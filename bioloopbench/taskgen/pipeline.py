"""Run the task-generation pipeline.

  python -m bioloopbench.taskgen.pipeline --config configs/taskgen.json --out generated/dev1
  python -m bioloopbench.taskgen.pipeline ... --topics cloning,pcr --stages corpus,extract

Stages are corpus, extract, build, calibrate. Each writes its result under the
run directory, and model replies are cached, so a run can be resumed or
extended without repeating work.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from . import llm, prompts
from .calibrate import calibrate_task
from .casefile import build_casefile
from .corpus import fetch_xml, parse_article, search
from .extract import extract_article
from .gatekeeper import public_view
from .siblings import assemble_task, group_symptoms
from .store import Funnel, load_config, pmap, read_json, write_json

STAGES = ("corpus", "extract", "build", "calibrate")


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def load_article(run, pmcid):
    return parse_article(fetch_xml(pmcid, run / "corpus" / "xml"))


def stage_corpus(run, topic, cfg, funnel):
    tdir = run / "topics" / topic["id"]
    phrases = topic.get("search_phrases")
    if not phrases:
        phrases = llm.chat_json(cfg["models"]["generator"], prompts.TOPIC_QUERIES.format(
            topic_name=topic["name"], topic_description=topic["description"]),
            tag=f"queries:{topic['id']}",
            validate=lambda r: [] if isinstance(r.get("phrases"), list) and r["phrases"] else ["need a 'phrases' list"])["phrases"]
    cutoff = cfg["test_cutoff_date"]
    hits = search(phrases)
    rng = random.Random(f"{cfg['seed']}:{topic['id']}")
    pools = {"development": sorted((h for h in hits if h["date"] and h["date"] < cutoff), key=lambda h: h["pmcid"]),
             "test": sorted((h for h in hits if h["date"] >= cutoff), key=lambda h: h["pmcid"])}
    wanted = {"development": cfg["dev_articles_per_topic"], "test": cfg["test_articles_per_topic"]}
    funnel.reset(topic["id"], "corpus")
    chosen = []
    for split, pool in pools.items():
        funnel.add(topic["id"], "corpus", f"{split}_articles_found", len(pool))
        rng.shuffle(pool)
        chosen += [dict(h, split=split) for h in pool[: wanted[split]]]
    for (h, art) in pmap(lambda h: load_article(run, h["pmcid"]), chosen, cfg["workers"]):
        if isinstance(art, Exception):
            h["error"] = str(art)
            funnel.add(topic["id"], "corpus", "fetch_or_parse_failed")
        else:
            h["n_entries"] = len(art["troubleshooting"])
            # The article's own date decides the split; the search index can lag.
            h["date"] = art["date"] or h["date"]
            h["split"] = "test" if h["date"] >= cutoff else "development"
            funnel.add(topic["id"], "corpus", f"{h['split']}_articles_selected")
    write_json(tdir / "articles.json", {"topic": topic, "search_phrases": phrases, "cutoff": cutoff, "articles": chosen})


def stage_extract(run, topic, cfg, funnel):
    tdir = run / "topics" / topic["id"]
    arts = [a for a in read_json(tdir / "articles.json")["articles"] if not a.get("error")]
    funnel.reset(topic["id"], "extract")

    def work(a):
        recs = extract_article(load_article(run, a["pmcid"]), topic, cfg)
        for r in recs:
            r["split"] = a["split"]
        return recs

    drafts = []
    for a, recs in pmap(work, arts, cfg["workers"]):
        if isinstance(recs, Exception):
            log(f"  extract failed for {a['pmcid']}: {recs}")
            funnel.add(topic["id"], "extract", "article_failed")
            continue
        funnel.add(topic["id"], "extract", "articles_read")
        if not recs:
            funnel.add(topic["id"], "extract", "articles_with_no_qualifying_entry")
        for r in recs:
            funnel.add(topic["id"], "extract", "cases_" + r["status"])
            for reason in r["reasons"]:
                funnel.add(topic["id"], "extract", "reject:" + reason)
        drafts += recs
    write_json(tdir / "drafts.json", drafts)


def stage_build(run, topic, cfg, funnel):
    tdir = run / "topics" / topic["id"]
    cases = [c for c in read_json(tdir / "drafts.json", []) if c["status"] == "accepted"]
    funnel.reset(topic["id"], "build")
    if not cases:
        write_json(tdir / "groups.json", [])
        return
    groups = group_symptoms(cases, topic, cfg)
    write_json(tdir / "groups.json", groups)
    by_id = {c["case_id"]: c for c in cases}
    jobs, per_article = [], {}
    for g in groups:
        members = [by_id[i] for i in g["case_ids"]]
        funnel.add(topic["id"], "build", "symptom_groups")
        for split in ("development", "test"):
            anchors = sorted((m for m in members if m["split"] == split),
                             key=lambda m: (-len(m["causes"]), m["case_id"]))
            taken = 0
            for a in anchors:
                if taken >= cfg["max_tasks_per_group_per_split"]:
                    break
                if per_article.get(a["pmcid"], 0) >= cfg["max_tasks_per_article"]:
                    continue
                per_article[a["pmcid"]] = per_article.get(a["pmcid"], 0) + 1
                taken += 1
                jobs.append((a, members, g["label"]))

    def work(job):
        anchor, members, label = job
        task_id = f"{topic['id']}-{anchor['case_id']}"
        causes, why = assemble_task(anchor, members, cfg)
        if why:
            return {"task_id": task_id, "status": "rejected", "reject_reason": why}
        art = load_article(run, anchor["pmcid"])
        source_text = "\n".join([art["title"], art["abstract"], *(s["text"] for s in art["sections"]),
                                 *(e["heading"] + "\n" + e["text"] for e in art["troubleshooting"])])
        cf, trail, why = build_casefile(task_id, anchor, causes, source_text, cfg)
        base = {"task_id": task_id, "topic": topic["id"], "split": anchor["split"], "group_label": label,
                "anchor": {k: anchor[k] for k in ("case_id", "pmcid", "doi", "article_title", "date", "entry_index",
                                                  "technique", "symptom_quote", "setup", "observed_failure")},
                "models": cfg["models"], "audits": {"casefile": trail}}
        if why:
            return {**base, "causes": causes, "status": "rejected", "reject_reason": why}
        kept = [c for c in causes if c["id"] in cf["cause_ids"]]
        return {**base, "causes": kept, **cf, "status": "built", "reject_reason": None}

    for job, task in pmap(work, jobs, cfg["workers"]):
        if isinstance(task, Exception):
            log(f"  build failed for {job[0]['case_id']}: {task}")
            funnel.add(topic["id"], "build", "task_failed")
            continue
        funnel.add(topic["id"], "build", "tasks_" + task["status"])
        if task["reject_reason"]:
            funnel.add(topic["id"], "build", "reject:" + task["reject_reason"])
        write_json(run / "private" / f"{task['task_id']}.json", task)


def stage_calibrate(run, topic, cfg, funnel):
    funnel.reset(topic["id"], "calibrate")
    paths = sorted((run / "private").glob(f"{topic['id']}-*.json"))
    tasks = [t for t in map(read_json, paths) if t.get("opening") and t["status"] in ("built", "accepted")
             or t.get("reject_reason") in ("judge_cannot_score", "no_instance_passed_calibration")]
    for t, done in pmap(lambda t: calibrate_task(t, cfg), tasks, cfg["workers"]):
        if isinstance(done, Exception):
            log(f"  calibration failed for {t['task_id']}: {done}")
            funnel.add(topic["id"], "calibrate", "task_failed")
            continue
        write_json(run / "private" / f"{done['task_id']}.json", done)
        funnel.add(topic["id"], "calibrate", "tasks_" + done["status"])
        if done["reject_reason"]:
            funnel.add(topic["id"], "calibrate", "reject:" + done["reject_reason"])
        for inst in done["instances"]:
            funnel.add(topic["id"], "calibrate", f"instances_{inst['status']}_{done['split']}")
            for r in inst["reasons"]:
                funnel.add(topic["id"], "calibrate", "instance_reject:" + r)
        pub = run / "public" / f"{done['task_id']}.json"
        if done["status"] == "accepted":
            write_json(pub, public_view(done))
        elif pub.exists():
            pub.unlink()


RUNNERS = {"corpus": stage_corpus, "extract": stage_extract, "build": stage_build, "calibrate": stage_calibrate}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--topics-file", default="configs/seed_topics.json")
    ap.add_argument("--out", required=True, help="run directory")
    ap.add_argument("--topics", help="comma-separated topic ids (default: all)")
    ap.add_argument("--stages", default=",".join(STAGES))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    run = Path(args.out)
    run.mkdir(parents=True, exist_ok=True)
    llm.CACHE_DIR = run / "llm_cache"
    llm.MAX_COST_USD = cfg["max_cost_usd"]
    write_json(run / "config.json", cfg)
    topics = json.loads(Path(args.topics_file).read_text())["topics"]
    if args.topics:
        want = args.topics.split(",")
        unknown = set(want) - {t["id"] for t in topics}
        if unknown:
            ap.error(f"unknown topics: {sorted(unknown)}")
        topics = [t for t in topics if t["id"] in want]
    stages = args.stages.split(",")
    if set(stages) - set(STAGES):
        ap.error(f"stages must be among {STAGES}")
    funnel = Funnel(run / "funnel.json")
    for topic in topics:
        for stage in STAGES:
            if stage in stages:
                log(f"[{topic['id']}] {stage}")
                RUNNERS[stage](run, topic, cfg, funnel)
                funnel.save()
                log("   ", json.dumps(funnel.data[topic["id"]].get(stage, {})))
    history = read_json(run / "usage.json", [])
    write_json(run / "usage.json", history + [{"topics": [t["id"] for t in topics], "stages": stages, "usage": llm.USAGE}])
    log(f"spend this invocation: ${llm.total_cost():.2f}")
    log("model usage:", json.dumps(llm.USAGE))


if __name__ == "__main__":
    main()
