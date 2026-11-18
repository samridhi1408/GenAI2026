"""Run the entry-task pipeline.

  python -m bioloopbench.entrytasks.pipeline --config configs/entrytasks.json --out generated/entry1
  python -m bioloopbench.entrytasks.pipeline ... --topics cloning --stages corpus,extract

Stages are corpus, extract, calibrate. The corpus stage is the one used by
bioloopbench.taskgen: a seed topic only decides which articles are sampled;
every troubleshooting entry in a sampled article is considered, whatever
technique it concerns. Model replies are cached under the run directory, so a
run can be resumed or extended without repeating work.

<run>/
  corpus/xml/                    cached article XML
  topics/<topic>/articles.json   sampled articles and their split
  topics/<topic>/entries.json    task ids produced from this topic's articles
  protocols/<pmcid>.txt          protocol text the gatekeeper answers from
  private/<task_id>.json         full record: fixes, quotes, provenance, audit, calibration
  public/<task_id>.json          what an evaluated agent may see
  funnel.json                    counts of what was kept and rejected, and why
  manifest.csv, manifest.jsonl   one row per task: dates, identifiers, split, status, models
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import csv
import random
import subprocess
import time

from ..taskgen import llm
from ..taskgen import prompts as taskgen_prompts
from ..taskgen.corpus import protocol_text, search
from ..taskgen.pipeline import load_article
from ..taskgen.store import DEFAULTS, Funnel, pmap, read_json, write_json
from .calibrate import calibrate_task
from .env import public_view
from .extract import extract_entries

STAGES = ("corpus", "extract", "calibrate")

ENTRY_DEFAULTS = {
    "articles_per_topic": None,    # if set, sample this many per topic: every test-split article first,
                                   # then development articles; overrides the two per-split counts
    "fetch_workers": 6,            # parallel article downloads (kept low to be polite to Europe PMC)
    "question_budget": 6,          # questions an agent may ask per task in ask mode
    "symptom_only_samples": 3,     # solver attempts from the opening alone
}


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    out = {**DEFAULTS, **ENTRY_DEFAULTS, **cfg}
    out["models"] = {**DEFAULTS["models"], **cfg.get("models", {})}
    missing = [r for r, m in out["models"].items() if not m]
    if missing:
        raise ValueError(f"config is missing models for roles: {', '.join(missing)}")
    return out


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def protocol_path(run, pmcid):
    return Path(run) / "protocols" / f"{pmcid}.txt"


ARTICLE_META = ("pmid", "doi", "journal", "volume", "issue", "pub_year", "pub_dates", "history_dates",
                "first_publication_date", "first_index_date")
MANIFEST_FIELDS = ("task_id", "status", "reasons", "split", "date", "first_publication_date", "pub_year",
                   "pub_date_epub", "pub_date_collection", "date_received", "date_accepted", "first_index_date",
                   "pmcid", "pmid", "doi", "journal", "sampled_for_topic", "entry_index", "technique", "n_fixes",
                   "n_step_refs", "symptom_solved", "symptom_only_top1_hits", "symptom_only_samples",
                   "generator", "auditor", "judge", "solver", "extracted_at", "pipeline_commit")


def _commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).parent, timeout=10).stdout.strip()
    except Exception:
        return ""


class Progress:
    """Writes <run>/progress.json as work items finish, for the dashboard."""

    def __init__(self, run, topic, stage, total):
        self.path, self.lock = Path(run) / "progress.json", __import__("threading").Lock()
        self.state = {"topic": topic, "stage": stage, "total": total, "done": 0, "failed": 0,
                      "started": time.time(), "updated": time.time()}
        write_json(self.path, self.state)

    def wrap(self, fn):
        def run(x):
            ok = False
            try:
                out = fn(x)
                ok = True
                return out
            finally:
                with self.lock:
                    self.state["done"] += 1
                    self.state["failed"] += not ok
                    self.state["updated"] = time.time()
                    self.state["usage"] = llm.USAGE
                    write_json(self.path, self.state)
        return run


def stage_corpus(run, topic, cfg, funnel):
    """Sample articles for a topic. An article already sampled for another
    topic in this run is left out, so every article belongs to one topic."""
    tdir = run / "topics" / topic["id"]
    phrases = topic.get("search_phrases") or read_json(tdir / "articles.json", {}).get("search_phrases")
    if not phrases:
        phrases = llm.chat_json(cfg["models"]["generator"], taskgen_prompts.TOPIC_QUERIES.format(
            topic_name=topic["name"], topic_description=topic["description"]),
            tag=f"queries:{topic['id']}",
            validate=lambda r: [] if isinstance(r.get("phrases"), list) and r["phrases"] else ["need a 'phrases' list"])["phrases"]
    taken = set()
    for other in sorted((run / "topics").glob("*/articles.json")):
        if other.parent.name != topic["id"]:
            taken |= {a["pmcid"] for a in read_json(other)["articles"]}
    cutoff = cfg["test_cutoff_date"]
    hits = search(phrases, limit=10000)
    funnel.reset(topic["id"], "corpus")
    funnel.add(topic["id"], "corpus", "articles_found", len(hits))
    funnel.add(topic["id"], "corpus", "already_in_another_topic", sum(h["pmcid"] in taken for h in hits))
    hits = [h for h in hits if h["pmcid"] not in taken]
    rng = random.Random(f"{cfg['seed']}:{topic['id']}")
    pools = {"development": sorted((h for h in hits if h["date"] and h["date"] < cutoff), key=lambda h: h["pmcid"]),
             "test": sorted((h for h in hits if h["date"] >= cutoff), key=lambda h: h["pmcid"])}
    wanted = {"development": cfg["dev_articles_per_topic"], "test": cfg["test_articles_per_topic"]}
    if cfg.get("articles_per_topic"):
        n_test = min(len(pools["test"]), cfg["articles_per_topic"])
        wanted = {"test": n_test, "development": cfg["articles_per_topic"] - n_test}
    chosen = []
    for split, pool in pools.items():
        funnel.add(topic["id"], "corpus", f"{split}_articles_available", len(pool))
        rng.shuffle(pool)
        chosen += [dict(h, split=split) for h in pool[: wanted[split]]]
    fetch = Progress(run, topic["id"], "corpus", len(chosen)).wrap(lambda h: load_article(run, h["pmcid"]))
    for (h, art) in pmap(fetch, chosen, cfg["fetch_workers"]):
        if isinstance(art, Exception):
            h["error"] = str(art)[:200]
            funnel.add(topic["id"], "corpus", "fetch_or_parse_failed")
            continue
        h["n_entries"] = len(art["troubleshooting"])
        # The article's own date decides the split; the search index can lag.
        h["date"] = art["date"] or h["date"]
        h["split"] = "test" if h["date"] >= cutoff else "development"
        for k in ("pmid", "doi", "journal"):
            h[k] = art.get(k) or h.get(k, "")
        h.update(pub_dates=art["pub_dates"], history_dates=art["history_dates"], volume=art["volume"], issue=art["issue"])
        funnel.add(topic["id"], "corpus", f"{h['split']}_articles_selected")
        funnel.add(topic["id"], "corpus", "entries_in_selected_articles", h["n_entries"])
    write_json(tdir / "articles.json", {"topic": topic, "search_phrases": phrases, "cutoff": cutoff,
                                        "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "articles": chosen})


def write_manifest(run):
    """One row per task with its dates and provenance, as CSV and JSON lines,
    so tasks can be re-split by any date without opening the records."""
    rows = []
    for path in sorted((Path(run) / "private").glob("*.json")):
        t = read_json(path)
        art, cal, models = t.get("article", {}), t.get("calibration", {}), t.get("models", {})
        rows.append({
            "task_id": t["task_id"], "status": t["status"], "reasons": ";".join(t.get("reasons", [])),
            "split": t.get("split", ""), "date": t.get("date", ""),
            "first_publication_date": art.get("first_publication_date", ""), "pub_year": art.get("pub_year", ""),
            "pub_date_epub": art.get("pub_dates", {}).get("epub", ""),
            "pub_date_collection": art.get("pub_dates", {}).get("collection", ""),
            "date_received": art.get("history_dates", {}).get("received", ""),
            "date_accepted": art.get("history_dates", {}).get("accepted", ""),
            "first_index_date": art.get("first_index_date", ""),
            "pmcid": t["pmcid"], "pmid": art.get("pmid", ""), "doi": t.get("doi", ""), "journal": art.get("journal", ""),
            "sampled_for_topic": t.get("sampled_for_topic", ""), "entry_index": t.get("entry_index", ""),
            "technique": t.get("technique", ""), "n_fixes": len(t.get("fixes", [])),
            "n_step_refs": len(t.get("step_refs", [])), "symptom_solved": t.get("symptom_solved", ""),
            "symptom_only_top1_hits": cal.get("symptom_only", {}).get("top1_hits", ""),
            "symptom_only_samples": len(cal.get("symptom_only", {}).get("first_match_ranks", [])) or "",
            **{r: models.get(r, "") for r in ("generator", "auditor", "judge", "solver")},
            "extracted_at": t.get("extracted_at", ""), "pipeline_commit": t.get("pipeline_commit", ""),
        })
    with open(Path(run) / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)
    with open(Path(run) / "manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def stage_extract(run, topic, cfg, funnel):
    tdir = run / "topics" / topic["id"]
    arts = [a for a in read_json(tdir / "articles.json")["articles"] if not a.get("error")]
    funnel.reset(topic["id"], "extract")
    commit, now = _commit(), time.strftime("%Y-%m-%dT%H:%M:%S%z")

    def work(a):
        art = load_article(run, a["pmcid"])
        records, skipped = extract_entries(art, cfg)
        path = protocol_path(run, a["pmcid"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(protocol_text(art, cfg["max_protocol_chars"]))
        return len(art["troubleshooting"]), records, skipped

    ids = []
    for a, res in pmap(Progress(run, topic["id"], "extract", len(arts)).wrap(work), arts, cfg["workers"]):
        if isinstance(res, Exception):
            log(f"  extract failed for {a['pmcid']}: {type(res).__name__}")
            funnel.add(topic["id"], "extract", "article_failed")
            funnel.add(topic["id"], "extract", f"article_failed:{type(res).__name__}")
            continue
        n_entries, records, skipped = res
        funnel.add(topic["id"], "extract", "articles_read")
        funnel.add(topic["id"], "extract", "entries_seen", n_entries)
        for code, n in skipped.items():
            funnel.add(topic["id"], "extract", "skipped:" + code, n)
        for r in records:
            r.update(split=a["split"], sampled_for_topic=topic["id"], models=cfg["models"],
                     article={k: a.get(k, "") for k in ARTICLE_META}, n_entries_in_article=n_entries,
                     test_cutoff_date=cfg["test_cutoff_date"], extracted_at=now, pipeline_commit=commit)
            funnel.add(topic["id"], "extract", f"tasks_{r['status']}_{a['split']}")
            funnel.add(topic["id"], "extract", "fixes_kept", len(r["fixes"]))
            if r["step_refs"]:
                funnel.add(topic["id"], "extract", "tasks_with_step_ref")
            for reason in r["reasons"]:
                funnel.add(topic["id"], "extract", "reject:" + reason)
            write_json(run / "private" / f"{r['task_id']}.json", r)
            ids.append(r["task_id"])
    write_json(tdir / "entries.json", ids)


def stage_calibrate(run, topic, cfg, funnel):
    funnel.reset(topic["id"], "calibrate")
    ids = read_json(run / "topics" / topic["id"] / "entries.json", [])
    tasks = [t for t in (read_json(run / "private" / f"{i}.json") for i in ids)
             if t and (t["status"] in ("extracted", "accepted") or t["reasons"] == ["judge_cannot_score"])]

    def work(t):
        return calibrate_task(t, protocol_path(run, t["pmcid"]).read_text(), cfg)

    for t, done in pmap(Progress(run, topic["id"], "calibrate", len(tasks)).wrap(work), tasks, cfg["workers"]):
        if isinstance(done, Exception):
            log(f"  calibration failed for {t['task_id']}: {type(done).__name__}")
            funnel.add(topic["id"], "calibrate", "task_failed")
            funnel.add(topic["id"], "calibrate", f"task_failed:{type(done).__name__}")
            continue
        write_json(run / "private" / f"{done['task_id']}.json", done)
        funnel.add(topic["id"], "calibrate", f"tasks_{done['status']}_{done['split']}")
        for reason in done["reasons"]:
            funnel.add(topic["id"], "calibrate", "reject:" + reason)
        pub = run / "public" / f"{done['task_id']}.json"
        if done["status"] == "accepted":
            if done["symptom_solved"]:
                funnel.add(topic["id"], "calibrate", f"symptom_solved_{done['split']}")
            write_json(pub, public_view(done))
        elif pub.exists():
            pub.unlink()


RUNNERS = {"corpus": stage_corpus, "extract": stage_extract, "calibrate": stage_calibrate}


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
    try:
        for topic in topics:
            for stage in STAGES:
                if stage in stages:
                    log(f"[{topic['id']}] {stage}")
                    RUNNERS[stage](run, topic, cfg, funnel)
                    funnel.save()
                    log("   ", json.dumps(funnel.data[topic["id"]].get(stage, {})))
    finally:
        funnel.save()
        log(f"manifest: {write_manifest(run)} tasks")
        history = read_json(run / "usage.json", [])
        write_json(run / "usage.json", history + [{"topics": [t["id"] for t in topics], "stages": stages, "usage": llm.USAGE}])
        log(f"spend this invocation: ${llm.total_cost():.2f}")


if __name__ == "__main__":
    main()
