"""Stage A: source-grounded extraction of troubleshooting cases, then audit."""

from __future__ import annotations

import json

from . import prompts
from .corpus import protocol_text
from .grounding import longest_shared_run, quote_in_source
from .llm import chat_json


def _entries_text(article):
    return "\n\n".join(f"[Entry {e['index']}] {e['heading']}\n{e['text']}" for e in article["troubleshooting"])


def check_drafts(reply, article, max_shared):
    """Deterministic validation of an extraction reply. Returns problem strings."""
    problems = []
    if not isinstance(reply, dict) or not isinstance(reply.get("cases"), list):
        return ["reply must be an object with a 'cases' list"]
    entries = {e["index"]: e["heading"] + "\n" + e["text"] for e in article["troubleshooting"]}
    full = article["title"] + "\n" + article["abstract"] + "\n" + \
        "\n".join(s["text"] for s in article["sections"]) + "\n" + "\n".join(entries.values())
    for n, c in enumerate(reply["cases"], 1):
        where = f"case {n}"
        src = entries.get(c.get("entry_index"))
        if src is None:
            problems.append(f"{where}: entry_index {c.get('entry_index')!r} is not one of the entries")
            continue
        for field in ("technique", "symptom_key", "observed_failure", "setup"):
            if not isinstance(c.get(field), str) or not c[field].strip():
                problems.append(f"{where}: '{field}' is missing")
        if not quote_in_source(c.get("symptom_quote") or "", src):
            problems.append(f"{where}: symptom_quote is not an exact span of entry {c['entry_index']}")
        if not isinstance(c.get("causes"), list) or not c["causes"]:
            problems.append(f"{where}: needs at least one cause")
            continue
        for k, cause in enumerate(c["causes"], 1):
            for field in ("cause", "fix"):
                if not isinstance(cause.get(field), str) or not cause[field].strip():
                    problems.append(f"{where} cause {k}: '{field}' is missing")
            if cause.get("cause_support") not in ("explicit", "implied_by_solution"):
                problems.append(f"{where} cause {k}: cause_support must be 'explicit' or 'implied_by_solution'")
            for field in ("cause_quote", "fix_quote"):
                if not quote_in_source(cause.get(field) or "", src):
                    problems.append(f"{where} cause {k}: {field} is not an exact span of entry {c['entry_index']}")
        for field in ("observed_failure", "setup"):
            run = longest_shared_run(c.get(field) or "", full)
            if run > max_shared:
                problems.append(f"{where}: '{field}' copies a {run}-word run from the article; rephrase it")
    return problems


def extract_article(article, topic, cfg):
    """Return a list of case records for one article, each with status 'accepted'
    or 'rejected' and the reasons. Raises LLMError if the model never produces a
    reply that passes the deterministic checks."""
    if not article["troubleshooting"]:
        return []
    proto = protocol_text(article, cfg["max_protocol_chars"])
    entries = _entries_text(article)
    fill = dict(topic_name=topic["name"], topic_description=topic["description"], protocol=proto, entries=entries)
    reply = chat_json(cfg["models"]["generator"], prompts.EXTRACT.format(title=article["title"], **fill),
                      tag=f"extract:{topic['id']}:{article['pmcid']}",
                      validate=lambda r: check_drafts(r, article, cfg["max_shared_run_words"]))
    drafts = reply["cases"]
    if not drafts:
        return []

    def check_audit(a):
        got = {c.get("entry_index"): c for c in a.get("cases", [])}
        out = []
        for d in drafts:
            g = got.get(d["entry_index"])
            if g is None:
                out.append(f"no audit for entry_index {d['entry_index']}")
            elif len(g.get("causes", [])) != len(d["causes"]):
                out.append(f"entry {d['entry_index']}: audit must list {len(d['causes'])} causes in order")
        return out

    audit = chat_json(cfg["models"]["auditor"],
                      prompts.AUDIT_EXTRACT.format(drafts=json.dumps(drafts, indent=1, ensure_ascii=False), **fill),
                      tag=f"audit_extract:{topic['id']}:{article['pmcid']}", validate=check_audit)
    by_entry = {c["entry_index"]: c for c in audit["cases"]}
    records = []
    for d in drafts:
        a = by_entry[d["entry_index"]]
        reasons = []
        if not a.get("on_topic"):
            reasons.append("off_topic")
        if not a.get("bench_observable"):
            reasons.append("not_bench_observable")
        if a.get("failure_leaks_cause"):
            reasons.append("failure_leaks_cause")
        if a.get("setup_unsupported"):
            reasons.append("setup_unsupported")
        if a.get("identifies_source"):
            reasons.append("identifies_source")
        kept = [dict(c, audit=ac) for c, ac in zip(d["causes"], a["causes"])
                if ac.get("supported") and ac.get("specific")]
        if not kept:
            reasons.append("no_supported_cause")
        records.append({
            "case_id": f"{article['pmcid']}_e{d['entry_index']}",
            "topic": topic["id"], "pmcid": article["pmcid"], "doi": article["doi"],
            "article_title": article["title"], "date": article["date"],
            "entry_index": d["entry_index"], "technique": d["technique"], "symptom_key": d["symptom_key"],
            "symptom_quote": d["symptom_quote"], "observed_failure": d["observed_failure"], "setup": d["setup"],
            "causes": kept, "dropped_causes": len(d["causes"]) - len(kept),
            "audit": {k: v for k, v in a.items() if k != "causes"},
            "status": "rejected" if reasons else "accepted", "reasons": reasons,
        })
    return records
