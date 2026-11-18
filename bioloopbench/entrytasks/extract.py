"""Extraction: one task record per qualifying troubleshooting entry, then audit."""

from __future__ import annotations

import json
import re

from ..taskgen.corpus import protocol_text
from ..taskgen.grounding import longest_shared_run, quote_in_source
from ..taskgen.llm import chat_json
from . import prompts


def entries_text(article):
    return "\n\n".join(f"[Entry {e['index']}] {e['heading']}\n{e['text']}" for e in article["troubleshooting"])


def make_opening(context, problem):
    return f"{context.strip()}\n\nWhat went wrong: {problem.strip()}"


def check_drafts(reply, article, max_shared):
    """Deterministic validation of an extraction reply. Returns problem strings."""
    if not isinstance(reply, dict) or not isinstance(reply.get("tasks"), list) or not isinstance(reply.get("skipped"), list):
        return ["reply must be an object with 'tasks' and 'skipped' lists"]
    problems = []
    entries = {e["index"]: e["heading"] + "\n" + e["text"] for e in article["troubleshooting"]}
    full = article["title"] + "\n" + article["abstract"] + "\n" + \
        "\n".join(s["text"] for s in article["sections"]) + "\n" + "\n".join(entries.values())
    seen = [t.get("entry_index") for t in reply["tasks"]] + [s.get("entry_index") for s in reply["skipped"]]
    if sorted(map(str, seen)) != sorted(map(str, entries)):
        problems.append(f"every entry index in {sorted(entries)} must appear exactly once across 'tasks' and 'skipped'")
    for s in reply["skipped"]:
        if s.get("reason_code") not in prompts.SKIP_CODES:
            problems.append(f"skipped entry {s.get('entry_index')}: reason_code must be one of {list(prompts.SKIP_CODES)}")
    for n, t in enumerate(reply["tasks"], 1):
        where = f"task {n}"
        src = entries.get(t.get("entry_index"))
        if src is None:
            problems.append(f"{where}: entry_index {t.get('entry_index')!r} is not one of the entries")
            continue
        for field in ("technique", "context", "problem"):
            if not isinstance(t.get(field), str) or not t[field].strip():
                problems.append(f"{where}: '{field}' is missing")
        if not quote_in_source(t.get("problem_quote") or "", src):
            problems.append(f"{where}: problem_quote is not an exact span of entry {t['entry_index']}")
        refs = t.get("step_refs")
        if not isinstance(refs, list) or any(not isinstance(r, str) for r in refs):
            problems.append(f"{where}: step_refs must be a list of strings")
        else:
            for r in refs:
                if not re.search(rf"(?<![0-9a-z]){re.escape(r.lower())}(?![0-9])", src.lower()):
                    problems.append(f"{where}: step reference {r!r} does not appear in entry {t['entry_index']}")
        if not isinstance(t.get("fixes"), list) or not t["fixes"]:
            problems.append(f"{where}: needs at least one fix (or move the entry to 'skipped')")
            continue
        for k, fix in enumerate(t["fixes"], 1):
            if not isinstance(fix, dict) or not isinstance(fix.get("fix"), str) or not fix["fix"].strip():
                problems.append(f"{where} fix {k}: 'fix' is missing")
                continue
            if not quote_in_source(fix.get("fix_quote") or "", src):
                problems.append(f"{where} fix {k}: fix_quote is not an exact span of entry {t['entry_index']}")
            if fix.get("cause") is not None and not isinstance(fix["cause"], str):
                problems.append(f"{where} fix {k}: cause must be a string or null")
        for field in ("context", "problem"):
            run = longest_shared_run(t.get(field) or "", full)
            if run > max_shared:
                problems.append(f"{where}: '{field}' copies a {run}-word run from the article; rephrase it")
    return problems


def extract_entries(article, cfg):
    """Return (records, skipped) for one article. Each record has status
    'accepted' or 'rejected' with reasons; ``skipped`` maps a skip reason code
    to the number of entries the generator set aside. Raises LLMError if the
    model never produces a reply that passes the deterministic checks."""
    if not article["troubleshooting"]:
        return [], {}
    fill = dict(protocol=protocol_text(article, cfg["max_protocol_chars"]), entries=entries_text(article))
    reply = chat_json(cfg["models"]["generator"], prompts.EXTRACT.format(title=article["title"], **fill),
                      tag=f"entry_extract:{article['pmcid']}",
                      validate=lambda r: check_drafts(r, article, cfg["max_shared_run_words"]))
    skipped = {}
    for s in reply["skipped"]:
        skipped[s["reason_code"]] = skipped.get(s["reason_code"], 0) + 1
    drafts = reply["tasks"]
    if not drafts:
        return [], skipped

    def check_audit(a):
        got = {t.get("entry_index"): t for t in a.get("tasks", [])} if isinstance(a.get("tasks"), list) else {}
        out = []
        for d in drafts:
            g = got.get(d["entry_index"])
            if g is None:
                out.append(f"no audit for entry_index {d['entry_index']}")
            elif not isinstance(g.get("fixes"), list) or len(g["fixes"]) != len(d["fixes"]):
                out.append(f"entry {d['entry_index']}: audit must list {len(d['fixes'])} fixes in order")
        return out

    audit = chat_json(cfg["models"]["auditor"],
                      prompts.AUDIT_EXTRACT.format(drafts=json.dumps(drafts, indent=1, ensure_ascii=False), **fill),
                      tag=f"entry_audit:{article['pmcid']}", validate=check_audit)
    by_entry = {t["entry_index"]: t for t in audit["tasks"]}
    records = []
    for d in drafts:
        a = by_entry[d["entry_index"]]
        reasons = []
        if not a.get("bench_observable"):
            reasons.append("not_bench_observable")
        if a.get("problem_leaks_fix"):
            reasons.append("problem_leaks_fix")
        if a.get("context_unsupported"):
            reasons.append("context_unsupported")
        if a.get("identifies_source"):
            reasons.append("identifies_source")
        kept = [f for f, af in zip(d["fixes"], a["fixes"]) if af.get("supported") and af.get("concrete")]
        if not kept:
            reasons.append("no_supported_fix")
        records.append({
            "task_id": f"{article['pmcid']}_e{d['entry_index']}",
            "pmcid": article["pmcid"], "doi": article["doi"], "article_title": article["title"],
            "date": article["date"], "entry_index": d["entry_index"], "technique": d["technique"],
            "context": d["context"], "problem": d["problem"], "problem_quote": d["problem_quote"],
            "opening": make_opening(d["context"], d["problem"]),
            "step_refs": d["step_refs"],
            "fixes": [{"id": f"F{i}", "fix": f["fix"].strip(), "fix_quote": f["fix_quote"], "cause": f.get("cause")}
                      for i, f in enumerate(kept, 1)],
            "dropped_fixes": len(d["fixes"]) - len(kept),
            "budget": cfg["question_budget"],
            "audit": a, "status": "rejected" if reasons else "extracted", "reasons": reasons,
        })
    return records, skipped
