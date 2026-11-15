"""Stage C: the opening, the diagnostic-evidence matrix and hidden case notes.

The matrix (tests x causes) is written in one call so results are consistent
across sibling causes, then checked structurally and by an independent auditor.
"""

from __future__ import annotations

import json
from itertools import combinations

from . import prompts
from .grounding import longest_shared_run
from .llm import chat_json


def classes(test, cause_ids):
    return {c: test["results"][c]["outcome_class"] for c in cause_ids}


def inseparable_pairs(tests, cause_ids):
    return [(a, b) for a, b in combinations(cause_ids, 2)
            if all(t["results"][a]["outcome_class"] == t["results"][b]["outcome_class"] for t in tests)]


def min_tests_to_identify(tests, cause_ids, true_cause):
    """Fewest tests whose results, under ``true_cause``, rule out every other cause."""
    others = [c for c in cause_ids if c != true_cause]
    for k in range(0, len(tests) + 1):
        for subset in combinations(tests, k):
            if all(any(t["results"][o]["outcome_class"] != t["results"][true_cause]["outcome_class"]
                       for t in subset) for o in others):
                return k
    return None


def check_casefile(cf, cause_ids, source_text, max_shared, min_tests=4):
    problems = []
    if not isinstance(cf, dict):
        return ["reply must be a JSON object"]
    if not isinstance(cf.get("opening"), str) or len(cf["opening"]) < 200:
        problems.append("opening is missing or too short to give the needed setup")
    else:
        run = longest_shared_run(cf["opening"], source_text)
        if run > max_shared:
            problems.append(f"opening copies a {run}-word run from the source article; rephrase it")
    tests = cf.get("tests")
    if not isinstance(tests, list) or len(tests) < min_tests:
        return problems + [f"need at least {min_tests} tests"]
    seen = set()
    for t in tests:
        tid = t.get("test_id")
        if not tid or tid in seen:
            problems.append(f"test ids must be unique and present (got {tid!r})")
        seen.add(tid)
        if not t.get("name") or not t.get("procedure"):
            problems.append(f"{tid}: name and procedure are required")
        res = t.get("results")
        if not isinstance(res, dict) or sorted(res) != sorted(cause_ids):
            problems.append(f"{tid}: results must have exactly the keys {cause_ids}")
            continue
        for c, cell in res.items():
            if not isinstance(cell, dict) or not isinstance(cell.get("observation"), str) or not cell["observation"].strip():
                problems.append(f"{tid}/{c}: observation is missing")
            if not isinstance(cell, dict) or not isinstance(cell.get("outcome_class"), str) or not cell["outcome_class"].strip():
                problems.append(f"{tid}/{c}: outcome_class is missing")
        if not problems:
            obs = {}
            for c, cell in res.items():
                obs.setdefault(" ".join(cell["observation"].split()).lower(), set()).add(cell["outcome_class"])
            if any(len(v) > 1 for v in obs.values()):
                problems.append(f"{tid}: identical observations were given different outcome classes")
    for key in ("case_notes", "reference_diagnosis"):
        if not isinstance(cf.get(key), dict) or sorted(cf[key]) != sorted(cause_ids):
            problems.append(f"{key} must have exactly the keys {cause_ids}")
    if not problems:
        for a, b in inseparable_pairs(tests, cause_ids):
            problems.append(f"no test tells {a} from {b}: give at least one test where their results differ")
    return problems


def _problem_count(audit):
    return sum(len(audit.get(k, [])) for k in ("opening_problems", "test_problems", "cell_problems", "diagnosis_problems"))


def build_casefile(task_id, anchor, causes, source_text, cfg):
    """Return (casefile, audit_trail, reject_reason). May drop causes or tests
    that still fail the audit after one repair round; ``casefile['cause_ids']``
    lists the causes that survive."""
    ids = [c["id"] for c in causes]
    fill = dict(setup=anchor["setup"], observed_failure=anchor["observed_failure"])
    validate = lambda cf: check_casefile(cf, ids, source_text, cfg["max_shared_run_words"])
    prompt = prompts.CASEFILE.format(causes="\n".join(f"{c['id']}: {c['statement']}" for c in causes), **fill)
    cf = chat_json(cfg["models"]["generator"], prompt, tag=f"casefile:{task_id}", validate=validate, max_tokens=32000)

    def audit_of(cf, round_no):
        def check(a):
            tids = {t["test_id"] for t in cf["tests"]}
            out = []
            for p in a.get("cell_problems", []):
                if p.get("test_id") not in tids or p.get("cause_id") not in ids:
                    out.append(f"cell problem refers to unknown test or cause: {p.get('test_id')}/{p.get('cause_id')}")
            for p in a.get("test_problems", []):
                if p.get("test_id") not in tids:
                    out.append(f"test problem refers to unknown test {p.get('test_id')}")
            return out
        shown = {k: cf[k] for k in ("opening", "tests", "reference_diagnosis")}
        shown["causes"] = {c["id"]: c["statement"] for c in causes}
        return chat_json(cfg["models"]["auditor"],
                         prompts.AUDIT_CASEFILE.format(casefile=json.dumps(shown, indent=1, ensure_ascii=False), **fill),
                         tag=f"audit_casefile:{task_id}:{round_no}", validate=check)

    trail = []
    audit = audit_of(cf, 0)
    trail.append(audit)
    if _problem_count(audit):
        cf = chat_json(cfg["models"]["generator"],
                       [{"role": "user", "content": prompt},
                        {"role": "assistant", "content": json.dumps(cf, ensure_ascii=False)},
                        {"role": "user", "content": prompts.REPAIR_CASEFILE.format(
                            problems=json.dumps(audit, indent=1, ensure_ascii=False))}],
                       tag=f"repair_casefile:{task_id}", validate=validate, max_tokens=32000)
        audit = audit_of(cf, 1)
        trail.append(audit)

    if audit.get("opening_problems"):
        return None, trail, "opening_failed_audit"
    if audit.get("diagnosis_problems"):
        return None, trail, "diagnosis_failed_audit"
    bad_tests = {p["test_id"] for p in audit.get("test_problems", [])}
    bad_causes = {p["cause_id"] for p in audit.get("cell_problems", []) if p["test_id"] not in bad_tests}
    keep_ids = [i for i in ids if i not in bad_causes]
    tests = [t for t in cf["tests"] if t["test_id"] not in bad_tests]
    if len(keep_ids) < cfg["min_causes"]:
        return None, trail, "evidence_failed_audit"
    for t in tests:
        t["results"] = {c: t["results"][c] for c in keep_ids}
    if len(tests) < 3 or inseparable_pairs(tests, keep_ids):
        return None, trail, "evidence_not_separable_after_audit"
    out = {"opening": cf["opening"], "tests": tests, "cause_ids": keep_ids,
           "case_notes": {c: cf["case_notes"][c] for c in keep_ids},
           "reference_diagnosis": {c: cf["reference_diagnosis"][c] for c in keep_ids},
           "dropped_after_audit": {"causes": sorted(bad_causes), "tests": sorted(bad_tests)}}
    need = {c: min_tests_to_identify(tests, keep_ids, c) for c in keep_ids}
    out["min_tests_to_identify"] = need
    out["budget"] = min(len(tests), max(need.values()) + 2)
    return out, trail, None
