"""Stage B: group cases by presenting symptom and assemble competing causes.

A task is anchored on one accepted case. Its candidate causes are the anchor
article's own causes plus causes that other articles give for the same symptom,
kept only where they apply to the anchor setup. Each cause records its origin,
so the faithfulness of every hidden instance can be stated exactly:

  own            stated for this failure by the anchor article
  transplanted   stated for the same symptom by another article; applicability
                 to the anchor setup judged by the assembler and the auditor
  model_proposed proposed by the model (only if enabled in the config)
"""

from __future__ import annotations

from . import prompts
from .llm import chat_json


def group_symptoms(cases, topic, cfg):
    """Return [{label, case_ids}] covering every accepted case exactly once."""
    ids = [c["case_id"] for c in cases]
    if len(ids) <= 1:
        return [{"label": c["symptom_key"], "case_ids": [c["case_id"]]} for c in cases]
    listing = "\n".join(f"{c['case_id']} | {c['technique']} | {c['symptom_key']}" for c in cases)

    def check(r):
        seen = [i for g in r.get("groups", []) for i in g.get("case_ids", [])]
        out = []
        if sorted(seen) != sorted(ids):
            missing, extra = set(ids) - set(seen), set(seen) - set(ids)
            dup = {i for i in seen if seen.count(i) > 1}
            if missing: out.append(f"missing case ids: {sorted(missing)}")
            if extra: out.append(f"unknown case ids: {sorted(extra)}")
            if dup: out.append(f"case ids in more than one group: {sorted(dup)}")
        return out

    reply = chat_json(cfg["models"]["generator"], prompts.CLUSTER.format(cases=listing),
                      tag=f"cluster:{topic['id']}", validate=check)
    return [{"label": g["label"], "case_ids": g["case_ids"]} for g in reply["groups"]]


def assemble_task(anchor, group_cases, cfg):
    """Return (causes, reject_reason). ``causes`` is a list of dicts with id,
    statement, origin and provenance; reject_reason is None on success."""
    own = {f"O{i}": c for i, c in enumerate(anchor["causes"], 1)}
    foreign = {}
    for other in group_cases:
        if other["pmcid"] == anchor["pmcid"]:
            continue
        for c in other["causes"]:
            foreign[f"F{len(foreign) + 1}"] = (other, c)
    n_extra = cfg["allow_model_proposed_causes"]
    if len(own) + len(foreign) + n_extra < cfg["min_causes"]:
        return None, "too_few_candidate_causes"

    def check(r):
        out = []
        cs = r.get("causes")
        if not isinstance(cs, list):
            return ["'causes' must be a list"]
        if len(cs) > cfg["max_causes"]:
            out.append(f"at most {cfg['max_causes']} causes")
        refs = [c.get("ref") for c in cs if c.get("origin") != "model_proposed"]
        if len(refs) != len(set(refs)):
            out.append("each ref may be used once")
        for c in cs:
            o, ref = c.get("origin"), c.get("ref")
            if o == "own" and ref not in own: out.append(f"unknown own ref {ref!r}")
            elif o == "transplanted" and ref not in foreign: out.append(f"unknown foreign ref {ref!r}")
            elif o == "model_proposed" and not n_extra: out.append("model_proposed causes are not allowed")
            elif o not in ("own", "transplanted", "model_proposed"): out.append(f"bad origin {o!r}")
            if not isinstance(c.get("statement"), str) or not c["statement"].strip():
                out.append("every cause needs a statement")
        if sum(c.get("origin") == "model_proposed" for c in cs) > n_extra:
            out.append(f"at most {n_extra} model_proposed causes")
        return out

    prompt = prompts.ASSEMBLE.format(
        setup=anchor["setup"], observed_failure=anchor["observed_failure"],
        own_causes="\n".join(f"{k}: {c['cause']}" for k, c in own.items()),
        foreign_causes="\n".join(f"{k}: {c['cause']}  (in a protocol using: {o['technique']})"
                                 for k, (o, c) in foreign.items()) or "(none)",
        model_proposed_clause=prompts.MODEL_PROPOSED_CLAUSE.format(n=n_extra) if n_extra else "")
    reply = chat_json(cfg["models"]["generator"], prompt, tag=f"assemble:{anchor['case_id']}", validate=check)

    causes = []
    for c in reply["causes"]:
        rec = {"id": f"C{len(causes) + 1}", "statement": c["statement"].strip(), "origin": c["origin"],
               "why_applicable": c.get("why_applicable", "")}
        if c["origin"] == "own":
            src = own[c["ref"]]
            rec["provenance"] = {"case_id": anchor["case_id"], "pmcid": anchor["pmcid"],
                                 "cause_quote": src["cause_quote"], "fix_quote": src["fix_quote"],
                                 "cause_support": src["cause_support"], "fix": src["fix"]}
        elif c["origin"] == "transplanted":
            other, src = foreign[c["ref"]]
            rec["provenance"] = {"case_id": other["case_id"], "pmcid": other["pmcid"],
                                 "cause_quote": src["cause_quote"], "fix_quote": src["fix_quote"],
                                 "cause_support": src["cause_support"], "fix": src["fix"],
                                 "original_statement": src["cause"]}
        else:
            rec["provenance"] = None
        causes.append(rec)
    if len(causes) < cfg["min_causes"]:
        return None, "too_few_applicable_causes"

    ids = [c["id"] for c in causes]

    def check_audit(a):
        got = [c.get("id") for c in a.get("causes", [])]
        out = [] if sorted(got) == sorted(ids) else [f"audit must cover exactly {ids}"]
        for p in a.get("indistinct_pairs", []):
            if len(p) != 2 or any(x not in ids for x in p):
                out.append(f"bad pair {p}")
        return out

    audit = chat_json(cfg["models"]["auditor"], prompts.AUDIT_ASSEMBLE.format(
        setup=anchor["setup"], observed_failure=anchor["observed_failure"],
        causes="\n".join(f"{c['id']}: {c['statement']}" for c in causes)),
        tag=f"audit_assemble:{anchor['case_id']}", validate=check_audit)
    verdict = {c["id"]: c for c in audit["causes"]}
    drop = {i for i in ids if not (verdict[i].get("applicable") and verdict[i].get("produces_failure"))}
    # For each indistinct pair drop one member, preferring to keep the anchor's own cause.
    rank = {"own": 0, "transplanted": 1, "model_proposed": 2}
    by_id = {c["id"]: c for c in causes}
    for a_id, b_id in audit.get("indistinct_pairs", []):
        if a_id in drop or b_id in drop:
            continue
        loser = max((a_id, b_id), key=lambda i: (rank[by_id[i]["origin"]], i))
        drop.add(loser)
    kept = [dict(c, assembly_audit=verdict[c["id"]]) for c in causes if c["id"] not in drop]
    if len(kept) < cfg["min_causes"]:
        return None, "too_few_causes_after_audit"
    for i, c in enumerate(kept, 1):  # renumber so ids are contiguous
        c["id"] = f"C{i}"
        c["assembly_audit"] = {k: v for k, v in c["assembly_audit"].items() if k != "id"}
    return kept, None
