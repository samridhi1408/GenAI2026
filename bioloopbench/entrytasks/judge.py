"""Scoring: the judge says which reference fix, if any, each proposed fix
matches. Ranks and step localisation are then computed in code."""

from __future__ import annotations

import re

from ..taskgen.llm import chat_json
from . import prompts


def match_fixes(task, proposals, model):
    """Return one entry per proposal: a reference fix id, or 'none'."""
    if not proposals:
        return []
    ids = [f["id"] for f in task["fixes"]]
    labels = [f"P{i}" for i in range(1, len(proposals) + 1)]

    def check(r):
        m = r.get("matches")
        if not isinstance(m, list) or [x.get("proposal") for x in m if isinstance(x, dict)] != labels:
            return [f"matches must list exactly {labels}, in order"]
        return [f"{x['proposal']}: matches must be one of {ids} or 'none'" for x in m if x.get("matches") not in (*ids, "none")]

    r = chat_json(model, prompts.MATCH.format(
        opening=task["opening"],
        fixes="\n".join(f"{f['id']}: {f['fix']}" for f in task["fixes"]),
        proposals="\n".join(f"{p}: {text}" for p, text in zip(labels, proposals))),
        tag=f"entry_match:{task['task_id']}", validate=check, max_tokens=4000)
    return [x["matches"] for x in r["matches"]]


def _step_key(s):
    return re.sub(r"[^0-9a-z]", "", str(s).lower())


def score_episode(task, episode, model):
    """Return the scores for one episode.

    first_match_rank   1-based rank of the first proposal matching a reference
                       fix, or None
    top1, top3         whether that rank is 1, or at most 3
    fixes_recovered    how many distinct reference fixes were matched
    step_correct       whether the named step is one the entry refers to; None
                       when the task has no step reference or none was asked for
    """
    out = {"matches": [], "first_match_rank": None, "top1": False, "top3": False, "fixes_recovered": 0,
           "n_reference_fixes": len(task["fixes"]), "questions_asked": episode.get("questions_asked", 0),
           "step_correct": None}
    final = episode.get("final")
    if not final:
        return out
    matches = match_fixes(task, final["fixes"], model)
    hits = [i for i, m in enumerate(matches, 1) if m != "none"]
    out.update(matches=matches, first_match_rank=hits[0] if hits else None, top1=bool(hits) and hits[0] == 1,
               top3=bool(hits) and hits[0] <= 3, fixes_recovered=len({m for m in matches if m != "none"}))
    if task.get("step_refs") and final.get("step") not in (None, ""):
        out["step_correct"] = _step_key(final["step"]) in {_step_key(s) for s in task["step_refs"]}
    return out
