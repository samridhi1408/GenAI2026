"""Checks that run models against an extracted task before it is accepted.

  judge sanity   the judge must match a paraphrase of every reference fix to
                 that fix, and a deliberately vague piece of advice to none. A
                 task the judge cannot score reliably is rejected.
  symptom only   the solver answers from the opening alone, several times. The
                 result is recorded, not used to reject: a task the solver
                 gets right without asking anything is kept and flagged
                 ``symptom_solved`` so results can be reported with and
                 without such tasks.
"""

from __future__ import annotations

import math

from ..taskgen.llm import chat_json
from . import prompts
from .agent import run_episode
from .judge import match_fixes, score_episode


def calibrate_task(task, protocol, cfg):
    """Fill task['calibration'] and set task['status'] to accepted or rejected."""
    m = cfg["models"]
    ids = [f["id"] for f in task["fixes"]]
    cal = {}

    def check(r):
        p = r.get("paraphrases")
        return [] if isinstance(p, dict) and sorted(p) == sorted(ids) and isinstance(r.get("vague"), str) \
            else [f"need paraphrases for exactly {ids} and a 'vague' string"]
    para = chat_json(m["generator"], prompts.PARAPHRASE.format(
        fixes="\n".join(f"{f['id']}: {f['fix']}" for f in task["fixes"])),
        tag=f"entry_paraphrase:{task['task_id']}", validate=check)
    got = match_fixes(task, [para["paraphrases"][i] for i in ids] + [para["vague"]], m["judge"])
    cal["judge_sanity"] = {"paraphrase_matched": dict(zip(ids, got[:-1])), "vague_matched": got[-1],
                           "passed": got[:-1] == ids and got[-1] == "none"}
    if not cal["judge_sanity"]["passed"]:
        task.update(calibration=cal, status="rejected", reasons=["judge_cannot_score"])
        return task

    n = cfg["symptom_only_samples"]
    ranks = []
    for s in range(n):
        ep = run_episode(task, protocol, m["solver"], m["gatekeeper"], mode="symptom_only", sample=s)
        ranks.append(score_episode(task, ep, m["judge"])["first_match_rank"])
    top1 = sum(r == 1 for r in ranks)
    cal["symptom_only"] = {"solver": m["solver"], "first_match_ranks": ranks, "top1_hits": top1,
                           "top3_hits": sum(r is not None for r in ranks)}
    task.update(calibration=cal, status="accepted", reasons=[],
                symptom_solved=n > 0 and top1 >= math.ceil(n * 2 / 3))
    return task
