"""Blind scoring: the judge is shown the candidate causes but not which is true,
and says which one a text asserts. Correctness is then decided in code."""

from __future__ import annotations

from . import prompts
from .llm import chat_json


def classify(task, text, model):
    """Return (asserted, rationale); asserted is a cause id, 'multiple' or 'none'."""
    ids = task["cause_ids"]
    causes = "\n".join(f"{c}: {task['reference_diagnosis'][c]}" for c in ids)

    def check(r):
        return [] if r.get("asserted") in (*ids, "multiple", "none") else [f"asserted must be one of {ids}, 'multiple' or 'none'"]

    r = chat_json(model, prompts.CLASSIFY.format(opening=task["opening"], causes=causes, text=text),
                  tag=f"classify:{task['task_id']}", validate=check, max_tokens=4000)
    return r["asserted"], r.get("rationale", "")


def score_episode(task, episode, model, per_round=True):
    """Add judge output to an episode record: whether the final diagnosis is
    correct, and which candidate the top-ranked hypothesis asserted each round."""
    truth = episode["cause_id"]
    out = {"correct": False, "asserted": None}
    if episode.get("final"):
        asserted, why = classify(task, episode["final"]["diagnosis"], model)
        out.update(correct=asserted == truth, asserted=asserted, rationale=why)
    if per_round:
        track = []
        for rnd in episode["rounds"]:
            top = rnd["hypotheses"][0]["description"] if rnd.get("hypotheses") else ""
            track.append(classify(task, top, model)[0] if top else "none")
        out["top_hypothesis_track"] = track
    return out
