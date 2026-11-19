"""Reference agent loop. Used by the calibration gates and for evaluation.

Modes:
  interactive    the agent requests experiments up to the task budget
  no_experiment  the agent must diagnose from the opening alone (prior and
                 memorisation baseline)
  oracle         the agent is handed every prepared test result at once
                 (solvability check, and the one-shot comparison condition)
Feedback (interactive only):
  true           results from the gatekeeper
  withheld       the action is accepted but its result is not shown
"""

from __future__ import annotations

import json

from . import prompts
from .gatekeeper import Environment
from .llm import LLMError, chat_json


def _check_turn(r):
    out = []
    h = r.get("hypotheses")
    if not isinstance(h, list) or len(h) != 3 or any(not isinstance(x, dict) or not x.get("id") or not x.get("description") for x in h):
        out.append("hypotheses must be a list of exactly three objects with id and description")
    a = r.get("action")
    if not isinstance(a, dict) or a.get("type") not in ("experiment", "diagnose"):
        out.append("action.type must be 'experiment' or 'diagnose'")
    elif a["type"] == "experiment" and not (isinstance(a.get("request"), str) and a["request"].strip()):
        out.append("an experiment action needs a request")
    elif a["type"] == "diagnose" and not (isinstance(a.get("diagnosis"), str) and a["diagnosis"].strip()):
        out.append("a diagnose action needs a diagnosis")
    return out


def run_episode(task, cause_id, agent_model, gatekeeper_model, *, mode="interactive", feedback="true",
                sample=0, facts_dir=None):
    budget = task["budget"] if mode == "interactive" else 0
    env = Environment(task, cause_id, gatekeeper_model, facts_dir=facts_dir, budget=budget)
    first = "YOUR COLLEAGUE'S REPORT:\n" + env.opening()
    if mode == "oracle":
        done = "\n".join(f"- {t['name']} ({t['procedure']})\n  Result: {t['results'][cause_id]['observation']}"
                         for t in task["tests"])
        first += "\n\nThe lab has already run these tests:\n" + done
    first += (f"\n\nYou may request up to {budget} diagnostic actions." if budget
              else "\n\nNo further diagnostic actions are available. Give your diagnosis now.")
    messages = [{"role": "user", "content": first}]
    episode = {"task_id": task["task_id"], "cause_id": cause_id, "agent_model": agent_model, "mode": mode,
               "feedback": feedback, "sample": sample, "rounds": [], "final": None, "status": "incomplete"}
    tag = f"agent:{task['task_id']}:{cause_id}:{mode}:{feedback}:{sample}"
    for _ in range(budget + Environment.MAX_REFUSALS + 3):
        try:
            turn = chat_json(agent_model, messages, system=prompts.AGENT_SYSTEM, tag=tag, validate=_check_turn)
        except LLMError as e:
            episode.update(status="error", error=str(e))
            return episode
        action = turn["action"]
        rnd = {"hypotheses": turn["hypotheses"], "reasoning": turn.get("reasoning", ""), "action": action}
        episode["rounds"].append(rnd)
        if action["type"] == "diagnose":
            episode.update(final={"diagnosis": action["diagnosis"], "explanation": action.get("explanation", "")},
                           status="completed")
            return episode
        res = env.step(action["request"])
        rnd["result"] = res
        if res["kind"] == "refuse":
            shown = f"The lab did not run this: {res['observation']}"
        elif feedback == "withheld":
            shown = "The action was carried out, but its result is not available to you."
        else:
            shown = f"RESULT: {res['observation']}"
        left = res["remaining_budget"]
        shown += (f"\n\nActions remaining: {left}." if left else
                  "\n\nNo actions remain. Give your diagnosis now.")
        messages += [{"role": "assistant", "content": json.dumps(turn, ensure_ascii=False)}, {"role": "user", "content": shown}]
    episode["status"] = "no_diagnosis"
    return episode

