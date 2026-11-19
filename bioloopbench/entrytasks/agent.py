"""Reference agent loop, used by calibration and by evaluation.

Modes:
  ask            the agent sees the opening and may ask the lab questions, up
                 to the task budget, before answering
  full_protocol  the agent is handed the whole protocol and answers at once;
                 it is also asked which step the problem relates to
  symptom_only   the agent answers from the opening alone (prior-knowledge and
                 memorisation baseline)
"""

from __future__ import annotations

import json

from ..taskgen.llm import LLMError, chat_json
from . import prompts
from .env import AskEnvironment

MODES = ("ask", "full_protocol", "symptom_only")
MAX_FIXES = 3
MAX_FIX_WORDS = 60


def _system(mode):
    step = mode == "full_protocol"
    return prompts.AGENT_SYSTEM.format(
        how=prompts.AGENT_HOW[mode],
        action=prompts.AGENT_ASK_ACTION if mode == "ask" else prompts.AGENT_NO_ASK_ACTION,
        step_field=prompts.AGENT_STEP_FIELD if step else "",
        step_rule=prompts.AGENT_STEP_RULE if step else "")


def _check_turn(r, can_ask):
    a = r.get("action")
    if not isinstance(a, dict) or a.get("type") not in ("ask", "answer"):
        return ["action.type must be 'ask' or 'answer'"]
    if a["type"] == "ask":
        if not can_ask:
            return ["no questions can be asked; give your answer"]
        return [] if isinstance(a.get("question"), str) and a["question"].strip() else ["an ask action needs a question"]
    fixes = a.get("fixes")
    if not isinstance(fixes, list) or not 1 <= len(fixes) <= MAX_FIXES or any(not isinstance(f, str) or not f.strip() for f in fixes):
        return [f"an answer needs one to {MAX_FIXES} fixes, each a non-empty string"]
    return [f"fix {i} is longer than {MAX_FIX_WORDS} words; give one specific action"
            for i, f in enumerate(fixes, 1) if len(f.split()) > MAX_FIX_WORDS]


def run_episode(task, protocol, agent_model, gatekeeper_model, *, mode="ask", sample=0, facts_dir=None):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    budget = task["budget"] if mode == "ask" else 0
    env = AskEnvironment(task, protocol, gatekeeper_model, facts_dir=facts_dir, budget=budget)
    first = "YOUR COLLEAGUE'S REPORT:\n" + env.opening()
    if mode == "full_protocol":
        first += "\n\nTHE PROTOCOL THAT WAS FOLLOWED:\n" + protocol
    first += (f"\n\nYou may ask up to {budget} questions." if budget else "\n\nGive your answer now.")
    messages = [{"role": "user", "content": first}]
    episode = {"task_id": task["task_id"], "agent_model": agent_model, "mode": mode, "sample": sample,
               "rounds": [], "questions_asked": 0, "final": None, "status": "incomplete"}
    tag = f"entry_agent:{task['task_id']}:{mode}:{sample}"
    system = _system(mode)
    for _ in range(budget + AskEnvironment.MAX_REFUSALS + 3):
        can_ask = env.remaining > 0
        try:
            turn = chat_json(agent_model, messages, system=system, tag=tag, validate=lambda r: _check_turn(r, can_ask))
        except LLMError as e:
            episode.update(status="error", error=str(e))
            return episode
        action = turn["action"]
        rnd = {"reasoning": turn.get("reasoning", ""), "action": action}
        episode["rounds"].append(rnd)
        if action["type"] == "answer":
            episode.update(final={"fixes": [f.strip() for f in action["fixes"]],
                                  "step": action.get("step") if mode == "full_protocol" else None},
                           status="completed")
            return episode
        res = env.step(action["question"])
        rnd["result"] = res
        if res["consumed"]:
            episode["questions_asked"] += 1
        shown = {"answer": "THE LAB: ", "not_recorded": "THE LAB (not in the notebook): ",
                 "refuse": "The lab did not answer this: "}[res["kind"]] + res["answer"]
        left = res["remaining_budget"]
        shown += f"\n\nQuestions remaining: {left}." if left else "\n\nNo questions remain. Give your answer now."
        messages += [{"role": "assistant", "content": json.dumps(turn, ensure_ascii=False)}, {"role": "user", "content": shown}]
    episode["status"] = "no_answer"
    return episode
