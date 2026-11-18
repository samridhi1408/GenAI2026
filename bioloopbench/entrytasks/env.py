"""The interactive environment: a gatekeeper answers questions about how the
experiment was carried out, from the protocol text alone.

The gatekeeper is never shown the troubleshooting section, so it cannot give
the answer away; it holds only what a lab notebook would. Answers are stored
per task and replayed, so every agent that asks the same question gets the
same answer.
"""

from __future__ import annotations

import threading
from pathlib import Path

from ..taskgen.grounding import normalise
from ..taskgen.llm import chat_json
from ..taskgen.store import read_json, write_json
from . import prompts

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()

KINDS = ("answer", "not_recorded", "refuse")


def _lock_for(key):
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


class AskEnvironment:
    MAX_REFUSALS = 3  # refusals beyond this use up a question, to stop free probing

    def __init__(self, task, protocol, model, facts_dir=None, budget=None):
        self.task, self.protocol, self.model = task, protocol, model
        self.facts_path = Path(facts_dir) / f"{task['task_id']}.json" if facts_dir else None
        self.remaining = task["budget"] if budget is None else budget
        self.refusals = 0

    def opening(self):
        return self.task["opening"]

    def _ask(self, question, facts):
        established = "\n".join(f"- Question: {f['question']}\n  Answer: {f['answer']}"
                                for f in facts.values() if f["kind"] != "refuse") or "(none yet)"
        system = prompts.GATEKEEPER.format(opening=self.task["opening"], protocol=self.protocol, established=established)

        def check(r):
            out = []
            if r.get("kind") not in KINDS:
                out.append(f"kind must be one of {list(KINDS)}")
            if not (isinstance(r.get("answer"), str) and r["answer"].strip()):
                out.append("answer is required")
            return out

        r = chat_json(self.model, f"QUESTION FROM THE SCIENTIST:\n{question}", system=system,
                      tag=f"entry_gatekeeper:{self.task['task_id']}", validate=check, max_tokens=4000)
        return {"kind": r["kind"], "answer": r["answer"].strip()}

    def step(self, question, record=True):
        """Answer one question. Returns kind, answer, consumed (whether a
        question was used up) and remaining_budget."""
        if self.remaining <= 0:
            return {"kind": "refuse", "answer": "No questions remain.", "consumed": False, "remaining_budget": 0}
        key = normalise(question)
        with _lock_for(self.task["task_id"]):
            facts = read_json(self.facts_path, {}) if self.facts_path else {}
            if key in facts:
                ans = {k: facts[key][k] for k in ("kind", "answer")}
            else:
                ans = self._ask(question, facts)
                if record and self.facts_path:
                    facts[key] = {"question": question, **ans}
                    write_json(self.facts_path, facts)
        consumed = ans["kind"] != "refuse"
        if not consumed:
            self.refusals += 1
            consumed = self.refusals > self.MAX_REFUSALS
        if consumed:
            self.remaining -= 1
        return {**ans, "consumed": consumed, "remaining_budget": self.remaining}


def public_view(task):
    """The only fields of a task an evaluated agent may be given."""
    return {"task_id": task["task_id"], "split": task["split"], "opening": task["opening"], "budget": task["budget"]}
