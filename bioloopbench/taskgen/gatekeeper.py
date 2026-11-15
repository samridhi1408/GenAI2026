"""The interactive environment: a gatekeeper model answers free-text requests
from a hidden case file (after SDBench's Gatekeeper).

Requests that match a prepared test return the audited result verbatim. Other
requests get a result the gatekeeper derives from the case notes; that result
is stored per instance and shown to later gatekeeper calls, so an instance's
world stays consistent across turns and across runs.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from . import prompts
from .grounding import normalise
from .llm import chat_json
from .store import read_json, write_json

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(key):
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


def instance_id(task, cause_id):
    return f"{task['task_id']}__{cause_id}"


class Environment:
    MAX_REFUSALS = 3  # refusals beyond this consume budget, to stop free probing

    def __init__(self, task, cause_id, model, facts_dir=None, budget=None):
        if cause_id not in task["cause_ids"]:
            raise ValueError(f"{cause_id} is not a cause of {task['task_id']}")
        self.task, self.cause_id, self.model = task, cause_id, model
        self.iid = instance_id(task, cause_id)
        self.facts_path = Path(facts_dir) / f"{self.iid}.json" if facts_dir else None
        self.remaining = task["budget"] if budget is None else budget
        self.refusals = 0
        self._tests = {t["test_id"]: t for t in task["tests"]}

    def opening(self):
        return self.task["opening"]

    def _ask(self, request, facts):
        tests = "\n".join(f"{t['test_id']}: {t['name']} - {t['procedure']}" for t in self.task["tests"])
        established = "\n".join(f"- Request: {f['request']}\n  Result: {f['observation']}"
                                for f in facts.values() if f["kind"] == "result" and not f["matched_test"]) or "(none yet)"
        system = prompts.GATEKEEPER.format(opening=self.task["opening"], case_notes=self.task["case_notes"][self.cause_id],
                                           tests=tests, established=established)

        def check(r):
            out = []
            if r.get("kind") not in ("result", "refuse"):
                out.append("kind must be 'result' or 'refuse'")
            if r.get("matched_test") not in (None, *self._tests):
                out.append(f"matched_test must be null or one of {list(self._tests)}")
            if r.get("matched_test") is None and not (isinstance(r.get("observation"), str) and r["observation"].strip()):
                out.append("observation is required")
            return out

        r = chat_json(self.model, f"REQUEST FROM THE AGENT:\n{request}", system=system,
                      tag=f"gatekeeper:{self.iid}", validate=check, max_tokens=4000)
        if r["kind"] == "result" and r.get("matched_test"):
            obs = self._tests[r["matched_test"]]["results"][self.cause_id]["observation"]
            return {"kind": "result", "matched_test": r["matched_test"], "observation": obs}
        return {"kind": r["kind"], "matched_test": None, "observation": r["observation"].strip()}

    def step(self, request, record=True):
        """Carry out one request. Returns kind, observation, matched_test,
        consumed (whether budget was spent) and remaining_budget."""
        if self.remaining <= 0:
            return {"kind": "refuse", "observation": "No diagnostic actions remain.", "matched_test": None,
                    "consumed": False, "remaining_budget": 0}
        key = normalise(request)
        with _lock_for(self.iid):
            facts = read_json(self.facts_path, {}) if self.facts_path else {}
            if key in facts:
                ans = {k: facts[key][k] for k in ("kind", "matched_test", "observation")}
            else:
                ans = self._ask(request, facts)
                if record and self.facts_path:
                    facts[key] = {"request": request, **ans}
                    write_json(self.facts_path, facts)
        consumed = ans["kind"] == "result"
        if not consumed:
            self.refusals += 1
            consumed = self.refusals > self.MAX_REFUSALS
        if consumed:
            self.remaining -= 1
        return {**ans, "consumed": consumed, "remaining_budget": self.remaining}

    def informative(self, test_id):
        """Whether a prepared test's result under the true cause differs from its
        result under at least one other cause."""
        res = self._tests[test_id]["results"]
        mine = res[self.cause_id]["outcome_class"]
        return any(res[c]["outcome_class"] != mine for c in self.task["cause_ids"] if c != self.cause_id)


def public_view(task):
    """The only fields of a task an evaluated agent may be given."""
    return {"task_id": task["task_id"], "topic": task["topic"], "split": task["split"],
            "opening": task["opening"], "budget": task["budget"]}
