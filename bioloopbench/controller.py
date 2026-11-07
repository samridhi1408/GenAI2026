"""Run a benchmark session using JSON submissions on standard input."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from uuid import uuid4

from .simulator import observation_for
from .trajectory import make_round
from .validate import ROOT, parse_json, read_json, validate_agent_submission, validate_record, validators


class InteractionError(ValueError):
    """Rejected participant input; the session state remains unchanged."""


class Controller:
    """Trusted session state. Expose submit/start responses, not this object."""

    def __init__(self, task, instance, lookup, instructions=""):
        for record in (task, instance, lookup):
            errors = validate_record(record)
            if errors:
                raise ValueError("Invalid benchmark definition: " + "; ".join(errors))
        for record in (instance, lookup):
            if (record["task_id"], record["task_version"]) != (task["task_id"], task["task_version"]):
                raise ValueError("Task references do not match")
        if instance["simulator"] != {"kind": "lookup", "definition_ref": lookup["simulator_id"]}:
            raise ValueError("Instance does not reference this lookup simulator")
        states = instance["ground_truth"]["hidden_state_ids"]
        if len(states) != 1 or states[0] not in lookup["action_results"]:
            raise ValueError("A single explicitly mapped hidden state is required")
        allowed = {action["action_id"] for action in task["allowed_actions"]}
        if set(lookup["action_results"][states[0]]) != allowed:
            raise ValueError("Simulator action coverage does not match the task")
        if task["budget"]["max_diagnostic_actions"] is None:
            raise ValueError("The diagnostic-action budget must be assigned")
        if "allow_repeated_actions" not in task:
            raise ValueError("The task must specify allow_repeated_actions")
        self._task = deepcopy(task)
        self._lookup = deepcopy(lookup)
        self._state = states[0]
        self._instructions = instructions
        self._remaining = task["budget"]["max_diagnostic_actions"]
        self._performed = set()
        self._log = {
            "record_type": "trajectory", "schema_version": task["schema_version"],
            "run_id": "run_" + uuid4().hex, "task_id": task["task_id"],
            "task_version": task["task_version"], "instance_id": instance["instance_id"],
            "status": "draft", "rounds": [], "pending_round": None, "final": None,
        }

    @property
    def run_id(self):
        return self._log["run_id"]

    @property
    def closed(self):
        return self._log["status"] != "draft"

    def _next(self):
        if self.closed:
            return "none"
        if self._log["pending_round"] is not None:
            return "update"
        return "act_or_finish" if self._remaining else "finish"

    def _response(self, event, **fields):
        return {"event": event, **deepcopy(fields),
                "remaining_budget": {"diagnostic_actions": self._remaining},
                "expected_submission": self._next()}

    def start(self):
        return self._response("session_started", run_id=self.run_id,
                              task=self._task, instructions=self._instructions)

    def error_response(self, message):
        return self._response("error", message=message)

    @staticmethod
    def _check_submission(payload, phase):
        errors = validate_agent_submission(payload, phase)
        if errors:
            raise InteractionError("Invalid submission: " + "; ".join(errors))
        field = "hypotheses_before" if phase == "before_action" else "hypotheses_after"
        hypotheses = payload[field]
        ids = [h["hypothesis_id"] for h in hypotheses]
        descriptions = [h["description"].strip().casefold() for h in hypotheses]
        if len(set(ids)) != 3 or len(set(descriptions)) != 3 or not all(descriptions):
            raise InteractionError("Provide three distinct hypothesis IDs and descriptions")

    def _available_evidence(self):
        ids = {"initial_failure"}
        ids.update(row["observation"]["observation_id"] for row in self._log["rounds"])
        if self._log["pending_round"] is not None:
            ids.add(self._log["pending_round"]["observation"]["observation_id"])
        return ids

    def _check_evidence(self, payload):
        if not set(payload["evidence_ids"]) <= self._available_evidence():
            raise InteractionError("Evidence IDs must refer to observations already received")

    def submit(self, message):
        """Accept an act/update/finish/abort envelope with agent-owned fields."""
        if self.closed:
            raise InteractionError("Session is already closed")
        if not isinstance(message, dict) or not isinstance(message.get("type"), str):
            raise InteractionError("Provide a JSON object with a type field")
        payload = {key: value for key, value in message.items() if key != "type"}
        command = message["type"]
        if command == "act":
            return self._act(payload)
        if command == "update":
            return self._update(payload)
        if command == "finish":
            return self._finish(payload)
        if command == "abort":
            if payload:
                raise InteractionError("Abort accepts only the type field")
            return self.abort()
        raise InteractionError("Unknown type; use act, update, finish, or abort")

    def _act(self, payload):
        if self._log["pending_round"] is not None:
            raise InteractionError("Update hypotheses after the previous observation first")
        if self._remaining == 0:
            raise InteractionError("The action budget is exhausted; submit a final diagnosis")
        self._check_submission(payload, "before_action")
        action = payload["action"]
        action_id = action["action_id"]
        if action_id not in {a["action_id"] for a in self._task["allowed_actions"]}:
            raise InteractionError("The selected action is not permitted")
        if action.get("parameters"):
            raise InteractionError("These lookup actions do not accept parameters")
        if action_id in self._performed and not self._task["allow_repeated_actions"]:
            raise InteractionError("The selected action cannot be repeated")
        index = len(self._log["rounds"]) + 1
        observation = observation_for(self._lookup, self._state, action_id,
                                      observation_id=f"observation_{index}")
        pending = deepcopy({"round_index": index, **payload, "observation": observation,
                            "remaining_budget": {"diagnostic_actions": self._remaining - 1}})
        self._log["pending_round"] = pending
        self._remaining -= 1
        self._performed.add(action_id)
        return self._response("observation", round_index=index, observation=observation)

    def _update(self, payload):
        pending = self._log["pending_round"]
        if pending is None:
            raise InteractionError("Select an action and receive its observation before updating")
        self._check_submission(payload, "after_observation")
        self._check_evidence(payload)
        row = make_round(
            round_index=pending["round_index"],
            before_action={"hypotheses_before": pending["hypotheses_before"], "action": pending["action"]},
            environment_feedback={"observation": pending["observation"], "remaining_budget": pending["remaining_budget"]},
            after_observation=payload,
        )
        self._log["rounds"].append(row)
        self._log["pending_round"] = None
        return self._response("revision_recorded")

    def _finish(self, payload):
        if self._log["pending_round"] is not None:
            raise InteractionError("Record your revision after the last observation before finishing")
        validator = validators()["trajectory"].evolve(
            schema={"$ref": "urn:bioloopbench:schema:common#/$defs/final"})
        errors = [error.message for error in validator.iter_errors(payload)]
        if errors:
            raise InteractionError("Invalid final submission: " + "; ".join(errors))
        required = self._task.get("required_output", {}).get("final_fields", [])
        if not set(required) <= set(payload):
            raise InteractionError("Provide every final field required by the task")
        self._check_evidence(payload)
        self._log["final"] = deepcopy(payload)
        self._log["status"] = "completed"
        return self._response("finished", final=payload)

    def abort(self):
        if self.closed:
            raise InteractionError("Session is already closed")
        self._log["status"] = "aborted"
        return self._response("aborted")

    def trajectory(self):
        """Evaluator-only snapshot, including an interrupted pending round."""
        return deepcopy(self._log)

    def save(self, path):
        """Atomically write an evaluator-only trajectory."""
        errors = validate_record(self._log)
        if errors:
            raise ValueError("Invalid trajectory: " + "; ".join(errors))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                             prefix=".trajectory-", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(self._log, stream, indent=2, ensure_ascii=False, allow_nan=False)
                stream.write("\n")
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()


def load_controller(task_id, instance_id=None, root=ROOT):
    """Curator-side loading. Instance selection is not an agent command."""
    if not isinstance(task_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", task_id):
        raise ValueError("Invalid task ID")
    root = Path(root)
    task = read_json(root / "tasks" / task_id / "task.json")
    instances = read_json(root / "hidden" / task_id / "hidden_instances.json")
    errors = validate_record(instances)
    if errors:
        raise ValueError("Invalid hidden-instance records")
    candidates = [item for item in instances["instances"] if instance_id is None or item["instance_id"] == instance_id]
    if len(candidates) != 1:
        raise ValueError("Select exactly one available instance with --instance")
    lookup = read_json(root / "simulator/private" / task_id / "lookup.json")
    instructions = (root / "tasks" / task_id / "instruction.md").read_text()
    return Controller(task, candidates[0], lookup, instructions)


def run_session(controller, path, input_stream, output_stream):
    """JSON-lines transport. Only public responses go to output_stream."""
    def emit(value):
        output_stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        output_stream.flush()

    controller.save(path)
    emit(controller.start())
    try:
        for line in input_stream:
            if not line.strip():
                continue
            try:
                message = parse_json(line)
                response = controller.submit(message)
            except (InteractionError, ValueError) as error:
                emit(controller.error_response(str(error)))
                continue
            controller.save(path)
            emit(response)
            if controller.closed:
                return 0 if response["event"] == "finished" else 1
    except KeyboardInterrupt:
        pass
    response = controller.abort()
    controller.save(path)
    emit(response)
    return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="task_001", help="Public task ID")
    parser.add_argument("--instance", help="Instance ID selected by the operator")
    parser.add_argument("--output", type=Path, help="Evaluator-only trajectory path")
    args = parser.parse_args(argv)
    try:
        controller = load_controller(args.task, args.instance)
        config = read_json(ROOT / "configs/benchmark.json")
        output = args.output or ROOT / config["paths"]["local_trajectories"] / controller.run_id / "trajectory.json"
        if output.exists():
            raise ValueError("Output already exists; choose a new trajectory path")
        return run_session(controller, output, sys.stdin, sys.stdout)
    except (OSError, ValueError) as error:
        print(f"Controller error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
