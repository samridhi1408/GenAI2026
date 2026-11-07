"""Session control and JSON-lines transport checks using neutral input."""

from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from bioloopbench.controller import Controller, InteractionError, load_controller, run_session
from bioloopbench.validate import ROOT, read_json, validate_bundle, validate_record


def hypotheses():
    return [{"hypothesis_id": f"hypothesis_{rank}", "description": f"Placeholder explanation {rank}", "rank": rank}
            for rank in (1, 2, 3)]


def act(action_id="diagnostic_action_1"):
    return {"type": "act", "hypotheses_before": hypotheses(), "action": {"action_id": action_id}}


def update(observation_id="observation_1"):
    return {"type": "update", "hypotheses_after": hypotheses(), "revision_summary": "Interpretation of the received result.",
            "evidence_ids": [observation_id]}


def finish(evidence_ids=None):
    return {"type": "finish", "diagnosis": "Placeholder explanation 1",
            "evidence_ids": ["initial_failure"] if evidence_ids is None else evidence_ids,
            "explanation": "Diagnosis based on the available observations."}


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.task = read_json(ROOT / "templates/task.json")
        self.task["budget"]["max_diagnostic_actions"] = 3
        self.task["allow_repeated_actions"] = True
        self.instance = read_json(ROOT / "templates/hidden_instance.json")
        self.instance["ground_truth"]["diagnosis"] = "Private reference diagnosis."
        self.instance["simulator"] = {"kind": "lookup", "definition_ref": "simulator_001"}
        self.lookup = {
            "record_type": "simulator_lookup", "schema_version": self.task["schema_version"],
            "simulator_id": "simulator_001", "task_id": self.task["task_id"],
            "task_version": self.task["task_version"], "deterministic": True,
            "action_results": {self.instance["ground_truth"]["hidden_state_ids"][0]: {
                "diagnostic_action_1": {"observation": "A recorded experimental result."}}},
        }
        self.controller = Controller(self.task, self.instance, self.lookup, "Public interaction instructions.")

    def unchanged_after_rejection(self, message):
        before = self.controller.trajectory()
        with self.assertRaises(InteractionError):
            self.controller.submit(message)
        self.assertEqual(self.controller.trajectory(), before)

    def test_three_repeated_actions_require_updates_and_final_at_zero_budget(self):
        contents, ids = [], []
        for index in (1, 2, 3):
            response = self.controller.submit(act())
            contents.append(response["observation"]["content"])
            ids.append(response["observation"]["observation_id"])
            self.assertEqual(response["remaining_budget"]["diagnostic_actions"], 3 - index)
            self.assertEqual(response["expected_submission"], "update")
            self.unchanged_after_rejection(act())
            self.unchanged_after_rejection(finish())
            self.controller.submit(update(response["observation"]["observation_id"]))
        self.assertEqual(len(set(contents)), 1)
        self.assertEqual(len(set(ids)), 3)
        self.unchanged_after_rejection(act())
        response = self.controller.submit(finish(ids))
        self.assertEqual(response["event"], "finished")
        self.assertNotIn("correct", response)
        self.assertEqual(validate_record(self.controller.trajectory()), [])
        provenance = read_json(ROOT / "templates/provenance.json")
        self.assertEqual(validate_bundle([self.task, self.instance, self.lookup, provenance,
                                         self.controller.trajectory()]), [])
        self.unchanged_after_rejection(finish())

    def test_invalid_input_does_not_spend_budget_or_change_phase(self):
        for message in (None, [], {}, {"type": "reset"}, act("unknown_action"), update()):
            self.unchanged_after_rejection(message)
        invalid = act()
        invalid["observation"] = {"content": "Model-authored result"}
        self.unchanged_after_rejection(invalid)
        invalid = act()
        invalid["hypotheses_before"][1]["rank"] = 1
        self.unchanged_after_rejection(invalid)
        invalid = act()
        invalid["hypotheses_before"][1]["hypothesis_id"] = "hypothesis_1"
        self.unchanged_after_rejection(invalid)
        invalid = act()
        invalid["action"]["parameters"] = {"unconfigured_parameter": 1}
        self.unchanged_after_rejection(invalid)
        self.assertEqual(self.controller.start()["remaining_budget"]["diagnostic_actions"], 3)

    def test_updates_and_finals_cannot_cite_unseen_evidence_or_omit_explanation(self):
        self.controller.submit(act())
        self.unchanged_after_rejection(update("observation_99"))
        invalid = update()
        invalid["remaining_budget"] = {"diagnostic_actions": 999}
        self.unchanged_after_rejection(invalid)
        self.controller.submit(update())
        self.unchanged_after_rejection(finish(["observation_99"]))
        invalid = finish()
        invalid.pop("explanation")
        self.unchanged_after_rejection(invalid)
        self.assertEqual(self.controller.submit(finish(["observation_1"]))["event"], "finished")
        self.assertEqual(len(self.controller.trajectory()["rounds"]), 1)

    def test_public_responses_are_copied_and_keep_hidden_metadata_private(self):
        response = self.controller.start()
        response["task"]["allowed_actions"].clear()
        response["remaining_budget"]["diagnostic_actions"] = 999
        self.assertEqual(len(self.controller.start()["task"]["allowed_actions"]), 1)
        self.assertEqual(self.controller.start()["remaining_budget"]["diagnostic_actions"], 3)
        response = self.controller.submit(act())
        response["observation"]["content"] = "Changed outside the controller."
        self.assertEqual(self.controller.trajectory()["pending_round"]["observation"]["content"],
                         "A recorded experimental result.")
        public = json.dumps([self.controller.start(), response, self.controller.error_response("Invalid request")])
        for value in ("Private reference diagnosis.", "hidden_state_A", "instance_001", "action_results", "ground_truth"):
            self.assertNotIn(value, public)

    def test_abort_preserves_pending_feedback_and_log_cannot_be_marked_completed(self):
        self.controller.submit(act())
        self.controller.abort()
        log = self.controller.trajectory()
        self.assertEqual(log["status"], "aborted")
        self.assertIsNotNone(log["pending_round"])
        self.assertEqual(validate_record(log), [])
        log["status"] = "completed"
        log["final"] = {key: value for key, value in finish().items() if key != "type"}
        self.assertTrue(validate_record(log))
        self.unchanged_after_rejection(update())

    def test_repeat_restriction_is_configurable(self):
        self.task["allow_repeated_actions"] = False
        self.controller = Controller(self.task, self.instance, self.lookup)
        self.controller.submit(act())
        self.controller.submit(update())
        self.unchanged_after_rejection(act())

    def test_loader_requires_an_explicit_mapped_instance_and_assigned_policies(self):
        invalid = deepcopy(self.task)
        invalid["budget"]["max_diagnostic_actions"] = None
        with self.assertRaises(ValueError):
            Controller(invalid, self.instance, self.lookup)
        invalid = deepcopy(self.task)
        invalid.pop("allow_repeated_actions")
        with self.assertRaises(ValueError):
            Controller(invalid, self.instance, self.lookup)
        invalid = deepcopy(self.instance)
        invalid["task_version"] = "wrong_version"
        with self.assertRaises(ValueError):
            Controller(self.task, invalid, self.lookup)
        with self.assertRaises(ValueError):
            load_controller("../hidden")

    def test_json_lines_transport_saves_each_step_without_private_stdout(self):
        commands = ['{"type":"act","type":"update"}', json.dumps(act()),
                    json.dumps(update()), json.dumps(finish(["observation_1"]))]
        output, checkpoints = io.StringIO(), []
        original_save = self.controller.save
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.json"

            def save_and_check(destination):
                original_save(destination)
                checkpoints.append(read_json(destination))

            with patch.object(self.controller, "save", side_effect=save_and_check):
                code = run_session(self.controller, path, io.StringIO("\n".join(commands)), output)
            self.assertEqual(code, 0)
            self.assertEqual(read_json(path)["status"], "completed")
            self.assertTrue(any(log["pending_round"] is not None for log in checkpoints))
        events = [json.loads(line)["event"] for line in output.getvalue().splitlines()]
        self.assertEqual(events, ["session_started", "error", "observation", "revision_recorded", "finished"])
        self.assertNotIn("ground_truth", output.getvalue())
        self.assertNotIn("Private reference diagnosis.", output.getvalue())

    def test_eof_aborts_and_saves_the_unfinished_round(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.json"
            code = run_session(self.controller, path, io.StringIO(json.dumps(act()) + "\n"), io.StringIO())
            self.assertEqual(code, 1)
            log = read_json(path)
            self.assertEqual(log["status"], "aborted")
            self.assertIsNotNone(log["pending_round"])
            self.assertEqual(validate_record(log), [])

    @unittest.skipUnless((ROOT / "hidden/task_001/hidden_instances.json").exists(), "Private case fixtures are not present")
    def test_executable_cli_with_existing_case(self):
        task = read_json(ROOT / "tasks/task_001/task.json")
        instance = read_json(ROOT / "hidden/task_001/hidden_instances.json")["instances"][0]
        commands = [act(task["allowed_actions"][0]["action_id"]), update(), finish(["observation_1"])]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.json"
            result = subprocess.run([sys.executable, "-m", "bioloopbench.controller", "--task", "task_001",
                                     "--output", str(path)], cwd=ROOT, text=True,
                                    input="\n".join(json.dumps(x) for x in commands),
                                    capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(read_json(path)["status"], "completed")
            self.assertEqual(validate_record(read_json(path)), [])
            self.assertNotIn("ground_truth", result.stdout)
            for hidden_state in instance["ground_truth"]["hidden_state_ids"]:
                self.assertNotIn(hidden_state, result.stdout)
            self.assertNotIn("source_ids", result.stdout)


if __name__ == "__main__":
    unittest.main()
