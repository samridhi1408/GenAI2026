"""Separate model submissions from trusted experimental feedback."""

from copy import deepcopy
import unittest

from bioloopbench.simulator import observation_for
from bioloopbench.trajectory import make_round
from bioloopbench.validate import ROOT, read_json, validate_agent_submission, validate_record


class FieldOwnershipTests(unittest.TestCase):
    def setUp(self):
        hypotheses = [{"hypothesis_id": f"hypothesis_{rank}", "description": f"Placeholder {rank}", "rank": rank}
                      for rank in (1, 2, 3)]
        self.before = {"hypotheses_before": deepcopy(hypotheses), "action": {"action_id": "diagnostic_action_1"}}
        self.after = {"hypotheses_after": deepcopy(hypotheses), "revision_summary": "Explanation of the received result.",
                      "evidence_ids": ["observation_1"]}
        self.lookup = {"action_results": {"state_A": {"diagnostic_action_1": {
            "observation": "Returned experimental result.", "private_note": "Evaluator-only interpretation."}}}}
        self.feedback = {"observation": observation_for(self.lookup, "state_A", "diagnostic_action_1",
                                                       observation_id="observation_1"),
                         "remaining_budget": {"diagnostic_actions": 0}}

    def test_task_contract_excludes_environment_fields_from_agent_output(self):
        for path in (ROOT / "templates/task.json", ROOT / "tasks/task_001/task.json"):
            task = read_json(path)
            agent_fields = set(task["required_output"]["per_round_fields"])
            environment_fields = set(task["environment_output"]["per_round_fields"])
            self.assertFalse(agent_fields & environment_fields)
            self.assertEqual(environment_fields, {"observation", "remaining_budget", "round_index"})
            malformed = deepcopy(task)
            malformed["required_output"]["per_round_fields"].append("observation")
            self.assertTrue(validate_record(malformed))

    def test_agent_submissions_reject_environment_owned_fields(self):
        for phase, submission in (("before_action", self.before), ("after_observation", self.after)):
            self.assertEqual(validate_agent_submission(submission, phase), [])
            for key, value in (("observation", self.feedback["observation"]),
                               ("remaining_budget", {"diagnostic_actions": 999}),
                               ("round_index", 1), ("source", "environment")):
                invalid = deepcopy(submission)
                invalid[key] = value
                self.assertTrue(validate_agent_submission(invalid, phase))
                kwargs = {"before_action": self.before, "after_observation": self.after}
                kwargs[phase] = invalid
                with self.assertRaises(ValueError):
                    make_round(round_index=1, environment_feedback=self.feedback, **kwargs)

    def test_environment_observation_is_inserted_without_private_metadata(self):
        row = make_round(round_index=1, before_action=self.before,
                         environment_feedback=self.feedback, after_observation=self.after)
        self.assertEqual(row["observation"], {"observation_id": "observation_1",
                                             "content": "Returned experimental result.", "source": "environment"})
        self.assertNotIn("private_note", row["observation"])
        self.assertEqual(row["remaining_budget"], {"diagnostic_actions": 0})
        log = read_json(ROOT / "templates/trajectory.json")
        log["rounds"] = [row]
        self.assertEqual(validate_record(log), [])
        self.feedback["observation"]["content"] = "Changed after recording."
        self.before["hypotheses_before"].clear()
        self.assertEqual(row["observation"]["content"], "Returned experimental result.")
        self.assertEqual(len(row["hypotheses_before"]), 3)

    def test_log_requires_environment_source_marker(self):
        for source in (None, "agent", "model"):
            invalid = deepcopy(self.feedback)
            if source is None:
                invalid["observation"].pop("source")
            else:
                invalid["observation"]["source"] = source
            with self.assertRaises(ValueError):
                make_round(round_index=1, before_action=self.before,
                           environment_feedback=invalid, after_observation=self.after)


if __name__ == "__main__":
    unittest.main()
