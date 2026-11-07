"""Case record integrity and observation-boundary checks; no model execution."""

from copy import deepcopy
import unittest

from bioloopbench.simulator import observation_for
from bioloopbench.validate import ROOT, read_json, validate_bundle, validate_record

PRIVATE_FILES = [
    ROOT / "hidden/task_001/hidden_instances.json",
    ROOT / "hidden/task_001/provenance.json",
    ROOT / "simulator/private/task_001/lookup.json",
]


@unittest.skipUnless(all(p.exists() for p in PRIVATE_FILES), "Private case fixtures are not present")
class CaseStructureTests(unittest.TestCase):
    def setUp(self):
        self.task = read_json(ROOT / "tasks/task_001/task.json")
        self.instances, self.provenance, self.lookup = map(read_json, PRIVATE_FILES)
        self.records = [self.task, self.instances, self.provenance, self.lookup]
        self.instance = self.instances["instances"][0]
        self.state = self.instance["ground_truth"]["hidden_state_ids"][0]

    def test_case_records_validate_with_assigned_budget(self):
        for record in self.records:
            self.assertEqual(validate_record(record), [])
        self.assertEqual(validate_bundle(self.records), [])
        self.assertEqual(self.task["budget"]["max_diagnostic_actions"], 3)
        self.assertEqual(self.instance["split"], "development")

    def test_action_mapping_coverage_and_observation_only_responses(self):
        allowed = {x["action_id"] for x in self.task["allowed_actions"]}
        self.assertEqual(allowed, {"D1", "D2", "D3", "D4"})
        self.assertEqual(set(self.lookup["action_results"][self.state]), allowed)
        for action in allowed:
            response = observation_for(self.lookup, self.state, action, observation_id="observation_1")
            self.assertEqual(set(response), {"observation_id", "content", "source"})
            self.assertEqual(response["source"], "environment")
            self.assertEqual(response, observation_for(self.lookup, self.state, action, observation_id="observation_1"))
            self.assertNotIn(self.state, response["content"])
        contaminated = deepcopy(self.lookup)
        contaminated["action_results"][self.state]["D1"]["interpretation_hidden_from_agent"] = "private interpretation"
        self.assertTrue(validate_record(contaminated))
        self.assertEqual(set(observation_for(contaminated, self.state, "D1", observation_id="observation_1")),
                         {"observation_id", "content", "source"})
        with self.assertRaises(ValueError):
            observation_for(self.lookup, self.state, "unlisted_action", observation_id="observation_1")

    def test_public_files_do_not_contain_answer_keys_or_source_material(self):
        public = (ROOT / "tasks/task_001/task.json").read_text() + (ROOT / "tasks/task_001/instruction.md").read_text()
        for private_value in (self.state, self.instance["ground_truth"]["diagnosis"],
                              '"ground_truth"', '"action_results"', '"source_ids"'):
            self.assertNotIn(private_value, public)
        for source in self.provenance["sources"]:
            self.assertNotIn(source["title"], public)
            if source["doi"]:
                self.assertNotIn(source["doi"], public)

    def test_provenance_preserves_sources_and_caveats(self):
        self.assertEqual(len(self.provenance["sources"]), 21)
        self.assertEqual({x["action_id"] for x in self.provenance["action_evidence"]},
                         {x["action_id"] for x in self.task["allowed_actions"]})
        self.assertTrue(all(x["caveats"] for x in self.provenance["action_evidence"]))

    def test_missing_mapping_and_unknown_sources_are_rejected(self):
        broken_lookup = deepcopy(self.lookup)
        broken_lookup["action_results"][self.state].pop("D4")
        errors = validate_bundle([self.task, self.instances, self.provenance, broken_lookup])
        self.assertTrue(any("action coverage" in e for e in errors))
        broken_provenance = deepcopy(self.provenance)
        broken_provenance["action_evidence"][0]["source_ids"] = ["missing_source"]
        errors = validate_bundle([self.task, self.instances, broken_provenance, self.lookup])
        self.assertTrue(any("Unknown evidence source" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
