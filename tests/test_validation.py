"""Generic schema and consistency tests using neutral placeholders only."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import tempfile
import unittest

from bioloopbench.validate import ROOT, main, read_json, validate_bundle, validate_record


def template(kind):
    return read_json(ROOT / "templates" / f"{kind}.json")


def ranked_hypotheses():
    return [{"hypothesis_id": f"hypothesis_{rank}",
             "description": f"Placeholder explanation {rank}", "rank": rank}
            for rank in (1, 2, 3)]


def trajectory():
    """One neutral record shape, not a simulated experiment."""
    record = template("trajectory")
    record["rounds"] = [{
        "round_index": 1,
        "hypotheses_before": ranked_hypotheses(),
        "action": {"action_id": "diagnostic_action_1"},
        "observation": {"observation_id": "observation_1", "content": "TBD", "source": "environment"},
        "hypotheses_after": ranked_hypotheses(),
        "revision_summary": "TBD", "evidence_ids": ["observation_1"],
        "remaining_budget": {"diagnostic_actions": 0},
    }]
    return record


class ValidationTests(unittest.TestCase):
    def test_neutral_templates_and_their_links(self):
        records = [template(kind) for kind in ("task", "hidden_instance", "provenance", "trajectory")]
        for record in records:
            self.assertEqual(validate_record(record), [])
        self.assertEqual(validate_bundle(records), [])

    def test_public_task_rejects_hidden_fields(self):
        record = template("task")
        record["ground_truth"] = {"hidden_state_ids": ["hidden_state_A"]}
        self.assertTrue(validate_record(record))
        record = template("task")
        record["allowed_actions"].append(deepcopy(record["allowed_actions"][0]))
        self.assertIn("allowed_actions contains duplicate action IDs", validate_record(record))

    def test_required_fields_types_and_versions(self):
        for field, value in (("schema_version", "unknown"), ("budget", {"max_diagnostic_actions": -1}),
                             ("budget", {"max_diagnostic_actions": True})):
            record = template("task")
            record[field] = value
            self.assertTrue(validate_record(record))
        record = template("hidden_instance")
        del record["ground_truth"]
        self.assertTrue(validate_record(record))
        for malformed in ([], None, {}, {"record_type": []}):
            self.assertTrue(validate_record(malformed))

    def test_hidden_record_does_not_impose_single_cause(self):
        record = template("hidden_instance")
        record["ground_truth"]["hidden_state_ids"].append("hidden_state_B")
        self.assertEqual(validate_record(record), [])

    def test_review_metadata_requires_reviewer_and_date_when_approved(self):
        record = template("provenance")
        record["curation"]["status"] = "approved"
        self.assertTrue(validate_record(record))
        record["curation"].update(reviewer="reviewer_1", reviewed_on="2026-11-06")
        self.assertEqual(validate_record(record), [])
        record["curation"]["reviewed_on"] = "2026-13-40"
        self.assertTrue(validate_record(record))

    def test_exactly_three_hypotheses_in_both_lists(self):
        record = trajectory()
        self.assertEqual(validate_record(record), [])
        for field in ("hypotheses_before", "hypotheses_after"):
            for count in (0, 1, 2, 4):
                with self.subTest(field=field, count=count):
                    record = trajectory()
                    record["rounds"][0][field] = [
                        {"hypothesis_id": f"hypothesis_{i}", "description": f"Placeholder {i}", "rank": i}
                        for i in range(1, count + 1)]
                    self.assertTrue(validate_record(record))

    def test_rank_order_and_no_ties(self):
        for field in ("hypotheses_before", "hypotheses_after"):
            for ranks in ((1, 1, 3), (1, 3, 2), (0, 2, 3), (1, 2, 4), (True, 2, 3),
                          ("1", 2, 3), (1.5, 2, 3)):
                with self.subTest(field=field, ranks=ranks):
                    record = trajectory()
                    for hypothesis, rank in zip(record["rounds"][0][field], ranks):
                        hypothesis["rank"] = rank
                    self.assertTrue(validate_record(record))

    def test_reranking_and_agent_generated_replacements_are_allowed(self):
        record = trajectory()
        before = record["rounds"][0]["hypotheses_before"]
        after = [deepcopy(before[2]), deepcopy(before[0]),
                 {"hypothesis_id": "hypothesis_4", "description": "Different placeholder explanation", "rank": 3}]
        for rank, hypothesis in enumerate(after, 1):
            hypothesis["rank"] = rank
        record["rounds"][0]["hypotheses_after"] = after
        self.assertEqual(validate_record(record), [])

    def test_duplicate_hypotheses_and_confidence_fields_are_rejected(self):
        for field in ("hypotheses_before", "hypotheses_after"):
            record = trajectory()
            record["rounds"][0][field][1]["hypothesis_id"] = "hypothesis_1"
            self.assertTrue(validate_record(record))
            record = trajectory()
            record["rounds"][0][field][1]["description"] = "  PLACEHOLDER EXPLANATION 1  "
            self.assertTrue(validate_record(record))
            for unwanted_field, value in (("confidence", .5), ("probability", .5), ("confidence_level", "high")):
                record = trajectory()
                record["rounds"][0][field][0][unwanted_field] = value
                self.assertTrue(validate_record(record))

    def test_rank_contract_has_no_provided_candidates(self):
        record = template("task")
        self.assertEqual(record["required_output"]["hypothesis_format"],
                         {"representation": "ranked", "count": 3, "source": "agent_generated"})
        record["candidate_hypotheses"] = ["Placeholder explanation"]
        self.assertTrue(validate_record(record))

    def test_evidence_chronology_and_final_record(self):
        record = trajectory()
        record["rounds"][0]["evidence_ids"] = ["future_observation"]
        self.assertTrue(validate_record(record))
        record["rounds"][0]["evidence_ids"] = ["initial_failure", "observation_1"]
        record["status"] = "completed"
        self.assertTrue(validate_record(record))
        record["final"] = {"diagnosis": "TBD", "evidence_ids": ["observation_1"], "explanation": "Placeholder explanation."}
        self.assertEqual(validate_record(record), [])
        with_confidence = deepcopy(record)
        with_confidence["final"]["confidence"] = .5
        self.assertTrue(validate_record(with_confidence))
        record["final"]["evidence_ids"] = ["unknown_observation"]
        self.assertTrue(validate_record(record))

    def test_round_sequence_and_observation_uniqueness(self):
        record = trajectory()
        row = deepcopy(record["rounds"][0])
        record["rounds"].append(row)
        errors = validate_record(record)
        self.assertTrue(any("contiguous" in e for e in errors))
        self.assertTrue(any("not unique" in e for e in errors))
        self.assertTrue(any("budget must decrease" in e for e in errors))

    def test_cross_record_references_and_split_groups(self):
        task, instance, provenance = template("task"), template("hidden_instance"), template("provenance")
        self.assertEqual(validate_bundle([task, instance, provenance, trajectory()]), [])
        other = deepcopy(instance)
        other.update(instance_id="instance_002", split="test")
        self.assertTrue(any("multiple splits" in e for e in validate_bundle([task, instance, other, provenance])))
        instance["task_version"] = "different_version"
        self.assertTrue(any("version mismatch" in e for e in validate_bundle([task, instance, provenance])))
        instance["provenance_ids"] = ["missing_provenance"]
        self.assertTrue(any("provenance" in e for e in validate_bundle([task, instance, provenance])))
        self.assertTrue(any("Unknown task" in e for e in validate_bundle([instance])))

    def test_bundle_rejects_unlisted_actions_and_budget_mismatch(self):
        task, instance, provenance = template("task"), template("hidden_instance"), template("provenance")
        log = trajectory()
        log["rounds"][0]["action"]["action_id"] = "unlisted_action"
        log["rounds"][0]["remaining_budget"]["diagnostic_actions"] = 1
        errors = validate_bundle([task, instance, provenance, log])
        self.assertTrue(any("Unpermitted action" in e for e in errors))
        self.assertTrue(any("budget disagrees" in e for e in errors))

    def test_repeated_actions_count_against_the_budget(self):
        task = read_json(ROOT / "tasks/task_001/task.json")
        self.assertTrue(task["allow_repeated_actions"])
        log = trajectory()
        log["instance_id"] = None
        row = deepcopy(log["rounds"][0])
        row["action"]["action_id"] = task["allowed_actions"][0]["action_id"]
        log["rounds"] = []
        for index in range(1, 4):
            repeated = deepcopy(row)
            repeated["round_index"] = index
            repeated["observation"]["observation_id"] = f"observation_{index}"
            repeated["evidence_ids"] = [f"observation_{index}"]
            repeated["remaining_budget"]["diagnostic_actions"] = 3 - index
            log["rounds"].append(repeated)
        self.assertEqual(validate_record(log), [])
        self.assertEqual(validate_bundle([task, log]), [])
        restricted_task = deepcopy(task)
        restricted_task["allow_repeated_actions"] = False
        self.assertTrue(any("Repeated action" in e for e in validate_bundle([restricted_task, log])))
        extra = deepcopy(log["rounds"][-1])
        extra["round_index"] = 4
        extra["observation"]["observation_id"] = "observation_4"
        extra["evidence_ids"] = ["observation_4"]
        log["rounds"].append(extra)
        self.assertTrue(any("budget disagrees" in e for e in validate_bundle([task, log])))

    def test_strict_json_and_cli_error_status(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            for content in ('{"record_type":"task","record_type":"trajectory"}', '{"rank":NaN}'):
                path.write_text(content)
                with self.assertRaises(ValueError):
                    read_json(path)
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(main([str(path)]), 1)
            path.write_text('{"record_type":"unknown"}')
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main([str(path)]), 1)
                self.assertEqual(main([str(path.parent / "missing.json")]), 1)

    def test_default_cli_checks_only_neutral_templates(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main([]), 0)
        self.assertIn("Validated 4 record(s)", output.getvalue())


if __name__ == "__main__":
    unittest.main()
