"""Validate benchmark record structure and cross-record consistency."""

import argparse
from functools import lru_cache
import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_FILES = {
    "task": "task.schema.json",
    "hidden_instance": "hidden_instance.schema.json",
    "hidden_instances": "hidden_instances.schema.json",
    "provenance": "provenance.schema.json",
    "trajectory": "trajectory.schema.json",
    "simulator_lookup": "simulator_lookup.schema.json",
}


def parse_json(source):
    def reject_constant(value):
        raise ValueError(f"Non-JSON numeric constant: {value}")

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(source, parse_constant=reject_constant,
                      object_pairs_hook=unique_keys)


def read_json(path):
    return parse_json(Path(path).read_text())


@lru_cache(maxsize=1)
def validators():
    schemas = {p.name: read_json(p) for p in (ROOT / "schemas").glob("*.schema.json")}
    registry = Registry()
    for schema in schemas.values():
        Draft202012Validator.check_schema(schema)
        registry = registry.with_resource(schema["$id"], Resource.from_contents(schema))
    return {kind: Draft202012Validator(schemas[name], registry=registry,
                                       format_checker=FormatChecker())
            for kind, name in SCHEMA_FILES.items()}


def duplicates(values):
    return len(values) != len(set(values))


def validate_agent_submission(submission, phase):
    """Validate agent-authored fields without accepting environment fields."""
    definitions = {"before_action": "agent_before_action",
                   "after_observation": "agent_after_observation"}
    if phase not in definitions:
        return ["Unknown agent submission phase"]
    validator = validators()["trajectory"].evolve(
        schema={"$ref": "urn:bioloopbench:schema:common#/$defs/" + definitions[phase]})
    return [error.message for error in validator.iter_errors(submission)]


def validate_record(record):
    """Return structural and within-record consistency errors."""
    if (not isinstance(record, dict) or not isinstance(record.get("record_type"), str)
            or record["record_type"] not in SCHEMA_FILES):
        return ["record_type must identify a supported benchmark record"]
    try:
        json.dumps(record, allow_nan=False)
    except (ValueError, TypeError) as error:
        return [str(error)]
    kind = record["record_type"]
    errors = [f"{'.'.join(map(str, e.absolute_path)) or '$'}: {e.message}"
              for e in sorted(validators()[kind].iter_errors(record), key=lambda e: str(list(e.path)))]
    if errors:
        return errors
    if kind == "task" and duplicates([a["action_id"] for a in record["allowed_actions"]]):
        errors.append("allowed_actions contains duplicate action IDs")
    if kind == "provenance" and duplicates([s["source_id"] for s in record["sources"]]):
        errors.append("sources contains duplicate source IDs")
    if kind == "provenance" and duplicates([x["action_id"] for x in record.get("action_evidence", [])]):
        errors.append("action_evidence contains duplicate action IDs")
    if kind == "trajectory":
        seen = {"initial_failure"}
        previous_budget = None
        rows = list(record["rounds"])
        if record.get("pending_round") is not None:
            rows.append(record["pending_round"])
        for index, row in enumerate(rows, 1):
            if row["round_index"] != index:
                errors.append("round_index must be contiguous starting at one")
            for field in ("hypotheses_before", "hypotheses_after"):
                if field not in row:
                    continue
                if duplicates([h["hypothesis_id"] for h in row[field]]):
                    errors.append(f"round {index}: duplicate hypothesis IDs in {field}")
                if duplicates([h["description"].strip().casefold() for h in row[field]]):
                    errors.append(f"round {index}: duplicate hypothesis descriptions in {field}")
            observation_id = row["observation"]["observation_id"]
            if observation_id in seen:
                errors.append(f"round {index}: observation ID is not unique")
            seen.add(observation_id)
            if not set(row.get("evidence_ids", [])) <= seen:
                errors.append(f"round {index}: unknown or future evidence ID")
            budget = row["remaining_budget"]["diagnostic_actions"]
            if previous_budget is not None and budget >= previous_budget:
                errors.append(f"round {index}: remaining budget must decrease")
            previous_budget = budget
        if record["final"] and not set(record["final"]["evidence_ids"]) <= seen:
            errors.append("final: unknown evidence ID")
    return errors


def expand_records(records):
    """Accept plural instance files without changing individual instance records."""
    return [item for record in records for item in
            (record["instances"] if record["record_type"] == "hidden_instances" else [record])]


def validate_bundle(records):
    """Check links across an already structurally valid collection of records."""
    records = expand_records(records)
    errors, index, groups = [], {}, {}
    key_fields = {"task": "task_id", "hidden_instance": "instance_id",
                  "provenance": "provenance_id", "trajectory": "run_id",
                  "simulator_lookup": "simulator_id"}
    for record in records:
        key = (record["record_type"], record[key_fields[record["record_type"]]])
        if key in index:
            errors.append(f"Duplicate record ID: {key[1]}")
        index[key] = record
    for record in records:
        kind = record["record_type"]
        if kind == "task":
            if record["budget"]["max_diagnostic_actions"] is None:
                errors.append(f"Unassigned diagnostic-action budget: {record['task_id']}")
            continue
        task = index.get(("task", record["task_id"]))
        if task is None:
            errors.append(f"Unknown task reference: {record['task_id']}")
        elif kind != "provenance" and record["task_version"] != task["task_version"]:
            errors.append(f"Task version mismatch for {record['task_id']}")
        if kind == "simulator_lookup" and task is not None:
            allowed = {a["action_id"] for a in task["allowed_actions"]}
            for results in record["action_results"].values():
                if set(results) != allowed:
                    errors.append(f"Simulator action coverage differs from task: {record['simulator_id']}")
        if kind == "provenance":
            sources = {s["source_id"] for s in record["sources"]}
            allowed = {a["action_id"] for a in task["allowed_actions"]} if task else set()
            for evidence in record.get("action_evidence", []):
                if evidence["action_id"] not in allowed:
                    errors.append(f"Unknown provenance action: {evidence['action_id']}")
                if not set(evidence["source_ids"]) <= sources:
                    errors.append(f"Unknown evidence source for action: {evidence['action_id']}")
        if kind == "hidden_instance":
            group = record["scenario_group_id"]
            if group in groups and groups[group] != record["split"]:
                errors.append(f"Scenario group spans multiple splits: {group}")
            groups[group] = record["split"]
            for provenance_id in record["provenance_ids"]:
                provenance = index.get(("provenance", provenance_id))
                if provenance is None or provenance["task_id"] != record["task_id"]:
                    errors.append(f"Unresolved or mismatched provenance: {provenance_id}")
            if record["simulator"]["kind"] == "lookup":
                reference = record["simulator"]["definition_ref"]
                lookup = index.get(("simulator_lookup", reference))
                if (lookup is None or lookup["task_id"] != record["task_id"]
                        or lookup["task_version"] != record["task_version"]):
                    errors.append(f"Unresolved or mismatched simulator: {reference}")
                elif not set(record["ground_truth"]["hidden_state_ids"]) <= set(lookup["action_results"]):
                    errors.append(f"Simulator lacks the instance's hidden state: {reference}")
        if kind == "trajectory":
            instance_id = record["instance_id"]
            if instance_id is not None:
                instance = index.get(("hidden_instance", instance_id))
                if (instance is None or instance["task_id"] != record["task_id"]
                        or instance["task_version"] != record["task_version"]):
                    errors.append(f"Unresolved or mismatched instance: {instance_id}")
            if task is not None:
                allowed = {a["action_id"] for a in task["allowed_actions"]}
                performed = set()
                rows = list(record["rounds"])
                if record.get("pending_round") is not None:
                    rows.append(record["pending_round"])
                for row in rows:
                    if row["action"]["action_id"] not in allowed:
                        errors.append(f"Unpermitted action: {row['action']['action_id']}")
                    action_id = row["action"]["action_id"]
                    if task.get("allow_repeated_actions") is False and action_id in performed:
                        errors.append(f"Repeated action is not permitted: {action_id}")
                    performed.add(action_id)
                    maximum = task["budget"]["max_diagnostic_actions"]
                    if maximum is not None and row["remaining_budget"]["diagnostic_actions"] != maximum - row["round_index"]:
                        errors.append("Trajectory remaining budget disagrees with task budget")
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Individual JSON records or directories")
    parser.add_argument("--repo", action="store_true", help="Check benchmark roots and cross-record links")
    args = parser.parse_args(argv)
    if args.repo and args.paths:
        parser.error("Use --repo or explicit paths, not both")
    roots = ([ROOT / name for name in ("tasks", "hidden", "test_set", "runs", "simulator/private")]
             if args.repo else args.paths or [ROOT / "templates"])
    validators()  # Check schema definitions even when the data roots are empty.
    files, errors, records = set(), [], []
    for path in roots:
        if not path.exists():
            if not args.repo:
                errors.append(f"Path does not exist: {path}")
        elif path.is_file():
            files.add(path.resolve())
        else:
            files.update(p.resolve() for p in path.rglob("*.json"))
    for path in sorted(files):
        try:
            record = read_json(path)
        except (OSError, ValueError) as error:
            errors.append(f"{path}: {error}")
            continue
        record_errors = validate_record(record)
        errors.extend(f"{path}: {error}" for error in record_errors)
        if not record_errors:
            records.append(record)
            if args.repo:
                if path.is_relative_to(ROOT / "tasks") and record["record_type"] != "task":
                    errors.append(f"{path}: private record placed in public tasks directory")
                if (path.is_relative_to(ROOT / "test_set") and any(
                        item["record_type"] == "hidden_instance" and item["split"] != "test"
                        for item in expand_records([record]))):
                    errors.append(f"{path}: held-out instance must use split=test")
    if not errors and (args.repo or not args.paths):
        errors.extend(validate_bundle(records))
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"Validated {len(records)} record(s); schema definitions checked.")
    if not records:
        print("No records found in the selected paths.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
