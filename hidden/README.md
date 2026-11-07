# Hidden benchmark records

Organize private records by task:

```text
hidden/<task_id>/
  instances/<instance_id>.json
  provenance/<provenance_id>.json
```

Use the hidden-instance and provenance schemas. Instance records link their simulator definitions. Store held-out records under `test_set/` or in evaluator-controlled storage.

Contents other than this README are ignored by Git. Restrict access to curator and evaluator processes.
