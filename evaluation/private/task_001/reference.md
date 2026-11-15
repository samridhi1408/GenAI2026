# Evaluator references

- Accepted diagnosis and hidden-state identifier: `hidden/task_001/hidden_instances.json`, instance `instance_001`, `ground_truth`.
- Evidence strengths, interpretations, and biological caveats: `hidden/task_001/provenance.json`, `action_evidence`.
- Evidence-combination guidance: the same provenance record, `evidence_integration_notes`.
- Returned observations: simulator `simulator_001`, stored in `simulator/private/task_001/lookup.json`.

Use these canonical records to assess the diagnosis and supporting evidence. Keep this reference and its linked records restricted to the evaluator.

## Ranked-hypothesis evaluation

Each round records exactly three agent-generated hypotheses before and after the observation, using stable hypothesis IDs and ranks 1–3. The benchmark does not provide candidate explanations. Track whether the correct hidden cause rises in rank when supported, whether contradicted hypotheses fall, and whether an earlier incorrect rank-1 hypothesis persists despite contradictory evidence. Record when a hypothesis is introduced, revised, or dropped; a hidden cause absent from the three entries is not an observed fourth-ranked hypothesis.

Assess rank changes against the diagnostic evidence and accepted diagnosis; ranking changes alone do not establish correctness.
