# Task instructions

Task ID: `task_001`

## Goal

TBD

## Protocol and context

TBD

## Observed failure

TBD

## Permitted interaction and budget

TBD: describe the actions and budget defined in `task.json`.

## Hypothesis ranking

Generate exactly three distinct hypotheses from the supplied context and observations. The benchmark does not supply candidate hypotheses. Each entry contains an agent-created `hypothesis_id`, a `description`, and a `rank` of 1, 2, or 3. Return entries in rank order, with rank 1 most likely.

Record exactly three hypotheses in both `hypotheses_before` and `hypotheses_after` for every round. After each diagnostic observation, re-rank the same or revised set and explain the changes. Retain IDs for continuing hypotheses and use new IDs for different explanations. Use rank order only, without confidence scores, probabilities, percentages, or confidence categories.

Submit `hypotheses_before` and `action`, then wait for the environment to return the observation. Respond with `hypotheses_after`, `revision_summary`, and `evidence_ids`. The environment populates the observation, its ID and source, round index, and remaining budget. Submit only the fields listed in `required_output`; the environment/logger combines these with feedback in the trajectory.

Keep the ranking unchanged when the new observation does not justify a change.

## Final response

Provide `diagnosis`, `evidence_ids`, and `explanation` as specified in `task.json`. The final diagnosis should reflect the latest rank-1 hypothesis. Keep instructions consistent with the public task record and keep hidden benchmark information outside the instructions.
