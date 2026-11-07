# Task interface rules

Public definition: `tasks/task_001/task.json`.

- Accept diagnostic action IDs from the public definition's `allowed_actions` list.
- Read the diagnostic-action limit from the public definition's `budget`.
- Require a configured action budget before starting an interaction.
- Permit repeated actions according to `allow_repeated_actions` in the public task. Count every selection against the budget, including repeats. Repeated selections use the same deterministic observation mapping, with a unique observation ID for each returned result.
- Require exactly three agent-generated hypotheses at ranks 1, 2, and 3 before the action and after its observation, using the trajectory schema. Candidate hypotheses remain agent-generated.
- Obtain the selected action's observation from the trusted simulator and return only that observation to the agent.
- Populate `observation` with the returned content, an environment-assigned ID, and `source: environment`. The environment also populates the round index and remaining budget.
- Validate agent submissions separately before the action and after the observation. Reject submissions containing environment-owned fields.
- Keep hidden instances, lookup definitions, provenance, and evaluator references outside the agent's accessible files and tools.

Use the existing trajectory schema to record interactions. The environment and simulator protocols are declared in `bioloopbench/interfaces.py`; observation extraction is implemented in `bioloopbench/simulator.py`.

Use `bioloopbench/trajectory.py` to assemble a round from separate agent submissions and trusted environment feedback. Read observation content from the active instance's simulator response, never from a model-authored trajectory.

The executable session controller is `bioloopbench/controller.py`. Launch it with `.venv/bin/python -m bioloopbench.controller --task task_001`; input messages are documented in `environment/README.md`.
