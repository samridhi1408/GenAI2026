# Environment interface

The environment interface exposes public task context and provides a boundary for permitted diagnostic actions, observations, and budget tracking.

The Python protocol is declared in `bioloopbench/interfaces.py`. Environment implementations should return public observations while keeping hidden state and curator metadata private.

The agent supplies hypotheses and an action request, then responds to returned feedback. The environment supplies observation content, observation IDs, source labels, round indices, and budget bookkeeping. Use `bioloopbench/trajectory.py` to combine these separately owned fields into a trajectory round.

## Run a session

From the repository root:

```sh
.venv/bin/python -m bioloopbench.controller --task task_001
```

The controller loads the task and its private instance/mapping, prints public task information, and accepts one JSON object per line. Hypothesis descriptions in the examples below are placeholders to replace with your own explanations.

1. Submit three ranked hypotheses and an action:

```json
{"type":"act","hypotheses_before":[{"hypothesis_id":"h1","description":"Your first explanation","rank":1},{"hypothesis_id":"h2","description":"Your second explanation","rank":2},{"hypothesis_id":"h3","description":"Your third explanation","rank":3}],"action":{"action_id":"D1"}}
```

The controller returns an observation with an ID such as `observation_1`, its environment source, and the remaining budget. Each accepted action consumes one budget slot, including repeated actions. Invalid requests leave the budget and session unchanged.

2. Respond to that observation with revised hypotheses:

```json
{"type":"update","hypotheses_after":[{"hypothesis_id":"h2","description":"Your second explanation","rank":1},{"hypothesis_id":"h1","description":"Your first explanation","rank":2},{"hypothesis_id":"h3","description":"Your third explanation","rank":3}],"revision_summary":"Explain what the received observation supports or contradicts.","evidence_ids":["observation_1"]}
```

Rank changes depend on the evidence; the ordering above illustrates the format. Preserve the same order when the result does not warrant a change. Submit an update after every observation, including the last observation at zero remaining budget.

3. Continue with another `act` message or submit the final diagnosis:

```json
{"type":"finish","diagnosis":"Your final diagnosis","evidence_ids":["observation_1"],"explanation":"Explain how the received evidence supports the diagnosis."}
```

The final diagnosis should agree with the latest rank-1 hypothesis. The controller validates fields and evidence references; scientific interpretation belongs to evaluation.

Use `{"type":"abort"}` to stop without a final diagnosis. End-of-input or Ctrl-C also closes the session as aborted. An action awaiting its revision is preserved under `pending_round` in the saved trajectory.

## Session storage and transport

Trajectories are saved after each accepted submission to `runs/<run_id>/trajectory.json`. The startup response includes `run_id`. Set a different evaluator-only destination with `--output /path/to/trajectory.json`; existing output files are not overwritten at startup. If multiple hidden instances are available, the operator selects one with `--instance`.

Standard output contains only public responses. Exit code 0 indicates a completed final submission, 1 an aborted session, and 2 a setup/storage error. Private instance IDs in evaluator logs, hidden states, reference answers, mappings, and provenance remain outside the participant response stream.

Run the controller as a trusted process. Restrict evaluated clients to its input/output interface; keep the repository, process memory, startup arguments, and trajectory files inaccessible to those clients. The local manual-input CLI is not an operating-system sandbox.
