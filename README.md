# BioLoopBench

BioLoopBench evaluates how scientific agents use experimental feedback to troubleshoot failed biological experiments. It captures hypothesis formation, diagnostic action selection, evidence-driven revision, and final diagnosis under a limited experimental budget.

## Repository structure

```text
configs/                  Benchmark metadata and data locations
schemas/                  JSON Schemas for benchmark records
templates/                Authoring templates
tasks/                    Agent-visible instructions and task definitions
hidden/                   Private instances and curator provenance
environment/              Public interaction interface
simulator/                Hidden state and observation interface
evaluation/               Metric definitions
test_set/                 Held-out benchmark records
bioloopbench/             Session controller, validation, and logging utilities
tests/                    Controller and validation tests
docs/design.md            Record conventions
```

## Benchmark records

The hierarchy is **task family → task → hidden instance**. Multiple hidden instances can share one task. A trajectory records an interaction with an instance.

| Record | Contents | Visibility |
|---|---|---|
| `task` | Goal, protocol/context, observed failure, permitted actions, budget | Agent-visible |
| `hidden_instance` | Ground truth, simulator reference, split, scenario group, curation | Curator/evaluator |
| `provenance` | Sources, construction notes, leakage controls, review | Curator/evaluator |
| `trajectory` | Agent hypotheses/actions/revisions and environment observations/budget, plus final diagnosis | Evaluator |

Schemas define the required fields. Templates provide editable records. Shared field definitions are in `schemas/common.schema.json`.

The interaction format uses exactly three agent-generated hypotheses, ranked 1 through 3, before and after each diagnostic observation. Stable IDs track continuing hypotheses across rank changes. The final response contains a diagnosis, supporting observation IDs, and an explanation.

The agent submits hypotheses and an action, receives an environment-generated observation, and submits its revision. The environment/logger records both submissions and its own feedback in the trajectory. `required_output` lists agent fields; `environment_output` lists environment-populated fields.

## Setup and validation

Requires Python 3.10+.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m bioloopbench.validate
.venv/bin/python -m unittest discover -s tests -v
```

The default validation command checks schemas and templates. To validate individual records or the repository's benchmark records:

```sh
.venv/bin/python -m bioloopbench.validate path/to/record.json
.venv/bin/python -m bioloopbench.validate --repo
```

`--repo` checks JSON records in `tasks/`, `hidden/`, `test_set/`, `runs/`, and `simulator/private/`, including cross-record references and lookup coverage. Unassigned diagnostic-action budgets are reported as validation errors.

## Run an interaction

```sh
.venv/bin/python -m bioloopbench.controller --task task_001
```

Submit JSON messages to select actions, revise ranked hypotheses after observations, and give a final diagnosis. The controller tracks the budget and records the trajectory under `runs/`. Private case files must be available to the trusted controller. See the [command format and examples](environment/README.md).

Hidden records, private simulator/evaluator data, held-out data, and local trajectories are ignored by Git by default. An evaluated agent receives public task files and selected observations through a restricted interface; private repository files remain accessible only to the curator/evaluator.

See [record conventions](docs/design.md) for field semantics and validation coverage.
