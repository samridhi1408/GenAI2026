# Entry tasks

`bioloopbench/entrytasks/` builds one troubleshooting task from each qualifying troubleshooting entry of a STAR Protocols article. It replaces the sibling-cause design in `bioloopbench/taskgen/` (see `docs/taskgen.md`), which lost almost all of its input to the requirement for three or more rival causes per task. It reuses that package's corpus stage, model client, grounding checks and run store.

## What a task is

The agent is told what went wrong in an experiment. It may ask a limited number of questions about how the experiment was carried out, then submits a ranked list of up to three fixes. A fix is correct when it matches one the authors report for that entry.

Nothing the agent sees or is scored against is invented by a model, except paraphrases, which are checked against the source.

## Where each part comes from

| Part | Source | Check |
|---|---|---|
| Context and problem (the opening) | Paraphrased from the entry and protocol | Problem quote must string-match the entry; no run of more than `max_shared_run_words` words shared with the article; auditor confirms it is supported, observable at the bench, does not give the fix away and does not identify the article |
| Reference fixes | The entry's own solution text | Each fix carries a verbatim quote that must string-match the entry; auditor keeps only fixes that are supported and concrete |
| Step references | Step numbers the entry mentions | Each must appear in the entry text |
| Answers to the agent's questions | Gatekeeper model reading the protocol text only | The gatekeeper never sees the troubleshooting section; answers are stored per task and replayed |

## Stages

1. **corpus** — shared with `taskgen`: search Europe PMC, split by `test_cutoff_date`, sample, download and parse. A seed topic only decides which articles are sampled.
2. **extract** — the generator turns every entry into a task or sets it aside with a reason code (`not_observable`, `no_concrete_fix`, `other`). Deterministic checks run first; then the auditor reviews each draft. A task ends `extracted` or `rejected`.
3. **calibrate** — two checks per extracted task:
   - *Judge sanity:* the judge must match a paraphrase of every reference fix to that fix and a deliberately vague piece of advice to none. Otherwise the task is rejected (`judge_cannot_score`).
   - *Symptom-only baseline:* the solver answers from the opening alone, `symptom_only_samples` times. The result is recorded, not used to reject. A task solved at rank 1 in at least two thirds of samples is flagged `symptom_solved`.

`funnel.json` counts every outcome, skip code, rejection reason and failure type.

## Running

```sh
python -m bioloopbench.entrytasks.pipeline --config configs/entrytasks.json --out generated/pilot1 --topics cloning
python -m bioloopbench.entrytasks.evaluate --run generated/pilot1 --model gemini:gemini-3.8-flash --split test
```

`--stages` selects stages. Model replies are cached under the run directory, so re-running resumes. API keys are described in `docs/api-keys.md`; note the expiry date of the Gemini key there.

Run layout:

```
<run>/
  corpus/xml/                    cached article XML
  topics/<topic>/articles.json   sampled articles and their split
  topics/<topic>/entries.json    task ids produced from this topic's articles
  protocols/<pmcid>.txt          protocol text the gatekeeper answers from
  private/<task_id>.json         full record: fixes, quotes, provenance, audit, calibration
  public/<task_id>.json          what an evaluated agent may see
  facts/<task_id>.json           gatekeeper answers already given, replayed on repeat questions
  results/<model>/               episodes, scores and summary from evaluate
  funnel.json                    counts of what was kept and rejected, and why
```

`private/` holds the answers and must not be given to an evaluated agent.

## Evaluation modes and scores

| Mode | What the agent gets |
|---|---|
| `ask` | The opening, and up to `question_budget` questions to the gatekeeper |
| `full_protocol` | The opening and the whole protocol; also asked which step the problem relates to |
| `symptom_only` | The opening alone (prior-knowledge and memorisation baseline) |

Scores per episode: rank of the first proposal that matches a reference fix (`top1`, `top3`), number of distinct reference fixes recovered, questions asked, and step accuracy (only in `full_protocol` mode, only for tasks with a step reference). Rates are over all tasks attempted, so an episode that errors counts as a miss. Summaries report `top1` both over all tasks and excluding `symptom_solved` tasks.

Refused questions do not use up the budget until `MAX_REFUSALS` is passed, after which each one does.

## Memorisation controls

- The test split holds only articles published on or after `test_cutoff_date`.
- The symptom-only baseline is recorded for every task and reported alongside results.
- Agent-facing text is paraphrased and the auditor rejects openings that identify the article.
- Evaluated agents get no web access; the gatekeeper is their only source of information.

## Tests

`tests/test_entrytasks.py` covers extraction checks, the gatekeeper environment, the judge and scoring, the agent loop, calibration, both pipeline stages and evaluation, with scripted model replies. It needs no network or API key.

```sh
python -m unittest discover -s tests
```

## Known limits

- With every role on one model family (as in the current Gemini-only config), the auditor and judge share blind spots with the generator and solver. Use a different family for `auditor` and `judge` when credits allow.
- Direct Gemini calls report no cost, so `max_cost_usd` only limits OpenRouter roles.
- An article is dropped whole if the generator refuses it or fails the deterministic checks three times.
