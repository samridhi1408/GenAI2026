# Automated task generation

`bioloopbench/taskgen/` turns a list of seed topics into interactive troubleshooting tasks. It is separate from the hand-built `task_001` records and the fixed-menu controller, which are unchanged.

## What a generated task is

A task is one failed experiment with three to five candidate root causes. Every cause is a separate hidden instance that shares the same opening, so the opening alone never identifies the answer. The agent requests diagnostic actions in free text, a gatekeeper model answers from the hidden case file, and a judge decides which candidate the final diagnosis asserts.

## Where each part comes from

| Part | Source | Check |
|---|---|---|
| Setup, observed failure | Paraphrased from the anchor article | Auditor model reads it against the article; must not share a run of more than 9 words with it |
| Cause and remedy (`own`) | Anchor article's Troubleshooting entry | Verbatim quote must string-match the entry; auditor confirms support |
| Cause (`transplanted`) | Another article's entry for the same symptom | Quote string-matches its own article; applicability to this setup judged by two models |
| Diagnostic results | Written by the generator model | Auditor model reviews every test-by-cause cell; one repair round, then failing causes are dropped |
| Results for unprepared requests | Derived by the gatekeeper at run time | Stored per instance and replayed, so they stay consistent |

Diagnostic results are model-derived, not taken from the papers, which do not report them. Only the cause and remedy are source-faithful.

## Stages

1. **corpus** — search Europe PMC for open-access STAR Protocols articles on the topic, split by `test_cutoff_date`, sample, download and parse.
2. **extract** — the generator selects qualifying troubleshooting entries; deterministic quote checks; the auditor rejects anything unsupported, off-topic or leaking.
3. **build** — group cases by symptom, assemble competing causes, write and audit the case file.
4. **calibrate** — judge sanity, no-experiment baseline, oracle solvability, gatekeeper probes.

Instances end as `accepted`, `rejected`, or `opening_solved` (the solver named the cause from the opening alone; kept in the record, excluded from the default set). `funnel.json` counts every outcome and rejection reason.

## Running

Set a model for each role in `configs/taskgen.json` as `provider:model_id` (`gemini`, `anthropic`, `openai`; see `bioloopbench/taskgen/llm.py`). Use a different model family for `auditor` than for `generator` if you can; the audit is worth less when both share blind spots.

```sh
.venv/bin/python -m bioloopbench.taskgen.pipeline --config configs/taskgen.json --out generated/dev1 --topics cloning
```

Omit `--topics` to run every topic in `configs/seed_topics.json`. `--stages` selects stages. Model replies are cached under the run directory, so re-running resumes. `generated/` is ignored by Git; `private/` holds answers and must not be given to an evaluated agent, which may see only `public/`.

## Memorisation controls

- The test split holds only articles published on or after `test_cutoff_date`.
- The no-experiment baseline is recorded for every task.
- Agent-facing text is paraphrased and never names the article.
- Evaluated agents get no web access; the gatekeeper is their only source of information.

## Tests

`tests/test_taskgen.py` covers every stage with scripted model replies and needs no network or API key.
