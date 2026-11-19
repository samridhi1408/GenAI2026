# Entry tasks: handoff report

Status as of 19 November 2026. Branch `entry-tasks`.

## Summary

We have **5,161 troubleshooting tasks** generated automatically from **1,448 open-access STAR Protocols articles**. Each task is one real troubleshooting entry from a published protocol:

- the model is told what went wrong (the **opening**),
- it proposes up to three **fixes**, ranked,
- it is scored against the fixes the article's authors report (the **reference fixes**).

How much the model is allowed to know about the experiment before answering is the experimental variable. We run three settings, from least to most information: **symptom only**, **ask** (it may question a simulated lab), and **full protocol**.

The data is in a private Hugging Face dataset repo, `awhecmu/bioloopbench-entry5k`. Ask Andre for access. It is not in this Git repository, because it contains the reference answers and full article texts.

## 1. What a task looks like

The example below is **made up** (it is the fixture from our unit tests), but it has exactly the shape of a real task.

**Opening** (shown to the model):

> A backbone was cut with a type IIS enzyme and joined to an insert using T4 ligase. The product was put into chemically competent DH5alpha and spread on selective agar.
>
> What went wrong: The selective plate was empty the next morning.

**Reference fixes** (hidden; used for scoring):

| ID | Fix | Cause, if the authors give one |
|---|---|---|
| F1 | Repeat with a fresh aliquot of competent cells. | The cells have lost competence. |
| F2 | Use fresh ligase buffer. | — |

**Step reference** (hidden): the protocol step the entry points to, here step `4`.

**Protocol** (optionally shown): the article's method text with the troubleshooting section removed, about 9,400 words on average.

A model's answer is a ranked list such as `["Thaw a new aliquot of cells.", "Check the incubator."]`. A judge model decides which reference fix, if any, each proposal matches. Here the first proposal matches F1, so the task is solved at rank 1.

To read real tasks, open any file in `private/` of the dataset, or build a readable sample of evaluation conversations with `python -m bioloopbench.entrytasks.transcripts` (section 4).

## 2. How much data we have

| | Accepted tasks | Articles |
|---|---|---|
| Total | 5,161 | 1,448 |
| Development split (published before 1 July 2026) | 4,573 | 1,296 |
| Test split (published 1 July 2026 or later) | 588 | 152 |

Topics. A topic only decides which articles were sampled; every troubleshooting entry in a sampled article was considered, so a task's actual technique can differ from its topic.

| Topic id | Topic | Accepted | Of which test |
|---|---|---|---|
| `cloning` | Plasmid cloning and bacterial transformation | 683 | 116 |
| `pcr` | PCR and qPCR amplification | 729 | 67 |
| `western` | Western blot and protein detection | 703 | 74 |
| `protein` | Recombinant protein expression and purification | 729 | 28 |
| `crispr` | CRISPR genome editing | 226 | 9 |
| `delivery` | Transfection and viral transduction | 707 | 32 |
| `immunostaining` | Immunofluorescence staining | 745 | 221 |
| `seqlib` | Sequencing library preparation | 639 | 41 |

Each task also has a free-text `technique` label written by the generator. There are about 3,400 distinct labels, so they are descriptive rather than a clean taxonomy. A count of them is in `technique_counts.csv` in the run folder on Andre's machine (not in the uploaded archive).

By publication year:

| 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|
| 213 | 918 | 905 | 664 | 655 | 847 | 959 |

Shape of the tasks:

- **Fixes per task:** 2.3 on average. 31% have one, 35% two, 20% three, 15% four or more.
- **Step reference:** 3,740 tasks (72%) name at least one protocol step.
- **Tasks per article:** 3.6 on average, median 3, maximum 18.
- **Opening length:** about 60 words.
- **Solvable from the opening alone:** 1,346 tasks (26%) are flagged `symptom_solved` (see section 5). In the test split it is 149 of 588.

## 3. Using the data

Download `entry5k_full.tar.gz` from the Hugging Face repo and unpack it into `generated/`, giving `generated/entry5k/`.

| Path | Contents |
|---|---|
| `manifest.csv`, `manifest.jsonl` | **Start here.** One row per task: status, split, dates, identifiers, topic, counts, models used. |
| `public/<task_id>.json` | What an evaluated model may see: `task_id`, `split`, `opening`, `budget`. Accepted tasks only. |
| `private/<task_id>.json` | The full record, including reference fixes. Never give this to an evaluated model. |
| `protocols/<pmcid>.txt` | Protocol text for the full-protocol setting and for the simulated lab. |
| `corpus/xml/<pmcid>.xml` | The source articles as downloaded from Europe PMC. |
| `topics/<topic>/articles.json` | Which articles were sampled for each topic, with their metadata. |
| `funnel.json` | Counts of what was kept and dropped at every stage. |
| `llm_cache/` | Every model reply from generation. Lets the pipeline be re-run without paying again. |

A task ID is `<PMCID>_e<entry number>`, for example `PMC1234567_e3`.

Fields of a private record that matter most:

| Field | Meaning |
|---|---|
| `opening` | `context` + "What went wrong:" + `problem`. Paraphrased from the article. |
| `fixes` | List of `{id, fix, fix_quote, cause}`. `fix_quote` is the verbatim span of the article that supports the fix. |
| `step_refs` | Protocol step numbers the entry mentions. |
| `problem_quote` | Verbatim span of the entry describing the problem. |
| `status`, `reasons` | `accepted` or `rejected`, with rejection reasons. |
| `symptom_solved` | True if the solver got it right from the opening alone in at least 2 of 3 attempts. |
| `calibration` | The judge sanity check and the three symptom-only attempts. |
| `date`, `article` | Publication date and the article metadata block (below). |
| `split` | `test` or `development` under the 1 July 2026 cutoff. |
| `budget` | Questions allowed in the ask setting (6). |

### Dates and re-splitting

Every accepted task has a publication date, so tasks can be re-split by any cutoff using the manifest alone.

| Manifest column | Source |
|---|---|
| `date` | Earliest publication date stated in the article. This decides `split`. |
| `first_publication_date`, `first_index_date`, `pub_year` | Europe PMC's record. |
| `pub_date_epub`, `pub_date_collection` | The article's own dates, as written (a collection date may have no day). |
| `date_received`, `date_accepted` | **Empty for every task**: the article files did not carry manuscript history. |

Articles do not overlap between topics, and all tasks from one article share a split.

## 4. How we evaluate models

The three settings form a ladder. The same model answers the same task with more information each time.

| Setting | The model gets | What it measures |
|---|---|---|
| `symptom_only` | The opening | Prior knowledge and memorisation. This is the floor. |
| `ask` | The opening, plus up to 6 questions to a simulated lab | Whether the model can find out what it needs by asking |
| `full_protocol` | The opening and the whole protocol | Reasoning with everything available. This is the practical ceiling. Also asked which step the problem relates to. |

In the ask setting a **gatekeeper** model plays the lab. It answers only from the protocol text and never sees the troubleshooting section, so it cannot give the answer away. Its answers are stored and replayed, so two models asking the same question get the same answer. The model may answer at any point; six questions is a cap, not a requirement.

Scores per task:

- **Top-1 / top-3:** whether the first matching proposal is at rank 1, or within the three.
- **Fixes recovered:** how many distinct reference fixes were matched.
- **Questions asked** (ask setting).
- **Step accuracy** (full-protocol setting, tasks with a step reference).

Summaries report top-1 both over all tasks and excluding `symptom_solved` tasks. The second number is the more informative one: it removes tasks that can be answered without knowing anything about the experiment.

Running an evaluation:

```sh
python -m bioloopbench.entrytasks.evaluate --run generated/entry5k \
    --model openrouter:<model id> --split test --limit 50
```

Results go to `generated/entry5k/results/<model>/`. To read conversations:

```sh
python -m bioloopbench.entrytasks.transcripts --run generated/entry5k --model openrouter:<model id> --n 8
```

`evaluate` selects tasks by `--split` only. Evaluating on a custom date range currently needs a small change to filter by task IDs from the manifest.

### Pilot result

One small pilot: Gemini 3.8 Flash on 20 test tasks from an earlier 28-article run. Treat it as a smoke test, not a finding.

| Setting | Top-1 | Top-3 | Top-1 excluding symptom-solved (12 tasks) |
|---|---|---|---|
| Symptom only | 45% | 65% | 8% |
| Ask | 25% | 40% | 8% |
| Full protocol | 55% | 85% | 33% |

The ladder separates the floor from the ceiling on the harder tasks (8% to 33%). **Asking did not help**, and overall it scored below symptom only. Andre read the transcripts and saw nothing broken; his guess is that the question rounds make the model overcomplicate its answer. The model used nearly all six questions every time (5.7 on average). This is the main open question about the evaluation design. No evaluation has been run on the 5,161-task set yet.

## 5. How the tasks were generated

```
Europe PMC search ──► sample articles ──► parse ──► extract ──► audit ──► calibrate ──► accepted task
  (per topic)          (205 per topic)    (XML)   (generator)  (auditor)  (judge, solver)
```

1. **Corpus.** For each topic, search Europe PMC for open-access STAR Protocols articles. Take every article published on or after the cutoff, then fill up to 205 with earlier ones at random. Skip articles already taken by another topic.
2. **Parse.** Split each article into protocol text and troubleshooting entries (usually "Problem N" / "Potential solution" pairs).
3. **Extract.** A generator model turns each entry into a draft task, or sets it aside with a reason. Code then checks the draft: every quote must appear verbatim in the entry, step references must appear in the entry, and the opening may not copy more than 9 consecutive words from the article.
4. **Audit.** A second model reviews each draft against the article. It rejects tasks whose problem is not observable at the bench, whose opening gives away the fix, is unsupported by the article, or identifies the article. It drops individual fixes that are unsupported or not concrete.
5. **Calibrate.** Two checks per task. The judge must match a paraphrase of each reference fix to that fix and must not match a deliberately vague answer; otherwise the task is rejected. Then a solver answers from the opening alone three times; the result is recorded as `symptom_solved` but does not reject the task.

Nothing the model sees or is scored against is invented: the fixes are the authors', and the opening is a checked paraphrase.

Funnel for the full run:

| Stage | Count |
|---|---|
| Articles sampled | 1,504 |
| Articles that failed extraction (model refusal or failed checks) | 7 |
| Troubleshooting entries seen | 7,969 |
| Draft tasks passing the code checks | 5,586 |
| Rejected by the auditor | 419 |
| Rejected by the judge sanity check | 2 |
| Failed calibration (model refusal) | 4 |
| **Accepted** | **5,161** |

The entries not drafted (about 2,400) were set aside by the generator, mostly for having no concrete fix or no bench-observable problem. Exact counts per reason and topic are in `funnel.json`.

Models used for generation, all Google Gemini through the direct API:

| Role | Model |
|---|---|
| Generator, gatekeeper, solver | `gemini-3.8-flash` |
| Auditor, judge | `gemini-3.1-pro-preview` |

Estimated cost of the run was about $500 at list prices, roughly $0.10 per accepted task.

## 6. Known limits and open questions

- **Asking does not yet help** (section 4). Unresolved.
- **One model family.** Generator, auditor, judge and solver are all Gemini, so they may share blind spots. The judge sanity check rejected only 2 of 5,167 tasks, which suggests it is too easy in this configuration. The `symptom_solved` flag is also relative to Gemini Flash; a stronger model will solve more from the opening alone.
- **Nobody has reviewed the tasks at scale.** Andre read eight pilot transcripts. Quality rests on the automated checks.
- **Step accuracy is an exact match** on the step number, so sub-step labels may be scored as misses (50% in the pilot).
- **CRISPR is small** (226 tasks from 74 articles against about 200 articles for other topics). Not investigated; probably most of its search hits had already been taken by earlier topics.
- **The test split is small and uneven:** 588 tasks, 221 of them from immunostaining.
- **Manuscript received and accepted dates are missing** for all tasks.
- **Memorisation.** Articles are open access and probably in training data for anything published before a model's cutoff. The controls are the date split, the symptom-only floor, paraphrased openings that do not name the article, and no web access for evaluated models.

## 7. Practical notes

- **Gemini key.** The key used for generation ran on credits that expired on 19 November 2026. The `gemini:` backend refuses calls after that date (`GEMINI_KEY_STOP_AFTER` in `bioloopbench/taskgen/llm.py`). To run anything now, set model roles to `openrouter:<model id>` in the config, or update the constant after replacing the key. See `docs/api-keys.md`.
- **Evaluation needs a gatekeeper and a judge**, taken from the run's `config.json`. Both are currently `gemini:` models, so change them there before evaluating.
- **Tests.** `python -m unittest discover -s tests` runs 94 tests with scripted model replies; no network or key needed.
- **Older design.** `bioloopbench/taskgen/` is the earlier sibling-cause pipeline. It produced no usable tasks and is kept only because the entry-task code reuses its corpus, model client and checking modules. `docs/taskgen.md` describes it.

## 8. Where things are

| What | Where |
|---|---|
| Pipeline code | `bioloopbench/entrytasks/` |
| Stage-by-stage design notes | `docs/entrytasks.md` |
| Config of the 5,161-task run | `configs/entry5k.json` |
| Runner script used | `scripts/run_entry5k.sh` |
| Data | Hugging Face, private dataset `awhecmu/bioloopbench-entry5k` |
| Tests | `tests/test_entrytasks.py` |
