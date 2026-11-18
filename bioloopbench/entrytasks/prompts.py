"""Prompt templates for entry tasks. Each says what its output is used for."""

PROJECT = """\
You are helping build BioLoopBench, a benchmark that tests whether AI agents can troubleshoot failed \
biology experiments. Each task comes from one entry in the Troubleshooting section of a published STAR \
Protocols article. The agent is told what went wrong, may ask questions about how the experiment was \
carried out, and must then propose fixes. A proposal is correct when it matches a fix the authors give for \
that entry.

Faithfulness to the article matters more than yield: a task that says something the article does not \
support, or that gives its own answer away, damages the benchmark."""

SKIP_CODES = ("not_observable", "no_concrete_fix", "other")

EXTRACT = PROJECT + """

Your job now is extraction. Below is one article and its numbered troubleshooting entries. Decide for \
every entry whether it can become a task. An entry qualifies when both of these hold:
- It describes a failure a scientist would observe at the bench or in the data (not a design decision, a \
software installation error, or general advice).
- The article gives at least one concrete fix: a specific change to a material, reagent, parameter, \
instrument setting or handling step that someone could carry out. "Optimise the conditions", "repeat the \
experiment" and "contact the authors" are not concrete fixes.
The entry does not need to state why the failure happened, and it does not need to be about any particular \
technique.

For each qualifying entry produce:
- entry_index: the entry's number.
- technique: the technique involved, in a few words.
- context: two to four sentences, in your own words, saying what the experiment is and what stage had been \
reached when the problem appeared, so that a scientist who has not read the article understands the \
situation. Every fact must come from the article. Do not name the article, its authors, or coined method \
names that would identify it. Do not mention any fix, and do not give details that point to one.
- problem: one or two sentences, in your own words, describing only what was observed. It must not hint at \
the reason or the remedy.
- problem_quote: the words from the entry that state the problem, copied exactly.
- step_refs: the protocol step numbers the entry says the problem relates to, as strings exactly as \
written (for example ["14"] or ["3b"]); an empty list if the entry names none.
- fixes: one item per distinct concrete fix the entry gives, each with
    fix: the corrective action in your own words, as one specific instruction;
    fix_quote: the words in the entry that give it, copied exactly;
    cause: the reason for the failure that this fix addresses, in your own words, if the entry states or \
clearly implies one; otherwise null.
  Split alternatives ("do A, or alternatively B") into separate fixes. Leave out vague advice.

Every entry must appear exactly once, either in "tasks" or in "skipped". For a skipped entry give a \
reason_code: "not_observable" (not a bench or data failure), "no_concrete_fix", or "other".

Quotes are checked by exact string match against the entry text, so copy them character for character and \
keep each to a single contiguous span. The context and problem are checked the other way: they must not \
reuse long runs of the article's wording.

Reply with JSON only:
{{"tasks": [{{"entry_index": 1, "technique": "...", "context": "...", "problem": "...", \
"problem_quote": "...", "step_refs": ["14"], "fixes": [{{"fix": "...", "fix_quote": "...", \
"cause": "..."}}]}}],
 "skipped": [{{"entry_index": 2, "reason_code": "no_concrete_fix", "reason": "..."}}]}}

ARTICLE TITLE: {title}

PROTOCOL TEXT:
{protocol}

TROUBLESHOOTING ENTRIES:
{entries}"""

AUDIT_EXTRACT = PROJECT + """

Your job now is to audit draft tasks that another model extracted from one article. You are the check on \
faithfulness, so read each draft against the source text. Flag real problems; do not reject a draft for \
matters of style.

For each draft task answer:
- bench_observable: is the problem something seen at the bench or in the data?
- problem_leaks_fix: would a reader of the context and problem alone be told, or all but told, one of the \
listed fixes? Naming the stage or the material involved is not a leak; stating what was done wrong is.
- context_unsupported: list any factual claims in the context that the article text below does not support \
(empty list if none).
- identifies_source: does the context name the article, its authors, or a coined method name that would \
identify it?
- for each fix, by position:
    supported: does the troubleshooting entry give this fix?
    concrete: is it a specific action someone could carry out, not general advice?

Reply with JSON only:
{{"tasks": [{{"entry_index": 1, "bench_observable": true, "problem_leaks_fix": false, \
"context_unsupported": [], "identifies_source": false, \
"fixes": [{{"supported": true, "concrete": true, "note": ""}}]}}]}}

DRAFT TASKS:
{drafts}

PROTOCOL TEXT:
{protocol}

TROUBLESHOOTING ENTRIES:
{entries}"""

GATEKEEPER = """\
You hold the lab notebook for an experiment that went wrong. A scientist who was not present is trying to \
work out how to fix it and will ask you how the experiment was set up and carried out. The notebook is the \
protocol below, which was followed as written.

WHAT THE SCIENTIST WAS TOLD:
{opening}

THE PROTOCOL AS FOLLOWED:
{protocol}

ANSWERS ALREADY GIVEN (stay consistent with them):
{established}

Rules:
- Answer one specific question per request about what was used or done: a material, an amount, a \
concentration, a time, a temperature, an instrument setting, the order of steps, how something was \
handled. Answer in one to three sentences with the facts the protocol records.
- Report what was done, not advice. Do not pass on the protocol's warnings, tips, notes on what can go \
wrong, or expected outcomes, and do not say what should have been done differently.
- If the protocol does not record what is asked, say so: kind "not_recorded".
- Refuse (kind "refuse") requests for the cause, a fix, a hint or an opinion; requests for the whole \
protocol, a whole section, or a summary of everything; and requests that bundle several unrelated \
questions. Say briefly that one specific question about the procedure is needed.

Reply with JSON only:
{{"kind": "answer" or "not_recorded" or "refuse", "answer": "..."}}"""

MATCH = """\
You are scoring a troubleshooting benchmark. An experiment had the problem described below. The authors of \
the protocol report the reference fixes listed. Decide, for each proposed fix, whether it is the same \
corrective action as one of the reference fixes.

THE SITUATION:
{opening}

REFERENCE FIXES:
{fixes}

PROPOSED FIXES:
{proposals}

Rules:
- A proposal matches a reference fix when carrying it out would mean making essentially the same change: \
the same step, material or parameter, changed in the same direction. Different wording counts as a match, \
and so does a different exact amount in the same direction.
- A proposal does not match if it targets something else, changes the same thing in the opposite \
direction, or is general advice that does not single out the reference fix.
- A proposal that lists several unrelated actions in the hope that one is right matches nothing.
- Judge each proposal on its own. Two proposals may match the same reference fix.

Reply with JSON only:
{{"matches": [{{"proposal": "P1", "matches": "F2" or "none", "rationale": "one sentence"}}]}}"""

PARAPHRASE = """\
Below are fixes for a failed experiment, one per line. Rewrite each in clearly different words, as another \
scientist might phrase the same instruction, keeping the meaning and the level of specificity. Then write \
one further item, "vague": a piece of general advice about the same area of the experiment that does not \
single out any of the listed fixes.

{fixes}

Reply with JSON only: {{"paraphrases": {{"F1": "...", "F2": "..."}}, "vague": "..."}}"""

AGENT_SYSTEM = """\
You are an experienced bench scientist helping a colleague whose experiment went wrong. Your job is to \
propose how to fix it.

{how}

At every turn reply with JSON only:
{{"reasoning": "two or three sentences on what you think is going on and why",
 "action": {action}}}

When you are ready to commit, or have no questions left, use this action instead:
 "action": {{"type": "answer", "fixes": ["most likely fix", "second", "third"]{step_field}}}

Give one to three fixes, most likely first. Each fix must be one specific corrective action: what to \
change, and how. A fix that lists several alternatives, or that is general advice such as "optimise the \
conditions", is scored as wrong.{step_rule}"""

AGENT_HOW = {
    "ask": "You were not present, so you may ask the lab how the experiment was set up and carried out, "
           "one specific question at a time. The lab answers from its notebook; it will not offer opinions, "
           "causes or fixes. You have a limited number of questions, so ask the ones whose answers would "
           "most change what you recommend.",
    "full_protocol": "You are given the full protocol that was followed. No questions can be asked.",
    "symptom_only": "You have only your colleague's report. No questions can be asked.",
}
AGENT_ASK_ACTION = '{"type": "ask", "question": "..."}'
AGENT_NO_ASK_ACTION = '{"type": "answer", "fixes": ["..."]}'
AGENT_STEP_FIELD = ', "step": "the number of the protocol step where the problem arises, as written in the protocol, or null"'
AGENT_STEP_RULE = "\n\nFor \"step\", give the single step number you think the problem relates to."
