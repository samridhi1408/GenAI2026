"""Compile evaluation episodes into a Markdown file for a person to read.

  python -m bioloopbench.entrytasks.transcripts --run generated/pilot1 --model gemini:gemini-3.8-flash --n 8

For each chosen task the file shows the opening, the reference fixes, the full
ask-mode conversation with the gatekeeper, the agent's final answer with the
judge's matches, and the same agent's symptom-only and full-protocol answers
for comparison. Tasks where asking did worse than not asking come first.

The file contains reference answers: it is for reviewers, never for an
evaluated agent. Only counts are printed to the terminal.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from ..taskgen.store import read_json

KIND = {"answer": "answered", "not_recorded": "not recorded", "refuse": "refused"}


def _rank(score):
    r = score.get("first_match_rank")
    return f"first match at rank {r}" if r else "no match"


def _answer(lines, title, row):
    lines.append(f"### {title} — {_rank(row['score'])}")
    final = row["episode"].get("final")
    if not final:
        lines.append(f"_No answer (status: {row['episode'].get('status')})._\n")
        return
    matches = row["score"].get("matches") or []
    for i, fix in enumerate(final["fixes"]):
        m = matches[i] if i < len(matches) else "?"
        lines.append(f"{i + 1}. {fix}  \n   → judge: **{m}**")
    if final.get("step") not in (None, ""):
        lines.append(f"\nStep named: `{final['step']}` — step correct: {row['score'].get('step_correct')}")
    lines.append("")


def category(rows):
    """Order tasks so the most informative comparisons come first."""
    ask, sym = rows["ask"]["score"], rows.get("symptom_only", {}).get("score", {})
    a, s = ask.get("first_match_rank"), sym.get("first_match_rank")
    if s and (not a or a > s):
        return 0, "asking did worse than symptom only"
    if not a and not s:
        return 1, "missed in both"
    if a and (not s or a < s):
        return 2, "asking did better than symptom only"
    return 3, "same rank in both"


def compile_transcripts(run, model, n):
    run = Path(run)
    res = run / "results" / re.sub(r"[^A-Za-z0-9_.-]+", "_", model)
    by_mode = {}
    for mode in ("ask", "symptom_only", "full_protocol"):
        for row in read_json(res / f"{mode}.json", []):
            by_mode.setdefault(row["episode"]["task_id"], {})[mode] = row
    tasks = sorted((category(rows), tid) for tid, rows in by_mode.items() if "ask" in rows)
    # Take from each category in turn, worst-for-asking first, so the file is a mix.
    groups = {}
    for (order, label), tid in tasks:
        groups.setdefault((order, label), []).append(tid)
    chosen = []
    while len(chosen) < n and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(chosen) < n:
                chosen.append((key, groups[key].pop(0)))
            if key[0] == 0 and groups[key] and len(chosen) < n:  # two of these per round
                chosen.append((key, groups[key].pop(0)))
    chosen.sort()

    lines = [f"# Ask-mode transcripts: {model}", "",
             f"Run `{run}`. {len(chosen)} of {len(tasks)} tasks. **Contains reference answers — reviewers only.**", "",
             "| Task | Group | Ask | Symptom only | Full protocol | Questions |", "|---|---|---|---|---|---|"]
    for (_, label), tid in chosen:
        r = by_mode[tid]
        cell = lambda m: _rank(r[m]["score"]) if m in r else "—"
        lines.append(f"| {tid} | {label} | {cell('ask')} | {cell('symptom_only')} | {cell('full_protocol')} | "
                     f"{r['ask']['score'].get('questions_asked', 0)} |")
    for (_, label), tid in chosen:
        r = by_mode[tid]
        task = read_json(run / "private" / f"{tid}.json")
        lines += ["", "---", "", f"## {tid} — {label}", "",
                  f"Technique: {task.get('technique')} · symptom-solved flag: {task.get('symptom_solved')} · "
                  f"step references: {task.get('step_refs')}", "", "### Opening shown to the agent", "",
                  "> " + task["opening"].replace("\n", "\n> "), "", "### Reference fixes (hidden from the agent)", ""]
        for f in task["fixes"]:
            lines.append(f"- **{f['id']}**: {f['fix']}" + (f"  \n  Cause: {f['cause']}" if f.get("cause") else ""))
        lines += ["", "### Ask-mode conversation", ""]
        for i, rnd in enumerate(r["ask"]["episode"].get("rounds", []), 1):
            act = rnd["action"]
            if rnd.get("reasoning"):
                lines.append(f"_Agent reasoning (round {i}):_ {rnd['reasoning']}\n")
            if act["type"] == "ask":
                result = rnd.get("result", {})
                used = "" if result.get("consumed", True) else ", did not use up a question"
                lines += [f"**Q{i}.** {act['question']}", "",
                          f"**Lab ({KIND.get(result.get('kind'), result.get('kind'))}{used}):** {result.get('answer', '')}", ""]
        _answer(lines, "Final answer after asking", r["ask"])
        if "symptom_only" in r:
            _answer(lines, "Answer from the symptom alone", r["symptom_only"])
        if "full_protocol" in r:
            _answer(lines, "Answer with the full protocol", r["full_protocol"])
    out = res / "ask_transcripts.md"
    out.write_text("\n".join(lines) + "\n")
    counts = {}
    for (_, label), _tid in chosen:
        counts[label] = counts.get(label, 0) + 1
    return out, counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=8, help="number of tasks to include")
    args = ap.parse_args(argv)
    out, counts = compile_transcripts(args.run, args.model, args.n)
    print(f"wrote {out} ({out.stat().st_size} bytes): {counts}")


if __name__ == "__main__":
    main()
