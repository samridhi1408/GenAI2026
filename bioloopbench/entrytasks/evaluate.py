"""Run a model over accepted entry tasks and aggregate its scores.

  python -m bioloopbench.entrytasks.evaluate --run generated/entry1 --model openrouter:<id> \\
      --modes ask,full_protocol,symptom_only --split test

Episodes and scores are written to <run>/results/<model>/<mode>.json, and a
summary to <run>/results/<model>/summary.json. The evaluated model is given
only the public view of a task (and, in full_protocol mode, the protocol).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from ..taskgen import llm
from ..taskgen.store import pmap, read_json, write_json
from .agent import MODES, run_episode
from .judge import score_episode
from .pipeline import protocol_path


def summarise(rows):
    """Aggregate per-episode scores. Rates are over all tasks attempted, so an
    episode that errored or gave no answer counts as a miss."""
    n = len(rows)
    if not n:
        return {"n": 0}
    stepped = [r["step_correct"] for r in rows if r["step_correct"] is not None]
    hard = [r for r in rows if not r["symptom_solved"]]
    return {
        "n": n,
        "completed": sum(r["status"] == "completed" for r in rows),
        "top1": sum(r["top1"] for r in rows) / n,
        "top3": sum(r["top3"] for r in rows) / n,
        "top1_excluding_symptom_solved": sum(r["top1"] for r in hard) / len(hard) if hard else None,
        "n_excluding_symptom_solved": len(hard),
        "mean_questions_asked": sum(r["questions_asked"] for r in rows) / n,
        "step_accuracy": sum(stepped) / len(stepped) if stepped else None,
        "n_step_scored": len(stepped),
    }


def evaluate(run, model, modes, split, cfg, limit=None):
    run = Path(run)
    tasks = [read_json(run / "private" / p.name) for p in sorted((run / "public").glob("*.json"))]
    tasks = [t for t in tasks if t["status"] == "accepted" and (split == "all" or t["split"] == split)][:limit]
    out_dir = run / "results" / re.sub(r"[^A-Za-z0-9_.-]+", "_", model)
    summary = {"model": model, "split": split, "modes": {}}
    for mode in modes:
        def work(t, mode=mode):
            protocol = protocol_path(run, t["pmcid"]).read_text()
            public = {k: t[k] for k in ("task_id", "split", "opening", "budget")}
            ep = run_episode(public, protocol, model, cfg["models"]["gatekeeper"], mode=mode,
                             facts_dir=run / "facts" if mode == "ask" else None)
            return {"episode": ep, "score": score_episode(t, ep, cfg["models"]["judge"])}
        rows, records = [], []
        for t, res in pmap(work, tasks, cfg["workers"]):
            if isinstance(res, Exception):
                res = {"episode": {"task_id": t["task_id"], "status": "error", "error": type(res).__name__},
                       "score": {"top1": False, "top3": False, "questions_asked": 0, "step_correct": None}}
            records.append(res)
            rows.append({**res["score"], "status": res["episode"]["status"], "symptom_solved": t.get("symptom_solved", False)})
        write_json(out_dir / f"{mode}.json", records)
        summary["modes"][mode] = summarise(rows)
    write_json(out_dir / "summary.json", summary)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory produced by the pipeline")
    ap.add_argument("--model", required=True, help="provider:model_id of the agent to evaluate")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--split", default="test", choices=("test", "development", "all"))
    ap.add_argument("--limit", type=int)
    args = ap.parse_args(argv)
    modes = args.modes.split(",")
    if set(modes) - set(MODES):
        ap.error(f"modes must be among {MODES}")
    cfg = read_json(Path(args.run) / "config.json")
    llm.CACHE_DIR = Path(args.run) / "llm_cache"
    llm.MAX_COST_USD = cfg["max_cost_usd"]
    summary = evaluate(args.run, args.model, modes, args.split, cfg, args.limit)
    print(json.dumps(summary, indent=2))
    print(f"spend this invocation: ${llm.total_cost():.2f}", file=sys.stderr)


if __name__ == "__main__":
    main()
