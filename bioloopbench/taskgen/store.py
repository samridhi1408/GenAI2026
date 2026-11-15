"""Run directory layout, configuration and small IO helpers.

<run>/
  corpus/xml/                 cached article XML (shared across topics)
  llm_cache/                  cached model replies
  topics/<topic>/articles.json   selected articles and their split
  topics/<topic>/drafts.json     extracted cases with audit results
  topics/<topic>/groups.json     symptom groups
  private/<task_id>.json      full task record: causes, evidence, provenance, audits, calibration
  public/<task_id>.json       what an evaluated agent may see
  funnel.json                 counts of what was kept and rejected, and why
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .llm import BudgetExceeded

DEFAULTS = {
    "models": {"generator": None, "auditor": None, "gatekeeper": None, "judge": None, "solver": None},
    "test_cutoff_date": "2026-07-01",   # articles first published on or after this date form the test split
    "dev_articles_per_topic": 24,
    "test_articles_per_topic": 16,
    "seed": 0,
    "max_protocol_chars": 120000,
    "max_shared_run_words": 9,          # longest wording an agent-facing text may share with its source
    "min_causes": 3,
    "max_causes": 5,
    "allow_model_proposed_causes": 0,   # extra textbook causes the assembler may add (0 = source-grounded only)
    "max_tasks_per_group_per_split": 2,
    "max_tasks_per_article": 2,
    "no_experiment_samples": 3,
    "oracle_samples": 1,
    "workers": 6,
    "max_cost_usd": None,               # stop the run when reported spend reaches this (OpenRouter reports cost)
}


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    out = {**DEFAULTS, **cfg}
    out["models"] = {**DEFAULTS["models"], **cfg.get("models", {})}
    missing = [r for r, m in out["models"].items() if not m]
    if missing:
        raise ValueError(f"config is missing models for roles: {', '.join(missing)}")
    return out


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    tmp.replace(path)


class Funnel:
    """Counts outcomes per stage so the yield of every gate is reported."""

    def __init__(self, path):
        self.path = Path(path)
        self.data = read_json(path, {})
        self._lock = threading.Lock()

    def reset(self, topic, stage):
        with self._lock:
            self.data.setdefault(topic, {})[stage] = {}

    def add(self, topic, stage, outcome, n=1):
        with self._lock:
            d = self.data.setdefault(topic, {}).setdefault(stage, {})
            d[outcome] = d.get(outcome, 0) + n

    def save(self):
        with self._lock:
            write_json(self.path, self.data)


def pmap(fn, items, workers):
    """Map in threads, returning (item, result_or_exception) pairs in order."""
    def run(x):
        try:
            return fn(x)
        except BudgetExceeded:
            raise
        except Exception as e:  # one failed article or task must not sink the run
            return e
    with ThreadPoolExecutor(max(1, workers)) as ex:
        return list(zip(items, ex.map(run, items)))
