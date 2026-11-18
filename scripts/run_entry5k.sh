#!/bin/bash
# Generate ~5,000 entry tasks with the Gemini key. Safe to re-run: model replies
# are cached, so it resumes. STOPS for good once the UTC date passes STOP_AFTER
# (the Gemini credits expire; see docs/api-keys.md).
#   scripts/run_entry5k.sh            run in the foreground
#   nohup scripts/run_entry5k.sh &    keep running after the terminal closes
cd "$(dirname "$0")/.." || exit 1
RUN=${RUN:-generated/entry5k}
STOP_AFTER=2026-11-19
PY=${PY:-../GenAI2026/.venv/bin/python}
eval "$(grep -E '^export GEMINI_API_KEY=' ~/.bash_profile)"; export GEMINI_API_KEY
unset GEMINI_KEY_STOP_AFTER   # never override the guard in llm.py from here
mkdir -p "$RUN"
say() { echo "$(date '+%F %T') $*" >> "$RUN/runner.log"; }
expired() { [[ "$(date -u +%F)" > "$STOP_AFTER" ]]; }
echo "{\"pid\": $$, \"started\": \"$(date '+%F %T')\", \"stop_after_utc\": \"$STOP_AFTER\"}" > "$RUN/runner.json"
for pass in 1 2 3 4; do
  if expired; then say "STOPPED: past $STOP_AFTER UTC, the Gemini key must not be used"; exit 0; fi
  stages="corpus,extract,calibrate"; [[ $pass -gt 1 ]] && stages="extract,calibrate"   # later passes retry failures from cache
  say "pass $pass starting ($stages)"
  caffeinate -i "$PY" -m bioloopbench.entrytasks.pipeline --config configs/entry5k.json --out "$RUN" --stages "$stages" >> "$RUN/pipeline.log" 2>&1
  say "pass $pass finished (exit $?)"
done
say "DONE: all passes finished"
