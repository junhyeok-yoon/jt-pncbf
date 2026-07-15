#!/usr/bin/env bash
# v2.5.2 Stage 2 — 3 unicycle exact_m0 seeds, 2-concurrent: {42,99} together, then 12345.
# Provenance: committed at v2.5.2 close from the scratchpad orchestrator. Wave 2 (seed 12345) was NOT run
# in the actual v2.5.2 execution (J-arc amendment §1 reordered J-arc ahead of wave 2 and halted after); the
# orchestrator main was killed before the `run_seed 12345` line. Retained here verbatim; `SP`/paths are the
# original session scratchpad, kept as historical provenance.
set -u
cd /home/junhyeok/MIT/jt-pncbf
PY=/home/junhyeok/miniconda3/envs/pncbf/bin/python
SP=/tmp/claude-1000/-home-junhyeok-MIT-jt-pncbf/31d93785-ac11-4206-bf50-a4c3de145dff/scratchpad
export PYTHONPATH=/home/junhyeok/MIT/jt-pncbf

run_seed () {
  local seed=$1
  $PY "$SP/s2_run.py" --seed "$seed" --stage full --steps 50000 --tag "full$seed" \
      > "$SP/s2_full${seed}.log" 2>&1
  echo "seed $seed exit=$? at $(date -u +%FT%TZ)" >> "$SP/s2_orchestrate.log"
}

echo "ORCH START $(date -u +%FT%TZ)" > "$SP/s2_orchestrate.log"
# Wave 1: seeds 42 and 99 concurrently.
run_seed 42 &
P42=$!
run_seed 99 &
P99=$!
echo "launched 42(pid $P42) 99(pid $P99)" >> "$SP/s2_orchestrate.log"
wait $P42
wait $P99
echo "wave1 done $(date -u +%FT%TZ)" >> "$SP/s2_orchestrate.log"
# Wave 2: seed 12345.
run_seed 12345
echo "ORCH DONE $(date -u +%FT%TZ)" >> "$SP/s2_orchestrate.log"
