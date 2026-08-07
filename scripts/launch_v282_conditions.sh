#!/usr/bin/env bash
# v2.8.2 S1 — launch the four conditions (CTRL/M1/M2/M3), staggered + fully detached, expandable_segments.
# CTRL = all infeasibility machinery OFF (exp_config defaults). M1 = empty_mode prox (γ=s̄=2.200).
# M2 = M1 + w_infeas 0.001969 (units-fixed, s1_premeasure.md §6) + δ 1.124. M3 = M2 + w_du 0.000318.
set -u
cd /home/junhyeok/MIT/jt-pncbf
export PYTHONPATH=/home/junhyeok/MIT/jt-pncbf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True     # ~5x VRAM cut (10.3->2.2 GB/run); mandatory for concurrency
PY=/home/junhyeok/miniconda3/envs/pncbf/bin/python
VINIT="data/runs/v2.7.6/set__20260725-043415__seed42/v2.7.6__oc__20260725-043415__seed42/checkpoints/best.pt"
LOGD=data/runs/v2.8.2/launch_logs; mkdir -p "$LOGD"
COMMON="--seed 42 --steps 30000 --value-init $VINIT --collector continuing --stage full"

launch () {                                                 # $1=label  $2...=condition-specific flags
  local label="$1"; shift
  setsid nice -n 10 $PY scripts/v276_stage2_jt_run.py $COMMON "$@" > "$LOGD/$label.log" 2>&1 < /dev/null &
  echo "$(date +%H:%M:%S) launched $label pid $!" >> "$LOGD/launch.log"
}

launch CTRL
sleep 15
launch M1 --empty-mode prox --empty-prox-temp 2.200
sleep 15
launch M2 --empty-mode prox --empty-prox-temp 2.200 --w-infeas 0.001969 --infeas-delta 1.124
sleep 15
launch M3 --empty-mode prox --empty-prox-temp 2.200 --w-infeas 0.001969 --infeas-delta 1.124 --w-du 0.000318
echo "$(date +%H:%M:%S) all four launched" >> "$LOGD/launch.log"
