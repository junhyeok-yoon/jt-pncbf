#!/usr/bin/env bash
# v2.8.2 within-version iteration — alpha_unsafe training axis. Three cells {30,10,3} at alpha_safe=2.0, the CTRL
# recipe with filter.alpha_unsafe as the ONLY difference (CTRL is the alpha_unsafe=100 fourth point). Independent
# processes, own run ids, staggered, no orchestrator coupling — any one may be stopped/restarted alone.
set -u
cd /home/junhyeok/MIT/jt-pncbf
export PYTHONPATH=/home/junhyeok/MIT/jt-pncbf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/junhyeok/miniconda3/envs/pncbf/bin/python
VINIT="data/runs/v2.7.6/set__20260725-043415__seed42/v2.7.6__oc__20260725-043415__seed42/checkpoints/best.pt"
LOGD=data/runs/v2.8.2/launch_logs; mkdir -p "$LOGD"
COMMON="--seed 42 --steps 30000 --value-init $VINIT --collector continuing --stage full"

launch () {                                                # $1=label  $2=alpha_unsafe
  local label="$1" au="$2"
  setsid nice -n 10 $PY scripts/v276_stage2_jt_run.py $COMMON --alpha-unsafe "$au" > "$LOGD/$label.log" 2>&1 < /dev/null &
  echo "$(date +%H:%M:%S) launched $label (alpha_unsafe=$au) pid $!" >> "$LOGD/launch_au.log"
}

launch AU30 30
sleep 15
launch AU10 10
sleep 15
launch AU3 3
echo "$(date +%H:%M:%S) all three alpha_unsafe cells launched" >> "$LOGD/launch_au.log"
