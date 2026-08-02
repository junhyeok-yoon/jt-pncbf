"""v2.8.0 S1 M3 — end-to-end throughput of the two projection realizations. Runs 200 JT training steps on
quadrotor_3d (same seed, same config, warm-started from the v2.7.6 OC value) under each of
filter.projection = {enumerate, dual_solve}, and reports wall-clock per step. Decision trigger: if the dual
solve is more than 25% slower end-to-end, stop before M4 (Researcher fallback, changes.md §6). Bench run dir
is discarded."""
from __future__ import annotations
import json, shutil, time
from pathlib import Path
import torch
import src.frameworks.jt_pncbf.train as T
from src.frameworks.jt_pncbf.train import run_training

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
VALUE_INIT = REPO / "data/runs/v2.7.6/set__20260725-043415__seed42/v2.7.6__oc__20260725-043415__seed42/checkpoints/best.pt"
OUTROOT = REPO / "data/runs/_s1_bench"
OUT = REPO / "data/runs/v2.8.0/s1_projection"; OUT.mkdir(parents=True, exist_ok=True)
_orig = T.load_effective_config


def _patch(proj):
    def _p():
        c = _orig()
        c["run"]["system"] = "quadrotor_3d"
        c["collection"]["collector"] = "continuing"
        c["collection"]["inject_frac"] = 0.0
        c["loss"]["policy"]["sat_excess_threshold"] = [4.905, 4.905, 4.905, 4.905]
        c["filter"]["projection"] = proj
        return c
    return _p


res = {"device": "cuda" if torch.cuda.is_available() else "cpu", "n_steps": 200}
for proj in ("enumerate", "dual_solve"):
    shutil.rmtree(OUTROOT, ignore_errors=True)
    T.load_effective_config = _patch(proj)
    t0 = time.time()
    run_training(stage="full", output_root=OUTROOT, seed=42, system="quadrotor_3d",
                 n_steps_override=200, schedule_n_steps_override=200, value_init_ckpt=VALUE_INIT)
    dt = time.time() - t0
    res[proj] = {"total_s": round(dt, 2), "per_step_ms": round(dt / 200 * 1000, 2)}
    print(f"{proj}: total {dt:.1f}s  per_step {dt/200*1000:.1f}ms", flush=True)
T.load_effective_config = _orig
ratio = res["dual_solve"]["per_step_ms"] / res["enumerate"]["per_step_ms"]
res["dual_over_enum_ratio"] = round(ratio, 4)
res["dual_more_than_25pct_slower"] = bool(ratio > 1.25)
res["note"] = "end-to-end wall/step incl. identical startup+collection+final smoke eval; ratio is the decision metric"
(OUT / "throughput.json").write_text(json.dumps(res, indent=2) + "\n")
shutil.rmtree(OUTROOT, ignore_errors=True)
print("THROUGHPUT:", json.dumps(res))
