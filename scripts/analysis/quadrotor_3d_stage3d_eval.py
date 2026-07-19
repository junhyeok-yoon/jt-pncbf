"""v2.7.2 Stage-3D — k=5 empty-branch fallback SCALING MEASUREMENT (eval-only; NOT an adoption trial).
Runs the canonical full-pool (n2000) evaluate() under filter.empty_fallback mode=none (M6 baseline) and
mode=kstep k=5, with scene-bootstrap CIs for cps and collision. Writes per-mode episode CSVs + an aggregate
JSON to scratchpad. Deployed 3D default REMAINS mode=none regardless of outcome."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path

import torch

from src.eval.bootstrap import within_seed_ci
from src.eval.build_pools import DEFAULT_OUTPUT_DIR
from src.eval.evaluate import evaluate, EVAL_EPISODE_COLUMNS
from src.frameworks.jt_pncbf.train import load_framework_from_checkpoint

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--outdir", required=True)
ap.add_argument("--n", type=int, default=2000)
a = ap.parse_args()

POOL = DEFAULT_OUTPUT_DIR / "eval_full_quadrotor-3d_n2000_seed23456.pkl"
outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)


def _run(mode: str):
    filt = copy.deepcopy(ck["config"]["filter"])
    if mode == "none":
        filt.pop("empty_fallback", None)                     # absent == mode=none (f1 parity) -> reproduces M6
    else:
        filt["empty_fallback"] = {"mode": "kstep", "k": 5}
    fw, cfg, _ = load_framework_from_checkpoint(Path(a.ckpt), config_overrides={"filter": filt})
    t0 = time.perf_counter()
    res = evaluate(fw, POOL, cfg, mode="final", step=int(ck["step"]),
                   ckpt_name=Path(a.ckpt).name, max_scenes=a.n, include_lqr_baseline=False)
    wall = time.perf_counter() - t0
    rows = res.episode_rows
    boot = ck["config"]["eval"]["bootstrap"]
    ci = within_seed_ci(rows, n_resample=int(boot["n_resample"]), seed=int(boot["seed"]))
    # per-mode episode CSV (for S3 attribution)
    with (outdir / f"episodes_{mode}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EVAL_EPISODE_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in EVAL_EPISODE_COLUMNS})
    agg = {
        "mode": mode, "n": len(rows), "wall_s": round(wall, 1),
        "cps": ci["mean"]["cps"], "cps_ci": [ci["ci"]["cps"]["lo"], ci["ci"]["cps"]["hi"]],
        "collision": ci["mean"]["collision"],
        "collision_ci": [ci["ci"]["collision"]["lo"], ci["ci"]["collision"]["hi"]],
        "reach": ci["mean"]["reach"], "oob": ci["mean"]["oob"], "stuck": ci["mean"]["stuck"],
        "timeout": ci["mean"]["timeout"], "infeasibility": ci["mean"]["infeasibility"],
        "empty_step_frac_mean": round(sum(float(r["empty_step_frac"]) for r in rows) / len(rows), 6),
    }
    return agg


out = {"none": _run("none"), "kstep_k5": _run("kstep")}
out["delta"] = {
    "d_cps": round(out["kstep_k5"]["cps"] - out["none"]["cps"], 6),
    "d_collision": round(out["kstep_k5"]["collision"] - out["none"]["collision"], 6),
    "wall_ratio": round(out["kstep_k5"]["wall_s"] / max(out["none"]["wall_s"], 1e-9), 3),
}
(outdir / "stage3d_aggregate.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
