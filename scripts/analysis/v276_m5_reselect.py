"""v2.7.6 M5 — re-select the JT headline checkpoint on the band-feasible FULL pool (04_eval s6.2): banded cps
is the selection metric, evaluated over cadence checkpoints (NOT the in-loop best.pt). Reports banded cps +
scene-bootstrap 95% CI per candidate and names the winner (max mean cps) with its SHA-256 prefix. Candidates
default to the competitive tail (in-loop cps stabilized > ~0.65); pass explicit steps to override. Eval-only."""
from __future__ import annotations

import copy, hashlib, json, sys
from pathlib import Path

import numpy as np
import torch

from src.eval.evaluate import evaluate
from src.eval.bootstrap import within_seed_ci
from src.eval.run_full import _load_framework as load_framework_from_checkpoint

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
POOLS = REPO / "data/runs/v2.7.6/pools"
STEM = "eval_bandfeasible_quadrotor-3d-d2r_n2000_seed42"
CKDIR = REPO / "data/runs/v2.7.6/set__20260725-043415__seed42/v2.7.6__jt__20260725-052127__seed42/checkpoints"


def _sha8(p: Path) -> str:
    h = hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()[:8]


def score(ckpt: Path):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    boot = ck["config"]["eval"]["bootstrap"]; n_boot = int(boot["n_resample"]); bseed = int(boot["seed"])
    over = {"env": {"dt": 0.05, "stuck_window_steps": 60, "stuck_radius": 0.10, "band_collision_limit": 4.0},
            "eval": {"max_steps": 200, "dt_ctrl": 0.05}, "filter": copy.deepcopy(ck["config"]["filter"])}
    over["filter"]["empty_fallback"] = {"mode": "none", "k": int(over["filter"].get("empty_fallback", {}).get("k", 10))}
    fw, cfg, _ = load_framework_from_checkpoint(ckpt, config_overrides=over)
    res = evaluate(fw, POOLS / f"{STEM}.pkl", cfg, mode="final", step=int(ck["step"]),
                   ckpt_name=ckpt.name, max_scenes=None, include_lqr_baseline=False)
    ci = within_seed_ci(list(res.episode_rows), n_resample=n_boot, seed=bseed)
    return {"step": int(ck["step"]), "sha8": _sha8(ckpt),
            "cps": round(float(ci["mean"]["cps"]), 6),
            "cps_ci95": [round(float(ci["ci"]["cps"]["lo"]), 6), round(float(ci["ci"]["cps"]["hi"]), 6)],
            "reach": round(float(ci["mean"]["reach"]), 5), "collision": round(float(ci["mean"]["collision"]), 5),
            "infeasibility": round(float(ci["mean"]["infeasibility"]), 5)}


if __name__ == "__main__":
    OUT = REPO / "data/runs/v2.7.6/stage2_eval/m5_reselect.json"
    if len(sys.argv) > 1:
        steps = [int(s) for s in sys.argv[1].split(",")]
    else:
        steps = list(range(21000, 50001, 1500))            # competitive tail
    results = []
    for s in steps:
        ck = CKDIR / f"step_{s:06d}.pt"
        if not ck.exists():
            print(f"  step {s}: MISSING", flush=True); continue
        r = score(ck); results.append(r)
        print(f"  step {r['step']:>5} sha {r['sha8']} banded_cps {r['cps']:+.6f} {r['cps_ci95']} "
              f"reach {r['reach']:.4f} coll {r['collision']:.4f} infeas {r['infeasibility']:.4f}", flush=True)
    best = max(results, key=lambda r: r["cps"]) if results else None
    rec = {"pool": STEM, "selection_metric": "banded_cps", "candidates": results, "winner": best}
    OUT.write_text(json.dumps(rec, indent=2) + "\n")
    if best:
        print(f"\nWINNER: step {best['step']} sha {best['sha8']} banded_cps {best['cps']:+.6f} {best['cps_ci95']}", flush=True)
