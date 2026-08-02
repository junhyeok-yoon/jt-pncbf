"""v2.7.6 close step 3 (amendment 1) — re-score the standing quadrotor_3d comparator (v2.7.2 M6 JT, sha8
4baaf031) from data/secured_data on the current canonical pool eval_full_quadrotor-3d-d2r_n2000_seed23456
(0ef3751b), under BOTH scorings and BOTH fallback settings the new rows use, on GPU (matching canonical_eval).
04_eval s1 History note prescribes re-scoring standing comparators across the banded predicate change. The
v2.7.2 checkpoint was trained under a different IC distribution (6-DOF perturbed tilt, random obstacles;
pre-full-SO(3), pre-d2r), so the comparison remains across IC distributions regardless of the numbers."""
from __future__ import annotations

import copy, json, time
from pathlib import Path

import torch

from src.eval.evaluate import evaluate
from src.eval.bootstrap import within_seed_ci
from src.eval.run_full import _load_framework as load_fw
from scripts.analysis.v276_m7_fallback_trial import _split, _filter_for

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
POOL = REPO / "data/eval_pools/eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl"   # canonical 0ef3751b (shared home)
CK = REPO / "data/secured_data/v2.7.2/seed42/checkpoints/best.pt"                # v2.7.2 secured, sha8 4baaf031
OUT = REPO / "data/runs/v2.7.6/stage2_eval/rescore_v272_canonical.json"
COMPS = ("cps", "reach", "collision", "oob", "stuck", "timeout", "infeasibility")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run(mode, scoring):
    ck = torch.load(CK, map_location="cpu", weights_only=False)
    boot = ck["config"]["eval"]["bootstrap"]; n_boot = int(boot["n_resample"]); bseed = int(boot["seed"])
    bc = 0.0 if scoring == "legacy" else 4.0
    over = {"env": {"dt": 0.05, "stuck_window_steps": 60, "stuck_radius": 0.10, "band_collision_limit": bc},
            "eval": {"max_steps": 200, "dt_ctrl": 0.05}, "filter": _filter_for(ck, mode)}
    fw, cfg, _ = load_fw(CK, config_overrides=over)
    for net in ("value_net", "policy_net"):
        m = getattr(fw, net, None)
        if m is not None:
            m.to(DEV)
    t0 = time.perf_counter()
    res = evaluate(fw, POOL, cfg, mode="final", step=int(ck.get("step", 0)),
                   ckpt_name="v272_best.pt", max_scenes=None, include_lqr_baseline=False)
    wall = round(time.perf_counter() - t0, 1)
    rows = list(res.episode_rows)
    ci = within_seed_ci(rows, n_resample=n_boot, seed=bseed)
    agg = {"mode": mode, "scoring": scoring, "n": len(rows), "wall_s": wall, "device": str(DEV)}
    for m in COMPS:
        agg[m] = round(float(ci["mean"][m]), 6); agg[m + "_ci"] = [round(float(ci["ci"][m]["lo"]), 6), round(float(ci["ci"][m]["hi"]), 6)]
    agg["split"] = _split(res, n_boot, bseed)
    s = agg["split"]
    print(f"[v272/{mode}/{scoring}] cps {agg['cps']:+.5f} {agg['cps_ci']} reach {agg['reach']:.4f} coll {agg['collision']:.4f} "
          f"band {s['band_total']['count']} (fl {s['band_floor']['count']}/cl {s['band_ceiling']['count']}) wall {wall}s", flush=True)
    return agg


if __name__ == "__main__":
    cells = [run(mode, sc) for sc in ("legacy", "banded") for mode in ("none", "kstep")]
    OUT.write_text(json.dumps({"comparator": "v2.7.2 M6 JT", "sha8": "4baaf031", "pool": POOL.stem,
        "pool_sha": "0ef3751b", "ck_source": "data/secured_data/v2.7.2/seed42/checkpoints/best.pt",
        "training_IC": "6-DOF perturbed tilt, obstacle_distribution=random (pre-full-SO(3), pre-d2r)",
        "cells": cells}, indent=2) + "\n")
    print("rescore done ->", OUT, flush=True)
