"""v2.7.6 close-out — canonical-pool numbers. Evaluate on eval_full_quadrotor-3d-d2r_n2000_seed23456
(sha 0ef3751b, immutable) so the version has a lineage-comparable row. This pool carries NEITHER the band
clearance NOR the band-feasible conditioning. Cells: JT step 42000 (09c33bf4) and v2.7.4 (244f4f83), each
mode=none and kstep k=5, LEGACY first (lineage-comparable) then BANDED (this version's definition). Full
component breakdown + scene-bootstrap 95% CIs + collision split (cylinder/band/floor/ceiling). Forced onto GPU
(results device-invariant) since the M7 CPU path was the bottleneck. Eval-only. Artifacts under stage2_eval/."""
from __future__ import annotations

import copy, json, time
from pathlib import Path

import torch

from src.eval.evaluate import evaluate
from src.eval.bootstrap import within_seed_ci
from src.eval.run_full import _load_framework as load_fw
from scripts.analysis.v276_m7_fallback_trial import _split, _filter_for, POOLS

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
STEM = "eval_full_quadrotor-3d-d2r_n2000_seed23456"
OUT = REPO / "data/runs/v2.7.6/stage2_eval"
JT42 = REPO / "data/runs/v2.7.6/set__20260725-043415__seed42/v2.7.6__jt__20260725-052127__seed42/checkpoints/step_042000.pt"
V274 = REPO / "data/runs/v2.7.4/set__20260720-083533__seed42/v2.7.4__20260720-091830__seed42/checkpoints/best.pt"
COMPS = ("cps", "reach", "collision", "oob", "stuck", "timeout", "infeasibility")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run(ckpt, arm, mode, scoring):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    boot = ck["config"]["eval"]["bootstrap"]; n_boot = int(boot["n_resample"]); bseed = int(boot["seed"])
    bc = 0.0 if scoring == "legacy" else 4.0
    over = {"env": {"dt": 0.05, "stuck_window_steps": 60, "stuck_radius": 0.10, "band_collision_limit": bc},
            "eval": {"max_steps": 200, "dt_ctrl": 0.05}, "filter": _filter_for(ck, mode)}
    fw, cfg, _ = load_fw(ckpt, config_overrides=over)
    for net in ("value_net", "policy_net"):
        m = getattr(fw, net, None)
        if m is not None:
            m.to(DEV)
    t0 = time.perf_counter()
    res = evaluate(fw, POOLS / f"{STEM}.pkl", cfg, mode="final", step=int(ck["step"]),
                   ckpt_name=Path(ckpt).name, max_scenes=None, include_lqr_baseline=False)
    wall = round(time.perf_counter() - t0, 1)
    rows = list(res.episode_rows)
    ci = within_seed_ci(rows, n_resample=n_boot, seed=bseed)
    agg = {"arm": arm, "mode": mode, "scoring": scoring, "n": len(rows), "wall_s": wall, "device": str(DEV)}
    for m in COMPS:
        agg[m] = round(float(ci["mean"][m]), 6); agg[m + "_ci"] = [round(float(ci["ci"][m]["lo"]), 6), round(float(ci["ci"][m]["hi"]), 6)]
    agg["split"] = _split(res, n_boot, bseed)
    sp = agg["split"]
    print(f"[{arm}/{mode}/{scoring}] cps {agg['cps']:+.5f} {agg['cps_ci']} reach {agg['reach']:.4f} coll {agg['collision']:.4f} "
          f"| cyl {sp['cylinder']['count']} band {sp['band_total']['count']} (floor {sp['band_floor']['count']}/ceil {sp['band_ceiling']['count']}) "
          f"infeas {agg['infeasibility']:.4f} wall {wall}s dev {DEV}", flush=True)
    return agg


if __name__ == "__main__":
    results = []
    # (a) JT legacy, (b) v274 legacy, (c) all four banded
    plan = [(JT42, "jt42000", "none", "legacy"), (JT42, "jt42000", "kstep", "legacy"),
            (V274, "v274", "none", "legacy"), (V274, "v274", "kstep", "legacy"),
            (JT42, "jt42000", "none", "banded"), (JT42, "jt42000", "kstep", "banded"),
            (V274, "v274", "none", "banded"), (V274, "v274", "kstep", "banded")]
    for ckpt, arm, mode, scoring in plan:
        results.append(run(ckpt, arm, mode, scoring))
        (OUT / "canonical_eval.json").write_text(json.dumps({"pool": STEM, "pool_sha": "0ef3751b",
            "note": "canonical pool: no band clearance, no band-feasible conditioning", "cells": results}, indent=2) + "\n")
    print("canonical eval done ->", OUT / "canonical_eval.json", flush=True)
