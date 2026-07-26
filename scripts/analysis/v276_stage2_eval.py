"""v2.7.6 Stage-2 dual eval — score a checkpoint on a pool under legacy (band_collision=0) AND banded
(band_collision=4.0), deployed default settings, scene-bootstrap 95% CIs + band_exit. Reused by M5; run here
first for the v2.7.4 checkpoint on the band-feasible pool (the band-blind baseline needed for M5 and the
cps-floor question). band_collision only changes outcome scoring/termination; two evaluate() passes give the
two scorings faithfully. No git, no securing."""
from __future__ import annotations

import copy, json, sys
from pathlib import Path

import numpy as np
import torch

from src.eval.evaluate import evaluate
from src.eval.bootstrap import within_seed_ci
from src.eval.band_exit import episode_band_exit
from src.eval.run_full import _load_framework as load_framework_from_checkpoint

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
POOLS = REPO / "data/runs/v2.7.6/pools"
METRICS = ("cps", "reach", "collision", "oob", "stuck", "timeout", "infeasibility")


def eval_dual(ckpt: Path, stem: str, tag: str, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    boot = ck["config"]["eval"]["bootstrap"]; n_boot = int(boot["n_resample"]); bseed = int(boot["seed"])
    res_out = {}
    for scoring, bc in (("legacy", 0.0), ("banded", 4.0)):
        over = {"env": {"dt": 0.05, "stuck_window_steps": 60, "stuck_radius": 0.10, "band_collision_limit": bc},
                "eval": {"max_steps": 200, "dt_ctrl": 0.05}, "filter": copy.deepcopy(ck["config"]["filter"])}
        over["filter"]["empty_fallback"] = {"mode": "none", "k": int(over["filter"].get("empty_fallback", {}).get("k", 10))}
        fw, cfg, _ = load_framework_from_checkpoint(ckpt, config_overrides=over)
        assert abs(float(cfg["env"].get("band_collision_limit", 0.0)) - bc) < 1e-9
        res = evaluate(fw, POOLS / f"{stem}.pkl", cfg, mode="final", step=int(ck["step"]),
                       ckpt_name=ckpt.name, max_scenes=None, include_lqr_baseline=False)
        rows = list(res.episode_rows)
        be = np.array([episode_band_exit(tr.filtered.states, int(float(r["n_steps"])), fw.system)
                       for r, tr in zip(rows, res.trajectories)])
        ci = within_seed_ci(rows, n_resample=n_boot, seed=bseed)
        idx = np.random.default_rng(bseed).integers(0, len(rows), (n_boot, len(rows)))
        agg = {m: {"mean": round(float(ci["mean"][m]), 6),
                   "ci95": [round(float(ci["ci"][m]["lo"]), 6), round(float(ci["ci"][m]["hi"]), 6)]} for m in METRICS}
        agg["band_exit"] = {"mean": round(float(be.mean()), 6),
                            "ci95": [round(float(np.percentile(be[idx].mean(1), 2.5)), 6),
                                     round(float(np.percentile(be[idx].mean(1), 97.5)), 6)]}
        res_out[scoring] = agg
        print(f"[{tag}/{scoring}] cps={agg['cps']['mean']:.6f}{agg['cps']['ci95']} "
              f"collision={agg['collision']['mean']:.4f} oob={agg['oob']['mean']:.4f} "
              f"reach={agg['reach']['mean']:.4f} band_exit={agg['band_exit']['mean']:.4f}")
    rec = {"tag": tag, "ckpt": str(ckpt), "stem": stem, "step": int(ck["step"]), "dual": res_out}
    (outdir / f"{tag}.json").write_text(json.dumps(rec, indent=2) + "\n")
    return rec


if __name__ == "__main__":
    V274 = REPO / "data/runs/v2.7.4/set__20260720-083533__seed42/v2.7.4__20260720-091830__seed42/checkpoints/best.pt"
    OUT = REPO / "data/runs/v2.7.6/stage2_eval"
    eval_dual(V274, "eval_bandfeasible_quadrotor-3d-d2r_n2000_seed42", "v274_bandfeasible", OUT)
