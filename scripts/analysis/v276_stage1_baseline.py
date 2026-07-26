"""v2.7.6 Stage-1 M4 — baseline eval of the v2.7.4 checkpoint on the band-feasible and full-range pools.

Deployed default settings (dt=dt_ctrl=0.05, max_steps=200, stuck_window=60, empty_fallback=none). Per pool:
writes eval_episodes.csv (standard EVAL_EPISODE_COLUMNS + band_exit) and eval_metrics.csv, with cps and its
components via the 04_eval s5 scene-bootstrap (within_seed_ci), plus the band_exit fraction with a CI from
the SAME resampling (seed/n_resample). band_exit is a recorded diagnostic, NOT a cps term. No git, no securing.
"""
from __future__ import annotations

import copy, csv, json
from pathlib import Path

import numpy as np
import torch

from src.eval.evaluate import evaluate, EVAL_EPISODE_COLUMNS
from src.eval.bootstrap import within_seed_ci
from src.eval.band_exit import episode_band_exit, BAND_FLOOR, BAND_CEILING
from src.frameworks.jt_pncbf.train import load_framework_from_checkpoint

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
CKPT = REPO / "data/runs/v2.7.4/set__20260720-083533__seed42/v2.7.4__20260720-091830__seed42/checkpoints/best.pt"
POOLS = REPO / "data/runs/v2.7.6/pools"
OUT = REPO / "data/runs/v2.7.6/stage1_baseline"; OUT.mkdir(parents=True, exist_ok=True)
POOLSET = {"band_feasible": "eval_bandfeasible_quadrotor-3d-d2r_n2000_seed42",
           "full_range": "eval_fullrange_quadrotor-3d-d2r_n2000_seed42"}
METRICS = ("cps", "reach", "collision", "oob", "stuck", "timeout", "infeasibility")


def load_fw():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    over = {"env": {"dt": 0.05, "stuck_window_steps": 60, "stuck_radius": 0.10},
            "eval": {"max_steps": 200, "dt_ctrl": 0.05},
            "filter": copy.deepcopy(ck["config"]["filter"])}
    over["filter"]["empty_fallback"] = {"mode": "none", "k": int(over["filter"].get("empty_fallback", {}).get("k", 10))}
    fw, cfg, _ = load_framework_from_checkpoint(CKPT, config_overrides=over)
    assert abs(float(cfg["env"]["dt"]) - 0.05) < 1e-12 and int(cfg["eval"]["max_steps"]) == 200
    assert cfg["filter"]["empty_fallback"]["mode"] == "none"
    return fw, cfg, ck


def run_pool(fw, cfg, ck, key, stem, n_boot, bseed, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    res = evaluate(fw, POOLS / f"{stem}.pkl", cfg, mode="final", step=int(ck["step"]),
                   ckpt_name=CKPT.name, max_scenes=None, include_lqr_baseline=False)
    rows = [dict(r) for r in res.episode_rows]
    be = np.empty(len(rows))
    for i, (r, tr) in enumerate(zip(rows, res.trajectories)):
        f = episode_band_exit(tr.filtered.states, int(float(r["n_steps"])), fw.system)
        r["band_exit"] = f; be[i] = f
    # eval_episodes.csv (standard columns + band_exit)
    cols = list(EVAL_EPISODE_COLUMNS) + ["band_exit"]
    with (outdir / "eval_episodes.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for r in rows:
            w.writerow(r)
    # 04_eval s5 scene-bootstrap for cps + components
    ci = within_seed_ci(rows, n_resample=n_boot, seed=bseed)
    # band_exit bootstrap with the SAME resampling
    idx = np.random.default_rng(bseed).integers(0, len(rows), size=(n_boot, len(rows)))
    be_rs = be[idx].mean(axis=1)
    be_ci = [float(np.percentile(be_rs, 2.5)), float(np.percentile(be_rs, 97.5))]
    agg = {m: {"mean": round(float(ci["mean"][m]), 6),
               "ci95": [round(float(ci["ci"][m]["lo"]), 6), round(float(ci["ci"][m]["hi"]), 6)]} for m in METRICS}
    agg["saturation_rate"] = round(float(res.eval_row["saturation_rate"]), 6)
    agg["band_exit"] = {"mean": round(float(be.mean()), 6), "ci95": [round(be_ci[0], 6), round(be_ci[1], 6)],
                        "count": int(be.sum())}
    # eval_metrics.csv (aggregate row)
    mrow = {"pool": key, "stem": stem, "n": len(rows), "step": int(ck["step"]),
            "band_floor": BAND_FLOOR, "band_ceiling": BAND_CEILING}
    for m in METRICS:
        mrow[m] = agg[m]["mean"]; mrow[f"{m}_ci_lo"] = agg[m]["ci95"][0]; mrow[f"{m}_ci_hi"] = agg[m]["ci95"][1]
    mrow["band_exit"] = agg["band_exit"]["mean"]; mrow["band_exit_ci_lo"] = be_ci[0]; mrow["band_exit_ci_hi"] = be_ci[1]
    with (outdir / "eval_metrics.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(mrow.keys())); w.writeheader(); w.writerow(mrow)
    rec = {"key": key, "stem": stem, "n": len(rows), "agg": agg,
           "csv": {"episodes": str(outdir / "eval_episodes.csv"), "metrics": str(outdir / "eval_metrics.csv")}}
    (outdir / "summary.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(f"[{key}] cps={agg['cps']['mean']:.6f}{agg['cps']['ci95']} reach={agg['reach']['mean']:.4f} "
          f"coll={agg['collision']['mean']:.4f} oob={agg['oob']['mean']:.4f} "
          f"band_exit={agg['band_exit']['mean']:.4f}{agg['band_exit']['ci95']} (n={int(be.sum())})")
    return rec


def main():
    fw, cfg, ck = load_fw()
    boot = ck["config"]["eval"]["bootstrap"]; n_boot = int(boot["n_resample"]); bseed = int(boot["seed"])
    out = {}
    for key, stem in POOLSET.items():
        out[key] = run_pool(fw, cfg, ck, key, stem, n_boot, bseed, OUT / key)
    # diff band_exit (band - full), unpaired bootstrap
    def load_be(key, stem):
        import csv as _csv
        rows = list(_csv.DictReader(open(OUT / key / "eval_episodes.csv")))
        return np.array([float(r["band_exit"]) for r in rows])
    beb = load_be("band_feasible", POOLSET["band_feasible"]); bef = load_be("full_range", POOLSET["full_range"])
    rng = np.random.default_rng(bseed)
    ib = rng.integers(0, len(beb), (n_boot, len(beb))); iff = rng.integers(0, len(bef), (n_boot, len(bef)))
    d = beb[ib].mean(1) - bef[iff].mean(1)
    out["band_exit_diff_band_minus_full"] = {"delta": round(float(beb.mean() - bef.mean()), 6),
        "ci95": [round(float(np.percentile(d, 2.5)), 6), round(float(np.percentile(d, 97.5)), 6)]}
    (OUT / "stage1_baseline.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nband_exit: band-feasible {out['band_feasible']['agg']['band_exit']['mean']:.4f} vs "
          f"full-range {out['full_range']['agg']['band_exit']['mean']:.4f} "
          f"delta={out['band_exit_diff_band_minus_full']['delta']:+.4f} CI{out['band_exit_diff_band_minus_full']['ci95']}")


if __name__ == "__main__":
    main()
