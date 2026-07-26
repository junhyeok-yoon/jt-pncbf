"""v2.7.6 Stage-1c M4 — re-run the Stage-1 baseline on the band-clearance pool (checkpoint 244f4f83, seed 42,
deployed defaults). Reports cps + components (04_eval s5 within_seed_ci), band_exit + CI, and the
floor/ceiling split. Writes eval_episodes.csv / eval_metrics.csv per pool under data/runs/v2.7.6/
stage1c_baseline/. Compares against the Stage-1 band-feasible baseline. No git, no securing.
"""
from __future__ import annotations

import csv, json
from pathlib import Path

import numpy as np

from src.eval.evaluate import evaluate, EVAL_EPISODE_COLUMNS
from src.eval.bootstrap import within_seed_ci
from src.eval.band_exit import episode_band_exit, BAND_FLOOR, BAND_CEILING
from scripts.analysis.v276_stage1_baseline import load_fw, POOLS

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
OUT = REPO / "data/runs/v2.7.6/stage1c_baseline"; OUT.mkdir(parents=True, exist_ok=True)
POOLSET = {"band_feasible": "eval_bandfeasible_quadrotor-3d-d2r_n2000_seed42",
           "full_range": "eval_fullrange_quadrotor-3d-d2r_n2000_seed42"}
METRICS = ("cps", "reach", "collision", "oob", "stuck", "timeout", "infeasibility")
STAGE1_BAND = {"cps": 0.853365, "band_exit": 0.0985}   # Stage-1 (pre-clearance) band-feasible baseline


def run_pool(fw, cfg, ck, key, stem, n_boot, bseed):
    outdir = OUT / key; outdir.mkdir(parents=True, exist_ok=True)
    res = evaluate(fw, POOLS / f"{stem}.pkl", cfg, mode="final", step=int(ck["step"]),
                   ckpt_name="best.pt", max_scenes=None, include_lqr_baseline=False)
    rows = [dict(r) for r in res.episode_rows]
    be = np.zeros(len(rows)); n_floor = n_ceil = 0
    for i, (r, tr) in enumerate(zip(rows, res.trajectories)):
        n = int(float(r["n_steps"]))
        z = fw.system.position(tr.filtered.states)[:, 0, 2].detach().cpu().numpy()
        z = z[: min(n, z.shape[0] - 1) + 1]
        exited = (z < BAND_FLOOR) | (z > BAND_CEILING)
        f = 1.0 if exited.any() else 0.0
        r["band_exit"] = f; be[i] = f
        if f:
            if z[int(np.argmax(exited))] < BAND_FLOOR: n_floor += 1
            else: n_ceil += 1
    with (outdir / "eval_episodes.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(EVAL_EPISODE_COLUMNS) + ["band_exit"], extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    ci = within_seed_ci(rows, n_resample=n_boot, seed=bseed)
    idx = np.random.default_rng(bseed).integers(0, len(rows), size=(n_boot, len(rows)))
    be_ci = [round(float(np.percentile(be[idx].mean(1), 2.5)), 6), round(float(np.percentile(be[idx].mean(1), 97.5)), 6)]
    agg = {m: {"mean": round(float(ci["mean"][m]), 6),
               "ci95": [round(float(ci["ci"][m]["lo"]), 6), round(float(ci["ci"][m]["hi"]), 6)]} for m in METRICS}
    agg["saturation_rate"] = round(float(res.eval_row["saturation_rate"]), 6)
    agg["band_exit"] = {"mean": round(float(be.mean()), 6), "ci95": be_ci, "count": int(be.sum()),
                        "floor": n_floor, "ceiling": n_ceil}
    mrow = {"pool": key, "stem": stem, "n": len(rows)}
    for m in METRICS:
        mrow[m] = agg[m]["mean"]; mrow[f"{m}_ci_lo"] = agg[m]["ci95"][0]; mrow[f"{m}_ci_hi"] = agg[m]["ci95"][1]
    mrow["band_exit"] = agg["band_exit"]["mean"]; mrow["band_exit_ci_lo"] = be_ci[0]; mrow["band_exit_ci_hi"] = be_ci[1]
    mrow["band_exit_floor"] = n_floor; mrow["band_exit_ceiling"] = n_ceil
    with (outdir / "eval_metrics.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(mrow.keys())); w.writeheader(); w.writerow(mrow)
    print(f"[{key}] cps={agg['cps']['mean']:.6f}{agg['cps']['ci95']} reach={agg['reach']['mean']:.4f} "
          f"coll={agg['collision']['mean']:.4f} oob={agg['oob']['mean']:.4f} "
          f"band_exit={agg['band_exit']['mean']:.4f}{be_ci} (n={int(be.sum())}, floor={n_floor} ceil={n_ceil})")
    return {"key": key, "stem": stem, "n": len(rows), "agg": agg}


def main():
    fw, cfg, ck = load_fw()
    boot = ck["config"]["eval"]["bootstrap"]; n_boot = int(boot["n_resample"]); bseed = int(boot["seed"])
    out = {}
    for key, stem in POOLSET.items():
        out[key] = run_pool(fw, cfg, ck, key, stem, n_boot, bseed)
    b = out["band_feasible"]["agg"]
    out["vs_stage1_band_feasible"] = {
        "stage1_cps": STAGE1_BAND["cps"], "stage1c_cps": b["cps"]["mean"],
        "stage1_band_exit": STAGE1_BAND["band_exit"], "stage1c_band_exit": b["band_exit"]["mean"],
        "band_exit_delta_1c_minus_1": round(b["band_exit"]["mean"] - STAGE1_BAND["band_exit"], 6)}
    (OUT / "stage1c_baseline.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nband-feasible: Stage-1 cps {STAGE1_BAND['cps']:.4f} band_exit {STAGE1_BAND['band_exit']:.4f} -> "
          f"Stage-1c cps {b['cps']['mean']:.4f} band_exit {b['band_exit']['mean']:.4f} "
          f"(delta {out['vs_stage1_band_feasible']['band_exit_delta_1c_minus_1']:+.4f})")


if __name__ == "__main__":
    main()
