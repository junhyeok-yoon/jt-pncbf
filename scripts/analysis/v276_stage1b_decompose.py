"""v2.7.6 Stage-1b — decompose the residual band_exit (197 episodes) on the band-feasible pool.

Eval-only re-run of the Stage-1 baseline (same checkpoint 244f4f83, seed 42, deployed defaults, same pool),
recording per band-exit episode: first-exit step t_exit, first-exit boundary (floor/ceiling), initial tilt
theta_0 and that IC's D_down, distance-to-goal + goal_z at t_exit, and exit depths. No change to h_star /
filter / loss / sampler / cps / oob. Persists per-episode records under data/runs/v2.7.6/stage1b/.
Measurement only. No git, no securing.
"""
from __future__ import annotations

import csv, json, math
from pathlib import Path

import numpy as np

from src.eval.evaluate import evaluate
from src.eval.band_exit import BAND_FLOOR, BAND_CEILING
from scripts.analysis.v276_stage1_baseline import load_fw, POOLS
from scripts.analysis.v276_attitude_feasibility import D_down_single

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
OUT = REPO / "data/runs/v2.7.6/stage1b"; OUT.mkdir(parents=True, exist_ok=True)
STEM = "eval_bandfeasible_quadrotor-3d-d2r_n2000_seed42"
GOAL_RADIUS = 0.15
NEAR_GOAL = GOAL_RADIUS + 1.0        # 1.15 m
EARLY_STEPS = 40                     # 2 s at dt=0.05


def tilt_deg(q):
    q = np.asarray(q, float)
    return math.degrees(math.acos(min(1.0, max(-1.0, float(q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2)))))


def main():
    fw, cfg, ck = load_fw()
    res = evaluate(fw, POOLS / f"{STEM}.pkl", cfg, mode="final", step=int(ck["step"]),
                   ckpt_name="best.pt", max_scenes=None, include_lqr_baseline=False)
    recs = []
    n_floor = n_ceil = 0
    oob_flags = np.zeros(len(res.episode_rows))
    for i, (r, tr) in enumerate(zip(res.episode_rows, res.trajectories)):
        n = int(float(r["n_steps"]))
        pos = fw.system.position(tr.filtered.states)[:, 0, :].detach().cpu().numpy()   # [T+1,3]
        k = min(n, pos.shape[0] - 1)
        z = pos[:k + 1, 2]
        oob_flags[i] = 1.0 if (z.min() < -8.0 or z.max() > 8.0) else 0.0
        exited = (z < BAND_FLOOR) | (z > BAND_CEILING)
        if not exited.any():
            continue
        t_exit = int(np.argmax(exited))                       # first True index
        boundary = "floor" if z[t_exit] < BAND_FLOOR else "ceiling"
        if boundary == "floor": n_floor += 1
        else: n_ceil += 1
        sc = tr.scene
        goal = np.asarray(sc.goal, float)
        theta0 = tilt_deg(sc.initial_attitude_quat)
        vz0 = float(sc.initial_velocity[2]); om0 = float(np.linalg.norm(sc.initial_omega_vec))
        dd = D_down_single(math.radians(theta0), vz0, om0)
        dist_goal = float(np.linalg.norm(pos[t_exit] - goal))
        recs.append({"episode_idx": int(r["episode_idx"]), "t_exit": t_exit, "boundary": boundary,
                     "theta0_deg": round(theta0, 2), "D_down": round(dd, 4),
                     "vz0": round(vz0, 4), "omega0": round(om0, 4),
                     "dist_to_goal_at_exit": round(dist_goal, 4), "goal_z": round(float(goal[2]), 4),
                     "start_z": round(float(sc.start[2]), 4),
                     "exit_depth_at_first": round(abs(z[t_exit]) - 4.0, 4),
                     "max_excursion": round(max(BAND_FLOOR - z.min(), z.max() - BAND_CEILING), 4),
                     "outcome": r["outcome"]})
    N = len(recs)
    # persist per-episode records
    with (OUT / "band_exit_episodes.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0].keys())); w.writeheader(); w.writerows(recs)

    A = {k: np.array([r[k] for r in recs]) for k in ("t_exit", "theta0_deg", "goal_z", "dist_to_goal_at_exit", "D_down")}
    # (1) t_exit histogram + early/late
    bins = [0, 10, 20, 40, 80, 120, 200]
    hist = {f"{bins[j]}-{bins[j+1]}": int(((A['t_exit'] >= bins[j]) & (A['t_exit'] < bins[j+1])).sum()) for j in range(len(bins) - 1)}
    early = int((A["t_exit"] <= EARLY_STEPS).sum())
    # (2) near-goal share
    near = int((A["dist_to_goal_at_exit"] < NEAR_GOAL).sum())
    # (3) joint tilt x goal_z strata
    tilt_bins = [("[0,60)", 0, 60), ("[60,120)", 60, 120), ("[120,180]", 120, 180.01)]
    gz_bins = [("[-4,-3)", -4, -3), ("[-3,0)", -3, 0), ("[0,4]", 0, 4.01)]
    joint = {}
    for tl, lo, hi in tilt_bins:
        for gl, glo, ghi in gz_bins:
            m = (A["theta0_deg"] >= lo) & (A["theta0_deg"] < hi) & (A["goal_z"] >= glo) & (A["goal_z"] < ghi)
            joint[f"tilt{tl} x goalz{gl}"] = int(m.sum())
    # (4) counterfactual goal_z restriction (filter existing episodes)
    cf = {}
    for d in (0.5, 1.0, 1.5):
        remain = int(((A["goal_z"] >= -4 + d) & (A["goal_z"] <= 4 - d)).sum())
        cf[f"d={d}"] = {"goal_z_in[-4+d,4-d]": [round(-4 + d, 2), round(4 - d, 2)],
                        "exits_remaining": remain, "exits_removed": N - remain,
                        "removed_frac": round((N - remain) / N, 4)}
    # (5) floor/ceiling + oob CI
    rng = np.random.default_rng(int(ck["config"]["eval"]["bootstrap"]["seed"]))
    nb = int(ck["config"]["eval"]["bootstrap"]["n_resample"])
    idx = rng.integers(0, len(oob_flags), (nb, len(oob_flags)))
    oob_rs = oob_flags[idx].mean(1)
    oob_ci = [round(float(np.percentile(oob_rs, 2.5)), 6), round(float(np.percentile(oob_rs, 97.5)), 6)]

    report = {
        "n_band_exit": N, "pool": STEM, "near_goal_threshold_m": NEAR_GOAL,
        "1_t_exit": {"histogram": hist, "early_le40": early, "late_gt40": N - early,
                     "early_frac": round(early / N, 4), "median_t_exit": int(np.median(A["t_exit"]))},
        "2_near_goal": {"count": near, "frac": round(near / N, 4), "threshold_m": NEAR_GOAL,
                        "dist_to_goal_median": round(float(np.median(A["dist_to_goal_at_exit"])), 4)},
        "3_joint_tilt_goalz": joint,
        "4_counterfactual_goalz_restriction": cf,
        "5_floor_ceiling_and_oob": {"floor": n_floor, "ceiling": n_ceil,
                                    "oob_|z|>8_mean": round(float(oob_flags.mean()), 6), "oob_ci95": oob_ci,
                                    "oob_count": int(oob_flags.sum())},
    }
    (OUT / "stage1b_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
