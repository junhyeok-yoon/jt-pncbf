"""v2.7.6 R6.2 — ballistic-feasibility (recoverability) check on the eval-IC bounds.

Reuses the v2.6.2 3g-relaxed sound doom certificate (src/common/quadrotor_ballistic_doom.py). Relaxation
(verbatim from that module): point mass p_ddot=u, ||u||<=A=f_max/m+g, no attitude, no velocity clamp ->
AT LEAST AS CAPABLE as the plant, so relaxed-doom is a SOUND certificate of true doom. For quadrotor_3d
A = 19.62/1.0 + 9.81 = 29.43. Two sufficient doom tests per scene:
  (1) xy obstacle penetration -> is_doomed_ballistic on start[:2],vel[:2] vs the scene obstacle discs.
  (2) oob box escape -> per-axis reachable-set: deepest reachable z under full up-authority is the braking
      undershoot z0 - v_z_down^2/(2A); flagged-floor iff it crosses -oob_limit. Ceiling / walls symmetric.
Flagged = certified un-recoverable by the relaxed (over-capable) system. Registered prediction: ~0.
Run on the eval-IC pool (primary), plus the full-range seed42 and canonical seed23456 as the v2.6.x
reference. STATED LIMITATION (registered): the relaxed system is attitude-blind and its omnidirectional 3g
makes |v_z|<=1.5 floor recovery trivial -> flagged~0 is a necessary-but-weak check; it does NOT validate
the attitude-limited recovery-second slope. Read-only. No git, no securing.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.common.quadrotor_ballistic_doom import is_doomed_ballistic
from src.eval.build_pools import load_pool

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
POOLS = REPO / "data/runs/v2.7.6/pools"
OUT = REPO / "data/runs/v2.7.6/r62_feasibility"; OUT.mkdir(parents=True, exist_ok=True)
REG = json.loads((REPO / "data/runs/v2.7.6/registered_params.json").read_text())
A = float(REG["plant"]["accel_bound_A_3g"])          # 29.43
OOB = float(REG["plant"]["oob_limit"])               # 8.0
EPS = 1e-9

POOLSET = {
    "eval_ic_seed42":  "eval_evalicz_quadrotor-3d-d2r_n2000_seed42",
    "full_range_seed42": "eval_fullrange_quadrotor-3d-d2r_n2000_seed42",
    "canonical_seed23456": "eval_full_quadrotor-3d-d2r_n2000_seed23456",
}


def tilt_deg(quat):
    w, x, y, z = quat
    r22 = w * w - x * x - y * y + z * z
    return float(np.degrees(np.arccos(np.clip(r22, -1.0, 1.0))))


def axis_escape(p0, v0):
    """Deepest / highest reachable coord under full braking authority A (1D per axis).
    min reachable = p0 - max(0,-v0)^2/(2A) (brake a negative velocity); max = p0 + max(0,v0)^2/(2A)."""
    down = max(0.0, -v0) ** 2 / (2.0 * A)
    up = max(0.0, v0) ** 2 / (2.0 * A)
    return p0 - down, p0 + up


def check_pool(stem, use_buckets):
    pool = load_pool(POOLS / f"{stem}.pkl")
    n = len(pool.scenes)
    flagged = []
    n_obst = n_floor = n_ceil = n_wall = 0
    floor_deepest = []          # deepest reachable z (relaxed) per scene
    floor_margin = []           # deepest_z - (-OOB): >0 means safe
    for i, s in enumerate(pool.scenes):
        start = np.asarray(s.start, float); vel = np.asarray(s.initial_velocity, float)
        centers = np.asarray(s.obstacle_centers, float); radii = np.asarray(s.obstacle_radii, float)
        active = np.asarray(s.obstacle_active, bool)
        obst = is_doomed_ballistic(start[:2], vel[:2], centers, radii, active, A, use_buckets=use_buckets)
        # per-axis oob box escape
        zmin, zmax = axis_escape(start[2], vel[2])
        xmin, xmax = axis_escape(start[0], vel[0]); ymin, ymax = axis_escape(start[1], vel[1])
        floor = zmin < -OOB - EPS
        ceil = zmax > OOB + EPS
        wall = (xmin < -OOB - EPS) or (xmax > OOB + EPS) or (ymin < -OOB - EPS) or (ymax > OOB + EPS)
        floor_deepest.append(zmin); floor_margin.append(zmin + OOB)
        if obst: n_obst += 1
        if floor: n_floor += 1
        if ceil: n_ceil += 1
        if wall: n_wall += 1
        if obst or floor or ceil or wall:
            flagged.append({"episode_idx": i, "z": round(float(start[2]), 4),
                            "v_z": round(float(vel[2]), 4), "tilt_deg": round(tilt_deg(s.initial_attitude_quat), 2),
                            "obstacle": bool(obst), "floor": bool(floor), "ceiling": bool(ceil), "wall": bool(wall)})
    fm = np.array(floor_margin)
    return {"stem": stem, "n": n, "use_buckets": use_buckets,
            "flagged_total": len(flagged), "flagged_fraction": round(len(flagged) / n, 6),
            "by_mode": {"obstacle": n_obst, "floor": n_floor, "ceiling": n_ceil, "wall": n_wall},
            "flagged_ICs": flagged[:50],
            "relaxed_floor_margin_m": {"min": round(float(fm.min()), 4), "p1": round(float(np.percentile(fm, 1)), 4),
                                       "median": round(float(np.median(fm)), 4),
                                       "note": "deepest relaxed-reachable z minus (-oob_limit); >0 = certified floor-safe under 3g point mass"}}


def main():
    out = {"accel_bound_A": A, "oob_limit": OOB,
           "instrument": "src/common/quadrotor_ballistic_doom.py (3g-relaxed sound doom certificate)",
           "stated_limitation": REG["R6.2"]["stated_limitation"], "pools": {}}
    for key, stem in POOLSET.items():
        rec = check_pool(stem, use_buckets=True)
        rec_nb = check_pool(stem, use_buckets=False)     # sensitivity: exact radii (more flags)
        rec["flagged_fraction_no_buckets"] = rec_nb["flagged_fraction"]
        rec["flagged_total_no_buckets"] = rec_nb["flagged_total"]
        out["pools"][key] = rec
        print(f"[{key}] flagged {rec['flagged_total']}/{rec['n']} (frac {rec['flagged_fraction']}) "
              f"by_mode={rec['by_mode']}  floor_margin_min={rec['relaxed_floor_margin_m']['min']}m "
              f"| no-buckets flagged={rec_nb['flagged_total']}")
    # adjudication: prediction flagged~0 on eval-IC
    ic = out["pools"]["eval_ic_seed42"]
    out["R6.2_adjudication"] = {
        "eval_ic_flagged_fraction": ic["flagged_fraction"],
        "prediction_flagged_approx_zero": ic["flagged_fraction"] == 0.0,
        "verdict": "confirmed" if ic["flagged_total"] == 0 else "falsified",
        "verdict_scope": ("relaxed-3g-point-mass recoverability only; attitude-limited recovery-second slope "
                          "is NOT validated by this instrument (stated_limitation)")}
    (OUT / "r62_summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nR6.2 verdict (eval-IC): {out['R6.2_adjudication']['verdict']} "
          f"(flagged {ic['flagged_total']}/{ic['n']})")


if __name__ == "__main__":
    main()
