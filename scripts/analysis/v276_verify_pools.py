"""v2.7.6 M2 check — confirm the eval-IC pool satisfies the registered z-bounds exactly (0 violations) and
characterize both pools' z / v_z / tilt marginals. Read-only on the built pools."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.build_pools import load_pool

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
OUT = REPO / "data/runs/v2.7.6/pools"
REG = json.loads((REPO / "data/runs/v2.7.6/registered_params.json").read_text())
Z = REG["ic_eval_z_bounds"]


def tilt_deg(quat):
    w, x, y, z = quat
    r22 = w * w - x * x - y * y + z * z          # world-z component of body-up axis = cos(tilt)
    return float(np.degrees(np.arccos(np.clip(r22, -1.0, 1.0))))


def extract(pool):
    sz, gz, vz, tilt = [], [], [], []
    for s in pool.scenes:
        sz.append(float(s.start[2])); gz.append(float(s.goal[2]))
        vz.append(float(s.initial_velocity[2])); tilt.append(tilt_deg(s.initial_attitude_quat))
    return map(np.array, (sz, gz, vz, tilt))


def dist(a):
    return {"n": int(a.size), "min": round(float(a.min()), 4), "p10": round(float(np.percentile(a, 10)), 4),
            "median": round(float(np.median(a)), 4), "p90": round(float(np.percentile(a, 90)), 4),
            "max": round(float(a.max()), 4)}


def main():
    out = {}
    for tag, stem in (("full_range_seed42", "eval_fullrange_quadrotor-3d-d2r_n2000_seed42"),
                      ("eval_ic_seed42", "eval_evalicz_quadrotor-3d-d2r_n2000_seed42")):
        pool = load_pool(OUT / f"{stem}.pkl")
        sz, gz, vz, tilt = extract(pool)
        rec = {"stem": stem, "start_z": dist(sz), "goal_z": dist(gz), "v_z": dist(vz), "tilt_deg": dist(tilt),
               "frac_start_below_-4": round(float((sz < -4.0).mean()), 4),
               "frac_tilt_gt_60deg": round(float((tilt > 60.0).mean()), 4)}
        if tag == "eval_ic_seed42":
            vz_down = np.maximum(0.0, -vz); vz_up = np.maximum(0.0, vz)
            zmax = np.minimum(Z["z_ceiling_base"], Z["z_ceiling_base"] - Z["z_ceiling_v_up_slope"] * vz_up)
            zmin = Z["z_floor_base"] + Z["z_floor_v_down_slope"] * vz_down
            eps = 1e-9
            viol_lo = int((sz < zmin - eps).sum()); viol_hi = int((sz > zmax + eps).sum())
            viol_goal = int((np.abs(gz) > Z["goal_z_half"] + eps).sum())
            rec["bound_violations"] = {"below_floor": viol_lo, "above_ceiling": viol_hi,
                                       "goal_out_of_range": viol_goal,
                                       "ALL_ZERO": (viol_lo == viol_hi == viol_goal == 0)}
            rec["slack_start_to_floor_m"] = {"median": round(float(np.median(sz - zmin)), 4),
                                             "min": round(float((sz - zmin).min()), 4)}
        out[tag] = rec
    (OUT / "pool_verify.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
