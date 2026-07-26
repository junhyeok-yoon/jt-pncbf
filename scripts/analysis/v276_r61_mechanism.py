"""v2.7.6 R6.1 mechanism — why did the eval-IC split raise floor exits? Stratify floor-exit rate by start_z
band and by tilt, joining per-episode outcomes (r61_screen JSONs) to the pool scenes. Read-only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.build_pools import load_pool

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
POOLS = REPO / "data/runs/v2.7.6/pools"
SCR = REPO / "data/runs/v2.7.6/r61_screen"
OUT = SCR


def tilt_deg(q):
    w, x, y, z = q
    return float(np.degrees(np.arccos(np.clip(w * w - x * x - y * y + z * z, -1.0, 1.0))))


def analyze(pool_stem, screen_key):
    pool = load_pool(POOLS / f"{pool_stem}.pkl")
    d = json.load(open(SCR / f"{screen_key}.json"))
    floor = np.array(d["arr"]["oob_floor"])
    sz = np.array([float(s.start[2]) for s in pool.scenes])
    vz = np.array([float(s.initial_velocity[2]) for s in pool.scenes])
    tl = np.array([tilt_deg(s.initial_attitude_quat) for s in pool.scenes])
    bands = [("z<-4", sz < -4), ("-4<=z<-2", (sz >= -4) & (sz < -2)),
             ("-2<=z<0", (sz >= -2) & (sz < 0)), ("z>=0", sz >= 0)]
    by_z = {name: {"n": int(m.sum()), "floor_exits": int(floor[m].sum()),
                   "floor_rate": round(float(floor[m].mean()), 4) if m.sum() else None} for name, m in bands}
    desc = tl > 60
    by_tilt = {"tilt>60_descending": {"n": int(desc.sum()), "floor_exits": int(floor[desc].sum()),
                                      "floor_rate": round(float(floor[desc].mean()), 4)},
               "tilt<=60_holding": {"n": int((~desc).sum()), "floor_exits": int(floor[~desc].sum()),
                                    "floor_rate": round(float(floor[~desc].mean()), 4)}}
    return {"stem": pool_stem, "n": len(pool.scenes), "total_floor_exits": int(floor.sum()),
            "floor_rate_by_start_z": by_z, "floor_rate_by_tilt": by_tilt,
            "frac_start_below_-4": round(float((sz < -4).mean()), 4)}


def main():
    out = {"baseline_fullrange": analyze("eval_fullrange_quadrotor-3d-d2r_n2000_seed42", "baseline_seed42"),
           "treatment_evalic": analyze("eval_evalicz_quadrotor-3d-d2r_n2000_seed42", "treatment_seed42")}
    # how many of the treatment's floor exits come from the deep band it newly samples (z<-4)?
    t = out["treatment_evalic"]
    deep = t["floor_rate_by_start_z"]["z<-4"]
    out["mechanism_reading"] = {
        "treatment_deep_band_z<-4": deep,
        "treatment_total_floor_exits": t["total_floor_exits"],
        "share_of_treatment_floor_from_deep_band": round(deep["floor_exits"] / max(t["total_floor_exits"], 1), 4),
        "note": ("full-range never samples z<-4; the eval-IC floor (-6.0 for non-descending v_z) newly "
                 "populates that band. If a large share of the treatment's floor exits come from z<-4, the "
                 "increase is caused by the deeper starts the v_z-only bound admits, not protected against.")}
    (OUT / "r61_mechanism.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
