"""v2.7.4 theory F1 — residual failure accounting by initial-tilt stratum (completes T1).

T1 reported collisions by stratum (11/23/7) and established that all 33 arena exits are vertical. F1 adds the
oob side: per stratum n, oob count and rate, collision count and rate, and the COMBINED failure rate on one
denominator; and for the oob episodes, the exit sign (floor vs ceiling) and the median |v_z| over the final
1.0 s before exit. Counts, not interpretation.

Method: same canonical re-roll as T1 (frozen d2r full pool n=2000, M6 checkpoint/config, batch 2000, dt 0.05,
max_steps 200); canonical outcomes/event steps from the run's own eval_episodes.csv. Eval-only.
"""
from __future__ import annotations

import csv, json
from pathlib import Path

import numpy as np
import torch

from src.eval.build_pools import load_pool, EVAL_POOLS_DIR
from src.eval.evaluate import _filter_adapter, _tensor_options, make_batched_scene
from src.eval.rollout import rollout_eval
from src.envs.scene_batch import initial_states_from_batch
from src.envs.quadrotor_3d import _quat_to_R
from src.frameworks.jt_pncbf.train import load_framework_from_checkpoint

RUN = Path("data/runs/v2.7.4/set__20260720-083533__seed42/v2.7.4__20260720-091830__seed42")
CKPT = RUN / "checkpoints/best.pt"
POOL = EVAL_POOLS_DIR / "eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl"
OUT = Path("data/runs/v2.7.4/theory"); OUT.mkdir(parents=True, exist_ok=True)

fw, cfg, _ = load_framework_from_checkpoint(CKPT)
system = fw.system
dt = float(cfg["env"]["dt"]); max_steps = int(cfg["eval"]["max_steps"])
oob_limit = float(cfg["env"]["oob_limit"])
scenes = load_pool(POOL).scenes
dtype, device = _tensor_options(system, fw)
bs = make_batched_scene(scenes, device=device, dtype=dtype)
res = rollout_eval(system, fw.policy, _filter_adapter(fw), bs, initial_states_from_batch(bs),
                   max_steps=max_steps, dt=dt, config=cfg)
S = res.states.detach().to(torch.float64).numpy()
T1, N = S.shape[0], S.shape[1]

outc, ev = {}, {}
with (RUN / "eval_episodes.csv").open() as f:
    for r in csv.DictReader(f):
        if r["mode"] == "final":
            i = int(r["episode_idx"]); outc[i] = r["outcome"]; ev[i] = int(float(r["n_steps"]))

q0 = torch.tensor(S[0, :, 3:7])
tilt0 = np.degrees(np.arccos(np.clip(_quat_to_R(q0)[:, 2, 2].numpy(), -1, 1)))
def strat_of(t): return "[0,60)" if t < 60 else ("[60,120)" if t < 120 else "[120,180]")
STRATA = ["[0,60)", "[60,120)", "[120,180]"]
WIN = int(round(1.0 / dt))

report = {"pool": POOL.name, "n_episodes": N, "oob_limit": oob_limit, "dt": dt,
          "window_steps_for_1s": WIN, "strata": {}}
for s in STRATA:
    idx = [i for i in range(N) if strat_of(tilt0[i]) == s and i in outc]
    n = len(idx)
    coll = [i for i in idx if outc[i] == "collision"]
    oob = [i for i in idx if outc[i] == "oob"]
    floor_n = ceil_n = 0
    vz1s = []
    for i in oob:
        k = int(np.clip(ev[i], 0, T1 - 1))
        pz = S[k, i, 2]
        if pz < -oob_limit: floor_n += 1
        elif pz > oob_limit: ceil_n += 1
        lo = max(0, k - WIN + 1)
        vz1s.append(float(np.mean(np.abs(S[lo:k + 1, i, 9]))))
    report["strata"][s] = {
        "n_episodes": n,
        "collision_count": len(coll), "collision_rate": round(len(coll) / n, 5) if n else None,
        "oob_count": len(oob), "oob_rate": round(len(oob) / n, 5) if n else None,
        "combined_failure_count": len(coll) + len(oob),
        "combined_failure_rate": round((len(coll) + len(oob)) / n, 5) if n else None,
        "oob_exit_sign": {"floor_below_minus_limit": floor_n, "ceiling_above_plus_limit": ceil_n},
        "oob_vz_final_1s_median": round(float(np.median(vz1s)), 5) if vz1s else None,
        "oob_vz_final_1s_iqr": ([round(float(np.percentile(vz1s, 25)), 5),
                                 round(float(np.percentile(vz1s, 75)), 5)] if vz1s else None),
    }
tot = {"collision": sum(v["collision_count"] for v in report["strata"].values()),
       "oob": sum(v["oob_count"] for v in report["strata"].values())}
tot["combined"] = tot["collision"] + tot["oob"]
report["totals"] = tot
(OUT / "f1_oob_by_stratum.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
