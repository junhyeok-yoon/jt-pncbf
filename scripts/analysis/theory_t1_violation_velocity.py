"""v2.7.4 theory T1 — violation-moment velocity decomposition (tests rem:vert-pred / prop:vert / cor:vert-filter).

Registered prediction (BEFORE data): over the M6 JT canonical eval, at the LAST step before each residual
collision, |v_z| / (|v_xy| + eps) is LARGER in the high-tilt stratum [120,180] than in [0,60), with the
stratum medians separated. Falsified if the distributions overlap with no median separation, or the ordering
reverses.

Method: re-roll the frozen d2r full pool through the canonical eval path with the M6 checkpoint and config
(batch 2000, dt 0.05, max_steps 200, stuck window 60) — the same path that reproduces the M6 headline
bit-identically — and read velocities out of the recorded states. Canonical outcomes/event steps come from the
run's own eval_episodes.csv. Also splits residual failures into cylinder contacts vs arena-bound exits
(floor/ceiling vs lateral), since cor:vert-filter(b) puts the arena bounds outside the avoid set. Eval-only.
"""
from __future__ import annotations

import copy, csv, json
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
EPS = 1e-6

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
fw, cfg, _ = load_framework_from_checkpoint(CKPT)
system = fw.system
dt = float(cfg["env"]["dt"]); max_steps = int(cfg["eval"]["max_steps"])
oob_limit = float(cfg["env"]["oob_limit"])
scenes = load_pool(POOL).scenes
dtype, device = _tensor_options(system, fw)
bs = make_batched_scene(scenes, device=device, dtype=dtype)
res = rollout_eval(system, fw.policy, _filter_adapter(fw), bs, initial_states_from_batch(bs),
                   max_steps=max_steps, dt=dt, config=cfg)
S = res.states.detach().to(torch.float64).numpy()          # [T+1, N, 13]
T1, N = S.shape[0], S.shape[1]

# canonical outcomes + event steps from the run's own artifact
outc, ev = {}, {}
with (RUN / "eval_episodes.csv").open() as f:
    for r in csv.DictReader(f):
        if r["mode"] == "final":
            i = int(r["episode_idx"]); outc[i] = r["outcome"]; ev[i] = int(float(r["n_steps"]))

# initial tilt per episode
q0 = torch.tensor(S[0, :, 3:7])
tilt0 = np.degrees(np.arccos(np.clip(_quat_to_R(q0)[:, 2, 2].numpy(), -1, 1)))

def strat_of(t):
    return "[0,60)" if t < 60 else ("[60,120)" if t < 120 else "[120,180]")

def q(a, p): return float(np.percentile(a, p)) if len(a) else None
def stats(a):
    a = np.asarray(a, float)
    return {"n": int(a.size), "median": round(q(a, 50), 5) if a.size else None,
            "iqr": [round(q(a, 25), 5), round(q(a, 75), 5)] if a.size else None} if a.size else {"n": 0}

WIN = int(round(1.0 / dt))                                  # 1.0 s window
rows = []
for i, o in outc.items():
    if o != "collision":
        continue
    k = int(np.clip(ev[i] - 1, 0, T1 - 1))                  # LAST step before the violation
    v = S[k, i, 7:10]
    vxy, vz = float(np.hypot(v[0], v[1])), float(abs(v[2]))
    lo = max(0, k - WIN + 1)
    seg = S[lo:k + 1, i, 7:10]
    mxy = float(np.mean(np.hypot(seg[:, 0], seg[:, 1]))); mz = float(np.mean(np.abs(seg[:, 2])))
    rows.append({"ep": i, "stratum": strat_of(tilt0[i]), "tilt0": round(float(tilt0[i]), 2),
                 "vxy": vxy, "vz": vz, "ratio": vz / (vxy + EPS),
                 "vxy_1s": mxy, "vz_1s": mz, "ratio_1s": mz / (mxy + EPS)})

report = {"source_rollout": str(CKPT), "pool": POOL.name, "n_episodes": N, "dt": dt,
          "window_steps_for_1s": WIN, "n_collisions_total": len(rows), "strata": {}}
for s in ("[0,60)", "[60,120)", "[120,180]"):
    g = [r for r in rows if r["stratum"] == s]
    report["strata"][s] = {
        "n_collisions": len(g),
        "at_violation_step": {"vz": stats([r["vz"] for r in g]), "vxy": stats([r["vxy"] for r in g]),
                              "ratio_vz_over_vxy": stats([r["ratio"] for r in g])},
        "mean_over_final_1.0s": {"vz": stats([r["vz_1s"] for r in g]), "vxy": stats([r["vxy_1s"] for r in g]),
                                 "ratio_vz_over_vxy": stats([r["ratio_1s"] for r in g])},
    }

# --- residual-failure split: cylinder contact vs arena-bound exit (floor/ceiling vs lateral) ---
split = {"collision_cylinder_contact": 0, "oob_total": 0, "oob_vertical_floor_or_ceiling": 0,
         "oob_lateral_xy": 0, "oob_both": 0}
for i, o in outc.items():
    if o == "collision":
        split["collision_cylinder_contact"] += 1
    elif o == "oob":
        split["oob_total"] += 1
        k = int(np.clip(ev[i], 0, T1 - 1)); p = S[k, i, 0:3]
        vert = abs(p[2]) > oob_limit; lat = (abs(p[0]) > oob_limit) or (abs(p[1]) > oob_limit)
        if vert and lat: split["oob_both"] += 1
        elif vert: split["oob_vertical_floor_or_ceiling"] += 1
        elif lat: split["oob_lateral_xy"] += 1
report["residual_failure_split"] = split
report["oob_limit"] = oob_limit
(OUT / "t1_violation_velocity.json").write_text(json.dumps(report, indent=2) + "\n")
json.dump(rows, open(OUT / "t1_per_collision.json", "w"), indent=1)
print(json.dumps(report, indent=2))
