"""v2.7.4 theory G4 — goal altitude of the floor-exit episodes vs the recovering controls.

Registered before the data: if the floor exits are overshooting a legitimate descent, their goal_z is lower
than the controls' and the goal lies BELOW them at basement entry for a majority. If instead they lost
altitude against their objective, goal_z is comparable to the controls' and the goal lies ABOVE them.
Reported as counts and medians; no further interpretation.

Artifact sufficiency (checked, not assumed): goal_z/start_z come straight from the frozen pool, but the saved
records do NOT suffice for the rest — G3 persisted the 217-episode control group as AGGREGATES ONLY (no ids,
no per-episode z(t_4)), and the M6 run persisted eval_action_stream.npz (actions) with NO states. So the same
deterministic canonical re-roll used by G1/G3 is repeated here to recover per-episode z(t_4) for both entering
groups and the post-entry z trajectory for the 33. Eval-only; no training, no checkpoint written.
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
from src.eval.run_full import _load_framework

RUN = Path("data/runs/v2.7.4/set__20260720-083533__seed42/v2.7.4__20260720-091830__seed42")
CKPT = RUN / "checkpoints/best.pt"
POOL = EVAL_POOLS_DIR / "eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl"
OUT = Path("data/runs/v2.7.4/theory"); OUT.mkdir(parents=True, exist_ok=True)

fw, cfg, _ = _load_framework(CKPT)
system = fw.system
dt = float(cfg["env"]["dt"]); max_steps = int(cfg["eval"]["max_steps"])
oob_limit = float(cfg["env"]["oob_limit"]); z_floor = -oob_limit
world_lim = float(cfg["env"]["world_lim"]); z_scene = -world_lim

scenes = load_pool(POOL).scenes
goal_z = np.array([float(np.asarray(s.goal, float)[2]) for s in scenes])
start_z = np.array([float(np.asarray(s.start, float)[2]) for s in scenes])

dtype, device = _tensor_options(system, fw)
bs = make_batched_scene(scenes, device=device, dtype=dtype)
res = rollout_eval(system, fw.policy, _filter_adapter(fw), bs, initial_states_from_batch(bs),
                   max_steps=max_steps, dt=dt, config=cfg)
S = res.states.detach().to(torch.float64).numpy()
T1, N = S.shape[0], S.shape[1]
z = S[:, :, 2]

outc, ev = {}, {}
with (RUN / "eval_episodes.csv").open() as f:
    for r in csv.DictReader(f):
        if r["mode"] == "final":
            i = int(r["episode_idx"]); outc[i] = r["outcome"]; ev[i] = int(float(r["n_steps"]))

def t4_of(i):
    k_end = int(np.clip(ev[i], 1, T1 - 1))
    zs = z[:k_end + 1, i]
    b = np.nonzero(zs < z_scene)[0]
    return (int(b[0]), k_end) if b.size else (None, k_end)

floor_eps = [i for i in range(N) if outc.get(i) == "oob" and z[int(np.clip(ev[i], 0, T1 - 1)), i] < z_floor]
goal_eps = [i for i in range(N) if outc.get(i) == "goal"]
goal_enter = [i for i in goal_eps if t4_of(i)[0] is not None]

def q(a_, p): return round(float(np.percentile(a_, p)), 4) if len(a_) else None
def dist(a_):
    a_ = np.asarray([x for x in a_ if x is not None], float)
    return {"n": int(a_.size), "median": q(a_, 50), "iqr": [q(a_, 25), q(a_, 75)]} if a_.size else {"n": 0}

def entry_stats(group):
    d, above = [], 0
    for i in group:
        k4, _ = t4_of(i)
        if k4 is None: continue
        gap = float(goal_z[i] - z[k4, i])            # >0 => goal ABOVE the vehicle at entry
        d.append(gap); above += int(gap > 0)
    return d, above

d_floor, above_floor = entry_stats(floor_eps)
d_ctl, above_ctl = entry_stats(goal_enter)

# tracking check: did the vertical distance to goal ever DECREASE after basement entry (vs its value at t_4)?
track = 0; track_detail = []
for i in floor_eps:
    k4, k_end = t4_of(i)
    if k4 is None: continue
    gap0 = abs(z[k4, i] - goal_z[i])
    post = np.abs(z[k4:k_end + 1, i] - goal_z[i])
    closed = bool(post.min() < gap0 - 1e-9)
    track += int(closed)
    track_detail.append({"ep": i, "gap_at_t4": round(float(gap0), 4),
                         "min_gap_after_t4": round(float(post.min()), 4), "ever_closed": closed})

rep = {
    "artifact_sufficiency": ("goal_z/start_z read directly from the frozen pool; the saved records did NOT "
        "suffice for z(t_4) of the 217 controls (G3 stored aggregates only, no ids) nor for the post-entry "
        "trajectory of the 33 (the M6 run persists actions, not states), so the deterministic canonical "
        "re-roll used by G1/G3 was repeated."),
    "pool": POOL.name, "n_episodes": N, "z_scene_floor": z_scene, "z_arena_floor": z_floor,
    "groups": {
        "floor_exits": {"n": len(floor_eps),
                        "goal_z": dist(goal_z[floor_eps]), "start_z": dist(start_z[floor_eps])},
        "goal_reaching_entering_basement": {"n": len(goal_enter),
                        "goal_z": dist(goal_z[goal_enter]), "start_z": dist(start_z[goal_enter])},
        "all_goal_reaching": {"n": len(goal_eps),
                        "goal_z": dist(goal_z[goal_eps]), "start_z": dist(start_z[goal_eps])}},
    "goal_minus_z_at_entry": {
        "floor_exits": {**dist(d_floor), "n_goal_ABOVE_vehicle": above_floor,
                        "frac_goal_ABOVE_vehicle": round(above_floor / max(len(d_floor), 1), 4)},
        "goal_reaching_entering_basement": {**dist(d_ctl), "n_goal_ABOVE_vehicle": above_ctl,
                        "frac_goal_ABOVE_vehicle": round(above_ctl / max(len(d_ctl), 1), 4)}},
    "floor_exits_still_tracking_after_entry": {
        "n": len(track_detail), "n_vertical_gap_ever_decreased": track,
        "frac": round(track / max(len(track_detail), 1), 4), "per_episode": track_detail},
}
(OUT / "g4_goal_altitude.json").write_text(json.dumps(rep, indent=2) + "\n")
print(json.dumps({k: rep[k] for k in ("groups", "goal_minus_z_at_entry")}, indent=2))
print("tracking:", json.dumps({k: v for k, v in rep["floor_exits_still_tracking_after_entry"].items() if k != "per_episode"}))
