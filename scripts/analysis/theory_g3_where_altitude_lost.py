"""v2.7.4 theory G3 — is the altitude loss inside the scene region, or only in the arena basement?

Registered prediction (BEFORE data): the majority of the margin is lost ABOVE the scene-region floor — median
of (m(0) - m(t_4)) / (m(0) - m(t_commit)) exceeds 0.5 — and at t_4 most of the 33 are already above the
60-degree altitude-holding limit (cos(tilt) < 1/TWR). Falsified if the majority of the margin is lost below
-4.0, or if most episodes cross -4.0 still able to hold altitude.

t_4 = first time z falls below the scene-region floor (-world_lim as read). Margin m uses G1's closed form
verbatim. Constants are READ FROM CODE and printed. Eval-only, same deterministic canonical re-roll as G1.
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
from src.eval.run_full import _load_framework

RUN = Path("data/runs/v2.7.4/set__20260720-083533__seed42/v2.7.4__20260720-091830__seed42")
CKPT = RUN / "checkpoints/best.pt"
POOL = EVAL_POOLS_DIR / "eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl"
OUT = Path("data/runs/v2.7.4/theory"); OUT.mkdir(parents=True, exist_ok=True)

fw, cfg, _ = _load_framework(CKPT)
system = fw.system
dt = float(cfg["env"]["dt"]); max_steps = int(cfg["eval"]["max_steps"])
oob_limit = float(cfg["env"]["oob_limit"]); z_floor = -oob_limit
world_lim = float(cfg["env"]["world_lim"]); z_scene = -world_lim      # scene-region floor
g = float(system.gravity); mass = float(system.mass); f_max = float(system.f_rotor_max)
TWR = (4 * f_max) / (mass * g); a_up = (TWR - 1.0) * g
alpha = (2.0 * float(system.moment_arm) * f_max) / float(system.inertia[0])
tilt_hold_limit = float(np.degrees(np.arccos(min(1.0, 1.0 / TWR))))   # cos(tilt) = 1/TWR
CONST = {"oob_limit": oob_limit, "arena_floor_z": z_floor,
         "world_lim_scene_region": world_lim, "scene_region_floor_z": z_scene,
         "scene_sampler": "start_z, goal_z ~ U(-world_lim, +world_lim) (src/envs/scene_init.py)",
         "unpenalized_band_m": round(oob_limit - world_lim, 4),
         "gravity": g, "mass": mass, "thrust_to_weight": round(TWR, 6),
         "a_up": round(a_up, 6), "alpha_rad_s2": round(alpha, 4),
         "altitude_holding_tilt_limit_deg": round(tilt_hold_limit, 4)}
print(json.dumps(CONST, indent=2), flush=True)

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

tilt = np.degrees(np.arccos(np.clip(
    _quat_to_R(torch.tensor(S[:, :, 3:7].reshape(-1, 4)))[:, 2, 2].numpy(), -1, 1))).reshape(T1, N)
z = S[:, :, 2]; vz = S[:, :, 9]
th = np.radians(np.maximum(tilt, 0.0)); t_rot = 2.0 * np.sqrt(th / alpha)
M = (z - z_floor) - (np.abs(vz) * t_rot + 0.5 * g * t_rot ** 2 + vz ** 2 / (2.0 * a_up))

floor_eps = [i for i in range(N) if outc.get(i) == "oob" and z[int(np.clip(ev[i], 0, T1 - 1)), i] < z_floor]

def q(a_, p): return round(float(np.percentile(a_, p)), 4) if len(a_) else None
def dist(a_):
    a_ = np.asarray([x for x in a_ if x is not None], float)
    return {"n": int(a_.size), "median": q(a_, 50), "iqr": [q(a_, 25), q(a_, 75)]} if a_.size else {"n": 0}

rows, no_t4 = [], 0
for i in floor_eps:
    k_end = int(np.clip(ev[i], 1, T1 - 1))
    zs = z[:k_end + 1, i]
    below = np.nonzero(zs < z_scene)[0]
    if below.size == 0:
        no_t4 += 1
        rows.append({"ep": i, "t4_exists": False}); continue
    k4 = int(below[0])
    ms = M[:k_end + 1, i]
    neg = np.nonzero(ms < 0)[0]
    kc = int(neg[0]) if neg.size else -1
    m0, m4 = float(ms[0]), float(ms[k4])
    mc = float(ms[kc]) if kc >= 0 else None
    above = m0 - m4                                   # margin spent above the scene-region floor
    below_amt = (m4 - mc) if kc >= 0 else None        # margin spent below it, up to commitment
    total = (m0 - mc) if kc >= 0 else None
    rows.append({
        "ep": i, "t4_exists": True, "t4_step": k4, "t4_time_s": round(k4 * dt, 3),
        "z_at_t4": round(float(z[k4, i]), 4), "vz_at_t4": round(float(vz[k4, i]), 4),
        "tilt_at_t4": round(float(tilt[k4, i]), 3), "m_at_t4": round(m4, 4), "m0": round(m0, 4),
        "commit_step": kc, "commit_time_s": (round(kc * dt, 3) if kc >= 0 else None),
        "m_at_commit": (round(mc, 4) if mc is not None else None),
        "margin_above": round(above, 4),
        "margin_below": (round(below_amt, 4) if below_amt is not None else None),
        "margin_total": (round(total, 4) if total is not None else None),
        "frac_above": (round(above / total, 4) if total not in (None, 0) else None),
        "committed_before_t4": (bool(kc >= 0 and kc < k4)),
        "descending_gt_1ms": bool(vz[k4, i] < -1.0),
        "above_hold_limit": bool(tilt[k4, i] > tilt_hold_limit)})

R = [r for r in rows if r.get("t4_exists")]
fracs = [r["frac_above"] for r in R if r["frac_above"] is not None]
committed_before = sum(1 for r in R if r["committed_before_t4"])
rep = {
    "constants_read_from_code": CONST, "pool": POOL.name, "n_episodes": N,
    "n_floor_exits": len(floor_eps), "n_without_t4_never_entered_band": no_t4,
    "state_at_t4": {"elapsed_s": dist([r["t4_time_s"] for r in R]), "z": dist([r["z_at_t4"] for r in R]),
                    "vz": dist([r["vz_at_t4"] for r in R]), "tilt_deg": dist([r["tilt_at_t4"] for r in R]),
                    "margin_m_at_t4": dist([r["m_at_t4"] for r in R])},
    "margin_split": {
        "margin_above_scene_floor": dist([r["margin_above"] for r in R]),
        "margin_below_scene_floor": dist([r["margin_below"] for r in R]),
        "margin_total_to_commit": dist([r["margin_total"] for r in R]),
        "fraction_above_median": (round(float(np.median(fracs)), 4) if fracs else None),
        "fraction_above_iqr": ([q(np.array(fracs), 25), q(np.array(fracs), 75)] if fracs else None),
        "n_committed_BEFORE_t4_inside_scene_region": committed_before,
        "n_committed_after_or_at_t4": len(R) - committed_before},
    "descent_established_at_t4": {
        "frac_descending_faster_than_1ms": round(float(np.mean([r["descending_gt_1ms"] for r in R])), 4) if R else None,
        "n_descending_faster_than_1ms": int(sum(r["descending_gt_1ms"] for r in R)),
        "frac_above_altitude_holding_limit": round(float(np.mean([r["above_hold_limit"] for r in R])), 4) if R else None,
        "n_above_altitude_holding_limit": int(sum(r["above_hold_limit"] for r in R)),
        "limit_deg": round(tilt_hold_limit, 3)},
    "per_episode": rows,
}

# ---- control group: non-failing (goal) episodes that enter the band ----
goal_eps = [i for i in range(N) if outc.get(i) == "goal"]
ctl = []
for i in goal_eps:
    k_end = int(np.clip(ev[i], 1, T1 - 1))
    zs = z[:k_end + 1, i]
    below = np.nonzero(zs < z_scene)[0]
    if below.size == 0: continue
    k4 = int(below[0])
    ctl.append({"t4_s": k4 * dt, "vz": float(vz[k4, i]), "tilt": float(tilt[k4, i]),
                "min_z": float(zs.min()), "depth_below": float(z_scene - zs.min())})
rep["control_group_non_failing_goal"] = {
    "n_goal_episodes": len(goal_eps),
    "n_entering_band_below_scene_floor": len(ctl),
    "frac_of_goal_entering_band": round(len(ctl) / max(len(goal_eps), 1), 5),
    "t4_elapsed_s": dist([c["t4_s"] for c in ctl]),
    "vz_at_crossing": dist([c["vz"] for c in ctl]),
    "tilt_at_crossing_deg": dist([c["tilt"] for c in ctl]),
    "max_depth_below_scene_floor_m": dist([c["depth_below"] for c in ctl]),
    "min_z_reached": dist([c["min_z"] for c in ctl]),
    "frac_descending_faster_than_1ms_at_crossing": round(float(np.mean([c["vz"] < -1.0 for c in ctl])), 4) if ctl else None,
    "frac_above_hold_limit_at_crossing": round(float(np.mean([c["tilt"] > tilt_hold_limit for c in ctl])), 4) if ctl else None}

(OUT / "g3_where_altitude_lost.json").write_text(json.dumps(rep, indent=2) + "\n")
print(json.dumps({k: rep[k] for k in ("n_floor_exits", "n_without_t4_never_entered_band", "state_at_t4",
                                      "margin_split", "descent_established_at_t4",
                                      "control_group_non_failing_goal")}, indent=2))
