"""v2.7.4 theory G1 — floor exits: doomed at the IC, or margin lost in flight?

Registered prediction (BEFORE data): m(0) is lower for the floor-exit episodes than for non-failing episodes,
with the medians separated. Falsified if the two distributions overlap without separation.

Recoverability margin (as registered):
    t_rot  = 2*sqrt(theta/alpha)                     bang-bang time to upright
    d_stop = |v_z|*t_rot + 0.5*g*t_rot^2 + v_z^2/(2*a_up)
    m      = (z - z_floor) - d_stop
The three d_stop terms are reported separately so the dominant one is visible. Constants are READ FROM CODE
(see the printed header), not assumed.

Method: the frozen d2r full pool re-rolled through the canonical eval path with the M6 checkpoint/config
(batch 2000, dt 0.05, max_steps 200) — the deterministic path that reproduces the M6 headline. Outcomes from
the run's own eval_episodes.csv. Eval-only.
"""
from __future__ import annotations

import csv, json, math
from pathlib import Path

import numpy as np
import torch

from src.common.quadrotor_barrier import make_barrier_fn
from src.common.value_net import make_h_fn
from src.eval.build_pools import load_pool, EVAL_POOLS_DIR
from src.eval.evaluate import _filter_adapter, _tensor_options, make_batched_scene
from src.eval.rollout import rollout_eval
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.envs.quadrotor_3d import _quat_to_R
from src.eval.run_full import _load_framework

RUN = Path("data/runs/v2.7.4/set__20260720-083533__seed42/v2.7.4__20260720-091830__seed42")
CKPT = RUN / "checkpoints/best.pt"
POOL = EVAL_POOLS_DIR / "eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl"
OUT = Path("data/runs/v2.7.4/theory"); OUT.mkdir(parents=True, exist_ok=True)

fw, cfg, _ = _load_framework(CKPT)
system = fw.system
dt = float(cfg["env"]["dt"]); max_steps = int(cfg["eval"]["max_steps"])
# ---- constants READ FROM CODE ----
oob_limit = float(cfg["env"]["oob_limit"]); z_floor = -oob_limit
g = float(system.gravity); mass = float(system.mass)
f_max = float(system.f_rotor_max); n_rot = 4
TWR = (n_rot * f_max) / (mass * g)
a_up = (TWR - 1.0) * g
l_arm = float(system.moment_arm); Jx = float(system.inertia[0])
tau_max_roll = 2.0 * l_arm * f_max
alpha = tau_max_roll / Jx
CONST = {"oob_limit": oob_limit, "oob_rule": "any(|p_axis| > oob_limit) -> floor at z = -oob_limit",
         "z_floor": z_floor, "gravity": g, "mass": mass, "f_rotor_max": f_max,
         "thrust_to_weight": round(TWR, 6), "a_up_max_net_upward": round(a_up, 6),
         "moment_arm_l": round(l_arm, 6), "Jx": Jx,
         "max_roll_pitch_torque": round(tau_max_roll, 6),
         "alpha_roll_pitch_rad_s2": round(alpha, 4)}
print(json.dumps(CONST, indent=2), flush=True)

scenes = load_pool(POOL).scenes
dtype, device = _tensor_options(system, fw)
bs = make_batched_scene(scenes, device=device, dtype=dtype)
res = rollout_eval(system, fw.policy, _filter_adapter(fw), bs, initial_states_from_batch(bs),
                   max_steps=max_steps, dt=dt, config=cfg)
S = res.states.detach().to(torch.float64).numpy()          # [T+1, N, 13]
T1, N = S.shape[0], S.shape[1]

outc, ev = {}, {}
with (RUN / "eval_episodes.csv").open() as f:
    for r in csv.DictReader(f):
        if r["mode"] == "final":
            i = int(r["episode_idx"]); outc[i] = r["outcome"]; ev[i] = int(float(r["n_steps"]))

Rall = _quat_to_R(torch.tensor(S[:, :, 3:7].reshape(-1, 4)))
tilt = np.degrees(np.arccos(np.clip(Rall[:, 2, 2].numpy(), -1, 1))).reshape(T1, N)   # [T+1,N] instantaneous
z = S[:, :, 2]; vz = S[:, :, 9]

def margin(theta_deg, vz_, z_):
    th = np.radians(np.maximum(theta_deg, 0.0))
    t_rot = 2.0 * np.sqrt(th / alpha)
    t1 = np.abs(vz_) * t_rot
    t2 = 0.5 * g * t_rot ** 2
    t3 = vz_ ** 2 / (2.0 * a_up)
    d_stop = t1 + t2 + t3
    return (z_ - z_floor) - d_stop, t1, t2, t3, d_stop, t_rot

M_all, T1_, T2_, T3_, D_, TR_ = margin(tilt, vz, z)

# floor-exit episodes: recorded oob AND final z below the floor
floor_eps = [i for i in range(N) if outc.get(i) == "oob" and z[int(np.clip(ev[i], 0, T1 - 1)), i] < z_floor]
h_star_fn = make_barrier_fn(float(cfg["env"][system.name]["c_gain"]), float(cfg["env"]["h_scale"]))
v_fn = make_h_fn(fw.value_net, system)

rows = []
for i in floor_eps:
    k_end = int(np.clip(ev[i], 1, T1 - 1))
    m_series = M_all[:k_end + 1, i]
    m0 = float(m_series[0])
    neg = np.nonzero(m_series < 0)[0]
    kx = int(neg[0]) if neg.size else -1
    rec = {"ep": i, "m0": round(m0, 4), "doomed_at_IC": bool(m0 < 0),
           "d_stop0": round(float(D_[0, i]), 4), "term_vz_trot": round(float(T1_[0, i]), 4),
           "term_half_g_trot2": round(float(T2_[0, i]), 4), "term_vz2_2aup": round(float(T3_[0, i]), 4),
           "t_rot0": round(float(TR_[0, i]), 4), "tilt0": round(float(tilt[0, i]), 2),
           "z0": round(float(z[0, i]), 3), "vz0": round(float(vz[0, i]), 3),
           "crossing_step": kx, "crossing_time_s": (round(kx * dt, 3) if kx >= 0 else None),
           "tilt_at_crossing": (round(float(tilt[kx, i]), 2) if kx > 0 else None),
           "z_at_crossing": (round(float(z[kx, i]), 3) if kx >= 0 else None),
           "vz_at_crossing": (round(float(vz[kx, i]), 3) if kx >= 0 else None)}
    if kx >= 0:
        X = torch.tensor(S[kx:kx + 1, i], dtype=torch.float32)
        b1 = batch_scenes([scenes[i]], device=torch.device("cpu"), dtype=torch.float32)
        with torch.no_grad():
            rec["h_star_at_crossing"] = round(float(h_star_fn(X, b1).reshape(-1)[0]), 5)
            rec["V_hat_at_crossing"] = round(float(v_fn(X, b1).reshape(-1)[0]), 5)
    rows.append(rec)

doomed = [r for r in rows if r["doomed_at_IC"]]
lost = [r for r in rows if not r["doomed_at_IC"]]

def q(a_, p): return round(float(np.percentile(a_, p)), 4) if len(a_) else None
def dist(a_):
    a_ = np.asarray(a_, float)
    return {"n": int(a_.size), "median": q(a_, 50), "iqr": [q(a_, 25), q(a_, 75)],
            "min": q(a_, 0), "max": q(a_, 100)} if a_.size else {"n": 0}

# control group: m(0) for ALL episodes; non-failing = recorded outcome 'goal'
m0_all = M_all[0, :]
strat = lambda t: "[0,60)" if t < 60 else ("[60,120)" if t < 120 else "[120,180]")
tilt0 = tilt[0, :]
control = {}
for s in ("[0,60)", "[60,120)", "[120,180]"):
    idx = [i for i in range(N) if strat(tilt0[i]) == s and outc.get(i) == "goal"]
    control[s] = dist(m0_all[idx])
control["ALL_non_failing_goal"] = dist(m0_all[[i for i in range(N) if outc.get(i) == "goal"]])
floor_m0 = [r["m0"] for r in rows]

report = {
    "constants_read_from_code": CONST,
    "pool": POOL.name, "n_episodes": N, "n_floor_exits": len(rows),
    "margin_definition": "m = (z - z_floor) - [ |v_z|*t_rot + 0.5*g*t_rot^2 + v_z^2/(2*a_up) ], t_rot = 2*sqrt(theta/alpha)",
    "split": {"doomed_at_IC_m0_lt_0": len(doomed), "margin_lost_m0_gt_0": len(lost)},
    "floor_exit_m0": dist(floor_m0),
    "margin_lost_crossing": {
        "elapsed_time_s": dist([r["crossing_time_s"] for r in lost if r["crossing_time_s"] is not None]),
        "tilt_deg_at_crossing": dist([r["tilt_at_crossing"] for r in lost if r["tilt_at_crossing"] is not None]),
        "h_star_at_crossing": dist([r["h_star_at_crossing"] for r in lost if "h_star_at_crossing" in r]),
        "V_hat_at_crossing": dist([r["V_hat_at_crossing"] for r in lost if "V_hat_at_crossing" in r]),
    },
    "d_stop_terms_at_IC_over_floor_exits": {
        "term_vz_trot": dist([r["term_vz_trot"] for r in rows]),
        "term_half_g_trot2": dist([r["term_half_g_trot2"] for r in rows]),
        "term_vz2_2aup": dist([r["term_vz2_2aup"] for r in rows]),
        "d_stop_total": dist([r["d_stop0"] for r in rows]),
    },
    "control_group_m0_non_failing_by_stratum": control,
    "per_episode": rows,
}
(OUT / "g1_floor_margin.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: report[k] for k in
                  ("n_floor_exits", "split", "floor_exit_m0", "margin_lost_crossing",
                   "d_stop_terms_at_IC_over_floor_exits", "control_group_m0_non_failing_by_stratum")}, indent=2))
