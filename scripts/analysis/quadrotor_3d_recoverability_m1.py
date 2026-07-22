"""v2.7.4 M1 — full-SO(3) recoverability calibration for one IC menu item (G1). Measurement only.

For a menu item (ic_omega_max, ic_v_max) with ic_so3=on and the frozen d2r cylinder geometry, at the IN-LOOP
seed 12345, n scenes:
  (1) acceptance: rejection rate of the scene/IC acceptance loop (clearance + unavoidable-collision;
      the v2.6.2 recoverability filter is planar-only, no-op here), tilt histogram BEFORE vs AFTER acceptance,
      and the realized post-clamp |omega| (state bound omega_max clamps IC omega in wrap_state).
  (2) physical recoverability, CLOSED FORM (no rollout): from tilt theta, rate |omega|, speed |v0|,
      bang-bang time-to-upright t = 2*sqrt(theta/alpha) + |omega|/alpha with alpha = tau_roll_max/Jx = 117.9
      rad/s^2; free fall 0.5*g*t^2 must fit the altitude above the region floor, lateral drift |v0|*t must fit
      the clearance to the nearest active cylinder. Fraction recoverable, overall and by tilt stratum.
  (3) nominal-only failure: cascaded hover-PD (system.lqr_action), UNFILTERED, rolled to eval max_steps;
      collision/stuck/oob/timeout/cps.
Writes one JSON under --run-dir (measurement persistence). Strata: [0,60),[60,120),[120,180] deg.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.common.rk4 import rk4_step
from src.common.outcomes import step_outcomes, resolve_outcome
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.envs.quadrotor_3d import _quat_to_R
from src.envs import scene_init as SI
from src.frameworks.jt_pncbf.train import load_effective_config, make_system

ap = argparse.ArgumentParser()
ap.add_argument("--omega-max", type=float, required=True)
ap.add_argument("--v-max", type=float, required=True)
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--seed", type=int, default=12345)
ap.add_argument("--run-dir", required=True)
ap.add_argument("--label", required=True)
a = ap.parse_args()

ALPHA = 117.9          # rad/s^2 = tau_roll_max (1.1792) / Jx (0.01)
STRATA = [(0.0, 60.0), (60.0, 120.0), (120.0, 180.0001)]

cfg = load_effective_config()
cfg["run"]["system"] = "quadrotor_3d"
q3 = cfg["env"]["quadrotor_3d"]
q3["ic_so3"] = True
q3["ic_omega_max"] = float(a.omega_max)
q3["ic_v_max"] = float(a.v_max)
system = make_system(cfg)
g = float(q3["gravity"]); world_lim = float(cfg["env"]["world_lim"])
omega_bound = float(cfg["env"]["bounds"]["quadrotor_3d"]["omega_max"])
dt = float(cfg["env"]["dt"]); max_steps = int(cfg["eval"]["max_steps"])
params = SI._SceneModeParams(mode="eval",
                             min_start_goal_dist=float(cfg["eval"]["scene"]["min_start_goal_dist"]),
                             start_goal_clearance=float(cfg["eval"]["scene"]["start_goal_clearance"]))


def _tilt_deg_from_quat(quat):
    q = torch.tensor(np.asarray(quat)[None, :], dtype=torch.float64)
    R = _quat_to_R(q)
    return float(np.degrees(np.arccos(np.clip(float(R[0, 2, 2]), -1.0, 1.0))))


# ---- sample n accepted scenes, instrumenting rejection + raw/accepted tilt (replicates _sample_scene loop)
rng = np.random.default_rng(a.seed)
accepted = []
raw_tilts, accepted_tilts = [], []
n_attempts = 0
while len(accepted) < a.n:
    for _ in range(SI._MAX_RETRIES):
        n_attempts += 1
        start, goal = SI._sample_start_goal(rng, cfg, params)
        centers, radii, active = SI._sample_obstacles(rng, cfg, start, goal, "quadrotor_3d")
        scene = SI._make_scene(rng, cfg, "quadrotor_3d", params.mode, centers, radii, active, start, goal)
        raw_tilts.append(_tilt_deg_from_quat(scene.initial_attitude_quat))
        if not SI._has_start_goal_clearance(scene, params.start_goal_clearance):
            continue
        if not SI._passes_unavoidable_collision_filter(scene, cfg, params):
            continue
        if not SI._passes_recoverability_filter(scene, cfg):
            continue
        accepted.append(scene)
        accepted_tilts.append(raw_tilts[-1])
        break
raw_tilts = np.array(raw_tilts); accepted_tilts = np.array(accepted_tilts)
rejection_rate = 1.0 - len(accepted) / n_attempts


def _hist(t):
    h, _ = np.histogram(t, bins=[0, 30, 60, 90, 120, 150, 180])
    return [int(x) for x in h]


# ---- realized post-clamp |omega| (wrap_state clamps per-axis to +-omega_bound) ----
omega_raw = np.array([np.linalg.norm(np.asarray(s.initial_omega_vec)) for s in accepted])
omega_clamped = np.array([np.linalg.norm(np.clip(np.asarray(s.initial_omega_vec), -omega_bound, omega_bound))
                          for s in accepted])

# ---- (2) closed-form recoverability ----
tilts = np.array([_tilt_deg_from_quat(s.initial_attitude_quat) for s in accepted])
recov = np.zeros(len(accepted), dtype=bool)
for i, s in enumerate(accepted):
    theta = np.radians(tilts[i])
    omega = float(np.linalg.norm(np.clip(np.asarray(s.initial_omega_vec), -omega_bound, omega_bound)))
    v0 = float(np.linalg.norm(np.asarray(s.initial_velocity))) if s.initial_velocity is not None else 0.0
    t = 2.0 * np.sqrt(max(theta, 0.0) / ALPHA) + omega / ALPHA
    fall = 0.5 * g * t * t
    drift = v0 * t
    p = np.asarray(s.start, float)
    floor_clear = float(p[2] + world_lim)                       # altitude above region floor (-world_lim)
    cen = np.asarray(s.obstacle_centers, float); rad = np.asarray(s.obstacle_radii, float)
    act = np.asarray(s.obstacle_active, bool)
    if act.any():
        cyl_clear = float(np.min(np.linalg.norm(cen[act][:, :2] - p[:2], axis=1) - rad[act]))
    else:
        cyl_clear = np.inf
    recov[i] = (fall <= floor_clear) and (drift <= cyl_clear)


def _by_stratum(mask_vals, tilts_arr):
    out = []
    for lo, hi in STRATA:
        sel = (tilts_arr >= lo) & (tilts_arr < hi)
        n = int(sel.sum())
        frac = float(mask_vals[sel].mean()) if n else None
        out.append({"stratum_deg": f"[{int(lo)},{int(min(hi,180))})", "n": n, "recoverable_frac": frac})
    return out


# ---- (3) nominal-only unfiltered rollout ----
bs = batch_scenes(accepted, device=torch.device("cpu"), dtype=torch.float32)
x = system.wrap_state(initial_states_from_batch(bs).float())
goal_t = torch.as_tensor(np.stack([np.asarray(s.goal) for s in accepted]), dtype=torch.float32)
states = [x]
with torch.no_grad():
    for _ in range(max_steps):
        u = system.lqr_action(x, goal_t)
        x = rk4_step(system, x, u, dt)
        states.append(x)
S = torch.stack(states, dim=0)
masks = step_outcomes(S, bs, system, cfg)
res_out = resolve_outcome(masks)
outc = np.array(res_out.outcome)
n = len(accepted)
reach = float((outc == "goal").mean()); collision = float((outc == "collision").mean())
oob = float((outc == "oob").mean()); stuck = float((outc == "stuck").mean()); timeout = float((outc == "timeout").mean())
cps = reach - 2 * collision - stuck - 0.5 * (oob + timeout)

result = {
    "label": a.label, "omega_max": a.omega_max, "v_max": a.v_max, "n": n, "seed": a.seed,
    "acceptance": {
        "n_attempts": n_attempts, "n_accepted": n, "rejection_rate": round(rejection_rate, 4),
        "tilt_hist_bins_deg": [0, 30, 60, 90, 120, 150, 180],
        "tilt_hist_raw": _hist(raw_tilts), "tilt_hist_accepted": _hist(accepted_tilts),
        "tilt_mean_raw": round(float(raw_tilts.mean()), 2), "tilt_mean_accepted": round(float(accepted_tilts.mean()), 2),
        "axis_name_false_flag": bool(rejection_rate > 0.50),
    },
    "omega_realized": {
        "ic_omega_max": a.omega_max, "state_omega_bound": omega_bound,
        "mean_raw": round(float(omega_raw.mean()), 3), "mean_post_clamp": round(float(omega_clamped.mean()), 3),
        "frac_clamped": round(float((omega_raw > omega_bound * np.sqrt(3) + 1e-9).mean()), 3),
        "frac_any_axis_clamped": round(float(np.mean([bool(np.any(np.abs(np.asarray(s.initial_omega_vec)) > omega_bound))
                                                      for s in accepted])), 3),
    },
    "recoverability": {
        "alpha": ALPHA, "overall_frac": round(float(recov.mean()), 4),
        "by_stratum": _by_stratum(recov.astype(float), tilts),
    },
    "nominal_only": {"reach": round(reach, 4), "collision": round(collision, 4), "oob": round(oob, 4),
                     "stuck": round(stuck, 4), "timeout": round(timeout, 4), "cps": round(cps, 4)},
}
run_dir = Path(a.run_dir); run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / f"{a.label}.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
