"""v2.8.0 cone_split amendment B — the episodes that recovered level attitude and THEN hit the floor.

M2 counts N_floor floor collisions in the complement (theta>60); M3 counts N_fail recovery failures
caused by the floor, where M3's condition runs only to the leveling instant. The difference is the set
of episodes that regained level attitude and crossed the floor AFTERWARDS (M3 scores them as recoveries).
This script isolates that set (floor AND obstacle channels), re-rolling the given arm's best.pt on the
same complement ICs and the same deployed config + shipped fallback as M3. For each such episode it
records the initial clearance and tilt, the altitude and vertical velocity at the leveling instant, the
step the floor/obstacle was crossed, the floor_feasibility class, and whether the crossing followed
directly from the recovery (levelled with too little altitude, monotone descent) or later after
independent flight (climbed back up, then crossed). These are NOT folded into either the recovery count
or the policy-failure count; they are their own small set.

Usage: --ckpt <best.pt> --projection {dual_solve|enumerate} --tag <name>
Sidecar: data/runs/v2.8.0/cone_postrecovery_<tag>.json"""
from __future__ import annotations
import argparse, copy, json
from pathlib import Path
import numpy as np, torch

from src.eval.build_pools import load_pool
from src.eval.run_full import _load_framework as load_fw
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.common.rk4 import rk4_step
from src.common.outcomes import step_outcomes, resolve_outcome
from src.envs.quadrotor_3d import _quat_to_R

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
POOL = REPO / "data/secured_data/pools/eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BAND, DT, STEPS = 4.0, 0.05, 200
DIRECT_GAP_S = 0.50        # crossing within this of the leveling instant = followed directly from recovery;
                           # a larger gap = the vehicle flew on and crossed later (independent flight).

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--projection", required=True, choices=["dual_solve", "enumerate"])
ap.add_argument("--tag", required=True)
a = ap.parse_args()

ff = json.loads((REPO / "data/runs/v2.8.0/floor_feasibility.json").read_text())
cls = np.array(["unrecoverable" if d["unrecoverable"] else ("recoverable" if d["recoverable"] else "undetermined")
                for d in ff["ic"]])
tilt0_all = np.array([d["tilt_deg"] for d in ff["ic"]])
recov_idx = np.where(tilt0_all > 60.0)[0]                      # the complement (theta>60), global indices

ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
filt = copy.deepcopy(ck["config"]["filter"]); filt["empty_fallback"] = {"mode": "kstep", "phases": 1, "k": 3}
filt["projection"] = a.projection
over = {"env": {"dt": 0.05, "stuck_window_steps": 60, "stuck_radius": 0.10, "band_collision_limit": 4.0,
                "goal_angrate_radius": 0.48}, "eval": {"max_steps": STEPS, "dt_ctrl": 0.05}, "filter": filt}
fw, cfg, _ = load_fw(a.ckpt, config_overrides=over)
for n in ("value_net", "policy_net"):
    m = getattr(fw, n, None); m.to(DEV) if m is not None else None

scenes_all = load_pool(POOL).scenes
scenes = [scenes_all[i] for i in recov_idx]
bs = batch_scenes(scenes, device=DEV, dtype=torch.float32)
x = initial_states_from_batch(bs)
states = [x]
with torch.no_grad():
    for _ in range(STEPS):
        un = fw.policy(x, bs); u = fw.filter(x, un, bs)[0]
        x = fw.system.wrap_state(rk4_step(fw.system, x, u, DT)); states.append(x)
S = torch.stack(states, 0)                                    # [T+1,B,13]
R = _quat_to_R(S[..., 3:7])
tilt = torch.rad2deg(torch.arccos(R[..., 2, 2].clamp(-1, 1))).cpu().numpy()  # [T+1,B]
z = S[..., 2].cpu().numpy(); vz = S[..., 9].cpu().numpy()
masks = step_outcomes(S, bs, fw.system, cfg)
bl = masks.collided_band_lower.cpu().numpy()
bu = masks.collided_band_upper.cpu().numpy()
ob = masks.collided_obstacle.cpu().numpy()
B = z.shape[1]

# t_level[b] = smallest t s.t. tilt stays <=60 from t on; STEPS+1 if never
below = tilt <= 60.0
stays = np.flip(np.logical_and.accumulate(np.flip(below, 0), 0), 0)
t_level = np.full(B, STEPS + 1)
for b in range(B):
    w = np.where(stays[:, b])[0]
    if w.size:
        t_level[b] = w[0]

def first_true(m):
    out = np.full(B, STEPS + 2)
    for b in range(B):
        w = np.where(m[:, b])[0]
        if w.size:
            out[b] = w[0]
    return out
t_floor, t_ceil, t_obst = first_true(bl), first_true(bu), first_true(ob)

# recovered (M3 definition): leveled AND no crossing at or before the leveling instant
no_cross_before = (t_floor > t_level) & (t_ceil > t_level) & (t_obst > t_level)
recovered = (t_level <= STEPS) & no_cross_before

# each recovered episode that later crosses is attributed to its FIRST post-leveling crossing (its
# resolved cause) -- no double-counting, so counts reconcile to M2's resolved band_lower/obstacle.
first_post = np.minimum(np.minimum(t_floor, t_ceil), t_obst)    # earliest crossing overall (> t_level for recovered)
chan_of = np.where(first_post == t_floor, "floor",
           np.where(first_post == t_obst, "obstacle",
           np.where(first_post == t_ceil, "ceiling", "none")))

def post_recovery(chan):
    """recovered episodes whose FIRST post-leveling crossing is `chan` (in (t_level, STEPS])."""
    sel = np.where(recovered & (first_post <= STEPS) & (chan_of == chan))[0]
    rows = []
    for b in sel:
        tl, tc = int(t_level[b]), int(first_post[b])
        seg = z[tl:tc + 1, b]
        rebound = float(seg.max() - z[tl, b])                  # how far it climbed above leveling altitude first
        gap = (tc - tl) * DT
        direct = gap <= DIRECT_GAP_S                           # crossed right after leveling vs flew on
        if direct:
            attr = "direct (crossed right after leveling)" if chan != "floor" else "direct (levelled too low)"
        else:
            attr = "later (independent flight)"
        rows.append({
            "global_idx": int(recov_idx[b]),
            "init_clearance_m": float(z[0, b] + BAND),
            "init_tilt_deg": float(tilt[0, b]),
            "altitude_at_level_m": float(z[tl, b]),
            "vz_at_level_ms": float(vz[tl, b]),
            "t_level_s": float(tl * DT),
            "t_cross_s": float(tc * DT),
            "gap_s": float(gap),
            "z_rebound_after_level_m": rebound,
            "ff_class": str(cls[recov_idx[b]]),
            "attribution": attr,
        })
    return rows

floor_set = post_recovery("floor")
obst_set = post_recovery("obstacle")

rep = {
    "tag": a.tag, "ckpt": str(a.ckpt), "projection": a.projection,
    "n_complement": int(B), "direct_gap_threshold_s": DIRECT_GAP_S,
    "post_recovery_floor": {"n": len(floor_set), "episodes": floor_set},
    "post_recovery_obstacle": {"n": len(obst_set), "episodes": obst_set},
}
# reconciliation (mirror M3): recovery failures caused by floor = floor is earliest crossing AT/BEFORE
# leveling. resolved band_lower in complement = those failures + floor-first post-recovery (nothing crossed
# before leveling for the latter, so floor IS their resolved first-crossing) -> should equal M2's 66.
m3_floor_fail = int(((~recovered) & (t_floor <= np.minimum(t_level, STEPS)) &
                     (t_floor <= t_ceil) & (t_floor <= t_obst)).sum())
m3_obst_fail = int(((~recovered) & (t_obst <= np.minimum(t_level, STEPS)) &
                    (t_obst < t_floor) & (t_obst <= t_ceil)).sum())
rep["reconciliation"] = {
    "M3_floor_recovery_failures_at_or_before_level": m3_floor_fail,
    "post_recovery_floor_n": len(floor_set),
    "resolved_band_lower_complement": m3_floor_fail + len(floor_set),
    "note_band_lower": f"{m3_floor_fail} (M3 floor failures) + {len(floor_set)} (recovered-then-floor) "
                       f"= {m3_floor_fail + len(floor_set)}; compare M2 complement band_lower (66).",
    "M3_obstacle_recovery_failures_at_or_before_level": m3_obst_fail,
    "post_recovery_obstacle_n": len(obst_set),
    "resolved_obstacle_complement": m3_obst_fail + len(obst_set),
}
(REPO / f"data/runs/v2.8.0/cone_postrecovery_{a.tag}.json").write_text(json.dumps(rep, indent=2) + "\n")
print(f"[{a.tag}] complement n={B}  M3-floor-failures={m3_floor_fail}  post-recovery-floor={len(floor_set)}  "
      f"resolved-band_lower={m3_floor_fail+len(floor_set)} (must match this arm's M2 complement floor)  "
      f"post-recovery-obstacle={len(obst_set)}")
for r in floor_set:
    print(f"  FLOOR idx {r['global_idx']}: clr {r['init_clearance_m']:.2f}m tilt0 {r['init_tilt_deg']:.0f} | "
          f"@level z {r['altitude_at_level_m']:.2f} vz {r['vz_at_level_ms']:.2f} t {r['t_level_s']:.2f}s -> "
          f"cross {r['t_cross_s']:.2f}s (gap {r['gap_s']:.2f}s) | {r['ff_class']} | {r['attribution']}")
for r in obst_set:
    print(f"  OBST  idx {r['global_idx']}: clr {r['init_clearance_m']:.2f}m tilt0 {r['init_tilt_deg']:.0f} | "
          f"@level z {r['altitude_at_level_m']:.2f} vz {r['vz_at_level_ms']:.2f} -> cross {r['t_cross_s']:.2f}s | {r['attribution']}")
