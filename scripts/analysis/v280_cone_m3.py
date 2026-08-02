"""v2.8.0 cone_split M3 — recovery on the complement (theta>60), light GPU.

Re-roll the dual arm's adopted checkpoint on the recovery subset under the deployed config + shipped
fallback, logging per-step tilt and z. Recovery = tilt falls to and STAYS <=60deg with NO floor/ceiling
crossing up to that moment. Reported as a CURVE over time (fraction recovered within t), not one cutoff.
Denominators: whole complement AND the floor_feasibility 'recoverable' subset; 'undetermined' ICs held
out of both, reported as their own count. Goal-reach-after-recovery reported separately. Sidecar:
data/runs/v2.8.0/cone_split_m3.json."""
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
_DUAL = REPO / "data/runs/v2.8.0/set__20260731-015517__seed42/v2.8.0__jt__20260731-015517__seed42/checkpoints/best.pt"
_ap = argparse.ArgumentParser()
_ap.add_argument("--ckpt", default=str(_DUAL))
_ap.add_argument("--projection", default="dual_solve", choices=["dual_solve", "enumerate"])
_ap.add_argument("--tag", default="")           # "" -> cone_split_m3.json (dual); else cone_split_m3_<tag>.json
_A = _ap.parse_args()
CK = Path(_A.ckpt)
POOL = REPO / "data/secured_data/pools/eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BAND, DT, STEPS = 4.0, 0.05, 200

ff = json.loads((REPO / "data/runs/v2.8.0/floor_feasibility.json").read_text())
cls = np.array(["unrecoverable" if d["unrecoverable"] else ("recoverable" if d["recoverable"] else "undetermined")
                for d in ff["ic"]])
tilt0 = np.array([d["tilt_deg"] for d in ff["ic"]])
recov_idx = np.where(tilt0 > 60.0)[0]                          # the complement

ck = torch.load(CK, map_location="cpu", weights_only=False)
filt = copy.deepcopy(ck["config"]["filter"]); filt["empty_fallback"] = {"mode": "kstep", "phases": 1, "k": 3}
filt["projection"] = _A.projection
over = {"env": {"dt": 0.05, "stuck_window_steps": 60, "stuck_radius": 0.10, "band_collision_limit": 4.0,
                "goal_angrate_radius": 0.48}, "eval": {"max_steps": STEPS, "dt_ctrl": 0.05}, "filter": filt}
fw, cfg, _ = load_fw(CK, config_overrides=over)
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
tilt = torch.rad2deg(torch.arccos(R[..., 2, 2].clamp(-1, 1)))  # [T+1,B]
masks = step_outcomes(S, bs, fw.system, cfg)
res = resolve_outcome(masks)
goal_reached = np.array([o == "goal" for o in res.outcome])
bl = masks.collided_band_lower.cpu().numpy()                   # [T+1,B]
bu = masks.collided_band_upper.cpu().numpy()
ob = masks.collided_obstacle.cpu().numpy()
tl = tilt.cpu().numpy(); B = tl.shape[1]

# t_level[b] = smallest t s.t. tilt[s,b] <= 60 for all s>=t (stays below); inf if never
below = tl <= 60.0
stays = np.flip(np.logical_and.accumulate(np.flip(below, 0), 0), 0)   # stays[t] = all below from t on
t_level = np.full(B, STEPS + 1)
for b in range(B):
    w = np.where(stays[:, b])[0]
    if w.size:
        t_level[b] = w[0]
# first floor/ceiling/obstacle crossing step per episode
def first_true(m):
    out = np.full(B, STEPS + 2)
    for b in range(B):
        w = np.where(m[:, b])[0]
        if w.size:
            out[b] = w[0]
    return out
t_floor, t_ceil, t_obst = first_true(bl), first_true(bu), first_true(ob)
# recovery: t_level exists AND no crossing at or before t_level
no_cross_before = (t_floor > t_level) & (t_ceil > t_level) & (t_obst > t_level)
recovered = (t_level <= STEPS) & no_cross_before
# failure mode for the rest: earliest crossing before leveling, else never_level
def failure_mode(b):
    if recovered[b]:
        return "recovered"
    cand = {"floor": t_floor[b], "ceiling": t_ceil[b], "obstacle": t_obst[b]}
    cand = {k: v for k, v in cand.items() if v <= min(t_level[b], STEPS)}
    if cand:
        return min(cand, key=cand.get)
    return "never_level"
modes = np.array([failure_mode(b) for b in range(B)])

# denominators
rec_sub = cls[recov_idx] == "recoverable"
und_sub = cls[recov_idx] == "undetermined"
# recovery curve: fraction recovered within t (over whole complement, and over recoverable subset)
ts = np.arange(0, STEPS + 1)
curve_all = [float(((recovered) & (t_level <= t)).mean()) for t in ts]
curve_rec = [float(((recovered & rec_sub) & (t_level <= t)).sum() / max(1, rec_sub.sum())) for t in ts]
tt_level = t_level[recovered] * DT                            # time-to-level among successes (s)

rep = {
    "checkpoint": f"{_A.projection} {CK.name}", "ckpt_path": str(CK), "n_complement": int(B),
    "boundary": "theta>60 (cor:hold-so3, not searched)",
    "recovered_whole_complement": {"n": int(recovered.sum()), "frac": float(recovered.mean())},
    "recovered_over_recoverable_subset": {"n": int((recovered & rec_sub).sum()), "denom": int(rec_sub.sum()),
        "frac": float((recovered & rec_sub).sum() / max(1, rec_sub.sum()))},
    "undetermined_held_out": {"n": int(und_sub.sum()), "recovered_among_them": int((recovered & und_sub).sum())},
    "time_to_level_s": {"median": float(np.median(tt_level)) if tt_level.size else None,
                        "p90": float(np.quantile(tt_level, 0.9)) if tt_level.size else None,
                        "max": float(tt_level.max()) if tt_level.size else None},
    "failure_modes": {k: int((modes == k).sum()) for k in ["recovered", "floor", "ceiling", "obstacle", "never_level"]},
    "recovery_curve_frac_within_t": {"t_s": [float(t * DT) for t in ts[::20]],
        "whole_complement": [curve_all[t] for t in ts[::20]], "recoverable_subset": [curve_rec[t] for t in ts[::20]]},
    "goal_reach_after_recovery": {"n_recovered": int(recovered.sum()),
        "reached_goal": int((recovered & goal_reached).sum()),
        "frac": float((recovered & goal_reached).sum() / max(1, recovered.sum()))},
}
_out = "cone_split_m3.json" if not _A.tag else f"cone_split_m3_{_A.tag}.json"
(REPO / "data/runs/v2.8.0" / _out).write_text(json.dumps(rep, indent=2) + "\n")
print("M3 recovery (%s, complement n=%d):" % (_A.projection, B))
print(f"  recovered whole {rep['recovered_whole_complement']['frac']:.4f} | over recoverable {rep['recovered_over_recoverable_subset']['frac']:.4f} (n {rep['recovered_over_recoverable_subset']['n']}/{rep['recovered_over_recoverable_subset']['denom']}) | undetermined held out {rep['undetermined_held_out']['n']}")
print(f"  failure modes: {rep['failure_modes']}")
print(f"  time-to-level median {rep['time_to_level_s']['median']}s | goal after recovery {rep['goal_reach_after_recovery']['frac']:.4f}")
