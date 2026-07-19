"""v2.7.2 M6 probes for quadrotor_3d (MEASURE ONLY — the k-step empty-branch fallback is NOT enabled).

P1 gravity-direction decodability : linear probe of g^b from the value net's penultimate features (R^2).
P2 tilt strata                    : collision/reach rate by initial tilt band angle(body-up, world-up).
P3 descent anatomy                : reach rate + final altitude error by |goal_z - start_z| band.
P4 ||L_g V_hat|| at collisions    : authority at the states where episodes collide (vs the M3 near-B0 gate).
P5 empty-branch rate              : fraction of active filter steps whose CBF feasible set is empty (blind row).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.common.filter_hardnet import _cbf_terms
from src.common.kstep_fallback import slice_scene
from src.common.outcomes import step_outcomes, resolve_outcome
from src.common.rk4 import rk4_step
from src.common.value_net import make_h_fn
from src.eval.build_pools import load_pool, DEFAULT_OUTPUT_DIR
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.frameworks.jt_pncbf.train import load_framework_from_checkpoint

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--n", type=int, default=2000)
a = ap.parse_args()

fw, cfg, ck = load_framework_from_checkpoint(Path(a.ckpt))
system = fw.system
vnet = fw.value_net if hasattr(fw, "value_net") else fw._value_net
h_fn = make_h_fn(vnet, system)
dt = float(cfg["env"]["dt"]); max_steps = int(cfg["eval"]["max_steps"])

pool = load_pool(DEFAULT_OUTPUT_DIR / "eval_full_quadrotor-3d_n2000_seed23456.pkl")
scenes = pool.scenes[: a.n]
bs = batch_scenes(scenes, device=torch.device("cpu"), dtype=torch.float32)
x = system.wrap_state(initial_states_from_batch(bs).float())
B = x.shape[0]


def _tilt_deg(xq):                                            # angle(body-up, world-up)
    R22 = _R(xq[:, 3:7])[:, 2, 2].clamp(-1, 1)
    return torch.rad2deg(torch.arccos(R22))


def _R(q):
    from src.envs.quadrotor_3d import _quat_to_R
    return _quat_to_R(q)


tilt0 = _tilt_deg(x).numpy()
start_z = x[:, 2].numpy().copy()
goal_z = bs.goal[:, 2].numpy()

# ---- roll policy + filter, collecting states + empty/singular flags ----
states = [x.clone()]
empty_steps = []          # [T] count of empty rows among still-active episodes
singular_steps = []
active = torch.ones(B, dtype=torch.bool)
empty_active_total = 0; singular_active_total = 0; active_step_total = 0
with torch.no_grad():
    for t in range(max_steps):
        un = fw.policy(x, bs)
        u, _ = fw.filter(x, un, bs)
        em = fw._filter.last_empty.bool(); sg = fw._filter.last_singular.bool()
        empty_active_total += int((em & active).sum()); singular_active_total += int((sg & active).sum())
        active_step_total += int(active.sum())
        x = rk4_step(system, x, u, dt)
        states.append(x.clone())
        # mark resolved (collision/goal/oob/stuck) episodes inactive going forward (approx via outcomes below)
S = torch.stack(states, dim=0)                                # [T+1, B, 13]

# ---- outcomes on the rolled trajectory ----
masks = step_outcomes(S, bs, system, cfg)
res = resolve_outcome(masks)
outcome = np.array(res.outcome)
event_step = res.event_step.numpy()
collided = outcome == "collision"
reached = outcome == "goal"

# P5 empty-branch rate (over active filter steps)
p5 = {"empty_branch_rate": round(empty_active_total / max(1, active_step_total), 6),
      "singular_rate": round(singular_active_total / max(1, active_step_total), 6),
      "active_filter_steps": active_step_total}

# P2 tilt strata
bands = [(0, 30), (30, 60), (60, 90), (90, 120)]
p2 = []
for lo, hi in bands:
    m = (tilt0 >= lo) & (tilt0 < hi)
    n = int(m.sum())
    p2.append({"band_deg": f"[{lo},{hi})", "n": n,
               "collision_rate": round(float(collided[m].mean()), 4) if n else None,
               "reach_rate": round(float(reached[m].mean()), 4) if n else None})

# P3 descent anatomy: reach + final altitude error by |dz| band
dz = np.abs(goal_z - start_z)
zfinal_err = np.abs(S[-1, :, 2].numpy() - goal_z)
p3 = []
for lo, hi in [(0, 2), (2, 4), (4, 8)]:
    m = (dz >= lo) & (dz < hi)
    n = int(m.sum())
    p3.append({"dz_band": f"[{lo},{hi})", "n": n,
               "reach_rate": round(float(reached[m].mean()), 4) if n else None,
               "final_altitude_err_med": round(float(np.median(zfinal_err[m])), 4) if n else None})

# P4 ||L_g V_hat|| at collision states (the step just before the collision event)
p4 = {"n_collisions": int(collided.sum())}
if int(collided.sum()) > 0:
    idx = np.nonzero(collided)[0]
    ev = np.clip(event_step[idx], 0, S.shape[0] - 1)
    xc = S[ev, idx]                                          # [Nc, 13] collision-time states
    sub = slice_scene(bs, torch.as_tensor(collided))
    u0 = torch.zeros(xc.shape[0], int(system.action_dim))
    _, _, lg = _cbf_terms(system, h_fn, xc, sub, u0, create_graph=False)
    lgn = torch.linalg.norm(lg, dim=1).detach().numpy()
    p4.update({"lg_min": round(float(lgn.min()), 4), "lg_median": round(float(np.median(lgn)), 4),
               "lg_mean": round(float(lgn.mean()), 4),
               "lg_degen_frac_lt_1e-3": round(float((lgn < 1e-3).mean()), 4)})

# P1 gravity decodability: linear probe g^b <- penultimate features of value member 0 (on the IC states,
# each with its own scene so the observation is well-defined).
feats = {}
member = vnet.members[0]
with torch.no_grad():
    x0 = system.wrap_state(initial_states_from_batch(bs).float())
    obs0 = system.observation(x0, bs)
    hook = member[-2].register_forward_hook(lambda m, i, o: feats.__setitem__("h", o.detach()))
    _ = member(obs0)
    hook.remove()
    H = feats["h"].numpy()                                   # [B, hidden]  penultimate activations
    g_b = obs0[:, 9:12].numpy()                              # target gravity-in-body
# least-squares linear probe with bias, report R^2 per component and mean
Hb = np.concatenate([H, np.ones((H.shape[0], 1))], axis=1)
W, *_ = np.linalg.lstsq(Hb, g_b, rcond=None)
pred = Hb @ W
ss_res = ((g_b - pred) ** 2).sum(axis=0); ss_tot = ((g_b - g_b.mean(0)) ** 2).sum(axis=0)
r2 = 1.0 - ss_res / np.clip(ss_tot, 1e-12, None)
p1 = {"gravity_decode_R2_xyz": [round(float(v), 4) for v in r2], "gravity_decode_R2_mean": round(float(r2.mean()), 4)}

print(json.dumps({"ckpt": a.ckpt, "n": B,
                  "P1_gravity_decodability": p1, "P2_tilt_strata": p2,
                  "P3_descent_anatomy": p3, "P4_lg_at_collisions": p4,
                  "P5_empty_branch": p5}, indent=2))
