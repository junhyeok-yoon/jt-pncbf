"""v2.7.4 theory G2 — which input channel carries L_g V_hat, and does it degenerate at tilt 90 deg?

Registered predictions (BEFORE data):
 (i) If L_g V_hat inherits h_star's thrust-channel structure, total authority peaks near tilt 90 and the
     thrust share is high throughout. Falsified if authority is flat in tilt, or peaks away from 90, or the
     thrust share is small.
 (ii) The torque channels carry the majority of the authority. Falsified if the thrust share exceeds one half
     in the median bin.

Sampling rule reused VERBATIM from the M3 authority gate (scripts/analysis/quadrotor_3d_m3_gate.py): same
checkpoint's V_hat (M3 value best.pt), same d2r in-loop pool, same boundary band |V_hat| < 0.15, same fixed
nominal rollout (system.lqr_action) over eval.max_steps, same _cbf_terms path, same box_halfwidth 2.4525 for
the scalar authority A = sum_i |L_g V_hat_i| * halfwidth and drift D = L_f V_hat. ONLY THE REPORTING CHANGES:
10-degree tilt bins instead of three coarse strata, plus the per-channel decomposition in wrench coordinates.

Wrench decomposition: with the fixed mixer M (rotor forces -> (f_thr, tau_x, tau_y, tau_z)), the wrench-space
row is L_g V_hat @ M^{-1}. Channel authority = |component| * (that channel's available range), where the range
is the image of the per-rotor box [0, f_rotor_max]^4 under that mixer row = f_max * sum|row entries| — so
thrust and torque are compared on a common footing rather than by raw gradient size. Eval-only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.common.filter_hardnet import _cbf_terms
from src.common.kstep_fallback import slice_scene
from src.common.rk4 import rk4_step
from src.common.value_net import ValueNetEnsemble, make_h_fn
from src.eval.build_pools import load_pool, pool_stem, pool_variant, resolve_pool_or_raise
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.envs.quadrotor_3d import _quat_to_R
from src.frameworks.jt_pncbf.train import make_system

M3 = Path("data/runs/v2.7.4/set__20260720-083533__seed42/v2.7.4__20260720-083533__seed42/checkpoints/best.pt")
OUT = Path("data/runs/v2.7.4/theory"); OUT.mkdir(parents=True, exist_ok=True)
BAND = 0.15
HALF = 2.4525                      # 4.905 / 2, verbatim from the M3 gate

ck = torch.load(M3, map_location="cpu", weights_only=False)
cfg = ck["config"]
system = make_system(cfg)
vnet = ValueNetEnsemble(system.obs_dim, cfg); vnet.load_state_dict(ck["v_s_state"]); vnet.eval()
for p in vnet.parameters():
    p.requires_grad_(False)
h_fn = make_h_fn(vnet, system)

variant = pool_variant(cfg, "quadrotor_3d")
stem = pool_stem("inloop", "quadrotor_3d", int(cfg["eval"]["in_loop"]["n"]),
                 int(cfg["eval"]["in_loop"]["seed"]), "random", variant)
pool = load_pool(resolve_pool_or_raise(stem))
bs = batch_scenes(pool.scenes, device=torch.device("cpu"), dtype=torch.float32)
x = system.wrap_state(initial_states_from_batch(bs).float())
goal = bs.goal
dt = float(cfg["env"]["dt"]); max_steps = int(cfg["eval"]["max_steps"])
u0 = torch.zeros(x.shape[0], int(system.action_dim))

Minv = system.mixer_inv.to(torch.float64)                       # wrench -> rotor ; row @ Minv = wrench-space row
f_max = float(system.f_rotor_max)
rng = (f_max * system.mixer.abs().sum(dim=1)).to(torch.float64).numpy()   # available range per wrench channel
CH = ["f_thr", "tau_x", "tau_y", "tau_z"]
print("channel ranges (image of the per-rotor box):", dict(zip(CH, [round(float(v), 5) for v in rng])), flush=True)

quats = np.stack([np.asarray(s.initial_attitude_quat) for s in pool.scenes])
init_tilt = np.degrees(np.arccos(np.clip(_quat_to_R(torch.tensor(quats, dtype=torch.float64))[:, 2, 2].numpy(), -1, 1)))
ep_index = torch.arange(x.shape[0])

A_all, D_all, W_all, tI_all, tE_all = [], [], [], [], []
for t in range(max_steps):
    with torch.no_grad():
        un = system.lqr_action(x, goal)
        h = h_fn(x, bs).reshape(-1)
    near = (h.abs() < BAND)
    if bool(near.any()):
        xin = x[near].clone()
        _h, lf, lg = _cbf_terms(system, h_fn, xin, slice_scene(bs, near), u0[: xin.shape[0]], create_graph=False)
        lg64 = lg.detach().to(torch.float64)
        A_all.append((lg.abs().sum(dim=1) * HALF).detach().numpy())
        D_all.append(lf.detach().numpy())
        W_all.append((lg64 @ Minv).numpy())                     # wrench-space row
        tI_all.append(init_tilt[ep_index[near].numpy()])        # episode INITIAL tilt (M3-gate convention)
        R = _quat_to_R(xin[:, 3:7].to(torch.float64))
        tE_all.append(np.degrees(np.arccos(np.clip(R[:, 2, 2].numpy(), -1, 1))))   # INSTANTANEOUS tilt
    with torch.no_grad():
        x = rk4_step(system, x, un, dt)

A = np.concatenate(A_all); D = np.concatenate(D_all); W = np.concatenate(W_all, axis=0)
tilt_init = np.concatenate(tI_all); tilt_inst = np.concatenate(tE_all)
chan_auth = np.abs(W) * rng[None, :]                            # [n,4] common-footing channel authority
tot = chan_auth.sum(axis=1)
thrust_share = chan_auth[:, 0] / np.maximum(tot, 1e-12)
print(f"n_boundary_states = {A.size} (M3 gate recorded 4372)", flush=True)

def bins(tilt_vec, label):
    out = []
    for lo in range(0, 180, 10):
        m = (tilt_vec >= lo) & (tilt_vec < lo + 10 + (0.001 if lo == 170 else 0))
        n = int(m.sum())
        if n == 0:
            out.append({"bin_deg": f"[{lo},{lo+10})", "n": 0}); continue
        out.append({
            "bin_deg": f"[{lo},{lo+10})", "n": n,
            "median_A_scalar": round(float(np.median(A[m])), 5),
            "median_total_channel_authority": round(float(np.median(tot[m])), 5),
            "median_thrust_share": round(float(np.median(thrust_share[m])), 5),
            "median_channel_authority": {c: round(float(np.median(chan_auth[m, j])), 5) for j, c in enumerate(CH)},
            "frac_A_lt_D": round(float((A[m] < D[m]).mean()), 5),
        })
    return {"binned_by": label, "bins": out}

by_inst = bins(tilt_inst, "INSTANTANEOUS tilt at the boundary state (primary: the channel-structure question)")
by_init = bins(tilt_init, "episode INITIAL tilt (M3-gate convention, for continuity with 10.14/9.46/11.01)")

nz = [b for b in by_inst["bins"] if b["n"] > 0]
peak = max(nz, key=lambda b: b["median_total_channel_authority"])
med_bin = sorted(nz, key=lambda b: -b["n"])[0]
report = {
    "sampling_rule": "verbatim from scripts/analysis/quadrotor_3d_m3_gate.py (band 0.15, halfwidth 2.4525, "
                     "d2r in-loop pool, fixed nominal rollout, _cbf_terms); reporting changed only",
    "ckpt": str(M3), "pool": stem, "band": BAND, "box_halfwidth": HALF,
    "n_boundary_states": int(A.size),
    "channel_ranges": dict(zip(CH, [round(float(v), 5) for v in rng])),
    "overall": {"median_A_scalar": round(float(np.median(A)), 5),
                "median_total_channel_authority": round(float(np.median(tot)), 5),
                "median_thrust_share": round(float(np.median(thrust_share)), 5),
                "median_channel_authority": {c: round(float(np.median(chan_auth[:, j])), 5) for j, c in enumerate(CH)},
                "frac_A_lt_D": round(float((A < D).mean()), 5)},
    "peak_bin_by_total_channel_authority": {"bin": peak["bin_deg"], "value": peak["median_total_channel_authority"]},
    "largest_n_bin": {"bin": med_bin["bin_deg"], "n": med_bin["n"], "thrust_share": med_bin["median_thrust_share"]},
    "by_instantaneous_tilt": by_inst, "by_initial_tilt": by_init,
}
(OUT / "g2_authority_channels.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: report[k] for k in ("n_boundary_states", "channel_ranges", "overall",
                                         "peak_bin_by_total_channel_authority", "largest_n_bin")}, indent=2))
