"""v2.7.6 Stage-2 M5 — thrust-channel share of ||L_g V_hat||, by the SAME channel decomposition that produced
the 0.0158 reference (theory_g2 / M3 authority gate): nominal-rollout boundary states (|V_hat|<0.15), wrench
row = L_g V_hat @ mixer_inv, channel authority = |component| * channel range, thrust share = f_thr / total.
Same sampling rule as 0.0158 -> the only change is the V_hat, isolating whether it inherited the band term's
thrust-channel structure. Reports overall median AND the band-boundary subset (|z| near 4). Read-only."""
from __future__ import annotations

import json, math
from pathlib import Path

import numpy as np
import torch

from src.common.filter_hardnet import _cbf_terms
from src.common.value_net import make_h_fn
from src.common.rk4 import rk4_step
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.eval.build_pools import load_pool
from src.eval.run_full import _load_framework

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
POOLS = REPO / "data/runs/v2.7.6/pools"
BAND = 0.15; HALF = 2.4525; CH = ["f_thr", "tau_x", "tau_y", "tau_z"]


def lg_thrust_share(ckpt: Path, stem: str, band_z_lo=3.0):
    fw, cfg, ck = _load_framework(ckpt, config_overrides={"env": {"dt": 0.05}, "eval": {"max_steps": 200}})
    system = fw.system
    h_fn = make_h_fn(fw.value_net, system)
    pool = load_pool(POOLS / f"{stem}.pkl")
    bs = batch_scenes(pool.scenes, device=torch.device("cpu"), dtype=torch.float32)
    x = system.wrap_state(initial_states_from_batch(bs).float()); goal = bs.goal
    dt = float(cfg["env"]["dt"]); max_steps = int(cfg["eval"]["max_steps"])
    u0 = torch.zeros(x.shape[0], int(system.action_dim))
    Minv = system.mixer_inv.to(torch.float64)
    f_max = float(system.f_rotor_max)
    rng = (f_max * system.mixer.abs().sum(dim=1)).to(torch.float64).numpy()
    W_all, Z_all = [], []
    for _ in range(max_steps):
        with torch.no_grad():
            un = system.lqr_action(x, goal)
            h = h_fn(x, bs).reshape(-1)
        near = (h.abs() < BAND)
        if bool(near.any()):
            xin = x[near].clone()
            from src.common.kstep_fallback import slice_scene as _slice
            _h, lf, lg = _cbf_terms(system, h_fn, xin, _slice(bs, near), u0[: xin.shape[0]], create_graph=False)
            W_all.append((lg.detach().to(torch.float64) @ Minv).numpy())
            Z_all.append(system.position(xin)[..., 2].detach().numpy())
        with torch.no_grad():
            x = rk4_step(system, x, un, dt)
    W = np.concatenate(W_all, axis=0); Z = np.concatenate(Z_all)
    chan_auth = np.abs(W) * rng[None, :]; tot = chan_auth.sum(axis=1)
    thrust_share = chan_auth[:, 0] / np.maximum(tot, 1e-12)
    band_m = np.abs(Z) >= band_z_lo
    out = {"ckpt": ckpt.name, "step": int(ck["step"]), "stem": stem, "n_boundary_states": int(W.shape[0]),
           "channel_ranges": {c: round(float(rng[j]), 4) for j, c in enumerate(CH)},
           "median_thrust_share_overall": round(float(np.median(thrust_share)), 5),
           "median_channel_authority": {c: round(float(np.median(chan_auth[:, j])), 5) for j, c in enumerate(CH)},
           "reference_v274_0.0158": 0.0158,
           "band_boundary_subset_|z|>=%.1f" % band_z_lo: {
               "n": int(band_m.sum()),
               "median_thrust_share": round(float(np.median(thrust_share[band_m])), 5) if band_m.any() else None}}
    return out


if __name__ == "__main__":
    import sys as _s
    r = lg_thrust_share(Path(_s.argv[1]), _s.argv[2] if len(_s.argv) > 2 else "eval_bandfeasible_quadrotor-3d-d2r_n2000_seed42")
    print(json.dumps(r, indent=2))
