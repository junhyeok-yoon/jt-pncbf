"""v2.7.2 M3 degeneracy/authority gate for quadrotor_3d.

The learned value V_hat drives a HardNet CBF filter whose feasibility needs first-order control authority
||L_g V_hat|| > 0 near the safety boundary B_0 (theory note O3). eps_g/R2 (planar 6-state near-B0 sampler)
is INACTIVE for quadrotor_3d, so this gate verifies that omitting it did not leave V_hat degenerate.

Method: roll the fixed nominal on the in-loop pool; at sampled steps collect on-distribution states, keep
those near the LEARNED boundary (|V_hat| < band), and measure ||L_g V_hat|| (via _cbf_terms, which
reconstructs g(x) control-affine). Reports min/median/mean and degen_frac = P(||L_g V_hat|| < floor).
HALT to Researcher if the distribution is degenerate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.common.filter_hardnet import _cbf_terms
from src.common.kstep_fallback import slice_scene
from src.common.rk4 import rk4_step
from src.common.value_net import ValueNetEnsemble, make_h_fn
from src.eval.build_pools import load_pool, DEFAULT_OUTPUT_DIR
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.frameworks.jt_pncbf.train import make_system

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--band", type=float, default=0.15)      # |V_hat| < band  => near the learned boundary
ap.add_argument("--floor", type=float, default=1.0e-3)   # ||L_g V_hat|| below this = degenerate authority
ap.add_argument("--every", type=int, default=5)          # sample every k-th rollout step
a = ap.parse_args()

ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
cfg = ck["config"]
system = make_system(cfg)
vnet = ValueNetEnsemble(system.obs_dim, cfg)
vnet.load_state_dict(ck["v_s_state"])
vnet.eval()
for p in vnet.parameters():
    p.requires_grad_(False)
h_fn = make_h_fn(vnet, system)

pool = load_pool(DEFAULT_OUTPUT_DIR / "eval_inloop_quadrotor-3d_n500_seed12345.pkl")
bs = batch_scenes(pool.scenes, device=torch.device("cpu"), dtype=torch.float32)
x = system.wrap_state(initial_states_from_batch(bs).float())
goal = bs.goal
dt = float(cfg["env"]["dt"])
max_steps = int(cfg["eval"]["max_steps"])
u0 = torch.zeros(x.shape[0], int(system.action_dim))

lg_norms = []
h_at = []
for t in range(max_steps):
    with torch.no_grad():
        un = system.lqr_action(x, goal)
        h = h_fn(x, bs).reshape(-1)
    if t % a.every == 0:
        near = (h.abs() < a.band)
        if bool(near.any()):
            xin = x[near].clone()
            _, _, lg = _cbf_terms(system, h_fn, xin, slice_scene(bs, near), u0[: xin.shape[0]],
                                  create_graph=False)
            lg_norms.append(torch.linalg.norm(lg, dim=1).detach())
            h_at.append(h[near].detach())
    with torch.no_grad():
        x = rk4_step(system, x, un, dt)

lgn = torch.cat(lg_norms).numpy()
res = {
    "ckpt": a.ckpt, "best_step": int(ck.get("step", -1)),
    "n_boundary_states": int(lgn.size), "band": a.band, "floor": a.floor,
    "lg_min": float(lgn.min()), "lg_p05": float(np.percentile(lgn, 5)),
    "lg_median": float(np.median(lgn)), "lg_mean": float(lgn.mean()), "lg_max": float(lgn.max()),
    "degen_frac": float((lgn < a.floor).mean()),
}
res["gate"] = "PASS (non-degenerate)" if (res["lg_median"] > 10 * a.floor and res["degen_frac"] < 0.10) \
    else "HALT (degenerate authority)"
print(json.dumps(res, indent=2))
