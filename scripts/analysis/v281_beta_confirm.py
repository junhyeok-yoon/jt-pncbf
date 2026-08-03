"""v2.8.1 S1 beta-screen Step 1 confirmation — capture the failing state and prove the soft-rank position-gradient
explosion is beta-driven. Patches filter_cbfqp._cbf_terms to capture the state x with the first |L_f h|>1e10, then
reports at beta in {2,6,12,50}: ‖d(obstacle-block obs)/d p_xy‖ (the soft-rank position gradient), the resulting
L_f h, and the soft-rank slot top-mass. Saves the state for a regression test."""
from pathlib import Path

import numpy as np
import torch

import src.common.filter_cbfqp as FQ

_orig = FQ._cbf_terms
CAP = {"x": None, "scene": None, "sys": None, "hfn": None}


def patched(system, h_fn, x, scene, u_nom):
    h, lf, lg = _orig(system, h_fn, x, scene, u_nom)
    if CAP["x"] is None:
        big = (lf.abs() > 1e10)
        if bool(big.any()):
            b = int(lf.abs().argmax())
            CAP.update(x=x[b:b + 1].detach().cpu().clone(), scene=scene, sys=system, hfn=h_fn, b=b,
                       lf=float(lf[b]), lg=lg[b].detach().cpu().numpy(), h=float(h[b]))
    return h, lf, lg


FQ._cbf_terms = patched

import src.frameworks.oc_pncbf.train as T
_ocfg = T.load_effective_config


def cfg6():
    c = _ocfg()
    c["run"]["system"] = "quadrotor_3d"; c["training"]["oc_pncbf"]["epochs"] = 100
    c["collection"]["inject_frac"] = 0.0; c["collection"]["collector"] = "continuing"
    c["env"]["band_hazard"] = {"enabled": True, "limit": 4.0}; c["env"]["band_collision_limit"] = 0.0
    c["filter"]["empty_fallback"] = {"mode": "none", "k": 10}
    c.setdefault("obs", {}).setdefault("quadrotor_3d", {}).update({"encoder": "soft_topk", "beta": 6.0})
    return c


T.load_effective_config = cfg6
from src.frameworks.oc_pncbf.train import run_training
from src.common.observation import scene_obstacle_tensors, soft_topk_obstacles, SOFT_DC, SOFT_INNER

try:
    run_training(stage="full", system="quadrotor_3d", seed=42, output_root=Path("data"), device="auto")
except Exception:
    pass

if CAP["x"] is None:
    print("did not capture an extreme-L_f state")
else:
    sysm = CAP["sys"]; x = CAP["x"].to(next(sysm.value_net.parameters()).device if hasattr(sysm, "value_net") else "cpu")
    dev = CAP["scene"].goal.device
    x = CAP["x"].to(dev)
    c, r, a = scene_obstacle_tensors(CAP["scene"], dev, x.dtype)
    b = CAP["b"]
    cb, rb, ab = c[b:b + 1, ..., :2], r[b:b + 1], a[b:b + 1]
    sd = np.sort((torch.linalg.norm(c[b, :, :2] - x[0, :2], dim=1) - r[b]).detach().cpu().numpy())
    print(f"captured failing state: b={b} h={CAP['h']:.4g} L_f(beta6)={CAP['lf']:.4g} max|L_g|={np.abs(CAP['lg']).max():.4g}")
    print(f"  6 nearest surf_dists={np.round(sd[:6],4)} min={sd.min():.4f}")
    print("  soft-rank position-gradient of the obstacle block vs beta:")
    for beta in (2.0, 6.0, 12.0, 50.0):
        pos = x[0:1, :2].detach().clone().requires_grad_(True)
        rel, rad, tm = soft_topk_obstacles(pos, cb, rb, ab, int(sysm.k_obs), beta=beta, d_c=SOFT_DC, inner=SOFT_INNER, return_indices=True)
        g = torch.autograd.grad(rel.sum() + rad.sum(), pos)[0]
        gmax = float(g.abs().max())
        print(f"    beta={beta:>4}: ||d(block)/d p_xy||_max={gmax:.4g} | top_mass={np.round(tm[0].detach().cpu().numpy(),4)} | fwd_finite={bool(torch.isfinite(rel).all())}")
    np.save("data/runs/v2.8.1/s1_diagnostics/beta6_failing_state.npy", CAP["x"].numpy())
    print("  saved failing state -> data/runs/v2.8.1/s1_diagnostics/beta6_failing_state.npy")
