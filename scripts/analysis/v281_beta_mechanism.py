"""v2.8.1 S1 beta-screen — pin the exact numerical mechanism on the captured failing state.

Captures the state x AND its scene at the first |L_f h|>1e10 during the beta=6 crash, then replays an instrumented
copy of the soft_topk iterated loop at beta in {2,6,12,50}, printing per slot: S=pw.sum(), the residual r
(min / the top obstacle's r after the (1-p) update), p.max, and ‖d rel / d pos‖. Settles whether the blow-up is
the clamp (S>>1e-30 -> inert) or subtractive cancellation in r_s = prod(1-p) as p->1 in float32."""
from pathlib import Path

import numpy as np
import torch

import src.common.filter_cbfqp as FQ
from src.common.observation import _distance_kernel, SOFT_DC, SOFT_INNER, scene_obstacle_tensors

_orig = FQ._cbf_terms
CAP = {}


def patched(system, h_fn, x, scene, u_nom):
    h, lf, lg = _orig(system, h_fn, x, scene, u_nom)
    if not CAP and bool((lf.abs() > 1e10).any()):
        b = int(lf.abs().argmax())
        c, r, a = scene_obstacle_tensors(scene, x.device, x.dtype)
        CAP.update(pos=x[b, :2].detach().cpu().clone(), c=c[b, :, :2].detach().cpu().clone(),
                   r=r[b].detach().cpu().clone(), a=a[b].detach().cpu().clone(), k=int(system.k_obs), lf=float(lf[b]))
    return h, lf, lg


FQ._cbf_terms = patched
import src.frameworks.oc_pncbf.train as T
_ocfg = T.load_effective_config


def cfg6():
    c = _ocfg(); c["run"]["system"] = "quadrotor_3d"; c["training"]["oc_pncbf"]["epochs"] = 100
    c["collection"]["inject_frac"] = 0.0; c["collection"]["collector"] = "continuing"
    c["env"]["band_hazard"] = {"enabled": True, "limit": 4.0}; c["env"]["band_collision_limit"] = 0.0
    c["filter"]["empty_fallback"] = {"mode": "none", "k": 10}
    c.setdefault("obs", {}).setdefault("quadrotor_3d", {}).update({"encoder": "soft_topk", "beta": 6.0})
    return c


T.load_effective_config = cfg6
from src.frameworks.oc_pncbf.train import run_training
try:
    run_training(stage="full", system="quadrotor_3d", seed=42, output_root=Path("data"), device="auto")
except Exception:
    pass

if not CAP:
    print("no capture"); raise SystemExit
pos0, c, r, a, k = CAP["pos"], CAP["c"], CAP["r"], CAP["a"], CAP["k"]
sd = torch.linalg.norm(c - pos0, dim=1) - r
print(f"captured state: L_f(beta6)={CAP['lf']:.4g} n_active={int(a.sum())}")
print(f"  surf_dists(active, sorted)={np.round(np.sort(sd[a].numpy()),4)}")


def instrumented(pos0, beta):
    pos = pos0.clone().requires_grad_(True)               # [2] leaf
    P = pos.reshape(1, 2)
    rel = c.reshape(1, -1, 2) - P.unsqueeze(1)            # [1,n,2]
    surfd = torch.linalg.norm(rel, dim=-1) - r.reshape(1, -1)   # [1,n]
    ab = a.reshape(1, -1)
    sigma = torch.where(ab, _distance_kernel(surfd, SOFT_INNER, SOFT_DC), torch.zeros_like(surfd))
    neg = torch.finfo(surfd.dtype).min / 4.0
    logit = torch.where(ab, -beta * surfd, torch.full_like(surfd, neg))
    rem = ab.to(surfd.dtype)
    rows = []
    slots_rel = []
    for s in range(k):
        w = torch.softmax(logit, dim=1)
        pw = rem * w
        S = pw.sum(dim=1, keepdim=True)
        p = pw / S.clamp_min(1e-30)
        slots_rel.append((p.unsqueeze(-1) * sigma.unsqueeze(-1) * rel).sum(1))
        rows.append((s, float(S), float(p.max()), float(rem.min()), float((rem * (1 - p)).min())))
        rem = (rem * (1.0 - p)).clamp_min(0.0)
    out = torch.stack(slots_rel, 1)
    g = torch.autograd.grad(out.sum(), pos, retain_graph=False)[0]
    return rows, float(g.abs().max())


for beta in (2.0, 6.0, 12.0, 50.0):
    rows, gmax = instrumented(pos0, beta)
    print(f"\nbeta={beta}: ||d rel/d pos||_max={gmax:.4g}")
    for s, S, pmax, rmin, rnext in rows:
        print(f"  slot {s}: S=pw.sum={S:.4e} p.max={pmax:.8f} rem.min(before)={rmin:.4e} (rem*(1-p)).min(after)={rnext:.4e}")
