"""v2.8.1 S1 beta-screen — contamination check (gates whether the completed beta=2.0 cell + V1-V5 stand).

Over the states the completed beta=2.0 value run (022629) actually visited, report the obstacle-block
||d obs / d p_xy|| distribution (max, p99, p99.9, fraction above 10x median) under BOTH the OLD (pre-fix) and the
redesigned encoder, the beta=2.0 output move, and whether the run's own encoder-gradient ever entered the
pathological range (>1e6, the surrogate for L_f V_hat blow-up). If outputs are materially unchanged and the tail
is clean -> the completed cell + V1-V5 stand; else beta=2.0 is re-run under the fix and V1-V5 re-read."""
import json
from pathlib import Path

import numpy as np
import torch

from src.eval.run_full import _load_framework
from src.eval.build_pools import load_pool
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.common.rk4 import rk4_step
from src.common.observation import (soft_topk_obstacles, scene_obstacle_tensors, _distance_kernel,
                                    SOFT_DC, SOFT_INNER)

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
CK = REPO / "data/runs/v2.8.1/set__20260802-022629__seed42/v2.8.1__oc__20260802-022629__seed42/checkpoints/best.pt"
INLOOP = REPO / "data/eval_pools/eval_inloop_quadrotor-3d-d2r_n500_seed12345.pkl"
OUT = REPO / "data/runs/v2.8.1/s1_diagnostics"; OUT.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NS, MAXS = 256, 200


def soft_topk_OLD(pos, c, r, a, k, beta):
    """Pre-fix encoder (division normaliser + subtractive multiplicative residual)."""
    cb = c if c.ndim == 3 else c.unsqueeze(0).expand(pos.shape[0], -1, -1)
    rb = r if r.ndim == 2 else r.unsqueeze(0).expand(pos.shape[0], -1)
    ab = a if a.ndim == 2 else a.unsqueeze(0).expand(pos.shape[0], -1)
    rel = cb - pos.unsqueeze(1); sd = torch.linalg.norm(rel, dim=-1) - rb
    rel_m = torch.where(ab.unsqueeze(-1), rel, torch.zeros_like(rel))
    sigma = torch.where(ab, _distance_kernel(sd, SOFT_INNER, SOFT_DC), torch.zeros_like(sd))
    neg = torch.finfo(sd.dtype).min / 4.0
    logit = torch.where(ab, -beta * sd, torch.full_like(sd, neg))
    rr = ab.to(sd.dtype); rels = []
    for _ in range(k):
        w = torch.softmax(logit, dim=1); pw = rr * w
        p = pw / pw.sum(dim=1, keepdim=True).clamp_min(1e-30)
        rels.append(((p * sigma).unsqueeze(-1) * rel_m).sum(1)); rr = (rr * (1 - p)).clamp_min(0.0)
    return torch.stack(rels, 1)


def block_grad_norm(fn, pos, c, r, a, k, beta):
    pos = pos.detach().clone().requires_grad_(True)
    out = fn(pos, c, r, a, k, beta)
    if isinstance(out, tuple):
        out = out[0]
    g = torch.autograd.grad(out.sum(), pos)[0]                      # d(sum obstacle-block)/d p_xy
    return g.norm(dim=1).detach()


def dist(v):
    v = np.asarray(v, float)
    q = np.percentile(v, [50, 90, 99, 99.9])
    return {"max": float(v.max()), "p99.9": float(q[3]), "p99": float(q[2]), "p90": float(q[1]),
            "median": float(q[0]), "frac_gt_10x_median": float((v > 10 * q[0]).mean()),
            "frac_gt_1e6": float((v > 1e6).mean())}


fw, cfg, ck = _load_framework(str(CK))
for _n in ("value_net", "policy_net"):
    _m = getattr(fw, _n, None)
    if _m is not None:
        _m.to(DEV)
sysm = fw.system; k = int(sysm.k_obs); beta = float(sysm.soft_beta)
scenes = load_pool(INLOOP).scenes[:NS]
bs = batch_scenes(scenes, device=DEV, dtype=torch.float32)
x = initial_states_from_batch(bs)
old_g, new_g, outmove = [], [], []
for _ in range(MAXS):
    x = x.detach()
    u_nom = fw.policy(x, bs); _res = fw.filter(x, u_nom, bs)
    u_safe = _res[0] if isinstance(_res, tuple) else _res
    c, r, a = scene_obstacle_tensors(bs, x.device, x.dtype)
    p_xy = x[:, :2]
    old_g.append(block_grad_norm(soft_topk_OLD, p_xy, c[..., :2], r, a, k, beta).cpu().numpy())
    new_g.append(block_grad_norm(lambda P, C, R, A, K, B: soft_topk_obstacles(P, C, R, A, K, beta=B, d_c=SOFT_DC, inner=SOFT_INNER),
                                 p_xy, c[..., :2], r, a, k, beta).cpu().numpy())
    with torch.no_grad():
        o_old = soft_topk_OLD(p_xy, c[..., :2], r, a, k, beta)
        o_new = soft_topk_obstacles(p_xy, c[..., :2], r, a, k, beta=beta, d_c=SOFT_DC, inner=SOFT_INNER)[0]
        outmove.append((o_old - o_new).abs().reshape(p_xy.shape[0], -1).max(1).values.cpu().numpy())
    x = sysm.wrap_state(rk4_step(sysm, x, u_safe.detach(), 0.05))
OG = np.concatenate(old_g); NG = np.concatenate(new_g); OM = np.concatenate(outmove)
rec = {"beta": beta, "n_states": int(OG.size),
       "obstacle_block_grad_norm_OLD": dist(OG), "obstacle_block_grad_norm_NEW": dist(NG),
       "output_move_OLD_vs_NEW": {"max": float(OM.max()), "p99": float(np.percentile(OM, 99)),
                                  "median": float(np.median(OM)), "frac_gt_1e-3": float((OM > 1e-3).mean())}}
(OUT / "beta2_contamination.json").write_text(json.dumps(rec, indent=2) + "\n")
print(f"beta={beta} states={OG.size}")
print(f"  OLD block-grad: max={rec['obstacle_block_grad_norm_OLD']['max']:.3g} p99.9={rec['obstacle_block_grad_norm_OLD']['p99.9']:.3g} "
      f"median={rec['obstacle_block_grad_norm_OLD']['median']:.3g} frac>1e6={rec['obstacle_block_grad_norm_OLD']['frac_gt_1e6']:.4f}")
print(f"  NEW block-grad: max={rec['obstacle_block_grad_norm_NEW']['max']:.3g} p99.9={rec['obstacle_block_grad_norm_NEW']['p99.9']:.3g} "
      f"median={rec['obstacle_block_grad_norm_NEW']['median']:.3g} frac>1e6={rec['obstacle_block_grad_norm_NEW']['frac_gt_1e6']:.4f}")
print(f"  OUTPUT move OLD->NEW: max={rec['output_move_OLD_vs_NEW']['max']:.3g} median={rec['output_move_OLD_vs_NEW']['median']:.3g} "
      f"frac>1e-3={rec['output_move_OLD_vs_NEW']['frac_gt_1e-3']:.3f}")
print(f"wrote {OUT/'beta2_contamination.json'}")
