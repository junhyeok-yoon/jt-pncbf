"""v2.8.1 S1 V4 — far-obstacle blindness in in-loop collision episodes.

Roll out the deployed policy+filter on the in-loop pool. A CYLINDER-collision episode is one whose min active-
cylinder surface distance goes <= 0 at some step before it first enters the goal region; the collision step
t_c is the first such step. Among collision episodes with t_c >= 10, report the fraction whose NEAREST active
cylinder surface distance 10 steps before contact exceeded d_c = 3.0 m — the states where the soft encoder's
compact support (sigma = 0 beyond 3.0 m) left the certificate blind to the obstacle it was about to hit. Read
beside G2's all-zero obstacle-block fraction (0.0471). Read-only on live dirs; output to s1_diagnostics/."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.eval.run_full import _load_framework
from src.eval.build_pools import load_pool
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.common.rk4 import rk4_step

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
INLOOP = REPO / "data/eval_pools/eval_inloop_quadrotor-3d-d2r_n500_seed12345.pkl"
OUT = REPO / "data/runs/v2.8.1/s1_diagnostics"; OUT.mkdir(parents=True, exist_ok=True)
D_C = 3.0
LOOKBACK = 10


def analyze(ckpt, label, n_scenes, max_steps, device):
    dev = torch.device(device)
    fw, cfg, ck = _load_framework(ckpt)
    for n in ("value_net", "policy_net"):
        m = getattr(fw, n, None)
        if m is not None:
            m.to(dev)
    sysm = fw.system
    dt = float(cfg["env"]["dt"])
    goal_radius = float(cfg["env"]["goal_radius"])
    scenes = load_pool(INLOOP).scenes[:n_scenes]
    bs = batch_scenes(scenes, device=dev, dtype=torch.float32)
    x = initial_states_from_batch(bs)
    c = bs.obstacle_centers[..., :2]                    # [B,K,2]
    r = bs.obstacle_radii                               # [B,K]
    act = bs.obstacle_active                            # [B,K]
    goal = bs.goal                                      # [B,3]

    surf_hist, goald_hist = [], []
    for _ in range(max_steps):
        x = x.detach()
        u_nom = fw.policy(x, bs)
        _res = fw.filter(x, u_nom, bs)
        u_safe = _res[0] if isinstance(_res, tuple) else _res
        p = x[:, :3]
        d = torch.norm(p[:, None, :2] - c, dim=2) - r    # [B,K] surface distance
        d = torch.where(act, d, torch.full_like(d, float("inf")))
        surf_hist.append(d.min(dim=1).values.detach().cpu())          # [B] nearest active cylinder surf dist
        goald_hist.append(torch.norm(p - goal, dim=1).detach().cpu()) # [B] distance to goal (3d position)
        x = sysm.wrap_state(rk4_step(sysm, x, u_safe.detach(), dt))
    S = torch.stack(surf_hist, 0).numpy()               # [T,B]
    Gd = torch.stack(goald_hist, 0).numpy()             # [T,B]
    T, B = S.shape

    def first_true(mask):                               # first t index True per column, or -1
        out = np.full(B, -1)
        for b in range(B):
            w = np.nonzero(mask[:, b])[0]
            out[b] = w[0] if w.size else -1
        return out
    t_coll = first_true(S <= 0.0)
    t_reach = first_true(Gd < goal_radius)
    is_coll = (t_coll >= 1) & ((t_reach < 0) | (t_coll < t_reach))    # collided before reaching
    coll_idx = np.nonzero(is_coll)[0]
    with_lb = [b for b in coll_idx if t_coll[b] >= LOOKBACK]
    far = [b for b in with_lb if S[t_coll[b] - LOOKBACK, b] > D_C]
    surf_at_lb = [float(S[t_coll[b] - LOOKBACK, b]) for b in with_lb]

    rec = {"label": label, "encoder": getattr(sysm, "encoder", "?"), "n_scenes": n_scenes, "max_steps": max_steps,
           "d_c": D_C, "lookback_steps": LOOKBACK, "goal_radius": goal_radius,
           "n_collision_episodes": int(is_coll.sum()),
           "n_collision_with_lookback": len(with_lb),
           "n_collision_tc_lt_lookback": int(len(coll_idx) - len(with_lb)),
           "far_blindness_fraction": (len(far) / len(with_lb)) if with_lb else None,
           "n_far": len(far),
           "surf_at_lookback_median": float(np.median(surf_at_lb)) if surf_at_lb else None,
           "surf_at_lookback_p95": float(np.percentile(surf_at_lb, 95)) if surf_at_lb else None,
           "G2_all_zero_obstacle_block_frac_reference": 0.0471}
    (OUT / f"v4_blindness_{label}.json").write_text(json.dumps(rec, indent=2) + "\n")
    frac = rec["far_blindness_fraction"]
    print(f"[{label}] enc={rec['encoder']} coll_eps={rec['n_collision_episodes']} "
          f"(with lookback {len(with_lb)}) | far-blind fraction (>d_c 10 steps pre-contact)="
          f"{frac if frac is None else round(frac,4)} n_far={len(far)} | "
          f"surf@lb median={rec['surf_at_lookback_median']}", flush=True)
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--n-scenes", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    analyze(a.ckpt, a.label, a.n_scenes, a.max_steps, a.device)
