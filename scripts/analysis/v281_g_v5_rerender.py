"""v2.8.1 S1 V5 — re-render a trajectory-control grid with the fixed plotter (one line per control component).

Rolls out the deployed policy+filter on the in-loop pool, builds EpisodeControlSpec for episodes that show
filter intervention, and calls the (now-fixed) plot_trajectory_control_grid. For the m=4 per-rotor plant the
grid must now draw four safe + four nominal control lines (u1..u4), where the pre-fix plotter drew only two."""
import argparse
from pathlib import Path

import numpy as np
import torch

from src.eval.run_full import _load_framework
from src.eval.build_pools import load_pool
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.common.rk4 import rk4_step
from src.eval.plotting import EpisodeControlSpec, plot_trajectory_control_grid, EPISODES_PER_REPORT, PANELS_PER_FIGURE

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
INLOOP = REPO / "data/eval_pools/eval_inloop_quadrotor-3d-d2r_n500_seed12345.pkl"
OUT = REPO / "data/runs/v2.8.1/s1_diagnostics"; OUT.mkdir(parents=True, exist_ok=True)
TOL = 1e-6


def main(ckpt, label, n_scenes, max_steps, device):
    dev = torch.device(device)
    fw, cfg, ck = _load_framework(ckpt)
    for n in ("value_net", "policy_net"):
        m = getattr(fw, n, None)
        if m is not None:
            m.to(dev)
    sysm = fw.system
    dt = float(cfg["env"]["dt"])
    scenes = load_pool(INLOOP).scenes[:n_scenes]
    bs = batch_scenes(scenes, device=dev, dtype=torch.float32)
    x = initial_states_from_batch(bs)
    S, UN, US = [], [], []
    for _ in range(max_steps):
        x = x.detach()
        u_nom = fw.policy(x, bs)
        _res = fw.filter(x, u_nom, bs)
        u_safe = _res[0] if isinstance(_res, tuple) else _res
        S.append(x.detach().cpu().numpy()); UN.append(u_nom.detach().cpu().numpy()); US.append(u_safe.detach().cpu().numpy())
        x = sysm.wrap_state(rk4_step(sysm, x, u_safe.detach(), dt))
    S = np.stack(S, 1); UN = np.stack(UN, 1); US = np.stack(US, 1)   # [B,T,*]
    B = S.shape[0]
    goal_np = bs.goal.cpu().numpy(); cxy = bs.obstacle_centers[..., :2].cpu().numpy()
    rr = bs.obstacle_radii.cpu().numpy(); aa = bs.obstacle_active.cpu().numpy()
    goal_radius = float(cfg["env"]["goal_radius"])

    def outcome_of(b):
        p = S[b][:, :3]
        gd = np.linalg.norm(p - goal_np[b], axis=1)
        surf = np.linalg.norm(p[:, None, :2] - cxy[b][None], axis=2) - rr[b][None]
        surf = np.where(aa[b][None], surf, np.inf).min(axis=1)
        if (gd < goal_radius).any():
            return "goal"
        if (surf <= 0).any():
            return "collision"
        return "timeout"

    specs = []
    for b in range(B):
        mask = np.linalg.norm(US[b] - UN[b], axis=1) > TOL
        if not mask.any():
            continue
        specs.append(EpisodeControlSpec(scene=scenes[b], pool_index=b, outcome=outcome_of(b),
                                        event_step=int(np.argmax(mask)), filtered_states=S[b],
                                        intervention_mask=mask[:S[b].shape[0] - 1], u_nom=UN[b], u_safe=US[b]))
        if len(specs) >= EPISODES_PER_REPORT:
            break
    ep = specs[:PANELS_PER_FIGURE]
    out = OUT / f"v5_grid_A_{label}.png"
    plot_trajectory_control_grid(ep, out, cfg, role=f"V5 re-render (fixed plotter) · {label}",
                                 system_name=sysm.name, letter="A", u_bounds=sysm.u_bounds,
                                 total_selected=len(specs), shortfall=max(0, EPISODES_PER_REPORT - len(specs)))
    print(f"[V5 {label}] action_dim={US.shape[2]} | intervention episodes={len(specs)} | wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--n-scenes", type=int, default=48)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    main(a.ckpt, a.label, a.n_scenes, a.max_steps, a.device)
