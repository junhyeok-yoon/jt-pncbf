"""v2.6.0 M6 — quadrotor CBF contour (the 6D-appropriate figure the auto-eval skips, run_full.py:252).

The learned deployed certificate h(x)=V_hat is rendered on the POSITION plane (px,py) for representative
eval scenes, across three APPROACH-SPEED slices s = v^T Re (velocity along the body thrust axis Re, the
h_star velocity channel that breaks the position-only B_0 degeneracy, Thm 5.3). At theta=0, Re=(0,1) so the
slice is v=(0,s), omega=0. Panels overlay the active obstacles, the goal, and the h=0 safety boundary.
Convention (matches the DI contour + the sigmoid gate gate_in=sigmoid(-h)): h<0 = SAFE (blue), h>0 = unsafe
(red), the h=0 level set is the filter boundary. best.pt (learned V_hat), coolwarm/TwoSlopeNorm centered 0.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from types import SimpleNamespace

from src.common.value_net import ValueNetEnsemble, make_h_fn
from src.eval.build_pools import load_pool
from src.frameworks.jt_pncbf.train import make_system

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
POOL = REPO / "data/secured_data/pools/eval_full_quadrotor-planar_n2000_seed23456.pkl"
SCENE_IDS = [0, 1]                       # a clean 2-obstacle scene + a dense 6-obstacle scene
APPROACH_SLICES = [(-2.0, "descending  v·Re=-2"), (0.0, "hover  v·Re=0"), (+2.0, "ascending  v·Re=+2")]
RES = 200
LEVELS = np.linspace(-1.0, 1.0, 41)


def main():
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else max(REPO.glob("data/v2.6.0__*seed42"))
    out_path = run_dir / "figures" / "cbf_contour_m6.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ck = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    cfg = ck["config"]; step = int(ck.get("step", -1))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    system = make_system(cfg)
    vnet = ValueNetEnsemble(system.obs_dim, cfg).to(dev, torch.float32)
    vnet.load_state_dict(ck["v_s_state"]); vnet.eval()
    for p in vnet.parameters():
        p.requires_grad_(False)
    h_fn = make_h_fn(vnet, system)

    world_lim = float(cfg["env"]["world_lim"])
    goal_radius = float(cfg["env"]["goal_radius"])
    ax_lin = torch.linspace(-world_lim, world_lim, RES, device=dev, dtype=torch.float32)
    gx, gy = torch.meshgrid(ax_lin, ax_lin, indexing="xy")
    pos = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)         # [R*R, 2]
    axn = ax_lin.cpu().numpy()

    scenes = load_pool(POOL).scenes
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    nrow, ncol = len(SCENE_IDS), len(APPROACH_SLICES)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 4.6 * nrow), dpi=150)
    axes = np.atleast_2d(axes)
    fig.suptitle(f"v2.6.0 M6 · learned CBF h(x)=V_hat · quadrotor_planar · {run_dir.name}/best.pt @ step {step}\n"
                 f"position plane, theta=0, omega=0; columns = approach speed s=v·Re (h_star velocity channel). "
                 f"h<0 safe (blue), h=0 boundary (black), h>0 unsafe (red).", fontsize=11)
    mappable = None
    for r, sid in enumerate(SCENE_IDS):
        sc = scenes[sid]
        active = np.asarray(sc.obstacle_active).astype(bool)
        centers = np.asarray(sc.obstacle_centers)[active]
        radii = np.asarray(sc.obstacle_radii)[active]
        goal = np.asarray(sc.goal)
        scene_t = SimpleNamespace(
            obstacle_centers=torch.as_tensor(np.asarray(sc.obstacle_centers), dtype=torch.float32, device=dev),
            obstacle_radii=torch.as_tensor(np.asarray(sc.obstacle_radii), dtype=torch.float32, device=dev),
            obstacle_active=torch.as_tensor(np.asarray(sc.obstacle_active), dtype=torch.bool, device=dev),
            goal=torch.as_tensor(goal, dtype=torch.float32, device=dev))
        for c, (s, label) in enumerate(APPROACH_SLICES):
            ax = axes[r, c]
            B = pos.shape[0]
            x = torch.zeros(B, 6, device=dev, dtype=torch.float32)
            x[:, :2] = pos
            x[:, 4] = s                                                # vy = s (theta=0 -> Re=(0,1), v·Re=vy)
            with torch.no_grad():
                h = torch.clamp(h_fn(x, scene_t), -1.0, 1.0).reshape(RES, RES).cpu().numpy()
            mappable = ax.contourf(axn, axn, h, levels=LEVELS, cmap="coolwarm", norm=norm, extend="both")
            if h.min() <= 0.0 <= h.max():
                ax.contour(axn, axn, h, levels=[0.0], colors="black", linewidths=1.6)
            for ctr, rad in zip(centers, radii):
                ax.add_patch(plt.Circle((ctr[0], ctr[1]), rad, fill=False, ec="k", lw=1.4, ls="--"))
                ax.add_patch(plt.Circle((ctr[0], ctr[1]), rad, fill=True, fc="k", alpha=0.12))
            ax.plot(goal[0], goal[1], marker="*", ms=18, mfc="gold", mec="k", mew=1.0, zorder=5)
            ax.add_patch(plt.Circle((goal[0], goal[1]), goal_radius, fill=False, ec="gold", lw=1.2))
            ax.set_xlim(-world_lim, world_lim); ax.set_ylim(-world_lim, world_lim)
            ax.set_aspect("equal"); ax.set_title(f"scene {sid} ({int(active.sum())} obs) · {label}", fontsize=9)
            if c == 0:
                ax.set_ylabel("py")
            if r == nrow - 1:
                ax.set_xlabel("px")
    fig.subplots_adjust(right=0.90, top=0.88, hspace=0.20, wspace=0.15)
    cax = fig.add_axes([0.92, 0.12, 0.015, 0.72])
    fig.colorbar(mappable, cax=cax, label="h(x) = V_hat  (clamped [-1,1])")
    fig.savefig(out_path, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
