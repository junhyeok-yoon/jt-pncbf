"""v2.8.1 S1 V1 (part 1) + V3 — matched CBF contours and the sigma-knot imprint overlay.

V1b: with the CURRENT plotting code and the SAME in-loop scene, render each run's xy V_hat contour
     (plot_quadrotor3d_cbf_contour: 3 panels, V_hat fill + h_star=0 + cylinders) so 043415 (hard) and v2.8.1
     (soft) sit side by side at the matched step.
V3:  on the v2.8.1 hover panel, overlay dashed circles of radius r_i+2.5 m and r_i+3.0 m about each active
     cylinder (the soft distance-kernel knots) and report whether the open-space filament/band structure of
     V_hat coincides with them. Also reports E[V_hat | |h_star|<band] on the grid (the signed V-units offset
     between the V_hat=0 and h_star=0 sets)."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from src.eval.run_full import _load_framework
from src.eval.build_pools import load_pool
from src.eval.plotting import plot_quadrotor3d_cbf_contour
from src.common.quadrotor_barrier import value_target_barrier

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
INLOOP = REPO / "data/eval_pools/eval_inloop_quadrotor-3d-d2r_n500_seed12345.pkl"
OUT = REPO / "data/runs/v2.8.1/s1_diagnostics"; OUT.mkdir(parents=True, exist_ok=True)
SOFT_INNER, SOFT_DC = 2.5, 3.0


def _scene(scene, dt, dev):
    return SimpleNamespace(
        obstacle_centers=torch.tensor(np.asarray(scene.obstacle_centers, np.float64)[:, :2], dtype=dt, device=dev),
        obstacle_radii=torch.tensor(np.asarray(scene.obstacle_radii, np.float64), dtype=dt, device=dev),
        obstacle_active=torch.tensor(np.asarray(scene.obstacle_active, bool), dtype=torch.bool, device=dev),
        goal=torch.tensor(np.asarray(scene.goal, np.float64), dtype=dt, device=dev))


def v1b(ckpt, label, scene, device):
    fw, cfg, ck = _load_framework(ckpt)
    for n in ("value_net", "policy_net"):
        m = getattr(fw, n, None)
        if m is not None:
            m.to(torch.device(device))
    p = OUT / f"v1_contour_{label}.png"
    plot_quadrotor3d_cbf_contour(scene, p, cfg, fw.system, fw.value_net,
                                 role=f"V1 matched OC contour · {label}", resolution=121)
    print(f"[V1b {label}] wrote {p}")
    return fw, cfg


def v3(fw, cfg, label, scene, device, resolution=161):
    """Hover v=0 panel with the sigma-knot circles + grid signed offset E[V_hat | |h*|<band]."""
    dev = torch.device(device)
    value_net, system = fw.value_net, fw.system
    try:
        param = next(value_net.parameters()); dt = param.dtype
    except StopIteration:
        dt = torch.float32
    world = float(cfg["env"]["world_lim"])
    axis = np.linspace(-world, world, resolution)
    gx, gy = np.meshgrid(axis, axis, indexing="xy")
    G = resolution * resolution
    p = torch.tensor(np.stack([gx.reshape(-1), gy.reshape(-1), np.zeros(G)], 1), dtype=dt, device=dev)
    sc = _scene(scene, dt, dev)
    x = torch.zeros(G, 13, dtype=dt, device=dev); x[:, :3] = p; x[:, 3] = 1.0     # hover, v=0
    with torch.no_grad():
        vhat = value_net.deployed_h(system.observation(x, sc)).reshape(-1)
        hstar = value_target_barrier(system, x, sc, cfg).reshape(-1)
    vg = vhat.cpu().numpy().reshape(resolution, resolution)
    hg = hstar.cpu().numpy().reshape(resolution, resolution)
    centers = np.asarray(scene.obstacle_centers, np.float64)[:, :2]
    radii = np.asarray(scene.obstacle_radii, np.float64)
    active = np.asarray(scene.obstacle_active, bool)

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    cf = ax.contourf(axis, axis, np.clip(vg, -1, 1), levels=41, cmap="RdBu_r")
    ax.contour(axis, axis, vg, levels=[0.0], colors="white", linewidths=1.6)          # V_hat=0
    if hg.min() <= 0 <= hg.max():
        ax.contour(axis, axis, hg, levels=[0.0], colors="black", linewidths=1.2)       # h_star=0
    for k in np.nonzero(active)[0]:
        ax.add_patch(Circle((centers[k, 0], centers[k, 1]), radii[k], fill=False, edgecolor="0.1", lw=1.4))
        ax.add_patch(Circle((centers[k, 0], centers[k, 1]), radii[k] + SOFT_INNER, fill=False,
                            edgecolor="0.2", lw=1.0, ls="--"))                          # r+2.5 (kernel inner)
        ax.add_patch(Circle((centers[k, 0], centers[k, 1]), radii[k] + SOFT_DC, fill=False,
                            edgecolor="0.45", lw=1.0, ls=":"))                          # r+3.0 (kernel cutoff)
    ax.set_xlim(-world, world); ax.set_ylim(-world, world); ax.set_aspect("equal")
    ax.set_title(f"V3 sigma-knot imprint · {label} (hover v=0)\nwhite=V_hat=0, black=h*=0, dashed=r+2.5, dotted=r+3.0")
    fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    pth = OUT / f"v3_knots_{label}.png"; fig.savefig(pth, dpi=140, bbox_inches="tight"); plt.close(fig)

    off = {}
    for band in (0.05, 0.10, 0.20):
        near = np.abs(hg) < band
        off[f"E[V_hat||h*|<{band}]"] = float(vg[near].mean()) if near.any() else None
    rec = {"label": label, "signed_offset_Vunits_on_grid": off, "png": str(pth)}
    (OUT / f"v3_knots_{label}.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(f"[V3 {label}] wrote {pth} | signed offset E[V_hat||h*|<0.1]={off.get('E[V_hat||h*|<0.1]')}")
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--scene-index", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--v3", action="store_true", help="also render the sigma-knot overlay (v2.8.1 soft run)")
    a = ap.parse_args()
    scene = load_pool(INLOOP).scenes[a.scene_index]
    fw, cfg = v1b(a.ckpt, a.label, scene, a.device)
    if a.v3:
        v3(fw, cfg, a.label, scene, a.device)
