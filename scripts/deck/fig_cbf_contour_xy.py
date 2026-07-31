"""v2.7.7 M18b (Amdt 6/7) — xy-plane certificate contour, AFTER-only (absolute capability of the final system).
Deployed V̂ of the after certificate (09c33bf4, dim-34) on the (p_x, p_y) plane at a fixed slice, with cylinder
cross-sections overlaid. Eval-only forward of value_net.deployed_h(system.observation(·)) on a grid (no src edit).
Full slice-condition line (same convention as M17). Message: the final certificate carves the lateral cylinder
barriers cleanly at this altitude."""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from src.eval.run_full import _load_framework
from src.eval.build_pools import load_pool
from scripts.deck.deck_style import save, C_VERIFY, AFTER
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm

CK = Path("data/previous_runs/v2.7.6/set__20260725-043415__seed42/v2.7.6__jt__20260725-052127__seed42/checkpoints/step_042000.pt")
INLOOP = Path("data/eval_pools/eval_inloop_quadrotor-3d-d2r_n500_seed12345.pkl")
RES, WORLD, PZ = 140, 4.5, 0.0
CMAP, NORM = "coolwarm", TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
SLICE = ("slice: p_z=0 m, (p_x, p_y) swept  ·  v=(v_x, v_y, v_z)=(0, 0, 0) m/s  ·  "
         "tilt 0° (hover, q=[1,0,0,0])  ·  ω=(0, 0, 0) rad/s")
NOTE = "all other velocity and rate components are held at zero; only (p_x, p_y) vary"

scene = load_pool(INLOOP).scenes[0]
C = np.asarray(scene.obstacle_centers, np.float64)[:, :2]; R = np.asarray(scene.obstacle_radii, np.float64)
A = np.asarray(scene.obstacle_active, bool); goal = np.asarray(scene.goal, np.float64).reshape(-1)[:3]
ax_lin = np.linspace(-WORLD, WORLD, RES); gx, gy = np.meshgrid(ax_lin, ax_lin)

fw, cfg, _ = _load_framework(CK)
sys_ = fw.system; dev = next(fw.value_net.parameters()).device; dt = torch.float32
scene_t = SimpleNamespace(obstacle_centers=torch.tensor(np.asarray(scene.obstacle_centers), dtype=dt, device=dev),
                          obstacle_radii=torch.tensor(np.asarray(scene.obstacle_radii), dtype=dt, device=dev),
                          obstacle_active=torch.tensor(np.asarray(scene.obstacle_active), dtype=torch.bool, device=dev),
                          goal=torch.tensor(np.asarray(scene.goal), dtype=dt, device=dev))
x = torch.zeros(RES * RES, 13, dtype=dt, device=dev)
x[:, 0] = torch.tensor(gx.reshape(-1), dtype=dt, device=dev)
x[:, 1] = torch.tensor(gy.reshape(-1), dtype=dt, device=dev)
x[:, 2] = PZ; x[:, 3] = 1.0
with torch.no_grad():
    V = np.clip(fw.value_net.deployed_h(sys_.observation(x, scene_t)).reshape(-1).cpu().numpy().reshape(RES, RES), -1, 1)

fig, ax = plt.subplots(figsize=(7.6, 7.2))
cf = ax.contourf(ax_lin, ax_lin, V, levels=np.linspace(-1, 1, 41), cmap=CMAP, norm=NORM, extend="both")
if V.min() <= 0 <= V.max():
    ax.contour(ax_lin, ax_lin, V, levels=[0.0], colors="black", linewidths=1.6)
for j in np.nonzero(A)[0]:
    ax.add_patch(mpatches.Circle((C[j, 0], C[j, 1]), R[j], facecolor="none", edgecolor="black", lw=1.5, ls="--"))
ax.plot([goal[0]], [goal[1]], "*", ms=18, color=C_VERIFY, mec="black", zorder=6)
ax.set_aspect("equal"); ax.set_xlim(-WORLD, WORLD); ax.set_ylim(-WORLD, WORLD)
ax.set_xlabel("p_x (m)", fontsize=11); ax.set_ylabel("p_y (m)", fontsize=11); ax.tick_params(labelsize=9)
fig.colorbar(ScalarMappable(norm=NORM, cmap=CMAP), ax=ax, fraction=0.046, pad=0.03, label="V̂ (clamped [-1,1])")
fig.suptitle(f"Certificate xy-section (hover) — {AFTER}", fontsize=13, fontweight="bold")
fig.text(0.5, 0.10, SLICE, ha="center", fontsize=9)
fig.text(0.5, 0.065, NOTE + "  ·  dashed circles = cylinders; V̂>0 (red) unsafe, V̂<0 (blue) safe.",
         ha="center", fontsize=8, style="italic", color="#555555")
fig.subplots_adjust(bottom=0.18, top=0.93)
p = save(fig, "fig_cbf_contour_xy.png")
print(f"M18b -> {p.name}; after V̂ range [{V.min():.2f},{V.max():.2f}]; obstacles {int(A.sum())}")
