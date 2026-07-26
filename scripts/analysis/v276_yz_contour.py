"""v2.7.6 Stage-2 — yz CBF contour section (04_eval s3b extended). Sweeps (py, pz) over [-world_lim,world_lim]^2
at px=0; three v_z columns (-1.5, 0, +1.5) with hover attitude (v_z the only variable); two scenes as rows.
Follows s3b conventions (coolwarm, clamp[-1,1], black h_star=0 contour, shared colorbar, per-panel V_hat
min/max). Overlays: obstacle cylinders as vertical py-bands; band surfaces |z|=4 as reference lines; the
analytic band-branch h_star=0 as a distinct dashed line. Reports the V_hat=0 crossing on the py=0 axis per
column. CPU-only, low thread count + nice (06_workflow s6.4 non-disturbance); read-only on the live run;
outputs to data/runs/v2.7.6/stage2_figures/."""
from __future__ import annotations

import json, math, sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
torch.set_num_threads(2)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from src.common.quadrotor_barrier import value_target_barrier
from src.eval.build_pools import load_pool
from src.eval.run_full import _load_framework

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
POOL = REPO / "data/runs/v2.7.6/pools/eval_bandfeasible_quadrotor-3d-d2r_n2000_seed42.pkl"
OUT = REPO / "data/runs/v2.7.6/stage2_figures"; OUT.mkdir(parents=True, exist_ok=True)
LEVELS = np.linspace(-1.0, 1.0, 41); RES = 121


def pick_scenes(n=2):
    pool = load_pool(POOL); out = []
    for i, s in enumerate(pool.scenes):
        c = np.asarray(s.obstacle_centers)[:, :2]; r = np.asarray(s.obstacle_radii); a = np.asarray(s.obstacle_active, bool)
        if any(abs(c[j, 0]) < r[j] for j in np.nonzero(a)[0]):        # obstacle crosses px=0 -> visible band
            out.append(s)
        if len(out) >= n:
            break
    return out


def render(ckpt: Path, tag: str):
    fw, cfg, ck = _load_framework(ckpt, config_overrides={"env": {"dt": 0.05}, "eval": {"max_steps": 200}})
    system = fw.system; value_net = fw.value_net
    world = float(cfg["env"]["world_lim"])
    c_z = math.pi / float(cfg["env"]["bounds"]["quadrotor_3d"]["omega_max"])
    dt = torch.float32; dev = torch.device("cpu")
    ax = np.linspace(-world, world, RES); gy, gz = np.meshgrid(ax, ax, indexing="xy")  # gy=py (x-axis), gz=pz (y-axis)
    G = RES * RES
    cols = [("v_z=-1.5 (descending)", -1.5), ("v_z=0 (hover)", 0.0), ("v_z=+1.5 (ascending)", 1.5)]
    scenes = pick_scenes(2)
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    fig, axes = plt.subplots(len(scenes), 3, figsize=(16.5, 5.4 * len(scenes)), dpi=140, squeeze=False)
    fig.suptitle(f"v2.7.6 · {tag} · quadrotor_3d V_hat yz section (px=0), step {int(ck['step'])}", fontsize=13)
    crossings = {}
    for ri, scene in enumerate(scenes):
        C = np.asarray(scene.obstacle_centers, np.float64)[:, :2]; R = np.asarray(scene.obstacle_radii, np.float64)
        A = np.asarray(scene.obstacle_active, bool)
        scene_t = SimpleNamespace(
            obstacle_centers=torch.tensor(C, dtype=dt), obstacle_radii=torch.tensor(R, dtype=dt),
            obstacle_active=torch.tensor(A, dtype=torch.bool), goal=torch.tensor(np.asarray(scene.goal, np.float64), dtype=dt))
        for ci, (label, vz) in enumerate(cols):
            axp = axes[ri][ci]
            p = torch.tensor(np.stack([np.zeros(G), gy.reshape(-1), gz.reshape(-1)], axis=1), dtype=dt)  # px=0
            x = torch.zeros(G, 13, dtype=dt); x[:, :3] = p
            x[:, 3] = 1.0                                            # hover quat identity
            x[:, 9] = vz                                             # v_z (vx=vy=0, omega=0)
            with torch.no_grad():
                vhat = value_net.deployed_h(system.observation(x, scene_t)).reshape(-1)
                hstar = value_target_barrier(system, x, scene_t, cfg).reshape(-1)
            vg = torch.clamp(vhat, -1, 1).numpy().reshape(RES, RES); hg = hstar.numpy().reshape(RES, RES)
            cf = axp.contourf(ax, ax, vg, levels=LEVELS, cmap="coolwarm", norm=norm, extend="both")
            if hg.min() <= 0.0 <= hg.max():
                axp.contour(ax, ax, hg, levels=[0.0], colors="black", linewidths=1.6)
            # overlays: obstacle vertical py-bands (px=0 cross-section)
            for j in np.nonzero(A)[0]:
                if abs(C[j, 0]) < R[j]:
                    half = math.sqrt(R[j] ** 2 - C[j, 0] ** 2)
                    axp.axvspan(C[j, 1] - half, C[j, 1] + half, color="0.4", alpha=0.18)
            axp.axhline(4.0, color="0.2", ls="-", lw=0.8); axp.axhline(-4.0, color="0.2", ls="-", lw=0.8)  # band |z|=4
            # analytic band-branch h_star=0 (dashed): floor pz=-4-c_z*vz, ceiling pz=4-c_z*vz
            for pz0 in (-4.0 - c_z * vz, 4.0 - c_z * vz):
                if -world <= pz0 <= world:
                    axp.axhline(pz0, color="magenta", ls="--", lw=1.3)
            # free-space py column = max obstacle clearance across py (isolates the vertical/band branch)
            clr = np.full(RES, np.inf)
            for j in np.nonzero(A)[0]:
                clr = np.minimum(clr, np.abs(ax - C[j, 1]) if abs(C[j, 0]) >= R[j]
                                 else np.maximum(np.abs(ax - C[j, 1]) - math.sqrt(max(R[j]**2 - C[j,0]**2, 0)), 0.0))
            fp = int(np.argmax(clr)); col_v = vg[:, fp]
            zc = [round(float(ax[k] + (ax[k+1]-ax[k]) * (0-col_v[k]) / (col_v[k+1]-col_v[k])), 3)
                  for k in range(RES-1) if (col_v[k] <= 0 <= col_v[k+1]) or (col_v[k] >= 0 >= col_v[k+1])]
            axp.axvline(ax[fp], color="lime", ls=":", lw=0.8)          # mark the free-space readout column
            # z-invariance: max |V_hat(py,pz) - V_hat(py,pz=0)| over the grid (control must be ~0)
            zinv = float(np.max(np.abs(vg - vg[RES // 2:RES // 2 + 1, :])))
            crossings[f"row{ri}_{label}"] = {"free_space_py": round(float(ax[fp]), 3),
                "vhat0_crossing_pz": zc, "analytic_band0_pz": [round(-4.0 - c_z * vz, 3), round(4.0 - c_z * vz, 3)],
                "z_invariance_max_dev": round(zinv, 4)}
            axp.set_xlim(-world, world); axp.set_ylim(-world, world); axp.set_aspect("equal")
            axp.set_xlabel("p_y"); axp.set_ylabel("p_z")
            axp.set_title(f"row{ri} {label}\nV_hat [{float(vhat.min()):.3f}, {float(vhat.max()):.3f}]", fontsize=9)
    fig.colorbar(cf, ax=axes, fraction=0.02, pad=0.02)
    fpath = OUT / f"yz_contour_{tag}.png"
    fig.savefig(fpath, bbox_inches="tight"); plt.close(fig)
    rec = {"tag": tag, "ckpt": str(ckpt), "step": int(ck["step"]), "c_z": round(c_z, 4),
           "figure": str(fpath), "vhat0_crossings": crossings}
    (OUT / f"yz_contour_{tag}.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(f"[{tag}] wrote {fpath.name}")
    print(json.dumps(crossings, indent=2))
    return rec


if __name__ == "__main__":
    render(Path(sys.argv[1]), sys.argv[2])
