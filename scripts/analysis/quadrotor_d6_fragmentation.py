"""v2.6.1 D6 — CBF contour + boundary-fragmentation across attitude/rate slices (read-only).

Renders the deployed certificate h(x)=V_hat on the position plane across theta x omega slices, for the
EARLY (~step 6000) and FINAL (best.pt) checkpoints, and computes a boundary-fragmentation metric of the
h=0 level set per slice (perimeter / area, plus an isoperimetric ratio P/(2 sqrt(pi A)); ~1 = one smooth
blob, >>1 = ragged/fragmented). Slices: theta in {-pi/2,0,pi/2} x omega in {-1,0,1}. On/off-manifold is
labelled at SLICE granularity from the visited |theta|,|omega| percentiles reported by the main diag
(theta~0, omega in [-1,1] are on-manifold; theta=+-pi/2 is a tail slice). Writes a figure to figures/ and
a JSON fragmentation table to scratchpad. No training / config / git change.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from src.common.value_net import ValueNetEnsemble, make_h_fn
from src.eval.build_pools import load_pool
from src.frameworks.jt_pncbf.train import make_system

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
SP = Path("/tmp/claude-1000/-home-junhyeok-MIT-jt-pncbf/31d93785-ac11-4206-bf50-a4c3de145dff/scratchpad")
POOL = REPO / "data/secured_data/pools/eval_full_quadrotor-planar_n2000_seed23456.pkl"
RES = 200
THETAS = [(-np.pi / 2, "-pi/2"), (0.0, "0"), (np.pi / 2, "+pi/2")]
OMEGAS = [(-1.0, "-1"), (0.0, "0"), (1.0, "+1")]
SCENE_ID = 1                       # dense multi-obstacle scene


def load_h(run_dir, ckpt, dev):
    ck = torch.load(run_dir / "checkpoints" / ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]; step = int(ck.get("step", -1))
    system = make_system(cfg)
    vnet = ValueNetEnsemble(system.obs_dim, cfg).to(dev, torch.float32)
    vnet.load_state_dict(ck["v_s_state"]); vnet.eval()
    for p in vnet.parameters():
        p.requires_grad_(False)
    return cfg, step, system, make_h_fn(vnet, system)


def frag_metrics(h, cell):
    """h [R,R] on a position grid; cell = world size of one cell. Returns area (world^2), perimeter
    (world length of h=0 level set, 4-neighbour sign-change edges), P/A, and isoperimetric ratio."""
    unsafe = h > 0.0
    area = float(unsafe.sum()) * cell * cell
    # sign-change edges between adjacent cells (horizontal + vertical)
    edges = int((unsafe[:, 1:] != unsafe[:, :-1]).sum() + (unsafe[1:, :] != unsafe[:-1, :]).sum())
    perim = edges * cell
    if area <= 0.0:
        return dict(area=0.0, perimeter=perim, P_over_A=0.0, isoperimetric=0.0, n_components=0)
    iso = perim / (2.0 * np.sqrt(np.pi * area))
    # connected components of the unsafe set (4-neighbour) as a raggedness proxy
    ncomp = _n_components(unsafe)
    return dict(area=area, perimeter=perim, P_over_A=perim / area, isoperimetric=float(iso),
                n_components=ncomp)


def _n_components(mask):
    m = mask.copy(); n = 0; R = m.shape[0]
    stack = []
    for i in range(R):
        for j in range(R):
            if m[i, j]:
                n += 1; stack.append((i, j)); m[i, j] = False
                while stack:
                    a, b = stack.pop()
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        x, y = a + da, b + db
                        if 0 <= x < R and 0 <= y < R and m[x, y]:
                            m[x, y] = False; stack.append((x, y))
    return n


def main():
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/v2.6.1__20260715-173808__seed42")
    if not run_dir.is_absolute():
        run_dir = REPO / run_dir
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    early_ck = "step_006000.pt"
    cfg, step_f, system, hf_final = load_h(run_dir, "best.pt", dev)
    _, step_e, _, hf_early = load_h(run_dir, early_ck, dev)

    world_lim = float(cfg["env"]["world_lim"]); goal_radius = float(cfg["env"]["goal_radius"])
    cell = 2.0 * world_lim / (RES - 1)
    ax_lin = torch.linspace(-world_lim, world_lim, RES, device=dev, dtype=torch.float32)
    gx, gy = torch.meshgrid(ax_lin, ax_lin, indexing="xy")
    pos = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    axn = ax_lin.cpu().numpy()

    scenes = load_pool(POOL).scenes; sc = scenes[SCENE_ID]
    active = np.asarray(sc.obstacle_active).astype(bool)
    centers = np.asarray(sc.obstacle_centers)[active]; radii = np.asarray(sc.obstacle_radii)[active]
    goal = np.asarray(sc.goal)
    scene_t = SimpleNamespace(
        obstacle_centers=torch.as_tensor(np.asarray(sc.obstacle_centers), dtype=torch.float32, device=dev),
        obstacle_radii=torch.as_tensor(np.asarray(sc.obstacle_radii), dtype=torch.float32, device=dev),
        obstacle_active=torch.as_tensor(np.asarray(sc.obstacle_active), dtype=torch.bool, device=dev),
        goal=torch.as_tensor(goal, dtype=torch.float32, device=dev))

    def h_slice(h_fn, th, om):
        B = pos.shape[0]
        x = torch.zeros(B, 6, device=dev, dtype=torch.float32)
        x[:, :2] = pos; x[:, 2] = th
        # velocity along thrust axis Re=(-sin th, cos th): set v=0 (hover slice) to isolate attitude/rate
        x[:, 5] = om
        with torch.no_grad():
            return h_fn(x, scene_t).reshape(RES, RES).cpu().numpy()

    # manifold labelling (from main diag visited percentiles): theta~0 on, +-pi/2 tail; omega in [-1,1] on
    def manifold(th, om):
        on_th = abs(th) < 0.6              # p50~0, p95~1.5 -> |th|<0.6 firmly on-manifold, pi/2 is tail
        on_om = abs(om) <= 4.0             # omega p95=4 -> |om|<=1 fully on-manifold
        return "on" if (on_th and on_om) else "tail"

    table = {"early": {"step": step_e}, "final": {"step": step_f}}
    for tag, h_fn in (("early", hf_early), ("final", hf_final)):
        for th, thl in THETAS:
            for om, oml in OMEGAS:
                h = h_slice(h_fn, th, om)
                m = frag_metrics(h, cell)
                m["hmin"] = float(h.min()); m["hmax"] = float(h.max())
                m["manifold"] = manifold(th, om)
                table[tag][f"theta={thl},omega={oml}"] = m

    json.dump(table, open(SP / "quadrotor_d6_fragmentation.json", "w"), indent=2)

    # figure: FINAL ckpt, 3x3 theta(rows) x omega(cols)
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0); LEV = np.linspace(-1, 1, 41)
    fig, axes = plt.subplots(len(THETAS), len(OMEGAS), figsize=(4.6 * 3, 4.4 * 3), dpi=130)
    fig.suptitle(f"v2.6.1 D6 · h(x)=V_hat · {run_dir.name}/best.pt @ step {step_f} · scene {SCENE_ID} "
                 f"({int(active.sum())} obs) · v=0 hover slice\nrows theta in {{-pi/2,0,+pi/2}}, cols omega "
                 f"in {{-1,0,+1}}; h<0 safe (blue), h=0 boundary (black). P/(2√(πA)) = fragmentation.", fontsize=11)
    mp = None
    for r, (th, thl) in enumerate(THETAS):
        for c, (om, oml) in enumerate(OMEGAS):
            ax = axes[r, c]; h = h_slice(hf_final, th, om); m = table["final"][f"theta={thl},omega={oml}"]
            mp = ax.contourf(axn, axn, h, levels=LEV, cmap="coolwarm", norm=norm, extend="both")
            if h.min() <= 0 <= h.max():
                ax.contour(axn, axn, h, levels=[0.0], colors="black", linewidths=1.4)
            for ctr, rad in zip(centers, radii):
                ax.add_patch(plt.Circle((ctr[0], ctr[1]), rad, fill=False, ec="k", lw=1.2, ls="--"))
            ax.plot(goal[0], goal[1], marker="*", ms=15, mfc="gold", mec="k", mew=1.0, zorder=5)
            ax.set_xlim(-world_lim, world_lim); ax.set_ylim(-world_lim, world_lim); ax.set_aspect("equal")
            ax.set_title(f"theta={thl}, omega={oml} [{m['manifold']}] · frag={m['isoperimetric']:.2f} "
                         f"ncomp={m['n_components']}", fontsize=8.5)
            if c == 0:
                ax.set_ylabel("py")
            if r == len(THETAS) - 1:
                ax.set_xlabel("px")
    fig.subplots_adjust(right=0.90, top=0.90, hspace=0.18, wspace=0.15)
    cax = fig.add_axes([0.92, 0.12, 0.014, 0.74]); fig.colorbar(mp, cax=cax, label="h=V_hat [-1,1]")
    out = run_dir / "figures" / "cbf_contour_d6_slices.png"; out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out}")
    print(json.dumps(table, indent=2))
    print(f"saved {SP / 'quadrotor_d6_fragmentation.json'}")


if __name__ == "__main__":
    main()
