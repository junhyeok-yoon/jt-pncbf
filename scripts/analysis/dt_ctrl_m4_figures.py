"""v2.7.5 M4.3/M4.4 — trajectory and field visualizations for the dt_ctrl arms. DIAGNOSTIC ONLY.

Reads ONLY recorded artifacts (states_<arm>.npz, action_stream_<arm>.npz, per_episode_<arm>.csv, and the
registered m4_scene_selection.json). Every annotated number is read from those artifacts; nothing is
recomputed into a different value. Time axes are ALWAYS in SECONDS (arm A has 200 steps and C has 1000 for the
same 10 s, so a step axis would misstate the comparison).

M4.3 per-scene 5-panel figure, B and C overlaid, A included: (a) top-down XY with obstacles to scale,
start/goal, collision point; (b) altitude vs time; (c) h_star and V-hat vs time with the zero line;
(d) per-rotor u_cmd solid / u_nom dashed with the [0, 4.905] box; (e) filter-active and empty-branch.
M4.4 (1) contact sheet of top-down paths, B|C side by side, tinted by RECORDED outcome; (2) field slices of
h_star and V-hat on an (x,y) grid at the vehicle's recorded altitude with the other coordinates held at their
recorded values at that instant.
"""
from __future__ import annotations

import csv, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

from src.common.quadrotor_barrier import make_barrier_fn
from src.common.value_net import make_h_fn
from src.eval.build_pools import load_pool, EVAL_POOLS_DIR
from src.envs.scene_batch import batch_scenes
from src.eval.run_full import _load_framework

RD = Path("data/runs/v2.7.5/dt_ctrl_arms")
FIG = RD / "figures"; FIG.mkdir(parents=True, exist_ok=True)
CKPT = Path("data/runs/v2.7.4/set__20260720-083533__seed42/v2.7.4__20260720-091830__seed42/checkpoints/best.pt")
POOL = EVAL_POOLS_DIR / "eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl"
ARMS = ["A_20Hz_coarse", "B_20Hz_fine", "C_100Hz_fine"]
COL = {"A_20Hz_coarse": "0.45", "B_20Hz_fine": "tab:blue", "C_100Hz_fine": "tab:red"}
LBL = {"A_20Hz_coarse": "A 20Hz coarse", "B_20Hz_fine": "B 20Hz fine", "C_100Hz_fine": "C 100Hz"}
OUTC_TINT = {"goal": "#e8f5e9", "collision": "#ffebee", "timeout": "#fff8e1", "stuck": "#ede7f6", "oob": "#e1f5fe"}
UMAX = 4.905

sel = json.loads((RD / "m4_scene_selection.json").read_text())
IDS = sel["selected_ids"]
RULE = sel["rule_per_id"]

scenes = load_pool(POOL).scenes
fw, cfg, _ = _load_framework(CKPT)
system = fw.system
c_gain = float(cfg["env"][system.name]["c_gain"]); h_scale = float(cfg["env"]["h_scale"])
h_star_fn = make_barrier_fn(c_gain, h_scale)
v_fn = make_h_fn(fw.value_net, system)

D = {}
for a in ARMS:
    z = np.load(RD / f"states_{a}.npz"); s = np.load(RD / f"action_stream_{a}.npz")
    rows = {int(r["episode_idx"]): r for r in csv.DictReader(open(RD / f"per_episode_{a}.csv"))}
    D[a] = {"states": z["states"], "interv": z["intervention"], "empty": z["empty"],
            "dt": float(z["dt"]), "u_cmd": s["u_cmd"], "u_nom": s["u_nom"], "n": s["n_steps"], "rows": rows}


def h_and_v(arm, ep, k):
    """h_star and V-hat along the recorded trajectory (batched over time)."""
    X = torch.tensor(D[arm]["states"][ep, :k + 1], dtype=torch.float32)
    bs = batch_scenes([scenes[ep]] * X.shape[0], device=torch.device("cpu"), dtype=torch.float32)
    with torch.no_grad():
        hs = h_star_fn(X, bs).numpy().reshape(-1)
        vh = v_fn(X, bs).numpy().reshape(-1)
    return hs, vh


def draw_obstacles(ax, sc):
    cen = np.asarray(sc.obstacle_centers, float); rad = np.asarray(sc.obstacle_radii, float)
    act = np.asarray(sc.obstacle_active, bool)
    for j in np.nonzero(act)[0]:
        ax.add_patch(plt.Circle((cen[j, 0], cen[j, 1]), rad[j], color="0.55", alpha=0.30, lw=0, zorder=1))


# ---------------- M4.3 per-scene figures ----------------
made = []
for ep in IDS:
    sc = scenes[ep]
    fig, axs = plt.subplots(5, 1, figsize=(9.2, 13.2), gridspec_kw={"height_ratios": [2.0, 1, 1.15, 1.25, 0.55]})
    ax_xy, ax_z, ax_h, ax_u, ax_f = axs
    for a in ARMS:
        k = int(min(D[a]["n"][ep], D[a]["states"].shape[1] - 1)); k = max(k, 1)
        dt = D[a]["dt"]; t = np.arange(k + 1) * dt
        S = D[a]["states"][ep, :k + 1]
        oc = D[a]["rows"][ep]["outcome"]
        ax_xy.plot(S[:, 0], S[:, 1], color=COL[a], lw=1.5, alpha=0.95, zorder=3,
                   label=f"{LBL[a]} → {oc}")
        if oc == "collision":
            ax_xy.plot(S[-1, 0], S[-1, 1], marker="x", ms=11, mew=2.6, color=COL[a], zorder=5)
            ax_z.plot(t[-1], S[-1, 2], marker="x", ms=9, mew=2.2, color=COL[a], zorder=5)
        ax_z.plot(t, S[:, 2], color=COL[a], lw=1.4)
        hs, vh = h_and_v(a, ep, k)
        ax_h.plot(t, hs, color=COL[a], lw=1.4, ls="-")
        ax_h.plot(t, vh, color=COL[a], lw=1.2, ls=":")
        u_c = D[a]["u_cmd"][ep, :k]; u_n = D[a]["u_nom"][ep, :k]; tu = np.arange(k) * dt
        if a != "A_20Hz_coarse":                       # keep panel (d) readable: B and C only
            for r in range(4):
                ax_u.plot(tu, u_c[:, r], color=COL[a], lw=0.85, alpha=0.85)
                ax_u.plot(tu, u_n[:, r], color=COL[a], lw=0.7, ls="--", alpha=0.45)
            iv = D[a]["interv"][ep, :k].astype(float); em = D[a]["empty"][ep, :k].astype(float)
            off = 0.0 if a == "B_20Hz_fine" else 1.0
            ax_f.fill_between(tu, off, off + iv * 0.42, color=COL[a], alpha=0.55, lw=0, step="post")
            ax_f.plot(tu[em > 0], np.full(int((em > 0).sum()), off + 0.62), ls="none", marker="|",
                      ms=5, color="magenta", alpha=0.85)
    st = np.asarray(sc.start, float); gl = np.asarray(sc.goal, float)
    draw_obstacles(ax_xy, sc)
    ax_xy.plot(st[0], st[1], "o", ms=8, mfc="w", mec="k", mew=1.6, zorder=6)
    ax_xy.plot(gl[0], gl[1], "*", ms=15, mfc="gold", mec="k", mew=0.9, zorder=6)
    ax_xy.set_aspect("equal"); ax_xy.set_xlabel("x (m)"); ax_xy.set_ylabel("y (m)")
    ax_xy.legend(fontsize=8, loc="best"); ax_xy.grid(alpha=0.25)
    ax_xy.set_title(f"(a) top-down XY — obstacles to scale (infinite vertical cylinders ⇒ circular section); ○ start, ★ goal, ✕ collision", fontsize=9)
    ax_z.set_ylabel("altitude z (m)"); ax_z.set_xlabel("time (s)"); ax_z.grid(alpha=0.25)
    ax_z.set_title("(b) altitude vs time", fontsize=9)
    ax_h.axhline(0.0, color="k", lw=1.0, ls="-", alpha=0.8)
    ax_h.set_ylabel("h*, V̂"); ax_h.set_xlabel("time (s)"); ax_h.grid(alpha=0.25)
    ax_h.set_title("(c) h* (solid) and V̂ (dotted) vs time — zero line drawn", fontsize=9)
    ax_u.axhline(0.0, color="k", lw=0.9, ls="--"); ax_u.axhline(UMAX, color="k", lw=0.9, ls="--")
    ax_u.set_ylim(-0.35, UMAX + 0.35); ax_u.set_ylabel("rotor cmd (N)"); ax_u.set_xlabel("time (s)")
    ax_u.set_title("(d) per-rotor u_cmd (solid) / u_nom (dashed), 4 rotors, box [0, 4.905]", fontsize=9)
    ax_f.set_yticks([0.2, 1.2]); ax_f.set_yticklabels(["B", "C"]); ax_f.set_ylim(-0.1, 1.75)
    ax_f.set_xlabel("time (s)"); ax_f.set_title("(e) filter-active (bars) · empty-branch (magenta ticks)", fontsize=9)
    fig.suptitle(f"v2.7.5 M4 · scene {ep} · rule: {RULE[str(ep)]} · DIAGNOSTIC ONLY (pinned batch 2000; "
                 f"no number revised)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    p = FIG / f"m4_scene_{ep:04d}.png"; fig.savefig(p, dpi=105); plt.close(fig); made.append(str(p))
    print("wrote", p, flush=True)

# ---------------- M4.4.1 contact sheet ----------------
ncols = 6; nrows = int(np.ceil(len(IDS) / ncols))
for arm in ("B_20Hz_fine", "C_100Hz_fine"):
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.35 * ncols, 2.5 * nrows), squeeze=False)
    for idx, ep in enumerate(IDS):
        ax = axes[idx // ncols][idx % ncols]
        sc = scenes[ep]; oc = D[arm]["rows"][ep]["outcome"]
        ax.set_facecolor(OUTC_TINT.get(oc, "#ffffff"))
        draw_obstacles(ax, sc)
        k = int(min(D[arm]["n"][ep], D[arm]["states"].shape[1] - 1))
        S = D[arm]["states"][ep, :max(k, 1) + 1]
        ax.plot(S[:, 0], S[:, 1], color=COL[arm], lw=1.2)
        if oc == "collision":
            ax.plot(S[-1, 0], S[-1, 1], "x", ms=7, mew=2.0, color="k")
        ax.plot(sc.start[0], sc.start[1], "o", ms=3.5, mfc="w", mec="k", mew=1.0)
        ax.plot(sc.goal[0], sc.goal[1], "*", ms=8, mfc="gold", mec="k", mew=0.5)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{ep} · {oc}", fontsize=7.5)
    for j in range(len(IDS), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(f"v2.7.5 M4.4 contact sheet — arm {LBL[arm]} · panel tint = RECORDED outcome "
                 f"(green reach / red collision / amber timeout / violet stuck / blue oob) · DIAGNOSTIC ONLY", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = FIG / f"m4_contact_sheet_{arm}.png"; fig.savefig(p, dpi=110); plt.close(fig); made.append(str(p))
    print("wrote", p, flush=True)

# ---------------- M4.4.2 field slices ----------------
fwd = sel["rule1_forward_flips_B_collide_C_reach"]["ids"]
rev = sel["rule1_reverse_flips_C_collide_B_reach"]["ids"]
med = sel["rule3_median_cps"]["ids"]
picks = [(fwd[0], "forward flip (B collide → C reach)") if fwd else None,
         (rev[0], "reverse flip (C collide → B reach)") if rev else None,
         (med[0], "median cps") if med else None]
picks = [p for p in picks if p]
G = 121
for ep, why in picks:
    sc = scenes[ep]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.0))
    for col, arm in enumerate(("B_20Hz_fine", "C_100Hz_fine")):
        k = int(min(D[arm]["n"][ep], D[arm]["states"].shape[1] - 1)); k = max(k, 1)
        S = D[arm]["states"][ep, :k + 1]
        kk = k - 1 if D[arm]["rows"][ep]["outcome"] == "collision" else k // 2
        kk = int(np.clip(kk, 0, k))
        ref = S[kk]                                             # recorded state at that instant
        z0 = float(ref[2])
        W = float(cfg["env"]["world_lim"])
        gx = np.linspace(-W, W, G); gy = np.linspace(-W, W, G)
        GX, GY = np.meshgrid(gx, gy, indexing="ij")
        X = np.tile(ref, (G * G, 1)).astype(np.float64)
        X[:, 0] = GX.ravel(); X[:, 1] = GY.ravel(); X[:, 2] = z0    # vary x,y; hold z and ALL other coords
        Xt = torch.tensor(X, dtype=torch.float32)
        bsg = batch_scenes([sc] * (G * G), device=torch.device("cpu"), dtype=torch.float32)
        with torch.no_grad():
            HS = h_star_fn(Xt, bsg).numpy().reshape(G, G)
            VH = v_fn(Xt, bsg).numpy().reshape(G, G)
        for row, (F, nm) in enumerate(((HS, "h*"), (VH, "V̂"))):
            ax = axes[row][col]
            m = float(np.abs(F).max())
            im = ax.pcolormesh(GX, GY, F, cmap="RdBu_r", vmin=-m, vmax=m, shading="auto")
            ax.contour(GX, GY, F, levels=[0.0], colors="k", linewidths=1.6)
            draw_obstacles(ax, sc)
            ax.plot(S[:, 0], S[:, 1], color=COL[arm], lw=1.5)
            ax.plot(ref[0], ref[1], "o", ms=6, mfc="w", mec="k", mew=1.4)
            ax.set_aspect("equal"); ax.set_title(f"{nm} · {LBL[arm]} · z={z0:.2f} m, t={kk*D[arm]['dt']:.2f} s", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"v2.7.5 M4.4 field slice — scene {ep} ({why}). (x, y) grid at the vehicle's RECORDED altitude; "
                 f"velocity, attitude and body rates held at their recorded values at that instant.\n"
                 f"V̂ is POLICY-CONDITIONED: this is a slice under the deployed policy, NOT a level set of the "
                 f"true safe set. Black line = zero contour. DIAGNOSTIC ONLY.", fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = FIG / f"m4_field_slice_{ep:04d}.png"; fig.savefig(p, dpi=110); plt.close(fig); made.append(str(p))
    print("wrote", p, flush=True)

(RD / "m4_figures_index.json").write_text(json.dumps({"figures": made, "n": len(made)}, indent=2) + "\n")
print(f"DONE — {len(made)} figures")
