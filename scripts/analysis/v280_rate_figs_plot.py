"""v2.8.0 rate figures F1-F5 from the re-rolled per-step data (roll_<arm>_<proj>.npz + scene_info.npz).
Numbers only; no interpretation. PNGs under data/runs/v2.8.0/rate_figs/."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
FIG = REPO / "data/runs/v2.8.0/rate_figs"
S3 = REPO / "data/runs/v2.8.0/s3_eval"
RATES = [("A", 20), ("C", 100), ("D", 500)]                 # 20 / 100 / 500 Hz
RC = {"A": "#1f77b4", "C": "#ff7f0e", "D": "#2ca44e"}       # color per rate
ROTORC = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
FMAX = 4.905
roles = ["clean1", "clean2", "recovery", "floor"]
ROLE_TITLE = {"clean1": "clean reach A", "clean2": "clean reach B", "recovery": "recovery (θ₀ 174.5°)",
              "floor": "floor collision"}

info = np.load(FIG / "scene_info.npz", allow_pickle=True)
IDX = info["idx"]
rolls = {f"{a}_{p}": np.load(FIG / f"roll_{a}_{p}.npz") for a, _ in RATES for p in ("enumerate", "dual_solve")}
tvmed = {}
for a, _ in RATES:
    for p in ("enumerate", "dual_solve"):
        tvmed[f"{a}_{p}"] = json.loads((S3 / f"rate_{a}_{p}.json").read_text())["tv_per_s"]["median"]


def tvec(d, n):
    return np.arange(n) * float(d["dt_sim"])


# ---------- F1: command stream, per rotor, per (rate x projection) cell, per episode ----------
for ei, role in enumerate(roles):
    fig, axes = plt.subplots(2, 3, figsize=(15, 6), sharey=True)
    for r, (arm, hz) in enumerate(RATES):
        for c, proj in enumerate(("enumerate", "dual_solve")):
            ax = axes[c, r]; d = rolls[f"{arm}_{proj}"]; u = d["u"][:, ei, :]
            t = tvec(d, u.shape[0])
            for k in range(4):
                ax.plot(t, u[:, k], color=ROTORC[k], lw=0.6, label=f"rotor {k}" if (r == 0 and c == 0) else None)
            ax.set_title(f"{hz} Hz · {proj}  |  TV/s med {tvmed[f'{arm}_{proj}']:.1f}", fontsize=9)
            ax.set_xlim(0, 10); ax.set_ylim(-0.1, FMAX + 0.1)
            if c == 1:
                ax.set_xlabel("time (s)")
            if r == 0:
                ax.set_ylabel(f"{proj}\nper-rotor thrust (N)")
    axes[0, 0].legend(fontsize=7, ncol=4, loc="upper right")
    fig.suptitle(f"F1 command stream — episode {int(IDX[ei])} ({ROLE_TITLE[role]})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(FIG / f"F1_{role}.png", dpi=110); plt.close(fig)
    print(f"F1_{role}.png")

# ---------- F2: 0.5 s zoom where enumerate command variation is highest ----------
# find (episode, window-start) maximizing D-enumerate TV over a 0.5 s window
de = rolls["D_enumerate"]; du_d = de["u"]; dt_d = float(de["dt_sim"]); win = int(round(0.5 / dt_d))
best = (-1.0, 0, 0)
for ei in range(len(roles)):
    tv_step = np.linalg.norm(np.diff(du_d[:, ei, :], axis=0), axis=1)   # [T-1]
    csum = np.concatenate([[0], np.cumsum(tv_step)])
    for s in range(0, len(tv_step) - win, max(1, win // 4)):
        w = csum[min(s + win, len(csum) - 1)] - csum[s]
        if w > best[0]:
            best = (w, ei, s)
_, zei, zs0 = best; zt0 = zs0 * dt_d
fig, axes = plt.subplots(2, 3, figsize=(15, 6), sharey=True)
for r, (arm, hz) in enumerate(RATES):
    for c, proj in enumerate(("enumerate", "dual_solve")):
        ax = axes[c, r]; d = rolls[f"{arm}_{proj}"]; u = d["u"][:, zei, :]; t = tvec(d, u.shape[0])
        m = (t >= zt0) & (t <= zt0 + 0.5)
        for k in range(4):
            ax.plot(t[m], u[m, k], color=ROTORC[k], lw=0.8)
        ax.set_title(f"{hz} Hz · {proj}", fontsize=9); ax.set_xlim(zt0, zt0 + 0.5); ax.set_ylim(-0.1, FMAX + 0.1)
        if c == 1:
            ax.set_xlabel("time (s)")
        if r == 0:
            ax.set_ylabel(f"{proj}\nthrust (N)")
fig.suptitle(f"F2 zoom 0.5 s — episode {int(IDX[zei])} ({ROLE_TITLE[roles[zei]]}), window [{zt0:.2f},{zt0+0.5:.2f}] s "
             f"(max enumerate TV window)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(FIG / "F2_zoom.png", dpi=110); plt.close(fig)
print(f"F2_zoom.png (episode {int(IDX[zei])}, window {zt0:.2f}s)")

# ---------- F3: 3-D trajectory per episode, 3 rates overlaid, one axes per projection ----------
def draw_scene(ax, ei):
    centers = info["obstacle_centers"][ei]; radii = info["obstacle_radii"][ei]; active = info["obstacle_active"][ei]
    zc = np.linspace(-4, 4, 2); th = np.linspace(0, 2 * np.pi, 24)
    for j in range(len(radii)):
        if not bool(active[j]):
            continue
        cx, cy = centers[j][0], centers[j][1]; rr = float(radii[j])
        TH, ZC = np.meshgrid(th, zc)
        ax.plot_surface(cx + rr * np.cos(TH), cy + rr * np.sin(TH), ZC, color="gray", alpha=0.18, linewidth=0)
    # floor & ceiling planes
    xx, yy = np.meshgrid(np.linspace(-8, 8, 2), np.linspace(-8, 8, 2))
    ax.plot_surface(xx, yy, -4 + 0 * xx, color="brown", alpha=0.08)
    ax.plot_surface(xx, yy, 4 + 0 * xx, color="blue", alpha=0.06)

for ei, role in enumerate(roles):
    fig = plt.figure(figsize=(14, 6))
    for c, proj in enumerate(("enumerate", "dual_solve")):
        ax = fig.add_subplot(1, 2, c + 1, projection="3d")
        for arm, hz in RATES:
            d = rolls[f"{arm}_{proj}"]; p = d["pos"][:, ei, :]
            ax.plot(p[:, 0], p[:, 1], p[:, 2], color=RC[arm], lw=1.1, label=f"{hz} Hz")
            coll = d["band_lower"][:, ei] | d["obstacle"][:, ei] | d["band_upper"][:, ei]
            if coll.any():
                s = int(np.argmax(coll)); ax.scatter(p[s, 0], p[s, 1], p[s, 2], color=RC[arm], marker="x", s=60)
        st = info["start"][ei]; gl = info["goal"][ei]
        ax.scatter(*st, color="black", marker="o", s=40, label="start")
        ax.scatter(*gl, color="green", marker="*", s=120, label="goal")
        draw_scene(ax, ei)
        ax.set_title(f"{proj}", fontsize=10); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_zlim(-4.5, 4.5)
        if c == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f"F3 trajectory — episode {int(IDX[ei])} ({ROLE_TITLE[role]}); ✕ = collision", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / f"F3_{role}.png", dpi=110); plt.close(fig)
    print(f"F3_{role}.png")

# ---------- F4: tilt & altitude vs time for the recovery episode ----------
rei = roles.index("recovery")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
for arm, hz in RATES:
    for proj, ls in (("enumerate", "-"), ("dual_solve", "--")):
        d = rolls[f"{arm}_{proj}"]; t = tvec(d, d["tilt"].shape[0])
        a1.plot(t, d["tilt"][:, rei], color=RC[arm], ls=ls, lw=1.0, label=f"{hz}Hz {proj[:4]}")
        a2.plot(t, d["pos"][:, rei, 2], color=RC[arm], ls=ls, lw=1.0)
a1.axhline(60, color="k", ls=":", lw=1); a1.set_title("tilt angle (°)"); a1.set_xlabel("time (s)"); a1.set_ylabel("tilt (°)")
a1.legend(fontsize=7, ncol=3); a1.set_xlim(0, 10)
a2.axhline(-4, color="brown", ls=":", lw=1); a2.axhline(4, color="blue", ls=":", lw=1)
a2.set_title("altitude z (m)"); a2.set_xlabel("time (s)"); a2.set_ylabel("z (m)"); a2.set_xlim(0, 10)
fig.suptitle(f"F4 tilt & altitude — recovery episode {int(IDX[rei])} (θ₀ 174.5°); dotted = 60° cone / floor / ceiling", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(FIG / "F4_recovery.png", dpi=110); plt.close(fig)
print("F4_recovery.png")

# ---------- F5: dual_solve command TV per step, empty-branch steps shaded ----------
# episode with the most empty-branch steps across the dual cells
emp_counts = [sum(int(rolls[f"{a}_dual_solve"]["empty"][:, e].sum()) for a, _ in RATES) for e in range(len(roles))]
fei = int(np.argmax(emp_counts)); EMPTY_SHARE = {"A": 0.234, "C": 0.339, "D": 0.510}
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
for r, (arm, hz) in enumerate(RATES):
    ax = axes[r]; d = rolls[f"{arm}_dual_solve"]; u = d["u"][:, fei, :]; em = d["empty"][:, fei]
    t = tvec(d, u.shape[0]); tv = np.concatenate([[0.0], np.linalg.norm(np.diff(u, axis=0), axis=1)])
    ax.plot(t, tv, color=RC[arm], lw=0.7)
    # shade empty-branch steps
    in_e = False; s0 = 0
    for i in range(len(em)):
        if em[i] and not in_e:
            in_e = True; s0 = i
        elif not em[i] and in_e:
            in_e = False; ax.axvspan(t[s0], t[i], color="red", alpha=0.15)
    if in_e:
        ax.axvspan(t[s0], t[-1], color="red", alpha=0.15)
    ax.set_title(f"{hz} Hz dual_solve — aggregate empty-branch TV share {EMPTY_SHARE[arm]:.3f} (red = empty-branch steps)", fontsize=9)
    ax.set_ylabel("‖Δu‖ per step"); ax.set_xlim(0, 10)
axes[-1].set_xlabel("time (s)")
fig.suptitle(f"F5 residual chatter (dual_solve) — episode {int(IDX[fei])} ({ROLE_TITLE[roles[fei]]})", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(FIG / "F5_residual.png", dpi=110); plt.close(fig)
print(f"F5_residual.png (episode {int(IDX[fei])})")

# record selection + F2/F5 episode choices for the report
(FIG / "selection.json").write_text(json.dumps({
    "episodes": {roles[i]: int(IDX[i]) for i in range(len(roles))},
    "F2_zoom_episode": int(IDX[zei]), "F2_window_s": [round(zt0, 3), round(zt0 + 0.5, 3)],
    "F5_episode": int(IDX[fei]), "empty_step_counts_by_episode": {roles[i]: emp_counts[i] for i in range(len(roles))},
}, indent=2) + "\n")
print("ALL FIGURES DONE")
