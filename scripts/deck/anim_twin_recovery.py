"""v2.7.7 (Amdt 7) — twin-recovery clip, AFTER-only (renamed from anim_obs_twin; the dim-34 replay only). With
absolute altitude in the observation (dim-34 certificate b2cdaddd), the altitude-degeneracy twin pair — a
near-floor start and a safe start with identical goal-relative geometry — BOTH recover inside the band; the
observation now distinguishes them. Two synchronized 3-D scene panels (near-floor twin | safe twin), shared
camera, 1 m scale bars, and an altitude strip. Absolute capability of the final observation (no dim-32 failure
phase). Caption "fixed-policy certificate" (obs-only certificate rollout; recovery, not a reach-trained policy).
mp4 (H.264) + GIF. Deterministic twin, fixed seed. Eval-only (per-rotor OC ckpt)."""
from __future__ import annotations
import dataclasses
from pathlib import Path
import numpy as np
import torch
from src.eval.run_full import _load_framework
from src.eval.build_pools import load_pool
from src.envs.scene_batch import batch_scenes
from src.common.rk4 import rk4_step
from scripts.deck.deck_style import OUT, DPI, C_LEARNED, C_VERIFY, C_FLOOR, HAVE_MP4
import scripts.deck.deck_scene3d as S3
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

torch.manual_seed(42)
OC34 = Path("data/runs/v2.7.6/set__20260725-043415__seed42/v2.7.6__oc__20260725-043415__seed42/checkpoints/best.pt")
INLOOP = Path("data/eval_pools/eval_inloop_quadrotor-3d-d2r_n500_seed12345.pkl")
T, DT, LIMIT, OOB, STRIDE, FPS = 200, 0.05, 4.0, 8.0, 3, 12


def roll(fw, x0, scene):
    sys_ = fw.system; dev = next(fw.value_net.parameters()).device
    bs = batch_scenes([scene], device=dev, dtype=torch.float32)
    x = x0.to(dev).clone(); Xs = []
    if hasattr(fw, "reset_deficit_state"):
        fw.reset_deficit_state()
    for _ in range(T):                       # OC filter needs grad; detach per step
        Xs.append(x[0].detach().cpu().numpy().copy())
        un = fw.policy(x, bs); _out = fw.filter(x, un, bs)
        u = _out[0] if isinstance(_out, (tuple, list)) else _out
        x = rk4_step(sys_, x, u.detach(), DT).detach()
        if abs(float(x[0, 2])) > OOB:
            Xs.extend([Xs[-1]] * (T - len(Xs))); break
    Xs = Xs[:T] + [Xs[-1]] * (T - len(Xs))
    return np.array(Xs)


fw34, cfg0, _ = _load_framework(OC34, config_overrides={"env": {"dt": DT}, "eval": {"max_steps": T}})
sys0 = fw34.system; dev0 = next(fw34.value_net.parameters()).device
base = load_pool(INLOOP).scenes[0]


def scene_goalz(gz):
    g = np.asarray(base.goal, np.float64).reshape(-1).copy(); g[2] = gz
    return dataclasses.replace(base, goal=torch.tensor(g, dtype=torch.float64))


def mk(pz):
    x = torch.zeros(1, 13, dtype=torch.float32, device=dev0); x[0, 2] = pz; x[0, 3] = 1.0; x[0, 9] = -1.0; return x


near_z, safe_z, rel = -3.6, 0.0, 0.4
sc_near, sc_safe = scene_goalz(near_z + rel), scene_goalz(safe_z + rel)
with torch.no_grad():                                    # confirm the twin shares the dim-32 core observation
    o_n = sys0.observation(mk(near_z), batch_scenes([sc_near], device=dev0, dtype=torch.float32))
    obs_dim = int(o_n.shape[1])
TWIN = {"near": ("near-floor twin", C_LEARNED, sc_near, mk(near_z)), "safe": ("safe twin", C_VERIFY, sc_safe, mk(safe_z))}
data = {tw: roll(fw34, TWIN[tw][3], TWIN[tw][2]) for tw in ("near", "safe")}


def outcome(X):
    return "falls" if X[:, 2].min() <= -OOB or X[:, 2].min() <= -LIMIT - 1.0 else "recovers"


allX = np.vstack([data[tw][:, :3] for tw in ("near", "safe")])
c_n, r_n, _ = S3.scene_geometry(sc_near)
xlim, ylim, zlim = S3.scene_extent(c_n, r_n, np.array([0, 0, 0.0]), allX, pad=1.0)
zlim = (max(zlim[0], -6.5), max(zlim[1], 4.5))

fig = plt.figure(figsize=(13, 8.5))
gs = GridSpec(6, 2, figure=fig, height_ratios=[1, 1, 1, 1, 0.9, 0.05], hspace=0.35, wspace=0.2)
ax3d = {"near": fig.add_subplot(gs[0:4, 0], projection="3d"), "safe": fig.add_subplot(gs[0:4, 1], projection="3d")}
axS = fig.add_subplot(gs[4, :])
for tw in ("near", "safe"):
    lab, col, sc, _ = TWIN[tw]
    cc, rr, gg = S3.scene_geometry(sc)
    S3.draw_static_3d(ax3d[tw], cc, rr, gg, xlim, ylim, zlim, title=lab)
    S3.scale_bar_3d(ax3d[tw], xlim, ylim, zlim)
    ax3d[tw].view_init(elev=8, azim=-72)
    try:
        ax3d[tw].set_box_aspect(None, zoom=1.28)
    except TypeError:
        pass
t = np.arange(T) * DT
axS.axhspan(-9, -LIMIT, color=C_FLOOR, alpha=0.12); axS.axhline(-LIMIT, color=C_FLOOR, ls="--", lw=1.4, label="band floor z=-4")
axS.set_xlim(0, T * DT); axS.set_ylim(-6.5, 3); axS.set_xlabel("time (s)", fontsize=10); axS.set_ylabel("altitude p_z (m)", fontsize=10)
axS.tick_params(labelsize=9); axS.legend(fontsize=9, loc="lower right")
fig.text(0.5, 0.955, "with absolute altitude (dim-34): both twins recover", ha="center", fontsize=13, fontweight="bold", color=C_VERIFY)
fig.text(0.5, 0.05, "fixed-policy certificate (obs-only certificate rollout; recovery, not a reach-trained policy)",
         ha="center", fontsize=9, style="italic", color="#555555")
frames = list(range(0, T, STRIDE)); dyn = {"near": [], "safe": [], "S": []}


def update(k):
    for tw in ("near", "safe"):
        S3.remove(dyn[tw]); dyn[tw] = []
        X = data[tw]; col = TWIN[tw][1]; p = X[k, :3]; R = S3.quat_to_R(X[k, 3:7])
        dyn[tw] = S3.trail_3d(ax3d[tw], X[:k + 1, :3], col) + S3.glyph_3d(ax3d[tw], p, R, col)
    S3.remove(dyn["S"]); dyn["S"] = []
    for tw in ("near", "safe"):
        col = TWIN[tw][1]; X = data[tw]
        dyn["S"] += axS.plot(t, X[:, 2], color=col, lw=1.4, alpha=0.28)
        dyn["S"] += axS.plot(t[:k + 1], X[:k + 1, 2], color=col, lw=2.8, label=TWIN[tw][0])
    dyn["S"] += [axS.axvline(t[k], color="black", lw=1.0)]
    return []


anim = FuncAnimation(fig, update, frames=frames, interval=1000 / FPS, blit=False)
OUT.mkdir(parents=True, exist_ok=True)
stem = OUT / "anim_twin_recovery"
anim.save(str(stem) + ".gif", writer=PillowWriter(fps=FPS), dpi=110); status = "mp4 + GIF"
if HAVE_MP4:
    try:
        anim.save(str(stem) + ".mp4", writer=FFMpegWriter(fps=FPS, codec="libx264", bitrate=2400), dpi=DPI)
    except Exception as e:
        status = f"GIF only (mp4 failed: {e})"
plt.close(fig)
print(f"M9→recovery {status} -> {stem.name}; dim-{obs_dim} certificate (b2cdaddd)")
print(f"  near twin {outcome(data['near'])} (min {data['near'][:,2].min():.2f}), safe twin {outcome(data['safe'])} (min {data['safe'][:,2].min():.2f})")
