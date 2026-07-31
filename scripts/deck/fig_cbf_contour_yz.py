"""v2.7.7 M17 (Amdt 6/7) — yz CBF-contour slide asset: THE one before/after pair (the only asset that carries the
contrast). One decisive slice condition (hover, descending v_z=-1.5). SELF-CONTAINED (Amdt 7): regenerates its two
source panels to a scratch dir — the before (244f4f83) 2x6 grid via the frozen renderer
src.eval.plotting.plot_quadrotor3d_yz_contour, and the recorded after (09c33bf4) grid copied from the v2.7.6
artifact — then image-crops grid col0 (hover v_z=-1.5), scene row 0, from each and lays them side by side. No
deck-assets appendix files are required. Message: after the vertical extension a strong V̂>0 barrier forms just
below z=±4; before, none (altitude-blind). Every panel states the FULL slice condition."""
from __future__ import annotations
import shutil
from pathlib import Path
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
from src.eval.run_full import _load_framework
from src.eval.build_pools import load_pool
from src.eval.plotting import plot_quadrotor3d_yz_contour
from scripts.deck.deck_style import save, BEFORE, AFTER

SCRATCH = Path("/tmp/claude-1000/-home-junhyeok-MIT-jt-pncbf/31d93785-ac11-4206-bf50-a4c3de145dff/scratchpad/m17_src")
CK_PRE = Path("data/runs/v2.7.4/set__20260720-083533__seed42/v2.7.4__20260720-091830__seed42/checkpoints/best.pt")
JT_YZ = Path("data/previous_runs/v2.7.6/stage2_eval/yz_headline_step42000.png")   # recorded after (09c33bf4) grid
INLOOP = Path("data/eval_pools/eval_inloop_quadrotor-3d-d2r_n500_seed12345.pkl")
CMAP, NORM = "coolwarm", TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
SLICE = ("slice: p_x=0 m, (p_y, p_z) swept  ·  v=(v_x, v_y, v_z)=(0, 0, −1.5) m/s  ·  "
         "tilt 0° (hover, q=[1,0,0,0])  ·  ω=(0, 0, 0) rad/s")
NOTE = "all other velocity and rate components are held at zero; only (p_y, p_z) vary within each panel"


def _regen_sources():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    pre = SCRATCH / "yz_before.png"; jt = SCRATCH / "yz_after.png"
    fw, cfg, _ = _load_framework(CK_PRE)
    cfg = dict(cfg); cfg["env"] = dict(cfg["env"]); cfg["env"]["band_hazard"] = {"enabled": True, "limit": 4.0}
    sc = load_pool(INLOOP).scenes[:2]
    plot_quadrotor3d_yz_contour(fw.system, fw.value_net, cfg, sc, pre, role="before yz")
    shutil.copyfile(JT_YZ, jt)
    return pre, jt


def _content(im):
    return im[..., :3].min(axis=2) < 0.92


def _runs(mask, minw):
    runs, s = [], None
    for x in range(len(mask)):
        if mask[x] and s is None:
            s = x
        elif not mask[x] and s is not None:
            runs.append((s, x)); s = None
    if s is not None:
        runs.append((s, len(mask)))
    return [(a, b) for a, b in runs if b - a > minw]


def crop_col0_row0(im):
    H, W = im.shape[:2]; c = _content(im)
    rows = np.where(c.mean(axis=1) > 0.15)[0]; r0, r1 = rows.min(), rows.max()
    cols = _runs(c[r0:r1].mean(axis=0) > 0.12, W * 0.03)            # 6 columns; col0 = hover v_z=-1.5
    a, b = cols[0]
    rowruns = _runs(c[:, a:b].mean(axis=1) > 0.35, H * 0.10)        # dense contour band; row 0 = scene 0
    ra, rb = rowruns[0]; side = min(b - a, rb - ra)
    return im[ra:ra + side, a:a + side]


pre, jt = _regen_sources()
imgs = {BEFORE: mpimg.imread(pre), AFTER: mpimg.imread(jt)}
fig, axes = plt.subplots(1, 2, figsize=(11.0, 6.2))
for ax, arm in zip(axes, [BEFORE, AFTER]):
    ax.imshow(crop_col0_row0(imgs[arm])); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(arm, fontsize=13, fontweight="bold")
    ax.set_xlabel("p_y (m)  →", fontsize=9); ax.set_ylabel("p_z (m)  ↑", fontsize=9)
fig.suptitle("Certificate yz-section (hover, descending): a band barrier forms only after the vertical extension",
             fontsize=13, fontweight="bold", y=0.99)
fig.text(0.5, 0.15, SLICE, ha="center", fontsize=9.5)
fig.text(0.5, 0.115, NOTE, ha="center", fontsize=8.5, style="italic", color="#555555")
fig.text(0.5, 0.075, "band floor/ceiling z=±4 (dark lines), analytic h*=0 (magenta); V̂>0 (red) = certified unsafe, "
         "V̂<0 (blue) = certified safe.", ha="center", fontsize=8.5, style="italic", color="#555555")
fig.subplots_adjust(left=0.05, right=0.87, top=0.90, bottom=0.24, wspace=0.10)
cax = fig.add_axes([0.89, 0.28, 0.02, 0.56])
fig.colorbar(ScalarMappable(norm=NORM, cmap=CMAP), cax=cax, label="V̂ (clamped [-1,1])")
p = save(fig, "fig_cbf_contour_yz.png")
print(f"M17 -> {p.name}; before|after single pair at hover v_z=-1.5 (self-contained: sources regenerated to scratch)")
