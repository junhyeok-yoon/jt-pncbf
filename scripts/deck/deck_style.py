"""v2.7.7 deck assets — shared style contract. Colors: blue=design-bought, orange=learned/delegated,
green=measured-verification; surfaces cylinder=gray, floor=red, ceiling=purple. PNG 200 dpi. Eval-only helper
module (no src/ or config edits)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DPI = 200
OUT = Path("/home/junhyeok/MIT/jt-pncbf/data/runs/v2.7.7/deck_assets")

# v2.7.7 amendment 1: bundled static ffmpeg for mp4 animations (no system change).
try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    HAVE_MP4 = True
except Exception:
    HAVE_MP4 = False

# Amendment 5 reach-first reframing: drop competitive comparator/ours wording; frame as the vertical extension
# (adding absolute (p_z, v_z) to the observation AND the hazard). Short labels for panels; long for captions.
BEFORE = "before vertical extension"
AFTER = "after vertical extension"
# Amendment 7: the two bar charts keep both groups (the counts ARE the measurement) but use neutral,
# non-competitive wording. All other assets are after-only (absolute capability of the final system).
WITHOUT_TERM = "without the vertical term"
WITH_TERM = "with the vertical term"
BEFORE_LONG = "before adding (p_z, v_z) to obs & hazard"
AFTER_LONG = "after adding (p_z, v_z) to obs & hazard"
COMPARATOR = BEFORE      # kept as import aliases; the visible text is now before/after (no competition)
OURS = AFTER

# semantic colors
C_DESIGN = "#1f5fbf"   # blue  — design-bought
C_LEARNED = "#e07b00"  # orange — learned / delegated
C_VERIFY = "#2e8b57"   # green — measured verification
# surface colors
C_CYL = "#7f7f7f"      # gray   — cylinder
C_FLOOR = "#d62728"    # red    — floor
C_CEIL = "#7e57c2"     # purple — ceiling
plt.rcParams["font.family"] = "DejaVu Sans"


def save(fig, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p
