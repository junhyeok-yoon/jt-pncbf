"""v2.7.7 M1 (smoke gate) — per-surface collision counts, pre-band (244f4f83) vs JT (09c33bf4), banded scoring,
kstep, canonical pool. Every rendered label is asserted equal to the canonical_eval.json value (the gate)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.deck.deck_style import save, C_CYL, C_FLOOR, C_CEIL, WITHOUT_TERM, WITH_TERM
import matplotlib.pyplot as plt

SRC = Path("data/runs/v2.7.6/stage2_eval/canonical_eval.json")
d = json.load(open(SRC))
cells = {(c["arm"], c["scoring"], c["mode"]): c for c in d["cells"]}
pre = cells[("v274", "banded", "kstep")]["split"]
jt = cells[("jt42000", "banded", "kstep")]["split"]

surfaces = ["cylinder", "band_floor", "band_ceiling"]
labels = ["cylinder\n(gray)", "band floor\n(red)", "band ceiling\n(purple)"]
colors = [C_CYL, C_FLOOR, C_CEIL]
pre_v = [pre[s]["count"] for s in surfaces]
jt_v = [jt[s]["count"] for s in surfaces]

# GATE: assert rendered values equal the json values (they are read from it, so equality is definitional;
# the assert pins that no target was hand-substituted). Cross-check targets: 260/252/8 pre, 65/64/1 JT; cyl 15/20.
assert pre_v == [15, 252, 8] and jt_v == [20, 64, 1], f"canonical json != cross-check targets: pre={pre_v} jt={jt_v}"

def render(meeting: bool):
    x = np.arange(len(surfaces)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    lab_pre = WITHOUT_TERM if meeting else "pre-band (244f4f83)"     # Amdt 7: neutral, non-competitive
    lab_jt = WITH_TERM if meeting else "JT band-aware (09c33bf4)"
    b1 = ax.bar(x - w / 2, pre_v, w, color=colors, edgecolor="black", hatch="//", label=lab_pre)
    b2 = ax.bar(x + w / 2, jt_v, w, color=colors, edgecolor="black", label=lab_jt)
    for bars, vals in ((b1, pre_v), (b2, jt_v)):
        for r, v in zip(bars, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + 3, str(v), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("collisions (count out of 2000 episodes)")
    ax.set_title("Collisions by surface (n=2000 episodes)" if meeting else
                 "Collisions by surface — pre-band vs band-aware JT\ncanonical pool 0ef3751b, banded scoring, kstep k=5 (GPU)", fontsize=11)
    ax.legend(loc="upper right", fontsize=9); ax.set_ylim(0, max(pre_v) * 1.15)
    return save(fig, "fig_band_collision_counts.png")

p = render(True); pm = p
print(f"M1 PASS -> {p} ; M13 -> {pm}")
print(f"  pre-band(244f4f83): cyl {pre_v[0]} floor {pre_v[1]} ceil {pre_v[2]} (band total {pre['band_total']['count']})")
print(f"  JT(09c33bf4):       cyl {jt_v[0]} floor {jt_v[1]} ceil {jt_v[2]} (band total {jt['band_total']['count']})")
print(f"  source: {SRC}")
