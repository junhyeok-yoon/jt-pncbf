"""v2.7.7 M6 — failure mechanism by tilt stratum: cylinder-collision rate vs floor-exit rate over
[0,60)/[60,120)/[120,180] deg. The two mechanisms cross (collision falls, floor rises with tilt).
Source: docs/versions/v2.7.4/theory_measurements.md:180-187 (v2.7.4 close artifact, canonical pool)."""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
from scripts.deck.deck_style import save, C_CYL, C_FLOOR
import matplotlib.pyplot as plt

SRC = Path("docs/versions/v2.7.4/theory_measurements.md")
txt = SRC.read_text()
# parse the three stratum rows: | [a,b) | n | coll | coll_rate | oob | oob_rate | ...
rows = re.findall(r"\| \[(\d+),(\d+)[\)\]] \| \d+ \| \d+ \| \*?\*?([\d.]+)\*?\*? \| \d+ \| \*?\*?([\d.]+)\*?\*? \|", txt)
strata = [(f"[{a},{b})", float(cr), float(orr)) for a, b, cr, orr in rows[:3]]
cyl_rate = [s[1] for s in strata]; floor_rate = [s[2] for s in strata]
assert cyl_rate == [0.0230, 0.0227, 0.0138] and floor_rate == [0.0125, 0.0148, 0.0236], f"parsed {cyl_rate},{floor_rate}"

def render(meeting: bool):
    x = np.arange(3); w = 0.38
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    b1 = ax.bar(x - w / 2, cyl_rate, w, color=C_CYL, edgecolor="black", label="cylinder collision rate")
    b2 = ax.bar(x + w / 2, floor_rate, w, color=C_FLOOR, edgecolor="black", label="floor-exit rate")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.0004, f"{r.get_height():.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(["[0,60)", "[60,120)", "[120,180]"])
    ax.set_xlabel("initial tilt stratum (deg)"); ax.set_ylabel("failure rate (fraction of stratum)")
    if meeting:
        ax.axvspan(-0.5, 0.5, color="#dfe9df", alpha=0.5, zorder=0)
        ax.text(0, 0.0265, "inside holding cone", ha="center", va="top", fontsize=8, style="italic", color="#2e6b2e")
    ax.set_title("Failure rate by initial tilt (n=2000 episodes)" if meeting else
                 "Failure mechanism crosses with tilt (pre-band 244f4f83, canonical)\ncylinder falls 0.0230->0.0138, floor rises 0.0125->0.0236", fontsize=10.5)
    ax.legend(loc="upper center", fontsize=9); ax.set_ylim(0, 0.028)
    return save(fig, "fig_tilt_strata.png")

p = render(True); pm = p
print(f"M6 PASS -> {p} ; M13 -> {pm}\n  cylinder {cyl_rate} | floor {floor_rate}\n  source: {SRC}:180-187")
