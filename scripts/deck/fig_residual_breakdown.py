"""v2.7.7 M3 — pre-band (244f4f83) residual-failure split, 74 = 41 cylinder + 33 floor (0 ceiling, 0 lateral).
Source: docs/versions/v2.7.4/theory_measurements.md:48-58 (v2.7.4 close artifact, canonical pool)."""
from __future__ import annotations
import re
from pathlib import Path
from scripts.deck.deck_style import save, C_CYL, C_FLOOR, C_CEIL
import matplotlib.pyplot as plt

SRC = Path("docs/versions/v2.7.4/theory_measurements.md")
txt = SRC.read_text()
# load the counts from the residual-failure-split table (parse, do not hardcode)
cyl = int(re.search(r"cylinder contact \(`collision`\) \| \*\*(\d+)\*\*", txt).group(1))
vert = int(re.search(r"vertical \(floor/ceiling\)\*\* \| \*\*(\d+)\*\*", txt).group(1))
lat = int(re.search(r"lateral \(xy\) \| \*\*(\d+)\*\*", txt).group(1))
# tilt-strata table pins all 33 vertical exits as floor (0 ceiling) — parse the totals row
floor = int(re.search(r"\| totals \| 2000 \|.*?\*\*(\d+)\*\* \| \*\*0\*\*", txt).group(1))
ceiling = vert - floor
total = cyl + vert + lat
assert (cyl, floor, ceiling, lat, total) == (41, 33, 0, 0, 74), f"parsed {(cyl,floor,ceiling,lat,total)} != target"

def render(meeting: bool):
    lat_name = "side walls" if meeting else "lateral (xy)"   # M13 rename in meeting variant
    parts = [("cylinder", cyl, C_CYL), ("band floor", floor, C_FLOOR), ("band ceiling", ceiling, C_CEIL), (lat_name, lat, "#cccccc")]
    fig, ax = plt.subplots(figsize=(4.6, 5.4)); bottom = 0
    for name, v, col in parts:
        ax.bar(0, v, 0.6, bottom=bottom, color=col, edgecolor="black", label=f"{name} = {v}")
        if v > 0:
            ax.text(0, bottom + v / 2, f"{name}\n{v}", ha="center", va="center", fontsize=10, fontweight="bold",
                    color="white" if col != "#cccccc" else "black")
        bottom += v
    ax.set_xlim(-0.6, 0.6); ax.set_xticks([]); ax.set_ylabel("residual failures (count out of 2000)")
    ax.set_title(f"Residual failures (n=2000 episodes)" if meeting else
                 f"Pre-band residual failures = {total}\n244f4f83, canonical pool, legacy scoring", fontsize=11)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    return save(fig, "fig_residual_breakdown.png")

p = render(True); pm = p
print(f"M3 PASS -> {p} ; M13 -> {pm}\n  74 = {cyl} cylinder + {floor} floor + {ceiling} ceiling + {lat} lateral\n  source: {SRC}:48-58,180-190")
