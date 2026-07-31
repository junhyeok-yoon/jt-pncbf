"""v2.7.7 M5 — ZOH inter-sample violating periods (2.45%% -> 0.379%% from dt 0.05 -> 0.01), the C2 s^2 protected
layer, and a per-step latency table. Sources: docs/versions/v2.7.5/theory_measurements.md:71-74,172 (violating
periods + C2); data/previous_runs/v2.7.6/stage2_eval/m8_latency.json (latency)."""
from __future__ import annotations
import csv, json, re
from pathlib import Path
import numpy as np
from scripts.deck.deck_style import save, OUT, C_VERIFY, C_DESIGN, C_LEARNED
import matplotlib.pyplot as plt

TH = Path("docs/versions/v2.7.5/theory_measurements.md"); t = TH.read_text()
viol_05 = float(re.search(r"3159 ?/ ?129158 = ([\d.]+) ?%", t).group(1))       # 2.45
viol_01 = float(re.search(r"489 ?/ ?129158 = ([\d.]+) ?%", t).group(1))        # 0.379
C2 = float(re.search(r"C2 = ([\d.]+)", t).group(1))                            # 19.59
assert (viol_05, viol_01) == (2.45, 0.379), f"parsed {(viol_05, viol_01)}"

LAT = Path("data/previous_runs/v2.7.6/stage2_eval/m8_latency.json"); L = json.load(open(LAT))
lat_rows = []
for c in L["cells"]:
    if c["label"] == "non_empty_base":
        lat_rows.append(("base (non-empty)", 0, c["median_ms"], c["p95_ms"], c["H4_p95_under_10ms"]))
    else:
        lat_rows.append((c["label"], c.get("n_candidate_evals", 0), c.get("empty_median_ms"), c.get("empty_p95_ms"), c.get("empty_H4_p95_under_10ms")))
with (OUT / "latency_table.csv").open("w", newline="") as f:
    w = csv.writer(f); w.writerow(["arm", "n_candidate_evals", "p50_ms", "p95_ms", "H4_p95_under_10ms"]); w.writerows(lat_rows)

Rq = float(re.search(r"quadratic.*?R² = ([\d.]+)", t, re.DOTALL).group(1))      # 0.895 (envelope quadratic fit)
C2_STAT = f"C2 = envelope quadratic-fit coefficient (V-envelope ≈ C2·s², R²={Rq})"

def render(meeting: bool):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    bars = axL.bar(["20 Hz\n(dt=0.05 s)", "100 Hz\n(dt=0.01 s)"], [viol_05, viol_01], color=C_VERIFY, edgecolor="black", width=0.55)
    for r, v in zip(bars, [viol_05, viol_01]):
        axL.text(r.get_x() + r.get_width() / 2, v + 0.04, f"{v}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    axL.set_ylabel("inter-sample violating control periods (%)"); axL.set_ylim(0, 2.9)
    axL.set_title("Inter-sample sign-crossings vs control rate\n2.45% -> 0.379% (6.5x)" if meeting else
                  f"ZOH first-substep sign-crossings\n2.45% -> 0.379% (6.5x), {C2_STAT}", fontsize=9.5)
    s = np.linspace(0, 0.05, 100)
    axR.plot(s * 1000, C2 * s**2, color=C_DESIGN, lw=2, label=f"C2·s² envelope (C2={C2}, R²={Rq})")
    for dt, col in ((0.05, C_LEARNED), (0.01, C_VERIFY)):
        axR.axvline(dt * 1000, color=col, ls="--", lw=1.2)
        axR.text(dt * 1000, C2 * dt**2, f" {int(1/dt)} Hz\n V-margin {C2*dt**2:.4f}", fontsize=8, va="bottom", color=col)
    axR.set_xlabel("inter-sample time s (ms)"); axR.set_ylabel("V-margin bound C2·s²")
    axR.set_title("Quadratic protected layer vs control period", fontsize=10.5); axR.legend(fontsize=8)
    axR.text(0.5, -0.22, C2_STAT, transform=axR.transAxes, ha="center", fontsize=7.5, style="italic")
    return save(fig, "fig_zoh_latency.png")

p = render(True); pm = p
print(f"M5 PASS -> {p} ; M13 -> {pm} + latency_table.csv ({len(lat_rows)} arms)")
print(f"  violating 2.45%->0.379%, C2={C2} (R²={Rq}); base p95 {lat_rows[0][3]}ms, p1_k3 p95 {[r[3] for r in lat_rows if r[0]=='p1_k3'][0]}ms")
print(f"  sources: {TH}:71-74,172 ; {LAT}")
