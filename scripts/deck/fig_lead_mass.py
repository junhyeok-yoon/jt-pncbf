"""v2.7.7 M4 — analytic lead-term V-mass fraction P(c;w) (NO data; formula given verbatim in amendment 1):
  P(c;w) = max{0, 0.5*(1 - TWR^-1*(1 + w/(g*c)))};  sup_c P = 0.5*(1 - TWR^-1)
  Instance: TWR=2, g=9.81, w=1.5, c_z=0.8 -> P=0.2022; ceiling 0.25.
Inset: c_min(theta) = w/(g*(TWR*cos theta - 1)), diverging as TWR*cos theta -> 1 (theta -> 60 deg for TWR=2).
Gate: P(0.8;1.5) == 0.2022 (3 dp) before rendering."""
from __future__ import annotations
import numpy as np
from scripts.deck.deck_style import save, C_DESIGN, C_LEARNED, C_VERIFY
import matplotlib.pyplot as plt

TWR, g, w, cz = 2.0, 9.81, 1.5, 0.8
def P(c, w=w): return np.maximum(0.0, 0.5 * (1 - (1 / TWR) * (1 + w / (g * c))))
ceiling = 0.5 * (1 - 1 / TWR)
Pcz = float(P(cz))
assert round(Pcz, 3) == 0.202 and ceiling == 0.25, f"gate fail: P(0.8;1.5)={Pcz}, ceiling={ceiling}"

def render(meeting: bool):
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    c = np.linspace(0.05, 6.0, 400)
    ax.plot(c, P(c), color=C_DESIGN, lw=2.2, label="P(c; w=1.5 m/s)")
    ax.axhline(ceiling, color="gray", ls="--", lw=1.3, label=f"ceiling sup_c P = 0.5(1-1/TWR) = {ceiling}")
    ax.plot([cz], [Pcz], "o", color=C_LEARNED, ms=9, zorder=5)
    ax.annotate(f"c_z=0.8 -> P={Pcz:.4f}", (cz, Pcz), (cz + 0.6, Pcz - 0.045), color=C_LEARNED, fontsize=10,
                arrowprops=dict(arrowstyle="->", color=C_LEARNED))
    ax.set_xlabel("lead coefficient c (s)"); ax.set_ylabel("lead-term V-mass fraction  P(c; w)")
    ax.set_title("Lead-term V-mass fraction vs lead coefficient" if meeting else
                 "Analytic lead-mass P(c;w) = max{0, 0.5(1 - TWR⁻¹(1 + w/(g c)))}\nTWR=2, g=9.81, w=1.5 m/s", fontsize=10.5)
    ax.set_xlim(0, 6); ax.set_ylim(0, 0.28); ax.legend(loc="lower center", fontsize=9)
    # inset: c_min(theta) diverging at 60 deg — placed lower-right (empty region; the main curve plateaus high)
    axi = ax.inset_axes([0.58, 0.14, 0.37, 0.40])
    th = np.linspace(0, 59.0, 300); denom = TWR * np.cos(np.radians(th)) - 1
    cmin = np.where(denom > 1e-6, w / (g * denom), np.nan)
    axi.plot(th, cmin, color=C_VERIFY, lw=1.8); axi.axvline(60, color="red", ls=":", lw=1.2)
    axi.set_xlim(0, 64); axi.text(58, 5.4, "60°", color="red", fontsize=8, va="top", ha="right")
    axi.set_xlabel("tilt θ (deg)", fontsize=7); axi.set_ylabel("c_min(θ) (s)", fontsize=7)
    axi.set_title("min lead c for self-consistency", fontsize=7); axi.set_ylim(0, 6); axi.tick_params(labelsize=6)
    return save(fig, "fig_lead_mass.png")

p = render(True); pm = p
print(f"M4 PASS -> {p} ; M13 -> {pm}\n  P(0.8;1.5)={Pcz:.4f} (gate 0.202 ok), ceiling {ceiling}; formula = amendment 1 (no data)")
