"""v2.7.7 M2 — |L_g V_hat| thrust-channel share, recomputed eval-only on the CANONICAL pool (0ef3751b) for the
pre-band comparator (244f4f83) and the band-aware JT (09c33bf4), overall and at the band boundary (|z|>=3).
Uses the recorded channel decomposition (v276_stage2_lg_share.lg_thrust_share), POOLS repointed to the shared
data/eval_pools/. Targets (band-feasible dumps): pre-band overall torque ~98.4%; ~0.074 -> 0.204/0.711 — the
canonical recompute may differ; artifact values are used and any gap is flagged."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import scripts.analysis.v276_stage2_lg_share as LG
from scripts.deck.deck_style import save, OUT, C_DESIGN, C_LEARNED, C_CYL, WITHOUT_TERM, WITH_TERM
import matplotlib.pyplot as plt

LG.POOLS = Path("data/eval_pools")   # v2.7.6 pools were archived; canonical lives here
STEM = "eval_full_quadrotor-3d-d2r_n2000_seed23456"
CK = {"preband": Path("data/runs/v2.7.4/set__20260720-083533__seed42/v2.7.4__20260720-091830__seed42/checkpoints/best.pt"),
      "jt": Path("data/runs/v2.7.6/set__20260725-043415__seed42/v2.7.6__jt__20260725-052127__seed42/checkpoints/step_042000.pt")}
CACHE = OUT / "authority_split_canonical.json"
if CACHE.exists():                                        # reuse the eval-only recompute (headless re-run cheap)
    res = json.load(open(CACHE))
else:
    res = {}
    for name, ck in CK.items():
        r = LG.lg_thrust_share(ck, STEM)
        res[name] = {"overall": r["median_thrust_share_overall"],
                     "boundary": r["band_boundary_subset_|z|>=3.0"]["median_thrust_share"],
                     "n": r["n_boundary_states"], "step": r["step"]}
    CACHE.write_text(json.dumps(res, indent=2) + "\n")

def render(meeting: bool):
    labels = ["without term\noverall", "without term\nband boundary", "with term\noverall", "with term\nband boundary"]
    vals = [res["preband"]["overall"], res["preband"]["boundary"], res["jt"]["overall"], res["jt"]["boundary"]]
    cols = [C_CYL, C_CYL, C_DESIGN, C_DESIGN]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    bars = ax.bar(range(4), vals, color=cols, edgecolor="black", width=0.6)
    for r, v in zip(bars, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.012, f"{v:.3f}\n(torque {1-v:.3f})", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels)
    ax.set_ylabel("thrust-channel share of |L_g V̂|")
    ax.set_title("Certificate authority: thrust-channel share (n=%d boundary states)" % res["jt"]["n"] if meeting else
                 "|L_g V̂| thrust-channel share — pre-band (244f4f83) vs JT (09c33bf4)\ncanonical pool 0ef3751b, GPU", fontsize=10.5)
    ax.set_ylim(0, max(vals) * 1.2)
    return save(fig, "fig_authority_split.png")

p = render(True); pm = p
tgt = {"pre overall~0.016": res["preband"]["overall"], "pre boundary~0.074": res["preband"]["boundary"],
       "jt overall~0.204": res["jt"]["overall"], "jt boundary~0.711": res["jt"]["boundary"]}
print(f"M2 PASS -> {p} ; M13 -> {pm}")
print("  canonical recompute:", {k: round(v, 4) for k, v in tgt.items()})
