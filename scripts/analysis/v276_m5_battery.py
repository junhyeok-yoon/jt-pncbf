"""v2.7.6 M5 battery — at the re-selected headline (JT step 42000, sha 09c33bf4) and the v2.7.4 band-blind
baseline (244f4f83), on BOTH pools (band-feasible + full-range): (i) dual legacy/banded cps + components +
band_exit with scene-bootstrap CIs (eval_dual), and (ii) the banded collision-surface split (cylinder / band
floor / ceiling). Gives the headline numbers, prediction (1)/(1-informative) and (3), and the JT-vs-v2.7.4
full-range band comparison (the registered test of the band term beating a band-blind policy). Eval-only."""
from __future__ import annotations

from pathlib import Path

from scripts.analysis.v276_stage2_eval import eval_dual, REPO
from scripts.analysis.v276_collision_split import classify, STEMS

OUT = REPO / "data/runs/v2.7.6/stage2_eval"
CS_OUT = OUT / "collision_split"
JT42 = REPO / "data/runs/v2.7.6/set__20260725-043415__seed42/v2.7.6__jt__20260725-052127__seed42/checkpoints/step_042000.pt"
V274 = REPO / "data/runs/v2.7.4/set__20260720-083533__seed42/v2.7.4__20260720-091830__seed42/checkpoints/best.pt"
BF, FR = STEMS["bandfeasible"], STEMS["fullrange"]

ARMS = [(JT42, "jt42000"), (V274, "v274")]
POOLS = [(BF, "bandfeasible"), (FR, "fullrange")]

for ckpt, arm in ARMS:
    for stem, pool in POOLS:
        sfx = "" if pool == "bandfeasible" else f"_{pool}"
        eval_dual(ckpt, stem, f"{arm}_{pool}", OUT)
        classify(ckpt, f"{arm}_{pool}_split", CS_OUT, stem)
print("M5 battery done.", flush=True)
