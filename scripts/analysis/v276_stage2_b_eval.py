"""v2.7.6 Stage-2 M4(b) — pre-JT comparator: the Stage-2 OC value (nominal cascaded-PD + HardNet with the
band-aware V_hat) on the band-feasible and full-range pools, dual (legacy/banded), full n2000. This is the
ledger-row-137 role under the new hazard, and the comparable OC figure (06_workflow s5). Its full-pool
infeasibility is the reference for the Stage-2 empty-branch prediction (4)."""
from pathlib import Path
from scripts.analysis.v276_stage2_eval import eval_dual

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
OC = REPO / "data/runs/v2.7.6/set__20260725-024027__seed42/v2.7.6__oc__20260725-024027__seed42/checkpoints/best.pt"
OUT = REPO / "data/runs/v2.7.6/stage2_eval"

for stem, tag in (("eval_bandfeasible_quadrotor-3d-d2r_n2000_seed42", "oc_a_bandfeasible"),
                  ("eval_fullrange_quadrotor-3d-d2r_n2000_seed42", "oc_a_fullrange")):
    eval_dual(OC, stem, tag, OUT)
