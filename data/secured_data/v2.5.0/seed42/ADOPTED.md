# v2.5.0 seed42 — ADOPTED (closed-version representative snapshot; NOT a new SOTA)

Source run dir: `data/v2.5.0__20260709-204711__seed42` (Stage B-2 run of record, seed 42).

**Headline B-2 checkpoint = `checkpoints/step_007500.pt`** (step 7500; md5 `bc0674519b7266f06db2f3d32d283719`,
byte-identical to `data/v2.5.0__20260708-212214__seed42/checkpoints/b2fast_best.pt`) — this is the
**n2000-reselected** best that anchors every B-2 headline number (legacy 0.5893 / cps_v2 0.8327).
NOTE: `checkpoints/best.pt` (md5 `c8237c1cbba30cd944dcbe529ad772c5`, **step 15000**) is the run's
**in-loop-cps-selected** best — a DIFFERENT checkpoint, retained per 04_eval §7.5's `{best.pt, final.pt}`
convention but NOT the B-2 headline. `checkpoints/final.pt` = step 30000. Cite `step_007500.pt` for B-2.

**Not SOTA.** v2.5.0 introduces no new DI SOTA — B-2 cps_v2 0.8327 (maneuver dual-deploy, D-slow) <
v2.3.0 0.9010. This snapshot is the version's closed representative run, not a SOTA snapshot; it is not
SOTA-bolded in the ledger (04_eval §2.4 / 06_workflow §6.3).

## Headline numbers (disk-cited)
- Native-filter eval (this dir's `eval_metrics.csv` / `metrics.csv`): the in-run learned-HardNet eval.
- **Anchor for the B-2 headline** = the maneuver dual-deploy re-eval:
  `data/v2.5.0__20260708-212214__seed42/deploy_b2f_Dslow.json` — legacy cps **0.5893260591**, cps_v2
  **0.8326756375702142**, reach 0.925, coll 0.0055, stuck 0.043, timeout 0.0265 (VM_FAST=1, gamma_m=0.02,
  D-slow dt 0.05).

## Figures
`figures/trajectory_grid_A.png`, `trajectory_grid_B.png`, `cbf_contour.png` (§7.5 required), plus
diagnostics `contour_slices.png`, `stuck_panels_{1,2,3}.png`.

**Figure provenance (read this):** the final figures are an **eval-only re-render** of the B-2 best,
produced in `data/v2.5.0__20260710-024230__seed42_evalfigs_b2best/` after a blocking consistency gate that
reproduced the deployed system to numerical identity (legacy **0.5893260591** / cps_v2
**0.8326756376**, matching `deploy_b2f_Dslow.json`). Grids A/B route through the canonical
`src/eval/plotting.py:plot_trajectory_control_grid`. **`cbf_contour.png` is rendered via the Option-2
analysis module** `scripts/analysis/eval_figs_b2best.py` (a non-canonical plotter route that calls
plotting.py primitives), because the deployed CBF here is the analytic **maneuver barrier V_M** and the
canonical `plot_cbf_contours` assumes a `ValueNetEnsemble`. The 2-line **DIFF-C** that would let the
canonical plotter take an `h_fn` **remains a proposal** (`docs/versions/v2.5.0/eval_figs_b2best.md`),
NOT applied. `contour_slices.png` row-2 (learned V̂) and `stuck_panels_*` are eval-only diagnostics.

## Excluded per §7.5
No TensorBoard event files. `reselect_n2000_b2fast.json` and `status.json` retained for provenance.
