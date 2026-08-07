# v2.8.2 CTRL — secured snapshot (NOT a SOTA claim)

Researcher-approved copy (`06_workflow` §6.3). The source run directory remains intact where it was filed; this is a
**copy**, never a move.

## Provenance
- **Run id:** `v2.8.2__jt__20260803-063606__seed42`
- **Checkpoint:** `best.pt` at **step 24000** (run completed to `current_step` 30000)
- **Seed:** 42
- **n_steps:** 30000
- **Terminal:** `goal_angrate_radius` (ω_G) = **0.30**
- **Encoder:** `hard_topk`
- **Value-init:** shared v2.7.6 OC `set__20260725-043415__seed42/…/checkpoints/best.pt`; policy fresh
- **Shipped fallback:** `{kstep, phases 1, k 3}`; **projection:** `dual_solve`
- **best.pt SHA256:** `c89f9aef0cdb5499c3f0e7d868c2e9e4650632d845245d2a53421c6b4658e285`

## Scored cells (dual three-cell, canonical n2000, shipped fallback, ω_G=0.30; `dual_CTRL.json`)
| cell | cps | reach | collision (obs/floor/ceil) | stuck | timeout | infeas | sat |
|------|----:|------:|----------------------------|------:|--------:|-------:|----:|
| gate | **0.7785** | 0.9205 | 0.0455 (.0115/.0335/.0005) | 0.0015 | 0.0325 | 0.1107 | 0.4478 |
| cps_tilt60 | **0.9046** | 0.9595 | 0.0090 (.0065/.0015/.0010) | 0.0005 | 0.0310 | 0.0695 | 0.4058 |
| cps_bandopen | **0.8563** | 0.9415 | 0.0180 (.0180/0/0) | 0.0015 | 0.0330 | 0.0941 | 0.4364 |

`cps_bandopen` band-crossing rate: 0.0340. **T2 PASSES**: gate reach 0.9205 ≥ 0.8875.

## Why secured despite not carrying a bold row
This checkpoint is **not SOTA-bold** — gate cps 0.7785 does not beat the standing 3-D bold (v2.8.0, 0.7919), and the
0.30-vs-0.48 terminal mismatch is unresolved (a terminal-matched re-score at ω_G=0.48 is registered separately). It is
secured under `06_workflow` §6.3, which allows a version to hold a snapshot without holding a bold row, because:

1. **It is the control** every v2.8.2 delta (the M1/M2/M3 conditions and the α_unsafe axis) is measured against.
2. **It carries the T2 attribution** that closed the open question from v2.8.1: CTRL passes T2 (reach 0.9205) under
   the same ω_G=0.30 terminal at which v2.8.1's soft-encoder run FAILED T2 (reach 0.8620) — so v2.8.1's failure
   attributes to the **encoder**, not the terminal.

**This snapshot is a control/attribution record, not a SOTA claim.** No bold applied; no promotion of a headline cps.
