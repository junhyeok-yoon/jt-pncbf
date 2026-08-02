# v2.7.6 — band-hazard JT (experiments sub-tree, NOT SOTA)

Secured diagnostic per `04_eval` §7.5 / `06_workflow` §6.3. This is the v2.7.6 headline JT run (vertical domain
surfaces + observation fix), kept for the record. It is **not the SOTA snapshot and is never bolded in the
ledger** (§2.4): the bold within quadrotor_3d remains v2.7.2 (`4baaf031`), because the standing comparator
cannot be re-scored on the current plant/pool (it predates the v2.7.3 per-rotor actuator refactor and the
full-SO(3)/d2r IC distribution), so the close-step-3 §2(c) comparison is **unavailable**.

- Headline checkpoint: `checkpoints/step_042000.pt`, sha8 `09c33bf4` — re-selected on the canonical
  band-feasible full pool (M5), NOT the run's in-loop `best.pt` (@48000, also included for completeness).
- Canonical-pool numbers (`eval_full_quadrotor-3d-d2r_n2000_seed23456`, `0ef3751b`, GPU): banded cps 0.69287
  [none] / 0.80508 [kstep k=5]; legacy 0.80288 / 0.90783. Full close record:
  `docs/versions/v2.7.6_results.md`, `docs/versions/v2.7.6/close_retrieve.md`.
- Source run: `data/previous_runs/v2.7.6/set__20260725-043415__seed42/v2.7.6__jt__20260725-052127__seed42/`.
- Excluded per §7.5: TensorBoard events, per-step training `metrics.csv`.
- Registered falsifier fired (prediction 3, legacy full-range no-regress); deploy-contingent (removed by the
  kstep fallback). See the phase report and results doc.
