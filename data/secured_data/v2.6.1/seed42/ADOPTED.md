# v2.6.1 — ADOPTED (seed 42, quadrotor_planar)

Secured snapshot of the v2.6.1 M6 result: the v2.6.0 winning pipeline (learned OC value → plain JT +
terminal + detach, no shield) reproduced on the **corrected plant** (torque box ±1.0) with a **velocity
terminal**. Direction-setting version — single seed 42 (the multi-seed escalation was aborted in favor of
the diagnostic suite; Researcher-authorized). Prepared per `06_workflow §2.5`.

## Run identity
- run_id: `v2.6.1__20260715-173808__seed42`
- best checkpoint: `checkpoints/best.pt` — step **46500** (peak in-loop cps 0.6018)
- git_commit at run time: see the run-dir `config.yaml` provenance (config working-tree changes uncommitted)
- value warm-start (parent): M1 OC V̂ step 42000, run `v2.6.1__20260715-170907__seed42` — moved to
  `data/previous_runs/` on close (identity pinned by sha below).

## Headline (canonical run_full path, full pool n2000/seed23456)
- cps_v2 **0.5753** [0.5333, 0.6163]; reach 0.8205; collision **0.0810** [0.069, 0.094]; oob 0; stuck 0;
  timeout 0.0985; inf_v2 0.1132.
- vs v2.6.1 M3 within-version baseline (same pool): cps 0.1931 → 0.5753 (**+0.382**, CIs non-overlapping:
  M3-hi 0.244 < M6-lo 0.533); collision 0.1810 → 0.0810 (−55%). Both hypotheses CONFIRMED (H1 torque box →
  collision < 0.143; H2 velocity terminal → timeout 0.0985 < 0.163, reach up).
- vs v2.6.0 M6 (0.292, tau_max=0.2): ~2× — DIFFERENT control bound; informative, not a pure supersession
  (changes.md §5). Ledger row flagged for Researcher classification.
- Gates: P1/M2 PASS (near-B₀ degen 0.000, median ‖L_g V̂‖ 16.72); P2 PASS (cps climbed, grad clean —
  velocity-terminal explosion risk refuted); P3 CONFIRMED (159 residual coll/7.95%, 0/159 collinear).

## Config delta from v2.6.0 (the two axes)
Axis A: `env.bounds.quadrotor_planar.tau_max` 0.2→**1.0**; `loss.policy.sat_excess_threshold`
[19.62,0.2]→**[19.62,1.0]**. Axis B: `loss.policy.w_terminal_v` NEW **30.0** + the `losses.py` velocity
terminal `+ discount·w_terminal_v·‖v_T‖`. Carried: `w_terminal=30`, `detach_filter_coeffs=true`, `c_gain=0.3`,
clamp_tanh head. Full config = `config.yaml`. (Those keys live in the tracked but currently-uncommitted
`src/configs/*` + `losses.py` — git is the Researcher's step.)

## Pinned SHA-256
| artifact | sha256 |
|---|---|
| `checkpoints/best.pt` (this snapshot) | `4334d7987c0d4bd5d79d00f0ccaac4cc774b0bfde573521fc8da3ddb43656035` |
| M1 value warm-start `…170907…/best.pt` (parent) | `1099eaf3b430c05e1efd81cbf358b9b4ef0401766cd297b050909664568c4696` |
| eval pool full `eval_full_quadrotor-planar_n2000_seed23456.pkl` | `92df837bad44658dd9a1755df19b39c6c4bb1be1bc6b7169c7e92baaf0dec531` |
| eval pool inloop `eval_inloop_quadrotor-planar_n500_seed12345.pkl` | `4c8af29c550bce8102333ff4504886e4256cd50842bc08354141475115467e19` |

## Snapshot contents (this directory)
- `checkpoints/best.pt`, `config.yaml`, `eval_metrics.csv`, `eval_episodes.csv`, `pool_manifest.json`,
  `status.json`, `figures/*` (cbf_contour.png final, cbf_contour_d6_slices.png diagnostic,
  trajectory_grid_A/B.png, inloop/ = 34 grids + 34 per-eval CBF contours).
- `report.md` = the authored build-log `docs/versions/v2.6.1/phase_v261_report.md`.
- **Excluded** (per instruction): `metrics.csv` (bulky per-step training log; remains in the run dir at
  `data/previous_runs/v2.6.1__20260715-173808__seed42/metrics.csv`).

## Notes / open items
- **Single seed 42.** Escalation aborted: seed-99 M1 done (best 0.197@42000), seed-99 M5 aborted mid-run at
  step 32999 (best_cps 0.567@22500), seed-12345 never launched — all moved to `data/previous_runs/`, intact.
- Diagnostic suite (this version's real output): `fundamental_diagnostics.md`, `role_function_diag.md`,
  `predictive_filter_eval.md` (lookahead null), `torque_box_plausibility.md`, `task_cost_surface.md`,
  `architecture_surface.md`; close inventory in `close_facts.md`.
- Open follow-ups (tagged in `report.md`): §4.4 terminal (utility confirmed, 03_train edit deferred);
  corrected plant constants pending promotion into 01_env/03_train/changes.md §4; feasibility/liveness theory
  additions not yet merged in-repo.
