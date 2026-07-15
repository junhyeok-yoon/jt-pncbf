# v2.6.0 — ADOPTED (seed 42, quadrotor_planar)

Secured snapshot of the M6 result (first navigable learned JT policy through the differentiable filter on
the underactuated planar quadrotor). Prepared per `06_workflow` §2.5. Single seed 42.

## Run identity
- run_id: `v2.6.0__20260715-042141__seed42`
- best checkpoint: `checkpoints/best.pt` — step **46500** (peak in-loop cps 0.2270)
- git_commit at run time: `f5edc2513bb630a40e7a44cf6a2c229d999e631c` (DIRTY — config working-tree changes uncommitted)
- value warm-start (parent): M1 OC V̂ (step 34500), run `v2.6.0__20260715-010357__seed42` — now at
  `data/previous_runs/v2.6.0__20260715-010357__seed42/checkpoints/best.pt` (moved on close; identity pinned
  by sha below). NB: the copied `config.yaml` records the pre-move path `data/v2.6.0__…010357…/best.pt`.

## Headline (canonical run_full path, full pool n2000/seed23456)
- cps_v2 **0.2920** [0.2398, 0.3382]; reach 0.6935; collision 0.1430; oob 0.0; stuck 0.0005;
  timeout 0.1630; inf_v2 0.1117.
- vs M3 pre-JT baseline (same pool): cps 0.1738 → 0.2920 (**+0.1182**, bootstrap CIs non-overlapping:
  M3-hi 0.2263 < M6-lo 0.2398); collision 0.1850 → 0.1430 (−23%).
- Gates: **P1/M2 PASS** (near-B₀ learned degen 0.000, median ‖L_g V̂‖ 18.02); **P2 PASS** (cps climbed,
  no first-order stall); **P3 PASS** (0/288 residual collisions collinear, ‖L_g V̂‖ median 1.05).
- (v2.6.0, quadrotor_planar) SOTA — the system's first full-pool row = its baseline; cps NOT comparable
  across systems (never ranked vs DI/unicycle).

## Pinned SHA-256
| artifact | sha256 |
|---|---|
| `checkpoints/best.pt` (this snapshot) | `db795949c5165c5596765aa83e20efd6aa3409873c35d79d6a387db620b8ab7f` |
| M1 value warm-start `…010357…/best.pt` (parent) | `c6c583802d2e427f5c9f877d12ed78ea20750fd4da8483a52507570585b9acf8` |
| eval pool full `eval_full_quadrotor-planar_n2000_seed23456.pkl` | `92df837bad44658dd9a1755df19b39c6c4bb1be1bc6b7169c7e92baaf0dec531` |
| eval pool inloop `eval_inloop_quadrotor-planar_n500_seed12345.pkl` | `4c8af29c550bce8102333ff4504886e4256cd50842bc08354141475115467e19` |

## Snapshot contents (this directory)
- `checkpoints/best.pt`, `config.yaml`, `eval_metrics.csv`, `eval_episodes.csv`, `pool_manifest.json`,
  `status.json`, `figures/*` (cbf_contour_m6.png, trajectory_grid_A/B.png, inloop/*).
- `report.md` = the authored build-log `docs/versions/v2.6.0/phase_s1_report.md` (the substantive record).
- `run_report_auto.md` = the run-dir auto-generated JT report template (kept for provenance).
- **Excluded** (per instruction): `metrics.csv` (bulky per-step training log; remains in the run dir at
  `data/previous_runs/v2.6.0__20260715-042141__seed42/metrics.csv`). All four v2.6.0 run dirs (M1, aborted
  M4/M5, M6) were moved to `data/previous_runs/` on close, intact.

## Reproduction
Config = `config.yaml` (carries the winning delta: `detach_filter_coeffs=true`, `w_terminal=30`,
`clamp_tanh` head, `z_target=1.0`, per-channel `sat_excess_threshold=[19.62,0.2]`, `c_gain=0.3`,
`value_init_ckpt` = M1 best.pt). Those keys live in the tracked but currently-uncommitted `src/configs/*`
(git is the Researcher's step). n_steps 50000, schedule_n_steps 42000, seed 42.

## Notes / open items
- Single seed 42; multi-seed confirmation is a Researcher call.
- Diagnostics (D1–D5) and the residual characterization are in `report.md` §M6 residual diagnostics.
- Eval-path note: the diagnostics used the dual_arm/hand-roll path (best.pt cps 0.2743) vs run_full 0.2920
  (~0.018 gap); the headline uses run_full. See the PROTOCOL FOLLOW-UP items in `report.md`.
