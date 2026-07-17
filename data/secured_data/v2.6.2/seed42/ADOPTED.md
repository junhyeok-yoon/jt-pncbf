# v2.6.2 — ADOPTED (seed 42, quadrotor_planar)

Secured snapshot of the v2.6.2 M6 result: the v2.6.1 winning pipeline (learned OC value → plain JT +
terminal + detach, no shield, corrected plant tau_max=1.0) with a **situation-dependent velocity objective**
— dense goal-gated **settling** + obstacle-gated **braking-envelope** approach term (running-cost redesign in
`policy_bptt_loss`; NO value/filter/structure change). Single seed 42 close (Researcher decision). Prepared
per `06_workflow §2.5`.

## Run identity
- run_id: `v2.6.2__20260716-182949__seed42`
- best checkpoint: `checkpoints/best.pt` — step **30000** (peak in-loop cps 0.7341)
- git_commit at run time: `9310114…DIRTY` (config working-tree changes uncommitted — git is the Researcher's step)
- value warm-start (parent): REUSED **v2.6.1 M1** OC V̂ step 42000, run `v2.6.1__20260715-170907__seed42`
  (identity pinned by sha below; M1 NOT retrained — value path byte-identical to v2.6.1).

## Headline (canonical run_full path, full pool n2000/seed23456)
- cps_v2 **0.7311** [0.6925, 0.7648]; reach 0.905; collision **0.0670** [0.0565, 0.0780]; oob 0; stuck 0.003;
  timeout 0.025; inf_v2 0.0815; sat_rate 0.4268.
- vs v2.6.1 M6 within-version-plant baseline (same pool, same plant): cps 0.5753 → **0.7311 (+0.156,
  CI-SEPARATED**: v2.6.1-hi 0.616 < brake-lo 0.693); collision 0.0810 → 0.0670 (−17%). **H-appr CONFIRMED
  with margin** (collision CI entirely below 0.0810). vs the gaussian-FULL within-version step 0.6097 = +0.121.
- vs the pre-JT M3 anchor (0.1931, same pool): +0.538.
- Gates: P1/M2 reused-from-v2.6.1 PASS (value path unchanged; near-B₀ degen 0.000, median ‖L_g V̂‖ 16.72);
  **P3 CONFIRMED** (123 residual collisions, 0/123 collinear `S_h`, learned ‖L_g V̂‖ median 0.453, degen 8.9%
  — certificate sound; the residual is NOT a value failure).
- Mechanism (H-appr) **MIXED** (honest): inward speed @1 m p50 **1.785** (< broken gaussian 1.94, NOT <
  v2.6.1's 1.66 → PARTIAL); filter-infeasible frac at residual collisions 1.000 (per-collision unchanged),
  absolute count 159→123 (−23%); envelope occupancy near obstacles 0.124. The brake-envelope cut collision by
  reducing the NUMBER of fast-approach commits (residual set 165→130), not by making residuals feasible.
- Insertion variants (`eval_metrics.csv`): lqr cps −0.383; frozen 0.6649; live 0.6792.

## Config delta from v2.6.1 (the mechanism — one axis, running cost)
`loss.policy`, quadrotor-scoped (DI/unicycle keep the legacy `d2` branch, byte-identical parity):
- **dense goal-gated settling** (NEW): `w_settle=1.0 · exp(−‖p−g‖²/settle_rho²)·‖v‖²`, `settle_rho=0.30`.
- **braking-envelope approach** (NEW, amendment 2): `w_appr=30.0 · Σ_k relu(s_k·tau_brake − surf_k)²`,
  `tau_brake=0.6`, `deficit_form="sq_cap"` (replaced the inert gaussian gate).
- **`w_terminal_v` 30→0**: settling moved from the sparse ‖v_T‖ terminal to the dense goal-gated term.
- goal-distance stays **quadratic** `Σ(p−g)²` (the Huber variant was reverted — amendment 1, audit confound).
- Carried: `w_terminal=30`, `detach_filter_coeffs=true`, `tau_max=1.0`, `lambda_v=0.01`, `mu_u=0.001`.
- Full config = `config.yaml`. (Keys live in tracked-but-uncommitted `src/configs/*` + `losses.py` — git is
  the Researcher's step.)

## Pinned SHA-256
| artifact | sha256 |
|---|---|
| `checkpoints/best.pt` (this snapshot) | `b33e67873cade9262b9fdea88acec28389bf3b27b405349188aa7600d2189d61` |
| M1 value warm-start `…170907…/best.pt` (parent, REUSED) | `1099eaf3b430c05e1efd81cbf358b9b4ef0401766cd297b050909664568c4696` |
| eval pool full `eval_full_quadrotor-planar_n2000_seed23456.pkl` | `92df837bad44658dd9a1755df19b39c6c4bb1be1bc6b7169c7e92baaf0dec531` |
| eval pool inloop `eval_inloop_quadrotor-planar_n500_seed12345.pkl` | `4c8af29c550bce8102333ff4504886e4256cd50842bc08354141475115467e19` |

⚠ The pool manifest's `sampler_params` (v_init_max=0.5, tau_max=0.2) are STALE relative to the pool's actual
contents (‖v0‖≤1.4999, |ω0|≤1.0, eval plant tau_max=1.0). The pkl matches its manifest sha (above) and is
used frozen/consistently across v2.6.0/1/2 — comparability unaffected. See `data/pools/POOL_MANIFEST_ERRATA.md`.

## Snapshot contents (this directory)
- `checkpoints/best.pt`, `config.yaml`, `eval_metrics.csv`, `eval_episodes.csv`, `pool_manifest.json`,
  `status.json`, `figures/*` (105 files: cbf_contour.png, trajectory_grid_A/B.png final, +
  `inloop/` = 34 grids + per-eval CBF contours).
- `report.md` = the authored build-log `docs/versions/v2.6.2/phase_v262_report.md` (full, 659+ lines).
- **Excluded** (per instruction): `metrics.csv` (bulky per-step training log; remains in the run dir).

## Notes / open items
- **Single seed 42** (Researcher decision; no multi-seed escalation this version).
- **IC-exclusion line CLOSED** by the doom census: the provable-doom set is EMPTY under the 3g relaxed system
  (ballistic 0/2000 + 4D-HJ 0/2000, both gates pass); attitude-cost doom is uncertifiable (6D dense HJ measured
  infeasible, 178–1661 GB). Full-range IC pool is the official quadrotor distribution.
- **Gravity-observability discovered** (this version's key diagnostic finding): the gravity-blind body-frame
  obs aliases upright↔inverted (θ undecodable in the high-tilt/early failure cell, 98.6°; 1-step aliased-
  dynamics closure gap ≈0.4 m/s; the feasibility certificate falsely dooms 61.5% of savable high-tilt ICs).
  The kinematic "thrust-misfire" mechanism was DISCONFIRMED. → next version: observation augmentation.
- **Deflation** sweep was a NULL (not a collision lever; reshuffles collision↔timeout) — ledger NOTE row,
  non-SOTA. The **recoverability filter** code (`src/common/quadrotor_recoverability.py`, `scene_init.py`)
  stays committed but **INERT** (recov_margin unset; criterion rejected at FP anatomy).
- Diagnostic suite (this version's real output): `collision_regression_diag.md`, `close_facts.md`, and the
  `phase_v262_report.md` sections (residual anatomy, deflation, FP anatomy, doom census, gravity-observability).
- Open follow-ups (see `report.md` close section): theory repo merge (feasibility/obs/liveness sections not yet
  in the canonical note). The §4.4-terminal and corrected-plant-constants promotions are being applied by the
  Strategist at this close.
