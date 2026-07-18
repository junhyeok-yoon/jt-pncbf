# v2.7.0 — ADOPTED (seed 42, quadrotor_planar) — SOTA HEADLINE

Secured snapshot of the **v2.7.0 iteration-1 (observation gravity-direction restoration)** result: the quadrotor
observation was extended dim 20 → 22 by appending the body-frame gravity direction `(sin θ, cos θ)` after the
obstacle block (`01_env §3.3`), value retrained from scratch (dim-incompatible warm-start), everything else
frozen at the v2.6.2 winning config. **This is the (v2.7.0, quadrotor_planar) full-range SOTA, single-seed**
(Researcher seed-economy decision 2026-07-14). Prepared per `06_workflow §2.5`.

## Run identity (value → JT chain)
- **JT headline run** (this snapshot): `v2.7.0__20260717-025050__seed42`, best.pt step **46500** (sha8 `22b902a6`).
- **value (M2) run** (`value_run/`): `v2.7.0__20260717-021858__seed42`, oc V̂ best.pt step **45000**
  (sha8 `4660ca90`), in-loop best cps 0.5581. Full artifacts also retained in `data/previous_runs/`.

## Headline (full pool n2000 seed23456, canonical `evaluate()` batch-2000)
- cps **0.8232** [0.7910, 0.8523]; reach 0.9400; collision **0.0455** [0.0370, 0.0550]; oob 0.0000;
  stuck 0.0015; timeout 0.0130; infeasibility 0.0593. Recompute matches the stored column to <1e-3 (CIs agree
  within bootstrap-RNG noise). Insertion cps: lqr −0.3920 / frozen 0.7841 / live 0.8043.
- **vs v2.6.2** (0.7311): cps +0.0921 CI-separated; collision −32% CI-separated. Supersedes v2.6.2.
- Pre-JT (M3, nominal+HardNet on V̂): cps 0.6490 [0.6124, 0.6823] / collision 0.0495.
- Registered hypotheses: **S1/S2/P2 CONFIRMED**; **S3/P1 not confirmed** (see `docs/versions/v2.7.0_results.md`
  §2 and `phase_v270_report.md §M6`).

## Pinned SHA-256
| artifact | sha256 |
|---|---|
| `checkpoints/best.pt` (JT headline @46500) | `22b902a6e3ab634662a75154fb45c4ca662c8973907845b9eccc37b34d708a47` |
| `value_run/checkpoints/best.pt` (M2 value @45000) | `4660ca902444eb8f6fb097ff2b1d689204e80477cba7d5e2b7dedc66b26e22d5` |
| eval pool full `eval_full_quadrotor-planar_n2000_seed23456.pkl` | `92df837bad44658dd9a1755df19b39c6c4bb1be1bc6b7169c7e92baaf0dec531` |
| eval pool inloop `eval_inloop_quadrotor-planar_n500_seed12345.pkl` | `4c8af29c550bce8102333ff4504886e4256cd50842bc08354141475115467e19` |

## Snapshot contents (this directory)
- JT run: `checkpoints/{best.pt, final.pt}`, `config.yaml`, `eval_metrics.csv`, `eval_episodes.csv`,
  `pool_manifest.json`, `status.json`, `git_commit.txt`, `figures/*`.
- `value_run/`: `checkpoints/best.pt`, `config.yaml`, `status.json`, `eval_metrics.csv`, `eval_episodes.csv`,
  `pool_manifest.json`.
- `report.md` = the authored close verdict `docs/versions/v2.7.0_results.md`.
- **Excluded** (per instruction / §7.5): `metrics.csv` (per-step training log; retained in `previous_runs/`),
  TensorBoard event files.

## Notes
- Single seed 42; multi-seed escalation deferred to the post-corridor-axis configuration.
- Residual-collision counts are code-path-dependent: collision RATE 0.0455 = 91/2000 (canonical `evaluate()`);
  P1/P3 shares use the manual-loop count 94; anatomy uses the agreed set 78. Each stated with its denominator
  in `docs/versions/v2.7.0/retrieve_close_facts.md §7.2`.
- Pool manifest `sampler_params` STALE-but-frozen caveat unchanged (see `POOL_MANIFEST_ERRATA.md`).
