# ADOPTED — v2.8.0 quadrotor_3d SOTA deliverable (seed 42)

Copy-only secured promotion executed at the v2.8.0 close (2026-08-02), per Researcher approval
("approve promotion: dual deliverable step 42000 (cf948104) → data/secured_data/v2.8.0/seed42/").
Originals remain in place at the source run dir; this is a copy, not a move.

- **System / version:** quadrotor_3d, v2.8.0 (single-seed, seed 42).
- **Deliverable:** dual arm (`filter.projection = dual_solve`), joint-training best.pt at step 42000,
  new terminal (ω_G = 0.48), shipped fallback `kstep phases 1 k 3`.
- **Source run dir:** `data/runs/v2.8.0/set__20260731-015517__seed42/v2.8.0__jt__20260731-015517__seed42/`
- **git commit at copy (git_commit.txt):** `73b9700ad7f639ce9029e0d652a40bd1a6e2f034 DIRTY`
- **Ledger row:** `docs/ledger.md` L253 (bolded as the quadrotor_3d SOTA at this close — first eligible
  per-rotor-plant baseline; no prior 3-D bold existed, v2.7.2 was released as plant-incompatible).

## Headline evaluation (canonical pool `eval_full_quadrotor-3d-d2r_n2000_seed23456`, n=2000)
- **cps 0.7919** [0.7623, 0.8232]; reach 0.9295; collision 0.0440 (obst .0095 / floor .0335 / ceil .0010);
  stuck 0.0000; timeout 0.0265; infeasibility 0.1211; saturation 0.4809. (`s3_eval/m4_dual.json`)
- Dual scoring (standalone lineage, `m3_v2.8.0_dual.json`): **cps_tilt60 0.9206** [0.905, 0.935],
  **cps_bandopen 0.8818** [0.859, 0.901] (band-crossing rate 0.0345; 51/69 reached goal after crossing).
- Driver lineage (`w4_driver_reference.json`): cps_tilt60 0.9230 / cps_bandopen 0.8862 (cross-lineage
  band +0.0024 / +0.0044; within-lineage spread 0).

## Pinned SHA-256 (secured copies)
| file | sha256 |
|---|---|
| checkpoints/best.pt (the deliverable, sha8 cf948104) | `cf9481047a33bc0775f1529b03e60a13322ee1ff40814c8077bb3de1a23557ea` |
| checkpoints/step_042000.pt | `d74ba44fcdf49a62dc871bcfc048cec98e2c0bb7f51d6caef5676992fbabb9a2` |
| config.yaml | `62f045c163e45708ce66f89acb5848c6a995269c44ef4fc5603d6b4fc140a24e` |
| pool_manifest.json | `97c1ac66737d6883521ebc2e8dfa12fcbbbc4dcbdf2a4102a460f723988fd315` |
| eval_metrics.csv | `800f298c5d29dea8b2cbb05e0e57f0b70591a5bd46a257a9f267940360743a87` |
| eval_episodes.csv | `68e872a87613febbb52e0a80dfff3373baa96923d8690d8bee4d671aea546296` |
| report.md | `d45e996fcf8462f0d0d7061a07a01f0be23d10775c4b8160664b5f972dffe0a3` |
| status.json | `1e3284d75b44e7e2e0acbeb4956e082aa782eae2eea819f25c03a05566f0f1cd` |
| canonical pool `0ef3751b` (pool_sha256, referenced) | `0ef3751bd4d095d8658df92cca009d250d5b2f01d8325aad7dc262e2aadd4695` |

## Secured file set
`checkpoints/{best.pt, step_042000.pt}`, `figures/` (full subtree), `config.yaml`, `eval_metrics.csv`,
`eval_episodes.csv`, `pool_manifest.json`, `git_commit.txt`, `report.md`, `status.json`. Per-step
`metrics.csv` and non-standard artifacts (band_offset_inloop.csv, eval_action_stream.npz, jac_classes.csv,
tensorboard/, other-step checkpoints, final.pt) were **excluded** per the §2.5 standard set.

## Notes
- Single-seed close (seed 42); multi-seed escalation not run (Researcher decision).
- D4 (actuator-lag) prediction is **unscored** at this close (`rem:lag-null`; the τ-sweep completed 4/30
  cells — see `docs/versions/v2.8.0_results.md` §2 and `close_retrieve.md`).
- git staging of this directory (`git add -f`) is a git mutation and was **not** performed by the Executor
  (this dispatch's header allows read-only `git status` only); it is left for the Researcher.
