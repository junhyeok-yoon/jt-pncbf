# PPO baseline — quadrotor_3d SOTA (copy-only artifact)

**This directory is a COPY. The original stays in place so every existing citation keeps resolving.**

| field | value |
|---|---|
| original path | `data/runs/v2.8.2/ppo_baseline/v2.8.2__ppo__20260805-114505__seed42` (mtime 2026-08-05 13:02:15) |
| run-id | `v2.8.2__ppo__20260805-114505__seed42` ("rev-12") |
| config | `<run>/config.yaml` — copied verbatim |
| checkpoint | `<run>/checkpoints/best.pt` — 1,836,045 bytes, mtime 2026-08-05 12:57:29 |
| **checkpoint sha256[:16]** | **`45485f71f040ef60`** (verified identical in both copies) |
| scoring artifact | `<run>/rev12_score/ppo_ppo_rev12_114505.json` (mtime 2026-08-05 13:02:26) + three per-cell episode CSVs |
| standard cell | `fullscr40`, pool seed 623456 (`data/secured_data/pools/eval_fullscr40_quadrotor-3d-d2r_n2000_seed623456.pkl`), n = 2000 |
| **eval_batch_size** | **NOT RECORDED in the scoring artifact — see `docs/versions/v2.8.3/baseline_ppo_secure.md`** |
| gate cps | **0.6940** [0.65574, 0.73251] |

**Selection**: best by cps on the standard (gate) cell across all 14 stored PPO runs. The margin over
rev-11 is **+0.00525**, which is **7.3× smaller than the CI half-width (0.0384)** — rev-12 and rev-11 are
statistically indistinguishable on this cell, and rev-11 leads on `cps_tilt60` and gate reach. See the
build-log before treating this as a separated result.

**Class**: PPO baseline — NO certificate, NO filter. `infeasibility = 0` by construction. Different
deployment class from the JT rows; not comparable to a filtered arm without stating that.
