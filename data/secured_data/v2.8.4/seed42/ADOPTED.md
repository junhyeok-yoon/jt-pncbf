# ADOPTED — v2.8.4 quadrotor_3d, ceiling arm C = 1.3 (seed 42)

Copy-only secured promotion (`06_workflow` §2.5 item 3, §6.3; `04_eval` §7.5). The source run directory
remains intact where it was filed; every file below is a **copy**, never a move, and no file under
`data/runs/` was modified, renamed or deleted.

The 2026-08-08 bolding pass (`docs/versions/v2.8.4/ledger_bold.md` DO-2) placed `best.pt` and
`config.yaml` here as a partial copy. This record completes that promotion to the standard file set; the
two files already present were verified byte-identical to source and **left untouched**.

## Provenance

- **System / version:** quadrotor_3d, v2.8.4 (single seed, seed 42).
- **Run id:** `v2.8.4__jt__20260808-133610__seed42`
- **Source run dir:**
  `data/runs/v2.8.4/set__20260808-133610__seed42/v2.8.4__jt__20260808-133610__seed42/`
- **Adopted checkpoint:** `checkpoints/best.pt` @ **step 24000**
  (`status.json`: `best_step 24000`, `best_cps 0.7664492567765719`, `current_step 30000`, `phase done`).
- **Deliverable identity:** ceiling arm, `network.value.ceiling = 1.3`, read back off the run's own
  `config.yaml`. This is the axis that distinguishes the run from the v2.8.2 CTRL at matched step 24000.
- **git commit at run creation (`git_commit.txt`):** `5848943cc1691f9c84852daf4a4adc1daa6be26c DIRTY`
- **Ledger line backed:** `docs/ledger.md` **L271** — the current bold quadrotor_3d row
  (`v2.8.4 | quadrotor_3d | 2026-08-08 | v2.8.4__jt__20260808-133610__seed42 | ARM (C=1.3) | 42 |
  cps_v2 0.8308`).

## The eval cell, read from the run's own `config.yaml`

| key | value in `config.yaml` |
|---|---|
| `env.goal_radius` / `goal_speed_radius` / `goal_angrate_radius` | 0.15 / 0.3 / 0.3 |
| `eval.max_steps` | 200 |
| `eval.full` | `n 2000`, `seed 23456` |
| `eval.in_loop` | `n 500`, `seed 12345` |
| `eval.bootstrap` | `n_resample 1000`, `seed 20260508` |
| `eval.cadence` | 1500 |
| `filter.projection` | `dual_solve` |
| `filter.alpha_safe` / `alpha_unsafe` | 2.0 / 100.0 |
| `filter.gamma_margin` | 0.0 |
| `filter.empty_mode` | `argmin` |
| `filter.empty_fallback` | `{mode: none, k: 10}` |
| `network.value.ceiling` | **1.3** |

**Two facts the reader must not conflate** (both verifiable from the files in this directory):

1. The **training/eval config's** `filter.empty_fallback` is `{mode: none, k: 10}`. The ledger row's
   eval cell names the **shipped deploy fallback** `{kstep, phases 1, k 3}`. That is a deploy-time
   override applied by the scoring harness, not a setting of this run's `config.yaml`.
2. This run's own `pool_manifest.json` pins the canonical `full` pool
   `pool_sha256 0ef3751b…` (n 2000, seed 23456). The ledger row L271 was scored on the **screened
   `fullcb` pool `3682a4e3…`**, in a separate rescore session whose artifact is
   `data/runs/v2.8.4/ceiling/stageC3_scores.json`. The `eval_metrics.csv` / `eval_episodes.csv`
   secured here are therefore the run's **own** end-of-run evaluation on the canonical pool, not the
   number the ledger row carries.

Run's own end-of-run evaluation (`eval_metrics.csv`, `mode = final`, step 24000, `pool full` seed 23456,
n 2000): cps 0.6773, reach 0.8915, collision 0.0765, infeasibility 0.1488, saturation 0.4906.
The ledger row's **cps_v2 0.8308** on `fullcb 3682a4e3` is the number the bold rests on.

## Pinned SHA-256 — every file in this snapshot

| file | bytes | sha256 | state |
|---|---:|---|---|
| `checkpoints/best.pt` (the adopted checkpoint) | 5 453 797 | `50dd979397239245a05de54e96cbafdc5fedb66c5f89b92420f42986c09330a6` | copied 2026-08-10 |
| `checkpoints/final.pt` | 5 453 907 | `c4f9a195aaa30d6d6fb365ad937ff36689fa6108562555e072a95bc8ada325ce` | copied 2026-08-10 |
| `config.yaml` | 7 981 | `3601fe454a7f243db3a704e0acd820b2902c2396db36e9f1ba1c1ebc74652468` | already present (2026-08-08), byte-identical to source |
| `eval_metrics.csv` | 8 339 | `0b5a7103576ff4552ed64a5efd7cf9ff45b80d7c2c277d67a433c41fd4d96418` | copied 2026-08-10 |
| `eval_episodes.csv` | 10 526 812 | `e4fab4ee220662a69c97ccd833f0a4fe6c16965e6385a869f1b15472176d2934` | copied 2026-08-10 |
| `pool_manifest.json` | 66 997 | `9328bca7000948cc8796f68735aa62a6630249fb649a3c53086378a644ab3614` | copied 2026-08-10 |
| `git_commit.txt` | 47 | `db8c66dd8490e4724dc04b001dc52a3466b3c2ad59a4234471c1fd12bd79eba9` | copied 2026-08-10 |
| `report.md` | 469 | `c9abaa1ee8528c2ebc7ff2da4e4f56724a86694b7a25735397fc95206aeaa212` | copied 2026-08-10 |
| `status.json` | 190 | `970976f7d6bb6335aea7c9a46451e1340ea14d76fd967f4141acf0fc00ea1764` | copied 2026-08-10 |
| `figures/trajectory_grid_A.png` | 472 776 | `4fce57300a7317aca47e1c4fe2d2c261bf3eed129ad0bcc1a4ddb481c7faf451` | copied 2026-08-10 |
| `figures/trajectory_grid_B.png` | 380 599 | `62183d97ec311ad1f9f94c609b72df58acc17ddb4fac740b979714448a81831d` | copied 2026-08-10 |
| `figures/cbf_contour.png` | 418 493 | `2063a61f65661e4ddb2732486258bc92bda8401f654fb2f5eb495f3da557c03b` | copied 2026-08-10 |
| `best.pt` (snapshot root — see below) | 5 453 797 | `50dd979397239245a05de54e96cbafdc5fedb66c5f89b92420f42986c09330a6` | already present (2026-08-08), left in place |

Every copy was verified byte-for-byte against its source by SHA-256 after writing. No mismatch.

## Layout note the Researcher should rule on

`best.pt` sits at the snapshot root as well as at the `04_eval` §7.5 path `checkpoints/best.pt`. The
root copy is the 2026-08-08 partial promotion; it is byte-identical to `checkpoints/best.pt` (same
digest `50dd9793…`). It was **not deleted** — this task is copy-only and never removes existing secured
content. Staging both duplicates 5 453 797 bytes in the repository; the root copy can be dropped from
the staging list, or removed, at the Researcher's direction.

## Excluded from this snapshot, with the reason

| excluded | reason |
|---|---|
| `metrics.csv` (126 642 B, per-step training telemetry) | excluded by `04_eval` §7.5 / `06_workflow` §2.5 item 3 |
| `tensorboard/` | excluded by `04_eval` §7.5 (event files) |
| `checkpoints/step_*.pt` (20 files) | not part of the §7.5 set; the adopted step is recoverable from `best.pt` itself (`step`/`best_step` 24000) and from `status.json` |
| `figures/inloop/` (81 per-step figures, ≈ 38 MB) | per-step training telemetry; §7.5 names exactly `figures/{trajectory_grid_A.png, trajectory_grid_B.png, cbf_contour.png}`, which is what was copied (the same reading as the v2.7.4 and v2.7.6 snapshots; v2.8.0 copied the whole subtree) |
| `eval/` subtree, `eval_action_stream.npz`, `band_offset_inloop.csv` | non-standard run artifacts, not named by the §7.5 set |

No file of the standard set was absent from the source run: all twelve resolved and were copied or found
already present.

## Status

- Single-seed (seed 42); no `data/secured_data/v2.8.4/aggregate/` — none was produced.
- **Not yet committed.** `git add` is a git mutation and is the Researcher's action (`06_workflow`
  §2.5 item 5, §2.6). Until it lands, §6.3's "present **and committed**" is only half satisfied.
