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

---

# 2026-08-10 — completion to the standard file set

Everything above is the original 2026-08-03 record and is unchanged. This section is an **addition**:
the 2026-08-03 promotion carried seven files and no `checkpoints/`, `figures/`, `eval_episodes.csv` or
`pool_manifest.json`. Those were copied on 2026-08-10 so the snapshot is the standard set of
`04_eval` §7.5 / `06_workflow` §2.5 item 3. Copy-only; nothing under `data/runs/` was touched, and no
file already here was overwritten.

## Why this snapshot has to be complete

`docs/ledger.md` **L267** (cps_v2 **0.8424** on `eval_fullscr41_quadrotor-3d-d2r_n2000_seed723456`,
pool `9a016919`) and **L268** (cps_v2 **0.8557** on
`eval_inloopv2_quadrotor-3d-d2r-mixed_n2000_seed145678`, pool `50bc2060`) — the `box_klamp` `K0_shipped`
deployment cells — are the two current-basis rows that outscore the bold quadrotor_3d row (L271, 0.8308
on `fullcb 3682a4e3`). Both were scored on **this** checkpoint. Whichever basis the Researcher settles
on for the quadrotor_3d bold, this checkpoint must be backed under §6.3.

**How that was resolved, from the artifacts and not from memory:**

1. Both rows' `parent` cell reads `v2.8.2__jt__20260803-063606__seed42`.
2. Both rows' `eval_source` cell names `ckpt v2.8.2__jt__20260803-063606__seed42/checkpoints/best.pt @
   step 24000`.
3. The K0 artifacts named in those rows —
   `data/runs/v2.8.4/box_klamp/K0_shipped__fullscr41.json` and `…__inloopv2.json` — each carry their own
   `"checkpoint"` field, both reading
   `/home/junhyeok/MIT/jt-pncbf/data/runs/v2.8.2/set__20260803-063606__seed42/v2.8.2__jt__20260803-063606__seed42/checkpoints/best.pt`.

All three agree. The source `checkpoints/best.pt` hashes to `c89f9aef…`, matching the digest this file
already pinned on 2026-08-03.

## The eval cell, read from this run's own `config.yaml`

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
| `network.value.ceiling` | *(absent — this is the CTRL arm; the v2.8.4 ARM sets 1.3)* |

Two cautions, both checkable against the files here:

1. The config's `filter.empty_fallback` is `{mode: none, k: 10}`. The K0 rows' cell names the **shipped
   deploy fallback** `{kstep, phases 1, k 3}` — a deploy-time override in the scoring harness, matching
   the `effective_filter_cell` block recorded inside each K0 artifact, not a setting of this
   `config.yaml`.
2. This run's `pool_manifest.json` pins the canonical `full` pool `0ef3751b…`. The K0 rows were scored
   on the **screened** pools `9a016919` (fullscr41) and `50bc2060` (inloopv2). So `eval_metrics.csv` /
   `eval_episodes.csv` here are this run's own end-of-run evaluation on the canonical pool
   (`mode = final`, step 24000: cps 0.6670, reach 0.8925, collision 0.0845, infeasibility 0.1484,
   saturation 0.4785), not the K0 numbers.

## Pinned SHA-256 — every file in this snapshot

| file | bytes | sha256 | state |
|---|---:|---|---|
| `checkpoints/best.pt` (the checkpoint L267/L268 were scored on) | 5 453 797 | `c89f9aef0cdb5499c3f0e7d868c2e9e4650632d845245d2a53421c6b4658e285` | copied 2026-08-10 |
| `checkpoints/final.pt` | 5 453 907 | `bc71d8e09e54d8af65a963c4521832cf46926f81ae735c3d613fefd2d7f81caa` | copied 2026-08-10 |
| `config.yaml` | 7 964 | `003cc0f89287ca1e17304850db8ca0de3bc9166d454af53d31f9add683428ec8` | already present (2026-08-03), byte-identical to source |
| `eval_metrics.csv` | 8 311 | `c958b5ad0431e92de877b5bad6edf4447238b6b180dff9805db31762457890a2` | already present, byte-identical to source |
| `eval_episodes.csv` | 9 964 980 | `fb7879375bd0fca14b8c29a273289b67e260a196129285fd883239ad6a6188eb` | copied 2026-08-10 |
| `pool_manifest.json` | 32 980 | `72eda38c50d022935504ee8f7b25753957aef8cdc275b3e9aaaf9ef6d260918d` | copied 2026-08-10 |
| `git_commit.txt` | 47 | `2855d35e6720d58cdfa9b82e10ba37b19969b043169dfbfa64c5002265dc3141` | already present, byte-identical to source |
| `report.md` | 469 | `1fd792d7b8df220853cdf61ace2a8df802c71c99ad95d4835f1a860f22b06b56` | already present, byte-identical to source |
| `status.json` | 190 | `233165b0aee58b764d02cbc1069e23fc226510a3b58e21ff3843900a34e8ac6c` | already present, byte-identical to source |
| `figures/trajectory_grid_A.png` | 516 708 | `a2d560670427b9b1fb2f860b2426c3151d5b2d94b04f2e9be4a052ba1cb883e4` | copied 2026-08-10 |
| `figures/trajectory_grid_B.png` | 487 036 | `1dd18f709250e44352a302a24800caab757efe8de1f6d3c4e123370f33d2b984` | copied 2026-08-10 |
| `figures/cbf_contour.png` | 423 464 | `aa567f01894507862c08da25b6afbe2c5a5a49c2394a107853ca9ce5aef0e559` | copied 2026-08-10 |
| `best.pt` (snapshot root — see below) | 5 453 797 | `c89f9aef0cdb5499c3f0e7d868c2e9e4650632d845245d2a53421c6b4658e285` | already present (2026-08-03), left in place |
| `dual_CTRL.json` (extra, beyond the §7.5 set) | 3 693 | `04bf916abec933c78b17515eb1571a492fae22215dee8daac90d334437dae415` | already present, kept |

`git_commit.txt` reads `c77a13a0011981c75c16b048eff25ee26f41cbbb DIRTY`. Every 2026-08-10 copy was
verified byte-for-byte against its source by SHA-256 after writing. No mismatch.

## Layout note the Researcher should rule on

`best.pt` sits at the snapshot root as well as at the §7.5 path `checkpoints/best.pt`. The root copy is
the 2026-08-03 promotion; it is byte-identical (same digest `c89f9aef…`). It was **not deleted** — this
task is copy-only and never removes existing secured content. Staging both duplicates 5 453 797 bytes;
the root copy can be dropped from the staging list, or removed, at the Researcher's direction.

## Excluded from this snapshot, with the reason

| excluded | reason |
|---|---|
| `metrics.csv` (121 273 B, per-step training telemetry) | excluded by `04_eval` §7.5 / `06_workflow` §2.5 item 3 |
| `tensorboard/` | excluded by `04_eval` §7.5 (event files) |
| `checkpoints/step_*.pt` (20 files) | not part of the §7.5 set; the adopted step is recoverable from `best.pt` and `status.json` (`best_step 24000`) |
| `figures/inloop/` (81 per-step figures, ≈ 38 MB) | per-step training telemetry; §7.5 names exactly the three top-level figures, which is what was copied |
| `eval/` subtree, `eval_action_stream.npz`, `band_offset_inloop.csv` | non-standard run artifacts, not named by the §7.5 set |

No file of the standard set was absent from the source run: all twelve resolved and were copied or found
already present.

## Status

- Single seed (seed 42); no `data/secured_data/v2.8.2/aggregate/` — none was produced.
- **Not yet committed.** `git add` is the Researcher's action (`06_workflow` §2.5 item 5, §2.6).
