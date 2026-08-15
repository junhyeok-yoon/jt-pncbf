# ADOPTED — v2.9.1 quadrotor_3d, `SCENELAW_C1000_SIG0` (seed 42)

Copy-only secured promotion (`06_workflow` §2.5 item 3, §6.3; `04_eval` §7.5). The source run
directory remains intact where it was filed; every file below is a **copy**, never a move, and no
file under `data/runs/` was modified, renamed or deleted. Nothing already under
`data/secured_data/` was modified or removed by this promotion — `data/secured_data/v2.9.1/` did
not exist before it and held no prior adoption.

## Provenance

- **System / version:** quadrotor_3d, v2.9.1, **single seed, seed 42**, read from the run's own
  `config.yaml` (`run.system`, `run.version`, `run.seed`, lines 183-186).
- **Run id:** `v2.9.1__jt__20260813-121049__seed42`
- **Source run dir:**
  `data/runs/v2.9.1/set__20260813-121049__seed42/v2.9.1__jt__20260813-121049__seed42/`
- **Terminal state:** `phase done` at `current_step` 10000 (round 1000), `halt_reason null`,
  `best_step` 9300, `best_cps` 0.8463254671694636 (the **in-loop** cell, pool `inloopv2` seed
  145678) — from the run's own `status.json` copied here. `report.md` records `wallclock_s`
  11833.431 and an in-loop peak of 0.846984 at step 9750, which is 0.000658 above the selected
  step and inside the 0.002 `early_stop_min_delta` hysteresis, so it never displaced the selection.
- **Value-init lineage:** `run.value_init_run_id` `v2.7.6__oc__20260725-043415__seed42`,
  `run.value_init_sha8` `b2cdaddd` (`config.yaml:187-188`).
- **git commit at run creation (`git_commit.txt`):**
  `58f1706f8cedc9cbb8166d28ac6da57c1aaae279 DIRTY` — the tree was dirty at launch; recorded, not
  repaired.

## THE ADOPTED CHECKPOINT

**ADOPTED: `checkpoints/step_009300.pt` @ step 9300 (round 930).**

- full sha256 `5407906c7998a0d3f26d1aadd0c592700fe5e31e3cc1630baa26ad92cefd62dc` (sha8 `5407906c`)
- bytes 5 459 367

**Its relation to `best.pt`.** Step 9300 is the step `status.json` records as `best_step`, so
`best.pt` carries the **same weights**. That is not an assumption here: both files were loaded on
CPU after the copy and their `pi_state` (6 tensors), `v_s_state` (16) and `v_s_target_state` (16)
compare **element-wise identical**, and both record `step` 9300. The two files are nevertheless
**different bytes** — `best.pt` is 5 454 053 B, sha256
`373fba81ad31a80f3bf95d22b5b717cd73da646ed6fe4c7818e982b445d0429d` — because the two save paths
serialise the same payload differently. The fixed-step file is the one adopted, so that the ledger
row's checkpoint is named by an immutable step rather than by a name a later run could re-point.

## The registered row this checkpoint backs

- **Ledger line backed:** `docs/ledger.md` **L304** — the bold `quadrotor_3d` row of v2.9.1, alias
  `best.pt SCENELAW_C1000_SIG0`, `cps_v2` 0.8722. It became the bold row in the same edit that
  released the v2.8.5 `exp0.125 @45000 ADOPTED` row (L294) from bold.
- **Adoption step / round:** **step 9300 / round 930.**
- **Backing artifact:** `data/runs/v2.9.1/rescore/row__v291__SCENELAW_C1000_SIG0__step9300.json`
  (`cell_id` `v291__SCENELAW_C1000_SIG0__step9300`). Every field of the ledger row was read from
  that JSON.

| field | value, read from the row JSON |
|---|---|
| `cps_v2` | **0.872223** (`GUARD_cps` 0.8722230754153334) |
| 95 % CI | [0.8471, 0.8962] (`cps_ci_lo` 0.8471190303566686, `cps_ci_hi` 0.8962370590105604) |
| round / step | 930 / 9300 |
| reach | 0.9550 (CI [0.9460, 0.9640]) |
| collision (obstacle / band_lo / band_up) | 0.0250 (0.0080 / 0.0170 / 0.0000), CI [0.0185, 0.0320] |
| oob / stuck / timeout | 0.0000 / 0.0005 / 0.0195 |
| infeasibility (episode) | 0.075090 |
| EMPTY / SINGULAR (active) | 0.060537 / 0.006893 |
| EMPTY / SINGULAR (episode mean step frac) | 0.067216 / 0.007974 |
| EMPTY / SINGULAR (episode share any) | 0.5715 / 0.0705 |
| saturation rate | 0.3643 |
| active steps | 141 584 |
| outcomes over 2000 episodes | goal 1910, collision 50, timeout 39, stuck 1 |

**Classification.** The bold row is a `06_workflow` §2.5 **basis** classification — the highest
usable evaluation on the current basis — and **NOT** a `04_eval` §5 CI-separated beat. Against the
row it supersedes (v2.8.5 `exp0.125` @ step 45000, `cps` 0.87012531260843, CI
[0.8471596779146777, 0.8918571394394254]):

- the difference is **+0.002098** read on the two rows' own artifacts — **0.25×** the 0.0083
  admissibility floor. Read instead against that row's 4-decimal ledger cell 0.8701 the difference
  is +0.002123, 0.26× the floor; both readings are far below the floor and neither changes the
  classification.
- the two 95 % **intervals overlap** ([0.8471, 0.8962] against [0.8472, 0.8919]).
- both rows are **single seed 42**.

No separation is claimed in either direction, on any channel.

**What is new relative to the superseded row.** Three things, and they are read off the two
`config.yaml` files secured here, which differ in 26 lines and nothing else:

1. the **v2.9.1 training scene law** — five clearance floors moved together:
   `init_feasibility_margin` 0.05 → **0.0**, and `start_obstacle_clearance` **0.0**,
   `goal_obstacle_clearance` **0.05**, `band_margin` **0.0**, `arena_margin` **0.0**, which the
   v2.8.5 `config.yaml` does not carry as keys at all (`config.yaml:74-81` here against
   `data/secured_data/v2.8.5/seed42/config.yaml:74-77`);
2. **`schedules.sigma.sigma_min` 0.3 → 0.0** (`config.yaml:264`) — the sigma floor removed;
3. the **compressed schedule at one fifth the update total** — `training.jt.n_steps` 50000 →
   **10000** (`config.yaml:196`) at `collection.jt.collect_every` 100 → **10**
   (`config.yaml:283`), i.e. 10 000 gradient updates over 1000 collection rounds against 50 000
   over 500; with it `schedule_n_steps` 50000 → 5000, `vs_warmup_steps` 2000 → 200,
   `sigma_pi.decay_steps` 5000 → 500, and the two logging cadences `eval.cadence` 1500 → 150 and
   `metrics_log_every` 200 → 20.

`sigma_min` was **not** a registered axis of v2.9.1 — `changes.md` §3 declares one substantive
axis, the clearance floors on the training scene law — and no verdict on `sigma_min` as an axis is
drawn here or in the ledger row. The three registered observables for this run are scored against
`SCENELAW_C1000` in `docs/versions/v2.9.1/final_scoring.md` §12.5, a pairing that is not one of
`changes.md` §2's two: A1 HOLDS, A2 FALSIFIED, A3 HOLDS.

## WHY `config.yaml` IS PART OF THIS SNAPSHOT

`config.yaml` is the **reference recipe the three remaining JT cells derive from**. The scene law,
the sigma floor and the compressed schedule above are all and only readable from it; without it
those cells cannot be rebuilt from the secured artifact alone. It is copied here verbatim, 8 115 B,
sha256 `aaba3cfdfd7c078c586d94b208da0dd4a1100936793a13586a5f951988c1cbee`.

## The eval cell the row was scored on — read back off the row JSON, not off `config.yaml`

| key | value in the row's `cell_read_back` |
|---|---|
| pool | `eval_fullcb_quadrotor-3d-d2r_n2000_seed823456.pkl`, `pool_sha8` `3682a4e3`, n 2000 |
| `filter_class` / `projection` | `HardNetFilter` / `dual_solve` |
| `empty_fallback` | `{kstep, phases 1, k 3}` |
| `alpha` (safe, unsafe) | (2.0, 100.0) |
| `gamma_margin` | 0.0 |
| `terminal` | (0.15, 0.3, 0.3) |
| `value_ceiling` / `n_vs` | 1.3 / 2 |
| `eval_batch_size` | 2000 |
| `box_klamp_enabled` | false |
| `empty_source` / `singular_source` | `last_empty` / `last_singular` |
| hazard geometry | `geom_form exp`, `ell` 0.125 (`config.yaml:432-434`) |

**Two facts the reader must not conflate** (both verifiable from the files in this directory):

1. This run's `config.yaml` sets `filter.empty_fallback: {mode: kstep, k: 3, phases: 1}` and
   `filter.empty_mode: argmin`; the pool the row was scored on is the **`fullcb` pool
   `3682a4e3…`** named in a separate re-score session, whereas this run's own
   `pool_manifest.json` pins the canonical `full` pool (n 2000, seed 23456).
2. The `eval_metrics.csv` / `eval_episodes.csv` secured here are **this run's own** evaluations on
   its own cells — its `final` row at step 9300 on pool `full` seed 23456 reads `cps`
   0.7598021894493454, and its `final_insertion_{lqr,frozen,live}` rows read −0.857 / 0.681693 /
   0.700439. **None of these is the number the ledger row carries.** The ledger row's 0.872223
   comes only from the row JSON named above.

## Pinned SHA-256 — every file in this snapshot

`ADOPTED.md` is the identity record and the only file in the set that carries digests
(`06_workflow` §6.1). No digest below appears in any tracked prose document.

| file | bytes | sha256 | state |
|---|---:|---|---|
| `checkpoints/step_009300.pt` (**the adopted checkpoint**) | 5 459 367 | `5407906c7998a0d3f26d1aadd0c592700fe5e31e3cc1630baa26ad92cefd62dc` | copied 2026-08-13 |
| `checkpoints/best.pt` (step 9300 — same weights, different bytes; not the adopted file) | 5 454 053 | `373fba81ad31a80f3bf95d22b5b717cd73da646ed6fe4c7818e982b445d0429d` | copied 2026-08-13 |
| `checkpoints/final.pt` (step 10000 — **not** adopted) | 5 454 163 | `eaadf06c5cda55979e5eadd235f530955de657804a563674246f99e10f9e3072` | copied 2026-08-13 |
| `config.yaml` | 8 115 | `aaba3cfdfd7c078c586d94b208da0dd4a1100936793a13586a5f951988c1cbee` | copied 2026-08-13 |
| `git_commit.txt` | 47 | `47e7328714818c5a47316e712912a37d57a1c63cba6fb4905ebdf140a2bcc6c8` | copied 2026-08-13 |
| `eval_metrics.csv` | 24 442 | `fb20f96f4d3f71ca0ac28da738d1a8b469d9be480bcb45c8a2e74eeb479d1724` | copied 2026-08-13 |
| `eval_episodes.csv` | 31 417 203 | `1762b908f62895623a019a731e511ba31ec1d0edfececf709e7f60fbb74f3e63` | copied 2026-08-13 |
| `pool_manifest.json` | 66 997 | `9328bca7000948cc8796f68735aa62a6630249fb649a3c53086378a644ab3614` | copied 2026-08-13 |
| `status.json` | 189 | `c7ba672f6620630dc24bc2f0e5f4a6c015d204fd0c377374efbcf12b24fed849` | copied 2026-08-13 |
| `report.md` | 468 | `a310cca4bf348a782f9d5623305e0dea0c244b43a4571369bd2fb4126dde202e` | copied 2026-08-13 |
| `figures/trajectory_grid_A.png` | 501 881 | `3ee31b0f3f5c17238a532c6d708eca7f5183e1cda991524780e844c9d36d8ac6` | copied 2026-08-13 |
| `figures/trajectory_grid_B.png` | 371 544 | `06141b11d9eff81ad48e808791d8d65362e2556cf7b8654a3350972335a2df43` | copied 2026-08-13 |
| `figures/cbf_contour.png` | 352 594 | `5260446da06ac6ee933416e8a482805423720ad9ee124e07db4f43a4ddd775f9` | copied 2026-08-13 |

Every copy was verified by sha256 against its source after writing. **No mismatch, 13 of 13.**
`ADOPTED.md` (this file) is the record itself and carries no self-hash.

**The §7.5 file set is COMPLETE for this run:** `config.yaml`, `git_commit.txt`,
`eval_metrics.csv`, `eval_episodes.csv`, `pool_manifest.json`, `status.json`,
`checkpoints/{best.pt, final.pt}`, `figures/{trajectory_grid_A,trajectory_grid_B,cbf_contour}.png`,
`report.md`, `ADOPTED.md` — plus the adopted fixed-step checkpoint, which §7.5's list does not name
and which is required to back the ledger row.

## Excluded from this snapshot, with the reason

| excluded | reason |
|---|---|
| `metrics.csv` (425 659 B, per-step training telemetry) | excluded by `04_eval` §7.5 / `06_workflow` §6.3 |
| `tensorboard/` (0 files on disk) | excluded by `04_eval` §7.5 (event files) |
| `checkpoints/step_*.pt` other than `step_009300.pt` (66 of 67 files) | not part of the §7.5 set; only the adopted step is required to back the bold row |
| `figures/inloop/` (268 files, 115 277 542 B) | per-step training telemetry; §7.5 names only the three final-eval figures, and all three are present |
| `eval/{bandopen,mixed,tilt60}/` (9 files, 1 318 144 B) | per-cell dual-scoring evaluations; not named by the §7.5 set, and no number in the bold row comes from them |
| `eval_action_stream.npz` (3 345 010 B), `band_offset_inloop.csv` (16 326 B) | non-standard run artifacts, not named by the §7.5 set |

## Status

- **No prior adoption existed.** `data/secured_data/v2.9.1/` did not exist before this promotion,
  so nothing was superseded, overwritten or preserved-aside here. The snapshot this one supersedes
  in the ledger lives at `data/secured_data/v2.8.5/seed42/` and was **not touched**: every
  pre-existing file under `data/secured_data/` was sha256-manifested before and after this
  promotion and every digest is unchanged.
- **Single seed (seed 42).** No `data/secured_data/v2.9.1/aggregate/` — no multi-seed aggregation
  was performed.
- **Not yet committed.** `git add` is a git mutation and is the Researcher's action
  (`06_workflow` §2.5 item 5, §2.6). Until it lands, §6.3's "present **and committed**" is only
  half satisfied.
