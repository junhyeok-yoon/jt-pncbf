# ADOPTED — v2.8.5 quadrotor_3d, exponential hazard geometry ell = 0.125 (seed 42)

Copy-only secured promotion (`06_workflow` §2.5 item 3, §6.3; `04_eval` §7.5), taken under
`v2.8.5_results.md` §13.4 **C** — *"Secured promotion: SOTA-of-record only. One snapshot,
`data/secured_data/v2.8.5/seed42/` for run `set__20260811-185046__seed42`, with `step_045000.pt` as the
adopted checkpoint."* The source run directory remains intact where it was filed; every file below is a
**copy**, never a move, and no file under `data/runs/` was modified, renamed or deleted.

## Provenance

- **System / version:** quadrotor_3d, v2.8.5, **single seed, seed 42**, read from the run's own
  `config.yaml` (`run.system`, `run.version`, `run.seed`).
- **Run id:** `v2.8.5__jt__20260811-185046__seed42`
- **Source run dir:**
  `data/runs/v2.8.5/set__20260811-185046__seed42/v2.8.5__jt__20260811-185046__seed42/`
- **Terminal state:** `done` at step 50000 (round 500), `halt_reason null`, from the run's own
  `status.json` copied here.
- **Deliverable identity:** exponential hazard geometry, `hazard.geom_form = exp`, `hazard.ell = 0.125`
  (`config.yaml:428-430`). The v2.8.5 clip control carries no `hazard` block at all.
- **git commit at run creation (`git_commit.txt`):**
  `bdf2e8d049417ed2cff106c3ecb3f874f6737746 DIRTY` — the tree was dirty at launch; recorded, not
  repaired.

## THE ADOPTED CHECKPOINT IS A FIXED-STEP CHECKPOINT, NOT THIS RUN'S `best.pt`

**ADOPTED: `checkpoints/step_045000.pt` @ step 45000 (round 450).**

- full sha256 `89659767854f5e3c89f964652f797614ae04017dd2fa6815a781bd095844ace0` (sha8 `89659767`)
- bytes 5 459 239

**Why a fixed step and not `best.pt`.** This run's `best.pt` is a **different file at step 50000**
(sha256 `88dffc20951eb11e797f330e28f327a2621354bc7a2464722066d73b088165d2`). `best.pt` is selected by
`train.py` on the **in-loop** cell, where this run's record is `best_cps 0.823929131839131` at
`best_step 50000` (`status.json`, copied here). The version adopts the **registered-cell argmax**
instead, which falls at step 45000: on the registered cell `step_045000.pt` scores **0.870125** while
`best.pt` scores **0.861442**. The in-loop cell and the registered cell are different pools with
different filter settings, and they do not agree on which step is best. This follows the precedent the
same version set for the exp0.175 lineage, whose superseded adoption is preserved in this directory.

**Consequence a reader must carry: any check that reads `best.pt` for this snapshot reads the wrong
file.** `best.pt` and `final.pt` are copied here because `04_eval` §7.5 names them, not because either
is adopted.

## The registered row this checkpoint backs

- **Ledger line backed:** `docs/ledger.md` **L294** — the bold `quadrotor_3d` row of v2.8.5, alias
  `exp0.125 @45000 ADOPTED`, `cps_v2` 0.8701.
- **Backing artifact:** `data/runs/v2.8.5_rescore/row__exp0.125__step45000.json` (`cell_id`
  `exp0.125__step45000`). Every field of the ledger row was read from that JSON at write time
  (`v2.8.5_results.md` §13.4 D).

| field | value, read from the row JSON |
|---|---|
| `cps_v2` | **0.870125** (`GUARD_cps` 0.87012531260843) |
| 95 % CI | [0.8472, 0.8919] (`cps_ci_lo` 0.8471596779146777, `cps_ci_hi` 0.8918571394394254) |
| round / step | 450 / 45000 |
| reach | 0.9525 |
| collision (obstacle / band_lo / band_up) | 0.0230 (0.0060 / 0.0165 / 0.0005) |
| oob / stuck / timeout | 0.0000 / 0.0000 / 0.0245 |
| infeasibility (episode) | 0.080416 |
| EMPTY / SINGULAR (active) | 0.069726 / 0.007066 |
| saturation rate | 0.4296 |
| active steps | 141 668 |
| outcomes over 2000 episodes | goal 1905, collision 46, timeout 49 |

**Classification.** The bold row is a `06_workflow` §2.5 **basis** classification — the highest usable
evaluation on the current basis — and **NOT** a `04_eval` §5 CI-separated beat. The row is single-seed
and its CI overlaps those of exp0.25, exp0.10 and CLIP50 at every round (`v2.8.5_results.md` §1.2,
§13.4 A).

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
| `empty_source` | `last_empty` |

**Two facts the reader must not conflate** (both verifiable from the files in this directory):

1. This run's `config.yaml` sets `filter.empty_fallback: {mode: none, k: 10}` and `filter.empty_mode:
   argmin`. The bold row's cell names the shipped deploy fallback `{kstep, phases 1, k 3}`. That is a
   deploy-time override applied by the scoring harness, **not** a setting of this run's `config.yaml`.
2. This run's own `pool_manifest.json` pins the canonical `full` pool (n 2000, seed 23456). The bold row
   was scored on the **`fullcb` pool `3682a4e3…`** in a separate re-score session. The
   `eval_metrics.csv` / `eval_episodes.csv` secured here are this run's own evaluations on its own
   cells, and are **not** the number the ledger row carries.

## A PRIOR ADOPTION WAS SUPERSEDED, AND ITS RECORD IS PRESERVED HERE

This directory previously held the promotion of **`step_033000.pt` from the exp0.175 lineage**
(`set__20260810-033417__seed42`, cps 0.8520). `v2.8.5_results.md` §13.4 A **withdraws and supersedes**
that promotion — "*that row is registered but not bold*" — and §13.4 C makes this directory the single
snapshot for run `set__20260811-185046__seed42`.

**Nothing of the prior adoption was deleted.** Before any file here was overwritten, the complete prior
snapshot was copied to

`data/secured_data/v2.8.5/seed42/superseded__20260812__exp0.175_step033000/`

and every copy was verified byte-identical (md5, both sides) after writing. It holds the prior
`ADOPTED.md`, `config.yaml`, `eval_metrics.csv`, `eval_episodes.csv`, `pool_manifest.json`,
`git_commit.txt`, `status.json`, `checkpoints/step_033000.pt` and the prior `checkpoints/best.pt`
(exp0.175's step 28500) — 9 files. The prior `ADOPTED.md` cites a ledger line **L288** that does not
correspond to any line of the ledger as it stands; that citation is part of the preserved historical
record and is not corrected here.

`checkpoints/step_033000.pt` **also remains at the root of this snapshot**, alongside the adopted
`step_045000.pt`. It is retained rather than deleted, and it is **not** adopted. Its identity is
recorded in the digest table below so it cannot be mistaken for part of the current §7.5 set.

## Pinned SHA-256 — every file in this snapshot

`ADOPTED.md` is the identity record and the only file in the set that carries digests
(`06_workflow` §6.1). No digest below appears in any tracked prose document.

| file | bytes | sha256 | state |
|---|---:|---|---|
| `checkpoints/step_045000.pt` (**the adopted checkpoint**) | 5 459 239 | `89659767854f5e3c89f964652f797614ae04017dd2fa6815a781bd095844ace0` | copied 2026-08-12 |
| `checkpoints/best.pt` (step 50000 — **not** adopted) | 5 453 925 | `88dffc20951eb11e797f330e28f327a2621354bc7a2464722066d73b088165d2` | copied 2026-08-12 |
| `checkpoints/final.pt` (step 50000 — **not** adopted) | 5 454 035 | `4690077a63de3108dea64406225957712170d5a5d5532ca18558dc358b813901` | copied 2026-08-12 |
| `config.yaml` | 8 019 | `405beb2981fcb0e7f772665c9d020d1ab7db58bac2cb0e510e0c60a2af5cb203` | copied 2026-08-12 |
| `git_commit.txt` | 47 | `aaeb97ff53703e1be65c51d03adb90b8aa5c5061163b596faaf48ec461eef6e6` | copied 2026-08-12 |
| `eval_metrics.csv` | 12 888 | `e3f89b8c9f3a8062dd6daa53d846fcd0d65d90e259fa0d934b77b80310df74e3` | copied 2026-08-12 |
| `eval_episodes.csv` | 16 975 961 | `79e2fbfb800e11ac8d016940e78885a2c9eaa25e1069b083a7e035362983f703` | copied 2026-08-12 |
| `pool_manifest.json` | 66 997 | `9328bca7000948cc8796f68735aa62a6630249fb649a3c53086378a644ab3614` | copied 2026-08-12 |
| `status.json` | 189 | `dbc1363d2de01ac21c44d4f4ea99c6acf80ed6074128e1419cb9e23d22470576` | copied 2026-08-12 |
| `report.md` | 469 | `00de9b09146692835f1c94006dd9b504b4990601fec85fa12074bc7f10a8946e` | copied 2026-08-12 |
| `figures/trajectory_grid_A.png` | 559 100 | `ddccb7e015f2fe1f09ef035c2765849ff26e98fca8628075196c47b5d26753ff` | copied 2026-08-12 |
| `figures/trajectory_grid_B.png` | 409 445 | `81b722043661ab63c065c08919f1d57e303023202b5e94836c64734888d2abb3` | copied 2026-08-12 |
| `figures/cbf_contour.png` | 388 172 | `be5647f6ef0f084c94e93303a42bcc6a18d79ce3040ea40080d05e5d89841f1a` | copied 2026-08-12 |
| `checkpoints/step_033000.pt` (**superseded prior adoption**, exp0.175 lineage — retained, not adopted, not part of the §7.5 set) | 5 459 239 | `73adf0bef2df80c4f72d1166dd5a78f420895d41ad3d2337b369b550cfce88f9` | retained from the prior snapshot |

Every copy was verified byte-for-byte against its source by md5 after writing. **No mismatch, 13 of 13.**
`ADOPTED.md` (this file) is the record itself and carries no self-hash.

**The §7.5 file set is COMPLETE for this run** — unlike the superseded exp0.175 candidate, which had
neither `final.pt` nor `report.md` nor figures because it died of a CUDA OOM at step 37199.

## Excluded from this snapshot, with the reason

| excluded | reason |
|---|---|
| `metrics.csv` (212 181 B, per-step training telemetry) | excluded by `04_eval` §7.5 / `06_workflow` §6.3 |
| `tensorboard/` | excluded by `04_eval` §7.5 (event files) |
| `checkpoints/step_*.pt` other than `step_045000.pt` (33 files) | not part of the §7.5 set; only the adopted step is required to back the bold row |
| `figures/inloop/` | per-step training telemetry; §7.5 names only the three final-eval figures, and all three are present |
| `eval/{bandopen,mixed,tilt60}/` | per-cell dual-scoring evaluations; not named by the §7.5 set, and no number in the bold row comes from them |
| `eval_action_stream.npz`, `band_offset_inloop.csv`, `handoff.json` | non-standard run artifacts, not named by the §7.5 set |

## Status

- **Single seed (seed 42).** No `data/secured_data/v2.8.5/aggregate/` — no multi-seed aggregation was
  performed, and `v2.8.5_results.md` §13.4 I holds multi-seed escalation as not to be considered for now.
- **Not yet committed.** `git add` is a git mutation and is the Researcher's action (`06_workflow`
  §2.5 item 5, §2.6). Until it lands, §6.3's "present **and committed**" is only half satisfied.
