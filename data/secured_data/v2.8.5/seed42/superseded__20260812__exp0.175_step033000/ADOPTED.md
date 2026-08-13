# ADOPTED — v2.8.5 quadrotor_3d, exponential hazard geometry ell = 0.175 (seed 42)

Copy-only secured promotion (`06_workflow` §2.5 item 3, §6.3; `04_eval` §7.5). The source run directory
remains intact where it was filed; every file below is a **copy**, never a move, and no file under
`data/runs/` was modified, renamed or deleted.

## Provenance

- **System / version:** quadrotor_3d, v2.8.5 (single seed, seed 42), read from the run's own
  `config.yaml` (`run.system`, `run.version`, `run.seed`).
- **Run id:** `v2.8.5__jt__20260810-033417__seed42`
- **Source run dir:**
  `data/runs/v2.8.5/set__20260810-033417__seed42/v2.8.5__jt__20260810-033417__seed42/`
- **ADOPTED CHECKPOINT: `checkpoints/step_033000.pt` @ step 33000 — NOT `best.pt`.**
  - full sha256 `73adf0bef2df80c4f72d1166dd5a78f420895d41ad3d2337b369b550cfce88f9` (sha8 `73adf0be`)
  - `best.pt` is a **different** file at step 28500
    (sha256 `20de4f6549864443ea60889673244722fcfec429537a4b310a45123d6a5495e6`) and is **not** what the
    bold ledger row rests on. It is copied here because `04_eval` §7.5 names it, not because it is
    adopted. Any check that reads `best.pt` for this snapshot reads the wrong file.
- **Deliverable identity:** exponential hazard geometry, `hazard.geom_form = exp`, `hazard.ell = 0.175`,
  read back off the run's own `config.yaml`. The v2.8.5 clip control carries no `hazard` block at all.
- **git commit at run creation (`git_commit.txt`):** `bdf2e8d049417ed2cff106c3ecb3f874f6737746`
- **Ledger line backed:** `docs/ledger.md` **L288** — the current bold quadrotor_3d row
  (`v2.8.5 | quadrotor_3d | 2026-08-11 | v2.8.5__jt__20260810-033417__seed42 | 1D exp0.175 | 42 |
  cps_v2 0.8520`), whose eval cell names this checkpoint by `sha8 73adf0be`.

## `final.pt` is ABSENT from this snapshot, and why

**There is no `final.pt` in the source run and none in this snapshot: the run died at step 37199 of a
CUDA out-of-memory error before it ever wrote one.** `final.pt` is written by the end-of-training path,
which this process never reached. The evidence is on disk:

- `status.json` (copied here): `current_step 37199`, `phase training`, `halt_reason null`,
  `best_step 28500`, `best_cps 0.8022952679331686`, `updated_at 2026-08-10T23:50:06+00:00`.
- `data/runs/v2.8.5/launch/stdout__A.log` ends in
  `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 512.00 MiB.` raised at
  `src/frameworks/oc_pncbf/collection.py:830` in `_cat_optional`, under
  `continuing_collector.advance_round -> buffer.append_batch`.
- The source `checkpoints/` directory therefore ends at `step_036000.pt`; there is no `final.pt` to
  copy. This is an absence of the artifact, not an omission of this promotion.

The same death is why the run has **no final full-pool evaluation**: its `eval_metrics.csv` holds 24
rows, all `mode = in_loop` on the `inloopv2` pool (seed 145678, n 2000), steps 1500 … 36000, and no
`mode = final` row. The adopted step's in-loop reading is
`step 33000, cps 0.8041488838857861, reach 0.9265, collision 0.0375, infeasibility 0.09783705371404633`;
that is an in-loop number on a different pool and is **not** the number the bold row carries.

## The eval cell, read from the run's own `config.yaml`

| key | value in `config.yaml` |
|---|---|
| `run.system` / `run.version` / `run.seed` | `quadrotor_3d` / `v2.8.5` / `42` |
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
| `network.value.ceiling` | 1.3 |
| `hazard.geom_form` / `hazard.ell` | **`exp` / `0.175`** |

**Two facts the reader must not conflate** (both verifiable from the files in this directory):

1. This run's `config.yaml` sets `filter.empty_fallback: {mode: none, k: 10}`. The bold ledger row's
   eval cell names the shipped deploy fallback `{kstep, phases 1, k 3}`. That is a deploy-time override
   applied by the scoring harness, not a setting of this run's `config.yaml`.
2. This run's own `pool_manifest.json` pins the canonical `full` pool `pool_sha256 0ef3751b…`
   (n 2000, seed 23456). The bold row was scored on the **`fullcb` pool `3682a4e3…`** in a separate
   re-score session whose artifact is
   `data/runs/v2.8.5/rescore/row__1D__exp0.175__step33000.json`. The `eval_metrics.csv` /
   `eval_episodes.csv` secured here are this run's own in-loop evaluations, not the number the ledger
   row carries.

## Pinned SHA-256 — every file in this snapshot

| file | bytes | sha256 | state |
|---|---:|---|---|
| `checkpoints/step_033000.pt` (**the adopted checkpoint**) | 5 459 239 | `73adf0bef2df80c4f72d1166dd5a78f420895d41ad3d2337b369b550cfce88f9` | copied 2026-08-12 |
| `checkpoints/best.pt` (step 28500 — **not** adopted) | 5 453 925 | `20de4f6549864443ea60889673244722fcfec429537a4b310a45123d6a5495e6` | copied 2026-08-12 |
| `config.yaml` | 8 019 | `621e0ae797339037534271d1ae7231f24459b9f005efef9b9c1d050787e32f4e` | copied 2026-08-12 |
| `eval_metrics.csv` | 8 492 | `163f31f26d11f6027bdc6a81a632b4a7d91611e76cb7997d625c3eb5df48af77` | copied 2026-08-12 |
| `eval_episodes.csv` | 10 827 344 | `d893711d99e80d40e9a1afca8529d04d6e6b621ef0911994c798840cddd864f1` | copied 2026-08-12 |
| `pool_manifest.json` | 66 997 | `9328bca7000948cc8796f68735aa62a6630249fb649a3c53086378a644ab3614` | copied 2026-08-12 |
| `git_commit.txt` | 41 | `df1c1ac1435e290c7af9ad967ae8d84c0f551ad90082de47b4a60d3b039199ab` | copied 2026-08-12 |
| `status.json` | 194 | `41be1ca886607a56e482ac8be194294a4c8939bf2630504755552c5d87c0dab6` | copied 2026-08-12 |

Every copy was verified byte-for-byte against its source by SHA-256 after writing. No mismatch.
`ADOPTED.md` (this file) is the record itself and carries no self-hash.

## Excluded from this snapshot, with the reason

| excluded | reason |
|---|---|
| `metrics.csv` (156 674 B, per-step training telemetry) | excluded by `04_eval` §7.5 / `06_workflow` §6.3 |
| `tensorboard/` | excluded by `04_eval` §7.5 (event files) |
| `checkpoints/final.pt` | **does not exist** — see the section above; the run died at step 37199 (CUDA OOM) before writing one |
| `checkpoints/step_*.pt` other than `step_033000.pt` (23 files) | not part of the §7.5 set; only the adopted step is required to back the bold row |
| `report.md`, `figures/{trajectory_grid_A,trajectory_grid_B,cbf_contour}.png` | do not exist — they are written by the final-eval path this run never reached |
| `figures/inloop/` (96 per-step figures) | per-step training telemetry; §7.5 names only the three final-eval figures, and none of those exists here |
| `band_offset_inloop.csv` | non-standard run artifact, not named by the §7.5 set |

## Status

- Single-seed (seed 42); no `data/secured_data/v2.8.5/aggregate/` — none was produced.
- **Not yet committed.** `git add` is a git mutation and is the Researcher's action (`06_workflow`
  §2.5 item 5, §2.6). Until it lands, §6.3's "present **and committed**" is only half satisfied.
