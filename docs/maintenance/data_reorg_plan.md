# data/ reorganization plan (PROPOSAL — read-only audit; nothing moved; awaiting approval)

Audit of `data/` clutter + a reference-safe move plan. **Nothing has been moved, renamed, deleted,
or edited.** The only safe, reference-clean move is archiving the 20 v2.1.0 lookahead run dirs that
sit at the `data/` root; everything else is either git-tracked + heavily referenced (`secured_data/`),
hard-coded by the v2.2.0 scripts and cited in v2.2.0 docs (`diagnostics/`), or an already-organized
archive subdir referenced by one script (`previous_runs/`). The move script
`scripts/maintenance/reorganize_data.sh` (dry-run by default) is NOT executed.

**Decisive constraint — `.gitignore`:** `data/*` is git-ignored **except** `!data/secured_data/`
(`.gitignore:41-43`). So **only `data/secured_data/` is git-tracked** (58 files); the lookahead runs,
`diagnostics/`, and `previous_runs/` are untracked. → moves of those use **plain `mv`** (not
`git mv`, which would fail on untracked paths). `secured_data/` is tracked and **must not move**.

## Part 1 — inventory (what is where, what references it)

`data/` root has 23 dirs, 0 loose files: 3 organized subdirs + 20 lookahead run dirs (the clutter).

| top-level entry | size | class | git | references (file:line) | move? |
|---|---|---|---|---|---|
| `secured_data/` | 48M | (a/b) canonical pools + SOTA checkpoints | **tracked** | many (below) | **NO** |
| `diagnostics/` | 138M | (c) v2.2.0 diagnostics + dissection sets | ignored | many v2.2.0 scripts + docs (below) | **NO** |
| `previous_runs/` | 1.4G | (e) v2.0.0/v2.0.1 training runs + misc | ignored | `stage1_scod_build.py:59` + historical doc paths | **LEAVE** (recommend) |
| `v2.1.0__…lookahead…` ×20 | 29M total | (d) v2.1.0 lookahead eval-only ablation | ignored | **none (0 refs anywhere)** | **ARCHIVE** |

### 1a/1b — `secured_data/` (canonical; tracked; MUST NOT MOVE)
Contents: `pools/` (`eval_full_di_n500_seed23456.pkl` + `.manifest.json`, `eval_inloop_di_n200_seed12345.pkl` + manifest); `v2.0.1/{seed42,seed12345,seed99}/` (each: checkpoints/, config.yaml, eval_*.csv, figures/, pool_manifest.json, report.md, status.json, ADOPTED.md) + `aggregate/`; `oc_pncbf/v2.0.0__20260529-001441__seed42/` (OC baseline). **SOTA checkpoint = `data/secured_data/v2.0.1/seed42/checkpoints/best.pt`.**

References (all REQUIRE this path; do not move):
- `data/secured_data/pools` — `src/eval/build_pools.py:23` (`DEFAULT_OUTPUT_DIR`), `src/eval/run_full.py:50`, `src/frameworks/jt_pncbf/train.py:56`, `src/frameworks/oc_pncbf/train.py:50`, `scripts/verification/{pool_preview.py:14, hardnet_oc_eval.py:23, gpu_profile.py:35}`.
- `data/secured_data/pools/eval_full_di_n500_seed23456.pkl` — `scripts/analysis/stage1_scod_build.py:62`.
- `data/secured_data/v2.0.1/seed42/checkpoints/best.pt` (SOTA, the `CKPT` constant) — `scripts/verification/jt_failure_dissect.py:31`; `scripts/analysis/{stage0_reliability_probe.py:53, stage1_scod_build.py:60, stage2_hocbf_comparison.py:45, stage2_hocbf_deploy_n2000.py:55, stage2_policy_obstacle_response.py:35, stage2_velocity_dependence_char.py:42}` (and indirectly every v2.2.0 script that imports `stage2_hocbf_deploy_n2000.CKPT`).
- `data/secured_data/oc_pncbf/v2.0.0__…/checkpoints/best.pt` — `scripts/verification/hardnet_oc_eval.py:26`.
- Docs cite `data/secured_data/...` as canonical: `docs/protocol/{00,04,05,06}.md`, `docs/versions/v2.0.1_results.md`, `docs/versions/v2.2.0/orientation_audit.md` (`…/aggregate/multi_seed_report.md`).

### 1c — `diagnostics/` (v2.2.0 outputs; hard-coded by scripts + cited by v2.2.0 docs; DO NOT MOVE)
Subdirs: `v2.2.0_hocbf/`, `v2.2.0_stage2_largeN/`, `v2.2.0_stage1_scod/`, `v2.2.0_stage0_probe/`, `v2.2.0_scod_diagnostic/`, `v2.1.0_failure_signal_pilot[_repro]/`, and two `v2.0.1__…failure_dissect_n2500_…{194620,194915}/` dissection sets.
- **The N=30000 and N=2000 eval pools live HERE, not in secured_data:** `diagnostics/v2.2.0_stage2_largeN/eval_full_di_n30000_seed20260605.pkl` and `diagnostics/v2.2.0_hocbf/eval_deployN2000_di_n2000_seed20260617.pkl`. They are **regenerable diagnostic pools** (seeded), cited by path in `stage2_decomposition.md` / `stage2_hocbf_deploy_n2000.md` — NOT canonical/secured. Leave in place.
- Hard-coded `OUT`/dir constants (moving/renaming breaks them): `v2.2.0_stage2_largeN`(+`/episodes`) — `stage2_eval.py:50`, `stage2_decomposition.py:27`, `stage2_stuck_mechanism.py:36`, `stage2_l2qp_compare.py:29`, `stage2_policy_obstacle_response.py:36`, `stage2_stuck_collision_decomp.py:47`, `stage2_authority_collapse.py:48`, `stage2_hocbf_comparison.py:56`; `v2.2.0_hocbf` — `stage2_hocbf_comparison.py:46`, `stage2_hocbf_deploy_n2000.py:56`, `stage2_velocity_dependence_char.py:43`; `v2.2.0_stage1_scod`(+ `scod_fisher.pt`,`/episodes`) — `stage1_scod_build.py:63`, `stage1_decomposition.py:26-27`, `stage1_gate_and_log.py:3`, `scod_failure_diagnostic.py:38`; `v2.2.0_stage0_probe` — `stage0_reliability_probe.py:55`; `v2.2.0_scod_diagnostic` — `scod_failure_diagnostic.py:39`; `…failure_dissect…194915` — `scod_failure_diagnostic.py:36` + docs (`v2.1.0_results.md`, `v2.1.0/detection_readiness_diagnostic.md`, `v2.0.1/{jt_pretraining_build,failure_modes_dissection}.md`); `v2.1.0_failure_signal_pilot[_repro]` — `failure_signal_pilot.py:23,27`.
- **Optional, low priority:** `diagnostics/v2.0.1__…failure_dissect…194620/` has **0 references** (the cited set is `…194915`, 13 refs). It is an older duplicate dissection run, safe to archive — but it lives inside the already-organized `diagnostics/` and is small, so it is listed as an OPTIONAL move (off by default in the script).

### 1d — the 20 v2.1.0 lookahead run dirs (the clutter; fully reference-safe to archive)
`data/v2.1.0__20260530-*__lookahead_*` (20 dirs, 29M total): the L0–L5 × 3-seed lookahead-alpha eval-only ablation + the N0/β0 baseline + one `_smoke`. Each holds config.yaml, eval_*.csv, figures/, git_commit.txt, pool_manifest.json, report.md, status.json, tensorboard/.
- **References: ZERO.** `grep -rn "v2.1.0__20260530\|lookahead_N" src/ scripts/ docs/` = 0 matches. The DEFAULT_OUTPUT_ROOT=`data` constant (`jt_pncbf/train.py:55`, `oc_pncbf/train.py:49`) only governs where NEW runs are written — it does not reference these existing dirs, so moving them is safe.

### 1.3 — are any v2.1.0 runs canonical / cited as SOTA?
**No.** Per `docs/ledger.md:32` and `docs/versions/v2.1.0_results.md:307-313` (confirmed in `orientation_audit.md` §2.7): v2.1.0 ran no training run, added **no bolded ledger row**; the lookahead ablation was an `eval_only` 6-arm × 3-seed dose-response, **all arms regressed vs L0**, the axis was rejected, and the runs are recorded in `v2.1.0_results.md` "rather than as ledger rows. Saved eval outputs … retained on disk." They are retained-on-disk artifacts, **not cited by path** anywhere → **fully archivable** (no securing needed).

## Part 2 — proposed target layout (NOT executed)

Minimal, reference-safe. The only move is the 20 lookahead dirs → a single archive subdir.

```
data/
  secured_data/        # UNCHANGED — canonical, git-tracked (pools, v2.0.1/seed*, oc_pncbf, aggregate)
  diagnostics/         # UNCHANGED — v2.2.0_* outputs + dissection sets (hard-coded by scripts + docs)
  previous_runs/       # UNCHANGED (recommended) — already an organized archive of v2.0.x runs
  archive/
    v2.1.0_lookahead/  # NEW — the 20 v2.1.0 lookahead eval-only ablation runs moved here
      v2.1.0__20260530-..._lookahead_..._seed.../   (×20)
```

| item | destination | reference-safe? | files/lines to update if moved |
|---|---|---|---|
| 20 `data/v2.1.0__*lookahead*` | `data/archive/v2.1.0_lookahead/` | **SAFE (0 refs)** | none |
| `data/secured_data/` | (stays) | n/a — DO NOT MOVE | n/a |
| `data/diagnostics/` | (stays) | n/a — DO NOT MOVE (breaks all v2.2.0 scripts + doc paths) | n/a |
| `data/previous_runs/` | (stays; recommended) | NOT safe if moved | `scripts/analysis/stage1_scod_build.py:59` (`PREV_RUN`), + historical doc paths in `docs/versions/v2.0.1/{jt_pretraining_build,failure_modes_dissection}.md` |
| `diagnostics/…failure_dissect…194620` (optional) | `data/archive/old_dissection/` | SAFE (0 refs) — OPTIONAL | none (the cited set is `…194915`) |

**Why `previous_runs/` is NOT moved (recommendation):** it is already a single organized subdir (not root clutter), it is 1.4G, and moving it would break `stage1_scod_build.py:59` (the SCOD-Fisher rebuild source) and leave the historical v2.0.x doc paths further out of date for marginal benefit. If a future cleanup wants it under `archive/`, that move is listed in the script as an OPTIONAL, off-by-default block with the one required `sed` to `stage1_scod_build.py:59` noted (NOT applied).

**Nothing is secured-promoted:** no v2.1.0 artifact is canonical (Part 1.3); the N=30000/N=2000 pools under `diagnostics/` are regenerable, not authoritative, so they stay in `diagnostics/` (moving them to `secured_data/` would require updating the hard-coded script paths + the doc citations — not recommended).

## Part 3 — the move script (written, NOT run)

`scripts/maintenance/reorganize_data.sh` — dry-run by default (`echo`s the commands); `--apply` to
execute; `--with-optional` to also archive the unreferenced `…194620` dissection set. It uses plain
`mv` (paths are git-ignored), refuses to touch `secured_data/`, and is idempotent (skips if already
moved). **It has not been executed.** No `sed`/config edit is applied by the script; the single
optional `previous_runs` reference update is printed as a manual step only.

| move | from | to | safe/needs-update |
|---|---|---|---|
| 1 (default) | `data/v2.1.0__*lookahead*` (20 dirs, 29M) | `data/archive/v2.1.0_lookahead/` | **SAFE — 0 refs** |
| 2 (optional, `--with-optional`) | `data/diagnostics/v2.0.1__…failure_dissect…194620` | `data/archive/old_dissection/` | SAFE — 0 refs |
| (not in script) | `data/previous_runs/` | `data/archive/previous_runs/` | NEEDS-UPDATE — `stage1_scod_build.py:59` + 2 historical docs; left as a documented manual option |

## Correction applied (2026-06-17) — lookahead runs relocated to `previous_runs/`

The original plan archived the lookahead runs to `data/archive/v2.1.0_lookahead/`. For consistency
with the existing prior-runs archive, the **20 v2.1.0 lookahead run dirs were relocated to
`data/previous_runs/v2.1.0_lookahead/`** (a single grouping subdir, since they are one ablation's 20
variants — L0–L5 × 3 seeds + N0/β0 baseline + smoke — which keeps `previous_runs/` from ballooning;
this differs slightly from the flat v2.0.x layout there but is cleaner for a 20-dir batch). The move
was reference-safe (re-confirmed 0 references in src/scripts/docs; `previous_runs/` is git-ignored, so
plain `mv` and no git-state change), and `data/previous_runs/v2.0.1__20260529-171057__seed42`
(`stage1_scod_build.py:59`) is a distinct existing dir unaffected by adding the new subdir.
`secured_data/` (58 tracked files) and the `diagnostics/v2.2.0_*` sets are byte-untouched. The
emptied `data/archive/v2.1.0_lookahead/` was removed.

**Resolved:** the earlier `--with-optional` run had moved the **unreferenced** older dissection set
(`…failure_dissect…194620`, 0 code refs) to `data/archive/old_dissection/`. Per user decision it was
**moved back to `data/diagnostics/`**, beside its cited sibling `…194915` (its origin/natural home),
and the now-empty **`data/archive/` was removed**. Final `data/` top level: `diagnostics/`,
`previous_runs/`, `secured_data/` (no `archive/`).
