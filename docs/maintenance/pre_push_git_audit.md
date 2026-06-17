# Pre-push git audit (READ-ONLY — nothing staged/committed/pushed; proposal only)

Full working-tree state + a commit plan for the v2.2.0 work. **No `git add/commit/push/checkout/tag`
was run; no file was edited except this report.** HEAD is on `main` tracking `origin/main`, both at
`060a6b5` (v2.1.0) — **no unpushed commits; only uncommitted working changes.**

## ⚠ Headline finding (read first)

**`docs/versions/` is git-ignored** (`.gitignore:38` — "Version / development documents, kept
locally, not committed"). `git ls-files docs/versions/` = **0 tracked files**. So **every v2.2.0
diagnostic report under `docs/versions/v2.2.0/*.md` is NOT committable** — they stay local by repo
policy. A commit of "the v2.2.0 work" therefore captures the **scripts + the new module + the version
bump**, but **NOT the written findings**. If the reports are meant to live in the repo, `.gitignore:38`
must be changed first (a separate decision — not assumed here). Same applies to the not-yet-written
`v2.2.0_results.md` (it would be ignored too).

## Part 1 — full working-tree state

`git status --porcelain`: 4 modified (tracked), 24 untracked entries. Grouped:

### (a) v2.2.0 diagnostic docs — `docs/versions/v2.2.0/*.md`
- **All git-IGNORED** (`.gitignore:38 docs/versions/`); 0 tracked, 0 shown in `git status`.
- Includes all 19 reports + `_results_source_extraction.md`. `v2.2.0_results.md` **does not exist yet**
  (only `_results_source_extraction.md` is present) — and would also be ignored if written.
- → **not part of any commit.** (Local-only.)

### (b) v2.2.0 analysis scripts — `scripts/analysis/*.py` (all 21 untracked/new, committable)
`scod_failure_diagnostic, stage0_reliability_probe, stage1_decomposition, stage1_gate_and_log,
stage1_gate_corrected, stage1_scod_build, stage2_authority_collapse, stage2_confusion_3x3,
stage2_decomposition, stage2_detection_rates, stage2_eval, stage2_hocbf_comparison,
stage2_hocbf_deploy_n2000, stage2_l2qp_compare, stage2_leadtime, stage2_lower_alpha_deploy,
stage2_policy_obstacle_response, stage2_stuck_collision_decomp, stage2_stuck_mechanism,
stage2_stuck_obs_grids, stage2_velocity_dependence_char` — all `??` (new). None tracked.

### (c) maintenance files (untracked/new this session, committable)
- `scripts/maintenance/reorganize_data.sh` (`??`).
- `docs/maintenance/` (`??`) — `data_reorg_plan.md` + this `pre_push_git_audit.md`. **`docs/maintenance/`
  is NOT ignored** (only `docs/versions/` is), so these ARE committable.

### (d) version bump — `src/_version.py` + `src/configs/exp_config.yaml` (+ `pyproject.toml`)
The v2.2.0 version triple (all `M`, intended):
- `src/_version.py`: `__version__ = "v2.1.0"` → `"v2.2.0"`.
- `src/configs/exp_config.yaml`: `run.version: "v2.1.0"` → `"v2.2.0"`.
- `pyproject.toml`: `version = "2.1.0"` → `"2.2.0"`. **This is the third leg of the version bump — NOT
  stray.** (The task flagged pyproject as possibly unintended; the diff shows it is the version bump.)

### (e) pre-existing modified tracked files — diffs shown
- **`pyproject.toml`** — see (d): the version bump `2.1.0 → 2.2.0`. **Recommend INCLUDE** (part of the bump).
- **`docs/ledger.md`** — the diff adds **(1) one blank line** and **(2) the v2.1.0 lookahead-rejection
  note**:
  > `Note: v2.1.0 ran no training run; SOTA unchanged (v2.0.1 seed-42, cps 0.8852, remains version-SOTA). A lookahead-alpha filter was tested as a 6-arm x 3-seed eval_only ablation (L0-L5); all arms regressed vs L0 and the axis was rejected, … retained on disk.`
  - **This is REAL content, not a stray IDE/blank-line-only edit** — it is the v2.1.0 ledger note that
    was never committed (the v2.1.0 commit `060a6b5` committed code but not this note; `orientation_audit`
    §2.7 already cited it from the working tree). It is a **v2.1.0** note (there is **no v2.2.0 ledger
    row** — correct, since v2.2.0 is measurement-only with no cps/SOTA change).
  - **USER DECISION (not decided here):** (i) **include** it — documents v2.1.0 in the committed ledger
    (recommended; it is legitimate and otherwise lost); (ii) **discard** `git checkout -- docs/ledger.md`
    — drops both the note and the blank line; (iii) split it into its own commit. The lone added blank
    line is cosmetic/benign either way.

### (f) anything else modified/untracked
- `src/common/filter_hocbf.py` (`??`) — the new analytic HOCBF filter module (the one **new `src/`
  production-code** file, used by the comparison scripts). Committable; recommend INCLUDE.
- Nothing stray/unexpected found: the 4 `M` files are the version triple + the ledger note; all `??`
  are this session's v2.2.0 scripts/module/maintenance files. No scratch, no accidental data files.

## Part 1.2 — what is git-IGNORED (NOT in any commit)
- **`docs/versions/`** (`.gitignore:38`) → all v2.2.0 reports + `v2.2.0_results.md` (when written). Local-only.
- **`data/*` except `data/secured_data/`** (`.gitignore:41-43`). So the **data/ reorg of this session
  is filesystem-only and produces ZERO git changes**: the lookahead relocation
  (`data/previous_runs/v2.1.0_lookahead/`) and the dissection move-back (`data/diagnostics/…194620`)
  are under ignored paths. `secured_data/` (58 tracked files) is unchanged. Confirmed: `git status`
  shows no `data/` entries.

## Part 1.3 — branch / log / tags
- Branch `main`, tracking `origin/main`; **HEAD == origin/main == `060a6b5`** → working tree is **not
  ahead/behind**; a new commit would be exactly 1 ahead and push fast-forwards.
- `git log --oneline -5`: `060a6b5` v2.1.0 → `13102db` v2.0.1 → `97f0657` initial v2.0.0.
- Tags local: `v2.0.1`, `v2.1.0`. **`git ls-remote --tags origin` confirms `v2.1.0` (060a6b5) and
  `v2.0.1` are already on origin** → "push v2.1.0" is NOT needed (already pushed); no tag re-push.

## Part 2 — proposed commit plan (NOT executed)

**One commit** (v2.2.0 measurement-only diagnosis), containing the committable substance:
- **Include:** version triple (`src/_version.py`, `src/configs/exp_config.yaml`, `pyproject.toml`);
  the new module `src/common/filter_hocbf.py`; all 21 `scripts/analysis/*.py`;
  `scripts/maintenance/reorganize_data.sh`; `docs/maintenance/` (`data_reorg_plan.md` +
  `pre_push_git_audit.md`).
- **User decision:** `docs/ledger.md` (the real v2.1.0 note — recommend include).
- **Cannot include (gitignored):** `docs/versions/v2.2.0/*.md` (the reports), `v2.2.0_results.md`
  (not written; would be ignored). The data/ reorg (no git change).
- **No cps/SOTA claim** in the message — v2.2.0 changed no cps (measurement-only); SOTA stays
  v2.0.1 seed-42 cps 0.8852.

**Pending per the user's earlier instruction:** `v2.2.0_results.md` has **not** been written. Because
`docs/versions/` is gitignored, it would not be committed regardless; if the user wants it in the repo,
that requires changing `.gitignore:38` (separate decision). Flag, do not assume.

**Tag:** propose `v2.2.0` (for consistency — v2.1.0, also diagnosis-only, was tagged), but **do not
create** (the user tags deliberately; a measurement-only version tag is optional).

Proposed commit message:
```
v2.2.0: deploy-time reliability + failure-mechanism diagnosis (measurement-only; SOTA unchanged)

No training, no control/filter/cps change. Adds the v2.2.0 diagnostic harnesses + the analytic
HOCBF oracle module:
- SCOD epistemic + CBF-residual reliability axes; residual is the practical failure detector,
  SCOD the collision-vs-stuck type axis (analysis scripts).
- analytic relative-degree-2 HOCBF filter (src/common/filter_hocbf.py) for the stuck/collision
  oracle comparison.
- collision root cause = learned V_S control-authority (L_g h = dh/dv) collapse; a velocity-aware
  value is the located fix target. Lowering alpha_unsafe refuted (worsens collisions).
- data/ tidy (filesystem-only; data/ is gitignored) + reorg plan.

cps/SOTA unchanged: v2.0.1 seed-42 cps 0.8852 remains version-SOTA. Diagnostic reports live under
docs/versions/v2.2.0/ (gitignored, local-only).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```

## Part 3 — exact commands (written, NOT run — await approval)
```bash
# from repo root, on main (HEAD == origin/main == 060a6b5)

# 1) version bump + new module + scripts + maintenance docs (NOT docs/versions — gitignored)
git add src/_version.py src/configs/exp_config.yaml pyproject.toml
git add src/common/filter_hocbf.py
git add scripts/analysis/*.py scripts/maintenance/reorganize_data.sh
git add docs/maintenance/

# 2) docs/ledger.md — USER DECISION (real v2.1.0 note + 1 blank line):
git add docs/ledger.md                 # (A) include the v2.1.0 note  [recommended]
# git checkout -- docs/ledger.md       # (B) discard it (drops the note + blank line)

# 3) sanity: confirm exactly what is staged, and that NO docs/versions or data/ path slipped in
git status                              # expect: staged set above; docs/versions/* + data/* absent (ignored)
git diff --cached --stat

# 4) commit (message above; multi-line via -F or repeated -m)
git commit -m "v2.2.0: deploy-time reliability + failure-mechanism diagnosis (measurement-only; SOTA unchanged)" \
           -m "<body as above>" \
           -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"

# 5) push (fast-forward; 1 commit ahead)
git push origin main

# 6) OPTIONAL, deliberate tag (do only if the user wants a v2.2.0 tag):
# git tag -a v2.2.0 -m "v2.2.0: measurement-only diagnosis; SOTA unchanged"
# git push origin v2.2.0
```

## STOP — awaiting approval
Nothing was staged, committed, pushed, tagged, or checked out. **Two items need a user decision before
committing:** (1) `docs/ledger.md` — include the v2.1.0 note or discard; (2) whether the v2.2.0
**reports** should be committable at all (currently `docs/versions/` is gitignored → they are not). On
approval, run the Part-3 block.
