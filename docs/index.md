# JT-PNCBF

Self-built research infrastructure for **Observation-Conditioned PNCBF** and
**Joint Training** for safe navigation in randomized obstacle environments.

This site is the single source of truth (SSOT). It is read by the Researcher,
the Strategist, and the Executor.

## Current state (dashboard)

!!! note "STATUS — keep this section short and always current"
    - **v2.8.5 close PREPARED — the version is NOT yet declared closed.** Active version:
      **v2.8.5**. Record: `docs/versions/v2.8.5_results.md`. **Every arm of this version is a single
      run at seed 42; no result below is a multi-seed aggregate.**
    - **Seven conditions, and where each stopped** (all seed 42, all under `data/runs/v2.8.5/`; the
      terminal is each run's own `status.json`, the `best.pt` step its `best_step`):
      `ell` 0.10 **done @50000**, best 43500 · `ell` 0.125 **done @50000**, best 50000 ·
      `ell` 0.175 **DEAD @37199 (CUDA OOM)**, best 28500, **no `final.pt` and no `report.md`** ·
      `ell` 0.25 **done @50000**, best 40500 · `ell` 0.35 **done @50000**, best 42000 ·
      CLIP50 (budget-matched clip control) **done @50000**, best 42000 ·
      COMPRESS500 **SIGTERM @4779 = round 477.9**, best 4350, 22 rounds of its budget unrun,
      **no `final.pt` and no `report.md`**. The two truncated runs' maxima are maxima of truncated
      series and are not comparable to one taken over 500 rounds.
    - **The axis returned no separation except an upper-side penalty.** On the registered cell the
      ten pairs among `ell` 0.10 / 0.125 / 0.175 / 0.25 and the clip control separate at **zero**
      rounds ≥ 300 against the 0.0083 admissibility floor; only `ell = 0.35` separates, and it is
      worse in every pair. The two frontier conditions (`ell` 0.125, 0.10) both completed 500 rounds
      and their placeholders are **closed** — the axis is **not** bracketed from below.
    - **The compressed-update condition is registered to v2.8.5, not delegated to v2.9.0** — the
      earlier delegation is withdrawn. Its verdict is **NOT RESOLVED**: it does not separate from its
      clip control at any round ≥ 300, and the matched terminal comparison the axis was designed
      around does not exist because it was signalled at round 477.9.
    - **Pool of record: `fullcb`** — `eval_fullcb_quadrotor-3d-d2r_n2000_seed823456`, n 2000,
      unchanged; its digest lives in the pool manifest, not here. `scripts/check_ledger.py` exit 0,
      **277 rows, 3 bolds, 0 violations**. **Sixteen v2.8.5 rows registered at L281–L296**: seven
      registered-cell `best.pt` rows (one per condition), the bold `ell` 0.125 @45000 row, five
      runs' own `final` rows on pool `full` seed 23456, two `B1` budget rows at step 30000, and the
      superseded `ell` 0.175 @33000 row, which stays registered but **not bold**.
    - **Bold state:** double_integrator v2.3.0 (flagged — `cps_v2` reads `-`, and the row names no
      checkpoint); quadrotor_planar v2.7.1 0.9036; **quadrotor_3d v2.8.5 `ell` 0.125 @45000 =
      0.8701** [0.8472, 0.8919] (ledger **L294**, adopted checkpoint `step_045000.pt`, secured at
      `data/secured_data/v2.8.5/seed42/`, digest pinned in that snapshot's `ADOPTED.md`). **The
      adopted checkpoint is a FIXED-STEP checkpoint, not that run's `best.pt`**, which is at step
      50000 and scores 0.8614 on the same cell. This is a **basis** classification under
      `06_workflow` §2.5 — single seed 42, and its CI overlaps `ell` 0.25, `ell` 0.10 and CLIP50 at
      every round — and is **not** a `04_eval` §5 CI-separated beat. Both the v2.8.4 ARM row (L271)
      and the v2.8.5 `ell` 0.175 row (L288) are un-bolded and carry standing supersession lines.
    - **Presence check (§6.3), every system:** 3 of 3 bold rows PRESENT **by digest**, 5 of 5
      checkpoint entries (the double_integrator bold is a 3-seed aggregate), each digest matched
      against the snapshot's own `ADOPTED.md`. `unicycle` carries 5 rows and no bold row, so §6.3 is
      vacuous for it. **"Committed" is unverified here** — staging is the Researcher's action.
    - Baselines: `data/baselines/{ppo, backup_cbf}/` and `data/secured_data/baselines/` populated.
      backup_cbf has **no checkpoint of its own** — two filter cells over the CTRL checkpoint; every
      citation carries *"hand-designed policy + online rollout certificate (implicit/backup CBF);
      NOT an analytic closed-form CBF."* **PPO ledger placement (PF-3) still awaits the Researcher.**
    - **Open items.** **C0** — `cps_v2` applicability on `quadrotor_3d`, an axis and not
      bookkeeping, since resolving it re-scores every registered row; it must not be silently
      dropped. **PF-5** — `train.py:693` gates the zero-value step on the safety-channel type rather
      than on `k_v`; tagged, not applied. **C21** — the two secured v2.7.0 `report.md` sentences
      citing `collection.py:204`'s hard-coded 0.0, disposition undecided, and the training-time
      empty rate unmeasurable while it stands. **C22** — the two literal `max=1.0` clamps inside
      `rpcbf_target`, the box_klamp basis question, and `docs/protocol.zip` alongside the
      authoritative `docs/protocol/`. **C23** — the automatic ledger-registration rule did not
      fire once across the version; all sixteen rows above were registered at the close, not as
      part of completing each eval. **C24** — the
      shell `grep` shim honours `.gitignore` and silently omits `tests/`, `scripts/analysis/` and
      `scripts/deck/`, so any earlier repo-wide sweep should be re-run and every count must name the
      binary that produced it. Also open: `verify.sh`'s state before launch (C6, unestablishable
      from disk); `eval_batch_size` still neither a config key nor a column of `eval_metrics.csv`;
      the 31 unselectored `signed_h` sites, none repaired. **v2.8.5's own protocol follow-ups are
      tagged PF-6 … PF-16** in `docs/versions/v2.8.5/build_log.md` §11; nine of the eleven are open.


## Map

- **Protocol** — rules, prohibitions, eval/env standards, workflow. Start here.
- **Versions** — one report per version (`vMAJOR.MINOR.PATCH`); kept locally.
- **Ledger** — one row per run (config, seeds, checkpoint hash, metrics, verdict).