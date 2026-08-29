# JT-PNCBF

Self-built research infrastructure for **Observation-Conditioned PNCBF** and
**Joint Training** for safe navigation in randomized obstacle environments.

This site is the single source of truth (SSOT). It is read by the Researcher,
the Strategist, and the Executor.

## Current state (dashboard)

!!! note "STATUS — keep this section short and always current"
    - **v2.9.3 CLOSED** on **single seed 42** (record `docs/versions/v2.9.3_results.md`;
      close-sequence step 3 done 2026-08-31, steps 4 and 5 remain). `src/_version.py`,
      `pyproject.toml` and `src/configs/exp_config.yaml` all read **v2.9.3** and agree; the bump
      follows the push. Its block is **L341–L398**, 58 rows. The version's directory is
      `docs/versions/v2.9.3/`, 75 build-logs.
    - **v2.9.3 basis of record: L372**, `L363 COLDABL on fullvia` — `cps` **0.8903**, reach 0.9620,
      collision 0.0225, on `eval_fullvia_quadrotor-3d-d2r_n2000_seed823456` at n 2000, `eval.max_steps`
      400, `best.pt` @ step 31500, **single seed 42**. It is the joint pair's row on the pool the eval
      protocol specifies for `quadrotor_3d`.
    - **Bold rows after the close: L356, L357, L358, L372** — one per system, `bolds=4` for the whole
      ledger. **The 3-D quadrotor's bold moved from L359 to L372 on 2026-08-31** under the
      pool-of-record rule: per system the highest `cps_v2` on the pool the eval protocol specifies,
      within the paper's condition class (cold start, shipped input limit, deployed filter), at n 2000
      h400. **Intervals and the 0.0083 floor are not bold criteria** — the bold names the row the paper
      reports, not the row shown to be better, and every gap L372 opens on its own pool is inside the
      floor and is called a match in its verdict cell. TWR×2 rows are a different plant and are never
      eligible. L359 was un-bolded and is otherwise unchanged.
    - **Promotion: `data/secured_data/v2.9.3/seed42/`.** Four runs copied at the close — L356
      `set__20260819-190133__seed42`, L357 `set__20260819-190141__seed42`, L358
      `set__20260820-011541__seed42`, L363/L372 `set__20260820-215728__seed42` — each with
      `checkpoints/best.pt`, `config.yaml`, `status.json`, plus the 14 score and per-episode artifacts
      those rows name, under `artifacts/`. `ADOPTED.md` carries the digests. **The `06_workflow` §6.3
      presence check passes 4 of 4.** Originals were copied, never moved.
    - **v2.9.3 verdict, one line: a large single-seed measurement campaign that closed several axes
      negatively and set none.** It has **no registered axis** — `docs/versions/v2.9.3/changes.md` was
      named as pending and never written, so every row is a measurement or a diagnostic and none is an
      axis result. Cold start at a matched 40000-step budget **does not separate from warm on any of
      four systems** (all four intervals cover zero); the penalty route is exhausted at `cps` 0.71425
      against the pair's 0.89025; the depth-four analytic HOCBF reaches −0.2252. Its most reusable
      products are bases, not results: the `fullvia` pool, the 400-step cap, the cold-ablation
      checkpoint and the reproduction gates that tie them together.
    - **v2.9.3 open items: 20** (`docs/versions/v2.9.3_results.md` §13), of which 19 are open and O8 is
      closed. Two SOTA flags stand and neither is taken: **L375** on a doubled input limit, the
      version's only CI-separated beat and a different plant; and the `fullcb` rows L364/L365/L366 that
      outrank the row L359 used to be.
    - **PROTOCOL FOLLOW-UP (v2.9.3): applied at close.** Nine rules, `PF-v293-1` … `PF-v293-9`, are in
      the tree; close-sequence **step 4 is done**. One line each:
        - **PF-v293-1** — bold marks the row of record: highest `cps_v2` on the **pool of record**, the
          pool the registered evaluation cell names for that system. `06_workflow` §2.5.
        - **PF-v293-2** — rows on a **changed plant** (altered input limits, thrust-to-weight, bounds)
          belong to no class the bold ranks and are never eligible. `06_workflow` §2.5.
        - **PF-v293-3** — registration is **never deferred to close**; a close finds unregistered
          evaluations **by enumeration** of every `score__*.json` at `n >= 2000` and records each
          omission against the task that produced it. `06_workflow` §2.5.
        - **PF-v293-4** — **reproduction is the gate** for work standing on a registered row, and a
          re-scoring takes its **cap, pool and batch size from the cell it re-scores onto**.
          `06_workflow` §2.5.
        - **PF-v293-5** — a producer **never carries a pool identifier or source tag as a literal**;
          both are derived from the pool it loaded. `06_workflow` §2.5.
        - **PF-v293-6** — **every exported number names its artifact**; a value copied from a build-log
          is not an export. `06_workflow` §2.5.
        - **PF-v293-7** — **diagnostics take no ledger row**; their record is the version's
          `results.md`. `06_workflow` §2.5.
        - **PF-v293-8** — a number quoted in a **verdict cell** is the row's own or another registered
          row's, named by line; **a number that has no row is not carried there**. `06_workflow` §2.5.
        - **PF-v293-9** — **every score artifact carries the interval columns**, analytic and nominal
          conditions included, and a row whose artifact persists none is flagged by
          `check_ledger.py`. `04_eval` §7.2.
      **One tension the delta leaves open:** `04_eval` §6.2's pool table still names
      `eval_fullcb_quadrotor-3d-d2r_n2000_seed823456` as `quadrotor_3d`'s full pool, while PF-v293-1
      bolds L372 on `fullvia`. Reported, not edited — `docs/protocol/` is untouched by this close.
      The carried item from the previous version, `PF-v292-1`, is unchanged.
    - **v2.9.2 CLOSED** on **single seed 42** (record `docs/versions/v2.9.2_results.md`; close-sequence
      step 3 in its §16). Earlier closed: v2.8.5, v2.9.0, v2.9.1. Its block is L316–L340.
    - **v2.9.2 verdict, one line: the registered axis held on its own observable, no improvement was
      detected, and nothing was promoted.** The `c_gain` 0.3 → 0.0 ablation's falsifier **A1 is NOT
      falsified on either condition** — `coll_obstacle` rises 0.0310 → 0.0500 (OC, 62 → 100 episodes)
      and 0.0075 → 0.0090 (JT, 15 → 18). The OC effect is clean (paired +0.0190, CI [+0.0095,
      +0.0280], band channels identical to the episode); **the JT margin of 3 episodes crosses zero**
      (CI [−0.0020, +0.0050]) and sits below the 0.0083 floor. `cps` falls on both (−0.0624 and
      −0.0184), carried on the JT condition by infeasibility and `SINGULAR` (0.0955 → 0.2485), not
      by collisions.
    - **The adopted basis is `eval.max_steps` = 400, and it is NOT comparable to the 200-step rows.**
      Rows **L316–L340** are v2.9.2's block; L321–L340 are on the 400-step basis. The horizon is worth
      more than the admissibility floor on four rows (L305 +0.0511, L306 +0.0372, L312 +0.0173, L311
      +0.0136) and is **exactly inert on every unfiltered nominal** (0 of 2000 episodes change). Any
      comparison across the two caps must say which basis it uses. Note also that the shipped
      `episode_timeout` and `eval.max_steps` are both still **200**, so `03_train` R3's binding of
      training-label horizon to eval horizon does not hold for the adopted basis
      (`v2.9.2_results.md` §7.4.4).
    - **Bold set unchanged (3, one per system — `unicycle` has none):** **quadrotor_3d L304 =
      0.8722** · **quadrotor_planar L313 = 0.8519** · **double_integrator L314 = 0.8682** (3-seed
      mean). **No row was bolded, re-bolded or re-scored at this close.** All three remain scored at
      the 200-step cap while the version reports at 400 — the basis question is open and is with the
      Researcher (`v2.9.2_results.md` §13.2).
    - **Presence check (§6.3), every system, re-run at this close:** **3 of 3 bold rows PRESENT by
      digest**, 5 of 5 snapshot entries, each checkpoint matched by SHA-256 against its own
      `ADOPTED.md`. One file is present and unpinned —
      `v2.7.0/seed42_iter5/checkpoints/final.pt` — which is not an adopted checkpoint and does not
      affect the check. **"Committed" is unverified here**; staging is the Researcher's action.
    - **Ledger:** `scripts/check_ledger.py` exit 0 — **321 rows, 43 blocks, 3 bolds, 0 violations,
      79 warnings**. Rule 11 caps `verdict` and `eval_source` at 400 characters each, with the
      pre-cap text of all 124 relocated cells in `docs/ledger_verdicts.md` (321 sections).
    - **NOTHING WAS PROMOTED, AND THE PROMOTION DECISION IS PENDING.** No write was made under
      `data/secured_data/` at this close, no `ADOPTED.md` was authored, and no approval is on record
      for a v2.9.2 seed snapshot. `data/secured_data/v2.9.2/experiments/` holds four secured
      **diagnostics** written by an earlier v2.9.2 dispatch — §6.3 experiments entries, bold-ineligible
      and not a promotion.
    - **`00_constitution` §3 Prohibition 5 sweep completed** across `docs/ledger.md` and
      `docs/ledger_verdicts.md`, not only this version's rows: **zero** occurrences remain in either
      file. The uppercase condition label was renamed to **`REF`**, which renames the alias of **L271**
      and **L278** to `REF (C=1.3)` and their anchors to `…--ref-c-1-3--{1,2}`; build-logs closed
      before 2026-08-18 still cite the old spelling, and the mapping is recorded at the head of
      `docs/ledger_verdicts.md`. No number, date, path or bold marker changed.
    - **Open at this close** (`v2.9.2_results.md` §13, with what would close each): the `04_eval` §1 /
      `filter_hardnet.py` divergence on the singular clause — measured on **20 cells**, 14 % to 84 % of
      the deployed term, six row families shifting past the floor and the two definitions **ordering OC
      and JT oppositely**; the three bold rows off-basis; `docs/ledger_verdicts.md` and
      `scripts/maintenance/*.py` tracking status; the contaminated MPPI wall time (790.6 s, three GPU
      jobs resident); and **the deployed pair reaching 0 of 280 on the obstacle-free grid** while
      reaching 0.9770 on `fullcb`.
    - **PROTOCOL FOLLOW-UP.** No v2.9.2 build-log raised one. **PF-v292-1** is recorded at
      `docs/versions/v2.9.2/protocol_followup.md`: the undeclared scene contract at
      `src/common/kstep_fallback.py:99`, which the one failing test (**1 failed, 243 passed, 2 skipped**)
      turns on and by which the standing **PF-3** now bites. Two conflicts are reported and not acted
      on: `04_eval` §3b's specified yz plane against the drawn plane, and **`06_workflow` §2.5 against
      `scripts/check_ledger.py` on the ledger ordering key** — §2.5 says artifact timestamps are not an
      ordering key, the gate keys on earliest artifact mtime first.
    - **Deferred, all still open.** The `04_eval` §6.2 **pool-registration delta** is drafted and not
      installed — until it settles, L307–L310 and L314 are comparable to each other and **not** to any
      historical DI/unicycle row. The **protocol delta is deferred** in full: `PF-1 … PF-10` in
      `v2.9.1_results.md` §10.5, none applied. **No source change this cycle** — PF-2 and PF-3 stay as
      recorded defects, and PF-3 has grown to **four** `empty_fallback` sub-blocks with a now-false
      comment at `exp_config.yaml:335`.
    - **Carried from earlier versions, unchanged.** Baselines `data/baselines/{ppo, backup_cbf}/` and
      `data/secured_data/baselines/` populated; backup_cbf has **no checkpoint of its own**. **PPO
      ledger placement still awaits the Researcher** — v2.9.2 resolved which network the factorial
      cell uses (`step_166896.pt`, `cps` +0.6770, reproducing **L262**) and that it is **not** L249.
      Open items **C0**, **C21**–**C24**, `verify.sh`'s pre-launch state (C6), `eval_batch_size` still
      neither a config key nor a column of `eval_metrics.csv`, and the 31 unselectored `signed_h`
      sites. v2.8.5's `PF-6 … PF-16` are in `docs/versions/v2.8.5/build_log.md` §11; nine of eleven
      remain open.

## Map

- **Protocol** — rules, prohibitions, eval/env standards, workflow. Start here.
- **Versions** — one report per version (`vMAJOR.MINOR.PATCH`); kept locally.
- **Ledger** — one row per run (config, seeds, checkpoint hash, metrics, verdict).