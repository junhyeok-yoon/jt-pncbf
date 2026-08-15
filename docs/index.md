# JT-PNCBF

Self-built research infrastructure for **Observation-Conditioned PNCBF** and
**Joint Training** for safe navigation in randomized obstacle environments.

This site is the single source of truth (SSOT). It is read by the Researcher,
the Strategist, and the Executor.

## Current state (dashboard)

!!! note "STATUS — keep this section short and always current"
    - **v2.9.1 CLOSED** (record `docs/versions/v2.9.1_results.md`; close decisions in its §13).
      Earlier closed: v2.8.5, v2.9.0. **No new version is open** — `src/_version.py`,
      `pyproject.toml` and `src/configs/exp_config.yaml` all still read **v2.9.1** and agree.
      **Every result below is single seed 42, with one exception: ledger L314 is a 3-seed
      aggregate (42, 12345, 99).** There is no `data/secured_data/v2.9.1/aggregate/`.
    - **v2.9.1 verdict, one line: the registered axis was falsified, a new basis row was secured,
      and no improvement was detected.** **A1 (`coll_band_lower`) is FALSIFIED on both registered
      pairs** — moving the five training clearance floors did not lower band-lower collision; the
      paired mean difference is **positive** on both (+0.000156 and +0.000396, 0.02× and 0.05× the
      0.0083 admissibility floor). L304 is secured as the `quadrotor_3d` basis row at **0.8722**,
      **+0.002098** over the superseded v2.8.5 row — **0.25×** the floor, with **overlapping** CIs
      ([0.8471, 0.8962] vs [0.8472, 0.8919]) and single seed on both sides. That is a
      `06_workflow` §2.5 **BASIS** classification and **not** an `04_eval` §5 CI-separated beat.
    - **Bold set (3, one per system — `unicycle` has none):**
      **quadrotor_3d L304 = 0.8722**, checkpoint `step_009300.pt`, secured at
      `data/secured_data/v2.9.1/seed42/` · **quadrotor_planar L313 = 0.8519**, checkpoint
      `data/secured_data/v2.7.0/seed42_iter5/checkpoints/best.pt` (`3b27d691…`) ·
      **double_integrator L314 = 0.8682** (3-seed mean), checkpoints
      `data/secured_data/v2.3.0/seed{42,12345,99}/checkpoints/best.pt`.
      **L313 and L314 moved onto their rows at this close** (from L132 and L62) as **basis
      re-scores of those rows' own checkpoints** — both new figures are **lower** than the rows they
      supersede (0.851927 vs 0.9036; 0.868247 vs 0.8698), so **neither is an improvement claim**.
      L132 and L62 keep their content and carry standing supersession lines.
    - **Presence check (§6.3), every system:** 3 of 3 bold rows PRESENT **by digest**, 5 of 5
      checkpoint entries (double_integrator is a 3-seed set), each matched against the snapshot's own
      `ADOPTED.md`. No promotion was needed for the bold moves — each new bold row scores its
      predecessor's checkpoint. **"Committed" is unverified here** — staging is the Researcher's action.
    - **Ledger:** `scripts/check_ledger.py` exit 0 — **296 rows, 42 blocks, 3 bolds, 0 violations,
      79 warnings**. v2.9.1 registers **L304–L315**. One warning is expected and disclosed: bold
      L313's `parent` is named by digest, which the run-id matcher cannot resolve.
    - **Table I, the eight cross-system cells (L305–L312), all n = 2000, all seed 42.** JT exceeds OC
      on all four systems: `double_integrator` 0.8805 → **0.9048** · `unicycle` 0.6705 → **0.8731** ·
      `quadrotor_planar` 0.6591 → **0.7757** · `quadrotor_3d` 0.6818 → **0.8680**. This is
      **structural coverage, not a tested hypothesis** — it was not pre-registered, and no CI is
      available on any of the eight rows.
    - **Promotion at close: nothing further.** `data/secured_data/` stands at **1724 files** —
      `v2.9.1/seed42/` (the L304 snapshot) plus `v2.9.1/experiments/` (**eight** entries, the Table I
      cells). An experiments entry is **bold-ineligible** under §6.3, which is why
      `double_integrator`'s highest current-basis figure (L309, 0.904757) is not the bold row.
    - **Pool of record: `fullcb`** — `eval_fullcb_quadrotor-3d-d2r_n2000_seed823456`, **unchanged by
      decision**. The z-law audit established that it was not built under the `04_eval` §6.1 vertical
      bound (252 of 2000 scenes outside), but that screen is not an impossibility instrument — L312's
      own artifact records 51 collisions in total, so **at least 201 flagged scenes did not collide**
      — and the corrected constrained-input doom certificate certifies **0 of 2000** vertically doomed
      with a non-vacuous gate 4. **Obstacle-coupled impossibility remains outside every instrument on
      record.** No `quadrotor_3d` row is re-scored.
    - **Deferred at this close, all still open.** The `04_eval` §6.2 **pool-registration delta** is
      drafted and not installed, to be discussed separately — until it settles, L307–L310 and L314 are
      comparable to each other and **not** to any historical DI/unicycle row. The **protocol delta is
      deferred** in full: `PF-1 … PF-10` are listed with before→after sketches in
      `v2.9.1_results.md` §10.5 and none is applied. **No source change this cycle** — PF-2
      (`filter_hardnet.py:192` has no caller-mode guard, so the eval-only property is
      config-enforced, not code-enforced) and PF-3 (`exp_config.yaml`'s three `empty_fallback`
      sub-blocks) stay as recorded defects.
    - **Carried from earlier versions, unchanged.** Baselines `data/baselines/{ppo, backup_cbf}/` and
      `data/secured_data/baselines/` populated; backup_cbf has **no checkpoint of its own** — two
      filter cells over the CTRL checkpoint, cited as *"hand-designed policy + online rollout
      certificate (implicit/backup CBF); NOT an analytic closed-form CBF."* **PPO ledger placement
      still awaits the Researcher.** Open items **C0** (`cps_v2` applicability on `quadrotor_3d` — an
      axis, not bookkeeping, since resolving it re-scores every registered row), **C21**, **C22**,
      **C23**, **C24**, `verify.sh`'s pre-launch state (C6), `eval_batch_size` still neither a config
      key nor a column of `eval_metrics.csv`, and the 31 unselectored `signed_h` sites. v2.8.5's
      `PF-6 … PF-16` are in `docs/versions/v2.8.5/build_log.md` §11; nine of eleven remain open.

## Map

- **Protocol** — rules, prohibitions, eval/env standards, workflow. Start here.
- **Versions** — one report per version (`vMAJOR.MINOR.PATCH`); kept locally.
- **Ledger** — one row per run (config, seeds, checkpoint hash, metrics, verdict).