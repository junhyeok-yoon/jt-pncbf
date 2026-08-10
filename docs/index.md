# JT-PNCBF

Self-built research infrastructure for **Observation-Conditioned PNCBF** and
**Joint Training** for safe navigation in randomized obstacle environments.

This site is the single source of truth (SSOT). It is read by the Researcher,
the Strategist, and the Executor.

## Current state (dashboard)

!!! note "STATUS — keep this section short and always current"
    - **v2.8.4 CLOSED (2026-08-10), tag `v2.8.4` = commit `30afc59c`.** Diagnostic version, single
      seed (42), no CI-separated improvement: two registered label-side axes (ceiling `C = 1.3`,
      gamma `horizon_final` 33), both |Δcps| below the 0.0083 admissibility floor. The gamma
      falsifier is scored and does **not** fire. Verdict, whole record, and what the tag holds and
      omits: `docs/versions/v2.8.4_results.md` (§0.1). **Active version: v2.8.5 — OPEN**,
      `docs/versions/v2.8.5/changes.md` installed.
    - **In flight: four v2.8.5 runs, seed 42, 50000 steps, concurrent on one GPU.** Arms A / B / C
      are the exponential hazard at `ell` 0.175 / 0.25 / 0.35 (PIDs 1335693 / 1336567 / 1337512);
      CLIP50 is the budget-matched clip control the v2.8.4 comparators lacked, its PID recorded in
      `docs/versions/v2.8.5/build_log.md` at launch. Parity and sanity gates green before launch.
      Matched-step tables go to the Researcher; nothing is registered, ranked or bolded from them.
    - **Pool of record: `fullcb`** — `eval_fullcb_quadrotor-3d-d2r_n2000_seed823456`, sha256
      `3682a4e3…`, unchanged. `scripts/check_ledger.py` exit 0, 261 rows, 3 bolds, 0 violations.
    - **Bold state:** double_integrator v2.3.0 (flagged — `cps_v2` reads `-`, and the row names no
      checkpoint); quadrotor_planar v2.7.1 0.9036; **quadrotor_3d v2.8.4 0.8308** (ledger L271,
      secured and committed). **Basis question open:** two box_klamp K0 rows score higher on other
      pools (L268 0.8557 `inloopv2`, L267 0.8424 `fullscr41`) on the v2.8.2 CTRL checkpoint, which
      is now also secured and committed — so either basis is backed.
    - **Presence check (§6.3), every system:** quadrotor_3d PASS; quadrotor_planar PARTIAL — the
      checkpoint resolves to `v2.7.0/seed42_iter5` by exact digest, but no
      `data/secured_data/v2.7.1/` records the cell that produced its headline; double_integrator
      resolves by version+seed inference only.
    - **What the `v2.8.4` tag omits**, recorded in results §0.1 and repaired forward on the same
      branch rather than by moving the tag: `src/frameworks/oc_pncbf/value_target.py` (one of the
      ceiling axis's four source files, present on disk but unstaged), this dashboard, and
      `tests/test_dual_scoring_wiring.py` (which repairs the version's one known-red test; the
      harness reads 244 passed / 2 skipped / 0 failed after it). The tag also carries `v2.8.5` in
      `src/_version.py` and four default-off v2.8.5 source files, whose flag-off bit-identity
      against the stored v2.8.4 artifact is proved on 37 fields (results §1.6).
    - Baselines: `data/baselines/{ppo, backup_cbf}/` and `data/secured_data/baselines/` populated.
      backup_cbf has **no checkpoint of its own** — two filter cells over the CTRL checkpoint; every
      citation carries *"hand-designed policy + online rollout certificate (implicit/backup CBF);
      NOT an analytic closed-form CBF."* **PPO ledger placement (PF-3) still awaits the Researcher.**
    - Open, carried into v2.8.5: **`cps_v2` applicability on `quadrotor_3d`** — an axis, not
      bookkeeping, since resolving it re-scores every registered row; whether `scripts/verify.sh`
      was green after v2.8.5's `src/` edits and **before** its arms launched; the two literal
      `max=1.0` clamps inside `rpcbf_target` against `pncbf_target`'s resolved ceiling; sigma C2
      switching share (no such column exists in the evaluator); 1915/45/40; PF-2, PF-4, PF-5; the
      training-time empty rate, unmeasurable while `collection.py:204` returns a hard-coded 0.0;
      `eval_batch_size` still not a column of `eval_metrics.csv`; the two secured v2.7.0
      `report.md` sentences citing that hard-coded zero; `docs/protocol.zip` tracked alongside the
      authoritative `docs/protocol/`.


## Map

- **Protocol** — rules, prohibitions, eval/env standards, workflow. Start here.
- **Versions** — one report per version (`vMAJOR.MINOR.PATCH`); kept locally.
- **Ledger** — one row per run (config, seeds, checkpoint hash, metrics, verdict).