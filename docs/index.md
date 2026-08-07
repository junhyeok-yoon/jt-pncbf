# JT-PNCBF

Self-built research infrastructure for **Observation-Conditioned PNCBF** and
**Joint Training** for safe navigation in randomized obstacle environments.

This site is the single source of truth (SSOT). It is read by the Researcher,
the Strategist, and the Executor.

## Current state (dashboard)

!!! note "STATUS — keep this section short and always current"
    - Active version: **v2.8.3** — OPEN (2026-08-06). v2.8.2 CLOSED (2026-08-07): diagnostic version, no new SOTA and no bold change; every registered training axis was falsified, unscored or a specification error. See `docs/versions/v2.8.2_results.md`.
    - Data layout: **repaired** and consolidated under `data/runs/<version>/`; `data/previous_runs/` **retired (gone)**. `.gitignore` `secured_data` negations restored; the canonical `-d2r` pools placed in `data/secured_data/pools/`. See `data_layout_repair.md`, `data_repair_part2.md`.
    - Bold state (§2.4/§6.3): one bold row per system — double_integrator v2.3.0 0.8698 (flagged: `cps_v2` pending re-scoring), quadrotor_planar v2.7.1 0.9036, and **quadrotor_3d v2.8.0 0.7919 (NEW this close — first eligible per-rotor-plant baseline; dual scoring `cps_tilt60` 0.9206 / `cps_bandopen` 0.8818; secured `cf948104`)**. v2.7.2 0.9329 was released (pre-per-rotor plant, checkpoint does not load). The §6.3 presence check passes: every bold row's checkpoint is in `secured_data` (3-D `cf948104`, planar `3b27d691`, DI v2.3.0 seed42/99/12345).
    - Open prerequisites: register the paper's headline scoring once (0.8793, bracket 0.8051/0.9078); name the pool wherever thrust-share is quoted; a canonical-pool measurement at the deployed fallback `{kstep, phases 1, k 3}`; a loadable per-rotor-plant baseline (S3); escalation to {42, 99, 12345}; the deck-assembly pipeline is not under version control.
    - Open at close: D4 actuator-lag `rem:lag-null` (τ-sweep completed 4/30 cells, prediction unscored); multi-seed {42,99,12345} not escalated (v2.8.0 closes single-seed); `protocol_delta_dual_scoring.md` §X.1–X.6 is a draft awaiting the Researcher's install into `docs/protocol/`; theory.tex delivered (7 labels present, uncommitted).
    - In flight: **v2.8.3 sigma-hazard repair-A arm** — `v2.8.3__jt__20260807-005658__seed42` (run dir `data/runs/v2.8.3/set__20260807-005658__seed42/`), step **6497** at this edit, PID 4090911, `halt_reason` null. Arm 1 (deployed two-valued alpha) COLLAPSED and is retained as a record only (`set__20260806-235320__seed42`, stopped step 4968, reach 0.0). Also open: the v2.8.2 dt axis arms dt=0.02 (died at step 10834 by shell teardown) and dt=0.01 (never launched, sequential behind 0.02).

## Map

- **Protocol** — rules, prohibitions, eval/env standards, workflow. Start here.
- **Versions** — one report per version (`vMAJOR.MINOR.PATCH`); kept locally.
- **Ledger** — one row per run (config, seeds, checkpoint hash, metrics, verdict).