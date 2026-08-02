# JT-PNCBF

Self-built research infrastructure for **Observation-Conditioned PNCBF** and
**Joint Training** for safe navigation in randomized obstacle environments.

This site is the single source of truth (SSOT). It is read by the Researcher,
the Strategist, and the Executor.

## Current state (dashboard)

!!! note "STATUS — keep this section short and always current"
    - Active version: **v2.8.0 — CLOSED (2026-08-02)**. Verdict: **no improvement detected (single-seed, seed 42)**; the version's yield is correctness/regularity — exact single-scalar `dual_solve` projection (`prop:lambda-solve`, G1/G2/G3 pass; `verify.sh` 212 passed / 2 skipped), rate-bounded command TV, angular-settling terminal, and the installed dual-scoring standard (`cps_tilt60`/`cps_bandopen`) — not score. Deliverable secured (copy-only): dual arm `best.pt`@42000 (`cf948104`) → `data/secured_data/v2.8.0/seed42/` (`git add -f` staging pending Researcher). Full record: `docs/versions/v2.8.0_results.md`, `close_retrieve.md`.
    - Data layout: **repaired** and consolidated under `data/runs/<version>/`; `data/previous_runs/` **retired (gone)**. `.gitignore` `secured_data` negations restored; the canonical `-d2r` pools placed in `data/secured_data/pools/`. See `data_layout_repair.md`, `data_repair_part2.md`.
    - Bold state (§2.4/§6.3): one bold row per system — double_integrator v2.3.0 0.8698 (flagged: `cps_v2` pending re-scoring), quadrotor_planar v2.7.1 0.9036, and **quadrotor_3d v2.8.0 0.7919 (NEW this close — first eligible per-rotor-plant baseline; dual scoring `cps_tilt60` 0.9206 / `cps_bandopen` 0.8818; secured `cf948104`)**. v2.7.2 0.9329 was released (pre-per-rotor plant, checkpoint does not load). The §6.3 presence check passes: every bold row's checkpoint is in `secured_data` (3-D `cf948104`, planar `3b27d691`, DI v2.3.0 seed42/99/12345).
    - Open prerequisites: register the paper's headline scoring once (0.8793, bracket 0.8051/0.9078); name the pool wherever thrust-share is quoted; a canonical-pool measurement at the deployed fallback `{kstep, phases 1, k 3}`; a loadable per-rotor-plant baseline (S3); escalation to {42, 99, 12345}; the deck-assembly pipeline is not under version control.
    - Open at close: D4 actuator-lag `rem:lag-null` (τ-sweep completed 4/30 cells, prediction unscored); multi-seed {42,99,12345} not escalated (v2.8.0 closes single-seed); `protocol_delta_dual_scoring.md` §X.1–X.6 is a draft awaiting the Researcher's install into `docs/protocol/`; theory.tex delivered (7 labels present, uncommitted).
    - In flight: none. **Next: v2.8.1** (Researcher-decided, two stages) — Stage 1: ω_G 0.30 / w_settle_ang 1.0 + soft top-k encoder (new lineage); Stage 2: smoothness on the Stage-1 checkpoint — empty-branch continuization (`prop:empty-prox`), infeasibility-margin cost (`prop:sinfty-decomp`), policy rate term; arms A(①)/B(①+②)/C(①+②+③).

## Map

- **Protocol** — rules, prohibitions, eval/env standards, workflow. Start here.
- **Versions** — one report per version (`vMAJOR.MINOR.PATCH`); kept locally.
- **Ledger** — one row per run (config, seeds, checkpoint hash, metrics, verdict).