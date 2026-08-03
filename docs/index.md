# JT-PNCBF

Self-built research infrastructure for **Observation-Conditioned PNCBF** and
**Joint Training** for safe navigation in randomized obstacle environments.

This site is the single source of truth (SSOT). It is read by the Researcher,
the Strategist, and the Executor.

## Current state (dashboard)

!!! note "STATUS — keep this section short and always current"
    - Active version: v2.8.1 — CLOSED (2026-08-02). Stage 1 only; Stage 2 never dispatched. Deliverable v2.8.1__jt__20260802-112624__seed42, best.pt step 36000, gate cps 0.6496 / cps_tilt60 0.7715 / cps_bandopen 0.7396. T2 FAILED (reach 0.8620 < 0.8875); terminal not relaxed, nothing promoted, nothing secured. 3-D bold remains v2.8.0. Principal finding: prop:obs-chain is rate-conditional — the crossing ||delta u|| ratio falls 32.27x -> 1.54x at 500 Hz but does not improve at the 20 Hz deployment rate (1.84x -> 2.34x), so the soft-rank encoder axis is retired. Next: v2.8.2 — revert to hard_topk (soft_topk implementation and gates preserved), infeasibility as the subject, OC 30k / JT 30k, reduced batch and per-step training volume, infeasibility judged on early-trajectory behaviour.
    - Data layout: **repaired** and consolidated under `data/runs/<version>/`; `data/previous_runs/` **retired (gone)**. `.gitignore` `secured_data` negations restored; the canonical `-d2r` pools placed in `data/secured_data/pools/`. See `data_layout_repair.md`, `data_repair_part2.md`.
    - Bold state (§2.4/§6.3): one bold row per system — double_integrator v2.3.0 0.8698 (flagged: `cps_v2` pending re-scoring), quadrotor_planar v2.7.1 0.9036, and **quadrotor_3d v2.8.0 0.7919 (NEW this close — first eligible per-rotor-plant baseline; dual scoring `cps_tilt60` 0.9206 / `cps_bandopen` 0.8818; secured `cf948104`)**. v2.7.2 0.9329 was released (pre-per-rotor plant, checkpoint does not load). The §6.3 presence check passes: every bold row's checkpoint is in `secured_data` (3-D `cf948104`, planar `3b27d691`, DI v2.3.0 seed42/99/12345).
    - Open prerequisites: register the paper's headline scoring once (0.8793, bracket 0.8051/0.9078); name the pool wherever thrust-share is quoted; a canonical-pool measurement at the deployed fallback `{kstep, phases 1, k 3}`; a loadable per-rotor-plant baseline (S3); escalation to {42, 99, 12345}; the deck-assembly pipeline is not under version control.
    - Open at close: D4 actuator-lag `rem:lag-null` (τ-sweep completed 4/30 cells, prediction unscored); multi-seed {42,99,12345} not escalated (v2.8.0 closes single-seed); `protocol_delta_dual_scoring.md` §X.1–X.6 is a draft awaiting the Researcher's install into `docs/protocol/`; theory.tex delivered (7 labels present, uncommitted).
    - In flight: none. **Next: v2.8.1** (Researcher-decided, two stages) — Stage 1: ω_G 0.30 / w_settle_ang 1.0 + soft top-k encoder (new lineage); Stage 2: smoothness on the Stage-1 checkpoint — empty-branch continuization (`prop:empty-prox`), infeasibility-margin cost (`prop:sinfty-decomp`), policy rate term; arms A(①)/B(①+②)/C(①+②+③).

## Map

- **Protocol** — rules, prohibitions, eval/env standards, workflow. Start here.
- **Versions** — one report per version (`vMAJOR.MINOR.PATCH`); kept locally.
- **Ledger** — one row per run (config, seeds, checkpoint hash, metrics, verdict).