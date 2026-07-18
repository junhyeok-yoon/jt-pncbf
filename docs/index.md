# JT-PNCBF

Self-built research infrastructure for **Observation-Conditioned PNCBF** and
**Joint Training** for safe navigation in randomized obstacle environments.

This site is the single source of truth (SSOT). It is read by the Researcher,
the Strategist, and the Executor.

## Current state (dashboard)

!!! note "STATUS — keep this section short and always current"
    - Active version: v2.7.1 CLOSED — corridor-cell state injection (registered axis) + 3 eval-only iterations (Stage-1/1b/1c: k-step empty-branch filter fallback); single seed 42 (quadrotor); multi-seed DEFERRED to the adopted config. **Headline = iter-5 checkpoint (3b27d691, continuing collector, inject 0) + empty-branch fallback k=5 — new (quadrotor_planar) SOTA-bold, supersedes v2.7.0 (0.8232):** cps_v2 **0.9036** [0.8807, 0.9248], reach 0.9655, collision **0.0215** [0.0155, 0.0280] (full_n2000 seed23456); CI-separated above L115 on cps and collision. The three k=5 candidate rows (v2.7.1+k5 0.9152, iter-5+k5 0.9036, iter-1+k5 0.9009) are mutually CI-overlapping → bold goes to the simplest config carrying no refuted mechanism (no injection interaction). **Registered axis (corridor injection) REFUTED-WITH-ASSETS:** cell is dynamically transient (visitation 0.04%→0.07%), H-perf/H-mech not met; assets kept (‖L_gV̂‖ at collisions 0.4915→1.4481; motivated P-B′). **Principal finding — k=5 empty-branch fallback:** P-B′ improvable 0.535→0.761→0.958 (k=1/5/10); Stage-1c differential quadrotor CI-sep ≫ unicycle point-only > DI null → mechanism is first-order blindness (underactuation). k=5 is the pre-registered cell (k=10 failed, non-monotonic). Adopted quadrotor deployed default = mode=kstep k=5 (DI/unicycle none). NOT compared across systems (DI SOTA remains v2.3.0 0.8698 | 0.9005). In flight: none. Next: Stage-2 (BPTT-integrated fallback) registered, not auto-started; corridor question open but no longer gating.

## Map

- **Protocol** — rules, prohibitions, eval/env standards, workflow. Start here.
- **Versions** — one report per version (`vMAJOR.MINOR.PATCH`); kept locally.
- **Ledger** — one row per run (config, seeds, checkpoint hash, metrics, verdict).