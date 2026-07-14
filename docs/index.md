# JT-PNCBF

Self-built research infrastructure for **Observation-Conditioned PNCBF** and
**Joint Training** for safe navigation in randomized obstacle environments.

This site is the single source of truth (SSOT). It is read by the Researcher,
the Strategist, and the Executor.

## Current state (dashboard)

!!! note "STATUS — keep this section short and always current"
    - Active version: v2.5.1 CLOSED — Certified Policy Iteration -> exact single-backup certificate. Best (guarantee-carrying): E_s99 arm-B cps_v2 0.8422, collision 0.000. Best filter-only: pooled 3-seed exact_m0 arm-A cps_v2 0.8160 [0.801, 0.829], collision 0.0018 (seed42 0.8226 [0.797, 0.847]). NOT SOTA — DI SOTA remains v2.3.0 legacy 0.8698 | cps_v2 0.9005 (3-seed). Hand-designed maneuver library removed on both sides: exact single-backup filter matches the v2.5.0 17-maneuver barrier (0.8327 [0.805, 0.856]) within overlapping CIs, and the learned bail-out family certifies more volume than the library (cert-ratio 1.0155). S' shield: zero verified-start collisions in every evaluation. Registered CPI performance gate P-PI2 REJECTED (0.678/0.721/0.660). In flight: none. Next: v2.5.2 (unicycle port).

## Map

- **Protocol** — rules, prohibitions, eval/env standards, workflow. Start here.
- **Versions** — one report per version (`vMAJOR.MINOR.PATCH`); kept locally.
- **Ledger** — one row per run (config, seeds, checkpoint hash, metrics, verdict).