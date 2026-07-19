# JT-PNCBF

Self-built research infrastructure for **Observation-Conditioned PNCBF** and
**Joint Training** for safe navigation in randomized obstacle environments.

This site is the single source of truth (SSOT). It is read by the Researcher,
the Strategist, and the Executor.

## Current state (dashboard)

!!! note "STATUS — keep this section short and always current"
    - Active version: v2.7.2 CLOSED — quadrotor_3d bring-up (13-state quaternion, infinite vertical cylinders, altitude goals, 6-DOF perturbed tilt starts); single seed 42; multi-seed DEFERRED to the post-difficulty-axis config. **Lineage headline (bold within quadrotor_3d) = M6 JT (best.pt@33000, sha8 4baaf031, mode=none deployed default): cps 0.9329 [0.91764, 0.94889], collision 0.0095 [0.0055, 0.0135], reach 0.9720** (full_n2000 seed23456). **H1 CONFIRMED** — carried by cps, CI-separated above pre-JT M4 0.8481 [0.82416, 0.87233] (gap +0.04531); collision halves at the point estimate (0.0200→0.0095) with CIs TOUCHING at 0.0135 (no collision-CI-sep claimed). **H2 CONFIRMED** — P2 tilt strata FLAT incl. past-horizontal starts + M1 q3 yaw-invariance; P1 restated as a learned-representation datum (penultimate-feature probe R² 0.83, NOT an obs-sufficiency test). **Stage-3D k=5 empty-branch fallback = SCALING MEASUREMENT, not adoption; 3D deployed default REMAINS mode=none:** collision 0.0095→0.0025 (within-seed CI sep by one scene = 1/2000), cps 0.9586 CI-OVERLAP; 15 fixed/1 new flips, all on empty-branch episodes; deploy-time cost 21.2× (pruning backlog). Blind-row signature TRANSFERS (P4 ‖L_gV̂‖ at 19 collisions median 0.411, degen 0.316; P5 empty 1.51%). NOT compared across systems (planar SOTA v2.7.1 0.9036; DI v2.3.0 0.8698 | 0.9005 unchanged). Provenance: M2 difficulty + M3 gate are build-log-narrative (stdout-only); all headline/probe/Stage numbers artifact-backed (RCF §8). Secured snapshot PENDING (Researcher "secure both" decision); runs moved to previous_runs. In flight: none. Next: v2.7.3 geometry-difficulty re-registration (restores statistical power; enables a fallback adoption trial), then v2.7.4 full-SO(3) starts.

## Map

- **Protocol** — rules, prohibitions, eval/env standards, workflow. Start here.
- **Versions** — one report per version (`vMAJOR.MINOR.PATCH`); kept locally.
- **Ledger** — one row per run (config, seeds, checkpoint hash, metrics, verdict).