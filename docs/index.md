# JT-PNCBF

Self-built research infrastructure for **Observation-Conditioned PNCBF** and
**Joint Training** for safe navigation in randomized obstacle environments.

This site is the single source of truth (SSOT). It is read by the Researcher,
the Strategist, and the Executor.

## Current state (dashboard)

!!! note "STATUS — keep this section short and always current"
    - Active version: v2.6.2 CLOSED — planar-quadrotor situation-dependent velocity objective (dense goal-gated settling + braking-envelope approach term; running-cost redesign, no value/filter change; single seed 42). M6 full-range baseline: cps_v2 **0.7311** [0.6925, 0.7648], reach 0.905, collision 0.0670 [0.0565, 0.078], timeout 0.025 (full_n2000 seed23456); +0.156 CI-separated vs v2.6.1 M6 0.5753 (same plant). H-appr CONFIRMED with margin (collision CI below 0.0810); P3 CONFIRMED (123 residual, 0/123 collinear). **IC-exclusion line CLOSED by the doom census** — provable-doom set empty under the 3g relaxed system (ballistic 0/2000 + 4D-HJ 0/2000, both gates pass); attitude-cost doom uncertifiable (6D dense HJ measured infeasible); the full-range IC pool is the official quadrotor distribution. **Gravity-observability discovered**: the gravity-blind body-frame obs aliases upright↔inverted (θ undecodable in the high-tilt/early failure cell 98.6°; 1-step aliased-dynamics closure gap ≈0.4 m/s; the feasibility certificate falsely dooms 61.5% of savable high-tilt ICs); the kinematic thrust-misfire mechanism DISCONFIRMED. Deflation sweep NULL (not a lever). NOT compared across systems (DI SOTA remains v2.3.0 0.8698 | 0.9005). In flight: none. Next: v2.7.0 — observation augmentation (sin/cos θ), full retrain.

## Map

- **Protocol** — rules, prohibitions, eval/env standards, workflow. Start here.
- **Versions** — one report per version (`vMAJOR.MINOR.PATCH`); kept locally.
- **Ledger** — one row per run (config, seeds, checkpoint hash, metrics, verdict).