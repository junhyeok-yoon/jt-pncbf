# JT-PNCBF

Self-built research infrastructure for **Observation-Conditioned PNCBF** and
**Joint Training** for safe navigation in randomized obstacle environments.

This site is the single source of truth (SSOT). It is read by the Researcher,
the Strategist, and the Executor.

## Current state (dashboard)

!!! note "STATUS — keep this section short and always current"
    - Active version: v2.6.1 CLOSED — planar-quadrotor (underactuated) plant-correction + velocity terminal, direction-setting (single seed 42). M6 corrected-plant baseline: cps_v2 **0.5753** [0.533, 0.616], reach 0.8205, collision 0.0810 [0.069, 0.094], timeout 0.0985 (full_n2000 seed23456). Both hypotheses CONFIRMED: H1 torque box 0.2→1.0 (plant-coherent) cut collision below 0.143 (−55% vs the M3 pre-JT 0.1810; vs v2.6.0 M6 0.143 −43%); H2 velocity terminal (w_terminal_v=30) cut timeout to 0.0985. cps +0.382 vs within-version M3 0.1931 (CI-separated). Gates P1/P2/P3 all PASS/CONFIRMED (near-B₀ degen 0.000; 159 residual coll/7.95%, 0/159 collinear). NEW SYSTEM — corrected plant (tau_max 1.0); flag Researcher for classification vs the v2.6.0 tau_max=0.2 row (not auto-superseded); NOT compared across systems (DI SOTA remains v2.3.0 0.8698 | 0.9005). Diagnostic suite (this version's core output): fundamental (D2 25.8% collisions avoidable, D3 value goal-agnostic), role-function (π navigates alone reach 0.61 but unsafe coll 0.39; filter rescues 64.4% of standalone failures; V̂ sound, 0% collision mis-cert), predictive-filter lookahead NULL, torque-box plausibility, task-cost surface, architecture surface. Seed escalation aborted (seed-99 M5 killed @32k, seed-12345 not launched). In flight: none. Next: v2.6.2 — situation-dependent (settling-aware) velocity objective (dense navigation credit: move goal-speed penalty into the running cost, keep goal-distance gradient from vanishing near goal).

## Map

- **Protocol** — rules, prohibitions, eval/env standards, workflow. Start here.
- **Versions** — one report per version (`vMAJOR.MINOR.PATCH`); kept locally.
- **Ledger** — one row per run (config, seeds, checkpoint hash, metrics, verdict).