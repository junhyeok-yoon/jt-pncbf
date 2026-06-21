# JT-PNCBF

Self-built research infrastructure for **Observation-Conditioned PNCBF** and
**Joint Training** for safe navigation in randomized obstacle environments.

This site is the single source of truth (SSOT). It is read by the Researcher,
the Strategist, and the Executor.

## Current state (dashboard)

!!! note "STATUS — keep this section short and always current"
    - **Active version:** v2.3.0 (closed). v2.3.1 next (fast-approach
      coverage: admission unit-fix + v_init_max).
    - **DI SOTA:** JT DI **cps 0.8698** (v2.3.0, D_pi=2k/D_V=1M @50k, 3-seed
      mean), secured under `data/secured_data/v2.3.0/`. Point-estimate
      promotion: cross-seed 95% CI [0.8527, 0.8869] overlaps the prior 0.8568
      (v2.0.1) — not a CI-confirmed beat. Prior SOTA superseded (kept under
      `data/secured_data/v2.0.1/`). The DI cps cap is projection infeasibility
      on recoverable states, not collisions.
    - **Unicycle (new system, v2.2.2):** baseline of record cps 0.4981; **best
      cps 0.6243** (collision-precursor injection, +0.126 over baseline,
      non-overlapping CIs, navigation-driven). Secured under
      `data/secured_data/v2.2.2/seed42/` (single-seed; trained at
      goal_speed_radius 0.30, also 0.6720 at 0.50). No converged multi-seed yet.
    - **Pools:** N=2000 is the standard pool. DI + unicycle pools in
      `data/secured_data/pools/` (the DI n2000 pool is not yet promoted there).
    - **In flight:** none.
    - **Open items (v2.3.1):** fast-approach-coverage axis — the start-state
      admission filter has a verified unit-bug (projected distance in the speed
      slot, `scene_init.py:255`), inert at `v_init_max=0.5` but activating under
      faster starts. Plan: fix admission physics (B) + raise `v_init_max` (A) to
      generate short-clearance fast-approach states. See
      `docs/versions/v2.3.0_results.md`.

## Map

- **Protocol** — rules, prohibitions, eval/env standards, workflow. Start here.
- **Versions** — one report per version (`vMAJOR.MINOR.PATCH`); kept locally.
- **Ledger** — one row per run (config, seeds, checkpoint hash, metrics, verdict).