# JT-PNCBF

Self-built research infrastructure for **Observation-Conditioned PNCBF** and
**Joint Training** for safe navigation in randomized obstacle environments.

This site is the single source of truth (SSOT). It is read by the Researcher,
the Strategist, and the Executor.

## Current state (dashboard)

!!! note "STATUS — keep this section short and always current"
    - **Active version:** v2.2.2 (closing). v2.3.0 about to open
      (collection/replay data-quality).
    - **DI SOTA (unchanged):** JT DI **cps 0.8568** (v2.0.1 seed-42, n2000
      pool), secured under `data/secured_data/v2.0.1/`. v2.2.2's injection
      base ties it (0.8607, CI-overlap) with a collision advantage and is the
      practical DI best; no DI run beat SOTA, so none is secured. The DI
      residual collision is diagnosed (9/9 over-speed) and the value-side
      gradient family is exhausted (input-rate, u-reg, Sobolev, gated-Sobolev,
      RPCBF robust deploy all failed/tied/regressed).
    - **Unicycle (new system, v2.2.2):** baseline of record cps 0.4981; **best
      cps 0.6243** (collision-precursor injection, +0.126 over baseline,
      non-overlapping CIs, navigation-driven). Secured under
      `data/secured_data/v2.2.2/seed42/` (single-seed; trained at
      goal_speed_radius 0.30, also 0.6720 at 0.50). No converged multi-seed yet.
    - **Pools:** N=2000 is the standard pool. DI + unicycle pools in
      `data/secured_data/pools/` (the DI n2000 pool is not yet promoted there).
    - **In flight:** none.
    - **Open items (v2.3.0):** the collection/replay data-quality defect — an
      untuned 1M-trajectory buffer dilutes fresh on-policy data to a
      sub-percent slice of every batch (replay ratio 0.12× by 100k); lower
      `buffer_cap` to a recency window (injection + adaptive sigma backstop
      collapse independent of buffer size). Also recorded: a true
      training-continue (buffer save/load) and unicycle-tuned injection. See
      `docs/versions/v2.2.2/results.md`.

## Map

- **Protocol** — rules, prohibitions, eval/env standards, workflow. Start here.
- **Versions** — one report per version (`vMAJOR.MINOR.PATCH`); kept locally.
- **Ledger** — one row per run (config, seeds, checkpoint hash, metrics, verdict).