# JT-PNCBF

Self-built research infrastructure for **Observation-Conditioned PNCBF** and
**Joint Training** for safe navigation in randomized obstacle environments.

This site is the single source of truth (SSOT). It is read by the Researcher,
the Strategist, and the Executor.

## Current state (dashboard)

!!! note "STATUS — keep this section short and always current"
    - **Active version:** v2.0.1 (closed). v2.1.0 not yet opened.
    - **Joint Training DI:** adopted (v2.0.1). JT through HardNet with
      decoupled schedule clock; three-seed validated. Secured under
      `data/secured_data/v2.0.1/`.
    - **Single-seed SOTA:** JT DI cps 0.8853 (seed 42, step 34000, N=500).
    - **Mean SOTA:** JT DI cps 0.8777 ± 0.0074 over seeds 42 / 12345 / 99
      (pooled 95% CI [0.8373, 0.9152]).
    - **In flight:** none.
    - **Open items:** v2.1.0 next-axis decision (lookahead-alpha /
      multi-system / OC multi-seed reference) — see
      `docs/versions/v2.0.1/results.md` §4. Eval init audit (current vs.
      prior codebase) pending. Strict-rule verdict vs OC v2.0.0 is
      "no improvement detected" (CI overlap artifact from OC being
      single-seed; substance is +0.08 cps separation). Residual reactive
      headroom ~0.025 cps per failure dissection.

## Map

- **Protocol** — rules, prohibitions, eval/env standards, workflow. Start here.
- **Versions** — one report per version (`vMAJOR.MINOR.PATCH`); kept locally.
- **Ledger** — one row per run (config, seeds, checkpoint hash, metrics, verdict).