# JT-PNCBF

Self-built research infrastructure for **Observation-Conditioned PNCBF** and
**Joint Training** for safe navigation in randomized obstacle environments.

This site is the single source of truth (SSOT). It is read by the Researcher,
the Strategist, and the Executor.

## Current state (dashboard)

!!! note "STATUS — keep this section short and always current"
    - Active version: v2.7.7 CLOSED — eval-only deck assets (34 files, 27 milestones, 2 authorized skips). The headline result is a scoring rule, not an asset: tilt-at-crossing re-scoring of `09c33bf4` on pool `0ef3751b`, kstep k=5, **cps 0.8793** (reach 0.9595, collision 0.0200) — between 0.8051 with the z limit and 0.9078 without, sanity PASS. It charges only the 19 band crossings inside the 60° holding cone. The analytic feasibility route is closed by its own audit (the flag is not a lower bound), so 0.9032 / 0.9103 are bracketed and unusable. No training, no new checkpoint, no `src`/config change. Single seed 42; no CI and no device recorded for any v2.7.7 evaluation.
    - Ledger: 3 `eval_only` rows (M21, M22, M27) at `docs/ledger.md:189-191`, **none bolded**. Lineage bold (quadrotor_3d) UNCHANGED = v2.7.2 M6 JT 0.9329; comparison unavailable (pre-d2r geometry). `scripts/verify.sh` 183 passed / 2 skipped / 0 failed.
    - Correcting the previous STATUS text: the v2.7.4 and v2.7.6 secured snapshots DO exist (`data/secured_data/v2.7.6/experiments/jt_band_hazard/`, `v2.7.4/seed42/`, 2026-07-26 01:16), and the v2.7.6 registered prediction 3 is FALSIFIED, not deploy-contingent.
    - Open prerequisites: register the paper's headline scoring once (0.8793, with 0.8051/0.9078 as the bracket); name the pool wherever thrust-share is quoted (0.0158 is band-feasible, 0.0727 canonical); a dimension-general exact projection at m=4 (`02_control` §6.1); a canonical-pool measurement at the deployed fallback `{kstep, phases 1, k 3}`; a loadable baseline on the per-rotor plant; escalation to {42, 99, 12345}; the deck-assembly pipeline is not under version control.
    - In flight: none. Next: v2.8.0 — audit and retrain toward the paper's final policy.

## Map

- **Protocol** — rules, prohibitions, eval/env standards, workflow. Start here.
- **Versions** — one report per version (`vMAJOR.MINOR.PATCH`); kept locally.
- **Ledger** — one row per run (config, seeds, checkpoint hash, metrics, verdict).