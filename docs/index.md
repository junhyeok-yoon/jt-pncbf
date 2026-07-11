# JT-PNCBF

Self-built research infrastructure for **Observation-Conditioned PNCBF** and
**Joint Training** for safe navigation in randomized obstacle environments.

This site is the single source of truth (SSOT). It is read by the Researcher,
the Strategist, and the Executor.

## Current state (dashboard)

!!! note "STATUS — keep this section short and always current"
    - **Active version:** v2.5.0 **CLOSED** (analytic MC-PNCBF: Stage A/B/B-2 + Stage R/R-2 + ARM-C + cps_v2 instrumentation). Best = B-2 step_007500: legacy 0.5893 | cps_v2 0.8327 [0.8053, 0.8564] @D-slow (0.5970 | 0.8358 @D-fast). Structural safety confirmed (2 runs × 20 ckpts, collision 0.002–0.0065 band). Deploy-rate gain V_M-specific (ARM-C: learned filter regresses at dt 0.01). NOT SOTA — DI SOTA v2.3.0 legacy 0.8698 | cps_v2 0.9005 (3-seed).
    - **Stage B (policy-only JT through analytic V_M, seed 42, 30k, gamma_m=0.02):** STRUCTURAL SAFETY CONFIRMED (P-B2 — collision flat ~0.005, training-independent across all 20 checkpoints; no learned safety object). Best step 9000 (n2000): canonical cps 0.5893 [0.565,0.611] (fast re-baseline 0.5831 | cps_v2 0.8254) | empty-only 0.8265 (> Stage-A A1' 0.8088 & OC 0.797, < P-B4 0.85 & v2.3.0 0.870); reach 0.9225 (< P-B1 0.94). Policy peaks ~step 9k then drifts down; proj_mag no decline (P-B3 mixed). Not SOTA. Stage C / next-axis is the Researcher's call.
    - **DI SOTA:** v2.3.0 cps 0.8698 (3-seed) | cps_v2 0.9005 (n2000 re-score, 3-seed mean: seed42 0.9010 / seed99 0.8937 / seed12345 0.9069); prior 0.8568 (v2.0.1).
    - **Unicycle SOTA:** 0.6243 (v2.2.2, single-seed).
    - MC-PNCBF Stage A (training-free analytic barrier): SOUND + FEASIBLE (P-A1 witness; collision ~0.01 vs LQR 0.34; discrete-CBF-ok 0.91-0.92) but arm-1 n2000 cps 0.5616 [0.536,0.584] << 0.80 gate (far-field zero-authority + LQR conservatism).
    - **V_M fast path:** compiled (torch.compile) production default (~4.9x, 204 ms/step); barrier bit-parity (V_M<=1e-6, L_g<=1e-5); ~0.006 systematic n2000 cps offset handled by fast-vs-fast re-baseline (comparators: Stage-B best 0.5831/0.8254, Stage-A A1'-LQR 0.5625/0.8076).
    - **Stage B-2 (filter-friction w=0.05, fast V_M path, seed 42, 30k):** CLOSED. Best step_007500 (n2000) canonical 0.5893 | cps_v2 0.8327 | empty 0.8327 @D-slow (0.5970/0.8358 @D-fast); reach 0.9250, coll 0.0055, stuck 0.0430. Marginal over Stage-B (cps_v2 +0.007, stuck -0.006, P-F3 structural safety confirmed) but P-F1 (proj_mag halving) / P-F2 (reach>=0.94) / P-F4 (cps_v2>=0.85) REFUTED — friction did not reduce intervention; stuck gain is open-area (P-L4). cps-v2 metric amendment: cps_v2==empty-only lifts ALL V-CBF systems (v2.3.0 0.8624->0.9010). NOT SOTA. Next lever = policy/task-cost side.
    - **In flight:** none.

## Map

- **Protocol** — rules, prohibitions, eval/env standards, workflow. Start here.
- **Versions** — one report per version (`vMAJOR.MINOR.PATCH`); kept locally.
- **Ledger** — one row per run (config, seeds, checkpoint hash, metrics, verdict).