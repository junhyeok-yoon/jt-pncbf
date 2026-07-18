# JT-PNCBF

Self-built research infrastructure for **Observation-Conditioned PNCBF** and
**Joint Training** for safe navigation in randomized obstacle environments.

This site is the single source of truth (SSOT). It is read by the Researcher,
the Strategist, and the Executor.

## Current state (dashboard)

!!! note "STATUS — keep this section short and always current"
    - Active version: v2.7.0 CLOSED — quadrotor observation gravity-direction restoration (obs dim 20→22, append body-frame `(sin θ, cos θ)`; value retrained from scratch, else frozen) + 4 within-version iterations; single seed 42. **Headline = iteration-1 (SOTA single-seed), supersedes v2.6.2 (0.7311):** cps_v2 **0.8232** [0.7910, 0.8523], reach 0.940, collision **0.0455** [0.0370, 0.0550], timeout 0.013 (full_n2000 seed23456); +0.092 CI-separated vs v2.6.2; collision −32% CI-separated. **S1/S2 CONFIRMED** (θ decodable to ~2°, aliasing eliminated), **P2 CONFIRMED**; **S3/P1 not confirmed** (t=0 false-doom + extreme-tilt residual persist). Iterations: **iter-2** IC-band oversampling REFUTED (null, CI-overlap); **iter-3** bptt_T=60 refuted (training collapse); **iter-5** continuing-batch collector (Track A) **ADOPTED as default** — H-nondegradation on par (cps 0.8343 [0.8038, 0.8632], NOT SOTA-separated), H-throughput NOT MET (0.94× eps/hr, +2% pipeline), adoption on label-provenance grounds. `collector=continuing` default going forward (legacy retained for parity). NOT compared across systems (DI SOTA remains v2.3.0 0.8698 | 0.9005). In flight: none. Next: v2.8.0 — corridor-cell state injection (near-surface × high-tilt × inward-speed).

## Map

- **Protocol** — rules, prohibitions, eval/env standards, workflow. Start here.
- **Versions** — one report per version (`vMAJOR.MINOR.PATCH`); kept locally.
- **Ledger** — one row per run (config, seeds, checkpoint hash, metrics, verdict).