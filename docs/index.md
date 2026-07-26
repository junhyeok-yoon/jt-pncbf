# JT-PNCBF

Self-built research infrastructure for **Observation-Conditioned PNCBF** and
**Joint Training** for safe navigation in randomized obstacle environments.

This site is the single source of truth (SSOT). It is read by the Researcher,
the Strategist, and the Executor.

## Current state (dashboard)

!!! note "STATUS — keep this section short and always current"
    - Active version: v2.7.6 CLOSED — vertical domain surfaces (arena band |z|=4 as a collision surface + h_star vertical branch) and the observation fix (dim 32→34, `obs_band_z`) that made them learnable, retrained OC→JT under one h_star; single seed 42, eval-only headline. **Headline checkpoint = JT step 42000 (sha8 `09c33bf4`) on the canonical `eval_full_quadrotor-3d-d2r_n2000_seed23456` (sha `0ef3751b`), banded scoring (this version's predicate): cps 0.69287 [0.6559, 0.7314] at fallback=none, 0.80508 [0.7744, 0.8349] at kstep k=5; legacy scoring 0.80288 / 0.90783** (GPU). All 45 v2.7.6 ledger rows are `eval_only`, **none bolded**. **Lineage bold (within quadrotor_3d) UNCHANGED = v2.7.2 M6 JT 0.9329** — §5 comparison unavailable (the v2.7.2 headline is on the pre-d2r canonical geometry, not bit-identical to this version's d2r canonical; and the matched legacy cell 0.80288 is CI-separated *below* 0.9329 regardless). **Band collisions fall CI-separated vs v2.7.4 on every pool** (band-feasible 185→8, full-range 286→99, 96–99% floor); tilt-anticipation confirmed (yz V̂=0 sits +0.78/+1.03 above the attitude-free hazard line); thrust-channel share 0.0158→0.20357 (band boundary 0.71). **Registered falsifier fired:** prediction 3 (legacy cps no-regress on full-range) FALSIFIED (0.7986 [0.770, 0.826] vs v2.7.4 0.8580 [0.835, 0.881]); deploy-contingent — removed by the kstep fallback (0.9085 vs 0.8990). `scripts/verify.sh` **183 passed / 2 skipped / 0 failed**; both blocking pool SHAs (`0ef3751b`, `db0b9eb5`) bit-identical; M8 vectorization gate reproduced M7 at 0 mismatches. **Open prerequisites (results §4):** a dimension-general exact projection (the empty-branch enumerator is exact only for 2-D action, this system is 4-D, so the filter is approximate on non-empty rows and patched on empty ones, and the policy gets no gradient where a box corner wins); no canonical-pool measurement at the deployed fallback `{kstep, phases 1, k 3}`; the secured chain is broken at v2.7.3 (v2.7.4/2.7.5/2.7.6 unsecured, `04_eval §5`); and the JT-vs-alternating central claim is untested. Runs moved to `data/previous_runs/v2.7.6/`; secured snapshot NOT taken. In flight: none. Next: Researcher decisions on config promotion (band-hazard default) and the v2.7.4→v2.7.6 secured promotions.

## Map

- **Protocol** — rules, prohibitions, eval/env standards, workflow. Start here.
- **Versions** — one report per version (`vMAJOR.MINOR.PATCH`); kept locally.
- **Ledger** — one row per run (config, seeds, checkpoint hash, metrics, verdict).