# JT-PNCBF

Self-built research infrastructure for **Observation-Conditioned PNCBF** and
**Joint Training** for safe navigation in randomized obstacle environments.

This site is the single source of truth (SSOT). It is read by the Researcher,
the Strategist, and the Executor.

## Current state (dashboard)

!!! note "STATUS — keep this section short and always current"
    - **v2.8.3 CLOSED (2026-08-08) — diagnostic version, NO SOTA candidate, NO bold change.** Six axes
      were screened; none was adopted. Results in `docs/versions/v2.8.3_results.md`; close fact-gather
      in `docs/versions/v2.8.3/close_facts.md`. **Active version: v2.8.4 — OPEN**, all three version
      strings read `v2.8.4`; `docs/versions/v2.8.4/changes.md` is still an empty placeholder.
    - **Pool of record: `fullcb`** — `eval_fullcb_quadrotor-3d-d2r_n2000_seed823456`, sha256
      `3682a4e38ab3405d0afd4cfc119a73225eee4ef945cf4a58eed23b2eb6118517`, built from the canonical pool
      under ONE screen (certain impossibility under the coupled, rollout-verified bound), re-verified
      **0/2000**. All thirteen standing rows are re-scored on one cell on it (ledger, 2026-08-08).
      Predecessor pools are unmodified and still resolve.
    - Bold state (§2.4/§6.3) **UNCHANGED**: double_integrator v2.3.0 0.8698 (flagged, `cps_v2` pending
      re-scoring), quadrotor_planar v2.7.1 0.9036, quadrotor_3d v2.8.0 0.7919 (secured `cf948104`).
      **CLASSIFICATION FLAG for the Researcher:** R1 (CTRL + HardNet) scores **0.8291** on `fullcb`,
      above the standing 0.7919, but on a **different pool stem** — a classification question, not a
      promotion. Bold deliberately left unchanged and no v2.8.3 row is bolded.
    - **Presence check (§6.3), every system:** v2.3.0 secured PRESENT (11 `.pt`); v2.8.0 secured PRESENT
      (2 `.pt`); **quadrotor_planar v2.7.1 has NO `data/secured_data/v2.7.1/`** — its row cites a v2.7.0
      artifact (`v2.7.0 iter-5 secured best.pt (3b27d691)`) and `data/secured_data/v2.7.0/` does exist.
      Flagged, not repaired.
    - **In flight: nothing.** No training or eval process is running; the GPU is idle. The sigma arm
      (`…20260807-005658`) and the u_prev arm (`…20260807-115736`) both COMPLETED 30000 steps; four
      further v2.8.3 run directories read `phase: training` but hold no process and are abandoned.
    - Baselines: `data/baselines/{ppo, backup_cbf}/` and `data/secured_data/baselines/{ppo, backup_cbf}/`
      populated. **backup_cbf has NO checkpoint of its own** — it is two filter cells over the CTRL
      checkpoint; every citation carries the STATUS line *"hand-designed policy + online rollout
      certificate (implicit/backup CBF); NOT an analytic closed-form CBF."* **PPO baseline ledger row
      still not placed** (PF-3, awaits Researcher).
    - Open, carried into v2.8.4: sigma C2 switching share (**no such column exists** in the evaluator);
      1915/45/40 unreconciled; PF-2, PF-3, PF-4, PF-5 open; the box_klamp float32 residual (diagnosed,
      unfixed by instruction); read (c)'s horizon inadmissibility (measured inert on the active window,
      unrepaired in the code); `eval_batch_size` is still not a column of `eval_metrics.csv`.


## Map

- **Protocol** — rules, prohibitions, eval/env standards, workflow. Start here.
- **Versions** — one report per version (`vMAJOR.MINOR.PATCH`); kept locally.
- **Ledger** — one row per run (config, seeds, checkpoint hash, metrics, verdict).