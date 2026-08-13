# JT-PNCBF

Self-built research infrastructure for **Observation-Conditioned PNCBF** and
**Joint Training** for safe navigation in randomized obstacle environments.

This site is the single source of truth (SSOT). It is read by the Researcher,
the Strategist, and the Executor.

## Current state (dashboard)

!!! note "STATUS — keep this section short and always current"
    - **v2.8.5 CLOSED** (record `docs/versions/v2.8.5_results.md`) and **v2.9.0 CLOSED** (record
      `docs/versions/v2.9.0_results.md`). Active version: **v2.9.1**, open, three conditions
      training. `src/_version.py`, `pyproject.toml` and `src/configs/exp_config.yaml` all read
      **v2.9.1** and agree. **Every run in every version below is a single run at seed 42; no result
      here is a multi-seed aggregate.**
    - **v2.9.0 — both registered hypotheses scored; the performance one failed.**
      **A1 (`cps` on the registered cell) is FALSIFIED**: the compressed condition is **below its
      control at 13 of 14 matched rounds ≥ 300**, mean paired difference **−0.034378**. Per-round
      CIs overlap at **13 of the 14**; they are **disjoint at round 300** alone
      ([0.717244, 0.780145] against [0.808011, 0.861132]). **No separation is claimed in either
      direction** — one round's separation carries nothing, and the verdict rests on the sign count,
      which is what A1 registered. **A2 (eval share of end-to-end wall) is
      NOT falsified**: **7.26 %** whole-run and **8.32 %** worst residency segment against the
      59.11 % `COMPRESS500` recorded, clearing it by 7.11× at the worst segment.
    - **v2.9.0 conditions, and where each stopped** (all seed 42, under `data/runs/v2.9.0/`):
      COMPRESS1000 **done @10000 = round 1000** · BUFCAP10K **done @10000** · BUFCAP5K
      **done @10000** · BUFCAP2K **STOPPED @5721 = round 572.1** by Researcher instruction under a
      free-VRAM trigger, **no `final.pt` and no `report.md`**; its series is truncated and its
      maximum is not comparable to one taken over 1000 rounds. `halt_reason` is `null` on all four.
      The `buffer_cap` series (1e6 → 10k / 5k / 2k) is **reported with no verdict** — it was not a
      registered axis of the version.
    - **v2.9.0 changed no bold row and secured nothing.** `data/secured_data/v2.9.0/` does not
      exist, which is consistent with no v2.9.0 row being bold. Four v2.9.0 rows are registered at
      **L297–L300**, one per condition at its in-loop best. `scripts/check_ledger.py` exit 0,
      **281 rows, 41 blocks, 3 bolds, 0 violations**.
    - **v2.9.1 open — three conditions training**, all seed 42, all under `data/runs/v2.9.1/`, all
      `phase: training` with `halt_reason` null. Two carry the new training scene law against their
      own v2.9.0 controls (`buffer_cap` 1e6 and 10 000); the third is the second of those with
      `schedules.sigma.sigma_min` 0.3 → 0.0, a single-key pair. Gates 1–5 passed before launch —
      flag-off parity byte-identical for both samplers, the eval pool rebuilt to its own manifest
      digest, each config diff exactly its registered field set, and the schedule gate at exactly
      0.0. No PID is named here; run directories are the stable identifier.
    - **v2.8.5's seven conditions and its `ell` axis are closed** — the axis returned no separation
      except an upper-side penalty at `ell = 0.35`, and the compressed-update condition closed
      **NOT RESOLVED** after being signalled at round 477.9. Detail is in
      `docs/versions/v2.8.5_results.md`; v2.9.0 re-ran that condition to full budget against a
      matched control on a repaired evaluator and it **still lost at 13 of 14 matched rounds**, so
      the three causes v2.8.5 named for the NOT-RESOLVED verdict were removed and the axis failed
      anyway. **Sixteen v2.8.5 rows stay registered at L281–L296.**
    - **Pool of record: `fullcb`** — `eval_fullcb_quadrotor-3d-d2r_n2000_seed823456`, n 2000,
      unchanged; its digest lives in the pool manifest, not here. v2.9.0 built two replacement pools
      under corrected altitude and omega laws and measured that their `cps` differs from `fullcb`'s
      by less than the scene-sampling noise of an independent 2000-scene draw, so **no pool-level
      effect was established and the pool of record did not move**; both carry
      `pool_role = "NOT REGISTERED"` in their own manifests.
    - **Bold state:** double_integrator v2.3.0 (flagged — `cps_v2` reads `-`, and the row names no
      checkpoint); quadrotor_planar v2.7.1 0.9036; **quadrotor_3d v2.8.5 `ell` 0.125 @45000 =
      0.8701** [0.8472, 0.8919] (ledger **L294**, adopted checkpoint `step_045000.pt`, secured at
      `data/secured_data/v2.8.5/seed42/`, digest pinned in that snapshot's `ADOPTED.md`). **The
      adopted checkpoint is a FIXED-STEP checkpoint, not that run's `best.pt`**, which is at step
      50000 and scores 0.8614 on the same cell. This is a **basis** classification under
      `06_workflow` §2.5 — single seed 42, and its CI overlaps `ell` 0.25, `ell` 0.10 and CLIP50 at
      every round — and is **not** a `04_eval` §5 CI-separated beat. Both the v2.8.4 ARM row (L271)
      and the v2.8.5 `ell` 0.175 row (L288) are un-bolded and carry standing supersession lines.
    - **Presence check (§6.3), every system:** 3 of 3 bold rows PRESENT **by digest**, 5 of 5
      checkpoint entries (the double_integrator bold is a 3-seed aggregate), each digest matched
      against the snapshot's own `ADOPTED.md`. `unicycle` carries 5 rows and no bold row, so §6.3 is
      vacuous for it. **"Committed" is unverified here** — staging is the Researcher's action.
    - Baselines: `data/baselines/{ppo, backup_cbf}/` and `data/secured_data/baselines/` populated.
      backup_cbf has **no checkpoint of its own** — two filter cells over the CTRL checkpoint; every
      citation carries *"hand-designed policy + online rollout certificate (implicit/backup CBF);
      NOT an analytic closed-form CBF."* **PPO ledger placement (PF-3) still awaits the Researcher.**
    - **Open items.** **C0** — `cps_v2` applicability on `quadrotor_3d`, an axis and not
      bookkeeping, since resolving it re-scores every registered row; it must not be silently
      dropped. **PF-5** — `train.py:693` gates the zero-value step on the safety-channel type rather
      than on `k_v`; tagged, not applied. **C21** — the two secured v2.7.0 `report.md` sentences
      citing `collection.py:204`'s hard-coded 0.0, disposition undecided, and the training-time
      empty rate unmeasurable while it stands. **C22** — the two literal `max=1.0` clamps inside
      `rpcbf_target`, the box_klamp basis question, and `docs/protocol.zip` alongside the
      authoritative `docs/protocol/`. **C23** — the automatic ledger-registration rule did not
      fire once across the version; all sixteen rows above were registered at the close, not as
      part of completing each eval. **C24** — the
      shell `grep` shim honours `.gitignore` and silently omits `tests/`, `scripts/analysis/` and
      `scripts/deck/`, so any earlier repo-wide sweep should be re-run and every count must name the
      binary that produced it. Also open: `verify.sh`'s state before launch (C6, unestablishable
      from disk); `eval_batch_size` still neither a config key nor a column of `eval_metrics.csv`;
      the 31 unselectored `signed_h` sites, none repaired. **v2.8.5's own protocol follow-ups are
      tagged PF-6 … PF-16** in `docs/versions/v2.8.5/build_log.md` §11; nine of the eleven are open.


## Map

- **Protocol** — rules, prohibitions, eval/env standards, workflow. Start here.
- **Versions** — one report per version (`vMAJOR.MINOR.PATCH`); kept locally.
- **Ledger** — one row per run (config, seeds, checkpoint hash, metrics, verdict).