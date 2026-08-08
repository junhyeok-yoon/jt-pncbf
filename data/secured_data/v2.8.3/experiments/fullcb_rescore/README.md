# v2.8.3 experiment — `fullcb` uniform re-score

**Base version.** v2.8.2 CTRL (`v2.8.2__jt__20260803-063606__seed42`, `best.pt`,
sha256[:16] `c89f9aef0cdb5499`) and the artifacts standing at the v2.8.3 open.

**The exact change from that base.** Two things, and nothing else:

1. **A new evaluation pool.** `eval_fullcb_quadrotor-3d-d2r_n2000_seed823456`, sha256
   `3682a4e38ab3405d0afd4cfc119a73225eee4ef945cf4a58eed23b2eb6118517`, built from the **canonical**
   pool (`0ef3751b`) by removing exactly the 33 scenes the coupled, rollout-verified bound flags as
   certainly unavoidable (32 floor + 1 ceiling) and topping up 33 from the same sampler, each admitted
   only after passing that bound individually. The superseded v2.8.2 floor screen is NOT applied. The
   finished pool re-runs 0/2000 flagged.
2. **One scoring cell for every standing row**, so that for the first time all rows belong in one
   table: n 2000, `eval_batch_size` 2000, terminal (0.15, 0.3, 0.3), projection `dual_solve`,
   `empty_fallback {kstep, phases 1, k 3}`, alpha (2.0, 100.0), `gamma_margin` 0.0, QP slack penalty
   1e6 on QP rows, bootstrap seed 20260808. Every field is read back off the constructed objects.

**Verdict.** Diagnostic. **No SOTA candidate and no bold change.** R1 (CTRL + HardNet) scores
cps 0.8291 on this pool, above the standing bold row's 0.7919, but on a **different pool stem** — a
classification question for the Researcher, not a promotion. There is deliberately **no seed
snapshot** under `data/secured_data/v2.8.3/`: this version produced no SOTA candidate.

**Contents.** `fullcb_rows.json` (the cell block + all thirteen rows), thirteen `row__*.json`,
thirteen `per_episode__*.npz` (outcome, cause and five step-fraction metrics per episode, so every CI
is recomputable), `fullcb_build.json` (ancestor, the 33 removed indices with class and margin, the 33
per-top-up records, the final 0/2000 bound, the predecessor-unchanged proof),
`D1_reproduces_R1.json` (the box-Bellman anchor reproducing R1 to the digit, `ALL_EQUAL: true`),
`ci_reproduction_two_processes.json` (fixed-seed CI identical across two PIDs while
`abs(hash(label))` differs).

**Interpreted in.** `docs/versions/v2.8.3_results.md` — the uniform re-score section — and built in
`docs/versions/v2.8.3/pool_v2_rescore.md`.
