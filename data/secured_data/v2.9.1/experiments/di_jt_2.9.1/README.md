# `di_jt_2.9.1` — v2.9.1 secured experiment entry
**This is an experiments entry, not a SOTA snapshot.** It is never eligible for ledger SOTA bolding (`06_workflow` §2.5, §6.3). The v2.9.1 SOTA snapshot remains `data/secured_data/v2.9.1/seed42/`, which this promotion did not touch.
## Base version and the exact change from it
- **Base version:** v2.9.1.
- **System:** `double_integrator`.
- **Framework:** `jt_pncbf`.
- **Condition:** `JTDI_V291_R2`.
- **Source run:** `data/runs/v2.9.1/set__20260813-182126__seed42/v2.9.1__jt__20260813-182126__seed42` (copy only; the original is untouched).
- **Change from base:** this is the version's JT-PNCBF arm for this system, warm-started from **that system's own OC `best.pt`** (`training.jt.value_init_ckpt`; value only — `v_s_state` and `v_s_target_state`, no `pi_state`, so the policy trains from scratch). Trained with the empty-branch fallback OFF: `filter.empty_fallback` resolves to `{mode none, k 10}`, verified before launch by read-back from the CONSTRUCTED collection filter rather than from the config.

## Verdict
Registered as **ledger row L309**, `cps` **0.904757**, scored on the registered cell through `v282_agree_gate.gate_overrides` — the same producer and cell as L304 and every other v2.9.1 row — on pool `eval_full_di_n2000_seed123456`, n 2000, ebs 2000, terminal (0.15, 0.3, 0.3). Single seed 42.

This entry is one cell of the version's **Table I coupling ladder**. Its pair on the identical pool and cell is **L307**: OC 0.880459 → JT 0.904757, **+0.024298**. The unfiltered LQR nominal on the same cell and pool scores `cps` -0.009500.

**No CI-separated claim is made.** Single seed on both sides of the pair, and `cps` is not commensurable across systems (`06_workflow` §2.5), so the ladder is read down each column and never across.

## Interpreting section
`docs/versions/v2.9.1_results.md` — Table I and the per-column pool/cell verification.
Detail: `docs/versions/v2.9.1/oc_arm_rows.md` (OC rows), `docs/versions/v2.9.1/jt_planar.md` (JT arms), `docs/versions/v2.9.1/final_scoring.md` §15 (the 3D JT run).

## File inventory
`04_eval` §7.5 file set as far as this run produced it: **12 of 12**, **19,845,783 bytes**.

All twelve §7.5 files present.

## Identity record — digests
This README is this entry's identity record and the only file here carrying digests (`06_workflow` §6.1); no hash appears in any prose file.

| file | sha256 |
|---|---|
| `config.yaml` | `bcdc006673d36708092b50b47065a8a9b1037e9b26a4540b3c9a321f7d4c198e` |
| `git_commit.txt` | `47e7328714818c5a47316e712912a37d57a1c63cba6fb4905ebdf140a2bcc6c8` |
| `eval_metrics.csv` | `ff16d5f4e76277094c09f6e1bf1c90c66f8aa41383eb0de95c5f30ca7424b617` |
| `eval_episodes.csv` | `3758c6093ac8f559c30ac20bc8f785fbbf8c605bbd37ed568bcc1e7dd32f0078` |
| `pool_manifest.json` | `fbb7f2389084269c912fa93f0d38b01c1df38d863f97813d1e70b3f08770d2a5` |
| `status.json` | `f52a6abc4c78c35d61c9191bb88df7a1b95debfec91f6c73ba609dde0d6d0e5d` |
| `checkpoints/best.pt` | `f066db688d6e93610ee9c39c8d38678dfe368eb89f80417d6cf8d50e6fef4b2b` |
| `checkpoints/final.pt` | `8adcbdbaec60425b2eb4813f0e188e23224bbe4de3e78d720ca500c1a487f6cb` |
| `figures/trajectory_grid_A.png` | `fc235eaadab510a777c42ab8c11ecf902a3a935c547db03cf45d09d91fc37bc5` |
| `figures/trajectory_grid_B.png` | `43173fb345fc47569fa179feeb016670e819fba4f8986688f2e6da8de0ee8ce0` |
| `figures/cbf_contour.png` | `308ed22e70b6e0e5c4ed65625e2ca6ade9c8ec6954a91770b92c5c66fe1970ef` |
| `report.md` | `2cb4ad9872001416842f27b7e5303b2e465b0e1b31b093ca796bac9f4dba2aba` |
