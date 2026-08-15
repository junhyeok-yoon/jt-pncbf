# `uni_jt_2.9.1` — v2.9.1 secured experiment entry
**This is an experiments entry, not a SOTA snapshot.** It is never eligible for ledger SOTA bolding (`06_workflow` §2.5, §6.3). The v2.9.1 SOTA snapshot remains `data/secured_data/v2.9.1/seed42/`, which this promotion did not touch.
## Base version and the exact change from it
- **Base version:** v2.9.1.
- **System:** `unicycle`.
- **Framework:** `jt_pncbf`.
- **Condition:** `JTUNI_V291_R2`.
- **Source run:** `data/runs/v2.9.1/set__20260813-182144__seed42/v2.9.1__jt__20260813-182144__seed42` (copy only; the original is untouched).
- **Change from base:** this is the version's JT-PNCBF arm for this system, warm-started from **that system's own OC `best.pt`** (`training.jt.value_init_ckpt`; value only — `v_s_state` and `v_s_target_state`, no `pi_state`, so the policy trains from scratch). Trained with the empty-branch fallback OFF: `filter.empty_fallback` resolves to `{mode none, k 10}`, verified before launch by read-back from the CONSTRUCTED collection filter rather than from the config.

## Verdict
Registered as **ledger row L310**, `cps` **0.873126**, scored on the registered cell through `v282_agree_gate.gate_overrides` — the same producer and cell as L304 and every other v2.9.1 row — on pool `eval_full_unicycle_n2000_seed123456`, n 2000, ebs 2000, terminal (0.15, 0.3, 0.3). Single seed 42.

This entry is one cell of the version's **Table I coupling ladder**. Its pair on the identical pool and cell is **L308**: OC 0.670486 → JT 0.873126, **+0.202640**. The unfiltered LQR nominal on the same cell and pool scores `cps` 0.002000.

**No CI-separated claim is made.** Single seed on both sides of the pair, and `cps` is not commensurable across systems (`06_workflow` §2.5), so the ladder is read down each column and never across.

## Interpreting section
`docs/versions/v2.9.1_results.md` — Table I and the per-column pool/cell verification.
Detail: `docs/versions/v2.9.1/oc_arm_rows.md` (OC rows), `docs/versions/v2.9.1/jt_planar.md` (JT arms), `docs/versions/v2.9.1/final_scoring.md` §15 (the 3D JT run).

## File inventory
`04_eval` §7.5 file set as far as this run produced it: **12 of 12**, **19,658,648 bytes**.

All twelve §7.5 files present.

## Identity record — digests
This README is this entry's identity record and the only file here carrying digests (`06_workflow` §6.1); no hash appears in any prose file.

| file | sha256 |
|---|---|
| `config.yaml` | `0bc69610861fd9988daa87e2837f8e2668862790e163882a55d6d719f78279ab` |
| `git_commit.txt` | `47e7328714818c5a47316e712912a37d57a1c63cba6fb4905ebdf140a2bcc6c8` |
| `eval_metrics.csv` | `95ea5b0b96ab45c89e25b83faf5773da3de0a2a8d96513d375cee0eca679234a` |
| `eval_episodes.csv` | `80abc65f2952424c6d94f975bd23eab88ea00752d95179cbf06f316d9e1154f3` |
| `pool_manifest.json` | `1981741b654b5c10da1f21e91dbc6816a5960dc2a744a4a2a6cc7bc4b9c14999` |
| `status.json` | `5096f419878682dd881a09d9ba700860e00d762bff7e22e8a3fc6319346bbdd8` |
| `checkpoints/best.pt` | `f99ddd62e397703f28427bb9a963b618fb288fd09467751253798c392d8fd1e6` |
| `checkpoints/final.pt` | `786801326ca9bf589f18fe9c0932c7c3d48615169add631419a45554aac3a1d9` |
| `figures/trajectory_grid_A.png` | `103385e5397ee66b1252ad5b4d3703520f9acd05aa20c23ffbbadd3f1497308a` |
| `figures/trajectory_grid_B.png` | `ae6d03037eaea40845c16c685ef396817c112f920d4c8a0a8b5d945460e46501` |
| `figures/cbf_contour.png` | `285410e9db1b37f90a3356971ae90e9a485474eebf6570d31344182fc48917ad` |
| `report.md` | `500bad67fe5475b745a93f18082e88b0bd58360722a098d12673e05a399ec395` |
