# `planar_jt_2.9.1` — v2.9.1 secured experiment entry
**This is an experiments entry, not a SOTA snapshot.** It is never eligible for ledger SOTA bolding (`06_workflow` §2.5, §6.3). The v2.9.1 SOTA snapshot remains `data/secured_data/v2.9.1/seed42/`, which this promotion did not touch.
## Base version and the exact change from it
- **Base version:** v2.9.1.
- **System:** `quadrotor_planar`.
- **Framework:** `jt_pncbf`.
- **Condition:** `JTPLANAR_V291_R4`.
- **Source run:** `data/runs/v2.9.1/set__20260813-215041__seed42/v2.9.1__jt__20260813-215041__seed42` (copy only; the original is untouched).
- **Change from base:** this is the version's JT-PNCBF arm for this system, warm-started from **that system's own OC `best.pt`** (`training.jt.value_init_ckpt`; value only — `v_s_state` and `v_s_target_state`, no `pi_state`, so the policy trains from scratch). Trained with the empty-branch fallback OFF: `filter.empty_fallback` resolves to `{mode none, k 10}`, verified before launch by read-back from the CONSTRUCTED collection filter rather than from the config.

## Verdict
Registered as **ledger row L311**, `cps` **0.775740**, scored on the registered cell through `v282_agree_gate.gate_overrides` — the same producer and cell as L304 and every other v2.9.1 row — on pool `eval_full_quadrotor-planar_n2000_seed23456`, n 2000, ebs 2000, terminal (0.15, 0.3, 0.3). Single seed 42.

This entry is one cell of the version's **Table I coupling ladder**. Its pair on the identical pool and cell is **L306**: OC 0.659150 → JT 0.775740, **+0.116590**. The unfiltered LQR nominal on the same cell and pool scores `cps` -0.206000.

**No CI-separated claim is made.** Single seed on both sides of the pair, and `cps` is not commensurable across systems (`06_workflow` §2.5), so the ladder is read down each column and never across.

## Interpreting section
`docs/versions/v2.9.1_results.md` — Table I and the per-column pool/cell verification.
Detail: `docs/versions/v2.9.1/oc_arm_rows.md` (OC rows), `docs/versions/v2.9.1/jt_planar.md` (JT arms), `docs/versions/v2.9.1/final_scoring.md` §15 (the 3D JT run).

## File inventory
`04_eval` §7.5 file set as far as this run produced it: **12 of 12**, **20,093,998 bytes**.

All twelve §7.5 files present.

## Identity record — digests
This README is this entry's identity record and the only file here carrying digests (`06_workflow` §6.1); no hash appears in any prose file.

| file | sha256 |
|---|---|
| `config.yaml` | `e722c149da3a516693252998294a00fb6b256aa47e983b52a7914dfafbc249b5` |
| `git_commit.txt` | `47e7328714818c5a47316e712912a37d57a1c63cba6fb4905ebdf140a2bcc6c8` |
| `eval_metrics.csv` | `b9edfaa1fa948a7ca4f2d70842a0ccb3b027d8288a4cba019361eebb7dd80b71` |
| `eval_episodes.csv` | `6aee4e20616a6deebf4250add33d4a265d42926a3eb32ed3168a24e1c7d03b01` |
| `pool_manifest.json` | `c182e643e80d19dd78d77a659dd57f2a57eded70ccdbb50d6d28cc883531983a` |
| `status.json` | `e41fd88964f60fe6ac556ca3fbd035a246f7880fc2585cd66e8f81a1fc8ac161` |
| `checkpoints/best.pt` | `27811e4071a477259336b27750e6f45e18eacb4fa2abd28bf707a32435297ce7` |
| `checkpoints/final.pt` | `50408556b4b9d5c0ec8c2d4e0b5d37a51c2c76837d06c263e16299797822802a` |
| `figures/trajectory_grid_A.png` | `402263c93d1fa2a0c9650bf922ca546cd1f302894899bbb91c7d24d19780fc7b` |
| `figures/trajectory_grid_B.png` | `a7878e9623134f7f6e1ef0b4b808ac4cf1b97f77da857bbb12ea452bf93c7265` |
| `figures/cbf_contour.png` | `326426fdaf83eea548d49946bf1f4ab30874e620a629d048587709701568a939` |
| `report.md` | `5baf46e6e7e607e14116a23848615c64050dffb20ebbc5ba8036500c5fc92c7b` |
