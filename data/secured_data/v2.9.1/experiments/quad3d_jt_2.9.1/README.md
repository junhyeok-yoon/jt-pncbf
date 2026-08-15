# `quad3d_jt_2.9.1` — v2.9.1 secured experiment entry
**This is an experiments entry, not a SOTA snapshot.** It is never eligible for ledger SOTA bolding (`06_workflow` §2.5, §6.3). The v2.9.1 SOTA snapshot remains `data/secured_data/v2.9.1/seed42/`, which this promotion did not touch.
## Base version and the exact change from it
- **Base version:** v2.9.1.
- **System:** `quadrotor_3d`.
- **Framework:** `jt_pncbf`.
- **Condition:** `JT3D_V291_OCR4`.
- **Source run:** `data/runs/v2.9.1/set__20260814-030107__seed42/v2.9.1__jt__20260814-030107__seed42` (copy only; the original is untouched).
- **Change from base:** this is the version's JT-PNCBF arm for this system, warm-started from **that system's own OC `best.pt`** (`training.jt.value_init_ckpt`; value only — `v_s_state` and `v_s_target_state`, no `pi_state`, so the policy trains from scratch). Trained with the empty-branch fallback OFF: `filter.empty_fallback` resolves to `{mode none, k 10}`, verified before launch by read-back from the CONSTRUCTED collection filter rather than from the config.

## Verdict
Registered as **ledger row L312**, `cps` **0.867987**, scored on the registered cell through `v282_agree_gate.gate_overrides` — the same producer and cell as L304 and every other v2.9.1 row — on pool `eval_fullcb_quadrotor-3d-d2r_n2000_seed823456 (sha8 3682a4e3)`, n 2000, ebs 2000, terminal (0.15, 0.3, 0.3). Single seed 42.

This entry is one cell of the version's **Table I coupling ladder**. Its pair on the identical pool and cell is **L305**: OC 0.681835 → JT 0.867987, **+0.186152**. The unfiltered LQR nominal on the same cell and pool scores `cps` -0.647000.

**No CI-separated claim is made.** Single seed on both sides of the pair, and `cps` is not commensurable across systems (`06_workflow` §2.5), so the ladder is read down each column and never across.

Against the bold row **L304** (0.872223) on the identical pool and cell this row is **−0.004236**, i.e. 0.51× the 0.0083 admissibility floor and below it; its point estimate lies inside L304's own 95% CI [0.8471, 0.8962]. Per the Researcher's disposition fixed in advance of the number: Table I takes this row for the 3D JT cell and **the bold stays at L304**.

## Interpreting section
`docs/versions/v2.9.1_results.md` — Table I and the per-column pool/cell verification.
Detail: `docs/versions/v2.9.1/oc_arm_rows.md` (OC rows), `docs/versions/v2.9.1/jt_planar.md` (JT arms), `docs/versions/v2.9.1/final_scoring.md` §15 (the 3D JT run).

## File inventory
`04_eval` §7.5 file set as far as this run produced it: **12 of 12**, **43,636,366 bytes**.

All twelve §7.5 files present.

## Identity record — digests
This README is this entry's identity record and the only file here carrying digests (`06_workflow` §6.1); no hash appears in any prose file.

| file | sha256 |
|---|---|
| `config.yaml` | `fad6932ce256eadc6bd8b8e56ba97b0853b2a8ec77ea338fb679a026a35b8bb8` |
| `git_commit.txt` | `47e7328714818c5a47316e712912a37d57a1c63cba6fb4905ebdf140a2bcc6c8` |
| `eval_metrics.csv` | `cd90c034ebf7bacca9934362b114e7cec623b81b12fdd1305ea8a659a2a2decf` |
| `eval_episodes.csv` | `6e7c49847f7ad31cba60c31da21781dc027e9703fe8a95a91cc81a53b999a7e0` |
| `pool_manifest.json` | `9328bca7000948cc8796f68735aa62a6630249fb649a3c53086378a644ab3614` |
| `status.json` | `e90df34fffe0d93cd8f23752d0df799640bc4cfa97d8dbceacd6a318d3f4d4c1` |
| `checkpoints/best.pt` | `b37e29f895ec263caa53a5b658aa808d4a46c6aa1291ef37d559c7d174c4d69a` |
| `checkpoints/final.pt` | `d1378c76bff9c70e7394adff99cf9e7e5c6a2316af8c013b56b51dee00d034ff` |
| `figures/trajectory_grid_A.png` | `246bbc679fb7518c342e9ce763ed2b0d5ee324a9024ec12ccfaf396fd58694e8` |
| `figures/trajectory_grid_B.png` | `ee64447d9bef53e73accc9a1147d39449b0f5096b02165d6d915e0f4849a4ab9` |
| `figures/cbf_contour.png` | `c3b0c7b35dbd20005feb75d7b9c2d7241ebda9b455b0ac3e5fd7310e49cf2972` |
| `report.md` | `5fe8dd6c9a3cdca3dd600663e4a1042ddeb861e33f5175330b48eb18508cf9c5` |
