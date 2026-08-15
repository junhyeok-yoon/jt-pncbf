# `uni_oc_2.9.1` — v2.9.1 secured experiment entry
**This is an experiments entry, not a SOTA snapshot.** It is never eligible for ledger SOTA bolding (`06_workflow` §2.5, §6.3). The v2.9.1 SOTA snapshot remains `data/secured_data/v2.9.1/seed42/`, which this promotion did not touch.
## Base version and the exact change from it
- **Base version:** v2.9.1.
- **System:** `unicycle`.
- **Framework:** `oc_pncbf`.
- **Condition:** `OCUNI_V291`.
- **Source run:** `data/runs/v2.9.1/set__20260813-170955__seed42/v2.9.1__oc__20260813-170955__seed42` (copy only; the original is untouched).
- **Change from base:** this is the version's OC-PNCBF arm for this system — the certificate trained against the nominal policy, which the JT arm then warm-starts from. Schedule frozen (`schedules.gamma_disc.total_epochs` 10000 against a 1000-epoch run, so `target_rhs` is identically 0.0) and buffer capped at 10000 trajectory records — the two repairs that followed the tail-bootstrap ratchet collapse and the `_cat_optional` OOM.

## Verdict
Registered as **ledger row L308**, `cps` **0.670486**, scored on the registered cell through `v282_agree_gate.gate_overrides` — the same producer and cell as L304 and every other v2.9.1 row — on pool `eval_full_unicycle_n2000_seed123456`, n 2000, ebs 2000, terminal (0.15, 0.3, 0.3). Single seed 42.

This entry is one cell of the version's **Table I coupling ladder**. Its pair on the identical pool and cell is **L310**: OC 0.670486 → JT 0.873126, **+0.202640**. The unfiltered LQR nominal on the same cell and pool scores `cps` 0.002000.

**No CI-separated claim is made.** Single seed on both sides of the pair, and `cps` is not commensurable across systems (`06_workflow` §2.5), so the ladder is read down each column and never across.

## Interpreting section
`docs/versions/v2.9.1_results.md` — Table I and the per-column pool/cell verification.
Detail: `docs/versions/v2.9.1/oc_arm_rows.md` (OC rows), `docs/versions/v2.9.1/jt_planar.md` (JT arms), `docs/versions/v2.9.1/final_scoring.md` §15 (the 3D JT run).

## File inventory
`04_eval` §7.5 file set as far as this run produced it: **9 of 12**, **15,741,795 bytes**.

**Not produced by this run:**

- `figures/trajectory_grid_A.png`
- `figures/trajectory_grid_B.png`
- `figures/cbf_contour.png`

The three top-level figures are written only by the JT end-of-run path; the OC trainer writes its figures per-eval into `figures/inloop/` instead (201 files in this run), which `04_eval` §7.5 does not include in the secured set. They remain in the original run directory.

## Identity record — digests
This README is this entry's identity record and the only file here carrying digests (`06_workflow` §6.1); no hash appears in any prose file.

| file | sha256 |
|---|---|
| `config.yaml` | `df7a04bfc2d591b6be55110f12ab56fddaab4c6f4b2918d02186132413e3b4d7` |
| `git_commit.txt` | `47e7328714818c5a47316e712912a37d57a1c63cba6fb4905ebdf140a2bcc6c8` |
| `eval_metrics.csv` | `bf0a75c00bae4131e1cfad47cbecc382213b3fef815bd8927cda333831c16c64` |
| `eval_episodes.csv` | `6bc19f17be90bec0cfd9e7d51c9d43fa6fff4e72667d009c40e268500e51dd76` |
| `pool_manifest.json` | `1981741b654b5c10da1f21e91dbc6816a5960dc2a744a4a2a6cc7bc4b9c14999` |
| `status.json` | `9dac10e85c7371ef39308dcc4cff7955619c404f842252d416b9daaef11720b3` |
| `checkpoints/best.pt` | `40d075e4f1f731b4ce369cce25a476cdf2a71e42f5073b49082d687269f0caf6` |
| `checkpoints/final.pt` | `cb3e8ebda9db6cb36a3731856de0f6618d505e27f06e5e6d2288f2eef9015385` |
| `report.md` | `edf8c2187a0bccee1ae5f817fcb58f7f0ec4499b16d3d64e1b366df4bec13da1` |
