# `planar_oc_2.9.1` — v2.9.1 secured experiment entry
**This is an experiments entry, not a SOTA snapshot.** It is never eligible for ledger SOTA bolding (`06_workflow` §2.5, §6.3). The v2.9.1 SOTA snapshot remains `data/secured_data/v2.9.1/seed42/`, which this promotion did not touch.
## Base version and the exact change from it
- **Base version:** v2.9.1.
- **System:** `quadrotor_planar`.
- **Framework:** `oc_pncbf`.
- **Condition:** `OCPLANAR_V291_R3`.
- **Source run:** `data/runs/v2.9.1/set__20260813-164154__seed42/v2.9.1__oc__20260813-164154__seed42` (copy only; the original is untouched).
- **Change from base:** this is the version's OC-PNCBF arm for this system — the certificate trained against the nominal policy, which the JT arm then warm-starts from. Schedule frozen (`schedules.gamma_disc.total_epochs` 10000 against a 1000-epoch run, so `target_rhs` is identically 0.0) and buffer capped at 10000 trajectory records — the two repairs that followed the tail-bootstrap ratchet collapse and the `_cat_optional` OOM.

## Verdict
Registered as **ledger row L306**, `cps` **0.659150**, scored on the registered cell through `v282_agree_gate.gate_overrides` — the same producer and cell as L304 and every other v2.9.1 row — on pool `eval_full_quadrotor-planar_n2000_seed23456`, n 2000, ebs 2000, terminal (0.15, 0.3, 0.3). Single seed 42.

This entry is one cell of the version's **Table I coupling ladder**. Its pair on the identical pool and cell is **L311**: OC 0.659150 → JT 0.775740, **+0.116590**. The unfiltered LQR nominal on the same cell and pool scores `cps` -0.206000.

**No CI-separated claim is made.** Single seed on both sides of the pair, and `cps` is not commensurable across systems (`06_workflow` §2.5), so the ladder is read down each column and never across.

## Interpreting section
`docs/versions/v2.9.1_results.md` — Table I and the per-column pool/cell verification.
Detail: `docs/versions/v2.9.1/oc_arm_rows.md` (OC rows), `docs/versions/v2.9.1/jt_planar.md` (JT arms), `docs/versions/v2.9.1/final_scoring.md` §15 (the 3D JT run).

## File inventory
`04_eval` §7.5 file set as far as this run produced it: **9 of 12**, **16,298,383 bytes**.

**Not produced by this run:**

- `figures/trajectory_grid_A.png`
- `figures/trajectory_grid_B.png`
- `figures/cbf_contour.png`

The three top-level figures are written only by the JT end-of-run path; the OC trainer writes its figures per-eval into `figures/inloop/` instead (201 files in this run), which `04_eval` §7.5 does not include in the secured set. They remain in the original run directory.

## Identity record — digests
This README is this entry's identity record and the only file here carrying digests (`06_workflow` §6.1); no hash appears in any prose file.

| file | sha256 |
|---|---|
| `config.yaml` | `e90f0a659037be0eeb658372ca9705ef9fac30f1bd6c44d914f5987c69d45070` |
| `git_commit.txt` | `47e7328714818c5a47316e712912a37d57a1c63cba6fb4905ebdf140a2bcc6c8` |
| `eval_metrics.csv` | `867745f7a27e6f232c1df30d0e5110ed9fdf998123d9d37656f1749d154a7cf4` |
| `eval_episodes.csv` | `aa8934b8a4b8e1a2c2583536f3e64044bf5f5a0320485629885bed61bf0e079a` |
| `pool_manifest.json` | `c182e643e80d19dd78d77a659dd57f2a57eded70ccdbb50d6d28cc883531983a` |
| `status.json` | `cc709e46063b417451e4d56c5caf2dfd147a9aee4774cc4457734b734f251cde` |
| `checkpoints/best.pt` | `bdc692b51e4fa0f8d43839fcd1c4f4af20c0666e51e6116cad0a21197c5a15f3` |
| `checkpoints/final.pt` | `af661d2a2ec8d9d5d21f9fedd1f9e081ed4d979b72053bc00f63e17061e20f40` |
| `report.md` | `a20089475d0c888fe3cb5a2451f125192b1ae3a86772015e3310e529d915d12c` |
