# `quad3d_oc_2.9.1` — v2.9.1 secured experiment entry
**This is an experiments entry, not a SOTA snapshot.** It is never eligible for ledger SOTA bolding (`06_workflow` §2.5, §6.3). The v2.9.1 SOTA snapshot remains `data/secured_data/v2.9.1/seed42/`, which this promotion did not touch.
## Base version and the exact change from it
- **Base version:** v2.9.1.
- **System:** `quadrotor_3d`.
- **Framework:** `oc_pncbf`.
- **Condition:** `OC3D_V291_R4`.
- **Source run:** `data/runs/v2.9.1/set__20260813-164148__seed42/v2.9.1__oc__20260813-164148__seed42` (copy only; the original is untouched).
- **Change from base:** this is the version's OC-PNCBF arm for this system — the certificate trained against the nominal policy, which the JT arm then warm-starts from. Schedule frozen (`schedules.gamma_disc.total_epochs` 10000 against a 1000-epoch run, so `target_rhs` is identically 0.0) and buffer capped at 10000 trajectory records — the two repairs that followed the tail-bootstrap ratchet collapse and the `_cat_optional` OOM.

## Verdict
Registered as **ledger row L305**, `cps` **0.681835**, scored on the registered cell through `v282_agree_gate.gate_overrides` — the same producer and cell as L304 and every other v2.9.1 row — on pool `eval_fullcb_quadrotor-3d-d2r_n2000_seed823456 (sha8 3682a4e3)`, n 2000, ebs 2000, terminal (0.15, 0.3, 0.3). Single seed 42.

This entry is one cell of the version's **Table I coupling ladder**. Its pair on the identical pool and cell is **L312**: OC 0.681835 → JT 0.867987, **+0.186152**. The unfiltered LQR nominal on the same cell and pool scores `cps` -0.647000.

**No CI-separated claim is made.** Single seed on both sides of the pair, and `cps` is not commensurable across systems (`06_workflow` §2.5), so the ladder is read down each column and never across.

## Interpreting section
`docs/versions/v2.9.1_results.md` — Table I and the per-column pool/cell verification.
Detail: `docs/versions/v2.9.1/oc_arm_rows.md` (OC rows), `docs/versions/v2.9.1/jt_planar.md` (JT arms), `docs/versions/v2.9.1/final_scoring.md` §15 (the 3D JT run).

## File inventory
`04_eval` §7.5 file set as far as this run produced it: **9 of 12**, **38,958,834 bytes**.

**Not produced by this run:**

- `figures/trajectory_grid_A.png`
- `figures/trajectory_grid_B.png`
- `figures/cbf_contour.png`

The three top-level figures are written only by the JT end-of-run path; the OC trainer writes its figures per-eval into `figures/inloop/` instead (201 files in this run), which `04_eval` §7.5 does not include in the secured set. They remain in the original run directory.

## Identity record — digests
This README is this entry's identity record and the only file here carrying digests (`06_workflow` §6.1); no hash appears in any prose file.

| file | sha256 |
|---|---|
| `config.yaml` | `27b7c38b069b76481459b69fc7e892b8ccbbf4f8a015cea70127a73286736a35` |
| `git_commit.txt` | `47e7328714818c5a47316e712912a37d57a1c63cba6fb4905ebdf140a2bcc6c8` |
| `eval_metrics.csv` | `3ed79e93c8bf9f31f56e8b232548ad8a83bee3ad398210d59ed2578fd04e2a2b` |
| `eval_episodes.csv` | `25c6f597a8f7139c25af6af3e68edb613140f054f338cc4bc096c1d5df8e94b5` |
| `pool_manifest.json` | `9328bca7000948cc8796f68735aa62a6630249fb649a3c53086378a644ab3614` |
| `status.json` | `8fc9bb618dc40e418ec14fe0bc319f002116dd520bd277e572244915a78ff98c` |
| `checkpoints/best.pt` | `c162950ba746ddfd850cd14f48ef9d62422e90a37b2e31ad8960c8a86742fd0d` |
| `checkpoints/final.pt` | `5bf7c6ba439c4ef3cfd98ad6cf84d999446e74f5756b130b74f16421ac7b6977` |
| `report.md` | `93534076b964bf61d58edbb3f68869ebb72f398066887c13f2e18fb5ba814a42` |
