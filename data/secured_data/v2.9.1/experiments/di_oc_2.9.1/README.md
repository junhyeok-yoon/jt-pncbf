# `di_oc_2.9.1` — v2.9.1 secured experiment entry
**This is an experiments entry, not a SOTA snapshot.** It is never eligible for ledger SOTA bolding (`06_workflow` §2.5, §6.3). The v2.9.1 SOTA snapshot remains `data/secured_data/v2.9.1/seed42/`, which this promotion did not touch.
## Base version and the exact change from it
- **Base version:** v2.9.1.
- **System:** `double_integrator`.
- **Framework:** `oc_pncbf`.
- **Condition:** `OCDI_V291`.
- **Source run:** `data/runs/v2.9.1/set__20260813-170950__seed42/v2.9.1__oc__20260813-170950__seed42` (copy only; the original is untouched).
- **Change from base:** this is the version's OC-PNCBF arm for this system — the certificate trained against the nominal policy, which the JT arm then warm-starts from. Schedule frozen (`schedules.gamma_disc.total_epochs` 10000 against a 1000-epoch run, so `target_rhs` is identically 0.0) and buffer capped at 10000 trajectory records — the two repairs that followed the tail-bootstrap ratchet collapse and the `_cat_optional` OOM.

## Verdict
Registered as **ledger row L307**, `cps` **0.880459**, scored on the registered cell through `v282_agree_gate.gate_overrides` — the same producer and cell as L304 and every other v2.9.1 row — on pool `eval_full_di_n2000_seed123456`, n 2000, ebs 2000, terminal (0.15, 0.3, 0.3). Single seed 42.

This entry is one cell of the version's **Table I coupling ladder**. Its pair on the identical pool and cell is **L309**: OC 0.880459 → JT 0.904757, **+0.024298**. The unfiltered LQR nominal on the same cell and pool scores `cps` -0.009500.

**No CI-separated claim is made.** Single seed on both sides of the pair, and `cps` is not commensurable across systems (`06_workflow` §2.5), so the ladder is read down each column and never across.

## Interpreting section
`docs/versions/v2.9.1_results.md` — Table I and the per-column pool/cell verification.
Detail: `docs/versions/v2.9.1/oc_arm_rows.md` (OC rows), `docs/versions/v2.9.1/jt_planar.md` (JT arms), `docs/versions/v2.9.1/final_scoring.md` §15 (the 3D JT run).

## File inventory
`04_eval` §7.5 file set as far as this run produced it: **9 of 12**, **15,822,417 bytes**.

**Not produced by this run:**

- `figures/trajectory_grid_A.png`
- `figures/trajectory_grid_B.png`
- `figures/cbf_contour.png`

The three top-level figures are written only by the JT end-of-run path; the OC trainer writes its figures per-eval into `figures/inloop/` instead (201 files in this run), which `04_eval` §7.5 does not include in the secured set. They remain in the original run directory.

## Identity record — digests
This README is this entry's identity record and the only file here carrying digests (`06_workflow` §6.1); no hash appears in any prose file.

| file | sha256 |
|---|---|
| `config.yaml` | `5c1b3b8ba93ffd98df49334717a915d0d98b11d09bb05e32ea128d2cabaa8996` |
| `git_commit.txt` | `47e7328714818c5a47316e712912a37d57a1c63cba6fb4905ebdf140a2bcc6c8` |
| `eval_metrics.csv` | `b76f5cc680a5f7a5ecc6915c730814354af4b78a7ca9cb948a0503ca2413d296` |
| `eval_episodes.csv` | `9f3bca5b63ac50258ce9794a7323fa315f3a209c1deaa95f0bde882b87e7794c` |
| `pool_manifest.json` | `fbb7f2389084269c912fa93f0d38b01c1df38d863f97813d1e70b3f08770d2a5` |
| `status.json` | `1a5439a49d6e2b784bfe9c6602c82e1e1ea2ad4d379f0c170f153a6f2c5be123` |
| `checkpoints/best.pt` | `73c9145ce893c636cc3ef1d50a41d3f2b1d42dec4cc8f3b2d3f590a62c5b813f` |
| `checkpoints/final.pt` | `89bd4767182ddac9f1c0ab234bd3d7cc0403911676c0d68336cfcbd4ff5403ab` |
| `report.md` | `0788058d12a4cdad635fb6f493ffebb3f5c00f4ebba6b844a0693d8fd62832a2` |
