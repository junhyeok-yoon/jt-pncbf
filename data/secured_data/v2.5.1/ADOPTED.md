# v2.5.1 — Certified Policy Iteration iteration-0 — ADOPTED (canonical seed scheme)

Backup-only certificate regression: a learned, conservatively-calibrated surrogate of the deadband-brake
(m_0) stopping value with unclipped interior gradients. **No policy training, no filter, no JT, and no
episode evaluation this version** — hence no `eval_metrics.csv` and **no ledger row** (the 04_eval §5
cps comparison is inapplicable, recorded as such). Not a SOTA claim; comparison targets are the exact
oracles V_m0 and V_M17, not a prior version.

Canonical run timestamp `20260711-050924`; seeds `{42, 99, 12345}` (standing convention); labels from
`master_seed 42` via `SeedSequence.spawn(4)` [scene, state, oracle-subset, split]. Source (read-only)
`data/previous_runs/v2.5.1__20260711-050924__{seed42, seed99, seed12345, cpi_labels_seed42}`.

## Checkpoint identity (per seed)

| seed | file | md5 | sha256 | bytes |
|---|---|---|---|---|
| 42 | best.pt | `598333677442ae2ac91a4e435b0e8679` | `de804a22cb3ed28b107e6a3451ad49898d0ca09e6719f6f96d7ac2e368b3ec52` | 557,987 |
| 42 | final.pt | `23dc493ef82f02dfc578fc5a00978fa6` | `e3ae2f16013cf392552e7792e15d17ac7789bfaea10bc20f39b58be477b685b3` | 558,321 |
| 99 | best.pt | `aed0b5456fde16818f1a101911ce0b43` | `11c8e1fb50200f24e16a2a4a34a597bbff4a4ccd7acc10082c845d2bc75ed8ef` | 557,987 |
| 99 | final.pt | `e545fe8f517f0fdc795e8f4603cb3ea7` | `f574e231bbf033b95dcbb9a737eca9db7ab5aff9ff9e07d9d8376fa34c0fcd1d` | 558,321 |
| 12345 | best.pt | `400d3fed39af6e309b2e1208fbb0b763` | `9305b87f99eadb8d847d1ccb53223ecdc844d21ef421f62404e262901e17e1fa` | 557,987 |
| 12345 | final.pt | `ca62d15113ceb115e482e3e2c6be4ad2` | `1f6580f1c1efd7639163e04e296f116e19724489d3fac3084e5f0f85640464b6` | 558,385 |

Model: value trunk 3×256 Softplus(β=20), raw linear head, dim-19 DI observation, single member (`best.pt` =
best-val, `final.pt` = last epoch). Per-seed `audit/metrics.json` holds the calibration + test-audit records.

## Canonical scorecard (held-out test split; verdict is the Researcher's)

| prediction | threshold | canonical {42,99,12345} | verdict |
|---|---|---|---|
| **P-PI1a** frac(gap-set G satisfies criterion C) | ≥ 0.70 | **0.7767** | CONFIRMED |
| **P-PI1b** IoU{V̂≤0, V_m0≤0} | ≥ 0.95 | **0.9599** (all seeds ≥0.956) | CONFIRMED |
| **P-PI1b** coverage @ α0.10 (test-in-band) | ≥ 0.90 | **0.9096** mean (seed42 0.8923 marginal, CI[0.878,0.905]) | CONFIRMED (mean) |
| **P-PI1c** eps_q(0.10, band 0.125) | ≤ 0.05 (< δ_grid 0.0625) | **0.0189** mean, all seeds ≤0.047 | CONFIRMED (all seeds) |

Per-seed: best val pinball 0.05959 / 0.05761 / 0.05941; eps_q(0.10) 0.0097 / 0.0000 / 0.0472; IoU 0.9624 /
0.9560 / 0.9614; coverage 0.8923 / 0.9353 / 0.9012. Gap 8a: |G|=3,072, certified-fraction ratio 0.9852
[scene-bootstrap 95% CI 0.9846, 0.9858]. seed-42 figures F1–F6 under `seed42/figures/`.

Full method + numbers: `docs/versions/v2.5.1/phase_i0_report.md` (P0–P6 first-pass history + R0–R7
canonical). The first-pass ad-hoc-seed runs (labels 31415/…, seeds 42/43/44) were superseded by this
seed-convention correction; seed43/44 dirs were deleted by Researcher direction (see report R6).

## Labels
`labels/` is manifest-only (`manifest.json`, `oracle_summary.json`, `gap_audit.json`,
`results_summary.json`); raw npz shards + `oracle.npz` are regenerable from `master_seed 42` (see
`labels/README.md`). SHA-256 per shard is pinned in `manifest.json`.

---

# v2.5.1 — I3 exact-backup filter (E-axis) — ARTIFACT ARCHIVE (Researcher-approved 2026-07-13)

**Artifact archive, NOT SOTA — DI SOTA remains v2.3.0 cps_v2 0.9005.** The E-axis (`safety_channel` type
`exact_m0`: the unclipped single-backup certificate V_m0 as the deployed filter) is the only I3 axis that
met its pre-registered bars (P-E1 & P-E2 CONFIRMED). Deposited under `exact_backup/` + `certificate_chain/`
+ `tables/`. D2 (D-axis) is ABANDONED (P-D1 falsified) and NOT archived.

## E-axis identity records (exact_m0 policies; full n2000, arm-A cps_v2 | arm-B; P-I1-2 PASS all arms)
- seed 42  — source `data/previous_runs/v2.5.1__20260713-040300__seed42`, `exact_backup/pi_exactm0_seed42/best.pt`
  sha256 `7252db13e869b36a…`; arm-A 0.8226 [0.797,0.847] coll 0.0015 | arm-B 0.8367 (P-E1 PASS).
- seed 99  — source `data/previous_runs/v2.5.1__20260713-100029__seed99`, `exact_backup/pi_exactm0_seed99/best.pt`
  sha256 `8ede23f53a363061…`; arm-A 0.8280 coll 0.0015 | arm-B 0.8422 (BEST single-seed; best.pt selection
  confirmed on a fresh n1000 train-mode pool).
- seed 12345 — source `data/previous_runs/v2.5.1__20260713-100037__seed12345`, `exact_backup/pi_exactm0_seed12345/best.pt`
  sha256 `480e34bc6c8f920c…`; arm-A 0.7974 coll 0.0025 | arm-B 0.8147.
- **Pooled 3-seed arm-A cps_v2 = 0.8160 [0.801, 0.829]**, collision 0.0018 (P-E2 PASS). Channel = exact_m0
  (no checkpoint; deployed h_fn = m0_value_raw, 25-step deadband brake). Warm-started from the pathfinder
  it1 / R3 / R4 policies (paired vs the learned-V̂₁ it2_ws: +0.085/+0.091/+0.049).

## Certificate chain (`certificate_chain/`; manifest-only, regenerable from master-seed 42)
- `V_hat_1/best.pt` sha256 `b87b97a3208434bb…`; `V_hat_2p/best.pt` (A6 uniform-dist V₂′) sha256
  `31bc37a3caa182ef…`; `V_hat_2dm/best.pt` (P1 distribution-matched) sha256 `b8df7b0236ba8494…`.
- `family_registry.json` sha256 `48ec0170bc5694c7…` (T₁..T₃); `chain_audit.json` — all P-A1 checks 0
  violations, cert fraction ~0.873, cert-ratio /V_M17 ~1.015, gap recovery ~0.988.

Adopted-for-record only; deployment of the exact-backup channel is a Researcher decision. SOTA unchanged.
