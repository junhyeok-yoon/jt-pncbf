# v2.5.1 iteration-0 — backup-only certificate labels (canonical, manifest-only)

Base: v2.5.1 Certified Policy Iteration, iteration-0. Complete-MC labels of the deadband-brake (m_0)
stopping value `V_raw = max_{0<=k<=25} h_raw(x_k)` (unclipped signed_h; `clip(V_raw)` == maneuver_value{m_0}
exactly). Source run dir `data/previous_runs/v2.5.1__20260711-050924__cpi_labels_seed42`.

## Canonical seed scheme
All label-pipeline randomness derives from a single master seed via
`numpy.SeedSequence(master_seed=42).spawn(4)` in the fixed order **[scene, state, oracle-subset, split]**
(`src/frameworks/cpi/labels.py:label_streams`). Training seeds are the standing convention `{42, 99, 12345}`.

## What is secured here (manifest-only)
- `manifest.json` — 20,000-scene dataset: 1,241,100 states (uniform 1.20M + boundary 41,100),
  scene-split 0.72/0.08/0.10/0.10 (pairwise-disjoint), SHA-256 per npz shard, per-split summary stats,
  dt_vm 0.05, T_stop 25, label_batch 65536, float64 stop-and-hold 6.94e-18.
- `oracle_summary.json` — V_M17 (17-member library, REFERENCE path) on 241,100 states; fast-vs-reference
  max|Δ| 2.03e-6.
- `gap_audit.json` — 8a gap-set audit: |G|=3,072, frac_C 0.7767, certified-fraction ratio 0.9852.
- `results_summary.json` — full R3–R5 numbers.

## Raw data regeneration (npz NOT secured)
The npz dataset shards and `oracle.npz` are **NOT** copied here (Researcher decision): they are
deterministically regenerable from `master_seed 42` (SeedSequence.spawn(4)) at the pinned `label_batch`
65536 via `src.frameworks.cpi.labels.build_dataset` / `build_oracle`. The SHA-256 hashes in `manifest.json`
pin the exact shard contents.

See `docs/versions/v2.5.1/phase_i0_report.md` §R3 for the labeling method and full audit.
