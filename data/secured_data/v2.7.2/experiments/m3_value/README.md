# v2.7.2 experiment: `m3_value` (OC-PNCBF value dependency)

- **Base version:** v2.7.2 (quadrotor_3d bring-up).
- **Exact change from base:** none — this is not an ablation. It is the M3 OC-PNCBF value
  V^{h★,π_nominal} (best.pt @40500, sha8 e3ab0940) from run `v2.7.2__20260718-204313__seed42`, the
  frozen-nominal value that the version's headline M5 JT run (`seed42/`) was **value-initialized** from.
  It is kept here because `seed42/` holds exactly one snapshot (the SOTA JT) and the value run is a
  dependency, not the version's SOTA.
- **Verdict:** dependency artifact, NOT a SOTA claim. Never eligible for ledger SOTA bolding
  (`06_workflow` §6.3 / `04_eval` §2.4). The version's SOTA remains `data/secured_data/v2.7.2/seed42/`.
- **Interpreting section:** `docs/versions/v2.7.2_results.md` §3 (bring-up facts — M3 value + authority gate)
  and §1 (lineage headline); build-log `docs/versions/v2.7.2/phase_v272_report.md` M3 section.
- **File set:** the `04_eval` §7.5 set the run produced (config, metrics/eval CSVs, pool_manifest, report,
  checkpoints/{best,final}, figures/{trajectory_grid_A,B}); `cbf_contour.png` was skipped for 3-D (viz-only,
  build-log follow-up 3). SHA-256 pins are in `../../seed42/ADOPTED.md`.
