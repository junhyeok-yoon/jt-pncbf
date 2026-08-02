# v2.7.4 — secured snapshot (seed 42)

Promoted at the v2.7.6 close (2026-07-26, close step 3, amendment) per `04_eval` §7.5. v2.7.4 is the band-blind
JT parent (`244f4f83`) against which both v2.7.5 and v2.7.6 render their verdicts; it existed only in a
gitignored run directory, so the lineage was not reproducible from a clone without this promotion. This is a
lineage-comparator snapshot; the SOTA bold within quadrotor_3d remains v2.7.2 (`4baaf031`).

Source run: `data/runs/v2.7.4/set__20260720-083533__seed42/v2.7.4__20260720-091830__seed42/`.
Headline checkpoint: `checkpoints/best.pt` (best.pt@39000). Excluded per §7.5: TensorBoard events, per-step
training `metrics.csv`.

## Pinned SHA-256

| file | sha256 |
|------|--------|
| checkpoints/best.pt | 244f4f83622ba9811c7f4f97bd1295de6949137da19984bdfaab5c712a611f82 |
| checkpoints/final.pt | e166cdb084339170f299293849b2d83a9db9891a7626384dcd03947b97b86fa8 |
| config.yaml | 338d3d9f00daa2c8a81fae6f23c418022089e628f76436b0ff84236253cd5780 |
| eval_episodes.csv | e1a7d156193a517c18570b5f61c3cc79baa5dbebe2237871cac87fa886bbcab2 |
| eval_metrics.csv | 5ca3838bbb5215c93cc303c1088dc5c37e97f26dbf935928d4beeff44db646e5 |
| figures/cbf_contour.png | 1e0eb6bc26bd549050138baf5194a201d33224c5714b54763eb8f9db4f61c6fe |
| figures/trajectory_grid_A.png | aa59f8293086acd5343eeae226471fd0e9d22618be8166e145bb638ea5069845 |
| figures/trajectory_grid_B.png | 7e3af6916c9f50276fd51a441d986e7adf1d7e4609ae17ecea0df7044a3cf03e |
| git_commit.txt | 95cb4571309370872d131b1464c29b70b53710a44feea8a58f929b026ea069ed |
| pool_manifest.json | 84a941457934253d05b49d4451cd75562b6aafe3713b503915a37e85f7525c21 |
| report.md | 2ac99a8f3867ab96cf9300cbfcfa77d60396e4239fd39084d2b98da457fecec5 |
