# v2.5.0 experiments (secured diagnostics — NOT SOTA snapshots)

Analysis-only, no-training diagnostics kept for the record (04_eval §7.5, 06_workflow §6.3). None is
SOTA-bolded in the ledger. Build-logs remain local-only under `docs/versions/v2.5.0/` (the authoritative
narrative); this directory holds the numeric artifacts they cite.

## stage_R/ — certified reach-time R(x), training-free (Stage R & R-2)
Plan-family coverage + plan-follower evaluation on the DI n2000/n500 pools. Grammar R-1 (aborted, coverage
0.0945) and R-2 (gate passed, coverage 0.98; rescue + cost hypotheses refuted). Key files: `arm_r0*.json`
(coverage), `arm_r1.json`+`arm_r1_follow.pt`+`figs/` (follower n2000), `arm_r2*.json`+`stall_onsets.{pt,json}`
(stall rescue + true onset states + clusters), `arm_r3.json` (cost), `R2_{i..iv}_start.pt`,
`exhaustive_R_start.pt`, `retrieve_q1q2.json`. Narrative: `docs/versions/v2.5.0/reach_witness.md`.
Modules: `scripts/analysis/reach_witness.py`, `reach_eval.py`.

## arm_c/ — learned filter at deploy rates (close-prep T-A)
v2.3.0 learned-V HardNet filter re-eval via `scripts/analysis/deploy_rate_eval.py --filter learned`:
`deploy_v23learned_Dslow.json` (batch200), `_Dslow_b250.json` (gate-matching batch, reproduces the rescore
reference), `_Dfast.json` (dt 0.01). Finding: the D-fast gain is V_M-specific, not shared by the learned
filter. `tb_backfill.json`/`tb_v221.json` = the ledger cps_v2 backfill scoring log. Narrative:
`docs/versions/v2.5.0/close_prep_batch.md` T-A/T-B.

## rescore/ — ledger cps_v2 re-score set (n2000, deterministic)
`scripts/analysis/rescore_cps_v2.py` outputs for every ledger row scored to fill the cps_v2 column
(v2.0.1, v2.2.1, v2.3.0, v2.3.1, v2.4.x). Standard n2000 pool. Narrative:
`docs/versions/v2.5.0/ledger_v2_rescore.md` + `close_prep_batch.md` T-B.
