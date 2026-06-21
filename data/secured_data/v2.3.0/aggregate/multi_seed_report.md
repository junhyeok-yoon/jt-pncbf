# v2.3.0 Multi-Seed Aggregate — DI SOTA of record

Per-seed n2000 re-selected best on the full DI pool (N=2000, pool seed 23456) @ goal_speed_radius 0.30.
Per-seed 95% CIs are scene-bootstrap (1000 resamples). The cross-seed aggregate CI is the t-interval over the 3 seed means (random-effects).

| seed | run_id | adopted step | cps | 95% CI | reach | collision | stuck | timeout | infeasibility |
|---|---|---:|---:|:---:|---:|---:|---:|---:|---:|
| 42 | v2.3.0__20260620-065914__seed42 | 40500 | 0.8700 | [0.8488, 0.8888] | 0.9470 | 0.0075 | 0.0260 | 0.0195 | 0.0875 |
| 12345 | v2.3.0__20260620-135807__seed12345 | 49500 | 0.8766 | [0.8571, 0.8956] | 0.9555 | 0.0070 | 0.0230 | 0.0140 | 0.1154 |
| 99 | v2.3.0__20260620-164338__seed99 | 45000 | 0.8628 | [0.8417, 0.8828] | 0.9530 | 0.0085 | 0.0250 | 0.0135 | 0.1383 |

| aggregate | cps | reach | collision | stuck | timeout | infeasibility |
|---|---:|---:|---:|---:|---:|---:|
| mean +/- SD | 0.8698 +/- 0.0069 | 0.9518 +/- 0.0044 | 0.0077 +/- 0.0008 | 0.0247 +/- 0.0015 | 0.0157 +/- 0.0033 | 0.1137 +/- 0.0254 |
| cps seed-level 95% CI (t, 2 dof) | [0.8527, 0.8869] | | | | | |

3-seed mean cps **0.8698** (SD 0.0069, range 0.8628-0.8766), **+0.0130** over prior SOTA 0.8568.

## Verdict vs prior SOTA 0.8568 (v2.0.1)

New DI SOTA of record, promoted on a formal 3-seed comparison (mean 0.8698). Cross-seed 95% CI [0.8527, 0.8869] overlaps the prior 0.8568 -> a consistent point-estimate improvement, not a non-overlapping CI-confirmed beat. Standard config in protocol 03_train §4.2. Promoted by Researcher decision, not by the 04_eval §5 significance test.
