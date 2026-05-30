# v2.0.1 Multi-Seed Aggregate

Per-seed final evals use the full DI pool (N=500, pool seed 23456). Bootstrap CIs use 1000 within-seed resamples per seed with seed 20260508 and NumPy linear percentiles.

| seed | run_id | best_step | cps | reach | collision | stuck | oob | timeout | infeasibility | saturation_rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | v2.0.1__20260529-171057__seed42 | 34000 | 0.8853 | 0.9580 | 0.0060 | 0.0260 | 0.0000 | 0.0100 | 0.0990 | 0.1215 |
| 12345 | v2.0.1__20260529-222032__seed12345 | 36000 | 0.8775 | 0.9500 | 0.0060 | 0.0260 | 0.0000 | 0.0180 | 0.0851 | 0.1248 |
| 99 | v2.0.1__20260530-003059__seed99 | 22000 | 0.8705 | 0.9540 | 0.0040 | 0.0280 | 0.0000 | 0.0140 | 0.1351 | 0.1206 |

| aggregate | cps | reach | collision | stuck | oob | timeout | infeasibility | saturation_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mean +/- SD | 0.8777 +/- 0.0074 | 0.9540 +/- 0.0040 | 0.0053 +/- 0.0012 | 0.0267 +/- 0.0012 | 0.0000 +/- 0.0000 | 0.0140 +/- 0.0040 | 0.1064 +/- 0.0258 | 0.1223 +/- 0.0022 |
| pooled 95% CI | [0.8373, 0.9152] | [0.9340, 0.9720] | [0.0000, 0.0120] | [0.0140, 0.0420] | [0.0000, 0.0000] | [0.0040, 0.0280] | [0.0785, 0.1448] | [0.1055, 0.1400] |

## Comparison vs OC v2.0.0

JT pooled mean cps = 0.8777; pooled 95% CI = [0.8373, 0.9152]. OC v2.0.0 cps = 0.7970; OC 95% CI = [0.7444, 0.8489]. Verdict: no improvement detected.
