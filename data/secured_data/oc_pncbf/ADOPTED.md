# Adopted OC-PNCBF DI Baseline

This directory records the adopted canonical OC-PNCBF Double Integrator baseline for
v2.0.0.

- run_id: `v2.0.0__20260529-001441__seed42`
- source run directory: `data/v2.0.0__20260529-001441__seed42`
- secured artifact directory:
  `data/secured_data/oc_pncbf/v2.0.0__20260529-001441__seed42`
- adopted checkpoint: `checkpoints/best.pt`
- best step: `40000`
- best checkpoint SHA256:
  `face0df625ead6acb556dfabab3e02a55fb8d133308225b13825de196416757e`

Final full-pool metrics at default alphas (`alpha_safe = 2.0`,
`alpha_unsafe = 100.0`) on `eval_full_di_n500_seed23456`:

| metric | value |
|---|---:|
| cps | 0.796995304719 |
| reach | 0.928 |
| collision | 0.004 |
| oob | 0.000 |
| stuck | 0.062 |
| timeout | 0.006 |
| infeasibility | 0.193348984271 |
| saturation_rate | 0.151770513108 |

Evaluation pools:

| pool | scenes | seed | SHA256 |
|---|---:|---:|---|
| `eval_inloop_di_n200_seed12345.pkl` | 200 | 12345 | `072438b7745235b6c2ac38e617a261edd875d10b788182deb540dc7c9acf9d9c` |
| `eval_full_di_n500_seed23456.pkl` | 500 | 23456 | `c5e5095226d90bce0d28ac76e92e74966dc94a4d41d7f8b964212d20fd9978b4` |

The pool SHA values above match
`v2.0.0__20260529-001441__seed42/pool_manifest.json`.

The secured artifact set intentionally excludes the bulky per-step `metrics.csv`
training log. The full raw training curve remains in the gitignored source run
directory at `data/v2.0.0__20260529-001441__seed42/metrics.csv`; the secured set keeps
the checkpoint, effective config, pools, final evaluation CSVs, reports, status, and
figures needed to reproduce and inspect the adopted baseline.

This is the adopted canonical OC-PNCBF DI baseline for v2.0.0.
