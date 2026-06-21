# Adopted v2.3.0 Seed 12345

- Source run id: `v2.3.0__20260620-135807__seed12345`
- Source run dir: `data/previous_runs/v2.3.0__20260620-135807__seed12345/` (moved at close)
- Run git commit: `784b5ddc0113c611b17ab98e90d265eb5a459ed6 DIRTY`
- Seed: `12345`
- Config: D_pi=2k + D_V=1M @50k (policy_buffer_cap=2000, buffer_cap=1,000,000; injection OFF; goal_speed_radius 0.30)
- Adopted checkpoint (n2000 re-selected): `checkpoints/step_049500.pt` (step 49500)
- In-loop best.pt step: 49500
- `step_049500.pt` SHA-256: `05dafe78028538c44beaf81d5b5856b3adf9d1f9c71341ffae45dc9e5f59d735`
- `best.pt` SHA-256: `4071db3187b255e17537f16a6b0a27f0dcadb6fe892c0d0e889f8a2d30a4f7e3`
- `final.pt` SHA-256: `ff8563127e9fbd2426eed4607ede9ee7a82e58d9c7213fe4c4b5c7896a654184`

## Adopted n2000 eval (full pool, seed 23456, @ goal_speed_radius 0.30)

| cps | 95% CI | reach | collision | stuck | timeout | infeasibility |
|---:|:---:|---:|---:|---:|---:|---:|
| 0.8766 | [0.8571, 0.8956] | 0.9555 | 0.0070 | 0.0230 | 0.0140 | 0.1154 |

Part of the v2.3.0 DI SOTA-of-record 3-seed set (mean n2000 cps 0.8698). See `../aggregate/`.
