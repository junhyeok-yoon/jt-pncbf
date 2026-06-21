# Adopted v2.3.0 Seed 99

- Source run id: `v2.3.0__20260620-164338__seed99`
- Source run dir: `data/previous_runs/v2.3.0__20260620-164338__seed99/` (moved at close)
- Run git commit: `784b5ddc0113c611b17ab98e90d265eb5a459ed6 DIRTY`
- Seed: `99`
- Config: D_pi=2k + D_V=1M @50k (policy_buffer_cap=2000, buffer_cap=1,000,000; injection OFF; goal_speed_radius 0.30)
- Adopted checkpoint (n2000 re-selected): `checkpoints/step_045000.pt` (step 45000)
- In-loop best.pt step: 46500
- `step_045000.pt` SHA-256: `25d61c8bac28ab4a80636f66013be7eabb884f32d299d34d0e24da4d661bd672`
- `best.pt` SHA-256: `b084b18498f4a66e323bac7f726e1ac07144c09bcd07ff5a7c08c0c0be5fed8d`
- `final.pt` SHA-256: `478fed7bd6c6db2bb2513bb574294a66cd82204b92b76847abbef46f686dca0d`

## Adopted n2000 eval (full pool, seed 23456, @ goal_speed_radius 0.30)

| cps | 95% CI | reach | collision | stuck | timeout | infeasibility |
|---:|:---:|---:|---:|---:|---:|---:|
| 0.8628 | [0.8417, 0.8828] | 0.9530 | 0.0085 | 0.0250 | 0.0135 | 0.1383 |

Part of the v2.3.0 DI SOTA-of-record 3-seed set (mean n2000 cps 0.8698). See `../aggregate/`.
