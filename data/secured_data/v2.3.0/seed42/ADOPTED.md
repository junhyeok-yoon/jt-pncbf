# Adopted v2.3.0 Seed 42

- Source run id: `v2.3.0__20260620-065914__seed42`
- Source run dir: `data/previous_runs/v2.3.0__20260620-065914__seed42/` (moved at close)
- Run git commit: `784b5ddc0113c611b17ab98e90d265eb5a459ed6 DIRTY`
- Seed: `42`
- Config: D_pi=2k + D_V=1M @50k (policy_buffer_cap=2000, buffer_cap=1,000,000; injection OFF; goal_speed_radius 0.30)
- Adopted checkpoint (n2000 re-selected): `checkpoints/step_040500.pt` (step 40500)
- In-loop best.pt step: 43500
- `step_040500.pt` SHA-256: `9e84eaa9d7a28c65b9c6d86be14e0b82dfe509a568b45d16042566f428419d76`
- `best.pt` SHA-256: `8aab773bd032f08240a2896712353e8c4e601d87f0cf86bab66355cdc38c1111`
- `final.pt`: **absent** — this run was deliberately truncated (SIGINT) at step ~50390 for the clean 50k 3-seed set; the trainer's natural-completion artifacts (`final.pt`, `report.md`) were not written. The adopted checkpoint (step_040500.pt) and best.pt are present and intact.

## Adopted n2000 eval (full pool, seed 23456, @ goal_speed_radius 0.30)

| cps | 95% CI | reach | collision | stuck | timeout | infeasibility |
|---:|:---:|---:|---:|---:|---:|---:|
| 0.8700 | [0.8488, 0.8888] | 0.9470 | 0.0075 | 0.0260 | 0.0195 | 0.0875 |

Part of the v2.3.0 DI SOTA-of-record 3-seed set (mean n2000 cps 0.8698). See `../aggregate/`.
