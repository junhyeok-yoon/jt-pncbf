# v2.7.2 — ADOPTED (secured lineage snapshot, seed 42)

Lineage: **quadrotor_3d** (bring-up). Headline = M6 JT (best.pt @33000). Single-seed close; no aggregate.
`git_commit.txt = e0432d5` (the v2.7.2 close commit that this snapshot anchors).

## Pinned checkpoints (full SHA-256)

| role | secured path | source run_id | best_step | SHA-256 |
|---|---|---|---|---|
| JT best (headline) | `seed42/checkpoints/best.pt` | `v2.7.2__20260718-212348__seed42` | 33000 | `4baaf03155b76cc7eea62c1b7540cbe6d7e714bcce59d8cf7080d1525a3546f6` |
| JT final | `seed42/checkpoints/final.pt` | `v2.7.2__20260718-212348__seed42` | 50000 | `14bf5a360fc06b114c1f0046fc272632e9fa140439e82a524c7f115f3d3138f0` |
| M3 value best (dependency) | `experiments/m3_value/checkpoints/best.pt` | `v2.7.2__20260718-204313__seed42` | 40500 | `e3ab094090c8ece95cb2839dcb578dc7e218c8f7abe679610a3918b36238fc0b` |
| M3 value final (dependency) | `experiments/m3_value/checkpoints/final.pt` | `v2.7.2__20260718-204313__seed42` | 50000 | `aab1f29822628e6cf130608534b4425a96d626cc539afc43aea926813996837f` |

## Provenance
- The M5 JT run was value-initialized from the M3 value `best.pt` (dependency above).
- Source runs' run-time `git_commit.txt` was `ccbc524968acd42e39af0c5a97f573d10094be23 DIRTY` (produced on the
  v2.7.1 tree with the v2.7.2 code uncommitted); the secured `git_commit.txt` pins the close commit `e0432d5`.
- Headline metrics (`04_eval` canonical, full_n2000): cps 0.9329 [0.91764, 0.94889], collision 0.0095
  [0.0055, 0.0135] — see `docs/versions/v2.7.2_results.md` §1.
- Promotion authorized by the Researcher (v2.7.2 close-completion amendment).
