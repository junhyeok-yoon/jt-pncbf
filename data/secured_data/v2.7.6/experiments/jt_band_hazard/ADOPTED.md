# v2.7.6 band-hazard JT — secured (experiments sub-tree, NOT SOTA)

Promoted at the v2.7.6 close (2026-07-26, close step 3, amendment) per `04_eval` §7.5 into the `experiments/`
sub-tree (never SOTA-bolded, `06_workflow` §6.3). Placement is per close-step-3 §2(c): the bold does not move
(comparison unavailable — the standing comparator v2.7.2 `4baaf031` predates the per-rotor actuator refactor
and the full-SO(3)/d2r IC distribution and cannot be validly re-scored on the current plant/pool). See
`README.md`. Headline checkpoint `checkpoints/step_042000.pt` = sha8 `09c33bf4`. Excluded per §7.5: TensorBoard
events, per-step training `metrics.csv`.

## Pinned SHA-256

| file | sha256 |
|------|--------|
| checkpoints/step_042000.pt (headline) | 09c33bf4cbc60d3fb7a0f038f66c187fec168f2415c77664ad4ce8fc5c76d789 |
| checkpoints/best.pt (run in-loop @48000, non-headline) | a5c1e55674ea7b2373e35681c92cd78916259e933c8a50c0fba7eced007dafa5 |
| checkpoints/final.pt | 9a10f9f83348d6468be2544c57ee93fa2f0577265fb9172a636a8463fdc3acd1 |
| config.yaml | 9dd7a3be9e5692c813ac3be2630cf24f10f7eb4e1b6390826cdb8f1e46759892 |
| eval_episodes.csv | 8df8ab17f4f0455c1f3298f4f23bfaf015340004ffb37706976c879dd071897b |
| eval_metrics.csv | 197726b86053cd6b7ecdf3465bc863ec471d8b8015dbb2d9d73eb3e2066273ef |
| figures/cbf_contour.png | 3e929ac560b0739debe7efbb33495de8ae7464beeb7126cc36e4122e08f7e633 |
| figures/trajectory_grid_A.png | 4d571a993bbce1622bd8e6e29a16c9a923b8699c39e5370b8ed3f208ae70b36c |
| figures/trajectory_grid_B.png | e495d8ba0b17d7612b894f67fa9aef9a6f56dcfe4a57df355bdb247ba56289e0 |
| git_commit.txt | ad72a1d522ec4941e1ac7ab54ebe87a3240f9f7d6b5b2eca4450fe5e5a72ea64 |
| pool_manifest.json | 84a941457934253d05b49d4451cd75562b6aafe3713b503915a37e85f7525c21 |
| report.md | 568b0c389baf46a816c6900e4821dadc60e84da50d4e45af518b9732f502249b |
