# mppi_n2000_2.9.2 — MPPI at the registered scale, n = 2000 — per-episode dump and cell artifact

A `06_workflow` §6.3 **experiments** entry: kept for the record, **not** a version SOTA
snapshot. An experiments entry is **never eligible for ledger SOTA bolding** (§2.5).

Backs ledger row **L317**, whose `eval_source` carries the tilt partition these
arrays were computed from.

The FIRST MPPI evaluation at n 2000 and the first MPPI ledger row. PRIVILEGED model-based reference point, NOT a peer of a learned-certificate row: MPPI reads the full 13-D state and the exact obstacle field and rolls the true plant inside its own planner, and runs an identity filter, so `infeasibility` and `mean_proj_mag` are STRUCTURALLY INAPPLICABLE rather than zero. `cps`/`cps_v2` are n/a on L317 and no composite headline is claimed. Cell difference against the learned rows: ebs 200 here against their 2000; same pool, same n 2000, same seed 42.

## The spawn-tilt partition carried on the row

`spawn_tilt = arccos(R33)` in degrees from each scene's **own stored quaternion** in the
pool, not from any rollout. **60° is the altitude-holding limit at TWR 2**, not a tuned
threshold. Every figure below is arithmetic on the per-episode arrays in this entry.

| block | n | reach | collision | `coll_obstacle` | `coll_band_lower` | shares sum |
|---|---:|---:|---:|---:|---:|---:|
| ALL | 2000 | 0.2980 | 0.6525 | 0.3280 | 0.3235 | 1.000000000000 |
| tilt ≤ 60° | 486 | 0.4198 | 0.5144 | 0.4074 | 0.1029 | 1.000000000000 |
| tilt > 60° | 1514 | 0.2589 | 0.6968 | 0.3025 | 0.3943 | 1.000000000000 |

## Per-episode arrays

`scene_idx`, `outcome`, `collision_cause`, `first_collision_step`, `d_min`,
`spawn_tilt_deg`, `n_steps` — 2000 entries each, index-aligned to the pool's own order.

## Digests (this file is the only place they appear, §6.1)

| file | sha256 |
|---|---|
| `perepisode__MPPI_CASCRATE_N2000_V292.npz` | `e242baef7cf131ea00a8833f50964561fa0647aa6fb752e9fbdbe1d22452fca2` |
| `row__MPPI_CASCRATE_N2000_V292__n2000.json` | `b0c0ccfdaae2407639daf81fb0fa7a0d751c5f69ca58dcae129c8e3f33e92c73` |

## Verification

- **copy-only**: nothing under `data/runs/` was moved, renamed or deleted; the sources
  remain at `data/runs/v2.9.2/mppi/`.
- all **2** copied files SHA-256 verified against their sources after the copy;
  **0 mismatches**.
- no ledger row was created for this entry.

