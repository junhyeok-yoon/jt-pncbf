# l312_partition_2.9.2 — L312 re-scored on the registered cell — per-episode dump for the tilt partition

A `06_workflow` §6.3 **experiments** entry: kept for the record, **not** a version SOTA
snapshot. An experiments entry is **never eligible for ledger SOTA bolding** (§2.5).

Backs ledger row **L312**, whose `eval_source` carries the tilt partition these
arrays were computed from.

DIAGNOSTIC comparator for L317's spawn-tilt partition; it created no ledger row. The re-score passed a REPRODUCTION GATE first: all nine fields reproduced L312's own scoring artifact `data/runs/v2.9.1/launch/jtrow__quadrotor_3d.json` at exactly 0.000e+00 -- not the ledger's 4-dp cell -- before any partition was computed. Written to a NEW per-condition path (05_code 3); it overwrites no artifact any ledger row cites.

## The spawn-tilt partition carried on the row

`spawn_tilt = arccos(R33)` in degrees from each scene's **own stored quaternion** in the
pool, not from any rollout. **60° is the altitude-holding limit at TWR 2**, not a tuned
threshold. Every figure below is arithmetic on the per-episode arrays in this entry.

| block | n | reach | collision | `coll_obstacle` | `coll_band_lower` | shares sum |
|---|---:|---:|---:|---:|---:|---:|
| ALL | 2000 | 0.9535 | 0.0255 | 0.0075 | 0.0180 | 1.000000000000 |
| tilt ≤ 60° | 486 | 0.9753 | 0.0123 | 0.0103 | 0.0021 | 1.000000000000 |
| tilt > 60° | 1514 | 0.9465 | 0.0297 | 0.0066 | 0.0231 | 1.000000000000 |

## Per-episode arrays

`scene_idx`, `outcome`, `collision_cause`, `first_collision_step`, `d_min`,
`spawn_tilt_deg`, `n_steps` — 2000 entries each, index-aligned to the pool's own order.

## Digests (this file is the only place they appear, §6.1)

| file | sha256 |
|---|---|
| `perepisode__L312_PARTITION_V292.npz` | `91f090f9dc0647197cfe112d558141d63fc8922d50d453136a06e3666eaa3662` |
| `row__L312_PARTITION_V292__step9450.json` | `77a50f516cb28fd192e3d742fbeb42e341d5a0abb7ac691bce4952fe9e8149a9` |

## Verification

- **copy-only**: nothing under `data/runs/` was moved, renamed or deleted; the sources
  remain at `data/runs/v2.9.2/mppi/`.
- all **2** copied files SHA-256 verified against their sources after the copy;
  **0 mismatches**.
- no ledger row was created for this entry.

