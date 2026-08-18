# quad3d_oc_cgain0_2.9.2 — v2.9.2 c_gain-0 ablation cell (oc_pncbf)

A `06_workflow` §6.3 **experiments** entry: a Researcher-directed ablation kept for the record,
**not** a version SOTA snapshot. An experiments entry is **never eligible for ledger SOTA
bolding** (§2.5), and this entry is not bold.

## What this cell is

- Registered axis of v2.9.2: `env.quadrotor_3d.c_gain` **0.3 → 0.0**, the horizontal
  approach-term ablation. `changes.md` registers observable **A1 = `coll_obstacle`**.
- Derived from **L305**'s own persisted config (`data/runs/v2.9.1/set__20260813-164148__seed42/v2.9.1__oc__20260813-164148__seed42/config.yaml`)
  with the gated key set and nothing else.
- Run directory of record (not moved): `data/runs/v2.9.2/set__20260815-011200__seed42/v2.9.2__oc__20260815-011200__seed42`
- Scored on the **registered cell** by the same producer as L305–L312
  (`v282_agree_gate.gate_overrides`), pool `fullcb` sha8 `3682a4e3`, n 2000, ebs 2000, seed 42.

## Digests (this file is the only place they appear, §6.1)

| file | sha256 |
|---|---|
| `checkpoints/best.pt` | `a8233cf0e00a5691f31f9b544718fa696204e19049caae5453e09021db686712` |
| `checkpoints/final.pt` | `0ebeb93d7d04509571b940734dcac608525ef467e3e2d0717b943cd69ae98ff2` |
| `config.yaml` | `01597d4289bb7eb14222d882418602a67c7ac9ae46e4f6ad0809be38d690f9e5` |

## The row this entry backs

- condition **OC3D_CGAIN0_V292**, ckpt step **30000**
- `c_gain` of the scored checkpoint, read off its own embedded config: **0.0**
- reproduction gate over two independent scoring passes: **True**

| field | value |
|---|---:|
| `reach` | 0.86 |
| `collision` | 0.0765 |
| `oob` | 0.0 |
| `stuck` | 0.003 |
| `timeout` | 0.0605 |
| `infeasibility` | 0.1811251988621407 |
| `cps` | 0.6194124403413578 |
| `coll_obstacle` | 0.05 |
| `coll_band_lower` | 0.0265 |
| `coll_band_upper` | 0.0 |

## File set (`04_eval` §7.5, as produced)

- present: 9 of the named items; plus `figures/inloop/` with **268** per-eval figures
- not produced by this framework: figures/trajectory_grid_A.png, figures/trajectory_grid_B.png, figures/cbf_contour.png

The three top-level figures are not produced by the OC trainer; the amended §7.5 admits the
per-eval `figures/inloop/` tree in their place and the substitution is recorded here.

## Verification

- **copy-only**: nothing under `data/runs/` was moved, renamed or deleted.
- every one of the **277** copied files was SHA-256 verified against its source
  after the copy; **0 mismatches**.

