# quad3d_jt_cgain0_2.9.2 — v2.9.2 c_gain-0 ablation cell (jt_pncbf)

A `06_workflow` §6.3 **experiments** entry: a Researcher-directed ablation kept for the record,
**not** a version SOTA snapshot. An experiments entry is **never eligible for ledger SOTA
bolding** (§2.5), and this entry is not bold.

## What this cell is

- Registered axis of v2.9.2: `env.quadrotor_3d.c_gain` **0.3 → 0.0**, the horizontal
  approach-term ablation. `changes.md` registers observable **A1 = `coll_obstacle`**.
- Derived from **L312**'s own persisted config (`data/runs/v2.9.1/set__20260814-030107__seed42/v2.9.1__jt__20260814-030107__seed42/config.yaml`)
  with the gated key set and nothing else.
- Run directory of record (not moved): `data/runs/v2.9.2/set__20260815-014509__seed42/v2.9.2__jt__20260815-014509__seed42`
- Scored on the **registered cell** by the same producer as L305–L312
  (`v282_agree_gate.gate_overrides`), pool `fullcb` sha8 `3682a4e3`, n 2000, ebs 2000, seed 42.

## Digests (this file is the only place they appear, §6.1)

| file | sha256 |
|---|---|
| `checkpoints/best.pt` | `ec035448396040809312ecf600d20bbf9a41eee6c6e463549ba6ada1d6c00e85` |
| `checkpoints/final.pt` | `ca48ad48cb39db0ed92a8600c87e16de3d06978e018f01412d43b8b799872687` |
| `config.yaml` | `bd64ed3408f2ef9dd1c6a60f1043a4535fd6b956c11215f3af0e5710de002897` |

## The row this entry backs

- condition **JT3D_CGAIN0_V292**, ckpt step **9300**
- `c_gain` of the scored checkpoint, read off its own embedded config: **0.0**
- reproduction gate over two independent scoring passes: **True**

| field | value |
|---|---:|
| `reach` | 0.947 |
| `collision` | 0.025 |
| `oob` | 0.0 |
| `stuck` | 0.002 |
| `timeout` | 0.026 |
| `infeasibility` | 0.108171362541911 |
| `cps` | 0.8495485912374265 |
| `coll_obstacle` | 0.009 |
| `coll_band_lower` | 0.016 |
| `coll_band_upper` | 0.0 |

## File set (`04_eval` §7.5, as produced)

- present: 12 of the named items
- not produced by this framework: none

The three top-level figures are not produced by the OC trainer; the amended §7.5 admits the
per-eval `figures/inloop/` tree in their place and the substitution is recorded here.

## Verification

- **copy-only**: nothing under `data/runs/` was moved, renamed or deleted.
- every one of the **12** copied files was SHA-256 verified against its source
  after the copy; **0 mismatches**.

