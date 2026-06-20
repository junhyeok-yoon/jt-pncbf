# Adopted v2.2.2 Seed 42 — Unicycle collision-precursor injection (NEW-SYSTEM SOTA)

- System: `unicycle` (state `[x, y, theta, v]`, control `[a, omega]`).
- Role: **unicycle SOTA of record** — the best result on the unicycle, a new system in v2.2.2.
  The DI SOTA is **unchanged** (v2.0.1, `cps 0.8568` n2000); v2.2.2's DI work (injection base 0.8607)
  only ties it, so no DI run is secured for v2.2.2. This is the only v2.2.2 run secured.
- Source run id: `v2.2.2__20260619-083424__seed42`
- Source run dir (local history): `data/previous_runs/v2.2.2__20260619-083424__seed42/`
- Run-start git commit: `ae1dbadf9e8b34bda9e1de3942e797232eb30347` (`DIRTY`)
- Seed: `42`
- Best step: `28000` (in-loop best_cps 0.531899; n2000 re-selection true-best also @28000)
- `best.pt` SHA-256: `3e324c1c2a387d8c4779f07cf9fee9b3a5b95b9bf10c99e395dc9db0f0bde816`
- `final.pt` SHA-256: `c37aeaeb8c37d6b2626fa56af449907db7db5f1fe8e4f6a3579810a6c5f4b887`
- `best.pt` is the headline checkpoint (n2000 true-best @28000); `final.pt` is @42000 (past-peak).

## Final eval metrics (n2000, pool `eval_full_unicycle_n2000_seed23456`)

Reach-criterion note: the policy was **trained at `goal_speed_radius=0.30`**; `goal_speed_radius` is an
eval-time setting, so the **same `best.pt`** is scored at both criteria (re-rolled per criterion, since eval
terminates on goal-reach). Both are recorded so the criterion behind each number is unambiguous.

| criterion | cps | reach | collision | oob | stuck | timeout | infeasibility | sat_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.30 (trained) | 0.6243 | 0.8235 | 0.0030 | 0.0000 | 0.1165 | 0.0570 | 0.1606 | 0.1518 |
| 0.50 (relaxed) | 0.6720 | 0.8535 | 0.0030 | 0.0000 | 0.1150 | 0.0285 | 0.1543 | n/a |

CI (n2000, scene-bootstrap): @0.30 cps `[0.5934, 0.6560]`. Single-seed result (stated as such; no converged
multi-seed exists — seeds 12345/99 were stopped early at 26-27k and are not representative).

Reference (not secured): Step-1 unicycle baseline (no injection) n2000 @0.30 `cps 0.4981` `[0.4637, 0.5347]`
→ injection **+0.1262**, non-overlapping 95% CIs (statistically significant; navigation-driven:
reach 0.732→0.824, timeout 0.138→0.057). Baseline run id `v2.2.2__20260619-051204__seed42`
(in `data/previous_runs/`).
