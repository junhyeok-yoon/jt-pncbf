# v2.6.1 — torque-box correction + velocity terminal, retrain the winning pipeline (build-log)

Facts, not verdicts; the version verdict is the Researcher's at close. Every number cited by disk path.
Reproduces the v2.6.0 winning pipeline (learned OC-PNCBF value → plain JT + terminal + detach, no shield)
on the CORRECTED plant + a velocity terminal. Two axes (changes.md §3, Researcher-reconfirmed): (A)
torque-box correction, (B) velocity-aware terminal. Recipe known — NOT a re-exploration (BC/DAgger and the
terminal-vs-alternatives search are closed in v2.6.0).

## Milestone 0 — version bump (06_workflow §2.1)

`v2.6.0 → v2.6.1`. Active-version literals bumped: `src/_version.py` (`__version__="v2.6.1"`),
`pyproject.toml` (`version="2.6.1"`), `src/configs/exp_config.yaml` (`run.version:"v2.6.1"`). The other 62
`2.6.0` occurrences across `src/`/`scripts/`/`tests/` are **provenance comments** (`v2.6.0: …` attributing
features to the version that introduced them) — correctly left unchanged; no test asserts the version
literal (grep of `tests/` = provenance comments only). Run-id derives from `__version__`
(`oc_pncbf/train.py:164,717` → `config['run']['version']`); verified printed `__version__ = v2.6.1`, dry
run-id `v2.6.1__20260715-XXXXXX__seed42`, pyproject 2.6.1, exp_config v2.6.1. `docs/versions/v2.6.1/` exists.

## Milestone 1 — two-axis config/code delta (changes.md §4)

**Axis A — torque box (config-only):**
- `exp_config.env.bounds.quadrotor_planar.tau_max` **0.2 → 1.0** (`exp_config.yaml:20`). The HardNet
  projection box reads this key, so the filter uses ±1.0 automatically. Rationale: plant-coherent
  τ̄=L_J·f_max/2≈0.98 (arm L_J=√(J/m)=0.1 m); the old 0.2 was ~1/5 of that (torque_box_plausibility.md).
- `exp_config.loss.policy.sat_excess_threshold` torque component **[19.62, 0.2] → [19.62, 1.0]**
  (`exp_config.yaml:193`; dependent — L_sat must track the box so it stays 0 within the legal box).

**Axis B — velocity terminal (config + code):**
- `exp_config.loss.policy.w_terminal_v` **NEW = 30.0** (`exp_config.yaml:181`; 0.0 = off, DI/unicycle parity).
- `src/frameworks/jt_pncbf/losses.py` `policy_bptt_loss` terminal now
  `task_cost += discount * (w_terminal*‖p_T−g‖ + w_terminal_v*‖v_T‖)` with `speed_T = system.speed(x)`,
  differentiable through x_T, inside the same `gate_in`, rides the committed `detach_filter_coeffs=true`.
  Guard widened to `if w_term > 0.0 or w_term_v > 0.0`. (byte-identical when both weights 0 → DI/unicycle
  parity preserved.)

**Unchanged (hygiene):** `c_gain=0.3`, `m/J/g/f_max`, `v_max=2.5`, `omega_max=4.0`,
`detach_filter_coeffs=true`, `w_terminal=30`, clamp_tanh head + z_target + output_gain, IC set, committed
pools. No `base_config.yaml` edit.

**Gates (green):**
- Two `tests/test_quadrotor_planar.py` box assertions hardcoded the old ±0.2 torque box
  (`test_hardnet_direct_input_asymmetric_box:182`, `test_clamp_tanh_head_reaches_box_boundary…:232`) — they
  read the live exp_config, so the corrected box (1.0) tripped them. Updated the plant-value constants
  0.2→1.0 (the tests still verify boundary-reached + asymmetric box; only the torque value tracks Axis A).
- Quadrotor `stage='smoke'` on the corrected plant: **OC** clean (run `v2.6.1__…170629`, halt None);
  **JT** clean (run `…170647`) — the new ‖v_T‖ terminal path is finite: `L_in_task=204.33`,
  `grad_norm_pi=129.87` (healthy), `grad_leak_VS=0.0`, no NaN/Inf; `w_terminal_v` off at 0 = DI/unicycle
  parity (verify.sh DI/unicycle smoke unaffected). Smoke dirs removed.
- verify.sh: **127 passed** (after the two box-assertion updates).

**Per-eval CBF contour (viz, user-requested).** The v2.6.0 auto-eval skipped the quadrotor CBF contour
(no 2D velocity slice); now `write_cbf_contour_figure` dispatches on `system.name` and, for
`quadrotor_planar`, renders a **position-plane contour across three approach-speed slices** (θ=0, ω=0,
`v=(0,s)`, `s∈{−2,0,+2}`) for the first two eval scenes — the same figure as the post-hoc
`quadrotor_cbf_contour.py`, now saved **every eval** to `figures/inloop/step_XXXXXX_cbf_contour.png` (and
logged to TensorBoard) by both trainers' `_record_eval`. New `plotting.plot_quadrotor_cbf_contour`; the
quadrotor branch is wrapped in try/except so a figure error never propagates into training; the DI/unicycle
2D-slice contour is unchanged. Integration confirmed on a quadrotor smoke (per-eval PNG written, 161 KB);
verify.sh 127. **Scope note:** the M2 (M1-value) run `…170907` was already running when this was added, so
its code was pre-loaded — it gets NO live per-eval contours (retrofit would require a forbidden restart);
the M5 JT run and all subsequent evals get them live. (Viz-only, not gate-relevant.)

## Milestone 2 — relearn V^{h★,π} on the corrected plant (OC-PNCBF)

Launcher `scripts/m1_oc_run.py --seed 42 --epochs 100` (config diff = only `training.oc_pncbf.epochs`
1000→100, the launcher reads the corrected exp_config as its baseline so tau_max=1.0 is used automatically).
Run dir `data/v2.6.1__20260715-170907__seed42`; n_steps 50000; `c_gain=0.3`, `eps_g=0.05`/`w=5.0` committed.

**Complete** (`status.json`: current_step 50000, phase **done**, halt_reason None, no NaN). Best filter-on
nominal in-loop cps **0.2182 @ step 42000** (`best.pt`) — vs v2.6.0 M1 best 0.185 (+0.033), an early sign
the corrected torque box helps even the pre-JT filter (collision band ~0.16–0.20 vs v2.6.0 ~0.19–0.27).
In-loop trajectory healthy/oscillating; ‖L_g V̂‖ logged throughout. (No per-eval CBF contour on this run —
the viz was added after M2 launched; §M0/M1 scope note.)

## Milestone 3 — M2/P1 gate on the new-plant V̂ — **PASS** (HALT gate cleared)

`scripts/analysis/quadrotor_m2_gate.py` on `best.pt` (step 42000), 344 true near-B₀ samples
({|V̂|<0.1}∩{|φ|<0.1}); `scratchpad/quadrotor_m2_gate.json`:

| barrier | median ‖L_g‖ | degen_frac (‖L_g‖<0.05) |
|---|---|---|
| exact φ (position-only) | 0.0000 | 0.904 |
| exact h_star (c=0.3) | 0.3000 | 0.023 |
| **learned V̂** | **16.72** (min 0.442, p5 0.900, p10 2.97) | **0.000** |

Gate: (1) learned degen_frac 0.000 < 0.10 (≪ φ 0.904) → PASS; (2) not starved (median 16.72 ≫ 0.072, p10
2.97 ≫ 0.0072) → PASS. **M2 PASS on the corrected plant** — the ±1.0 torque box did NOT break B₀ first-order
authority; `c_gain=0.3` holds (no re-tune needed). Proceed to M4/M5.

## Milestone 4 — M3 pre-JT filtered-nominal eval (v2.6.1 within-version baseline)

`scripts/analysis/quadrotor_m3_eval.py` on `best.pt` (step 42000): HardNet on the new-plant V̂, filter-ON,
NOMINAL policy, full pool `eval_full_quadrotor-planar_n2000_seed23456`; `scratchpad/quadrotor_m3_eval.json`.

| | cps_v2 [CI] | reach | coll | oob | timeout | inf_v2 |
|---|---|---|---|---|---|---|
| v2.6.1 M3 (corrected plant) | **0.1931** [0.1410, 0.2438] | 0.6670 | 0.1810 | 0.0285 | 0.1235 | 0.1196 |
| v2.6.0 M3 (old plant, ref) | 0.1738 [0.1221, 0.2263] | 0.6610 | 0.1850 | 0.0190 | 0.1350 | 0.1341 |

The corrected ±1.0 torque box lifts the pre-JT filter baseline: cps +0.019, timeout 0.135→0.124, inf
0.134→0.120, collision ~flat (0.185→0.181); oob rises slightly (0.019→0.029 — the larger box lets the
un-cotrained nominal fly marginally more aggressively). This is the **v2.6.1 within-version M6 comparison
baseline** (04_eval §5). Ledger row added (eval_only, learned filter, pre-JT). Proceed to M5 (JT).

## Milestone 5 — JT (plain joint training, winning config + velocity terminal) — RUNNING

`scripts/jt_terminal_run.py --seed 42 --steps 50000 --value-init <new M1 best.pt>`. Run dir
`data/v2.6.1__20260715-173808__seed42`; value warm-start new M1, policy FRESH; run config confirmed
`w_terminal=30`, `w_terminal_v=30`, `detach_filter_coeffs=true`, `tau_max=1.0`. Per-eval CBF contours live.

**Grad control (velocity-terminal explosion check — the changes.md §6 watch):** through step 4500, post-warmup
`grad_norm_pi` median **58**, max 2601 (a single warmup→policy transition spike, 1/14 samples), `grad_leak_VS=0`.
The velocity terminal does NOT explode the gradient — it rides `detach_filter_coeffs` as the position terminal
does. (Continues to be watched.)

**Early trajectory (in-loop, first post-warmup evals):**

| step | cps | reach | coll | oob | stuck | timeout | note |
|---|---|---|---|---|---|---|---|
| 1500 | −0.704 | 0.000 | 0.118 | 0.744 | 0.000 | 0.138 | pre-warmup (policy fresh) |
| 3000 | −0.839 | 0.016 | 0.202 | 0.006 | 0.052 | 0.724 | 1st post-warmup; OOB→timeout |
| 4500 | −0.987 | 0.062 | 0.168 | 0.000 | **0.590** | 0.180 | reach climbing (>v2.6.0 M6 0.026) BUT stuck spikes to 0.59 |

**Observation (fact, not verdict):** the velocity terminal (`w_terminal_v=30`) drives an early **stuck** phase
(0.59 at step 4500) — the policy learns to STOP (minimizing ‖v_T‖) before reaching the goal, converting the
v2.6.0 timeout/hover mode into mid-field stopping. Open question over the next evals: does stuck resolve as the
position terminal + in-window task cost pull the policy to slow AT the goal (H2 as intended), or does it
persist (velocity terminal over-strong → loiter becomes stuck, not reach)? Per scope ("do not fish"),
`w_terminal_v` is NOT tuned mid-run; the outcome is reported factually and flagged for Researcher if it persists.

**Stuck RESOLVED (step 6000):** reach jumps 0.062→**0.628**, stuck collapses 0.590→**0.07**, timeout already
**0.108** (< H2 target 0.163), coll 0.194, cps 0.079. The early stuck was a transient training phase — the
policy learned to slow AT the goal (H2 mechanism working), not mid-field. This is FAR ahead of v2.6.0 M6
(reach 0.050 at step 6000; it reached 0.61 only at step ~22500) — the corrected box + velocity terminal give
much faster convergence. Grad stays controlled through step 6000: median 60, p95 **1841 (<2000)**, max 2601
(the lone warmup spike, 5% >2000 — better than v2.6.0 M6's 3.7%@max7357), `grad_leak_VS=0`. Remaining watch:
collision (H1 needs <0.143; 0.194 at step 6000).

| 7500 (in-loop) | cps 0.3245 | reach 0.736 | coll 0.154 | timeout 0.084 | already > v2.6.0 M6 final (cps 0.292/reach 0.694/timeout 0.163); H2 crushed, H1 near (coll 0.154 vs 0.143 target) |
| 9000 (in-loop) | cps 0.3958 | reach 0.778 | coll 0.156 | timeout 0.054 | cps/reach climbing; timeout 0.054; coll ~0.155 (H1 pending) |
| 10500 (in-loop) | cps 0.4602 | reach 0.806 | coll 0.144 | timeout 0.046 | coll AT H1 target 0.143; timeout crushed; cps 0.46 (+0.17 vs v2.6.0 M6) |
| 12000 (in-loop) | cps 0.5550 | reach 0.834 | coll 0.110 | timeout 0.052 | **coll 0.110 < H1 0.143**; H1+H2 confirmed in-loop; cps 0.555 ~2x v2.6.0 M6 |

**Grad clean through step 12000** (velocity-terminal risk refuted): median 77, p95 **409** (< v2.6.0 M6 1513), max 2601 (lone warmup spike, 2.0% >2000), grad_leak_VS=0. The velocity terminal riding detach is BETTER-behaved than v2.6.0 M6. H1 (coll<0.143) + H2 (timeout<0.163) both confirmed IN-LOOP by step 12000; full-pool M6 confirmation pending run completion.
| 13500 (in-loop) | cps 0.4038 | reach 0.780 | coll 0.154 | timeout 0.062 | oscillation pullback from 12k peak (0.555); best cps 0.555@12000 |
| 15000 (in-loop) | cps 0.5291 | reach 0.818 | coll 0.112 | timeout 0.070 | back near peak; coll 0.112 < H1; strong ~0.40-0.55 cps band |
| 16500 (in-loop) | cps 0.4720 | reach 0.802 | coll 0.130 | timeout 0.066 | steady strong band; coll 0.130 < H1 |
| 18000 (in-loop) | **cps 0.5842** | reach 0.840 | coll 0.098 | timeout 0.060 | NEW BEST; coll 0.098 lowest; cps 2x v2.6.0 M6 |
| 19500 (in-loop) | cps 0.5353 | reach 0.822 | coll 0.112 | timeout 0.064 | steady near best |
| 21000 (in-loop) | cps 0.2626 | reach 0.748 | coll 0.212 | timeout 0.038 | downswing (coll spike, eval noise); best 0.584@18000 |
| 22500 (in-loop) | cps 0.3664 | reach 0.774 | coll 0.172 | timeout 0.054 | partial recovery; wide oscillation band |
| 24000 (in-loop) | cps 0.4881 | reach 0.802 | coll 0.126 | timeout 0.072 | back up; coll 0.126 < H1 |
| 25500 (in-loop) | cps 0.4808 | reach 0.814 | coll 0.138 | timeout 0.048 | steady; coll 0.138 |
| 27000 (in-loop) | cps 0.5033 | reach 0.820 | coll 0.130 | timeout 0.050 | steady strong |
| 28500 (in-loop) | cps 0.4256 | reach 0.794 | coll 0.152 | timeout 0.054 | mild oscillation |
| 30000 (in-loop) | cps 0.5098 | reach 0.806 | coll 0.112 | timeout 0.082 | steady; coll 0.112 |
| 31500 (in-loop) | cps 0.5652 | reach 0.826 | coll 0.096 | timeout 0.078 | near best; coll 0.096 |
| 33000 (in-loop) | cps 0.5118 | reach 0.808 | coll 0.112 | timeout 0.078 | steady |
| 34500 (in-loop) | cps 0.5352 | reach 0.806 | coll 0.098 | timeout 0.096 | steady; coll 0.098 |
| 36000 (in-loop) | cps 0.4557 | reach 0.794 | coll 0.134 | timeout 0.072 | steady band |
| 37500 (in-loop) | cps 0.5775 | reach 0.822 | coll 0.084 | timeout 0.092 | near best cps; coll 0.084 lowest |
| 39000 (in-loop) | cps 0.5009 | reach 0.792 | coll 0.104 | timeout 0.104 | steady |
| 40500 (in-loop) | cps 0.4606 | reach 0.778 | coll 0.114 | timeout 0.108 | steady (schedule sat @42k near) |
| 42000 (in-loop) | cps 0.5765 | reach 0.814 | coll 0.072 | timeout 0.114 | schedule saturates; coll 0.072 new low |
| 43500 (in-loop) | cps 0.3043 | reach 0.730 | coll 0.162 | timeout 0.108 | post-sat downswing (noise) |
| 45000 (in-loop) | cps 0.5586 | reach 0.806 | coll 0.078 | timeout 0.116 | recovered; coll 0.078 |
| 46500 (in-loop) | **cps 0.6018** | reach 0.818 | coll 0.062 | timeout 0.120 | NEW BEST; cps clears 0.60; coll 0.062 lowest |
| 48000 (in-loop) | cps 0.3703 | reach 0.756 | coll 0.148 | timeout 0.096 | dip (noise); best 0.602@46500 |
| 49500 (in-loop) | cps 0.2827 | reach 0.748 | coll 0.194 | timeout 0.058 | tail downswing; best 0.602@46500 |
| 50000 (in-loop) | cps 0.5375 | reach 0.786 | coll 0.072 | timeout 0.142 | run end; best in-loop 0.602@46500 |

**M5 complete** (`status.json`: current_step 50000, phase **done**, halt_reason None, no NaN; best in-loop
cps **0.6018 @ step 46500**). Grad clean the whole run (velocity-terminal risk refuted). Per-eval CBF
contours saved to `figures/inloop/step_*_cbf_contour.png` (live viz).

## Milestone 6 — full-pool eval + P3 (seed 42)

**M6 full eval** (`best.pt` step 46500, frozen pool `eval_full_quadrotor-planar_n2000_seed23456`,
`eval_metrics.csv` mode=final):

| eval | cps_v2 [95% CI] | reach | collision [CI] | oob | stuck | timeout | inf_v2 |
|---|---|---|---|---|---|---|---|
| **v2.6.1 M6** | **0.5753 [0.5333, 0.6163]** | 0.8205 | **0.0810 [0.0690, 0.0940]** | 0.0 | 0.0 | 0.0985 | 0.1132 |
| v2.6.1 M3 (pre-JT) | 0.1931 [0.141, 0.244] | 0.6670 | 0.1810 | 0.0285 | 0.0 | 0.1235 | 0.1196 |
| v2.6.0 M6 (ref, old plant) | 0.2920 [0.240, 0.338] | 0.6935 | 0.1430 | 0.0 | 0.0005 | 0.1630 | 0.1117 |

Insertion variants: final_insertion_lqr −0.374, final_insertion_frozen 0.4723, final_insertion_live 0.5246.

**Hypotheses (full pool):**
- **H1 (torque box → collision < 0.143): CONFIRMED.** collision **0.0810** [0.069, 0.094] — the entire CI is
  below 0.143 (and below M3's 0.181). vs v2.6.0 M6 0.143: −43%.
- **H2 (velocity terminal → timeout < 0.163 & reach rises): CONFIRMED.** timeout **0.0985** < 0.163; reach
  0.8205 (> M3 0.667, > v2.6.0 M6 0.694).
- **Combined:** cps **0.5753** — vs the within-version M3 baseline 0.1931 = **+0.382** (CI-separated:
  M3-hi 0.244 < M6-lo 0.533); vs the v2.6.0 M6 reference 0.292 ≈ **2×** (control-bound differs — informative,
  not a pure supersession, per changes.md §5).

**P3 — collinear-residual gate** (`scripts/analysis/quadrotor_p3_collinear.py`, `scratchpad/quadrotor_p3.json`):
residual collisions **159 (7.95%)** (vs v2.6.0 M6's 288/14.4%); collinear `S_h` (|cos(∇φ,Re)|>0.9) frac
**0.000 (0/159)**; learned ‖L_g V̂‖ at collisions median **0.570** (degenerate 2.5%). **P3 CONFIRMED** — the
`S_h` brake-only failure does not transfer; authority retained; residual collisions down sharply.

**Seed-42 verdict-grade:** collision 0.081 < 0.143 with CI margin, pipeline healthy (grad clean, M2/P1 pass,
P3 confirmed) → escalate to canonical seeds {99, 12345} (seed-economy mandate, changes.md §5).

## Seed escalation — {99, 12345}

### Seed 99
- **M1** (relearn V): `data/v2.6.1__20260715-211111__seed99`, done, best filter-on cps 0.197@42000 (≈ seed42 0.218).
- **M2/P1 gate: PASS** — learned degen_frac 0.000 (< 0.10, ≪ φ 0.879), median ‖L_g V̂‖ 17.09, p10 1.72, not
  starved (`scratchpad/quadrotor_m2_gate.json`).
- **M3 pre-JT baseline**: cps_v2 0.2056 [0.151, 0.257], reach 0.6715, coll 0.1755, timeout 0.1140, inf 0.1281
  (≈ seed42 M3 0.1931). Within-version baseline for seed99.
- **M5 (JT)**: `data/v2.6.1__20260715-213730__seed99` — RUNNING (value warm-start seed99 M1, w_terminal=30 +
  w_terminal_v=30, detach=true, tau_max=1.0).
