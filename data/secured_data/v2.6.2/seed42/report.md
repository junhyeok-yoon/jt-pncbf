# v2.6.2 — situation-dependent velocity objective, retrain (build-log)

Facts, not verdicts; the version verdict is the Researcher's at close. Every number cited by disk path. One
mechanism — a situation-dependent velocity objective (running-cost redesign in `policy_bptt_loss`), two
spatially-gated components (goal settling + obstacle approach), shipped full with a goal-only ablation
(changes.md §3). NO value / filter / buffer / structure change. Reproduces the v2.6.1 winning pipeline
(learned OC value → plain JT + terminal + detach, corrected plant tau_max=1.0) with the new running cost.

## Milestone 0 — version bump (06_workflow §2.1)

`v2.6.1 → v2.6.2`. Active-version literals bumped: `src/_version.py` (`v2.6.2`), `pyproject.toml` (`2.6.2`),
`exp_config.yaml run.version` (`v2.6.2`). The one `tests/` `2.6.1` hit is a provenance comment
(`test_quadrotor_planar.py:229`), left. Verified printed `__version__ = v2.6.2`, dry run-id
`v2.6.2__20260716-XXXXXX__seed42`, pyproject 2.6.2, exp_config v2.6.2. `docs/versions/v2.6.2/` created.

## Milestone 1 — running-cost redesign (changes.md §4)

All in `src/frameworks/jt_pncbf/losses.py` `policy_bptt_loss` + `loss.policy` config. NO value/filter/buffer
change. **QUADROTOR-scoped** (gated `use_situational = system.name=='quadrotor_planar' and any new key set`)
because the obstacle-approach term needs the linear-velocity VECTOR `x[:,3:5]`, whose layout is
system-specific; **DI/unicycle take the legacy quadratic `d2` branch → byte-identical parity** (structural,
not weight-dependent). New running cost (per step, quadrotor), replacing `d2 + lambda_v·v2 + mu_u·u2`:

- **goal-distance Huber** (non-vanishing gradient): `w_goal · huber(‖p−g‖; huber_delta)`,
  `huber(r;δ)=0.5r² if r≤δ else δ(r−0.5δ)`. Keys `w_goal=1.0`, `huber_delta=0.15` (=goal_radius). `w_goal`
  absent → legacy `d2`.
- **dense goal-gated settling** (ADD): `w_settle · exp(−‖p−g‖²/settle_rho²) · ‖v‖²`. Keys `w_settle=1.0`,
  `settle_rho=0.30` (=2·goal_radius). `w_settle=0` → inert (goal-only ablation keeps this ON).
- **dense obstacle-gated approach** (ADD): `w_appr · Σ_k exp(−surf_k²/appr_d0²)·relu(−v·n̂_k)²` over the
  top-K obstacles (n̂_k = outward surface normal, surf_k = surface distance), **vectorized over K on-GPU, no
  Python obstacle loop** (obstacle geometry fetched once via `scene_obstacle_tensors`). Keys `w_appr=1.0`,
  `appr_d0=0.5`. `w_appr=0` → inert (the goal-only ablation arm).
- Retained `lambda_v=0.01`, `mu_u=0.001`, `w_terminal=30`. **`w_terminal_v` 30→0** (settling moved from the
  sparse ‖v_T‖ terminal to the dense goal-gated term — part of the mechanism).

**Gates:**
- Quadrotor `stage='smoke'`: policy active `L_in_task=101.9` (finite; < v2.6.1's 204 — Huber goal cost is
  smaller than quadratic+terminal), `grad_norm_pi=94.97`, `grad_leak_VS=0.0`, no NaN/Inf; config keys loaded
  (`w_terminal_v=0`, w_goal/w_settle/w_appr on). DI parity structural (legacy branch).
- verify.sh: **127 passed** (DI/unicycle parity confirmed — legacy branch, no test change needed).

## Milestone 2 — relearn V^{h★,π} — **REUSED v2.6.1 M1** (value path unchanged)

The M1 OC value path rolls the FIXED nominal LQR and regresses V^{h★,π_nominal} — it is unaffected by the
policy-loss redesign (M1 changed only `loss.policy`). The value-path config (env `tau_max=1.0`, `c_gain=0.3`,
`lg_authority eps_g=0.05`, lqr, `training.oc_pncbf`) is byte-identical to v2.6.1 → the M1 V̂ is identical.
**Reused** `data/previous_runs/v2.6.1__20260715-170907__seed42/checkpoints/best.pt` (step 42000, sha256
**1099eaf3b430c05e1efd81cbf358b9b4ef0401766cd297b050909664568c4696** = the v2.6.1 ADOPTED.md pin). M1 NOT
retrained (registered budget unaffected).

## Milestone 3 — M2/P1 gate + pre-JT baseline — **CITED from v2.6.1** (value path unchanged)

Since the value path is byte-identical, the gate/baseline are the v2.6.1 values (not recomputed):
- **M2/P1 gate: PASS** — on the reused M1 V̂ best.pt@42000: learned degen_frac **0.000** (< 0.10, ≪ φ 0.904),
  median ‖L_g V̂‖ **16.72**, p10 2.97, not starved (v2.6.1 `phase_v261_report.md §M3`).
- **M3 pre-JT baseline** (v2.6.2 within-version comparison anchor, same as v2.6.1's): cps_v2 **0.1931**
  [0.141, 0.244], reach 0.6670, collision 0.1810, timeout 0.1235, inf_v2 0.1196 (`ledger.md:107`,
  eval_only learned filter pre-JT). No new ledger row (identical value path/eval).

## AMENDMENT (2026-07-16, Strategist re-audit) — revert the Huber goal-distance; keep quadratic

**Why:** the Huber goal-distance component was mis-motivated — no smooth cost has a nonvanishing gradient at
its minimizer, and as specified (w_goal=1.0, huber_delta=0.15) it *weakened* the far-field goal gradient
~40× vs the original `2‖p−g‖`. Both ablation arms carried it, so a Huber-induced regression would be
unattributable. The v2.6.2 mechanism is the two gated SPEED terms; the distance form stays **quadratic**.

**Aborted stub (kept for provenance, NOT a result):** M4 run `data/v2.6.2__20260716-111140__seed42` was
aborted at **step 3699** (pre-warmup only; best_cps −0.704 @ step 1500 = the fresh-policy baseline, no policy
training yet). Dir retained.

**Revert diff (keys):**
- `losses.py` `policy_bptt_loss`: goal-distance term Huber → original `d2 = Σ(p−g)²` (the settling gate still
  computes `r=‖p−g‖`). `use_situational` now triggers on `w_settle>0 or w_appr>0` (dropped the `w_goal`
  condition).
- `exp_config.loss.policy`: **removed** `w_goal` and `huber_delta`. **Unchanged:** `w_settle=1.0`,
  `settle_rho=0.30`, `w_appr=1.0`, `appr_d0=0.5`, `w_terminal_v=0.0`, and everything else in changes.md §4.

**Gates (post-revert):** quadrotor smoke clean — `w_goal` absent from config, quadratic goal restored,
`L_in_task=163.4` (> the Huber 101.9, as expected — quadratic goal is larger), `grad_norm_pi=96.3`,
`grad_leak_VS=0.0`, no NaN; policy active. verify.sh: **127 passed**. DI/unicycle parity structural
(legacy branch, unchanged). changes.md stays as-opened (amendment lives here, per the v2.6.0 precedent).

**Grad note:** the elevated pre-clip `grad_norm_pi` observed in the aborted M4 (median ~1.4k vs v2.6.1 ~77)
was from the new dense gated terms, NOT Huber (Huber *lowers* the distance gradient); after the revert it may
sit higher still (quadratic far-field gradient restored + gated terms). Watch p95/max for explosion SCALE
only; weights are NOT tuned mid-run.

**Doc annotation** (append-only) added to `docs/versions/v2.6.1/predictive_filter_eval.md`: the tested
lookahead inflates alpha → on the safe interior (h<0, RHS = alpha·|h|) inflation LOOSENS the brake; the null
refutes the inflation variant only, the deflation variant is untested.

## Milestone 4 — JT FULL objective {goal(quadratic) + settling + obstacle-approach} — RUNNING

`scripts/jt_terminal_run.py --seed 42 --steps 50000 --value-init <reused M1>`. Run dir
`data/v2.6.2__20260716-112517__seed42` (config: quadratic goal, w_settle=1, w_appr=1, w_terminal_v=0,
detach=true, tau_max=1.0). Grad scale post-warmup (dense gated terms): median ~1235, max ~4314 through step
3000, `grad_leak_VS=0` — elevated vs v2.6.1 (~77) as expected (dense terms), no explosion scale.

**Early in-loop (first post-warmup evals):**

| step | cps | reach | coll | stuck | timeout | note |
|---|---|---|---|---|---|---|
| 1500 | −0.704 | 0.000 | 0.118 | 0.000 | 0.138 | pre-warmup (fresh policy) |
| 3000 | −0.931 | 0.010 | 0.266 | 0.012 | 0.690 | 1st post-warmup |
| 4500 | −0.718 | 0.046 | 0.162 | 0.030 | 0.762 | reach climbing |
| 6000 | −0.170 | 0.452 | 0.184 | 0.078 | 0.286 | breakthrough (≈ v2.6.1 M5 0.514@6000) |

**Early-stuck controlled:** stuck stays ≤0.078 (vs v2.6.1 M5's transient 0.59 at step 4500) — the dense
goal-gated settling replaced the sparse ‖v_T‖ terminal's blunt mid-field stopping (changes.md §6 risk not
realized).

**M4 complete** (`status.json`: phase done, best in-loop cps **0.6158 @ step 31500**, no halt). Grad
controlled the whole run (dense-terms scale, no explosion); early-stuck ≤0.078 (settling-gate fix held).

## Milestone 6 (part 1) — M4 FULL full-pool eval + P3

**M6 FULL eval** (best.pt@31500, frozen pool n2000/seed23456, `eval_metrics.csv` mode=final):

| eval | cps_v2 [95% CI] | reach | collision [CI] | timeout | inf_v2 |
|---|---|---|---|---|---|
| **v2.6.2 M4 FULL** | **0.6097 [0.5670, 0.6515]** | 0.8535 | **0.0955 [0.0830, 0.1085]** | **0.0510** | 0.0909 |
| v2.6.1 M6 (ref, same plant) | 0.5753 [0.5333, 0.6163] | 0.8205 | 0.0810 [0.069, 0.094] | 0.0985 | 0.1132 |
| Δ (FULL − v2.6.1) | **+0.0344** (CI-overlapping) | +0.0330 | **+0.0145** | **−0.0475 (−48%)** | −0.0223 |

Insertion variants: lqr −0.379, frozen 0.5383, live 0.5545.

- **H-goal (settling) — CONFIRMED (vs v2.6.1):** timeout **0.0985 → 0.0510 (−48%)**, reach **0.8205 →
  0.8535**. The dense goal-gated settling cut the near-goal loiter far more than v2.6.1's sparse ‖v_T‖
  terminal, AND the early-stuck did not worsen (≤0.078 vs v2.6.1's transient 0.59) — changes.md §6 held.
- **Collision ROSE vs v2.6.1** (0.0810 → 0.0955, +18%; P3 residuals 159 → 192). The obstacle-approach term
  (w_appr=1) did NOT reduce collision below v2.6.1 at the full-run level. Whether it HELPS relative to
  goal-only is the M5 ablation (H-obs = FULL − goal-only), pending.
- **cps rose** to 0.6097 (> v2.6.1 0.5753, +0.034) — driven by the timeout cut, net of the collision rise
  (arithmetic: +0.033 reach − 2·0.0145 coll − 0.5·(−0.0475) timeout ≈ +0.028). **CI overlaps v2.6.1**
  (v2.6.1-hi 0.616 vs FULL-lo 0.567) — not CI-separated.

**P3 (FULL best.pt)** (`scratchpad/quadrotor_p3_v262full.json`): residual collisions **192 (9.60%)**,
collinear `S_h` frac **0.000 (0/192)**, learned ‖L_g V̂‖ median **0.395** (degen 6.2%) → **P3 CONFIRMED** —
`S_h` does not transfer; certificate sound; the higher collision is NOT a value failure.

## Milestone 5 — JT GOAL-ONLY ablation (w_appr=0) — RUNNING

`scripts/jt_goalonly_run.py --seed 42 --steps 50000 --value-init <reused M1>` (patches `w_appr=0`, settling
ON). Run dir `data/v2.6.2__20260716-152305__seed42` (config confirmed w_appr=0.0, w_settle=1.0). Isolates the
obstacle-approach component: H-obs = (goal-only collision) − (FULL collision).

## AMENDMENT 2 (2026-07-16) — braking-envelope recalibration of the obstacle-approach term

Still v2.6.2 (NO version bump — promotion is the Researcher's). changes.md stays as-opened; the amendment
lives here (v2.6.0 precedent).

**A1 — M5 goal-only ABORTED** (Researcher-directed): run `data/v2.6.2__20260716-152305__seed42` killed at
**step 40568** (best in-loop cps 0.5439 @ step 37500; dir kept, NOT registered). The partial trajectory
tracked the FULL run closely (goal-only ≈ FULL in-loop throughout) — consistent with the Phase-1 prediction
that the gaussian approach term is inert.

**A2 — D4-lite** (eval_only(diagnostic), non-registered): full-pool eval of the goal-only best (@37500) vs
the FULL run's step_033000:

| ckpt | cps_v2 [CI] | reach | collision | timeout |
|---|---|---|---|---|
| goal-only best (w_appr=0) @37500 | 0.5790 [0.535, 0.619] | 0.8210 | **0.0825** | 0.0965 |
| FULL (w_appr=1) step_033000 | 0.5756 [0.529, 0.619] | 0.8375 | **0.1015** | 0.0610 |

cps CIs fully overlap; **goal-only collision (0.0825 ≈ v2.6.1's 0.0810) is LOWER than FULL (0.1015)** — the
gaussian approach term does NOT reduce collision (if anything mildly counterproductive). **D4 → the appr term
is inert-to-harmful as calibrated** (confirms Phase-1 B+C).

**B1 — H-appr (registered):** with a braking-envelope form that engages BEFORE the point-of-no-return
(earlier the faster the approach) at a weight competitive with the goal term, collision falls below the
v2.6.1 0.0810 on the full pool, while timeout stays ≤ ~0.06 and reach ≥ ~0.83 (settling untouched).
Falsified if collision ≥ 0.0955, or timeout/reach regress materially (over-braking). **Mechanism predictions
(checked at C2, required):** inward speed at the 1.0 m surface crossing drops below 1.66 (v2.6.2 FULL: 1.94,
v2.6.1: 1.66); the filter-infeasible fraction at residual collisions falls; envelope occupancy near obstacles
is substantial. **Grounds:** diag Phase-1 — (A) spillover refuted (v2.6.2 slower everywhere); (B) gate 0.018
at 1.0 m surface vs PNR ~0.9–1.0 m; (C) term p90 0.84 vs goal 30.4.

**B2 — form:** replace `w_appr·Σ_k exp(−dist_k²/appr_d0²)·relu(−v·n_k)²` with the braking-envelope deficit
`w_appr·Σ_k relu(s_k·tau_brake − dist_k)²`, `s_k = relu(−v·n_k)` (inward speed), `dist_k` = SURFACE distance
— exactly 0 when receding or outside the envelope, C¹, no division, engages EARLIER the FASTER the approach.
Config: `w_appr 1.0 → 30.0` (= the goal quadratic's near-obstacle p90 30.4; w_terminal=30 precedent);
REMOVE `appr_d0`; NEW `loss.policy.tau_brake = 0.6` s (PNR 0.35–0.4 s + margin; ≈ v_max/a_lat). Both
calibrated ONCE from the diagnostic; no mid-run tuning. Settling / quadratic goal / w_terminal / value /
filter / plant UNCHANGED.

**A3 — D5 checkpoint re-scan:** 5 FULL `step_*.pt` have collision < 0.0810 at cps within the best.pt CI
(steps 42000-50000, coll 0.065-0.075; step_050000 cps 0.6362 / coll 0.065 beats best.pt on BOTH). best.pt
selection STANDS (not re-selected). Detail: `collision_regression_diag.md` Phase 2. **A4 — verify.sh 127
passed** (Phase-1 deferred check; the situational-cost redesign is transparent to the DI-based policy-loss
tests — no test update needed).

**B2 gates:** quadrotor smoke clean (w_appr=30, tau_brake=0.6, appr_d0 removed; L_in_task=163.6,
grad_norm_pi=94.8, grad_leak_VS=0.0, no NaN). Braking-envelope unit sanity: deficit²=0 when receding (s=0),
0.49 approaching-fast-inside (s=2,surf=0.5,radius=1.2), 0 approaching-outside (s=1,surf=1.0,radius=0.6<1.0).
verify.sh **127 passed**. DI/unicycle parity structural (legacy branch).

## Phase C — JT retrain on the braking-envelope objective — RUNNING

`scripts/jt_terminal_run.py --seed 42 --steps 50000 --value-init <reused v2.6.1 M1, sha 1099eaf3…>`. Run dir
`data/v2.6.2__20260716-182949__seed42` (config: w_appr=30, tau_brake=0.6, w_settle=1, quadratic goal,
w_terminal=30, w_terminal_v=0, detach=true, tau_max=1.0). Grad SCALE (the brake-envelope is pathwise, no
filter-Jacobian coupling): one warmup-transition spike max **22516 @ step 2200** (first post-warmup step;
w_appr=30 makes it larger than prior runs), then median **159**, grads drop to ~1-4k, only 1/13 >10000;
`grad_leak_VS=0`. Not exploding. Early reach onset FAST (0.350 @ step 4500 vs the prior gaussian FULL's 0.046).

**C1 complete** (`status.json`: phase done, best in-loop cps **0.7341 @ step 30000**, no halt). Grad
controlled (one warmup spike 22516@2200, else median ~159); early over-braking stuck (0.144@6000) RESOLVED
by ~step 15000 (0.018).

## Milestone 6 (C2) — brake-envelope full-pool eval + P3 + mechanism checks

**M6 eval** (best.pt@30000, frozen pool n2000/seed23456, mode=final):

| eval | cps_v2 [95% CI] | reach | collision [CI] | timeout | inf_v2 |
|---|---|---|---|---|---|
| **v2.6.2 brake-envelope** | **0.7311 [0.6925, 0.7648]** | 0.9050 | **0.0670 [0.0565, 0.0780]** | **0.0250** | 0.0815 |
| v2.6.2 FULL (gaussian, prior) | 0.6097 [0.567, 0.652] | 0.8535 | 0.0955 | 0.0510 | 0.0909 |
| v2.6.1 M6 (ref, same plant) | 0.5753 [0.533, 0.616] | 0.8205 | 0.0810 [0.069, 0.094] | 0.0985 | 0.1132 |

Insertion: lqr −0.383, frozen 0.6649, live 0.6792.

- **H-appr (outcome) — CONFIRMED with margin.** collision **0.0670** [0.0565, 0.0780] < 0.0810 (CI entirely
  below; vs v2.6.1 −17%, vs the gaussian FULL 0.0955 −30% = the regression FIXED); timeout **0.0250** ≤ ~0.06;
  reach **0.9050** ≥ ~0.83. All three B1 falsification conditions passed.
- **cps 0.7311** — vs v2.6.1 0.5753 = **+0.156, CI-SEPARATED** (v2.6.1-hi 0.616 < brake-lo 0.693); vs the
  gaussian FULL 0.6097 = +0.121. Same plant → directly comparable.
- **P3 CONFIRMED** (best.pt@30000, `scratchpad/quadrotor_p3.json`): residual collisions **123**, collinear
  `S_h` frac 0.000, learned ‖L_g V̂‖ median 0.453 (degen 8.9%) — certificate sound.

**Mechanism checks (B1 required, honest — MIXED):**
| check | brake | v2.6.2 FULL | v2.6.1 | verdict |
|---|---|---|---|---|
| 1. inward speed @ 1.0 m surface crossing (p50) | **1.785** | 1.94 | 1.66 | PARTIAL — reduced vs the broken gaussian (1.94→1.785), NOT below v2.6.1's 1.66 |
| 2. filter-infeasible frac at residual collisions | 1.000 (123/123) | — | 1.000 (159/159) | per-collision UNCHANGED (100%); absolute count fell 159→123 (−23%); pool inf 0.113→0.082 |
| 3. envelope occupancy near obstacles (surf<1.0) | **0.124** | — | 0.143 (counterfactual) | term ON at ~12% of near-obstacle steps; below the v2.6.1 counterfactual → policy stays out of the violation region |

**Mechanism reading (fact):** the brake-envelope cut collision by reducing the NUMBER of fast-approach
collisions (residual set 165→130) + overall/near-obstacle speed (overall p50 2.117 vs FULL 2.27), NOT by
making the residual collisions feasible or dramatically slower at 1 m — the ones that still collide remain
the same 100%-infeasible fast-approach mode. So the OUTCOME hypothesis (H-appr) is confirmed with margin, the
MECHANISM predictions are partially met (fewer commits, not feasible-residuals). tau_brake/w_appr set ONCE
from the diagnostic; not tuned mid-run (a partial mechanism match is a reported result, not a knob).

## Residual-collision anatomy (D1) + lookahead-DEFLATION eval (D2)

Read-only, eval-only, on the brake-envelope best.pt@30000, full pool n2000/seed23456, dual_arm re-roll.
Baseline OFF reproduces M6 (coll 0.0615 vs 0.067, cps 0.7365 vs 0.7311; n_coll 123). `_lookahead_peak_h`
confirmed runnable with the deployed policy (policy_fn(x,scene), `filter_hardnet.py:334`); `delta=0.1`
(`exp_config.yaml:252`). Script `scripts/analysis/quadrotor_deflation_eval.py`, dump
`scratchpad/quadrotor_deflation.json`.

### D1 — residual-collision anatomy (n=123; born-doomed vs rescuable)

> **RETRO-ANNOTATION (§doom census, 2026-07-17):** the "born-doomed" label here is METHOD-RELATIVE — it means
> V̂-infeasibility (empty admissible set) OR brake-envelope-deficit at t=0, NOT provable doom. The doom census
> supersedes it: under the sound 3g relaxed system, ZERO of these ICs (0/72) is provably doomed (ballistic +
> 4D-HJ agree). These are extreme ICs the learned filter/policy handles sub-optimally, i.e. a training target —
> not excludable born-doomed ICs. Read the counts below as "V̂-infeasible-at-birth", not "inevitable collision".

**Three-way split (sums to 123 ✓):** BORN-DOOMED (infeasible OR envelope-deficit>0 at t=0) **72 (58.5%)** ·
EARLY-TRAPPED (feasible <1 s then lost) **41 (33.3%)** · LATE-TRAPPED (feasible ≥1 s then lost) **10 (8.1%)**.
Birth: **53.7%** infeasible-at-t=0, **35.0%** deficit>0-at-t=0; median feasible-prefix **0 steps** — the median
residual collision is infeasible from birth. → **the majority of residual collisions are born-doomed IC
artifacts, unrecoverable at deploy time.**

**IC cross-tab (collision class by IC band):**
| \|θ₀\| band | n | born-doomed | early | late |
|---|---|---|---|---|
| [0, π/6) (near-nominal) | 12 | 2 | 7 | 3 |
| [π/6, π/2) | 27 | 12 | 11 | 4 |
| [π/2, π] (tilted) | 84 | **58 (69%)** | 23 | 3 |
| inward v₀ ≥ 0.75 | 35 | **29 (83%)** | 5 | 1 |
| inward v₀ receding~0 | 51 | 17 | 27 | 7 |

**Deployment read:** born-doomed collisions concentrate in EXTREME ICs — 58/84 of the tilted band
(\|θ₀\|>π/2) and 29/35 of the fast-inward band (v₀≥0.75) are born-doomed. The **near-nominal band
(\|θ₀\|<π/6) has only 12/123 collisions (10%)**, mostly rescuable (7 early + 3 late, 2 born). So on the
realistic-attitude sub-distribution the residual collision is small and rescuable; the count is dominated by
extreme initial states.

### D2 — lookahead-DEFLATION sweep (conservative variant; eval_only, non-registered)

Deflate the SAFE branch only: `alpha_eff = alpha_safe/(1+beta·gap)` on h≤0, `gap=relu((h_peak−h)/delta)`,
h_peak from `_lookahead_peak_h` (deployed policy); alpha_unsafe UNCHANGED; delta=0.1. Runtime override only.

| cell | collision | timeout | cps_v2 | empty_inf | ‖du‖ | Δcoll |
|---|---|---|---|---|---|---|
| **OFF** | 0.0615 | 0.0300 | 0.7365 | 0.0800 | 1.437 | — |
| n3 β1 | 0.0605 | 0.0395 | 0.7297 | 0.0784 | 1.528 | −0.0010 |
| n3 β3 | 0.0580 | 0.0500 | 0.7171 | 0.0815 | 1.708 | −0.0035 |
| n8 β1 | 0.0615 | 0.0405 | 0.7242 | 0.0795 | 1.597 | +0.0000 |
| n8 β3 | 0.0600 | 0.0705 | 0.6738 | 0.0857 | 1.871 | −0.0015 |

**Per-class conversion (baseline collisions → goal/timeout under the cell):**
| cell | early-trapped rescued | late-trapped rescued | born-doomed moved |
|---|---|---|---|
| n3 β1 | 23/41 | 6/10 | 7/72 |
| n3 β3 | 21/41 | 9/10 | 9/72 |
| n8 β1 | 25/41 | 8/10 | 7/72 |
| n8 β3 | 24/41 | 8/10 | 7/72 |

**Deflation is NOT a collision lever (fact).** Every cell's Δcollision is ≤ 0.0035 (within eval noise); ALL
raise timeout (0.030→0.040–0.071) and lower cps (0.7365→0.674–0.730). The per-class conversion shows the
mechanism: deflation DOES rescue the rescuable set (early 21–25/41 ≈ 55–60%, late 6–9/10) — but the aggregate
collision is unchanged because the more-conservative filter **reshuffles** (creates ~as-many new collisions
among previously-safe episodes as it rescues), trading collision for timeout. **Born-doomed unmoved**
(7–9/72 ≈ 10%) — confirms the D1 classification (they are unrecoverable at deploy time; if they moved it would
falsify it). Registered as eval_only(deflation=n/β), NOT SOTA-eligible; flag Researcher.

### Verdict inputs (facts)

Of the 123 residual collisions: **58.5% born-doomed** (IC artifacts, concentrated in \|θ₀\|>π/2 + fast-inward
ICs — 90% of them; median infeasible-from-birth), **41.5% early/late-trapped** (rescuable in principle).
The conservative deflation variant CAN convert the rescuable set but only by reshuffling (net collision
unchanged, timeout↑, cps↓) — **deflation is not a usable collision lever**. The residual collision is
dominated by the extreme-IC born-doomed set, not a deploy-time-fixable filter deficiency; the near-nominal
attitude band already collides at only 10% of the residual.

## AMENDMENT 3 (2026-07-16) — attitude-aware IC recoverability filter: implemented, VALIDATION FAILED → STOP

Quadrotor only; no retrain / version bump / git. **PROTOCOL FOLLOW-UP: 01_env scene-init recoverability test
(quadrotor) + pool lineage (_recov1).**

**A — criterion (implemented + unit-tested).** `src/common/quadrotor_recoverability.py` `is_recoverable`:
reject an IC iff for ANY active obstacle `d_k < s_k·t_align + 0.5·a_adv·t_align² + (s_k+a_adv·t_align)²/(2·a_brake)
+ margin`, plant-constants only — `s_k=relu(−v₀·n̂_k)`, `d_k`=surface distance, `n̂_k`=outward normal;
`t_align`=min-time bang-bang rotation of `Re(θ₀)` onto `n̂_k` under `α_max=τ_max/J=100`, `ω_max=4.0`, from
`ω₀` (signed: helps if toward the target, adverse-reversal cost if away); `a_adv=g·max(0,n̂_{k,y})`;
`a_brake=f_max/m−g=9.81`. Wired ADDITIVELY into `scene_init._passes_recoverability_filter` (quadrotor only,
no-op if `env.quadrotor_planar.recov_margin` absent). Unit test
`test_recoverability_rejects_flipped_fast_inward_accepts_hover_far` PASSES (π-flip fast-inward near obstacle
→ reject; hover far → accept; receding → accept; adverse-ω costs > helping-ω; deterministic). verify.sh green.

**B — validation vs the D1 sets (re-roll of brake best.pt: 72 born-doomed collisions, 400 reach ICs).**
Required: TP (born-doomed flagged) ≥ 0.80 AND FP (reach flagged) ≤ 0.02.

| margin | TP (born flagged) | FP (reach flagged) | fresh-sampling reject | passes? |
|---|---|---|---|---|
| 0.0 | 0.792 | **0.217** | 0.279 | NO (TP<0.80, FP≫0.02) |
| 0.1 | 0.847 | **0.250** | 0.303 | NO (FP≫0.02) |
| 0.2 | 0.889 | **0.275** | 0.329 | NO (FP≫0.02) |

**NO margin passes → STOP (per the execute's stop condition; criterion NOT tuned beyond the grid).** The
test HAS signal (born-doomed flagged ~3× the reach rate) but the FALSE-POSITIVE floor (0.217 at margin 0) is
~10× the required 0.02: the analytic worst-case head-on-braking criterion flags ~1/4 of REACH ICs as doomed
because it ignores the learned policy's LATERAL avoidance (steering around a finite obstacle without head-on
deceleration) and its sub-worst-case execution. The criterion cannot separate born-doomed from reach at
FP≤0.02.

**Consequence:** C (pool regeneration) and D (re-eval battery) NOT run — a filter that rejects 25% of
reach-capable ICs would distort the pool, not clean it. `env.quadrotor_planar.recov_margin` is NOT set → the
scene_init filter stays INERT (no behavior change to existing sampling/pools; old pools untouched). The
criterion code + unit test are kept as a documented tool. **The born-doomed set is real (D1: 58.5%,
concentrated in |θ₀|>π/2 + fast-inward), but a purely-analytic worst-case recoverability test is too
conservative to filter them without over-rejecting recoverable ICs — a learned or lateral-avoidance-aware
recoverability model would be needed. Facts; the next-step verdict is the Strategist's.**

## CONTRADICTION LOG (2026-07-16) — recov1 ledger task cited a battery that was never run

A follow-up ledger task instructed appending five `pool=recov1` rows (brake-envelope 0.8272, gaussian-FULL
0.744, v2.6.1 0.700, v2.6.1-M3 0.318, LQR-only −0.318) "from the amendment-3 battery." **No such battery
exists:** amendment 3 STOPPED at B (recoverability criterion FP **0.217/0.250/0.275** at margins
{0.0,0.1,0.2}, all ≫ the required 0.02 — no margin passed), so per its stop-condition C (pool regeneration)
and D (re-eval battery) were NOT run. Disk-verified: no `_recov1` pool files, no battery table in this report
(the §amendment-3 "Consequence" line records C/D not run), no battery scratchpad (only
`quadrotor_recov_validate.json` = the failed validation). The five cited cps numbers appear in NO artifact.
**No ledger row was written from them (fabrication avoided).** Resolution (user-directed, Option 1 + ledger
hygiene): keep the STOP; annotate ALL 8 quadrotor_planar ledger rows as "POOL UNDER REVISION … historical
record, NOT a current verdict baseline; current comparability class pending a validated recoverability
criterion. flag Researcher." and remove all SOTA-bold from quadrotor rows (no current baseline until a
validated recov pool exists). No number deleted/renumbered/edited; DI/unicycle rows untouched.

---

## §criterion FP anatomy — why the amendment-3 kinematic test flags REACHABLE ICs (read-only, eval-only)

Diagnostic `scripts/analysis/quadrotor_recov_fp_anatomy.py`; single brake-envelope re-roll of
`data/v2.6.2__20260716-182949__seed42/checkpoints/best.pt` (filter ON) on the full n2000 pool, criterion
at margin 0.0. Artifact `scratchpad/quadrotor_recov_fp_anatomy.json`. `is_recoverable`/`recoverability_detail`
unchanged; recov_margin stays absent (filter inert). Purpose: characterise the FP=0.21 over-firing so a
future criterion can be corrected — NOT to re-adopt the criterion.

### D1 — the sets @margin 0.0 (full reach set, not a sample)
born-doomed = 73 (collision & infeasible@0 or brake-deficit>0@0). reach = 1815.
- **TP** (born & flagged) = 58 → TPR 0.7945
- **FN** (born & unflagged) = 15
- **FP** (reach & flagged) = **382 → FPR 0.2105** (consistent with the B-validation 0.2175 on its 400-sample)
- **TN** (reach & unflagged) = 1433

### D2 — which term over-fires, and are FPs near-threshold?
Binding-obstacle decomposition (means). **The FP and TP inputs are nearly identical on every kinematic
variable EXCEPT surface distance d_k** — the criterion cannot separate reachable from doomed:

| set | d_k | s_k | v_perp | Δθ (rad) | t_align | align-adv | brake | align/d_req | viol margin (d_req−d_k) |
|-----|-----|-----|--------|----------|---------|-----------|-------|-------------|--------------------------|
| FP  | **1.34** | 0.75 | 0.60 | 2.20 | — | 1.67 | 1.33 | 0.61 | +1.66 (p50 1.21, p90 3.99) |
| TP  | **0.49** | 0.75 | 0.65 | 2.35 | — | 2.06 | 1.77 | 0.58 | +3.33 |
| TN  | 3.9-ish | low | — | — | — | 0.38 | 0.17 | — | −1.67 |

- **The align-advance term dominates d_req** (mean 0.61 of the requirement for FP; ≈0.58 for TP). The
  requirement is set mostly by "coast head-on through the whole bang-bang rotation", not by the physical
  brake distance.
- **FPs are NOT near-threshold** — the violation margin is DEEP (p50 +1.21 m, p90 +3.99 m of predicted
  penetration). No small margin trim recovers them.
- s_k, v_perp, Δθ are statistically the same for FP and TP; d_k differs (1.34 vs 0.49). A criterion whose
  only working discriminator is d_k, but whose d_req≈3.0 m is far above the d_k the policy actually needs,
  cannot be fixed by re-thresholding.

### D3 — how the 382 FPs actually escape (trajectory re-use, filter ON)
Escape classification: **tangential-evasion 254 (66.5%)**, closure-overestimate 80 (20.9%),
faster-align 46 (12.0%), other 2.
- **Lateral evasion is the dominant over-firing mode.** tang0 = |v_perp|/|v| @0 mean 0.57 (p50 0.59) — for
  two-thirds of FPs the *majority* of the speed the criterion books as head-on closure is actually
  tangential; the vehicle slides past the obstacle and never needs the full head-on brake.
- **Full alignment is not required.** never-aligned fraction 0.17 (17% of FPs never reach the <20° thrust-to-
  normal cone yet still reach goal); realised t_align/criterion t_align p50 0.78 (aligns faster than the
  bang-bang worst case, aided by ω0 and partial-redirect sufficiency).
- **Penetration is massively overpredicted.** actual closest approach d_min p50 0.33 m (mean 0.50, stays
  outside), vs criterion-predicted penetration; overestimate (actual − predicted) mean **+2.16 m** (p50 1.81).

Root cause: the "rotate fully to the normal with ZERO concurrent braking, then brake ALL of −v·n as head-on
closure" worst-case is structurally too conservative for an underactuated body that (a) carries lateral
momentum past the obstacle and (b) brakes partially while it rotates.

### D4 — candidate-relaxation ROC (report only; gate = FPR ≤ 0.02 at TPR ≥ 0.80)
| variant | TPR (born flagged) | FPR (reach flagged) |
|---------|--------------------|---------------------|
| original @0.0 | 0.795 | 0.211 |
| original @−0.1 | 0.753 | 0.193 |
| original @−0.2 | 0.726 | 0.180 |
| normal-closure-only (drop align-advance) @0.0 | 0.685 | **0.087** |
| effective-velocity κ=0.5 @0.0 | 0.753 | 0.186 |
| effective-velocity κ=1.0 @0.0 | 0.726 | 0.173 |
| simultaneous align+brake @0.0 | 0.000 | 0.000 (degenerate — never flags) |

**No variant meets the gate.** Dropping the align-advance term (normal-closure-only) halves the FPR to
0.087 — confirming align-advance as the primary over-firing term — but only by conceding TPR to 0.685, and
0.087 is still ≫ 0.02. Discounting closure by lateral speed (effective-velocity) moves the ROC only
marginally because it also weakens the true positives (FP and TP share the same v_perp≈0.6). Simultaneous
align+brake collapses to a no-op (brake-from-s_k with full authority always fits). Negative margins trade
TPR for FPR along the same unfavourable curve.

### Verdict
The over-firing is **structural, not a threshold miscalibration**: (1) the align-advance term (≈60% of d_req)
rests on a head-on-closure-during-full-rotation premise that ~2/3 of reachable ICs violate by lateral
evasion; (2) FP and TP are separable only on d_k, which the inflated d_req swamps. This closed criterion is
NOT adoptable as an IC filter and remains inert (recov_margin unset). A viable successor would need to model
lateral drift-past and concurrent braking (i.e. a 2-D reachability/backup-set test, not a 1-D head-on
kinematic bound) — deferred; not attempted this version. verify.sh green (128 passed).

---

## §HJ criterion — P0 feasibility probe: STOP (dense 6D grid exceeds hardware by 1-2 orders of magnitude)

Goal: replace the failed kinematic recoverability test (§criterion FP anatomy — structurally over-conservative,
FPR 0.21) with a physically rigorous single-obstacle Hamilton-Jacobi AVOID table as the doom filter. No
learned artifact enters the criterion. Probe script `scripts/analysis/hj_p0_probe.py` (sha eb137a8b58795a91),
`hj_reachability` 0.7.0 (pip, --no-deps) on JAX 0.9.2 / CUDA. Hardware: RTX 5080 16 GB VRAM (13.2 free),
host RAM 22 GB available, 24 CPU.

### Formulation (implemented, correct, but not runnable at the required resolution)
Relative state x_rel = [px_rel, py_rel, theta, vx, vy, omega] (6D); static obstacle at origin so ṗ_rel = v.
Gravity FIXED in world frame — breaks rotational symmetry, so the state is genuinely 6D (x-mirror symmetry is
a correctness check only, NOT a reduction; no translational reduction either). Control-affine
`ControlAndDisturbanceAffineDynamics`: drift f=[vx,vy,ω,0,−g,0], control Jacobian columns
[f_thr: (0,0,0,−sinθ/m,cosθ/m,0)], [τ: (0,0,0,0,0,1/J)]; control box f_thr∈[0,19.62], τ∈[−1,1].
Doom value V*(x;R)=sup_u inf_t (‖p_rel‖−R): control_mode='max' (control maximizes surface distance),
min-over-time via the `backwards_reachable_tube` Hamiltonian postprocessor min(H,0). V*<0 ⟺ even the best
control cannot keep the distance non-negative ⟺ doomed. (Convention verified against solver.py: line 16
`backwards_reachable_tube = lambda x: jnp.minimum(x, 0)` is a Hamiltonian postprocessor giving the avoid tube.)
Domain p_rel∈[−4.5,4.5]², θ periodic, |v|≤2.5, |ω|≤4.

### Measured cost (single-run, RTX 5080)
| N/dim | cells | value array | accuracy | peak GPU | overhead (peak/value) | per-step |
|-------|-------|-------------|----------|----------|-----------------------|----------|
| 13 | 4.8 M | 19 MB | low | 0.97 GB | **50×** | 96 ms |
| 15 | 11.4 M | 46 MB | low | 2.12 GB | 47× | 294 ms |
| 17 | 24.1 M | 97 MB | low | 5.10 GB | 53× | 886 ms |
| 21 | 85.8 M | 343 MB | low | **OOM** (>11 GB single alloc) | ~32×+ | — |
| 21 | 85.8 M | 343 MB | high (WENO3) | **OOM** (13.9 GB working set, XLA remat log) | 40× | — |

The 6D Lax-Friedrichs + upwind scheme materialises ~50× the value array at the leanest accuracy ("low" =
first-order upwind + first-order TVD-RK) and ≥100× at the rigorous "high" (WENO3) accuracy needed for a
tight conservative certificate. **hj_reachability OOMs the GPU already at N=21 — below the requested range.**

### Full-grid projection (memory ~ N⁶); requested full grid = 31–45/dim per radius bucket, ×~5 buckets
| N/dim | value array | dense solver (~50×, low-acc floor) | frugal in-place (~3× copies, theoretical) | verdict |
|-------|-------------|-----------------------------------|-------------------------------------------|---------|
| 21 | 0.34 GB | 17.2 GB | 1.0 GB | > GPU; host-only |
| 25 | 0.98 GB | 48.8 GB | 2.9 GB | over GPU |
| 27 | 1.55 GB | 77.5 GB | 4.6 GB | over GPU |
| **31** | 3.55 GB | **177.5 GB** | 10.7 GB | over GPU + over host (dense) |
| 35 | 7.35 GB | 367.7 GB | 22.1 GB | INFEASIBLE (host edge even frugal) |
| 41 | 19.0 GB | 950 GB | 57.0 GB | INFEASIBLE |
| 45 | 33.2 GB | 1661 GB | 99.6 GB | INFEASIBLE (value array alone > host RAM) |

Per-step time at the requested resolution (low accuracy, ×(N/13)⁶ from the 96 ms anchor): N=31 ≈ 17 s/step;
convergence T_hj≈3.5 s at CFL dt≈0.02 → ~175 steps → ~50 min/bucket, ~4 h for 5 buckets at N=31 low; "high"
accuracy ~5× longer and ~2× the memory. N=41/45 add another 5–20×.

### STOP (P0 gate)
The dense-grid HJ table at the **required 31–45/dim** exceeds the hardware budget by **1–2 orders of magnitude**
with the actual solver (178–1661 GB vs 16 GB GPU / 22 GB host). The task's P0 instruction is explicit: *"STOP
and report if the full grid exceeds the GPU/host budget — do NOT silently coarsen"* and *"STOP at any failed
gate — do not improvise past it."* Accordingly P1–P3 are NOT run. What was NOT done, deliberately: (a) no
coarsening to N≤21 (below the resolution the certificate needs; the requested range is the gate); (b) no
hand-written frugal 6D WENO solver (the ~3×-copy in-place floor reaches only N=31 on host RAM at multi-hour
cost, and writing it is exactly the improvisation the gate forbids); (c) no learned-artifact substitute.
The Curse of Dimensionality is intrinsic to dense 6D level-set methods — this is a hardware/method wall, not
a bug. Criterion remains UNVALIDATED; recov_margin stays unset (filter inert). verify.sh green (128 passed).

Decision surfaced to the user (not auto-chosen): options include (1) a decomposition/lower-D relaxation
(e.g. per-obstacle 4D leader-frame with conservative velocity coupling), (2) a sampling/trajopt-only doom
certificate without the grid, (3) provisioning a larger-RAM/GPU host for N=31 frugal solve, or (4) shelving
the HJ route and keeping the pools flagged [P1] "pending a validated criterion".

---

## §doom census — provable IC-doom in the RELAXED 3g system: EMPTY (exclusion line closes with evidence)

Resolution of the P0 decision (Researcher-directed): 6D dense HJ stays CLOSED (measured infeasible);
trajopt-only and bigger-host REJECTED; ADOPTED = the attitude-free RELAXED system as a DIAGNOSTIC census,
plus the ballistic closed form as the criterion-eligible artifact. NO pool change, NO training, NO git.
Modules: `src/common/quadrotor_ballistic_doom.py` (sha 4d178c8992930309); scripts
`quadrotor_doom_census.py` (sha 367d63b7649679ff), `quadrotor_doom_census_B.py` (sha adea02e1c58f6a83).

### Soundness frame (verbatim)
The true per-axis acceleration is (f/m)·Re(θ) − g·e_y with f∈[0,f_max], so the true acceleration set is
contained in the ball of radius f_max/m + g = **3g = 29.43** about 0. The RELAXED system — a double integrator
p̈ = u, ‖u‖ ≤ 3g, no attitude, no velocity clamp — is therefore AT LEAST AS CAPABLE as the plant. Hence
relaxed-system inevitable penetration for SOME single obstacle k ⟹ true doom in the full scene (capability
containment + constraints-only-shrink). Both tests below live in this relaxed system and are
UNDER-approximations of true doom (**zero false exclusion by construction**). What NEITHER can certify is doom
that exists only through the attitude/alignment cost — that boundary is stated, not papered over.

### A — ballistic closed form (criterion-eligible)
Reachable-set doom certificate: under ‖u‖≤A the position reachable at time t is the disc centered at the
ballistic point b(t)=p0+v0·t with radius ½A t². If some t has the whole reachable disc inside an obstacle
disc, every trajectory is inside at t ⟹ doom:  ∃t≥0: ‖b(t)−c‖ + ½A t² < R  (R rounded DOWN to shared
0.05 buckets; smaller obstacle = fewer flags = conservative). g(t)=‖b(t)−c‖+½A t² is convex → unique min by
ternary search on [0,√(2R/A)]. Head-on case (v0=−s_k·n) reduces analytically to **d_k < s_k²/(2A)** (normal
closure vs braking distance), with a lateral-escape allowance built in off-axis. Plant constants only; pure.
Unit test `test_ballistic_doom_flags_surface_head_on_spares_grazing` PASSES: surface-contact head-on →
flagged; tangential-grazing / ample-clearance / receding / hover-far → not flagged; closed form matches the
ternary min to 1e-5; buckets round down. **A2 flag rates (A=29.43, radius rounded down), NO pool change:**
| set | flagged | rate | note |
|-----|---------|------|------|
| (i) all pool ICs | **0 / 2000** | 0.0000 | provable-doom set EMPTY |
| (ii) born-doomed (D1) | **0 / 72** | 0.0000 | context only (V̂-based label, not ground truth) |
| (iii) reach-outcome | **0 / 1807** | 0.0000 | **GATE PASS** (a reach outcome is a physical avoidance witness) |
No IC is flagged, so there is no flagged-IC stratum to report. The reach-gate holds: zero derivation bug.

### B — 4D relaxed-system HJ census (diagnostic; same soundness class, exact relaxed solve)
State [px_rel,py_rel,vx,vy]; ṗ=v, v̇=u, ‖u‖≤3g (ball control → custom argmax u*=A·∇_vV/‖∇_vV‖); static disc
at origin; V*(x;R)=sup_u inf_t(‖p_rel‖−R) via hj_reachability + tube postprocessor min(H,0). Grid pos=81,
vel=61 (24.4 M cells, 98 MB; v-domain ±4.0 ⊃ pool ‖v0‖≤1.5), "high"/WENO3. Per-bucket solve ~29 s,
**converges T_hj=0.50 s** (dV/chunk ≤ 1.3e-5; physical: braking time v/A≈0.14 s), 13 buckets {0.15…0.75}.
Doom lookup = corner-MAX (upper) interpolation + **eps_num=0.25** (= 2·dx_pos, a stated grid-error bound;
|∇_p V*|≈1 for a distance) — every conservatism AGAINST exclusion. **B2 census (same 3 sets):**
| set | flagged | rate |
|-----|---------|------|
| (i) all pool ICs | **0 / 2000** | 0.0000 |
| (ii) born-doomed | **0 / 72** | 0.0000 |
| (iii) reach-outcome | **0 / 1807** | 0.0000 (**GATE PASS**) |
Gates: reach=0 PASS; **A-flags ⊆ B-flags PASS** (both ∅, holds trivially); A-vs-B gap = 0. Solver validated
NON-degenerate: the R=0.5 table has 34,576 outside-disc cells with V*<0 (certified doom shell, depth to
−0.332), and a synthetic supra-grid doomed state (d_k=0.10, inward 4.0 → braking 0.27≫0.10) has V*<0 as
required. **Resolution note:** at pool speeds (‖v0‖≤1.5) the relaxed-3g doom shell OUTSIDE an obstacle is
only ≈ v²/2A ≈ 0.04 m thick — thinner than one position cell (dx=0.125) and far below eps_num=0.25 — so B
cannot add flags beyond A; A (analytic, no grid error) resolves the shell exactly and is the stronger
instrument. B independently corroborates zero.

### B3 registered caveat (verbatim)
This census is a diagnostic UNDER-approximation of true doom; it cannot certify attitude-cost doom (6D
measured infeasible, §HJ criterion). It is NOT a pool criterion in this task; any pool adoption is a separate
Researcher decision.

### Verdict
Both the analytic ballistic test and the exact 4D relaxed-system HJ solve find **ZERO provably-doomed ICs**
across all 2000 pool ICs (and 0/1807 reach, 0/72 born). The relaxed system carries 3g of authority while the
init filter already enforces an a=g stopping-distance clearance, so no pool IC is inevitably-collision even
under the weaker relaxed dynamics. The **IC-exclusion line CLOSES with evidence**: there is no
model-independent, attitude-free provable-doom set to exclude from the quadrotor pools; the residual
collisions are policy-suboptimality (a training target), not born-doomed ICs, to the extent the relaxed
system can certify. The pools remain flagged [P1] pending any (separate) Researcher decision on an
attitude-aware criterion — which the 6D measurement showed is not tractable on this hardware. verify.sh green.

---

## §gravity-observability — is the tilted-band failure an aliasing artifact of the gravity-blind obs? (read-only)

Hypothesis: obs rotates everything into the body frame and DROPS θ, so it is SO(2)-invariant — correct only if
the dynamics are SO(2)-equivariant. Gravity (fixed world (0,−g)) breaks that, so a memoryless obs-conditioned
π / V̂ cannot tell upright from inverted or reliably choose the recovery rotation. Diagnostic
`scripts/analysis/quadrotor_gravity_obs.py` (sha 8dd998a64ac5b859); brake-envelope best.pt re-rolled on the full pool
(coll 123, goal 1807, other 70). Facts only; the fix decision is the Strategist's/Researcher's.
Artifact `scratchpad/quadrotor_gravity_obs.json`. verify.sh green (129).

### D1 — thrust-misfire discrimination: DISCONFIRMS the active-misfire mechanism
MISFIRE step := f > 0.25·f_max AND Re·(−n_k) > 0.5 (actively thrusting toward the binding obstacle). Fraction
of misfire steps, last 10 pre-collision vs matched near-obstacle (surf<0.5) reach passages, by |θ| at the step:
| \|θ\| band | collisions (misfire frac, n steps) | reach near-passages (misfire frac, n) |
|-----------|-----------------------------------|---------------------------------------|
| [0,π/6)   | 0.076 (225) | **0.425 (5779)** |
| [π/6,π/2) | 0.127 (362) | 0.314 (926) |
| [π/2,π]   | **0.037 (383)** | 0.094 (171) |
The gravity-blindness prediction was "misfire HIGH and concentrated at \|θ\|>π/2 in collisions, low in
near-upright passages." The data is the **opposite**: collision misfire is LOWEST at high tilt (0.037) and
successful passages misfire MORE at every band (0.094–0.425). Per-collision misfire fraction is p50=0.0
(p90=0.2). → the competing reading holds: high-tilt collisions are NOT the vehicle actively thrusting into the
obstacle; they are drift / failure-to-recover (thrust not directed at the obstacle). **The "misfire" signature
is DISCONFIRMED.**

### D2 — θ decodability probe: CONFIRMS gravity-blindness in the failure regime
Ridge + 2-layer MLP, obs → (sinθ,cosθ), episode-split. Angular error (MLP, degrees):
| stratum | error° | | stratum | error° |
|---------|--------|-|---------|--------|
| overall (ridge / MLP) | 51.9 / **12.8** | | \|θ\|∈[0,π/6) | **6.0** |
| \|θ\|∈[π/6,π/2) | 35.7 | | \|θ\|∈[π/2,π] | **85.0** (≈random) |
| v-tercile low/mid/high | **1.1** / 6.1 / 31.1 | | t<0.5s / t≥0.5s | **67.3** / 9.9 |
| **failure cell (\|θ\|≥π/2 AND t<0.5s, n=369)** | **98.6** (worse than random) | | | |
θ is recoverable from obs only once gravity has imprinted the world frame into the velocity (v-low 1.1°, late
9.9°), and is **essentially undecodable in the failure cell** (high tilt + early: 98.6°). A memoryless net
genuinely cannot know θ there. **Decodability prediction CONFIRMED.**

### D3 — aliasing existence + 1-step dynamics divergence: CONFIRMED (exact + pervasive)
Aliased pairs (θ′=θ+δ, scene/velocity rotated by δ about the body) are IDENTICAL to the nets — obs gap ≤1e-6,
π gap ≤1.1e-5, h(t=0) gap ≤1e-6 (sanity ✓). Their TRUE 1-step divergence under the SAME control (p10/p50/p90):
| set | Δh | Δclosure-rate (m/s) |
|-----|-----|---------------------|
| collision-precursor (n=369) | 0.004 / 0.058 / 0.179 | 0.072 / **0.424** / 0.777 |
| benign near-obstacle (n=600) | 0.020 / 0.140 / 0.458 | 0.118 / **0.541** / 0.841 |
States the nets cannot tell apart diverge by a median ≈0.4–0.5 m/s of closure and Δh≈0.06–0.14 in a SINGLE
step. The aliased-dynamics gap is real and PERVASIVE (present at both precursor and benign; if anything larger
in the benign near-obstacle set) — i.e. a structural representational gap, not one concentrated only at
collision precursors. **Aliasing existence + non-trivial dynamics divergence CONFIRMED.**

### D4 — value-inheritance profile at t=0 (ALL 2000 pool ICs): CONFIRMS the false-doom certificate
V̂(t=0) and HardNet feasibility@t=0 by \|θ₀\| band, split by eventual outcome:
| \|θ₀\| band | n | infeasible@0 frac | of infeasible@0, frac that REACH | reach rate |
|------------|---|-------------------|----------------------------------|------------|
| [0,π/6)   | 337  | 0.089 | 24/30 = **0.800** | 0.929 |
| [π/6,π/2) | 659  | 0.091 | 47/60 = **0.783** | 0.926 |
| [π/2,π]   | 1004 | **0.168** | 104/169 = **0.615** | 0.881 |
High-tilt ICs start HardNet-infeasible ≈2× as often (16.8% vs 8.9%), yet **61.5% of the infeasible-at-birth
high-tilt ICs eventually REACH the goal** (80% in the near-upright band). The V̂-derived feasibility certificate
FALSELY marks demonstrably-savable states as doomed-at-birth, disproportionately in the tilted band — the
"value inherited the nominal's recovery incompetence" signature. **CONFIRMED** (and consistent with the
§doom census: none of these are provably doomed).

### Net reading (facts, not verdict)
The obs-level gravity blindness is REAL and concentrates in the tilted/early failure regime: θ is undecodable
there (D2, 98.6°), aliasing is exact with a ≈0.4 m/s 1-step closure gap (D3), and the feasibility certificate
spuriously dooms savable high-tilt ICs (D4, 61.5% still reach). BUT the failure is NOT active thrust-misfire
into the obstacle (D1 disconfirmed: high-tilt collision misfire 0.037, below successful passages) — it presents
as failure-to-recover / drift, consistent with a memoryless net that cannot resolve θ (hence the correct
recovery rotation) in exactly the regime where the dynamics are aliased. The representational hypothesis is
supported at the θ/value level; the specific kinematic "misfire" prediction is not. Fix decision deferred to
Strategist/Researcher (e.g. restore θ or a gravity-direction feature / history to the obs).

---

## Close — PROTOCOL FOLLOW-UP resolution (v2.6.2, seed 42; NO git — Researcher/Strategist apply)

Tag hygiene at close. Status of every follow-up tag touching this version:

- **RESOLVED — amendment-3 scene-init recoverability test + `_recov1` pool lineage** (tagged §amendment-3,
  line 321). The IC-exclusion line is CLOSED by the §doom census: the provable-doom set is EMPTY under the
  sound 3g relaxed system (ballistic 0/2000 + 4D-HJ 0/2000, both gates pass), and attitude-cost doom is
  uncertifiable (6D dense HJ measured infeasible). No `_recov1` pool was ever regenerated and none is needed;
  the full-range IC pool is the official quadrotor distribution. The recoverability code
  (`src/common/quadrotor_recoverability.py`, `src/envs/scene_init.py::_passes_recoverability_filter`) stays
  committed but **INERT** — `recov_margin` is unset in every config, so the filter is a no-op (criterion
  rejected at §criterion FP anatomy). No further action.

- **RESOLVED (delegated) — corrected-plant-constants + §4.4-terminal promotions** (carried from v2.6.1
  `phase_v261_report.md:227-236`). The corrected plant (tau_max 0.2→1.0, torque sat threshold) and the BPTT
  terminal (`w_terminal=30`; note `w_terminal_v` is now 0 in v2.6.2, so only the `‖p_T−g‖` terminal is active)
  are substantiated deviations; **promotion into `01_env`/`03_train`/`changes.md §4` is being applied by the
  Strategist at this close** (the docs/protocol edits are the Strategist's step, not the Executor's).

- **OPEN — theory repo merge.** The feasibility / observability / liveness theory-note sections referenced for
  the roadmap are NOT yet committed to the canonical note; the only theory tex in-repo remains
  `docs/versions/v2.6.0/underactuated_jt_theory.tex`. Carried forward (Researcher-gated).

No other tags resolved. verify.sh green (129) at close.
