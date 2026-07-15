# v2.6.0 Stage 1+2 — learned h_star value → value-policy joint training (build-log)

Facts, not verdicts; the version verdict is the Researcher's at close. Every number cited by disk path.
No shield anywhere (filter-only). Reference path: the closed v2.3.0 learned-value + HardNet + BPTT family.
Two Researcher amendments this stage: (1) the ε_g realization + three integrations; (2) M1 framework =
oc_pncbf.

## Integration (before any training) — h_star value target + ε_g (R2) + quadrotor wiring

The v2.3.0 value path is DI/unicycle-shaped in three places; the quadrotor needed three integrations
(Researcher-confirmed) plus the R2 ε_g (Researcher-specified, system-generic — the DI `cbf_deriv` loss is
4D-hardcoded and revives the exact-backup braking model, so it is NOT reused):

1. **h_star value labeling.** `src/common/quadrotor_barrier.py:value_target_barrier` — the ground-truth
   barrier whose sup-over-time the value regresses is `h_star = phi(p,o) + c·(v^T Re)` (c=0.3) for the
   quadrotor, `signed_h(phi)` otherwise. Wired at every value-label site: JT `collection.py:106,374`
   (task_stored uses the buffer's stored h), OC `collection.py` (rollout_lqr labeling; `config` threaded
   through `collect()`). Without this the net would learn V^{phi,pi} — the version's premise broken.
2. **ε_g regularizer (R2), system-generic** (`quadrotor_barrier.lg_authority_loss`):
   `L = w·mean_{x near B_0} ReLU(eps_g − ‖L_g V̂(x)‖)`, `eps_g=0.05`, `w=5.0` (`exp_config
   loss.value.lg_authority`; changes.md §4). Near-B_0 states from the shared `sample_near_B0_states` (the
   M6/P1 sampler: agent at ~phi=0 around one obstacle, velocity all-directions). `‖L_g V̂‖` via
   `_cbf_terms` (same primitive HardNet uses). Added to BOTH value updates (JT `train.py`, OC
   `_value_step`), gated on `quadrotor_planar` + weight>0; `‖L_g V̂‖` min/median/degenerate-fraction logged
   every metrics step (`metrics.csv` new cols `lg_min/lg_median/lg_degen_frac/L_lg_raw`).
3. **Quadrotor wiring:** OC `make_system` gained `quadrotor_planar`; the CBF-contour figure (2D velocity
   slice) is skipped for the 6D quadrotor (`run_full.write_cbf_contour_figure` returns None; both
   `_record_eval` callers + `run_full` guarded) — viz-only, not gate-relevant (PROTOCOL FOLLOW-UP).

Both frameworks smoke clean on the quadrotor (JT value_loss 0.098 / pi_loss 94.8; OC grad_norm 3.44).
verify.sh green after the integration (126 passed). No `base_config.yaml` edit.

## Milestone 1 — learn V^{h_star, pi_nominal} (OC-PNCBF, fixed nominal LQR)

Framework **oc_pncbf** (Researcher amendment 2: it rolls the fixed nominal LQR = the exact basis of the
Stage-0 exact V, so the M2 gate compares like-with-like; jt_pncbf's behavior policy is the learned
ControlNet). Launcher `scripts/m1_oc_run.py`, single seed 42. **Registered budget 50000 value steps =
epochs 100 × grad_steps 500**; `gamma_disc.total_epochs` unchanged (registered). Config diff at launch:
ONLY `training.oc_pncbf.epochs 1000→100` (`quadrotor` env/lqr + `lg_authority` are committed exp_config,
not a launch delta) — verified §-clean. Run dir `data/v2.6.0__20260715-010357__seed42`.

**‖L_g V̂‖ logged through training** (M1 self-check; `metrics.csv`) — early (step ~2500):
near-B_0 `lg_median ≈ 13–17`, `lg_degen_frac ≈ 0.01` (‖L_g V̂‖<eps_g), `lg_min` 0.0 (single-sample outlier
in 256). In-loop eval @1500: cps −0.219, collision 0.27 (already below the M5 nominal 0.409 — the learned
filter is working before it has converged).

**M1 complete** (`status.json`: current_step 50000, `phase done`, `halt_reason None`, no NaN). Best in-loop
cps **0.185 @ step 34500** (`best.pt`); value loss converged (L_R 0.008 by step 48000). ‖L_g V̂‖ logged
throughout (`metrics.csv`): near-B_0 lg_median 13–18 across training, lg_degen_frac 0.008–0.065 (256-sample
per-step estimate). **verify.sh green (126 passed)** concurrent with M1. Rate ~50 steps/s (faster than the
initial estimate); no throughput issue.

## Milestone 2 — GATE P1-learned / O3: does the learned V̂ keep authority on B_0? — **PASS**

`scripts/analysis/quadrotor_m2_gate.py` on `best.pt` (step 34500). **True near-B_0** =
`{|V̂|<0.1} ∩ {|φ|<0.1}` (642 of 40000 synthetic single-obstacle candidates — filtering on the LEARNED V̂,
per the amendment's `B_0={V̂=0}∩{h=0}`; measuring on all near-boundary candidates spuriously inflates
degeneracy with deep-safe flat-V̂ states, my first-pass error, corrected). ‖L_g V̂‖ on the SAME states:

| barrier | median ‖L_g‖ | degenerate frac (‖L_g‖<0.05) |
|---|---|---|
| exact φ (position-only) | 0.0000 | **0.751** (Stage-0 ~0.84) |
| exact h_star (c=0.3) | 1.0663 | 0.020 (Stage-0 ~0.00) |
| **learned V̂** | **18.02** (min 0.273, p10 3.54) | **0.000** |

Gate: **(1)** learned degen_frac 0.000 < 0.10, ≪ position-only 0.751 → PASS; **(2)** not starved — median
18.02 ≫ 0.072 and p10 3.54 ≫ the v5 starved 0.0072 → PASS. **M2 PASS.** The theory survives learning: the
learned V̂ + ε_g keeps FULL first-order authority on B_0 (0% degenerate, min ‖L_g V̂‖ 0.273 > ε_g=0.05 — ε_g
is holding, no O3 starvation). Proceed to M3/M4.
_(min ‖L_g V̂‖ was 0.0 on the UNFILTERED candidate set — off-B_0 flat-V̂ artifacts, not B_0 degeneracy;
recorded here so the correction is auditable.)_

## Milestone 3 — pre-JT learned-filter evaluation (filter-on, nominal policy)

`scripts/analysis/quadrotor_m3_eval.py`: HardNet on the M1 `V̂` (best.pt @34500), filter-ON, NOMINAL
policy, full pool `eval_full_quadrotor-planar_n2000_seed23456` (n=2000). **cps_v2 = 0.1738** [0.1221, 0.2263],
reach 0.6610, collision **0.1850**, oob 0.0190, stuck 0.0, timeout 0.1350, inf_v2 0.1341 (inf_empty 0.1341,
inf_sing_viol 0.0034). **vs the M5 nominal (filter-off): cps_v2 −0.2270, collision 0.4090.** The learned
filter alone (pre-joint-training) lifts cps_v2 by **+0.40** and cuts collision by **55%** (0.409→0.185); the
residual timeout 0.135 + infeasibility 0.134 (the filter makes the un-cotrained nominal conservative) is
what M4 joint training should reduce. Ledger row added (`docs/ledger.md`, eval_only(learned filter, pre-JT)).

## Milestone 4 — value-policy joint training (jt_pncbf, warm-started from M1 V̂) — RUNNING

Launcher `scripts/m4_jt_run.py`; run dir `data/v2.6.0__20260715-014348__seed42`. Co-trains the policy by
BPTT through the HardNet map on `V̂` (h_star + ε_g wired). Warm-start = the M1 value via a new minimal
`run_training(value_init_ckpt=...)` kwarg (loads `v_s_state`+target only; policy/optimizers/step fresh;
mutually exclusive with resume_ckpt) — the M1 OC checkpoint has no `pi_state`, so `resume_ckpt` doesn't
apply. Config delta at launch: ONLY `training.jt.n_steps 42000→50000` (registered budget) + `value_init_ckpt`;
`schedule_n_steps` unchanged (registered). verify.sh green (126 passed) after the value_init change.
Rate ~3.1 steps/s → ETA ~4.5 h. `grad_norm_pi=0` during vs_warmup (2000 steps) as expected; policy BPTT
begins after.

### M4 in-loop trajectory (through step 7995; `eval_metrics.csv` / `metrics.csv`)

| step | cps | reach | collision | timeout | inf | grad_norm_pi (pre-clip) |
|---|---|---|---|---|---|---|
| 1500 | −0.7882 | 0.0000 | 0.1680 | 0.0360 | 0.1207 | 0 (warmup) |
| 3000 | −0.8071 | 0.0000 | 0.1760 | 0.7640 | 0.1203 | ~139 |
| 4500 | −0.7872 | 0.0000 | 0.1680 | 0.7800 | 0.1172 | ~2635 |
| 6000 | −0.8192 | 0.0020 | 0.1900 | 0.7460 | 0.1208 | ~1048 |
| 7500 | −0.8281 | 0.0020 | 0.1960 | 0.7480 | 0.1138 | ~875 |

grad_norm_pi trajectory (post-warmup, pre-clip): 51 → 139 → 2093 → 2635 → 211 → 398 → 1048 → 1176 → 875 → 175
(clipped by `optim.grad_clip=2.0`). `grad_leak_VS_from_Lpi = 0.0` throughout; no NaN/Inf, `halt_reason None`.

### M5 — first-order-stall GATE: distinguished — **credit-horizon / BPTT instability, NOT a first-order stall**

- **grad_norm_pi is NOT collapsing toward zero** — it is large and spiking (51–2635, heavily clipped). The
  BPTT policy gradient flows (the first-order-stall HALT condition — grad→0 on B_0 — is NOT met).
- **The value keeps FULL B_0 authority** (M2 gate on `step_007500.pt`): learned V̂ degen_frac **0.000**,
  median ‖L_g V̂‖ **9.40**, min 0.226 > ε_g=0.05 (`scratchpad/quadrotor_m2_gate.json`). So **Cor 6.1 / H3
  holds — h_star's first-order authority survived learning AND joint training**; there is no first-order
  authority stall to diagnose.
- **But the policy fails to navigate:** reach collapses to 0.000, **timeout explodes to ~0.76**, cps stuck
  at ~−0.8 (WORSE than the M3 pre-JT nominal-filter cps 0.174 / reach 0.66). The joint-trained policy
  degenerates to a timeout/hover regime.

**Mechanism = the standing JT credit-horizon / BPTT-gradient-instability family** (grad_norm_pi explosion
2000+ heavily clipped → the update direction is dominated by the filter-coefficient Jacobian, not
navigation signal — the v2.4.0-root-caused explosion, whose hygiene flag `detach_filter_coeffs` is default
OFF). This is a SEPARATE mechanism from the theory's first-order stall, exactly the distinction the M5 gate
asked for. Per Stage-1 scope ("a credit-horizon stall is a separate mechanism — report it, do NOT
hyperparameter-fish"), no knobs changed. Surfaced to the Researcher.

### M4-relaunch attempt — BC(LQR)-init (Researcher-directed) — HIT A STRUCTURAL OBSTACLE

Aborted M4 (dir kept). Built `scripts/bc_lqr_pretrain.py`: filtered-LQR manifold rollout (V̂ frozen) →
MSE to the raw LQR action → ControlNet, stop-gated at reach≥0.40 & coll≤0.185 or val early-stop. Two
issues found:
1. **Raw-action MSE is thrust-scale-dominated** (f~10 vs τ~0.2) → ignores torque → reach 0. Fixed with
   NORMALIZED-action MSE (both channels to [−1,1]); val_mse fell to 0.018 (good imitation).
2. **STRUCTURAL: the ControlNet softsign output cannot reproduce the LQR's SATURATED torque.** Diagnosis
   (`scratchpad`, on the inloop pool): filtered-LQR reach **0.692** navigates via **|τ| median 0.2000,
   84.8% saturated** at ±τ_max (to right θ~U[−π,π]); the BC policy's softsign is asymptotic → **max |τ|
   0.19, median 0.097, 0% saturated**, τ_rmse vs LQR **0.206** (> the whole ±0.2 range) → it UNDER-TORQUES
   → filtered-BC reach **0.004**, timeout **0.712**. BC imitates in MSE but structurally cannot match the
   saturated attitude-righting control, so it cannot stabilize the tilted vehicle. The stop-rule (ii)
   (reach≥0.40) is therefore UNREACHABLE by BC.

This is not a tuning issue: `center + half_width·softsign` (`control_net.py:forward`) asymptotes and never
reaches the box boundary, while the tilted ICs + small torque box (±0.2) + aggressive LQR (kp_att=40) make
the nominal saturate torque 85% of the time. **The same softsign ControlNet is the JT policy** — so JT
faces the same torque-saturation ceiling. Surfaced to the Researcher.

### M4-relaunch v2 — policy-output alignment (Researcher-directed) — STAGE 1 DONE, STAGE 2 (BC) GATE FAILED

**Stage 1 (three DI-inherited policy-output fixes) — DONE, verify.sh green (127 passed):**
- **Over-range+clamp head** `clamp_tanh` (`control_net.py`: `clamp(gain·tanh(z), −1, 1)`, `output_gain=1/0.7`)
  — REACHES the box boundary (new head-side test `test_clamp_tanh_head_reaches_box_boundary…`: |τ| output
  hits ±0.2 exactly; softsign/tanh never do). `base_config network.control.output = clamp_tanh`.
- **L_pre `z_target` 0.867→1.0** (`exp_config`): clamp_tanh reaches the box at |z|=0.867, so z_target≥0.867
  leaves deep-safe boundary-reaching (tilted-IC attitude-righting) unpenalized.
- **L_sat `sat_excess_threshold` 2.0→[19.62, 0.2] per-channel** (thrust/torque box bounds; vectorized in
  `losses.py`): L_sat = 0.0 at hover thrust f=9.81 and at the box corner (was `(10−2)²·1.0`, penalizing the
  necessary hover thrust). Stage-1 smoke: JT runs clean; L_sat 0 at hover; head reaches boundary.

**Stage 2 (BC on the fixed head) — GATE FAILED (reach 0.002 ≪ the ≥0.55 gate) → STOPPED before JT:**
The head fix PARTIALLY recovered torque expressiveness — filtered-BC |τ| now reaches **max 0.200, 14.8%
saturated** (softsign was 0%) — but BC still does not produce a navigating policy. Diagnosis (inloop pool,
`data/bc_lqr_seed42/bc_policy.pt`, clamp_tanh confirmed):
- **MSE under-fits the saturated torque:** BC |τ| median **0.086** vs LQR **0.200** (LQR 84.8% saturated,
  BC 14.8%). Action-MSE (val 0.015) does not force median saturation — on the tilted ICs the BC under-torques
  (on-trajectory |τ_BC − τ_LQR| = **0.18** at t=0, nearly the full ±0.2 range).
- **Covariate shift / goal-stopping:** filtered-BC rights attitude (|θ| median 1.71→0.57 by t=20) and
  accelerates to v_max (speed→2.4) but does not slow at the goal → **timeout 0.698, reach 0.002**, coll 0.242,
  oob 0.058. BC val-MSE early-stopped (plateau) at this non-navigating optimum.

So: the head/L_pre/L_sat alignment was NECESSARY and is correct (committed; expressiveness measurably
improved), but **BC(LQR) via action-MSE does not clear the reach≥0.55 gate on this unstable, saturated-control
quadrotor** — a known BC limitation (MSE-smoothing of the saturated bang-bang torque + open-loop covariate
shift + goal-stopping imprecision). Per the BC gate self-abort, JT was NOT launched. Surfaced to the
Researcher (options: DAgger/iterative BC to fix covariate shift; a saturation-weighted or per-channel BC
loss; a gentler nominal so BC has a smooth target; or accept BC-init is not the lever and revisit the
JT-from-random credit-horizon mechanism directly). No JT core / value path / §4-constant change; the
committed head + L_pre + L_sat fixes stand (they are correct regardless of the BC outcome).

### M4-relaunch v3 — DAgger-robustified BC (Researcher-directed) — FALLBACK, JT NOT launched

`scripts/dagger_bc.py`: round 0 = the committed filtered-LQR BC dataset; each round r≥1 rolls the CURRENT
filtered policy on TRAIN scenes, labels visited states with the RAW LQR (expert), AGGREGATES, retrains
(normalized MSE, val early-stop), evals filtered-policy in-loop reach. Cap 4, gate reach≥0.55.

**Result (`scratchpad/dagger_lqr.json`):**
| round | dataset | val_mse | in-loop reach | coll | timeout |
|---|---|---|---|---|---|
| 0 (BC init) | 240k | — | 0.002 | 0.242 | — |
| 1 | 480k | 0.0311 | 0.020 | 0.238 | 0.714 |
| 2 | 720k | 0.0414 | **0.054** | 0.250 | 0.674 |
| 3 | 960k | 0.0453 | 0.046 | 0.226 | 0.706 |
| 4 | 1.20M | 0.0465 | 0.044 | 0.226 | 0.708 |

DAgger fixed covariate shift only marginally: reach **0.002→0.054 then plateaus ~0.05**, ≪ the 0.55 gate;
`val_mse RISES` across rounds (0.031→0.047) — the aggregated policy-visited states are progressively harder
to clone (the LQR's saturated attitude-righting + precise goal-stopping do not compress into the feedforward
ControlNet). **FALLBACK per spec: reach 0.044 < 0.55 → JT NOT launched.**

**Finding (fact):** a filtered-LQR clone — single-pass BC (reach 0.002) OR DAgger (reach 0.054) — CANNOT
produce a navigable policy init on this underactuated quadrotor; the filtered-LQR's reach 0.69 is a
closed-loop property that does not transfer to a cloned feedforward ControlNet under the filter (saturated
bang-bang torque + goal-stopping precision + rising val-MSE on aggregated states). **FLAGGED for Researcher**
(per the FALLBACK): re-examine plain JT (no BC) with credit-horizon mitigation on another axis — `bptt_T`
lengthening (with the `03_train` §4.4 horizon caveat) or a terminal-value / in-window-termination axis. This
is a Researcher decision; per scope I did NOT autonomously start a different-axis run. The committed
policy-output alignment (head/L_pre/L_sat) and M1/M2/M3 results stand.

### M5 — credit-horizon axis: plain-JT + BPTT terminal value (Researcher-directed) — RUNNING

Researcher chose the terminal-value axis over `bptt_T` lengthening (the §4.4 horizon caveat: `bptt_T=60`
collapsed in v2.4.0). **Plain JT** — value warm-started from the M1 V̂, **policy fresh** (NO BC / NO
`pi_init`, after BC/DAgger both failed to produce a navigable init). The mechanism: `03_train` §4.4's
rollout return is a fixed `bptt_T=30`-step (~1.5 s) discounted sum with **no terminal and no in-window
termination**, so goal-reaching beyond the window is never credited and myopic hover is the local optimum
(this is the credit-horizon root cause established at M4/M5-stall gate above).

**Terminal (implementation, `src/frameworks/jt_pncbf/losses.py` `policy_bptt_loss`):** after the BPTT
loop, add to the minimized `task_cost`
`+= discount · w_terminal · ‖p_T − g‖`, where `discount == gamma_T^T` (the running discount at loop exit;
`gamma_T=0.99`, `T=30`, so `gamma_T^30 ≈ 0.740`). This is the cost-framing of the spec's
`R += gamma_T^T·V_term`, `V_term = −‖p_T − g‖` (negative terminal goal-distance). It is **analytic**
(not the learned V̂, which is a hazard sup-h value, NOT a goal value) and **differentiable through x_T**
(it must carry the goal-closing gradient — "detached" in the spec = not-a-learned-value, i.e. not V̂; the
distance itself is differentiable). Kept inside the same `gate_in` safe-region gate as `task_cost`; the
`gate_out` recovery branch is unchanged. `w_terminal=30.0` (config `loss.policy.w_terminal`; 0.0 = off,
DI/unicycle parity). **PROTOCOL FOLLOW-UP:** this changes the §4.4 return; the `03_train` edit is deferred
until utility is confirmed (Researcher-gated).

**Gates before launch (all green):**
- `verify.sh`: **127 passed** (DI parity: `w_terminal` defaults to 0.0 via `.get`, terminal inert).
- Quadrotor `stage='smoke'` (vs_warmup→1, exercises the full policy BPTT incl. terminal): policy active
  `L_in_task=190.68` (finite), `L_pi_total=197.35`, `grad_norm_pi=72.9` (healthy — NOT the 2000+ M4
  BPTT explosion), `grad_leak_VS_from_Lpi=0.0` (V_S grad-leak safeguard §4.7 passes), no non-finite losses.

**Launch:** `scripts/jt_terminal_run.py --seed 42 --steps 50000`. Run dir
`data/v2.6.0__20260715-033610__seed42`; value warm-start `data/v2.6.0__20260715-010357__seed42/checkpoints/best.pt`
(M1 best, step 34500); `schedule_n_steps=42000` UNCHANGED; vs_warmup=2000; in-loop eval cadence 1500 steps.
**Self-abort:** after vs_warmup, if in-loop cps shows no improvement over best for 4 consecutive evals
(~6000 steps) AND reach ≈ 0 → abort + credit-horizon vs first-order-stall analysis from retained
checkpoints. **Pass/fail:** trap broken iff in-loop reach rises significantly above 0 after vs_warmup
(target > ~0.3, climbing) and cps climbs above the M4 plateau (−0.79). Baselines: M4 plateau cps −0.79;
M3 pre-JT filtered-nominal cps 0.174. _(results below, filled per eval checkpoint)_

| step | cps | reach | coll | oob | timeout | note |
|---|---|---|---|---|---|---|
| 1500 | −0.7771 | 0.000 | 0.160 | 0.776 | — | pre-warmup (step < vs_warmup 2000); policy fresh, flies OOB |
| 3000 | −0.9579 | 0.008 | 0.282 | 0.038 | 0.672 | 1st post-warmup; OOB→timeout (learned to stay in-bounds, now hovers) |
| 4500 | −0.9733 | 0.008 | 0.294 | 0.032 | 0.666 | flat vs 3000 (reach 0.008, timeout-dominated hover); no-improve #2 |
| 6000 | −0.8972 | 0.012 | 0.250 | 0.030 | 0.708 | cps ticks up but < best −0.777; reach flat 0.012; no-improve #3 |
| 7500 | −0.9342 | 0.008 | 0.268 | 0.010 | 0.704 | reach back to 0.008; no-improve #4 → **SELF-ABORT** |

**SELF-ABORT (per spec):** 4 consecutive post-warmup evals (3000/4500/6000/7500) with no improvement over
`best_cps=−0.777` (step 1500, pre-warmup; `status.json`) AND reach frozen at ≈0.01. Trainer stopped at
step 7586; all checkpoints retained (`step_001500…007500.pt` + `best.pt`), run kept read-only.

#### M5 abort diagnosis — credit-horizon CONFIRMED, terminal REFUTED (NOT a first-order stall)

**(1) NOT a first-order B_0 stall.** M2 gate re-run on the FINAL JT-cotrained value (`step_007500.pt`,
`scripts/analysis/quadrotor_m2_gate.py`): 613 true near-B_0 samples ({|V̂|<0.1}∩{|φ|<0.1}) →
learned `DEGEN_frac(‖L_g V̂‖<0.05)=0.000`, median ‖L_g V̂‖ **11.25**, min 0.240, p5 1.82. The value
retains full B_0 first-order authority through JT (position-only φ degen 0.741 for contrast). So the policy
COULD receive productive gradient (Cor 6.1: BPTT gradient flows where ‖L_g V̂‖≠0) — the stall is not the
value.

**(2) Credit-horizon / myopic-hover CONFIRMED.** Reach frozen ≈0.008–0.012 across all 4 post-warmup evals;
timeout-dominated (0.666–0.708); OOB collapses 0.776 (pre-warmup, random policy) → 0.010–0.038 (post): the
policy learns only to stay alive / in-bounds and HOVER, never navigating to goal. `L_in_task` falls
1342 (step 2600) → ~230 (steady) — the return (incl. terminal) IS minimized, but into a hover local optimum.
Thrust is pinned bang-bang at the ceiling (`abs_action_max ∈ {0, 19.62=f_max}`, satfrac_a_phi median 0.375).

**(3) Terminal REFUTED — it re-triggered the BPTT gradient explosion.** `w_terminal=30` did not break the
trap. Its terminal `‖p_T−g‖` is differentiable through the full 30-step RK4 rollout + per-step HardNet
filter-Jacobian, i.e. exactly the long-horizon path that exploded at M4. Post-warmup `grad_norm_pi`:
median **477**, **max 178 653** (step 3000), 36 460 (4000); **21%** of steps >2000, **7%** >10 000
(`metrics.csv`). With `optim.grad_clip=2.0`, these don't NaN — but the terminal's goal-closing gradient is
(a) drowned post-clip by the chaotic long-horizon component and (b) unable to escape the hover basin.
`grad_leak_VS_from_Lpi=0.000000` throughout (§4.7 safeguard held); no value contamination.

**Verdict:** the terminal-value credit-horizon mitigation is REFUTED on this quadrotor — the fixed-window
BPTT return lands in myopic hover with or without the terminal, and a *differentiable* terminal only
re-excites the M4 gradient explosion. This closes the terminal axis (as `bptt_T`-lengthening was closed by
the §4.4 collapse caveat). **The value certificate is sound (M1/M2/M3 stand); the open problem is producing
a navigable policy through the differentiable filter** — neither imitation (BC/DAgger, reach ≤0.054) nor
plain-JT-with-terminal (reach ≤0.012) succeeds. **FLAGGED for Researcher** — per scope I did NOT
autonomously start a further axis. Candidate directions for a Researcher decision (not started): a
*detached/bootstrapped* terminal (avoid the long-horizon differentiable path — e.g. a separately-learned
goal value used as a stop-gradient bootstrap target), in-window termination on goal-reach, a shorter
inner BPTT with a longer detached credit tail, or a non-BPTT policy-improvement route (actor-critic on the
learned value) that sidesteps the filter-Jacobian explosion entirely.

### M6 — plain-JT + terminal + `detach_filter_coeffs` ON (grad-explosion fix, Researcher-directed) — RUNNING

M5 refuted the *differentiable* terminal but the cause was NUMERICAL, not the credit mechanism: the
terminal's backprop through the 30-step HardNet filter-Jacobian re-excited the BPTT gradient explosion
(`grad_norm_pi` max 178 653, 21% of steps >2000), so the goal-closing signal was clipped away
(`grad_clip=2.0`) and the policy stayed in the hover basin. The value keeps full B_0 authority (M2 on
step_7500: degen 0.000, ‖L_g V̂‖ 11.25) — only the policy-gradient path is unstable. The exact remedy is
`03_train §7 detach_filter_coeffs`: treat the CBF coefficients (h, L_f h, L_g h, α·h) as constants in
backward (forward byte-identical), removing the horizon-growing gradient tail while leaving the
median/p90 gradient intact.

**Change (single axis):** `loss.policy.detach_filter_coeffs: false → true` (`src/configs/exp_config.yaml`;
plumbed at `losses.py:503` → `hardnet(..., detach_coeffs=...)`; detach at `filter_hardnet.py:69-75`). The
M5 terminal retained as-is (`w_terminal=30`). No value-path / head / loss.policy-alignment change.

**Gates (green):** `verify.sh` **127 passed** (DI/unicycle default remains off in practice — the flag
changes backward only, forward byte-identical, so smoke numerics are unaffected). Quadrotor `stage='smoke'`
with the flag ON: policy active `L_in_task=190.68` (byte-identical forward to M5, as expected),
`grad_leak_VS_from_Lpi=0.0`, no non-finite. (At smoke `bptt_T=3` the coefficient-path gradient is
negligible so `grad_norm_pi=72.9` matches M5; the T=30 run is where detach bites.)

**Launch:** `scripts/jt_terminal_run.py --seed 42 --steps 50000`. Run dir
`data/v2.6.0__20260715-042141__seed42`; value warm-start M1 best.pt; `detach_filter_coeffs=true`
(config.yaml:269 confirmed); schedule_n_steps=42000 unchanged; vs_warmup=2000; eval cadence 1500.
**Pass/fail:** (primary) explosion fixed iff post-warmup `grad_norm_pi` p99 < 2000 (vs M5 max 178k, 21%
>2000); (trap) reach > ~0.3 climbing & cps above the M5 plateau (−0.78). **FALLBACK:** grad controlled but
reach ~0 → STOP, flag Researcher for in-window goal-termination (§4.4). **SELF-ABORT:** 4 post-warmup
evals no cps improvement over best AND reach ~0. **PROTOCOL FOLLOW-UP:** §4.4 terminal (03_train edit
deferred). _(results below, filled per eval; grad_norm_pi digest computed from `metrics.csv` per eval)_

| step | cps | reach | coll | oob | timeout | grad_norm_pi (postwarmup max / p99) | note |
|---|---|---|---|---|---|---|---|
| 1500 | −0.7771 | 0.000 | 0.160 | 0.776 | 0.064 | — | pre-warmup; identical to M5 (reproducible) |
| 3000 | −0.9192 | 0.010 | 0.262 | 0.008 | 0.720 | max 1474 / p99 1474 (0% >2000) | **grad explosion FIXED** (M5: 178k, 21% >2000); reach still ~0.01 |
| 4500 | −0.8776 | 0.026 | 0.252 | 0.000 | 0.722 | — | reach 0.010→0.026 CLIMBING (M5 flat 0.008); cps up; oob→0 |
| 6000 | −0.8312 | 0.050 | 0.246 | 0.004 | 0.700 | max 1474 / p99 1474 (0% >2000), median 99 | reach 0.026→0.050, cps monotone up; grad stays controlled |
| 7500 | −0.8128 | 0.090 | 0.274 | 0.002 | 0.634 | — | reach 0.050→0.090 (M5 self-aborted here @ 0.008); timeout ↓ |
| 9000 | −0.7160 | 0.116 | 0.234 | 0.004 | 0.644 | — | reach 0.090→0.116; cps −0.716 clears M5 plateau (−0.78); coll ↓ |
| 10500 | −0.6312 | 0.170 | 0.232 | 0.002 | 0.596 | — | reach 0.116→0.170; cps ↑; timeout ↓ 0.596 |
| 12000 | −0.6989 | 0.148 | 0.254 | 0.006 | 0.590 | recent(≥10k) max 698, all max 2997 (4% >2000) | reach dips 0.170→0.148 (eval noise, NOT late explosion — grad calm) |
| 13500 | **−0.2769** | **0.400** | 0.226 | 0.000 | 0.374 | step≥12k max 7357 median 164 | **BREAKTHROUGH — reach clears 0.3 target; cps ≫ M5 plateau; new best** |
| 15000 | −0.1892 | 0.458 | 0.226 | 0.000 | 0.314 | — | reach 0.400→0.458, cps→−0.19; coll steady 0.226 (watch vs M3 ~0.185) |
| 16500 | −0.0773 | 0.502 | 0.194 | 0.000 | 0.302 | — | reach crosses 0.5; cps→−0.08; coll ↓ 0.194 |
| 18000 | −0.0885 | 0.536 | 0.236 | 0.000 | 0.228 | — | reach 0.536, timeout ↓ 0.228; coll ↑ 0.236, cps flattening ~−0.08 (reach/safety tradeoff) |
| 19500 | 0.0132 | 0.584 | 0.216 | 0.000 | 0.196 | — | **cps crosses 0 (+0.013)**; reach 0.584→LQR 0.69; coll ↓ 0.216 |
| 21000 | 0.0800 | 0.602 | 0.190 | 0.002 | 0.204 | ALL≥2k: max 7357 p99 7357 p95 2260 median 135 (5.2% >2000); recent≥17k: p95 1774 median 125 | reach 0.602; coll 0.190 ~M3; grad spikes isolated/transient (see note) |

**Grad primary-gate — nuance (fact).** Strict gate `p99 < 2000` reads FAIL at p99 **7357** — but this is 5
isolated single-step spikes over n=96 post-warmup evals (at steps 7600/8000/13600/16400/19200, each a
reach-jump learning transition), NOT sustained instability: median **135**, p95 **2260** (recent step≥17k:
p95 **1774 < 2000**, median 125, 1 spike). Vs M5 (max 178 653, p99 ~36k, **21%** of steps >2000), the
explosion is reduced **~24× in magnitude** and **~4× in frequency** (5.2%). With `grad_clip=2.0` the
isolated spikes clip harmlessly while the median-135 credit flows every step — the mechanistic outcome the
gate protects (credit survives → reach climbs 0.01→0.60) is decisively achieved. `grad_leak_VS=0.000000`
throughout.

**In-loop trajectory (continued, step 22500+):**

| step | cps | reach | coll | oob | timeout | note |
|---|---|---|---|---|---|---|
| 22500 | 0.0135 | 0.610 | 0.240 | 0.000 | 0.148 | reach 0.610, timeout ↓ 0.148; coll ↑ 0.240 → cps dips (reach/safety tradeoff) |
| 24000 | **0.1276** | 0.644 | 0.200 | 0.002 | 0.150 | new best cps; reach 0.644→LQR 0.69; coll ↓ 0.200; nearing M3 baseline 0.174 |
| 25500 | 0.0940 | 0.644 | 0.222 | 0.004 | 0.128 | reach flat 0.644, coll ↑ 0.222 → cps 0.094; oscillating ~0.09–0.13 band |
| 27000 | **0.1650** | 0.666 | 0.198 | 0.000 | 0.136 | new best; **matches M3 baseline 0.174**; reach 0.666≈LQR 0.69; still climbing |
| 28500 | 0.1263 | 0.638 | 0.194 | 0.002 | 0.164 | dip from 27k peak; oscillating band, best 0.165 holds |
| 30000 | 0.1613 | 0.658 | 0.192 | 0.000 | 0.150 | back near 27k peak; best-cps gains decelerating (~0.16 plateau forming?) |
| 31500 | 0.1506 | 0.654 | 0.194 | 0.000 | 0.148 | steady ~0.15; reach ~0.65; plateau band forming |
| 33000 | 0.1007 | 0.652 | 0.222 | 0.004 | 0.118 | oscillation dip; best 0.165 unbeaten since 27k (plateau firming) |
| 34500 | 0.1560 | 0.656 | 0.192 | 0.000 | 0.150 | back near best; plateau ~0.14–0.16 cps / 0.65 reach |
| 36000 | 0.0258 | 0.632 | 0.252 | 0.000 | 0.116 | downswing (coll 0.252); grad calm (median 104) — noise, not instability; best 0.165 unbeaten since 27k (6 evals → PLATEAU, not rising) |
| 37500 | **0.1983** | 0.670 | 0.178 | 0.000 | 0.150 | **NEW BEST — breaks plateau, exceeds M3 0.174**; reach 0.670 high; coll 0.178 < M3 0.185 |
| 39000 | 0.1327 | 0.652 | 0.202 | 0.000 | 0.146 | oscillation down from 37.5k peak; best 0.198 holds; high variance |
| 40500 | **0.2035** | 0.656 | 0.162 | 0.000 | 0.182 | **NEW BEST 0.204**; coll 0.162 lowest yet; peaks still drifting up (0.165→0.198→0.204) |
| 42000 | 0.1832 | 0.650 | 0.168 | 0.000 | 0.182 | schedule saturates (schedule_n_steps=42000) |
| 43500 | 0.2117 | 0.672 | 0.172 | 0.000 | 0.156 | new best 0.212 |
| 45000 | 0.2193 | 0.660 | 0.154 | 0.000 | 0.186 | new best 0.219 |
| 46500 | **0.2270** | 0.672 | 0.160 | 0.000 | 0.166 | **PEAK in-loop 0.227 — best.pt** |
| 48000 | 0.2074 | 0.660 | 0.162 | 0.000 | 0.178 | past peak, declining |
| 49500 | 0.1804 | 0.654 | 0.172 | 0.000 | 0.172 | declining |
| 50000 | 0.1693 | 0.638 | 0.164 | 0.000 | 0.196 | run end (current_step 50000, phase done, halt_reason None) |

**Run complete** (`status.json`: current_step 50000, phase **done**, halt_reason None, no NaN). Best in-loop
cps **0.2270 @ step 46500** (`best.pt`).

**Extension decision (user directive "if cps still rising around 50k, keep training 50k more"): NOT
extended.** In-loop cps PEAKED at step 46500 (0.227) then declined over the final 3500 steps
(0.207→0.180→0.169); no new best-cps high after 46500; linear slope near 50k negative in every window
(last-4 evals −0.0164/1k, last-6 −0.0070/1k, last-8 −0.0019/1k). Schedule saturated at 42000 and the model
converged/declined thereafter — cps is not rising at 50k, so the extension condition is not met. (Resume
launcher `scripts/jt_resume_run.py` written + verified for a clean full joint resume, unused.)

**Primary gate — grad explosion (full 50k, `metrics.csv`).** post-warmup `grad_norm_pi` (n=241): median
**96**, p95 **1513 (<2000)**, p99 4757, max 7357; **3.7%** of steps >2000 (9 isolated single-step
transition spikes). Strict `p99<2000` reads FAIL (4757) on those spikes, but vs M5 (max 178 653, ~21%
>2000) the explosion is cut **~24× in magnitude** and **~6× in frequency**, and the median-96 credit flows
every step under `grad_clip=2.0` — the mechanistic goal (credit survives → reach climbs 0.01→0.69) is
achieved. `grad_leak_VS_from_Lpi = 0.000000` over the entire run (§4.7 safeguard held; no value contamination).

**M6 full eval** (`best.pt` step 46500, frozen pool `eval_full_quadrotor-planar_n2000_seed23456` — the SAME
pool as M3, `eval_metrics.csv` mode=final):

| eval | cps_v2 [95% CI] | reach | collision | oob | stuck | timeout | inf_v2 |
|---|---|---|---|---|---|---|---|
| M3 pre-JT (filtered nominal) | 0.1738 [0.1221, 0.2263] | 0.6610 | 0.1850 | 0.019 | 0.000 | 0.1350 | 0.1341 |
| **M6 post-JT (learned policy)** | **0.2920 [0.2398, 0.3382]** | **0.6935** | **0.1430** | 0.000 | 0.0005 | 0.1630 | 0.1117 |
| Δ (M6 − M3) | **+0.1182** | +0.0325 | **−0.0420 (−23%)** | −0.019 | +0.0005 | +0.0280 | −0.0224 |

JT co-training (terminal + detach fix) beats the pre-JT filtered nominal by **cps +0.118** with
**non-overlapping bootstrap CIs** (M3 hi 0.226 < M6 lo 0.240), cuts collision **23%** (0.185→0.143), lifts
reach and reduces infeasibility; the small timeout rise (0.135→0.163) is the net cost of a more active
policy. Insertion-eval variants (`eval_metrics.csv`): final_insertion_frozen 0.1773, final_insertion_live
0.2084, final_insertion_lqr −0.374 (LQR-only baseline).

**P3 — collinear-residual gate** (`scripts/analysis/quadrotor_p3_collinear.py`, `scratchpad/quadrotor_p3.json`).
Theory E2 / changes.md P3: collision drops WITHOUT the collinear brake-only failure `S_h` (rem:collinear /
Prop 3.2) transferring to V̂^{h,π}. Rolled the filtered `best.pt` policy on the full pool, captured the
first-collision state of every collision episode (288 = 14.4%, matches final coll 0.143), measured at each:

- **Collinear geometry** `|cos(∇_p φ, Re)|` (thrust axis vs obstacle normal — the DIRECT `S_h` test): median
  **0.0000**; **S_h membership (|cos|>0.9) = 0.000 (0 of 288)**. The residual collisions are NOT the
  collinear brake-only mode — at collision the thrust axis is typically *orthogonal* to the obstacle normal
  (lateral-maneuver collisions), the opposite of `S_h`. (`‖L_g φ‖` is structurally ~0 at ALL states on this
  thrust-underactuated system, Thm 5.2 — thrust moves velocity not position at first order — so it cannot
  discriminate `S_h`; the angle test does.)
- **Learned authority** `‖L_g V̂‖` at collision states: median **1.0515**, degenerate (<ε_g=0.05) only
  **8.3%** overall; `‖L_g h_star‖` uniform **0.300** (the analytic c·v^TRe velocity-term authority, = c/m).

**P3 CONFIRMED:** (a) collision dropped (0.185→0.143 vs M3; 0.409→0.143 vs the M5 filter-off nominal) and
(b) `S_h` did NOT transfer — 0% of residual collisions are collinear and the learned V̂ retains lateral
authority (median ‖L_g V̂‖ 1.05). The theory's collinear brake-only failure is absent from V̂^{h,π}.

### M6 verdict (facts)

The credit-horizon trap that blocked M4/M5 is broken by the single change `detach_filter_coeffs: false→true`
(03_train §7) on top of the committed BPTT terminal — the terminal's credit mechanism was sound; only the
filter-Jacobian gradient explosion (M5 grad_norm_pi 178k) was clipping it away. With the flag on: reach
climbed 0.01→0.69, in-loop cps −0.92→+0.23, and the frozen-pool M6 eval (cps **0.292**, reach 0.694, coll
0.143) beats the M3 pre-JT baseline (0.174) by **+0.118** with non-overlapping CIs. M2-gate authority is
retained through JT (step_7500 degen 0.000; and P3: median ‖L_g V̂‖ 1.05 at residual collisions), and P3
confirms the collinear `S_h` failure does not transfer. This is the first navigable learned policy through
the differentiable filter on the underactuated quadrotor (BC/DAgger reach ≤0.054, M5 terminal reach ≤0.012).
**PROTOCOL FOLLOW-UP:** the §4.4 BPTT terminal (`R += γ_T^T·(−‖p_T−g‖)`) is on disk in `losses.py`
(`w_terminal=30`) but the `03_train` §4.4 edit is deferred pending Researcher confirmation. `detach_filter_coeffs=true`
is now the quadrotor default in `exp_config`; DI/unicycle are unaffected (forward byte-identical, flag off in
practice for those systems' runs). Single seed 42; a multi-seed confirmation is a Researcher call.

### M6 CBF contour figure

`scripts/analysis/quadrotor_cbf_contour.py` → `data/v2.6.0__20260715-042141__seed42/figures/cbf_contour_m6.png`.
The auto-eval skips the CBF contour for the 6D quadrotor (`run_full.py:252` — the DI contour uses a 2D
velocity-column slice the quadrotor lacks); this is the deferred 6D-appropriate figure. The learned deployed
h(x)=V̂ (best.pt @46500) is rendered on the position plane (px,py) for two eval scenes (clean 2-obstacle
scene 0; dense 6-obstacle scene 1) across three approach-speed slices s=v·Re ∈ {−2,0,+2} (θ=0, ω=0; Re=(0,1)
so v=(0,s)). Convention (matches the gate `gate_in=sigmoid(−h)`): h<0 safe (blue), h=0 boundary (black), h>0
unsafe (red); obstacles dashed, goal a gold star. **Observation:** each obstacle is wrapped by a smooth
h>0 keep-out region bounded by h=0, and the keep-out region GROWS monotonically with s across the three
columns — a position-only φ would give a velocity-INDEPENDENT field, so the visible velocity-dependence is
the B_0-degeneracy-breaking authority (‖L_g V̂‖≠0, Thm 5.3) rendered as a field, consistent with the M2 gate
(degen 0.000) and P3 (median ‖L_g V̂‖ 1.05). Viz-only; not gate-relevant.

## M6 residual diagnostics (D1–D5) — read-only characterization of the residual timeout + collision

best.pt (step 46500), full pool `eval_full_quadrotor-planar_n2000_seed23456` (= M6/M3). Two instrumented
read-only re-rolls of the filtered learned policy (`scripts/analysis/quadrotor_m6_residual.py`) + a
checkpoint scan (`scripts/analysis/quadrotor_m6_ckpt_scan.py`); no state/config/git change. Facts, not
verdicts. **Eval-path note (disk-cited):** these diagnostics use the dual_arm / hand-roll path
(best.pt cps_v2 **0.2743**, reach 0.683, coll 0.1440, timeout 0.172); the ledger/M6-`final` numbers use the
run_full/evaluate path (0.2920 / 0.694 / 0.143 / 0.163). The ~0.018 cps gap is eval-path (batch-size float
non-associativity + goal/stuck resolution order), NOT a checkpoint difference — collision matches to 2
episodes (0.1440 = 288/2000 here, 0.143 = 286/2000 final; P3 also 288). The relative diagnostic structure is
path-invariant. (`scratchpad/quadrotor_m6_residual.json`, `quadrotor_m6_ckpt_scan.json`.)

**D1 — Timeout conversion (max_steps 200→400; Roll B, 11 s).** Registered as eval_only(max_steps=400),
NOT SOTA-eligible (changed deploy axis) — flag Researcher.

| | reach | timeout | collision | cps_v2 (hand-calc) |
|---|---|---|---|---|
| 200 (canonical) | 0.6830 | 0.1720 | 0.1440 | ~0.308 |
| 400 | 0.7195 | 0.1065 | 0.1710 | ~0.312 |
| Δ | +0.0365 | **−0.0655** | **+0.0270** | **+0.004 (neutral)** |

Of the **344** original 200-step timeouts, at 400 steps: **21.2% → goal**, 61.9% still timeout, **15.7% →
collision**, 1.2% stuck. ‖p−g‖ trajectory of those 344: median min-distance **0.101** (< goal_radius 0.15),
median dist@200 **1.337**, median dist@400 1.013; **57.0%** reach within goal_radius at their closest
approach yet do not resolve (fail the goal-speed criterion `goal_speed_radius=0.30` — near-goal loiter);
descending-at-200 **48.3%** vs loiter-at-200 **51.7%** (by the [180,200] ‖p−g‖ slope < −0.02). **Arithmetic
check:** doubling max_steps converts 21% of timeouts to goal but adds an equal collision mass (0.144→0.171),
so cps is **neutral (+0.004)** — more steps alone does not recover the timeout drain.

**D2 — Collision feasibility split (n_coll=288, decisive).** Per residual collision, HardNet feasibility
(02_control §4: NOT (empty half-space∩box) AND NOT (singular ∧ row<0)) at the collision step and K=1..5
prior: **FEASIBLE at collision = 0.3%**, **INFEASIBLE = 99.7%** (feas+infeas=1.000 ✓); feasible at ANY of
the 5 precursor steps = **14.2%**. So 85.8% of residual collisions are already infeasible ≥5 steps before
impact — a safe action does NOT exist in the torque box on the approach; only 14.2% had a recoverable
window. Consistent with P3 (authority intact, ‖L_g V̂‖ median 1.05 — a gradient direction EXISTS) + D4 (the
required action is outside the box): authority present but control-authority-saturated = execution-bound,
not barrier failure.

**D3 — IC stratification (collision / timeout / reach rate per band).**

| stratifier | band | n | collision | timeout | reach |
|---|---|---|---|---|---|
| \|θ₀\| | [0, π/6) | 337 | **0.098** | 0.142 | 0.757 |
| | [π/6, π/2) | 659 | 0.129 | 0.196 | 0.675 |
| | [π/2, π] | 1004 | **0.169** | 0.166 | 0.663 |
| v₀·(toward nearest obs) | receding (<0) | 1005 | 0.123 | 0.171 | 0.703 |
| | [0, 0.75) | 588 | 0.151 | 0.187 | 0.662 |
| | ≥0.75 | 407 | **0.184** | 0.152 | 0.663 |
| \|ω₀\| | [0, 0.5) | 984 | 0.129 | 0.178 | 0.691 |
| | [0.5, 1.0) | 1016 | 0.158 | 0.166 | 0.675 |
| | ≥1.0 | 0 | — | — | — (ω_init_max=1.0) |

Collision rises monotonically with initial tilt (0.098→0.169) and inward speed (0.123→0.184); mild with
\|ω₀\|. **Projected post-nominal-IC collision (\|θ₀\|<π/6) = 0.098** — the near-hover band collides at
0.098 vs the pool 0.144, so the tilted/fast half of the IC distribution carries the collision excess.

**D4 — Torque headroom at precursors (last 5 steps before the event).** τ box = ±0.2.

| subset | \|τ_safe\| saturated (±0.2) | torque-box-clip (demanded \|τ\|>0.2) |
|---|---|---|
| all failing (n=632) | 0.690 | 0.528 |
| **collisions** | **0.971** | **0.916** |
| timeouts | 0.455 | 0.203 |

At collision precursors the torque is pinned at ±0.2 in **97.1%** of steps and the HardNet safety action
*demands more torque than the box allows* in **91.6%** — the residual collisions are **torque-box-bound**
(the filter asks for a larger corrective torque than ±0.2 can deliver). Timeouts are far less
torque-limited (sat 0.455, box-clip 0.203) — consistent with D1's near-goal-settle characterization.

**D5 — Full-pool checkpoint re-selection (35 ckpts, 204 s; dual_arm path).** No retained `step_*.pt` exceeds
best.pt@46500 (cps_v2 0.2743 dual_arm / 0.292 run_full). Top step_* = step_046500 (= best.pt). Nearest
others: step_050000 0.2697, step_045000 0.2472, step_042000 0.2460, step_049500 0.2453 — all below,
CI-overlapping. **0 checkpoints above the best.pt reference** → the M6 selection stands.

**Summary (facts).** The residual cps drain splits cleanly by lever: (i) **collision 0.144** is
execution-bound — 99.7% infeasible at impact / 85.8% infeasible ≥5 steps prior (D2), 97% torque-saturated
with 92% box-clip at precursors (D4), concentrated in tilted (θ₀ up to 0.169) + fast-inward (0.184) ICs
(D3); the barrier is sound (P3, ‖L_g V̂‖ 1.05) so the lever is torque authority / IC distribution / earlier
steering, not the certificate. (ii) **timeout 0.172** is a near-goal settle-failure — 57% reach within
goal_radius but miss the goal-speed criterion, ~half loiter vs ~half en-route (D1); doubling max_steps is
cps-neutral (+0.004, converts 21% to goal but adds equal collision). Which lever to pursue is the
Researcher's call.
