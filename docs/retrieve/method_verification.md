# jt — method-prose verification, eleven claims

**Read-only retrieve.** No run, no edit, no re-scoring, nothing about git. Every verdict cites
`file:line` at the entry point the **deployment path actually imports**, traced from
`src/eval/evaluate.py` → `src/eval/rollout.py:138 rollout_eval` → `src/common/filter_hardnet.py`
(`HardNetFilter`) → `src/common/value_net.py:50 make_h_fn` → `system.observation`.

**The cell these verdicts describe** is L328 / L321's registered cell, and every config value quoted
is read from that run's own persisted config,
`data/runs/v2.9.1/set__20260814-030107__seed42/v2.9.1__jt__20260814-030107__seed42/config.yaml`,
not from the repository defaults. Where the two differ it is stated.

Verdicts: **holds** · **holds-with-correction** · **fails**.

---

## 1. M1a — certificate target is the max-over-time hazard along the closed-loop rollout

**HOLDS WITH CORRECTION.** Two corrections: the operator is a *discounted* avoid recursion, not a
plain supremum, and the rollout is the *filtered* closed loop.

**Entry point.** `src/frameworks/jt_pncbf/losses.py:167 value_targets` → `:315 pncbf_target(h_seq, …)`
→ `src/frameworks/oc_pncbf/value_target.py:13 pncbf_target`. The L328 run has
`value_target.conditioning: task_stored`, so the branches at `losses.py:181` (`task_raw_lagged`) and
`:229` (`learned_recovery`) are not taken and `batch.h_sequence` is used directly (`losses.py:310`).

**Exact form** (`value_target.py:37-62`), with `γ = exp(−λ_disc·dt)`:

```
costs[t]  = clamp(h[t], -1.0, ceiling)                          ceiling = network.value.ceiling
lhs[T-1]  = costs[T-1]
lhs[t]    = max( costs[t],  (1-γ)·costs[t] + γ·lhs[t+1] )       # discounted avoid recursion
int_rhs[t]= (1-γ)·costs[t] + γ·int_rhs[t+1]
rhs_full  = int_rhs + γ^(T-1-t)·V_target(x_T)                   # bootstrapped tail
target    = clamp( lhs + target_rhs·relu(rhs_full - lhs), -1.0, ceiling )
```

- **The correction.** `lhs` is the *discounted* max-over-time. It reduces to the undiscounted running
  max only as `γ → 1`. The target also carries a one-sided bootstrap push
  `target_rhs·relu(rhs_full − lhs)` and a clamp — neither is a max over time.
- **`ceiling` = 1.3**, from `network.value.ceiling` (run config). The lower clamp is −1.0 and is
  deliberately not configurable (`value_target.py:20-22`).
- **The rollout is the filtered closed loop.** `batch.h_sequence` is labelled at collection by
  `src/frameworks/jt_pncbf/collection.py:465-471`: the rollout runs the current policy **under the
  HardNet filter** (`_make_collection_filter("hardnet", …)`, `collection.py:461`) at `sigma=0`, and
  `value_target_barrier` is evaluated at every stored state. So the closed loop is π **plus** the
  certificate filter, not π alone.
- **Bootstrap tail read-out is the ensemble `min`**, not the deployed mean:
  `value_net.target_h` = `torch.min(value_all, dim=1)` (`src/common/value_net.py:43-44`), called at
  `losses.py:295`.

## 2. M1a — sign convention: V ≤ 0 is the certified side

**HOLDS.**

- The hazard is positive-unsafe: `signed_h` returns `1 − 2·clamp(clearance/h_scale, 0, 1)`
  (`src/common/signed_h.py:76`), so it is **+1 inside a cylinder** and −1 beyond `h_scale`.
- The deployed row is built by the single constructor
  `src/common/filter_hardnet.py:596 _row_upper`, documented at `:606` as
  **`b = −L_f h − α·(h + gamma_margin)`**, with `gamma_margin = 0.0` on this cell
  (`filter.gamma_margin` absent ⇒ 0.0, `filter_hardnet.py:550`). The enforced constraint is therefore
  `L_g h · u ≤ −L_f h − α h`, i.e. `ḣ ≤ −α h`, which renders **{h ≤ 0}** forward invariant.
- The deployed `h` is the certificate itself: `make_h_fn` returns `value_net.deployed_h(obs)`
  (`src/common/value_net.py:55-59`), i.e. the ensemble **mean** clamped to `[−1, 1.3]`
  (`value_net.py:38-41,46-47`).

## 3. M1b — observation truncation: K, ordering, frame, obstacle slot dimension

**HOLDS.** All four parts.

**Entry point.** `src/envs/quadrotor_3d.py:106 observation`, called by the eval path through
`make_h_fn` (`value_net.py:56`) and by the policy.

| part | verdict | evidence |
|---|---|---|
| **K value** | **5** | `self.k_obs = int(config["env"]["k_obs"])` (`quadrotor_3d.py:43`); run config `env.k_obs: 5` |
| **ordering** | nearest by **surface distance**, ascending | `quadrotor_3d.py:117` calls `top_k_obstacles`; `src/common/observation.py:126-134`: `surface_distance = ‖centers − p‖ − radii`, inactive `masked_fill(+inf)`, `argsort(…, stable=True)`, first K |
| **frame** | **body** | `c_off_b = _rot_k(Rt, c_off_world)` with `Rt = R(q)ᵀ` (`quadrotor_3d.py:108,125`); `v_b`, `goal_b`, `g_b` likewise (`:120,122,124`) |
| **slot dimension** | **4** | `torch.cat([c_off_b, top_radii.unsqueeze(-1)], dim=-1)` (`quadrotor_3d.py:126`) — 3 rotated offset components + 1 radius |

Two details the prose should not omit:

- **The offset is planar before rotation.** Only `Δc_xy` is taken, with the z component set to zero
  (`c_off_world = cat([top_rel_xy, zeros], -1)`, `:119`), then rotated into body. The third slot
  component is therefore a rotation image of a horizontal vector, not an altitude offset.
- **The encoder is the hard top-K.** `self.encoder` resolves to `hard_topk` on this run
  (`quadrotor_3d.py:55`; run config `obs.encoder: hard_topk` and `obs.quadrotor_3d.encoder:
  hard_topk`), so the `soft_topk` branch at `:113-115` is not taken.
- **Empty slots are zero-padded**, not dropped (`observation.py:141-143,153+`).
- **Dimension check.** `obs_dim = 3+3+3+3+4·k_obs + 2` (`quadrotor_3d.py:50`) = **34** at k_obs 5 with
  `obs_band_z: True` — the trailing 2 are absolute `p_z` and world `v_z` (`:133`).

## 4. M2 — deployed hazard branch structure for quadrotor_3d, and the combination operator

**HOLDS.** Obstacle channel plus two band branches, combined by **maximum**.

**Entry point.** `src/common/quadrotor_barrier.py:97 value_target_barrier`.

```
phi      = signed_h(position(x)[..., :2], scene, h_scale, geom_form, ell)      # :112  obstacle channel
h        = phi + c * approach_barrier(x, scene, h_scale)                       # :118  (M3)
psi_up   = clamp(z - limit, max=psi_cap)                                       # :132  ceiling branch
psi_lo   = clamp(-z - limit, max=psi_cap)                                      # :133  floor branch
h        = max( h, max( psi_up + c_z*v_z , psi_lo - c_z*v_z ) )                # :134  COMBINATION
```

- **The combination operator is `torch.maximum`, nested twice** (`:134`) — the pointwise maximum over
  {obstacle-plus-approach, ceiling branch, floor branch}. Not a sum and not a soft-max.
- The obstacle channel is itself a max over cylinders (`signed_h.py:83`,
  `torch.max(h_all, dim=-1)`), with inactive slots forced to −1 (`:82`).
- **The band branches are config-gated and system-gated** (`:129-130`):
  `env.band_hazard.enabled` **True** and `system.name == "quadrotor_3d"` on this run.
- The band branches never read `h_scale` or `ell` (`:109-111` and `signed_h.py:56-58`), so the
  `geom_form` axis moves the obstacle channel only. This run uses `geom_form` **`clip`** (no `hazard`
  block in the run config ⇒ default, `signed_h.py:33-39`).

## 5. M3 — the approach term: exact form, gain key, registered value

**HOLDS.**

**Form.** `h_star = phi + c_gain · (v_xy · r̂)` — `src/common/quadrotor_barrier.py:118`, with the
system-interface implementation at `src/envs/quadrotor_3d.py:154-170`:

```
p_xy   = x[..., :2] ;  v_xy = x[..., 7:9]                                  # :160-161
toward = c_xy - p_xy                                                       # :164  agent -> cylinder centre
surf   = ‖toward‖ - radii ,  inactive -> +inf                              # :165-166
idx    = argmin(surf)                                                      # :167  nearest ACTIVE cylinder
r̂      = toward[idx] / max(‖toward[idx]‖, 1e-9)                            # :169
return  sum(v_xy * r̂)                                                      # :170  closing speed, positive = approaching
```

- **Gain key: `config["env"][system.name]["c_gain"]`** — `quadrotor_barrier.py:117`, i.e.
  `env.quadrotor_3d.c_gain`. **Registered value on this cell: 0.3.**
- **Selection matches `phi`'s**: both pick the minimum-surface-distance active cylinder
  (`quadrotor_3d.py:159`, `signed_h.py:78-83`).
- **Not the planar form.** `quadrotor_planar` uses `v ᵀRe`, the thrust-axis projection
  (`quadrotor_barrier.py:33-36`); `quadrotor_3d` uses the obstacle-directed horizontal closing speed
  above. Prose that gives `v ᵀRe` for the 3-D system would be wrong.

## 6. M3 — the band branches' vertical-velocity terms: exact form

**HOLDS.**

`src/common/quadrotor_barrier.py:132-134`:

```
psi_up + c_z * v_z        # ceiling branch, ascending v_z inflates the cost
psi_lo - c_z * v_z        # floor branch,  descending v_z inflates the cost
```

with, all read from config and never chosen in code (`:126-131`):

| quantity | source | value on this cell |
|---|---|---|
| `limit` | `env.band_hazard.limit` (`:127`) | **4.0** |
| `psi_cap` | `config["obstacle"]["per_system"]["quadrotor_3d"]["r_max"]` (`:128`) | **0.8** |
| `c_z` | `math.pi / config["env"]["bounds"]["quadrotor_3d"]["omega_max"]` (`:129`) | **π/4.0 = 0.7853981633974483** |
| `z` | `system.position(x)[..., 2]` (`:130`) | — |
| `v_z` | `x[..., 9]`, world vertical velocity (`:131`) | — |

The `clamp(..., max=psi_cap)` is on `z` only, not on `v_z`, so `∂h_z/∂v_z = ±c_z ≠ 0` including inside
the cap — which is what makes the vertical channel relative degree 1 (`:122-124`).

## 7. M4 — the filter at deployment

**HOLDS.** Four parts, all in `src/common/filter_hardnet.py`.

**(a) Projection realization: closed form — the exact single-scalar dual root. NOT candidate
enumeration on this cell.** `_select_projection` (`:313`) dispatches on `filter.projection`; the
registered cell sets **`dual_solve`** (recorded in the score artifacts' `cell.projection`), taking
`_dual_solve_projection` (`:340`), documented at `:349-353` as *"the unique nonnegative root of the
piecewise-linear non-increasing φ(λ) = aᵀclip(u_nom − λa, U)"* — exact and continuous in every
dimension. The `enumerate` realization (`_box_aware_projection`, `:260`) exists and is **not** the
deployed one; it is exact only for action dim ≤ 2, and the action dim here is 4.

**(b) Infeasible-step command: least-violation, from the candidate set.** On rows where the
half-space ∩ box is empty, `_dual_solve_projection` hands those rows to the enumeration path
(`:356-368`), which selects by
`least_bad_scores = relu(aᵀu − b) + _FEASIBILITY_TOL·‖u − u_nom‖²` then `argmin` (`:283-285`,
`:300-302`) — least violation, ties broken toward `u_nom`. `empty_mode` is **`argmin`** on this cell
(the `prox` softmin blend at `:291-299` is not taken).

**One eval-only override to note.** The registered scoring cell replaces `filter.empty_fallback` with
`{kstep, phases 1, k 3}` (the run's own training config carries `{mode: none, k: 10}`). Under it,
`:192-199` overwrites the least-violating action on empty rows with the first-phase control of the
two-phase k-step argmin. **The returned flag is deliberately not touched** (`:186-190`): the row still
counts infeasible.

**(c) Singular-row handling: the row is not special-cased; the projection degenerates to the box
clamp of `u_nom`.** `singular = ‖L_g h‖ < _SINGULAR_LG_THRESHOLD` with the threshold **5.0e-4**
(`:15`, `:144`). Nothing branches on `singular` for the action — it is only OR-ed into the returned
flag (`:213`). What executes is the ordinary projection with a vanishing row: `_base_projection`
(`:245-257`) divides the correction by `‖L_g h‖² + epsilon² + lg_reg_eps`, so as `‖L_g h‖ → 0` the
correction → 0 and the output → `clamp(u_nom, box)`; the dual solve then returns the same point since
the row constrains nothing. **A singular row therefore executes the nominal command, clipped to the
box, while being reported infeasible.**

**(d) Box bounds source: `system.u_bounds`.** `bounds = self.system.u_bounds.to(...)`
(`:121`), defined at `src/envs/quadrotor_3d.py:84-86` as `[[f_rotor_min, f_rotor_max]] * 4` — read
from `env.bounds.quadrotor_3d`, **`[0.0, 4.905]⁴`** on this cell. Per-rotor, not a torque box.

## 8. M5 — certificate loss form and target construction; policy loss form

**HOLDS WITH CORRECTION** on the certificate-loss description; **holds** on the policy loss.

**Certificate loss.** Mean-squared regression of the ensemble onto the detached target of claim 1:
the target is built by `value_targets` (`losses.py:167`) and `.detach()`ed at `:310,315-322`, so the
certificate loss is a supervised regression, not a bootstrapped TD loss with a live target. The
target read-out uses the **ensemble min** (`value_net.py:43-44`) while deployment uses the **ensemble
mean** (`:46-47`) — the two read-outs differ and prose that names one for both is wrong.

**Policy loss.** Assembled at `losses.py:1134-1139`:

```
total = task
      + lambda_a   * action_norm
      + lambda_s   * smoothness
      + lambda_sat * saturation_excess
      + lambda_pretanh * pretanh
      + w_outside  * outside
```

with `task = (gate_in * task_cost).mean()` and `outside = (gate_out * v_next).mean()`
(`:1132-1133`), `v_next` being the certificate evaluated at the **unfiltered** one-step successor
`rk4_step(system, states, u0, dt)` (`:1130-1131`). Optional terms (rate, u_reg, du, infeas, floor
recovery, deficit, friction, agree, horizon critic) are added at `:1143-1246` under their own weights;
on this cell the ones the config zeroes contribute nothing.

## 9. M5 — does the task gradient reach certificate parameters?

**HOLDS: it is stopped.** Three independent mechanisms, all in the shipped path.

1. **Optimizer partition.** `opt_pi = AdamW(policy_net.parameters(), …)`
   (`src/frameworks/jt_pncbf/train.py:430-434`) holds **only** policy parameters;
   `opt_vs = AdamW(value_net.parameters(), …)` (`:425-429`) only certificate parameters. No policy
   update can step θ_V.
2. **Explicit clear inside the policy update.** `_policy_updates` calls
   `value_net.zero_grad(set_to_none=True)` immediately after `optimizer.zero_grad(set_to_none=True)`
   (`train.py:1658-1659`), before `optimizer.step()` at `:1676` — so any θ_V gradient the loss graph
   does form is discarded and cannot accumulate into the next value step.
3. **Detach at the gate read-outs.** `v_now` (`losses.py:866`) and `v_gate` (`:911`) are `.detach()`ed
   at the call.

**One qualification the prose should carry.** The `outside` term at `losses.py:1131` calls
`value_net.deployed_h(...)` **without** a detach, so a θ_V path exists in the autograd graph; it is
neutralized by (1) and (2), not by a stop-gradient at the site. The claim "the task gradient does not
update the certificate" is exact; "the task gradient never touches θ_V in the graph" is not.

Separately, `detach_filter_coeffs` (`losses.py:784`, applied at `filter_hardnet.py:96-102`) controls
whether the policy BPTT gradient flows through the CBF coefficients' state dependence. It is
**false** on this cell, so it does not contribute a stop-gradient here.

## 10. M5 — update schedule and initialization

**HOLDS: alternating, not simultaneous; certificate warm-started, policy fresh.**

**Schedule — alternating, per macro step.** `train.py:560-561` reads `k_v = K_V`, `k_pi = K_pi`; the
macro step runs `_value_updates(..., n_updates=k_v)` (`:701-716`) and then, gated on
`step > vs_warmup_steps and k_pi > 0`, `_policy_updates(..., n_updates=k_pi)` (`:742-751`). On this
cell: **K_V = 3, K_pi = 1, vs_warmup_steps = 200**, `n_steps = 10000`, `bptt_T = 30`,
`schedule_n_steps = 5000`. So each macro step is three certificate updates followed by one policy
update, and the policy is not updated at all for the first 200 macro steps.

**Initialization — the two keys that decide it** are `training.jt.value_init_ckpt` and
`training.jt.pi_init_ckpt` (`train.py:356-357,370-373`).

| network | this cell | key |
|---|---|---|
| certificate | **warm start** from `data/runs/v2.9.1/set__20260813-164148__seed42/v2.9.1__oc__20260813-164148__seed42/checkpoints/best.pt` — the OC condition's own best checkpoint | `training.jt.value_init_ckpt` |
| policy | **fresh** (`null`) | `training.jt.pi_init_ckpt` |

Both loaders take one state dict only — `value_init_ckpt` loads `v_s_state`, `pi_init_ckpt` loads
`pi_state` (`train.py:370-373`) — and are **not** a joint resume; `train.py:501` records that the
other channels start fresh, unlike `resume_ckpt`.

## 11. Network architecture, and that no shared encoder exists

**HOLDS.** The Fig. 2 correction rests on this and the code supports it.

**Certificate — `ValueNetEnsemble`, `src/common/value_net.py:15-47`, members built by
`_make_value_member` (`:64-80`).**

| property | value | evidence |
|---|---|---|
| ensemble members | **2** | `n_vs` (`:18`); run config `network.value.n_vs: 2` |
| hidden layers per member | **3** | `n_layers` (`:20`); run config `n_layers: 3` |
| width | **256** | `hidden` (`:19`); run config `hidden: 256` |
| activation | **Softplus(β = 20.0)** | `:74`; run config `softplus_beta: 20.0` |
| head | `Linear(256 → 1)`, weight `N(0, 0.01)`, bias 0 | `:76-79` |
| read-out | `clamp(·, −1.0, 1.3)`; deployed = **mean** over members, target = **min** | `:38-47`; `ceiling: 1.3` |

**Policy — `ControlNet`, `src/common/control_net.py:14-62`.**

| property | value | evidence |
|---|---|---|
| hidden layers | **2** | `n_layers` (`:24`); run config `n_layers: 2` |
| width | **256** | `hidden` (`:23`); run config `hidden: 256` |
| activation | **leaky_relu** | `:25,35`; run config `activation: leaky_relu` |
| head | `Linear(256 → action_dim = 4)` | `:38`; `quadrotor_3d.py:39` |
| output map | `clamp(gain·tanh(z), −1, 1)` then affine to the box: `center + half_width·bounded` | `:48-55,59-62` |
| `output_gain` | **1.4286** | `:27`; run config `output_gain: 1.4286` |

**No shared encoder — three independent confirmations.**

1. **Both networks consume the raw observation directly.** `_make_value_member(obs_dim, …)` starts
   `in_dim = obs_dim` (`value_net.py:71`) and `ControlNet` starts `in_dim = obs_dim`
   (`control_net.py:32`). Neither takes a feature tensor from the other.
2. **No module is shared.** The value members live in `self.members` (`value_net.py:27-29`); the
   policy trunk in `self.trunk` (`control_net.py:37`). They are separate `nn.Module`s constructed
   separately and never cross-referenced.
3. **The parameter sets are disjoint at the optimizer.** `opt_vs` takes `value_net.parameters()` and
   `opt_pi` takes `policy_net.parameters()` (`train.py:425-434`) — a shared encoder would place the
   same tensors under both.

The only thing the two share is the **function** `system.observation(x, scene)`, which is a fixed,
parameter-free feature map (`quadrotor_3d.py:106-133`) evaluated independently on each side — for the
certificate via `make_h_fn` (`value_net.py:56`) and for the policy at its own call sites. That is a
shared *input*, not a shared encoder.

---

## Summary

| # | claim | verdict |
|---|---|---|
| 1 | M1a target = max-over-time hazard along the closed-loop rollout | **holds with correction** — discounted avoid recursion + bootstrap push + clamp; loop is *filtered* |
| 2 | M1a sign, V ≤ 0 certified | **holds** |
| 3 | M1b K, ordering, frame, slot dimension | **holds** — K 5, surface-distance ascending, body frame, 4 per slot |
| 4 | M2 branch structure and combination operator | **holds** — obstacle + floor + ceiling, combined by `max` |
| 5 | M3 approach term form, gain key, value | **holds** — `c_gain·(v_xy·r̂)`, `env.quadrotor_3d.c_gain` = 0.3 |
| 6 | M3 band vertical-velocity terms | **holds** — `psi_up + c_z v_z`, `psi_lo − c_z v_z`, `c_z` = π/4 |
| 7 | M4 filter: projection, infeasible command, singular row, box source | **holds** — closed-form dual root; least violation; nominal-clipped-to-box; `system.u_bounds` |
| 8 | M5 certificate and policy loss forms | **holds with correction** — regression on a detached target; min vs mean read-outs differ |
| 9 | M5 task gradient stopped at the certificate | **holds** — by optimizer partition + explicit `zero_grad`, not by a site stop-gradient |
| 10 | M5 schedule and initialization | **holds** — alternating K_V 3 / K_pi 1; certificate warm-started, policy fresh |
| 11 | network shapes and no shared encoder | **holds** |
