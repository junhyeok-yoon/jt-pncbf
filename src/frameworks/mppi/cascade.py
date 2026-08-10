"""v2.8.4 charter-"v5" STAGE 2 — CASCADED MPPI. A SEPARATE BASELINE, NOT VANILLA MPPI.

=================================================================================================
READ THIS FIRST: WHAT THIS IS AND HOW IT MUST BE REPORTED
=================================================================================================

This controller LEAVES THE VANILLA MPPI CLASS. Vanilla MPPI (`mppi_controller.py`) plans directly in
the plant's own input space: the sampled sequence is mapped to per-rotor forces by a STATE-INDEPENDENT
allocation and handed to the plant. The controller here plans an OUTER-LOOP command and closes an
INNER ATTITUDE LOOP underneath it, so what the plant receives at rollout step k depends on the
sample's own state at step k. That is a different algorithm with a different privilege profile, and
every number it produces is filed and reported as its OWN baseline row.

    IT MUST NEVER BE TABULATED BESIDE A VANILLA MPPI ROW WITHOUT AN EXPLICIT VARIANT COLUMN, AND ITS
    NUMBERS MUST NEVER BE BLENDED INTO, AVERAGED WITH, OR SUBSTITUTED FOR VANILLA MPPI NUMBERS.

`run_cell` stamps `cell.variant = "cascaded"` and rewrites `arm` on every cascaded record, so the
separation survives in the artifact and not only in this docstring.

=================================================================================================
THE PLAN VARIABLE — the one interpretation this file makes, stated plainly
=================================================================================================

The charter writes the planner as "MPPI over (total thrust T, desired attitude q_des)" with the inner
loop being "the PD attitude controller already implemented in src/common/filter_backup.py".

That named inner loop is `System.lqr_action` (src/envs/quadrotor_3d.py:173-189) — `filter_backup.py`'s
own module docstring says so in its first sentence, and `filter_backup.brake_action` reaches it through
`filter_backup.override_gains`, the documented WRAP mechanism ("nothing about the mixer, the thrust
projection, the attitude loop or the per-rotor box clip is re-implemented"). Reading its lines:

    a_des  = -kp_pos (p - goal) - kd_pos v                 <- OUTER loop
    f_des  = mass (a_des + gravity e3)                     <- desired WORLD force
    f_thr  = clamp(f_des . b3, min=0)                      <- TOTAL THRUST T (projected on the
                                                              CURRENT thrust axis)
    b3_des = f_des / ||f_des||                             <- DESIRED ATTITUDE (its body-up axis)
    tau    = J (kp_att (R^T (b3 x b3_des)) - kd_att omega)  <- INNER attitude PD
    f_rotor = clamp((f_thr, tau) @ mixer_inv^T)            <- rotor commands, box-clipped

BOTH of the charter's plan variables are functions of the SINGLE world-force vector f_des: T is its
projection onto the current thrust axis and q_des is its direction. They are not independent inputs of
this inner loop, and there is no third input. Moreover `b3 x b3_des` has no component along b3, so the
shipped attitude PD is YAW-BLIND: a full quaternion q_des would carry one degree of freedom the
controller cannot consume.

    THEREFORE the plan carries the DESIRED WORLD ACCELERATION a_des in R^3, one entry per decision
    step, and (T, q_des) are read off the shipped cascade's own lines as above. This is the reading of
    "(T, q_des)" that the named inner loop can actually accept; it is recorded here, in the cascade
    record of every cell, and in the build log, rather than left implicit.

HOW a_des IS FED IN, with NO re-implementation and NO gain typed. The plan must mean the TOTAL desired
acceleration, so the outer loop's own velocity-damping term is switched OFF — otherwise the realised
acceleration would be `a_des - kd_pos v` and the plan would not mean what it says. That is done with
`filter_backup.override_gains(system, kd_pos=0.0)`, exactly the mechanism and exactly the idiom
`brake_action` uses to switch the POSITION term off. The remaining outer term is then
`-kp_pos (p - goal)`, which is inverted by feeding the pseudo-goal

    goal_pseudo = p + a_des / kp_pos

with `kp_pos` READ OFF THE SYSTEM OBJECT (it comes from `config["lqr"]["quadrotor_3d"]`, the charter's
"read gains from its config"). Substituting gives `-kp_pos (p - goal_pseudo) = a_des` EXACTLY, so the
shipped `f_des = mass (a_des + gravity e3)` is the plan's own force and every line after it — the
thrust projection, the attitude error, the inner PD, the mixer inverse and the per-rotor box clip — is
the shipped code, byte for byte. THE ATTITUDE GAINS ARE NEVER TOUCHED. `cascade_smoke` MEASURES the
inversion (check (d)) rather than asserting it.

=================================================================================================
WHAT IS UNCHANGED FROM THE BASE CELL
=================================================================================================

The charter fixes everything but the planner: "Everything else (noise, lambda, terminal cost, plant,
predicate identity) unchanged."

  * COST — `stage_cost` / `terminal_cost` / `collision_mask` are imported from `cost.py` and called
    with the same `CostParams`. Not a copy, the same functions.
  * PLANT — `rk4_step(system, x, u, dt)`, the shared deployed map, as in vanilla.
  * NOISE — `_draw_noise` is INHERITED unchanged: stationary OU along the horizon axis, the same
    `correlation_steps` / `alpha`, the same per-channel scaling rule. Only the channel-scale VECTOR
    differs, because the plan's channels are accelerations rather than a wrench (below).
  * LAMBDA — the same relative rule, `lam_eff = max(lam * std_n(S_n), lam_eps_abs)`, per (scene, step).
  * PREDICATE IDENTITY — the collision predicate and the deployed terminal are the same objects, and
    the cell is scored by the same `src.eval.evaluate` path as every other v2.8.4 row.
  * DEGENERATE RULE, control hold, receding-horizon shift, hover centring — inherited.

SIGMA. `sigma` stays the base cell's value in PER-ROTOR-EQUIVALENT NEWTONS. The per-axis acceleration
std is computed at construction as

    sigma_axis = sigma * ||mixer_row_0|| / mass

i.e. the COLLECTIVE channel's own scale (the quantity vanilla applies to its F_total channel) divided
by the system's own mass to turn a force into an acceleration. The collective axis therefore keeps
exactly the marginal std vanilla gives it, and the two lateral axes inherit the same scale. Both
`||mixer_row_0||` and `mass` are read off the system object; nothing is typed. This is a CHOICE, made
because the vanilla channel scales are torques and have no acceleration counterpart, and it is
recorded per cell as `channel_scale_mode` rather than buried.

THE EFFECTIVE NOISE IS THE RAW NOISE. Vanilla re-projects post-clip rotor forces to get an honest
`effective_noise`, because its sampled wrench can leave the per-rotor box and be clipped BEFORE the
plant sees it. Here the plan is an acceleration and is NOT clipped: the sequence actually simulated is
exactly `U + eps`, so the honest effective perturbation is `eps` itself and the MPPI update is the
textbook `U <- U + sum_n w_n eps_n`. The clip still happens — inside the shipped cascade, on the rotor
forces — but it is downstream of the plan and is part of the plant's input map, not of the sampler.

B1/B2/B3 AND G1-G4 ARE OFF, as the base configuration requires. B2 is additionally UNREACHABLE here:
it seeds a body-wrench entry into the plan, and this plan carries no wrench. Construction raises if B2
is switched on rather than silently ignoring it.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from src.common.filter_backup import override_gains
from src.common.observation import scene_goal_tensor
from src.common.rk4 import rk4_step
from src.common.system import System
from src.frameworks.mppi.cost import CostParams, collision_mask, stage_cost, terminal_cost
from src.frameworks.mppi.mppi_controller import MPPIController, MPPIParams
from src.frameworks.mppi.recovery import RecoveryParams, tilt_cos, tilt_deg_from_cos


Tensor = torch.Tensor

PLAN_MODES = ("accel_world",)
CHANNEL_SCALE_MODES = ("from_collective_row_norm_over_mass",)


class CascadeController(MPPIController):
    """MPPI over the desired world acceleration, closed by the SHIPPED cascaded PD. See the module
    docstring — this is a SEPARATE baseline and never a vanilla MPPI row."""

    def __init__(
        self,
        system: System,
        config: Mapping[str, Any],
        params: MPPIParams,
        cost_params: CostParams,
        *,
        device: torch.device,
        dtype: torch.dtype,
        recovery: RecoveryParams | None = None,
        cascade: Mapping[str, Any],
    ) -> None:
        super().__init__(
            system, config, params, cost_params, device=device, dtype=dtype, recovery=recovery
        )
        plan_mode = str(cascade["plan"])
        if plan_mode not in PLAN_MODES:
            raise ValueError(f"mppi.v5.stage2.plan must be one of {PLAN_MODES}, got {plan_mode!r}.")
        scale_mode = str(cascade["channel_scale"])
        if scale_mode not in CHANNEL_SCALE_MODES:
            raise ValueError(
                f"mppi.v5.stage2.channel_scale must be one of {CHANNEL_SCALE_MODES}, got {scale_mode!r}."
            )
        if params.space != "wrench":
            # `_draw_noise`'s OU branch is inherited and is reached only on the non-legacy path.
            raise ValueError(
                "the cascaded planner inherits the OU sampler, which requires "
                f"mppi.sampling.space == 'wrench'; got {params.space!r}."
            )
        if recovery is not None and recovery.b2_enabled:
            raise NotImplementedError(
                "B2 seeds a body-wrench entry into the carried plan; the cascaded plan carries a "
                "desired world ACCELERATION and has no wrench to seed. Switch B2 off."
            )
        self.plan_mode = plan_mode
        self.channel_scale_mode = scale_mode

        # ---- PD gains, READ OFF THE SYSTEM OBJECT (which read them from config["lqr"][system]) ----
        self.kp_pos = float(system.kp_pos)
        self.kd_pos = float(system.kd_pos)
        self.kp_att = float(system.kp_att)
        self.kd_att = float(system.kd_att)
        if self.kp_pos == 0.0:
            raise ValueError(
                "the pseudo-goal inversion goal_pseudo = p + a_des / kp_pos needs a non-zero outer "
                "position gain; config['lqr'] gives kp_pos = 0."
            )

        # ---- plan geometry: the plan is 3-D (a_des), the plant input stays rotor-dimensional -------
        # `action_dim` is what the INHERITED plan bookkeeping (`reset`, `_shift`, `_draw_noise`) sizes
        # the plan with, so it is re-pointed at the PLAN dimension; the rotor dimension is kept
        # separately for the record. The per-rotor box `u_lo`/`u_hi` is untouched and unused here: the
        # box clip happens inside the shipped cascade, on the rotor forces.
        self.rotor_dim = int(system.action_dim)
        self.plan_dim = int(cascade["plan_dim"])
        if self.plan_dim != 3:
            raise ValueError(f"mppi.v5.stage2.plan_dim must be 3 for {plan_mode!r}, got {self.plan_dim}.")
        self.action_dim = self.plan_dim

        # ---- the hover anchor in acceleration space is EXACTLY the zero vector --------------------
        # a_des = 0 gives f_des = mass * gravity * e3: level attitude, collective = m g. So the S3
        # hover-centred decomposition u = u_hover + u_plan + eps carries over with anchor 0, and the
        # receding-horizon braking tail is the zero acceleration, i.e. hover.
        zeros = torch.zeros(self.plan_dim, device=device, dtype=dtype)
        self.hover_trim = zeros
        self.anchor = zeros.clone()
        self.plan_trim = zeros.clone()

        # ---- sigma: the collective channel's own scale, converted to an acceleration by the mass ---
        collective_row_norm = float(torch.linalg.norm(self.mixer[0]).item())
        self.sigma_axis = float(params.sigma) * collective_row_norm / self.mass
        self.collective_row_norm = collective_row_norm
        # `_draw_noise` multiplies the standard normals by this vector, so an all-equal vector gives
        # the three acceleration axes the same marginal std, and the OU recursion is inherited intact.
        self.sigma_channel = torch.full(
            (self.plan_dim,), self.sigma_axis, device=device, dtype=dtype
        )

    # ---- the inner loop: the SHIPPED cascaded PD, wrapped, never re-implemented -------------------
    def cascade_action(self, x: Tensor, a_des: Tensor) -> Tensor:
        """Desired world acceleration -> per-rotor forces, through `System.lqr_action` unmodified.

        `override_gains(system, kd_pos=0.0)` switches the outer loop's velocity-damping term off — the
        plan supplies the TOTAL desired acceleration, so leaving the damping in would realise
        `a_des - kd_pos v` instead — and the pseudo-goal `p + a_des / kp_pos` inverts the remaining
        outer term exactly. `kp_pos` is read off the system object. See the module docstring; the
        inversion is MEASURED by `cascade_smoke` check (d).

        `x` is [M, state_dim] and `a_des` is [M, 3]; returns [M, rotor_dim], already box-clipped by
        the shipped `_clamp_action`.
        """
        goal_pseudo = self.system.position(x) + a_des / self.kp_pos
        with override_gains(self.system, kd_pos=0.0):
            return self.system.lqr_action(x, goal_pseudo)

    def desired_wrench_record(self, x: Tensor, a_des: Tensor) -> dict[str, Tensor]:
        """The charter's two plan variables, read off the shipped cascade's own expressions, for the
        smoke and the diagnostics. Nothing here feeds the control."""
        f_des = self.mass * (a_des + torch.tensor(
            [0.0, 0.0, self.gravity], device=a_des.device, dtype=a_des.dtype
        ))
        b3 = self.system.thrust_axis(x)
        return {
            "f_des": f_des,
            "total_thrust_T": torch.clamp((f_des * b3).sum(dim=-1), min=0.0),
            "b3_des": f_des / torch.clamp(torch.linalg.norm(f_des, dim=-1, keepdim=True), min=1e-9),
        }

    # ---- the record a cell JSON carries ----------------------------------------------------------
    def cascade_record(self) -> dict[str, Any]:
        return {
            "variant": "cascaded",
            "variant_status": (
                "SEPARATE BASELINE. This row is CASCADED MPPI, not vanilla MPPI. It must never be "
                "tabulated beside a vanilla row without an explicit variant column and its numbers "
                "must never be blended with vanilla MPPI numbers."
            ),
            "plan": self.plan_mode,
            "plan_dim": self.plan_dim,
            "plan_variable": (
                "desired WORLD acceleration a_des in R^3, one entry per decision step. The charter's "
                "(total thrust T, desired attitude q_des) are read off the shipped cascade: "
                "T = clamp(f_des . b3, 0) and q_des's body-up axis = f_des / ||f_des||, with "
                "f_des = mass (a_des + gravity e3). Both are functions of the same f_des, and the "
                "shipped attitude error b3 x b3_des is yaw-blind, so a full q_des carries one degree "
                "of freedom this inner loop cannot consume."
            ),
            "inner_loop": (
                "System.lqr_action (src/envs/quadrotor_3d.py:173-189) — the PD attitude controller "
                "src/common/filter_backup.py names in its first sentence and reaches through "
                "filter_backup.override_gains. REUSED, not re-implemented: the thrust projection, the "
                "b3 x b3_des attitude error, the inner PD, the mixer inverse and the per-rotor box "
                "clip are the shipped lines."
            ),
            "outer_loop_wrap": (
                "override_gains(system, kd_pos=0.0) + pseudo-goal p + a_des / kp_pos; the damping term "
                "is switched off because the plan supplies the TOTAL desired acceleration, and the "
                "pseudo-goal inverts the remaining -kp_pos (p - goal) term exactly"
            ),
            "gains_read_from_config": {
                "source": "config['lqr'][run.system], via the System object; nothing typed here",
                "kp_pos": self.kp_pos, "kd_pos_shipped": self.kd_pos,
                "kd_pos_used_by_the_wrap": 0.0,
                "kp_att": self.kp_att, "kd_att": self.kd_att,
                "attitude_gains_touched": False,
            },
            "rotor_dim": self.rotor_dim,
            "u_bounds": self.system.u_bounds.tolist(),
            "channel_scale_mode": self.channel_scale_mode,
            "collective_row_norm": self.collective_row_norm,
            "mass": self.mass,
            "gravity": self.gravity,
            "sigma_per_rotor_equivalent_N": float(self.params.sigma),
            "sigma_per_axis_m_s2": self.sigma_axis,
            "sigma_expression": "sigma * ||mixer_row_0|| / mass",
            "anchor": [float(v) for v in self.anchor.tolist()],
            "anchor_note": "hover in acceleration space is EXACTLY the zero vector, so the hover-centred "
                           "decomposition and the braking tail carry over with a zero anchor",
            "effective_noise": (
                "the RAW OU perturbation. The plan is not clipped in plan space, so the sequence "
                "actually simulated is U + eps and the honest effective perturbation is eps itself; "
                "the per-rotor box clip lives inside the shipped cascade, downstream of the plan."
            ),
            "unchanged_from_base": [
                "cost (the same stage_cost / terminal_cost / collision_mask objects and CostParams)",
                "plant (rk4_step on the shared deployed map)",
                "OU noise structure, correlation_steps and alpha",
                "relative lambda rule",
                "deployed terminal predicate and the shared eval path",
                "degenerate rule, control hold, receding-horizon shift, hover centring",
            ],
            "recovery_and_v4_switches": {
                "B1": bool(self.recovery.b1_enabled) if self.recovery is not None else False,
                "B2": bool(self.recovery.b2_enabled) if self.recovery is not None else False,
                "B3": bool(self.recovery.b3_enabled) if self.recovery is not None else False,
                "G1": bool(self.cost.g1_enabled), "G2": bool(self.cost.g2_enabled),
                "G3": bool(self.cost.g3_enabled), "G4": bool(self.cost.g4_enabled),
                "B2_reachable": False,
                "B2_note": "structurally unreachable: B2 seeds a body-wrench entry and this plan "
                           "carries an acceleration. Construction raises if it is switched on.",
            },
        }

    def sampler_record(self) -> dict[str, Any]:
        """The vanilla record indexes the wrench allocation of a 4-D anchor, which does not exist
        here, so the cascaded record is written directly. Every field a cell JSON consumes is present.
        """
        return {
            "space": "accel_world (CASCADED — not the vanilla wrench plan)",
            "noise": self.params.noise,
            "lam_mode": self.params.lam_mode,
            "sigma_per_rotor_equivalent_N": float(self.params.sigma),
            "channel_scale_mode": self.channel_scale_mode,
            "channel_scale": [float(v) for v in self.sigma_channel.tolist()],
            "sigma_per_channel": [float(v) for v in self.sigma_channel.tolist()],
            "channel_names": ["a_x", "a_y", "a_z"],
            "ou": {
                "correlation_steps": float(self.params.ou_correlation_steps),
                "alpha": float(self.params.ou_alpha),
                "stationary": True,
                "axis": "horizon",
            },
            "trim_mode": "hover is the zero acceleration",
            "trim_wrench": [float(v) for v in self.trim_wrench.tolist()],
            "trim_rotor_per_rotor": float(self.trim_rotor[0].item()),
            "center": self.params.center,
            "center_decomposition": "u = u_hover + u_plan + eps with u_hover = 0 in acceleration space",
            "anchor": [float(v) for v in self.anchor.tolist()],
            "plan_trim": [float(v) for v in self.plan_trim.tolist()],
            "control_hold_m": self.control_hold,
            "horizon_decision_entries": int(self.params.horizon),
            "rollout_physical_steps": self.lookahead_steps,
            "effective_lookahead_s": self.lookahead_s,
            "dt": self.dt,
            "hold_semantics": (
                "the MPPI optimisation and the receding-horizon shift run at DECISION steps "
                "(t % m == 0); on a hold step the latched rotor action is re-applied"
            ),
            "terminal_cost_mode": self.cost.terminal_mode,
            "goal_v4": self.cost.goal_v4_record(),
            "terminal_radii_read_from_config": {
                "goal_radius": self.cost.goal_radius,
                "goal_speed_radius": self.cost.goal_speed_radius,
                "goal_angrate_radius": self.cost.goal_angrate_radius,
            },
            "mass": self.mass,
            "gravity": self.gravity,
            "mixer": [[float(v) for v in row] for row in self.mixer.double().tolist()],
            "mixer_inv": [[float(v) for v in row] for row in self.mixer_inv.double().tolist()],
            "lam_eps_abs": float(self.params.lam_eps_abs),
            "std_estimator": "unbiased (N-1 denominator), torch.Tensor.std default",
            "recovery": (
                self.recovery.record() if self.recovery is not None
                else {"present": False, "note": "B1/B2/B3 all off"}
            ),
            "cascade": self.cascade_record(),
        }

    # ---- the rollout: the ONLY lines that differ from vanilla are the two control lines -----------
    @torch.no_grad()
    def _rollout_chunk(
        self,
        x: Tensor,
        sampled: Tensor,
        goal: Tensor,
        centers: Tensor,
        radii: Tensor,
        active: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """One sample-chunk. `sampled` is [B, n_sub, H, 3] of PLANNED ACCELERATIONS.

        Structurally the vanilla `_rollout_chunk`: the same freeze-on-collision rule, the same
        charged-once C_crash, the same per-physical-step `stage_cost` with `x_prev`, the same
        `terminal_cost`, the same endpoint probe. The difference is that the plant's input is produced
        INSIDE the loop by the shipped cascaded PD from the sample's CURRENT state, because the inner
        loop is state-dependent — that is exactly what makes this a cascade and not vanilla MPPI.
        """
        batch, n, horizon, plan_dim = sampled.shape
        flat = batch * n
        state = x.unsqueeze(1).expand(batch, n, -1).reshape(flat, -1)
        goal_flat = goal.unsqueeze(1).expand(batch, n, -1).reshape(flat, -1)
        cost = torch.zeros(flat, device=self.device, dtype=self.dtype)
        dead = torch.zeros(flat, dtype=torch.bool, device=self.device)

        for k in range(horizon * self.control_hold):
            a_des = self.held_control(sampled, k).reshape(flat, plan_dim)
            control = self.cascade_action(state, a_des)           # THE SHIPPED CASCADED PD
            nxt = rk4_step(self.system, state, control, self.dt)  # THE SHARED DEPLOYED PLANT MAP
            previous = state
            state = torch.where(dead.unsqueeze(-1), state, nxt)   # freeze collided samples
            collided = collision_mask(
                self.system.position(state).view(batch, n, -1), centers, radii, active, self.cost
            ).reshape(flat)
            newly = collided & ~dead
            cost = cost + newly.to(self.dtype) * self.cost.c_crash
            dead = dead | collided
            cost = cost + stage_cost(
                self.system, state, goal_flat, self.cost, self.recovery, x_prev=previous
            )

        cost = cost + terminal_cost(self.system, state, goal_flat, self.cost)
        if self.endpoint_probe:
            with torch.random.fork_rng(devices=self._fork_devices):
                self._endpoint_chunk_dists.append(
                    torch.linalg.norm(self.system.position(state) - goal_flat, dim=-1).view(batch, n)
                )
        return cost.view(batch, n), dead.view(batch, n)

    # ---- one control step ------------------------------------------------------------------------
    @torch.no_grad()
    def act(self, x: Tensor, scene: Any) -> Tensor:
        """One CONTROL step; returns the applied PER-ROTOR action [B, rotor_dim].

        The weighting, the relative-lambda rule, the ESS definition, the degenerate rule, the hold
        schedule and the receding-horizon shift are the vanilla ones. Two things differ, both stated
        in the module docstring: the sampled sequences are accelerations (so no allocation and no
        re-projection happens in plan space, and the effective perturbation is the raw noise), and the
        applied action is produced by the shipped cascaded PD from the CURRENT state.

        B3 is off in every v5 cell, so no adaptive-temperature branch is written here; the pre/post
        adaptation diagnostics the harness reads are filled with the single set of weights actually
        used, which is what B3-off means.
        """
        batch = int(x.shape[0])
        if self.plan is None or self.plan.shape[0] != batch:
            self.reset(batch)
        assert self.plan is not None and self.degenerate_steps is not None

        if self.steps_since_reset % self.control_hold != 0:
            assert self.last_action is not None
            self.last_decision = False
            self.steps_since_reset += 1
            if self.endpoint_probe:
                self._endpoint_current.append(np.full(batch, np.nan, dtype=np.float32))
            return self.last_action.clone()

        x = x.to(device=self.device, dtype=self.dtype)
        centers, radii, active = self._scene_tensors(scene)
        goal = scene_goal_tensor(scene, x)                                   # [B,3]

        if self.steps_since_reset == 0:
            self.spawn_tilt_deg = tilt_deg_from_cos(tilt_cos(self.system, x))

        base = self.absolute(self.plan)                                      # [B,H,3] a_des
        noise = self._draw_noise(batch)                                      # [B,N,H,3] OU, inherited
        sampled = base.unsqueeze(1) + noise                                  # [B,N,H,3]
        # The plan is NOT clipped in plan space, so the sequence simulated IS base + noise and the
        # honest effective perturbation is the noise itself (see the module docstring).
        cost, all_collided = self._rollout_cost(x, sampled, goal, centers, radii, active)
        del sampled

        if self.endpoint_probe:
            with torch.random.fork_rng(devices=self._fork_devices):
                assert self.last_endpoint_dist is not None
                self.last_best_endpoint_dist = self.last_endpoint_dist.gather(
                    1, cost.argmin(dim=1, keepdim=True)
                ).squeeze(1)
                self._endpoint_current.append(
                    self.last_best_endpoint_dist.detach().to(torch.float32).cpu().numpy().copy()
                )

        shifted = cost - cost.min(dim=1, keepdim=True).values
        if self.params.lam_mode == "absolute":
            lam_eff: Any = self.params.lam
            lam_eff_row = torch.full(
                (batch,), float(self.params.lam), device=self.device, dtype=self.dtype
            )
        else:
            lam_eff = torch.clamp(
                float(self.params.lam) * cost.std(dim=1, keepdim=True),
                min=float(self.params.lam_eps_abs),
            )
            lam_eff_row = lam_eff.squeeze(1)
        weight = torch.exp(-shifted / lam_eff)
        partition = weight.sum(dim=1, keepdim=True)
        degenerate = all_collided | ~torch.isfinite(partition).squeeze(1) | (partition.squeeze(1) <= 0.0)
        weight = weight / torch.clamp(partition, min=torch.finfo(self.dtype).tiny)
        ess = 1.0 / torch.clamp(weight.square().sum(dim=1), min=torch.finfo(self.dtype).tiny)

        update = self.plan + (weight.unsqueeze(-1).unsqueeze(-1) * noise).sum(dim=1)
        del noise
        plan = torch.where(degenerate.view(-1, 1, 1), self.plan, update)

        head = self.absolute(plan[:, 0])                                     # [B,3] a_des to apply
        action = self.cascade_action(x, head)                                # [B,rotor_dim], box-clipped
        self.plan = self._shift(plan)
        self.last_degenerate = degenerate
        self.last_ess = ess
        self.last_ess_pre = ess
        self.last_lam_eff_pre = lam_eff_row
        self.last_b3_event = torch.zeros(batch, dtype=torch.bool, device=self.device)
        self.last_lam_eff = lam_eff_row
        self.last_action = action
        self.last_decision = True
        self.steps_since_reset += 1
        self.degenerate_steps = self.degenerate_steps + degenerate.long()
        assert self.b3_events is not None
        return action


# =================================================================================================
# CPU SMOKE — run BEFORE the Stage-2 screen, exactly as the charter orders
# =================================================================================================

def _level_state(system: System, position: Tensor, dtype: torch.dtype) -> Tensor:
    """A level, motionless state at `position`: identity quaternion (w, x, y, z) = (1, 0, 0, 0), zero
    linear velocity, zero body rate. Built through `system.wrap_state`, so it is normalised exactly as
    the plant normalises every state it integrates."""
    batch = position.shape[0]
    quat = torch.zeros(batch, 4, dtype=dtype)
    quat[:, 0] = 1.0
    rest = torch.zeros(batch, 6, dtype=dtype)
    return system.wrap_state(torch.cat([position, quat, rest], dim=1))


def _two_phase_profile(offset: Tensor, seconds: float, dt: float) -> Tensor:
    """The OPEN-LOOP straight-line acceleration profile: +A for the first half, -A for the second, with

        A = 4 * offset / seconds^2

    the amplitude for which an exact double integrator starting at rest covers `offset` and arrives
    with zero terminal velocity (displacement A T^2/8 in each half, terminal velocity A T/2 - A T/2).
    Returned as [n_steps, 3]. Nothing about the vehicle enters it: it is fixed at t = 0 and replayed
    without ever reading the state, which is what makes the outer loop OPEN.
    """
    n_steps = int(round(float(seconds) / float(dt)))
    amplitude = 4.0 * offset / (float(seconds) ** 2)
    sign = torch.ones(n_steps, dtype=offset.dtype)
    sign[n_steps // 2:] = -1.0
    return sign.view(-1, 1) * amplitude.view(1, -1)


@torch.no_grad()
def cascade_smoke(
    system: System,
    config: Mapping[str, Any],
    controller: "CascadeController",
    smoke_config: Mapping[str, Any],
) -> dict[str, Any]:
    """The charter's Stage-2 smoke, on CPU. MEASUREMENT ONLY — no threshold is registered and no cell
    is selected; every check reports the number it measured and the deployed radius it is read against.

    (a) HOVER FIXED POINT. A level, motionless state is driven with the plan's hover command (a_des = 0)
        for `hover_steps` steps through `cascade_action` + `rk4_step`. Reports the drift in position,
        speed, body rate and tilt.
    (b) STRAIGHT LINE, OPEN LOOP. An obstacle-free straight-line scene driven by the two-phase
        acceleration profile above, fixed at t = 0 and replayed without reading the state. Reports the
        closest approach and whether it is inside `env.goal_radius` (READ from the merged config).
    (c) STRAIGHT LINE, CLOSED OUTER LOOP. The same scene with a_des taken from the shipped outer law
        evaluated on the current state. Supporting evidence beside (b), never a substitute for it.
    (d) THE WRAP IS EXACT. For random states and a random goal, the acceleration the SHIPPED outer law
        produces is fed back through `cascade_action`, and the rotor forces are compared against
        `system.lqr_action(x, goal)` itself. Agreement to floating point is the statement that the
        cascade reuses the shipped controller rather than re-implementing it; the comparison is
        non-circular because the reference is the shipped call with the shipped gains.
    (e) HOVER COMMAND. `cascade_action` at the hover state with a_des = 0 against the hover trim
        `mass * gravity / rotor_dim` per rotor, both read off the system object.
    """
    dtype = controller.dtype
    dt = float(config["env"]["dt"])
    goal_radius = float(config["env"]["goal_radius"])
    generator = torch.Generator().manual_seed(int(controller.params.seed))

    # ---- (a) hover fixed point ------------------------------------------------------------------
    hover_steps = int(smoke_config["hover_steps"])
    x = _level_state(system, torch.zeros(1, 3, dtype=dtype), dtype)
    x0 = x.clone()
    zero_accel = torch.zeros(1, controller.plan_dim, dtype=dtype)
    drift_p, drift_v, drift_w, drift_tilt = [], [], [], []
    for _ in range(hover_steps):
        x = rk4_step(system, x, controller.cascade_action(x, zero_accel), dt)
        drift_p.append(float(torch.linalg.norm(system.position(x) - system.position(x0)).item()))
        drift_v.append(float(system.speed(x).item()))
        drift_w.append(float(system.angular_rate(x).item()))
        drift_tilt.append(float(tilt_deg_from_cos(tilt_cos(system, x)).item()))
    hover = {
        "steps": hover_steps,
        "command": "a_des = 0, the plan's hover entry",
        "max_position_drift_m": max(drift_p),
        "max_speed_m_s": max(drift_v),
        "max_angular_rate_rad_s": max(drift_w),
        "max_tilt_deg": max(drift_tilt),
        "final_position_drift_m": drift_p[-1],
        "rotor_command_N": [float(v) for v in controller.cascade_action(x0, zero_accel)[0].tolist()],
        "hover_trim_per_rotor_N": float(controller.trim_rotor[0].item()),
    }

    # ---- (b) / (c) straight line, no obstacle ----------------------------------------------------
    line = smoke_config["straight_line"]
    offset = torch.tensor([float(v) for v in line["offset"]], dtype=dtype)
    seconds = float(line["seconds"])
    start = torch.zeros(1, 3, dtype=dtype)
    goal = (start + offset.view(1, 3))
    profile = _two_phase_profile(offset, seconds, dt)                       # [n_steps, 3], OPEN LOOP

    x = _level_state(system, start.clone(), dtype)
    open_dist = []
    for k in range(profile.shape[0]):
        x = rk4_step(system, x, controller.cascade_action(x, profile[k].view(1, 3)), dt)
        open_dist.append(float(torch.linalg.norm(system.position(x) - goal).item()))
    open_loop = {
        "mode": "OPEN LOOP in the outer loop: the a_des sequence is fixed at t = 0 and never "
                "re-computed from the state; the inner attitude PD is closed, which is the cascade's "
                "own structure",
        "profile": "two-phase, amplitude 4 * offset / seconds^2",
        "offset_m": [float(v) for v in offset.tolist()],
        "seconds": seconds,
        "n_steps": int(profile.shape[0]),
        "goal_radius_read_from_config": goal_radius,
        "closest_approach_m": min(open_dist),
        "final_distance_m": open_dist[-1],
        "reaches_position_radius": bool(min(open_dist) <= goal_radius),
        "final_speed_m_s": float(system.speed(x).item()),
    }

    x = _level_state(system, start.clone(), dtype)
    closed_steps = int(round(2.0 * seconds / dt))
    closed_dist = []
    for _ in range(closed_steps):
        # the SHIPPED outer law, evaluated on the current state and pushed through the same wrap
        a_des = -system.kp_pos * (system.position(x) - goal) - system.kd_pos * (x[:, 7:10])
        x = rk4_step(system, x, controller.cascade_action(x, a_des), dt)
        closed_dist.append(float(torch.linalg.norm(system.position(x) - goal).item()))
    closed_loop = {
        "mode": "CLOSED outer loop — supporting evidence beside the open-loop check, never a "
                "substitute for it",
        "n_steps": closed_steps,
        "closest_approach_m": min(closed_dist),
        "final_distance_m": closed_dist[-1],
        "reaches_position_radius": bool(min(closed_dist) <= goal_radius),
        "final_speed_m_s": float(system.speed(x).item()),
    }

    # ---- (d) the wrap is exact -------------------------------------------------------------------
    n_probe = 256
    probe = torch.randn(n_probe, system.state_dim, generator=generator, dtype=dtype)
    probe = system.wrap_state(probe)
    probe_goal = torch.randn(n_probe, 3, generator=generator, dtype=dtype)
    a_shipped = -system.kp_pos * (system.position(probe) - probe_goal) - system.kd_pos * probe[:, 7:10]
    reference = system.lqr_action(probe, probe_goal)
    through_wrap = controller.cascade_action(probe, a_shipped)
    wrap = {
        "what": "the SHIPPED outer law's own acceleration, fed back through cascade_action, against "
                "system.lqr_action itself — non-circular, because the reference is the shipped call "
                "with the shipped gains",
        "n_probe_states": n_probe,
        "max_abs_rotor_difference_N": float((through_wrap - reference).abs().max().item()),
        "max_abs_rotor_force_N": float(reference.abs().max().item()),
        "exact_to_float": bool(torch.equal(through_wrap, reference)),
        "gains_used": {"kp_pos": system.kp_pos, "kd_pos_shipped": system.kd_pos,
                       "kd_pos_in_wrap": 0.0, "kp_att": system.kp_att, "kd_att": system.kd_att},
    }

    # ---- (e) hover command -----------------------------------------------------------------------
    hover_cmd = controller.cascade_action(_level_state(system, torch.zeros(1, 3, dtype=dtype), dtype),
                                          zero_accel)
    trim = controller.mass * controller.gravity / float(controller.rotor_dim)
    hover_command = {
        "rotor_command_N": [float(v) for v in hover_cmd[0].tolist()],
        "expected_trim_per_rotor_N": trim,
        "max_abs_difference_N": float((hover_cmd - trim).abs().max().item()),
        "mass_and_gravity_read_from_config": {"mass": controller.mass, "gravity": controller.gravity},
    }

    return {
        "what": "charter-'v5' Stage 2 CASCADED MPPI — CPU smoke. Measurement only; no threshold is "
                "registered and no cell is selected.",
        "device": "cpu",
        "dtype": str(dtype),
        "dt_read_from_config": dt,
        "a_hover_fixed_point": hover,
        "b_straight_line_open_loop": open_loop,
        "c_straight_line_closed_outer_loop": closed_loop,
        "d_wrap_reproduces_shipped_controller": wrap,
        "e_hover_command": hover_command,
        "cascade": controller.cascade_record(),
    }
