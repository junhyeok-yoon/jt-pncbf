"""v2.8.4 — LITERATURE-STANDARD QUADROTOR MPPI over (collective thrust T, desired body rates omega_des).

=================================================================================================
WHAT THIS BASELINE IS — the statement that must accompany every number it produces
=================================================================================================

Literature-standard quadrotor MPPI (L1-MPPI, Pravitra et al. 2020; PA-MPPI, RA-L 2026): the planner
samples in (collective thrust T, desired body rates omega_des) — attitude dynamics are delegated to an
inner loop, exactly as those works prescribe. Two asymmetries vs our method, both stated wherever this
baseline is reported: (1) privileged full-state + obstacle-field access; (2) the per-rotor box is NOT
handled by the planner — the inner loop allocates rotor thrusts and clips to [0, 4.905], so actuator
constraints are respected only after allocation, not during planning. The rotor-direct variant (v1-v5,
reach 0.0000 across 25+ configurations and N up to 8192) remains the same-interface comparison and is
reported as its own row.

The bracketed box above is the charter's own text; in every machine-readable record this file emits the
same sentence is formatted with the interval READ OFF `system.u_bounds`, so the words and the numbers
cannot drift apart.

    THE ASYMMETRY IS STRUCTURAL HERE, NOT A SOFTENING. The planner's rollout never forms a rotor
    command at all: it propagates translational dynamics + attitude kinematics driven by the sampled
    (T, omega_des) and has no per-rotor variable to bound. The rotor forces come into existence only
    when the executed command passes through the inner loop, where the mixer inverse allocates them and
    the per-rotor clip is applied. `rate_cascade_smoke` check (c) MEASURES that sampled plans do map to
    pre-clip rotor commands outside the box and reports the count with its denominator; every cell also
    reports the same count over its own executed commands.

=================================================================================================
HOW THIS DIFFERS FROM `cascade.py` (the charter-"v5" Stage-2 cascade) — READ BEFORE REUSING CODE
=================================================================================================

They are DIFFERENT INTERFACES and are separate baseline rows.

  * `cascade.py`'s plan carries the desired WORLD ACCELERATION a_des in R^3, and (T, q_des) are read
    off `System.lqr_action`'s own lines because that shipped outer law derives both from one world-force
    vector and is yaw-blind. Its rollout integrates THE FULL PLANT (`rk4_step` on rotor thrusts) with
    the shipped cascaded PD closed inside the loop, so a rotor command — and the per-rotor clip — exists
    at every rollout step.
  * THIS file's plan carries (T, w_x, w_y, w_z): a collective thrust plus DESIRED BODY RATES. The
    planner propagates ATTITUDE KINEMATICS (dot q = 1/2 q (x) (0, omega)) with the sampled rate as the
    input — attitude DYNAMICS are delegated to the inner loop and do not appear in the prediction — and
    the inner loop is a body-RATE tracking law, not an attitude-error PD around a desired attitude.

Neither the plan variable nor the outer law of `cascade.py` is reused. What IS reused: the cost objects,
the collision predicate, the OU sampler, the relative-lambda rule, the hover-centred decomposition, the
degenerate rule, the receding-horizon shift, the chunking, and the shared eval path.

CONTEXT, MEASURED AND NOT RE-DERIVED HERE: the v5 a_des cascade scored reach 0.0000 on all four cells at
n = 400 while driving omega_p50 from ~4.89 rad/s down to 0.20-0.29 — it removed the rate saturation the
diagnosis recorded and reach stayed exactly zero. A reach of 0.0000 from this (T, omega) cascade is
CONSISTENT WITH THAT RECORD, not a new discovery.

=================================================================================================
THE PLANNER'S PREDICTION MODEL (charter item 1) — L1-MPPI's form
=================================================================================================

`_RatePlannerModel.dynamics` is the shipped plant's own `dynamics` with exactly two edits:

    dot p     = v                                            (unchanged)
    dot q     = 1/2 q (x) (0, omega)                         (unchanged — the same `_quat_mul`/`_pure_quat`)
    dot v     = (T / m) R(q) e3 - g e3                       (the plant's line with `f_thr` := T, i.e.
                                                              WITHOUT the `u @ mixer.T` allocation)
    dot omega = 0                                            (the plant's `J^-1 (tau - omega x J omega)`
                                                              REMOVED: attitude dynamics are delegated)

`R(q) e3` is `system.thrust_axis(x)` and m, g are `system.mass` / `system.gravity` — read off the system
object, never typed. `wrap_state` is delegated to the real system, so the planner's quaternion
renormalisation, its ||v|| <= v_max clamp and its |omega| <= omega_max clamp are the plant's own.

Because attitude dynamics are delegated, the commanded rate IS the predicted rate: at every physical
rollout step the state's omega slot is overwritten with the (low-pass filtered, rate-limited) command
before the step is taken. That is what "delegated to an inner loop" means, and it is also what makes the
cost's own ||omega|| terms — which the deployed reach predicate needs — read the commanded rate.

LOW-PASS ON THE SAMPLED RATES (charter item 1, "their stated correction for discontinuous rate samples").
Before a sampled rate enters the kinematics it passes a first-order low-pass along the rollout axis,

    w_f[k] = beta * w_f[k-1] + (1 - beta) * w_cmd[k],      w_f[-1] := the CURRENT MEASURED body rate,

with `beta` read from config (`mppi.cascaded.rate_lowpass.beta`, asserted equal to exp(-1/steps) at
construction exactly as the OU pole is). Initialising the filter at the vehicle's own rate is what makes
the filtered command causal and continuous with the state the controller was handed; the same recursion
produces the EXECUTED command from the head entry, so the plan's first entry and the rollout's first
step see the identical filtered rate. `rate_cascade_smoke` check (e) MEASURES that the lag-1
autocorrelation of the rate channels increases through the filter.

RATE LIMIT (charter item 4). The sampled rate channels are clamped to +/- `system.omega_max`, the plant's
own representable range (the bound `system.wrap_state` enforces), read off the system object. THE
COLLECTIVE CHANNEL IS NOT CLAMPED: bounding T to its achievable interval would be planner-side handling
of an actuator constraint, which this baseline does not do and does not claim.

=================================================================================================
THE INNER LOOP (charter item 2) — body-rate P tracking, then the standard mixer, then the box clip
=================================================================================================

    tau      = J * k_rate * (omega_des - omega)                 J = `system.inertia`
    wrench   = (T, tau)
    f_rotor  = wrench @ system.mixer_inv.T                      the standard X-mixer allocation, the same
                                                                expression `quadrotor_3d.lqr_action:188`
                                                                allocates with (`MPPIController.allocate`)
    u        = clip(f_rotor, system.u_bounds)                    THE PER-ROTOR BOX, APPLIED HERE AND ONLY HERE

`k_rate` is the ATTITUDE LOOP'S OWN RATE GAIN, `system.kd_att`, read from `config["lqr"][run.system]` —
the gains `src/common/filter_backup.py` names in its docstring and reaches through `override_gains` —
multiplied by the screen's own factor (1 = the backup-CBF value, 2 = twice it). The form is the shipped
one restricted to the rate leg: `System.lqr_action` computes `tau = J (kp_att e_att_body - kd_att omega)`,
so at `omega_des = 0` and factor 1 this law is that expression's rate term byte for byte. Nothing is
re-tuned beyond the screen's factor, no gyroscopic feed-forward is added (the shipped loop carries none),
and the attitude-error leg has no counterpart here because a rate command replaces the desired attitude.

The arm length, the torque constant, the mixer entries, the inertia, the box, dt, omega_max, m, g and the
rate gain are ALL read from the system object or the config. Nothing above is typed.

=================================================================================================
WHAT IS UNCHANGED FROM THE v3 BASE (charter item 3)
=================================================================================================

  * COST — `stage_cost` / `terminal_cost` / `collision_mask` imported from `cost.py` and called with the
    same `CostParams`: goal distance, the settling terminal with the DEPLOYED terminal's constants (all
    three radii read from the effective env config), and the collision indicator on the SAME predicate
    the eval harness resolves outcomes with.
  * PLANT — the full plant (RK4, `env.dt`, rotor thrusts) remains the SIMULATION plant everywhere. The
    cascade lives inside the controller only; `src.eval.evaluate` integrates `rk4_step(system, x, u, dt)`
    on the executed per-rotor action exactly as for every other arm.
  * SAMPLER — the inherited stationary OU noise along the horizon, the relative-lambda rule, the ESS
    definition, the degenerate rule, the control hold, the receding-horizon shift and the hover-centred
    decomposition u = u_hover + u_plan + eps.
  * HOVER CENTRING is retained ON T: the anchor is (T_hover, 0, 0, 0) with T_hover = m * g read from the
    system object, and the zero body rate is hover in the three rate channels.

CHANNEL SCALES, a recorded CHOICE. `sigma` stays the base cell's value in PER-ROTOR-EQUIVALENT NEWTONS.
The collective channel keeps exactly the marginal std vanilla gives its F_total channel,
`sigma * ||mixer_row_0||`. The three rate channels take the RATE CHANGE the vanilla torque noise of that
channel produces in one control step,

    sigma_omega_j = sigma * ||mixer_row_{j+1}|| / J_j * dt,

every factor read off the system object or the config. It is a choice — a torque std has no rate
counterpart without a timescale — so it is recorded per cell as `channel_scale_mode` rather than buried.

B1/B2/B3 and G1-G4 are OFF, as the base configuration requires. B2 is additionally UNREACHABLE: it seeds
a body-wrench entry into the plan and this plan carries no wrench. Construction raises if it is on.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import torch

from src.common.observation import scene_goal_tensor
from src.common.rk4 import rk4_step
from src.common.system import System
from src.envs.quadrotor_3d import _pure_quat, _quat_mul
from src.frameworks.mppi.cost import CostParams, collision_mask, stage_cost, terminal_cost
from src.frameworks.mppi.mppi_controller import MPPIController, MPPIParams
from src.frameworks.mppi.recovery import RecoveryParams, tilt_cos, tilt_deg_from_cos


Tensor = torch.Tensor

PLAN_MODES = ("thrust_bodyrate",)
CHANNEL_SCALE_MODES = ("from_mixer_rows_over_inertia_times_dt",)

# The state layout of quadrotor_3d, as its own module docstring writes it:
#   x = [px,py,pz, qw,qx,qy,qz, vx,vy,vz, wx,wy,wz]
# These are SLICE INDICES into that layout, not physical constants.
_Q = slice(3, 7)
_V = slice(7, 10)
_W = slice(10, 13)


def cascade_sigma_channel(config: Any) -> list[float] | None:
    """v2.8.4 STAGE 3 — read a PER-CHANNEL sampling std off the config. DEFAULT OFF.

    `mppi_cascade.sigma_channel`, when present, is a 4-vector in the plan's OWN units,
    `[T (N), w_x, w_y, w_z (rad/s)]`, and REPLACES the plant-derived `sigma_channel` built in
    `__init__` from `sigma * [||mixer_row_0||, ||mixer_rows_1:4|| / J * dt]`. The scalar `sigma`
    (`MPPIParams.sigma`) then has no effect on the sampler, and that is stated in the per-cell record.

    ABSENT — the shipped case, and every configuration that predates this key — returns None and the
    plant-derived construction runs unchanged and byte-identical. That absent case is the parity gate,
    exactly as `hazard.geom_form`'s 'clip' default is (`src/common/signed_h.py:hazard_geom`).
    """
    block: Mapping[str, Any] = {}
    if config is not None:
        try:
            block = config.get("mppi_cascade") or {}          # type: ignore[union-attr]
        except AttributeError:
            block = {}
    raw = block.get("sigma_channel", None)
    if raw is None:
        return None
    values = [float(v) for v in raw]
    if len(values) != 4:
        raise ValueError(
            "mppi_cascade.sigma_channel must have 4 entries [T, w_x, w_y, w_z], got "
            f"{len(values)}."
        )
    if any(v <= 0.0 for v in values):
        raise ValueError(f"mppi_cascade.sigma_channel entries must be positive, got {values}.")
    return values


class _RatePlannerModel:
    """The PLANNER's prediction model: translational dynamics + attitude kinematics, input (T, omega).

    It exposes the two methods `src.common.rk4.rk4_step` calls — `dynamics` and `wrap_state` — so the
    planner integrates through the SHIPPED RK4 and the SHIPPED state normalisation, and the only thing
    that differs from the plant is the model itself (see the module docstring for the four lines).

    THIS IS THE PLANNER'S MODEL, NOT THE SIMULATION PLANT. The simulation plant is the full RK4 rotor
    -thrust plant and is untouched; this object is never handed to the eval harness.
    """

    def __init__(self, system: System) -> None:
        self.system = system
        self.mass = float(system.mass)          # read off the system object
        self.gravity = float(system.gravity)    # read off the system object

    def dynamics(self, x: Tensor, u: Tensor) -> Tensor:
        """`u` is [B, 4] = (T, w_x, w_y, w_z). The rate enters through the STATE's omega slot (which the
        rollout overwrites with the filtered command before every step), so this reproduces the plant's
        own `dot q` line verbatim; `dot omega = 0` is the delegation."""
        q = x[:, _Q]
        v = x[:, _V]
        omega = x[:, _W]
        thrust = u[:, 0]
        g_vec = torch.zeros_like(v)
        g_vec[:, 2] = self.gravity
        dv = (thrust.unsqueeze(-1) / self.mass) * self.system.thrust_axis(x) - g_vec
        dq = 0.5 * _quat_mul(q, _pure_quat(omega))
        return torch.cat([v, dq, dv, torch.zeros_like(omega)], dim=1)

    def wrap_state(self, x: Tensor) -> Tensor:
        """The real system's own normalisation — quaternion renormalisation, ||v|| <= v_max,
        |omega| <= omega_max. Delegated, never re-implemented."""
        return self.system.wrap_state(x)


class RateCascadeController(MPPIController):
    """MPPI over (collective thrust T, desired body rates omega_des), closed by a body-rate tracking
    inner loop. See the module docstring — this is a SEPARATE baseline row and never a vanilla MPPI row,
    and it is NOT the charter-"v5" a_des cascade either."""

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
            raise ValueError(f"mppi.cascaded.plan must be one of {PLAN_MODES}, got {plan_mode!r}.")
        scale_mode = str(cascade["channel_scale"])
        if scale_mode not in CHANNEL_SCALE_MODES:
            raise ValueError(
                f"mppi.cascaded.channel_scale must be one of {CHANNEL_SCALE_MODES}, got {scale_mode!r}."
            )
        if params.space != "wrench":
            # the inherited OU sampler's non-legacy branch is the one this plan uses
            raise ValueError(
                "the (T, omega) cascade inherits the OU sampler, which requires "
                f"mppi.sampling.space == 'wrench'; got {params.space!r}."
            )
        if recovery is not None and recovery.b2_enabled:
            raise NotImplementedError(
                "B2 seeds a body-wrench entry into the carried plan; this plan carries "
                "(collective thrust, desired body rates) and has no wrench to seed. Switch B2 off."
            )
        self.plan_mode = plan_mode
        self.channel_scale_mode = scale_mode
        self.plan_dim = int(cascade["plan_dim"])
        if self.plan_dim != int(system.action_dim):
            raise ValueError(
                f"mppi.cascaded.plan_dim must equal the rotor dimension ({int(system.action_dim)}) so "
                f"the inherited plan bookkeeping is reused unchanged; got {self.plan_dim}."
            )
        self.rotor_dim = int(system.action_dim)

        # ---- the inner loop's gain: the ATTITUDE LOOP's own rate gain, read off the system object ----
        # `system.kd_att` comes from config["lqr"][run.system] — the gains src/common/filter_backup.py
        # names and reaches through `override_gains`. The screen's factor is the only thing that moves.
        self.rate_gain_factor = float(cascade["rate_gain_factor"])
        self.kd_att = float(system.kd_att)
        self.kp_att = float(system.kp_att)
        self.k_rate = self.kd_att * self.rate_gain_factor
        self.inertia = system.inertia.to(device=device, dtype=dtype)      # J, read off the system object

        # ---- the plant's representable rate range, read off the system object -----------------------
        self.omega_max = float(system.omega_max)

        # ---- the sampled-rate low-pass ---------------------------------------------------------------
        lowpass = cascade["rate_lowpass"]
        self.lowpass_steps = float(lowpass["steps"])
        self.lowpass_beta = float(lowpass["beta"])
        expected = math.exp(-1.0 / self.lowpass_steps)
        if abs(self.lowpass_beta - expected) > 1e-12:
            raise ValueError(
                f"mppi.cascaded.rate_lowpass.beta = {self.lowpass_beta!r} does not equal "
                f"exp(-1 / steps) = {expected!r} to 1e-12 (steps = {self.lowpass_steps!r})."
            )

        # ---- the planner's prediction model ----------------------------------------------------------
        self.planner = _RatePlannerModel(system)

        # ---- hover centring on T: the anchor is (m*g, 0, 0, 0), inherited from the base class ---------
        # `MPPIController.__init__` built `trim_wrench = (mass*gravity, 0, 0, 0)` from the system
        # constants and set the anchor / plan trim from it under `space == 'wrench'`. In THIS plan the
        # first channel is the collective thrust T and the other three are body rates, so that same
        # vector is exactly the hover command: T_hover = m * g and zero body rate. Nothing is re-derived.
        self.t_hover = float(self.trim_wrench[0].item())

        # ---- channel scales: collective as vanilla, rates as the one-step rate change of vanilla's
        #      torque noise. A CHOICE, recorded rather than buried (see the module docstring). ----------
        row_norms = torch.linalg.norm(self.mixer, dim=1)                  # [4] = {2, 2l, 2l, 2c}
        self.collective_row_norm = float(row_norms[0].item())
        sigma = float(params.sigma)
        rate_scale = row_norms[1:4] / self.inertia * self.dt              # [3] rad/s per unit sigma
        self.sigma_channel = torch.cat(
            [torch.full((1,), sigma * self.collective_row_norm, device=device, dtype=dtype),
             sigma * rate_scale.to(device=device, dtype=dtype)]
        )                                                                  # [4]

        # ---- v2.8.4 STAGE 3, DEFAULT OFF: the per-channel override ------------------------------------
        # `mppi_cascade.sigma_channel` ABSENT (every configuration that predates this key, and every
        # cell of sections 1-9) leaves `self.sigma_channel` exactly as constructed above, byte for byte.
        # PRESENT, it replaces the four entries with the configured 4-vector in the plan's own units and
        # the scalar `sigma` no longer reaches the sampler. Recorded per cell, never inferred.
        self.sigma_channel_override = cascade_sigma_channel(config)
        if self.sigma_channel_override is not None:
            self.sigma_channel = torch.tensor(
                self.sigma_channel_override, device=device, dtype=dtype
            )                                                              # [4]

        # ---- the asymmetry evidence, accumulated over the whole cell ----------------------------------
        # Counts of EXECUTED commands whose pre-clip rotor allocation left the per-rotor box. Not reset by
        # `reset()`, so a cell's counter spans its whole scored set.
        self.preclip_entries = 0
        self.preclip_out_of_box = 0
        self.preclip_min = float("inf")
        self.preclip_max = float("-inf")
        self.rate_limit_entries = 0
        self.rate_limit_clamped = 0

    # ---- the box interval, formatted from the system's own bounds --------------------------------
    def box_text(self) -> str:
        lo = float(self.u_lo.min().item())
        hi = float(self.u_hi.max().item())
        return f"[{lo:g}, {hi:g}]"

    def asymmetry_statement(self) -> str:
        """The charter's own sentence, with the interval READ OFF `system.u_bounds`."""
        return (
            "the per-rotor box is NOT handled by the planner — the inner loop allocates rotor thrusts "
            f"and clips to {self.box_text()}, so actuator constraints are respected only after "
            "allocation, not during planning"
        )

    # ---- the rate limit (charter item 4) ---------------------------------------------------------
    def rate_limit(self, plan: Tensor) -> Tensor:
        """Clamp the three RATE channels to +/- `system.omega_max`, the plant's own representable range.
        The COLLECTIVE channel is returned untouched: bounding T would be planner-side handling of an
        actuator constraint, which this baseline does not do."""
        thrust = plan[..., :1]
        rates = torch.clamp(plan[..., 1:], min=-self.omega_max, max=self.omega_max)
        return torch.cat([thrust, rates], dim=-1)

    # ---- the inner loop (charter item 2) ----------------------------------------------------------
    def inner_loop(self, x: Tensor, command: Tensor, *, count: bool = False) -> Tensor:
        """(T, omega_des) -> per-rotor forces. `x` is [M, state_dim], `command` is [M, 4]; returns
        [M, rotor_dim] AFTER the per-rotor box clip — which is the only place the box is applied.

        tau = J * k_rate * (omega_des - omega), then the standard X-mixer allocation and the box clip,
        both through `MPPIController.allocate`, which is `wrench @ system.mixer_inv.T` clamped to
        `system.u_bounds` — the same expression `quadrotor_3d.lqr_action:188-189` uses.
        """
        omega = x[..., _W]
        tau = self.inertia * (self.k_rate * (command[..., 1:] - omega))
        wrench = torch.cat([command[..., :1], tau], dim=-1)
        if count:
            # ASYMMETRY EVIDENCE. The pre-clip allocation is formed only to be MEASURED; the control
            # path below is `self.allocate(wrench)`, unchanged. Wrapped in `fork_rng` under the standing
            # rule for added instrumentation.
            with torch.random.fork_rng(devices=self._fork_devices):
                pre = wrench @ self.mixer_inv.t()
                out = (pre < self.u_lo) | (pre > self.u_hi)
                self.preclip_entries += int(pre.numel())
                self.preclip_out_of_box += int(out.sum().item())
                self.preclip_min = min(self.preclip_min, float(pre.min().item()))
                self.preclip_max = max(self.preclip_max, float(pre.max().item()))
        return self.allocate(wrench)

    def preclip_record(self) -> dict[str, Any]:
        """The out-of-box pre-clip count with its denominator, over the EXECUTED commands of this cell."""
        entries = int(self.preclip_entries)
        return {
            "what": "per-rotor entries of the pre-clip allocation `wrench @ system.mixer_inv.T` of every "
                    "EXECUTED command, counted against the box read from system.u_bounds",
            "n_entries": entries,
            "n_outside_box": int(self.preclip_out_of_box),
            "frac_outside_box": (float(self.preclip_out_of_box) / entries) if entries else float("nan"),
            "preclip_min_N": None if entries == 0 else self.preclip_min,
            "preclip_max_N": None if entries == 0 else self.preclip_max,
            "box_read_from_system": [float(self.u_lo.min().item()), float(self.u_hi.max().item())],
            "asymmetry": self.asymmetry_statement(),
            "rate_limit": {
                "n_entries": int(self.rate_limit_entries),
                "n_clamped": int(self.rate_limit_clamped),
                "frac_clamped": (
                    float(self.rate_limit_clamped) / float(self.rate_limit_entries)
                    if self.rate_limit_entries else float("nan")
                ),
                "omega_max_read_from_system": self.omega_max,
            },
        }

    # ---- the record a cell JSON carries -----------------------------------------------------------
    def cascade_record(self) -> dict[str, Any]:
        return {
            "variant": "cascaded_rate",
            "variant_status": (
                "SEPARATE BASELINE ROW. This is literature-standard quadrotor MPPI over "
                "(collective thrust T, desired body rates omega_des) — L1-MPPI / PA-MPPI's interface. "
                "It is NOT vanilla rotor-direct MPPI and it is NOT the charter-'v5' a_des cascade. Its "
                "numbers must never be blended with, averaged with or substituted for either, and it "
                "may never sit in a table with them without an explicit variant column."
            ),
            "reference": "L1-MPPI, Pravitra et al. 2020; PA-MPPI, RA-L 2026",
            "plan": self.plan_mode,
            "plan_dim": self.plan_dim,
            "plan_variable": (
                "v = [T, w_x, w_y, w_z]: the collective thrust and the DESIRED BODY RATES. Attitude "
                "dynamics are delegated to the inner loop, exactly as those works prescribe."
            ),
            "differs_from_v5_cascade": (
                "the charter-'v5' cascade (src/frameworks/mppi/cascade.py) plans the desired WORLD "
                "ACCELERATION a_des in R^3 and rolls THE FULL PLANT forward with the shipped attitude "
                "PD closed inside the loop. Here the plan is (T, omega_des), the planner propagates "
                "attitude KINEMATICS with the sampled rate as the input, and the inner loop is a "
                "body-RATE tracking law. Different interface, different prediction model, separate row."
            ),
            "planner_model": {
                "form": "translational dynamics + attitude kinematics (L1-MPPI's form)",
                "dot_p": "v",
                "dot_q": "0.5 * q (x) (0, omega) — the plant's own line, same _quat_mul/_pure_quat",
                "dot_v": "(T / mass) * system.thrust_axis(x) - gravity e3 — the plant's line with "
                         "f_thr := T (no `u @ mixer.T` allocation)",
                "dot_omega": "0 — attitude DYNAMICS are delegated to the inner loop",
                "omega_slot": "overwritten with the low-pass filtered, rate-limited command before every "
                              "physical rollout step, so the commanded rate IS the predicted rate and "
                              "the cost's ||omega|| terms read it",
                "integrator": "src.common.rk4.rk4_step, the shipped RK4, at env.dt",
                "wrap_state": "delegated to the real system (quaternion renorm, ||v||<=v_max, "
                              "|omega|<=omega_max)",
                "simulation_plant": "UNCHANGED — the full RK4 rotor-thrust plant remains the simulation "
                                    "plant everywhere; the cascade lives inside the controller only",
                "mass": self.mass, "gravity": self.gravity, "dt": self.dt,
            },
            "rate_lowpass": {
                "law": "w_f[k] = beta * w_f[k-1] + (1 - beta) * w_cmd[k]",
                "init": "the CURRENT MEASURED body rate of the state the controller was handed",
                "beta": self.lowpass_beta,
                "steps": self.lowpass_steps,
                "consistency": "beta == exp(-1 / steps), asserted at construction to 1e-12",
                "why": "L1-MPPI's stated correction for discontinuous rate samples",
            },
            "rate_limit": {
                "bound_read_from_system": self.omega_max,
                "applies_to": "the three RATE channels only",
                "collective_not_limited": (
                    "the collective channel is NOT clamped: bounding T to its achievable interval would "
                    "be planner-side handling of an actuator constraint, which this baseline does not do"
                ),
            },
            "inner_loop": {
                "law": "tau = J * k_rate * (omega_des - omega); wrench = (T, tau); "
                       "f_rotor = wrench @ system.mixer_inv.T; u = clip(f_rotor, system.u_bounds)",
                "k_rate": self.k_rate,
                "k_rate_source": (
                    "system.kd_att (config['lqr'][run.system], the gains src/common/filter_backup.py "
                    "names and reaches through override_gains) times the screen's factor"
                ),
                "kd_att_read_from_config": self.kd_att,
                "kp_att_read_from_config": self.kp_att,
                "rate_gain_factor": self.rate_gain_factor,
                "inertia_read_from_system": [float(v) for v in self.inertia.double().tolist()],
                "form_note": (
                    "the shipped attitude loop is tau = J (kp_att e_att_body - kd_att omega); this is "
                    "that expression's RATE leg with the setpoint moved to omega_des, so at "
                    "omega_des = 0 and factor 1 it is the shipped rate term byte for byte. No "
                    "gyroscopic feed-forward is added — the shipped loop carries none — and nothing is "
                    "re-tuned beyond the screen's factor."
                ),
                "allocation": "MPPIController.allocate — wrench @ system.mixer_inv.T, clamped to "
                              "system.u_bounds; the same expression quadrotor_3d.lqr_action:188-189 uses",
                "mixer": [[float(v) for v in row] for row in self.mixer.double().tolist()],
                "mixer_inv": [[float(v) for v in row] for row in self.mixer_inv.double().tolist()],
                "u_bounds": self.system.u_bounds.tolist(),
            },
            "asymmetries": {
                "privilege": (
                    "privileged full-state + obstacle-field access: the planner reads the full 13-D "
                    "state and the scene's obstacle field directly, which no learned arm does"
                ),
                "actuator_box": self.asymmetry_statement(),
                "box_handled_in_planner": False,
                "planner_forms_a_rotor_command": False,
                "where_the_box_is_applied": "inside the inner loop, after the mixer allocation, on the "
                                            "executed command only",
            },
            "channel_scale_mode": self.channel_scale_mode,
            "channel_names": ["T", "w_x", "w_y", "w_z"],
            "sigma_per_rotor_equivalent_N": float(self.params.sigma),
            "sigma_per_channel": [float(v) for v in self.sigma_channel.tolist()],
            "sigma_expression": (
                "collective: sigma * ||mixer_row_0||  (vanilla's own F_total scale); "
                "rates: sigma * ||mixer_row_{j+1}|| / J_j * dt  (the rate change vanilla's torque noise "
                "of that channel produces in one control step). A recorded CHOICE."
            ),
            "collective_row_norm": self.collective_row_norm,
            "sigma_channel_override": self.sigma_channel_override,
            "sigma_channel_override_note": (
                "null = ABSENT, the shipped plant-derived construction above, byte-identical. A "
                "4-vector = `mppi_cascade.sigma_channel` was set and REPLACED it; the scalar sigma "
                "then does not reach the sampler."
            ),
            "hover_centering": {
                "anchor": [float(v) for v in self.anchor.tolist()],
                "T_hover": self.t_hover,
                "T_hover_source": "mass * gravity, read off the system object",
                "note": "hover centring is retained ON T; zero body rate is hover in the three rate "
                        "channels, so the inherited (m*g, 0, 0, 0) anchor is exactly the hover command",
            },
            "effective_noise": (
                "the POST-RATE-LIMIT perturbation, `rate_limit(U + eps) - U`. The sequence actually "
                "simulated is the limited one, so this is the honest analogue of vanilla's post-clip "
                "re-projection. The per-rotor box does not enter it at all: no rotor command exists in "
                "plan space."
            ),
            "unchanged_from_base": [
                "cost (the same stage_cost / terminal_cost / collision_mask objects and CostParams)",
                "the settling terminal with the deployed terminal's constants, all radii config-read",
                "the collision indicator on the SAME predicate as the eval harness",
                "the full RK4 rotor-thrust plant as the SIMULATION plant",
                "OU noise structure, correlation_steps and alpha",
                "relative lambda rule, ESS definition, degenerate rule",
                "control hold, receding-horizon shift, hover centring",
            ],
            "recovery_and_v4_switches": {
                "B1": bool(self.recovery.b1_enabled) if self.recovery is not None else False,
                "B2": bool(self.recovery.b2_enabled) if self.recovery is not None else False,
                "B3": bool(self.recovery.b3_enabled) if self.recovery is not None else False,
                "G1": bool(self.cost.g1_enabled), "G2": bool(self.cost.g2_enabled),
                "G3": bool(self.cost.g3_enabled), "G4": bool(self.cost.g4_enabled),
                "B2_reachable": False,
                "B2_note": "structurally unreachable: B2 seeds a body-wrench entry and this plan carries "
                           "(T, omega). Construction raises if it is switched on.",
            },
            "preclip_evidence": self.preclip_record(),
        }

    def sampler_record(self) -> dict[str, Any]:
        """The vanilla record indexes a wrench anchor's rotor allocation, which is not what this anchor
        is; the record is therefore written directly. Every field a cell JSON consumes is present."""
        return {
            "space": "thrust_bodyrate (CASCADED (T, omega_des) — not the vanilla rotor/wrench plan)",
            "noise": self.params.noise,
            "lam_mode": self.params.lam_mode,
            "sigma_per_rotor_equivalent_N": float(self.params.sigma),
            "channel_scale_mode": self.channel_scale_mode,
            "channel_scale": [float(v) for v in self.sigma_channel.tolist()],
            "sigma_per_channel": [float(v) for v in self.sigma_channel.tolist()],
            "channel_names": ["T", "w_x", "w_y", "w_z"],
            "channel_units": ["N", "rad/s", "rad/s", "rad/s"],
            "sigma_channel_override": self.sigma_channel_override,
            "ou": {
                "correlation_steps": float(self.params.ou_correlation_steps),
                "alpha": float(self.params.ou_alpha),
                "stationary": True,
                "axis": "horizon",
            },
            "trim_mode": "hover is (m*g, 0, 0, 0) in (T, omega) space",
            "trim_wrench": [float(v) for v in self.trim_wrench.tolist()],
            "trim_rotor_per_rotor": float(self.trim_rotor[0].item()),
            "center": self.params.center,
            "center_decomposition": "u = u_hover + u_plan + eps with u_hover = (m*g, 0, 0, 0)",
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

    # ---- the rollout: the planner's own model, never the plant ------------------------------------
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
        """One sample-chunk. `sampled` is [B, n_sub, H, 4] of RATE-LIMITED (T, omega_des) entries.

        Structurally the vanilla `_rollout_chunk` — the same freeze-on-collision rule, the same
        charged-once C_crash, the same per-physical-step `stage_cost` with `x_prev`, the same
        `terminal_cost`, the same endpoint probe. What differs: the sampled rate passes the low-pass,
        the filtered rate is written into the state's omega slot, and the step is taken through the
        PLANNER'S model rather than the plant. NO ROTOR COMMAND IS FORMED HERE AND NO BOX IS APPLIED.
        """
        batch, n, horizon, plan_dim = sampled.shape
        flat = batch * n
        state = x.unsqueeze(1).expand(batch, n, -1).reshape(flat, -1)
        goal_flat = goal.unsqueeze(1).expand(batch, n, -1).reshape(flat, -1)
        cost = torch.zeros(flat, device=self.device, dtype=self.dtype)
        dead = torch.zeros(flat, dtype=torch.bool, device=self.device)
        beta = self.lowpass_beta
        # the low-pass state starts at the vehicle's OWN measured body rate
        rate_f = state[:, _W].clone()

        for k in range(horizon * self.control_hold):
            command = self.held_control(sampled, k).reshape(flat, plan_dim)
            rate_f = beta * rate_f + (1.0 - beta) * command[:, 1:]
            # the commanded rate IS the predicted rate: attitude dynamics are delegated
            state = torch.cat([state[:, :10], rate_f], dim=1)
            nxt = rk4_step(self.planner, state, command, self.dt)     # THE PLANNER'S OWN MODEL
            previous = state
            state = torch.where(dead.unsqueeze(-1), state, nxt)       # freeze collided samples
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
        schedule and the receding-horizon shift are the vanilla ones. What differs: the sampled
        sequences are (T, omega_des), the rate channels are limited to the plant's representable range
        (so the effective perturbation is the post-limit one), and the applied action is produced by the
        body-rate inner loop from the CURRENT state — which is the only place a rotor command exists and
        the only place the per-rotor box is applied.

        B3 is off in every cell of this screen, so no adaptive-temperature branch is written here; the
        pre/post adaptation diagnostics the harness reads are filled with the single set of weights
        actually used, which is what B3-off means.
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

        base = self.absolute(self.plan)                                      # [B,H,4] (T, omega_des)
        noise = self._draw_noise(batch)                                      # [B,N,H,4] OU, inherited
        raw = base.unsqueeze(1) + noise
        sampled = self.rate_limit(raw)                                       # the plant's rate range
        with torch.random.fork_rng(devices=self._fork_devices):
            self.rate_limit_entries += int(raw[..., 1:].numel())
            self.rate_limit_clamped += int((raw[..., 1:] != sampled[..., 1:]).sum().item())
        effective_noise = sampled - base.unsqueeze(1)                        # post-limit, honest
        del raw, noise
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

        # the update is a convex combination of RATE-LIMITED sequences, so the carried plan's rate
        # channels never leave the plant's representable range.
        update = self.plan + (weight.unsqueeze(-1).unsqueeze(-1) * effective_noise).sum(dim=1)
        del effective_noise
        plan = torch.where(degenerate.view(-1, 1, 1), self.plan, update)

        head = self.rate_limit(self.absolute(plan[:, 0]))                    # [B,4] (T, omega_des)
        # the EXECUTED rate passes the same low-pass, initialised at the vehicle's own measured rate, so
        # the first entry of the plan and the first step of the rollout see the identical filtered rate.
        rate_applied = self.lowpass_beta * x[:, _W] + (1.0 - self.lowpass_beta) * head[:, 1:]
        command = torch.cat([head[:, :1], rate_applied], dim=-1)
        action = self.inner_loop(x, command, count=True)                     # [B,rotor_dim], box-clipped
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
