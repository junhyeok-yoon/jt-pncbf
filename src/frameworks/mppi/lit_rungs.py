"""v2.8.4 — the LITERATURE-CONFIGURATION ladder for the (T, omega_des) cascade.

SUBJECT. `src/frameworks/mppi/cascade_rate.py`. THAT FILE IS NOT MODIFIED BY THIS WORK — every rung lives
in the subclass below, so "existing default behaviour unchanged" is structural rather than asserted.
`cascade.py` (the v5 a_des plan) is not the subject and is not imported. `reach_rungs.py` (the previous
mission's ladder) is not imported either: its R2 seeded-rollout rung is FORBIDDEN here and cannot be
reached from this module because no seeding code exists in it. K_seed IS STRUCTURALLY ZERO.

WHAT IS NOT TOUCHED, ANYWHERE IN THIS MODULE
--------------------------------------------
  * the interface: the planner samples v = [T, w_x, w_y, w_z] and nothing else (the (T, tau) VARIANT is a
    separate class, `TorqueLitController`, and is a separate table column — never the same row);
  * the inner rate loop `tau = J k_rate (omega_des - omega)` — `RateCascadeController.inner_loop`, CALLED;
  * the post-allocation per-rotor clip — inside `MPPIController.allocate`, CALLED;
  * the full RK4 rotor-thrust SIMULATION plant — `src.common.rk4.rk4_step` on `system`, CALLED;
  * the collision predicate `cost.collision_mask` — CALLED, never re-implemented. L2 changes only what the
    cost DOES with the predicate's verdict, never the verdict;
  * the min-cost baseline subtraction, the ESS definition, the degenerate rule, the control hold, the
    receding-horizon shift, the hover centring and the OU sampler (except where L5 replaces the DRAW).

There is NO reference trajectory, NO global planner, NO map, NO waypoint sequence and NO seeded/geometric
rollout in this file.

THE OFF-STATE GUARANTEE
-----------------------
`LitCascadeController.act` dispatches to `super().act` — the shipped `RateCascadeController.act` — the
instant `LitFlags.act_flags_on` is false. No rung branch is entered, no extra draw is taken from the
generator and no tensor is formed, so the OFF state is the shipped code path itself. The parity is
nevertheless MEASURED (`lit_ladder.py`, the `L0_all_flags_off` cell) against
`data/runs/v2.8.4/mppi_cascaded/smoke.json`'s recorded 1.0560340480879828, not left as an argument.

THE REFERENCE CONFIGURATION (arXiv:2605.02147, Pacelli, Ratheesh and Theodorou, "Sampling-Based Control
via Entropy-Regularized Optimal Transport", Appendix C-C / Table VII, MPPI column) — TRANSCRIBED FROM THE
MISSION BRIEF, which states it verbatim:

    horizon 60, iterations 5, samples per iteration 500, inverse temperature 8.054
    running cost  w_goal ||p - g||^2 + w_obstacle I_crash + w_thrust thrust^2
                + w_torque torque^2 + w_velocity ||v||^2 + w_height I_height
    w_goal 0.242, w_obstacle 13.56, w_thrust 0.017, w_torque 0.021, w_velocity 0.01, w_height 10.2

The six cost weights were INDEPENDENTLY CONFIRMED against the arXiv HTML of the paper (they render as a
table this agent could read). The four parameter rows (horizon / iterations / samples / inverse
temperature) did NOT survive the HTML-to-text conversion legibly and are therefore taken from the brief;
Table III's own quadrotor row corroborates the ITERATION COUNT arithmetically (338.0 ms wall clock /
67.6 ms per iteration = 5.00). See `docs/versions/v2.8.4/mppi_lit.md`.

THE RUNGS
---------
L1  `l1_iterations`   ITERATIONS PER CONTROL STEP. The shipped implementation performs ONE
                      sample-score-weight-update per control step (`cascade_rate.py:670-714`); the
                      reference performs 5, RESAMPLING around the updated nominal each time. The inner
                      loop is the shipped update verbatim, re-entered; the receding-horizon shift, the
                      executed head and the inner rate loop happen ONCE, after the last iteration, exactly
                      as they do today. `l1_iterations == 1` is the shipped single update.

L2  `l2_cost`         COST FORM. "reference" replaces the deployed running cost
                      (`cost.stage_cost`: w_goal d^2 + gated relu speed/rate excess) by the reference
                      running cost above at the reference weights. `l2_terminal` = "none" additionally
                      drops the settling terminal, which has no counterpart in the reference cost;
                      "settling" keeps `cost.terminal_cost` unchanged. Both are measured.

                      THE COLLISION PREDICATE IS UNCHANGED. `cost.collision_mask` is CALLED. What changes
                      is only the WEIGHT its verdict is charged at (c_crash -> w_obstacle) and the fact
                      that the reference cost carries a SECOND indicator, I_height. Our own predicate is
                      a disjunction of an obstacle leg and a band leg (`cost.collision_mask:275-284`), and
                      the reference cost's two indicators are exactly those two legs. The legs are read
                      out BY CALLING `collision_mask` TWICE — once with the band suppressed (obstacle leg)
                      and once with `active` suppressed (band leg) — so no predicate is re-implemented and
                      their disjunction is the shipped verdict bit for bit (asserted every step under
                      `l2_assert_predicate`).

                      THE ROLLOUT'S CHARGE-ONCE MECHANICS ARE UNCHANGED: both indicators are charged once,
                      at their first occurrence, exactly as `cascade_rate.py:620` charges C_crash. Charging
                      an indicator at EVERY step of a frozen sample is the graded-vs-indicator question the
                      mission explicitly does not open (previously R5), and it is NOT opened here.

L3  `l3_horizon`      HORIZON. Ours is `mppi.cascaded.smoke.straight_line.mppi_horizon` = 40 decision
                      entries at `env.dt` (config.yaml:...; both read, never typed); theirs is 60. Applied
                      at CONSTRUCTION through `MPPIParams.horizon` — the shipped field — so no code path
                      changes at all. Recorded in the flags for the ablation record.

L4  `l4_temperature`  TEMPERATURE CONVENTION. The shipped rule is RELATIVE
                      (`cascade_rate.py:699-703`: `lam_eff = clamp(lam * cost.std(dim=1), lam_eps_abs)`,
                      selected by `mppi.sampling.lam_mode: relative`). The reference weights on
                      `exp(-beta S)` with beta = 8.054, i.e. an ABSOLUTE lambda = 1 / beta. The shipped
                      code ALREADY implements the absolute convention (`cascade_rate.py:694-697`,
                      `lam_mode == "absolute"`), so L4 is a CONSTRUCTION-TIME selection of an existing
                      branch and adds no code. Both conventions are run.

L5  `l5_noise`        PROPOSAL QUALITY. Modifies the DRAW only. Two citable options; AT MOST ONE is
                      implemented and which one is stated in the build log.
                        "colored"  — low-frequency / colored noise, Vlahov, Gibson, Fan, Buch, Theodorou,
                                     "Low Frequency Sampling in Model Predictive Path Integral Control",
                                     RA-L 9(5):4543-4550, 2024. The horizon-axis spectrum of the drawn
                                     perturbation is shaped as 1/f^exponent and renormalised so the
                                     per-channel marginal std is exactly the shipped `sigma_channel`.
                        "lognormal" — log-MPPI's normal-log-normal mixture, Mohamed, Yin, Liu, RA-L
                                     7(4):10240-10247, 2022. The shipped OU draw is multiplied by a
                                     per-(sample, channel) log-normal factor with mu = -sigma_ln^2 so the
                                     second moment is preserved and only the tails change.

L6                    WEIGHT TUNING. Not implemented in this module; if it is reached it is a search over
                      the six `ReferenceCost` fields driven from `lit_ladder.py`, on scenes disjoint from
                      every reported number.

Every constant is read from the system object or the config. Nothing is typed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping

import numpy as np
import torch

from src.common.rk4 import rk4_step
from src.common.system import System
from src.frameworks.mppi.cascade_rate import RateCascadeController, _W
from src.frameworks.mppi.cost import CostParams, collision_mask, stage_cost, terminal_cost


Tensor = torch.Tensor

L2_MODES = ("off", "reference")
L2_TERMINALS = ("settling", "none")
L2_THRUST_REFS = ("absolute", "hover_relative")
L4_MODES = ("off", "absolute")
L5_MODES = ("off", "colored", "lognormal")


@dataclass(frozen=True)
class ReferenceCost:
    """The reference running cost's six weights, TRANSCRIBED from the mission brief's statement of
    arXiv:2605.02147 Table VII (MPPI column). The six values were independently confirmed against the
    paper's arXiv HTML. They are DATA, not defaults resolved from our config, and are recorded verbatim on
    every cell that uses them so a reader can check the transcription without leaving the artifact."""

    w_goal: float = 0.242
    w_obstacle: float = 13.56
    w_thrust: float = 0.017
    w_torque: float = 0.021
    w_velocity: float = 0.01
    w_height: float = 10.2

    def record(self) -> dict[str, Any]:
        return {
            "source": "arXiv:2605.02147 Appendix C-C / Table VII, MPPI column, as transcribed in the "
                      "mission brief; the six weights independently confirmed against the arXiv HTML",
            "expression": "w_goal ||p - g||^2 + w_obstacle I_crash + w_thrust thrust^2 "
                          "+ w_torque torque^2 + w_velocity ||v||^2 + w_height I_height",
            "w_goal": self.w_goal, "w_obstacle": self.w_obstacle, "w_thrust": self.w_thrust,
            "w_torque": self.w_torque, "w_velocity": self.w_velocity, "w_height": self.w_height,
        }


@dataclass(frozen=True)
class LitFlags:
    """One field per rung. The all-default instance is the shipped implementation and `act_flags_on` is
    False, which is what makes the OFF state the shipped code path rather than a re-derivation of it.

    `l3_horizon`, `l4_*` and `n_samples` are CONSTRUCTION-TIME selections of fields the shipped
    `MPPIParams` already carries; they are recorded here so a cell's ablation is readable off one object,
    but they enter no branch in `act`."""

    # L1 — iterations per control step. 1 = the shipped single update.
    l1_iterations: int = 1
    # L2 — cost form
    l2_cost: str = "off"
    l2_terminal: str = "settling"
    l2_thrust_ref: str = "absolute"
    l2_assert_predicate: bool = True
    # L3 — horizon (0 = the config's own value, applied at construction)
    l3_horizon: int = 0
    # L4 — temperature convention (construction-time; "absolute" selects the shipped absolute branch)
    l4_temperature: str = "off"
    l4_beta: float = 0.0
    # L5 — proposal quality (the DRAW only)
    l5_noise: str = "off"
    l5_colored_exponent: float = 0.0
    l5_ln_sigma: float = 0.0
    # sample count (0 = the config's own value, applied at construction)
    n_samples: int = 0
    # instrumentation: track each sample's CLOSEST APPROACH over its rollout. Pure measurement — it draws
    # no random number and feeds nothing into the control.
    track_closest: bool = False

    def __post_init__(self) -> None:
        if self.l1_iterations < 1:
            raise ValueError(f"l1_iterations must be >= 1, got {self.l1_iterations!r}.")
        if self.l2_cost not in L2_MODES:
            raise ValueError(f"l2_cost must be one of {L2_MODES}, got {self.l2_cost!r}.")
        if self.l2_terminal not in L2_TERMINALS:
            raise ValueError(f"l2_terminal must be one of {L2_TERMINALS}, got {self.l2_terminal!r}.")
        if self.l2_thrust_ref not in L2_THRUST_REFS:
            raise ValueError(f"l2_thrust_ref must be one of {L2_THRUST_REFS}, got {self.l2_thrust_ref!r}.")
        if self.l4_temperature not in L4_MODES:
            raise ValueError(f"l4_temperature must be one of {L4_MODES}, got {self.l4_temperature!r}.")
        if self.l4_temperature == "absolute" and not self.l4_beta > 0.0:
            raise ValueError("l4_temperature 'absolute' needs a positive l4_beta (lambda = 1 / beta).")
        if self.l5_noise not in L5_MODES:
            raise ValueError(f"l5_noise must be one of {L5_MODES}, got {self.l5_noise!r}.")
        if self.l5_noise == "colored" and not self.l5_colored_exponent > 0.0:
            raise ValueError("l5_noise 'colored' needs a positive l5_colored_exponent.")
        if self.l5_noise == "lognormal" and not self.l5_ln_sigma > 0.0:
            raise ValueError("l5_noise 'lognormal' needs a positive l5_ln_sigma.")

    @property
    def act_flags_on(self) -> bool:
        """Whether any flag changes the CODE PATH inside `act`. `l3_horizon`, `l4_*` and `n_samples` are
        deliberately absent: they select existing `MPPIParams` fields at construction and are executed by
        the shipped code, so with only those set the shipped `act` still runs."""
        return bool(
            self.l1_iterations > 1
            or self.l2_cost != "off"
            or self.l2_terminal != "settling"
            or self.l5_noise != "off"
            or self.track_closest
        )

    @property
    def lam_mode(self) -> str:
        return "absolute" if self.l4_temperature == "absolute" else "relative"

    @property
    def lam_value(self) -> float | None:
        """The ABSOLUTE lambda the reference convention implies, `1 / beta`. None under the shipped
        relative rule, where `lam` keeps its config meaning (a multiplier on the cost spread)."""
        return (1.0 / float(self.l4_beta)) if self.l4_temperature == "absolute" else None

    def record(self) -> dict[str, Any]:
        return {
            "l1_iterations": self.l1_iterations,
            "l2_cost": self.l2_cost, "l2_terminal": self.l2_terminal,
            "l2_thrust_ref": self.l2_thrust_ref, "l2_assert_predicate": self.l2_assert_predicate,
            "l3_horizon": self.l3_horizon,
            "l4_temperature": self.l4_temperature, "l4_beta": self.l4_beta,
            "l4_lambda_absolute": self.lam_value, "l4_lam_mode_selected": self.lam_mode,
            "l5_noise": self.l5_noise, "l5_colored_exponent": self.l5_colored_exponent,
            "l5_ln_sigma": self.l5_ln_sigma,
            "n_samples": self.n_samples,
            "track_closest": self.track_closest,
            "act_flags_on": self.act_flags_on,
            "K_seed": 0,
            "K_seed_note": "STRUCTURALLY ZERO. This module contains no seeded / geometric rollout code at "
                           "all; `reach_rungs.py` (which does) is not imported and cannot be reached.",
            "off_state": "with act_flags_on False, act() dispatches to RateCascadeController.act — the "
                         "shipped code path itself, not a copy",
        }


class LitCascadeController(RateCascadeController):
    """The (T, omega_des) cascade with the literature-configuration ladder's rungs behind flags.

    `RateCascadeController` is SUBCLASSED, never edited: with every flag at its default this class's `act`
    is `RateCascadeController.act`, reached through `super()`, so the shipped behaviour is the shipped code
    and not a reproduction of it.
    """

    def __init__(
        self,
        *args: Any,
        flags: LitFlags | None = None,
        reference_cost: ReferenceCost | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.flags = flags if flags is not None else LitFlags()
        self.ref_cost = reference_cost if reference_cost is not None else ReferenceCost()

        # ---- L2's two predicate legs. Both are `cost.collision_mask` CALLED on a CostParams whose one
        #      relevant field is suppressed, so neither leg is a re-implementation of anything. --------
        # obstacle leg: the band limit set to 0 disables the band branch (`cost.collision_mask:281`).
        self._obstacle_only = replace(self.cost, band_limit=0.0)
        # band leg: `active` is passed all-false at the call site, which kills the obstacle branch.

        # ---- L5's horizon-axis spectral shaping, built once. -----------------------------------------
        self._l5_filter: Tensor | None = None
        if self.flags.l5_noise == "colored":
            horizon = int(self.params.horizon)
            freq = torch.fft.rfftfreq(horizon, d=1.0).to(device=self.device, dtype=self.dtype)
            gain = torch.ones_like(freq)
            # 1/f^(exponent/2) in AMPLITUDE == 1/f^exponent in POWER. The DC bin takes the first non-zero
            # bin's gain so the filter is finite; the whole filter is renormalised below anyway.
            nonzero = freq > 0
            gain[nonzero] = freq[nonzero].pow(-0.5 * float(self.flags.l5_colored_exponent))
            gain[~nonzero] = gain[nonzero][0] if bool(nonzero.any()) else 1.0
            self._l5_filter = gain

        # ---- diagnostics ------------------------------------------------------------------------
        # per-iteration, over the whole episode. index i holds iteration i's values across control steps.
        self.iter_increment: list[list[float]] = [[] for _ in range(int(self.flags.l1_iterations))]
        self.iter_best_closest: list[list[float]] = [[] for _ in range(int(self.flags.l1_iterations))]
        self.iter_min_closest: list[list[float]] = [[] for _ in range(int(self.flags.l1_iterations))]
        self.iter_ess: list[list[float]] = [[] for _ in range(int(self.flags.l1_iterations))]
        self.iter_lam: list[list[float]] = [[] for _ in range(int(self.flags.l1_iterations))]
        self.iter_cost_min: list[list[float]] = [[] for _ in range(int(self.flags.l1_iterations))]
        # L2 per-term shares, accumulated over the episode as summed contributions to the sample costs.
        self.term_totals: dict[str, float] = {}
        self._closest_chunks: list[Tensor] = []
        self.last_closest: Tensor | None = None
        self.predicate_checks = 0
        self.predicate_mismatches = 0

    # ---- the record a cell JSON carries ----------------------------------------------------------
    def lit_record(self) -> dict[str, Any]:
        return {
            "what": "the v2.8.4 literature-configuration ladder — one flag per rung, every OFF state the "
                    "shipped code path",
            "subject_file_modified": False,
            "subject": "src/frameworks/mppi/cascade_rate.py (subclassed, not edited)",
            "variant": "cascaded_rate",
            "flags": self.flags.record(),
            "reference_cost": self.ref_cost.record() if self.flags.l2_cost == "reference" else None,
            "unchanged": [
                "the (T, omega) interface", "the inner rate loop", "the post-allocation per-rotor clip",
                "the full RK4 rotor-thrust simulation plant",
                "the collision predicate cost.collision_mask (CALLED; L2 changes only the weight its "
                "verdict is charged at and reads its two legs by calling it twice)",
                "the min-cost baseline subtraction, the ESS definition, the degenerate rule, "
                "the control hold, the receding-horizon shift, hover centring",
            ],
            "no_seeded_rollouts": "K_seed = 0 structurally: this module contains no seeded / geometric "
                                  "rollout code and does not import reach_rungs.py",
            "no_reference_trajectory": "no reference trajectory, no global planner, no map, no waypoints",
            "collision_charge_mechanics": (
                "UNCHANGED — each indicator is charged ONCE, at its first occurrence, exactly as "
                "cascade_rate.py:620 charges C_crash. Per-step charging of a frozen sample is the "
                "graded-vs-indicator question the mission does not open, and it is not opened."
            ),
            "predicate_agreement": {
                "checked": int(self.predicate_checks),
                "mismatches": int(self.predicate_mismatches),
                "what": "obstacle_leg | band_leg == cost.collision_mask(...) verdict, asserted at every "
                        "physical rollout step of every L2 cell",
            },
        }

    # ---- L5: the draw --------------------------------------------------------------------------
    def _draw_noise(self, batch: int) -> Tensor:
        """L5 modifies THE PROPOSAL DISTRIBUTION ONLY. With `l5_noise == "off"` the shipped
        `MPPIController._draw_noise` is CALLED and nothing here executes."""
        if self.flags.l5_noise == "off":
            return super()._draw_noise(batch)
        if self.flags.l5_noise == "colored":
            n, horizon, m = self.params.n_samples, self.params.horizon, self.action_dim
            z = torch.randn(
                (batch, n, horizon, m), generator=self.generator, device=self.device, dtype=self.dtype
            )
            assert self._l5_filter is not None
            spectrum = torch.fft.rfft(z, dim=2) * self._l5_filter.view(1, 1, -1, 1)
            shaped = torch.fft.irfft(spectrum, n=horizon, dim=2)
            # renormalise to the SHIPPED per-channel marginal std, so L5 changes the SHAPE of the
            # perturbation's spectrum and not its size. The std is taken over the whole draw.
            scale = shaped.std(dim=(0, 1, 2), keepdim=True).clamp(min=torch.finfo(self.dtype).tiny)
            return shaped / scale * self.sigma_channel.view(1, 1, 1, -1)
        # lognormal: the SHIPPED OU draw times a per-(sample, channel) log-normal factor whose second
        # moment is 1, so only the tails change. log-MPPI's normal-log-normal mixture.
        eps = super()._draw_noise(batch)
        sigma_ln = float(self.flags.l5_ln_sigma)
        z2 = torch.randn(
            (batch, self.params.n_samples, 1, self.action_dim),
            generator=self.generator, device=self.device, dtype=self.dtype,
        )
        return eps * torch.exp(-sigma_ln * sigma_ln + sigma_ln * z2)

    # ---- the rollout ---------------------------------------------------------------------------
    @torch.no_grad()
    def _rollout_cost(
        self, x: Tensor, sampled: Tensor, goal: Tensor, centers: Tensor, radii: Tensor, active: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if not self.flags.act_flags_on:
            return super()._rollout_cost(x, sampled, goal, centers, radii, active)
        self._closest_chunks = []
        cost, dead = super()._rollout_cost(x, sampled, goal, centers, radii, active)
        if self._closest_chunks:
            self.last_closest = (
                torch.cat(self._closest_chunks, dim=1)
                if len(self._closest_chunks) > 1 else self._closest_chunks[0]
            )
            self._closest_chunks = []
        return cost, dead

    @torch.no_grad()
    def _rollout_chunk(
        self, x: Tensor, sampled: Tensor, goal: Tensor, centers: Tensor, radii: Tensor, active: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """`RateCascadeController._rollout_chunk` with (i) L2's cost form and (ii) the closest-approach
        instrumentation. With neither on, the PARENT IS CALLED and the OFF path is the shipped one.

        The freeze-on-collision rule, the charge-once C_crash mechanics, the SAME `collision_mask`, the
        SAME `terminal_cost` object and the endpoint probe are unchanged.
        """
        reference = self.flags.l2_cost == "reference"
        drop_terminal = self.flags.l2_terminal == "none"
        if not (reference or drop_terminal or self.flags.track_closest):
            return super()._rollout_chunk(x, sampled, goal, centers, radii, active)

        batch, n, horizon, plan_dim = sampled.shape
        flat = batch * n
        state = x.unsqueeze(1).expand(batch, n, -1).reshape(flat, -1)
        goal_flat = goal.unsqueeze(1).expand(batch, n, -1).reshape(flat, -1)
        cost = torch.zeros(flat, device=self.device, dtype=self.dtype)
        dead = torch.zeros(flat, dtype=torch.bool, device=self.device)
        beta = self.lowpass_beta
        rate_f = state[:, _W].clone()
        inactive = torch.zeros_like(active)
        crashed_once = torch.zeros(flat, dtype=torch.bool, device=self.device)
        height_once = torch.zeros(flat, dtype=torch.bool, device=self.device)
        closest = torch.linalg.norm(self.system.position(state) - goal_flat, dim=-1)
        totals = {"goal": 0.0, "thrust": 0.0, "torque": 0.0, "velocity": 0.0,
                  "obstacle": 0.0, "height": 0.0, "collision_shipped": 0.0, "stage_shipped": 0.0}

        for k in range(horizon * self.control_hold):
            command = self.held_control(sampled, k).reshape(flat, plan_dim)
            rate_f = beta * rate_f + (1.0 - beta) * command[:, 1:]
            # the INNER LOOP'S torque for this command, formed here only to be COSTED (L2's torque^2 term).
            # It is `RateCascadeController.inner_loop`'s own expression restricted to the wrench's rate
            # leg, with J and k_rate read off the system object; no rotor command is formed and no box is
            # applied, exactly as in the shipped rollout.
            torque = self.inertia * (self.k_rate * (command[:, 1:] - rate_f))
            state = torch.cat([state[:, :10], rate_f], dim=1)
            nxt = rk4_step(self.planner, state, command, self.dt)
            previous = state
            state = torch.where(dead.unsqueeze(-1), state, nxt)
            positions = self.system.position(state).view(batch, n, -1)
            collided = collision_mask(positions, centers, radii, active, self.cost).reshape(flat)

            if reference:
                obstacle = collision_mask(
                    positions, centers, radii, active, self._obstacle_only
                ).reshape(flat)
                height = collision_mask(
                    positions, centers, radii, inactive, self.cost
                ).reshape(flat)
                if self.flags.l2_assert_predicate:
                    with torch.random.fork_rng(devices=self._fork_devices):
                        self.predicate_checks += 1
                        self.predicate_mismatches += int(
                            ((obstacle | height) != collided).sum().item()
                        )
                new_crash = obstacle & ~crashed_once
                new_height = height & ~height_once
                cost = cost + new_crash.to(self.dtype) * self.ref_cost.w_obstacle
                cost = cost + new_height.to(self.dtype) * self.ref_cost.w_height
                crashed_once = crashed_once | obstacle
                height_once = height_once | height
                with torch.random.fork_rng(devices=self._fork_devices):
                    totals["obstacle"] += float(
                        (new_crash.to(self.dtype) * self.ref_cost.w_obstacle).sum().item()
                    )
                    totals["height"] += float(
                        (new_height.to(self.dtype) * self.ref_cost.w_height).sum().item()
                    )
            else:
                newly = collided & ~dead
                cost = cost + newly.to(self.dtype) * self.cost.c_crash
                with torch.random.fork_rng(devices=self._fork_devices):
                    totals["collision_shipped"] += float(
                        (newly.to(self.dtype) * self.cost.c_crash).sum().item()
                    )
            dead = dead | collided

            if reference:
                distance = torch.linalg.norm(self.system.position(state) - goal_flat, dim=-1)
                thrust = command[:, 0]
                if self.flags.l2_thrust_ref == "hover_relative":
                    thrust = thrust - self.t_hover
                goal_term = self.ref_cost.w_goal * distance.square()
                thrust_term = self.ref_cost.w_thrust * thrust.square()
                torque_term = self.ref_cost.w_torque * torque.square().sum(dim=-1)
                vel_term = self.ref_cost.w_velocity * self.system.speed(state).square()
                cost = cost + goal_term + thrust_term + torque_term + vel_term
                with torch.random.fork_rng(devices=self._fork_devices):
                    totals["goal"] += float(goal_term.sum().item())
                    totals["thrust"] += float(thrust_term.sum().item())
                    totals["torque"] += float(torque_term.sum().item())
                    totals["velocity"] += float(vel_term.sum().item())
            else:
                stage = stage_cost(
                    self.system, state, goal_flat, self.cost, self.recovery, x_prev=previous
                )
                cost = cost + stage
                with torch.random.fork_rng(devices=self._fork_devices):
                    totals["stage_shipped"] += float(stage.sum().item())

            if self.flags.track_closest:
                closest = torch.minimum(
                    closest, torch.linalg.norm(self.system.position(state) - goal_flat, dim=-1)
                )

        if not drop_terminal:
            terminal = terminal_cost(self.system, state, goal_flat, self.cost)
            cost = cost + terminal
            with torch.random.fork_rng(devices=self._fork_devices):
                totals["terminal"] = totals.get("terminal", 0.0) + float(terminal.sum().item())

        with torch.random.fork_rng(devices=self._fork_devices):
            for key, value in totals.items():
                self.term_totals[key] = self.term_totals.get(key, 0.0) + value
            if self.flags.track_closest:
                self._closest_chunks.append(closest.view(batch, n))

        if self.endpoint_probe:
            with torch.random.fork_rng(devices=self._fork_devices):
                self._endpoint_chunk_dists.append(
                    torch.linalg.norm(self.system.position(state) - goal_flat, dim=-1).view(batch, n)
                )
        return cost.view(batch, n), dead.view(batch, n)

    # ---- the weighting, factored out so L1's iterations re-enter EXACTLY the shipped update -------
    def _weights(self, cost: Tensor, all_collided: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """`cascade_rate.py:692-708` verbatim. Returns (weight, degenerate, ess, lam_eff_row)."""
        batch = int(cost.shape[0])
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
        return weight, degenerate, ess, lam_eff_row

    # ---- one control step -------------------------------------------------------------------------
    @torch.no_grad()
    def act(self, x: Tensor, scene: Any) -> Tensor:
        """`RateCascadeController.act` with L1's iteration loop. With every act-flag off the shipped
        method is called through `super()` and NOTHING below runs — no branch, no draw, no allocation.

        L1: the sample / score / weight / update block is re-entered `l1_iterations` times, RESAMPLING
        around the nominal the previous iteration produced. Everything outside the block — the hold
        schedule, the receding-horizon shift, the executed head, the low-pass on the executed rate and the
        inner rate loop — happens exactly ONCE, after the last iteration, as it does today.
        """
        if not self.flags.act_flags_on:
            return super().act(x, scene)

        from src.common.observation import scene_goal_tensor
        from src.frameworks.mppi.recovery import tilt_cos, tilt_deg_from_cos

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

        plan = self.plan
        degenerate = torch.zeros(batch, dtype=torch.bool, device=self.device)
        ess = torch.zeros(batch, dtype=self.dtype, device=self.device)
        lam_eff_row = torch.zeros(batch, dtype=self.dtype, device=self.device)
        for iteration in range(int(self.flags.l1_iterations)):
            base = self.absolute(plan)                                       # [B,H,4] (T, omega_des)
            noise = self._draw_noise(batch)                                  # [B,N,H,4]
            raw = base.unsqueeze(1) + noise
            sampled = self.rate_limit(raw)                                   # the plant's rate range
            with torch.random.fork_rng(devices=self._fork_devices):
                self.rate_limit_entries += int(raw[..., 1:].numel())
                self.rate_limit_clamped += int((raw[..., 1:] != sampled[..., 1:]).sum().item())
            effective_noise = sampled - base.unsqueeze(1)
            del raw, noise
            cost, all_collided = self._rollout_cost(x, sampled, goal, centers, radii, active)
            del sampled

            if self.endpoint_probe:
                with torch.random.fork_rng(devices=self._fork_devices):
                    assert self.last_endpoint_dist is not None
                    self.last_best_endpoint_dist = self.last_endpoint_dist.gather(
                        1, cost.argmin(dim=1, keepdim=True)
                    ).squeeze(1)

            weight, degenerate, ess, lam_eff_row = self._weights(cost, all_collided)
            update = plan + (weight.unsqueeze(-1).unsqueeze(-1) * effective_noise).sum(dim=1)
            del effective_noise
            new_plan = torch.where(degenerate.view(-1, 1, 1), plan, update)

            # ---- L1's per-iteration diagnostics. Pure measurement; no random number, no control path.
            if self.flags.l1_iterations > 1 or self.flags.track_closest:
                with torch.random.fork_rng(devices=self._fork_devices):
                    self.iter_increment[iteration].append(
                        float(torch.linalg.norm(new_plan[0] - plan[0]).item())
                    )
                    self.iter_ess[iteration].append(float(ess[0].item()))
                    self.iter_lam[iteration].append(float(lam_eff_row[0].item()))
                    self.iter_cost_min[iteration].append(float(cost[0].min().item()))
                    if self.last_closest is not None:
                        best = int(cost[0].argmin().item())
                        self.iter_best_closest[iteration].append(
                            float(self.last_closest[0, best].item())
                        )
                        self.iter_min_closest[iteration].append(
                            float(self.last_closest[0].min().item())
                        )
            plan = new_plan

        if self.endpoint_probe and self.last_best_endpoint_dist is not None:
            with torch.random.fork_rng(devices=self._fork_devices):
                self._endpoint_current.append(
                    self.last_best_endpoint_dist.detach().to(torch.float32).cpu().numpy().copy()
                )

        head = self.rate_limit(self.absolute(plan[:, 0]))                    # [B,4] (T, omega_des)
        rate_applied = self.lowpass_beta * x[:, _W] + (1.0 - self.lowpass_beta) * head[:, 1:]
        command = torch.cat([head[:, :1], rate_applied], dim=-1)
        action = self.inner_loop(x, command, count=True)                     # box-clipped, unchanged
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
# THE OPTIONAL (T, tau) VARIANT — a SEPARATE baseline column, never the same row as cascaded_rate
# =================================================================================================
class _TorquePlannerModel:
    """The planner's prediction model for the (T, tau) variant: THE SHIPPED PLANT'S OWN `dynamics` with
    exactly one edit — the rotor allocation `u @ mixer.T` is not performed, because the plan already
    carries the wrench. `dot omega = J^-1 (tau - omega x J omega)` is the plant's own line, RESTORED (the
    (T, omega) cascade deletes it; here attitude DYNAMICS are inside the planner).

    The MIXER and the POST-ALLOCATION PER-ROTOR CLIP are untouched and are applied to the EXECUTED command
    exactly as they are for the cascaded variant — only the inner RATE LOOP is removed from the planner.
    """

    def __init__(self, system: System, device: torch.device, dtype: torch.dtype) -> None:
        from src.envs.quadrotor_3d import _pure_quat, _quat_mul
        self.system = system
        self.mass = float(system.mass)
        self.gravity = float(system.gravity)
        self.inertia = system.inertia.to(device=device, dtype=dtype)
        self._pure_quat = _pure_quat
        self._quat_mul = _quat_mul

    def dynamics(self, x: Tensor, u: Tensor) -> Tensor:
        q, v, omega = x[:, 3:7], x[:, 7:10], x[:, 10:13]
        thrust, torque = u[:, 0], u[:, 1:4]
        g_vec = torch.zeros_like(v)
        g_vec[:, 2] = self.gravity
        dv = (thrust.unsqueeze(-1) / self.mass) * self.system.thrust_axis(x) - g_vec
        dq = 0.5 * self._quat_mul(q, self._pure_quat(omega))
        dw = (torque - torch.cross(omega, self.inertia * omega, dim=1)) / self.inertia
        return torch.cat([v, dq, dv, dw], dim=1)

    def wrap_state(self, x: Tensor) -> Tensor:
        return self.system.wrap_state(x)


class TorqueLitController(LitCascadeController):
    """(T, tau) MPPI: the reference's own interface, one level below ours.

    SEPARATE VARIANT COLUMN. Its numbers may never sit in a table with `cascaded_rate` numbers without an
    explicit variant column. The plan is v = [T, tau_x, tau_y, tau_z]; the planner integrates the SHIPPED
    plant's attitude dynamics; the executed command goes through the SAME `MPPIController.allocate`
    (mixer inverse + the per-rotor box clip) that every other arm uses. There is no inner rate loop at all
    — it is removed from the planner, which is the point of the variant.

    The rate LOW-PASS and the rate LIMIT have no meaning on a torque plan and are structurally absent:
    `rate_limit` becomes the identity and the low-pass is not applied. Both are recorded.
    """

    def rate_limit(self, plan: Tensor) -> Tensor:
        """IDENTITY. `omega_max` bounds a body RATE; this plan carries a body TORQUE, which has no such
        representable range on the system object. Nothing is invented to fill the gap: the channel is left
        unbounded, exactly as the collective channel is in the cascaded variant."""
        return plan

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.planner = _TorquePlannerModel(self.system, self.device, self.dtype)
        # the torque channels take the torque std vanilla's OWN wrench sampler gives them,
        # `sigma * ||mixer_row_j||` — the base class's `channel_scale`, read off the mixer.
        self.sigma_channel = float(self.params.sigma) * torch.linalg.norm(self.mixer, dim=1)

    def inner_loop(self, x: Tensor, command: Tensor, *, count: bool = False) -> Tensor:
        """(T, tau) -> per-rotor forces. There is NO rate loop: the plan IS the wrench. The mixer and the
        per-rotor box clip are `MPPIController.allocate`, unchanged."""
        if count:
            with torch.random.fork_rng(devices=self._fork_devices):
                pre = command @ self.mixer_inv.t()
                out = (pre < self.u_lo) | (pre > self.u_hi)
                self.preclip_entries += int(pre.numel())
                self.preclip_out_of_box += int(out.sum().item())
                self.preclip_min = min(self.preclip_min, float(pre.min().item()))
                self.preclip_max = max(self.preclip_max, float(pre.max().item()))
        return self.allocate(command)

    @torch.no_grad()
    def _rollout_chunk(
        self, x: Tensor, sampled: Tensor, goal: Tensor, centers: Tensor, radii: Tensor, active: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """The rollout for the torque plan: the SHIPPED plant's attitude dynamics inside the planner, no
        low-pass, no omega-slot overwrite. Same freeze rule, same charge-once, same predicate, same cost
        objects (or L2's reference form)."""
        reference = self.flags.l2_cost == "reference"
        drop_terminal = self.flags.l2_terminal == "none"
        batch, n, horizon, plan_dim = sampled.shape
        flat = batch * n
        state = x.unsqueeze(1).expand(batch, n, -1).reshape(flat, -1)
        goal_flat = goal.unsqueeze(1).expand(batch, n, -1).reshape(flat, -1)
        cost = torch.zeros(flat, device=self.device, dtype=self.dtype)
        dead = torch.zeros(flat, dtype=torch.bool, device=self.device)
        inactive = torch.zeros_like(active)
        crashed_once = torch.zeros(flat, dtype=torch.bool, device=self.device)
        height_once = torch.zeros(flat, dtype=torch.bool, device=self.device)
        closest = torch.linalg.norm(self.system.position(state) - goal_flat, dim=-1)

        for k in range(horizon * self.control_hold):
            command = self.held_control(sampled, k).reshape(flat, plan_dim)
            nxt = rk4_step(self.planner, state, command, self.dt)
            previous = state
            state = torch.where(dead.unsqueeze(-1), state, nxt)
            positions = self.system.position(state).view(batch, n, -1)
            collided = collision_mask(positions, centers, radii, active, self.cost).reshape(flat)
            if reference:
                obstacle = collision_mask(
                    positions, centers, radii, active, self._obstacle_only
                ).reshape(flat)
                height = collision_mask(positions, centers, radii, inactive, self.cost).reshape(flat)
                cost = cost + (obstacle & ~crashed_once).to(self.dtype) * self.ref_cost.w_obstacle
                cost = cost + (height & ~height_once).to(self.dtype) * self.ref_cost.w_height
                crashed_once = crashed_once | obstacle
                height_once = height_once | height
            else:
                cost = cost + (collided & ~dead).to(self.dtype) * self.cost.c_crash
            dead = dead | collided
            if reference:
                distance = torch.linalg.norm(self.system.position(state) - goal_flat, dim=-1)
                thrust = command[:, 0]
                if self.flags.l2_thrust_ref == "hover_relative":
                    thrust = thrust - self.t_hover
                cost = cost + self.ref_cost.w_goal * distance.square()
                cost = cost + self.ref_cost.w_thrust * thrust.square()
                cost = cost + self.ref_cost.w_torque * command[:, 1:4].square().sum(dim=-1)
                cost = cost + self.ref_cost.w_velocity * self.system.speed(state).square()
            else:
                cost = cost + stage_cost(
                    self.system, state, goal_flat, self.cost, self.recovery, x_prev=previous
                )
            if self.flags.track_closest:
                closest = torch.minimum(
                    closest, torch.linalg.norm(self.system.position(state) - goal_flat, dim=-1)
                )
        if not drop_terminal:
            cost = cost + terminal_cost(self.system, state, goal_flat, self.cost)
        if self.flags.track_closest:
            with torch.random.fork_rng(devices=self._fork_devices):
                self._closest_chunks.append(closest.view(batch, n))
        if self.endpoint_probe:
            with torch.random.fork_rng(devices=self._fork_devices):
                self._endpoint_chunk_dists.append(
                    torch.linalg.norm(self.system.position(state) - goal_flat, dim=-1).view(batch, n)
                )
        return cost.view(batch, n), dead.view(batch, n)

    @torch.no_grad()
    def act(self, x: Tensor, scene: Any) -> Tensor:
        """The (T, tau) act. Structurally `LitCascadeController.act` with the rate low-pass removed from
        the executed head — a torque command has no rate to filter."""
        from src.common.observation import scene_goal_tensor
        from src.frameworks.mppi.recovery import tilt_cos, tilt_deg_from_cos

        batch = int(x.shape[0])
        if self.plan is None or self.plan.shape[0] != batch:
            self.reset(batch)
        assert self.plan is not None and self.degenerate_steps is not None
        if self.steps_since_reset % self.control_hold != 0:
            assert self.last_action is not None
            self.last_decision = False
            self.steps_since_reset += 1
            return self.last_action.clone()
        x = x.to(device=self.device, dtype=self.dtype)
        centers, radii, active = self._scene_tensors(scene)
        goal = scene_goal_tensor(scene, x)
        if self.steps_since_reset == 0:
            self.spawn_tilt_deg = tilt_deg_from_cos(tilt_cos(self.system, x))

        plan = self.plan
        degenerate = torch.zeros(batch, dtype=torch.bool, device=self.device)
        ess = torch.zeros(batch, dtype=self.dtype, device=self.device)
        lam_eff_row = torch.zeros(batch, dtype=self.dtype, device=self.device)
        for iteration in range(int(self.flags.l1_iterations)):
            base = self.absolute(plan)
            noise = self._draw_noise(batch)
            sampled = base.unsqueeze(1) + noise
            effective_noise = noise
            del noise
            cost, all_collided = self._rollout_cost(x, sampled, goal, centers, radii, active)
            del sampled
            weight, degenerate, ess, lam_eff_row = self._weights(cost, all_collided)
            update = plan + (weight.unsqueeze(-1).unsqueeze(-1) * effective_noise).sum(dim=1)
            del effective_noise
            new_plan = torch.where(degenerate.view(-1, 1, 1), plan, update)
            with torch.random.fork_rng(devices=self._fork_devices):
                self.iter_increment[iteration].append(
                    float(torch.linalg.norm(new_plan[0] - plan[0]).item())
                )
                self.iter_ess[iteration].append(float(ess[0].item()))
                self.iter_lam[iteration].append(float(lam_eff_row[0].item()))
                self.iter_cost_min[iteration].append(float(cost[0].min().item()))
                if self.last_closest is not None:
                    best = int(cost[0].argmin().item())
                    self.iter_best_closest[iteration].append(float(self.last_closest[0, best].item()))
                    self.iter_min_closest[iteration].append(float(self.last_closest[0].min().item()))
            plan = new_plan

        command = self.absolute(plan[:, 0])                                   # [B,4] = (T, tau)
        action = self.inner_loop(x, command, count=True)                      # mixer + the box clip
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
        return action

    def lit_record(self) -> dict[str, Any]:
        record = super().lit_record()
        record["variant"] = "thrust_torque"
        record["variant_status"] = (
            "SEPARATE BASELINE COLUMN. MPPI over (collective thrust T, body torque tau) — the reference's "
            "own interface. It removes the inner RATE LOOP from the planner while keeping the mixer and "
            "the post-allocation per-rotor clip. It is NOT cascaded_rate and may never share a table with "
            "cascaded_rate, rotor-direct, seeded or reference numbers without an explicit variant column."
        )
        record["planner_model"] = {
            "form": "the SHIPPED plant's dynamics with the rotor allocation `u @ mixer.T` not performed "
                    "(the plan already carries the wrench); dot omega = J^-1 (tau - omega x J omega) is "
                    "the plant's own line, restored",
            "rate_lowpass": "structurally absent — a torque command has no rate to filter",
            "rate_limit": "identity — omega_max bounds a body RATE, and this plan carries a body TORQUE; "
                          "nothing is invented to fill the gap",
        }
        return record


# =================================================================================================
# construction
# =================================================================================================
def build_lit_controller(
    mppi_config: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    flags: LitFlags,
    reference_cost: ReferenceCost | None = None,
    variant: str = "cascaded_rate",
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[System, LitCascadeController]:
    """Construct the ladder's controller directly, mirroring `evaluate_mppi.build_framework`'s own
    `rate_cascade` branch entry for entry (as `reach_rungs.build_reach_controller` does), so NO existing
    file has to be edited to reach the subclass. The base configuration is `mppi.v5.base_cell`, read
    through `stages_v5.base_kwargs`; the cascade mapping is `screen_cascaded.cascade_kwargs`.

    L3 (horizon), L4 (temperature convention) and the sample count are applied HERE, on the shipped
    `MPPIParams` fields, so they change no code path."""
    from src.frameworks.jt_pncbf.train import make_system
    from src.frameworks.mppi.mppi_controller import MPPIParams
    from src.frameworks.mppi.recovery import RecoveryParams
    from src.frameworks.mppi.screen_cascaded import cascade_kwargs
    from src.frameworks.mppi.stages_v5 import base_kwargs

    smoke_cfg = mppi_config["cascaded"]["smoke"]
    kwargs = base_kwargs(mppi_config)
    # OUR horizon, read from the config the smoke itself runs at; L3 overrides it with the reference's.
    kwargs["horizon"] = int(smoke_cfg["straight_line"]["mppi_horizon"])
    if flags.l3_horizon > 0:
        kwargs["horizon"] = int(flags.l3_horizon)
    lam = float(kwargs["lam"]) if flags.lam_value is None else float(flags.lam_value)
    n_samples = (
        int(flags.n_samples) if flags.n_samples > 0
        else int(smoke_cfg["straight_line"]["mppi_n_samples"][0])
    )

    system = make_system(config)
    params = MPPIParams.from_config(
        mppi_config, n_samples=n_samples, horizon=kwargs["horizon"], lam=lam,
        sigma=kwargs["sigma"], seed=int(mppi_config["cascaded"]["scale"]["seed"]), sample_chunk=0,
        space=kwargs["space"], noise=kwargs["noise"], lam_mode=flags.lam_mode,
        center=kwargs["center"], control_hold=kwargs["control_hold"],
    )
    cost_params = CostParams.from_config(
        mppi_config, config["env"], c_crash=kwargs["c_crash"], terminal=kwargs["terminal"],
        g1=kwargs["g1"], g2=kwargs["g2"], g3=kwargs["g3"], g4=kwargs["g4"],
    )
    recovery = RecoveryParams.from_config(
        mppi_config, b1=kwargs["b1"], b2=kwargs["b2"], b3=kwargs["b3"]
    )
    cls = TorqueLitController if variant == "thrust_torque" else LitCascadeController
    controller = cls(
        system, config, params, cost_params, device=device, dtype=dtype, recovery=recovery,
        cascade=cascade_kwargs(mppi_config, float(smoke_cfg["rate_gain_factor"])),
        flags=flags, reference_cost=reference_cost,
    )
    return system, controller
