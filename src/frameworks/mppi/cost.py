"""v2.8.4 MPPI baseline — the running / terminal cost and the collision indicator.

Every coefficient is read from `src/frameworks/mppi/config.yaml` (block `mppi.cost`) or from the shipped
env config (`env.goal_radius`, `env.goal_speed_radius`, `env.goal_angrate_radius`,
`env.band_collision_limit`, `env.band_terminates`). NOTHING is hard-coded here — `CostParams.from_config`
indexes required keys and raises KeyError if one is missing, so a silent default can never creep in.

The five terms (the sampled sequence cost S is their sum over the horizon plus the terminal):

  1. goal distance     w_goal * ||p - g||^2                                        (running, always on)
  2. near-goal speed   w_vel     * gate(d) * relu(||v|| - v_G)^2                   (running)
  3. near-goal rate    w_angrate * gate(d) * relu(||omega|| - omega_G)^2           (running)
  4. collision         C_crash * 1{collided}, charged ONCE at the first collision  (running, indicator)
  5. terminal          see `terminal_cost` — `mppi.cost.terminal` selects
                       "settling" (S3 default) or "distance" (legacy, S1/S2)

A SIXTH term exists only when B1 (charter "v3", `mppi.recovery.b1.enabled`, default false) is switched
on: the tilt-scaled attitude leg `w_att(theta) * (1 - cos theta)`, added to the RUNNING cost and never to
the terminal. See `src/frameworks/mppi/recovery.py`. With B1 off the branch is not executed, so the cost
is the five terms above byte for byte.

Terms 2 and 3 exist because the deployed reach terminal is a CONJUNCTION,
`dist <= goal_radius AND speed <= goal_speed_radius AND angular_rate <= goal_angrate_radius`
(src/common/outcomes.py step_outcomes). A cost carrying only the position term produces a controller
that flies through the goal ball at speed and never latches the predicate: the episodes end in timeout
and the baseline is unfairly weak. Both gated terms are relu-excess over exactly the deployed
thresholds, so they are zero precisely on the set where the predicate can close.

`gate(d) = exp(-(d / (gate_scale * goal_radius))^2)` is a smooth near-goal window built from the third
deployed radius, `env.goal_radius`. It is smooth rather than a hard indicator so a sample that is one
step outside the ball still feels the settling pressure (at dt = 0.05 s and ||v|| <= 2.5 m/s a hard
indicator of width goal_radius is crossed in ~1.2 steps and carries almost no signal).

================================================================================================
S3 CHANGE 2 — THE SETTLING TERMINAL COST (`mppi.cost.terminal: settling`)
================================================================================================

The S1/S2 terminal was `w_terminal * ||p_H - g||`: it scores DISTANCE, while the deployed terminal
predicate is a simultaneous position AND velocity AND angular-rate settling. A plan that arrives fast
and overshoots scores identically to one that arrives slowly and latches, so the terminal credit never
pointed at the event the eval actually rewards. S3 replaces it with the predicate's own excess:

    phi(x_H) = w_terminal * [ relu(||p_H - g|| - goal_radius)^2
                            + relu(||v_H||    - goal_speed_radius)^2
                            + relu(||w_H||    - goal_angrate_radius)^2 ]

All three radii are READ from the effective deployed env config (the SAME numbers `step_outcomes` tests
and the same three the running terms 2-3 already use); none is typed here. The stage cost keeps its S1
form exactly — only the terminal changes.

WHY THE POSITION TERM IS ALSO RELU'D. The dispatch writes the position leg as a bare `||p - g||^2`, but
its own smoke requires the terminal cost to be ZERO EXACTLY on states satisfying the deployed terminal
predicate. A bare `||p - g||^2` is positive everywhere except `p == g`, so it cannot satisfy that. The
relu on the position leg is what makes the smoke property exact, and it is the reading of the dispatch's
own parenthesis "the 0.15-radius handled by the distance term" — the radius enters the distance term as
the relu offset. With all three legs relu'd, `phi == 0` iff `d <= goal_radius AND s <= goal_speed_radius
AND w <= goal_angrate_radius`, computed from the same norms `outcomes.step_outcomes` uses, i.e. exactly
the deployed conjunction and nothing else. `cpu_smoke.py` check (j) is that property test.

The legacy `w_terminal * ||p_H - g||` stays reachable as `mppi.cost.terminal: distance`, which is what
the backward-reproducibility gate runs and what reproduces S1 and S2.

================================================================================================
THE GOAL-ATTRACTION REDESIGN G1-G4 (charter "v4"; `mppi.goal_v4`, ALL FOUR DEFAULT OFF)
================================================================================================

Three of the four components live in this file; each is behind its own config switch and each defaults
to the charter-"v3" behaviour, so with the shipped config the two functions below are the pre-v4 ones
branch for branch.

G1 (`mppi.goal_v4.g1.enabled`) — the PROGRESS stage cost. The absolute position term
`w_goal * ||p - g||^2` is REPLACED by the per-step approach term

    -w_p_eff * ( ||p_{k-1} - g|| - ||p_k - g|| )

charged at every PHYSICAL rollout step, with `p_{k-1}` the state the rollout stepped FROM. It is a
reward for approaching (negative cost) and a charge for receding. `w_p_eff = w_p * w_p_scale` when G2 is
on and `w_p` otherwise. Two properties, stated because they are algebra rather than expectations:
  * the term TELESCOPES — summed over the horizon it is `w_p_eff * (||p_H - g|| - ||p_0 - g||)`, and
    `p_0` is common to all N samples of a scene at a given control step, so BETWEEN SAMPLES the term
    is exactly `w_p_eff * ||p_H - g||`;
  * a collided sample is FROZEN by the rollout, so its position increments are 0 for the remaining
    horizon: under G1 an early crash stops accruing position cost, and the discrimination between an
    early crash and survival rests on `C_crash` and on the terms G1 does not replace.

G2 (`mppi.goal_v4.g2.enabled`) — the progress weight becomes `w_p * w_p_scale`. G2 multiplies ONLY the
progress weight, so with G1 off nothing in the resolved cost changes; `CostParams.g2_inert` records that
per cell instead of leaving it implicit.

G3 (`mppi.goal_v4.g3.enabled`) — the terminal is linearised to `w_terminal * ||p_H - g||`, first power,
replacing the settling form. `w_terminal` is the SAME field the settling terminal reads. This expression
is byte-identical to the legacy `terminal: distance` branch and SHARES it rather than duplicating it.

G4 (`mppi.goal_v4.g4.enabled`) — an always-on `w_omega * ||omega||^2` running term, with no near-goal
gate and no tilt condition, ADDED to (not substituted for) the gated relu-excess rate term. Nothing in
the sampler is touched by G4.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from src.common.system import System


Tensor = torch.Tensor

TERMINAL_MODES = ("settling", "distance")


@dataclass(frozen=True)
class CostParams:
    """Resolved cost coefficients. Built only through `from_config`; no field has a default."""

    w_goal: float
    w_vel: float
    w_angrate: float
    w_terminal: float
    gate_scale: float
    c_crash: float
    terminal_mode: str      # "settling" (S3) | "distance" (legacy S1/S2)
    # deployed terminal radii, read from the env config (the SAME numbers step_outcomes tests)
    goal_radius: float
    goal_speed_radius: float
    goal_angrate_radius: float
    # deployed collision geometry, read from the env config
    band_limit: float
    band_terminates: bool
    # ---- charter "v4" goal-attraction redesign (mppi.goal_v4); all four default OFF ---------------
    g1_enabled: bool        # G1 — progress stage cost replaces the absolute position term
    g1_w_p: float           # w_p, the progress weight BEFORE G2's scale
    g2_enabled: bool        # G2 — apply w_p_scale to the progress weight
    g2_w_p_scale: float     # the "current value and 10x" sweep factor
    g3_enabled: bool        # G3 — linearised terminal, w_terminal * ||p_H - g||
    g4_enabled: bool        # G4 — always-on w_omega * ||omega||^2 running term
    g4_w_omega: float       # w_omega

    def __post_init__(self) -> None:
        if self.terminal_mode not in TERMINAL_MODES:
            raise ValueError(
                f"mppi.cost.terminal must be one of {TERMINAL_MODES}, got {self.terminal_mode!r}."
            )

    @property
    def gate_rho(self) -> float:
        """Width of the near-goal settling window, in metres."""
        return self.gate_scale * self.goal_radius

    @property
    def w_p_effective(self) -> float:
        """The progress weight actually charged: `w_p * w_p_scale` under G2, `w_p` otherwise."""
        return self.g1_w_p * self.g2_w_p_scale if self.g2_enabled else self.g1_w_p

    @property
    def g2_inert(self) -> bool:
        """True when G2 is on but G1 is off — G2 scales only the progress weight, so with the progress
        term not evaluated it changes no coefficient of the resolved cost. Recorded, not assumed."""
        return self.g2_enabled and not self.g1_enabled

    @property
    def linear_terminal(self) -> bool:
        """Whether the terminal is the first-power distance form — the legacy `terminal: distance`
        branch, which G3 selects. The two reach the SAME expression through one branch."""
        return self.g3_enabled or self.terminal_mode == "distance"

    @classmethod
    def from_config(
        cls,
        mppi_config: Mapping[str, Any],
        env_config: Mapping[str, Any],
        *,
        c_crash: float | None = None,
        terminal: str | None = None,
        g1: bool | None = None,
        g2: bool | None = None,
        g3: bool | None = None,
        g4: bool | None = None,
        w_p: float | None = None,
        w_omega: float | None = None,
    ) -> "CostParams":
        """`mppi_config` is the `mppi` block of src/frameworks/mppi/config.yaml; `env_config` is
        `config["env"]` of the effective (shipped) config. `c_crash` overrides the config value when the
        screening grid sweeps it; `terminal` selects the terminal-cost form (S3 "settling" vs the legacy
        "distance" the reproducibility gate runs).

        `g1`-`g4` override the charter-"v4" switches (None = the config value, which is False for all
        four); `w_p` / `w_omega` override the two v4 coefficients when the Round-2 sweep moves them. Every
        default is indexed as a required key, so a missing config entry raises rather than defaulting."""
        cost = mppi_config["cost"]
        v4 = mppi_config["goal_v4"]
        return cls(
            w_goal=float(cost["w_goal"]),
            w_vel=float(cost["w_vel"]),
            w_angrate=float(cost["w_angrate"]),
            w_terminal=float(cost["w_terminal"]),
            gate_scale=float(cost["gate_scale"]),
            c_crash=float(cost["c_crash"] if c_crash is None else c_crash),
            terminal_mode=str(cost["terminal"] if terminal is None else terminal),
            goal_radius=float(env_config["goal_radius"]),
            goal_speed_radius=float(env_config["goal_speed_radius"]),
            goal_angrate_radius=float(env_config["goal_angrate_radius"]),
            band_limit=float(env_config["band_collision_limit"]),
            band_terminates=bool(env_config["band_terminates"]),
            g1_enabled=bool(v4["g1"]["enabled"] if g1 is None else g1),
            g1_w_p=float(v4["g1"]["w_p"] if w_p is None else w_p),
            g2_enabled=bool(v4["g2"]["enabled"] if g2 is None else g2),
            g2_w_p_scale=float(v4["g2"]["w_p_scale"]),
            g3_enabled=bool(v4["g3"]["enabled"] if g3 is None else g3),
            g4_enabled=bool(v4["g4"]["enabled"] if g4 is None else g4),
            g4_w_omega=float(v4["g4"]["w_omega"] if w_omega is None else w_omega),
        )

    def goal_v4_record(self) -> dict[str, Any]:
        """Everything about the v4 switches a cell record must carry, with the numerics resolved."""
        return {
            "g1_progress_stage_cost": self.g1_enabled,
            "w_p_config": self.g1_w_p,
            "g2_rebalance": self.g2_enabled,
            "w_p_scale": self.g2_w_p_scale,
            "w_p_effective": self.w_p_effective,
            "g2_inert_because_g1_off": self.g2_inert,
            "g3_linear_terminal": self.g3_enabled,
            "w_terminal": self.w_terminal,
            "terminal_expression": (
                "w_terminal * ||p_H - g||" if self.linear_terminal
                else "w_terminal * [relu(d-r_p)^2 + relu(s-r_v)^2 + relu(w-r_w)^2]"
            ),
            "g4_always_on_angrate": self.g4_enabled,
            "w_omega": self.g4_w_omega,
            "all_off_equals_v3": not (
                self.g1_enabled or self.g2_enabled or self.g3_enabled or self.g4_enabled
            ),
        }


def collision_mask(
    positions: Tensor,
    centers: Tensor,
    radii: Tensor,
    active: Tensor,
    params: CostParams,
) -> Tensor:
    """The collision-by-class predicate, mirroring `src/common/outcomes.py`.

    This REPLICATES the eval harness's predicate rather than calling it, because `step_outcomes` consumes
    a [T, B, .] trajectory against a per-scene obstacle field, while MPPI needs the predicate on a
    [scene, sample, .] block whose sample axis is NOT a scene axis. The two pieces replicated are exactly:

      * the obstacle leg, `outcomes.py:_collided_exact` (lines 96-115): xy-cylinder contact,
        `((||p_xy - c|| < r) & active).any(-1)`, STRICT `<`, obstacles taking the first
        `centers.shape[-1]` position coordinates;
      * the band leg, `outcomes.py:54-64`: `band_lower = p_z <= -band_z`, `band_upper = p_z >= +band_z`
        for `band_z = env.band_collision_limit > 0`, and
        `collided = (obstacle | band_lower | band_upper) if band_terminates else obstacle`.

    Agreement with `step_outcomes(...).collided` is asserted on a batch of states by
    `src/frameworks/mppi/cpu_smoke.py` check (d) and recorded in
    `data/runs/v2.8.4/mppi_screen/cpu_smoke.json`.

    Shapes
    ------
    positions : [B, S, 3]  B scenes x S samples (S is the MPPI sample axis, or a time axis when checking)
    centers   : [B, K, C]  per-scene obstacle centres (C = 2 for the infinite vertical cylinders)
    radii     : [B, K]
    active    : [B, K]
    returns   : [B, S] bool
    """
    if positions.ndim != 3:
        raise ValueError(f"positions must be [B, S, 3], got {tuple(positions.shape)}.")
    if centers.ndim != 3:
        raise ValueError(f"centers must be [B, K, C], got {tuple(centers.shape)}.")

    pos_o = positions[..., : centers.shape[-1]]                       # [B,S,C]
    distance = torch.linalg.norm(pos_o.unsqueeze(-2) - centers.unsqueeze(-3), dim=-1)   # [B,S,K]
    obstacle = ((distance < radii.unsqueeze(-2)) & active.unsqueeze(-2)).any(dim=-1)    # [B,S]

    band_lower = torch.zeros_like(obstacle)
    band_upper = torch.zeros_like(obstacle)
    if params.band_limit > 0.0 and positions.shape[-1] >= 3:
        band_lower = positions[..., 2] <= -params.band_limit
        band_upper = positions[..., 2] >= params.band_limit
    return (obstacle | band_lower | band_upper) if params.band_terminates else obstacle


def progress_cost(
    system: System,
    x_prev: Tensor,
    x: Tensor,
    goal: Tensor,
    params: CostParams,
) -> Tensor:
    """G1's per-step approach term, `-w_p_eff * (||p_{k-1} - g|| - ||p_k - g||)`.

    NEGATIVE (a cost reduction) on a step that closes distance, POSITIVE on a step that opens it, ZERO on
    a frozen (collided) sample, whose two states are identical. `w_p_eff` carries G2's scale. This is the
    whole of G1: `stage_cost` charges it INSTEAD of `w_goal * ||p - g||^2`, never alongside it.
    """
    distance_prev = torch.linalg.norm(system.position(x_prev) - goal, dim=-1)
    distance = torch.linalg.norm(system.position(x) - goal, dim=-1)
    return -params.w_p_effective * (distance_prev - distance)


def stage_cost(
    system: System,
    x: Tensor,
    goal: Tensor,
    params: CostParams,
    recovery: Any | None = None,
    x_prev: Tensor | None = None,
) -> Tensor:
    """Running cost, terms 1-3, plus B1's attitude leg when it is switched on. `x` is [M, state_dim],
    `goal` is [M, 3]; returns [M].

    The collision indicator (term 4) is NOT here: it is charged once, at the first collision step, by the
    rollout in mppi_controller.py, which also needs the mask to freeze the sample.

    `recovery` is a `src.frameworks.mppi.recovery.RecoveryParams` or None. B1 (charter "v3") adds

        w_att(theta) * (1 - cos theta),   w_att(theta) = w0 * (1 + k_att * relu(theta-theta_ref)/theta_ref)

    with theta the CURRENT tilt of `x` read through `system.thrust_axis`. When `recovery` is None or its
    B1 switch is off the leg is not evaluated at all — the branch is skipped, not added as a zero — so
    the default path is the pre-B1 stage cost byte for byte. The leg is deliberately absent from
    `terminal_cost`: the settling terminal's defining property is that it vanishes exactly on the
    deployed reach predicate, which says nothing about tilt.

    charter "v4": `x_prev` is the state the rollout stepped FROM, required only when G1 is on — the
    progress term then REPLACES `w_goal * ||p - g||^2` (the two are never charged together). G4 adds
    `w_omega * ||omega||^2` with no gate and no tilt condition. With both switches off neither branch is
    executed, so the default path is the pre-v4 stage cost byte for byte.
    """
    position = system.position(x)
    distance = torch.linalg.norm(position - goal, dim=-1)
    if params.g1_enabled:
        if x_prev is None:
            raise ValueError(
                "G1 (mppi.goal_v4.g1.enabled) charges a per-step approach term, so stage_cost needs the "
                "state the rollout stepped from; x_prev was None."
            )
        cost = progress_cost(system, x_prev, x, goal, params)
    else:
        cost = params.w_goal * distance.square()

    gate = torch.exp(-(distance / params.gate_rho).square())
    speed_excess = torch.clamp(system.speed(x) - params.goal_speed_radius, min=0.0)
    rate_excess = torch.clamp(system.angular_rate(x) - params.goal_angrate_radius, min=0.0)
    cost = cost + params.w_vel * gate * speed_excess.square()
    cost = cost + params.w_angrate * gate * rate_excess.square()
    if params.g4_enabled:
        cost = cost + params.g4_w_omega * system.angular_rate(x).square()
    if recovery is not None and recovery.b1_enabled:
        from src.frameworks.mppi.recovery import attitude_cost      # local: keeps cost.py import-light
        cost = cost + attitude_cost(system, x, recovery)
    return cost


def terminal_cost(system: System, x: Tensor, goal: Tensor, params: CostParams) -> Tensor:
    """Terminal cost, term 5. `x` is [M, state_dim], `goal` is [M, 3]; returns [M].

    `params.terminal_mode`:

    "settling" (S3 CHANGE 2, the default) — the deployed terminal predicate's own excess,

        w_terminal * [ relu(d - goal_radius)^2 + relu(s - goal_speed_radius)^2
                       + relu(w - goal_angrate_radius)^2 ]

    with d = ||p_H - g||, s = ||v_H||, w = ||omega_H|| computed through the SAME system accessors
    `src.common.outcomes.step_outcomes` uses and the three radii READ from the effective deployed env
    config (never typed). It is therefore ZERO EXACTLY on the set where the reach predicate closes —
    `d <= r_p AND s <= r_v AND w <= r_w`, all three comparisons non-strict, exactly as `step_outcomes`
    writes them — and STRICTLY POSITIVE as soon as any one of the three is violated. The position leg
    carries the relu offset for the same reason the other two do: without it the terminal would be
    positive everywhere except p == g and the zero-set property would be false (see the module docstring).

    "distance" (LEGACY, S1/S2) — `w_terminal * ||p_H - g||`, linear (not squared) so that a long horizon
    does not let the quadratic running term dominate the terminal credit. Retained so the superseded
    screens stay exactly reproducible; it is what the backward-reproducibility gate runs.

    charter "v4" G3 (`mppi.goal_v4.g3.enabled`) selects the SAME first-power expression as the legacy
    "distance" mode and shares its branch — `params.linear_terminal` is the disjunction, so the two can
    never drift into two different linear terminals.
    """
    distance = torch.linalg.norm(system.position(x) - goal, dim=-1)
    if params.linear_terminal:
        return params.w_terminal * distance
    position_excess = torch.clamp(distance - params.goal_radius, min=0.0)
    speed_excess = torch.clamp(system.speed(x) - params.goal_speed_radius, min=0.0)
    rate_excess = torch.clamp(system.angular_rate(x) - params.goal_angrate_radius, min=0.0)
    return params.w_terminal * (
        position_excess.square() + speed_excess.square() + rate_excess.square()
    )
