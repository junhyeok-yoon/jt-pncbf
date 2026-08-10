"""v2.8.4 MPPI baseline — the three recovery components B1 / B2 / B3 (charter "v3").

LINEAGE — the directory names do NOT match the charter's v1/v2/v3 labels. State it once, here:

    charter "v1"  = the ORIGINAL 16-cell grid (N x H x lam x C_crash), 8 cells run
                    -> data/runs/v2.8.4/mppi_screen/          IMMUTABLE
    (unlabelled)  = the R1-R4 amendment's 12-cell grid (sigma x lam_rel x H)
                    -> data/runs/v2.8.4/mppi_screen_v2/       IMMUTABLE
    charter "v2"  = the hover-centered / settling-terminal / control-hold 16-cell grid
                    -> data/runs/v2.8.4/mppi_screen_v3/       IMMUTABLE
    charter "v3"  = THIS work
                    -> data/runs/v2.8.4/mppi_v3/, docs/versions/v2.8.4/mppi_v3.md

The package's own internal names for the same three screens are S1 / S2 / S3; S3 is the charter's "v2".

================================================================================================
WHAT THE CHARTER'S "v2" ACTUALLY MEASURED — read this before reading the components
================================================================================================

The charter's context says v2 "fixed the reach collapse" and that the residual failure "concentrates at
high spawn tilt". Neither is what `data/runs/v2.8.4/mppi_screen_v3/screen_rows.json` contains. Verbatim
from that file:

  * `reach` is 0.0000 in ALL 16 cells, and 0.0000 in BOTH tilt bands of every cell. The string `goal`
    appears in no cell's `outcome_counts`. No episode of the 6400 scored reached the goal.
  * The failure is not concentrated at high tilt. In the lowest-collision cell
    (`N1024_lam0.05_C100000_m1_H40_sig1`) collision is 0.2040 at tilt < 90 deg and 0.2764 at
    tilt >= 90 deg: worse at high tilt, but the low-tilt band fails almost as badly, and reach is zero
    in both.
  * All EIGHT m=4 cells carry identical headline numbers — collision 0.9500, oob 0.0500, cps -1.9250 —
    because the degenerate branch fired on 0.9997-1.0000 of decisions. At the 8 s lookahead every
    sampled rollout collides inside the horizon, so the MPPI update is never applied and the vehicle
    flies the carried hover-trim plan open-loop.

B1/B2/B3 are therefore applied to a controller that did not reach the goal at ANY tilt, not to a
residual high-tilt failure of an otherwise working controller. The charter is run as specified — that is
the Researcher's call — but nothing here is presented as polishing a residual.

================================================================================================
THE THREE COMPONENTS
================================================================================================

All three are config-switched and all three default OFF, so the shipped config reproduces the charter's
"v2" controller exactly. Every coefficient below is READ from `mppi.recovery` in
`src/frameworks/mppi/config.yaml`; none is typed in code, and `RecoveryParams.from_config` indexes
required keys so a missing one raises rather than silently defaulting.

B1 — RECOVERY-AWARE COST WEIGHT (`mppi.recovery.b1.enabled`)

    The stage cost gains an ATTITUDE leg whose weight is scaled by the CURRENT tilt of the rolled-out
    state:

        theta(x)   = angle between the body up-axis and world up
                   = degrees(arccos(clip(system.thrust_axis(x)[..., 2], -1, 1)))
        w_att(theta) = w0 * (1 + k_att * relu(theta - theta_ref) / theta_ref)
        leg(x)     = w_att(theta(x)) * (1 - cos theta(x))

    `w0` (`mppi.recovery.b1.w_att`), `k_att` and `theta_ref` (`mppi.recovery.theta_ref_deg`, 60 deg —
    the same boundary the reporting split uses, not a new number) are config fields.

    THE WEIGHT IS THE CHARTER'S FORMULA VERBATIM. The charter specifies the weight and not the leg it
    multiplies, and the charter's "v2" cost has no attitude term at all to scale (its five terms are goal
    distance, near-goal speed excess, near-goal angular-rate excess, the collision indicator, and the
    settling terminal). Two readings were available and the choice is recorded rather than hidden:

      (i)  scale the existing near-goal angular-rate weight, the only attitude-family term present. This
           makes "reduces to v2 at low tilt" literally exact, but the term is multiplied by
           `gate(d) = exp(-(d/0.9 m)^2)`, so at high tilt far from the goal — the exact situation B1 is
           for — the scaling multiplies something that is numerically zero, and B1 would be inert by
           construction. It would also penalise the angular RATE a recovery manoeuvre must produce.
      (ii) add an attitude leg carrying the charter's weight. `1 - cos theta` is zero exactly when the
           vehicle is upright, rises monotonically to 2 when inverted, is smooth, and needs no new
           constant. "Reduces to v2 at low tilt" then holds in substance rather than by algebra: at
           theta <= theta_ref the multiplier is exactly 1, and the leg itself decays to 0 as theta -> 0
           (0.015 * w0 at 10 deg).

    (ii) is implemented. This is an interpretation, it is reported as one, and B1 OFF removes the leg
    entirely so the default path is byte-identical to the charter's "v2".

    THE LEG IS IN THE STAGE COST ONLY, NEVER THE TERMINAL. The settling terminal's defining property is
    that it is zero exactly on the deployed reach predicate (`d <= goal_radius AND s <= goal_speed_radius
    AND w <= goal_angrate_radius`). The deployed predicate says nothing about tilt, so an attitude leg in
    the terminal would make it positive on states the eval scores as reached and destroy that property.

B2 — RECOVERY-INFORMED INITIAL PLAN (`mppi.recovery.b2.enabled`)

    At the FIRST decision step of an episode, for the scene rows whose SPAWN tilt exceeds `theta_ref`,
    the head of the nominal sequence is initialised with an attitude-first recovery instead of the flat
    hover trim: `block_steps` decision entries of

        (F_total, tau) = (m*g, tau_mag * a_hat)

    where `a_hat` is the unit body-frame axis that rotates the body up-axis onto world up — the SAME
    attitude-error construction the plant's own nominal controller uses
    (`src/envs/quadrotor_3d.py:lqr_action:181-184`: `e_att_world = cross(b3, b3_des)`, mapped to body by
    `R^T`) — with `b3_des` the world up-axis. Its yaw component is identically zero, so the initialised
    entry is a pure roll/pitch torque, i.e. a THRUST DIFFERENTIAL opposing the tilt, on top of the
    collective hover trim. The remaining entries stay at hover trim.

    `tau_mag = torque_frac * tau_half`, where `tau_half` is the half-width of the roll/pitch torque
    RANGE ACHIEVABLE OVER THE ACTUATOR BOX, computed from `system.mixer` and `system.u_bounds`
    (`channel_authority` below). Nothing about the airframe is typed here.

    This is a PLAN INITIALISATION ONLY. It seeds `self.plan` before the first sample is drawn; the
    sampler perturbs it, the exponential weighting scores it against every other sample, and the update
    moves it freely. It is not a control override and it does not persist: after the first decision the
    receding-horizon shift retires it one entry at a time like any other plan.

B3 — ADAPTIVE TEMPERATURE (`mppi.recovery.b3.enabled`)

    At a decision step, after the weights are formed at the cell's own lambda,

        ESS = (sum_n w_n)^2 / sum_n w_n^2

    is evaluated per scene row (identical to the `1 / sum_n w_norm^2` the controller already computes on
    the normalised weights — the two expressions are the same number). For the rows where ESS falls below
    `ess_frac_of_n * N`, lambda is multiplied by `lam_factor` and the weights, the partition, the
    degenerate mask and ESS are recomputed FOR THAT STEP ONLY. Nothing persists to the next step and the
    carried plan is untouched. The event is counted per episode and per decision step.

    The degenerate branch (all N rollouts of a scene collide -> skip the update, hold the carried plan)
    is evaluated on the POST-adaptation partition and remains the final fallback, exactly as before.

DEFAULTS. `enabled: false` on all three. With all three off, `RecoveryParams.off()` is what the
controller carries, `stage_cost` receives `None` for its attitude argument and does not evaluate it,
`reset()` writes the flat trim plan, and `act()` never recomputes the weights — the code paths are the
pre-B1/B2/B3 ones, not equivalent-looking replacements.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from src.common.system import System
from src.envs.quadrotor_3d import _quat_to_R


Tensor = torch.Tensor


@dataclass(frozen=True)
class RecoveryParams:
    """Resolved B1/B2/B3 coefficients. Built through `from_config` or `off`; no field has a default."""

    theta_ref_deg: float        # shared by B1, B2 and the reporting split (config: theta_ref_deg)

    b1_enabled: bool
    b1_w_att: float             # w0
    b1_k_att: float             # k_att

    b2_enabled: bool
    b2_block_steps: int         # decision entries at the head of the plan seeded with the recovery wrench
    b2_torque_frac: float       # fraction of the achievable roll/pitch torque half-authority

    b3_enabled: bool
    b3_ess_frac_of_n: float     # ESS fraction of N below which lambda is multiplied, that step only
    b3_lam_factor: float        # the multiplier

    def __post_init__(self) -> None:
        if self.theta_ref_deg <= 0.0:
            raise ValueError(f"mppi.recovery.theta_ref_deg must be > 0, got {self.theta_ref_deg!r}.")
        if self.b2_enabled and int(self.b2_block_steps) < 1:
            raise ValueError(
                f"mppi.recovery.b2.block_steps must be a positive integer, got {self.b2_block_steps!r}."
            )
        if self.b3_enabled and not (0.0 < self.b3_ess_frac_of_n <= 1.0):
            raise ValueError(
                "mppi.recovery.b3.ess_frac_of_n must lie in (0, 1] — ESS itself lies in [1, N], so the "
                f"fraction of N is a fraction; got {self.b3_ess_frac_of_n!r}."
            )
        if self.b3_enabled and self.b3_lam_factor <= 0.0:
            raise ValueError(f"mppi.recovery.b3.lam_factor must be > 0, got {self.b3_lam_factor!r}.")

    @property
    def any_enabled(self) -> bool:
        return bool(self.b1_enabled or self.b2_enabled or self.b3_enabled)

    @classmethod
    def from_config(
        cls,
        mppi_config: Mapping[str, Any],
        *,
        b1: bool | None = None,
        b2: bool | None = None,
        b3: bool | None = None,
    ) -> "RecoveryParams":
        """`mppi_config` is the `mppi` block of src/frameworks/mppi/config.yaml. `b1`/`b2`/`b3` override
        the config's `enabled` flags (the screen turns all three on, the ablation turns them on one at a
        time); None keeps the config value, which is False for all three."""
        recovery = mppi_config["recovery"]
        block1, block2, block3 = recovery["b1"], recovery["b2"], recovery["b3"]
        return cls(
            theta_ref_deg=float(recovery["theta_ref_deg"]),
            b1_enabled=bool(block1["enabled"] if b1 is None else b1),
            b1_w_att=float(block1["w_att"]),
            b1_k_att=float(block1["k_att"]),
            b2_enabled=bool(block2["enabled"] if b2 is None else b2),
            b2_block_steps=int(block2["block_steps"]),
            b2_torque_frac=float(block2["torque_frac"]),
            b3_enabled=bool(block3["enabled"] if b3 is None else b3),
            b3_ess_frac_of_n=float(block3["ess_frac_of_n"]),
            b3_lam_factor=float(block3["lam_factor"]),
        )

    def with_switches(self, *, b1: bool, b2: bool, b3: bool) -> "RecoveryParams":
        """The same coefficients with a different on/off triple (the ablation rows)."""
        return RecoveryParams(
            theta_ref_deg=self.theta_ref_deg,
            b1_enabled=bool(b1), b1_w_att=self.b1_w_att, b1_k_att=self.b1_k_att,
            b2_enabled=bool(b2), b2_block_steps=self.b2_block_steps,
            b2_torque_frac=self.b2_torque_frac,
            b3_enabled=bool(b3), b3_ess_frac_of_n=self.b3_ess_frac_of_n,
            b3_lam_factor=self.b3_lam_factor,
        )

    def record(self) -> dict[str, Any]:
        """The component record every cell JSON carries."""
        return {
            "theta_ref_deg": self.theta_ref_deg,
            "theta_ref_source": "mppi.recovery.theta_ref_deg — the same boundary the reporting split "
                                "uses; read from config, never typed in code",
            "B1_recovery_aware_cost_weight": {
                "enabled": self.b1_enabled,
                "w_att_w0": self.b1_w_att,
                "k_att": self.b1_k_att,
                "weight_law": "w_att(theta) = w0 * (1 + k_att * relu(theta - theta_ref) / theta_ref)",
                "leg": "w_att(theta) * (1 - cos theta), theta = angle(body up-axis, world up) from "
                       "system.thrust_axis; STAGE cost only, never the terminal",
                "max_multiplier_at_180_deg": 1.0 + self.b1_k_att * max(
                    0.0, (180.0 - self.theta_ref_deg) / self.theta_ref_deg
                ),
            },
            "B2_recovery_informed_initial_plan": {
                "enabled": self.b2_enabled,
                "block_steps": self.b2_block_steps,
                "torque_frac": self.b2_torque_frac,
                "fires_when": "spawn tilt > theta_ref, at the FIRST decision step of the episode only",
                "wrench": "(m*g, tau_mag * a_hat) with a_hat the unit body-frame axis rotating the body "
                          "up-axis onto world up (the lqr_action attitude-error construction, b3_des = "
                          "world up; its yaw component is identically zero) and tau_mag = torque_frac * "
                          "the roll/pitch torque half-authority of system.mixer over system.u_bounds",
                "nature": "plan initialisation only — the sampler perturbs it, the weighting scores it, "
                          "the update moves it freely, and the shift retires it entry by entry",
            },
            "B3_adaptive_temperature": {
                "enabled": self.b3_enabled,
                "ess_frac_of_n": self.b3_ess_frac_of_n,
                "lam_factor": self.b3_lam_factor,
                "rule": "ESS = (sum w)^2 / sum w^2 per scene row; where ESS < ess_frac_of_n * N, lambda "
                        "is multiplied by lam_factor and the weights / partition / degenerate mask / ESS "
                        "are recomputed FOR THAT DECISION STEP ONLY; nothing persists",
                "fallback": "the degenerate branch (all N rollouts collide -> hold the carried plan) is "
                            "evaluated on the post-adaptation partition and stays the final fallback",
            },
        }

    @classmethod
    def off(cls, mppi_config: Mapping[str, Any]) -> "RecoveryParams":
        """The coefficients with all three switches OFF — the charter's "v2" controller."""
        return cls.from_config(mppi_config, b1=False, b2=False, b3=False)


# ---- B1 ------------------------------------------------------------------------------------------
def tilt_cos(system: System, x: Tensor) -> Tensor:
    """cos(theta) for theta the angle between the body up-axis and world up.

    Read through `system.thrust_axis`, the system object's OWN accessor for the body up-axis in world
    (`src/envs/quadrotor_3d.py:144-146`, `R(q)[:, :, 2]`), so this is the same quantity the reporting
    split's `_quat_to_R(x[:, 3:7])[:, 2, 2]` is, obtained without touching the quaternion layout."""
    axis = getattr(system, "thrust_axis", None)
    if axis is None:
        raise AttributeError(
            f"system {getattr(system, 'name', system)!r} has no `thrust_axis` accessor, so the B1 "
            "attitude leg cannot read the tilt from the system object. B1 is quadrotor-specific."
        )
    return torch.clamp(axis(x)[..., 2], -1.0, 1.0)


def tilt_deg_from_cos(cos_theta: Tensor) -> Tensor:
    """theta in degrees. `arccos` of the clipped cosine, exactly as the reporting split defines it."""
    return torch.rad2deg(torch.arccos(cos_theta))


def attitude_weight(tilt_deg: Tensor, params: RecoveryParams) -> Tensor:
    """B1's weight, the charter's formula verbatim:

        w_att(theta) = w0 * (1 + k_att * relu(theta - theta_ref) / theta_ref)

    Equal to w0 for every theta <= theta_ref and rising linearly above it. w0, k_att and theta_ref are
    config fields; nothing here is typed."""
    excess = torch.clamp(tilt_deg - params.theta_ref_deg, min=0.0) / params.theta_ref_deg
    return params.b1_w_att * (1.0 + params.b1_k_att * excess)


def attitude_cost(system: System, x: Tensor, params: RecoveryParams) -> Tensor:
    """B1's stage-cost leg, `w_att(theta) * (1 - cos theta)`. Zero exactly when upright."""
    cos_theta = tilt_cos(system, x)
    return attitude_weight(tilt_deg_from_cos(cos_theta), params) * (1.0 - cos_theta)


# ---- B2 ------------------------------------------------------------------------------------------
def channel_authority(mixer: Tensor, u_bounds: Tensor) -> Tensor:
    """Half-width of the range each wrench channel can reach over the actuator box, [4].

    For channel j the reachable set of `sum_i M[j,i] * f_i` over `f_i in [lo_i, hi_i]` is the interval
    between `sum_i min(M[j,i]*lo_i, M[j,i]*hi_i)` and `sum_i max(...)`; the half-width is half that
    interval. Both `mixer` and `u_bounds` come from the system object, so no airframe number is typed.
    For the X-config mixer and a [0, f_max] box this evaluates to {2*f_max, 2*l*f_max, 2*l*f_max,
    2*c*f_max} — the exact half-authority of each channel."""
    lo, hi = u_bounds[:, 0], u_bounds[:, 1]
    products = torch.stack([mixer * lo.unsqueeze(0), mixer * hi.unsqueeze(0)], dim=0)   # [2,4,4]
    lower = products.min(dim=0).values.sum(dim=1)
    upper = products.max(dim=0).values.sum(dim=1)
    return 0.5 * (upper - lower)


def recovery_axis_body(x: Tensor) -> Tensor:
    """Unit body-frame axis of the rotation taking the body up-axis onto world up, [B, 3].

    The plant's own nominal controller builds the attitude error the same way
    (`quadrotor_3d.lqr_action:181-184`): `e_att_world = cross(b3, b3_des)` then `R^T e_att_world`, here
    with `b3_des` the world up-axis. Writing it out, `cross(b3, e3) = (b3_y, -b3_x, 0)` and the body-frame
    image has an identically zero yaw component, so the recovery entry is a pure roll/pitch torque — a
    thrust differential. Rows whose axis is degenerate (perfectly upright, or exactly inverted, where the
    axis is undefined) come back as the zero vector and B2 leaves them at hover trim."""
    rotation = _quat_to_R(x[..., 3:7])                                   # [B,3,3] body->world
    body_up = rotation[..., :, 2]                                        # b3 in world
    error_world = torch.stack(
        [body_up[..., 1], -body_up[..., 0], torch.zeros_like(body_up[..., 0])], dim=-1
    )                                                                    # cross(b3, world up)
    error_body = torch.einsum("bji,bj->bi", rotation, error_world)       # R^T @ error_world
    norm = torch.linalg.norm(error_body, dim=-1, keepdim=True)
    return torch.where(norm > 0.0, error_body / torch.clamp(norm, min=torch.finfo(x.dtype).tiny),
                       torch.zeros_like(error_body))


def recovery_wrench(
    x: Tensor, trim_wrench: Tensor, mixer: Tensor, u_bounds: Tensor, params: RecoveryParams
) -> Tensor:
    """B2's absolute recovery wrench per scene row, [B, 4] = (m*g, tau_mag * a_hat_xy, 0).

    The collective stays at the hover trim (read from the system constants by the caller) and the
    differential is `torque_frac` of the roll/pitch half-authority about the recovery axis."""
    authority = channel_authority(mixer, u_bounds).to(device=x.device, dtype=x.dtype)
    axis = recovery_axis_body(x)                                         # [B,3], yaw component == 0
    torque = axis * (params.b2_torque_frac * authority[1:4]).unsqueeze(0)
    collective = trim_wrench[0].expand(x.shape[0], 1)
    return torch.cat([collective, torque], dim=-1)


# ---- B3 ------------------------------------------------------------------------------------------
def effective_sample_size(weight: Tensor) -> Tensor:
    """ESS = (sum_n w_n)^2 / sum_n w_n^2 along the sample axis, [B].

    The charter's expression, written on the UNNORMALISED weights. It is the same number as
    `1 / sum_n w_norm^2` on the normalised weights, which is what the controller reports; both lie in
    [1, N], 1 being a hard argmin and N a uniform softmax."""
    total = weight.sum(dim=1)
    return total.square() / torch.clamp(weight.square().sum(dim=1), min=torch.finfo(weight.dtype).tiny)
