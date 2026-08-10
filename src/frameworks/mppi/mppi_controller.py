"""v2.8.4 MPPI baseline — sampler, exponential weighting, receding-horizon loop, eval adapter.

Standard information-theoretic MPPI (Williams, Aldrich, Theodorou; Williams et al. ICRA'17). At every
control step, from the current state x and the carried nominal sequence U in R^{H x m}:

  1. draw N control-sequence perturbations eps, V = U + eps, and map V into the plant's per-rotor box;
  2. roll all N sequences forward through the TRUE deployed plant map and accumulate the cost S_n;
  3. weight  w_n = exp(-(S_n - min_k S_k) / lambda_eff) / Z;
  4. update  U <- U + sum_n w_n (V_n^eff - U);
  5. apply u_t = alloc(U[0]), then shift U one step and append a braking (hover-trim) tail.

THE PLANT IS THE SHARED DEPLOYED MAP. Step 2 calls `src.common.rk4.rk4_step(system, x, u, dt)` with the
real `system` object built by `src.frameworks.jt_pncbf.train.make_system` and dt = `env.dt`. There is NO
surrogate model and no re-implementation of the dynamics anywhere in this package; `rk4_step` also applies
`system.wrap_state` (quaternion renormalisation, ||v|| <= v_max, |omega| <= omega_max) exactly as the
deployment rollout does. Every control the plant is ever asked to integrate lies inside the per-rotor box
read off `system.u_bounds`.

PRIVILEGE (stated here as well as in the README, because it is the reason this arm is not a peer of the
learned arms): MPPI reads the full 13-D state x and the scene's obstacle field (centres, radii, active
flags) directly, and it simulates the true plant. The learned certificate arms see only the 34-D
observation and carry no model. This is a deliberate asymmetry in MPPI's favour.

================================================================================================
THE v2.8.4 AMENDMENT (R1-R3): WRENCH-SPACE SAMPLING, OU NOISE, RELATIVE LAMBDA
================================================================================================

The first 16-cell screen (artifacts RETAINED under `data/runs/v2.8.4/mppi_screen/`) measured reach 0 in
every cell it ran, with ESS ~= 1: the softmax had collapsed to a hard argmin, so lambda and C_crash were
inert and MPPI was random shooting over N open-loop per-rotor sequences. That screen is a valid
measurement OF A DEGENERATED CONTROLLER. The amendment replaces the sampler:

R1 — CONTROL PARAMETERIZATION. The carried plan U is in BODY-WRENCH space,
`[B, H, 4] = (F_total, tau_x, tau_y, tau_z)`. It is mapped to rotor thrusts through the SYSTEM'S OWN
allocation relation, `f = wrench @ system.mixer_inv.T` — byte-identically the map
`src/envs/quadrotor_3d.py:lqr_action:188` uses — and then clipped per rotor to `system.u_bounds`. The
mixer, its inverse and the box are READ FROM THE SYSTEM OBJECT; nothing here is re-derived or hard-coded.

R2 — NOISE STRUCTURE. The nominal is the HOVER TRIM WRENCH `(mass * gravity, 0, 0, 0)`, with mass and
gravity read from `env.<system>`. Perturbations are Ornstein-Uhlenbeck (a first-order low-pass) ALONG THE
HORIZON AXIS, per channel, in the STATIONARY parameterisation

    eps[:, :, 0, j] = sigma_j * z_0
    eps[:, :, k, j] = alpha * eps[:, :, k-1, j] + sqrt(1 - alpha^2) * sigma_j * z_k,    z ~ N(0, 1) iid

so the MARGINAL std is exactly sigma_j at every k and the lag-1 autocorrelation is exactly alpha, with
`alpha = exp(-1 / correlation_steps)`. Both `correlation_steps` and `alpha` are config fields and the
loader asserts their consistency to 1e-12, so they cannot drift apart.

R3 — RELATIVE LAMBDA. lambda is set RELATIVE to the running spread of the sample costs, per (scene,
control step): `lam_eff = max(lam_rel * std_n(S_n), lam_eps_abs)` with `std_n` the UNBIASED (N-1
denominator) std over that scene's N sample costs at that step. This is what makes ESS collapse by scale
mismatch impossible: the exponent `(S_n - min_k S_k) / lam_eff` is invariant to any rescaling of S.

SIGMA UNITS — settled, do not re-derive.
`sigma` is a scalar in PER-ROTOR-EQUIVALENT NEWTONS (the units the earlier sigma probe used, which ran
{0.1, 0.3, 1.0}). The per-channel wrench std is `sigma_j = sigma * channel_scale_j` with

    channel_scale_j = || M_row_j ||    (row norms of `system.mixer`, computed at construction)
                    = {2.0, 2l, 2l, 2c} = {2.0, 0.2404163056, 0.2404163056, 0.032}

for l = arm_L / sqrt(2) and c = c_moment. RATIONALE: an ABSOLUTE reading of sigma would put
sigma = 0.6 N*m at 3.8x the entire achievable yaw-torque range (tau_z half-authority is
c * f_rotor_max * 2 = 0.15696 N*m) — i.e. pure clipping, and the grid would measure the clipper rather
than the sampler. The per-rotor-equivalent reading keeps the grid in the units of the earlier probe.

CONSEQUENCE, STATED HONESTLY. The mixer rows are orthogonal, so for i.i.d. per-rotor noise
`f ~ N(0, sigma^2 I)` the induced wrench covariance is `sigma^2 M M^T = diag((sigma * ||M_row_j||)^2)`.
With `channel_scale_j = ||M_row_j||` the MARGINAL wrench covariance at a given sigma is therefore EXACTLY
that of the old i.i.d. per-rotor noise. R1's reparameterisation is, under this normalization,
MEASURE-PRESERVING BY ITSELF. What actually changes the sampled distribution is (i) the OU correlation in
time (R2) and (ii) clipping applied AFTER allocation rather than before. R1 is not oversold here: its
value is that the plan, the trim and the update all live in the space the vehicle is actually actuated
in, and that the noise is correlated in the channels a quadrotor cares about.

RE-PROJECTION / EFFECTIVE NOISE. `system.mixer` is invertible but NOT surjective onto the per-rotor box:
a sampled wrench may allocate to rotor forces outside `[0, f_rotor_max]`, and those get clipped. The
rollout is scored on the CLIPPED ROTOR FORCES (the plant's true input), and the MPPI update uses the
EFFECTIVE wrench perturbation obtained by re-projecting the clipped rotor forces back through the mixer,
`eff = (f_clipped @ system.mixer.T) - U`. This is the honest analogue of the old post-clip
`effective_noise`: the weighted average is taken over sequences that were ACTUALLY SIMULATED. Because the
image of the box under M is convex and the weights are normalised, `U + sum_n w_n * eff_n` is itself a
convex combination of achievable wrenches, so the carried plan never leaves the achievable set.

DEGENERATE SAMPLES (unchanged). When every one of the N rollouts for a scene collides, the exponential
weighting carries no usable information: the S_n are all crash-dominated, and in the limit where they are
equal the softmax is uniform, so `sum_n w_n (V_n - U)` averages the noise back to ~0 and the "optimised"
plan is just U plus sampling noise. The case is REAL on this pool: the eval ICs are Haar-uniform on
SO(3), so a large fraction of episodes start inverted, and near a cylinder or the |p_z| = 4 band every
sample dies inside the horizon. The explicit rule, per `config.yaml: mppi.degenerate`:

    on a degenerate row the MPPI update is SKIPPED. The carried plan U is used as-is -- it is the
    previous step's optimised plan already shifted one step, with a hover-trim braking tail appended --
    the first entry is applied, and the shift-with-tail then runs as usual, so each further degenerate
    step retires one more optimised entry and appends one more braking entry. The event is COUNTED per
    episode (`degenerate_steps`).

================================================================================================
THE S3 CHANGES (this screen; artifacts data/runs/v2.8.4/mppi_screen_v3/)
================================================================================================

LINEAGE, so the names are unambiguous. S1 = the ORIGINAL 16-cell grid (N x H x lam x C_crash),
artifacts `data/runs/v2.8.4/mppi_screen/`, the dispatch's "v1". S2 = the R1-R4 amendment's 12-cell grid
(sigma x lam_rel x H), artifacts `data/runs/v2.8.4/mppi_screen_v2/`. S3 = THIS screen, artifacts
`data/runs/v2.8.4/mppi_screen_v3/`, which the dispatch calls "v2" — S3 is NOT the `mppi_screen_v2`
directory. All three are config-switched; S1 and S2 stay exactly reproducible.

S3 CHANGE 1 — HOVER-CENTERED SAMPLING (`mppi.sampling.center: hover|none`). The commanded control is
written as an EXPLICIT decomposition

    u = u_hover + u_plan + eps,

where `u_hover` is a FIXED anchor read from the system object (the hover trim wrench (m*g, 0, 0, 0),
whose rotor image is exactly `mass * gravity / action_dim` per rotor) and `u_plan` is the DEVIATION the
MPPI update moves. Concretely `self.plan` now carries the deviation and `self.anchor` the trim; the
absolute plan is `anchor + plan`. Under `center: none` the anchor is the zero vector and `self.plan` is
the absolute plan again — the S1/S2 path, bit-exact (adding a zero tensor is skipped, not just exact).

This is a RE-PARAMETERISATION of something the R-amendment already had in part: the carried plan was
already initialised to the hover trim and the braking tail was already the hover trim. What is new is
that the trim is now a fixed anchor the update cannot move away from and the plan is a deviation about
it. In exact arithmetic the two are ALGEBRAICALLY IDENTICAL on every branch — sampling
(`anchor + plan + eps == plan_abs + eps`), the weighted update (`anchor + plan + sum w (proj - anchor -
plan) == sum w proj`, the same convex combination), the degenerate hold (the same carried absolute plan)
and the shift tail (deviation 0 <=> absolute trim). They can differ only by floating-point associativity.
This is MEASURED, not asserted: `cpu_smoke.py` check (l) runs both centres over real pool scenes in
float64 and float32 and reports the deviation and whether the resolved metrics agree.

S3 CHANGE 2 — SETTLING TERMINAL COST (`mppi.cost.terminal: settling|distance`). See `cost.py`.

S3 CHANGE 3 — CONTROL HOLD (`mppi.sampling.control_hold: m`). The plan carries H DECISION entries; each
entry is applied for m consecutive PHYSICAL steps. Two consequences:

  * ROLLOUT. A sampled sequence is expanded to H*m rk4 steps, the entry at physical step k being
    `sampled[:, :, k // m]` (`held_control`), so every sampled plan is piecewise-constant with block
    length exactly m and the lookahead is H*m*dt seconds — 2.0 s at m=1 and 8.0 s at m=4 with H=40,
    dt=0.05. The cost is accumulated over all H*m physical steps, so a longer lookahead is not bought by
    coarsening the integration: the plant map, dt and the collision predicate are untouched.
  * CLOSED LOOP. The MPPI optimisation and the receding-horizon shift run ONCE PER m CONTROL STEPS, at
    the DECISION steps `t % m == 0`. On a non-decision (HOLD) step the controller does NOT sample, does
    NOT roll out, does NOT update and does NOT shift: it re-applies the action latched at the last
    decision step, byte-identically. This is the only reading under which the receding-horizon shift
    "retires one decision entry every m control steps and between shifts the same entry is re-applied"
    is literally true — if the update ran on hold steps the head entry would change and the re-applied
    action would not be the same one. Per-step diagnostics (ESS, lam_eff, the degenerate flag) exist
    only at decision steps and are reported over decision steps; hold steps are flagged, never counted
    as if they carried their own weighting. m = 1 makes every step a decision step and recovers the
    unheld controller on the identical code path.

BACKWARD REPRODUCIBILITY. `mppi.sampling.space` (wrench|rotor), `mppi.sampling.noise` (ou|iid),
`mppi.sampling.lam_mode` (relative|absolute), `mppi.sampling.center` (hover|none),
`mppi.sampling.control_hold` (m) and `mppi.cost.terminal` (settling|distance) select the controller. The
R-amendment made wrench/ou/relative the DEFAULT and S3 adds hover/m/settling; setting
rotor/iid/absolute/none/1/distance reproduces the superseded first screen EXACTLY, including the RNG
stream (the legacy branch draws `randn` with the identical shape from the identical generator and applies
the identical op sequence). The reproduction is asserted cell-for-cell against the retained artifacts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from src.common.observation import scene_goal_tensor, scene_obstacle_tensors
from src.common.rk4 import rk4_step
from src.common.system import System
from src.frameworks.mppi.cost import CostParams, collision_mask, stage_cost, terminal_cost
from src.frameworks.mppi.recovery import (
    RecoveryParams,
    channel_authority,
    recovery_wrench,
    tilt_cos,
    tilt_deg_from_cos,
)


Tensor = torch.Tensor

SPACES = ("wrench", "rotor")
NOISES = ("ou", "iid")
LAM_MODES = ("relative", "absolute")
CENTERS = ("hover", "none")


@dataclass(frozen=True)
class MPPIParams:
    """Sampler / weighting hyperparameters.

    The screening grid moves sigma, lam (as lam_rel) and horizon; n_samples and c_crash are fixed by the
    amendment. `space` / `noise` / `lam_mode` select the sampler and exist so the SUPERSEDED first screen
    stays reproducible (rotor / iid / absolute) alongside the amended default (wrench / ou / relative).
    """

    n_samples: int          # N
    horizon: int            # H
    lam: float              # lambda: an ABSOLUTE temperature if lam_mode == "absolute", else lam_rel
    sigma: float            # exploration std in PER-ROTOR-EQUIVALENT newtons (see module docstring)
    seed: int               # torch.Generator seed for the perturbations
    sample_chunk: int       # rollout batch cap along the sample axis; 0 = all N at once
    space: str              # "wrench" (R1, default) | "rotor" (legacy, for the reproducibility gate)
    noise: str              # "ou" (R2, default)     | "iid" (legacy)
    lam_mode: str           # "relative" (R3, default) | "absolute" (legacy)
    center: str             # "hover" (S3, default) | "none" (legacy): the sampling-centre anchor
    control_hold: int       # m — physical steps each DECISION entry is applied for (S3; 1 = unheld)
    ou_correlation_steps: float   # OU correlation length in control steps
    ou_alpha: float               # OU pole; MUST equal exp(-1 / ou_correlation_steps)
    lam_eps_abs: float            # floor on lam_eff, so a zero-variance row cannot divide by zero
    channel_scale_mode: str       # "from_mixer_row_norms" — how sigma is spread over the wrench channels

    def __post_init__(self) -> None:
        if self.space not in SPACES:
            raise ValueError(f"mppi.sampling.space must be one of {SPACES}, got {self.space!r}.")
        if self.noise not in NOISES:
            raise ValueError(f"mppi.sampling.noise must be one of {NOISES}, got {self.noise!r}.")
        if self.lam_mode not in LAM_MODES:
            raise ValueError(f"mppi.sampling.lam_mode must be one of {LAM_MODES}, got {self.lam_mode!r}.")
        if self.center not in CENTERS:
            raise ValueError(f"mppi.sampling.center must be one of {CENTERS}, got {self.center!r}.")
        if int(self.control_hold) < 1:
            raise ValueError(
                f"mppi.sampling.control_hold must be a positive integer, got {self.control_hold!r}."
            )
        # the two OU fields are both written in config.yaml so they are both auditable; they can never be
        # allowed to drift apart, so consistency is a hard assertion, not a coercion.
        expected = math.exp(-1.0 / float(self.ou_correlation_steps))
        if abs(float(self.ou_alpha) - expected) > 1e-12:
            raise ValueError(
                f"mppi.sampling.ou.alpha = {self.ou_alpha!r} does not equal "
                f"exp(-1 / correlation_steps) = {expected!r} to 1e-12 "
                f"(correlation_steps = {self.ou_correlation_steps!r})."
            )
        if self.lam_eps_abs <= 0.0:
            raise ValueError(f"mppi.sampling.lam_eps_abs must be > 0, got {self.lam_eps_abs!r}.")

    @classmethod
    def from_config(
        cls,
        mppi_config: Mapping[str, Any],
        *,
        n_samples: int | None = None,
        horizon: int | None = None,
        lam: float | None = None,
        sigma: float | None = None,
        seed: int | None = None,
        sample_chunk: int | None = None,
        space: str | None = None,
        noise: str | None = None,
        lam_mode: str | None = None,
        center: str | None = None,
        control_hold: int | None = None,
    ) -> "MPPIParams":
        sampling = mppi_config["sampling"]
        ou = sampling["ou"]
        return cls(
            n_samples=int(sampling["n_samples"] if n_samples is None else n_samples),
            horizon=int(sampling["horizon"] if horizon is None else horizon),
            lam=float(sampling["lam"] if lam is None else lam),
            sigma=float(sampling["sigma"] if sigma is None else sigma),
            seed=int(sampling["seed"] if seed is None else seed),
            sample_chunk=int(sampling["sample_chunk"] if sample_chunk is None else sample_chunk),
            space=str(sampling["space"] if space is None else space),
            noise=str(sampling["noise"] if noise is None else noise),
            lam_mode=str(sampling["lam_mode"] if lam_mode is None else lam_mode),
            center=str(sampling["center"] if center is None else center),
            control_hold=int(sampling["control_hold"] if control_hold is None else control_hold),
            ou_correlation_steps=float(ou["correlation_steps"]),
            ou_alpha=float(ou["alpha"]),
            lam_eps_abs=float(sampling["lam_eps_abs"]),
            channel_scale_mode=str(sampling["channel_scale"]),
        )


class MPPIController:
    """Batched MPPI over a batch of scenes. One carried plan U per scene row."""

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
    ) -> None:
        self.system = system
        self.params = params
        self.cost = cost_params
        # charter "v3" B1/B2/B3. None => every switch off; the recovery branches are then never entered.
        self.recovery = recovery
        self.device = device
        self.dtype = dtype
        self.dt = float(config["env"]["dt"])

        bounds = system.u_bounds.to(device=device, dtype=dtype)     # [m, 2]
        self.u_lo = bounds[:, 0]
        self.u_hi = bounds[:, 1]
        self.action_dim = int(system.action_dim)

        # ---- the allocation relation, READ FROM THE SYSTEM (never re-derived) --------------------
        # `system.mixer` maps per-rotor forces to the body wrench (quadrotor_3d.dynamics:94 does
        # `u @ mixer.T`); `system.mixer_inv` is its inverse and is the map lqr_action:188 allocates with.
        self.mixer = system.mixer.to(device=device, dtype=dtype)            # [4, 4] wrench <- rotor
        self.mixer_inv = system.mixer_inv.to(device=device, dtype=dtype)    # [4, 4] rotor  <- wrench
        # sigma is in per-rotor-equivalent newtons; the per-channel wrench std is sigma * ||M_row_j||.
        if params.channel_scale_mode != "from_mixer_row_norms":
            raise ValueError(
                f"Unsupported mppi.sampling.channel_scale {params.channel_scale_mode!r}; the only "
                "implemented mode is 'from_mixer_row_norms'."
            )
        self.channel_scale = torch.linalg.norm(self.mixer, dim=1)           # [4] = {2, 2l, 2l, 2c}
        self.sigma_channel = float(params.sigma) * self.channel_scale       # [4] per-channel wrench std

        # ---- the trim, READ FROM THE SYSTEM CONSTANTS -------------------------------------------
        physics = config["env"][str(config["run"]["system"])]
        self.mass = float(physics["mass"])
        self.gravity = float(physics["gravity"])
        trim_per_rotor = self.mass * self.gravity / float(self.action_dim)
        self.trim_rotor = torch.clamp(
            torch.full((self.action_dim,), trim_per_rotor, device=device, dtype=dtype),
            self.u_lo, self.u_hi,
        )
        # hover trim wrench = (m*g, 0, 0, 0); it is the exact mixer image of trim_rotor.
        self.trim_wrench = torch.zeros(self.action_dim, device=device, dtype=dtype)
        self.trim_wrench[0] = self.mass * self.gravity
        # the plan-space trim: the ABSOLUTE hover trim in whichever space the plan is carried in.
        self.hover_trim = self.trim_wrench if params.space == "wrench" else self.trim_rotor

        # ---- S3 CHANGE 1: the explicit hover anchor, u = u_hover + u_plan + eps ------------------
        # `self.plan` carries u_plan (the DEVIATION); `self.anchor` carries u_hover. Under center=none
        # the anchor is zero and `self.plan` is the absolute plan again — the S1/S2 parameterisation.
        # `self.centered` is checked before every add so the legacy path does not even execute `+ 0`,
        # which keeps it bit-exact rather than merely exact-in-principle.
        self.centered = params.center == "hover"
        self.anchor = self.hover_trim if self.centered else torch.zeros_like(self.hover_trim)
        # what reset() fills the plan with and what _shift() appends as the braking tail, IN PLAN SPACE:
        # exactly 0 when centered (absolute = the trim), the trim itself when not.
        self.plan_trim = self.hover_trim - self.anchor

        # ---- S3 CHANGE 3: control hold ----------------------------------------------------------
        self.control_hold = int(params.control_hold)
        self.lookahead_steps = int(params.horizon) * self.control_hold      # physical rk4 steps rolled
        self.lookahead_s = float(self.lookahead_steps) * self.dt

        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(int(params.seed))

        # per-batch state, set by reset()
        self.plan: Tensor | None = None              # [B, H, m] DEVIATION in PLAN space (wrench|rotor)
        self.degenerate_steps: Tensor | None = None  # [B] long, cumulative count over the episode
        self.last_degenerate: Tensor | None = None   # [B] bool, the most recent DECISION step's mask
        self.last_ess: Tensor | None = None          # [B] float, ESS of the last decision's weights
        self.last_lam_eff: Tensor | None = None      # [B] float, lam_eff of the last decision step
        self.last_action: Tensor | None = None       # [B, m] the latched per-rotor action (hold steps)
        self.last_decision: bool = False             # was the most recent act() a DECISION step?
        self.steps_since_reset: int = 0              # control-step counter; decisions at t % m == 0
        self._scene_cache: tuple[int, tuple[Tensor, Tensor, Tensor]] | None = None
        # charter "v3" per-batch bookkeeping
        self.last_ess_pre: Tensor | None = None      # [B] ESS BEFORE any B3 adaptation of that step
        self.last_lam_eff_pre: Tensor | None = None  # [B] lam_eff BEFORE any B3 adaptation of that step
        self.last_b3_event: Tensor | None = None     # [B] bool, did B3 adapt this row this decision?
        self.b3_events: Tensor | None = None         # [B] long, cumulative B3 adaptations this episode
        self.b2_fired: Tensor | None = None          # [B] bool, did B2 seed this episode's plan?
        self.b2_axis_degenerate: Tensor | None = None  # [B] bool, seeded row whose recovery axis vanished
        self.spawn_tilt_deg: Tensor | None = None    # [B] tilt at the episode's first decision step

        # ---- charter "v5" Stage 1: the BEST-SAMPLE ENDPOINT probe (default OFF) -------------------
        # When `endpoint_probe` is switched on, the rollout records, per (scene, sample), the distance
        # from the sample's ENDPOINT (the state after the last physical rollout step, frozen if that
        # sample collided) to the scene's goal, and `act` keeps the entry of the ARGMIN-COST sample at
        # every decision step. That is the "best-cost sample's endpoint inside r of the goal" column
        # the v5 Stage-1 table reports, and it cannot be recovered after the fact: the endpoint lives
        # only inside the rollout. The probe draws NO random numbers and performs NO arithmetic that
        # feeds the control — with the flag off none of its branches executes at all, so the retained
        # artifacts of every earlier screen reproduce byte for byte. `stages_v5.py` MEASURES that by
        # re-running the base cell with the probe ON and comparing every metric against the retained
        # mppi_screen_v3 row rather than asserting inertness.
        self.endpoint_probe: bool = False
        self._endpoint_chunk_dists: list[Tensor] = []   # per rollout-chunk [B, n_sub], within one act
        self.last_endpoint_dist: Tensor | None = None   # [B, N] of the most recent decision step
        self.last_best_endpoint_dist: Tensor | None = None   # [B] argmin-cost sample's endpoint dist
        self._endpoint_current: list[np.ndarray] = []   # per control step [B], within one eval chunk
        self.endpoint_chunks: list[np.ndarray] = []     # per eval chunk [T, B_chunk] float32
        self._fork_devices = [device.index or 0] if device.type == "cuda" else []

    # ---- introspection used by the cell record ---------------------------------------------------
    def sampler_record(self) -> dict[str, Any]:
        """Everything about the sampler that a cell JSON must carry, with the numerics resolved."""
        return {
            "space": self.params.space,
            "noise": self.params.noise,
            "lam_mode": self.params.lam_mode,
            "sigma_per_rotor_equivalent_N": float(self.params.sigma),
            "channel_scale_mode": self.params.channel_scale_mode,
            "channel_scale": [float(v) for v in self.channel_scale.tolist()],
            "sigma_per_channel": [float(v) for v in self.sigma_channel.tolist()],
            "channel_names": ["F_total", "tau_x", "tau_y", "tau_z"],
            "ou": {
                "correlation_steps": float(self.params.ou_correlation_steps),
                "alpha": float(self.params.ou_alpha),
                "stationary": True,
                "axis": "horizon",
            },
            "trim_mode": "from_system_constants",
            "trim_wrench": [float(v) for v in self.trim_wrench.tolist()],
            "trim_rotor_per_rotor": float(self.trim_rotor[0].item()),
            # ---- S3 CHANGE 1: the sampling centre ------------------------------------------------
            "center": self.params.center,
            "center_decomposition": (
                "u = u_hover + u_plan + eps; u_hover is the fixed anchor below (read from the system "
                "object), u_plan the deviation the MPPI update moves"
                if self.centered else
                "center=none (legacy S1/S2): the anchor is zero and the plan is carried in absolute "
                "units; the trim still seeds reset() and the shift tail"
            ),
            "anchor": [float(v) for v in self.anchor.tolist()],
            "anchor_rotor_per_rotor": float(self.allocate(self.anchor)[0].item())
            if self.params.space == "wrench" else float(self.anchor[0].item()),
            "plan_trim": [float(v) for v in self.plan_trim.tolist()],
            # ---- S3 CHANGE 3: control hold -------------------------------------------------------
            "control_hold_m": self.control_hold,
            "horizon_decision_entries": int(self.params.horizon),
            "rollout_physical_steps": self.lookahead_steps,
            "effective_lookahead_s": self.lookahead_s,
            "dt": self.dt,
            "hold_semantics": (
                "the MPPI optimisation and the receding-horizon shift run at DECISION steps "
                "(t % m == 0); on a hold step the latched action is re-applied with no sampling, no "
                "rollout, no update and no shift"
            ),
            # ---- S3 CHANGE 2: the terminal cost form (radii read from the deployed env config) ----
            "terminal_cost_mode": self.cost.terminal_mode,
            # ---- charter "v4" G1-G4: the goal-attraction switches and their resolved coefficients ----
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
            # ---- charter "v3": the B1/B2/B3 component record --------------------------------------
            "recovery": (
                self.recovery.record() if self.recovery is not None
                else {"present": False,
                      "note": "no RecoveryParams was supplied; B1/B2/B3 are all off and none of their "
                              "branches is reachable"}
            ),
            "b2_torque_half_authority_from_system": (
                [float(v) for v in
                 channel_authority(self.mixer, self.system.u_bounds.to(self.device, self.dtype)).tolist()]
            ),
        }

    # ---- plan bookkeeping -------------------------------------------------------------------
    def reset(self, batch_size: int) -> None:
        """Start a fresh episode batch: plan = the plan-space trim over the whole horizon (the ZERO
        deviation when hover-centered), counters zeroed, the control-hold phase back to a decision."""
        self.plan = self.plan_trim.view(1, 1, -1).expand(
            batch_size, self.params.horizon, self.action_dim
        ).clone()
        self.degenerate_steps = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self.last_degenerate = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        self.last_ess = torch.zeros(batch_size, dtype=self.dtype, device=self.device)
        self.last_lam_eff = torch.zeros(batch_size, dtype=self.dtype, device=self.device)
        self.last_action = None
        self.last_decision = False
        self.steps_since_reset = 0
        self._scene_cache = None
        self.last_ess_pre = torch.zeros(batch_size, dtype=self.dtype, device=self.device)
        self.last_lam_eff_pre = torch.zeros(batch_size, dtype=self.dtype, device=self.device)
        self.last_b3_event = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        self.b3_events = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self.b2_fired = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        self.b2_axis_degenerate = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        self.spawn_tilt_deg = torch.zeros(batch_size, dtype=self.dtype, device=self.device)
        # charter "v5": close the previous eval chunk's endpoint log. `reset` is called at every eval
        # chunk boundary, which is exactly where the [T, B_chunk] matrix must be closed.
        self.flush_endpoint_probe()

    def flush_endpoint_probe(self) -> None:
        """charter "v5": stack the current eval chunk's per-step best-endpoint log into a
        [T, B_chunk] matrix. Idempotent, and a no-op when the probe never ran."""
        if self._endpoint_current:
            self.endpoint_chunks.append(np.stack(self._endpoint_current, axis=0))
            self._endpoint_current = []

    def absolute(self, plan: Tensor) -> Tensor:
        """u_hover + u_plan. Under center=none the anchor is zero and the add is SKIPPED, so the legacy
        parameterisation is bit-exact rather than exact-up-to-adding-zero."""
        return plan + self.anchor if self.centered else plan

    def _shift(self, plan: Tensor) -> Tensor:
        """Receding horizon: drop the applied entry, append one braking-tail entry. Called ONCE PER
        DECISION, i.e. every `control_hold` control steps."""
        tail = self.plan_trim.view(1, 1, -1).expand(plan.shape[0], 1, self.action_dim)
        return torch.cat([plan[:, 1:], tail], dim=1)

    # ---- charter "v3" B2: the recovery-informed initial plan --------------------------------------
    @torch.no_grad()
    def _seed_recovery_plan(self, x: Tensor) -> None:
        """B2. At the FIRST decision step of an episode, replace the head of the nominal sequence with an
        attitude-first recovery for the scene rows whose SPAWN tilt exceeds `theta_ref`.

        `block_steps` decision entries become the absolute wrench `(m*g, tau_mag * a_hat)` — the hover
        collective plus a pure roll/pitch torque (a thrust differential) about the axis that rotates the
        body up-axis onto world up — and the remaining entries stay at hover trim. The write is on the
        PLAN, i.e. on the deviation `absolute - anchor`, so it is the same absolute sequence under both
        sampling centres. This runs BEFORE any sample is drawn: MPPI then perturbs, scores and updates
        this plan exactly as it would the flat one, and the receding-horizon shift retires it entry by
        entry. It is not a control override.
        """
        assert self.plan is not None and self.recovery is not None
        assert self.spawn_tilt_deg is not None
        fire = self.spawn_tilt_deg > self.recovery.theta_ref_deg                # [B]
        self.b2_fired = fire
        if not bool(fire.any()):
            return
        if self.params.space != "wrench":
            raise NotImplementedError(
                "B2 seeds a body-wrench recovery entry, so it requires mppi.sampling.space == 'wrench'; "
                f"got {self.params.space!r}. The legacy rotor-space plan has no wrench to seed."
            )
        wrench = recovery_wrench(
            x, self.trim_wrench, self.mixer, self.system.u_bounds.to(self.device, self.dtype),
            self.recovery,
        )                                                                       # [B,4] absolute
        self.b2_axis_degenerate = fire & (torch.linalg.norm(wrench[:, 1:4], dim=-1) <= 0.0)
        deviation = wrench - self.anchor if self.centered else wrench           # PLAN space
        block = min(int(self.recovery.b2_block_steps), int(self.params.horizon))
        head = deviation.unsqueeze(1).expand(-1, block, -1)
        self.plan[:, :block] = torch.where(
            fire.view(-1, 1, 1), head, self.plan[:, :block]
        )

    def held_control(self, sampled: Tensor, k: int) -> Tensor:
        """S3 CHANGE 3 — the DECISION entry applied at PHYSICAL rollout step `k`.

        `sampled` is [B, N, H, m] over H DECISION entries; entry j covers physical steps
        [j*hold, (j+1)*hold), so the sampled sequence is piecewise-constant with block length exactly
        `control_hold` and the rollout spans H * control_hold physical steps."""
        return sampled[:, :, k // self.control_hold]

    def _scene_tensors(self, scene: Any) -> tuple[Tensor, Tensor, Tensor]:
        """Obstacle field as [B,K,C] / [B,K] / [B,K]; cached per scene object (the eval harness holds one
        BatchedScene per chunk, so this resolves once per chunk)."""
        if self._scene_cache is not None and self._scene_cache[0] == id(scene):
            return self._scene_cache[1]
        centers, radii, active = scene_obstacle_tensors(scene, self.device, self.dtype)
        if centers.ndim == 2:                       # single (unbatched) Scene -> add the scene axis
            centers = centers.unsqueeze(0)
            radii = radii.unsqueeze(0)
            active = active.unsqueeze(0)
        tensors = (centers, radii, active)
        self._scene_cache = (id(scene), tensors)
        return tensors

    # ---- allocation (R1) -------------------------------------------------------------------------
    def allocate(self, wrench: Tensor) -> Tensor:
        """Wrench -> per-rotor forces through the SYSTEM'S OWN relation, then the per-rotor box clip.

        `wrench @ mixer_inv.T` is byte-identically the allocation `quadrotor_3d.lqr_action:188` uses; the
        box is `system.u_bounds`. Works on any trailing-dim-4 tensor."""
        return torch.clamp(wrench @ self.mixer_inv.t(), self.u_lo, self.u_hi)

    def project(self, rotor: Tensor) -> Tensor:
        """Per-rotor forces -> the wrench they actually produce, `rotor @ mixer.T` (the same expression
        `quadrotor_3d.dynamics:94` applies inside the plant). The mixer is not surjective onto the box, so
        this re-projection is what makes the post-clip effective noise honest."""
        return rotor @ self.mixer.t()

    # ---- perturbations (R2) ----------------------------------------------------------------------
    def _draw_noise(self, batch: int) -> Tensor:
        """[B, N, H, m] perturbations in PLAN space.

        wrench/ou  : stationary OU along the horizon, per channel, marginal std sigma * channel_scale_j.
        wrench/iid : the same marginals with no temporal correlation.
        rotor/iid  : the LEGACY draw, byte-identical to the superseded screen — `randn(...) * sigma` with
                     the identical shape from the identical generator, so the RNG stream is preserved.
        """
        n, horizon, m = self.params.n_samples, self.params.horizon, self.action_dim
        z = torch.randn(
            (batch, n, horizon, m), generator=self.generator, device=self.device, dtype=self.dtype
        )
        if self.params.space == "rotor":
            # LEGACY: isotropic per-rotor noise, scalar sigma. Kept exactly as the first screen ran it.
            return z * self.params.sigma
        eps = z * self.sigma_channel                                    # marginal std sigma_j at every k
        if self.params.noise == "iid":
            return eps
        alpha = float(self.params.ou_alpha)
        innovation = math.sqrt(max(0.0, 1.0 - alpha * alpha))
        # In-place stationary recursion along the horizon axis: eps[k] already holds sigma_j * z_k, so
        # eps[k] <- alpha * eps[k-1] + sqrt(1-alpha^2) * eps[k] is exactly the OU update.
        for k in range(1, horizon):
            eps[:, :, k].mul_(innovation).add_(eps[:, :, k - 1], alpha=alpha)
        return eps

    # ---- the controller ---------------------------------------------------------------------
    @torch.no_grad()
    def act(self, x: Tensor, scene: Any) -> Tensor:
        """One CONTROL step. `x` is [B, state_dim]; returns the applied PER-ROTOR action [B, action_dim].

        S3 CHANGE 3 — the control-hold schedule. `t = steps_since_reset` counts control steps within the
        episode batch. `t % control_hold == 0` is a DECISION step: the sampler, the rollout, the
        exponential weighting, the update and the receding-horizon shift all run, the head entry is
        applied, and the applied action is LATCHED. Every other step is a HOLD step, and the applied
        action is exactly that latched action re-issued — no sampling, no rollout, no update, no shift,
        so the head entry the shift will retire is the same entry throughout the block. `last_ess`,
        `last_lam_eff` and `last_degenerate` are left at their decision-step values and `last_decision`
        is set False, so the harness can report those diagnostics over decision steps only rather than
        counting a held step as if it carried its own weighting. control_hold = 1 makes every step a
        decision step on the identical code path.
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
                # a HOLD step runs no rollout, so it has no endpoint of its own: NaN, masked out
                # downstream exactly as the held ESS and lam_eff entries are.
                self._endpoint_current.append(np.full(batch, np.nan, dtype=np.float32))
            return self.last_action.clone()

        x = x.to(device=self.device, dtype=self.dtype)
        centers, radii, active = self._scene_tensors(scene)
        goal = scene_goal_tensor(scene, x)                                   # [B,3]

        # charter "v3" — at the episode's FIRST decision step record the SPAWN tilt from the state the
        # controller is actually handed (an independent cross-check of the pool-derived tilt array the
        # reporting split uses), then let B2 seed the plan if it is on. Both happen before any sample is
        # drawn. With B2 off nothing is written to the plan and it stays the flat trim.
        if self.steps_since_reset == 0:
            self.spawn_tilt_deg = tilt_deg_from_cos(tilt_cos(self.system, x))
            if self.recovery is not None and self.recovery.b2_enabled:
                self._seed_recovery_plan(x)

        base = self.absolute(self.plan)                                    # u_hover + u_plan, [B,H,m]
        noise = self._draw_noise(batch)                                      # [B,N,H,m], PLAN space
        if self.params.space == "rotor":
            # LEGACY PATH — the exact op sequence of the superseded screen: clip to the per-rotor box
            # BEFORE the rollout, and take the post-clip difference as the effective noise.
            rotor = torch.clamp(base.unsqueeze(1) + noise, self.u_lo, self.u_hi)        # [B,N,H,m]
            effective_noise = rotor - base.unsqueeze(1)
        else:
            # R1 PATH — sample in wrench space, allocate through the system's own relation, clip per
            # rotor, and re-project the CLIPPED rotor forces back to wrench space for the update.
            wrench = base.unsqueeze(1) + noise                               # [B,N,H,4] wrench
            rotor = self.allocate(wrench)                                    # [B,N,H,4] rotor, in-box
            effective_noise = self.project(rotor) - base.unsqueeze(1)        # [B,N,H,4] wrench
            del wrench
        del noise

        # THE ROLLOUT IS ALWAYS SCORED ON THE CLIPPED ROTOR FORCES — the plant's true input.
        cost, all_collided = self._rollout_cost(x, rotor, goal, centers, radii, active)  # [B,N], [B]

        # charter "v5" Stage 1: the ARGMIN-COST sample's endpoint distance at this decision step. Read
        # off tensors already computed; it draws nothing and changes nothing below it.
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
            lam_eff: Any = self.params.lam                                   # python float (legacy)
            lam_eff_row = torch.full((batch,), float(self.params.lam), device=self.device, dtype=self.dtype)
        else:
            # R3: lambda relative to the running std of the sample costs, per (scene, control step).
            # The exponent (S_n - min_k S_k) / lam_eff is then invariant to any rescaling of S, so the
            # softmax cannot collapse to a hard argmin through a scale mismatch.
            lam_eff = torch.clamp(
                float(self.params.lam) * cost.std(dim=1, keepdim=True), min=float(self.params.lam_eps_abs)
            )                                                                # [B,1]
            lam_eff_row = lam_eff.squeeze(1)
        weight = torch.exp(-shifted / lam_eff)
        partition = weight.sum(dim=1, keepdim=True)
        degenerate = all_collided | ~torch.isfinite(partition).squeeze(1) | (partition.squeeze(1) <= 0.0)
        weight = weight / torch.clamp(partition, min=torch.finfo(self.dtype).tiny)

        # ESS on the NORMALISED weights, so ESS in [1, N]: 1 = hard argmin, N = uniform.
        ess = 1.0 / torch.clamp(weight.square().sum(dim=1), min=torch.finfo(self.dtype).tiny)

        # ---- charter "v3" B3: adaptive temperature -----------------------------------------------
        # Where ESS = (sum w)^2 / sum w^2 falls below `ess_frac_of_n * N`, multiply lambda by
        # `lam_factor` and recompute the weights, the partition, the degenerate mask and ESS FOR THIS
        # DECISION STEP ONLY. Nothing persists: `lam_eff` itself is not mutated, the carried plan is
        # untouched, and the next step starts from the cell's own lambda again. The degenerate branch is
        # re-derived on the post-adaptation partition and remains the final fallback. With B3 off none of
        # this executes, so the weights are the pre-B3 ones byte for byte.
        ess_pre = ess
        lam_eff_pre = lam_eff_row
        b3_event = torch.zeros(batch, dtype=torch.bool, device=self.device)
        if self.recovery is not None and self.recovery.b3_enabled:
            b3_event = ess_pre < (self.recovery.b3_ess_frac_of_n * float(self.params.n_samples))
            if bool(b3_event.any()):
                lam_row2d = (
                    lam_eff if torch.is_tensor(lam_eff)
                    else torch.full((batch, 1), float(lam_eff), device=self.device, dtype=self.dtype)
                )
                lam_row2d = torch.where(
                    b3_event.view(-1, 1), lam_row2d * self.recovery.b3_lam_factor, lam_row2d
                )
                weight = torch.exp(-shifted / lam_row2d)
                partition = weight.sum(dim=1, keepdim=True)
                degenerate = (
                    all_collided | ~torch.isfinite(partition).squeeze(1) | (partition.squeeze(1) <= 0.0)
                )
                weight = weight / torch.clamp(partition, min=torch.finfo(self.dtype).tiny)
                ess = 1.0 / torch.clamp(weight.square().sum(dim=1), min=torch.finfo(self.dtype).tiny)
                lam_eff_row = lam_row2d.squeeze(1)

        # The update is written on the DEVIATION: u_plan <- u_plan + sum_n w_n (eff_n), with
        # eff_n = proj_n - (u_hover + u_plan), so the absolute plan becomes sum_n w_n proj_n exactly as
        # in the uncentered parameterisation — the anchor cancels algebraically and never drifts.
        update = self.plan + (weight.unsqueeze(-1).unsqueeze(-1) * effective_noise).sum(dim=1)
        if self.params.space == "rotor":
            # LEGACY: the box lives in u-space, so the clip is applied to the ABSOLUTE plan and the
            # result carried back to the deviation (a no-op under center=none, where the anchor is 0).
            updated = torch.clamp(self.absolute(update), self.u_lo, self.u_hi)
            if self.centered:
                updated = updated - self.anchor
        else:
            # No clip: the weights are normalised and the image of the box under the mixer is convex, so
            # `plan + sum_n w_n (proj_n - plan) = sum_n w_n proj_n` is a convex combination of achievable
            # wrenches and is therefore itself achievable. `allocate` clips anyway, defensively.
            updated = update
        # DEGENERATE BRANCH: skip the update, hold the carried plan (already shifted one step and
        # tail-appended at the end of the previous step) and count the event. The hold is on the
        # DEVIATION, so the absolute plan held is anchor + deviation — the same sequence the uncentered
        # branch holds.
        plan = torch.where(degenerate.view(-1, 1, 1), self.plan, updated)

        head = self.absolute(plan[:, 0])
        action = head.clone() if self.params.space == "rotor" else self.allocate(head)
        self.plan = self._shift(plan)
        self.last_degenerate = degenerate
        self.last_ess = ess
        self.last_ess_pre = ess_pre
        self.last_lam_eff_pre = lam_eff_pre
        self.last_b3_event = b3_event
        self.last_lam_eff = lam_eff_row
        self.last_action = action
        self.last_decision = True
        self.steps_since_reset += 1
        self.degenerate_steps = self.degenerate_steps + degenerate.long()
        assert self.b3_events is not None
        self.b3_events = self.b3_events + b3_event.long()
        return action

    @torch.no_grad()
    def _rollout_cost(
        self,
        x: Tensor,
        sampled: Tensor,
        goal: Tensor,
        centers: Tensor,
        radii: Tensor,
        active: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Roll the N sampled PER-ROTOR sequences of every scene through the deployed plant and accumulate S.

        A sample that collides is FROZEN at the collision state (mirroring the eval rollout, which holds x
        and zeroes u once an episode is physically done), is charged `C_crash` exactly once, and keeps
        being charged the running cost of that frozen state for the remaining horizon. Freezing without
        the continued charge would make an early crash CHEAPER than survival whenever
        C_crash < H * stage_cost, which at C_crash = 1e3, H = 40 and a far goal is a live inversion.

        S3 CHANGE 3: `sampled` holds H DECISION entries and the rollout runs H * control_hold PHYSICAL
        steps, the entry at physical step k being `held_control(sampled, k)`. C_crash is still charged
        once, and the stage cost is accumulated at EVERY physical step, so extending the lookahead by
        holding does not coarsen the integration or the collision predicate.

        MEMORY: the rollout's working set is [B*n_sub, .] with n_sub = min(N, params.sample_chunk).
        Samples are INDEPENDENT — each carries its own state, its own death flag and its own cost sum —
        so splitting the sample axis is exactly equivalent, not an approximation: it changes the wall
        time and the peak allocation, nothing else. (`sampled` is built in full before the split, so the
        RNG stream is identical too.) `cpu_smoke.py` check (e) asserts a chunked and an unchunked cell
        agree on every metric.

        Returns (cost [B, N], all_collided [B]).
        """
        batch, n, _, _ = sampled.shape
        step = int(self.params.sample_chunk) or n
        costs, deads = [], []
        self._endpoint_chunk_dists = []
        for start in range(0, n, step):
            chunk_cost, chunk_dead = self._rollout_chunk(
                x, sampled[:, start : start + step], goal, centers, radii, active
            )
            costs.append(chunk_cost)
            deads.append(chunk_dead)
        cost = torch.cat(costs, dim=1) if len(costs) > 1 else costs[0]
        dead = torch.cat(deads, dim=1) if len(deads) > 1 else deads[0]
        # charter "v5" Stage 1: re-assemble the per-sample endpoint distances over the SAME sample axis
        # the costs were concatenated along, so `cost.argmin(dim=1)` indexes both consistently.
        if self.endpoint_probe:
            self.last_endpoint_dist = (
                torch.cat(self._endpoint_chunk_dists, dim=1)
                if len(self._endpoint_chunk_dists) > 1 else self._endpoint_chunk_dists[0]
            )
            self._endpoint_chunk_dists = []
        return cost, dead.all(dim=1)

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
        """One sample-chunk of `_rollout_cost`. Returns (cost [B, n_sub], dead [B, n_sub])."""
        batch, n, horizon, m = sampled.shape
        flat = batch * n
        state = x.unsqueeze(1).expand(batch, n, -1).reshape(flat, -1)
        goal_flat = goal.unsqueeze(1).expand(batch, n, -1).reshape(flat, -1)
        cost = torch.zeros(flat, device=self.device, dtype=self.dtype)
        dead = torch.zeros(flat, dtype=torch.bool, device=self.device)

        for k in range(horizon * self.control_hold):                  # H DECISION entries, m steps each
            control = self.held_control(sampled, k).reshape(flat, m)
            nxt = rk4_step(self.system, state, control, self.dt)      # THE SHARED DEPLOYED PLANT MAP
            # charter "v4" G1 charges an approach INCREMENT, so the state the step was taken FROM is
            # carried into the stage cost. Binding the name costs nothing and changes no arithmetic; with
            # G1 off `stage_cost` never reads it. A frozen (collided) sample has state == previous, so
            # its increment is exactly zero.
            previous = state
            state = torch.where(dead.unsqueeze(-1), state, nxt)       # freeze collided samples
            collided = collision_mask(
                self.system.position(state).view(batch, n, -1), centers, radii, active, self.cost
            ).reshape(flat)
            newly = collided & ~dead
            cost = cost + newly.to(self.dtype) * self.cost.c_crash    # C_crash charged once
            dead = dead | collided
            cost = cost + stage_cost(
                self.system, state, goal_flat, self.cost, self.recovery, x_prev=previous
            )

        cost = cost + terminal_cost(self.system, state, goal_flat, self.cost)
        # charter "v5" Stage 1: the ENDPOINT of every sample of this chunk. Post-rollout arithmetic on
        # tensors the rollout already produced — it draws no random numbers and feeds nothing back into
        # `cost`, `dead` or the plan — and it is wrapped in `fork_rng` under the standing rule so added
        # instrumentation can never shift the stream a later draw takes from.
        if self.endpoint_probe:
            with torch.random.fork_rng(devices=self._fork_devices):
                self._endpoint_chunk_dists.append(
                    torch.linalg.norm(self.system.position(state) - goal_flat, dim=-1).view(batch, n)
                )
        return cost.view(batch, n), dead.view(batch, n)


class _DeviceAnchor(torch.nn.Module):
    """`src.eval.evaluate._tensor_options` resolves the rollout dtype/device from
    `framework.value_net`'s first parameter, falling back to `system.u_bounds` (float64/CPU). MPPI has no
    network, so this one-parameter module is what carries the requested dtype/device into the harness.
    It is never read, never trained, and never touches the control."""

    def __init__(self, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1, device=device, dtype=dtype))


class MPPIFramework:
    """Adapts MPPIController to the `src.eval.evaluate` framework protocol with an IDENTITY filter.

    MPPI is a planner, not a certificate arm: there is no barrier and nothing can be infeasible, so
    `filter` returns (u_nom, all-False) exactly as the PPO baseline's FilterFreeFramework does. The eval
    then reports `infeasibility == 0` and `mean_proj_mag == 0` by construction (STRUCTURALLY INAPPLICABLE,
    not a measured zero) and `cps = reach - 2*collision - stuck - 0.5*(oob + timeout)`.

    Episode bookkeeping: `rollout_eval` calls `policy` exactly `max_steps` times per scene chunk (at
    `eval.dt_ctrl == env.dt`, i.e. substeps = 1, which `evaluate_mppi.py` asserts), so the call counter
    below marks chunk boundaries and resets the carried plan there. The per-step degenerate masks, the
    per-step ESS and the per-step lam_eff of each chunk are stacked for the harness.

    S3 CHANGE 3: with a control hold m > 1 only every m-th call is a DECISION step; on a hold step the
    controller re-issues the latched action and computes no weights. A per-step DECISION mask is stacked
    alongside, and the degenerate flag / ESS / lam_eff of a hold step are written as the neutral values
    (False / NaN / NaN) so a held step can never be counted as if it carried its own weighting. At m = 1
    every step is a decision step and the logs are identical to the pre-S3 ones.
    """

    def __init__(
        self,
        system: System,
        controller: MPPIController,
        *,
        max_steps: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.system = system
        self.controller = controller
        self.max_steps = int(max_steps)
        self.value_net = _DeviceAnchor(device, dtype)
        self._calls = 0
        self._current: list[np.ndarray] = []
        self._current_ess: list[np.ndarray] = []
        self._current_lam: list[np.ndarray] = []
        self._current_decision: list[np.ndarray] = []
        self._current_b3: list[np.ndarray] = []
        self._current_ess_pre: list[np.ndarray] = []
        self.degenerate_chunks: list[np.ndarray] = []      # each [T, B_chunk] bool
        self.ess_chunks: list[np.ndarray] = []             # each [T, B_chunk] float32
        self.lam_eff_chunks: list[np.ndarray] = []         # each [T, B_chunk] float32
        self.decision_chunks: list[np.ndarray] = []        # each [T, B_chunk] bool (S3 control hold)
        # charter "v3": the B3 adaptation event mask and the PRE-adaptation ESS, per (step, episode)
        self.b3_chunks: list[np.ndarray] = []              # each [T, B_chunk] bool
        self.ess_pre_chunks: list[np.ndarray] = []         # each [T, B_chunk] float32
        # charter "v3": per-episode B2 firing, filled at each chunk's first decision step
        self.b2_fired_chunks: list[np.ndarray] = []        # each [B_chunk] bool
        self.b2_axis_degenerate_chunks: list[np.ndarray] = []
        self.spawn_tilt_chunks: list[np.ndarray] = []      # each [B_chunk] float64

    def policy(self, x: Tensor, scene: Any) -> Tensor:
        if self._calls % self.max_steps == 0:
            self.flush()
            self.controller.reset(int(x.shape[0]))
        action = self.controller.act(x, scene)
        assert self.controller.last_degenerate is not None
        assert self.controller.last_ess is not None and self.controller.last_lam_eff is not None
        decision = bool(self.controller.last_decision)
        batch = int(x.shape[0])
        if decision:
            self._current.append(self.controller.last_degenerate.detach().cpu().numpy().copy())
            self._current_ess.append(
                self.controller.last_ess.detach().to(torch.float32).cpu().numpy().copy()
            )
            self._current_lam.append(
                self.controller.last_lam_eff.detach().to(torch.float32).cpu().numpy().copy()
            )
            assert self.controller.last_b3_event is not None
            assert self.controller.last_ess_pre is not None
            self._current_b3.append(self.controller.last_b3_event.detach().cpu().numpy().copy())
            self._current_ess_pre.append(
                self.controller.last_ess_pre.detach().to(torch.float32).cpu().numpy().copy()
            )
        else:
            # a HOLD step carries no weighting of its own: neutral entries, masked out downstream.
            self._current.append(np.zeros(batch, dtype=bool))
            self._current_ess.append(np.full(batch, np.nan, dtype=np.float32))
            self._current_lam.append(np.full(batch, np.nan, dtype=np.float32))
            self._current_b3.append(np.zeros(batch, dtype=bool))
            self._current_ess_pre.append(np.full(batch, np.nan, dtype=np.float32))
        self._current_decision.append(np.full(batch, decision, dtype=bool))
        if self._calls % self.max_steps == 0:
            # the chunk's FIRST call has just run, so B2's firing decision (if any) is now made
            assert self.controller.b2_fired is not None
            assert self.controller.b2_axis_degenerate is not None
            assert self.controller.spawn_tilt_deg is not None
            self.b2_fired_chunks.append(self.controller.b2_fired.detach().cpu().numpy().copy())
            self.b2_axis_degenerate_chunks.append(
                self.controller.b2_axis_degenerate.detach().cpu().numpy().copy()
            )
            self.spawn_tilt_chunks.append(
                self.controller.spawn_tilt_deg.detach().to(torch.float64).cpu().numpy().copy()
            )
        self._calls += 1
        return action.to(device=x.device, dtype=x.dtype)

    def filter(self, x: Tensor, u_nom: Tensor, scene: Any) -> tuple[Tensor, Tensor]:
        # Identity filter: MPPI carries no certificate, so the executed action IS the planner's action.
        return u_nom, torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)

    def flush(self) -> None:
        """Close the current chunk's per-step logs. Call once after `evaluate` returns."""
        if self._current:
            self.degenerate_chunks.append(np.stack(self._current, axis=0))
            self.ess_chunks.append(np.stack(self._current_ess, axis=0))
            self.lam_eff_chunks.append(np.stack(self._current_lam, axis=0))
            self.decision_chunks.append(np.stack(self._current_decision, axis=0))
            self.b3_chunks.append(np.stack(self._current_b3, axis=0))
            self.ess_pre_chunks.append(np.stack(self._current_ess_pre, axis=0))
            self._current = []
            self._current_ess = []
            self._current_lam = []
            self._current_decision = []
            self._current_b3 = []
            self._current_ess_pre = []
