"""v2.8.4 — MPPI optimizer parity, items B / D / E, for the (T, omega_des) cascade.

SUBJECT. `src/frameworks/mppi/cascade_rate.py` — the (T, omega) plan with attitude kinematics in the
planner and a body-rate inner loop. `cascade.py` (the v5 a_des plan) is NOT the subject and is not
imported. Cost, inner loop, post-allocation rotor clip, collision predicate and the full RK4 rotor-thrust
simulation plant are UNCHANGED and are CALLED, never re-implemented. There is no reference trajectory and
no global planner anywhere in this file.

ADDITIVE. Nothing here is imported by any existing run; this module only reads the shipped objects. It
writes `data/runs/v2.8.4/mppi_parity/**` and nothing else.

WHAT IS COMPARED. Our `RateCascadeController` against `pytorch_mppi.MPPI` 0.9.1 (Williams et al. 2017,
algorithm 2), with our planner model and our cost objects wired into the reference as its three
callbacks. Function parity is the standard (02_control section 8), not trajectory bit-parity.

THE THREE CONTROLS, as the dispatch fixes them.

  (1) INITIALISATION. Our plan initialises FLAT at the hover trim (`mppi_controller.py:464-469`, filled
      with `plan_trim`, which under `center: hover` is the zero deviation whose absolute value is
      `hover_trim = (m*g, 0, 0, 0)`); the reference initialises `self.U` from a NOISE DRAW
      (`pytorch_mppi/mppi.py:144-145`, and again in `reset()` at `:290`) unless `U_init` is passed. Our
      flat trim is therefore passed as the reference's `U_init` (`mppi.py:52,141`) and as its `u_init`
      (`mppi.py:51,142`), which is the value its shift appends as the tail (`mppi.py:238`) and is exactly
      what our `_shift` appends (`mppi_controller.py:501-505`). Both nominals live in ABSOLUTE (T, omega)
      units, which is the space the planner model's `dynamics` consumes.

  (2) SHIFT ALIGNMENT. Ours shifts at the END of `act` (`cascade_rate.py:722`); the reference shifts at
      the START of `command` (`mppi.py:249-250`). They are aligned by calling the reference's own
      `shift_nominal_trajectory()` (`mppi.py:232`) explicitly at the top of each step and then calling
      `command(..., shift_nominal_trajectory=False)` — which `mppi.py:249-252` makes exactly equivalent
      to the default call. The observation point for both implementations is then the same: the nominal
      IN FORCE at the sampling moment of step t. Neutrality is MEASURED, not asserted: at step 0 the
      shift of a flat trim sequence with a trim tail is the identity, `torch.equal` is recorded, and at
      every later step `shift(u_nom_out(t-1)) == u_nom_in(t)` is checked with each side's OWN shift.

  (3) IDENTICAL NOISE. Our sampler draws from a controller-owned `torch.Generator`
      (`mppi_controller.py:581-605`); the reference draws `torch.randn` from the GLOBAL default generator
      (`mppi.py:203`). Identical noise IS reachable without editing either implementation: seed the
      controller's generator and torch's global generator with the SAME seed, keep the scene batch at 1
      so the two draw shapes (1, K, H, 4) and (K, H, 4) have the same element count and the same
      row-major order, and set the reference's `noise_sigma` to `diag(sigma_channel^2)` so its diagonal
      factor `sqrt(diag)` is our per-channel std. Identity is then VERIFIED with `torch.equal` at every
      step, never assumed. Matched Sigma additionally requires our sampler in its `noise: iid` mode
      (a shipped `MPPIParams.noise` option, `mppi_controller.py:183`): under `noise: ou` our draw is
      temporally correlated and no single Sigma of the reference's iid law can match it. Both are
      reported; nothing is tuned toward agreement.

The reference has no relative-lambda rule. Item D therefore matches lambda by assigning our own resolved
`lam_eff` (`cascade_rate.py:699-703`, read back off `controller.last_lam_eff`) to `mppi.lambda_` before
each `command`; item E, whose two legs must each stand alone, uses the ABSOLUTE temperature read from
`mppi.v5.base_cell.lam` on both sides and reports the resulting weighting spread as item E(iii) requires.

Run:  CUDA_VISIBLE_DEVICES="" python -m src.frameworks.mppi.parity_rate --stage all
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from src._version import __version__
from src.common.observation import scene_goal_tensor
from src.common.rk4 import rk4_step
from src.common.system import System
from src.envs.scene_batch import batch_scenes
from src.frameworks.mppi import smoke_cascade_rate as smoke
from src.frameworks.mppi.cascade_rate import RateCascadeController, _W
from src.frameworks.mppi.cost import collision_mask, stage_cost, terminal_cost
from src.frameworks.mppi.evaluate_mppi import (
    REPO,
    build_framework,
    effective_config,
    load_mppi_config,
)
from src.frameworks.mppi.screen_cascaded import cascade_kwargs
from src.frameworks.mppi.stages_v5 import base_kwargs

from pytorch_mppi.mppi import MPPI


Tensor = torch.Tensor

OUT_DIR = REPO / "data/runs/v2.8.4/mppi_parity"

# READ-ONLY. Recorded before and after every stage; never written by this file.
IMMUTABLE_DIRS = (
    REPO / "data/runs/v2.8.4/mppi_screen",
    REPO / "data/runs/v2.8.4/mppi_screen_v2",
    REPO / "data/runs/v2.8.4/mppi_screen_v3",
    REPO / "data/runs/v2.8.4/mppi_v3",
    REPO / "data/runs/v2.8.4/mppi_diag",
    REPO / "data/runs/v2.8.4/mppi_v4",
    REPO / "data/runs/v2.8.4/mppi_v5",
    REPO / "data/runs/v2.8.4/mppi_cascaded",
)

# ---- the tolerance, and why it is what it is ----------------------------------------------------
# float64 carries eps = 2^-52 ~ 2.22e-16. The two implementations accumulate the SAME terms in
# DIFFERENT associations: ours adds the crash charge and the stage cost to the running total in two
# separate statements (`cascade_rate.py:620,622`), the reference adds their sum once (`mppi.py:319`);
# ours sums the terminal after the loop (`cascade_rate.py:626`), the reference after its own loop
# (`mppi.py:328`). One rollout performs H = 40 steps x (4 RK4 dynamics evaluations + a handful of cost
# reductions) ~ 1e3 dependent floating-point accumulations, so re-association alone can move a cost by
# at most ~1e3 * eps ~ 2e-13 in RELATIVE terms. RTOL is set three orders of magnitude above that bound
# and ATOL guards quantities whose true value is zero:
RTOL = 1.0e-10
ATOL = 1.0e-12
TOLERANCE_NOTE = (
    "float64 eps = 2.220446049250313e-16. The two implementations accumulate the same terms in "
    "different associations (ours: cascade_rate.py:620 then :622 then :626; the reference: mppi.py:319 "
    "then :328), and one rollout is ~1e3 dependent accumulations, so re-association alone bounds the "
    "relative difference by ~1e3*eps ~ 2e-13. rtol = 1e-10 sits three orders of magnitude above that "
    "bound; atol = 1e-12 guards quantities whose exact value is zero. A difference ABOVE this tolerance "
    "cannot be explained by floating-point re-association and is therefore a FUNCTIONAL difference; a "
    "difference below it is not evidence of one. Function parity is the standard (02_control section 8), "
    "not trajectory bit-parity."
)


def within(a: Tensor | float, b: Tensor | float) -> bool:
    at = torch.as_tensor(a, dtype=torch.float64)
    bt = torch.as_tensor(b, dtype=torch.float64)
    return bool(torch.all((at - bt).abs() <= ATOL + RTOL * bt.abs()))


def maxdiff(a: Tensor, b: Tensor) -> dict[str, float]:
    d = (a.double() - b.double()).abs()
    reference = b.double().abs()
    nonzero = reference > 0.0
    max_rel = float((d[nonzero] / reference[nonzero]).max().item()) if bool(nonzero.any()) else 0.0
    return {
        "max_abs": float(d.max().item()),
        "max_rel_over_nonzero": max_rel,
        "bit_identical": bool(torch.equal(a, b)),
        "within_tolerance": within(a, b),
    }


# =================================================================================================
# process / immutability bookkeeping
# =================================================================================================
def proc_state() -> dict[str, Any]:
    pid = os.getpid()
    state = ""
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("State:"):
                state = line.split(":", 1)[1].strip()
                break
    except OSError:                                                          # pragma: no cover
        state = "unavailable"
    return {"pid": pid, "ppid": os.getppid(), "proc_state": state,
            "launched_with": "plain subprocess; NO setsid and NO nohup"}


def dir_state() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for p in IMMUTABLE_DIRS:
        st = p.stat()
        deep = max((q.stat().st_mtime_ns for q in p.rglob("*")), default=0)
        out[str(p.relative_to(REPO))] = {
            "mtime": st.st_mtime,
            "mtime_ns": st.st_mtime_ns,
            "mtime_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st.st_mtime))
                         + f".{st.st_mtime_ns % 1_000_000_000:09d}",
            "n_entries": len(list(p.iterdir())),
            "deepest_file_mtime_ns": deep,
        }
    return out


def write_report(name: str, report: dict[str, Any]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    if path.exists():
        path = path.with_name(f"{path.stem}__{time.strftime('%H%M%S')}{path.suffix}")
    path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"wrote {path}", flush=True)
    return path


# =================================================================================================
# ITEM B — the three reference callbacks
# =================================================================================================
class WiredCallbacks:
    """Our cascade's planner model and our cost objects, exposed as the reference's three callbacks.

    Every shipped function is CALLED. The only code here is the plumbing the reference's signatures
    require: the reference's `dynamics(state, u)` carries no rollout-local state, so the low-pass
    recursion is carried in the STATE's own omega slot — which is exactly where `cascade_rate.py:612`
    writes it, and where it survives the step because the planner's `dot omega = 0`
    (`cascade_rate.py:202`) and the plant's PER-COMPONENT `|omega| <= omega_max` wrap
    (`quadrotor_3d.wrap_state:197`) leave an in-range rate untouched. The collision freeze and the
    charged-once C_crash are per-rollout state and are held on this object, reset before every
    `command`. Agreement with our own `_rollout_cost` is MEASURED at every step, never assumed.

    Shipped functions called
    ------------------------
    src/frameworks/mppi/cascade_rate.py:190   `_RatePlannerModel.dynamics`   (through rk4_step)
    src/frameworks/mppi/cascade_rate.py:204   `_RatePlannerModel.wrap_state` (through rk4_step)
    src/common/rk4.py:11                      `rk4_step`
    src/frameworks/mppi/cost.py:238           `collision_mask`
    src/frameworks/mppi/cost.py:305           `stage_cost`
    src/frameworks/mppi/cost.py:359           `terminal_cost`
    src/common/system.py                      `System.position`, `System.speed`, `System.angular_rate`,
                                              `System.thrust_axis` (through the four above)
    """

    def __init__(
        self,
        controller: RateCascadeController,
        system: System,
        goal_row: Tensor,
        centers: Tensor,
        radii: Tensor,
        active: Tensor,
        k: int,
    ) -> None:
        self.c = controller
        self.system = system
        self.k = int(k)
        self.goal = goal_row.expand(self.k, -1)
        self.centers, self.radii, self.active = centers, radii, active
        self.reset()

    def reset(self) -> None:
        self.dead = torch.zeros(self.k, dtype=torch.bool, device=self.c.device)
        self.prev: Tensor | None = None
        zero = torch.zeros(self.k, dtype=self.c.dtype, device=self.c.device)
        self.stage_sum = zero.clone()
        self.crash_sum = zero.clone()
        self.terminal_sum = zero.clone()
        self.min_dist = torch.full((self.k,), float("inf"), dtype=self.c.dtype, device=self.c.device)

    # -- callback 1 -------------------------------------------------------------------------------
    def dynamics(self, state: Tensor, u: Tensor) -> Tensor:
        """`cascade_rate.py:610-615` with the low-pass carried in the state's own omega slot."""
        beta = self.c.lowpass_beta
        rate = beta * state[:, _W] + (1.0 - beta) * u[:, 1:]
        pre = torch.cat([state[:, :10], rate], dim=1)
        nxt = rk4_step(self.c.planner, pre, u, self.c.dt)        # THE PLANNER'S OWN MODEL, shipped
        self.prev = pre
        return torch.where(self.dead.unsqueeze(-1), pre, nxt)    # freeze collided samples

    # -- callback 2 -------------------------------------------------------------------------------
    def running_cost(self, state: Tensor, u: Tensor) -> Tensor:
        """`cascade_rate.py:616-624`: the SAME predicate, the charged-once C_crash, the SAME stage cost."""
        position = self.system.position(state)
        collided = collision_mask(
            position.view(1, self.k, -1), self.centers, self.radii, self.active, self.c.cost
        ).reshape(self.k)
        newly = collided & ~self.dead
        crash = newly.to(self.c.dtype) * self.c.cost.c_crash
        self.dead = self.dead | collided
        stage = stage_cost(self.system, state, self.goal, self.c.cost, self.c.recovery, x_prev=self.prev)
        self.crash_sum = self.crash_sum + crash
        self.stage_sum = self.stage_sum + stage
        self.min_dist = torch.minimum(
            self.min_dist, torch.linalg.norm(position - self.goal, dim=-1)
        )
        return crash + stage

    # -- callback 3 -------------------------------------------------------------------------------
    def terminal_state_cost(self, states: Tensor, actions: Tensor | None) -> Tensor:
        """`cascade_rate.py:626`: the settling terminal on the final rollout state."""
        final = states[0, :, -1]
        value = terminal_cost(self.system, final, self.goal, self.c.cost)
        self.terminal_sum = value
        return value

    # -- the decomposition item E(v) asks for -----------------------------------------------------
    def components(self) -> dict[str, Tensor]:
        return {"goal_running": self.stage_sum, "settling_terminal": self.terminal_sum,
                "collision": self.crash_sum}

    def total(self) -> Tensor:
        return self.stage_sum + self.crash_sum + self.terminal_sum


def roll_block(cb: WiredCallbacks, x: Tensor, sequences: Tensor) -> Tensor:
    """Roll `sequences` [M, H, 4] from state `x` [1, nx] through the wired callbacks; returns the total
    cost [M] and leaves the decomposition on `cb`. `cb` must already be reset."""
    state = x.expand(cb.k, -1)
    for t in range(sequences.shape[1]):
        u = sequences[:, t]
        state = cb.dynamics(state, u)
        cb.running_cost(state, u)
    cb.terminal_state_cost(state.view(1, cb.k, 1, -1), None)
    return cb.total()


def roll_one(cb_factory: Callable[[int], WiredCallbacks], sequence: Tensor, x: Tensor) -> float:
    """Item E(iv): roll ONE (T, omega) sequence [H, 4] and report its closest approach, through exactly
    the callbacks the reference and our own rollout share."""
    cb = cb_factory(1)
    roll_block(cb, x, sequence.unsqueeze(0))
    return float(cb.min_dist.min().item())


# =================================================================================================
# construction
# =================================================================================================
def build(
    mppi_config: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    n_samples: int,
    noise: str,
    lam_mode: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[System, Any, RateCascadeController]:
    smoke_cfg = mppi_config["cascaded"]["smoke"]
    kwargs = base_kwargs(mppi_config)
    kwargs["horizon"] = int(smoke_cfg["straight_line"]["mppi_horizon"])
    kwargs["noise"] = str(noise)
    kwargs["lam_mode"] = str(lam_mode)
    system, framework, _, _ = build_framework(
        config, mppi_config, n_samples=int(n_samples), sample_chunk=0,
        seed=int(mppi_config["cascaded"]["scale"]["seed"]),
        rate_cascade=cascade_kwargs(mppi_config, float(smoke_cfg["rate_gain_factor"])),
        device=device, dtype=dtype, **kwargs,
    )
    return system, framework, framework.controller


def make_reference(
    controller: RateCascadeController, system: System, cb: WiredCallbacks, lam: float
) -> MPPI:
    """The reference, at matched (K, T, lambda, Sigma, horizon, dynamics, cost).

    CONTROL 1 — `U_init` and `u_init` are our flat hover trim, read off the controller
    (`controller.hover_trim`, built at `mppi_controller.py:328-331` from the system constants).
    The box passed as `u_min`/`u_max` is our own `rate_limit` (`cascade_rate.py:328-334`): the three
    rate channels bounded by `system.omega_max` read off the controller, the collective UNBOUNDED, which
    `torch.clamp` with +/-inf reproduces exactly.
    """
    infinity = float("inf")
    omega_max = controller.omega_max
    dtype, device = controller.dtype, controller.device
    horizon = int(controller.params.horizon)
    sigma_channel = controller.sigma_channel
    return MPPI(
        dynamics=cb.dynamics,
        running_cost=cb.running_cost,
        nx=int(system.state_dim),
        noise_sigma=torch.diag(sigma_channel * sigma_channel),
        num_samples=int(controller.params.n_samples),
        horizon=horizon,
        device=device,
        terminal_state_cost=cb.terminal_state_cost,
        lambda_=float(lam),
        u_min=torch.tensor([-infinity, -omega_max, -omega_max, -omega_max], dtype=dtype, device=device),
        u_max=torch.tensor([infinity, omega_max, omega_max, omega_max], dtype=dtype, device=device),
        u_init=controller.hover_trim.clone(),
        U_init=controller.hover_trim.view(1, -1).expand(horizon, -1).clone(),
    )


def reference_shift(ref: MPPI, sequence: Tensor) -> Tensor:
    """Apply THE REFERENCE'S OWN shift (`mppi.py:232-238`) to an arbitrary sequence, restoring `ref.U`.
    Used only to form `shift(u_nom(t-1))` for item E(ii) with each side's own shift operator."""
    keep = ref.U
    ref.U = sequence.clone()
    ref.shift_nominal_trajectory()
    shifted = ref.U
    ref.U = keep
    return shifted


def execute(controller: RateCascadeController, x: Tensor, head_absolute: Tensor) -> Tensor:
    """`cascade_rate.py:716-721`: the executed command, through the shipped `rate_limit`, the shipped
    low-pass constant and the shipped `inner_loop` (which is the ONLY place a rotor command exists and
    the only place the per-rotor box is applied). `count=False` so a reference leg never contributes to
    the subject's own pre-clip counters."""
    head = controller.rate_limit(head_absolute.view(1, -1))
    rate_applied = controller.lowpass_beta * x[:, _W] + (1.0 - controller.lowpass_beta) * head[:, 1:]
    command = torch.cat([head[:, :1], rate_applied], dim=-1)
    return controller.inner_loop(x, command, count=False)


def our_noise(controller: RateCascadeController, batch: int) -> Tensor:
    """The perturbation `act` is ABOUT to draw, captured WITHOUT consuming the stream: the generator's
    state is snapshotted, the shipped `_draw_noise` (`mppi_controller.py:581`) is called, and the state
    is restored, so the subsequent `act` draws exactly this tensor."""
    state = controller.generator.get_state()
    noise = controller._draw_noise(batch)
    controller.generator.set_state(state)
    return noise


def reference_raw_noise(ref: MPPI) -> Tensor:
    """The reference's raw draw, captured the same way from the GLOBAL generator."""
    state = torch.random.get_rng_state()
    noise = ref._sample_noise((ref.K, ref.T))
    torch.random.set_rng_state(state)
    return noise


def weights_from(cost: Tensor, lam: float, dtype: torch.dtype) -> tuple[Tensor, float, dict[str, float]]:
    """The exponent argument (C_k - rho)/lambda, the normalised weights and the ESS 1/sum(w^2), computed
    from a cost vector for MEASUREMENT ONLY (item E(iii)). Both implementations use rho = min_k C_k —
    ours at `cascade_rate.py:692`, the reference at `mppi.py:255`. The result is validated against
    `controller.last_ess` / `mppi.omega` at every step and the residual is reported."""
    shifted = cost - cost.min()
    exponent = shifted / lam
    weight = torch.exp(-exponent)
    weight = weight / torch.clamp(weight.sum(), min=torch.finfo(dtype).tiny)
    ess = float(1.0 / torch.clamp(weight.square().sum(), min=torch.finfo(dtype).tiny).item())
    spread = {
        "min": float(exponent.min().item()), "p50": float(exponent.median().item()),
        "p95": float(exponent.quantile(0.95).item()), "max": float(exponent.max().item()),
        "mean": float(exponent.mean().item()),
        "frac_exponent_gt_745": float((exponent > 745.0).double().mean().item()),
    }
    return weight, ess, spread


def our_updated_nominal(
    controller: RateCascadeController, plan_before: Tensor, effective_noise: Tensor, weight: Tensor
) -> tuple[Tensor, dict[str, Any]]:
    """The UNSHIFTED updated nominal our `act` formed this step, in PLAN (deviation) space.

    `act` writes it at `cascade_rate.py:712-714` and then immediately shifts it away
    (`cascade_rate.py:722`), so it is not observable afterwards. It is reconstructed here from the
    quantities that ARE observable — the carried plan, the effective noise and the weights — and the
    reconstruction is then PROVED by pushing it through the shipped `_shift` and comparing against the
    plan `act` actually carried away. The proof is recorded per step; nothing rests on the assumption.
    """
    update = plan_before + (weight.view(-1, 1, 1) * effective_noise).sum(dim=0)
    degenerate = bool(controller.last_degenerate[0].item())
    if degenerate:
        update = plan_before
    shifted = controller._shift(update.unsqueeze(0))[0]
    proof = maxdiff(shifted, controller.plan[0])
    proof["degenerate_row"] = degenerate
    return update, proof


def summarise(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    t = torch.as_tensor(values, dtype=torch.float64)
    return {"min": float(t.min().item()), "p50": float(t.median().item()),
            "p95": float(t.quantile(0.95).item()), "max": float(t.max().item()),
            "mean": float(t.mean().item())}


# =================================================================================================
# ITEM D — the parity comparison
# =================================================================================================
def item_d(mppi_config: dict[str, Any], config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    device, dtype = torch.device("cpu"), torch.float64
    smoke_cfg = mppi_config["cascaded"]["smoke"]
    seed = int(mppi_config["cascaded"]["scale"]["seed"])
    dt = float(config["env"]["dt"])

    system, _, controller = build(
        mppi_config, config, n_samples=int(args.d_samples), noise="iid", lam_mode="relative",
        device=device, dtype=dtype,
    )
    scene = smoke.straight_line_scene(
        system, [float(v) for v in smoke_cfg["straight_line"]["offset"]], dtype
    )
    batched = batch_scenes([scene], device=device, dtype=dtype)
    x = smoke.level_state(system, torch.as_tensor(scene.start, dtype=dtype).view(1, 3), dtype)

    controller.reset(1)
    centers, radii, active = controller._scene_tensors(batched)
    goal_row = scene_goal_tensor(batched, x)
    k = int(controller.params.n_samples)
    horizon = int(controller.params.horizon)

    cb = WiredCallbacks(controller, system, goal_row, centers, radii, active, k)
    ref = make_reference(controller, system, cb, lam=float(mppi_config["v5"]["base_cell"]["lam"]))

    # CONTROL 3 — the two streams are put in the same state, once.
    controller.generator.manual_seed(seed)
    torch.manual_seed(seed)

    steps: list[dict[str, Any]] = []
    first_difference: dict[str, Any] | None = None
    shift_identity_step0: bool | None = None
    previous_out_ours: Tensor | None = None
    previous_out_ref: Tensor | None = None

    for t in range(int(args.d_steps)):
        plan_before = controller.plan.clone()                          # [1,H,4] deviation
        base = controller.absolute(plan_before)                        # [1,H,4] absolute, at sampling
        noise = our_noise(controller, 1)                               # what `act` will draw
        sampled = controller.rate_limit(base.unsqueeze(1) + noise)     # cascade_rate.py:672-673
        effective_noise = sampled - base.unsqueeze(1)                  # cascade_rate.py:677
        cost_ours, _ = controller._rollout_cost(x, sampled, goal_row, centers, radii, active)

        # CONTROL 2 — the reference's own shift, called explicitly, at the top of the step.
        u_before_shift = ref.U.clone()
        ref.shift_nominal_trajectory()
        if t == 0:
            shift_identity_step0 = bool(torch.equal(u_before_shift, ref.U))
        nominal_in_ref = ref.U.clone()

        raw_ref = reference_raw_noise(ref)
        noise_identical = bool(torch.equal(noise[0], raw_ref))

        action_ours = controller.act(x, batched)
        lam_eff = float(controller.last_lam_eff[0].item())
        ess_ours = float(controller.last_ess[0].item())

        ref.lambda_ = lam_eff                                          # matched temperature
        cb.reset()
        ref.command(x[0], shift_nominal_trajectory=False)
        nominal_out_ref = ref.U.clone()

        rollout_ref = cb.total()
        action_cost = ref.lambda_ * ref.noise * ref._noise_sigma_inv_diag
        u_at_cost = ref.perturbed_action - ref.noise
        perturbation_cost = torch.sum(u_at_cost * action_cost, dim=(1, 2))

        weight_ref = ref.omega
        ess_ref = float(1.0 / weight_ref.square().sum().item())
        weight_ours, ess_replay, spread_ours = weights_from(cost_ours[0], lam_eff, dtype)
        update_dev, proof = our_updated_nominal(controller, plan_before[0], effective_noise[0], weight_ours)
        nominal_out_ours = controller.absolute(update_dev)

        row: dict[str, Any] = {
            "step": t,
            "u_nom_in": maxdiff(nominal_in_ref, base[0]),
            "raw_noise_identical": noise_identical,
            "perturbed_action": maxdiff(ref.perturbed_action, sampled[0]),
            "effective_noise": maxdiff(ref.noise, effective_noise[0]),
            "rollout_cost": maxdiff(rollout_ref, cost_ours[0]),
            "total_cost": maxdiff(ref.cost_total, cost_ours[0]),
            "reference_total_minus_rollout_is_the_perturbation_cost":
                maxdiff(ref.cost_total - rollout_ref, perturbation_cost),
            "perturbation_cost": {
                "min": float(perturbation_cost.min().item()),
                "max": float(perturbation_cost.max().item()),
                "std": float(perturbation_cost.std().item()),
                "ratio_to_rollout_cost_std": float(
                    (perturbation_cost.std() / cost_ours[0].std()).item()
                ),
            },
            "weights": maxdiff(weight_ref, weight_ours),
            "u_nom_out": maxdiff(nominal_out_ref, nominal_out_ours),
            "ess_ours_recorded": ess_ours, "ess_ours_replay": ess_replay, "ess_reference": ess_ref,
            "ess_replay_matches_controller": within(ess_replay, ess_ours),
            "our_updated_nominal_reconstruction_proof": proof,
            "lam_matched": lam_eff,
            "argmin_sample_ours": int(cost_ours[0].argmin().item()),
            "argmin_sample_reference": int(ref.cost_total.argmin().item()),
            "argmin_agrees": bool(cost_ours[0].argmin().item() == ref.cost_total.argmin().item()),
            "exponent_spread_ours": spread_ours,
        }
        if previous_out_ours is not None:
            row["shift_consistency_ours"] = maxdiff(
                controller._shift(previous_out_ours.unsqueeze(0))[0], plan_before[0]
            )
            row["shift_consistency_reference"] = maxdiff(
                reference_shift(ref, previous_out_ref), nominal_in_ref
            )
        previous_out_ours, previous_out_ref = update_dev.clone(), nominal_out_ref.clone()
        steps.append(row)

        if first_difference is None:
            ladder = [
                ("u_nom_in — the nominal carried into the step", row["u_nom_in"]),
                ("the raw noise draw", {"within_tolerance": noise_identical}),
                ("the perturbed action sequences", row["perturbed_action"]),
                ("the effective (post-bound) noise", row["effective_noise"]),
                ("the ROLLOUT cost C_k", row["rollout_cost"]),
                ("the TOTAL sample cost C_k", row["total_cost"]),
                ("the weights w_k", row["weights"]),
                ("u_nom_out — the updated nominal", row["u_nom_out"]),
            ]
            for name, entry in ladder:
                if not entry["within_tolerance"]:
                    first_difference = {
                        "quantity": name, "step": t, "detail": entry,
                        "ladder_position": [n for n, _ in ladder].index(name),
                        "ladder": [n for n, _ in ladder],
                    }
                    break

        x = rk4_step(system, x, action_ours, dt)          # TEACHER FORCING: one plant, ours

    return {
        "what": "item D — the two nominal control sequences at matched (K, T, lambda, Sigma, horizon, "
                "dynamics, cost), TEACHER-FORCED: one plant (ours) drives both optimizers, so every "
                "difference is the optimizer's and not the state's.",
        "variant": "cascaded_rate",
        "configuration": {
            "K_n_samples": k, "T_horizon": horizon,
            "lambda": "matched per step to our own resolved lam_eff (cascade_rate.py:699-703), read "
                      "back off controller.last_lam_eff and assigned to mppi.lambda_ before command",
            "Sigma": "diag(sigma_channel^2), sigma_channel read off the controller "
                     "(cascade_rate.py:294-301)",
            "sigma_channel": [float(v) for v in controller.sigma_channel.tolist()],
            "noise_law": "iid on BOTH sides — our MPPIParams.noise = 'iid'. Under 'ou' our draw is "
                         "temporally correlated and NO Sigma of the reference's iid law matches it.",
            "dynamics": "the wired callbacks of item B",
            "cost": "the wired callbacks of item B",
            "dt": controller.dt, "control_hold": controller.control_hold,
            "scene": "the smoke's obstacle-free straight-line scene",
            "n_steps": int(args.d_steps),
        },
        "tolerance": {"rtol": RTOL, "atol": ATOL, "justification": TOLERANCE_NOTE},
        "control_1_initialisation": {
            "U_init": [float(v) for v in controller.hover_trim.tolist()],
            "U_init_source": "controller.hover_trim (mppi_controller.py:328-331), our flat trim",
            "u_init_is_the_same_vector": True,
            "reference_default_would_be": "a noise draw, mppi.py:144-145 and :290",
        },
        "control_2_shift": {
            "ours": "END of act, cascade_rate.py:722",
            "reference": "START of command, mppi.py:249-250; here called explicitly through the "
                         "reference's own shift_nominal_trajectory() (mppi.py:232) and command(..., "
                         "shift_nominal_trajectory=False), which mppi.py:249-252 makes equivalent",
            "shift_at_step_0_is_the_identity": shift_identity_step0,
            "why_neutral": "with control 1 in force the step-0 nominal is flat at the trim and the tail "
                           "the shift appends IS that trim, so the shift is the identity there; from "
                           "step 1 on both observe the nominal at the same point of the cycle, which "
                           "the per-step shift_consistency entries verify with each side's own shift",
        },
        "control_3_noise": {
            "identical_noise_reachable_without_editing_either_implementation": True,
            "how": "seed the controller-owned generator (mppi_controller.py:349-350) and torch's global "
                   "default generator with the same seed; keep the scene batch at 1 so our (1,K,H,4) "
                   "draw and the reference's (K,H,4) draw (mppi.py:203) have the same element count and "
                   "row-major order; set noise_sigma = diag(sigma_channel^2) so the reference's "
                   "diagonal factor sqrt(diag) IS our per-channel std (verified with torch.equal).",
            "verified_every_step_with_torch_equal": all(r["raw_noise_identical"] for r in steps),
            "paired_seed_fallback_over_42_99_12345_needed": False,
        },
        "first_difference": first_difference,
        "steps": steps,
    }


# =================================================================================================
# ITEM E — the obstacle-free straight-line scene, closed loop
# =================================================================================================
def _episode_record() -> dict[str, list]:
    return {key: [] for key in (
        "distance_m", "nominal_increment", "nominal_minus_hover", "ess", "ess_replay",
        "exponent_spread", "lam", "share_goal", "share_terminal", "share_collision",
        "argmin_closest_approach_m", "weighted_average_closest_approach_m",
        "horizon_coverage_m", "distance_to_goal_at_decision_m", "coverage_exceeds_distance",
        "wired_vs_internal_rollout_cost", "component_goal", "component_terminal", "component_collision",
        "shift_consistency_max_abs", "reconstruction_proof_max_abs",
    )}


def _fill(
    record: dict[str, list], controller: RateCascadeController, system: System, cb: WiredCallbacks,
    cost: Tensor, ess: float, ess_replay: float, spread: dict[str, float], lam: float,
    nominal_in: Tensor, nominal_out: Tensor, hover_sequence: Tensor, sampled: Tensor,
    x: Tensor, goal: Tensor, cb_factory: Callable[[int], WiredCallbacks], horizon: int,
) -> None:
    # (ii) — nominal_in(t) IS shift(nominal_out(t-1)); the identity is checked separately.
    record["nominal_increment"].append(float(torch.linalg.norm(nominal_out - nominal_in).item()))
    record["nominal_minus_hover"].append(float(torch.linalg.norm(nominal_out - hover_sequence).item()))
    # (iii)
    record["ess"].append(ess)
    record["ess_replay"].append(ess_replay)
    record["exponent_spread"].append(spread)
    record["lam"].append(lam)
    # (iv)
    argmin = int(cost.argmin().item())
    record["argmin_closest_approach_m"].append(roll_one(cb_factory, sampled[argmin], x))
    record["weighted_average_closest_approach_m"].append(roll_one(cb_factory, nominal_out, x))
    # (v)
    components = cb.components()
    mean = {name: float(value.mean().item()) for name, value in components.items()}
    total = sum(mean.values())
    record["component_goal"].append(mean["goal_running"])
    record["component_terminal"].append(mean["settling_terminal"])
    record["component_collision"].append(mean["collision"])
    record["share_goal"].append(mean["goal_running"] / total if total else float("nan"))
    record["share_terminal"].append(mean["settling_terminal"] / total if total else float("nan"))
    record["share_collision"].append(mean["collision"] / total if total else float("nan"))
    # (vi)
    coverage = float(horizon) * controller.dt * float(system.v_max)
    distance = float(torch.linalg.norm(system.position(x) - goal).item())
    record["horizon_coverage_m"].append(coverage)
    record["distance_to_goal_at_decision_m"].append(distance)
    record["coverage_exceeds_distance"].append(bool(coverage >= distance))


def run_ours(
    mppi_config: dict[str, Any], config: dict[str, Any], *, noise: str, lam_mode: str,
    n_samples: int, label: str,
) -> dict[str, Any]:
    device, dtype = torch.device("cpu"), torch.float64
    smoke_cfg = mppi_config["cascaded"]["smoke"]
    seed = int(mppi_config["cascaded"]["scale"]["seed"])
    dt = float(config["env"]["dt"])
    goal_radius = float(config["env"]["goal_radius"])
    lam_absolute = float(mppi_config["v5"]["base_cell"]["lam"])

    system, _, controller = build(
        mppi_config, config, n_samples=n_samples, noise=noise, lam_mode=lam_mode,
        device=device, dtype=dtype,
    )
    scene = smoke.straight_line_scene(
        system, [float(v) for v in smoke_cfg["straight_line"]["offset"]], dtype
    )
    batched = batch_scenes([scene], device=device, dtype=dtype)
    x = smoke.level_state(system, torch.as_tensor(scene.start, dtype=dtype).view(1, 3), dtype)
    goal = torch.as_tensor(scene.goal, dtype=dtype).view(1, 3)
    seconds = float(smoke_cfg["straight_line"]["seconds"])
    n_steps = int(round(seconds / dt))

    controller.reset(1)
    controller.generator.manual_seed(seed)
    centers, radii, active = controller._scene_tensors(batched)
    goal_row = scene_goal_tensor(batched, x)
    k = int(controller.params.n_samples)
    horizon = int(controller.params.horizon)

    def cb_factory(m: int) -> WiredCallbacks:
        return WiredCallbacks(controller, system, goal_row, centers, radii, active, m)

    cb = cb_factory(k)
    hover_sequence = controller.hover_trim.view(1, -1).expand(horizon, -1)

    record = _episode_record()
    previous_out: Tensor | None = None
    for _ in range(n_steps):
        plan_before = controller.plan.clone()
        base = controller.absolute(plan_before)
        noise_draw = our_noise(controller, 1)
        sampled = controller.rate_limit(base.unsqueeze(1) + noise_draw)
        effective_noise = sampled - base.unsqueeze(1)
        cost_internal, _ = controller._rollout_cost(x, sampled, goal_row, centers, radii, active)
        cb.reset()
        cost_wired = roll_block(cb, x, sampled[0])
        record["wired_vs_internal_rollout_cost"].append(maxdiff(cost_wired, cost_internal[0]))

        if previous_out is not None:
            record["shift_consistency_max_abs"].append(
                maxdiff(controller._shift(previous_out.unsqueeze(0))[0], plan_before[0])["max_abs"]
            )

        action = controller.act(x, batched)
        lam_eff = float(controller.last_lam_eff[0].item())
        ess_controller = float(controller.last_ess[0].item())
        weight, ess_replay, spread = weights_from(cost_internal[0], lam_eff, dtype)
        update_dev, proof = our_updated_nominal(controller, plan_before[0], effective_noise[0], weight)
        record["reconstruction_proof_max_abs"].append(proof["max_abs"])
        nominal_out = controller.absolute(update_dev)
        previous_out = update_dev.clone()

        _fill(
            record, controller, system, cb, cost_internal[0], ess_controller, ess_replay, spread,
            lam_eff, base[0], nominal_out, hover_sequence, sampled[0], x, goal, cb_factory, horizon,
        )

        x = rk4_step(system, x, action, dt)
        record["distance_m"].append(float(torch.linalg.norm(system.position(x) - goal).item()))

    return _close(
        record, controller, system, goal_radius, n_steps, seconds, label,
        implementation="ours — src/frameworks/mppi/cascade_rate.py RateCascadeController.act",
        noise=noise, lam_mode=lam_mode, lam_absolute=lam_absolute, horizon=horizon, k=k,
    )


def run_reference(
    mppi_config: dict[str, Any], config: dict[str, Any], *, n_samples: int, label: str,
) -> dict[str, Any]:
    device, dtype = torch.device("cpu"), torch.float64
    smoke_cfg = mppi_config["cascaded"]["smoke"]
    seed = int(mppi_config["cascaded"]["scale"]["seed"])
    dt = float(config["env"]["dt"])
    goal_radius = float(config["env"]["goal_radius"])
    lam_absolute = float(mppi_config["v5"]["base_cell"]["lam"])

    system, _, controller = build(
        mppi_config, config, n_samples=n_samples, noise="iid", lam_mode="absolute",
        device=device, dtype=dtype,
    )
    scene = smoke.straight_line_scene(
        system, [float(v) for v in smoke_cfg["straight_line"]["offset"]], dtype
    )
    batched = batch_scenes([scene], device=device, dtype=dtype)
    x = smoke.level_state(system, torch.as_tensor(scene.start, dtype=dtype).view(1, 3), dtype)
    goal = torch.as_tensor(scene.goal, dtype=dtype).view(1, 3)
    seconds = float(smoke_cfg["straight_line"]["seconds"])
    n_steps = int(round(seconds / dt))

    controller.reset(1)
    torch.manual_seed(seed)
    centers, radii, active = controller._scene_tensors(batched)
    goal_row = scene_goal_tensor(batched, x)
    k = int(controller.params.n_samples)
    horizon = int(controller.params.horizon)

    def cb_factory(m: int) -> WiredCallbacks:
        return WiredCallbacks(controller, system, goal_row, centers, radii, active, m)

    cb = cb_factory(k)
    ref = make_reference(controller, system, cb, lam=lam_absolute)
    hover_sequence = controller.hover_trim.view(1, -1).expand(horizon, -1)

    record = _episode_record()
    previous_out: Tensor | None = None
    for _ in range(n_steps):
        ref.shift_nominal_trajectory()                      # CONTROL 2, the reference's own shift
        nominal_in = ref.U.clone()
        if previous_out is not None:
            record["shift_consistency_max_abs"].append(
                maxdiff(reference_shift(ref, previous_out), nominal_in)["max_abs"]
            )
        cb.reset()
        ref.command(x[0], shift_nominal_trajectory=False)
        nominal_out = ref.U.clone()
        previous_out = nominal_out.clone()

        cost = ref.cost_total
        ess = float(1.0 / ref.omega.square().sum().item())
        _, ess_replay, spread = weights_from(cost, float(ref.lambda_), dtype)
        _fill(
            record, controller, system, cb, cost, ess, ess_replay, spread, float(ref.lambda_),
            nominal_in, nominal_out, hover_sequence, ref.perturbed_action, x, goal, cb_factory, horizon,
        )

        action = execute(controller, x, nominal_out[0])
        x = rk4_step(system, x, action, dt)
        record["distance_m"].append(float(torch.linalg.norm(system.position(x) - goal).item()))

    return _close(
        record, controller, system, goal_radius, n_steps, seconds, label,
        implementation="pytorch_mppi 0.9.1 — pytorch_mppi/mppi.py MPPI.command",
        noise="iid", lam_mode="absolute", lam_absolute=lam_absolute, horizon=horizon, k=k,
    )


def _close(
    record: dict[str, list], controller: RateCascadeController, system: System, goal_radius: float,
    n_steps: int, seconds: float, label: str, *, implementation: str, noise: str, lam_mode: str,
    lam_absolute: float, horizon: int, k: int,
) -> dict[str, Any]:
    distance = record["distance_m"]
    spreads = record["exponent_spread"]
    total_goal = sum(record["component_goal"])
    total_terminal = sum(record["component_terminal"])
    total_collision = sum(record["component_collision"])
    grand = total_goal + total_terminal + total_collision
    wired = record["wired_vs_internal_rollout_cost"]
    return {
        "variant": "cascaded_rate",
        "label": label,
        "implementation": implementation,
        "sampler_noise": noise,
        "lam_mode": lam_mode,
        "lam_absolute_read_from_config": lam_absolute if lam_mode == "absolute" else None,
        "N_samples": k, "horizon": horizon, "dt": controller.dt,
        "n_control_steps": n_steps, "seconds": seconds,
        "goal_radius_read_from_config": goal_radius,
        # (i)
        "closest_approach_m": min(distance),
        "final_distance_m": distance[-1],
        "reaches_position_radius": bool(min(distance) <= goal_radius),
        # (ii)
        "nominal_increment_norm": summarise(record["nominal_increment"]),
        "nominal_minus_hover_norm": summarise(record["nominal_minus_hover"]),
        "nominal_increment_definition":
            "||u_nom(t) - shift(u_nom(t-1))||_F with u_nom(t) the UPDATED nominal of step t and "
            "shift(u_nom(t-1)) the nominal carried into step t; the identity between the two is checked "
            "with each side's own shift operator and reported as shift_consistency_max_abs.",
        "shift_consistency_max_abs": max(record["shift_consistency_max_abs"], default=None),
        "our_updated_nominal_reconstruction_max_abs":
            max(record["reconstruction_proof_max_abs"], default=None),
        # (iii)
        "ess": summarise(record["ess"]),
        "ess_replay_vs_recorded_max_abs": max(
            (abs(a - b) for a, b in zip(record["ess"], record["ess_replay"])), default=None
        ),
        "exponent_spread_p50": summarise([s["p50"] for s in spreads]),
        "exponent_spread_max": summarise([s["max"] for s in spreads]),
        "exponent_frac_gt_745": summarise([s["frac_exponent_gt_745"] for s in spreads]),
        "exponent_definition": "(C_k - rho)/lambda with rho = min_k C_k; 745 is where exp(-e) "
                               "underflows to 0 in float64, so the fraction above it is the fraction of "
                               "samples that carry no weight at all",
        "lam_used": summarise(record["lam"]),
        # (iv)
        "argmin_rollout_closest_approach_m": summarise(record["argmin_closest_approach_m"]),
        "weighted_average_rollout_closest_approach_m":
            summarise(record["weighted_average_closest_approach_m"]),
        "iv_note": "BOTH are reported and NEITHER is concluded from: the minimum-cost sampled rollout "
                   "and the weighted-average nominal are two different objects, and the closed loop "
                   "executes only the head of the second.",
        # (v)
        "cost_shares_episode": {
            "goal_running": total_goal / grand if grand else float("nan"),
            "settling_terminal": total_terminal / grand if grand else float("nan"),
            "collision": total_collision / grand if grand else float("nan"),
        },
        "cost_shares_per_step_p50": {
            "goal_running": summarise(record["share_goal"]).get("p50"),
            "settling_terminal": summarise(record["share_terminal"]).get("p50"),
            "collision": summarise(record["share_collision"]).get("p50"),
        },
        "cost_component_note": "goal_running is the SUM over the horizon of `stage_cost` (cost.py:305), "
                               "which carries the w_goal distance term and the two gated settling terms; "
                               "settling_terminal is `terminal_cost` (cost.py:359) at the final rollout "
                               "state; collision is the charged-once C_crash. Shares are of the "
                               "SAMPLE-MEAN cost.",
        # (vi)
        "horizon_coverage_m": record["horizon_coverage_m"][0],
        "horizon_coverage_expression": "horizon * dt * system.v_max, all three read from the config / "
                                       "system object",
        "v_max_read_from_system": float(system.v_max),
        "distance_to_goal_at_decision_m": summarise(record["distance_to_goal_at_decision_m"]),
        "coverage_exceeds_distance_all_steps": all(record["coverage_exceeds_distance"]),
        "coverage_exceeds_distance_frac": (
            sum(record["coverage_exceeds_distance"]) / len(record["coverage_exceeds_distance"])
        ),
        # wiring validation
        "wired_rollout_matches_internal": {
            "checked_steps": len(wired),
            "all_within_tolerance": all(w["within_tolerance"] for w in wired),
            "all_bit_identical": all(w["bit_identical"] for w in wired),
            "max_abs_over_episode": max(w["max_abs"] for w in wired),
        } if wired else {"checked_steps": 0,
                         "note": "the reference leg IS the wired rollout; nothing to cross-check"},
        "per_step": {
            "distance_m": distance,
            "nominal_increment": record["nominal_increment"],
            "nominal_minus_hover": record["nominal_minus_hover"],
            "ess": record["ess"],
            "lam": record["lam"],
            "share_goal": record["share_goal"],
            "share_terminal": record["share_terminal"],
            "share_collision": record["share_collision"],
            "argmin_closest_approach_m": record["argmin_closest_approach_m"],
            "weighted_average_closest_approach_m": record["weighted_average_closest_approach_m"],
            "distance_to_goal_at_decision_m": record["distance_to_goal_at_decision_m"],
        },
    }


# =================================================================================================
# main
# =================================================================================================
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("d", "e", "all"), default="all")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--d-steps", type=int, default=20)
    parser.add_argument("--d-samples", type=int, default=1024)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    torch.set_num_threads(int(args.threads))
    t0 = time.time()
    mppi_config = load_mppi_config()
    config = effective_config(mppi_config)
    smoke_cfg = mppi_config["cascaded"]["smoke"]
    n_samples = int(smoke_cfg["straight_line"]["mppi_n_samples"][0])
    suffix = f"__{args.tag}" if args.tag else ""

    before = dir_state()
    header = {
        "what": "v2.8.4 MPPI optimizer parity, items B / D / E — the (T, omega_des) cascade",
        "version": __version__,
        "subject": "src/frameworks/mppi/cascade_rate.py (the (T, omega) plan). cascade.py, the v5 a_des "
                   "plan, is NOT the subject and is not imported here.",
        "variant": "cascaded_rate",
        "reference": "pytorch_mppi 0.9.1 (Williams et al. 2017, algorithm 2)",
        "process": proc_state(),
        "device": "cpu", "dtype": "torch.float64",
        "torch_threads": int(args.threads),
        "immutable_before": before,
        "no_selection": "MEASUREMENT ONLY. No pool, no cell selection, no ledger row, no promotion and "
                        "no final score. Cascaded, rotor-direct and parity numbers never share a table "
                        "without a variant column.",
    }

    if args.stage in ("d", "all"):
        print("[parity] item D ...", flush=True)
        report = dict(header)
        report["item_D"] = item_d(mppi_config, config, args)
        report["immutable_after"] = dir_state()
        report["immutable_unchanged"] = report["immutable_before"] == report["immutable_after"]
        report["wall_s"] = round(time.time() - t0, 2)
        write_report(f"item_D_parity{suffix}.json", report)

    if args.stage in ("e", "all"):
        print("[parity] item E ...", flush=True)
        legs = {}
        print("[parity]   E row A — ours, shipped cascaded configuration (ou / relative)", flush=True)
        legs["ours_shipped_ou_relative"] = run_ours(
            mppi_config, config, noise="ou", lam_mode="relative", n_samples=n_samples,
            label="ours, shipped cascaded configuration (OU noise, relative lambda)",
        )
        print("[parity]   E row B — ours, matched configuration (iid / absolute)", flush=True)
        legs["ours_matched_iid_absolute"] = run_ours(
            mppi_config, config, noise="iid", lam_mode="absolute", n_samples=n_samples,
            label="ours, matched configuration (iid noise, absolute lambda)",
        )
        print("[parity]   E row C — the reference, matched configuration", flush=True)
        legs["reference_matched_iid_absolute"] = run_reference(
            mppi_config, config, n_samples=n_samples,
            label="pytorch_mppi 0.9.1, matched configuration (iid noise, absolute lambda)",
        )
        report = dict(header)
        report["item_E"] = {
            "what": "item E — the smoke's obstacle-free straight-line scene, closed loop on the FULL "
                    "RK4 rotor-thrust plant, one scene, per implementation.",
            "scene": {
                "offset_m": [float(v) for v in smoke_cfg["straight_line"]["offset"]],
                "seconds": float(smoke_cfg["straight_line"]["seconds"]),
                "obstacle_free": True,
                "source": "src/frameworks/mppi/smoke_cascade_rate.py:66 straight_line_scene",
            },
            "legs": legs,
        }
        report["immutable_after"] = dir_state()
        report["immutable_unchanged"] = report["immutable_before"] == report["immutable_after"]
        report["wall_s"] = round(time.time() - t0, 2)
        write_report(f"item_E_smoke_scene{suffix}.json", report)

    print(f"immutable dirs unchanged: {dir_state() == before}", flush=True)
    return 0


if __name__ == "__main__":                                                   # pragma: no cover
    raise SystemExit(main())
