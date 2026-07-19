"""v2.6.0 Stage 0 M2 — the h_star barrier and the exact (numerical) PNCBF value V^{h,pi}.

h_star(x,o) = phi(p,o) + c (v^T Re)                          (theory note Def 8.1 / Thm 5.3)
  phi(p,o)  : the OC signed-distance avoid generator (src/common/signed_h.py), Top-K, unchanged.
  Re        : the body thrust axis (-sin th, cos th)          (planar, SO(2)).
  c(v^T Re) : a single scalar term, the approach speed along the thrust axis; obstacle-agnostic (R4).

V^{h,pi}(x) = sup_{t>=0} h(x_t^pi) along the closed-loop flow of the nominal policy pi  (Prop 4.1).
Computed here EXACTLY by a differentiable finite-horizon rollout + running max (NO learning). Its
input sensitivity L_g V (via src.common.filter_hardnet._cbf_terms, which reconstructs g(x)
control-affine) is the object the P1 gate measures near B_0 (note Sec 8.5 E1).
"""
from __future__ import annotations

import math
from typing import Any, Callable

import torch

from src.common.rk4 import rk4_step
from src.common.signed_h import signed_h

Tensor = torch.Tensor


def thrust_axis(x: Tensor) -> Tensor:
    """Re = (-sin theta, cos theta) — the body thrust axis (note Def 2.1, planar)."""
    theta = x[..., 2]
    return torch.stack([-torch.sin(theta), torch.cos(theta)], dim=-1)


def approach_speed(x: Tensor) -> Tensor:
    """v^T Re : linear velocity projected on the thrust axis (obstacle-agnostic, R4)."""
    v = x[..., 3:5]
    return torch.sum(v * thrust_axis(x), dim=-1)


def phi_value(x: Tensor, scene: Any, h_scale: float) -> Tensor:
    """Position-only avoid generator phi(p,o) (the existing OC signed-distance ramp)."""
    return signed_h(x[..., :2], scene, h_scale)


def h_star_value(x: Tensor, scene: Any, c: float, h_scale: float) -> Tensor:
    """h_star = phi(p,o) + c (v^T Re)."""
    return phi_value(x, scene, h_scale) + c * approach_speed(x)


def make_barrier_fn(c: float, h_scale: float, *, position_only: bool = False) -> Callable[[Tensor, Any], Tensor]:
    """Return h_fn(x, scene). position_only=True -> phi(p,o) (theory note baseline, Thm 5.2);
    else h_star (Thm 5.3). Suitable as the h_fn argument to _cbf_terms."""
    if position_only:
        return lambda x, scene: phi_value(x, scene, h_scale)
    return lambda x, scene: h_star_value(x, scene, c, h_scale)


def make_exact_value_fn(
    system: Any,
    c: float,
    h_scale: float,
    dt: float,
    horizon: int,
    *,
    position_only: bool = False,
    goal: Tensor | None = None,
) -> Callable[[Tensor, Any], Tensor]:
    """Return V_fn(x, scene) = sup_{k=0..horizon} h(x_k^pi), the EXACT numerical PNCBF value under
    the fixed nominal policy pi = system.lqr_action (Prop 4.1). Differentiable in x (rollout + running
    max), so _cbf_terms yields the envelope-theorem grad_x V and hence L_g V. No learning.

    `goal` is the per-batch goal for the nominal policy; if None it is read from `scene.goal`.
    """
    barrier = make_barrier_fn(c, h_scale, position_only=position_only)

    def value_fn(x: Tensor, scene: Any) -> Tensor:
        g = goal
        if g is None:
            g = torch.as_tensor(scene.goal, dtype=x.dtype, device=x.device)
        xk = x
        running = barrier(xk, scene)                       # h(x_0)
        for _ in range(horizon):
            u = system.lqr_action(xk, g)
            xk = rk4_step(system, xk, u, dt)
            running = torch.maximum(running, barrier(xk, scene))
        return running

    return value_fn


# --------------------------------------------------------------------------------------------------
# v2.6.0 Stage 1 — training integration: h_star value target, near-B_0 sampler, epsilon_g (R2) loss.
# --------------------------------------------------------------------------------------------------

def value_target_barrier(system: Any, x: Tensor, scene: Any, config: Any) -> Tensor:
    """The GROUND-TRUTH barrier h whose sup-over-time the value V^{h,pi} regresses. For a quadrotor it is
    h_star = phi(p,o) + c * approach (the version's premise); for DI/unicycle it is signed_h(phi).
    Replaces the bare `signed_h(position, ...)` at the value-labeling sites (OC + JT collection).

    v2.7.2: the approach augmentation is generalized through the SYSTEM INTERFACE (`system.approach_barrier`)
    — no branching on the system name. quadrotor_planar returns v^T Re (bit-identical to the module-level
    `approach_speed`, golden parity q5); quadrotor_3d returns v_xy . r_hat (cylinder closing speed). phi is
    taken on the xy footprint (`position(x)[..., :2]`); for the 2D systems position() is already xy so this
    is a no-op and their labels are unchanged."""
    h_scale = float(config["env"]["h_scale"])
    phi = signed_h(system.position(x)[..., :2], scene, h_scale)
    approach_fn = getattr(system, "approach_barrier", None)
    if approach_fn is not None:
        c = float(config["env"][system.name]["c_gain"])
        return phi + c * approach_fn(x, scene, h_scale)
    return phi


def sample_near_B0_states(system: Any, config: Any, n: int, device, dtype,
                          generator: "torch.Generator | None" = None):
    """Synthetic near-B_0 states for the quadrotor (the M6/P1 sampler): agent at ~phi=0 around a single
    obstacle, velocity in ALL directions (B_0's degenerate states are t*=0 = outward/tangential), goal
    beyond. Returns (x [n,6], scene) with per-state single-obstacle batched tensors. Reused by the
    epsilon_g loss (training) and the M2 gate (measurement)."""
    if getattr(system, "name", None) != "quadrotor_planar":
        raise ValueError("sample_near_B0_states is quadrotor_planar-specific.")
    h_scale = float(config["env"]["h_scale"])
    gdev = generator.device if generator is not None else torch.device("cpu")
    def _r(*shape):
        return torch.rand(*shape, generator=generator, device=gdev)
    r = 0.3 + 0.5 * _r(n)
    psi = (2 * _r(n) - 1) * math.pi
    band = (_r(n) - 0.5) * (0.9 * h_scale)
    dist = r + 0.5 * h_scale + band
    dirx = torch.stack([torch.cos(psi), torch.sin(psi)], dim=1)
    p = dist.unsqueeze(1) * dirx
    vpsi = (2 * _r(n) - 1) * math.pi
    vdir = torch.stack([torch.cos(vpsi), torch.sin(vpsi)], dim=1)
    speed = 0.3 + 1.9 * _r(n)
    v = speed.unsqueeze(1) * vdir
    theta = (2 * _r(n) - 1) * math.pi
    omega = (2 * _r(n) - 1) * 1.0
    x = torch.stack([p[:, 0], p[:, 1], theta, v[:, 0], v[:, 1], omega], dim=1).to(device=device, dtype=dtype)
    K = 12
    C = torch.zeros(n, K, 2, device=device, dtype=dtype)
    R = torch.zeros(n, K, device=device, dtype=dtype)
    A = torch.zeros(n, K, dtype=torch.bool, device=device)
    R[:, 0] = r.to(device=device, dtype=dtype); A[:, 0] = True
    goal = (-2.0 * dirx).to(device=device, dtype=dtype)
    from types import SimpleNamespace
    scene = SimpleNamespace(obstacle_centers=C, obstacle_radii=R, obstacle_active=A, goal=goal)
    return x, scene


def lg_authority_loss(system: Any, value_net: Any, config: Any,
                      generator: "torch.Generator | None" = None):
    """epsilon_g regularizer (R2), system-generic: L = mean_{x near B_0} ReLU(eps_g - ||L_g V_hat||).
    Protects the LEARNED value's first-order authority on B_0, which the theorems guarantee only for the
    exact V (note O3). Returns (loss_raw, diagnostics). Weight is applied by the caller."""
    import math as _math  # noqa: F401
    from src.common.filter_hardnet import _cbf_terms
    from src.common.value_net import make_h_fn
    cfg = config["loss"]["value"]["lg_authority"]
    eps_g = float(cfg["eps_g"]); n = int(cfg["n_samples"])
    param = next(value_net.parameters())
    x, scene = sample_near_B0_states(system, config, n, param.device, param.dtype, generator)
    h_fn = make_h_fn(value_net, system)
    u0 = torch.zeros(n, int(system.action_dim), device=param.device, dtype=param.dtype)
    _, _, lg = _cbf_terms(system, h_fn, x, scene, u0, create_graph=True)
    lg_norm = torch.linalg.norm(lg, dim=1)
    loss_raw = torch.relu(eps_g - lg_norm).mean()
    diag = {
        "lg_min": float(lg_norm.min().detach().cpu()),
        "lg_median": float(lg_norm.median().detach().cpu()),
        "lg_degen_frac": float((lg_norm < eps_g).float().mean().detach().cpu()),
    }
    return loss_raw, diag
