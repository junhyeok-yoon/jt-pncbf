"""Analytic maneuver-family barrier V_M (MC-PNCBF Stage A).

Framework-agnostic, GPU-batched, differentiable (NO torch.no_grad in the value path — the HardNet
filter differentiates V_M via autograd). V_M(x) = min over a shift-closed maneuver library of the
finite-horizon worst-case signed-h along an OPEN-LOOP analytic rollout, integrated with the exact
deployment scheme (rk4_step + the velocity clamp inside system.wrap_state) and the canonical
signed_h. No neural network, no bootstrap. Eval-only barrier: it consumes the FULL scene geometry
via signed_h (no Top-K truncation), so the 01_env observation convention does not bind here.

Library:
- m0 deadband brake (componentwise): u_i = -u_max*sign(v_i) if |v_i| > u_max*dt else -v_i/dt.
- m_{j,d} lateral: u = u_max*d for j steps (d = +/- unit-perp to (goal - p), fixed at plan time,
  open-loop; world-axis fallback if ||goal - p|| < 1e-6), then m0.
Horizon per maneuver T_m = j + T_stop, T_stop = ceil(v_max / (u_max*dt)); rolled to T_max = max_j +
T_stop (after a maneuver reaches rest the deadband brake holds it, so max-h is unchanged by the
extra steps). All controls are box-feasible by construction.
"""
from __future__ import annotations

import math
import os
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import torch

from src.common.rk4 import rk4_step
from src.common.signed_h import signed_h
from src.common.system import System

Tensor = torch.Tensor


def t_stop(config: Mapping[str, Any], system: System) -> int:
    dt = float(config["env"]["dt"])
    u_max = float(config["env"]["bounds"][system.name]["u_max"])
    v_max = float(config["env"]["bounds"][system.name]["v_max"])
    return int(math.ceil(v_max / (u_max * dt)))


def build_library(lateral_js: Sequence[int]) -> list[tuple[bool, int, float]]:
    """Return (is_lateral, j, sign) per maneuver: brake first, then +/- lateral for each j."""
    lib: list[tuple[bool, int, float]] = [(False, 0, 0.0)]
    for j in lateral_js:
        lib.append((True, int(j), +1.0))
        lib.append((True, int(j), -1.0))
    return lib


def _deadband_brake(v: Tensor, u_max: float, dt: float) -> Tensor:
    full = -u_max * torch.sign(v)
    stop = -v / dt
    return torch.where(v.abs() > u_max * dt, full, stop)


def _perp_directions(states: Tensor, scene: Any, system: System) -> Tensor:
    """Plan-time-fixed (detached, open-loop) unit-perp to (goal - p); world-axis fallback at goal."""
    b = states.shape[0]
    goal = torch.as_tensor(scene.goal, dtype=states.dtype, device=states.device)
    if goal.ndim == 1:
        goal = goal.unsqueeze(0).expand(b, -1)
    to_goal = goal - system.position(states)
    n = torch.linalg.norm(to_goal, dim=1, keepdim=True)
    unit = to_goal / n.clamp_min(1.0e-12)
    perp = torch.stack([-unit[:, 1], unit[:, 0]], dim=1)
    fallback = torch.zeros_like(perp)
    fallback[:, 0] = 1.0
    perp = torch.where(n > 1.0e-6, perp, fallback)
    return perp.detach()


def _expanded_scene(scene: Any, m: int, dtype: torch.dtype, device: torch.device) -> SimpleNamespace:
    c = torch.as_tensor(scene.obstacle_centers, dtype=dtype, device=device)
    r = torch.as_tensor(scene.obstacle_radii, dtype=dtype, device=device)
    a = torch.as_tensor(scene.obstacle_active, dtype=torch.bool, device=device)
    return SimpleNamespace(
        obstacle_centers=c.repeat_interleave(m, dim=0),
        obstacle_radii=r.repeat_interleave(m, dim=0),
        obstacle_active=a.repeat_interleave(m, dim=0),
    )


# v2.5.0 fast path: the reference rollout below is a 33-step python loop invoked per BPTT step (~4846
# CUDA kernel launches/call), launch-overhead-bound (GPU ~51% idle). `_rollout_impl` inlines EXACTLY the
# same float32 ops (signed_h + RK4 + DI velocity clamp + deadband brake) as a single pure-tensor function
# so torch.compile can fuse it (measured ~2x) and mode='reduce-overhead' can CUDA-graph the no-grad path.
# Reference path is preserved (fast=False) for parity + audit. NO algorithmic change.
_FAST_DEFAULT = os.environ.get("VM_FAST", "1") == "1"   # PRODUCTION default = compiled fast path (Researcher
# Option A; gate (c) retired). Barrier is bit-parity (V_M<=1e-6, L_g<=1e-5); compile fp-reorder gives a
# ~0.035 systematic aggregate-cps offset handled BY DESIGN (all verdict V_M comparisons are fast-vs-fast).
# Reference (bit-parity, uncompiled) retained for tests/audit via VM_FAST=0 / fast=False.
_COMPILE_MODE = os.environ.get("VM_COMPILE_MODE", "default")   # 'default'|'reduce-overhead'|'none' — default
# (compile-only) is used everywhere: it meets the throughput gate (~204 ms/step) and avoids CUDA-graph
# buffer-aliasing risk. The pipeline runs V_M through _cbf_terms(enable_grad) so the grad path dominates
# regardless; reduce-overhead is opt-in via env for the rare pure-no_grad calls only.
_compiled_cache: dict[str, Any] = {}
# The training BPTT differentiates the compiled V_M forward with create_graph=True (for L_g). torch.compile's
# donated-buffer optimization requires create_graph=False on the compiled backward -> disable it so the
# grad path is compatible with the filter's create_graph=True L_g computation (no numeric change).
try:
    import torch._functorch.config as _functorch_config
    _functorch_config.donated_buffer = False
except Exception:   # pragma: no cover - config path may differ across torch versions
    pass


def set_fast_path(enabled: bool) -> None:
    global _FAST_DEFAULT
    _FAST_DEFAULT = bool(enabled)


def _rollout_impl(x, is_lat, jrow, sgn, d_row, centers, radii, active, dt, u_max, v_max, h_scale, T_max):
    """Pure-tensor T_max-step open-loop rollout worst-h. Bit-parity with the reference (same ops/order):
    signed_h (rel/dist/clearance/ramp/inactive->-1/max/clamp), RK4 (DI dynamics cat([v,u])), DI wrap_state
    (velocity clamp to v_max), deadband brake. centers/radii/active are the M-expanded obstacle tensors."""
    def _sh(p):
        rel = p.unsqueeze(-2) - centers
        distance = torch.linalg.norm(rel, dim=-1)
        clearance = distance - radii
        h_all = 1.0 - 2.0 * torch.clamp(clearance / h_scale, min=0.0, max=1.0)
        h_all = torch.where(active, h_all, torch.full_like(h_all, -1.0))
        return torch.clamp(torch.max(h_all, dim=-1).values, min=-1.0, max=1.0)

    def _rk4(xx, u):
        hdt = 0.5 * dt
        k1 = torch.cat([xx[:, 2:4], u], dim=1)
        k2 = torch.cat([(xx + hdt * k1)[:, 2:4], u], dim=1)
        k3 = torch.cat([(xx + hdt * k2)[:, 2:4], u], dim=1)
        k4 = torch.cat([(xx + dt * k3)[:, 2:4], u], dim=1)
        xn = xx + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        speed = torch.linalg.norm(xn[:, 2:4], dim=1).clamp_min(1.0e-6)
        scale = torch.clamp(v_max / speed, max=1.0)
        return torch.cat([xn[:, :2], xn[:, 2:4] * scale.unsqueeze(1)], dim=1)

    maxh = _sh(x[:, :2])
    for k in range(T_max):
        v = x[:, 2:4]
        u_brake = torch.where(v.abs() > u_max * dt, -u_max * torch.sign(v), -v / dt)
        u = torch.where((is_lat & (k < jrow)).unsqueeze(1), u_max * d_row, u_brake)
        x = _rk4(x, u)
        maxh = torch.maximum(maxh, _sh(x[:, :2]))
    return maxh


def _get_rollout(mode: str):
    if mode not in _compiled_cache:
        _compiled_cache[mode] = _rollout_impl if mode == "none" else torch.compile(_rollout_impl, mode=mode)
    return _compiled_cache[mode]


def maneuver_maxh(
    states: Tensor,
    scene: Any,
    system: System,
    config: Mapping[str, Any],
    *,
    lateral_js: Sequence[int] = (),
    dt_override: float | None = None,
    fast: bool | None = None,
) -> Tensor:
    """Per-maneuver worst-case signed-h over the open-loop rollout. Returns [B, M].

    dt_override sets the certificate's internal rollout grid dt_Vm (deploy-rate eval); lateral_js are
    then step-counts at dt_override. Default (None) uses config env.dt (Stage A/B path, unchanged).
    fast=None uses the module default (compiled); fast=False forces the reference python loop."""
    dt = float(dt_override) if dt_override is not None else float(config["env"]["dt"])
    h_scale = float(config["env"]["h_scale"])
    u_max = float(config["env"]["bounds"][system.name]["u_max"])
    v_max = float(config["env"]["bounds"][system.name]["v_max"])
    tstop = int(math.ceil(v_max / (u_max * dt)))                          # T_stop at the effective dt
    lib = build_library(lateral_js)
    m = len(lib)
    b = states.shape[0]
    dev, dtype = states.device, states.dtype
    T_max = max((j for _, j, _ in lib), default=0) + tstop

    perp = _perp_directions(states, scene, system)                       # [B,2], detached
    x = states.repeat_interleave(m, dim=0)                               # [B*M, 4]
    scn = _expanded_scene(scene, m, dtype, dev)
    is_lat = torch.tensor([o[0] for o in lib], device=dev).repeat(b)     # [B*M]
    jrow = torch.tensor([o[1] for o in lib], device=dev, dtype=torch.long).repeat(b)
    sgn = torch.tensor([o[2] for o in lib], dtype=dtype, device=dev).repeat(b)
    d_row = perp.repeat_interleave(m, dim=0) * sgn.unsqueeze(1)          # [B*M, 2]

    use_fast = _FAST_DEFAULT if fast is None else fast
    if use_fast and dev.type == "cuda":
        # no-grad -> CUDA-graph-eligible mode; grad path -> plain compile (default) to avoid graph/autograd
        # capture conflicts (the training BPTT filter always runs under enable_grad for L_g).
        mode = _COMPILE_MODE if not torch.is_grad_enabled() else "default"
        maxh = _get_rollout(mode)(x, is_lat, jrow, sgn, d_row, scn.obstacle_centers, scn.obstacle_radii,
                                  scn.obstacle_active, dt, u_max, v_max, h_scale, T_max)
        return maxh.reshape(b, m)

    maxh = signed_h(system.position(x), scn, h_scale)                    # k=0 (reference path)
    for k in range(T_max):
        v = x[:, 2:4]
        u_brake = _deadband_brake(v, u_max, dt)
        u_lat = u_max * d_row
        in_lat = (is_lat & (k < jrow)).unsqueeze(1)
        u = torch.where(in_lat, u_lat, u_brake)
        x = rk4_step(system, x, u, dt)
        maxh = torch.maximum(maxh, signed_h(system.position(x), scn, h_scale))
    return maxh.reshape(b, m)


def maneuver_value(
    states: Tensor,
    scene: Any,
    system: System,
    config: Mapping[str, Any],
    *,
    lateral_js: Sequence[int] = (),
    softmin_beta: float = 0.0,
    dt_override: float | None = None,
    fast: bool | None = None,
) -> Tensor:
    """V_M = min (or softmin, if beta>0) over the library of the per-maneuver worst-case h. [B]."""
    maxh = maneuver_maxh(states, scene, system, config, lateral_js=lateral_js, dt_override=dt_override,
                         fast=fast)
    if softmin_beta > 0.0:
        return -(1.0 / softmin_beta) * torch.logsumexp(-softmin_beta * maxh, dim=1)
    return maxh.min(dim=1).values


def maneuver_policy(
    states: Tensor,
    scene: Any,
    system: System,
    config: Mapping[str, Any],
    *,
    lateral_js: Sequence[int] = (),
) -> Tensor:
    """First control u0 of the argmin maneuver at x (pi_M for the validity check). [B, action_dim]."""
    dt = float(config["env"]["dt"])
    u_max = float(config["env"]["bounds"][system.name]["u_max"])
    lib = build_library(lateral_js)
    with torch.no_grad():
        maxh = maneuver_maxh(states, scene, system, config, lateral_js=lateral_js)
        arg = maxh.argmin(dim=1)                                         # [B]
    perp = _perp_directions(states, scene, system)
    is_lat = torch.tensor([o[0] for o in lib], device=states.device)
    jrow = torch.tensor([o[1] for o in lib], device=states.device, dtype=torch.long)
    sgn = torch.tensor([o[2] for o in lib], dtype=states.dtype, device=states.device)
    v = states[:, 2:4]
    u_brake = _deadband_brake(v, u_max, dt)
    lat = is_lat[arg] & (jrow[arg] > 0)                                  # first step lateral?
    d = perp * sgn[arg].unsqueeze(1)
    return torch.where(lat.unsqueeze(1), u_max * d, u_brake)


def make_maneuver_h_fn(
    system: System,
    config: Mapping[str, Any],
    *,
    lateral_js: Sequence[int] = (),
    softmin_beta: float = 0.0,
    gamma_m: float = 0.0,
    dt_override: float | None = None,
    fast: bool | None = None,
) -> Callable[[Tensor, Any], Tensor]:
    """HardNet h_fn adapter: h_fn(x, scene) -> V_M(x) + gamma_m [B] (differentiable, no no_grad).

    gamma_m is baked in as h_eff = V_M + gamma_m — identical to adding it after _cbf_terms (a constant
    shift leaves grad V_M, hence L_f/L_g, unchanged; only alpha & row_upper see h_eff).
    dt_override sets the certificate rollout grid dt_Vm (deploy-rate eval; lateral_js in dt_Vm steps)."""
    def h_fn(x: Tensor, scene: Any) -> Tensor:
        v = maneuver_value(x, scene, system, config, lateral_js=lateral_js, softmin_beta=softmin_beta,
                           dt_override=dt_override, fast=fast)
        return v + gamma_m if gamma_m != 0.0 else v

    return h_fn


def build_safety_h_fn(system: System, config: Mapping[str, Any], value_net: Any = None):
    """Single safety-channel h_fn builder for BOTH collection filtering and policy BPTT (and eval).

    safety_channel.type == 'maneuver' -> analytic V_M + gamma_m (library J=1..J, both dirs);
    'value' (default) -> the learned make_h_fn(value_net) — value path bit-identical."""
    sc = config.get("safety_channel", {}) or {}
    if str(sc.get("type", "value")) == "maneuver":
        m = sc.get("maneuver", {}) or {}
        j_max = int(m.get("J", 8))
        lateral_js = list(range(1, j_max + 1))   # build_library adds BOTH +/- per j (both_dirs)
        return make_maneuver_h_fn(system, config, lateral_js=lateral_js,
                                  gamma_m=float(m.get("gamma_m", 0.0)))
    from src.common.value_net import make_h_fn
    return make_h_fn(value_net, system)
