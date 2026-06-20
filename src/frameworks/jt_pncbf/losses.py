from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

import torch
from torch import nn

from src.common.filter_hardnet import HardNetFilter, _base_alpha, _cbf_terms, _hardnet_params
from src.common.observation import top_k_obstacles
from src.common.rk4 import rk4_step
from src.common.signed_h import signed_h
from src.common.spline_max import cubic_spline_max
from src.common.system import System
from src.common.value_net import ValueNetEnsemble, make_h_fn
from src.envs.scene_batch import BatchedScene
from src.frameworks.oc_pncbf.collection import TensorTransitionBatch
from src.frameworks.oc_pncbf.value_target import pncbf_target


Tensor = torch.Tensor


@dataclass(frozen=True)
class ValueLossResult:
    total: Tensor
    reach: Tensor
    targets: Tensor


@dataclass(frozen=True)
class PolicyLossResult:
    total: Tensor
    task: Tensor
    action_norm: Tensor
    smoothness: Tensor
    saturation_excess: Tensor
    pretanh: Tensor
    outside: Tensor
    grad_leak: float
    action_abs_mean: Tensor
    action_abs_max: Tensor
    satfrac_a_phi: Tensor
    l_rate_raw: Tensor
    l_rate_weighted: Tensor
    mean_abs_du: Tensor
    l_u_raw: Tensor
    l_u_weighted: Tensor


def value_loss(
    *,
    system: System,
    value_net: ValueNetEnsemble,
    target_value_net: ValueNetEnsemble,
    batch: TensorTransitionBatch,
    lambda_disc: float,
    target_rhs: float,
    config: Mapping[str, Any],
) -> ValueLossResult:
    obs = system.observation(batch.states, batch.scene)
    targets = value_targets(
        system=system,
        target_value_net=target_value_net,
        batch=batch,
        lambda_disc=lambda_disc,
        target_rhs=target_rhs,
        config=config,
    )
    prediction = value_net(obs)
    reach = torch.mean((prediction - targets.unsqueeze(1)) ** 2)
    total = float(config["loss"]["value"]["lambda_R"]) * reach
    return ValueLossResult(total=total, reach=reach, targets=targets)


def value_targets(
    *,
    system: System,
    target_value_net: ValueNetEnsemble,
    batch: TensorTransitionBatch,
    lambda_disc: float,
    target_rhs: float,
    config: Mapping[str, Any],
) -> Tensor:
    with torch.no_grad():
        tail_obs = system.observation(batch.tail_states, batch.tail_scene)
        bootstrap_tail = target_value_net.target_h(tail_obs)
    targets = pncbf_target(
        batch.h_sequence,
        lambda_disc,
        float(config["env"]["dt"]),
        target_rhs,
        bootstrap_tail,
    ).detach()
    return targets.gather(0, batch.step_indices.unsqueeze(0)).squeeze(0)


@dataclass(frozen=True)
class CBFDerivLossResult:
    loss_raw: Tensor               # unweighted mean ReLU(m + gamma) over kept gate states
    diagnostics: dict[str, float]  # gate_* + lg_norm_gate_* scalars for logging


def cbf_deriv_feasibility_loss(
    *,
    system: System,
    value_net: ValueNetEnsemble,
    scene: BatchedScene,
    config: Mapping[str, Any],
    generator: torch.Generator | None = None,
    collect_diagnostics: bool = True,
) -> CBFDerivLossResult:
    """v2.2.1 box-feasibility CBF-derivative auxiliary term L_feas (value-side; oracle-free).

    Builds a fresh gate set S_gate of near-boundary fast-approach states on the given (real) scene
    layouts, drops doomed states (unavoidable-collision predicate), then restricts to the recoverable
    boundary via an h-band mask (deployed h in [h_band_lo, h_band_hi]), and penalizes states where no
    in-box action satisfies the descent condition:
        m(x) = L_f h + alpha(h) h - u_max * ||L_g h||_1 ,  L_feas = mean ReLU(m + gamma_strict).
    Reducible only by growing ||L_g h|| = ||dh/dv|| in the braking direction. Uses the DEPLOYED
    (mean-ensemble) h via make_h_fn + _cbf_terms; never the analytic 2(p-c).
    """
    cfg = config["loss"]["value"]["cbf_deriv"]
    gamma_strict = float(cfg["gamma_strict"])
    d_lo, d_gate = float(cfg["d_lo"]), float(cfg["d_gate"])
    s_lo, s_hi = float(cfg["s_lo"]), float(cfg["s_hi"])
    lateral_frac = float(cfg["lateral_frac"])
    exclude_doomed = bool(cfg["exclude_doomed"])
    # v2.2.1: recoverable-boundary h-band. Absent keys -> [-inf, +inf] (no band filter; backward compat).
    h_band_lo = float(cfg.get("h_band_lo", float("-inf")))
    h_band_hi = float(cfg.get("h_band_hi", float("inf")))
    di = config["env"]["bounds"]["double_integrator"]
    u_max = float(di["u_max"])
    v_max = float(di["v_max"])
    delta_feas = float(config["scene_train"]["init_feasibility_margin"])

    centers = scene.obstacle_centers          # [B, nmax, 2]
    radii = scene.obstacle_radii              # [B, nmax]
    active = scene.obstacle_active            # [B, nmax] bool
    B = centers.shape[0]
    device, dtype = centers.device, centers.dtype
    eps = 1.0e-9

    # placement obstacle: one random ACTIVE obstacle per scene (n_min=1 guarantees >=1 active)
    weights = active.to(dtype)
    weights = torch.where(weights.sum(dim=1, keepdim=True) > 0, weights, torch.ones_like(weights))
    sel = torch.multinomial(weights, 1, generator=generator).squeeze(1)        # [B]
    bidx = torch.arange(B, device=device)
    c0 = centers[bidx, sel]                                                     # [B,2]
    r0 = radii[bidx, sel]                                                       # [B]

    g = torch.randn(B, 2, generator=generator, device=device, dtype=dtype)
    u_rad = g / torch.linalg.norm(g, dim=1, keepdim=True).clamp_min(eps)        # outward unit dir
    d = d_lo + (d_gate - d_lo) * torch.rand(B, generator=generator, device=device, dtype=dtype)
    p = c0 + (r0 + d).unsqueeze(1) * u_rad                                      # [B,2]
    s = s_lo + (s_hi - s_lo) * torch.rand(B, generator=generator, device=device, dtype=dtype)
    v_in = -s.unsqueeze(1) * u_rad
    perp = torch.stack([-u_rad[:, 1], u_rad[:, 0]], dim=1)
    lat = (lateral_frac * s) * (2.0 * torch.rand(B, generator=generator, device=device, dtype=dtype) - 1.0)
    v = v_in + lat.unsqueeze(1) * perp
    vnorm = torch.linalg.norm(v, dim=1, keepdim=True)
    v = torch.where(vnorm > v_max, v * (v_max / vnorm.clamp_min(eps)), v)       # env velocity clamp
    x = torch.cat([p, v], dim=1)                                               # [B,4]

    # realized nearest active obstacle (env helper) for the doomed predicate
    top_rel, top_radii = top_k_obstacles(p, centers, radii, active, 1)
    cp = top_rel[:, 0, :]                                                       # c - p
    rr = top_radii[:, 0]
    dist = torch.linalg.norm(cp, dim=1)
    surf_clr = dist - rr
    v_close = torch.relu(torch.sum(v * cp, dim=1) / dist.clamp_min(eps))
    d_stop = v_close * v_close / (2.0 * u_max)
    doomed = surf_clr < (d_stop + delta_feas)                                   # unavoidable-collision
    keep_recoverable = ~doomed if exclude_doomed else torch.ones_like(doomed)

    # descent-condition margin on the DEPLOYED h; create_graph so gradients reach V_S params.
    # h is computed ONCE here and reused both for m(x) and for the h-band mask (no second forward).
    params = _hardnet_params(config)
    h_fn = make_h_fn(value_net, system)
    u_zero = torch.zeros(B, system.action_dim, device=device, dtype=dtype)
    h, lf_h, lg_h = _cbf_terms(system, h_fn, x, scene, u_zero, create_graph=True)
    alpha = _base_alpha(h, params)
    m = lf_h + alpha * h - u_max * torch.sum(torch.abs(lg_h), dim=1)            # u_max*||L_g h||_1
    viol = torch.relu(m + gamma_strict)

    # v2.2.1 h-band: ADDITIONAL filter (after doomed-exclusion) restricting S_gate to the
    # recoverable boundary, deployed h in [h_band_lo, h_band_hi]. Same deployed h as in m(x).
    h_det = h.detach()
    in_band = (h_det >= h_band_lo) & (h_det <= h_band_hi)
    keep = keep_recoverable & in_band                                          # final S_gate mask
    keepf = keep.to(dtype)
    # Masked mean with a >=1 denominator: numerically identical to sum(viol*keepf)/keepf.sum() when any
    # state is kept (clamp is the identity since keepf.sum() is an integer count >= 1), and 0 when none
    # are kept -- so the loss has NO per-step host sync (no int(keep.sum().item()) control-flow branch).
    loss_raw = torch.sum(viol * keepf) / keepf.sum().clamp_min(1.0)

    if collect_diagnostics:
        # pure-logging scalars (host syncs) -- computed only on metrics-logging-cadence steps.
        n_kept = int(keep.sum().item())
        if n_kept > 0:
            active_frac = float((((m + gamma_strict) > 0) & keep).sum().item()) / n_kept
            lg_sel = torch.linalg.norm(lg_h, dim=1)[keep].detach()
            lg_mean = float(lg_sel.mean().item())
            lg_p50 = float(lg_sel.median().item())
        else:
            active_frac, lg_mean, lg_p50 = 0.0, 0.0, 0.0
        diagnostics = {
            "gate_n_constructed": float(B),
            "gate_n_kept": float(n_kept),                                      # AFTER doomed AND h-band
            "gate_doomed_frac": (float(doomed.sum().item()) / B) if B > 0 else 0.0,  # true doomed frac
            "gate_active_frac": float(active_frac),
            "lg_norm_gate_mean": lg_mean,
            "lg_norm_gate_p50": lg_p50,
        }
    else:
        diagnostics = {}
    return CBFDerivLossResult(loss_raw=loss_raw, diagnostics=diagnostics)


@dataclass(frozen=True)
class SobolevLossResult:
    loss_raw: Tensor               # unweighted mean ||grad_x V_S - grad_x V_spline||^2 over the batch
    diagnostics: dict[str, float]  # sob_* scalars for logging


def _overspeed_gate(
    system: System, x: Tensor, scene: BatchedScene, gate_cfg: Mapping[str, Any], eps: float = 1.0e-9,
) -> tuple[Tensor, Tensor, Tensor]:
    """Smooth over-speed gate g(x) in [0,1], higher the more over-speed (DI). Computed by the caller
    under no_grad (a weighting, not part of the differentiated objective).

      g(x) = sigmoid((inward_speed - s0)/tau_s) * sigmoid((c0 - clearance)/tau_c)

    inward_speed = max(0, v . n_hat), n_hat the unit vector from the agent to the NEAREST active obstacle
    center; clearance = nearest-obstacle surface distance. Nearest-obstacle geometry mirrors signed_h
    (rel = center - p, distance = ||rel||, clearance = distance - radius, inactive -> +inf). Fires only
    when BOTH fast-approaching AND near an obstacle; receding/slow/far -> g ~ 0 (term inactive there)."""
    s0 = float(gate_cfg.get("s0", 1.0)); tau_s = float(gate_cfg.get("tau_s", 0.5))
    c0 = float(gate_cfg.get("c0", 0.3)); tau_c = float(gate_cfg.get("tau_c", 0.15))
    p = system.position(x)                                              # [B,2]
    v = x[:, 2:4]                                                       # velocity [B,2] (double integrator)
    centers = torch.as_tensor(scene.obstacle_centers, dtype=x.dtype, device=x.device)
    radii = torch.as_tensor(scene.obstacle_radii, dtype=x.dtype, device=x.device)
    active = torch.as_tensor(scene.obstacle_active, dtype=torch.bool, device=x.device)
    rel = centers - p.unsqueeze(-2)                                    # agent -> center [B,K,2]
    distance = torch.linalg.norm(rel, dim=-1)                          # [B,K]
    clearance_all = torch.where(active, distance - radii, torch.full_like(distance, float("inf")))
    nearest = torch.argmin(clearance_all, dim=-1)                      # [B]
    bidx = torch.arange(x.shape[0], device=x.device)
    clearance = clearance_all[bidx, nearest]                          # [B]
    rel_n = rel[bidx, nearest]                                         # [B,2] toward nearest center
    n_hat = rel_n / torch.linalg.norm(rel_n, dim=-1, keepdim=True).clamp_min(eps)
    inward = torch.clamp(torch.sum(v * n_hat, dim=-1), min=0.0)        # [B] inward approach speed
    g = torch.sigmoid((inward - s0) / tau_s) * torch.sigmoid((c0 - clearance) / tau_c)
    return g, inward, clearance


def spline_sobolev_loss(
    *,
    system: System,
    value_net: ValueNetEnsemble,
    policy_net: nn.Module | None,
    states: Tensor,
    scene: BatchedScene,
    config: Mapping[str, Any],
    collect_diagnostics: bool = True,
) -> SobolevLossResult:
    """Route D: RPCBF cubic-spline-max Sobolev gradient-matching on V_S (value-side, oracle-free).

    Label: roll the (unfiltered) LQR / detached policy H steps from x0, take the cubic-spline
    max-over-time of the ground-truth signed_h sequence (recovers the continuous-time dV/dv that a
    discrete max staircases away), and read its STATE Jacobian grad_x V_spline by autograd (1st-order,
    detached -- a fixed regression target). Prediction: grad_x of the DEPLOYED (mean-ensemble) h via the
    same make_h_fn path the HardNet filter consumes, with create_graph so the term backprops into V_S
    params (clones the L_feas second-order pattern). L_sobolev = mean ||grad_pred - grad_label||^2 over
    the full state vector (the velocity pair feeds L_g h, the position pair feeds L_f h).
    """
    sob_cfg = config["loss"]["value"]["sobolev"]
    horizon = int(sob_cfg["H"])
    rollout_policy = str(sob_cfg.get("rollout_policy", "lqr"))
    dt = float(config["env"]["dt"])
    h_scale = float(config["env"]["h_scale"])

    x0 = system.wrap_state(states.detach()).detach().requires_grad_(True)
    with torch.enable_grad():
        x = x0
        h_seq = [signed_h(system.position(x), scene, h_scale)]
        for _ in range(horizon):
            if rollout_policy == "lqr":
                u = system.lqr_action(x, _scene_goal(scene, x))
            elif rollout_policy == "policy_detached":
                if policy_net is None:
                    raise ValueError("sobolev rollout_policy='policy_detached' requires a policy_net.")
                u = policy_net(system.observation(x, scene)).detach()
            else:
                raise ValueError(f"Unknown sobolev rollout_policy {rollout_policy!r}.")
            x = rk4_step(system, x, u, dt)
            h_seq.append(signed_h(system.position(x), scene, h_scale))
        h_stack = torch.stack(h_seq, dim=1)                                  # [B, H+1]
        t_grid = torch.linspace(0.0, 1.0, horizon + 1, device=x0.device, dtype=x0.dtype)
        v_spline, _ = cubic_spline_max(t_grid, h_stack)                      # [B]
        grad_label = torch.autograd.grad(v_spline.sum(), x0)[0].detach()     # [B, state_dim], fixed target

        h_fn = make_h_fn(value_net, system)
        x_pred = x0.detach().requires_grad_(True)
        h_pred = h_fn(x_pred, scene).reshape(-1)
        grad_pred = torch.autograd.grad(h_pred.sum(), x_pred, create_graph=True)[0]

    # gated, velocity-only variant (gate.enabled=false -> original global full-vector Sobolev, bit-identical)
    gate_cfg = sob_cfg.get("gate", {}) or {}
    gate_enabled = bool(gate_cfg.get("enabled", False))
    velocity_only = bool(gate_cfg.get("velocity_only", False))
    if gate_enabled and velocity_only:
        diff = grad_pred[:, 2:] - grad_label[:, 2:]                       # velocity channels only (DI: vx,vy)
    else:
        diff = grad_pred - grad_label                                    # full state vector (plain Sobolev)
    per_state = torch.sum(diff * diff, dim=1)                             # [B]
    if gate_enabled:
        with torch.no_grad():
            g, gate_inward, gate_clear = _overspeed_gate(system, x0.detach(), scene, gate_cfg)
        loss_raw = torch.mean(g * per_state)                             # localized: only over-speed states
    else:
        g = None
        loss_raw = torch.mean(per_state)

    if collect_diagnostics:
        gl_v = torch.linalg.norm(grad_label[:, 2:], dim=1)
        gp_v = torch.linalg.norm(grad_pred[:, 2:].detach(), dim=1)
        if gate_enabled:
            gate_mean = float(g.mean().item())
            gate_frac_active = float((g > 0.5).to(g.dtype).mean().item())
            denom = float(g.sum().clamp_min(1.0e-9).item())
            gate_dhdv = float((g * gp_v).sum().item() / denom)           # dh/dv_pred weighted by the gate
        else:
            gate_mean = 0.0; gate_frac_active = 0.0; gate_dhdv = float(gp_v.mean().item())
        diagnostics = {
            "sob_grad_label_v_frac": float((gl_v > 1.0e-6).to(gl_v.dtype).mean().item()),
            "sob_grad_label_v_mean": float(gl_v.mean().item()),
            "sob_dhdv_pred_mean": float(gp_v.mean().item()),
            "sob_v_spline_mean": float(v_spline.detach().mean().item()),
            "sob_gate_mean": gate_mean,
            "sob_gate_frac_active": gate_frac_active,
            "sob_gate_dhdv_pred": gate_dhdv,
        }
    else:
        diagnostics = {}
    return SobolevLossResult(loss_raw=loss_raw, diagnostics=diagnostics)


def policy_bptt_loss(
    *,
    system: System,
    policy_net: nn.Module,
    value_net: ValueNetEnsemble,
    batch: TensorTransitionBatch,
    config: Mapping[str, Any],
    step: int = 0,
) -> PolicyLossResult:
    policy_cfg = config["loss"]["policy"]
    dt = float(config["env"]["dt"])
    bptt_t = int(config["training"]["jt"]["bptt_T"])
    gamma_t = float(policy_cfg["gamma_T"])
    lambda_v = float(policy_cfg["lambda_v"])
    mu_u = float(policy_cfg["mu_u"])
    tau_gate = float(policy_cfg["tau_gate"])
    hardnet = HardNetFilter(system, make_h_fn(value_net, system), config)

    _zero_grads(value_net.parameters())
    with frozen_params(value_net):
        x = system.wrap_state(batch.states.detach())
        scene = batch.scene
        v_now = value_net.deployed_h(system.observation(x, scene)).detach()
        gate_in = torch.sigmoid(-v_now / tau_gate)
        gate_out = torch.sigmoid(v_now / tau_gate)
        task_cost = x.new_zeros(x.shape[0])
        discount = 1.0
        nominal_actions: list[Tensor] = []
        safe_actions: list[Tensor] = []
        pretanh_values: list[Tensor] = []
        sat_excess_values: list[Tensor] = []
        pretanh_penalties: list[Tensor] = []

        for _ in range(bptt_t):
            obs = system.observation(x, scene)
            u_nom = policy_net(obs)
            nominal_actions.append(u_nom)
            if getattr(policy_net, "last_pretanh", None) is not None:
                z = policy_net.last_pretanh
                pretanh_values.append(z)
                excess_z = (z.abs() - float(policy_cfg["z_target"])).clamp_min(0.0)
                sq_excess = (excess_z * excess_z).mean(dim=1)
                if bool(policy_cfg["vs_gated_pretanh"]):
                    v_gate = value_net.deployed_h(system.observation(x, scene)).detach()
                    gate = torch.sigmoid((-v_gate - 0.02) / float(policy_cfg["vs_gate_tau"]))
                    sq_excess = sq_excess * gate
                pretanh_penalties.append(sq_excess.mean())

            sat_excess = (
                u_nom.abs() - float(policy_cfg["sat_excess_threshold"])
            ).clamp_min(0.0)
            sat_excess_values.append((sat_excess * sat_excess).sum(dim=1).mean())

            u_safe, _ = hardnet(x, scene, u_nom)
            safe_actions.append(u_safe)
            x = rk4_step(system, x, u_safe, dt)
            goal = _scene_goal(scene, x)
            pos_error = system.position(x) - goal
            d2 = torch.sum(pos_error * pos_error, dim=1)
            v2 = system.speed(x) * system.speed(x)
            u2 = torch.sum(u_safe * u_safe, dim=1)
            task_cost = task_cost + discount * (d2 + lambda_v * v2 + mu_u * u2)
            discount *= gamma_t

        action_stack = torch.stack(nominal_actions, dim=0)
        safe_stack = torch.stack(safe_actions, dim=0)
        # input-rate regulation (v2.2.2 add-on): DEAD-ZONE penalty on step-to-step change of the
        # APPLIED (post-HardNet) control u_safe. L_rate = mean_t max(0, ||u_t - u_{t-1}|| - delta)^2
        # (delta dead-zone leaves normal control free, only chattering above delta is penalized).
        # Effective weight follows a linear curriculum: w_eff = weight * min(1, step / ramp_steps).
        # mean_abs_du = mean ||du||_2 stays the RAW chattering diagnostic (NOT dead-zoned), comparable
        # across runs/weights.
        rate_cfg = policy_cfg.get("input_rate", {})
        rate_enabled = bool(rate_cfg.get("enabled", False))
        rate_weight = float(rate_cfg.get("weight", 0.0))
        rate_delta = float(rate_cfg.get("delta", 0.0))
        rate_ramp_steps = float(rate_cfg.get("ramp_steps", 0.0))
        rate_w_eff = rate_weight
        if rate_ramp_steps > 0.0:
            rate_w_eff = rate_weight * min(1.0, float(step) / rate_ramp_steps)
        if safe_stack.shape[0] > 1:
            du = safe_stack[1:] - safe_stack[:-1]                       # [T-1, B, A]
            du_norm = torch.linalg.norm(du, dim=2)                      # [T-1, B] = ||du||_2 per step
            excess = (du_norm - rate_delta).clamp_min(0.0)              # dead-zone: free below delta
            l_rate = torch.mean(excess * excess)
            mean_abs_du = du_norm.mean()
        else:
            l_rate = safe_stack.new_zeros(())
            mean_abs_du = safe_stack.new_zeros(())
        # input control-magnitude regularization (v2.2.2 add-on): LQR-style R term on the APPLIED
        # control, L_u = mean_t ||u_safe_t||^2 (ungated, undiscounted). u is the control whose integral
        # is speed, so penalizing |u| curbs accumulated approach speed (the over-speed-entry cause).
        ureg_cfg = policy_cfg.get("u_reg", {})
        ureg_enabled = bool(ureg_cfg.get("enabled", False))
        ureg_weight = float(ureg_cfg.get("weight", 0.0))
        l_u = torch.mean(torch.sum(safe_stack * safe_stack, dim=2))
        action_norm = torch.mean(torch.sum(action_stack * action_stack, dim=2))
        if action_stack.shape[0] > 1:
            smoothness = torch.mean(
                torch.sum((action_stack[1:] - action_stack[:-1]) ** 2, dim=2)
            )
        else:
            smoothness = action_stack.new_zeros(())
        saturation_excess = torch.stack(sat_excess_values).mean()
        pretanh = (
            torch.stack(pretanh_penalties).mean()
            if pretanh_penalties
            else action_stack.new_zeros(())
        )
        obs0 = system.observation(batch.states.detach(), batch.scene)
        u0 = policy_net(obs0)
        x_next_unfiltered = rk4_step(system, batch.states.detach(), u0, dt)
        v_next = value_net.deployed_h(system.observation(x_next_unfiltered, batch.scene))
        outside = (gate_out * v_next).mean()
        task = (gate_in * task_cost).mean()
        total = (
            task
            + float(policy_cfg["lambda_a"]) * action_norm
            + float(policy_cfg["lambda_s"]) * smoothness
            + float(policy_cfg["lambda_sat"]) * saturation_excess
            + float(policy_cfg["lambda_pretanh"]) * pretanh
            + float(policy_cfg["w_outside"]) * outside
        )
        # flag-off (or w_eff==0, e.g. step 0 of the ramp) leaves `total` byte-identical to baseline.
        if rate_enabled and rate_w_eff > 0.0:
            total = total + rate_w_eff * l_rate
            l_rate_weighted = (rate_w_eff * l_rate).detach()
        else:
            l_rate_weighted = l_rate.new_zeros(())
        # u-reg add (flag-off / weight 0 leaves `total` byte-identical to baseline).
        if ureg_enabled and ureg_weight > 0.0:
            total = total + ureg_weight * l_u
            l_u_weighted = (ureg_weight * l_u).detach()
        else:
            l_u_weighted = l_u.new_zeros(())

    return PolicyLossResult(
        total=total,
        task=task.detach(),
        action_norm=action_norm.detach(),
        smoothness=smoothness.detach(),
        saturation_excess=saturation_excess.detach(),
        pretanh=pretanh.detach(),
        outside=outside.detach(),
        grad_leak=grad_norm(value_net.parameters()),
        action_abs_mean=safe_stack.abs().mean().detach(),
        action_abs_max=safe_stack.abs().max().detach(),
        satfrac_a_phi=_satfrac(action_stack, system).detach(),
        l_rate_raw=l_rate.detach(),
        l_rate_weighted=l_rate_weighted,
        mean_abs_du=mean_abs_du.detach(),
        l_u_raw=l_u.detach(),
        l_u_weighted=l_u_weighted,
    )


@contextmanager
def frozen_params(module: nn.Module) -> Iterator[None]:
    states = [param.requires_grad for param in module.parameters()]
    for param in module.parameters():
        param.requires_grad_(False)
    try:
        yield
    finally:
        for param, state in zip(module.parameters(), states, strict=True):
            param.requires_grad_(state)


def grad_sq_norm(parameters: Any) -> Tensor | None:
    """Sum of squared grads as an on-device scalar tensor (NO host sync). None if no params have grads.

    Lets callers compare the gradient-leak against a threshold on-device every step and materialize the
    Python float only when logging or when the halt condition triggers.
    """
    total: Tensor | None = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        s = torch.sum(grad * grad)
        total = s if total is None else total + s
    return total


def grad_norm(parameters: Any) -> float:
    # Single host transfer (one .item() on the device-summed norm) instead of one per parameter.
    sq = grad_sq_norm(parameters)
    return 0.0 if sq is None else float(sq.sqrt().item())


def _zero_grads(parameters: Any) -> None:
    for parameter in parameters:
        parameter.grad = None


def _scene_goal(scene: BatchedScene, x: Tensor) -> Tensor:
    goal = scene.goal.to(device=x.device, dtype=x.dtype)
    if goal.ndim == 1:
        return goal.unsqueeze(0).expand(x.shape[0], -1)
    return goal


def _satfrac(actions: Tensor, system: System) -> Tensor:
    bounds = system.u_bounds.to(device=actions.device, dtype=actions.dtype)
    lower_dist = torch.abs(actions - bounds[:, 0])
    upper_dist = torch.abs(actions - bounds[:, 1])
    saturated = torch.any(torch.minimum(lower_dist, upper_dist) <= 1.0e-3, dim=-1)
    return saturated.to(dtype=actions.dtype).mean()
