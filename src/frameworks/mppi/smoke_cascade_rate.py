"""CPU smoke for the literature-standard (T, omega_des) cascaded MPPI. Runs BEFORE any GPU cell.

MEASUREMENT ONLY. No threshold is registered, no cell is selected and nothing is ranked; every check
reports the number it measured and the config-read quantity it is read against.

  (a) HOVER FIXED POINT under the FULL CASCADE on the REAL PLANT. A level, motionless state is driven
      with the plan's hover entry (T = m*g read off the system object, omega_des = 0) through the inner
      loop and `rk4_step` on the full rotor-thrust plant. Reports the drift in position, speed, body
      rate and tilt.
  (b) STRAIGHT LINE, NO OBSTACLE, reaching the position radius. Two legs on the same constructed scene:
      (b1) the deterministic outer law of check (f) — the shipped cascaded PD expressed in this
      interface — with the planner's rate limit applied, closed on the real plant; (b2) the (T, omega)
      MPPI PLANNER itself at a small N, so the closed loop that the screen runs is exercised end to end.
  (c) THE PER-ROTOR BOX AND THE ASYMMETRY EVIDENCE. On a constructed aggressive command the allocated
      rotor thrusts respect the box AFTER the clip; and the sampler's OWN draws about the hover anchor
      are allocated at real states to COUNT the pre-clip rotor commands that fall outside the box, with
      their denominator. That count is the evidence for asymmetry (2) — the planner has no per-rotor
      variable to bound, so its sampled plans can and do map outside the box, and the box is applied
      only after allocation.
  (d) THE COLLISION INDICATOR. The cost's `collision_mask` against `src.common.outcomes.step_outcomes`
      on a real rollout plus states placed exactly on the cylinder and band boundaries.
  (e) THE SAMPLED-RATE LOW-PASS. The lag-1 autocorrelation of the rate channels along the horizon,
      measured BEFORE and AFTER the filter; the filter is verified by the increase.
  (f) THE INTERFACE REPRODUCES THE SHIPPED CASCADED PD. With the rate setpoint
      omega_des = (kp_att / kd_att) * e_att_body and the inner-loop gain k_rate = kd_att, the cascade's
      torque is J (kp_att e_att_body - kd_att omega) — `System.lqr_action`'s own line. The rotor
      commands are compared against `system.lqr_action(x, goal)` itself, which is non-circular because
      the reference is the shipped call with the shipped gains. This is the statement that the (T,
      omega) interface can express the controller the repo already ships, not merely something near it.

Run:  CUDA_VISIBLE_DEVICES="" python -m src.frameworks.mppi.smoke_cascade_rate
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import torch

from src.common.outcomes import step_outcomes
from src.common.rk4 import rk4_step
from src.common.system import System
from src.envs.quadrotor_3d import _quat_to_R, _rot
from src.envs.scene_batch import batch_scenes
from src.envs.scene_init import Scene
from src.eval.build_pools import load_pool
from src.frameworks.mppi.cascade_rate import RateCascadeController, _W
from src.frameworks.mppi.cost import collision_mask


Tensor = torch.Tensor


def level_state(system: System, position: Tensor, dtype: torch.dtype) -> Tensor:
    """A level, motionless state at `position`: identity quaternion (w,x,y,z) = (1,0,0,0), zero linear
    velocity, zero body rate, built through `system.wrap_state` so it is normalised exactly as the plant
    normalises every state it integrates."""
    batch = position.shape[0]
    quat = torch.zeros(batch, 4, dtype=dtype)
    quat[:, 0] = 1.0
    rest = torch.zeros(batch, 6, dtype=dtype)
    return system.wrap_state(torch.cat([position, quat, rest], dim=1))


def straight_line_scene(system: System, offset: list[float], dtype: torch.dtype) -> Scene:
    """An OBSTACLE-FREE straight-line scene: one obstacle slot, inactive. Built here rather than drawn
    from the pool, so the check is a wiring test and not a second screen."""
    start = np.zeros(3, dtype=np.float64)
    goal = np.asarray(offset, dtype=np.float64)
    return Scene(
        obstacle_centers=np.zeros((1, 2), dtype=np.float64),
        obstacle_radii=np.zeros(1, dtype=np.float64),
        obstacle_active=np.zeros(1, dtype=bool),
        start=start,
        goal=goal,
        system=system.name,
        mode="eval",
        initial_velocity=np.zeros(3, dtype=np.float64),
        initial_attitude_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        initial_omega_vec=np.zeros(3, dtype=np.float64),
    )


def shipped_pd_command(system: System, x: Tensor, goal: Tensor) -> tuple[Tensor, Tensor]:
    """The SHIPPED cascaded PD, `System.lqr_action` (src/envs/quadrotor_3d.py:173-189), expressed in
    THIS interface. Returns (T, omega_des).

        a_des    = -kp_pos (p - goal) - kd_pos v            the shipped outer loop
        f_des    = mass (a_des + gravity e3)
        T        = clamp(f_des . b3, min=0)                 the shipped thrust projection
        b3_des   = f_des / ||f_des||
        e_att    = R^T (b3 x b3_des)                        the shipped attitude error, in body
        omega_des = (kp_att / kd_att) * e_att

    The rate setpoint is the shipped attitude PD's proportional term divided by its rate gain, so a
    rate loop `tau = J kd_att (omega_des - omega)` reproduces `tau = J (kp_att e_att - kd_att omega)`
    EXACTLY. Every gain, the mass and gravity are read off the system object.
    """
    p = system.position(x)
    v = x[:, 7:10]
    a_des = -system.kp_pos * (p - goal) - system.kd_pos * v
    e_up = torch.zeros_like(a_des)
    e_up[:, 2] = system.gravity
    f_des = system.mass * (a_des + e_up)
    R = _quat_to_R(x[:, 3:7])
    b3 = R[:, :, 2]
    thrust = (f_des * b3).sum(dim=1).clamp(min=0.0)
    b3_des = f_des / torch.clamp(torch.linalg.norm(f_des, dim=1, keepdim=True), min=1e-9)
    e_att_body = _rot(R.transpose(1, 2), torch.cross(b3, b3_des, dim=1))
    omega_des = (system.kp_att / system.kd_att) * e_att_body
    return thrust.unsqueeze(-1), omega_des


# =================================================================================================
# the checks
# =================================================================================================
@torch.no_grad()
def check_hover(system: System, controller: RateCascadeController, dt: float, steps: int) -> dict:
    """(a) the hover fixed point under the full cascade on the REAL plant."""
    dtype = controller.dtype
    x = level_state(system, torch.zeros(1, 3, dtype=dtype), dtype)
    x0 = x.clone()
    hover = torch.zeros(1, controller.plan_dim, dtype=dtype)
    hover[0, 0] = controller.t_hover                       # T_hover = m*g, read off the system object
    drift_p, drift_v, drift_w, drift_tilt = [], [], [], []
    for _ in range(steps):
        x = rk4_step(system, x, controller.inner_loop(x, hover), dt)
        drift_p.append(float(torch.linalg.norm(system.position(x) - system.position(x0)).item()))
        drift_v.append(float(system.speed(x).item()))
        drift_w.append(float(system.angular_rate(x).item()))
        cos_tilt = float(system.thrust_axis(x)[0, 2].item())
        drift_tilt.append(math.degrees(math.acos(min(1.0, max(-1.0, cos_tilt)))))
    return {
        "steps": steps,
        "command": "T = m*g (read off the system object), omega_des = 0 — the plan's hover entry",
        "T_hover_N": controller.t_hover,
        "plant": "the FULL rotor-thrust plant, rk4_step at env.dt",
        "max_position_drift_m": max(drift_p),
        "max_speed_m_s": max(drift_v),
        "max_angular_rate_rad_s": max(drift_w),
        "max_tilt_deg": max(drift_tilt),
        "final_position_drift_m": drift_p[-1],
        "rotor_command_N": [float(v) for v in controller.inner_loop(x0, hover)[0].tolist()],
        "hover_trim_per_rotor_N": float(controller.trim_rotor[0].item()),
        "max_abs_difference_from_trim_N": float(
            (controller.inner_loop(x0, hover) - controller.trim_rotor).abs().max().item()
        ),
        "PASS": None,
    }


@torch.no_grad()
def check_straight_line_outer(
    system: System, controller: RateCascadeController, scene: Scene, dt: float, seconds: float,
    goal_radius: float,
) -> dict:
    """(b1) the obstacle-free straight line under the deterministic outer law of check (f)."""
    dtype = controller.dtype
    start = torch.as_tensor(scene.start, dtype=dtype).view(1, 3)
    goal = torch.as_tensor(scene.goal, dtype=dtype).view(1, 3)
    x = level_state(system, start.clone(), dtype)
    n_steps = int(round(seconds / dt))
    dist, limited = [], 0
    for _ in range(n_steps):
        thrust, omega_des = shipped_pd_command(system, x, goal)
        command = controller.rate_limit(torch.cat([thrust, omega_des], dim=-1))
        limited += int((command[:, 1:] != omega_des).sum().item())
        x = rk4_step(system, x, controller.inner_loop(x, command), dt)
        dist.append(float(torch.linalg.norm(system.position(x) - goal).item()))
    return {
        "mode": "deterministic outer law (the shipped cascaded PD expressed as (T, omega_des)), the "
                "planner's rate limit applied, closed on the REAL plant through the inner loop",
        "offset_m": [float(v) for v in (goal - start)[0].tolist()],
        "seconds": seconds, "n_steps": n_steps,
        "goal_radius_read_from_config": goal_radius,
        "closest_approach_m": min(dist),
        "final_distance_m": dist[-1],
        "final_speed_m_s": float(system.speed(x).item()),
        "final_angular_rate_rad_s": float(system.angular_rate(x).item()),
        "rate_limit_engagements": limited,
        "reaches_position_radius": bool(min(dist) <= goal_radius),
    }


@torch.no_grad()
def check_straight_line_planner(
    system: System, framework: Any, scene: Scene, dt: float, seconds: float, goal_radius: float,
) -> dict:
    """(b2) the same scene flown by the (T, omega) MPPI PLANNER itself, at a small N on CPU."""
    controller = framework.controller
    dtype = controller.dtype
    batched = batch_scenes([scene], device=torch.device("cpu"), dtype=dtype)
    x = level_state(system, torch.as_tensor(scene.start, dtype=dtype).view(1, 3), dtype)
    goal = torch.as_tensor(scene.goal, dtype=dtype).view(1, 3)
    n_steps = int(round(seconds / dt))
    controller.reset(1)
    dist = []
    for _ in range(n_steps):
        u = controller.act(x, batched)
        x = rk4_step(system, x, u, dt)
        dist.append(float(torch.linalg.norm(system.position(x) - goal).item()))
    return {
        "mode": "the (T, omega_des) MPPI planner itself, closed on the REAL plant; a wiring test at a "
                "small N, never a second screen",
        "N": int(controller.params.n_samples), "H": int(controller.params.horizon),
        "lam": float(controller.params.lam), "sigma": float(controller.params.sigma),
        "rate_gain_k_rate": controller.k_rate,
        "seconds": seconds, "n_steps": n_steps,
        "goal_radius_read_from_config": goal_radius,
        "closest_approach_m": min(dist),
        "final_distance_m": dist[-1],
        "final_speed_m_s": float(system.speed(x).item()),
        "reaches_position_radius": bool(min(dist) <= goal_radius),
        "preclip_over_this_leg": controller.preclip_record(),
    }


@torch.no_grad()
def check_box(
    system: System, controller: RateCascadeController, n_states: int, n_plans: int, seed: int,
) -> dict:
    """(c) the box after allocation, and the pre-clip out-of-box count with its denominator."""
    dtype = controller.dtype
    generator = torch.Generator().manual_seed(int(seed))
    probe = system.wrap_state(torch.randn(n_states, system.state_dim, generator=generator, dtype=dtype))
    lo = float(controller.u_lo.min().item())
    hi = float(controller.u_hi.max().item())
    omega_max = controller.omega_max

    # ---- (c.i) a CONSTRUCTED AGGRESSIVE command: the largest collective the box can produce and the
    #            rate limit itself, at every sign pattern of the three rate channels.
    signs = torch.tensor(
        [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)], dtype=dtype
    )                                                                        # [8,3]
    collective_max = float((controller.u_hi.sum()).item())                   # the box's own collective
    aggressive = torch.cat(
        [torch.full((signs.shape[0], 1), collective_max, dtype=dtype), signs * omega_max], dim=-1
    )                                                                        # [8,4]
    states = probe.unsqueeze(1).expand(-1, aggressive.shape[0], -1).reshape(-1, system.state_dim)
    commands = aggressive.unsqueeze(0).expand(probe.shape[0], -1, -1).reshape(-1, controller.plan_dim)
    tau = controller.inertia * (controller.k_rate * (commands[:, 1:] - states[..., _W]))
    wrench = torch.cat([commands[:, :1], tau], dim=-1)
    pre = wrench @ controller.mixer_inv.t()
    post = controller.allocate(wrench)
    aggressive_out = ((pre < controller.u_lo) | (pre > controller.u_hi))

    # ---- (c.ii) THE ASYMMETRY EVIDENCE: the SAMPLER'S OWN draws about the hover anchor, rate-limited
    #             exactly as the planner limits them, allocated at the same random states.
    plan_noise = torch.randn(n_plans, controller.plan_dim, generator=generator, dtype=dtype)
    sampled = controller.anchor.unsqueeze(0) + plan_noise * controller.sigma_channel.unsqueeze(0)
    sampled = controller.rate_limit(sampled)
    idx = torch.randint(0, probe.shape[0], (n_plans,), generator=generator)
    at_states = probe[idx]
    tau_s = controller.inertia * (controller.k_rate * (sampled[:, 1:] - at_states[..., _W]))
    pre_s = torch.cat([sampled[:, :1], tau_s], dim=-1) @ controller.mixer_inv.t()
    post_s = controller.allocate(torch.cat([sampled[:, :1], tau_s], dim=-1))
    out_s = ((pre_s < controller.u_lo) | (pre_s > controller.u_hi))

    return {
        "box_read_from_system": [lo, hi],
        "omega_max_read_from_system": omega_max,
        "k_rate": controller.k_rate,
        "aggressive_command": {
            "what": "the box's own maximum collective (sum of the per-rotor upper bounds) with the rate "
                    "command at +/- omega_max in every sign pattern, evaluated at random states",
            "n_states": int(probe.shape[0]), "n_commands": int(aggressive.shape[0]),
            "collective_commanded_N": collective_max,
            "n_entries": int(pre.numel()),
            "preclip_min_N": float(pre.min().item()), "preclip_max_N": float(pre.max().item()),
            "n_preclip_outside_box": int(aggressive_out.sum().item()),
            "frac_preclip_outside_box": float(aggressive_out.double().mean().item()),
            "postclip_min_N": float(post.min().item()), "postclip_max_N": float(post.max().item()),
            "all_postclip_inside_box": bool((post >= lo).all() and (post <= hi).all()),
            "PASS": bool((post >= lo).all() and (post <= hi).all()),
        },
        "sampled_plans": {
            "what": "the ASYMMETRY EVIDENCE. The sampler's own draws about the hover anchor "
                    "(u_hover + eps, per-channel std from the cell's sigma), rate-limited exactly as the "
                    "planner limits them, allocated at random states. The planner has NO per-rotor "
                    "variable to bound, so these plans can and do map to pre-clip rotor commands outside "
                    "the box; the box is applied only after allocation.",
            "n_plans": int(n_plans),
            "n_entries": int(pre_s.numel()),
            "n_preclip_outside_box": int(out_s.sum().item()),
            "frac_preclip_outside_box": float(out_s.double().mean().item()),
            "n_plans_with_any_entry_outside_box": int(out_s.any(dim=-1).sum().item()),
            "frac_plans_with_any_entry_outside_box": float(out_s.any(dim=-1).double().mean().item()),
            "preclip_min_N": float(pre_s.min().item()), "preclip_max_N": float(pre_s.max().item()),
            "postclip_min_N": float(post_s.min().item()), "postclip_max_N": float(post_s.max().item()),
            "all_postclip_inside_box": bool((post_s >= lo).all() and (post_s <= hi).all()),
            "sigma_per_channel": [float(v) for v in controller.sigma_channel.tolist()],
            "asymmetry": controller.asymmetry_statement(),
            "PASS": bool(
                out_s.any() and (post_s >= lo).all() and (post_s <= hi).all()
            ),
        },
    }


@torch.no_grad()
def check_predicate(
    system: System, controller: RateCascadeController, config: Mapping[str, Any], scenes,
    n_rollout_steps: int, seed: int,
) -> dict:
    """(d) the cost's collision indicator against `src.common.outcomes.step_outcomes`, exactly."""
    dtype = controller.dtype
    device = torch.device("cpu")
    batched = batch_scenes(scenes, device=device, dtype=dtype)
    from src.envs.scene_batch import initial_states_from_batch
    x = system.wrap_state(initial_states_from_batch(batched))
    dt = float(config["env"]["dt"])
    generator = torch.Generator(device=device).manual_seed(int(seed))

    # a real rollout of the CASCADE, so the states tested are states this controller actually produces
    states = [x]
    for _ in range(n_rollout_steps):
        command = torch.cat(
            [torch.full((x.shape[0], 1), controller.t_hover, dtype=dtype),
             (2.0 * torch.rand((x.shape[0], 3), generator=generator, dtype=dtype) - 1.0)
             * controller.omega_max],
            dim=-1,
        )
        x = rk4_step(system, x, controller.inner_loop(x, command), dt)
        states.append(x)
    rollout_states = torch.stack(states, dim=0)

    # states placed exactly ON the cylinder and band boundaries, so the strict / non-strict conventions
    # are exercised rather than assumed
    band = float(config["env"]["band_collision_limit"])
    centers = torch.as_tensor(np.stack([s.obstacle_centers for s in scenes]), dtype=dtype)
    radii = torch.as_tensor(np.stack([s.obstacle_radii for s in scenes]), dtype=dtype)
    active = torch.as_tensor(np.stack([s.obstacle_active for s in scenes]), dtype=torch.bool)
    first = active.float().argmax(dim=1)
    rows = torch.arange(len(scenes))
    c0 = centers[rows, first]
    r0 = radii[rows, first]
    template = rollout_states[0].clone()
    probes = []
    for offset in (-0.05, -1e-9, 0.0, 1e-9, 0.05):
        probe = template.clone()
        probe[:, 0] = c0[:, 0] + r0 + offset
        probe[:, 1] = c0[:, 1]
        probes.append(probe)
    for z in (-band - 0.05, -band, -band + 1e-9, 0.0, band - 1e-9, band, band + 0.05):
        probe = template.clone()
        probe[:, 2] = z
        probes.append(probe)
    synthetic = torch.stack(probes, dim=0)

    all_states = torch.cat([rollout_states, synthetic], dim=0)
    reference = step_outcomes(all_states, batched, system, config).collided
    mine = collision_mask(
        system.position(all_states).permute(1, 0, 2),
        batched.obstacle_centers, batched.obstacle_radii, batched.obstacle_active,
        controller.cost,
    ).permute(1, 0)
    agree = (mine == reference)
    return {
        "what": "the cost's collision indicator (cost.collision_mask, the SAME object the rollout "
                "charges C_crash with) against src.common.outcomes.step_outcomes, the harness's own "
                "outcome predicate",
        "n_states_compared": int(agree.numel()),
        "n_rollout_steps": int(rollout_states.shape[0]),
        "n_boundary_probes": int(synthetic.shape[0]),
        "n_scenes": len(scenes),
        "band_collision_limit": band,
        "reference_positive_rate": float(reference.double().mean()),
        "mine_positive_rate": float(mine.double().mean()),
        "agreement_rate": float(agree.double().mean()),
        "n_disagreements": int((~agree).sum()),
        "PASS": bool(agree.all() and reference.any() and (~reference).any()),
    }


@torch.no_grad()
def check_lowpass(controller: RateCascadeController, n_sequences: int, seed: int) -> dict:
    """(e) the sampled-rate low-pass: the lag-1 autocorrelation along the horizon, before and after."""
    dtype = controller.dtype
    horizon = int(controller.params.horizon)
    generator = torch.Generator().manual_seed(int(seed))
    z = torch.randn((n_sequences, horizon, controller.plan_dim), generator=generator, dtype=dtype)
    raw = controller.anchor.view(1, 1, -1) + z * controller.sigma_channel.view(1, 1, -1)
    alpha = float(controller.params.ou_alpha)
    innovation = math.sqrt(max(0.0, 1.0 - alpha * alpha))
    ou = raw.clone()
    for k in range(1, horizon):
        ou[:, k] = innovation * ou[:, k] + alpha * ou[:, k - 1]              # the inherited OU recursion

    beta = controller.lowpass_beta
    filtered = torch.empty_like(ou)
    state = torch.zeros(n_sequences, 3, dtype=dtype)                         # a resting vehicle
    for k in range(horizon):
        state = beta * state + (1.0 - beta) * ou[:, k, 1:]
        filtered[:, k, 0] = ou[:, k, 0]
        filtered[:, k, 1:] = state

    def lag1(seq: Tensor) -> list[float]:
        centred = seq - seq.mean(dim=(0, 1), keepdim=True)
        num = (centred[:, 1:] * centred[:, :-1]).mean(dim=(0, 1))
        den = (centred * centred).mean(dim=(0, 1)).clamp(min=1e-30)
        return [float(v) for v in (num / den).tolist()]

    before = lag1(ou)
    after = lag1(filtered)
    increased = [bool(a > b) for a, b in zip(after[1:], before[1:])]
    return {
        "what": "lag-1 autocorrelation along the HORIZON axis of the sampled sequences, before and after "
                "the rate low-pass. The collective channel is not filtered and is reported as the "
                "control.",
        "n_sequences": int(n_sequences), "horizon": horizon,
        "channel_names": ["T", "w_x", "w_y", "w_z"],
        "beta": beta, "steps": controller.lowpass_steps,
        "ou_alpha": alpha,
        "lag1_before": before,
        "lag1_after": after,
        "rate_channels_autocorrelation_increased": increased,
        "collective_channel_unchanged": bool(
            torch.equal(filtered[:, :, 0], ou[:, :, 0])
        ),
        "PASS": bool(all(increased)),
    }


@torch.no_grad()
def check_planner_fidelity(
    system: System, controller: RateCascadeController, dt: float, n_states: int, seed: int,
) -> dict:
    """(g) the PLANNER'S MODEL against the FULL PLANT under the same command sequence.

    The planner's model assumes the inner loop tracks the commanded rate; the plant has attitude
    dynamics and a per-rotor box. From the same random initial states, one horizon of the same sampled
    (T, omega_des) sequence is rolled BOTH ways — through `_RatePlannerModel` as the planner does, and
    through the inner loop plus the full rotor-thrust plant as the simulation does — and the divergence
    is reported per horizon index. This is the error the delegation makes; it is MEASURED, not assumed,
    and no threshold is taken against it.
    """
    dtype = controller.dtype
    horizon = int(controller.params.horizon)
    generator = torch.Generator().manual_seed(int(seed))
    x0 = system.wrap_state(torch.randn(n_states, system.state_dim, generator=generator, dtype=dtype))
    noise = torch.randn(n_states, horizon, controller.plan_dim, generator=generator, dtype=dtype)
    command = controller.rate_limit(
        controller.anchor.view(1, 1, -1) + noise * controller.sigma_channel.view(1, 1, -1)
    )
    beta = controller.lowpass_beta
    x_plan, x_plant = x0.clone(), x0.clone()
    w_plan, w_plant = x0[..., _W].clone(), x0[..., _W].clone()
    position_err, rate_err = [], []
    for k in range(horizon):
        w_plan = beta * w_plan + (1.0 - beta) * command[:, k, 1:]
        x_plan = torch.cat([x_plan[:, :10], w_plan], dim=1)
        x_plan = rk4_step(controller.planner, x_plan, command[:, k], dt)
        w_plant = beta * w_plant + (1.0 - beta) * command[:, k, 1:]
        u = controller.inner_loop(x_plant, torch.cat([command[:, k, :1], w_plant], dim=-1))
        x_plant = rk4_step(system, x_plant, u, dt)
        position_err.append(torch.linalg.norm(system.position(x_plan) - system.position(x_plant), dim=-1))
        rate_err.append(torch.linalg.norm(x_plan[..., _W] - x_plant[..., _W], dim=-1))

    def band(seq: list[Tensor]) -> dict:
        stacked = torch.stack(seq)                                            # [H, n]
        idx = [0, horizon // 4, horizon // 2, horizon - 1]
        return {
            "at_horizon_index": idx,
            "p50": [float(stacked[i].median().item()) for i in idx],
            "p95": [float(stacked[i].quantile(0.95).item()) for i in idx],
            "max": [float(stacked[i].max().item()) for i in idx],
        }

    return {
        "what": "the planner's model vs the FULL plant under the SAME (T, omega_des) sequence, from the "
                "same random initial states — the error the delegation assumption makes over one "
                "planning horizon. Measurement only; no threshold is taken against it.",
        "n_states": int(n_states), "horizon": horizon, "dt": dt,
        "command": "the sampler's own draws about the hover anchor, rate-limited as the planner limits "
                   "them and low-pass filtered on both legs",
        "position_divergence_m": band(position_err),
        "body_rate_divergence_rad_s": band(rate_err),
        "PASS": None,
    }


@torch.no_grad()
def check_interface_reproduces_shipped(
    system: System, controller: RateCascadeController, n_probe: int, seed: int,
) -> dict:
    """(f) the (T, omega) interface expresses the shipped cascaded PD exactly."""
    dtype = controller.dtype
    generator = torch.Generator().manual_seed(int(seed))
    probe = system.wrap_state(torch.randn(n_probe, system.state_dim, generator=generator, dtype=dtype))
    goal = torch.randn(n_probe, 3, generator=generator, dtype=dtype)
    thrust, omega_des = shipped_pd_command(system, probe, goal)
    through = controller.inner_loop(probe, torch.cat([thrust, omega_des], dim=-1))
    reference = system.lqr_action(probe, goal)
    limited = controller.rate_limit(torch.cat([thrust, omega_des], dim=-1))
    would_limit = (limited[:, 1:] != omega_des)
    return {
        "what": "the SHIPPED cascaded PD written in this interface — omega_des = (kp_att/kd_att) "
                "e_att_body, k_rate = kd_att — against system.lqr_action itself. Non-circular: the "
                "reference is the shipped call with the shipped gains.",
        "n_probe_states": n_probe,
        "rate_gain_factor_of_this_controller": controller.rate_gain_factor,
        "k_rate": controller.k_rate, "kd_att": controller.kd_att, "kp_att": controller.kp_att,
        "max_abs_rotor_difference_N": float((through - reference).abs().max().item()),
        "max_abs_rotor_force_N": float(reference.abs().max().item()),
        "exact_to_float": bool(torch.equal(through, reference)),
        "rate_setpoint_would_hit_the_limit": {
            "n_channel_entries": int(would_limit.numel()),
            "n_limited": int(would_limit.sum().item()),
            "frac_limited": float(would_limit.double().mean().item()),
            "omega_max_read_from_system": controller.omega_max,
            "note": "reported because the planner DOES apply this limit to its own samples; the "
                    "comparison above is made without it so the algebraic identity is testable",
        },
        "PASS": None,
    }
