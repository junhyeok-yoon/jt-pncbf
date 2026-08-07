"""v2.8.2 in-house PPO baseline — training loop.

Policy only: NO certificate, NO filter, NO V_hat, NO CBF-QP, NO HardNet projection, NO dual_solve, NO
empty_fallback anywhere in the loop. The policy output reaches the plant through the action box (per-rotor box,
tanh then scale) and nothing else.

The batched rollout reuses src/common and src/envs verbatim — the training scene sampler (sample_train_scene
via _train_scene_sampler), scene batching (batch_scenes / initial_states_from_batch), the plant (rk4_step +
system.dynamics/observation), and the outcome predicates (step_outcomes / resolve_outcome) — so the plant,
sampler, and predicates are identical to the JT/OC group BY CONSTRUCTION. The run-dir / status.json /
metrics.csv / eval_metrics.csv / eval_episodes.csv / checkpoint layout mirrors src/frameworks/jt_pncbf/train.py.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from src._version import __version__
from src.common.eval_gate import eval_gate
from src.common.outcomes import resolve_outcome, step_outcomes
from src.common.rk4 import rk4_step
from src.common.signed_h import signed_h
from src.common.system import System
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.eval.evaluate import EVAL_EPISODE_COLUMNS, EVAL_METRIC_COLUMNS, evaluate
from src.eval.rollout import _physical_done_mask
from src.frameworks.jt_pncbf.train import (
    _append_csv,
    _git_commit_text,
    _init_csv,
    _make_summary_writer,
    _plain_data,
    _read_csv,
    _train_scene_sampler,
    _write_status,
    _write_tb_scalars,
    make_system,
)
from src.frameworks.ppo.nets import PPOPolicy, PPOValue


Tensor = torch.Tensor
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data"

METRIC_COLUMNS = [
    "iteration",
    "update_step",          # cumulative policy-gradient minibatch updates (== JT "step" for the 30000 budget)
    "env_steps",            # cumulative environment transitions consumed
    "wallclock_s",
    "steps_per_s",          # env transitions / s over the iteration (throughput)
    "iters_per_s",
    "n_active_transitions",
    "mean_episode_return",
    "train_reach",
    "train_collision",
    "train_oob",
    "train_stuck",
    "train_timeout",
    "pg_loss",
    "vf_loss",
    "entropy",
    "approx_kl",
    "clip_frac",
    "explained_variance",
    "log_std_mean",
    "grad_norm_pi",
    "grad_norm_vf",
    "diag_h_mean",          # DIAGNOSTIC margin h = signed_h over active steps; NEVER a reward term
    "diag_h_min",
    "diag_cos_theta_mean",  # DIAGNOSTIC mean e3^T R e3 over active steps (1=upright, -1=inverted)
    "diag_att_cost_mean",   # DIAGNOSTIC mean per-step attitude-penalty magnitude
    "diag_frac_in_goal_ball",  # DIAGNOSTIC fraction of active steps with dist<=0.3
    "diag_goal_bonus_per_ep",  # DIAGNOSTIC total accumulated r_goal bonus per episode (vs reach 30.0)
    "diag_goal_d_per_ep",      # DIAGNOSTIC r_goal distance channel accumulated per episode
    "diag_goal_v_per_ep",      # DIAGNOSTIC r_goal speed channel accumulated per episode
    "diag_goal_om_per_ep",     # DIAGNOSTIC r_goal angular-rate channel accumulated per episode
    "diag_shaping_accum_per_ep",  # DIAGNOSTIC potential shaping accumulated per episode
    "diag_easy_frac",          # DIAGNOSTIC curriculum easy-IC fraction this iteration
    "diag_closest_dist_med",   # DIAGNOSTIC median min-distance-to-goal over lanes
    "diag_omega_rp_closest",   # DIAGNOSTIC median roll/pitch rate at closest approach
    "diag_omega_yaw_closest",  # DIAGNOSTIC median yaw rate at closest approach
    "diag_d1_mean",         # DIAGNOSTIC mean 1st-difference control smoothness (normalized [0,1])
    "diag_d2_mean",         # DIAGNOSTIC mean 2nd-difference control smoothness (normalized [0,1])
    "n_born_terminal",      # count of born-terminal lanes dropped this iteration (event_step==0)
    "cuda_max_mem_mb",
    "figures",              # per-eval trajectory-figure status ("ok"/"ERROR:..."/""); lagged one eval; detail in eval/figure_status.csv
]

# v2.8.2 (labeling only, no score change): the PPO eval CSV = the shared EVAL_METRIC_COLUMNS with an `iter`
# column inserted right after the update-count `step`, so the uniform iteration cadence (150) is visible in the
# record without converting update_step->iter. PPO-local; the shared list (and the JT eval CSV) is untouched.
_PPO_EVAL_COLUMNS = list(EVAL_METRIC_COLUMNS)
if "iter" not in _PPO_EVAL_COLUMNS:
    _PPO_EVAL_COLUMNS.insert(_PPO_EVAL_COLUMNS.index("step") + 1, "iter")


@dataclass(frozen=True)
class PPOTrainingResult:
    run_dir: Path
    halted: bool
    halt_reason: str | None
    best_cps: float
    best_step: int
    final_update_step: int


# --------------------------------------------------------------------------------------------------------- #
# scene sampling / k derivation
# --------------------------------------------------------------------------------------------------------- #
def _sample_scene_batch(sampler, rng, config, system_name, n, device, dtype):
    scenes = [sampler(rng, config, system_name) for _ in range(int(n))]
    batched = batch_scenes(scenes, device=device, dtype=dtype)
    x0 = initial_states_from_batch(batched)
    return batched, x0


def measure_k_shaping(config, system, device, dtype, *, n_scenes: int = 4096,
                      seed: int = 20260804) -> tuple[float, float]:
    """DERIVE k = 5 / E[||p_0 - g||], the expectation measured on the TRAINING scene distribution (the
    distribution the dense shaping operates on). Returns (E[||p0-g||], k). Per Ng et al. the optimal policy is
    invariant to the shaping scale, so the measurement basis does not bias the comparison — it only conditions
    learning; the value is reported for transparency."""
    sampler = _train_scene_sampler(config)
    rng = np.random.default_rng(seed)
    batched, x0 = _sample_scene_batch(sampler, rng, config, system.name, n_scenes, device, dtype)
    with torch.no_grad():
        goal = torch.as_tensor(batched.goal, dtype=x0.dtype, device=x0.device)
        p0 = system.position(system.wrap_state(x0))
        dist0 = torch.linalg.norm(p0 - goal, dim=-1)
        e_dist = float(dist0.mean().item())
    return e_dist, 5.0 / max(e_dist, 1e-6)


# --------------------------------------------------------------------------------------------------------- #
# batched rollout + reward + GAE
# --------------------------------------------------------------------------------------------------------- #
@torch.no_grad()
def rollout_batch(system: System, policy: PPOPolicy, value: PPOValue, scene, x0: Tensor,
                  horizon: int, dt: float, config: Mapping[str, Any], generator: torch.Generator):
    """Full-episode batched rollout with eval-consistent freezing (physical_done lanes hold state and command
    zero, exactly as src.eval.rollout.rollout_eval). Returns the stored buffers and the complete state trace."""
    x = system.wrap_state(x0.detach())
    b = x.shape[0]
    states = [x]
    obs_steps, z_steps, logp_steps, val_steps, u_steps = [], [], [], [], []
    done_frozen = _physical_done_mask(system, scene, x.unsqueeze(0), config)
    for _ in range(horizon):
        obs = system.observation(x, scene)
        u, z, logp = policy.sample(obs, generator=generator)
        v = value(obs)
        u_exec = torch.where(done_frozen.unsqueeze(1), torch.zeros_like(u), u)
        x_next = rk4_step(system, x, u_exec, dt)
        x = torch.where(done_frozen.unsqueeze(1), x, x_next)
        states.append(x)
        obs_steps.append(obs)
        z_steps.append(z)
        logp_steps.append(logp)
        val_steps.append(v)
        u_steps.append(u)                               # SAMPLED action (pre freeze-zeroing) for smoothness
        done_frozen = done_frozen | _physical_done_mask(system, scene, x.unsqueeze(0), config)
    return {
        "states": torch.stack(states, dim=0),          # [T+1, B, D]
        "obs": torch.stack(obs_steps, dim=0),           # [T, B, obs_dim]
        "z": torch.stack(z_steps, dim=0),               # [T, B, action_dim]
        "logp": torch.stack(logp_steps, dim=0),         # [T, B]
        "val": torch.stack(val_steps, dim=0),           # [T, B]
        "u_sampled": torch.stack(u_steps, dim=0),       # [T, B, action_dim] sampled box action
    }


_TERMINAL_REWARD = {"goal": 30.0, "collision": -5.0, "oob": -1.0, "stuck": -1.0, "timeout": -1.0}  # v2.8.2 7th revision (Researcher): reach terminal +5 -> +30 (gamma^150 * 30 ~ 6.7 > shaping telescoping total ~5, so reaching now outranks it); collision stays -5. Operative value; config reward.reach kept in sync. Potential k=5 anchor (line ~128) UNCHANGED.


def compute_rewards_and_gae(roll, scene, system, config, *, gamma, lam, k_shaping, max_steps):
    """Build per-transition rewards from the registered design and compute GAE. Every episode terminates by the
    horizon with a scored terminal (reach/collision/oob/stuck/timeout), so the terminal value is 0 for all lanes
    (no bootstrap). Returns flattened active-transition tensors + diagnostics."""
    states = roll["states"]                                       # [T+1, B, D]
    val = roll["val"]                                             # [T, B]
    t_steps, b = val.shape
    device = states.device

    masks = step_outcomes(states, scene, system, config)
    resolved = resolve_outcome(masks)
    event_step = resolved.event_step.to(device)                  # [B], index into 0..T ; -1 == timeout
    outcomes = resolved.outcome                                  # list[str] length B

    # transition index at which each lane terminates: e>=1 -> e-1 ; timeout(e==-1) -> T-1 ; born-terminal(e==0) -> skip
    term_t = torch.where(event_step >= 1, event_step - 1,
                         torch.where(event_step < 0, torch.full_like(event_step, t_steps - 1),
                                     torch.full_like(event_step, -1)))
    t_idx = torch.arange(t_steps, device=device).unsqueeze(1)     # [T,1]
    active = (t_idx <= term_t.unsqueeze(0)) & (term_t.unsqueeze(0) >= 0)   # [T,B]

    goal = torch.as_tensor(scene.goal, dtype=states.dtype, device=device)
    dist = torch.linalg.norm(system.position(states) - goal, dim=-1)      # [T+1, B] full 3-D distance
    # Ng et al. potential-based shaping F = gamma*Phi(s') - Phi(s) with Phi(s) = -k*dist(s), i.e.
    # k*(dist_t - gamma*dist_{t+1}) — the gamma is REQUIRED for optimal-policy invariance (its omission was the
    # defect the addendum flagged; the gap (1-gamma)*k*d is small but the invariance guarantee needs it).
    shaping = k_shaping * (dist[:-1] - gamma * dist[1:])          # [T, B]
    per_step = -1.0 / float(max_steps)

    # terminal rewards read from config (reach = +30 since the 7th revision — the farming check and the
    # discounted-terminal fix both need the larger terminal). collision/oob/stuck/timeout unchanged.
    _tr = config["ppo"]["reward"]
    _term_map = {"goal": float(_tr.get("reach", 30.0)), "collision": float(_tr.get("collision", -5.0)),
                 "oob": float(_tr.get("oob", -1.0)), "stuck": float(_tr.get("stuck", -1.0)),
                 "timeout": float(_tr.get("timeout", -1.0))}
    terminal_reward = torch.tensor([_term_map[o] for o in outcomes], dtype=states.dtype, device=device)
    terminal_bonus = torch.zeros((t_steps, b), dtype=states.dtype, device=device)
    valid_lane = term_t >= 0
    if bool(valid_lane.any()):
        lanes = torch.nonzero(valid_lane, as_tuple=False).flatten()
        terminal_bonus[term_t[lanes], lanes] = terminal_reward[lanes]

    # Attitude deadband penalty (arena-fixed flight physics; quadrotor only — undefined without a thrust axis,
    # so OFF for the DI gate). r_att = -(w_att/T_max) * relu(cos(theta_hold) - cos(theta)), cos(theta)=e3^T R e3.
    # cos(theta_hold)=1/TWR is DERIVED from config (mass, gravity, rotor bound, n_rotors=action_dim); inside the
    # cone the term is exactly 0 (bank freely up to the tilt where horizontal accel peaks at theta_hold), outside
    # it penalizes the shortfall in altitude-hold authority. w_att=1.0 => a sustained fully-inverted episode (-1)
    # accumulates ~-1.5 over T_max=max_steps steps, the scale of the step penalty and well under +/-5 terminals.
    w_att = float(config["ppo"]["reward"].get("w_att", 0.0))
    r_att = torch.zeros((t_steps, b), dtype=states.dtype, device=device)
    cos_theta = None
    if w_att > 0.0 and hasattr(system, "thrust_axis"):
        cos_theta = system.thrust_axis(states)[..., 2]                     # [T+1, B] = e3^T R e3
        bnd = config["env"]["bounds"][system.name]
        phys = config["env"][system.name]
        twr = (int(system.action_dim) * float(bnd["f_rotor_max"])) / (float(phys["mass"]) * float(phys["gravity"]))
        cos_hold = 1.0 / twr
        r_att = -(w_att / float(max_steps)) * torch.relu(cos_hold - cos_theta[1:])   # on resulting states s_{t+1}

    # Dense positive GOAL-PROXIMITY bonus on the JOINT predicate (Researcher revision 9). The settling PENALTY
    # was net-negative on approach (its tax >= the potential's gain, so the rational optimum was to sit at d~2 m
    # — the measured closest-approach median), and everything near the goal was a cost while the reach terminal
    # was never sampled, so nothing pulled the policy in. r_goal = (w_goal/T_max)*exp(-[(d/d_ref)^2 +
    # (||v||/v_G)^2 + (||omega||/omega_G)^2]): maximal exactly where the reach predicate holds, decaying smoothly
    # outward -> a gradient in ALL THREE channels at once, never taxing the approach, covering the full ||omega||
    # (not the yaw axis alone). It REPLACES the settling penalty and the (implicit) angular-rate penalty. Farming
    # is bounded: reach ends the episode so the bonus is collected only just outside the predicate (<=~0.4*w_goal
    # per episode) << the +30 terminal; the accumulated bonus/episode is logged so this is checkable.
    # Revision 10 B1: SUM of INDEPENDENT per-channel bonuses (the rev-9 product exp was identically zero — one
    # off-channel, e.g. omega~3 => (omega/0.3)^2=112, zeroed the whole term AND its gradient in every channel;
    # goal_bonus/ep=0.00 recorded exactly that). A sum lets each channel carry its own gradient so no channel can
    # zero the others. v_ref/om_ref = 3x the terminal bounds (0.90) so there is gradient at the measured operating
    # point (omega~3), not 100 widths out. d_ref=1.0. Per-channel accumulation logged (a dead channel is visible).
    w_goal = float(config["ppo"]["reward"].get("w_goal", 0.0))
    d_ref = float(config["ppo"]["reward"].get("d_ref", 1.0))
    ref_mult = float(config["ppo"]["reward"].get("goal_ref_mult", 3.0))
    r_goal = torch.zeros((t_steps, b), dtype=states.dtype, device=device)
    gd = torch.zeros((t_steps, b), dtype=states.dtype, device=device)
    gv = torch.zeros((t_steps, b), dtype=states.dtype, device=device)
    gom = torch.zeros((t_steps, b), dtype=states.dtype, device=device)
    if w_goal > 0.0:
        v_ref = float(config["env"]["goal_speed_radius"]) * ref_mult
        om_g_base = float(config["env"].get("goal_angrate_radius", float("inf")))
        c = w_goal / float(max_steps)
        gd = c * torch.exp(-(dist[1:] / d_ref) ** 2)
        gv = c * torch.exp(-(system.speed(states)[1:] / v_ref) ** 2)
        if om_g_base == om_g_base and om_g_base != float("inf"):
            gom = c * torch.exp(-(system.angular_rate(states)[1:] / (om_g_base * ref_mult)) ** 2)
        r_goal = gd + gv + gom

    # Control-smoothness terms (Researcher addendum 5) on the SAMPLED action u (not u_exec — freeze-zeroing would
    # inject a spurious jump). d1 = ||u_t-u_{t-1}||/(sqrt(m)*range) (1 at a full simultaneous swing);
    # d2 = ||u_t-2u_{t-1}+u_{t-2}||/(2*sqrt(m)*range) (1 at a full alternating swing). r_smooth =
    # -(1/T_max)*(w_du*d1^2 + w_d2u*d2^2), squared (JT w_du form). Boundary d1[0]=0, d2[0]=d2[1]=0.
    u_s = roll.get("u_sampled")
    d1 = torch.zeros((t_steps, b), dtype=states.dtype, device=device)
    d2 = torch.zeros((t_steps, b), dtype=states.dtype, device=device)
    r_smooth = torch.zeros((t_steps, b), dtype=states.dtype, device=device)
    w_du = float(config["ppo"]["reward"].get("w_du", 0.0))
    w_d2u = float(config["ppo"]["reward"].get("w_d2u", 0.0))
    if u_s is not None and (w_du > 0.0 or w_d2u > 0.0) and t_steps >= 2:
        m_rotor = u_s.shape[-1]
        ub = system.u_bounds.to(device=device, dtype=states.dtype)
        u_range = float((ub[:, 1] - ub[:, 0])[0].item())                     # per-component box range (uniform)
        norm1 = math.sqrt(m_rotor) * u_range
        du = u_s[1:] - u_s[:-1]                                              # [T-1,B,m]
        d1[1:] = torch.linalg.norm(du, dim=-1) / norm1
        if t_steps >= 3:
            d2u = u_s[2:] - 2.0 * u_s[1:-1] + u_s[:-2]                       # [T-2,B,m]
            d2[2:] = torch.linalg.norm(d2u, dim=-1) / (2.0 * norm1)
        r_smooth = -(1.0 / float(max_steps)) * (w_du * d1 ** 2 + w_d2u * d2 ** 2)

    # Revision 9: the settling penalty and the yaw-rate penalty are REMOVED — the positive r_goal bonus subsumes
    # both (it rewards low ||v|| and low ||omega|| on every axis without gating the approach).
    reward = (per_step + shaping + terminal_bonus + r_att + r_goal + r_smooth) * active

    # GAE (backward). next_nonterminal[t] = 1 iff s_{t+1} is not terminal (t < term_t); else 0. Inactive-step
    # garbage is killed at the terminal step (next_nonterminal==0 there resets the carry) and masked out below.
    next_nonterminal = (t_idx < term_t.unsqueeze(0)).to(states.dtype)     # [T,B]
    adv = torch.zeros((t_steps, b), dtype=states.dtype, device=device)
    lastgae = torch.zeros(b, dtype=states.dtype, device=device)
    for t in range(t_steps - 1, -1, -1):
        m = next_nonterminal[t]
        next_v = val[t + 1] * m if t + 1 < t_steps else torch.zeros(b, dtype=states.dtype, device=device)
        delta = reward[t] + gamma * next_v - val[t]
        lastgae = delta + gamma * lam * m * lastgae
        adv[t] = lastgae
    returns = adv + val

    flat = active.reshape(-1)
    keep = torch.nonzero(flat, as_tuple=False).flatten()
    out = {
        "obs": roll["obs"].reshape(t_steps * b, -1)[keep],
        "z": roll["z"].reshape(t_steps * b, -1)[keep],
        "logp": roll["logp"].reshape(-1)[keep],
        "adv": adv.reshape(-1)[keep],
        "ret": returns.reshape(-1)[keep],
        "val": val.reshape(-1)[keep],
    }

    # diagnostics
    h = signed_h(system.position(states)[:-1], scene, float(config["env"]["h_scale"]))    # [T,B] margin
    h_active = h.reshape(-1)[keep]
    ep_return = (reward * active).sum(dim=0)                     # [B]
    # per-episode ACCUMULATED magnitudes (the check against farming/mis-scaling): the r_goal bonus and the
    # shaping, read beside the +30 reach terminal. mean over valid (non-born-terminal) lanes.
    goal_bonus_ep = (r_goal * active).sum(dim=0)                 # [B] accumulated positive goal bonus / episode
    goal_d_ep = (gd * active).sum(dim=0)                         # [B] per-channel accumulation (dead-channel check)
    goal_v_ep = (gv * active).sum(dim=0)
    goal_om_ep = (gom * active).sum(dim=0)
    shaping_ep = (shaping * active).sum(dim=0)                   # [B] accumulated potential shaping
    n_valid = int(valid_lane.sum().item())
    _vl = valid_lane
    def _m(t):
        return float(t[_vl].mean().item()) if n_valid else 0.0
    def _med(t):
        return float(t[_vl].median().item()) if n_valid else 0.0
    # closest-approach diagnostics (revision 9): min distance to goal per lane, and the omega split roll/pitch vs
    # yaw AT that step (quadrotor only; body rates at state indices 10:13 = wx,wy,wz).
    mind, mstep = dist.min(dim=0)                                # [B] over T+1 states
    rows_b = torch.arange(b, device=device)
    if getattr(system, "name", "") == "quadrotor_3d":
        om_at = states[mstep, rows_b, 10:13]                     # [B,3] body rates at closest approach
        rp_closest = torch.linalg.norm(om_at[:, :2], dim=-1)     # roll/pitch rate magnitude
        yaw_closest = om_at[:, 2].abs()                          # yaw rate magnitude
    else:
        rp_closest = torch.zeros(b, device=device)
        yaw_closest = torch.zeros(b, device=device)
    frac = {o: 0.0 for o in ("goal", "collision", "oob", "stuck", "timeout")}
    for o in outcomes:
        frac[o] = frac.get(o, 0.0) + 1.0
    n = max(1, len(outcomes))
    diag = {
        "n_active": int(keep.numel()),
        "mean_episode_return": float(ep_return[valid_lane].mean().item()) if n_valid else 0.0,
        "train_reach": frac["goal"] / n,
        "train_collision": frac["collision"] / n,
        "train_oob": frac["oob"] / n,
        "train_stuck": frac["stuck"] / n,
        "train_timeout": frac["timeout"] / n,
        "diag_h_mean": float(h_active.mean().item()) if h_active.numel() else 0.0,
        "diag_h_min": float(h_active.min().item()) if h_active.numel() else 0.0,
        "diag_cos_theta_mean": (float(cos_theta[1:].reshape(-1)[keep].mean().item())
                                if (cos_theta is not None and keep.numel()) else float("nan")),
        "diag_att_cost_mean": (float((-r_att).reshape(-1)[keep].mean().item()) if keep.numel() else 0.0),
        "diag_frac_in_goal_ball": (float((dist[1:].reshape(-1)[keep] <= 0.3).float().mean().item())
                                   if keep.numel() else 0.0),
        # accumulated-per-episode magnitudes (read beside reach terminal = 30.0):
        "diag_goal_bonus_per_ep": _m(goal_bonus_ep),           # total accumulated r_goal bonus / episode
        "diag_goal_d_per_ep": _m(goal_d_ep),                   # distance channel (dead-channel check)
        "diag_goal_v_per_ep": _m(goal_v_ep),                   # speed channel
        "diag_goal_om_per_ep": _m(goal_om_ep),                 # angular-rate channel
        "diag_shaping_accum_per_ep": _m(shaping_ep),           # potential shaping accumulated / episode
        # closest-approach trajectory (revision 9):
        "diag_closest_dist_med": _med(mind),                   # median min-distance-to-goal over lanes
        "diag_omega_rp_closest": _med(rp_closest),             # median roll/pitch rate at closest approach
        "diag_omega_yaw_closest": _med(yaw_closest),           # median yaw rate at closest approach
        "diag_d1_mean": (float(d1.reshape(-1)[keep].mean().item()) if keep.numel() else 0.0),
        "diag_d2_mean": (float(d2.reshape(-1)[keep].mean().item()) if keep.numel() else 0.0),
        "n_born_terminal": int((term_t < 0).sum().item()),   # lanes dropped (born-terminal, event_step==0)
    }
    return out, diag


# --------------------------------------------------------------------------------------------------------- #
# PPO update
# --------------------------------------------------------------------------------------------------------- #
def ppo_update(policy, value, opt_pi, opt_vf, buf, ppo_cfg, torch_gen):
    clip = float(ppo_cfg["clip_ratio"])
    ent_coef = float(ppo_cfg["ent_coef"])
    vf_coef = float(ppo_cfg["vf_coef"])
    max_grad_norm = float(ppo_cfg["max_grad_norm"])
    epochs = int(ppo_cfg["epochs"])
    n_minibatches = int(ppo_cfg["n_minibatches"])
    clip_vloss = bool(ppo_cfg["clip_vloss"])
    normalize_adv = bool(ppo_cfg["normalize_adv"])

    n = buf["obs"].shape[0]
    adv = buf["adv"]
    if normalize_adv and n > 1:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    device = buf["obs"].device

    pg_losses, vf_losses, ents, kls, clipfracs, gpi, gvf = [], [], [], [], [], [], []
    for _ in range(epochs):
        perm = torch.randperm(n, generator=torch_gen, device=device)
        # Minibatch SIZE follows from the batch (Researcher): if ppo.minibatch_size is set, use it (n_minibatches
        # then = ceil(n/mb_size), varying with the batch); else fall back to a fixed n_minibatches (ceil division
        # => exactly n_minibatches chunks). Either way updates/iter = epochs * #chunks; report the actual count.
        mb_target = int(ppo_cfg.get("minibatch_size", 0) or 0)
        mb_size = max(1, min(n, mb_target)) if mb_target > 0 else max(1, -(-n // n_minibatches))
        for start in range(0, n, mb_size):
            idx = perm[start:start + mb_size]
            if idx.numel() == 0:
                continue
            obs_mb, z_mb = buf["obs"][idx], buf["z"][idx]
            old_logp, adv_mb, ret_mb, val_mb = buf["logp"][idx], adv[idx], buf["ret"][idx], buf["val"][idx]

            new_logp, entropy = policy.evaluate(obs_mb, z_mb)
            ratio = (new_logp - old_logp).exp()
            surr1 = ratio * adv_mb
            surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv_mb
            pg_loss = -torch.min(surr1, surr2).mean()
            ent = entropy.mean()
            pi_loss = pg_loss - ent_coef * ent
            opt_pi.zero_grad(set_to_none=True)
            pi_loss.backward()
            g_pi = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            opt_pi.step()

            new_v = value(obs_mb)
            if clip_vloss:
                v_clipped = val_mb + torch.clamp(new_v - val_mb, -clip, clip)
                vf_loss = 0.5 * torch.max((new_v - ret_mb) ** 2, (v_clipped - ret_mb) ** 2).mean()
            else:
                vf_loss = 0.5 * ((new_v - ret_mb) ** 2).mean()
            opt_vf.zero_grad(set_to_none=True)
            (vf_coef * vf_loss).backward()
            g_vf = torch.nn.utils.clip_grad_norm_(value.parameters(), max_grad_norm)
            opt_vf.step()

            with torch.no_grad():
                logratio = new_logp - old_logp
                kls.append(float(((ratio - 1.0) - logratio).mean().item()))
                clipfracs.append(float(((ratio - 1.0).abs() > clip).float().mean().item()))
            pg_losses.append(float(pg_loss.item()))
            vf_losses.append(float(vf_loss.item()))
            ents.append(float(ent.item()))
            gpi.append(float(g_pi))
            gvf.append(float(g_vf))

    with torch.no_grad():
        var_ret = buf["ret"].var()
        ev = float(1.0 - (buf["ret"] - buf["val"]).var() / var_ret) if float(var_ret) > 0 else 0.0
    return {
        "pg_loss": float(np.mean(pg_losses)) if pg_losses else 0.0,
        "vf_loss": float(np.mean(vf_losses)) if vf_losses else 0.0,
        "entropy": float(np.mean(ents)) if ents else 0.0,
        "approx_kl": float(np.mean(kls)) if kls else 0.0,
        "clip_frac": float(np.mean(clipfracs)) if clipfracs else 0.0,
        "explained_variance": ev,
        "grad_norm_pi": float(np.mean(gpi)) if gpi else 0.0,
        "grad_norm_vf": float(np.mean(gvf)) if gvf else 0.0,
        "n_updates": len(pg_losses),
    }


# --------------------------------------------------------------------------------------------------------- #
# run-dir / eval / checkpoints
# --------------------------------------------------------------------------------------------------------- #
def build_policy_value(system: System, config: Mapping[str, Any], device, dtype):
    obs_dim = int(system.obs_dim)
    policy = PPOPolicy(obs_dim, system, config, init_log_std=float(config["ppo"]["init_log_std"]))
    value = PPOValue(obs_dim, config)
    return policy.to(device=device, dtype=dtype), value.to(device=device, dtype=dtype)


def load_policy_from_checkpoint(ckpt_path, device=None, dtype=torch.float32):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ck["config"]
    system = make_system(config)
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, value = build_policy_value(system, config, dev, dtype)
    policy.load_state_dict(ck["pi_state"])
    value.load_state_dict(ck["vf_state"])
    policy.eval()
    value.eval()
    return system, policy, value, config, ck


def _ppo_run_dir(output_root: Path, config: Mapping[str, Any], seed: int) -> Path:
    version = str(config["run"]["version"])
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{version}__ppo__{ts}__seed{seed}"     # ppo tag: self-describing, namespaced under ppo_baseline/
    run_dir = Path(output_root) / "runs" / version / "ppo_baseline" / run_id
    suffix = 1
    while run_dir.exists():
        run_dir = run_dir.parent / f"{run_id}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir()
    (run_dir / "tensorboard").mkdir()
    (run_dir / "config.yaml").write_text(yaml.safe_dump(_plain_data(config), sort_keys=False), encoding="utf-8")
    (run_dir / "git_commit.txt").write_text(_git_commit_text() + "\n", encoding="utf-8")
    _init_csv(run_dir / "metrics.csv", METRIC_COLUMNS)
    _init_csv(run_dir / "eval_metrics.csv", _PPO_EVAL_COLUMNS)
    _init_csv(run_dir / "eval_episodes.csv", EVAL_EPISODE_COLUMNS)
    return run_dir


def _save_ppo_checkpoint(path, policy, value, opt_pi, opt_vf, config, update_step, iteration,
                         env_steps, best_cps, best_step):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "version": __version__,
        "framework": "ppo",
        "step": int(update_step),
        "iteration": int(iteration),
        "env_steps": int(env_steps),
        "best_cps": float(best_cps),
        "best_step": int(best_step),
        "pi_state": policy.state_dict(),
        "vf_state": value.state_dict(),
        "opt_pi_state": opt_pi.state_dict(),
        "opt_vf_state": opt_vf.state_dict(),
        "config": _plain_data(config),
    }, path)


def run_inloop_eval(run_dir, writer, system, policy, value, config, inloop_pool_path, update_step, eg_lock,
                    eval_batch_size: int = 200, iteration: int = -1):
    from src.frameworks.ppo.filter_free import FilterFreeFramework
    fw = FilterFreeFramework(system, policy, value)
    eg = (config.get("eval", {}) or {}).get("gate", {}) or {}
    with eval_gate(eg_lock, enabled=bool(eg.get("enabled", True)), timeout_s=float(eg.get("timeout_s", 300.0)),
                   log=lambda m: print(m, flush=True)):
        policy.eval()
        # Chunk the (up to 2000-scene) eval so its transient allocations stay small; the un-chunked window
        # unfold otherwise fragments the caching allocator to several GB and breaches VRAM coexistence.
        result = evaluate(fw, inloop_pool_path, config, mode="in_loop", step=int(update_step),
                          ckpt_name=f"step_{update_step:06d}.pt", include_lqr_baseline=False,
                          eval_batch_size=int(eval_batch_size))
        policy.train()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()      # hand the eval's reserved cache back so concurrent-mission evals can use it
    result.eval_row["iter"] = int(iteration)   # v2.8.2 labeling: training iteration alongside update-count step
    cols = _PPO_EVAL_COLUMNS + [c for c in ("cps_full_half", "cps_tilt60_half") if c in result.eval_row]
    _append_csv(run_dir / "eval_metrics.csv", cols, [result.eval_row])
    _append_csv(run_dir / "eval_episodes.csv", EVAL_EPISODE_COLUMNS, result.episode_rows)
    _write_tb_scalars(writer, "eval/in_loop", result.eval_row, int(update_step))
    return result   # v2.8.2 B2: full result (trajectories) so the figure path reuses it -- no 2nd eval


# --------------------------------------------------------------------------------------------------------- #
# main training entry
# --------------------------------------------------------------------------------------------------------- #
def run_ppo_training(config: dict[str, Any], *, seed: int, inloop_pool_path, output_root: Path = DEFAULT_OUTPUT_ROOT,
                     device: str = "auto", one_third_reach_floor: float = 0.02) -> PPOTrainingResult:
    dev = torch.device("cuda" if (device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu")
    dtype = torch.float32
    config["run"]["version"] = __version__
    config["run"]["seed"] = int(seed)

    np.random.seed(seed)
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    torch_gen = torch.Generator(device=dev)
    torch_gen.manual_seed(seed)

    system = make_system(config)
    ppo_cfg = config["ppo"]
    # Training rollout horizon READ from config: eval.max_steps (the episode cap under which timeout is scored),
    # unless ppo.horizon explicitly overrides. State what it resolves to at launch.
    horizon = int(ppo_cfg["horizon"]) if ppo_cfg.get("horizon") else int(config["eval"]["max_steps"])
    ppo_cfg["horizon"] = horizon
    num_envs = int(ppo_cfg["num_envs"])
    n_iterations = int(ppo_cfg["n_iterations"])
    eval_cadence = int(ppo_cfg["eval_cadence_iters"])
    gamma = float(ppo_cfg["gamma"])
    lam = float(ppo_cfg["gae_lambda"])
    max_steps = int(config["eval"]["max_steps"])

    # k = 5 / E[||p0-g||] on the training distribution (report + stamp)
    if ppo_cfg["reward"].get("k_shaping") is None:
        e_dist, k_shaping = measure_k_shaping(config, system, dev, dtype)
        ppo_cfg["reward"]["k_shaping"] = float(k_shaping)
        ppo_cfg["reward"]["measured_E_p0_goal"] = float(e_dist)
    else:
        k_shaping = float(ppo_cfg["reward"]["k_shaping"])
        e_dist = float(ppo_cfg["reward"].get("measured_E_p0_goal", float("nan")))
    print(f"[ppo] k_shaping = 5 / E[||p0-g||] = 5 / {e_dist:.4f} = {k_shaping:.4f}", flush=True)

    # Attitude deadband penalty (quadrotor only): report + stamp the DERIVED threshold. cos(theta_hold)=1/TWR,
    # TWR = n_rotors * f_rotor_max / (m*g). Undefined without a thrust axis (DI) => term OFF there.
    w_att = float(ppo_cfg["reward"].get("w_att", 0.0))
    if w_att > 0.0 and hasattr(system, "thrust_axis"):
        _b = config["env"]["bounds"][system.name]
        _p = config["env"][system.name]
        _twr = (int(system.action_dim) * float(_b["f_rotor_max"])) / (float(_p["mass"]) * float(_p["gravity"]))
        import math as _math
        ppo_cfg["reward"]["_TWR"] = float(_twr)
        ppo_cfg["reward"]["_cos_theta_hold"] = float(1.0 / _twr)
        ppo_cfg["reward"]["_theta_hold_deg"] = float(_math.degrees(_math.acos(min(1.0, 1.0 / _twr))))
        print(f"[ppo] attitude penalty: w_att={w_att}  TWR={_twr:.4f}  cos(theta_hold)=1/TWR={1.0/_twr:.4f}  "
              f"theta_hold={ppo_cfg['reward']['_theta_hold_deg']:.2f} deg", flush=True)
    else:
        print(f"[ppo] attitude penalty OFF (w_att={w_att}, thrust_axis={hasattr(system, 'thrust_axis')})", flush=True)

    # Revision 9: dense positive goal-proximity bonus REPLACES the settling and yaw-rate penalties.
    w_goal = float(ppo_cfg["reward"].get("w_goal", 0.0))
    _reach = float(ppo_cfg["reward"].get("reach", 30.0))
    if w_goal > 0.0:
        _vg = float(config["env"]["goal_speed_radius"])
        _og = float(config["env"].get("goal_angrate_radius", float("inf")))
        _dr = float(ppo_cfg["reward"].get("d_ref", 1.0))
        _rm = float(ppo_cfg["reward"].get("goal_ref_mult", 3.0))
        print(f"[ppo] goal-proximity BONUS (rev 10, SUM of channels): w_goal={w_goal}  d_ref={_dr}  "
              f"v_ref=om_ref={_rm}x terminal={_rm*_vg:.2f}. r_goal=(w_goal/T_max)*[exp(-(d/d_ref)^2)+"
              f"exp(-(||v||/v_ref)^2)+exp(-(||omega||/om_ref)^2)]; positive, per-channel (no channel zeroes "
              f"others). Settling & yaw penalties REMOVED. reach terminal = {_reach:.0f}", flush=True)
    else:
        print(f"[ppo] goal-proximity bonus OFF (w_goal={w_goal})", flush=True)

    policy, value = build_policy_value(system, config, dev, dtype)
    opt_pi = torch.optim.Adam(policy.parameters(), lr=float(ppo_cfg["lr_pi"]), eps=float(ppo_cfg["adam_eps"]))
    opt_vf = torch.optim.Adam(value.parameters(), lr=float(ppo_cfg["lr_vf"]), eps=float(ppo_cfg["adam_eps"]))

    run_dir = _ppo_run_dir(output_root, config, seed)
    writer = _make_summary_writer(run_dir / "tensorboard")
    sampler = _train_scene_sampler(config)
    dt = float(config["env"]["dt"])
    eg_lock = Path(output_root) / "runs" / str(config["run"]["version"]) / "eval_gate.lock"

    # revision-10 B3: training-IC curriculum (PPO training sampler only; eval pools untouched -> Gate 2 safe).
    from src.frameworks.ppo.curriculum import easy_frac_at, sample_curriculum_scenes
    cur = dict(ppo_cfg.get("curriculum", {}) or {})
    cur_enabled = bool(cur.get("enabled", False))
    cur_f0 = float(cur.get("easy_frac0", 0.0))
    cur_anneal = float(cur.get("anneal_frac", 0.7))
    easy_bounds = {k: float(cur.get(k, d)) for k, d in
                   (("d_lo", 0.15), ("d_hi", 0.8), ("v_max", 0.25), ("om_max", 0.25), ("max_tilt", 0.35))}

    best_cps, best_step, halt_reason = -float("inf"), 0, None
    update_step, env_steps, total_born_terminal = 0, 0, 0
    _write_status(run_dir, stage="full", phase="training", current_step=0, best_step=0,
                  best_cps=best_cps, halt_reason=None)
    fig_pool = None                                          # loaded lazily on first eval (for per-eval grids)
    fig_status = ""                                           # per-eval figure status, written into the metrics row (lagged one eval); immediate detail in eval/figure_status.csv
    print(f"[ppo] run_dir={run_dir}  system={system.name}  obs_dim={system.obs_dim}  device={dev}  "
          f"horizon={horizon} (=eval.max_steps)  num_envs={num_envs}", flush=True)
    if cur_enabled:
        print(f"[ppo] TRAINING-IC CURRICULUM (rev 10 B3): easy_frac0={cur_f0} anneal_frac={cur_anneal} "
              f"(linear f0->0 by iter {int(cur_anneal*n_iterations)}); easy IC: dist~U[{easy_bounds['d_lo']},"
              f"{easy_bounds['d_hi']}]m, ||v0||<=U[0,{easy_bounds['v_max']}], ||omega0||<=U[0,{easy_bounds['om_max']}], "
              f"tilt<=U[0,{easy_bounds['max_tilt']}]rad. TRAINING sampler ONLY; eval pools/scene_init.py untouched "
              f"(Gate 2 safe). Asymmetry vs JT (analytic settling grad via w_settle/w_settle_ang) recorded.", flush=True)
    else:
        print(f"[ppo] training-IC curriculum OFF", flush=True)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_start = time.time()

    for iteration in range(1, n_iterations + 1):
        t_iter = time.time()
        easy_frac = easy_frac_at(iteration - 1, n_iterations, cur_f0, cur_anneal) if cur_enabled else 0.0
        if easy_frac > 0.0:
            _scn = sample_curriculum_scenes(np_rng, config, system.name, num_envs, easy_frac, easy_bounds)
            batched = batch_scenes(_scn, device=dev, dtype=dtype)
            x0 = initial_states_from_batch(batched)
        else:
            batched, x0 = _sample_scene_batch(sampler, np_rng, config, system.name, num_envs, dev, dtype)
        roll = rollout_batch(system, policy, value, batched, x0, horizon, dt, config, torch_gen)
        buf, diag = compute_rewards_and_gae(roll, batched, system, config, gamma=gamma, lam=lam,
                                            k_shaping=k_shaping, max_steps=max_steps)
        env_steps += diag["n_active"]
        total_born_terminal += diag["n_born_terminal"]
        if buf["obs"].shape[0] == 0:
            print(f"[ppo] iter {iteration}: no active transitions (all born-terminal); skipping update", flush=True)
            continue
        stats = ppo_update(policy, value, opt_pi, opt_vf, buf, ppo_cfg, torch_gen)
        update_step += stats["n_updates"]

        iter_dt = time.time() - t_iter
        row = {
            "iteration": iteration, "update_step": update_step, "env_steps": env_steps,
            "wallclock_s": time.time() - t_start,
            "steps_per_s": diag["n_active"] / max(iter_dt, 1e-9),
            "iters_per_s": 1.0 / max(iter_dt, 1e-9),
            "n_active_transitions": diag["n_active"], "mean_episode_return": diag["mean_episode_return"],
            "train_reach": diag["train_reach"], "train_collision": diag["train_collision"],
            "train_oob": diag["train_oob"], "train_stuck": diag["train_stuck"],
            "train_timeout": diag["train_timeout"],
            "pg_loss": stats["pg_loss"], "vf_loss": stats["vf_loss"], "entropy": stats["entropy"],
            "approx_kl": stats["approx_kl"], "clip_frac": stats["clip_frac"],
            "explained_variance": stats["explained_variance"],
            "log_std_mean": float(policy.log_std.mean().item()),
            "grad_norm_pi": stats["grad_norm_pi"], "grad_norm_vf": stats["grad_norm_vf"],
            "diag_h_mean": diag["diag_h_mean"], "diag_h_min": diag["diag_h_min"],
            "diag_cos_theta_mean": diag["diag_cos_theta_mean"], "diag_att_cost_mean": diag["diag_att_cost_mean"],
            "diag_frac_in_goal_ball": diag["diag_frac_in_goal_ball"],
            "diag_goal_bonus_per_ep": diag["diag_goal_bonus_per_ep"],
            "diag_goal_d_per_ep": diag["diag_goal_d_per_ep"],
            "diag_goal_v_per_ep": diag["diag_goal_v_per_ep"],
            "diag_goal_om_per_ep": diag["diag_goal_om_per_ep"],
            "diag_shaping_accum_per_ep": diag["diag_shaping_accum_per_ep"],
            "diag_easy_frac": easy_frac,
            "diag_closest_dist_med": diag["diag_closest_dist_med"],
            "diag_omega_rp_closest": diag["diag_omega_rp_closest"],
            "diag_omega_yaw_closest": diag["diag_omega_yaw_closest"],
            "diag_d1_mean": diag["diag_d1_mean"], "diag_d2_mean": diag["diag_d2_mean"],
            "n_born_terminal": diag["n_born_terminal"],
            "cuda_max_mem_mb": (torch.cuda.max_memory_allocated() / 1e6) if dev.type == "cuda" else 0.0,
            "figures": fig_status,
        }
        _append_csv(run_dir / "metrics.csv", METRIC_COLUMNS, [row])
        _write_tb_scalars(writer, "train", row, update_step)
        # revision-9 trajectory surfaced EARLY: reach, closest-approach dist + omega split (roll/pitch vs yaw),
        # accumulated goal bonus (farming check vs reach 30), log_std, approx_kl (overfitting check on batch reuse).
        print(f"[ppo] iter {iteration}/{n_iterations} upd={update_step} env={env_steps} easy={easy_frac:.2f} "
              f"reach={diag['train_reach']:.4f} coll={diag['train_collision']:.3f} "
              f"ret={diag['mean_episode_return']:.2f} kl={stats['approx_kl']:.4f} "
              f"closest_d={diag['diag_closest_dist_med']:.2f} om(rp={diag['diag_omega_rp_closest']:.2f},"
              f"yaw={diag['diag_omega_yaw_closest']:.2f}) goalb/ep(d={diag['diag_goal_d_per_ep']:.2f},"
              f"v={diag['diag_goal_v_per_ep']:.2f},om={diag['diag_goal_om_per_ep']:.2f}) "
              f"logstd={row['log_std_mean']:.2f} sps={row['steps_per_s']:.0f} vram={row['cuda_max_mem_mb']:.0f}MB", flush=True)

        # 1/3-budget shaping-failure early stop
        if iteration == max(1, n_iterations // 3) and diag["train_reach"] < one_third_reach_floor:
            halt_reason = f"reach_near_zero_at_third_budget ({diag['train_reach']:.4f} < {one_third_reach_floor})"
            print(f"[ppo] STOP: {halt_reason}", flush=True)
            break

        if iteration % eval_cadence == 0 or iteration == n_iterations:
            result = run_inloop_eval(run_dir, writer, system, policy, value, config,
                                     inloop_pool_path, update_step, eg_lock, iteration=iteration)
            eval_row = result.eval_row
            cps = float(eval_row["cps"])
            print(f"[ppo]   in-loop eval @upd {update_step}: cps={cps:.4f} reach={eval_row['reach']:.4f} "
                  f"coll={eval_row['collision']:.4f} infeas={eval_row['infeasibility']:.4f}", flush=True)
            # Per-eval trajectory grids: ALWAYS attempted (the diagnostic for an unfiltered policy — NOT optional).
            # Kept in try/except so a plotting bug never kills training, but a failure is NEVER silent: the full
            # traceback is written to the run dir AND the status is recorded (eval/figure_status.csv + the next
            # metrics row's 'figures' column). n_scenes kept modest so the LQR-baseline rollout's transient VRAM
            # stays small (the earlier OOM was the 200-scene LQR spike on a shared GPU).
            (run_dir / "eval").mkdir(parents=True, exist_ok=True)
            try:
                from src.frameworks.ppo.figures import write_trajectory_grids
                # v2.8.2 B2/B3: reuse the in-loop eval's `result` (NO 2nd PPO rollout); the eval gate now
                # covers ONLY the 32-scene LQR rollout INSIDE write_trajectory_grids (via eg_lock) --
                # panel selection, matplotlib and file writes run UNGATED. n_scenes = panel-selection subset.
                write_trajectory_grids(system, config, result, run_dir / "eval",
                                       step=update_step, n_scenes=96, eg_lock=eg_lock)
                _pngs = sorted(p.name for p in (run_dir / "eval").glob(f"*{update_step:06d}*.png"))
                fig_status = "ok" if _pngs else "ERROR:no_png_written"
                _append_csv(run_dir / "eval" / "figure_status.csv", ["step", "status", "detail"],
                            [{"step": update_step, "status": fig_status, "detail": ";".join(_pngs)[:300]}])
                print(f"[ppo]   per-eval figures @upd {update_step}: {fig_status} {_pngs}", flush=True)
            except Exception as _e:                      # viz must never fail training — but never SILENTLY
                import traceback as _tb
                fig_status = f"ERROR:{type(_e).__name__}"
                (run_dir / "eval" / f"figure_error_step{update_step:06d}.txt").write_text(_tb.format_exc())
                _append_csv(run_dir / "eval" / "figure_status.csv", ["step", "status", "detail"],
                            [{"step": update_step, "status": fig_status, "detail": f"{type(_e).__name__}: {_e}"}])
                print(f"[ppo]   per-eval figures FAILED @upd {update_step}: {type(_e).__name__}: {_e} "
                      f"-> eval/figure_error_step{update_step:06d}.txt", flush=True)
            if cps > best_cps:
                best_cps, best_step = cps, update_step
                _save_ppo_checkpoint(run_dir / "checkpoints/best.pt", policy, value, opt_pi, opt_vf,
                                     config, update_step, iteration, env_steps, best_cps, best_step)
            _save_ppo_checkpoint(run_dir / f"checkpoints/step_{update_step:06d}.pt", policy, value, opt_pi,
                                 opt_vf, config, update_step, iteration, env_steps, best_cps, best_step)
            _write_status(run_dir, stage="full", phase="training", current_step=update_step,
                          best_step=best_step, best_cps=best_cps, halt_reason=None)

    _save_ppo_checkpoint(run_dir / "checkpoints/final.pt", policy, value, opt_pi, opt_vf, config,
                         update_step, iteration, env_steps, best_cps, best_step)
    if not (run_dir / "checkpoints/best.pt").exists():
        _save_ppo_checkpoint(run_dir / "checkpoints/best.pt", policy, value, opt_pi, opt_vf, config,
                             update_step, iteration, env_steps, best_cps, best_step)
    _write_status(run_dir, stage="full", phase="halted" if halt_reason else "done",
                  current_step=update_step, best_step=best_step, best_cps=best_cps, halt_reason=halt_reason)
    writer.close()
    _wall = time.time() - t_start
    print(f"[ppo] DONE run_dir={run_dir} best_cps={best_cps:.4f}@{best_step} halt_reason={halt_reason}", flush=True)
    print(f"[ppo] BUDGET (comparable units): env_interactions={env_steps}  policy_updates={update_step}  "
          f"wall_s={_wall:.0f} ({_wall/3600:.2f} h) vs JT ~5.7 h / 30000 steps; "
          f"born_terminal_lanes_dropped={total_born_terminal}", flush=True)
    return PPOTrainingResult(run_dir=run_dir, halted=halt_reason is not None, halt_reason=halt_reason,
                             best_cps=best_cps, best_step=best_step, final_update_step=update_step)
