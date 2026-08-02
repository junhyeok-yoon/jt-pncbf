"""v2.8.0 S2 B7 — the angular reach condition.

(i)   quadrotor_3d: a near-goal, low-linear-speed, high-omega state is NOT goal_reached until
      ||omega|| <= goal_angrate_radius.
(ii)  the continuing collector's segment-close goal test agrees with outcomes.step_outcomes on the
      same states (the predicate is defined identically in both copies).
(iii) double_integrator and unicycle are unaffected: goal_reached is identical whether
      goal_angrate_radius is set or absent (angular_rate is a structural zero there).
"""
from __future__ import annotations
import copy
from types import SimpleNamespace

import torch

from src.common.outcomes import step_outcomes
from src.frameworks.jt_pncbf.train import load_effective_config, make_system


def _cfg(system, angrate=0.48):
    c = copy.deepcopy(load_effective_config())
    c["run"]["system"] = system
    c["env"]["goal_angrate_radius"] = angrate
    c["env"]["band_collision_limit"] = 0.0
    return c


def _scene(goal, dim):
    g = torch.tensor([goal], dtype=torch.float64)
    return SimpleNamespace(
        goal=g,
        obstacle_centers=torch.zeros(1, 1, 2, dtype=torch.float64),
        obstacle_radii=torch.zeros(1, 1, dtype=torch.float64),
        obstacle_active=torch.zeros(1, 1, dtype=torch.bool),
    )


def _q3d_state(p, v, omega):
    # [px,py,pz, qw,qx,qy,qz, vx,vy,vz, wx,wy,wz]
    x = torch.zeros(13, dtype=torch.float64)
    x[0:3] = torch.tensor(p); x[3] = 1.0
    x[7:10] = torch.tensor(v); x[10:13] = torch.tensor(omega)
    return x


def test_i_angular_gate_blocks_reach_until_omega_below_threshold():
    s = make_system(_cfg("quadrotor_3d"))
    cfg = _cfg("quadrotor_3d", angrate=0.48)
    goal = [0.0, 0.0, 0.0]
    sc = _scene(goal, 3)
    # near goal, low linear speed, HIGH omega (1.5 > 0.48) -> not reached
    x_hi = _q3d_state([0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [1.5, 0.0, 0.0]).reshape(1, 1, 13)
    m_hi = step_outcomes(x_hi, sc, s, cfg)
    assert not bool(m_hi.goal_reached.any()), "high-omega near-goal state wrongly flagged reached"
    # same but LOW omega (0.1 < 0.48) -> reached
    x_lo = _q3d_state([0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.1, 0.0, 0.0]).reshape(1, 1, 13)
    m_lo = step_outcomes(x_lo, sc, s, cfg)
    assert bool(m_lo.goal_reached.any()), "low-omega near-goal settled state wrongly not reached"
    # absent key (inf) -> the high-omega state IS reached (byte-identical to pre-angular predicate)
    cfg_off = _cfg("quadrotor_3d"); cfg_off["env"].pop("goal_angrate_radius")
    m_off = step_outcomes(x_hi, sc, s, cfg_off)
    assert bool(m_off.goal_reached.any()), "absent goal_angrate_radius should be inert (inf)"


def test_ii_collector_goal_test_agrees_with_outcomes():
    s = make_system(_cfg("quadrotor_3d"))
    cfg = _cfg("quadrotor_3d", angrate=0.48)
    goal = [0.0, 0.0, 0.0]
    sc = _scene(goal, 3)
    # a batch spanning the omega threshold
    xs = torch.stack([
        _q3d_state([0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [w, 0.0, 0.0]) for w in (0.0, 0.2, 0.47, 0.49, 1.0, 2.0)
    ]).reshape(1, -1, 13)
    m = step_outcomes(xs, sc, s, cfg)
    goal_outcomes = m.goal_reached[0]                          # [B]
    # the collector's segment-close predicate, computed the same way advance_round does
    p = s.position(xs[0]); dist = torch.linalg.norm(p - sc.goal, dim=1)
    spd = s.speed(xs[0]); arate = s.angular_rate(xs[0])
    gr = float(cfg["env"]["goal_radius"]); gs = float(cfg["env"]["goal_speed_radius"]); ga = float(cfg["env"]["goal_angrate_radius"])
    collector_goal = (dist <= gr) & (spd <= gs) & (arate <= ga)
    assert torch.equal(goal_outcomes, collector_goal), "collector goal test disagrees with outcomes.py"


def test_iii_di_and_unicycle_unaffected_by_angular_key():
    for system, sdim, mk in (("double_integrator", 4, lambda: _di_state()), ("unicycle", 4, lambda: _uni_state())):
        s = make_system(_cfg(system))
        sc = _scene([0.0, 0.0], 2)
        x = mk().reshape(1, 1, sdim)
        m_on = step_outcomes(x, sc, s, _cfg(system, angrate=0.48))
        cfg_off = _cfg(system); cfg_off["env"].pop("goal_angrate_radius")
        m_off = step_outcomes(x, sc, s, cfg_off)
        assert torch.equal(m_on.goal_reached, m_off.goal_reached), f"{system} reach changed by angular key"


def _di_state():
    x = torch.zeros(4, dtype=torch.float64); x[0:2] = 0.0; x[2:4] = torch.tensor([0.05, 0.0]); return x


def _uni_state():
    # [px,py,theta,speed]
    x = torch.zeros(4, dtype=torch.float64); x[3] = 0.05; return x
