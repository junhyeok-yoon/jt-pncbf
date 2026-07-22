"""v2.7.4 M0d — invariant tests for the full-SO(3) initial-condition sampler (NEW file).

(i)  statistical: cos(tilt) is uniform on [-1,1] over 20000 SO(3) draws (KS at the 1% level). This is the
     test that catches the uniform-Euler-angles bug, whose measure is badly non-uniform yet looks fine.
(ii) termination: a 180 deg-tilt state with no cylinder within 2 m does not terminate and does not register
     a collision outcome (changes.md §6 "no tilt-based termination").
(iii) observation: obs_dim is still 32 and the observation builder is untouched.
"""
import math

import numpy as np
import torch
from scipy import stats

from src.envs.quadrotor_3d import _quat_to_R
from src.envs.scene_init import _shoemake_quat
from src.common.outcomes import step_outcomes
from src.frameworks.jt_pncbf.train import load_effective_config, make_system


def _cfg():
    c = load_effective_config()
    c["run"]["system"] = "quadrotor_3d"
    return c


def _sys():
    return make_system(_cfg())


def test_i_cos_tilt_uniform_so3_ks():
    """cos(tilt) = R(q)[2,2] uniform on [-1,1] under Shoemake SO(3), KS test p > 0.01 (20000 draws)."""
    rng = np.random.default_rng(20240719)
    n = 20000
    quats = np.stack([_shoemake_quat(rng) for _ in range(n)])
    q = torch.tensor(quats, dtype=torch.float64)
    R = _quat_to_R(q)
    cos_tilt = R[:, 2, 2].clamp(-1.0, 1.0).numpy()      # body-up . world-up = angle(body-z, world-z)
    # uniform on [-1, 1] == scipy uniform(loc=-1, scale=2)
    ks_stat, p = stats.kstest(cos_tilt, "uniform", args=(-1.0, 2.0))
    assert p > 0.01, f"cos(tilt) not uniform on [-1,1]: KS={ks_stat:.4f} p={p:.4g}"
    # sanity: under uniform SO(3), P(tilt>90deg)=P(cos<0)=0.5 and P(tilt>150deg)=P(cos<-cos30)~6.7%
    assert abs(float((cos_tilt < 0).mean()) - 0.50) < 0.02
    assert abs(float((cos_tilt < -math.cos(math.radians(30))).mean()) - 0.0670) < 0.01


def test_ii_no_tilt_termination_or_collision_far_from_cylinder():
    """180 deg tilt, no cylinder within 2 m -> no collision and no termination (episode keeps flying)."""
    s = _sys()
    cfg = _cfg()
    # q = [0,1,0,0] is a 180 deg rotation about body-x: R = diag(1,-1,-1) -> body-up = (0,0,-1) -> tilt=180deg
    q180 = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    R = _quat_to_R(q180.unsqueeze(0))
    assert float(R[0, 2, 2]) < -0.999, "constructed state is not at 180 deg tilt"
    p = torch.zeros(3, dtype=torch.float64)
    v = torch.zeros(3, dtype=torch.float64)
    om = torch.zeros(3, dtype=torch.float64)
    x = torch.cat([p, q180, v, om]).reshape(1, 1, 13)          # [T=1, B=1, 13]
    from types import SimpleNamespace
    scene = SimpleNamespace(
        goal=torch.tensor([[8.0, 8.0, 8.0]], dtype=torch.float64),      # goal far
        obstacle_centers=torch.tensor([[[5.0, 0.0]]], dtype=torch.float64),  # one cylinder at (5,0): 5 m away
        obstacle_radii=torch.tensor([[0.5]], dtype=torch.float64),           # surface 4.5 m >> 2 m
        obstacle_active=torch.tensor([[True]]),
    )
    masks = step_outcomes(x, scene, s, cfg)
    assert not bool(masks.collided.any()), "180 deg tilt far from any cylinder wrongly registered a collision"
    assert not bool(masks.oob.any()), "in-bounds tilted state wrongly flagged out-of-bounds"
    assert not bool(masks.goal_reached.any()), "tilted state 11 m from goal wrongly flagged goal-reached"
    # nothing in step_outcomes depends on tilt -> tilt cannot terminate the episode


def test_iii_obs_dim_still_32():
    """obs_dim is unchanged (32) and the observation builder is not touched by the IC axis."""
    s = _sys()
    assert s.obs_dim == 32, f"obs_dim changed to {s.obs_dim}"
    # a forward pass produces a dim-32 observation
    from types import SimpleNamespace
    x = torch.zeros(4, 13, dtype=torch.float64); x[:, 3] = 1.0     # identity quat
    scene = SimpleNamespace(
        goal=torch.zeros(4, 3, dtype=torch.float64),
        obstacle_centers=torch.zeros(4, 5, 2, dtype=torch.float64),
        obstacle_radii=torch.ones(4, 5, dtype=torch.float64) * 0.3,
        obstacle_active=torch.zeros(4, 5, dtype=torch.bool),
    )
    obs = s.observation(x, scene)
    assert obs.shape[-1] == 32, f"observation dim {obs.shape[-1]} != 32"
