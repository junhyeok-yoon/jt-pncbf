"""v2.7.6 Stage-2: pin the band collision predicate (|p_z| >= limit) and the h_star vertical branch psi cap.

Both features are config-gated and DEFAULT OFF, so these tests explicitly enable them; the rest of the suite
(which does not) exercises the unchanged legacy behaviour."""
from __future__ import annotations

import copy
import math

import torch

from src.common.outcomes import step_outcomes
from src.common.quadrotor_barrier import value_target_barrier
from src.envs.scene_batch import batch_scenes
from src.envs.scene_init import sample_train_scene
from src.frameworks.jt_pncbf.train import load_effective_config, make_system

import numpy as np


def _cfg(banded=True):
    c = copy.deepcopy(load_effective_config())
    c["run"]["system"] = "quadrotor_3d"
    c["env"]["band_collision_limit"] = 4.0 if banded else 0.0
    c["env"]["band_hazard"] = {"enabled": banded, "limit": 4.0}
    return c


def _scene(cfg):
    return sample_train_scene(np.random.default_rng(0), cfg, "quadrotor_3d")


def _state(z, vz, xy=6.0):
    x = torch.zeros(1, 13, dtype=torch.float64)
    x[0, 0] = xy; x[0, 1] = xy; x[0, 2] = z; x[0, 3] = 1.0; x[0, 9] = vz
    return x


def test_band_collision_fires_on_z_surface_and_off_when_disabled():
    cfg = _cfg(banded=True); sc = _scene(cfg); sys = make_system(cfg)
    traj = torch.stack([_state(0.0, 0.0), _state(4.05, 0.0), _state(-4.2, 0.0)], dim=0)  # [3,1,13]
    collided = step_outcomes(traj, sc, sys, cfg).collided.reshape(-1).tolist()
    assert collided == [False, True, True]          # in-band no, ceiling yes, floor yes
    cfg_off = _cfg(banded=False)
    collided_off = step_outcomes(traj, sc, sys, cfg_off).collided.reshape(-1).tolist()
    assert collided_off == [False, False, False]    # legacy: |z|>=4 is NOT a collision


def test_oob_predicate_unchanged():
    # collision at |z|>=4 must not touch the oob predicate at |z|>8: a state at z=-8.5 is still oob.
    cfg = _cfg(banded=True); sc = _scene(cfg); sys = make_system(cfg)
    masks = step_outcomes(_state(-8.5, 0.0).unsqueeze(0)[0].unsqueeze(0), sc, sys, cfg)
    assert bool(masks.oob.reshape(-1)[0]) is True


def test_psi_cap_bounds_vertical_branch_at_r_max():
    cfg = _cfg(banded=True); sc = _scene(cfg); sys = make_system(cfg)
    bs = batch_scenes([sc], device=torch.device("cpu"), dtype=torch.float64)
    r_max = float(cfg["obstacle"]["per_system"]["quadrotor_3d"]["r_max"])
    # deep excursion z=12, v_z=0: vertical branch = min(z-4, r_max) + 0 = r_max (capped, not z-4=8).
    h = float(value_target_barrier(sys, _state(12.0, 0.0), bs, cfg))
    assert abs(h - r_max) < 1e-6


def test_vertical_relative_degree_one_dh_dvz_equals_pm_cz():
    cfg = _cfg(banded=True); sc = _scene(cfg); sys = make_system(cfg)
    bs = batch_scenes([sc], device=torch.device("cpu"), dtype=torch.float64)
    c_z = math.pi / float(cfg["env"]["bounds"]["quadrotor_3d"]["omega_max"])
    for z, vz, sign in [(4.5, 1.0, +1.0), (-4.5, -1.0, -1.0), (12.0, 1.0, +1.0)]:  # incl inside the cap
        x = _state(z, vz).requires_grad_(True)
        g = torch.autograd.grad(value_target_barrier(sys, x, bs, cfg).sum(), x)[0][0, 9].item()
        assert abs(g - sign * c_z) < 1e-9


def test_vertical_branch_off_is_legacy_bit_parity():
    cfg_on = _cfg(banded=True); cfg_off = _cfg(banded=False)
    cfg_nokey = _cfg(banded=False); del cfg_nokey["env"]["band_hazard"]   # legacy: no band_hazard key at all
    sc = _scene(cfg_on); sys = make_system(cfg_on)
    bs = batch_scenes([sc], device=torch.device("cpu"), dtype=torch.float64)
    x = _state(3.0, 0.5)   # ascending toward the ceiling: the vertical branch is genuinely active when ON
    h_off = float(value_target_barrier(sys, x, bs, cfg_off))
    h_nokey = float(value_target_barrier(sys, x, bs, cfg_nokey))
    h_on = float(value_target_barrier(sys, x, bs, cfg_on))
    assert h_off == h_nokey                          # disabled == no-key: legacy bit-parity, branch skipped
    assert h_on > h_off                              # enabled: the ascending-toward-ceiling branch raises h
    # and deep mid-band stationary: the vertical branch is dominated -> ON == OFF there
    x0 = _state(0.0, 0.0)
    assert float(value_target_barrier(sys, x0, bs, cfg_on)) == float(value_target_barrier(sys, x0, bs, cfg_off))


# ---- v2.7.6 obs_band_z (32 -> 34): symmetry group is now xy-translation + yaw, NOT z-translation ----

def _q3_scene_state(dt=torch.float64):
    from types import SimpleNamespace
    x = torch.zeros(3, 13, dtype=dt); x[:, 0] = 0.6; x[:, 1] = -0.2; x[:, 2] = 1.0; x[:, 3] = 1.0  # identity quat
    x[:, 7] = 0.3; x[:, 8] = -0.5; x[:, 9] = 0.8                          # v = (vx,vy,vz)
    sc = SimpleNamespace(goal=torch.tensor([[1.5, 1.0, 2.0]], dtype=dt).repeat(3, 1),
                         obstacle_centers=torch.tensor([[[1.2, 0.3]], [[1.2, 0.3]], [[1.2, 0.3]]], dtype=dt),
                         obstacle_radii=torch.full((3, 1), 0.4, dtype=dt),
                         obstacle_active=torch.ones(3, 1, dtype=torch.bool))
    return x, sc


def test_obs_band_z_is_xy_translation_invariant_not_z():
    """05_code s5.4 corrected: |z|=4 is a hazard, so z-translation leaves the symmetry group. The dim-34
    observation is invariant under a random XY shift of (start, goal, obstacles) and NOT under a z shift."""
    cfg = _cfg(banded=True); sys = make_system(cfg); assert sys.obs_dim == 34
    x, sc = _q3_scene_state()
    from types import SimpleNamespace
    obs0 = sys.observation(x, sc)
    dxy = torch.tensor([0.7, -1.3, 0.0], dtype=torch.float64)             # xy-only world shift
    xs = x.clone(); xs[:, :3] += dxy
    scs = SimpleNamespace(goal=sc.goal + dxy, obstacle_centers=sc.obstacle_centers + dxy[:2],
                          obstacle_radii=sc.obstacle_radii, obstacle_active=sc.obstacle_active)
    assert torch.allclose(sys.observation(xs, scs), obs0, atol=1e-9)      # xy-invariant
    # z shift of start+goal is NOT invariant (p_z slot changes) -> confirms z left the group
    dz = torch.tensor([0.0, 0.0, 1.6], dtype=torch.float64)
    xz = x.clone(); xz[:, :3] += dz
    scz = SimpleNamespace(goal=sc.goal + dz, obstacle_centers=sc.obstacle_centers,
                          obstacle_radii=sc.obstacle_radii, obstacle_active=sc.obstacle_active)
    assert not torch.allclose(sys.observation(xz, scz), obs0, atol=1e-6)


def test_obs_band_z_pz_slot_equals_pz_difference():
    """Two states differing ONLY in p_z (goal_z held fixed) produce different observations, and the
    difference in the p_z slot (index 32) equals the p_z difference exactly."""
    cfg = _cfg(banded=True); sys = make_system(cfg)
    x1, sc = _q3_scene_state(); x2 = x1.clone(); x2[:, 2] = x1[:, 2] + 2.0   # +2 m altitude only
    o1 = sys.observation(x1, sc); o2 = sys.observation(x2, sc)
    assert not torch.allclose(o1, o2, atol=1e-6)
    assert torch.allclose(o2[:, 32] - o1[:, 32], torch.full((3,), 2.0, dtype=torch.float64), atol=1e-9)
    assert torch.allclose(o2[:, 33], o1[:, 33], atol=1e-9)                 # v_z slot unchanged


# ---- v2.7.6 M8.1: phases=1 single-phase fallback candidate cardinality ----

def test_m8_phases1_candidate_cardinality_is_25():
    """M8.1 pin: phases=1 selects one piecewise-constant control from the grid, so its candidate cardinality
    is the grid size = 25 for quadrotor_3d (16 box corners + center + 8 per-axis extremes; the all-zero
    candidate coincides with the all-lower corner and dedupes). phases=2 searches the 25x25 = 625 grid^2."""
    import torch
    from src.common.kstep_fallback import grid_controls
    cfg = _cfg(banded=True); sys = make_system(cfg)
    G = grid_controls(sys, torch.device("cpu"))
    assert tuple(G.shape) == (25, 4), f"quadrotor_3d fallback grid {tuple(G.shape)} != (25, 4)"
