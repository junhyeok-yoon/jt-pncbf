"""v2.8.0 M1 — the collision cause channel (obstacle / band_lower / band_upper).

The cause is recorded ALONGSIDE `collided`, which must stay bit-identical to the legacy predicate
`obstacle | (|p_z| >= band_z)`. Priority when several fire at one step, in evaluation order:
obstacle > band_lower > band_upper. Reproduction gate: `test_collided_bit_identical_to_legacy`.
"""
from __future__ import annotations
import copy
from types import SimpleNamespace

import torch

from src.common.outcomes import step_outcomes, resolve_outcome, _collided_exact
from src.frameworks.jt_pncbf.train import load_effective_config, make_system

BAND = 4.0


def _cfg():
    c = copy.deepcopy(load_effective_config())
    c["run"]["system"] = "quadrotor_3d"
    c["env"]["band_collision_limit"] = BAND
    return c


def _sys():
    return make_system(_cfg())


def _scene(obstacle=False):
    if obstacle:
        centers = torch.tensor([[[0.0, 0.0]]], dtype=torch.float64)     # one cylinder at xy origin
        radii = torch.tensor([[0.8]], dtype=torch.float64)
        active = torch.tensor([[True]])
    else:
        centers = torch.tensor([[[50.0, 50.0]]], dtype=torch.float64)   # far away, inactive
        radii = torch.tensor([[0.3]], dtype=torch.float64)
        active = torch.tensor([[False]])
    return SimpleNamespace(goal=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
                           obstacle_centers=centers, obstacle_radii=radii, obstacle_active=active)


def _state(p):
    x = torch.zeros(13, dtype=torch.float64); x[0:3] = torch.tensor(p, dtype=torch.float64); x[3] = 1.0
    return x.reshape(1, 1, 13)


def _cause(x, scene):
    s = _sys(); cfg = _cfg()
    masks = step_outcomes(x, scene, s, cfg)
    res = resolve_outcome(masks)
    return masks, res.outcome[0], res.collision_cause[0]


def test_floor_only_records_band_lower():
    masks, outcome, cause = _cause(_state([0.0, 0.0, -4.5]), _scene(obstacle=False))
    assert bool(masks.collided.any()) and outcome == "collision"
    assert cause == "band_lower", cause


def test_ceiling_records_band_upper():
    masks, outcome, cause = _cause(_state([0.0, 0.0, 4.5]), _scene(obstacle=False))
    assert bool(masks.collided.any()) and outcome == "collision"
    assert cause == "band_upper", cause


def test_obstacle_graze_records_obstacle():
    # xy inside the cylinder, z well within the band -> obstacle only
    masks, outcome, cause = _cause(_state([0.1, 0.0, 0.0]), _scene(obstacle=True))
    assert bool(masks.collided.any()) and outcome == "collision"
    assert cause == "obstacle", cause


def test_simultaneous_obstacle_and_floor_records_obstacle_by_priority():
    # inside the cylinder AND below the floor at the same step -> obstacle wins (evaluation order)
    masks, outcome, cause = _cause(_state([0.1, 0.0, -4.5]), _scene(obstacle=True))
    assert bool(masks.collided.any()) and outcome == "collision"
    assert cause == "obstacle", cause


def test_no_collision_has_empty_cause():
    masks, outcome, cause = _cause(_state([0.0, 0.0, 0.0]), _scene(obstacle=False))
    assert not bool(masks.collided.any())
    assert cause == "", repr(cause)


def test_collided_bit_identical_to_legacy():
    """REPRODUCTION GATE: over a recorded-scale batch of 3-D positions, the new `collided`
    (obstacle | band_lower | band_upper) is bit-identical to the legacy `obstacle | (|p_z| >= band_z)`.
    Includes exact-boundary z = +/- BAND. A mismatch is a HALT."""
    s = _sys(); cfg = _cfg()
    torch.manual_seed(0)
    T, B = 40, 5000
    x = torch.zeros(T, B, 13, dtype=torch.float64)
    x[..., 0:3] = (torch.rand(T, B, 3, dtype=torch.float64) - 0.5) * 12.0   # z spans well past +/-BAND
    x[..., 2] = torch.where(torch.rand(T, B) < 0.02, torch.full((T, B), BAND, dtype=torch.float64), x[..., 2])
    x[..., 2] = torch.where(torch.rand(T, B) < 0.02, torch.full((T, B), -BAND, dtype=torch.float64), x[..., 2])
    x[..., 3] = 1.0
    scene = SimpleNamespace(
        goal=torch.zeros(B, 3, dtype=torch.float64),
        obstacle_centers=(torch.rand(B, 3, 2, dtype=torch.float64) - 0.5) * 6.0,
        obstacle_radii=torch.ones(B, 3, dtype=torch.float64) * 0.5,
        obstacle_active=torch.rand(B, 3) < 0.5,
    )
    positions = s.position(x)
    legacy = _collided_exact(positions, scene) | (torch.abs(positions[..., 2]) >= BAND)
    masks = step_outcomes(x, scene, s, cfg)
    assert torch.equal(masks.collided, legacy), "collided bit changed vs legacy — HALT"
    # and the cause channels union back to collided exactly
    union = masks.collided_obstacle | masks.collided_band_lower | masks.collided_band_upper
    assert torch.equal(union, masks.collided)
    # band_lower and band_upper are disjoint
    assert not bool((masks.collided_band_lower & masks.collided_band_upper).any())
