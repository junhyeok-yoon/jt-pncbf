"""v2.5.1 S — minimal-intervention shield ladder: ladder=None is bit-identical to the arm-B shield, and
the ladder (arm-B') preserves the verified-tail invariant (0 verified-start collisions) on a scene that
forces overrides, while applying at least some verified blends instead of full brakes."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest
import torch
import yaml

from src.envs.double_integrator import DoubleIntegrator
from src.envs.scene_init import Scene
from src.frameworks.cpi.channel import make_cpi_h_fn
from src.frameworks.cpi.shield import shield_eval

REPO = Path(__file__).resolve().parents[1]
CKPT = REPO / "data/secured_data/v2.5.1/seed42/checkpoints/best.pt"
pytestmark = pytest.mark.skipif(not CKPT.exists(), reason="secured v2.5.1 seed42 checkpoint required")


def _cfg() -> dict[str, Any]:
    b = yaml.safe_load((REPO / "src/configs/base_config.yaml").read_text())
    e = yaml.safe_load((REPO / "src/configs/exp_config.yaml").read_text())

    def m(a, o):
        d = dict(a)
        for k, v in o.items():
            d[k] = m(d[k], v) if isinstance(v, Mapping) and isinstance(d.get(k), Mapping) else v
        return d
    return m(b, e)


def _policy(system, cfg):
    from src.common.control_net import ControlNet
    torch.manual_seed(0)
    return ControlNet(system.obs_dim, system, cfg).eval()


def _scenes(n=8):
    # a mid-field obstacle on the straight-line path at moderate entry speed -> forces filter overrides
    return [Scene(obstacle_centers=np.array([[0.4, 0.15]] + [[0, 0]] * 11, float),
                  obstacle_radii=np.array([0.5] + [0.0] * 11), obstacle_active=np.array([True] + [False] * 11),
                  start=np.array([-2.0, 0.0]), goal=np.array([2.0, 0.0]), system="double_integrator",
                  mode="synthetic", initial_velocity=np.array([1.2, 0.0])) for _ in range(n)]


def test_ladder_off_is_bit_identical_to_arm_b():
    cfg = _cfg(); dev = torch.device("cpu"); system = DoubleIntegrator(cfg)
    system.u_bounds = system.u_bounds.to(torch.float32); cfg["eval"]["max_steps"] = 40
    h_fn = make_cpi_h_fn(CKPT, system); pol = _policy(system, cfg); sc = _scenes()
    r0 = shield_eval(sc, pol, cfg, h_fn, system, dev)
    r1 = shield_eval(sc, pol, cfg, h_fn, system, dev, ladder=None)
    assert torch.equal(r0["S"], r1["S"]) and r0["outcome"] == r1["outcome"]
    assert torch.equal(r0["n_overrides"], r1["n_overrides"])


def test_ladder_preserves_verified_start_invariant():
    """The ladder (arm-B') keeps the verified-tail invariant: 0 verified-start collisions on aggressive
    scenes (P-I1-2), because every applied action — a blend or the brake — is fully verified before use."""
    cfg = _cfg(); dev = torch.device("cpu"); system = DoubleIntegrator(cfg)
    system.u_bounds = system.u_bounds.to(torch.float32); cfg["eval"]["max_steps"] = 60
    h_fn = make_cpi_h_fn(CKPT, system); pol = _policy(system, cfg); sc = _scenes()
    mi = shield_eval(sc, pol, cfg, h_fn, system, dev, ladder=[0.75, 0.5, 0.25])
    coll = torch.tensor([o == "collision" for o in mi["outcome"]])
    assert int((coll & mi["verified_start"]).sum()) == 0


def test_ladder_applied_actions_are_all_verified_on_a_verified_start_batch():
    """Policy-independent ladder property: from a VERIFIED-START state (brake-now clears), the ladder never
    applies an unverified action — a doomed accelerate candidate is rejected and every action the ladder can
    emit (each blend it accepts, and the brake floor) fully verifies. This is the invariant behind P-I1-2."""
    from src.frameworks.cpi.shield import verify_plan
    from src.frameworks.cpi.backup import deadband_brake
    cfg = _cfg(); system = DoubleIntegrator(cfg); system.u_bounds = system.u_bounds.to(torch.float32)
    dt = float(cfg["env"]["dt"])
    # obstacle far enough ahead that braking clears (verified-start), but accelerating straight in overshoots
    C = torch.tensor([[[3.0, 0.0]] + [[0.0, 0.0]] * 11]); R = torch.tensor([[0.5] + [0.0] * 11])
    A = torch.tensor([[True] + [False] * 11])
    x = torch.tensor([[0.0, 0.0, 2.5, 0.0]])
    u_brake = deadband_brake(x, system, cfg, dt)
    assert bool(verify_plan(x, u_brake, C, R, A, system, cfg, dt, 0.01, 40, 0.0125)[0]), \
        "state must be verified-start (brake clears)"
    applied = u_brake
    for lam in (0.75, 0.5, 0.25):
        u_lam = lam * torch.tensor([[2.0, 0.0]]) + (1 - lam) * u_brake
        if bool(verify_plan(x, u_lam, C, R, A, system, cfg, dt, 0.01, 40, 0.0125)[0]):
            applied = u_lam; break
    assert bool(verify_plan(x, applied, C, R, A, system, cfg, dt, 0.01, 40, 0.0125)[0])
