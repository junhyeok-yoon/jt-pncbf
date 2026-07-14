"""S' shield: verify_plan correctness (clear -> pass, doomed -> fail), collision-free among verified-start
episodes, zero overrides on a clear scene, and determinism (same scene twice -> identical outcomes)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pytest
import torch
import yaml

from src.envs.double_integrator import DoubleIntegrator
from src.envs.scene_init import Scene
from src.frameworks.cpi.channel import make_cpi_h_fn
from src.frameworks.cpi.shield import shield_eval, verify_plan, _deadband

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


def _obs(centers, radii, n=12):
    C = np.zeros((n, 2)); Rr = np.zeros(n); A = np.zeros(n, bool)
    for i, (c, r) in enumerate(zip(centers, radii)):
        C[i] = c; Rr[i] = r; A[i] = True
    return C, Rr, A


def test_verify_plan_clear_and_doomed():
    cfg = _cfg(); system = DoubleIntegrator(cfg); system.u_bounds = system.u_bounds.to(torch.float32)
    C, R, A = _obs([(2.0, 0.0)], [0.5])
    Ct = torch.as_tensor(C, dtype=torch.float32)[None]; Rt = torch.as_tensor(R, dtype=torch.float32)[None]
    At = torch.as_tensor(A)[None]
    # clear: at (-3,0) moving away -> pass
    x_clear = torch.tensor([[-3.0, 0.0, -1.0, 0.0]])
    ok = verify_plan(x_clear, _deadband(x_clear[:, 2:4], 2.0, 0.05), Ct, Rt, At, system, 0.05, 0.01, 40, 0.0125, 2.0, 2.5)
    assert bool(ok[0])
    # doomed: just outside the obstacle at high speed straight into it -> brake cannot clear -> fail
    x_doom = torch.tensor([[1.4, 0.0, 2.5, 0.0]])
    ok = verify_plan(x_doom, torch.tensor([[2.0, 0.0]]), Ct, Rt, At, system, 0.05, 0.01, 40, 0.0125, 2.0, 2.5)
    assert not bool(ok[0])


def _policy(system, cfg):
    from src.common.control_net import ControlNet
    torch.manual_seed(0)
    return ControlNet(system.obs_dim, system, cfg).eval()


def test_shield_clear_scene_zero_overrides_and_verified_start_collision_free():
    cfg = _cfg(); dev = torch.device("cpu"); system = DoubleIntegrator(cfg); system.u_bounds = system.u_bounds.to(torch.float32)
    cfg["eval"]["max_steps"] = 40
    h_fn = make_cpi_h_fn(CKPT, system); pol = _policy(system, cfg)
    # clear scenes: single small far obstacle, start heading to goal in open space
    scenes = [Scene(obstacle_centers=np.array([[3.5, 3.5]] + [[0, 0]] * 11, float),
                    obstacle_radii=np.array([0.2] + [0.0] * 11), obstacle_active=np.array([True] + [False] * 11),
                    start=np.array([-2.0, 0.0]), goal=np.array([2.0, 0.0]), system="double_integrator",
                    mode="synthetic", initial_velocity=np.array([0.0, 0.0])) for _ in range(4)]
    r = shield_eval(scenes, pol, cfg, h_fn, system, dev)
    assert int(r["n_overrides"].sum()) == 0
    coll = torch.tensor([o == "collision" for o in r["outcome"]])
    assert int((coll & r["verified_start"]).sum()) == 0     # invariant: no verified-start collision


def test_shield_determinism():
    cfg = _cfg(); dev = torch.device("cpu"); system = DoubleIntegrator(cfg); system.u_bounds = system.u_bounds.to(torch.float32)
    cfg["eval"]["max_steps"] = 30
    h_fn = make_cpi_h_fn(CKPT, system); pol = _policy(system, cfg)
    sc = [Scene(obstacle_centers=np.array([[0.5, 0.3]] + [[0, 0]] * 11, float),
                obstacle_radii=np.array([0.4] + [0.0] * 11), obstacle_active=np.array([True] + [False] * 11),
                start=np.array([-2.0, 0.0]), goal=np.array([2.0, 0.0]), system="double_integrator",
                mode="synthetic", initial_velocity=np.array([1.0, 0.0]))]
    r1 = shield_eval(sc, pol, cfg, h_fn, system, dev); r2 = shield_eval(sc, pol, cfg, h_fn, system, dev)
    assert torch.equal(r1["S"], r2["S"]) and r1["outcome"] == r2["outcome"]
    assert torch.equal(r1["n_overrides"], r2["n_overrides"])
