from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch

import yaml

from src.common.control_net import ControlNet
from src.common.maneuver_value import build_safety_h_fn, maneuver_value
from src.common.value_net import ValueNetEnsemble, make_h_fn
from src.envs.double_integrator import DoubleIntegrator
from src.envs.scene_batch import batch_scenes
from src.envs.scene_init import Scene
from src.frameworks.jt_pncbf.losses import grad_norm, policy_bptt_loss

REPO_ROOT = Path(__file__).resolve().parents[1]
DTYPE = torch.float64
DEV = torch.device("cpu")


def _config() -> dict[str, Any]:
    base = yaml.safe_load((REPO_ROOT / "src/configs/base_config.yaml").read_text())
    exp = yaml.safe_load((REPO_ROOT / "src/configs/exp_config.yaml").read_text())
    return _deep_merge(base, exp)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    m = dict(base)
    for k, v in override.items():
        m[k] = _deep_merge(m[k], v) if isinstance(v, Mapping) and isinstance(m.get(k), Mapping) else v
    return m


def _scene() -> Scene:
    centers = np.zeros((12, 2)); radii = np.zeros(12); active = np.zeros(12, dtype=np.bool_)
    centers[0] = [0.6, 0.0]; radii[0] = 0.2; active[0] = True
    return Scene(obstacle_centers=centers, obstacle_radii=radii, obstacle_active=active,
                 start=np.array([-1.0, 0.0]), goal=np.array([1.4, 0.0]),
                 system="double_integrator", mode="synthetic", initial_velocity=np.array([0.0, 0.0]))


def _batch(B=6):
    states = torch.tensor(
        [[-1.2, -0.4, 0.20, 0.10], [-0.8, -0.2, 0.30, 0.15], [-0.4, 0.0, 0.35, 0.20],
         [0.0, 0.2, 0.30, 0.25], [0.4, 0.4, 0.25, 0.20], [0.8, 0.6, 0.20, 0.15]], dtype=DTYPE)
    return SimpleNamespace(states=states, scene=batch_scenes([_scene()] * B, device=DEV, dtype=DTYPE))


def test_value_mode_parity_collection_and_bptt_hfn() -> None:
    # value mode (default/parity): the shared builder returns make_h_fn(value_net) EXACTLY -> the
    # collection filter and the BPTT filter are bit-identical to pre-change in value mode.
    cfg = _deep_merge(_config(), {"safety_channel": {"type": "value"}})
    torch.manual_seed(0)
    system = DoubleIntegrator(cfg)
    value_net = ValueNetEnsemble(system.obs_dim, cfg).to(dtype=DTYPE)
    b = _batch()
    h_built = build_safety_h_fn(system, cfg, value_net)(b.states, b.scene)
    h_ref = make_h_fn(value_net, system)(b.states, b.scene)
    assert torch.equal(h_built, h_ref)


def test_maneuver_mode_hfn_equals_VM_plus_gamma() -> None:
    cfg = _deep_merge(_config(), {"safety_channel": {"type": "maneuver"}})  # default is value; activate maneuver
    assert cfg["safety_channel"]["type"] == "maneuver"
    torch.manual_seed(0)
    system = DoubleIntegrator(cfg)
    value_net = ValueNetEnsemble(system.obs_dim, cfg).to(dtype=DTYPE)
    b = _batch()
    h_built = build_safety_h_fn(system, cfg, value_net)(b.states, b.scene)
    gm = float(cfg["safety_channel"]["maneuver"]["gamma_m"])
    h_ref = maneuver_value(b.states, b.scene, system, cfg, lateral_js=list(range(1, 9))) + gm
    assert torch.allclose(h_built, h_ref, atol=1.0e-12)


def test_maneuver_policy_bptt_routing_no_value_grad() -> None:
    # maneuver mode: policy grads flow through the projection; the value net is NOT in the graph
    # (no value-net grads); total is finite. detach_filter_coeffs on (Stage-B setting).
    cfg = _deep_merge(_config(), {"safety_channel": {"type": "maneuver"},
                                  "loss": {"policy": {"detach_filter_coeffs": True}}})
    torch.manual_seed(1)
    system = DoubleIntegrator(cfg)
    value_net = ValueNetEnsemble(system.obs_dim, cfg).to(dtype=DTYPE)
    policy_net = ControlNet(system.obs_dim, system, cfg).to(dtype=DTYPE)
    b = _batch()
    policy_net.zero_grad(set_to_none=True); value_net.zero_grad(set_to_none=True)
    res = policy_bptt_loss(system=system, policy_net=policy_net, value_net=value_net, batch=b, config=cfg)
    assert torch.isfinite(res.total)
    res.total.backward()
    assert grad_norm(policy_net.parameters()) > 0.0
    assert all(p.grad is None for p in value_net.parameters())


def _friction_cfg(w):
    return _deep_merge(_config(), {"safety_channel": {"type": "maneuver"},
                                   "loss": {"policy": {"detach_filter_coeffs": True, "w_friction": w}}})


def _batch_active(B=6):
    # states NEAR the obstacle ([0.6,0], r0.2) moving toward it fast: V_M > -1 (near boundary) so the
    # filter ACTIVELY intervenes (u_safe != u_nom) -> friction term is nonzero (unlike far-field states).
    states = torch.tensor(
        [[-0.20, -0.10, 1.20, -0.20], [-0.20, 0.00, 1.20, 0.00], [-0.20, 0.10, 1.20, -0.20],
         [-0.20, -0.10, 1.40, -0.20], [-0.20, -0.10, 1.20, 0.20], [-0.20, 0.00, 1.20, 0.20]],
        dtype=DTYPE)  # V_M in (-0.85, -0.05): near-boundary, filter active
    return SimpleNamespace(states=states, scene=batch_scenes([_scene()] * B, device=DEV, dtype=DTYPE))


def test_friction_w0_parity_and_exact_add() -> None:
    # (a) w_friction=0.0 adds nothing (friction_loss==0); (b) w=0.05 adds EXACTLY w*||u_safe-u_nom||^2:
    # total(0.05) == total(0.0) + friction_loss(0.05) on the same deterministic rollout.
    torch.manual_seed(7)
    system = DoubleIntegrator(_config())
    value_net = ValueNetEnsemble(system.obs_dim, _config()).to(dtype=DTYPE)
    policy_net = ControlNet(system.obs_dim, system, _config()).to(dtype=DTYPE)
    b = _batch_active()
    r0 = policy_bptt_loss(system=system, policy_net=policy_net, value_net=value_net, batch=b, config=_friction_cfg(0.0))
    r1 = policy_bptt_loss(system=system, policy_net=policy_net, value_net=value_net, batch=b, config=_friction_cfg(0.05))
    assert float(r0.friction_loss) == 0.0
    assert float(r1.friction_loss) > 0.0
    assert torch.allclose(r1.total, r0.total + r1.friction_loss, atol=1.0e-6)


def test_friction_routing_policy_grad_no_value_grad() -> None:
    # friction reaches theta_pi (grad differs from w=0) via u_nom AND the u_safe projection path; no
    # value-net grads (value frozen/detached; friction does not touch V_M or filter coeffs).
    torch.manual_seed(8)
    system = DoubleIntegrator(_config())
    value_net = ValueNetEnsemble(system.obs_dim, _config()).to(dtype=DTYPE)
    policy_net = ControlNet(system.obs_dim, system, _config()).to(dtype=DTYPE)
    b = _batch_active()

    def grad_of(w):
        policy_net.zero_grad(set_to_none=True); value_net.zero_grad(set_to_none=True)
        res = policy_bptt_loss(system=system, policy_net=policy_net, value_net=value_net, batch=b, config=_friction_cfg(w))
        res.total.backward()
        g = torch.cat([p.grad.flatten() for p in policy_net.parameters() if p.grad is not None]).clone()
        vg = all(p.grad is None for p in value_net.parameters())
        return g, vg

    g0, vg0 = grad_of(0.0)
    g1, vg1 = grad_of(0.05)
    assert vg0 and vg1                              # no value-net grads either way
    assert not torch.allclose(g0, g1)               # friction changed the policy gradient
    assert torch.isfinite(g1).all()
