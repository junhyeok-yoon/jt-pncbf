from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch

from src.common.control_net import ControlNet
from src.common.filter_hardnet import HardNetFilter
from src.common.value_net import ValueNetEnsemble, make_h_fn
from src.envs.double_integrator import DoubleIntegrator
from src.envs.scene_batch import batch_scenes
from src.envs.scene_init import Scene
from src.frameworks.jt_pncbf.losses import grad_norm, policy_bptt_loss

REPO_ROOT = Path(__file__).resolve().parents[1]
DTYPE = torch.float64


def _scene(center=(1.6, 1.0), radius=0.25):
    c = np.zeros((12, 2)); r = np.zeros(12); a = np.zeros(12, bool)
    c[0] = np.asarray(center); r[0] = radius; a[0] = True
    return Scene(obstacle_centers=c, obstacle_radii=r, obstacle_active=a,
                 start=np.array([-1.2, -0.4]), goal=np.array([1.4, 0.8]),
                 system="double_integrator", mode="synthetic", initial_velocity=np.array([0.15, 0.05]))


def _setup(seed=2026, bptt_t=5):
    config = _deep_merge(_load_config(), {"training": {"jt": {"bptt_T": bptt_t}}})
    torch.manual_seed(seed)
    system = DoubleIntegrator(config)
    value_net = ValueNetEnsemble(system.obs_dim, config).to(dtype=DTYPE)
    policy_net = ControlNet(system.obs_dim, system, config).to(dtype=DTYPE)
    return config, system, value_net, policy_net


def _batch(system, B=6):
    torch.manual_seed(7)
    states = torch.tensor(
        [[-1.2, -0.4, 0.20, 0.10], [-0.8, -0.2, 0.30, 0.15], [-0.4, 0.0, 0.35, 0.20],
         [0.0, 0.2, 0.30, 0.25], [0.4, 0.4, 0.25, 0.20], [0.8, 0.6, 0.20, 0.15]], dtype=DTYPE)
    scene = batch_scenes([_scene()] * B, device=torch.device("cpu"), dtype=DTYPE)
    return SimpleNamespace(states=states, scene=scene)


def _on(config, v):  # config with detach flag set to v
    return _deep_merge(config, {"loss": {"policy": {"detach_filter_coeffs": v}}})


def test_forward_loss_value_identical_on_off() -> None:
    # detach changes only the backward graph, never forward numerics -> loss VALUE identical.
    config, system, value_net, policy_net = _setup()
    batch = _batch(system)
    kw = dict(system=system, policy_net=policy_net, value_net=value_net, batch=batch)
    off = policy_bptt_loss(config=_on(config, False), **kw)
    on = policy_bptt_loss(config=_on(config, True), **kw)
    assert torch.allclose(off.total.detach(), on.total.detach(), atol=0.0, rtol=0.0)


def test_filter_forward_u_safe_identical_on_off() -> None:
    config, system, value_net, policy_net = _setup()
    for p in value_net.parameters():
        p.requires_grad_(False)
    hardnet = HardNetFilter(system, make_h_fn(value_net, system), config)
    scene = batch_scenes([_scene()], device=torch.device("cpu"), dtype=DTYPE)
    x = torch.tensor([[-0.4, 0.0, 0.35, 0.20]], dtype=DTYPE)
    u_nom = policy_net(system.observation(x, scene))
    u_off, _ = hardnet(x, scene, u_nom, detach_coeffs=False)
    u_on, _ = hardnet(x, scene, u_nom, detach_coeffs=True)
    assert torch.allclose(u_off.detach(), u_on.detach(), atol=0.0, rtol=0.0)


def _analytic_h(x, scene):
    # Velocity-dependent CBF (=> nonzero L_g h = dh/dv), positive when near or approaching an
    # obstacle, so the projection is exercised in a controlled way (a random value net gives h<=0
    # everywhere at init, never activating the filter).
    p, v = x[:, :2], x[:, 2:4]
    centers = torch.as_tensor(scene.obstacle_centers, dtype=x.dtype, device=x.device)
    radii = torch.as_tensor(scene.obstacle_radii, dtype=x.dtype, device=x.device)
    active = torch.as_tensor(scene.obstacle_active, dtype=torch.bool, device=x.device)
    rel = p.unsqueeze(-2) - centers                                # [B,n,2] obstacle->p
    dist = torch.linalg.norm(rel, dim=-1).clamp_min(1e-6)
    clearance = dist - radii
    inward = torch.sum(v.unsqueeze(-2) * (-rel / dist.unsqueeze(-1)), dim=-1)   # inward speed [B,n]
    h_all = (0.4 - clearance) + 0.3 * inward
    h_all = torch.where(active, h_all, torch.full_like(h_all, -1e9))
    return torch.clamp(h_all.max(dim=-1).values, -1.0, 1.0)


def test_coefficient_path_gradient_cut_when_on() -> None:
    # u_nom is a FIXED leaf, so x reaches u_safe only via the CBF coefficients. detach on =>
    # u_safe is CONSTANT w.r.t. x (no grad_fn, coeff path fully cut); detach off, at an ACTIVE
    # (non-corner) state => u_safe depends on x with a nonzero gradient.
    config, system, _, _ = _setup()
    hardnet = HardNetFilter(system, _analytic_h, config)
    scene = batch_scenes([_scene()], device=torch.device("cpu"), dtype=DTYPE)
    x0 = torch.tensor([[0.30, 0.40, 0.908, 0.419]], dtype=DTYPE)    # mildly active (h<0, non-corner)
    u_nom = torch.zeros(1, system.action_dim, dtype=DTYPE)

    x_on = x0.clone().requires_grad_(True)
    u_on, _ = hardnet(x_on, scene, u_nom, detach_coeffs=True)
    assert not u_on.requires_grad                                  # coeff path fully cut

    x_off = x0.clone().requires_grad_(True)
    u_off, _ = hardnet(x_off, scene, u_nom, detach_coeffs=False)
    assert u_off.requires_grad and not torch.allclose(u_off.detach(), u_nom)   # active, coeff path present
    g_off = torch.autograd.grad(u_off.sum(), x_off)[0]
    assert float(g_off.abs().max()) > 1.0e-8


def test_detach_changes_bptt_state_chain_gradient() -> None:
    # The detach cuts u_safe's coefficient x-path -> it must change the BPTT STATE-chain gradient
    # d(loss)/dx0 (the amplification path that explodes at T=60), while the flag-off value is
    # deterministic/unchanged.
    config, system, _, policy_net = _setup()
    from src.common.rk4 import rk4_step
    hardnet = HardNetFilter(system, _analytic_h, config)
    scene = batch_scenes([_scene()] * 4, device=torch.device("cpu"), dtype=DTYPE)
    x0 = torch.tensor([[0.30, 0.40, 0.91, 0.42], [0.40, 0.50, 0.87, 0.48],
                       [0.50, 0.40, 0.90, 0.38], [0.35, 0.55, 0.85, 0.50]], dtype=DTYPE)
    goal = torch.as_tensor(scene.goal, dtype=DTYPE)
    dt = float(config["env"]["dt"])

    def state_grad(detach):
        # fix u_nom (detach the policy) so the only x-influence on u_safe is via the CBF
        # coefficients -> isolates the coefficient x-path (the BPTT state-chain gradient).
        x = x0.clone().requires_grad_(True)
        xx = x
        loss = torch.zeros((), dtype=DTYPE)
        for _ in range(5):
            u_nom = policy_net(system.observation(xx, scene)).detach()
            u_safe, _ = hardnet(xx, scene, u_nom, detach_coeffs=detach)
            xx = rk4_step(system, xx, u_safe, dt)
            loss = loss + torch.sum((system.position(xx) - goal) ** 2)
        return torch.autograd.grad(loss, x)[0].clone()

    g_off1 = state_grad(False)
    g_off2 = state_grad(False)
    g_on = state_grad(True)
    assert torch.allclose(g_off1, g_off2, atol=0.0, rtol=0.0)       # flag-off deterministic/unchanged
    assert not torch.allclose(g_off1, g_on)                        # detach changes the state-chain gradient


def test_routing_holds_with_flag_on() -> None:
    config, system, value_net, policy_net = _setup()
    batch = _batch(system)
    value_net.zero_grad(set_to_none=True); policy_net.zero_grad(set_to_none=True)
    policy_bptt_loss(system=system, policy_net=policy_net, value_net=value_net,
                     batch=batch, config=_on(config, True)).total.backward()
    assert grad_norm(value_net.parameters()) < 1.0e-12
    assert grad_norm(policy_net.parameters()) > 0.0


def _load_config() -> dict[str, Any]:
    import yaml
    base = yaml.safe_load((REPO_ROOT / "src/configs/base_config.yaml").read_text())
    exp = yaml.safe_load((REPO_ROOT / "src/configs/exp_config.yaml").read_text())
    return _deep_merge(base, exp)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
