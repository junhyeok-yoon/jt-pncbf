"""v2.5.1 A2(b) horizon-summary critic W: flag-off byte-identical parity, and critic ISOLATION —
(1) grep-level: W never appears in the certificate channel / shield / family / filter modules;
(2) gradient routing: dL_pi/d(theta_W)=0 (W frozen in the policy loss) yet dL_pi/d(theta_pi)!=0 THROUGH the
    W(x_T) tail (removing W changes the policy gradient); and the W-regression step routes NO gradient to
    theta_pi (grad-free rollout, pred depends on W only)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from src.common.control_net import ControlNet
from src.common.value_net import ValueNetEnsemble
from src.envs.double_integrator import DoubleIntegrator
from src.envs.scene_init import Scene
from src.frameworks.cpi.value import CPIValue
from src.frameworks.jt_pncbf.collection import make_replay_buffers
from src.frameworks.jt_pncbf.losses import grad_norm, policy_bptt_loss
from src.frameworks.jt_pncbf.train import _critic_updates, _polyak_update
from src.frameworks.oc_pncbf.collection import OCReplayBuffer

REPO_ROOT = Path(__file__).resolve().parents[1]
DTYPE = torch.float64


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config() -> dict[str, Any]:
    base = yaml.safe_load((REPO_ROOT / "src/configs/base_config.yaml").read_text())
    exp = yaml.safe_load((REPO_ROOT / "src/configs/exp_config.yaml").read_text())
    cfg = _deep_merge(base, exp)
    cfg["training"]["jt"]["bptt_T"] = 3
    return cfg


def _tiny_scene() -> Scene:
    centers = np.zeros((12, 2), dtype=np.float64)
    radii = np.zeros(12, dtype=np.float64)
    active = np.zeros(12, dtype=np.bool_)
    centers[0] = np.array([1.6, 1.0], dtype=np.float64)
    radii[0] = 0.25
    active[0] = True
    return Scene(
        obstacle_centers=centers, obstacle_radii=radii, obstacle_active=active,
        start=np.array([-1.0, -0.4], dtype=np.float64), goal=np.array([1.4, 0.8], dtype=np.float64),
        system="double_integrator", mode="synthetic",
        initial_velocity=np.array([0.10, 0.05], dtype=np.float64),
    )


def _fixture(config):
    system = DoubleIntegrator(config)
    value_net = ValueNetEnsemble(system.obs_dim, config).to(dtype=DTYPE)
    policy_net = ControlNet(system.obs_dim, system, config).to(dtype=DTYPE)
    buffer = OCReplayBuffer(capacity=8)
    states = torch.tensor(
        [[-1.0, -0.4, 0.10, 0.05], [-0.9, -0.35, 0.10, 0.05],
         [-0.8, -0.30, 0.10, 0.05], [-0.7, -0.25, 0.10, 0.05]], dtype=DTYPE)
    buffer.append(_tiny_scene(), states, torch.tensor([-0.4, -0.3, -0.2, -0.1], dtype=DTYPE))
    buffers = make_replay_buffers(capacity=8)
    buffers.policy = buffer
    return system, value_net, policy_net, buffers


def test_horizon_critic_flag_off_is_byte_identical():
    """The horizon_critic config block + the critic_net=None path leave L_pi (value AND gradients) bitwise
    identical to a config with no horizon_critic block at all."""
    cfg_on_block = _load_config()                         # has the horizon_critic block (enabled=false)
    cfg_no_block = _load_config()
    del cfg_no_block["training"]["jt"]["horizon_critic"]  # a config predating the feature

    def run(cfg):
        torch.manual_seed(1234)
        system, value_net, policy_net, buffers = _fixture(cfg)
        batch = buffers.policy.sample_tensor_batch(batch_size=3)
        value_net.zero_grad(set_to_none=True); policy_net.zero_grad(set_to_none=True)
        out = policy_bptt_loss(system=system, policy_net=policy_net, value_net=value_net,
                               batch=batch, config=cfg, critic_net=None)
        out.total.backward()
        g = torch.cat([p.grad.reshape(-1) for p in policy_net.parameters()])
        return float(out.total.detach()), g

    total_a, grad_a = run(cfg_no_block)
    total_b, grad_b = run(cfg_on_block)
    assert total_a == total_b, "flag-off total must be byte-identical to the pre-feature baseline"
    assert torch.equal(grad_a, grad_b), "flag-off policy gradient must be byte-identical"


def test_critic_absent_from_safety_channel_shield_family_filter():
    """grep-level isolation: the deployment/certification modules never reference the critic."""
    forbidden = ("critic", "horizon_critic", "_critic_updates")
    for rel in ("src/frameworks/cpi/channel.py", "src/frameworks/cpi/shield.py",
                "src/frameworks/cpi/family.py", "src/common/filter_hardnet.py"):
        text = (REPO_ROOT / rel).read_text().lower()
        for token in forbidden:
            assert token not in text, f"{token!r} must not appear in {rel}"


def test_policy_loss_freezes_W_but_routes_through_x_T():
    """dL_pi/d(theta_W)=0 (W frozen), dL_pi/d(theta_pi)!=0, and the W tail actually changes the policy
    gradient vs critic-off (proving the pathwise route through x_T is live)."""
    cfg = _load_config()
    cfg["training"]["jt"]["horizon_critic"]["enabled"] = True
    torch.manual_seed(1234)
    system, value_net, policy_net, buffers = _fixture(cfg)
    critic = CPIValue(obs_dim=system.obs_dim).to(dtype=DTYPE)
    batch = buffers.policy.sample_tensor_batch(batch_size=3)

    # critic-ON: W frozen -> no grad on theta_W; policy grad nonzero.
    value_net.zero_grad(set_to_none=True); policy_net.zero_grad(set_to_none=True); critic.zero_grad(set_to_none=True)
    out_on = policy_bptt_loss(system=system, policy_net=policy_net, value_net=value_net,
                              batch=batch, config=cfg, critic_net=critic)
    out_on.total.backward()
    assert grad_norm(critic.parameters()) == 0.0, "dL_pi/d(theta_W) must be exactly 0 (W is stop-grad)"
    assert grad_norm(policy_net.parameters()) > 0.0, "dL_pi/d(theta_pi) must be nonzero"
    grad_on = torch.cat([p.grad.reshape(-1) for p in policy_net.parameters()]).clone()

    # critic-OFF on the same inputs: the policy gradient must differ (the W(x_T) tail contributes).
    policy_net.zero_grad(set_to_none=True); value_net.zero_grad(set_to_none=True)
    out_off = policy_bptt_loss(system=system, policy_net=policy_net, value_net=value_net,
                               batch=batch, config=cfg, critic_net=None)
    out_off.total.backward()
    grad_off = torch.cat([p.grad.reshape(-1) for p in policy_net.parameters()])
    assert not torch.equal(grad_on, grad_off), "the W(x_T) tail must change the policy gradient"
    assert float(out_on.total.detach()) != float(out_off.total.detach())


def test_W_regression_routes_no_gradient_to_policy():
    """dL_W/d(theta_pi)=0: one W-step (grad-free rollout, pred depends on W only) leaves theta_pi grads
    untouched while producing a finite loss and updating W."""
    cfg = _load_config()
    cfg["training"]["jt"]["horizon_critic"]["enabled"] = True
    cfg["training"]["jt"]["horizon_critic"]["n_step"] = 5   # keep the CPU rollout short
    torch.manual_seed(1234)
    system, value_net, policy_net, buffers = _fixture(cfg)
    critic = CPIValue(obs_dim=system.obs_dim).to(dtype=DTYPE)
    critic_target = CPIValue(obs_dim=system.obs_dim).to(dtype=DTYPE)
    critic_target.load_state_dict(critic.state_dict()); critic_target.requires_grad_(False)
    opt_w = torch.optim.AdamW(critic.parameters(), lr=1.0e-3)
    w_before = torch.cat([p.detach().reshape(-1) for p in critic.parameters()]).clone()

    policy_net.zero_grad(set_to_none=True)
    scalars = _critic_updates(system=system, critic_net=critic, critic_target=critic_target, optimizer=opt_w,
                              value_net=value_net, policy_net=policy_net, buffers=buffers,
                              torch_generator=None, batch_size=3, config=cfg, log=True)
    assert scalars["critic_finite"]
    assert np.isfinite(scalars["L_W"])
    # no policy gradient produced by the W-step
    assert all(p.grad is None or float(p.grad.abs().sum()) == 0.0 for p in policy_net.parameters())
    # W actually moved
    w_after = torch.cat([p.detach().reshape(-1) for p in critic.parameters()])
    assert not torch.equal(w_before, w_after)
