from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch

import src.common.filter_hardnet as fh
from src.common.filter_hardnet import HardNetFilter
from src.common.value_net import ValueNetEnsemble
from src.envs.double_integrator import DoubleIntegrator
from src.envs.scene_batch import batch_scenes
from src.envs.scene_init import Scene
from src.frameworks.jt_pncbf.losses import grad_norm, policy_bptt_loss
from src.frameworks.jt_pncbf.train import JTPNCBFFramework, _build_control_net

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DTYPE = torch.float64
DEV = torch.device("cpu")


def _scene():
    c = np.zeros((12, 2)); r = np.zeros(12); a = np.zeros(12, bool)
    c[0] = np.array([1.4, 0.9]); r[0] = 0.25; a[0] = True
    return Scene(obstacle_centers=c, obstacle_radii=r, obstacle_active=a,
                 start=np.array([-1.0, -0.3]), goal=np.array([1.3, 0.8]),
                 system="double_integrator", mode="synthetic", initial_velocity=np.array([0.3, 0.15]))


def _config() -> dict[str, Any]:
    import yaml
    base = yaml.safe_load((REPO / "src/configs/base_config.yaml").read_text())
    exp = yaml.safe_load((REPO / "src/configs/exp_config.yaml").read_text())
    return _merge(base, exp)


def _merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _cfg(**policy):
    c = _config()
    c["loss"]["policy"] = {**c["loss"]["policy"], **policy}
    return c


def _setup(config, seed=7):
    torch.manual_seed(seed)
    system = DoubleIntegrator(config)
    value_net = ValueNetEnsemble(system.obs_dim, config).to(dtype=DTYPE)
    policy_net = _build_control_net(system, config).to(dtype=DTYPE)
    return system, value_net, policy_net


def _batch(B=6):
    states = torch.tensor(
        [[-1.0, -0.3, 0.30, 0.15], [-0.5, -0.1, 0.40, 0.20], [0.0, 0.1, 0.45, 0.25],
         [0.4, 0.3, 0.40, 0.30], [0.8, 0.5, 0.35, 0.25], [1.0, 0.6, 0.30, 0.20]], dtype=DTYPE)
    return SimpleNamespace(states=states, scene=batch_scenes([_scene()] * B, device=DEV, dtype=DTYPE))


def test_flag_off_parity_and_aux_not_invoked(monkeypatch) -> None:
    # Flag off (obs_deficit_feedback false, w_deficit 0): dim-19 policy, no aux path invoked, and the
    # loss/grad match a config with the key entirely absent.
    cfg_off = _cfg(obs_deficit_feedback=False, w_deficit=0.0)
    system, value_net, policy_net = _setup(cfg_off)
    assert policy_net.trunk[0].in_features == system.obs_dim              # dim-19, no augmentation
    batch = _batch()
    orig = HardNetFilter.__call__

    def guard(self, x, scene, u_nom, detach_coeffs=False, return_deficit_aux=False):
        assert not return_deficit_aux, "aux path invoked with obs_deficit off and w_deficit 0"
        return orig(self, x, scene, u_nom, detach_coeffs=detach_coeffs)

    monkeypatch.setattr(fh.HardNetFilter, "__call__", guard)

    def loss_grad(cfg):
        policy_net.zero_grad(set_to_none=True)
        res = policy_bptt_loss(system=system, policy_net=policy_net, value_net=value_net,
                               batch=batch, config=cfg)
        res.total.backward()
        g = torch.cat([p.grad.flatten() for p in policy_net.parameters() if p.grad is not None]).clone()
        return res.total.detach().clone(), g

    cfg_absent = _config()
    cfg_absent["loss"]["policy"] = {k: v for k, v in cfg_absent["loss"]["policy"].items()
                                    if k not in ("obs_deficit_feedback",)}
    cfg_absent["loss"]["policy"]["w_deficit"] = 0.0
    l0, g0 = loss_grad(cfg_off)
    l1, g1 = loss_grad(cfg_absent)
    assert torch.allclose(l0, l1, atol=0.0, rtol=0.0)
    assert torch.allclose(g0, g1, atol=0.0, rtol=0.0)


def test_zero_init_continuity() -> None:
    # Flag on at init: the two deficit input columns are zero, so the policy output is INVARIANT to the
    # deficit feature (feeding any delta_u equals feeding 0).
    cfg = _cfg(obs_deficit_feedback=True)
    system, _, policy_net = _setup(cfg)
    assert policy_net.trunk[0].in_features == system.obs_dim + system.action_dim
    obs = torch.randn(5, system.obs_dim, dtype=DTYPE)
    d1 = torch.randn(5, system.action_dim, dtype=DTYPE)
    d2 = torch.zeros(5, system.action_dim, dtype=DTYPE)
    out1 = policy_net(torch.cat([obs, d1], dim=1))
    out2 = policy_net(torch.cat([obs, d2], dim=1))
    assert torch.allclose(out1, out2, atol=1e-9)                          # deficit-invariant at init


def test_feature_sensitivity_and_gradient() -> None:
    # With nonzero deficit-column weights the policy output DEPENDS on the deficit feature, and those
    # weights receive gradient (the policy can learn to use the feature).
    cfg = _cfg(obs_deficit_feedback=True)
    system, _, policy_net = _setup(cfg)
    with torch.no_grad():
        policy_net.trunk[0].weight[:, system.obs_dim:] = 0.5
    obs = torch.randn(5, system.obs_dim, dtype=DTYPE)
    d1 = torch.randn(5, system.action_dim, dtype=DTYPE)
    with torch.no_grad():
        out0 = policy_net(torch.cat([obs, torch.zeros_like(d1)], dim=1))
        out1 = policy_net(torch.cat([obs, d1], dim=1))
    assert float(torch.linalg.norm(out1 - out0)) > 1e-6                   # forward sensitive to feature
    policy_net.zero_grad(set_to_none=True)
    policy_net(torch.cat([obs, d1], dim=1)).sum().backward()
    assert float(torch.linalg.norm(policy_net.trunk[0].weight.grad[:, system.obs_dim:])) > 1e-6


def test_gradient_isolation_no_leak_through_feature() -> None:
    # With obs_deficit on and w_deficit 0, the policy loss trains the policy (grad>0) with NO gradient
    # to V_S params. delta_u depends on V_S (through h); feeding it UNDETACHED would leak grad to V_S
    # via the coefficient Jacobian, so grad_leak==0 verifies the feature is detached (the Exp 2 point).
    cfg = _cfg(obs_deficit_feedback=True, w_deficit=0.0)
    system, value_net, policy_net = _setup(cfg)
    with torch.no_grad():
        policy_net.trunk[0].weight[:, system.obs_dim:] = 0.3                # make the feature live
    batch = _batch()
    value_net.zero_grad(set_to_none=True); policy_net.zero_grad(set_to_none=True)
    policy_bptt_loss(system=system, policy_net=policy_net, value_net=value_net,
                     batch=batch, config=cfg).total.backward()
    assert grad_norm(value_net.parameters()) < 1.0e-12                     # no leak through the feature
    assert grad_norm(policy_net.parameters()) > 0.0


def test_collection_bptt_deficit_agreement() -> None:
    # The delta_u_{t-1} the deployed path (framework.filter) stores equals the raw box-free deficit
    # (u_cbf_only - u_safe) that the BPTT rollout feeds, for the same state/action.
    cfg = _cfg(obs_deficit_feedback=True)
    system, value_net, policy_net = _setup(cfg)
    fw = JTPNCBFFramework(system, value_net, policy_net, cfg)
    scene = batch_scenes([_scene()] * 3, device=DEV, dtype=DTYPE)
    x0 = torch.tensor([[-1.0, -0.3, 0.3, 0.15], [0.6, 0.4, 0.5, 0.3], [1.0, 0.6, 0.4, 0.2]], dtype=DTYPE)
    u_nom = fw.policy(x0, scene)                                           # step-0 uses zero deficit
    fw.filter(x0, u_nom, scene)                                           # stores _prev_deficit
    # recompute the raw deficit directly from the same filter
    u_safe, _, u_cbf_only, _ = fw._filter(x0, scene, u_nom, return_deficit_aux=True)
    assert torch.allclose(fw._prev_deficit, (u_cbf_only - u_safe).detach(), atol=1e-12)
    assert fw._prev_deficit.requires_grad is False                        # detached feature
