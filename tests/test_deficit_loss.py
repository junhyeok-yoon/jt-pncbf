from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch

import src.common.filter_hardnet as fh
from src.common.control_net import ControlNet
from src.common.filter_hardnet import (
    HardNetFilter, _base_alpha, _base_projection, _box_aware_projection, _cbf_terms,
    _hardnet_params, _SINGULAR_LG_THRESHOLD,
)
from src.common.rk4 import rk4_step
from src.common.value_net import ValueNetEnsemble, make_h_fn
from src.envs.double_integrator import DoubleIntegrator
from src.envs.scene_batch import batch_scenes
from src.envs.scene_init import Scene
from src.frameworks.jt_pncbf.losses import grad_norm, policy_bptt_loss

REPO_ROOT = Path(__file__).resolve().parents[1]
DTYPE = torch.float64
DEV = torch.device("cpu")


def _scene(center=(1.6, 1.0), radius=0.25):
    c = np.zeros((12, 2)); r = np.zeros(12); a = np.zeros(12, bool)
    c[0] = np.asarray(center); r[0] = radius; a[0] = True
    return Scene(obstacle_centers=c, obstacle_radii=r, obstacle_active=a,
                 start=np.array([-1.2, -0.4]), goal=np.array([1.4, 0.8]),
                 system="double_integrator", mode="synthetic", initial_velocity=np.array([0.1, 0.05]))


def _vel_h(x, scene):
    # velocity-dependent CBF (nonzero L_g h), positive near/approaching an obstacle.
    p, v = x[:, :2], x[:, 2:4]
    c = torch.as_tensor(scene.obstacle_centers, dtype=x.dtype); r = torch.as_tensor(scene.obstacle_radii, dtype=x.dtype)
    act = torch.as_tensor(scene.obstacle_active, dtype=torch.bool)
    rel = p.unsqueeze(-2) - c; dist = torch.linalg.norm(rel, dim=-1).clamp_min(1e-6)
    inward = torch.sum(v.unsqueeze(-2) * (-rel / dist.unsqueeze(-1)), dim=-1)
    h = (0.4 - (dist - r)) + 0.3 * inward
    h = torch.where(act, h, torch.full_like(h, -1e9))
    return torch.clamp(h.max(dim=-1).values, -1.0, 1.0)


def _pos_h(x, scene):
    # position-only CBF => L_g h = dh/dv = 0 => SINGULAR (||L_g h|| = 0).
    p = x[:, :2]
    c = torch.as_tensor(scene.obstacle_centers, dtype=x.dtype); r = torch.as_tensor(scene.obstacle_radii, dtype=x.dtype)
    act = torch.as_tensor(scene.obstacle_active, dtype=torch.bool)
    dist = torch.linalg.norm(p.unsqueeze(-2) - c, dim=-1).clamp_min(1e-6)
    h = torch.where(act, 0.4 - (dist - r), torch.full_like(dist, -1e9))
    return torch.clamp(h.max(dim=-1).values, -1.0, 1.0)


def _analytic_cbf_only_raw(u_nom, a, b, params):
    # RAW box-free half-space projection (NO box clamp) = "the control the CBF asked for".
    lhs = torch.sum(a * u_nom, dim=1)
    viol = torch.relu(lhs - b)
    denom = torch.sum(a * a, dim=1) + params.epsilon ** 2 + params.lg_reg_eps
    return u_nom - a * (viol / denom).unsqueeze(1)


def _setup(seed=2026):
    config = _config()
    torch.manual_seed(seed)
    system = DoubleIntegrator(config)
    value_net = ValueNetEnsemble(system.obs_dim, config).to(dtype=DTYPE)
    policy_net = ControlNet(system.obs_dim, system, config).to(dtype=DTYPE)
    return config, system, value_net, policy_net


def _batch(system, B=6):
    states = torch.tensor(
        [[-1.2, -0.4, 0.20, 0.10], [-0.8, -0.2, 0.30, 0.15], [-0.4, 0.0, 0.35, 0.20],
         [0.0, 0.2, 0.30, 0.25], [0.4, 0.4, 0.25, 0.20], [0.8, 0.6, 0.20, 0.15]], dtype=DTYPE)
    return SimpleNamespace(states=states, scene=batch_scenes([_scene()] * B, device=DEV, dtype=DTYPE))


def test_w0_parity_and_aux_not_invoked(monkeypatch) -> None:
    config, system, value_net, policy_net = _setup()
    batch = _batch(system)
    orig = HardNetFilter.__call__

    def guard(self, x, scene, u_nom, detach_coeffs=False, return_deficit_aux=False):
        assert not return_deficit_aux, "aux path invoked with w_deficit == 0"
        return orig(self, x, scene, u_nom, detach_coeffs=detach_coeffs)

    monkeypatch.setattr(fh.HardNetFilter, "__call__", guard)
    kw = dict(system=system, policy_net=policy_net, value_net=value_net, batch=batch)

    def loss_and_grad(cfg):
        policy_net.zero_grad(set_to_none=True)
        res = policy_bptt_loss(config=cfg, **kw)
        res.total.backward()
        g = torch.cat([p.grad.flatten() for p in policy_net.parameters() if p.grad is not None]).clone()
        return res.total.detach().clone(), g

    l0, g0 = loss_and_grad(_merge(config, {"loss": {"policy": {"w_deficit": 0.0}}}))
    cfg_absent = _merge(config, {})
    cfg_absent["loss"]["policy"].pop("w_deficit", None)
    l1, g1 = loss_and_grad(cfg_absent)
    assert torch.allclose(l0, l1, atol=0.0, rtol=0.0)
    assert torch.allclose(g0, g1, atol=0.0, rtol=0.0)


def test_filter_aux_analytic_cases() -> None:
    config, system, value_net, _ = _setup()
    # v2.8.0 S2 B7: PIN RETAINED (projection=enumerate). Cases (B) and (C-moderate) assert that the box-aware
    # SELECTION differs from the raw box-free CBF ask (delta_u>0) on fixed states chosen so the ε-regularized
    # base-projection candidate exits the box. The exact dual_solve keeps the raw ask INSIDE the box on those
    # same states (e.g. C-moderate: raw = [-1.377,-0.787] within [-2,2]^2 -> delta_u ~ 4e-6), so the expectation
    # cannot be recomputed for the exact projection without redesigning the test states. This case validates the
    # retained enumeration realization; the exact default path is covered by test_dual_projection.py.
    config["filter"]["projection"] = "enumerate"
    for p in value_net.parameters():
        p.requires_grad_(False)
    params = _hardnet_params(config)
    hardnet = HardNetFilter(system, _vel_h, config)
    scene1 = batch_scenes([_scene()], device=DEV, dtype=DTYPE)

    def aux(x, u_nom, h_fn=_vel_h):
        hn = HardNetFilter(system, h_fn, config)
        u_safe, _, u_cbf, sing = hn(x, scene1, u_nom, return_deficit_aux=True)
        h, lf, lg = _cbf_terms(system, h_fn, x, scene1, u_nom, create_graph=False)
        b = -lf - _base_alpha(h, params) * h
        return u_safe.detach(), u_cbf.detach(), sing, lg.detach(), b.detach()

    bounds = system.u_bounds.to(DTYPE)
    cap = float(config["loss"]["policy"]["deficit_cap"])

    # (A) non-binding: filter inactive => u_cbf_raw == u_nom and u_safe == u_nom => delta_u == 0.
    xA = torch.tensor([[-1.0, -0.6, 0.0, 0.0]], dtype=DTYPE)          # far, safe
    uA = torch.tensor([[0.3, -0.2]], dtype=DTYPE)
    u_safe, u_cbf, sing, lg, b = aux(xA, uA)
    assert torch.allclose(u_cbf, _analytic_cbf_only_raw(uA, lg, b, params), atol=1e-9)
    assert torch.allclose(u_cbf, uA, atol=1e-9)                       # inactive (viol = 0)
    assert torch.allclose(u_cbf - u_safe, torch.zeros_like(uA), atol=1e-9)   # delta_u = 0

    # (B) constraint binds but the RAW box-free projection lands inside the box (weak binding):
    # box-aware coincides with it => delta_u == 0. Hand-built marginal half-space (direct math).
    a = torch.tensor([[1.0, 0.0]], dtype=DTYPE); bb = torch.tensor([0.2], dtype=DTYPE)
    u = torch.tensor([[0.2001, 0.5]], dtype=DTYPE)                    # violates by ~1e-4 (base stays feasible)
    u_raw = _analytic_cbf_only_raw(u, a, bb, params)                  # ~[0.2, 0.5], inside box
    u_box, _ = _box_aware_projection(u, _base_projection(u, a, bb, bounds, params), a, bb, bounds)
    assert bool((u_raw.abs() < 1.99).all())                          # raw projection inside box
    assert not torch.allclose(u_raw, u, atol=1e-6)                   # constraint binds
    assert torch.allclose(u_raw - u_box, torch.zeros_like(u), atol=1e-6)   # delta_u = 0

    # (C-moderate) box ∩ half-space feasible but the RAW projection exits the box (h<0 => alpha=2):
    # ||delta_u|| > 0 and u_cbf_raw matches the analytic RAW base projection.
    xCm = torch.tensor([[0.20, 0.20, 1.303, 0.744]], dtype=DTYPE)
    uCm = torch.zeros(1, 2, dtype=DTYPE)
    u_safe, u_cbf, sing, lg, b = aux(xCm, uCm)
    assert not bool(sing[0])
    assert torch.allclose(u_cbf, _analytic_cbf_only_raw(uCm, lg, b, params), atol=1e-9)  # RAW, no clamp
    assert float(torch.linalg.norm(u_cbf - u_safe)) > 1e-2           # ||delta_u|| > 0

    # (C-strong) empty-intersection / infeasible (h>0 => alpha_unsafe=100): the raw CBF ask is huge,
    # ||delta_u|| >> cap, so the used deficit is norm-capped to exactly `cap` (the axis target region;
    # the clamped definition zeroed this — the reason for the correction).
    xCs = torch.tensor([[1.30, 0.85, 1.2, 0.7]], dtype=DTYPE)         # near obstacle, strong inbound, h>0
    uCs = torch.zeros(1, 2, dtype=DTYPE)
    u_safe, u_cbf, sing, lg, b = aux(xCs, uCs)
    assert not bool(sing[0])
    assert float(b[0]) < 0.0                                          # alpha_unsafe*h dominates => empty
    delta = u_cbf - u_safe
    raw_norm = float(torch.linalg.norm(delta))
    assert raw_norm > cap                                             # raw ask exceeds the cap
    used = delta * (cap / (torch.linalg.norm(delta, dim=1, keepdim=True) + 1e-9)).clamp(max=1.0)
    assert abs(float(torch.linalg.norm(used)) - cap) < 1e-6          # ||delta_u_used|| == cap

    # (D) singular step (position-only h => ||L_g h|| = 0) is flagged singular -> masked to 0.
    xD = torch.tensor([[1.30, 0.85, 0.0, 0.0]], dtype=DTYPE)
    uD = torch.tensor([[1.5, 1.0]], dtype=DTYPE)
    u_safe, u_cbf, sing, lg, b = aux(xD, uD, h_fn=_pos_h)
    assert float(torch.linalg.norm(lg)) < _SINGULAR_LG_THRESHOLD and bool(sing[0])
    nonsingular = (~sing).to(DTYPE)
    masked = nonsingular * torch.sum((u_cbf - u_safe) ** 2, dim=1)
    assert float(masked.abs().max()) == 0.0                          # singular contributes exactly 0


def test_routing_holds_with_w_deficit() -> None:
    config, system, value_net, policy_net = _setup()
    batch = _batch(system)
    value_net.zero_grad(set_to_none=True); policy_net.zero_grad(set_to_none=True)
    policy_bptt_loss(system=system, policy_net=policy_net, value_net=value_net,
                     batch=batch, config=_merge(config, {"loss": {"policy": {"w_deficit": 1.0}}})).total.backward()
    assert grad_norm(value_net.parameters()) < 1.0e-12
    assert grad_norm(policy_net.parameters()) > 0.0


def test_deficit_gradient_reaches_policy() -> None:
    # Minimal differentiable rollout with a CONTROLLED box-binding filter (h>0 states) -> the deficit
    # term L_deficit = mean ||delta_u||^2 has nonzero gradient to the policy.
    config, system, _, policy_net = _setup()
    hardnet = HardNetFilter(system, _vel_h, config)
    scene = batch_scenes([_scene()] * 4, device=DEV, dtype=DTYPE)
    x = torch.tensor([[0.20, 0.20, 1.303, 0.744], [0.30, 0.30, 1.30, 0.70],
                      [0.25, 0.25, 1.35, 0.72], [0.35, 0.40, 1.25, 0.68]], dtype=DTYPE)
    dt = float(config["env"]["dt"])
    policy_net.zero_grad(set_to_none=True)
    terms = []
    for _ in range(4):
        u_nom = policy_net(system.observation(x, scene))
        u_safe, _, u_cbf, sing = hardnet(x, scene, u_nom, return_deficit_aux=True)
        delta_u = u_cbf - u_safe
        ns = (~sing).to(DTYPE)
        terms.append((ns * torch.sum(delta_u * delta_u, dim=1)).mean())
        x = rk4_step(system, x, u_safe, dt)
    l_def = torch.stack(terms).mean()
    assert float(l_def) > 0.0                                         # deficit active (box binds)
    l_def.backward()
    assert grad_norm(policy_net.parameters()) > 0.0


def test_deficit_form_huber_vs_sqcap_gradient() -> None:
    # On an empty-intersection state where ||delta_u|| > deficit_cap, the sq_cap term is constant
    # (zero gradient to the policy) while the Huber term keeps a nonzero, bounded (<= cap) gradient.
    config, system, _, policy_net = _setup()
    cap = float(config["loss"]["policy"]["deficit_cap"])
    hardnet = HardNetFilter(system, _vel_h, config)
    scene = batch_scenes([_scene()], device=DEV, dtype=DTYPE)
    x = torch.tensor([[1.30, 0.85, 1.2, 0.7]], dtype=DTYPE)           # h>0, empty intersection

    def deficit_term(form):
        policy_net.zero_grad(set_to_none=True)
        u_nom = policy_net(system.observation(x, scene))
        u_safe, _, u_cbf, sing = hardnet(x, scene, u_nom, return_deficit_aux=True)
        delta_u = u_cbf - u_safe
        r = torch.linalg.norm(delta_u, dim=1)
        assert float(r) > cap and not bool(sing[0])                   # capped, non-singular
        if form == "huber":
            val = torch.where(r <= cap, 0.5 * r * r, cap * (r - 0.5 * cap))
        else:
            scale = (cap / (torch.linalg.norm(delta_u, dim=1, keepdim=True) + 1e-9)).clamp(max=1.0)
            du = delta_u * scale
            val = torch.sum(du * du, dim=1)
        ((~sing).to(DTYPE) * val).mean().backward()
        return grad_norm(policy_net.parameters())

    g_sqcap = deficit_term("sq_cap")
    g_huber = deficit_term("huber")
    assert g_sqcap < 1.0e-9                                           # sq_cap: zero grad above the cap
    assert g_huber > 1.0e-6                                           # huber: nonzero grad above the cap

    # Per-step Huber gradient w.r.t. delta_u is bounded by deficit_cap (|dL/dr| = cap for r > cap).
    du = torch.tensor([[3.0, 4.0]], dtype=DTYPE, requires_grad=True)  # ||du|| = 5 > cap
    r = torch.linalg.norm(du, dim=1)
    torch.where(r <= cap, 0.5 * r * r, cap * (r - 0.5 * cap)).sum().backward()
    assert float(torch.linalg.norm(du.grad)) <= cap + 1.0e-9          # bounded gradient


def _config() -> dict[str, Any]:
    import yaml
    base = yaml.safe_load((REPO_ROOT / "src/configs/base_config.yaml").read_text())
    exp = yaml.safe_load((REPO_ROOT / "src/configs/exp_config.yaml").read_text())
    return _merge(base, exp)


def _merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged
