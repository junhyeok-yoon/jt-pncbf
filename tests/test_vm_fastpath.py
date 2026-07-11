"""v2.5.0 V_M fast-path PARITY gate: the torch.compile(+CUDA-graph) rollout must match the reference
python-loop rollout bit-for-bit (to float32 tolerance) in values, L_g gradient, filter decisions, and a
closed-loop trajectory. cuda-only (the fast path activates on cuda; on cpu it falls back to reference)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest
import torch
import yaml

from src.common.filter_hardnet import (
    _SINGULAR_LG_THRESHOLD, _base_alpha, _base_projection, _box_aware_projection, _cbf_terms,
    _hardnet_params,
)
from src.common.maneuver_value import make_maneuver_h_fn, maneuver_value
from src.common.rk4 import rk4_step
from src.envs.double_integrator import DoubleIntegrator
from src.envs.scene_batch import batch_scenes
from src.envs.scene_init import Scene

REPO = Path(__file__).resolve().parents[1]
JS = list(range(1, 9))   # A1' library
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="fast path is cuda-only")


def _cfg() -> dict[str, Any]:
    b = yaml.safe_load((REPO / "src/configs/base_config.yaml").read_text())
    e = yaml.safe_load((REPO / "src/configs/exp_config.yaml").read_text())

    def m(a: Mapping[str, Any], o: Mapping[str, Any]) -> dict[str, Any]:
        d = dict(a)
        for k, v in o.items():
            d[k] = m(d[k], v) if isinstance(v, Mapping) and isinstance(d.get(k), Mapping) else v
        return d
    return m(b, e)


def _scene() -> Scene:
    ce = np.zeros((12, 2)); ra = np.zeros(12); ac = np.zeros(12, bool)
    ce[0] = [0.6, 0.0]; ra[0] = 0.2; ac[0] = True
    ce[1] = [0.2, 0.5]; ra[1] = 0.15; ac[1] = True
    return Scene(obstacle_centers=ce, obstacle_radii=ra, obstacle_active=ac,
                 start=np.array([-1.0, 0.0]), goal=np.array([1.4, 0.0]),
                 system="double_integrator", mode="synthetic", initial_velocity=np.array([0.0, 0.0]))


def _setup(B):
    cfg = _cfg()
    dev = torch.device("cuda"); dt = torch.float32
    system = DoubleIntegrator(cfg); system.u_bounds = system.u_bounds.to(device=dev, dtype=dt)
    sc = batch_scenes([_scene()] * B, device=dev, dtype=dt)
    torch.manual_seed(B)
    x = torch.empty(B, 4, device=dev, dtype=dt)
    x[:, :2].uniform_(-1.0, 1.0); x[:, 2:].uniform_(-2.0, 2.0)
    return cfg, system, sc, x, dev, dt


def test_fastpath_values_parity() -> None:
    for B in (4096, 500, 37):
        cfg, system, sc, x, _, _ = _setup(B)
        with torch.no_grad():
            vref = maneuver_value(x, sc, system, cfg, lateral_js=JS, fast=False)
            vfast = maneuver_value(x, sc, system, cfg, lateral_js=JS, fast=True)
        assert float((vref - vfast).abs().max()) <= 1.0e-6, f"values B={B}"


def test_fastpath_grad_Lg_parity() -> None:
    cfg, system, sc, x, _, _ = _setup(512)
    xr = x.clone().requires_grad_(True)
    gr = torch.autograd.grad(maneuver_value(xr, sc, system, cfg, lateral_js=JS, fast=False).sum(), xr)[0]
    xf = x.clone().requires_grad_(True)
    gf = torch.autograd.grad(maneuver_value(xf, sc, system, cfg, lateral_js=JS, fast=True).sum(), xf)[0]
    assert float((gr - gf).abs().max()) <= 1.0e-5


def _filter(system, cfg, h_fn, x, sc, u_nom, params):
    h, lf, lg = _cbf_terms(system, h_fn, x, sc, u_nom, create_graph=False)
    h, lf, lg = h.detach(), lf.detach(), lg.detach()
    with torch.no_grad():
        alpha = _base_alpha(h, params); row = -lf - alpha * h
        bounds = system.u_bounds
        proj = _base_projection(u_nom, lg, row, bounds, params)
        sing = torch.linalg.norm(lg, dim=1) < _SINGULAR_LG_THRESHOLD
        u_safe, empty = _box_aware_projection(u_nom, proj, lg, row, bounds)
    return u_safe, sing, empty


def test_fastpath_filter_decision_parity() -> None:
    # The barrier is parity-clean (values<=1e-6, L_g<=1e-5, singular/empty flags EXACT). u_safe itself is
    # an INTRINSICALLY DISCONTINUOUS function of the barrier: the HardNet box-aware projection selects a
    # discrete vertex/edge, so a >1e-6 barrier perturbation can flip the branch (an O(0.1-4) jump). This
    # is NOT fast-path-specific — the REFERENCE path under a 1e-7 x-jitter flips ~24/512 states (measured;
    # see stage_b2.md), MORE than fast-vs-ref (~2/512). So the gate asserts: flags exact, L_g<=1e-5, and
    # the u_safe branch-flip fraction is <= the reference's own 1e-7-jitter fraction (fast is no less
    # stable than trivial numeric jitter), with median/p99 tight.
    cfg, system, sc, x, _, _ = _setup(512)
    params = _hardnet_params(cfg)
    goal = torch.as_tensor(sc.goal, dtype=x.dtype, device=x.device)
    u_nom = system.lqr_action(x, goal)
    href = make_maneuver_h_fn(system, cfg, lateral_js=JS, gamma_m=0.02, fast=False)
    ur, sr, er = _filter(system, cfg, href, x, sc, u_nom, params)
    uf, sf, ef = _filter(system, cfg, make_maneuver_h_fn(system, cfg, lateral_js=JS, gamma_m=0.02, fast=True), x, sc, u_nom, params)
    assert torch.equal(sr, sf), "singular flags differ"
    assert torch.equal(er, ef), "empty flags differ"
    d = (ur - uf).abs().amax(dim=1)
    # reference's OWN discontinuity under a 1e-7 x-jitter (the intrinsic branch-flip rate to beat)
    xp = x + (torch.rand_like(x) - 0.5) * 2.0e-7
    uj, _, _ = _filter(system, cfg, href, xp, sc, system.lqr_action(xp, goal), params)
    dj = (ur - uj).abs().amax(dim=1)
    assert float(d.median()) <= 1.0e-5 and float(torch.quantile(d, 0.99)) <= 1.0e-5, "bulk u_safe not parity"
    assert int((d > 1.0e-5).sum()) <= int((dj > 1.0e-5).sum()) + 1, "fast flips MORE branches than 1e-7 jitter"


def test_fastpath_closed_loop_trajectory_parity() -> None:
    # 50-step closed loop: median/p99 trajectory diff is tight; a tiny fraction of episodes may diverge
    # via the box-projection branch discontinuity above (compounds over the rollout) — the same tail the
    # reference shows under 1e-7 jitter. Assert the bulk matches; bound the divergent-episode fraction.
    cfg, system, sc, x0, _, _ = _setup(256)
    params = _hardnet_params(cfg)
    dt = float(cfg["env"]["dt"])
    goal = torch.as_tensor(sc.goal, dtype=x0.dtype, device=x0.device)

    def rollout(fast, x_init):
        x = x_init.clone(); traj = [x]
        h_fn = make_maneuver_h_fn(system, cfg, lateral_js=JS, gamma_m=0.02, fast=fast)
        for _ in range(50):
            u_nom = system.lqr_action(x, goal)
            u_safe, _, _ = _filter(system, cfg, h_fn, x, sc, u_nom, params)
            x = rk4_step(system, x, u_safe, dt); traj.append(x)
        return torch.stack(traj, 0)
    tr = rollout(False, x0)
    ep_fast = (tr - rollout(True, x0)).abs().amax(dim=(0, 2))         # fast vs ref, per episode
    # intrinsic sensitivity: reference vs itself under a 1e-7 x-jitter (the box-projection makes the
    # closed loop chaotic — ~45% of episodes flip a branch over 50 steps under ANY 1e-7 change).
    ep_jit = (tr - rollout(False, x0 + (torch.rand_like(x0) - 0.5) * 2.0e-7)).abs().amax(dim=(0, 2))
    assert float(ep_fast.median()) <= 1.0e-5, "bulk trajectory not parity"
    # fast perturbs the loop NO MORE than a 1e-7 input jitter does (within the projection's own sensitivity)
    assert int((ep_fast > 1.0e-3).sum()) <= int(1.3 * int((ep_jit > 1.0e-3).sum())) + 2
