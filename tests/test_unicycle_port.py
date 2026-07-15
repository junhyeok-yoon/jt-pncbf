"""v2.5.2 Stage 1 — unicycle port of the exact-backup safety chain (changes.md §2). Five checks, all on
CPU (no checkpoint, 0 skips): (1) F1 unicycle brake rest-completion at the float32 floor with theta held;
(2) L_g h relative degree 1 in BOTH a and omega (§2.2a) with the median-|dV/dtheta| floor; (3) exact_m0 ==
labeler j=0 (values/grads) on the unicycle; (4) DI brake bit-parity after the refactor; (5) generic segment
sub-integration reproduces the DI closed form in the non-clamping regime."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from src.common.filter_hardnet import _cbf_terms, _SINGULAR_LG_THRESHOLD
from src.common.observation import scene_goal_tensor
from src.common.rk4 import rk4_step
from src.envs.double_integrator import DoubleIntegrator
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.envs.unicycle import Unicycle
from src.frameworks.cpi.backup import deadband_brake, t_stop
from src.frameworks.cpi.channel import make_exact_m0_h_fn
from src.frameworks.cpi.labels import label_streams, m0_value_raw, sample_scenes, stack_scene_obstacles
from src.frameworks.cpi.shield import _seg_min_clearance

REPO = Path(__file__).resolve().parents[1]


def _cfg(system_name="double_integrator") -> dict[str, Any]:
    b = yaml.safe_load((REPO / "src/configs/base_config.yaml").read_text())
    e = yaml.safe_load((REPO / "src/configs/exp_config.yaml").read_text())

    def m(a, o):
        d = dict(a)
        for k, v in o.items():
            d[k] = m(d[k], v) if isinstance(v, Mapping) and isinstance(d.get(k), Mapping) else v
        return d
    cfg = m(b, e); cfg["run"]["system"] = system_name
    cfg["safety_channel"] = {"type": "exact_m0", "gamma_margin": 0.0}
    return cfg


def test_unicycle_rest_completion_F1():
    cfg = _cfg("unicycle"); sysm = Unicycle(cfg); sysm.u_bounds = sysm.u_bounds.to(torch.float32)
    dt = float(cfg["env"]["dt"]); v_max = float(cfg["env"]["bounds"]["unicycle"]["v_max"])
    torch.manual_seed(0); B = 4096
    x = torch.empty(B, 4)
    x[:, :2].uniform_(-4.0, 4.0); x[:, 2].uniform_(-math.pi, math.pi)
    x[:, 3].uniform_(-v_max, v_max)
    theta0 = x[:, 2].clone()
    ts = t_stop(sysm, cfg, dt); assert ts == 25
    for _ in range(ts):
        x = rk4_step(sysm, x, deadband_brake(x, sysm, cfg, dt), dt)
    assert float(x[:, 3].abs().max()) <= 7.5e-9, float(x[:, 3].abs().max())     # rest at the float32 floor
    assert float((x[:, 2] - theta0).abs().max()) <= 1e-6                        # heading held (omega=0)


def _realistic_unicycle_states(cfg, sysm, device, n_scenes=400, roll=30):
    """Sample unicycle scenes and roll the LQR policy so states populate the interior; return |v|>1.0 states
    and their per-state obstacle scene (NOT dead-ahead — the generic distribution §2.2a is about)."""
    DT = torch.float32
    scenes = sample_scenes(n_scenes, label_streams(cfg)["scene"], cfg)
    bs = batch_scenes(scenes, device=device, dtype=DT); x = initial_states_from_batch(bs).to(DT)
    with torch.no_grad():
        for _ in range(roll):
            x = rk4_step(sysm, x, sysm.lqr_action(x, scene_goal_tensor(bs, x)), float(cfg["env"]["dt"]))
    m = x[:, 3].abs() > 1.0
    from types import SimpleNamespace
    scn = SimpleNamespace(obstacle_centers=bs.obstacle_centers[m], obstacle_radii=bs.obstacle_radii[m],
                          obstacle_active=bs.obstacle_active[m], goal=bs.goal[m])
    return x[m], scn


def test_unicycle_Lg_h_relative_degree_both_controls():
    cfg = _cfg("unicycle"); dev = torch.device("cpu"); sysm = Unicycle(cfg); sysm.u_bounds = sysm.u_bounds.to(torch.float32)
    h_fn = make_exact_m0_h_fn(sysm, cfg)
    torch.manual_seed(0)
    xs, scn = _realistic_unicycle_states(cfg, sysm, dev)
    _, _, lg = _cbf_terms(sysm, h_fn, xs, scn, torch.zeros(xs.shape[0], 2), create_graph=False)
    lg = lg.detach()
    dVdv = lg[:, 0].abs().cpu().numpy(); dVdth = lg[:, 1].abs().cpu().numpy()   # [dV/dv (a), dV/dtheta (omega)]
    frac_sing = float(((lg[:, 0].abs() < _SINGULAR_LG_THRESHOLD) | (lg[:, 1].abs() < _SINGULAR_LG_THRESHOLD)).float().mean())
    med_th = float(np.median(dVdth))
    # §2.2a: relative degree 1 in BOTH controls; tightened Gate 4 — median |dV/dtheta| must clear 0.1.
    assert med_th >= 0.1, f"median |dV/dtheta| = {med_th} < 0.1 (dead heading channel)"
    assert dVdv.max() > 0.0 and dVdth.max() > 0.0     # both controls carry authority somewhere
    # (frac_sing high is expected — far-field states have V_m0 flat; not asserted, reported by the run)
    _ = frac_sing


def test_exact_m0_equals_labeler_j0_unicycle():
    cfg = _cfg("unicycle"); sysm = Unicycle(cfg); sysm.u_bounds = sysm.u_bounds.to(torch.float64)
    h_fn = make_exact_m0_h_fn(sysm, cfg)
    torch.manual_seed(0); B = 256
    x = torch.empty(B, 4, dtype=torch.float64)
    x[:, :2].uniform_(-3, 3); x[:, 2].uniform_(-math.pi, math.pi); x[:, 3].uniform_(-2.5, 2.5)
    K = 12; C = torch.zeros(B, K, 2, dtype=torch.float64); R = torch.zeros(B, K, dtype=torch.float64)
    A = torch.zeros(B, K, dtype=torch.bool)
    C[:, 0] = torch.tensor([1.0, 0.5]); R[:, 0] = 0.4; A[:, 0] = True
    from types import SimpleNamespace
    scn = SimpleNamespace(obstacle_centers=C, obstacle_radii=R, obstacle_active=A, goal=torch.zeros(B, 2, dtype=torch.float64))
    hv = h_fn(x, scn)
    ref = m0_value_raw(x, C, R, A, sysm, cfg, float(cfg["env"]["dt"]))
    assert float((hv - ref).abs().max()) <= 1e-6
    # gradients agree (channel h_fn IS m0_value_raw, so bit-identical; assert <= 1e-5)
    xr1 = x.clone().requires_grad_(True); g1, = torch.autograd.grad(h_fn(xr1, scn).sum(), xr1)
    xr2 = x.clone().requires_grad_(True); g2, = torch.autograd.grad(m0_value_raw(xr2, C, R, A, sysm, cfg, float(cfg["env"]["dt"])).sum(), xr2)
    assert float((g1 - g2).abs().max()) <= 1e-5


def test_di_brake_bit_parity_after_refactor():
    cfg = _cfg("double_integrator"); sysm = DoubleIntegrator(cfg); sysm.u_bounds = sysm.u_bounds.to(torch.float64)
    dt = float(cfg["cpi"]["labels"]["dt_vm"]); u_max = 2.0
    torch.manual_seed(1); x = torch.empty(2048, 4, dtype=torch.float64); x.uniform_(-3, 3)
    refactored = deadband_brake(x, sysm, cfg, dt)
    v = x[:, 2:4]
    inlined = torch.where(v.abs() > u_max * dt, -u_max * torch.sign(v), -v / dt)   # v2.5.1 inlined DI brake
    assert torch.equal(refactored, inlined)


def test_segment_clearance_parity_di_nonclamping():
    cfg = _cfg("double_integrator"); sysm = DoubleIntegrator(cfg); sysm.u_bounds = sysm.u_bounds.to(torch.float64)
    torch.manual_seed(0); B = 64
    x = torch.empty(B, 4, dtype=torch.float64); x[:, :2].uniform_(-3, 3); x[:, 2:].uniform_(-1.0, 1.0)  # non-clamping
    u = torch.empty(B, 2, dtype=torch.float64).uniform_(-2, 2)
    K = 12; C = torch.zeros(B, K, 2, dtype=torch.float64); R = torch.zeros(B, K, dtype=torch.float64)
    A = torch.zeros(B, K, dtype=torch.bool); C[:, 0] = torch.tensor([1.0, 0.5]); R[:, 0] = 0.4; A[:, 0] = True
    dt_ctrl, dt_check = 0.05, 0.01
    gen = _seg_min_clearance(x, u, sysm, dt_ctrl, dt_check, C, R, A)
    p0 = x[:, :2]; v0 = x[:, 2:4]; n = round(dt_ctrl / dt_check)
    ts = torch.linspace(dt_check, dt_ctrl, n, dtype=torch.float64)
    best = torch.full((B,), float("inf"), dtype=torch.float64)
    for t in ts:
        p = p0 + v0 * t + 0.5 * u * (t * t)
        dist = torch.linalg.norm(p.unsqueeze(1) - C, dim=-1)
        clr = torch.where(A, dist - R, torch.full_like(dist, float("inf"))); best = torch.minimum(best, clr.amin(1))
    assert float((gen - best).abs().max()) <= 1e-9
