"""v2.6.0 Stage 0 M1 — planar quadrotor direct dynamics self-checks (theory note Def 2.1).
(a) hover equilibrium f_thr=m*g, tau=0 gives x_dot~=0; (b) ballistic (zero input) matches exact
projectile / RK4 is exact for the quadratic; (c) wrap_state clamps ||v||<=v_max and |omega|<=omega_max;
(d) RK4 reversibility on a short horizon (no clamp regime)."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from src.common.control_net import ControlNet
from src.common.rk4 import rk4_step
from src.envs.quadrotor_planar import QuadrotorPlanar

REPO = Path(__file__).resolve().parents[1]


def _cfg() -> dict[str, Any]:
    b = yaml.safe_load((REPO / "src/configs/base_config.yaml").read_text())
    e = yaml.safe_load((REPO / "src/configs/exp_config.yaml").read_text())

    def m(a, o):
        d = dict(a)
        for k, v in o.items():
            d[k] = m(d[k], v) if isinstance(v, Mapping) and isinstance(d.get(k), Mapping) else v
        return d

    return m(b, e)


def _sys() -> QuadrotorPlanar:
    s = QuadrotorPlanar(_cfg())
    s.u_bounds = s.u_bounds.to(torch.float64)
    return s


def test_hover_equilibrium():
    s = _sys()
    m, g = s.mass, s.gravity
    x = torch.zeros(8, 6, dtype=torch.float64)
    x[:, 0].uniform_(-3, 3)          # arbitrary position, all else 0 (theta=0, v=0, omega=0)
    x[:, 1].uniform_(-3, 3)
    u = torch.tensor([[m * g, 0.0]], dtype=torch.float64).expand(8, 2)
    xdot = s.dynamics(x, u)
    assert float(xdot.abs().max()) <= 1e-12, float(xdot.abs().max())
    assert abs(m * g - 9.81) < 1e-9    # hover thrust == m*g == 9.81


def test_ballistic_zero_input_matches_projectile():
    s = _sys()
    g = s.gravity
    dt = 0.05
    x = torch.zeros(1, 6, dtype=torch.float64)
    x[0, 2] = 0.7                      # attitude irrelevant at zero thrust
    x[0, 3] = 0.4                      # vx0
    x[0, 4] = 0.0                      # vy0
    u = torch.zeros(1, 2, dtype=torch.float64)   # no thrust, no torque
    xs = x.clone()
    N = 5
    for _ in range(N):
        xs = rk4_step(s, xs, u, dt)
    t = N * dt
    # exact projectile: vx const, vy = -g t, px = px0 + vx0 t, py = -0.5 g t^2, theta/omega unchanged
    assert abs(float(xs[0, 3]) - 0.4) <= 1e-12
    assert abs(float(xs[0, 4]) - (-g * t)) <= 1e-9
    assert abs(float(xs[0, 0]) - (0.4 * t)) <= 1e-9
    assert abs(float(xs[0, 1]) - (-0.5 * g * t * t)) <= 1e-9
    assert abs(float(xs[0, 2]) - 0.7) <= 1e-12   # zero torque -> attitude held
    assert float(torch.linalg.norm(xs[0, 3:5])) <= s.v_max   # stayed under the clamp


def test_wrap_state_clamps_velocity_and_omega():
    s = _sys()
    x = torch.tensor([[0.0, 0.0, 3.5, 3.0, 4.0, 10.0]], dtype=torch.float64)  # ||v||=5>2.5, omega=10>4
    xw = s.wrap_state(x)
    speed = float(torch.linalg.norm(xw[0, 3:5]))
    assert speed <= s.v_max + 1e-9 and abs(speed - s.v_max) <= 1e-9      # clamped to v_max
    # direction preserved (v was (3,4), unit (0.6,0.8))
    assert abs(float(xw[0, 3]) - s.v_max * 0.6) <= 1e-9
    assert abs(float(xw[0, 4]) - s.v_max * 0.8) <= 1e-9
    assert abs(float(xw[0, 5]) - s.omega_max) <= 1e-9                    # omega clamped to +4
    assert -math.pi < float(xw[0, 2]) <= math.pi                        # theta wrapped


def test_rk4_reversibility_short_horizon():
    s = _sys()
    dt = 0.05
    x0 = torch.tensor([[0.5, -0.3, 0.1, 0.2, -0.1, 0.0]], dtype=torch.float64)
    u = torch.tensor([[9.81, 0.05]], dtype=torch.float64)   # near hover + small torque; no clamp fires
    x = x0.clone()
    N = 5
    for _ in range(N):
        x = rk4_step(s, x, u, dt)
    # integrate backward with -dt from x_N; should recover x0 (wrap_state is identity in this regime)
    for _ in range(N):
        x = rk4_step(s, x, u, -dt)
    assert float((x - x0).abs().max()) <= 1e-5, float((x - x0).abs().max())


# ---- M2: barrier h_star + gradients ----

def _scene(centers, radii, goal):
    from types import SimpleNamespace
    n = 12
    C = torch.zeros(n, 2, dtype=torch.float64)
    R = torch.zeros(n, dtype=torch.float64)
    A = torch.zeros(n, dtype=torch.bool)
    for i, (cc, rr) in enumerate(zip(centers, radii)):
        C[i] = torch.tensor(cc, dtype=torch.float64); R[i] = rr; A[i] = True
    return SimpleNamespace(obstacle_centers=C, obstacle_radii=R, obstacle_active=A,
                           goal=torch.tensor(goal, dtype=torch.float64))


def test_h_star_analytic_vs_fd_gradient():
    from src.common.quadrotor_barrier import h_star_value
    cfg = _cfg(); h_scale = float(cfg["env"]["h_scale"]); c = 0.5
    sc = _scene([(0.0, 0.0)], [0.5], (2.0, 2.0))
    torch.manual_seed(0)
    # place states in the phi ramp band around the single obstacle (clearance/h_scale in (0,1)):
    # radius 0.5, h_scale 0.35 -> clearance in (0, 0.35) => distance in (0.5, 0.85).
    B = 64
    ang = torch.empty(B, dtype=torch.float64).uniform_(-3.14, 3.14)
    dist = torch.empty(B, dtype=torch.float64).uniform_(0.55, 0.80)   # strictly inside the ramp
    x = torch.zeros(B, 6, dtype=torch.float64)
    x[:, 0] = dist * torch.cos(ang); x[:, 1] = dist * torch.sin(ang)
    x[:, 2].uniform_(-3.14, 3.14); x[:, 3].uniform_(-2.0, 2.0); x[:, 4].uniform_(-2.0, 2.0); x[:, 5].uniform_(-1, 1)

    xr = x.clone().requires_grad_(True)
    hv = h_star_value(xr, sc, c, h_scale)
    grad_analytic, = torch.autograd.grad(hv.sum(), xr)

    eps = 1e-6
    grad_fd = torch.zeros_like(x)
    for d in range(6):
        xp = x.clone(); xp[:, d] += eps
        xm = x.clone(); xm[:, d] -= eps
        grad_fd[:, d] = (h_star_value(xp, sc, c, h_scale) - h_star_value(xm, sc, c, h_scale)) / (2 * eps)
    assert float((grad_analytic - grad_fd).abs().max()) <= 1e-5, float((grad_analytic - grad_fd).abs().max())


def test_h_star_augmentation_is_obstacle_agnostic():
    from src.common.quadrotor_barrier import h_star_value, phi_value, approach_speed
    cfg = _cfg(); h_scale = float(cfg["env"]["h_scale"]); c = 0.5
    sc1 = _scene([(0.0, 0.0)], [0.5], (2.0, 2.0))
    sc2 = _scene([(1.0, -1.0), (-2.0, 0.5)], [0.4, 0.6], (3.0, 0.0))
    torch.manual_seed(1)
    x = torch.randn(50, 6, dtype=torch.float64)
    aug1 = h_star_value(x, sc1, c, h_scale) - phi_value(x, sc1, h_scale)
    aug2 = h_star_value(x, sc2, c, h_scale) - phi_value(x, sc2, h_scale)
    # the augmentation c(v^T Re) is identical across scenes (o-independent, R4) and equals c*approach_speed
    assert float((aug1 - aug2).abs().max()) <= 1e-12
    assert float((aug1 - c * approach_speed(x)).abs().max()) <= 1e-12


# ---- M3: direct-input box-aware HardNet on (f_thr, tau) with the asymmetric box ----

def test_hardnet_direct_input_asymmetric_box():
    from src.common.filter_hardnet import HardNetFilter, _cbf_terms, _base_alpha, _hardnet_params
    from src.common.quadrotor_barrier import make_exact_value_fn
    cfg = _cfg(); h_scale = float(cfg["env"]["h_scale"]); dt = float(cfg["env"]["dt"]); c = 0.5
    s = _sys()
    sc = _scene([(0.0, 0.0)], [0.5], (2.0, 2.0))
    v_fn = make_exact_value_fn(s, c, h_scale, dt, horizon=25)
    filt = HardNetFilter(s, v_fn, cfg)

    torch.manual_seed(3)
    B = 96
    x = torch.zeros(B, 6, dtype=torch.float64)
    x[:, 0].uniform_(0.55, 1.6); x[:, 1].uniform_(-0.6, 0.6)   # around the obstacle boundary
    x[:, 2].uniform_(-3.14, 3.14); x[:, 3].uniform_(-1.5, 1.5); x[:, 4].uniform_(-1.5, 1.5); x[:, 5].uniform_(-1, 1)
    u_nom = s.lqr_action(x, sc.goal)
    u_out, infeasible = filt(x, sc, u_nom)
    u_out = u_out.detach()

    lo = s.u_bounds[:, 0].to(torch.float64); hi = s.u_bounds[:, 1].to(torch.float64)
    # (1) output lands in the ASYMMETRIC box [0,19.62] x [-1.0,1.0]  (v2.6.1: torque box 0.2->1.0)
    assert bool((u_out[:, 0] >= lo[0] - 1e-7).all() and (u_out[:, 0] <= hi[0] + 1e-7).all())
    assert bool((u_out[:, 1] >= lo[1] - 1e-7).all() and (u_out[:, 1] <= hi[1] + 1e-7).all())
    assert abs(float(hi[0]) - 19.62) < 1e-9 and abs(float(hi[1]) - 1.0) < 1e-9   # asymmetric, not a square

    # reconstruct the CBF constraint row  L_g V . u <= row_upper  ( = -L_f V - alpha V )
    h, lf, lg = _cbf_terms(s, v_fn, x, sc, u_nom, create_graph=False)
    params = _hardnet_params(cfg); alpha = _base_alpha(h, params)
    row_upper = (-lf - alpha * h).detach()
    lg = lg.detach()
    slack_nom = row_upper - torch.sum(lg * u_nom, dim=1)          # >=0 => nominal already feasible
    feasible = slack_nom >= -1e-9

    # (2) on a feasible state the nominal passes UNALTERED
    if bool(feasible.any()):
        du = (u_out[feasible] - u_nom[feasible]).abs().max()
        assert float(du) <= 1e-6, float(du)

    # (3) corner/empty fallback fires ONLY when the half-space misses the box: for every flagged-infeasible
    # non-singular row, the best in-box point still violates the constraint (min over box corners > row_upper).
    lg_norm = torch.linalg.norm(lg, dim=1)
    nonsingular = lg_norm >= 5.0e-4
    corners = torch.tensor([[lo[0], lo[1]], [lo[0], hi[1]], [hi[0], lo[1]], [hi[0], hi[1]]], dtype=torch.float64)
    best_inbox = torch.min(corners @ lg.T, dim=0).values        # min_{box} L_g.u  (linear => at a corner)
    flagged = infeasible & nonsingular
    if bool(flagged.any()):
        # half-space {L_g.u <= row_upper} misses the box  <=>  min_{box} L_g.u > row_upper
        assert bool((best_inbox[flagged] > row_upper[flagged] - 1e-7).all())
    # and where NOT flagged, the output satisfies the constraint
    ok = (~infeasible) & nonsingular
    if bool(ok.any()):
        viol = (torch.sum(lg * u_out, dim=1) - row_upper)[ok].max()
        assert float(viol) <= 1e-5, float(viol)


# ---- v2.6.0 policy-output alignment: clamp_tanh head reaches the box boundary ----

def test_clamp_tanh_head_reaches_box_boundary_and_stays_in_box():
    cfg = _cfg()
    cfg["network"]["control"]["output"] = "clamp_tanh"
    system = QuadrotorPlanar(cfg)
    net = ControlNet(system.obs_dim, system, cfg).double()
    net.u_bounds = net.u_bounds.double()
    lo = system.u_bounds[:, 0].double(); hi = system.u_bounds[:, 1].double()
    obs0 = torch.zeros(4, system.obs_dim, dtype=torch.float64)
    with torch.no_grad():
        net.head.bias.fill_(100.0)                       # large +preactivation -> clamp -> upper box
        out_hi = net(obs0)
        net.head.bias.fill_(-100.0)                      # large -preactivation -> clamp -> lower box
        out_lo = net(obs0)
    # boundary REACHED (softsign/tanh never would): f_max=19.62, tau_max=1.0 ; f_min=0, -tau_max (v2.6.1 box 0.2->1.0)
    assert torch.allclose(out_hi, hi.expand_as(out_hi), atol=1e-6), out_hi[0]
    assert torch.allclose(out_lo, lo.expand_as(out_lo), atol=1e-6), out_lo[0]
    assert abs(float(out_hi[0, 1]) - 1.0) < 1e-6 and abs(float(out_lo[0, 1]) + 1.0) < 1e-6   # torque box
    # random preactivation stays IN the asymmetric box
    with torch.no_grad():
        torch.manual_seed(0)
        net.head.bias.uniform_(-6, 6); net.head.weight.uniform_(-6, 6)
        out = net(torch.randn(256, system.obs_dim, dtype=torch.float64))
    assert bool((out[:, 0] >= lo[0] - 1e-9).all() and (out[:, 0] <= hi[0] + 1e-9).all())
    assert bool((out[:, 1] >= lo[1] - 1e-9).all() and (out[:, 1] <= hi[1] + 1e-9).all())
