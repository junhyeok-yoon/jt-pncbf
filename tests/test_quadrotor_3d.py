"""v2.7.2 M1 — quadrotor_3d system bring-up tests (q1..q8).

q1  quaternion-norm invariance (wrap_state + RK4 keep the attitude a unit quaternion; double-cover canonical).
q2  hover fixed-point (identity attitude at goal, v=omega=0 -> nominal ~ hover; dynamics ~ 0).
q3  yaw-symmetry (global world-yaw of state+scene -> obs bit-identical; a pure tilt is observable via g^b).
q4  obs layout (dim 32; block indices v^b/omega^b/goal^b/g^b/cylinder decoded exactly at identity attitude).
q5  golden bit-parity (DI/unicycle value_target_barrier == signed_h; planar h_star == phi + c v^T Re, unchanged).
q6  IC-attitude distribution (tilt phi ~ U[0,2pi/3] w/ no inversion; tilt-axis + yaw uniform).
q7  convention cross-checks (R^T R=I; g^b for known tilts; free fall; hover; e3-torque yaw-only; nominal recovery).
q8  autograd sanity (finite-difference vs autograd Jacobian of one RK4 step).
"""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import torch

from src.common.quadrotor_barrier import value_target_barrier
from src.common.rk4 import rk4_step
from src.common.signed_h import signed_h
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.envs.scene_init import sample_train_scene
from src.frameworks.jt_pncbf.train import load_effective_config, make_system

torch.manual_seed(0)


def _cfg(system="quadrotor_3d"):
    c = load_effective_config()
    c["run"]["system"] = system
    return c


def _sys(system="quadrotor_3d"):
    return make_system(_cfg(system))


def _rand_quat(n, dtype=torch.float64, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, 4, generator=g, dtype=dtype)
    q = q / torch.linalg.norm(q, dim=1, keepdim=True)
    return q * torch.sign(q[:, :1] + (q[:, :1] == 0))            # canonical w>=0


def _R(q):
    from src.envs.quadrotor_3d import _quat_to_R
    return _quat_to_R(q)


def _batch(n=16, seed=1, dtype=torch.float32, system="quadrotor_3d"):
    cfg = _cfg(system)
    s = make_system(cfg)
    rng = np.random.default_rng(seed)
    scenes = [sample_train_scene(rng, cfg, system) for _ in range(n)]
    bs = batch_scenes(scenes, device=torch.device("cpu"), dtype=dtype)
    x = s.wrap_state(initial_states_from_batch(bs).to(dtype))
    return s, x, bs, cfg


# ---------------------------------------------------------------------------- q1
def test_q1_quaternion_norm_invariance():
    s, x, bs, cfg = _batch(n=32, dtype=torch.float64)
    dt = float(cfg["env"]["dt"])
    qn = torch.linalg.norm(x[:, 3:7], dim=1)
    assert torch.allclose(qn, torch.ones_like(qn), atol=1e-9)     # wrap_state renormalizes
    assert bool((x[:, 3] >= -1e-12).all())                        # canonical w>=0
    g = torch.Generator().manual_seed(3)
    for _ in range(20):                                           # roll: RK4+wrap must keep |q|=1
        u = s.lqr_action(x, bs.goal) + 0.2 * torch.randn(x.shape[0], 4, generator=g, dtype=x.dtype)
        x = rk4_step(s, x, u, dt)
        qn = torch.linalg.norm(x[:, 3:7], dim=1)
        assert torch.allclose(qn, torch.ones_like(qn), atol=1e-9)
        assert bool((x[:, 3] >= -1e-12).all())


# ---------------------------------------------------------------------------- q2
def test_q2_hover_fixed_point():
    s = _sys()
    n = 4
    goal = torch.tensor([[1.0, -0.5, 0.3]], dtype=torch.float64).repeat(n, 1)
    x = torch.zeros(n, 13, dtype=torch.float64)
    x[:, :3] = goal                                              # at goal
    x[:, 3] = 1.0                                                # identity quaternion
    # v = omega = 0
    u = s.lqr_action(x, goal)
    assert torch.allclose(u[:, 0], torch.full((n,), s.mass * s.gravity, dtype=torch.float64), atol=1e-6)
    assert torch.allclose(u[:, 1:4], torch.zeros(n, 3, dtype=torch.float64), atol=1e-6)
    dx = s.dynamics(x, u)
    assert torch.allclose(dx, torch.zeros_like(dx), atol=1e-6)   # true fixed point


# ---------------------------------------------------------------------------- q3
def _yaw_quat(psi, n, dtype):
    z = torch.zeros(n, dtype=dtype)
    return torch.stack([torch.cos(psi / 2) + z, z, z, torch.sin(psi / 2) + z], dim=1)


def _ns_scene(goal, centers, radii, active):
    from types import SimpleNamespace
    return SimpleNamespace(goal=goal, obstacle_centers=centers, obstacle_radii=radii, obstacle_active=active)


def test_q3_yaw_symmetry_and_tilt_observability():
    from src.envs.quadrotor_3d import _quat_mul
    s = _sys()
    dt = torch.float64
    n, K = 8, 5
    g = torch.Generator().manual_seed(7)
    p = torch.randn(n, 3, generator=g, dtype=dt)
    q = _rand_quat(n, dtype=dt, seed=11)
    v = torch.randn(n, 3, generator=g, dtype=dt)
    om = torch.randn(n, 3, generator=g, dtype=dt)
    x = torch.cat([p, q, v, om], dim=1)
    goal = torch.randn(n, 3, generator=g, dtype=dt)
    centers = torch.randn(n, K, 2, generator=g, dtype=dt)
    radii = 0.3 + 0.2 * torch.rand(n, K, generator=g, dtype=dt)
    active = torch.ones(n, K, dtype=torch.bool)
    obs0 = s.observation(x, _ns_scene(goal, centers, radii, active))

    # ---- (a) global world-yaw by psi: obs must be bit-identical ----
    psi = torch.full((n,), 0.6, dtype=dt)
    c, sn = torch.cos(psi), torch.sin(psi)
    Rz = torch.zeros(n, 3, 3, dtype=dt)
    Rz[:, 0, 0] = c; Rz[:, 0, 1] = -sn; Rz[:, 1, 0] = sn; Rz[:, 1, 1] = c; Rz[:, 2, 2] = 1.0
    Rz2 = Rz[:, :2, :2]
    p2 = torch.einsum("bij,bj->bi", Rz, p)
    v2 = torch.einsum("bij,bj->bi", Rz, v)
    q2 = _quat_mul(_yaw_quat(psi, n, dt), q)                     # new attitude = Rz . R(q)
    x2 = torch.cat([p2, q2, v2, om], dim=1)                      # body rates omega unchanged
    goal2 = torch.einsum("bij,bj->bi", Rz, goal)
    centers2 = torch.einsum("bij,bkj->bki", Rz2, centers)
    obs_yaw = s.observation(x2, _ns_scene(goal2, centers2, radii, active))
    assert torch.allclose(obs0, obs_yaw, atol=1e-9), \
        f"yaw is not a symmetry of the observation (max {float((obs0 - obs_yaw).abs().max()):.2e})"

    # ---- (b) a pure tilt is OBSERVABLE: g^b (obs[9:12]) must change ----
    phi = 0.4
    q_tilt = torch.tensor([math.cos(phi / 2), math.sin(phi / 2), 0.0, 0.0], dtype=dt).expand(n, 4)
    x3 = x.clone()
    x3[:, 3:7] = _quat_mul(q_tilt, q)                            # tilt the attitude only
    obs_tilt = s.observation(x3, _ns_scene(goal, centers, radii, active))
    assert not torch.allclose(obs0[:, 9:12], obs_tilt[:, 9:12], atol=1e-6), "tilt left g^b unchanged"


# ---------------------------------------------------------------------------- q4
def test_q4_obs_layout_indices():
    s = _sys()
    assert s.obs_dim == 32
    dt = torch.float64
    n, K = 3, 5
    p = torch.tensor([[0.2, -0.1, 0.5]], dtype=dt).repeat(n, 1)
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dt).repeat(n, 1)  # identity: body frame == world frame
    v = torch.tensor([[0.3, -0.7, 0.9]], dtype=dt).repeat(n, 1)
    om = torch.tensor([[0.1, 0.2, -0.3]], dtype=dt).repeat(n, 1)
    x = torch.cat([p, q, v, om], dim=1)
    goal = torch.tensor([[1.0, 1.0, 2.0]], dtype=dt).repeat(n, 1)
    centers = torch.zeros(n, K, 2, dtype=dt)
    centers[:, 0] = torch.tensor([1.2, 0.3])                     # nearest cylinder
    centers[:, 1] = torch.tensor([-2.0, -2.0])
    radii = torch.zeros(n, K, dtype=dt); radii[:, 0] = 0.4; radii[:, 1] = 0.5
    active = torch.zeros(n, K, dtype=torch.bool); active[:, 0] = True; active[:, 1] = True
    obs = s.observation(x, _ns_scene(goal, centers, radii, active))
    assert obs.shape == (n, 32)
    assert torch.allclose(obs[:, 0:3], v, atol=1e-9)             # v^b == v (identity attitude)
    assert torch.allclose(obs[:, 3:6], om, atol=1e-9)            # omega^b
    assert torch.allclose(obs[:, 6:9], goal - p, atol=1e-9)      # goal^b
    assert torch.allclose(obs[:, 9:12], torch.tensor([0.0, 0.0, -1.0], dtype=dt).expand(n, 3), atol=1e-9)  # g^b
    # first cylinder block: c_off^b = (Dc_xy, 0), r ; nearest is centers[0]
    dc = torch.tensor([1.2 - 0.2, 0.3 - (-0.1), 0.0], dtype=dt).expand(n, 3)
    assert torch.allclose(obs[:, 12:15], dc, atol=1e-9)
    assert torch.allclose(obs[:, 15], torch.full((n,), 0.4, dtype=dt), atol=1e-9)


# ---------------------------------------------------------------------------- q5
def test_q5_golden_bit_parity():
    # DI + unicycle: value_target_barrier == signed_h (no approach term)
    for sysname in ("double_integrator", "unicycle"):
        s, x, bs, cfg = _batch(n=24, dtype=torch.float64, system=sysname, seed=5)
        h = value_target_barrier(s, x, bs, cfg)
        ref = signed_h(s.position(x)[..., :2], bs, float(cfg["env"]["h_scale"]))
        assert torch.equal(h, ref), f"{sysname}: value_target_barrier diverged from signed_h"
    # planar: h_star == phi + c * (v^T Re), bit-identical to the pre-generalization module formula
    from src.common.quadrotor_barrier import approach_speed, phi_value
    s, x, bs, cfg = _batch(n=24, dtype=torch.float64, system="quadrotor_planar", seed=6)
    h = value_target_barrier(s, x, bs, cfg)
    c = float(cfg["env"]["quadrotor_planar"]["c_gain"])
    ref = phi_value(x, bs, float(cfg["env"]["h_scale"])) + c * approach_speed(x)
    assert torch.equal(h, ref), "planar h_star label changed under the interface generalization"


# ---------------------------------------------------------------------------- q6
def test_q6_ic_attitude_distribution():
    cfg = _cfg()
    rng = np.random.default_rng(2026)
    n = 6000
    scenes = [sample_train_scene(rng, cfg, "quadrotor_3d") for _ in range(n)]
    quats = np.stack([sc.initial_attitude_quat for sc in scenes])
    q = torch.tensor(quats, dtype=torch.float64)
    R = _R(q)
    b3 = R[:, :, 2]                                              # body-up in world
    cosang = b3[:, 2].clamp(-1.0, 1.0)
    phi = torch.arccos(cosang).numpy()                          # tilt angle == phi_tilt
    # SET-ONCE: no inversion, tilt in [0, 2pi/3]
    assert phi.min() >= -1e-9 and phi.max() <= 2 * math.pi / 3 + 1e-6, (phi.min(), phi.max())
    # past-horizontal mass ~ 25% (phi in (pi/2, 2pi/3), out of [0,2pi/3] uniform -> (2pi/3-pi/2)/(2pi/3)=0.25)
    assert 0.21 <= float((phi > math.pi / 2).mean()) <= 0.29
    assert abs(float(phi.mean()) - math.pi / 3) < 0.04          # U[0,2pi/3] mean = pi/3
    hist, _ = np.histogram(phi, bins=6, range=(0, 2 * math.pi / 3))
    assert hist.min() > 0.75 * n / 6 and hist.max() < 1.25 * n / 6  # coarse uniformity
    # tilt-axis azimuth uniform: b3_xy = sin(phi)(sin a, -cos a) -> azimuth uniform iff a uniform
    b3xy = b3[:, :2].numpy()
    az = np.arctan2(b3xy[:, 1], b3xy[:, 0])
    assert abs(np.cos(az).mean()) < 0.05 and abs(np.sin(az).mean()) < 0.05
    # yaw uniform: de-tilt R by R_a(phi)^T then extract yaw from the residual R_z(psi)
    a_az = np.arctan2(b3xy[:, 0], -b3xy[:, 1])                  # tilt-axis azimuth alpha
    Rn = R.numpy(); psis = np.empty(n)
    for i in range(n):
        ph, al = phi[i], a_az[i]
        ax = np.array([math.cos(al), math.sin(al), 0.0])
        Kx = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        Ra = np.eye(3) + math.sin(ph) * Kx + (1 - math.cos(ph)) * (Kx @ Kx)
        Rz = Ra.T @ Rn[i]
        psis[i] = math.atan2(Rz[1, 0], Rz[0, 0])
    assert abs(np.cos(psis).mean()) < 0.05 and abs(np.sin(psis).mean()) < 0.05


# ---------------------------------------------------------------------------- q7
def test_q7_convention_cross_checks():
    s = _sys()
    dt = torch.float64
    # (a) R^T R = I
    q = _rand_quat(64, dtype=dt, seed=21)
    R = _R(q)
    RtR = torch.einsum("bji,bjk->bik", R, R)
    assert torch.allclose(RtR, torch.eye(3, dtype=dt).expand(64, 3, 3), atol=1e-10)
    # (b) g^b for a known tilt phi about world-x: world-up-in-body angle to body-up == phi
    from src.envs.quadrotor_3d import _quat_mul
    phi = 0.5
    qt = torch.tensor([[math.cos(phi / 2), math.sin(phi / 2), 0.0, 0.0]], dtype=dt)
    Rt = _R(qt)
    g_b = torch.einsum("bji,j->bi", Rt, torch.tensor([0.0, 0.0, -1.0], dtype=dt))  # R^T(-e3)
    ang = torch.arccos((-g_b[:, 2]).clamp(-1, 1))               # angle(world-up-in-body, body-up e3)
    assert torch.allclose(ang, torch.tensor([phi], dtype=dt), atol=1e-9)
    assert torch.allclose(torch.linalg.norm(g_b, dim=1), torch.ones(1, dtype=dt), atol=1e-12)
    # (c) free fall (f=0): dv = -g e3 exactly
    x = torch.zeros(4, 13, dtype=dt); x[:, 3] = 1.0; x[:, 7:10] = torch.randn(4, 3)
    u = torch.zeros(4, 4, dtype=dt); u[:, 0] = 0.0
    dv = s.dynamics(x, u)[:, 7:10]
    assert torch.allclose(dv, torch.tensor([0.0, 0.0, -s.gravity], dtype=dt).expand(4, 3), atol=1e-12)
    # (d) hover at identity: f=mg -> dv = 0
    u[:, 0] = s.mass * s.gravity
    dv = s.dynamics(x, u)[:, 7:10]
    assert torch.allclose(dv, torch.zeros(4, 3, dtype=dt), atol=1e-12)
    # (e) e3 torque -> yaw only + g^b invariant
    x = torch.zeros(1, 13, dtype=dt); x[:, 3] = 1.0
    u = torch.zeros(1, 4, dtype=dt); u[:, 0] = s.mass * s.gravity; u[:, 3] = 0.5   # tau_z only
    for _ in range(10):
        x = rk4_step(s, x, u, 0.02)
    b3 = _R(x[:, 3:7])[:, :, 2]
    assert torch.allclose(b3, torch.tensor([0.0, 0.0, 1.0], dtype=dt).expand(1, 3), atol=1e-6)  # still upright
    g_b = torch.einsum("bji,j->bi", _R(x[:, 3:7]), torch.tensor([0.0, 0.0, -1.0], dtype=dt))
    assert torch.allclose(g_b, torch.tensor([0.0, 0.0, -1.0], dtype=dt).expand(1, 3), atol=1e-6)
    # (f) nominal recovers hover from a 30-deg tilt
    phi0 = math.radians(30.0)
    x = torch.zeros(1, 13, dtype=dt)
    x[:, 3:7] = torch.tensor([[math.cos(phi0 / 2), math.sin(phi0 / 2), 0.0, 0.0]], dtype=dt)
    goal = torch.zeros(1, 3, dtype=dt)
    tilt0 = math.acos(float(_R(x[:, 3:7])[0, 2, 2]))
    for _ in range(400):
        u = s.lqr_action(x, goal)
        x = rk4_step(s, x, u, 0.01)
    tilt_final = math.acos(min(1.0, float(_R(x[:, 3:7])[0, 2, 2])))
    assert tilt_final < 0.05 and tilt_final < tilt0, f"nominal did not recover hover (tilt {tilt0:.3f}->{tilt_final:.3f})"


# ---------------------------------------------------------------------------- q8
def test_q8_autograd_matches_finite_difference():
    s = _sys()
    dt = 0.05
    # moderate state so wrap_state acts smoothly (|v|<v_max, |omega|<omega_max, q near unit)
    x0 = torch.tensor([[0.1, -0.2, 0.3, 1.0, 0.05, -0.03, 0.02, 0.4, -0.3, 0.2, 0.1, -0.2, 0.15]],
                      dtype=torch.float64)
    x0[:, 3:7] = x0[:, 3:7] / torch.linalg.norm(x0[:, 3:7], dim=1, keepdim=True)
    u0 = torch.tensor([[9.81, 0.02, -0.01, 0.03]], dtype=torch.float64)

    def step(xv):
        return rk4_step(s, xv, u0, dt).reshape(-1)

    J_auto = torch.autograd.functional.jacobian(step, x0, vectorize=True).reshape(13, 13)
    eps = 1e-6
    J_fd = torch.zeros(13, 13, dtype=torch.float64)
    base = step(x0)
    for j in range(13):
        xp = x0.clone(); xp[0, j] += eps
        xm = x0.clone(); xm[0, j] -= eps
        J_fd[:, j] = (step(xp) - step(xm)) / (2 * eps)
    err = float((J_auto - J_fd).abs().max())
    assert err < 1e-5, f"autograd vs finite-diff Jacobian mismatch (max {err:.2e})"
