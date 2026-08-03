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


def _sys_encoder(encoder, system="quadrotor_3d"):
    # v2.8.1 S1: build with an explicit obstacle encoder (per-system override).
    c = _cfg(system)
    c.setdefault("obs", {}).setdefault(system, {})["encoder"] = encoder
    return make_system(c)


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
    # v = omega = 0. v2.7.3 per-rotor plant: hover = equal rotor forces mg/4; wrench = (mg, 0, 0, 0).
    u = s.lqr_action(x, goal)
    assert torch.allclose(u, torch.full((n, 4), s.mass * s.gravity / 4.0, dtype=torch.float64), atol=1e-5)
    wrench = u @ s.mixer.to(u.dtype).t()
    assert torch.allclose(wrench[:, 0], torch.full((n,), s.mass * s.gravity, dtype=torch.float64), atol=1e-5)
    assert torch.allclose(wrench[:, 1:4], torch.zeros(n, 3, dtype=torch.float64), atol=1e-5)
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


# ------------------------------------------------------------- G7 (v2.8.1 S1: soft_topk invariance harness)
def test_g7_soft_topk_group_invariance_and_carried_z():
    """v2.8.1 S1 G7 (05_code s5.4). q3 / test_band_hazard pin the invariance harness on the hard path (and now
    run under the soft default only incidentally); this pins it EXPLICITLY on the deployed `soft_topk` encoder.
    Under soft_topk the dim-34 quadrotor_3d observation must be invariant over the retained symmetry group —
    horizontal (xy) translation of (state, goal, obstacles) and global world-yaw — even though the broad
    soft-rank mixing (beta = 1/(0.5 m)) makes the group act on every slot at once, not on a single argmin.
    Companion POSITIVE assertions: the carried vertical coordinates p_z and v_z (band_hazard left z OUT of the
    group, 05_code s5.4) stay OBSERVABLE — a state differing only in p_z, or only in v_z, changes the obs."""
    from src.envs.quadrotor_3d import _quat_mul
    s = _sys_encoder("soft_topk")
    assert s.encoder == "soft_topk" and s.obs_dim == 34
    dt = torch.float64
    n, K = 8, 5
    g = torch.Generator().manual_seed(23)
    p = torch.randn(n, 3, generator=g, dtype=dt)
    q = _rand_quat(n, dtype=dt, seed=29)
    v = torch.randn(n, 3, generator=g, dtype=dt)
    om = torch.randn(n, 3, generator=g, dtype=dt)
    x = torch.cat([p, q, v, om], dim=1)
    goal = torch.randn(n, 3, generator=g, dtype=dt)
    centers = 1.1 * torch.randn(n, K, 2, generator=g, dtype=dt)   # mostly within the d_c=3.0 m kernel -> slots mix
    radii = 0.3 + 0.2 * torch.rand(n, K, generator=g, dtype=dt)
    active = torch.ones(n, K, dtype=torch.bool)
    obs0 = s.observation(x, _ns_scene(goal, centers, radii, active))

    # (a) horizontal translation of the whole scene: obs invariant
    dxy = torch.tensor([0.7, -1.3, 0.0], dtype=dt)
    xt = x.clone(); xt[:, :3] += dxy
    obs_t = s.observation(xt, _ns_scene(goal + dxy, centers + dxy[:2], radii, active))
    assert torch.allclose(obs0, obs_t, atol=1e-9), \
        f"soft_topk: horizontal translation is not a symmetry (max {float((obs0 - obs_t).abs().max()):.2e})"

    # (b) global world-yaw by psi: obs invariant (p, v world-rotate; attitude gets the yaw; body rates unchanged)
    psi = torch.full((n,), 0.6, dtype=dt)
    cz, sz = torch.cos(psi), torch.sin(psi)
    Rz = torch.zeros(n, 3, 3, dtype=dt)
    Rz[:, 0, 0] = cz; Rz[:, 0, 1] = -sz; Rz[:, 1, 0] = sz; Rz[:, 1, 1] = cz; Rz[:, 2, 2] = 1.0
    xy = x.clone()
    xy[:, :3] = torch.einsum("bij,bj->bi", Rz, p)
    xy[:, 3:7] = _quat_mul(_yaw_quat(psi, n, dt), q)
    xy[:, 7:10] = torch.einsum("bij,bj->bi", Rz, v)
    goal_y = torch.einsum("bij,bj->bi", Rz, goal)
    centers_y = torch.einsum("bij,bkj->bki", Rz[:, :2, :2], centers)
    obs_y = s.observation(xy, _ns_scene(goal_y, centers_y, radii, active))
    assert torch.allclose(obs0, obs_y, atol=1e-9), \
        f"soft_topk: world-yaw is not a symmetry (max {float((obs0 - obs_y).abs().max()):.2e})"

    # (c) carried p_z is OBSERVABLE: a pure p_z shift changes the obs
    xz = x.clone(); xz[:, 2] += 1.6
    assert not torch.allclose(obs0, s.observation(xz, _ns_scene(goal, centers, radii, active)), atol=1e-6), \
        "soft_topk: p_z left the observation unchanged"
    # (d) carried v_z is OBSERVABLE: a pure v_z shift changes the obs
    xvz = x.clone(); xvz[:, 9] += 0.9
    assert not torch.allclose(obs0, s.observation(xvz, _ns_scene(goal, centers, radii, active)), atol=1e-6), \
        "soft_topk: v_z left the observation unchanged"


# ---------------------------------------------------------------------------- q4
def test_q4_obs_layout_indices():
    # v2.8.1 S1: the exact obstacle-block layout reference is the HARD encoder's slot-0 value (the reference
    # this assertion was derived from). soft_topk leaves the packing identical and changes only the block's
    # content; its packing is covered by test_q4b_obs_layout_soft_topk below.
    s = _sys_encoder("hard_topk")
    assert s.encoder == "hard_topk"
    assert s.obs_dim == 34                                       # v2.7.6: + p_z, v_z (obs_band_z)
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
    assert obs.shape == (n, 34)
    assert torch.allclose(obs[:, 0:3], v, atol=1e-9)             # v^b == v (identity attitude)
    assert torch.allclose(obs[:, 3:6], om, atol=1e-9)            # omega^b
    assert torch.allclose(obs[:, 6:9], goal - p, atol=1e-9)      # goal^b
    assert torch.allclose(obs[:, 9:12], torch.tensor([0.0, 0.0, -1.0], dtype=dt).expand(n, 3), atol=1e-9)  # g^b
    # first cylinder block: c_off^b = (Dc_xy, 0), r ; nearest is centers[0]
    dc = torch.tensor([1.2 - 0.2, 0.3 - (-0.1), 0.0], dtype=dt).expand(n, 3)
    assert torch.allclose(obs[:, 12:15], dc, atol=1e-9)
    assert torch.allclose(obs[:, 15], torch.full((n,), 0.4, dtype=dt), atol=1e-9)
    # v2.7.6 obs_band_z: last two slots are absolute p_z, v_z
    assert torch.allclose(obs[:, 32], torch.full((n,), 0.5, dtype=dt), atol=1e-9)   # p_z
    assert torch.allclose(obs[:, 33], torch.full((n,), 0.9, dtype=dt), atol=1e-9)   # v_z


# ---------------------------------------------------------------------------- q4b
def test_q4b_obs_layout_soft_topk():
    # v2.8.1 S1: packing stays covered under the DEPLOYED encoder. Same obs_dim and block boundaries; the
    # obstacle block [12:16] responds to the nearest active cylinder. With a SINGLE active cylinder inside the
    # distance kernel (sigma == 1), the soft-rank slot 0 collapses to that cylinder exactly, so the layout is
    # asserted without duplicating the soft-mixture test (test_soft_topk_encoder.py).
    s = _sys_encoder("soft_topk")
    assert s.encoder == "soft_topk"
    assert s.obs_dim == 34
    dt = torch.float64
    n, K = 3, 5
    p = torch.tensor([[0.2, -0.1, 0.5]], dtype=dt).repeat(n, 1)
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dt).repeat(n, 1)  # identity attitude
    v = torch.tensor([[0.3, -0.7, 0.9]], dtype=dt).repeat(n, 1)
    om = torch.tensor([[0.1, 0.2, -0.3]], dtype=dt).repeat(n, 1)
    x = torch.cat([p, q, v, om], dim=1)
    goal = torch.tensor([[1.0, 1.0, 2.0]], dtype=dt).repeat(n, 1)
    centers = torch.zeros(n, K, 2, dtype=dt)
    centers[:, 0] = torch.tensor([1.2, 0.3])                        # the single active cylinder (surf dist 0.68 < d_c)
    radii = torch.zeros(n, K, dtype=dt); radii[:, 0] = 0.4
    active = torch.zeros(n, K, dtype=torch.bool); active[:, 0] = True
    obs = s.observation(x, _ns_scene(goal, centers, radii, active))
    assert obs.shape == (n, 34)                                     # layout unchanged
    assert torch.allclose(obs[:, 0:3], v, atol=1e-9)               # v^b
    assert torch.allclose(obs[:, 3:6], om, atol=1e-9)              # omega^b
    assert torch.allclose(obs[:, 6:9], goal - p, atol=1e-9)       # goal^b
    assert torch.allclose(obs[:, 9:12], torch.tensor([0.0, 0.0, -1.0], dtype=dt).expand(n, 3), atol=1e-9)  # g^b
    dc = torch.tensor([1.2 - 0.2, 0.3 - (-0.1), 0.0], dtype=dt).expand(n, 3)
    assert torch.allclose(obs[:, 12:15], dc, atol=1e-9)           # block [12:15] responds to the nearest cylinder
    assert torch.allclose(obs[:, 15], torch.full((n,), 0.4, dtype=dt), atol=1e-9)  # its radius
    # empty slots 1..k-1 zero-pad exactly (only one active obstacle)
    assert torch.allclose(obs[:, 16:32], torch.zeros(n, 16, dtype=dt), atol=1e-9)


def test_q4c_encoder_resolves_hard_default_for_predating_checkpoint():
    # v2.8.1 S1 (checkpoint compatibility, obs_band_z precedent): a system built from a config that PREDATES
    # the encoder key resolves to hard_topk (so a v2.8.0 checkpoint never silently re-scores under soft_topk);
    # an explicit per-system override selects soft_topk; a global-only key applies with no per-system override.
    c_old = _cfg("quadrotor_3d"); c_old.pop("obs", None)
    assert make_system(c_old).encoder == "hard_topk"
    c_soft = _cfg("quadrotor_3d"); c_soft.setdefault("obs", {}).setdefault("quadrotor_3d", {})["encoder"] = "soft_topk"
    assert make_system(c_soft).encoder == "soft_topk"
    c_glob = _cfg("quadrotor_3d"); c_glob["obs"] = {"encoder": "hard_topk"}
    assert make_system(c_glob).encoder == "hard_topk"


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
    # v2.7.4 M0d: this pins the LEGACY tilt-capped sampler (ic_so3 off), which still exists and was proven
    # byte-identical to v2.7.3 in M0c. The v2.7.4 full-SO(3) sampler (ic_so3 on) is covered by
    # tests/test_v274_ic.py (cos(tilt) ~ U[-1,1] KS test). Force the switch off here so this regression test
    # continues to exercise the sampler it was written for regardless of the committed default.
    cfg = _cfg()
    cfg["env"]["quadrotor_3d"]["ic_so3"] = False
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
    # v2.7.3 per-rotor plant: actions are motor forces; build them from a desired wrench via the mixer inverse.
    def _rotor(wrench):                                          # wrench (…,4)=(f_thr,tau) -> per-rotor forces
        return wrench.to(dt) @ s.mixer_inv.to(dt).t()
    # (c) free fall (all motors off): dv = -g e3 exactly
    x = torch.zeros(4, 13, dtype=dt); x[:, 3] = 1.0; x[:, 7:10] = torch.randn(4, 3)
    u = torch.zeros(4, 4, dtype=dt)                              # f_i = 0 -> f_thr = 0
    dv = s.dynamics(x, u)[:, 7:10]
    assert torch.allclose(dv, torch.tensor([0.0, 0.0, -s.gravity], dtype=dt).expand(4, 3), atol=1e-12)
    # (d) hover at identity: equal rotor forces mg/4 -> f_thr = mg, tau = 0 -> dv = 0
    u = torch.full((4, 4), s.mass * s.gravity / 4.0, dtype=dt)
    dv = s.dynamics(x, u)[:, 7:10]
    assert torch.allclose(dv, torch.zeros(4, 3, dtype=dt), atol=1e-12)
    # (e) e3 (yaw) torque at hover thrust -> yaw only + g^b invariant
    x = torch.zeros(1, 13, dtype=dt); x[:, 3] = 1.0
    u = _rotor(torch.tensor([[s.mass * s.gravity, 0.0, 0.0, 0.05]]))   # wrench (mg,0,0,tau_z)
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
