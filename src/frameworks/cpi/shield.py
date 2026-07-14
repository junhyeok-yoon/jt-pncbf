"""S' successor-verification shield (eval-time only; never used at training time).

Per control step: the filter proposes a candidate u_safe; the shield verifies the plan
[one dt_ctrl candidate segment + m_0 brake-to-rest from the successor] by sampling GEOMETRIC clearance
min_i(dist_i - r_i) at dt_check = 0.01 inside each ZOH segment (closed-form quadratic position). The plan
passes iff min clearance >= v_max*dt_check/2 = 0.0125 m. On failure the current-state m_0 (deadband brake)
action is applied — its plan is the verified tail of the previous step's check (verified-tail invariant),
so verified-start episodes stay collision-free independently of V_hat accuracy.

Fully batched over the pool. Records per episode: verified_start, n_checks, n_overrides, and a collision
flag among verified-start episodes (must be 0 -> P-I1-2).
"""
from __future__ import annotations

import math
from typing import Any

import torch

from src.common.filter_hardnet import (_base_alpha, _base_projection, _box_aware_projection, _cbf_terms,
                                        _hardnet_params, _SINGULAR_LG_THRESHOLD)
from src.common.rk4 import rk4_step

Tensor = torch.Tensor


def _deadband(v, u_max, dt):
    return torch.where(v.abs() > u_max * dt, -u_max * torch.sign(v), -v / dt)


def _seg_min_clearance(p0, v0, u, dt, dt_check, C, R, A):
    """Min over the ZOH segment [0,dt] (sampled every dt_check) of min_i(dist_i - r_i). Quadratic position
    p(t)=p0+v0 t+0.5 u t^2. p0/v0/u [B,2]; C[B,K,2] R[B,K] A[B,K] -> [B] min clearance over the segment."""
    n = max(1, int(round(dt / dt_check)))
    ts = torch.linspace(dt_check, dt, n, device=p0.device, dtype=p0.dtype)  # sub-samples (exclude t=0; covered by prev step)
    best = torch.full((p0.shape[0],), float("inf"), device=p0.device, dtype=p0.dtype)
    for t in ts:
        p = p0 + v0 * t + 0.5 * u * (t * t)
        dist = torch.linalg.norm(p.unsqueeze(1) - C, dim=-1)             # [B,K]
        clr = torch.where(A, dist - R, torch.full_like(dist, float("inf")))
        best = torch.minimum(best, clr.amin(dim=1))
    return best


def verify_plan(x0, u_cand, C, R, A, system, dt_ctrl, dt_check, t_brake, thresh, u_max, v_max):
    """Verify [candidate segment + m_0 brake-to-rest from the successor]. Returns pass mask [B]."""
    p0 = x0[:, :2]; v0 = x0[:, 2:4]
    mc = _seg_min_clearance(p0, v0, u_cand, dt_ctrl, dt_check, C, R, A)   # candidate segment
    ok = mc >= thresh
    x = rk4_step(system, x0, u_cand, dt_ctrl)                            # exact ZOH successor
    for _ in range(t_brake):
        v = x[:, 2:4]
        if bool((torch.linalg.norm(v, dim=1) <= 1e-9).all()):
            break
        u = _deadband(v, u_max, dt_ctrl)
        mc = _seg_min_clearance(x[:, :2], v, u, dt_ctrl, dt_check, C, R, A)
        ok = ok & (mc >= thresh)
        x = rk4_step(system, x, u, dt_ctrl)
    return ok


def shield_eval(scenes, policy_net, config, h_fn, system, device, *, dt_check=0.01, thresh=0.0125,
                t_brake=40, dtype=torch.float32, chunk=250, ladder=None):
    """Filter-then-shield closed-loop eval over `scenes`. Returns dict: S [T+1,N,4], outcomes, event_step,
    verified_start [N], n_overrides [N], n_checks [N]. Filter = cpi h_fn projection (HardNet base).

    ladder=None (default) reproduces the arm-B shield exactly (bit-identical): on verification failure the
    m_0 brake is applied. ladder=[0.75,0.5,0.25] enables the MINIMAL-INTERVENTION fallback (arm-B'): on
    failure, try the largest blend u = lam*u_cand + (1-lam)*u_brake that FULLY verifies (segment +
    brake-from-successor), decreasing lam; the first pass wins; else pure brake. Every applied action is
    fully verified before use, so the verified-tail invariant (hence P-I1-2) is unchanged; the ladder only
    reduces over-braking (stuck)."""
    from src.common.outcomes import resolve_outcome, step_outcomes
    from src.envs.scene_batch import batch_scenes, initial_states_from_batch
    dt = float(config["env"]["dt"]); u_max = float(config["env"]["bounds"]["double_integrator"]["u_max"])
    v_max = float(config["env"]["bounds"]["double_integrator"]["v_max"]); max_steps = int(config["eval"]["max_steps"])
    params = _hardnet_params(config); bounds = system.u_bounds
    allS, outc, ev, vst, ovr, nch = [], [], [], [], [], []
    for s0 in range(0, len(scenes), chunk):
        bs = batch_scenes(scenes[s0:s0 + chunk], device=device, dtype=dtype)
        x = initial_states_from_batch(bs).to(dtype); B = x.shape[0]
        C = torch.as_tensor(bs.obstacle_centers, dtype=dtype, device=device)
        R = torch.as_tensor(bs.obstacle_radii, dtype=dtype, device=device)
        A = torch.as_tensor(bs.obstacle_active, dtype=torch.bool, device=device)
        if C.ndim == 2:
            C = C.expand(B, -1, -1); R = R.expand(B, -1); A = A.expand(B, -1)
        vs = verify_plan(x, _deadband(x[:, 2:4], u_max, dt), C, R, A, system, dt, dt_check, t_brake, thresh, u_max, v_max)
        n_ovr = torch.zeros(B, dtype=torch.long, device=device); n_chk = torch.zeros(B, dtype=torch.long, device=device)
        states = [x]
        with torch.no_grad():
            for _ in range(max_steps):
                un = policy_net(system.observation(x, bs))
                h, lf, lg = _cbf_terms(system, h_fn, x, bs, un, create_graph=False)
                h, lf, lg = h.detach(), lf.detach(), lg.detach()
                alpha = _base_alpha(h, params); row = -lf - alpha * h
                proj = _base_projection(un, lg, row, bounds, params)
                u_cand, _ = _box_aware_projection(un, proj, lg, row, bounds)
                ok = verify_plan(x, u_cand, C, R, A, system, dt, dt_check, t_brake, thresh, u_max, v_max)
                u_brake = _deadband(x[:, 2:4], u_max, dt)
                u = torch.where(ok.unsqueeze(1), u_cand, u_brake)
                n_ovr += (~ok).long(); n_chk += 1
                if ladder:
                    # minimal-intervention fallback: for failed episodes, take the largest verified blend.
                    need = ~ok
                    for lam in ladder:
                        if not bool(need.any()):
                            break
                        u_lam = lam * u_cand + (1.0 - lam) * u_brake
                        ok_lam = need & verify_plan(x, u_lam, C, R, A, system, dt, dt_check, t_brake,
                                                    thresh, u_max, v_max)
                        u = torch.where(ok_lam.unsqueeze(1), u_lam, u)
                        need = need & ~ok_lam
                        n_chk += 1
                x = rk4_step(system, x, u, dt); states.append(x)
        S = torch.stack(states, 0); masks = step_outcomes(S, bs, system, config); res = resolve_outcome(masks)
        allS.append(S.cpu()); outc += res.outcome; ev.append(res.event_step.cpu())
        vst.append(vs.cpu()); ovr.append(n_ovr.cpu()); nch.append(n_chk.cpu())
    return {"S": torch.cat(allS, 1), "outcome": outc, "event_step": torch.cat(ev),
            "verified_start": torch.cat(vst), "n_overrides": torch.cat(ovr), "n_checks": torch.cat(nch)}
