"""S' successor-verification shield (eval-time only; never used at training time).

Per control step: the filter proposes a candidate u_safe; the shield verifies the plan
[one dt_ctrl candidate segment + m_0 brake-to-rest from the successor] by sampling GEOMETRIC clearance
min_i(dist_i - r_i) inside each ZOH segment. The candidate/brake segments are sub-integrated with the
ACTUAL system dynamics (rk4_step) every dt_check, so the check is exact for the DI straight/quadratic path
and for the unicycle arc alike. The plan passes iff every sampled clearance >= thresh, with

    thresh = v_max(system) * dt_check / 2          (02_control §8 grid-excursion bound; DERIVED, not literal)

On failure the current-state m_0 (deadband brake, system-dispatched) action is applied — its plan is the
verified tail of the previous step's check (verified-tail invariant), so verified-start episodes stay
collision-free independently of V_hat accuracy.

Fully batched over the pool. Records per episode: verified_start, n_checks, n_overrides, and a collision
flag among verified-start episodes (must be 0 -> P-I1-2). System-generic: the brake, the speed test, and
the segment integration all dispatch on `system` (backup.py + system.dynamics/system.speed).
"""
from __future__ import annotations

from typing import Any

import torch

from src.common.filter_hardnet import (_base_alpha, _base_projection, _box_aware_projection, _cbf_terms,
                                        _hardnet_params, _SINGULAR_LG_THRESHOLD)
from src.common.rk4 import rk4_step
from src.frameworks.cpi.backup import deadband_brake, speed_max

Tensor = torch.Tensor


def _seg_min_clearance(x0, u, system, dt_ctrl, dt_check, C, R, A):
    """Min geometric clearance min_i(dist_i - r_i) over the ZOH segment (0, dt_ctrl], sub-integrated with the
    ACTUAL dynamics via rk4_step every dt_check (u held constant; endpoint included, t=0 excluded — covered by
    the previous step). System-generic: DI gives the quadratic path, unicycle the constant-(a,omega) arc."""
    n = max(1, int(round(dt_ctrl / dt_check)))
    best = torch.full((x0.shape[0],), float("inf"), device=x0.device, dtype=x0.dtype)
    x = x0
    for _ in range(n):
        x = rk4_step(system, x, u, dt_check)                            # sub-step; ZOH holds u
        dist = torch.linalg.norm(x[:, :2].unsqueeze(1) - C, dim=-1)     # [B,K]
        clr = torch.where(A, dist - R, torch.full_like(dist, float("inf")))
        best = torch.minimum(best, clr.amin(dim=1))
    return best


def verify_plan(x0, u_cand, C, R, A, system, config, dt_ctrl, dt_check, t_brake, thresh):
    """Verify [candidate segment + m_0 brake-to-rest from the successor]. Returns pass mask [B]. The brake
    and the rest test dispatch on `system` (backup.deadband_brake, system.speed)."""
    mc = _seg_min_clearance(x0, u_cand, system, dt_ctrl, dt_check, C, R, A)   # candidate segment
    ok = mc >= thresh
    x = rk4_step(system, x0, u_cand, dt_ctrl)                                 # exact ZOH successor
    for _ in range(t_brake):
        if bool((system.speed(x) <= 1e-9).all()):
            break
        u = deadband_brake(x, system, config, dt_ctrl)
        mc = _seg_min_clearance(x, u, system, dt_ctrl, dt_check, C, R, A)
        ok = ok & (mc >= thresh)
        x = rk4_step(system, x, u, dt_ctrl)
    return ok


def shield_eval(scenes, policy_net, config, h_fn, system, device, *, dt_check=None, t_brake=40,
                dtype=torch.float32, chunk=250, ladder=None):
    """Filter-then-shield closed-loop eval over `scenes`. Returns dict: S [T+1,N,state_dim], outcomes,
    event_step, verified_start [N], n_overrides [N], n_checks [N]. Filter = cpi h_fn projection (HardNet base).

    dt_check: sub-integration resolution; None -> config `filter.shield.dt_check` else 0.01. thresh is
    DERIVED from `v_max(system) * dt_check / 2` (not the v2.5.1 0.0125 literal). ladder=None (default)
    reproduces the arm-B shield exactly (bit-identical control law: on failure apply the m_0 brake).
    ladder=[0.75,0.5,0.25] enables the MINIMAL-INTERVENTION fallback (arm-B'): on failure, apply the largest
    verified blend u = lam*u_cand + (1-lam)*u_brake, decreasing lam, first pass wins, else pure brake. Every
    applied action is fully verified before use, so the verified-tail invariant (P-I1-2) is unchanged."""
    from src.common.outcomes import resolve_outcome, step_outcomes
    from src.envs.scene_batch import batch_scenes, initial_states_from_batch
    dt = float(config["env"]["dt"]); max_steps = int(config["eval"]["max_steps"])
    if dt_check is None:
        dt_check = float(config.get("filter", {}).get("shield", {}).get("dt_check", 0.01))
    thresh = speed_max(system, config) * dt_check / 2.0                       # 02_control §8, DERIVED
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
        vs = verify_plan(x, deadband_brake(x, system, config, dt), C, R, A, system, config, dt, dt_check, t_brake, thresh)
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
                ok = verify_plan(x, u_cand, C, R, A, system, config, dt, dt_check, t_brake, thresh)
                u_brake = deadband_brake(x, system, config, dt)
                u = torch.where(ok.unsqueeze(1), u_cand, u_brake)
                n_ovr += (~ok).long(); n_chk += 1
                if ladder:
                    # minimal-intervention fallback: for failed episodes, take the largest verified blend.
                    need = ~ok
                    for lam in ladder:
                        if not bool(need.any()):
                            break
                        u_lam = lam * u_cand + (1.0 - lam) * u_brake
                        ok_lam = need & verify_plan(x, u_lam, C, R, A, system, config, dt, dt_check, t_brake, thresh)
                        u = torch.where(ok_lam.unsqueeze(1), u_lam, u)
                        need = need & ~ok_lam
                        n_chk += 1
                x = rk4_step(system, x, u, dt); states.append(x)
        S = torch.stack(states, 0); masks = step_outcomes(S, bs, system, config); res = resolve_outcome(masks)
        allS.append(S.cpu()); outc += res.outcome; ev.append(res.event_step.cpu())
        vst.append(vs.cpu()); ovr.append(n_ovr.cpu()); nch.append(n_chk.cpu())
    return {"S": torch.cat(allS, 1), "outcome": outc, "event_step": torch.cat(ev),
            "verified_start": torch.cat(vst), "n_overrides": torch.cat(ovr), "n_checks": torch.cat(nch)}
