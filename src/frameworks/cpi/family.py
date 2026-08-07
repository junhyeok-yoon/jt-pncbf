"""Family labeler v2 (bail-out-min family).

V_k(x) = min over {m_0} ∪ {j-step prefixes of each family transport, then m_0 brake-to-rest}, j = 0..T
(T = 40). h is the UNCLIPPED signed_h ramp on the dt_vm grid (identical to iteration-0). The transport is
the DEPLOY-time filtered policy (filter_{V_hat} ∘ pi) so label transport == deployed transport exactly.

Efficient form: roll each transport once (running-max h over the prefix); at each prefix state take the
m_0 brake value; plan value at prefix j = max(running-max-h[0..j], brake-value(x_j)); V_k = min over j and
transports. The j=0 term IS the m_0 value (= iteration-0 V_raw). Bail-out-min (not fixed-time switching) is
required for shift-closure (02_control §8).
"""
from __future__ import annotations

import torch

from src.common.filter_hardnet import (_base_alpha, _row_upper, _base_projection, _box_aware_projection, _cbf_terms,
                                        _hardnet_params)
from src.frameworks.cpi.labels import h_raw_position, m0_value_raw

Tensor = torch.Tensor


def _deadband(v, u_max, dt):
    return torch.where(v.abs() > u_max * dt, -u_max * torch.sign(v), -v / dt)


def _batched_scene(C, R, A, G):
    from types import SimpleNamespace
    return SimpleNamespace(obstacle_centers=C, obstacle_radii=R, obstacle_active=A, goal=G)


def build_transports(pairs, system, config):
    """Build the CUMULATIVE family transport list T_1..T_k from ordered (vhat_ckpt, pi_ckpt) pairs.

    Cumulative retention: family_k = {m_0} ∪ {T_1, ..., T_k}, T_j = filter_{V_hat_{j-1}} ∘ pi_j (the deploy
    pair of iteration j). This is REQUIRED for shift-closure and cross-iteration monotonicity: single-step
    retention (keeping only the two most recent transports) drops T_1..T_{k-2}, so family_k ⊉ family_{k-1}
    and V_k = min(family_k) can EXCEED V_{k-1} (the k=3 P-I1-3 HALT). Each pair is
    (V_hat_{j-1} checkpoint used as the deployed CBF, pi_j policy checkpoint); index 0 is T_1.
    """
    from src.common.control_net import ControlNet
    from src.frameworks.cpi.channel import make_cpi_h_fn

    dev = system.u_bounds.device
    dt = system.u_bounds.dtype
    odim = system.obs_dim + (system.action_dim
                             if config.get("loss", {}).get("policy", {}).get("obs_deficit_feedback") else 0)
    transports = []
    for vhat_ckpt, pi_ckpt in pairs:
        pol = ControlNet(odim, system, config).to(device=dev, dtype=dt)
        pol.load_state_dict(torch.load(pi_ckpt, map_location=dev, weights_only=False)["pi_state"])
        pol.eval()
        transports.append((pol, make_cpi_h_fn(vhat_ckpt, system)))
    return transports


def family_value(states, C, R, A, G, transports, system, config, *, t_bailout=40, chunk=65536):
    """V_k over the bail-out family. states [N,4]; per-state scenes [N,K,*], G [N,2]. transports = list of
    (policy_net, h_fn) deploy pairs. Returns V_k [N] (unclipped, min over the family)."""
    dt = float(config["env"]["dt"])   # (v2.5.1's u_max read here was dead — brake goes through m0_value_raw)
    h_scale = float(config["env"]["h_scale"]); params = _hardnet_params(config); bounds = system.u_bounds
    out = []
    for s0 in range(0, states.shape[0], chunk):
        sl = slice(s0, s0 + chunk)
        x0 = states[sl]; Cs, Rs, As, Gs = C[sl], R[sl], A[sl], G[sl]
        # j=0 member: m_0 brake value from x0 (unclipped) == iteration-0 V_raw
        V = m0_value_raw(x0, Cs, Rs, As, system, config, dt)
        for policy_net, h_fn in transports:
            scn = _batched_scene(Cs, Rs, As, Gs)
            x = x0.clone()
            rm_h = h_raw_position(x[:, :2], Cs, Rs, As, h_scale)                  # running max over transport prefix, h(x0)
            with torch.no_grad():
                for _ in range(t_bailout):
                    un = policy_net(system.observation(x, scn))
                    h, lf, lg = _cbf_terms(system, h_fn, x, scn, un, create_graph=False)
                    h, lf, lg = h.detach(), lf.detach(), lg.detach()
                    alpha = _base_alpha(h, params); row = _row_upper(lf, alpha, h, params)
                    proj = _base_projection(un, lg, row, bounds, params)
                    u, _ = _box_aware_projection(un, proj, lg, row, bounds)   # deployed transport action
                    from src.common.rk4 import rk4_step
                    x = rk4_step(system, x, u, dt)
                    rm_h = torch.maximum(rm_h, h_raw_position(x[:, :2], Cs, Rs, As, h_scale))
                    brake_j = m0_value_raw(x, Cs, Rs, As, system, config, dt)   # brake-to-rest from x_j
                    V = torch.minimum(V, torch.maximum(rm_h, brake_j))
        out.append(V)
    return torch.cat(out)
