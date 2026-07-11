"""Stage R (candidate) — certified reach-time R(x), training-free (analysis-only; production untouched).

R(x) = min duration over maneuver-sequence plans (goal-aligned thrust segments + MANDATORY terminal
deadband brake), each plan verified collision-free by EXACT rollout (same RK4 + ZOH + velocity clamp +
signed_h as V_M). Grid dt=dt_Vm=0.05, gamma_m=0.02, T_stop=25. Plan dimension is fused (repeat_interleave)
like maneuver_value fuses |M| — no Python loops over batch or plans.

Frame at (p,g): e_par=(g-p)/||g-p|| (world-axis fallback when degenerate), e_perp=rot90(e_par);
D={+e_par,-e_par,+e_perp,-e_perp} FIXED at plan time. Primitive (d,j): u=u_max*d for j steps, j in J_grid.
Plan depth-k: <=k primitives then m_0 brake. Admitted iff k_g (first step with ||p-g||<=0.15 AND
speed<=0.30) exists AND h<=-gamma_m for all k<=k_g AND in-arena. dur=dt*k_g. R=min dur over admitted plans.
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from src.common.rk4 import rk4_step
from src.common.signed_h import signed_h

J_GRID = (2, 4, 6, 8, 12, 16, 20)
GOAL_R = 0.15
GOAL_SPD = 0.30
GAMMA_M = 0.02


def build_plans(j_grid=J_GRID, depth=2):
    """Enumerate plans as (d1,j1,d2,j2); j2=0 => depth-1 (single primitive + brake). d in {0,1,2,3}."""
    plans = []
    for d1 in range(4):
        for j1 in j_grid:
            plans.append((d1, j1, 0, 0))                     # depth-1: primitive then brake
    if depth >= 2:
        for d1 in range(4):
            for j1 in j_grid:
                for d2 in range(4):
                    for j2 in j_grid:
                        plans.append((d1, j1, d2, j2))
    return np.asarray(plans, dtype=np.int64)                 # [P,4]


def _frame(states, scene, system, n_dirs=4):
    """Return D [B,n_dirs,2], plan-time-fixed (detached). Order (D8):
    0:+e_par 1:-e_par 2:+e_perp 3:-e_perp 4:+diag1 5:-diag1 6:+diag2 7:-diag2 (diag=(e_par±e_perp)/sqrt2).
    First 4 == the R-1 D4 set (so config (i) reproduces R-1 exactly)."""
    b = states.shape[0]
    g = torch.as_tensor(scene.goal, dtype=states.dtype, device=states.device)
    if g.ndim == 1:
        g = g.unsqueeze(0).expand(b, -1)
    tg = g - system.position(states); n = torch.linalg.norm(tg, dim=1, keepdim=True)
    epar = tg / n.clamp_min(1.0e-12)
    fb = torch.zeros_like(epar); fb[:, 0] = 1.0
    epar = torch.where(n > 1.0e-6, epar, fb)                 # world-axis fallback
    eperp = torch.stack([-epar[:, 1], epar[:, 0]], dim=1)
    dirs = [epar, -epar, eperp, -eperp]
    if n_dirs >= 8:
        s = 1.0 / math.sqrt(2.0)
        dirs += [s * (epar + eperp), -s * (epar + eperp), s * (epar - eperp), -s * (epar - eperp)]
    return torch.stack(dirs, dim=1).detach(), g             # [B,n_dirs,2], goal [B,2]


# ---- Stage R-2 revised grammar: D8 + cruise segments + brake/capture terminals, depth<=3 ----
# Plan encoding: int array [P,10] columns
#   [k1,d1,j1, k2,d2,j2, k3,d3,j3, term]  where kX: 0=thrust 1=cruise 2=empty; dX: dir idx; jX: length;
#   term: 0=brake(deadband,T_stop=25) 1=capture(LQR feedback, T_capture=100).
T_CAPTURE = 100


def build_plans_r2(n_dirs=8, j_grid=J_GRID, max_depth=3, cruise=True, terminals=(0, 1)):
    """Enumerate all 1..max_depth segment-sequences x terminals. Segment = thrust(dir,j) or cruise(j)."""
    segs = [(0, d, j) for d in range(n_dirs) for j in j_grid]          # thrust
    if cruise:
        segs += [(1, 0, j) for j in j_grid]                            # cruise (dir ignored)
    EMPTY = (2, 0, 0)
    plans = []
    seqs = []
    for s1 in segs:                                                    # depth-1
        seqs.append((s1, EMPTY, EMPTY))
    if max_depth >= 2:
        for s1 in segs:
            for s2 in segs:
                seqs.append((s1, s2, EMPTY))
    if max_depth >= 3:
        for s1 in segs:
            for s2 in segs:
                for s3 in segs:
                    seqs.append((s1, s2, s3))
    for (a, b, c) in seqs:
        for term in terminals:
            plans.append((a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2], term))
    return np.asarray(plans, dtype=np.int64)                           # [P,10]


def reach_R2(states, scene, system, config, plans, *, gamma_m=GAMMA_M, plan_chunk=None,
             return_dur=False, return_mind2g=False, n_dirs=8):
    """Exhaustive R over R-2 plans [P,10] with plan-dim chunking + early-stop. Returns R [B] (running min).
    return_dur/return_mind2g valid ONLY for a single block (small plan set, e.g. beam) — returns [B,P]."""
    dt = float(config["env"]["dt"]); h_scale = float(config["env"]["h_scale"])
    u_max, v_max = _u_max_v_max(config, system); oob = float(config["env"]["oob_limit"])
    tstop = int(math.ceil(v_max / (u_max * dt)))
    dev, dtype = states.device, states.dtype; B = states.shape[0]; P = plans.shape[0]
    D, goal = _frame(states, scene, system, n_dirs=n_dirs)             # [B,n_dirs,2], [B,2]
    c = torch.as_tensor(scene.obstacle_centers, dtype=dtype, device=dev)
    r = torch.as_tensor(scene.obstacle_radii, dtype=dtype, device=dev)
    a = torch.as_tensor(scene.obstacle_active, dtype=torch.bool, device=dev)
    R = torch.full((B,), float("inf"), dtype=dtype, device=dev)
    full_dur = torch.full((B, P), float("inf"), dtype=dtype, device=dev) if return_dur else None
    full_m2g = torch.full((B, P), float("inf"), dtype=dtype, device=dev) if return_mind2g else None
    pc = P if plan_chunk is None else plan_chunk
    for p0 in range(0, P, pc):
        pt = torch.as_tensor(plans[p0:p0 + pc], device=dev); Pb = pt.shape[0]
        k1, d1, j1, k2, d2, j2, k3, d3, j3, term = (pt[:, i] for i in range(10))
        b1 = j1; b2 = j1 + j2; b3 = j1 + j2 + j3
        termlen = torch.where(term == 0, torch.tensor(tstop, device=dev), torch.tensor(T_CAPTURE, device=dev))
        plen = b3 + termlen
        Tmax = int(plen.max())
        x = states.repeat_interleave(Pb, dim=0)                        # [B*Pb,4]
        gr = goal.repeat_interleave(Pb, dim=0)
        # segment control vectors (thrust -> u_max*D8[dir]; cruise/empty -> 0), gathered per (b,plan)
        def sv(kX, dX):
            g = D[:, dX, :].reshape(B * Pb, 2)                         # D[:,dX] -> [B,Pb,2]
            return torch.where((kX == 0).repeat(B).unsqueeze(1), u_max * g, torch.zeros_like(g))
        sv1, sv2, sv3 = sv(k1, d1), sv(k2, d2), sv(k3, d3)
        b1r, b2r, b3r, plenr, termr = (t.repeat(B) for t in (b1, b2, b3, plen, term))
        scn = SimpleNamespace(obstacle_centers=c.repeat_interleave(Pb, 0), obstacle_radii=r.repeat_interleave(Pb, 0),
                              obstacle_active=a.repeat_interleave(Pb, 0))
        cummax_h = signed_h(system.position(x), scn, h_scale)
        oob_seen = system.position(x).abs().amax(dim=1) > oob
        kg = torch.full((B * Pb,), -1, dtype=torch.long, device=dev)
        hcap = torch.full((B * Pb,), float("inf"), dtype=dtype, device=dev)
        oobcap = torch.ones(B * Pb, dtype=torch.bool, device=dev)
        mind2g = torch.linalg.norm(system.position(x) - gr, dim=1)
        for k in range(Tmax):
            v = x[:, 2:4]
            u_brake = torch.where(v.abs() > u_max * dt, -u_max * torch.sign(v), -v / dt)
            u_lqr = system.lqr_action(x, gr)
            term_u = torch.where((termr == 1).unsqueeze(1), u_lqr, u_brake)
            a1 = (k < b1r).unsqueeze(1); a2 = ((k >= b1r) & (k < b2r)).unsqueeze(1)
            a3 = ((k >= b2r) & (k < b3r)).unsqueeze(1); at = ((k >= b3r) & (k < plenr)).unsqueeze(1)
            u = torch.where(a1, sv1, torch.where(a2, sv2, torch.where(a3, sv3, torch.where(at, term_u, u_brake))))
            x = rk4_step(system, x, u, dt)
            p = system.position(x); spd = system.speed(x); h = signed_h(p, scn, h_scale)
            cummax_h = torch.maximum(cummax_h, h); oob_seen = oob_seen | (p.abs().amax(dim=1) > oob)
            mind2g = torch.minimum(mind2g, torch.linalg.norm(p - gr, dim=1))
            hit = (torch.linalg.norm(p - gr, dim=1) <= GOAL_R) & (spd <= GOAL_SPD) & (kg < 0) & (k < plenr)
            kg = torch.where(hit, torch.full_like(kg, k + 1), kg)
            hcap = torch.where(hit, cummax_h, hcap); oobcap = torch.where(hit, oob_seen, oobcap)
            if k % 8 == 7:
                settled = (kg >= 0) | (cummax_h > -gamma_m) | oob_seen | (k >= plenr - 1)
                if bool(settled.all()):
                    break
        admitted = (kg >= 0) & (hcap <= -gamma_m) & (~oobcap)
        dur = torch.where(admitted, kg.to(dtype) * dt, torch.full((B * Pb,), float("inf"), dtype=dtype, device=dev))
        dur = dur.reshape(B, Pb)
        R = torch.minimum(R, dur.min(dim=1).values)
        if return_dur: full_dur[:, p0:p0 + pc] = dur
        if return_mind2g: full_m2g[:, p0:p0 + pc] = mind2g.reshape(B, Pb)
    out = [R]
    if return_dur: out.append(full_dur)
    if return_mind2g: out.append(full_m2g)
    return out[0] if len(out) == 1 else tuple(out)


def _u_max_v_max(config, system):
    u = float(config["env"]["bounds"][system.name]["u_max"]); v = float(config["env"]["bounds"][system.name]["v_max"])
    return u, v


def reach_R(states, scene, system, config, plans, *, gamma_m=GAMMA_M, return_dur=False, return_mind2g=False):
    """Exhaustive R over `plans` (a [P,4] int array). Returns R [B] (inf if none admitted).
    If return_dur: also returns dur [B,P] (inf where not admitted) for the argmin follower.
    If return_mind2g: also returns mind2g [B,P] = closest approach ||p-g|| over the rollout (for beam ranking)."""
    dt = float(config["env"]["dt"]); h_scale = float(config["env"]["h_scale"])
    u_max, v_max = _u_max_v_max(config, system)
    oob = float(config["env"]["oob_limit"]); tstop = int(math.ceil(v_max / (u_max * dt)))
    dev, dtype = states.device, states.dtype
    B = states.shape[0]; P = plans.shape[0]
    D, goal = _frame(states, scene, system)                  # [B,4,2], [B,2]
    pt = torch.as_tensor(plans, device=dev)                  # [P,4]
    d1i, j1, d2i, j2 = pt[:, 0], pt[:, 1], pt[:, 2], pt[:, 3]
    Tmax = int(max(int(j1.max()), 0) + int(j2.max()) + tstop)
    # fuse plan dim: [B*P, ...]
    x = states.repeat_interleave(P, dim=0)                    # [BP,4]
    d1 = D[:, d1i, :].reshape(B * P, 2); d2 = D[:, d2i, :].reshape(B * P, 2)
    j1r = j1.repeat(B); j2r = j2.repeat(B)                    # [BP]
    gr = goal.repeat_interleave(P, dim=0)                    # [BP,2]
    c = torch.as_tensor(scene.obstacle_centers, dtype=dtype, device=dev)
    r = torch.as_tensor(scene.obstacle_radii, dtype=dtype, device=dev)
    a = torch.as_tensor(scene.obstacle_active, dtype=torch.bool, device=dev)
    scn = SimpleNamespace(obstacle_centers=c.repeat_interleave(P, 0), obstacle_radii=r.repeat_interleave(P, 0),
                          obstacle_active=a.repeat_interleave(P, 0))
    cummax_h = signed_h(system.position(x), scn, h_scale)     # k=0 (plan state)
    oob_seen = system.position(x).abs().amax(dim=1) > oob
    kg = torch.full((B * P,), -1, dtype=torch.long, device=dev)
    mind2g = torch.linalg.norm(system.position(x) - gr, dim=1)
    for k in range(Tmax):
        v = x[:, 2:4]
        u_brake = torch.where(v.abs() > u_max * dt, -u_max * torch.sign(v), -v / dt)
        in1 = (k < j1r).unsqueeze(1); in2 = ((k >= j1r) & (k < j1r + j2r)).unsqueeze(1)
        u = torch.where(in1, u_max * d1, torch.where(in2, u_max * d2, u_brake))
        x = rk4_step(system, x, u, dt)
        p = system.position(x); spd = system.speed(x); h = signed_h(p, scn, h_scale)
        cummax_h = torch.maximum(cummax_h, h); oob_seen = oob_seen | (p.abs().amax(dim=1) > oob)
        mind2g = torch.minimum(mind2g, torch.linalg.norm(p - gr, dim=1))
        hit = (torch.linalg.norm(p - gr, dim=1) <= GOAL_R) & (spd <= GOAL_SPD) & (kg < 0)
        kg = torch.where(hit, torch.full_like(kg, k + 1), kg)   # k+1: steps taken to reach
    admitted = (kg >= 0) & (cummax_h <= -gamma_m) & (~oob_seen)
    dur = torch.where(admitted, kg.to(dtype) * dt, torch.full((B * P,), float("inf"), dtype=dtype, device=dev))
    dur = dur.reshape(B, P); R = dur.min(dim=1).values
    out = [R]
    if return_dur: out.append(dur)
    if return_mind2g: out.append(mind2g.reshape(B, P))
    return out[0] if len(out) == 1 else tuple(out)


_FOLLOWER_PLANS_R2 = None


def follower_plan_r2(state1, scene, system, config, n_dirs=8):
    """R-2 follower oracle: exhaustive over depth<=2 D8+cruise+{brake,capture} (8064 plans, static
    coverage 0.988 ~ full depth-3 0.98; an upper bound on full-depth exhaustive R). Returns the argmin
    admitted plan per state, decoded for open-loop replay.
      -> has_plan [B], R [B], (sv1,sv2,sv3 [B,2], b1,b2,b3 [B], term [B])  (sv=segment control vector;
         b*=step boundaries; term 0=brake 1=capture-LQR)."""
    global _FOLLOWER_PLANS_R2
    if _FOLLOWER_PLANS_R2 is None:
        # follower per-replan oracle: 4-dir depth<=2 + {brake,capture} (1624 plans, ~0.11s/250-states,
        # static cov 0.972 ~ full depth-3 0.98). A plan-subset of full depth-3 => R is an UPPER BOUND on
        # exhaustive R (spot-checked). 10x cheaper than D8+cruise => replanning stays tractable at n2000.
        _FOLLOWER_PLANS_R2 = build_plans_r2(n_dirs=4, max_depth=2, cruise=False, terminals=(0, 1))
    plans = _FOLLOWER_PLANS_R2
    dev = state1.device; B = state1.shape[0]; u_max, _ = _u_max_v_max(config, system)
    R, dur = reach_R2(state1, scene, system, config, plans, return_dur=True, n_dirs=n_dirs)
    best = dur.argmin(dim=1); has_plan = torch.isfinite(R)
    D, _ = _frame(state1, scene, system, n_dirs=n_dirs)                # [B,n_dirs,2]
    pt = torch.as_tensor(plans, device=dev)[best]                     # [B,10]
    ar = torch.arange(B, device=dev)
    def sv(kX, dX):
        g = D[ar, dX]; return torch.where((kX == 0).unsqueeze(1), u_max * g, torch.zeros_like(g))
    k1, d1, j1, k2, d2, j2, k3, d3, j3, term = (pt[:, i] for i in range(10))
    sv1, sv2, sv3 = sv(k1, d1), sv(k2, d2), sv(k3, d3)
    return has_plan, R, (sv1, sv2, sv3, j1, j1 + j2, j1 + j2 + j3, term)


def follower_plan(state1, scene, system, config, j_grid=J_GRID, beam_W=8):
    """Beam depth-2: pick argmin-duration admitted plan per state (upper bound on exhaustive R).
    Depth-1 exhaustive; expand the top-W first-primitives (ranked by closest-approach distance-to-goal,
    UNION over batch => shared plan set) to depth-2. Returns:
      has_plan [B] bool, R [B], and committed replay tuple (d1_vec [B,2], j1 [B], d2_vec [B,2], j2 [B])."""
    dev = state1.device; B = state1.shape[0]
    d1p = build_plans(j_grid, depth=1)                                  # 28 depth-1 (primitive + brake)
    _, mind2g1 = reach_R(state1, scene, system, config, d1p, return_mind2g=True)   # [B,28] closest approach
    topW = min(beam_W, d1p.shape[0])
    order = torch.argsort(mind2g1, dim=1)[:, :topW]                     # top-W primitives by closest approach
    firsts = np.unique(d1p[order.reshape(-1).cpu().numpy()][:, :2], axis=0)        # unique (d1,j1), shared
    plans = list(map(tuple, d1p))
    for (d1, j1) in firsts:
        for d2 in range(4):
            for j2 in j_grid:
                plans.append((int(d1), int(j1), int(d2), int(j2)))
    plansA = np.asarray(plans, dtype=np.int64)
    R, dur = reach_R(state1, scene, system, config, plansA, return_dur=True)
    best = dur.argmin(dim=1)                                            # argmin duration (inf-safe)
    has_plan = torch.isfinite(R)
    D, _ = _frame(state1, scene, system)
    pt = torch.as_tensor(plansA, device=dev)[best]                     # [B,4]
    d1i, j1, d2i, j2 = pt[:, 0], pt[:, 1], pt[:, 2], pt[:, 3]
    d1v = D[torch.arange(B, device=dev), d1i]; d2v = D[torch.arange(B, device=dev), d2i]
    return has_plan, R, (d1v, j1, d2v, j2)
