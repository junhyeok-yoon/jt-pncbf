"""Stage R follower + exhaustive-R evaluation harness (analysis-only; production untouched).

- exhaustive_R(scenes, depth): batched exhaustive R(x) at start states (ARM-R0).
- follow(scenes): open-loop replanning plan-follower (beam depth-2, replan_every=5; NO filter, NO policy).
    No-plan states brake (m_0) and retry every step; has-plan states run open-loop 5 steps then re-plan.
- outcomes/metrics via src.common.outcomes (same resolver as production eval).

All rollouts reuse rk4_step + signed_h (same RK4 + ZOH + velocity clamp as V_M). Grid dt=dt_Vm=0.05.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import yaml

from src.common.outcomes import resolve_outcome, step_outcomes
from src.common.rk4 import rk4_step
from src.envs.double_integrator import DoubleIntegrator
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.eval.build_pools import load_pool
import scripts.analysis.reach_witness as RW

REPO = Path("/home/junhyeok/MIT/jt-pncbf")


def load_cfg():
    def m(a, o):
        d = dict(a)
        for k, v in o.items():
            d[k] = m(d[k], v) if isinstance(v, dict) and isinstance(d.get(k), dict) else v
        return d
    return m(yaml.safe_load((REPO / "src/configs/base_config.yaml").read_text()),
             yaml.safe_load((REPO / "src/configs/exp_config.yaml").read_text()))


def make_system(cfg, dev, DT=torch.float32):
    s = DoubleIntegrator(cfg); s.u_bounds = s.u_bounds.to(device=dev, dtype=DT); return s


def exhaustive_R(scenes, system, cfg, depth, *, chunk=200, dev=None, DT=torch.float32):
    """Exhaustive R at start states over ALL depth-{1,2} plans (28 / 812). Returns R [N] on cpu."""
    dev = dev or torch.device("cuda")
    plans = RW.build_plans(RW.J_GRID, depth)
    out = []
    for s0 in range(0, len(scenes), chunk):
        bs = batch_scenes(scenes[s0:s0 + chunk], device=dev, dtype=DT)
        x = initial_states_from_batch(bs).to(DT)
        with torch.no_grad():
            R = RW.reach_R(x, bs, system, cfg, plans)
        out.append(R.cpu())
    return torch.cat(out)


def exhaustive_R2(scenes, system, cfg, plans, *, chunk=100, plan_chunk=12000, n_dirs=8,
                  dev=None, DT=torch.float32):
    """Exhaustive R-2 grammar reach-time at start states, chunking BOTH states and the plan dim."""
    dev = dev or torch.device("cuda"); out = []
    for s0 in range(0, len(scenes), chunk):
        bs = batch_scenes(scenes[s0:s0 + chunk], device=dev, dtype=DT)
        x = initial_states_from_batch(bs).to(DT)
        with torch.no_grad():
            R = RW.reach_R2(x, bs, system, cfg, plans, plan_chunk=plan_chunk, n_dirs=n_dirs)
        out.append(R.cpu())
    return torch.cat(out)


def classify_starts(scenes):
    """R-1 start classification (matches arm_r0_why.json): per-start dist, straight-blocked, min clearance."""
    dist = np.zeros(len(scenes)); blocked = np.zeros(len(scenes), bool); clear = np.zeros(len(scenes))
    for i, s in enumerate(scenes):
        p = np.asarray(s.start, float); g = np.asarray(s.goal, float)
        c = np.asarray(s.obstacle_centers, float); r = np.asarray(s.obstacle_radii, float)
        a = np.asarray(s.obstacle_active, bool)
        d = g - p; L2 = float((d * d).sum()); best = 1e9; blk = False
        for k in range(len(r)):
            if not a[k]:
                continue
            t = 0.0 if L2 == 0 else max(0.0, min(1.0, float(((c[k] - p) * d).sum()) / L2))
            cp = p + t * d; dd = float(np.linalg.norm(cp - c[k])) - float(r[k])
            best = min(best, dd); blk = blk or dd < 0
        dist[i] = float(np.linalg.norm(g - p)); blocked[i] = blk; clear[i] = best
    return dist, blocked, clear


def follow(scenes, system, cfg, *, replan_every=5, beam_W=8, max_steps=None, chunk=250,
           dev=None, DT=torch.float32, start_states=None, budget_steps=None):
    """Run the plan-follower on `scenes`. Returns dict of stacked tensors (cpu):
       S [T+1,N,4], U [T,N,2], outcome (list), event_step [N], has_plan_ever [N], replan_count [N]."""
    dev = dev or torch.device("cuda")
    dt = float(cfg["env"]["dt"]); u_max, _ = RW._u_max_v_max(cfg, system)
    max_steps = int(cfg["eval"]["max_steps"]) if max_steps is None else int(max_steps)
    allS, allU, outc, allev, allhp, allrc = [], [], [], [], [], []
    for s0 in range(0, len(scenes), chunk):
        bs = batch_scenes(scenes[s0:s0 + chunk], device=dev, dtype=DT)
        if start_states is None:
            x = initial_states_from_batch(bs).to(DT)
        else:
            x = start_states[s0:s0 + chunk].to(device=dev, dtype=DT).clone()
        B = x.shape[0]
        cd1 = torch.zeros(B, 2, device=dev, dtype=DT); cd2 = torch.zeros(B, 2, device=dev, dtype=DT)
        cj1 = torch.zeros(B, dtype=torch.long, device=dev); cj2 = torch.zeros(B, dtype=torch.long, device=dev)
        chas = torch.zeros(B, dtype=torch.bool, device=dev)
        cstart = torch.zeros(B, dtype=torch.long, device=dev)
        next_replan = torch.zeros(B, dtype=torch.long, device=dev)
        hp_ever = torch.zeros(B, dtype=torch.bool, device=dev); rc = torch.zeros(B, dtype=torch.long, device=dev)
        states = [x]; ctrls = []
        with torch.no_grad():
            for t in range(max_steps):
                due = t >= next_replan
                if bool(due.any()):
                    hp, _R, (d1v, j1, d2v, j2) = RW.follower_plan(x, bs, system, cfg, beam_W=beam_W)
                    cd1[due] = d1v[due]; cd2[due] = d2v[due]; cj1[due] = j1[due]; cj2[due] = j2[due]
                    chas[due] = hp[due]; cstart[due] = t; hp_ever |= (hp & due); rc += due.long()
                    next_replan[due] = t + torch.where(hp[due], torch.tensor(replan_every, device=dev),
                                                       torch.tensor(1, device=dev))
                s = (t - cstart)
                in1 = (chas & (s < cj1)).unsqueeze(1); in2 = (chas & (s >= cj1) & (s < cj1 + cj2)).unsqueeze(1)
                v = x[:, 2:4]; ub = torch.where(v.abs() > u_max * dt, -u_max * torch.sign(v), -v / dt)
                u = torch.where(in1, u_max * cd1, torch.where(in2, u_max * cd2, ub))
                x = rk4_step(system, x, u, dt); states.append(x); ctrls.append(u)
        S = torch.stack(states, 0); U = torch.stack(ctrls, 0)
        masks = step_outcomes(S, bs, system, cfg); res = resolve_outcome(masks)
        allS.append(S.cpu()); allU.append(U.cpu()); outc += res.outcome
        allev.append(res.event_step.cpu()); allhp.append(hp_ever.cpu()); allrc.append(rc.cpu())
    return dict(S=torch.cat(allS, 1), U=torch.cat(allU, 1), outcome=outc,
                event_step=torch.cat(allev), has_plan_ever=torch.cat(allhp), replan_count=torch.cat(allrc))


def follow_r2(scenes, system, cfg, *, replan_every=5, max_steps=None, chunk=250,
              dev=None, DT=torch.float32, start_states=None, budget_steps=None):
    """R-2 plan-follower: open-loop replanning (beam=depth<=2 D8+cruise+{brake,capture}); no filter/policy.
    No-plan states brake and retry every step. Returns dict of cpu tensors (S,U,outcome,event_step,
    has_plan_ever,replan_count) + R0 (finite exhaustive-R2 depth<=2 at start, for time-vs-R analysis)."""
    dev = dev or torch.device("cuda")
    dt = float(cfg["env"]["dt"]); u_max, _ = RW._u_max_v_max(cfg, system)
    max_steps = int(cfg["eval"]["max_steps"]) if max_steps is None else int(max_steps)
    allS, allU, outc, allev, allhp, allrc, allR0 = [], [], [], [], [], [], []
    for s0 in range(0, len(scenes), chunk):
        bs = batch_scenes(scenes[s0:s0 + chunk], device=dev, dtype=DT)
        if start_states is None:
            x = initial_states_from_batch(bs).to(DT)
        else:
            x = start_states[s0:s0 + chunk].to(device=dev, dtype=DT).clone()
        B = x.shape[0]
        goal = torch.as_tensor(bs.goal, dtype=DT, device=dev)
        if goal.ndim == 1:
            goal = goal.unsqueeze(0).expand(B, -1)
        z2 = torch.zeros(B, 2, device=dev, dtype=DT)
        csv1, csv2, csv3 = z2.clone(), z2.clone(), z2.clone()
        cb1 = torch.zeros(B, dtype=torch.long, device=dev); cb2 = cb1.clone(); cb3 = cb1.clone()
        cterm = torch.zeros(B, dtype=torch.long, device=dev); chas = torch.zeros(B, dtype=torch.bool, device=dev)
        cstart = torch.zeros(B, dtype=torch.long, device=dev); next_replan = torch.zeros(B, dtype=torch.long, device=dev)
        hp_ever = torch.zeros(B, dtype=torch.bool, device=dev); rc = torch.zeros(B, dtype=torch.long, device=dev)
        R0 = None; states = [x]; ctrls = []
        with torch.no_grad():
            for t in range(max_steps):
                due = t >= next_replan
                if bool(due.any()):
                    hp, Rr, (s1, s2, s3, b1, b2, b3, term) = RW.follower_plan_r2(x, bs, system, cfg)
                    if t == 0:
                        R0 = Rr.cpu()
                    for a, n in [(csv1, s1), (csv2, s2), (csv3, s3)]:
                        a[due] = n[due]
                    cb1[due] = b1[due]; cb2[due] = b2[due]; cb3[due] = b3[due]; cterm[due] = term[due]
                    chas[due] = hp[due]; cstart[due] = t; hp_ever |= (hp & due); rc += due.long()
                    next_replan[due] = t + torch.where(hp[due], torch.tensor(replan_every, device=dev),
                                                       torch.tensor(1, device=dev))
                s = (t - cstart)
                v = x[:, 2:4]; ub = torch.where(v.abs() > u_max * dt, -u_max * torch.sign(v), -v / dt)
                ulqr = system.lqr_action(x, goal)
                term_u = torch.where((cterm == 1).unsqueeze(1), ulqr, ub)
                a1 = (chas & (s < cb1)).unsqueeze(1); a2 = (chas & (s >= cb1) & (s < cb2)).unsqueeze(1)
                a3 = (chas & (s >= cb2) & (s < cb3)).unsqueeze(1); atrm = (chas & (s >= cb3)).unsqueeze(1)
                u = torch.where(a1, csv1, torch.where(a2, csv2, torch.where(a3, csv3, torch.where(atrm, term_u, ub))))
                x = rk4_step(system, x, u, dt); states.append(x); ctrls.append(u)
        S = torch.stack(states, 0); U = torch.stack(ctrls, 0)
        masks = step_outcomes(S, bs, system, cfg); res = resolve_outcome(masks)
        allS.append(S.cpu()); allU.append(U.cpu()); outc += res.outcome
        allev.append(res.event_step.cpu()); allhp.append(hp_ever.cpu()); allrc.append(rc.cpu()); allR0.append(R0)
    return dict(S=torch.cat(allS, 1), U=torch.cat(allU, 1), outcome=outc, event_step=torch.cat(allev),
                has_plan_ever=torch.cat(allhp), replan_count=torch.cat(allrc), R0=torch.cat(allR0))


def recertify_dt001(S, U, scenes, system, cfg, ep_idx, *, dev=None, DT=torch.float32,
                    gamma_m=RW.GAMMA_M, sub=5):
    """Re-roll the executed control prefix of selected episodes on a dt/sub grid (ZOH each control).
    Returns, per selected episode, the finer-grid max signed_h over the prefix (grid-optimism check)."""
    dev = dev or torch.device("cuda")
    from src.common.signed_h import signed_h
    from types import SimpleNamespace
    dt = float(cfg["env"]["dt"]); hs = float(cfg["env"]["h_scale"]); fine = dt / sub
    res = []
    for i in ep_idx:
        sc = scenes[i]
        scn = SimpleNamespace(obstacle_centers=torch.as_tensor(sc.obstacle_centers, dtype=DT, device=dev),
                              obstacle_radii=torch.as_tensor(sc.obstacle_radii, dtype=DT, device=dev),
                              obstacle_active=torch.as_tensor(sc.obstacle_active, dtype=torch.bool, device=dev))
        x = S[0, i:i + 1].to(device=dev, dtype=DT).clone()
        maxh = signed_h(system.position(x), scn, hs)
        with torch.no_grad():
            for t in range(U.shape[0]):
                u = U[t, i:i + 1].to(device=dev, dtype=DT)
                for _ in range(sub):
                    x = rk4_step(system, x, u, fine); maxh = torch.maximum(maxh, signed_h(system.position(x), scn, hs))
        res.append(float(maxh))
    return res
