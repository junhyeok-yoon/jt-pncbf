"""v2.6.2 residual-collision anatomy (D1) + lookahead-DEFLATION sweep (D2) on the brake-envelope best.pt.
Read-only, eval-only, full pool n2000/seed23456, dual_arm re-roll path. D1: classify the ~130 residual
collisions born-doomed / early-trapped / late-trapped + IC strata cross-tab. D2: the AUDIT-reopened
CONSERVATIVE predictive variant — deflate alpha on the SAFE branch only (alpha_safe/(1+beta*gap), h<=0;
alpha_unsafe UNCHANGED), h_peak from _lookahead_peak_h with the deployed policy; grid n{3,8}xbeta{1,3} + OFF
baseline; per-cell collision/timeout/cps/empty-inf(by density)/mean||du||; per-class conversion of the D1
EARLY/LATE-TRAPPED sets. Runtime override only (never committed). Facts.
"""
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.common.filter_hardnet import (_base_alpha, _base_projection, _box_aware_projection, _cbf_terms,
                                        _hardnet_params, _lookahead_peak_h, _SINGULAR_LG_THRESHOLD)
from src.common.observation import scene_obstacle_tensors
from src.common.outcomes import resolve_outcome, step_outcomes
from src.common.rk4 import rk4_step
from src.common.value_net import ValueNetEnsemble, make_h_fn
from src.eval.build_pools import load_pool
from src.eval.dual_arm import cps_v2
from src.eval.evaluate import first_physical_event_step
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.frameworks.jt_pncbf.train import make_system, _build_control_net

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
SP = Path("/tmp/claude-1000/-home-junhyeok-MIT-jt-pncbf/31d93785-ac11-4206-bf50-a4c3de145dff/scratchpad")
POOL = REPO / "data/secured_data/pools/eval_full_quadrotor-planar_n2000_seed23456.pkl"
RUN = REPO / "data/v2.6.2__20260716-182949__seed42"
TAU_BRAKE = 0.6; DELTA = 0.1
GRID = [(3, 1.0), (3, 3.0), (8, 1.0), (8, 3.0)]
DENS = [(1, 2, "1-2"), (3, 5, "3-5"), (6, 8, "6-8"), (9, 99, "9+")]


def load(dev):
    ck = torch.load(RUN / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    cfg = ck["config"]
    system = make_system(cfg); system.u_bounds = system.u_bounds.to(dev, torch.float32)
    vnet = ValueNetEnsemble(system.obs_dim, cfg).to(dev, torch.float32)
    vnet.load_state_dict(ck["v_s_state"]); vnet.eval()
    for p in vnet.parameters():
        p.requires_grad_(False)
    policy = _build_control_net(system, cfg).to(dev); policy.load_state_dict(ck["pi_state"]); policy.eval()
    return ck, cfg, system, vnet, make_h_fn(vnet, system), policy


def roll(system, h_fn, policy, cfg, scenes, dev, deflate=None, want_anat=False, chunk=250):
    """Filtered roll; deflate=(n,beta) applies alpha_safe/(1+beta*gap) on the h<=0 branch. Returns per-episode
    (outcome, empty_rate, dun_mean, n_obs) + (if want_anat) per-episode anatomy dict."""
    params = _hardnet_params(cfg); bounds = system.u_bounds
    max_steps = int(cfg["eval"]["max_steps"]); dt = float(cfg["env"]["dt"])
    pol = lambda x, sc: policy(system.observation(x, sc))
    ep = []; anat = []
    for s0 in range(0, len(scenes), chunk):
        bs = batch_scenes(scenes[s0:s0 + chunk], device=dev, dtype=torch.float32)
        x = initial_states_from_batch(bs).float(); B = x.shape[0]
        oc, orad, oact = scene_obstacle_tensors(bs, dev, torch.float32)
        n_obs = oact.to(torch.int64).sum(1).cpu().numpy()
        th0 = x[:, 2].abs().cpu().numpy()
        # inward v_0 to nearest obstacle
        p0 = x[:, :2]; v0 = x[:, 3:5]; rel0 = p0.unsqueeze(1) - oc; d0c = torch.linalg.norm(rel0, dim=2)
        surf0 = (d0c - orad).masked_fill(~oact.bool(), float("inf")); j0 = surf0.argmin(1)
        n0 = (rel0[torch.arange(B), j0]); n0 = n0 / torch.linalg.norm(n0, dim=1, keepdim=True).clamp_min(1e-9)
        inward_v0 = torch.relu(-torch.sum(v0 * n0, dim=1)).cpu().numpy()
        # birth deficit (envelope) at t=0
        nrm0 = rel0 / d0c.unsqueeze(2).clamp_min(1e-9); inw0_all = torch.relu(-torch.sum(v0.unsqueeze(1) * nrm0, dim=2))
        deficit0 = (torch.relu(inw0_all * TAU_BRAKE - (d0c - orad)) * oact.bool().float()).max(1).values.cpu().numpy()
        empt = torch.zeros(max_steps, B, dtype=torch.bool, device=dev)
        dun = torch.zeros(max_steps, B, device=dev)
        states = [x.clone()]
        with torch.no_grad():
            for t in range(max_steps):
                un = pol(x, bs)
                h, lf, lg = _cbf_terms(system, h_fn, x, bs, un, create_graph=False)
                h, lf, lg = h.detach(), lf.detach(), lg.detach()
                alpha = _base_alpha(h, params)
                if deflate is not None:
                    n_la, beta = deflate
                    dp = replace_params(params, n_la)
                    h_peak = _lookahead_peak_h(system=system, h_fn=h_fn, policy_fn=pol, x=x, scene=bs, params=dp)
                    gap = torch.relu((h_peak - h) / DELTA)
                    factor = 1.0 / (1.0 + beta * gap)
                    alpha = torch.where(h <= 0.0, alpha * factor, alpha)     # DEFLATE safe branch only
                row = -lf - alpha * h
                proj = _base_projection(un, lg, row, bounds, params)
                u, empty = _box_aware_projection(un, proj, lg, row, bounds)
                empt[t] = empty; dun[t] = torch.linalg.norm(u - un, dim=1)
                x = rk4_step(system, x, u, dt); states.append(x.clone())
        S = torch.stack(states, 0); masks = step_outcomes(S, bs, system, cfg); res = resolve_outcome(masks)
        collided = masks.collided
        an = first_physical_event_step(masks); an = torch.where(an >= 0, an, torch.full_like(an, max_steps))
        act = torch.arange(max_steps, device=dev).unsqueeze(1) < an.unsqueeze(0)
        for i in range(B):
            am = act[:, i]; na = int(am.sum())
            er = float((empt[:, i] & am).sum()) / na if na else 0.0
            dm = float(dun[:, i][am].mean()) if na else 0.0
            ep.append((res.outcome[i], er, dm, int(n_obs[i]), s0 + i))
            if want_anat and res.outcome[i] == "collision":
                tc = min(int(torch.argmax(collided[:, i].to(torch.int8))), max_steps - 1)
                em = empt[:tc, i].cpu().numpy() if tc > 0 else np.array([], dtype=bool)
                # feasible prefix: steps from t=0 feasible before first infeasible
                fp = int(np.argmax(em)) if em.any() else tc            # first True index; if none, all feasible
                if not em.any():
                    fp = tc
                anat.append(dict(idx=s0 + i, tc=tc, infeas0=bool(empt[0, i]), deficit0=float(deficit0[i]),
                                 feas_prefix=fp, feas5=(not bool(empt[min(5, tc), i])),
                                 feas10=(not bool(empt[min(10, tc), i])), feas20=(not bool(empt[min(20, tc), i])),
                                 th0=float(th0[i]), inward_v0=float(inward_v0[i])))
    return ep, anat


def replace_params(params, n_la):
    from dataclasses import replace
    return replace(params, lookahead_enabled=True, lookahead_n=n_la, lookahead_beta=1.0, lookahead_delta=DELTA)


def summ(ep):
    o = np.array([e[0] for e in ep]); em = np.array([e[1] for e in ep]); dm = np.array([e[2] for e in ep])
    nb = np.array([e[3] for e in ep])
    rate = lambda k: float((o == k).mean())
    out = dict(collision=rate("collision"), timeout=rate("timeout"), reach=rate("goal"),
               cps_v2=cps_v2(o, em), empty_inf=float(em.mean()), dun_mean=float(dm.mean()))
    out["by_density_inf"] = {lab: (float(em[(nb >= lo) & (nb <= hi)].mean())
                                   if ((nb >= lo) & (nb <= hi)).any() else 0.0) for lo, hi, lab in DENS}
    return out


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck, cfg, system, vnet, h_fn, policy = load(dev)
    scenes = load_pool(POOL).scenes
    t0 = time.time()
    base_ep, anat = roll(system, h_fn, policy, cfg, scenes, dev, deflate=None, want_anat=True)
    base = summ(base_ep)
    base_outcome = {e[4]: e[0] for e in base_ep}
    print(f"BASELINE (OFF) coll={base['collision']:.4f} timeout={base['timeout']:.4f} cps={base['cps_v2']:.4f} "
          f"inf={base['empty_inf']:.4f} n_coll={len(anat)} ({time.time()-t0:.0f}s)", flush=True)

    # D1 three-way classification
    for a in anat:
        if a["infeas0"] or a["deficit0"] > 0:
            a["cls"] = "born_doomed"
        elif a["feas_prefix"] >= 20:
            a["cls"] = "late_trapped"
        else:
            a["cls"] = "early_trapped"
    cls = np.array([a["cls"] for a in anat])
    n = len(anat)
    d1 = dict(n_collision=n,
              born_doomed=int((cls == "born_doomed").sum()),
              early_trapped=int((cls == "early_trapped").sum()),
              late_trapped=int((cls == "late_trapped").sum()),
              feas_prefix_p50=float(np.median([a["feas_prefix"] for a in anat])),
              frac_infeas0=float(np.mean([a["infeas0"] for a in anat])),
              frac_deficit0_pos=float(np.mean([a["deficit0"] > 0 for a in anat])))
    # IC strata cross-tab (collision-class fractions within band n)
    th = np.array([a["th0"] for a in anat]); iv = np.array([a["inward_v0"] for a in anat])
    pi = np.pi
    def xtab(arr, bins):
        r = {}
        for lo, hi, lab in bins:
            m = (arr >= lo) & (arr < hi)
            r[lab] = dict(n=int(m.sum()), born=int((cls[m] == "born_doomed").sum()),
                          early=int((cls[m] == "early_trapped").sum()), late=int((cls[m] == "late_trapped").sum()))
        return r
    d1["by_theta0"] = xtab(th, [(0, pi/6, "[0,pi/6)"), (pi/6, pi/2, "[pi/6,pi/2)"), (pi/2, pi+1e-6, "[pi/2,pi]")])
    d1["by_inward_v0"] = xtab(iv, [(-1e9, 1e-9, "receding~0"), (1e-9, 0.75, "[0,0.75)"), (0.75, 1e9, ">=0.75")])

    # D2 deflation sweep
    early_idx = set(a["idx"] for a in anat if a["cls"] == "early_trapped")
    late_idx = set(a["idx"] for a in anat if a["cls"] == "late_trapped")
    born_idx = set(a["idx"] for a in anat if a["cls"] == "born_doomed")
    cells = {"OFF": base}
    conv = {}
    for (n_la, beta) in GRID:
        tc = time.time()
        ce, _ = roll(system, h_fn, policy, cfg, scenes, dev, deflate=(n_la, beta), want_anat=False)
        key = f"n{n_la}_b{beta}"
        cells[key] = summ(ce)
        co = {e[4]: e[0] for e in ce}
        def rescued(idxs):
            idxs = [i for i in idxs if co.get(i) in ("goal", "timeout")]
            return len(idxs)
        conv[key] = dict(early_rescued=rescued(early_idx), early_total=len(early_idx),
                         late_rescued=rescued(late_idx), late_total=len(late_idx),
                         born_moved=rescued(born_idx), born_total=len(born_idx))
        c = cells[key]
        print(f"  {key}: coll={c['collision']:.4f} timeout={c['timeout']:.4f} cps={c['cps_v2']:.4f} "
              f"inf={c['empty_inf']:.4f} du={c['dun_mean']:.3f} (dcoll={c['collision']-base['collision']:+.4f}) "
              f"| early_resc {conv[key]['early_rescued']}/{len(early_idx)} late_resc {conv[key]['late_rescued']}/{len(late_idx)} "
              f"born_moved {conv[key]['born_moved']}/{len(born_idx)} ({time.time()-tc:.0f}s)", flush=True)

    out = dict(run=RUN.name, step=int(ck.get("step", -1)), D1=d1, cells=cells, conversion=conv)
    json.dump(out, open(SP / "quadrotor_deflation.json", "w"), indent=2)
    print("saved", SP / "quadrotor_deflation.json", flush=True)
    # arithmetic check
    assert d1["born_doomed"] + d1["early_trapped"] + d1["late_trapped"] == n, "3-way split != residual count"
    print(f"D1 3-way sum check OK: {d1['born_doomed']}+{d1['early_trapped']}+{d1['late_trapped']}={n}", flush=True)


if __name__ == "__main__":
    main()
