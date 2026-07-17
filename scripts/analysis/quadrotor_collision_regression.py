"""v2.6.2 collision-regression diagnostic (D1-D3 + D5) — why did collision rise (0.0810->0.0955) while
timeout/reach improved? Read-only, INFERENCE-ONLY, CPU (GPU is occupied by the training M5; CPU => zero GPU
contention). Instrumented dual_arm-path re-roll of each checkpoint's filtered policy on the full pool
n2000/seed23456, capturing per-step states / u_nom / empty-flag; then D1 speed distributions, D2 objective-
term shares + gate occupancy (FULL only), D3 collision anatomy, D5 checkpoint re-scan. Phase 1 = {FULL,
v2.6.1}; the goal-only column + D4 append later (Phase 2). Facts only.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from src.common.filter_hardnet import (_base_alpha, _base_projection, _box_aware_projection, _cbf_terms,
                                        _hardnet_params, _SINGULAR_LG_THRESHOLD)
from src.common.observation import scene_obstacle_tensors
from src.common.outcomes import resolve_outcome, step_outcomes
from src.common.rk4 import rk4_step
from src.common.value_net import ValueNetEnsemble, make_h_fn
from src.eval.build_pools import load_pool
from src.eval.dual_arm import _rollout, _arm_summary
from src.eval.evaluate import first_physical_event_step
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.frameworks.jt_pncbf.train import make_system, _build_control_net

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
SP = Path("/tmp/claude-1000/-home-junhyeok-MIT-jt-pncbf/31d93785-ac11-4206-bf50-a4c3de145dff/scratchpad")
POOL = REPO / "data/secured_data/pools/eval_full_quadrotor-planar_n2000_seed23456.pkl"
DEV = torch.device("cpu")
SURF_BINS = [(-1e9, 0.25, "<0.25"), (0.25, 0.5, "0.25-0.5"), (0.5, 1.0, "0.5-1.0"), (1.0, 2.0, "1.0-2.0")]


def load_ck(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    system = make_system(cfg); system.u_bounds = system.u_bounds.to(DEV, torch.float32)
    vnet = ValueNetEnsemble(system.obs_dim, cfg).to(DEV, torch.float32)
    vnet.load_state_dict(ck["v_s_state"]); vnet.eval()
    for p in vnet.parameters():
        p.requires_grad_(False)
    policy = _build_control_net(system, cfg).to(DEV); policy.load_state_dict(ck["pi_state"]); policy.eval()
    return ck, cfg, system, vnet, make_h_fn(vnet, system), policy


def roll_instrumented(system, h_fn, policy, cfg, scenes, want_terms, chunk=250):
    """Filtered dual_arm-path roll; returns per-episode records + aggregate arrays for D1/D2/D3."""
    params = _hardnet_params(cfg); bounds = system.u_bounds
    max_steps = int(cfg["eval"]["max_steps"]); dt = float(cfg["env"]["dt"])
    goal_radius = float(cfg["env"].get("goal_radius", 0.15))
    pc = cfg["loss"]["policy"]
    lam_v = float(pc["lambda_v"]); mu_u = float(pc["mu_u"])
    w_settle = float(pc.get("w_settle", 0.0)); settle_rho = float(pc.get("settle_rho", 0.30))
    w_appr = float(pc.get("w_appr", 0.0)); appr_d0 = float(pc.get("appr_d0", 0.5))
    pol = lambda x, sc: policy(system.observation(x, sc))

    speed_all, speed_benign = [], []              # ||v|| samples (subsampled)
    inward_by_bin = {lab: [] for _, _, lab in SURF_BINS}
    goalentry_speed = []; reach_len = []
    terms = {k: [] for k in ["goal", "lam_v_v2", "settle", "appr", "mu_u"]}     # per-step realized magnitudes
    terms_stratum = {s: {k: [] for k in terms} for s in ["benign", "near_obs", "near_goal"]}
    gate_d_occ = []; gate_rho_occ = []
    coll_rec = []                                  # per collision: dict

    for s0 in range(0, len(scenes), chunk):
        bs = batch_scenes(scenes[s0:s0 + chunk], device=DEV, dtype=torch.float32)
        x = initial_states_from_batch(bs).float(); B = x.shape[0]
        oc, orad, oact = scene_obstacle_tensors(bs, DEV, torch.float32)        # [B,K,2],[B,K],[B,K]
        th0 = x[:, 2].abs().cpu().numpy(); v0 = x[:, 3:5]
        states = [x.clone()]; empt = torch.zeros(max_steps, B, dtype=torch.bool)
        with torch.no_grad():
            for t in range(max_steps):
                un = pol(x, bs)
                h, lf, lg = _cbf_terms(system, h_fn, x, bs, un, create_graph=False)
                alpha = _base_alpha(h.detach(), params); row = -lf.detach() - alpha * h.detach()
                proj = _base_projection(un, lg.detach(), row, bounds, params)
                u, empty = _box_aware_projection(un, proj, lg.detach(), row, bounds)
                empt[t] = empty
                x = rk4_step(system, x, u, dt); states.append(x.clone())
        S = torch.stack(states, 0)                                            # [T+1,B,6]
        masks = step_outcomes(S, bs, system, cfg); res = resolve_outcome(masks)
        an = first_physical_event_step(masks); an = torch.where(an >= 0, an, torch.full_like(an, max_steps))
        act = torch.arange(max_steps).unsqueeze(1) < an.unsqueeze(0)          # [T,B] active steps
        collided = masks.collided

        # per-step geometry on active states (vectorized)
        P = system.position(S[:max_steps])                                    # [T,B,2]
        V = S[:max_steps, :, 3:5]                                             # [T,B,2]
        spd = torch.linalg.norm(V, dim=2)                                     # [T,B]
        rel = P.unsqueeze(2) - oc.unsqueeze(0)                                # [T,B,K,2]
        dist_c = torch.linalg.norm(rel, dim=3)                               # [T,B,K]
        surf = dist_c - orad.unsqueeze(0)                                     # [T,B,K]
        surf_masked = surf.masked_fill(~oact.unsqueeze(0).bool(), float("inf"))
        nn_surf, nn_j = surf_masked.min(dim=2)                               # [T,B] nearest surface dist
        normal = rel / dist_c.unsqueeze(3).clamp_min(1e-9)
        vdotn = torch.sum(V.unsqueeze(2) * normal, dim=3)                    # [T,B,K]
        gidx = nn_j.unsqueeze(2)
        vdotn_nn = torch.gather(vdotn, 2, gidx).squeeze(2)                   # [T,B] to nearest
        inward_nn = torch.relu(-vdotn_nn)                                     # [T,B]
        with torch.no_grad():
            Vhat = h_fn(S[:max_steps].reshape(-1, 6), _tile_scene(bs, max_steps)).reshape(max_steps, B)
        goal = bs.goal if bs.goal.ndim > 1 else bs.goal.unsqueeze(0).expand(B, -1)
        gdist = torch.linalg.norm(P - goal.unsqueeze(0), dim=2)             # [T,B]

        actf = act.float()
        # D1 speed
        sp = spd[act]; speed_all.append(sp[::7].cpu().numpy())
        benign = act & (nn_surf > 1.0) & (Vhat < -0.5)
        speed_benign.append(spd[benign].cpu().numpy())
        for lo, hi, lab in SURF_BINS:
            m = act & (nn_surf >= lo) & (nn_surf < hi)
            if m.any():
                inward_by_bin[lab].append(inward_nn[m].cpu().numpy())
        for i in range(B):
            am = act[:, i]
            if res.outcome[i] == "goal":
                gr = int(an[i]); reach_len.append(gr)
                # speed at first goal-radius entry
                ent = (gdist[:, i] <= goal_radius) & am
                if ent.any():
                    goalentry_speed.append(float(spd[torch.nonzero(ent)[0, 0], i]))
        # D2 term shares (FULL only: want_terms)
        if want_terms:
            g_rho = torch.exp(-(gdist * gdist) / (settle_rho * settle_rho))
            g_d = torch.exp(-(surf * surf) / (appr_d0 * appr_d0)) * oact.unsqueeze(0).bool().float()
            inward_all = torch.relu(-vdotn)
            appr_term = w_appr * torch.sum(g_d * inward_all * inward_all, dim=2)   # [T,B]
            settle_term = w_settle * g_rho * spd * spd
            d2 = gdist * gdist
            lam_term = lam_v * spd * spd
            # mu_u term needs u_safe; approximate with |u|~ from states not stored -> use 0 placeholder note
            comp = {"goal": d2, "lam_v_v2": lam_term, "settle": settle_term, "appr": appr_term}
            gate_d_occ.append(float(((g_d.amax(dim=2) > 0.1) & act).float().sum() / act.float().sum().clamp_min(1)))
            gate_rho_occ.append(float(((g_rho > 0.1) & act).float().sum() / act.float().sum().clamp_min(1)))
            for k, arr in comp.items():
                terms[k].append(arr[act][::7].cpu().numpy())
            strat = {"benign": act & (nn_surf > 1.0) & (Vhat < -0.5),
                     "near_obs": act & (nn_surf < 1.0),
                     "near_goal": act & (gdist < 0.5)}
            for sname, sm in strat.items():
                if sm.any():
                    for k, arr in comp.items():
                        terms_stratum[sname][k].append(arr[sm][::5].cpu().numpy())
        # D3 collision anatomy
        for i in range(B):
            if res.outcome[i] != "collision":
                continue
            tc = min(int(torch.argmax(collided[:, i].to(torch.int8))), max_steps - 1)  # clamp: nn_surf/empt are [max_steps]
            # last-feasible offset: trace empt backward from tc
            lastf = None
            for k in range(1, min(41, tc + 1)):
                if not bool(empt[tc - k, i]):
                    lastf = k; break
            # inward speed at surface-dist==1.0 crossing (last time nn_surf crosses 1.0 before tc)
            pre = nn_surf[:tc + 1, i]
            cross = None
            for tt in range(tc, 0, -1):
                if pre[tt] <= 1.0 and pre[tt - 1] > 1.0:
                    cross = float(inward_nn[tt, i]); break
            inw0 = float(torch.sum(v0[i] * _toward_nearest(x0i(S, i), oc[i], orad[i], oact[i])))
            coll_rec.append(dict(t_coll=tc, last_feasible_off=(lastf if lastf is not None else -1),
                                 inward_at_1m=(cross if cross is not None else -1.0),
                                 th0=float(th0[i]), inward_v0=inw0))
    return dict(speed_all=speed_all, speed_benign=speed_benign, inward_by_bin=inward_by_bin,
                goalentry_speed=goalentry_speed, reach_len=reach_len, terms=terms,
                terms_stratum=terms_stratum, gate_d_occ=gate_d_occ, gate_rho_occ=gate_rho_occ,
                coll_rec=coll_rec)


def _tile_scene(bs, T):
    from types import SimpleNamespace
    # replicate the batched scene T times along batch dim to match S[:T].reshape(-1,6)
    def rep(a):
        return a.repeat_interleave(1, 0) if a.ndim == 0 else a
    B = bs.obstacle_centers.shape[0]
    return SimpleNamespace(
        obstacle_centers=bs.obstacle_centers.unsqueeze(0).expand(T, B, *bs.obstacle_centers.shape[1:]).reshape(T * B, *bs.obstacle_centers.shape[1:]),
        obstacle_radii=bs.obstacle_radii.unsqueeze(0).expand(T, B, *bs.obstacle_radii.shape[1:]).reshape(T * B, *bs.obstacle_radii.shape[1:]),
        obstacle_active=bs.obstacle_active.unsqueeze(0).expand(T, B, *bs.obstacle_active.shape[1:]).reshape(T * B, *bs.obstacle_active.shape[1:]),
        goal=(bs.goal.unsqueeze(0).expand(T, B, 2).reshape(T * B, 2) if bs.goal.ndim > 1
              else bs.goal.unsqueeze(0).expand(T * B, 2)))


def x0i(S, i):
    return S[0, i]


def _toward_nearest(x0, centers, radii, active):
    p = x0[:2]
    d = torch.linalg.norm(p.unsqueeze(0) - centers, dim=1) - radii
    d = d.masked_fill(~active.bool(), float("inf"))
    j = int(torch.argmin(d)); to = centers[j] - p
    return to / torch.linalg.norm(to).clamp_min(1e-9)


def q(a):
    a = np.concatenate(a) if isinstance(a, list) and a else (np.asarray(a) if len(a) else np.array([0.0]))
    return dict(p50=float(np.percentile(a, 50)), p90=float(np.percentile(a, 90)), max=float(np.max(a)), n=int(a.size))


def summarize(name, R):
    out = {"name": name}
    out["speed"] = dict(overall=q(R["speed_all"]), benign=q(R["speed_benign"]))
    out["inward_by_surf"] = {lab: q(v) if v else None for lab, v in R["inward_by_bin"].items()}
    gs = np.array(R["goalentry_speed"]) if R["goalentry_speed"] else np.array([])
    out["goalentry_speed"] = dict(p50=float(np.percentile(gs, 50)) if gs.size else None,
                                  p90=float(np.percentile(gs, 90)) if gs.size else None, n=int(gs.size))
    rl = np.array(R["reach_len"]) if R["reach_len"] else np.array([])
    out["reach_len_p50"] = float(np.percentile(rl, 50)) if rl.size else None
    cr = R["coll_rec"]; n = len(cr)
    out["n_collision"] = n
    if n:
        tc = np.array([c["t_coll"] for c in cr]); lf = np.array([c["last_feasible_off"] for c in cr])
        iw = np.array([c["inward_at_1m"] for c in cr]); th = np.array([c["th0"] for c in cr])
        out["collision"] = dict(
            t_coll_p50=float(np.percentile(tc, 50)), frac_early_lt40=float((tc < 40).mean()),
            frac_late_gt80=float((tc > 80).mean()),
            runway_p50=float(np.percentile(lf[lf >= 0], 50)) if (lf >= 0).any() else None,
            frac_feasible_ge20=float((lf >= 20).mean()),
            inward_at_1m_p50=float(np.percentile(iw[iw >= 0], 50)) if (iw >= 0).any() else None,
            frac_tilted_th0_gt_halfpi=float((th > np.pi / 2).mean()))
    if R["terms"]["goal"]:
        out["term_shares"] = {k: q(v) for k, v in R["terms"].items() if v}
        out["term_shares_stratum"] = {s: {k: q(v) for k, v in d.items() if v}
                                      for s, d in R["terms_stratum"].items()}
        out["gate_d_occ"] = float(np.mean(R["gate_d_occ"])) if R["gate_d_occ"] else None
        out["gate_rho_occ"] = float(np.mean(R["gate_rho_occ"])) if R["gate_rho_occ"] else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True, help="name=path pairs")
    ap.add_argument("--terms-for", default="FULL", help="which name gets D2 term shares")
    ap.add_argument("--out", default=str(SP / "collision_regression.json"))
    a = ap.parse_args()
    scenes = load_pool(POOL).scenes
    results = {}
    for spec in a.ckpts:
        name, path = spec.split("=", 1)
        t0 = time.time()
        ck, cfg, system, vnet, h_fn, policy = load_ck(path)
        R = roll_instrumented(system, h_fn, policy, cfg, scenes, want_terms=(name == a.terms_for))
        results[name] = summarize(name, R)
        results[name]["ckpt"] = path; results[name]["step"] = int(ck.get("step", -1))
        print(f"{name}: n_coll={results[name]['n_collision']} speed_p90={results[name]['speed']['overall']['p90']:.3f} "
              f"goalentry_p50={results[name]['goalentry_speed']['p50']} ({time.time()-t0:.0f}s)", flush=True)
    json.dump(results, open(a.out, "w"), indent=2)
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
