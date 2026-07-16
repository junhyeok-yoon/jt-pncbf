"""v2.6.1 fundamental diagnostics (D1-D7) — read-only on state.

Instruments a single filtered learned-policy re-roll of best.pt on the full pool (n2000/seed23456)
to answer the bang-bang (D1), collision point-of-no-return (D2), value navigation-blindness (D3),
filter-conservativeness-by-density (D4), value-calibration (D5) questions, plus a contour /
boundary-fragmentation pass (D6) and a dimension-agnostic certificate-metric table (D7). Per-step
data is not persisted by training, so every number here is derived by re-rolling best.pt read-only
(no training / config / git change). Writes a JSON dump to scratchpad and a D6 figure to figures/.

Nothing is fabricated; states/outcomes come from src.common.outcomes (the eval primitives). The deployed
certificate is h(x)=V_hat (make_h_fn -> value_net.value, clamped [-1,1]). Ground-truth barrier for D5 is
h_star = phi(p,o) + c v^T Re (src.common.quadrotor_barrier), c=c_gain, h_scale from config.
"""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from src.common.filter_hardnet import (_base_alpha, _base_projection, _box_aware_projection, _cbf_terms,
                                        _hardnet_params, _SINGULAR_LG_THRESHOLD)
from src.common.outcomes import resolve_outcome, step_outcomes
from src.common.quadrotor_barrier import h_star_value, approach_speed, thrust_axis
from src.common.rk4 import rk4_step
from src.common.value_net import ValueNetEnsemble, make_h_fn
from src.common.observation import scene_obstacle_tensors
from src.eval.build_pools import load_pool
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.frameworks.jt_pncbf.train import make_system, _build_control_net

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
SP = Path("/tmp/claude-1000/-home-junhyeok-MIT-jt-pncbf/31d93785-ac11-4206-bf50-a4c3de145dff/scratchpad")
POOL = REPO / "data/secured_data/pools/eval_full_quadrotor-planar_n2000_seed23456.pkl"

F_MIN, F_MAX, TAU_MAX = 0.0, 19.62, 1.0
F_HOVER = 9.81                     # m*g
BENIGN_SURF = 1.0                  # nearest-obstacle surface-dist threshold for "benign"
BENIGN_VHAT = -0.5                 # V_hat large-negative threshold for "benign"
NEAR_GOAL = 0.5                    # ||p-g|| < 0.5 = goal region (D3)
D2_WINDOW = 40                     # ~2 s backward trace
ONE_SEC = 20                       # 1 s = 20 steps @ dt=0.05


def load(run_dir, ckpt, dev):
    ck = torch.load(run_dir / "checkpoints" / ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    system = make_system(cfg); system.u_bounds = system.u_bounds.to(dev, torch.float32)
    vnet = ValueNetEnsemble(system.obs_dim, cfg).to(dev, torch.float32)
    vnet.load_state_dict(ck["v_s_state"]); vnet.eval()
    for p in vnet.parameters():
        p.requires_grad_(False)
    policy = _build_control_net(system, cfg).to(dev); policy.load_state_dict(ck["pi_state"]); policy.eval()
    return ck, cfg, system, vnet, make_h_fn(vnet, system), policy


def surface_dist(pos, bs):
    """min over active obstacles of ||p-c||-r, per state. pos [B,2]."""
    centers = bs.obstacle_centers.to(pos.dtype); radii = bs.obstacle_radii.to(pos.dtype)
    active = bs.obstacle_active.to(torch.bool)
    d = torch.linalg.norm(pos.unsqueeze(1) - centers, dim=-1) - radii
    return d.masked_fill(~active, float("inf")).min(dim=1).values


def roll_chunk(system, h_fn, policy, cfg, bs, dev, max_steps):
    params = _hardnet_params(cfg); bounds = system.u_bounds; dt = float(cfg["env"]["dt"])
    x = initial_states_from_batch(bs).to(dev, torch.float32); B = x.shape[0]
    states = [x]
    unom = torch.zeros(max_steps, B, system.action_dim, device=dev)
    usafe = torch.zeros(max_steps, B, system.action_dim, device=dev)
    hstep = torch.zeros(max_steps, B, device=dev)
    empt = torch.zeros(max_steps, B, dtype=torch.bool, device=dev)
    sviol = torch.zeros(max_steps, B, dtype=torch.bool, device=dev)
    with torch.no_grad():
        for t in range(max_steps):
            un = policy(system.observation(x, bs))
            h, lf, lg = _cbf_terms(system, h_fn, x, bs, un, create_graph=False)
            h, lf, lg = h.detach(), lf.detach(), lg.detach()
            alpha = _base_alpha(h, params); row = -lf - alpha * h
            proj = _base_projection(un, lg, row, bounds, params)
            sing = torch.linalg.norm(lg, dim=1) < _SINGULAR_LG_THRESHOLD
            u, empty = _box_aware_projection(un, proj, lg, row, bounds)
            unom[t] = un; usafe[t] = u; hstep[t] = h; empt[t] = empty
            sviol[t] = sing & (row < 0.0)
            x = rk4_step(system, x, u, dt); states.append(x)
    S = torch.stack(states, 0)
    masks = step_outcomes(S, bs, system, cfg); res = resolve_outcome(masks)
    feas = ~(empt | sviol)
    return S, res, unom, usafe, hstep, empt, sviol, feas, masks.collided


def grad_p_vhat(h_fn, x, scene_ns):
    """grad_x V_hat(x) -> full [B,6]; deployed (clamped) certificate."""
    xr = x.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        h = h_fn(xr, scene_ns).reshape(-1)
        g = torch.autograd.grad(h.sum(), xr, create_graph=False)[0]
    return g.detach(), h.detach()


def index_scene(bs, idx):
    return SimpleNamespace(
        obstacle_centers=bs.obstacle_centers[idx], obstacle_radii=bs.obstacle_radii[idx],
        obstacle_active=bs.obstacle_active[idx], goal=bs.goal[idx])


def main():
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/v2.6.1__20260715-173808__seed42")
    if not run_dir.is_absolute():
        run_dir = REPO / run_dir
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck, cfg, system, vnet, h_fn, policy = load(run_dir, "best.pt", dev)
    step = int(ck.get("step", -1)); max_steps = int(cfg["eval"]["max_steps"]); chunk = 250
    c_gain = float(cfg["env"]["quadrotor_planar"]["c_gain"]); h_scale = float(cfg["env"]["h_scale"])
    scenes = load_pool(POOL).scenes; N = len(scenes)

    # accumulators
    ep = []                                    # per-episode records
    # D1 benign chatter (separate for policy u_nom and deployed u_safe)
    ben_n = 0
    ben_df_nom = []; ben_df_safe = []          # |Delta f| in benign steps
    ben_f_lo_safe = 0; ben_f_hi_safe = 0       # benign steps at thrust extremes (deployed)
    ben_f_lo_nom = 0; ben_f_hi_nom = 0
    ben_intervene = []                         # ||u_safe-u_nom|| in benign steps
    # global box-extreme occupancy (deployed)
    tot_steps = 0; f_at_lo = 0; f_at_hi = 0; tau_at_ext = 0
    du_f_all = []; du_tau_all = []             # control-rate samples (deployed), subsampled
    # D2
    d2 = []
    # D3
    d3_proj = []; d3_gradp = []; d3_sat = 0; d3_n = 0
    d3_to_proj = []                            # timeout-episode near-goal projections
    d3_term_grad = []; d3_run_grad = []        # analytic terminal vs running task grad @ near-goal (timeout eps)
    # D4 by obstacle count
    d4 = {}                                    # nobs -> dict lists
    # D5 calibration
    cal_e = []; cal_e_band = []; cal_unsound = 0; cal_cons = 0; cal_tot = 0; cal_band_tot = 0
    # D7 conservativeness (set-membership from trajectories) + Lipschitz
    cs_true_safe = 0; cs_true_safe_cert_unsafe = 0    # conservative
    cs_cert_safe = 0; cs_cert_safe_collide = 0        # unsound
    lip_gradnorm = []
    # D6 manifold occupancy
    theta_abs = []; omega_abs = []

    w_term = float(cfg["loss"]["policy"].get("w_terminal", 0.0))
    w_term_v = float(cfg["loss"]["policy"].get("w_terminal_v", 0.0))
    gamma_T = float(cfg["loss"]["policy"].get("gamma_T", 0.99))
    lambda_v = float(cfg["loss"]["policy"].get("lambda_v", 0.0))
    bptt_T = int(cfg["training"]["jt"].get("bptt_T", 30))
    disc_T = gamma_T ** bptt_T

    rng = np.random.default_rng(0)
    t0 = time.time()
    for s0 in range(0, N, chunk):
        bs = batch_scenes(scenes[s0:s0 + chunk], device=dev, dtype=torch.float32)
        S, res, unom, usafe, hstep, empt, sviol, feas, collided = roll_chunk(
            system, h_fn, policy, cfg, bs, dev, max_steps)
        B = S.shape[1]
        nobs = bs.obstacle_active.to(torch.int64).sum(dim=1).cpu().numpy()
        goal = bs.goal                                   # [B,2]
        outc = np.array(res.outcome); esteps = res.event_step.cpu().numpy()
        tot_steps += max_steps * B

        # per-step surface dist + h_star for all states
        pos = S[..., :2]                                 # [T+1,B,2]
        dist_g = torch.linalg.norm(pos - goal.unsqueeze(0), dim=-1)   # [T+1,B]
        surf = torch.stack([surface_dist(S[t, :, :2], bs) for t in range(S.shape[0])], 0)  # [T+1,B]
        hstar = torch.stack([h_star_value(S[t], bs, c_gain, h_scale) for t in range(S.shape[0])], 0)  # [T+1,B]

        # ---- D1: box-extremes + control-rate (deployed) ----
        f_at_lo += int((usafe[:, :, 0] <= F_MIN + 0.05).sum())
        f_at_hi += int((usafe[:, :, 0] >= F_MAX - 0.05).sum())
        tau_at_ext += int((usafe[:, :, 1].abs() >= TAU_MAX - 0.01).sum())
        du = usafe[1:] - usafe[:-1]                       # [T-1,B,2]
        du_f = du[:, :, 0].abs().reshape(-1).cpu().numpy()
        du_tau = du[:, :, 1].abs().reshape(-1).cpu().numpy()
        du_f_all.append(rng.choice(du_f, size=min(20000, du_f.size), replace=False))
        du_tau_all.append(rng.choice(du_tau, size=min(20000, du_tau.size), replace=False))

        # benign steps (deployed states S[0..T-1] aligned to actions t=0..T-1)
        vhat_step = hstep                                 # V_hat at S[t], t=0..T-1
        surf_step = surf[:-1]                             # [T,B]
        benign = (surf_step > BENIGN_SURF) & (vhat_step < BENIGN_VHAT)   # [T,B]
        ben_mask_du = benign[1:]                          # align to du (t=1..T-1)
        if ben_mask_du.any():
            m = ben_mask_du.reshape(-1)
            ben_df_safe.append(du[:, :, 0].abs().reshape(-1)[m].cpu().numpy())
            ben_df_nom.append((unom[1:, :, 0] - unom[:-1, :, 0]).abs().reshape(-1)[m].cpu().numpy())
            ben_intervene.append(torch.linalg.norm((usafe - unom)[1:], dim=2).reshape(-1)[m].cpu().numpy())
        bm = benign.reshape(-1)
        ben_n += int(bm.sum())
        ben_f_lo_safe += int((usafe[:, :, 0].reshape(-1)[bm] <= F_MIN + 1.0).sum())
        ben_f_hi_safe += int((usafe[:, :, 0].reshape(-1)[bm] >= F_MAX - 1.0).sum())
        ben_f_lo_nom += int((unom[:, :, 0].reshape(-1)[bm] <= F_MIN + 1.0).sum())
        ben_f_hi_nom += int((unom[:, :, 0].reshape(-1)[bm] >= F_MAX - 1.0).sum())

        # per-episode control-rate up to event (for outcome correlation)
        for i in range(B):
            end = esteps[i] if esteps[i] >= 0 else max_steps
            end = max(2, end)
            cr = float(torch.linalg.norm(du[:end - 1, i], dim=1).mean())
            ep.append(dict(idx=s0 + i, outcome=outc[i], event_step=int(esteps[i]),
                           ctrl_rate=cr, nobs=int(nobs[i])))

        # ---- D2: collision runway ----
        for i in range(B):
            if outc[i] != "collision":
                continue
            tc = esteps[i] if esteps[i] >= 0 else int(torch.argmax(collided[:, i].to(torch.int8)))
            w0 = max(0, tc - D2_WINDOW)
            fw = feas[w0:tc, i]                            # feasibility on approach (exclusive of tc)
            n_feas = int(fw.sum()); wlen = int(fw.numel())
            # last feasible step before collision -> offset
            fidx = torch.nonzero(fw, as_tuple=False).flatten()
            last_feas_off = int(tc - (w0 + int(fidx[-1]))) if fidx.numel() else (wlen + 1)
            # terminal infeasible streak length (consecutive infeasible immediately before tc)
            streak = 0
            for k in range(tc - 1, w0 - 1, -1):
                if not bool(feas[k, i]):
                    streak += 1
                else:
                    break
            d2.append(dict(idx=s0 + i, tc=int(tc), wlen=wlen, n_feas=n_feas,
                           feas_at_coll=bool(feas[min(tc, max_steps - 1), i]),
                           last_feas_off=last_feas_off, infeas_streak=streak))

        # ---- D3: near-goal grad_p V_hat . goal-dir ----
        ng = dist_g[:-1] < NEAR_GOAL                       # [T,B]; states S[0..T-1]
        gi, gt = torch.nonzero(ng, as_tuple=True)          # step, batch
        if gi.numel():
            sel = rng.choice(gi.numel(), size=min(5000, gi.numel()), replace=False)
            gi_s = gi[sel]; gt_s = gt[sel]
            xg = S[gi_s, gt_s]                              # [M,6]
            sc = index_scene(bs, gt_s)
            gfull, hval = grad_p_vhat(h_fn, xg, sc)
            gp = gfull[:, :2]
            gdir = (goal[gt_s] - xg[:, :2])
            gdir = gdir / torch.linalg.norm(gdir, dim=1, keepdim=True).clamp_min(1e-9)
            proj = torch.sum(gp * gdir, dim=1)             # >0 = grad points toward goal (V increases toward goal)
            gpn = torch.linalg.norm(gp, dim=1)
            d3_proj.append(proj.cpu().numpy()); d3_gradp.append(gpn.cpu().numpy())
            d3_sat += int((hval.abs() >= 0.999).sum()); d3_n += int(hval.numel())
            lip_gradnorm.append(torch.linalg.norm(gfull, dim=1).cpu().numpy())
            # timeout-episode subset
            to_ep = torch.as_tensor(outc == "timeout", device=dev)[gt_s]
            if to_ep.any():
                d3_to_proj.append(proj[to_ep].cpu().numpy())
                # analytic terminal vs windowed running task-grad @ these near-goal timeout states
                dgoal = torch.linalg.norm(xg[to_ep, :2] - goal[gt_s][to_ep], dim=1)
                vspd = torch.linalg.norm(xg[to_ep, 3:5], dim=1)
                # terminal grad wrt (p,v): d/dp[w_term*||p-g||]=w_term (unit), d/dv[w_term_v*||v||]=w_term_v
                term_g = disc_T * torch.sqrt(torch.tensor(float(w_term ** 2 + w_term_v ** 2), device=dev)) \
                    * torch.ones_like(dgoal)
                # running per-step task grad: d2=||p-g||^2 + lambda_v||v||^2 -> grad_p=2||p-g||, grad_v=2 lambda_v||v||
                run_g = torch.sqrt((2 * dgoal) ** 2 + (2 * lambda_v * vspd) ** 2)
                d3_term_grad.append(term_g.cpu().numpy()); d3_run_grad.append(run_g.cpu().numpy())

        # ---- D4: intervention by obstacle count ----
        dnorm = torch.linalg.norm(usafe - unom, dim=2)    # [T,B]
        intervene = (dnorm > 1e-3)
        at_box = ((usafe[:, :, 0] <= F_MIN + 0.05) | (usafe[:, :, 0] >= F_MAX - 0.05) |
                  (usafe[:, :, 1].abs() >= TAU_MAX - 0.01))
        for i in range(B):
            k = int(nobs[i])
            r = d4.setdefault(k, dict(n_ep=0, interv=[], dmag=[], empty=[], box=[]))
            r["n_ep"] += 1
            r["interv"].append(float(intervene[:, i].float().mean()))
            r["dmag"].append(float(dnorm[intervene[:, i], i].mean()) if intervene[:, i].any() else 0.0)
            r["empty"].append(float(empt[:, i].float().mean()))
            r["box"].append(float(at_box[:, i].float().mean()))

        # ---- D5: calibration V_hat(S[t]) vs sup_{s>=t} h_star ----
        # reverse cummax of hstar over time
        sup_h = torch.flip(torch.cummax(torch.flip(hstar, [0]), dim=0).values, [0])   # [T+1,B]
        vh = hstep                                          # V_hat at S[0..T-1]
        sh = torch.clamp(sup_h[:-1], -1.0, 1.0)             # aligned target, clamped like V_hat
        e = (vh - sh).reshape(-1).cpu().numpy()
        cal_e.append(rng.choice(e, size=min(30000, e.size), replace=False))
        cal_tot += e.size
        cal_unsound += int((vh < sh - 0.05).sum())         # V_hat under-predicts danger (UNSOUND)
        cal_cons += int((vh > sh + 0.05).sum())            # V_hat over-predicts danger (conservative)
        band = (sh.reshape(-1).abs() < 0.3)                # near-boundary
        if band.any():
            cal_e_band.append(e[band.cpu().numpy()])
            cal_band_tot += int(band.sum())

        # ---- D7: set-membership conservativeness (does trajectory collide after t?) ----
        # future-collision flag per (t,i): any collision at step >= t
        coll_cum = torch.flip(torch.cummax(torch.flip(collided.to(torch.int8), [0]), dim=0).values, [0]).bool()
        future_coll = coll_cum[:-1]                        # [T,B] aligned to S[0..T-1]
        cert_safe = vh <= 0.0
        true_safe = ~future_coll
        cs_true_safe += int(true_safe.sum()); cs_true_safe_cert_unsafe += int((true_safe & ~cert_safe).sum())
        cs_cert_safe += int(cert_safe.sum()); cs_cert_safe_collide += int((cert_safe & future_coll).sum())

        # D6 manifold occupancy
        theta_abs.append(S[..., 2].abs().reshape(-1).cpu().numpy())
        omega_abs.append(S[..., 5].abs().reshape(-1).cpu().numpy())

    wall = time.time() - t0

    # ---- aggregate ----
    outc = np.array([r["outcome"] for r in ep]); rate = lambda k: float((outc == k).mean())
    du_f = np.concatenate(du_f_all); du_tau = np.concatenate(du_tau_all)
    cr = np.array([r["ctrl_rate"] for r in ep])
    def _corr(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
    D1 = dict(
        box_extremes=dict(
            f_at_min_frac=f_at_lo / tot_steps, f_at_max_frac=f_at_hi / tot_steps,
            tau_at_ext_frac=tau_at_ext / tot_steps),
        control_rate_deployed=dict(
            df_mean=float(du_f.mean()), df_p50=float(np.percentile(du_f, 50)),
            df_p95=float(np.percentile(du_f, 95)), df_max=float(du_f.max()),
            df_frac_gt_half_box=float((du_f > 0.5 * F_MAX).mean()),
            dtau_mean=float(du_tau.mean()), dtau_p95=float(np.percentile(du_tau, 95)),
            dtau_frac_gt_half_box=float((du_tau > 0.5 * TAU_MAX).mean())),
        ctrl_rate_by_outcome=dict(
            reach=float(cr[outc == "goal"].mean()), collision=float(cr[outc == "collision"].mean()),
            timeout=float(cr[outc == "timeout"].mean())),
        ctrl_rate_corr=dict(
            vs_collision=_corr(cr, (outc == "collision").astype(float)),
            vs_timeout=_corr(cr, (outc == "timeout").astype(float))),
        benign=dict(
            n_benign_steps=ben_n, frac_of_all=ben_n / tot_steps,
            df_safe_mean=float(np.concatenate(ben_df_safe).mean()) if ben_df_safe else 0.0,
            df_safe_frac_gt_half_box=float((np.concatenate(ben_df_safe) > 0.5 * F_MAX).mean()) if ben_df_safe else 0.0,
            df_nom_mean=float(np.concatenate(ben_df_nom).mean()) if ben_df_nom else 0.0,
            df_nom_frac_gt_half_box=float((np.concatenate(ben_df_nom) > 0.5 * F_MAX).mean()) if ben_df_nom else 0.0,
            intervene_mean=float(np.concatenate(ben_intervene).mean()) if ben_intervene else 0.0,
            f_lo_frac_safe=ben_f_lo_safe / max(1, ben_n), f_hi_frac_safe=ben_f_hi_safe / max(1, ben_n),
            f_lo_frac_nom=ben_f_lo_nom / max(1, ben_n), f_hi_frac_nom=ben_f_hi_nom / max(1, ben_n)))

    n_coll = len(d2)
    wlens = np.array([r["wlen"] for r in d2]); nf = np.array([r["n_feas"] for r in d2])
    lfo = np.array([r["last_feas_off"] for r in d2]); strk = np.array([r["infeas_streak"] for r in d2])
    feas_at = np.array([r["feas_at_coll"] for r in d2])
    long_runway = (nf > ONE_SEC)                     # >1 s of feasible steps in the 2 s pre-collision window
    short_runway = (strk >= ONE_SEC)                 # infeasible for >=1 s continuously before collision
    D2 = dict(
        n_collision=n_coll,
        frac_feasible_at_collision=float(feas_at.mean()),
        median_n_feas_in40=float(np.median(nf)), median_window_len=float(np.median(wlens)),
        median_last_feasible_offset=float(np.median(lfo)),
        median_infeasible_streak=float(np.median(strk)),
        frac_long_runway_gt1s=float(long_runway.mean()),      # avoidable: safe action existed >1s
        frac_short_runway_infeas_ge1s=float(short_runway.mean()),  # unavoidable-ish: trapped >=1s
        frac_feasible_until_le5steps=float((lfo <= 5).mean()))     # safe action existed within 5 steps of impact

    d3p = np.concatenate(d3_proj); d3g = np.concatenate(d3_gradp)
    d3to = np.concatenate(d3_to_proj) if d3_to_proj else np.array([])
    D3 = dict(
        n_near_goal_states=d3_n,
        gradp_vhat_saturated_frac=d3_sat / max(1, d3_n),
        proj_on_goaldir=dict(mean=float(d3p.mean()), median=float(np.median(d3p)),
                             std=float(d3p.std()), frac_positive=float((d3p > 0).mean()),
                             frac_abs_lt_0p05=float((np.abs(d3p) < 0.05).mean())),
        gradp_vhat_norm=dict(mean=float(d3g.mean()), median=float(np.median(d3g)),
                             p95=float(np.percentile(d3g, 95))),
        timeout_near_goal_proj=dict(
            n=int(d3to.size), mean=float(d3to.mean()) if d3to.size else 0.0,
            frac_positive=float((d3to > 0).mean()) if d3to.size else 0.0),
        analytic_grad_near_goal_timeout=dict(
            terminal_grad_mean=float(np.concatenate(d3_term_grad).mean()) if d3_term_grad else 0.0,
            running_grad_mean=float(np.concatenate(d3_run_grad).mean()) if d3_run_grad else 0.0,
            running_grad_p95=float(np.percentile(np.concatenate(d3_run_grad), 95)) if d3_run_grad else 0.0,
            note="terminal=disc_T*sqrt(w_term^2+w_term_v^2) applied only at x_T; running=||grad task_cost|| per step"),
        weights=dict(w_terminal=w_term, w_terminal_v=w_term_v, gamma_T=gamma_T,
                     bptt_T=bptt_T, disc_T=disc_T, lambda_v=lambda_v))

    D4 = {}
    for k in sorted(d4):
        r = d4[k]
        D4[str(k)] = dict(n_ep=r["n_ep"], intervention_rate=float(np.mean(r["interv"])),
                          mean_dmag=float(np.mean(r["dmag"])), empty_infeas_rate=float(np.mean(r["empty"])),
                          box_boundary_frac=float(np.mean(r["box"])))

    cae = np.concatenate(cal_e); caeb = np.concatenate(cal_e_band) if cal_e_band else np.array([])
    D5 = dict(
        n_states=cal_tot,
        mean_signed_error=float(cae.mean()), mae=float(np.abs(cae).mean()),
        frac_unsound_vhat_under=cal_unsound / cal_tot,      # V_hat < true sup-h (under-certifies danger)
        frac_conservative_vhat_over=cal_cons / cal_tot,     # V_hat > true sup-h (over-certifies danger)
        near_boundary=dict(n=cal_band_tot, mean_signed_error=float(caeb.mean()) if caeb.size else 0.0,
                           mae=float(np.abs(caeb).mean()) if caeb.size else 0.0))

    lipn = np.concatenate(lip_gradnorm)
    D7 = dict(
        feasibility_fraction=1.0 - D2["frac_short_runway_infeas_ge1s"],   # aggregate: 1 - unavoidable-share
        setmembership=dict(
            true_safe=cs_true_safe, true_safe_certified_unsafe=cs_true_safe_cert_unsafe,
            conservative_frac=cs_true_safe_cert_unsafe / max(1, cs_true_safe),
            cert_safe=cs_cert_safe, cert_safe_collide_later=cs_cert_safe_collide,
            unsound_frac=cs_cert_safe_collide / max(1, cs_cert_safe)),
        value_regularity_lipschitz=dict(
            mean_gradnorm=float(lipn.mean()), p95=float(np.percentile(lipn, 95)),
            max=float(lipn.max())))

    # ---- B_0 authority (D7) via the near-B0 sampler (same primitive as lg_authority loss) ----
    from src.common.quadrotor_barrier import sample_near_B0_states
    gen = torch.Generator(device="cpu").manual_seed(0)
    xb, scb = sample_near_B0_states(system, cfg, 4096, dev, torch.float32, gen)
    _, _, lgb = _cbf_terms(system, h_fn, xb, scb, torch.zeros(4096, 2, device=dev), create_graph=False)
    lgn = torch.linalg.norm(lgb.detach(), dim=1).cpu().numpy()
    eps_g = float(cfg["loss"]["value"]["lg_authority"]["eps_g"])
    D7["B0_authority"] = dict(eps_g=eps_g, lg_min=float(lgn.min()), lg_median=float(np.median(lgn)),
                              lg_degen_frac=float((lgn < eps_g).mean()))

    D6_manifold = dict(
        theta_abs=dict(p50=float(np.percentile(np.concatenate(theta_abs), 50)),
                       p95=float(np.percentile(np.concatenate(theta_abs), 95)),
                       p99=float(np.percentile(np.concatenate(theta_abs), 99))),
        omega_abs=dict(p50=float(np.percentile(np.concatenate(omega_abs), 50)),
                       p95=float(np.percentile(np.concatenate(omega_abs), 95)),
                       p99=float(np.percentile(np.concatenate(omega_abs), 99))))

    report = dict(run=run_dir.name, step=step, n=N, max_steps=max_steps, wall_s=round(wall, 1),
                  outcomes=dict(reach=rate("goal"), collision=rate("collision"),
                                timeout=rate("timeout"), oob=rate("oob"), stuck=rate("stuck")),
                  D1=D1, D2=D2, D3=D3, D4=D4, D5=D5, D6_manifold=D6_manifold, D7=D7)
    json.dump(report, open(SP / "quadrotor_fundamental_diag.json", "w"), indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nsaved {SP / 'quadrotor_fundamental_diag.json'}  ({wall:.0f}s)")


if __name__ == "__main__":
    main()
