"""v2.6.2 amendment-3 FP anatomy — WHY does the worst-case kinematic recoverability test flag REACHABLE ICs?

Read-only / eval-only. The amendment-3 criterion (is_recoverable) over-fires on reach ICs (FP=0.2175 at
margin 0 >> the 0.02 gate), so it was NOT adopted (recov_margin stays absent -> filter inert). This script
characterises the over-firing so a future criterion can be corrected. NEVER overwrites amendment-3 artifacts.

Pipeline (single brake-envelope re-roll, filter ON, reused for all four analyses):
  D1  reconstruct the per-IC sets @margin 0.0 -- FP (reach & flagged), TP (born-doomed & flagged),
      TN (reach & unflagged), FN (born-doomed & unflagged).  born-doomed == collision & (infeasible@0 OR
      brake-envelope deficit>0 @0), matching residual-anatomy / validate.py.
  D2  which criterion term over-fires -- align-advance vs brake term, Delta_theta, t_align, violation margin
      (d_req - d_k), across FP / TP / TN.
  D3  re-use each FP episode trajectory -- heading-to-normal angle vs bang-bang t_align; tangential-speed
      fraction |v_perp|/|v| @0 and growth; actual closest approach vs criterion-predicted penetration.
      Classify each FP escape: tangential-evasion / faster-align / closure-overestimate / other.
  D4  candidate-relaxation ROC (report only) -- (a) normal-closure-only (drop align-advance), (b) effective
      velocity max(0, s_k - kappa*|v_perp|) for kappa in {0.5,1.0}, (c) simultaneous align+brake, (d) original
      at margins {-0.1,-0.2}.  TPR (born flagged) / FPR (reach flagged) per variant.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch

from src.common.filter_hardnet import (_base_alpha, _base_projection, _box_aware_projection, _cbf_terms,
                                        _hardnet_params)
from src.common.observation import scene_obstacle_tensors
from src.common.outcomes import resolve_outcome, step_outcomes
from src.common.quadrotor_recoverability import (_min_time_align, is_recoverable, plant_params,
                                                 recoverability_detail)
from src.common.rk4 import rk4_step
from src.common.value_net import ValueNetEnsemble, make_h_fn
from src.eval.build_pools import load_pool
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.frameworks.jt_pncbf.train import make_system, _build_control_net

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
SP = Path("/tmp/claude-1000/-home-junhyeok-MIT-jt-pncbf/31d93785-ac11-4206-bf50-a4c3de145dff/scratchpad")
POOL = REPO / "data/secured_data/pools/eval_full_quadrotor-planar_n2000_seed23456.pkl"
RUN = REPO / "data/runs/v2.6.2/set__20260716-182949__seed42/v2.6.2__20260716-182949__seed42"
TAU_BRAKE = 0.6
MARGIN = 0.0


def scene_ic(sc):
    return (np.asarray(sc.start, np.float64), np.asarray(sc.initial_velocity, np.float64),
            float(sc.initial_attitude or 0.0), float(sc.initial_omega or 0.0),
            np.asarray(sc.obstacle_centers, np.float64), np.asarray(sc.obstacle_radii, np.float64),
            np.asarray(sc.obstacle_active, bool))


def pct(a, ps=(10, 50, 90)):
    a = np.asarray(a, float)
    if a.size == 0:
        return {f"p{p}": None for p in ps}
    return {f"p{p}": round(float(np.percentile(a, p)), 4) for p in ps}


def summ(a):
    a = np.asarray(a, float)
    if a.size == 0:
        return dict(n=0, mean=None, **pct(a))
    return dict(n=int(a.size), mean=round(float(a.mean()), 4), **pct(a))


# ----- D4 relaxation variants: return True == FLAGGED (unrecoverable) -----
def variant_flag(ic, plant, kind, kappa=0.0, margin=0.0):
    """Recompute the flag under a relaxation. ic=(p0,v0,theta0,omega0,centers,radii,active)."""
    p0, v0, theta0, omega0, centers, radii, active = ic
    A = plant["alpha_max"]; W = plant["omega_max"]; g = plant["g"]; ab = plant["a_brake"]
    re = np.array([-math.sin(theta0), math.cos(theta0)])
    for c, r, act in zip(centers, radii, active):
        if not bool(act):
            continue
        rel = p0 - c; dist = float(np.linalg.norm(rel))
        if dist < 1e-9:
            return True
        n = rel / dist; d_k = dist - float(r); s_k = max(0.0, -float(np.dot(v0, n)))
        if s_k <= 0.0:
            continue
        v_perp = float(np.linalg.norm(v0 - float(np.dot(v0, n)) * n))
        cross = re[0]*n[1]-re[1]*n[0]; dot = float(np.dot(re, n)); delta = math.atan2(cross, dot)
        t_align = _min_time_align(delta, float(omega0), A, W)
        a_adv = g * max(0.0, float(n[1]))
        if kind == "normal_closure_only":                    # drop align-advance, keep brake from v_end
            v_end = s_k + a_adv * t_align
            d_req = (v_end*v_end)/(2.0*ab)
        elif kind == "eff_velocity":                         # discount closure by lateral evasion
            s_eff = max(0.0, s_k - kappa*v_perp)
            v_end = s_eff + a_adv*t_align
            d_req = s_eff*t_align + 0.5*a_adv*t_align*t_align + (v_end*v_end)/(2.0*ab)
        elif kind == "simultaneous":                         # brake overlaps align: brake from s_k, no advance
            d_req = (s_k*s_k)/(2.0*ab)
        else:                                                # "original"
            v_end = s_k + a_adv*t_align
            d_req = s_k*t_align + 0.5*a_adv*t_align*t_align + (v_end*v_end)/(2.0*ab)
        if d_k < d_req + margin:
            return True
    return False


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(RUN / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    cfg = ck["config"]; plant = plant_params(cfg)
    system = make_system(cfg); system.u_bounds = system.u_bounds.to(dev, torch.float32)
    vnet = ValueNetEnsemble(system.obs_dim, cfg).to(dev, torch.float32)
    vnet.load_state_dict(ck["v_s_state"]); vnet.eval()
    for p in vnet.parameters():
        p.requires_grad_(False)
    policy = _build_control_net(system, cfg).to(dev); policy.load_state_dict(ck["pi_state"]); policy.eval()
    h_fn = make_h_fn(vnet, system); params = _hardnet_params(cfg); bounds = system.u_bounds
    ms = int(cfg["eval"]["max_steps"]); dt = float(cfg["env"]["dt"])
    scenes = load_pool(POOL).scenes; N = len(scenes)
    print(f"pool N={N} plant={plant}", flush=True)

    born_idx, reach_idx = [], []
    traj = {}                                                # gi -> (T+1,6) trajectory (born + reach only)
    for s0 in range(0, N, 250):
        bs = batch_scenes(scenes[s0:s0+250], device=dev, dtype=torch.float32)
        x = initial_states_from_batch(bs).float(); B = x.shape[0]
        oc, orad, oact = scene_obstacle_tensors(bs, dev, torch.float32)
        p0 = x[:, :2]; v0 = x[:, 3:5]; rel0 = p0.unsqueeze(1) - oc; d0c = torch.linalg.norm(rel0, dim=2)
        nrm0 = rel0 / d0c.unsqueeze(2).clamp_min(1e-9)
        inw0 = torch.relu(-torch.sum(v0.unsqueeze(1)*nrm0, dim=2))
        deficit0 = (torch.relu(inw0*TAU_BRAKE - (d0c-orad))*oact.bool().float()).max(1).values
        empt0 = torch.zeros(B, dtype=torch.bool, device=dev)
        states = [x.clone()]
        with torch.no_grad():
            for t in range(ms):
                un = policy(system.observation(x, bs))
                h, lf, lg = _cbf_terms(system, h_fn, x, bs, un, create_graph=False)
                alpha = _base_alpha(h.detach(), params); row = -lf.detach() - alpha*h.detach()
                proj = _base_projection(un, lg.detach(), row, bounds, params)
                u, empty = _box_aware_projection(un, proj, lg.detach(), row, bounds)
                if t == 0:
                    empt0 = empty
                x = rk4_step(system, x, u, dt); states.append(x.clone())
        S = torch.stack(states, 0)                           # (T+1, B, 6)
        res = resolve_outcome(step_outcomes(S, bs, system, cfg))
        Snp = S.cpu().numpy()
        for i in range(B):
            gi = s0 + i
            if res.outcome[i] == "collision" and (bool(empt0[i]) or float(deficit0[i]) > 0):
                born_idx.append(gi); traj[gi] = Snp[:, i, :]
            elif res.outcome[i] == "goal":
                reach_idx.append(gi); traj[gi] = Snp[:, i, :]
        print(f"  batch {s0}: born={len(born_idx)} reach={len(reach_idx)}", flush=True)

    # -------- D1: reconstruct the per-IC sets @margin 0 --------
    det_born = {gi: recoverability_detail(*scene_ic(scenes[gi]), plant=plant, margin=MARGIN) for gi in born_idx}
    det_reach = {gi: recoverability_detail(*scene_ic(scenes[gi]), plant=plant, margin=MARGIN) for gi in reach_idx}
    TP = [gi for gi in born_idx if det_born[gi]["flagged"]]
    FN = [gi for gi in born_idx if not det_born[gi]["flagged"]]
    FP = [gi for gi in reach_idx if det_reach[gi]["flagged"]]
    TN = [gi for gi in reach_idx if not det_reach[gi]["flagged"]]
    d1 = dict(n_born=len(born_idx), n_reach=len(reach_idx),
              TP=len(TP), FN=len(FN), FP=len(FP), TN=len(TN),
              TPR=round(len(TP)/max(1, len(born_idx)), 4), FPR=round(len(FP)/max(1, len(reach_idx)), 4))
    print("D1", json.dumps(d1), flush=True)

    # -------- D2: term decomposition FP vs TP vs TN --------
    def terms(idxs, det):
        rows = [det[gi] for gi in idxs]
        # only obstacles that produced a closing term have finite parts (binding exists) for TN some are receding
        clos = [r for r in rows if r["s_k"] > 0]
        return dict(
            frac_binding=round(len(clos)/max(1, len(rows)), 4),
            align_advance=summ([r["align_advance"] for r in clos]),
            brake_dist=summ([r["brake_dist"] for r in clos]),
            align_over_dreq=summ([r["align_advance"]/max(1e-9, r["d_req"]) for r in clos]),
            delta_theta=summ([r["delta_theta"] for r in clos]),
            t_align=summ([r["t_align"] for r in clos]),
            s_k=summ([r["s_k"] for r in clos]),
            d_k=summ([r["d_k"] for r in clos]),
            v_perp=summ([r["v_perp"] for r in clos]),
            viol_margin=summ([r["d_req"]-r["d_k"] for r in clos]),      # >0 == flagged; magnitude == depth
        )
    d2 = dict(FP=terms(FP, det_reach), TP=terms(TP, det_born), TN=terms(TN, det_reach))
    print("D2 FP align_advance", d2["FP"]["align_advance"], "brake", d2["FP"]["brake_dist"], flush=True)
    print("D2 FP align/d_req", d2["FP"]["align_over_dreq"], "viol_margin", d2["FP"]["viol_margin"], flush=True)

    # -------- D3: FP escape mechanism from trajectories --------
    radii_pool = {gi: np.asarray(scenes[gi].obstacle_radii, np.float64) for gi in FP}
    d3_rows = []
    classes = {"tangential-evasion": 0, "faster-align": 0, "closure-overestimate": 0, "other": 0}
    for gi in FP:
        r = det_reach[gi]; oi = r["obs_idx"]; c = r["center"]; rad = r["radius"]; n0 = r["n"]
        Straj = traj[gi]; Pt = Straj[:, :2]; Vt = Straj[:, 3:5]; Th = Straj[:, 2]
        rel = Pt - c; dist = np.linalg.norm(rel, axis=1); dk_t = dist - rad
        nt = rel / np.clip(dist, 1e-9, None)[:, None]
        # heading-to-normal angle: thrust axis Re(theta) vs instantaneous outward normal
        re_t = np.stack([-np.sin(Th), np.cos(Th)], 1)
        cos_align = np.clip(np.sum(re_t*nt, axis=1), -1, 1)
        ang = np.arccos(cos_align)                                       # 0 == aligned to brake
        # actual time to reach the aligned cone (thrust within ~20deg of outward normal)
        aligned = np.where(ang < math.radians(20))[0]
        t_align_actual = float(aligned[0]*dt) if aligned.size else float("inf")
        # tangential fraction |v_perp|/|v|
        vmag = np.linalg.norm(Vt, axis=1)
        vn = np.sum(Vt*nt, axis=1)
        vperp = np.linalg.norm(Vt - vn[:, None]*nt, axis=1)
        tang0 = float(vperp[0]/max(1e-6, vmag[0]))
        tang_max = float(np.max(vperp[:10]/np.clip(vmag[:10], 1e-6, None)))
        d_min_actual = float(np.min(dk_t))
        d_min_pred = r["slack"]                                          # d_k - d_req (<0 == predicted penetration)
        overest = d_min_actual - d_min_pred
        # classify
        if tang0 >= 0.40:
            cls = "tangential-evasion"
        elif t_align_actual <= 0.7*max(1e-6, r["t_align"]):
            cls = "faster-align"
        elif overest >= 0.30:
            cls = "closure-overestimate"
        else:
            cls = "other"
        classes[cls] += 1
        d3_rows.append(dict(gi=gi, cls=cls, t_align_crit=round(r["t_align"], 4),
                            t_align_actual=round(t_align_actual, 4), tang0=round(tang0, 4),
                            tang_max=round(tang_max, 4), d_min_actual=round(d_min_actual, 4),
                            d_min_pred=round(float(d_min_pred), 4), overest=round(overest, 4)))
    d3 = dict(classes=classes,
              tang0=summ([r["tang0"] for r in d3_rows]),
              t_align_ratio=summ([r["t_align_actual"]/max(1e-6, r["t_align_crit"])
                                  for r in d3_rows if math.isfinite(r["t_align_actual"])]),
              overest=summ([r["overest"] for r in d3_rows]),
              d_min_actual=summ([r["d_min_actual"] for r in d3_rows]),
              never_aligned_frac=round(np.mean([not math.isfinite(r["t_align_actual"]) for r in d3_rows]), 4))
    print("D3 classes", classes, flush=True)
    print("D3 tang0", d3["tang0"], "overest", d3["overest"], flush=True)

    # -------- D4: relaxation ROC --------
    born_ics = {gi: scene_ic(scenes[gi]) for gi in born_idx}
    reach_ics = {gi: scene_ic(scenes[gi]) for gi in reach_idx}

    def roc(kind, kappa=0.0, margin=0.0):
        tpr = np.mean([variant_flag(born_ics[gi], plant, kind, kappa, margin) for gi in born_idx])
        fpr = np.mean([variant_flag(reach_ics[gi], plant, kind, kappa, margin) for gi in reach_idx])
        return dict(TPR=round(float(tpr), 4), FPR=round(float(fpr), 4))

    d4 = {
        "original@0.0": roc("original", margin=0.0),
        "original@-0.1": roc("original", margin=-0.1),
        "original@-0.2": roc("original", margin=-0.2),
        "normal_closure_only@0.0": roc("normal_closure_only", margin=0.0),
        "eff_velocity_k0.5@0.0": roc("eff_velocity", kappa=0.5, margin=0.0),
        "eff_velocity_k1.0@0.0": roc("eff_velocity", kappa=1.0, margin=0.0),
        "simultaneous@0.0": roc("simultaneous", margin=0.0),
    }
    print("D4", json.dumps(d4), flush=True)

    out = dict(margin=MARGIN, run=str(RUN.name), pool=str(POOL.name),
               D1=d1, D2=d2, D3=d3, D3_rows=d3_rows, D4=d4)
    json.dump(out, open(SP / "quadrotor_recov_fp_anatomy.json", "w"), indent=2)
    print("WROTE", SP / "quadrotor_recov_fp_anatomy.json", flush=True)


if __name__ == "__main__":
    main()
