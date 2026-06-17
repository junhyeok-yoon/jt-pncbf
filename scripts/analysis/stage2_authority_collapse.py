"""v2.2.0 Stage 2 — WHY does L_g h = d h / d v collapse at the boundary? (read-only diagnosis)

Diagnostic 3 established PNCBF collisions are 100% control-authority failures: V_S alarms (h->1)
but L_g h -> 0, so the alpha_unsafe=100 braking demand is infeasible. This script diagnoses the
CAUSE of the L_g h collapse. No fix. Read-only: V_S queried by forward + autograd on stored states.

L_g h chain rule (verified): g maps u into the velocity channels, so L_g h = grad_x h restricted to
the velocity components = d h / d v. obs = [vx,vy, gx-px,gy-py, 5x(cx-px,cy-py,r)]; only obs[0:2]
depend on v (identity), so d h / d v = d h / d obs[0:2] = grad_x(h)[2:4]; the "position sensitivity"
is grad_x(h)[0:2] (through the rel-goal/rel-obstacle channels). The deployed h = mean over the
2-member ensemble of clamp(member(obs), -1, 1); the clamp zeroes the gradient of any saturated
member.

Parts: A characterize the collapse (||L_g h|| vs h, colliding vs non-colliding); B attribute it
(head clip-saturation; alpha_unsafe critical-alpha what-if; gradient-vs-density tail; velocity-vs-
position sensitivity); C contrast with the analytic HOCBF L_g psi1 = -2(p-c); D verdict.

Run: /home/junhyeok/miniconda3/envs/pncbf/bin/python scripts/analysis/stage2_authority_collapse.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (str(REPO_ROOT), str(REPO_ROOT / "scripts/analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)

import stage2_hocbf_deploy_n2000 as D  # noqa: E402
from src.common.filter_hardnet import _cbf_terms, _hardnet_params  # noqa: E402
from src.common.value_net import make_h_fn  # noqa: E402
from src.envs.scene_batch import batch_scenes  # noqa: E402
from src.envs.scene_init import Scene  # noqa: E402
from src.frameworks.jt_pncbf.train import load_framework_from_checkpoint, make_system  # noqa: E402

OUT = D.OUT
LARGEN_DIR = REPO_ROOT / "data/diagnostics/v2.2.0_stage2_largeN/episodes"
U_MAX = 2.0
ALPHA_UNSAFE = 100.0
ALPHA_SAFE = 2.0
SINGULAR_TOL = 5.0e-4
NEAR_CLEAR = 0.6          # near-obstacle clearance for the non-colliding reference set
CLIP = 1.0
H_BINS = np.linspace(0.0, 1.0, 21)


def load_scene(z):
    return Scene(obstacle_centers=np.asarray(z["obstacle_centers"], float),
                 obstacle_radii=np.asarray(z["obstacle_radii"], float),
                 obstacle_active=np.asarray(z["obstacle_active"], bool),
                 start=np.asarray(z["start"], float), goal=np.asarray(z["goal"], float),
                 system="double_integrator", mode="eval",
                 initial_velocity=np.asarray(z["initial_velocity"], float))


def nearest_pc(states, scene):
    """p - c for the nearest active obstacle (by surface distance) per step. Returns [S,2], r [S]."""
    p = states[:, :2]
    c = np.asarray(scene.obstacle_centers, float); r = np.asarray(scene.obstacle_radii, float)
    a = np.asarray(scene.obstacle_active, bool)
    rel = p[:, None, :] - c[None, :, :]
    surf = np.where(a[None, :], np.linalg.norm(rel, axis=2) - r[None, :], np.inf)
    j = np.argmin(surf, axis=1)
    si = np.arange(states.shape[0])
    return rel[si, j], r[j]


def collect_collision_approaches():
    """Per collision NPZ: steps from first alarm (h>0) to event_action. Stored fields."""
    fs = sorted(glob.glob(str(LARGEN_DIR / "ep_*_collision.npz")))
    rows = {k: [] for k in ("h", "lg_norm", "lf_h", "clearance", "lgx", "lgy", "pcx", "pcy", "pcr")}
    for f in fs:
        z = np.load(f, allow_pickle=False)
        md = json.loads(str(np.asarray(z["metadata_json"]).item()))
        T = len(z["h"]); ev = int(md["event_step"]); ea = max(0, min(ev - 1, T - 1)) if ev > 0 else T - 1
        h = np.asarray(z["h"], float)
        alarm = np.where(h[: ea + 1] > 0.0)[0]
        lo = int(alarm[0]) if alarm.size else ea
        idx = np.arange(lo, ea + 1)
        if idx.size == 0:
            continue
        lg = np.asarray(z["lg_h"], float)[idx]
        pc, pr = nearest_pc(np.asarray(z["states"], float)[idx], load_scene(z))
        rows["h"].append(h[idx]); rows["lg_norm"].append(np.asarray(z["lg_norm"], float)[idx])
        rows["lf_h"].append(np.asarray(z["lf_h"], float)[idx]); rows["clearance"].append(np.asarray(z["clearance"], float)[idx])
        rows["lgx"].append(lg[:, 0]); rows["lgy"].append(lg[:, 1])
        rows["pcx"].append(pc[:, 0]); rows["pcy"].append(pc[:, 1]); rows["pcr"].append(pr)
    return {k: np.concatenate(v) for k, v in rows.items()}


def collect_goal_near():
    """Non-colliding near-obstacle reference: goal-episode steps with clearance < NEAR_CLEAR."""
    fs = sorted(glob.glob(str(LARGEN_DIR / "ep_*_goal.npz")))
    rows = {k: [] for k in ("h", "lg_norm", "clearance")}
    for f in fs:
        z = np.load(f, allow_pickle=False)
        cl = np.asarray(z["clearance"], float)
        m = cl < NEAR_CLEAR
        if not m.any():
            continue
        rows["h"].append(np.asarray(z["h"], float)[m]); rows["lg_norm"].append(np.asarray(z["lg_norm"], float)[m])
        rows["clearance"].append(cl[m])
    return {k: np.concatenate(v) for k, v in rows.items()}


def manifold_clearance_lg():
    """Visited-manifold proxy = ALL goal-episode steps: clearance histogram + lg_norm per clearance bin."""
    fs = sorted(glob.glob(str(LARGEN_DIR / "ep_*_goal.npz")))
    cl, lg = [], []
    for f in fs:
        z = np.load(f, allow_pickle=False)
        cl.append(np.asarray(z["clearance"], float)); lg.append(np.asarray(z["lg_norm"], float))
    return np.concatenate(cl), np.concatenate(lg)


def med_per_bin(x, y, bins):
    out = np.full(len(bins) - 1, np.nan)
    for i in range(len(bins) - 1):
        m = (x >= bins[i]) & (x < bins[i + 1])
        if m.sum() >= 5:
            out[i] = np.median(y[m])
    return out


# ---- autograd recompute (reconstruction check, pre-clip gradient, vel-vs-pos split) ---

def recompute_grads(states, scene, fw, system, device, dtype):
    """Returns h, lf_h, lg(=dh/dv [S,2]), pos_grad(=dh/dp [S,2]), and PRE-CLIP dh_raw/dv [S,2]."""
    h_fn = make_h_fn(fw.value_net, system)
    S = states.shape[0]
    bscene = batch_scenes([scene] * S, device=device, dtype=dtype)
    x = torch.as_tensor(states, device=device, dtype=dtype).requires_grad_(True)
    obs = system.observation(x, bscene)
    h = fw.value_net.deployed_h(obs)                      # mean of clamp(member,-1,1)
    grad = torch.autograd.grad(h.sum(), x, create_graph=False)[0]
    pos_grad = grad[:, 0:2].detach().cpu().numpy(); lg = grad[:, 2:4].detach().cpu().numpy()
    h_np = h.detach().cpu().numpy()
    # pre-clip: mean of raw member outputs (no clamp)
    x2 = torch.as_tensor(states, device=device, dtype=dtype).requires_grad_(True)
    obs2 = system.observation(x2, bscene)
    h_raw = fw.value_net.forward_all(obs2).mean(dim=1)
    grad_raw = torch.autograd.grad(h_raw.sum(), x2, create_graph=False)[0]
    lg_raw = grad_raw[:, 2:4].detach().cpu().numpy()
    raw_members = fw.value_net.forward_all(obs2).detach().cpu().numpy()   # [S, n_vs] pre-clip member outputs
    return h_np, lg, pos_grad, lg_raw, raw_members


def main():
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); dtype = torch.float32
    fw, config, _ = load_framework_from_checkpoint(D.CKPT)
    system = make_system(config)
    fw.value_net.to(device, dtype).eval(); fw.policy_net.to(device, dtype).eval()
    params = _hardnet_params(config)
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {"thresholds": {"singular_tol": SINGULAR_TOL, "near_clearance": NEAR_CLEAR, "u_max": U_MAX,
                          "alpha_safe": ALPHA_SAFE, "alpha_unsafe": ALPHA_UNSAFE, "clip": CLIP},
           "value_head": {"n_vs": 2, "n_layers": 3, "hidden": 256, "softplus_beta": 20.0,
                          "head": "linear_clipped (clamp to [-1,1])"},
           "lg_chain_rule": "L_g h = grad_x(h)[velocity] = d h / d obs[0:2]; verified vs stored lg_h."}

    coll = collect_collision_approaches()
    goal = collect_goal_near()
    print(f"collision-approach steps: {coll['h'].size} | goal near-obstacle steps: {goal['h'].size}")

    # ---- reconstruction check on a collision-impact sample ----
    fs = sorted(glob.glob(str(LARGEN_DIR / "ep_*_collision.npz")))
    z = np.load(fs[0], allow_pickle=False); T = len(z["h"])
    h_np, lg, pos_grad, lg_raw, raw = recompute_grads(np.asarray(z["states"], float)[:T], load_scene(z), fw, system, device, dtype)
    rep["reconstruction_check"] = {
        "max_abs_h": float(np.max(np.abs(h_np - np.asarray(z["h"], float)))),
        "max_abs_lg_norm": float(np.max(np.abs(np.linalg.norm(lg, axis=1) - np.asarray(z["lg_norm"], float))))}
    print("reconstruction:", rep["reconstruction_check"])

    # ---- Part A ----
    lg1_coll = np.abs(coll["lgx"]) + np.abs(coll["lgy"])           # ||L_g h||_1
    auth_demand = coll["lf_h"] + ALPHA_UNSAFE * coll["h"]          # need u_max*||A||_1 >= this for feasible row
    authority_lost = U_MAX * lg1_coll < auth_demand
    curve_coll = med_per_bin(coll["h"], coll["lg_norm"], H_BINS)
    curve_goal = med_per_bin(goal["h"], goal["lg_norm"], H_BINS)
    # h at which authority lost (median over collision steps with h>0): first h-bin where >50% authority_lost
    frac_lost_bin = np.array([np.mean(authority_lost[(coll["h"] >= H_BINS[i]) & (coll["h"] < H_BINS[i + 1])])
                              if ((coll["h"] >= H_BINS[i]) & (coll["h"] < H_BINS[i + 1])).sum() >= 5 else np.nan
                              for i in range(len(H_BINS) - 1)])
    sing_bin = np.array([np.mean(coll["lg_norm"][(coll["h"] >= H_BINS[i]) & (coll["h"] < H_BINS[i + 1])] < SINGULAR_TOL)
                         if ((coll["h"] >= H_BINS[i]) & (coll["h"] < H_BINS[i + 1])).sum() >= 5 else np.nan
                         for i in range(len(H_BINS) - 1)])
    rep["partA_collapse"] = {
        "median_lg_norm_by_h_bin_collision": [None if np.isnan(v) else float(v) for v in curve_coll],
        "median_lg_norm_by_h_bin_goalnear": [None if np.isnan(v) else float(v) for v in curve_goal],
        "h_bin_edges": [float(b) for b in H_BINS],
        "frac_authority_lost_by_h_bin": [None if np.isnan(v) else float(v) for v in frac_lost_bin],
        "median_lg_norm_collision_h_gt_0.8": float(np.median(coll["lg_norm"][coll["h"] > 0.8])),
        "median_lg_norm_collision_h_in_0.0_0.3": float(np.median(coll["lg_norm"][(coll["h"] > 0) & (coll["h"] < 0.3)])) if ((coll["h"] > 0) & (coll["h"] < 0.3)).any() else None,
        "frac_authority_lost_overall": float(np.mean(authority_lost)),
        "frac_singular_overall": float(np.mean(coll["lg_norm"] < SINGULAR_TOL))}
    # A.2 matched (h, clearance) comparison
    matched = []
    for hlo, hhi in [(0.3, 0.6), (0.6, 0.8), (0.8, 1.01)]:
        for clo, chi in [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6)]:
            mc = (coll["h"] >= hlo) & (coll["h"] < hhi) & (coll["clearance"] >= clo) & (coll["clearance"] < chi)
            mg = (goal["h"] >= hlo) & (goal["h"] < hhi) & (goal["clearance"] >= clo) & (goal["clearance"] < chi)
            if mc.sum() >= 5 and mg.sum() >= 5:
                matched.append({"h": [hlo, hhi], "clearance": [clo, chi],
                                "n_coll": int(mc.sum()), "n_goal": int(mg.sum()),
                                "med_lg_collision": float(np.median(coll["lg_norm"][mc])),
                                "med_lg_goalnear": float(np.median(goal["lg_norm"][mg]))})
    rep["partA2_matched"] = matched

    # ---- Part B.1 head clip-saturation (recompute on collision-approach states, per episode) ----
    n_clip = 0; n_tot = 0; ratios = []; raw_clipped_frac = []
    for f in fs:
        z = np.load(f, allow_pickle=False)
        md = json.loads(str(np.asarray(z["metadata_json"]).item()))
        T = len(z["h"]); ev = int(md["event_step"]); ea = max(0, min(ev - 1, T - 1)) if ev > 0 else T - 1
        h = np.asarray(z["h"], float); alarm = np.where(h[: ea + 1] > 0.0)[0]
        lo = int(alarm[0]) if alarm.size else ea; idx = np.arange(lo, ea + 1)
        st = np.asarray(z["states"], float)[idx]
        hh, lg2, pg2, lg_raw2, raw2 = recompute_grads(st, load_scene(z), fw, system, device, dtype)
        n_tot += hh.size
        n_clip += int(np.sum(np.abs(hh) >= CLIP - 1e-6))
        raw_clipped_frac.append(np.mean(np.abs(raw2) >= CLIP - 1e-6))   # per-member clip fraction
        post = np.linalg.norm(lg2, axis=1); pre = np.linalg.norm(lg_raw2, axis=1)
        ratios.append(post / np.clip(pre, 1e-9, None))
    ratios = np.concatenate(ratios)
    rep["partB1_head_clip"] = {
        "frac_impact_states_h_at_clip(|h|>=1)": n_clip / n_tot,
        "median_per_member_clip_fraction": float(np.mean(np.concatenate([np.atleast_1d(x) for x in raw_clipped_frac]))),
        "median_postclip_over_preclip_lgnorm_ratio": float(np.median(ratios)),
        "frac_steps_postclip_lt_10pct_of_preclip": float(np.mean(ratios < 0.1)),
        "note": "post/pre << 1 => the clamp (not the Softplus) zeroes the velocity gradient; pre-clip gradient is alive."}

    # ---- Part B.2 critical-alpha what-if (stored A,b at collision approach) ----
    h_pos = coll["h"] > 1e-6
    crit_alpha = (U_MAX * lg1_coll[h_pos] - coll["lf_h"][h_pos]) / coll["h"][h_pos]
    rep["partB2_critical_alpha"] = {
        "definition": "largest alpha with min_box(A.u)=-u_max*||A||_1 <= -L_f h - alpha*h  =>  alpha <= (u_max*||A||_1 - L_f h)/h",
        "median": float(np.median(crit_alpha)), "p10": float(np.percentile(crit_alpha, 10)),
        "p90": float(np.percentile(crit_alpha, 90)),
        "frac_le_0_authority_bound_gain_cannot_fix": float(np.mean(crit_alpha <= 0.0)),
        "frac_feasible_at_alpha_safe_2": float(np.mean(crit_alpha >= ALPHA_SAFE)),
        "frac_feasible_at_alpha_unsafe_100": float(np.mean(crit_alpha >= ALPHA_UNSAFE)),
        "note": "READ-ONLY algebraic what-if on stored A,b; a real lower-alpha deploy is a separate test."}

    # ---- Part B.3 gradient vs manifold density (clearance) ----
    mcl, mlg = manifold_clearance_lg()
    cl_bins = np.linspace(0.0, 1.5, 16)
    dens, _ = np.histogram(mcl, bins=cl_bins, density=False)
    lg_by_cl = med_per_bin(mcl, mlg, cl_bins)
    tail_frac = float(np.mean(mcl < 0.1))
    rep["partB3_gradient_vs_density"] = {
        "manifold": "all goal-episode steps (success-manifold proxy)", "n_steps": int(mcl.size),
        "clearance_bin_edges": [float(b) for b in cl_bins],
        "density_counts": [int(d) for d in dens],
        "median_lg_norm_by_clearance_bin": [None if np.isnan(v) else float(v) for v in lg_by_cl],
        "near_boundary_tail_frac_clearance_lt_0.1": tail_frac,
        "median_lg_norm_clearance_lt_0.2": float(np.median(mlg[mcl < 0.2])) if (mcl < 0.2).any() else None,
        "median_lg_norm_clearance_gt_1.0": float(np.median(mlg[mcl > 1.0])) if (mcl > 1.0).any() else None}

    # ---- Part B.4 velocity vs position sensitivity (recompute grad split) ----
    vel_mag, pos_mag, h_for = [], [], []
    for f in fs:
        z = np.load(f, allow_pickle=False)
        md = json.loads(str(np.asarray(z["metadata_json"]).item()))
        T = len(z["h"]); ev = int(md["event_step"]); ea = max(0, min(ev - 1, T - 1)) if ev > 0 else T - 1
        h = np.asarray(z["h"], float); alarm = np.where(h[: ea + 1] > 0.0)[0]
        lo = int(alarm[0]) if alarm.size else ea; idx = np.arange(lo, ea + 1)
        st = np.asarray(z["states"], float)[idx]
        hh, lg2, pg2, _, _ = recompute_grads(st, load_scene(z), fw, system, device, dtype)
        vel_mag.append(np.linalg.norm(lg2, axis=1)); pos_mag.append(np.linalg.norm(pg2, axis=1)); h_for.append(hh)
    vel_mag = np.concatenate(vel_mag); pos_mag = np.concatenate(pos_mag); h_for = np.concatenate(h_for)
    rep["partB4_velocity_vs_position"] = {
        "median_velocity_grad(L_g h)": float(np.median(vel_mag)),
        "median_position_grad(dh/dp)": float(np.median(pos_mag)),
        "median_ratio_vel_over_pos": float(np.median(vel_mag / np.clip(pos_mag, 1e-9, None))),
        "vel_grad_by_h_bin": [None if np.isnan(v) else float(v) for v in med_per_bin(h_for, vel_mag, H_BINS)],
        "pos_grad_by_h_bin": [None if np.isnan(v) else float(v) for v in med_per_bin(h_for, pos_mag, H_BINS)]}

    # ---- Part C contrast with analytic HOCBF L_g psi1 = -2(p-c) ----
    pc = np.stack([coll["pcx"], coll["pcy"]], axis=1)
    lg_vec = np.stack([coll["lgx"], coll["lgy"]], axis=1)
    hocbf_A = -2.0 * pc                                            # row coefficient on u
    mag_h = np.linalg.norm(hocbf_A, axis=1); mag_lg = coll["lg_norm"]
    ratio = mag_lg / np.clip(mag_h, 1e-9, None)
    cos = np.sum(lg_vec * hocbf_A, axis=1) / np.clip(np.linalg.norm(lg_vec, axis=1) * mag_h, 1e-9, None)
    rep["partC_hocbf_contrast"] = {
        "median_||L_g h||": float(np.median(mag_lg)), "median_||L_g psi1||=2||p-c||": float(np.median(mag_h)),
        "median_magnitude_ratio_learned_over_analytic": float(np.median(ratio)),
        "ratio_by_h_bin": [None if np.isnan(v) else float(v) for v in med_per_bin(coll["h"], ratio, H_BINS)],
        "median_direction_cosine(L_g h, -2(p-c))": float(np.median(cos)),
        "frac_cosine_gt_0.7_aligned": float(np.mean(cos > 0.7)),
        "frac_cosine_lt_0_misoriented": float(np.mean(cos < 0.0)),
        "note": "L_g psi1=-2(p-c) is bounded away from 0 near the obstacle; the learned L_g h is the row coefficient d h/d v."}

    _figs(coll, goal, curve_coll, curve_goal, crit_alpha, mcl, cl_bins, dens, lg_by_cl,
          vel_mag, pos_mag, h_for, ratio, cos)
    (OUT / "authority_collapse_summary.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("summary ->", OUT / "authority_collapse_summary.json")
    print("B1 head-clip:", rep["partB1_head_clip"])
    print("B2 critical-alpha:", {k: rep["partB2_critical_alpha"][k] for k in ("median", "frac_le_0_authority_bound_gain_cannot_fix", "frac_feasible_at_alpha_safe_2")})
    print("B4 vel/pos:", {k: rep["partB4_velocity_vs_position"][k] for k in ("median_velocity_grad(L_g h)", "median_position_grad(dh/dp)", "median_ratio_vel_over_pos")})
    print("C ratio/cos:", rep["partC_hocbf_contrast"]["median_magnitude_ratio_learned_over_analytic"], rep["partC_hocbf_contrast"]["median_direction_cosine(L_g h, -2(p-c))"])
    return 0


def _figs(coll, goal, curve_coll, curve_goal, crit_alpha, mcl, cl_bins, dens, lg_by_cl,
          vel_mag, pos_mag, h_for, ratio, cos):
    hc = 0.5 * (H_BINS[:-1] + H_BINS[1:])
    fig, ax = plt.subplots(2, 3, figsize=(18, 11), dpi=140)
    a = ax[0, 0]
    a.semilogy(hc, curve_coll, "o-", color="#d62728", label="collision-approach")
    a.semilogy(hc, curve_goal, "s--", color="#1f77b4", label="goal near-obstacle")
    a.axhline(SINGULAR_TOL, color="k", ls=":", lw=1, label="singular 5e-4")
    a.set_xlabel("h = V_S"); a.set_ylabel("median ||L_g h||"); a.set_title("A: control authority vs h (log)")
    a.legend(fontsize=8)
    a = ax[0, 1]
    a.hist(np.clip(crit_alpha, -20, 120), bins=60, color="#7f4fbf")
    a.axvline(ALPHA_UNSAFE, color="r", ls="--", label="alpha_unsafe=100"); a.axvline(ALPHA_SAFE, color="g", ls="--", label="alpha_safe=2")
    a.axvline(0, color="k", ls=":"); a.set_xlabel("critical alpha (feasible row iff alpha<=this)")
    a.set_title("B2: critical-alpha what-if"); a.legend(fontsize=8)
    a = ax[0, 2]
    clc = 0.5 * (cl_bins[:-1] + cl_bins[1:])
    a.bar(clc, dens / dens.sum(), width=(cl_bins[1] - cl_bins[0]) * 0.9, alpha=0.4, color="0.6", label="manifold density")
    a2 = a.twinx(); a2.semilogy(clc, lg_by_cl, "o-", color="#d62728", label="median ||L_g h||")
    a.set_xlabel("clearance"); a.set_ylabel("visited density"); a2.set_ylabel("median ||L_g h||")
    a.set_title("B3: gradient vs manifold density"); a.legend(fontsize=8, loc="upper right")
    a = ax[1, 0]
    a.semilogy(hc, [np.nan if v is None else v for v in med_per_bin(h_for, vel_mag, H_BINS)], "o-", color="#d62728", label="||dh/dv|| (L_g h)")
    a.semilogy(hc, [np.nan if v is None else v for v in med_per_bin(h_for, pos_mag, H_BINS)], "s--", color="#2ca02c", label="||dh/dp||")
    a.set_xlabel("h"); a.set_ylabel("median grad"); a.set_title("B4: velocity vs position sensitivity"); a.legend(fontsize=8)
    a = ax[1, 1]
    a.semilogy(hc, [np.nan if v is None else v for v in med_per_bin(coll["h"], ratio, H_BINS)], "o-", color="#d62728")
    a.set_xlabel("h"); a.set_ylabel("||L_g h|| / ||2(p-c)||"); a.set_title("C: learned/analytic authority ratio")
    a = ax[1, 2]
    a.hist(cos, bins=40, color="#1f77b4"); a.axvline(0, color="k", ls=":")
    a.set_xlabel("cos(L_g h, -2(p-c))"); a.set_title("C: gradient direction vs analytic")
    fig.tight_layout(); fig.savefig(OUT / "authority_collapse.png"); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
