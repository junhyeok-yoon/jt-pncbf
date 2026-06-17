"""v2.2.0 Stage 2 — does the learned nominal policy steer around obstacles, or go straight at goal?

Read-only: the deployed ControlNet is loaded only to recompute u_nom on STORED states (states +
stored scene -> obs via DoubleIntegrator.observation) and for an obs-perturbation sensitivity
probe. No training, no env rollout, no Fisher. Only this report + script + figures. Deterministic.

obs (DI, dim 19): [vx, vy, rel_gx, rel_gy, then 5 x (rel_cx, rel_cy, r)], rel_c = center - position.
ĝ = (rel_gx, rel_gy) normalized; direction agent->nearest-obstacle = center - pos = -nearest_rel_pos.

Run: /home/junhyeok/miniconda3/envs/pncbf/bin/python scripts/analysis/stage2_policy_obstacle_response.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.frameworks.jt_pncbf.train import load_framework_from_checkpoint, make_system  # noqa: E402

CKPT = REPO_ROOT / "data/secured_data/v2.0.1/seed42/checkpoints/best.pt"
OUT = REPO_ROOT / "data/diagnostics/v2.2.0_stage2_largeN"
EPISODE_DIR = OUT / "episodes"
LABELS = OUT / "stage2_failure_labels.csv"
STUCK_WINDOW, STUCK_RADIUS = 60, 0.10
D_PATH, BEARING = 0.5, 45.0
PROBE_SEED, PROBE_DELTA, PROBE_STRIDE = 20260616, 0.2, 7


def unit(v):
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9, None)


def ang(a, b):
    return np.degrees(np.arccos(np.clip(np.einsum("md,md->m", unit(a), unit(b)), -1, 1)))


def physical_onset(states):
    pos = states[:, :2]; T = pos.shape[0]
    for t in range(STUCK_WINDOW, T):
        if np.linalg.norm(pos[t - STUCK_WINDOW:t + 1] - pos[t - STUCK_WINDOW], axis=1).max() <= STUCK_RADIUS:
            return max(0, t - STUCK_WINDOW)
    return None


def main():
    torch.manual_seed(0)
    fw, config, _ = load_framework_from_checkpoint(CKPT)
    system = make_system(config)
    policy = fw.policy_net.to("cpu", torch.float32).eval()
    labels = {int(r["episode_idx"]): r for r in csv.DictReader(LABELS.open())} if LABELS.exists() else {}

    cols = ("outcome", "stuck_phase", "near", "inpath", "bearing", "theta_goal", "gp_nom", "gp_safe",
            "ang_nom_safe", "defl_tan", "out_nom", "out_filt", "clearance")
    R = {c: [] for c in cols}
    max_mismatch = 0.0
    probe_samples = []
    rng = np.random.default_rng(PROBE_SEED)

    for path in sorted(EPISODE_DIR.glob("ep_*.npz")):
        z = np.load(path, allow_pickle=False)
        md = json.loads(str(np.asarray(z["metadata_json"]).item()))
        outcome = str(md["outcome"]); n = int(len(z["h"])); ns = min(int(md.get("active_steps", n)), n)
        states = np.asarray(z["states"], float)
        scene = SimpleNamespace(goal=np.asarray(z["goal"], float),
                                obstacle_centers=np.asarray(z["obstacle_centers"], float),
                                obstacle_radii=np.asarray(z["obstacle_radii"], float),
                                obstacle_active=np.asarray(z["obstacle_active"], bool))
        x = torch.as_tensor(states[:ns], dtype=torch.float32)
        with torch.no_grad():
            obs = system.observation(x, scene)
            un = policy(obs).numpy()
        max_mismatch = max(max_mismatch, float(np.max(np.abs(un - z["u_nom"][:ns]))) if ns else 0.0)
        us = np.asarray(z["u_safe"], float)[:ns]
        g = np.asarray(z["goal_rel_pos"], float)[:ns]; nr = np.asarray(z["nearest_rel_pos"], float)[:ns]
        cl = np.asarray(z["clearance"], float)[:ns]
        ghat = unit(g); dir_obs = unit(-nr); outward = unit(nr)
        bearing = ang(dir_obs, g)
        near = cl < D_PATH; inpath = near & (bearing < BEARING)
        gp_nom = np.einsum("md,md->m", un, ghat); gp_safe = np.einsum("md,md->m", us, ghat)
        defl = un - gp_nom[:, None] * ghat
        tang = np.stack([-dir_obs[:, 1], dir_obs[:, 0]], axis=1)
        defl_tan = np.degrees(np.arccos(np.clip(np.abs(np.einsum("md,md->m", unit(defl), tang)), -1, 1)))
        sp = np.zeros(ns, bool)
        on = physical_onset(states)
        if on is not None and outcome == "stuck":
            sp[on:min(ns, on + STUCK_WINDOW)] = True
        vals = {"outcome": np.array([outcome] * ns), "stuck_phase": sp, "near": near, "inpath": inpath,
                "bearing": bearing, "theta_goal": ang(un, g), "gp_nom": gp_nom, "gp_safe": gp_safe,
                "ang_nom_safe": ang(un, us), "defl_tan": defl_tan,
                "out_nom": np.einsum("md,md->m", un, outward), "out_filt": np.einsum("md,md->m", us - un, outward),
                "clearance": cl}
        for c in cols:
            R[c].append(vals[c])
        for t in range(0, ns, PROBE_STRIDE):
            probe_samples.append((states[t], scene))

    R = {c: np.concatenate(v) for c, v in R.items()}
    outc = R["outcome"]
    if max_mismatch > 1e-3:
        raise SystemExit(f"STOP: recomputed u_nom mismatch {max_mismatch:.3e} (obs reconstruction wrong)")

    rep = {"recon_max_mismatch_unom": max_mismatch,
           "counts": {o: int((outc == o).sum()) for o in ("goal", "stuck", "collision", "timeout")},
           "thresholds": {"D_path": D_PATH, "bearing_deg": BEARING}}

    def med(a):
        return float(np.median(a)) if a.size else None

    # Part A
    tg = R["theta_goal"]
    rep["partA"] = {
        "theta_goal_inpath_median": med(tg[R["inpath"]]), "n_inpath": int(R["inpath"].sum()),
        "theta_goal_not_inpath_median": med(tg[~R["inpath"]]), "n_not_inpath": int((~R["inpath"]).sum()),
        "gap_median_deg": (med(tg[R["inpath"]]) - med(tg[~R["inpath"]])) if R["inpath"].any() else None,
        "deflection_to_tangent_median_deg_inpath": med(R["defl_tan"][R["inpath"]]),
        "sensitivity": {f"D{dp}_B{bg}": {
            "inpath_theta_median": med(tg[(R["clearance"] < dp) & (R["bearing"] < bg)]),
            "n_inpath": int(((R["clearance"] < dp) & (R["bearing"] < bg)).sum()),
            "gap_vs_rest": (med(tg[(R["clearance"] < dp) & (R["bearing"] < bg)]) -
                            med(tg[~((R["clearance"] < dp) & (R["bearing"] < bg))]))
            if ((R["clearance"] < dp) & (R["bearing"] < bg)).any() else None}
            for dp in (0.3, 0.5, 1.0) for bg in (30.0, 45.0, 60.0)}}

    # Part B
    def block(mask):
        return {"n": int(mask.sum()), "theta_goal_median": med(tg[mask]),
                "gp_nom_median": med(R["gp_nom"][mask]), "gp_safe_median": med(R["gp_safe"][mask]),
                "ang_nom_safe_median": med(R["ang_nom_safe"][mask])}
    rep["partB"] = {"success_near_obstacle": block((outc == "goal") & R["near"]),
                    "stuck_phase": block(R["stuck_phase"]),
                    "goal_far_from_obstacle": block((outc == "goal") & (~R["near"]))}

    # Part C.1 — avoidance: nominal outward vs CBF outward at near-obstacle steps
    nm = R["near"]; on_nom = R["out_nom"][nm]; on_filt = R["out_filt"][nm]
    denom = np.abs(on_nom) + np.abs(on_filt)
    frac_cbf = np.where(denom > 1e-9, np.abs(on_filt) / denom, np.nan)
    rep["partC1_avoidance"] = {
        "n_near_steps": int(nm.sum()),
        "outward_from_nominal_median": med(on_nom), "outward_from_filter_median": med(on_filt),
        "frac_avoidance_from_cbf_median": float(np.nanmedian(frac_cbf)),
        "frac_steps_nominal_pushes_toward_obstacle": float(np.mean(on_nom < 0))}

    # Part C.2 — obs-perturbation sensitivity (re-query policy on rebuilt full obs)
    rep["partC2_sensitivity"] = probe(policy, system, probe_samples, rng)

    (OUT / "stage2_policy_obstacle_response_summary.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    _figs(R)
    _print(rep)
    return 0


def probe(policy, system, samples, rng):
    d_obs, d_goal = [], []
    with torch.no_grad():
        for x0, scene in samples:
            x = torch.as_tensor(x0[None], dtype=torch.float32)
            obs = system.observation(x, scene)
            base = policy(obs)
            o1 = obs.clone()
            o1[0, 4:7] = obs[0, 4:7] + torch.as_tensor(unit(rng.normal(0, 1, 3)) * PROBE_DELTA, dtype=torch.float32)
            o2 = obs.clone()
            o2[0, 2:4] = obs[0, 2:4] + torch.as_tensor(unit(rng.normal(0, 1, 2)) * PROBE_DELTA, dtype=torch.float32)
            d_obs.append(float(torch.linalg.norm(policy(o1) - base)))
            d_goal.append(float(torch.linalg.norm(policy(o2) - base)))
    d_obs = np.asarray(d_obs); d_goal = np.asarray(d_goal)
    return {"n": len(samples), "delta": PROBE_DELTA,
            "median_dunom_obstacle_perturb": float(np.median(d_obs)),
            "median_dunom_goal_perturb": float(np.median(d_goal)),
            "ratio_obstacle_over_goal": float(np.median(d_obs) / max(np.median(d_goal), 1e-9))}


def _figs(R):
    tg = R["theta_goal"]; outc = R["outcome"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), dpi=130)
    ax[0].hist(tg[R["inpath"]], bins=40, density=True, alpha=0.6, label="in-path obstacle", color="#d62728")
    ax[0].hist(tg[~R["inpath"]], bins=40, density=True, alpha=0.6, label="no in-path obstacle", color="#7f7f7f")
    ax[0].set_title("theta_goal = angle(u_nom, ĝ)"); ax[0].set_xlabel("deg"); ax[0].legend(fontsize=8, frameon=False)
    sn = (outc == "goal") & R["near"]; sp = R["stuck_phase"]
    ax[1].hist(tg[sn], bins=40, density=True, alpha=0.6, label="success near-obs", color="#2ca02c")
    ax[1].hist(tg[sp], bins=40, density=True, alpha=0.6, label="stuck phase", color="#1f77b4")
    ax[1].set_title("goal-straightness: success vs stuck"); ax[1].set_xlabel("theta_goal deg"); ax[1].legend(fontsize=8, frameon=False)
    nm = R["near"]
    ax[2].hist(R["out_nom"][nm], bins=40, alpha=0.6, label="u_nom·outward", color="#2ca02c")
    ax[2].hist(R["out_filt"][nm], bins=40, alpha=0.6, label="(u_safe-u_nom)·outward", color="#d62728")
    ax[2].axvline(0, color="0.3", lw=1); ax[2].set_title("avoidance accel (away from obstacle)"); ax[2].legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "stage2_policy_obstacle_response.png"); plt.close(fig)


def _print(rep):
    print("recon u_nom max mismatch:", f"{rep['recon_max_mismatch_unom']:.3e}", "| counts:", rep["counts"])
    a = rep["partA"]
    print(f"Part A: theta_goal in-path={a['theta_goal_inpath_median']:.1f}deg (n={a['n_inpath']}) "
          f"not-in-path={a['theta_goal_not_inpath_median']:.1f}deg gap={a['gap_median_deg']:.1f}deg "
          f"defl-to-tangent(in-path)={a['deflection_to_tangent_median_deg_inpath']:.1f}deg")
    print("  sensitivity (inpath theta median / gap):",
          {k: (round(v["inpath_theta_median"], 1), round(v["gap_vs_rest"], 1)) for k, v in a["sensitivity"].items() if k in ("D0.3_B45.0", "D0.5_B45.0", "D1.0_B45.0", "D0.5_B30.0", "D0.5_B60.0")})
    for k, d in rep["partB"].items():
        print(f"Part B {k}: n={d['n']} theta_goal={d['theta_goal_median']} gp_nom={d['gp_nom_median']} "
              f"gp_safe={d['gp_safe_median']} ang(unom,usafe)={d['ang_nom_safe_median']}")
    c = rep["partC1_avoidance"]
    print(f"Part C.1: outward nominal med={c['outward_from_nominal_median']:.3f} filter med={c['outward_from_filter_median']:.3f} "
          f"frac-from-CBF={c['frac_avoidance_from_cbf_median']:.3f} frac-nominal-toward-obstacle={c['frac_steps_nominal_pushes_toward_obstacle']:.3f}")
    s = rep["partC2_sensitivity"]
    print(f"Part C.2: |Δu_nom| obstacle={s['median_dunom_obstacle_perturb']:.4f} goal={s['median_dunom_goal_perturb']:.4f} "
          f"ratio obs/goal={s['ratio_obstacle_over_goal']:.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
