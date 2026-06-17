"""v2.2.0 Stage 2 — analytic HOCBF vs PNCBF comparison harness (Parts 1-2 + 4x4 gain grid).

Implements the verification (HOCBF algebra + exact 2D QP vs cvxpy), the controlled collinear
scene constructor, the numeric gain sweep, and the 4x4 trajectory grid for visual gain selection.
Part 3 (controlled + real-scene + velocity comparison) is added after gain confirmation.

Runs env rollouts (rollout_eval) under a FIXED nominal (LQR or learned policy) with a swappable
filter (HOCBF / PNCBF-HardNet) — only the filter/safety-function differs. Read-only on the secured
checkpoint, committed pools, deployed policy/V_S/HardNet; the HOCBF is a new separate module.
Deterministic (LQR + HOCBF + rollout have no RNG; seeds reported for any constructed scenes).

Run: /home/junhyeok/miniconda3/envs/pncbf/bin/python scripts/analysis/stage2_hocbf_comparison.py
"""

from __future__ import annotations

import gc
import glob
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.filter_hocbf import HOCBFFilter, _qp2d  # noqa: E402
from src.common.value_net import make_h_fn  # noqa: E402
from src.common.filter_hardnet import HardNetFilter  # noqa: E402
from src.common.outcomes import resolve_outcome, step_outcomes  # noqa: E402
from src.envs.scene_init import Scene  # noqa: E402
from src.envs.scene_batch import batch_scenes, initial_states_from_batch  # noqa: E402
from src.eval.rollout import rollout_eval  # noqa: E402
from src.frameworks.jt_pncbf.train import load_framework_from_checkpoint, make_system  # noqa: E402

CKPT = REPO_ROOT / "data/secured_data/v2.0.1/seed42/checkpoints/best.pt"
OUT = REPO_ROOT / "data/diagnostics/v2.2.0_hocbf"
R_MARGIN = 0.05
# canonical collinear scene
C_START = np.array([-2.5, 0.0]); C_GOAL = np.array([2.5, 0.0])
C_OFFSET = 0.02          # tiny perpendicular symmetry-break (perfect collinearity is an unstable tie)
C_RADIUS = 0.5
GRID_A1 = [0.5, 2.0, 4.0, 8.0]
GRID_A2 = [0.5, 2.0, 4.0, 8.0]
SWEEP = [0.5, 1.0, 2.0, 4.0, 8.0]
HOCBF_A1, HOCBF_A2 = 2.0, 2.0          # confirmed gains for the Part-3 comparison
STUCK_DIR = REPO_ROOT / "data/diagnostics/v2.2.0_stage2_largeN/episodes"


def make_collinear(radius=C_RADIUS, half=2.5, offset=C_OFFSET, v0=(0.0, 0.0), start=C_START, goal=C_GOAL):
    s = np.asarray(start, float); g = np.asarray(goal, float)
    mid = 0.5 * (s + g)
    seg = g - s; perp = np.array([-seg[1], seg[0]]); perp = perp / (np.linalg.norm(perp) + 1e-12)
    c = mid + offset * perp
    return Scene(obstacle_centers=c.reshape(1, 2), obstacle_radii=np.array([radius]),
                 obstacle_active=np.array([True]), start=s, goal=g, system="double_integrator",
                 mode="eval", initial_velocity=np.asarray(v0, float))


def run_batch(system, scenes, config, filter_fn, policy_fn, device, dtype):
    bscene = batch_scenes(scenes, device=device, dtype=dtype)
    x0 = initial_states_from_batch(bscene)
    res = rollout_eval(system, policy_fn, filter_fn, bscene, x0,
                       int(config["eval"]["max_steps"]), float(config["env"]["dt"]), config)
    masks = step_outcomes(res.states, bscene, system, config)
    resolved = resolve_outcome(masks)
    return res, resolved, bscene


def lqr_policy(system):
    def f(x, scene):
        goal = torch.as_tensor(scene.goal, dtype=x.dtype, device=x.device)
        if goal.ndim == 1:
            goal = goal.unsqueeze(0).expand(x.shape[0], -1)
        return system.lqr_action(x, goal)
    return f


def hocbf_filter_fn(hocbf):
    return lambda x, u_nom, scene: hocbf(x, scene, u_nom)


def min_clearance(states, scene, system):
    pos = system.position(states)[:, 0, :].detach().cpu().numpy()       # [T+1,2] single episode
    c = np.asarray(scene.obstacle_centers[0]); r = float(scene.obstacle_radii[0])
    return float(np.min(np.linalg.norm(pos - c, axis=1) - r))


# ---- Part 1: verification -------------------------------------------------------------

def verify_algebra(rng):
    errs, row_errs = [], []
    for _ in range(2000):
        p = rng.normal(0, 1.5, 2); v = rng.normal(0, 1.0, 2); c = rng.normal(0, 1.5, 2)
        r = abs(rng.normal(0.5, 0.2)) + 0.1; a1 = rng.uniform(0.2, 8); a2 = rng.uniform(0.2, 8)
        u = rng.uniform(-2, 2, 2); rsafe = r + R_MARGIN
        def psi1(pp, vv):
            return 2 * np.dot(pp - c, vv) + a1 * (np.dot(pp - c, pp - c) - rsafe ** 2)
        eps = 1e-6
        fd = (psi1(p + eps * v, v + eps * u) - psi1(p, v)) / eps
        analytic = 2 * np.dot(v, v) + 2 * np.dot(p - c, u) + 2 * a1 * np.dot(p - c, v)
        errs.append(abs(fd - analytic))
        # row: A u <= b  <=>  psi1_dot + a2 psi1 >= 0
        A = -2 * (p - c); b = 2 * np.dot(v, v) + 2 * a1 * np.dot(p - c, v) + a2 * psi1(p, v)
        lhs = np.dot(A, u) - b                  # <= 0 iff feasible
        cbf = analytic + a2 * psi1(p, v)        # >= 0 iff feasible
        row_errs.append(abs((-lhs) - cbf))      # -lhs should equal cbf
    return {"psi1_dot_fd_vs_analytic_max": float(np.max(errs)),
            "row_vs_cbf_condition_max": float(np.max(row_errs))}


def verify_qp(rng):
    try:
        import cvxpy as cp
    except Exception:
        return {"cvxpy": "unavailable"}
    maxdiff = 0.0; n_feas = 0
    for _ in range(150):
        m = rng.integers(1, 4)
        A = rng.normal(0, 1, (1, m + 4, 2)); b = rng.normal(0.5, 1.0, (1, m + 4))
        A[0, m:] = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]); b[0, m:] = 2.0
        un = rng.uniform(-2, 2, (1, 2))
        u_mine, infeas = _qp2d(torch.tensor(un), torch.tensor(A), torch.tensor(b), 2.0)
        u = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(u - un[0])),
                          [A[0] @ u <= b[0]])
        prob.solve(solver=cp.OSQP, eps_abs=1e-10, eps_rel=1e-10)
        if prob.status == cp.OPTIMAL and not bool(infeas[0]):
            n_feas += 1
            maxdiff = max(maxdiff, float(np.linalg.norm(u.value - u_mine[0].numpy())))
    return {"cvxpy": "ok", "n_feasible_compared": n_feas, "max_abs_diff_qp_vs_cvxpy": maxdiff}


# ---- Part 2: gain sweep + 4x4 grid ---------------------------------------------------

def validation_scenes():
    scenes = []
    for radius in (0.3, 0.5, 0.7):
        for half in (2.0, 2.5):
            for v0 in ((0.0, 0.0), (0.5, 0.0)):
                scenes.append(make_collinear(radius=radius, half=half, v0=v0,
                                             start=np.array([-half, 0.0]), goal=np.array([half, 0.0])))
    return scenes


def gain_sweep(system, config, device, dtype):
    scenes = validation_scenes()
    table = {}
    for a1 in SWEEP:
        for a2 in SWEEP:
            hocbf = HOCBFFilter(system, config, a1, a2, R_MARGIN)
            _, resolved, _ = run_batch(system, scenes, config, hocbf_filter_fn(hocbf), lqr_policy(system), device, dtype)
            oc = resolved.outcome
            table[f"a1={a1}_a2={a2}"] = {"reach": oc.count("goal"), "collision": oc.count("collision"),
                                         "stuck": oc.count("stuck"), "timeout": oc.count("timeout"),
                                         "oob": oc.count("oob"), "n": len(oc)}
    return table, len(scenes)


def grid_4x4(system, config, device, dtype):
    scene = make_collinear()
    lqr_only_states = None
    # LQR-only reference (no filter)
    def passthrough(x, u_nom, scene):
        return u_nom, torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
    res0, _, _ = run_batch(system, [scene], config, passthrough, lqr_policy(system), device, dtype)
    lqr_ref = system.position(res0.states)[:, 0, :].detach().cpu().numpy()

    cells = []
    fig, axes = plt.subplots(4, 4, figsize=(16, 16), dpi=160)
    for i, a1 in enumerate(GRID_A1):
        for j, a2 in enumerate(GRID_A2):
            hocbf = HOCBFFilter(system, config, a1, a2, R_MARGIN)
            res, resolved, _ = run_batch(system, [scene], config, hocbf_filter_fn(hocbf), lqr_policy(system), device, dtype)
            oc = resolved.outcome[0]; ev = int(resolved.event_step[0].item())
            traj = system.position(res.states)[:, 0, :].detach().cpu().numpy()
            mc = min_clearance(res.states, scene, system)
            steps = ev if ev >= 0 else int(res.states.shape[0] - 1)
            cells.append({"alpha1": a1, "alpha2": a2, "outcome": oc, "min_clearance": mc,
                          "steps_to_goal": steps if oc == "goal" else None})
            ax = axes[i, j]
            color = {"goal": "green", "collision": "red", "stuck": "orange", "timeout": "orange", "oob": "red"}[oc]
            c = np.asarray(scene.obstacle_centers[0]); r = float(scene.obstacle_radii[0])
            ax.add_patch(Circle(c, r, color="0.6", alpha=0.7))
            ax.add_patch(Circle(c, r + R_MARGIN, fill=False, ls="--", color="0.4"))
            ax.plot(lqr_ref[:, 0], lqr_ref[:, 1], color="0.7", lw=1.0, ls=":", alpha=0.8)
            ax.plot(traj[:, 0], traj[:, 1], color="#1f77b4", lw=1.6)
            ax.scatter([scene.start[0]], [scene.start[1]], c="green", s=40, marker="o", zorder=5)
            ax.scatter([scene.goal[0]], [scene.goal[1]], c="red", s=80, marker="*", zorder=5)
            ax.set_title(f"a1={a1}, a2={a2} | {oc}\nminClr={mc:.3f}  steps={steps}", color=color, fontsize=10)
            for sp in ax.spines.values():
                sp.set_color(color); sp.set_linewidth(2.5)
            ax.set_aspect("equal"); ax.set_xlim(-3.0, 3.0); ax.set_ylim(-2.0, 2.0)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("v2.2.0 HOCBF gain grid (LQR nominal, canonical collinear scene): rows alpha1, cols alpha2\n"
                 "gray=obstacle, dashed=r_safe, dotted=LQR-only, blue=HOCBF; border green=reach/red=collision/orange=stuck",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "hocbf_gain_grid_4x4.png"
    fig.savefig(path); plt.close(fig)
    return path, cells, scene


# ---- Part 3: controlled comparison (nominal fixed, filter swapped) -------------------

OUTCOMES = ("goal", "collision", "stuck", "timeout", "oob")


def learned_policy(fw):
    return lambda x, scene: fw.policy(x, scene)


def counts_of(outc):
    return {o: int(list(outc).count(o)) for o in OUTCOMES}


def run_chunked(system, scenes, config, filter_fn, policy_fn, device, dtype, chunk, want_states=False):
    """Rollout over scenes in chunks; detach + free the autograd graph per chunk (PNCBF create_graph)."""
    max_steps = int(config["eval"]["max_steps"]); dt = float(config["env"]["dt"])
    outcomes, event_steps = [], []
    states_all = [] if want_states else None
    for s in range(0, len(scenes), chunk):
        sub = scenes[s:s + chunk]
        bscene = batch_scenes(sub, device=device, dtype=dtype)
        x0 = initial_states_from_batch(bscene)
        res = rollout_eval(system, policy_fn, filter_fn, bscene, x0, max_steps, dt, config)
        masks = step_outcomes(res.states, bscene, system, config)
        resolved = resolve_outcome(masks)
        outcomes.extend(list(resolved.outcome))
        event_steps.extend([int(e) for e in resolved.event_step.tolist()])
        if want_states:
            states_all.append(res.states.detach().to("cpu"))
        del res, masks, resolved, bscene, x0
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    states = torch.cat(states_all, dim=1) if want_states else None      # [T+1, N, 4]
    return outcomes, event_steps, states


def part3_collinear(system, config, fw, device, dtype):
    """3.1 — controlled collinear grid (radius x distance x speed) x {LQR, learned} x {HOCBF, PNCBF}."""
    radii = (0.3, 0.5, 0.7); halves = (2.0, 2.5, 3.0); speeds = (0.0, 0.5, 1.0)
    scenes, scene_meta = [], []
    for radius in radii:
        for half in halves:
            for sp in speeds:
                scenes.append(make_collinear(radius=radius, half=half, v0=(sp, 0.0),
                                             start=np.array([-half, 0.0]), goal=np.array([half, 0.0])))
                scene_meta.append({"radius": radius, "half_dist": half, "approach_speed": sp})
    hocbf = HOCBFFilter(system, config, HOCBF_A1, HOCBF_A2, R_MARGIN)
    nominals = {"lqr": lqr_policy(system), "learned": learned_policy(fw)}
    filters = {"hocbf": hocbf_filter_fn(hocbf), "pncbf": fw.filter}
    per_scene, summary = {}, {}
    for nname, pol in nominals.items():
        for fname, filt in filters.items():
            outc, _, _ = run_chunked(system, scenes, config, filt, pol, device, dtype, chunk=len(scenes))
            per_scene[f"{nname}+{fname}"] = outc
            summary[f"{nname}+{fname}"] = counts_of(outc)
            print(f"  collinear {nname}+{fname}: {summary[f'{nname}+{fname}']}")
    contrast = {}
    for nname in nominals:
        p = per_scene[f"{nname}+pncbf"]; h = per_scene[f"{nname}+hocbf"]
        stuck_idx = [i for i, o in enumerate(p) if o == "stuck"]
        coll_idx = [i for i, o in enumerate(p) if o == "collision"]
        contrast[nname] = {
            "pncbf_stuck_n": len(stuck_idx),
            "hocbf_on_pncbf_stuck": counts_of([h[i] for i in stuck_idx]),
            "pncbf_collision_n": len(coll_idx),
            "hocbf_on_pncbf_collision": counts_of([h[i] for i in coll_idx]),
        }
        print(f"  contrast[{nname}]: PNCBF stuck={len(stuck_idx)} -> HOCBF {contrast[nname]['hocbf_on_pncbf_stuck']}")
    return {"grid": {"radii": list(radii), "half_dists": list(halves), "approach_speeds": list(speeds),
                     "n_scenes": len(scenes)},
            "outcome_counts": summary, "per_scene": {k: list(v) for k, v in per_scene.items()},
            "scene_meta": scene_meta, "contrast_pncbf_stuck_vs_hocbf": contrast}


def reconstruct_stuck_scenes(limit=None):
    files = sorted(glob.glob(str(STUCK_DIR / "ep_*_stuck.npz")))
    if limit is not None:
        files = files[:limit]
    scenes, stored_states, idxs = [], [], []
    for f in files:
        z = np.load(f, allow_pickle=False)
        scenes.append(Scene(
            obstacle_centers=np.asarray(z["obstacle_centers"], float),
            obstacle_radii=np.asarray(z["obstacle_radii"], float),
            obstacle_active=np.asarray(z["obstacle_active"], bool),
            start=np.asarray(z["start"], float), goal=np.asarray(z["goal"], float),
            system="double_integrator", mode="eval",
            initial_velocity=np.asarray(z["initial_velocity"], float)))
        stored_states.append(np.asarray(z["states"], np.float32))   # [201,4] deployed PNCBF trajectory
        idxs.append(int(Path(f).stem.split("_")[1]))
    return scenes, stored_states, idxs, files


def part3_real(system, config, fw, device, dtype):
    """3.2 — real Stage-2 PNCBF-stuck scenes: reconstruct, re-run PNCBF (verify), run HOCBF, count resolutions."""
    scenes, stored_states, idxs, files = reconstruct_stuck_scenes()
    n = len(scenes)
    print(f"  reconstructed {n} PNCBF-stuck scenes from NPZ")
    learned = learned_policy(fw)
    # PNCBF re-run on reconstructed scenes (verifies reconstruction reproduces the deployed stuck result)
    pncbf_outc, _, pncbf_states = run_chunked(system, scenes, config, fw.filter, learned, device, dtype,
                                              chunk=200, want_states=True)
    # reproduction fidelity vs stored deployed trajectory
    stored = np.stack(stored_states, axis=0)                        # [N,201,4]
    rerun = pncbf_states.permute(1, 0, 2).numpy()                   # [N,201,4]
    max_state_diff = float(np.max(np.abs(rerun - stored)))
    n_repro_stuck = int(list(pncbf_outc).count("stuck"))
    # HOCBF on the SAME reconstructed scenes, identical learned nominal; full obstacle awareness (K=12)
    hocbf = HOCBFFilter(system, config, HOCBF_A1, HOCBF_A2, R_MARGIN, k_obs=12)
    hocbf_outc, _, _ = run_chunked(system, scenes, config, hocbf_filter_fn(hocbf), learned, device, dtype, chunk=600)
    # resolution among the episodes PNCBF re-run confirms as stuck
    repro_stuck_idx = [i for i, o in enumerate(pncbf_outc) if o == "stuck"]
    hocbf_on_stuck = counts_of([hocbf_outc[i] for i in repro_stuck_idx])
    resolved_idx = [i for i in repro_stuck_idx if hocbf_outc[i] == "goal"]
    print(f"  PNCBF re-run reproduces stuck: {n_repro_stuck}/{n} (max|state diff vs stored|={max_state_diff:.2e})")
    print(f"  HOCBF on PNCBF-stuck scenes: {hocbf_on_stuck}")
    return ({"n_scenes": n, "pncbf_rerun_counts": counts_of(pncbf_outc),
             "pncbf_reproduces_stuck": n_repro_stuck, "max_state_diff_vs_stored": max_state_diff,
             "hocbf_counts_all": counts_of(hocbf_outc),
             "hocbf_on_pncbf_stuck": hocbf_on_stuck,
             "n_resolved_to_goal": len(resolved_idx),
             "resolution_rate_to_goal": len(resolved_idx) / max(1, len(repro_stuck_idx))},
            {"scenes": scenes, "stored_states": stored_states, "idxs": idxs,
             "pncbf_outc": pncbf_outc, "hocbf_outc": hocbf_outc, "resolved_idx": resolved_idx})


def _vel_components(states, goal):
    """states [T+1,4] numpy; returns dict of per-step speed, goal-progress vel, tangential speed, dist."""
    pos = states[:, :2]; vel = states[:, 2:4]
    gvec = goal[None, :] - pos
    gdist = np.linalg.norm(gvec, axis=1)
    gdir = gvec / np.clip(gdist[:, None], 1e-9, None)
    vprog = np.sum(vel * gdir, axis=1)
    vtang = np.linalg.norm(vel - vprog[:, None] * gdir, axis=1)
    return {"speed": np.linalg.norm(vel, axis=1), "v_progress": vprog, "v_tangential": vtang,
            "dist_to_goal": gdist, "pos": pos}


def part3_velocity(system, config, fw, device, dtype, real_bundle):
    """3.3 — velocity-profile comparison HOCBF vs PNCBF on a representative collinear and real scene."""
    dt = float(config["env"]["dt"])
    hocbf1 = HOCBFFilter(system, config, HOCBF_A1, HOCBF_A2, R_MARGIN)
    hocbf12 = HOCBFFilter(system, config, HOCBF_A1, HOCBF_A2, R_MARGIN, k_obs=12)

    # representative collinear scene (LQR nominal: clean goal-straight): radius 0.5, half 2.5, approach 0.5
    col = make_collinear(radius=0.5, half=2.5, v0=(0.5, 0.0), start=np.array([-2.5, 0.0]), goal=np.array([2.5, 0.0]))
    col_runs = {}
    for fname, filt in (("hocbf", hocbf_filter_fn(hocbf1)), ("pncbf", fw.filter)):
        res, resolved, _ = run_batch(system, [col], config, filt, lqr_policy(system), device, dtype)
        col_runs[fname] = {"states": res.states[:, 0, :].detach().cpu().numpy(), "outcome": resolved.outcome[0]}

    # representative real scene: prefer one HOCBF resolves (PNCBF stuck -> HOCBF goal), else first stuck
    resolved_idx = real_bundle["resolved_idx"]
    pick = resolved_idx[0] if resolved_idx else 0
    rscene = real_bundle["scenes"][pick]; rstored = real_bundle["stored_states"][pick]
    real_runs = {"pncbf": {"states": np.asarray(rstored, float),
                           "outcome": real_bundle["pncbf_outc"][pick]}}
    res, resolved, _ = run_batch(system, [rscene], config, hocbf_filter_fn(hocbf12), learned_policy(fw), device, dtype)
    real_runs["hocbf"] = {"states": res.states[:, 0, :].detach().cpu().numpy(), "outcome": resolved.outcome[0]}

    # figure: cols [collinear, real]; rows [trajectory, speed, goal-progress velocity]
    fig, axes = plt.subplots(3, 2, figsize=(13, 13), dpi=150)
    panels = [("collinear (LQR nominal)", col, col_runs), (f"real stuck scene idx={real_bundle['idxs'][pick]} (learned nominal)", rscene, real_runs)]
    col_colors = {"hocbf": "#1f77b4", "pncbf": "#d62728"}
    for cidx, (title, scene, runs) in enumerate(panels):
        ax = axes[0, cidx]
        ca = np.asarray(scene.obstacle_centers); ra = np.asarray(scene.obstacle_radii)
        act = np.asarray(scene.obstacle_active)
        for c, r, a in zip(ca, ra, act):
            if a:
                ax.add_patch(Circle(c, r, color="0.6", alpha=0.55))
                ax.add_patch(Circle(c, r + R_MARGIN, fill=False, ls="--", color="0.45", lw=0.8))
        for fname, run in runs.items():
            comp = _vel_components(run["states"], np.asarray(scene.goal, float))
            ax.plot(comp["pos"][:, 0], comp["pos"][:, 1], color=col_colors[fname], lw=1.6,
                    label=f"{fname} ({run['outcome']})")
        ax.scatter([scene.start[0]], [scene.start[1]], c="green", s=45, marker="o", zorder=5)
        ax.scatter([scene.goal[0]], [scene.goal[1]], c="red", s=90, marker="*", zorder=5)
        ax.set_title(f"trajectory — {title}", fontsize=10); ax.set_aspect("equal"); ax.legend(fontsize=8, frameon=False)
        t = np.arange(runs["hocbf"]["states"].shape[0]) * dt
        for fname, run in runs.items():
            comp = _vel_components(run["states"], np.asarray(scene.goal, float))
            axes[1, cidx].plot(t, comp["speed"], color=col_colors[fname], lw=1.4, label=f"{fname} |v|")
            axes[2, cidx].plot(t, comp["v_progress"], color=col_colors[fname], lw=1.4, label=f"{fname} v·ĝ")
            axes[2, cidx].plot(t, comp["v_tangential"], color=col_colors[fname], lw=1.0, ls=":", alpha=0.8)
        axes[1, cidx].axhline(0, color="0.7", lw=0.8); axes[1, cidx].set_title("speed |v|", fontsize=10)
        axes[1, cidx].set_xlabel("time [s]"); axes[1, cidx].legend(fontsize=8, frameon=False)
        axes[2, cidx].axhline(0, color="0.7", lw=0.8)
        axes[2, cidx].set_title("goal-progress v·ĝ (solid), tangential |v_⊥| (dotted)", fontsize=10)
        axes[2, cidx].set_xlabel("time [s]"); axes[2, cidx].legend(fontsize=8, frameon=False)
    fig.suptitle("v2.2.0 Part 3.3 — velocity profile: HOCBF (blue) vs PNCBF (red)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "part3_velocity_profiles.png"
    fig.savefig(path); plt.close(fig)

    def stats(run, scene):
        comp = _vel_components(run["states"], np.asarray(scene.goal, float))
        T = run["states"].shape[0]
        active = slice(0, T)
        return {"outcome": run["outcome"], "mean_speed": float(np.mean(comp["speed"][active])),
                "mean_v_progress": float(np.mean(comp["v_progress"][active])),
                "final_dist_to_goal": float(comp["dist_to_goal"][-1]),
                "min_dist_to_goal": float(np.min(comp["dist_to_goal"]))}
    return {"path": str(path), "picked_real_idx": int(real_bundle["idxs"][pick]),
            "real_is_resolved_example": bool(resolved_idx),
            "collinear": {f: stats(col_runs[f], col) for f in col_runs},
            "real": {f: stats(real_runs[f], rscene) for f in real_runs}}


def run_part3(system, config, fw, device, dtype):
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {"hocbf_gains": {"alpha1": HOCBF_A1, "alpha2": HOCBF_A2}, "r_margin": R_MARGIN,
           "lookahead": "disabled (config filter.lookahead is None) -> PNCBF filter is nominal-independent"}
    print("[3.1] controlled collinear grid ...")
    rep["part3_1_collinear"] = part3_collinear(system, config, fw, device, dtype)
    print("[3.2] real Stage-2 stuck scenes ...")
    real_summary, real_bundle = part3_real(system, config, fw, device, dtype)
    rep["part3_2_real_stuck"] = real_summary
    print("[3.3] velocity profiles ...")
    rep["part3_3_velocity"] = part3_velocity(system, config, fw, device, dtype, real_bundle)
    (OUT / "part3_summary.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("part3 ->", OUT / "part3_summary.json")
    return rep


def run_parts12(system, config, device, dtype, rng):
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {"r_margin": R_MARGIN, "u_max": system.u_bounds.tolist(),
           "canonical_scene": {"start": C_START.tolist(), "goal": C_GOAL.tolist(),
                               "obstacle_center_offset_perp": C_OFFSET, "radius": C_RADIUS,
                               "r_safe": C_RADIUS + R_MARGIN}}
    rep["verify_algebra"] = verify_algebra(rng)
    rep["verify_qp"] = verify_qp(rng)
    print("verify algebra:", rep["verify_algebra"])
    print("verify qp:", rep["verify_qp"])
    sweep, n_val = gain_sweep(system, config, device, dtype)
    rep["gain_sweep"] = {"n_validation_scenes": n_val, "table": sweep}
    best = max(sweep.items(), key=lambda kv: (kv[1]["reach"], -kv[1]["collision"], -kv[1]["stuck"]))
    rep["sweep_best_by_numeric"] = {"gains": best[0], **best[1]}
    print("sweep best (numeric):", best[0], best[1])
    grid_path, cells, _ = grid_4x4(system, config, device, dtype)
    rep["grid_4x4"] = {"path": str(grid_path), "alpha1_rows": GRID_A1, "alpha2_cols": GRID_A2, "cells": cells}
    (OUT / "hocbf_gain_summary.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("4x4 grid ->", grid_path)
    return rep


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "part3"
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    fw, config, _ = load_framework_from_checkpoint(CKPT)
    system = make_system(config)
    fw.value_net.to(device, dtype).eval(); fw.policy_net.to(device, dtype).eval()
    rng = np.random.default_rng(20260616)

    if mode in ("all", "parts12"):
        run_parts12(system, config, device, dtype, rng)
    if mode in ("all", "part3"):
        run_part3(system, config, fw, device, dtype)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
