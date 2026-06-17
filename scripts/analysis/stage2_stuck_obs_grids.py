"""v2.2.0 Stage 2 — Diagnostic 1: obs-distribution 4x4 grids for HOCBF-stuck and PNCBF-stuck.

Re-runs the N=2000 deployment (same pool seed 20260617; deterministic -> reproduces the
deploy_n2000_summary counts: PNCBF 86 stuck / 23 collision, HOCBF 133 stuck), caches per-scene
outcomes + the stuck/collision trajectories to deploy_n2000_traces.npz (consumed by the
collision/stuck decomposition script too), and builds two 4x4 trajectory grids so the researcher
can visually judge whether stuck cases are deep planning-hard pockets or shallow ones.

Read-only on the deployed policy/V_S/HardNet, secured checkpoint, committed pools; HOCBF is the
existing module. Re-running the deploy is the env work "needed to log obs/states at stuck states"
(no stored states exist for the N=2000 pool). Deterministic.

Run: /home/junhyeok/miniconda3/envs/pncbf/bin/python scripts/analysis/stage2_stuck_obs_grids.py
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (str(REPO_ROOT), str(REPO_ROOT / "scripts/analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)

import stage2_hocbf_deploy_n2000 as D  # noqa: E402  (pool builder + constants)
from src.common.filter_hocbf import HOCBFFilter  # noqa: E402
from src.common.outcomes import resolve_outcome, step_outcomes  # noqa: E402
from src.eval.rollout import rollout_eval  # noqa: E402
from src.envs.scene_batch import batch_scenes, initial_states_from_batch  # noqa: E402
from src.frameworks.jt_pncbf.train import load_framework_from_checkpoint, make_system  # noqa: E402

OUT = D.OUT
TRACES = OUT / "deploy_n2000_traces.npz"
STUCK_WINDOW = 60
STUCK_RADIUS = 0.10
NEAR_R = 1.0                 # radius for "n obstacles within R of stall"
WORLD_LIM = 4.0


def physical_onset(states):
    pos = states[:, :2]; T = pos.shape[0]
    for t in range(STUCK_WINDOW, T):
        if np.linalg.norm(pos[t - STUCK_WINDOW:t + 1] - pos[t - STUCK_WINDOW], axis=1).max() <= STUCK_RADIUS:
            return max(0, t - STUCK_WINDOW)
    return None


def _run_capture(system, scenes, config, filter_fn, policy_fn, device, dtype, chunk, want_u=False):
    """Rollout in chunks; return per-scene outcome/event + detached states (and u_nom/u_safe if want_u)."""
    max_steps = int(config["eval"]["max_steps"]); dt = float(config["env"]["dt"])
    outcomes, events = [], []
    states_list, unom_list, usafe_list = [], [], []
    for s in range(0, len(scenes), chunk):
        sub = scenes[s:s + chunk]
        bscene = batch_scenes(sub, device=device, dtype=dtype)
        x0 = initial_states_from_batch(bscene)
        res = rollout_eval(system, policy_fn, filter_fn, bscene, x0, max_steps, dt, config)
        masks = step_outcomes(res.states, bscene, system, config)
        resolved = resolve_outcome(masks)
        outcomes.extend(list(resolved.outcome))
        events.extend([int(e) for e in resolved.event_step.tolist()])
        states_list.append(res.states.detach().to("cpu"))        # [T+1, b, 4]
        if want_u:
            unom_list.append(res.u_nom.detach().to("cpu"))        # [T, b, 2]
            usafe_list.append(res.u_safe.detach().to("cpu"))
        del res, masks, resolved, bscene, x0
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"    {min(s + chunk, len(scenes))}/{len(scenes)}")
    states = torch.cat(states_list, dim=1).numpy()                # [T+1, N, 4]
    unom = torch.cat(unom_list, dim=1).numpy() if want_u else None
    usafe = torch.cat(usafe_list, dim=1).numpy() if want_u else None
    return np.array(outcomes), np.array(events, int), states, unom, usafe


def build_or_load_traces(fw, config, system, device, dtype, rebuild=False):
    if TRACES.exists() and not rebuild:
        return dict(np.load(TRACES, allow_pickle=False))
    pool, _, _ = D.build_n2000_pool(config)
    scenes = pool.scenes
    print("[traces] PNCBF deploy re-run (capturing states + u) ...")
    p_out, p_ev, p_states, p_unom, p_usafe = _run_capture(
        system, scenes, config, fw.filter, fw.policy, device, dtype, D.PNCBF_CHUNK, want_u=True)
    print("[traces] HOCBF deploy re-run ...")
    hocbf = HOCBFFilter(system, config, D.HOCBF_A1, D.HOCBF_A2, D.R_MARGIN, k_obs=D.K_DEPLOY)
    hf = lambda x, u, sc: hocbf(x, sc, u)
    h_out, h_ev, h_states, _, _ = _run_capture(
        system, scenes, config, hf, fw.policy, device, dtype, D.HOCBF_CHUNK, want_u=False)

    p_stuck = np.where(p_out == "stuck")[0]
    h_stuck = np.where(h_out == "stuck")[0]
    p_col = np.where(p_out == "collision")[0]
    print(f"[traces] PNCBF stuck={len(p_stuck)} collision={len(p_col)} | HOCBF stuck={len(h_stuck)}")
    cache = {
        "pncbf_outcomes": p_out, "hocbf_outcomes": h_out,
        "pncbf_event": p_ev, "hocbf_event": h_ev,
        "pncbf_stuck_idx": p_stuck, "pncbf_stuck_states": p_states[:, p_stuck, :].transpose(1, 0, 2),
        "hocbf_stuck_idx": h_stuck, "hocbf_stuck_states": h_states[:, h_stuck, :].transpose(1, 0, 2),
        "pncbf_col_idx": p_col, "pncbf_col_states": p_states[:, p_col, :].transpose(1, 0, 2),
        "pncbf_col_unom": p_unom[:, p_col, :].transpose(1, 0, 2),
        "pncbf_col_usafe": p_usafe[:, p_col, :].transpose(1, 0, 2),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(TRACES, **cache)
    print(f"[traces] wrote {TRACES}")
    return cache


def episode_geometry(states, scene):
    """stall point, final dist-to-goal, clearance at stall, n obstacles within NEAR_R, nearest-5 idx."""
    pos = states[:, :2]
    on = physical_onset(states)
    stall_i = on + STUCK_WINDOW if on is not None else pos.shape[0] - 1
    stall_i = min(stall_i, pos.shape[0] - 1)
    stall = pos[stall_i]
    goal = np.asarray(scene.goal, float)
    centers = np.asarray(scene.obstacle_centers, float); radii = np.asarray(scene.obstacle_radii, float)
    active = np.asarray(scene.obstacle_active, bool)
    surf = np.where(active, np.linalg.norm(centers - stall, axis=1) - radii, np.inf)
    order = np.argsort(surf)
    near5 = order[:min(D.K_DEPLOY, int(active.sum()))]
    n_within = int(np.sum((np.linalg.norm(centers - stall, axis=1) <= NEAR_R) & active))
    return {"stall_i": stall_i, "stall": stall, "final_dist_to_goal": float(np.linalg.norm(goal - pos[-1])),
            "clearance_at_stall": float(np.min(surf)), "n_obs_within_R": n_within, "near5": near5}


def grid(filter_name, idxs, states_arr, scenes, out_path):
    # trap-depth proxy = final distance-to-goal; select 16 spread across the sorted range
    geos = [episode_geometry(states_arr[k], scenes[idxs[k]]) for k in range(len(idxs))]
    proxy = np.array([g["final_dist_to_goal"] for g in geos])
    order = np.argsort(proxy)
    pick = np.unique(np.linspace(0, len(order) - 1, 16).round().astype(int))
    sel = order[pick][:16]

    fig, axes = plt.subplots(4, 4, figsize=(18, 18), dpi=140)
    for ax_i, k in enumerate(sel):
        ax = axes[ax_i // 4, ax_i % 4]
        gi = geos[k]; scene = scenes[idxs[k]]; st = states_arr[k]
        centers = np.asarray(scene.obstacle_centers, float); radii = np.asarray(scene.obstacle_radii, float)
        active = np.asarray(scene.obstacle_active, bool)
        for j, (c, r, a) in enumerate(zip(centers, radii, active)):
            if not a:
                continue
            ringed = j in gi["near5"]
            ax.add_patch(Circle(c, r, color=("#c44" if ringed else "0.6"), alpha=0.45))
            ax.add_patch(Circle(c, r + D.R_MARGIN, fill=False, ls="--", color="0.4", lw=0.7))
            if ringed:
                ax.add_patch(Circle(c, r, fill=False, ec="#c00", lw=1.8))   # highlight obs-horizon
        # trajectory up to stall, colour gradient start->stall
        traj = st[:gi["stall_i"] + 1, :2]
        if len(traj) > 1:
            segs = np.stack([traj[:-1], traj[1:]], axis=1)
            lc = LineCollection(segs, cmap="viridis", array=np.linspace(0, 1, len(segs)), lw=1.8)
            ax.add_collection(lc)
        ax.scatter([scene.start[0]], [scene.start[1]], c="lime", s=45, marker="o", ec="k", lw=0.5, zorder=6)
        ax.scatter([scene.goal[0]], [scene.goal[1]], c="red", s=110, marker="*", ec="k", lw=0.5, zorder=6)
        ax.scatter([gi["stall"][0]], [gi["stall"][1]], c="black", s=55, marker="X", zorder=7)
        ax.set_title(f"{filter_name} idx{idxs[k]} | dG={gi['final_dist_to_goal']:.2f} "
                     f"clr={gi['clearance_at_stall']:.2f}\n#obs<{NEAR_R:.0f}m={gi['n_obs_within_R']} "
                     f"(proxy dG={gi['final_dist_to_goal']:.2f})", fontsize=8)
        ax.set_xlim(-WORLD_LIM, WORLD_LIM); ax.set_ylim(-WORLD_LIM, WORLD_LIM)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"v2.2.0 {filter_name}-stuck obs-distribution grid (16 of {len(idxs)}; "
                 f"sorted by trap-depth proxy = final dist-to-goal)\n"
                 f"viridis traj start(dark)->stall(yellow), X=stall, *=goal, o=start; "
                 f"red-ringed = nearest-5 obs horizon; dashed = r+{D.R_MARGIN}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path); plt.close(fig)
    return {"path": str(out_path), "n_total": len(idxs), "selected_idx": [int(idxs[k]) for k in sel],
            "proxy_selected": [float(proxy[k]) for k in sel],
            "proxy_range": [float(proxy.min()), float(proxy.max())],
            "proxy_median": float(np.median(proxy))}


def main():
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    fw, config, _ = load_framework_from_checkpoint(D.CKPT)
    system = make_system(config)
    fw.value_net.to(device, dtype).eval(); fw.policy_net.to(device, dtype).eval()
    pool, _, _ = D.build_n2000_pool(config)
    scenes = pool.scenes

    cache = build_or_load_traces(fw, config, system, device, dtype)
    p_idx = cache["pncbf_stuck_idx"]; h_idx = cache["hocbf_stuck_idx"]
    print(f"[grids] PNCBF stuck {len(p_idx)} | HOCBF stuck {len(h_idx)} "
          f"| PNCBF collisions {len(cache['pncbf_col_idx'])}")
    pg = grid("PNCBF", p_idx, cache["pncbf_stuck_states"], scenes, OUT / "pncbf_stuck_grid_4x4.png")
    hg = grid("HOCBF", h_idx, cache["hocbf_stuck_states"], scenes, OUT / "hocbf_stuck_grid_4x4.png")
    print("PNCBF grid ->", pg["path"])
    print("HOCBF grid ->", hg["path"])
    print("PNCBF proxy range/median:", pg["proxy_range"], pg["proxy_median"])
    print("HOCBF proxy range/median:", hg["proxy_range"], hg["proxy_median"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
