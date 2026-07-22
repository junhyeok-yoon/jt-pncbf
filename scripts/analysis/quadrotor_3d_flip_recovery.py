"""v2.7.4 M6 amendment — flip-recovery time-series figures on the extreme stratum (measurement only).

Stratum (registered before the data exists): canonical d2r full-pool episodes with initial tilt >= 150 deg
AND `blocked` true (an active cylinder disk meets the xy-projection of the start->goal segment — the v2.7.3
M1 definition, reused verbatim from quadrotor_3d_blocked_stratify). Rank survivors by initial |omega| desc;
take the top 4 that ended in `reach` and the top 4 that ended in `collision` (canonical outcomes from the M6
eval_episodes.csv). Fewer than 4 -> take what exists and say so; empty collision group -> say so, never
back-fill. Re-roll ONLY the selected episodes from the frozen pool with full state recording; assert each
re-rolled outcome matches the canonical outcome (HALT on mismatch — determinism is load-bearing). Five
stacked panels per episode + one selection JSON under <run-dir>/figures/flip_recovery/. No verdict language.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.common.outcomes import step_outcomes, resolve_outcome
from src.common.rk4 import rk4_step
from src.eval.build_pools import load_pool, EVAL_POOLS_DIR, DEFAULT_OUTPUT_DIR
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.envs.quadrotor_3d import _quat_to_R
from src.eval.run_full import _load_framework

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)                 # M6 best.pt (JT)
ap.add_argument("--run-dir", required=True)              # M6 run dir (figures/flip_recovery/ written here)
ap.add_argument("--pool", default="eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl")
ap.add_argument("--episodes-csv", default=None)          # canonical outcomes; default <run-dir>/eval_episodes.csv
a = ap.parse_args()

run_dir = Path(a.run_dir)
out_dir = run_dir / "figures" / "flip_recovery"; out_dir.mkdir(parents=True, exist_ok=True)
pool_path = (EVAL_POOLS_DIR / a.pool) if (EVAL_POOLS_DIR / a.pool).exists() else (DEFAULT_OUTPUT_DIR / a.pool)
pool = load_pool(pool_path)
scenes = pool.scenes
fw, cfg, _ = _load_framework(Path(a.ckpt))
system = fw.system
dt = float(cfg["env"]["dt"]); max_steps = int(cfg["eval"]["max_steps"])
world_lim = float(cfg["env"]["world_lim"])


def _seg_point_dist(p0, p1, c):
    ab = p1 - p0
    t = np.clip(np.dot(c - p0, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(p0 + t * ab - c))


def _blocked(s) -> bool:
    return any(_seg_point_dist(np.asarray(s.start, float)[:2], np.asarray(s.goal, float)[:2],
                               np.asarray(s.obstacle_centers, float)[j]) < np.asarray(s.obstacle_radii, float)[j]
               for j in np.nonzero(np.asarray(s.obstacle_active, bool))[0])


def _tilt_deg(quat):
    R = _quat_to_R(torch.tensor(np.asarray(quat)[None], dtype=torch.float64))
    return float(np.degrees(np.arccos(np.clip(float(R[0, 2, 2]), -1.0, 1.0))))


# ---- per-episode ICs + blocked ----
init_tilt = np.array([_tilt_deg(s.initial_attitude_quat) for s in scenes])
init_omega = np.array([float(np.linalg.norm(np.asarray(s.initial_omega_vec))) for s in scenes])
init_v = np.array([float(np.linalg.norm(np.asarray(s.initial_velocity))) for s in scenes])
blocked = np.array([_blocked(s) for s in scenes], dtype=bool)

# ---- canonical outcomes ----
ep_csv = Path(a.episodes_csv) if a.episodes_csv else (run_dir / "eval_episodes.csv")
outcome = {}
with ep_csv.open() as f:
    for r in csv.DictReader(f):
        if r["mode"] == "final":
            outcome[int(r["episode_idx"])] = r["outcome"]

# ---- stratum + mechanical selection ----
stratum = [i for i in range(len(scenes)) if init_tilt[i] >= 150.0 and blocked[i] and i in outcome]
stratum.sort(key=lambda i: -init_omega[i])                      # rank by |omega| descending
reach_ids = [i for i in stratum if outcome[i] == "goal"][:4]
coll_ids = [i for i in stratum if outcome[i] == "collision"][:4]
selected = reach_ids + coll_ids
notes = []
if len(reach_ids) < 4: notes.append(f"only {len(reach_ids)} reach episodes in stratum (<4)")
if len(coll_ids) < 4: notes.append(f"only {len(coll_ids)} collision episodes in stratum (<4)")
if not coll_ids: notes.append("COLLISION GROUP EMPTY — no collision episode in the tilt>=150 & blocked stratum")


# ---- re-roll the selected episodes with full state recording ----
def _euler_and_tilt(quat_bt):
    R = _quat_to_R(quat_bt)                                     # [T,3,3]
    roll = torch.atan2(R[:, 2, 1], R[:, 2, 2])
    pitch = torch.asin(torch.clamp(-R[:, 2, 0], -1.0, 1.0))
    yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])
    tilt = torch.arccos(torch.clamp(R[:, 2, 2], -1.0, 1.0))
    return (np.degrees(roll.numpy()), np.degrees(pitch.numpy()),
            np.degrees(yaw.numpy()), np.degrees(tilt.numpy()))


# Re-roll the SELECTED episodes through the SAME canonical path run_full uses (evaluate.rollout_eval), so the
# states/intervention/empty are bit-identical to the canonical eval and the outcome assertion is exact.
from src.eval.rollout import rollout_eval
from src.eval.evaluate import _filter_adapter, _tensor_options, make_batched_scene

sel_records = {}
if selected:
    # Roll the FULL pool in ONE batch, exactly as the canonical eval (a borderline episode's outcome is
    # batch-composition-sensitive via batched-matmul reduction order, so only the same 2000-batch reproduces
    # it bit-identically). The full RolloutResult (~201x2000x13 f32 ~= 21 MB) is held transiently in RAM and
    # NOT persisted; only the selected episodes' states are kept.
    dtype, device = _tensor_options(system, fw)
    bscene_full = make_batched_scene(scenes, device=device, dtype=dtype)
    x0 = initial_states_from_batch(bscene_full)
    result = rollout_eval(system, fw.policy, _filter_adapter(fw), bscene_full, x0,
                          max_steps=max_steps, dt=dt, config=cfg)
    for ep in selected:
        Sj = result.states[:, ep, :].detach().to(torch.float64)          # [T+1, 13]
        masks = step_outcomes(result.states[:, ep:ep + 1, :], scenes[ep], system, cfg)
        rolled_outcome = resolve_outcome(masks).outcome[0]
        if rolled_outcome != outcome[ep]:
            raise SystemExit(f"HALT: re-roll outcome '{rolled_outcome}' != canonical '{outcome[ep]}' for episode "
                             f"{ep} — the eval is not deterministic; that is a larger problem than these figures.")
        interv = result.intervention_mask[:, ep].detach().cpu().numpy()
        empty = (result.empty[:, ep].detach().cpu().numpy() if result.empty is not None
                 else np.zeros(interv.shape, dtype=bool))
        sel_records[ep] = dict(S=Sj.numpy(), interv=interv, empty=empty, scene=scenes[ep], outcome=outcome[ep])


# ---- figures ----
def _dist_nearest_surface(pos_xy, scene):
    cen = np.asarray(scene.obstacle_centers, float)[:, :2]; rad = np.asarray(scene.obstacle_radii, float)
    act = np.asarray(scene.obstacle_active, bool)
    if not act.any():
        return np.full(pos_xy.shape[0], np.inf)
    d = np.linalg.norm(pos_xy[:, None, :] - cen[None, act, :], axis=2) - rad[None, act]
    return d.min(axis=1)


def _shade(ax, mask, t, color, alpha=0.15):
    m = np.asarray(mask); i = 0
    while i < len(m):
        if m[i]:
            j = i
            while j < len(m) and m[j]:
                j += 1
            ax.axvspan(t[i], t[min(j, len(t) - 1)], color=color, alpha=alpha, lw=0)
            i = j
        else:
            i += 1


tt_upright = {}
paths = []
for ep in selected:
    rec = sel_records[ep]; S = rec["S"]; scene = rec["scene"]
    Tn = S.shape[0]; t = np.arange(Tn) * dt
    p = S[:, 0:3]; quat = torch.tensor(S[:, 3:7])
    roll, pitch, yaw, tilt = _euler_and_tilt(quat)
    yaw_unwrap = np.degrees(np.unwrap(np.radians(yaw)))
    surf = _dist_nearest_surface(p[:, :2], scene)
    below30 = np.where(tilt < 30.0)[0]
    ttu = float(t[below30[0]]) if below30.size else None
    tt_upright[ep] = ttu
    interv_step = np.append(rec["interv"], rec["interv"][-1]); empty_step = np.append(rec["empty"], rec["empty"][-1])

    fig, axs = plt.subplots(5, 1, figsize=(9, 11), sharex=True,
                            gridspec_kw={"height_ratios": [1, 1, 1.2, 1, 0.6]})
    axs[0].plot(t, p[:, 0], label="x"); axs[0].plot(t, p[:, 1], label="y"); axs[0].set_ylabel("x,y (m)"); axs[0].legend(loc="upper right", fontsize=8)
    axs[1].plot(t, p[:, 2], color="tab:green", label="z"); axs[1].axhline(-world_lim, color="k", ls="--", lw=0.8, label="region floor")
    axs[1].set_ylabel("z (m)"); axs[1].legend(loc="upper right", fontsize=8)
    axs[2].plot(t, roll, label="roll", color="tab:blue"); axs[2].plot(t, pitch, label="pitch", color="tab:orange")
    axs[2].plot(t, tilt, label="tilt", color="k", lw=1.8); axs[2].axhline(30.0, color="grey", ls=":", lw=0.8)
    _shade(axs[2], np.abs(pitch) > 80.0, t, "red", 0.10)       # Euler singular near |pitch|>80
    axs[2].set_ylabel("angle (deg)"); axs[2].legend(loc="upper right", fontsize=8)
    axs[3].plot(t, yaw_unwrap, color="tab:purple", label="yaw (unwrapped)"); axs[3].set_ylabel("yaw (deg)"); axs[3].legend(loc="upper right", fontsize=8)
    axs[4].plot(t, surf, color="tab:brown"); axs[4].axhline(0.0, color="r", lw=0.9); axs[4].set_ylabel("dist to\nsurface (m)"); axs[4].set_xlabel("time (s)")
    for ax in axs:
        _shade(ax, interv_step, t, "tab:blue", 0.08)           # filter intervened
        if ttu is not None:
            ax.axvline(ttu, color="green", lw=1.4, ls="-")     # time-to-upright
        for k in np.where(empty_step)[0]:                      # empty-branch steps (fallback would act) — distinct marks
            ax.axvline(t[k], color="magenta", lw=0.6, alpha=0.5)
    fig.suptitle(f"ep {ep} | tilt0 {init_tilt[ep]:.0f}deg | |omega0| {init_omega[ep]:.2f} | |v0| {init_v[ep]:.2f} "
                 f"| outcome {rec['outcome']} | t_upright {('%.2fs' % ttu) if ttu else 'never'}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = out_dir / f"ep{ep:04d}_{rec['outcome']}.png"; fig.savefig(out, dpi=110); plt.close(fig)
    paths.append(str(out))

# ---- selection JSON ----
sel_json = {
    "stratum_definition": "initial tilt >= 150 deg AND blocked (v2.7.3 M1 seg-intersect definition)",
    "stratum_size_before_selection": len(stratum),
    "ranked_by": "initial |omega| descending",
    "n_reach_selected": len(reach_ids), "n_collision_selected": len(coll_ids),
    "notes": notes,
    "selected": [
        dict(episode_idx=ep, initial_tilt_deg=round(float(init_tilt[ep]), 2),
             initial_omega=round(float(init_omega[ep]), 3), initial_v=round(float(init_v[ep]), 3),
             outcome=outcome[ep], time_to_upright_s=(round(tt_upright[ep], 3) if tt_upright[ep] is not None else None))
        for ep in selected],
    "non_recovering": [ep for ep in selected if tt_upright[ep] is None],
    "figure_paths": paths,
}
(out_dir / "flip_recovery_selection.json").write_text(json.dumps(sel_json, indent=2) + "\n")
print(json.dumps(sel_json, indent=2))
print("SELECTED_EPISODE_IDS " + ",".join(str(ep) for ep in selected))
