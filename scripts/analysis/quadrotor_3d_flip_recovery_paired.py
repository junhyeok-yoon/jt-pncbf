"""v2.7.4 M6 amendment (paired) — flip-recovery time-series on the extreme stratum, PRE-JT vs JT on the
SAME episodes (measurement only).

Supersedes the top-4-reach / top-4-collision single-arm rule: collisions are sparse in this stratum (3 of 51
pre-JT, fewer under JT), so a JT-only reel would be all-reach. Instead both policies are rolled on the SAME
selected episodes and overlaid, which answers the recovery claim directly (does the aircraft that the pre-JT
filter loses right itself under JT?).

Stratum (registered before the data): canonical d2r full-pool episodes with initial tilt >= 150 deg AND
`blocked` (v2.7.3 M1 segment-intersect definition, verbatim). Selection, in priority order, never hand-picked
and never back-filled from an easier stratum:
  1. Outcome flips pre-JT collision -> JT goal, every one up to 6, ranked by initial |omega| desc.
  2. JT collisions in the stratum, all of them (if zero, say zero).
  3. Top |omega| episodes to fill up to 8 total.
Each figure states which category it came from. Both arms are re-rolled on the FULL pool through the canonical
rollout_eval path (a borderline outcome is batch-composition-sensitive, so only the same 2000-batch reproduces
it) and each arm's re-rolled outcome is asserted against ITS canonical outcome (HALT on mismatch). Five stacked
panels per episode with pre-JT (dashed, muted) and JT (solid) overlaid; each arm's time-to-upright marked
separately; filter-intervention shading and empty-branch marks drawn for the JT arm. No verdict language.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

from src.common.outcomes import step_outcomes, resolve_outcome
from src.eval.build_pools import load_pool, EVAL_POOLS_DIR, DEFAULT_OUTPUT_DIR
from src.envs.scene_batch import initial_states_from_batch
from src.envs.quadrotor_3d import _quat_to_R
from src.eval.run_full import _load_framework
from src.eval.rollout import rollout_eval
from src.eval.evaluate import _filter_adapter, _tensor_options, make_batched_scene

ap = argparse.ArgumentParser()
ap.add_argument("--jt-ckpt", required=True)              # M6 best.pt (JT)
ap.add_argument("--pre-ckpt", required=True)             # M4/M3 value best.pt (pre-JT: nominal + HardNet)
ap.add_argument("--run-dir", required=True)              # M6 run dir (figures/flip_recovery/ written here)
ap.add_argument("--jt-episodes-csv", default=None)       # default <run-dir>/eval_episodes.csv
ap.add_argument("--pre-episodes-csv", required=True)     # M4 run dir eval_episodes.csv
ap.add_argument("--pool", default="eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl")
a = ap.parse_args()

run_dir = Path(a.run_dir)
out_dir = run_dir / "figures" / "flip_recovery"; out_dir.mkdir(parents=True, exist_ok=True)
pool_path = (EVAL_POOLS_DIR / a.pool) if (EVAL_POOLS_DIR / a.pool).exists() else (DEFAULT_OUTPUT_DIR / a.pool)
scenes = load_pool(pool_path).scenes


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


def _read_outcomes(path):
    o = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["mode"] == "final":
                o[int(r["episode_idx"])] = r["outcome"]
    return o


# ---- per-episode ICs + blocked ----
init_tilt = np.array([_tilt_deg(s.initial_attitude_quat) for s in scenes])
init_omega = np.array([float(np.linalg.norm(np.asarray(s.initial_omega_vec))) for s in scenes])
init_v = np.array([float(np.linalg.norm(np.asarray(s.initial_velocity))) for s in scenes])
blocked = np.array([_blocked(s) for s in scenes], dtype=bool)

jt_csv = Path(a.jt_episodes_csv) if a.jt_episodes_csv else (run_dir / "eval_episodes.csv")
outc_jt = _read_outcomes(jt_csv)
outc_pre = _read_outcomes(a.pre_episodes_csv)

# ---- stratum + paired mechanical selection ----
stratum = [i for i in range(len(scenes)) if init_tilt[i] >= 150.0 and blocked[i] and i in outc_jt and i in outc_pre]
stratum.sort(key=lambda i: -init_omega[i])                       # rank by |omega| descending
flips = [i for i in stratum if outc_pre[i] == "collision" and outc_jt[i] == "goal"]
jt_coll = [i for i in stratum if outc_jt[i] == "collision"]

selected, category = [], {}
for i in flips[:6]:
    selected.append(i); category[i] = "flip_preJT_collision_to_JT_goal"
for i in jt_coll:
    if i not in selected:
        selected.append(i); category[i] = "JT_collision"
for i in stratum:                                                # top |omega| fill to 8
    if len(selected) >= 8:
        break
    if i not in selected:
        selected.append(i); category[i] = "top_omega_fill"

notes = []
notes.append(f"flip group (preJT collision -> JT goal): {len(flips)} in stratum")
notes.append(f"JT collision group: {len(jt_coll)} in stratum" + (" — ZERO" if not jt_coll else ""))
notes.append(f"top-|omega| fill added: {sum(1 for i in selected if category[i]=='top_omega_fill')}")


# ---- roll BOTH arms on the full pool through the canonical path ----
def roll_full(ckpt_path):
    fw, cfg, _ = _load_framework(Path(ckpt_path))
    system = fw.system
    dt = float(cfg["env"]["dt"]); max_steps = int(cfg["eval"]["max_steps"])
    world_lim = float(cfg["env"]["world_lim"])
    dtype, device = _tensor_options(system, fw)
    bscene = make_batched_scene(scenes, device=device, dtype=dtype)
    x0 = initial_states_from_batch(bscene)
    result = rollout_eval(system, fw.policy, _filter_adapter(fw), bscene, x0,
                          max_steps=max_steps, dt=dt, config=cfg)
    return result, cfg, dt, world_lim, system


res_jt, cfg_jt, dt, world_lim, system = roll_full(a.jt_ckpt)
res_pre, _, dt_pre, _, _ = roll_full(a.pre_ckpt)
assert abs(dt - dt_pre) < 1e-12, "dt mismatch between arms"


def extract(result, ep, canonical):
    S = result.states[:, ep, :].detach().to(torch.float64)
    masks = step_outcomes(result.states[:, ep:ep + 1, :], scenes[ep], system, cfg_jt)
    rolled = resolve_outcome(masks).outcome[0]
    if rolled != canonical[ep]:
        raise SystemExit(f"HALT: re-roll outcome '{rolled}' != canonical '{canonical[ep]}' for episode {ep} "
                         f"— the eval is not deterministic; that is a larger problem than these figures.")
    interv = result.intervention_mask[:, ep].detach().cpu().numpy()
    empty = (result.empty[:, ep].detach().cpu().numpy() if result.empty is not None
             else np.zeros(interv.shape, dtype=bool))
    return S.numpy(), interv, empty


def euler_tilt(S):
    R = _quat_to_R(torch.tensor(S[:, 3:7]))
    roll = np.degrees(torch.atan2(R[:, 2, 1], R[:, 2, 2]).numpy())
    pitch = np.degrees(torch.asin(torch.clamp(-R[:, 2, 0], -1.0, 1.0)).numpy())
    yaw = np.degrees(torch.atan2(R[:, 1, 0], R[:, 0, 0]).numpy())
    tilt = np.degrees(torch.arccos(torch.clamp(R[:, 2, 2], -1.0, 1.0)).numpy())
    return roll, pitch, yaw, tilt


def dist_surface(pos_xy, scene):
    cen = np.asarray(scene.obstacle_centers, float)[:, :2]; rad = np.asarray(scene.obstacle_radii, float)
    act = np.asarray(scene.obstacle_active, bool)
    if not act.any():
        return np.full(pos_xy.shape[0], np.inf)
    return (np.linalg.norm(pos_xy[:, None, :] - cen[None, act, :], axis=2) - rad[None, act]).min(axis=1)


def ttu_of(tilt, t):
    below = np.where(tilt < 30.0)[0]
    return float(t[below[0]]) if below.size else None


def shade(ax, mask, t, color, alpha):
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


JT_C, PRE_C = "tab:blue", "0.45"
sel_json_rows, paths = [], []
for ep in selected:
    Sj, interv_j, empty_j = extract(res_jt, ep, outc_jt)
    Sp, _, _ = extract(res_pre, ep, outc_pre)
    Tn = Sj.shape[0]; t = np.arange(Tn) * dt
    rj, pj, yj, tj = euler_tilt(Sj)
    rp, pp, yp, tp = euler_tilt(Sp)
    yj_u = np.degrees(np.unwrap(np.radians(yj))); yp_u = np.degrees(np.unwrap(np.radians(yp)))
    surf_j = dist_surface(Sj[:, :2], scenes[ep]); surf_p = dist_surface(Sp[:, :2], scenes[ep])
    ttu_j = ttu_of(tj, t); ttu_p = ttu_of(tp, t)
    interv_step = np.append(interv_j, interv_j[-1]); empty_step = np.append(empty_j, empty_j[-1])

    fig, axs = plt.subplots(5, 1, figsize=(9, 11.5), sharex=True,
                            gridspec_kw={"height_ratios": [1, 1, 1.3, 1, 0.7]})
    # panel 0: x / y
    axs[0].plot(t, Sp[:, 0], color=PRE_C, ls="--", lw=1.0); axs[0].plot(t, Sp[:, 1], color=PRE_C, ls=":", lw=1.0)
    axs[0].plot(t, Sj[:, 0], color="tab:red", lw=1.4, label="x"); axs[0].plot(t, Sj[:, 1], color="tab:green", lw=1.4, label="y")
    axs[0].set_ylabel("x,y (m)"); axs[0].legend(loc="upper right", fontsize=8, ncol=2)
    # panel 1: z + region floor
    axs[1].plot(t, Sp[:, 2], color=PRE_C, ls="--", lw=1.0)
    axs[1].plot(t, Sj[:, 2], color="tab:green", lw=1.4, label="z (JT)")
    axs[1].axhline(-world_lim, color="k", ls="--", lw=0.8, label="region floor")
    axs[1].set_ylabel("z (m)"); axs[1].legend(loc="upper right", fontsize=8)
    # panel 2: roll/pitch/tilt (both arms); tilt is the headline
    axs[2].plot(t, tp, color=PRE_C, ls="--", lw=1.3)
    axs[2].plot(t, rp, color="tab:blue", ls=":", lw=0.8, alpha=0.5); axs[2].plot(t, pp, color="tab:orange", ls=":", lw=0.8, alpha=0.5)
    axs[2].plot(t, rj, color="tab:blue", lw=1.0, label="roll (JT)"); axs[2].plot(t, pj, color="tab:orange", lw=1.0, label="pitch (JT)")
    axs[2].plot(t, tj, color="k", lw=1.9, label="tilt (JT)"); axs[2].axhline(30.0, color="grey", ls=":", lw=0.8)
    shade(axs[2], np.abs(pj) > 80.0, t, "red", 0.10)              # JT Euler-singular near |pitch|>80
    shade(axs[2], np.abs(pp) > 80.0, t, "grey", 0.08)             # pre-JT singular spans
    axs[2].set_ylabel("angle (deg)"); axs[2].legend(loc="upper right", fontsize=8, ncol=2)
    # panel 3: unwrapped yaw
    axs[3].plot(t, yp_u, color=PRE_C, ls="--", lw=1.1); axs[3].plot(t, yj_u, color="tab:purple", lw=1.3, label="yaw JT (unwrapped)")
    axs[3].set_ylabel("yaw (deg)"); axs[3].legend(loc="upper right", fontsize=8)
    # panel 4: distance to nearest surface
    axs[4].plot(t, surf_p, color=PRE_C, ls="--", lw=1.1); axs[4].plot(t, surf_j, color="tab:brown", lw=1.3)
    axs[4].axhline(0.0, color="r", lw=0.9); axs[4].set_ylabel("dist to\nsurface (m)"); axs[4].set_xlabel("time (s)")
    for ax in axs:
        shade(ax, interv_step, t, "tab:blue", 0.07)               # JT filter intervened
        for k in np.where(empty_step)[0]:                         # JT empty-branch steps
            ax.axvline(t[k], color="magenta", lw=0.5, alpha=0.4)
        if ttu_j is not None:
            ax.axvline(ttu_j, color="green", lw=1.5, ls="-")      # JT time-to-upright
        if ttu_p is not None:
            ax.axvline(ttu_p, color="darkorange", lw=1.3, ls="--")  # pre-JT time-to-upright
    handles = [Line2D([0], [0], color="k", lw=1.9, label="JT (solid)"),
               Line2D([0], [0], color=PRE_C, lw=1.3, ls="--", label="pre-JT (dashed)"),
               Line2D([0], [0], color="green", lw=1.5, label=f"JT t_upright {('%.2fs'%ttu_j) if ttu_j else 'never'}"),
               Line2D([0], [0], color="darkorange", lw=1.3, ls="--", label=f"preJT t_upright {('%.2fs'%ttu_p) if ttu_p else 'never'}")]
    axs[0].legend(handles=[axs[0].lines[2], axs[0].lines[3]] , loc="upper right", fontsize=8, ncol=2)
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"ep {ep} [{category[ep]}] | tilt0 {init_tilt[ep]:.0f}deg |omega0| {init_omega[ep]:.2f} "
                 f"|v0| {init_v[ep]:.2f} | preJT->{outc_pre[ep]}  JT->{outc_jt[ep]}", fontsize=10)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    out = out_dir / f"paired_ep{ep:04d}_{outc_pre[ep]}_to_{outc_jt[ep]}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    paths.append(str(out))
    sel_json_rows.append(dict(
        episode_idx=ep, category=category[ep], initial_tilt_deg=round(float(init_tilt[ep]), 2),
        initial_omega=round(float(init_omega[ep]), 3), initial_v=round(float(init_v[ep]), 3),
        preJT_outcome=outc_pre[ep], JT_outcome=outc_jt[ep],
        preJT_time_to_upright_s=(round(ttu_p, 3) if ttu_p is not None else None),
        JT_time_to_upright_s=(round(ttu_j, 3) if ttu_j is not None else None)))

sel_json = {
    "design": "paired pre-JT vs JT on the SAME selected episodes",
    "stratum_definition": "initial tilt >= 150 deg AND blocked (v2.7.3 M1 seg-intersect definition)",
    "stratum_size_before_selection": len(stratum),
    "selection_priority": ["flip preJT-collision->JT-goal (up to 6, |omega| desc)",
                           "JT collisions in stratum (all)", "top |omega| fill to 8"],
    "n_flip": len(flips), "n_JT_collision": len(jt_coll), "n_selected": len(selected),
    "notes": notes,
    "jt_ckpt": a.jt_ckpt, "pre_ckpt": a.pre_ckpt,
    "selected": sel_json_rows,
    "figure_paths": paths,
}
(out_dir / "flip_recovery_paired_selection.json").write_text(json.dumps(sel_json, indent=2) + "\n")
print(json.dumps(sel_json, indent=2))
print("SELECTED_EPISODE_IDS " + ",".join(str(ep) for ep in selected))
