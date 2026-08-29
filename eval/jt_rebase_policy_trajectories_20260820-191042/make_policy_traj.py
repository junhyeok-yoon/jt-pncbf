"""v2.9.3 jt_rebase -- the FIXED NOMINAL POLICY vs the LEARNED POLICY, both deployed ALONE,
re-taken on the COLD-started 40 000-step 3-D checkpoint.

    python eval/jt_rebase_policy_trajectories_20260820-191042/make_policy_traj.py

WHAT THIS IS. A diagnostic. It re-takes the trajectory comparison between

  nominal (unfiltered)   the FIXED nominal policy the OC-PNCBF construction consumes:
                         `QuadrotorQuad3D.lqr_action` (src/envs/quadrotor_3d.py:189), the cascaded
                         hover-PD -> wrench -> mixer-inverse -> per-rotor box clip. Untouched.
                         This is condition C3 of scripts/analysis/v292_pi_only.py (item 5).
  learned (unfiltered)   the jointly trained policy network alone: `fw.policy`, certificate ABSENT,
                         HardNet projection ABSENT, observation unchanged.
                         This is condition C1 of scripts/analysis/v292_pi_only.py (item 5).

NO certificate, NO projection, NO filter, on either policy. Both are driven through the SAME
producer -- `v292_pi_only.rollout_passthrough` -- from the SAME initial states, so every pair is
matched scene-for-scene.

THE CHECKPOINT is resolved from the COLD pointer
`data/runs/v2.9.3/jt_rebase/jtrow__quadrotor_3d__COLD40K.json` through the producer's own opt-in
`JT_ROW_ART` override, set below BEFORE `v292_pi_only` is imported (that module reads the env at
import time). The registered WARM artifact is never read, never edited, never moved.

TWO SCENE CLASSES.
  CLASS A  the registered scene law with the obstacle set EMPTY -- `v292_pi_only.build_grid`
           (item 4), 280 scenes, constructed in memory, NEVER written as a pool.
  CLASS B  the registered pool's own scenes, obstacles ACTIVE --
           data/secured_data/pools/eval_fullcb_quadrotor-3d-d2r_n2000_seed823456.pkl, n = 2000.

THE RULE THAT GOVERNS CLASS B. A collision does NOT terminate the episode and is NOT scored as an
outcome. The rollout continues to the horizon THROUGH the obstacle. That is exactly
`v292_pi_only.rollout_passthrough` (item 3): `collided` is suppressed from BOTH the freeze mask and
the outcome resolution, and the shadow contacts are counted over the live window. Obstacle entry is
recorded as an ANNOTATION -- where and when -- never as a stopping condition. This departs from the
registered outcome predicates, so this is a DIAGNOSTIC and NOT a scored cell: no ledger row, no cps.

SCENE SELECTION was fixed BEFORE any rollout ran and before any plot existed, and is on disk at
data/runs/v2.9.3/jt_rebase/policy_traj/selection_rule_preregistered.json. It is restated verbatim
in SELECTION_RULE below and re-emitted into the summary.

REUSE, not reimplementation:
  scripts/analysis/v292_pi_only.py        resolve_identity, build_grid, rollout_passthrough
  scripts/analysis/v292_pi_only_figs_pool.py  clearance (signed distance to the nearest cylinder)
  scripts/analysis/v282_agree_gate.py     gate_overrides (the registered cell)
  eval/fig1_teaser_intervention_20260819/make_fig1.py  draw_scene, endmark, rcParams, hue canon

WRITE SCOPE. PNGs + this script's manifest under this directory; numeric artifacts under
data/runs/v2.9.3/jt_rebase/policy_traj/. Nothing else is written, moved or deleted. No git, no
training, no src edit, no ledger edit, no promotion.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
import warnings
from pathlib import Path

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
# The COLD pointer, through the producer's own opt-in override. MUST precede the import below.
os.environ["JT_ROW_ART"] = str(REPO / "data/runs/v2.9.3/jt_rebase/jtrow__quadrotor_3d__COLD40K.json")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/analysis"))
sys.path.insert(0, str(REPO / "eval/fig1_teaser_intervention_20260819"))
import v292_pi_only as P                                   # noqa: E402
import v292_pi_only_figs_pool as PF                        # noqa: E402
import make_fig1 as F1                                     # noqa: E402
from v282_agree_gate import gate_overrides                 # noqa: E402

HERE = Path(__file__).resolve().parent
NUM = REPO / "data/runs/v2.9.3/jt_rebase/policy_traj"
POOL = REPO / "data/secured_data/pools/eval_fullcb_quadrotor-3d-d2r_n2000_seed823456.pkl"
CHUNK_B = 500

# --- the two policies. Names follow the paper's canon for the nominal; the learned policy alone is
# --- NOT the paper's "joint (ours)" (that string names the full filtered stack), so it is labelled
# --- for what it is.
NAME = {"nominal": "nominal (unfiltered)", "learned": "learned policy (unfiltered)"}
HUE = {"nominal": F1.HUE["nominal"], "learned": F1.HUE["jt"]}      # grey / red, house canon
C_START, C_GOAL = F1.C_START, F1.C_GOAL
C_ENTRY = "#e08214"

SELECTION_RULE = {
    "A": ("Fixed grid coordinates, no dependence on any measured quantity. Keep every grid scene "
          "whose cell is azimuth 0 deg, elevation 0 deg, distance 4.0 m -- one scene per registered "
          "initial-tilt value {0, 60, 90, 120, 180} deg. Ordered by ascending tilt."),
    "B": ("Two clauses, both scanned in ascending pool index, on the SHIPPED obstacle-contact "
          "channel StepOutcomeMasks.collided_obstacle evaluated over each policy's OWN live window. "
          "B1 = the FIRST THREE scenes where the fixed nominal enters an obstacle and the learned "
          "policy does NOT. B2 = the FIRST THREE scenes where NEITHER policy enters an obstacle."),
}

matplotlib.rcParams.update({
    "font.size": 8.0, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 8.0,
    "axes.linewidth": 0.7, "savefig.transparent": False,
})


# ------------------------------------------------------------------------------------ measurement
def measure(roll, bs, system, cfg):
    """Per-scene numbers + per-step profiles, all over the LIVE window of each episode.

    The live window is the runner's own `alive_x` / `alive_u` masks (monotone prefixes), so the
    window is exactly the one the pass-through rollout integrated; nothing is reconstructed.
    """
    from src.common.outcomes import StepOutcomeMasks, resolve_outcome, step_outcomes

    st = roll["states"]                                    # [T+1, B, 13]
    u = roll["u_dep"]                                      # [T,   B, 4]
    ax_ = roll["alive_x"]                                  # [T+1, B]
    au = roll["alive_u"]                                   # [T,   B]
    T1, B = st.shape[0], st.shape[1]
    dt = float(cfg["env"]["dt"])
    lo = system.u_bounds[:, 0].to(st.device, st.dtype)
    hi = system.u_bounds[:, 1].to(st.device, st.dtype)

    masks = step_outcomes(st, bs, system, cfg, actions=u)
    four = StepOutcomeMasks(collided=torch.zeros_like(masks.collided),
                            goal_reached=masks.goal_reached, oob=masks.oob, stuck=masks.stuck,
                            window_displacement=masks.window_displacement)
    res = resolve_outcome(four)
    outcome = np.array(res.outcome, dtype=object)
    ev = res.event_step.detach().cpu().numpy()
    assert (outcome != "collision").all(), "collision resolved under the pass-through predicate"

    p = system.position(st)                                # [T+1,B,3]
    n_live_x = ax_.sum(dim=0)                              # [B]
    n_live_u = au.sum(dim=0)                               # [B]

    # path length: the step increments the integrator actually took. Frozen steps contribute 0 by
    # construction (x is held), and the mask makes that explicit rather than assumed.
    dstep = torch.linalg.norm(p[1:] - p[:-1], dim=-1)      # [T,B]
    path_len = (dstep * ax_[1:].to(dstep.dtype)).sum(dim=0)

    # time to goal: the resolving step of the SHIPPED three-leg reach predicate, in seconds.
    t_goal = np.where(outcome == "goal", ev.astype(np.float64) * dt, np.nan)

    speed = system.speed(st)                               # [T+1,B]
    sp_live = speed.masked_fill(~ax_, float("nan"))
    sp_np = sp_live.detach().cpu().numpy().astype(np.float32)
    with warnings.catch_warnings():                     # an all-frozen column is legal, not an error
        warnings.simplefilter("ignore", RuntimeWarning)
        sp_mean = np.nanmean(sp_np, axis=0)
        sp_p50 = np.nanmedian(sp_np, axis=0)
        sp_max = np.nanmax(sp_np, axis=0)

    # actuator authority. u_bounds = [0, f_rotor_max] per rotor, so the box fraction of a command is
    # (u - lo)/(hi - lo) and its per-step scalar is the mean over the four rotors.
    frac = ((u - lo) / (hi - lo)).mean(dim=-1)             # [T,B]
    frac_live = frac.masked_fill(~au, float("nan"))
    fr_np = frac_live.detach().cpu().numpy().astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        au_mean = np.nanmean(fr_np, axis=0)
        au_p50 = np.nanmedian(fr_np, axis=0)
        au_max = np.nanmax(fr_np, axis=0)
    impulse = (u.sum(dim=-1) * au.to(u.dtype)).sum(dim=0) * dt          # N*s, summed over rotors
    at_bound = ((u <= lo + 1e-9) | (u >= hi - 1e-9)) & au.unsqueeze(-1)
    sat_share = at_bound.sum(dim=(0, 2)).to(u.dtype) / (n_live_u.to(u.dtype) * u.shape[-1]).clamp(min=1)
    both = au[1:] & au[:-1]
    rate = (torch.abs(u[1:] - u[:-1]).sum(dim=-1) * both.to(u.dtype)).sum(dim=0) / \
           (both.sum(dim=0).to(u.dtype).clamp(min=1) * dt)             # N/s, L1 over rotors

    # obstacle occupancy: the SHIPPED channel, over the live window. Terminates nothing.
    obs_ch = masks.collided_obstacle & ax_
    in_obs = obs_ch.sum(dim=0).to(st.dtype) / n_live_x.to(st.dtype).clamp(min=1)
    any_obs = obs_ch.any(dim=0)
    big = torch.full_like(obs_ch, False)
    idx = torch.where(obs_ch, torch.arange(T1, device=st.device).view(-1, 1).expand_as(obs_ch),
                      torch.full_like(big, T1, dtype=torch.long))
    entry_step = idx.min(dim=0).values
    entry_step = torch.where(any_obs, entry_step, torch.full_like(entry_step, -1))
    rows = torch.arange(B, device=st.device)
    entry_pos = p[entry_step.clamp(min=0), rows]                       # [B,3]
    entry_pos = torch.where(any_obs.unsqueeze(-1), entry_pos, torch.full_like(entry_pos, float("nan")))

    band = (masks.collided_band_lower | masks.collided_band_upper) & ax_
    in_band = band.sum(dim=0).to(st.dtype) / n_live_x.to(st.dtype).clamp(min=1)

    # deepest penetration: the reused signed-clearance helper, over the live window.
    clr = PF.clearance(st[:T1 - 1], bs, ax_[:T1 - 1])                  # [T,B], +inf off-window
    min_clr = clr.min(dim=0).values

    return dict(
        outcome=outcome.astype("U12"),
        event_step=ev.astype(np.int64),
        n_live_steps=n_live_x.detach().cpu().numpy().astype(np.int64),
        path_length_m=path_len.detach().cpu().numpy().astype(np.float64),
        time_to_goal_s=t_goal,
        speed_mean=sp_mean, speed_p50=sp_p50, speed_max=sp_max,
        authority_mean_frac_box=au_mean, authority_p50_frac_box=au_p50,
        authority_max_frac_box=au_max,
        impulse_Ns=impulse.detach().cpu().numpy().astype(np.float64),
        saturation_share=sat_share.detach().cpu().numpy().astype(np.float64),
        cmd_rate_N_per_s=rate.detach().cpu().numpy().astype(np.float64),
        inside_obstacle_frac=in_obs.detach().cpu().numpy().astype(np.float64),
        entered_obstacle=any_obs.detach().cpu().numpy(),
        obstacle_entry_step=entry_step.detach().cpu().numpy().astype(np.int64),
        obstacle_entry_pos=entry_pos.detach().cpu().numpy().astype(np.float64),
        inside_band_frac=in_band.detach().cpu().numpy().astype(np.float64),
        min_clearance_m=min_clr.detach().cpu().numpy().astype(np.float64),
        speed_profile=sp_np,                                            # [T+1,B] nan off-window
        authority_profile=fr_np,                                        # [T,  B] nan off-window
        traj=p.detach().cpu().numpy().astype(np.float32),               # [T+1,B,3]
        alive_x=ax_.detach().cpu().numpy(),
    )


def run_policy(fw, cfg, scenes, which, chunk):
    """One policy over one scene class. Both policies see the SAME x0 for the same scene."""
    from src.envs.scene_batch import batch_scenes as mk, initial_states_from_batch
    from src.eval.evaluate import _tensor_options

    system = fw.system
    dtype, device = _tensor_options(system, fw)
    identity = lambda x, u, s: (u, torch.zeros(x.shape[0], dtype=torch.bool, device=x.device))
    parts, t0 = [], time.time()
    for i in range(0, len(scenes), chunk):
        bs = mk(list(scenes[i:i + chunk]), device=device, dtype=dtype)
        x0 = initial_states_from_batch(bs)
        if which == "learned":                       # C1: certificate + projection ABSENT
            fw._prev_u = None
            pol, cap = fw.policy, bool(getattr(fw, "_u_prev_on", False))
        else:                                        # C3: the fixed LQR/PD nominal, unfiltered
            def pol(x, s):
                g = torch.as_tensor(s.goal, dtype=x.dtype, device=x.device)
                return system.lqr_action(x, g)
            cap = False
        roll = P.rollout_passthrough(fw, system, pol, identity, bs, x0, cfg, capture_prev_u=cap)
        parts.append(measure(roll, bs, system, cfg))
        del roll, bs, x0
        torch.cuda.empty_cache()
    out = {}
    for k in parts[0]:
        v = [q[k] for q in parts]
        out[k] = np.concatenate(v, axis=1) if k in ("speed_profile", "authority_profile",
                                                    "traj", "alive_x") else np.concatenate(v, axis=0)
    out["wall_s"] = round(time.time() - t0, 1)
    return out


def agg(m, sel=None):
    """Class-level summary of one policy. `sel` restricts to a subset; None = the whole class."""
    sl = slice(None) if sel is None else np.asarray(sel)

    def st(a, finite_only=False):
        a = np.asarray(a, dtype=np.float64)[sl]
        if finite_only:
            a = a[np.isfinite(a)]
        if a.size == 0:
            return {"n": 0}
        return {"n": int(a.size), "mean": float(a.mean()), "p50": float(np.median(a)),
                "max": float(a.max()), "min": float(a.min())}

    o = m["outcome"][sl]
    n = int(o.size)
    return {
        "n": n,
        "outcome_counts": {k: int((o == k).sum()) for k in ("goal", "oob", "timeout", "stuck")},
        "reach_share": float((o == "goal").mean()),
        "path_length_m": st(m["path_length_m"]),
        "time_to_goal_s": st(m["time_to_goal_s"], finite_only=True),
        "speed_mean_mps": st(m["speed_mean"]),
        "speed_p50_mps": st(m["speed_p50"]),
        "speed_max_mps": st(m["speed_max"]),
        "authority_mean_frac_box": st(m["authority_mean_frac_box"]),
        "authority_max_frac_box": st(m["authority_max_frac_box"]),
        "impulse_Ns": st(m["impulse_Ns"]),
        "saturation_share": st(m["saturation_share"]),
        "cmd_rate_N_per_s": st(m["cmd_rate_N_per_s"]),
        "live_steps": st(m["n_live_steps"]),
        "inside_obstacle_frac": st(m["inside_obstacle_frac"]),
        "episodes_entering_obstacle": int(np.asarray(m["entered_obstacle"])[sl].sum()),
        "inside_band_frac": st(m["inside_band_frac"]),
        "min_clearance_m": st(m["min_clearance_m"], finite_only=True),
    }


# ---------------------------------------------------------------------------------------- figures
def draw_paths(scenes, M, sel, titles, path_png, suptitle, world, ncol, fit=False):
    nrow = int(np.ceil(len(sel) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.55 * nrow), squeeze=False,
                            constrained_layout=True)
    for ax, si, ttl in zip(axs.ravel(), sel, titles):
        sc = scenes[int(si)]
        F1.draw_scene(ax, sc, world)                     # house helper: cylinders + equal aspect
        g = np.asarray(sc.goal, np.float64).reshape(-1)[:3]
        for k in ("nominal", "learned"):
            m = M[k]
            e = int(m["n_live_steps"][si]) - 1
            tr = m["traj"][:e + 1, int(si), :]
            ax.plot(tr[:, 0], tr[:, 1], color=HUE[k], lw=1.5, zorder=4 if k == "learned" else 3,
                    label=NAME[k], solid_capstyle="round")
            F1.endmark(ax, tr[:, :2], str(m["outcome"][si]), HUE[k], z=9)
            if bool(m["entered_obstacle"][si]):
                q = m["obstacle_entry_pos"][int(si)]
                ax.plot([q[0]], [q[1]], "X", ms=7.0, color=C_ENTRY, mec="black", mew=0.6, zorder=10)
        ax.plot([g[0]], [g[1]], "*", ms=10, color=C_GOAL, mec="black", mew=0.4, zorder=8)
        s0 = M["nominal"]["traj"][0, int(si), :]
        ax.plot([s0[0]], [s0[1]], "o", ms=5.0, color=C_START, mec="black", mew=0.4, zorder=8)
        ax.set_title(ttl, fontsize=8.0)
        ax.grid(alpha=0.2)
        if fit:                       # obstacle-free class: no arena furniture to anchor the view
            pts = [np.asarray(g[:2]).reshape(1, 2)]
            for k in ("nominal", "learned"):
                e = int(M[k]["n_live_steps"][si]) - 1
                pts.append(M[k]["traj"][:e + 1, int(si), :2])
            q = np.concatenate(pts, axis=0)
            cx, cy = 0.5 * (q[:, 0].min() + q[:, 0].max()), 0.5 * (q[:, 1].min() + q[:, 1].max())
            r = 0.55 * max(float(np.ptp(q[:, 0])), float(np.ptp(q[:, 1])), 0.5)
            ax.set_xlim(cx - r, cx + r); ax.set_ylim(cy - r, cy + r)
    for ax in axs.ravel()[len(sel):]:
        ax.axis("off")
    for r in range(nrow):
        axs[r][0].set_ylabel("y position (m)")
    h = [plt.Line2D([], [], color=HUE[k], lw=1.6, label=NAME[k]) for k in ("nominal", "learned")]
    h += [plt.Line2D([], [], color=C_START, marker="o", ls="none", ms=5, mec="black", mew=0.4,
                     label="start"),
          plt.Line2D([], [], color=C_GOAL, marker="*", ls="none", ms=10, mec="black", mew=0.4,
                     label="goal"),
          plt.Line2D([], [], color=C_ENTRY, marker="X", ls="none", ms=7, mec="black", mew=0.6,
                     label="first obstacle entry (non-terminating)"),
          plt.Line2D([], [], color="black", marker="o", ls="none", ms=5, mfc="none", mew=1.2,
                     label="arrival"),
          plt.Line2D([], [], color="black", marker="s", ls="none", ms=4.2, mfc="none", mew=1.2,
                     label="ran to horizon")]
    fig.legend(handles=h, loc="outside lower center", ncol=4, frameon=False, fontsize=7.6,
               handlelength=2.0, columnspacing=1.4)
    fig.suptitle(suptitle, fontsize=9.0)
    fig.savefig(path_png, dpi=200)
    plt.close(fig)


def draw_profiles(M, sel, titles, path_png, suptitle, dt, world_band):
    nrow = len(sel)
    fig, axs = plt.subplots(nrow, 3, figsize=(11.4, 2.15 * nrow), squeeze=False)
    for r, (si, ttl) in enumerate(zip(sel, titles)):
        for k in ("nominal", "learned"):
            m = M[k]
            e = int(m["n_live_steps"][si]) - 1
            t = np.arange(e + 1) * dt
            axs[r][0].plot(t, m["speed_profile"][:e + 1, int(si)], color=HUE[k], lw=1.3,
                           label=NAME[k])
            axs[r][1].plot(t, m["traj"][:e + 1, int(si), 2], color=HUE[k], lw=1.3)
            eu = max(e, 1)
            axs[r][2].plot(np.arange(eu) * dt, m["authority_profile"][:eu, int(si)], color=HUE[k],
                           lw=1.3)
            if bool(m["entered_obstacle"][si]):
                te = int(m["obstacle_entry_step"][si]) * dt
                for c in range(3):
                    axs[r][c].axvline(te, color=C_ENTRY, ls="--", lw=1.0, zorder=1)
        axs[r][1].axhline(world_band, color="0.4", ls=":", lw=0.9)
        axs[r][1].axhline(-world_band, color="0.4", ls=":", lw=0.9)
        axs[r][0].set_ylabel("speed (m/s)")
        axs[r][1].set_ylabel("altitude z (m)")
        axs[r][2].set_ylabel("box fraction")
        axs[r][0].set_title(ttl, fontsize=8.0, loc="left")
        for c in range(3):
            axs[r][c].grid(alpha=0.2)
            if r == nrow - 1:
                axs[r][c].set_xlabel("time (s)")
    axs[0][0].legend(fontsize=7.4, loc="upper right", frameon=False)
    axs[0][2].set_title("mean per-rotor command / box width", fontsize=8.0, loc="right")
    fig.suptitle(suptitle, fontsize=9.0)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(path_png, dpi=200)
    plt.close(fig)


def draw_aggregate(M, path_png, suptitle, with_obstacle):
    keys = [("path_length_m", "path length (m)", None),
            ("time_to_goal_s", "time to goal (s), reaching episodes", None),
            ("speed_max", "peak speed (m/s)", None),
            ("authority_mean_frac_box", "mean actuator authority (box fraction)", None),
            ("impulse_Ns", "actuator impulse (N s)", None),
            ("cmd_rate_N_per_s", "command rate (N/s, L1 over rotors)", None)]
    if with_obstacle:
        keys.append(("inside_obstacle_frac", "fraction of rollout inside an obstacle", None))
    ncol = 4
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.7 * nrow), squeeze=False)
    for ax, (key, lab, _) in zip(axs.ravel(), keys):
        data = {k: np.asarray(M[k][key], np.float64) for k in ("nominal", "learned")}
        data = {k: v[np.isfinite(v)] for k, v in data.items()}
        lohi = np.concatenate(list(data.values()))
        if lohi.size == 0:
            ax.axis("off"); continue
        lo, hi = float(lohi.min()), float(lohi.max())
        if hi <= lo:
            hi = lo + 1e-6
        bins = np.linspace(lo, hi, 45)
        for k in ("nominal", "learned"):
            ax.hist(data[k], bins=bins, histtype="step", lw=1.5, color=HUE[k], label=NAME[k])
        for k in ("nominal", "learned"):
            ax.axvline(float(np.median(data[k])), color=HUE[k], ls="--", lw=1.0)
        ax.set_xlabel(lab); ax.set_ylabel("episodes"); ax.grid(alpha=0.2)
    for ax in axs.ravel()[len(keys):]:
        ax.axis("off")
    axs[0][0].legend(fontsize=7.4, frameon=False)
    fig.suptitle(suptitle + "   (dashed = median)", fontsize=9.0)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path_png, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------------------------------- main
def main() -> int:
    NUM.mkdir(parents=True, exist_ok=True)
    free = int(os.popen("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits"
                        ).read().strip().split("\n")[0])
    print(f"[gpu] {free} MiB free", flush=True)
    assert free >= 5000, f"GPU gate: only {free} MiB free"

    ident = P.resolve_identity()
    print("[identity] " + json.dumps({k: ident[k] for k in
          ("row_artifact", "row_ckpt_step", "step_matches_row", "strict_load", "ok")}), flush=True)
    assert ident["ok"], "checkpoint identity failed"
    ckpt = REPO / ident["row_ckpt"]

    from src.frameworks.jt_pncbf.train import load_framework_from_checkpoint
    from src.eval.build_pools import load_pool, sha256_file
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fw, cfg, ck = load_framework_from_checkpoint(ckpt, config_overrides=copy.deepcopy(gate_overrides(ckpt)))
    for nm in ("value_net", "policy_net"):
        m = getattr(fw, nm, None)
        if m is not None:
            m.to(dev)
    dt = float(cfg["env"]["dt"])
    world = float(cfg["env"]["world_lim"])
    band = float(cfg["env"]["band_collision_limit"])
    ubox = fw.system.u_bounds.detach().cpu().numpy().tolist()
    hover = float(fw.system.mass * fw.system.gravity / 4.0)
    print(f"[cell] dt {dt}  max_steps {cfg['eval']['max_steps']}  world {world}  band {band}  "
          f"u_bounds {ubox[0]}  hover/rotor {hover:.4f} N", flush=True)

    scenes_A, meta_A, dropped_A = P.build_grid(cfg)
    pool = load_pool(POOL)
    scenes_B = list(pool.scenes)
    pool_sha = sha256_file(POOL)[:8]
    print(f"[classA] grid kept {len(scenes_A)} dropped {len(dropped_A)} (obstacle-free, in memory)",
          flush=True)
    print(f"[classB] pool {POOL.name} sha8 {pool_sha} n {len(scenes_B)} (obstacles ACTIVE)", flush=True)

    MA, MB = {}, {}
    for k in ("nominal", "learned"):
        MA[k] = run_policy(fw, cfg, scenes_A, k, chunk=len(scenes_A))
        o = MA[k]["outcome"]
        print(f"  [A/{k:8s}] {MA[k]['wall_s']:5.1f}s reach {(o=='goal').mean():.4f} "
              f"path p50 {np.median(MA[k]['path_length_m']):.3f} m", flush=True)
    for k in ("nominal", "learned"):
        MB[k] = run_policy(fw, cfg, scenes_B, k, chunk=CHUNK_B)
        o = MB[k]["outcome"]
        print(f"  [B/{k:8s}] {MB[k]['wall_s']:5.1f}s reach {(o=='goal').mean():.4f} "
              f"path p50 {np.median(MB[k]['path_length_m']):.3f} m  "
              f"entered obstacle {int(MB[k]['entered_obstacle'].sum())}/{len(scenes_B)}", flush=True)

    # ------------------------------------------------------------------ selection, rule then apply
    az = np.array([m["azimuth_deg"] for m in meta_A])
    el = np.array([m["elevation_deg"] for m in meta_A])
    di = np.array([m["distance_m"] for m in meta_A])
    ti = np.array([m["tilt_deg"] for m in meta_A])
    okA = (az == 0.0) & (el == 0.0) & (di == 4.0)
    selA = [int(i) for i in np.where(okA)[0][np.argsort(ti[okA])]]
    titlesA = [f"grid {i}: tilt {ti[i]:.0f} deg, az 0, el 0, d 4.0 m" for i in selA]
    print(f"[ruleA] {int(okA.sum())} scenes satisfy; selected {selA} "
          f"tilts {[float(ti[i]) for i in selA]}", flush=True)

    entN = MB["nominal"]["entered_obstacle"]
    entL = MB["learned"]["entered_obstacle"]
    b1_all = np.where(entN & ~entL)[0]
    b2_all = np.where(~entN & ~entL)[0]
    selB1 = [int(i) for i in b1_all[:3]]
    selB2 = [int(i) for i in b2_all[:3]]
    selB = selB1 + selB2
    titlesB = ([f"pool {i}: nominal enters, learned does not" for i in selB1]
               + [f"pool {i}: neither enters" for i in selB2])
    print(f"[ruleB] clause B1 satisfied by {len(b1_all)} scenes -> {selB1}; "
          f"clause B2 satisfied by {len(b2_all)} scenes -> {selB2}", flush=True)

    # ---------------------------------------------------------------------------------- figures
    pngs = {}
    draw_paths(scenes_A, MA, selA, titlesA, HERE / "classA_paths_xy.png",
               "Class A -- the registered scene law with the obstacle set EMPTY.\nBoth policies "
               "deployed alone, matched initial conditions, cold 40k checkpoint.",
               world, ncol=5, fit=True)
    pngs["classA_paths_xy.png"] = "class A top-down paths, five selected grid scenes"
    draw_profiles(MA, selA, titlesA, HERE / "classA_profiles.png",
                  "Class A -- speed, altitude and actuator authority against time", dt, band)
    pngs["classA_profiles.png"] = "class A per-scene speed / altitude / authority profiles"
    draw_aggregate(MA, HERE / "classA_aggregate.png",
                   f"Class A, all {len(scenes_A)} obstacle-free scenes", with_obstacle=False)
    pngs["classA_aggregate.png"] = "class A distributions over all 280 scenes"

    draw_paths(scenes_B, MB, selB, titlesB, HERE / "classB_paths_xy.png",
               "Class B -- the registered pool's own scenes, obstacles ACTIVE.\nCollision does NOT "
               "terminate: every path runs to its own horizon THROUGH the obstacle;\nthe cross marks "
               "first entry.", world, ncol=3)
    pngs["classB_paths_xy.png"] = "class B top-down paths, six selected pool scenes, obstacles drawn"
    draw_profiles(MB, selB, titlesB, HERE / "classB_profiles.png",
                  "Class B -- speed, altitude and actuator authority against time "
                  "(dashed vertical = first obstacle entry)", dt, band)
    pngs["classB_profiles.png"] = "class B per-scene speed / altitude / authority profiles"
    draw_aggregate(MB, HERE / "classB_aggregate.png",
                   f"Class B, all {len(scenes_B)} pool scenes, obstacles active, collision "
                   f"non-terminating", with_obstacle=True)
    pngs["classB_aggregate.png"] = "class B distributions over all 2000 scenes"

    # ---------------------------------------------------------------------------------- artifacts
    scalar_keys = [k for k in MA["nominal"] if k not in
                   ("speed_profile", "authority_profile", "traj", "alive_x", "wall_s")]
    axes_extra = {"classA": {"grid_azimuth_deg": az, "grid_elevation_deg": el,
                             "grid_distance_m": di, "grid_tilt_deg": ti},
                  "classB": {}}
    for tag, M, sel in (("classA", MA, selA), ("classB", MB, selB)):
        np.savez_compressed(
            NUM / f"perscene__{tag}.npz",
            selected_indices=np.asarray(sel, np.int64), **axes_extra[tag],
            **{f"{k}__{q}": np.asarray(M[k][q]) for k in M for q in scalar_keys})
        np.savez_compressed(
            NUM / f"profiles__{tag}.npz",
            dt=np.float64(dt),
            selected_indices=np.asarray(sel, np.int64),
            **{f"{k}__speed_profile": M[k]["speed_profile"] for k in M},
            **{f"{k}__authority_profile": M[k]["authority_profile"] for k in M},
            **{f"{k}__traj_selected": M[k]["traj"][:, np.asarray(sel, int), :] for k in M},
            **{f"{k}__alive_x_selected": M[k]["alive_x"][:, np.asarray(sel, int)] for k in M})

    def per_scene_rows(M, sel):
        rows = []
        for si in sel:
            r = {"scene": int(si)}
            for k in ("nominal", "learned"):
                m = M[k]
                r[k] = {
                    "outcome": str(m["outcome"][si]),
                    "live_steps": int(m["n_live_steps"][si]),
                    "path_length_m": float(m["path_length_m"][si]),
                    "time_to_goal_s": (None if not np.isfinite(m["time_to_goal_s"][si])
                                       else float(m["time_to_goal_s"][si])),
                    "speed_mean_mps": float(m["speed_mean"][si]),
                    "speed_p50_mps": float(m["speed_p50"][si]),
                    "speed_max_mps": float(m["speed_max"][si]),
                    "authority_mean_frac_box": float(m["authority_mean_frac_box"][si]),
                    "authority_max_frac_box": float(m["authority_max_frac_box"][si]),
                    "impulse_Ns": float(m["impulse_Ns"][si]),
                    "saturation_share": float(m["saturation_share"][si]),
                    "cmd_rate_N_per_s": float(m["cmd_rate_N_per_s"][si]),
                    "inside_obstacle_frac": float(m["inside_obstacle_frac"][si]),
                    "entered_obstacle": bool(m["entered_obstacle"][si]),
                    "obstacle_entry_step": int(m["obstacle_entry_step"][si]),
                    "obstacle_entry_time_s": (None if int(m["obstacle_entry_step"][si]) < 0
                                              else float(m["obstacle_entry_step"][si]) * dt),
                    "obstacle_entry_pos_m": (None if not bool(m["entered_obstacle"][si]) else
                                             [float(v) for v in m["obstacle_entry_pos"][si]]),
                    "min_clearance_m": float(m["min_clearance_m"][si]),
                }
            rows.append(r)
        return rows

    summary = {
        "what": "v2.9.3 jt_rebase -- fixed nominal policy vs learned policy, BOTH DEPLOYED ALONE "
                "(no certificate, no projection, no filter), on the COLD 40k 3-D checkpoint",
        "diagnostic": True,
        "ledger_row": None,
        "cps": None,
        "why_no_row": "the pass-through predicate suppresses `collided` from both the freeze mask "
                      "and the outcome resolution, which departs from the registered outcome "
                      "predicates; this is therefore not a scored cell",
        "producer": "eval/jt_rebase_policy_trajectories_20260820-191042/make_policy_traj.py",
        "reused": {
            "rollout + grid + identity": "scripts/analysis/v292_pi_only.py "
                                         "(resolve_identity, build_grid, rollout_passthrough)",
            "signed clearance": "scripts/analysis/v292_pi_only_figs_pool.py clearance",
            "cell": "scripts/analysis/v282_agree_gate.py gate_overrides",
            "plot helpers": "eval/fig1_teaser_intervention_20260819/make_fig1.py "
                            "(draw_scene, endmark, hue canon, rcParams)",
        },
        "checkpoint": {
            "pointer": ident["row_artifact"], "ckpt": ident["row_ckpt"],
            "ckpt_step": int(ident["ckpt_step_recorded"]),
            "step_matches_row": bool(ident["step_matches_row"]),
            "strict_load": bool(ident["strict_load"]),
            "warm_pointer_replaced": "data/runs/v2.9.1/launch/jtrow__quadrotor_3d.json "
                                     "(best.pt @ 9450, ledger L328/L312) -- READ-ONLY, untouched",
        },
        "policies": {
            "nominal": "QuadrotorQuad3D.lqr_action (src/envs/quadrotor_3d.py:189), the fixed "
                       "nominal the OC-PNCBF construction consumes, unfiltered. = v292_pi_only C3",
            "learned": "fw.policy alone, certificate and HardNet projection ABSENT, observation "
                       "unchanged. = v292_pi_only C1",
        },
        "cell": {"overrides": "v282_agree_gate.gate_overrides", "dt": dt,
                 "max_steps": int(cfg["eval"]["max_steps"]), "seed": 42,
                 "u_bounds": ubox, "hover_thrust_per_rotor_N": hover,
                 "world_lim": world, "band_collision_limit": band,
                 "goal_radius": float(cfg["env"]["goal_radius"]),
                 "goal_speed_radius": float(cfg["env"]["goal_speed_radius"]),
                 "goal_angrate_radius": float(cfg["env"].get("goal_angrate_radius", float("inf")))},
        "scene_classes": {
            "A": {"what": "registered scene law, obstacle set EMPTY", "n": len(scenes_A),
                  "built_by": "v292_pi_only.build_grid", "requested": len(meta_A) + len(dropped_A),
                  "dropped": len(dropped_A), "never_written_as_pool": True},
            "B": {"what": "the registered pool's own scenes, obstacles ACTIVE",
                  "n": len(scenes_B), "pool": str(POOL.relative_to(REPO)), "pool_sha8": pool_sha},
        },
        "selection_rule": SELECTION_RULE,
        "selection_rule_preregistered_at":
            "data/runs/v2.9.3/jt_rebase/policy_traj/selection_rule_preregistered.json",
        "selection": {
            "A": {"n_satisfying": int(okA.sum()), "indices": selA,
                  "tilt_deg": [float(ti[i]) for i in selA]},
            "B": {"clause_B1_n_satisfying": int(b1_all.size), "clause_B1_indices": selB1,
                  "clause_B2_n_satisfying": int(b2_all.size), "clause_B2_indices": selB2,
                  "indices": selB},
        },
        "class_tables": {
            "A": {k: agg(MA[k]) for k in ("nominal", "learned")},
            "B": {k: agg(MB[k]) for k in ("nominal", "learned")},
        },
        "class_B_split_by_selection_clause": {
            "B1_nominal_enters_learned_does_not": {k: agg(MB[k], b1_all) for k in ("nominal", "learned")},
            "B2_neither_enters": {k: agg(MB[k], b2_all) for k in ("nominal", "learned")},
        },
        "per_scene": {"A": per_scene_rows(MA, selA), "B": per_scene_rows(MB, selB)},
        "figures": {k: str((HERE / k).relative_to(REPO)) for k in pngs},
        "figure_captions": pngs,
        "warm_figures_replaced": [
            "data/runs/v2.9.2/pi_only/figures/grid_trajectories.png",
            "data/runs/v2.9.2/pi_only/figures/fullcb_trajectories.png",
            "data/runs/v2.9.2/pi_only/figures/grid_settling.png",
            "data/runs/v2.9.2/pi_only/figures/fullcb_clearance_hist.png",
        ],
        "warm_document": "docs/versions/v2.9.2/pi_only.md",
        "numeric_artifacts": [str((NUM / f).relative_to(REPO)) for f in
                             ("perscene__classA.npz", "perscene__classB.npz",
                              "profiles__classA.npz", "profiles__classB.npz",
                              "summary.json", "selection_rule_preregistered.json")],
        "walls_s": {"A": {k: MA[k]["wall_s"] for k in MA}, "B": {k: MB[k]["wall_s"] for k in MB}},
        "gpu_free_MiB_at_launch": free,
    }
    (NUM / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    (HERE / "manifest.json").write_text(json.dumps(
        {"producer": summary["producer"], "figures": summary["figures"],
         "numeric_artifacts": summary["numeric_artifacts"],
         "selection_rule": SELECTION_RULE, "selection": summary["selection"],
         "checkpoint": summary["checkpoint"], "diagnostic": True, "ledger_row": None, "cps": None},
        indent=1) + "\n", encoding="utf-8")

    for f in sorted(HERE.glob("*.png")):
        print(f"  wrote {f.relative_to(REPO)} ({f.stat().st_size // 1024} KiB)", flush=True)
    for f in sorted(NUM.glob("*")):
        print(f"  wrote {f.relative_to(REPO)} ({f.stat().st_size // 1024} KiB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
