"""Paper figure 1 — the teaser. Where the filter intervenes.

Three controllers, all rolled at eval.max_steps 400 on the registered cell
(`v282_agree_gate.gate_overrides`, registered pool, n 2000, eval_batch_size 2000, seed 42):

  nominal (unfiltered)   LQR, no certificate, no filter  (_rollout_lqr_batch)
  OC-PNCBF               L321's checkpoint (LQR nominal + its certificate)
  joint (ours)           L328's checkpoint (jointly trained policy + certificate)

Legend strings are the paper's canon EXACTLY: `nominal (unfiltered)`, `OC-PNCBF`, `joint (ours)`.

SCENE SELECTION IS A RULE, NOT A CHOICE. Scan the pool in ascending index and take the FIRST scene
on which the unfiltered nominal COLLIDES and BOTH learned controllers REACH. The rule is applied to
the outcome vectors of the same rollouts the trajectories are drawn from.

THE POINT OF THE FIGURE is where the projection intervenes. An intervening step is one where the
EXECUTED command differs from the policy's own proposal by more than a threshold:

    || u_safe(t) - u_nom(t) ||_2 > 1e-3    (rotor-thrust units, N)

which is the harness's own criterion -- `rollout_eval` builds exactly this as `intervention_mask`
(src/eval/rollout.py) and 04_eval s3 selects intervention episodes by it. No second implementation:
the mask drawn is the mask the harness returns. The nominal has no filter and carries one colour.

TWO VARIANTS, on the SAME selection rule:
  fig1_teaser_one_scene.pdf     one scene, one controller per panel, each learned path two-coloured
                                by the mask; the nominal single-coloured.
  fig1_teaser_three_scenes.pdf  three scenes (the first three satisfying the rule), one per panel,
                                each panel overlaying ALL THREE controllers; controller identity is
                                the line hue and the intervening segments carry the same amber halo
                                in every panel, so the intervention colouring is common to all.

07_tex_deck I3: no version stamp, run-id, pool name, scene index, checkpoint digest or ledger row
label is drawn.

Read-only on data/secured_data. No src edit, no config key on disk, no existing artifact touched.
"""
from __future__ import annotations
import copy, json, os, sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts/analysis"))
from v282_agree_gate import gate_overrides

HERE = Path(__file__).resolve().parent
# v2.9.3 jt_rebase: opt-in output override, same idiom as JT_ROW_ART. Unset -> this
# directory verbatim (default), so the warm figure on disk is never overwritten. Set ->
# data/runs/v2.9.3/jt_rebase/figures/fig1. Only the OUTPUT moves; HERE stays the code dir.
OUT = Path(os.environ["FIG_OUT"]) if os.environ.get("FIG_OUT") else HERE
POOL = REPO / "data/secured_data/pools/eval_fullcb_quadrotor-3d-d2r_n2000_seed823456.pkl"
OC_ART = REPO / "data/runs/v2.9.1/launch/ocrow__quadrotor_3d.json"   # L321's checkpoint
# v2.9.3 jt_rebase: opt-in override. Unset -> the warm L328 pointer verbatim (default,
# so every closed v2.9.1/v2.9.2 artifact stays reproducible). Set -> the cold pointer
# data/runs/v2.9.3/jt_rebase/jtrow__quadrotor_3d__COLD40K.json. The registered warm
# artifact itself is never edited and never moved.
JT_ART = Path(os.environ["JT_ROW_ART"]) if os.environ.get("JT_ROW_ART") else \
         REPO / "data/runs/v2.9.1/launch/jtrow__quadrotor_3d.json"   # L328's checkpoint
CAP = 400
EVAL_BATCH = 2000

# v2.9.3 jt_rebase: the JT row identity is READ FROM the pointer artifact, not hard-coded.
# The registered warm pointer carries no row field of its own, so the warm default keeps its
# registered value L328 unchanged; any other pointer must carry `ledger_row`, or
# `provenance.h400_sibling_row` (this producer rolls at cap 400), or the field is recorded as
# null rather than guessed.
WARM_JT_ART = REPO / "data/runs/v2.9.1/launch/jtrow__quadrotor_3d.json"


def jt_ledger_row():
    a = json.loads(JT_ART.read_text())
    r = a.get("ledger_row") or a.get("provenance", {}).get("h400_sibling_row")
    if r:
        return r
    return "L328" if JT_ART.resolve() == WARM_JT_ART.resolve() else None

W_DOUBLE_IN = 7.00
FIG_H_ONE, FIG_H_THREE = 3.05, 3.05
PT_LABEL, PT_TICK = 8.0, 7.0
INTERVENE_EPS = 1.0e-3            # metres/newtons: || u_safe - u_nom ||_2, the harness's own value

CANON = {"nominal": "nominal (unfiltered)", "oc": "OC-PNCBF", "jt": "joint (ours)"}
HUE = {"nominal": "#555555", "oc": "#2b6cb0", "jt": "#c1272d"}
C_INACTIVE, C_ACTIVE = "#2b6cb0", "#e08214"       # variant A: filter inactive | filter active
C_HALO = "#e08214"                                # variant B: common intervention halo
C_START, C_GOAL = "#2ca02c", "#d62728"

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": PT_TICK, "axes.labelsize": PT_LABEL, "axes.titlesize": PT_LABEL,
    "xtick.labelsize": PT_TICK, "ytick.labelsize": PT_TICK, "legend.fontsize": PT_LABEL,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "savefig.transparent": False,
})


# ---------------------------------------------------------------------------------------------
def roll(kind, art, scenes, dev):
    """Full-pool rollout at the registered cell, cap 400. Returns states, u_nom, u_safe, mask, ..."""
    from src.envs.scene_batch import batch_scenes, initial_states_from_batch
    from src.eval.evaluate import _tensor_options, _filter_adapter, _rollout_lqr_batch
    from src.eval.rollout import rollout_eval
    from src.common.outcomes import step_outcomes, resolve_outcome
    a = json.loads(art.read_text())
    ck = REPO / a["ckpt"]
    if kind == "jt":
        from src.frameworks.jt_pncbf.train import load_framework_from_checkpoint as LD
    else:
        from src.frameworks.oc_pncbf.train import load_framework_from_checkpoint as LD
    ov = copy.deepcopy(gate_overrides(ck)); ov["eval"]["max_steps"] = CAP
    fw, cfg, _ = LD(ck, config_overrides=ov)
    for nm in ("value_net", "policy_net"):
        m = getattr(fw, nm, None)
        if m is not None:
            m.to(dev)
    dt_, device = _tensor_options(fw.system, fw)
    bs = batch_scenes(list(scenes), device=device, dtype=dt_)
    x0 = initial_states_from_batch(bs)
    dt = float(cfg["env"]["dt"])
    if kind == "nominal":
        with torch.no_grad():
            r = _rollout_lqr_batch(fw.system, bs, x0, max_steps=CAP, dt=dt, config=cfg)
    else:
        r = rollout_eval(fw.system, fw.policy, _filter_adapter(fw), bs, x0,
                         max_steps=CAP, dt=dt, config=cfg)
    res = resolve_outcome(step_outcomes(r.states, bs, fw.system, cfg, actions=r.u_safe))
    box = fw.system.u_bounds
    box_range = float((box[:, 1] - box[:, 0]).max())
    out = dict(
        states=r.states.detach().cpu().numpy(),
        u_nom=r.u_nom.detach().cpu().numpy(),
        u_safe=r.u_safe.detach().cpu().numpy(),
        mask=r.intervention_mask.detach().cpu().numpy(),
        outcome=np.array(res.outcome, dtype=object),
        cause=np.array(list(res.collision_cause) or [""] * len(scenes), dtype=object),
        event_step=res.event_step.detach().cpu().numpy(),
        cfg=cfg, box_range=box_range, ckpt=a.get("ckpt"), ckpt_step=a.get("ckpt_step"))
    del fw, r, bs, x0
    torch.cuda.empty_cache()
    return out


def episode(d, si):
    """(n_drawn_states, positions[n,3], n_active_action_steps, mask over those steps, outcome)."""
    ev = int(d["event_step"][si])
    T = d["states"].shape[0]
    e = ev if ev >= 0 else T - 1
    e = max(1, min(e, T - 1))
    return e, d["states"][:e + 1, si, :3], d["mask"][:e, si], str(d["outcome"][si])


def two_colour_path(ax, p, mask, c_off, c_on, lw=1.5, z=3):
    """Segment t joins state t to state t+1 and takes the colour of action step t."""
    seg = np.stack([p[:-1], p[1:]], axis=1)
    col = np.where(mask[: len(seg)], c_on, c_off)
    lc = LineCollection(seg, colors=list(col), linewidths=lw, zorder=z, capstyle="round")
    ax.add_collection(lc)


def halo_path(ax, p, mask, hue, lw=1.4, halo_lw=3.2, z=4):
    """One hue for the controller; the intervening segments carry a common amber halo beneath it."""
    seg = np.stack([p[:-1], p[1:]], axis=1)
    m = mask[: len(seg)].astype(bool)
    if m.any():
        ax.add_collection(LineCollection(seg[m], colors=[C_HALO], linewidths=halo_lw,
                                         zorder=z - 1, alpha=0.85, capstyle="round"))
    ax.add_collection(LineCollection(seg, colors=[hue], linewidths=lw, zorder=z,
                                     capstyle="round"))


def draw_scene(ax, scene, world):
    C = np.asarray(scene.obstacle_centers, np.float64)
    R = np.asarray(scene.obstacle_radii, np.float64)
    A = np.asarray(scene.obstacle_active, bool)
    for j in np.nonzero(A)[0]:
        ax.add_patch(mpatches.Circle((C[j, 0], C[j, 1]), R[j], facecolor="0.86",
                                     edgecolor="0.45", lw=0.5, zorder=1))
    ax.set_aspect("equal"); ax.set_xlim(-world, world); ax.set_ylim(-world, world)
    ax.set_xlabel("x position (m)")


def endmark(ax, p, outcome, colour, z=9):
    """The termination of the path. Drawn ABOVE the goal star so an arrival is never hidden by it."""
    mk, ms = dict(collision=("X", 6.0), goal=("o", 5.0)).get(outcome, ("s", 4.2))
    ax.plot([p[-1, 0]], [p[-1, 1]], mk, color=colour, ms=ms,
            mfc=(colour if outcome == "collision" else "none"),
            mec=("black" if outcome == "collision" else colour),
            mew=(0.6 if outcome == "collision" else 1.2), zorder=z)


# ---------------------------------------------------------------------------------------------
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from src.eval.build_pools import load_pool, sha256_file
    dev = torch.device("cuda")
    pool = load_pool(POOL)
    pool_sha = sha256_file(POOL)[:8]
    scenes = list(pool.scenes)
    n_pool = len(scenes)
    print(f"pool sha8 {pool_sha}, {n_pool} scenes; rolling three conditions at cap {CAP}", flush=True)

    D = {}
    for kind, art in (("nominal", OC_ART), ("oc", OC_ART), ("jt", JT_ART)):
        D[kind] = roll(kind, art, scenes, dev)
        o = D[kind]["outcome"]
        print(f"  {kind:8s} rolled: goal {int((o=='goal').sum())}, collision "
              f"{int((o=='collision').sum())}, timeout {int((o=='timeout').sum())}, "
              f"oob {int((o=='oob').sum())}, stuck {int((o=='stuck').sum())}", flush=True)

    cfg = D["jt"]["cfg"]
    world = float(cfg["env"]["world_lim"])
    limit = float(cfg["env"].get("band_collision_limit", 4.0))
    dt_ctrl = float(cfg.get("eval", {}).get("dt_ctrl", cfg["env"]["dt"]))
    box_range = D["jt"]["box_range"]

    ok = ((D["nominal"]["outcome"] == "collision") & (D["oc"]["outcome"] == "goal")
          & (D["jt"]["outcome"] == "goal"))
    sat = np.where(ok)[0]
    n_sat = int(ok.sum())
    if n_sat == 0:
        raise SystemExit("STOP: no scene satisfies the selection rule")
    take_one = int(sat[0])
    take_three = [int(i) for i in sat[:3]]
    print(f"selection: {n_sat} of {n_pool} scenes satisfy the rule; "
          f"first {take_one}; first three {take_three}", flush=True)
    # persist the vectors the rule was applied to, so the selection is checkable without re-rolling
    np.savez_compressed(
        OUT / "outcomes.npz",
        outcome_nominal=D["nominal"]["outcome"].astype("U16"),
        outcome_oc=D["oc"]["outcome"].astype("U16"),
        outcome_jt=D["jt"]["outcome"].astype("U16"),
        event_step_nominal=D["nominal"]["event_step"],
        event_step_oc=D["oc"]["event_step"],
        event_step_jt=D["jt"]["event_step"],
        satisfying_indices=sat.astype(np.int64),
        intervening_steps_oc=D["oc"]["mask"].sum(axis=0).astype(np.int64),
        intervening_steps_jt=D["jt"]["mask"].sum(axis=0).astype(np.int64))

    # ---- intervention shares, measured on the drawn episodes -----------------------------------
    def shares(si):
        rec = {}
        for kind in ("nominal", "oc", "jt"):
            e, p, m, oc_ = episode(D[kind], si)
            rec[kind] = dict(
                controller=CANON[kind], outcome=oc_,
                collision_cause=(str(D[kind]["cause"][si]) if oc_ == "collision" else None),
                event_step=int(D[kind]["event_step"][si]), n_active_steps=int(e),
                n_intervening_steps=(0 if kind == "nominal" else int(m.sum())),
                intervening_share=(0.0 if kind == "nominal" else float(m.mean())),
                has_filter=(kind != "nominal"),
                duration_s=round(e * dt_ctrl, 3),
                altitude_range_m=[float(p[:, 2].min()), float(p[:, 2].max())],
                final_position_m=[float(v) for v in p[-1]],
                final_distance_to_goal_m=float(np.linalg.norm(
                    p[-1] - np.asarray(scenes[si].goal, np.float64).reshape(-1)[:3])),
                max_disc_norm=(None if kind == "nominal"
                               else float(np.linalg.norm(
                                   D[kind]["u_safe"][:e, si] - D[kind]["u_nom"][:e, si],
                                   axis=-1).max())))
        return rec

    per_scene_shares = {str(si): shares(si) for si in take_three}
    for si in take_three:
        for k in ("nominal", "oc", "jt"):
            r = per_scene_shares[str(si)][k]
            print(f"  scene {si:4d} {r['controller']:22s} {r['outcome']:9s} "
                  f"{r['n_intervening_steps']:4d}/{r['n_active_steps']:4d} steps intervening "
                  f"({r['intervening_share']*100:5.1f} %)"
                  + (f"  cause {r['collision_cause']}" if r["collision_cause"] else ""), flush=True)

    # pool-wide context: the share over every episode's active window
    pool_share = {}
    for kind in ("oc", "jt"):
        ev = D[kind]["event_step"]
        T = D[kind]["states"].shape[0]
        act = np.where(ev >= 0, ev, T - 1).astype(int)
        act = np.clip(act, 1, T - 1)
        m = D[kind]["mask"]
        tot_i = int(sum(int(m[: act[i], i].sum()) for i in range(n_pool)))
        tot_a = int(act.sum())
        pool_share[kind] = dict(controller=CANON[kind], n_intervening_steps=tot_i,
                                n_active_steps=tot_a, share=tot_i / tot_a)
        print(f"  pool-wide {CANON[kind]:22s} {tot_i}/{tot_a} steps intervening "
              f"({100*tot_i/tot_a:.2f} %)", flush=True)

    handles_common = [
        plt.Line2D([], [], color=C_START, marker="o", ls="none", ms=4.6, mec="black", mew=0.4,
                   label="start"),
        plt.Line2D([], [], color=C_GOAL, marker="*", ls="none", ms=9, mec="black", mew=0.4,
                   label="goal"),
        plt.Line2D([], [], color="black", marker="X", ls="none", ms=6.0, label="collision"),
        plt.Line2D([], [], color="black", marker="o", ls="none", ms=5.0, mfc="none", mew=1.2,
                   label="arrival")]

    # ---------------- variant A: one scene, one controller per panel ---------------------------
    si = take_one
    sc = scenes[si]
    goal = np.asarray(sc.goal, np.float64).reshape(-1)[:3]
    start = D["jt"]["states"][0, si, :3]
    figA, axA = plt.subplots(1, 3, figsize=(W_DOUBLE_IN, FIG_H_ONE), constrained_layout=True)
    for col, kind in enumerate(("nominal", "oc", "jt")):
        ax = axA[col]
        draw_scene(ax, sc, world)
        e, p, m, oc_ = episode(D[kind], si)
        if kind == "nominal":
            ax.add_collection(LineCollection(np.stack([p[:-1, :2], p[1:, :2]], axis=1),
                                             colors=[HUE["nominal"]], linewidths=1.5, zorder=3,
                                             capstyle="round"))
            endc = HUE["nominal"]
        else:
            two_colour_path(ax, p[:, :2], m, C_INACTIVE, C_ACTIVE, lw=1.6)
            endc = C_ACTIVE if (len(m) and m[-1]) else C_INACTIVE
        endmark(ax, p[:, :2], oc_, endc)
        ax.plot([goal[0]], [goal[1]], "*", ms=9, color=C_GOAL, mec="black", mew=0.4, zorder=8)
        ax.plot([start[0]], [start[1]], "o", ms=4.6, color=C_START, mec="black", mew=0.4, zorder=8)
        ax.set_title(CANON[kind], fontsize=PT_LABEL)
        if col == 0:
            ax.set_ylabel("y position (m)")
        ax.tick_params(pad=1.5)
    hA = [plt.Line2D([], [], color=C_INACTIVE, lw=1.6, label="filter inactive"),
          plt.Line2D([], [], color=C_ACTIVE, lw=1.6, label="filter active"),
          plt.Line2D([], [], color=HUE["nominal"], lw=1.6, label="no filter")] + handles_common
    figA.legend(handles=hA, loc="outside lower center", ncol=4, frameon=False, fontsize=PT_TICK,
                handlelength=1.9, columnspacing=1.2, borderpad=0.1)
    pdfA = OUT / "fig1_teaser_one_scene.pdf"
    figA.savefig(pdfA, format="pdf")
    figA.savefig(OUT / "preview_one_scene.png", format="png", dpi=300)
    plt.close(figA)

    # ---------------- variant B: three scenes, all three controllers per panel -----------------
    figB, axB = plt.subplots(1, 3, figsize=(W_DOUBLE_IN, FIG_H_THREE), constrained_layout=True)
    for col, sj in enumerate(take_three):
        ax = axB[col]
        scj = scenes[sj]
        draw_scene(ax, scj, world)
        gj = np.asarray(scj.goal, np.float64).reshape(-1)[:3]
        stj = D["jt"]["states"][0, sj, :3]
        # EVERY halo is laid down before ANY hue line, so one controller's intervention halo can
        # never bury another controller's path where the two overlap.
        for kind, zc in (("oc", 4), ("jt", 5)):
            _, p, m, _ = episode(D[kind], sj)
            seg = np.stack([p[:-1, :2], p[1:, :2]], axis=1)
            mm = m[: len(seg)].astype(bool)
            if mm.any():
                ax.add_collection(LineCollection(seg[mm], colors=[C_HALO], linewidths=3.2,
                                                 zorder=zc, alpha=0.85, capstyle="round"))
        for kind, zc in (("nominal", 3), ("oc", 6), ("jt", 7)):
            e, p, m, oc_ = episode(D[kind], sj)
            ax.add_collection(LineCollection(np.stack([p[:-1, :2], p[1:, :2]], axis=1),
                                             colors=[HUE[kind]], linewidths=1.4,
                                             zorder=zc, capstyle="round"))
            endmark(ax, p[:, :2], oc_, HUE[kind], z=9 + zc)
        ax.plot([gj[0]], [gj[1]], "*", ms=9, color=C_GOAL, mec="black", mew=0.4, zorder=8)
        ax.plot([stj[0]], [stj[1]], "o", ms=4.6, color=C_START, mec="black", mew=0.4, zorder=8)
        ax.set_title(f"({'abc'[col]})", fontsize=PT_LABEL, fontweight="bold", loc="left", pad=2.0)
        if col == 0:
            ax.set_ylabel("y position (m)")
        ax.tick_params(pad=1.5)
    hB = [plt.Line2D([], [], color=HUE[k], lw=1.6, label=CANON[k]) for k in ("nominal", "oc", "jt")]
    hB += [plt.Line2D([], [], color=C_HALO, lw=3.2, alpha=0.85, label="filter active")]
    hB += handles_common
    figB.legend(handles=hB, loc="outside lower center", ncol=4, frameon=False, fontsize=PT_TICK,
                handlelength=1.9, columnspacing=1.2, borderpad=0.1)
    pdfB = OUT / "fig1_teaser_three_scenes.pdf"
    figB.savefig(pdfB, format="pdf")
    figB.savefig(OUT / "preview_three_scenes.png", format="png", dpi=300)
    plt.close(figB)

    # ---------------- manifest -----------------------------------------------------------------
    sys.path.insert(0, str(REPO / "eval/fig4_certificate_zero_level_20260819"))
    import measure_pdf
    meas = {p.name: measure_pdf.measure(p, target_width_in=W_DOUBLE_IN) for p in (pdfA, pdfB)}
    (OUT / "pdf_measurements.json").write_text(json.dumps(meas, indent=2) + "\n")

    jt_row = jt_ledger_row()
    man = dict(
        figures=[pdfA.name, pdfB.name],
        column="double", target_width_in=W_DOUBLE_IN,
        column_choice_reason="three arenas side by side; at the 3.40 in single-column target each "
                             "panel would be 1.1 in wide (about 10 pt per metre of arena) and the "
                             "7 pt tick labels would dominate the trajectories the figure exists "
                             "to show",
        panel_layout=dict(one_scene="1 row x 3 columns, one controller per panel, one scene",
                          three_scenes="1 row x 3 columns, one scene per panel, all three "
                                       "controllers overlaid in every panel"),
        font_pt=dict(label=PT_LABEL, tick=PT_TICK, title=PT_LABEL, legend=PT_TICK, smallest=PT_TICK),
        cell="v282_agree_gate.gate_overrides with eval.max_steps 400", eval_max_steps=CAP,
        eval_batch_size=EVAL_BATCH, seed=42,
        pool=str(POOL.relative_to(REPO)), pool_sha8=pool_sha, n_pool=n_pool,
        dt_ctrl_s=dt_ctrl, world_lim=world, band_collision_limit=limit,
        legend_canon=CANON,
        conditions=dict(
            nominal=dict(label=CANON["nominal"], what="unfiltered LQR, no certificate, no filter",
                         checkpoint=None, ledger_row=None),
            oc=dict(label=CANON["oc"], what="L321's checkpoint, LQR nominal + its certificate",
                    checkpoint=D["oc"]["ckpt"], checkpoint_step=D["oc"]["ckpt_step"],
                    ledger_row="L321"),
            jt=dict(label=CANON["jt"],
                    what=f"{jt_row or 'the JT pointer'}'s checkpoint, jointly trained "
                         f"policy+certificate",
                    checkpoint=D["jt"]["ckpt"], checkpoint_step=D["jt"]["ckpt_step"],
                    ledger_row=jt_row, pointer_artifact=str(JT_ART),
                    condition=json.loads(JT_ART.read_text()).get("condition"))),
        selection_rule="ascending pool index; the FIRST scene on which the unfiltered nominal "
                       "COLLIDES and BOTH learned controllers REACH, read off the outcome vectors "
                       "of the same cap-400 rollouts the trajectories are drawn from",
        n_scenes_satisfying=n_sat, n_pool_scenes=n_pool,
        satisfying_indices_first20=[int(i) for i in sat[:20]],
        scene_index_one_scene=take_one, scene_indices_three_scenes=take_three,
        intervention_threshold=dict(
            definition="a step is intervening iff || u_safe(t) - u_nom(t) ||_2 > eps",
            eps=INTERVENE_EPS, units="newtons (rotor thrust), the action space's own units",
            eps_as_fraction_of_action_box=INTERVENE_EPS / box_range, action_box_range_N=box_range,
            source="src/eval/rollout.py rollout_eval -> RolloutResult.intervention_mask; the mask "
                   "drawn IS the mask the harness returns, not a second implementation",
            u_nom_is="the policy's own proposal before projection (the learned policy for the "
                     "joint pair, the LQR command for OC-PNCBF); the unfiltered nominal has no "
                     "filter and no intervening step by construction",
            denominator="the episode's ACTIVE steps, i.e. steps 0..event_step-1, the same steps "
                        "the drawn path covers"),
        intervention_shares_per_scene=per_scene_shares,
        intervention_share_pool_wide=pool_share,
        outcome_counts={k: {o: int((D[k]["outcome"] == o).sum())
                            for o in ("goal", "collision", "timeout", "oob", "stuck")}
                        for k in ("nominal", "oc", "jt")},
        colouring=dict(
            one_scene="two-colour path: filter inactive %s, filter active %s; the unfiltered "
                      "nominal is one colour %s" % (C_INACTIVE, C_ACTIVE, HUE["nominal"]),
            three_scenes="controller identity is the line hue (%s); the intervening segments carry "
                         "the SAME amber halo %s in every panel, so the intervention colouring is "
                         "common across controllers. EVERY halo is drawn before ANY hue line, so "
                         "one controller's halo never buries another's path; where two paths "
                         "coincide the higher-zorder hue (joint) still occludes the lower (OC), "
                         "which is the unavoidable cost of overlaying three controllers on one "
                         "arena and is the reason this variant is not recommended over the other"
                         % (json.dumps(HUE), C_HALO)),
        projection_caveat="both variants draw the xy projection only. The obstacles are INFINITE "
                          "vertical cylinders, so the xy projection is complete for obstacle "
                          "avoidance -- no altitude change lets a path cross a drawn circle. What "
                          "the projection hides is the floor/ceiling band at |p_z| = "
                          "band_collision_limit and how much vertical motion each path carries; "
                          "the per-episode altitude range is recorded beside each share, and it is "
                          "large (several metres on every drawn episode), so a caption should not "
                          "describe these as planar paths. Every drawn nominal collision has "
                          "collision_cause 'obstacle', not a band collision, so no drawn "
                          "termination is invisible in this projection.",
        measured=meas)
    (OUT / "manifest_fig1.json").write_text(json.dumps(man, indent=2) + "\n")
    for k, v in meas.items():
        print(f"measured {k}: MediaBox {v['mediabox_width_in']} x {v['mediabox_height_in']} in, "
              f"Tf {v['distinct_tf_pt']} pt, all >= 6pt {v['all_text_ge_6pt']}, "
              f"images {v['n_image_xobjects']}", flush=True)
    print(f"wrote {pdfA}\nwrote {pdfB}", flush=True)


if __name__ == "__main__":
    main()
