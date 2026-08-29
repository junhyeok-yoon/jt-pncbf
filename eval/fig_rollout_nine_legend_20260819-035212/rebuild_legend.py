"""fig_rollout_nine.pdf — legend regeneration only.

Rebuilds the nine-scene rollout figure with the paper's controller canon in the legend and NO
other change. The three condition labels are overridden in memory; colours, line widths, markers,
panel titles and the (a)-(s) panel labels are untouched, and the drawing code is the shipped
producer's own -- this file imports `draw_traj_pair` and `episode` from it rather than restating
them, so no drawing logic is duplicated or altered.

The shipped producer RE-SIMULATES from the registered checkpoints. A determinism gate therefore
runs first, and rendering is skipped entirely unless it passes.

Read-only on the repository: no existing script, figure or artifact is edited, overwritten or
moved. Everything this run produces lands in this directory.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts/analysis"))
import v292_fig_rollouts as RO
import v292_fig_nine as NINE
import v292_mppi_n2000 as M

HERE = Path(__file__).resolve().parent
TAB1 = REPO / "data/runs/v2.9.2/horizon400_tableI"
PERSISTED = {"nominal": "perepisode__NOM_quadrotor_3d_h400.npz",
             "cert":    "perepisode__L305_h400.npz",
             "pair":    "perepisode__L312_h400.npz"}

# The ONLY change: the three condition labels. Colours, keys, order and line styles are copied
# from the shipped table unchanged.
NEW_LABEL = {"nominal": "nominal (unfiltered)", "cert": "OC-PNCBF", "pair": "joint (ours)"}


def main() -> int:
    report: dict = {"script_rebuilt_from": "scripts/analysis/v292_fig_nine.py",
                    "replays_or_resimulates": "re-simulates (loads the registered checkpoints and "
                                              "rolls the full pool at eval.max_steps 400)",
                    "legend_change": {k: {"old": lab, "new": NEW_LABEL[k]}
                                      for k, lab, _c, _s in RO.COND for kk in [k] if kk == k},
                    "colors_unchanged": {k: c for k, _lab, c, _s in RO.COND}}

    from src.eval.build_pools import load_pool
    dev = torch.device("cuda")
    pool = load_pool(RO.POOL)
    all_scenes = list(pool.scenes)
    tilt = M.spawn_tilt_deg(pool)

    full = {}
    for kind, art in (("nominal", RO.OC_ART), ("cert", RO.OC_ART), ("pair", RO.JT_ART)):
        full[kind] = RO.roll(kind, art, all_scenes, dev)
        print(f"  rolled {kind}", flush=True)

    # ---------------- determinism gate ---------------------------------------------------------
    # manifest_nine.json pins the scene lists, initial conditions and geometry but carries NO
    # per-scene outcome or termination time, so the reference used here is the registered
    # per-episode vector of the same cell at the same cap, which does carry both.
    scenes_all = NINE.BLOCK1 + NINE.BLOCK2
    gate = {"reference": {k: str((TAB1 / f).relative_to(REPO)) for k, f in PERSISTED.items()},
            "note": ("manifest_nine.json carries no per-scene outcome or termination time; the "
                     "registered per-episode vectors of the same cells at cap 400 are used instead"),
            "cap": RO.CAP, "per_condition": {}, "nine_scene_detail": []}
    ok = True
    for kind in ("nominal", "cert", "pair"):
        st, out, ev, _cfg = full[kind]
        d = np.load(TAB1 / PERSISTED[kind], allow_pickle=True)
        ref_out = d["outcome"].astype(str)
        # schema differs by condition: the two learned vectors carry , the nominal carries
        #  (-1 == no physical termination, i.e. the episode ran to the cap).
        if "n_steps" in d:
            ref_t = np.asarray(d["n_steps"]).astype(int)
            ref_t_field = "n_steps"
        else:
            _e = np.asarray(d["event_step"]).astype(int)
            ref_t = np.where(_e >= 0, _e, RO.CAP).astype(int)
            ref_t_field = "event_step (-1 -> cap)"
        got_out = np.asarray(out).astype(str)
        got_t = np.where(np.asarray(ev) >= 0, np.asarray(ev), RO.CAP).astype(int)
        o_mis = int((got_out != ref_out).sum())
        t_mis = int((got_t != ref_t).sum())
        gate["per_condition"][kind] = {
            "n": int(ref_out.size),
            "reference_time_field": ref_t_field,
            "outcome_mismatches_full_pool": o_mis,
            "termination_time_mismatches_full_pool": t_mis,
            "nine_scene_outcome_mismatches": int(sum(got_out[s] != ref_out[s] for s in scenes_all)),
            "nine_scene_time_mismatches": int(sum(got_t[s] != ref_t[s] for s in scenes_all)),
        }
        ok = ok and o_mis == 0 and t_mis == 0
        for s in scenes_all:
            gate["nine_scene_detail"].append(dict(
                scene=int(s), condition=kind,
                outcome_reference=str(ref_out[s]), outcome_resimulated=str(got_out[s]),
                termination_step_reference=int(ref_t[s]), termination_step_resimulated=int(got_t[s]),
                match=bool(got_out[s] == ref_out[s] and got_t[s] == ref_t[s])))
        print(f"  gate {kind}: outcome mismatches {o_mis}, time mismatches {t_mis} (of {ref_out.size})",
              flush=True)
    gate["pass"] = bool(ok)
    report["determinism_gate"] = gate
    (HERE / "determinism_gate.json").write_text(json.dumps(gate, indent=2) + "\n")
    if not ok:
        report["rendered"] = False
        report["stopped_because"] = "determinism gate failed; no figure was rendered"
        (HERE / "diff_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print("GATE FAILED — nothing rendered", flush=True)
        return 2

    # ---------------- the ONLY edit: the three legend strings -----------------------------------
    RO.COND = [(k, NEW_LABEL[k], c, s) for k, _lab, c, s in RO.COND]

    # ---------------- figure 1, verbatim from v292_fig_nine.main() ------------------------------
    cfg = full["pair"][3]
    world = float(cfg["env"]["world_lim"])
    limit = float(cfg["env"].get("band_collision_limit", 4.0))
    dt_ctrl = float(cfg.get("eval", {}).get("dt_ctrl", cfg["env"]["dt"]))
    xmax = 1
    for si in scenes_all:
        for k in ("nominal", "cert", "pair"):
            xmax = max(xmax, NINE.episode(full, k, si)[0])

    NCOL, PT_LABEL, PT_TICK = NINE.NCOL, NINE.PT_LABEL, NINE.PT_TICK
    fig, axes = plt.subplots(4, NCOL, figsize=(NINE.W_DOUBLE_IN, NINE.W_DOUBLE_IN * 0.98),
                             constrained_layout=True)
    blocks = [(0, NINE.BLOCK1), (2, NINE.BLOCK2)]
    for r0, blk in blocks:
        for col in range(NCOL):
            if col >= len(blk):
                axes[r0][col].axis("off"); axes[r0 + 1][col].axis("off")
                continue
            si = blk[col]
            NINE.draw_traj_pair(axes[r0][col], axes[r0 + 1][col], full, si, all_scenes[si],
                                world, limit, dt_ctrl, xmax, float(tilt[si]), col == 0)
    handles = [plt.Line2D([], [], color=c, lw=1.4, label=lab) for _, lab, c, _ in RO.COND]
    handles += [plt.Line2D([], [], color="#2ca02c", marker="o", ls="none", ms=4.5, mec="black",
                           mew=0.4, label="start"),
                plt.Line2D([], [], color="#d62728", marker="*", ls="none", ms=9, mec="black",
                           mew=0.4, label="goal"),
                plt.Line2D([], [], color="black", marker="X", ls="none", ms=5.5, label="collision"),
                plt.Line2D([], [], color="black", marker="o", ls="none", ms=4, mfc="none",
                           label="arrival")]
    leg = fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False,
                     fontsize=PT_LABEL, handlelength=2.0, columnspacing=1.3)
    for r0, blk in blocks:
        for col in range(len(blk)):
            axes[r0][col].text(0.04, 0.96, f"({'abcdefghijklmnopqrst'[r0 * NCOL + col]})",
                               transform=axes[r0][col].transAxes, va="top", ha="left",
                               fontsize=PT_TICK, fontweight="bold",
                               bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.8))
            axes[r0 + 1][col].text(0.04, 0.96, f"({'abcdefghijklmnopqrst'[(r0 + 1) * NCOL + col]})",
                                   transform=axes[r0 + 1][col].transAxes, va="top", ha="left",
                                   fontsize=PT_TICK, fontweight="bold",
                                   bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.8))
    out_pdf = HERE / "fig_rollout_nine.pdf"
    fig.savefig(out_pdf, format="pdf")

    fig.canvas.draw()
    lb = leg.get_window_extent()
    W, H = fig.get_size_inches()
    report["canvas_in"] = [round(float(W), 4), round(float(H), 4)]
    report["legend_bbox_figure_fraction"] = dict(
        x0=float(lb.x0 / (W * fig.dpi)), x1=float(lb.x1 / (W * fig.dpi)),
        y0=float(lb.y0 / (H * fig.dpi)), y1=float(lb.y1 / (H * fig.dpi)))
    bb = axes[0][0].get_window_extent()
    report["arena_panel"] = dict(
        arena_panel_width_in=round(bb.width / fig.dpi, 4),
        arena_panel_width_pt=round(bb.width / fig.dpi * 72.0, 2),
        data_span_m=2 * world,
        pt_per_metre=round((bb.width / fig.dpi * 72.0) / (2 * world), 3))
    plt.close(fig)
    report["rendered"] = True
    report["output_pdf"] = str(out_pdf.relative_to(REPO))
    (HERE / "render_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out_pdf}", flush=True)
    print("legend bbox (figure fraction):", json.dumps(report["legend_bbox_figure_fraction"]), flush=True)
    print("arena panel:", json.dumps(report["arena_panel"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
