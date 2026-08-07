"""Trajectory-control grid figures for the PPO baseline (04_eval §3), reused at EVERY in-loop eval and by the
standalone script. Reuses the JT panel primitives (src.eval.plotting), so the grids render in the JT style:
per-panel trajectory (deployed PPO solid, LQR-only nominal dotted; start/goal/collision markers; title =
outcome) beside the control panel with ONE LINE PER ACTION COMPONENT (m=4). Panels are selected to SPAN
OUTCOMES (reach/collision/timeout, then stuck/oob) rather than the first N. The caption states the CBF-QP
filtered trace is ABSENT BY CONSTRUCTION (no certificate, no filter), not omitted.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

_PREFERRED_ORDER = ["goal", "collision", "timeout", "stuck", "oob"]


def _select_spanning(trajectories, n_panels):
    buckets = defaultdict(list)
    for idx, tr in enumerate(trajectories):
        buckets[tr.filtered_outcome].append((idx, tr))
    order = [o for o in _PREFERRED_ORDER if o in buckets] + [o for o in buckets if o not in _PREFERRED_ORDER]
    selected, cursor = [], {o: 0 for o in order}
    while len(selected) < n_panels and any(cursor[o] < len(buckets[o]) for o in order):
        for o in order:
            if cursor[o] < len(buckets[o]):
                selected.append(buckets[o][cursor[o]])
                cursor[o] += 1
                if len(selected) >= n_panels:
                    break
    return selected, {o: len(buckets[o]) for o in order}


def write_trajectory_grids(system, config: Mapping[str, Any], result, out_dir: Path,
                           *, step: int, n_scenes: int = 96, eg_lock=None,
                           name_prefix: str | None = None) -> list[Path]:
    """v2.8.2 B2/B3: draw outcome-spanning grids from an ALREADY-COMPUTED in-loop eval `result`
    (NO second PPO rollout). Panels are selected from the first `n_scenes` trajectories of `result`
    (the same subset the former standalone eval scored, so selection is unchanged). The LQR baseline
    is rolled ONLY for the selected panel scenes (PANELS_PER_FIGURE*2 = 32), batched in ONE call --
    per-scene independent, so each selected panel's LQR trace is bit-identical to before. If `eg_lock`
    is given, ONLY that 32-scene LQR rollout is serialized under the eval gate (B3); panel selection,
    matplotlib rendering and file writes run AFTER release. Figures: <prefix>_{A,B}.png."""
    from src.eval.evaluate import _rollout_lqr_batch
    from src.envs.scene_batch import batch_scenes as make_batched_scene, initial_states_from_batch
    from src.common.eval_gate import eval_gate
    from src.eval.plotting import (
        EpisodeControlSpec, PANELS_PER_FIGURE, _legend_handles, _plot_control_panel,
        _plot_trajectory_panel, _to_numpy,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected, counts = _select_spanning(result.trajectories[: int(n_scenes)], PANELS_PER_FIGURE * 2)

    # --- the ONLY GPU work: roll the LQR baseline for JUST the selected panel scenes (one batched
    #     call). Gate this alone if a lock is given (B3); everything after runs ungated. ---
    panel_lqr: list = [None] * len(selected)
    if selected:
        sel_scenes = [tr.scene for _, tr in selected]
        ref = selected[0][1].filtered.states
        bscene = make_batched_scene(sel_scenes, device=ref.device, dtype=ref.dtype)
        x0 = initial_states_from_batch(bscene)
        max_steps = int(config["eval"]["max_steps"])
        dt = float(config["env"]["dt"])

        def _roll():
            with torch.no_grad():
                return _rollout_lqr_batch(system, bscene, x0, max_steps=max_steps, dt=dt, config=config)

        if eg_lock is not None:
            with eval_gate(eg_lock, enabled=True, timeout_s=300.0, log=lambda m: print(m, flush=True)):
                lqr_batch = _roll()
        else:
            lqr_batch = _roll()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        panel_lqr = [lqr_batch[:, k : k + 1, :] for k in range(len(selected))]

    world_lim = float(config["env"]["world_lim"])
    bounds = _to_numpy(system.u_bounds).astype(float)
    role = f"PPO baseline (no certificate / no filter) · deployed PPO vs LQR-only · step {step}"
    absent = [o for o in ("goal", "collision", "timeout") if counts.get(o, 0) == 0]
    caption = ("Deployed = unfiltered PPO policy (solid, one line per rotor command). Baseline = LQR-only "
               "nominal (grey). The CBF-QP-filtered trace is ABSENT BY CONSTRUCTION (no certificate, no filter), "
               "not omitted. Panels selected to span outcomes.")
    if absent:
        caption += f" Outcome(s) with zero episodes: {', '.join(absent)}."
    prefix = name_prefix or f"ppo_trajectory_grid_step{int(step):06d}"

    panels = [(pi, tr, panel_lqr[k]) for k, (pi, tr) in enumerate(selected)]
    paths = []
    for fi, letter in enumerate(["A", "B"]):
        chunk = panels[fi * PANELS_PER_FIGURE:(fi + 1) * PANELS_PER_FIGURE]
        if not chunk:
            break
        specs = []
        for pool_index, tr, lqr_states in chunk:
            specs.append(EpisodeControlSpec(
                scene=tr.scene, pool_index=pool_index, outcome=tr.filtered_outcome,
                event_step=tr.filtered_event_step, filtered_states=tr.filtered.states[:, 0, :],
                intervention_mask=tr.filtered.intervention_mask[:, 0],
                u_nom=tr.filtered.u_safe[:, 0, :], u_safe=tr.filtered.u_safe[:, 0, :],
                nominal_states=None if lqr_states is None else lqr_states[:, 0, :]))
        fig, axes = plt.subplots(4, 4, figsize=(16, 16), dpi=120)
        fig.suptitle(f"{role} · {system.name} · Figure {letter}", fontsize=13)
        for slot in range(PANELS_PER_FIGURE):
            row, col = slot // 2, (slot % 2) * 2
            if slot < len(specs):
                _plot_trajectory_panel(axes[row, col], specs[slot], world_lim)
                _plot_control_panel(axes[row, col + 1], specs[slot], bounds, drop_nominal=True)
            else:
                axes[row, col].axis("off")
                axes[row, col + 1].axis("off")
        tl, cl = _legend_handles(min(bounds.shape[0], 4))
        fig.legend(handles=tl, loc="lower center", bbox_to_anchor=(0.5, 0.052), ncol=6, frameon=False, fontsize=10)
        fig.legend(handles=cl, loc="lower center", bbox_to_anchor=(0.5, 0.022), ncol=5, frameon=False, fontsize=10)
        fig.text(0.5, 0.002, caption, ha="center", va="bottom", fontsize=9, wrap=True)
        fig.tight_layout(rect=(0.0, 0.085, 1.0, 0.955))
        p = out_dir / f"{prefix}_{letter}.png"
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)
    return paths
