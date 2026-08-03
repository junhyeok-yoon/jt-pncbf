from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping

from matplotlib import image as mpimg
import numpy as np
import torch
import yaml

from src._version import __version__
from src.common.outcomes import StepOutcomeMasks, resolve_outcome, step_outcomes
from src.common.rk4 import rk4_step
from src.common.signed_h import signed_h
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.envs.scene_init import Scene
from src.eval.build_pools import obstacle_distribution_name, pool_stem
from src.eval.evaluate import (
    EVAL_EPISODE_COLUMNS,
    EVAL_METRIC_COLUMNS,
    EvaluationResult,
    active_action_steps,
    active_bool_fraction,
    eval_min_window_displacement,
    evaluate,
    first_physical_event_step,
    saturation_step_fraction,
    stuck_bin_fractions,
)
from src.eval.bootstrap import within_seed_ci
from src.eval.plotting import (
    EPISODES_PER_REPORT,
    PANELS_PER_FIGURE,
    EpisodeControlSpec,
    plot_cbf_contours,
    plot_quadrotor_cbf_contour,
    plot_trajectory_control_grid,
)
from src.eval.rollout import RolloutResult


REPO_ROOT = Path(__file__).resolve().parents[2]
POOL_DIR = REPO_ROOT / "data/secured_data/pools"


@dataclass(frozen=True)
class FullEvalResult:
    run_dir: Path
    eval_result: EvaluationResult
    figure_paths: list[Path]
    filter_works_count: int
    intervention_episode_count: int
    plotted_episode_count: int


_FINAL_CELL_SPECS = {
    "mixed": ("eval_inloop_quadrotor-3d-d2r-mixed_n2000_seed45678", True),
    "tilt60": ("eval_navcone_quadrotor-3d-d2r_n2000_seed34567", True),
    "bandopen": ("eval_full_quadrotor-3d-d2r_n2000_seed23456", False),
}


def final_cell_specs(config: Mapping[str, Any], system_name: str) -> list[tuple[str, str | None, bool]]:
    """v2.8.0 W2: the final-eval cells. quadrotor_3d -> mixed / tilt60 / bandopen (restrictable via
    eval.final.cells, default all three); every other system -> a single 'main' cell (today's behaviour,
    pool resolved as before, band terminating). Returns (cell_name, pool_stem_or_None, band_terminates)."""
    if system_name != "quadrotor_3d":
        return [("main", None, True)]
    requested = config.get("eval", {}).get("final", {}).get("cells", ["mixed", "tilt60", "bandopen"])
    return [(c, _FINAL_CELL_SPECS[c][0], _FINAL_CELL_SPECS[c][1]) for c in requested if c in _FINAL_CELL_SPECS]


def dual_scoring_ledger_rows(
    *, date: str, parent: str, run_id: str, seeds: str, blended_cps: float,
    tilt60: Mapping[str, Any], bandopen: Mapping[str, Any],
) -> list[str]:
    """v2.8.0 W3: emit the two adjacent eval_only ledger rows for a quadrotor_3d final eval — one carrying
    cps_tilt60, one carrying cps_bandopen as `value (crossing rate)` — sharing parent/date/run id, with the
    mixed cell's blended cps recorded in both rows' eval_source for continuity with the historical cps column.
    18 columns (schema of Part L); no unescaped `|` in text (norm written as ‖). Each returned string parses
    to the header's cell count under the R-audit parser."""
    def _row(cell, cps_tilt60_cell, cps_bandopen_cell, src, verdict):
        o = cell
        return ("| %s | quadrotor_3d | %s | %s | %s | %.4f | %s | %s | %s | %.4f | %.4f | %.4f | %.4f | "
                "%.4f | %.4f | %.4f | %.4f | %s |" % (
                    __version__, date, parent, seeds, blended_cps, cps_tilt60_cell, cps_bandopen_cell, src,
                    o["reach"], o["collision"], o["oob"], o["stuck"], o["timeout"], o["infeasibility"],
                    o["saturation_rate"], o["cps"], verdict))
    src = "eval_only(dual-scoring final, run %s; mixed blended cps %.4f)" % (run_id, blended_cps)
    row_tilt = _row(tilt60, "%.4f" % tilt60["cps"], "", src,
                    "cps_tilt60 cell (navcone e54d8cd6, band terminating); read with cps_bandopen row; "
                    "mixed blended %.4f; unbolded" % blended_cps)
    row_band = _row(bandopen, "", "%.4f (%.4f)" % (bandopen["cps"], bandopen["crossing_rate"]), src,
                    "cps_bandopen cell (canonical 0ef3751b, band-open); value (crossing rate); read with "
                    "cps_tilt60 row; mixed blended %.4f; unbolded" % blended_cps)
    return [row_tilt, row_band]


def run_final_cells(
    run_dir: Path,
    framework: Any,
    config: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
    max_scenes: int | None = None,
) -> dict[str, dict[str, Any]]:
    """v2.8.0 W2: run the dual-scoring final cells for quadrotor_3d (mixed / tilt60 / bandopen), each on its
    own pool with its own band_terminates setting, writing a named artifact set under run_dir/eval/<cell>/
    (eval_metrics.csv with the half sub-scores where applicable, eval_episodes.csv with collision_cause and
    band_crossings, config.yaml). Non-3D systems and the 'main' cell are handled by run_full_eval's standing
    single eval; run_final_cells only produces the pool-backed 3-D cells."""
    import copy
    from src.eval.build_pools import resolve_pool_or_raise
    # Each cell reuses the passed framework (within-process eval reuse is deterministic) and differs only by
    # its pool and its env.band_terminates readout. NOTE (v2.8.0 W4): the n2000 GPU eval is not bit-reproducible
    # across *processes* (~0.002-0.004; a handful of borderline episodes flip); cells reproduce their standalone
    # sidecars within that band, not bit-for-bit. See docs/versions/v2.8.0/wiring_and_chatter.md (W4).
    out: dict[str, dict[str, Any]] = {}
    for name, pool_stem, band_term in final_cell_specs(config, framework.system.name):
        if pool_stem is None:
            continue
        cell_cfg = copy.deepcopy(_plain_data(config))
        cell_cfg.setdefault("env", {})["band_terminates"] = bool(band_term)
        res = evaluate(framework, resolve_pool_or_raise(pool_stem), cell_cfg, mode="final",
                       step=int(checkpoint["step"]), ckpt_name=checkpoint_path.name,
                       max_scenes=max_scenes, include_lqr_baseline=False)
        cell_dir = run_dir / "eval" / name
        cell_dir.mkdir(parents=True, exist_ok=True)
        cols = EVAL_METRIC_COLUMNS + [c for c in ("cps_full_half", "cps_tilt60_half") if c in res.eval_row]
        _append_csv(cell_dir / "eval_metrics.csv", cols, [res.eval_row])
        _append_csv(cell_dir / "eval_episodes.csv", EVAL_EPISODE_COLUMNS, res.episode_rows)
        (cell_dir / "config.yaml").write_text(yaml.safe_dump(cell_cfg, sort_keys=False), encoding="utf-8")
        out[name] = {"metrics": cell_dir / "eval_metrics.csv", "episodes": cell_dir / "eval_episodes.csv",
                     "config": cell_dir / "config.yaml", "cps": float(res.eval_row["cps"]),
                     "band_terminates": bool(band_term)}
    return out


def run_full_eval(
    run_dir: Path,
    *,
    ckpt_path: Path | None = None,
    use_last: bool = False,
    max_scenes: int | None = None,
    config_overrides: Mapping[str, Any] | None = None,
    include_insertion: bool = True,
) -> FullEvalResult:
    checkpoint_path = _checkpoint_path(run_dir, ckpt_path, use_last)
    framework, config, checkpoint = _load_framework(
        checkpoint_path,
        config_overrides=config_overrides,
    )
    _initialize_eval_dir_if_needed(run_dir, config, checkpoint_path)
    pool_path = _full_pool_path(config)
    eval_result = evaluate(
        framework,
        pool_path,
        config,
        mode="final",
        step=int(checkpoint["step"]),
        ckpt_name=checkpoint_path.name,
        max_scenes=max_scenes,
        include_lqr_baseline=True,
    )
    _append_csv(run_dir / "eval_metrics.csv", EVAL_METRIC_COLUMNS, [eval_result.eval_row])
    _append_csv(run_dir / "eval_episodes.csv", EVAL_EPISODE_COLUMNS, eval_result.episode_rows)
    _persist_action_stream(run_dir, eval_result, ckpt_name=checkpoint_path.name)

    figure_result = write_trajectory_figures(
        run_dir=run_dir,
        eval_result=eval_result,
        config=config,
        system_name=framework.system.name,
        u_bounds=framework.system.u_bounds,
        role="Final eval",
        output_dir=run_dir / "figures",
        filename_template="trajectory_grid_{letter}.png",
    )
    contour_path = write_cbf_contour_figure(
        eval_result=eval_result,
        config=config,
        system=framework.system,
        value_net=framework.value_net,
        output_path=run_dir / "figures/cbf_contour.png",
        role="Final eval CBF contour",
    )
    if contour_path is not None:                       # v2.6.0: None => contour skipped (quadrotor)
        _write_tb_image(
            run_dir / "tensorboard",
            "eval/final/cbf_contour",
            contour_path,
            int(checkpoint["step"]),
        )
    if include_insertion:
        insertion_eval_rows, insertion_episode_rows = _run_online_insertion(
            framework,
            eval_result,
            config,
            step=int(checkpoint["step"]),
            ckpt_name=checkpoint_path.name,
        )
        _append_csv(run_dir / "eval_metrics.csv", EVAL_METRIC_COLUMNS, insertion_eval_rows)
        _append_csv(run_dir / "eval_episodes.csv", EVAL_EPISODE_COLUMNS, insertion_episode_rows)
    # v2.8.0 W2: for quadrotor_3d the final eval additionally emits the dual-scoring cells (mixed / tilt60 /
    # bandopen) as named artifact sets under run_dir/eval/<cell>/. Non-3D systems are untouched (the standing
    # single eval above is their one 'main' cell).
    if framework.system.name == "quadrotor_3d":
        run_final_cells(run_dir, framework, config, checkpoint, checkpoint_path, max_scenes=max_scenes)
    filter_works_count = sum(
        1
        for trajectory in eval_result.trajectories[:32]
        if trajectory.lqr_outcome == "collision"
        and trajectory.filtered_outcome != "collision"
    )
    _write_status_done(run_dir, checkpoint, eval_result)
    _rewrite_report(run_dir)
    return FullEvalResult(
        run_dir=run_dir,
        eval_result=eval_result,
        figure_paths=[*figure_result.paths, *( [contour_path] if contour_path is not None else [] )],
        filter_works_count=filter_works_count,
        intervention_episode_count=figure_result.intervention_episode_count,
        plotted_episode_count=figure_result.plotted_episode_count,
    )


def _persist_action_stream(
    run_dir: Path,
    eval_result: Any,
    *,
    ckpt_name: str,
) -> Path | None:
    """v2.7.3 M6 amendment: persist the per-step deployed action stream for every canonical-eval episode.

    Writes the commanded action u_cmd (= filtered u_safe, all four rotor commands) and, since it is already
    in scope from the same rollout, the pre-filter nominal u_nom — indexed by step and episode id — as a
    single compressed array alongside eval_episodes.csv. This reuses tensors already computed by the rollout
    (no second forward pass). It exists so the post-hoc command-swing / first-order-motor-lag misdelivery
    number can be computed later without re-running the eval; no metric is computed here.

    episode_idx / n_steps are copied from eval_result.episode_rows (the same join keys written to
    eval_episodes.csv); trajectories align 1:1 with those rows. Actions are stored float32 (thrust in N,
    range ~[0, 4.905]); the untrimmed length is max_steps for every episode, so n_steps gives the physical
    horizon for trimming the zero-padded tail after episode termination.
    """
    trajectories = getattr(eval_result, "trajectories", None)
    episode_rows = getattr(eval_result, "episode_rows", None)
    if not trajectories or not episode_rows:
        return None
    if len(trajectories) != len(episode_rows):                    # alignment is the whole contract; refuse if broken
        raise RuntimeError(
            f"action-stream persistence: {len(trajectories)} trajectories vs "
            f"{len(episode_rows)} episode rows — cannot align."
        )
    cmd = [traj.filtered.u_safe[:, 0, :].detach().to(dtype=torch.float32).cpu().numpy() for traj in trajectories]
    nom = [traj.filtered.u_nom[:, 0, :].detach().to(dtype=torch.float32).cpu().numpy() for traj in trajectories]
    u_cmd = np.stack(cmd, axis=0)                                 # [n_episodes, max_steps, action_dim]
    u_nom = np.stack(nom, axis=0)
    episode_idx = np.asarray([int(row["episode_idx"]) for row in episode_rows], dtype=np.int64)
    n_steps = np.asarray([int(row["n_steps"]) for row in episode_rows], dtype=np.int64)
    step_index = np.arange(u_cmd.shape[1], dtype=np.int64)
    path = run_dir / "eval_action_stream.npz"
    np.savez_compressed(
        path,
        u_cmd=u_cmd,
        u_nom=u_nom,
        episode_idx=episode_idx,
        n_steps=n_steps,
        step_index=step_index,
        ckpt_name=np.asarray(ckpt_name),
        mode=np.asarray("final"),
        note=np.asarray("u_cmd=filtered u_safe; u_nom=pre-filter nominal; tail past n_steps is done-zeroed"),
    )
    n_rows = int(u_cmd.shape[0]) * int(u_cmd.shape[1])
    print(
        f"[action-stream] wrote {path} "
        f"shape u_cmd/u_nom {u_cmd.shape} ({n_rows} episode-steps) "
        f"size {path.stat().st_size / 1e6:.2f} MB",
        flush=True,
    )
    return path


def _load_framework(
    checkpoint_path: Path,
    config_overrides: Mapping[str, Any] | None = None,
) -> tuple[Any, Mapping[str, Any], dict[str, Any]]:
    # v2.8.0 B6.2: load to the run device rather than forcing CPU (was map_location="cpu").
    _run_dev = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=_run_dev, weights_only=False)
    framework = str(checkpoint.get("framework", "oc_pncbf"))
    if framework == "oc_pncbf":
        from src.frameworks.oc_pncbf.train import load_framework_from_checkpoint

        return load_framework_from_checkpoint(
            checkpoint_path,
            config_overrides=config_overrides,
        )
    if framework == "jt_pncbf":
        from src.frameworks.jt_pncbf.train import load_framework_from_checkpoint

        return load_framework_from_checkpoint(
            checkpoint_path,
            config_overrides=config_overrides,
        )
    raise ValueError(f"Unsupported checkpoint framework: {framework!r}")


def _checkpoint_path(
    run_dir: Path,
    ckpt_path: Path | None,
    use_last: bool,
) -> Path:
    if ckpt_path is not None:
        return ckpt_path
    if use_last:
        return run_dir / "checkpoints/final.pt"
    return run_dir / "checkpoints/best.pt"


def _full_pool_path(config: Mapping[str, Any]) -> Path:
    # v2.7.3 M0c: variant (from config) selects the scene-variant pool; hard-assert existence + SHA match.
    from src.eval.build_pools import pool_variant, resolve_pool_or_raise

    n_scenes = int(config["eval"]["full"]["n"])
    seed = int(config["eval"]["full"]["seed"])
    system_name = str(config["run"]["system"])
    stem = pool_stem("full", system_name, n_scenes, seed, obstacle_distribution_name(config),
                     pool_variant(config, system_name))
    return resolve_pool_or_raise(stem)          # secured then eval_pools (v2.7.4 -d2r lives in eval_pools)


@dataclass(frozen=True)
class FigureWriteResult:
    paths: list[Path]
    intervention_episode_count: int
    plotted_episode_count: int
    shortfall: int


def write_trajectory_figures(
    *,
    run_dir: Path,
    eval_result: EvaluationResult,
    config: Mapping[str, Any],
    system_name: str,
    u_bounds: torch.Tensor,
    role: str,
    output_dir: Path,
    filename_template: str,
) -> FigureWriteResult:
    _ = run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = _selected_intervention_episodes(eval_result)
    intervention_count = len(selected)
    shortfall = max(0, EPISODES_PER_REPORT - intervention_count)
    selected = selected[:EPISODES_PER_REPORT]
    paths = []
    for letter, start in (("A", 0), ("B", PANELS_PER_FIGURE)):
        figure_episodes = selected[start : start + PANELS_PER_FIGURE]
        output_path = output_dir / filename_template.format(letter=letter)
        plot_trajectory_control_grid(
            episodes=[
                _episode_control_spec(pool_index, episode)
                for pool_index, episode in figure_episodes
            ],
            output_path=output_path,
            config=config,
            role=role,
            system_name=system_name,
            letter=letter,
            u_bounds=u_bounds,
            total_selected=intervention_count,
            shortfall=shortfall,
        )
        paths.append(output_path)
    return FigureWriteResult(
        paths=paths,
        intervention_episode_count=intervention_count,
        plotted_episode_count=len(selected),
        shortfall=shortfall,
    )


def write_cbf_contour_figure(
    *,
    eval_result: EvaluationResult,
    config: Mapping[str, Any],
    system: Any,
    value_net: Any,
    output_path: Path,
    role: str,
) -> Path | None:
    # v2.6.1: the 6D quadrotor has no 2D velocity-column slice, so it uses a position-plane contour across
    # three approach-speed slices (h_star velocity channel). Viz-only, not gate-relevant; wrapped so a
    # figure error never propagates into the eval / training loop.
    if getattr(system, "name", None) == "quadrotor_planar":
        try:
            return plot_quadrotor_cbf_contour(
                scenes=eval_result.pool.scenes[:2],
                output_path=output_path,
                config=config,
                system=system,
                value_net=value_net,
                role=role,
            )
        except Exception:
            return None
    # v2.7.3 amendment: quadrotor_3d fixed 3-panel V_hat contour (recipe registered in plotting.py). Uses the
    # IN-LOOP d2 pool scene 0 (held for every frame, independent of the eval pool). Viz-only; wrapped.
    if getattr(system, "name", None) == "quadrotor_3d":
        try:
            from src.eval.build_pools import load_pool, pool_stem, pool_variant, resolve_pool_or_raise
            from src.eval.plotting import plot_quadrotor3d_cbf_contour
            stem = pool_stem("inloop", "quadrotor_3d", int(config["eval"]["in_loop"]["n"]),
                             int(config["eval"]["in_loop"]["seed"]), obstacle_distribution_name(config),
                             pool_variant(config, "quadrotor_3d"))
            scene0 = load_pool(resolve_pool_or_raise(stem)).scenes[0]   # secured|eval_pools (v2.7.4 -d2r)
            return plot_quadrotor3d_cbf_contour(scene0, output_path, config, system, value_net, role)
        except Exception:
            return None
    # other >2D systems have no 2D velocity slice; skip (viz-only).
    if getattr(system, "name", None) not in {"double_integrator", "unicycle"}:
        return None
    plot_cbf_contours(
        scenes=eval_result.pool.scenes[:2],
        output_path=output_path,
        config=config,
        system=system,
        value_net=value_net,
        role=role,
    )
    return output_path


def log_png_to_tensorboard(writer: Any, tag: str, path: Path, step: int) -> None:
    if not hasattr(writer, "add_image"):
        return
    image = mpimg.imread(path)
    writer.add_image(tag, image, int(step), dataformats="HWC")


def _selected_intervention_episodes(eval_result: EvaluationResult) -> list[tuple[int, Any]]:
    return [
        (idx, episode)
        for idx, episode in enumerate(eval_result.trajectories)
        if bool(episode.filtered.intervention_mask.any())
    ]


def _episode_control_spec(pool_index: int, episode: Any) -> EpisodeControlSpec:
    return EpisodeControlSpec(
        scene=episode.scene,
        pool_index=pool_index,
        outcome=episode.filtered_outcome,
        event_step=episode.filtered_event_step,
        filtered_states=episode.filtered.states[:, 0, :],
        intervention_mask=episode.filtered.intervention_mask[:, 0],
        u_nom=episode.filtered.u_nom[:, 0, :],
        u_safe=episode.filtered.u_safe[:, 0, :],
        nominal_states=None if episode.lqr_states is None else episode.lqr_states[:, 0, :],
    )


def _run_online_insertion(
    framework: Any,
    eval_result: EvaluationResult,
    config: Mapping[str, Any],
    *,
    step: int,
    ckpt_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    insertion_cfg = config["eval"]["insertion"]
    t_insert = int(insertion_cfg["t_insert"])
    radius = float(insertion_cfg["radius"])
    inserted_scenes = [
        _insert_scene_obstacle(
            episode.scene,
            episode.filtered.states[:, 0, :],
            framework.system,
            t_insert,
            radius,
        )
        for episode in eval_result.trajectories
    ]

    modes: dict[str, list[tuple[Scene, Scene, RolloutResult]]] = {
        "final_insertion_lqr": [],
        "final_insertion_frozen": [],
    }
    for episode, inserted_scene in zip(
        eval_result.trajectories,
        inserted_scenes,
        strict=True,
    ):
        if episode.lqr_states is None:
            raise ValueError("Final insertion requires LQR baseline trajectories.")
        modes["final_insertion_lqr"].append(
            (
                episode.scene,
                inserted_scene,
                _lqr_result_from_states(
                    episode.lqr_states,
                    episode.scene,
                    framework.system,
                ),
            )
        )
        modes["final_insertion_frozen"].append(
            (episode.scene, inserted_scene, episode.filtered)
        )

    live_result = _rollout_live_insertion(
        framework,
        [episode.scene for episode in eval_result.trajectories],
        inserted_scenes,
        config,
        t_insert,
    )
    modes["final_insertion_live"] = [
        (
            episode.scene,
            inserted_scene,
            _slice_rollout(live_result, idx),
        )
        for idx, (episode, inserted_scene) in enumerate(
            zip(eval_result.trajectories, inserted_scenes, strict=True)
        )
    ]

    eval_rows = []
    episode_rows = []
    for mode, mode_items in modes.items():
        rows = [
            _insertion_episode_row(
                mode=mode,
                step=step,
                ckpt_name=ckpt_name,
                episode_idx=idx,
                original_scene=original_scene,
                inserted_scene=inserted_scene,
                result=result,
                system=framework.system,
                config=config,
                t_insert=t_insert,
            )
            for idx, (original_scene, inserted_scene, result) in enumerate(mode_items)
        ]
        episode_rows.extend(rows)
        eval_rows.append(
            _insertion_eval_row(
                mode=mode,
                step=step,
                ckpt_name=ckpt_name,
                pool=eval_result.pool,
                episode_rows=rows,
                config=config,
            )
        )
    return eval_rows, episode_rows


def _insert_scene_obstacle(
    scene: Scene,
    states: torch.Tensor,
    system: Any,
    t_insert: int,
    radius: float,
) -> Scene:
    centers = np.asarray(scene.obstacle_centers, dtype=np.float64).copy()
    radii = np.asarray(scene.obstacle_radii, dtype=np.float64).copy()
    active = np.asarray(scene.obstacle_active, dtype=np.bool_).copy()
    positions = system.position(states).detach().cpu().numpy()
    insert_step = min(max(0, t_insert), positions.shape[0] - 1)
    # Insert in the obstacle coordinate space: the first centers.shape[-1] position coords (xy footprint for
    # infinite vertical cylinders; a no-op when position dim == center dim, e.g. DI/unicycle/planar).
    d = centers.shape[-1]
    center = (0.5 * (positions[0] + positions[insert_step]))[:d]

    centers = np.concatenate([centers, center.reshape(1, d)], axis=0)
    radii = np.concatenate([radii, np.asarray([radius], dtype=np.float64)], axis=0)
    active = np.concatenate([active, np.asarray([True], dtype=np.bool_)], axis=0)
    return replace(
        scene,
        obstacle_centers=centers,
        obstacle_radii=radii,
        obstacle_active=active,
    )


def _rollout_live_insertion(
    framework: Any,
    original_scenes: list[Scene],
    inserted_scenes: list[Scene],
    config: Mapping[str, Any],
    t_insert: int,
) -> RolloutResult:
    value_net = getattr(framework, "value_net", None)
    if value_net is None:
        dtype = framework.system.u_bounds.dtype
        device = framework.system.u_bounds.device
    else:
        parameter = next(value_net.parameters())
        dtype = parameter.dtype
        device = parameter.device

    original_batch = batch_scenes(original_scenes, device=device, dtype=dtype)
    inserted_batch = batch_scenes(inserted_scenes, device=device, dtype=dtype)
    x = framework.system.wrap_state(initial_states_from_batch(original_batch))
    max_steps = int(config["eval"]["max_steps"])
    dt = float(config["env"]["dt"])

    states = [x]
    u_nom_steps = []
    u_safe_steps = []
    infeasible_steps = []
    physical_done = _insertion_physical_done(
        framework.system,
        original_batch,
        inserted_batch,
        torch.stack(states, dim=0),
        config,
        t_insert,
    )
    empty_steps: list[torch.Tensor] = []
    singular_steps: list[torch.Tensor] = []
    _ff = getattr(framework, "_filter", None)                        # v2.7.1 S1d: split-logging source
    for step in range(max_steps):
        scene = original_batch if step < t_insert else inserted_batch
        u_nom = framework.policy(x, scene)
        filtered = framework.filter(x, u_nom, scene)
        u_safe = filtered[0]
        infeasible = filtered[1].to(device=x.device, dtype=torch.bool)
        _le = getattr(_ff, "last_empty", None)
        empty_b = (_le.to(device=x.device, dtype=torch.bool) if _le is not None
                   else infeasible.clone())
        _ls = getattr(_ff, "last_singular", None)
        singular_b = (_ls.to(device=x.device, dtype=torch.bool) if _ls is not None
                      else torch.zeros_like(infeasible))
        done_action = physical_done.unsqueeze(1)
        u_nom = torch.where(done_action, torch.zeros_like(u_nom), u_nom)
        u_safe = torch.where(done_action, torch.zeros_like(u_safe), u_safe)
        infeasible = torch.where(
            physical_done,
            torch.zeros_like(infeasible, dtype=torch.bool),
            infeasible,
        )
        x_next = rk4_step(framework.system, x, u_safe, dt)
        x = torch.where(done_action, x, x_next)
        states.append(x)
        u_nom_steps.append(u_nom)
        u_safe_steps.append(u_safe)
        infeasible_steps.append(infeasible)
        empty_steps.append(torch.where(physical_done, torch.zeros_like(empty_b), empty_b))
        singular_steps.append(torch.where(physical_done, torch.zeros_like(singular_b), singular_b))
        physical_done = physical_done | _insertion_physical_done(
            framework.system,
            original_batch,
            inserted_batch,
            torch.stack(states, dim=0),
            config,
            t_insert,
        )

    u_nom_tensor = torch.stack(u_nom_steps, dim=0)
    u_safe_tensor = torch.stack(u_safe_steps, dim=0)
    return RolloutResult(
        states=torch.stack(states, dim=0),
        u_nom=u_nom_tensor,
        u_safe=u_safe_tensor,
        intervention_mask=torch.linalg.norm(u_safe_tensor - u_nom_tensor, dim=-1) > 1.0e-3,
        infeasible=torch.stack(infeasible_steps, dim=0),
        empty=torch.stack(empty_steps, dim=0),
        singular=torch.stack(singular_steps, dim=0),
    )


def _insertion_physical_done(
    system: Any,
    original_scene: Any,
    inserted_scene: Any,
    states: torch.Tensor,
    config: Mapping[str, Any],
    t_insert: int,
) -> torch.Tensor:
    masks = _step_outcomes_with_insertion(
        states,
        original_scene,
        inserted_scene,
        system,
        config,
        t_insert,
    )
    return masks.collided[-1] | masks.goal_reached[-1] | masks.oob[-1]


def _states_only_result(states: torch.Tensor, action_dim: int) -> RolloutResult:
    n_steps = states.shape[0] - 1
    actions = states.new_zeros((n_steps, states.shape[1], action_dim))
    bools = torch.zeros((n_steps, states.shape[1]), dtype=torch.bool, device=states.device)
    return RolloutResult(
        states=states,
        u_nom=actions,
        u_safe=actions.clone(),
        intervention_mask=bools.clone(),
        infeasible=bools.clone(),
    )


def _lqr_result_from_states(states: torch.Tensor, scene: Scene, system: Any) -> RolloutResult:
    step_states = states[:-1, 0, :]
    goal = torch.as_tensor(scene.goal, dtype=states.dtype, device=states.device)
    goals = goal.unsqueeze(0).expand(step_states.shape[0], -1)
    actions = system.lqr_action(step_states, goals).unsqueeze(1)
    bools = torch.zeros(
        (actions.shape[0], 1),
        dtype=torch.bool,
        device=states.device,
    )
    return RolloutResult(
        states=states,
        u_nom=actions,
        u_safe=actions.clone(),
        intervention_mask=bools.clone(),
        infeasible=bools.clone(),
    )


def _slice_rollout(result: RolloutResult, batch_index: int) -> RolloutResult:
    _bi = slice(batch_index, batch_index + 1)
    return RolloutResult(
        states=result.states[:, batch_index : batch_index + 1, :],
        u_nom=result.u_nom[:, batch_index : batch_index + 1, :],
        u_safe=result.u_safe[:, batch_index : batch_index + 1, :],
        intervention_mask=result.intervention_mask[:, batch_index : batch_index + 1],
        infeasible=result.infeasible[:, batch_index : batch_index + 1],
        empty=None if result.empty is None else result.empty[:, _bi],
        singular=None if result.singular is None else result.singular[:, _bi],
    )


def _insertion_episode_row(
    *,
    mode: str,
    step: int,
    ckpt_name: str,
    episode_idx: int,
    original_scene: Scene,
    inserted_scene: Scene,
    result: RolloutResult,
    system: Any,
    config: Mapping[str, Any],
    t_insert: int,
) -> dict[str, Any]:
    masks = _step_outcomes_with_insertion(
        result.states,
        original_scene,
        inserted_scene,
        system,
        config,
        t_insert,
    )
    resolved = resolve_outcome(masks)
    outcome = resolved.outcome[0]
    event_step = int(resolved.event_step[0].item())
    physical_event_step = int(first_physical_event_step(masks)[0].item())
    min_window_displacement = float(
        eval_min_window_displacement(masks, physical_event_step)[0]
        .detach()
        .cpu()
        .item()
    )
    reach = 1.0 if outcome == "goal" else 0.0
    collision = 1.0 if outcome == "collision" else 0.0
    oob = 1.0 if outcome == "oob" else 0.0
    stuck = 1.0 if outcome == "stuck" else 0.0
    timeout = 1.0 if outcome == "timeout" else 0.0
    active_steps = active_action_steps(physical_event_step, result.u_safe.shape[0])
    infeasible = active_bool_fraction(result.infeasible, active_steps)
    empty_frac = active_bool_fraction(result.empty, active_steps) if result.empty is not None else infeasible
    singular_frac = active_bool_fraction(result.singular, active_steps) if result.singular is not None else 0.0
    saturation = saturation_step_fraction(result, system, active_steps=active_steps)
    projection = torch.linalg.norm(result.u_safe - result.u_nom, dim=-1)
    positions = system.position(result.states)
    h_values = _signed_h_with_insertion(
        positions,
        original_scene,
        inserted_scene,
        float(config["env"]["h_scale"]),
        t_insert,
    )
    path_delta = positions[1:] - positions[:-1]
    if outcome == "goal" and 0 <= event_step < result.states.shape[0]:
        angrate_at_reach = float(system.angular_rate(result.states[event_step])[0].item())
    else:
        angrate_at_reach = float("nan")
    return {
        "mode": mode,
        "step": int(step),
        "ckpt_name": ckpt_name,
        "episode_idx": int(episode_idx),
        "outcome": outcome,
        "n_steps": int(event_step if event_step >= 0 else result.u_safe.shape[0]),
        "cps_episode": reach
        - 2.0 * collision
        - stuck
        - 0.5 * (oob + timeout)
        - 0.3 * infeasible,
        "reach": reach,
        "collision": collision,
        "oob": oob,
        "stuck": stuck,
        "timeout": timeout,
        "infeasible_step_frac": infeasible,
        "empty_step_frac": empty_frac,
        "singular_step_frac": singular_frac,
        "saturation_step_frac": saturation,
        "min_window_displacement": min_window_displacement,
        "mean_proj_mag": float(projection.mean().item()) if projection.numel() else 0.0,
        "max_h": float(h_values.max().item()),
        "traj_path_len": float(torch.linalg.norm(path_delta, dim=-1).sum().item()),
        "angrate_at_reach": angrate_at_reach,
        "collision_cause": (resolved.collision_cause[0] if resolved.collision_cause else ""),
        # v2.8.0 schema parity: the insertion path always runs band-terminating, so a band crossing is the
        # terminal collision (0/1 with its step); the floor-permissive readout is exercised via evaluate().
        "band_crossings": (1 if (resolved.collision_cause and resolved.collision_cause[0] in ("band_lower", "band_upper")) else 0),
        "first_crossing_step": (int(event_step) if (resolved.collision_cause and resolved.collision_cause[0] in ("band_lower", "band_upper")) else -1),
    }


def _step_outcomes_with_insertion(
    states: torch.Tensor,
    original_scene: Scene,
    inserted_scene: Scene,
    system: Any,
    config: Mapping[str, Any],
    t_insert: int,
) -> StepOutcomeMasks:
    original = step_outcomes(states, original_scene, system, config)
    inserted = step_outcomes(states, inserted_scene, system, config)
    time_mask = (
        torch.arange(states.shape[0], device=states.device).unsqueeze(1) >= t_insert
    )
    return StepOutcomeMasks(
        collided=torch.where(time_mask, inserted.collided, original.collided),
        goal_reached=torch.where(time_mask, inserted.goal_reached, original.goal_reached),
        oob=torch.where(time_mask, inserted.oob, original.oob),
        stuck=inserted.stuck,
        window_displacement=inserted.window_displacement,
        # v2.8.0: mux the collision cause channels the same way (band surfaces are insertion-invariant;
        # obstacle changes because the inserted obstacle can add contacts after t_insert).
        collided_obstacle=torch.where(time_mask, inserted.collided_obstacle, original.collided_obstacle),
        collided_band_lower=torch.where(time_mask, inserted.collided_band_lower, original.collided_band_lower),
        collided_band_upper=torch.where(time_mask, inserted.collided_band_upper, original.collided_band_upper),
    )


def _signed_h_with_insertion(
    positions: torch.Tensor,
    original_scene: Scene,
    inserted_scene: Scene,
    h_scale: float,
    t_insert: int,
) -> torch.Tensor:
    original = signed_h(positions, original_scene, h_scale)
    inserted = signed_h(positions, inserted_scene, h_scale)
    time_mask = (
        torch.arange(positions.shape[0], device=positions.device).unsqueeze(1) >= t_insert
    )
    return torch.where(time_mask, inserted, original)


def _insertion_eval_row(
    *,
    mode: str,
    step: int,
    ckpt_name: str,
    pool: Any,
    episode_rows: list[dict[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    bootstrap_cfg = config["eval"]["bootstrap"]
    ci_result = within_seed_ci(
        episode_rows,
        n_resample=int(bootstrap_cfg["n_resample"]),
        seed=int(bootstrap_cfg["seed"]),
    )
    mean = ci_result["mean"]
    ci = ci_result["ci"]
    saturation_rate = float(
        np.mean([float(row["saturation_step_frac"]) for row in episode_rows])
    )
    stuck_bins = stuck_bin_fractions(episode_rows)
    return {
        "mode": mode,
        "step": int(step),
        "ckpt_name": ckpt_name,
        "pool_name": pool.name,
        "pool_seed": int(pool.seed),
        "n_scenes": int(len(episode_rows)),
        "cps": mean["cps"],
        "reach": mean["reach"],
        "collision": mean["collision"],
        "oob": mean["oob"],
        "stuck": mean["stuck"],
        "timeout": mean["timeout"],
        "infeasibility": mean["infeasibility"],
        "saturation_rate": saturation_rate,
        **stuck_bins,
        "alpha_safe": float(config["filter"]["alpha_safe"]),
        "alpha_unsafe": float(config["filter"]["alpha_unsafe"]),
        "cps_ci_lo": ci["cps"]["lo"],
        "cps_ci_hi": ci["cps"]["hi"],
        "reach_ci_lo": ci["reach"]["lo"],
        "reach_ci_hi": ci["reach"]["hi"],
        "collision_ci_lo": ci["collision"]["lo"],
        "collision_ci_hi": ci["collision"]["hi"],
        "stuck_ci_lo": ci["stuck"]["lo"],
        "stuck_ci_hi": ci["stuck"]["hi"],
        "infeasibility_ci_lo": ci["infeasibility"]["lo"],
        "infeasibility_ci_hi": ci["infeasibility"]["hi"],
    }


def _append_csv(path: Path, columns: list[str], rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        return
    _ensure_csv_columns(path, columns)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=columns, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _write_tb_image(tensorboard_dir: Path, tag: str, path: Path, step: int) -> None:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError:
        return
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    try:
        log_png_to_tensorboard(writer, tag, path, step)
    finally:
        writer.close()


def _ensure_csv_columns(path: Path, columns: list[str]) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open(newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        existing_columns = reader.fieldnames or []
        if existing_columns == columns:
            return
        rows = list(reader)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _initialize_eval_dir_if_needed(
    run_dir: Path,
    config: Mapping[str, Any],
    checkpoint_path: Path,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    (run_dir / "tensorboard").mkdir(exist_ok=True)
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        config_path.write_text(
            yaml.safe_dump(_plain_data(config), sort_keys=False),
            encoding="utf-8",
        )
    git_path = run_dir / "git_commit.txt"
    if not git_path.exists():
        git_path.write_text(_git_commit_text() + "\n", encoding="utf-8")
    source_manifest = checkpoint_path.parent.parent / "pool_manifest.json"
    target_manifest = run_dir / "pool_manifest.json"
    if source_manifest.exists() and not target_manifest.exists():
        shutil.copy2(source_manifest, target_manifest)


def _plain_data(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain_data(item) for item in value]
    if isinstance(value, tuple):
        return [_plain_data(item) for item in value]
    return value


def _git_commit_text() -> str:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
        dirty = subprocess.run(
            ["git", "diff", "--quiet"],
            cwd=REPO_ROOT,
            check=False,
        ).returncode != 0
    except Exception:
        return "unknown"
    return f"{commit} DIRTY" if dirty else commit


def _write_status_done(
    run_dir: Path,
    checkpoint: Mapping[str, Any],
    eval_result: EvaluationResult,
) -> None:
    status = {
        "stage": "full",
        "phase": "done",
        "current_step": int(checkpoint["step"]),
        "best_step": int(checkpoint.get("best_step", checkpoint["step"])),
        "best_cps": float(checkpoint.get("best_cps", eval_result.eval_row["cps"])),
        "halt_reason": None,
        "updated_at": _now_iso(),
    }
    (run_dir / "status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _rewrite_report(run_dir: Path) -> None:
    from src.frameworks.oc_pncbf.train import _write_report

    git_commit = (run_dir / "git_commit.txt").read_text(encoding="utf-8").strip()
    _write_report(
        run_dir,
        run_id=run_dir.name,
        git_commit=git_commit,
        wallclock_s=0.0,
        halt_reason=None,
    )


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run OC-PNCBF final evaluation.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--last", action="store_true")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--eval-only-tag", type=str, default=None)
    parser.add_argument("--lookahead-enabled", action="store_true")
    parser.add_argument("--lookahead-n", type=int, default=None)
    parser.add_argument("--lookahead-beta", type=float, default=None)
    parser.add_argument("--lookahead-delta", type=float, default=None)
    parser.add_argument("--skip-insertion", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir
    if run_dir is None:
        if args.ckpt is None or args.eval_only_tag is None:
            parser.error("--run-dir is required unless --ckpt and --eval-only-tag are provided.")
        run_dir = _fresh_eval_only_dir(args.output_root, args.eval_only_tag)
    config_overrides = _lookahead_config_overrides(
        enabled=args.lookahead_enabled,
        n=args.lookahead_n,
        beta=args.lookahead_beta,
        delta=args.lookahead_delta,
    )
    result = run_full_eval(
        run_dir,
        ckpt_path=args.ckpt,
        use_last=args.last,
        max_scenes=args.max_scenes,
        config_overrides=config_overrides,
        include_insertion=not args.skip_insertion,
    )
    for path in result.figure_paths:
        print(path)
    print(f"filter_works_count={result.filter_works_count}")
    print(f"intervention_episode_count={result.intervention_episode_count}")
    print(
        "plotted_intervention_episodes="
        f"{result.plotted_episode_count}/{EPISODES_PER_REPORT}"
    )
    return 0


def _lookahead_config_overrides(
    *,
    enabled: bool,
    n: int | None,
    beta: float | None,
    delta: float | None,
) -> dict[str, Any]:
    if not enabled and n is None and beta is None and delta is None:
        return {}
    lookahead: dict[str, Any] = {"enabled": bool(enabled)}
    if n is not None:
        lookahead["N"] = int(n)
    if beta is not None:
        lookahead["beta"] = float(beta)
    if delta is not None:
        lookahead["delta"] = float(delta)
    return {"filter": {"lookahead": lookahead}}


def _fresh_eval_only_dir(output_root: Path, tag: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = output_root / f"{__version__}__{timestamp}__{tag}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"{base.name}_{suffix}"
        suffix += 1
    return candidate


if __name__ == "__main__":
    raise SystemExit(main())
