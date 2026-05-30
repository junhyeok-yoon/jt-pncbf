from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from src._version import __version__
from src.common.filter_hardnet import HardNetFilter
from src.eval.build_pools import obstacle_distribution_name, pool_stem
from src.eval.evaluate import EVAL_EPISODE_COLUMNS, EVAL_METRIC_COLUMNS, evaluate
from src.eval.run_full import write_cbf_contour_figure, write_trajectory_figures
from src.frameworks.oc_pncbf.train import load_framework_from_checkpoint


REPO_ROOT = Path(__file__).resolve().parents[2]
POOL_DIR = REPO_ROOT / "data/secured_data/pools"
DEFAULT_CKPT = (
    REPO_ROOT
    / "data/secured_data/oc_pncbf/v2.0.0__20260529-001441__seed42/checkpoints/best.pt"
)


@dataclass(frozen=True)
class FilterSummary:
    cps: float
    reach: float
    collision: float
    oob: float
    stuck: float
    timeout: float
    infeasibility: float
    saturation_rate: float
    intervention_step_frac: float
    filter_works_count: int


class HardNetDeployFramework:
    def __init__(self, base_framework: Any, config: Mapping[str, Any]) -> None:
        self.base = base_framework
        self.system = base_framework.system
        self.value_net = base_framework.value_net
        self.value_net.requires_grad_(False)
        self._filter = HardNetFilter(self.system, base_framework._h_fn, config)

    def policy(self, x: torch.Tensor, scene: Any) -> torch.Tensor:
        return self.base.policy(x, scene)

    def filter(
        self,
        x: torch.Tensor,
        u_nom: torch.Tensor,
        scene: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        u_safe, infeasible = self._filter(x.detach(), scene, u_nom.detach())
        return u_safe.detach(), infeasible.detach()

    def value(self, x: torch.Tensor, scene: Any) -> torch.Tensor:
        return self.base.value(x, scene)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the adopted OC-PNCBF checkpoint with HardNet deployment.",
    )
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--eval-batch-size", type=int, default=200)
    args = parser.parse_args()

    output_dir = _fresh_output_dir(args.output_root)
    output_dir.mkdir(parents=True)
    (output_dir / "figures").mkdir()
    (output_dir / "figures/inloop").mkdir(parents=True)

    hardnet_base, config, checkpoint = load_framework_from_checkpoint(args.ckpt)
    hardnet = HardNetDeployFramework(hardnet_base, config)
    cbf_qp, _, _ = load_framework_from_checkpoint(args.ckpt)

    _write_csv_header(output_dir / "eval_metrics.csv", EVAL_METRIC_COLUMNS)
    _write_csv_header(output_dir / "eval_episodes.csv", EVAL_EPISODE_COLUMNS)
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    hardnet_results = {}
    for pool_name, mode, fig_dir in (
        ("inloop", "hardnet_in_loop", output_dir / "figures/inloop"),
        ("full", "hardnet_final", output_dir / "figures"),
    ):
        pool_path = _pool_path(pool_name, config)
        result = evaluate(
            hardnet,
            pool_path,
            config,
            mode=mode,
            step=int(checkpoint["step"]),
            ckpt_name=args.ckpt.name,
            include_lqr_baseline=True,
            eval_batch_size=args.eval_batch_size,
        )
        hardnet_results[pool_name] = result
        _append_csv(output_dir / "eval_metrics.csv", EVAL_METRIC_COLUMNS, [result.eval_row])
        _append_csv(output_dir / "eval_episodes.csv", EVAL_EPISODE_COLUMNS, result.episode_rows)
        write_trajectory_figures(
            run_dir=output_dir,
            eval_result=result,
            config=config,
            system_name=hardnet.system.name,
            u_bounds=hardnet.system.u_bounds,
            role="HardNet deploy eval",
            output_dir=fig_dir,
            filename_template="trajectory_grid_{letter}.png",
        )
        write_cbf_contour_figure(
            eval_result=result,
            config=config,
            system=hardnet.system,
            value_net=hardnet.value_net,
            output_path=fig_dir / "cbf_contour.png",
            role="HardNet deploy CBF contour",
        )

    cbf_full = evaluate(
        cbf_qp,
        _pool_path("full", config),
        config,
        mode="cbf_qp_reference_final",
        step=int(checkpoint["step"]),
        ckpt_name=args.ckpt.name,
        include_lqr_baseline=True,
        eval_batch_size=args.eval_batch_size,
    )
    hardnet_full = hardnet_results["full"]
    hardnet_summary = _summary(hardnet_full)
    cbf_summary = _summary(cbf_full)
    divergences = _divergences(cbf_full, hardnet_full)

    comparison = {
        "output_dir": str(output_dir),
        "checkpoint": str(args.ckpt),
        "checkpoint_step": int(checkpoint["step"]),
        "version": __version__,
        "halfspace": {
            "A": "L_g h",
            "b": "-L_f h - alpha*h",
            "alpha": "alpha_safe if h <= 0 else alpha_unsafe",
            "h_fn": "make_h_fn(ValueNetEnsemble, system) deployed value",
        },
        "hardnet_feasibility_definition": (
            "HardNet infeasible is singular ||L_g h|| < 5e-4 or empty "
            "half-space/box intersection. It is not CBF-QP slack activity."
        ),
        "summaries": {
            "cbf_qp": cbf_summary.__dict__,
            "hardnet": hardnet_summary.__dict__,
        },
        "divergences": divergences,
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(output_dir)
    print("FILTER,cps,reach,collision,oob,stuck,timeout,infeasibility,saturation_rate,intervention_step_frac,filter_works_count")
    print(_summary_csv("CBF-QP", cbf_summary))
    print(_summary_csv("HardNet", hardnet_summary))
    print(f"divergence_count={len(divergences)}")
    for item in divergences[:20]:
        print(json.dumps(item, sort_keys=True))
    return 0


def _fresh_output_dir(output_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = output_root / f"{__version__}__{timestamp}__hardnet_oc_seed42"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"{base.name}_{suffix}"
        suffix += 1
    return candidate


def _pool_path(pool_name: str, config: Mapping[str, Any]) -> Path:
    if pool_name == "inloop":
        n_scenes = int(config["eval"]["in_loop"]["n"])
        seed = int(config["eval"]["in_loop"]["seed"])
    elif pool_name == "full":
        n_scenes = int(config["eval"]["full"]["n"])
        seed = int(config["eval"]["full"]["seed"])
    else:
        raise ValueError(f"Unknown pool: {pool_name!r}")
    system_name = str(config["run"]["system"])
    return POOL_DIR / (
        f"{pool_stem(pool_name, system_name, n_scenes, seed, obstacle_distribution_name(config))}.pkl"
    )


def _summary(result: Any) -> FilterSummary:
    row = result.eval_row
    return FilterSummary(
        cps=float(row["cps"]),
        reach=float(row["reach"]),
        collision=float(row["collision"]),
        oob=float(row["oob"]),
        stuck=float(row["stuck"]),
        timeout=float(row["timeout"]),
        infeasibility=float(row["infeasibility"]),
        saturation_rate=float(row["saturation_rate"]),
        intervention_step_frac=_intervention_step_fraction(result),
        filter_works_count=_filter_works_count(result),
    )


def _intervention_step_fraction(result: Any) -> float:
    active = 0
    intervened = 0
    for episode, row in zip(result.trajectories, result.episode_rows, strict=True):
        n_steps = int(row["n_steps"])
        mask = episode.filtered.intervention_mask[:n_steps, 0]
        active += int(mask.numel())
        intervened += int(mask.sum().item())
    return 0.0 if active == 0 else intervened / active


def _filter_works_count(result: Any) -> int:
    return sum(
        1
        for trajectory in result.trajectories[:32]
        if trajectory.lqr_outcome == "collision"
        and trajectory.filtered_outcome != "collision"
    )


def _divergences(cbf_result: Any, hardnet_result: Any) -> list[dict[str, Any]]:
    items = []
    for idx, (cbf_ep, hard_ep, cbf_row, hard_row) in enumerate(
        zip(
            cbf_result.trajectories,
            hardnet_result.trajectories,
            cbf_result.episode_rows,
            hardnet_result.episode_rows,
            strict=True,
        )
    ):
        if cbf_ep.filtered_outcome == hard_ep.filtered_outcome:
            continue
        items.append(
            {
                "episode_idx": idx,
                "cbf_qp_outcome": cbf_ep.filtered_outcome,
                "hardnet_outcome": hard_ep.filtered_outcome,
                "lqr_outcome": hard_ep.lqr_outcome,
                "cbf_qp_infeasible_step_frac": float(cbf_row["infeasible_step_frac"]),
                "hardnet_infeasible_step_frac": float(hard_row["infeasible_step_frac"]),
                "cbf_qp_saturation_step_frac": float(cbf_row["saturation_step_frac"]),
                "hardnet_saturation_step_frac": float(hard_row["saturation_step_frac"]),
                "cbf_qp_mean_proj_mag": float(cbf_row["mean_proj_mag"]),
                "hardnet_mean_proj_mag": float(hard_row["mean_proj_mag"]),
            }
        )
    return items


def _summary_csv(label: str, summary: FilterSummary) -> str:
    values = [
        label,
        summary.cps,
        summary.reach,
        summary.collision,
        summary.oob,
        summary.stuck,
        summary.timeout,
        summary.infeasibility,
        summary.saturation_rate,
        summary.intervention_step_frac,
        summary.filter_works_count,
    ]
    return ",".join(str(value) for value in values)


def _write_csv_header(path: Path, columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        csv.DictWriter(file_obj, fieldnames=columns).writeheader()


def _append_csv(
    path: Path,
    columns: list[str],
    rows: list[Mapping[str, Any]],
) -> None:
    with path.open("a", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=columns, extrasaction="ignore")
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


if __name__ == "__main__":
    raise SystemExit(main())
