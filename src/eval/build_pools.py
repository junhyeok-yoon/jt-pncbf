from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import yaml

from src.envs.scene_init import Scene
from src.envs.scene_init_eval import sample_eval_scene
from src.envs.scene_init_fixed import sample_eval_fixed_scene


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/secured_data/pools"
BASE_CONFIG_PATH = REPO_ROOT / "src/configs/base_config.yaml"
POOL_FORMAT_VERSION = 1


@dataclass(frozen=True)
class PoolSpec:
    name: str
    n_scenes: int
    seed: int


@dataclass(frozen=True)
class EvaluationPool:
    name: str
    system: str
    n_scenes: int
    seed: int
    scenes: list[Scene]


@dataclass(frozen=True)
class PoolArtifacts:
    pool_path: Path
    manifest_path: Path
    sha256: str


def default_pool_specs(config: Mapping[str, Any]) -> list[PoolSpec]:
    return [
        PoolSpec(
            name="inloop",
            n_scenes=int(config["eval"]["in_loop"]["n"]),
            seed=int(config["eval"]["in_loop"]["seed"]),
        ),
        PoolSpec(
            name="full",
            n_scenes=int(config["eval"]["full"]["n"]),
            seed=int(config["eval"]["full"]["seed"]),
        ),
    ]


def build_pool(
    config: Mapping[str, Any],
    system: str,
    spec: PoolSpec,
) -> EvaluationPool:
    rng = np.random.default_rng(seed=spec.seed)
    sampler = _eval_sampler(config)
    scenes = [
        sampler(rng, config, system)
        for _ in range(spec.n_scenes)
    ]
    return EvaluationPool(
        name=spec.name,
        system=system,
        n_scenes=spec.n_scenes,
        seed=spec.seed,
        scenes=scenes,
    )


def write_pool(
    pool: EvaluationPool,
    config: Mapping[str, Any],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    created_at: str | None = None,
    git_commit: str | None = None,
) -> PoolArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    obstacle_distribution = obstacle_distribution_name(config)
    stem = pool_stem(
        pool.name,
        pool.system,
        pool.n_scenes,
        pool.seed,
        obstacle_distribution,
    )
    pool_path = output_dir / f"{stem}.pkl"
    manifest_path = output_dir / f"{stem}.manifest.json"

    with pool_path.open("wb") as pool_file:
        pickle.dump(_pool_payload(pool), pool_file, protocol=pickle.HIGHEST_PROTOCOL)

    sha = sha256_file(pool_path)
    manifest = {
        "n_scenes": pool.n_scenes,
        "system": pool.system,
        "seed": pool.seed,
        "pool_format_version": POOL_FORMAT_VERSION,
        "obstacle_distribution": obstacle_distribution,
        "sampler_params": sampler_param_snapshot(config, pool.system),
        "pool_sha256": sha,
        "created_at": created_at or _now_iso(),
        "git_commit": git_commit if git_commit is not None else git_head_or_unknown(),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return PoolArtifacts(
        pool_path=pool_path,
        manifest_path=manifest_path,
        sha256=sha,
    )


def build_default_pools(
    config: Mapping[str, Any] | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    system: str = "double_integrator",
) -> list[PoolArtifacts]:
    if config is None:
        config = load_base_config()

    artifacts = []
    git_commit = git_head_or_unknown()
    for spec in default_pool_specs(config):
        pool = build_pool(config, system, spec)
        artifacts.append(
            write_pool(
                pool,
                config,
                output_dir=output_dir,
                git_commit=git_commit,
            )
        )
    return artifacts


def load_pool(path: Path) -> EvaluationPool:
    with path.open("rb") as pool_file:
        payload = pickle.load(pool_file)
    if isinstance(payload, EvaluationPool):
        return payload
    if not isinstance(payload, Mapping):
        raise TypeError(f"Unexpected pool object in {path}: {type(payload)!r}")
    if "scenes" in payload:
        return EvaluationPool(
            name=str(payload["name"]),
            system=str(payload["system"]),
            n_scenes=int(payload["n_scenes"]),
            seed=int(payload["seed"]),
            scenes=list(payload["scenes"]),
        )

    return EvaluationPool(
        name=_pool_name_from_path(path),
        system=str(payload["system"]),
        n_scenes=int(payload["n_scenes"]),
        seed=int(payload["seed"]),
        scenes=_scenes_from_payload(payload),
    )


def pool_stem(
    pool_name: str,
    system: str,
    n_scenes: int,
    seed: int,
    obstacle_distribution: str = "random",
) -> str:
    distribution = _normalize_obstacle_distribution(obstacle_distribution)
    suffix = "_fixed" if distribution == "fixed_centered" else ""
    return f"eval_{pool_name}_{_system_tag(system)}{suffix}_n{n_scenes}_seed{seed}"


def sampler_param_snapshot(
    config: Mapping[str, Any],
    system: str,
) -> dict[str, Any]:
    return {
        "obstacle_distribution": obstacle_distribution_name(config),
        "obstacle": _plain_data(config["obstacle"]),
        "eval_scene": _plain_data(config["eval"]["scene"]),
        "env": {
            "world_lim": float(config["env"]["world_lim"]),
            "v_init_max": float(config["env"]["v_init_max"]),
            "bounds": _plain_data(config["env"]["bounds"][system]),
        },
        "fixed_obstacle": _fixed_obstacle_snapshot(config),
        "scene_train": {
            "init_feasibility_margin": float(
                config["scene_train"]["init_feasibility_margin"]
            ),
        },
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pool_payload(pool: EvaluationPool) -> dict[str, Any]:
    obstacle_centers = np.stack([scene.obstacle_centers for scene in pool.scenes])
    obstacle_radii = np.stack([scene.obstacle_radii for scene in pool.scenes])
    obstacle_active = np.stack([scene.obstacle_active for scene in pool.scenes])
    start = np.stack([scene.start for scene in pool.scenes])
    goal = np.stack([scene.goal for scene in pool.scenes])
    init_velocity = _stack_init_velocity(pool.scenes)
    return {
        "pool_format_version": POOL_FORMAT_VERSION,
        "system": pool.system,
        "seed": pool.seed,
        "n_scenes": pool.n_scenes,
        "obstacle_centers": obstacle_centers,
        "obstacle_radii": obstacle_radii,
        "obstacle_active": obstacle_active,
        "start": start,
        "goal": goal,
        "init_velocity": init_velocity,
    }


def _stack_init_velocity(scenes: list[Scene]) -> np.ndarray:
    values = []
    for scene in scenes:
        if scene.initial_velocity is not None:
            values.append(scene.initial_velocity)
        else:
            if scene.initial_speed is None or scene.initial_heading is None:
                raise ValueError("Scene is missing initial velocity representation.")
            values.append(
                np.array(
                    [scene.initial_speed, scene.initial_heading],
                    dtype=np.float64,
                )
            )
    return np.stack(values)


def _scenes_from_payload(payload: Mapping[str, Any]) -> list[Scene]:
    system = str(payload["system"])
    n_scenes = int(payload["n_scenes"])
    init_velocity = np.asarray(payload["init_velocity"], dtype=np.float64)
    scenes = []
    for idx in range(n_scenes):
        if system == "double_integrator":
            initial_velocity = init_velocity[idx]
            initial_speed = None
            initial_heading = None
        else:
            initial_velocity = None
            initial_speed = float(init_velocity[idx, 0])
            initial_heading = float(init_velocity[idx, 1])
        scenes.append(
            Scene(
                obstacle_centers=np.asarray(payload["obstacle_centers"][idx]),
                obstacle_radii=np.asarray(payload["obstacle_radii"][idx]),
                obstacle_active=np.asarray(payload["obstacle_active"][idx]),
                start=np.asarray(payload["start"][idx]),
                goal=np.asarray(payload["goal"][idx]),
                system=system,
                mode="eval",
                initial_velocity=initial_velocity,
                initial_speed=initial_speed,
                initial_heading=initial_heading,
            )
        )
    return scenes


def _pool_name_from_path(path: Path) -> str:
    parts = path.stem.split("_")
    if len(parts) >= 2 and parts[0] == "eval":
        return parts[1]
    return "unknown"


def load_base_config() -> Mapping[str, Any]:
    return yaml.safe_load(BASE_CONFIG_PATH.read_text(encoding="utf-8"))


def git_head_or_unknown() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(
            "Warning: git HEAD unavailable; recording unknown in pool manifest.",
            file=sys.stderr,
        )
        return "unknown"
    return result.stdout.strip()


def _system_tag(system: str) -> str:
    if system == "double_integrator":
        return "di"
    return system.replace("_", "-")


def obstacle_distribution_name(config: Mapping[str, Any]) -> str:
    env_cfg = config.get("env", {})
    if not isinstance(env_cfg, Mapping):
        raise TypeError("config['env'] must be a mapping.")
    return _normalize_obstacle_distribution(
        str(env_cfg.get("obstacle_distribution", "random"))
    )


def _eval_sampler(config: Mapping[str, Any]) -> Any:
    distribution = obstacle_distribution_name(config)
    if distribution == "random":
        return sample_eval_scene
    if distribution == "fixed_centered":
        return sample_eval_fixed_scene
    raise ValueError(f"Unsupported obstacle distribution: {distribution!r}")


def _normalize_obstacle_distribution(value: str) -> str:
    if value in {"random", "fixed_centered"}:
        return value
    raise ValueError(f"Unsupported obstacle distribution: {value!r}")


def _fixed_obstacle_snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "center": [0.0, 0.0],
        "radius": float(config["obstacle"]["r_max"]),
        "active_count": 1,
    }


def _plain_data(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain_data(item) for item in value]
    if isinstance(value, tuple):
        return [_plain_data(item) for item in value]
    return value


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic eval pools.")
    parser.add_argument("--system", default="double_integrator")
    parser.add_argument(
        "--obstacle-distribution",
        choices=["random", "fixed_centered"],
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    args = parser.parse_args()

    config = load_base_config()
    if args.obstacle_distribution is not None:
        config["env"]["obstacle_distribution"] = args.obstacle_distribution
    artifacts = build_default_pools(
        config=config,
        output_dir=args.output_dir,
        system=args.system,
    )
    for artifact in artifacts:
        print(artifact.pool_path)
        print(artifact.manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
