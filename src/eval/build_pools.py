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
    variant: str = "",
) -> PoolArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    obstacle_distribution = obstacle_distribution_name(config)
    stem = pool_stem(
        pool.name,
        pool.system,
        pool.n_scenes,
        pool.seed,
        obstacle_distribution,
        variant,
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
    variant: str = "",
) -> list[PoolArtifacts]:
    if config is None:
        config = load_base_config()
    # v2.7.3 M0c: default the naming variant from config so build and selection cannot disagree; an explicit
    # `variant` arg (CLI) overrides for naming only. The scene spec itself is read from config (per_system).
    if not variant:
        variant = pool_variant(config, system)

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
                variant=variant,
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
    variant: str = "",
) -> str:
    distribution = _normalize_obstacle_distribution(obstacle_distribution)
    suffix = "_fixed" if distribution == "fixed_centered" else ""
    tag = _system_tag(system)
    if variant:                                          # v2.7.3: scene-variant tag (e.g. "d2"); empty = today's name
        tag = f"{tag}-{variant}"
    return f"eval_{pool_name}_{tag}{suffix}_n{n_scenes}_seed{seed}"


def pool_variant(config: Mapping[str, Any], system: str) -> str:
    """Scene-variant tag for `system`, sourced from `obstacle.per_system[system].variant` (empty if absent).
    Build and selection both read this so they cannot disagree (v2.7.3 M0c)."""
    return str(
        config.get("obstacle", {}).get("per_system", {}).get(system, {}).get("variant", "")
    )


EVAL_POOLS_DIR = REPO_ROOT / "data/eval_pools"


def resolve_pool_or_raise(stem: str) -> Path:
    """v2.7.4: resolve a pool `.pkl` by stem, preferring the secured pools dir (frozen d2 / v2.7.2 / DI pools)
    and falling back to the writable `data/eval_pools` — the v2.7.4 `-d2r` pools live there because
    `data/secured_data/` is write-forbidden this version. SHA-verifies whichever is found; secured is always
    tried first so frozen-pool resolution is unchanged."""
    for base in (DEFAULT_OUTPUT_DIR, EVAL_POOLS_DIR):
        candidate = base / f"{stem}.pkl"
        if candidate.exists():
            return verify_pool_or_raise(candidate)
    raise FileNotFoundError(
        f"Pool {stem}.pkl not found in {DEFAULT_OUTPUT_DIR} or {EVAL_POOLS_DIR}"
    )


def verify_pool_or_raise(pool_path: Path) -> Path:
    """Assert the resolved pool `.pkl` exists AND its SHA-256 matches its manifest (v2.7.3 M0c hard gate —
    silently loading the wrong/absent pool is the failure this prevents). Returns the path on success."""
    if not pool_path.exists():
        raise FileNotFoundError(f"Resolved eval pool is absent: {pool_path}")
    manifest_path = pool_path.with_suffix(".manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Pool manifest is absent: {manifest_path}")
    recorded = str(json.loads(manifest_path.read_text(encoding="utf-8")).get("pool_sha256", ""))
    actual = sha256_file(pool_path)
    if recorded != actual:
        raise ValueError(
            f"Pool SHA-256 mismatch for {pool_path.name}: manifest {recorded[:12]}… vs disk {actual[:12]}…"
        )
    return pool_path


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
    payload = {
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
    if pool.scenes[0].system == "quadrotor_planar":
        # quadrotor_planar carries attitude + angular rate in addition to the 2D linear velocity.
        payload["init_attitude"] = np.array(
            [float(scene.initial_attitude) for scene in pool.scenes], dtype=np.float64
        )
        payload["init_omega"] = np.array(
            [float(scene.initial_omega) for scene in pool.scenes], dtype=np.float64
        )
    if pool.scenes[0].system == "quadrotor_3d":
        # quadrotor_3d: start/goal already 3D (stacked above); 3D init_velocity; full attitude + rates.
        payload["init_attitude_quat"] = np.stack([s.initial_attitude_quat for s in pool.scenes])
        payload["init_omega_vec"] = np.stack([s.initial_omega_vec for s in pool.scenes])
    return payload


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
    init_attitude = payload.get("init_attitude")
    init_omega = payload.get("init_omega")
    scenes = []
    for idx in range(n_scenes):
        initial_attitude = None
        initial_omega = None
        if system == "double_integrator":
            initial_velocity = init_velocity[idx]
            initial_speed = None
            initial_heading = None
        elif system == "quadrotor_planar":
            initial_velocity = init_velocity[idx]
            initial_speed = None
            initial_heading = None
            initial_attitude = float(init_attitude[idx])
            initial_omega = float(init_omega[idx])
        elif system == "quadrotor_3d":
            initial_velocity = init_velocity[idx]
            initial_speed = None
            initial_heading = None
        else:
            initial_velocity = None
            initial_speed = float(init_velocity[idx, 0])
            initial_heading = float(init_velocity[idx, 1])
        initial_attitude_quat = None
        initial_omega_vec = None
        if system == "quadrotor_3d":
            initial_attitude_quat = np.asarray(payload["init_attitude_quat"][idx], dtype=np.float64)
            initial_omega_vec = np.asarray(payload["init_omega_vec"][idx], dtype=np.float64)
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
                initial_attitude=initial_attitude,
                initial_omega=initial_omega,
                initial_attitude_quat=initial_attitude_quat,
                initial_omega_vec=initial_omega_vec,
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
    parser.add_argument(
        "--variant",
        default="",
        help="v2.7.3 scene-variant tag for naming (e.g. d2); empty reproduces today's filenames. "
        "The scene spec itself lives in config (obstacle.per_system); this names the output only.",
    )
    args = parser.parse_args()

    config = load_base_config()
    if args.obstacle_distribution is not None:
        config["env"]["obstacle_distribution"] = args.obstacle_distribution
    artifacts = build_default_pools(
        config=config,
        output_dir=args.output_dir,
        system=args.system,
        variant=args.variant,
    )
    for artifact in artifacts:
        print(artifact.pool_path)
        print(artifact.manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
