from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.outcomes import resolve_outcome, step_outcomes  # noqa: E402
from src.envs.double_integrator import DoubleIntegrator  # noqa: E402
from src.envs.scene_init import Scene, sample_train_scene  # noqa: E402
from src.eval.plotting import TrajectorySpec, plot_scene_grid  # noqa: E402
from src.frameworks.oc_pncbf.collection import (  # noqa: E402
    OCReplayBuffer,
    collect,
)


OUTPUT_PATH = REPO_ROOT / "data/verification/oc_collection_di_A.png"
N_SCENES = 16
SEED = 202605


def main() -> int:
    config = _load_config()
    system = DoubleIntegrator(config)
    rng = np.random.default_rng(seed=SEED)
    buffer = OCReplayBuffer(capacity=N_SCENES * int(config["training"]["oc_pncbf"]["horizon"]))

    def sampler(local_rng: np.random.Generator) -> Scene:
        return sample_train_scene(local_rng, config, system.name)

    collect(
        system,
        sampler,
        rng,
        n_episodes=N_SCENES,
        max_steps=int(config["training"]["oc_pncbf"]["horizon"]),
        dt=float(config["env"]["dt"]),
        buffer=buffer,
        h_scale=float(config["env"]["h_scale"]),
    )

    trajectories = []
    outcomes = []
    event_steps = []
    scenes = []
    for record in buffer.traj_view[:N_SCENES]:
        states = record.states.unsqueeze(1)
        masks = step_outcomes(states, record.scene, system, config)
        resolved = resolve_outcome(masks)
        scenes.append(record.scene)
        trajectories.append([TrajectorySpec(states=record.states)])
        outcomes.append(resolved.outcome[0])
        event_steps.append(int(resolved.event_step[0]))

    plot_scene_grid(
        scenes=scenes,
        output_path=OUTPUT_PATH,
        config=config,
        role="OC collection",
        system_name=system.name,
        letter="A",
        start_index=0,
        trajectories=trajectories,
        outcomes=outcomes,
        event_steps=event_steps,
        draw_final_velocity=True,
    )
    if not OUTPUT_PATH.exists():
        raise FileNotFoundError(f"Missing output PNG: {OUTPUT_PATH}")
    print(OUTPUT_PATH)
    return 0


def _load_config() -> Mapping[str, Any]:
    base = yaml.safe_load((REPO_ROOT / "src/configs/base_config.yaml").read_text())
    exp = yaml.safe_load((REPO_ROOT / "src/configs/exp_config.yaml").read_text())
    return _deep_merge(base, exp)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


if __name__ == "__main__":
    raise SystemExit(main())
