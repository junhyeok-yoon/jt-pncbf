from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from matplotlib import image as mpimg
import numpy as np
import torch
import yaml

from src.envs.double_integrator import DoubleIntegrator
from src.envs.scene_init import Scene
from src.envs.unicycle import Unicycle
from src.eval.plotting import plot_cbf_contours


REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeContourValue(torch.nn.Module):
    def deployed_h(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.shape[1] == 19:
            raw = obs[:, 2] / 4.0
        elif obs.shape[1] == 18:
            raw = obs[:, 1] / 4.0
        else:
            raise ValueError(f"Unexpected observation dim: {obs.shape[1]}.")
        return torch.clamp(raw, -1.0, 1.0)

    def target_h(self, obs: torch.Tensor) -> torch.Tensor:
        return self.deployed_h(obs)


def test_cbf_contour_renders_for_both_systems(tmp_path: Path) -> None:
    config = _load_config()
    value_net = FakeContourValue()
    cases = [
        (
            DoubleIntegrator(config),
            [_scene("double_integrator", 0), _scene("double_integrator", 1)],
        ),
        (Unicycle(config), [_scene("unicycle", 0), _scene("unicycle", 1)]),
    ]

    for system, scenes in cases:
        output_path = tmp_path / f"{system.name}_cbf_contour.png"
        result = plot_cbf_contours(
            scenes=scenes,
            output_path=output_path,
            config=config,
            system=system,
            value_net=value_net,
            resolution=36,
        )

        assert output_path.exists()
        assert -1.0 <= result.h_min <= result.h_max <= 1.0
        assert result.zero_contour_panels > 0
        assert result.velocity_arrow_count == 4
        assert result.zero_velocity_dot_count == 2
        image = mpimg.imread(output_path)
        assert image.ndim == 3
        black_pixels = np.all(image[..., :3] < 0.05, axis=-1)
        assert int(black_pixels.sum()) > 100


def _scene(system: str, offset: int) -> Scene:
    centers, radii, active = _obstacles()
    centers[0] = np.array([0.5 + offset, 1.0], dtype=np.float64)
    centers[1] = np.array([-1.5, -1.0 - offset], dtype=np.float64)
    radii[:2] = np.array([0.6, 0.45], dtype=np.float64)
    active[:2] = True
    kwargs: dict[str, Any] = {}
    if system == "double_integrator":
        kwargs["initial_velocity"] = np.zeros(2, dtype=np.float64)
    else:
        kwargs["initial_speed"] = 0.0
        kwargs["initial_heading"] = 0.0
    return Scene(
        obstacle_centers=centers,
        obstacle_radii=radii,
        obstacle_active=active,
        start=np.array([-3.0, 0.0], dtype=np.float64),
        goal=np.array([0.0, 0.0], dtype=np.float64),
        system=system,
        mode="unit",
        **kwargs,
    )


def _obstacles() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_max = int(_load_config()["obstacle"]["n_max"])
    return (
        np.zeros((n_max, 2), dtype=np.float64),
        np.zeros(n_max, dtype=np.float64),
        np.zeros(n_max, dtype=np.bool_),
    )


def _load_config() -> dict[str, Any]:
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
