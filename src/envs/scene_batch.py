from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from src.envs.scene_init import Scene


Tensor = torch.Tensor


@dataclass(frozen=True)
class BatchedScene:
    obstacle_centers: Tensor
    obstacle_radii: Tensor
    obstacle_active: Tensor
    start: Tensor
    goal: Tensor
    system: str
    mode: str
    initial_velocity: Tensor | None = None
    initial_speed: Tensor | None = None
    initial_heading: Tensor | None = None
    initial_attitude: Tensor | None = None
    initial_omega: Tensor | None = None
    initial_attitude_quat: Tensor | None = None   # quadrotor_3d: [B,4] body->world quaternion
    initial_omega_vec: Tensor | None = None        # quadrotor_3d: [B,3] body rates
    # v2.9.2 item A (EVAL-ONLY, default None => every existing path is byte-identical): the obstacle
    # block of the observation frozen at its step-zero value, [B, 4*k_obs]. Carried on the SCENE and
    # not on the system so that it row-subsets correctly wherever the scene does -- in particular
    # `kstep_fallback.slice_scene`, which the registered cell's empty-branch fallback calls on a
    # boolean mask of the batch. A cache held on the system would desynchronize there.
    frozen_obs_block: Tensor | None = None


def batch_scenes(
    scenes: Sequence[Scene],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> BatchedScene:
    if not scenes:
        raise ValueError("scenes must be non-empty.")
    system = scenes[0].system
    if any(scene.system != system for scene in scenes):
        raise ValueError("Cannot batch scenes from different systems.")
    mode = scenes[0].mode

    obstacle_centers = torch.as_tensor(
        np.stack([scene.obstacle_centers for scene in scenes]),
        dtype=dtype,
        device=device,
    )
    obstacle_radii = torch.as_tensor(
        np.stack([scene.obstacle_radii for scene in scenes]),
        dtype=dtype,
        device=device,
    )
    obstacle_active = torch.as_tensor(
        np.stack([scene.obstacle_active for scene in scenes]),
        dtype=torch.bool,
        device=device,
    )
    start = torch.as_tensor(
        np.stack([scene.start for scene in scenes]),
        dtype=dtype,
        device=device,
    )
    goal = torch.as_tensor(
        np.stack([scene.goal for scene in scenes]),
        dtype=dtype,
        device=device,
    )

    initial_speed = None
    initial_heading = None
    initial_attitude = None
    initial_omega = None
    initial_attitude_quat = None
    initial_omega_vec = None
    if scenes[0].initial_velocity is not None:
        initial_velocity = torch.as_tensor(
            np.stack([scene.initial_velocity for scene in scenes]),
            dtype=dtype,
            device=device,
        )
        if system == "quadrotor_planar":
            initial_attitude = torch.as_tensor(
                [float(scene.initial_attitude) for scene in scenes],
                dtype=dtype,
                device=device,
            )
            initial_omega = torch.as_tensor(
                [float(scene.initial_omega) for scene in scenes],
                dtype=dtype,
                device=device,
            )
        elif system == "quadrotor_3d":
            # start/goal are already full 3D (stacked above); attach the full attitude quaternion + rates.
            initial_attitude_quat = torch.as_tensor(
                np.stack([scene.initial_attitude_quat for scene in scenes]),
                dtype=dtype,
                device=device,
            )
            initial_omega_vec = torch.as_tensor(
                np.stack([scene.initial_omega_vec for scene in scenes]),
                dtype=dtype,
                device=device,
            )
    else:
        initial_velocity = None
        initial_speed = torch.as_tensor(
            [float(scene.initial_speed) for scene in scenes],
            dtype=dtype,
            device=device,
        )
        initial_heading = torch.as_tensor(
            [float(scene.initial_heading) for scene in scenes],
            dtype=dtype,
            device=device,
        )

    return BatchedScene(
        obstacle_centers=obstacle_centers,
        obstacle_radii=obstacle_radii,
        obstacle_active=obstacle_active,
        start=start,
        goal=goal,
        system=system,
        mode=mode,
        initial_velocity=initial_velocity,
        initial_speed=initial_speed,
        initial_heading=initial_heading,
        initial_attitude=initial_attitude,
        initial_omega=initial_omega,
        initial_attitude_quat=initial_attitude_quat,
        initial_omega_vec=initial_omega_vec,
    )


def initial_states_from_batch(scene: BatchedScene) -> Tensor:
    if scene.system == "quadrotor_3d":
        # [px,py,pz, qw,qx,qy,qz, vx,vy,vz, wx,wy,wz]  (13D)
        return torch.cat(
            [scene.start, scene.initial_attitude_quat, scene.initial_velocity, scene.initial_omega_vec],
            dim=1,
        )
    if scene.system == "quadrotor_planar":
        # [px, py, theta, vx, vy, omega]
        return torch.stack(
            [
                scene.start[:, 0],
                scene.start[:, 1],
                scene.initial_attitude,
                scene.initial_velocity[:, 0],
                scene.initial_velocity[:, 1],
                scene.initial_omega,
            ],
            dim=1,
        )
    if scene.initial_velocity is not None:
        return torch.cat([scene.start, scene.initial_velocity], dim=1)
    if scene.initial_speed is None or scene.initial_heading is None:
        raise ValueError("Batched scene is missing initial speed or heading.")
    return torch.stack(
        [
            scene.start[:, 0],
            scene.start[:, 1],
            scene.initial_heading,
            scene.initial_speed,
        ],
        dim=1,
    )
