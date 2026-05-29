from __future__ import annotations

from typing import Any

import torch


Tensor = torch.Tensor


def scene_obstacle_tensors(
    scene: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor]:
    centers = torch.as_tensor(scene.obstacle_centers, dtype=dtype, device=device)
    radii = torch.as_tensor(scene.obstacle_radii, dtype=dtype, device=device)
    active = torch.as_tensor(scene.obstacle_active, dtype=torch.bool, device=device)
    return centers, radii, active


def scene_goal_tensor(scene: Any, x: Tensor) -> Tensor:
    goal = torch.as_tensor(scene.goal, dtype=x.dtype, device=x.device)
    if goal.ndim == 1:
        return goal.unsqueeze(0).expand(x.shape[0], -1)
    if goal.ndim == 2 and goal.shape[0] == x.shape[0]:
        return goal
    raise ValueError(f"Goal shape {tuple(goal.shape)} is incompatible with batch {x.shape[0]}.")


def top_k_obstacles(
    positions: Tensor,
    centers: Tensor,
    radii: Tensor,
    active: Tensor,
    k: int,
) -> tuple[Tensor, Tensor]:
    batch_size = positions.shape[0]
    if centers.ndim == 2:
        centers_batch = centers.unsqueeze(0).expand(batch_size, -1, -1)
        radii_batch = radii.unsqueeze(0).expand(batch_size, -1)
        active_batch = active.unsqueeze(0).expand(batch_size, -1)
    elif centers.ndim == 3 and centers.shape[0] == batch_size:
        centers_batch = centers
        radii_batch = radii
        active_batch = active
    else:
        raise ValueError(
            f"Obstacle center shape {tuple(centers.shape)} is incompatible "
            f"with batch {batch_size}."
        )

    rel = centers_batch - positions.unsqueeze(1)
    surface_distance = torch.linalg.norm(rel, dim=-1) - radii_batch
    surface_distance = surface_distance.masked_fill(~active_batch, torch.inf)

    n_obstacles = centers_batch.shape[1]
    k_select = min(k, n_obstacles)
    sorted_indices = torch.argsort(surface_distance, dim=1, stable=True)
    indices = sorted_indices[:, :k_select]
    distances = torch.gather(surface_distance, dim=1, index=indices)

    gather_rel = indices.unsqueeze(-1).expand(-1, -1, 2)
    top_rel = torch.gather(rel, dim=1, index=gather_rel)
    top_radii = torch.gather(radii_batch, 1, indices)

    valid = torch.isfinite(distances)
    top_rel = top_rel.masked_fill(~valid.unsqueeze(-1), 0.0)
    top_radii = top_radii.masked_fill(~valid, 0.0)

    if k_select == k:
        return top_rel, top_radii

    pad_count = k - k_select
    rel_pad = torch.zeros(
        positions.shape[0],
        pad_count,
        2,
        dtype=positions.dtype,
        device=positions.device,
    )
    radii_pad = torch.zeros(
        positions.shape[0],
        pad_count,
        dtype=positions.dtype,
        device=positions.device,
    )
    return torch.cat([top_rel, rel_pad], dim=1), torch.cat(
        [top_radii, radii_pad],
        dim=1,
    )
