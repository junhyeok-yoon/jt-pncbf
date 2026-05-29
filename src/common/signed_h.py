from __future__ import annotations

from typing import Any

import torch


Tensor = torch.Tensor


def signed_h(p: Tensor, scene: Any, h_scale: float) -> Tensor:
    if h_scale <= 0.0:
        raise ValueError(f"h_scale must be positive, got {h_scale}.")

    centers = torch.as_tensor(scene.obstacle_centers, dtype=p.dtype, device=p.device)
    radii = torch.as_tensor(scene.obstacle_radii, dtype=p.dtype, device=p.device)
    active = torch.as_tensor(scene.obstacle_active, dtype=torch.bool, device=p.device)

    rel = p.unsqueeze(-2) - centers
    distance = torch.linalg.norm(rel, dim=-1)
    clearance = distance - radii
    h_all = 1.0 - 2.0 * torch.clamp(clearance / h_scale, min=0.0, max=1.0)
    h_all = torch.where(active, h_all, torch.full_like(h_all, -1.0))
    h_value = torch.max(h_all, dim=-1).values
    return torch.clamp(h_value, min=-1.0, max=1.0)
