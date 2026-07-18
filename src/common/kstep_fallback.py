"""v2.7.1 Stage-1 — shared k-step empty-branch fallback (eval-only default-off).

Two-phase piecewise-constant candidate selection over a fixed control grid G: for k steps split into
phase lengths ceil(k/2) + floor(k/2), enumerate all (u1, u2) in G x G, roll the plant, and pick the pair
minimising V_hat(x_k) - V_hat(x0). Returns the FIRST-phase control u1* of the argmin (receding-horizon: the
filter applies u1* this step and re-plans next call, no carried state). Deterministic fixed-order tie-break
(a outer, b inner; strict-less update keeps the first). Runs under torch.no_grad. Imported by both the HardNet
filter (empty branch) and `residual_mechanism_diag.py` P-B' (no fork).
"""
from __future__ import annotations

from typing import Any

import torch

from src.common.rk4 import rk4_step

Tensor = torch.Tensor


def grid_controls(system: Any, device: Any, dtype: Any = torch.float32) -> Tensor:
    """Per-system two-phase control grid from the box U (|G| <= 16).
    - quadrotor_planar: box corners + hover (mg,0) + max-thrust±max|tau| pair (Stage-1/1b grid, UNCHANGED).
    - generic (DI / unicycle / any box): box corners + center + zero (if in box) + per-axis extremes
      (min/max on axis i, other axes at center). Deterministic (sorted unique)."""
    from itertools import product
    b = system.u_bounds.to(device, dtype)                                 # [adim, 2]
    adim = int(b.shape[0])
    lo = [float(b[i, 0]) for i in range(adim)]; hi = [float(b[i, 1]) for i in range(adim)]
    mid = [(lo[i] + hi[i]) / 2.0 for i in range(adim)]
    if getattr(system, "name", "") == "quadrotor_planar":
        mg = 9.81
        cand = [(lo[0], lo[1]), (lo[0], hi[1]), (hi[0], lo[1]), (hi[0], hi[1]),
                (mg, 0.0), (hi[0], hi[1]), (hi[0], lo[1])]
    else:
        cand = [tuple(hi[i] if bit else lo[i] for i, bit in enumerate(bits))
                for bits in product((0, 1), repeat=adim)]                 # 2^adim box corners
        cand.append(tuple(mid))                                           # center
        if all(lo[i] <= 0.0 <= hi[i] for i in range(adim)):
            cand.append(tuple(0.0 for _ in range(adim)))                  # zero (if inside the box)
        for i in range(adim):                                            # per-axis extremes, others at center
            for side in (lo[i], hi[i]):
                c = list(mid); c[i] = side; cand.append(tuple(c))
    return torch.tensor(sorted(set(cand)), dtype=dtype, device=device)    # [k, adim]


def kstep_select(x: Tensor, scene: Any, h_fn, system: Any, G: Tensor, k: int, dt: float):
    """Return (u1_star [n, adim], margin [n]) — the first-phase control of the argmin two-phase sequence and its
    V_hat(x_k)-V_hat(x0) margin. Deterministic (a-outer/b-inner fixed order, strict-less update keeps the first)."""
    ng = G.shape[0]; k1 = (k + 1) // 2; k2 = k - k1; n = x.shape[0]
    with torch.no_grad():
        v0 = h_fn(x, scene).reshape(-1)
        best_m = torch.full((n,), float("inf"), device=x.device, dtype=v0.dtype)
        best_a = torch.zeros(n, dtype=torch.long, device=x.device)
        for a in range(ng):
            u1 = G[a].unsqueeze(0).expand(n, -1)
            for b in range(ng):
                u2 = G[b].unsqueeze(0).expand(n, -1)
                xx = x
                for _ in range(k1):
                    xx = rk4_step(system, xx, u1, dt)
                for _ in range(k2):
                    xx = rk4_step(system, xx, u2, dt)
                m = h_fn(xx, scene).reshape(-1) - v0
                upd = m < best_m                                          # strict => first (a,b) wins ties
                best_m = torch.where(upd, m, best_m)
                best_a = torch.where(upd, torch.full_like(best_a, a), best_a)
        return G[best_a], best_m


def slice_scene(scene: Any, mask: Tensor) -> Any:
    """Row-subset a BatchedScene by a boolean mask: slice every batched Tensor field (first dim == batch),
    keep scalars (system/mode) and None. Preserves type."""
    import dataclasses
    n = int(mask.shape[0])
    upd = {}
    for f in dataclasses.fields(scene):
        v = getattr(scene, f.name)
        if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == n:
            upd[f.name] = v[mask]
    return dataclasses.replace(scene, **upd)
