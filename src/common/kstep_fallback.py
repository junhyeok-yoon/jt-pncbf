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


def _tile_scene(scene: Any, reps: int, n: int) -> Any:
    """Repeat every batched field (first dim == n) reps times, row i -> rows [i*reps : (i+1)*reps]
    (repeat_interleave), matching the candidate-major state tiling below."""
    import dataclasses
    upd = {}
    for f in dataclasses.fields(scene):
        v = getattr(scene, f.name)
        if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == n:
            upd[f.name] = v.repeat_interleave(reps, dim=0)
    return dataclasses.replace(scene, **upd)


def kstep_select(x: Tensor, scene: Any, h_fn, system: Any, G: Tensor, k: int, dt: float, phases: int = 2):
    """Return (u1_star [n, adim], margin [n]) — the first-phase control of the argmin candidate sequence and its
    V_hat(x_k)-V_hat(x0) margin. VECTORIZED (v2.7.6 M8.0): one batched k-step rollout over all candidates instead
    of a Python double loop; the deterministic tie-break is preserved exactly (candidates are laid out in the same
    a-outer/b-inner row-major order the loop used, and torch.argmin returns the first minimiser = the strict-less
    'keep first' rule). phases=2 (default, unchanged): two-phase (u1 for ceil(k/2), u2 for floor(k/2)) over G x G
    (ng^2 candidates). phases=1 (M8.1): one control over all k steps (ng candidates)."""
    ng = int(G.shape[0]); n = int(x.shape[0]); adim = int(G.shape[1])
    with torch.no_grad():
        v0 = h_fn(x, scene).reshape(-1)                                   # [n]
        if phases == 1:
            ncand = ng; k1, k2 = k, 0
            U1 = G                                                        # [ncand, adim]
            U2 = None
        else:
            ncand = ng * ng; k1 = (k + 1) // 2; k2 = k - k1
            cc = torch.arange(ncand, device=x.device)
            U1 = G[cc // ng]; U2 = G[cc % ng]                            # a-outer / b-inner, c = a*ng + b
        X = x.unsqueeze(1).expand(n, ncand, x.shape[1]).reshape(n * ncand, x.shape[1])
        U1f = U1.unsqueeze(0).expand(n, ncand, adim).reshape(n * ncand, adim)
        sc = _tile_scene(scene, ncand, n)
        for _ in range(k1):
            X = rk4_step(system, X, U1f, dt)
        if phases == 2 and k2 > 0:
            U2f = U2.unsqueeze(0).expand(n, ncand, adim).reshape(n * ncand, adim)
            for _ in range(k2):
                X = rk4_step(system, X, U2f, dt)
        m = (h_fn(X, sc).reshape(-1) - v0.repeat_interleave(ncand)).reshape(n, ncand)
        best_c = torch.argmin(m, dim=1)                                   # first minimiser = 'keep first' tie-break
        best_m = m.gather(1, best_c.unsqueeze(1)).reshape(-1)
        u1_star = G[best_c] if phases == 1 else G[best_c // ng]           # phase-1 control of the argmin
        return u1_star, best_m


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
