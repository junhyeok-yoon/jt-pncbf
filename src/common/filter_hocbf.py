"""Analytic higher-order CBF (HOCBF) safety filter for the double integrator.

Relative-degree-2 construction per obstacle i (point p, velocity v, center c_i, radius r_i):
  h0_i   = ||p - c_i||^2 - r_safe_i^2                  (>0 safe; r_safe = r + margin)
  psi0_i = h0_i
  psi1_i = d/dt psi0_i + alpha1 psi0_i = 2 (p-c_i)·v + alpha1 h0_i
  enforce psi1_i_dot + alpha2 psi1_i >= 0, with
    psi1_i_dot = 2||v||^2 + 2 (p-c_i)·u + 2 alpha1 (p-c_i)·v
  => 2 (p-c_i)·u >= -[ 2||v||^2 + 2 alpha1 (p-c_i)·v + alpha2 psi1_i ]
  => row in the project A·u <= b convention:
       A_i = -2 (p-c_i),  b_i = 2||v||^2 + 2 alpha1 (p-c_i)·v + alpha2 psi1_i
Forward invariance for the DI holds with class-K alpha1, alpha2 > 0 given psi0_i(0)>=0,
psi1_i(0)>=0 (standard HOCBF; here linear class-K). One row per nearest-K active obstacle.

The safe action solves min ||u - u_nom||^2 s.t. all HOCBF rows AND |u_j| <= u_max, as an EXACT
2-variable QP by KKT active-set candidate enumeration (0/1/2 active constraints). Infeasible
(rows ∩ box empty) returns the least-violating box-feasible action and flags infeasible.

Only this filter is new; it plugs into the same rollout interface as the deployed HardNet filter
(filter_fn(x, scene, u_nom) -> (u_safe, infeasible)).
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Mapping

import torch

Tensor = torch.Tensor
_DET_TOL = 1e-12
_FEAS_TOL = 1e-7


class HOCBFFilter:
    def __init__(self, system: Any, config: Mapping[str, Any], alpha1: float, alpha2: float,
                 r_margin: float = 0.05, k_obs: int = 5) -> None:
        self.system = system
        self.alpha1 = float(alpha1)
        self.alpha2 = float(alpha2)
        self.r_margin = float(r_margin)
        self.k_obs = int(k_obs)
        self.u_max = float(config["env"]["bounds"]["double_integrator"]["u_max"])

    def __call__(self, x: Tensor, scene: Any, u_nom: Tensor) -> tuple[Tensor, Tensor]:
        if x.ndim != 2 or u_nom.ndim != 2:
            raise ValueError("x and u_nom must be batched rank-2 tensors.")
        device, dtype = u_nom.device, u_nom.dtype
        p = x[:, :2]
        v = x[:, 2:4]
        centers = torch.as_tensor(scene.obstacle_centers, device=device, dtype=dtype)
        radii = torch.as_tensor(scene.obstacle_radii, device=device, dtype=dtype)
        active = torch.as_tensor(scene.obstacle_active, device=device, dtype=torch.bool)
        if centers.ndim == 2:
            centers = centers.unsqueeze(0).expand(p.shape[0], -1, -1)
            radii = radii.unsqueeze(0).expand(p.shape[0], -1)
            active = active.unsqueeze(0).expand(p.shape[0], -1)

        rel = p.unsqueeze(1) - centers                       # (p - c) [B, nmax, 2]
        surf = torch.linalg.norm(rel, dim=2) - radii
        surf = surf.masked_fill(~active, float("inf"))
        k = min(self.k_obs, surf.shape[1])
        sel = torch.topk(surf, k, dim=1, largest=False).indices   # nearest k by surface dist
        b_idx = torch.arange(p.shape[0], device=device).unsqueeze(1)
        rel_k = rel[b_idx, sel]                               # [B,k,2]
        radii_k = radii[b_idx, sel]                           # [B,k]
        active_k = active[b_idx, sel]                         # [B,k]

        r_safe = radii_k + self.r_margin
        h0 = torch.sum(rel_k * rel_k, dim=2) - r_safe * r_safe          # [B,k]
        relv = torch.sum(rel_k * v.unsqueeze(1), dim=2)                  # (p-c)·v [B,k]
        v2 = torch.sum(v * v, dim=1, keepdim=True)                       # ||v||^2 [B,1]
        psi1 = 2.0 * relv + self.alpha1 * h0
        a_obs = -2.0 * rel_k                                             # [B,k,2]
        b_obs = 2.0 * v2 + 2.0 * self.alpha1 * relv + self.alpha2 * psi1  # [B,k]
        # padded/inactive selected obstacles: never binding (A=0, b=+inf)
        a_obs = torch.where(active_k.unsqueeze(2), a_obs, torch.zeros_like(a_obs))
        b_obs = torch.where(active_k, b_obs, torch.full_like(b_obs, float("inf")))

        a_box = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], device=device, dtype=dtype)
        a_box = a_box.unsqueeze(0).expand(p.shape[0], -1, -1)
        b_box = torch.full((p.shape[0], 4), self.u_max, device=device, dtype=dtype)
        A = torch.cat([a_obs, a_box], dim=1)                            # [B, k+4, 2]
        b = torch.cat([b_obs, b_box], dim=1)                            # [B, k+4]
        return _qp2d(u_nom, A, b, self.u_max)


def _qp2d(u_nom: Tensor, A: Tensor, b: Tensor, u_max: float) -> tuple[Tensor, Tensor]:
    """Exact min ||u-u_nom||^2 s.t. A u <= b (rows include the box). 2-var QP by enumeration."""
    B, m, _ = A.shape
    device, dtype = u_nom.device, u_nom.dtype
    cand = [u_nom, torch.clamp(u_nom, -u_max, u_max)]                   # 0-active + box fallback
    # single-active: project u_nom onto each line A_j·u = b_j
    nj2 = torch.sum(A * A, dim=2)                                       # [B,m]
    Au_nom = torch.einsum("bmd,bd->bm", A, u_nom)
    for j in range(m):
        coef = (Au_nom[:, j] - b[:, j]) / torch.where(nj2[:, j] > _DET_TOL, nj2[:, j], torch.ones_like(nj2[:, j]))
        pj = u_nom - A[:, j, :] * coef.unsqueeze(1)
        pj = torch.where((nj2[:, j] > _DET_TOL).unsqueeze(1), pj, torch.full_like(pj, float("nan")))
        cand.append(pj)
    # two-active: intersection vertices
    for j, k in combinations(range(m), 2):
        a0, a1 = A[:, j, :], A[:, k, :]
        det = a0[:, 0] * a1[:, 1] - a0[:, 1] * a1[:, 0]
        ok = torch.abs(det) > _DET_TOL
        detb = torch.where(ok, det, torch.ones_like(det))
        ux = (b[:, j] * a1[:, 1] - b[:, k] * a0[:, 1]) / detb
        uy = (a0[:, 0] * b[:, k] - a1[:, 0] * b[:, j]) / detb
        vtx = torch.stack([ux, uy], dim=1)
        vtx = torch.where(ok.unsqueeze(1), vtx, torch.full_like(vtx, float("nan")))
        cand.append(vtx)
    C = torch.stack(cand, dim=1)                                        # [B, K, 2]
    finite = torch.isfinite(C).all(dim=2)
    Au = torch.einsum("bmd,bkd->bkm", A, C)                             # [B,K,m]
    feas = (Au <= b.unsqueeze(1) + _FEAS_TOL).all(dim=2) & finite
    dist = torch.sum((C - u_nom.unsqueeze(1)) ** 2, dim=2)
    inf = torch.full_like(dist, float("inf"))
    score = torch.where(feas, dist, inf)
    any_feas = feas.any(dim=1)
    # least-violating fallback among box-feasible finite candidates
    box_ok = (torch.abs(C[:, :, 0]) <= u_max + 1e-6) & (torch.abs(C[:, :, 1]) <= u_max + 1e-6) & finite
    viol = torch.sum(torch.clamp(Au - b.unsqueeze(1), min=0.0), dim=2)
    lv_score = torch.where(box_ok, viol, inf)
    idx_feas = torch.argmin(score, dim=1)
    idx_lv = torch.argmin(lv_score, dim=1)
    idx = torch.where(any_feas, idx_feas, idx_lv)
    u = C[torch.arange(B, device=device), idx]
    u = torch.nan_to_num(u, nan=0.0)
    u = torch.clamp(u, -u_max, u_max)
    return u, ~any_feas
