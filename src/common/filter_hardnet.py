from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Mapping

import torch

from src.common.rk4 import rk4_step
from src.common.system import System


Tensor = torch.Tensor

_SINGULAR_LG_THRESHOLD = 5.0e-4
_FEASIBILITY_TOL = 1.0e-9
_DENOM_TOL = 1.0e-12


@dataclass(frozen=True)
class _HardNetParams:
    epsilon: float
    lg_reg_eps: float
    box_aware: bool
    alpha_safe: float
    alpha_unsafe: float
    lookahead_enabled: bool
    lookahead_n: int
    lookahead_beta: float
    lookahead_delta: float
    dt: float
    empty_fallback_mode: str = "none"       # v2.7.1 Stage-1: none | kstep (eval-only; none = bit-parity)
    empty_fallback_k: int = 10
    empty_fallback_phases: int = 2          # v2.7.6 M8.1: 2 = two-phase (default, unchanged) | 1 = single-phase
    projection: str = "dual_solve"          # v2.8.0 S1: dual_solve (exact, prop:lambda-solve) | enumerate (legacy)
    empty_mode: str = "argmin"              # v2.8.2 M1 (prop:empty-prox): argmin (v2.8.0 discrete least-violating,
                                            # DEFAULT) | prox (continuous softmin over the SAME candidates)
    empty_prox_temp: float | None = None    # v2.8.2: prox continuity scale gamma; NO default (S5-derived; required iff prox)
    gamma_margin: float = 0.0               # v2.8.2: ROW-ONLY safety retreat (02_control §6.1). b = -L_f h - alpha*(h+gamma).
                                            # DEFAULT 0.0 => byte-identical to the pre-v2.8.2 row (guarded in _row_upper).
                                            # Deliberately NOT installed at h_fn level: h+gamma would move _base_alpha's
                                            # `h <= 0` region test, flipping alpha 2.0 -> 100.0 on h in (-gamma, 0].


class HardNetFilter:
    def __init__(
        self,
        system: System,
        h_fn: Callable[[Tensor, Any], Tensor],
        config: Mapping[str, Any],
        *,
        policy_fn: Callable[[Tensor, Any], Tensor] | None = None,
    ) -> None:
        self.system = system
        self.h_fn = h_fn
        self.policy_fn = policy_fn
        self.params = _hardnet_params(config)

    def __call__(
        self,
        x: Tensor,
        scene: Any,
        u_nom: Tensor,
        detach_coeffs: bool = False,
        return_deficit_aux: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor]:
        if x.ndim != 2 or u_nom.ndim != 2:
            raise ValueError("x and u_nom must be batched rank-2 tensors.")
        if x.shape[0] != u_nom.shape[0]:
            raise ValueError("x and u_nom batch sizes must match.")

        h, lf_h, lg_h = _cbf_terms(
            self.system,
            self.h_fn,
            x,
            scene,
            u_nom,
            create_graph=True,
        )
        _lg_live, _h_live, _lf_live = lg_h, h, lf_h   # v2.8.2 M2: pre-detach LIVE coeffs (box-aware stash below);
        #                                               survives detach_filter_coeffs so s_inf keeps its policy grad
        if detach_coeffs:
            # v2.4.0 Step 5 (audit C1 fix): treat the CBF coefficients (h, L_f h, L_g h and every
            # projection/box-argmin quantity derived from them) as CONSTANTS w.r.t. the state in
            # backward. Forward numerics are byte-identical (detach changes no values); the policy
            # BPTT gradient then flows to pi_theta only through the u_nom pathway, not through the
            # coefficients' state dependence (the source of the T=60 gradient explosion, audit C1).
            h, lf_h, lg_h = h.detach(), lf_h.detach(), lg_h.detach()
        alpha = _base_alpha(h, self.params)
        if (
            self.params.lookahead_enabled
            and self.params.lookahead_n > 0
            and self.params.lookahead_beta != 0.0
        ):
            if self.policy_fn is None:
                raise ValueError("HardNet lookahead requires a policy_fn.")
            h_peak = _lookahead_peak_h(
                system=self.system,
                h_fn=self.h_fn,
                policy_fn=self.policy_fn,
                x=x,
                scene=scene,
                params=self.params,
            )
            gap = torch.relu((h_peak - h.detach()) / self.params.lookahead_delta)
            alpha = alpha * (1.0 + self.params.lookahead_beta * gap)
        row_upper = _row_upper(lf_h, alpha, h, self.params)
        bounds = self.system.u_bounds.to(device=u_nom.device, dtype=u_nom.dtype)

        projected = _base_projection(u_nom, lg_h, row_upper, bounds, self.params)
        lg_norm = torch.linalg.norm(lg_h, dim=1)
        singular = lg_norm < _SINGULAR_LG_THRESHOLD

        if return_deficit_aux:
            # v2.4.1: u_cbf_raw = the RAW box-free half-space projection ("the control the CBF asked
            # for"), UNCLAMPED to the actuator box (may land far outside it — that is the point; it
            # carries signal at empty intersection where the clamped output would coincide with the
            # least-violating fallback). This is _base_projection's correction WITHOUT its box clamp.
            violation = torch.relu(torch.sum(lg_h * u_nom, dim=1) - row_upper)
            denom = torch.sum(lg_h * lg_h, dim=1) + self.params.epsilon ** 2 + self.params.lg_reg_eps
            u_cbf_raw = u_nom - lg_h * (violation / denom).unsqueeze(1)

        if not self.params.box_aware:
            if return_deficit_aux:
                return projected, singular, u_cbf_raw, singular
            return projected, singular

        box_projected, empty_intersection = _select_projection(
            self.params.projection,
            u_nom,
            projected,
            lg_h,
            row_upper,
            bounds,
            empty_mode=self.params.empty_mode,
            empty_prox_temp=self.params.empty_prox_temp,
        )
        # v2.8.0 Phase-2 C1 instrumentation (additive; does NOT change the returned action or flag): stash the
        # constraint row a=L_g h, the RHS b=row_upper, and the geometric box projection (pre-fallback) so the
        # per-step chatter dump can recover a, b, and the box-clipped coordinate set. Read via last_a/last_b/…
        self.last_a = lg_h.detach()
        self.last_b = row_upper.detach()
        self.last_box_projected = box_projected.detach()
        # v2.8.2 M2 (prop:sinfty-decomp): also expose the LIVE (non-detached) row a=L_g h and RHS b=row_upper so
        # the policy loss can form the empty-branch depth s_inf = a·min_corner(box) − b with the policy/state
        # gradient intact. value_net is frozen in the policy loss (theta_V detached => zero theta_V grad, G4),
        # and s_inf is linear in a,b so grad(s_inf) carries no 1/||a|| homogeneity (unlike v2.2.1/v2.4.1). These
        # references are overwritten every step; with w_infeas=0 the loss never reads them (zero extra cost).
        # Built from the PRE-detach coeffs (so detach_filter_coeffs, which detaches the projection row, does not
        # also kill s_inf's policy gradient). Uses the base alpha (lookahead, off by default, is not folded in).
        self.last_a_live = _lg_live
        self.last_b_live = _row_upper(_lf_live, _base_alpha(_h_live, self.params), _h_live, self.params)
        # v2.7.1 Stage-1: k-step empty-branch ACTION fallback (eval-only, default off). On rows where the
        # geometric intersection is empty, replace the least-violating action with the first-phase control of
        # the two-phase k-step argmin. INVARIANT: the returned flag `singular | empty_intersection` is NOT
        # touched — the row stays counted infeasible (metric/cps comparability, see retrieve §5). Singular-only
        # rows keep the current selection (empty-only scope). Non-box-aware path never reaches here (early
        # return above), so DI/unicycle parity holds.
        if self.params.empty_fallback_mode == "kstep" and bool(empty_intersection.any()):
            from src.common.kstep_fallback import grid_controls, kstep_select, slice_scene
            m = empty_intersection
            G = grid_controls(self.system, x.device, x.dtype)
            u1_star, _ = kstep_select(x[m], slice_scene(scene, m), self.h_fn, self.system, G,
                                      self.params.empty_fallback_k, self.params.dt,
                                      phases=self.params.empty_fallback_phases)
            box_projected = box_projected.clone()
            box_projected[m] = u1_star.to(box_projected.dtype)
        self.last_empty = empty_intersection.detach()          # split logging (S1d); additive, not the flag
        self.last_singular = singular.detach()
        if return_deficit_aux:
            return box_projected, singular | empty_intersection, u_cbf_raw, singular
        return box_projected, singular | empty_intersection


def _cbf_terms(
    system: System,
    h_fn: Callable[[Tensor, Any], Tensor],
    x: Tensor,
    scene: Any,
    u_nom: Tensor,
    *,
    create_graph: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    x_req = x if x.requires_grad and create_graph else x.detach().clone()
    x_req = x_req.requires_grad_(True)
    with torch.enable_grad():
        h = h_fn(x_req, scene).reshape(-1)
        grad_h = torch.autograd.grad(
            h.sum(),
            x_req,
            create_graph=create_graph,
            retain_graph=create_graph,
        )[0]

        zero_u = torch.zeros_like(u_nom)
        f_x = system.dynamics(x_req, zero_u)
        g_columns = []
        for action_idx in range(u_nom.shape[1]):
            basis = torch.zeros_like(u_nom)
            basis[:, action_idx] = 1.0
            g_columns.append(system.dynamics(x_req, basis) - f_x)
        g_x = torch.stack(g_columns, dim=2)

        lf_h = torch.sum(grad_h * f_x, dim=1)
        lg_h = torch.einsum("bs,bsa->ba", grad_h, g_x)
    return h, lf_h, lg_h


def _base_projection(
    u_nom: Tensor,
    halfspace_normal: Tensor,
    row_upper: Tensor,
    bounds: Tensor,
    params: _HardNetParams,
) -> Tensor:
    lhs = torch.sum(halfspace_normal * u_nom, dim=1)
    violation = torch.relu(lhs - row_upper)
    denom = torch.sum(halfspace_normal * halfspace_normal, dim=1)   # ||L_g h||^2
    denom = denom + params.epsilon**2 + params.lg_reg_eps             # lg_reg_eps: low-speed singularity reg
    correction = halfspace_normal * (violation / denom).unsqueeze(1)
    return _clamp_to_bounds(u_nom - correction, bounds)


def _box_aware_projection(
    u_nom: Tensor,
    base_projected: Tensor,
    halfspace_normal: Tensor,
    row_upper: Tensor,
    bounds: Tensor,
    empty_mode: str = "argmin",
    empty_prox_temp: float | None = None,
) -> tuple[Tensor, Tensor]:
    candidates = _candidate_actions(
        u_nom,
        base_projected,
        halfspace_normal,
        row_upper,
        bounds,
    )
    lhs = torch.einsum("ba,bna->bn", halfspace_normal, candidates)
    feasible = lhs <= row_upper.unsqueeze(1) + _FEASIBILITY_TOL
    distance_sq = torch.sum((candidates - u_nom.unsqueeze(1)) ** 2, dim=2)

    inf = torch.full_like(distance_sq, torch.inf)
    feasible_scores = torch.where(feasible, distance_sq, inf)
    feasible_idx = torch.argmin(feasible_scores, dim=1)

    violation = torch.relu(lhs - row_upper.unsqueeze(1))
    least_bad_scores = violation + _FEASIBILITY_TOL * distance_sq
    least_bad_idx = torch.argmin(least_bad_scores, dim=1)

    has_feasible = torch.any(feasible, dim=1)
    batch_idx = torch.arange(u_nom.shape[0], device=u_nom.device)
    if empty_mode == "prox":
        # v2.8.2 M1 (prop:empty-prox): continuous, gradient-carrying empty-branch map. Replace the discrete
        # least-violating argmin with a softmin-weighted blend of the SAME candidate set (temperature = the
        # S5-derived continuity scale gamma; gamma -> 0 recovers argmin). Feasible rows keep the exact
        # min-distance feasible candidate; only empty rows take the smooth prox map, so the empty branch
        # carries d(u_safe)/d(state) instead of a zero/argmin Jacobian.
        if empty_prox_temp is None:
            raise ValueError("filter.empty_prox_temp must be set (no default) when empty_mode='prox'.")
        weights = torch.softmax(-least_bad_scores / float(empty_prox_temp), dim=1)   # [B, n_cand]
        prox_action = torch.sum(weights.unsqueeze(-1) * candidates, dim=1)           # [B, A]
        selected = torch.where(has_feasible.unsqueeze(1), candidates[batch_idx, feasible_idx], prox_action)
    else:                                                          # "argmin" (default) — v2.8.0 behaviour, bit-for-bit
        selected_idx = torch.where(has_feasible, feasible_idx, least_bad_idx)
        selected = candidates[batch_idx, selected_idx]

    empty_intersection = _empty_halfspace_box(
        halfspace_normal,
        row_upper,
        bounds,
    )
    return selected, empty_intersection


def _select_projection(
    projection: str,
    u_nom: Tensor,
    base_projected: Tensor,
    halfspace_normal: Tensor,
    row_upper: Tensor,
    bounds: Tensor,
    empty_mode: str = "argmin",
    empty_prox_temp: float | None = None,
) -> tuple[Tensor, Tensor]:
    """v2.8.0 S1: dispatch the box-aware projection realization behind filter.projection.

    Both realizations return (selected_action, empty_intersection) and are reachable at every
    call site. `enumerate` = the legacy finite-candidate selection (prop:sel; exact only for
    action dim <= 2, prop:enum-exact(b)). `dual_solve` = the exact single-scalar dual root of
    prop:lambda-solve (exact and continuous in every dimension)."""
    if projection == "enumerate":
        return _box_aware_projection(u_nom, base_projected, halfspace_normal, row_upper, bounds,
                                     empty_mode, empty_prox_temp)
    if projection == "dual_solve":
        return _dual_solve_projection(u_nom, base_projected, halfspace_normal, row_upper, bounds,
                                      empty_mode, empty_prox_temp)
    raise ValueError(
        f"Unknown filter.projection {projection!r}; expected 'dual_solve' or 'enumerate'."
    )


def _dual_solve_projection(
    u_nom: Tensor,
    base_projected: Tensor,
    halfspace_normal: Tensor,
    row_upper: Tensor,
    bounds: Tensor,
    empty_mode: str = "argmin",
    empty_prox_temp: float | None = None,
) -> tuple[Tensor, Tensor]:
    """Exact projection of prop:lambda-solve: the Euclidean projection of u_nom onto the single
    row {a^T u <= b} intersected with the box U, computed as the unique nonnegative root of the
    piecewise-linear non-increasing phi(lambda)=a^T clip(u_nom - lambda a, U). Feasible rows get
    the exact minimizer; empty rows (prop:lambda-solve does not apply) keep the UNCHANGED
    least-violating enumeration output (bit-parity, scored by A1n)."""
    empty = _empty_halfspace_box(halfspace_normal, row_upper, bounds)
    projected = _exact_dual(u_nom, halfspace_normal, row_upper, bounds)
    if bool(empty.any()):
        mask = empty
        enum_sub, _ = _box_aware_projection(
            u_nom[mask],
            base_projected[mask],
            halfspace_normal[mask],
            row_upper[mask],
            bounds,
            empty_mode,
            empty_prox_temp,
        )
        projected = projected.clone()
        projected[mask] = enum_sub.to(projected.dtype)
    return projected, empty


def _dual_solve_projection_branchless(
    u_nom: Tensor,
    base_projected: Tensor,
    halfspace_normal: Tensor,
    row_upper: Tensor,
    bounds: Tensor,
    empty_mode: str = "argmin",
    empty_prox_temp: float | None = None,
) -> tuple[Tensor, Tensor]:
    """v2.8.2 MISSION probe — BRANCHLESS equivalent of `_dual_solve_projection`. Computes the enumerate path for
    ALL rows and selects with `torch.where(empty, enum, dual)` instead of masking empty rows behind
    `if bool(empty.any())`. Output is byte-identical to `_dual_solve_projection` (the box-aware selection is
    row-independent, so `enum_all[empty] == enum_sub`). ADDITIVE and NOT wired into any deployed path — measurement
    only. Removes the sole host sync in the projection so the filter forward can be CUDA-graph captured. Cost: the
    enumerate path now runs on the ~88% non-empty rows too; the probe measures whether the launch-overhead saved by
    graph capture outweighs that."""
    empty = _empty_halfspace_box(halfspace_normal, row_upper, bounds)
    dual = _exact_dual(u_nom, halfspace_normal, row_upper, bounds)
    enum_all, _ = _box_aware_projection(u_nom, base_projected, halfspace_normal, row_upper, bounds,
                                        empty_mode, empty_prox_temp)
    projected = torch.where(empty.unsqueeze(1), enum_all.to(dual.dtype), dual)
    return projected, empty


def _exact_dual(
    u_nom: Tensor,
    a: Tensor,
    b: Tensor,
    bounds: Tensor,
) -> Tensor:
    """Batched closed-form solve of a^T clip(u_nom - lambda a, U) = b, lambda >= 0.

    The active interval and clipped set are found on detached copies (they are locally constant
    off the measure-zero breakpoint set, exactly prop:lambda-solve's a.e. statement); lambda* and
    the returned action are then written in closed form of the LIVE (u_nom, a, b) so autograd sees
    I - a a^T / ||a||_I^2 on the interior block. No batch loop; no iterative root finder."""
    batch, dim = u_nom.shape
    low = bounds[:, 0].to(device=u_nom.device, dtype=u_nom.dtype)     # [dim]
    high = bounds[:, 1].to(device=u_nom.device, dtype=u_nom.dtype)    # [dim]
    clip0 = torch.clamp(u_nom, low, high)

    ud, ad, bd = u_nom.detach(), a.detach(), b.detach()
    amask = ad.abs() > _DENOM_TOL                                     # [B,dim] moving coords
    safe_a = torch.where(amask, ad, torch.ones_like(ad))
    bp_hi = torch.where(amask, (ud - high) / safe_a, torch.full_like(ud, -1.0))
    bp_lo = torch.where(amask, (ud - low) / safe_a, torch.full_like(ud, -1.0))
    bp = torch.cat([bp_hi, bp_lo], dim=1)                             # [B,2*dim]
    keep = bp >= 0.0
    bp = torch.where(keep, bp, torch.full_like(bp, -1.0))
    lam_max = torch.clamp(bp.amax(dim=1, keepdim=True), min=0.0)      # [B,1] largest real breakpoint
    bp = torch.where(keep, bp, lam_max.expand_as(bp))                 # invalid -> finite sentinel
    lam = torch.cat(
        [torch.zeros(batch, 1, device=u_nom.device, dtype=u_nom.dtype), bp], dim=1
    )                                                                 # [B, 2*dim+1]
    lam_sorted, _ = torch.sort(lam, dim=1)
    u_lam = torch.clamp(
        ud.unsqueeze(1) - lam_sorted.unsqueeze(2) * ad.unsqueeze(1), low, high
    )                                                                 # [B,K,dim]
    phi = torch.sum(ad.unsqueeze(1) * u_lam, dim=2)                   # [B,K] non-increasing in lambda
    le = phi <= bd.unsqueeze(1) + _FEASIBILITY_TOL
    has = le.any(dim=1)
    kstar = torch.where(
        has, torch.argmax(le.to(torch.int8), dim=1),
        torch.zeros(batch, dtype=torch.long, device=u_nom.device),
    )
    rows = torch.arange(batch, device=u_nom.device)
    lam_hi = lam_sorted[rows, kstar]
    lam_lo = lam_sorted[rows, torch.clamp(kstar - 1, min=0)]
    lam_mid = 0.5 * (lam_lo + lam_hi)                                 # strictly inside the active interval
    uj = ud - lam_mid.unsqueeze(1) * ad
    interior = (uj > low) & (uj < high) & amask                      # [B,dim] locally-constant active set
    clipped_val = torch.clamp(uj, low, high)                          # clipped coords sit at a bound
    already = phi0_le(ad, clip0.detach(), bd)                         # row already satisfied by clip(u_nom)

    w = torch.where(interior, u_nom, clipped_val)                     # LIVE on interior coords
    num = torch.sum(a * w, dim=1) - b
    denom = torch.sum(torch.where(interior, a * a, torch.zeros_like(a)), dim=1)
    denom = torch.where(denom > _DENOM_TOL, denom, torch.ones_like(denom))
    lam_star = num / denom
    projected = torch.clamp(u_nom - lam_star.unsqueeze(1) * a, low, high)
    return torch.where(already.unsqueeze(1), clip0, projected)


def phi0_le(a_det: Tensor, clip0_det: Tensor, b_det: Tensor) -> Tensor:
    """a^T clip(u_nom, U) <= b : the row is already satisfied by the box-clipped nominal."""
    return torch.sum(a_det * clip0_det, dim=1) <= b_det + _FEASIBILITY_TOL


def _candidate_actions(
    u_nom: Tensor,
    base_projected: Tensor,
    halfspace_normal: Tensor,
    row_upper: Tensor,
    bounds: Tensor,
) -> Tensor:
    # The box-aware enumerator is specified for the current 2-D action systems.
    action_dim = u_nom.shape[1]
    low = bounds[:, 0]
    high = bounds[:, 1]
    clamped_nom = _clamp_to_bounds(u_nom, bounds)
    candidates = [clamped_nom, _clamp_to_bounds(base_projected, bounds)]

    for corner_bits in product((0, 1), repeat=action_dim):
        corner_values = [
            high[idx] if bit else low[idx] for idx, bit in enumerate(corner_bits)
        ]
        corner = torch.stack(corner_values).to(device=u_nom.device, dtype=u_nom.dtype)
        candidates.append(corner.unsqueeze(0).expand(u_nom.shape[0], -1))

    for fixed_idx in range(action_dim):
        for fixed_side in (0, 1):
            fixed_value = high[fixed_idx] if fixed_side else low[fixed_idx]
            for solve_idx in range(action_dim):
                if solve_idx == fixed_idx:
                    continue
                cand = _replace_column(
                    clamped_nom,
                    fixed_idx,
                    fixed_value.expand(u_nom.shape[0]),
                )
                current_lhs = torch.sum(halfspace_normal * cand, dim=1)
                residual = (
                    row_upper
                    - current_lhs
                    + halfspace_normal[:, solve_idx] * cand[:, solve_idx]
                )
                denom = halfspace_normal[:, solve_idx]
                solved = residual / torch.where(
                    torch.abs(denom) > _DENOM_TOL,
                    denom,
                    torch.ones_like(denom),
                )
                solved_column = torch.where(
                    torch.abs(denom) > _DENOM_TOL,
                    solved,
                    cand[:, solve_idx],
                )
                cand = _replace_column(cand, solve_idx, solved_column)
                candidates.append(_clamp_to_bounds(cand, bounds))

    return torch.stack(candidates, dim=1)


def _empty_halfspace_box(
    halfspace_normal: Tensor,
    row_upper: Tensor,
    bounds: Tensor,
) -> Tensor:
    low = bounds[:, 0]
    high = bounds[:, 1]
    minimizing_corner = torch.where(halfspace_normal >= 0.0, low, high)
    min_lhs = torch.sum(halfspace_normal * minimizing_corner, dim=1)
    return min_lhs > row_upper + _FEASIBILITY_TOL


def _clamp_to_bounds(u: Tensor, bounds: Tensor) -> Tensor:
    return torch.clamp(u, min=bounds[:, 0], max=bounds[:, 1])


def _replace_column(tensor: Tensor, column_idx: int, values: Tensor) -> Tensor:
    columns = [tensor[:, idx] for idx in range(tensor.shape[1])]
    columns[column_idx] = values
    return torch.stack(columns, dim=1)


def _hardnet_params(config: Mapping[str, Any]) -> _HardNetParams:
    hardnet_cfg = config["filter"]["hardnet"]
    lookahead_cfg = config["filter"].get("lookahead", {})
    # v2.8.0 B6.1: per-system empty_fallback. A nested dict under the system's name overrides the global
    # scalars; an absent per-system entry leaves the global block untouched (byte-identical).
    ef_cfg = dict(config["filter"].get("empty_fallback", {}))
    _sys = str(config.get("run", {}).get("system", ""))
    _per_sys = ef_cfg.get(_sys)
    if isinstance(_per_sys, dict):
        ef_cfg = {**ef_cfg, **_per_sys}
    # v2.8.2: gamma_margin (row-only). Same per-system override path as empty_fallback: a scalar is
    # global; a dict may carry a per-system key (falling back to "default", then 0.0). Absent => 0.0,
    # which takes _row_upper's guarded branch and is byte-identical to the pre-v2.8.2 row.
    _gm = config["filter"].get("gamma_margin", 0.0)
    if isinstance(_gm, dict):
        _gm = _gm.get(_sys, _gm.get("default", 0.0))
    return _HardNetParams(
        epsilon=float(hardnet_cfg["epsilon"]),
        lg_reg_eps=float(config["filter"].get("lg_reg_eps", 0.0)),
        box_aware=bool(hardnet_cfg["box_aware"]),
        alpha_safe=float(config["filter"]["alpha_safe"]),
        alpha_unsafe=float(config["filter"]["alpha_unsafe"]),
        lookahead_enabled=bool(lookahead_cfg.get("enabled", False)),
        lookahead_n=int(lookahead_cfg.get("N", 0)),
        lookahead_beta=float(lookahead_cfg.get("beta", 0.0)),
        lookahead_delta=max(float(lookahead_cfg.get("delta", 0.1)), 1.0e-8),
        dt=float(config.get("env", {}).get("dt", 0.05)),
        empty_fallback_mode=str(ef_cfg.get("mode", "none")),
        empty_fallback_k=int(ef_cfg.get("k", 10)),
        empty_fallback_phases=int(ef_cfg.get("phases", 2)),
        projection=str(config["filter"].get("projection", "dual_solve")),
        empty_mode=str(config["filter"].get("empty_mode", "argmin")),   # v2.8.2 M1; default argmin = v2.8.0 behaviour
        empty_prox_temp=(None if config["filter"].get("empty_prox_temp") is None
                         else float(config["filter"]["empty_prox_temp"])),   # NO silent default (S5-derived)
        gamma_margin=float(_gm),
    )


def _row_upper(lf_h: Tensor, alpha: Tensor, h: Tensor, params: _HardNetParams) -> Tensor:
    """THE single constructor for the deployed CBF row RHS b (02_control §6.1).

        b = -L_f h - alpha * (h + gamma_margin)

    gamma_margin = 0.0 (default) takes the guarded branch and is BYTE-IDENTICAL to the pre-v2.8.2
    expression `-lf_h - alpha * h` (not merely numerically equal). Row-only by construction: alpha is
    passed in already selected by _base_alpha(h, ...) on the RAW h, so the margin never moves the
    safe/unsafe region test. NOTE: 02_control §6.1 also specifies a V_SHIFT = 1e-3 term in h_eff which
    the shipped row has never carried; V_SHIFT is deliberately NOT added here (recorded divergence)."""
    g = float(params.gamma_margin)
    if g == 0.0:
        return -lf_h - alpha * h
    return -lf_h - alpha * (h + g)


def _base_alpha(h: Tensor, params: _HardNetParams) -> Tensor:
    return torch.where(
        h <= 0.0,
        torch.full_like(h, params.alpha_safe),
        torch.full_like(h, params.alpha_unsafe),
    )


def _lookahead_peak_h(
    *,
    system: System,
    h_fn: Callable[[Tensor, Any], Tensor],
    policy_fn: Callable[[Tensor, Any], Tensor],
    x: Tensor,
    scene: Any,
    params: _HardNetParams,
) -> Tensor:
    horizon = max(0, int(params.lookahead_n))
    x_la = system.wrap_state(x.detach())
    with torch.no_grad():
        h_peak = h_fn(x_la, scene).reshape(-1).detach()
    bounds = system.u_bounds.to(device=x_la.device, dtype=x_la.dtype)

    for _ in range(horizon):
        with torch.no_grad():
            u_nom = policy_fn(x_la, scene).detach()
        h, lf_h, lg_h = _cbf_terms(
            system,
            h_fn,
            x_la,
            scene,
            u_nom,
            create_graph=False,
        )
        alpha = _base_alpha(h, params)
        row_upper = _row_upper(lf_h, alpha, h, params)
        projected = _base_projection(u_nom, lg_h, row_upper, bounds, params)
        if params.box_aware:
            projected, _ = _select_projection(
                params.projection,
                u_nom,
                projected,
                lg_h,
                row_upper,
                bounds,
                empty_mode=params.empty_mode,
                empty_prox_temp=params.empty_prox_temp,
            )
        with torch.no_grad():
            x_la = rk4_step(system, x_la, projected.detach(), params.dt)
            h_next = h_fn(x_la, scene).reshape(-1).detach()
            h_peak = torch.maximum(h_peak, h_next)
    return h_peak
