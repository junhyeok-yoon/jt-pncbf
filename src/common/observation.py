from __future__ import annotations

from typing import Any

import torch


Tensor = torch.Tensor


# ---------------------------------------------------------------------------------------------------
# v2.8.3 U-PREV AXIS — the previous EXECUTED control as a POLICY-observation channel (quadrotor_3d only).
#
# WHY. On the empty branch the BPTT gradient through the filter dies (`rem:empty-grad`, `prop:sel(a)`),
# and the policy input carries nothing about the filter — so on exactly the states the filter rejects the
# policy has NO channel of any kind. Appending u_safe(t-1) opens an INPUT channel there.
#
# WHAT. obs 34 -> 38 for the POLICY ONLY: `[obs(34), u_safe(t-1) (4)]`, per-rotor, RAW NEWTONS (no
# normalisation, no centring — the same units the actuator box [f_rotor_min, f_rotor_max] is written in).
# `system.observation` itself is UNCHANGED at 34: the VALUE net V_S must stay a function of the state
# alone (the HardNet filter differentiates V_S w.r.t. x to form L_f/L_g, and the value target is
# max_t h(x_t) with h a pure state function), so only the policy net widens.
#
# t = 0 CONVENTION. HOVER TRIM m*g/4 per rotor, NOT zeros. u lives in the box [0, f_rotor_max]^4 and the
# thrust that holds altitude is m*g split over 4 rotors, so the trim is the physically neutral "no
# previous action" default and the only value continuous with a hovering history. A zero-centred default
# would repeat the `L_a` / `mu_u` mis-specification (treating 0 as the neutral input for a strictly
# non-negative rotor-force channel). Provenance of the constants: src/configs/exp_config.yaml:46
# (mass: 1.0) and :47 (gravity: 9.81) under env.quadrotor_3d -> 1.0 * 9.81 / 4 = 2.4525 N.
#
# The SAME convention holds in training, collection, in-loop eval and final eval; every site asserts it.
U_PREV_SYSTEM = "quadrotor_3d"


def u_prev_feedback_on(config: Any) -> bool:
    """The axis flag. Absent (every shipped config and every existing checkpoint) -> False -> the
    untouched dim-34 policy path, byte-identical to CTRL."""
    return bool((config.get("obs", {}) or {}).get("u_prev_feedback", False))


def u_prev_trim(system: Any, config: Any) -> float:
    """t=0 default per rotor: hover trim m*g/4 (see the axis note above)."""
    phys = config["env"][U_PREV_SYSTEM]
    trim = float(phys["mass"]) * float(phys["gravity"]) / float(system.action_dim)
    if not (trim > 0.0):
        raise ValueError(f"u_prev trim must be positive (got {trim}); a zero default is NOT the axis.")
    return trim


def u_prev_extra_dim(system: Any, config: Any) -> int:
    """Extra POLICY input columns contributed by the axis (0 when off). quadrotor_3d ONLY."""
    if not u_prev_feedback_on(config):
        return 0
    if system.name != U_PREV_SYSTEM:
        raise ValueError(
            f"obs.u_prev_feedback is quadrotor_3d-only but system is {system.name!r}."
        )
    if u_prev_feedback_on(config) and bool(
        (config.get("loss", {}) or {}).get("policy", {}).get("obs_deficit_feedback", False)
    ):
        raise ValueError(
            "obs.u_prev_feedback and loss.policy.obs_deficit_feedback both append action_dim columns "
            "to the policy observation; enabling both is ambiguous. Enable exactly one."
        )
    return int(system.action_dim)


def u_prev_init(system: Any, config: Any, batch: int, *, device: Any, dtype: Any) -> Tensor:
    """[batch, action_dim] filled with the hover trim — the t=0 value of the u_prev channel."""
    return torch.full((int(batch), int(system.action_dim)), u_prev_trim(system, config),
                      device=device, dtype=dtype)


def append_u_prev(obs: Tensor, u_prev: Tensor) -> Tensor:
    """[B,34] (+) [B,4] -> [B,38]. `u_prev` is always DETACHED by the caller: the axis opens an INPUT
    channel, not a second gradient path through the filter (the deployed and collection paths are
    no_grad, so a gradient-carrying training path would be a train/deploy semantic mismatch)."""
    if u_prev.shape[0] != obs.shape[0]:
        raise ValueError(f"u_prev batch {tuple(u_prev.shape)} != obs batch {tuple(obs.shape)}.")
    return torch.cat([obs, u_prev.detach().to(device=obs.device, dtype=obs.dtype)], dim=1)


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
    return_indices: bool = False,
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

    # v2.8.0 Phase-2 C1/D3 instrumentation (additive, opt-in): the selected obstacle indices per step, in
    # rank order, with invalid (inactive/inf-distance) slots set to -1. Does not affect existing callers.
    ind_out = None
    if return_indices:
        ind_out = indices.masked_fill(~valid, -1)

    if k_select == k:
        if return_indices:
            return top_rel, top_radii, ind_out
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
    top_rel_out = torch.cat([top_rel, rel_pad], dim=1)
    top_radii_out = torch.cat([top_radii, radii_pad], dim=1)
    if return_indices:
        idx_pad = torch.full((positions.shape[0], pad_count), -1, dtype=ind_out.dtype, device=ind_out.device)
        return top_rel_out, top_radii_out, torch.cat([ind_out, idx_pad], dim=1)
    return top_rel_out, top_radii_out


# v2.8.1 S1 — continuous (soft-rank) obstacle encoder. Derived constants, NOT tunable knobs:
#   SOFT_TOPK_BETA = 1/(0.5 m) = 2.0 /m — tied to the smoothstep width, fixes the soft-rank sharpness.
#   distance kernel sigma(d): 1 for surface distance d <= SOFT_INNER, smoothstep to 0 at SOFT_DC (width 0.5 m),
#   so obstacles beyond d_c vanish continuously. The hard limit (beta -> inf, sigma == 1 i.e. d_c -> inf) is
#   `top_k_obstacles` bit-for-bit (delegated below; gate G1). Continuous in the state everywhere (no
#   rank-crossing or k-th-exchange jump), invariant to obstacle index permutation (softmax + sum over obstacles).
SOFT_TOPK_BETA: float = 2.0
SOFT_INNER: float = 2.5
SOFT_DC: float = 3.0


def _distance_kernel(surface_distance: Tensor, inner: float, d_c: float) -> Tensor:
    """sigma(d): 1 for d<=inner, smoothstep down to 0 at d_c, 0 beyond. d_c == inf -> identically 1."""
    if d_c == float("inf"):
        return torch.ones_like(surface_distance)
    t = ((surface_distance - inner) / (d_c - inner)).clamp(0.0, 1.0)
    return 1.0 - t * t * (3.0 - 2.0 * t)                    # 1 - smoothstep(t)


def _log1mexp(x: Tensor) -> Tensor:
    """Numerically stable log(1 - exp(x)) for x <= 0 (Mächler 2012, two-branch): near x=0 use log(-expm1(x)),
    far use log1p(-exp(x)). Both branches use stable primitives (expm1 / log1p) with accurate autograd, so the
    log-space cross-slot exclusion in soft_topk_obstacles never forms `1 - exp(x)` by subtraction."""
    return torch.where(x > -0.6931471805599453,            # -log(2)
                       torch.log(-torch.expm1(x)),
                       torch.log1p(-torch.exp(x)))


def soft_topk_obstacles(
    positions: Tensor,
    centers: Tensor,
    radii: Tensor,
    active: Tensor,
    k: int,
    beta: float = SOFT_TOPK_BETA,
    d_c: float = SOFT_DC,
    inner: float = SOFT_INNER,
    return_indices: bool = False,
) -> tuple[Tensor, Tensor]:
    """Continuous soft-rank replacement for `top_k_obstacles`, same output shapes ([B,k,2],[B,k]) and obs_dim.

    Each of the k slots is an iterated-softmax soft-rank mixture over obstacles by SURFACE distance (sharpness
    beta), with the slot content multiplied by the distance kernel sigma(d) so far obstacles vanish. Inactive
    obstacles carry exactly zero weight. The (beta -> inf, sigma == 1) limit is `top_k_obstacles` bit-for-bit."""
    # Exact hard limit: delegate so beta->inf, sigma==1 reproduces the current encoder bit-wise (gate G1).
    if beta == float("inf") and d_c == float("inf"):
        return top_k_obstacles(positions, centers, radii, active, k, return_indices=return_indices)

    batch_size = positions.shape[0]
    if centers.ndim == 2:
        centers_b = centers.unsqueeze(0).expand(batch_size, -1, -1)
        radii_b = radii.unsqueeze(0).expand(batch_size, -1)
        active_b = active.unsqueeze(0).expand(batch_size, -1)
    elif centers.ndim == 3 and centers.shape[0] == batch_size:
        centers_b, radii_b, active_b = centers, radii, active
    else:
        raise ValueError(f"Obstacle center shape {tuple(centers.shape)} is incompatible with batch {batch_size}.")

    rel = centers_b - positions.unsqueeze(1)                          # [B, n, D]
    surface_distance = torch.linalg.norm(rel, dim=-1) - radii_b       # [B, n]
    # mask inactive obstacles out of every channel (finite so 0-weight * value never yields nan)
    rel_m = torch.where(active_b.unsqueeze(-1), rel, torch.zeros_like(rel))
    radii_m = torch.where(active_b, radii_b, torch.zeros_like(radii_b))
    sigma = torch.where(active_b, _distance_kernel(surface_distance, inner, d_c), torch.zeros_like(surface_distance))
    neg = torch.finfo(surface_distance.dtype).min / 4.0
    logit = torch.where(active_b, -beta * surface_distance, torch.full_like(surface_distance, neg))  # [B,n]

    # v2.8.1 S1 beta-screen — redesign (a): separate normalization (WHICH obstacle a slot attends) from occupancy
    # (WHETHER the slot exists). Occupancy is a DISCRETE fact of the active mask (occ_s = 1 iff > s obstacles are
    # active), so it carries the zero-pad contract AND contributes nothing to d(obs)/d(p_xy) — the prior form's
    # tiny-normalizer 1/S explosion (from the subtractive `1-p` residual collapsing sum(r*w)~1e-8) cannot exist by
    # construction. Attention a_s is a stable masked softmax; cross-slot exclusion is carried in LOG space via a
    # stable log(1-a) (log1mexp on the log-attention, clamped just below 0), never as a `1 - sum` subtraction.
    # beta->inf,d_c->inf delegates to top_k above; large finite beta approaches it softly (a_s -> s-th nearest).
    n_active = active_b.sum(dim=1, keepdim=True).to(surface_distance.dtype)   # [B,1] discrete occupancy source
    log_surv = torch.zeros_like(surface_distance)                            # log survival weight (0 = available)
    n_obs = surface_distance.shape[1]
    _eye = torch.eye(n_obs, dtype=torch.bool, device=surface_distance.device).unsqueeze(0)   # [1,n,n]
    slots_rel, slots_rad, slots_topmass = [], [], []
    for s in range(int(k)):
        zs = logit + log_surv                                                # logits with prior slots excluded
        a = torch.softmax(zs, dim=1)                                         # attention (WHICH obstacle), stable
        occ = (n_active > float(s)).to(surface_distance.dtype)               # [B,1] WHETHER slot exists (mask only)
        asig = a * sigma                                                     # distance-gated attention
        slots_rel.append(occ * (asig.unsqueeze(-1) * rel_m).sum(dim=1))      # [B, D]  (zero-padded by occ)
        slots_rad.append(occ.squeeze(-1) * (asig * radii_m).sum(dim=1))      # [B]
        slots_topmass.append(occ.squeeze(-1) * a.max(dim=1).values)          # informative: top attention share
        # multiplicative survival in log space. log(1 - a_i) is formed as the COMPLEMENTARY masked-logsumexp
        # (logsumexp over j != i, minus the full logsumexp) — never `lse - exp(zs_i)` — so it carries no
        # catastrophic cancellation as a_i -> 1 (which is what corrupted the log1mexp gradient and gave the b12
        # dominant-config / b30 tie FD gaps). The difference is between two well-separated log-sums, so both the
        # value and its autograd are accurate at every finite beta and dtype.
        lse = torch.logsumexp(zs, dim=1, keepdim=True)                       # [B,1]
        zs_excl = zs.unsqueeze(1).expand(-1, n_obs, -1).masked_fill(_eye, neg)   # [B,n,n], drop j==i
        log_surv = log_surv + (torch.logsumexp(zs_excl, dim=2) - lse)        # [B,n] = log(1 - a_i)

    top_rel = torch.stack(slots_rel, dim=1)                           # [B, k, D]
    top_radii = torch.stack(slots_rad, dim=1)                         # [B, k]
    if return_indices:
        # soft encoder has no discrete indices; expose per-slot top-mass share instead (for the price metrics)
        return top_rel, top_radii, torch.stack(slots_topmass, dim=1)
    return top_rel, top_radii
