from __future__ import annotations

import torch

Tensor = torch.Tensor


def _not_a_knot_matrix(n_pts: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    # Coefficient matrix for the second-derivative moments M_0..M_{n_pts-1} of a uniform-grid
    # cubic spline with not-a-knot end conditions (S''' continuous across the first/last interior
    # knots). The matrix is grid-spacing independent; h enters only through the right-hand side.
    n_seg = n_pts - 1
    amat = torch.zeros(n_pts, n_pts, device=device, dtype=dtype)
    interior = torch.arange(1, n_seg, device=device)
    amat[interior, interior - 1] = 1.0
    amat[interior, interior] = 4.0
    amat[interior, interior + 1] = 1.0
    amat[0, 0], amat[0, 1], amat[0, 2] = 1.0, -2.0, 1.0
    amat[n_seg, n_seg - 2], amat[n_seg, n_seg - 1], amat[n_seg, n_seg] = 1.0, -2.0, 1.0
    return amat


def cubic_spline_max(t: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    """Differentiable maximum of the not-a-knot cubic spline through knots (t, y).

    t: (n_pts,) ascending uniform grid. y: (..., n_pts) knot values. Returns (maxval (...),
    maxloc (...)). The spline moments are solved with torch.linalg.solve (differentiable in y).
    Per segment the cubic derivative (a quadratic) is rooted with a stable formula; roots inside the
    segment with negative second derivative are candidate interior maxima, compared against all knots.
    The interior-critical-point location is detached when evaluating the cubic: by the envelope theorem
    the location's first-order contribution vanishes at a maximizer, so the gradient flows exactly
    through the spline moments and knot values without differentiating the root solve (robust, no sqrt
    singularities). Only maxval carries the downstream gradient; maxloc is a detached diagnostic.
    """
    if t.ndim != 1 or t.shape[0] < 4:
        raise ValueError("t must be 1-D with at least 4 knots (not-a-knot needs n_seg >= 3).")
    n_pts = t.shape[0]
    n_seg = n_pts - 1
    device, dtype = y.device, y.dtype
    h = (t[-1] - t[0]) / float(n_seg)                                # 0-dim tensor, no host sync

    lead = y.shape[:-1]
    yf = y.reshape(-1, n_pts)                                        # (M, n_pts)
    m_batch = yf.shape[0]

    amat = _not_a_knot_matrix(n_pts, device, dtype)
    rhs = yf.new_zeros(n_pts, m_batch)
    second_diff = yf[:, 2:] - 2.0 * yf[:, 1:-1] + yf[:, :-2]         # (M, n_seg-1)
    rhs[1:n_seg, :] = (6.0 / (h * h)) * second_diff.transpose(0, 1)
    moments = torch.linalg.solve(amat, rhs).transpose(0, 1)         # (M, n_pts)

    mi = moments[:, :-1]
    mip = moments[:, 1:]
    yi = yf[:, :-1]
    yip = yf[:, 1:]

    a_q = (mip - mi) / (2.0 * h)
    b_q = mi
    c_q = -mi * h / 2.0 + (yip - yi) / h - (h / 6.0) * (mip - mi)
    disc = b_q * b_q - 4.0 * a_q * c_q
    valid_disc = disc > 0.0
    sqrt_disc = torch.sqrt(disc.clamp_min(0.0))
    sign_b = torch.where(b_q >= 0.0, torch.ones_like(b_q), -torch.ones_like(b_q))
    qv = -0.5 * (b_q + sign_b * sqrt_disc)
    eps = 1.0e-12
    root1 = qv / torch.where(a_q.abs() > eps, a_q, torch.ones_like(a_q))
    root2 = c_q / torch.where(qv.abs() > eps, qv, torch.ones_like(qv))

    seg_left = t[:-1].reshape(1, n_seg)

    def candidate(root: Tensor, root_ok: Tensor) -> tuple[Tensor, Tensor]:
        in_range = (root >= 0.0) & (root <= h)
        b = torch.minimum(root.detach().clamp_min(0.0), h)          # envelope theorem: detach location
        a = h - b
        spp = mi + (mip - mi) * b / h                               # S''(b)
        val = (
            (mi * a ** 3 + mip * b ** 3) / (6.0 * h)
            + (yi * a + yip * b) / h
            - (h / 6.0) * (mi * a + mip * b)
        )
        ok = root_ok & valid_disc & in_range & (spp < 0.0)
        val = torch.where(ok, val, torch.full_like(val, float("-inf")))
        loc = (seg_left + b).expand_as(val)
        return val, loc

    val1, loc1 = candidate(root1, a_q.abs() > eps)
    val2, loc2 = candidate(root2, qv.abs() > eps)
    knot_loc = t.reshape(1, n_pts).expand(m_batch, n_pts)

    cand_val = torch.cat([val1, val2, yf], dim=1)
    cand_loc = torch.cat([loc1, loc2, knot_loc], dim=1)
    maxval, argmax = cand_val.max(dim=1)
    maxloc = cand_loc.gather(1, argmax.unsqueeze(1)).squeeze(1).detach()

    return maxval.reshape(lead), maxloc.reshape(lead)
