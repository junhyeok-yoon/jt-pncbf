"""v2.8.1 S1 — gates for the continuous soft-rank obstacle encoder (`soft_topk_obstacles`).

G1 (limit parity): at beta -> inf with sigma == 1 (d_c == inf) the soft encoder reproduces `top_k_obstacles`
BIT-WISE across a battery — exact distance ties, k-th-position exchanges, fewer-than-k active, all-inactive.
Plus the two structural properties the encoder must have for the continuity gate to be meaningful:
permutation invariance in obstacle index, and continuity across a k-th-position rank crossing (no jump)."""
from __future__ import annotations
import math

import torch

from src.common.observation import soft_topk_obstacles, top_k_obstacles

DT = torch.float64
INF = float("inf")


def _scene(centers, radii, active):
    c = torch.tensor(centers, dtype=DT)
    r = torch.tensor(radii, dtype=DT)
    a = torch.tensor(active, dtype=torch.bool)
    return c, r, a


def _bitwise_equal(pos, centers, radii, active, k):
    hard = top_k_obstacles(pos, centers, radii, active, k)
    soft = soft_topk_obstacles(pos, centers, radii, active, k, beta=INF, d_c=INF)
    return torch.equal(hard[0], soft[0]) and torch.equal(hard[1], soft[1])


def test_g1_hard_limit_bitwise_battery():
    k = 5
    # (a) generic random battery
    g = torch.Generator().manual_seed(3)
    pos = torch.randn(16, 2, generator=g, dtype=DT)
    centers = torch.randn(16, 7, 2, generator=g, dtype=DT)
    radii = 0.2 + 0.3 * torch.rand(16, 7, generator=g, dtype=DT)
    active = torch.rand(16, 7, generator=g) > 0.3
    assert _bitwise_equal(pos, centers, radii, active, k), "generic battery not bit-wise"

    # (b) EXACT distance ties: two obstacles at identical surface distance from the drone
    pos_t = torch.zeros(1, 2, dtype=DT)
    centers_t = torch.tensor([[[1.0, 0.0], [-1.0, 0.0], [0.0, 2.0], [3.0, 0.0], [0.0, -4.0]]], dtype=DT)
    radii_t = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5]], dtype=DT)   # obs0,obs1 tie at surface dist 0.5
    active_t = torch.ones(1, 5, dtype=torch.bool)
    assert _bitwise_equal(pos_t, centers_t, radii_t, active_t, k), "distance-tie not bit-wise"

    # (c) k-th-position exchange: the k-th and (k+1)-th nearest at nearly equal distance
    pos_x = torch.zeros(1, 2, dtype=DT)
    centers_x = torch.tensor([[[0.5, 0], [1.0, 0], [1.5, 0], [2.0, 0], [2.5000001, 0], [2.4999999, 0]]], dtype=DT)
    radii_x = torch.zeros(1, 6, dtype=DT)
    active_x = torch.ones(1, 6, dtype=torch.bool)
    assert _bitwise_equal(pos_x, centers_x, radii_x, active_x, k), "k-th exchange not bit-wise"

    # (d) fewer-than-k active
    active_f = torch.tensor([[True, True, False, False, False, False]], dtype=torch.bool)
    assert _bitwise_equal(pos_x, centers_x, radii_x, active_f, k), "fewer-than-k not bit-wise"

    # (e) all-inactive scene
    active_z = torch.zeros(1, 6, dtype=torch.bool)
    assert _bitwise_equal(pos_x, centers_x, radii_x, active_z, k), "all-inactive not bit-wise"


def test_g1b_permutation_invariance_finite_beta():
    k = 5
    pos = torch.zeros(1, 2, dtype=DT)
    centers = torch.tensor([[[1.0, 0.2], [-0.7, 1.1], [2.0, -1.0], [0.3, 0.4], [-1.5, -1.5], [2.5, 2.5]]], dtype=DT)
    radii = torch.tensor([[0.4, 0.3, 0.5, 0.2, 0.6, 0.3]], dtype=DT)
    active = torch.ones(1, 6, dtype=torch.bool)
    perm = torch.tensor([3, 0, 5, 1, 4, 2])
    r0 = soft_topk_obstacles(pos, centers, radii, active, k)              # derived beta/d_c
    r1 = soft_topk_obstacles(pos, centers[:, perm], radii[:, perm], active[:, perm], k)
    assert torch.allclose(r0[0], r1[0], atol=1e-12) and torch.allclose(r0[1], r1[1], atol=1e-12), \
        "soft encoder is not permutation-invariant"


def test_g1c_continuity_across_kth_exchange():
    # Two obstacles swap rank as the drone moves; the soft encoder's output must be continuous through the
    # crossing (small step for a small state step), whereas the hard encoder jumps.
    k = 5
    centers = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]], dtype=DT)       # symmetric about x=0
    radii = torch.zeros(1, 2, dtype=DT)
    active = torch.ones(1, 2, dtype=torch.bool)
    eps = 1e-4
    pos_a = torch.tensor([[-eps, 0.0]], dtype=DT)                        # obs0 nearer
    pos_b = torch.tensor([[+eps, 0.0]], dtype=DT)                        # obs1 nearer (rank swapped)
    soft_a = soft_topk_obstacles(pos_a, centers, radii, active, k)[0]
    soft_b = soft_topk_obstacles(pos_b, centers, radii, active, k)[0]
    hard_a = top_k_obstacles(pos_a, centers, radii, active, k)[0]
    hard_b = top_k_obstacles(pos_b, centers, radii, active, k)[0]
    soft_jump = float((soft_a - soft_b).abs().max())
    hard_jump = float((hard_a - hard_b).abs().max())
    assert soft_jump < 1e-2, f"soft encoder jumped at the crossing: {soft_jump}"
    assert hard_jump > 1.0, f"hard encoder should jump at the crossing (control): {hard_jump}"


# ---- v2.8.1 S1 beta-screen — gates for the redesigned (occupancy-gated, log-space) soft-rank ----------------
# FD-agreement tolerance for the SCREEN band. Rationale (06_workflow: state the tolerance and why it suffices):
# the screen selects on OC in-loop collision measured on n500 (quantized at 1/500 = 0.002) against a <=0.06 rule;
# an obs-gradient error below FD_TOL=1e-2 is far below what could move that metric, so a passing encoder measures
# beta, not numerical error. The redesign is FD-exact to ~1e-3 across the screen band {2,6,12}.
FD_TOL = 1e-2


def _battery32():
    g = torch.Generator().manual_seed(11)
    out = []
    out.append((torch.zeros(1, 2, dtype=torch.float32),                         # dominant near + far (sigma=0)
                torch.tensor([[[2.39, 0.0], [5.2, 0.5], [0, 0], [0, 0], [0, 0]]], dtype=torch.float32),
                torch.full((1, 5), 0.4, dtype=torch.float32),
                torch.tensor([[True, True, False, False, False]])))
    out.append((torch.zeros(1, 2, dtype=torch.float32),                         # near-ties (not bit-exact)
                torch.tensor([[[1.0, 0], [1.0001, 0], [-1.0, .02], [2, 2], [-2, -2]]], dtype=torch.float32),
                torch.full((1, 5), 0.4, dtype=torch.float32), torch.ones(1, 5, dtype=torch.bool)))
    out.append((torch.randn(4, 2, generator=g, dtype=torch.float32),            # generic random
                torch.randn(4, 6, 2, generator=g, dtype=torch.float32),
                0.2 + 0.3 * torch.rand(4, 6, generator=g, dtype=torch.float32),
                torch.rand(4, 6, generator=g) > 0.3))
    return out


def test_g8_fd_agreement_screen_band_float32():
    """FD agreement (true derivative) across the tie battery at the SCREEN betas {2,6,12}, float32. This is what
    the pre-fix encoder failed (grad ~1e24). The redesign carries occupancy from the mask and the cross-slot
    exclusion in log space, so d(obs)/d(p_xy) is finite and matches central differences to < FD_TOL."""
    for pos, c, r, a in _battery32():
        for beta in (2.0, 6.0, 12.0):
            pos_g = pos.clone().requires_grad_(True)
            rel, rad = soft_topk_obstacles(pos_g, c, r, a, 5, beta=beta)
            g = torch.autograd.grad((rel.sum() + rad.sum()), pos_g)[0]
            fd = torch.zeros_like(pos)
            h = 1e-3
            for j in range(2):
                e = torch.zeros_like(pos); e[:, j] = h
                with torch.no_grad():
                    fp = soft_topk_obstacles(pos + e, c, r, a, 5, beta=beta)
                    fm = soft_topk_obstacles(pos - e, c, r, a, 5, beta=beta)
                sp = fp[0].reshape(pos.shape[0], -1).sum(1) + fp[1].sum(1)    # per-batch total(rel+rad) at +e
                sm = fm[0].reshape(pos.shape[0], -1).sum(1) + fm[1].sum(1)    # at -e
                fd[:, j] = (sp - sm) / (2 * h)
            err = float((g - fd).abs().max())
            assert torch.isfinite(g).all() and err < FD_TOL, f"FD disagreement beta={beta}: {err}"


def test_g8b_gradient_finite_and_bounded_high_beta():
    """In the crash regime (beta up to 50, float32) the gradient stays finite and bounded — the pre-fix encoder
    reached ~1e24 here (numerically defeating the CBF-QP). Bound is generous; the point is no explosion."""
    for pos, c, r, a in _battery32():
        for beta in (6.0, 12.0, 30.0, 50.0):
            pos_g = pos.clone().requires_grad_(True)
            rel, rad = soft_topk_obstacles(pos_g, c, r, a, 5, beta=beta)
            g = torch.autograd.grad((rel.sum() + rad.sum()), pos_g)[0]
            assert torch.isfinite(g).all() and float(g.abs().max()) < 1.0e3, f"grad blew up beta={beta}"


def test_g8c_zero_pad_fewer_than_k():
    """Occupancy carries the zero-pad contract: with m < k active obstacles, slots m..k-1 are EXACTLY zero."""
    dt = DT
    centers = torch.tensor([[[1.5, 0.0], [-2.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]], dtype=dt)
    radii = torch.full((1, 5), 0.4, dtype=dt)
    for m in (1, 2, 3):
        active = torch.tensor([[i < m for i in range(5)]])
        rel, rad = soft_topk_obstacles(torch.zeros(1, 2, dtype=dt), centers, radii, active, 5, beta=6.0)
        assert bool((rel[:, m:] == 0).all()) and bool((rad[:, m:] == 0).all()), f"slots >= {m} not zero-padded"


def test_g8d_per_slot_gradient_liveness():
    """No slot silently loses its gradient (the pre-fix beta=50 'grad 1.0' was dead slots 1-4, unreported)."""
    g = torch.Generator().manual_seed(5)
    c = 1.1 * torch.randn(1, 5, 2, generator=g, dtype=DT)
    r = torch.full((1, 5), 0.4, dtype=DT); a = torch.ones(1, 5, dtype=torch.bool)
    for s in range(5):
        pos = torch.zeros(1, 2, dtype=DT, requires_grad=True)
        rel, _ = soft_topk_obstacles(pos, c, r, a, 5, beta=12.0)
        gs = torch.autograd.grad(rel[:, s].sum(), pos, retain_graph=True)[0]
        assert float(gs.abs().max()) > 0.0, f"slot {s} gradient is dead"


def test_g1d_soft_path_large_finite_beta_finite_float32():
    """G1 limit EXERCISES THE SOFT PATH (not the beta=inf delegation): at large FINITE beta in float32 the
    soft-rank forward and gradient must stay finite and bounded. delegation at beta=inf + float64 tests is why
    the float32 high-beta pathology was originally missed. (The soft rank does NOT converge to top_k on this
    path: iterated soft removal leaves a dominant obstacle tied with the next at high beta — a property of the
    formulation, present in the original too; bit-exact top_k is provided by the beta=inf delegation, test_g1.)"""
    g = torch.Generator().manual_seed(7)
    for beta in (50.0, 200.0):
        pos = torch.randn(8, 2, generator=g, dtype=torch.float32, requires_grad=True)
        c = torch.randn(8, 6, 2, generator=g, dtype=torch.float32)
        r = 0.2 + 0.3 * torch.rand(8, 6, generator=g, dtype=torch.float32)
        a = torch.rand(8, 6, generator=g) > 0.3
        rel, rad = soft_topk_obstacles(pos, c, r, a, 5, beta=beta, d_c=float("inf"))   # SOFT path, finite beta
        assert torch.isfinite(rel).all() and torch.isfinite(rad).all(), f"soft path non-finite at beta={beta}"
        gg = torch.autograd.grad((rel.sum() + rad.sum()), pos)[0]
        assert torch.isfinite(gg).all() and float(gg.abs().max()) < 1.0e3, f"soft-path grad blew up beta={beta}"
