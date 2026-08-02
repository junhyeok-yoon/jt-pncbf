"""v2.8.0 S1 — permanent gates for the exact dual-solve projection (prop:lambda-solve).

Generalizes the real-state gates G1/G2/G3 (scripts/analysis/v280_s1_gates.py) to fast synthetic
rows so scripts/verify.sh keeps them green. Also checks the config-flag dispatch and the empty-row
bit-parity (A1n invariant)."""
from __future__ import annotations

import itertools

import torch

from src.common.filter_hardnet import (
    _HardNetParams,
    _base_projection,
    _box_aware_projection,
    _dual_solve_projection,
    _empty_halfspace_box,
    _exact_dual,
    _hardnet_params,
    _select_projection,
)

TOL = 1.0e-9
_P0 = _HardNetParams(
    epsilon=0.0, lg_reg_eps=0.0, box_aware=True, alpha_safe=2.0, alpha_unsafe=100.0,
    lookahead_enabled=False, lookahead_n=0, lookahead_beta=0.0, lookahead_delta=0.1, dt=0.05,
)


def _rows(m: int, n: int, seed: int):
    torch.manual_seed(seed)
    low = torch.tensor([-1.0 - 0.3 * j for j in range(m)], dtype=torch.float64)
    high = torch.tensor([1.0 + 0.2 * j for j in range(m)], dtype=torch.float64)
    bounds = torch.stack([low, high], dim=1)
    u_nom = (torch.rand(n, m, dtype=torch.float64) - 0.5) * 6.0
    a = torch.randn(n, m, dtype=torch.float64)
    a[torch.rand(n, m) < 0.15] = 0.0
    b = torch.randn(n, dtype=torch.float64) * 2.0
    return u_nom, a, b, bounds


def _brute_ref(u_nom, a, b, bounds):
    n, m = u_nom.shape
    low, high = bounds[:, 0], bounds[:, 1]
    codes = torch.tensor(list(itertools.product((0, 1, 2), repeat=m)))
    interior = (codes == 2)[None]
    boundv = torch.where(codes == 0, low, high).double()[None]
    un, av = u_nom[:, None, :], a[:, None, :]
    num = (av * torch.where(interior, un, boundv)).sum(-1) - b[:, None]
    den = torch.where(interior, av * av, torch.zeros_like(av)).sum(-1)
    lam = num / den.clamp_min(1e-300)
    u = torch.where(interior, un - lam[:, :, None] * av, boundv)
    ok = interior.any(-1) & (den > 1e-15) & (u >= low - 1e-9).all(-1) & (u <= high + 1e-9).all(-1) \
        & ((av * u).sum(-1) <= b[:, None] + 1e-7)
    dist = torch.where(ok, ((u - un) ** 2).sum(-1), torch.full_like(num, float("inf")))
    best = dist.argmin(1)
    rows = torch.arange(n)
    ref, bestd = u[rows, best], dist[rows, best]
    c0 = torch.clamp(u_nom, low, high)
    use0 = ((a * c0).sum(-1) <= b + 1e-9) & (((c0 - u_nom) ** 2).sum(-1) <= bestd)
    return torch.where(use0[:, None], c0, ref)


def test_g1_parity_m2():
    """m<=2: the enumeration is exact (prop:enum-exact(b)), so dual == enumerate on satisfiable rows."""
    u_nom, a, b, bounds = _rows(2, 20000, 1)
    sat = ~_empty_halfspace_box(a, b, bounds)
    base = _base_projection(u_nom, a, b, bounds, _P0)
    dual = _exact_dual(u_nom, a, b, bounds)[sat]
    enum, _ = _box_aware_projection(u_nom, base, a, b, bounds)
    assert float((dual - enum[sat]).abs().amax()) <= TOL


def test_g2_exact_m3_m4():
    """m in {3,4}: dual equals an independent brute-force active-set projection to 1e-9;
    the enumeration is inexact on a nonzero fraction (the point of the axis)."""
    for m in (3, 4):
        u_nom, a, b, bounds = _rows(m, 6000, 7 + m)
        sat = ~_empty_halfspace_box(a, b, bounds)
        base = _base_projection(u_nom, a, b, bounds, _P0)
        dual = _exact_dual(u_nom, a, b, bounds)
        enum, _ = _box_aware_projection(u_nom, base, a, b, bounds)
        ref = _brute_ref(u_nom, a, b, bounds)
        assert float((dual[sat] - ref[sat]).abs().amax()) <= TOL, f"dual inexact at m={m}"
        # dual output is always feasible (box + row) on satisfiable rows
        lhs = (a * dual).sum(1)
        assert bool(((lhs[sat] <= b[sat] + 1e-8)).all())
        assert bool((dual[sat] >= bounds[:, 0] - 1e-8).all() and (dual[sat] <= bounds[:, 1] + 1e-8).all())
        enum_bad = torch.linalg.norm(enum[sat] - ref[sat], dim=1) > TOL
        assert float(enum_bad.float().mean()) > 0.02, f"enumeration should differ at m={m}"


def test_g3_jacobian_no_clip_and_all_clip():
    """no-clip projected rows: Jacobian = I - a a^T/||a||^2 (rank m-1); fully-clipped rows: zero."""
    m = 4
    torch.manual_seed(3)
    low = torch.full((m,), -2.0, dtype=torch.float64); high = torch.full((m,), 2.0, dtype=torch.float64)
    bounds = torch.stack([low, high], dim=1)
    # rows that project with all coords interior: u_nom interior, a moderate, b slightly below a^T u_nom
    u_nom = (torch.rand(3000, m, dtype=torch.float64) - 0.5) * 2.0
    a = torch.randn(3000, m, dtype=torch.float64)
    b = (a * u_nom).sum(1) - 0.5
    un = u_nom.clone().requires_grad_(True)
    proj = _exact_dual(un, a, b, bounds)
    J = torch.stack([torch.autograd.grad(proj[:, i].sum(), un, retain_graph=True)[0] for i in range(m)], dim=1)
    clip = ((proj - low).abs() < 1e-7) | ((proj - high).abs() < 1e-7)
    noclip = clip.sum(1) == 0
    aa = a / torch.linalg.norm(a, dim=1, keepdim=True)
    E = torch.eye(m, dtype=torch.float64)[None] - torch.einsum("bi,bj->bij", aa, aa)
    assert int(noclip.sum()) > 100
    assert float((J.detach()[noclip] - E[noclip]).abs().amax()) <= 1e-6
    sv = torch.linalg.svdvals(J.detach()[noclip])
    assert bool((sv[:, m - 1] < 1e-6).all()) and bool((sv[:, : m - 1] > 1e-6).all())


def test_empty_row_bit_parity():
    """A1n invariant: on empty (unsatisfiable) rows, dual_solve returns the SAME least-violating
    action as the enumeration (the change is scoped to the feasible branch)."""
    u_nom, a, b, bounds = _rows(4, 8000, 11)
    # genuine empty rows that retain authority (exclude the all-singular L_g h = 0 set, which the
    # singular flag handles separately and which the epsilon=0 base-projection would send to NaN)
    empty = _empty_halfspace_box(a, b, bounds) & (a.abs().amax(1) > 1e-12)
    assert int(empty.sum()) > 50
    base = _base_projection(u_nom, a, b, bounds, _P0)
    enum, _ = _box_aware_projection(u_nom, base, a, b, bounds)
    dual, empty2 = _dual_solve_projection(u_nom, base, a, b, bounds)
    assert bool((empty2 == _empty_halfspace_box(a, b, bounds)).all())
    assert torch.equal(dual[empty], enum[empty])   # bit-parity of the least-violating action


def test_flag_dispatch_both_realizations():
    """filter.projection selects both realizations; unknown value raises."""
    u_nom, a, b, bounds = _rows(4, 256, 5)
    base = _base_projection(u_nom, a, b, bounds, _P0)
    d, _ = _select_projection("dual_solve", u_nom, base, a, b, bounds)
    e, _ = _select_projection("enumerate", u_nom, base, a, b, bounds)
    assert d.shape == u_nom.shape and e.shape == u_nom.shape
    try:
        _select_projection("nope", u_nom, base, a, b, bounds)
        assert False, "unknown projection should raise"
    except ValueError:
        pass


def test_hardnet_params_projection_default():
    """_hardnet_params reads filter.projection and defaults to dual_solve when absent."""
    cfg = {"filter": {"hardnet": {"epsilon": 0.0, "box_aware": True}, "alpha_safe": 2.0,
                      "alpha_unsafe": 100.0}, "env": {"dt": 0.05}}
    assert _hardnet_params(cfg).projection == "dual_solve"
    cfg["filter"]["projection"] = "enumerate"
    assert _hardnet_params(cfg).projection == "enumerate"
