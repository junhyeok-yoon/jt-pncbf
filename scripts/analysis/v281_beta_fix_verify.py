"""v2.8.1 S1 beta-screen — acceptance test for the log-domain soft_topk fix.

Checks the Researcher's required properties: finite-difference agreement (true-derivative) across a tie battery
and the 2-50 beta range; finite + bounded gradients at every finite beta (the old encoder exploded to ~1e24);
per-slot gradient liveness; inactive-slot non-participation; beta->inf still exactly top_k_obstacles; and MEASURES
the beta=2.0 forward movement vs the old (buggy) encoder."""
import numpy as np
import torch

from src.common.observation import (soft_topk_obstacles, top_k_obstacles, _distance_kernel, SOFT_DC, SOFT_INNER)

DT = torch.float64  # FD agreement in float64 (clean); float32 finiteness checked separately


def soft_topk_OLD(positions, centers, radii, active, k, beta, d_c=SOFT_DC, inner=SOFT_INNER):
    """The pre-fix buggy encoder (division normaliser + subtractive multiplicative residual), for comparison."""
    B = positions.shape[0]
    cb = centers.unsqueeze(0).expand(B, -1, -1) if centers.ndim == 2 else centers
    rb = radii.unsqueeze(0).expand(B, -1) if radii.ndim == 1 else radii
    ab = active.unsqueeze(0).expand(B, -1) if active.ndim == 1 else active
    rel = cb - positions.unsqueeze(1)
    sd = torch.linalg.norm(rel, dim=-1) - rb
    rel_m = torch.where(ab.unsqueeze(-1), rel, torch.zeros_like(rel))
    rad_m = torch.where(ab, rb, torch.zeros_like(rb))
    sigma = torch.where(ab, _distance_kernel(sd, inner, d_c), torch.zeros_like(sd))
    neg = torch.finfo(sd.dtype).min / 4.0
    logit = torch.where(ab, -beta * sd, torch.full_like(sd, neg))
    r = ab.to(sd.dtype); rels = []
    for _ in range(k):
        w = torch.softmax(logit, dim=1); pw = r * w
        p = pw / pw.sum(dim=1, keepdim=True).clamp_min(1e-30)
        rels.append(((p * sigma).unsqueeze(-1) * rel_m).sum(1))
        r = (r * (1.0 - p)).clamp_min(0.0)
    return torch.stack(rels, 1)


def battery():
    S = {}
    S["dominant_near_far"] = (torch.tensor([[[2.39, 0.0], [5.2, 0.5], [0, 0], [0, 0], [0, 0]]], dtype=DT),
                              torch.full((1, 5), 0.4, dtype=DT), torch.tensor([[True, True, False, False, False]]))
    ang = torch.linspace(0, 2 * np.pi, 6, dtype=DT)[:5]
    tie5 = torch.stack([3.318 * torch.cos(ang), 3.318 * torch.sin(ang)], 1).unsqueeze(0)
    tie5 = torch.cat([torch.tensor([[[2.39, 0.0]]], dtype=DT), tie5[:, :4]], 1)  # 1 near + 4 tied
    S["near_plus_tie4"] = (tie5, torch.full((1, 5), 0.4, dtype=DT), torch.ones(1, 5, dtype=torch.bool))
    S["near_ties"] = (torch.tensor([[[1.0, 0], [1.0001, 0], [-1.0, .02], [2, 2], [-2, -2]]], dtype=DT),
                      torch.full((1, 5), 0.4, dtype=DT), torch.ones(1, 5, dtype=torch.bool))
    S["fewer_k"] = (torch.tensor([[[1.5, 0], [-2, 1], [0, 0], [0, 0], [0, 0]]], dtype=DT),
                    torch.full((1, 5), 0.4, dtype=DT), torch.tensor([[True, True, False, False, False]]))
    S["all_inactive"] = (torch.tensor([[[1.5, 0], [-2, 1], [0, 3], [2, 2], [-2, -2]]], dtype=DT),
                         torch.full((1, 5), 0.4, dtype=DT), torch.zeros(1, 5, dtype=torch.bool))
    return S


BETAS = [2.0, 4.0, 6.0, 8.0, 12.0, 20.0, 30.0, 50.0]
k = 5
print("=== FD agreement + finiteness (fixed) vs old-encoder gradient magnitude ===")
worst_fd = 0.0
for name, (c, r, a) in battery().items():
    for beta in BETAS:
        pos = torch.zeros(1, 2, dtype=DT, requires_grad=True)
        rel, _ = soft_topk_obstacles(pos, c, r, a, k, beta=beta, d_c=SOFT_DC, inner=SOFT_INNER)
        g = torch.autograd.grad(rel.sum(), pos, retain_graph=False)[0]
        # central finite difference
        fd = torch.zeros_like(pos)
        h = 1e-6
        for j in range(2):
            e = torch.zeros_like(pos); e[0, j] = h
            with torch.no_grad():
                fp = soft_topk_obstacles(pos + e, c, r, a, k, beta=beta, d_c=SOFT_DC, inner=SOFT_INNER)[0].sum()
                fm = soft_topk_obstacles(pos - e, c, r, a, k, beta=beta, d_c=SOFT_DC, inner=SOFT_INNER)[0].sum()
            fd[0, j] = (fp - fm) / (2 * h)
        err = float((g - fd).abs().max())
        worst_fd = max(worst_fd, err)
        posO = torch.zeros(1, 2, dtype=DT, requires_grad=True)
        relO = soft_topk_OLD(posO, c, r, a, k, beta=beta)
        gO = torch.autograd.grad(relO.sum(), posO)[0]
        gmax = float(g.abs().max()); gOmax = float(gO.abs().max())
        flag = "" if (np.isfinite(gmax) and gmax < 1e3 and err < 1e-3) else "  <-- CHECK"
        if beta in (2.0, 6.0, 50.0):
            print(f"  {name:<16} beta={beta:>4}: fixed|grad|max={gmax:9.4g} FDerr={err:.2e} | OLD|grad|max={gOmax:9.4g}{flag}")
print(f"worst FD error across battery x betas = {worst_fd:.2e}")

print("\n=== beta=2.0 forward movement (fixed vs old) ===")
mx = 0.0
for name, (c, r, a) in battery().items():
    pos = torch.zeros(1, 2, dtype=DT)
    with torch.no_grad():
        new = soft_topk_obstacles(pos, c, r, a, k, beta=2.0)[0]
        old = soft_topk_OLD(pos, c, r, a, k, beta=2.0)
    mx = max(mx, float((new - old).abs().max()))
print(f"  max|fixed - old| at beta=2.0 across battery = {mx:.3e}")

print("\n=== top_k limit (beta=inf, d_c=inf) delegation ===")
c, r, a = battery()["near_ties"]
pos = torch.zeros(1, 2, dtype=DT)
lim = soft_topk_obstacles(pos, c, r, a, k, beta=float("inf"), d_c=float("inf"))
hard = top_k_obstacles(pos, c, r, a, k)
print(f"  beta=inf,d_c=inf == top_k bit-wise: {bool(torch.equal(lim[0], hard[0]) and torch.equal(lim[1], hard[1]))}")

print("\n=== per-slot gradient liveness + inactive non-participation ===")
c, r, a = battery()["near_plus_tie4"]
for s in range(k):
    pos = torch.zeros(1, 2, dtype=DT, requires_grad=True)
    rel, _ = soft_topk_obstacles(pos, c, r, a, k, beta=6.0)
    g = torch.autograd.grad(rel[:, s].sum(), pos, retain_graph=True)[0]
    print(f"  slot {s}: |grad|={float(g.abs().max()):.4g} live={float(g.abs().max())>0}")
ci, ri, ai = battery()["all_inactive"]
pos = torch.zeros(1, 2, dtype=DT)
rel_i, rad_i = soft_topk_obstacles(pos, ci, ri, ai, k, beta=6.0)
print(f"  all-inactive: rel all-zero={bool((rel_i==0).all())} radii all-zero={bool((rad_i==0).all())}")

print("\n=== float32 finiteness/boundedness (the regime that crashed) ===")
for name in ("dominant_near_far", "near_plus_tie4"):
    c, r, a = battery()[name]
    c32, r32, a32 = c.float(), r.float(), a
    for beta in (6.0, 12.0, 50.0):
        pos = torch.zeros(1, 2, dtype=torch.float32, requires_grad=True)
        rel, _ = soft_topk_obstacles(pos, c32, r32, a32, k, beta=beta)
        g = torch.autograd.grad(rel.sum(), pos)[0]
        print(f"  {name:<16} beta={beta:>4} f32: |grad|max={float(g.abs().max()):.4g} finite={bool(torch.isfinite(g).all())}")
