"""v2.8.0 S1 correctness gates for the exact dual-solve projection (prop:lambda-solve).

G1  m=2 parity   : on real planar rollout states with a satisfiable row, dual == enumerate to 1e-9
                   (prop:enum-exact(b) makes them equal at m<=2). FAIL -> HALT.
G2  m=4 exactness: on real 3-D rollout states, dual matches an independent brute-force active-set
                   projection to 1e-9; report the enumeration's error distribution (the empirical
                   content of prop:enum-exact(b) at the deployed operating point).
G3  Jacobian     : autograd d u / d u_nom vs central finite differences; the three clip classes.

Real states are captured closed-loop with the legacy (enumerate) filter. The projection comparison
is a pure function of (u_nom, a=L_g h, b=row_upper, box); it is done in float64 for the 1e-9 gates.
Artifacts persisted under data/runs/v2.8.0/s1_projection/."""
from __future__ import annotations
import itertools, json
from pathlib import Path
import numpy as np
import torch

from src.eval.run_full import _load_framework
from src.eval.build_pools import load_pool
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.common.rk4 import rk4_step
from src.common.filter_hardnet import (
    _cbf_terms, _base_alpha, _base_projection, _select_projection,
    _empty_halfspace_box, _exact_dual, _box_aware_projection,
)

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
OUT = REPO / "data/runs/v2.8.0/s1_projection"; OUT.mkdir(parents=True, exist_ok=True)
PLANAR_CK = REPO / "data/secured_data/v2.7.0/seed42/checkpoints/best.pt"
PLANAR_POOL = REPO / "data/secured_data/pools/eval_full_quadrotor-planar_n2000_seed23456.pkl"
JT3D = REPO / "data/runs/v2.7.6/set__20260725-043415__seed42/v2.7.6__jt__20260725-052127__seed42/checkpoints/step_042000.pt"
POOL3D = REPO / "data/eval_pools/eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FTOL = 1.0e-9


def capture(ckpt, pool_path, n_scenes, max_steps):
    """Closed-loop rollout under the legacy enumerate filter; return real (u_nom, a, b) at every step."""
    fw, cfg, _ = _load_framework(ckpt)
    system = fw.system
    hf = fw._filter
    h_fn, params = hf.h_fn, hf.params
    for name in ("value_net", "policy_net"):
        net = getattr(fw, name, None)
        if net is not None:
            net.to(DEV)
    scenes = load_pool(pool_path).scenes[:n_scenes]
    bscene = batch_scenes(scenes, device=DEV, dtype=torch.float32)
    x = initial_states_from_batch(bscene)
    bounds = system.u_bounds.to(device=DEV, dtype=torch.float32)
    dt = float(params.dt)
    U, A, B = [], [], []
    for _ in range(max_steps):
        with torch.no_grad():
            u_nom = fw.policy(x, bscene)
        h, lf, lg = _cbf_terms(system, h_fn, x, bscene, u_nom, create_graph=False)
        alpha = _base_alpha(h, params)
        row_upper = -lf - alpha * h
        U.append(u_nom.detach()); A.append(lg.detach()); B.append(row_upper.detach())
        with torch.no_grad():
            base = _base_projection(u_nom, lg, row_upper, bounds, params)
            u_safe, _ = _select_projection("enumerate", u_nom, base, lg, row_upper, bounds)
            x = system.wrap_state(rk4_step(system, x, u_safe, dt))
    return (torch.cat(U).double(), torch.cat(A).double(), torch.cat(B).double(),
            bounds.double(), system.name)


def brute_ref(u_nom, a, b, bounds):
    """Independent exact projection onto {a^T u <= b} ∩ box: 3^m active-set enumeration (lambda from the
    interior equality) + the constraint-inactive candidate clip(u_nom); min-distance feasible. Vectorized."""
    Bn, m = u_nom.shape
    low, high = bounds[:, 0], bounds[:, 1]
    codes = torch.tensor(list(itertools.product((0, 1, 2), repeat=m)), device=u_nom.device)  # [P,m] 0=lo,1=hi,2=int
    interior = (codes == 2)[None]                                             # [1,P,m]
    boundv = torch.where(codes == 0, low, high).double()[None]                # [1,P,m]
    un, av = u_nom[:, None, :], a[:, None, :]
    w = torch.where(interior, un, boundv)
    num = (av * w).sum(-1) - b[:, None]                                       # [B,P]
    den = torch.where(interior, av * av, torch.zeros_like(av)).sum(-1)        # [B,P]
    lam = num / den.clamp_min(1e-300)
    u = torch.where(interior, un - lam[:, :, None] * av, boundv)              # [B,P,m]
    boxok = (u >= low - 1e-9).all(-1) & (u <= high + 1e-9).all(-1)
    consok = (av * u).sum(-1) <= b[:, None] + 1e-7
    valid = interior.any(-1) & (den > 1e-15) & boxok & consok
    dist = torch.where(valid, ((u - un) ** 2).sum(-1), torch.full_like(num, float("inf")))
    best = dist.argmin(1)
    rows = torch.arange(Bn, device=u_nom.device)
    ref = u[rows, best]
    bestd = dist[rows, best]
    c0 = torch.clamp(u_nom, low, high)
    c0ok = (a * c0).sum(-1) <= b + 1e-9
    d0 = ((c0 - u_nom) ** 2).sum(-1)
    use0 = c0ok & (d0 <= bestd)
    return torch.where(use0[:, None], c0, ref)


def jac_autograd(fn, un, a, b, bounds):
    un = un.clone().requires_grad_(True)
    proj = fn(un, a, b, bounds)
    m = un.shape[1]
    J = torch.zeros(un.shape[0], m, m, dtype=un.dtype, device=un.device)
    for i in range(m):
        g = torch.autograd.grad(proj[:, i].sum(), un, retain_graph=True)[0]
        J[:, i, :] = g
    return J.detach(), proj.detach()


def jac_fd(fn, un, a, b, bounds, eps=1e-6):
    m = un.shape[1]
    J = torch.zeros(un.shape[0], m, m, dtype=un.dtype, device=un.device)
    for k in range(m):
        e = torch.zeros_like(un); e[:, k] = eps
        with torch.no_grad():
            J[:, :, k] = (fn(un + e, a, b, bounds) - fn(un - e, a, b, bounds)) / (2 * eps)
    return J


def enum_fn(un, a, b, bounds):
    base = _base_projection(un, a, b, bounds,
                            type("P", (), {"epsilon": 0.0, "lg_reg_eps": 0.0})())
    sel, _ = _box_aware_projection(un, base, a, b, bounds)
    return sel


report = {}

# ---------------- G1: m=2 parity ----------------
U, A, Bb, bounds, name = capture(PLANAR_CK, PLANAR_POOL, n_scenes=2000, max_steps=200)
empty = _empty_halfspace_box(A, Bb, bounds)
sat = ~empty
Us, As, Bs = U[sat], A[sat], Bb[sat]
base_s = _base_projection(Us, As, Bs, bounds, type("P", (), {"epsilon": 0.0, "lg_reg_eps": 0.0})())
dual_s = _exact_dual(Us, As, Bs, bounds)
enum_s, _ = _box_aware_projection(Us, base_s, As, Bs, bounds)
g1_max = float((dual_s - enum_s).abs().amax()) if Us.shape[0] else float("nan")
report["G1"] = {"system": name, "n_total": int(U.shape[0]), "n_satisfiable": int(sat.sum()),
                "max_abs_dual_minus_enum": g1_max, "pass": bool(g1_max <= FTOL)}
np.savez_compressed(OUT / "g1_planar.npz", diff=(dual_s - enum_s).abs().amax(1).cpu().numpy())
print(f"G1 planar: n_sat={int(sat.sum())} max|dual-enum|={g1_max:.3e} pass={report['G1']['pass']}")
if not report["G1"]["pass"]:
    bad = ((dual_s - enum_s).abs().amax(1) > FTOL)
    print(f"G1 HALT: {int(bad.sum())} rows disagree; e.g. max {float((dual_s-enum_s).abs().amax()):.3e}")
    (OUT / "gates.json").write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit("G1 FAILED — HALT (see report)")

# ---------------- G2: m=4 exactness + enumeration error distribution ----------------
U, A, Bb, bounds, name = capture(JT3D, POOL3D, n_scenes=2000, max_steps=40)
empty = _empty_halfspace_box(A, Bb, bounds)
sat = ~empty
Us, As, Bs = U[sat], A[sat], Bb[sat]
base_s = _base_projection(Us, As, Bs, bounds, type("P", (), {"epsilon": 0.0, "lg_reg_eps": 0.0})())
dual_s = _exact_dual(Us, As, Bs, bounds)
enum_s, _ = _box_aware_projection(Us, base_s, As, Bs, bounds)
ref = brute_ref(Us, As, Bs, bounds)
dual_err = (dual_s - ref).abs().amax(1)
enum_err = torch.linalg.norm(enum_s - ref, dim=1)          # ||u_enum - P||
dual_ref_max = float(dual_err.max())
enum_bad = enum_err > FTOL
# sanity: is the reference trustworthy? (dual, an independent exact solver, matches it)
ref_trustworthy = dual_ref_max <= 1e-7
report["G2"] = {
    "system": name, "n_satisfiable": int(sat.shape[0] if sat.ndim == 0 else sat.sum()),
    "n_rows_compared": int(Us.shape[0]),
    "max_abs_dual_minus_ref": dual_ref_max, "dual_pass": bool(dual_ref_max <= FTOL),
    "reference_trustworthy": bool(ref_trustworthy),
    "enum_frac_diff_gt_1e9": float(enum_bad.float().mean()),
    "enum_err_median_on_bad": float(enum_err[enum_bad].median()) if int(enum_bad.sum()) else 0.0,
    "enum_err_p90_on_bad": float(torch.quantile(enum_err[enum_bad], 0.90)) if int(enum_bad.sum()) else 0.0,
    "enum_err_max": float(enum_err.max()),
}
np.savez_compressed(OUT / "g2_3d.npz", dual_err=dual_err.cpu().numpy(), enum_err=enum_err.cpu().numpy())
print(f"G2 3D: n={Us.shape[0]} max|dual-ref|={dual_ref_max:.3e} dual_pass={report['G2']['dual_pass']} "
      f"enum_frac>1e-9={report['G2']['enum_frac_diff_gt_1e9']:.3f} "
      f"enum_err median/p90(bad)={report['G2']['enum_err_median_on_bad']:.3e}/{report['G2']['enum_err_p90_on_bad']:.3e}")

# ---------------- G3: Jacobian structure (need-project rows) ----------------
phi0 = torch.sum(As * torch.clamp(Us, bounds[:, 0], bounds[:, 1]), dim=1)
need = phi0 > Bs + FTOL
idx = torch.nonzero(need, as_tuple=False).flatten()
if idx.numel() > 4000:
    idx = idx[torch.linspace(0, idx.numel() - 1, 4000).long()]
Ug, Ag, Bg = Us[idx], As[idx], Bs[idx]
m = Ug.shape[1]
I = torch.eye(m, dtype=Ug.dtype, device=Ug.device)


def classify(proj):
    low, high = bounds[:, 0], bounds[:, 1]
    at = (proj - low).abs().min(  # distance to nearest bound
        )
    clip = ((proj - low).abs() < 1e-7) | ((proj - high).abs() < 1e-7)
    nclip = clip.sum(1)
    return nclip


g3 = {}
for label, fn in (("dual", _exact_dual), ("enumerate", enum_fn)):
    J, proj = jac_autograd(fn, Ug, Ag, Bg, bounds)
    Jfd = jac_fd(fn, Ug.clone(), Ag, Bg, bounds)
    fd_max = float((J - Jfd).abs().amax())
    nclip = classify(proj)
    noclip = nclip == 0; partial = (nclip > 0) & (nclip < m); allclip = nclip == m
    # no-clip: J == I - a a^T/||a||^2, rank m-1
    aa = Ag / torch.linalg.norm(Ag, dim=1, keepdim=True)
    E = I[None] - torch.einsum("bi,bj->bij", aa, aa)
    noclip_err = float((J[noclip] - E[noclip]).abs().amax()) if int(noclip.sum()) else 0.0
    Jn = J[noclip]
    if Jn.shape[0]:
        sv = torch.linalg.svdvals(Jn)                      # [n,m]
        rank_ok = bool(((sv[:, :m - 1] > 1e-6).all(1) & (sv[:, m - 1] < 1e-6)).float().mean() > 0.99)
    else:
        rank_ok = True
    partial_nz = float((torch.linalg.matrix_norm(J[partial]) > 1e-9).float().mean()) if int(partial.sum()) else 1.0
    allclip_zero = float((torch.linalg.matrix_norm(J[allclip]) < 1e-9).float().mean()) if int(allclip.sum()) else 1.0
    g3[label] = {
        "n": int(idx.numel()),
        "share_no_clip": float(noclip.float().mean()),
        "share_partial": float(partial.float().mean()),
        "share_all_clip": float(allclip.float().mean()),
        "no_clip_jac_err_vs_projector": noclip_err,
        "no_clip_rank_m_minus_1_ok": rank_ok,
        "partial_nonzero_share": partial_nz,
        "all_clip_zero_share": allclip_zero,
        "autograd_vs_fd_max": fd_max,
    }
    print(f"G3 {label}: no-clip {g3[label]['share_no_clip']:.3f} partial {g3[label]['share_partial']:.3f} "
          f"all-clip {g3[label]['share_all_clip']:.3f} | no-clip Jac err {noclip_err:.2e} rank_ok {rank_ok} "
          f"| partial_nz {partial_nz:.3f} allclip_zero {allclip_zero:.3f} | fd_max {fd_max:.2e}")
report["G3"] = g3

(OUT / "gates.json").write_text(json.dumps(report, indent=2) + "\n")
print("GATES DONE ->", OUT / "gates.json")
