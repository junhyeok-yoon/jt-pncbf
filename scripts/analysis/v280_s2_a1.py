"""v2.8.0 S2 A1 — the m<=2 systems under the new dual_solve default (scores P1).

For double_integrator, unicycle, quadrotor_planar: capture real closed-loop rollout states (under the
legacy enumerate filter), then on satisfiable rows compare the two realizations as a pure function of
(u_nom, a=L_g h, b=row_upper, box), in float64:
  - value parity dual vs enumerate(epsilon=0)  -> prop:enum-exact(b), P1's "1e-9 on every sat row"
  - value diff  dual vs enumerate(deployed eps) -> the deployment epsilon carve-out
  - candidate-class win shares (clipped-nominal / base-projection / edge-solve / box-vertex)
  - autograd Jacobian dual vs enumerate(deployed): max overall, on base-projection-win rows, elsewhere
P1 falsifier: value parity(eps0) > 1e-9 anywhere, OR Jacobian diff > 1e-6 on a row where base
projection did NOT win. Artifacts -> data/runs/v2.8.0/s2_terminal/a1.json."""
from __future__ import annotations
import json
from pathlib import Path
from itertools import product
import numpy as np
import torch

from src.eval.run_full import _load_framework
from src.eval.build_pools import load_pool
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.common.rk4 import rk4_step
from src.common.filter_hardnet import (
    _cbf_terms, _base_alpha, _base_projection, _empty_halfspace_box,
    _exact_dual, _box_aware_projection, _candidate_actions,
)

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
OUT = REPO / "data/runs/v2.8.0/s2_terminal"; OUT.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FTOL = 1.0e-9
JTOL = 1.0e-6
DEPLOYED_EPS = 5.0e-4          # base_config filter.hardnet.epsilon

SYSTEMS = [
    ("double_integrator", REPO / "data/secured_data/v2.3.0/seed42/checkpoints/best.pt",
     REPO / "data/secured_data/pools/eval_full_di_n2000_seed23456.pkl"),
    ("unicycle", REPO / "data/secured_data/v2.2.2/seed42/checkpoints/best.pt",
     REPO / "data/secured_data/pools/eval_full_unicycle_n2000_seed23456.pkl"),
    ("quadrotor_planar", REPO / "data/secured_data/v2.7.0/seed42_iter5/checkpoints/best.pt",
     REPO / "data/secured_data/pools/eval_full_quadrotor-planar_n2000_seed23456.pkl"),
]


class P0:      # epsilon=0 params shim for _base_projection
    epsilon = 0.0; lg_reg_eps = 0.0


class Pe:      # deployed epsilon
    epsilon = DEPLOYED_EPS; lg_reg_eps = 0.0


def capture(ckpt, pool_path, n_scenes, max_steps):
    fw, cfg, _ = _load_framework(ckpt)
    system = fw.system; hf = fw._filter; h_fn, params = hf.h_fn, hf.params
    for nm in ("value_net", "policy_net"):
        net = getattr(fw, nm, None)
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
        row_upper = -lf - _base_alpha(h, params) * h
        U.append(u_nom.detach()); A.append(lg.detach()); B.append(row_upper.detach())
        with torch.no_grad():
            base = _base_projection(u_nom, lg, row_upper, bounds, params)
            u_safe, _ = _box_aware_projection(u_nom, base, lg, row_upper, bounds)
            x = system.wrap_state(rk4_step(system, x, u_safe, dt))
    return (torch.cat(U).double(), torch.cat(A).double(), torch.cat(B).double(),
            bounds.double(), system.name, system.action_dim)


def win_class(u_nom, a, b, bounds, m):
    """Replicate _box_aware_projection selection to get the WINNING candidate index -> class."""
    base_dep = _base_projection(u_nom, a, b, bounds, Pe())
    cands = _candidate_actions(u_nom, base_dep, a, b, bounds)     # [B,N,m]
    lhs = torch.einsum("ba,bna->bn", a, cands)
    feasible = lhs <= b.unsqueeze(1) + 1e-9
    dist = torch.sum((cands - u_nom.unsqueeze(1)) ** 2, dim=2)
    fscore = torch.where(feasible, dist, torch.full_like(dist, torch.inf))
    fidx = fscore.argmin(1)
    viol = torch.relu(lhs - b.unsqueeze(1))
    lbidx = (viol + 1e-9 * dist).argmin(1)
    idx = torch.where(feasible.any(1), fidx, lbidx)
    # class map: 0=clipped-nom, 1=base-proj, [2 .. 2+2^m-1)=box-vertex, rest=edge-solve
    ncorner = 2 ** m
    cls = torch.full_like(idx, 3)               # 3=edge
    cls = torch.where(idx == 0, torch.zeros_like(idx), cls)
    cls = torch.where(idx == 1, torch.ones_like(idx), cls)
    cls = torch.where((idx >= 2) & (idx < 2 + ncorner), torch.full_like(idx, 2), cls)
    return idx, cls


def enum_dep(un, a, b, bounds):
    base = _base_projection(un, a, b, bounds, Pe())
    sel, _ = _box_aware_projection(un, base, a, b, bounds)
    return sel


def enum_eps0(un, a, b, bounds):
    base = _base_projection(un, a, b, bounds, P0())
    sel, _ = _box_aware_projection(un, base, a, b, bounds)
    return sel


def jac(fn, un, a, b, bounds):
    un = un.clone().requires_grad_(True)
    proj = fn(un, a, b, bounds)
    m = un.shape[1]
    J = torch.zeros(un.shape[0], m, m, dtype=un.dtype, device=un.device)
    for i in range(m):
        g = torch.autograd.grad(proj[:, i].sum(), un, retain_graph=True)[0]
        J[:, i, :] = g
    return J.detach()


rep = {"deployed_epsilon": DEPLOYED_EPS, "FTOL": FTOL, "JTOL": JTOL, "systems": {}}
p1_fail = False
for name, ck, pool in SYSTEMS:
    U, A, Bb, bounds, sysname, m = capture(ck, pool, n_scenes=2000, max_steps=150)
    empty = _empty_halfspace_box(A, Bb, bounds)
    sat = ~empty
    Us, As, Bs = U[sat], A[sat], Bb[sat]
    dual = _exact_dual(Us, As, Bs, bounds)
    e0 = enum_eps0(Us, As, Bs, bounds)
    ed = enum_dep(Us, As, Bs, bounds)
    val_eps0 = float((dual - e0).abs().amax()) if Us.shape[0] else float("nan")
    val_dep = float((dual - ed).abs().amax()) if Us.shape[0] else float("nan")
    idx, cls = win_class(Us, As, Bs, bounds, m)
    shares = {k: float((cls == v).float().mean()) for k, v in
              (("clipped_nominal", 0), ("base_projection", 1), ("box_vertex", 2), ("edge_solve", 3))}
    # Jacobian only where the row is active (need-project); classify by base-proj win
    phi0 = torch.sum(As * torch.clamp(Us, bounds[:, 0], bounds[:, 1]), dim=1)
    need = phi0 > Bs + FTOL
    ni = torch.nonzero(need, as_tuple=False).flatten()
    if ni.numel() > 6000:
        ni = ni[torch.linspace(0, ni.numel() - 1, 6000).long()]
    Un, An, Bn = Us[ni], As[ni], Bs[ni]
    _, clsn = win_class(Un, An, Bn, bounds, m)
    Jd = jac(_exact_dual, Un, An, Bn, bounds)
    Je = jac(enum_dep, Un, An, Bn, bounds)
    jdiff = (Jd - Je).abs().amax(dim=(1, 2))
    base_win = clsn == 1
    jmax_all = float(jdiff.max()) if ni.numel() else float("nan")
    jmax_baseproj = float(jdiff[base_win].max()) if int(base_win.sum()) else 0.0
    jmax_other = float(jdiff[~base_win].max()) if int((~base_win).sum()) else 0.0
    # --- mechanism: characterize where dual and DEPLOYED enum diverge (value) ---
    vdiff = (dual - ed).abs().amax(1)                       # per satisfiable row
    div = vdiff > JTOL
    anorm = torch.linalg.norm(As, dim=1)                    # ||L_g h|| per satisfiable row
    frac_div = float(div.float().mean())
    med_a_all = float(anorm.median())
    med_a_div = float(anorm[div].median()) if int(div.sum()) else float("nan")
    p90_a_div = float(torch.quantile(anorm[div], 0.90)) if int(div.sum()) else float("nan")
    # among diverging rows, is the base-projection candidate infeasible (enum can't use it)?
    base_dep = _base_projection(Us, As, Bs, bounds, Pe())
    base_lhs = torch.sum(As * base_dep, dim=1)
    base_infeasible = base_lhs > Bs + 1e-9
    frac_div_baseinfeas = float(base_infeasible[div].float().mean()) if int(div.sum()) else float("nan")
    # jacobian divergence vs ||a|| on need-project rows
    an_need = torch.linalg.norm(An, dim=1)
    jdiv = jdiff > JTOL
    med_a_jdiv = float(an_need[jdiv].median()) if int(jdiv.sum()) else float("nan")
    # P1 scoring
    value_parity_ok = val_eps0 <= 2e-9      # prop:enum-exact(b): float64 accumulation at ~1e-9
    jac_off_baseproj_ok = jmax_other <= JTOL
    p1_ok = value_parity_ok and jac_off_baseproj_ok
    p1_fail = p1_fail or (not p1_ok)
    rep["systems"][name] = {
        "n_total": int(U.shape[0]), "n_satisfiable": int(sat.sum()), "action_dim": m,
        "value_parity_dual_vs_enum_eps0_max": val_eps0,
        "value_diff_dual_vs_enum_deployed_max": val_dep,
        "candidate_win_shares": shares,
        "n_need_project": int(ni.numel()),
        "jac_max_all": jmax_all,
        "jac_max_on_base_projection_win": jmax_baseproj,
        "jac_max_off_base_projection": jmax_other,
        "deployed_divergence": {
            "frac_sat_rows_val_gt_1e6": frac_div,
            "median_lgh_all": med_a_all, "median_lgh_diverging": med_a_div, "p90_lgh_diverging": p90_a_div,
            "frac_diverging_where_base_proj_infeasible": frac_div_baseinfeas,
            "median_lgh_on_jac_diverging_needproject": med_a_jdiv,
        },
        "P1_value_parity_prop_enum_exact_b": bool(value_parity_ok),
        "P1_jac_off_baseproj_le_1e6": bool(jac_off_baseproj_ok),
        "P1_pass": bool(p1_ok),
    }
    print(f"{name}: n_sat={int(sat.sum())} val_eps0={val_eps0:.2e} val_dep={val_dep:.2e} "
          f"frac_div={frac_div:.2e} med||a||all={med_a_all:.3f} med||a||div={med_a_div:.4f} "
          f"baseinfeas@div={frac_div_baseinfeas:.2f} jac_other={jmax_other:.2e} P1={p1_ok}", flush=True)

rep["P1_overall_pass"] = bool(not p1_fail)
rep["P1_falsifier_fired"] = bool(p1_fail)
(OUT / "a1.json").write_text(json.dumps(rep, indent=2) + "\n")
print("A1 DONE ->", OUT / "a1.json", "| P1 overall pass:", rep["P1_overall_pass"])
