"""v2.8.0 A — which branch the gradient-free set lives in (eval-only).

For each checkpoint, roll the deployed policy+filter (shipped fallback kstep phases1 k3, new terminal) on
the canonical pool and classify EVERY filtered step into feasible-branch (subdivided as S1 did: no clipped
coord / 0<|A|<m-1 / |A|=m-1 / |A|=m) and empty-branch (subdivided by whether the least-violating argmin is a
box vertex). Then attribute the total zero-Jacobian set to each branch.

Zero-Jacobian definition (from S1 G3, which established the class->gradient map):
  - dual_solve: |A|>=m-1  (|A|=m box vertex has zero Jacobian; |A|=m-1 has one free coord pinned by the
    equality a^T u = b, a function of b and the clipped bounds, not of u_nom -> zero Jacobian);
  - enumerate:  |A|=m     (the enumeration is discontinuous / vertex-valued only at the box vertex).
  The empty branch returns the least-violating enumeration argmin, a discontinuous vertex-valued map -> its
  Jacobian is zero on essentially all empty rows (S1 A1n: empty vertex share ~1.0). Each arm is classified
  under its DEPLOYED projection.

POPULATION CAVEAT (stated in the report): this is measured on the EVALUATION state distribution at a fixed
checkpoint; P2's 0.28->0.15 is on the TRAINING state distribution from the replay buffer. Different
populations -- this decomposes the same classes, it does not replace P2, and the two numbers are not equated.

Sidecar: data/runs/v2.8.0/s3_eval/branch_decomp.json"""
from __future__ import annotations
import copy, json
from pathlib import Path
import numpy as np, torch
from src.eval.run_full import _load_framework
from src.eval.build_pools import load_pool
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.common.rk4 import rk4_step
from src.common.filter_hardnet import (_cbf_terms, _base_alpha, _base_projection,
    _empty_halfspace_box, _box_aware_projection, _dual_solve_projection)

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
POOL = REPO / "data/secured_data/pools/eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl"
OUT = REPO / "data/runs/v2.8.0/s3_eval"; OUT.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VTOL = 1e-6
DUAL = REPO / "data/runs/v2.8.0/set__20260731-015517__seed42/v2.8.0__jt__20260731-015517__seed42/checkpoints"
ENUM = REPO / "data/runs/v2.8.0/set__20260731-015707__seed42/v2.8.0__jt__20260731-015707__seed42/checkpoints"
CKPTS = [("dual_step42000", DUAL / "best.pt", "dual_solve"),
         ("dual_step34500", DUAL / "step_034500.pt", "dual_solve"),
         ("enum_step34500", ENUM / "step_034500.pt", "enumerate")]


def load(ckpt, proj):
    ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    filt = copy.deepcopy(ck["config"]["filter"])
    filt["empty_fallback"] = {"mode": "kstep", "phases": 1, "k": 3}; filt["projection"] = proj
    over = {"env": {"dt": 0.05, "stuck_window_steps": 60, "stuck_radius": 0.10, "band_collision_limit": 4.0,
                    "goal_angrate_radius": 0.48}, "eval": {"max_steps": 200, "dt_ctrl": 0.05}, "filter": filt}
    fw, cfg, _ = _load_framework(str(ckpt), config_overrides=over)
    for n in ("value_net", "policy_net"):
        m = getattr(fw, n, None); m.to(DEV) if m is not None else None
    return fw


def capture(fw, proj, n_scenes=2000, max_steps=200):
    system, hf = fw.system, fw._filter
    h_fn, params = hf.h_fn, hf.params
    scenes = load_pool(POOL).scenes[:n_scenes]
    bscene = batch_scenes(scenes, device=DEV, dtype=torch.float32)
    x = initial_states_from_batch(bscene)
    bounds = system.u_bounds.to(DEV, torch.float32)
    P = type("P", (), {"epsilon": 0.0, "lg_reg_eps": 0.0})()
    empties, nclips = [], []
    for _ in range(max_steps):
        with torch.no_grad():
            u_nom = fw.policy(x, bscene)
        h, lf, lg = _cbf_terms(system, h_fn, x, bscene, u_nom, create_graph=False)
        row = -lf - _base_alpha(h, params) * h
        with torch.no_grad():
            empty = _empty_halfspace_box(lg, row, bounds)
            base = _base_projection(u_nom, lg, row, bounds, P)
            proj_u = (_dual_solve_projection if proj == "dual_solve" else _box_aware_projection)(
                u_nom, base, lg, row, bounds)[0]
            low, high = bounds[:, 0], bounds[:, 1]
            nclip = (((proj_u - low).abs() < VTOL) | ((proj_u - high).abs() < VTOL)).sum(dim=1)
            empties.append(empty.cpu()); nclips.append(nclip.cpu())
            u_safe, _ = fw.filter(x, u_nom, bscene)
            x = system.wrap_state(rk4_step(system, x, u_safe, 0.05))
    return torch.cat(empties).numpy().astype(bool), torch.cat(nclips).numpy()


def decompose(empty, nclip, proj, m=4):
    N = empty.size; feas = ~empty
    frac = lambda mask: float(mask.sum()) / N
    classes = {
        "feasible_noclip": frac(feas & (nclip == 0)),
        "feasible_partial_0_to_m2": frac(feas & (nclip > 0) & (nclip < m - 1)),
        "feasible_m_minus_1": frac(feas & (nclip == m - 1)),
        "feasible_vertex_m": frac(feas & (nclip == m)),
        "empty_vertex_m": frac(empty & (nclip == m)),
        "empty_nonvertex": frac(empty & (nclip < m)),
    }
    zj = (nclip >= m - 1) if proj == "dual_solve" else (nclip == m)
    zj_n = int(zj.sum())
    return {
        "n_steps": int(N), "projection": proj,
        "empty_step_fraction": frac(empty),
        "class_shares_of_all_filtered_steps": classes,
        "zero_jac_definition": ("|A|>=m-1" if proj == "dual_solve" else "|A|=m"),
        "zero_jac_total_fraction": frac(zj),
        "zero_jac_n": zj_n,
        "zero_jac_empty_branch_share": float((zj & empty).sum()) / zj_n if zj_n else None,
        "zero_jac_feasible_branch_share": float((zj & feas).sum()) / zj_n if zj_n else None,
        "zero_jac_empty_fraction_of_all": frac(zj & empty),
        "zero_jac_feasible_fraction_of_all": frac(zj & feas),
    }


rep = {"pool": "0ef3751b", "fallback": "kstep phases1 k3", "terminal": "new (omega_G=0.48)", "device": str(DEV),
       "population_caveat": "eval state distribution at a fixed checkpoint; P2's 0.15 is the training "
                            "replay-buffer distribution -- different populations, not to be equated.",
       "checkpoints": {}}
for name, ckpt, proj in CKPTS:
    fw = load(ckpt, proj)
    empty, nclip = capture(fw, proj)
    d = decompose(empty, nclip, proj)
    rep["checkpoints"][name] = d
    c = d["class_shares_of_all_filtered_steps"]
    print(f"[{name}] {proj}: empty_frac {d['empty_step_fraction']:.4f} | "
          f"feas(noclip {c['feasible_noclip']:.3f}/partial {c['feasible_partial_0_to_m2']:.3f}/"
          f"m-1 {c['feasible_m_minus_1']:.3f}/vertex {c['feasible_vertex_m']:.3f}) "
          f"empty(vertex {c['empty_vertex_m']:.3f}/nonvertex {c['empty_nonvertex']:.4f})", flush=True)
    print(f"    zero-Jac total {d['zero_jac_total_fraction']:.4f} -> empty-branch {d['zero_jac_empty_branch_share']:.3f} "
          f"/ feasible-branch {d['zero_jac_feasible_branch_share']:.3f}", flush=True)
(OUT / "branch_decomp.json").write_text(json.dumps(rep, indent=2) + "\n")
print("-> ", OUT / "branch_decomp.json")
