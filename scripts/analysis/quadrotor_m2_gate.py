"""v2.6.0 Stage 1 M2 — GATE: does the LEARNED V_hat keep first-order authority on B_0? (P1-learned / O3)

On near-B_0 states, measure ||L_g V_hat|| for the learned M1 value and compare to the EXACT position-only
V^{phi,pi} (Stage-0: ~84% degenerate) on the SAME states, and to exact h_star (Stage-0: ~0%). Gate:
  (1) learned degenerate fraction (||L_g V_hat|| < eps_g=0.05) MATERIALLY below position-only's ~84%;
  (2) the learned ||L_g V_hat|| distribution not collapsed toward the starved 0.0072 regime (median >> it).
HALT if either fails (theory-note O3: epsilon_g not holding => joint training would stall).

Usage: python scripts/analysis/quadrotor_m2_gate.py <run_dir> [ckpt=best.pt]
"""
import json
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from src.common.filter_hardnet import _cbf_terms, _SINGULAR_LG_THRESHOLD
from src.common.quadrotor_barrier import make_exact_value_fn, phi_value, sample_near_B0_states
from src.common.value_net import ValueNetEnsemble, make_h_fn
from src.frameworks.jt_pncbf.train import make_system

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
SP = Path("/tmp/claude-1000/-home-junhyeok-MIT-jt-pncbf/31d93785-ac11-4206-bf50-a4c3de145dff/scratchpad")
EPS_G = 0.05
STARVED = 0.0072


def _lg(system, fn, x, scene):
    _, _, lg = _cbf_terms(system, fn, x, scene, torch.zeros(x.shape[0], 2, device=x.device, dtype=x.dtype),
                          create_graph=False)
    return torch.linalg.norm(lg.detach(), dim=1)


def _stats(t):
    a = t.cpu().numpy()
    return dict(median=float(np.median(a)), min=float(a.min()), p1=float(np.percentile(a, 1)),
                p5=float(np.percentile(a, 5)), p10=float(np.percentile(a, 10)),
                degen_frac=float((a < EPS_G).mean()))


def main():
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else max(REPO.glob("data/v2.6.0__*seed42"))
    ckpt = sys.argv[2] if len(sys.argv) > 2 else "best.pt"
    ck = torch.load(run_dir / "checkpoints" / ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    system = make_system(cfg)
    system.u_bounds = system.u_bounds.to(device=dev, dtype=dtype)
    vnet = ValueNetEnsemble(system.obs_dim, cfg).to(device=dev, dtype=dtype)
    vnet.load_state_dict(ck["v_s_state"]); vnet.eval()
    h_scale = float(cfg["env"]["h_scale"]); dt = float(cfg["env"]["dt"]); c = float(cfg["env"]["quadrotor_planar"]["c_gain"])

    g = torch.Generator(device=dev).manual_seed(20260715)
    xc, sc = sample_near_B0_states(system, cfg, 40000, dev, dtype, g)
    # TRUE near-B_0 = {|V_hat|<0.1} INT {|phi|<0.1} (amendment: B_0={V_hat=0} INT {h=0}). Filter to the
    # states where the LEARNED value is marginal AND the barrier is at the boundary; measuring ||L_g V||
    # on all near-boundary candidates (incl. deep-safe flat-V_hat states) spuriously inflates degeneracy.
    hL = make_h_fn(vnet, system)
    with torch.no_grad():
        Vhat0 = hL(xc, sc).reshape(-1)
        phi0 = phi_value(xc, sc, h_scale)
    near = (Vhat0.abs() < 0.10) & (phi0.abs() < 0.10)
    idx = near.nonzero(as_tuple=True)[0]
    x = xc[idx]
    scene = type(sc)(obstacle_centers=sc.obstacle_centers[idx], obstacle_radii=sc.obstacle_radii[idx],
                     obstacle_active=sc.obstacle_active[idx], goal=sc.goal[idx])
    n_true = int(x.shape[0])

    lg_learned = _lg(system, hL, x, scene)                                            # learned V_hat
    lg_phi = _lg(system, make_exact_value_fn(system, c, h_scale, dt, 25, position_only=True), x, scene)   # exact phi
    lg_hstar = _lg(system, make_exact_value_fn(system, c, h_scale, dt, 25, position_only=False), x, scene)  # exact h_star

    sL, sP, sH = _stats(lg_learned), _stats(lg_phi), _stats(lg_hstar)
    print(f"run: {run_dir.name}  ckpt: {ckpt}  step: {ck.get('step')}  candidates: 40000  true near-B_0 (|V_hat|<0.1 & |phi|<0.1): {n_true}")
    print(f"  exact phi (position-only): median={sP['median']:.4f}  DEGEN_frac={sP['degen_frac']:.3f}  (Stage-0 ~0.84)")
    print(f"  exact h_star (c={c}):       median={sH['median']:.4f}  DEGEN_frac={sH['degen_frac']:.3f}  (Stage-0 ~0.00)")
    print(f"  LEARNED V_hat:             median={sL['median']:.4f}  min={sL['min']:.5f}  p1={sL['p1']:.5f} "
          f"p5={sL['p5']:.5f}  DEGEN_frac(<{EPS_G})={sL['degen_frac']:.3f}")

    cond1 = sL["degen_frac"] < 0.10                       # materially below position-only ~0.84
    # not starved: the DISTRIBUTION is not collapsed toward the v5 0.0072 regime (median + p10 well above
    # it). A single min outlier is not starvation; starvation = the bulk collapsed.
    cond2 = sL["median"] > 10 * STARVED and sL["p10"] > STARVED
    P1_learned = cond1 and cond2
    print(f"\n  (1) learned degen_frac {sL['degen_frac']:.3f} < 0.10 (<< phi {sP['degen_frac']:.3f}): {'PASS' if cond1 else 'FAIL'}")
    print(f"  (2) not starved: median {sL['median']:.4f} > {10*STARVED:.4f} & p10 {sL['p10']:.5f} > {STARVED}: {'PASS' if cond2 else 'FAIL'}")
    print(f"\nM2 (P1-learned / O3) GATE: {'PASS' if P1_learned else 'FAIL -> HALT'}")

    json.dump({"run": run_dir.name, "ckpt": ckpt, "step": ck.get("step"), "eps_g": EPS_G,
               "learned": sL, "exact_phi": sP, "exact_hstar": sH,
               "cond1_degen": cond1, "cond2_notstarved": cond2, "P1_learned": bool(P1_learned)},
              open(SP / "quadrotor_m2_gate.json", "w"), indent=2)
    print("saved:", SP / "quadrotor_m2_gate.json")


if __name__ == "__main__":
    main()
