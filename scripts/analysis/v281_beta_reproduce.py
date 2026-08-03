"""v2.8.1 S1 beta-screen Step 1 (Part B, definitive) — reproduce the beta=6 crash and dump the EXACT failing QP
case datum. Patches filter_cbfqp._solve_cbf_qp_case to, on the first non-SOLVED status, print (h, L_f h, L_g h,
row_upper, u_nom, bounds) finiteness + magnitudes — the row datum at the failing step — then re-raise."""
from pathlib import Path

import numpy as np

import src.common.filter_cbfqp as FQ

_orig = FQ._solve_cbf_qp_case
STATE = {"done": False}


def fin(v):
    return bool(np.isfinite(np.asarray(v, dtype=np.float64)).all())


def patched(u_nom, h, lf_h, lg_h, bounds, params):
    try:
        return _orig(u_nom, h, lf_h, lg_h, bounds, params)
    except RuntimeError as e:
        if not STATE["done"]:
            STATE["done"] = True
            alpha = params["alpha_safe"] if h <= 0.0 else params["alpha_unsafe"]
            row_upper = -float(lf_h) - alpha * (float(h) + params["v_shift"] + params["gamma_margin"])
            lg = np.asarray(lg_h, dtype=np.float64)
            print("\n==== EXACT FAILING QP CASE DATUM (beta=6) ====", flush=True)
            print(f"  raised: {str(e)[:70]}", flush=True)
            print(f"  h={float(h):.6g} fin={fin(h)} | L_f h={float(lf_h):.6g} fin={fin(lf_h)}", flush=True)
            print(f"  L_g h={lg} fin={fin(lg)} max|L_g|={np.abs(lg).max():.6g}", flush=True)
            print(f"  alpha={alpha} row_upper={row_upper:.6g} fin={fin(row_upper)}", flush=True)
            print(f"  u_nom={np.asarray(u_nom)} fin={fin(u_nom)} | bounds fin={fin(bounds)}", flush=True)
            print(f"  ALL DATUM FINITE = {fin(h) and fin(lf_h) and fin(lg) and fin(row_upper) and fin(u_nom) and fin(bounds)}", flush=True)
        raise


FQ._solve_cbf_qp_case = patched

import src.frameworks.oc_pncbf.train as T
_ocfg = T.load_effective_config


def cfg6():
    c = _ocfg()
    c["run"]["system"] = "quadrotor_3d"
    c["training"]["oc_pncbf"]["epochs"] = 100
    c["collection"]["inject_frac"] = 0.0
    c["collection"]["collector"] = "continuing"
    c["env"]["band_hazard"] = {"enabled": True, "limit": 4.0}
    c["env"]["band_collision_limit"] = 0.0
    c["filter"]["empty_fallback"] = {"mode": "none", "k": 10}
    c.setdefault("obs", {}).setdefault("quadrotor_3d", {}).update({"encoder": "soft_topk", "beta": 6.0})
    return c


T.load_effective_config = cfg6
from src.frameworks.oc_pncbf.train import run_training

try:
    run_training(stage="full", system="quadrotor_3d", seed=42, output_root=Path("data"), device="auto")
    print("run finished without crash (unexpected)")
except Exception as e:
    print(f"\nrun raised (as expected): {type(e).__name__}: {str(e)[:120]}", flush=True)
print("dumped =", STATE["done"])
