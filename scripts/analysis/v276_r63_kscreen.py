"""v2.7.6 R6.3 M1 — tilt-conditioned floor k-screen.

For k in {0,0.5,1,1.5,2}: build the tilt-floor eval-IC pool (seed 42), verify the bounds (0 violations,
never below -4), and run the M0 attitude-aware instrument. k* = smallest k with flagged fraction <= 1/2000.
Also re-checks the inertness gates (canonical seed23456 and the R6.1 v_z pool must still reproduce
bit-identically after the sampler edit). ic_eval_z is injected in-memory only (no config promotion).
No git, no securing.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml

from src.eval.build_pools import PoolSpec, build_pool, write_pool, load_pool, pool_variant
from src.envs.scene_init import _shoemake_quat  # noqa: F401 (kept for parity; tilt via quat below)
from scripts.analysis.v276_attitude_feasibility import check_pool, MODEL_CONSTANTS

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
SYSTEM = "quadrotor_3d"
OUT = REPO / "data/runs/v2.7.6/pools"
OUTK = REPO / "data/runs/v2.7.6/r63_kscreen"; OUTK.mkdir(parents=True, exist_ok=True)
REG = json.loads((REPO / "data/runs/v2.7.6/registered_params.json").read_text())
K_GRID = REG["R6.3"]["M1_k_grid"]
K_THRESH = 1                         # flagged <= 1 of 2000
CANON_SHA = "0ef3751b"; R61_VZ_SHA = "220fb1e6"


def _merged():
    b = yaml.safe_load((REPO / "src/configs/base_config.yaml").read_text())
    e = yaml.safe_load((REPO / "src/configs/exp_config.yaml").read_text())
    def m(a, o):
        d = dict(a)
        for k, v in o.items():
            d[k] = m(d[k], v) if isinstance(v, Mapping) and isinstance(d.get(k), Mapping) else v
        return d
    return m(b, e)


def _tilt_cfg(cfg, k):
    cfg = copy.deepcopy(cfg)
    cfg["env"]["quadrotor_3d"]["ic_eval_z"] = {
        "floor_mode": "tilt", "tilt_floor_base": -4.0, "tilt_floor_k": float(k), "twr": 2.0,
        "z_ceiling_base": 3.9, "z_ceiling_v_up_slope": 0.5, "goal_z_half": 3.5}
    return cfg


def _vz_cfg(cfg):
    cfg = copy.deepcopy(cfg)
    cfg["env"]["quadrotor_3d"]["ic_eval_z"] = REG["ic_eval_z_bounds"]  # R6.1 v_z block (floor_mode default vz)
    return cfg


def verify_bounds(pool, k):
    viol = 0
    for s in pool.scenes:
        q = np.asarray(s.initial_attitude_quat, float)
        cos_t = q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2
        deficit = max(0.0, 0.5 - float(cos_t))
        zmin = -4.0 + k * deficit
        vz_up = max(0.0, float(s.initial_velocity[2]))
        zmax = min(3.9, 3.9 - 0.5 * vz_up)
        sz = float(s.start[2]); gz = float(s.goal[2])
        if sz < zmin - 1e-9 or sz > zmax + 1e-9 or abs(gz) > 3.5 + 1e-9 or sz < -4.0 - 1e-9:
            viol += 1
    return viol


def main():
    base = _merged()
    variant = pool_variant(base, SYSTEM)
    summary = {"model_constants": MODEL_CONSTANTS, "k_grid": K_GRID, "k_threshold_flagged": K_THRESH,
               "inertness": {}, "k_screen": {}}

    # inertness re-gates (sampler edit must not perturb existing paths)
    g = build_pool(base, SYSTEM, PoolSpec("full", 2000, 23456))
    ga = write_pool(g, base, output_dir=OUTK, git_commit="uncommitted-v2.7.6", variant=variant)
    vz = build_pool(_vz_cfg(base), SYSTEM, PoolSpec("evalicz", 2000, 42))
    va = write_pool(vz, _vz_cfg(base), output_dir=OUTK, git_commit="uncommitted-v2.7.6", variant=variant)
    summary["inertness"] = {"canonical_sha": ga.sha256[:16], "canonical_ok": ga.sha256.startswith(CANON_SHA),
                            "r61_vz_sha": va.sha256[:16], "r61_vz_ok": va.sha256.startswith(R61_VZ_SHA)}
    print(f"[INERTNESS] canonical {ga.sha256[:12]} ok={summary['inertness']['canonical_ok']} | "
          f"R6.1-vz {va.sha256[:12]} ok={summary['inertness']['r61_vz_ok']}")
    if not (summary["inertness"]["canonical_ok"] and summary["inertness"]["r61_vz_ok"]):
        (OUTK / "kscreen_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        raise SystemExit("HALT: sampler edit perturbed an existing path (inertness gate failed).")

    k_star = None
    for k in K_GRID:
        cfg = _tilt_cfg(base, k)
        pool = build_pool(cfg, SYSTEM, PoolSpec(f"tiltk{str(k).replace('.', 'p')}", 2000, 42))
        art = write_pool(pool, cfg, output_dir=OUT, git_commit="uncommitted-v2.7.6", variant=variant)
        viol = verify_bounds(pool, k)
        feas = check_pool(pool)
        rec = {"k": k, "stem": art.pool_path.stem, "sha256": art.sha256[:16], "bound_violations": viol,
               "flagged_total": feas["flagged_total"], "flagged_fraction": feas["flagged_fraction"],
               "model_min_z": feas["model_min_z"], "flagged_ICs": feas["flagged_ICs"][:20]}
        summary["k_screen"][str(k)] = rec
        ok = feas["flagged_total"] <= K_THRESH
        if k_star is None and ok:
            k_star = k
        print(f"[k={k}] flagged {feas['flagged_total']}/2000 (viol={viol}) min_z={feas['model_min_z']['min']} "
              f"{'<= thresh' if ok else '> thresh'}")
    summary["k_star"] = k_star
    summary["k_star_rule"] = f"smallest k with flagged <= {K_THRESH} of 2000"
    (OUTK / "kscreen_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\n[k-screen] k* = {k_star}")


if __name__ == "__main__":
    main()
