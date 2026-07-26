"""v2.7.6 Stage-1 M3 — build the band-feasible eval pool + inertness gates, report admission / Haar / slack.

Builds (all quadrotor_3d / d2r / n2000, mode=eval):
  (band)      seed 42, ic_eval_z floor_mode=band  -> the band-feasible pool.
  (gate-canon) seed 23456, NO ic_eval_z          -> SHA MUST stay 0ef3751b (BLOCKING).
  (gate-full)  seed 42,    NO ic_eval_z          -> SHA MUST stay db0b9eb5 (BLOCKING).
Reports: admission rate (nonempty-interval), attitude marginal vs Haar (KS on cos theta ~ U[-1,1]),
slack = (z0+4)-D_down distribution over admitted ICs (must be >= 0), and the M2 attitude-redraw count.
ic_eval_z injected in-memory only (no config promotion). No git, no securing.
"""
from __future__ import annotations

import copy, json, math
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml

from src.eval.build_pools import PoolSpec, build_pool, write_pool, load_pool, pool_variant
from src.envs.scene_init import band_stats, reset_band_stats
from scripts.analysis.v276_attitude_feasibility import D_down_single, D_up, MODEL_CONSTANTS

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
SYSTEM = "quadrotor_3d"
OUT = REPO / "data/runs/v2.7.6/pools"
OUTS = REPO / "data/runs/v2.7.6/stage1_build"; OUTS.mkdir(parents=True, exist_ok=True)
CANON_SHA = "0ef3751b"; FULL_SHA = "db0b9eb5"
BAND = {"floor_mode": "band", "band_floor": -4.0, "band_ceiling": 4.0}


def _merged():
    b = yaml.safe_load((REPO / "src/configs/base_config.yaml").read_text())
    e = yaml.safe_load((REPO / "src/configs/exp_config.yaml").read_text())
    def m(a, o):
        d = dict(a)
        for k, v in o.items():
            d[k] = m(d[k], v) if isinstance(v, Mapping) and isinstance(d.get(k), Mapping) else v
        return d
    return m(b, e)


def _band_cfg(cfg):
    cfg = copy.deepcopy(cfg); cfg["env"]["quadrotor_3d"]["ic_eval_z"] = dict(BAND); return cfg


def ks_uniform_m1_1(x):
    """KS statistic of x against U[-1,1]; returns (D, approx p). CDF_0(t)=(t+1)/2 on [-1,1]."""
    x = np.sort(np.asarray(x, float)); n = x.size
    cdf = np.clip((x + 1.0) / 2.0, 0.0, 1.0)
    d_plus = np.max(np.arange(1, n + 1) / n - cdf)
    d_minus = np.max(cdf - np.arange(0, n) / n)
    D = float(max(d_plus, d_minus))
    p = float(math.exp(-2.0 * n * D * D))                # conservative asymptotic tail
    return D, p


def cos_theta(scene):
    q = np.asarray(scene.initial_attitude_quat, float)
    return float(q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2)


def main():
    base = _merged(); variant = pool_variant(base, SYSTEM)
    out = {"model_constants": MODEL_CONSTANTS, "inertness": {}, "band_pool": {}}

    # --- inertness gates (BLOCKING) ---
    g = build_pool(base, SYSTEM, PoolSpec("full", 2000, 23456))
    ga = write_pool(g, base, output_dir=OUTS, git_commit="uncommitted-v2.7.6", variant=variant)
    fr = build_pool(base, SYSTEM, PoolSpec("fullrange", 2000, 42))
    fa = write_pool(fr, base, output_dir=OUTS, git_commit="uncommitted-v2.7.6", variant=variant)
    out["inertness"] = {"canonical_sha": ga.sha256[:16], "canonical_ok": ga.sha256.startswith(CANON_SHA),
                        "fullrange_sha": fa.sha256[:16], "fullrange_ok": fa.sha256.startswith(FULL_SHA)}
    print(f"[INERTNESS] canonical {ga.sha256[:12]} ok={out['inertness']['canonical_ok']} | "
          f"fullrange {fa.sha256[:12]} ok={out['inertness']['fullrange_ok']}")
    if not (out["inertness"]["canonical_ok"] and out["inertness"]["fullrange_ok"]):
        (OUTS / "stage1_build.json").write_text(json.dumps(out, indent=2) + "\n")
        raise SystemExit("HALT: inertness SHA changed — training/default path perturbed. STOP.")

    # --- band-feasible pool ---
    reset_band_stats()
    cfg = _band_cfg(base)
    bp = build_pool(cfg, SYSTEM, PoolSpec("bandfeasible", 2000, 42))
    ba = write_pool(bp, cfg, output_dir=OUT, git_commit="uncommitted-v2.7.6", variant=variant)
    st = band_stats()
    admission = round(st["scenes"] / max(st["attempts"], 1), 6)

    # slack + attitude marginal over the 2000 admitted scenes
    slack = np.empty(2000); ct = np.empty(2000); Dd = np.empty(2000)
    for i, s in enumerate(bp.scenes):
        cthe = cos_theta(s); ct[i] = cthe
        theta = math.acos(min(1.0, max(-1.0, cthe)))
        d = D_down_single(theta, float(s.initial_velocity[2]), float(np.linalg.norm(s.initial_omega_vec)))
        Dd[i] = d
        slack[i] = (float(s.start[2]) + 4.0) - d
    D_band, p_band = ks_uniform_m1_1(ct)
    ct_canon = np.array([cos_theta(s) for s in g.scenes])          # Haar reference (default path)
    D_ref, p_ref = ks_uniform_m1_1(ct_canon)
    min_slack = float(slack.min())

    out["band_pool"] = {
        "stem": ba.pool_path.stem, "sha256": ba.sha256[:16], "n": 2000,
        "redraw_stats": st, "admission_rate": admission,
        "attitude_marginal_KS_vs_U[-1,1]": {"band_D": round(D_band, 5), "band_p_approx": round(p_band, 4),
            "canonical_ref_D": round(D_ref, 5), "canonical_ref_p_approx": round(p_ref, 4),
            "preserved": bool(D_band <= D_ref * 1.5 + 0.02)},
        "slack_m": {"min": round(min_slack, 5), "p1": round(float(np.percentile(slack, 1)), 4),
                    "median": round(float(np.median(slack)), 4), "max": round(float(slack.max()), 4),
                    "all_nonneg": bool(min_slack >= -1e-9)},
        "D_down_over_pool": {"min": round(float(Dd.min()), 4), "median": round(float(np.median(Dd)), 4),
                             "p99": round(float(np.percentile(Dd, 99)), 4), "max": round(float(Dd.max()), 4)},
        "start_z": {"min": round(float(min(s.start[2] for s in bp.scenes)), 4),
                    "max": round(float(max(s.start[2] for s in bp.scenes)), 4)},
    }
    (OUTS / "stage1_build.json").write_text(json.dumps(out, indent=2) + "\n")

    print(f"[band pool] {ba.pool_path.name} sha={ba.sha256[:12]}")
    print(f"  redraws: empty_interval={st['empty_interval_redraws']} obstacle_scene_rejects={st['obstacle_scene_rejects']} "
          f"attempts={st['attempts']} -> admission_rate={admission}")
    print(f"  attitude KS vs U[-1,1]: band D={D_band:.4f} (ref Haar D={D_ref:.4f}) preserved={out['band_pool']['attitude_marginal_KS_vs_U[-1,1]']['preserved']}")
    print(f"  slack min={min_slack:.5f} (all_nonneg={out['band_pool']['slack_m']['all_nonneg']}) median={out['band_pool']['slack_m']['median']}")
    print(f"  D_down: median={out['band_pool']['D_down_over_pool']['median']} max={out['band_pool']['D_down_over_pool']['max']}")
    # STOP conditions
    if admission < 0.70:
        raise SystemExit(f"STOP: admission {admission} < 0.70 — report slack + binding term, do NOT tune.")
    if min_slack < -1e-9:
        raise SystemExit(f"STOP/FAIL: admitted slack < 0 ({min_slack}).")
    if not out["band_pool"]["attitude_marginal_KS_vs_U[-1,1]"]["preserved"]:
        raise SystemExit("STOP: attitude marginal shifted materially from Haar — redraw leaking.")
    print("M3 DONE — band-feasible pool + inertness gates OK.")


if __name__ == "__main__":
    main()
