"""v2.7.6 Stage-1c M3 — rebuild the band-feasible pool with band clearance on goal_z and the start ceiling.

delta_eval = eval.scene.start_goal_clearance is READ from config (never chosen). If delta_eval >= 1.0 STOP.
goal_z ~ U(-4+delta, +4-delta); z_init_max = min(+4-delta, +4-D_up); z_init_min = -4 + D_down (unchanged).
Re-verifies both inertness SHAs (0ef3751b, db0b9eb5), reports admission / Haar / slack, and the fraction of
the PREVIOUS band pool's scenes the new goal_z bound would exclude. Overwrites the seed-42 band pool stem
(new SHA). ic_eval_z injected in-memory only. No git, no securing.
"""
from __future__ import annotations

import copy, json, math
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml

from src.eval.build_pools import PoolSpec, build_pool, write_pool, load_pool, pool_variant
from src.envs.scene_init import band_stats, reset_band_stats
from scripts.analysis.v276_attitude_feasibility import D_down_single, MODEL_CONSTANTS

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
SYSTEM = "quadrotor_3d"
OUT = REPO / "data/runs/v2.7.6/pools"
OUTS = REPO / "data/runs/v2.7.6/stage1c_build"; OUTS.mkdir(parents=True, exist_ok=True)
CANON_SHA = "0ef3751b"; FULL_SHA = "db0b9eb5"
BAND = {"floor_mode": "band", "band_floor": -4.0, "band_ceiling": 4.0}
STEM = "eval_bandfeasible_quadrotor-3d-d2r_n2000_seed42"


def _merged():
    b = yaml.safe_load((REPO / "src/configs/base_config.yaml").read_text())
    e = yaml.safe_load((REPO / "src/configs/exp_config.yaml").read_text())
    def m(a, o):
        d = dict(a)
        for k, v in o.items():
            d[k] = m(d[k], v) if isinstance(v, Mapping) and isinstance(d.get(k), Mapping) else v
        return d
    return m(b, e)


def ks_uniform_m1_1(x):
    x = np.sort(np.asarray(x, float)); n = x.size
    cdf = np.clip((x + 1.0) / 2.0, 0.0, 1.0)
    D = float(max(np.max(np.arange(1, n + 1) / n - cdf), np.max(cdf - np.arange(0, n) / n)))
    return D, float(math.exp(-2.0 * n * D * D))


def cos_theta(s):
    q = np.asarray(s.initial_attitude_quat, float)
    return float(q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2)


def main():
    base = _merged(); variant = pool_variant(base, SYSTEM)
    delta = float(base["eval"]["scene"]["start_goal_clearance"])
    out = {"model_constants": MODEL_CONSTANTS, "delta_eval": delta,
           "goal_z_bound": [round(-4.0 + delta, 4), round(4.0 - delta, 4)],
           "z_ceiling_bound": round(4.0 - delta, 4)}
    print(f"[delta_eval] = {delta} (from eval.scene.start_goal_clearance)")
    if delta >= 1.0:
        (OUTS / "stage1c_build.json").write_text(json.dumps(out | {"STOP": "delta_eval >= 1.0"}, indent=2) + "\n")
        raise SystemExit(f"STOP: delta_eval {delta} >= 1.0 removes a quarter of the band — Researcher decision.")

    # exclusion of the PREVIOUS band pool under the new goal_z bound (load BEFORE overwrite)
    prev = OUT / f"{STEM}.pkl"
    if prev.exists():
        old = load_pool(prev)
        gz = np.array([float(s.goal[2]) for s in old.scenes])
        sz = np.array([float(s.start[2]) for s in old.scenes])
        out["previous_pool"] = {
            "sha_note": "pre-1c band pool (overwritten below)",
            "goal_z_excluded_frac": round(float(((gz < -4 + delta) | (gz > 4 - delta)).mean()), 4),
            "goal_z_excluded_count": int(((gz < -4 + delta) | (gz > 4 - delta)).sum()),
            "start_z_above_ceiling_frac": round(float((sz > 4 - delta).mean()), 4),
            "start_z_above_ceiling_count": int((sz > 4 - delta).sum())}

    # inertness gates (BLOCKING)
    g = build_pool(base, SYSTEM, PoolSpec("full", 2000, 23456))
    ga = write_pool(g, base, output_dir=OUTS, git_commit="uncommitted-v2.7.6", variant=variant)
    fr = build_pool(base, SYSTEM, PoolSpec("fullrange", 2000, 42))
    fa = write_pool(fr, base, output_dir=OUTS, git_commit="uncommitted-v2.7.6", variant=variant)
    out["inertness"] = {"canonical_sha": ga.sha256[:16], "canonical_ok": ga.sha256.startswith(CANON_SHA),
                        "fullrange_sha": fa.sha256[:16], "fullrange_ok": fa.sha256.startswith(FULL_SHA)}
    print(f"[INERTNESS] canonical {ga.sha256[:12]} ok={out['inertness']['canonical_ok']} | "
          f"fullrange {fa.sha256[:12]} ok={out['inertness']['fullrange_ok']}")
    if not (out["inertness"]["canonical_ok"] and out["inertness"]["fullrange_ok"]):
        (OUTS / "stage1c_build.json").write_text(json.dumps(out, indent=2) + "\n")
        raise SystemExit("HALT: inertness SHA changed — training/default path perturbed. STOP.")

    # rebuild band-feasible pool (overwrites STEM)
    reset_band_stats()
    cfg = copy.deepcopy(base); cfg["env"]["quadrotor_3d"]["ic_eval_z"] = dict(BAND)
    bp = build_pool(cfg, SYSTEM, PoolSpec("bandfeasible", 2000, 42))
    ba = write_pool(bp, cfg, output_dir=OUT, git_commit="uncommitted-v2.7.6", variant=variant)
    st = band_stats(); admission = round(st["scenes"] / max(st["attempts"], 1), 6)

    slack = np.empty(2000); ct = np.empty(2000)
    gz_new = np.array([float(s.goal[2]) for s in bp.scenes]); sz_new = np.array([float(s.start[2]) for s in bp.scenes])
    for i, s in enumerate(bp.scenes):
        cthe = cos_theta(s); ct[i] = cthe
        d = D_down_single(math.acos(min(1.0, max(-1.0, cthe))), float(s.initial_velocity[2]),
                          float(np.linalg.norm(s.initial_omega_vec)))
        slack[i] = (float(s.start[2]) + 4.0) - d
    D_band, _ = ks_uniform_m1_1(ct); D_ref, _ = ks_uniform_m1_1(np.array([cos_theta(s) for s in g.scenes]))
    min_slack = float(slack.min())
    out["band_pool"] = {"stem": STEM, "sha256": ba.sha256[:16], "n": 2000, "redraw_stats": st,
        "admission_rate": admission,
        "attitude_marginal": {"band_D": round(D_band, 5), "canonical_ref_D": round(D_ref, 5),
                              "preserved": bool(D_band <= D_ref * 1.5 + 0.02)},
        "slack_m": {"min": round(min_slack, 5), "median": round(float(np.median(slack)), 4),
                    "all_nonneg": bool(min_slack >= -1e-9)},
        "goal_z": {"min": round(float(gz_new.min()), 4), "max": round(float(gz_new.max()), 4)},
        "start_z": {"min": round(float(sz_new.min()), 4), "max": round(float(sz_new.max()), 4)}}
    (OUTS / "stage1c_build.json").write_text(json.dumps(out, indent=2) + "\n")

    print(f"[band pool] {STEM} sha={ba.sha256[:12]}  admission={admission} "
          f"(empty_interval={st['empty_interval_redraws']} obstacle_rejects={st['obstacle_scene_rejects']})")
    print(f"  Haar KS band D={D_band:.4f} (ref {D_ref:.4f}) preserved={out['band_pool']['attitude_marginal']['preserved']}")
    print(f"  slack min={min_slack:.5f} all_nonneg={out['band_pool']['slack_m']['all_nonneg']}")
    print(f"  goal_z range [{gz_new.min():.3f},{gz_new.max():.3f}]  start_z max={sz_new.max():.3f}")
    if "previous_pool" in out:
        print(f"  previous pool goal_z excluded by new bound: {out['previous_pool']['goal_z_excluded_count']}/2000 "
              f"({out['previous_pool']['goal_z_excluded_frac']})")
    if admission < 0.70: raise SystemExit(f"STOP: admission {admission} < 0.70.")
    if min_slack < -1e-9: raise SystemExit(f"STOP/FAIL: slack < 0 ({min_slack}).")
    if not out["band_pool"]["attitude_marginal"]["preserved"]: raise SystemExit("STOP: Haar marginal shifted.")
    print("Stage-1c build DONE.")


if __name__ == "__main__":
    main()
