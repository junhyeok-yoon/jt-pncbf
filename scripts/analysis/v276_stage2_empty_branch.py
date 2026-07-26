"""v2.7.6 Stage-2 — empty-branch instrumentation (the infeasible-branch threat to the JT gradient mechanism).

For a checkpoint + pool, roll out the deployed policy+filter and, on EMPTY-branch steps (half-space/box
intersection empty), report:
  (2) the selected-candidate decomposition: box vertex vs clipped-nominal vs edge-solve — the same partition
      rem:empty-resolved used (71.1 / 28.9 / 0), by re-running the exact filter enumerator on the states.
  (3) the share of empty steps whose ACTIVE h_star branch is the BAND branch (psi +/- c_z v_z) rather than an
      obstacle branch — whether the band term is the source.
Also returns overall infeasibility (empty|singular fraction over active steps). Post-hoc: reads the eval
trajectories' recorded empty flags (RolloutResult.empty); no change to the filter or training path.
"""
from __future__ import annotations

import copy, json, math
from pathlib import Path

import numpy as np
import torch

from src.common.filter_hardnet import (
    _cbf_terms, _base_projection, _base_alpha, _hardnet_params, _candidate_actions,
    _empty_halfspace_box, _clamp_to_bounds, _SINGULAR_LG_THRESHOLD, _FEASIBILITY_TOL,
)
from src.common.signed_h import signed_h
from src.common.value_net import make_h_fn
from src.eval.evaluate import evaluate
from src.eval.run_full import _load_framework
from src.envs.scene_batch import batch_scenes

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
POOLS = REPO / "data/runs/v2.7.6/pools"
MAX_EMPTY = 20000     # subsample cap on empty-step states for the decomposition


def _classify_selected(u_nom, base_proj, lg, row_upper, bounds):
    """Re-run the box-aware enumerator; on each row return selected candidate CLASS index:
    0=clipped-nominal (clamped u_nom or base projection), 1=box vertex (a 2^m corner), 2=edge-solve
    (fix-one-solve-one). Mirrors _box_aware_projection's least-violating argmin on the empty branch."""
    A = u_nom.shape[1]
    cands = _candidate_actions(u_nom, base_proj, lg, row_upper, bounds)           # [B, K, A]
    n_clip = 2                                                                    # clamped_nom, base_projection
    n_corner = 2 ** A
    lhs = torch.einsum("ba,bka->bk", lg, cands)
    violation = torch.relu(lhs - row_upper.unsqueeze(1))
    dist_sq = torch.sum((cands - u_nom.unsqueeze(1)) ** 2, dim=2)
    sel = torch.argmin(violation + _FEASIBILITY_TOL * dist_sq, dim=1)             # empty-branch least-violating
    cls = torch.where(sel < n_clip, torch.zeros_like(sel),
                      torch.where(sel < n_clip + n_corner, torch.ones_like(sel), 2 * torch.ones_like(sel)))
    return cls


def _hstar_branches(system, x, scene, cfg):
    """Return (obstacle_branch, band_branch_max) per state, to find the ACTIVE branch."""
    h_scale = float(cfg["env"]["h_scale"])
    phi = signed_h(system.position(x)[..., :2], scene, h_scale)
    c = float(cfg["env"][system.name]["c_gain"])
    obst = phi + c * system.approach_barrier(x, scene, h_scale)
    b = cfg["env"]["band_hazard"]; limit = float(b["limit"])
    psi_cap = float(cfg["obstacle"]["per_system"]["quadrotor_3d"]["r_max"])
    c_z = math.pi / float(cfg["env"]["bounds"]["quadrotor_3d"]["omega_max"])
    z = system.position(x)[..., 2]; v_z = x[..., 9]
    band = torch.maximum(torch.clamp(z - limit, max=psi_cap) + c_z * v_z,
                         torch.clamp(-z - limit, max=psi_cap) - c_z * v_z)
    return obst, band


def diagnose(ckpt: Path, stem: str, max_scenes=None):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    over = {"env": {"dt": 0.05, "stuck_window_steps": 60, "stuck_radius": 0.10, "band_collision_limit": 4.0},
            "eval": {"max_steps": 200, "dt_ctrl": 0.05}, "filter": copy.deepcopy(ck["config"]["filter"])}
    over["filter"]["empty_fallback"] = {"mode": "none", "k": int(over["filter"].get("empty_fallback", {}).get("k", 10))}
    # ensure band_hazard present for h_star branch attribution
    over.setdefault("env", {})["band_hazard"] = ck["config"]["env"].get("band_hazard", {"enabled": True, "limit": 4.0})
    fw, cfg, _ = _load_framework(ckpt, config_overrides=over)
    sys = fw.system
    h_fn = make_h_fn(fw.value_net, sys)                           # V_hat as the filter barrier (OC + JT alike)
    res = evaluate(fw, POOLS / f"{stem}.pkl", cfg, mode="final", step=int(ck["step"]),
                   ckpt_name=ckpt.name, max_scenes=max_scenes, include_lqr_baseline=False)
    params = _hardnet_params(cfg)
    # collect empty-branch (state, u_nom) across episodes (from recorded RolloutResult.empty + u_nom)
    xs, us, scenes = [], [], []
    n_active = n_empty = n_singular = 0
    for tr in res.trajectories:
        st = tr.filtered.states                                   # [T+1,1,D]
        un = tr.filtered.u_nom                                    # [T,1,A]
        emp = getattr(tr.filtered, "empty", None)                 # [T,1] bool
        sing = getattr(tr.filtered, "singular", None)
        if emp is None:
            continue
        T = emp.shape[0]
        n_active += T
        e = emp.reshape(-1).bool(); s = sing.reshape(-1).bool() if sing is not None else torch.zeros(T, dtype=torch.bool)
        n_empty += int(e.sum()); n_singular += int(s.sum())
        idx = torch.nonzero(e, as_tuple=False).reshape(-1)
        for t in idx.tolist():
            xs.append(st[t, 0]); us.append(un[t, 0]); scenes.append(tr.scene)
    infeas = round((n_empty + n_singular) / max(n_active, 1), 6)
    out = {"ckpt": ckpt.name, "step": int(ck["step"]), "stem": stem,
           "infeasibility_empty_or_singular": infeas, "empty_frac": round(n_empty / max(n_active, 1), 6),
           "n_active_steps": n_active, "n_empty_steps": n_empty}
    if not xs:
        out["note"] = "no empty steps"; return out
    if len(xs) > MAX_EMPTY:
        sel = np.random.default_rng(0).choice(len(xs), MAX_EMPTY, replace=False)
        xs = [xs[i] for i in sel]; us = [us[i] for i in sel]; scenes = [scenes[i] for i in sel]
    X = torch.stack(xs, dim=0).to(torch.float32)
    u_nom = torch.stack(us, dim=0).to(torch.float32)
    bs = batch_scenes(scenes, device=X.device, dtype=X.dtype)
    h, lf, lg = _cbf_terms(sys, h_fn, X, bs, u_nom, create_graph=False)
    alpha = _base_alpha(h, params)
    row_upper = -lf - alpha * h
    bounds = sys.u_bounds.to(X.device, X.dtype)
    base_proj = _base_projection(u_nom, lg, row_upper, bounds, params)
    empty = _empty_halfspace_box(lg, row_upper, bounds)           # confirm these are empty
    cls = _classify_selected(u_nom, base_proj, lg, row_upper, bounds)
    m = empty                                                     # restrict to truly-empty (should be ~all)
    cls_e = cls[m]
    dec = {"box_vertex": round(float((cls_e == 1).float().mean()), 4),
           "clipped_nominal": round(float((cls_e == 0).float().mean()), 4),
           "edge_solve": round(float((cls_e == 2).float().mean()), 4),
           "n": int(m.sum()), "ref_rem_empty_resolved": [71.1, 28.9, 0.0]}
    obst, band = _hstar_branches(sys, X, bs, cfg)
    band_bind = round(float((band[m] > obst[m]).float().mean()), 4)
    out["candidate_decomposition_on_empty"] = dec
    out["band_branch_active_share_on_empty"] = band_bind
    return out


if __name__ == "__main__":
    import sys as _s
    ckpt = Path(_s.argv[1]); stem = _s.argv[2] if len(_s.argv) > 2 else "eval_bandfeasible_quadrotor-3d-d2r_n2000_seed42"
    ms = int(_s.argv[3]) if len(_s.argv) > 3 else None
    r = diagnose(ckpt, stem, max_scenes=ms)
    print(json.dumps(r, indent=2))
