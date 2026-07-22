"""v2.7.3 M3 authority/degeneracy gate for quadrotor_3d (per-rotor plant).

The v2.7.2 floor (1e-3 on ||L_g V_hat||) was a coordinate norm in WRENCH units and does not transfer to the
per-rotor parametrization. This gate measures, from the SAME control-affine filter-coefficient path the
deployed HardNet filter uses (_cbf_terms -> h, L_f V_hat, L_g V_hat), what feasibility actually depends on:

  authority  A = sum_i |(L_g V_hat)_i| * box_halfwidth   (box_halfwidth = 4.905/2 = 2.4525; max commandable
             V_hat rate over the per-rotor box)
  drift      D = L_f V_hat                                (the drift the constraint must overcome)

Sampling: roll the fixed nominal on the (d2) in-loop pool; keep states near the learned boundary |V_hat| < band.
Reports median and p05 of A and D, frac(A < D), and the legacy raw ||L_g V_hat|| (continuity only). HALT only on
MANIFEST degeneracy: frac(A < D) > 0.50 OR median A < 1e-2. The operational threshold is set later, not here.
Persists the full arrays + summary JSON (06_workflow §3.2).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.common.filter_hardnet import _cbf_terms
from src.common.kstep_fallback import slice_scene
from src.common.rk4 import rk4_step
from src.common.value_net import ValueNetEnsemble, make_h_fn
from src.eval.build_pools import DEFAULT_OUTPUT_DIR, load_pool, pool_stem, pool_variant, resolve_pool_or_raise
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.envs.quadrotor_3d import _quat_to_R
from src.frameworks.jt_pncbf.train import make_system

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--band", type=float, default=0.15)
ap.add_argument("--box-halfwidth", type=float, default=2.4525)   # 4.905 / 2
ap.add_argument("--run-dir", required=True)
ap.add_argument("--label", default="m3_gate")
ap.add_argument("--pool", default=None, help="explicit pool .pkl; else resolve the config stem (secured|eval_pools)")
a = ap.parse_args()
STRATA = [(0.0, 60.0), (60.0, 120.0), (120.0, 180.001)]

ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
cfg = ck["config"]
system = make_system(cfg)
vnet = ValueNetEnsemble(system.obs_dim, cfg); vnet.load_state_dict(ck["v_s_state"]); vnet.eval()
for p in vnet.parameters():
    p.requires_grad_(False)
h_fn = make_h_fn(vnet, system)

variant = pool_variant(cfg, "quadrotor_3d")
stem = pool_stem("inloop", "quadrotor_3d", int(cfg["eval"]["in_loop"]["n"]),
                 int(cfg["eval"]["in_loop"]["seed"]), "random", variant)
pool = load_pool(Path(a.pool)) if a.pool else load_pool(resolve_pool_or_raise(stem))  # secured|eval_pools (v2.7.4)
bs = batch_scenes(pool.scenes, device=torch.device("cpu"), dtype=torch.float32)
x = system.wrap_state(initial_states_from_batch(bs).float())
goal = bs.goal
dt = float(cfg["env"]["dt"]); max_steps = int(cfg["eval"]["max_steps"])
u0 = torch.zeros(x.shape[0], int(system.action_dim))

# per-episode INITIAL tilt (deg) for stratification (tilt from initial_attitude_quat -> R[2,2])
quats = np.stack([np.asarray(s.initial_attitude_quat) for s in pool.scenes])
R0 = _quat_to_R(torch.tensor(quats, dtype=torch.float64))
init_tilt = np.degrees(np.arccos(np.clip(R0[:, 2, 2].numpy(), -1.0, 1.0)))   # [n_episodes]

A_all, D_all, lg_all, ep_all = [], [], [], []
ep_index = torch.arange(x.shape[0])
for t in range(max_steps):
    with torch.no_grad():
        un = system.lqr_action(x, goal)
        h = h_fn(x, bs).reshape(-1)
    near = (h.abs() < a.band)
    if bool(near.any()):
        xin = x[near].clone()
        _h, lf, lg = _cbf_terms(system, h_fn, xin, slice_scene(bs, near), u0[: xin.shape[0]], create_graph=False)
        A_all.append((lg.abs().sum(dim=1) * a.box_halfwidth).detach())
        D_all.append(lf.detach())
        lg_all.append(torch.linalg.norm(lg, dim=1).detach())
        ep_all.append(ep_index[near].clone())                 # which episode each boundary state came from
    with torch.no_grad():
        x = rk4_step(system, x, un, dt)

A = torch.cat(A_all).numpy(); D = torch.cat(D_all).numpy(); lg = torch.cat(lg_all).numpy()
ep = torch.cat(ep_all).numpy()
tilt_of_state = init_tilt[ep]                                 # initial tilt of the episode each state belongs to
frac_AltD = float((A < D).mean())
med_A = float(np.median(A))


def _stratum_gate(lo, hi):
    m = (tilt_of_state >= lo) & (tilt_of_state < hi)
    n = int(m.sum())
    if n == 0:
        return {"stratum_deg": f"[{int(lo)},{int(min(hi,180))})", "n": 0}
    aS, dS = A[m], D[m]
    return {"stratum_deg": f"[{int(lo)},{int(min(hi,180))})", "n": n,
            "median_A": round(float(np.median(aS)), 5), "median_D": round(float(np.median(dS)), 5),
            "frac_A_lt_D": round(float((aS < dS).mean()), 5),
            "degenerate": bool(float((aS < dS).mean()) > 0.50 or float(np.median(aS)) < 1e-2)}


by_stratum = [_stratum_gate(lo, hi) for lo, hi in STRATA]
agg_halt = bool(frac_AltD > 0.50 or med_A < 1e-2)
hi_tilt_only = (not agg_halt) and any(s.get("degenerate") for s in by_stratum)
res = {
    "label": a.label, "ckpt": a.ckpt, "pool": (a.pool or stem), "n_boundary_states": int(A.size), "band": a.band,
    "box_halfwidth": a.box_halfwidth,
    "authority_A": {"median": round(med_A, 5), "p05": round(float(np.percentile(A, 5)), 5),
                    "mean": round(float(A.mean()), 5)},
    "drift_D": {"median": round(float(np.median(D)), 5), "p05": round(float(np.percentile(D, 5)), 5),
                "p95": round(float(np.percentile(D, 95)), 5)},
    "frac_A_lt_D": round(frac_AltD, 5),
    "legacy_raw_norm_lgV": {"median": round(float(np.median(lg)), 5), "p05": round(float(np.percentile(lg, 5)), 5)},
    "by_tilt_stratum": by_stratum,
    "HALT_manifest_degeneracy": agg_halt,
    "high_tilt_only_degeneracy_NONHALT": hi_tilt_only,
}
run_dir = Path(a.run_dir); run_dir.mkdir(parents=True, exist_ok=True)
np.savez(run_dir / f"{a.label}_dist.npz", A=A, D=D, lg=lg)
(run_dir / f"{a.label}.json").write_text(json.dumps(res, indent=2) + "\n")
print(json.dumps(res, indent=2))
