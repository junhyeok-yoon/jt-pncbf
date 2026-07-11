"""MC-PNCBF Stage A — training-free analytic maneuver-barrier evaluation (read-only, no training).

Deploys the analytic V_M (src/common/maneuver_value.py) as the HardNet h_fn with a clamped-LQR
nominal, on the frozen full-eval pools, via the standard evaluate() pipeline (cps + CIs + episode
rows). Adds: premise check (V_brake, V_M on all initial states); per-step singular/empty split BY
REGION (V_M<=0 vs >0); authority stats (||row A|| = ||L_g V_M||); discrete CBF-condition probe on
{V_M<=0}; per-filter-call wallclock. One (pool, arm) per invocation.

Filter uses the SAME HardNet internals as deployment (_cbf_terms/_base_projection/_box_aware_
projection) but with create_graph=False (first-order L_g V_M only; numerically identical to the
training create_graph=True path — higher-order graph unused in eval) and a detached output.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from src.common.filter_hardnet import (
    _SINGULAR_LG_THRESHOLD, _base_alpha, _base_projection, _box_aware_projection,
    _cbf_terms, _empty_halfspace_box, _hardnet_params,
)
from src.common.maneuver_value import make_maneuver_h_fn, maneuver_value, t_stop
from src.common.outcomes import step_outcomes
from src.envs.double_integrator import DoubleIntegrator
from src.eval.build_pools import load_pool
from src.eval.evaluate import evaluate, first_physical_event_step
from src.envs.scene_batch import batch_scenes, initial_states_from_batch

REPO = Path(__file__).resolve().parents[2]
POOLS = {
    "n500": REPO / "data/secured_data/pools/eval_full_di_n500_seed23456.pkl",
    "n2000": REPO / "data/secured_data/pools/eval_full_di_n2000_seed23456.pkl",
}


def _deep_merge(base, override):
    m = dict(base)
    for k, v in override.items():
        m[k] = _deep_merge(m[k], v) if isinstance(v, dict) and isinstance(m.get(k), dict) else v
    return m


def _config():
    base = yaml.safe_load((REPO / "src/configs/base_config.yaml").read_text())
    exp = yaml.safe_load((REPO / "src/configs/exp_config.yaml").read_text())
    return _deep_merge(base, exp)


class StageAFramework:
    """EvaluatedFramework: clamped-LQR nominal + HardNet(V_M) filter."""

    value_net = None

    def __init__(self, system, config, lateral_js, softmin_beta=0.0, nominal_only=False, gamma_m=0.0):
        self.system = system
        self.config = config
        self.lateral_js = lateral_js
        self.nominal_only = nominal_only
        self.gamma_m = float(gamma_m)   # G7: filter margin, h_eff = V_M + gamma_m (driver-side; not in src)
        self.h_fn = make_maneuver_h_fn(system, config, lateral_js=lateral_js, softmin_beta=softmin_beta)
        self.params = _hardnet_params(config)
        self.n_calls = 0
        self.filter_time = 0.0

    def policy(self, x, scene):
        goal = torch.as_tensor(scene.goal, dtype=x.dtype, device=x.device)
        if goal.ndim == 1:
            goal = goal.unsqueeze(0).expand(x.shape[0], -1)
        return self.system.lqr_action(x, goal)   # DoubleIntegrator.lqr_action already clamps to box

    def filter(self, x, u_nom, scene):
        if self.nominal_only:                       # LQR-only baseline (pass-through, never infeasible)
            return u_nom, torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        t0 = time.time()
        h, lf, lg = _cbf_terms(self.system, self.h_fn, x, scene, u_nom, create_graph=False)
        h, lf, lg = h.detach(), lf.detach(), lg.detach()
        h = h + self.gamma_m   # h_eff = V_M + gamma_m (constant shift: L_f/L_g unchanged; alpha & row use h_eff)
        with torch.no_grad():
            alpha = _base_alpha(h, self.params)
            row_upper = -lf - alpha * h
            bounds = self.system.u_bounds.to(device=u_nom.device, dtype=u_nom.dtype)
            projected = _base_projection(u_nom, lg, row_upper, bounds, self.params)
            singular = torch.linalg.norm(lg, dim=1) < _SINGULAR_LG_THRESHOLD
            if self.params.box_aware:
                u_safe, empty = _box_aware_projection(u_nom, projected, lg, row_upper, bounds)
                infeasible = singular | empty
            else:
                u_safe, infeasible = projected, singular
        if x.device.type == "cuda":
            torch.cuda.synchronize()
        self.n_calls += 1
        self.filter_time += time.time() - t0
        return u_safe.detach(), infeasible.detach()


def _region_split(fw, config, scenes, max_steps, dt, eval_batch, alpha_s):
    """Re-roll (deployed path) and split per-step singular/empty by region {V_M<=0 vs >0}; authority
    stats; discrete CBF-condition probe on {V_M<=0}. Aggregated over ACTIVE steps."""
    system = fw.system
    dev = system.u_bounds.device
    dtype = system.u_bounds.dtype
    agg = {k: 0.0 for k in (
        "active", "cert_active", "uncert_active",
        "sing_cert", "sing_uncert", "empty_cert", "empty_uncert",
        "probe_ok_cert", "lg_sum", "lg_sq", "lg_n")}
    lg_samples = []
    params = fw.params
    bounds = system.u_bounds.to(device=dev, dtype=dtype)
    for s0 in range(0, len(scenes), eval_batch):
        batch = scenes[s0:s0 + eval_batch]
        bscene = batch_scenes(batch, device=dev, dtype=dtype)
        x = initial_states_from_batch(bscene)
        states = [x]
        with torch.no_grad():
            for _ in range(max_steps):
                u_nom = fw.policy(x, bscene)
                u_safe, _ = fw.filter(x, u_nom, bscene)
                x = system.wrap_state(x)  # keep clamp semantics
                from src.common.rk4 import rk4_step
                x = rk4_step(system, x, u_safe, dt)
                states.append(x)
        S = torch.stack(states, dim=0)   # [T+1, B, 4]
        masks = step_outcomes(S, bscene, system, config)
        event = first_physical_event_step(masks)
        active_n = torch.where(event >= 0, event, torch.full_like(event, max_steps))
        for t in range(max_steps):
            xt = S[t]
            active_t = torch.tensor(t, device=dev) < active_n
            if not active_t.any():
                continue
            u_nom = fw.policy(xt, bscene)
            h, lf, lg = _cbf_terms(system, fw.h_fn, xt, bscene, u_nom, create_graph=False)
            h, lf, lg = h.detach(), lf.detach(), lg.detach()
            with torch.no_grad():
                alpha = _base_alpha(h, params)
                row_upper = -lf - alpha * h
                proj = _base_projection(u_nom, lg, row_upper, bounds, params)
                sing = torch.linalg.norm(lg, dim=1) < _SINGULAR_LG_THRESHOLD
                if params.box_aware:
                    u_safe, empty = _box_aware_projection(u_nom, proj, lg, row_upper, bounds)
                else:
                    u_safe, empty = proj, torch.zeros_like(sing)
                cert = (h <= 0.0)
                A = active_t
                Ac, Au = A & cert, A & (~cert)
                agg["active"] += float(A.sum())
                agg["cert_active"] += float(Ac.sum())
                agg["uncert_active"] += float(Au.sum())
                agg["sing_cert"] += float((sing & Ac).sum())
                agg["sing_uncert"] += float((sing & Au).sum())
                agg["empty_cert"] += float((empty & Ac).sum())
                agg["empty_uncert"] += float((empty & Au).sum())
                lgn = torch.linalg.norm(lg, dim=1)
                agg["lg_sum"] += float(lgn[A].sum()); agg["lg_sq"] += float((lgn[A] ** 2).sum())
                agg["lg_n"] += float(A.sum())
                if len(lg_samples) < 20000:
                    lg_samples.append(lgn[A].detach().cpu())
                # discrete CBF-condition probe on {V_M<=0} active steps
                from src.common.rk4 import rk4_step
                xn = rk4_step(system, xt, u_safe, dt)
                vn = maneuver_value(xn, bscene, system, config, lateral_js=fw.lateral_js)
                ok = vn <= (1.0 - alpha_s * dt) * h + 1.0e-9
                agg["probe_ok_cert"] += float((ok & Ac).sum())
    lgc = torch.cat(lg_samples) if lg_samples else torch.zeros(1)
    def rate(a, b): return agg[a] / agg[b] if agg[b] else 0.0
    return {
        "active_steps": int(agg["active"]),
        "cert_frac": rate("cert_active", "active"),
        "singular_rate_cert": rate("sing_cert", "cert_active"),
        "singular_rate_uncert": rate("sing_uncert", "uncert_active"),
        "empty_rate_cert": rate("empty_cert", "cert_active"),
        "empty_rate_uncert": rate("empty_uncert", "uncert_active"),
        "discrete_cbf_ok_frac_cert": rate("probe_ok_cert", "cert_active"),
        "lg_mean": agg["lg_sum"] / agg["lg_n"] if agg["lg_n"] else 0.0,
        "lg_p10": float(torch.quantile(lgc, 0.10)), "lg_p50": float(torch.quantile(lgc, 0.50)),
        "lg_p90": float(torch.quantile(lgc, 0.90)),
    }


def _premise(fw, config, pool_path):
    system = fw.system
    dev, dtype = system.u_bounds.device, system.u_bounds.dtype
    scenes = load_pool(pool_path).scenes
    bscene = batch_scenes(scenes, device=dev, dtype=dtype)
    x0 = initial_states_from_batch(bscene)
    with torch.no_grad():
        vb = maneuver_value(x0, bscene, system, config, lateral_js=())
        vm = maneuver_value(x0, bscene, system, config, lateral_js=fw.lateral_js)
    return {"n": len(scenes), "frac_Vbrake_le0": float((vb <= 0).float().mean()),
            "frac_VM_le0": float((vm <= 0).float().mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", choices=list(POOLS), required=True)
    ap.add_argument("--lateral-js", type=str, default="")   # e.g. "4,8"
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--eval-batch", type=int, default=250)
    ap.add_argument("--nominal-only", action="store_true", help="LQR-only baseline (no filter)")
    ap.add_argument("--out", type=Path, default=REPO / "docs/versions/v2.5.0")
    a = ap.parse_args()
    lateral_js = [int(s) for s in a.lateral_js.split(",") if s.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = _config()
    config["run"]["version"] = "v2.5.0"
    system = DoubleIntegrator(config)
    system.u_bounds = system.u_bounds.to(device)   # forces eval dtype/device (float64, cuda)
    fw = StageAFramework(system, config, lateral_js, nominal_only=a.nominal_only)
    dt = float(config["env"]["dt"])
    alpha_s = float(config["filter"]["alpha_safe"])
    max_steps = int(config["eval"]["max_steps"])
    pool_path = POOLS[a.pool]
    a.out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    premise = _premise(fw, config, pool_path) if not a.nominal_only else {"skipped": "nominal_only"}
    ev = evaluate(fw, pool_path, config, mode="stage_a", ckpt_name=f"{a.tag}",
                  eval_batch_size=a.eval_batch)
    row = ev.eval_row
    fw.n_calls = 0; fw.filter_time = 0.0
    region = ({} if a.nominal_only
              else _region_split(fw, config, load_pool(pool_path).scenes, max_steps, dt, a.eval_batch, alpha_s))
    wall = time.time() - t0

    out = {"tag": a.tag, "pool": a.pool, "lateral_js": lateral_js, "T_stop": t_stop(config, system),
           "premise": premise,
           "metrics": {k: row.get(k) for k in (
               "cps", "cps_ci_lo", "cps_ci_hi", "reach", "reach_ci_lo", "reach_ci_hi",
               "collision", "collision_ci_lo", "collision_ci_hi", "oob", "stuck",
               "stuck_ci_lo", "stuck_ci_hi", "timeout", "infeasibility",
               "infeasibility_ci_lo", "infeasibility_ci_hi", "n_scenes")},
           "region_split": region,
           "wallclock_s": wall,
           "filter_calls": region.get("active_steps"), }
    (a.out / f"{a.tag}.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
