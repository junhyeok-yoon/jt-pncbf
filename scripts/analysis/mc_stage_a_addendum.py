"""Stage-A addendum: per-episode region-tagged re-roll for like-for-like cps (G2), premise split
(G3), and shift-closed arm + invariance violations (G4). One arm per invocation, n2000, no training.

Reuses StageAFramework (clamped LQR + HardNet(V_M)); rolls the deployed path, then a per-step pass
tags each ACTIVE step with {infeasible=singular|empty, empty, active_band=infeasible & V_M>-1+delta}
and V_M(x_k) (for cert-invariance violations V_M(x_k)<=0 & V_M(x_{k+1})>0). Emits per-episode metrics
and the pool-level cps under three infeas conventions (reach/coll/stuck/oob/timeout unchanged).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.common.filter_hardnet import (
    _SINGULAR_LG_THRESHOLD, _base_alpha, _base_projection, _box_aware_projection,
    _cbf_terms, _empty_halfspace_box,
)
from src.common.maneuver_value import maneuver_value
from src.common.outcomes import resolve_outcome, step_outcomes
from src.common.rk4 import rk4_step
from src.eval.build_pools import load_pool
from src.eval.evaluate import first_physical_event_step
from src.envs.double_integrator import DoubleIntegrator
from src.envs.scene_batch import batch_scenes, initial_states_from_batch

import scripts.analysis.mc_stage_a as base

REPO = Path(__file__).resolve().parents[2]
DELTA = 0.05


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lateral-js", type=str, default="")
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--pool", choices=list(base.POOLS), default="n2000")
    ap.add_argument("--eval-batch", type=int, default=250)
    ap.add_argument("--gamma-m", type=float, default=0.0)
    ap.add_argument("--out", type=Path, default=REPO / "docs/versions/v2.5.0")
    a = ap.parse_args()
    lateral_js = [int(s) for s in a.lateral_js.split(",") if s.strip()]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = base._config(); cfg["run"]["version"] = "v2.5.0"
    system = DoubleIntegrator(cfg); system.u_bounds = system.u_bounds.to(dev)
    fw = base.StageAFramework(system, cfg, lateral_js, gamma_m=a.gamma_m)
    dt = float(cfg["env"]["dt"]); max_steps = int(cfg["eval"]["max_steps"])
    alpha_s = float(cfg["filter"]["alpha_safe"])
    reg = {k: 0.0 for k in ("cert_n", "sing_cert", "emp_cert", "ok_cert")}
    bounds = system.u_bounds
    scenes = load_pool(base.POOLS[a.pool]).scenes

    ep = []  # per-episode dicts
    for s0 in range(0, len(scenes), a.eval_batch):
        chunk = scenes[s0:s0 + a.eval_batch]
        bscene = batch_scenes(chunk, device=dev, dtype=bounds.dtype)
        x = initial_states_from_batch(bscene)
        with torch.no_grad():
            v_start = maneuver_value(x, bscene, system, cfg, lateral_js=lateral_js)
        states = [x]
        with torch.no_grad():
            for _ in range(max_steps):
                u_nom = fw.policy(x, bscene)
                u_safe, _ = fw.filter(x, u_nom, bscene)
                x = rk4_step(system, x, u_safe, dt)
                states.append(x)
        S = torch.stack(states, dim=0)                        # [T+1, B, 4]
        masks = step_outcomes(S, bscene, system, cfg)
        resolved = resolve_outcome(masks)
        event = first_physical_event_step(masks)
        active_n = torch.where(event >= 0, event, torch.full_like(event, max_steps))
        B = x.shape[0]
        # per-step region flags over the whole horizon (active mask applied per-episode later)
        inf_c = torch.zeros(max_steps, B, dtype=torch.bool, device=dev)
        emp = torch.zeros(max_steps, B, dtype=torch.bool, device=dev)
        sng = torch.zeros(max_steps, B, dtype=torch.bool, device=dev)
        vm = torch.zeros(max_steps + 1, B, dtype=bounds.dtype, device=dev)
        vm[0] = v_start
        for t in range(max_steps):
            xt = S[t]
            u_nom = fw.policy(xt, bscene)
            h, lf, lg = _cbf_terms(system, fw.h_fn, xt, bscene, u_nom, create_graph=False)
            h, lf, lg = h.detach() + fw.gamma_m, lf.detach(), lg.detach()   # h_eff matches the filter
            with torch.no_grad():
                alpha = _base_alpha(h, fw.params)
                row = -lf - alpha * h
                proj = _base_projection(u_nom, lg, row, bounds, fw.params)
                sing = torch.linalg.norm(lg, dim=1) < _SINGULAR_LG_THRESHOLD
                _, empty = _box_aware_projection(u_nom, proj, lg, row, bounds)
                inf_c[t] = sing | empty
                emp[t] = empty
                sng[t] = sing
                vm[t + 1] = maneuver_value(S[t + 1], bscene, system, cfg, lateral_js=lateral_js)
        vm_step = vm[:max_steps]                               # V_M(x_t) at each active step t
        t_idx = torch.arange(max_steps, device=dev).unsqueeze(1)
        active = t_idx < active_n.unsqueeze(0)
        band = inf_c & (vm_step > (-1.0 + DELTA))
        inviol = (vm[:max_steps] <= 0.0) & (vm[1:max_steps + 1] > 0.0) & active
        cert_mask = (vm_step <= 0.0) & active
        okmask = vm[1:max_steps + 1] <= (1.0 - alpha_s * dt) * vm_step + 1.0e-9
        reg["cert_n"] += float(cert_mask.sum())
        reg["sing_cert"] += float((sng & cert_mask).sum())
        reg["emp_cert"] += float((emp & cert_mask).sum())
        reg["ok_cert"] += float((okmask & cert_mask).sum())
        for i in range(B):
            outc = resolved.outcome[i]
            na = int(active_n[i].item())
            am = active[:, i]
            n_act = int(am.sum().item())
            def rate(flag):
                return float((flag[:, i] & am).sum().item()) / n_act if n_act else 0.0
            ep.append({
                "outcome": outc, "n_active": n_act, "start_cert": bool(v_start[i].item() <= 0.0),
                "start_speed": float(system.speed(states[0][i:i + 1]).item()),
                "inf_canonical": rate(inf_c), "inf_empty": rate(emp), "inf_band": rate(band),
                "inviol": int(inviol[:, i].sum().item()),
            })

    def orate(o): return float(np.mean([1.0 if e["outcome"] == o else 0.0 for e in ep]))
    reach, coll, oob = orate("goal"), orate("collision"), orate("oob")
    stuck, timeout = orate("stuck"), orate("timeout")
    def infrate(key): return float(np.mean([e[key] for e in ep]))
    inf_can, inf_emp, inf_band = infrate("inf_canonical"), infrate("inf_empty"), infrate("inf_band")
    def cps(inf): return reach - 2 * coll - stuck - 0.5 * (oob + timeout) - 0.3 * inf
    cert = [e for e in ep if e["start_cert"]]; unc = [e for e in ep if not e["start_cert"]]
    def split(grp, o): return float(np.mean([1.0 if e["outcome"] == o else 0.0 for e in grp])) if grp else None
    def rrate(k): return reg[k] / reg["cert_n"] if reg["cert_n"] else 0.0
    out = {
        "tag": a.tag, "lateral_js": lateral_js, "gamma_m": a.gamma_m, "n_episodes": len(ep),
        "region_cert": {"singular_cert": rrate("sing_cert"), "empty_cert": rrate("emp_cert"),
                        "discrete_cbf_ok_cert": rrate("ok_cert")},
        "outcomes": {"reach": reach, "collision": coll, "oob": oob, "stuck": stuck, "timeout": timeout},
        "infeas_rates": {"canonical": inf_can, "empty_only": inf_emp, "active_band": inf_band},
        "cps": {"canonical": cps(inf_can), "empty_only": cps(inf_emp), "active_band": cps(inf_band)},
        "premise_split": {
            "n_cert": len(cert), "n_uncert": len(unc),
            "cert": {o: split(cert, o) for o in ("goal", "collision", "stuck", "timeout", "oob")},
            "uncert": {o: split(unc, o) for o in ("goal", "collision", "stuck", "timeout", "oob")},
            "uncert_start_speed_mean": float(np.mean([e["start_speed"] for e in unc])) if unc else None,
            "uncert_start_speed_p90": float(np.percentile([e["start_speed"] for e in unc], 90)) if unc else None,
        },
        "cert_invariance_violations_total": int(sum(e["inviol"] for e in ep)),
        "cert_invariance_violation_episodes": int(sum(1 for e in ep if e["inviol"] > 0)),
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{a.tag}.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
