"""v2.4.2 R4 counterfactual tail cross (read-only, throwaway).

Attribute the 6.6x tail-push growth to schedule-vs-V. On the same fixed probe as tail_ratchet, hold
the re-roll FIXED (late 045000 policy => one h-sequence + one x_30 per probe state), and cross:
{schedule constants (gamma,target_rhs) of 016500, of 045000} x {V_target net of 016500, of 045000}.
Cell value = mean of the tail push target_rhs * relu(rhs_full - lhs), with lhs/int_rhs at the cell's
gamma and rhs_full = int_rhs[0] + gamma^Tb * V_targetNet(x_30). (Re-roll fixed to the late policy so the
endpoint states x_30 are the late-policy off-support endpoints where elevation is measured.)
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from src.common.brake_rollout import brake_h_rollout
from src.eval.build_pools import load_pool
from src.eval.evaluate import _tensor_options
from src.eval.run_full import _load_framework
from src.envs.scene_batch import batch_scenes
from src.frameworks.oc_pncbf.value_target import compute_disc_avoid_terms

REPO = Path(__file__).resolve().parents[2]
POOL = REPO / "data/secured_data/pools/eval_inloop_di_n500_seed12345.pkl"


def _sched_at(run_dir, step):
    rows = list(csv.DictReader(open(run_dir / "metrics.csv")))
    by = {int(r["step"]): r for r in rows}
    ks = sorted(by)
    i = bisect.bisect_left(ks, step)
    cand = [ks[j] for j in (i - 1, i) if 0 <= j < len(ks)]
    r = by[min(cand, key=lambda x: abs(x - step))]
    return float(r["gamma_disc_active"]), float(r["target_rhs_active"])


def _make_states(scenes, k, vmax, seed):
    g = torch.Generator().manual_seed(seed)
    sel = [si for si in range(len(scenes)) for _ in range(k)]
    n = len(sel)
    speed = torch.rand(n, generator=g) * vmax
    ang = torch.rand(n, generator=g) * 2.0 * math.pi
    v = torch.stack([speed * torch.cos(ang), speed * torch.sin(ang)], dim=1)
    starts = torch.tensor(np.array([scenes[i].start for i in sel]), dtype=torch.float64)
    return torch.cat([starts, v.to(torch.float64)], dim=1), [scenes[i] for i in sel]


def main() -> int:
    ap = argparse.ArgumentParser(description="counterfactual tail cross (read-only).")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--early", type=Path, required=True)
    ap.add_argument("--late", type=Path, required=True)
    ap.add_argument("--early-step", type=int, default=16500)
    ap.add_argument("--late-step", type=int, default=45000)
    ap.add_argument("--pool", type=Path, default=POOL)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=REPO / "docs/versions/v2.4.2")
    a = ap.parse_args()

    scenes = load_pool(a.pool).scenes
    states, scenes_sel = _make_states(scenes, a.k, 2.0, a.seed)
    fw_l, cfg_l, _ = _load_framework(a.late)
    fw_e, _, _ = _load_framework(a.early)
    system = fw_l.system
    dtype, device = _tensor_options(system, fw_l)
    batched = batch_scenes(scenes_sel, device=device, dtype=dtype)
    dt = float(cfg_l["env"]["dt"])
    h_scale = float(cfg_l["env"]["h_scale"])
    u_max = float(cfg_l["env"]["bounds"][system.name]["u_max"])
    T_b = int(cfg_l["value_target"].get("raw_lagged", {}).get("T_b", 30))
    ub = system.u_bounds.to(device=device, dtype=dtype)
    lo, hi = ub[:, 0], ub[:, 1]
    x = states.to(device=device, dtype=dtype)

    def _late_pol(_x, obs):
        return torch.clamp(fw_l.policy_net(obs), min=lo, max=hi)

    with torch.no_grad():
        h_seq, tail_obs = brake_h_rollout(x, batched, system, system.observation,
                                          T_b, u_max, 0.0, dt, h_scale, policy_fn=_late_pol)
        costs = torch.clamp(h_seq, -1.0, 1.0).transpose(0, 1).contiguous()   # [T_b+1, B]
        sched = {"S_early": _sched_at(a.run_dir, a.early_step), "S_late": _sched_at(a.run_dir, a.late_step)}
        tails = {"V_early": fw_e.value_net.target_h(tail_obs).reshape(-1),
                 "V_late": fw_l.value_net.target_h(tail_obs).reshape(-1)}
        cells = {}
        for sname, (gamma, trhs) in sched.items():
            lam = -math.log(gamma) / dt
            lhs, int_rhs, disc = compute_disc_avoid_terms(costs, lam, dt)
            for vname, tail in tails.items():
                rhs_full = int_rhs[0] + disc[0] * tail
                push = trhs * torch.relu(rhs_full - lhs[0])
                cells[f"{sname} x {vname}"] = float(push.mean().item())

    out = {"run_dir": str(a.run_dir), "reroll_policy": "late (fixed)", "n_probe": int(x.shape[0]),
           "schedule": {k: {"gamma": v[0], "target_rhs": v[1]} for k, v in sched.items()},
           "V_target_mean": {k: float(v.mean().item()) for k, v in tails.items()},
           "tail_push_cells": cells}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "tail_cross.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
