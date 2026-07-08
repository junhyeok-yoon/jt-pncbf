"""v2.4.2 forensics G — tail-ratchet quantification (read-only, throwaway).

For a FIXED probe batch (pool-scene starts + sampled velocity spread), reconstruct the
raw_lagged label pipeline for two checkpoints using each checkpoint's OWN policy as the pi_b
proxy (the true pi_b lags by tau_b=0.005 => tiny lag; PROXY CAVEAT recorded) and its value net:
a T_b-step unfiltered deterministic re-roll -> lhs (discounted-avoid of the h-sequence at the
schedule-active gamma of that step) and rhs_full = int_rhs + gamma^T_b * V_target(x_T). Reports
the one-sided tail push target_rhs*relu(rhs_full-lhs), fraction rhs_full>lhs, signed mean, and the
label shift Delta_y = y_full - lhs (== y with target_rhs forced to 0). Growing push 016500->045000
dominated by an elevated bootstrap tail = H2' ratchet confirmed.
"""
from __future__ import annotations

import argparse
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


def _sched_at(run_dir: Path, step: int):
    import bisect
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
    states = torch.cat([starts, v.to(torch.float64)], dim=1)
    return states, [scenes[i] for i in sel]


def _q(a):
    a = np.asarray(a, dtype=float)
    return {"n": int(a.size), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90))}


def _analyze(ckpt: Path, run_dir: Path, states, scenes_sel):
    fw, cfg, ck = _load_framework(ckpt)
    system = fw.system
    dtype, device = _tensor_options(system, fw)
    step = int(ck.get("step", -1))
    gamma, trhs = _sched_at(run_dir, step)
    dt = float(cfg["env"]["dt"])
    h_scale = float(cfg["env"]["h_scale"])
    u_max = float(cfg["env"]["bounds"][system.name]["u_max"])
    T_b = int(cfg["value_target"].get("raw_lagged", {}).get("T_b", 30))
    batched = batch_scenes(scenes_sel, device=device, dtype=dtype)
    ub = system.u_bounds.to(device=device, dtype=dtype)
    lo, hi = ub[:, 0], ub[:, 1]
    x = states.to(device=device, dtype=dtype)

    def pol(_x, obs):
        return torch.clamp(fw.policy_net(obs), min=lo, max=hi)

    with torch.no_grad():
        h_seq, tail_obs = brake_h_rollout(x, batched, system, system.observation,
                                          T_b, u_max, 0.0, dt, h_scale, policy_fn=pol)
        costs = torch.clamp(h_seq, -1.0, 1.0).transpose(0, 1).contiguous()   # [T_b+1, B]
        lam = -math.log(gamma) / dt
        lhs, int_rhs, disc = compute_disc_avoid_terms(costs, lam, dt)
        lhs0, int0, disc0 = lhs[0], int_rhs[0], disc[0]                       # disc0 = gamma^T_b
        tail = fw.value_net.target_h(tail_obs).reshape(-1)
        rhs_full = int0 + disc0 * tail
        diff = (rhs_full - lhs0)
        push = trhs * torch.relu(diff)
        y_full = torch.clamp(lhs0 + push, -1.0, 1.0)
        y_tailoff = torch.clamp(lhs0, -1.0, 1.0)
        delta_y = y_full - y_tailoff
    return {
        "checkpoint_step": step, "gamma_disc_active": gamma, "target_rhs_active": trhs,
        "gamma_pow_Tb": float(disc0.mean().item()), "T_b": T_b,
        "frac_rhs_gt_lhs": float((rhs_full > lhs0).float().mean().item()),
        "mean_diff_signed": float(diff.mean().item()),
        "mean_tail_Vtarget": float(tail.mean().item()),
        "mean_lhs": float(lhs0.mean().item()),
        "mean_rhs_full": float(rhs_full.mean().item()),
        "tail_push": _q(push.cpu().numpy()),
        "delta_y": _q(delta_y.cpu().numpy()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="tail-ratchet quantification (read-only).")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--early", type=Path, required=True)
    ap.add_argument("--late", type=Path, required=True)
    ap.add_argument("--pool", type=Path, default=POOL)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=REPO / "docs/versions/v2.4.2")
    a = ap.parse_args()
    scenes = load_pool(a.pool).scenes
    states, scenes_sel = _make_states(scenes, a.k, 2.0, a.seed)
    out = {"pool": str(a.pool), "n_probe": int(states.shape[0]),
           "proxy_note": "pi_b approximated by each checkpoint's own policy (tau_b=0.005 => tiny lag)",
           "early": _analyze(a.early, a.run_dir, states, scenes_sel),
           "late": _analyze(a.late, a.run_dir, states, scenes_sel)}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "tail_ratchet.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
