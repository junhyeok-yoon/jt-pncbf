"""v2.4.2 R3 tail-state coverage — off-support elevation test (read-only, throwaway).

On the same fixed probe as tail_ratchet (pool starts + velocity spread), characterize the T_b-step
re-roll endpoints x_30 for two checkpoints: frac with true h(x_30)>0 (post-collision endpoints),
speed(x_30) summary, and the signed error (h-hat(x_30) - V_brake(x_30)) at endpoints vs at the
on-support reference (probe START states x_0). Endpoint error >> on-support error and growing = the
off-support-elevation mechanism (labels bootstrapped from states the deployed policy never visits).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from src.common.brake_rollout import brake_h_rollout
from src.common.rk4 import rk4_step
from src.common.signed_h import signed_h
from src.eval.build_pools import load_pool
from src.eval.evaluate import _tensor_options
from src.eval.run_full import _load_framework
from src.envs.scene_batch import batch_scenes, initial_states_from_batch

REPO = Path(__file__).resolve().parents[2]
POOL = REPO / "data/secured_data/pools/eval_inloop_di_n500_seed12345.pkl"


def _make_states(scenes, k, vmax, seed):
    g = torch.Generator().manual_seed(seed)
    sel = [si for si in range(len(scenes)) for _ in range(k)]
    n = len(sel)
    speed = torch.rand(n, generator=g) * vmax
    ang = torch.rand(n, generator=g) * 2.0 * math.pi
    v = torch.stack([speed * torch.cos(ang), speed * torch.sin(ang)], dim=1)
    starts = torch.tensor(np.array([scenes[i].start for i in sel]), dtype=torch.float64)
    return torch.cat([starts, v.to(torch.float64)], dim=1), [scenes[i] for i in sel]


def _q(a):
    a = np.asarray(a, dtype=float)
    return {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90))}


def _reroll_final(fw, cfg, states, batched):
    system = fw.system
    dtype, device = _tensor_options(system, fw)
    dt = float(cfg["env"]["dt"])
    T_b = int(cfg["value_target"].get("raw_lagged", {}).get("T_b", 30))
    ub = system.u_bounds.to(device=device, dtype=dtype)
    lo, hi = ub[:, 0], ub[:, 1]
    x = states.to(device=device, dtype=dtype)
    with torch.no_grad():
        for _ in range(T_b):
            u = torch.clamp(fw.policy_net(system.observation(x, batched)), min=lo, max=hi)
            x = rk4_step(system, x, u, dt)
    return x


def _vbrake(fw, cfg, x, batched):
    system = fw.system
    u_max = float(cfg["env"]["bounds"][system.name]["u_max"])
    eps_v = float(cfg["value_target"]["brake"]["eps_v"])
    T_b = max(30, int(cfg["value_target"]["brake"]["T_b"]))
    dt = float(cfg["env"]["dt"])
    h_scale = float(cfg["env"]["h_scale"])
    h_seq, _ = brake_h_rollout(x, batched, system, system.observation, T_b, u_max, eps_v, dt, h_scale)
    return torch.clamp(h_seq, -1.0, 1.0).max(dim=1).values


def _analyze(ckpt, states, scenes_sel):
    fw, cfg, ck = _load_framework(ckpt)
    system = fw.system
    dtype, device = _tensor_options(system, fw)
    batched = batch_scenes(scenes_sel, device=device, dtype=dtype)
    x_ref = initial_states_from_batch(batched)   # true pool-start reference (on-support, deployment start)
    h_scale = float(cfg["env"]["h_scale"])
    x30 = _reroll_final(fw, cfg, states, batched)   # endpoints from the velocity-spread re-roll
    with torch.no_grad():
        h_end = signed_h(system.position(x30), batched, h_scale).reshape(-1)
        spd = system.speed(x30).reshape(-1)
        hhat_end = fw._filter.h_fn(x30, batched).reshape(-1)
        vb_end = _vbrake(fw, cfg, x30, batched)
        hhat_0 = fw._filter.h_fn(x_ref, batched).reshape(-1)
        vb_0 = _vbrake(fw, cfg, x_ref, batched)
    err_end = (hhat_end - vb_end).cpu().numpy()
    err_0 = (hhat_0 - vb_0).cpu().numpy()
    return {
        "step": int(ck.get("step", -1)),
        "frac_h_end_gt0": float((h_end > 0).float().mean().item()),
        "speed_end": _q(spd.cpu().numpy()),
        "err_endpoint_x30": _q(err_end),
        "err_onsupport_start_x0": _q(err_0),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="tail-state coverage / off-support elevation (read-only).")
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
           "early": _analyze(a.early, states, scenes_sel),
           "late": _analyze(a.late, states, scenes_sel)}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "tail_state_coverage.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
