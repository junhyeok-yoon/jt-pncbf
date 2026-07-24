"""v2.7.5 M3 — deployment latency: wall-clock of ONE control computation at batch=1 on the v2.7.4 checkpoint.

One control step = policy forward + V-hat gradient (the CBF term ∇h) + HardNet projection — the exact path
that runs each control period at deploy. Measured at batch=1 (the deploy condition), median + p95 over many
timed calls, against the 10 ms budget that 100 Hz implies. Reports device (GPU/host). A 100 Hz claim is void
if p95 exceeds 10 ms regardless of cps. dt_ctrl does not change the per-call cost (same forward+grad+project),
so latency is checkpoint/hardware-bound and measured once. No training, no secured_data writes.
"""
from __future__ import annotations

import json
import platform
import statistics
import time
from pathlib import Path

import torch

from src.eval.build_pools import load_pool, EVAL_POOLS_DIR
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.frameworks.jt_pncbf.train import load_framework_from_checkpoint

CKPT = Path("data/runs/v2.7.4/set__20260720-083533__seed42/v2.7.4__20260720-091830__seed42/checkpoints/best.pt")
POOL = EVAL_POOLS_DIR / "eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl"
OUTDIR = Path("data/runs/v2.7.5/dt_ctrl_arms"); OUTDIR.mkdir(parents=True, exist_ok=True)
N_WARMUP, N_TIMED = 50, 500

fw, cfg, ck = load_framework_from_checkpoint(CKPT)
system = fw.system
dev = next(fw.policy_net.parameters()).device if hasattr(fw, "policy_net") else torch.device("cpu")
scenes = load_pool(POOL).scenes[:1]                     # batch = 1 (deploy condition)
bs = batch_scenes(scenes, device=dev, dtype=torch.float32)
x = system.wrap_state(initial_states_from_batch(bs).float().to(dev))


def one_control_step(x, bs):
    # policy forward + filter (the filter internally computes the V-hat gradient ∇h and the HardNet projection)
    un = fw.policy(x, bs)
    u, _ = fw.filter(x, un, bs)
    return u


cuda = dev.type == "cuda"
for _ in range(N_WARMUP):
    one_control_step(x, bs)
if cuda:
    torch.cuda.synchronize()

times_ms = []
for _ in range(N_TIMED):
    if cuda:
        torch.cuda.synchronize(); t0 = time.perf_counter(); one_control_step(x, bs); torch.cuda.synchronize()
    else:
        t0 = time.perf_counter(); one_control_step(x, bs)
    times_ms.append((time.perf_counter() - t0) * 1e3)

times_ms.sort()
p = lambda q: times_ms[min(len(times_ms) - 1, int(q * len(times_ms)))]
gpu_name = torch.cuda.get_device_name(0) if cuda else "cpu-only"
out = {
    "batch": 1, "n_warmup": N_WARMUP, "n_timed": N_TIMED,
    "device": str(dev), "gpu": gpu_name, "host": platform.node(), "cpu": platform.processor() or platform.machine(),
    "budget_ms_100Hz": 10.0,
    "median_ms": round(statistics.median(times_ms), 4),
    "p95_ms": round(p(0.95), 4), "p99_ms": round(p(0.99), 4),
    "min_ms": round(times_ms[0], 4), "max_ms": round(times_ms[-1], 4),
    "H4_p95_under_budget": bool(p(0.95) < 10.0),
    "note": "one control step = policy forward + V-hat gradient + HardNet projection; batch=1 deploy condition.",
}
(OUTDIR / "latency.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
