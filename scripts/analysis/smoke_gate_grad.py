"""Quantitative smoke gate for the C1 detach (v2.4.0 Step 5): per-sample policy-gradient norms of
the PRODUCTION policy_bptt_loss at T=60 with detach_filter_coeffs on/off, on the Step 4
step_009000.pt start states. PASS iff (flag on) per-sample max <= 5000 AND median within 2x of the
unmodified scratch T=60 median (34.0). Read-only.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

from src.eval.build_pools import load_pool
from src.eval.evaluate import _filter_adapter, _tensor_options
from src.eval.rollout import rollout_eval
from src.eval.run_full import _load_framework
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.frameworks.jt_pncbf.losses import grad_norm, policy_bptt_loss

REPO = Path(__file__).resolve().parents[2]
INLOOP_POOL = REPO / "data/secured_data/pools/eval_inloop_di_n500_seed12345.pkl"
N_START, REF_MEDIAN = 512, 34.0


def per_sample_grads(framework, config, starts, start_scenes, detach):
    policy = framework.policy_net
    policy.requires_grad_(True)
    cfg = {**config}
    cfg["loss"] = {**config["loss"], "policy": {**config["loss"]["policy"], "detach_filter_coeffs": detach}}
    cfg["training"] = {**config["training"], "jt": {**config["training"]["jt"], "bptt_T": 60}}
    norms = []
    for i in range(len(starts)):
        batch = SimpleNamespace(states=starts[i:i + 1], scene=batch_scenes([start_scenes[i]],
                                device=starts.device, dtype=starts.dtype))
        policy.zero_grad(set_to_none=True)
        policy_bptt_loss(system=framework.system, policy_net=policy, value_net=framework.value_net,
                         batch=batch, config=cfg).total.backward()
        norms.append(grad_norm(policy.parameters()))
    return torch.tensor(norms)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--pool", type=Path, default=INLOOP_POOL)
    args = ap.parse_args()
    framework, config, _ = _load_framework(args.ckpt)
    system = framework.system
    dtype, device = _tensor_options(system, framework)
    scenes = load_pool(args.pool).scenes
    batched = batch_scenes(scenes, device=device, dtype=dtype)
    x0 = initial_states_from_batch(batched)
    with torch.no_grad():
        res = rollout_eval(system, framework.policy, _filter_adapter(framework), batched, x0,
                           int(config["eval"]["max_steps"]), float(config["env"]["dt"]), config=config)
    T, B, _ = res.states.shape
    gen = torch.Generator(device="cpu").manual_seed(20)
    bi = torch.randint(0, B, (N_START,), generator=gen)
    ti = torch.randint(0, T, (N_START,), generator=gen)
    starts = res.states[ti, bi].to(device)
    start_scenes = [scenes[int(b)] for b in bi]

    def summ(v):
        v = v.sort().values
        return {"median": float(v.median()), "p90": float(v[int(0.9 * len(v))]),
                "p99": float(v[int(0.99 * len(v))]), "max": float(v.max())}

    on = summ(per_sample_grads(framework, config, starts, start_scenes, True))
    off = summ(per_sample_grads(framework, config, starts, start_scenes, False))
    passed = (on["max"] <= 5000.0) and (REF_MEDIAN / 2.0 <= on["median"] <= REF_MEDIAN * 2.0)
    print(f"PRODUCTION policy_bptt_loss T=60 per-sample grad norm (N={N_START}):")
    print(f"  detach ON : {on}")
    print(f"  detach OFF: {off}")
    print(f"  GATE: max_on={on['max']:.1f} (<=5000) median_on={on['median']:.2f} "
          f"(in [{REF_MEDIAN/2:.1f},{REF_MEDIAN*2:.1f}]) -> {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
