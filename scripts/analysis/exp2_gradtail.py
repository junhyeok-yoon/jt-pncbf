"""v2.4.1 Exp 2 — grad-tail gate for the obs-deficit channel.

Per-sample policy-gradient norms of the PRODUCTION policy_bptt_loss at T=30:
  OFF = the v2.3.0 dim-19 policy (obs_deficit off);
  ON  = the SAME policy extended to dim-21 with LIVE deficit input columns (weight 0.3) and
        obs_deficit_feedback on, so delta_u_{t-1} actually feeds the policy.
Because the deficit feature is fed DETACHED, its gradient does NOT traverse the filter
coefficient Jacobian (the Huber failure mode); the tail should stay near baseline.
PASS iff p99_on <= 3*p99_off AND max_on <= 5*max_off (and reported vs the audit ref 173/8481).
Read-only.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from src.common.control_net import ControlNet
from src.eval.build_pools import load_pool
from src.eval.evaluate import _filter_adapter, _tensor_options
from src.eval.rollout import rollout_eval
from src.eval.run_full import _load_framework
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.frameworks.jt_pncbf.losses import grad_norm, policy_bptt_loss

REPO = Path(__file__).resolve().parents[2]
INLOOP_POOL = REPO / "data/secured_data/pools/eval_inloop_di_n500_seed12345.pkl"
N_START = 512
REF_T30 = {"p99": 173.0, "max": 8481.1}


def _cfg(config, obs_deficit):
    cfg = copy.deepcopy(config)
    cfg["loss"]["policy"] = {**cfg["loss"]["policy"], "w_deficit": 0.0,
                             "obs_deficit_feedback": bool(obs_deficit)}
    cfg["training"]["jt"] = {**cfg["training"]["jt"], "bptt_T": 30}
    return cfg


def _dim21_policy(system, config, base: ControlNet, deficit_w=0.3):
    net = ControlNet(system.obs_dim + system.action_dim, system, config).to(
        dtype=next(base.parameters()).dtype)
    with torch.no_grad():
        for name, p in base.named_parameters():
            if name == "trunk.0.weight":
                net.trunk[0].weight[:, : system.obs_dim] = p
                net.trunk[0].weight[:, system.obs_dim :] = deficit_w   # LIVE deficit columns
            else:
                dict(net.named_parameters())[name].copy_(p)
    return net


def _per_sample(system, policy, value_net, cfg, starts, start_scenes, device, dtype):
    policy.requires_grad_(True)
    norms = []
    for i in range(len(starts)):
        batch = SimpleNamespace(states=starts[i:i + 1],
                                scene=batch_scenes([start_scenes[i]], device=device, dtype=dtype))
        policy.zero_grad(set_to_none=True)
        policy_bptt_loss(system=system, policy_net=policy, value_net=value_net,
                         batch=batch, config=cfg).total.backward()
        norms.append(grad_norm(policy.parameters()))
    return torch.tensor(norms)


def _stats(v):
    v = v.sort().values
    return {"median": float(v.median()), "p90": float(v[int(0.9 * len(v))]),
            "p99": float(v[int(0.99 * len(v))]), "max": float(v.max())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=REPO / "data/secured_data/v2.3.0/seed42/checkpoints/best.pt")
    ap.add_argument("--deficit-weight", type=float, default=0.3)
    ap.add_argument("--out", type=Path, default=REPO / "docs/versions/v2.4.1")
    args = ap.parse_args()

    framework, config, _ = _load_framework(args.ckpt)
    system = framework.system
    dtype, device = _tensor_options(system, framework)
    scenes = load_pool(INLOOP_POOL).scenes
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

    off = _stats(_per_sample(system, framework.policy_net, framework.value_net,
                             _cfg(config, False), starts, start_scenes, device, dtype))
    policy21 = _dim21_policy(system, _cfg(config, True), framework.policy_net, deficit_w=args.deficit_weight).to(device)
    on = _stats(_per_sample(system, policy21, framework.value_net,
                            _cfg(config, True), starts, start_scenes, device, dtype))
    p99_ok = on["p99"] <= 3.0 * off["p99"]
    max_ok = on["max"] <= 5.0 * off["max"]
    passed = p99_ok and max_ok
    result = {"ckpt": str(args.ckpt), "T": 30, "off_dim19": off, "on_dim21_obs_deficit": on,
              "ref_audit_T30": REF_T30, "p99_ratio": on["p99"] / off["p99"],
              "max_ratio": on["max"] / off["max"], "passed": passed}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "exp2_gradtail.json").write_text(json.dumps(result, indent=2))
    print(f"T=30 per-sample policy grad (N={N_START}):")
    print(f"  OFF (dim-19):            {off}")
    print(f"  ON  (dim-21 obs_deficit): {on}")
    print(f"  ratios: p99 {result['p99_ratio']:.2f}x (<=3), max {result['max_ratio']:.2f}x (<=5)")
    print(f"  audit ref T30: {REF_T30}   GATE: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
