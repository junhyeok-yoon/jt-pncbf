"""v2.4.1 Stage 4c — grad-tail gate for the control-deficit loss.

Per-sample policy-gradient norms of the PRODUCTION policy_bptt_loss at T=30 with w_deficit ON
(calibrated) vs OFF (unmodified baseline), on rollout-sampled start states from the v2.3.0 seed42
net (same sampling as the v2.4.0 audit). PASS iff, relative to the OFF baseline at T=30,
    p99_on <= 3 * p99_off   AND   max_on <= 5 * max_off .
Also reports the v2.4.0 audit's fixed unmodified-T=30 reference (p99 173, max 8481) for continuity.
Read-only.
"""
from __future__ import annotations

import argparse
import copy
import json
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
N_START = 512
REF_T30 = {"p99": 173.0, "max": 8481.1}   # v2.4.0 audit unmodified T=30 (task-return persample)


def _cfg(config, w_deficit, bptt_t=30, deficit_form="sq_cap"):
    cfg = copy.deepcopy(config)
    cfg["loss"]["policy"]["w_deficit"] = float(w_deficit)
    cfg["loss"]["policy"]["deficit_form"] = str(deficit_form)
    cfg["training"]["jt"]["bptt_T"] = int(bptt_t)
    return cfg


def _per_sample(framework, cfg, starts, start_scenes, device, dtype):
    policy = framework.policy_net
    policy.requires_grad_(True)
    norms = []
    for i in range(len(starts)):
        batch = SimpleNamespace(states=starts[i:i + 1],
                                scene=batch_scenes([start_scenes[i]], device=device, dtype=dtype))
        policy.zero_grad(set_to_none=True)
        policy_bptt_loss(system=framework.system, policy_net=policy, value_net=framework.value_net,
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
    ap.add_argument("--w-deficit", type=float, default=1.0)
    ap.add_argument("--deficit-form", type=str, default="sq_cap")
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

    off = _stats(_per_sample(framework, _cfg(config, 0.0, deficit_form=args.deficit_form),
                             starts, start_scenes, device, dtype))
    on = _stats(_per_sample(framework, _cfg(config, args.w_deficit, deficit_form=args.deficit_form),
                            starts, start_scenes, device, dtype))
    p99_ok = on["p99"] <= 3.0 * off["p99"]
    max_ok = on["max"] <= 5.0 * off["max"]
    passed = p99_ok and max_ok
    result = {"ckpt": str(args.ckpt), "w_deficit": args.w_deficit, "T": 30,
              "off_w0": off, "on": on, "ref_audit_T30": REF_T30,
              "p99_ratio": on["p99"] / off["p99"], "max_ratio": on["max"] / off["max"],
              "gate_p99_le_3x": p99_ok, "gate_max_le_5x": max_ok, "passed": passed}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "calib_4c_gradtail.json").write_text(json.dumps(result, indent=2))
    print(f"T=30 per-sample policy grad (N={N_START}):")
    print(f"  OFF (w_d=0): {off}")
    print(f"  ON  (w_d={args.w_deficit}): {on}")
    print(f"  ratios: p99 {result['p99_ratio']:.2f}x (<=3), max {result['max_ratio']:.2f}x (<=5)")
    print(f"  audit fixed ref T30: {REF_T30}")
    print(f"  GATE: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
