"""v2.4.1 Stage 4b — control-deficit w_d ratio calibration.

On one fixed batch of 512 BPTT start states (v2.3.0 seed42 baseline net), compute
    g_task = || grad_theta( gated task + outside loss ) ||          (regs + deficit zeroed)
    g_def  = || grad_theta( w_d * L_deficit ) ||  =  || grad_W - grad_0 ||   (linearity, same batch)
    r = g_def / g_task
Adapt w_d until r in [0.01, 0.5]: r<0.01 -> w_d*=10 (cap 100); r>0.5 -> w_d/=10 (floor 1e-3).
If the batch has no box-binding step (g_def==0), rebuild it from fast-approach states.
Read-only (loads a frozen checkpoint); writes a calibration log/JSON under docs/versions/v2.4.1/.
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
LOW, HIGH = 0.01, 0.5
FLOOR, CAP = 1.0e-3, 100.0


def _task_only_cfg(config: dict, w_deficit: float, deficit_form: str = "sq_cap") -> dict:
    cfg = copy.deepcopy(config)
    p = cfg["loss"]["policy"]
    p["deficit_form"] = str(deficit_form)
    # keep task_cost (lambda_v/mu_u are inside task) + w_outside; zero the shaping regularizers.
    p["lambda_a"] = 0.0
    p["lambda_s"] = 0.0
    p["lambda_sat"] = 0.0
    p["lambda_pretanh"] = 0.0
    p["input_rate"] = {**p.get("input_rate", {}), "enabled": False}
    p["u_reg"] = {**p.get("u_reg", {}), "enabled": False}
    p["w_deficit"] = float(w_deficit)
    return cfg


def _grad_vec(framework, cfg, batch) -> torch.Tensor:
    policy = framework.policy_net
    policy.zero_grad(set_to_none=True)
    policy_bptt_loss(system=framework.system, policy_net=policy, value_net=framework.value_net,
                     batch=batch, config=cfg).total.backward()
    return torch.cat([p.grad.detach().flatten() for p in policy.parameters() if p.grad is not None])


def _active_frac(framework, cfg, batch) -> float:
    policy = framework.policy_net
    policy.zero_grad(set_to_none=True)
    res = policy_bptt_loss(system=framework.system, policy_net=policy, value_net=framework.value_net,
                           batch=batch, config=cfg)
    return float(res.deficit_active_frac)


def _build_batch(framework, config, n, dtype, device, seed):
    # Roll out the baseline policy over the in-loop pool and sample n states across trajectories
    # (mid-trajectory near-obstacle states exercise the CBF/box binding the deficit needs).
    system = framework.system
    scenes = load_pool(INLOOP_POOL).scenes
    batched = batch_scenes(scenes, device=device, dtype=dtype)
    x0 = initial_states_from_batch(batched)
    with torch.no_grad():
        res = rollout_eval(system, framework.policy, _filter_adapter(framework), batched, x0,
                           int(config["eval"]["max_steps"]), float(config["env"]["dt"]), config=config)
    T, B, _ = res.states.shape
    gen = torch.Generator(device="cpu").manual_seed(seed)
    bi = torch.randint(0, B, (n,), generator=gen)
    ti = torch.randint(0, T, (n,), generator=gen)
    states = res.states[ti, bi].to(device=device, dtype=dtype)
    bscene = batch_scenes([scenes[int(b)] for b in bi], device=device, dtype=dtype)
    return SimpleNamespace(states=states, scene=bscene)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=REPO / "data/secured_data/v2.3.0/seed42/checkpoints/best.pt")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--w-init", type=float, default=1.0)
    ap.add_argument("--deficit-form", type=str, default="sq_cap")
    ap.add_argument("--fixed", action="store_true", help="measure r at --w-init only (no adaptation)")
    ap.add_argument("--out", type=Path, default=REPO / "docs/versions/v2.4.1")
    args = ap.parse_args()

    torch.manual_seed(0)
    framework, config, _ = _load_framework(args.ckpt)
    framework.policy_net.requires_grad_(True)
    dtype, device = _tensor_options(framework.system, framework)

    batch = _build_batch(framework, config, args.n, dtype, device, seed=20)
    active_frac = _active_frac(framework, _task_only_cfg(config, 1.0, args.deficit_form), batch)
    # ensure the batch exercises the deficit; else resample with a different seed.
    tries = 0
    while active_frac <= 0.0 and tries < 5:
        tries += 1
        batch = _build_batch(framework, config, args.n, dtype, device, seed=20 + tries)
        active_frac = _active_frac(framework, _task_only_cfg(config, 1.0, args.deficit_form), batch)

    g0 = _grad_vec(framework, _task_only_cfg(config, 0.0, args.deficit_form), batch)          # task + outside only
    g_task = float(torch.linalg.norm(g0))

    w_d = float(args.w_init)
    table = []
    for _ in range(12):
        gW = _grad_vec(framework, _task_only_cfg(config, w_d, args.deficit_form), batch)
        g_def = float(torch.linalg.norm(gW - g0))                         # = ||w_d * grad L_deficit||
        r = g_def / g_task if g_task > 0 else float("inf")
        table.append({"w_d": w_d, "g_task": g_task, "g_def": g_def, "r": r})
        print(f"w_d={w_d:.4g}  g_task={g_task:.4g}  g_def={g_def:.4g}  r={r:.4g}  active_frac={active_frac:.3f}")
        if args.fixed:
            break
        if LOW <= r <= HIGH:
            break
        if r < LOW:
            if w_d >= CAP:
                break
            w_d = min(w_d * 10.0, CAP)
        else:
            if w_d <= FLOOR:
                break
            w_d = max(w_d / 10.0, FLOOR)

    final = table[-1]
    ok = LOW <= final["r"] <= HIGH
    result = {"ckpt": str(args.ckpt), "n": args.n, "deficit_active_frac": active_frac,
              "table": table, "adopted_w_d": final["w_d"], "final_r": final["r"], "in_range": ok}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "calib_4b_ratio.json").write_text(json.dumps(result, indent=2))
    print(f"\nADOPTED w_d={final['w_d']:.4g}  r={final['r']:.4g}  IN_RANGE={ok}  active_frac={active_frac:.3f}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
