"""Stage A — per-step closed-loop Jacobian audit for the bptt_T=60 gradient explosion (read-only).

Along T=60 windows under pi(ckpt)+HardNet+RK4, estimate the per-step spectral norm ||J_k||_2 of
x_{k+1} = rk4_step(x_k, HardNet(x_k, pi(obs(x_k)))) w.r.t. x_k, and record the step's filter/CBF
features. ||J_k||_2 is estimated by a finite-difference random-probe power iteration (robust; a
LOWER BOUND on the true spectral norm — no double-backward through the filter's internal
autograd(L_g h)). Per-window Lyapunov exponent = mean_k log||J_k||.

Start states are a visited-state proxy for the training D_pi path (deterministic eval rollout;
CAVEAT: no sigma_pi exploration noise, so a lower bound on the pathological-state share).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from src.common.filter_hardnet import _cbf_terms, _base_alpha, _hardnet_params, _SINGULAR_LG_THRESHOLD
from src.common.rk4 import rk4_step
from src.common.value_net import make_h_fn
from src.eval.build_pools import load_pool
from src.eval.evaluate import _filter_adapter, _tensor_options
from src.eval.rollout import rollout_eval
from src.eval.run_full import _load_framework
from src.envs.scene_batch import batch_scenes, initial_states_from_batch

REPO = Path(__file__).resolve().parents[2]
INLOOP_POOL = REPO / "data/secured_data/pools/eval_inloop_di_n500_seed12345.pkl"
N_WINDOWS, T_FULL, N_PROBES, EPS = 256, 60, 10, 1.0e-4


def sigma_max(step_fn, x, n_probes=N_PROBES, eps=EPS):
    """Finite-difference random-probe power estimate of ||J||_2 per sample (block-diagonal J)."""
    with torch.no_grad():
        fx = step_fn(x)
        best = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        best_v = None
        for _ in range(n_probes):
            v = torch.randn_like(x)
            v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)
            jv = (step_fn(x + eps * v) - fx) / eps
            s = jv.norm(dim=1)
            take = s > best
            best = torch.where(take, s, best)
            best_v = v if best_v is None else torch.where(take.unsqueeze(1), v, best_v)
        # 2 power-refinement steps on the best direction
        for _ in range(2):
            jv = (step_fn(x + eps * best_v) - fx) / eps
            best_v = jv / jv.norm(dim=1, keepdim=True).clamp_min(1e-12)
            s = ((step_fn(x + eps * best_v) - fx) / eps).norm(dim=1)
            best = torch.maximum(best, s)
        return best


def audit(ckpt: Path, pool_path: Path):
    framework, config, checkpoint = _load_framework(ckpt)
    system = framework.system
    dtype, device = _tensor_options(system, framework)
    dt = float(config["env"]["dt"])
    oob_lim = float(config["env"]["oob_limit"])
    params = _hardnet_params(config)
    h_fn = make_h_fn(framework.value_net, system)
    policy_net, filt = framework.policy_net, framework._filter

    scenes = load_pool(pool_path).scenes
    batched = batch_scenes(scenes, device=device, dtype=dtype)
    x0 = initial_states_from_batch(batched)
    filter_fn = _filter_adapter(framework)
    with torch.no_grad():
        res = rollout_eval(system, framework.policy, filter_fn, batched, x0,
                           int(config["eval"]["max_steps"]), dt, config=config)
    T, B, _ = res.states.shape
    gen = torch.Generator(device="cpu").manual_seed(20)
    bi = torch.randint(0, B, (N_WINDOWS,), generator=gen)
    ti = torch.randint(0, T, (N_WINDOWS,), generator=gen)
    x = res.states[ti, bi].to(device)
    sb = batch_scenes([scenes[int(b)] for b in bi], device=device, dtype=dtype)

    def step_fn(xx):
        u_nom = policy_net(system.observation(xx, sb))
        u_safe, _ = filt(xx, sb, u_nom)
        return rk4_step(system, xx, u_safe, dt)

    logJ = []                       # per (step, window)
    feats = {k: [] for k in ("filter_active", "lg_norm", "h", "unsafe_alpha", "speed", "oob")}
    per_window_logJ = torch.zeros(T_FULL, N_WINDOWS, device=device)
    x = system.wrap_state(x)
    for k in range(T_FULL):
        with torch.no_grad():
            u_nom = policy_net(system.observation(x, sb))
            u_safe, _ = filt(x, sb, u_nom)
            h, lf_h, lg_h = _cbf_terms(system, h_fn, x, sb, u_nom, create_graph=False)
            lg_norm = torch.linalg.norm(lg_h, dim=1)
            fa = (torch.linalg.norm(u_safe - u_nom, dim=1) > 1e-3)
            unsafe = (h > 0.0)
            speed = system.speed(x)
            oob = torch.any(torch.abs(system.position(x)) > oob_lim, dim=1)
        sk = sigma_max(step_fn, x)                                   # [N]
        per_window_logJ[k] = torch.log(sk.clamp_min(1e-12))
        logJ.append(sk.detach().cpu())
        feats["filter_active"].append(fa.cpu()); feats["lg_norm"].append(lg_norm.cpu())
        feats["h"].append(h.cpu()); feats["unsafe_alpha"].append(unsafe.cpu())
        feats["speed"].append(speed.cpu()); feats["oob"].append(oob.cpu())
        with torch.no_grad():
            x = step_fn(x)

    J = torch.stack(logJ, dim=0)                                    # [T, N] log-sigma
    F = {k: torch.stack(v, dim=0) for k, v in feats.items()}        # [T, N]
    lyap = per_window_logJ.mean(dim=0).cpu()                        # [N]
    # conditional on filter_active x lg-bucket
    def bucket(lg):
        return torch.where(lg < 1e-3, 0, torch.where(lg < 1e-2, 1, 2))
    lgb = bucket(F["lg_norm"])
    cond = {}
    for fa_v in (0, 1):
        for lb in (0, 1, 2):
            m = (F["filter_active"].long() == fa_v) & (lgb == lb)
            if m.any():
                cond[f"fa{fa_v}_lg{lb}"] = {"n": int(m.sum()), "mean_logJ": float(J[m].mean()),
                                            "p90_logJ": float(J[m].flatten().quantile(0.9))}
    # top-decile step feature profile
    thr = J.flatten().quantile(0.9)
    top = J >= thr
    def tprof(key):
        return float(F[key][top].float().mean())
    return {
        "checkpoint": str(ckpt), "checkpoint_step": int(checkpoint.get("step", -1)),
        "alpha_safe": params.alpha_safe, "alpha_unsafe": params.alpha_unsafe,
        "epsilon": params.epsilon, "lg_reg_eps": params.lg_reg_eps, "box_aware": params.box_aware,
        "singular_threshold": _SINGULAR_LG_THRESHOLD,
        "n_windows": N_WINDOWS, "T": T_FULL,
        "logJ_overall": {"mean": float(J.mean()), "p50": float(J.flatten().quantile(0.5)),
                         "p90": float(J.flatten().quantile(0.9)), "p99": float(J.flatten().quantile(0.99)),
                         "max": float(J.max())},
        "sigma_gt1_frac": float((J > 0.0).float().mean()),
        "lyapunov_mean": float(lyap.mean()), "lyapunov_std": float(lyap.std()),
        "lyapunov_positive_window_frac": float((lyap > 0).float().mean()),
        "conditional_logJ": cond,
        "top_decile_profile": {"filter_active": tprof("filter_active"),
                               "unsafe_alpha": tprof("unsafe_alpha"),
                               "mean_lg_norm": float(F["lg_norm"][top].mean()),
                               "oob": tprof("oob"), "mean_speed": float(F["speed"][top].mean())},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--pool", type=Path, default=INLOOP_POOL)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--out", type=Path, default=REPO / "docs/versions/v2.4.0")
    args = ap.parse_args()
    stats = audit(args.ckpt, args.pool)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.tag}.jac.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"OUT={args.out / f'{args.tag}.jac.json'}")
    print("  logJ:", stats["logJ_overall"], " sigma>1 frac:", round(stats["sigma_gt1_frac"], 3))
    print("  lyapunov mean/std:", round(stats["lyapunov_mean"], 4), round(stats["lyapunov_std"], 4),
          " pos-window frac:", round(stats["lyapunov_positive_window_frac"], 3))
    print("  conditional_logJ:", stats["conditional_logJ"])
    print("  top-decile profile:", stats["top_decile_profile"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
