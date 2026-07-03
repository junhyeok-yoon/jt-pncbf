"""Stages B + C — per-sample policy-gradient T-scaling and mechanism ablations (read-only).

B: per-sample gradient norms of the (task-return) policy loss for T in {15,30,60}; report
   median/p90/p99/max, top-1% grad-norm share, log-linear fit of log(median|p90) vs T, and
   Spearman correlations of the T=60 per-sample grad norm with window features.
C: single-batch autograd surgeries on the SCRATCH loss at T=60 — detach-filter-Jacobian (C1),
   singularity masking tau in {1e-3,5e-3,2e-2} (C2), gamma_T in {0.95,0.90} (C3), oob-window
   exclusion (C4) — each vs the unmodified T=60 and T=30 to see which collapses the tail.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from src.common.filter_hardnet import _cbf_terms, _hardnet_params
from src.common.rk4 import rk4_step
from src.common.outcomes import _collided_exact
from src.common.value_net import make_h_fn
from src.eval.build_pools import load_pool
from src.eval.evaluate import _filter_adapter, _tensor_options
from src.eval.rollout import rollout_eval
from src.eval.run_full import _load_framework
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from scripts.analysis._bptt_scratch import task_return_persample, persample_grad_norms

REPO = Path(__file__).resolve().parents[2]
INLOOP_POOL = REPO / "data/secured_data/pools/eval_inloop_di_n500_seed12345.pkl"
N_START, CHUNK = 512, 128


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ar = a.argsort().argsort().float()
    br = b.argsort().argsort().float()
    ar = (ar - ar.mean()) / ar.std().clamp_min(1e-12)
    br = (br - br.mean()) / br.std().clamp_min(1e-12)
    return float((ar * br).mean())


def _stats(v: torch.Tensor) -> dict:
    v = v.flatten().sort().values
    n = v.numel()
    top1 = v[int(0.99 * n):].sum() / v.sum().clamp_min(1e-12)
    return {"median": float(v.median()), "p90": float(v[int(0.9 * n)]), "p99": float(v[int(0.99 * n)]),
            "max": float(v.max()), "top1pct_share": float(top1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--pool", type=Path, default=INLOOP_POOL)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--out", type=Path, default=REPO / "docs/versions/v2.4.0")
    args = ap.parse_args()

    framework, config, checkpoint = _load_framework(args.ckpt)
    system = framework.system
    dtype, device = _tensor_options(system, framework)
    policy, value_net = framework.policy_net, framework.value_net
    policy.requires_grad_(True)
    dt = float(config["env"]["dt"]); oob_lim = float(config["env"]["oob_limit"])
    h_fn = make_h_fn(value_net, system); params = _hardnet_params(config)

    scenes_all = load_pool(args.pool).scenes
    batched = batch_scenes(scenes_all, device=device, dtype=dtype)
    x0 = initial_states_from_batch(batched)
    with torch.no_grad():
        res = rollout_eval(system, framework.policy, _filter_adapter(framework), batched, x0,
                           int(config["eval"]["max_steps"]), dt, config=config)
    T, B, _ = res.states.shape
    gen = torch.Generator(device="cpu").manual_seed(20)
    bi = torch.randint(0, B, (N_START,), generator=gen)
    ti = torch.randint(0, B if False else T, (N_START,), generator=gen)
    starts = res.states[ti, bi].to(device)
    start_scenes = [scenes_all[int(b)] for b in bi]

    def grad_norms(Tb, gamma=None, detach=False, tau=None, mask=None):
        idx = range(N_START) if mask is None else [i for i in range(N_START) if mask[i]]
        idx = list(idx)
        out = []
        for i in range(0, len(idx), CHUNK):
            js = idx[i:i + CHUNK]
            xs = starts[js]
            sc = batch_scenes([start_scenes[j] for j in js], device=device, dtype=dtype)
            batch = SimpleNamespace(states=xs, scene=sc)
            policy.zero_grad(set_to_none=True)
            loss = task_return_persample(system=system, policy_net=policy, value_net=value_net,
                                         batch=batch, config=config, bptt_t=Tb, gamma_t=gamma,
                                         detach_filter_jac=detach, singular_mask_tau=tau)
            out.append(persample_grad_norms(loss, policy.parameters()).detach().cpu())
        return torch.cat(out) if out else torch.zeros(0)

    # ---- Stage B: T-scaling ----
    B_stats = {}
    gvecs = {}
    for Tb in (15, 30, 60):
        g = grad_norms(Tb)
        gvecs[Tb] = g
        B_stats[str(Tb)] = _stats(g)
    Ts = [15, 30, 60]
    slope_med = (math.log(B_stats["60"]["median"]) - math.log(B_stats["15"]["median"])) / (60 - 15)
    slope_p90 = (math.log(B_stats["60"]["p90"]) - math.log(B_stats["15"]["p90"])) / (60 - 15)

    # ---- B2: feature correlations at T=60 (no_grad window features) ----
    with torch.no_grad():
        sc_all = batch_scenes(start_scenes, device=device, dtype=dtype)
        x = system.wrap_state(starts)
        min_lg_active = torch.full((N_START,), float("inf"), device=device)
        n_active = torch.zeros(N_START, device=device)
        has_oob = torch.zeros(N_START, dtype=torch.bool, device=device)
        has_goal = torch.zeros(N_START, dtype=torch.bool, device=device)
        for _ in range(60):
            u_nom = framework.policy_net(system.observation(x, sc_all))
            u_safe, _ = framework._filter(x, sc_all, u_nom)
            _, _, lg = _cbf_terms(system, h_fn, x, sc_all, u_nom, create_graph=False)
            lgn = torch.linalg.norm(lg, dim=1)
            fa = torch.linalg.norm(u_safe - u_nom, dim=1) > 1e-3
            n_active += fa.float()
            min_lg_active = torch.where(fa, torch.minimum(min_lg_active, lgn), min_lg_active)
            x = rk4_step(system, x, u_safe, dt)
            p = system.position(x); sp = system.speed(x)
            gpos = torch.as_tensor(sc_all.goal, dtype=dtype, device=device)
            has_goal |= (torch.linalg.norm(p - gpos, dim=1) <= float(config["env"]["goal_radius"])) & (sp <= float(config["env"]["goal_speed_radius"]))
            has_oob |= torch.any(torch.abs(p) > oob_lim, dim=1)
        min_lg_active = torch.where(torch.isinf(min_lg_active), torch.zeros_like(min_lg_active), min_lg_active)
    g60 = gvecs[60].to(device)
    corr = {
        "spearman_gnorm_vs_min_lg_active": _spearman(g60, min_lg_active),
        "spearman_gnorm_vs_n_active": _spearman(g60, n_active),
        "spearman_gnorm_vs_has_oob": _spearman(g60, has_oob.float()),
        "spearman_gnorm_vs_has_goal": _spearman(g60, has_goal.float()),
    }
    thr = g60.flatten().quantile(0.99)
    top = g60 >= thr
    top_profile = {"n": int(top.sum()), "mean_n_active": float(n_active[top].mean()),
                   "mean_min_lg_active": float(min_lg_active[top].mean()),
                   "frac_has_oob": float(has_oob[top].float().mean()),
                   "frac_has_goal": float(has_goal[top].float().mean())}

    # ---- Stage C: ablations at T=60 ----
    C = {"unmodified_T60": B_stats["60"], "reference_T30": B_stats["30"]}
    C["C1_detach_filter_jac"] = _stats(grad_norms(60, detach=True))
    for tau in (1e-3, 5e-3, 2e-2):
        C[f"C2_singular_mask_tau{tau:g}"] = _stats(grad_norms(60, tau=tau))
    for gm in (0.95, 0.90):
        C[f"C3_gamma{gm:g}"] = _stats(grad_norms(60, gamma=gm))
    keep = [(not bool(has_oob[i])) for i in range(N_START)]
    C["C4_oob_excluded"] = _stats(grad_norms(60, mask=keep))
    C["C4_oob_excluded"]["n_kept"] = int(sum(keep))

    stats = {"checkpoint": str(args.ckpt), "checkpoint_step": int(checkpoint.get("step", -1)),
             "n_start": N_START, "alpha_unsafe": params.alpha_unsafe,
             "B_Tscaling": B_stats, "log_slope_median_per_step": slope_med, "log_slope_p90_per_step": slope_p90,
             "B2_correlations": corr, "B2_top1pct_profile": top_profile, "C_ablations": C}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.tag}.gradscale.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"OUT={args.out / f'{args.tag}.gradscale.json'}")
    for Tb in ("15", "30", "60"):
        print(f"  T={Tb}: {B_stats[Tb]}")
    print(f"  log-slope median/p90 per step: {slope_med:.4f} / {slope_p90:.4f}")
    print(f"  B2 corr: {corr}")
    print(f"  B2 top1% profile: {top_profile}")
    print("  --- C ablations (median/p90/max) ---")
    for k, v in C.items():
        print(f"  {k}: median={v['median']:.3f} p90={v['p90']:.3f} max={v['max']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
