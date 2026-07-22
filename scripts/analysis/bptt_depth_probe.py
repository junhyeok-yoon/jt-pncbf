"""v2.7.4 Phase 0 / Part B — BPTT depth-vs-horizon gradient probe (eval-only, forward+backward, NO update).

The bptt_T=60 collapse confounded DEPTH (number of BPTT steps T) with physical HORIZON (H = T*dt): going
30->60 at fixed dt=0.05 doubled BOTH. This probe separates them by sweeping (dt, T) so that some cells share
H while differing in T. It reconstructs the policy BPTT rollout faithfully from the SAME pieces training uses
(system, HardNet filter on the M3 value, rk4_step, the exact running + terminal cost), with the two B-2
confound neutralizations applied so the cells are comparable:
  (i)  running task cost multiplied by dt (approximates the same time-integral at both timesteps);
  (ii) per-step objective discount gamma_T rescaled as gamma_T ** (dt/dt0) so per-second discounting is
       fixed (dt0=0.05, gamma_T=0.99 -> 0.995988 at dt=0.02).
The value discount is dt-aware already (gamma_from_lambda) and the value is used frozen-in-graph here (not
retrained), so it is left alone. Policy = M5 best; filter value = M3 (B-1). detach_filter_coeffs follows the
config (True): the filter backward is then a projection onto a convex set (non-expansive), so the predicted
depth failure is a VANISHING gradient, which the exactly-zero-entry fraction is meant to reveal.

Per (dt,T) cell and per fixed IC draw: ||grad_policy||, ||grad_value||, per-layer policy-grad norm, fraction
of policy-grad entries exactly zero, count of non-finite grad entries, peak CUDA memory. 5 fixed IC draws
(fixed pool slices, reused across every cell so only (dt,T) varies); median/min/max reported. One JSON out.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.common.filter_hardnet import HardNetFilter
from src.common.maneuver_value import build_safety_h_fn
from src.common.observation import scene_obstacle_tensors
from src.common.rk4 import rk4_step
from src.eval.build_pools import DEFAULT_OUTPUT_DIR, load_pool
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.frameworks.jt_pncbf.losses import _scene_goal
from src.eval.run_full import _load_framework   # dispatches by checkpoint["framework"] (jt vs oc)

ap = argparse.ArgumentParser()
ap.add_argument("--policy-ckpt", required=True)          # M5 best.pt
ap.add_argument("--value-ckpt", required=True)           # M3 value best.pt (filter safety value, B-1)
ap.add_argument("--pool", default="eval_inloop_quadrotor-3d-d2_n500_seed12345.pkl")
ap.add_argument("--batch", type=int, default=64)         # scenes per IC draw
ap.add_argument("--draws", type=int, default=5)          # fixed IC draws
ap.add_argument("--out", required=True)
a = ap.parse_args()

DT0 = 0.05
GRID = [(0.05, 15), (0.05, 30), (0.05, 60), (0.02, 37), (0.02, 75), (0.02, 150)]

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fw, cfg, _ = _load_framework(Path(a.policy_ckpt))        # M5 (jt): policy + value
system = fw.system
policy_net = fw.policy_net.to(dev)
# filter value = M3 (B-1): dispatch by framework field (oc value-only checkpoint), keep its value_net
fw_v, _, _ = _load_framework(Path(a.value_ckpt))
value_net = fw_v.value_net.to(dev)

pol = cfg["loss"]["policy"]
lambda_v = float(pol["lambda_v"]); mu_u = float(pol["mu_u"])
w_settle = float(pol.get("w_settle", 0.0)); settle_rho = float(pol.get("settle_rho", 0.30))
w_appr = float(pol.get("w_appr", 0.0)); tau_brake = float(pol.get("tau_brake", 0.6))
w_term = float(pol.get("w_terminal", 0.0)); w_term_v = float(pol.get("w_terminal_v", 0.0))
gamma_T0 = float(pol["gamma_T"])
detach_coeffs = bool(pol.get("detach_filter_coeffs", False))
situational = hasattr(system, "horizontal_velocity") and (w_settle > 0.0 or w_appr > 0.0)

pool = load_pool(DEFAULT_OUTPUT_DIR / a.pool)
scenes_all = pool.scenes
# fixed IC draws = fixed disjoint slices of the frozen pool (deterministic, reused across all cells)
draws = [scenes_all[i * a.batch:(i + 1) * a.batch] for i in range(a.draws)]
dtype = next(value_net.parameters()).dtype

hardnet = HardNetFilter(system, build_safety_h_fn(system, cfg, value_net), cfg)


def _rollout_gradnorm(scene_list, dt, T):
    """One forward+backward of the (neutralized) policy BPTT task cost; returns grad stats."""
    for p in policy_net.parameters():
        if p.grad is not None:
            p.grad = None
    for p in value_net.parameters():
        if p.grad is not None:
            p.grad = None
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    bscene = batch_scenes(scene_list, device=dev, dtype=dtype)
    x = system.wrap_state(initial_states_from_batch(bscene).to(device=dev, dtype=dtype))
    gamma_T = gamma_T0 ** (dt / DT0)                       # B-2(ii) per-second-invariant discount
    if situational and w_appr > 0.0:
        obs_c, obs_r, obs_a = scene_obstacle_tensors(bscene, x.device, x.dtype)
    task_cost = x.new_zeros(x.shape[0]); discount = 1.0
    for _ in range(T):
        obs = system.observation(x, bscene)
        u_nom = policy_net(obs)
        u_safe, _ = hardnet(x, bscene, u_nom, detach_coeffs=detach_coeffs)
        x = rk4_step(system, x, u_safe, dt)
        goal = _scene_goal(bscene, x)
        pos_err = system.position(x) - goal
        v2 = system.speed(x) * system.speed(x)
        u2 = torch.sum(u_safe * u_safe, dim=1)
        d2 = torch.sum(pos_err * pos_err, dim=1)
        step_cost = d2 + lambda_v * v2 + mu_u * u2
        if situational and w_settle > 0.0:
            r = torch.linalg.norm(pos_err, dim=1)
            step_cost = step_cost + w_settle * torch.exp(-(r * r) / (settle_rho * settle_rho)) * v2
        if situational and w_appr > 0.0:
            p_xy = system.position(x)[:, : obs_c.shape[-1]]
            vel = system.horizontal_velocity(x)
            rel = p_xy.unsqueeze(1) - obs_c
            dist_c = torch.linalg.norm(rel, dim=2)
            surf = dist_c - obs_r
            normal = rel / dist_c.unsqueeze(2).clamp_min(1.0e-9)
            v_dot_n = torch.sum(vel.unsqueeze(1) * normal, dim=2)
            inward = torch.relu(-v_dot_n)
            deficit = torch.relu(inward * tau_brake - surf) * obs_a.to(surf.dtype)
            step_cost = step_cost + w_appr * torch.sum(deficit * deficit, dim=1)
        task_cost = task_cost + discount * dt * step_cost   # B-2(i) running cost * dt
        discount = discount * gamma_T
    if w_term > 0.0 or w_term_v > 0.0:
        goal_T = _scene_goal(bscene, x)
        dist_T = torch.linalg.norm(system.position(x) - goal_T, dim=1)
        speed_T = system.speed(x)
        task_cost = task_cost + discount * (w_term * dist_T + w_term_v * speed_T)
    loss = task_cost.mean()
    loss.backward()

    def _grad_stats(params):
        g = [p.grad.detach() for p in params if p.grad is not None]
        if not g:
            return dict(norm=0.0, zero_frac=1.0, n_nonfinite=0, n_elems=0)
        flat = torch.cat([t.reshape(-1) for t in g])
        return dict(norm=float(torch.linalg.norm(flat)),
                    zero_frac=float((flat == 0).float().mean()),
                    n_nonfinite=int((~torch.isfinite(flat)).sum()),
                    n_elems=int(flat.numel()))
    pstat = _grad_stats(list(policy_net.parameters()))
    vstat = _grad_stats(list(value_net.parameters()))
    # per-layer policy grad norm
    per_layer = {name: float(torch.linalg.norm(p.grad.detach()))
                 for name, p in policy_net.named_parameters() if p.grad is not None}
    peak_mb = (torch.cuda.max_memory_allocated() / 1e6) if dev.type == "cuda" else 0.0
    return dict(grad_policy=pstat["norm"], grad_value=vstat["norm"],
                policy_zero_frac=pstat["zero_frac"], policy_nonfinite=pstat["n_nonfinite"],
                value_nonfinite=vstat["n_nonfinite"], peak_mem_mb=round(peak_mb, 1),
                per_layer=per_layer, loss=float(loss.detach()))


def _agg(vals):
    import statistics
    return dict(median=round(statistics.median(vals), 6), min=round(min(vals), 6), max=round(max(vals), 6))


cells = []
for (dt, T) in GRID:
    runs = [_rollout_gradnorm(d, dt, T) for d in draws]
    gp = [r["grad_policy"] for r in runs]; gv = [r["grad_value"] for r in runs]
    cell = dict(
        dt=dt, T=T, H=round(dt * T, 4), gamma_T=round(gamma_T0 ** (dt / DT0), 6),
        grad_policy=_agg(gp), grad_value=_agg(gv),
        policy_zero_frac=_agg([r["policy_zero_frac"] for r in runs]),
        policy_nonfinite=max(r["policy_nonfinite"] for r in runs),
        value_nonfinite=max(r["value_nonfinite"] for r in runs),
        peak_mem_mb=_agg([r["peak_mem_mb"] for r in runs]),
        loss=_agg([r["loss"] for r in runs]),
        per_layer_median={k: round(float(sorted(r["per_layer"][k] for r in runs)[len(runs) // 2]), 6)
                          for k in runs[0]["per_layer"]},
    )
    cells.append(cell)
    print(f"[cell] dt={dt} T={T} H={cell['H']} gammaT={cell['gamma_T']} "
          f"||gp||={cell['grad_policy']['median']:.4g} ||gv||={cell['grad_value']['median']:.4g} "
          f"zerofrac={cell['policy_zero_frac']['median']:.3f} peakMB={cell['peak_mem_mb']['median']}", flush=True)

out = dict(
    meta=dict(policy_ckpt=a.policy_ckpt, value_ckpt=a.value_ckpt, pool=a.pool, batch=a.batch,
              draws=a.draws, device=dev.type, dt0=DT0, gamma_T0=gamma_T0,
              detach_filter_coeffs=detach_coeffs, situational=situational,
              neutralization="running_cost*dt ; gamma_T**(dt/dt0) ; value discount untouched"),
    grid=cells,
)
Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
print(f"[done] wrote {a.out}", flush=True)
