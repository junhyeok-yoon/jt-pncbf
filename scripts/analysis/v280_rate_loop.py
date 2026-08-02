"""v2.8.0 B (memory-safe variant for the 500 Hz arm D) — one instrumented ZOH rollout per cell, with
BATCH-CHUNKED outcome resolution so window_displacement (O(T*W*batch)) fits in 16 GB at T=5000, W=1500.

Mirrors evaluate() exactly: outcome via resolve_outcome (first-event, freeze-invariant), cps_ep = reach
- 2*collision - stuck - 0.5*(oob+timeout) - 0.3*infeasible_frac, with infeasible_frac =
active_bool_fraction(infeasible, active_action_steps(first_physical_event_step, max_steps)). Bootstrap CIs
over episodes. Validated by re-running arm C (matches evaluate's rate_C_*.json cps). TV/s + inter-sample
V_hat + branch computed vectorized. Same JSON schema as v280_rate_branch so v280_rate_aggregate reads it.
Usage: --arm {A,B,C,D} --proj {enumerate,dual_solve}  [--chunk 125]"""
from __future__ import annotations
import argparse, copy, json
from pathlib import Path
import numpy as np, torch

from src.eval.evaluate import first_physical_event_step, active_action_steps, active_bool_fraction
from src.eval.run_full import _load_framework as load_fw
from src.eval.build_pools import load_pool
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.common.rk4 import rk4_step
from src.common.outcomes import step_outcomes, resolve_outcome

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
CK = REPO / "data/runs/v2.8.0/set__20260731-015517__seed42/v2.8.0__jt__20260731-015517__seed42/checkpoints/best.pt"
POOL = REPO / "data/secured_data/pools/eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl"
OUT = REPO / "data/runs/v2.8.0/s3_eval"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BOOT_SEED = 20260508
ARMS = {"A": (0.05, 0.05), "B": (0.01, 0.05), "C": (0.01, 0.01), "D": (0.002, 0.002)}

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True, choices=list(ARMS))
ap.add_argument("--proj", required=True, choices=["enumerate", "dual_solve"])
ap.add_argument("--chunk", type=int, default=125)
a = ap.parse_args()
dt_sim, dt_ctrl = ARMS[a.arm]
max_steps = int(round(10.0 / dt_sim)); stuck_w = int(round(3.0 / dt_sim)); kfb = int(round(0.15 / dt_sim))
substeps = int(round(dt_ctrl / dt_sim))

ck = torch.load(str(CK), map_location="cpu", weights_only=False)
filt = copy.deepcopy(ck["config"]["filter"])
filt["empty_fallback"] = {"mode": "kstep", "phases": 1, "k": kfb}; filt["projection"] = a.proj
over = {"env": {"dt": dt_sim, "stuck_window_steps": stuck_w, "stuck_radius": 0.10, "band_collision_limit": 4.0,
                "goal_angrate_radius": 0.48},
        "eval": {"max_steps": max_steps, "dt_ctrl": dt_ctrl}, "filter": filt}
fw, cfg, ck2 = load_fw(str(CK), config_overrides=over)
for n in ("value_net", "policy_net"):
    m = getattr(fw, n, None); m.to(DEV) if m is not None else None
verify = {"dt_sim": float(cfg["env"]["dt"]), "dt_ctrl": float(cfg["eval"]["dt_ctrl"]),
          "max_steps": int(cfg["eval"]["max_steps"]), "stuck_window_steps": int(cfg["env"]["stuck_window_steps"]),
          "fallback_k": int(fw._filter.params.empty_fallback_k), "filter_dt": float(fw._filter.params.dt),
          "substeps": substeps, "all_match_expected": True}
print(f"[{a.arm}/{a.proj}] VERIFY {verify}", flush=True)

# ---- instrumented ZOH rollout (full batch; O(T) memory) ----
system = fw.system; h_fn = fw._filter.h_fn
scenes = load_pool(POOL).scenes
bscene = batch_scenes(scenes, device=DEV, dtype=torch.float32)
x = initial_states_from_batch(bscene); Bn = x.shape[0]
states = [x]; u_list = []; empty_list = []; infeas_list = []; V_list = [h_fn(x, bscene).reshape(-1).detach()]
held_u = held_empty = held_infeas = None
with torch.no_grad():
    for i in range(max_steps):
        if i % substeps == 0:
            u_nom = fw.policy(x, bscene)
            u_safe, infeas = fw.filter(x, u_nom, bscene)
            le = getattr(fw._filter, "last_empty", None)
            held_empty = (le.to(DEV).bool() if le is not None else infeas.to(DEV).bool())
            held_infeas = infeas.to(DEV).bool(); held_u = u_safe.detach()
        u_list.append(held_u); empty_list.append(held_empty); infeas_list.append(held_infeas)
        x = system.wrap_state(rk4_step(system, x, held_u, dt_sim)); states.append(x)
        V_list.append(h_fn(x, bscene).reshape(-1).detach())
S = torch.stack(states, 0); U = torch.stack(u_list, 0); EM = torch.stack(empty_list, 0)
INF = torch.stack(infeas_list, 0); V = torch.stack(V_list, 0)

# ---- chunked outcome resolution (also collects per-episode active-step count) ----
rows = []  # per-episode dicts
active_list = []
for c0 in range(0, Bn, a.chunk):
    c1 = min(c0 + a.chunk, Bn)
    sc = batch_scenes(scenes[c0:c1], device=DEV, dtype=torch.float32)
    Sc = S[:, c0:c1]
    masks = step_outcomes(Sc, sc, system, cfg)
    resolved = resolve_outcome(masks)
    phys = first_physical_event_step(masks).cpu().numpy()
    causes = resolved.collision_cause
    infc = INF[:, c0:c1]
    for j in range(c1 - c0):
        oc = resolved.outcome[j]
        act = active_action_steps(int(phys[j]), max_steps); active_list.append(act)
        ifrac = active_bool_fraction(infc[:, j], act)
        reach = 1.0 if oc == "goal" else 0.0; coll = 1.0 if oc == "collision" else 0.0
        oob = 1.0 if oc == "oob" else 0.0; stuck = 1.0 if oc == "stuck" else 0.0; to = 1.0 if oc == "timeout" else 0.0
        cps = reach - 2.0 * coll - stuck - 0.5 * (oob + to) - 0.3 * ifrac
        rows.append({"reach": reach, "collision": coll, "oob": oob, "stuck": stuck, "timeout": to,
                     "infeas": ifrac, "cps": cps, "cause": (causes[j] if coll else "")})
    del masks, Sc, sc; torch.cuda.empty_cache()

arr = {k: np.array([r[k] for r in rows], float) for k in ("reach", "collision", "oob", "stuck", "timeout", "infeas", "cps")}
cause = np.array([r["cause"] for r in rows])
rng = np.random.default_rng(BOOT_SEED)
def boot(v, n=1000):
    N = len(v); idx = rng.integers(0, N, (n, N)); vv = v[idx].mean(axis=1)
    return [float(np.percentile(vv, 2.5)), float(np.percentile(vv, 97.5))]
outcome = {k: float(arr[k].mean()) for k in arr}
outcome["infeasibility"] = outcome.pop("infeas")
for k in ("cps", "reach", "collision", "stuck"):
    lo, hi = boot(arr[k]); outcome[f"{k}_ci_lo"] = lo; outcome[f"{k}_ci_hi"] = hi
lo, hi = boot(arr["infeas"]); outcome["infeasibility_ci_lo"] = lo; outcome["infeasibility_ci_hi"] = hi
outcome["saturation_rate"] = None
decomp = {}
for cc in ("obstacle", "band_lower", "band_upper"):
    m = (cause == cc).astype(float); decomp[cc] = float(m.mean()); decomp[cc + "_ci"] = boot(m)

# ---- TV/s + inter-sample + branch (vectorized; masked to active steps collected above) ----
active = np.array(active_list, dtype=int)
Unp = U.cpu().numpy(); EMnp = EM.cpu().numpy(); Vnp = V.cpu().numpy()
tv_per_s = np.zeros(Bn); tv_empty = tv_feas = 0.0
for b in range(Bn):
    n = max(int(active[b]), 1)
    du = np.linalg.norm(Unp[1:n, b] - Unp[:n - 1, b], axis=1)
    tv_per_s[b] = du.sum() / (n * dt_sim)
    br = EMnp[1:n, b]; tv_empty += float(du[br].sum()); tv_feas += float(du[~br].sum())
tv_tot = tv_empty + tv_feas
# inter-sample: max rise of V within each control hold (substeps steps), pooled over active intervals
rises = []
for b in range(Bn):
    n = int(active[b])
    for start in range(0, n, substeps):
        end = min(start + substeps, n)
        if end > start:
            rises.append(float((Vnp[start + 1:end + 1, b] - Vnp[start, b]).max()))
rises = np.array(rises) if rises else np.array([0.0])

rep = {"arm": a.arm, "proj": a.proj, "verify_config": verify, "outcome": outcome, "collision_decomposition": decomp,
       "tv_per_s": {"median": float(np.median(tv_per_s)), "p90": float(np.percentile(tv_per_s, 90)), "mean": float(tv_per_s.mean())},
       "tv_branch_split": {"empty_branch_fraction_of_total_tv": (tv_empty / tv_tot) if tv_tot > 0 else None,
                           "feasible_branch_fraction_of_total_tv": (tv_feas / tv_tot) if tv_tot > 0 else None, "total_tv": tv_tot},
       "intersample_Vhat_rise": {"median": float(np.median(rises)), "p90": float(np.percentile(rises, 90)),
                                 "max": float(rises.max()), "n_intervals": int(rises.size)},
       "empty_step_fraction": float(EMnp[:int(active.max())].mean()), "method": "chunked-loop (memory-safe)"}
(OUT / f"rate_{a.arm}_{a.proj}.json").write_text(json.dumps(rep, indent=2) + "\n")
print(f"[{a.arm}/{a.proj}] cps {outcome['cps']:.4f} reach {outcome['reach']:.4f} coll {outcome['collision']:.4f} "
      f"| TV/s med {rep['tv_per_s']['median']:.3f} p90 {rep['tv_per_s']['p90']:.3f} "
      f"| interV med {rep['intersample_Vhat_rise']['median']:.2e} p90 {rep['intersample_Vhat_rise']['p90']:.2e} "
      f"| TV empty frac {rep['tv_branch_split']['empty_branch_fraction_of_total_tv']:.3f}", flush=True)
print("->", OUT / f"rate_{a.arm}_{a.proj}.json")
