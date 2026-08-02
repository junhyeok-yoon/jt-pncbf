"""v2.8.0 B — deploy control-rate x filter-projection (eval-only), one cell per invocation.

Re-runs the v2.7.5 arm structure on the dual arm's deliverable checkpoint (step 42000), crossed with
filter.projection. Seconds are held: max_steps, stuck_window_steps and the k-step fallback's k are rescaled
per arm (max_steps = 10s/dt_sim, stuck = 3s/dt_sim, k = 0.15s/dt_sim since kstep_select simulates env.dt);
every rescaled value is read back from the WRITTEN effective config and reported. Per cell:
  - outcome vector + per-component CIs + collision decomposition  (via evaluate(); arm A enumerate is the
    harness-reproduction gate against the recorded 20 Hz M5 row);
  - TV/s of the commanded (post-ZOH) action, median & p90 over episodes;
  - inter-sample barrier violation = max rise of V_hat within a control hold interval, median & p90;
  - TV/s decomposed by the branch (empty vs feasible) active at each step.
Usage: --arm {A,B,C,D} --proj {enumerate,dual_solve}. Sidecar: s3_eval/rate_<arm>_<proj>.json"""
from __future__ import annotations
import argparse, copy, json
from pathlib import Path
import numpy as np, torch

from src.eval.evaluate import evaluate, first_physical_event_step
from src.eval.run_full import _load_framework as load_fw
from src.eval.build_pools import load_pool
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.common.rk4 import rk4_step
from src.common.outcomes import step_outcomes, resolve_outcome

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
CK = REPO / "data/runs/v2.8.0/set__20260731-015517__seed42/v2.8.0__jt__20260731-015517__seed42/checkpoints/best.pt"
POOL = REPO / "data/secured_data/pools/eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl"
OUT = REPO / "data/runs/v2.8.0/s3_eval"; OUT.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BOOT_SEED = 20260508
ARMS = {  # dt_sim, dt_ctrl ; max_steps=10s/dt_sim, stuck=3s/dt_sim, k=0.15s/dt_sim (kstep uses env.dt=dt_sim)
    "A": (0.05, 0.05), "B": (0.01, 0.05), "C": (0.01, 0.01), "D": (0.002, 0.002),
}

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True, choices=list(ARMS))
ap.add_argument("--proj", required=True, choices=["enumerate", "dual_solve"])
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

# ---- verify the rescaled parameters from the WRITTEN effective config ----
verify = {"dt_sim": float(cfg["env"]["dt"]), "dt_ctrl": float(cfg["eval"]["dt_ctrl"]),
          "max_steps": int(cfg["eval"]["max_steps"]), "stuck_window_steps": int(cfg["env"]["stuck_window_steps"]),
          "fallback_k": int(fw._filter.params.empty_fallback_k), "filter_dt": float(fw._filter.params.dt),
          "substeps": substeps}
expect = {"dt_sim": dt_sim, "dt_ctrl": dt_ctrl, "max_steps": max_steps, "stuck_window_steps": stuck_w,
          "fallback_k": kfb, "filter_dt": dt_sim, "substeps": substeps}
verify["all_match_expected"] = all(abs(verify[k] - expect[k]) < 1e-9 for k in expect)
print(f"[{a.arm}/{a.proj}] VERIFY {verify}", flush=True)
if not verify["all_match_expected"]:
    raise SystemExit(f"HALT: rescaled config not written as expected: {verify} vs {expect}")

# ---- (1) outcome vector + CIs + decomposition via evaluate() ----
res = evaluate(fw, POOL, cfg, mode="final", step=int(ck2["step"]), ckpt_name=f"{a.arm}_{a.proj}",
               max_scenes=None, include_lqr_baseline=False)
r = res.eval_row; eps = res.episode_rows
cause = np.array([e.get("collision_cause", "") for e in eps])
rng = np.random.default_rng(BOOT_SEED)
def boot(v, n=1000):
    N = len(v); idx = rng.integers(0, N, size=(n, N)); vals = v[idx].mean(axis=1)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
COMP = ["cps", "reach", "collision", "oob", "stuck", "timeout", "infeasibility", "saturation_rate"]
CI = ["cps_ci_lo", "cps_ci_hi", "reach_ci_lo", "reach_ci_hi", "collision_ci_lo", "collision_ci_hi",
      "stuck_ci_lo", "stuck_ci_hi", "infeasibility_ci_lo", "infeasibility_ci_hi"]
outcome = {k: (float(r[k]) if r.get(k) is not None else None) for k in COMP + CI}
decomp = {}
for c in ("obstacle", "band_lower", "band_upper"):
    m = (cause == c).astype(float); decomp[c] = float(m.mean()); decomp[c + "_ci"] = boot(m)

# ---- (2) instrumented ZOH rollout: TV/s, inter-sample V_hat, branch (O(T), no per-step done-freeze;
#          first-event outcome resolution is freeze-invariant, so pre-done TV/V_hat are exact) ----
system = fw.system; h_fn = fw._filter.h_fn
scenes = load_pool(POOL).scenes
bscene = batch_scenes(scenes, device=DEV, dtype=torch.float32)
x = initial_states_from_batch(bscene); B = x.shape[0]
states = [x]; u_list = []; empty_list = []; V_list = [h_fn(x, bscene).reshape(-1).detach()]
held_u = None; held_empty = None
with torch.no_grad():
    for i in range(max_steps):
        if i % substeps == 0:
            u_nom = fw.policy(x, bscene)
            u_safe, _ = fw.filter(x, u_nom, bscene)
            le = getattr(fw._filter, "last_empty", None)
            held_empty = (le.to(DEV).bool() if le is not None else torch.zeros(B, dtype=torch.bool, device=DEV))
            held_u = u_safe.detach()
        u_list.append(held_u); empty_list.append(held_empty)
        x = system.wrap_state(rk4_step(system, x, held_u, dt_sim))
        states.append(x); V_list.append(h_fn(x, bscene).reshape(-1).detach())
S = torch.stack(states, 0)                                   # [T+1,B,D]
U = torch.stack(u_list, 0)                                   # [T,B,A]
EM = torch.stack(empty_list, 0)                              # [T,B]
V = torch.stack(V_list, 0)                                   # [T+1,B]
masks = step_outcomes(S, bscene, system, cfg)
done_np = first_physical_event_step(masks).cpu().numpy()      # -1 if never
active = np.where(done_np < 0, max_steps, done_np).astype(int)  # steps taken per episode

Unp = U.cpu().numpy(); EMnp = EM.cpu().numpy(); Vnp = V.cpu().numpy()
tv_per_s, tv_empty, tv_feas = [], 0.0, 0.0
for b in range(B):
    n = max(active[b], 1)
    du = np.linalg.norm(Unp[1:n, b] - Unp[:n - 1, b], axis=1)  # increments over active control steps
    tv = float(du.sum()); tv_per_s.append(tv / (n * dt_sim))
    # branch of the step whose command produced the increment (step t+1)
    br = EMnp[1:n, b]
    tv_empty += float(du[br].sum()); tv_feas += float(du[~br].sum())
tv_per_s = np.array(tv_per_s)
tv_tot = tv_empty + tv_feas
# inter-sample: max rise of V within each control hold interval, pooled over active intervals
rises = []
for b in range(B):
    n = active[b]
    for start in range(0, n, substeps):
        end = min(start + substeps, n)
        if end <= start:
            continue
        rise = float((Vnp[start + 1:end + 1, b] - Vnp[start, b]).max()) if end > start else 0.0
        rises.append(rise)
rises = np.array(rises) if rises else np.array([0.0])

rep = {"arm": a.arm, "proj": a.proj, "verify_config": verify,
       "outcome": outcome, "collision_decomposition": decomp,
       "tv_per_s": {"median": float(np.median(tv_per_s)), "p90": float(np.percentile(tv_per_s, 90)),
                    "mean": float(tv_per_s.mean())},
       "tv_branch_split": {"empty_branch_fraction_of_total_tv": (tv_empty / tv_tot) if tv_tot > 0 else None,
                           "feasible_branch_fraction_of_total_tv": (tv_feas / tv_tot) if tv_tot > 0 else None,
                           "total_tv": tv_tot},
       "intersample_Vhat_rise": {"median": float(np.median(rises)), "p90": float(np.percentile(rises, 90)),
                                 "max": float(rises.max()), "n_intervals": int(rises.size)},
       "empty_step_fraction": float(EMnp[:int(active.max())].mean())}
(OUT / f"rate_{a.arm}_{a.proj}.json").write_text(json.dumps(rep, indent=2) + "\n")
print(f"[{a.arm}/{a.proj}] cps {outcome['cps']:.4f} reach {outcome['reach']:.4f} coll {outcome['collision']:.4f} "
      f"| TV/s med {rep['tv_per_s']['median']:.3f} p90 {rep['tv_per_s']['p90']:.3f} "
      f"| interV med {rep['intersample_Vhat_rise']['median']:.2e} p90 {rep['intersample_Vhat_rise']['p90']:.2e} "
      f"| TV empty-branch frac {rep['tv_branch_split']['empty_branch_fraction_of_total_tv']}", flush=True)
print(f"-> {OUT / f'rate_{a.arm}_{a.proj}.json'}")
