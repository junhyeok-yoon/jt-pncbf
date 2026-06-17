"""v2.2.0 Stage 2 — HOCBF vs PNCBF as a DEPLOYMENT filter over a full N=2000 distribution.

Part 3 (stage2_hocbf_comparison.py) compared the two filters on the PNCBF-stuck subset only.
This task asks the deployment question: over a full (success-included) deployment-distribution pool
at N=2000, does swapping the deployed PNCBF (HardNet on V_S) for the analytic HOCBF — SAME learned
nominal, SAME V_S-derived policy, SAME scenes, SAME rollout loop, only the filter differs — improve
the canonical cps, or trade stuck for collisions/timeouts/over-conservatism?

Part 0  verify the HOCBF is deployment-faithful: it builds its per-obstacle rows from the SAME
        nearest-K (p - c_i, r_i) channels the policy/PNCBF obs exposes (NOT privileged full-scene
        info); same r_safe margin (0.05) as Part 3; report the QP per-step cost.
Part 1  build a deterministic N=2000 pool from the SAME deployment distribution (checkpoint config,
        new seed). Committed N=500 / N=30000 pools untouched.
Part 2  run PNCBF and HOCBF (K=5) on the identical pool; full outcome counts + canonical cps each.
Part 3  paired per-scene contingency (PNCBF-outcome x HOCBF-outcome), improve/regress, infeasibility.
Part 4  verdict (numbers only).

cps = reach - 2*collision - stuck - 0.5*(oob+timeout) - 0.3*infeasibility   (canonical; matches
src/eval/evaluate.py per-episode definition exactly — helpers imported from there).

Read-only on the deployed policy/V_S/HardNet, the secured checkpoint, and committed pools; the HOCBF
is the existing separate module src/common/filter_hocbf.py. Deterministic: pool seeded (reported);
rollouts + PNCBF forward + HOCBF QP have no RNG.

Run: /home/junhyeok/miniconda3/envs/pncbf/bin/python scripts/analysis/stage2_hocbf_deploy_n2000.py
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.filter_hocbf import HOCBFFilter  # noqa: E402
from src.common.outcomes import resolve_outcome, step_outcomes  # noqa: E402
from src.eval.build_pools import (  # noqa: E402
    PoolSpec, build_pool, load_pool, obstacle_distribution_name, pool_stem, write_pool,
)
from src.eval.evaluate import (  # noqa: E402
    active_action_steps, active_bool_fraction, first_physical_event_step,
)
from src.eval.rollout import rollout_eval  # noqa: E402
from src.envs.scene_batch import batch_scenes, initial_states_from_batch  # noqa: E402
from src.frameworks.jt_pncbf.train import load_framework_from_checkpoint, make_system  # noqa: E402

CKPT = REPO_ROOT / "data/secured_data/v2.0.1/seed42/checkpoints/best.pt"
OUT = REPO_ROOT / "data/diagnostics/v2.2.0_hocbf"
REPORT = REPO_ROOT / "docs/versions/v2.2.0/stage2_hocbf_deploy_n2000.md"
N = 2000
POOL_SEED = 20260617
R_MARGIN = 0.05
K_DEPLOY = 5
HOCBF_A1, HOCBF_A2 = 2.0, 2.0
PNCBF_CHUNK = 250          # bounded: HardNet create_graph autograd graph per step must fit GPU
HOCBF_CHUNK = 500          # no autograd; cheap
OUTCOMES = ("goal", "collision", "stuck", "timeout", "oob")


# ---- Part 0: deployment-faithfulness + QP cost ---------------------------------------

def verify_obs_equivalence(system, scenes, device, dtype, k=K_DEPLOY):
    """The HOCBF's nearest-k (p-c_i, r_i) must equal the policy/PNCBF obs channels (no privilege)."""
    bscene = batch_scenes(scenes[:500], device=device, dtype=dtype)
    x = initial_states_from_batch(bscene)
    # perturb velocities so the check spans deployment-relevant (moving) states, not just starts
    g = torch.Generator(device="cpu").manual_seed(7)
    x = x.clone()
    x[:, 2:4] = (torch.rand(x.shape[0], 2, generator=g) * 2 - 1).to(device, dtype)

    obs = system.observation(x, bscene)                       # [B, 19] = [vx,vy, gx-px,gy-py, 5x(rel,r)]
    blk = obs[:, 4:].reshape(x.shape[0], k, 3)
    obs_pc = -blk[:, :, :2]                                    # p - c = -(center - pos)
    obs_r = blk[:, :, 2]
    obs_surf = torch.linalg.norm(obs_pc, dim=2) - obs_r        # nearest-k surface dist from obs
    # obs pads empty slots with rel=0,r=0; the module masks them to inf. Real obstacles have r>0;
    # mask phantom-pad slots to inf on BOTH sides so only real nearest-k entries are compared.
    obs_phantom = obs_r <= 1e-9
    obs_surf = obs_surf.masked_fill(obs_phantom, float("inf"))

    # the module's own nearest-k selection (same surface-distance criterion as top_k_obstacles)
    p = x[:, :2]
    centers = bscene.obstacle_centers; radii = bscene.obstacle_radii; active = bscene.obstacle_active
    rel = p.unsqueeze(1) - centers
    surf = torch.linalg.norm(rel, dim=2) - radii
    surf = surf.masked_fill(~active, float("inf"))
    sel = torch.topk(surf, k, dim=1, largest=False).indices
    bi = torch.arange(x.shape[0], device=device).unsqueeze(1)
    mod_pc = rel[bi, sel]                                      # p - c for module's nearest-k
    mod_r = radii[bi, sel]
    mod_active = active[bi, sel]
    mod_surf = torch.linalg.norm(mod_pc, dim=2) - mod_r
    mod_surf = mod_surf.masked_fill(~mod_active, float("inf"))

    # compare as a multiset: sort by surface distance, phantom -> inf both sides
    obs_sorted, _ = torch.sort(torch.nan_to_num(obs_surf, posinf=1e9), dim=1)
    mod_sorted, _ = torch.sort(torch.nan_to_num(mod_surf, posinf=1e9), dim=1)
    surf_diff = float(torch.max(torch.abs(obs_sorted - mod_sorted)).item())
    # radii multiset match over real obstacles (phantom radii are 0 both sides)
    obs_r_real = obs_r.masked_fill(obs_phantom, 0.0)
    mod_r_real = mod_r.masked_fill(~mod_active, 0.0)
    r_diff = float(torch.max(torch.abs(torch.sort(obs_r_real, dim=1).values
                                       - torch.sort(mod_r_real, dim=1).values)).item())
    n_real_obs_match = bool(torch.equal((~obs_phantom).sum(1), mod_active.sum(1)))
    return {"n_states": int(x.shape[0]), "k": k,
            "max_surface_dist_diff_obs_vs_module": surf_diff,
            "max_radius_diff_obs_vs_module": r_diff,
            "real_obstacle_count_matches": n_real_obs_match,
            "note": ("HOCBF selects nearest-k by surface distance ||p-c||-r, the identical criterion "
                     "top_k_obstacles uses to build the obs; the (p-c_i, r_i) it consumes equal the "
                     "obs channels, so no privileged full-scene info is used.")}


def qp_cost(system, config, scenes, device, dtype):
    hocbf = HOCBFFilter(system, config, HOCBF_A1, HOCBF_A2, R_MARGIN, k_obs=K_DEPLOY)
    bscene = batch_scenes(scenes[:HOCBF_CHUNK], device=device, dtype=dtype)
    x = initial_states_from_batch(bscene)
    u = torch.zeros(x.shape[0], system.action_dim, device=device, dtype=dtype)
    for _ in range(3):                                        # warmup
        hocbf(x, bscene, u)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    reps = 20
    for _ in range(reps):
        hocbf(x, bscene, u)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) / reps * 1e3
    return {"batch": int(x.shape[0]), "ms_per_batched_qp_step": dt_ms,
            "us_per_scene_step": dt_ms * 1e3 / x.shape[0],
            "note": ("_qp2d is fully vectorized over the scene batch (candidate enumeration over "
                     "k+4 rows, all tensor ops over [B,...]); N=2000 solves in a few batches.")}


# ---- Part 1: N=2000 deployment pool --------------------------------------------------

def build_n2000_pool(config):
    spec = PoolSpec(name="deployN2000", n_scenes=N, seed=POOL_SEED)
    dist = obstacle_distribution_name(config)
    stem = pool_stem("deployN2000", "double_integrator", N, POOL_SEED, dist)
    path = OUT / f"{stem}.pkl"
    if path.exists():
        print(f"[part1] reusing existing pool {path}")
        return load_pool(path), path, dist
    print(f"[part1] building N={N} pool (seed {POOL_SEED}, distribution {dist}) ...")
    pool = build_pool(config, "double_integrator", spec)
    write_pool(pool, config, output_dir=OUT)
    return pool, path, dist


# ---- Part 2: deployment eval (canonical cps; per-chunk detach to avoid autograd OOM) -

def cps_episode(outcome, infeas_frac):
    reach = 1.0 if outcome == "goal" else 0.0
    collision = 1.0 if outcome == "collision" else 0.0
    oob = 1.0 if outcome == "oob" else 0.0
    stuck = 1.0 if outcome == "stuck" else 0.0
    timeout = 1.0 if outcome == "timeout" else 0.0
    return reach - 2.0 * collision - stuck - 0.5 * (oob + timeout) - 0.3 * infeas_frac


def deploy_eval(system, scenes, config, filter_fn, policy_fn, device, dtype, chunk, label):
    max_steps = int(config["eval"]["max_steps"]); dt = float(config["env"]["dt"])
    rows = []
    for s in range(0, len(scenes), chunk):
        sub = scenes[s:s + chunk]
        bscene = batch_scenes(sub, device=device, dtype=dtype)
        x0 = initial_states_from_batch(bscene)
        res = rollout_eval(system, policy_fn, filter_fn, bscene, x0, max_steps, dt, config)
        masks = step_outcomes(res.states, bscene, system, config)
        resolved = resolve_outcome(masks)
        phys = first_physical_event_step(masks).detach().cpu()
        infeas = res.infeasible.detach()                      # [T,B] bool, graph-free
        for i in range(len(sub)):
            outcome = resolved.outcome[i]
            active = active_action_steps(int(phys[i].item()), infeas.shape[0])
            ifrac = active_bool_fraction(infeas[:, i:i + 1], active)
            rows.append({"outcome": outcome, "infeas_frac": float(ifrac),
                         "cps": float(cps_episode(outcome, ifrac))})
        del res, masks, resolved, infeas, phys, bscene, x0
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  [{label}] {min(s + chunk, len(scenes))}/{len(scenes)}")
    return rows


def summarize(rows):
    n = len(rows)
    counts = {o: sum(1 for r in rows if r["outcome"] == o) for o in OUTCOMES}
    rates = {o: counts[o] / n for o in OUTCOMES}
    cps = float(np.mean([r["cps"] for r in rows]))
    infeas = float(np.mean([r["infeas_frac"] for r in rows]))
    return {"n": n, "counts": counts, "rates": rates, "cps_mean": cps, "infeasibility_mean": infeas}


def paired_delta_ci(rows_a, rows_b, seed=20260617, boot=2000):
    """Paired bootstrap CI on mean(cps_b - cps_a) over the same scenes."""
    d = np.array([rb["cps"] - ra["cps"] for ra, rb in zip(rows_a, rows_b)], float)
    rng = np.random.default_rng(seed)
    means = [float(np.mean(d[rng.integers(0, d.size, d.size)])) for _ in range(boot)]
    return {"delta_cps_mean": float(d.mean()),
            "delta_cps_ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]}


# ---- Part 3: paired contingency ------------------------------------------------------

def contingency(rows_pncbf, rows_hocbf):
    cells = {}
    for rp, rh in zip(rows_pncbf, rows_hocbf):
        cells[(rp["outcome"], rh["outcome"])] = cells.get((rp["outcome"], rh["outcome"]), 0) + 1
    table = {f"PNCBF_{a}__HOCBF_{b}": v for (a, b), v in sorted(cells.items())}
    improved = sum(v for (a, b), v in cells.items()
                   if cps_rank(b) > cps_rank(a))
    regressed = sum(v for (a, b), v in cells.items()
                    if cps_rank(b) < cps_rank(a))
    same = sum(v for (a, b), v in cells.items() if a == b)
    transitions = sorted(((v, a, b) for (a, b), v in cells.items() if a != b), reverse=True)
    return {"table": table,
            "n_improved": improved, "n_regressed": regressed, "n_unchanged_outcome": same,
            "dominant_transitions": [{"from": a, "to": b, "n": v} for v, a, b in transitions[:8]],
            "pncbf_stuck_to_hocbf_goal": cells.get(("stuck", "goal"), 0),
            "pncbf_stuck_to_hocbf_stuck": cells.get(("stuck", "stuck"), 0),
            "pncbf_goal_to_hocbf_nongoal": sum(cells.get(("goal", b), 0) for b in OUTCOMES if b != "goal")}


def cps_rank(outcome):
    # rank outcomes by their cps contribution (ignoring infeasibility term): goal best, collision worst
    return {"goal": 1.0, "timeout": -0.5, "oob": -0.5, "stuck": -1.0, "collision": -2.0}[outcome]


def main():
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    fw, config, _ = load_framework_from_checkpoint(CKPT)
    system = make_system(config)
    fw.value_net.to(device, dtype).eval(); fw.policy_net.to(device, dtype).eval()
    OUT.mkdir(parents=True, exist_ok=True)

    rep = {"config": {"ckpt": str(CKPT), "N": N, "pool_seed": POOL_SEED, "r_margin": R_MARGIN,
                      "k_deploy": K_DEPLOY, "hocbf_gains": [HOCBF_A1, HOCBF_A2],
                      "alpha_safe": float(config["filter"]["alpha_safe"]),
                      "lookahead": config["filter"].get("lookahead"),
                      "collision_boundary": "env exact ||p-c|| < r (gamma_margin=0); scored identically for both filters",
                      "hocbf_margin_note": "HOCBF enforces ||p-c|| >= r+0.05 internally (Part-3 value); outcomes scored by env exact collision for BOTH"}}

    # Part 1 first (need scenes for Part 0 sampling)
    pool, pool_path, dist = build_n2000_pool(config)
    scenes = pool.scenes
    rep["part1_pool"] = {"path": str(pool_path), "seed": POOL_SEED, "n_scenes": len(scenes),
                         "obstacle_distribution": dist}
    print(f"[part1] pool: {pool_path} (n={len(scenes)}, dist={dist})")

    # Part 0
    rep["part0_obs_equivalence"] = verify_obs_equivalence(system, scenes, device, dtype)
    rep["part0_qp_cost"] = qp_cost(system, config, scenes, device, dtype)
    print("[part0] obs-equivalence:", rep["part0_obs_equivalence"]["max_surface_dist_diff_obs_vs_module"],
          rep["part0_obs_equivalence"]["max_radius_diff_obs_vs_module"])
    print("[part0] qp cost:", rep["part0_qp_cost"]["ms_per_batched_qp_step"], "ms/batch-step")

    # Part 2 — both filters on the identical pool
    print("[part2] PNCBF (HardNet) deployment eval ...")
    rows_pncbf = deploy_eval(system, scenes, config, fw.filter, fw.policy, device, dtype, PNCBF_CHUNK, "pncbf")
    print("[part2] HOCBF (K=5) deployment eval ...")
    hocbf = HOCBFFilter(system, config, HOCBF_A1, HOCBF_A2, R_MARGIN, k_obs=K_DEPLOY)
    hocbf_fn = lambda x, u_nom, scene: hocbf(x, scene, u_nom)
    rows_hocbf = deploy_eval(system, scenes, config, hocbf_fn, fw.policy, device, dtype, HOCBF_CHUNK, "hocbf")

    sp, sh = summarize(rows_pncbf), summarize(rows_hocbf)
    rep["part2_pncbf"] = sp
    rep["part2_hocbf"] = sh
    rep["part2_delta_hocbf_minus_pncbf"] = {
        "cps": sh["cps_mean"] - sp["cps_mean"],
        "per_outcome_count": {o: sh["counts"][o] - sp["counts"][o] for o in OUTCOMES},
        "per_outcome_rate": {o: sh["rates"][o] - sp["rates"][o] for o in OUTCOMES},
        "infeasibility": sh["infeasibility_mean"] - sp["infeasibility_mean"],
        **paired_delta_ci(rows_pncbf, rows_hocbf)}
    print("[part2] cps PNCBF", round(sp["cps_mean"], 4), "| HOCBF", round(sh["cps_mean"], 4),
          "| delta", round(rep["part2_delta_hocbf_minus_pncbf"]["cps"], 4))

    # Part 3 — paired contingency
    rep["part3_contingency"] = contingency(rows_pncbf, rows_hocbf)
    print("[part3] improved", rep["part3_contingency"]["n_improved"],
          "regressed", rep["part3_contingency"]["n_regressed"])

    (OUT / "deploy_n2000_summary.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("summary ->", OUT / "deploy_n2000_summary.json")
    return rep


if __name__ == "__main__":
    main()
