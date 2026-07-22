"""v2.7.4 M7 — kstep empty-branch fallback ADOPTION trial on SO(3) ICs (eval-only, best.pt@39000).

Two arms on the SAME M6 best checkpoint and the SAME frozen d2r full pool (batch-2000):
  none    — checkpoint filter as-is (config already carries empty_fallback.mode=none); MUST reproduce the
            M6 headline cps 0.8508 (HALT if it does not — that would mean the eval path changed).
  kstep_5 — filter.empty_fallback = {mode:kstep, k:5}; the fallback acts ONLY on empty-branch steps.

Registered hypothesis H-B-fallback (before the data): kstep k=5 reduces collision vs none with CIs
separated, AND every outcome flip is confined to empty-branch episodes, AND reach/timeout do not degrade
(CIs overlap none or improve). ADOPT only if all three hold. Applies the same three-clause rule as v2.7.3 M7,
mechanically. Does NOT edit 02_control or any config default. Writes aggregate + attribution + per-arm episode
CSVs under <run-dir>/fallback_trial/ (measurement persistence). No verdict language.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.common.filter_hardnet import _cbf_terms
from src.common.rk4 import rk4_step
from src.common.value_net import make_h_fn
from src.eval.bootstrap import within_seed_ci
from src.eval.build_pools import load_pool, EVAL_POOLS_DIR, DEFAULT_OUTPUT_DIR
from src.eval.evaluate import evaluate, EVAL_EPISODE_COLUMNS
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.frameworks.jt_pncbf.train import load_framework_from_checkpoint

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--run-dir", required=True)
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--roll-n", type=int, default=600)          # instrumented subset for firing/chatter
ap.add_argument("--pool", default="eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl")
ap.add_argument("--m6-cps", type=float, default=0.8508367)   # M6 headline to reproduce (none arm)
ap.add_argument("--repro-tol", type=float, default=1e-4)     # bit-identical tolerance
a = ap.parse_args()

pool_path = (EVAL_POOLS_DIR / a.pool) if (EVAL_POOLS_DIR / a.pool).exists() else (DEFAULT_OUTPUT_DIR / a.pool)
outdir = Path(a.run_dir) / "fallback_trial"; outdir.mkdir(parents=True, exist_ok=True)
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
boot = ck["config"]["eval"]["bootstrap"]


def _filter_for(mode: str):
    filt = copy.deepcopy(ck["config"]["filter"])
    if mode == "none":
        filt["empty_fallback"] = {"mode": "none", "k": int(filt.get("empty_fallback", {}).get("k", 10))}
    else:
        filt["empty_fallback"] = {"mode": "kstep", "k": 5}
    return filt


def _run(mode: str):
    fw, cfg, _ = load_framework_from_checkpoint(Path(a.ckpt), config_overrides={"filter": _filter_for(mode)})
    t0 = time.perf_counter()
    res = evaluate(fw, pool_path, cfg, mode="final", step=int(ck["step"]),
                   ckpt_name=Path(a.ckpt).name, max_scenes=a.n, include_lqr_baseline=False)
    wall = time.perf_counter() - t0
    rows = res.episode_rows
    ci = within_seed_ci(rows, n_resample=int(boot["n_resample"]), seed=int(boot["seed"]))
    with (outdir / f"episodes_{mode}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EVAL_EPISODE_COLUMNS); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in EVAL_EPISODE_COLUMNS})
    agg = {"mode": mode, "n": len(rows), "wall_s": round(wall, 1)}
    for m in ("cps", "reach", "collision", "oob", "stuck", "timeout", "infeasibility"):
        agg[m] = round(float(ci["mean"][m]), 6)
        agg[m + "_ci"] = [round(float(ci["ci"][m]["lo"]), 6), round(float(ci["ci"][m]["hi"]), 6)]
    agg["empty_step_frac_mean"] = round(sum(float(r["empty_step_frac"]) for r in rows) / len(rows), 6)
    return agg, rows


agg_none, rows_none = _run("none")
# --- REPRODUCTION GATE: none arm must reproduce the M6 headline cps ---
d_repro = agg_none["cps"] - a.m6_cps
print(f"[repro] none cps = {agg_none['cps']:.6f}  vs M6 headline {a.m6_cps:.6f}  delta = {d_repro:+.6f}")
if abs(d_repro) > a.repro_tol:
    raise SystemExit(f"HALT: none arm cps {agg_none['cps']:.6f} does not reproduce M6 headline "
                     f"{a.m6_cps:.6f} (delta {d_repro:+.6f} > tol {a.repro_tol}); the eval path changed.")
agg_kstep, rows_kstep = _run("kstep")

# ---- flip anatomy ----
none = {int(r["episode_idx"]): r for r in rows_none}
kstep = {int(r["episode_idx"]): r for r in rows_kstep}
ids = sorted(set(none) & set(kstep))
col_n = {i: int(float(none[i]["collision"])) for i in ids}
col_k = {i: int(float(kstep[i]["collision"])) for i in ids}
emp = {i: float(none[i]["empty_step_frac"]) for i in ids}
fixed = [i for i in ids if col_n[i] == 1 and col_k[i] == 0]     # kstep averts a none-collision
new = [i for i in ids if col_n[i] == 0 and col_k[i] == 1]       # kstep introduces a collision
out_flip = [i for i in ids if none[i]["outcome"] != kstep[i]["outcome"]]

# ||L_g V_hat|| at t0 for all episodes
fw, cfg, _ = load_framework_from_checkpoint(Path(a.ckpt))
system = fw.system
h_fn = make_h_fn(fw.value_net, system)
scenes = load_pool(pool_path).scenes[: a.n]
bs = batch_scenes(scenes, device=torch.device("cpu"), dtype=torch.float32)
x0 = system.wrap_state(initial_states_from_batch(bs).float())
u0 = torch.zeros(x0.shape[0], int(system.action_dim))
_, _, lg = _cbf_terms(system, h_fn, x0, bs, u0, create_graph=False)
lg0 = torch.linalg.norm(lg, dim=1).detach().numpy()
lo_auth = lg0 < 1e-3


def _part(idlist):
    if not idlist:
        return {"n": 0}
    e = np.array([emp[i] > 0 for i in idlist]); la = lo_auth[idlist]
    return {"n": len(idlist), "ids": idlist[:50], "had_empty_frac": round(float(e.mean()), 3),
            "lo_authority_t0_frac": round(float(la.mean()), 3),
            "lg0_median": round(float(np.median(lg0[idlist])), 4)}


# confinement: fraction of outcome flips that are empty-branch episodes (v2.7.3 analog = 1.0)
flip_empty = [i for i in out_flip if emp[i] > 0]
confinement = round(len(flip_empty) / max(1, len(out_flip)), 4) if out_flip else None

# ---- instrumented rollout: firing counts + chatter ||Δu|| on empty steps (both modes) ----
dt = float(cfg["env"]["dt"]); max_steps = int(cfg["eval"]["max_steps"])
bs_r = batch_scenes(scenes[: a.roll_n], device=torch.device("cpu"), dtype=torch.float32)


def _roll(mode):
    fw2, cfg2, _ = load_framework_from_checkpoint(Path(a.ckpt), config_overrides={"filter": _filter_for(mode)})
    x = system.wrap_state(initial_states_from_batch(bs_r).float())
    prev_u = None; empty_steps = 0; du_sum = 0.0; du_cnt = 0
    with torch.no_grad():
        for _ in range(max_steps):
            un = fw2.policy(x, bs_r); u, _ = fw2.filter(x, un, bs_r)
            em = fw2._filter.last_empty.bool()
            empty_steps += int(em.sum())
            if prev_u is not None and bool(em.any()):
                du = torch.linalg.norm(u[em] - prev_u[em], dim=1)
                du_sum += float(du.sum()); du_cnt += int(em.sum())
            prev_u = u
            x = rk4_step(system, x, u, dt)
    return {"total_empty_steps": empty_steps, "mean_du_on_empty": round(du_sum / max(1, du_cnt), 5)}


roll_none = _roll("none"); roll_k = _roll("kstep")

# ---- adoption rule (three clauses, mechanical) ----
c1_collision_sep = agg_kstep["collision_ci"][1] < agg_none["collision_ci"][0]     # kstep hi < none lo
c2_confined = (confinement == 1.0) if confinement is not None else True           # all flips empty-branch
# reach not degraded: kstep reach CI overlaps none's OR improves (kstep lo >= none lo counts as not-degraded)
c3_reach_ok = (agg_kstep["reach_ci"][1] >= agg_none["reach_ci"][0]) and (agg_kstep["reach"] >= agg_none["reach"] - 1e-9 or
              agg_kstep["reach_ci"][0] <= agg_none["reach_ci"][1])
# timeout not degraded: kstep timeout CI overlaps none's OR improves (kstep not CI-worse-above)
c3_timeout_ok = not (agg_kstep["timeout_ci"][0] > agg_none["timeout_ci"][1])
c3_no_paralysis = bool(c3_reach_ok and c3_timeout_ok)
adopt = bool(c1_collision_sep and c2_confined and c3_no_paralysis)

report = {
    "ckpt": a.ckpt, "pool": a.pool, "n": a.n,
    "arms": {"none": agg_none, "kstep_k5": agg_kstep},
    "delta": {"d_cps": round(agg_kstep["cps"] - agg_none["cps"], 6),
              "d_collision": round(agg_kstep["collision"] - agg_none["collision"], 6),
              "wall_ratio": round(agg_kstep["wall_s"] / max(agg_none["wall_s"], 1e-9), 3)},
    "repro_gate": {"none_cps": agg_none["cps"], "m6_headline": a.m6_cps, "delta": round(d_repro, 8),
                   "reproduced": bool(abs(d_repro) <= a.repro_tol)},
    "flips": {"n_outcome_flips": len(out_flip), "fixed_collisions": _part(fixed),
              "new_collisions": _part(new), "empty_branch_confinement_frac": confinement,
              "flip_ids": out_flip[:50],
              "flip_table": [{"episode": i, "none": none[i]["outcome"], "kstep": kstep[i]["outcome"],
                              "empty_step_frac_none": round(emp[i], 4), "lg0": round(float(lg0[i]), 4),
                              "lo_authority_t0": bool(lo_auth[i])} for i in out_flip[:50]]},
    "collided_none_partition": _part([i for i in ids if col_n[i] == 1]),
    "lg0_all": {"median": round(float(np.median(lg0)), 4), "lo_authority_t0_frac": round(float(lo_auth.mean()), 4)},
    "firing_counts": {"roll_subset_n": a.roll_n, "none_empty_steps": roll_none["total_empty_steps"],
                      "kstep_empty_steps": roll_k["total_empty_steps"],
                      "note": "empty flags are mode-invariant; counts on the roll subset"},
    "chatter_mean_du_on_empty": {"none": roll_none["mean_du_on_empty"], "kstep": roll_k["mean_du_on_empty"]},
    "adoption_rule": {
        "clause1_collision_CI_separated_below": {"pass": c1_collision_sep,
            "kstep_collision_ci_hi": agg_kstep["collision_ci"][1], "none_collision_ci_lo": agg_none["collision_ci"][0]},
        "clause2_flips_confined_to_empty_branch": {"pass": c2_confined, "confinement_frac": confinement},
        "clause3_no_conservative_paralysis": {"pass": c3_no_paralysis, "reach_ok": bool(c3_reach_ok),
            "timeout_ok": bool(c3_timeout_ok),
            "kstep_reach_ci": agg_kstep["reach_ci"], "none_reach_ci": agg_none["reach_ci"],
            "kstep_timeout_ci": agg_kstep["timeout_ci"], "none_timeout_ci": agg_none["timeout_ci"]},
        "ADOPT": adopt,
        "deployed_default_for_SO3": ("kstep k=5 (rule fired — default flip is a Strategist protocol edit at close)"
                                     if adopt else "none (rule did not fire — default stays none for SO(3))"),
    },
}
(outdir / "m7_fallback_trial.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
