"""v2.7.2 Stage-3D — attribution (S3). Cross-tabs the mode=none vs mode=kstep outcome flips with
{had empty steps} and ||L_g V_hat|| at t0 (the P4 low-authority tail); counts fallback firings; measures the
chattering step-change ||Δu|| on empty steps (none vs kstep); reports the wall-clock ratio from S2. Eval-only."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch

from src.common.filter_hardnet import _cbf_terms
from src.common.rk4 import rk4_step
from src.common.value_net import make_h_fn
from src.eval.build_pools import load_pool, DEFAULT_OUTPUT_DIR
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.frameworks.jt_pncbf.train import load_framework_from_checkpoint

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--dir", required=True)                      # scratchpad/stage3d with episodes_{none,kstep}.csv
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--roll-n", type=int, default=600)           # subset for the instrumented chatter/firing roll
a = ap.parse_args()
D = Path(a.dir)


def _load(mode):
    rows = {}
    with (D / f"episodes_{mode}.csv").open() as f:
        for r in csv.DictReader(f):
            rows[int(r["episode_idx"])] = r
    return rows


none, kstep = _load("none"), _load("kstep")
ids = sorted(set(none) & set(kstep))
col_n = {i: int(float(none[i]["collision"])) for i in ids}
col_k = {i: int(float(kstep[i]["collision"])) for i in ids}
emp = {i: float(none[i]["empty_step_frac"]) for i in ids}     # had-empty (baseline) per episode
fixed = [i for i in ids if col_n[i] == 1 and col_k[i] == 0]   # none collided, kstep did not
new = [i for i in ids if col_n[i] == 0 and col_k[i] == 1]     # kstep introduced a collision
out_flip = [i for i in ids if none[i]["outcome"] != kstep[i]["outcome"]]

# ---- ||L_g V_hat|| at t0 for all episodes ----
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
fw, cfg, _ = load_framework_from_checkpoint(Path(a.ckpt))
system = fw.system
h_fn = make_h_fn(fw.value_net, system)
pool = load_pool(DEFAULT_OUTPUT_DIR / "eval_full_quadrotor-3d_n2000_seed23456.pkl")
scenes = pool.scenes[: a.n]
bs = batch_scenes(scenes, device=torch.device("cpu"), dtype=torch.float32)
x0 = system.wrap_state(initial_states_from_batch(bs).float())
u0 = torch.zeros(x0.shape[0], int(system.action_dim))
_, _, lg = _cbf_terms(system, h_fn, x0, bs, u0, create_graph=False)
lg0 = torch.linalg.norm(lg, dim=1).detach().numpy()          # ||L_g V_hat|| at t0 per episode
lo_auth = lg0 < 1e-3                                          # degenerate-authority tail at t0

def _part(idlist):
    if not idlist:
        return {"n": 0}
    e = np.array([emp[i] > 0 for i in idlist]); la = lo_auth[idlist]
    return {"n": len(idlist), "ids": idlist[:50], "had_empty_frac": round(float(e.mean()), 3),
            "lo_authority_t0_frac": round(float(la.mean()), 3),
            "lg0_median": round(float(np.median(lg0[idlist])), 4)}

# ---- instrumented rollout: firing counts + chatter ||Δu|| on empty steps (both modes) ----
dt = float(cfg["env"]["dt"]); max_steps = int(cfg["eval"]["max_steps"])

# instrumented rollout on a subset (empty flags are mode-INVARIANT — the fallback changes the empty-row
# ACTION only, not which rows are empty; f2 confirms; so firing counts are read on this subset and scaled note).
bs_r = batch_scenes(scenes[: a.roll_n], device=torch.device("cpu"), dtype=torch.float32)

def _roll(mode):
    filt = copy.deepcopy(ck["config"]["filter"])
    if mode == "none":
        filt.pop("empty_fallback", None)
    else:
        filt["empty_fallback"] = {"mode": "kstep", "k": 5}
    fw2, cfg2, _ = load_framework_from_checkpoint(Path(a.ckpt), config_overrides={"filter": filt})
    x = system.wrap_state(initial_states_from_batch(bs_r).float())
    prev_u = None
    empty_steps = 0
    du_empty_sum = 0.0; du_empty_cnt = 0
    with torch.no_grad():
        for _ in range(max_steps):
            un = fw2.policy(x, bs_r); u, _ = fw2.filter(x, un, bs_r)
            em = fw2._filter.last_empty.bool()
            empty_steps += int(em.sum())
            if prev_u is not None and bool(em.any()):
                du = torch.linalg.norm(u[em] - prev_u[em], dim=1)
                du_empty_sum += float(du.sum()); du_empty_cnt += int(em.sum())
            prev_u = u
            x = rk4_step(system, x, u, dt)
    return {"total_empty_steps": empty_steps,
            "mean_du_on_empty": round(du_empty_sum / max(1, du_empty_cnt), 5)}

roll_none = _roll("none"); roll_k = _roll("kstep")

s2 = json.loads((D / "stage3d_aggregate.json").read_text())
report = {
    "flips": {"fixed_collisions": _part(fixed), "new_collisions": _part(new),
              "n_outcome_flips": len(out_flip)},
    "collided_none_partition": _part([i for i in ids if col_n[i] == 1]),
    "lg0_all": {"median": round(float(np.median(lg0)), 4),
                "lo_authority_t0_frac": round(float(lo_auth.mean()), 4)},
    "firing_counts": {"roll_subset_n": a.roll_n, "none_empty_steps": roll_none["total_empty_steps"],
                      "kstep_empty_steps": roll_k["total_empty_steps"],
                      "note": "empty flags are mode-invariant (f2); counts on the roll subset"},
    "chatter_mean_du_on_empty": {"none": roll_none["mean_du_on_empty"], "kstep": roll_k["mean_du_on_empty"]},
    "wall": {"none_s": s2["none"]["wall_s"], "kstep_s": s2["kstep_k5"]["wall_s"],
             "ratio": s2["delta"]["wall_ratio"]},
}
(D / "stage3d_attribution.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
