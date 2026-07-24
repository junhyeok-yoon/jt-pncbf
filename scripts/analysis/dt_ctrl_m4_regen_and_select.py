"""v2.7.5 M4.1/M4.2 — regenerate the missing per-episode artifacts at the PINNED batch, gate them against the
registered numbers, then register the scene selection BEFORE any figure is made.

M4.1 audit found the arms persisted only u_cmd/u_nom/n_steps/dt/dt_ctrl: no states, no per-step h_star/V-hat,
no filter flags, and no per_episode.csv. Scene selection (M4.2) requires RECORDED outcome labels, so those must
exist first.

Rather than a subset re-run (which would change the batch size and force a `viz_only` numeric caveat), this
re-runs each arm through the SAME canonical evaluate() path, SAME config, SAME checkpoint, at the SAME pinned
batch 2000 that produced the registered numbers. That path is deterministic (Arm A reproduced bit-identically
in M1 and again post-fix), so the regenerated per-episode labels ARE the recorded ones, and the extracted
states are fully comparable to the pinned arms.

HARD GATE: every regenerated aggregate must match the arm's registered metrics_<arm>.json. Any mismatch stops
the script (M4.5) rather than plotting either value.

Writes per arm: per_episode_<arm>.csv, states_<arm>.npz (states + filter/empty flags). Then registers the
selection to m4_scene_selection.json with the rule that picked each id.
"""
from __future__ import annotations

import copy, csv, json
from pathlib import Path

import numpy as np
import torch

from src.eval.build_pools import EVAL_POOLS_DIR
from src.eval.evaluate import evaluate
from src.frameworks.jt_pncbf.train import load_framework_from_checkpoint

CKPT = Path("data/runs/v2.7.4/set__20260720-083533__seed42/v2.7.4__20260720-091830__seed42/checkpoints/best.pt")
POOL = EVAL_POOLS_DIR / "eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl"
RD = Path("data/runs/v2.7.5/dt_ctrl_arms")
ARMS = {
    "A_20Hz_coarse": dict(dt=0.05, max_steps=200,  dt_ctrl=0.05, stuck_window=60),
    "B_20Hz_fine":   dict(dt=0.01, max_steps=1000, dt_ctrl=0.05, stuck_window=300),
    "C_100Hz_fine":  dict(dt=0.01, max_steps=1000, dt_ctrl=0.01, stuck_window=300),
}
GATE_KEYS = ("cps", "reach", "collision", "oob", "stuck", "timeout", "infeasibility")

ck = torch.load(CKPT, map_location="cpu", weights_only=False)

def cfg_over(spec):
    o = {"env": {"dt": spec["dt"], "stuck_window_steps": spec["stuck_window"], "stuck_radius": 0.10},
         "eval": {"max_steps": spec["max_steps"], "dt_ctrl": spec["dt_ctrl"]},
         "filter": copy.deepcopy(ck["config"]["filter"])}
    o["filter"]["empty_fallback"] = {"mode": "none", "k": 10}
    return o

from src.eval.bootstrap import within_seed_ci
gate_report = {}
per_arm_rows = {}
for name, spec in ARMS.items():
    fw, cfg, _ = load_framework_from_checkpoint(CKPT, config_overrides=cfg_over(spec))
    res = evaluate(fw, POOL, cfg, mode="final", step=int(ck["step"]), ckpt_name=CKPT.name,
                   max_scenes=None, include_lqr_baseline=False)     # eval_batch_size None -> pinned 2000
    rows = res.episode_rows
    boot = ck["config"]["eval"]["bootstrap"]
    ci = within_seed_ci(rows, n_resample=int(boot["n_resample"]), seed=int(boot["seed"]))
    registered = json.loads((RD / f"metrics_{name}.json").read_text())
    diffs = {k: round(float(ci["mean"][k]) - float(registered[k]), 9) for k in GATE_KEYS}
    ok = all(abs(v) <= 1e-6 for v in diffs.values())
    gate_report[name] = {"match": ok, "diffs_vs_registered": diffs,
                         "regenerated": {k: round(float(ci["mean"][k]), 6) for k in GATE_KEYS},
                         "registered": {k: float(registered[k]) for k in GATE_KEYS}}
    print(f"[{name}] gate match={ok} diffs={diffs}", flush=True)
    if not ok:
        (RD / "m4_gate_report.json").write_text(json.dumps(gate_report, indent=2) + "\n")
        raise SystemExit(f"STOP (M4.5): regenerated aggregates disagree with the registered numbers for {name}: {diffs}")

    # per-episode recorded labels
    with (RD / f"per_episode_{name}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode_idx", "outcome", "n_steps", "cps_episode"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in ("episode_idx", "outcome", "n_steps", "cps_episode")})
    per_arm_rows[name] = {int(r["episode_idx"]): r for r in rows}

    # full states + filter/empty flags (pinned batch, so comparable)
    st = np.stack([t.filtered.states[:, 0, :].detach().to(torch.float32).cpu().numpy() for t in res.trajectories], 0)
    iv = np.stack([t.filtered.intervention_mask[:, 0].detach().cpu().numpy() for t in res.trajectories], 0)
    em = np.stack([(t.filtered.empty[:, 0].detach().cpu().numpy() if t.filtered.empty is not None
                    else np.zeros(t.filtered.intervention_mask.shape[0], bool)) for t in res.trajectories], 0)
    np.savez_compressed(RD / f"states_{name}.npz", states=st, intervention=iv, empty=em,
                        dt=np.float64(spec["dt"]), dt_ctrl=np.float64(spec["dt_ctrl"]))
    print(f"[{name}] states {st.shape} written", flush=True)

(RD / "m4_gate_report.json").write_text(json.dumps(gate_report, indent=2) + "\n")

# ---------------- M4.2 selection, registered BEFORE any figure ----------------
B, C = per_arm_rows["B_20Hz_fine"], per_arm_rows["C_100Hz_fine"]
ids = sorted(set(B) & set(C))
fwd = [i for i in ids if B[i]["outcome"] == "collision" and C[i]["outcome"] == "goal"]
rev = [i for i in ids if C[i]["outcome"] == "collision" and B[i]["outcome"] == "goal"]

# TV/s per scene per arm from the persisted action streams (same computation as the registered aggregate)
def tv_per_s(arm):
    z = np.load(RD / f"action_stream_{arm}.npz")
    u, ns, dt = z["u_cmd"].astype(np.float64), z["n_steps"], float(z["dt"])
    bw = 4.905
    out = np.zeros(u.shape[0])
    for i in range(u.shape[0]):
        k = max(1, int(ns[i])); seg = u[:k + 1, ] if False else u[i, :k + 1]
        out[i] = (np.abs(np.diff(seg, axis=0)).sum() / (k * dt * bw * u.shape[2])) if seg.shape[0] > 1 else 0.0
    return out
tvB, tvC = tv_per_s("B_20Hz_fine"), tv_per_s("C_100Hz_fine")
d_tv = tvC - tvB
tv_top = [int(i) for i in np.argsort(-d_tv)[:6]]

cpsB = np.array([float(B[i]["cps_episode"]) for i in ids])
cpsC = np.array([float(C[i]["cps_episode"]) for i in ids])
medB, medC = float(np.median(cpsB)), float(np.median(cpsC))
dist = np.abs(cpsB - medB) + np.abs(cpsC - medC)
med_near = [int(ids[j]) for j in np.argsort(dist)[:6]]

sel, rule = [], {}
def add(lst, r):
    for i in lst:
        if i not in sel and len(sel) < 24:
            sel.append(int(i)); rule[str(int(i))] = r
add(fwd, "rule1_forward_flip_B_collision_to_C_goal")
add(rev, "rule1_reverse_flip_C_collision_to_B_goal")
add(tv_top, "rule2_top6_u_cmd_TVps_increase_B_to_C")
add(med_near, "rule3_nearest_median_cps_both_arms")

selection = {
    "registered_before_any_figure": True,
    "source": "per_episode_{B,C}.csv recorded outcome labels (regenerated at the pinned batch 2000; gate matched)",
    "rule1_forward_flips_B_collide_C_reach": {"count": len(fwd), "ids": fwd},
    "rule1_reverse_flips_C_collide_B_reach": {"count": len(rev), "ids": rev},
    "rule2_top6_tv_increase": {"ids": tv_top, "delta_tv_per_s": [round(float(d_tv[i]), 5) for i in tv_top]},
    "rule3_median_cps": {"ids": med_near, "median_cps_B": round(medB, 5), "median_cps_C": round(medC, 5)},
    "selected_ids": sel, "n_selected": len(sel), "rule_per_id": rule,
}
(RD / "m4_scene_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
print(json.dumps({k: selection[k] for k in
                  ("rule1_forward_flips_B_collide_C_reach", "rule1_reverse_flips_C_collide_B_reach",
                   "rule2_top6_tv_increase", "rule3_median_cps", "n_selected")}, indent=2))
