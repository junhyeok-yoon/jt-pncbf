"""v2.7.5 — deploy control-rate axis: run Arms A/B/C (eval-only) on the v2.7.4 checkpoint.

Arm A (dt_sim=dt_ctrl=0.05, max_steps=200, stuck_window=60) is the M1 baseline-reproduction gate: cps must
reproduce the registered v2.7.4 value 0.8508 within 1e-3, else STOP (harness drift). Arms B (0.01/0.05) and
C (0.01/0.01) hold 10 s / 3 s physically (max_steps=1000, stuck_window=300, r_stuck=0.10). Frozen across arms:
checkpoint, d2r pool, eval batch=2000 (full pool, one batch), alpha, h_star, observation, policy, filter,
empty_fallback.mode=none. Per arm, from the canonical evaluate() path: cps + 5 outcome fractions + infeasibility
+ saturation_rate with scene-bootstrap 95% CIs, cps decomposed by term, and command TV/s per channel
(u_cmd=filtered u_safe, u_nom=pre-filter) median/p90 over episodes. Persists per-arm action stream npz. No
training, no secured_data writes; checkpoint read read-only in place. Headline is B-vs-C; A-vs-B is the
integration-grid effect, reported separately.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

from src.eval.bootstrap import within_seed_ci
from src.eval.build_pools import EVAL_POOLS_DIR
from src.eval.evaluate import evaluate
from src.frameworks.jt_pncbf.train import load_framework_from_checkpoint

CKPT = Path("data/runs/v2.7.4/set__20260720-083533__seed42/v2.7.4__20260720-091830__seed42/checkpoints/best.pt")
POOL = EVAL_POOLS_DIR / "eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl"
OUTDIR = Path("data/runs/v2.7.5/dt_ctrl_arms"); OUTDIR.mkdir(parents=True, exist_ok=True)
REG_CPS = 0.8508367            # registered v2.7.4 M6 cps (M1 gate ±1e-3)

ARMS = {
    "A_20Hz_coarse": dict(dt=0.05, max_steps=200,  dt_ctrl=0.05, stuck_window=60),
    "B_20Hz_fine":   dict(dt=0.01, max_steps=1000, dt_ctrl=0.05, stuck_window=300),
    "C_100Hz_fine":  dict(dt=0.01, max_steps=1000, dt_ctrl=0.01, stuck_window=300),
}


def run_arm(name, spec):
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    over = {
        "env": {"dt": spec["dt"], "stuck_window_steps": spec["stuck_window"], "stuck_radius": 0.10},
        "eval": {"max_steps": spec["max_steps"], "dt_ctrl": spec["dt_ctrl"]},
        "filter": copy.deepcopy(ck["config"]["filter"]),
    }
    over["filter"]["empty_fallback"] = {"mode": "none", "k": int(over["filter"].get("empty_fallback", {}).get("k", 10))}
    fw, cfg, _ = load_framework_from_checkpoint(CKPT, config_overrides=over)
    # sanity: the merged config carries the rescaled duration + stuck window (M2 pass criterion)
    assert abs(float(cfg["env"]["dt"]) - spec["dt"]) < 1e-12
    assert int(cfg["eval"]["max_steps"]) == spec["max_steps"]
    assert int(cfg["env"]["stuck_window_steps"]) == spec["stuck_window"]
    assert abs(float(cfg["eval"]["dt_ctrl"]) - spec["dt_ctrl"]) < 1e-12
    assert cfg["filter"]["empty_fallback"]["mode"] == "none"

    res = evaluate(fw, POOL, cfg, mode="final", step=int(ck["step"]), ckpt_name=CKPT.name,
                   max_scenes=None, include_lqr_baseline=False)   # eval_batch_size None -> full pool (2000)
    rows = res.episode_rows
    boot = ck["config"]["eval"]["bootstrap"]
    ci = within_seed_ci(rows, n_resample=int(boot["n_resample"]), seed=int(boot["seed"]))

    def mc(m): return round(float(ci["mean"][m]), 6), [round(float(ci["ci"][m]["lo"]), 6), round(float(ci["ci"][m]["hi"]), 6)]
    agg = {"arm": name, **spec, "n": len(rows)}
    for m in ("cps", "reach", "collision", "oob", "stuck", "timeout", "infeasibility"):
        agg[m], agg[m + "_ci"] = mc(m)
    agg["saturation_rate"] = round(float(res.eval_row["saturation_rate"]), 6)

    # cps decomposition by term (from the CI means; sums to cps)
    r, c, s, o, t, inf = (agg["reach"], agg["collision"], agg["stuck"], agg["oob"], agg["timeout"], agg["infeasibility"])
    terms = {"+reach": round(r, 6), "-2*collision": round(-2 * c, 6), "-stuck": round(-s, 6),
             "-0.5*oob": round(-0.5 * o, 6), "-0.5*timeout": round(-0.5 * t, 6), "-0.3*infeasibility": round(-0.3 * inf, 6)}
    agg["cps_decomposition"] = terms
    agg["cps_decomposition_sum"] = round(sum(terms.values()), 6)

    # --- command TV per second, per channel, for u_cmd (u_safe) and u_nom ---
    u_bounds = fw.system.u_bounds.detach().cpu().to(torch.float64).numpy()   # [action_dim, 2]
    box_w = (u_bounds[:, 1] - u_bounds[:, 0])                                 # per-channel box width
    A = int(fw.system.action_dim)
    nstep = {int(r_["episode_idx"]): int(float(r_["n_steps"])) for r_ in rows}
    cmd_stack = [tr.filtered.u_safe[:, 0, :].detach().to(torch.float64).cpu().numpy() for tr in res.trajectories]
    nom_stack = [tr.filtered.u_nom[:, 0, :].detach().to(torch.float64).cpu().numpy() for tr in res.trajectories]

    def tv_per_s(stack):
        # per episode, per channel: sum|du| over active steps / (active_seconds * box_w); active_seconds = n_steps*dt
        out = np.full((len(stack), A), np.nan)
        for i, U in enumerate(stack):
            ns = max(1, nstep.get(i, U.shape[0]))
            seg = U[:ns + 1]                                     # active trajectory (pre done-zero tail)
            if seg.shape[0] < 2:
                out[i, :] = 0.0; continue
            tv = np.abs(np.diff(seg, axis=0)).sum(axis=0)        # [A] total variation per channel
            out[i, :] = tv / (ns * spec["dt"] * box_w)           # per second, box-normalized
        return out

    tv_cmd = tv_per_s(cmd_stack); tv_nom = tv_per_s(nom_stack)
    agg["tv_per_s_box_normalized"] = {
        "note": "TV/s = sum_t|du|/(n_steps*dt*box_width); per channel; ONLY per-second is dt-comparable, per-step |du| is not.",
        "box_width_per_channel": [round(float(x), 4) for x in box_w],
        "u_cmd_median": [round(float(np.nanmedian(tv_cmd[:, k])), 5) for k in range(A)],
        "u_cmd_p90": [round(float(np.nanpercentile(tv_cmd[:, k], 90)), 5) for k in range(A)],
        "u_nom_median": [round(float(np.nanmedian(tv_nom[:, k])), 5) for k in range(A)],
        "u_nom_p90": [round(float(np.nanpercentile(tv_nom[:, k], 90)), 5) for k in range(A)],
        "u_cmd_median_over_channels": round(float(np.nanmedian(tv_cmd)), 5),
        "u_cmd_p90_over_channels": round(float(np.nanpercentile(tv_cmd, 90)), 5),
        "u_nom_median_over_channels": round(float(np.nanmedian(tv_nom)), 5),
        "u_nom_p90_over_channels": round(float(np.nanpercentile(tv_nom, 90)), 5),
    }
    # persist action stream (float32, truncated tail already done-zeroed by rollout)
    np.savez_compressed(OUTDIR / f"action_stream_{name}.npz",
                        u_cmd=np.stack(cmd_stack, 0).astype(np.float32),
                        u_nom=np.stack(nom_stack, 0).astype(np.float32),
                        n_steps=np.array([nstep.get(i, 0) for i in range(len(rows))], dtype=np.int64),
                        dt=np.float64(spec["dt"]), dt_ctrl=np.float64(spec["dt_ctrl"]))
    (OUTDIR / f"metrics_{name}.json").write_text(json.dumps(agg, indent=2) + "\n")
    print(f"[{name}] cps={agg['cps']:.6f} coll={agg['collision']:.4f} infeas={agg['infeasibility']:.4f} "
          f"sat={agg['saturation_rate']:.4f} TVs_cmd_med={agg['tv_per_s_box_normalized']['u_cmd_median_over_channels']}")
    return agg


# --- Arm A first: M1 gate ---
def load_or_run(name, spec):
    """Resume-safe: a complete metrics JSON on disk is reused (an arm is only written after all of its
    computation succeeds), so an externally-interrupted session re-runs ONLY the missing arms."""
    p = OUTDIR / f"metrics_{name}.json"
    if p.exists():
        d = json.loads(p.read_text())
        print(f"[{name}] REUSED existing complete artifact (cps={d['cps']:.6f}) — not re-run")
        return d
    return run_arm(name, spec)


a = load_or_run("A_20Hz_coarse", ARMS["A_20Hz_coarse"])
d = a["cps"] - REG_CPS
print(f"[M1 GATE] Arm A cps {a['cps']:.6f} vs registered {REG_CPS:.6f} delta {d:+.6f}")
if abs(d) > 1e-3:
    raise SystemExit(f"HALT M1: Arm A cps {a['cps']:.6f} not within 1e-3 of {REG_CPS:.6f} (delta {d:+.6f}) — harness drift.")
print("[M1 GATE] PASS — proceeding to Arms B, C")

b = load_or_run("B_20Hz_fine", ARMS["B_20Hz_fine"])
c = load_or_run("C_100Hz_fine", ARMS["C_100Hz_fine"])
json.dump({"A": a, "B": b, "C": c}, open(OUTDIR / "arms_summary.json", "w"), indent=2)
print("DONE — all three arms written to", OUTDIR)
