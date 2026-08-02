"""v2.8.0 S3 R2 — reconstruct the P2 Jacobian-class series from checkpoints.

The dual arm logged P2 every step (jac_classes.csv, 48000 rows). The enum arm's crashed run wrote NO
jac_classes.csv (incremental flush was added only for the resume), so its P2 for steps 2000-34500 is lost;
only the resume (34500-50000) is logged. This script reconstructs the clip-class fractions from a
checkpoint by rolling the checkpoint's own policy+filter (its stored TRAINING filter config: projection per
arm, empty_fallback none, band_hazard) on a FIXED in-loop batch for bptt_T steps and recomputing exactly
the fractions losses.py logs (no-clip / 0<|A|<m / |A|=m / |A|=m-1 / intervention). Because the training
rollout draws states from the replay buffer while this reconstruction draws them from the in-loop pool,
the reconstruction is a PROXY, not a bit-copy -- so it is VALIDATED on the dual arm (reconstructed vs
logged at the 1500-step checkpoint grid); only if the dual validation agrees is the enum reconstruction
trustworthy, otherwise the enum lost segment is reported as 'not recorded'.

Usage: --ckpt-dir <dir> --logged <jac_classes.csv|none> --tag <name>  [--max-step N]
Sidecar: data/runs/v2.8.0/s3_eval/p2_reconstruct_<tag>.json"""
from __future__ import annotations
import argparse, copy, csv, json, re
from pathlib import Path
import numpy as np, torch

from src.eval.build_pools import load_pool
from src.eval.run_full import _load_framework as load_fw
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.common.rk4 import rk4_step

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
POOL = REPO / "data/secured_data/pools/eval_inloop_quadrotor-3d-d2r_n500_seed12345.pkl"
OUT = REPO / "data/runs/v2.8.0/s3_eval"; OUT.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BPTT_T, DT = 30, 0.05
CLASSES = ["frac_noclip", "frac_partial", "frac_vertex", "frac_free1", "frac_intervention",
           "zero_jac_frac_all_real", "zero_jac_frac_dual"]

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt-dir", required=True)
ap.add_argument("--logged", required=True)          # path to jac_classes.csv, or "none"
ap.add_argument("--tag", required=True)
ap.add_argument("--max-step", type=int, default=10**9)
a = ap.parse_args()

scenes = load_pool(POOL).scenes
bs = batch_scenes(scenes, device=DEV, dtype=torch.float32)


def clip_classes(safe_stack, action_stack, ub):
    ss = safe_stack.reshape(-1, safe_stack.shape[-1]); m = ss.shape[-1]
    clip = ((ss - ub[:, 0]).abs() < 1e-6) | ((ss - ub[:, 1]).abs() < 1e-6)
    nclip = clip.sum(dim=1)
    interv = (torch.linalg.norm(safe_stack - action_stack, dim=-1).reshape(-1) > 1e-3)
    return {"n_rows": int(ss.shape[0]),
            "frac_noclip": float((nclip == 0).float().mean()),
            "frac_partial": float(((nclip > 0) & (nclip < m)).float().mean()),
            "frac_vertex": float((nclip == m).float().mean()),
            "frac_free1": float((nclip == m - 1).float().mean()),
            "frac_intervention": float(interv.float().mean()),
            "zero_jac_frac_all_real": float((nclip == m).float().mean()),
            "zero_jac_frac_dual": float((nclip >= m - 1).float().mean())}


def reconstruct(ckpt):
    ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    filt = copy.deepcopy(ck["config"]["filter"])         # the TRAINING filter (projection per arm, fallback none)
    over = {"env": {"dt": DT}, "filter": filt}
    fw, cfg, ck2 = load_fw(str(ckpt), config_overrides=over)
    for n in ("value_net", "policy_net"):
        m = getattr(fw, n, None); m.to(DEV) if m is not None else None
    x = initial_states_from_batch(bs); noms, safes = [], []
    with torch.no_grad():
        for _ in range(BPTT_T):
            un = fw.policy(x, bs); us = fw.filter(x, un, bs)[0]
            noms.append(un); safes.append(us)
            x = fw.system.wrap_state(rk4_step(fw.system, x, us, DT))
    ub = fw.system.u_bounds.to(device=DEV, dtype=x.dtype)
    rec = clip_classes(torch.stack(safes, 0), torch.stack(noms, 0), ub)
    return int(ck2["step"]), rec


ckpts = sorted(Path(a.ckpt_dir).glob("step_*.pt"),
               key=lambda p: int(re.search(r"step_(\d+)", p.name).group(1)))
series = []
for c in ckpts:
    st = int(re.search(r"step_(\d+)", c.name).group(1))
    if st > a.max_step:
        continue
    step, rec = reconstruct(c)
    rec["step"] = step; series.append(rec)
    print(f"  recon step {step:>6}: noclip {rec['frac_noclip']:.3f} partial {rec['frac_partial']:.3f} "
          f"vertex {rec['frac_vertex']:.3f} free1 {rec['frac_free1']:.3f} interv {rec['frac_intervention']:.3f}", flush=True)

rep = {"tag": a.tag, "ckpt_dir": a.ckpt_dir, "bptt_T": BPTT_T, "pool": "eval_inloop_n500_seed12345",
       "note": "reconstruction proxy: rollout from in-loop pool states (training used replay-buffer states)",
       "reconstructed": series}

# --- validation against logged, if provided ---
if a.logged.lower() != "none" and Path(a.logged).exists():
    logged = {}
    with open(a.logged) as f:
        for row in csv.DictReader(f):
            logged[int(row["step"])] = {k: float(row[k]) for k in CLASSES}
    diffs = {k: [] for k in CLASSES}; pairs = []
    for s in series:
        st = s["step"]
        if st in logged:
            pairs.append(st)
            for k in CLASSES:
                diffs[k].append(abs(s[k] - logged[st][k]))
    agree = {k: {"mean_abs_diff": float(np.mean(diffs[k])) if diffs[k] else None,
                 "max_abs_diff": float(np.max(diffs[k])) if diffs[k] else None} for k in CLASSES}
    rep["validation"] = {"n_matched_steps": len(pairs), "grid": pairs, "agreement": agree,
                         "verdict": "AGREES" if all((agree[k]["mean_abs_diff"] or 0) < 0.05 for k in CLASSES)
                                    else "DISAGREES (>0.05 mean abs diff on some class)"}
    print(f"\nVALIDATION vs logged ({len(pairs)} matched steps):")
    for k in CLASSES:
        print(f"   {k:26s} mean|Δ| {agree[k]['mean_abs_diff']:.4f}  max|Δ| {agree[k]['max_abs_diff']:.4f}")
    print(f"   VERDICT: {rep['validation']['verdict']}")

(OUT / f"p2_reconstruct_{a.tag}.json").write_text(json.dumps(rep, indent=2) + "\n")
print(f"-> {OUT/('p2_reconstruct_'+a.tag+'.json')}")
