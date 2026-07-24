"""v2.7.5 — verify the eval graph-retention fix: (§2) numerically inert, (§3) leak gone.

§2 inertness gate: re-run Arm A (dt_sim=dt_ctrl=0.05, max_steps=200, stuck 60, batch 2000) with the fix and
require cps BIT-IDENTICAL to the pre-fix completed Arm A (0.850837) — identical, not within 1e-3 — so no
forward value changed and Arm B stays valid.

§3 leak gate: peak host RSS vs control-computation count (200 vs 1000) at a fixed batch, each measured in its
OWN subprocess (ru_maxrss is a per-process high-water mark). Gate: RSS approximately FLAT in step count rather
than scaling with it. Uses a reduced probe batch (diagnostic, not an eval arm — the 2000 pin governs arms).

Writes JSON; nothing is printed as the sole record (06_workflow §3.2).
"""
from __future__ import annotations

import copy
import json
import os
import resource
import subprocess
import sys
from pathlib import Path

import torch

CKPT = Path("data/runs/v2.7.4/set__20260720-083533__seed42/v2.7.4__20260720-091830__seed42/checkpoints/best.pt")
OUTDIR = Path("data/runs/v2.7.5/dt_ctrl_arms"); OUTDIR.mkdir(parents=True, exist_ok=True)
REF_A_CPS = 0.850837          # pre-fix completed Arm A (metrics_A_20Hz_coarse.json)


def _cfg_for(dt, max_steps, dt_ctrl, stuck_window, ck):
    over = {"env": {"dt": dt, "stuck_window_steps": stuck_window, "stuck_radius": 0.10},
            "eval": {"max_steps": max_steps, "dt_ctrl": dt_ctrl},
            "filter": copy.deepcopy(ck["config"]["filter"])}
    over["filter"]["empty_fallback"] = {"mode": "none", "k": int(over["filter"].get("empty_fallback", {}).get("k", 10))}
    return over


def probe_child(max_steps, dt, dt_ctrl, n_scenes):
    """Child mode: roll `n_scenes` episodes and report peak RSS (MiB) for THIS process."""
    from src.eval.build_pools import load_pool, EVAL_POOLS_DIR
    from src.eval.evaluate import _filter_adapter, _tensor_options, make_batched_scene
    from src.eval.rollout import rollout_eval
    from src.envs.scene_batch import initial_states_from_batch
    from src.frameworks.jt_pncbf.train import load_framework_from_checkpoint
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    fw, cfg, _ = load_framework_from_checkpoint(CKPT, config_overrides=_cfg_for(dt, max_steps, dt_ctrl, 300 if dt < 0.05 else 60, ck))
    scenes = load_pool(EVAL_POOLS_DIR / "eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl").scenes[:n_scenes]
    dtype, device = _tensor_options(fw.system, fw)
    bscene = make_batched_scene(scenes, device=device, dtype=dtype)
    x0 = initial_states_from_batch(bscene)
    rollout_eval(fw.system, fw.policy, _filter_adapter(fw), bscene, x0,
                 max_steps=max_steps, dt=dt, config=cfg)
    peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(json.dumps({"peak_rss_mib": round(peak_mib, 1)}))


if len(sys.argv) > 1 and sys.argv[1] == "child":
    probe_child(int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]))
    raise SystemExit(0)

result = {}

# ---------- §2 inertness gate: Arm A re-run at the pinned batch 2000 ----------
from src.eval.bootstrap import within_seed_ci
from src.eval.build_pools import EVAL_POOLS_DIR
from src.eval.evaluate import evaluate
from src.frameworks.jt_pncbf.train import load_framework_from_checkpoint

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
fw, cfg, _ = load_framework_from_checkpoint(CKPT, config_overrides=_cfg_for(0.05, 200, 0.05, 60, ck))
res = evaluate(fw, EVAL_POOLS_DIR / "eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl", cfg, mode="final",
               step=int(ck["step"]), ckpt_name=CKPT.name, max_scenes=None, include_lqr_baseline=False)
boot = ck["config"]["eval"]["bootstrap"]
ci = within_seed_ci(res.episode_rows, n_resample=int(boot["n_resample"]), seed=int(boot["seed"]))
cps_fixed = round(float(ci["mean"]["cps"]), 6)
result["inertness_gate"] = {
    "arm_A_cps_prefix_reference": REF_A_CPS, "arm_A_cps_with_fix": cps_fixed,
    "delta": round(cps_fixed - REF_A_CPS, 9),
    "bit_identical": bool(cps_fixed == REF_A_CPS),
    "n": len(res.episode_rows), "eval_batch": 2000,
}
(OUTDIR / "fixcheck.json").write_text(json.dumps(result, indent=2) + "\n")
if cps_fixed != REF_A_CPS:
    raise SystemExit(f"HALT §2: Arm A cps with fix {cps_fixed} != pre-fix {REF_A_CPS} — fix is NOT inert.")

# ---------- §3 leak gate: peak RSS vs control-computation count, separate subprocesses ----------
PROBE_N = 400          # diagnostic batch (arms keep the pinned 2000)
probes = []
for (ms, dt, dtc) in [(200, 0.05, 0.05), (1000, 0.01, 0.01)]:
    out = subprocess.run([sys.executable, __file__, "child", str(ms), str(dt), str(dtc), str(PROBE_N)],
                         capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "."})
    peak = json.loads(out.stdout.strip().splitlines()[-1])["peak_rss_mib"]
    probes.append({"max_steps": ms, "dt": dt, "dt_ctrl": dtc, "control_computations": ms,
                   "probe_batch": PROBE_N, "peak_rss_mib": peak})
r200, r1000 = probes[0]["peak_rss_mib"], probes[1]["peak_rss_mib"]
result["leak_gate"] = {
    "probes": probes,
    "ratio_1000_over_200": round(r1000 / max(r200, 1e-9), 3),
    "control_computation_ratio": 5.0,
    "flat_in_step_count": bool(r1000 / max(r200, 1e-9) < 1.5),
    "note": "peak RSS is a per-process high-water mark (ru_maxrss); each probe ran in its own subprocess. "
            "Pre-fix, retained graphs scaled with control-computation count; flat here means the leak is gone.",
}
(OUTDIR / "fixcheck.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
