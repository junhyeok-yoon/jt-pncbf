"""J-arc (v2.5.2 phase J) — does the S' shield transfer to the v2.3.0 LEARNED filter? EVAL-ONLY.
Runs the committed src/eval/dual_arm.py (arms A/B/B') on the v2.3.0 ADOPTED DI checkpoints (the n2000
re-selected steps per data/secured_data/v2.3.0/aggregate/multi_seed_metrics.json: 42@40500, 99@45000,
12345@49500 — the step_0*.pt files, NOT best.pt), full pool eval_full_di_n2000_seed23456, chunk=250.

Config: the checkpoint's OWN v2.3.0 config (filter.alpha_safe=2.0/alpha_unsafe=100/hardnet.eps=5e-4,
h_scale=0.35). ONLY overrides: safety_channel.type=value (barrier = ckpt ValueNet) and the Stage-1
DERIVED shield constants filter.shield.dt_check=0.01 -> thresh=v_max*dt_check/2=0.0125. No margin, nothing
tuned. Pooled = per-seed MEAN (matches the v2.3.0 aggregate method; cps_legacy mean 0.8698 of record).

Provenance: committed at v2.5.2 close from the scratchpad driver that produced the phase_j_report.md /
ledger J-arc numbers. Launch (whole GPU, no guards; wave-1 exited): `PYTHONPATH=. python
scripts/analysis/j_arc_run.py`. The `SP` output path below is the original session scratchpad (retained
verbatim as historical provenance; redirect it if re-running)."""
import json, time
from pathlib import Path

import numpy as np
import torch

from src.common.maneuver_value import build_safety_h_fn
from src.eval.build_pools import load_pool
from src.eval.dual_arm import dual_arm_eval
from src.frameworks.jt_pncbf.train import load_framework_from_checkpoint

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
SP = Path("/tmp/claude-1000/-home-junhyeok-MIT-jt-pncbf/31d93785-ac11-4206-bf50-a4c3de145dff/scratchpad")
POOL = REPO / "data/secured_data/pools/eval_full_di_n2000_seed23456.pkl"
ADOPTED = {42: "step_040500.pt", 99: "step_045000.pt", 12345: "step_049500.pt"}
OV = {"safety_channel": {"type": "value"}, "filter": {"shield": {"dt_check": 0.01}}}

dev = torch.device("cuda")
scenes = load_pool(POOL).scenes
print(f"pool n={len(scenes)}", flush=True)

per_seed = {}
for s, fname in ADOPTED.items():
    t0 = time.time()
    ck_path = REPO / f"data/secured_data/v2.3.0/seed{s}/checkpoints/{fname}"
    fw, cfg, ck = load_framework_from_checkpoint(ck_path, config_overrides=OV)
    system = fw.system
    system.u_bounds = system.u_bounds.to(device=dev, dtype=torch.float32)
    pol = fw.policy_net.to(device=dev, dtype=torch.float32).eval()
    vnet = fw.value_net.to(device=dev, dtype=torch.float32).eval()
    h_fn = build_safety_h_fn(system, cfg, vnet)                 # value channel = ckpt ValueNet
    res = dual_arm_eval(scenes, pol, cfg, system, dev, arms=("A", "B", "Bprime"), chunk=250, h_fn=h_fn)
    res["_meta"] = {"seed": s, "ckpt": str(ck_path.relative_to(REPO)), "step": ck.get("step"),
                    "thresh": float(cfg.get("filter", {}).get("shield", {}).get("dt_check", 0.01)) * 2.5 / 2.0,
                    "wall_s": round(time.time() - t0, 1)}
    per_seed[s] = res
    A, B, Bp = res["arm_A"], res["arm_B"], res["arm_Bprime"]
    print(f"seed {s} [{fname}] {res['_meta']['wall_s']}s: "
          f"A cps_v2={A['cps_v2']:.4f} legacy={A['cps_legacy']:.4f} reach={A['reach']:.4f} coll={A['collision']:.4f} "
          f"inf_empty={A['inf_empty']:.4f} inf_sv={A['inf_singular_violated']:.4f} | "
          f"B cps_v2={B['cps_v2']:.4f} vstart_coll={B['verified_start_collisions']} ovr/ep={B['overrides_per_ep']:.3f} | "
          f"B' cps_v2={Bp['cps_v2']:.4f} vstart_coll={Bp['verified_start_collisions']}", flush=True)

# Pooled = per-seed mean (v2.3.0 aggregate method)
def mean(arm, key):
    return float(np.mean([per_seed[s][arm][key] for s in ADOPTED]))

pooled = {}
for arm in ("arm_A", "arm_B", "arm_Bprime"):
    pooled[arm] = {k: mean(arm, k) for k in
                   ("cps_v2", "cps_legacy", "reach", "collision", "oob", "stuck", "timeout",
                    "inf_empty", "inf_singular_violated", "inf_v2", "inf_canonical")
                   if k in per_seed[42][arm]}
# arm A infeasibility split; arms B/B' shield diagnostics
for arm in ("arm_B", "arm_Bprime"):
    for k in ("verified_start_frac", "overrides_per_ep", "checks_per_ep", "shield_overhead_x"):
        pooled[arm][k] = mean(arm, k)
    pooled[arm]["verified_start_collisions_total"] = int(sum(per_seed[s][arm]["verified_start_collisions"] for s in ADOPTED))

# ---- GATES ----
# P-J0 gate is on cps_legacy (per-seed cps_v2 of record not fully tabulated in secured_data; amendment §2).
# Per-seed cps_legacy references of record (v2.3.0 aggregate per_seed.cps):
CPS_LEG_REF = {42: 0.870, 99: 0.8628, 12345: 0.8766}
A2 = pooled["arm_A"]["cps_v2"]; Aleg = pooled["arm_A"]["cps_legacy"]
per_seed_legacy = {s: per_seed[s]["arm_A"]["cps_legacy"] for s in ADOPTED}
per_seed_v2 = {s: per_seed[s]["arm_A"]["cps_v2"] for s in ADOPTED}
pj0_perseed = {s: abs(per_seed_legacy[s] - CPS_LEG_REF[s]) <= 0.010 for s in ADOPTED}
pj0_pooled_legacy = abs(Aleg - 0.8698) <= 0.010
pj0 = all(pj0_perseed.values()) and pj0_pooled_legacy
pj0_v2_corrob = abs(A2 - 0.9005) <= 0.010            # reported for the record, NOT gating
vstart_B = pooled["arm_B"]["verified_start_collisions_total"]
vstart_Bp = pooled["arm_Bprime"]["verified_start_collisions_total"]
pj1 = (vstart_B == 0) and (vstart_Bp == 0)
B2 = pooled["arm_B"]["cps_v2"]
pj2 = B2 > 0.8422
reach_drop_ok = pooled["arm_B"]["reach"] >= pooled["arm_A"]["reach"] - 0.05
drop = A2 - B2
attrib = (pooled["arm_B"]["stuck"] - pooled["arm_A"]["stuck"]) + (pooled["arm_B"]["timeout"] - pooled["arm_A"]["timeout"])
pj3 = reach_drop_ok and (drop <= 0 or attrib >= 0.60 * drop)
pj4 = pooled["arm_B"]["overrides_per_ep"] > 1.0

print("\n=== POOLED (per-seed mean) ===", flush=True)
print(f"arm A: cps_v2={A2:.4f} cps_legacy={Aleg:.4f} reach={pooled['arm_A']['reach']:.4f} coll={pooled['arm_A']['collision']:.4f}", flush=True)
print(f"arm B: cps_v2={B2:.4f} reach={pooled['arm_B']['reach']:.4f} vstart_coll_total={vstart_B} ovr/ep={pooled['arm_B']['overrides_per_ep']:.3f}", flush=True)
print(f"arm B': cps_v2={pooled['arm_Bprime']['cps_v2']:.4f} vstart_coll_total={vstart_Bp} ovr/ep={pooled['arm_Bprime']['overrides_per_ep']:.3f}", flush=True)
print("\n=== GATES ===", flush=True)
print("P-J0 [GATE] on cps_legacy (per-seed + pooled); cps_v2 reported for record:", flush=True)
for s in ADOPTED:
    print(f"    seed {s}: legacy {per_seed_legacy[s]:.4f} ~ ref {CPS_LEG_REF[s]} (d={per_seed_legacy[s]-CPS_LEG_REF[s]:+.4f}) {'PASS' if pj0_perseed[s] else 'FAIL'} | cps_v2={per_seed_v2[s]:.4f}", flush=True)
print(f"    pooled: legacy {Aleg:.4f} ~ 0.8698 (d={Aleg-0.8698:+.4f}) {'PASS' if pj0_pooled_legacy else 'FAIL'} | cps_v2 {A2:.4f} vs 0.9005 (d={A2-0.9005:+.4f}) {'corroborates' if pj0_v2_corrob else 'NOTE-divergent'}", flush=True)
print(f"P-J0 OVERALL: {'PASS' if pj0 else 'FAIL -> HALT (value channel broken, comparison void)'}", flush=True)
print(f"P-J1 [HALT] verified-start collisions B={vstart_B} B'={vstart_Bp}: {'PASS' if pj1 else 'VIOLATED -> HALT+DUMP'}", flush=True)
print(f"P-J2 arm-B cps_v2 {B2:.4f} > 0.8422: {'PASS' if pj2 else 'FALSIFIED'}", flush=True)
print(f"P-J3 reach>=A-0.05 ({pooled['arm_B']['reach']:.4f}>={pooled['arm_A']['reach']-0.05:.4f}) & >=60% drop via stuck+to (drop={drop:.4f}, attrib={attrib:.4f}): {'PASS' if pj3 else 'FALSIFIED'}", flush=True)
print(f"P-J4 arm-B ovr/ep {pooled['arm_B']['overrides_per_ep']:.3f} > 1.0: {'PASS(high tax)' if pj4 else 'FALSIFIED(low override)'}", flush=True)

out = {"per_seed": {str(s): per_seed[s] for s in per_seed}, "pooled": pooled,
       "gates": {"P_J0": pj0, "P_J1": pj1, "P_J2": pj2, "P_J3": pj3, "P_J4": pj4},
       "pj0_detail": {"per_seed_legacy": {str(s): per_seed_legacy[s] for s in ADOPTED},
                      "per_seed_legacy_pass": {str(s): pj0_perseed[s] for s in ADOPTED},
                      "per_seed_cps_v2": {str(s): per_seed_v2[s] for s in ADOPTED},
                      "pooled_legacy": Aleg, "pooled_cps_v2": A2, "cps_v2_corroborates_0.9005": pj0_v2_corrob},
       "refs": {"pj0_cps_v2": 0.9005, "pj0_cps_legacy": 0.8698, "cps_leg_ref": CPS_LEG_REF,
                "pj2_bar": 0.8422, "pj4_v251_ovr": 0.1995}}
json.dump(out, open(SP / "j_arc_result.json", "w"), indent=2, default=str)
print("\nDONE" if pj0 and pj1 else ("\nHALT: P-J0 FAILED" if not pj0 else "\nHALT: P-J1 VIOLATED"), flush=True)
