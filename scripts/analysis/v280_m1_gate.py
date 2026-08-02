"""v2.8.0 collision-decomposition M1 reproduction gate (corrected).

Re-evaluate the v2.7.6 headline checkpoint 09c33bf4 (step_042000) on the canonical pool 0ef3751b under the
EXACT recorded config (v2.7.6_results.md:17 / canonical_eval.json jt42000 kstep cell): banded scoring,
empty_fallback kstep **phases 2** k5, enumerate projection, native dt_ctrl. The angular-reach terminal added
in S2 is made inert here (goal_angrate_radius = 1e9) to isolate the collision channel. Assert collision ==
0.0425 (HALT if off) and check the decomposition equals the recorded split (obstacle 20 / band_floor 64 /
band_ceiling 1). Also verify, on the real 2000-episode trajectory, that the new `collided` is bit-identical
to the legacy predicate.

NOTE (correction over the completion pass): the completion pass used empty_fallback **phases 1** (and
dt_ctrl 0.05), which produced collision ~0.0480 and prompted a (now retracted) stuck / S1-filter-drift
account. With the recorded phases-2 config the anchor reproduces bit-for-bit; there is no reproduction
discrepancy. See docs/versions/v2.8.0/collision_decomposition.md. Sidecar: s3_eval/m1_gate.json"""
from __future__ import annotations
import copy, json
from pathlib import Path
import numpy as np, torch

from src.eval.build_pools import load_pool
from src.eval.run_full import _load_framework as load_fw
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.common.rk4 import rk4_step
from src.common.outcomes import step_outcomes, _collided_exact

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
POOL = REPO / "data/secured_data/pools/eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl"
CKPT = REPO / "data/runs/v2.7.6/set__20260725-043415__seed42/v2.7.6__jt__20260725-052127__seed42/checkpoints/step_042000.pt"
OUT = REPO / "data/runs/v2.8.0/s3_eval/m1_gate.json"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BAND, DT, STEPS = 4.0, 0.05, 200
EXPECT = {"collision": 0.0425, "reach": 0.9375, "obstacle": 20, "band_lower": 64, "band_upper": 1}

ck = torch.load(str(CKPT), map_location="cpu", weights_only=False)
filt = copy.deepcopy(ck["config"]["filter"])
filt["empty_fallback"] = {"mode": "kstep", "phases": 2, "k": 5}    # the RECORDED config (phases 2, not 1)
filt["projection"] = "enumerate"                                   # v2.7.6 was pre-S1 finite-candidate
over = {"env": {"dt": DT, "stuck_window_steps": 60, "stuck_radius": 0.10, "band_collision_limit": BAND,
                "goal_angrate_radius": 1e9},                       # inert angular conjunct -> old terminal
        "eval": {"max_steps": STEPS}, "filter": filt}              # native dt_ctrl (do not override)
fw, cfg, ck2 = load_fw(str(CKPT), config_overrides=over)
for n in ("value_net", "policy_net"):
    m = getattr(fw, n, None); m.to(DEV) if m is not None else None

# aggregate reproduction via evaluate()
from src.eval.evaluate import evaluate
res = evaluate(fw, POOL, cfg, mode="final", step=int(ck2["step"]), ckpt_name=CKPT.name,
               max_scenes=None, include_lqr_baseline=False)
r = res.eval_row
cause = np.array([e.get("collision_cause", "") for e in res.episode_rows])
decomp = {c: int((cause == c).sum()) for c in ("obstacle", "band_lower", "band_upper")}
coll = float(r["collision"])
coll_ok = abs(coll - EXPECT["collision"]) < 1e-9
split_ok = all(decomp[c] == EXPECT[c] for c in ("obstacle", "band_lower", "band_upper"))

# real-trajectory bit-identity of new collided vs legacy predicate
bs = batch_scenes(load_pool(POOL).scenes, device=DEV, dtype=torch.float32)
x = initial_states_from_batch(bs); traj = [x]
with torch.no_grad():
    for _ in range(STEPS):
        un = fw.policy(x, bs); u = fw.filter(x, un, bs)[0]
        x = fw.system.wrap_state(rk4_step(fw.system, x, u, DT)); traj.append(x)
S = torch.stack(traj, 0); pos = fw.system.position(S)
new_collided = step_outcomes(S, bs, fw.system, cfg).collided.cpu().numpy()
legacy = (_collided_exact(pos, bs) | (torch.abs(pos[..., 2]) >= BAND)).cpu().numpy()
bit_identical = bool(np.array_equal(new_collided, legacy)); mism = int((new_collided != legacy).sum())

rep = {"ckpt": "09c33bf4 step_042000", "pool": "0ef3751b", "config": "banded, kstep phases2 k5, enumerate, native dt_ctrl",
       "collision": coll, "reach": float(r["reach"]), "decomposition": decomp,
       "expected": EXPECT, "collision_pass": coll_ok, "decomposition_pass": split_ok,
       "real_trajectory_bit_identity": {"collided_equals_legacy": bit_identical, "mismatches": mism},
       "gate_pass": bool(coll_ok and split_ok and bit_identical),
       "note": "phases2 (recorded) reproduces the artifact bit-for-bit; the completion pass's phases1 gave "
               "0.0480 and a since-retracted stuck/S1-drift account."}
OUT.write_text(json.dumps(rep, indent=2) + "\n")
print(f"M1 GATE (phases2 recorded config): collision {coll:.4f} (expect {EXPECT['collision']}) -> {'PASS' if coll_ok else 'FAIL'}")
print(f"  decomposition obstacle {decomp['obstacle']}/band_lower {decomp['band_lower']}/band_upper {decomp['band_upper']} "
      f"(expect 20/64/1) -> {'PASS' if split_ok else 'FAIL'}")
print(f"  reach {float(r['reach']):.4f} (expect 0.9375) | real-trajectory bit-identity {'CONFIRMED' if bit_identical else 'FAILED'} (mism {mism})")
if not (coll_ok and split_ok and bit_identical):
    raise SystemExit(f"HALT: M1 gate did not pass (coll_ok={coll_ok} split_ok={split_ok} bit_identical={bit_identical})")
print("M1 GATE: PASS")
