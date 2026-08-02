"""v2.8.0 M2 gates for env.band_terminates (both HALT on failure).

Gate 1 (inert when on): with band_terminates=true (default), re-evaluating the dual deliverable reproduces
the recorded m4_dual row bit-for-bit (cps/reach/collision + obstacle/floor/ceiling decomposition).
Gate 2 (obstacle-preserving when off): rolling the same scenes under band_terminates true vs false, the
obstacle channel is bit-identical at every step up to the first state divergence (= the first band
crossing, where the terminating run stops and the permissive run continues). Defined only on the common
prefix; stated as such. Sidecar: data/runs/v2.8.0/m2_gates.json"""
from __future__ import annotations
import copy, json
from pathlib import Path
import numpy as np, torch

from src.eval.evaluate import evaluate
from src.eval.run_full import _load_framework as load_fw
from src.eval.build_pools import load_pool
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.common.rk4 import rk4_step
from src.common.outcomes import step_outcomes

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
CK = REPO / "data/runs/v2.8.0/set__20260731-015517__seed42/v2.8.0__jt__20260731-015517__seed42/checkpoints/best.pt"
POOL = REPO / "data/secured_data/pools/eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl"
OUT = REPO / "data/runs/v2.8.0"; OUT.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REC = json.loads((REPO / "data/runs/v2.8.0/s3_eval/m4_dual.json").read_text())


def build(band_terminates):
    ck = torch.load(str(CK), map_location="cpu", weights_only=False)
    filt = copy.deepcopy(ck["config"]["filter"]); filt["empty_fallback"] = {"mode": "kstep", "phases": 1, "k": 3}
    filt["projection"] = "dual_solve"
    over = {"env": {"dt": 0.05, "stuck_window_steps": 60, "stuck_radius": 0.10, "band_collision_limit": 4.0,
                    "goal_angrate_radius": 0.48, "band_terminates": band_terminates},
            "eval": {"max_steps": 200, "dt_ctrl": 0.05}, "filter": filt}
    fw, cfg, ck2 = load_fw(str(CK), config_overrides=over)
    for n in ("value_net", "policy_net"):
        m = getattr(fw, n, None); m.to(DEV) if m is not None else None
    return fw, cfg, ck2


rep = {}
# ---- Gate 1: inert when on ----
fw, cfg, ck2 = build(True)
res = evaluate(fw, POOL, cfg, mode="final", step=int(ck2["step"]), ckpt_name="g1", max_scenes=None, include_lqr_baseline=False)
r = res.eval_row; cause = np.array([e.get("collision_cause", "") for e in res.episode_rows])
dec = {c: float((cause == c).mean()) for c in ("obstacle", "band_lower", "band_upper")}
g1 = {"cps": float(r["cps"]), "reach": float(r["reach"]), "collision": float(r["collision"]),
      "obstacle": dec["obstacle"], "band_lower": dec["band_lower"], "band_upper": dec["band_upper"],
      "recorded": {"cps": REC["outcome"]["cps"], "reach": REC["outcome"]["reach"], "collision": REC["outcome"]["collision"],
                   "obstacle": REC["collision_obstacle"], "band_lower": REC["collision_band_lower"], "band_upper": REC["collision_band_upper"]}}
g1["pass"] = (abs(g1["cps"] - REC["outcome"]["cps"]) < 1e-9 and abs(g1["collision"] - REC["outcome"]["collision"]) < 1e-9
              and abs(g1["reach"] - REC["outcome"]["reach"]) < 1e-9 and abs(g1["obstacle"] - REC["collision_obstacle"]) < 1e-9
              and abs(g1["band_lower"] - REC["collision_band_lower"]) < 1e-9 and abs(g1["band_upper"] - REC["collision_band_upper"]) < 1e-9)
rep["gate1_inert_when_on"] = g1
print(f"GATE1 inert-when-on: cps {g1['cps']:.4f} vs rec {REC['outcome']['cps']:.4f} | coll {g1['collision']:.4f} "
      f"decomp {g1['obstacle']:.4f}/{g1['band_lower']:.4f}/{g1['band_upper']:.4f} -> {'PASS' if g1['pass'] else 'FAIL'}", flush=True)

# ---- Gate 2: obstacle-preserving when off (common prefix) ----
NSC = 400
scenes = load_pool(POOL).scenes[:NSC]
def roll(band_terminates):
    fw, cfg, _ = build(band_terminates); system = fw.system
    bscene = batch_scenes(scenes, device=DEV, dtype=torch.float32); x = initial_states_from_batch(bscene)
    S = [x]
    with torch.no_grad():
        for _ in range(200):
            un = fw.policy(x, bscene); u = fw.filter(x, un, bscene)[0]
            x = system.wrap_state(rk4_step(system, x, u, 0.05)); S.append(x)
    S = torch.stack(S, 0)
    m = step_outcomes(S, bscene, system, cfg)
    return S, m.collided_obstacle.cpu().numpy(), (m.collided_band_lower | m.collided_band_upper).cpu().numpy()
S_t, obst_t, band_t = roll(True)
S_f, obst_f, band_f = roll(False)
# first state divergence per episode
diff = (S_t - S_f).abs().amax(dim=-1).cpu().numpy()          # [T+1,B]
T1, B = obst_t.shape
first_div = np.full(B, T1)
for b in range(B):
    nz = np.nonzero(diff[:, b] > 1e-9)[0]
    if nz.size:
        first_div[b] = nz[0]
# compare obstacle on common prefix [0, first_div)
mismatches = 0; n_cross = 0
for b in range(B):
    p = first_div[b]
    mismatches += int((obst_t[:p, b] != obst_f[:p, b]).sum())
    if band_f[:, b].any():
        n_cross += 1
g2 = {"n_scenes": NSC, "n_episodes_with_band_crossing": int(n_cross),
      "obstacle_mismatches_on_common_prefix": int(mismatches),
      "median_first_divergence_step_crossing_eps": float(np.median([first_div[b] for b in range(B) if band_f[:, b].any()])) if n_cross else None,
      "pass": mismatches == 0}
rep["gate2_obstacle_preserving_when_off"] = g2
rep["gate2_note"] = ("comparison defined only on the common prefix [0, first state divergence); after a band "
                     "crossing the permissive run continues where the terminating run stopped, so the trajectories "
                     "legitimately diverge and the obstacle channel is not comparable there.")
print(f"GATE2 obstacle-preserving-when-off: {n_cross}/{NSC} episodes cross the band; obstacle mismatches on "
      f"common prefix = {mismatches} -> {'PASS' if g2['pass'] else 'FAIL'}", flush=True)

(OUT / "m2_gates.json").write_text(json.dumps(rep, indent=2) + "\n")
if not (g1["pass"] and g2["pass"]):
    raise SystemExit(f"HALT: M2 gate failed (gate1={g1['pass']} gate2={g2['pass']})")
print("M2 GATES: BOTH PASS")
