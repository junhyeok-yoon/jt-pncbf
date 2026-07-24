"""v2.7.4 theory G5 — during the fall, is the policy trying to arrest and the filter overriding, or is the
policy not trying?

PRECONDITION (checked, not assumed): u_nom in eval_action_stream.npz must be the LEARNED policy's pre-filter
output, not the cascaded-PD nominal. The checkpoint's framework is jt_pncbf and
JTPNCBFFramework.policy(x, scene) = self.policy_net(self._policy_obs(x, scene)) — the learned net. (The PD
nominal is OCPNCBFFramework.policy = system.lqr_action, i.e. the M4 pre-JT arm, not this run.) The script
re-asserts this and stops if the framework is not jt_pncbf.

Actions are read from the persisted artifact as specified. The 33 floor-exit ids and their t_4 come from
G3's per-episode records. The 217 basement-entering goal ids were NOT persisted by G4, and the action stream
carries no per-step empty/intervention flags, so the same deterministic canonical re-roll used by G1/G3/G4 is
repeated to recover those ids, their t_4, and the per-step flags. The re-rolled actions are cross-checked
against the npz. Eval-only.
"""
from __future__ import annotations

import csv, json
from pathlib import Path

import numpy as np
import torch

from src.eval.build_pools import load_pool, EVAL_POOLS_DIR
from src.eval.evaluate import _filter_adapter, _tensor_options, make_batched_scene
from src.eval.rollout import rollout_eval
from src.envs.scene_batch import initial_states_from_batch
from src.eval.run_full import _load_framework

RUN = Path("data/runs/v2.7.4/set__20260720-083533__seed42/v2.7.4__20260720-091830__seed42")
CKPT = RUN / "checkpoints/best.pt"
POOL = EVAL_POOLS_DIR / "eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl"
OUT = Path("data/runs/v2.7.4/theory"); OUT.mkdir(parents=True, exist_ok=True)
TOL = 1.0e-3                     # codebase intervention tolerance

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
if str(ck.get("framework")) != "jt_pncbf":
    raise SystemExit(f"STOP: framework is {ck.get('framework')}, so u_nom is not the learned policy output.")
fw, cfg, _ = _load_framework(CKPT)
system = fw.system
dt = float(cfg["env"]["dt"]); max_steps = int(cfg["eval"]["max_steps"])
world_lim = float(cfg["env"]["world_lim"]); z_scene = -world_lim
oob_limit = float(cfg["env"]["oob_limit"]); z_floor = -oob_limit
Mx = system.mixer.to(torch.float64).numpy()                 # rotor -> (f_thr, tau_x, tau_y, tau_z)
F_TOTAL_MAX = 4.0 * float(system.f_rotor_max)               # 19.62 N

z_np = np.load(RUN / "eval_action_stream.npz", allow_pickle=True)
U_NOM = z_np["u_nom"].astype(np.float64)                    # [E,T,4] learned-policy proposal
U_CMD = z_np["u_cmd"].astype(np.float64)                    # [E,T,4] filter output
NSTEP = z_np["n_steps"].astype(int)

scenes = load_pool(POOL).scenes
dtype, device = _tensor_options(system, fw)
bs = make_batched_scene(scenes, device=device, dtype=dtype)
res = rollout_eval(system, fw.policy, _filter_adapter(fw), bs, initial_states_from_batch(bs),
                   max_steps=max_steps, dt=dt, config=cfg)
S = res.states.detach().to(torch.float64).numpy()
EMPTY = (res.empty.detach().cpu().numpy().astype(bool) if res.empty is not None
         else np.zeros((max_steps, S.shape[1]), bool))      # [T,E]
INTERV = res.intervention_mask.detach().cpu().numpy().astype(bool)
U_NOM_RR = res.u_nom.detach().to(torch.float64).numpy()     # [T,E,4] for the cross-check
T1, N = S.shape[0], S.shape[1]
z = S[:, :, 2]

outc, ev = {}, {}
with (RUN / "eval_episodes.csv").open() as f:
    for r in csv.DictReader(f):
        if r["mode"] == "final":
            i = int(r["episode_idx"]); outc[i] = r["outcome"]; ev[i] = int(float(r["n_steps"]))

# cross-check: persisted npz u_nom vs re-rolled u_nom over active steps
diffs = []
for i in range(0, N, 97):
    k = max(1, min(int(NSTEP[i]), T1 - 1))
    diffs.append(float(np.abs(U_NOM[i, :k] - U_NOM_RR[:k, i]).max()))
xcheck = {"n_sampled_episodes": len(diffs), "max_abs_diff_u_nom": round(float(max(diffs)), 12)}

# groups
g3 = json.loads((OUT / "g3_where_altitude_lost.json").read_text())
floor = {int(p["ep"]): int(p["t4_step"]) for p in g3["per_episode"] if p.get("t4_exists")}
ctl = {}
for i in range(N):
    if outc.get(i) != "goal":
        continue
    k_end = int(np.clip(ev[i], 1, T1 - 1))
    b = np.nonzero(z[:k_end + 1, i] < z_scene)[0]
    if b.size:
        ctl[i] = int(b[0])

def q(a, p): return round(float(np.percentile(a, p)), 4) if len(a) else None
def dist(a):
    a = np.asarray(a, float)
    return {"n": int(a.size), "median": q(a, 50), "iqr": [q(a, 25), q(a, 75)]} if a.size else {"n": 0}

def measure(group, label):
    thr_n, thr_c, ovr, dthr, rp_n, rp_c, iv_f, em_f, frac_max_n, frac_max_c = ([] for _ in range(10))
    n_steps_total = 0
    for i, k4 in group.items():
        kend = max(1, min(int(NSTEP[i]), T1 - 1))
        if kend <= k4:                       # transit window must contain at least one step
            continue
        un = U_NOM[i, k4:kend]; uc = U_CMD[i, k4:kend]      # [w,4]
        n_steps_total += un.shape[0]
        tn = un.sum(axis=1); tc = uc.sum(axis=1)            # total commanded thrust
        thr_n.append(np.median(tn)); thr_c.append(np.median(tc))
        frac_max_n.append(np.median(tn) / F_TOTAL_MAX); frac_max_c.append(np.median(tc) / F_TOTAL_MAX)
        ovr.append(np.median(np.linalg.norm(uc - un, axis=1)))
        dthr.append(np.median(tc - tn))
        wn = un @ Mx.T; wc = uc @ Mx.T                       # wrench (f_thr, tau_x, tau_y, tau_z)
        rp_n.append(np.median(np.hypot(wn[:, 1], wn[:, 2])))
        rp_c.append(np.median(np.hypot(wc[:, 1], wc[:, 2])))
        iv_f.append(float(INTERV[k4:kend, i].mean()))
        em_f.append(float(EMPTY[k4:kend, i].mean()))
    return {"group": label, "n_episodes": len(thr_n), "n_transit_steps": n_steps_total,
            "total_thrust_u_nom_N": dist(thr_n), "total_thrust_u_cmd_N": dist(thr_c),
            "frac_of_max_19.62_u_nom": dist(frac_max_n), "frac_of_max_19.62_u_cmd": dist(frac_max_c),
            "override_norm_u_cmd_minus_u_nom": dist(ovr),
            "signed_delta_total_thrust_cmd_minus_nom_N": dist(dthr),
            "rollpitch_torque_u_nom_Nm": dist(rp_n), "rollpitch_torque_u_cmd_Nm": dist(rp_c),
            "intervention_fraction_of_transit_steps": dist(iv_f),
            "empty_branch_fraction_of_transit_steps": dist(em_f)}

rep = {"precondition": {"framework": str(ck.get("framework")),
        "u_nom_is": "LEARNED policy output (JTPNCBFFramework.policy -> policy_net); NOT the cascaded-PD nominal",
        "npz_note": str(z_np["note"])},
       "gap_named": ("G4 did not persist the 217 control ids, and eval_action_stream.npz carries no per-step "
                     "empty/intervention flags; both were recovered by the same deterministic canonical re-roll "
                     "used by G1/G3/G4. Actions are taken from the persisted npz as specified."),
       "npz_vs_reroll_u_nom_crosscheck": xcheck,
       "constants": {"f_total_max_N": F_TOTAL_MAX, "z_scene_floor": z_scene, "z_arena_floor": z_floor,
                     "intervention_tolerance": TOL, "dt": dt},
       "window": "per episode, [t_4, n_steps) — the basement transit",
       "floor_exits": measure(floor, "33 floor exits"),
       "controls": measure(ctl, "basement-entering goal-reaching")}
(OUT / "g5_policy_vs_filter.json").write_text(json.dumps(rep, indent=2) + "\n")
print(json.dumps(rep, indent=2))
