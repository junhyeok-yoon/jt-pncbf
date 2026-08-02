"""v2.8.0 Phase-2 C1 — one instrumented rollout pass (feeds D1/D2/D3/D5).

Runs the dual-deliverable checkpoint on the canonical pool at one control rate x one filter.projection, and
persists per control step: u_nom, u_cmd, a=L_g h, b=row_upper, the box-clipped coordinate set, the empty and
singular branch flags, and the top-k obstacle index list. Seconds are held: max_steps=10s/dt_sim,
stuck_window_steps=3s/dt_sim, empty-fallback k=0.15s/dt_sim; every rescaled value is read back from the WRITTEN
effective config and HALTs if it does not match (an unrescaled step-denominated parameter invalidates the arm).
Everything in C2/C3/C4/C6 is computed from these files with no further GPU.

Usage: --proj {dual_solve,enumerate} --rate {20,100,500}
Sidecar: data/runs/v2.8.0/c1/stream_<rate>hz_<proj>.npz  (+ meta_<rate>hz_<proj>.json)
"""
from __future__ import annotations
import argparse, copy, json
from pathlib import Path
import numpy as np, torch

from src.eval.evaluate import first_physical_event_step
from src.eval.run_full import _load_framework as load_fw
from src.eval.build_pools import load_pool
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.common.rk4 import rk4_step
from src.common.outcomes import step_outcomes
from src.common.observation import scene_obstacle_tensors, top_k_obstacles

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
CK = REPO / "data/runs/v2.8.0/set__20260731-015517__seed42/v2.8.0__jt__20260731-015517__seed42/checkpoints/best.pt"
POOL = REPO / "data/secured_data/pools/eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl"
OUT = REPO / "data/runs/v2.8.0/c1"; OUT.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RATE_ARMS = {20: (0.05, 0.05), 100: (0.01, 0.01), 500: (0.002, 0.002)}  # rate -> (dt_sim, dt_ctrl)

ap = argparse.ArgumentParser()
ap.add_argument("--proj", required=True, choices=["dual_solve", "enumerate"])
ap.add_argument("--rate", required=True, type=int, choices=list(RATE_ARMS))
a = ap.parse_args()
dt_sim, dt_ctrl = RATE_ARMS[a.rate]
max_steps = int(round(10.0 / dt_sim)); stuck_w = int(round(3.0 / dt_sim)); kfb = int(round(0.15 / dt_sim))
substeps = int(round(dt_ctrl / dt_sim))

ck = torch.load(str(CK), map_location="cpu", weights_only=False)
filt = copy.deepcopy(ck["config"]["filter"])
filt["empty_fallback"] = {"mode": "kstep", "phases": 1, "k": kfb}; filt["projection"] = a.proj
over = {"env": {"dt": dt_sim, "stuck_window_steps": stuck_w, "stuck_radius": 0.10, "band_collision_limit": 4.0,
                "goal_angrate_radius": 0.48},
        "eval": {"max_steps": max_steps, "dt_ctrl": dt_ctrl}, "filter": filt}
fw, cfg, ck2 = load_fw(str(CK), config_overrides=over)
for n in ("value_net", "policy_net"):
    m = getattr(fw, n, None)
    if m is not None:
        m.to(DEV)

# ---- verify the rescaled parameters from the WRITTEN effective config (HALT if any drift) ----
verify = {"dt_sim": float(cfg["env"]["dt"]), "dt_ctrl": float(cfg["eval"]["dt_ctrl"]),
          "max_steps": int(cfg["eval"]["max_steps"]), "stuck_window_steps": int(cfg["env"]["stuck_window_steps"]),
          "fallback_k": int(fw._filter.params.empty_fallback_k), "filter_dt": float(fw._filter.params.dt),
          "substeps": substeps, "projection": str(fw._filter.params.projection)}
expect = {"dt_sim": dt_sim, "dt_ctrl": dt_ctrl, "max_steps": max_steps, "stuck_window_steps": stuck_w,
          "fallback_k": kfb, "filter_dt": dt_sim, "substeps": substeps, "projection": a.proj}
verify["all_match_expected"] = all((abs(verify[k] - expect[k]) < 1e-9) if isinstance(expect[k], (int, float))
                                   else (verify[k] == expect[k]) for k in expect)
print(f"[{a.rate}Hz/{a.proj}] VERIFY {verify}", flush=True)
if not verify["all_match_expected"]:
    raise SystemExit(f"HALT: rescaled config not written as expected: {verify} vs {expect}")

# ---- instrumented ZOH rollout ----
system = fw.system
scenes = load_pool(POOL).scenes
bscene = batch_scenes(scenes, device=DEV, dtype=torch.float32)
x = initial_states_from_batch(bscene); B = x.shape[0]
k_obs = int(system.k_obs)
ub = system.u_bounds.to(DEV)                                   # [A,2]
lo = ub[:, 0].reshape(1, -1); hi = ub[:, 1].reshape(1, -1)
BTOL = 1e-6

states = [x]
cap = {k: [] for k in ("u_nom", "u_cmd", "a", "b", "clipped", "empty", "singular", "topk")}
held = {}
with torch.no_grad():
    for i in range(max_steps):
        if i % substeps == 0:
            u_nom = fw.policy(x, bscene)
            u_safe, _ = fw.filter(x, u_nom, bscene)
            f = fw._filter
            a_row = f.last_a.to(DEV); b_row = f.last_b.reshape(-1).to(DEV)
            boxp = f.last_box_projected.to(DEV)
            clipped = (boxp <= lo + BTOL) | (boxp >= hi - BTOL)
            le = getattr(f, "last_empty", None)
            ls = getattr(f, "last_singular", None)
            empty = le.to(DEV).bool() if le is not None else torch.zeros(B, dtype=torch.bool, device=DEV)
            singular = ls.to(DEV).bool() if ls is not None else torch.zeros(B, dtype=torch.bool, device=DEV)
            # top-k obstacle indices for the current state (exact encoder ordering)
            p_xy = x[:, :2]
            centers, radii, active = scene_obstacle_tensors(bscene, x.device, x.dtype)
            _, _, topk = top_k_obstacles(p_xy, centers[..., :2], radii, active, k_obs, return_indices=True)
            held = {"u_nom": u_nom.detach(), "u_cmd": u_safe.detach(), "a": a_row, "b": b_row,
                    "clipped": clipped, "empty": empty, "singular": singular, "topk": topk.to(torch.int32)}
        for key in cap:
            cap[key].append(held[key].cpu().numpy())
        x = system.wrap_state(rk4_step(system, x, held["u_cmd"], dt_sim))
        states.append(x)

S = torch.stack(states, 0)                                    # [T+1,B,D]
# step_outcomes' window_displacement is O((T-W)*(W+1)*B); at 500 Hz (T=5000, W=1500) the full batch is 117 GiB,
# so resolve the per-episode done step in episode chunks (identical result; only memory changes).
from src.common.kstep_fallback import slice_scene
Bc = 100
done_parts = []
for s0 in range(0, B, Bc):
    s1 = min(s0 + Bc, B)
    msk = torch.zeros(B, dtype=torch.bool, device=DEV); msk[s0:s1] = True
    sub_masks = step_outcomes(S[:, s0:s1], slice_scene(bscene, msk), system, cfg)
    done_parts.append(first_physical_event_step(sub_masks).cpu().numpy())
done_np = np.concatenate(done_parts)
active = np.where(done_np < 0, max_steps, done_np).astype(int)  # control steps taken per episode

# stack captures to [T,B,...] and truncate per-episode to active length -> flat arrays
arr = {k: np.stack(v, 0) for k, v in cap.items()}             # each [T,B,...]
flat = {k: [] for k in arr}
ep_id, t_in_ep = [], []
offsets = [0]
for b in range(B):
    n = int(active[b])
    if n <= 0:
        offsets.append(offsets[-1]); continue
    for k in arr:
        flat[k].append(arr[k][:n, b])
    ep_id.append(np.full(n, b, dtype=np.int32)); t_in_ep.append(np.arange(n, dtype=np.int32))
    offsets.append(offsets[-1] + n)
flat = {k: (np.concatenate(v, 0) if v else np.empty((0,) + arr[k].shape[2:], dtype=arr[k].dtype)) for k, v in flat.items()}
ep_id = np.concatenate(ep_id) if ep_id else np.empty(0, np.int32)
t_in_ep = np.concatenate(t_in_ep) if t_in_ep else np.empty(0, np.int32)

np.savez_compressed(
    OUT / f"stream_{a.rate}hz_{a.proj}.npz",
    u_nom=flat["u_nom"].astype(np.float32), u_cmd=flat["u_cmd"].astype(np.float32),
    a=flat["a"].astype(np.float32), b=flat["b"].astype(np.float32),
    clipped=flat["clipped"].astype(bool), empty=flat["empty"].astype(bool),
    singular=flat["singular"].astype(bool), topk=flat["topk"].astype(np.int32),
    ep_id=ep_id, t_in_ep=t_in_ep, ep_offsets=np.array(offsets, np.int64),
    ep_active=active.astype(np.int64), u_bounds=ub.cpu().numpy().astype(np.float32),
)
meta = {"rate_hz": a.rate, "proj": a.proj, "dt_sim": dt_sim, "dt_ctrl": dt_ctrl, "substeps": substeps,
        "max_steps": max_steps, "stuck_window_steps": stuck_w, "fallback_k": kfb, "k_obs": k_obs,
        "n_episodes": int(B), "n_steps_total": int(flat["u_cmd"].shape[0]),
        "mean_active": float(active.mean()), "verify_config": verify, "pool": POOL.name,
        "empty_step_fraction": float(flat["empty"].mean()) if flat["empty"].size else 0.0,
        "singular_step_fraction": float(flat["singular"].mean()) if flat["singular"].size else 0.0}
(OUT / f"meta_{a.rate}hz_{a.proj}.json").write_text(json.dumps(meta, indent=2) + "\n")
print(f"[{a.rate}Hz/{a.proj}] steps={meta['n_steps_total']} mean_active={meta['mean_active']:.1f} "
      f"empty_frac={meta['empty_step_fraction']:.4f} sing_frac={meta['singular_step_fraction']:.4f} "
      f"-> {OUT / f'stream_{a.rate}hz_{a.proj}.npz'}", flush=True)
