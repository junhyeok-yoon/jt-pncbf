"""v2.8.1 S1 D3 — crossing control-discontinuity ratio ||Δu|| (the soft encoder's core claim).

Faithful reproduction of the v2.8.0 D3 protocol (wiring_and_chatter Part-C table + v280_c1_rollout/c2346_analysis):
ZOH rollout at a given control rate, per-step u_cmd + hard top-k order captured, each episode TRUNCATED to its
active length via first_physical_event_step (post-termination hover steps excluded — the masking my first pass
lacked). ||Δu|| = ||u_cmd[t]-u_cmd[t-1]|| WITHIN episode; a crossing is a top-k order change t-1->t.
D3 = median(||Δu|| @crossing) / median(||Δu|| @non-crossing). Recorded HARD baseline: 1.85x @20Hz, ~33x @500Hz.
Run for hard (v2.8.0, validates the baseline) and soft (beta=30 JT deliverable). Terminal 0.48 for BOTH (the
baseline methodology; the encoder is the only difference) so the ratio isolates the encoder."""
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
from src.common.kstep_fallback import slice_scene
from src.common.observation import scene_obstacle_tensors, top_k_obstacles

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
POOL = REPO / "data/secured_data/pools/eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl"
OUT = REPO / "data/runs/v2.8.1"; OUT.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOFT = REPO / "data/runs/v2.8.1/set__20260802-093419__seed42/v2.8.1__jt__20260802-112624__seed42/checkpoints/best.pt"
HARD = REPO / "data/runs/v2.8.0/set__20260731-015517__seed42/v2.8.0__jt__20260731-015517__seed42/checkpoints/best.pt"
RATE_DT = {20: 0.05, 100: 0.01, 500: 0.002}
ap = argparse.ArgumentParser()
ap.add_argument("--rate", type=int, required=True, choices=list(RATE_DT))
ap.add_argument("--nsub", type=int, required=True)
a = ap.parse_args()
dt = RATE_DT[a.rate]
max_steps = int(round(10.0 / dt)); stuck_w = int(round(3.0 / dt)); kfb = int(round(0.15 / dt))


def du_ratio(ckpt, tag):
    ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    filt = copy.deepcopy(ck["config"]["filter"])
    filt["empty_fallback"] = {"mode": "kstep", "phases": 1, "k": kfb}; filt["projection"] = "dual_solve"
    over = {"env": {"dt": dt, "stuck_window_steps": stuck_w, "stuck_radius": 0.10, "band_collision_limit": 4.0,
                    "goal_angrate_radius": 0.48}, "eval": {"max_steps": max_steps, "dt_ctrl": dt}, "filter": filt}
    fw, cfg, ck2 = load_fw(str(ckpt), config_overrides=over)
    for n in ("value_net", "policy_net"):
        m = getattr(fw, n, None)
        if m is not None:
            m.to(DEV)
    sysm = fw.system; k = int(sysm.k_obs)
    scenes = load_pool(POOL).scenes[:a.nsub]
    bscene = batch_scenes(scenes, device=DEV, dtype=torch.float32); B = a.nsub
    x = initial_states_from_batch(bscene)
    us, topk, states = [], [], [x]
    with torch.no_grad():
        for _ in range(max_steps):
            u_nom = fw.policy(x, bscene); u_safe, _ = fw.filter(x, u_nom, bscene)
            p_xy = x[:, :2]; c, r, act = scene_obstacle_tensors(bscene, x.device, x.dtype)
            _, _, tk = top_k_obstacles(p_xy, c[..., :2], r, act, k, return_indices=True)
            us.append(u_safe.detach().cpu().numpy()); topk.append(tk.cpu().numpy().astype(np.int32))
            x = sysm.wrap_state(rk4_step(sysm, x, u_safe.detach(), dt)); states.append(x)
    U = np.stack(us, 0); TK = np.stack(topk, 0)                                   # [T,B,·]
    # active length per episode (post-termination steps excluded), chunked like C1
    S = torch.stack(states, 0); Bc = 100; done_parts = []
    for s0 in range(0, B, Bc):
        s1 = min(s0 + Bc, B)
        msk = torch.zeros(B, dtype=torch.bool, device=DEV); msk[s0:s1] = True
        done_parts.append(first_physical_event_step(step_outcomes(S[:, s0:s1], slice_scene(bscene, msk),
                                                                  sysm, cfg)).cpu().numpy())
    done = np.concatenate(done_parts); active = np.where(done < 0, max_steps, done).astype(int)
    dc, dnc = [], []
    for b in range(B):
        n = int(active[b])
        if n < 2:
            continue
        du = np.linalg.norm(U[1:n, b] - U[:n - 1, b], axis=1)                     # within-episode ||Δu||
        cr = ~(TK[1:n, b] == TK[:n - 1, b]).all(axis=1)                           # top-k order change
        dc.append(du[cr]); dnc.append(du[~cr])
    dc = np.concatenate(dc) if dc else np.array([]); dnc = np.concatenate(dnc) if dnc else np.array([])
    mc = float(np.median(dc)) if dc.size else float("nan"); mnc = float(np.median(dnc)) if dnc.size else float("nan")
    ratio = mc / mnc if mnc and mnc > 0 else float("nan")
    cr_rate = dc.size / (dc.size + dnc.size) if (dc.size + dnc.size) else float("nan")
    print(f"[{tag}/{a.rate}Hz] enc={sysm.encoder} beta={getattr(sysm,'soft_beta',0.0)} n_ep={B} "
          f"mean_active={active.mean():.0f} | crossing_rate={cr_rate:.4f} | med||Δu|| cross={mc:.4f} "
          f"non={mnc:.4f} | D3 ratio={ratio:.2f}x", flush=True)
    return {"encoder": sysm.encoder, "beta": float(getattr(sysm, "soft_beta", 0.0)), "rate_hz": a.rate,
            "n_ep": B, "mean_active": float(active.mean()), "crossing_rate": float(cr_rate),
            "med_du_cross": mc, "med_du_non": mnc, "ratio": ratio, "n_cross": int(dc.size), "n_non": int(dnc.size)}


h = du_ratio(HARD, "hard_v2.8.0")
s = du_ratio(SOFT, "soft_beta30")
rec = {"rate_hz": a.rate, "nsub": a.nsub, "hard_v2.8.0": h, "soft_beta30": s,
       "reduction_x": (h["ratio"] / s["ratio"]) if s["ratio"] and s["ratio"] == s["ratio"] and s["ratio"] > 0 else None}
(OUT / f"d3_du_{a.rate}hz.json").write_text(json.dumps(rec, indent=2) + "\n")
print(f"\nD3 @{a.rate}Hz: HARD {h['ratio']:.2f}x  SOFT(beta30) {s['ratio']:.2f}x  "
      f"-> reduction {rec['reduction_x']:.2f}x" if rec["reduction_x"] else f"\nD3 @{a.rate}Hz incomplete")
