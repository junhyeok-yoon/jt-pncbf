"""Backup-only (m_0) certificate labels for CPI iteration-0.

V_raw(x, S) = max_{0<=k<=T_stop} h_raw(x_k) under the deadband brake m_0, integrated with the exact
deployment scheme (rk4_step + DI velocity clamp) on grid dt_vm. h_raw is the UNCLIPPED per-active-obstacle
ramp h_raw = max_i (1 - 2*(dist_i - r_i)/h_scale); inactive obstacles are excluded from the max.

Parity: clip(V_raw, -1, 1) == maneuver_value REFERENCE restricted to {m_0}. The codebase's signed_h is
h = clip(1 - 2*clamp(clearance/h_scale, 0, 1), -1, 1); by the identity clip(1 - 2*clearance/h_scale) =
1 - 2*clamp(clearance/h_scale, 0, 1) and clip<->max commutation, clip(V_raw) equals signed_h's brake value
exactly (proven in tests/test_cpi_label_parity.py). The unclipped V_raw exposes interior gradients that the
clipped barrier saturates away; its zero set is identical to the clipped barrier's (clip preserves sign).

NOTE (PROTOCOL FOLLOW-UP): the design-spec literal h_raw = (r-dist)/h_scale is the 01_env §1.4 ramp, which
differs from the implemented signed_h ramp (different slope and zero location). The parity target is the
implemented maneuver_value, so h_raw here is the unclipped signed_h; see the phase-i0 report.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch

from src.common.rk4 import rk4_step
from src.envs.scene_init import sample_train_scene

Tensor = torch.Tensor
NEG_INF = -1.0e30


def t_stop(config: Mapping[str, Any]) -> int:
    env = config["env"]; u_max = float(env["bounds"]["double_integrator"]["u_max"])
    v_max = float(env["bounds"]["double_integrator"]["v_max"]); dt = float(config["cpi"]["labels"]["dt_vm"])
    return int(math.ceil(v_max / (u_max * dt)))


def label_streams(config: Mapping[str, Any]) -> dict[str, np.random.Generator]:
    """All label-pipeline randomness from one master seed: SeedSequence(master_seed).spawn(4) in the fixed
    order [scene, state, oracle-subset, split]. No other seed constant exists in the cpi code."""
    ms = int(config["cpi"]["labels"]["master_seed"])
    scene_ss, state_ss, oracle_ss, split_ss = np.random.SeedSequence(ms).spawn(4)
    return {"scene": np.random.default_rng(scene_ss), "state": np.random.default_rng(state_ss),
            "oracle": np.random.default_rng(oracle_ss), "split": np.random.default_rng(split_ss)}


def sample_scenes(n: int, rng: np.random.Generator, config: Mapping[str, Any]) -> list[Any]:
    """n training scenes via the 03_train §1 sampler (§1.3 acceptance predicates apply). rng = scene stream."""
    return [sample_train_scene(rng, config, "double_integrator") for _ in range(n)]


def stack_scene_obstacles(scenes: list[Any], device, dtype=torch.float32):
    """Per-scene obstacle tensors: centers [S,K,2], radii [S,K], active [S,K], goal [S,2]."""
    C = torch.as_tensor(np.stack([s.obstacle_centers for s in scenes]), dtype=dtype, device=device)
    R = torch.as_tensor(np.stack([s.obstacle_radii for s in scenes]), dtype=dtype, device=device)
    A = torch.as_tensor(np.stack([s.obstacle_active for s in scenes]), dtype=torch.bool, device=device)
    G = torch.as_tensor(np.stack([np.asarray(s.goal, np.float64) for s in scenes]), dtype=dtype, device=device)
    return C, R, A, G


def h_raw_position(p: Tensor, C: Tensor, R: Tensor, A: Tensor, h_scale: float) -> Tensor:
    """Unclipped per-active-obstacle ramp at positions p [N,2] with per-state obstacles [N,K,*]. -> [N]."""
    dist = torch.linalg.norm(p.unsqueeze(1) - C, dim=-1)                 # [N,K]
    h = 1.0 - 2.0 * (dist - R) / h_scale
    h = torch.where(A, h, torch.full_like(h, NEG_INF))
    return h.max(dim=1).values


def outside_all_obstacles(p: Tensor, C: Tensor, R: Tensor, A: Tensor) -> Tensor:
    """Position strictly outside every active obstacle (dist > r); the sampler acceptance predicate. [N] bool."""
    dist = torch.linalg.norm(p.unsqueeze(1) - C, dim=-1)
    inside = (dist <= R) & A
    return ~inside.any(dim=1)


def m0_value_raw(states: Tensor, C: Tensor, R: Tensor, A: Tensor, system, config, dt_vm: float) -> Tensor:
    """Batched V_raw over the m_0 brake rollout. states [N,4]; per-state obstacles [N,K,*]. -> V_raw [N]."""
    env = config["env"]; h_scale = float(env["h_scale"])
    u_max = float(env["bounds"]["double_integrator"]["u_max"])
    tstop = t_stop(config)
    x = states
    vraw = h_raw_position(x[:, :2], C, R, A, h_scale)
    for _ in range(tstop):
        v = x[:, 2:4]
        u = torch.where(v.abs() > u_max * dt_vm, -u_max * torch.sign(v), -v / dt_vm)   # deadband brake m_0
        x = rk4_step(system, x, u, dt_vm)
        vraw = torch.maximum(vraw, h_raw_position(x[:, :2], C, R, A, h_scale))
    return vraw


def _label_in_batches(states, C, R, A, system, config, dt_vm, batch):
    out = []
    for s0 in range(0, states.shape[0], batch):
        out.append(m0_value_raw(states[s0:s0 + batch], C[s0:s0 + batch], R[s0:s0 + batch],
                                A[s0:s0 + batch], system, config, dt_vm))
    return torch.cat(out) if out else states.new_zeros(0)


def sample_uniform_states(scenes_obs, state_rng, n_per_scene, v_max, world_lim, device, dtype=torch.float32,
                          retry_cap=200):
    """Per scene: n_per_scene positions uniform in [-world_lim,world_lim]^2 rejected until outside all active
    obstacles (cap `retry_cap` then keep last draw); velocity dir~U(S^1), speed~U[0,v_max]. Fully vectorized
    across scenes. Returns pos [S*n,2], vel [S*n,2], scene_id [S*n]."""
    C, R, A, _ = scenes_obs; S = C.shape[0]; N = S * n_per_scene
    sid = torch.arange(S, device=device).repeat_interleave(n_per_scene)
    pos = torch.as_tensor(state_rng.uniform(-world_lim, world_lim, size=(N, 2)), dtype=dtype, device=device)
    bad = ~outside_all_obstacles(pos, C[sid], R[sid], A[sid])
    for _ in range(retry_cap):
        if not bool(bad.any()):
            break
        nb = int(bad.sum())
        pos[bad] = torch.as_tensor(state_rng.uniform(-world_lim, world_lim, size=(nb, 2)), dtype=dtype, device=device)
        bad = ~outside_all_obstacles(pos, C[sid], R[sid], A[sid])
    speed = torch.as_tensor(state_rng.uniform(0.0, v_max, size=N), dtype=dtype, device=device)
    ang = torch.as_tensor(state_rng.uniform(0.0, 2 * math.pi, size=N), dtype=dtype, device=device)
    vel = torch.stack([speed * torch.cos(ang), speed * torch.sin(ang)], dim=1)
    return pos, vel, sid


def sample_boundary_states(pos_u, vel_u, sid_u, vraw_u, scenes_obs, state_rng, config, device, dtype=torch.float32):
    """Boundary refinement: per scene, among uniform states with |V_raw|<=boundary_band in ascending |V_raw|
    order, spawn 2 jittered copies each (pos~N(0,jp^2), vel~N(0,jv^2) rescaled to <=v_max, re-reject jittered
    pos outside obstacles up to 20 redraws) until states_per_scene_boundary or candidates exhausted."""
    lc = config["cpi"]["labels"]; band = float(lc["boundary_band"]); cap = int(lc["states_per_scene_boundary"])
    jp = float(lc["jitter_pos_sigma"]); jv = float(lc["jitter_vel_sigma"])
    v_max = float(config["env"]["bounds"]["double_integrator"]["v_max"])
    C, R, A, _ = scenes_obs; S = C.shape[0]
    sid_np = sid_u.cpu().numpy(); vr_np = vraw_u.cpu().numpy()
    posc, velc, sidc = [], [], []
    for s in range(S):
        idx = np.where(sid_np == s)[0]
        cand = idx[np.abs(vr_np[idx]) <= band]
        cand = cand[np.argsort(np.abs(vr_np[cand]))]                    # ascending |V_raw|
        made = 0
        for ci in cand:
            if made >= cap:
                break
            for _ in range(2):
                if made >= cap:
                    break
                p0 = pos_u[ci].cpu().numpy(); v0 = vel_u[ci].cpu().numpy(); placed = False
                for _try in range(20):
                    pj = p0 + state_rng.normal(0.0, jp, size=2)
                    pt = torch.as_tensor(pj, dtype=dtype, device=device).unsqueeze(0)
                    if bool(outside_all_obstacles(pt, C[s:s + 1], R[s:s + 1], A[s:s + 1])[0]):
                        vj = v0 + state_rng.normal(0.0, jv, size=2)
                        sp = float(np.linalg.norm(vj))
                        if sp > v_max:
                            vj = vj * (v_max / sp)
                        posc.append(pj); velc.append(vj); sidc.append(s); made += 1; placed = True
                        break
                if not placed:
                    break
    if not posc:
        z = torch.zeros(0, 2, dtype=dtype, device=device)
        return z, z, torch.zeros(0, dtype=torch.long, device=device)
    pos = torch.as_tensor(np.stack(posc), dtype=dtype, device=device)
    vel = torch.as_tensor(np.stack(velc), dtype=dtype, device=device)
    sid = torch.as_tensor(np.asarray(sidc), dtype=torch.long, device=device)
    return pos, vel, sid


def split_by_scene(n_scenes: int, fractions, rng: np.random.Generator):
    """Disjoint scene-id partition into train/val/calib/test (rng = split stream). name->set(scene_ids)."""
    perm = rng.permutation(n_scenes)
    f = np.asarray(fractions); cuts = (np.cumsum(f) * n_scenes).round().astype(int)
    names = ["train", "val", "calib", "test"]
    parts = np.split(perm, cuts[:-1])
    return {names[i]: set(parts[i].tolist()) for i in range(4)}


def sha256_bytes(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _batched_observation(system, pos, vel, sid, C, R, A, G, batch):
    """dim-19 DI observation per state; per-state scenes gathered via sid from per-scene tensors. Chunked."""
    out = []
    x = torch.cat([pos, vel], dim=1)
    for s0 in range(0, x.shape[0], batch):
        sl = slice(s0, s0 + batch); si = sid[sl]
        scn = SimpleNamespace(obstacle_centers=C[si], obstacle_radii=R[si], obstacle_active=A[si], goal=G[si])
        out.append(system.observation(x[sl], scn))
    return torch.cat(out)


def build_dataset(config, out_dir: Path, system, device, n_scenes=None, dtype=torch.float32,
                  shard_size=250000):
    """Full label pipeline: sample scenes -> uniform + boundary states -> V_raw -> obs -> split by scene ->
    npz shards + manifest.json (SHA-256, per-split counts + V_raw summary). Returns the manifest dict."""
    lc = config["cpi"]["labels"]; dt_vm = float(lc["dt_vm"]); lb = int(lc["label_batch"])
    v_max = float(config["env"]["bounds"]["double_integrator"]["v_max"]); world = float(config["env"]["world_lim"])
    n_scenes = int(lc["n_scenes"]) if n_scenes is None else int(n_scenes)
    streams = label_streams(config)
    scenes = sample_scenes(n_scenes, streams["scene"], config)
    obs_t = stack_scene_obstacles(scenes, device, dtype); C, R, A, G = obs_t
    state_rng = streams["state"]
    # uniform states + labels
    pu, vu, su = sample_uniform_states(obs_t, state_rng, int(lc["states_per_scene_uniform"]), v_max, world, device, dtype)
    xu = torch.cat([pu, vu], 1)
    vru = _label_in_batches(xu, C[su], R[su], A[su], system, config, dt_vm, lb)
    # boundary states + labels
    pb, vb, sb = sample_boundary_states(pu, vu, su, vru, obs_t, state_rng, config, device, dtype)
    if pb.shape[0]:
        xb = torch.cat([pb, vb], 1); vrb = _label_in_batches(xb, C[sb], R[sb], A[sb], system, config, dt_vm, lb)
    else:
        vrb = vru.new_zeros(0)
    pos = torch.cat([pu, pb], 0); vel = torch.cat([vu, vb], 0); sid = torch.cat([su, sb], 0)
    vraw = torch.cat([vru, vrb], 0)
    kind = torch.cat([torch.zeros(pu.shape[0], dtype=torch.int8), torch.ones(pb.shape[0], dtype=torch.int8)]).to(device)
    assert torch.isfinite(vraw).all(), "non-finite label detected"
    obs = _batched_observation(system, pos, vel, sid, C, R, A, G, lb)
    # split by scene
    parts = split_by_scene(n_scenes, config["cpi"]["split"]["fractions"], streams["split"])
    sid_np = sid.cpu().numpy(); sp2name = np.empty(n_scenes, dtype=object)
    for name, ids in parts.items():
        for i in ids:
            sp2name[i] = name
    split_of = np.array([sp2name[s] for s in sid_np])
    out_dir.mkdir(parents=True, exist_ok=True)
    obs_np = obs.cpu().numpy(); vraw_np = vraw.cpu().numpy(); pos_np = pos.cpu().numpy()
    vel_np = vel.cpu().numpy(); kind_np = kind.cpu().numpy()
    shards = []; summary = {}
    for name in ["train", "val", "calib", "test"]:
        mask = np.where(split_of == name)[0]
        v = vraw_np[mask]
        summary[name] = {"n": int(mask.size), "n_uniform": int((kind_np[mask] == 0).sum()),
                         "n_boundary": int((kind_np[mask] == 1).sum()),
                         "vraw_q": [float(np.percentile(v, p)) for p in (1, 10, 50, 90, 99)] if v.size else [],
                         "frac_vraw_gt0": float((v > 0).mean()) if v.size else 0.0,
                         "frac_absvraw_le_0.125": float((np.abs(v) <= 0.125).mean()) if v.size else 0.0}
        for si, s0 in enumerate(range(0, mask.size, shard_size)):
            idx = mask[s0:s0 + shard_size]; fn = f"{name}_{si:03d}.npz"
            np.savez(out_dir / fn, obs=obs_np[idx], vraw=vraw_np[idx], scene_id=sid_np[idx],
                     pos=pos_np[idx], vel=vel_np[idx], kind=kind_np[idx], state_id=idx.astype(np.int64))
            shards.append({"file": fn, "split": name, "n": int(idx.size), "sha256": sha256_bytes(out_dir / fn)})
    # 10k float64 stop-and-hold check on a random subsample (state stream, continued; master-seed derived)
    sub = state_rng.integers(0, pos.shape[0], size=min(10000, pos.shape[0]))
    stop_max = _stop_check_float64(pos[sub], vel[sub], system, config)
    manifest = {"version": config["run"]["version"], "n_scenes": n_scenes, "n_states": int(pos.shape[0]),
                "dt_vm": dt_vm, "label_batch": lb, "t_stop": t_stop(config),
                "split_scene_counts": {k: len(v) for k, v in parts.items()},
                "summary": summary, "shards": shards,
                "stop_check_float64_max_speed_10k": stop_max,
                "label_dtype": "float32", "h_raw": "1 - 2*(dist-r)/h_scale (unclipped signed_h)"}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def build_oracle(config, out_dir: Path, dataset_dir: Path, system, device, dtype=torch.float32):
    """V_M17 (full 17-member library, REFERENCE path) on n_uniform + n_boundary states drawn from the
    labeled set (oracle stream = SeedSequence(master_seed).spawn(4)[2]). Stores V_M17, V_m0 (= clip(V_raw)),
    pos, vel, and the
    per-state obstacle tensors for the criterion-C audit. Returns a summary dict."""
    from src.common.maneuver_value import maneuver_maxh
    import time
    oc = config["cpi"]["labels"]["oracle_subset"]; nu = int(oc["n_uniform"]); nb = int(oc["n_boundary"])
    lat_js = [1, 2, 3, 4, 5, 6, 7, 8]; dt_vm = float(config["cpi"]["labels"]["dt_vm"])
    man = json.load(open(dataset_dir / "manifest.json"))
    P, V, SID, VR, KIND, STID = [], [], [], [], [], []
    for sh in man["shards"]:
        d = np.load(dataset_dir / sh["file"])
        P.append(d["pos"]); V.append(d["vel"]); SID.append(d["scene_id"]); VR.append(d["vraw"])
        KIND.append(d["kind"]); STID.append(d["state_id"])
    P = np.concatenate(P); V = np.concatenate(V); SID = np.concatenate(SID); VR = np.concatenate(VR)
    KIND = np.concatenate(KIND); STID = np.concatenate(STID)
    streams = label_streams(config); rng = streams["oracle"]
    u_idx = np.where(KIND == 0)[0]; b_idx = np.where(KIND == 1)[0]
    sel_u = rng.choice(u_idx, size=min(nu, u_idx.size), replace=False)
    sel_b = rng.choice(b_idx, size=min(nb, b_idx.size), replace=False)
    sel = np.concatenate([sel_u, sel_b])
    scenes = sample_scenes(int(config["cpi"]["labels"]["n_scenes"]), streams["scene"], config)
    pos = torch.as_tensor(P[sel], dtype=dtype, device=device); vel = torch.as_tensor(V[sel], dtype=dtype, device=device)
    sid = SID[sel]; vraw = VR[sel]
    x = torch.cat([pos, vel], 1)
    vm17 = torch.empty(sel.size, dtype=dtype, device=device)
    t0 = time.time()
    for s in np.unique(sid):
        m = np.where(sid == s)[0]
        mh = maneuver_maxh(x[m], scenes[int(s)], system, config, lateral_js=lat_js, dt_override=dt_vm, fast=False)
        vm17[m] = mh.min(dim=1).values
    wall = time.time() - t0
    # fast-vs-reference max|delta| on a 10k overlap
    ov = rng.choice(sel.size, size=min(10000, sel.size), replace=False); dmax = 0.0
    for s in np.unique(sid[ov]):
        m = ov[sid[ov] == s]
        fastv = maneuver_maxh(x[m], scenes[int(s)], system, config, lateral_js=lat_js, dt_override=dt_vm, fast=True).min(dim=1).values
        dmax = max(dmax, float((fastv - vm17[m]).abs().max()))
    C, R, A, _ = stack_scene_obstacles(scenes, device, dtype)
    sidt = torch.as_tensor(sid, dtype=torch.long, device=device)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "oracle.npz", state_id=STID[sel], scene_id=sid, pos=P[sel], vel=V[sel],
             vm17=vm17.cpu().numpy(), vm0=np.clip(vraw, -1, 1), vraw=vraw, kind=KIND[sel],
             centers=C[sidt].cpu().numpy(), radii=R[sidt].cpu().numpy(), active=A[sidt].cpu().numpy())
    summ = {"n_uniform": int(sel_u.size), "n_boundary": int(sel_b.size), "wall_s": round(wall, 1),
            "states_per_s": round(sel.size / wall, 1) if wall else None,
            "fast_vs_reference_max_delta_10k": dmax, "path": "reference (fast=False)"}
    (out_dir / "oracle_summary.json").write_text(json.dumps(summ, indent=2) + "\n")
    return summ


def _stop_check_float64(pos, vel, system, config):
    from src.envs.double_integrator import DoubleIntegrator
    cfg = config; dev = pos.device
    s64 = DoubleIntegrator(cfg); s64.u_bounds = s64.u_bounds.to(device=dev, dtype=torch.float64)
    dt = float(cfg["cpi"]["labels"]["dt_vm"]); u_max = float(cfg["env"]["bounds"]["double_integrator"]["u_max"])
    x = torch.cat([pos, vel], 1).to(torch.float64)
    for _ in range(t_stop(cfg)):
        v = x[:, 2:4]
        u = torch.where(v.abs() > u_max * dt, -u_max * torch.sign(v), -v / dt)
        x = rk4_step(s64, x, u, dt)
    return float(torch.linalg.norm(x[:, 2:4], dim=1).max())
