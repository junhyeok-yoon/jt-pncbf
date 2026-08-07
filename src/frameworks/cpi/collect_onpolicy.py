"""P1.1 on-policy collection (distribution-matched evaluation, Addendum 4).

Rolls a registry transport T = filter_{V_hat} o pi (the DEPLOY-time filtered policy) with the standard
episode rollout on fresh training-mode scenes, and records the visited states (subsampled). These states
are later labeled with the cumulative bail-out family (same labeler as A6) and merged with the it0 uniform
dataset to regress a distribution-matched certificate.

Seeding: extends the SAME master-seed-42 chain — np.random.SeedSequence(master_seed).spawn(7); children
[0..3] are the it0 streams [scene,state,oracle,split] (spawn is order/count-deterministic, so they are
bit-identical to spawn(4)), children [4,5,6] are the P1 streams {scenes, subsample, split}. NO new seed
constant is introduced.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.common.control_net import ControlNet
from src.common.filter_hardnet import (_base_alpha, _row_upper, _base_projection, _box_aware_projection, _cbf_terms,
                                        _hardnet_params)
from src.common.rk4 import rk4_step
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.envs.scene_init import sample_train_scene
from src.frameworks.cpi.channel import make_cpi_h_fn


def p1_streams(config: Mapping[str, Any]) -> dict[str, np.random.Generator]:
    """P1 streams from the master-seed-42 chain: spawn(7) -> [scene,state,oracle,split, P1_scenes,
    P1_subsample, P1_split]. Children 0..3 equal the it0 label_streams exactly."""
    ms = int(config["cpi"]["labels"]["master_seed"])
    ch = np.random.SeedSequence(ms).spawn(7)
    return {"scene": np.random.default_rng(ch[4]), "subsample": np.random.default_rng(ch[5]),
            "split": np.random.default_rng(ch[6])}


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def collect_onpolicy(config, out_dir, system, device, vhat_ckpt, pi_ckpt, *, n_scenes=4000,
                     max_steps=None, subsample=4, cap=150, scene_batch=250, dtype=torch.float32):
    """Roll T=(vhat_ckpt, pi_ckpt) on n_scenes training scenes; keep every `subsample`-th visited state per
    scene (random per-scene phase from the P1 subsample stream), capped at `cap`/scene. Writes states.npz
    (pos, vel, scene_id, obs) + the per-scene obstacle arrays + manifest.json. Returns the manifest."""
    out_dir = Path(out_dir); (out_dir).mkdir(parents=True, exist_ok=True)
    max_steps = int(config["eval"]["max_steps"]) if max_steps is None else int(max_steps)
    dt = float(config["env"]["dt"]); params = _hardnet_params(config); bounds = system.u_bounds
    streams = p1_streams(config)
    scenes = [sample_train_scene(streams["scene"], config, system.name) for _ in range(n_scenes)]
    phase = streams["subsample"].integers(0, subsample, size=n_scenes)

    ck = torch.load(pi_ckpt, map_location=device, weights_only=False)
    odim = system.obs_dim + (system.action_dim
                             if config.get("loss", {}).get("policy", {}).get("obs_deficit_feedback") else 0)
    pol = ControlNet(odim, system, config).to(device=device, dtype=dtype)
    pol.load_state_dict(ck["pi_state"]); pol.eval()
    h_fn = make_cpi_h_fn(vhat_ckpt, system)

    POS = []; VEL = []; SID = []; OBS = []
    for s0 in range(0, n_scenes, scene_batch):
        chunk = scenes[s0:s0 + scene_batch]
        bs = batch_scenes(chunk, device=device, dtype=dtype)
        x = initial_states_from_batch(bs).to(dtype).to(device)
        B = x.shape[0]
        xs = []; obss = []
        with torch.no_grad():
            for _ in range(max_steps):
                obs = system.observation(x, bs)                      # dim-19 base obs of x_t
                xs.append(x.clone()); obss.append(obs.clone())
                un = pol(obs)
                h, lf, lg = _cbf_terms(system, h_fn, x, bs, un, create_graph=False)
                h, lf, lg = h.detach(), lf.detach(), lg.detach()
                alpha = _base_alpha(h, params); row = _row_upper(lf, alpha, h, params)
                proj = _base_projection(un, lg, row, bounds, params)
                us, _ = _box_aware_projection(un, proj, lg, row, bounds)
                x = rk4_step(system, x, us, dt)
        Xt = torch.stack(xs, 0)                                       # [T,B,4]
        Ot = torch.stack(obss, 0)                                     # [T,B,obs_dim]
        T = Xt.shape[0]
        t_idx = torch.arange(T, device=device).unsqueeze(1)          # [T,1]
        ph = torch.as_tensor(phase[s0:s0 + B], device=device).unsqueeze(0)  # [1,B]
        mask = ((t_idx - ph) % subsample == 0) & (t_idx >= ph)       # [T,B]
        # per-scene cap: keep the earliest `cap` kept steps
        if cap is not None:
            cs = mask.long().cumsum(0)
            mask = mask & (cs <= cap)
        sel = mask.nonzero(as_tuple=False)                           # [N,2] = (t, local_scene)
        t_sel, b_sel = sel[:, 0], sel[:, 1]
        X = Xt[t_sel, b_sel]; O = Ot[t_sel, b_sel]
        POS.append(X[:, :2].cpu().numpy()); VEL.append(X[:, 2:4].cpu().numpy())
        SID.append((b_sel + s0).cpu().numpy()); OBS.append(O.cpu().numpy())

    POS = np.concatenate(POS); VEL = np.concatenate(VEL); SID = np.concatenate(SID); OBS = np.concatenate(OBS)
    # per-scene obstacle arrays (for the family labeler)
    C = np.stack([s.obstacle_centers for s in scenes]).astype(np.float32)
    R = np.stack([s.obstacle_radii for s in scenes]).astype(np.float32)
    A = np.stack([s.obstacle_active for s in scenes])
    G = np.stack([np.asarray(s.goal, np.float64) for s in scenes]).astype(np.float32)
    np.savez(out_dir / "states.npz", pos=POS, vel=VEL, scene_id=SID, obs=OBS)
    np.savez(out_dir / "scene_obstacles.npz", C=C, R=R, A=A, G=G)

    per_scene = np.bincount(SID, minlength=n_scenes)
    manifest = {
        "transport": {"vhat_ckpt": str(vhat_ckpt), "pi_ckpt": str(pi_ckpt),
                      "vhat_sha256": hashlib.sha256(Path(vhat_ckpt).read_bytes()).hexdigest(),
                      "pi_sha256": hashlib.sha256(Path(pi_ckpt).read_bytes()).hexdigest()},
        "n_scenes": int(n_scenes), "max_steps": int(max_steps), "subsample": int(subsample), "cap": int(cap),
        "n_states": int(POS.shape[0]), "states_per_scene_mean": float(per_scene.mean()),
        "states_per_scene_min": int(per_scene.min()), "states_per_scene_max": int(per_scene.max()),
        "seed_chain": "SeedSequence(master_seed).spawn(7)[4,5,6]={scenes,subsample,split}",
        "sha256": {"pos": _sha(POS), "vel": _sha(VEL), "scene_id": _sha(SID), "obs": _sha(OBS),
                   "C": _sha(C), "R": _sha(R), "A": _sha(A), "G": _sha(G)},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def label_family_onpolicy(config, out_dir, system, device, transport_pairs, *, dtype=torch.float32):
    """Label the collected on-policy states with the CUMULATIVE bail-out family over transport_pairs
    (ordered (vhat_ckpt, pi_ckpt) list, same labeler as A6). Returns V [N] (np) in states.npz order and
    writes labels.npz (V, state order == states.npz). No new randomness."""
    from src.frameworks.cpi.family import build_transports, family_value
    out_dir = Path(out_dir)
    st = np.load(out_dir / "states.npz"); so = np.load(out_dir / "scene_obstacles.npz")
    pos = torch.as_tensor(st["pos"], dtype=dtype, device=device)
    vel = torch.as_tensor(st["vel"], dtype=dtype, device=device)
    sid = torch.as_tensor(st["scene_id"], dtype=torch.long, device=device)
    x = torch.cat([pos, vel], 1)
    C = torch.as_tensor(so["C"], dtype=dtype, device=device); R = torch.as_tensor(so["R"], dtype=dtype, device=device)
    A = torch.as_tensor(so["A"], dtype=torch.bool, device=device); G = torch.as_tensor(so["G"], dtype=dtype, device=device)
    transports = build_transports(transport_pairs, system, config)
    V = family_value(x, C[sid], R[sid], A[sid], G[sid], transports, system, config, t_bailout=40).cpu().numpy()
    assert np.isfinite(V).all(), "non-finite on-policy family label"
    np.savez(out_dir / "labels.npz", V=V, scene_id=st["scene_id"])
    return V


def scene_config_hashes(C, R, A, G):
    """Per-scene content hash of the obstacle+goal config (for the it0-vs-on-policy disjointness assert)."""
    out = []
    for i in range(C.shape[0]):
        blob = np.concatenate([C[i].ravel(), R[i].ravel(), A[i].astype(np.float64).ravel(), G[i].ravel()])
        out.append(hashlib.sha1(np.ascontiguousarray(blob, dtype=np.float64).tobytes()).hexdigest())
    return out


if __name__ == "__main__":
    import argparse
    import yaml
    from src.envs.double_integrator import DoubleIntegrator
    from src.envs.unicycle import Unicycle

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--vhat", required=True)
    ap.add_argument("--pi", required=True)
    ap.add_argument("--n-scenes", type=int, default=4000)
    ap.add_argument("--subsample", type=int, default=4)
    ap.add_argument("--cap", type=int, default=150)
    a = ap.parse_args()
    REPO = Path(__file__).resolve().parents[3]

    def mrg(x, o):
        d = dict(x)
        for k, v in o.items():
            d[k] = mrg(d[k], v) if isinstance(v, dict) and isinstance(d.get(k), dict) else v
        return d
    cfg = mrg(yaml.safe_load(open(REPO / "src/configs/base_config.yaml")),
              yaml.safe_load(open(REPO / "src/configs/exp_config.yaml")))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _cls = Unicycle if str(cfg.get("run", {}).get("system", "double_integrator")) == "unicycle" else DoubleIntegrator
    sysm = _cls(cfg); sysm.u_bounds = sysm.u_bounds.to(device=dev, dtype=torch.float32)
    m = collect_onpolicy(cfg, a.out, sysm, dev, a.vhat, a.pi,
                         n_scenes=a.n_scenes, subsample=a.subsample, cap=a.cap)
    print(json.dumps({k: m[k] for k in ("n_states", "states_per_scene_mean", "states_per_scene_min",
          "states_per_scene_max", "n_scenes")}), flush=True)
    print("DONE", flush=True)
