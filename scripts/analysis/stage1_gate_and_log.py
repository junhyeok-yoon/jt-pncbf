"""v2.2.0 Stage 1 (resume) — observation-aware OOD gate + unbiased full-pool logging.

Reuses the saved SCOD Fisher (data/diagnostics/v2.2.0_stage1_scod/scod_fisher.pt); does NOT
rebuild it. Redesigns the Step-4 gate to test ONLY observation-changing OOD directions, then —
if the gate passes (every valid OOD type AUROC >= 0.75) — runs the Step-5 all-episode logging
eval over the committed full pool. Read-only on the Fisher, checkpoints, pool, configs.

Run: /home/junhyeok/miniconda3/envs/pncbf/bin/python scripts/analysis/stage1_gate_and_log.py
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (str(REPO_ROOT), str(REPO_ROOT / "scripts/verification"), str(REPO_ROOT / "scripts/analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)

import stage1_scod_build as B  # noqa: E402
from src.frameworks.jt_pncbf.train import load_framework_from_checkpoint, make_system  # noqa: E402


GATE_SEED = 20260603
N_GATE = 500
OOD_GATE_AUROC = 0.75
WORLD_LIM = 4.0
V_MAX = 2.5            # training velocity ceiling (post-step clamp); rollout speeds <= 2.5
GATE_SUMMARY = B.OUT_DIR / "gate_resume_summary.json"
GATE_FIG = B.OUT_DIR / "ood_gate_scod_resume.png"


def load_art() -> dict:
    f = torch.load(B.FISHER_PATH, map_location="cpu", weights_only=False)
    return {
        "sigma_out": f["sigma_out"],
        "epsilon": f["epsilon"],
        "members": [{"eigvals": m["eigvals"], "eigvecs": m["eigvecs"], "k": m["k"]} for m in f["members"]],
    }


def _indist_obstacles(rng):
    k = int(rng.integers(1, 13))
    c = rng.uniform(-WORLD_LIM, WORLD_LIM, size=(k, 2))
    r = rng.uniform(0.15, 0.80, size=k)
    return c, r


def make_fast_velocity(n, rng, device, dtype):
    """In-distribution obstacle field; velocity components beyond v_max=2.5 (genuine OOD:
    rollout speeds are clamped to 2.5, so |v| in [3.0,5.0] per component is outside the
    training velocity manifold). Only the velocity obs dims (vx, vy) are perturbed."""
    centers, radii, active, goals, x0 = [], [], [], [], []
    lim = WORLD_LIM - 0.3
    for _ in range(n):
        c, r = _indist_obstacles(rng)
        start = rng.uniform(-lim, lim, size=2)
        goal = rng.uniform(-lim, lim, size=2)
        vel = rng.choice([-1.0, 1.0], size=2) * rng.uniform(3.0, 5.0, size=2)
        centers.append(c); radii.append(r); active.append(np.ones(c.shape[0], bool))
        goals.append(goal); x0.append(np.concatenate([start, vel]))
    scene = B._pad_scene(centers, radii, active, goals, device, dtype)
    return torch.as_tensor(np.stack(x0), device=device, dtype=dtype), scene


def make_near_obstacle(n, rng, device, dtype):
    """In-distribution obstacle field plus ONE obstacle forced to a surface distance in
    [0.01,0.10] from the agent start (training enforces start clearance >= 0.10, so the
    nearest-obstacle obs dims fall below the trained start range). Only nearest-obstacle
    relative position is perturbed; velocity stays in the training initial range."""
    centers, radii, active, goals, x0 = [], [], [], [], []
    lim = WORLD_LIM - 0.3
    for _ in range(n):
        c, r = _indist_obstacles(rng)
        start = rng.uniform(-lim, lim, size=2)
        goal = rng.uniform(-lim, lim, size=2)
        theta = rng.uniform(-np.pi, np.pi)
        r0 = float(rng.uniform(0.15, 0.80))
        surf = float(rng.uniform(0.01, 0.10))
        d = r0 + surf
        center = start + d * np.array([np.cos(theta), np.sin(theta)])
        c = np.concatenate([c, center.reshape(1, 2)], axis=0)
        r = np.concatenate([r, [r0]])
        vel = rng.uniform(-0.5, 0.5, size=2)
        centers.append(c); radii.append(r); active.append(np.ones(c.shape[0], bool))
        goals.append(goal); x0.append(np.concatenate([start, vel]))
    scene = B._pad_scene(centers, radii, active, goals, device, dtype)
    return torch.as_tensor(np.stack(x0), device=device, dtype=dtype), scene


def nearest_surface_distance(x0, scene, device, dtype):
    pos = x0[:, :2]
    c = scene.obstacle_centers if torch.is_tensor(scene.obstacle_centers) else torch.as_tensor(scene.obstacle_centers, device=device, dtype=dtype)
    r = scene.obstacle_radii if torch.is_tensor(scene.obstacle_radii) else torch.as_tensor(scene.obstacle_radii, device=device, dtype=dtype)
    a = scene.obstacle_active if torch.is_tensor(scene.obstacle_active) else torch.as_tensor(scene.obstacle_active, device=device, dtype=torch.bool)
    if c.ndim == 2:
        c = c.unsqueeze(0).expand(pos.shape[0], -1, -1); r = r.unsqueeze(0).expand(pos.shape[0], -1); a = a.unsqueeze(0).expand(pos.shape[0], -1)
    d = torch.linalg.norm(c - pos.unsqueeze(1), dim=-1) - r
    d = d.masked_fill(~a, float("inf"))
    return float(d.min(dim=1).values.median().item())


def run_gate(deployed, system, art, device, dtype):
    rng = np.random.default_rng(GATE_SEED)
    x0_in, scene_in = B.make_indist_batch(N_GATE, rng, system, device, dtype)
    with torch.no_grad():
        scod_in = B.scod_scalar(deployed, system.observation(x0_in, scene_in), art)[0].cpu().numpy()
    valid = {
        "big_radius": B.make_ood_batch("big_radius", N_GATE, rng, device, dtype),
        "centers_outside": B.make_ood_batch("centers_outside", N_GATE, rng, device, dtype),
        "fast_velocity": make_fast_velocity(N_GATE, rng, device, dtype),
        "near_obstacle": make_near_obstacle(N_GATE, rng, device, dtype),
    }
    excluded = {"more_obstacles": B.make_ood_batch("more_obstacles", N_GATE, rng, device, dtype)}

    figdata = {"in-dist": scod_in}
    results = {"in_dist_median": float(np.median(scod_in)),
               "in_dist_nearest_surf_median": nearest_surface_distance(x0_in, scene_in, device, dtype),
               "valid": {}, "excluded": {}}
    for kind, (x0, scene) in valid.items():
        with torch.no_grad():
            scod = B.scod_scalar(deployed, system.observation(x0, scene), art)[0].cpu().numpy()
        labels = np.concatenate([np.ones_like(scod), np.zeros_like(scod_in)])
        scores = np.concatenate([scod, scod_in])
        auc = B.checked_auc(labels, scores)
        results["valid"][kind] = {"auroc": auc, "ood_median": float(np.median(scod)),
                                  "nearest_surf_median": nearest_surface_distance(x0, scene, device, dtype),
                                  "pass": bool(auc >= OOD_GATE_AUROC)}
        figdata[kind] = scod
    for kind, (x0, scene) in excluded.items():
        with torch.no_grad():
            scod = B.scod_scalar(deployed, system.observation(x0, scene), art)[0].cpu().numpy()
        labels = np.concatenate([np.ones_like(scod), np.zeros_like(scod_in)])
        scores = np.concatenate([scod, scod_in])
        results["excluded"][kind] = {"auroc": B.checked_auc(labels, scores),
                                     "ood_median": float(np.median(scod)),
                                     "nearest_surf_median": nearest_surface_distance(x0, scene, device, dtype),
                                     "note": "obs-invariant: count beyond Top-K=5 not encoded; excluded from gate"}
        figdata[kind] = scod
    results["all_valid_pass"] = all(v["pass"] for v in results["valid"].values())
    _plot(figdata, results)
    return results


def _plot(figdata, results):
    B.OUT_DIR.mkdir(parents=True, exist_ok=True)
    order = ["in-dist", "big_radius", "centers_outside", "fast_velocity", "near_obstacle", "more_obstacles"]
    data = [np.log10(np.clip(figdata[k], 1e-12, None)) for k in order]
    fig, ax = plt.subplots(figsize=(9.0, 4.6), dpi=140)
    ax.boxplot(data, tick_labels=order, showfliers=False)
    ax.set_ylabel("log10 SCOD epistemic (max over members)")
    auc_txt = "  ".join(f"{k}:{v['auroc']:.3f}{'*' if not v['pass'] else ''}" for k, v in results["valid"].items())
    ax.set_title(f"Redesigned OOD gate (valid types) — {auc_txt}  | excluded more_obstacles:"
                 f"{results['excluded']['more_obstacles']['auroc']:.3f}", fontsize=8)
    ax.axvline(5.5, color="0.7", ls="--", lw=1)
    fig.tight_layout()
    fig.savefig(GATE_FIG)
    plt.close(fig)


def main() -> int:
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device, dtype = B.device_dtype()
    fw, config, _ = load_framework_from_checkpoint(B.SECURED_CKPT)
    B._BASE_CONFIG = config
    system = make_system(config)
    fw.system = system
    deployed = fw.value_net.to(device=device, dtype=dtype).eval()
    art = load_art()

    print("[gate] redesigned observation-aware OOD gate ...")
    gate = run_gate(deployed, system, art, device, dtype)
    for k, v in gate["valid"].items():
        print(f"[gate] {k}: AUROC={v['auroc']:.4f} pass={v['pass']} "
              f"(ood_med={v['ood_median']:.3f}, surf_med={v['nearest_surf_median']:.3f})")
    print(f"[gate] EXCLUDED more_obstacles: AUROC={gate['excluded']['more_obstacles']['auroc']:.4f} "
          f"(surf_med={gate['excluded']['more_obstacles']['nearest_surf_median']:.3f})")
    print(f"[gate] in-dist surf_med={gate['in_dist_nearest_surf_median']:.3f}")

    gate["gate_passed"] = gate["all_valid_pass"]
    GATE_SUMMARY.write_text(json.dumps(B._jsonable(gate), indent=2), encoding="utf-8")

    if not gate["all_valid_pass"]:
        raise SystemExit("STOP: redesigned OOD gate failed (a valid obs-changing OOD type < 0.75). "
                         "See gate_resume_summary.json.")

    print("[step5] full-pool logging eval ...")
    logging_info = B.run_full_pool_logging(fw, art, device, dtype)
    print(f"[step5] wrote {logging_info['n_episodes']} NPZs -> {logging_info['npz_dir']}")
    print(f"[step5] outcomes: {logging_info['outcome_counts']}")
    gate["logging"] = logging_info
    GATE_SUMMARY.write_text(json.dumps(B._jsonable(gate), indent=2), encoding="utf-8")
    print(f"[done] summary -> {GATE_SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
