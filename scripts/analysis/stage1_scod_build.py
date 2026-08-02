"""v2.2.0 Stage 1 — faithful SCOD build + OOD gate + unbiased full-pool logging.

Read-only on all checkpoints/pools/configs. Writes ONLY: the SCOD Fisher artifact, a build
summary json, OOD figure, and per-episode NPZs under the new diagnostics dir.

Pipeline (hard gate at Step 4):
  Step 2: roll out the 22 surviving policy+value snapshots (steps 2000..42000) with their
          recovered per-step sigma and the target-net HardNet filter, pooling (state, obs)
          as a grade-2 approximation of the training value-buffer manifold D_V.
  Step 3: closed-form last-layer dataset Fisher per deployed member -> top-k eigenspace.
  Step 4: OOD sanity GATE — SCOD must separate three out-of-distribution obstacle fields
          from in-distribution (AUROC >= 0.75 each), else raise (STOP).
  Step 5: all-episode logging eval over the committed full pool with the deployed step-34000
          V_S, adding per-member values, target-h, the SCOD scalar, and the CBF residual.

Run: /home/junhyeok/miniconda3/envs/pncbf/bin/python scripts/analysis/stage1_scod_build.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.metrics import roc_auc_score  # noqa: E402

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
VERIF_DIR = REPO_ROOT / "scripts/verification"
if str(VERIF_DIR) not in sys.path:
    sys.path.insert(0, str(VERIF_DIR))

from src._version import __version__  # noqa: E402
from src.common.control_net import ControlNet  # noqa: E402
from src.common.value_net import ValueNetEnsemble, make_h_fn  # noqa: E402
from src.envs.scene_batch import BatchedScene, batch_scenes, initial_states_from_batch  # noqa: E402
from src.envs.scene_init import sample_train_scene  # noqa: E402
from src.frameworks.jt_pncbf.collection import collect_policy_rollouts, make_replay_buffers  # noqa: E402
from src.frameworks.jt_pncbf.train import load_framework_from_checkpoint, make_system  # noqa: E402
from src.eval.build_pools import load_pool  # noqa: E402
from src.eval.evaluate import evaluate  # noqa: E402

import jt_failure_dissect as dissect  # the per-step NPZ logger (imported, NOT modified)  # noqa: E402


PREV_RUN = REPO_ROOT / "data/runs/v2.0.1/v2.0.1__20260529-171057__seed42"
SECURED_CKPT = REPO_ROOT / "data/secured_data/v2.0.1/seed42/checkpoints/best.pt"
METRICS_CSV = PREV_RUN / "metrics.csv"
FULL_POOL = REPO_ROOT / "data/secured_data/pools/eval_full_di_n500_seed23456.pkl"
OUT_DIR = REPO_ROOT / "data/diagnostics/v2.2.0_stage1_scod"
EPISODE_DIR = OUT_DIR / "episodes"
FISHER_PATH = OUT_DIR / "scod_fisher.pt"
SUMMARY_PATH = OUT_DIR / "build_summary.json"

SNAPSHOT_STEPS = list(range(2000, 42001, 2000))
SCENES_PER_SNAPSHOT = 100
SCENE_SEED = 20260601
OOD_SEED = 20260602
N_OOD = 500
EPSILON = 1.0
K_RANGE = (20, 64)
ENERGY_KEEP = 0.99
OOD_GATE_AUROC = 0.75
WORLD_LIM = 4.0


# --------------------------------------------------------------------------------------
# helpers


def device_dtype() -> tuple[torch.device, torch.dtype]:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return dev, torch.float32


def read_snapshot_sigma() -> dict[int, float]:
    import csv

    rows = {int(r["step"]): r for r in csv.DictReader(METRICS_CSV.open())}
    sigma: dict[int, float] = {}
    for step in SNAPSHOT_STEPS:
        if step not in rows:
            raise SystemExit(f"STOP: sigma for step {step} not in metrics.csv (gate).")
        sigma[step] = float(rows[step]["sigma"])
    return sigma


def final_value_mse() -> float:
    import csv

    rows = list(csv.DictReader(METRICS_CSV.open()))
    tail = [float(r["L_R"]) for r in rows[-5:] if r.get("L_R")]
    return float(np.mean(tail)) if tail else 1.0


def load_snapshot_actors(step: int, device: torch.device, dtype: torch.dtype):
    ck = torch.load(PREV_RUN / f"checkpoints/step_{step:06d}.pt", map_location="cpu", weights_only=False)
    config = ck["config"]
    system = make_system(config)
    policy = ControlNet(system.obs_dim, system, config).to(device=device, dtype=dtype)
    policy.load_state_dict(ck["pi_state"])
    policy.eval()
    target = ValueNetEnsemble(system.obs_dim, config).to(device=device, dtype=dtype)
    target.load_state_dict(ck["v_s_target_state"])
    target.eval()
    return system, policy, target, config


def member_penultimate(member: torch.nn.Sequential, obs: torch.Tensor) -> torch.Tensor:
    return member[:-1](obs)  # Softplus activation feeding the final Linear(256, 1)


def augmented_jacobian(member: torch.nn.Sequential, obs: torch.Tensor) -> torch.Tensor:
    # Last-layer Jacobian of the raw scalar output w.r.t. [weight(256), bias(1)] = [penult, 1].
    penult = member_penultimate(member, obs)
    ones = torch.ones(penult.shape[0], 1, device=penult.device, dtype=penult.dtype)
    return torch.cat([penult, ones], dim=1)  # [N, 257]


def checked_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    labels, scores = labels[finite], scores[finite]
    if len(np.unique(labels)) < 2:
        return float("nan")
    sk = float(roc_auc_score(labels, scores))
    pos, neg = scores[labels == 1], scores[labels == 0]
    diff = pos[:, None] - neg[None, :]
    pw = float((np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / (pos.size * neg.size))
    if abs(sk - pw) > 1e-9:
        raise AssertionError(f"AUROC mismatch sklearn={sk} pairwise={pw}")
    return sk


# --------------------------------------------------------------------------------------
# Step 2 — pooled training-manifold dataset by multi-snapshot rollout


def build_pooled_fisher(deployed: ValueNetEnsemble, system, sigma_by_step: dict[int, float],
                        device: torch.device, dtype: torch.dtype):
    n_vs = len(deployed.members)
    dim = deployed.members[0][-1].in_features + 1  # 257
    fisher_raw = [torch.zeros(dim, dim, dtype=torch.float64, device=device) for _ in range(n_vs)]
    counts: list[tuple[int, float, int]] = []
    total = 0

    np_rng = np.random.default_rng(SCENE_SEED)
    torch_rng = torch.Generator(device=device)
    torch_rng.manual_seed(SCENE_SEED)
    horizon = 100

    for step in SNAPSHOT_STEPS:
        snap_system, policy, target, config = load_snapshot_actors(step, device, dtype)
        horizon = int(config["training"]["oc_pncbf"]["horizon"])
        sigma = sigma_by_step[step]
        buffers = make_replay_buffers(capacity=SCENES_PER_SNAPSHOT)
        collect_policy_rollouts(
            system=snap_system,
            policy_net=policy,
            value_net=target,                       # target net -> make_h_fn deployed(mean) of target
            scene_sampler=lambda r: sample_train_scene(r, config, "double_integrator"),
            rng=np_rng,
            torch_generator=torch_rng,
            n_episodes=SCENES_PER_SNAPSHOT,
            max_steps=horizon,
            dt=float(config["env"]["dt"]),
            buffer=buffers.value,
            config=config,
            sigma=sigma,
            storage_device=device,
            storage_dtype=dtype,
            collection_filter="hardnet",
        )
        buf = buffers.value
        scene = BatchedScene(
            obstacle_centers=buf._tensor_centers,
            obstacle_radii=buf._tensor_radii,
            obstacle_active=buf._tensor_active,
            start=torch.zeros((buf._tensor_states.shape[0], 2), device=device, dtype=dtype),
            goal=buf._tensor_goals,
            system="double_integrator",
            mode="train",
        )
        with torch.no_grad():
            obs = system.observation(buf._tensor_states, scene)         # deployed obs reconstruction
            for m in range(n_vs):
                a = augmented_jacobian(deployed.members[m], obs).double()
                fisher_raw[m] += a.t() @ a
        n_trans = int(buf._tensor_states.shape[0])
        total += n_trans
        counts.append((step, sigma, n_trans))
        del buffers, buf, obs

    return fisher_raw, counts, total, horizon


# --------------------------------------------------------------------------------------
# Step 3 — last-layer dataset Fisher eigenspace + SCOD online closure


def build_scod(fisher_raw: list[torch.Tensor], sigma_out: float) -> dict[str, Any]:
    sigma_out_sq = float(sigma_out) ** 2
    members = []
    for m, m_raw in enumerate(fisher_raw):
        fisher = (m_raw / sigma_out_sq).cpu()
        evals, evecs = torch.linalg.eigh(fisher)           # ascending
        evals = torch.clamp(evals.flip(0), min=0.0)        # descending, non-negative
        evecs = evecs.flip(1)
        total_energy = float(evals.sum().item())
        cum = torch.cumsum(evals, 0) / max(total_energy, 1e-30)
        k_energy = int(torch.searchsorted(cum, torch.tensor(ENERGY_KEEP)).item()) + 1
        k = int(min(max(k_energy, K_RANGE[0]), K_RANGE[1], evals.numel()))
        members.append(
            {
                "eigvals": evals,
                "eigvecs": evecs,
                "k": k,
                "k_energy": k_energy,
                "decay": evals[:12].tolist(),
                "total_energy": total_energy,
                "energy_at_k": float(cum[k - 1].item()),
            }
        )
    return {"sigma_out": float(sigma_out), "epsilon": EPSILON, "members": members}


def scod_per_member(member: torch.nn.Sequential, obs: torch.Tensor, art: dict[str, Any], m: int) -> torch.Tensor:
    sigma_out = art["sigma_out"]
    eps = art["epsilon"]
    info = art["members"][m]
    k = info["k"]
    U = info["eigvecs"][:, :k].to(obs.device, obs.dtype)        # [257, k]
    lam = info["eigvals"][:k].to(obs.device, obs.dtype)         # [k]
    with torch.no_grad():
        g = augmented_jacobian(member, obs) / sigma_out          # [N, 257]
        proj = g @ U                                            # [N, k]
        coef = (eps ** 4) * lam / (1.0 + (eps ** 2) * lam)      # [k]
        unc = (eps ** 2) * (g * g).sum(dim=1) - (coef.unsqueeze(0) * proj * proj).sum(dim=1)
    return torch.clamp(unc, min=0.0)


def scod_scalar(deployed: ValueNetEnsemble, obs: torch.Tensor, art: dict[str, Any]) -> tuple[torch.Tensor, list[torch.Tensor]]:
    per = [scod_per_member(deployed.members[m], obs, art, m) for m in range(len(deployed.members))]
    stacked = torch.stack(per, dim=0)
    return stacked.max(dim=0).values, per     # ensemble rule: max over members (safety-conservative)


# --------------------------------------------------------------------------------------
# Step 4 — OOD gate


def _pad_scene(centers_list, radii_list, active_list, goals, device, dtype):
    nmax = max(c.shape[0] for c in centers_list)
    b = len(centers_list)
    centers = torch.zeros(b, nmax, 2, device=device, dtype=dtype)
    radii = torch.zeros(b, nmax, device=device, dtype=dtype)
    active = torch.zeros(b, nmax, dtype=torch.bool, device=device)
    for i, (c, r, a) in enumerate(zip(centers_list, radii_list, active_list)):
        n = c.shape[0]
        centers[i, :n] = torch.as_tensor(c, device=device, dtype=dtype)
        radii[i, :n] = torch.as_tensor(r, device=device, dtype=dtype)
        active[i, :n] = torch.as_tensor(a, device=device, dtype=torch.bool)
    return SimpleNamespace(
        obstacle_centers=centers, obstacle_radii=radii, obstacle_active=active,
        goal=torch.as_tensor(np.stack(goals), device=device, dtype=dtype),
    )


def make_ood_batch(kind: str, n: int, rng: np.random.Generator, device, dtype):
    centers_list, radii_list, active_list, goals, x0 = [], [], [], [], []
    lim = WORLD_LIM - 0.3
    for _ in range(n):
        if kind == "more_obstacles":
            k = int(rng.integers(16, 21))
            c = rng.uniform(-WORLD_LIM, WORLD_LIM, size=(k, 2))
            r = rng.uniform(0.15, 0.80, size=k)
        elif kind == "big_radius":
            k = int(rng.integers(1, 13))
            c = rng.uniform(-WORLD_LIM, WORLD_LIM, size=(k, 2))
            r = rng.uniform(1.0, 1.4, size=k)
        elif kind == "centers_outside":
            k = int(rng.integers(1, 13))
            sign = rng.choice([-1.0, 1.0], size=(k, 2))
            c = sign * rng.uniform(4.5, 7.5, size=(k, 2))
            r = rng.uniform(0.15, 0.80, size=k)
        else:
            raise ValueError(kind)
        start = rng.uniform(-lim, lim, size=2)
        goal = rng.uniform(-lim, lim, size=2)
        vel = rng.uniform(-0.5, 0.5, size=2)
        centers_list.append(c)
        radii_list.append(r)
        active_list.append(np.ones(c.shape[0], dtype=bool))
        goals.append(goal)
        x0.append(np.concatenate([start, vel]))
    scene = _pad_scene(centers_list, radii_list, active_list, goals, device, dtype)
    return torch.as_tensor(np.stack(x0), device=device, dtype=dtype), scene


def make_indist_batch(n: int, rng: np.random.Generator, system, device, dtype):
    scenes = [sample_train_scene(rng, _BASE_CONFIG, "double_integrator") for _ in range(n)]
    batched = batch_scenes(scenes, device=device, dtype=dtype)
    return initial_states_from_batch(batched), batched


def run_ood_gate(deployed, system, art, device, dtype) -> dict[str, Any]:
    rng = np.random.default_rng(OOD_SEED)
    x0_in, scene_in = make_indist_batch(N_OOD, rng, system, device, dtype)
    with torch.no_grad():
        scod_in, _ = scod_scalar(deployed, system.observation(x0_in, scene_in), art)
    scod_in = scod_in.cpu().numpy()
    results = {"in_dist_median": float(np.median(scod_in)), "kinds": {}}
    figdata = {"in-dist": scod_in}
    for kind in ("more_obstacles", "big_radius", "centers_outside"):
        x0, scene = make_ood_batch(kind, N_OOD, rng, device, dtype)
        with torch.no_grad():
            scod_ood, _ = scod_scalar(deployed, system.observation(x0, scene), art)
        scod_ood = scod_ood.cpu().numpy()
        labels = np.concatenate([np.ones_like(scod_ood), np.zeros_like(scod_in)])
        scores = np.concatenate([scod_ood, scod_in])
        auc = checked_auc(labels, scores)
        results["kinds"][kind] = {
            "auroc": auc,
            "ood_median": float(np.median(scod_ood)),
            "pass": bool(auc >= OOD_GATE_AUROC),
        }
        figdata[kind] = scod_ood
    results["all_pass"] = all(v["pass"] for v in results["kinds"].values())
    _plot_ood(figdata, results)
    return results


def _plot_ood(figdata: dict[str, np.ndarray], results: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=140)
    order = ["in-dist", "more_obstacles", "big_radius", "centers_outside"]
    data = [np.log10(np.clip(figdata[k], 1e-12, None)) for k in order]
    ax.boxplot(data, labels=order, showfliers=False)
    ax.set_ylabel("log10 SCOD epistemic (max over members)")
    title = "OOD gate: " + ", ".join(
        f"{k.split('_')[0]} AUROC={results['kinds'][k]['auroc']:.3f}" for k in results["kinds"]
    )
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ood_gate_scod.png")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Step 5 — unbiased full-pool logging


def run_full_pool_logging(deployed_framework, art, device, dtype) -> dict[str, Any]:
    EPISODE_DIR.mkdir(parents=True, exist_ok=True)
    config = deployed_framework.config
    system = deployed_framework.system
    value_net = deployed_framework.value_net
    h_fn = make_h_fn(value_net, system, use_target=False)
    eval_result = evaluate(
        deployed_framework, FULL_POOL, config,
        mode="diagnostic_stage1_fullpool",
        step=34000, ckpt_name=SECURED_CKPT.name,
        include_lqr_baseline=False, eval_batch_size=250,
    )
    outcome_counts: dict[str, int] = {}
    for idx, episode in enumerate(eval_result.trajectories):
        outcome = episode.filtered_outcome
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        trace = dissect._episode_trace(
            framework=deployed_framework, h_fn=h_fn, config=config,
            scene=episode.scene, result=episode.filtered,
            outcome=outcome, event_step=int(episode.filtered_event_step),
        )
        states = episode.filtered.states[:, 0, :].detach()
        x = states[:-1]
        with torch.no_grad():
            obs = system.observation(x, episode.scene)
            value_all = value_net.value_all(obs)               # [T, n_vs] clipped members
            target_h = value_net.target_h(obs)                 # [T] min ensemble
            scod, per = scod_scalar(value_net, obs, art)        # [T] max over members
        trace["value_member0"] = dissect._np(value_all[:, 0])
        trace["value_member1"] = dissect._np(value_all[:, 1])
        trace["target_h"] = dissect._np(target_h)
        trace["scod_epistemic"] = dissect._np(scod)
        trace["scod_member0"] = dissect._np(per[0])
        trace["scod_member1"] = dissect._np(per[1])
        trace["ensemble_disagreement"] = dissect._np(torch.abs(value_all[:, 0] - value_all[:, 1]))
        dissect._write_episode_npz(EPISODE_DIR, idx, outcome, "ep", trace)
    return {"npz_dir": str(EPISODE_DIR), "outcome_counts": outcome_counts,
            "n_episodes": len(eval_result.trajectories)}


# --------------------------------------------------------------------------------------
# main

_BASE_CONFIG: dict[str, Any] = {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device, dtype = device_dtype()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    global _BASE_CONFIG
    deployed_framework, config, _ = load_framework_from_checkpoint(SECURED_CKPT)
    _BASE_CONFIG = config
    system = make_system(config)
    deployed = deployed_framework.value_net.to(device=device, dtype=dtype)
    deployed.eval()
    deployed_framework.system = system

    sigma_by_step = read_snapshot_sigma()
    sigma_out = float(np.sqrt(max(final_value_mse(), 1e-12)))

    print(f"[step2] rolling out {len(SNAPSHOT_STEPS)} snapshots x {SCENES_PER_SNAPSHOT} scenes ...")
    fisher_raw, counts, total, horizon = build_pooled_fisher(deployed, system, sigma_by_step, device, dtype)
    print(f"[step2] pooled transitions total={total}  horizon={horizon}")

    print(f"[step3] building per-member SCOD Fisher (sigma_out={sigma_out:.4f}, eps={EPSILON}) ...")
    art = build_scod(fisher_raw, sigma_out)
    for m, info in enumerate(art["members"]):
        print(f"[step3] member{m}: k={info['k']} (energy {info['energy_at_k']:.4f}), "
              f"top eigvals={[round(v,3) for v in info['decay'][:6]]}")

    torch.save(
        {
            "version": __version__,
            "epsilon": EPSILON,
            "sigma_out": sigma_out,
            "ensemble_rule": "max",
            "snapshot_counts": counts,
            "total_samples": total,
            "scene_seed": SCENE_SEED,
            "members": [
                {"k": info["k"], "k_energy": info["k_energy"], "eigvals": info["eigvals"],
                 "eigvecs": info["eigvecs"], "total_energy": info["total_energy"]}
                for info in art["members"]
            ],
        },
        FISHER_PATH,
    )

    print("[step4] OOD gate ...")
    ood = run_ood_gate(deployed, system, art, device, dtype)
    for k, v in ood["kinds"].items():
        print(f"[step4] {k}: AUROC={v['auroc']:.4f} pass={v['pass']}")
    if not ood["all_pass"]:
        summary = {"gate_passed": False, "ood": ood, "snapshot_counts": counts,
                   "total_samples": total, "sigma_out": sigma_out}
        SUMMARY_PATH.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
        raise SystemExit("STOP: OOD gate failed (an OOD type has AUROC < 0.75). See build_summary.json.")

    print("[step5] full-pool logging eval ...")
    logging_info = run_full_pool_logging(deployed_framework, art, device, dtype)
    print(f"[step5] wrote {logging_info['n_episodes']} NPZs to {logging_info['npz_dir']}")
    print(f"[step5] outcomes: {logging_info['outcome_counts']}")

    summary = {
        "gate_passed": True,
        "version": __version__,
        "sigma_out": sigma_out,
        "epsilon": EPSILON,
        "ensemble_rule": "max",
        "k_per_member": [info["k"] for info in art["members"]],
        "k_energy_per_member": [info["k_energy"] for info in art["members"]],
        "eig_decay_member0": art["members"][0]["decay"],
        "eig_decay_member1": art["members"][1]["decay"],
        "snapshot_counts": counts,
        "total_samples": total,
        "scene_seed": SCENE_SEED,
        "ood": ood,
        "logging": logging_info,
        "fisher_artifact": str(FISHER_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print(f"[done] summary -> {SUMMARY_PATH}")
    return 0


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    return obj


if __name__ == "__main__":
    raise SystemExit(main())
