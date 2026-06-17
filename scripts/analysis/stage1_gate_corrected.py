"""v2.2.0 Stage 1 (resume 2) — corrected deployment-referenced OOD gate + full-pool logging.

Reuses the saved Fisher (does NOT rebuild). Corrects the two prior gate faults:
  (a) reference: in-distribution reference is now the DEPLOYMENT rollout manifold SCOD was fit
      on (the dissection goal-class timesteps, full-rollout in-deployment states), not the
      narrow start-state distribution that produced the 0.687 near_obstacle artifact;
  (b) valid OOD only: Part 1 tests three genuinely out-of-manifold directions; more_obstacles
      (obs-invariant under Top-K=5) and near_obstacle (an in-manifold low-density tail, per the
      diagnostic Q2) are EXCLUDED with measured justification, reported for transparency;
  (c) Part 2 additionally requires SCOD to flag the real failures we care about: deployment-
      referenced real-collision-vs-goal AUROC >= 0.75.

Gate passes only if Part 1 (all three synthetic OOD >= 0.75) AND Part 2 (>= 0.75). If it
passes, Step 5 runs the unbiased full-pool logging eval. Gate computed on CPU (matches the
diagnostic's deployment-reference numbers exactly); Step 5 logging runs on GPU.

Run: /home/junhyeok/miniconda3/envs/pncbf/bin/python scripts/analysis/stage1_gate_corrected.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (str(REPO_ROOT), str(REPO_ROOT / "scripts/analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)

import stage1_scod_build as B  # noqa: E402
import stage1_gate_and_log as G  # make_fast_velocity, make_near_obstacle  # noqa: E402
import scod_failure_diagnostic as D  # load_episodes, active_series  # noqa: E402
from src.frameworks.jt_pncbf.train import load_framework_from_checkpoint, make_system  # noqa: E402

GATE_SEED = 20260604
N_OOD = 500
THRESH = 0.75
LEAD5 = 5
OUT_DIR = B.OUT_DIR
SUMMARY = OUT_DIR / "gate_corrected_summary.json"
FIG = OUT_DIR / "ood_gate_corrected.png"


def load_art() -> dict:
    f = torch.load(B.FISHER_PATH, map_location="cpu", weights_only=False)
    return {"sigma_out": f["sigma_out"], "epsilon": f["epsilon"],
            "members": [{"eigvals": m["eigvals"], "eigvecs": m["eigvecs"], "k": m["k"]} for m in f["members"]]}


def scod_of(deployed, system, x0, scene):
    with torch.no_grad():
        return B.scod_scalar(deployed, system.observation(x0, scene), ART)[0].cpu().numpy()


def auc_vs_ref(ood_scores, ref_scores):
    labels = np.concatenate([np.ones_like(ood_scores), np.zeros_like(ref_scores)])
    scores = np.concatenate([ood_scores, ref_scores])
    return B.checked_auc(labels, scores)


ART: dict = {}


def main() -> int:
    global ART
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cpu = torch.device("cpu")
    dtype = torch.float32
    fw, config, _ = load_framework_from_checkpoint(B.SECURED_CKPT)
    system = make_system(config)
    deployed_cpu = fw.value_net.to(device=cpu, dtype=dtype).eval()
    ART = load_art()

    # Deployment reference = dissection goal-class timesteps (full-rollout, in-deployment states
    # under the deployed policy on in-distribution scenes) — the manifold SCOD was fit on.
    episodes, recon = D.load_episodes(deployed_cpu, system, ART, cpu, dtype)
    if recon > 1e-6:
        raise SystemExit(f"STOP: reconstruction mismatch {recon:.3e}")
    by = {o: [e for e in episodes if e["outcome"] == o] for o in ("collision", "stuck", "goal")}
    ref = np.concatenate([D.active_series(e, "scod") for e in by["goal"]])

    rng = np.random.default_rng(GATE_SEED)
    # Part 1 — synthetic out-of-manifold OOD vs deployment reference
    part1 = {}
    figdata = {"deployment-ref(goal ts)": ref}
    for kind, maker in (("big_radius", lambda: B.make_ood_batch("big_radius", N_OOD, rng, cpu, dtype)),
                        ("centers_outside", lambda: B.make_ood_batch("centers_outside", N_OOD, rng, cpu, dtype)),
                        ("fast_velocity", lambda: G.make_fast_velocity(N_OOD, rng, cpu, dtype))):
        x0, scene = maker()
        s = scod_of(deployed_cpu, system, x0, scene)
        auc = auc_vs_ref(s, ref)
        part1[kind] = {"auroc": auc, "median": float(np.median(s)), "pass": bool(auc >= THRESH)}
        figdata[kind] = s

    # Excluded (reported, not counted) vs the same deployment reference
    excl = {}
    for kind, maker in (("more_obstacles", lambda: B.make_ood_batch("more_obstacles", N_OOD, rng, cpu, dtype)),
                        ("near_obstacle", lambda: G.make_near_obstacle(N_OOD, rng, cpu, dtype))):
        x0, scene = maker()
        s = scod_of(deployed_cpu, system, x0, scene)
        excl[kind] = {"auroc": auc_vs_ref(s, ref), "median": float(np.median(s)),
                      "reason": "obs-invariant (Top-K=5 count)" if kind == "more_obstacles"
                      else "in-manifold low-density tail (diagnostic Q2: 0.53% of rollout states, in-spec)"}
        figdata[kind] = s

    # Part 2 — real-collision detection vs deployment reference (per-timestep, near-event lead-5)
    coll_near = np.concatenate([e["scod"][max(0, e["event_action"] - LEAD5 + 1): e["event_action"] + 1]
                                for e in by["collision"]])
    part2_ts = auc_vs_ref(coll_near, ref)
    # episode-max vs goal-control episodes (biased-sample context)
    part2_epi = D.assoc_episode(by["collision"], by["goal"], "scod")[0]

    part1_pass = all(v["pass"] for v in part1.values())
    part2_pass = bool(part2_ts >= THRESH)
    gate_passed = part1_pass and part2_pass

    summary = {
        "reference": "dissection goal-class timesteps (full-rollout deployment states)",
        "n_ref_timesteps": int(ref.size), "ref_scod_median": float(np.median(ref)),
        "part1_synthetic_ood": part1, "part1_pass": part1_pass,
        "part2_real_collision": {"deployment_timestep_auroc": part2_ts,
                                 "episode_max_vs_goal_controls": float(part2_epi),
                                 "n_collision_near_steps": int(coll_near.size), "pass": part2_pass},
        "excluded": excl,
        "gate_passed": gate_passed,
    }
    _plot(figdata, summary)
    SUMMARY.write_text(json.dumps(B._jsonable(summary), indent=2), encoding="utf-8")

    print(f"[gate] reference: goal-class deployment timesteps (n={ref.size}, SCOD median {np.median(ref):.2f})")
    for k, v in part1.items():
        print(f"[gate] P1 {k}: AUROC={v['auroc']:.4f} pass={v['pass']} (median {v['median']:.2f})")
    print(f"[gate] P2 real-collision deployment-timestep AUROC={part2_ts:.4f} pass={part2_pass} "
          f"(episode-max vs goal-controls {part2_epi:.4f})")
    for k, v in excl.items():
        print(f"[gate] EXCLUDED {k}: AUROC={v['auroc']:.4f} ({v['reason']})")
    print(f"[gate] PART1_PASS={part1_pass} PART2_PASS={part2_pass} -> GATE_PASSED={gate_passed}")

    if not gate_passed:
        raise SystemExit("STOP: corrected gate failed. See gate_corrected_summary.json.")

    print("[step5] full-pool logging eval (GPU) ...")
    gpu, _ = B.device_dtype()
    fw.value_net.to(device=gpu, dtype=dtype).eval()
    fw.policy_net.to(device=gpu, dtype=dtype).eval()
    fw.system = system
    info = B.run_full_pool_logging(fw, ART, gpu, dtype)
    print(f"[step5] wrote {info['n_episodes']} NPZs -> {info['npz_dir']}")
    print(f"[step5] outcomes: {info['outcome_counts']}")
    summary["logging"] = info
    SUMMARY.write_text(json.dumps(B._jsonable(summary), indent=2), encoding="utf-8")
    print(f"[done] -> {SUMMARY}")
    return 0


def _plot(figdata, summary):
    order = ["deployment-ref(goal ts)", "big_radius", "centers_outside", "fast_velocity", "near_obstacle", "more_obstacles"]
    data = [np.log10(np.clip(figdata[k], 1e-12, None)) for k in order]
    fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=140)
    ax.boxplot(data, tick_labels=order, showfliers=False)
    ax.set_ylabel("log10 SCOD epistemic (max over members)")
    p1 = summary["part1_synthetic_ood"]
    txt = "  ".join(f"{k}:{v['auroc']:.3f}" for k, v in p1.items())
    ax.set_title(f"Corrected gate vs deployment ref — P1[{txt}]  "
                 f"P2 real-collision:{summary['part2_real_collision']['deployment_timestep_auroc']:.3f}", fontsize=8)
    ax.axvline(1.5, color="0.7", ls="--", lw=1)
    fig.tight_layout(); fig.savefig(FIG); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
