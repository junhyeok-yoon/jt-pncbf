"""v2.2.0 Stage 2 (Steps 1-3) — N=30000 deployment eval with conditional per-step logging + labels.

Step 1: build a deterministic N=30000 eval pool from the SAME deployment distribution (base
        obstacle/scene + eval overrides; sample_eval_scene), new seed; write the pool file.
Step 2: chunked (eval_batch_size~1500, consume-discard) all-episode eval with the deployed
        step-34000 V_S (bit-identical control); per saved episode log existing dissection fields
        PLUS per-member values, target-h, SCOD (saved Fisher closure), 2-member, CBF residual.
        Write NPZ only for failures (collision/stuck/timeout/oob) or until a 1000 goal quota.
Step 3: label saved failures by reusing the dissection's own classifiers — mechanism via
        jt_failure_dissect._classify_mechanism (pure trace fn) and layer via
        jt_failure_layer_attribution._attribute_episode (A* geometry) — to stage2_failure_labels.csv.

Read-only on the Fisher/checkpoints/committed pool/configs. Reuses the saved Fisher (not rebuilt).
Run: /home/junhyeok/miniconda3/envs/pncbf/bin/python scripts/analysis/stage2_eval.py
"""

from __future__ import annotations

import csv
import gc
import json
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (str(REPO_ROOT), str(REPO_ROOT / "scripts/analysis"), str(REPO_ROOT / "scripts/verification")):
    if p not in sys.path:
        sys.path.insert(0, p)

import stage1_scod_build as B  # scod_scalar, device_dtype, FISHER_PATH, SECURED_CKPT  # noqa: E402
import jt_failure_dissect as dissect  # _episode_trace, _write_episode_npz, _np, _classify_mechanism  # noqa: E402
import jt_failure_layer_attribution as layerlib  # _attribute_episode  # noqa: E402
from src.common.value_net import make_h_fn  # noqa: E402
from src.eval.build_pools import EvaluationPool, PoolSpec, build_pool, write_pool  # noqa: E402
from src.eval.evaluate import evaluate  # noqa: E402
from src.frameworks.jt_pncbf.train import load_framework_from_checkpoint, make_system  # noqa: E402

N = 30000
NEW_SEED = 20260605
CHUNK = 250          # bounded so the retained HardNet autograd graph (create_graph per step) fits 16GB
GOAL_QUOTA = 1000
FAILURE_OUTCOMES = {"collision", "stuck", "timeout", "oob"}

OUT = REPO_ROOT / "data/diagnostics/v2.2.0_stage2_largeN"
EPISODE_DIR = OUT / "episodes"
LABELS = OUT / "stage2_failure_labels.csv"
SUMMARY = OUT / "stage2_eval_summary.json"


def load_art() -> dict:
    f = torch.load(B.FISHER_PATH, map_location="cpu", weights_only=False)
    return {"sigma_out": f["sigma_out"], "epsilon": f["epsilon"],
            "members": [{"eigvals": m["eigvals"], "eigvecs": m["eigvecs"], "k": m["k"]} for m in f["members"]]}


def main() -> int:
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    EPISODE_DIR.mkdir(parents=True, exist_ok=True)   # NEVER auto-wipe; resumable/idempotent append
    device, dtype = B.device_dtype()

    fw, ckpt_config, _ = load_framework_from_checkpoint(B.SECURED_CKPT)
    system = make_system(ckpt_config)
    fw.system = system
    fw.value_net.to(device=device, dtype=dtype).eval()
    fw.policy_net.to(device=device, dtype=dtype).eval()
    value_net = fw.value_net
    art = load_art()
    h_fn = make_h_fn(value_net, system, use_target=False)

    # Step 1 — build + write the N=30000 deployment-distribution pool
    config = deepcopy(ckpt_config)
    config["eval"]["full"]["n"] = N
    config["eval"]["full"]["seed"] = NEW_SEED
    spec = PoolSpec(name="full", n_scenes=N, seed=NEW_SEED)
    from src.eval.build_pools import load_pool, pool_stem, obstacle_distribution_name
    pool_path = OUT / f"{pool_stem('full', 'double_integrator', N, NEW_SEED, obstacle_distribution_name(config))}.pkl"
    if pool_path.exists():
        print(f"[step1] reusing existing deterministic pool {pool_path}")
        pool = load_pool(pool_path)
    else:
        print(f"[step1] building N={N} pool (seed {NEW_SEED}, deployment distribution) ...")
        pool = build_pool(config, "double_integrator", spec)
        write_pool(pool, config, output_dir=OUT)
    artifacts_path = pool_path

    # Step 2 — chunked eval + conditional logging (consume-discard per chunk)
    outcome_counts: dict[str, int] = {}
    saved_counts: dict[str, int] = {}
    goal_saved = len(list(EPISODE_DIR.glob("ep_*_goal.npz")))   # resume goal quota from disk
    scenes = pool.scenes
    for start in range(0, N, CHUNK):
        chunk = scenes[start:start + CHUNK]
        chunk_pool = EvaluationPool(name="stage2_chunk", system="double_integrator",
                                    n_scenes=len(chunk), seed=NEW_SEED, scenes=chunk)
        result = evaluate(fw, chunk_pool, config, mode="diagnostic_stage2_largeN",
                          step=34000, ckpt_name=B.SECURED_CKPT.name,
                          include_lqr_baseline=False, eval_batch_size=CHUNK)
        for local, ep in enumerate(result.trajectories):
            outcome = ep.filtered_outcome
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
            save = outcome in FAILURE_OUTCOMES or (outcome == "goal" and goal_saved < GOAL_QUOTA)
            if not save:
                continue
            idx = start + local
            if (EPISODE_DIR / f"ep_{idx:05d}_{outcome}.npz").exists():   # idempotent resume: never overwrite/wipe
                saved_counts[outcome] = saved_counts.get(outcome, 0) + 1
                if outcome == "goal":
                    goal_saved += 1
                continue
            trace = dissect._episode_trace(framework=fw, h_fn=h_fn, config=config, scene=ep.scene,
                                           result=ep.filtered, outcome=outcome,
                                           event_step=int(ep.filtered_event_step))
            x = ep.filtered.states[:-1, 0, :].detach()
            with torch.no_grad():
                obs = system.observation(x, ep.scene)
                members = value_net.value_all(obs)
                target_h = value_net.target_h(obs)
                scod, per = B.scod_scalar(value_net, obs, art)
            trace["value_member0"] = dissect._np(members[:, 0])
            trace["value_member1"] = dissect._np(members[:, 1])
            trace["target_h"] = dissect._np(target_h)
            trace["scod_epistemic"] = dissect._np(scod)
            trace["scod_member0"] = dissect._np(per[0])
            trace["scod_member1"] = dissect._np(per[1])
            trace["ensemble_disagreement"] = dissect._np(torch.abs(members[:, 0] - members[:, 1]))
            dissect._write_episode_npz(EPISODE_DIR, idx, outcome, "ep", trace)
            saved_counts[outcome] = saved_counts.get(outcome, 0) + 1
            if outcome == "goal":
                goal_saved += 1
        del result
        gc.collect()                      # break autograd (create_graph) ref cycles before next chunk
        if device.type == "cuda":
            torch.cuda.empty_cache()
        done = min(start + CHUNK, N)
        print(f"[step2] {done}/{N} | outcomes={outcome_counts} | saved={saved_counts}")

    # Step 3 — labels for saved failure episodes (reuse dissection classifiers)
    print("[step3] labeling saved failures (mechanism: _classify_mechanism; layer: _attribute_episode A*) ...")
    rows = []
    for path in sorted(EPISODE_DIR.glob("ep_*.npz")):
        z = np.load(path, allow_pickle=False)
        md = json.loads(str(np.asarray(z["metadata_json"]).item()))
        outcome = str(md["outcome"])
        trace = {k: z[k] for k in z.files if k != "metadata_json"}
        trace["event_step"] = int(md["event_step"])
        trace["outcome"] = outcome
        idx = int(path.stem.split("_")[1])
        mechanism = layer = "n/a"
        if outcome in ("collision", "stuck"):
            mechanism = dissect._classify_mechanism(outcome, trace)
            try:
                layer = layerlib._attribute_episode(path, z, outcome).layer
            except Exception as exc:  # A* fallback noted per-row
                layer = f"layer_error:{type(exc).__name__}"
        ea = dissect._event_action_index(trace)
        rows.append({"episode_idx": idx, "outcome": outcome, "mechanism": mechanism, "layer": layer,
                     "event_clearance": float(z["clearance"][ea]),
                     "event_closing_speed": float(z["closing_speed"][ea])})
    with LABELS.open("w", newline="", encoding="utf-8") as fobj:
        w = csv.DictWriter(fobj, fieldnames=["episode_idx", "outcome", "mechanism", "layer",
                                             "event_clearance", "event_closing_speed"])
        w.writeheader()
        w.writerows(rows)
    print(f"[step3] labels -> {LABELS} ({len(rows)} rows)")

    summary = {"n": N, "seed": NEW_SEED, "chunk": CHUNK, "goal_quota": GOAL_QUOTA,
               "pool_path": str(artifacts_path),
               "episode_dir": str(EPISODE_DIR), "labels_path": str(LABELS),
               "outcome_counts": outcome_counts, "saved_counts": saved_counts,
               "n_saved_total": sum(saved_counts.values())}
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] outcomes over {N}: {outcome_counts}")
    print(f"[done] saved: {saved_counts} (total {sum(saved_counts.values())}) -> {SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
