"""v2.7.1 Stage-1 — eval-only k-step empty-branch fallback. Three canonical evals on the M5 best.pt (full pool
n2000 seed23456): mode=none (baseline reproduction), kstep k=5, kstep k=10. Per mode: cps + collision +
scene-bootstrap CI, wall-clock; per-episode split empty/singular rates; flip lists vs mode=none. No training."""
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.eval.evaluate import evaluate
from src.frameworks.jt_pncbf.train import load_framework_from_checkpoint

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
SP = Path("/tmp/claude-1000/-home-junhyeok-MIT-jt-pncbf/31d93785-ac11-4206-bf50-a4c3de145dff/scratchpad")
CKPT = REPO / "data/v2.7.1__20260718-114933__seed42/checkpoints/best.pt"
POOL = REPO / "data/secured_data/pools/eval_full_quadrotor-planar_n2000_seed23456.pkl"


def boot(vals, seed=20260718, n=10000):
    v = np.asarray(vals); rng = np.random.default_rng(seed)
    b = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)])
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def run_mode(mode, k):
    over = {"filter": {"empty_fallback": {"mode": mode, "k": int(k)}}}
    framework, config, ck = load_framework_from_checkpoint(CKPT, config_overrides=over)
    t0 = time.time()
    res = evaluate(framework, POOL, config, mode="final", step=int(ck["step"]), ckpt_name="best.pt",
                   include_lqr_baseline=False)
    wall = time.time() - t0
    rows = [r for r in res.episode_rows if r.get("mode") == "final"]
    return rows, wall


def summarize(rows):
    cps = [float(r["cps_episode"]) for r in rows]; coll = [float(r["collision"]) for r in rows]
    cm, clo, chi = boot(cps); om, olo, ohi = boot(coll)
    return dict(n=len(rows), cps=cm, cps_ci=[clo, chi], collision=om, collision_ci=[olo, ohi],
                reach=float(np.mean([r["reach"] for r in rows])),
                timeout=float(np.mean([r["timeout"] for r in rows])),
                empty_step_frac=float(np.mean([float(r.get("empty_step_frac", 0)) for r in rows])),
                singular_step_frac=float(np.mean([float(r.get("singular_step_frac", 0)) for r in rows])),
                infeasible_step_frac=float(np.mean([float(r.get("infeasible_step_frac", 0)) for r in rows])),
                saturation=float(np.mean([float(r.get("saturation_step_frac", 0)) for r in rows])),
                mean_proj_mag=float(np.mean([float(r.get("mean_proj_mag", 0)) for r in rows])))


def flips(base_rows, alt_rows):
    b = {int(r["episode_idx"]): r for r in base_rows}; a = {int(r["episode_idx"]): r for r in alt_rows}
    out = []
    for i in sorted(b):
        bo, ao = b[i]["outcome"], a[i]["outcome"]
        if bo != ao:
            out.append(dict(episode=i, frm=bo, to=ao, had_empty=bool(float(b[i].get("empty_step_frac", 0)) > 0)))
    return out


def main():
    results = {}
    allrows = {}
    for tag, mode, k in [("none", "none", 10), ("kstep_k5", "kstep", 5), ("kstep_k10", "kstep", 10)]:
        print(f"=== running {tag} ===", flush=True)
        rows, wall = run_mode(mode, k)
        allrows[tag] = rows
        s = summarize(rows); s["wall_s"] = round(wall, 1)
        results[tag] = s
        # per-mode episode CSV
        import csv
        with open(SP / f"stage1_{tag}_episodes.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"  {tag}: cps={s['cps']:.4f} {s['cps_ci']} coll={s['collision']:.4f} {s['collision_ci']} "
              f"empty={s['empty_step_frac']:.4f} sing={s['singular_step_frac']:.4f} wall={s['wall_s']}s", flush=True)
    # flips vs none
    results["flips_none_to_k5"] = flips(allrows["none"], allrows["kstep_k5"])
    results["flips_none_to_k10"] = flips(allrows["none"], allrows["kstep_k10"])
    # firing count per episode (empty steps), pool-wide, from k10 rows (empty_step_frac * active steps ~ proportional)
    json.dump(results, open(SP / "stage1_empty_fallback_eval.json", "w"), indent=2, default=str)
    print("\n=== FLIPS none->k10 ===", flush=True)
    for f in results["flips_none_to_k10"]:
        print(f"  ep {f['episode']}: {f['frm']} -> {f['to']} (had_empty={f['had_empty']})", flush=True)
    print(f"\nnone->k5 flips: {len(results['flips_none_to_k5'])}  none->k10 flips: {len(results['flips_none_to_k10'])}", flush=True)
    print("WROTE", SP / "stage1_empty_fallback_eval.json", flush=True)


if __name__ == "__main__":
    main()
