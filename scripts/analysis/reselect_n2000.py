"""n2000 checkpoint re-selection (read-only).

For a completed run, evaluate its top-K in-loop checkpoints on the frozen full pool at a fixed
goal_speed_radius and select the best by cps. Emits JSON + markdown with cps + scene-bootstrap
95% CI + full decomposition (reach/collision/infeasibility/stuck/timeout/oob) per checkpoint, and
the headline best vs the reported baselines. Nothing is trained or modified; the run directory is
read-only (results are written under --out).

Usage:
  python -m scripts.analysis.reselect_n2000 --run-dir RUN [--pool PATH] [--top-k 8]
         [--gsr 0.30] [--tag NAME] [--out DIR]
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

from src.eval.evaluate import evaluate
from src.eval.run_full import _load_framework

REPO = Path(__file__).resolve().parents[2]
DEFAULT_POOL = REPO / "data/secured_data/pools/eval_full_di_n2000_seed23456.pkl"
BASELINES = {"v2.3.0 (3-seed)": 0.8698, "v2.0.1": 0.8568}
KEYS = ("cps", "cps_ci_lo", "cps_ci_hi", "reach", "collision", "infeasibility",
        "stuck", "timeout", "oob")


def _top_k_steps(run_dir: Path, k: int) -> list[int]:
    rows = [r for r in csv.DictReader(open(run_dir / "eval_metrics.csv"))
            if r.get("mode") == "in_loop"]
    rows.sort(key=lambda r: float(r["cps"]), reverse=True)
    return [int(r["step"]) for r in rows[:k]]


def main() -> int:
    ap = argparse.ArgumentParser(description="n2000 checkpoint re-selection (read-only).")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--gsr", type=float, default=0.30)
    ap.add_argument("--steps", type=str, default=None, help="comma-separated steps (override auto top-k)")
    ap.add_argument("--tag", type=str, default="reselect_n2000")
    ap.add_argument("--out", type=Path, default=REPO / "docs/versions/v2.4.0")
    args = ap.parse_args()

    steps = ([int(s) for s in args.steps.split(",")] if args.steps
             else _top_k_steps(args.run_dir, args.top_k))
    rows = []
    for st in steps:
        ckpt = args.run_dir / f"checkpoints/step_{st:06d}.pt"
        framework, config, checkpoint = _load_framework(ckpt)
        cfg = copy.deepcopy(dict(config))
        cfg["env"] = dict(cfg["env"])
        cfg["env"]["goal_speed_radius"] = float(args.gsr)
        res = evaluate(framework, args.pool, cfg, mode="final", step=st,
                       ckpt_name=ckpt.name, include_lqr_baseline=False)
        er = res.eval_row
        row = {"step": st, **{k: float(er[k]) for k in KEYS if k in er}}
        rows.append(row)
        print(f"step {st:>6}  cps {row['cps']:+.4f} [{row['cps_ci_lo']:+.4f},{row['cps_ci_hi']:+.4f}] "
              f"reach {row['reach']:.3f} coll {row['collision']:.3f} infeas {row['infeasibility']:.3f} "
              f"stuck {row['stuck']:.3f} timeout {row['timeout']:.3f} oob {row.get('oob', 0):.3f}", flush=True)

    best = max(rows, key=lambda r: r["cps"])
    payload = {"run_dir": str(args.run_dir), "pool": str(args.pool), "gsr": args.gsr,
               "rows": rows, "best": best, "baselines": BASELINES}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.tag}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    md = [f"# n2000 re-selection — {args.tag}", "",
          f"run: `{args.run_dir}`  |  pool: `{args.pool.name}`  |  gsr={args.gsr}  |  SINGLE-SEED (seed 42)", "",
          "| step | cps | 95% CI | reach | coll | infeas | stuck | timeout | oob |",
          "|---:|---:|:--:|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(rows, key=lambda r: r["cps"], reverse=True):
        md.append(f"| {r['step']} | {r['cps']:+.4f} | [{r['cps_ci_lo']:+.4f}, {r['cps_ci_hi']:+.4f}] | "
                  f"{r['reach']:.3f} | {r['collision']:.3f} | {r['infeasibility']:.3f} | "
                  f"{r['stuck']:.3f} | {r['timeout']:.3f} | {r.get('oob', 0):.3f} |")
    md += ["", f"**Headline (best): step {best['step']} — cps {best['cps']:+.4f} "
               f"[{best['cps_ci_lo']:+.4f}, {best['cps_ci_hi']:+.4f}]** (single-seed).", "",
           "Baselines on the same frozen pool: " +
           ", ".join(f"{k} {v:.4f}" for k, v in BASELINES.items()) + "."]
    (args.out / f"{args.tag}.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"\nBEST step {best['step']} cps {best['cps']:+.4f} [{best['cps_ci_lo']:+.4f},{best['cps_ci_hi']:+.4f}]")
    print(f"OUT={args.out / f'{args.tag}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
