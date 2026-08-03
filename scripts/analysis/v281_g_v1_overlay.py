"""v2.8.1 S1 V1 (part 2) — in-loop metric overlays vs step for the two matched OC value runs.

Overlays cps / reach / collision / infeasibility against step for v2.7.6 043415 (hard) and v2.8.1 (soft),
straight from each run's eval_metrics.csv. The matched-contour side-by-side is produced by
v281_g_v1v3_contours.py; this is the trajectory-of-training comparison that answers whether the two OC runs
differ at all as objects, or only in the remembered sharpness of a single contour."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/home/junhyeok/MIT/jt-pncbf/data/runs/v2.8.1/s1_diagnostics")


def load(csv_path):
    rows = [r for r in csv.DictReader(open(csv_path)) if r["mode"] == "in_loop"]
    step = [int(r["step"]) for r in rows]
    return {"step": step, "cps": [float(r["cps"]) for r in rows], "reach": [float(r["reach"]) for r in rows],
            "collision": [float(r["collision"]) for r in rows], "infeasibility": [float(r["infeasibility"]) for r in rows]}


def main(ref_csv, new_csv, ref_label, new_label):
    A = load(ref_csv); B = load(new_csv)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, key in zip(axes.ravel(), ["cps", "reach", "collision", "infeasibility"]):
        ax.plot(A["step"], A[key], "-o", ms=3, label=f"{ref_label} (hard)", color="0.35")
        ax.plot(B["step"], B[key], "-o", ms=3, label=f"{new_label} (soft)", color=(0.0, 0.31, 0.72))
        ax.set_title(key); ax.set_xlabel("step"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("v2.8.1 S1 V1 — matched OC in-loop metrics vs step (n500 inloop pool)")
    fig.tight_layout()
    p = OUT / "v1_inloop_overlay.png"; fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    # final-step comparison
    print(f"final {ref_label}: cps={A['cps'][-1]:.4f} reach={A['reach'][-1]:.3f} coll={A['collision'][-1]:.3f} @step{A['step'][-1]}")
    print(f"final {new_label}: cps={B['cps'][-1]:.4f} reach={B['reach'][-1]:.3f} coll={B['collision'][-1]:.3f} @step{B['step'][-1]}")
    print(f"wrote {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-csv", required=True)
    ap.add_argument("--new-csv", required=True)
    ap.add_argument("--ref-label", default="043415")
    ap.add_argument("--new-label", default="v2.8.1")
    a = ap.parse_args()
    main(a.ref_csv, a.new_csv, a.ref_label, a.new_label)
