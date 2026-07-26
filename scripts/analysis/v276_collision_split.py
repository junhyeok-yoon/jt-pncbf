"""v2.7.6 — split the banded 'collision' outcome by SURFACE on the band-feasible pool: cylinder (xy obstacle)
vs band (|p_z|>=4), band further into floor (p_z<=-4) and ceiling (p_z>=+4). Per-episode: recompute the first
collision step from the recorded trajectory (rollouts continue through collision) and classify at that step
with the exact step_outcomes predicates. Counts + scene-bootstrap 95% CIs. Eval-only, read-only on the live
run dir; outputs elsewhere. Run for (b) pre-JT (OC a best) and the JT best (step 30000) before M5."""
from __future__ import annotations

import copy, json, sys
from pathlib import Path

import numpy as np
import torch

from src.eval.evaluate import evaluate
from src.eval.run_full import _load_framework as load_framework_from_checkpoint
from src.common.outcomes import _collided_exact

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
POOLS = REPO / "data/runs/v2.7.6/pools"
STEMS = {"bandfeasible": "eval_bandfeasible_quadrotor-3d-d2r_n2000_seed42",
         "fullrange": "eval_fullrange_quadrotor-3d-d2r_n2000_seed42"}


def classify(ckpt: Path, tag: str, outdir: Path, STEM: str):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    boot = ck["config"]["eval"]["bootstrap"]; n_boot = int(boot["n_resample"]); bseed = int(boot["seed"])
    over = {"env": {"dt": 0.05, "stuck_window_steps": 60, "stuck_radius": 0.10, "band_collision_limit": 4.0},
            "eval": {"max_steps": 200, "dt_ctrl": 0.05}, "filter": copy.deepcopy(ck["config"]["filter"])}
    over["filter"]["empty_fallback"] = {"mode": "none", "k": int(over["filter"].get("empty_fallback", {}).get("k", 10))}
    fw, cfg, _ = load_framework_from_checkpoint(ckpt, config_overrides=over)
    res = evaluate(fw, POOLS / f"{STEM}.pkl", cfg, mode="final", step=int(ck["step"]),
                   ckpt_name=ckpt.name, max_scenes=None, include_lqr_baseline=False)
    trs = res.trajectories
    n = len(trs)
    # per-episode one-hot indicators (sum over the disjoint labels == total collision indicator). Use the
    # eval's own resolved outcome + event step (filtered_outcome / filtered_event_step) so the split is exactly
    # consistent with the reported collision fraction; states are [T, B=1, 13] so squeeze the batch axis.
    cyl = np.zeros(n); bfloor = np.zeros(n); bceil = np.zeros(n); both = np.zeros(n); other = np.zeros(n)
    for i, tr in enumerate(trs):
        if str(tr.filtered_outcome) != "collision":
            continue
        st = torch.as_tensor(np.asarray(tr.filtered.states)[:, 0, :], dtype=torch.float64)  # [T, 13]
        ev = int(tr.filtered_event_step)
        pos_ev = st[ev, :3]
        cb = bool(_collided_exact(pos_ev, tr.scene))                 # cylinder at the collision step
        pz = float(pos_ev[2]); bb = abs(pz) >= 4.0                   # band at the collision step
        if bb and not cb:
            (bfloor if pz <= -4.0 else bceil)[i] = 1.0
        elif cb and not bb:
            cyl[i] = 1.0
        elif bb and cb:                                              # simultaneous: attribute to band, count 'both'
            (bfloor if pz <= -4.0 else bceil)[i] = 1.0; both[i] = 1.0
        else:                                                        # neither predicate at the eval's event step
            other[i] = 1.0

    rng = np.random.default_rng(bseed)
    idx = rng.integers(0, n, (n_boot, n))

    def stat(vec):
        m = float(vec.mean()); bs = vec[idx].mean(1)
        return {"count": int(vec.sum()), "frac": round(m, 5),
                "ci95": [round(float(np.percentile(bs, 2.5)), 5), round(float(np.percentile(bs, 97.5)), 5)]}

    band_total = bfloor + bceil
    total_coll = cyl + band_total + other
    out = {"tag": tag, "ckpt": str(ckpt), "step": int(ck["step"]), "stem": STEM, "n": n,
           "collision_total": stat(total_coll), "cylinder": stat(cyl), "band_total": stat(band_total),
           "band_floor": stat(bfloor), "band_ceiling": stat(bceil),
           "band_and_cylinder_same_step": int(both.sum()), "collision_unclassified": int(other.sum())}
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{tag}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"[{tag}] step {out['step']} n={n} | collision {out['collision_total']['frac']:.4f}"
          f"{out['collision_total']['ci95']} = cylinder {out['cylinder']['frac']:.4f}{out['cylinder']['ci95']}"
          f" + band {out['band_total']['frac']:.4f}{out['band_total']['ci95']}"
          f" (floor {out['band_floor']['count']}, ceiling {out['band_ceiling']['count']},"
          f" both-surfaces {out['band_and_cylinder_same_step']}, unclassified {out['collision_unclassified']})", flush=True)
    return out


if __name__ == "__main__":
    OUT = REPO / "data/runs/v2.7.6/stage2_eval/collision_split"
    OC_A = REPO / "data/runs/v2.7.6/set__20260725-043415__seed42/v2.7.6__oc__20260725-043415__seed42/checkpoints/best.pt"
    JT = REPO / "data/runs/v2.7.6/set__20260725-043415__seed42/v2.7.6__jt__20260725-052127__seed42/checkpoints/step_030000.pt"
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    poolkey = sys.argv[2] if len(sys.argv) > 2 else "bandfeasible"
    STEM = STEMS[poolkey]; sfx = "" if poolkey == "bandfeasible" else f"_{poolkey}"
    if which in ("both", "b"):
        classify(OC_A, f"b_preJT_OC_a_dim34{sfx}", OUT, STEM)
    if which in ("both", "jt"):
        classify(JT, f"jt_best_step30000{sfx}", OUT, STEM)
