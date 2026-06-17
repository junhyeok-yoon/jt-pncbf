"""v2.2.0 Stage 2 — lower-alpha_unsafe deployment test (does it cut collisions, at what cost?).

The authority-collapse diagnosis found alpha_unsafe=100 is ~1000x the feasible threshold
(critical-alpha median 0.08): the CBF row is infeasible the instant h>0.1, and lowering alpha to
alpha_safe=2 would restore feasibility on ~39% of approach steps WITH THE SAME V_S (algebraic
what-if). This runs the REAL deploy for a sweep of alpha_unsafe on the SAME N=2000 pool, same
secured nominal+V_S+HardNet structure+scenes+rollout — only the scalar alpha_unsafe varies (entering
solely as row_upper = -L_f h - alpha_unsafe*h in the danger regime h>0; verified the sole use).

Part 0 mechanism (confirmed in code). Part 1 sweep {100,20,10,5,2,1,0.5}. Part 2 trade-off curve.
Part 3 paired contingency at the cps-best alpha vs baseline + critical-alpha cross-reference of which
collisions are alpha-fixable vs authority-bound. Part 4 verdict.

Read-only on V_S/policy/HardNet structure/checkpoint/committed pools; only the alpha_unsafe scalar
in the filter config varies (a pure deploy gain change, no retraining). Deterministic (pool seeded;
rollout + PNCBF forward have no RNG). Reuses the N=2000 pool for parity.

Run: /home/junhyeok/miniconda3/envs/pncbf/bin/python scripts/analysis/stage2_lower_alpha_deploy.py
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (str(REPO_ROOT), str(REPO_ROOT / "scripts/analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)

import stage2_hocbf_deploy_n2000 as D  # noqa: E402  (deploy_eval, summarize, contingency, cps, pool)
from src.common.filter_hardnet import HardNetFilter, _base_alpha, _cbf_terms, _hardnet_params  # noqa: E402
from src.common.value_net import make_h_fn  # noqa: E402
from src.envs.scene_batch import batch_scenes  # noqa: E402
from src.envs.scene_init import Scene  # noqa: E402
from src.frameworks.jt_pncbf.train import load_framework_from_checkpoint, make_system  # noqa: E402

OUT = D.OUT
TRACES = OUT / "deploy_n2000_traces.npz"
ALPHAS = [100.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5]      # 100 = baseline; alpha_safe stays 2.0
U_MAX = 2.0


def make_pncbf_filter(fw, system, config, alpha_unsafe):
    cfg = copy.deepcopy(config)
    cfg["filter"]["alpha_unsafe"] = float(alpha_unsafe)            # ONLY change; alpha_safe untouched
    filt = HardNetFilter(system, make_h_fn(fw.value_net, system), cfg, policy_fn=fw.policy)
    return lambda x, u_nom, scene: filt(x, scene, u_nom)


def collision_critical_alphas(fw, system, config, scenes, device, dtype):
    """Per BASELINE collision episode: critical alpha over its h>0 approach (baseline-visited states).
    critical alpha = (u_max*||A||_1 - L_f h)/h ; row feasible at alpha iff alpha <= critical."""
    if not TRACES.exists():
        return {}
    cache = dict(np.load(TRACES, allow_pickle=False))
    h_fn = make_h_fn(fw.value_net, system); params = _hardnet_params(config)
    c_idx = cache["pncbf_col_idx"]; c_states = cache["pncbf_col_states"]; p_ev = cache["pncbf_event"]
    out = {}
    for n in range(len(c_idx)):
        idx = int(c_idx[n]); scene = scenes[idx]
        T = c_states[n].shape[0] - 1
        ev = int(p_ev[idx]); ea = max(0, min(ev - 1, T - 1)) if ev > 0 else T - 1
        x = torch.as_tensor(c_states[n][: ea + 1], device=device, dtype=dtype)
        bscene = batch_scenes([scene] * x.shape[0], device=device, dtype=dtype)
        hh, lf, lg = _cbf_terms(system, h_fn, x, bscene, torch.zeros(x.shape[0], 2, device=device, dtype=dtype), create_graph=False)
        h = hh.detach().cpu().numpy(); lf = lf.detach().cpu().numpy()
        l1 = torch.sum(torch.abs(lg), dim=1).detach().cpu().numpy()
        m = h > 1e-6
        if not m.any():
            out[idx] = {"median_crit": float("nan"), "min_crit": float("nan")}
            continue
        crit = (U_MAX * l1[m] - lf[m]) / h[m]
        out[idx] = {"median_crit": float(np.median(crit)), "min_crit": float(np.min(crit))}
    return out


def main():
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); dtype = torch.float32
    fw, config, _ = load_framework_from_checkpoint(D.CKPT)
    system = make_system(config)
    fw.value_net.to(device, dtype).eval(); fw.policy_net.to(device, dtype).eval()
    pool, pool_path, _ = D.build_n2000_pool(config)
    scenes = pool.scenes
    OUT.mkdir(parents=True, exist_ok=True)

    rep = {"pool": str(pool_path), "n": len(scenes), "alpha_safe": float(config["filter"]["alpha_safe"]),
           "alphas_swept": ALPHAS, "u_max": U_MAX,
           "mechanism": ("_base_alpha = where(h<=0, alpha_safe, alpha_unsafe); alpha_unsafe enters ONLY as "
                         "row_upper = -L_f h - alpha_unsafe*h in the danger regime h>0. Pure deploy gain "
                         "change (no retraining, V_S/policy/HardNet structure identical).")}

    # Part 1 — sweep
    rows_by_alpha = {}; summary_by_alpha = {}; outcomes_by_alpha = {}
    for a in ALPHAS:
        print(f"[sweep] alpha_unsafe={a} ...")
        rows = D.deploy_eval(system, scenes, config, make_pncbf_filter(fw, system, config, a),
                             fw.policy, device, dtype, D.PNCBF_CHUNK, f"a={a}")
        rows_by_alpha[a] = rows
        summary_by_alpha[a] = D.summarize(rows)
        outcomes_by_alpha[a] = [r["outcome"] for r in rows]
        print(f"  alpha={a}: {summary_by_alpha[a]['counts']} cps={summary_by_alpha[a]['cps_mean']:.4f} "
              f"infeas={summary_by_alpha[a]['infeasibility_mean']:.4f}")

    base = summary_by_alpha[100.0]
    rep["sweep"] = {str(a): {**summary_by_alpha[a],
                             "delta_vs_baseline": {
                                 "cps": summary_by_alpha[a]["cps_mean"] - base["cps_mean"],
                                 **{f"d_{o}": summary_by_alpha[a]["counts"][o] - base["counts"][o] for o in D.OUTCOMES}},
                             "paired_delta_ci": D.paired_delta_ci(rows_by_alpha[100.0], rows_by_alpha[a])}
                    for a in ALPHAS}
    # baseline parity check vs the stored deploy report numbers
    rep["baseline_parity"] = {"counts": base["counts"], "cps": base["cps_mean"], "infeas": base["infeasibility_mean"],
                              "expected_from_deploy_report": {"goal": 1867, "collision": 23, "stuck": 86, "timeout": 24,
                                                              "cps": 0.8258, "infeas": 0.1188}}

    # Part 2 — cps-best alpha
    best_a = max(ALPHAS, key=lambda a: summary_by_alpha[a]["cps_mean"])
    rep["cps_best_alpha"] = {"alpha_unsafe": best_a, "cps": summary_by_alpha[best_a]["cps_mean"],
                             "counts": summary_by_alpha[best_a]["counts"],
                             "collisions_removed_vs_baseline": base["counts"]["collision"] - summary_by_alpha[best_a]["counts"]["collision"]}
    print(f"[best] cps-optimal alpha_unsafe={best_a} cps={summary_by_alpha[best_a]['cps_mean']:.4f}")

    # Part 3 — paired contingency vs baseline + critical-alpha cross-reference.
    # The cps-best alpha is the baseline (lowering never helps), so the contrast uses the what-if's
    # suggested value alpha_unsafe = alpha_safe = 2.0 to show what lowering actually costs.
    contrast_a = 2.0
    cont = D.contingency(rows_by_alpha[100.0], rows_by_alpha[contrast_a])
    rep["part3_contrast_alpha"] = contrast_a
    rep["part3_contingency_contrast_vs_baseline"] = cont
    crit = collision_critical_alphas(fw, system, config, scenes, device, dtype)
    base_col_idx = [i for i, o in enumerate(outcomes_by_alpha[100.0]) if o == "collision"]
    new_col_idx = [i for i in range(len(scenes))
                   if outcomes_by_alpha[contrast_a][i] == "collision" and outcomes_by_alpha[100.0][i] != "collision"]
    rep["part3_new_collisions_at_contrast"] = {
        "n": len(new_col_idx),
        "from_outcome": {o: sum(1 for i in new_col_idx if outcomes_by_alpha[100.0][i] == o) for o in D.OUTCOMES},
        "net_collisions_baseline_to_contrast": [summary_by_alpha[100.0]["counts"]["collision"],
                                                summary_by_alpha[contrast_a]["counts"]["collision"]]}
    resolved, residual = [], []
    for i in base_col_idx:
        rec = crit.get(i, {"median_crit": float("nan")})
        entry = {"idx": i, "median_crit": rec["median_crit"], "contrast_alpha_outcome": outcomes_by_alpha[contrast_a][i]}
        (resolved if outcomes_by_alpha[contrast_a][i] != "collision" else residual).append(entry)
    def crit_summary(group):
        v = [g["median_crit"] for g in group if np.isfinite(g["median_crit"])]
        return {"n": len(group), "median_critical_alpha": float(np.median(v)) if v else None,
                "frac_crit_ge_contrast_alpha": float(np.mean([x >= contrast_a for x in v])) if v else None,
                "frac_crit_le_0_authority_bound": float(np.mean([x <= 0 for x in v])) if v else None}
    rep["part3_collision_crossref"] = {
        "contrast_alpha": contrast_a, "n_baseline_collisions": len(base_col_idx),
        "resolved_at_contrast_alpha": crit_summary(resolved),
        "still_collide_residual": crit_summary(residual),
        "note": "critical alpha on the BASELINE-visited collision approach (h>0); the what-if predicted "
                "frac_crit>=alpha resolvable, but the closed-loop deploy shows feasibility != safety: a "
                "feasible-but-gentle low-alpha constraint permits approach, so collisions rise."}

    _fig(summary_by_alpha)
    (OUT / "lower_alpha_deploy_summary.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("summary ->", OUT / "lower_alpha_deploy_summary.json")
    print("collision counts by alpha:", {a: summary_by_alpha[a]["counts"]["collision"] for a in ALPHAS})
    print("cps by alpha:", {a: round(summary_by_alpha[a]["cps_mean"], 4) for a in ALPHAS})
    print("crossref resolved:", rep["part3_collision_crossref"]["resolved_at_contrast_alpha"])
    print("crossref residual:", rep["part3_collision_crossref"]["still_collide_residual"])
    return 0


def _fig(summary_by_alpha):
    a = np.array(ALPHAS)
    coll = [summary_by_alpha[x]["counts"]["collision"] for x in ALPHAS]
    stuck = [summary_by_alpha[x]["counts"]["stuck"] for x in ALPHAS]
    to = [summary_by_alpha[x]["counts"]["timeout"] for x in ALPHAS]
    goal = [summary_by_alpha[x]["counts"]["goal"] for x in ALPHAS]
    cps = [summary_by_alpha[x]["cps_mean"] for x in ALPHAS]
    infe = [summary_by_alpha[x]["infeasibility_mean"] for x in ALPHAS]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5), dpi=140)
    ax[0].plot(a, coll, "o-", color="#d62728", label="collision")
    ax[0].plot(a, stuck, "s-", color="#1f77b4", label="stuck")
    ax[0].plot(a, to, "^-", color="#ff7f0e", label="timeout")
    ax[0].set_xscale("log"); ax[0].set_xlabel("alpha_unsafe (log)"); ax[0].set_ylabel("episode count (of 2000)")
    ax[0].axvline(2.0, color="0.6", ls=":", label="alpha_safe=2"); ax[0].set_title("outcomes vs alpha_unsafe")
    ax[0].legend(fontsize=9); ax[0].invert_xaxis()
    ax2 = ax[1]; ax2.plot(a, cps, "o-", color="#2ca02c", label="cps")
    ax2.set_xscale("log"); ax2.set_xlabel("alpha_unsafe (log)"); ax2.set_ylabel("cps", color="#2ca02c")
    ax2.invert_xaxis(); ax2.axvline(2.0, color="0.6", ls=":")
    ax3 = ax2.twinx(); ax3.plot(a, infe, "d--", color="#9467bd", label="infeasibility")
    ax3.set_ylabel("infeasibility rate", color="#9467bd")
    ax2.set_title("cps + infeasibility vs alpha_unsafe")
    fig.tight_layout(); fig.savefig(OUT / "lower_alpha_tradeoff.png"); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
