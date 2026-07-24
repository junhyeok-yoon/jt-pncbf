"""v2.7.5 M5 — POST-HOC diagnostic analysis of the dt_ctrl arms (A1-A5, 17 tests).

Registered verdicts are FINAL and are not revised here (H1 rejected, H2 direction-only, H3 failed, H4 passed
on p95). Everything is recorded as an observed property. Single seed, EXPLORATORY. Reads only the recorded
artifacts regenerated at the pinned batch 2000 (all three arms passed the 1e-6 gate). Plan and thresholds are
fixed in m5_analysis_plan.json, written before any computation.
"""
from __future__ import annotations

import csv, json, math
from pathlib import Path

import numpy as np
import torch

from src.common.quadrotor_barrier import make_barrier_fn
from src.eval.build_pools import load_pool, EVAL_POOLS_DIR
from src.envs.scene_batch import batch_scenes
from src.eval.run_full import _load_framework

RD = Path("data/runs/v2.7.5/dt_ctrl_arms")
POOL = EVAL_POOLS_DIR / "eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl"
CKPT = Path("data/runs/v2.7.4/set__20260720-083533__seed42/v2.7.4__20260720-091830__seed42/checkpoints/best.pt")
PLAN = json.loads((RD / "m5_analysis_plan.json").read_text())
ARMS = ["A_20Hz_coarse", "B_20Hz_fine", "C_100Hz_fine"]
OUTC = ["goal", "collision", "timeout", "stuck", "oob"]
RNG = np.random.default_rng(20260723)
UBOX = 4.905

D = {}
for a in ARMS:
    z = np.load(RD / f"states_{a}.npz"); s = np.load(RD / f"action_stream_{a}.npz")
    rows = {int(r["episode_idx"]): r for r in csv.DictReader(open(RD / f"per_episode_{a}.csv"))}
    D[a] = {"S": z["states"], "iv": z["intervention"], "em": z["empty"], "dt": float(z["dt"]),
            "uc": s["u_cmd"], "un": s["u_nom"], "n": s["n_steps"], "rows": rows}
N = D["B_20Hz_fine"]["S"].shape[0]
ids = list(range(N))
oc = {a: {i: D[a]["rows"][i]["outcome"] for i in ids} for a in ARMS}
cps = {a: np.array([float(D[a]["rows"][i]["cps_episode"]) for i in ids]) for a in ARMS}
nst = {a: np.array([int(D[a]["n"][i]) for i in ids]) for a in ARMS}
R = {}

def boot_ci(v, n=4000):
    v = np.asarray(v, float)
    if v.size == 0: return [None, None]
    bs = np.array([v[RNG.integers(0, v.size, v.size)].mean() for _ in range(n)])
    return [round(float(np.percentile(bs, 2.5)), 6), round(float(np.percentile(bs, 97.5)), 6)]

# ---------------- A1 ----------------
tab = {ob: {oco: 0 for oco in OUTC} for ob in OUTC}
for i in ids: tab[oc["B_20Hz_fine"][i]][oc["C_100Hz_fine"][i]] += 1
R["T1_contingency_B_rows_C_cols"] = tab
R["T1_offdiagonal_total"] = int(sum(tab[a_][b_] for a_ in OUTC for b_ in OUTC if a_ != b_))

fwd = [i for i in ids if oc["B_20Hz_fine"][i] == "collision" and oc["C_100Hz_fine"][i] == "goal"]
rev = [i for i in ids if oc["C_100Hz_fine"][i] == "collision" and oc["B_20Hz_fine"][i] == "goal"]
b_, c_ = len(fwd), len(rev); n_disc = b_ + c_
# T2 exact binomial P(X >= b_ | n, 0.5), two-sided also reported
p_ge = sum(math.comb(n_disc, k) for k in range(b_, n_disc + 1)) / 2 ** n_disc
p_two = min(1.0, 2 * p_ge)
# T3/T4 McNemar
mc_cc = (abs(b_ - c_) - 1) ** 2 / (b_ + c_)
mc_nc = (b_ - c_) ** 2 / (b_ + c_)
from math import erfc
chi2_p = lambda x: erfc(math.sqrt(x / 2.0))          # 1 dof survival
R["T2_exact_binomial"] = {"forward": b_, "reverse": c_, "n_discordant": n_disc,
                          "P_X_ge_forward_one_sided": round(p_ge, 6), "two_sided": round(p_two, 6)}
R["T3_mcnemar_with_cc"] = {"chi2": round(mc_cc, 5), "p": round(chi2_p(mc_cc), 6)}
R["T4_mcnemar_no_cc"] = {"chi2": round(mc_nc, 5), "p": round(chi2_p(mc_nc), 6)}

d_cb = cps["C_100Hz_fine"] - cps["B_20Hz_fine"]
R["T5_paired_cps_C_minus_B"] = {"n": N, "mean": round(float(d_cb.mean()), 6),
                                "median": round(float(np.median(d_cb)), 6), "boot95_mean": boot_ci(d_cb),
                                "frac_positive": round(float((d_cb > 0).mean()), 5)}
d_ba = cps["B_20Hz_fine"] - cps["A_20Hz_coarse"]
R["T6_paired_cps_B_minus_A"] = {"n": N, "mean": round(float(d_ba.mean()), 6),
                                "median": round(float(np.median(d_ba)), 6), "boot95_mean": boot_ci(d_ba),
                                "frac_positive": round(float((d_ba > 0).mean()), 5)}

# ---------------- A2 ----------------
fw_, cfg, _ = _load_framework(CKPT)
system = fw_.system
h_star_fn = make_barrier_fn(float(cfg["env"][system.name]["c_gain"]), float(cfg["env"]["h_scale"]))
scenes = load_pool(POOL).scenes
TDIV = 0.10

def runs_of(mask):
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]: j += 1
            out.append((i, j)); i = j
        else: i += 1
    return out

def a2_group(group):
    per = []
    for i in group:
        kB = max(1, int(nst["B_20Hz_fine"][i])); kC = max(1, int(nst["C_100Hz_fine"][i]))
        emB = D["B_20Hz_fine"]["em"][i, :kB].astype(bool); emC = D["C_100Hz_fine"]["em"][i, :kC].astype(bool)
        pB = D["B_20Hz_fine"]["S"][i, :, 0:3]; pC = D["C_100Hz_fine"]["S"][i, :, 0:3]
        K = min(pB.shape[0], pC.shape[0])
        dd = np.linalg.norm(pC[:K] - pB[:K], axis=1)
        td = int(np.argmax(dd > TDIV)) if (dd > TDIV).any() else -1
        fe = int(np.argmax(emC)) if emC.any() else -1
        order = ("no_empty_in_C" if fe < 0 else ("no_divergence" if td < 0 else
                 ("empty_BEFORE_tdiv" if fe < td else ("empty_AFTER_tdiv" if fe > td else "same_step"))))
        # altitude rate at contact (C if C collided else B)
        arm = "C_100Hz_fine" if oc["C_100Hz_fine"][i] == "collision" else "B_20Hz_fine"
        kk = max(1, int(nst[arm][i])); vz = float(D[arm]["S"][i, min(kk, D[arm]["S"].shape[1] - 1), 9])
        per.append({"ep": i,
            "infeas_frac_B": round(float(emB.mean()), 5), "infeas_frac_C": round(float(emC.mean()), 5),
            "n_empty_intervals_B": len(runs_of(emB)), "n_empty_intervals_C": len(runs_of(emC)),
            "empty_dur_s_B": round(len(runs_of(emB)) and float(emB.sum()) * D["B_20Hz_fine"]["dt"] or 0.0, 4),
            "empty_dur_s_C": round(float(emC.sum()) * D["C_100Hz_fine"]["dt"], 4),
            "t_div_s": (round(td * D["C_100Hz_fine"]["dt"], 4) if td >= 0 else None),
            "t_first_empty_C_s": (round(fe * D["C_100Hz_fine"]["dt"], 4) if fe >= 0 else None),
            "ordering": order, "vz_at_contact": round(vz, 4), "contact_arm": arm})
    return per

per_rev, per_fwd = a2_group(rev), a2_group(fwd)
def summ(per, key):
    v = [p[key] for p in per if p[key] is not None]
    return {"n": len(v), "median": round(float(np.median(v)), 5) if v else None} if v else {"n": 0}
def order_counts(per):
    c = {}
    for p in per: c[p["ordering"]] = c.get(p["ordering"], 0) + 1
    return c
R["T7_infeasibility_fraction"] = {
  "reverse_flips": {"n": len(rev), "median_B": summ(per_rev, "infeas_frac_B")["median"], "median_C": summ(per_rev, "infeas_frac_C")["median"]},
  "forward_flips_CONTROL": {"n": len(fwd), "median_B": summ(per_fwd, "infeas_frac_B")["median"], "median_C": summ(per_fwd, "infeas_frac_C")["median"]}}
R["T8_empty_interval_structure"] = {
  "reverse_flips": {"median_n_intervals_B": summ(per_rev, "n_empty_intervals_B")["median"], "median_n_intervals_C": summ(per_rev, "n_empty_intervals_C")["median"],
                    "median_dur_s_B": summ(per_rev, "empty_dur_s_B")["median"], "median_dur_s_C": summ(per_rev, "empty_dur_s_C")["median"]},
  "forward_flips_CONTROL": {"median_n_intervals_B": summ(per_fwd, "n_empty_intervals_B")["median"], "median_n_intervals_C": summ(per_fwd, "n_empty_intervals_C")["median"],
                            "median_dur_s_B": summ(per_fwd, "empty_dur_s_B")["median"], "median_dur_s_C": summ(per_fwd, "empty_dur_s_C")["median"]}}
R["T9_ordering_empty_vs_tdiv"] = {"threshold_m": TDIV, "reverse_flips": order_counts(per_rev),
                                  "forward_flips_CONTROL": order_counts(per_fwd)}
R["T10_altitude_rate_at_contact"] = {
  "reverse_flips": {"n": len(rev), "median_vz": summ(per_rev, "vz_at_contact")["median"],
                    "frac_descending": round(float(np.mean([p["vz_at_contact"] < 0 for p in per_rev])), 4) if per_rev else None},
  "forward_flips_CONTROL": {"n": len(fwd), "median_vz": summ(per_fwd, "vz_at_contact")["median"],
                            "frac_descending": round(float(np.mean([p["vz_at_contact"] < 0 for p in per_fwd])), 4) if per_fwd else None}}

# ---------------- A3 ----------------
def tvps(arm, i, k0=None, k1=None, which="uc", mask=None):
    k = max(1, int(nst[arm][i])); dt = D[arm]["dt"]
    lo = 0 if k0 is None else k0
    hi = k if k1 is None else min(k, k1)
    if hi - lo < 2: return None
    seg = D[arm][which][i, lo:hi + 1].astype(np.float64)
    du = np.abs(np.diff(seg, axis=0))
    if mask is not None:
        m = mask[lo:hi][:du.shape[0]]
        if m.sum() < 1: return None
        return float(du[m].sum() / (m.sum() * dt * UBOX * 4))
    return float(du.sum() / ((hi - lo) * dt * UBOX * 4))

both_goal = [i for i in ids if oc["B_20Hz_fine"][i] == "goal" and oc["C_100Hz_fine"][i] == "goal"]
def med_p90(vals):
    v = np.array([x for x in vals if x is not None], float)
    return {"n": int(v.size), "median": round(float(np.median(v)), 5), "p90": round(float(np.percentile(v, 90)), 5)}
tB = med_p90([tvps("B_20Hz_fine", i) for i in both_goal]); tC = med_p90([tvps("C_100Hz_fine", i) for i in both_goal])
R["T11_tv_both_reached"] = {"B": tB, "C": tC,
    "pct_change_median": round(100 * (tC["median"] / tB["median"] - 1), 2),
    "pct_change_p90": round(100 * (tC["p90"] / tB["p90"] - 1), 2),
    "registered_full_pool_reference": {"median_pct": 40.5, "p90_pct": 95.5}}
cw = [min(int(nst["B_20Hz_fine"][i]), int(nst["C_100Hz_fine"][i])) for i in ids]
wB = med_p90([tvps("B_20Hz_fine", i, 0, cw[i]) for i in ids]); wC = med_p90([tvps("C_100Hz_fine", i, 0, cw[i]) for i in ids])
R["T12_tv_common_window"] = {"B": wB, "C": wC,
    "pct_change_median": round(100 * (wC["median"] / wB["median"] - 1), 2),
    "pct_change_p90": round(100 * (wC["p90"] / wB["p90"] - 1), 2)}
act, ina = {}, {}
for a in ("B_20Hz_fine", "C_100Hz_fine"):
    va, vi = [], []
    for i in ids:
        k = max(1, int(nst[a][i])); m = D[a]["iv"][i, :k].astype(bool)
        va.append(tvps(a, i, mask=m)); vi.append(tvps(a, i, mask=~m))
    act[a] = med_p90(va); ina[a] = med_p90(vi)
R["T13_tv_filter_active_vs_inactive"] = {"filter_ACTIVE": act, "filter_INACTIVE": ina}
nB = med_p90([tvps("B_20Hz_fine", i, which="un") for i in ids]); nC = med_p90([tvps("C_100Hz_fine", i, which="un") for i in ids])
cB = med_p90([tvps("B_20Hz_fine", i) for i in ids]); cC = med_p90([tvps("C_100Hz_fine", i) for i in ids])
R["T14_u_nom_vs_u_cmd_full_pool"] = {"u_cmd": {"B": cB, "C": cC, "pct_median": round(100*(cC["median"]/cB["median"]-1),2)},
                                     "u_nom": {"B": nB, "C": nC, "pct_median": round(100*(nC["median"]/nB["median"]-1),2)}}

# ---------------- A4 ----------------
HTH, SWING, MINS = 0.05, UBOX / 2.0, 1.0
def boundary_riding(arm):
    hits = []
    for i in ids:
        k = max(1, int(nst[arm][i])); dt = D[arm]["dt"]
        X = torch.tensor(D[arm]["S"][i, :k + 1], dtype=torch.float32)
        bs1 = batch_scenes([scenes[i]] * X.shape[0], device=torch.device("cpu"), dtype=torch.float32)
        with torch.no_grad(): hs = h_star_fn(X, bs1).numpy().reshape(-1)
        u = D[arm]["uc"][i, :k].astype(np.float64)
        sw = np.zeros(k, bool)
        if k >= 2: sw[:k-1] = (np.abs(np.diff(u, axis=0)).max(axis=1) > SWING)
        cond = D[arm]["iv"][i, :k].astype(bool) & (np.abs(hs[:k]) < HTH) & sw
        need = int(round(MINS / dt))
        if any((e - s) >= need for s, e in runs_of(cond)): hits.append(i)
    return hits
brB, brC = boundary_riding("B_20Hz_fine"), boundary_riding("C_100Hz_fine")
R["T15_boundary_riding_counts"] = {"detector": PLAN["stated_thresholds"]["A4_boundary_riding_detector"],
                                   "B": len(brB), "C": len(brC), "B_ids": brB[:40], "C_ids": brC[:40]}
def xtab(hits, arm):
    return {o: int(sum(1 for i in hits if oc[arm][i] == o)) for o in OUTC}
R["T16_boundary_riding_xtab_and_stuck"] = {
  "B_outcomes_of_riding_scenes": xtab(brB, "B_20Hz_fine"), "C_outcomes_of_riding_scenes": xtab(brC, "C_100Hz_fine"),
  "stuck_fraction_B": round(float(np.mean([oc["B_20Hz_fine"][i] == "stuck" for i in ids])), 5),
  "stuck_fraction_C": round(float(np.mean([oc["C_100Hz_fine"][i] == "stuck" for i in ids])), 5),
  "timeout_fraction_B": round(float(np.mean([oc["B_20Hz_fine"][i] == "timeout" for i in ids])), 5),
  "timeout_fraction_C": round(float(np.mean([oc["C_100Hz_fine"][i] == "timeout" for i in ids])), 5)}

# ---------------- A5 ----------------
from src.envs.quadrotor_3d import _quat_to_R
S0 = D["B_20Hz_fine"]["S"][:, 0, :]
tilt0 = np.degrees(np.arccos(np.clip(_quat_to_R(torch.tensor(S0[:, 3:7], dtype=torch.float64))[:, 2, 2].numpy(), -1, 1)))
spd0 = np.linalg.norm(S0[:, 7:10], axis=1)
def seg_pt(p0, p1, c):
    ab = p1 - p0; t = np.clip(np.dot(c - p0, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
    return float(np.linalg.norm(p0 + t * ab - c))
clr, blk = np.zeros(N), np.zeros(N, bool)
for i in ids:
    s = scenes[i]; cen = np.asarray(s.obstacle_centers, float); rad = np.asarray(s.obstacle_radii, float)
    a_ = np.asarray(s.obstacle_active, bool); st = np.asarray(s.start, float); gl = np.asarray(s.goal, float)
    if a_.any():
        clr[i] = float(np.min(np.linalg.norm(st[None, :2] - cen[a_][:, :2], axis=1) - rad[a_]))
        blk[i] = any(seg_pt(st[:2], gl[:2], cen[j, :2]) < rad[j] for j in np.nonzero(a_)[0])
    else: clr[i] = np.inf
noflip = [i for i in ids if i not in set(fwd) | set(rev)]
def grp(idx, name):
    idx = np.array(idx, int)
    return {"group": name, "n": int(idx.size),
            "tilt0_median": round(float(np.median(tilt0[idx])), 3), "tilt0_boot95": boot_ci(tilt0[idx]),
            "clearance_median": round(float(np.median(clr[idx])), 4), "clearance_boot95": boot_ci(clr[idx]),
            "blocked_frac": round(float(blk[idx].mean()), 4), "blocked_boot95": boot_ci(blk[idx].astype(float)),
            "speed0_median": round(float(np.median(spd0[idx])), 4), "speed0_boot95": boot_ci(spd0[idx])}
R["T17_subgroup_covariates"] = {"NOTE": "exploratory subgroup analysis on a SINGLE seed; CIs are on the mean",
                                "forward_flips": grp(fwd, "forward"), "reverse_flips": grp(rev, "reverse"),
                                "no_flip": grp(noflip, "no_flip")}

R["_meta"] = {"total_tests_planned": PLAN["total_test_count"], "total_tests_reported": 17,
              "n_scenes": N, "forward_flips": b_, "reverse_flips": c_}
(RD / "m5_posthoc_results.json").write_text(json.dumps(R, indent=2) + "\n")
json.dump({"reverse": per_rev, "forward": per_fwd}, open(RD / "m5_a2_per_scene.json", "w"), indent=1)
print(json.dumps({k: R[k] for k in R if k.startswith(("T1_", "T2", "T3", "T4", "T5", "T6", "_meta"))}, indent=2))
print("\nA2:", json.dumps({k: R[k] for k in ("T7_infeasibility_fraction","T9_ordering_empty_vs_tdiv","T10_altitude_rate_at_contact")}, indent=2))
print("\nA3:", json.dumps({k: R[k] for k in ("T11_tv_both_reached","T12_tv_common_window","T14_u_nom_vs_u_cmd_full_pool")}, indent=2))
print("\nA4:", json.dumps({k: R[k] for k in ("T15_boundary_riding_counts","T16_boundary_riding_xtab_and_stuck")}, indent=2)[:1200])
print("\nA5:", json.dumps(R["T17_subgroup_covariates"], indent=2))
