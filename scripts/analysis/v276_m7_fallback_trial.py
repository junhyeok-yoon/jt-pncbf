"""v2.7.6 M7 — kstep empty-branch fallback ADOPTION trial (eval-only), extending the v2.7.4 M7 driver
(quadrotor_3d_m7_fallback_trial.py). Checkpoint = the M5 selection JT step 42000 (sha 09c33bf4); deploy
settings = the M5 defaults; empty_fallback overridden in memory only. Per 02_control s4 the fallback does NOT
alter the infeasible flag or the infeasibility metric, so the -0.3*infeasibility term of the M5 legacy
full-range gap (-0.0197 of -0.0594) cannot move: at most -0.0397 is reachable by this arm.

Arms (pool seed 42): full-range REQUIRED, then band-feasible if wall-clock permits. Three arms, both scorings:
  JT mode=none (reproduction gate vs M5), JT mode=kstep k=5, v2.7.4 (244f4f83) mode=kstep k=5.
Per arm: full cps component breakdown + scene-bootstrap 95% CIs; collision split cylinder/band(floor/ceiling)
with counts; wall-clock. Flip anatomy (JT none-vs-kstep, both directions separately) cross-tabbed against the
episode empty-branch step count and ||L_g V_hat|| at t0, plus a chattering measure. Adoption rule A&B&C applied
MECHANICALLY (record which branch fired; no adjudication). eval-only; no git/securing/config promotion; no
protocol edit. Artifacts under <JT run-dir>/fallback_trial/ (05_code s3 run-id convention)."""
from __future__ import annotations

import copy, csv, json, sys, time
from pathlib import Path

import numpy as np
import torch

from src.common.filter_hardnet import _cbf_terms
from src.common.value_net import make_h_fn
from src.eval.bootstrap import within_seed_ci
from src.eval.evaluate import evaluate, EVAL_EPISODE_COLUMNS
from src.eval.build_pools import load_pool
from src.eval.run_full import _load_framework as load_fw
from src.common.outcomes import _collided_exact
from src.envs.scene_batch import batch_scenes, initial_states_from_batch

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
POOLS = REPO / "data/runs/v2.7.6/pools"
JT = REPO / "data/runs/v2.7.6/set__20260725-043415__seed42/v2.7.6__jt__20260725-052127__seed42"
JT42 = JT / "checkpoints/step_042000.pt"
V274 = REPO / "data/runs/v2.7.4/set__20260720-083533__seed42/v2.7.4__20260720-091830__seed42/checkpoints/best.pt"
OUT = JT / "fallback_trial"; OUT.mkdir(parents=True, exist_ok=True)
STEMS = {"fullrange": "eval_fullrange_quadrotor-3d-d2r_n2000_seed42",
         "bandfeasible": "eval_bandfeasible_quadrotor-3d-d2r_n2000_seed42"}
# M5 reference (reproduction gate): JT step 42000, full-range, empty_fallback=none
M5_REF = {"fullrange": {"legacy": 0.798637, "banded": 0.680976}}
COMPS = ("cps", "reach", "collision", "oob", "stuck", "timeout", "infeasibility")
REPRO_TOL = 1e-4


def _filter_for(ck, mode):
    filt = copy.deepcopy(ck["config"]["filter"])
    kdef = int(filt.get("empty_fallback", {}).get("k", 10))
    filt["empty_fallback"] = {"mode": "none", "k": kdef} if mode == "none" else {"mode": "kstep", "k": 5}
    return filt


def _split(res, n_boot, bseed):
    trs = res.trajectories; n = len(trs)
    cyl = np.zeros(n); bfloor = np.zeros(n); bceil = np.zeros(n); both = 0
    for i, tr in enumerate(trs):
        if str(tr.filtered_outcome) != "collision":
            continue
        _s = tr.filtered.states
        _s = _s.detach().cpu().numpy() if isinstance(_s, torch.Tensor) else np.asarray(_s)  # device-agnostic
        st = torch.as_tensor(_s[:, 0, :], dtype=torch.float64)
        ev = int(tr.filtered_event_step); pz = float(st[ev, 2])
        cb = bool(_collided_exact(st[ev, :3], tr.scene)); bb = abs(pz) >= 4.0
        if bb and not cb:
            (bfloor if pz <= -4 else bceil)[i] = 1.0
        elif cb and not bb:
            cyl[i] = 1.0
        elif bb and cb:
            (bfloor if pz <= -4 else bceil)[i] = 1.0; both += 1
    rng = np.random.default_rng(bseed); idx = rng.integers(0, n, (n_boot, n))
    def st_(v): return {"count": int(v.sum()), "frac": round(float(v.mean()), 5),
                        "ci95": [round(float(np.percentile(v[idx].mean(1), 2.5)), 5),
                                 round(float(np.percentile(v[idx].mean(1), 97.5)), 5)]}
    return {"cylinder": st_(cyl), "band_floor": st_(bfloor), "band_ceiling": st_(bceil),
            "band_total": st_(bfloor + bceil), "both_surfaces_same_step": both}


def run_arm(ckpt, pool_key, mode, scoring, tag):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    boot = ck["config"]["eval"]["bootstrap"]; n_boot = int(boot["n_resample"]); bseed = int(boot["seed"])
    bc = 0.0 if scoring == "legacy" else 4.0
    over = {"env": {"dt": 0.05, "stuck_window_steps": 60, "stuck_radius": 0.10, "band_collision_limit": bc},
            "eval": {"max_steps": 200, "dt_ctrl": 0.05}, "filter": _filter_for(ck, mode)}
    fw, cfg, _ = load_fw(ckpt, config_overrides=over)
    t0 = time.perf_counter()
    res = evaluate(fw, POOLS / f"{STEMS[pool_key]}.pkl", cfg, mode="final", step=int(ck["step"]),
                   ckpt_name=Path(ckpt).name, max_scenes=None, include_lqr_baseline=False)
    wall = round(time.perf_counter() - t0, 1)
    rows = list(res.episode_rows)
    ci = within_seed_ci(rows, n_resample=n_boot, seed=bseed)
    agg = {"tag": tag, "arm": mode, "scoring": scoring, "pool": pool_key, "n": len(rows), "wall_s": wall,
           "step": int(ck["step"])}
    for m in COMPS:
        agg[m] = round(float(ci["mean"][m]), 6); agg[m + "_ci"] = [round(float(ci["ci"][m]["lo"]), 6),
                                                                   round(float(ci["ci"][m]["hi"]), 6)]
    agg["split"] = _split(res, n_boot, bseed)
    with (OUT / f"episodes_{tag}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EVAL_EPISODE_COLUMNS); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in EVAL_EPISODE_COLUMNS})
    print(f"[{tag}] {mode}/{scoring}/{pool_key} cps {agg['cps']:+.5f} {agg['cps_ci']} reach {agg['reach']:.4f} "
          f"coll {agg['collision']:.4f} infeas {agg['infeasibility']:.4f} band {agg['split']['band_total']['frac']:.4f} "
          f"cyl {agg['split']['cylinder']['frac']:.4f} wall {wall}s", flush=True)
    return agg, rows


def lg0_at_t0(ckpt, pool_key, n):
    fw, cfg, _ = load_fw(ckpt)
    system = fw.system; h_fn = make_h_fn(fw.value_net, system)
    scenes = load_pool(POOLS / f"{STEMS[pool_key]}.pkl").scenes[:n]
    bs = batch_scenes(scenes, device=torch.device("cpu"), dtype=torch.float32)
    x0 = system.wrap_state(initial_states_from_batch(bs).float())
    u0 = torch.zeros(x0.shape[0], int(system.action_dim))
    _, _, lg = _cbf_terms(system, h_fn, x0, bs, u0, create_graph=False)
    return torch.linalg.norm(lg, dim=1).detach().numpy()


def flip_anatomy(rows_none, rows_kstep, lg0, scoring):
    none = {int(r["episode_idx"]): r for r in rows_none}
    kst = {int(r["episode_idx"]): r for r in rows_kstep}
    ids = sorted(set(none) & set(kst))
    def ecount(r): return round(float(r["empty_step_frac"]) * float(r["n_steps"]))
    emp = {i: ecount(none[i]) for i in ids}
    flips = [i for i in ids if none[i]["outcome"] != kst[i]["outcome"]]
    # direction by cps_episode change (recovered = kstep better; newly_lost = kstep worse)
    recovered = [i for i in flips if float(kst[i]["cps_episode"]) > float(none[i]["cps_episode"])]
    newly_lost = [i for i in flips if float(kst[i]["cps_episode"]) < float(none[i]["cps_episode"])]
    unflipped = [i for i in ids if i not in flips]
    def dist(idl):
        if not idl: return {"n": 0}
        e = np.array([emp[i] for i in idl]); lg = np.array([lg0[i] for i in idl])
        return {"n": len(idl), "empty_count_median": float(np.median(e)), "empty_count_mean": round(float(e.mean()), 2),
                "empty_count_p90": float(np.percentile(e, 90)), "frac_with_empty": round(float((e > 0).mean()), 3),
                "lg0_median": round(float(np.median(lg)), 4), "lg0_lo_authority_frac": round(float((lg < 1e-3).mean()), 3)}
    # clause C: flipped empty-count distribution above unflipped (median test + Mann-Whitney one-sided)
    fe = np.array([emp[i] for i in flips]); ue = np.array([emp[i] for i in unflipped])
    try:
        from scipy.stats import mannwhitneyu
        mw = float(mannwhitneyu(fe, ue, alternative="greater").pvalue) if len(fe) and len(ue) else float("nan")
    except Exception:
        mw = float("nan")
    c_above = bool(len(fe) and len(ue) and np.median(fe) > np.median(ue))
    return {"scoring": scoring, "n_flips": len(flips), "recovered": dist(recovered), "newly_lost": dist(newly_lost),
            "flipped_all": dist(flips), "unflipped": dist(unflipped),
            "clauseC_flipped_median_above_unflipped": c_above, "mannwhitney_greater_p": round(mw, 5),
            "recovered_ids": recovered[:40], "newly_lost_ids": newly_lost[:40]}


def chatter(ckpt, pool_key, mode, roll_n):
    from src.common.rk4 import rk4_step
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    over = {"filter": _filter_for(ck, mode)}
    fw, cfg, _ = load_fw(ckpt, config_overrides=over)
    system = fw.system; dt = float(cfg["env"]["dt"]); max_steps = int(cfg["eval"]["max_steps"])
    scenes = load_pool(POOLS / f"{STEMS[pool_key]}.pkl").scenes[:roll_n]
    bs = batch_scenes(scenes, device=torch.device("cpu"), dtype=torch.float32)
    x = system.wrap_state(initial_states_from_batch(bs).float())
    prev = None; empty = 0; du_s = 0.0; du_c = 0
    with torch.no_grad():
        for _ in range(max_steps):
            un = fw.policy(x, bs); u, _ = fw.filter(x, un, bs)
            em = fw._filter.last_empty.bool(); empty += int(em.sum())
            if prev is not None and bool(em.any()):
                du = torch.linalg.norm(u[em] - prev[em], dim=1); du_s += float(du.sum()); du_c += int(em.sum())
            prev = u; x = rk4_step(system, x, u, dt)
    return {"empty_steps": empty, "mean_du_on_empty": round(du_s / max(1, du_c), 5)}


if __name__ == "__main__":
    pools = sys.argv[1].split(",") if len(sys.argv) > 1 else ["fullrange", "bandfeasible"]
    report = {"ckpt_jt": str(JT42), "ckpt_v274": str(V274), "sha_jt": "09c33bf4", "scope_limit":
              "02_control s4: fallback does not change the infeasible flag / infeasibility metric; -0.3*infeas term "
              "(-0.0197 of the -0.0594 M5 legacy full-range gap) cannot move -> at most -0.0397 reachable.",
              "clause_B_interpretation": "kstep banded cps NOT CI-separated below none banded cps on EVERY pool run.",
              "clause_C_pool": "full-range, banded outcomes (the pool carrying the M5 legacy regression).",
              "pools": {}}
    store = {}
    for pk in pools:
        arms = {}
        for (ck, arm, mode) in [(JT42, "jt_none", "none"), (JT42, "jt_kstep", "kstep"), (V274, "v274_kstep", "kstep")]:
            arms[arm] = {}
            for scoring in ("legacy", "banded"):
                agg, rows = run_arm(ck, pk, mode, scoring, f"{arm}_{scoring}_{pk}")
                arms[arm][scoring] = {"agg": agg, "rows": rows}
        repro = {}
        if pk in M5_REF:
            for sc in ("legacy", "banded"):
                got = arms["jt_none"][sc]["agg"]["cps"]; ref = M5_REF[pk][sc]
                repro[sc] = {"got": got, "m5_ref": ref, "delta": round(got - ref, 8), "ok": bool(abs(got - ref) <= REPRO_TOL)}
            if not all(repro[sc]["ok"] for sc in repro):
                report["pools"][pk] = {"repro_gate": repro, "HALT": "none arm does not reproduce M5; harness changed"}
                (OUT / "m7_fallback_trial.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
                print("HALT: reproduction gate failed", flush=True); raise SystemExit(1)
        lg0 = lg0_at_t0(JT42, pk, arms["jt_none"]["legacy"]["agg"]["n"])
        flips = {sc: flip_anatomy(arms["jt_none"][sc]["rows"], arms["jt_kstep"][sc]["rows"], lg0, sc) for sc in ("banded", "legacy")}
        ch = {"none": chatter(JT42, pk, "none", 600), "kstep": chatter(JT42, pk, "kstep", 600)}
        store[pk] = {"arms": arms, "flips": flips}
        report["pools"][pk] = {"repro_gate": repro,
            "arms": {a: {sc: arms[a][sc]["agg"] for sc in arms[a]} for a in arms},
            "flip_anatomy": flips, "chatter": ch,
            "wall_s_total": round(sum(arms[a][sc]["agg"]["wall_s"] for a in arms for sc in arms[a]), 1)}
        (OUT / "m7_fallback_trial.json").write_text(json.dumps(report, indent=2, default=str) + "\n")

    # ---- adoption rule A&B&C (mechanical), evaluated across the pools run ----
    A = B = C = None; notes = {}
    if "fullrange" in store:
        jl = store["fullrange"]["arms"]["jt_kstep"]["legacy"]["agg"]; vl = store["fullrange"]["arms"]["v274_kstep"]["legacy"]["agg"]
        A = not (jl["cps_ci"][1] < vl["cps_ci"][0])
        notes["A"] = {"jt_kstep_legacy_cps": jl["cps"], "jt_kstep_legacy_ci": jl["cps_ci"],
                      "v274_kstep_legacy_cps": vl["cps"], "v274_kstep_legacy_ci": vl["cps_ci"]}
        C = bool(store["fullrange"]["flips"]["banded"]["clauseC_flipped_median_above_unflipped"])
        notes["C"] = {"flipped_empty_median": store["fullrange"]["flips"]["banded"]["flipped_all"].get("empty_count_median"),
                      "unflipped_empty_median": store["fullrange"]["flips"]["banded"]["unflipped"].get("empty_count_median"),
                      "mannwhitney_greater_p": store["fullrange"]["flips"]["banded"]["mannwhitney_greater_p"]}
    Bp = {}
    for pk in store:
        kb = store[pk]["arms"]["jt_kstep"]["banded"]["agg"]; nb = store[pk]["arms"]["jt_none"]["banded"]["agg"]
        Bp[pk] = not (kb["cps_ci"][1] < nb["cps_ci"][0])
    B = all(Bp.values()) if Bp else None
    notes["B"] = {"per_pool": Bp}
    fired = bool(A and B and C)
    report["adoption_rule"] = {"A_legacy_fullrange_not_sep_below_v274kstep": A, "B_banded_kstep_not_sep_below_none_all_pools": B,
                               "C_flips_concentrate_in_empty_branch_fullrange": C, "FIRES_A_and_B_and_C": fired,
                               "failed_clauses": [n for n, v in (("A", A), ("B", B), ("C", C)) if not v], "detail": notes}
    print(f"\nADOPTION A={A} B={B} C={C} -> FIRES={fired}  failed={[n for n,v in (('A',A),('B',B),('C',C)) if not v]}", flush=True)
    (OUT / "m7_fallback_trial.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print("M7 done ->", OUT / "m7_fallback_trial.json", flush=True)
