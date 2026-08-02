"""v2.8.0 Phase-2 C2/C3/C4/C6 — D1/D2/D3/D5 from the persisted C1 streams. No GPU.

D1 (C2, attribution): counterfactual refilter — recompute the geometric projection F(u_nom,a,b) at
  (u_nom_t, a_{t-1}, b_{t-1}) [policy channel] and (u_nom_{t-1}, a_t, b_t) [certificate channel]; the cross
  term is the residual. TV shares by source: policy / certificate / cross (on feasible->feasible steps),
  empty-branch (both-empty steps), branch_transition (X2: empty flag differs t-1->t). F = _base_projection +
  _select_projection (the same functions the filter uses), on CPU.
D2 (C3, amplifier): 1/||a_I||, a_I = a on box-UNclipped coords; stats by S1 Jacobian class (|A|=clipped count)
  and by branch; Spearman rank-corr with ||Δu||; TV share of the top amp decile; ||a|| vs ||a_I||.
D3 (C4, crossings): top-k order changes between steps (within-set swap vs k-th exchange); ||Δu||,||Δa||,|Δb|
  at crossing vs non-crossing; TV share crossing steps carry.
D5 (C6, spectrum): per-rotor command PSD, power fraction above 5 Hz and 10 Hz per cell.

Sidecar: data/runs/v2.8.0/c1/d_analysis.json"""
from __future__ import annotations
import copy, json
from pathlib import Path
import numpy as np, torch
from scipy.stats import spearmanr

from src.common.filter_hardnet import _base_projection, _select_projection, _hardnet_params

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
CK = REPO / "data/runs/v2.8.0/set__20260731-015517__seed42/v2.8.0__jt__20260731-015517__seed42/checkpoints/best.pt"
C1 = REPO / "data/runs/v2.8.0/c1"
CELLS = [(r, p) for p in ("dual_solve", "enumerate") for r in (20, 100, 500)]
_CKCFG = torch.load(str(CK), map_location="cpu", weights_only=False)["config"]


def params_for(proj, dt_sim):
    cfg = copy.deepcopy(_CKCFG)
    cfg["filter"]["projection"] = proj
    cfg["filter"]["empty_fallback"] = {"mode": "kstep", "phases": 1, "k": 3}
    cfg.setdefault("env", {})["dt"] = float(dt_sim)
    cfg.setdefault("run", {})["system"] = "quadrotor_3d"
    return _hardnet_params(cfg)


def F(u_nom, a, b, bounds, params, proj):
    base = _base_projection(u_nom, a, b, bounds, params)
    box, _ = _select_projection(proj, u_nom, base, a, b, bounds)
    return box


def pairs(offsets, active):
    prev, cur = [], []
    for e in range(len(active)):
        s, en = int(offsets[e]), int(offsets[e + 1])
        if en - s >= 2:
            idx = np.arange(s, en)
            prev.append(idx[:-1]); cur.append(idx[1:])
    if not prev:
        return np.empty(0, int), np.empty(0, int)
    return np.concatenate(prev), np.concatenate(cur)


def stats(v):
    v = np.asarray(v, float)
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
            "p99": float(np.percentile(v, 99)), "max": float(v.max()), "mean": float(v.mean())}


def analyze(rate, proj):
    f = C1 / f"stream_{rate}hz_{proj}.npz"
    if not f.exists():
        return None
    d = np.load(f)
    meta = json.loads((C1 / f"meta_{rate}hz_{proj}.json").read_text())
    dt_ctrl = float(meta["dt_ctrl"])
    # float32 to match the GPU rollout's exact stored inputs — the enumerate projection's discrete candidate
    # argmin flips under float64, so faithful reconstruction requires the same precision the rollout used.
    u_nom = torch.tensor(d["u_nom"], dtype=torch.float32)
    u_cmd = torch.tensor(d["u_cmd"], dtype=torch.float32)
    a = torch.tensor(d["a"], dtype=torch.float32)
    b = torch.tensor(d["b"], dtype=torch.float32)
    bounds = torch.tensor(d["u_bounds"], dtype=torch.float32)
    clipped = d["clipped"]; empty = d["empty"]; topk = d["topk"]
    offsets = d["ep_offsets"]; active = d["ep_active"]
    params = params_for(proj, meta["dt_sim"])
    prev, cur = pairs(offsets, active)
    P, C = torch.tensor(prev), torch.tensor(cur)

    out = {"rate_hz": rate, "proj": proj, "n_steps": int(u_cmd.shape[0]), "n_pairs": int(prev.size),
           "empty_step_fraction": float(empty.mean()), "dt_ctrl": dt_ctrl}

    # ---------- D1 / C2 : attribution ----------
    delta = (u_cmd[C] - u_cmd[P])
    tv_step = torch.linalg.norm(delta, dim=1).numpy()
    tv_total = float(tv_step.sum())
    emp_p, emp_c = empty[prev], empty[cur]
    trans = emp_p != emp_c
    emptyb = (~trans) & emp_c
    feas = (~trans) & (~emp_c)
    base_ref = F(u_nom[P], a[P], b[P], bounds, params, proj)
    policy_cf = F(u_nom[C], a[P], b[P], bounds, params, proj)
    cert_cf = F(u_nom[P], a[C], b[C], bounds, params, proj)
    policy_vec = (policy_cf - base_ref).numpy()
    cert_vec = (cert_cf - base_ref).numpy()
    cross_vec = delta.numpy() - policy_vec - cert_vec
    # per-pair reconstruction fidelity (base_ref must reconstruct the applied u_cmd on feasible-prev steps;
    # dual_solve is continuous -> exact; enumerate's discrete candidate argmin flips off the GPU float path
    # at its discontinuities -> those pairs are un-attributable by counterfactual refilter).
    recon_all = np.linalg.norm(base_ref.numpy() - u_cmd[P].numpy(), axis=1)
    faithful = feas & (recon_all < 1e-3)
    disc = feas & (recon_all >= 1e-3)               # feasible but at a projection discontinuity
    recon_err_feas = recon_all[feas] if feas.any() else np.zeros(0)
    fnrm = lambda v, m: float(np.linalg.norm(v[m], axis=1).sum())
    tv_policy, tv_cert, tv_cross = fnrm(policy_vec, faithful), fnrm(cert_vec, faithful), fnrm(cross_vec, faithful)
    tv_faithful = float(tv_step[faithful].sum()); tv_disc = float(tv_step[disc].sum())
    tv_emptyb = float(tv_step[emptyb].sum()); tv_trans = float(tv_step[trans].sum())
    chan = tv_policy + tv_cert + tv_cross
    out["D1_attribution"] = {
        "tv_total": tv_total,
        "self_check_F_reconstructs_ucmd_maxerr": float(recon_err_feas.max()) if recon_err_feas.size else 0.0,
        "frac_feasible_pairs_at_discontinuity": float((recon_err_feas >= 1e-3).mean()) if recon_err_feas.size else 0.0,
        "n_pairs": {"faithful_feasible": int(faithful.sum()), "discontinuity_feasible": int(disc.sum()),
                    "empty_branch": int(emptyb.sum()), "branch_transition": int(trans.sum())},
        # partitions total TV (sums to ~1): where the chatter occurs
        "source_tv_fraction": {
            "feasible_attributable": tv_faithful / tv_total if tv_total else None,
            "feasible_discontinuity": tv_disc / tv_total if tv_total else None,
            "empty_branch": tv_emptyb / tv_total if tv_total else None,
            "branch_transition": tv_trans / tv_total if tv_total else None},
        # within faithful-feasible TV, the channel split (policy+cert+cross vectors sum to Δ; shares normalized)
        "feasible_channel_share": {
            "policy": tv_policy / chan if chan else None,
            "certificate": tv_cert / chan if chan else None,
            "cross": tv_cross / chan if chan else None}}

    # ---------- D2 / C3 : amplifier 1/||a_I|| ----------
    a_np = d["a"].astype(np.float64)
    unclip = ~clipped
    aI_norm = np.sqrt((a_np ** 2 * unclip).sum(1))
    a_norm = np.linalg.norm(a_np, 1 and 1, axis=1) if False else np.linalg.norm(a_np, axis=1)
    nclip = clipped.sum(1)
    m = clipped.shape[1]
    cls = np.where(nclip == 0, "no_clip", np.where(nclip == m, "vertex", "partial"))
    valid = aI_norm > 1e-12
    amp = np.where(valid, 1.0 / np.maximum(aI_norm, 1e-30), np.inf)
    # amp aligned to the 'cur' step of each pair for TV correlation
    amp_cur = amp[cur]; valid_cur = valid[cur]
    rho = None
    if valid_cur.sum() > 10:
        rho = float(spearmanr(amp_cur[valid_cur], tv_step[valid_cur]).statistic)
    # TV share of top amp decile (over pairs, by amp at cur)
    top_share = None
    if valid_cur.sum() > 10:
        vc = np.where(valid_cur, amp_cur, -np.inf)
        thr = np.percentile(vc[valid_cur], 90)
        topmask = valid_cur & (amp_cur >= thr)
        top_share = float(tv_step[topmask].sum() / tv_total) if tv_total else None
    out["D2_amplifier"] = {
        "amp_all_valid": stats(amp[valid]),
        "by_class": {c: {"share": float((cls == c).mean()),
                         "amp": stats(amp[(cls == c) & valid])} for c in ("no_clip", "partial", "vertex")},
        "by_branch": {"feasible": stats(amp[valid & (~empty)]), "empty": stats(amp[valid & empty])},
        "spearman_amp_vs_du": rho, "tv_share_top_amp_decile": top_share,
        "a_norm": stats(a_norm), "aI_norm": stats(aI_norm[valid]),
        "n_all_clipped_steps_excluded": int((~valid).sum()),
        "note": "single-row CBF (scalar certificate): a_I = a restricted to box-unclipped control coords; |A|=clipped count gives the S1 class."}

    # ---------- D3 / C4 : top-k crossings ----------
    tp = topk[prev].copy(); tc = topk[cur].copy()
    sp = np.sort(np.where(tp < 0, 10 ** 6, tp), axis=1)
    sc = np.sort(np.where(tc < 0, 10 ** 6, tc), axis=1)
    same_set = (sp == sc).all(1)
    same_order = (tp == tc).all(1)
    crossing = ~same_order
    within_swap = crossing & same_set
    exchange = crossing & (~same_set)
    da = torch.linalg.norm(a[C] - a[P], dim=1).numpy()
    db = torch.abs(b[C] - b[P]).numpy()
    def cn(mask):
        return {"du": stats(tv_step[mask]), "da": stats(da[mask]), "db": stats(db[mask]), "n": int(mask.sum())}
    out["D3_crossings"] = {
        "crossing_rate": float(crossing.mean()), "within_topk_swap_rate": float(within_swap.mean()),
        "kth_exchange_rate": float(exchange.mean()),
        "at_crossing": cn(crossing), "at_non_crossing": cn(~crossing),
        "tv_share_crossing_steps": float(tv_step[crossing].sum() / tv_total) if tv_total else None}

    # ---------- D5 / C6 : spectrum ----------
    u = d["u_cmd"].astype(np.float64); A = u.shape[1]
    frac5, frac10 = [], []
    for e in range(len(active)):
        s, en = int(offsets[e]), int(offsets[e + 1])
        n = en - s
        if n < 8:
            continue
        seg = u[s:en]                                  # [n, A]
        freqs = np.fft.rfftfreq(n, d=dt_ctrl)
        pw = np.abs(np.fft.rfft(seg - seg.mean(0), axis=0)) ** 2   # [F, A], DC removed
        tot = pw.sum(0)
        tot = np.where(tot > 0, tot, 1.0)
        f5 = pw[freqs > 5.0].sum(0) / tot
        f10 = pw[freqs > 10.0].sum(0) / tot
        frac5.append(f5.mean()); frac10.append(f10.mean())
    out["D5_spectrum"] = {
        "nyquist_hz": rate / 2.0,
        "power_fraction_above_5hz": stats(frac5),
        "power_fraction_above_10hz": stats(frac10),
        "n_episodes_scored": len(frac5)}
    return out


def main():
    results = {}
    for rate, proj in CELLS:
        r = analyze(rate, proj)
        if r is None:
            print(f"[skip] {rate}Hz/{proj} — no stream")
            continue
        results[f"{rate}hz_{proj}"] = r
        da = r["D1_attribution"]; src = da["source_tv_fraction"]; ch = da["feasible_channel_share"]
        d2 = r["D2_amplifier"]; d3 = r["D3_crossings"]
        print(f"[{rate}Hz/{proj}] TV source: feas-attr {src['feasible_attributable']:.3f} "
              f"disc {src['feasible_discontinuity']:.3f} empty {src['empty_branch']:.3f} trans {src['branch_transition']:.3f} "
              f"| chan(policy/cert/cross) {ch['policy']:.2f}/{ch['certificate']:.2f}/{ch['cross']:.2f} "
              f"| disc-frac {da['frac_feasible_pairs_at_discontinuity']:.3f}")
        print(f"          D2 amp med {d2['amp_all_valid'].get('median')!r} rho {d2['spearman_amp_vs_du']!r} "
              f"top-decile TV {d2['tv_share_top_amp_decile']!r} | D3 crossing_rate {d3['crossing_rate']:.4f} "
              f"tv_share {d3['tv_share_crossing_steps']:.4f}")
    (C1 / "d_analysis.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"\n-> {C1 / 'd_analysis.json'}  ({len(results)} cells)")


if __name__ == "__main__":
    main()
