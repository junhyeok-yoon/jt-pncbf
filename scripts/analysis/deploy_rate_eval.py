"""v2.5.0 deploy-rate eval (EVAL ONLY, no training). Roll the Stage-B best policy (or clamped LQR)
through the analytic V_M HardNet filter at independent rates: dt_plant (plant grid + collision-detection
fidelity), dt_ctrl (control/filter re-eval period, ZOH between), dt_Vm (certificate internal rollout).

TIME-base: episode horizon 10.0 s, stuck window 3.0 s / disp 0.10, reach instantaneous, oob spatial —
all converted to steps at dt_plant. Collision on the plant grid. Infeasibility & cert-invariance are
reported per-unit-TIME and per-episode-any (per-step means are not comparable across dt); the cps
infeasibility term uses mean over active CONTROL steps (fraction of filter decisions), which IS
comparable across arms. One arm per invocation."""
import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch
from src.common.control_net import ControlNet
from src.common.filter_hardnet import (_SINGULAR_LG_THRESHOLD, _base_alpha, _base_projection,
    _box_aware_projection, _cbf_terms, _hardnet_params)
from src.common.maneuver_value import make_maneuver_h_fn, maneuver_value
from src.common.outcomes import resolve_outcome, step_outcomes
from src.common.rk4 import rk4_step
from src.eval.build_pools import load_pool
from src.eval.evaluate import first_physical_event_step
from src.envs.double_integrator import DoubleIntegrator
from src.envs.scene_batch import batch_scenes, initial_states_from_batch

REPO = Path(__file__).resolve().parents[2]


def infeasible_v2(singular, empty, row):
    """cps-v2 per-step infeasible flag: empty-intersection OR (singular AND the CBF row is NOT satisfied).
    At singular steps L_g V ~ 0, so the CBF row L_g V . u <= row_upper is u-INDEPENDENT and holds iff
    row_upper (= -L_f V - alpha*V) >= 0. A far-field-flat step with the row already satisfied is FEASIBLE
    (the nominal needs no control authority), so it must NOT count as infeasible. row = row_upper."""
    return empty | (singular & (row < 0.0))
RUN = REPO / "data/v2.5.0__20260708-212214__seed42"
POOL = REPO / "data/secured_data/pools/eval_full_di_n2000_seed23456.pkl"
HORIZON_SEC, STUCK_SEC, DELTA, GAMMA_M = 10.0, 3.0, 0.05, 0.02
LAT_DUR = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]   # A1' lateral durations (s), both dirs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt-plant", type=float, required=True)
    ap.add_argument("--dt-ctrl", type=float, required=True)
    ap.add_argument("--dt-vm", type=float, required=True)
    ap.add_argument("--nominal", choices=["policy", "lqr"], default="policy")
    ap.add_argument("--ckpt", default="step_009000.pt")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--filter", choices=["maneuver", "learned"], default="maneuver")
    ap.add_argument("--run", default=str(RUN))          # run dir holding checkpoints/ (override for other ckpts)
    ap.add_argument("--outdir", default=None)           # writable output dir (secured_data run dirs are read-only)
    a = ap.parse_args()
    run_dir = Path(a.run); outdir = Path(a.outdir) if a.outdir else run_dir
    dev = torch.device("cuda")
    dtp, dtc, dtv = a.dt_plant, a.dt_ctrl, a.dt_vm
    n_sub = int(round(dtc / dtp))                       # plant substeps per control update
    max_plant = int(round(HORIZON_SEC / dtp))
    lat_js = [int(round(d / dtv)) for d in LAT_DUR]     # lateral durations -> dt_Vm steps

    ck = torch.load(run_dir / "checkpoints" / a.ckpt, map_location=dev, weights_only=False)
    cfg = ck["config"]; cfg.setdefault("safety_channel", {})["type"] = "maneuver"
    cfg["env"]["stuck_window_steps"] = int(round(STUCK_SEC / dtp))    # TIME-converted stuck window
    system = DoubleIntegrator(cfg)
    dtype = torch.float32
    system.u_bounds = system.u_bounds.to(device=dev, dtype=dtype)
    bounds = system.u_bounds
    params = _hardnet_params(cfg); alpha_s = float(cfg["filter"]["alpha_safe"])
    # h_fn = the barrier the filter projects onto; cert_val/cert_gamma = the barrier for cert diagnostics.
    # maneuver branch: byte-identical to the pre-flag behavior. learned branch: deployed learned V_hat.
    if a.filter == "learned":
        from src.common.value_net import ValueNetEnsemble, make_h_fn
        vn = ValueNetEnsemble(system.obs_dim, cfg).to(device=dev, dtype=dtype)
        vn.load_state_dict(ck["v_s_state"]); vn.eval()
        h_fn = make_h_fn(vn, system)                                  # dt_Vm n/a on this branch
        cert_val = lambda x, scene: vn.deployed_h(system.observation(x, scene))
        cert_gamma = 0.0
    else:
        h_fn = make_maneuver_h_fn(system, cfg, lateral_js=lat_js, gamma_m=GAMMA_M, dt_override=dtv)
        cert_val = lambda x, scene: maneuver_value(x, scene, system, cfg, lateral_js=lat_js, dt_override=dtv)
        cert_gamma = GAMMA_M
    pol = None
    if a.nominal == "policy":
        pol = ControlNet(system.obs_dim, system, cfg).to(device=dev, dtype=dtype)
        pol.load_state_dict(ck["pi_state"]); pol.eval()

    def nominal(x, scene):
        if pol is not None:
            return pol(system.observation(x, scene))
        goal = torch.as_tensor(scene.goal, dtype=x.dtype, device=x.device)
        if goal.ndim == 1: goal = goal.unsqueeze(0).expand(x.shape[0], -1)
        return system.lqr_action(x, goal)

    def filt(x, u_nom, scene):
        h, lf, lg = _cbf_terms(system, h_fn, x, scene, u_nom, create_graph=False)
        h, lf, lg = h.detach(), lf.detach(), lg.detach()
        with torch.no_grad():
            alpha = _base_alpha(h, params); row = -lf - alpha * h
            proj = _base_projection(u_nom, lg, row, bounds, params)
            sing = torch.linalg.norm(lg, dim=1) < _SINGULAR_LG_THRESHOLD
            u_safe, empty = _box_aware_projection(u_nom, proj, lg, row, bounds)
        # cps-v2: at singular steps L_g V ~ 0 so the CBF row is u-independent; row_satisfied <=> row_upper
        # (= -L_f V - alpha*V) >= 0 (i.e. L_f V + alpha*V <= 0). A far-field-flat step with the row already
        # satisfied is FEASIBLE, not infeasible. row returned for the v2 flag.
        return u_safe.detach(), sing, empty, lg.detach(), row.detach()

    scenes = load_pool(POOL).scenes
    ep = []
    reg = {k: 0.0 for k in ("cert_n", "sing_cert", "emp_cert", "ok_cert")}
    lg_s, pm_all, pm_band = [], [], []
    t0 = time.time()
    for s0 in range(0, len(scenes), a.batch):
        bscene = batch_scenes(scenes[s0:s0 + a.batch], device=dev, dtype=dtype)
        x = initial_states_from_batch(bscene).to(dtype)
        B = x.shape[0]
        with torch.no_grad():
            v_start = cert_val(x, bscene) + cert_gamma
        n_ctrl = max_plant // n_sub
        vm_c = torch.zeros(n_ctrl + 1, B, dtype=dtype, device=dev)         # V_M at control steps (raw)
        inf_c = torch.zeros(n_ctrl, B, dtype=torch.bool, device=dev)
        emp_c = torch.zeros_like(inf_c); sng_c = torch.zeros_like(inf_c)
        band_c = torch.zeros_like(inf_c); pmag_c = torch.zeros(n_ctrl, B, dtype=dtype, device=dev)
        inf2_c = torch.zeros_like(inf_c)          # cps-v2 flag: empty OR (singular AND row NOT satisfied)
        states = [x]; u_hold = None; ci = 0
        for t in range(max_plant):
            if t % n_sub == 0:
                u_nom = nominal(x, bscene)
                u_safe, sing, empty, lg, row = filt(x, u_nom, bscene)
                u_hold = u_safe
                with torch.no_grad():
                    h_now = cert_val(x, bscene)
                    vm_c[ci] = h_now
                    inf_c[ci] = sing | empty; emp_c[ci] = empty; sng_c[ci] = sing
                    inf2_c[ci] = infeasible_v2(sing, empty, row)   # empty | (singular & row_upper<0)
                    band_c[ci] = (sing | empty) & (h_now > (-1.0 + DELTA))
                    pm = torch.linalg.norm(u_safe - u_nom, dim=1); pmag_c[ci] = pm
                    lgn = torch.linalg.norm(lg, dim=1)
                if len(lg_s) < 40000: lg_s.append(lgn.cpu())
                ci += 1
            x = rk4_step(system, x, u_hold, dtp)
            states.append(x)
        with torch.no_grad():
            vm_c[ci] = cert_val(x, bscene)
        S = torch.stack(states, 0)
        masks = step_outcomes(S, bscene, system, cfg); resolved = resolve_outcome(masks)
        event = first_physical_event_step(masks)
        active_plant = torch.where(event >= 0, event, torch.full_like(event, max_plant))  # plant steps
        # map plant active horizon -> control steps
        cidx = torch.arange(n_ctrl, device=dev).unsqueeze(1)
        active_c = (cidx * n_sub) < active_plant.unsqueeze(0)
        cert = (vm_c[:n_ctrl] <= 0.0) & active_c
        okmask = vm_c[1:n_ctrl + 1] <= (1.0 - alpha_s * dtc) * vm_c[:n_ctrl] + 1e-9
        inviol = (vm_c[:n_ctrl] <= 0.0) & (vm_c[1:n_ctrl + 1] > 0.0) & active_c
        reg["cert_n"] += float(cert.sum()); reg["sing_cert"] += float((sng_c & cert).sum())
        reg["emp_cert"] += float((emp_c & cert).sum()); reg["ok_cert"] += float((okmask & cert).sum())
        for i in range(B):
            am = active_c[:, i]; na = int(am.sum())
            atime = float(active_plant[i]) * dtp                          # episode active seconds
            rate = lambda f: float((f[:, i] & am).sum()) / na if na else 0.0
            band_m = am & (vm_c[:n_ctrl, i] > (-1.0 + DELTA))
            ep.append({"outcome": resolved.outcome[i], "start_cert": bool(v_start[i] <= 0),
                       "start_speed": float(system.speed(states[0][i:i + 1])),
                       "inf_canonical": rate(inf_c), "inf_empty": rate(emp_c), "inf_band": rate(band_c),
                       "inf_v2": rate(inf2_c),
                       "inf_any": bool((inf_c[:, i] & am).any()), "inf_per_s": float((inf_c[:, i] & am).sum()) / atime if atime else 0.0,
                       "inviol_any": bool((inviol[:, i]).any()), "inviol_per_s": float(inviol[:, i].sum()) / atime if atime else 0.0,
                       "pm_all": float(pmag_c[am, i].mean()) if na else 0.0,
                       "pm_band": float(pmag_c[band_m, i].mean()) if int(band_m.sum()) else float("nan")})
    wall = time.time() - t0
    orate = lambda o: float(np.mean([e["outcome"] == o for e in ep]))
    reach, coll, oob, stuck, timeout = (orate(o) for o in ("goal", "collision", "oob", "stuck", "timeout"))
    inf = {k: float(np.mean([e[k] for e in ep])) for k in ("inf_canonical", "inf_empty", "inf_band", "inf_v2")}
    cps = lambda i: reach - 2 * coll - stuck - 0.5 * (oob + timeout) - 0.3 * i
    cert = [e for e in ep if e["start_cert"]]; unc = [e for e in ep if not e["start_cert"]]
    sp = lambda g, o: float(np.mean([e["outcome"] == o for e in g])) if g else None
    rr = lambda k: reg[k] / reg["cert_n"] if reg["cert_n"] else 0.0
    lgc = torch.cat(lg_s) if lg_s else torch.zeros(1)
    pmb = np.array([e["pm_band"] for e in ep]); pmb = pmb[~np.isnan(pmb)]
    out = {"tag": a.tag, "dt_plant": dtp, "dt_ctrl": dtc, "dt_vm": dtv, "nominal": a.nominal,
           "n_ep": len(ep), "outcomes": {"reach": reach, "collision": coll, "oob": oob, "stuck": stuck, "timeout": timeout},
           "infeas_ctrlstep": inf,
           "cps": {"canonical": cps(inf["inf_canonical"]), "empty_only": cps(inf["inf_empty"]),
                   "active_band": cps(inf["inf_band"]), "v2": cps(inf["inf_v2"])},
           "cps_v2": cps(inf["inf_v2"]),   # first-class: infeas on empty|(singular & row-not-satisfied)
           "cert_start_split": {"n_cert": len(cert), "n_uncert": len(unc),
               "cert_coll": sp(cert, "collision"), "uncert_coll": sp(unc, "collision"), "uncert_reach": sp(unc, "goal"),
               "uncert_speed_mean": float(np.mean([e["start_speed"] for e in unc])) if unc else None},
           "region_cert": {"singular_cert": rr("sing_cert"), "empty_cert": rr("emp_cert"), "discrete_cbf_ok_cert": rr("ok_cert")},
           "infeas_per_s_mean": float(np.mean([e["inf_per_s"] for e in ep])),
           "infeas_episode_any": float(np.mean([e["inf_any"] for e in ep])),
           "inviol_per_s_mean": float(np.mean([e["inviol_per_s"] for e in ep])),
           "inviol_episode_any": float(np.mean([e["inviol_any"] for e in ep])),
           "authority": {"lg_mean": float(lgc.mean()), "lg_p50": float(torch.quantile(lgc, .5)), "lg_p90": float(torch.quantile(lgc, .9))},
           "proj_mag": {"all_ctrl_mean": float(np.mean([e["pm_all"] for e in ep])), "active_band_mean": float(pmb.mean()) if pmb.size else None},
           "coll_ci": _boot_ci([e["outcome"] == "collision" for e in ep]),
           "cert_coll_ci": _boot_ci([e["outcome"] == "collision" for e in cert]),
           "wallclock_s": round(wall, 1)}
    out["filter"] = a.filter; out["ckpt_path"] = str(run_dir / "checkpoints" / a.ckpt)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"deploy_{a.tag}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0


def _boot_ci(flags, n=1000):
    a = np.asarray(flags, dtype=float)
    if a.size == 0: return [None, None]
    idx = np.random.default_rng(0).integers(0, a.size, size=(n, a.size))
    bs = a[idx].mean(axis=1)
    return [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]


if __name__ == "__main__":
    raise SystemExit(main())
