"""v2.8.0 S1 M4 probes A1 / A1n / A2 (paired dual_solve vs enumerate), eval-only on the v2.7.6 headline
checkpoint 09c33bf4 / canonical pool / deployed fallback (kstep k=5, banded). Persists to
data/runs/v2.8.0/s1_projection/probe.json."""
from __future__ import annotations
import copy, json
from pathlib import Path
import numpy as np, torch
from src.eval.run_full import _load_framework
from src.eval.build_pools import load_pool
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.common.rk4 import rk4_step
from src.common.filter_hardnet import (_cbf_terms, _base_alpha, _base_projection, _empty_halfspace_box,
    _box_aware_projection, _dual_solve_projection)

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
JT3D = REPO / "data/runs/v2.7.6/set__20260725-043415__seed42/v2.7.6__jt__20260725-052127__seed42/checkpoints/step_042000.pt"
POOL3D = REPO / "data/eval_pools/eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl"
OUT = REPO / "data/runs/v2.8.0/s1_projection"; OUT.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VTOL = 1e-6


def load(proj, fallback):
    ck = torch.load(JT3D, map_location="cpu", weights_only=False)
    filt = copy.deepcopy(ck["config"]["filter"]); filt["empty_fallback"] = fallback; filt["projection"] = proj
    over = {"env": {"dt": 0.05, "stuck_window_steps": 60, "stuck_radius": 0.10, "band_collision_limit": 4.0},
            "eval": {"max_steps": 200, "dt_ctrl": 0.05}, "filter": filt}
    fw, cfg, _ = _load_framework(JT3D, config_overrides=over)
    for n in ("value_net", "policy_net"):
        m = getattr(fw, n, None);  m.to(DEV) if m is not None else None
    return fw


def capture(fw, n_scenes=2000, max_steps=200):
    system, hf = fw.system, fw._filter
    h_fn, params = hf.h_fn, hf.params
    scenes = load_pool(POOL3D).scenes[:n_scenes]
    bscene = batch_scenes(scenes, device=DEV, dtype=torch.float32)
    x = initial_states_from_batch(bscene)
    bounds = system.u_bounds.to(DEV, torch.float32)
    E, US, UN, A, Bb = [], [], [], [], []
    for _ in range(max_steps):
        with torch.no_grad():
            u_nom = fw.policy(x, bscene)
        h, lf, lg = _cbf_terms(system, h_fn, x, bscene, u_nom, create_graph=False)
        row = -lf - _base_alpha(h, params) * h
        with torch.no_grad():
            empty = _empty_halfspace_box(lg, row, bounds)
            u_safe, _ = fw.filter(x, u_nom, bscene)
            E.append(empty); US.append(u_safe); UN.append(u_nom); A.append(lg); Bb.append(row)
            x = system.wrap_state(rk4_step(system, x, u_safe, 0.05))
    return (torch.cat(E), torch.cat(US), torch.cat(UN), torch.cat(A), torch.cat(Bb), bounds)


def vertex_share(u, bounds, sub=None):
    low, high = bounds[:, 0], bounds[:, 1]
    v = (((u - low).abs() < VTOL) | ((u - high).abs() < VTOL)).all(dim=1)
    return float(v[sub].float().mean()) if sub is not None else float(v.float().mean())


rep = {"checkpoint": "09c33bf4", "pool": "0ef3751b", "fallback": "kstep k=5 (banded)", "device": str(DEV)}

# ---- A1 (feasible-branch vertex share) + capture per arm ----
caps = {}
for proj in ("enumerate", "dual_solve"):
    fw = load(proj, {"mode": "kstep", "k": 5})
    E, US, UN, A, Bb, bounds = capture(fw)
    sat = ~E
    caps[proj] = (E, US, UN, A, Bb, bounds)
    rep.setdefault("A1", {})[proj] = {
        "n_filtered": int(E.numel()), "n_satisfiable": int(sat.sum()),
        "feasible_branch_vertex_share": vertex_share(US, bounds, sat),
    }
    print(f"A1 {proj}: sat={int(sat.sum())} feasible-branch vertex share={rep['A1'][proj]['feasible_branch_vertex_share']:.4f}", flush=True)
rep["A1"]["prediction_dual_under_0.10"] = rep["A1"]["dual_solve"]["feasible_branch_vertex_share"] < 0.10

# ---- A1n (empty-branch least-violating rule unchanged): on the enum arm's empty states, dual vs enum ----
E, US, UN, A, Bb, bounds = caps["enumerate"]
empty = E
base = _base_projection(UN, A, Bb, bounds, type("P", (), {"epsilon": 0.0, "lg_reg_eps": 0.0})())
enum_none, _ = _box_aware_projection(UN, base, A, Bb, bounds)      # least-violating (mode=none)
dual_none, _ = _dual_solve_projection(UN, base, A, Bb, bounds)
ok_a = A.abs().amax(1) > 1e-12
em = empty & ok_a
rep["A1n"] = {
    "n_empty_rows": int(empty.sum()), "n_empty_with_authority": int(em.sum()),
    "max_abs_action_diff_empty": float((dual_none[em] - enum_none[em]).abs().amax()) if int(em.sum()) else 0.0,
    "empty_vertex_share_dual": vertex_share(dual_none, bounds, em),
    "empty_vertex_share_enum": vertex_share(enum_none, bounds, em),
}
rep["A1n"]["bit_parity"] = bool(rep["A1n"]["max_abs_action_diff_empty"] == 0.0)
print(f"A1n: empty={int(empty.sum())} max|dual-enum| on empty={rep['A1n']['max_abs_action_diff_empty']:.2e} "
      f"bit_parity={rep['A1n']['bit_parity']}", flush=True)


# ---- A2 (matched-perturbation: action-diff / state-diff at the first >100x separation growth) ----
def a2(proj, n_scenes=512, max_steps=120, sub=4):
    fw = load(proj, {"mode": "kstep", "k": 5})
    system, hf = fw.system, fw._filter
    scenes = load_pool(POOL3D).scenes[:n_scenes]
    bscene = batch_scenes(scenes, device=DEV, dtype=torch.float32)
    x1 = initial_states_from_batch(bscene); x2 = x1.clone()
    ratios = torch.full((n_scenes,), float("nan"), device=DEV)
    done = torch.zeros(n_scenes, dtype=torch.bool, device=DEV)
    prev_dx = torch.zeros(n_scenes, device=DEV)
    for _ in range(max_steps):
        with torch.no_grad():
            un1 = fw.policy(x1, bscene); un2 = fw.policy(x2, bscene)
            u1, _ = fw.filter(x1, un1, bscene); u2, _ = fw.filter(x2, un2, bscene)
            dx = torch.linalg.norm((x1 - x2).reshape(n_scenes, -1), dim=1)
            du = torch.linalg.norm((u1 - u2).reshape(n_scenes, -1), dim=1)
            grow = (prev_dx > 1e-12) & (dx > 100.0 * prev_dx) & (~done)
            r = du / dx.clamp_min(1e-30)
            ratios[grow] = r[grow]; done |= grow
            prev_dx = dx
            x1 = system.wrap_state(rk4_step(system, x1, u1, 0.05))
            xs = x2                                                  # finer integration for the second trajectory
            for _s in range(sub):
                xs = system.wrap_state(rk4_step(system, xs, u2, 0.05 / sub))
            x2 = xs
    v = ratios[torch.isfinite(ratios)]
    return v


for proj in ("enumerate", "dual_solve"):
    v = a2(proj)
    rep.setdefault("A2", {})[proj] = {
        "n_events": int(v.numel()),
        "median_ratio": float(v.median()) if v.numel() else float("nan"),
        "p90_ratio": float(torch.quantile(v, 0.90)) if v.numel() else float("nan"),
        "max_ratio": float(v.max()) if v.numel() else float("nan"),
    }
    print(f"A2 {proj}: n={int(v.numel())} median={rep['A2'][proj]['median_ratio']:.3e} p90={rep['A2'][proj]['p90_ratio']:.3e}", flush=True)
if rep["A2"]["enumerate"]["p90_ratio"] and rep["A2"]["dual_solve"]["p90_ratio"]:
    rep["A2"]["p90_orders_of_magnitude_lower_dual_vs_enum"] = float(
        np.log10(rep["A2"]["enumerate"]["p90_ratio"] / max(rep["A2"]["dual_solve"]["p90_ratio"], 1e-30)))

(OUT / "probe.json").write_text(json.dumps(rep, indent=2) + "\n")
print("PROBE DONE ->", OUT / "probe.json")
