"""v2.6.2 §doom census — B (4D relaxed-system HJ) diagnostic census. Read-only; NO pool change; NO training.

Exact avoid value of the RELAXED double integrator (state [px_rel, py_rel, vx, vy]; ṗ=v, v̇=u, ||u||<=A:=3g;
static disc radius R at origin): V*(x;R) = sup_u inf_t (||p_rel|| - R), control_mode='max' + tube
postprocessor min(H,0). Ball control constraint (not a box) -> custom argmax u* = A * grad_v V / ||grad_v V||.
Tighter than the ballistic closed form A (same relaxed system, exact vs reachable-disc bound) -> A-flag set
MUST be a subset of B-flag set. Census on the SAME 3 sets (indices cached by quadrotor_doom_census.py).
Gates: (i) A-flags subset of B-flags; (ii) reach-outcome flags = 0. Doom lookup uses corner-MAX (upper)
interpolation + eps_num >= grid-error bound (all conservatism AGAINST exclusion). Diagnostic only, NOT a
pool criterion in this task.
"""
import json
import time as _time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

import hj_reachability as hj
from src.common.quadrotor_ballistic_doom import (accel_bound, is_doomed_ballistic, radius_bucket_down,
                                                 BUCKET_LO, BUCKET_HI, BUCKET_STEP)
from src.eval.build_pools import load_pool

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
SP = Path("/tmp/claude-1000/-home-junhyeok-MIT-jt-pncbf/31d93785-ac11-4206-bf50-a4c3de145dff/scratchpad")
POOL = REPO / "data/secured_data/pools/eval_full_quadrotor-planar_n2000_seed23456.pkl"
RUN = REPO / "data/v2.6.2__20260716-182949__seed42"

P_LIM = 5.0
V_DOM = 4.0
N_POS = 81
N_VEL = 61
A = 29.43


class RelaxedDI(hj.ControlAndDisturbanceAffineDynamics):
    """p=[px,py,vx,vy]; ṗ=v, v̇=u, ||u||<=A. Ball control (override argmax); avoid = control maximizes."""
    def __init__(self, A):
        self.A = A
        cs = hj.sets.Box(jnp.array([-A, -A]), jnp.array([A, A]))            # bounding box (argmax overridden)
        ds = hj.sets.Box(jnp.array([0.0]), jnp.array([0.0]))
        super().__init__(control_mode="max", disturbance_mode="min", control_space=cs, disturbance_space=ds)

    def open_loop_dynamics(self, state, time):
        return jnp.array([state[2], state[3], 0.0, 0.0])

    def control_jacobian(self, state, time):
        return jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    def disturbance_jacobian(self, state, time):
        return jnp.zeros((4, 1))

    def optimal_control_and_disturbance(self, state, time, grad_value):
        d = grad_value @ self.control_jacobian(state, time)                # = grad_v V  (2,)
        norm = jnp.linalg.norm(d)
        u = jnp.where(norm > 1e-12, self.A * d / jnp.maximum(norm, 1e-12), jnp.zeros(2))
        return u, jnp.zeros(1)

    def partial_max_magnitudes(self, state, time, value, grad_value_box):
        del value, grad_value_box
        return jnp.array([jnp.abs(state[2]), jnp.abs(state[3]), self.A, self.A])


def make_grid(npos, nvel):
    lo = jnp.array([-P_LIM, -P_LIM, -V_DOM, -V_DOM])
    hi = jnp.array([P_LIM, P_LIM, V_DOM, V_DOM])
    return hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(lo, hi), (npos, npos, nvel, nvel))


def solve_bucket(grid, dyn, settings, R, tol=2e-3, t_chunk=0.25, t_max=6.0):
    """Solve V*(.;R) to convergence; return (values ndarray, T_hj, final max|dV|/chunk)."""
    p = grid.states[..., :2]
    v0 = jnp.linalg.norm(p, axis=-1) - R
    v = v0
    t = 0.0
    last = None
    while -t < t_max:
        vn = hj.step(settings, dyn, grid, t, v, t - t_chunk, progress_bar=False)
        vn.block_until_ready()
        dv = float(jnp.max(jnp.abs(vn - v)))
        v = vn
        t -= t_chunk
        last = dv
        if dv < tol:
            break
    return np.asarray(v), -t, last


def upper_lookup(values, R, xq):
    """corner-MAX (upper) interpolation of V* at query points xq (M,4). Out-of-grid -> +inf (safe).
    Returns V*_upper (M,)."""
    lo = np.array([-P_LIM, -P_LIM, -V_DOM, -V_DOM])
    hi = np.array([P_LIM, P_LIM, V_DOM, V_DOM])
    ns = np.array(values.shape)
    dx = (hi - lo) / (ns - 1)
    out = np.full(xq.shape[0], np.inf)
    frac = (xq - lo) / dx
    i0 = np.floor(frac).astype(int)
    inside = np.all((i0 >= 0) & (i0 <= ns - 2), axis=1)
    for m in np.where(inside)[0]:
        base = i0[m]
        best = -np.inf
        for corner in range(16):
            idx = base + np.array([(corner >> k) & 1 for k in range(4)])
            best = max(best, float(values[idx[0], idx[1], idx[2], idx[3]]))
        out[m] = best
    return out


def scene_pairs(sc):
    """Per active obstacle: relative 4D state x_rel = [px_rel,py_rel,vx,vy] and rounded-down R."""
    p0 = np.asarray(sc.start, np.float64); v0 = np.asarray(sc.initial_velocity, np.float64)
    C = np.asarray(sc.obstacle_centers, np.float64); Rr = np.asarray(sc.obstacle_radii, np.float64)
    act = np.asarray(sc.obstacle_active, bool)
    xs, Rs = [], []
    for c, r, a in zip(C, Rr, act):
        if not a:
            continue
        xs.append(np.array([p0[0] - c[0], p0[1] - c[1], v0[0], v0[1]]))
        Rs.append(radius_bucket_down(float(r)))
    return xs, Rs


def main():
    dev = jax.devices()[0]
    print(f"device={dev} grid pos={N_POS} vel={N_VEL} cells={N_POS**2*N_VEL**2:,} "
          f"value={N_POS**2*N_VEL**2*4/1e6:.0f} MB", flush=True)
    dyn = RelaxedDI(A)
    settings = hj.SolverSettings.with_accuracy(
        "high", hamiltonian_postprocessor=hj.solver.backwards_reachable_tube)

    # memory/time probe on the coarsest bucket
    grid = make_grid(N_POS, N_VEL)
    tic = _time.time()
    vals, T_hj, dvf = solve_bucket(grid, dyn, settings, R=0.5)
    solve_s = _time.time() - tic
    dxp = 2 * P_LIM / (N_POS - 1)
    eps_num = 2.0 * dxp                                # grid-error bound: |grad_p V*|~1 (distance) -> 2*dx
    print(f"[probe R=0.5] solve={solve_s:.1f}s T_hj={T_hj:.2f}s final_dV/chunk={dvf:.2e} "
          f"Vmin={vals.min():.3f} dx_pos={dxp:.3f} eps_num={eps_num:.3f}", flush=True)

    scenes = load_pool(POOL).scenes; N = len(scenes)
    z = np.load(SP / "doom_census_rollout.npz")
    born = list(z["born"]); reach = list(z["reach"])

    # buckets actually present (rounded down)
    present = sorted({radius_bucket_down(float(r)) for sc in scenes
                      for r, a in zip(sc.obstacle_radii, sc.obstacle_active) if a})
    print(f"radius buckets present (rounded down): {present}", flush=True)
    tables = {}
    for R in present:
        vals, T_hj, dvf = solve_bucket(grid, dyn, settings, R=R)
        tables[R] = vals
        print(f"  bucket R={R:.2f}: T_hj={T_hj:.2f}s dV/chunk={dvf:.2e} Vmin={vals.min():.3f}", flush=True)

    def b_flag(gi):
        xs, Rs = scene_pairs(scenes[gi])
        if not xs:
            return False
        for x, R in zip(xs, Rs):
            vu = upper_lookup(tables[R], R, x[None, :])[0]
            if vu < -eps_num:
                return True
        return False

    f_all = [gi for gi in range(N) if b_flag(gi)]
    f_born = [gi for gi in born if b_flag(gi)]
    f_reach = [gi for gi in reach if b_flag(gi)]

    # A-flag set (from A json) for the subset gate
    aj = json.load(open(SP / "quadrotor_doom_census_A.json"))
    a_all = set(aj["flagged_all_idx"])
    subset_ok = a_all.issubset(set(f_all))

    print(f"\nB 4D-HJ flag rates (relaxed 3g, exact solve, upper interp, eps_num={eps_num:.3f}):", flush=True)
    print(f"  (i)   all pool ICs : {len(f_all)}/{N} = {len(f_all)/N:.4f}", flush=True)
    print(f"  (ii)  born-doomed  : {len(f_born)}/{len(born)} = {len(f_born)/max(1,len(born)):.4f} (context)", flush=True)
    print(f"  (iii) reach ICs    : {len(f_reach)}/{len(reach)} = {len(f_reach)/max(1,len(reach)):.4f}  [GATE: must be 0]", flush=True)
    print(f"GATE reach=0 : {'PASS' if len(f_reach)==0 else 'FAIL'}", flush=True)
    print(f"GATE A subset of B : {'PASS' if subset_ok else 'FAIL'}  (|A_flags|={len(a_all)} |B_flags|={len(f_all)})", flush=True)
    print(f"A-vs-B gap (B extra flags beyond A): {len(set(f_all)-a_all)}", flush=True)

    out = dict(A=A, npos=N_POS, nvel=N_VEL, eps_num=eps_num, buckets=present, T_hj=T_hj,
               flag_all=len(f_all), flag_born=len(f_born), flag_reach=len(f_reach),
               reach_gate=len(f_reach) == 0, subset_gate=bool(subset_ok),
               b_extra_beyond_a=len(set(f_all) - a_all), flagged_all_idx=[int(i) for i in f_all])
    json.dump(out, open(SP / "quadrotor_doom_census_B.json", "w"), indent=2)
    print("WROTE", SP / "quadrotor_doom_census_B.json", flush=True)


if __name__ == "__main__":
    main()
