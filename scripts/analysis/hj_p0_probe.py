"""v2.6.2 §HJ criterion — P0 feasibility probe (STOP-gated).

Single-obstacle 6D relative-state HJ AVOID computation for the planar quadrotor. Relative state
x_rel = [px_rel, py_rel, theta, vx, vy, omega]; obstacle static at origin, so p_rel dynamics = v.
Gravity is FIXED in the world frame (breaks rotational symmetry -> genuinely 6D; x-mirror symmetry is a
correctness check only, NOT a reduction). Doom value V*(x;R) = sup_u inf_t (||p_rel|| - R): control MAX-
imizes surface distance (control_mode='max'), min-over-time via the backwards_reachable_tube Hamiltonian
postprocessor. V* < 0 <=> even the best control cannot avoid <=> doomed.

Plant (01_env 3.3): m=1, J=0.01, g=9.81; f_thr in [0,19.62], tau in [-1,1]; |v|<=2.5, |omega|<=4.
Domain: p_rel in [-4.5,4.5]^2, theta periodic, |v|<=2.5, |omega|<=4.

P0 measures at a COARSE 21/dim grid: peak GPU bytes, per-step wallclock, and projects the full-grid cost
(31/dim, 45/dim). STOP (report, do not coarsen) if the projected full grid exceeds the 16 GB VRAM / 22 GB
host budget. Velocity/rate clamp handling: the grid edges use hj's extrapolate_away_from_zero BC (outward
flux leaves the value unchanged); the physical system's |v|<=v_max, |omega|<=omega_max box means true
trajectories cannot exit the grid interior, so edge extrapolation is conservative for interior ICs (probe
only records this; P1 states the final handling).
"""
import time as _time

import jax
import jax.numpy as jnp
import numpy as np

import hj_reachability as hj

M, J, G = 1.0, 0.01, 9.81
F_MIN, F_MAX, TAU_MAX = 0.0, 19.62, 1.0
V_MAX, OMEGA_MAX = 2.5, 4.0
P_LIM = 4.5


class QuadRel(hj.ControlAndDisturbanceAffineDynamics):
    def __init__(self):
        control_space = hj.sets.Box(jnp.array([F_MIN, -TAU_MAX]), jnp.array([F_MAX, TAU_MAX]))
        disturbance_space = hj.sets.Box(jnp.array([0.0]), jnp.array([0.0]))
        super().__init__(control_mode="max", disturbance_mode="min",
                         control_space=control_space, disturbance_space=disturbance_space)

    def open_loop_dynamics(self, state, time):
        _, _, _, vx, vy, _ = state
        return jnp.array([vx, vy, state[5], 0.0, -G, 0.0])

    def control_jacobian(self, state, time):
        theta = state[2]
        return jnp.array([[0.0, 0.0],
                          [0.0, 0.0],
                          [0.0, 0.0],
                          [-jnp.sin(theta) / M, 0.0],
                          [jnp.cos(theta) / M, 0.0],
                          [0.0, 1.0 / J]])

    def disturbance_jacobian(self, state, time):
        return jnp.zeros((6, 1))


def make_grid(n):
    lo = np.array([-P_LIM, -P_LIM, -np.pi, -V_MAX, -V_MAX, -OMEGA_MAX])
    hi = np.array([P_LIM, P_LIM, np.pi, V_MAX, V_MAX, OMEGA_MAX])
    ns = (n, n, n, n, n, n)
    return hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(jnp.array(lo), jnp.array(hi)), ns, periodic_dims=2)


def terminal_values(grid, R):
    p = grid.states[..., :2]
    return jnp.linalg.norm(p, axis=-1) - R


def probe_on(device, N, accuracy):
    """Return (per_step_s, peak_bytes, dv, value_gb) or raise. Runs on the given jax device."""
    with jax.default_device(device):
        dyn = QuadRel()
        settings = hj.SolverSettings.with_accuracy(
            accuracy, hamiltonian_postprocessor=hj.solver.backwards_reachable_tube)
        grid = make_grid(N)
        v0 = terminal_values(grid, R_PROBE)

        def one_step(v, t, dt):
            return hj.step(settings, dyn, grid, t, v, t - dt, progress_bar=False)

        dt_probe = 0.02
        v = one_step(v0, 0.0, dt_probe)
        v.block_until_ready()                                  # compile
        nstep = 5
        tic = _time.time()
        t = 0.0
        for _ in range(nstep):
            v = one_step(v, t, dt_probe)
            t -= dt_probe
        v.block_until_ready()
        per_step = (_time.time() - tic) / nstep
        try:
            peak = device.memory_stats().get("peak_bytes_in_use", 0)
        except Exception:
            peak = 0
        dv = float(jnp.max(jnp.abs(v - v0)))
        return per_step, peak, dv, v0.nbytes / 1e9


R_PROBE = 0.5


def main():
    gpu = jax.devices("gpu")[0]
    cpu = jax.devices("cpu")[0]
    print(f"gpu={gpu} cpu={cpu}", flush=True)

    ACC = "low"       # leanest scheme (first-order upwind + first-order TVD-RK) -> memory floor
    dt_probe = 0.02
    print(f"accuracy={ACC} (memory-floor probe)\n", flush=True)

    per_step = peak = dv = None
    N = None
    for cand, device, tag in [(21, gpu, "GPU"), (17, gpu, "GPU"), (21, cpu, "CPU"), (15, cpu, "CPU")]:
        try:
            ps, pk, d, vgb = probe_on(device, cand, ACC)
            per_step, peak, dv, N = ps, pk, d, cand
            over = (pk / (vgb * 1e9)) if pk else float("nan")
            print(f"[OK {tag} N={cand}] per_step={ps*1000:.0f} ms  peak={pk/1e9:.2f} GB  "
                  f"value_arr={vgb:.3f} GB  overhead={over:.0f}x  max|dV|(5 steps)={d:.3f}", flush=True)
            break
        except Exception as e:
            print(f"[FAIL {tag} N={cand}] {str(e)[:110]}", flush=True)
    if per_step is None:
        print("PROBE FAILED at all sizes -> P0 STOP (cannot even run a coarse grid).", flush=True)
        return

    overhead = peak / ((N ** 6) * 4) if peak else 40.0        # empirical peak/value_array
    # convergence horizon estimate: cross ~9 m domain at v_max 2.5 (~3.6 s) + align/brake. T_hj ~ 3.5 s.
    T_hj_est = 3.5
    steps_conv = T_hj_est / dt_probe
    print(f"\nmeasured overhead (peak/value_array) = {overhead:.0f}x at N={N}", flush=True)
    print(f"convergence: dt_probe={dt_probe}s, T_hj_est~{T_hj_est}s -> ~{steps_conv:.0f} steps/solve", flush=True)

    print("\n=== FULL-GRID PROJECTION (memory & per-step ~ N^6) ===", flush=True)
    for Nf in (21, 25, 27, 31, 35, 41, 45):
        fac = (Nf / N) ** 6
        val_gb = (Nf ** 6) * 4 / 1e9
        proj_peak = val_gb * overhead
        proj_step_ms = per_step * 1000 * fac
        proj_solve_min = proj_step_ms / 1000 * steps_conv / 60
        buckets = 5
        fits = "GPU-OK" if proj_peak < 13.2 else ("host-OK" if proj_peak < 22 else "OVER-BUDGET")
        print(f"  N={Nf}: cells={Nf**6:,} value={val_gb:.2f} GB  peak~{proj_peak:.1f} GB [{fits}]  "
              f"step~{proj_step_ms:.0f} ms  solve~{proj_solve_min:.1f} min  x{buckets}buckets~{proj_solve_min*buckets:.0f} min",
              flush=True)
    print("\nBUDGET: GPU VRAM 16 GB (13.2 free), host RAM 22 GB available.", flush=True)


if __name__ == "__main__":
    main()
