"""v2.7.7 M26 (Amdt 13/14) — filter-active trajectory grid, quadrotor_3d (after 09c33bf4) on the canonical pool.
Amdt-14: REACH episodes only (exclude collision / band violation / stuck / oob / timeout), require >=1 filter-active
step, rank by (#active steps, smallest clearance), top 6. Red = filter active, blue = nominal. Titles/footers
stripped (system name only). Eval-only; no src edits. Selection rule + chosen indices printed for the manifest."""
from scripts.deck._filter_active import roll_active, select_top6, render_grid_3d

CK = "data/previous_runs/v2.7.6/set__20260725-043415__seed42/v2.7.6__jt__20260725-052127__seed42/checkpoints/step_042000.pt"
POOL = "data/eval_pools/eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl"
X, act, scenes, dt, goal_r = roll_active(CK, POOL, N=400)
order, na, mc, reached, info = select_top6(X, act, scenes, goal_r)
p = render_grid_3d(order, X, act, scenes, "3-D quadrotor", "fig_traj_grid_3d.png")
print(f"M26 3d -> {p.name}; reach+active candidates {info['n_reach_active_candidates']}; chosen {info['chosen_indices']}; "
      f"active {info['chosen_active_steps']}; min_clear {info['chosen_min_clearance']}")
