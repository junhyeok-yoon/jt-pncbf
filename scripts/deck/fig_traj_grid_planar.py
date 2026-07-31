"""v2.7.7 M26 (Amdt 13/14) — filter-active trajectory grid, planar quadrotor. Amdt-14: REACH episodes only, require >=1
filter-active step, rank by (#active steps, smallest clearance), top 6. Red = filter active, blue = nominal.
Titles/footers stripped (system name only). Eval-only; no src edits. Chosen indices printed for the manifest."""
from scripts.deck._filter_active import roll_active, select_top6, render_grid_2d

CK = "data/secured_data/v2.7.0/seed42_iter5/checkpoints/best.pt"
POOL = "data/secured_data/pools/eval_full_quadrotor-planar_n2000_seed23456.pkl"
X, act, scenes, dt, goal_r = roll_active(CK, POOL, N=400)
order, na, mc, reached, info = select_top6(X, act, scenes, goal_r)
p = render_grid_2d(order, X, act, scenes, "planar quadrotor", "fig_traj_grid_planar.png")
print(f"M26 planar -> {p.name}; reach+active candidates {info['n_reach_active_candidates']}; chosen {info['chosen_indices']}; "
      f"active {info['chosen_active_steps']}; min_clear {info['chosen_min_clearance']}")
