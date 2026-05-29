# 01 — Environment

In RL terms this document defines the **environment**: the world the agent acts in, the
state it evolves, the observation it emits, and the rules that decide collision, goal, and
termination. It does **not** define the agent (nominal policy, neural CBF, safety filter,
safety margin) — that is `02_control` — nor how scenes are sampled for training and
evaluation — that is `03_train` and `04_eval`.

Two systems are defined: **Double Integrator** and **Unicycle**. A planar quadrotor is
discussed only as a future extension (§3.3).

Run-level knobs (`dt`, control bounds, `v_init_max`, episode length, RNG seed, and other
tunables) live in `src/configs/` — see `base_config.yaml` for the rarely-changed task and
filter parameters and `exp_config.yaml` for the per-experiment hyperparameters. This
document fixes the structure and the canonical problem-defining constants.

---

## 1. Common task definition

### 1.1 Navigation task

The agent drives a single point-mass-like system from a start state to a goal region while
avoiding a field of circular obstacles, inside a bounded square arena. Obstacle layouts are
randomized per episode; generalizing across these layouts is the point of the work.

### 1.2 Arena and out-of-bounds

- The arena is the square `world_lim = 4.0`, i.e. nominal positions lie in $[-4, 4]^2$.
- The out-of-bounds (OOB) boundary is `oob_limit = 2 * world_lim = 8.0`.
- OOB **terminates** the episode (outcome = OOB). The state is **not** clamped or pushed
  back: forcing the state would alter the dynamics. Leaving $[-8, 8]^2$ simply ends the
  episode as a failure-to-stay-in-bounds.

### 1.3 Obstacle field

The task is defined over randomized fields of axis-free circular obstacles. The canonical
field characteristics are:

- Number of obstacles per scene: 1 to 12.
- Radius of each obstacle: 0.15 to 0.80.
- Layouts are corridor-biased: a fraction (canonically 0.6) of scenes place obstacles around
  the start-goal segment to create corridor-like passages; the remainder are uniform.

Values live in `base_config.obstacle.*`. The **sampling procedure** (how a concrete scene is
drawn, start/goal clearance, minimum start-goal distance, rejection rules, initial velocity)
differs between training and evaluation and is therefore specified in `03_train` and
`04_eval`, not here (§1.7).

Inactive obstacle slots (when a scene has fewer than the maximum) are represented by
**zero-padding** (center and radius zero). Activity is carried by an **explicit boolean
active mask**, which is authoritative: every consumer (signed-h, observation, collision
predicate) reads the mask and ignores inactive slots regardless of their padded center or
radius. Activity is not inferred from "radius $> 0$" — the mask is the single source of
truth, so that a legitimately small radius can never be confused with padding and padding
values can never leak into a computation.

### 1.4 Safety signal: signed-h and collision predicate

The environment exposes the ground-truth geometric safety of a state. This is data the
agent learns from; how conservatively the agent then acts (its safety margin) is a
control-side choice defined in `02_control`.

Let $\text{dist}_i = \|p - c_i\|$ be the distance from the position $p = (px, py)$ to
obstacle center $c_i$ with radius $r_i$, over active obstacles.

- **Signed safety value** $h(x)$:
  $h_i = (r_i - \text{dist}_i) / h_{\text{scale}}$, then
  $h(x) = \mathrm{clip}(\max_i h_i,\ -1,\ 1)$.
  - $h(x) > 0$ inside / overlapping an obstacle (unsafe); $h(x) < 0$ outside (safe).
  - $h(x) = 0$ lies exactly on the collision boundary $\text{dist}_i = r_i$. No safety
    margin is baked in — the signal marks true contact, not an inflated obstacle.
  - $h_{\text{scale}} > 0$ is a fixed normalization constant
    (`base_config.env.h_scale`, default 1.0) controlling the slope; it is not a safety
    margin.
- **Collision predicate** (boolean, used for outcomes): collided iff
  $\text{dist}_i < r_i$ for any active obstacle. **No margin** — contact is exact.

The two are consistent: $h(x) \geq 0$ iff the position is at or inside an obstacle
boundary.

### 1.5 Goal and success

- Goal region: position within `goal_radius = 0.15` of the goal, with speed below
  `goal_speed_radius = 0.30`.
  - Double Integrator speed is $\|(vx, vy)\|$; Unicycle speed is $|v|$.
- These two radii are common to both systems. They may be revised later after a pure-LQR
  sanity check, but are fixed at these values for now.

### 1.6 Outcome semantics and termination

Outcomes are detected per step in this fixed priority order; the **episode outcome** is the first predicate that fires at any step, and is locked once detected (later events of the same or lower priority do not change it):

1. **Collision** — at least one active obstacle has $\text{dist} < r$.
2. **Goal reached** — position and speed predicates of §1.5, only if not collided.
3. **Out of bounds** — §1.2, only if neither collided nor reached.
4. **Stuck** — the agent has barely moved over a fixed time window. Concretely, for each step $t \ge W_{\text{stuck}}$, define the window-displacement
   $$
   \mathrm{disp}_t = \max_{k = 0, \dots, W_{\text{stuck}}} \big\| p_{t-k} - p_{t - W_{\text{stuck}}} \big\|,
   $$
   i.e. the maximum displacement of any position in the window from the window-anchor (the position $W_{\text{stuck}}$ steps ago). Stuck fires when $\mathrm{disp}_t \le r_{\text{stuck}}$. Default $W_{\text{stuck}} = 60$ steps ($= 3$ s at $\mathrm{dt} = 0.05$), $r_{\text{stuck}} = 0.10$ m. The episode's `min_window_displacement` is $\min_{t \ge W_{\text{stuck}}} \mathrm{disp}_t$ (the most-stuck moment); recorded as a diagnostic on every episode whether or not the stuck predicate fires.
5. **Timeout** — the step counter reaches the configured maximum length (`base_config.eval.max_steps` for eval; the configured training horizon for training) with none of the above firing.

So a step that simultaneously satisfies collision and goal counts as a collision; a step that satisfies goal and OOB counts as a goal; a window that satisfies the stuck predicate after the agent has already been classified as collided/reached/OOB does not change the outcome.

**Detection vs termination are separate.** The list above resolves *which outcome label an episode carries*. Whether the rollout actually stops at that step depends on the rollout mode:

- **Training rollout** (`03_train` §3): outcomes are detected per step for diagnostic, but the rollout runs the full configured horizon regardless. No early termination.
- **Evaluation rollout** (`04_eval` §2): the rollout terminates on the FIRST of {collision, goal, out-of-bounds, timeout}, mirroring real deployment for these physical outcomes. **Stuck is detected and the outcome is locked when it fires, but the rollout does NOT terminate** — it continues to `max_steps`. Real deployment does not auto-terminate when a controller oscillates in place, and continuing the rollout allows the trajectory plot to show whether the system recovers or remains stuck.

### 1.7 Scene initialization (defined elsewhere)

How a concrete scene is sampled — start/goal placement, start/goal obstacle clearance,
minimum start-goal distance, initial velocity, the unavoidable-collision rejection filter,
and any fixed (non-random) layouts — is a training/evaluation concern and differs between
the two. It is therefore defined in `03_train` §1 (training) and `04_eval` §6 (evaluation).

---

## 2. Integration and control limits

- **Integrator: RK4.** The scheme is fixed (changing it changes the discrete-time system
  and breaks comparability). A single RK4 step integrates the continuous dynamics holding
  the control constant over the step.
- **Time step `dt`** is `base_config.env.dt` (default 0.05). If numerical accuracy ever
  requires it, lower `dt` in config; do not change the scheme.
- **Control clamping.** The action is clamped to the system's control bounds (§3) before
  integration.
- **Velocity clamping (Double Integrator).** After each RK4 step, the DI velocity is
  scaled down so its speed does not exceed $v_{\max} = 2.5$
  (`base_config.env.bounds.double_integrator.v_max`): with
  $s = \lVert (vx, vy) \rVert$, the post-step velocity is multiplied by
  $\min(1,\ v_{\max} / \max(s, \epsilon))$. This bounds the worst-case approach speed, so
  the velocity-dependent worst-case future risk that $V_S$ must learn stays finite; it is
  a deliberate part of the dynamics, not a representation convention. The Unicycle has its
  own forward-speed limit through its control/state structure and is not additionally
  clamped here.
- **Angle wrapping (Unicycle only).** After each step, $\theta$ is wrapped to
  $[-\pi, \pi]$. This is a representation convention and does not alter the dynamics.

---

## 3. Systems

### 3.0 Observation model (common principles)

The agent never sees absolute position $(px, py)$; observations are translation-invariant.
Both systems use **Top-K obstacle conditioning** with $k_{\text{obs}} = 5$
(`base_config.env.k_obs`):

- The $k_{\text{obs}}$ nearest **active** obstacles (ranked by ascending surface distance
  $\text{dist}_i - r_i$, using the active mask of §1.3, never radius) are included; if
  fewer are active, remaining slots are zero-padded.
- **Tie-breaking is deterministic by ascending obstacle index.** When two obstacles share
  the same surface distance, the lower index is kept. This is the convention `torch.topk`
  provides and is required for bit-reproducible evaluation pools; ties are vanishingly
  rare under continuous sampling, so no stochastic rule is used.
- Each obstacle contributes 3 features: two relative-position components and the radius.
- The exact frame (world-relative vs body-frame) is system-specific (below), reflecting
  each system's natural symmetry.

The neural CBF and policy (`02_control`) take exactly this observation as input; they do
not redefine it.

### 3.1 Double Integrator

- **State** (dim 4): $(px, py, vx, vy)$.
- **Control** (dim 2): $(ax, ay)$, box-bounded
  $|ax|, |ay| \leq u_{\max} = 2.0$
  (`base_config.env.bounds.double_integrator.u_max`).
- **Dynamics** (control-affine, $\dot x = f(x) + g(x) u$):
  - $\dot{px} = vx$
  - $\dot{py} = vy$
  - $\dot{vx} = ax$
  - $\dot{vy} = ay$
- **Observation** (dim 19, world-relative): $[vx,\, vy,\, gx - px,\, gy - py]$ followed by
  the Top-5 obstacles, each $(cx_i - px,\, cy_i - py,\, r_i)$. Translation-invariant
  (absolute position excluded); not body-frame (the system has no heading).

### 3.2 Unicycle

- **State** (dim 4): $(px, py, \theta, v)$, with $\theta$ wrapped to $[-\pi, \pi]$.
- **Control** (dim 2): $(a, \omega)$, where $a$ is linear acceleration and $\omega$ is the
  turn-rate command ($\dot\theta = \omega$); both are control inputs. Box-bounded
  $|a| \leq a_{\max} = 2.0$, $|\omega| \leq \omega_{\max} = 3.0$
  (`base_config.env.bounds.unicycle`).
- **Dynamics** (control-affine):
  - $\dot{px} = v \cos\theta$
  - $\dot{py} = v \sin\theta$
  - $\dot\theta = \omega$
  - $\dot v = a$
- **Observation** (dim 18, body-frame):
  $[v,\, \text{goal}^{\text{body}}_x,\, \text{goal}^{\text{body}}_y]$ followed by the
  Top-5 obstacles, each $(\text{obs}^{\text{body}}_x,\, \text{obs}^{\text{body}}_y,\,
  r_i)$, where goal and obstacle relative positions are rotated into the body frame by
  $\theta$. Absolute position and $\theta$ are excluded, so the observation is invariant
  to both translation and rotation.

### 3.3 Remark: planar quadrotor (future extension)

A planar quadrotor is underactuated and is not implemented in v2 initially. It can be
brought under the same interface by wrapping a PD attitude controller that maps a desired
acceleration command to thrust and torque, reducing the quadrotor to a
double-integrator-like acceleration-controlled system. The Double Integrator safety
machinery then transfers with the PD inner loop absorbing the underactuation. This is a
planned extension, not a current definition.
