# 01 — Environment

In RL terms this document defines the **environment**: the world the agent acts in, the
state it evolves, the observation it emits, and the rules that decide collision, goal, and
termination. It does **not** define the agent (nominal policy, neural CBF, safety filter,
safety margin) — that is `02_control` — nor how scenes are sampled for training and
evaluation — that is `03_train` and `04_eval`.

Three systems are defined: **Double Integrator**, **Unicycle**, and the **planar quadrotor**
(§3.3), the last being underactuated (thrust + torque, no direct lateral force).

Run-level knobs (`dt`, control bounds, `v_init_max`, episode length, RNG seed, and other
tunables) live in `src/configs/` — see `base_config.yaml` for the rarely-changed task and
filter parameters and `exp_config.yaml` for the per-experiment hyperparameters. This
document fixes the structure and the canonical problem-defining constants.

---

## 1. Common task definition

**Provenance of constants.** A constant introduced into a system, or carried across from
another, records how it was fixed: derived from the plant, screened from a distribution, or
transferred. A transfer states whether it was verified on the receiving system, and an
unverified transfer stands as unscreened until it is screened there.

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
obstacle center $c_i$ with radius $r_i$, over active obstacles, and let the clearance to
obstacle $i$ be $\text{clr}_i = \text{dist}_i - r_i$.

- **Signed safety value** $h(x)$ (`src/common/signed_h.py`):
  $h_i = 1 - 2\,\mathrm{clamp}(\text{clr}_i / h_{\text{scale}},\ 0,\ 1)$, then
  $h(x) = \mathrm{clip}(\max_i h_i,\ -1,\ 1)$.
  - $h(x) = +1$ at and inside contact ($\text{clr}_i \le 0$); $h(x) = -1$ once every
    clearance reaches $h_{\text{scale}}$; between the two the ramp is linear with slope
    $-2 / h_{\text{scale}}$ per unit of clearance.
  - **Zero level set:** $h(x) = 0$ at clearance $\text{clr}_i = h_{\text{scale}} / 2$
    (0.5 m at the default $h_{\text{scale}}$). The zero set is an *inflated* obstacle
    boundary, not the contact surface — the safety signal the agent learns carries a
    built-in geometric margin of $h_{\text{scale}} / 2$, and every certificate, filter,
    and terminal built on $h$ inherits it.
  - $h_{\text{scale}} > 0$ (`base_config.env.h_scale`, default 1.0) sets both the ramp
    slope and the inflation. It is distinct from the control-side
    $\gamma_{\text{margin}}$ of `02_control` §5.1, which shifts $h$ and is applied on top
    of it.
- **Collision predicate** (boolean, used for outcomes): collided iff
  $\text{dist}_i < r_i$ for any active obstacle. **No margin** — contact is exact.

The two are ordered, not identical. A state on the $h$ zero set is collision-free by
$h_{\text{scale}} / 2$; contact is $h(x) = +1$. Reported `collision` outcomes always use
the exact contact predicate, never the $h$ sign.

### 1.5 Goal and success

- Goal region: position within `goal_radius = 0.15` of the goal, with speed below
  `goal_speed_radius = 0.30`.
  - Double Integrator speed is $\|(vx, vy)\|$; Unicycle speed is $|v|$.
- These two radii are common to both systems. They may be revised later after a pure-LQR
  sanity check, but are fixed at these values for now.

### 1.6 Outcome semantics and termination

Outcomes are detected per step in this fixed priority order; the **episode outcome** is the first predicate that fires at any step, and is locked once detected (later events of the same or lower priority do not change it):

1. **Collision** — at least one active obstacle has $\text{dist} < r$, or the agent has met a
   **domain surface** that its system declares physical: for `quadrotor_3d`, the arena floor and
   ceiling $|p_z| \ge \text{world\_lim}$ (§3.4). A domain surface is a collision, not an
   out-of-bounds: the obstacle set is infinite vertical cylinders, so the volume those cylinders
   span is closed below and above by physical surfaces, and meeting one ends the episode exactly as
   meeting a cylinder does. Systems that declare no domain surface are unaffected.
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

The agent never sees absolute horizontal position $(px, py)$; observations are invariant to
horizontal translation. A coordinate along which the system declares a domain surface (§1.6) is
**not** a symmetry direction and is carried explicitly — see the design rule in §3.4.
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

### 3.3 Planar quadrotor (underactuated)

- **State** (dim 6): $(px, py, \theta, vx, vy, \omega)$, with attitude $\theta$ maintained in
  $[-\pi, \pi]$.
- **Control** (dim 2): $(f_{\text{thr}}, \tau)$ — scalar body-axis thrust
  $f_{\text{thr}}$ and torque $\tau$, box-bounded $f_{\text{thr}} \in [0,\, 19.62]$,
  $\tau \in [-1.0,\, 1.0]$ (`exp_config.env.quadrotor_planar`). The channels are **decoupled
  inputs**: thrust drives $\dot v$ through the attitude-steered axis, torque drives $\dot\omega$
  only. There is no rotor/arm model — thrust and torque are independent controls — but the
  torque bound is sized **plant-coherently**: with $\mathsf m = 1.0$ and $J = 0.01$ the implied
  arm length is $L_J = \sqrt{J/\mathsf m} = 0.1$ m, whose coherent torque scale is
  $\bar\tau \approx L_J f_{\max}/2 \approx 0.98$. An undersized torque bound is a filter
  **feasibility** defect, not a mere agility choice: the box magnitude in the authority-loaded
  channel must dominate the required safety correction (the theory note's box-feasibility
  characterization).
- **Dynamics** (control-affine, $\dot x = f(x) + g(x) u$), with $Re = (-\sin\theta,\ \cos\theta)$
  the body thrust axis, mass $\mathsf m$, inertia $J$, gravity $g$ along $-y$:
  - $\dot{px} = vx$
  - $\dot{py} = vy$
  - $\dot{vx} = \tfrac{1}{\mathsf m}\, f_{\text{thr}}\, (-\sin\theta)$
  - $\dot{vy} = \tfrac{1}{\mathsf m}\, f_{\text{thr}}\, \cos\theta - g$
  - $\dot\theta = \omega$
  - $\dot\omega = \tau / J$
- **Canonical constants:** $\mathsf m = 1.0$, $J = 0.01$, $g = 9.81$, speed clamp
  $v_{\max} = 2.5$, rate clamp $\omega_{\max} = 4.0$ (`exp_config.env.quadrotor_planar`). After
  each RK4 step $\|(vx, vy)\|$ is scaled to $v_{\max}$ and $|\omega|$ to $\omega_{\max}$
  (velocity/rate clamp per §2), and $\theta$ is wrapped to $[-\pi, \pi]$.
- **Observation** (dim 22, body-frame):
  $[v^{\text{body}}_x,\, v^{\text{body}}_y,\, \omega,\, \text{goal}^{\text{body}}_x,\,
  \text{goal}^{\text{body}}_y]$ followed by the Top-5 obstacles, each
  $(\text{obs}^{\text{body}}_x,\, \text{obs}^{\text{body}}_y,\, r_i)$, followed by the attitude
  features $(\sin\theta,\ \cos\theta)$ — equivalently the body-frame gravity direction.
  **Design rule:** the observation may quotient only by the symmetry group of the closed-loop
  dynamics-plus-task. Translations are a symmetry (absolute position is excluded); rotations
  are **not** for this system — gravity fixes a world direction, so a body-frame encoding that
  also drops $\theta$ over-quotients and aliases dynamically distinct states (upright vs
  inverted), an information loss no training can remove. The attitude therefore enters
  explicitly via $(\sin\theta, \cos\theta)$; the angular rate $\omega$ is a raw scalar. (The
  Unicycle and DI are kinematic/gravity-free, so their full-SE(2)/translation quotients remain
  exact — this rule changes nothing there.)

The barrier machinery that acts on this system (the analytic attitude-augmented $h_\star$ and the
direct box-aware HardNet projection onto $(f_{\text{thr}}, \tau)$) is defined agent-side in
`02_control`, not here.

### 3.4 3D quadrotor (quadrotor_3d, underactuated)

- **State** (dim 13): position $p \in \mathbb{R}^3$, unit quaternion $q$ (attitude, $\|q\| = 1$),
  linear velocity $v \in \mathbb{R}^3$, body rates $\omega \in \mathbb{R}^3$.
- **Control** (dim 4): $u = (f_1, f_2, f_3, f_4)$ — **per-rotor thrusts**, box-bounded
  $f_i \in [0,\, \overline f_{\text{rotor}}]$ with $\overline f_{\text{rotor}} = 4.905$
  (`exp_config.env.bounds.quadrotor_3d`). Total thrust and body torque are **derived** through the
  X-mixer, not independently bounded: with arm $L = 0.17$, moment arm $l = L/\sqrt2$, and drag
  coefficient $c_{\text{m}} = 0.016$,
  $$
  f_{\text{thr}} = \textstyle\sum_i f_i,\quad
  \tau_x = l\,(f_1 + f_2 - f_3 - f_4),\quad
  \tau_y = l\,(-f_1 + f_2 + f_3 - f_4),\quad
  \tau_z = c_{\text{m}}\,(f_1 - f_2 + f_3 - f_4).
  $$
  The achievable set in $(f_{\text{thr}}, \tau)$ is therefore a **polytope, not a box**, and the two
  channels contend for the same actuator budget: the maximum roll/pitch torque
  $\tau_{\max} = 2 l \overline f_{\text{rotor}} = 1.179$ requires two rotors at zero, leaving
  $f_{\text{thr}} = \mathsf m g$, so the thrust-to-weight ratio available at torque magnitude $|\tau|$ is
  $\mathrm{TWR}_{\text{eff}}(\tau) = \mathrm{TWR}\,(1 - |\tau| / \tau_{\max}) + |\tau| / \tau_{\max}$,
  falling from $\mathrm{TWR} = 2$ at zero torque to $1$ at full torque. Any recovery or reachability
  argument that grants full thrust while commanding full torque is unsound. Plant-coherent with the
  planar system: $\mathsf m = 1.0$, $g = 9.81$, $J = \mathrm{diag}(0.01,\, 0.01,\, 0.02)$.
- **Dynamics** ($R(q)$ the body-to-world rotation, $e_3$ the world up-axis, $\otimes$ quaternion product):
  - $\dot p = v$
  - $\dot q = \tfrac{1}{2}\, q \otimes (0, \omega)$
  - $\dot v = \tfrac{1}{\mathsf m}\, f_{\text{thr}}\, R(q)\, e_3 - g\, e_3$
  - $\dot\omega = J^{-1}\,(\tau - \omega \times J\omega)$
  RK4 integration with quaternion **renormalization each step** (double-cover safe).
- **Canonical constants:** $\mathsf m = 1.0$, $J = \mathrm{diag}(0.01, 0.01, 0.02)$, $g = 9.81$; speed
  clamp $v_{\max}$ and rate clamp $\omega_{\max}$ carried from the planar values; $dt$ and
  $\text{max\_steps} = 200$ unchanged.
- **Obstacles:** infinite vertical cylinders (center $c_{xy}$, radius $r$). Surface distance
  $\phi = \|p_{xy} - c_{xy}\| - r$; radial direction $\hat r$ horizontal and globally smooth. Top-5 by
  surface distance, active mask authoritative, tie rule identical to §3.0.
- **Observation** (dim 34, full body frame): $[v^{b}(3),\, \omega^{b}(3),\, \text{goal}^{b}(3),\,
  g^{b}(3),\, p_z,\, v_z]$ followed by the Top-5 cylinders, each $(c_{\text{off}}^{b}(3),\, r_i)$; the
  two scalars $p_z$ (absolute altitude) and $v_z$ (world vertical velocity) are appended in world
  coordinates. All remaining vectors are expressed
  in the body frame $R(q)^\top$, with $g^{b} = R(q)^\top(-e_3)$ and $c_{\text{off}}^{b} =
  R(q)^\top(\Delta c_{xy},\, 0)$ (information-complete for infinite cylinders). The state $\omega$ is
  **already** the body-frame rate (the kinematics use $\dot q = \tfrac12 q\otimes(0,\omega)$), so
  $\omega^{b} = \omega$ is reported directly — no $R(q)^\top$ is applied to it; only the world-frame
  quantities ($v$, $\text{goal}-p$, $-e_3$, $\Delta c_{xy}$) are rotated. The closed-loop symmetry
  group is $G_{\text{dyn}} = \text{horizontal translations} \times \text{yaw}$; $g^{b}$ restores exactly
  the gravity-referenced tilt component a full-rotation quotient would alias, and $(p_z, v_z)$ carry the
  vertical coordinate that the floor/ceiling surfaces (§1.6) and the vertical hazard term make
  observable. Both added scalars are invariant under horizontal translation and yaw, so the retained
  quotient is exactly preserved. Double-cover safe (all quantities pass through $R(q)$).

  **Design rule (observation vs hazard).** The observation may quotient only symmetries that the
  hazard *also* has. Equivalently: every argument of $h_\star$ must be determined by the observation,
  modulo the retained group. Adding a hazard term that reads a coordinate the observation has
  quotiented away does not make that term hard to learn — it makes it **unlearnable**, since states
  with different hazard map to one observation and the value cannot represent its own target. The
  check is performed when the hazard changes, not when the run is deployed: for each new term, name
  the coordinates it reads and point to where each is recoverable from the observation. Vertical
  translation left $G_{\text{dyn}}$ the moment $|p_z| = \text{world\_lim}$ became a physical surface,
  which is why $p_z$ is carried; $v_z$ is recoverable as $-(v^{b} \cdot g^{b})$ and is carried anyway,
  so that a lead term linear in descent rate is an affine function of the input rather than a bilinear
  one the network must synthesize.
- **Scene:** region $[-4, 4]^3$. Cylinder xy-centers, radii, counts, active mask, and the start/goal
  **xy-clearance** rules inherited from the planar scene distribution verbatim (clearance in xy only —
  cylinders are infinite). Start and goal $z$ sampled **independently** uniform in $[-4, 4]$ (altitude
  change is generically part of the task).
- **Initial pose** (6-DOF perturbed): position as above; attitude **Haar-uniform on $SO(3)$**, drawn
  by Shoemake's method, so $\cos\theta \sim U[-1, 1]$ for the thrust-axis tilt $\theta$ and inversions are
  included. Three quarters of the draws therefore start past the altitude-holding limit
  $\theta_{\text{hold}} = \arccos(\mathrm{TWR}^{-1}) = 60^\circ$, beyond which no admissible thrust
  arrests a descent; this is a deliberate stress on the vertical channel, not an accident of sampling.
  $\|v_0\|$ from the planar speed distribution with uniform 3D direction; per-axis $\omega_0$ from the
  planar range.
- **Barrier / nominal:** the hazard carries a horizontal branch per cylinder and two vertical branches
  for the domain surfaces of §1.6,
  $$
  h_\star = \max\Big\{\ \max_i\big[\phi_i + 0.3\,(v_{xy}\cdot\hat r_i)\big],\quad
  \min(\psi_{\text{lo}}, \overline\psi) - c_z v_z,\quad
  \min(\psi_{\text{up}}, \overline\psi) + c_z v_z\ \Big\},
  $$
  with vertical margins $\psi_{\text{lo}} = -p_z - \text{world\_lim}$,
  $\psi_{\text{up}} = p_z - \text{world\_lim}$. Both vertical branches take the $\psi + c\,\dot\psi$ form of
  the horizontal one: the margin alone has $\partial\psi/\partial v_z = 0$ and so relative degree 2, and the
  lead term restores relative degree 1 with $L_g$ supported on the thrust channel. The **cap**
  $\overline\psi = r_{\max}$ is required because there is no ground-contact model, so $\psi$ is unbounded
  below the floor; capping it at the largest contribution a cylinder can make keeps both surfaces on one
  scale in the $\max$. The lead gain $c_z = 0.8$ is the flip time at the plant's own rate limit,
  $\pi/\omega_{\max}$; a shorter lookahead turns the branch positive only after the descent can no longer be
  arrested, and the horizontal $0.3$ does not transfer because the binding vertical timescale is attitude
  recovery, not closure. $h_{\text{scale}}$ unchanged. Nominal is a hover **cascaded-PD** (thrust-axis
  projection + attitude error), replacing a hover-linearized LQR because the Haar attitude distribution
  lies far outside a hover linearization's validity region; the box-aware HardNet projection onto the
  per-rotor box is defined agent-side in `02_control`. Per-system rule:
  `filter.empty_fallback = {mode: kstep, phases: 1, k: 3}` (`02_control` §4).