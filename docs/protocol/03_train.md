# 03 — Training

This document defines how the value network (and, in Joint Training, the policy) is
learned. It assumes the environment (`01_env`) and the agent machinery (`02_control`); it
defines training **structure, procedure, and algorithms**. All numeric values live in
`src/configs/` (`base_config.yaml` and `exp_config.yaml`); evaluation is `04_eval`.

Equations are written in LaTeX for MkDocs Material's MathJax rendering. They are the
authoritative form used in any paper derived from this work.

## 0.1 Clean-baseline principle

The baseline is a minimal, theoretically grounded reproduction of each framework.
Auxiliary mechanisms (extra anchor classes, late-floor adaptive noise, lateral bonuses,
gradient-floor losses) are excluded from the baseline; they are listed in §7 as one-axis
additions, each independently ablatable. Keeping the baseline minimal is what makes its
results attributable.

---

## 1. Scene initialization (training)

The training sampler differs from the evaluation sampler only in the parameter values
listed below; the sampling **mechanism** (obstacle field generation, start/goal placement,
rejection filters) is the same. The eval-side overrides are defined in `04_eval` §6.

### 1.1 Draw order (authoritative)

Because corridor-biased obstacle placement is defined relative to the start-goal segment,
start and goal must be drawn before the obstacle centers. One sampling attempt draws, in
this fixed order:

1. active obstacle count $n \sim \text{Uniform}\{n_{\text{obs,min}}, \dots,
   n_{\text{obs,max}}\}$;
2. start $s$;
3. goal $g$;
4. the corridor decision (Bernoulli $p_{\text{corr}}$);
5. the $n$ obstacle centers (corridor band around the $s$–$g$ segment, or uniform);
6. the $n$ obstacle radii;
7. the initial velocity (and heading, for the Unicycle).

The whole attempt is then checked against the acceptance predicates of §1.3. If any
predicate fails, the **entire scene is rejected and the attempt is repeated** (a fresh
draw of all of the above), with a hard retry cap of 1000; exceeding the cap raises rather
than relaxing any clearance, distance, or margin. This fixed order, together with an
explicitly passed RNG (no module-level random state), is what makes evaluation pools
bit-reproducible. Inactive slots beyond $n$ are zero-padded per `01_env` §1.3.

### 1.2 Distributions

- **Obstacle count, radii:** $n \sim \text{Uniform}\{n_{\text{obs,min}},
  n_{\text{obs,max}}\}$; each radius $\sim \text{Uniform}[r_{\min}, r_{\max}]$.
- **Obstacle centers:** with probability $p_{\text{corr}}$ placed in a corridor band
  around the $s$–$g$ segment (along-range $1.5\,L$, perpendicular $1.0\,L$,
  $L = \|g - s\|$); otherwise uniform in the arena. All parameters in
  `base_config.obstacle.*`.
- **Start and goal:** drawn uniformly in
  $[-(\text{world\_lim} - 0.3),\ \text{world\_lim} - 0.3]^2$. For systems whose scene region is 3-D,
  the vertical coordinate is drawn uniformly over the full arena band for both start and goal, with
  no clearance from the band surfaces. Training must experience the region the value is meant to
  score, and a start adjacent to a hazard surface is such a region.
- **Attitude and body rates (3-D systems):** attitude Haar-uniform on $SO(3)$, per-axis $\omega_0$
  from the configured range (`01_env` §3.4). The vertical span is **not** narrowed for tilted draws.
  Evaluation restricts it — a start that no admissible control can hold is a measurement artefact
  there (`04_eval` §6.1) — but training must experience the region the value is meant to score, and
  a doomed start yields a correct $\sup_t h$ target. Excluding it would leave the value untrained
  exactly where the certificate is later asked to speak.
- **Initial velocity (not zero):** each component (Double Integrator) or the scalar speed
  (Unicycle) drawn $v_0 \sim \text{Uniform}(-v_{\text{init,max}}, v_{\text{init,max}})$,
  $v_{\text{init,max}} = \texttt{env.v\_init\_max}$. For the Unicycle the heading
  $\theta_0 \sim \text{Uniform}[-\pi, \pi]$.

### 1.3 Acceptance predicates

A drawn attempt (§1.1) is accepted iff all of the following hold; otherwise the whole
scene is resampled.

**Distance and clearance.**

$$
\|g - s\| \geq d_{\min}^{\text{train}},
\quad
\min_i \big(\|s - c_i\| - r_i\big) \geq \delta^{\text{train}},
\quad
\min_i \big(\|g - c_i\| - r_i\big) \geq \delta^{\text{train}},
$$

with $d_{\min}^{\text{train}}$ and $\delta^{\text{train}}$ from
`base_config.scene_train`, over active obstacles only.

**Unavoidable-collision rejection.** A start $(s, v_0)$ that cannot avoid contact even
with maximum braking is rejected. For every active obstacle $i$, let
$\hat u_i = (c_i - s) / \|c_i - s\|$ be the unit direction from the start toward the
obstacle center (well-defined since the clearance predicate above forces
$\|c_i - s\| \ge r_i + \delta^{\text{train}} > 0$); compute the inward speed — the
initial velocity projected onto that direction —

$$
v^{\text{in}}_i = \max\!\big(0,\ v_0 \cdot \hat u_i\big),
$$

the corresponding stopping distance under maximum braking

$$
d^{\text{stop}}_i = \frac{(v^{\text{in}}_i)^2}{2\, u_{\max}},
$$

and reject the start iff there exists $i$ with

$$
\|c_i - s\| \;<\; r_i + d^{\text{stop}}_i + \delta^{\text{feas}},
$$

where $\delta^{\text{feas}} = \texttt{scene\_train.init\_feasibility\_margin}$ (default
0.05) and $u_{\max}$ is the system's acceleration bound (Double Integrator: $u_{\max}$;
Unicycle: $a_{\max}$ along the heading direction). When $v_0 = 0$, $v^{\text{in}}_i = 0$
and the test reduces to the clearance buffer already enforced above.

### 1.4 Scene schema

A sampled scene is a fixed-shape record carrying: obstacle centers `[n_max, 2]` and radii
`[n_max]` (zero-padded), a boolean active mask `[n_max]` (authoritative per `01_env`
§1.3), start position `[2]`, goal position `[2]`, and initial velocity in the system's
representation (`[2]` for DI; scalar speed plus heading for Unicycle). The bit-reproducible
draw order and the retry cap are specified in §1.1.

### 1.5 Train vs eval

Training and evaluation share one sampler; only the following differ between the two
modes:

- start-goal minimum distance $d_{\min}$ (train tighter than eval);
- start/goal obstacle clearance $\delta$ (train tighter than eval);
- any obstacle-field difficulty overrides specified by `04_eval`.

Evaluation values are not restated here. See `04_eval` for the eval-side parameters and
`base_config.eval.*` for the canonical values.

---

## 2. Common value-learning scaffold

Both frameworks learn the value network of `02_control` (signed $h$ with the raw-linear
output head clipped to $[-1, 1]$ only at read-out, per `02_control` §3.2) with the same
target form and the same optimizer/Polyak convention. The training MSE regresses the raw
forward output against the clamped target; deployed $h$ uses the mean-ensemble clipped
value and the target uses the min-ensemble clipped value (`02_control` §3.4). Only the
data distribution and the policy involvement differ between the frameworks.

### 2.1 Value target (switchable)

The regression target form is selected per experiment by
`exp_config.value_target.type`, one of `pncbf` (one-step discounted avoid) or `rpcbf`
(multi-step softmax). The JT trainer's active target is `pncbf`. The `rpcbf` form and the
dispatch helper are defined but are not wired into the JT value loss, so selecting `rpcbf`
does not change the JT target; a change to the JT target form is its own axis and carries
the wiring with it.

#### 2.1.1 PNCBF — one-step discounted avoid

With per-step cost $c_t = h(x_t)$ (signed safety value from `01_env` §1.4, clamped to
$[-1, 1]$) and discount

$$
\gamma_t = \exp\!\big(-\lambda_t\, dt\big),
$$

the avoid value is the backward recurrence

$$
A_T = c_T,
\qquad
A_t = \max\!\Big(c_t,\ (1 - \gamma_t)\, c_t + \gamma_t\, A_{t+1}\Big),
$$

with the truncated tail bootstrapped from the target network
$A_{T+1} \leftarrow V_S^{\text{target}}(x_{T+1})$, then the result clamped to
$[-1, 1]$. The discount rate $\lambda_t$ is curriculum-scheduled (§4.6); its config name
is `schedules.gamma_disc`.

The target presented to the value regressor is then mixed toward the full bootstrap by the
$\text{tgt\_rhs}$ schedule (§4.6):

$$
\tilde A_t = A_t + \tau_{\text{rhs}} \cdot \max\!\big(A_t^{\text{full}} - A_t,\ 0\big),
$$

where $A_t^{\text{full}}$ is the un-truncated avoid recurrence.

#### 2.1.2 RPCBF — multi-step smooth-max over an $H$-step rollout

Under the target policy, simulate $H = \texttt{value\_target.rpcbf\_H}$ steps forward
starting from $x_t$, producing $\{x_{t+k}\}_{k=0}^{H}$ ($H + 1$ states, including $x_t$)
with each rollout step under $V_S^{\text{target}}$-filtered actions. The multi-step target
is the **normalized** log-mean-exp smooth-max

$$
V^{\text{rp}}_t = \frac{1}{\beta}\log\!\left(\frac{1}{H+1}\sum_{k=0}^{H}
\exp\!\big(\beta\, h(x_{t+k})\big)\right),
\qquad \beta = \texttt{value\_target.rpcbf\_beta},
$$

then clamped to $[-1, 1]$. This lies between $\mathrm{mean}_k\, h(x_{t+k})$ and
$\max_k\, h(x_{t+k})$ and increases monotonically toward $\max_k h(x_{t+k})$ as
$\beta \to \infty$. The $\tfrac{1}{H+1}$ normalization makes the target **horizon-invariant**
(the un-normalized log-sum-exp carries a $\tfrac{1}{\beta}\log(H+1)$ bias that depends on
$H$ and consumes a meaningful fraction of the clamped $[-1,1]$ range). When
`value_target.type` is `rpcbf`, $V^{\text{rp}}_t$ replaces $\tilde A_t$ as the regression
target. A blend $(1-\rho)\tilde A_t + \rho V^{\text{rp}}_t$ is reserved as a one-axis
addition (§7).

**Trade-off — loss of the $V \ge h$ upper bound.** The PNCBF / RPCBF guarantee that the
policy value function is a valid CBF rests on $V^{h,\pi}(x) \ge h(x)$: the value
upper-bounds the worst future safety cost, so its zero sublevel set is forward-invariant
(So et al. 2024, Thm. 1; Knoedler 2025, Eq. 7). The **un-normalized** log-sum-exp target
satisfies this automatically, since it is always $\ge \max_k h(x_{t+k}) \ge h(x_t)$. The
**normalized** log-mean-exp target defined above does **not**: lying between mean and max,
it can fall below $\max_k h$, so the regressed $V_S$ is a smooth under-estimate of the
true max-over-time value and is therefore slightly **optimistic on the unsafe side**.
The default value loss accepts this trade-off for horizon-invariance and to avoid
output-head saturation within $[-1, 1]$; conservativeness on the unsafe region can be
restored by switching the value target to the un-normalized log-sum-exp form as a one-axis
alternative (§7); the unsafe anchor is prohibited (§4.3).

### 2.2 Value loss, optimizer, target network

Let the ensemble be $\{V_S^{(m)}\}_{m=1}^{n_{\text{vs}}}$. For minibatch $\mathcal{B}$
drawn from the value buffer with regression target $y_t$ (§2.1), the **avoid-regression
loss** is

$$
\mathcal{L}_{R} \;=\; \frac{1}{n_{\text{vs}}|\mathcal{B}|}
\sum_{m=1}^{n_{\text{vs}}}\sum_{t \in \mathcal{B}}
\Big(V_S^{(m)}(x_t,\, \text{obs}_t) - y_t\Big)^2.
$$

Both frameworks use the same value-loss form

$$
\mathcal{L}_{V} = \lambda_R\, \mathcal{L}_{R} + \lambda_A\, \mathcal{L}_{A}
+ \lambda_C\, \mathcal{L}_{C},
\qquad \lambda_R, \lambda_A, \lambda_C \in \texttt{loss.value}.
$$

**By default $\lambda_A = \lambda_C = 0$ for both frameworks** (pure PNCBF MSE
recurrence). §4.3 defines the anchor mechanism that those terms invoke; it is retained in
the spec for optional reactivation by setting nonzero anchor weights, but it is inactive
in the JT and OC-PNCBF defaults.

**Optimizer.** AdamW with `optim.lr_VS`, `optim.weight_decay`. Weight decay is applied to
all parameters in the value network (no exclusions). After each backward pass, gradients
are clipped:

$$
\|\nabla_\theta \mathcal{L}_V\|_2 \;\leftarrow\; \min\!\big(\|\nabla_\theta \mathcal{L}_V\|_2,\
\texttt{optim.grad\_clip}\big),
$$

then `opt.step()` is called. No learning-rate scheduler is used; learning rates are
constant for the run.

**Target network.** A target copy $V_S^{\text{target}}$ is initialized as a `deepcopy` of
$V_S$ with `requires_grad = False`. After each value optimizer step it is updated by Polyak
averaging,

$$
\theta_{V_S^{\text{target}}} \;\leftarrow\; (1 - \tau_P)\, \theta_{V_S^{\text{target}}}
+ \tau_P\, \theta_{V_S},
\qquad \tau_P = \texttt{optim.tau\_polyak},
$$

applied parameter-wise to every registered parameter (including every ensemble member).
The Polyak update runs **per value optimizer step** (`optim.polyak_per_v_step = true` is
the default). The macro-step alternative (one Polyak update per macro step regardless of
$K_V$) is available but not the default.

### 2.3 Collection distribution and the V-to-zero collapse

A structural requirement of this learning scheme: **the value network needs both safe and
unsafe trajectory signal**. If collection produces only safe rollouts, the avoid target is
uniformly safe, $V_S$ regresses toward zero everywhere, the CBF goes inactive, and safety
collapses. This is a collection-distribution property, not an exploration bug.

The framework lineage from `00_constitution` makes the asymmetry explicit.
**OC-PNCBF** uses a fixed LQR nominal that collides in many random scenes (especially
corridors), so its rollouts naturally carry unsafe signal — collection is
self-regularizing. **Joint Training** extends OC-PNCBF by learning the policy through
HardNet (lifting the LQR ceiling), but inherits the matching risk: once policy + filter
are good, rollouts become safe and the unsafe signal vanishes. This is structural, not a
transient bug. The mitigation is to collect the value buffer under action noise
$\sigma > 0$ (§4.2), with $\sigma$ targeted by an adaptive rule based on the unsafe
fraction of recent rollouts (§4.6).

### 2.4 Collection rollout — the continuing-batch collector

Collection rollouts (OC §3.1 and both JT buffers §4.2) are produced by one collector with
two modes: `collector = continuing | legacy` (default `continuing`; `legacy` — fixed-horizon
fresh-IC rounds — is retained solely for ablation and bit-parity testing). The continuing
mode is normative:

1. **Outcome-agnostic with respect to physical events.** Collision and OOB never terminate a
   collection episode; outcome predicates remain per-step diagnostics (`01_env` §1.6).
   Obstacles are not part of the plant dynamics, so post-contact pass-through trajectories
   are valid trajectories of the free dynamics: interior and post-contact states are
   retained in the buffer as unsafe-side value anchors — they carry the strongest unsafe
   labels and pin the certificate's gradient direction at the safety boundary. Masking,
   truncating, or relabeling them is prohibited.
2. **Information-saturation truncation.** An episode is truncated only when continued
   stepping adds no further label information: (R1) the eval goal criterion holds, plus
   `k_hover` further steps (settling data is retained deliberately); (R2) stationarity —
   total position displacement below `stationary_thresh` over `stationary_window`
   consecutive steps (covers stuck-like dwelling without importing an outcome concept into
   collection); (R3) per-episode timeout `episode_timeout` = the eval `max_steps`, so
   training labels cover the same horizon evaluation scores.
3. **Persistent rows with round carry-over.** Collection rounds are a scheduling quantity
   only. Each batch row carries persistent episode state across rounds (state, scene,
   per-episode step counter, exploration-noise state); an episode cut by a round boundary
   continues in the next round from the same state. Consequence: no episode phase is
   systematically under- or over-sampled — early- and late-episode states enter the buffer
   at their natural visitation rates.
4. **Refill on truncation only.** A truncated row is reinitialized with a fresh scene + IC
   and its counter reset; the batch shape stays constant and every simulated step is a
   useful step (no post-goal hover flooding of the buffer).
5. **Per-segment labeling.** Buffers store variable-length episode segments (a segment ends
   at a truncation or a round boundary). The label recursion closes every segment with the
   target-net bootstrap at that segment's tail state and never crosses an episode boundary
   within a row. A round-boundary segment is an ordinary truncation: the bootstrap
   summarizes the continuation whose realized data the next round supplies.
6. **Buffers store states, not observations.** Observations are recomputed at consumption
   time, so buffer contents are independent of the observation definition.

Keys: `continuing.k_hover = 20`, `continuing.stationary_window = 20`,
`continuing.stationary_thresh = 0.05`, `continuing.episode_timeout = 200`. Required tests
whenever the collector is touched: legacy bit-parity; segment isolation (a segment's labels
invariant to the next episode sharing its row); R1/R2/R3 unit firing; round-boundary and
mid-round-refill carry-over; a fixed-seed semantic-equivalence check against the frozen
reference implementation.

---

## 3. Framework A — OC-PNCBF (value-only)

A fixed nominal policy (LQR); learn only $V_S$. The CBF-QP is a deployment/eval filter,
not part of training collection.

### 3.1 Collection

Roll out the fixed LQR nominal (`02_control` §2) on sampled scenes, recording the
signed-h history. No CBF filter is applied during collection. Trajectories enter a FIFO
buffer (`collection.oc_pncbf.buffer_capacity`).

**Rollout termination follows the continuing-batch collector (§2.4).** Collision and OOB
never terminate (post-collision and post-OOB states are exactly the strongest unsafe labels
for PNCBF's avoid-target backward recurrence and are kept in the buffer); information-
saturated tails truncate and refill per §2.4 R1–R3 with round carry-over. Stuck detection
during training remains diagnostic-only and does not alter collection.

### 3.2 Loop

Per epoch (`exp_config.training.oc_pncbf`):

1. Collect `collection.oc_pncbf.collect_size` rollouts of length
   `training.oc_pncbf.horizon`.
2. Take `training.oc_pncbf.grad_steps_per_epoch` gradient steps on $\mathcal{L}_V$
   (with $\lambda_A = \lambda_C = 0$ by default) over minibatches of size
   `optim.batch_size_oc`.
3. Polyak-update the target network per the convention of §2.2.

Total epochs `training.oc_pncbf.epochs`.

### 3.3 Filtering

The learned $V_S$ is used by the CBF-QP filter only at evaluation (`04_eval`). Training
never filters. This keeps the value target a property of the nominal policy, as PNCBF
intends.

---

## 4. Framework B — Joint Training (value and policy)

Co-train $V_S$ and the control network through the differentiable HardNet filter.

### 4.1 Co-training loop

Each macro step runs the following block. The full pseudocode is in §5; the per-section
text below details each component.

1. (Optional) collect rollouts into the two buffers if
   $n \bmod \texttt{collection.jt.collect\_every} = 0$ (§4.2).
2. $K_V$ value steps on $\mathcal{L}_V$ (§2.2, §4.3 optional anchors) over minibatches of
   size `optim.batch_size_jt_value` drawn from $\mathcal{D}_V$. Polyak target update after
   each value step.
3. After the value-only warmup (§4.5), $K_\pi$ policy steps on $\mathcal{L}_\pi$ (§4.4)
   over BPTT rollouts of length `training.jt.bptt_T` and policy minibatches of size
   `optim.batch_size_jt` (BPTT memory budget) drawn from $\mathcal{D}_\pi$.

The value-step minibatch size and the policy-step BPTT minibatch size are independent
config keys (`optim.batch_size_jt_value` vs `optim.batch_size_jt`). This separation is
necessary because the BPTT minibatch is memory-bounded by `bptt_T` while the value step
is not, so the value update can use a much larger batch than the policy update.

### 4.2 Two-buffer collection

Two buffers serve the two updates:

- $\mathcal{D}_V$ (exploratory) — collected under the current policy plus Gaussian action
  noise. The pre-filter action is $\hat u_t = \pi_\theta(\text{obs}_t) + \sigma_n \epsilon$
  with $\epsilon \sim \mathcal{N}(0, I)$, then HardNet-filtered with $V_S^{\text{target}}$.
  $\sigma_n$ is adaptive (§4.6).
- $\mathcal{D}_\pi$ (on-policy) — collected with little or no noise
  (`schedules.sigma_pi`), reflecting the actions the policy actually executes. Filter is
  applied as in $\mathcal{D}_V$.

Both rollouts are detached (collection does not back-prop). Per collection cycle,
`collection.jt.episodes_per_collect` episodes feed each buffer. Each buffer stores both a
trajectory view (for labeling) and a transition view (for V and policy minibatching), and
samples its minibatches uniformly at random (with replacement) over the transition view.
Buffer capacities are FIFO trajectory counts: once a buffer exceeds its cap, the oldest
trajectory is evicted, so a cap is the sliding-window length (in trajectories) of retained
collection history. $\mathcal{D}_V$ and the structural **precursor** buffer (inert unless the
precursor-injection addition (§7) is active) use `collection.jt.buffer_cap`. The on-policy
$\mathcal{D}_\pi$ uses `collection.jt.policy_buffer_cap`, an independent cap; when it is unset
or equal to `buffer_cap`, all buffers share one capacity. Standard configuration:
`buffer_cap` = 1,000,000 ($\mathcal{D}_V$ + precursor); `policy_buffer_cap` = 2,000
($\mathcal{D}_\pi$).

Both buffers are fed by the continuing-batch collector (§2.4): entries are variable-length
episode segments; the trajectory view labels each segment to its own bootstrap-closed tail
(no recursion across an episode boundary within a row), and the transition view samples
uniformly over transitions exactly as before. Buffer contents are states; observations are
recomputed at consumption (§2.4 item 6).

### 4.3 Weak value supervision and minibatch composition (PROHIBITED)

*The anchor mechanism is prohibited. $\lambda_A$ and $\lambda_C$ are an arbitrary loss with no
grounding in the theory; the anchor weights are zero and per-iteration anchor minibatches are
not built, and the value step uses only the transition minibatch with $\mathcal{L}_R$ (§2.2).
The section remains so that what is prohibited can be read; references to anchor labels in §5
are skipped.*

In addition to the avoid target, $V_S$ may be anchored by two weak label sets re-extracted
from collected trajectories at every value-step iteration. With per-trajectory step index
$t$, time-to-next-collision $\tau^{\text{col}}_t$ (steps until the next collision, $\infty$
if the trajectory is collision-free), and future signed-h window
$h^{\text{fut}}_t = \max_{k = 0..k_{\text{safe}}} h(x_{t+k})$:

- **$\mathcal{D}_A$ (unsafe anchor)** — include $x_t$ if either
  $\tau^{\text{col}}_t \leq k_A$ (pre-collision window) or $h(x_t) \geq -m_{\text{phys}}$
  (near-unsafe physical band, when $m_{\text{phys}} > 0$). Target hinge $y^A = +
  \gamma_{\text{margin}}$.
- **$\mathcal{D}_C$ (safe anchor)** — collision-free trajectory and
  $h^{\text{fut}}_t \leq -\delta_C$. Target hinge $y^C = -\gamma_{\text{margin}}$.

The exact hinge losses are

$$
\mathcal{L}_A = \frac{1}{|\mathcal{D}_A|}\!\sum_{x \in \mathcal{D}_A}\!
\mathrm{ReLU}\!\big(\gamma_{\text{margin}} - V_S(x) + 10^{-3}\big)^2,
\qquad
\mathcal{L}_C = \frac{1}{|\mathcal{D}_C|}\!\sum_{x \in \mathcal{D}_C}\!
\mathrm{ReLU}\!\big(V_S(x) + \gamma_{\text{margin}} + 10^{-3}\big)^2.
$$

Labeling constants: $k_A$, $\delta_C$, $k_{\text{safe}}$, $m_{\text{phys}}$ from
`labeling.*` in `exp_config.yaml`.

**Per-iteration minibatch composition (when anchors are active).** Each value step builds
a minibatch with:

- $|\mathcal{B}_{\text{trans}}| = \texttt{optim.batch\_size\_jt\_value}$ transitions
  sampled uniformly from the transition view of $\mathcal{D}_V$ (carries $\mathcal{L}_R$);
- $|\mathcal{B}_A| = \texttt{optim.batch\_size\_jt\_value} / 4$ states sampled with
  replacement from the freshly labeled $\mathcal{D}_A$;
- $|\mathcal{B}_C| = \texttt{optim.batch\_size\_jt\_value} / 4$ states sampled with
  replacement from the freshly labeled $\mathcal{D}_C$.

If a label set is empty (rare early in training), that anchor contributes zero loss for
that iteration. When anchors are inactive, only $\mathcal{B}_{\text{trans}}$ is built.

### 4.4 Policy loss — BPTT through HardNet and dynamics

The policy $\pi_\theta$ is trained by BPTT of a task return through the HardNet projection
and the RK4 dynamics, with $V_S$ **detached** so no gradient reaches its parameters
(`02_control` §7; gradient-leak threshold $10^{-9}$).

#### 4.4.1 Detach mechanism (required)
 Detaching the scalar value of $h$ is insufficient —
HardNet still needs $\partial h / \partial x$ to project. The correct mechanism is to set
`requires_grad_(False)` on all $V_S$ parameters for the duration of the policy backward
pass (or an equivalent frozen-parameter context), then restore it. This lets $\partial h /
\partial x$ flow through the projection while blocking any gradient from reaching $V_S$'s
weights. The gradient-leak halt (§4.7) verifies this held.

#### 4.4.2 Rollout and task cost

Over horizon $T = \texttt{training.jt.bptt\_T}$, the initial states $x_0$ are sampled
uniformly from the transition view of $\mathcal{D}_\pi$; goals and obstacles are taken
from the same buffer entry (not freshly generated). At each step the policy proposes the
pre-projection action $u^{\text{nom}}_t = \pi_\theta(\text{obs}_t)$, the HardNet projects
it to $u^{\text{safe}}_t$, and the state advances
$x_{t+1} = \text{RK4}(x_t, u^{\text{safe}}_t)$. The task cost per step is

$$
c^{\text{task}}_t = \|p_t - g\|^2 + \lambda_v\, \|v_t\|^2
+ w_s\, e^{-\|p_t-g\|^2/\rho_s^2}\, \|v_t\|^2
+ w_a \sum_{k} \mathrm{relu}\big(s_{k,t}\,\tau_b - d_{k,t}\big)^2
+ \mu_u\, \|u^{\text{safe}}_t\|^2,
$$

with $\lambda_v$ and $\mu_u$ from `loss.policy`, and two **situational velocity terms** (keys
`loss.policy.{w_settle, settle_rho, w_appr, tau_brake}`; all default $0$, which recovers the
plain cost — DI/Unicycle parity):

- **Goal-gated settling** ($w_s = $ `w_settle`, $\rho_s = $ `settle_rho`): penalizes speed only
  near the goal, converting arrival into settling densely at every step. This is the running
  replacement for the sparse velocity terminal below; the two should not be active together.
- **Braking-envelope approach** ($w_a = $ `w_appr`, $\tau_b = $ `tau_brake`): with
  $s_k = \mathrm{relu}(-v^\top n_k)$ the inward closure speed toward obstacle $k$ and $d_k$ its
  **surface** distance, the per-obstacle deficit is
  $m_k \cdot \mathrm{relu}(s_k \tau_b - d_k)$, where $m_k$ is the **active-obstacle mask**
  (padded or inactive slots carry $m_k = 0$); the mask is part of the term's definition, since
  an unmasked sum over padded slots contributes phantom deficit. The deficit is a soft
  stopping-distance constraint — exactly $0$ when receding or outside the envelope, engaging
  earlier the faster the approach — and it shapes the demand side of filter feasibility.

#### 4.4.3 Return and terminal

The discounted return is

$$
R(\theta; x_0) = -\sum_{t=0}^{T-1} \gamma_T^t\, c^{\text{task}}_t
\;-\; \gamma_T^{T}\,\big( w_{\text{term}}\, \|p_T - g\| + w_{\text{term},v}\, \|v_T\| \big),
\qquad \gamma_T = \texttt{loss.policy.gamma\_T}.
$$

The final term is an **end-of-horizon terminal** with weights
$w_{\text{term}} = \texttt{loss.policy.w\_terminal}$ and
$w_{\text{term},v} = \texttt{loss.policy.w\_terminal\_v}$: the position part credits closing
distance to the goal beyond the BPTT window, which the windowed sum alone cannot reward. It uses
the analytic goal distance (not the learned $V_S$, which is a hazard value), and is
differentiable through $x_T$. Setting $w_{\text{term}} = 0$ recovers the plain windowed return;
it is $0$ by default and used only where the fixed window $T$ is short relative to the time to
reach the goal. The velocity part $w_{\text{term},v}$ is a **sparse** settling signal, fired once per window and
blind to where; its default is $0$, the dense goal-gated term above superseding it.
Per-system standard: quadrotor
$w_s = 1.0$, $\rho_s = 0.30$, $w_a = 30$, $\tau_b = 0.6$, $w_{\text{term}} = 30$,
$w_{\text{term},v} = 0$; DI/Unicycle: all four situational keys $0$.

The rollout integrates the cost over the full fixed horizon $T$ and does not terminate or mask
mid-window when a trajectory reaches the goal, penetrates an obstacle, or leaves the arena;
$\text{wrap\_state}$ bounds velocity but not position. It applies the single end-of-horizon
terminal above but **no in-window terminal or termination**.
The window therefore carries return accrued after a physically-terminal event, and that
interacts with horizon length; the reserved cps-floor and early-stop halts (§4.7-3..5) exist
against it. A change to in-window termination or to the terminal value is its own axis.

#### 4.4.4 Region gating and the policy loss

The objective is **region-wise**, gated by the current $V_S$ via soft indicators (with
$V_S$ detached for the gates):

$$
g_{\text{in}}(x) = \sigma\!\big(-V_S(x)/\tau_{\text{gate}}\big),
\qquad
g_{\text{out}}(x) = \sigma\!\big(V_S(x)/\tau_{\text{gate}}\big),
\qquad \tau_{\text{gate}} = \texttt{loss.policy.tau\_gate}.
$$

Inside the safe region (gate $g_{\text{in}}$) the policy maximizes $R$; outside (gate
$g_{\text{out}}$) it minimizes the terminal value $V_S(x_T)$ to recover toward safety. The
full policy loss is

$$
\boxed{\ \mathcal{L}_\pi = \mathbb{E}_{x_0\sim\mathcal{D}_\pi}\!\Big[
g_{\text{in}}(x_0)\cdot(-R(\theta; x_0))
\;+\; w_{\text{out}}\cdot g_{\text{out}}(x_0)\cdot V_S(x_T)
\Big]
\;+\; \mathcal{L}_{\text{reg}}\ }
$$

with $w_{\text{out}} = \texttt{loss.policy.w\_outside}$.

#### 4.4.5 Regularizers and optimizer

All four regularizers are evaluated on the *pre-projection* policy outputs along the
BPTT rollout. Let $a^{\text{nom}}_t = u^{\text{nom}}_t = \pi_\theta(\text{obs}_t)$ at step
$t$, and let $z_t$ denote the policy's pre-activation output (pre-tanh or pre-softsign).

$$
\mathcal{L}_{\text{a}} \;=\; \lambda_{\text{a}}\, \mathbb{E}\Big[\sum_i
(a^{\text{nom}}_{t,i})^2\Big],
\qquad
\mathcal{L}_{\text{s}} \;=\; \lambda_{\text{s}}\, \mathbb{E}\Big[\sum_i
(a^{\text{nom}}_{t,i} - a^{\text{nom}}_{t-1,i})^2\Big]_{t \geq 1},
$$

$$
\mathcal{L}_{\text{sat}} \;=\; \lambda_{\text{sat}}\, \mathbb{E}\Big[\sum_i
\mathrm{ReLU}\!\big(|a^{\text{nom}}_{t,i}| - u^{\text{thr}}\big)^2\Big],
\qquad u^{\text{thr}} = \texttt{loss.policy.sat\_excess\_threshold},
$$

$$
\mathcal{L}_{\text{pre}} \;=\; \lambda_{\text{pre}}\, \mathbb{E}\Big[g_{\text{vs}}(x_t)
\cdot \mathrm{mean}_i\, \mathrm{ReLU}\!\big(|z_{t,i}| - z^{\text{target}}\big)^2\Big],
$$

with $z^{\text{target}} = \mathrm{atanh}(0.70) \approx 0.867$
(`loss.policy.z_target`) and the optional $V_S$-gated weight

$$
g_{\text{vs}}(x) = \sigma\!\big((-V_S^{\text{det}}(x) - 0.02) / \tau_{\text{vs-gate}}\big),
\qquad \tau_{\text{vs-gate}} = \texttt{loss.policy.vs\_gate\_tau}
$$

active iff `loss.policy.vs_gated_pretanh = true`; otherwise $g_{\text{vs}} \equiv 1$.
Total: $\mathcal{L}_{\text{reg}} = \mathcal{L}_{\text{a}} + \mathcal{L}_{\text{s}}
+ \mathcal{L}_{\text{sat}} + \mathcal{L}_{\text{pre}}$.

Optimizer: AdamW with `optim.lr_pi`, same `weight_decay` and `grad_clip` as the value
optimizer (§2.2).

### 4.5 $V_S$ warmup and schedule clock

Two integer clocks govern Joint Training and are **independent**:

- $n_{\text{steps}} = \texttt{training.jt.n\_steps}$ — total macro steps to train.
- $n_{\text{sched\_total}} = \texttt{training.jt.schedule\_n\_steps}$ — total macro steps
  over which curriculum schedules unroll. **Independent of $n_{\text{steps}}$**; may be
  larger (schedule does not fully unroll within training), smaller (schedule completes
  before training ends), or equal. If unset, defaults to $n_{\text{steps}}$.

The CLI exposes both as `--jt-n-steps` and `--schedule-n-steps`.

$V_S$ is trained alone for $\texttt{training.jt.vs\_warmup\_steps}$ macro steps before
policy updates begin (during warmup, $K_\pi$ is forced to 0). Curriculum schedules use the
**offset step**

$$
n_{\text{sched}} = \max\!\big(0,\ n - \texttt{vs\_warmup\_steps}\big),
$$

so the curriculum starts when the policy activates, not during warmup. Without this offset
the curriculum advances during warmup and the policy activates mid-curriculum.

### 4.6 Schedules

All schedules use the offset step $n_{\text{sched}}$ above and operate over the **schedule
effective horizon**

$$
N_{\text{eff}} = n_{\text{sched\_total}} - \texttt{vs\_warmup\_steps}.
$$

When $n_{\text{sched\_total}} > n_{\text{steps}}$ the schedule unrolls only partially over
training; when $n_{\text{sched\_total}} \le n_{\text{steps}}$ schedules reach and hold
their final values before training ends. Setting $n_{\text{sched\_total}}$ independently
of $n_{\text{steps}}$ is the canonical way to gentle or steepen the bootstrap/horizon
ramp without changing training length.

Each schedule is parameterized by an initial value $v_0$, a final value $v_\infty$, a
warmup fraction $f_w$, and a phase-1 fraction $f_1$. The base **two-segment** form

$$
\text{val}(s) =
\begin{cases}
v_0 & s < f_w N_{\text{eff}}, \\[2pt]
v_0 + \dfrac{s - f_w N_{\text{eff}}}{(f_1 - f_w) N_{\text{eff}}}\, (v_\infty - v_0)
  & f_w N_{\text{eff}} \leq s < f_1 N_{\text{eff}}, \\[6pt]
v_\infty & s \geq f_1 N_{\text{eff}}.
\end{cases}
$$

is used by `target_rhs` and `sigma_pi`. The `gamma_disc` schedule uses a **three-segment**
form (below) because both the effective horizon $H$ and the rate $\lambda$ ramp.

**`gamma_disc`** — the avoid-value discount $\gamma_t = \exp(-\lambda_t\, dt)$. The
schedule is horizon-anchored over three segments:

1. **Warmup** ($s < f_w N_{\text{eff}}$): hold horizon at $H_0$, so
   $\lambda_t = -\ln(1 - 1/H_0)/dt$.
2. **Phase 1** ($f_w N_{\text{eff}} \le s < f_1 N_{\text{eff}}$): ramp horizon linearly
   from $H_0$ to $H_\infty$; compute $\lambda_t = -\ln(1 - 1/H_t)/dt$ at every step.
3. **Phase 2** ($s \ge f_1 N_{\text{eff}}$): hold horizon at $H_\infty$ and ramp
   $\lambda_t$ linearly from its phase-1 endpoint down to the floor
   $\lambda_{\text{floor}}$ over the remaining schedule horizon.

Anchors: $H_0 = 20$ steps, $H_\infty = 100$ steps, $\lambda_{\text{floor}} = 0.05$,
$f_w = 0.10$, $f_1 = 0.80$. Because horizons are expressed in steps and $\lambda$ divides
by $dt$, the resulting $\gamma$ is correct at any $dt$ (the discount horizon in real time
scales with $dt$, but the step-horizon is invariant). The discount is never specified as
a raw $\gamma$ literal; it is always computed from $H_t$ (or $\lambda_t$ in phase 2) and
$dt$.

**Floor note (learned-value runs).** The theory note's schedule-blindness corollary places
the slow-drift detectability floor at $H_0 \ge 37$ under the standard tripwire cadence; the
$H_0 = 20$ anchor predates it. This is inactive under the maneuver channel ($K_V = 0$); any
future learned-value run must either raise $H_0 \ge 37$ or justify the deviation in
`changes.md`.

**`target_rhs`** — mixing of the target toward the full bootstrap (§2.1): ramps from $0$
(pure MC max-over-time) to a hold value, engaging the discounted TD-bootstrap correction
only after $V_S$ has stabilized. Defaults: $v_0 = 0.0$, $v_\infty = 0.9$, $f_w = 0.10$,
$f_1 = 0.80$.

**`sigma`** — $\mathcal{D}_V$ collection noise (§4.2), **adaptive by unsafe-fraction
target**. Let

$$
\rho_n^{\text{unsafe}} = \frac{1}{|\mathcal{B}_{\text{coll}}|}\sum_{\text{episode}}\!
\mathbf{1}\!\big[\exists t:\ h_{\text{geom}}(x_t) > 0\big]
$$

be the unsafe-episode fraction in the latest collection batch (using the instantaneous
geometric signed-h of `01_env` §1.4, not the value network). The target sigma is

$$
\sigma_n^{\text{tgt}} =
\begin{cases}
\sigma_{\max} & \text{if } \rho_n^{\text{unsafe}} < \rho_{\text{target}}, \\
\sigma_{\min} & \text{otherwise},
\end{cases}
$$

and sigma EMAs toward target:

$$
\sigma_n = \mathrm{clip}\!\big((1 - \beta_\sigma)\, \sigma_{n-1} + \beta_\sigma\,
\sigma_n^{\text{tgt}},\ \sigma_{\min},\ \sigma_{\max}\big),
$$

with $\rho_{\text{target}} = \texttt{schedules.sigma.rho\_target}$ (default $0.10$),
$\beta_\sigma = \texttt{schedules.sigma.beta\_sigma}$ (default $0.05$),
$\sigma_{\min} = \texttt{schedules.sigma.sigma\_min}$ (default $0.30$),
$\sigma_{\max} = \texttt{schedules.sigma.sigma\_max}$ (default $2.0$),
$\sigma_0 = \texttt{schedules.sigma.init}$ (default $0.5$). The mechanism is: when the
collection batch is too safe, push $\sigma$ toward the noise ceiling so perturbed actions
reach unsafe states; when unsafe signal is already adequate, decay $\sigma$ toward the
floor. The pinning of $\sigma_n^{\text{tgt}}$ at $\sigma_{\max}$ for several consecutive
cycles is reserved as a halt signal (§4.7; currently inactive).

*Note: earlier protocol drafts defined `sigma` by an EMA of projection magnitude with
knobs `beta_proj`, `alpha_dom`. Those knobs remain in `exp_config.yaml` for compatibility
but are unused by the active JT trainer and are not part of this spec.*

**`sigma_pi`** — $\mathcal{D}_\pi$ noise decay (§4.2):

$$
\sigma_{\pi,n} = \sigma_{\pi,0}\cdot \max\!\big(0,\ 1 - n / D\big),
\qquad
\sigma_{\pi,0} = \texttt{schedules.sigma\_pi.init},\quad
D = \texttt{schedules.sigma\_pi.decay\_steps}.
$$

Default $\sigma_{\pi,0} = 0$ (on-policy collection is zero-noise; the policy buffer
already reflects executed behavior). Late-training schedule additions are listed in §7.

### 4.7 Halt protocol

The trainer halts and writes a final checkpoint whenever any of the **active** conditions
below triggers. Thresholds in `exp_config.halt`.

**Active halts:**

1. **NaN / Inf in $V$ or policy loss.** Hard halt at the offending step; save current
   state.
2. **$V_S$ gradient leak.** During every policy step, compute
   $\|\nabla_{V_S\text{params}} \mathcal{L}_\pi\|_2$; if this exceeds
   `halt.vs_grad_leak_threshold` ($10^{-9}$), halt. This enforces the detach convention
   of `02_control` §7.

**Reserved halts (defined in spec, not currently activated by the trainer):** the
following halt conditions are specified here so their semantics, thresholds, and config
keys are fixed, but the JT trainer does not currently wire them into its halt path.
Activation of any reserved halt is a one-axis future addition (§7) and must be added to
the trainer's halt path before the corresponding `exp_config.halt.*` threshold has any
runtime effect.

3. **$\sigma$ pinned at $\sigma_{\max}$.** If $\sigma_n^{\text{tgt}}$ equals
   $\sigma_{\max}$ for `halt.sigma_max_cycles` consecutive collection cycles, halt:
   exploration is failing to find unsafe signal even at the noise ceiling.
4. **`cps` floor.** Once warmup is past, if the in-loop `cps` drops below
   `halt.cps_floor` (default $-0.5$), halt: training has collapsed. The floor is a threshold on a
   metric, so it is only meaningful against that metric's reachable range: a change to the outcome
   predicates or to `cps` shifts the whole range and the floor is re-derived from the new one, never
   carried across or relaxed to fit. A floor placed outside the reachable range on a given stage is a
   specification error (`00_constitution` §4), and a stage whose policy is fixed cannot move `cps` at
   all, so the floor carries no information there and is not applied.
5. **Early stop on no improvement.** If the best in-loop `cps` has not improved by
   `halt.early_stop_min_delta` within `halt.early_stop_patience` macro steps, halt.

Two further detectors are listed as reserved one-axis additions only and have no spec or
config keys yet: a deep-saturation detector (fraction of value-head outputs with
$|h| > 0.95$ sustained above a threshold) and a catastrophic-failure detector (per-eval
`cps` slope below a threshold). Their thresholds depend on baseline statistics not yet
collected.

---

**Advisory gradient watch (no auto-kill).** If the pre-clip policy gradient norm exceeds
10x its running median on 2 consecutive logging points, the Executor sends an immediate
notification with the values and continues training; stopping is a Researcher decision.
This is an advisory channel, not a halt condition: clipped-gradient degradation can evade
the NaN/leak hard halts while still destroying the policy, and the watch exists to surface
it in time.

## 5. Algorithm — one macro step of Joint Training

The following is the authoritative ordering. The OC-PNCBF loop is a subset (only the
collection and value blocks; no policy step; no warmup).

> **Inputs:** macro-step index $n$, value net $V_S$, target $V_S^{\text{target}}$, policy
> $\pi_\theta$, buffers $\mathcal{D}_V, \mathcal{D}_\pi$, configs.
>
> 1. $n_{\text{sched}} \leftarrow \max(0,\ n - \texttt{vs\_warmup\_steps})$ over
>    $N_{\text{eff}} = n_{\text{sched\_total}} - \texttt{vs\_warmup\_steps}$.
> 2. $\gamma_{\text{disc},n} \leftarrow \texttt{schedules.gamma\_disc}(n_{\text{sched}})$,
>    $\tau_{\text{rhs},n} \leftarrow \texttt{schedules.target\_rhs}(n_{\text{sched}})$.
> 3. **Collection** (if $n = 1$ or $n \bmod \texttt{collect\_every} = 0$):
>    1. Draw scene batch with `episodes_per_collect` scenes (§1.1–§1.3).
>    2. Roll out $\pi_\theta + \mathcal{N}(0, \sigma_n^2 I) +$ HardNet on
>       $V_S^{\text{target}}$; append to $\mathcal{D}_V$ (trajectory + transition views).
>    3. Roll out $\pi_\theta + \mathcal{N}(0, \sigma_{\pi,n}^2 I) +$ HardNet on
>       $V_S^{\text{target}}$; append to $\mathcal{D}_\pi$.
>    4. Measure $\rho_n^{\text{unsafe}}$ over the latest collection batch and update
>       $\sigma_n$ via the EMA-toward-target rule (§4.6).
> 4. **Value updates.** For $k = 1 \ldots K_V$:
>    1. (Optional, anchors active only) Sample $\texttt{v\_label\_episodes}$ trajectories
>       from $\mathcal{D}_V$ and rebuild labels $\mathcal{D}_A, \mathcal{D}_C$ (§4.3).
>    2. Build minibatch: $\mathcal{B}_{\text{trans}}\sim\mathcal{D}_V^{\text{trans}}$ of
>       size $\texttt{optim.batch\_size\_jt\_value}$, plus $\mathcal{B}_A, \mathcal{B}_C$
>       at sizes specified in §4.3 only when anchors are active.
>    3. Compute the value target $y$ (§2.1; RPCBF override if active).
>    4. Compute $\mathcal{L}_V = \lambda_R \mathcal{L}_R + \lambda_A \mathcal{L}_A
>       + \lambda_C \mathcal{L}_C$ (with $\lambda_A = \lambda_C = 0$ by default);
>       finite-check; halt on NaN/Inf (§4.7-1).
>    5. Zero V grads; backward; clip $\|\nabla_{\theta_V}\|_2 \leq
>       \texttt{grad\_clip}$; `opt_vs.step()`.
>    6. Polyak update $V_S^{\text{target}} \leftarrow (1-\tau_P)V_S^{\text{target}}
>       + \tau_P V_S$ (§2.2, per-V-step convention).
> 5. **Policy updates** (skip if $n \leq \texttt{vs\_warmup\_steps}$). For
>    $k = 1 \ldots K_\pi$:
>    1. Sample $x_0$, goal, obstacles from $\mathcal{D}_\pi^{\text{trans}}$ at
>       minibatch size $\texttt{optim.batch\_size\_jt}$.
>    2. With $V_S$ params set to `requires_grad = False`, BPTT-roll out $T$ steps under
>       $\pi_\theta +$ HardNet $+$ RK4; accumulate $R$ (§4.4).
>    3. Compute $\mathcal{L}_\pi$; finite-check; halt on NaN/Inf (§4.7-1).
>    4. Zero policy grads; backward; check $\|\nabla_{V_S\text{params}} \mathcal{L}_\pi\|_2
>       \leq 10^{-9}$ else halt (§4.7-2); clip
>       $\|\nabla_{\theta_\pi}\|_2 \leq \texttt{grad\_clip}$; `opt_pi.step()`.
> 6. **Logging.** Append a row to `metrics.csv` and a TensorBoard scalar batch with all
>    loss components, schedule values, and gradient diagnostics.
> 7. **In-loop evaluation.** If $n \bmod \texttt{eval.cadence} = 0$, run a full pool
>    rollout (`04_eval` §2.1), append a row to `eval_metrics.csv`, append per-episode
>    rows to `eval_episodes.csv`, log to TensorBoard, and update `best.pt` by `cps`.
> 8. **Halt checks** (active halts §4.7-1, §4.7-2 only; reserved halts §4.7-3..5 are
>    not currently wired into this step).
> 9. **Final.** When $n = n_{\text{steps}}$ or an active halt triggers, save `final.pt`,
>    auto-generate `report.md`, run the full evaluation (`04_eval` §2.2).

---

## 6. Smoke stage

The trainer accepts `--stage smoke` to verify the full loop end-to-end with cheap settings
before any real run. Smoke mode automatically applies the following caps:

| key | smoke override |
|---|---|
| `n_steps` | $\min(n_{\text{steps}}, 10)$ |
| `collect_every` | 1 |
| `episodes_per_collect` | $\min(\cdot, 8)$ |
| `batch_size_jt` | $\min(\cdot, 64)$ |
| `batch_size_jt_value` | $\min(\cdot, 64)$ |
| `v_label_episodes` | $\min(\cdot, 8)$ |
| `log_every` | 1 |
| `eval.cadence` | 5 |
| `bptt_T` | $\min(\cdot, 10)$ |

Beyond shortening, smoke runs two gradient-routing assertions:

1. After the first value step:
   $\|\nabla_{V_S}\mathcal{L}_V\| > 0$ and $\|\nabla_{\pi}\mathcal{L}_V\| = 0$.
2. After the first policy step:
   $\|\nabla_\pi \mathcal{L}_\pi\| > 0$ and $\|\nabla_{V_S}\mathcal{L}_\pi\| < 10^{-12}$.

A failure of either halts immediately. The convention is: **a smoke run must pass before
any full run is launched** for the same `exp_config`.

---

## 7. One-axis additions

Mechanisms reserved for later versions, each to be introduced individually with a stated
hypothesis and a clean ablation against the baseline:

- **Anchor reactivation** — turn on $\mathcal{L}_A$ and/or $\mathcal{L}_C$ from §4.3 by
  setting nonzero anchor weights; the minibatch composition rule of §4.3 then applies.
- **Value-target blend** — $\rho V^{\text{rp}} + (1-\rho)\tilde A$ instead of the binary
  choice between PNCBF and RPCBF.
- **Un-normalized RPCBF target** — the strict $V \ge h$ log-sum-exp form
  $\frac{1}{\beta}\log\sum_k \exp(\beta h_{t+k})$ instead of the normalized log-mean-exp
  (§2.1.2), recovering the analytic CBF upper-bound at the cost of an $H$-dependent bias.
- **Auxiliary value-loss terms** — $\varepsilon$-gradient floor, consistency loss,
  Lipschitz loss.
- **Additional anchors** — $\mathcal{D}_{\text{far}}$, $\mathcal{D}_{\text{brake}}$.
- **Late-floor sigma** on $\mathcal{D}_V$ collection noise.
- **IC injection** (`inject_frac`, default 0 = bit-parity) — collection-side oversampling of
  a declared cell predicate using fresh standard-sampler draws only (never eval-pool states
  or their rollouts). Any use requires a registered natural-mass pre-gate demonstrating the
  cell is actually rare under the sampler; predicates whose natural mass is not small
  cannot be meaningfully oversampled. The single-band tilt predicate
  ($|\theta_0| > \pi/2$, natural mass $\approx 0.5$) is a refuted instantiation and may not
  be reintroduced. Corridor-cell START-injection is refuted: when the target cell is dynamically transient,
  episode-start injection cannot materially raise state-level visitation (a structural ceiling independent of the
  natural-mass pre-gate). The pre-gate therefore also requires a statement on the cell's dynamical persistence
  before any injection axis is registered.
- **Lateral/collinear bonus** in the policy task return, and collinear scene sampling.
- **Policy action smoothness curriculum** (smoothness weight ramped over training).
- **Alternative output heads** for the value network.
- **Policy architecture variants** — residual head, attention encoder.
- **Filter-coefficient gradient detachment** — an optional policy-BPTT hygiene flag
  `loss.policy.detach_filter_coeffs`. When enabled, the HardNet projection's
  state-dependent CBF coefficients ($L_f h$, $L_g h$, $\alpha h$) are computed in the
  forward pass exactly as normal but treated as constants w.r.t. the state in the backward
  pass, so the policy gradient reaches $\pi_\theta$ only through the $u^{\text{nom}}$
  pathway and the state chain, not through the coefficients' state dependence. Forward
  numerics are unchanged (detach alters no values); flag-off is bit-identical to the
  standard path. Differentiating through those coefficients — the $h$-Hessian, the $\alpha$
  jump at $h=0$, the box-argmin geometry — compounds multiplicatively over a long BPTT window
  and produces a heavy per-sample policy-gradient tail; the flag removes that tail and leaves
  the bulk of the gradient distribution unchanged. Default **off** for
  the Double Integrator and Unicycle; **on** for the planar quadrotor, where the long-horizon
  coefficient-gradient tail otherwise dominates the BPTT update and clips the task-credit away
  (forward is byte-identical either way, so this changes only the backward path).
- **Reserved halts (§4.7-3..5) activation** — wire the sigma-pin, cps-floor, and
  early-stop halts into the trainer's halt path. Partially outstanding: the cps-floor halt is wired
  in the OC trainer and not in the JT trainer, so the spec and the code disagree on its status;
  reconciling the two is part of this item.
- **Saturation and catastrophic-failure halt detectors** with thresholds set once baseline
  statistics are available.
- **Collision-precursor injection** — a third (precursor) buffer of synthetic near-obstacle
  initial states, mixed into each value minibatch at a fixed fraction to sharpen $V_S$'s
  danger boundary in collision-critical states that on-policy collection under-visits. Each
  precursor is built around a random active obstacle: a clearance $d \sim U[d_{\text{lo}},
  d_{\text{hi}}]$ sets the radial offset $r_i + d$ from the obstacle center, an inward speed
  $s \sim U[s_{\text{lo}}, s_{\text{hi}}]$ sets the velocity toward the obstacle, and a
  lateral spread (`lateral_frac`) perturbs it perpendicular; the velocity is capped at the
  system's `v_max` (and, for the Unicycle, re-expressed as heading + speed). The precursor
  buffer is populated each collection by rolling the current policy through the HardNet
  filter at $\sigma = 0$ over the fixed horizon (no early stop), labeled by the same
  max-over-time signed-$h$ target as every other buffer state. At each value update,
  $\text{round}(\texttt{fraction}\cdot\texttt{batch\_size})$ states are drawn from the
  precursor buffer and the remainder from $\mathcal{D}_V$, a single MSE over the mixed
  minibatch; the injected fraction is per-batch and independent of buffer sizes, and flag-off
  (or an empty precursor buffer) reduces bit-identically to the two-buffer baseline. Config:
  `loss.value.precursor_injection.{enabled, fraction, d_lo, d_hi, s_lo, s_hi, lateral_frac}`.

---

## 8. Config references

All values are in `src/configs/`:

- `base_config.yaml` — task (env, obstacle, scene_train), filter constants
  (HardNet $\varepsilon$, CBF-QP parameters, $\gamma_{\text{margin}}$), network structure
  (value, control), LQR weights, eval pools and scene. Locked.
- `exp_config.yaml` — `run` identity; `training.{jt|oc_pncbf}` (including
  `training.jt.n_steps`, `training.jt.schedule_n_steps`, `training.jt.vs_warmup_steps`,
  `training.jt.K_V`, `training.jt.K_pi`, `training.jt.bptt_T`); `optim` (including
  `optim.batch_size_jt`, `optim.batch_size_jt_value`, `optim.batch_size_oc`,
  `optim.lr_VS`, `optim.lr_pi`, `optim.weight_decay`, `optim.grad_clip`,
  `optim.tau_polyak`); `value_target`; `schedules`; `collection`; `loss.{value|policy}`;
  `labeling`; `halt`; `filter.alpha_{safe|unsafe}`; `eval.cadence`.

This document does not duplicate values; conflicts are resolved in favor of the configs,
which are written into every run's `data/<run_id>/config.yaml` (per `05_code` §3).

---

## 9. In-loop and final evaluation

### In-loop evaluation (during training)

Every `eval.cadence` macro steps (`exp_config.yaml`) the trainer runs an evaluation pass
on the **pinned in-loop pool** (`base_config.eval.in_loop`, $N = 500$, seed 12345) and
appends one row to `eval_metrics.csv`, the per-episode breakdown to `eval_episodes.csv`,
and the scalar values to TensorBoard. The **best-by-`cps`** checkpoint is saved to
`data/<run_id>/checkpoints/best.pt`. Mode details, output schema, and trajectory plots
live in `04_eval` §2.1, §3, §7.

### Final evaluation (after training)

When training terminates (by completion or halt), the trainer triggers a single **full
evaluation** on `data/<run_id>/checkpoints/best.pt` (override to `--last` for the last
checkpoint), using the disjoint full pool (`base_config.eval.full`, $N = 2000$,
seed 23456), with all three trajectory variants (LQR-only / learned no-CBF / filtered),
trajectory PNGs, online insertion, and bootstrap statistics. Mode details, output layout,
and reporting conventions live entirely in `04_eval` §2.2, §3, §4, §5, §7.