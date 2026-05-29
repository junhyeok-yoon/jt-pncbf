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
  $[-(\text{world\_lim} - 0.3),\ \text{world\_lim} - 0.3]^2$.
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
with maximum braking in the velocity direction is rejected. For every active obstacle $i$,
let $\hat v_0 = v_0 / \|v_0\|$ (when $\|v_0\| > 0$); compute the inward velocity component

$$
v^{\text{in}}_i = \max\!\big(0,\ \hat v_0 \cdot (c_i - s)\big),
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
Unicycle: $a_{\max}$ along the heading direction). When $v_0 = 0$ this reduces to the
clearance buffer $\delta^{\text{train}}$ already enforced above.

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

Both frameworks learn the value network of `02_control` (signed $h$, tanh-bounded
$[-1, 1]$) with the same target form and the same optimizer/Polyak convention. Only the
data distribution and the policy involvement differ.

### 2.1 Value target (switchable)

The regression target form is selected per experiment by
`exp_config.value_target.type`, one of `pncbf` (one-step discounted avoid) or `rpcbf`
(multi-step softmax).

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
target. (A blend $(1-\rho)\tilde A_t + \rho V^{\text{rp}}_t$ is a one-axis addition, §7.)

**Trade-off — loss of the $V \ge h$ upper bound (deliberate v2.0.0 choice).** The PNCBF /
RPCBF guarantee that the policy value function is a valid CBF rests on
$V^{h,\pi}(x) \ge h(x)$: the value upper-bounds the worst future safety cost, so its zero
sublevel set is forward-invariant (So et al. 2024, Thm. 1; Knoedler 2025, Eq. 7). The
**un-normalized** log-sum-exp target satisfies this automatically, since it is always
$\ge \max_k h(x_{t+k}) \ge h(x_t)$. The **normalized** log-mean-exp target chosen here does
**not**: lying between mean and max, it can fall below $\max_k h$, so the regressed $V_S$ is
a smooth under-estimate of the true max-over-time value and is therefore slightly
**optimistic on the unsafe side** (it may report a state as less dangerous than the
worst-case bound would). This is accepted in v2.0.0 for two reasons: (1) horizon-invariance
and avoidance of tanh-head saturation within $[-1,1]$, as above; (2) the $\mathcal{D}_A$
unsafe anchor (§4.3) independently forces $V_S \ge \gamma_{\text{margin}}$ on pre-collision
and near-unsafe states, supplying the lower bound on $V_S$ over the unsafe region that the
log-mean-exp target gives up — so conservativeness is recovered through supervision rather
than through the target's analytic upper-bound property. Restoring the strict $V \ge h$
target (un-normalized log-sum-exp, accepting the $H$-dependent bias) is available as a
one-axis alternative (§7) should the anchor compensation prove insufficient.

### 2.2 Value loss, optimizer, target network

Let the ensemble be $\{V_S^{(m)}\}_{m=1}^{n_{\text{vs}}}$. For minibatch $\mathcal{B}$
drawn from the value buffer with regression target $y_t$ (§2.1), the **avoid-regression
loss** is

$$
\mathcal{L}_{R} \;=\; \frac{1}{n_{\text{vs}}|\mathcal{B}|}
\sum_{m=1}^{n_{\text{vs}}}\sum_{t \in \mathcal{B}}
\Big(V_S^{(m)}(x_t,\, \text{obs}_t) - y_t\Big)^2.
$$

JT adds the weak-supervision anchors of §4.3; OC-PNCBF does not. The total value loss is

$$
\mathcal{L}_{V} = \lambda_R\, \mathcal{L}_{R} + \lambda_A\, \mathcal{L}_{A}
+ \lambda_C\, \mathcal{L}_{C},
\qquad \lambda_R, \lambda_A, \lambda_C \in \texttt{loss.value}.
$$

The ensemble training target uses $\min_m$ over members; the deployed $h$ uses $\max_m$,
per `02_control` §3.4.

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
In v2.0.0 this update runs **per value optimizer step** (`optim.polyak_per_v_step = true`).
The macro-step alternative (one Polyak update per macro step regardless of $K_V$) is
available but not the default.

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
$\sigma > 0$ (§4.2), so perturbed actions still reach unsafe states.

---

## 3. Framework A — OC-PNCBF (value-only)

A fixed nominal policy (LQR); learn only $V_S$. The CBF-QP is a deployment/eval filter,
not part of training collection.

### 3.1 Collection

Roll out the fixed LQR nominal (`02_control` §2) on sampled scenes, recording the
signed-h history. No CBF filter is applied during collection. Trajectories enter a FIFO
buffer (`collection.oc_pncbf.buffer_capacity`).

**No early termination during training rollout.** Per `01_env` §1.6, outcome predicates
(collision, goal, OOB, stuck, timeout) are evaluated per step for diagnostic, but the
training rollout runs the full configured horizon regardless of which outcome fires. Post-
collision and post-OOB states remain informative for PNCBF's avoid-target backward
recurrence (they are exactly the strongest unsafe labels), so they are kept in the
buffer. Stuck detection during training is logged for diagnostic only and does not alter
collection.

### 3.2 Loop

Per epoch (`exp_config.training.oc_pncbf`):

1. Collect `collection.oc_pncbf.collect_size` rollouts of length
   `training.oc_pncbf.horizon`.
2. Take `training.oc_pncbf.grad_steps_per_epoch` gradient steps on $\mathcal{L}_V$
   (with $\lambda_A = \lambda_C = 0$, since OC-PNCBF does not use weak anchors) over
   minibatches of size `optim.batch_size_oc`.
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
2. $K_V$ value steps on $\mathcal{L}_V$ (§2.2 + §4.3) over minibatches of size
   `optim.batch_size_jt`. Polyak target update after each value step.
3. After the value-only warmup (§4.5), $K_\pi$ policy steps on $\mathcal{L}_\pi$ (§4.4)
   over BPTT rollouts of length `training.jt.bptt_T`.

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
`collection.jt.episodes_per_collect` episodes feed each buffer; buffer capacity is
`collection.jt.buffer_cap`. Each buffer stores both a trajectory view (for labeling) and a
transition view (for V minibatching).

### 4.3 Weak value supervision and minibatch composition

In addition to the avoid target, $V_S$ is anchored by two weak label sets re-extracted
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

**Per-iteration minibatch composition.** Each value step builds a minibatch with:

- $|\mathcal{B}_{\text{trans}}| = \texttt{optim.batch\_size\_jt}$ transitions sampled
  uniformly from the transition view of $\mathcal{D}_V$ (carries $\mathcal{L}_R$);
- $|\mathcal{B}_A| = \texttt{optim.batch\_size\_jt} / 4$ states sampled with replacement
  from the freshly labeled $\mathcal{D}_A$;
- $|\mathcal{B}_C| = \texttt{optim.batch\_size\_jt} / 4$ states sampled with replacement
  from the freshly labeled $\mathcal{D}_C$.

If a label set is empty (rare early in training), that anchor contributes zero loss for
that iteration.

### 4.4 Policy loss — BPTT through HardNet and dynamics

The policy $\pi_\theta$ is trained by BPTT of a task return through the HardNet projection
and the RK4 dynamics, with $V_S$ **detached** so no gradient reaches its parameters
(`02_control` §7; gradient-leak threshold $10^{-9}$).

**Detach mechanism (required).** Detaching the scalar value of $h$ is insufficient —
HardNet still needs $\partial h / \partial x$ to project. The correct mechanism is to set
`requires_grad_(False)` on all $V_S$ parameters for the duration of the policy backward
pass (or an equivalent frozen-parameter context), then restore it. This lets $\partial h /
\partial x$ flow through the projection while blocking any gradient from reaching $V_S$'s
weights. The gradient-leak halt (§4.7) verifies this held.

Over horizon $T = \texttt{training.jt.bptt\_T}$, the initial states $x_0$ are sampled
uniformly from the transition view of $\mathcal{D}_\pi$; goals and obstacles are taken
from the same buffer entry (not freshly generated). At each step the policy proposes the
pre-projection action $u^{\text{nom}}_t = \pi_\theta(\text{obs}_t)$, the HardNet projects
it to $u^{\text{safe}}_t$, and the state advances
$x_{t+1} = \text{RK4}(x_t, u^{\text{safe}}_t)$. The task cost per step is

$$
c^{\text{task}}_t = \|p_t - g\|^2 + \lambda_v\, \|v_t\|^2 + \mu_u\, \|u^{\text{safe}}_t\|^2,
$$

with $\lambda_v$ and $\mu_u$ from `loss.policy`. The discounted return is

$$
R(\theta; x_0) = -\sum_{t=0}^{T-1} \gamma_T^t\, c^{\text{task}}_t,
\qquad \gamma_T = \texttt{loss.policy.gamma\_T}.
$$

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

**Regularizers.** All four are evaluated on the *pre-projection* policy outputs along the
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
g_{\text{vs}}(x) = \sigma\!\big((-V_S^{\text{det}}(x) - 0.02) / \tau_{\text{vs-gate}}\big)
$$

active iff `loss.policy.vs_gated_pretanh = true`; otherwise $g_{\text{vs}} \equiv 1$.
Total: $\mathcal{L}_{\text{reg}} = \mathcal{L}_{\text{a}} + \mathcal{L}_{\text{s}}
+ \mathcal{L}_{\text{sat}} + \mathcal{L}_{\text{pre}}$.

**Optimizer.** AdamW with `optim.lr_pi`, same `weight_decay` and `grad_clip` as the value
optimizer (§2.2).

### 4.5 $V_S$ warmup and schedule offset

$V_S$ is trained alone for `training.jt.vs_warmup_steps` macro steps before policy updates
begin (during warmup, $K_\pi$ is forced to 0). All curriculum schedules use an **offset
step**

$$
n_{\text{sched}} = \max\!\big(0,\ n - \texttt{vs\_warmup\_steps}\big),
$$

so the curriculum starts when the policy activates, not during warmup. Without this offset
the curriculum advances during warmup and the policy activates mid-curriculum.

### 4.6 Schedules

All schedules use the offset step $n_{\text{sched}}$ above and operate over the effective
horizon $N_{\text{eff}} = n_{\text{steps}} - \texttt{vs\_warmup\_steps}$. Each schedule is
parameterized by an initial value $v_0$, a final value $v_\infty$, a warmup fraction
$f_w$, and a phase-1 fraction $f_1$:

$$
\text{val}(s) =
\begin{cases}
v_0 & s < f_w N_{\text{eff}}, \\[2pt]
v_0 + \dfrac{s - f_w N_{\text{eff}}}{(f_1 - f_w) N_{\text{eff}}}\, (v_\infty - v_0)
  & f_w N_{\text{eff}} \leq s < f_1 N_{\text{eff}}, \\[6pt]
v_\infty & s \geq f_1 N_{\text{eff}}.
\end{cases}
$$

**`gamma_disc`** — the avoid-value discount $\gamma_t$ used in the §2.1 recurrence, derived
from a continuous rate $\lambda_t$ via $\gamma_t = \exp(-\lambda_t\, dt)$. The schedule is
**horizon-anchored**, not a direct $\gamma$ ramp: it interpolates an effective horizon
$H_t$ (in steps) and maps it to the rate by $\lambda_t = -\ln(1 - 1/H_t) / dt$. Anchors:
horizon $H_0 = 20$ steps (warmup), ramping to $H_\infty = 100$ steps, with a final
rate floor $\lambda_{\text{floor}} = 0.05$; warmup fraction $f_w = 0.10$, phase-1 fraction
$f_1 = 0.80$. Because the horizon is expressed in steps and $\lambda$ divides by $dt$, the
resulting $\gamma$ is automatically correct at any $dt$ (the discount horizon in real time
scales with $dt$, but the step-horizon is invariant). The discount is never specified as a
raw $\gamma$ literal; it is always computed from $H_t$ and $dt$.

**`target_rhs`** — mixing of the target toward the full bootstrap (§2.1): $\beta_t$ ramps
from $0$ (pure MC max-over-time) to a hold value, engaging the discounted TD-bootstrap
correction only after $V_S$ has stabilized. Defaults: $v_0 = 0.0$, $v_\infty = 0.9$,
$f_w = 0.10$, $f_1 = 0.80$.

**`sigma`** — $\mathcal{D}_V$ collection noise (§4.2), **adaptive** by an EMA on the
projection magnitude. Let

$$
p_n = \frac{1}{|\mathcal{B}_{\text{coll}}|}\sum_t \|u^{\text{safe}}_t - u^{\text{nom}}_t\|
$$

be the average projection magnitude over the latest collection batch. The two EMAs are

$$
\bar p_n = (1 - \beta_p)\, \bar p_{n-1} + \beta_p\, p_n,
\qquad
\sigma^{\text{tgt}}_n = \mathrm{clip}\!\big(\alpha_{\text{dom}}\cdot \bar p_n,\
\sigma_{\min},\ \sigma_{\max}\big),
$$

$$
\sigma_n = \mathrm{clip}\!\big((1 - \beta_\sigma)\, \sigma_{n-1} + \beta_\sigma\,
\sigma^{\text{tgt}}_n,\ \sigma_{\min},\ \sigma_{\max}\big),
$$

with $\beta_p = \texttt{schedules.sigma.beta\_proj}$,
$\beta_\sigma = \texttt{schedules.sigma.beta\_sigma}$,
$\alpha_{\text{dom}} = \texttt{schedules.sigma.alpha\_dom}$. The pinning of
$\sigma^{\text{tgt}}_n$ at $\sigma_{\max}$ for several cycles is a halt signal (§4.7).

**`sigma_pi`** — $\mathcal{D}_\pi$ noise decay (§4.2):

$$
\sigma_{\pi,n} = \sigma_{\pi,0}\cdot \max\!\big(0,\ 1 - n / D\big),
\qquad
\sigma_{\pi,0} = \texttt{schedules.sigma\_pi.init},\quad
D = \texttt{schedules.sigma\_pi.decay\_steps}.
$$

In v2.0.0 the default $\sigma_{\pi,0} = 0$ (no decay needed; on-policy collection is
already zero-noise). Late-training schedule additions are listed in §7.

### 4.7 Halt protocol

The trainer halts and writes a final checkpoint whenever any of the conditions below
triggers. Thresholds in `exp_config.halt`.

1. **NaN / Inf in $V$ or policy loss.** Hard halt at the offending step; save current
   state.
2. **$V_S$ gradient leak.** During every policy step, compute
   $\|\nabla_{V_S\text{params}} \mathcal{L}_\pi\|_2$; if this exceeds
   `halt.vs_grad_leak_threshold` ($10^{-9}$), halt. This enforces the detach convention
   of `02_control` §7.
3. **$\sigma$ pinned at $\sigma_{\max}$.** If $\sigma^{\text{tgt}}_n$ equals
   $\sigma_{\max}$ for `halt.sigma_max_cycles` consecutive collection cycles, halt:
   exploration is failing to find unsafe signal even at the noise ceiling.
4. **`cps` floor.** Once warmup is past, if the in-loop `cps` drops below
   `halt.cps_floor` (default $-0.5$), halt: training has collapsed.
5. **Early stop on no improvement.** If the best in-loop `cps` has not improved by
   `halt.early_stop_min_delta` within `halt.early_stop_patience` macro steps, halt.

Two additional detectors are listed as planned but **not** included in v2.0.0: a
deep-saturation detector (fraction of value-head outputs with $|h| > 0.95$ sustained
above a threshold) and a catastrophic-failure detector (per-eval `cps` slope below a
threshold). They are reserved as one-axis additions (§7) until v2.0.0 establishes the
baseline statistics needed to set their thresholds.

---

## 5. Algorithm — one macro step of v2.0.0 Joint Training

The following is the authoritative ordering. The OC-PNCBF loop is a subset (only the
collection and value blocks; no policy step; no warmup).

> **Inputs:** macro-step index $n$, value net $V_S$, target $V_S^{\text{target}}$, policy
> $\pi_\theta$, buffers $\mathcal{D}_V, \mathcal{D}_\pi$, configs.
>
> 1. $n_{\text{sched}} \leftarrow \max(0,\ n - \texttt{vs\_warmup\_steps})$.
> 2. $\gamma_{\text{disc},n} \leftarrow \texttt{schedules.gamma\_disc}(n_{\text{sched}})$,
>    $\tau_{\text{rhs},n} \leftarrow \texttt{schedules.target\_rhs}(n_{\text{sched}})$.
> 3. **Collection** (if $n = 1$ or $n \bmod \texttt{collect\_every} = 0$):
>    1. Draw scene batch with `episodes_per_collect` scenes (§1.1–§1.3).
>    2. Roll out $\pi_\theta + \mathcal{N}(0, \sigma_n^2 I) +$ HardNet on
>       $V_S^{\text{target}}$; append to $\mathcal{D}_V$ (trajectory + transition views).
>    3. Roll out $\pi_\theta + \mathcal{N}(0, \sigma_{\pi,n}^2 I) +$ HardNet on
>       $V_S^{\text{target}}$; append to $\mathcal{D}_\pi$.
>    4. Update $\bar p_n, \sigma^{\text{tgt}}_n, \sigma_n$ (§4.6); update the sigma-pinning
>       counter for halt (§4.7-3).
> 4. **Value updates.** For $k = 1 \ldots K_V$:
>    1. Sample $\texttt{v\_label\_episodes}$ trajectories from $\mathcal{D}_V$; rebuild
>       labels $\mathcal{D}_A, \mathcal{D}_C$ (§4.3).
>    2. Build minibatch:
>       $\mathcal{B}_{\text{trans}}\sim\mathcal{D}_V^{\text{trans}}$,
>       $\mathcal{B}_A\sim\mathcal{D}_A$, $\mathcal{B}_C\sim\mathcal{D}_C$
>       at sizes specified in §4.3.
>    3. Compute the value target $y$ (§2.1; RPCBF override if active).
>    4. Compute $\mathcal{L}_V = \lambda_R \mathcal{L}_R + \lambda_A \mathcal{L}_A
>       + \lambda_C \mathcal{L}_C$; finite-check; halt on NaN/Inf (§4.7-1).
>    5. Zero V grads; backward; clip $\|\nabla_{\theta_V}\|_2 \leq
>       \texttt{grad\_clip}$; `opt_vs.step()`.
>    6. Polyak update $V_S^{\text{target}} \leftarrow (1-\tau_P)V_S^{\text{target}}
>       + \tau_P V_S$ (§2.2, per-V-step convention).
> 5. **Policy updates** (skip if $n \leq \texttt{vs\_warmup\_steps}$). For
>    $k = 1 \ldots K_\pi$:
>    1. Sample $x_0$, goal, obstacles from $\mathcal{D}_\pi^{\text{trans}}$.
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
>    rows to `eval_episodes.csv`, log to TensorBoard, and update `best.pt` by `cps`
>    (`exp_config.halt.early_stop_min_delta` threshold).
> 8. **Halt checks** (cps floor §4.7-4, early-stop patience §4.7-5).
> 9. **Final.** When $n = n_{\text{steps}}$ or any halt triggers, save `final.pt`,
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

## 7. One-axis additions (planned, not in v2.0.0)

Mechanisms reserved for later versions, each to be introduced individually with a stated
hypothesis and a clean ablation against the baseline:

- **Value-target blend** — $\rho V^{\text{rp}} + (1-\rho)\tilde A$ instead of the binary
  v2.0.0 choice between PNCBF and RPCBF.
- **Un-normalized RPCBF target** — the strict $V \ge h$ log-sum-exp form
  $\frac{1}{\beta}\log\sum_k \exp(\beta h_{t+k})$ instead of the v2.0.0 normalized
  log-mean-exp (§2.1.2), recovering the analytic CBF upper-bound at the cost of an
  $H$-dependent bias. To try if the $\mathcal{D}_A$ anchor proves insufficient to keep
  $V_S$ conservative on the unsafe region.
- **Auxiliary value-loss terms** — $\varepsilon$-gradient floor, consistency loss,
  Lipschitz loss.
- **Additional anchors** — $\mathcal{D}_{\text{far}}$, $\mathcal{D}_{\text{brake}}$.
- **Late-floor sigma** on $\mathcal{D}_V$ collection noise.
- **Lateral/collinear bonus** in the policy task return, and collinear scene sampling.
- **Policy action smoothness curriculum** (smoothness weight ramped over training).
- **Unbounded value head** with cost-domain target clamping (alternative to the tanh head
  of `02_control` §3.2).
- **Policy architecture variants** — residual head, attention encoder.
- **Saturation and catastrophic-failure halt detectors** (§4.7 remarks).
- **Per-step action rate limiting.**

Each introduction follows the one-axis recommendation of the constitution.

---

## 8. Config references

All values are in `src/configs/`:

- `base_config.yaml` — task (env, obstacle, scene_train), filter constants
  (HardNet $\varepsilon$, CBF-QP parameters, $\gamma_{\text{margin}}$), network structure
  (value, control), LQR weights, eval pools and scene. Locked.
- `exp_config.yaml` — `run` identity; `training.{jt|oc_pncbf}`; `optim`; `value_target`;
  `schedules`; `collection`; `loss.{value|policy}`; `labeling`; `halt`;
  `filter.alpha_{safe|unsafe}`; `eval.cadence`.

This document does not duplicate values; conflicts are resolved in favor of the configs,
which are written into every run's `data/<run_id>/config.yaml` (per `05_code` §3).

---

## 9. In-loop and final evaluation

### In-loop evaluation (during training)

Every `eval.cadence` macro steps (`exp_config.yaml`) the trainer runs an evaluation pass
on the **pinned in-loop pool** (`base_config.eval.in_loop`, $N = 200$, seed 12345) and
appends one row to `eval_metrics.csv`, the per-episode breakdown to `eval_episodes.csv`,
and the scalar values to TensorBoard. The **best-by-`cps`** checkpoint is saved to
`data/<run_id>/checkpoints/best.pt`. Mode details, output schema, and trajectory plots
live in `04_eval` §2.1, §3, §7.

### Final evaluation (after training)

When training terminates (by completion or halt), the trainer triggers a single **full
evaluation** on `data/<run_id>/checkpoints/best.pt` (override to `--last` for the last
checkpoint), using the disjoint full pool (`base_config.eval.full`, $N = 500$,
seed 23456), with all three trajectory variants (LQR-only / learned no-CBF / filtered),
trajectory PNGs, online insertion, and bootstrap statistics. Mode details, output layout,
and reporting conventions live entirely in `04_eval` §2.2, §3, §4, §5, §7.
