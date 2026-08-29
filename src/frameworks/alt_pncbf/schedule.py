"""v2.9.3 ALT-PNCBF block schedule — the ONE thing this framework adds to the deployed macro-step loop.

Pure: no torch, no I/O, no RNG, no config loading. `block_at` is a total function of the macro-step index
and three integers, so the block active at ANY macro step of a finished run is recoverable exactly from
that run's persisted `config.yaml` alone, independent of what was logged.

INVARIANT 5 IS ENFORCED HERE. Block lengths are configuration IN MACRO STEPS, and `validate_against_loop`
refuses any schedule that would let a "block" degenerate into a single macro step carrying a large update
count: a block shorter than one collection period (`collection.jt.collect_every`) cannot draw fresh data
under the network it is held against, so it is not a policy-iteration block and is rejected before the
run directory is created.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

VALUE = "value"
POLICY = "policy"
BLOCK_KINDS = (VALUE, POLICY)

# metrics.csv encoding of the `block` column (numeric so TensorBoard and every CSV reader take it).
BLOCK_CODE = {VALUE: 0.0, POLICY: 1.0}


@dataclass(frozen=True)
class BlockSchedule:
    """A deterministic two-phase cycle over macro steps: `first` for its length, then the other."""

    value_block: int
    policy_block: int
    first: str = VALUE

    @property
    def cycle(self) -> int:
        return self.value_block + self.policy_block

    def block_at(self, step: int) -> str:
        """The block active at MACRO step `step` (1-based, as the trainer's loop counts)."""
        if step < 1:
            raise ValueError(f"macro steps are 1-based; got step={step}.")
        phase = (step - 1) % self.cycle
        if self.first == VALUE:
            return VALUE if phase < self.value_block else POLICY
        return POLICY if phase < self.policy_block else VALUE

    def index_at(self, step: int) -> int:
        """0-based index of the alternation CYCLE containing `step` (both blocks share one index)."""
        if step < 1:
            raise ValueError(f"macro steps are 1-based; got step={step}.")
        return (step - 1) // self.cycle

    def blocks(self, n_steps: int) -> list[tuple[int, int, str, int]]:
        """Every block of a run of `n_steps` macro steps as (start_step, end_step, kind, cycle_index)."""
        out: list[tuple[int, int, str, int]] = []
        start = 1
        while start <= n_steps:
            kind = self.block_at(start)
            end = start
            while end + 1 <= n_steps and self.block_at(end + 1) == kind:
                end += 1
            out.append((start, end, kind, self.index_at(start)))
            start = end + 1
        return out

    def as_config(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "value_block": int(self.value_block),
            "policy_block": int(self.policy_block),
            "first": str(self.first),
        }


def block_schedule_from_config(alt_cfg: Mapping[str, Any]) -> BlockSchedule:
    """Build the schedule from `training.alt`. NO key is defaulted: a block length is never invented."""
    if not alt_cfg:
        raise ValueError(
            "ALT-PNCBF requires a `training.alt` config subtree with keys "
            "{enabled, value_block, policy_block, first}; none was present. Block lengths are never "
            "defaulted — supply them through the launcher's load_effective_config redirect, or through "
            "run_training(value_block_override=..., policy_block_override=...)."
        )
    if not bool(alt_cfg.get("enabled", False)):
        raise ValueError(
            "training.alt.enabled is false. An ALT-PNCBF run with alternation disabled is a joint run "
            "with extra machinery, not a control; use src.frameworks.jt_pncbf.train for that."
        )
    for key in ("value_block", "policy_block"):
        if key not in alt_cfg:
            raise ValueError(f"training.alt.{key} is required and has no default.")
    first = str(alt_cfg.get("first", VALUE))
    if first not in BLOCK_KINDS:
        raise ValueError(f"training.alt.first must be one of {BLOCK_KINDS}, got {first!r}.")
    return BlockSchedule(
        value_block=int(alt_cfg["value_block"]),
        policy_block=int(alt_cfg["policy_block"]),
        first=first,
    )


def validate_against_loop(
    schedule: BlockSchedule,
    *,
    n_steps: int,
    vs_warmup_steps: int,
    collect_every: int,
    k_v: int,
    k_pi: int,
) -> None:
    """Refuse, before any run directory exists, every schedule that is not an honest alternation.

    Each rule below is a way the object could stop being policy iteration while still running.
    """
    if schedule.value_block < 1 or schedule.policy_block < 1:
        raise ValueError(
            f"both blocks must be at least one macro step long, got value_block="
            f"{schedule.value_block}, policy_block={schedule.policy_block}. A zero-length block is a "
            "single-phase run, not an alternation."
        )
    if k_v < 1 or k_pi < 1:
        raise ValueError(
            f"training.jt.K_V and training.jt.K_pi must both be >= 1 in an alternating run "
            f"(got K_V={k_v}, K_pi={k_pi}). The alternation is expressed by the BLOCK SCHEDULE, never "
            "by zeroing an update count: a zeroed count would make one block a no-op rather than a "
            "held-fixed counterpart."
        )
    period = max(1, int(collect_every))
    if schedule.value_block < period or schedule.policy_block < period:
        raise ValueError(
            f"INVARIANT 5: a block must span at least one collection period "
            f"(collection.jt.collect_every={period} macro steps), got value_block="
            f"{schedule.value_block}, policy_block={schedule.policy_block}. A shorter block cannot draw "
            "data under the network it is held against, so it reuses a buffer filled under an earlier "
            "policy and is not policy iteration. A block realized as ONE macro step with a large update "
            "count is refused here."
        )
    if n_steps < schedule.cycle:
        raise ValueError(
            f"training.jt.n_steps={n_steps} is shorter than one alternation cycle "
            f"({schedule.value_block}+{schedule.policy_block}={schedule.cycle} macro steps); the run "
            "would never reach its second block."
        )
    # The JT warmup gate (`step > vs_warmup_steps`) would silently turn an early policy block into a dead
    # macro step: no value update (it is a policy block) and no policy update (warmup). Refuse instead.
    first_policy_step = 1 if schedule.first == POLICY else schedule.value_block + 1
    if first_policy_step <= int(vs_warmup_steps):
        raise ValueError(
            f"the first policy macro step is {first_policy_step}, inside the value-only warmup "
            f"(vs_warmup_steps={vs_warmup_steps}); those macro steps would update neither network. "
            "Either set training.alt.first='value' with value_block >= vs_warmup_steps, or set "
            "vs_warmup_steps=0."
        )


# ----------------------------------------------------------------------------------------------------
# WHICH COLLECTION PASSES A BLOCK RUNS.  `training.alt.collect` and `training.alt.n_episodes_active_scale`.
#
# `collect_jt` (jt_pncbf/collection.py:207) runs TWO independent passes per collection: one into `D_V`
# (`buffers.value`, at `sigma_v`) and one into `D_pi` (`buffers.policy`, at `sigma_pi`), plus an optional
# precursor pass into `buffers.precursor`.  `D_V` is read only by `_value_updates`
# (jt_pncbf/train.py:1423,:1446) and `D_pi` only by `_policy_updates` (jt_pncbf/train.py:1657), so each
# pass feeds exactly one block's updates.  `collect: active_only` runs the ACTIVE block's pass alone.
#
# BOTH KEYS DEFAULT SO THAT A CONFIG NAMING NEITHER IS THE ALTBLK TRAINER, BYTE FOR BYTE:
# `both` takes the untouched `collect_jt` call and `1` leaves `collection.jt.episodes_per_collect` alone.
# `as_config` OMITS a key at its default, so a default run's persisted `config.yaml` is unchanged and an
# axis diff against it reads `<absent> -> active_only`, the idiom the launchers already use for
# `training.alt.enabled`.  Absence means the default, exactly as for `first` in `block_schedule_from_config`.
# ----------------------------------------------------------------------------------------------------

COLLECT_BOTH = "both"
COLLECT_ACTIVE_ONLY = "active_only"
COLLECT_MODES = (COLLECT_BOTH, COLLECT_ACTIVE_ONLY)

DEFAULT_N_EPISODES_ACTIVE_SCALE = 1

# The pass each block owns.  A pass is ACTIVE at a macro step iff the block that step runs owns it.
PASS_BUFFER = {VALUE: "D_V", POLICY: "D_pi"}


@dataclass(frozen=True)
class CollectPolicy:
    """Which of `collect_jt`'s two passes run at a collection, and with how many episodes each."""

    mode: str = COLLECT_BOTH
    n_episodes_active_scale: int = DEFAULT_N_EPISODES_ACTIVE_SCALE

    @property
    def engaged(self) -> bool:
        """True iff this is anything other than the ALTBLK default, i.e. iff behaviour can differ."""
        return (self.mode != COLLECT_BOTH
                or self.n_episodes_active_scale != DEFAULT_N_EPISODES_ACTIVE_SCALE)

    def runs_pass(self, pass_kind: str, block: str) -> bool:
        """Does the `pass_kind` pass run during a `block` macro step?"""
        if pass_kind not in BLOCK_KINDS:
            raise ValueError(f"pass_kind must be one of {BLOCK_KINDS}, got {pass_kind!r}.")
        if block not in BLOCK_KINDS:
            raise ValueError(f"block must be one of {BLOCK_KINDS}, got {block!r}.")
        if self.mode == COLLECT_BOTH:
            return True
        return pass_kind == block

    def n_episodes_for(self, pass_kind: str, block: str, base: int) -> int:
        """`n_episodes` for the `pass_kind` pass during a `block` macro step.

        The scale multiplies the ACTIVE pass and nothing else, so under `both` the inactive pass keeps
        the deployed `episodes_per_collect` and under `active_only` the inactive pass does not run at all.
        """
        if pass_kind not in BLOCK_KINDS:
            raise ValueError(f"pass_kind must be one of {BLOCK_KINDS}, got {pass_kind!r}.")
        if block not in BLOCK_KINDS:
            raise ValueError(f"block must be one of {BLOCK_KINDS}, got {block!r}.")
        if int(base) <= 0:
            raise ValueError(f"base n_episodes must be positive, got {base}.")
        if pass_kind == block:
            return int(base) * int(self.n_episodes_active_scale)
        return int(base)

    def as_config(self) -> dict[str, Any]:
        """Only NON-DEFAULT keys, so a default run's `training.alt` subtree is unchanged."""
        out: dict[str, Any] = {}
        if self.mode != COLLECT_BOTH:
            out["collect"] = str(self.mode)
        if self.n_episodes_active_scale != DEFAULT_N_EPISODES_ACTIVE_SCALE:
            out["n_episodes_active_scale"] = int(self.n_episodes_active_scale)
        return out


def collect_policy_from_config(alt_cfg: Mapping[str, Any] | None) -> CollectPolicy:
    """Build the collection policy from `training.alt`. BOTH keys default (unlike the block lengths).

    A block length is never invented because a wrong one silently changes the object under study; these
    two default because their defaults ARE the already-registered ALTBLK object, and defaulting them is
    what makes an ALTBLK config run byte-identically under this code.
    """
    cfg = dict(alt_cfg or {})
    mode = str(cfg.get("collect", COLLECT_BOTH))
    if mode not in COLLECT_MODES:
        raise ValueError(
            f"training.alt.collect must be one of {COLLECT_MODES}, got {mode!r}. "
            f"'{COLLECT_BOTH}' runs both of collect_jt's passes every collection (the ALTBLK "
            f"behaviour); '{COLLECT_ACTIVE_ONLY}' runs only the active block's own pass."
        )
    raw = cfg.get("n_episodes_active_scale", DEFAULT_N_EPISODES_ACTIVE_SCALE)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or int(raw) != raw:
        raise ValueError(
            f"training.alt.n_episodes_active_scale must be a positive integer, got {raw!r}."
        )
    scale = int(raw)
    if scale < 1:
        raise ValueError(
            f"training.alt.n_episodes_active_scale must be >= 1, got {scale}. A scale of 0 would make "
            "the active pass collect nothing, which is a dead block, not a smaller one."
        )
    return CollectPolicy(mode=mode, n_episodes_active_scale=scale)


def validate_collect_policy(
    policy: CollectPolicy,
    schedule: BlockSchedule,
    *,
    collect_every: int,
) -> None:
    """Refuse, before any run directory exists, a collection policy that cannot feed its own blocks.

    Under `active_only` INVARIANT 5 stops being only about drawing data under the held network and
    becomes about drawing data AT ALL: a block shorter than one collection period would run its updates
    against a buffer its own block never fed. `validate_against_loop` already refuses that; this repeats
    the check with the reason that applies here, so the refusal message names the right cause.
    """
    if policy.mode not in COLLECT_MODES:
        raise ValueError(f"training.alt.collect must be one of {COLLECT_MODES}, got {policy.mode!r}.")
    if policy.n_episodes_active_scale < 1:
        raise ValueError(
            f"training.alt.n_episodes_active_scale must be >= 1, got {policy.n_episodes_active_scale}."
        )
    if policy.mode != COLLECT_ACTIVE_ONLY:
        return
    period = max(1, int(collect_every))
    if schedule.value_block < period or schedule.policy_block < period:
        raise ValueError(
            f"training.alt.collect={COLLECT_ACTIVE_ONLY!r} feeds D_V only during value blocks and D_pi "
            f"only during policy blocks, so a block shorter than one collection period "
            f"(collection.jt.collect_every={period} macro steps) would run its updates against a buffer "
            f"its own block never filled; got value_block={schedule.value_block}, "
            f"policy_block={schedule.policy_block}."
        )
