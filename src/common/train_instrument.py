"""v2.8.2 — training-loop instrumentation (prerequisite, not an axis; changes.md v2.8.2 §4).

Per-phase wall-clock and peak-VRAM helpers shared by the OC and JT trainers. v2.8.1's close could
recover neither the phase split nor peak VRAM from what was written, so the batch-sizing decision
(changes.md §5 S0) had no measurement to stand on. These are additive: they only widen the metrics
row, never change training math.

Usage (each trainer holds a `PhaseTimers` next to `start_time`):
    timers = PhaseTimers()
    _t = time.time(); collect(...);            timers.t_collect += time.time() - _t
    _t = time.time(); step(...);               timers.t_bptt    += time.time() - _t
    ...
    row = _metrics_row(..., **timers.as_row())   # adds t_collect/t_bptt/t_eval/t_ckpt/cuda_max_mem_mb
    if step % metrics_log_every == 0:
        _append_csv(...); timers.reset()          # per-interval split (accumulate, reset after each write)
"""
from __future__ import annotations

import torch

# Column names contributed to METRIC_COLUMNS by this instrumentation (kept in one place so both
# trainers stay in sync). Wall-clock in SECONDS spent in each phase since the last logged row;
# cuda_max_mem_mb is the process peak allocated (MiB-scale, /1e6) at the time the row is written.
INSTRUMENT_COLUMNS = ["t_collect", "t_bptt", "t_eval", "t_eval_wait", "t_ckpt", "cuda_max_mem_mb"]


def cuda_max_mem_mb() -> float:
    """Peak CUDA bytes allocated so far / 1e6; 0.0 when CUDA is unavailable (CPU / smoke)."""
    return float(torch.cuda.max_memory_allocated()) / 1e6 if torch.cuda.is_available() else 0.0


class PhaseTimers:
    """Four per-phase second accumulators + the peak-VRAM read, reset after each logged row so the
    written value is the split SINCE the previous log (collect/eval/ckpt run on coarser cadences than
    metrics_log_every, so a non-resetting counter would report cost against the wrong interval)."""

    __slots__ = ("t_collect", "t_bptt", "t_eval", "t_eval_wait", "t_ckpt")

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.t_collect = 0.0
        self.t_bptt = 0.0
        self.t_eval = 0.0
        self.t_eval_wait = 0.0        # v2.8.2: cross-process eval-gate queueing (NEVER billed to training)
        self.t_ckpt = 0.0

    def as_row(self) -> dict[str, float]:
        return {
            "t_collect": float(self.t_collect),
            "t_bptt": float(self.t_bptt),
            "t_eval": float(self.t_eval),
            "t_eval_wait": float(self.t_eval_wait),
            "t_ckpt": float(self.t_ckpt),
            "cuda_max_mem_mb": cuda_max_mem_mb(),
        }
