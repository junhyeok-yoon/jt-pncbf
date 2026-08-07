"""v2.8.2 — cross-process in-loop-eval serialization gate.

An in-loop eval fire is ~50 s and saturates every CPU core; with several JT conditions training concurrently,
overlapping fires would contend and distort both the fires and the training steps around them. This gate lets at
most ONE eval run across all concurrent processes: an eval acquires an exclusive advisory lock (fcntl.flock) on a
SHARED lock file before running and releases it after; others block until it frees.

Robustness: flock is held by the open file DESCRIPTION and is auto-released by the OS when the holder process dies
(the fd is closed on exit) — so a dead process can NEVER hold a stale lock and the gate can NEVER deadlock (this
is stronger than a mtime/PID stale-timeout). A bounded ACQUIRE timeout is still enforced: if the lock cannot be
taken within timeout_s (pathological), the eval proceeds UNGATED and logs it, so a stuck holder can never stall
training forever.

Bit-identity: `enabled=False` bypasses the gate entirely; with a single run the lock is taken instantly, so the
wait is ~0 and the eval result is unchanged (the gate serializes timing only, never the computation). The context
manager yields the seconds spent WAITING so the caller bills it to a `t_eval_wait` phase — never to training.
"""
from __future__ import annotations

import errno
import fcntl
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator


@contextmanager
def eval_gate(
    lock_path: str | os.PathLike,
    *,
    enabled: bool = True,
    timeout_s: float | None = 300.0,
    poll_s: float = 0.5,
    log: Callable[[str], None] | None = None,
) -> Iterator[float]:
    """Hold an exclusive cross-process lock for the duration of the block. Yields seconds waited to acquire.

    timeout_s is None => UNBOUNDED wait (blocking flock): the block NEVER proceeds ungated. Use this for the
    end-of-run full eval (~13 GB): several runs finishing together must serialize their finals, and a bounded
    timeout that degrades to ungated is an OOM path there. A bounded timeout_s (the in-loop fire, ~17 MB) proceeds
    ungated on expiry so training can never stall forever. Either way, flock auto-releases on holder death — no
    deadlock in the unbounded path.
    """
    if not enabled:
        yield 0.0
        return
    lp = Path(lock_path)
    lp.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lp), os.O_CREAT | os.O_RDWR, 0o644)
    acquired = False
    waited = 0.0
    try:
        t0 = time.time()
        if timeout_s is None:
            while True:                                      # UNBOUNDED: block until acquired; never ungated
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX)           # blocking; OS releases it if the holder dies
                    acquired = True
                    waited = time.time() - t0
                    break
                except OSError as exc:
                    if exc.errno == errno.EINTR:
                        continue                             # interrupted by a signal — retry the blocking wait
                    raise
        else:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    waited = time.time() - t0
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EAGAIN, errno.EACCES):
                        raise
                    waited = time.time() - t0
                    if waited >= timeout_s:
                        if log is not None:
                            log(f"eval_gate: acquire timeout {timeout_s:.0f}s exceeded; proceeding UNGATED")
                        break                                # proceed ungated — never stall training forever
                    time.sleep(poll_s)
        if acquired:                                         # record holder pid (diagnostic only; flock is the lock)
            try:
                os.ftruncate(fd, 0)
                os.write(fd, f"{os.getpid()} {time.time():.3f}\n".encode())
            except OSError:
                pass
        yield waited
    finally:
        if acquired:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(fd)
