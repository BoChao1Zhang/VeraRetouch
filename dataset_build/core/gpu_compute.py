"""Per-GPU compute leases for the core layer.

The teacher renderer, pyiqa, and (live) SAM3 are all single-GPU, non-thread-safe
tenants that must not run concurrently on the *same* card. ``GpuCompute`` hands
out an exclusive per-device lease so those tenants serialize. The strict
acquisition order is **lease -> render_lock** (never the reverse) and the lease
is **non-reentrant** per device (a kind that already holds a card must not
re-lease it), matching the invariant in the v2 design — taking them out of order
or re-entering would deadlock.

In the Phase-2 build path only the renderer touches the GPU (the masker is a
``CachedMasker`` FS read and build never calls IQA), so a lease is single-tenant
and uncontended — ``RenderClient`` is constructed with ``gpu=None`` there and
falls back to ``render_lock`` alone, preserving today's exact behavior. The lease
matters when render and IQA share a card in the QA path.

Pure stdlib threading; no torch import here.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional


class GpuCompute:
    """Exclusive, non-reentrant per-device compute leases + least-busy pick."""

    def __init__(self, devices: Optional[Iterable[str]] = None) -> None:
        self._devices: List[str] = list(devices) if devices else ["cuda:0"]
        self._locks: Dict[str, threading.Lock] = {d: threading.Lock() for d in self._devices}
        # Waiters+holder count per device, for least-busy selection. Guarded by _meta.
        self._meta = threading.Lock()
        self._pending: Dict[str, int] = {d: 0 for d in self._devices}
        # Track which threads hold which device to catch reentrancy.
        self._holders: Dict[str, Optional[int]] = {d: None for d in self._devices}

    def _ensure(self, device: str) -> None:
        if device not in self._locks:
            self._locks[device] = threading.Lock()
            with self._meta:
                self._pending.setdefault(device, 0)
                self._holders.setdefault(device, None)

    @contextmanager
    def lease(self, device: str = "cuda:0"):
        """Acquire the exclusive compute lease for ``device``.

        Non-reentrant: re-leasing a device already held by the same thread is a
        programming error (would deadlock against the lease->render_lock order).
        """
        self._ensure(device)
        tid = threading.get_ident()
        if self._holders.get(device) == tid:
            raise RuntimeError(f"non-reentrant GPU lease re-acquired on {device}")
        with self._meta:
            self._pending[device] += 1
        try:
            self._locks[device].acquire()
            self._holders[device] = tid
            try:
                yield device
            finally:
                self._holders[device] = None
                self._locks[device].release()
        finally:
            with self._meta:
                self._pending[device] -= 1

    def least_busy(self, devices: Optional[Iterable[str]] = None) -> str:
        """Return the device with the fewest pending+holding leasers."""
        cands = list(devices) if devices else self._devices
        with self._meta:
            return min(cands, key=lambda d: self._pending.get(d, 0))
