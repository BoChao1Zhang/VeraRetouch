"""Overlap Where-B's data supply with its GPU work.  Order-preserving (PERF-1).

The training loop was

    for micro in sampler:
        samples = [self.dataset[i] for i, _ in micro]   # <- blocking NFS + decode
        batch = self.builder.build(samples, modes)      # <- the GPU work

with no dataloader, no workers and no prefetch anywhere: every record read,
image read, mask read, JPEG decode and PNG decode happened on the training
thread while both H100s sat idle.  Measured (nfs-ro, 12 micro-batches of 8 taken
from the untouched tail of the sampler order, W01/W02 streaming concurrently):

    627 ms per micro-batch = 78 ms per sample
      read_image     32.0%     read_oracle 13.1%     read_mask   12.7%
      read_genctx    11.5%     read_record  8.5%     -> 77.8% pure NFS latency
      prepare_image  10.9%     + 10 % small CPU tensor ops

At grad-accum 4 that is 2.51 s of the observed 4.4 s optimizer step, which is
why ``nvidia-smi`` showed 17-23% utilisation on both cards.

What this module changes: **when** ``dataset[i]`` runs, never **what** it
returns.  Micro-batches are still consumed in exactly the sampler's order, each
sample is still built by the same ``WhereBDataset.__getitem__``, and an
exception still reaches the training loop -- from the micro-batch it belongs to,
in order.  ``WhereBDataset.__getitem__`` is a pure function of the frozen index
(no RNG, no shared mutable state beyond the fd caches, which are per-thread
since PERF-1), so moving it onto worker threads cannot change a single trained
weight.  Python threads (not processes) are the right tool here because every
expensive part of the work -- pread, PIL decode, torch resize -- releases the
GIL, and processes would have to pickle a PIL image and a 512x768 mask per
sample back to the parent.

Set ``Q3VL_PREFETCH_WORKERS=0`` to restore the exactly-serial behaviour.
"""

from __future__ import annotations

import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, Iterator, Sequence

__all__ = ["PREFETCH_ENV", "DEFAULT_PREFETCH_WORKERS", "default_workers",
           "SamplePrefetcher", "prefetch_warmers"]

PREFETCH_ENV = "Q3VL_PREFETCH_WORKERS"
#: 6 in-flight micro-batches hide 627 ms of supply behind a ~475 ms GPU step
#: with headroom, and cost ~150 MB of queued samples at micro-batch 8.
DEFAULT_PREFETCH_WORKERS = 6

Micro = Sequence[tuple[int, str]]


def default_workers() -> int:
    raw = os.environ.get(PREFETCH_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_PREFETCH_WORKERS
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_PREFETCH_WORKERS


def prefetch_warmers(builder) -> Callable[[Any, str], None] | None:
    """A warmer that pays for the published-store reads ``builder`` will make.

    ``BatchBuilder`` reads the generated ``<where>`` record inside
    ``context_for`` and the oracle latent inside ``targets_for``; together they
    were 24.6% of the supply cost and neither is reachable from
    ``dataset[i]``.  Reading them on the prefetch thread puts the bytes in the
    store's blob cache (and in the page cache) before the training thread asks.

    Returns ``None`` when there is nothing to warm, so the caller can skip the
    call entirely.  Warming is an optimisation and is therefore duck-typed on
    ``prime``: a builder wired to a stand-in store (the mock end-to-end tests, an
    offline analysis) simply gets no warmer instead of an AttributeError, and
    the samples it produces are byte-identical either way.
    """
    from .context import GENERATED
    from .stores import GenContextStore, OracleStore

    prime_gen = getattr(getattr(builder, "genctx", None), "prime", None)
    prime_orc = getattr(getattr(builder, "oracle", None), "prime", None)
    if prime_gen is None and prime_orc is None:
        return None

    def warm(sample, mode: str) -> None:
        sid = sample.sample_id
        if prime_gen is not None and mode == GENERATED:
            prime_gen(sid, GenContextStore.SUFFIX)
        if prime_orc is not None and not sample.is_global:
            prime_orc(sid, OracleStore.SUFFIX)

    return warm


class SamplePrefetcher:
    """Build the next few micro-batches' samples on worker threads.

    ``batches`` is any iterable of micro-batches -- the trainer passes the
    :class:`~q3vl.whereb.context.BalancedContextSampler` straight through, and
    evaluation passes its own chunks -- and iteration yields
    ``(micro, samples)`` in that same order.

    ``workers <= 0`` runs the original serial loop, so the fallback path is one
    ``if`` rather than a second implementation.
    """

    def __init__(
        self,
        dataset,
        batches: Iterable[Micro],
        *,
        workers: int | None = None,
        depth: int | None = None,
        warm: Callable[[Any, str], None] | None = None,
    ):
        self.dataset = dataset
        self.batches = batches
        self.workers = default_workers() if workers is None else int(workers)
        # one in-flight micro-batch per worker: fewer starves the pool, more only
        # buys queue depth that is already covered by the GPU step being longer
        # than the supply step.
        self.depth = max(1, int(depth)) if depth else max(1, self.workers)
        self.warm = warm

    # -- one micro-batch ----------------------------------------------------
    def _build(self, micro: Micro) -> list[Any]:
        out = []
        for i, mode in micro:
            sample = self.dataset[i]
            if self.warm is not None:
                self.warm(sample, mode)
            out.append(sample)
        return out

    def facts(self) -> dict[str, Any]:
        return {"workers": self.workers, "depth": self.depth,
                "warm": self.warm is not None, "env": PREFETCH_ENV}

    def __iter__(self) -> Iterator[tuple[Micro, list[Any]]]:
        if self.workers <= 0:
            for micro in self.batches:
                yield micro, self._build(micro)
            return

        it = iter(self.batches)
        pending: deque = deque()
        pool = ThreadPoolExecutor(self.workers, thread_name_prefix="q3vl-prefetch")

        def submit_next() -> bool:
            try:
                micro = next(it)
            except StopIteration:
                return False
            pending.append((micro, pool.submit(self._build, micro)))
            return True

        try:
            for _ in range(self.depth):
                if not submit_next():
                    break
            while pending:
                micro, fut = pending.popleft()
                # .result() re-raises in the training thread, at the micro-batch
                # the failure belongs to -- same visible failure as the serial
                # loop, just with some later batches already built.
                samples = fut.result()
                submit_next()
                yield micro, samples
        finally:
            for _micro, fut in pending:
                fut.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
