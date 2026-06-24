"""WS-C Phase 1: a one-thread render queue so the GPU teacher never idles.

The build main loop was synchronous: plan -> GPU render (1-2s, blocking) ->
submit clean -> repeat. The GPU sat idle during each plan/prepare/submit window.
RenderWorker moves render onto its own thread fed by a bounded queue: the
producer (main loop) enqueues a render thunk and gets a Future back immediately,
then plans/prepares the next batch while the worker renders. maxsize gives
natural backpressure (the prefetch depth) and caps RAM.

Only render is decoupled this way — it is the sole GPU-single-tenant hot-path
bottleneck. vLLM already runs out-of-process behind the broker, SAM3 is offline
precomputed, IQA is QA-only. ponytail: no task queue for the cold paths, and a
plain thunk queue (not a broker) since the worker is in-process and can't vanish.
"""
from __future__ import annotations

import queue
import threading
from concurrent.futures import Future
from typing import Any, Callable, Optional

_SENTINEL = object()


class RenderWorker:
    """Single daemon thread running submitted render thunks in FIFO order.

    One worker only: the GPU teacher is serial (render_lock), so extra workers
    would just contend. The thunk is whatever the caller passes (kept as the
    existing ``stream._render_after_batch`` call, so the render path — cap,
    downscale, LUT-None handling — is byte-identical to the inline path). If the
    worker thread dies (e.g. CUDA OOM), the next submit() raises so the producer
    can fail the process for an outer restart + --resume.
    """

    def __init__(self, prefetch_depth: int = 3) -> None:
        self._q: "queue.Queue[Any]" = queue.Queue(maxsize=max(1, int(prefetch_depth)))
        self._err: Optional[BaseException] = None
        self._t = threading.Thread(target=self._run, name="render-worker", daemon=True)
        self._t.start()

    def _run(self) -> None:
        while True:
            job = self._q.get()
            if job is _SENTINEL:
                return
            fn, fut = job
            if not fut.set_running_or_notify_cancel():
                continue
            try:
                fut.set_result(fn())
            except BaseException as e:  # noqa: BLE001 - keep worker alive, surface to caller
                self._err = e
                fut.set_exception(e)

    def submit(self, fn: Callable[[], Any]) -> "Future":
        """Enqueue a render thunk; returns immediately with a Future. Blocks only
        when the queue is full (== prefetch depth reached) — that is the
        backpressure that keeps the GPU at most `prefetch_depth` batches ahead."""
        if not self._t.is_alive():
            raise RuntimeError(f"render worker is dead: {self._err!r}")
        fut: "Future[Any]" = Future()
        self._q.put((fn, fut))
        return fut

    def alive(self) -> bool:
        return self._t.is_alive()

    def close(self) -> None:
        try:
            self._q.put_nowait(_SENTINEL)
        except queue.Full:
            self._q.put(_SENTINEL)
        self._t.join(timeout=5.0)


if __name__ == "__main__":
    # self-check: futures resolve in FIFO order, errors propagate, worker survives.
    import time

    def mk(i):
        def fn():
            time.sleep(0.02)
            return i * 10
        return fn

    w = RenderWorker(prefetch_depth=2)
    futs = [w.submit(mk(i)) for i in range(5)]
    assert [f.result() for f in futs] == [0, 10, 20, 30, 40]

    def boom():
        raise ValueError("kaboom")

    bad = w.submit(boom)
    try:
        bad.result()
        raise AssertionError("expected error")
    except ValueError as e:
        assert "kaboom" in str(e)
    assert w.alive()  # worker survives a thunk failure
    w.close()
    print("ok")
