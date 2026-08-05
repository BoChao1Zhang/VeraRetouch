"""A resident CPU process pool for the inner oracle fits.

Why this exists
---------------
D11 established that the per-image L-BFGS belongs on the CPU (float64 over a
(1536, 71) design with a strong-Wolfe line search is thousands of tiny kernels;
an H100 runs it 3.5x slower than a CPU core).  But one core is not enough
either: at 1.6-2.9 s per fit and 64 fits per step (batch 32 x 2 readouts), a
serial calibration step costs ~100 s, i.e. ~37 days per arm.

The fits inside a step are *embarrassingly* parallel -- each one sees a single
image's ``phi`` and its own mask, and shares nothing -- so they go to a resident
pool of single-threaded workers.  Resident, because a 48-core pool costs a
second or two to build and a calibration epoch has ~1,300 steps.

Numerics
--------
Thread count changes results: BLAS reduction order is not associative, and the
same fit measured 1-thread vs 8-thread differs by ~3e-8 in ``w_raw``.  So
**one thread per fit is the canonical setting**, in the pool and in the serial
path alike.  That is what makes "same seed => same numbers" true regardless of
how many workers happen to be configured, and it is what
``test_fitpool.py::test_pool_matches_serial_bitwise`` pins.

Worker count is chosen by the caller, not by ``os.cpu_count()``: this box also
has to run the S1 packing job and the genctx generation, and a pool that eats
every core would simply move the bottleneck.

Start method
------------
``spawn``, not ``fork``.  The parent has already initialised OpenBLAS (numpy and
torch both do it on import), and a forked child inherits a thread pool whose
threads do not exist.  Measured: under ``fork`` **every one of the 10 starts of
every fit raised**, so each fit came back ``all_starts_failed`` -- i.e. the pool
looked like it worked and quietly returned garbage.  Under ``spawn`` the same fit
reproduces the parent's loss to the last bit.  ``spawn`` costs a couple of
seconds per worker once, against a ~1,300-step epoch.

:meth:`FitPool.self_check` re-runs one real fit in a worker and compares it to
the parent before any calibration starts, so this class of breakage can never
again be mistaken for "these samples were hard to fit".
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch

from .config import FitConfig
from .oracle import FitResult, fit_latent

__all__ = ["FitTask", "FitPool", "run_fits_serial", "pin_single_thread"]

# every library that might spin up its own thread pool inside a worker
_THREAD_ENV = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)


def _pin_openblas() -> bool:
    """Set OpenBLAS to one thread *at runtime*, in an already-running process.

    ``torch.set_num_threads`` governs ATen only; numpy keeps its own OpenBLAS,
    and the ``*_NUM_THREADS`` env vars are read when that library loads -- which
    has already happened by the time any of this code runs.  So the informed
    L-BFGS start (``np.linalg.lstsq``) stayed multi-threaded in the parent while
    a spawned worker got a genuinely single-threaded one, and the two disagreed
    in the 13th digit.  Calling the library's own setter closes that gap.
    """
    import ctypes

    try:
        with open(f"/proc/{os.getpid()}/maps", encoding="utf-8") as fh:
            paths = {tok for line in fh for tok in line.split()
                     if "openblas" in tok.lower() and ".so" in tok}
    except OSError:
        return False
    for path in sorted(paths):
        try:
            lib = ctypes.CDLL(path)
        except OSError:
            continue
        for name in ("openblas_set_num_threads64_", "openblas_set_num_threads",
                     "goto_set_num_threads"):
            fn = getattr(lib, name, None)
            if fn is not None:
                fn(ctypes.c_int(1))
                return True
    return False


def pin_single_thread() -> bool:
    """Force single-threaded numerics in this process; returns True if OpenBLAS
    was reachable.

    One thread is the *canonical* setting for the oracle fit, not an
    optimisation: BLAS reduction order is not associative, so a fit run with 8
    threads differs from the same fit run with 1 by ~3e-8 in ``w_raw``.  Pinning
    it is what makes "same seed => same numbers" hold regardless of how many
    workers a machine happens to be able to spare.
    """
    for var in _THREAD_ENV:
        os.environ[var] = "1"
    torch.set_num_threads(1)
    return _pin_openblas()


def _worker_init() -> None:
    pin_single_thread()


@dataclass
class FitTask:
    """One image's fits.  Both readouts travel together so the (1536, 71)
    ``phi`` is pickled once per sample rather than once per readout."""

    key: str                       # sample_id
    phi: torch.Tensor              # (P, 71) float64, detached, on CPU
    target: torch.Tensor           # (P,) float64, on CPU
    readouts: tuple[str, ...]
    cfg: FitConfig
    seed_offset: int = 0


def _run_task(task: FitTask) -> tuple[str, dict[str, FitResult]]:
    out: dict[str, FitResult] = {}
    for readout in task.readouts:
        cfg = type(task.cfg)(**{**task.cfg.__dict__,
                                "seed": task.cfg.seed + task.seed_offset})
        out[readout] = fit_latent(task.phi, task.target, readout, cfg)
    return task.key, out


def run_fits_serial(tasks: Iterable[FitTask]) -> dict[str, dict[str, FitResult]]:
    """Reference implementation -- identical numbers, no processes."""
    return {k: v for k, v in (_run_task(t) for t in tasks)}


def _probe_threads(_: int) -> tuple[int, str | None]:
    return torch.get_num_threads(), os.environ.get("OMP_NUM_THREADS")


class FitPool:
    """Resident single-threaded process pool.  ``None`` workers means serial."""

    def __init__(self, n_workers: int | None = None, mp_context: str = "spawn"):
        self.n_workers = n_workers
        self.mp_context = mp_context
        self._ex: ProcessPoolExecutor | None = None
        if n_workers:
            pin_single_thread()
            import multiprocessing as mp

            self._ex = ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=mp.get_context(mp_context),
                initializer=_worker_init,
            )

    def __enter__(self) -> "FitPool":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def run(self, tasks: Sequence[FitTask]) -> dict[str, dict[str, FitResult]]:
        """``{sample_id: {readout: FitResult}}`` for a whole batch."""
        if self._ex is None:
            return run_fits_serial(tasks)
        return {k: v for k, v in self._ex.map(_run_task, tasks)}

    def warmup(self) -> None:
        """Start the workers now rather than inside the first timed step."""
        if self._ex is not None:
            list(self._ex.map(_probe_threads, range(self.n_workers or 0)))

    def self_check(self, task: FitTask) -> dict[str, Any]:
        """Run one real fit in a worker and in the parent, and compare.

        This is the guard that ``fork`` needed: a broken worker environment makes
        every start raise, which the fit reports as ``all_starts_failed`` -- a
        perfectly ordinary-looking rejection.  Without this check a whole
        calibration epoch can come back "no sample could be fitted" and look like
        a data problem.  Raises rather than returning a flag: there is no sane
        way to continue.
        """
        if self._ex is None:
            return {"mode": "serial", "checked": False}
        parent = _run_task(task)[1]
        worker = self._ex.submit(_run_task, task).result()[1]
        for readout, ref in parent.items():
            got = worker[readout]
            if got.status != ref.status or got.loss != ref.loss:
                raise RuntimeError(
                    f"fit pool self-check failed for {readout}: worker returned "
                    f"status={got.status!r} loss={got.loss!r} "
                    f"(n_failed_starts={got.n_failed_starts}/{got.n_starts}) while the "
                    f"parent returned status={ref.status!r} loss={ref.loss!r}. "
                    f"start_method={self.mp_context!r}. A worker whose BLAS is broken "
                    f"fails every start and reports it as 'all_starts_failed', which is "
                    f"indistinguishable from a hard sample -- do not run a calibration "
                    f"epoch against this pool."
                )
            if ref.latent is not None and not torch.equal(got.latent.w_raw, ref.latent.w_raw):
                raise RuntimeError(
                    f"fit pool self-check: {readout} latent differs from the serial "
                    f"reference by {float((got.latent.w_raw - ref.latent.w_raw).abs().max()):.3e}. "
                    f"The pool must change the wall clock and nothing else."
                )
        return {"mode": "process_pool", "checked": True,
                "start_method": self.mp_context,
                "losses": {r: f.loss for r, f in parent.items()}}

    def close(self) -> None:
        if self._ex is not None:
            self._ex.shutdown(wait=True)
            self._ex = None

    def facts(self) -> dict[str, Any]:
        return {
            "n_workers": self.n_workers or 1,
            "mode": "process_pool" if self._ex is not None else "serial",
            "start_method": self.mp_context,
            "threads_per_worker": 1,
            "torch_num_threads": torch.get_num_threads(),
        }
