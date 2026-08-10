"""The sample prefetcher is scheduling, not semantics (PERF-1).

Everything here is an equivalence test: whatever the worker count, the training
loop must see the same micro-batches, in the same order, carrying the same
samples, and must observe a failure at the same micro-batch it would have
serially.  If any of these break, the arm's weights are no longer a function of
the seed alone and every Where-B number becomes irreproducible.
"""

from __future__ import annotations

import threading

import pytest

from q3vl.whereb.context import GENERATED, GT
from q3vl.whereb.prefetch import (
    DEFAULT_PREFETCH_WORKERS,
    PREFETCH_ENV,
    SamplePrefetcher,
    default_workers,
    prefetch_warmers,
)


class _Sample:
    def __init__(self, i: int, is_global: bool = False):
        self.sample_id = f"s{i:04d}"
        self.index = i
        self.is_global = is_global


class _Dataset:
    """Counts accesses and records the thread each one ran on."""

    def __init__(self, n: int = 64, boom: int | None = None):
        self.n = n
        self.boom = boom
        self.seen: list[int] = []
        self.threads: set[str] = set()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> _Sample:
        if i == self.boom:
            raise ValueError(f"sample {i} is broken")
        with self._lock:
            self.seen.append(i)
            self.threads.add(threading.current_thread().name)
        return _Sample(i, is_global=(i % 3 == 0))


def _batches(n_batches: int = 8, size: int = 4):
    out = []
    k = 0
    for _ in range(n_batches):
        micro = []
        for j in range(size):
            micro.append((k, GT if j % 2 == 0 else GENERATED))
            k += 1
        out.append(micro)
    return out


@pytest.mark.parametrize("workers", [0, 1, 4, 8])
def test_order_and_content_are_independent_of_the_worker_count(workers):
    batches = _batches()
    serial = [[s.index for s in b] for _m, b in
              SamplePrefetcher(_Dataset(), batches, workers=0)]
    got = [[s.index for s in b] for _m, b in
           SamplePrefetcher(_Dataset(), batches, workers=workers)]
    assert got == serial
    assert serial == [[i for i, _m in micro] for micro in batches]


@pytest.mark.parametrize("workers", [0, 4])
def test_the_micro_batch_is_handed_back_unchanged(workers):
    batches = _batches()
    for micro, samples in SamplePrefetcher(_Dataset(), batches, workers=workers):
        assert [i for i, _m in micro] == [s.index for s in samples]


def test_workers_actually_run_off_the_main_thread():
    ds = _Dataset()
    list(SamplePrefetcher(ds, _batches(), workers=4))
    assert any(name.startswith("q3vl-prefetch") for name in ds.threads)


def test_zero_workers_stays_on_the_main_thread():
    ds = _Dataset()
    list(SamplePrefetcher(ds, _batches(), workers=0))
    assert ds.threads == {threading.current_thread().name}


@pytest.mark.parametrize("workers", [0, 4])
def test_a_broken_sample_raises_at_its_own_micro_batch(workers):
    """Micro-batch 3 of 4-sample batches owns index 13."""
    ds = _Dataset(boom=13)
    seen = []
    with pytest.raises(ValueError, match="sample 13"):
        for micro, _samples in SamplePrefetcher(ds, _batches(), workers=workers):
            seen.append(micro[0][0])
    assert seen == [0, 4, 8]              # the three batches before it, in order


def test_the_pool_is_shut_down_when_the_consumer_stops_early():
    ds = _Dataset(n=4096)
    before = threading.active_count()
    it = iter(SamplePrefetcher(ds, _batches(64), workers=4))
    next(it)
    it.close()
    for _ in range(200):                                        # threads exit lazily
        if threading.active_count() <= before + 4:
            break
        threading.Event().wait(0.01)
    assert threading.active_count() <= before + 4


def test_lazy_batches_are_not_drained_up_front():
    """Evaluation passes a generator; the prefetcher must not materialise it."""
    pulled = []

    def gen():
        for micro in _batches(32):
            pulled.append(micro[0][0])
            yield micro

    it = iter(SamplePrefetcher(_Dataset(n=4096), gen(), workers=2, depth=2))
    next(it)
    assert len(pulled) <= 4               # depth + the one just handed out


# -- warmers ---------------------------------------------------------------

class _Store:
    SUFFIX = ".x"

    def __init__(self):
        self.primed: list[str] = []
        self._lock = threading.Lock()

    def prime(self, sample_id: str, suffix: str) -> bool:
        with self._lock:
            self.primed.append(sample_id)
        return True


class _Builder:
    def __init__(self, genctx=None, oracle=None):
        self.genctx = genctx
        self.oracle = oracle


def test_warmers_prime_only_what_the_builder_will_read():
    gen, orc = _Store(), _Store()
    warm = prefetch_warmers(_Builder(gen, orc))
    warm(_Sample(1, is_global=False), GT)
    warm(_Sample(2, is_global=False), GENERATED)
    warm(_Sample(3, is_global=True), GENERATED)
    assert gen.primed == ["s0002", "s0003"]      # generated context only
    assert orc.primed == ["s0001", "s0002"]      # local samples only


def test_a_builder_without_stores_gets_no_warmer():
    assert prefetch_warmers(_Builder()) is None


def test_a_store_without_prime_is_skipped_rather_than_crashing():
    """Stand-in stores in the mock end-to-end tests have no prime()."""
    class _Bare:
        pass

    warm = prefetch_warmers(_Builder(genctx=_Bare(), oracle=_Store()))
    warm(_Sample(1), GENERATED)                  # must not raise


def test_the_warmer_runs_on_the_prefetch_thread():
    gen = _Store()
    ds = _Dataset()
    names: set[str] = set()

    def warm(sample, mode):
        names.add(threading.current_thread().name)
        gen.prime(sample.sample_id, ".x")

    list(SamplePrefetcher(ds, _batches(), workers=4, warm=warm))
    assert any(n.startswith("q3vl-prefetch") for n in names)
    assert len(gen.primed) == 32


# -- the knob --------------------------------------------------------------

def test_the_env_variable_selects_the_worker_count(monkeypatch):
    monkeypatch.delenv(PREFETCH_ENV, raising=False)
    assert default_workers() == DEFAULT_PREFETCH_WORKERS
    monkeypatch.setenv(PREFETCH_ENV, "0")
    assert default_workers() == 0
    monkeypatch.setenv(PREFETCH_ENV, "3")
    assert default_workers() == 3
    monkeypatch.setenv(PREFETCH_ENV, "nonsense")
    assert default_workers() == DEFAULT_PREFETCH_WORKERS
