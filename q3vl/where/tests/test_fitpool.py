"""The CPU fit pool must buy wall clock and nothing else.

The one property that matters: same seed => same numbers, pool or no pool.
A parallel implementation that quietly perturbs the fit would turn every arm
comparison into a comparison of scheduling noise.
"""

from __future__ import annotations

import time

import pytest
import torch

from q3vl.where.calibrate import Calibrator
from q3vl.where.config import CalibConfig, FitConfig, SEM_DIM
from q3vl.where.fitpool import FitPool, FitTask, pin_single_thread, run_fits_serial
from q3vl.where.phi import build_phi_dir
from q3vl.where.tests.test_calibrate import _mock_sample

DT = torch.float64


def _task(i: int, readouts=("band", "cband12"), gh=12, gw=16) -> FitTask:
    g = torch.Generator().manual_seed(100 + i)
    sem = torch.randn(gh * gw, SEM_DIM, generator=g, dtype=DT)
    img = torch.rand(3, gh, gw, generator=g, dtype=DT)
    phi = build_phi_dir(sem, img, gh, gw).phi_dir
    ys = torch.linspace(-1, 1, gh, dtype=DT)
    xs = torch.linspace(-1, 1, gw, dtype=DT)
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    target = torch.sigmoid((0.4 - ((X - 0.1 * i) ** 2 + Y ** 2)) * 8.0).reshape(-1)
    return FitTask(key=f"s{i}", phi=phi, target=target, readouts=readouts,
                   cfg=FitConfig(n_random=2, max_iter=40, seed=0), seed_offset=i)


def _fingerprint(fits) -> dict:
    out = {}
    for key, per in sorted(fits.items()):
        for r, f in sorted(per.items()):
            out[(key, r)] = (
                f.loss, f.status, f.reject_reason,
                None if f.latent is None else f.latent.w_raw.clone(),
                None if f.latent is None else f.latent.w0.clone(),
            )
    return out


def _assert_identical(a, b):
    assert set(a) == set(b)
    for k in a:
        la, sa, ra, wa, w0a = a[k]
        lb, sb, rb, wb, w0b = b[k]
        assert la == lb, f"{k}: loss {la!r} != {lb!r}"      # exact, not approx
        assert sa == sb and ra == rb, k
        if wa is None:
            assert wb is None
        else:
            assert torch.equal(wa, wb), f"{k}: w_raw differs by {float((wa-wb).abs().max()):.3e}"
            assert torch.equal(w0a, w0b), k


def test_pool_matches_serial_bitwise():
    """The headline guarantee: 4 workers, 6 samples, 12 fits -- every latent
    identical to the serial reference bit for bit (`torch.equal`, not allclose)."""
    pin_single_thread()
    tasks = [_task(i) for i in range(6)]
    serial = _fingerprint(run_fits_serial(tasks))
    with FitPool(n_workers=4) as pool:
        parallel = _fingerprint(pool.run(tasks))
    _assert_identical(serial, parallel)


def test_worker_count_does_not_change_the_numbers():
    """Because each worker is single-threaded, the result cannot depend on how
    many of them there are -- so a pool sized for the machine's other load is
    still the same experiment."""
    pin_single_thread()
    tasks = [_task(i) for i in range(4)]
    ref = _fingerprint(run_fits_serial(tasks))
    for n in (1, 2, 5):
        with FitPool(n_workers=n) as pool:
            _assert_identical(ref, _fingerprint(pool.run(tasks)))


def test_pool_none_is_the_serial_path():
    pin_single_thread()
    tasks = [_task(i) for i in range(3)]
    with FitPool(n_workers=None) as pool:
        assert pool.facts()["mode"] == "serial"
        _assert_identical(_fingerprint(run_fits_serial(tasks)),
                          _fingerprint(pool.run(tasks)))


def test_single_thread_is_the_canonical_setting():
    """BLAS reduction order is not associative: the same fit run with 1 vs 8
    threads differs by ~3e-8 in w_raw.  Pinning one thread per fit is what makes
    'same seed => same numbers' independent of the pool size, so the pin has to
    actually be in force."""
    pin_single_thread()
    assert torch.get_num_threads() == 1
    import os
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        assert os.environ[var] == "1"


def test_workers_are_single_threaded():
    from q3vl.where.fitpool import _probe_threads

    with FitPool(n_workers=2) as pool:
        got = list(pool._ex.map(_probe_threads, range(2)))
    assert all(n == 1 and env == "1" for n, env in got), got


def test_spawn_is_the_start_method():
    """`fork` is not usable here: the parent has already initialised OpenBLAS, and
    a forked child inherits a thread pool whose threads do not exist.  Measured:
    under fork every one of the 10 starts of every fit raised, so each fit came
    back `all_starts_failed` -- the pool *looked* like it worked and returned
    garbage.  Under spawn the same fit is bit-identical to the parent."""
    assert FitPool(n_workers=None).mp_context == "spawn"
    with FitPool(n_workers=1) as pool:
        assert pool.facts()["start_method"] == "spawn"


def test_self_check_compares_a_real_fit_against_the_parent():
    """The guard that fork needed: a broken worker fails every start and reports
    it as an ordinary rejection, so a whole epoch can come back "nothing fitted"
    and look like a data problem."""
    task = _task(0, readouts=("band",))
    with FitPool(n_workers=1) as pool:
        rep = pool.self_check(task)
    assert rep["checked"] is True and rep["start_method"] == "spawn"
    assert "band" in rep["losses"]
    assert FitPool(n_workers=None).self_check(task)["checked"] is False


def test_self_check_raises_when_a_worker_disagrees(monkeypatch):
    import q3vl.where.fitpool as fp

    task = _task(0, readouts=("band",))
    with FitPool(n_workers=1) as pool:
        broken = fp.FitResult(latent=None, readout="band", loss=float("inf"),
                              status="rejected", reject_reason="all_starts_failed",
                              n_starts=10, n_failed_starts=10)

        class _Fut:
            @staticmethod
            def result():
                return ("s0", {"band": broken})

        monkeypatch.setattr(pool._ex, "submit", lambda *a, **k: _Fut())
        with pytest.raises(RuntimeError, match="self-check failed"):
            pool.self_check(task)


def test_a_failing_start_records_why():
    """`all_starts_failed` with no reason attached is what made the fork breakage
    look like a hard sample."""
    from q3vl.where.oracle import fit_latent

    t = _task(0, readouts=("band",))
    phi = t.phi.clone()
    phi[0, 0] = float("nan")
    res = fit_latent(phi, t.target, "band", FitConfig(n_random=1, max_iter=10, seed=0))
    assert res.status == "rejected"
    assert "start_errors" in res.to_dict()
    assert "start_errors" in res.rejection_row()


def test_tasks_carry_both_readouts_so_phi_travels_once():
    t = _task(0)
    assert t.readouts == ("band", "cband12")
    res = run_fits_serial([t])
    assert set(res["s0"]) == {"band", "cband12"}


def test_calibrator_step_is_identical_with_and_without_a_pool():
    """End to end: a full BA-3-Joint step through the pool must produce the same
    loss, the same gradient norm and the same projector as the serial step."""
    pin_single_thread()
    cfg = CalibConfig(arm="BA-3-Joint", batch_size=3,
                      inner_fit=FitConfig(n_random=1, max_iter=20, seed=0))
    batch = [_mock_sample(i) for i in range(3)]

    a = Calibrator(cfg, total_steps=2)
    out_a = a.step(batch)
    b = Calibrator(cfg, total_steps=2)
    with FitPool(n_workers=3) as pool:
        out_b = b.step(batch, pool=pool)

    assert out_a["loss"] == out_b["loss"]
    assert out_a["grad_norm"] == out_b["grad_norm"]
    assert out_a["n_used_per_readout"] == out_b["n_used_per_readout"]
    assert out_a["fit_loss_per_readout"] == out_b["fit_loss_per_readout"]
    assert torch.equal(a.projector.weight, b.projector.weight)
    assert a.projector.digest() == b.projector.digest()


def test_step_reports_its_timing_breakdown():
    """The schedule is decided from these numbers, so they ship with every step."""
    cfg = CalibConfig(arm="BA-1-Band", batch_size=2,
                      inner_fit=FitConfig(n_random=1, max_iter=10, seed=0))
    out = Calibrator(cfg, total_steps=1).step([_mock_sample(i) for i in range(2)])
    t = out["timing_s"]
    assert set(t) == {"phi_gpu", "fit", "outer", "backward", "total"}
    assert t["total"] >= t["fit"] > 0


def test_pool_actually_runs_concurrently():
    """Guards against a 'pool' that silently degraded to serial."""
    pin_single_thread()
    tasks = [_task(i, readouts=("band",)) for i in range(8)]
    t0 = time.perf_counter()
    run_fits_serial(tasks)
    serial = time.perf_counter() - t0
    with FitPool(n_workers=8) as pool:
        pool.warmup()
        t0 = time.perf_counter()
        pool.run(tasks)
        parallel = time.perf_counter() - t0
    assert parallel < 0.7 * serial, f"serial {serial:.2f}s vs pool {parallel:.2f}s"


def test_facts_report_the_configuration():
    with FitPool(n_workers=2) as pool:
        f = pool.facts()
    assert f["n_workers"] == 2 and f["mode"] == "process_pool"
    assert f["threads_per_worker"] == 1
