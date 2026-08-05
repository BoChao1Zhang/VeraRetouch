"""Contracts the pending jobs must honour, checked without running them.

These scripts cannot be executed until the GPUs free up, so the parts that are
pure interface -- the z grid Where-B will consume, the conventions published
next to it, the pipeline's rejection bookkeeping -- are pinned here instead.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from q3vl.where.basis import Latent
from q3vl.where.config import (
    CBAND_NORMALIZATION, CURVE_Z_N, S_DOMAIN, S_OOD_FRAC_MAX,
)
from q3vl.where.readout import apply_readout, param_shapes
from q3vl.where.scripts.make_oracle_latents import CURVE_Z, curve_of

DT = torch.float64


# --- B-7: the r*(z) grid Where-B consumes ----------------------------------

def test_curve_grid_is_the_protocol_grid():
    """Protocol 5.5: `L_curve = mean_z |R(z;rho_pred) - r*(z)|, z=linspace(-3,3,257)`.
    A 121-point vector cannot be differenced against a 257-point one, so Where-B
    would have to interpolate (error injected into a supervision target) or
    deviate from 5.5."""
    assert CURVE_Z_N == 257
    assert CURVE_Z.shape == (257,)
    assert np.allclose(CURVE_Z, np.linspace(-3.0, 3.0, 257))
    assert CURVE_Z[0] == -3.0 and CURVE_Z[-1] == 3.0
    assert S_DOMAIN == (-3.0, 3.0)


@pytest.mark.parametrize("readout", ["band", "cband12"])
def test_curve_of_samples_that_grid_under_the_declared_convention(readout):
    g = torch.Generator().manual_seed(3)
    rho = {k: (torch.randn(s, generator=g, dtype=DT) if s else
               torch.randn((), generator=g, dtype=DT))
           for k, s in param_shapes(readout).items()}
    lat = Latent(readout, torch.zeros((), dtype=DT), torch.zeros((), dtype=DT),
                 torch.randn(71, generator=g, dtype=DT), rho)
    curve = curve_of(lat)
    assert len(curve) == 257
    want = apply_readout(readout, torch.tensor(CURVE_Z, dtype=DT), rho,
                         normalization=CBAND_NORMALIZATION)
    assert np.allclose(np.asarray(curve), want.numpy(), atol=1e-12)
    json.dumps(curve)                       # must survive the shard payload
    assert all(0.0 <= v <= 1.0 for v in curve)


def test_the_normalisation_convention_is_published_not_implied():
    """N-25: r*(z) is sampled under `logsumexp`; Where-B must recompute
    `R(z;rho_pred)` the same way or the two sides of L_curve are different
    functions.  The field is part of the payload, so it travels with the data."""
    import inspect

    from q3vl.where.scripts import make_oracle_latents as job

    src = inspect.getsource(job)
    assert 'd["cband_normalization"] = CBAND_NORMALIZATION' in src, \
        "per-sample payload must declare the convention"
    assert '"cband_normalization": CBAND_NORMALIZATION' in src, \
        "the run report must declare it too"
    assert CBAND_NORMALIZATION == "logsumexp"


# --- N-20: the pre-registered domain gate ----------------------------------

def test_domain_gate_is_pre_registered_and_used_by_both_consumers():
    import inspect

    from q3vl.where import preflight
    from q3vl.where.scripts import sweep_upsample

    assert S_OOD_FRAC_MAX == 0.01
    assert "S_OOD_FRAC_MAX" in inspect.getsource(preflight.check_calibration_health)
    sweep_src = inspect.getsource(sweep_upsample.main)
    assert "admissible" in sweep_src, "selection must filter before it ranks"
    # the ranking key must not be reachable without passing the gate first
    assert sweep_src.index("admissible = ") < sweep_src.index("best = max(admissible")


# --- N-19: late drops are recorded, not raised ------------------------------

def test_pipeline_records_mask_io_failures_instead_of_raising(monkeypatch):
    """One unreadable .cgt.png out of 75,544 must cost one sample, not a GPU-day."""
    from q3vl.where.maskdata import MaskLookupError
    from q3vl.where.pipeline import WhereADataSource

    src = WhereADataSource.__new__(WhereADataSource)
    src.visual = None
    src.processor = None
    src.device = "cpu"
    src.exclude_low = False
    src.attach_hi = False
    src.maskviews = None
    src.rejections = []

    class _BoomResolver:
        def resolve(self, record):
            raise MaskLookupError("no .cgt.png for this sample")

    class _Store:
        def read(self, ref):
            return b"jpeg-bytes"        # decoded by the patched prepare_image

    src.resolver = _BoomResolver()
    src.store = _Store()

    record = {
        "sample_id": "sft_bad", "build": "l1", "render_mode": "local",
        "winner_confidence": "normal", "source_sample_id": "batch-0_0_candidate_x",
        "image": {"out_h": 512, "out_w": 768, "oriented_h": 1024, "oriented_w": 1536,
                  "origin": {"root": "/nowhere"}},
    }

    class _Ref:
        members = {"image": object()}

    monkeypatch.setattr("q3vl.where.pipeline.prepare_image",
                        lambda data: (_FakeImg(), _FakeGeom()))
    out = src.prepare(_Ref(), record)
    assert out is None
    assert len(src.rejections) == 1
    assert src.rejections[0]["reason"] == "mask_io"
    assert src.rejections[0]["sample_id"] == "sft_bad"
    assert "MaskLookupError" in src.rejections[0]["error"]


class _FakeImg:
    size = (768, 512)


class _FakeGeom:
    out_h, out_w, grid_h, grid_w = 512, 768, 32, 48


def test_count_eligible_declares_itself_an_upper_bound():
    """N-19: `prepare()` can still drop a sample the count accepted, so the
    driver reconciles the gap instead of failing blind after a whole epoch."""
    import inspect

    from q3vl.where.pipeline import WhereADataSource
    from q3vl.where.scripts import run_calibration

    doc = WhereADataSource.count_eligible.__doc__ or ""
    assert "upper\n        bound" in doc or "upper bound" in doc.replace("\n", " ")
    src = inspect.getsource(run_calibration.main)
    assert "gap_explained" in src
    assert "late_drops" in src
    # the fatal branch must be conditional on the gap being unexplained
    assert 'if gap != explained:' in src


# --- BA-0 must not run an epoch that fits nothing --------------------------

def test_ba0_has_no_readouts_so_a_training_pass_would_be_empty():
    """Caught at S4 launch: BA-0-Fixed's ARM_READOUTS is (), so `step()` fits
    nothing, computes no loss and takes no gradient.  Running the epoch anyway
    is a full pass over 75,544 samples (~4.6 h of data loading) that leaves B
    exactly at its seeded-orthogonal initialisation."""
    from q3vl.where.calibrate import Calibrator, arm_readouts
    from q3vl.where.config import CalibConfig, FitConfig
    from q3vl.where.tests.test_calibrate import _mock_sample

    assert arm_readouts("BA-0-Fixed") == ()
    cal = Calibrator(CalibConfig(arm="BA-0-Fixed",
                                 inner_fit=FitConfig(n_random=1, max_iter=5)))
    assert cal.trains_projector is False
    before = cal.projector.digest()
    out = cal.step([_mock_sample(0)])
    assert out["n_used_per_readout"] == {}
    assert out["grad_norm"] is None
    assert cal.n_fits == 0, "an epoch step for BA-0 performs no fit at all"
    assert cal.projector.digest() == before


def test_driver_skips_the_epoch_for_a_non_training_arm():
    import inspect

    from q3vl.where.scripts import run_calibration

    src = inspect.getsource(run_calibration.main)
    assert "skip_epoch = not cal.trains_projector" in src
    assert "[] if skip_epoch else _batched" in src, \
        "the epoch loop must be short-circuited, not merely logged"
    assert '"epoch_skipped": skip_epoch' in src
    # and the pool self-check must still happen, on the readouts evaluation uses
    assert '("band", "cband12")' in src, \
        "a skipped epoch must still check the pool, with non-empty readouts"


def test_self_check_on_empty_readouts_would_be_vacuous():
    """Why the check moved: with no readouts it compares two empty dicts."""
    from q3vl.where.fitpool import FitPool, FitTask
    from q3vl.where.config import FitConfig

    task = FitTask(key="s", phi=torch.zeros(4, 71, dtype=torch.float64),
                   target=torch.zeros(4, dtype=torch.float64),
                   readouts=(), cfg=FitConfig(n_random=1, max_iter=5))
    with FitPool(n_workers=1) as pool:
        rep = pool.self_check(task)
    assert rep["checked"] is True and rep["losses"] == {}, (
        "an empty-readout self-check passes without testing anything -- which is "
        "exactly what BA-0 was doing at launch")


def test_curve_of_is_device_independent():
    """The S5 job moves each latent to the calibrator's device for evaluation, so
    a curve built from a bare `torch.tensor(CURVE_Z)` mixes devices and raises --
    which is exactly what happened on the first S5 split.  r*(z) is a
    serialisation artifact: CPU float64 is its canonical form."""
    import inspect

    from q3vl.where.scripts import make_oracle_latents as job

    src = inspect.getsource(job.curve_of)
    assert ".cpu()" in src, "the readout params must be pulled to CPU"
    assert ".double()" in src, "and evaluated in float64"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU free")
def test_curve_of_accepts_a_latent_on_the_gpu():
    g = torch.Generator().manual_seed(1)
    rho = {k: (torch.randn(s, generator=g, dtype=DT) if s else
               torch.randn((), generator=g, dtype=DT))
           for k, s in param_shapes("band").items()}
    lat = Latent("band", torch.zeros((), dtype=DT), torch.zeros((), dtype=DT),
                 torch.randn(71, generator=g, dtype=DT), rho)
    on_cpu = curve_of(lat)
    on_gpu = curve_of(lat.to("cuda"))
    assert len(on_gpu) == 257
    assert np.allclose(np.asarray(on_cpu), np.asarray(on_gpu), atol=1e-12)


def test_s5_fits_go_through_the_pool_not_one_at_a_time():
    """The first S5 launch created the pool and then fitted through
    `cal.fit_sample`, one fit at a time in-process: ~2.5 s per full-config fit
    x 151,088 fits for train is ~100 h.  The fits must go through `run_fits`,
    the same seam the calibration loop uses."""
    import inspect

    from q3vl.where.scripts import make_oracle_latents as job

    src = inspect.getsource(job.main)
    assert "cal.run_fits(tasks, pool)" in src, "S5 must dispatch through the pool"
    assert "cal.fit_sample(" not in src, "no per-fit serial path may remain"
    assert "cal._chunks(" in src, "and it must be chunked, not one sample per dispatch"
    # the seed must still be the global sample index, or the numbers move
    assert "tasks[-1].seed_offset = base + k" in src
