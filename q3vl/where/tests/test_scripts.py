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
