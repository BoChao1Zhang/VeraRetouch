"""Protocol 4.2 -- one guided upsample, of the scalar, after the combination."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from q3vl.where.basis import Latent
from q3vl.where.config import PHI_DIR_DIM, SEM_DIM, UpsampleConfig
from q3vl.where.readout import param_shapes
from q3vl.where.upsample import (
    ChannelOrderError, area_resize, box_mean, combine_then_upsample,
    guided_upsample, luma_guide,
)

DT = torch.float64


def _guide_step(H=64, W=64):
    g = torch.zeros(1, 1, H, W, dtype=DT)
    g[..., W // 2:] = 1.0
    return g


def test_multichannel_input_is_refused():
    """'You may not upsample the 64 channels first and combine afterwards.'"""
    s = torch.randn(1, SEM_DIM, 8, 8, dtype=DT)
    with pytest.raises(ChannelOrderError):
        guided_upsample(s, _guide_step(32, 32))
    # even two channels is a violation
    with pytest.raises(ChannelOrderError):
        guided_upsample(torch.randn(1, 2, 8, 8, dtype=DT), _guide_step(32, 32))


def test_scalar_input_is_accepted():
    out = guided_upsample(torch.randn(1, 1, 8, 8, dtype=DT), _guide_step(32, 32))
    assert out.shape == (1, 1, 32, 32)


def test_constant_field_is_preserved():
    s = torch.full((1, 1, 8, 8), -1.75, dtype=DT)
    out = guided_upsample(s, _guide_step(64, 64))
    assert torch.allclose(out, torch.full_like(out, -1.75), atol=1e-9)


def test_edges_are_sharper_than_bilinear():
    """Edge-awareness is the reason this filter is here at all."""
    H = W = 128
    guide = _guide_step(H, W)
    low = area_resize(guide, (8, 8)) * 6.0 - 3.0            # s_low follows the step
    out = guided_upsample(low, guide, UpsampleConfig(radius_low=1, eps=1e-4))
    bil = F.interpolate(low, size=(H, W), mode="bilinear", align_corners=False)

    row = H // 2
    lo, hi = float(out[0, 0, row].min()), float(out[0, 0, row].max())
    span = hi - lo
    width = lambda x: int(((x[0, 0, row] > lo + 0.1 * span)  # noqa: E731
                           & (x[0, 0, row] < hi - 0.1 * span)).sum())
    assert width(out) < width(bil), "guided upsample must beat bilinear at the edge"
    # a low-res cell is 16 full-res pixels wide; the transition must stay well
    # inside one cell
    assert width(out) <= 8
    assert width(bil) >= 2 * width(out)


def test_box_mean_matches_a_manual_window():
    x = torch.arange(25, dtype=DT).reshape(1, 1, 5, 5)
    out = box_mean(x, 1)
    # interior pixel (2,2): mean of the 3x3 neighbourhood
    assert float(out[0, 0, 2, 2]) == pytest.approx(float(x[0, 0, 1:4, 1:4].mean()))


def test_luma_guide():
    img = torch.zeros(1, 3, 2, 2, dtype=DT)
    img[:, 1] = 1.0
    assert torch.allclose(luma_guide(img), torch.full((1, 1, 2, 2), 0.7152, dtype=DT))


def test_combine_then_upsample_enforces_the_order():
    gh, gw = 8, 12
    H, W = gh * 16, gw * 16
    g = torch.Generator().manual_seed(0)
    phi = torch.randn(gh * gw, PHI_DIR_DIM, generator=g, dtype=DT)
    lat = Latent("band", torch.zeros((), dtype=DT), torch.zeros((), dtype=DT),
                 torch.randn(PHI_DIR_DIM, generator=g, dtype=DT),
                 {k: (torch.zeros(s, dtype=DT) if s else torch.zeros((), dtype=DT))
                  for k, s in param_shapes("band").items()})
    guide = torch.rand(1, 1, H, W, generator=g, dtype=DT)
    s_hi, s_lo = combine_then_upsample(phi, lat, gh, gw, guide)
    assert s_hi.shape == (1, 1, H, W)
    assert s_lo.shape == (gh * gw,)          # scalar per grid cell, before upsample
    with pytest.raises(ValueError):
        combine_then_upsample(phi[:-1], lat, gh, gw, guide)


def test_area_resize_down_is_an_average():
    x = torch.arange(16, dtype=DT).reshape(1, 1, 4, 4)
    out = area_resize(x, (2, 2))
    assert float(out[0, 0, 0, 0]) == pytest.approx(float(x[0, 0, :2, :2].mean()))


def test_guide_shape_is_validated():
    with pytest.raises(ValueError):
        guided_upsample(torch.randn(1, 1, 4, 4, dtype=DT), torch.randn(1, 3, 8, 8, dtype=DT))
    with pytest.raises(ValueError):
        guided_upsample(torch.randn(1, 4, 4, dtype=DT), _guide_step(8, 8))


def test_gradient_flows_through_the_upsample():
    s = torch.randn(1, 1, 8, 8, dtype=DT, requires_grad=True)
    guided_upsample(s, _guide_step(32, 32)).sum().backward()
    assert s.grad is not None and float(s.grad.abs().sum()) > 0


# --- REVIEW-impl-WhereA B-4 / N-2 ------------------------------------------

def test_batch_axis_cannot_smuggle_channels_past_the_guard():
    """N-2: the channel guard alone is bypassable by folding the 64 semantic
    channels into the batch axis; the batch/guide agreement closes it."""
    sem_as_batch = torch.randn(SEM_DIM, 1, 8, 12, dtype=DT)
    with pytest.raises(ChannelOrderError, match="batch"):
        guided_upsample(sem_as_batch, _guide_step(64, 64))
    # a genuine batch of images is still fine
    out = guided_upsample(torch.randn(3, 1, 8, 12, dtype=DT),
                          _guide_step(64, 64).repeat(3, 1, 1, 1))
    assert out.shape == (3, 1, 64, 64)


def test_guided_upsample_pushes_s_out_of_domain_and_reports_it():
    """B-4: this overshoot is real and used to be invisible.  The report must
    show it *before* the clamp, so 'the field left (-3,3)' is never silent."""
    H = W = 128
    guide = torch.rand(1, 1, H, W, generator=torch.Generator().manual_seed(3), dtype=DT)
    s_low = torch.where(torch.rand(1, 1, 8, 8, generator=torch.Generator().manual_seed(4), dtype=DT) > 0.5,
                        torch.tensor(3.0, dtype=DT), torch.tensor(-3.0, dtype=DT))

    # eps controls how aggressively the filter extrapolates from the guide, so
    # it also controls how far past the domain it can throw s.  A small eps makes
    # the overshoot reproducible; the D5 winner (eps=1e-2) is deliberately
    # gentler, which is part of why the sweep's domain gate selected it.
    raw_cfg = UpsampleConfig(radius_low=2, eps=1e-3, clamp_domain=False)
    s_raw, rep_raw = guided_upsample(s_low, guide, raw_cfg, return_domain_report=True)
    assert rep_raw["raw_max"] > 3.0 or rep_raw["raw_min"] < -3.0, \
        "the filter is expected to overshoot on an uncorrelated guide"
    assert rep_raw["frac_out_of_domain"] > 0
    assert rep_raw["clamped"] is False
    assert float(s_raw.max()) == pytest.approx(rep_raw["raw_max"])

    clamped_cfg = UpsampleConfig(radius_low=2, eps=1e-3, clamp_domain=True)
    s_clamped, rep = guided_upsample(s_low, guide, clamped_cfg, return_domain_report=True)
    assert rep["clamped"] is True
    # the report still shows the *pre-clamp* truth
    assert rep["raw_max"] == pytest.approx(rep_raw["raw_max"])
    assert rep["frac_out_of_domain"] == pytest.approx(rep_raw["frac_out_of_domain"])
    assert float(s_clamped.max()) <= 3.0 + 1e-12
    assert float(s_clamped.min()) >= -3.0 - 1e-12

    # whatever the parameters, the domain fields are always reported
    _, rep_default = guided_upsample(s_low, guide, UpsampleConfig(),
                                     return_domain_report=True)
    assert set(rep_default) >= {"raw_min", "raw_max", "frac_out_of_domain", "clamped"}


def test_clamping_leaves_in_domain_values_untouched():
    """The clamp must restore the producer's invariant, not rescale the field
    (a tanh squash would move every in-domain value; per-image rescaling is a
    red line)."""
    guide = torch.rand(1, 1, 64, 64, generator=torch.Generator().manual_seed(6), dtype=DT)
    s_low = torch.rand(1, 1, 8, 8, generator=torch.Generator().manual_seed(7), dtype=DT) * 2 - 1
    a = guided_upsample(s_low, guide, UpsampleConfig(clamp_domain=False))
    b = guided_upsample(s_low, guide, UpsampleConfig(clamp_domain=True))
    inside = a.abs() < 3.0
    assert bool(inside.all()), "this fixture should stay inside the domain"
    assert torch.allclose(a[inside], b[inside], atol=0)


def test_domain_default_matches_the_producer():
    from q3vl.where.config import S_DOMAIN, S_SCALE
    assert S_DOMAIN == (-S_SCALE, S_SCALE)
    assert UpsampleConfig().domain == S_DOMAIN
    assert UpsampleConfig().clamp_domain is True


def test_combine_then_upsample_can_return_the_domain_report():
    gh, gw = 8, 12
    g = torch.Generator().manual_seed(0)
    phi = torch.randn(gh * gw, PHI_DIR_DIM, generator=g, dtype=DT)
    lat = Latent("band", torch.zeros((), dtype=DT), torch.tensor(2.0, dtype=DT),
                 torch.randn(PHI_DIR_DIM, generator=g, dtype=DT),
                 {k: (torch.zeros(s, dtype=DT) if s else torch.zeros((), dtype=DT))
                  for k, s in param_shapes("band").items()})
    guide = torch.rand(1, 1, gh * 16, gw * 16, generator=g, dtype=DT)
    s_hi, s_lo, rep = combine_then_upsample(phi, lat, gh, gw, guide,
                                            return_domain_report=True)
    assert set(rep) >= {"domain", "raw_min", "raw_max", "frac_out_of_domain", "clamped"}
    assert float(s_lo.abs().max()) < 3.0        # producer invariant
    assert float(s_hi.abs().max()) <= 3.0 + 1e-12
