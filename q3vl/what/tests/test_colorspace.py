"""Protocol 9.2 -- Lab has to be dimensionless before it reaches a loss."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from q3vl.what.colorspace import (
    chroma_hue,
    delta_e00,
    hue_angular_error_deg,
    hue_cos_diff,
    normalised_chroma,
    srgb_to_lab,
    srgb_to_lab_norm,
)
from q3vl.what.config import CHROMA_NORM


def test_lab_matches_the_glut_repro_implementation():
    from model.glut_repro.model_rdg import srgb_to_lab as ref

    torch.manual_seed(0)
    rgb = torch.rand(512, 3)
    assert torch.allclose(srgb_to_lab(rgb), ref(rgb), atol=1e-4)


def test_lab_agrees_with_skimage_up_to_the_white_point_convention():
    """Cross-check against an independent implementation.

    The residual ~0.015 Lab units is not precision: skimage's D65 white point is
    the ASTM triple ``(0.95047, 1.0, 1.08883)`` while this module derives it from
    the ``(0.3127, 0.3290)`` chromaticity, giving ``(0.950456, 1, 1.088754)``.
    Both are "D65".  The repo's own pinned implementation
    (``model/glut_repro``) uses the chromaticity form, and matching *it* exactly
    is what keeps this campaign's numbers comparable with the previous one, so
    that is the tighter assertion above.
    """
    pytest.importorskip("skimage")
    from skimage.color import rgb2lab

    torch.manual_seed(1)
    rgb = torch.rand(64, 64, 3, dtype=torch.float64)
    diff = np.abs(srgb_to_lab(rgb).numpy() - rgb2lab(rgb.numpy()))
    assert diff.max() < 0.05


def test_delta_e00_matches_the_glut_repro_implementation():
    from model.glut_repro.model_rdg import delta_e00 as ref

    torch.manual_seed(2)
    a, b = torch.rand(1024, 3), torch.rand(1024, 3)
    assert torch.allclose(delta_e00(a, b), ref(a, b), atol=1e-4)


def test_delta_e00_of_identical_colours_is_zero():
    torch.manual_seed(3)
    a = torch.rand(256, 3)
    assert float(delta_e00(a, a).max()) < 1e-4


def test_the_unit_error_is_real_and_the_normalised_form_fixes_it():
    """The A0 accident, measured: raw Lab a/b are 128x the normalised ones and
    L is 100x, so a loss mixing raw Lab with RGB is dominated by Lab."""
    torch.manual_seed(4)
    rgb = torch.rand(4096, 3)
    raw, norm = srgb_to_lab(rgb), srgb_to_lab_norm(rgb)
    assert float(raw[..., 0].abs().max()) > 90.0
    assert float(norm.abs().max()) <= 1.5
    ratio_ab = float(raw[..., 1].abs().max() / norm[..., 1].abs().max())
    assert abs(ratio_ab - 128.0) < 1e-2
    ratio_l = float(raw[..., 0].abs().max() / norm[..., 0].abs().max())
    assert abs(ratio_l - 100.0) < 1e-2


def test_hue_cos_diff_equals_the_atan2_form():
    torch.manual_seed(5)
    a, b = torch.rand(2048, 3), torch.rand(2048, 3)
    la, lb = srgb_to_lab_norm(a), srgb_to_lab_norm(b)
    ha, hb = chroma_hue(la)[1], chroma_hue(lb)[1]
    want = 1.0 - torch.cos(ha - hb)
    assert torch.allclose(hue_cos_diff(la, lb), want, atol=1e-5)


def test_hue_cos_diff_has_no_branch_cut_and_is_bounded():
    """The reason for not using atan2: opposite hues either side of +-pi."""
    la = torch.tensor([[0.5, 1.0, 0.001]])          # hue just above 0
    lb = torch.tensor([[0.5, 1.0, -0.001]])         # hue just below 2pi
    assert float(hue_cos_diff(la, lb)) < 1e-4
    opposite = torch.tensor([[0.5, -1.0, 0.0]])
    assert abs(float(hue_cos_diff(la, opposite)) - 2.0) < 1e-3
    d = hue_cos_diff(torch.rand(1000, 3), torch.rand(1000, 3))
    assert float(d.min()) >= 0.0 and float(d.max()) <= 2.0


def test_achromatic_points_do_not_produce_nan():
    grey = torch.full((8, 3), 0.5)
    la = srgb_to_lab_norm(grey)
    d = hue_cos_diff(la, la)
    assert torch.isfinite(d).all()
    assert torch.isfinite(normalised_chroma(la)).all()


def test_normalised_chroma_uses_a_global_constant_not_a_batch_max():
    """A per-image or per-batch normalisation is a campaign red line."""
    lo = srgb_to_lab_norm(torch.tensor([[0.5, 0.5, 0.5]]))
    hi = srgb_to_lab_norm(torch.tensor([[1.0, 0.0, 0.0]]))
    c_alone = float(normalised_chroma(hi))
    c_together = float(normalised_chroma(torch.cat([lo, hi]))[1])
    assert abs(c_alone - c_together) < 1e-9
    assert 0.0 <= c_alone <= 1.0
    assert math.isclose(CHROMA_NORM, math.sqrt(2.0), rel_tol=1e-12)


def test_hue_angular_error_is_wrapped_to_180():
    torch.manual_seed(6)
    a, b = torch.rand(512, 3), torch.rand(512, 3)
    d = hue_angular_error_deg(a, b)
    assert float(d.min()) >= 0.0 and float(d.max()) <= 180.0 + 1e-4
