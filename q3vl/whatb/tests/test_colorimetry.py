"""dE00 against the authors' own test table; Lab against known anchors."""

from __future__ import annotations

import pytest
import torch

from q3vl.whatb.colorimetry import (
    chroma_hue,
    delta_e00,
    delta_e76,
    linear_to_srgb,
    srgb_to_lab,
    srgb_to_linear,
)

# Sharma, Wu & Dalal (2005) CIEDE2000 test data, all 34 pairs, fetched
# 2026-08-15 from
#   https://hajim.rochester.edu/ece/sites/gsharma/ciede2000/dataNprograms/
#   ciede2000testdata.txt      (HTTP 200, 1830 B,
#   sha256 44aebb39107128328add54fbef5ac8ee89909e50508f448a1580adea2058a4b8)
# columns: L1 a1 b1 L2 a2 b2 dE00
SHARMA_PAIRS = [
    (50.0000, 2.6772, -79.7751, 50.0000, 0.0000, -82.7485, 2.0425),
    (50.0000, 3.1571, -77.2803, 50.0000, 0.0000, -82.7485, 2.8615),
    (50.0000, 2.8361, -74.0200, 50.0000, 0.0000, -82.7485, 3.4412),
    (50.0000, -1.3802, -84.2814, 50.0000, 0.0000, -82.7485, 1.0000),
    (50.0000, -1.1848, -84.8006, 50.0000, 0.0000, -82.7485, 1.0000),
    (50.0000, -0.9009, -85.5211, 50.0000, 0.0000, -82.7485, 1.0000),
    (50.0000, 0.0000, 0.0000, 50.0000, -1.0000, 2.0000, 2.3669),
    (50.0000, -1.0000, 2.0000, 50.0000, 0.0000, 0.0000, 2.3669),
    (50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0009, 7.1792),
    (50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0010, 7.1792),
    (50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0011, 7.2195),
    (50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0012, 7.2195),
    (50.0000, -0.0010, 2.4900, 50.0000, 0.0009, -2.4900, 4.8045),
    (50.0000, -0.0010, 2.4900, 50.0000, 0.0010, -2.4900, 4.8045),
    (50.0000, -0.0010, 2.4900, 50.0000, 0.0011, -2.4900, 4.7461),
    (50.0000, 2.5000, 0.0000, 50.0000, 0.0000, -2.5000, 4.3065),
    (50.0000, 2.5000, 0.0000, 73.0000, 25.0000, -18.0000, 27.1492),
    (50.0000, 2.5000, 0.0000, 61.0000, -5.0000, 29.0000, 22.8977),
    (50.0000, 2.5000, 0.0000, 56.0000, -27.0000, -3.0000, 31.9030),
    (50.0000, 2.5000, 0.0000, 58.0000, 24.0000, 15.0000, 19.4535),
    (50.0000, 2.5000, 0.0000, 50.0000, 3.1736, 0.5854, 1.0000),
    (50.0000, 2.5000, 0.0000, 50.0000, 3.2972, 0.0000, 1.0000),
    (50.0000, 2.5000, 0.0000, 50.0000, 1.8634, 0.5757, 1.0000),
    (50.0000, 2.5000, 0.0000, 50.0000, 3.2592, 0.3350, 1.0000),
    (60.2574, -34.0099, 36.2677, 60.4626, -34.1751, 39.4387, 1.2644),
    (63.0109, -31.0961, -5.8663, 62.8187, -29.7946, -4.0864, 1.2630),
    (61.2901, 3.7196, -5.3901, 61.4292, 2.2480, -4.9620, 1.8731),
    (35.0831, -44.1164, 3.7933, 35.0232, -40.0716, 1.5901, 1.8645),
    (22.7233, 20.0904, -46.6940, 23.0331, 14.9730, -42.5619, 2.0373),
    (36.4612, 47.8580, 18.3852, 36.2715, 50.5065, 21.2231, 1.4146),
    (90.8027, -2.0831, 1.4410, 91.1528, -1.6435, 0.0447, 1.4441),
    (90.9257, -0.5406, -0.9208, 88.6381, -0.8985, -0.7239, 1.5381),
    (6.7747, -0.2908, -2.4247, 5.8714, -0.0985, -2.2286, 0.6377),
    (2.0776, 0.0795, -1.1350, 0.9033, -0.0636, -0.5514, 0.9082),
]


def test_delta_e00_matches_sharma_test_data():
    d = torch.tensor(SHARMA_PAIRS, dtype=torch.float64)
    got = delta_e00(d[:, 0:3], d[:, 3:6])
    err = (got - d[:, 6]).abs()
    # the published table is rounded to four decimals
    assert float(err.max()) < 1e-4, f"worst pair err {float(err.max())}"


def test_delta_e00_is_symmetric_and_zero_on_identity():
    d = torch.tensor(SHARMA_PAIRS, dtype=torch.float64)
    a, b = d[:, 0:3], d[:, 3:6]
    assert torch.allclose(delta_e00(a, b), delta_e00(b, a), atol=1e-10)
    assert float(delta_e00(a, a).abs().max()) == 0.0


def test_delta_e00_broadcasts_and_keeps_shape():
    x = torch.rand(4, 5, 3, dtype=torch.float64)
    y = torch.rand(4, 5, 3, dtype=torch.float64)
    assert delta_e00(x, y).shape == (4, 5)
    assert delta_e76(x, y).shape == (4, 5)


def test_srgb_to_lab_anchors():
    rgb = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
                       dtype=torch.float64)
    lab = srgb_to_lab(rgb)
    assert abs(float(lab[0, 0]) - 100.0) < 1e-3
    assert abs(float(lab[0, 1])) < 1e-2 and abs(float(lab[0, 2])) < 1e-2
    assert float(lab[1].abs().max()) < 1e-9
    # mid grey: neutral, and its lightness is the well-known ~53.39
    assert abs(float(lab[2, 1])) < 1e-2 and abs(float(lab[2, 2])) < 1e-2
    assert 53.0 < float(lab[2, 0]) < 53.8


def test_eotf_round_trip():
    x = torch.rand(1000, 3, dtype=torch.float64)
    assert torch.allclose(linear_to_srgb(srgb_to_linear(x)), x, atol=1e-12)


def test_chroma_hue_mask_is_the_frozen_eps():
    lab = torch.tensor([[50.0, 0.0, 0.0], [50.0, 3.0, 4.0]], dtype=torch.float64)
    c, h, valid = chroma_hue(lab, eps_c=1e-3)
    assert bool(valid[0]) is False and bool(valid[1]) is True
    assert abs(float(c[1]) - 5.0) < 1e-12
    assert torch.allclose(h[1], torch.tensor([0.6, 0.8], dtype=torch.float64))
    # C -> 0 gives a finite (not NaN) hue vector: the mask, not 0/0
    assert torch.isfinite(h[0]).all()


# --------------------------------------------------------------------------- #
# B1: neutral colours must have a FINITE backward, not just a finite forward
# --------------------------------------------------------------------------- #
NEUTRAL = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.5, 0.5, 0.5),
           (0.04045, 0.04045, 0.04045), (1e-8, 1e-8, 1e-8)]


@pytest.mark.parametrize("rgb", NEUTRAL)
def test_srgb_to_lab_gradient_is_finite_at_neutral_colours(rgb):
    y = torch.tensor([list(rgb)], dtype=torch.float64, requires_grad=True)
    out = srgb_to_lab(y)
    assert bool(torch.isfinite(out).all())
    out.sum().backward()
    assert bool(torch.isfinite(y.grad).all()), f"{rgb}: {y.grad}"


@pytest.mark.parametrize("rgb", NEUTRAL)
def test_chroma_hue_gradient_is_finite_at_neutral_colours(rgb):
    """``C = sqrt(a^2+b^2)`` at ``a == b == 0`` was ``0/0`` (review blocker B1)."""
    y = torch.tensor([list(rgb)], dtype=torch.float64, requires_grad=True)
    c, h, valid = chroma_hue(srgb_to_lab(y))
    assert bool(torch.isfinite(c).all()) and bool(torch.isfinite(h).all())
    (c.sum() + h.sum()).backward()
    assert bool(torch.isfinite(y.grad).all()), f"{rgb}: {y.grad}"


def test_the_forward_of_the_fix_is_unchanged():
    """The guards may not move a single published number."""
    torch.manual_seed(0)
    x = torch.rand(64, 3, dtype=torch.float64)
    lab = srgb_to_lab(x)
    a, b = lab[..., 1], lab[..., 2]
    c_ref = torch.sqrt((a * a + b * b).clamp_min(0.0))
    c, h, _ = chroma_hue(lab)
    assert torch.equal(c, c_ref)
    assert torch.equal(h, torch.stack((a / c_ref.clamp_min(1e-3),
                                       b / c_ref.clamp_min(1e-3)), dim=-1))
    # ... including at the exact neutral where the old spelling produced NaN
    zero = torch.zeros(1, 3, dtype=torch.float64)
    c0, h0, valid0 = chroma_hue(srgb_to_lab(zero))
    assert float(c0) == 0.0 and float(h0.abs().sum()) == 0.0 and not bool(valid0)


def test_delta_e_gradients_are_finite_for_an_identical_pair():
    """W8: dE76 / dE00 of a converged prediction against its own target."""
    lab = torch.tensor([[50.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64,
                       requires_grad=True)
    ref = lab.detach().clone()
    for fn in (delta_e76, delta_e00):
        if lab.grad is not None:
            lab.grad = None
        out = fn(lab, ref)
        assert bool(torch.isfinite(out).all())
        out.sum().backward()
        assert bool(torch.isfinite(lab.grad).all()), fn.__name__


def test_every_arm_l_hc_has_a_finite_gradient_at_a_neutral_prediction():
    """The five arms consume the shared functions; qdual no longer forks them."""
    from q3vl.whatb.arms import affonly, carrier, g4d, idgate, qdual

    target = torch.tensor([[[0.4, 0.2, 0.1], [0.0, 0.0, 0.0]]])
    op = torch.full((1, 4), 0.5)

    def _pred():
        # exactly black, exactly white: both are legal clamped predictions
        return torch.tensor([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]], requires_grad=True)

    cases = {
        "carrier": lambda p: carrier.hue_chroma_loss(p, target)[0],
        "affonly": lambda p: affonly.loss_terms(p, target, op).total,
        "idgate": lambda p: idgate.glut_loss(p, target, op).total,
        "g4d": lambda p: g4d.l_hc(p, target)[0],
        "qdual": lambda p: qdual.hue_chroma_loss(p, target)[0],
    }
    for name, fn in cases.items():
        p = _pred()
        out = fn(p)
        assert bool(torch.isfinite(out).all()), name
        out.backward()
        assert bool(torch.isfinite(p.grad).all()), f"{name}: {p.grad}"
