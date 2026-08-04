"""Protocol 4.3 -- the two readouts, symbol by symbol."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from q3vl.where.config import (
    BAND_H_HI, BAND_H_LO, BAND_K_HI, BAND_K_LO, CBAND_EPS, CBAND_M,
    CBAND_SIG_HI, CBAND_SIG_LO,
)
from q3vl.where.readout import (
    apply_readout, band_params, bounded_sigmoid, bounds_report, cband_centres,
    cband_params, describe_params, inv_bounded_sigmoid, mirror_params, param_shapes,
)

DT = torch.float64


def _t(v):
    return torch.tensor(v, dtype=DT)


def test_band_matches_hand_computation():
    z = torch.linspace(-4, 4, 41, dtype=DT)
    raw = {"mu": _t(0.3), "h_raw": _t(0.7), "k_raw": _t(-0.2), "pi_raw": _t(1.1)}
    m = apply_readout("band", z, raw)

    p = band_params(raw)
    k, h, mu, pi = float(p["k"]), float(p["h"]), 0.3, float(p["pi"])
    zz = z.numpy()
    sig = lambda x: 1.0 / (1.0 + np.exp(-x))  # noqa: E731
    b = sig(k * (zz - mu + h)) - sig(k * (zz - mu - h))
    want = pi * b + (1 - pi) * (1 - b)
    assert np.allclose(m.numpy(), want, atol=1e-12)


def test_band_bounds_hold_under_extreme_raw():
    for raw_val in (-1e6, -50.0, 0.0, 50.0, 1e6):
        raw = {"mu": _t(0.0), "h_raw": _t(raw_val), "k_raw": _t(raw_val),
               "pi_raw": _t(raw_val)}
        p = band_params(raw)
        assert 0.0 < float(p["h"]), "protocol 4.3 requires h > 0"
        assert BAND_H_LO <= float(p["h"]) <= BAND_H_HI
        assert BAND_K_LO <= float(p["k"]) <= BAND_K_HI, "k must stay in [1, 40]"
        assert 0.0 <= float(p["pi"]) <= 1.0


def test_band_polarity_flips_the_mask():
    z = torch.linspace(-3, 3, 61, dtype=DT)
    base = {"mu": _t(0.0), "h_raw": _t(0.0), "k_raw": _t(1.0)}
    m_band = apply_readout("band", z, {**base, "pi_raw": _t(30.0)})    # pi -> 1
    m_notch = apply_readout("band", z, {**base, "pi_raw": _t(-30.0)})  # pi -> 0
    assert torch.allclose(m_band + m_notch, torch.ones_like(z), atol=1e-9)


def test_cband_matches_hand_computation():
    z = torch.linspace(-3, 3, 37, dtype=DT)
    g = torch.Generator().manual_seed(0)
    raw = {
        "sig_raw": torch.randn(CBAND_M, generator=g, dtype=DT),
        "o_raw": torch.randn(CBAND_M, generator=g, dtype=DT),
        "c_raw": torch.randn(CBAND_M, generator=g, dtype=DT),
    }
    m = apply_readout("cband12", z, raw)

    p = cband_params(raw)
    mu = p["mu"].numpy()
    sigma = p["sigma"].numpy()
    o = p["o"].numpy()
    c = p["c"].numpy()
    zz = z.numpy()[:, None]
    gi = o * np.exp(-0.5 * ((zz - mu) / sigma) ** 2)
    want = (c * gi).sum(1) / (gi.sum(1) + CBAND_EPS)
    assert np.allclose(m.numpy(), want, atol=1e-12)


def test_cband_centres_are_the_fixed_symmetric_grid():
    mu = cband_centres(dtype=DT)
    assert mu.shape == (CBAND_M,)
    assert np.allclose(mu.numpy(), np.linspace(-3, 3, CBAND_M))
    # symmetry is what makes the mirror an index reversal
    assert torch.allclose(mu, -torch.flip(mu, dims=(0,)), atol=1e-12)


def test_cband_sigma_is_bounded_sigmoid_not_bare_exp():
    for raw_val in (-1e6, -30.0, 0.0, 30.0, 1e6):
        raw = {k: torch.full((CBAND_M,), raw_val, dtype=DT)
               for k in ("sig_raw", "o_raw", "c_raw")}
        p = cband_params(raw)
        assert bool(((p["sigma"] >= CBAND_SIG_LO) & (p["sigma"] <= CBAND_SIG_HI)).all())
        assert bool(((p["o"] >= 0) & (p["o"] <= 1)).all())
        assert bool(((p["c"] >= 0) & (p["c"] <= 1)).all())


def test_cband_all_c_on_gives_one():
    z = torch.linspace(-2, 2, 21, dtype=DT)
    raw = {"sig_raw": torch.zeros(CBAND_M, dtype=DT),
           "o_raw": torch.zeros(CBAND_M, dtype=DT),
           "c_raw": torch.full((CBAND_M,), 30.0, dtype=DT)}
    m = apply_readout("cband12", z, raw)
    assert torch.allclose(m, torch.ones_like(m), atol=1e-6)


@pytest.mark.parametrize("readout", ["band", "cband12"])
def test_mirror_is_exact(readout):
    """m(-z; mirror(rho)) == m(z; rho): the identity the sign rule rests on."""
    g = torch.Generator().manual_seed(7)
    z = torch.linspace(-3.5, 3.5, 71, dtype=DT)
    for _ in range(20):
        raw = {k: (torch.randn(s, generator=g, dtype=DT) if s else
                   torch.randn((), generator=g, dtype=DT))
               for k, s in param_shapes(readout).items()}
        m = apply_readout(readout, z, raw)
        m_mirror = apply_readout(readout, -z, mirror_params(readout, raw))
        assert torch.allclose(m, m_mirror, atol=1e-12), readout


@pytest.mark.parametrize("readout", ["band", "cband12"])
def test_mirror_is_an_involution(readout):
    g = torch.Generator().manual_seed(3)
    raw = {k: (torch.randn(s, generator=g, dtype=DT) if s else
               torch.randn((), generator=g, dtype=DT))
           for k, s in param_shapes(readout).items()}
    twice = mirror_params(readout, mirror_params(readout, raw))
    for k in raw:
        assert torch.allclose(raw[k], twice[k], atol=1e-14)


def test_bounded_sigmoid_roundtrip():
    for v in (0.03, 0.5, 1.0, 2.4):
        raw = inv_bounded_sigmoid(v, BAND_H_LO, BAND_H_HI)
        back = float(bounded_sigmoid(torch.tensor(raw, dtype=DT), BAND_H_LO, BAND_H_HI))
        assert abs(back - v) < 1e-9


def test_batched_parameters_broadcast():
    z = torch.randn(4, 25, dtype=DT)
    raw_b = {"mu": torch.randn(4, dtype=DT), "h_raw": torch.randn(4, dtype=DT),
             "k_raw": torch.randn(4, dtype=DT), "pi_raw": torch.randn(4, dtype=DT)}
    m = apply_readout("band", z, raw_b)
    assert m.shape == (4, 25)
    for i in range(4):
        single = {k: v[i] for k, v in raw_b.items()}
        assert torch.allclose(m[i], apply_readout("band", z[i], single), atol=1e-12)

    raw_c = {k: torch.randn(4, CBAND_M, dtype=DT)
             for k in ("sig_raw", "o_raw", "c_raw")}
    mc = apply_readout("cband12", z, raw_c)
    assert mc.shape == (4, 25)
    for i in range(4):
        single = {k: v[i] for k, v in raw_c.items()}
        assert torch.allclose(mc[i], apply_readout("cband12", z[i], single), atol=1e-12)


def test_bounds_report_and_describe():
    raw = {"mu": _t(0.1), "h_raw": _t(0.0), "k_raw": _t(0.0), "pi_raw": _t(0.0)}
    rep = bounds_report("band", raw)
    assert rep["h_in_bounds"] and rep["k_in_bounds"] and rep["pi_in_bounds"]
    d = describe_params("band", raw)
    assert set(d) == {"mu", "h", "k", "pi"}

    rawc = {k: torch.zeros(CBAND_M, dtype=DT) for k in ("sig_raw", "o_raw", "c_raw")}
    repc = bounds_report("cband12", rawc)
    assert repc["sigma_in_bounds"] and repc["mu_grid_fixed"]


def test_unknown_readout_rejected():
    with pytest.raises(ValueError):
        apply_readout("nope", torch.zeros(3, dtype=DT), {})
    with pytest.raises(ValueError):
        param_shapes("nope")


# --- REVIEW-impl-WhereA B-4: the CBand12 denominator collapse ---------------

@pytest.mark.parametrize("sigma", [0.30, 0.20, 0.10])
def test_cband_logsumexp_matches_the_eps_formula_where_it_is_well_conditioned(sigma):
    """The two evaluations are the same formula; only the underflow regime differs.

    sigma=0.10 is the boundary: measured, the two forms start to separate just
    below it (sigma=0.05 already differs by 4.3e-3), so this case is what will
    catch a future change to CBAND_SIG_LO/HI (REVIEW-impl-WhereA N-26).
    """
    z = torch.linspace(-3, 3, 241, dtype=DT)
    g = torch.Generator().manual_seed(11)
    sig_raw0 = inv_bounded_sigmoid(sigma, CBAND_SIG_LO, CBAND_SIG_HI)
    for _ in range(10):
        raw = {
            "sig_raw": torch.full((CBAND_M,), sig_raw0, dtype=DT),
            "o_raw": torch.randn(CBAND_M, generator=g, dtype=DT),
            "c_raw": torch.randn(CBAND_M, generator=g, dtype=DT),
        }
        a = apply_readout("cband12", z, raw, normalization="eps")
        b = apply_readout("cband12", z, raw, normalization="logsumexp")
        assert torch.allclose(a, b, atol=1e-9), (sigma, float((a - b).abs().max()))


def test_the_two_forms_separate_below_the_boundary():
    """Pins the direction of the divergence: it is the eps form that collapses,
    and it starts doing so around sigma ~ 0.10.  If this ever stops holding, the
    equivalence test above has become vacuous."""
    z = torch.linspace(-3, 3, 241, dtype=DT)
    raw = {
        "sig_raw": torch.full((CBAND_M,), inv_bounded_sigmoid(0.05, CBAND_SIG_LO, CBAND_SIG_HI), dtype=DT),
        "o_raw": torch.zeros(CBAND_M, dtype=DT),
        "c_raw": torch.full((CBAND_M,), 4.0, dtype=DT),
    }
    a = apply_readout("cband12", z, raw, normalization="eps")
    b = apply_readout("cband12", z, raw, normalization="logsumexp")
    assert float((a - b).abs().max()) > 1e-3
    assert float(a.min()) < float(b.min()), "the eps form is the one that drops"


def test_eps_form_collapses_to_zero_at_the_sigma_lower_bound():
    """Pins the failure the review measured, so a revert cannot go unnoticed:
    with sigma at 0.025 and centres 6/11 apart, `sum g_i` underflows below
    CBAND_EPS between two centres and past the end centres, and every c_i=0.98
    still reads out as exactly 0."""
    raw = {
        "sig_raw": torch.full((CBAND_M,), inv_bounded_sigmoid(CBAND_SIG_LO, CBAND_SIG_LO, CBAND_SIG_HI), dtype=DT),
        "o_raw": torch.full((CBAND_M,), 4.0, dtype=DT),      # o ~ 0.982
        "c_raw": torch.full((CBAND_M,), 4.0, dtype=DT),      # c ~ 0.982
    }
    grid = cband_centres(dtype=DT)
    midpoint = float((grid[5] + grid[6]) / 2)                 # between two centres
    probes = torch.tensor([midpoint, float(grid[5]), 3.0, 3.3, 4.0], dtype=DT)

    legacy = apply_readout("cband12", probes, raw, normalization="eps")
    assert float(legacy[0]) == pytest.approx(0.0, abs=1e-12), "midpoint should collapse"
    assert float(legacy[3]) == pytest.approx(0.0, abs=1e-12), "z=3.3 should collapse"
    assert float(legacy[4]) == pytest.approx(0.0, abs=1e-12), "z=4.0 should collapse"
    assert float(legacy[1]) == pytest.approx(0.982014, abs=1e-5), "on-centre is fine"

    fixed = apply_readout("cband12", probes, raw, normalization="logsumexp")
    assert bool((fixed > 0.97).all()), f"logsumexp must not collapse: {fixed.tolist()}"


def test_logsumexp_is_the_default():
    from q3vl.where.config import CBAND_NORMALIZATION
    assert CBAND_NORMALIZATION == "logsumexp"
    raw = {
        "sig_raw": torch.full((CBAND_M,), inv_bounded_sigmoid(CBAND_SIG_LO, CBAND_SIG_LO, CBAND_SIG_HI), dtype=DT),
        "o_raw": torch.full((CBAND_M,), 4.0, dtype=DT),
        "c_raw": torch.full((CBAND_M,), 4.0, dtype=DT),
    }
    assert float(apply_readout("cband12", torch.tensor([3.3], dtype=DT), raw)) > 0.97


def test_logsumexp_survives_underflowed_opacity():
    """sigmoid(-800) is exactly 0.0 in float64; log(0) would poison the softmax."""
    raw = {"sig_raw": torch.zeros(CBAND_M, dtype=DT),
           "o_raw": torch.full((CBAND_M,), -800.0, dtype=DT),
           "c_raw": torch.zeros(CBAND_M, dtype=DT)}
    m = apply_readout("cband12", torch.linspace(-3, 3, 25, dtype=DT), raw)
    assert bool(torch.isfinite(m).all())
    assert bool(((m >= 0) & (m <= 1)).all())


def test_logsumexp_keeps_the_mask_in_the_unit_interval():
    g = torch.Generator().manual_seed(5)
    z = torch.linspace(-6, 6, 401, dtype=DT)
    for _ in range(30):
        raw = {k: torch.randn(CBAND_M, generator=g, dtype=DT)
               for k in ("sig_raw", "o_raw", "c_raw")}
        m = apply_readout("cband12", z, raw)
        assert float(m.min()) >= -1e-12 and float(m.max()) <= 1 + 1e-12


def test_mirror_still_exact_under_logsumexp():
    g = torch.Generator().manual_seed(17)
    z = torch.linspace(-4, 4, 81, dtype=DT)
    for _ in range(10):
        raw = {k: torch.randn(CBAND_M, generator=g, dtype=DT)
               for k in ("sig_raw", "o_raw", "c_raw")}
        a = apply_readout("cband12", z, raw)
        b = apply_readout("cband12", -z, mirror_params("cband12", raw))
        assert torch.allclose(a, b, atol=1e-12)
