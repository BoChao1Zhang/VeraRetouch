"""Protocol 4.2 -- ``s_low``, the unit direction, and the sign canonicalisation."""

from __future__ import annotations

import pytest
import torch

from q3vl.where.basis import (
    Latent, alpha_of, canonicalize, is_canonical, mask_from_latent, s_low,
    sign_index, w_dir_of,
)
from q3vl.where.config import CBAND_M, PHI_DIR_DIM, S_SCALE
from q3vl.where.readout import param_shapes

DT = torch.float64


def _rand_latent(readout: str, seed: int = 0, negate: bool = False) -> Latent:
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(PHI_DIR_DIM, generator=g, dtype=DT)
    if negate:
        # force the largest-|.| coefficient negative
        i = int(torch.argmax(w.abs()))
        if w[i] > 0:
            w = -w
    rho = {k: (torch.randn(s, generator=g, dtype=DT) if s else
               torch.randn((), generator=g, dtype=DT))
           for k, s in param_shapes(readout).items()}
    return Latent(readout, torch.randn((), generator=g, dtype=DT),
                  torch.randn((), generator=g, dtype=DT), w, rho)


def _phi(seed: int = 1, p: int = 200) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(p, PHI_DIR_DIM, generator=g, dtype=DT)


def test_s_low_matches_the_protocol_formula():
    lat = _rand_latent("band", 0)
    phi = _phi()
    want = S_SCALE * torch.tanh(
        (lat.w0 + torch.nn.functional.softplus(lat.alpha_raw)
         * (phi @ (lat.w_raw / lat.w_raw.norm()))) / S_SCALE
    )
    assert torch.allclose(s_low(phi, lat), want, atol=1e-14)


def test_s_low_is_bounded_and_alpha_positive():
    for seed in range(5):
        lat = _rand_latent("cband12", seed)
        s = s_low(_phi(seed), lat)
        assert float(s.abs().max()) < S_SCALE
        assert float(lat.alpha) > 0.0


def test_w_dir_is_unit_norm():
    for seed in range(5):
        lat = _rand_latent("band", seed)
        assert abs(float(lat.w_dir.norm()) - 1.0) < 1e-12


def test_alpha_softplus_stays_positive_for_extreme_raw():
    for v in (-1e6, -100.0, 0.0, 100.0):
        assert float(alpha_of(torch.tensor(v, dtype=DT))) >= 0.0
    assert float(alpha_of(torch.tensor(-1e3, dtype=DT))) < 1e-100


@pytest.mark.parametrize("readout", ["band", "cband12"])
def test_canonicalize_enforces_the_sign_rule(readout):
    lat = _rand_latent(readout, seed=2, negate=True)
    assert not is_canonical(lat)
    can = canonicalize(lat)
    assert is_canonical(can)
    wd = can.w_dir
    assert float(wd[sign_index(wd)]) > 0, "largest-|.| coefficient must be positive"


@pytest.mark.parametrize("readout", ["band", "cband12"])
def test_canonicalize_leaves_the_mask_untouched(readout):
    """This is the whole point: +-w is a non-identifiability, not a degree of
    freedom.  Removing it must not move a single pixel."""
    phi = _phi(5, p=512)
    for seed in range(12):
        lat = _rand_latent(readout, seed=seed, negate=(seed % 2 == 0))
        m0, s0 = mask_from_latent(phi, lat)
        can = canonicalize(lat)
        m1, s1 = mask_from_latent(phi, can)
        assert torch.allclose(m0, m1, atol=1e-12), f"{readout} seed={seed}"
        if not is_canonical(lat):
            assert torch.allclose(s1, -s0, atol=1e-12), "the flip must negate s exactly"


@pytest.mark.parametrize("readout", ["band", "cband12"])
def test_canonicalize_is_idempotent(readout):
    lat = _rand_latent(readout, seed=4, negate=True)
    once = canonicalize(lat)
    twice = canonicalize(once)
    assert torch.allclose(once.w_raw, twice.w_raw, atol=1e-14)
    assert torch.allclose(once.w0, twice.w0, atol=1e-14)


def test_sign_index_picks_largest_absolute_coefficient():
    w = torch.zeros(PHI_DIR_DIM, dtype=DT)
    w[7] = -0.9
    w[13] = 0.5
    assert sign_index(w_dir_of(w)) == 7


def test_zero_direction_is_not_silently_flipped():
    lat = Latent("band", torch.tensor(0.5, dtype=DT), torch.tensor(0.0, dtype=DT),
                 torch.zeros(PHI_DIR_DIM, dtype=DT),
                 {k: (torch.zeros(s, dtype=DT) if s else torch.zeros((), dtype=DT))
                  for k, s in param_shapes("band").items()})
    can = canonicalize(lat)
    assert torch.allclose(can.w_raw, lat.w_raw)
    assert float(can.w0) == 0.5


def test_latent_roundtrips_through_json_dict():
    lat = canonicalize(_rand_latent("cband12", 9))
    d = lat.to_dict()
    assert d["canonical"] is True
    assert len(d["w_dir"]) == PHI_DIR_DIM
    assert len(d["rho"]["sigma"]) == CBAND_M
    back = Latent.from_dict(d)
    phi = _phi(3)
    assert torch.allclose(mask_from_latent(phi, lat)[0], mask_from_latent(phi, back)[0], atol=1e-12)


def test_wrong_dimension_is_rejected():
    with pytest.raises(ValueError):
        Latent("band", torch.zeros((), dtype=DT), torch.zeros((), dtype=DT),
               torch.zeros(70, dtype=DT),
               {k: (torch.zeros(s, dtype=DT) if s else torch.zeros((), dtype=DT))
                for k, s in param_shapes("band").items()})
    with pytest.raises(ValueError):
        Latent("band", torch.zeros((), dtype=DT), torch.zeros((), dtype=DT),
               torch.zeros(PHI_DIR_DIM, dtype=DT), {"mu": torch.zeros((), dtype=DT)})
