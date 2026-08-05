"""Protocol 7.1 -- the frozen functional encoder must be exactly reproducible."""

from __future__ import annotations

import math

import pytest
import torch

from q3vl.what.config import SRHT_SEED, ZGT_DIM, ZGT_GRID
from q3vl.what.srht import SRHT, default_srht, encode_z_gt, fwht, identity_grid, u_of_table


def test_fwht_matches_explicit_hadamard():
    n = 64
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    x = torch.randn(5, n, dtype=torch.float64)
    assert torch.allclose(fwht(x), x @ h.double().T, atol=1e-9)


def test_fwht_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        fwht(torch.zeros(3, 7))


def test_identity_grid_and_u():
    g = identity_grid()
    assert g.shape == (ZGT_GRID ** 3, 3)
    assert float(g.min()) == 0.0 and float(g.max()) == 1.0
    # u of the identity LUT is exactly zero
    assert float(u_of_table(g).abs().max()) == 0.0
    assert u_of_table(g.unsqueeze(0).expand(4, -1, -1)).shape == (4, ZGT_GRID ** 3 * 3)


def test_projection_is_deterministic_across_instances():
    a, b = default_srht(), SRHT(ZGT_GRID ** 3 * 3, ZGT_DIM, seed=SRHT_SEED)
    v = torch.randn(3, a.n_in)
    assert torch.equal(a(v), b(v))
    assert a.digest() == b.digest()
    assert SRHT(a.n_in, seed=SRHT_SEED + 1).digest() != a.digest()


def test_isometry_in_expectation_and_distance_preservation():
    """SRHT is a JL transform: norms are preserved in expectation, distances
    within the usual O(sqrt(log n / k)) factor.  Measured, not asserted."""
    s = default_srht()
    v = torch.randn(128, s.n_in) * 0.05
    ratio_norm = (s(v).norm(dim=-1) / v.norm(dim=-1))
    assert abs(float(ratio_norm.mean()) - 1.0) < 0.02
    iu = torch.triu_indices(64, 64, offset=1)
    d0 = (v[:64][iu[0]] - v[:64][iu[1]]).norm(dim=-1)
    d1 = (s(v[:64])[iu[0]] - s(v[:64])[iu[1]]).norm(dim=-1)
    r = d1 / d0
    assert 0.75 < float(r.min()) and float(r.max()) < 1.25
    assert abs(float(r.mean()) - 1.0) < 0.03


def test_z_gt_is_unit_norm_and_centre_shifts_it():
    vals = torch.rand(4, ZGT_GRID ** 3, 3)
    z = encode_z_gt(vals)
    assert z.shape == (4, ZGT_DIM)
    assert torch.allclose(z.norm(dim=-1), torch.ones(4), atol=1e-5)
    centre = u_of_table(vals).mean(0)
    z_c = encode_z_gt(vals, centre)
    assert not torch.allclose(z, z_c)
    assert torch.allclose(z_c.norm(dim=-1), torch.ones(4), atol=1e-5)


def test_scale_is_the_srht_normalisation():
    s = default_srht()
    assert math.isclose(s.scale, math.sqrt(s.pad / s.k), rel_tol=1e-12)
    assert s.rows.numel() == ZGT_DIM
    assert int(s.rows.unique().numel()) == ZGT_DIM      # sampled without replacement
