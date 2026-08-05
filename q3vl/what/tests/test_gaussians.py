"""Protocol 7.6 -- parameter domains, SPD, the identity at init, and the 2x trap."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from q3vl.what.config import ANCHOR_GRID, FG_TOTAL_PER_SAMPLE, LutConfig, N_SLOTS, SB_TOTAL_PER_SAMPLE
from q3vl.what.gaussians import (
    GeometryBank,
    PRIM_LAYOUT_FG,
    PRIM_LAYOUT_SB,
    anchor_points,
    bake,
    decode_global,
    decode_primitives,
    gaussian_log_density,
    identity_raw,
    mixture_weights,
    parameter_report,
    render,
    softplus_inv,
)
from q3vl.what.lut import lattice_points

CFG = LutConfig()


def _fg_params(z_prim, z_glob, cfg=CFG):
    p = decode_primitives(z_prim, cfg, PRIM_LAYOUT_FG, anchors=anchor_points(cfg.anchor_grid))
    p.update(decode_global(z_glob, cfg))
    return p


# --- anchors and layouts ----------------------------------------------------

def test_anchor_grid_is_4x4x3_cell_centres():
    a = anchor_points()
    assert a.shape == (N_SLOTS, 3)
    assert ANCHOR_GRID == (4, 4, 3)
    assert float(a.min()) > 0.0 and float(a.max()) < 1.0
    assert sorted({round(float(v), 6) for v in a[:, 0]}) == [0.125, 0.375, 0.625, 0.875]
    assert len({round(float(v), 6) for v in a[:, 2]}) == 3


def test_per_sample_parameter_counts_match_the_protocol():
    assert sum(b - a for a, b in PRIM_LAYOUT_FG.values()) == 23
    assert sum(b - a for a, b in PRIM_LAYOUT_SB.values()) == 14
    assert N_SLOTS * 23 + 12 == FG_TOTAL_PER_SAMPLE == 1116
    assert N_SLOTS * 14 + 12 == SB_TOTAL_PER_SAMPLE


# --- constrained domains ----------------------------------------------------

def test_mu_stays_in_the_cube_for_extreme_raw_values():
    # saturating raw values: the sigmoid can reach the closed cube in float32,
    # which is still "confined to the RGB cube" (protocol 7.6)
    z = torch.full((2, N_SLOTS, 23), 1e4)
    z[1] = -1e4
    p = _fg_params(z, torch.zeros(2, 12))
    assert bool(((p["mu"] >= 0.0) & (p["mu"] <= 1.0)).all())
    # at any magnitude short of float32 sigmoid saturation (~|z| > 9) it is
    # strictly interior
    z = torch.full((2, N_SLOTS, 23), 8.0)
    z[1] = -8.0
    p = _fg_params(z, torch.zeros(2, 12))
    assert bool(((p["mu"] > 0.0) & (p["mu"] < 1.0)).all())


def _sigma_at(raw: float, cfg=CFG) -> float:
    z = torch.full((1, N_SLOTS, 23), raw)
    return float(_fg_params(z, torch.zeros(1, 12), cfg)["sigma"][0, 0, 0])


def test_sigma_is_positive_has_a_floor_and_grows_linearly_not_exponentially():
    assert _sigma_at(-1e4) >= CFG.sigma_lo - 1e-9
    assert _sigma_at(-1e4) > 0.0
    # at raw = 0 the diagonal is exactly sigma_init
    assert abs(_sigma_at(0.0) - CFG.sigma_init) < 1e-5
    # softplus is asymptotically the identity: a fixed raw increment produces the
    # same absolute increment.  A bare exp would produce a fixed *ratio*, which is
    # what the campaign red line forbids.
    d1 = _sigma_at(20.0) - _sigma_at(10.0)
    d2 = _sigma_at(30.0) - _sigma_at(20.0)
    assert abs(d1 - 10.0) < 1e-2 and abs(d2 - 10.0) < 1e-2
    assert abs(d2 - d1) < 1e-2                      # linear, not multiplicative


def test_bounded_sigmoid_alternative_stays_in_its_band():
    cfg = replace(CFG, sigma_param="bounded_sigmoid")
    lo, hi = cfg.sigma_lo, cfg.sigma_lo + cfg.sigma_span
    for v in (-1e4, 0.0, 1e4):
        s = _fg_params(torch.full((1, N_SLOTS, 23), v), torch.zeros(1, 12), cfg)["sigma"]
        assert lo - 1e-9 <= float(s.min()) and float(s.max()) <= hi + 1e-9
    s0 = _fg_params(torch.zeros(1, N_SLOTS, 23), torch.zeros(1, 12), cfg)["sigma"]
    assert abs(float(s0[0, 0, 0]) - cfg.sigma_init) < 1e-5


def test_softplus_inv_round_trip():
    for y in (1e-4, 0.18, 3.0, 25.0):
        assert abs(float(torch.nn.functional.softplus(torch.tensor(softplus_inv(y)))) - y) < 1e-4


def test_covariance_is_spd():
    torch.manual_seed(0)
    p = _fg_params(torch.randn(2, N_SLOTS, 23), torch.zeros(2, 12))
    s, o = p["sigma"], p["off"]
    L = torch.zeros(2, N_SLOTS, 3, 3)
    L[..., 0, 0], L[..., 1, 1], L[..., 2, 2] = s[..., 0], s[..., 1], s[..., 2]
    L[..., 1, 0], L[..., 2, 0], L[..., 2, 1] = o[..., 0], o[..., 1], o[..., 2]
    sigma = L @ L.transpose(-1, -2)
    eig = torch.linalg.eigvalsh(sigma)
    assert float(eig.min()) > 0.0


def test_log_density_matches_a_hand_computed_gaussian():
    """One isotropic Gaussian, closed form, no forward substitution involved."""
    mu = torch.tensor([[[0.3, 0.4, 0.5]]])
    sig = torch.full((1, 1, 3), 0.2)
    off = torch.zeros(1, 1, 3)
    x = torch.tensor([[[0.3, 0.4, 0.5], [0.5, 0.4, 0.5]]])
    got = gaussian_log_density(x, mu, sig, off)
    want_peak = -1.5 * math.log(2 * math.pi) - 3 * math.log(0.2)
    assert abs(float(got[0, 0, 0]) - want_peak) < 1e-5
    assert abs(float(got[0, 0, 1]) - (want_peak - 0.5 * (0.2 / 0.2) ** 2)) < 1e-5


def test_weights_are_normalised():
    torch.manual_seed(1)
    p = _fg_params(torch.randn(2, N_SLOTS, 23) * 0.3, torch.zeros(2, 12))
    q = mixture_weights(p, torch.rand(2, 256, 3), CFG.mixture_eps)
    s = q.sum(1)
    assert float(s.max()) <= 1.0 + 1e-6
    assert float(s.min()) > 0.99


# --- the identity, and the 2x trap ------------------------------------------

def test_zero_init_is_exactly_the_identity():
    z_prim, z_glob = identity_raw(N_SLOTS, PRIM_LAYOUT_FG)
    p = _fg_params(z_prim, z_glob)
    x = lattice_points(9).unsqueeze(0)
    assert float((render(p, x, CFG) - x).abs().max()) < 1e-4


def test_the_literal_protocol_formula_gives_2x():
    """The measurement behind ``GLOBAL_AFFINE_MODE``: identity-centred *global*
    plus identity-centred locals is f(x) = 2x, the campaign's named red line."""
    cfg = replace(CFG, global_affine_mode="identity_centered", clamp_output=False)
    z_prim, z_glob = identity_raw(N_SLOTS, PRIM_LAYOUT_FG)
    p = decode_primitives(z_prim, cfg, PRIM_LAYOUT_FG, anchors=anchor_points(cfg.anchor_grid))
    p.update(decode_global(z_glob, cfg))
    x = lattice_points(5).unsqueeze(0)
    y = render(p, x, cfg)
    assert float((y - 2 * x).abs().max()) < 1e-3
    with pytest.raises(ValueError):
        decode_global(z_glob, replace(CFG, global_affine_mode="nonsense"))


def test_clamp_keeps_the_output_in_the_cube():
    torch.manual_seed(2)
    p = _fg_params(torch.randn(1, N_SLOTS, 23) * 3, torch.randn(1, 12) * 5)
    y = render(p, torch.rand(1, 512, 3), CFG)
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0


# --- gradients and the shared bank ------------------------------------------

def test_gradient_reaches_every_raw_group():
    torch.manual_seed(3)
    z_prim = torch.randn(2, N_SLOTS, 23, requires_grad=True)
    z_glob = torch.randn(2, 12, requires_grad=True)
    render(_fg_params(z_prim, z_glob), torch.rand(2, 128, 3), CFG).sum().backward()
    assert torch.isfinite(z_prim.grad).all() and torch.isfinite(z_glob.grad).all()
    for name, (a, b) in PRIM_LAYOUT_FG.items():
        assert float(z_prim.grad[..., a:b].abs().sum()) > 0.0, name


def test_geometry_bank_starts_at_the_anchors_and_is_trainable():
    bank = GeometryBank(CFG)
    g = bank(3)
    assert g["mu"].shape == (3, N_SLOTS, 3)
    assert torch.allclose(g["mu"][0], anchor_points(), atol=1e-5)
    assert bank.raw.requires_grad and bank.raw.numel() == N_SLOTS * 9
    g["mu"].sum().backward()
    assert float(bank.raw.grad.abs().sum()) > 0.0


def test_sb_decode_uses_the_shared_geometry():
    bank = GeometryBank(CFG)
    geom = bank(2)
    p = decode_primitives(torch.zeros(2, N_SLOTS, 14), CFG, PRIM_LAYOUT_SB, geometry=geom)
    assert torch.allclose(p["mu"], geom["mu"])
    # the same geometry for every sample in the batch -- that is the point of SB48
    assert torch.allclose(p["mu"][0], p["mu"][1])
    with pytest.raises(ValueError):
        decode_primitives(torch.zeros(2, N_SLOTS, 14), CFG, PRIM_LAYOUT_SB)


def test_bake_shape_and_lattice_consistency():
    torch.manual_seed(4)
    p = _fg_params(torch.randn(2, N_SLOTS, 23) * 0.2, torch.zeros(2, 12))
    cube = bake(p, CFG, 17)
    assert cube.shape == (2, 17, 17, 17, 3)
    pts = lattice_points(17).unsqueeze(0).expand(2, -1, -1)
    assert torch.allclose(cube.reshape(2, -1, 3), render(p, pts, CFG), atol=1e-5)


def test_parameter_report_flags_the_domains():
    p = _fg_params(torch.randn(2, N_SLOTS, 23), torch.zeros(2, 12))
    rep = parameter_report(p)
    assert rep["mu_in_cube"] and rep["sigma_positive"] and rep["spd"] and rep["all_finite"]
