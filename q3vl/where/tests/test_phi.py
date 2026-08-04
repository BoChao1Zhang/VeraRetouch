"""Protocol 4.2 -- geo5 / range / residualised semantics."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from q3vl.where.config import PHI_DIR_DIM, RESID_BLOCK_DIM, SEM_DIM, PhiConfig
from q3vl.where.phi import (
    build_phi_dir, design_block, geo5_grid, legendre_p2, norm_coords,
    range_channels, residualize, semantic_slice, standardize,
)

DT = torch.float64


def test_legendre_p2():
    t = torch.linspace(-1, 1, 11, dtype=DT)
    assert torch.allclose(legendre_p2(t), 0.5 * (3 * t * t - 1), atol=1e-14)
    assert abs(float(legendre_p2(torch.tensor(1.0, dtype=DT))) - 1.0) < 1e-14


def test_coords_respect_the_true_aspect_ratio():
    """Protocol 4.1: coordinates follow each image's real aspect ratio; the
    image is never squashed to a square."""
    gh, gw = 32, 56                     # a 512x896 spec-5 image
    X, Y = norm_coords(gh, gw, dtype=DT)
    assert pytest.approx(float(Y.max()), abs=1e-9) == 1.0 - 1.0 / gh
    ar = gw / gh
    assert float(X.max()) == pytest.approx(ar - ar / gw, abs=1e-9)
    # square images stay square
    Xs, Ys = norm_coords(20, 20, dtype=DT)
    assert float(Xs.max()) == pytest.approx(float(Ys.max()), abs=1e-12)


def test_geo5_columns_and_order():
    gh, gw = 6, 8
    g = geo5_grid(gh, gw, dtype=DT)
    assert g.shape == (gh * gw, 5)
    X, Y = norm_coords(gh, gw, dtype=DT)
    x, y = X.reshape(-1), Y.reshape(-1)
    assert torch.allclose(g[:, 0], x, atol=1e-14)
    assert torch.allclose(g[:, 1], y, atol=1e-14)
    assert torch.allclose(g[:, 2], legendre_p2(x), atol=1e-14)
    assert torch.allclose(g[:, 3], legendre_p2(y), atol=1e-14)
    assert torch.allclose(g[:, 4], x * y, atol=1e-14)


def test_geo5_is_row_major():
    gh, gw = 4, 5
    g = geo5_grid(gh, gw, dtype=DT)
    x = g[:, 0].reshape(gh, gw)
    assert torch.allclose(x[0], x[1], atol=1e-14)      # x constant down a column
    assert float(x[0, 0]) < float(x[0, -1])


def test_range_channels_on_known_colours():
    img = torch.zeros(3, 1, 4, dtype=DT)
    img[:, 0, 0] = torch.tensor([1.0, 1.0, 1.0], dtype=DT)   # white
    img[:, 0, 1] = torch.tensor([0.0, 0.0, 0.0], dtype=DT)   # black
    img[:, 0, 2] = torch.tensor([1.0, 0.0, 0.0], dtype=DT)   # pure red
    img[:, 0, 3] = torch.tensor([0.5, 0.5, 0.5], dtype=DT)   # grey
    L, S = range_channels(img)
    assert float(L[0]) == pytest.approx(1.0)
    assert float(L[1]) == pytest.approx(0.0)
    assert float(L[2]) == pytest.approx(0.2126)
    assert float(S[0]) == pytest.approx(0.0)     # white is unsaturated
    assert float(S[2]) == pytest.approx(1.0)     # pure red is fully saturated
    assert float(S[3]) == pytest.approx(0.0)


def test_standardize():
    v = torch.randn(100, 3, dtype=DT) * 5 + 2
    z = standardize(v, 1e-12)
    assert torch.allclose(z.mean(0), torch.zeros(3, dtype=DT), atol=1e-10)
    assert torch.allclose(z.std(0, unbiased=False), torch.ones(3, dtype=DT), atol=1e-9)


def test_residualize_removes_the_design_block():
    g = torch.Generator().manual_seed(0)
    A = torch.randn(300, RESID_BLOCK_DIM, generator=g, dtype=DT)
    A[:, 0] = 1.0
    E = A @ torch.randn(RESID_BLOCK_DIM, 4, generator=g, dtype=DT) \
        + 0.1 * torch.randn(300, 4, generator=g, dtype=DT)
    Er, diag = residualize(E, A, 1e-6)
    assert diag["resid_corr_before"] > 0.5
    assert diag["resid_corr_after"] < 1e-6
    assert diag["design_gram_cond"] > 0


def _fake_inputs(gh=8, gw=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    sem = torch.randn(gh * gw, SEM_DIM, generator=g, dtype=DT)
    img = torch.rand(3, gh, gw, generator=g, dtype=DT)
    return sem, img, gh, gw


def test_phi_dir_shape_and_layout():
    sem, img, gh, gw = _fake_inputs()
    parts = build_phi_dir(sem, img, gh, gw)
    assert parts.phi_dir.shape == (gh * gw, PHI_DIR_DIM)
    assert len(parts.names) == PHI_DIR_DIM
    assert parts.names[:5] == ("x", "y", "P2x", "P2y", "xy")
    assert parts.names[5:7] == ("L", "S")
    assert parts.names[7] == "e1" and parts.names[-1] == f"e{SEM_DIM}"
    # the protocol's column order, verified against the parts themselves
    assert torch.allclose(parts.phi_dir[:, :5], parts.geo5, atol=1e-14)
    assert torch.allclose(parts.phi_dir[:, 5], parts.L, atol=1e-14)
    assert torch.allclose(parts.phi_dir[:, 6], parts.S, atol=1e-14)
    assert torch.allclose(parts.phi_dir[:, semantic_slice()], parts.semantic, atol=1e-14)


def test_semantics_are_orthogonal_to_geometry_and_range():
    """The whole point of the residualisation: the projector cannot smuggle
    coordinates, lightness or saturation into the semantic block."""
    gh, gw = 10, 14
    g = torch.Generator().manual_seed(1)
    img = torch.rand(3, gh, gw, generator=g, dtype=DT)
    L, S = range_channels(img)
    L = standardize(L.unsqueeze(1), 1e-6).squeeze(1)
    S = standardize(S.unsqueeze(1), 1e-6).squeeze(1)
    A = design_block(geo5_grid(gh, gw, dtype=DT), L, S)
    # a projector that literally outputs x, y, L and S
    sem = torch.zeros(gh * gw, SEM_DIM, dtype=DT)
    sem[:, 0] = A[:, 1]
    sem[:, 1] = A[:, 2]
    sem[:, 2] = L
    sem[:, 3] = S
    sem[:, 4:] = torch.randn(gh * gw, SEM_DIM - 4, generator=g, dtype=DT)
    parts = build_phi_dir(sem, img, gh, gw)
    # the four cheating channels are annihilated exactly, not amplified into
    # unit-variance round-off
    assert parts.diag["n_dead_semantic"] == 4
    assert float(parts.semantic[:, :4].abs().max()) == 0.0
    # the surviving channels really are orthogonal to the design block
    from q3vl.where.phi import _max_abs_corr
    assert _max_abs_corr(A, parts.semantic[:, 4:], 1e-6) < 1e-6


def test_semantics_are_standardised():
    sem, img, gh, gw = _fake_inputs(seed=2)
    parts = build_phi_dir(sem, img, gh, gw)
    m = parts.semantic.mean(0)
    s = parts.semantic.std(0, unbiased=False)
    assert parts.diag["n_dead_semantic"] == 0
    assert float(m.abs().max()) < 1e-9
    # exactly 1 - eps/(std+eps): standardize divides by (std + STD_EPS)
    assert float((s - 1).abs().max()) < 1e-5


def test_gradient_reaches_the_semantic_block():
    """This is the *only* path by which the shared projector B is trained."""
    sem, img, gh, gw = _fake_inputs(seed=3)
    sem = sem.clone().requires_grad_(True)
    parts = build_phi_dir(sem, img, gh, gw)
    parts.phi_dir.sum().backward()
    assert sem.grad is not None
    assert float(sem.grad.abs().sum()) > 0


def test_constant_image_does_not_explode():
    gh, gw = 6, 6
    img = torch.full((3, gh, gw), 0.42, dtype=DT)
    g = torch.Generator().manual_seed(4)
    sem = torch.randn(gh * gw, SEM_DIM, generator=g, dtype=DT)
    parts = build_phi_dir(sem, img, gh, gw)
    assert torch.isfinite(parts.phi_dir).all()


def test_shape_mismatches_are_rejected():
    sem, img, gh, gw = _fake_inputs()
    with pytest.raises(ValueError):
        build_phi_dir(sem[:, :10], img, gh, gw)
    with pytest.raises(ValueError):
        build_phi_dir(sem, img, gh + 1, gw)
    with pytest.raises(ValueError):
        build_phi_dir(sem, img[:, :3], gh, gw)


def test_residualisation_can_be_disabled_for_ablation_only():
    sem, img, gh, gw = _fake_inputs(seed=5)
    parts = build_phi_dir(sem, img, gh, gw, PhiConfig(residualize=False))
    assert parts.diag["resid_corr_after"] is None
    assert parts.phi_dir.shape[1] == PHI_DIR_DIM


def test_float32_path_matches_float64_closely():
    sem, img, gh, gw = _fake_inputs(seed=6)
    p64 = build_phi_dir(sem, img, gh, gw)
    p32 = build_phi_dir(sem.float(), img.float(), gh, gw)
    assert np.allclose(p32.phi_dir.double().numpy(), p64.phi_dir.numpy(), atol=1e-4)
