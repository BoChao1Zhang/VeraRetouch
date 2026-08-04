"""Q_where: learnable 2D canvas positions and the true-aspect-ratio grid."""

from __future__ import annotations

import torch

from q3vl.where.phi import geo5_grid
from q3vl.whereb.qwhere import (
    MetaCanvasQueryBank,
    PositionEncoder,
    fourier_pos_encode,
    fpre_grid_positions,
)


def test_initial_canvas_positions_are_a_uniform_grid_in_the_unit_square():
    p = MetaCanvasQueryBank.initial_positions(4)
    assert p.shape == (16, 2)
    assert torch.allclose(p[:, 0][:4], torch.tensor([-0.75, -0.25, 0.25, 0.75]))
    assert float(p.min()) == -0.75 and float(p.max()) == 0.75
    # row-major: y is constant inside a row
    assert torch.allclose(p[:4, 1], torch.full((4,), -0.75))


def test_positions_are_learnable_and_tokens_are_learnable():
    bank = MetaCanvasQueryBank(8, 32, seed=0)
    names = {n for n, _ in bank.named_parameters()}
    assert names == {"tokens", "positions"}
    assert bank.positions.requires_grad and bank.tokens.requires_grad
    assert bank.n_queries == 64


def test_query_bank_is_shared_across_the_batch():
    bank = MetaCanvasQueryBank(4, 16, seed=1)
    enc = PositionEncoder(16, 4, 8.0)
    q = bank(3, enc)
    assert q.shape == (3, 16, 16)
    assert torch.equal(q[0], q[2])


def test_position_gradient_flows_to_the_learnable_positions():
    bank = MetaCanvasQueryBank(4, 16, seed=1)
    enc = PositionEncoder(16, 4, 8.0)
    bank(2, enc).sum().backward()
    assert bank.positions.grad is not None
    assert torch.isfinite(bank.positions.grad).all()
    assert float(bank.positions.grad.abs().sum()) > 0


def test_fourier_features_are_sin_cos_pairs_and_bounded():
    xy = torch.tensor([[0.0, 0.0], [0.5, -0.25]])
    f = fourier_pos_encode(xy, n_bands=4, max_freq=8.0)
    assert f.shape == (2, 16)
    assert float(f.abs().max()) <= 1.0
    # at the origin every sin is 0 and every cos is 1
    assert torch.allclose(f[0, :8], torch.zeros(8), atol=1e-6)
    assert torch.allclose(f[0, 8:], torch.ones(8), atol=1e-6)


def test_distinct_positions_get_distinct_encodings():
    xy = torch.tensor([[0.1, 0.2], [0.1, 0.2001], [-0.7, 0.4]])
    f = fourier_pos_encode(xy, n_bands=16, max_freq=8.0)
    assert not torch.allclose(f[0], f[2])
    assert torch.allclose(f[0], f[1], atol=1e-2)      # nearby -> nearby


def test_fpre_positions_use_the_true_aspect_ratio_not_a_square():
    """Protocol 4.1: never squashed to 512x512.  Short side spans [-1, 1] and a
    grid cell is square, so the long side runs past 1 in proportion."""
    gh, gw = 32, 48
    pos = fpre_grid_positions(gh, gw)                 # a 2:3 image
    x, y = pos[:, 0], pos[:, 1]
    step = 2.0 / gh                                   # short side spans [-1, 1]
    assert abs(float(y.max()) - (1 - step / 2)) < 1e-6
    assert float(x.max()) > 1.0                       # long side goes past 1
    # square cells: identical spacing on both axes
    assert abs(float(x[1] - x[0]) - step) < 1e-6
    assert abs(float(y[gw] - y[0]) - step) < 1e-6
    # full extent (cell edges, not centres) reproduces the image aspect ratio
    ext_x = float(x.max() - x.min()) + step
    ext_y = float(y.max() - y.min()) + step
    assert abs(ext_x / ext_y - gw / gh) < 1e-6
    square = fpre_grid_positions(20, 20)
    assert abs(float(square[:, 0].max()) - float(square[:, 1].max())) < 1e-9


def test_fpre_positions_agree_with_the_where_a_geo5_block():
    """Where-B's position encoding and Where-A's geo5 must mean the same (x, y)."""
    pos = fpre_grid_positions(6, 9, dtype=torch.float64)
    geo = geo5_grid(6, 9, dtype=torch.float64)
    assert torch.allclose(pos[:, 0], geo[:, 0])
    assert torch.allclose(pos[:, 1], geo[:, 1])


def test_canvas_map_restores_the_2d_layout():
    bank = MetaCanvasQueryBank(4, 8, seed=0)
    t = torch.arange(2 * 16 * 8, dtype=torch.float32).reshape(2, 16, 8)
    m = bank.canvas_map(t)
    assert m.shape == (2, 4, 4, 8)
    assert torch.equal(m[0, 1, 2], t[0, 1 * 4 + 2])
