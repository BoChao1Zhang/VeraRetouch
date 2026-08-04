"""Protocol 5.1 -- ``Q_where``: a MetaCanvas query bank with learnable 2D positions.

    "``Q_where`` is a MetaCanvas query with learnable 2D positions.  It reads
     only: all ``<where>...</where>`` token hidden states (``H_where``), the
     merger-pre ``F_pre``, and a two-dimensional position encoding under the
     *true* aspect ratio.  It does not read ``<color>`` hidden states, and it
     does not read ``I_tar``."

Two different position encodings live here and they must share one coordinate
convention, otherwise a query's learned position means nothing relative to the
feature grid it attends to:

* the canvas queries carry a **learnable** ``(x, y)`` per query, initialised on
  a uniform ``n x n`` grid over ``[-1, 1]^2``;
* the ``F_pre`` tokens carry a **fixed** ``(x, y)`` from
  :func:`q3vl.where.phi.norm_coords` -- the short side spans ``[-1, 1]`` and the
  long side spans ``[-AR, +AR]``, i.e. a grid cell is square in feature space
  and the image is never squashed to 512x512 (protocol 4.1).

Both go through the same Fourier feature map, so "query at (0.3, -0.5)" and
"feature token at (0.3, -0.5)" are the same place.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from q3vl.where.phi import norm_coords

from .config import ConnectorConfig

__all__ = ["fourier_pos_encode", "PositionEncoder", "MetaCanvasQueryBank",
           "fpre_grid_positions"]


def fourier_pos_encode(xy: torch.Tensor, n_bands: int, max_freq: float) -> torch.Tensor:
    """``(..., 2) -> (..., 4 * n_bands)`` sin/cos features.

    Frequencies are log-spaced in ``[1, max_freq]`` cycles per unit; with the
    short side spanning ``[-1, 1]`` the highest band resolves ~1/16 of the short
    side, which is exactly the ``F_pre`` cell size at 512/16 = 32 cells.
    """
    if xy.shape[-1] != 2:
        raise ValueError(f"expected (..., 2) coordinates, got {tuple(xy.shape)}")
    freqs = torch.logspace(
        0.0, math.log10(max_freq), n_bands, device=xy.device, dtype=xy.dtype
    )
    ang = xy.unsqueeze(-1) * freqs * math.pi          # (..., 2, n_bands)
    ang = ang.flatten(-2)                             # (..., 2 * n_bands)
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class PositionEncoder(nn.Module):
    """Fourier features -> ``dim``.  One instance per stream, shared by the
    canvas queries and the ``F_pre`` tokens so both live in the same space."""

    def __init__(self, dim: int, n_bands: int, max_freq: float):
        super().__init__()
        self.n_bands = n_bands
        self.max_freq = max_freq
        self.proj = nn.Linear(4 * n_bands, dim)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.proj(fourier_pos_encode(xy, self.n_bands, self.max_freq))


class MetaCanvasQueryBank(nn.Module):
    """``n x n`` learnable query tokens, each with its own learnable ``(x, y)``."""

    def __init__(self, canvas: int, dim: int, seed: int = 0):
        super().__init__()
        self.canvas = canvas
        self.dim = dim
        self.n_queries = canvas * canvas
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.tokens = nn.Parameter(
            torch.randn(self.n_queries, dim, generator=g) * (dim ** -0.5)
        )
        self.positions = nn.Parameter(self.initial_positions(canvas))

    @staticmethod
    def initial_positions(canvas: int) -> torch.Tensor:
        """Cell centres of a uniform ``canvas x canvas`` grid over [-1, 1]^2."""
        t = (torch.arange(canvas, dtype=torch.float32) + 0.5) / canvas * 2.0 - 1.0
        y, x = torch.meshgrid(t, t, indexing="ij")
        return torch.stack([x.reshape(-1), y.reshape(-1)], dim=1)

    def forward(self, batch: int, pos_encoder: PositionEncoder) -> torch.Tensor:
        """``(batch, n_queries, dim)``."""
        q = self.tokens + pos_encoder(self.positions.to(self.tokens.dtype))
        return q.unsqueeze(0).expand(batch, -1, -1)

    def canvas_map(self, tokens: torch.Tensor) -> torch.Tensor:
        """``(B, n*n, C) -> (B, n, n, C)`` for the protocol 13 canvas figures."""
        b, n, c = tokens.shape
        if n != self.n_queries:
            raise ValueError(f"{n} tokens != canvas {self.canvas}x{self.canvas}")
        return tokens.reshape(b, self.canvas, self.canvas, c)


def fpre_grid_positions(
    grid_h: int, grid_w: int, *, device=None, dtype=torch.float32
) -> torch.Tensor:
    """``(grid_h * grid_w, 2)`` row-major ``(x, y)`` at the true aspect ratio.

    Delegates to :func:`q3vl.where.phi.norm_coords` so the Where-B position
    encoding and the Where-A ``geo5`` block cannot disagree about where a token
    is.
    """
    X, Y = norm_coords(grid_h, grid_w, device=device, dtype=dtype)
    return torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)


def default_connector_config() -> ConnectorConfig:
    return ConnectorConfig()
