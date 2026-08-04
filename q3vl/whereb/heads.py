"""Protocol 5.1 / 5.2 -- attention pools and the global parameter heads.

    "The output head produces only the global ``w0, w_dir, alpha, rho``; it does
     not produce dense logits."

There is deliberately no path in this module from a canvas token to a spatial
map: every head consumes a *pooled* ``(B, dim)`` vector, so the only way spatial
detail can reach the mask is through ``phi_dir`` and the analytic basis, exactly
as protocol 4.2 requires.

Initialisation matters more than usual here, because the zero-initialised cross
gates (protocol 5.1) make the canvas input-independent at step 0: whatever the
head's bias says *is* the model's first prediction for every image.  The output
projections are therefore zero-weight with an explicit, in-bounds bias:

* ``w0 = 0``, ``alpha = softplus(alpha_raw) ~= 1``;
* ``w_raw`` = a fixed seeded unit vector -- **not** zeros, because
  ``w_dir = w_raw / (||w_raw|| + 1e-12)`` has a 1e12-scale gradient at the
  origin and would blow up on the first step;
* the readout bias reproduces the informed start that the Where-A oracle fit
  uses (``band``: k=8, h~=1, pi~=0.88; ``cband12``: sigma ~= 0.45 * grid step).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from q3vl.where.config import CBAND_M, CBAND_SIG_HI, CBAND_SIG_LO
from q3vl.where.readout import inv_bounded_sigmoid, param_shapes

from .config import PHI_DIR_DIM
from .connector import MultiheadAttention

__all__ = ["AttentionPool", "AxisOutput", "RhoOutput", "LatentHeads",
           "rho_numel", "rho_split"]


def rho_numel(readout: str) -> int:
    return sum(int(torch.tensor(s).prod()) if s else 1
               for s in param_shapes(readout).values())


def rho_split(readout: str, flat: torch.Tensor) -> dict[str, torch.Tensor]:
    """``(B, n_rho)`` -> the named raw parameters with protocol shapes."""
    out: dict[str, torch.Tensor] = {}
    off = 0
    for name, shape in param_shapes(readout).items():
        n = 1
        for s in shape:
            n *= s
        chunk = flat[..., off:off + n]
        out[name] = chunk.squeeze(-1) if shape == () else chunk
        off += n
    if off != flat.shape[-1]:
        raise AssertionError(f"rho vector has {flat.shape[-1]} entries, used {off}")
    return out


def _softplus_inv(y: float) -> float:
    return float(y + math.log(-math.expm1(-y))) if y < 20 else float(y)


class AttentionPool(nn.Module):
    """One learnable probe cross-attending the canvas -> ``(B, dim)``."""

    def __init__(self, dim: int, n_heads: int, seed: int = 0):
        super().__init__()
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.probe = nn.Parameter(torch.randn(1, dim, generator=g) * (dim ** -0.5))
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = MultiheadAttention(dim, n_heads)
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, canvas: torch.Tensor) -> torch.Tensor:
        b = canvas.shape[0]
        q = self.probe.unsqueeze(0).expand(b, -1, -1)
        return self.norm_out(self.attn(q, self.norm_kv(canvas), None)).squeeze(1)


def _trunk(dim: int) -> nn.Module:
    return nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU())


class AxisOutput(nn.Module):
    """``pooled -> (w0, w_raw, alpha_raw)``."""

    def __init__(self, dim: int, seed: int = 0):
        super().__init__()
        self.proj = nn.Linear(dim, 1 + PHI_DIR_DIM + 1)
        g = torch.Generator(device="cpu").manual_seed(seed)
        w0 = torch.randn(PHI_DIR_DIM, generator=g)
        w0 = w0 / w0.norm()
        with torch.no_grad():
            self.proj.weight.zero_()
            bias = torch.empty(1 + PHI_DIR_DIM + 1)
            bias[0] = 0.0                                   # w0
            bias[1:1 + PHI_DIR_DIM] = w0                    # w_raw: unit vector
            bias[-1] = _softplus_inv(1.0)                   # alpha ~= 1
            self.proj.bias.copy_(bias)

    def forward(self, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y = self.proj(pooled)
        return y[..., 0], y[..., 1:1 + PHI_DIR_DIM], y[..., -1]


class RhoOutput(nn.Module):
    """``pooled -> the raw readout parameters`` (bounded maps live in readout.py)."""

    def __init__(self, dim: int, readout: str):
        super().__init__()
        self.readout = readout
        self.n = rho_numel(readout)
        self.proj = nn.Linear(dim, self.n)
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.bias.copy_(self.default_bias(readout))

    @staticmethod
    def default_bias(readout: str) -> torch.Tensor:
        if readout == "band":
            return torch.tensor([
                0.0,                                           # mu
                inv_bounded_sigmoid(1.0, 0.02, 2.50),          # h ~= 1.0
                inv_bounded_sigmoid(8.0, 1.0, 40.0),           # k = 8
                2.0,                                           # pi ~= 0.88
            ], dtype=torch.float32)
        if readout == "cband12":
            step = 6.0 / (CBAND_M - 1)
            sig0 = min(max(0.45 * step, CBAND_SIG_LO * 1.05), CBAND_SIG_HI * 0.95)
            sig_raw = inv_bounded_sigmoid(sig0, CBAND_SIG_LO, CBAND_SIG_HI)
            return torch.cat([
                torch.full((CBAND_M,), sig_raw),
                torch.zeros(CBAND_M),                          # o = 0.5
                torch.zeros(CBAND_M),                          # c = 0.5
            ]).float()
        raise ValueError(f"unknown readout {readout!r}")

    def forward(self, pooled: torch.Tensor) -> dict[str, torch.Tensor]:
        return rho_split(self.readout, self.proj(pooled))


class LatentHeads(nn.Module):
    """Pools + heads for one arm.

    ``n_pools == 1``  -> ``MC*-Joint``: one attention pool, one *joint* head
                         (a shared trunk with two output projections).
    ``n_pools == 2``  -> ``MC16-SplitHead`` / ``MC16-DualCanvas``: independent
                         pools and independent heads for ``w`` and ``rho``.
    """

    def __init__(self, dim: int, n_heads: int, readout: str, n_pools: int, seed: int = 0):
        super().__init__()
        if n_pools not in (1, 2):
            raise ValueError(f"n_pools must be 1 or 2, got {n_pools}")
        self.n_pools = n_pools
        self.readout = readout
        if n_pools == 1:
            self.pool = AttentionPool(dim, n_heads, seed=seed)
            self.trunk = _trunk(dim)
            self.axis_out = AxisOutput(dim, seed=seed + 1)
            self.rho_out = RhoOutput(dim, readout)
        else:
            self.pool_axis = AttentionPool(dim, n_heads, seed=seed)
            self.pool_rho = AttentionPool(dim, n_heads, seed=seed + 100)
            self.trunk_axis = _trunk(dim)
            self.trunk_rho = _trunk(dim)
            self.axis_out = AxisOutput(dim, seed=seed + 1)
            self.rho_out = RhoOutput(dim, readout)

    def forward(
        self, canvas_axis: torch.Tensor, canvas_rho: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if self.n_pools == 1:
            h = self.trunk(self.pool(canvas_axis))
            w0, w_raw, alpha_raw = self.axis_out(h)
            rho = self.rho_out(h)
        else:
            ha = self.trunk_axis(self.pool_axis(canvas_axis))
            hr = self.trunk_rho(self.pool_rho(canvas_rho))
            w0, w_raw, alpha_raw = self.axis_out(ha)
            rho = self.rho_out(hr)
        return w0, w_raw, alpha_raw, rho
