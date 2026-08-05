"""Protocol 7.1 -- the frozen, deterministic functional encoder for ``z_gt``.

    u(T)  = flatten(T(x) - x),  x a fixed 17^3 RGB grid
    z_gt  = L2Norm(SRHT_1024(u(T) - mean_train_u))

    "The SRHT uses a fixed public seed, the centre is computed on the train LUTs
     only, and no LUT lookup is learned.  A random projection preserves function
     distance and avoids training a second target encoder that could leak or
     collapse."

Why an SRHT and not a dense Gaussian matrix: ``u`` has ``17^3 * 3 = 14739``
entries, so a dense ``1024 x 14739`` projection is 15M floats that would have to
be shipped with every checkpoint and re-derived identically by any reader.  The
SRHT is three integers' worth of state (seed, pad, k) plus an ``O(n log n)``
transform, and it is the standard Johnson-Lindenstrauss construction:

    Phi = sqrt(n / k) * S * H * D

with ``D`` a random +-1 diagonal, ``H`` the *normalised* Walsh-Hadamard matrix
(``H H^T = I``), and ``S`` a uniform sample of ``k`` of the ``n`` rows.  Then
``E ||Phi v||^2 = ||v||^2`` and pairwise distances are preserved up to the usual
JL factor; ``tests/test_srht.py`` measures that on real LUT differences rather
than asserting it.

Everything here is a pure function of ``(seed, n, k)``: two processes, two
machines and two ranks produce bit-identical projections, which is what makes
``z_gt`` a *target* rather than another moving part.
"""

from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
import torch

from .config import SRHT_PAD, SRHT_SEED, ZGT_DIM, ZGT_GRID

__all__ = ["identity_grid", "u_of_table", "fwht", "SRHT", "default_srht",
           "encode_z_gt", "pairwise_rms", "running_mean_u"]


# --- the fixed evaluation grid ---------------------------------------------

def identity_grid(size: int = ZGT_GRID, device=None,
                  dtype=torch.float32) -> torch.Tensor:
    """``(size^3, 3)`` -- the fixed RGB lattice ``u`` is measured on.

    Row-major with R slowest, matching :func:`q3vl.what.lut.lattice_points`, so
    ``u`` from a baked cube and ``u`` from an analytic renderer index the same
    colours.
    """
    ax = torch.linspace(0.0, 1.0, size, device=device, dtype=dtype)
    r, g, b = torch.meshgrid(ax, ax, ax, indexing="ij")
    return torch.stack([r, g, b], dim=-1).reshape(-1, 3)


def u_of_table(values: torch.Tensor, grid: torch.Tensor | None = None) -> torch.Tensor:
    """``u(T) = flatten(T(x) - x)`` from ``T(x)`` evaluated on the fixed grid.

    ``values`` is ``(..., size^3, 3)``; the result is ``(..., size^3 * 3)``.
    """
    g = identity_grid(ZGT_GRID, values.device, values.dtype) if grid is None else grid
    if values.shape[-2:] != g.shape:
        raise ValueError(
            f"expected T(x) with shape (..., {g.shape[0]}, 3), got {tuple(values.shape)}"
        )
    return (values - g).reshape(*values.shape[:-2], -1)


# --- the transform ----------------------------------------------------------

def fwht(x: torch.Tensor) -> torch.Tensor:
    """In-place-free fast Walsh-Hadamard transform along the last axis.

    Unnormalised (``H_n`` with entries +-1); the caller divides by ``sqrt(n)``.
    ``n`` must be a power of two.
    """
    n = x.shape[-1]
    if n & (n - 1):
        raise ValueError(f"FWHT needs a power-of-two length, got {n}")
    y = x.clone()
    h = 1
    while h < n:
        y = y.reshape(*y.shape[:-1], n // (2 * h), 2, h)
        a, b = y[..., 0, :], y[..., 1, :]
        y = torch.stack([a + b, a - b], dim=-2)
        y = y.reshape(*y.shape[:-3], n)
        h *= 2
    return y


class SRHT:
    """A frozen ``n -> k`` subsampled randomised Hadamard transform.

    Not an ``nn.Module``: it has no parameters, is never trained, and must not be
    able to end up in an optimiser parameter group by accident.  The signs and
    the row sample are derived from ``seed`` alone.
    """

    def __init__(self, n_in: int, k: int = ZGT_DIM, pad: int = SRHT_PAD,
                 seed: int = SRHT_SEED):
        if pad & (pad - 1):
            raise ValueError(f"pad must be a power of two, got {pad}")
        if pad < n_in:
            raise ValueError(f"pad {pad} < input dimension {n_in}")
        if k > pad:
            raise ValueError(f"k {k} > pad {pad}")
        self.n_in, self.k, self.pad, self.seed = int(n_in), int(k), int(pad), int(seed)
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.signs = (torch.randint(0, 2, (pad,), generator=g, dtype=torch.int8) * 2 - 1)
        self.rows = torch.randperm(pad, generator=g)[:k].sort().values
        self.scale = float(np.sqrt(pad / k))

    # -- the projection -----------------------------------------------------
    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        """``(..., n_in) -> (..., k)``."""
        if v.shape[-1] != self.n_in:
            raise ValueError(f"expected (..., {self.n_in}), got {tuple(v.shape)}")
        pad = torch.zeros(*v.shape[:-1], self.pad, device=v.device, dtype=v.dtype)
        pad[..., : self.n_in] = v
        pad = pad * self.signs.to(v.device, v.dtype)
        y = fwht(pad) / float(np.sqrt(self.pad))
        return y[..., self.rows.to(y.device)] * self.scale

    # -- provenance ---------------------------------------------------------
    def digest(self) -> str:
        h = hashlib.sha256()
        h.update(f"{self.n_in}|{self.k}|{self.pad}|{self.seed}".encode())
        h.update(self.signs.numpy().tobytes())
        h.update(self.rows.numpy().astype(np.int32).tobytes())
        return h.hexdigest()

    def facts(self) -> dict[str, object]:
        return {
            "n_in": self.n_in, "k": self.k, "pad": self.pad, "seed": self.seed,
            "scale": self.scale, "digest": self.digest(),
            "kind": "SRHT(sqrt(n/k) * S * H * D)",
        }


_DEFAULT: SRHT | None = None


def default_srht() -> SRHT:
    """The one projection the whole campaign uses (17^3 * 3 -> 1024)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = SRHT(ZGT_GRID ** 3 * 3, ZGT_DIM, SRHT_PAD, SRHT_SEED)
    return _DEFAULT


# --- the full target --------------------------------------------------------

def encode_z_gt(
    values: torch.Tensor,
    center: torch.Tensor | None = None,
    srht: SRHT | None = None,
) -> torch.Tensor:
    """``z_gt = L2Norm(SRHT(u(T) - mean_train_u))``.

    ``center`` is ``mean_train_u`` (``(n_in,)``); ``None`` means "not centred
    yet", which is only legitimate in unit tests and in the job that *computes*
    the centre.  Training must pass it -- :class:`q3vl.what.data.WhatBatchBuilder`
    refuses to run without one.
    """
    s = srht or default_srht()
    u = u_of_table(values)
    if center is not None:
        u = u - center.to(u.device, u.dtype)
    z = s(u)
    return z / (z.norm(dim=-1, keepdim=True) + 1e-12)


def pairwise_rms(n: int, sum_u: torch.Tensor, sum_sq: float) -> float:
    """The constant ``C`` of amendment A-2, in closed form.

    ``C`` is the root mean square of ``||u_i - u_j||`` over **all** ``N(N-1)/2``
    train pairs.  Materialising those pairs is unnecessary::

        sum_{i<j} ||u_i - u_j||^2 = N * sum_i ||u_i||^2 - || sum_i u_i ||^2
        C^2 = 2 * ( N * sum_sq - ||sum_u||^2 ) / ( N * (N-1) )

    so one streaming pass over the train LUTs -- the same pass that accumulates
    ``mean_train_u`` -- yields the exact constant with no sampling and no
    quadratic cost.  ``sum_u`` is ``sum_i u_i``; ``sum_sq`` is ``sum_i ||u_i||^2``.
    """
    if n < 2:
        raise ValueError(f"C needs at least two train LUTs, got {n}")
    total = 2.0 * (n * float(sum_sq) - float(sum_u.double().pow(2).sum())) / (n * (n - 1))
    if total <= 0.0:
        raise ValueError(
            f"pairwise mean square came out {total}; every train LUT would have to "
            "be identical for that, which cannot be right"
        )
    return float(total ** 0.5)


def running_mean_u(chunks: Iterable[torch.Tensor]) -> torch.Tensor:
    """Streaming ``mean_train_u`` over an iterable of ``(B, n_in)`` blocks.

    Used by ``scripts/make_zgt_center.py``: the train LUT set is ~3.1k tables and
    ``u`` is 14739-dim, so the full stack is only ~180 MiB -- but the job also
    runs on the full 3.4k-LUT corpus in one pass, and a streaming mean means the
    peak does not depend on how many LUTs a future build adds.
    """
    total: torch.Tensor | None = None
    n = 0
    for c in chunks:
        c = c.double()
        total = c.sum(0) if total is None else total + c.sum(0)
        n += int(c.shape[0])
    if total is None or n == 0:
        raise ValueError("no LUTs supplied; mean_train_u is undefined")
    return (total / n).float()
