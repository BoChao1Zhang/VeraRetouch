"""Protocol 9.1 -- the 2048 RGB query points ``L_func`` is evaluated on.

    "For every sample take 2048 RGB query points:
      - 1024 from a fixed stratified uniform-RGB sampler;
      - 1024 from the current ``I_in``'s natural RGB, sampled for local samples
        with weights from the frozen ``m_pred`` and for global samples from the
        whole image.
     ... The uniform and natural halves are 50% each and are reported separately,
     to avoid fitting only the colours that are frequent in natural images."

The uniform half is *fixed*: one 1024-point set, drawn once from
``UNIFORM_SEED``, shared by every sample, every arm and every evaluation.  That
is what makes "uniform-half CIEDE2000" comparable across arms at all -- a
resampled uniform set would put a different, if small, amount of sampling noise
into every number.  8^3 = 512 strata with 2 jittered points each covers the cube
isotropically; a plain uniform draw of 1024 points leaves visible holes on a
cube of volume 1.

The natural half is per sample and deterministic in ``(seed, sample_id)``: an
evaluation re-run must land on the same colours, or the "natural-half" column
moves for reasons that have nothing to do with the checkpoint.
"""

from __future__ import annotations

import hashlib

import torch

from .config import (
    N_QUERY_NATURAL,
    N_QUERY_UNIFORM,
    UNIFORM_PER_STRATUM,
    UNIFORM_SEED,
    UNIFORM_STRATA,
)

__all__ = ["uniform_query_points", "natural_query_points", "sample_seed",
           "QUERY_KINDS", "query_kind_index"]

QUERY_KINDS = ("uniform", "natural")

_UNIFORM: torch.Tensor | None = None


def uniform_query_points(device=None, dtype=torch.float32) -> torch.Tensor:
    """The fixed stratified 1024-point uniform sample -> ``(1024, 3)``."""
    global _UNIFORM
    if _UNIFORM is None:
        n = UNIFORM_STRATA
        g = torch.Generator(device="cpu").manual_seed(UNIFORM_SEED)
        ax = torch.arange(n, dtype=torch.float64)
        r, gg, b = torch.meshgrid(ax, ax, ax, indexing="ij")
        base = torch.stack([r, gg, b], dim=-1).reshape(-1, 3)          # (512, 3)
        base = base.repeat_interleave(UNIFORM_PER_STRATUM, dim=0)      # (1024, 3)
        jitter = torch.rand(base.shape, generator=g, dtype=torch.float64)
        pts = (base + jitter) / n
        if pts.shape[0] != N_QUERY_UNIFORM:
            raise AssertionError(f"{pts.shape[0]} uniform points, need {N_QUERY_UNIFORM}")
        _UNIFORM = pts.float()
    return _UNIFORM.to(device=device, dtype=dtype)


def sample_seed(base_seed: int, sample_id: str) -> int:
    """A stable per-sample seed; ``hash()`` is salted per process and unusable."""
    h = hashlib.sha256(f"{base_seed}|{sample_id}".encode()).digest()
    return int.from_bytes(h[:8], "big") % (2 ** 31 - 1)


def natural_query_points(
    image: torch.Tensor,                 # (3, H, W) in [0,1]
    n: int = N_QUERY_NATURAL,
    *,
    weights: torch.Tensor | None = None,  # (H, W) frozen m_pred, or None
    seed: int = 0,
) -> torch.Tensor:
    """``(n, 3)`` colours drawn from the image's own pixels.

    ``weights`` is the frozen ``m_pred`` for a local sample and ``None`` for a
    global one (protocol 9.1).  Sampling is *with* replacement: the alternative
    would silently change the effective weighting when the mask concentrates on
    fewer than ``n`` pixels, which is exactly the small-mask regime the local
    strata are there to measure.
    """
    if image.dim() != 3 or image.shape[0] != 3:
        raise ValueError(f"expected (3,H,W), got {tuple(image.shape)}")
    flat = image.reshape(3, -1).t()                                # (P, 3)
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    if weights is None:
        idx = torch.randint(0, flat.shape[0], (n,), generator=g)
    else:
        w = weights.reshape(-1).float().clamp_min(0.0)
        if w.shape[0] != flat.shape[0]:
            raise ValueError(
                f"weights have {w.shape[0]} entries, image has {flat.shape[0]} pixels"
            )
        if float(w.sum()) <= 0.0:
            # a fully-zero predicted mask still has to produce colours; falling
            # back to the whole image is the honest choice and is reported by
            # the caller's ``mask_mass`` statistic rather than hidden.
            idx = torch.randint(0, flat.shape[0], (n,), generator=g)
        else:
            idx = torch.multinomial(w, n, replacement=True, generator=g)
    return flat[idx.to(flat.device)].clamp(0.0, 1.0)


def query_kind_index(n_uniform: int = N_QUERY_UNIFORM,
                     n_natural: int = N_QUERY_NATURAL,
                     device=None) -> torch.Tensor:
    """``0`` for the uniform half, ``1`` for the natural half (reporting split)."""
    return torch.cat([
        torch.zeros(n_uniform, dtype=torch.long, device=device),
        torch.ones(n_natural, dtype=torch.long, device=device),
    ])
