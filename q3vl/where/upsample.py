"""Protocol 4.2 -- one edge-aware guided upsample of the **scalar** ``s_low``.

    "Only the scalar ``s_low`` gets a single edge-aware guided upsample to give
     ``s(p)`` at the original aspect ratio; you may not upsample the 64 channels
     first and combine afterwards."

The rule is enforced structurally: :func:`guided_upsample` refuses any input
with more than one channel, and :func:`combine_then_upsample` is the only
sanctioned path from ``(phi_dir, latent)`` to a full-resolution ``s``.

Algorithm: the fast guided filter (He & Sun, arXiv:1505.00996), i.e. the linear
coefficients ``(a, b)`` of the guided filter (He, Sun & Tang, ECCV 2010 /
TPAMI 2013) are estimated on the low-resolution grid, bilinearly upsampled and
applied to the full-resolution guide:

    a_k = (mean(I p) - mean(I) mean(p)) / (var(I) + eps)
    b_k = mean(p) - a_k mean(I)
    q_i = mean(a)_i * I_i + mean(b)_i
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .basis import Latent, s_low
from .config import UpsampleConfig

__all__ = [
    "box_mean",
    "guided_upsample",
    "combine_then_upsample",
    "area_resize",
    "luma_guide",
    "ChannelOrderError",
]


class ChannelOrderError(RuntimeError):
    """Raised when something tries to upsample a multi-channel basis."""


def box_mean(x: torch.Tensor, radius: int) -> torch.Tensor:
    """``(B,C,H,W)`` box mean with window ``2r+1`` and replicate padding."""
    if radius <= 0:
        return x
    r = min(radius, min(x.shape[-2], x.shape[-1]) - 1)
    if r <= 0:
        return x
    pad = (r, r, r, r)
    return F.avg_pool2d(F.pad(x, pad, mode="replicate"), kernel_size=2 * r + 1, stride=1)


def area_resize(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Area-average resize (down) / bilinear (up) for ``(B,C,H,W)``."""
    h, w = size
    if x.shape[-2] == h and x.shape[-1] == w:
        return x
    mode = "area" if (h <= x.shape[-2] and w <= x.shape[-1]) else "bilinear"
    if mode == "area":
        return F.interpolate(x, size=size, mode="area")
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def luma_guide(img: torch.Tensor) -> torch.Tensor:
    """``(B,3,H,W)`` sRGB in [0,1] -> ``(B,1,H,W)`` Rec.709 luma."""
    if img.dim() != 4 or img.shape[1] != 3:
        raise ValueError(f"expected (B,3,H,W), got {tuple(img.shape)}")
    return (0.2126 * img[:, 0:1] + 0.7152 * img[:, 1:2] + 0.0722 * img[:, 2:3])


def guided_upsample(
    s_low_map: torch.Tensor,
    guide_hi: torch.Tensor,
    cfg: UpsampleConfig | None = None,
    return_domain_report: bool = False,
):
    """Edge-aware upsample of a **scalar** field.

    ``s_low_map``: ``(B,1,h,w)``.  ``guide_hi``: ``(B,1,H,W)`` in [0,1].
    The batch axes must agree: one scalar field per guide image.  That is not
    cosmetic -- without it, 64 semantic channels folded into the batch axis
    (``sem.reshape(64,1,h,w)``) would sail past the channel guard and give
    exactly the per-channel upsample protocol 4.2 forbids
    (REVIEW-impl-WhereA N-2).

    Returns ``s_hi`` or, with ``return_domain_report=True``, ``(s_hi, report)``
    where the report carries the *pre-clamp* range and the fraction of pixels the
    filter pushed outside the declared domain (CLAUDE.md s-cache contract: the
    consumer states its domain and asserts the raw data lives in it).
    """
    cfg = cfg or UpsampleConfig()
    if s_low_map.dim() != 4:
        raise ValueError(f"s_low must be (B,1,h,w), got {tuple(s_low_map.shape)}")
    if s_low_map.shape[1] != 1:
        raise ChannelOrderError(
            f"guided upsample got {s_low_map.shape[1]} channels. Protocol 4.2: the "
            "64 semantic channels must be combined into the scalar s_low *first*; "
            "upsampling the channels and combining afterwards is forbidden."
        )
    if guide_hi.dim() != 4 or guide_hi.shape[1] != 1:
        raise ValueError(f"guide must be (B,1,H,W), got {tuple(guide_hi.shape)}")
    if s_low_map.shape[0] != guide_hi.shape[0]:
        raise ChannelOrderError(
            f"batch mismatch: {s_low_map.shape[0]} scalar fields vs "
            f"{guide_hi.shape[0]} guide images. Protocol 4.2 allows exactly one "
            "guided upsample of one scalar per image; folding basis channels into "
            "the batch axis is the forbidden per-channel upsample in disguise."
        )

    H, W = guide_hi.shape[-2:]
    h, w = s_low_map.shape[-2:]
    guide_low = area_resize(guide_hi, (h, w))
    r = cfg.radius_low

    mI = box_mean(guide_low, r)
    mP = box_mean(s_low_map, r)
    corr = box_mean(guide_low * s_low_map, r)
    var = box_mean(guide_low * guide_low, r) - mI * mI
    a = (corr - mI * mP) / (var + cfg.eps)
    b = mP - a * mI
    a_hi = F.interpolate(box_mean(a, r), size=(H, W), mode="bilinear", align_corners=False)
    b_hi = F.interpolate(box_mean(b, r), size=(H, W), mode="bilinear", align_corners=False)
    s_hi = a_hi * guide_hi + b_hi

    lo, hi = cfg.domain
    report = {
        "domain": [lo, hi],
        "raw_min": float(s_hi.detach().min()),
        "raw_max": float(s_hi.detach().max()),
        "frac_below": float((s_hi.detach() < lo).to(s_hi.dtype).mean()),
        "frac_above": float((s_hi.detach() > hi).to(s_hi.dtype).mean()),
        "clamped": bool(cfg.clamp_domain),
    }
    report["frac_out_of_domain"] = report["frac_below"] + report["frac_above"]
    if cfg.clamp_domain:
        s_hi = s_hi.clamp(lo, hi)
    return (s_hi, report) if return_domain_report else s_hi


def combine_then_upsample(
    phi_dir: torch.Tensor,
    latent: Latent,
    grid_h: int,
    grid_w: int,
    guide_hi: torch.Tensor,
    cfg: UpsampleConfig | None = None,
    return_domain_report: bool = False,
):
    """The only sanctioned order: combine 71 channels -> scalar -> one upsample.

    Returns ``(s_hi (1,1,H,W), s_low (P,))``, plus the domain report when asked.
    """
    if phi_dir.shape[0] != grid_h * grid_w:
        raise ValueError(
            f"phi_dir has {phi_dir.shape[0]} rows but grid is {grid_h}x{grid_w}"
        )
    s = s_low(phi_dir, latent)                       # (P,) -- already scalar
    s_map = s.reshape(1, 1, grid_h, grid_w)
    s_hi, report = guided_upsample(s_map, guide_hi, cfg, return_domain_report=True)
    return (s_hi, s, report) if return_domain_report else (s_hi, s)
