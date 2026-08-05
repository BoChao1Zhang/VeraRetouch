"""Protocol 7.2 -- Gaussian-aligned visual pooling.

    a_i(p) = m_pred(p) * N(I_in(p); mu_i, Sigma_i)
    v_i    = sum_p a_i(p) [RGB(p), Lab(p), V(F_pre(p))] / (sum_p a_i(p) + eps)

    "``v_i`` additionally concatenates ``log(sum a_i)`` and a valid bit; with no
     valid pixel it falls back to a masked global pool rather than producing NaN."

Everything is computed in the log domain.  ``N`` is a full normalised 3-D
Gaussian density: at ``sigma = 0.02`` its peak is ``(2 pi)^-1.5 * 0.02^-3 ~ 8e3``
and its value five sigmas out is ``e^-12.5`` smaller, so the ratio across one
image spans ~40 orders of magnitude.  Forming ``a_i`` and then dividing would
overflow to ``inf/inf`` for a tight Gaussian and underflow to ``0/0`` for a
distant one; ``softmax(log m + log N)`` over the pixel axis is the same quantity,
is exact, and makes the "no valid pixel" case a finite test on a logsumexp
instead of a division by zero.

Two grids, deliberately:

* the pooling itself runs on the ``F_pre`` grid (``H/16 x W/16``), because
  ``F_pre(p)`` is only defined there.  ``I_in`` is area-downsampled to it and
  ``m_pred`` is the low-resolution readout ``m_low``;
* protocol 9.1's *natural query colours* are drawn at full image resolution with
  the guided-upsampled ``m_hi`` -- see :mod:`q3vl.what.queries`.

Mask conditioning is an interface property, not a constant: protocol 6 gives
``m_pred`` to ``WC-1`` and ``WC-3`` only, so ``WC-0``/``WC-2`` and the strict
no-where controls pool with ``a_i(p) = N(...)`` alone.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .colorspace import srgb_to_lab_norm
from .config import FPRE_DIM, POOL_EPS, POOL_FEATURE_DIM, POOL_VALID_TAU, V_PROJ_DIM
from .gaussians import gaussian_log_density

__all__ = ["VisionProjector", "aligned_pool", "global_visual_pool", "roi_bg_pool",
           "LOG_MASS_SCALE", "LOG_MASS_CLAMP"]

#: DECISION: ``log(sum a_i)`` spans tens of nats while every other entry of
#: ``v_i`` is O(1).  It is clamped and divided by 10 so a single feature cannot
#: dominate the input projection at initialisation.  The clamp bounds are far
#: outside anything a normalised density on a 1e3-pixel grid produces.
LOG_MASS_CLAMP = 40.0
LOG_MASS_SCALE = 10.0
_NEG_INF = -1e30


class VisionProjector(nn.Module):
    """``V: 1024 -> 256`` of protocol 7.2."""

    def __init__(self, in_dim: int = FPRE_DIM, out_dim: int = V_PROJ_DIM):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, f_pre: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(f_pre))


def _pixel_features(rgb: torch.Tensor, v_feat: torch.Tensor) -> torch.Tensor:
    """``[RGB, Lab_norm, V(F_pre)]`` -> ``(B, P, 3 + 3 + 256)``.

    Lab is the *dimensionless* form (protocol 9.2's unit discipline applies to
    every tensor in this stage, not only to the loss).
    """
    return torch.cat([rgb, srgb_to_lab_norm(rgb), v_feat], dim=-1)


def aligned_pool(
    geometry: dict[str, torch.Tensor],
    rgb: torch.Tensor,                       # (B, P, 3) in [0,1], F_pre grid
    v_feat: torch.Tensor,                    # (B, P, 256) = V(F_pre)
    *,
    m_pred: torch.Tensor | None = None,      # (B, P) in [0,1] or None
    valid: torch.Tensor | None = None,       # (B, P) bool, padding mask
    tau: float = POOL_VALID_TAU,
    collect_stats: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """``v_i`` for all 48 slots -> ``(B, N, POOL_FEATURE_DIM)``.

    Differentiable in ``geometry``: protocol 7.4 requires the gradient to reach
    the provisional ``mu, Sigma`` through this pooling, so nothing here is
    detached.
    """
    B, P, _ = rgb.shape
    feat = _pixel_features(rgb, v_feat)                       # (B, P, F)
    ld = gaussian_log_density(rgb, geometry["mu"], geometry["sigma"],
                              geometry["off"])                # (B, N, P)
    log_a = ld
    if m_pred is not None:
        log_a = log_a + torch.log(m_pred.clamp_min(1e-12)).unsqueeze(1)
    if valid is not None:
        log_a = log_a.masked_fill(~valid.unsqueeze(1), _NEG_INF)

    log_mass = torch.logsumexp(log_a, dim=-1)                 # (B, N)
    is_valid = log_mass > torch.log(torch.tensor(tau, device=rgb.device,
                                                 dtype=rgb.dtype))
    # softmax over p == a_i / sum_p a_i, computed without ever forming a_i
    w = torch.softmax(log_a, dim=-1)
    pooled = torch.einsum("bnp,bpf->bnf", w, feat)

    # fallback: the masked global pool (protocol 7.2), broadcast to every slot
    fb_w = _fallback_weights(m_pred, valid, B, P, rgb.dtype, rgb.device)
    fallback = torch.einsum("bp,bpf->bf", fb_w, feat).unsqueeze(1).expand_as(pooled)
    pooled = torch.where(is_valid.unsqueeze(-1), pooled, fallback)

    extra = torch.stack([
        log_mass.clamp(-LOG_MASS_CLAMP, LOG_MASS_CLAMP) / LOG_MASS_SCALE,
        is_valid.to(pooled.dtype),
    ], dim=-1)
    v = torch.cat([pooled, extra], dim=-1)
    if v.shape[-1] != POOL_FEATURE_DIM:
        raise AssertionError(
            f"v_i is {v.shape[-1]}-dim, protocol 7.2 says {POOL_FEATURE_DIM}"
        )
    # review nit N-3: the finite assertion and the six scalars below each force a
    # device->host sync.  Once per micro-batch that is a measurable tax on a loop
    # whose forward is otherwise sync-free, and the quantities are only ever read
    # on logging steps.  ``collect_stats=False`` skips them; the *structural*
    # shape check above is free and always runs.
    if not collect_stats:
        return v, {"mask_conditioned": m_pred is not None, "stats_collected": False}
    if not torch.isfinite(v).all():
        raise AssertionError("aligned pooling produced a non-finite v_i")
    stats = {
        "n_invalid_slots": int((~is_valid).sum()),
        "invalid_fraction": float((~is_valid).float().mean()),
        "log_mass_min": float(log_mass.detach().min()),
        "log_mass_max": float(log_mass.detach().max()),
        "mask_conditioned": m_pred is not None,
        "stats_collected": True,
    }
    return v, stats


def _fallback_weights(m_pred, valid, B: int, P: int, dtype, device) -> torch.Tensor:
    """Masked global pool weights, normalised over the valid pixels."""
    if m_pred is not None:
        w = m_pred.clamp_min(0.0)
    else:
        w = torch.ones(B, P, dtype=dtype, device=device)
    if valid is not None:
        w = w * valid.to(dtype)
    s = w.sum(dim=-1, keepdim=True)
    # a fully empty row (no valid pixel at all) falls back to a flat average
    flat = torch.full_like(w, 1.0 / max(P, 1))
    return torch.where(s > POOL_EPS, w / (s + POOL_EPS), flat)


# --- protocol 6 pools -------------------------------------------------------

def global_visual_pool(f_pre: torch.Tensor,
                       valid: torch.Tensor | None = None) -> torch.Tensor:
    """``WC-0``'s "global visual pooling": the plain mean of ``F_pre``."""
    if valid is None:
        return f_pre.mean(dim=1)
    w = valid.to(f_pre.dtype)
    return (f_pre * w.unsqueeze(-1)).sum(1) / (w.sum(1, keepdim=True) + POOL_EPS)


def roi_bg_pool(f_pre: torch.Tensor, m_pred: torch.Tensor,
                valid: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """Protocol 6::

        F_roi = sum_p m(p) F_pre(p) / (sum_p m(p) + eps)
        F_bg  = sum_p (1-m(p)) F_pre(p) / (sum_p (1-m(p)) + eps)
    """
    w = m_pred.clamp(0.0, 1.0)
    if valid is not None:
        w = w * valid.to(w.dtype)
        inv = (1.0 - m_pred.clamp(0.0, 1.0)) * valid.to(w.dtype)
    else:
        inv = 1.0 - w
    roi = (f_pre * w.unsqueeze(-1)).sum(1) / (w.sum(1, keepdim=True) + POOL_EPS)
    bg = (f_pre * inv.unsqueeze(-1)).sum(1) / (inv.sum(1, keepdim=True) + POOL_EPS)
    return roi, bg
