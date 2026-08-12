"""The five pre-registered PR-AMORT loss terms.

    L = 1.00 * BCE_soft(m, .cgt, valid cells)
      + 0.10 * SDF boundary penalty
      + 0.05 * area-band penalty (tau = 0.15)
      + w_fake  * empty-mask term on foreign-instruction samples (p = 0.15)
      + w_sep   * swap-subject paired separation

Red lines honoured here: **IoU and dice are never optimisation targets** (they
are reported only); the prior field is never a target; ``antonym`` gets no loss
term of any kind.

Why antonym gets nothing
------------------------
E3's first draft read the antonym column backwards and briefly concluded M4.
The corrected reading: ``antonym`` flips only colour-direction words and keeps
the subject phrase, and the Where mask is by protocol a function of the subject,
not of the colour direction -- so "mask does not move" is the PASS, W01 already
achieves it (``median_abs_delta = 0.0``), and a separation/complementarity loss
built on it would actively destroy a control that currently passes.  NOTES §7.2
calls this "the closest thing to a trap in this round".  It is reported, never
trained.

Why the separation term takes the form it does
----------------------------------------------
The disease E3 actually measured is not "the instruction never arrives" -- it
does (swap-subject costs 0.145, fixed phrase 0.230).  It is that arriving buys
only **+0.033 over a zero-parameter centre prior**.  So the separation term is a
*triplet margin in mask space* against the partner's GT on the **same source
image**:

    L_sep = relu(margin - [ d(m, gt_partner) - d(m, gt_own) ])

with ``d`` a mean absolute difference over valid cells.  The decisive property:
a head that emits a centre blob is **equidistant** from ``gt_own`` and
``gt_partner``, scores a margin of exactly 0, and is therefore maximally
penalised.  No amount of copying the geometric prior can reduce this term -- only
actually resolving *which* subject the instruction names can.  That is the
quantity worth optimising, stated directly.

It costs one extra GT mask load and **no extra VLM forward**: ``ShuffleIndex``
groups by ``(source_image_id, render_mode)``, so the partner is a different edit
of the same picture and its mask is just another ``.cgt`` read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch

__all__ = ["LossWeights", "signed_distance_field", "bce_soft", "sdf_boundary",
           "area_band", "empty_mask", "paired_separation", "amort_sample_loss",
           "AmortLoss", "aggregate", "excess_curvature", "monotonicity_hinge",
           "eikonal"]


@dataclass(frozen=True)
class LossWeights:
    """Pre-registered into ``config/`` before the first step; never tuned later."""

    bce: float = 1.00
    sdf: float = 0.10
    area: float = 0.05
    fake: float = 0.20
    sep: float = 0.30
    #: area-band half width; |pred_area/gt_area - 1| inside tau is free
    area_tau: float = 0.15
    #: triplet margin for the swap-subject separation, in mean-abs-difference units
    sep_margin: float = 0.05
    #: probability that a training sample is served a foreign instruction
    fake_prob: float = 0.15
    #: --- P2 structural pack (family-gated, analytic families only) ---------
    #: Off by default: these are the prescription being probed, not part of the
    #: pre-registered baseline loss.
    curv: float = 0.0
    mono: float = 0.0
    #: SHAPE3: soft eikonal on the s field (distance-field reparameterisation)
    eik: float = 0.0
    #: strong-monotonicity slope floor.  RESEARCH §4.7: a bare monotonicity
    #: hinge is satisfiable by a flat field, so the constraint has to demand a
    #: slope, not merely a non-negative one.
    mono_eps: float = 0.02

    def to_dict(self) -> dict[str, float]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def signed_distance_field(gt: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Kervadec-style signed distance to the GT boundary, in grid cells.

    Negative inside the region, positive outside, normalised by the grid
    diagonal so the term's scale does not drift with image aspect ratio.  If the
    GT is degenerate (all in or all out) the field is all-zero, which makes the
    boundary term vanish for that sample rather than inventing a gradient.
    """
    from scipy.ndimage import distance_transform_edt

    g = (gt.detach().float() > threshold).cpu().numpy()
    if g.all() or not g.any():
        return torch.zeros_like(gt)
    outside = distance_transform_edt(~g)
    inside = distance_transform_edt(g)
    phi = outside - inside
    diag = float(np.hypot(*g.shape))
    return torch.from_numpy((phi / diag).astype(np.float32)).to(gt.device)


def _valid_mean(x: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
    if valid is None:
        return x.mean()
    v = valid.to(x.dtype)
    return (x * v).sum() / v.sum().clamp_min(1.0)


def bce_soft(m: torch.Tensor, gt: torch.Tensor,
             valid: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    """BCE against a **soft** target -- the ``.cgt`` alpha, not a binarisation.

    The target is an edit-falloff region; its soft band is signal (4.8% of a
    semantic mask, 23-87% of a geometric one), so thresholding it first would
    discard the very structure the boundary term then tries to recover.
    """
    m = m.clamp(eps, 1.0 - eps)
    t = gt.clamp(0.0, 1.0)
    return _valid_mean(-(t * m.log() + (1.0 - t) * (1.0 - m).log()), valid)


def sdf_boundary(m: torch.Tensor, phi: torch.Tensor,
                 valid: torch.Tensor | None = None) -> torch.Tensor:
    """``mean(phi * m)`` -- mass outside the region is charged by its distance."""
    return _valid_mean(phi * m, valid)


def area_band(m: torch.Tensor, gt: torch.Tensor, tau: float,
              valid: torch.Tensor | None = None) -> torch.Tensor:
    """Free inside ``|pred/gt - 1| <= tau``, hinge-linear outside.

    A band, not a point target: pinning the area exactly would make the head
    trade placement for area and is not what the criterion asks.
    """
    a_p = _valid_mean(m, valid)
    a_g = _valid_mean(gt, valid).clamp_min(1e-4)
    return torch.relu((a_p / a_g - 1.0).abs() - tau)


def empty_mask(m: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    """Foreign instruction -> nothing is being referred to -> emit nothing."""
    return _valid_mean(m, valid)


def paired_separation(m: torch.Tensor, gt_own: torch.Tensor,
                      gt_partner: torch.Tensor, margin: float,
                      valid: torch.Tensor | None = None) -> torch.Tensor:
    """Triplet margin against the same image's other edit target.

    Returns 0 when the prediction is already ``margin`` closer to its own target
    than to the partner's.  A centre-prior-shaped output sits at exactly 0
    advantage and is charged the full margin.
    """
    d_own = _valid_mean((m - gt_own).abs(), valid)
    d_par = _valid_mean((m - gt_partner).abs(), valid)
    return torch.relu(margin - (d_par - d_own))


def _grad(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gy = torch.zeros_like(y)
    gx = torch.zeros_like(y)
    gy[1:-1, :] = (y[2:, :] - y[:-2, :]) * 0.5
    gx[:, 1:-1] = (y[:, 2:] - y[:, :-2]) * 0.5
    return gy, gx


def excess_curvature(m: torch.Tensor, gt: torch.Tensor, eps: float = 1e-3,
                     k_max: float = 50.0) -> torch.Tensor:
    """Penalise iso-contour curvature **in excess of the GT's own**.

    Excess form, not absolute (RESEARCH §4.7): an absolute curvature penalty
    also flattens the legitimately curved families -- a small ellipse has real
    curvature and punishing it would trade one failure for another.  Only
    curvature the target does not have is charged.
    """
    # eps=1e-3 (not 1e-6) and a curvature clamp.  Measured 2026-08-11: with
    # eps=1e-6 the normalised gradient n = grad/|grad| explodes wherever the
    # field is locally flat, its derivative explodes further, and training goes
    # NaN between step ~50 and ~200 -- which silently destroyed 3 of 4 P2 arms
    # (all three landed on the identical degenerate IoU 0.1692).
    gy, gx = _grad(m)
    mag = torch.sqrt(gy ** 2 + gx ** 2 + eps ** 2)
    ny, nx = gy / mag, gx / mag
    dny, _ = _grad(ny)
    _, dnx = _grad(nx)
    k_m = (dny + dnx).abs().clamp(max=k_max)
    with torch.no_grad():
        ggy, ggx = _grad(gt)
        gmag = torch.sqrt(ggy ** 2 + ggx ** 2 + eps ** 2)
        gny, gnx = ggy / gmag, ggx / gmag
        dgny, _ = _grad(gny)
        _, dgnx = _grad(gnx)
        k_g = (dgny + dgnx).abs().clamp(max=k_max)
        band = (gmag > gmag.median()).float()
    return ((torch.relu(k_m - k_g) * band).sum() / band.sum().clamp_min(1.0))


def monotonicity_hinge(m: torch.Tensor, gt: torch.Tensor,
                       eps_slope: float = 0.02) -> torch.Tensor:
    """Charge directional derivatives that disagree in sign with the GT's.

    The direction field is the GT's own gradient direction (training-time
    privileged information, permitted for a training loss), and the hinge
    demands a slope of at least ``eps_slope`` rather than merely a non-negative
    one -- without that floor a constant field satisfies the constraint
    perfectly (RESEARCH §4.7).
    """
    gy, gx = _grad(m)
    with torch.no_grad():
        ggy, ggx = _grad(gt)
        gmag = torch.sqrt(ggy ** 2 + ggx ** 2)
        band = gmag > gmag.median()
        dy = ggy / (gmag + 1e-6)
        dx = ggx / (gmag + 1e-6)
    proj = gy * dy + gx * dx
    return (torch.relu(eps_slope - proj)[band]).mean() if band.any() \
        else m.sum() * 0.0


def eikonal(s_field: torch.Tensor, target: float = 1.0) -> torch.Tensor:
    """``mean((|grad s| - 1)^2)`` -- the soft distance-field constraint.

    Applied on the **s / logit** field, never on the saturated mask: RESEARCH
    §4.0 is explicit that an eikonal term in y space is meaningless wherever the
    sigmoid has flattened, which is most of the field.
    """
    gy, gx = _grad(s_field)
    mag = torch.sqrt(gy ** 2 + gx ** 2 + 1e-8)
    return ((mag - target) ** 2).mean()


@dataclass
class AmortLoss:
    total: torch.Tensor
    terms: dict[str, torch.Tensor] = field(default_factory=dict)
    stats: dict[str, float] = field(default_factory=dict)


def amort_sample_loss(
    m_pred: torch.Tensor,
    gt: torch.Tensor,
    w: LossWeights,
    *,
    valid: torch.Tensor | None = None,
    phi_sdf: torch.Tensor | None = None,
    gt_partner: torch.Tensor | None = None,
    is_fake: bool = False,
    structural: bool = False,
) -> AmortLoss:
    """One sample.  ``m_pred`` and ``gt`` live on the same grid, in [0, 1]."""
    terms: dict[str, torch.Tensor] = {}
    zero = m_pred.sum() * 0.0

    if is_fake:
        # A foreign instruction has no ground truth here: charging BCE against
        # this image's mask would teach the head to ignore the text, which is the
        # opposite of the point.  Only the empty-mask term applies.
        terms["fake"] = empty_mask(m_pred, valid)
        total = w.fake * terms["fake"]
        for k in ("bce", "sdf", "area", "sep", "curv", "mono"):
            terms[k] = zero
        with torch.no_grad():
            fake_area = float(_valid_mean(m_pred, valid))
        return AmortLoss(total=total, terms=terms,
                         stats={"is_fake": 1.0, "pred_area": fake_area})

    terms["bce"] = bce_soft(m_pred, gt, valid)
    terms["sdf"] = sdf_boundary(m_pred, phi_sdf, valid) if phi_sdf is not None else zero
    terms["area"] = area_band(m_pred, gt, w.area_tau, valid)
    terms["sep"] = (paired_separation(m_pred, gt, gt_partner, w.sep_margin, valid)
                    if gt_partner is not None else zero)
    terms["fake"] = zero
    # P2: family-gated structural pack.  `structural` is set by the caller only
    # for analytic families -- applying a monotonicity or curvature prior to a
    # semantic silhouette would be actively wrong.
    terms["curv"] = (excess_curvature(m_pred, gt) if (structural and w.curv)
                     else zero)
    terms["mono"] = (monotonicity_hinge(m_pred, gt, w.mono_eps)
                     if (structural and w.mono) else zero)

    total = (w.bce * terms["bce"] + w.sdf * terms["sdf"]
             + w.area * terms["area"] + w.sep * terms["sep"]
             + w.curv * terms["curv"] + w.mono * terms["mono"])

    with torch.no_grad():
        a_p = float(_valid_mean(m_pred, valid))
        a_g = float(_valid_mean(gt, valid))
        stats = {
            "is_fake": 0.0, "pred_area": a_p, "gt_area": a_g,
            "area_ratio": a_p / max(a_g, 1e-6),
            "pred_std": float(m_pred.std()), "gt_std": float(gt.std()),
            "has_partner": float(gt_partner is not None),
        }
    return AmortLoss(total=total, terms=terms, stats=stats)


def aggregate(losses: Sequence[AmortLoss]) -> tuple[torch.Tensor, dict[str, Any]]:
    """Mean over the micro-batch, plus the early-warning columns.

    ``area_ratio_median``, ``std_ratio_median`` and the fake fraction are logged
    every step because they are the registered stop-and-check triggers: an area
    ratio above 1.5 inside the first 20% of steps, or a std ratio below 0.5, is
    the over-coverage signature W01/W02 died of, and it is visible long before
    any evaluation runs.
    """
    if not losses:
        raise ValueError("no losses to aggregate")
    total = torch.stack([x.total for x in losses]).mean()
    out: dict[str, Any] = {}
    keys = sorted({k for x in losses for k in x.terms})
    for k in keys:
        vals = [x.terms[k] for x in losses if k in x.terms]
        out[f"L_{k}"] = float(torch.stack(vals).mean().detach())
    real = [x for x in losses if x.stats.get("is_fake", 0.0) < 0.5]
    out["n"] = len(losses)
    out["n_fake"] = len(losses) - len(real)
    if real:
        ar = sorted(x.stats["area_ratio"] for x in real)
        out["area_ratio_median"] = ar[len(ar) // 2]
        sr = sorted(x.stats["pred_std"] / max(x.stats["gt_std"], 1e-6) for x in real)
        out["std_ratio_median"] = sr[len(sr) // 2]
        out["pred_area_mean"] = float(np.mean([x.stats["pred_area"] for x in real]))
        out["gt_area_mean"] = float(np.mean([x.stats["gt_area"] for x in real]))
        out["frac_with_partner"] = float(np.mean([x.stats["has_partner"] for x in real]))
    return total, out
