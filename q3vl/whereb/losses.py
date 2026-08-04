"""Protocol 5.5 -- the Where loss, symbol by symbol.

    L_mask  = (1 - softIoU(m_pred, m_gt))
            + 0.25 * balanced_BCE(m_pred, m_gt)
            + 0.10 * boundary_F1_loss_3px(m_pred, m_gt)

    L_s     = Huber(s_pred / 3, s* / 3)
    L_curve = mean_z |R(z; rho_pred) - r*(z)|,  z = linspace(-3, 3, 257)
    L_dir   = 1 - cosine(w_dir_pred, w_dir*)

    first 30% of optimizer steps:  L = L_mask + 1.00 L_s + 1.00 L_curve + 0.10 L_dir
    remaining 70%:                 L = L_mask + 0.25 L_s + 0.25 L_curve + 0.05 L_dir

``boundary_F1_loss_3px`` is the differentiable boundary-F1 surrogate of
Bokhovkin & Burnaev, *Boundary Loss for Remote Sensing Imagery Semantic
Segmentation*, arXiv:1905.07852 (title, authors and equations verified against
the paper on 2026-08-05):

    y_b      = pool(1 - y, theta0) - (1 - y)          # boundary map
    y_b_ext  = pool(y_b, theta)                       # tolerance band
    P        = sum(p_b * gt_b_ext) / sum(p_b)
    R        = sum(gt_b * p_b_ext) / sum(gt_b)
    L_BF1    = 1 - 2PR / (P + R)

with ``theta0 = 3`` (a 1px-wide boundary) and a tolerance *radius* of 3 px
(pooling window 7), which is what "3px" means as a distance.

Every loss here is defined for the two degenerate cases the data actually
contains: a global sample whose GT mask is all ones (no boundary at all) and a
sample without an oracle latent (auxiliaries masked out, ``L_mask`` still on).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from q3vl.where.readout import apply_readout

from .config import (
    AUX_DENOMINATOR,
    BOUNDARY_KERNEL,
    BOUNDARY_TOL_PX,
    CURVE_Z_HI,
    CURVE_Z_LO,
    CURVE_Z_N,
    HUBER_DELTA,
    MASK_BCE_W,
    MASK_BF1_W,
    MASK_IOU_W,
    MASK_WEIGHT,
    S_SCALE,
    STAGE1_FRACTION,
    STAGE1_WEIGHTS,
    STAGE2_WEIGHTS,
)

__all__ = [
    "soft_iou", "balanced_bce", "boundary_map", "boundary_f1_loss", "mask_loss",
    "loss_s", "loss_curve", "loss_dir", "curve_grid", "schedule_weights",
    "SampleLoss", "sample_loss", "aggregate",
]

_EPS = 1e-6
_BOUNDARY_TAU = 1e-3          # total soft boundary mass below which "no boundary"


# --- mask terms -------------------------------------------------------------

def soft_iou(m: torch.Tensor, t: torch.Tensor, kind: str = "minmax") -> torch.Tensor:
    """``sum(min)/sum(max)`` -- the PLAN v2 L205 / E2 / Where-A definition."""
    if kind == "minmax":
        return torch.minimum(m, t).sum() / (torch.maximum(m, t).sum() + _EPS)
    if kind == "prod":
        inter = (m * t).sum()
        return inter / (m.sum() + t.sum() - inter + _EPS)
    raise ValueError(f"unknown soft-IoU kind {kind!r}")


def balanced_bce(m: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Per-image class-balanced BCE with soft targets.

    ``w_pos = 0.5 / mean(t)`` and ``w_neg = 0.5 / (1 - mean(t))`` make the
    positive and the negative mass contribute equally, so a 2%-area mask cannot
    be beaten by predicting zero everywhere.  For a global sample (``t == 1``)
    the negative term is exactly zero and the huge ``w_neg`` never multiplies
    anything non-zero.
    """
    p = m.clamp(_EPS, 1.0 - _EPS)
    pos = t.mean()
    w_pos = 0.5 / (pos + _EPS)
    w_neg = 0.5 / (1.0 - pos + _EPS)
    return -(w_pos * t * torch.log(p) + w_neg * (1.0 - t) * torch.log(1.0 - p)).mean()


def _as_map(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        return x[None, None]
    if x.dim() == 3:
        return x[None]
    if x.dim() == 4:
        return x
    raise ValueError(f"expected a 2D/3D/4D map, got {tuple(x.shape)}")


def _pool(x: torch.Tensor, k: int) -> torch.Tensor:
    return F.max_pool2d(x, kernel_size=k, stride=1, padding=k // 2)


def boundary_map(y: torch.Tensor, kernel: int = BOUNDARY_KERNEL) -> torch.Tensor:
    """``pool(1 - y, theta0) - (1 - y)`` -- arXiv:1905.07852 eq. (boundary map)."""
    inv = 1.0 - y
    return (_pool(inv, kernel) - inv).clamp(0.0, 1.0)


def boundary_f1_loss(
    m: torch.Tensor,
    t: torch.Tensor,
    kernel: int = BOUNDARY_KERNEL,
    tol_px: int = BOUNDARY_TOL_PX,
) -> torch.Tensor:
    m4, t4 = _as_map(m), _as_map(t)
    bp, bt = boundary_map(m4, kernel), boundary_map(t4, kernel)
    ext = 2 * tol_px + 1
    bp_ext, bt_ext = _pool(bp, ext), _pool(bt, ext)
    sp, st = bp.sum(), bt.sum()
    prec = (bp * bt_ext).sum() / (sp + _EPS)
    rec = (bt * bp_ext).sum() / (st + _EPS)
    f1 = 2.0 * prec * rec / (prec + rec + _EPS)
    loss = 1.0 - f1
    # A GT with no boundary (global mask) and a prediction with no boundary is a
    # perfect match, not a total miss; without this the constant 1.0 would sit
    # in every global sample's loss with zero gradient.
    both_empty = (st < _BOUNDARY_TAU) & (sp < _BOUNDARY_TAU)
    return torch.where(both_empty, torch.zeros_like(loss), loss)


def mask_loss(
    m: torch.Tensor, t: torch.Tensor, *, iou_kind: str = "minmax",
    tol_px: int = BOUNDARY_TOL_PX, spatial: bool = True,
) -> dict[str, torch.Tensor]:
    """Protocol 5.5 ``L_mask`` and its three components."""
    iou = soft_iou(m.reshape(-1), t.reshape(-1), iou_kind)
    bce = balanced_bce(m.reshape(-1), t.reshape(-1))
    bf1 = (boundary_f1_loss(m, t, tol_px=tol_px) if spatial
           else torch.zeros((), device=m.device, dtype=m.dtype))
    total = MASK_IOU_W * (1.0 - iou) + MASK_BCE_W * bce + MASK_BF1_W * bf1
    return {"L_mask": total, "soft_iou": iou, "bce": bce, "bf1_loss": bf1}


# --- oracle auxiliaries -----------------------------------------------------

def curve_grid(device=None, dtype=torch.float32) -> torch.Tensor:
    return torch.linspace(CURVE_Z_LO, CURVE_Z_HI, CURVE_Z_N, device=device, dtype=dtype)


def loss_s(s_pred: torch.Tensor, s_star: torch.Tensor) -> torch.Tensor:
    return F.huber_loss(s_pred / S_SCALE, s_star / S_SCALE, delta=HUBER_DELTA)


def loss_curve(
    readout: str, rho_pred: dict[str, torch.Tensor], r_star: torch.Tensor,
    z: torch.Tensor | None = None,
) -> torch.Tensor:
    zz = z if z is not None else curve_grid(r_star.device, r_star.dtype)
    r_pred = apply_readout(readout, zz, rho_pred)
    return (r_pred - r_star).abs().mean()


def loss_dir(w_dir_pred: torch.Tensor, w_dir_star: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(
        w_dir_pred.reshape(1, -1), w_dir_star.reshape(1, -1), dim=1
    ).squeeze(0)


# --- schedule ---------------------------------------------------------------

def schedule_weights(
    step: int, total_steps: int, stage1_fraction: float = STAGE1_FRACTION
) -> dict[str, float]:
    """Protocol 5.5.  ``step`` is a 0-based *optimizer* step.

    The boundary is ``round(stage1_fraction * total_steps)``; step
    ``boundary - 1`` is still stage 1 and step ``boundary`` is stage 2.
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    boundary = int(round(stage1_fraction * total_steps))
    stage = 1 if step < boundary else 2
    w = STAGE1_WEIGHTS if stage == 1 else STAGE2_WEIGHTS
    return {"stage": stage, "boundary_step": boundary, "mask": MASK_WEIGHT, **w}


# --- one sample -------------------------------------------------------------

@dataclass
class SampleLoss:
    """One sample's contribution, with the mask term and the oracle auxiliaries
    kept **separate** so :func:`aggregate` can give them different denominators
    (ruling D-B16 / review blocker B5)."""

    total: torch.Tensor                       # mask_term + aux_term, per sample
    parts: dict[str, torch.Tensor] = field(default_factory=dict)
    scalars: dict[str, float] = field(default_factory=dict)
    has_oracle: bool = False
    mask_term: torch.Tensor | None = None
    aux_term: torch.Tensor | None = None


def sample_loss(
    *,
    readout: str,
    fields: dict[str, torch.Tensor],
    rho_pred: dict[str, torch.Tensor],
    mask_target: torch.Tensor,
    weights: dict[str, float],
    s_star: torch.Tensor | None = None,
    r_star: torch.Tensor | None = None,
    w_dir_star: torch.Tensor | None = None,
    mask_space: str = "hi",
    iou_kind: str = "minmax",
    tol_px: int = BOUNDARY_TOL_PX,
) -> SampleLoss:
    """Protocol 5.5 for one sample.

    ``mask_target`` must live in the same space as ``mask_space`` selects
    (``hi`` -> the spec-5 image grid, ``low`` -> the ``F_pre`` grid).
    Auxiliaries are skipped -- not zero-filled -- when the sample has no oracle
    latent (every global sample, and any local sample whose Where-A fit was
    rejected).
    """
    m = fields["m_hi"] if mask_space == "hi" else fields["m_low"]
    if mask_space == "hi" and "m_hi" not in fields:
        raise KeyError("mask_space='hi' needs the guided-upsampled m_hi")
    mt = mask_target.reshape(m.shape) if m.numel() == mask_target.numel() else mask_target
    mk = mask_loss(m, mt, iou_kind=iou_kind, tol_px=tol_px, spatial=True)

    mask_term = weights["mask"] * mk["L_mask"]
    aux_term: torch.Tensor | None = None
    parts: dict[str, torch.Tensor] = {"L_mask": mk["L_mask"]}
    scalars: dict[str, float] = {
        "soft_iou": float(mk["soft_iou"].detach()),
        "bce": float(mk["bce"].detach()),
        "bf1_loss": float(mk["bf1_loss"].detach()),
    }

    has_oracle = s_star is not None and r_star is not None and w_dir_star is not None
    if has_oracle:
        ls = loss_s(fields["s_low"], s_star)
        lc = loss_curve(readout, rho_pred, r_star)
        ld = loss_dir(fields["w_dir"], w_dir_star)
        aux_term = weights["s"] * ls + weights["curve"] * lc + weights["dir"] * ld
        parts.update({"L_s": ls, "L_curve": lc, "L_dir": ld})
        scalars.update({
            "L_s": float(ls.detach()), "L_curve": float(lc.detach()),
            "L_dir": float(ld.detach()),
            "s_std_ratio": float(
                (fields["s_low"].std(unbiased=False)
                 / (s_star.std(unbiased=False) + _EPS)).detach()
            ),
        })
    total = mask_term if aux_term is None else mask_term + aux_term
    return SampleLoss(total=total, parts=parts, scalars=scalars,
                      has_oracle=has_oracle, mask_term=mask_term, aux_term=aux_term)


def aggregate(
    losses: Sequence[SampleLoss], contexts: Sequence[str] | None = None,
    aux_denominator: str = AUX_DENOMINATOR,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Micro-batch reduction, plus a per-context breakdown (protocol 5.4).

        L = mean_i(L_mask_i) + (sum over samples WITH an oracle of the weighted
                                auxiliaries) / n_with_oracle

    Ruling D-B16 (review blocker B5).  ``L_mask`` applies to every sample, so it
    is averaged over the batch.  The three oracle auxiliaries only exist for
    samples that *have* an oracle latent -- every global sample and every
    rejected Where-A fit is excluded -- so averaging them over the batch too
    would silently multiply protocol 5.5's nominal weights by the local fraction
    of the split (train: 47.45%).  Stage 1's ``1.00 L_s + 1.00 L_curve`` would
    then really be ``~0.47``, closer to stage 2's ``0.25`` than to stage 1, and
    the two-stage contrast that stage 1 exists for would be flattened.

    ``aux_denominator="batch"`` restores the diluted behaviour; it is kept only
    so the effect can be reproduced.  ``stats`` always reports
    ``aux_effective_scale`` = the factor the nominal weights were actually
    multiplied by, so a future reader can never be misled about it again.

    "Every loss is computed on the GT and the generated sub-batch alike", so the
    per-context breakdown is a report, never a re-weighting.
    """
    if not losses:
        raise ValueError("empty batch")
    n = len(losses)
    n_oracle = sum(l.has_oracle for l in losses)
    mask_terms = [l.mask_term if l.mask_term is not None else l.total for l in losses]
    total = torch.stack(mask_terms).mean()
    aux = [l.aux_term for l in losses if l.aux_term is not None]
    if aux:
        aux_sum = torch.stack(aux).sum()
        if aux_denominator == "with_oracle":
            denom, scale = float(len(aux)), 1.0
        elif aux_denominator == "batch":
            denom, scale = float(n), len(aux) / n
        else:
            raise ValueError(f"unknown aux_denominator {aux_denominator!r}")
        total = total + aux_sum / denom
    else:
        scale = 0.0
    stats: dict[str, Any] = {
        "n": n, "n_with_oracle": n_oracle,
        "oracle_fraction": n_oracle / n,
        "aux_denominator": aux_denominator,
        "aux_effective_scale": scale,
    }
    keys = sorted({k for l in losses for k in l.scalars})
    for k in keys:
        vals = [l.scalars[k] for l in losses if k in l.scalars]
        if vals:
            stats[k] = sum(vals) / len(vals)
    stats["loss"] = float(total.detach())
    if contexts is not None:
        if len(contexts) != len(losses):
            raise ValueError("contexts and losses must align")
        by: dict[str, list[SampleLoss]] = {}
        for c, l in zip(contexts, losses):
            by.setdefault(c, []).append(l)
        stats["by_context"] = {
            c: {
                "n": len(v),
                "loss": float(torch.stack([x.total for x in v]).mean().detach()),
                **{k: (sum(x.scalars[k] for x in v if k in x.scalars)
                       / max(1, sum(k in x.scalars for x in v)))
                   for k in keys if any(k in x.scalars for x in v)},
            }
            for c, v in by.items()
        }
    return total, stats
