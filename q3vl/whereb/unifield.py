"""Shared machinery for the three ``RESEARCH_unified-field-prediction`` Gate-0 cards.

Three independent proposals (A = CH-NPE, B = FAFM, C = FPD) each put a
zero-training gate in front of their training arm.  All three gates need the
same four things, so they live here once:

1. **The guided upsample as an explicit linear operator** (:class:`GuidedOp`).
   ``q3vl.where.upsample.guided_upsample`` implements the *fast guided filter*
   with the **image** as guide and the field as the filtered signal::

       a = (mean(I p) - mean(I) mean(p)) / (var(I) + eps)
       b = mean(p) - a mean(I)
       q = mean(a) I + mean(b)

   Every ``p``-dependent term (``mean(I p)``, ``mean(p)``) is linear in ``p``
   and ``var(I)`` does not involve ``p`` at all, so the map ``p -> q`` is an
   honest linear operator ``A_I`` -- *provided the domain clamp is off*.  The
   clamp is the only nonlinearity in the chain, which is exactly why proposal B
   writes its renderer as ``clip(A_I c, 0, 1)`` with the clip **outside** the
   operator.  :class:`GuidedOp` therefore always runs with ``clamp_domain=False``
   and exposes the clip separately.

2. **The adjoint** ``A_I^T``, obtained by reverse-mode AD.  For a linear map the
   vector-Jacobian product *is* the adjoint, exactly, so this is not an
   approximation -- and :meth:`GuidedOp.adjoint_check` asserts the defining
   identity ``<A c, r> == <c, A^T r>`` rather than trusting it.

3. **The ridge/neck projection** ``c* = (A^T A + eps I)^-1 A^T y`` by conjugate
   gradients on the normal equations (:func:`ridge_project`), with an exact
   Cholesky cross-check available on small subsets (:func:`gram_exact`).

4. **The mandated criteria columns** (:func:`field_row`).  CLAUDE.md 2026-08-05
   bans AUC outright and fixes the replacement: soft-IoU / hard-IoU at
   matched-GT-area top-k, grid-level boundary F1, a zero-parameter centre-prior
   column on the same support and the same top-k rule, the ``a/(2-a)`` random
   floor, and area strata.  Nothing here computes an AUC.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch

from q3vl.where.config import UpsampleConfig
from q3vl.where.upsample import area_resize, guided_upsample, luma_guide

__all__ = [
    "FAMILIES",
    "AREA_STRATA",
    "family_of",
    "load_families",
    "GuidedOp",
    "ridge_project",
    "gram_exact",
    "field_row",
    "area_stratum",
    "agg",
    "median",
]

#: The four generative families of the local build.  ``slot_id`` in the source
#: build's ``.vrmeta.json`` is the only surviving label (amort_p1 NOTES section 1);
#: ``semantic`` is the "contour" family of the research document, the other three
#: are its "geometric primitive" families.  ``global`` render_mode samples are the
#: document's "全域常量" family and are handled by the caller (they are excluded
#: from the local population everywhere in this stage).
FAMILIES = ("radial", "linear", "band", "semantic")

#: Area strata mandated by amort_p1 NOTES section 7.6.  The ``>=0.45`` stratum holds
#: ~47% of V_where and its random floor is already 0.525, so it is always
#: reported as its own row rather than folded into the headline.
AREA_STRATA = ((0.0, 0.15), (0.15, 0.30), (0.30, 0.45), (0.45, 1.01))


def family_of(slot_id: str | None) -> str:
    """``"radial-1"`` -> ``"radial"``.  Mirrors ``mask_type_stats.family_of``."""
    return str(slot_id).rsplit("-", 1)[0] if slot_id else "unknown"


def load_families(ds, indices: Sequence[int], workers: int = 48) -> dict[str, str]:
    """``sample_id -> family`` for the given dataset rows.

    Reads the construction-side ``.vrmeta.json`` through ``MaskResolver``; the
    split index, the ``.rec.json`` and the maskview ``meta`` all drop the label
    (amort_p1 NOTES section 1).  75k rows take ~1.2 min on 48 threads, so this is
    never worth caching.
    """
    from q3vl.where.maskdata import MaskResolver

    res = MaskResolver(verify="none", suffix=".vrmeta.json")
    recs = [ds.record(i) for i in indices]

    def one(rec: dict) -> tuple[str, str]:
        try:
            vm = json.loads(res.read_bytes(res.resolve(rec)).decode())
            return rec["sample_id"], family_of(vm.get("slot_id"))
        except Exception:                                    # noqa: BLE001
            return rec["sample_id"], "unknown"

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        return dict(ex.map(one, recs))


# --------------------------------------------------------------------------- #
#  A_I -- the guided upsample as a linear operator                             #
# --------------------------------------------------------------------------- #

class GuidedOp:
    """``A_I : R^{h x w} -> R^{H x W}``, the frozen image-guided upsample.

    The clamp is forced off (``clamp_domain=False``): with it on the map is
    piecewise-affine, not linear, and every closed form downstream (the ridge
    projection, the spectrum, the superposition test) would be measuring the
    clip instead of the operator.  Proposal B renders ``clip(A_I c, 0, 1)`` with
    the clip applied afterwards, which is what :meth:`render` does.
    """

    def __init__(self, guide_hi: torch.Tensor, grid_h: int, grid_w: int,
                 cfg: UpsampleConfig | None = None, dtype=torch.float64):
        if guide_hi.dim() == 2:
            guide_hi = guide_hi[None, None]
        elif guide_hi.dim() == 3:
            guide_hi = guide_hi[None]
        if guide_hi.shape[1] != 1:
            raise ValueError(f"guide must be (1,1,H,W), got {tuple(guide_hi.shape)}")
        base = cfg or UpsampleConfig()
        # never clamp inside the operator -- see the class docstring
        self.cfg = UpsampleConfig(radius_low=base.radius_low, eps=base.eps,
                                  guide=base.guide, clamp_domain=False,
                                  domain=base.domain)
        self.guide = guide_hi.to(dtype)
        self.dtype = dtype
        self.grid_h, self.grid_w = int(grid_h), int(grid_w)
        self.n_low = self.grid_h * self.grid_w
        self.H, self.W = int(self.guide.shape[-2]), int(self.guide.shape[-1])
        self.n_hi = self.H * self.W

    # -- forward / adjoint --------------------------------------------------
    # Both always take and return **flat batched** tensors, ``(B, n_low)`` and
    # ``(B, n_hi)``.  Shape-sniffing was tried and is a trap here: ``(1, n_low)``
    # and ``(h, w)`` are both 2-D with the same element count, so "is this one
    # field or a batch?" has no correct answer.  Callers reshape explicitly.
    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """``(..., h*w)`` -> ``(B, H*W)``, unclamped."""
        cm = c.reshape(-1, 1, self.grid_h, self.grid_w).to(self.dtype)
        g = self.guide.expand(cm.shape[0], -1, -1, -1)
        return guided_upsample(cm, g, self.cfg).reshape(cm.shape[0], self.n_hi)

    def adjoint(self, r: torch.Tensor) -> torch.Tensor:
        """``A_I^T``: ``(..., H*W)`` -> ``(B, h*w)``.

        Reverse-mode AD of a *linear* map is exactly its adjoint -- no
        approximation.  :meth:`adjoint_check` asserts it rather than assuming it.
        """
        rm = r.reshape(-1, self.n_hi).to(self.dtype)
        c = torch.zeros(rm.shape[0], self.n_low, dtype=self.dtype,
                        device=rm.device, requires_grad=True)
        out = self.forward(c)
        (g,) = torch.autograd.grad(out, c, grad_outputs=rm)
        return g

    def normal(self, c: torch.Tensor, eps: float) -> torch.Tensor:
        """``(A^T A + eps I) c``, ``(B, n_low)`` in and out."""
        return self.adjoint(self.forward(c)) + eps * c.reshape(-1, self.n_low)

    def render(self, c: torch.Tensor, lo: float = 0.0, hi: float = 1.0) -> torch.Tensor:
        """``clip(A_I c, lo, hi)`` as ``(H, W)`` -- proposal B's renderer, one field."""
        return self.forward(c)[0].reshape(self.H, self.W).clamp(lo, hi)

    # -- self-checks --------------------------------------------------------
    def adjoint_check(self, seed: int = 0) -> float:
        """``|<A c, r> - <c, A^T r>| / |<A c, r>|``.  Should be ~1e-16 in fp64."""
        g = torch.Generator().manual_seed(seed)
        c = torch.randn(self.n_low, generator=g, dtype=self.dtype).to(self.guide.device)
        r = torch.randn(self.n_hi, generator=g, dtype=self.dtype).to(self.guide.device)
        lhs = float(torch.dot(self.forward(c).reshape(-1), r.reshape(-1)))
        rhs = float(torch.dot(c.reshape(-1), self.adjoint(r).reshape(-1)))
        return abs(lhs - rhs) / max(abs(lhs), 1e-30)

    def superposition_residual(self, c1: torch.Tensor, c2: torch.Tensor
                               ) -> dict[str, float]:
        """G0a / E0a: relative ``||A(c1+c2) - A c1 - A c2||``.

        Two denominators are reported because the research document only says
        "/ 尺度": ``rel_sum`` divides by ``||A(c1+c2)||`` (the natural relative
        error of the combined response) and ``rel_parts`` by
        ``||A c1|| + ||A c2||`` (the conservative one -- it cannot be inflated by
        cancellation between the two responses).
        """
        y1 = self.forward(c1).reshape(-1)
        y2 = self.forward(c2).reshape(-1)
        ys = self.forward(c1.reshape(-1) + c2.reshape(-1)).reshape(-1)
        resid = float((ys - y1 - y2).norm())
        return {
            "abs": resid,
            "rel_sum": resid / max(float(ys.norm()), 1e-30),
            "rel_parts": resid / max(float(y1.norm()) + float(y2.norm()), 1e-30),
        }


def gram_exact(op: GuidedOp, chunk: int = 256) -> torch.Tensor:
    """Explicit ``A^T A`` (``n_low x n_low``) by pushing the identity through.

    ``n_low`` is ~1536 here, so this is a few seconds and gives both an exact
    Cholesky solve and the true spectrum -- the quantity proposal B promises to
    put in its REPORT ("谱可直接进 REPORT").  Used on a subset to certify the CG
    solve used on the full population.
    """
    cols = []
    eye = torch.eye(op.n_low, dtype=op.dtype, device=op.guide.device)
    for i in range(0, op.n_low, chunk):
        cols.append(op.adjoint(op.forward(eye[i:i + chunk])))
    return torch.cat(cols, dim=0).T.contiguous()


def ridge_project(op: GuidedOp, y_hi: torch.Tensor, eps: float,
                  max_iter: int = 300, tol: float = 1e-10
                  ) -> tuple[torch.Tensor, dict[str, Any]]:
    """``c* = (A^T A + eps I)^-1 A^T y`` by conjugate gradients.

    The normal-equation system is symmetric positive definite for ``eps > 0``,
    so CG is the right solver and its residual is a certificate: the returned
    info carries the final relative residual, which the callers assert on.
    """
    b = op.adjoint(y_hi.reshape(-1)).reshape(1, op.n_low)
    x = torch.zeros_like(b)
    r = b - op.normal(x, eps)
    p = r.clone()
    rs = float((r * r).sum())
    b_norm = max(float(b.norm()), 1e-30)
    it = 0
    for it in range(1, max_iter + 1):
        Ap = op.normal(p, eps)
        denom = float((p * Ap).sum())
        if denom <= 0:
            break
        alpha = rs / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = float((r * r).sum())
        if rs_new ** 0.5 / b_norm < tol:
            rs = rs_new
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x.reshape(op.grid_h, op.grid_w), {
        "cg_iters": it,
        "cg_rel_residual": rs ** 0.5 / b_norm,
        "eps": eps,
    }


# --------------------------------------------------------------------------- #
#  criteria columns                                                            #
# --------------------------------------------------------------------------- #

def area_stratum(area: float) -> str:
    for lo, hi in AREA_STRATA:
        if lo <= area < hi:
            return f"{lo:.2f}-{hi:.2f}" if hi <= 1.0 else f">={lo:.2f}"
    return ">=0.45"


def field_row(pred_hi: torch.Tensor, gt_hi: torch.Tensor,
              grid_h: int, grid_w: int, *, prefix: str = "") -> dict[str, Any]:
    """The mandated criteria columns for one predicted field.

    Two resolutions, on purpose:

    * ``*_softiou_hi`` -- soft-IoU (min/max form) of the *soft* fields at
      delivery resolution.  This is the quantity the research document's
      reconstruction gates (B G0b 0.93/0.88, A E0c 0.85/0.75, C E0 0.92/0.80)
      are written against: they ask how much of the GT field survives a trip
      through the frozen decoder, not how a thresholded version scores.
    * ``*_softiou`` / ``*_hard_iou`` / ``*_gbf1`` -- the project's standing grid
      columns on the ``F_pre`` grid, binarised by matched-GT-area top-k (never a
      per-field threshold), next to the zero-parameter centre prior on the same
      support and the ``a/(2-a)`` random floor.

    No AUC in any form (CLAUDE.md red line).
    """
    from q3vl.whereb.metrics import (
        center_prior_field, grid_boundary_f1, gt_area_k, hard_iou, soft_iou_value,
        topk_mask,
    )

    p = prefix
    pred_hi = pred_hi.double().reshape(1, 1, *pred_hi.shape[-2:])
    gt_hi = gt_hi.double().reshape(1, 1, *gt_hi.shape[-2:])
    row: dict[str, Any] = {
        f"{p}softiou_hi": soft_iou_value(pred_hi[0, 0], gt_hi[0, 0]),
    }

    pred16 = area_resize(pred_hi, (grid_h, grid_w))[0, 0]
    gt16 = area_resize(gt_hi, (grid_h, grid_w))[0, 0]
    gt16b = (gt16 > 0.5).double()
    k = gt_area_k(gt16)
    area = float(gt16b.mean())
    cp = center_prior_field(grid_h, grid_w).double()

    row.update({
        f"{p}softiou": soft_iou_value(topk_mask(pred16, k), gt16b),
        f"{p}hard_iou": hard_iou(topk_mask(pred16, k), gt16b),
        f"{p}gbf1": grid_boundary_f1(topk_mask(pred16, k), gt16b),
        "centre_prior_softiou": soft_iou_value(topk_mask(cp, k), gt16b),
        "centre_prior_gbf1": grid_boundary_f1(topk_mask(cp, k), gt16b),
        "random_floor": area / (2 - area) if area < 1 else 1.0,
        "area_frac": area,
        "area_stratum": area_stratum(area),
        "gt_area_k": k,
    })
    return row


def median(xs: Iterable[float]) -> float | None:
    v = [float(x) for x in xs if x is not None and np.isfinite(x)]
    if not v:
        return None
    from q3vl.whereb.metrics import percentile
    return percentile(v, 0.5)


def agg(xs: Iterable[float]) -> dict[str, Any]:
    """``n / mean / median / p10 / p90`` -- nearest-rank order statistics."""
    from q3vl.whereb.metrics import percentile

    v = [float(x) for x in xs if x is not None and np.isfinite(x)]
    if not v:
        return {"n": 0}
    return {
        "n": len(v),
        "mean": float(np.mean(v)),
        "median": percentile(v, 0.5),
        "p10": percentile(v, 0.10),
        "p90": percentile(v, 0.90),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
    }


def by_group(rows: Sequence[dict[str, Any]], key: str, field: str
             ) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    groups: dict[str, list[float]] = {}
    for r in rows:
        groups.setdefault(str(r.get(key)), []).append(r.get(field))
    for g, vals in sorted(groups.items()):
        out[g] = agg(vals)
    return out


def guide_of(sample) -> torch.Tensor:
    """``(1,1,H,W)`` Rec.709 luma guide for one dataset sample."""
    return luma_guide(sample.image_tensor().double().unsqueeze(0))
