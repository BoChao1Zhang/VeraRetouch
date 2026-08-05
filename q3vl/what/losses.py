"""Protocol 9 -- the What loss, symbol by symbol.

    L_func       = mean Charbonnier(T_pred(x) - T_gt(x))      # 2048 query points
    L_hc         = mean [ normalize(C_gt) * (1 - cos(h_pred - h_gt)) ]
    R_sparse     = mean(binary_entropy(opacity) + binary_entropy(existence)) / 2
    L_style_cos  = 1 - cosine(z_style, z_gt)
    L_style_dist = Huber(d_style(i,j), d_func(T_i,T_j))       # pairs inside a batch
    L_var        = mean max(0, 1 - std(z_style_dim))
    L_cov        = normalized squared off-diagonal covariance
    L_bake       = mean Charbonnier(T_tetra(bake_33(T_pred), x) - T_pred(x))

    L_what = 1.00 L_func + 10.00 L_hc + 0.001 R_sparse + 0.05 L_style_cos
           + 0.05 L_style_dist + 0.02 (L_var + L_cov) + 0.10 L_bake

    "This recipe is the single starting point and main configuration for all
     twelve What arms; weights are not tuned per arm.  Only unit errors,
     non-finite values or implementation bugs may be fixed at preflight; the loss
     may not be changed after seeing the main results and the failing arm re-run."

Two things this module is built to make impossible rather than merely unlikely:

1. **A unit error in the Lab term.**  Everything that reaches ``L_hc`` comes from
   :func:`q3vl.what.colorspace.srgb_to_lab_norm`, i.e. ``(L/100, a/128, b/128)``.
   Protocol 9.2 names the failure explicitly ("the ~229x gradient amplification of
   old A0 must not be reproduced"), so :func:`grad_norm_ratio` computes
   ``||d L_func|| : ||d (10 L_hc)||`` from real gradients and the trainer writes it
   into every log row on a fixed schedule.
2. **``I_tar`` leaking into the loss.**  Protocol 9.5: "``I_tar`` does not enter
   ``L_what``".  No function here takes a rendered image; the target of ``L_func``
   is the LUT function itself.

``L_var``/``L_cov`` use the stop-gradient FIFO queue branch of protocol 9.3
("if each arm runs on a single GPU"), which is the protocol 11 schedule.  Queue
size, push rule and warm-up are constants in :mod:`q3vl.what.config` and identical
for every arm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F

from .colorspace import hue_cos_diff, normalised_chroma, srgb_to_lab_norm
from .config import (
    BAKE_SIZE,
    CHARBONNIER_EPS,
    HUBER_DELTA,
    LOSS_WEIGHTS,
    LutConfig,
    STYLE_QUEUE_MIN,
    STYLE_QUEUE_SIZE,
    VAR_EPS,
    VAR_TARGET,
)
from .gaussians import bake
from .lut import tetra_lookup

__all__ = [
    "charbonnier", "loss_func", "loss_hue_chroma", "binary_entropy", "loss_sparse",
    "loss_style_cos", "loss_style_dist", "pairwise_d_func", "StyleQueue",
    "loss_var_cov", "bake_readback", "loss_bake", "WhatLoss", "compute_loss",
    "grad_norm_ratio",
]

_EPS = 1e-8


# --- primitives -------------------------------------------------------------

def charbonnier(diff: torch.Tensor, eps: float = CHARBONNIER_EPS) -> torch.Tensor:
    """``sqrt(d^2 + eps^2)`` -- the smooth L1 GLUT/CGLUT reconstruct with."""
    return torch.sqrt(diff * diff + eps * eps)


def loss_func(t_pred: torch.Tensor, t_gt: torch.Tensor,
              kind: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """``L_func`` plus the mandatory uniform/natural split (protocol 9.1).

    ``kind`` is 0 for the uniform half and 1 for the natural half; the split is
    *reported*, never re-weighted -- the protocol fixes 50/50 by construction of
    the query set.
    """
    per_point = charbonnier(t_pred - t_gt).mean(-1)                # (B, Q)
    out = {"L_func": per_point.mean()}
    if kind is not None:
        k = kind if kind.dim() == 2 else kind.unsqueeze(0).expand_as(per_point)
        for i, name in enumerate(("uniform", "natural")):
            sel = k == i
            out[f"L_func_{name}"] = (
                (per_point * sel).sum() / sel.sum().clamp_min(1)
                if bool(sel.any()) else per_point.new_zeros(())
            )
    return out


def loss_hue_chroma(t_pred: torch.Tensor, t_gt: torch.Tensor) -> torch.Tensor:
    """``mean [ normalize(C_gt) * (1 - cos(h_pred - h_gt)) ]``, all dimensionless.

    Both Lab conversions go through the normalised form; ``normalize(C_gt)``
    divides by a global constant (``sqrt(2)``), never by a per-image or per-batch
    maximum.
    """
    lp = srgb_to_lab_norm(t_pred)
    lg = srgb_to_lab_norm(t_gt)
    return (normalised_chroma(lg) * hue_cos_diff(lp, lg)).mean()


def binary_entropy(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    q = p.clamp(eps, 1.0 - eps)
    return -(q * torch.log(q) + (1.0 - q) * torch.log(1.0 - q))


def loss_sparse(params: dict[str, torch.Tensor]) -> torch.Tensor:
    """GLUT/CGLUT's opacity sparsity, extended to the existence gate."""
    return 0.5 * (binary_entropy(params["opacity"]).mean()
                  + binary_entropy(params["existence"]).mean())


# --- style code -------------------------------------------------------------

def loss_style_cos(z_style: torch.Tensor, z_gt: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(z_style, z_gt, dim=-1)).mean()


def pairwise_d_func(u_gt: torch.Tensor, scale: float) -> torch.Tensor:
    """``d_func(T_i, T_j) = ||u(T_i) - u(T_j)||_2 / C`` -- amendment A-2.

    ``u_gt`` is the raw ``flatten(T(x) - x)`` on the fixed 17^3 grid: **not** the
    SRHT code and **not** L2-normalised.  ``C`` is a fixed constant precomputed
    over the whole train LUT set (:func:`q3vl.what.srht.pairwise_rms`), published
    next to ``mean_train_u``.

    Why the raw ``u`` and not ``z_gt``: ``z_gt = L2Norm(SRHT(u - mean))`` throws
    the *magnitude* of ``u`` away, so two presets differing only in strength (the
    same look at 50%) would get ``d_func = 0`` while their real function distance
    is large.  Since ``L_style_cos`` is also direction-only, using ``z_gt`` here
    would leave nothing in ``L_what`` supervising the edit magnitude of
    ``z_style`` -- and this corpus is Lightroom presets, where strength variants
    are the common case (review blocker B-3).

    Centring cancels in a difference, so the caller may pass centred or raw ``u``.
    """
    b = u_gt.shape[0]
    iu = torch.triu_indices(b, b, offset=1, device=u_gt.device)
    return (u_gt[iu[0]] - u_gt[iu[1]]).norm(dim=-1) / scale


def loss_style_dist(z_style: torch.Tensor, u_gt: torch.Tensor, scale: float,
                    delta: float = HUBER_DELTA) -> torch.Tensor:
    """Pairwise distance matching inside the micro-batch (protocol 9.3 / A-2).

        d_style(i, j) = || z_hat_i - z_hat_j ||,   z_hat = z_style / ||z_style||
        d_func(i, j)  = || u_i - u_j || / C

    ``d_style`` lives in [0, 2]; ``d_func`` has RMS 1 over the train set by
    construction of ``C``.  That is what ``C`` is for -- without it the Huber
    would be comparing a bounded quantity against an unbounded one and the term
    would either vanish or dominate depending on the corpus.
    """
    b = z_style.shape[0]
    if b < 2:
        return z_style.new_zeros(())
    if scale <= 0.0:
        raise ValueError(
            f"d_func scale C must be positive, got {scale}; it is a published "
            "train-set constant (see scripts/make_zgt_center.py), never a "
            "per-batch statistic"
        )
    zs = F.normalize(z_style, dim=-1)
    iu = torch.triu_indices(b, b, offset=1, device=z_style.device)
    d_style = (zs[iu[0]] - zs[iu[1]]).norm(dim=-1)
    d_func = pairwise_d_func(u_gt.to(z_style.dtype), scale)
    return F.huber_loss(d_style, d_func, delta=delta)


class StyleQueue:
    """Protocol 9.3's stop-gradient FIFO statistics queue.

    One arm per GPU means there is no all-gather to take batch statistics over,
    and a variance/covariance term computed on a micro-batch of 4 in 1024
    dimensions is noise.  The queue holds the last ``size`` **detached**
    ``z_style`` vectors; the statistics are taken over ``queue + current batch``
    and only the current batch carries gradient.  Contents and update rule are
    fixed for every arm (protocol 9.3's last sentence), so this class has no
    per-arm knobs.
    """

    def __init__(self, size: int = STYLE_QUEUE_SIZE, min_size: int = STYLE_QUEUE_MIN):
        self.size = int(size)
        self.min_size = int(min_size)
        #: rows stay on whatever device they arrived on (review nit N-12): moving
        #: 256 x 1024 to the host and back once per micro-batch is ~1 MB/step of
        #: pure sync for statistics that are consumed on the device anyway.
        self.buf: list[torch.Tensor] = []

    def __len__(self) -> int:
        return len(self.buf)

    def push(self, z: torch.Tensor) -> None:
        for row in z.detach().float():
            self.buf.append(row)
        while len(self.buf) > self.size:
            self.buf.pop(0)

    def stack(self, device=None, dtype=torch.float32) -> torch.Tensor | None:
        if not self.buf:
            return None
        return torch.stack(self.buf).to(device=device, dtype=dtype)

    def facts(self) -> dict[str, Any]:
        return {"size": self.size, "min_size": self.min_size, "n": len(self.buf)}


def loss_var_cov(z_style: torch.Tensor, queue: StyleQueue | None = None
                 ) -> dict[str, torch.Tensor]:
    """``L_var`` and ``L_cov`` over ``queue + batch`` with the queue detached."""
    parts = [z_style]
    if queue is not None:
        q = queue.stack(z_style.device, z_style.dtype)
        if q is not None:
            parts.append(q)
    z = torch.cat(parts, dim=0)
    m, d = z.shape
    if m < 2 or (queue is not None and m < queue.min_size):
        zero = z_style.new_zeros(())
        return {"L_var": zero, "L_cov": zero, "n_stats": m}
    zc = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(zc.var(dim=0, unbiased=False) + VAR_EPS)
    l_var = F.relu(VAR_TARGET - std).mean()
    cov = (zc.T @ zc) / (m - 1)
    off = cov - torch.diag_embed(torch.diagonal(cov))
    l_cov = (off ** 2).sum() / d
    return {"L_var": l_var, "L_cov": l_cov, "n_stats": m}


# --- bake consistency -------------------------------------------------------

def bake_readback(params: dict[str, torch.Tensor], cfg: LutConfig,
                  x: torch.Tensor, size: int = BAKE_SIZE) -> torch.Tensor:
    """``T_tetra(bake_33(T_pred), x)`` -- the delivery form, at the query points.

    ``bake`` evaluates the analytic mixture on the uniform lattice and
    ``tetra_lookup`` reads it back the way a ``.cube`` host would.  Both are
    differentiable, so ``L_bake`` shapes the function rather than only measuring
    it (protocol 9.4: "this constrains the trained function to still hold in its
    delivered form").
    """
    cube = bake(params, cfg, size)
    return tetra_lookup(cube, x.clamp(0.0, 1.0))


def loss_bake(t_pred: torch.Tensor, t_read: torch.Tensor) -> torch.Tensor:
    return charbonnier(t_read - t_pred).mean()


# --- the total --------------------------------------------------------------

@dataclass
class WhatLoss:
    total: torch.Tensor
    parts: dict[str, torch.Tensor] = field(default_factory=dict)
    scalars: dict[str, float] = field(default_factory=dict)


def compute_loss(
    *,
    t_pred: torch.Tensor,                 # (B, Q, 3)
    t_gt: torch.Tensor,                   # (B, Q, 3)
    x: torch.Tensor,                      # (B, Q, 3) the query points themselves
    params: dict[str, torch.Tensor],
    z_style: torch.Tensor,                # (B, 1024)
    z_gt: torch.Tensor,                   # (B, 1024)
    u_gt: torch.Tensor,                   # (B, 14739) raw flatten(T(x) - x)
    d_func_scale: float,                  # published train-set constant C (A-2)
    lut_cfg: LutConfig,
    query_kind: torch.Tensor | None = None,
    t_read: torch.Tensor | None = None,   # bake readback at the same points
    queue: StyleQueue | None = None,
    weights: dict[str, float] | None = None,
) -> WhatLoss:
    """Protocol 9.5 for one micro-batch.  ``I_tar`` is not a parameter of it."""
    w = dict(LOSS_WEIGHTS if weights is None else weights)
    parts: dict[str, torch.Tensor] = {}
    fn = loss_func(t_pred, t_gt, query_kind)
    parts["L_func"] = fn["L_func"]
    parts["L_hc"] = loss_hue_chroma(t_pred, t_gt)
    parts["R_sparse"] = loss_sparse(params)
    parts["L_style_cos"] = loss_style_cos(z_style, z_gt)
    parts["L_style_dist"] = loss_style_dist(z_style, u_gt, d_func_scale)
    vc = loss_var_cov(z_style, queue)
    parts["L_var"], parts["L_cov"] = vc["L_var"], vc["L_cov"]
    if t_read is None:
        t_read = bake_readback(params, lut_cfg, x)
    parts["L_bake"] = loss_bake(t_pred, t_read)

    total = (w["L_func"] * parts["L_func"]
             + w["L_hc"] * parts["L_hc"]
             + w["R_sparse"] * parts["R_sparse"]
             + w["L_style_cos"] * parts["L_style_cos"]
             + w["L_style_dist"] * parts["L_style_dist"]
             + w["L_varcov"] * (parts["L_var"] + parts["L_cov"])
             + w["L_bake"] * parts["L_bake"])

    scalars = {k: float(v.detach()) for k, v in parts.items()}
    scalars.update({k: float(v.detach()) for k, v in fn.items() if k != "L_func"})
    scalars["n_style_stats"] = float(vc["n_stats"])
    scalars["loss"] = float(total.detach())
    return WhatLoss(total=total, parts=parts, scalars=scalars)


# --- protocol 9.2's mandatory gradient-norm ratio ---------------------------

def grad_norm_ratio(
    params: Iterable[torch.nn.Parameter],
    l_func: torch.Tensor,
    l_hc: torch.Tensor,
    w_hc: float = LOSS_WEIGHTS["L_hc"],
    w_func: float = LOSS_WEIGHTS["L_func"],
) -> dict[str, float]:
    """``||grad(1.00 L_func)||`` vs ``||grad(10.00 L_hc)||`` -- protocol 9.2.

    "The training log records the ratio of ``grad_norm(L_func)`` to
     ``grad_norm(10 * L_hc)`` separately."  Two extra backward passes, so the
    trainer calls this on a schedule; the value is what would have caught A0's
    229x unit error on the first step instead of at the end of the run.
    """
    ps = [p for p in params if p.requires_grad]
    if not ps:
        return {}

    def _norm(loss: torch.Tensor) -> float:
        gs = torch.autograd.grad(loss, ps, retain_graph=True, allow_unused=True)
        sq = sum(float((g.detach() ** 2).sum()) for g in gs if g is not None)
        return float(sq ** 0.5)

    gf = _norm(w_func * l_func)
    gh = _norm(w_hc * l_hc)
    return {
        "grad_norm_L_func": gf,
        "grad_norm_w_L_hc": gh,
        "grad_ratio_hc_over_func": gh / gf if gf > _EPS else float("inf"),
    }


def aggregate_scalars(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    keys = sorted({k for r in rows for k in r})
    out: dict[str, float] = {}
    for k in keys:
        vals = [r[k] for r in rows if k in r]
        if vals:
            out[k] = sum(vals) / len(vals)
    return out
