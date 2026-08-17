"""EPR-030 loss set: ``L0`` = plain L1, plus three optional terms, all off.

``L0``
------
::

    L = mean( | f_theta(x) - L_l(x) | )        # RGB function values, mean over
                                              # colours and channels

One term, weight 1, nothing else.  Sources, both opened:

* **NILUT / CNILUT** (AAAI 2024) Eq.(6) ``L = sum_i || Phi(x_i) - phi(x_i) ||_1``
  -- the only published recipe that supervises in the same place this project
  does (function values on a colour set, not images); the reference
  implementation is ``fit.py:77-78``
  (https://raw.githubusercontent.com/mv-lab/nilut/main/fit.py), whose own
  comment on choosing L1 over L2 is ``# more stable than L2``.
* the archived transformer-backend trainer ``model/glut_repro/train_rdg.py``:
  ``loss_main = (render(p, x) - y).abs().mean()`` -- code fact only, that
  campaign's conclusions are not usable.

Three optional additive terms (default weight 0, implemented, not enabled)
--------------------------------------------------------------------------
================  =======  ===============================================
flag              value    form / source
================  =======  ===============================================
``--lambda-sparse``  1e-4  GLUT Eq.8 opacity entropy at the **family**-stable
                          regulariser weight (3DLUT / AdaInt / SepLUT all ship
                          ``sparse_factor = 0.0001``); the running arms use
                          0.001.
``--lambda-hc``      1.0   ``mean( (C / C.detach().mean()) * (1 - cos dhue) )``.
                          Weight 1 is CLUT-Net's direction term
                          (``utils/losses.py:24-25``, l1 and cos both weight 1,
                          ``sum(loss_ls).backward()``).  The ``C`` normalisation
                          is this project's own measured configuration -- the
                          only one under which ``L_rec`` fell and
                          ``cross_std`` rose (HANDOFF section 1.3).
``--lambda-mono``    10.0  monotonicity hinge ``mean(relu(v[i] - v[i+1]))`` over
                          the three axes of a sampled grid -- 3DLUT
                          ``models.py:340-359`` / ``mn_cons``, at the
                          family-stable ``monotonicity_factor = 10``
                          (3DLUT, AdaInt, 4D LUT all use 10).
================  =======  ===============================================

Two run-time assertions, because "defined but never wired" has happened three
times in this campaign
----------------------------------------------------------------------------
* :func:`assert_l0_pure` -- with ``--loss l0`` the two extra weights must be
  exactly 0; anything else raises before the first step.
* :func:`assert_l0_ran` -- :func:`l0_losses` bumps a process-wide counter, and
  the first quick eval asserts the counter is non-zero.  A loss module that was
  imported but never called fails there instead of at step 117,440.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from q3vl.whatb.colorimetry import chroma_hue, srgb_to_lab
from q3vl.whatb.glut import EPS

__all__ = [
    "LOSS_CHOICES",
    "LAMBDA_SPARSE_L0",
    "LAMBDA_HC_L0",
    "LAMBDA_MONO_L0",
    "HC_EPS",
    "L0Weights",
    "L0Output",
    "l1_reconstruction",
    "opacity_entropy",
    "hue_chroma_term",
    "monotonicity_hinge",
    "l0_losses",
    "l0_call_count",
    "reset_l0_call_count",
    "assert_l0_pure",
    "assert_l0_ran",
    "L0NotCalled",
    "term_grad_norm_shares",
]

#: ``--loss``.  Only ``l0`` exists on this arm; the spelling is a flag so the
#: pre-registration file records *which* recipe ran, not just its weights.
LOSS_CHOICES: tuple[str, ...] = ("l0",)

#: the family-stable values the three optional terms take when switched on
LAMBDA_SPARSE_L0: float = 1e-4
LAMBDA_HC_L0: float = 1.0
LAMBDA_MONO_L0: float = 10.0

#: the frozen block's ``C -> 0`` guard, unchanged (``h = (a,b)/max(C, eps_C)``
#: plus the hard mask ``1[C >= eps_C]``); only the *weighting* of the term
#: changes here, never its ``C -> 0`` handling.
HC_EPS: float = 1e-3

_CALLS: dict[str, int] = {"l0_losses": 0}


class L0NotCalled(AssertionError):
    """``l0_losses`` was never called: the loss is defined but not wired."""


def l0_call_count() -> int:
    """How many times :func:`l0_losses` has run in this process."""
    return int(_CALLS["l0_losses"])


def reset_l0_call_count() -> None:
    """Tests and re-entrant runs."""
    _CALLS["l0_losses"] = 0


def assert_l0_pure(*, lambda_hc: float, lambda_sparse: float,
                   lambda_mono: float = 0.0, loss: str = "l0") -> dict[str, float]:
    """``--loss l0`` means one term.  Raise if any extra weight is non-zero."""
    if loss not in LOSS_CHOICES:
        raise ValueError(f"--loss must be one of {LOSS_CHOICES}, got {loss!r}")
    extras = {"lambda_hc": float(lambda_hc), "lambda_sparse": float(lambda_sparse),
              "lambda_mono": float(lambda_mono)}
    hot = {k: v for k, v in extras.items() if v != 0.0}
    if hot:
        raise AssertionError(
            f"--loss l0 is the single L1 term; these weights are not 0: {hot}.  "
            "An additive row must be published as its own ablation line, not "
            "folded into the main arm's headline.")
    return extras


def assert_l0_ran(*, where: str = "quick_eval") -> int:
    """The counter check.  Called from the first quick eval, never skipped."""
    n = l0_call_count()
    if n <= 0:
        raise L0NotCalled(
            f"{where}: q3vl.whatb.losses_l0.l0_losses has been called {n} times -- "
            "the pre-registered L0 loss was never executed by the training loop")
    return n


# --------------------------------------------------------------------------- #
# the terms
# --------------------------------------------------------------------------- #
def l1_reconstruction(y_hat: Tensor, y: Tensor) -> Tensor:
    """``mean |y_hat - y|`` over colours and channels -- NILUT Eq.(6)."""
    if y_hat.shape != y.shape:
        raise ValueError(f"shape mismatch {tuple(y_hat.shape)} vs {tuple(y.shape)}")
    return (y_hat - y.to(device=y_hat.device, dtype=y_hat.dtype)).abs().mean()


def opacity_entropy(opacity: Tensor, *, eps: float = EPS) -> Tensor:
    """GLUT Eq.8 ``R_sparse``: the negated binary entropy of the opacities."""
    o = opacity
    return -(o * torch.log(o + eps) + (1.0 - o) * torch.log(1.0 - o + eps)).mean()


def hue_chroma_term(y_hat: Tensor, y: Tensor, *, eps_c: float = HC_EPS,
                    cnorm: bool = True, mask: bool = True) -> tuple[Tensor, int]:
    """``mean( w * (1 - <h_hat, h>) )`` with ``w = C`` or ``C / C.detach().mean()``.

    The ``C -> 0`` handling is the frozen block's, verbatim: hues divide by
    ``max(C, eps_c)`` (inside :func:`~q3vl.whatb.colorimetry.chroma_hue`) and the
    term is multiplied by the hard mask ``1[C >= eps_c]`` taken on the
    **target**'s chroma.  ``cnorm`` divides the chroma weight by its own
    detached batch mean, which is the only measured configuration under which
    this project's ``L_rec`` fell (HANDOFF section 1.3); the divisor is detached,
    so the term stays a weighting of the same quantity.

    Returns ``(loss, n_masked)``.
    """
    y = y.to(device=y_hat.device, dtype=y_hat.dtype)
    c, h, valid = chroma_hue(srgb_to_lab(y).detach(), eps_c)
    _, h_hat, _ = chroma_hue(srgb_to_lab(y_hat), eps_c)
    w = c
    if cnorm:
        w = c / c.detach().mean().clamp_min(eps_c)
    term = w * (1.0 - (h_hat * h).sum(dim=-1))
    m = valid.to(term.dtype)
    if not mask:
        m = torch.ones_like(m)
    n_valid = m.sum()
    loss = (term * m).sum() / n_valid.clamp_min(1.0)
    return loss, int(valid.numel() - int(valid.sum()))


def monotonicity_hinge(values: Tensor, n: int) -> Tensor:
    """``mean(relu(v[i] - v[i+1]))`` along the three axes of an ``n^3`` grid.

    ``values`` is ``(B, n^3, 3)`` -- ``f_theta`` evaluated on the *uniform*
    ``n^3`` sRGB grid in ``meshgrid(..., indexing="ij")`` order, so axis 0 is R,
    1 is G, 2 is B.  Form and reduction are 3DLUT's ``mn_cons``
    (``models.py:340-359``: ``dif = LUT[..., :-1] - LUT[..., 1:]``,
    ``mn = mean(relu(dif))``), applied to the sampled transform rather than to a
    stored LUT tensor because this carrier has no stored lattice.
    """
    n = int(n)
    if values.dim() != 3 or values.shape[-1] != 3 or values.shape[1] != n ** 3:
        raise ValueError(f"expected (B, {n ** 3}, 3) grid values, got {tuple(values.shape)}")
    v = values.reshape(values.shape[0], n, n, n, 3)
    parts = [
        torch.relu(v[:, :-1] - v[:, 1:]).mean(),
        torch.relu(v[:, :, :-1] - v[:, :, 1:]).mean(),
        torch.relu(v[:, :, :, :-1] - v[:, :, :, 1:]).mean(),
    ]
    return torch.stack(parts).mean()


# --------------------------------------------------------------------------- #
# the assembled loss
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class L0Weights:
    """Which terms are on, and at what weight.  ``as_dict`` goes in the record."""

    lambda_hc: float = 0.0
    lambda_sparse: float = 0.0
    lambda_mono: float = 0.0
    hc_cnorm: bool = True
    hc_eps: float = HC_EPS
    hc_mask: bool = True
    mono_grid: int = 9

    @property
    def pure_l1(self) -> bool:
        return (self.lambda_hc == 0.0 and self.lambda_sparse == 0.0
                and self.lambda_mono == 0.0)

    def as_dict(self) -> dict[str, Any]:
        return {"lambda_rec": 1.0, "lambda_hc": self.lambda_hc,
                "lambda_sparse": self.lambda_sparse, "lambda_mono": self.lambda_mono,
                "hc_cnorm": self.hc_cnorm, "hc_eps": self.hc_eps,
                "hc_mask": self.hc_mask, "mono_grid": self.mono_grid,
                "pure_l1": self.pure_l1}


@dataclass
class L0Output:
    """The total, every named part, and the ``steps.jsonl`` columns."""

    total: Tensor
    l_rec: Tensor
    l_hc: Tensor
    l_sparse: Tensor
    l_mono: Tensor
    n_hc_masked: int
    n_colors: int
    terms: dict[str, Tensor] = field(default_factory=dict)

    def row(self, *, n_luts_in_batch: int, mining_ratio_value: float,
            extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """The step row.  Column names are the frozen ``FIRST_STEP_COLUMNS`` set.

        ``L_hc`` / ``L_sparse`` are written even when their weight is 0 (they are
        then the measured value of an inactive term, or 0.0): the publication
        gate asserts the column *exists*, and a row that drops a frozen column
        because the term is off is the "silently absent" failure mode.
        """
        row: dict[str, Any] = {
            "L_rec": float(self.l_rec.detach()),
            "L_hc": float(self.l_hc.detach()),
            "L_sparse": float(self.l_sparse.detach()),
            "L_mono": float(self.l_mono.detach()),
            "n_colors": int(self.n_colors),
            "n_luts_in_batch": int(n_luts_in_batch),
            "mining_ratio": float(mining_ratio_value),
            "n_hc_masked": int(self.n_hc_masked),
            "loss": float(self.total.detach()),
        }
        if extra:
            row.update(dict(extra))
        return row


def l0_losses(y_hat: Tensor, y: Tensor, *, opacity: Tensor | None = None,
              grid_values: Tensor | None = None,
              weights: L0Weights = L0Weights(), eps: float = EPS) -> L0Output:
    """``L = L1 [+ l_sparse R_sparse] [+ l_hc L_hc] [+ l_mono L_mono]``.

    Every optional term is off by default, so the default return is exactly
    NILUT Eq.(6).  Calling this function is what :func:`assert_l0_ran` counts.
    """
    _CALLS["l0_losses"] += 1
    l_rec = l1_reconstruction(y_hat, y)
    total = l_rec
    terms: dict[str, Tensor] = {"L_rec": l_rec}

    l_hc = y_hat.new_zeros(())
    n_masked = 0
    if weights.lambda_hc != 0.0:
        l_hc, n_masked = hue_chroma_term(y_hat, y, eps_c=weights.hc_eps,
                                         cnorm=weights.hc_cnorm, mask=weights.hc_mask)
        total = total + float(weights.lambda_hc) * l_hc
        terms["L_hc"] = float(weights.lambda_hc) * l_hc

    l_sparse = y_hat.new_zeros(())
    if weights.lambda_sparse != 0.0:
        if opacity is None:
            raise ValueError("--lambda-sparse is non-zero but no opacity was passed "
                             "(GlutAux.opacity); refusing to skip a term silently")
        l_sparse = opacity_entropy(opacity, eps=eps)
        total = total + float(weights.lambda_sparse) * l_sparse
        terms["L_sparse"] = float(weights.lambda_sparse) * l_sparse

    l_mono = y_hat.new_zeros(())
    if weights.lambda_mono != 0.0:
        if grid_values is None:
            raise ValueError("--lambda-mono is non-zero but no grid values were passed; "
                             "the hinge needs f_theta on the uniform n^3 grid")
        l_mono = monotonicity_hinge(grid_values, weights.mono_grid)
        total = total + float(weights.lambda_mono) * l_mono
        terms["L_mono"] = float(weights.lambda_mono) * l_mono

    n_colors = (int(y_hat.shape[0] * y_hat.shape[1]) if y_hat.dim() == 3
                else int(y_hat.numel() // 3))
    return L0Output(total=total, l_rec=l_rec, l_hc=l_hc, l_sparse=l_sparse,
                    l_mono=l_mono, n_hc_masked=int(n_masked), n_colors=n_colors,
                    terms=terms)


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _global_norm(grads: Sequence[Tensor | None]) -> float:
    sq = [float((g.detach() ** 2).sum()) for g in grads if g is not None]
    return float(sum(sq) ** 0.5)


def term_grad_norm_shares(terms: Mapping[str, Tensor],
                          params: Sequence[torch.nn.Parameter]
                          ) -> dict[str, dict[str, float]]:
    """``|| grad_theta (lambda_i L_i) ||`` per term, and each one's share.

    This is GradNorm's ``G_W^(i)(t) = || grad_W w_i(t) L_i(t) ||_2``
    (http://proceedings.mlr.press/v80/chen18a/chen18a.pdf Eq.1), computed
    **for monitoring only** -- no weight is adjusted from it, in either
    direction.  Terms are expected pre-multiplied by their weight.

    Costs one extra backward per term, so callers run it every ``--diag-every``
    steps, never every step.  ``retain_graph=True`` on all but the last.
    """
    names = [k for k, v in terms.items() if v is not None and v.requires_grad]
    out: dict[str, dict[str, float]] = {}
    if not names or not params:
        return out
    norms: dict[str, float] = {}
    for i, name in enumerate(names):
        grads = torch.autograd.grad(terms[name], list(params), retain_graph=True,
                                    allow_unused=True)
        norms[name] = _global_norm(grads)
    total = sum(norms.values())
    for name in names:
        out[name] = {"grad_norm": norms[name],
                     "share": (norms[name] / total) if total > 0 else 0.0}
    return out
