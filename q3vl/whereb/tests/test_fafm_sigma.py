"""``sigma_max_sq`` must track the operator's resolution, not a stale constant.

REVIEW-impl-amort-uni U5: ``SIGMA_MAX_SQ_FULLRES = 256.0`` was measured at full
delivery resolution and then consumed against the quarter-resolution operator the
FAFM trainer actually applies, where the true value is 16.  The metric term of
``Lambda`` -- the case's headline "task-aligned metric" and its literal execution
of constraint 3 -- therefore carried ~6% of its intended weight.

These tests pin the identity that makes that unrepresentable::

    ||A_I||_2^2 = n_hi / n_low

and assert it against an explicit Gram eigendecomposition of the real frozen
operator at several resolutions.
"""

from __future__ import annotations

import pytest
import torch

from q3vl.whereb.fafm import SIGMA_MAX_SQ_FULLRES, sigma_max_sq
from q3vl.whereb.unifield import GuidedOp, gram_exact


def _guide(h: int, w: int) -> torch.Tensor:
    """A *smooth* guide -- the regime every real photographic luma guide is in.

    A uniform-noise guide is deliberately NOT used: there the constant mode stops
    being dominant and the closed form becomes a loose lower bound (measured 27.6
    vs 16.0), which would be testing a regime this operator never sees.
    """
    y = torch.linspace(0, torch.pi, h, dtype=torch.float64)[:, None]
    x = torch.linspace(0, torch.pi, w, dtype=torch.float64)[None, :]
    return (0.5 + 0.4 * torch.sin(y) * torch.cos(x))[None, None]


@pytest.mark.parametrize("div", [1, 2, 4])
def test_constant_mode_gain_is_exactly_the_closed_form(div: int) -> None:
    """``||A 1||^2 / ||1||^2 == n_hi / n_low`` -- an identity, for ANY guide.

    This is the part that is exact; it is what makes the closed form trustworthy
    rather than fitted.
    """
    gh, gw = 8, 12
    op = GuidedOp(_guide(gh * 16 // div, gw * 16 // div), gh, gw)
    ones = torch.ones(op.n_low, dtype=torch.float64)
    rayleigh = float((op.forward(ones) ** 2).sum() / (ones ** 2).sum())
    assert rayleigh == pytest.approx(sigma_max_sq(op.n_hi, op.n_low), rel=1e-9)


@pytest.mark.parametrize("div", [1, 2, 4])
def test_sigma_max_sq_matches_measured_gram(div: int) -> None:
    """On a smooth guide the constant mode IS the top eigenvector: within 1%."""
    gh, gw = 8, 12
    op = GuidedOp(_guide(gh * 16 // div, gw * 16 // div), gh, gw)
    gram = gram_exact(op).double()
    measured = float(torch.linalg.eigvalsh(0.5 * (gram + gram.T)).max())
    predicted = sigma_max_sq(op.n_hi, op.n_low)
    assert predicted == pytest.approx(op.n_hi / op.n_low)
    assert measured == pytest.approx(predicted, rel=0.01), (
        f"div={div}: measured {measured} vs closed form {predicted}"
    )


def test_full_resolution_constant_is_reproduced() -> None:
    """The published Gate-0 constant is the closed form at the full-res ratio.

    Guards the provenance claim in ``SIGMA_MAX_SQ_FULLRES``'s docstring: 256 is
    (512*768)/(32*48), i.e. output pixels per coarse cell at H/16 -> H.
    """
    assert sigma_max_sq(512 * 768, 32 * 48) == pytest.approx(SIGMA_MAX_SQ_FULLRES)


def test_quarter_resolution_is_sixteen_not_the_full_res_constant() -> None:
    """The exact regression U5 describes: the trainer's operator is 16, not 256."""
    quarter = sigma_max_sq((512 // 4) * (768 // 4), 32 * 48)
    assert quarter == pytest.approx(16.0)
    assert SIGMA_MAX_SQ_FULLRES / quarter == pytest.approx(16.0)


def test_lambda_metric_loss_requires_an_explicit_sigma() -> None:
    """No default: a stale default is how the 16x underweight happened."""
    import inspect

    from q3vl.whereb.fafm import lambda_metric_loss

    sig = inspect.signature(lambda_metric_loss)
    param = sig.parameters["sigma_max_sq_value"]
    assert param.default is inspect.Parameter.empty


def test_metric_term_weight_is_resolution_invariant() -> None:
    """A constant residual must score the same through operators of any resolution.

    This is the behavioural statement of U5: before the fix, the same residual
    measured through the quarter-resolution operator scored 16x too small.
    """
    from q3vl.whereb.fafm import lambda_metric_loss

    gh, gw = 8, 12
    vals = []
    for div in (1, 2, 4):
        op = GuidedOp(_guide(gh * 16 // div, gw * 16 // div), gh, gw)
        smsq = sigma_max_sq(op.n_hi, op.n_low)
        c_hat = torch.ones(1, 1, gh, gw, dtype=torch.float64)
        c_star = torch.zeros(1, 1, gh, gw, dtype=torch.float64)

        def apply_A(r: torch.Tensor, _op=op) -> torch.Tensor:
            return _op.forward(r)

        vals.append(float(lambda_metric_loss(c_hat, c_star, apply_A, 0.0, smsq)))
    assert max(vals) / min(vals) == pytest.approx(1.0, rel=0.02), vals
