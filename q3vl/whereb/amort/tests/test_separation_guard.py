"""Regression tests for the swap-subject separation term (review U1, nit N10).

U1 was not a coding slip -- the loss was correct in isolation and wrong against
the data it would be fed.  A unit test on the hinge alone would have passed, so
these tests pin the two *data* properties that made it wrong:

1. a pair whose two GTs are closer than the margin cannot satisfy the hinge even
   at the correct answer, so it must be rejected rather than trained on;
2. a pair that shares a subject noun is an antonym pair in disguise, and putting
   separation pressure on it is exactly what NOTES §7.2 forbids.
"""

from __future__ import annotations

import pytest
import torch

from q3vl.whereb.amort.losses import LossWeights, paired_separation


def test_hinge_is_unsatisfiable_when_gts_are_closer_than_the_margin():
    """At the perfect prediction the term must be 0 for a legal pair, >0 for a
    degenerate one -- which is precisely why degenerate pairs must be filtered."""
    margin = 0.05
    gt = torch.zeros(8, 8)
    gt[2:6, 2:6] = 1.0

    # legal pair: the partner is far away, so the perfect prediction satisfies it
    far = torch.zeros(8, 8)
    far[0:2, 6:8] = 1.0
    assert float(paired_separation(gt, gt, far, margin)) == 0.0

    # degenerate pair: partner is (nearly) the same mask.  Even m == gt_own
    # leaves the hinge active -- a constant-magnitude gradient at the right
    # answer.  This is the 15.2% of V_where pairs U1 measured.
    near = gt.clone()
    near[2, 2] = 0.0                       # d(gt_own, gt_partner) = 1/64 << margin
    assert float(paired_separation(gt, gt, near, margin)) > 0.0

    # identical GTs: the worst case, and 9 real pairs are bit-identical.  The
    # hinge sits at exactly the full margin -- no achievable prediction reduces
    # it, so the gradient never goes away.
    assert float(paired_separation(gt, gt, gt.clone(), margin)) == pytest.approx(margin)


def test_builder_rejects_same_subject_and_degenerate_pairs():
    """The guard, exercised through the builder's own logic without any I/O."""

    class _Stub:
        """Only the pieces `_partner_gt_low` touches."""

        sep_margin = 0.05

        def __init__(self):
            self.sep_reject: dict[str, int] = {}
            self._partner_cache: dict[str, torch.Tensor] = {}
            self._nouns: dict[str, set[str]] = {}
            self.device = torch.device("cpu")

        _nouns_of = lambda self, sid: self._nouns.get(sid, set())  # noqa: E731

    from q3vl.whereb.amort.data import AmortBatchBuilder

    stub = _Stub()
    gt = torch.zeros(8, 8)
    gt[2:6, 2:6] = 1.0

    # same-subject pair -> rejected before any mask is loaded
    stub._nouns = {"a": {"lizard"}, "b": {"lizard"}}
    stub.shuffle_index = type("S", (), {"partner_of": staticmethod(lambda s: "b")})()
    stub.dataset = object()
    stub.id_to_index = {"b": 0}
    out = AmortBatchBuilder._partner_gt_low(stub, "a", 8, 8, gt)
    assert out is None
    assert stub.sep_reject.get("same_subject") == 1

    # different subjects but near-identical GTs -> rejected on the margin
    stub.sep_reject = {}
    stub._nouns = {"a": {"lizard"}, "b": {"vase"}}
    stub._partner_cache = {"b@8x8": gt.clone()}
    out = AmortBatchBuilder._partner_gt_low(stub, "a", 8, 8, gt)
    assert out is None
    assert stub.sep_reject.get("gt_too_close") == 1

    # different subjects and a genuinely different GT -> accepted
    stub.sep_reject = {}
    far = torch.zeros(8, 8)
    far[0:2, 6:8] = 1.0
    stub._partner_cache = {"b@8x8": far}
    out = AmortBatchBuilder._partner_gt_low(stub, "a", 8, 8, gt)
    assert out is not None
    assert stub.sep_reject.get("accepted") == 1


def test_antonym_never_gets_a_loss_term():
    """NOTES §7.2: antonym is a negative control, reported and never trained.

    Guards the weight table itself -- the failure U1 describes arrives by giving
    antonym-like pairs a separation weight, so the absence of any antonym term is
    worth pinning explicitly.
    """
    w = LossWeights()
    assert not any("antonym" in f for f in w.to_dict())
