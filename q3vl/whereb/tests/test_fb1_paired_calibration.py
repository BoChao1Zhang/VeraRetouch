"""F-B1: the directional paired difference is area-confounded without calibration.

Binarising the prediction to ``k = |GT_A|`` cells caps
``IoU(pred_k, GT_B) <= |GT_A|/|GT_B|``, so whenever the partner's region is
bigger the cross score is mechanically depressed and ``self - cross`` comes out
positive **for free** -- with no instruction information involved at all.

The three adversarial constructions are the reviewer's, reproduced here as a
regression: equal-area (the cancellation works), lopsided, and concentric (two
GTs sharing a centre, differing only in area).  The requirement the fix has to
meet is stated on the last one: after calibration against the centre prior, a
zero-information field's **net** delta must be ~0.
"""

from __future__ import annotations

import pytest
import torch

from q3vl.whereb.config import PAIRED_AREA_RATIO_RANGE
from q3vl.whereb.evaluate import _instruction_paired, _paired_rows
from q3vl.whereb.metrics import center_prior_field

GH, GW = 32, 32


def _disc(cy, cx, r):
    yy, xx = torch.meshgrid(torch.arange(GH).float(), torch.arange(GW).float(),
                            indexing="ij")
    return (((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r).float()


def _square(cy, cx, half):
    m = torch.zeros(GH, GW)
    m[max(0, cy - half):cy + half, max(0, cx - half):cx + half] = 1.0
    return m


def _fields(gt_a, gt_b, pred=None):
    """Two samples on one image.  `pred=None` -> the field IS the centre prior."""
    prior = center_prior_field(GH, GW)
    p = prior if pred is None else pred
    return {
        "sA": {"grid_pred": p.clone(), "grid_gt": gt_a,
               "source_image_id": "img", "instruction": "A"},
        "sB": {"grid_pred": p.clone(), "grid_gt": gt_b,
               "source_image_id": "img", "instruction": "B"},
    }


# --- the three adversarial constructions ------------------------------------

def test_equal_area_pairs_cancel_as_designed():
    """The mechanism the docstring claims really does work -- when areas match."""
    f = _fields(_square(8, 8, 5), _square(24, 24, 5))
    rows = _paired_rows(f, use_prior=True)
    raw = sum(r["self_iou"] - r["cross_iou"] for r in rows) / len(rows)
    assert abs(raw) < 1e-9, f"equal-area centre prior should score 0, got {raw}"


def test_lopsided_area_gives_a_zero_information_field_a_free_positive():
    f = _fields(_square(16, 16, 3), _square(16, 16, 9))      # 36 vs 324 cells
    rows = _paired_rows(f, use_prior=True)
    raw = sum(r["self_iou"] - r["cross_iou"] for r in rows) / len(rows)
    assert raw > 0.05, (
        f"the area confound did not reproduce (raw={raw:.4f}); if it is genuinely "
        "gone, F-B1's rationale needs re-checking rather than this test deleted"
    )


def test_concentric_pair_is_the_decisive_case():
    """Two GTs sharing a centre carry identical positional information, so an
    honest statistic must score ~0 -- the raw one scores far above that."""
    f = _fields(_disc(16, 16, 3.5), _disc(16, 16, 11.5))
    rows = _paired_rows(f, use_prior=True)
    raw = sum(r["self_iou"] - r["cross_iou"] for r in rows) / len(rows)
    assert raw > 0.3, f"concentric confound did not reproduce (raw={raw:.4f})"


# --- what the fix has to deliver --------------------------------------------

@pytest.mark.parametrize("name,gt_a,gt_b", [
    ("equal_area", _square(8, 8, 5), _square(24, 24, 5)),
    ("lopsided", _square(16, 16, 3), _square(16, 16, 9)),
    ("concentric", _disc(16, 16, 3.5), _disc(16, 16, 11.5)),
])
def test_calibrated_delta_is_zero_for_a_zero_information_field(name, gt_a, gt_b):
    """THE requirement: calibrate the centre prior against itself and the net
    directionality must vanish, on all three constructions."""
    out = _instruction_paired(_fields(gt_a, gt_b))
    cal = out["all_pairs"]["calibrated_delta"]
    assert cal == pytest.approx(0.0, abs=1e-9), (
        f"{name}: centre prior kept a net delta of {cal} after calibration"
    )


def test_calibration_column_is_reported_next_to_the_raw_value():
    out = _instruction_paired(_fields(_disc(16, 16, 3.5), _disc(16, 16, 11.5)))
    blk = out["all_pairs"]
    assert blk["delta"] is not None                 # raw, still reported
    assert blk["delta_center_prior"] is not None    # the calibration column
    assert blk["calibrated_delta"] is not None      # the headline
    assert blk["delta"] == pytest.approx(blk["delta_center_prior"])  # same field
    assert "calibrated_p_value" in blk and "calibrated_ci95" in blk


def test_a_field_that_really_follows_the_instruction_keeps_a_positive_margin():
    """Calibration must not flatten a genuine signal: a field that matches its
    own GT must still beat the centre prior after the area effect is removed."""
    gt_a, gt_b = _square(8, 8, 5), _square(24, 24, 7)
    out = _instruction_paired(_fields(gt_a, gt_b, pred=gt_a * 10.0))
    assert out["all_pairs"]["calibrated_delta"] > 0.2


def test_two_pairs_cannot_reach_significance_and_the_test_says_so():
    """A sign-flip test on n=2 has four assignments, so the smallest two-sided p
    it can report is ~0.5.  Reporting 1.0 here is the honest answer, not a bug --
    pinned so nobody 'fixes' it into a false positive."""
    out = _instruction_paired(_fields(_square(8, 8, 5), _square(24, 24, 7),
                                      pred=_square(8, 8, 5) * 10.0))
    assert out["all_pairs"]["n_pairs"] == 2
    assert out["all_pairs"]["calibrated_p_value"] >= 0.5


def test_a_powered_construction_does_reach_significance():
    """Same signal across many images: now the permutation test can see it."""
    fields = {}
    for g in range(12):
        a = _square(8, 8, 5)
        b = _square(24, 24, 6)
        fields[f"a{g}"] = {"grid_pred": a * 10.0, "grid_gt": a,
                           "source_image_id": f"img{g}", "instruction": "A"}
        fields[f"b{g}"] = {"grid_pred": b * 10.0, "grid_gt": b,
                           "source_image_id": f"img{g}", "instruction": "B"}
    out = _instruction_paired(fields)
    assert out["all_pairs"]["n_pairs"] == 24
    assert out["all_pairs"]["calibrated_delta"] > 0.5
    assert out["all_pairs"]["calibrated_p_value"] <= 0.05
    assert out["all_pairs"]["calibrated_ci95"][0] > 0


# --- the area-balanced second view ------------------------------------------

def test_area_balanced_subset_excludes_lopsided_pairs():
    lo, hi = PAIRED_AREA_RATIO_RANGE
    assert (lo, hi) == (0.5, 2.0)
    out = _instruction_paired(_fields(_square(16, 16, 3), _square(16, 16, 9)))
    assert out["n_pairs_total"] == 2
    assert out["n_pairs_area_balanced"] == 0        # 36/324 and 324/36 both out
    assert out["area_balanced"]["n_pairs"] == 0


def test_area_balanced_subset_keeps_comparable_pairs():
    out = _instruction_paired(_fields(_square(8, 8, 5), _square(24, 24, 6)))
    assert out["n_pairs_area_balanced"] == out["n_pairs_total"] == 2
    assert out["area_ratio_observed"]["min"] >= 0.5


def test_the_output_says_it_is_a_reported_column_not_a_gate():
    out = _instruction_paired(_fields(_square(8, 8, 5), _square(24, 24, 5)))
    assert "not a" in out["note"] and "gate" in out["note"]
    from q3vl.whereb.config import GATES
    assert not any("instruction_paired" in k for k, _, _ in GATES)
