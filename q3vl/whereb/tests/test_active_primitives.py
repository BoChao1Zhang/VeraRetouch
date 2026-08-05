"""Active-primitive-count stratum (Where-A fragility, reported not gated).

Where-A's failure analysis: fits with exactly one active CBand12 primitive are
systematically fragile in the guided upsample's hi tier -- 30.8% of samples,
median hi-vs-low drop +0.012 against +0.006 at >= 3 primitives, and all three
collapse cases were single-primitive narrow-band extrapolations.  Where-B
predicts the same rho, so it can inherit the fragility; the hi-tier columns are
stratified so it shows up instead of averaging away.

Main-agent ruling: **reporting only** -- no loss, gate or training change.
"""

from __future__ import annotations

import pytest
import torch

from q3vl.whereb import config as C
from q3vl.whereb import metrics as M
from q3vl.whereb.evaluate import strata_report


def _cband_rho(n_active: int, m: int = 12):
    """c_raw large -> c ~ 1 (on), very negative -> c ~ 0 (off)."""
    c_raw = torch.full((m,), -8.0)
    c_raw[:n_active] = 8.0
    return {"sig_raw": torch.zeros(m), "o_raw": torch.zeros(m), "c_raw": c_raw}


# --- the counter ------------------------------------------------------------

@pytest.mark.parametrize("n", [0, 1, 2, 3, 7, 12])
def test_counts_primitives_above_the_threshold(n):
    assert M.active_primitive_count("cband12", _cband_rho(n)) == n


def test_threshold_is_the_declared_constant():
    assert C.ACTIVE_PRIMITIVE_ON_THRESHOLD == 0.5
    # c just above / just below 0.5 (c = sigmoid(c_raw), so raw 0 -> c = 0.5)
    rho = {"sig_raw": torch.zeros(12), "o_raw": torch.zeros(12),
           "c_raw": torch.tensor([0.1] * 3 + [-0.1] * 9)}
    assert M.active_primitive_count("cband12", rho) == 3


def test_band_readout_is_not_applicable():
    """One learnable band-pass is one primitive by construction, so a count of
    1 would be a misleading number rather than a measurement."""
    rho = {"mu": torch.tensor(0.0), "h_raw": torch.tensor(0.0),
           "k_raw": torch.tensor(0.0), "pi_raw": torch.tensor(2.0)}
    assert M.active_primitive_count("band", rho) is None
    assert M.active_primitive_bucket(None) == "n/a"


# --- the buckets ------------------------------------------------------------

@pytest.mark.parametrize("n,want", [(0, "1"), (1, "1"), (2, "2"), (3, ">=3"),
                                    (5, ">=3"), (12, ">=3")])
def test_bucketing_matches_the_where_a_strata(n, want):
    assert M.active_primitive_bucket(n) == want


def test_the_fragile_stratum_is_its_own_bucket():
    """The whole point: n=1 must not be pooled with n>=3."""
    assert M.active_primitive_bucket(1) != M.active_primitive_bucket(3)


# --- the reported column ----------------------------------------------------

def test_hi_lo_drop_column_exists_and_has_the_where_a_sign_convention():
    """Where-A reports the drop as positive when hi is worse than low."""
    gt = torch.zeros(8, 8)
    gt[2:6, 2:6] = 1.0
    perfect = M.sample_metrics(gt, gt, grid_pred=gt, grid_gt=gt)
    assert perfect["hi_lo_soft_iou_drop"] == pytest.approx(0.0, abs=1e-5)
    worse_hi = M.sample_metrics(gt * 0.5, gt, grid_pred=gt, grid_gt=gt)
    assert worse_hi["hi_lo_soft_iou_drop"] > 0


def test_summarise_reports_the_drop_median():
    rows = [{"render_mode": "local", "soft_iou": 0.8,
             "hi_lo_soft_iou_drop": 0.012 + 0.001 * i} for i in range(5)]
    assert M.summarise(rows)["hi_lo_soft_iou_drop_median"] == pytest.approx(0.014)


def test_strata_report_splits_the_hi_tier_by_primitive_count():
    rows = []
    for i in range(6):
        bucket = "1" if i < 3 else ">=3"
        rows.append({"render_mode": "local", "context": "generated",
                     "soft_iou": 0.70 if bucket == "1" else 0.80,
                     "hi_lo_soft_iou_drop": 0.012 if bucket == "1" else 0.006,
                     "active_primitive_bucket": bucket, "active_primitives":
                     1 if bucket == "1" else 4})
    rep = strata_report(rows)
    assert "active_primitive_bucket" in rep, "the stratum is not being reported"
    assert set(rep["active_primitive_bucket"]) == {"1", ">=3"}
    single = rep["active_primitive_bucket"]["1"]
    many = rep["active_primitive_bucket"][">=3"]
    # the Where-A signature: the single-primitive stratum degrades more
    assert single["hi_lo_soft_iou_drop_median"] > many["hi_lo_soft_iou_drop_median"]
    assert single["local_soft_iou_median"] < many["local_soft_iou_median"]


def test_extra_strata_key_is_declared_and_wired():
    assert C.EXTRA_STRATA_KEYS == ("active_primitive_bucket",)
    assert "active_primitive_bucket" not in C.STRATA_KEYS   # computed, not meta


# --- the ruling: reported, never gated --------------------------------------

def test_the_stratum_changes_no_gate_and_no_loss():
    import inspect

    from q3vl.whereb import losses

    assert not any("primitive" in k for k, _, _ in C.GATES)
    assert not any("primitive" in k for k, _ in C.SELECTION_ORDER)
    assert "primitive" not in inspect.getsource(losses).lower()
    assert C.MASK_IOU_W == 1.00 and C.MASK_BCE_W == 0.25 and C.MASK_BF1_W == 0.10


def test_attribution_note_records_the_risk_and_its_provenance():
    risk = M.ATTRIBUTION_NOTE["known_blind_spots"]["single_active_primitive_fragility"]
    assert "Where-A" in risk and "REVIEW-result" in risk
    assert "30.8%" in risk and "+0.012" in risk and "+0.006" in risk
    assert "n/a" in risk                      # the Band caveat
    assert "no loss, gate or training change" in risk
