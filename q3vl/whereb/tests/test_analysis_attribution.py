"""WEVAL-1 attribution, tables and the rules the report refuses to break.

The decision tree is the part of this tool that turns numbers into a sentence
("the bottleneck is the generated context"), so each branch is exercised in
isolation: one row engineered to trip exactly one mechanism, and a clean row that
must trip none.
"""

from __future__ import annotations

import pytest
import torch

from q3vl.whereb.analysis.attribution import (
    FIELD_MECHANISMS, MECHANISMS, PRIORITY, AnalysisThresholds, attribute_sample,
    mechanism_summary,
)
from q3vl.whereb.analysis.report import render_report
from q3vl.whereb.analysis.tables import (
    LOW_CONFIDENCE_N, overall_table, per_class_tables, worst_class,
)


def clean_row(**kw):
    """A sample that is fine on every measurable axis."""
    row = {
        "sample_id": "s0", "context": "generated", "render_mode": "local",
        "soft_iou": 0.82, "grid_soft_iou": 0.82, "grid_hard_iou": 0.80,
        "grid_boundary_f1": 0.55, "center_prior_hard_iou": 0.40,
        "center_prior_boundary_f1": 0.20, "oracle_soft_iou": 0.95,
        "oracle_grid_hard_iou": 0.95, "oracle_grid_boundary_f1": 0.98,
        "s_std_ratio": 0.90, "hi_lo_soft_iou_drop": 0.001,
        "pred_mean": 0.30, "gt_mean": 0.30, "format_failure": False,
        "winner_confidence": "normal", "upscaled": False, "build": "l1",
    }
    row.update(kw)
    return row


def clean_fields(**kw):
    f = {"w_dir_cos": 0.9, "iou_pred": 0.82, "iou_oracle_s_pred_rho": 0.83,
         "iou_pred_s_oracle_rho": 0.84, "active_primitives": 4}
    f.update(kw)
    return f


# --- one branch at a time ---------------------------------------------------

def test_a_clean_sample_is_unexplained_not_labelled():
    a = attribute_sample(clean_row(), gt_row=clean_row(), fields=clean_fields())
    assert a["mechanisms"] == ["unexplained"]
    assert a["primary"] == "unexplained"
    assert a["not_tested"] == []


@pytest.mark.parametrize("kw,mech", [
    ({"format_failure": True}, "format_failure"),
    ({"oracle_soft_iou": 0.4}, "oracle_ceiling"),
    ({"s_std_ratio": 0.05}, "s_collapse"),
    ({"hi_lo_soft_iou_drop": 0.2}, "upsample_collapse"),
    ({"pred_mean": 0.9, "gt_mean": 0.3}, "area_mismatch"),
    ({"grid_hard_iou": 0.1, "center_prior_hard_iou": 0.5}, "below_center_prior"),
])
def test_each_eval_only_mechanism_fires_on_its_own_row(kw, mech):
    a = attribute_sample(clean_row(**kw), gt_row=clean_row(), fields=clean_fields())
    assert mech in a["mechanisms"], a
    others = set(a["mechanisms"]) - {mech}
    assert not others, f"{mech} row also tripped {others}"


def test_context_quality_needs_the_gt_row_to_be_good_and_better():
    bad_gen = clean_row(soft_iou=0.40)
    good_gt = clean_row(soft_iou=0.80)
    a = attribute_sample(bad_gen, gt_row=good_gt, fields=clean_fields())
    assert "context_quality" in a["mechanisms"]
    assert a["evidence"]["context_gap"] == pytest.approx(0.40)

    # both contexts equally bad -> the context is not what differs
    b = attribute_sample(bad_gen, gt_row=clean_row(soft_iou=0.42),
                         fields=clean_fields())
    assert "context_quality" not in b["mechanisms"]

    # GT itself is bad -> the comparison carries no information
    c = attribute_sample(clean_row(soft_iou=0.10), gt_row=clean_row(soft_iou=0.30),
                         fields=clean_fields())
    assert "context_quality" not in c["mechanisms"]


def test_the_swap_test_separates_s_errors_from_rho_errors():
    row = clean_row(soft_iou=0.30)
    s_bad = attribute_sample(row, gt_row=row,
                             fields=clean_fields(iou_pred=0.30,
                                                 iou_oracle_s_pred_rho=0.85,
                                                 iou_pred_s_oracle_rho=0.31))
    assert "s_error" in s_bad["mechanisms"] and "rho_error" not in s_bad["mechanisms"]

    rho_bad = attribute_sample(row, gt_row=row,
                               fields=clean_fields(iou_pred=0.30,
                                                   iou_oracle_s_pred_rho=0.32,
                                                   iou_pred_s_oracle_rho=0.88))
    assert "rho_error" in rho_bad["mechanisms"]
    assert "s_error" not in rho_bad["mechanisms"]


def test_wrong_direction_is_caught_by_the_cosine():
    a = attribute_sample(clean_row(), gt_row=clean_row(),
                         fields=clean_fields(w_dir_cos=0.05))
    assert "s_direction" in a["mechanisms"]


def test_single_primitive_needs_both_the_count_and_a_hi_tier_loss():
    fired = attribute_sample(clean_row(hi_lo_soft_iou_drop=0.01), gt_row=clean_row(),
                             fields=clean_fields(active_primitives=1))
    assert "single_primitive" in fired["mechanisms"]
    # a single primitive that upsamples fine is not a failure
    quiet = attribute_sample(clean_row(hi_lo_soft_iou_drop=-0.01), gt_row=clean_row(),
                             fields=clean_fields(active_primitives=1))
    assert "single_primitive" not in quiet["mechanisms"]


# --- not_tested is not absence ---------------------------------------------

def test_without_fields_the_field_mechanisms_are_not_tested_not_absent():
    a = attribute_sample(clean_row(), gt_row=clean_row(), fields=None)
    assert FIELD_MECHANISMS <= set(a["not_tested"])
    assert not (FIELD_MECHANISMS & set(a["mechanisms"]))


def test_a_sample_without_an_oracle_cannot_be_blamed_on_the_ceiling():
    row = clean_row()
    row.pop("oracle_soft_iou")
    a = attribute_sample(row, gt_row=clean_row(), fields=clean_fields())
    assert "oracle_ceiling" in a["not_tested"]
    assert "oracle_ceiling" not in a["mechanisms"]


def test_band_readout_reports_single_primitive_as_not_tested():
    a = attribute_sample(clean_row(), gt_row=clean_row(),
                         fields=clean_fields(active_primitives=None))
    assert "single_primitive" in a["not_tested"]


# --- priority ---------------------------------------------------------------

def test_primary_is_the_most_upstream_mechanism_that_fired():
    a = attribute_sample(
        clean_row(soft_iou=0.2, oracle_soft_iou=0.3, s_std_ratio=0.01,
                  pred_mean=0.9, gt_mean=0.3),
        gt_row=clean_row(soft_iou=0.8), fields=clean_fields(w_dir_cos=0.0))
    assert set(a["mechanisms"]) >= {"oracle_ceiling", "context_quality",
                                    "s_direction", "s_collapse", "area_mismatch"}
    assert a["primary"] == "oracle_ceiling"
    assert PRIORITY.index("oracle_ceiling") < PRIORITY.index("s_collapse")


def test_priority_covers_every_mechanism_exactly_once():
    assert sorted(PRIORITY) == sorted(MECHANISMS)
    assert PRIORITY[-1] == "unexplained"


# --- summary ----------------------------------------------------------------

def test_mechanism_summary_shares_and_bottleneck():
    labelled = [
        attribute_sample(clean_row(oracle_soft_iou=0.3), gt_row=clean_row()),
        attribute_sample(clean_row(oracle_soft_iou=0.3), gt_row=clean_row()),
        attribute_sample(clean_row(s_std_ratio=0.01), gt_row=clean_row()),
    ]
    s = mechanism_summary(labelled)
    assert s["n_tail"] == 3
    assert s["primary_counts"]["oracle_ceiling"] == 2
    assert s["primary_share"]["oracle_ceiling"] == pytest.approx(2 / 3)
    assert sum(s["primary_share"].values()) == pytest.approx(1.0)
    assert s["bottleneck"] == "oracle_ceiling"
    # field mechanisms were untestable on all three, and say so
    assert s["not_tested_counts"]["s_error"] == 3


def test_thresholds_are_data_not_code():
    strict = AnalysisThresholds(oracle_ceiling=0.99)
    a = attribute_sample(clean_row(), gt_row=clean_row(), thresholds=strict)
    assert "oracle_ceiling" in a["mechanisms"]
    assert AnalysisThresholds().to_dict()["oracle_ceiling"] == 0.70


# --- tables -----------------------------------------------------------------

def _rows(n_a: int, n_b: int):
    rows = [clean_row(sample_id=f"a{i}", soft_iou=0.8) for i in range(n_a)]
    rows += [clean_row(sample_id=f"b{i}", soft_iou=0.2, grid_hard_iou=0.2)
             for i in range(n_b)]
    return rows


def _labels(rows):
    return {r["sample_id"]: {"area": "large" if r["sample_id"].startswith("a")
                             else "small"} for r in rows}


def test_per_class_numbers_come_from_the_same_aggregator_as_the_board():
    from q3vl.whereb.metrics import summarise

    rows = _rows(30, 25)
    t = per_class_tables(rows, _labels(rows), ["area"])
    sub = [r for r in rows if r["sample_id"].startswith("b")]
    assert t["area"]["small"]["local_soft_iou_median"] == \
        summarise(sub)["local_soft_iou_median"]


def test_every_class_table_carries_the_centre_prior_columns():
    rows = _rows(30, 25)
    t = per_class_tables(rows, _labels(rows), ["area"])
    for cell in t["area"].values():
        assert cell["center_prior_hard_iou"] is not None
        assert "center_prior_delta_hard_iou" in cell
        assert "center_prior_delta_hard_iou_p" in cell


def test_no_output_key_mentions_auc():
    rows = _rows(30, 25)
    t = per_class_tables(rows, _labels(rows), ["area"])
    keys = {k for cell in t["area"].values() for k in cell}
    keys |= set(overall_table(rows))
    assert not [k for k in keys if "auc" in k.lower()], "AUC is banned campaign-wide"


def test_small_classes_are_flagged_and_never_win_the_worst_class_vote():
    rows = _rows(30, 3)
    t = per_class_tables(rows, _labels(rows), ["area"])
    assert t["area"]["small"]["low_confidence"] is True
    assert t["area"]["large"]["low_confidence"] is False
    # "small" has the worse median but only 3 samples
    assert t["area"]["small"]["local_soft_iou_median"] < \
        t["area"]["large"]["local_soft_iou_median"]
    assert worst_class(t["area"]) == "large"
    assert worst_class(t["area"], min_n=2) == "small"
    assert LOW_CONFIDENCE_N == 20


def test_unlabelled_rows_are_dropped_loudly_not_silently():
    rows = _rows(5, 5)
    labels = _labels(rows)
    labels.pop("a0")
    kept = per_class_tables(rows, labels, ["area"])
    assert sum(c["n"] for c in kept["area"].values()) == 9
    shown = per_class_tables(rows, labels, ["area"], include_unlabelled=True)
    assert shown["area"]["unlabelled"]["n"] == 1


# --- the top-k rule the columns rest on ------------------------------------

def test_matched_area_topk_is_what_the_hard_iou_column_measures():
    """Red line: thresholding is always top-k matching the GT area.

    Demonstrated rather than asserted: a per-field 0.5 threshold on the very same
    field gives a *different* number, which is exactly why a tuned threshold is
    banned -- it lets a field buy coverage it did not earn.
    """
    from q3vl.whereb.metrics import (
        gt_area_k, hard_iou, sample_metrics, topk_mask,
    )

    gt = torch.zeros(8, 8)
    gt[2:6, 2:6] = 1.0                          # 16 cells
    # a field that is confidently ON over 25 cells, brightest at the top-left --
    # so a 0.5 threshold claims all 25 while the matched-area rule must pick the
    # 16 brightest, which sit off the GT
    field = torch.full((8, 8), 0.05)
    yy, xx = torch.meshgrid(torch.arange(8.0), torch.arange(8.0), indexing="ij")
    field[1:6, 1:6] = (0.9 - 0.01 * (yy + xx))[1:6, 1:6]

    met = sample_metrics(field, gt, grid_pred=field, grid_gt=gt)
    k = gt_area_k(gt)
    assert met["grid_k"] == k == 16
    assert met["grid_hard_iou"] == pytest.approx(
        hard_iou(topk_mask(field, k), gt > 0.5))

    tuned = hard_iou((field > 0.5).float(), gt > 0.5)
    assert int((field > 0.5).sum()) == 25, "the tuned threshold claims more cells"
    assert tuned > met["grid_hard_iou"] + 0.1, (
        "a per-field threshold buys coverage the matched-area rule denies it; "
        "that gap is why the rule is a red line")


# --- the report refuses to render a red-line violation ---------------------

def _payload(cell: dict):
    return {
        "meta": {"arm": "W00", "step": "1", "split": "V_where", "eval_dir": "x",
                 "main_context": "generated", "n_rows": 1, "n_contexts": 1,
                 "n_masked": 1, "taxonomy": __import__(
                     "q3vl.whereb.analysis.taxonomy", fromlist=["TaxonomyConfig"]
                 ).TaxonomyConfig().to_dict(),
                 "thresholds": AnalysisThresholds().to_dict(),
                 "tail_cut": AnalysisThresholds().tail_soft_iou,
                 "tail_decile": AnalysisThresholds().tail_decile,
                 "tail_decile_cut": 0.12, "n_viz_samples": 0},
        "dimensions": ["area"],
        "per_class": {"generated": {"area": {"large": cell}}, "gt": {}},
        "class_distribution": {"area": {"large": 1}},
        "geometry_stats": {},
        "overall": {"generated": {"n_local": 1, "n_global": 0}},
        "worst_class": {"area": "large"},
        "tail": [],
        "tail_summary": mechanism_summary([]),
        "population_summary": {},
        "conclusions": [],
    }


def test_both_tail_cuts_are_declared_and_the_ceiling_is_marked_provisional():
    """Main-agent ruling 2026-08-10: report the absolute cut AND the decile."""
    t = AnalysisThresholds()
    assert t.tail_soft_iou == 0.30
    assert t.tail_decile == 0.10
    import inspect

    from q3vl.whereb.analysis import attribution as A

    src = inspect.getsource(A.AnalysisThresholds)
    assert "PROVISIONAL" in src, "the ceiling's provisional status must be stated"
    # the number itself is what a report has to quote, so it is in the dump
    assert t.to_dict()["oracle_ceiling"] == 0.70
    assert t.to_dict()["tail_decile"] == 0.10


def test_report_carries_the_auxiliary_decile_column_and_the_provisional_note():
    cell = {"center_prior_hard_iou": 0.4, "local_soft_iou_median": 0.5}
    payload = _payload(cell)
    payload["tail_summary_decile"] = mechanism_summary(
        [attribute_sample(clean_row(oracle_soft_iou=0.3), gt_row=clean_row())])
    md = render_report(payload)
    assert "最差 10%" in md
    assert "provisional" in md.lower()
    assert "工程判断" in md, "the priority order must be declared as a judgement"


def test_report_still_renders_without_a_decile_summary():
    """Older payloads (and the unit fixtures) must not crash the renderer."""
    md = render_report(_payload({"center_prior_hard_iou": 0.4}))
    assert "长尾归因" in md


def test_report_refuses_a_table_without_the_centre_prior_column():
    with pytest.raises(AssertionError, match="centre-prior"):
        render_report(_payload({"local_soft_iou_median": 0.5}))


def test_report_refuses_an_auc_column():
    with pytest.raises(AssertionError, match="AUC"):
        render_report(_payload({"center_prior_hard_iou": 0.4, "auc_target": 0.9}))
