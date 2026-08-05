"""Protocol 5.6: gate arithmetic and the lexicographic selection rule."""

from __future__ import annotations

import torch

from q3vl.whereb.config import GATE_FAILED_TAG
from q3vl.whereb.metrics import (
    arm_metrics,
    center_prior_field,
    grid_boundary_f1,
    hard_iou,
    topk_mask,
    boundary_f1,
    evaluate_gates,
    lexicographic_best,
    percentile,
    sample_metrics,
    summarise,
)

PASSING = {
    "local_soft_iou_median": 0.80,
    "soft_iou_vs_oracle_ratio": 0.90,
    "local_soft_iou_p10": 0.60,
    "grid_boundary_f1_vs_oracle_ratio": 0.80,
    "center_prior_delta_hard_iou": 0.22,
    "center_prior_delta_hard_iou_p": 0.01,
    "instruction_shuffle_iou_drop": 0.25,
    "s_std_ratio_median": 0.70,
    "global_soft_iou": 0.99,
    "gt_generated_iou_gap": 0.03,
}


def test_percentile_endpoints():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(xs, 0.0) == 1.0
    assert percentile(xs, 1.0) == 5.0
    assert percentile(xs, 0.5) == 3.0
    assert percentile([], 0.5) is None


def test_boundary_f1_metric_is_one_for_an_exact_match():
    y = torch.zeros(24, 24)
    y[6:18, 6:18] = 1.0
    assert abs(boundary_f1(y, y) - 1.0) < 1e-6
    assert boundary_f1(torch.ones(12, 12), torch.ones(12, 12)) == 1.0


def test_sample_metrics_reports_the_oracle_ratio_inputs():
    t = torch.zeros(1, 1, 20, 20)
    t[..., 5:15, 5:15] = 1.0
    m = t.clone()
    met = sample_metrics(m, t, s_pred=torch.randn(30), s_star=torch.randn(30),
                         m_oracle=t.clone())
    assert abs(met["soft_iou"] - 1.0) < 1e-5
    assert abs(met["oracle_soft_iou"] - 1.0) < 1e-5
    assert "s_std_ratio" in met
    assert "auc_target" not in met          # amendment A-5


def test_summarise_splits_local_and_global():
    rows = [
        {"soft_iou": 0.8, "grid_boundary_f1": 0.7,          "render_mode": "local", "oracle_soft_iou": 0.9,
         "grid_boundary_f1": 0.7, "grid_hard_iou": 0.6,
         "center_prior_hard_iou": 0.3, "center_prior_boundary_f1": 0.2,
         "s_std_ratio": 0.7},
        {"soft_iou": 0.6, "grid_boundary_f1": 0.5,          "render_mode": "local", "oracle_soft_iou": 0.9,
         "grid_boundary_f1": 0.5, "grid_hard_iou": 0.4,
         "center_prior_hard_iou": 0.3, "center_prior_boundary_f1": 0.2,
         "s_std_ratio": 0.5},
        {"soft_iou": 0.99, "grid_boundary_f1": 1.0, "render_mode": "global"},
    ]
    s = summarise(rows)
    assert s["n_local"] == 2 and s["n_global"] == 1
    assert s["global_soft_iou"] == 0.99
    assert s["local_soft_iou_median"] in (0.6, 0.8)
    assert abs(s["soft_iou_vs_oracle_ratio"] - 0.8 / 0.9) < 1e-6 or \
        abs(s["soft_iou_vs_oracle_ratio"] - 0.6 / 0.9) < 1e-6


def test_arm_metrics_derives_the_two_cross_context_gates():
    per = {
        "generated": {"local_soft_iou_median": 0.80},
        "gt": {"local_soft_iou_median": 0.83},
        "shuffled": {"local_soft_iou_median": 0.52},
        "null": {"local_soft_iou_median": 0.40},
    }
    m = arm_metrics(per)
    assert abs(m["gt_generated_iou_gap"] - 0.03) < 1e-9
    assert abs(m["instruction_shuffle_iou_drop"] - 0.28) < 1e-9
    assert m["main_context"] == "generated"
    assert set(m["per_context"]) == set(per)


def test_all_gates_pass_on_a_passing_board():
    g = evaluate_gates(PASSING)
    assert g["passed"] and g["n_passed"] == g["n_gates"] == len(PASSING)
    assert g["tag"] is None


def test_a_single_failure_fails_the_board_and_tags_it():
    bad = dict(PASSING, gt_generated_iou_gap=0.09)
    g = evaluate_gates(bad)
    assert not g["passed"] and g["tag"] == GATE_FAILED_TAG
    row = next(r for r in g["rows"] if r["metric"] == "gt_generated_iou_gap")
    assert row["op"] == "<=" and not row["passed"]


def test_a_missing_metric_is_a_failure_not_a_pass():
    partial = {k: v for k, v in PASSING.items() if k != "grid_boundary_f1_vs_oracle_ratio"}
    g = evaluate_gates(partial)
    assert not g["passed"]
    assert next(r for r in g["rows"]
                if r["metric"] == "grid_boundary_f1_vs_oracle_ratio")["reason"] == "missing"


def test_boundary_values_are_inclusive():
    exact = dict(PASSING, local_soft_iou_median=0.75, gt_generated_iou_gap=0.05)
    assert evaluate_gates(exact)["passed"]


def test_lexicographic_selection_follows_the_protocol_order():
    cands = [
        dict(PASSING, arm="W01", local_soft_iou_median=0.80, grid_boundary_f1=0.60,
             local_soft_iou_p10=0.60, n_trainable_params=10),
        dict(PASSING, arm="W02", local_soft_iou_median=0.80, grid_boundary_f1=0.70,
             local_soft_iou_p10=0.55, n_trainable_params=20),
        dict(PASSING, arm="W03", local_soft_iou_median=0.79, grid_boundary_f1=0.99,
             local_soft_iou_p10=0.70, n_trainable_params=5),
    ]
    best = lexicographic_best(cands)
    assert best["best"]["arm"] == "W02"          # ties on IoU -> boundary F1 wins
    assert best["ranking"][0] == "W02"
    assert best["gate"]["passed"] and best["tag"] is None


def test_parameter_count_breaks_a_full_tie():
    a = dict(PASSING, arm="A", grid_boundary_f1=0.7, n_trainable_params=100)
    b = dict(PASSING, arm="B", grid_boundary_f1=0.7, n_trainable_params=50)
    assert lexicographic_best([a, b])["best"]["arm"] == "B"


def test_gate_failure_still_selects_but_tags_the_winner():
    cands = [dict(PASSING, arm="W01", local_soft_iou_median=0.40, grid_boundary_f1=0.3),
             dict(PASSING, arm="W02", local_soft_iou_median=0.30, grid_boundary_f1=0.3)]
    out = lexicographic_best(cands)
    assert out["best"]["arm"] == "W01"
    assert out["tag"] == GATE_FAILED_TAG
    assert not out["any_candidate_passed_all_gates"]
