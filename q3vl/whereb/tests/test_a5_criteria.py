"""Amendment A-5: the criteria side, aligned to the 2026-08-05 red lines.

Three things these tests hold:

1. **AUC is gone**, not merely unused -- no gate row, no metric key, no producer.
2. **The centre-prior baseline is correct and comparable**: same support, same
   matched-area top-k rule, and it is genuinely zero-parameter.
3. **The grid-level boundary F1 fixes the pathology the pixel-level 3px one
   has**: a random top-k field beats a centre prior on the pixel metric (the red
   line quotes 0.0394 vs 0.0327) because that metric mostly counts boundary
   *length*.  On the grid, where the top-k rule gives both fields exactly ``k``
   cells, the ordering has to come back.

The loss is deliberately NOT touched by any of this; `test_the_loss_is_untouched`
is what stops a well-meaning future edit from "finishing the job".
"""

from __future__ import annotations

import inspect
import math

import pytest
import torch

from q3vl.whereb import config as C
from q3vl.whereb import metrics as M
from q3vl.whereb import losses as L


# --- 1. AUC is gone ---------------------------------------------------------

def test_no_gate_row_mentions_auc():
    keys = [k for k, _, _ in C.GATES]
    assert not any("auc" in k.lower() for k in keys), keys
    assert not any("auc" in k.lower() for k, _ in C.SELECTION_ORDER)


def test_the_auc_producer_is_deleted_not_merely_unused():
    assert not hasattr(M, "auc_target"), (
        "auc_target still exists; the red line says new experiments must not "
        "produce the metric, not merely avoid gating on it"
    )
    assert not any("auc" in n.lower() for n in M.__all__)


def test_sample_metrics_emits_no_auc_key():
    gt = torch.zeros(6, 8)
    gt[2:4, 3:6] = 1.0
    out = M.sample_metrics(gt, gt, grid_pred=gt, grid_gt=gt)
    assert not any("auc" in k.lower() for k in out), sorted(out)


def test_summarise_and_attribution_carry_no_auc():
    rows = [{"soft_iou": 0.8, "grid_boundary_f1": 0.5, "grid_hard_iou": 0.6,
             "center_prior_hard_iou": 0.3, "center_prior_boundary_f1": 0.2,
             "render_mode": "local"}]
    assert not any("auc" in k.lower() for k in M.summarise(rows))
    blob = repr(M.ATTRIBUTION_NOTE).lower()
    assert "auc" not in blob


# --- 2. top-k thresholding and the centre prior -----------------------------

def test_topk_mask_selects_exactly_k_cells():
    torch.manual_seed(0)
    f = torch.randn(6, 8)
    for k in (0, 1, 7, 47, 48):
        m = M.topk_mask(f, k)
        assert int(m.sum()) == k, k
        assert set(m.unique().tolist()) <= {0.0, 1.0}


def test_topk_mask_picks_the_largest_values():
    f = torch.tensor([[1.0, 5.0], [3.0, 2.0]])
    m = M.topk_mask(f, 2)
    assert m.tolist() == [[0.0, 1.0], [1.0, 0.0]]


def test_topk_k_out_of_range_is_clamped_not_crashing():
    f = torch.randn(3, 3)
    assert int(M.topk_mask(f, 99).sum()) == 9
    assert int(M.topk_mask(f, -5).sum()) == 0


def test_gt_area_k_is_the_matched_area():
    gt = torch.zeros(5, 5)
    gt[1:3, 1:4] = 1.0                      # 6 cells
    assert M.gt_area_k(gt) == 6
    # soft edges count as region iff above the 0.5 midpoint
    gt2 = torch.full((4, 4), 0.4)
    assert M.gt_area_k(gt2) == 0


def test_center_prior_is_maximal_at_the_frame_centre():
    f = M.center_prior_field(9, 9)
    assert f.shape == (9, 9)
    assert torch.argmax(f.reshape(-1)).item() == 4 * 9 + 4      # dead centre
    # monotone: strictly decreasing away from the centre along a row
    row = f[4]
    assert row[4] > row[2] > row[0]


def test_center_prior_uses_the_true_aspect_ratio_convention():
    """Same coordinate convention as phi's geo5, so distance is isotropic in
    image space rather than in cell-index space."""
    from q3vl.where.phi import norm_coords

    gh, gw = 4, 12
    X, Y = norm_coords(gh, gw, dtype=torch.float32)
    want = -torch.sqrt(X ** 2 + Y ** 2)
    assert torch.allclose(M.center_prior_field(gh, gw), want)
    # a wide frame reaches further in x than in y, as the aspect ratio demands
    f = M.center_prior_field(4, 12)
    assert f[0, 0] < f[0, 5]


def test_center_prior_is_zero_parameter():
    """It must not depend on the prediction, the GT, or anything learned."""
    sig = inspect.signature(M.center_prior_field)
    assert set(sig.parameters) == {"grid_h", "grid_w", "device", "dtype"}
    a = M.center_prior_field(6, 8)
    b = M.center_prior_field(6, 8)
    assert torch.equal(a, b)


def test_center_prior_column_uses_the_same_k_as_the_prediction():
    gt = torch.zeros(8, 8)
    gt[1:4, 2:6] = 1.0                       # k = 12
    torch.manual_seed(0)
    pred = torch.rand(8, 8)
    out = M.sample_metrics(gt, gt, grid_pred=pred, grid_gt=gt)
    assert out["grid_k"] == 12
    # both binarisations own exactly k cells -> the comparison is like-for-like
    assert int(M.topk_mask(pred, out["grid_k"]).sum()) == 12
    assert int(M.topk_mask(M.center_prior_field(8, 8), out["grid_k"]).sum()) == 12
    assert "center_prior_hard_iou" in out and "center_prior_boundary_f1" in out


# --- 3. the discriminative-power regression ---------------------------------

GH, GW, UP = 32, 48, 16          # the real F_pre grid and the spec-5 upsample


def _square(cy, cx, r):
    m = torch.zeros(GH, GW)
    m[max(0, cy - r):cy + r, max(0, cx - r):cx + r] = 1.0
    return m


def _up(m):
    """Grid mask -> spec-5 pixel mask, the way the pipeline actually does it."""
    import torch.nn.functional as F

    return F.interpolate(m[None, None], scale_factor=UP, mode="nearest")[0, 0]


def _fields():
    """An off-centre subject and four candidate fields, all with the same k.

    A *centred* GT would make the centre prior correct by construction, which is
    why the subject is off-centre -- that is also the realistic case.
    """
    torch.manual_seed(0)
    gt = _square(9, 12, 5)
    k = M.gt_area_k(gt)
    return gt, k, {
        "good": _square(10, 13, 5),                       # right subject, 1 cell off
        "half": (lambda m: (m.__setitem__((slice(9, 14), slice(7, 17)), 0.0), m)[1])(
            gt.clone()),                                  # partially correct
        "random": M.topk_mask(torch.rand(GH, GW), k),     # scattered, 3x the perimeter
        "prior": M.topk_mask(M.center_prior_field(GH, GW), k),
    }


def test_pixel_3px_boundary_f1_cannot_reward_a_near_perfect_field():
    """Why A-5 drops the pixel-level 3px criterion.

    At the spec-5 resolution a 3px tolerance is 3/16 = **0.19 grid cells**, so a
    field that is off by a single grid cell -- an excellent field -- has its
    entire boundary outside the tolerance and collects almost nothing, while a
    partially-wrong field whose boundary happens to coincide with the GT's scores
    an order of magnitude higher.  Measured here: good 0.0220 vs half 0.5827.
    The red line quotes the same failure in its own setting (a random top-k
    scoring 0.0394 against a centre prior's 0.0327): the metric rewards boundary
    coincidence and boundary *length*, not shape agreement.
    """
    gt, _k, f = _fields()
    pix = {n: M.boundary_f1(_up(v), _up(gt), tol_px=3) for n, v in f.items()}
    assert pix["half"] > 10 * pix["good"], (
        f"the pathology did not reproduce: good={pix['good']:.4f} "
        f"half={pix['half']:.4f}. If it is genuinely gone, A-5's rationale needs "
        "re-checking rather than this test being deleted."
    )
    # and a shredded field is within a whisker of the near-perfect one
    assert pix["good"] / max(pix["random"], 1e-9) < 3.0


def test_grid_boundary_f1_separates_a_good_field_from_noise():
    """What A-5 puts in the gate instead: on the grid, with the matched-area
    top-k rule, the near-perfect field beats scattered noise by ~28x (measured)
    instead of 1.6x, and the ordering good > half > random > centre-prior holds."""
    gt, _k, f = _fields()
    g = {n: M.grid_boundary_f1(v, gt) for n, v in f.items()}
    assert g["good"] > g["half"] > g["random"] >= g["prior"], g
    assert g["good"] / max(g["random"], 1e-9) > 10.0, g
    # hard-IoU, the coverage column, must agree about the ordering
    h = {n: M.hard_iou(v, gt) for n, v in f.items()}
    assert h["good"] > h["half"] > h["random"] >= h["prior"], h


def test_no_single_column_is_sufficient_which_is_why_there_are_three():
    """A compact-but-displaced field and scattered noise are both bad, and
    boundary F1 alone cannot order them meaningfully -- the coverage column and
    the centre-prior column are what make the criterion complete."""
    gt, k, _f = _fields()
    displaced = _square(22, 34, 5)
    assert M.grid_boundary_f1(displaced, gt) == 0.0
    assert M.hard_iou(displaced, gt) == 0.0
    # the centre prior is the calibrated "no information" reference for both
    prior = M.topk_mask(M.center_prior_field(GH, GW), k)
    assert M.hard_iou(prior, gt) < 0.2


def test_grid_boundary_f1_endpoints():
    gt = _square(9, 12, 5)
    assert M.grid_boundary_f1(gt, gt) == pytest.approx(1.0)
    empty = torch.zeros_like(gt)
    assert M.grid_boundary_f1(empty, empty) == 1.0        # no boundary either side
    assert M.grid_boundary_f1(empty, gt) == 0.0


def test_grid_boundary_f1_tolerance_is_in_cells_not_pixels():
    assert C.GRID_BOUNDARY_TOL_CELLS == 1
    sig = inspect.signature(M.grid_boundary_f1)
    assert "tol_cells" in sig.parameters
    assert "tol_px" not in sig.parameters


# --- paired delta -----------------------------------------------------------

def test_paired_delta_detects_a_real_margin():
    a = [0.60, 0.62, 0.58, 0.65, 0.61, 0.59, 0.63, 0.60]
    b = [0.30, 0.33, 0.29, 0.35, 0.31, 0.28, 0.34, 0.30]
    d = M.paired_delta(a, b, seed=0)
    assert d["n"] == 8
    assert d["delta"] > 0.25
    assert d["p_value"] <= 0.05
    assert d["ci95"][0] > 0


def test_paired_delta_reports_no_margin_when_there_is_none():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(1)
    a = torch.randn(64, generator=g).tolist()
    b = torch.randn(64, generator=g).tolist()
    d = M.paired_delta(a, b, seed=0)
    assert d["p_value"] > 0.05
    assert d["ci95"][0] < 0 < d["ci95"][1]


def test_paired_delta_handles_an_empty_pairing():
    d = M.paired_delta([], [])
    assert d["n"] == 0 and d["delta"] is None and d["p_value"] is None


def test_summarise_emits_the_centre_prior_delta_and_p():
    rows = []
    for i in range(12):
        rows.append({"render_mode": "local", "soft_iou": 0.8,
                     "grid_hard_iou": 0.60 + 0.01 * (i % 3),
                     "center_prior_hard_iou": 0.30,
                     "grid_boundary_f1": 0.50, "center_prior_boundary_f1": 0.20})
    s = M.summarise(rows)
    assert s["center_prior_delta_hard_iou"] > 0.25
    assert s["center_prior_delta_hard_iou_p"] <= 0.05
    assert s["center_prior_delta_boundary_f1"] == pytest.approx(0.30)
    assert len(s["center_prior_delta_hard_iou_ci95"]) == 2


# --- gates ------------------------------------------------------------------

def test_the_gate_table_matches_amendment_a5():
    got = {k: (op, thr) for k, op, thr in C.GATES}
    assert got == {
        "local_soft_iou_median": (">=", 0.75),
        "soft_iou_vs_oracle_ratio": (">=", 0.85),
        "local_soft_iou_p10": (">=", 0.55),
        "grid_boundary_f1_vs_oracle_ratio": (">=", 0.75),
        "center_prior_delta_hard_iou": (">", 0.0),
        "center_prior_delta_hard_iou_p": ("<=", 0.05),
        "instruction_shuffle_iou_drop": (">=", 0.20),
        "s_std_ratio_median": (">=", 0.60),
        "global_soft_iou": (">=", 0.98),
        "gt_generated_iou_gap": ("<=", 0.05),
    }
    assert C.SELECTION_ORDER[1][0] == "grid_boundary_f1"


def test_strictly_greater_than_gate_is_honoured():
    """`center_prior_delta_hard_iou > 0` must not pass on exactly zero."""
    base = {k: (thr + 1.0 if op == ">=" else thr - 1.0) for k, op, thr in C.GATES}
    base["center_prior_delta_hard_iou_p"] = 0.01
    base["gt_generated_iou_gap"] = 0.0
    base["center_prior_delta_hard_iou"] = 0.0
    g = M.evaluate_gates(base)
    row = next(r for r in g["rows"] if r["metric"] == "center_prior_delta_hard_iou")
    assert not row["passed"], "a zero margin over the centre prior must fail"
    base["center_prior_delta_hard_iou"] = 1e-6
    assert next(r for r in M.evaluate_gates(base)["rows"]
                if r["metric"] == "center_prior_delta_hard_iou")["passed"]


def test_three_negative_controls_are_declared():
    assert C.INSTRUCTION_NEGATIVE_CONTROLS == (
        "shuffled", "irrelevant_words", "fixed_phrase")
    assert C.TOPK_RULE == "match_gt_area"


# --- the loss must NOT have moved -------------------------------------------

def test_the_loss_is_untouched_by_a5():
    """Protocol 9.5/10.4 forbid changing a loss after the fact.

    A-5 revises criteria only.  `L_mask` keeps its PIXEL-level 3px boundary term
    even though the *criterion* is now grid-level -- and that divergence is
    deliberate, because it makes the criterion something the training does not
    directly optimise.
    """
    assert C.MASK_IOU_W == 1.00 and C.MASK_BCE_W == 0.25 and C.MASK_BF1_W == 0.10
    assert C.BOUNDARY_TOL_PX == 3 and C.BOUNDARY_KERNEL == 3
    sig = inspect.signature(L.boundary_f1_loss)
    assert "tol_px" in sig.parameters
    assert sig.parameters["tol_px"].default == 3
    # the loss still uses the pixel-level surrogate, not the new grid metric
    src = inspect.getsource(L.mask_loss)
    assert "boundary_f1_loss" in src and "grid_boundary_f1" not in src


def test_loss_and_criterion_are_different_functions_on_purpose():
    torch.manual_seed(0)
    gt = _square(9, 12, 5)
    k = M.gt_area_k(gt)
    pred = M.topk_mask(torch.rand(GH, GW), k)
    loss_val = float(L.boundary_f1_loss(pred[None, None], gt[None, None], tol_px=3))
    crit = M.grid_boundary_f1(pred, gt)
    assert 0.0 <= loss_val <= 1.0 and 0.0 <= crit <= 1.0
    assert abs((1.0 - loss_val) - crit) > 1e-6, (
        "the criterion collapsed onto the loss; A-5 requires them to differ"
    )
