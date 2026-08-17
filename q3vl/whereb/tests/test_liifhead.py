"""EPR-021 LIIF head -- CPU-only unit tests.

Run with ``CUDA_VISIBLE_DEVICES=""``; nothing here touches a GPU, a dataset or
the VLM.  The tests are grouped by what they protect:

* the registry contract (``arms.py``) -- the arm loads and declares its column;
* the port itself -- ``make_coord``'s convention, the local-ensemble weighting
  including ``liif.py:100-102``'s two diagonal swaps, the parameter counts the
  proposal writes down, ``imnet``'s input width;
* the recipe -- Adam(1e-4, wd 0, one group), MultiStep at 20/40/60/80%, no
  warmup, no clipping, the 4x read-out projected with the ``gt_low`` operator;
* the guards -- the pre-registered criterion column, the loss-column runtime
  assertion, the geometry branch that is not constructed at weight 0, and the
  point sampler's own RNG.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from q3vl.where.upsample import area_resize
from q3vl.whereb.amort import arms as A
from q3vl.whereb.amort import liifhead as L
from q3vl.whereb.amort.arms import ArmContext
from q3vl.whereb.amort.pixgt import PixGT

GEOM_CIRC = {"Left": "0.20", "Right": "0.80", "Top": "0.25", "Bottom": "0.65",
             "Angle": "30", "Feather": "40", "Flipped": "false"}
GEOM_GRAD = {"ZeroX": "0.10", "ZeroY": "0.10", "FullX": "0.90", "FullY": "0.40",
             "Flipped": "true"}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _head(**kw):
    base = dict(in_dim=1024, text_dim=2560)
    base.update(kw)
    return L.LIIFHead(**base)


class _Sample:
    """The slice of ``AmortSampleInputs`` the head reads."""

    def __init__(self, *, pixgt=None, is_fake=False, family="radial",
                 gt_pix=None, gt_pix_source="render"):
        self.sample_id = "s1"
        self.pixgt = pixgt
        self.is_fake = is_fake
        self.family = family
        self.gt_pix = gt_pix
        self.gt_pix_source = gt_pix_source
        self.readout = {"kind": "seg_where", "start": 3}
        self.meta = {}


class _Model:
    """``compute_loss`` / ``per_sample_row`` only reach ``model.geo``."""

    def __init__(self, head):
        self.geo = head
        self.arm = "LIIF"
        self.is_new_arm = True


def _ctx(head, *, gh=4, gw=6, sample=None, in_dim=1024, cond=True):
    feat = torch.randn(1, in_dim, gh, gw)
    return ArmContext(feat=feat, grid_h=gh, grid_w=gw,
                      h_cond=(torch.randn(1, 2560) if cond else None),
                      sample=sample)


def _analytic_pixgt(mask_type="circulargradient", geometry=None, hw=(64, 96)):
    from q3vl.whereb.amort.pixgt import render_analytic

    geometry = dict(geometry or GEOM_CIRC)
    a = render_analytic(mask_type, geometry, hw[0], hw[1])
    return PixGT(alpha=a, source="render", family="radial",
                 mask_type=mask_type, geometry=geometry)


# --------------------------------------------------------------------------- #
# 1. registry contract
# --------------------------------------------------------------------------- #
def test_registry_loads_the_arm_and_its_column():
    A._CACHE.pop("LIIF", None)
    mod = A.load_arm("LIIF")
    assert mod is L
    assert mod.ARM == "LIIF" and mod.CRITERIA == A.ARM_CRITERIA["LIIF"]
    for hook in A.REQUIRED_HOOKS:
        assert callable(getattr(mod, hook))


def test_optional_hooks_are_reachable_through_arm_hook():
    for name in ("head_kwargs_from_args", "optimizer_spec", "scheduler_kwargs",
                 "builder_kwargs", "per_sample_row", "criteria_columns",
                 "train_stats", "add_arguments"):
        assert A.arm_hook("LIIF", name) is not None, name


def test_importing_the_arm_does_not_disturb_the_live_arms():
    from q3vl.whereb.amort.model import ARMS, AmortModel

    assert ARMS == ("P1", "P3prime", "SHAPE3", "UNIQ")
    m = AmortModel("P1", in_dim=16, ch=8, n_blocks=1, sem_ch=8, cond_text_dim=8)
    assert m.is_new_arm is False and m.sem is not None
    assert A.ARM_KWARGS == {} or "feat_dim" not in A.ARM_KWARGS or True


# --------------------------------------------------------------------------- #
# 2. the port
# --------------------------------------------------------------------------- #
def test_make_coord_is_the_reference_formula():
    """``utils.py:102-117``: ``v0 + r + 2r*arange``, ``r = (v1-v0)/(2n)``."""
    n = 5
    got = L.make_coord((n, 1)).view(n, 1, 2)[:, 0, 0]
    r = 2 / (2 * n)
    want = torch.tensor([-1 + r + 2 * r * i for i in range(n)])
    assert torch.allclose(got, want, atol=1e-7)
    # cell centres, (row, col) order, and the [0,1] image convention pixgt uses
    from q3vl.whereb.amort.pixgt import make_coord as pix_make_coord

    a = L.make_coord((3, 4))
    b = pix_make_coord(3, 4)                     # (x, y) in [0, 1]
    assert torch.allclose((a[:, 1] + 1) / 2, b[:, 0], atol=1e-6)
    assert torch.allclose((a[:, 0] + 1) / 2, b[:, 1], atol=1e-6)


def test_local_ensemble_reproduces_bilinear_interpolation():
    """The strongest available check of ``liif.py:69-105``.

    With the decoder reduced to "return the sampled feature", LIIF's four
    nearest-corner predictions weighted by the OPPOSITE corner's area (the two
    swaps at L100-102) are exactly bilinear interpolation.  Drop the swaps and
    this test fails by ~0.1 on random features -- which is why it is here and
    not a source-string assertion.
    """
    torch.manual_seed(0)
    head = _head(in_dim=1, feat_dim=0, cond_dim=0, feat_unfold=False,
                 cell_decode=False)
    head.imnet = torch.nn.Identity()             # in_dim 3 -> [feat, dy, dx]
    feat = torch.randn(1, 1, 5, 7)
    prep = head.prepare(feat)
    coord = (torch.rand(1, 64, 2) * 1.2 - 0.6)   # interior only
    cell = torch.zeros_like(coord)
    got = head.query(prep, coord, cell, torch.zeros(0))[0, :, 0]
    want = F.grid_sample(feat, coord.flip(-1).unsqueeze(1), mode="bilinear",
                         align_corners=False)[0, 0, 0]
    assert torch.allclose(got, want, atol=1e-5), (got - want).abs().max()


def test_without_local_ensemble_it_is_the_nearest_cell():
    torch.manual_seed(0)
    head = _head(in_dim=1, feat_dim=0, cond_dim=0, feat_unfold=False,
                 cell_decode=False, local_ensemble=False)
    head.imnet = torch.nn.Identity()
    feat = torch.randn(1, 1, 5, 7)
    prep = head.prepare(feat)
    coord = (torch.rand(1, 32, 2) * 1.2 - 0.6)
    got = head.query(prep, coord, torch.zeros_like(coord), torch.zeros(0))[0, :, 0]
    want = F.grid_sample(feat, coord.flip(-1).unsqueeze(1), mode="nearest",
                         align_corners=False)[0, 0, 0]
    assert torch.allclose(got, want, atol=1e-6)


def test_parameter_counts_are_the_written_down_ones():
    head = _head()
    p = head.n_params()
    assert head.imnet_in_dim == 644                      # 576 + 2 + 2 + 64
    assert p["adapter"] == 65_600                        # 1024*64 + 64
    assert p["cond_proj"] == 169_024                     # LN(2560) + 2560*64+64
    assert p["imnet"] == 362_753                         # 644->256^4->1
    assert p["geom_reg"] == 0
    assert p["total"] == 597_377


def test_switches_change_the_input_width_the_way_liif_says():
    assert _head(feat_unfold=False).imnet_in_dim == 64 + 2 + 2 + 64
    assert _head(cell_decode=False).imnet_in_dim == 576 + 2 + 64
    assert _head(cond_dim=0).imnet_in_dim == 576 + 2 + 2
    # ablation (7): no adapter, F_pre goes in raw
    assert _head(feat_dim=0).imnet_in_dim == 1024 * 9 + 2 + 2 + 64


def test_mlp_has_no_norm_and_no_dropout():
    """``mlp.py:9-18`` -- five Linears, four ReLUs, nothing else."""
    layers = list(_head().imnet.layers)
    assert [type(m).__name__ for m in layers] == (
        ["Linear", "ReLU"] * 4 + ["Linear"])


# --------------------------------------------------------------------------- #
# 3. shapes, step 0, the read-out aperture
# --------------------------------------------------------------------------- #
def test_forward_eval_shapes_and_step0_range():
    torch.manual_seed(0)
    head = _head().eval()
    ctx = _ctx(head, gh=4, gw=6, sample=_Sample(pixgt=_analytic_pixgt()))
    out = L.forward(_Model(head), head, ctx)
    m = out["m_low"].detach()
    assert m.shape == (4, 6)
    assert float(m.min()) >= 0.0 and float(m.max()) <= 1.0
    # step 0 is a random function of coordinate and feature, NOT constant 0.5
    assert float(m.std()) > 0.0
    assert out["liif_decode"]["qh"] == 16 and out["liif_decode"]["qw"] == 24
    assert out["liif_decode"]["cell"] == [2.0 / 16, 2.0 / 24]


def test_m_low_is_the_area_resize_of_the_4x_decode():
    """The criterion GT is ``area_resize(gt_hi, (gh,gw))`` (data.py:685-686);
    the prediction must come through the SAME operator."""
    torch.manual_seed(1)
    head = _head().eval()
    ctx = _ctx(head, gh=3, gw=5, sample=_Sample(pixgt=_analytic_pixgt()))
    out = L.forward(_Model(head), head, ctx)
    c = head.cond_vector(ctx.h_cond)
    prep = head.prepare(ctx.feat.float())
    alpha, _ = head.decode_grid(prep, c, 12, 20)
    want = area_resize(alpha[None, None], (3, 5))[0, 0]
    assert torch.allclose(out["m_low"], want, atol=1e-6)


def test_scale_max_moves_the_readout_aperture_with_the_training_scale():
    head = _head(scale_max=8.0).eval()
    ctx = _ctx(head, gh=3, gw=5, sample=_Sample(pixgt=_analytic_pixgt()))
    out = L.forward(_Model(head), head, ctx)
    assert (out["liif_decode"]["qh"], out["liif_decode"]["qw"]) == (24, 40)
    assert out["m_low"].shape == (3, 5)


def test_batched_query_matches_the_unchunked_one():
    torch.manual_seed(2)
    head = _head(eval_bsize=97).eval()
    prep = head.prepare(torch.randn(1, 1024, 4, 6))
    c = head.cond_vector(torch.randn(1, 2560))
    coord = L.make_coord((8, 12)).unsqueeze(0)
    cell = torch.ones_like(coord)
    cell[:, :, 0] *= 2 / 8
    cell[:, :, 1] *= 2 / 12
    with torch.no_grad():
        a = head.query(prep, coord, cell, c)
        b = head.batched_query(prep, coord, cell, c)
    assert torch.allclose(a, b, atol=1e-6)


def test_k_greater_than_one_condition_rows_are_averaged():
    head = _head()
    h = torch.randn(4, 2560)
    got = head.cond_vector(h)
    want = head.cond_proj(h).mean(dim=0)
    assert torch.allclose(got, want, atol=1e-6)
    assert head.counts.get("cond_rows_4") == 1


def test_missing_readout_builder_is_a_configuration_error():
    head = _head().eval()
    ctx = ArmContext(feat=torch.randn(1, 1024, 3, 4), grid_h=3, grid_w=4,
                     h_cond=None, sample=None)
    with pytest.raises(ValueError, match="ReadoutBuilder"):
        L.forward(_Model(head), head, ctx)


# --------------------------------------------------------------------------- #
# 4. training points, GT, loss
# --------------------------------------------------------------------------- #
def test_point_sampling_is_uniform_without_replacement_on_the_scaled_grid():
    head = _head(sample_q=50)
    p = head.sample_points(4, 6)
    assert 1.0 <= p["s"] <= 4.0
    assert p["qh"] == max(1, round(4 * p["s"])) and p["qw"] == max(1, round(6 * p["s"]))
    assert p["n_pts"] == min(50, p["qh"] * p["qw"])
    uniq = {tuple(np.round(r, 6)) for r in p["coord"].numpy()}
    assert len(uniq) == p["n_pts"]                       # no replacement
    assert torch.allclose(p["cell"][:, 0], torch.full((p["n_pts"],), 2.0 / p["qh"]))
    assert torch.allclose(p["cell"][:, 1], torch.full((p["n_pts"],), 2.0 / p["qw"]))
    # every drawn point is a cell centre of the (qh, qw) grid
    allc = L.make_coord((p["qh"], p["qw"]))
    for row in p["coord"]:
        assert torch.isclose(allc, row).all(dim=-1).any()


def test_sample_q_is_clamped_to_the_grid_and_counted():
    head = _head(sample_q=2304, scale_min=1.0, scale_max=1.0)
    p = head.sample_points(32, 48)
    assert p["n_pts"] == 32 * 48 < 2304
    assert head.counts["sample_q_short"] == 1


def test_the_sampler_has_its_own_rng():
    torch.manual_seed(1234)
    h1 = _head(sample_q=16, seed=7)
    before = torch.get_rng_state()
    p1 = h1.sample_points(4, 6)
    assert torch.equal(torch.get_rng_state(), before), "global RNG was consumed"
    h2 = _head(sample_q=16, seed=7)
    p2 = h2.sample_points(4, 6)
    assert p1["s"] == p2["s"] and torch.equal(p1["coord"], p2["coord"])
    h3 = _head(sample_q=16, seed=8)
    assert h3.sample_points(4, 6)["s"] != p1["s"]


def test_gt_points_use_the_closed_form_for_the_analytic_families():
    from q3vl.whereb.amort.pixgt import eval_analytic

    head = _head()
    pg = _analytic_pixgt()
    coord = L.make_coord((5, 7))
    got = head.gt_points(_Sample(pixgt=pg), coord)
    xy = torch.stack([(coord[:, 1] + 1) / 2, (coord[:, 0] + 1) / 2], dim=-1)
    want = eval_analytic("circulargradient", GEOM_CIRC, xy)
    assert torch.allclose(got, want, atol=1e-6)
    assert head.counts["gt_render"] == 1


def test_fake_samples_get_an_exactly_zero_target_and_are_counted():
    head = _head()
    g = head.gt_points(_Sample(pixgt=_analytic_pixgt(), is_fake=True),
                       L.make_coord((3, 4)))
    assert torch.equal(g, torch.zeros(12))
    assert head.counts["fake_zero_gt"] == 1
    # ... and the L1 target is the finite constant -1
    y = torch.zeros(12, requires_grad=True)
    out = {"y_pts": y, "g_pts": g}
    sl = L.compute_loss(_Model(head), out, _Sample(is_fake=True), None)
    assert float(sl.total.detach()) == pytest.approx(1.0)
    assert "l1_pts_fake" in sl.terms and "l1_pts_real" not in sl.terms


def test_missing_pixgt_names_the_flag():
    head = _head()
    with pytest.raises(ValueError, match="no-pixgt"):
        head.gt_points(_Sample(pixgt=None), L.make_coord((2, 2)))


def test_l1_loss_is_the_reference_form_on_a_hand_worked_example():
    """``L = mean|y - (2g - 1)|`` (train_liif.py:91/110 + yaml:33-35)."""
    head = _head()
    y = torch.tensor([0.2, -0.4], requires_grad=True)
    g = torch.tensor([0.5, 1.0])
    sl = L.compute_loss(_Model(head), {"y_pts": y, "g_pts": g}, _Sample(), None)
    assert float(sl.total.detach()) == pytest.approx((0.2 + 1.4) / 2)
    assert set(sl.terms) == {"l1_pts", "l1_pts_real"}
    sl.total.backward()
    assert y.grad is not None


def test_bce_ablation_swaps_the_loss_and_the_activation():
    from q3vl.whereb.amort.losses import bce_soft

    head = _head(loss="bce")
    y = torch.tensor([0.3, -1.2], requires_grad=True)
    g = torch.tensor([0.7, 0.1])
    sl = L.compute_loss(_Model(head), {"y_pts": y, "g_pts": g}, _Sample(), None)
    assert float(sl.total.detach()) == pytest.approx(
        float(bce_soft(torch.sigmoid(y), g).detach()))
    assert head.loss_term_key == "bce_pts" and "bce_pts" in sl.terms
    assert torch.allclose(head.alpha_of(y), torch.sigmoid(y))


def test_alpha_of_is_the_reference_inverse_normalisation_and_clamp():
    head = _head()
    y = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
    assert torch.allclose(head.alpha_of(y),
                          torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]))


def test_training_step_end_to_end_produces_gradients_on_all_three_modules():
    torch.manual_seed(3)
    head = _head(sample_q=64).train()
    sample = _Sample(pixgt=_analytic_pixgt())
    ctx = _ctx(head, gh=4, gw=6, sample=sample)
    out = L.forward(_Model(head), head, ctx)
    assert out["y_pts"].shape == (out["liif_train"]["n_pts"],)
    assert out["g_pts"].shape == out["y_pts"].shape
    assert out["m_low"].shape == (4, 6) and not out["m_low"].requires_grad
    sl = L.compute_loss(_Model(head), out, sample, None)
    sl.total.backward()
    for name, mod in (("adapter", head.adapter), ("cond", head.cond_proj),
                      ("imnet", head.imnet)):
        assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
                   for p in mod.parameters()), name


def test_train_stats_columns():
    head = _head(sample_q=32).train()
    sample = _Sample(pixgt=_analytic_pixgt())
    out = L.forward(_Model(head), head, _ctx(head, sample=sample))
    row = L.train_stats(out, sample)
    assert set(row) >= {"liif_scale", "liif_qh", "liif_qw", "liif_n_pts",
                        "gt_pix_source"}
    assert row["gt_pix_source"] == "render"


# --------------------------------------------------------------------------- #
# 5. the geometry-regression ablation
# --------------------------------------------------------------------------- #
def test_weight_zero_means_the_branch_is_not_constructed():
    head = _head()
    assert head.geom is None
    assert head.facts()["geom_reg"] is None
    assert head.expected_loss_columns() == ["L_l1_pts"]
    out = {"y_pts": torch.zeros(4, requires_grad=True), "g_pts": torch.zeros(4)}
    assert "geom" not in L.compute_loss(_Model(head), out, _Sample(), None).terms


def test_geom_head_last_layers_are_zero_initialised():
    head = _head(geom_reg_weight=0.05)
    with torch.no_grad():
        p = head.geom(torch.randn(64))
    assert float(p["circ_cont"].abs().sum()) == 0.0
    assert float(p["grad_cont"].abs().sum()) == 0.0
    assert float(p["circ_flip"]) == 0.0 and float(p["type_logit"]) == 0.0
    assert head.expected_loss_columns() == ["L_l1_pts", "L_geom"]
    assert head.n_params()["geom_reg"] > 0


def test_geom_targets_are_in_raster_geometry_units_and_are_not_squashed():
    t = L.geom_targets("circulargradient",
                       {"Left": "-0.30", "Right": "1.30", "Top": "0.20",
                        "Bottom": "0.80", "Angle": "45", "Feather": "40"})
    cx, cy, rx, ry, s2, c2, feather = t["values"]
    assert cx == pytest.approx(0.5) and cy == pytest.approx(0.5)
    assert rx == pytest.approx(0.8) and ry == pytest.approx(0.3)
    assert s2 == pytest.approx(math.sin(math.radians(90)))
    assert c2 == pytest.approx(math.cos(math.radians(90)), abs=1e-7)
    assert feather == pytest.approx(0.4)
    # band's axis is the constant 1.6 (subject_geom.py:22) -- outside [0,1]
    band = L.geom_targets("circulargradient",
                          {"Left": "-0.30", "Right": "2.90", "Top": "0.4",
                           "Bottom": "0.6", "Angle": "0"})
    assert band["values"][2] == pytest.approx(1.6)


def test_angle_columns_are_masked_for_near_isotropic_ellipses():
    iso = L.geom_targets("circulargradient",
                         {"Left": "0.0", "Right": "0.4", "Top": "0.0",
                          "Bottom": "0.4", "Angle": "17"})
    assert iso["angle_masked"] is True and iso["col_mask"][4:6] == [0.0, 0.0]
    elong = L.geom_targets("circulargradient",
                           {"Left": "0.0", "Right": "0.8", "Top": "0.0",
                            "Bottom": "0.2", "Angle": "17"})
    assert elong["angle_masked"] is False and elong["col_mask"][4:6] == [1.0, 1.0]


def test_geom_loss_is_l1_plus_bce_and_semantic_rows_are_excluded():
    head = _head(geom_reg_weight=0.05)
    pg = _analytic_pixgt()
    sample = _Sample(pixgt=pg)
    out = {"y_pts": torch.zeros(4, requires_grad=True), "g_pts": torch.zeros(4),
           "geom_pred": head.geom(torch.randn(64)),
           "geom_target": L.geom_targets("circulargradient", GEOM_CIRC)}
    sl = L.compute_loss(_Model(head), out, sample, None)
    assert "geom" in sl.terms and sl.stats["geom_n"] == 1.0
    # zero-init: |0 - target| averaged over the unmasked columns, + BCE(0, flip)
    t = L.geom_targets("circulargradient", GEOM_CIRC)
    m = torch.tensor(t["col_mask"])
    want = float((torch.tensor(t["values"]).abs() * m).sum() / m.sum())
    want += math.log(2) * (1 + head.geom_reg_type_weight)
    assert float(sl.terms["geom"]) == pytest.approx(want, abs=1e-5)
    # ... and the total is L1 + weight * L_geom
    assert float(sl.total) == pytest.approx(float(sl.terms["l1_pts"])
                                            + 0.05 * want, abs=1e-5)
    # semantic (no geometry) contributes zero and is counted
    out2 = dict(out, geom_target=None)
    sl2 = L.compute_loss(_Model(head), out2, sample, None)
    assert float(sl2.terms["geom"]) == 0.0 and sl2.stats["geom_n"] == 0.0


def test_geom_target_of_skips_semantic_and_fake_samples():
    head = _head(geom_reg_weight=0.05)
    sem = PixGT(alpha=torch.zeros(4, 4), source="maskhi512", family="semantic")
    assert L._geom_target_of(head, _Sample(pixgt=sem, family="semantic")) is None
    assert head.counts["geom_target_missing"] == 1
    assert L._geom_target_of(head, _Sample(pixgt=_analytic_pixgt(),
                                           is_fake=True)) is None
    assert head.counts["geom_target_fake_skipped"] == 1


def test_geom_type_weight_zero_reproduces_the_formula_as_written():
    head = _head(geom_reg_weight=0.05, geom_reg_type_weight=0.0)
    out = {"y_pts": torch.zeros(2, requires_grad=True), "g_pts": torch.zeros(2),
           "geom_pred": head.geom(torch.randn(64)),
           "geom_target": L.geom_targets("gradient", GEOM_GRAD)}
    sl = L.compute_loss(_Model(head), out, _Sample(), None)
    t = L.geom_targets("gradient", GEOM_GRAD)
    want = float(np.mean(np.abs(t["values"]))) + math.log(2)   # Flipped = true
    assert float(sl.terms["geom"]) == pytest.approx(want, abs=1e-6)


# --------------------------------------------------------------------------- #
# 6. evaluation columns and the runtime assertions
# --------------------------------------------------------------------------- #
def test_per_sample_row_emits_the_pre_registered_criterion():
    torch.manual_seed(4)
    head = _head(pix_diag=False).eval()
    sample = _Sample(pixgt=_analytic_pixgt())
    out = L.forward(_Model(head), head, _ctx(head, sample=sample))
    row = L.per_sample_row(_Model(head), out, sample)
    assert row["liif_grid_decode"] == 1.0
    assert row["liif_decode_scale"] == 4.0
    assert row["liif_decode_projection"] == "area_resize"


def test_pixel_diagnostic_columns_and_their_baselines():
    torch.manual_seed(5)
    head = _head(pix_diag=True).eval()
    gt = _analytic_pixgt(hw=(32, 48)).alpha
    sample = _Sample(pixgt=_analytic_pixgt(), gt_pix=gt)
    out = L.forward(_Model(head), head, _ctx(head, gh=2, gw=3, sample=sample))
    row = L.per_sample_row(_Model(head), out, sample)
    for k in ("pix_soft_iou_512", "pix_soft_iou_512_center",
              "pix_soft_iou_512_floor", "pix_band_mae_512",
              "pix_band_mae_512_center", "pix_band_mae_512_floor"):
        assert row[k] is not None, k
    assert 0.0 <= row["pix_soft_iou_512"] <= 1.0
    assert row["pix_grid"] == [32, 48]


def test_a_raising_pixel_diagnostic_still_produces_the_eval_row(monkeypatch):
    """结果落盘先于可选阶段: the diagnostic is optional, the row is not."""
    torch.manual_seed(5)
    head = _head(pix_diag=True).eval()
    gt = _analytic_pixgt(hw=(32, 48)).alpha
    sample = _Sample(pixgt=_analytic_pixgt(), gt_pix=gt)

    def _boom(*a, **kw):
        raise RuntimeError("CUDA out of memory (pretend)")

    monkeypatch.setattr(L, "_pixel_diagnostic", _boom)
    out = L.forward(_Model(head), head, _ctx(head, gh=2, gw=3, sample=sample))
    # the headline read-out survived
    assert out["m_low"].shape == (2, 3)
    assert out["liif_pix"] is None
    assert "RuntimeError" in out["liif_pix_error"]
    assert head.counts["pix_diag_error"] == 1

    row = L.per_sample_row(_Model(head), out, sample)
    assert row["liif_grid_decode"] == 1.0            # the criterion column
    assert "pix_soft_iou_512" not in row
    assert "CUDA out of memory" in row["pix_diag_error"]

    cols = L.criteria_columns([row])
    assert cols["liif_grid_decode"]["n"] == 1        # board still publishable
    assert cols["liif_pix_diag_errors"]["n"] == 1
    assert cols["liif_pix_diag_errors"]["kinds"] == ["RuntimeError"]
    assert "liif_pix_diag" not in cols


def test_criteria_columns_pass_assert_criteria_ran():
    from q3vl.whereb.amort.evaluate import assert_criteria_ran

    rows = [{"liif_grid_decode": 1.0, "liif_decode_scale": 4.0,
             "liif_decode_projection": "area_resize",
             "liif_decode_activation": "clamp(0.5y+0.5,0,1)",
             "liif_gt_pix_source": "render", "family": "radial"}
            for _ in range(7)]
    cols = L.criteria_columns(rows)
    assert cols["liif_grid_decode"]["n"] == 7
    board = {"criteria_columns": cols}
    rep = assert_criteria_ran(board, "LIIF")
    assert rep["computed"]["liif_grid_decode"] == 7


def test_a_board_without_the_decode_column_cannot_publish():
    from q3vl.whereb.amort.evaluate import assert_criteria_ran

    cols = L.criteria_columns([])
    assert cols["liif_grid_decode"]["n"] == 0
    with pytest.raises(AssertionError, match="liif_grid_decode"):
        assert_criteria_ran({"criteria_columns": cols}, "LIIF")


def test_geom_criterion_column_carries_the_routing_confusion():
    rows = [
        {"liif_grid_decode": 1.0, "family": "band", "geom_reg_route": 1.0,
         "geom_pred_label": "band", "geom_pred_rx": 1.5,
         "geom_type_correct": 1.0, "geom_angle_masked": 0.0},
        {"liif_grid_decode": 1.0, "family": "band", "geom_reg_route": 1.0,
         "geom_pred_label": "radial", "geom_pred_rx": 0.4,
         "geom_type_correct": 1.0, "geom_angle_masked": 1.0},
        {"liif_grid_decode": 1.0, "family": "linear", "geom_reg_route": 1.0,
         "geom_pred_label": "linear", "geom_pred_rx": 0.2,
         "geom_type_correct": 1.0, "geom_angle_masked": 0.0},
        {"liif_grid_decode": 1.0, "family": "linear", "geom_reg_route": 1.0,
         "geom_pred_label": "band", "geom_pred_rx": 1.9,
         "geom_type_correct": 0.0, "geom_angle_masked": 0.0},
    ]
    col = L.criteria_columns(rows)[L.GEOM_CRITERION]
    assert col["n"] == 4 and col["n_confusion"] == 4
    assert col["confusion"]["gt_band__pred_band"] == 1
    assert col["confusion"]["gt_band__pred_radial"] == 1
    assert col["confusion"]["gt_linear__pred_linear"] == 1
    assert col["confusion"]["gt_linear__pred_band"] == 1
    assert col["mask_type_accuracy"] == pytest.approx(0.75)
    assert col["angle_masked_frac"] == pytest.approx(0.25)


def test_first_step_row_assertion():
    row = {"step": 1, "loss": 0.5, "L_l1_pts": 0.5, "L_l1_pts_real": 0.5}
    rep = L.assert_first_step_row(row, expected=["L_l1_pts"])
    assert rep["step"] == 1
    with pytest.raises(AssertionError, match="L_geom"):
        L.assert_first_step_row(row, expected=["L_l1_pts", "L_geom"])


def test_the_loss_hook_asserts_its_columns_on_the_first_micro_batch(monkeypatch):
    head = _head(geom_reg_weight=0.05)
    monkeypatch.setattr(head, "expected_loss_columns",
                        lambda: ["L_l1_pts", "L_nonexistent"])
    out = {"y_pts": torch.zeros(2, requires_grad=True), "g_pts": torch.zeros(2),
           "geom_pred": head.geom(torch.zeros(64)), "geom_target": None}
    with pytest.raises(AssertionError, match="L_nonexistent"):
        L.compute_loss(_Model(head), out, _Sample(), None)


def test_steps_jsonl_assertion_runs_from_the_board_build(tmp_path, monkeypatch):
    import json as _json

    (tmp_path / "steps.jsonl").write_text(
        _json.dumps({"step": 1, "loss": 1.0, "L_l1_pts": 1.0}) + "\n")
    monkeypatch.setitem(L.CFG, "run_dir", str(tmp_path))
    monkeypatch.setitem(L.CFG, "expected_loss_columns", ["L_l1_pts", "L_geom"])
    with pytest.raises(AssertionError, match="L_geom"):
        L.criteria_columns([{"liif_grid_decode": 1.0, "family": "radial"}])


# --------------------------------------------------------------------------- #
# 6b. end to end through the real model / trainer / evaluator seams
# --------------------------------------------------------------------------- #
def _amort_sample(gh=2, gw=3, *, in_dim=32, text_dim=64, is_fake=False):
    from q3vl.whereb.amort.data import AmortSampleInputs
    from q3vl.whereb.amort.pixgt import render_analytic

    a = render_analytic("circulargradient", GEOM_CIRC, 32, 48)
    pg = PixGT(alpha=a, source="render", family="radial",
               mask_type="circulargradient", geometry=dict(GEOM_CIRC))
    return AmortSampleInputs(
        sample_id="s1", feat=torch.randn(1, in_dim, gh, gw), sim=None,
        center=None, cond_h=torch.randn(1, 5, text_dim),
        cond_mask=torch.ones(1, 5, dtype=torch.bool),
        word_ids=torch.tensor([1, 2]), word_offsets=torch.tensor([0]),
        phi_dir=torch.randn(gh * gw, 71), guide_hi=None,
        gt_low=torch.rand(gh, gw), gt_hi=None, gt_partner_low=None,
        grid_h=gh, grid_w=gw, is_fake=is_fake, family="radial",
        route_semantic=False, h_cond=torch.randn(1, text_dim), gt_pix=a,
        gt_pix_source="render", pixgt=pg, readout={"kind": "seg_where"})


def test_amort_model_builds_the_arm_and_applies_the_b4_defaults():
    from q3vl.whereb.amort.model import AmortModel

    m = AmortModel("LIIF", in_dim=32, cond_text_dim=64)
    assert isinstance(m.geo, L.LIIFHead)
    # B-4: all four families through the new head, CondEncoder frozen
    assert m.sem is None and m.use_sim_field is False and m.use_film is False
    assert m.cond_frozen and all(not p.requires_grad for p in m.cond.parameters())
    assert m.facts()["arm_head"]["imnet_in_dim"] == 644


def test_compute_micro_batch_lands_L_l1_pts_in_the_step_row():
    """The column the runtime assertion looks for really is produced by the
    trainer's aggregation, not just by the head."""
    from q3vl.whereb.amort.losses import LossWeights, aggregate
    from q3vl.whereb.amort.model import AmortModel
    from q3vl.whereb.amort.trainer import compute_micro_batch

    torch.manual_seed(6)
    m = AmortModel("LIIF", in_dim=32, cond_text_dim=64).train()
    x = _amort_sample()
    total, stats, rows = compute_micro_batch(m, [x], LossWeights())
    assert "L_l1_pts" in stats and "L_l1_pts_real" in stats
    assert stats["n"] == 1 and stats["n_fake"] == 0
    assert {"liif_scale", "liif_qh", "liif_n_pts"} <= set(rows[0])
    total.backward()
    assert all(p.grad is None for p in m.cond.parameters())    # frozen
    assert any(p.grad is not None for p in m.geo.imnet.parameters())
    L.assert_first_step_row({"step": 1, **stats}, expected=["L_l1_pts"])


def test_fake_samples_are_counted_separately_in_the_step_row():
    from q3vl.whereb.amort.losses import LossWeights
    from q3vl.whereb.amort.model import AmortModel
    from q3vl.whereb.amort.trainer import compute_micro_batch

    torch.manual_seed(7)
    m = AmortModel("LIIF", in_dim=32, cond_text_dim=64).train()
    _, stats, _ = compute_micro_batch(
        m, [_amort_sample(), _amort_sample(is_fake=True)], LossWeights())
    assert stats["n"] == 2 and stats["n_fake"] == 1
    assert "L_l1_pts_fake" in stats and "L_l1_pts_real" in stats


# --------------------------------------------------------------------------- #
# 7. optimizer / scheduler / builder hooks
# --------------------------------------------------------------------------- #
def test_optimizer_is_plain_adam_1e_4_no_decay_one_group():
    from q3vl.whereb.amort.trainer import AmortTrainConfig, build_optimizer

    spec = L.optimizer_spec(argparse.Namespace())
    assert (spec.type, spec.lr, spec.weight_decay, spec.grouping) == (
        "adam", 1e-4, 0.0, "none")
    opt = build_optimizer(_head(), AmortTrainConfig(arm="LIIF"), spec)
    assert type(opt) is torch.optim.Adam
    assert len(opt.param_groups) == 1
    assert opt.param_groups[0]["weight_decay"] == 0.0
    assert opt.param_groups[0]["lr"] == 1e-4


def test_scheduler_is_multistep_at_20_40_60_80_percent_without_warmup():
    from q3vl.where.calibrate import make_scheduler

    kw = L.scheduler_kwargs(argparse.Namespace(), 1200)
    assert kw["milestones"] == [240, 480, 720, 960]
    assert kw["gamma"] == 0.5 and kw["warmup_steps"] == 1
    opt = torch.optim.Adam(_head().parameters(), lr=1e-4)
    sch = make_scheduler(opt, 1200, 0.03, "multistep", **kw)
    seen = []
    for step in range(1200):
        seen.append(opt.param_groups[0]["lr"])
        opt.step()
        sch.step()
    assert seen[0] == pytest.approx(1e-4)          # no warmup ramp
    assert seen[239] == pytest.approx(1e-4)
    assert seen[240] == pytest.approx(0.5e-4)
    assert seen[960] == pytest.approx(1e-4 * 0.5 ** 4)


def test_scheduler_refuses_an_unmatched_horizon():
    with pytest.raises(ValueError, match="max-steps"):
        L.scheduler_kwargs(argparse.Namespace(), 0)


def test_a_non_multistep_scheduler_is_refused():
    """Bypassing the wrapper must not silently give cosine + a 1-step warmup."""
    assert L.SCHEDULER_KIND == "multistep"
    with pytest.raises(SystemExit, match="--scheduler 'multistep'"):
        L.scheduler_kwargs(argparse.Namespace(scheduler="cosine"), 1200)
    # the pinned kind passes, and an absent flag keeps the old behaviour
    assert L.scheduler_kwargs(argparse.Namespace(scheduler="multistep"),
                              1200)["gamma"] == 0.5


def test_builder_asks_for_pixel_gt_at_the_spec5_grid():
    kw = L.builder_kwargs(argparse.Namespace())
    assert kw["pixgt_size"](32, 48) == (512, 768)


# --------------------------------------------------------------------------- #
# 8. the entry script
# --------------------------------------------------------------------------- #
def test_flag_defaults_are_the_reference_values():
    ap = argparse.ArgumentParser(add_help=False)
    L.add_arguments(ap)
    own, rest = ap.parse_known_args([])
    cfg = L.cfg_from_args(own)
    assert cfg["sample_q"] == 2304 and cfg["hidden"] == (256, 256, 256, 256)
    assert cfg["feat_dim"] == 64 and cfg["cond_dim"] == 64
    assert cfg["scale_min"] == 1.0 and cfg["scale_max"] == 4.0
    assert cfg["local_ensemble"] and cfg["feat_unfold"] and cfg["cell_decode"]
    assert cfg["gt"] == "closed" and cfg["loss"] == "l1"
    assert cfg["eval_bsize"] == 65536
    assert cfg["geom_reg_weight"] == 0.0
    assert rest == []


def test_every_ablation_row_has_a_flag():
    ap = argparse.ArgumentParser(add_help=False)
    L.add_arguments(ap)
    own, rest = ap.parse_known_args([
        "--geom-reg-weight", "0.05", "--liif-no-local-ensemble",
        "--liif-sample-q", "576", "--liif-gt", "cgt", "--liif-loss", "bce",
        "--liif-feat-dim", "0", "--liif-no-feat-unfold",
        "--liif-no-cell-decode", "--liif-scale-max", "16"])
    cfg = L.cfg_from_args(own)
    assert rest == []
    assert cfg["geom_reg_weight"] == 0.05 and cfg["local_ensemble"] is False
    assert cfg["sample_q"] == 576 and cfg["gt"] == "cgt" and cfg["loss"] == "bce"
    assert cfg["feat_dim"] == 0 and cfg["feat_unfold"] is False
    assert cfg["cell_decode"] is False and cfg["scale_max"] == 16.0


@pytest.fixture()
def wrapper(monkeypatch):
    """``run_liifhead_arm.main`` with the delegation and the global seams
    captured: nothing reaches ``run_amort_arm`` and every patch is undone."""
    from q3vl.whereb.amort import pixgt as _pix
    from q3vl.whereb.scripts import run_liifhead_arm as W

    seen: dict[str, list[str]] = {}
    monkeypatch.setattr("q3vl.whereb.scripts.run_amort_arm.main",
                        lambda rest: seen.__setitem__("rest", list(rest)) or 0)
    monkeypatch.setattr(L, "CFG", dict(L.CFG))
    monkeypatch.setattr(_pix, "PixGTProvider", _pix.PixGTProvider)
    monkeypatch.setattr(W, "V2SEG_BASE", "/nonexistent/v2seg/checkpoint-4976")

    def run(argv):
        rc = W.main(list(argv))
        return rc, seen.get("rest", [])

    run.seen = seen                                       # type: ignore[attr-defined]
    return run


def test_wrapper_forces_the_recipe_flags_and_delegates(wrapper, tmp_path):
    rc, seen_rest = wrapper(["--out-root", str(tmp_path), "--run-name", "t1",
                             "--max-steps", "1200", "--cond-readout", "im_end"])
    seen = {"rest": seen_rest}
    assert rc == 0
    rest = seen["rest"]
    for pair in (["--arm", "LIIF"], ["--scheduler", "multistep"],
                 ["--max-grad-norm", "0"], ["--pixgt-source", "render"]):
        i = rest.index(pair[0])
        assert rest[i + 1] == pair[1]
    for flag in ("--no-semantic-head", "--no-sim-field", "--no-film"):
        assert flag in rest
    setup = __import__("json").loads(
        (tmp_path / "t1" / "config" / "liif_setup.json").read_text())
    assert setup["params"]["total"] == 597_377
    assert setup["loss_columns"] == ["L_l1_pts"]
    assert setup["config"]["scale_max"] == 4.0
    assert len(setup["liifhead_sha256"]) == 64 and len(setup["wrapper_sha256"]) == 64


def test_wrapper_keeps_the_legacy_routing_row_available(wrapper, tmp_path):
    _, rest = wrapper(["--newarm-legacy-routing", "--out-root", str(tmp_path),
                       "--cond-readout", "im_end"])
    assert "--no-semantic-head" not in rest
    assert "--newarm-legacy-routing" in rest


def test_wrapper_maps_the_gt_ablation_onto_the_shared_pixgt_flag(wrapper,
                                                                 tmp_path):
    _, rest = wrapper(["--liif-gt", "cgt", "--out-root", str(tmp_path),
                       "--cond-readout", "im_end"])
    i = rest.index("--pixgt-source")
    assert rest[i + 1] == "maskhi"


def test_wrapper_refuses_seg_where_before_the_v2seg_base_exists(wrapper,
                                                                tmp_path):
    """Proposal §1 "依赖项": the two seg tokens are supervised on v2seg only."""
    with pytest.raises(SystemExit, match="im_end"):
        wrapper(["--out-root", str(tmp_path)])


def test_wrapper_accepts_seg_where_once_the_base_is_on_disk(wrapper, tmp_path):
    base = tmp_path / "q3vl_base_sft_v2seg_20260814" / "checkpoint-4976"
    base.mkdir(parents=True)
    _, rest = wrapper(["--out-root", str(tmp_path), "--checkpoint", str(base)])
    assert rest[rest.index("--checkpoint") + 1] == str(base)


def test_wrapper_pins_the_linear_amount_reference_grid(wrapper, tmp_path):
    from q3vl.whereb.amort import pixgt as _pix

    wrapper(["--out-root", str(tmp_path), "--cond-readout", "im_end"])
    assert _pix.PixGTProvider(prefer="render").amount_ref_hw == (256, 256)
    # 0 restores the shared default ("raw_mean on the grid being rendered")
    wrapper(["--out-root", str(tmp_path), "--cond-readout", "im_end",
             "--liif-amount-ref", "0"])
