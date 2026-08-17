"""EPR-019 SAMDEC: the ported SAM decoder, its loss, and that both are WIRED.

CPU only (``CUDA_VISIBLE_DEVICES=""``).  What is pinned here is the part of the
proposal a board cannot show by itself:

* the port's **parameter count**, module by module, against the numbers §2 of
  the proposal derives from the upstream files -- a silently wrong width (a
  ``GroupNorm`` instead of ``LayerNorm2d``, a missing hypernet) would still
  train and still produce a plausible board;
* the **loss arithmetic**, against values computed by hand from SAM §A and
  ``sam2/training/loss_fns.py``, including the empty-GT (``is_fake``) case the
  ``+1`` dice smoothing and the ``clamp(min=1)`` IoU denominator exist for;
* the **winner-takes-all**: only the lowest ``20*focal + 1*dice`` candidate
  carries gradient, and the IoU term is taken at that same index;
* the **runtime assertions**: the pre-registered ``samdec_cand`` column and the
  ``L_focal`` / ``L_dice`` / ``L_iouhead`` / ``L_sup_cells`` training witnesses,
  both of which must refuse a board rather than let it publish;
* that with the arm not selected, nothing here is imported and no live arm moves.
"""

from __future__ import annotations

import argparse
import inspect
import math

import pytest
import torch

from q3vl.whereb.amort import samdec
from q3vl.whereb.amort.samdec import SAMDecHead, SamLossConfig, sam_mask_loss

LN2 = math.log(2.0)


@pytest.fixture(autouse=True)
def _restore_variant():
    """``samdec.VARIANT`` is module state the wrapper writes; never leak it."""
    snapshot = dict(samdec.VARIANT)
    yield
    samdec.VARIANT.clear()
    samdec.VARIANT.update(snapshot)


def _args(**kw):
    base = dict(cond_readout="seg_where", readout_qtok=0, readout_nseg=1,
                seed=20260810)
    base.update(kw)
    return argparse.Namespace(**base)


# --------------------------------------------------------------------------- #
# registry contract
# --------------------------------------------------------------------------- #
def test_the_module_satisfies_the_arm_contract():
    from q3vl.whereb.amort.arms import (ARM_CRITERIA, REQUIRED_HOOKS, load_arm)

    mod = load_arm("SAMDEC")          # validates ARM / CRITERIA / hooks
    assert mod is samdec
    assert samdec.ARM == "SAMDEC"
    assert samdec.CRITERIA == ARM_CRITERIA["SAMDEC"] == ("samdec_cand",)
    for hook in REQUIRED_HOOKS:
        assert callable(getattr(samdec, hook))
    for hook in ("add_arguments", "head_kwargs_from_args", "optimizer_spec",
                 "scheduler_kwargs", "builder_kwargs", "readout_spec",
                 "per_sample_row", "criteria_columns", "train_stats"):
        assert callable(getattr(samdec, hook)), hook


def test_no_pretrained_weights_are_ever_loaded():
    """EPR-019 is EPR-018 minus the checkpoint; a ``load_state_dict`` here would
    silently make the two arms the same experiment."""
    src = inspect.getsource(samdec)
    assert "load_state_dict" not in src and "torch.load" not in src
    head = SAMDecHead(in_dim=16, text_dim=32)
    assert head.facts()["pretrained_weights_loaded"] is False


# --------------------------------------------------------------------------- #
# structure: the parameter count is the port
# --------------------------------------------------------------------------- #
#: proposal §2, derived line by line from the upstream files
EXPECT_GROUPS = {
    "neck": 852_992,          # 1024*256 + 256 + 256*256*9 + 256
    "prompt_proj": 655_616,   # 2560*256 + 256
    "no_mask_embed": 256,
    "decoder": 4_058_340,     # transformer 3_291_264 + upscaling 73_952
                              # + 4 hypernets 559_232 + iou head 132_612 + 1_280
}
EXPECT_TOTAL = sum(EXPECT_GROUPS.values())            # 5_567_204


def test_parameter_count_matches_the_proposal_module_by_module():
    head = SAMDecHead()
    f = head.facts()
    assert f["params_by_group"] == EXPECT_GROUPS
    assert f["n_params"] == f["n_trainable"] == EXPECT_TOTAL == 5_567_204


def test_the_decoder_sub_blocks_are_sam_s_own_widths():
    d = SAMDecHead().decoder
    n = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    assert n(d.transformer) == 3_291_264
    assert n(d.output_upscaling) == 73_952
    assert n(d.output_hypernetworks_mlps) == 559_232
    assert n(d.iou_prediction_head) == 132_612
    assert d.iou_token.weight.numel() + d.mask_tokens.weight.numel() == 1_280
    assert d.num_mask_tokens == 4                      # 3 multimask + 1
    # LayerNorm2d, not GroupNorm(1, C): per-position over channels
    assert isinstance(d.output_upscaling[1], samdec.LayerNorm2d)
    assert len(d.transformer.layers) == 2
    assert d.transformer.layers[0].skip_first_layer_pe
    assert not d.transformer.layers[1].skip_first_layer_pe
    assert d.transformer.layers[0].cross_attn_token_to_image.internal_dim == 128
    assert d.transformer.layers[0].self_attn.internal_dim == 256


def test_layernorm2d_is_channelwise_not_groupnorm():
    ln = samdec.LayerNorm2d(4)
    x = torch.randn(1, 4, 3, 5)
    y = ln(x)
    # each spatial position is standardised across channels
    assert torch.allclose(y.mean(1), torch.zeros(1, 3, 5), atol=1e-5)
    assert torch.allclose(y.std(1, unbiased=False), torch.ones(1, 3, 5), atol=1e-3)


def test_position_embedding_is_a_buffer_and_never_trained():
    head = SAMDecHead(in_dim=16, text_dim=32)
    pe = head.pe_layer.positional_encoding_gaussian_matrix
    assert pe.shape == (2, 128)
    assert not isinstance(pe, torch.nn.Parameter)
    assert all(p is not pe for p in head.parameters())
    assert head.facts()["pe_is_buffer"] is True
    # and it is defined for ANY grid, which is what a varying H/16 grid needs
    assert head.pe_layer((3, 5)).shape == (256, 3, 5)
    assert head.pe_layer((7, 4)).shape == (256, 7, 4)


def test_the_iou_head_ablation_removes_exactly_those_two_modules():
    off = SAMDecHead(iou_head=False)
    assert off.decoder.iou_token is None
    assert off.decoder.iou_prediction_head is None
    assert off.facts()["n_params"] == EXPECT_TOTAL - 256 - 132_612


def test_multimask_1_only_changes_the_slice_not_the_parameters():
    assert SAMDecHead(multimask=1).facts()["n_params"] == EXPECT_TOTAL
    assert SAMDecHead(multimask=1).n_candidates == 1
    assert SAMDecHead(multimask=3).n_candidates == 3
    with pytest.raises(ValueError, match="samdec-multimask"):
        SAMDecHead(multimask=2)


# --------------------------------------------------------------------------- #
# forward: shapes and the step-0 property
# --------------------------------------------------------------------------- #
def _head_and_inputs(gh=3, gw=4, in_dim=16, text_dim=32, **kw):
    head = SAMDecHead(in_dim=in_dim, text_dim=text_dim, **kw)
    feat = torch.randn(1, in_dim, gh, gw)
    h = torch.randn(1, text_dim)
    return head, feat, h


def test_forward_produces_three_candidates_at_4x():
    gh, gw = 3, 4
    head, feat, h = _head_and_inputs(gh, gw)
    logits, iou = head(feat, h, gh, gw)
    assert logits.shape == (3, 4 * gh, 4 * gw)
    assert iou.shape == (3,)
    assert torch.isfinite(logits).all() and torch.isfinite(iou).all()


def test_forward_single_candidate_and_iou_head_off():
    gh, gw = 3, 4
    head, feat, h = _head_and_inputs(gh, gw, multimask=1)
    logits, iou = head(feat, h, gh, gw)
    assert logits.shape == (1, 4 * gh, 4 * gw) and iou.shape == (1,)

    head, feat, h = _head_and_inputs(gh, gw, iou_head=False)
    logits, iou = head(feat, h, gh, gw)
    assert logits.shape == (3, 4 * gh, 4 * gw)
    assert torch.equal(iou, torch.zeros(3))


def test_forward_refuses_a_grid_that_disagrees_with_the_features():
    head, feat, h = _head_and_inputs(3, 4)
    with pytest.raises(ValueError, match="disagrees"):
        head(feat, h, 4, 4)
    with pytest.raises(ValueError, match="condition rows"):
        head(feat, torch.randn(1, 7), 3, 4)


def test_arm_forward_returns_m_low_on_the_criterion_grid():
    gh, gw = 3, 4
    from q3vl.whereb.amort.arms import ArmContext

    head, feat, h = _head_and_inputs(gh, gw)
    ctx = ArmContext(feat=feat, grid_h=gh, grid_w=gw, h_cond=h)
    out = samdec.forward(None, head, ctx)
    assert out["m_low"].shape == (gh, gw)
    m = out["m_low"].detach()
    assert float(m.min()) >= 0.0 and float(m.max()) <= 1.0
    assert out["m_4x"].shape == (4 * gh, 4 * gw)
    assert out["samdec"]["n_cand"] == 3
    assert out["samdec"]["n_prompt_tokens"] == 1
    # SAM's own GT-free selection
    assert out["samdec"]["sel"] == int(out["samdec"]["iou_pred"].argmax())


def test_m_low_is_the_area_projection_of_the_selected_candidate():
    """The criterion grid is reached with the SAME operator ``gt_low`` uses."""
    from q3vl.where.upsample import area_resize
    from q3vl.whereb.amort.arms import ArmContext

    gh, gw = 3, 4
    head, feat, h = _head_and_inputs(gh, gw)
    out = samdec.forward(None, head, ArmContext(feat=feat, grid_h=gh, grid_w=gw,
                                                h_cond=h))
    sel = out["samdec"]["sel"]
    want = area_resize(torch.sigmoid(out["samdec"]["logits"][sel])[None, None],
                       (gh, gw))[0, 0]
    assert torch.allclose(out["m_low"], want)


def test_step0_field_is_not_constant():
    """Proposal §3.3: at initialisation the field is near 0.5 but NOT spatially
    constant -- a constant field would mean the image path is dead."""
    from q3vl.whereb.amort.arms import ArmContext

    gh, gw = 4, 6
    head, feat, h = _head_and_inputs(gh, gw)
    m = samdec.forward(None, head, ArmContext(feat=feat, grid_h=gh, grid_w=gw,
                                              h_cond=h))["m_low"].detach()
    assert float(m.std()) > 1e-6


def test_iou_head_off_deploys_candidate_zero():
    from q3vl.whereb.amort.arms import ArmContext

    gh, gw = 3, 4
    head, feat, h = _head_and_inputs(gh, gw, iou_head=False)
    out = samdec.forward(None, head, ArmContext(feat=feat, grid_h=gh, grid_w=gw,
                                                h_cond=h))
    assert out["samdec"]["sel"] == 0


def test_missing_h_cond_names_the_flag():
    from q3vl.whereb.amort.arms import ArmContext

    head, feat, _ = _head_and_inputs(3, 4)
    with pytest.raises(ValueError, match="ReadoutBuilder"):
        samdec.forward(None, head, ArmContext(feat=feat, grid_h=3, grid_w=4))


def test_k_gt_1_readouts_become_k_sparse_tokens():
    """The pre-registered NOVEL default of §4 (qtok / nseg rows): one token per
    condition row, all through the same ``prompt_proj``."""
    from q3vl.whereb.amort.arms import ArmContext

    gh, gw = 3, 4
    head, feat, _ = _head_and_inputs(gh, gw)
    out = samdec.forward(None, head, ArmContext(feat=feat, grid_h=gh, grid_w=gw,
                                                h_cond=torch.randn(4, 32)))
    assert out["samdec"]["n_prompt_tokens"] == 4
    assert head.facts()["prompt_token_hist"] == {4: 1}


def test_span_prompt_reads_the_where_span_rows_and_guards_the_readout():
    from q3vl.whereb.amort.arms import ArmContext

    gh, gw = 3, 4
    head, feat, _ = _head_and_inputs(gh, gw, prompt="span")
    ctx = ArmContext(feat=feat, grid_h=gh, grid_w=gw, h_cond=torch.randn(1, 32),
                     h_where=torch.randn(1, 6, 32))
    out = samdec.forward(None, head, ctx)
    assert out["samdec"]["n_prompt_tokens"] == 6

    class _S:
        readout = {"kind": "seg_where"}

    ctx.sample = _S()
    with pytest.raises(AssertionError, match="where_span_pool"):
        samdec.forward(None, head, ctx)


# --------------------------------------------------------------------------- #
# the loss, against hand-computed values
# --------------------------------------------------------------------------- #
def test_focal_and_dice_match_the_closed_form():
    """z = 0 everywhere, t = 1 everywhere, N = 16:

    focal = alpha * ln2 * (1 - 0.5)^2 = 0.25 * 0.693147 * 0.25
    dice  = 1 - (2*0.5*16 + 1) / (0.5*16 + 16 + 1) = 1 - 17/25
    IoU   = |{z>0} & {t>0.5}| / max(|union|, 1) = 0
    """
    logits = torch.zeros(1, 4, 4)
    target = torch.ones(4, 4)
    res = sam_mask_loss(logits, torch.tensor([0.5]), target)
    assert float(res["focal"]) == pytest.approx(0.25 * LN2 * 0.25, rel=1e-6)
    assert float(res["dice"]) == pytest.approx(0.32, rel=1e-6)
    assert float(res["iou_true"][0]) == pytest.approx(0.0)
    assert float(res["iouhead"]) == pytest.approx(0.25, rel=1e-6)
    assert float(res["total"]) == pytest.approx(
        20 * 0.25 * LN2 * 0.25 + 0.32 + 0.25, rel=1e-6)
    assert res["sup_cells"] == 16


def test_empty_gt_is_defined_and_its_iou_target_is_zero():
    """``is_fake`` (p = 0.15 of the training stream) has an all-zero GT: dice's
    ``+1`` smoothing and the ``clamp(min=1)`` denominator keep both terms
    finite, which is why the fake stream stays in the loss (NOTES 4)."""
    logits = torch.zeros(1, 4, 4)
    res = sam_mask_loss(logits, torch.tensor([0.0]), torch.zeros(4, 4))
    assert float(res["focal"]) == pytest.approx(0.75 * LN2 * 0.25, rel=1e-6)
    assert float(res["dice"]) == pytest.approx(1 - 1 / 9, rel=1e-6)
    assert float(res["iou_true"][0]) == 0.0
    assert torch.isfinite(res["total"])


def test_the_soft_alpha_iou_target_uses_the_0_5_threshold():
    """SAM2 thresholds the GT at ``> 0``; on a soft alpha that counts the whole
    feather band.  0.5 is the threshold every criterion already uses."""
    logits = torch.full((1, 2, 2), 10.0)          # predicts everything
    target = torch.tensor([[0.9, 0.4], [0.2, 0.8]])
    res = sam_mask_loss(logits, torch.tensor([0.0]), target)
    assert float(res["iou_true"][0]) == pytest.approx(2 / 4)
    assert samdec.IOU_GT_THRESHOLD == 0.5


def test_wta_backpropagates_only_from_the_lowest_loss_candidate():
    target = torch.ones(4, 4)
    good = torch.full((4, 4), 4.0, requires_grad=True)
    bad1 = torch.full((4, 4), -4.0, requires_grad=True)
    bad2 = torch.full((4, 4), -8.0, requires_grad=True)
    logits = torch.stack([bad1, good, bad2])
    res = sam_mask_loss(logits, torch.tensor([0.1, 0.2, 0.3]), target)
    assert res["sel"] == 1
    res["total"].backward()
    assert good.grad is not None and float(good.grad.abs().sum()) > 0
    assert bad1.grad is None or float(bad1.grad.abs().sum()) == 0.0
    assert bad2.grad is None or float(bad2.grad.abs().sum()) == 0.0


def test_the_iou_term_is_taken_at_the_winner_index():
    """``loss_fns.py:277-282`` -- "to be consistent w/ SAM"."""
    target = torch.ones(4, 4)
    logits = torch.stack([torch.full((4, 4), -4.0), torch.full((4, 4), 4.0)])
    iou_pred = torch.tensor([0.9, 0.25])
    res = sam_mask_loss(logits, iou_pred, target)
    assert res["sel"] == 1
    # candidate 1 predicts everything -> real IoU 1.0; the term is (0.25 - 1)^2
    assert float(res["iou_true"][1]) == pytest.approx(1.0)
    assert float(res["iouhead"]) == pytest.approx(0.5625, rel=1e-6)


def test_loss_ablations_switch_exactly_one_term_each():
    logits = torch.zeros(1, 4, 4)
    target = torch.ones(4, 4)
    full = sam_mask_loss(logits, torch.tensor([0.0]), target,
                         SamLossConfig(kind="focal_dice"))
    no_dice = sam_mask_loss(logits, torch.tensor([0.0]), target,
                            SamLossConfig(kind="focal"))
    bce_dice = sam_mask_loss(logits, torch.tensor([0.0]), target,
                             SamLossConfig(kind="bce_dice"))
    assert float(no_dice["dice"]) == 0.0
    assert float(no_dice["focal"]) == pytest.approx(float(full["focal"]))
    assert float(bce_dice["focal"]) == pytest.approx(LN2, rel=1e-6)   # plain BCE
    assert float(bce_dice["dice"]) == pytest.approx(float(full["dice"]))
    with pytest.raises(ValueError, match="samdec-loss"):
        SamLossConfig(kind="nope")


def test_iou_head_off_drops_the_iou_term():
    res = sam_mask_loss(torch.zeros(1, 4, 4), None, torch.ones(4, 4),
                        SamLossConfig(iou_head=False))
    assert float(res["iouhead"]) == 0.0
    assert float(res["total"]) == pytest.approx(20 * 0.25 * LN2 * 0.25 + 0.32,
                                                rel=1e-6)


def test_the_target_must_be_on_the_decoder_grid():
    with pytest.raises(ValueError, match=r"4\*gh"):
        sam_mask_loss(torch.zeros(1, 8, 8), None, torch.ones(4, 4))


# --------------------------------------------------------------------------- #
# initialisation / RNG
# --------------------------------------------------------------------------- #
def test_same_seed_gives_the_same_head_regardless_of_the_global_stream():
    torch.manual_seed(1234)
    a = SAMDecHead(in_dim=16, text_dim=32, seed=7)
    torch.rand(17)                                  # move the global stream on
    b = SAMDecHead(in_dim=16, text_dim=32, seed=7)
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb and torch.equal(pa, pb)
    assert torch.equal(a.pe_layer.positional_encoding_gaussian_matrix,
                       b.pe_layer.positional_encoding_gaussian_matrix)
    c = SAMDecHead(in_dim=16, text_dim=32, seed=8)
    assert not torch.equal(a.neck[0].weight, c.neck[0].weight)


def test_building_the_head_does_not_consume_the_global_rng():
    """The trainer's sample order and the negative controls draw from the global
    stream; a head that shifts it would change the data an arm sees."""
    torch.manual_seed(99)
    reference = torch.randn(5)
    torch.manual_seed(99)
    SAMDecHead(in_dim=16, text_dim=32, seed=7)
    assert torch.equal(reference, torch.randn(5))


# --------------------------------------------------------------------------- #
# the trainer seam
# --------------------------------------------------------------------------- #
class _X:
    """The fields ``compute_micro_batch`` and this arm's loss actually read."""

    def __init__(self, gh=3, gw=4, dim=16, tdim=32, is_fake=False, pix=True):
        self.sample_id = "s1"
        self.feat = torch.randn(1, dim, gh, gw)
        self.sim = self.center = self.geom = self.guide_hi = None
        self.cond_h = torch.randn(1, 5, tdim)
        self.cond_mask = torch.ones(1, 5, dtype=torch.bool)
        self.word_ids = torch.tensor([0])
        self.word_offsets = torch.tensor([0])
        self.phi_dir = torch.randn(gh * gw, 71)
        self.gt_low = torch.rand(gh, gw)
        self.gt_hi = self.gt_partner_low = None
        self.grid_h, self.grid_w = gh, gw
        self.is_fake = is_fake
        self.family = "radial"
        self.route_semantic = False
        self.h_cond = torch.randn(1, tdim)
        self.gt_pix = torch.rand(4 * gh, 4 * gw) if pix else None
        self.gt_pix_source = "cgt1024"
        self.pixgt = None
        self.readout = {"kind": "seg_where"}
        self.meta = {}


def _model(**kw):
    from q3vl.whereb.amort.model import AmortModel

    return AmortModel("SAMDEC", in_dim=16, cond_text_dim=32, **kw)


def test_amort_model_builds_the_head_with_the_b4_defaults():
    m = _model()
    assert isinstance(m.geo, SAMDecHead)
    assert m.is_new_arm and m.new_arm_defaults
    assert m.sem is None and m.cond_frozen
    assert not any(p.requires_grad for p in m.cond.parameters())
    assert m.facts()["arm_head"]["arm"] == "SAMDEC"


def test_compute_micro_batch_emits_the_pre_registered_witness_columns():
    from q3vl.whereb.amort.losses import LossWeights
    from q3vl.whereb.amort.trainer import compute_micro_batch

    m = _model()
    total, stats, rows = compute_micro_batch(m, [_X(), _X(is_fake=True)],
                                             LossWeights())
    assert total.requires_grad
    for col in ("L_focal", "L_dice", "L_iouhead", "L_sup_cells"):
        assert col in stats, col
    assert stats["L_sup_cells"] == pytest.approx(4 * 3 * 4 * 4)   # 16x the grid
    samdec.assert_steps_row(stats)
    # the seven-term ST_LANG stack did not run
    assert not any(k.startswith(("L_bce", "L_sdf", "L_area", "L_sep"))
                   for k in stats)
    # the trainer's own early-warning columns survive
    assert "area_ratio_median" in stats and "std_ratio_median" in stats
    assert rows[0]["gt_pix_source"] == "cgt1024"
    assert rows[0]["samdec_n_cand"] == 3


def test_the_fake_stream_is_supervised_against_an_empty_target():
    from q3vl.whereb.amort.arms import ArmContext
    from q3vl.whereb.amort.losses import LossWeights

    m = _model()
    x = _X(is_fake=True)
    x.gt_pix = torch.ones(4 * x.grid_h, 4 * x.grid_w)      # a full-frame mask
    out = samdec.forward(m, m.geo, ArmContext(feat=x.feat, grid_h=x.grid_h,
                                              grid_w=x.grid_w, h_cond=x.h_cond))
    sl = samdec.compute_loss(m, out, x, LossWeights())
    assert torch.isfinite(sl.total)
    # the target really was zeroed: the IoU of any prediction against an empty
    # GT is 0, so the head's regression target is 0 as well
    assert sl.stats["samdec_iou_true"] == 0.0


def test_a_missing_pixel_gt_names_the_flag_that_produces_it():
    from q3vl.whereb.amort.arms import ArmContext
    from q3vl.whereb.amort.losses import LossWeights

    m = _model()
    x = _X(pix=False)
    out = samdec.forward(m, m.geo, ArmContext(feat=x.feat, grid_h=x.grid_h,
                                              grid_w=x.grid_w, h_cond=x.h_cond))
    with pytest.raises(ValueError, match="no-pixgt"):
        samdec.compute_loss(m, out, x, LossWeights())
    x2 = _X()
    x2.gt_pix = torch.rand(5, 5)
    with pytest.raises(AssertionError, match=r"4\*gh"):
        samdec.compute_loss(m, out, x2, LossWeights())


def test_assert_steps_row_refuses_a_row_that_lost_a_column():
    row = {"L_focal": 0.1, "L_dice": 0.2, "L_iouhead": 0.0, "L_sup_cells": 192.0}
    assert samdec.assert_steps_row(row)["sup_cells"] == 192.0
    for drop in ("L_focal", "L_dice", "L_iouhead", "L_sup_cells"):
        bad = {k: v for k, v in row.items() if k != drop}
        with pytest.raises(AssertionError, match="witness"):
            samdec.assert_steps_row(bad)
    with pytest.raises(AssertionError, match="cannot be empty"):
        samdec.assert_steps_row({**row, "L_sup_cells": 0.0})


def test_train_stats_hook_columns():
    from q3vl.whereb.amort.arms import ArmContext

    m = _model()
    x = _X()
    out = samdec.forward(m, m.geo, ArmContext(feat=x.feat, grid_h=x.grid_h,
                                              grid_w=x.grid_w, h_cond=x.h_cond))
    st = samdec.train_stats(out, x)
    assert st["samdec_sup_cells"] == 4 * 3 * 4 * 4
    assert st["samdec_n_cand"] == 3 and st["samdec_n_prompt_tokens"] == 1


def test_two_real_optimizer_steps_write_the_witness_row_to_steps_jsonl(tmp_path):
    """End to end on CPU: the loop, the SAM optimiser, the SAM schedule, and the
    file the runtime assertion reads afterwards."""
    import json

    from q3vl.whereb.amort.losses import LossWeights
    from q3vl.whereb.amort.trainer import AmortTrainConfig, AmortTrainer

    class _Builder:
        genctx = None

        def build(self, samples, modes):
            return [_X(is_fake=(m == "foreign")) for m in modes]

        def facts(self):
            return {}

    m = _model()
    cfg = AmortTrainConfig(arm="SAMDEC", micro_batch=2, effective_batch=2,
                           max_steps=2, eval_steps=1000, save_steps=1000,
                           precision="fp32", scheduler="multistep")
    tr = AmortTrainer(m, _Builder(), list(range(8)), list(range(8)), cfg,
                      LossWeights(), run_dir=tmp_path, device="cpu",
                      optimizer_spec=samdec.optimizer_spec(_args()),
                      scheduler_kwargs=samdec.scheduler_kwargs(_args(), 2))
    before = m.geo.prompt_proj.weight.detach().clone()
    tr.train()
    rows = [json.loads(l) for l in (tmp_path / "steps.jsonl").read_text().splitlines() if l]
    assert rows and rows[0]["step"] == 1
    assert samdec.assert_steps_row(rows[0])["sup_cells"] == 4 * 3 * 4 * 4
    assert not torch.equal(before, m.geo.prompt_proj.weight.detach())
    assert tr.setup()["optimizer_spec"]["lr"] == 8e-4


# --------------------------------------------------------------------------- #
# the evaluator seam and the pre-registered column
# --------------------------------------------------------------------------- #
def _eval_row(model, x):
    from q3vl.whereb.amort.arms import ArmContext

    out = samdec.forward(model, model.geo,
                         ArmContext(feat=x.feat, grid_h=x.grid_h,
                                    grid_w=x.grid_w, h_cond=x.h_cond))
    return samdec.per_sample_row(model, out, x)


def test_per_sample_row_carries_the_candidate_diagnostics():
    m = _model()
    row = _eval_row(m, _X())
    assert row["samdec_n_cand"] == 3 and len(row["samdec_cand_ious"]) == 3
    assert 0.0 <= row["samdec_best_of_3"] <= 1.0
    assert row["samdec_best_of_3"] >= row["samdec_sel_iou"] - 1e-9
    assert row["samdec_iou_mae"] == pytest.approx(
        abs(row["samdec_iou_pred"] - row["samdec_sel_iou"]))
    assert row["samdec_sel_is_best"] == (row["samdec_sel"] == row["samdec_best_cand"])


def test_criteria_columns_feed_the_runtime_assertion():
    from q3vl.whereb.amort.evaluate import assert_criteria_ran

    m = _model()
    rows = [_eval_row(m, _X()) for _ in range(3)]
    cols = samdec.criteria_columns(rows)
    assert set(cols) == {"samdec_cand"}
    block = cols["samdec_cand"]
    assert block["n"] == 3 and block["n_cand"] == 3
    assert block["iou_mae"]["n"] == 3
    assert sum(block["sel_hist"].values()) == 3
    rep = assert_criteria_ran({"criteria_columns": cols}, "SAMDEC")
    assert rep["required"] == ["samdec_cand"] and rep["computed"]["samdec_cand"] == 3


def test_an_empty_column_refuses_the_board():
    from q3vl.whereb.amort.evaluate import assert_criteria_ran

    cols = samdec.criteria_columns([])
    assert cols["samdec_cand"]["n"] == 0
    with pytest.raises(AssertionError, match="cannot adjudicate"):
        assert_criteria_ran({"criteria_columns": cols}, "SAMDEC")


def test_head_facts_declare_no_deep_supervision_branches():
    """``deep_supervision_tags`` reads the head's facts; SAM §A says auxiliary
    deep supervision is unhelpful, so this arm must promise none."""
    from q3vl.whereb.amort.evaluate import deep_supervision_tags

    assert deep_supervision_tags(SAMDecHead(in_dim=16, text_dim=32).facts()) == []


# --------------------------------------------------------------------------- #
# optimiser / schedule / builder hooks
# --------------------------------------------------------------------------- #
def test_optimizer_spec_is_sam_appendix_a():
    spec = samdec.optimizer_spec(_args())
    assert spec.type == "adamw"
    assert spec.lr == 8e-4 and spec.weight_decay == 0.1
    assert spec.betas is None          # -> torch default (0.9, 0.999) = SAM's
    assert spec.grouping == "dim"


def test_the_optimizer_really_gets_8e_4_and_the_dim_grouping():
    """§3.2: wd 0.1 on the ``dim > 1`` group, 0 on norms/biases (this repo's
    split; SAM's training code was never released -- NOTES 3)."""
    from q3vl.whereb.amort.trainer import AmortTrainConfig, build_optimizer

    m = _model()
    opt = build_optimizer(m, AmortTrainConfig(arm="SAMDEC"),
                          samdec.optimizer_spec(_args()))
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.param_groups[0]["lr"] == 8e-4
    assert opt.defaults["betas"] == (0.9, 0.999)          # SAM §A
    assert [g["weight_decay"] for g in opt.param_groups] == [0.1, 0.0]
    trainable = {id(p) for p in m.parameters() if p.requires_grad}
    grouped = {id(p) for g in opt.param_groups for p in g["params"]}
    assert grouped == trainable                            # nothing dropped
    # the frozen CondEncoder is not in any group (B-4 parameter audit)
    assert not (grouped & {id(p) for p in m.cond.parameters()})


def test_scheduler_kwargs_carry_sam_s_schedule_onto_1200_steps():
    kw = samdec.scheduler_kwargs(_args(), 1200)
    assert kw == {"warmup_steps": 3, "milestones": [800, 1156], "gamma": 0.1}


def test_the_schedule_really_decays_by_10x_at_those_steps():
    from q3vl.where.calibrate import make_scheduler

    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))], lr=8e-4)
    sched = make_scheduler(opt, 1200, 0.03, "multistep",
                           **samdec.scheduler_kwargs(_args(), 1200))
    f = sched.lr_lambdas[0]
    assert f(0) == pytest.approx(1 / 3)      # linear warmup over 3 steps
    assert f(3) == pytest.approx(1.0)
    assert f(799) == pytest.approx(1.0)
    assert f(800) == pytest.approx(0.1)
    assert f(1155) == pytest.approx(0.1)
    assert f(1156) == pytest.approx(0.01)


def test_builder_kwargs_ask_for_the_4x_pixel_gt():
    kw = samdec.builder_kwargs(_args())
    assert kw["pixgt_size"](32, 48) == (128, 192)
    assert kw["pixgt_size"](3, 4) == (12, 16)


def test_readout_spec_passes_the_flag_through_and_guards_the_span_row():
    spec = samdec.readout_spec(_args())
    assert spec.kind == "seg_where" and spec.n_vectors == 1
    assert samdec.readout_spec(_args(cond_readout="im_end")).kind == "im_end"
    samdec.set_variant(prompt="span")
    with pytest.raises(SystemExit, match="where_span_pool"):
        samdec.readout_spec(_args())
    assert samdec.readout_spec(_args(cond_readout="where_span_pool")).kind == \
        "where_span_pool"


def test_config_of_prefers_the_run_seed():
    assert samdec.config_of(_args(seed=4242))["seed"] == 4242
    assert samdec.config_of(None)["seed"] == samdec.VARIANT["seed"]


def test_set_variant_rejects_a_typo():
    with pytest.raises(KeyError, match="unknown SAMDEC variant"):
        samdec.set_variant(w_focl=20.0)


# --------------------------------------------------------------------------- #
# the entry script
# --------------------------------------------------------------------------- #
def test_the_entry_exposes_every_flag_of_the_proposal():
    from q3vl.whereb.scripts import run_samdec_arm

    ap = argparse.ArgumentParser(add_help=False)
    samdec.add_arguments(ap)
    txt = ap.format_help()
    for flag in ("--samdec-multimask", "--samdec-loss", "--samdec-gt",
                 "--samdec-iou-head", "--no-samdec-iou-head", "--samdec-prompt",
                 "--samdec-lr", "--samdec-wd", "--samdec-sched",
                 "--samdec-focal-alpha", "--samdec-focal-gamma",
                 "--samdec-w-focal", "--samdec-w-dice", "--samdec-w-iou"):
        assert flag in txt, flag
    assert run_samdec_arm.GT_TO_PIXGT == {"png": "cgt1024", "raster": "render"}


def test_the_entry_derives_the_shared_flags_and_records_them():
    from q3vl.whereb.scripts.run_samdec_arm import plan

    own, rest, setup = plan(["--run-name", "amort_SAMDEC", "--max-steps", "1200"])
    assert rest[:2] == ["--arm", "SAMDEC"]
    assert "--pixgt-source" in rest and rest[rest.index("--pixgt-source") + 1] == "cgt1024"
    assert rest[rest.index("--scheduler") + 1] == "multistep"
    assert float(rest[rest.index("--lr") + 1]) == 8e-4
    assert float(rest[rest.index("--weight-decay") + 1]) == 0.1
    assert "--max-steps" in rest and "1200" in rest
    assert setup["variant"]["multimask"] == 3 and setup["variant"]["loss"] == "focal_dice"
    assert setup["loss"]["w_focal"] == 20.0 and setup["loss"]["w_dice"] == 1.0
    assert setup["loss"]["focal_alpha"] == 0.25 and setup["loss"]["focal_gamma"] == 2.0
    assert setup["pretrained_weights_loaded"] is False
    assert setup["cli_flags"]["samdec_iou_head"] is True


def test_the_entry_maps_the_raster_ablation_and_refuses_a_contradiction():
    from q3vl.whereb.scripts.run_samdec_arm import plan

    _own, rest, _s = plan(["--samdec-gt", "raster"])
    assert rest[rest.index("--pixgt-source") + 1] == "render"
    with pytest.raises(SystemExit, match="Pick one"):
        plan(["--samdec-gt", "png", "--pixgt-source", "render"])
    with pytest.raises(SystemExit, match="cannot be used with SAMDEC"):
        plan(["--no-pixgt"])
    with pytest.raises(SystemExit, match="run_samdec_arm is the SAMDEC entry"):
        plan(["--arm", "UNIQ"])


def test_the_entry_records_itself_in_the_run_setup_seam(monkeypatch, tmp_path):
    """``arms.ARM_SETUP`` is what ``run_amort_arm`` merges into
    ``run_setup.json["new_arm"]``; the record has to be there before the run,
    not reconstructed from the command line afterwards."""
    import json

    from q3vl.whereb.amort import arms as A
    from q3vl.whereb.scripts import run_samdec_arm

    monkeypatch.setattr(A, "ARM_SETUP", {})
    calls: dict[str, object] = {}

    def _no_write(rest, setup):
        calls["rest"] = rest
        return None

    def _fake_base(rest):
        calls["delegated"] = rest
        return 3

    monkeypatch.setattr(run_samdec_arm, "_write_setup", _no_write)
    monkeypatch.setattr("q3vl.whereb.scripts.run_amort_arm.main", _fake_base)
    rc = run_samdec_arm.main(["--samdec-loss", "bce_dice", "--out-root",
                              str(tmp_path)])
    assert rc == 3
    rec = A.ARM_SETUP["samdec"]
    assert rec["arm"] == "SAMDEC" and rec["criteria"] == ["samdec_cand"]
    assert rec["variant"]["loss"] == "bce_dice"
    assert rec["derived_flags"]["scheduler"] == "multistep"
    json.dumps(rec)                     # must be serialisable into run_setup.json


def test_the_entry_writes_its_own_frozen_record(tmp_path, monkeypatch):
    import json

    from q3vl.whereb.scripts.run_samdec_arm import _write_setup, plan

    _own, rest, setup = plan(["--out-root", str(tmp_path), "--run-name", "r1"])
    run_dir = _write_setup(rest, setup)
    rec = json.loads((run_dir / "config" / "samdec_setup.json").read_text())
    assert len(rec["samdec_sha256"]) == 64 and len(rec["wrapper_sha256"]) == 64
    assert rec["schedule_source"]["warmup_iters_of_90k"] == 250
    assert rec["sources"], "the upstream files this port was written against"


def test_the_entry_propagates_the_variant_to_the_head():
    from q3vl.whereb.scripts.run_samdec_arm import plan

    plan(["--samdec-multimask", "1", "--no-samdec-iou-head",
          "--samdec-loss", "focal"])
    head = samdec.build_head(in_dim=16, text_dim=32, args=None)
    assert head.n_candidates == 1 and head.use_iou_head is False
    assert samdec.loss_config(None).kind == "focal"


# --------------------------------------------------------------------------- #
# not selected = not there
# --------------------------------------------------------------------------- #
def test_the_live_arms_do_not_import_or_see_this_module():
    import q3vl.whereb.amort.model as model_mod
    from q3vl.whereb.amort.losses import LossWeights
    from q3vl.whereb.amort.model import ARMS, AmortModel

    assert "samdec" not in inspect.getsource(model_mod)
    assert ARMS == ("P1", "P3prime", "SHAPE3", "UNIQ")
    # no `sam_*` weight was added to the shared loss record
    assert not any(k.startswith("sam_") for k in LossWeights().to_dict())
    m = AmortModel("P1", in_dim=8, ch=8, n_blocks=1, cond_text_dim=16)
    assert m.is_new_arm is False and m.sem is not None and not m.cond_frozen
