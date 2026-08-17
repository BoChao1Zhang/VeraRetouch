"""EPR-022 MATTE: the port is the port, and the port is WIRED.

CPU only (``CUDA_VISIBLE_DEVICES=""`` is enough; nothing here touches a GPU).

What is pinned, and why each one:

* the channel table, the fusion in/out chain and the parameter count against the
  numbers computed by hand from ``detail_capture.py`` -- a silently different
  ``in_chans`` produces a head that trains and a board that means something else;
* the four criterion terms against closed-form arithmetic, including the literal
  ``262144`` and the ``0.01`` sparsity pair, and both NOVEL zero guards **with
  the un-guarded value shown next to them** so the guard is visibly doing work;
* step 0 = the unconditional ViTMatte (FiLM last layer zero-initialised), which
  is the only initialisation claim the proposal makes;
* the diagnostic control's RNG is private, and building the head does not move
  the global stream that every other module's initialisation is drawn from;
* the pre-registered ``pix_readout`` column really refuses an empty board;
* the whole seam end to end -- ``AmortModel.forward_geo`` ->
  ``compute_micro_batch`` -> ``assert_criteria_ran`` -- on a toy grid.
"""

from __future__ import annotations

import argparse
import math

import pytest
import torch
import torch.nn.functional as F

from q3vl.whereb.amort import arms as A
from q3vl.whereb.amort import matte as M
from q3vl.whereb.amort.data import AmortSampleInputs

# Toy F_pre grid.  4x6 rather than 2x3 because the ported 5-level laplacian
# pyramid reflect-pads by 2 at every level and therefore needs min(H, W) >= 48 --
# a real property of the port (spec-5's short side is 512), pinned below.
GH, GW = 4, 6
H, W = GH * 16, GW * 16            # 64 x 96


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def test_registry_loads_matte_and_the_names_agree():
    mod = A.load_arm("MATTE")
    assert mod is M
    assert M.ARM == "MATTE"
    assert M.CRITERIA == A.ARM_CRITERIA["MATTE"] == ("pix_readout",)
    for hook in A.REQUIRED_HOOKS:
        assert callable(getattr(M, hook))


@pytest.mark.parametrize("hook", ["add_arguments", "head_kwargs_from_args",
                                  "optimizer_spec", "scheduler_kwargs",
                                  "builder_kwargs", "per_sample_row",
                                  "criteria_columns", "train_stats"])
def test_optional_hooks_are_all_provided(hook):
    assert A.arm_hook("MATTE", hook) is not None


# --------------------------------------------------------------------------- #
# the port: channel table + parameter count
# --------------------------------------------------------------------------- #
def test_detail_capture_channel_table_is_vitmattes():
    d = M.Detail_Capture(in_chans=1024, img_chans=3)
    # ConvStream: conv_chans = [img_chans] + out_chans   (detail_capture.py:40-41)
    assert d.conv_chans == [3, 48, 96, 192]
    # fus_channs = [in_chans] + fusion_out                (L118-119)
    assert d.fus_channs == [1024, 256, 128, 64, 32]
    ins = [b.conv.conv.in_channels for b in d.fusion_blks]
    outs = [b.conv.conv.out_channels for b in d.fusion_blks]
    # in = fus_channs[i] + conv_chans[-(i+1)]             (L123)
    assert ins == [1024 + 192, 256 + 96, 128 + 48, 64 + 3] == [1216, 352, 176, 67]
    assert outs == [256, 128, 64, 32]
    # Basic_Conv3x3: 3x3, bias=False, stride 2 in the stream / 1 in the fusion
    c0 = d.convstream.convs[0].conv
    assert (c0.kernel_size, c0.stride, c0.padding, c0.bias) == ((3, 3), (2, 2), (1, 1), None)
    assert d.fusion_blks[0].conv.conv.stride == (1, 1)
    # Matting_Head 32 -> 16 (3x3) -> 1 (1x1), both with the default bias
    mh = list(d.matting_head.matting_convs)
    assert (mh[0].in_channels, mh[0].out_channels, mh[0].kernel_size) == (32, 16, (3, 3))
    assert (mh[3].in_channels, mh[3].out_channels, mh[3].kernel_size) == (16, 1, (1, 1))
    assert mh[0].bias is not None and mh[3].bias is not None


def test_parameter_count_matches_the_proposals_hand_count():
    head = M.MatteHead(in_dim=1024, text_dim=2560, img_source="luma")
    # luma path is 1 input channel; the proposal counts the 3-channel one
    head3 = M.MatteHead(in_dim=1024, text_dim=2560, img_source="rgb") \
        if M._rgb_available() else None
    convstream = (3 * 48 * 9 + 48 * 96 * 9 + 96 * 192 * 9) + 2 * (48 + 96 + 192)
    fusion = (1216 * 256 * 9 + 352 * 128 * 9 + 176 * 64 * 9 + 67 * 32 * 9) \
        + 2 * (256 + 128 + 64 + 32)
    mhead = (32 * 16 * 9 + 16) + 2 * 16 + (16 * 1 + 1)
    film = (2560 * 256 + 256) + (256 * 960 + 960)
    want = convstream + fusion + mhead + film
    assert want == 4_445_137                       # 4.4M, the proposal's figure
    # the luma head differs in two places: the ConvStream's first conv AND the
    # last fusion block, whose input is `fus_channs[3] + conv_chans[0]`
    assert head.n_params() == want - (3 - 1) * 48 * 9 - (3 - 1) * 32 * 9
    if head3 is not None:
        assert head3.n_params() == want
    assert abs(want / 1e6 - 4.4) < 0.05


def test_res_half_stops_one_level_early():
    head = M.MatteHead(in_dim=8, text_dim=6, res=0.5, img_source="luma")
    assert len(head.detail.fusion_blks) == 3
    assert head.fusion_out == (256, 128, 64)
    assert head.out_stride == 2
    ins = [b.conv.conv.in_channels for b in head.detail.fusion_blks]
    assert ins == [8 + 192, 256 + 96, 128 + 48]


# --------------------------------------------------------------------------- #
# shapes + step-0 properties
# --------------------------------------------------------------------------- #
def _head(**kw):
    kw.setdefault("in_dim", 8)
    kw.setdefault("text_dim", 6)
    kw.setdefault("img_source", "luma")
    return M.MatteHead(**kw)


def test_forward_shapes_and_alpha_domain():
    head = _head()
    feat = torch.randn(1, 8, GH, GW)
    img = torch.randn(1, 1, H, W)
    alpha = head(feat, img, torch.randn(6))
    assert tuple(alpha.shape) == (1, 1, H, W)
    a = alpha.detach()
    assert float(a.min()) > 0.0 and float(a.max()) < 1.0


def test_forward_shapes_at_half_resolution():
    head = _head(res=0.5)
    alpha = head(torch.randn(1, 8, GH, GW), torch.randn(1, 1, H, W), torch.randn(6))
    assert tuple(alpha.shape) == (1, 1, H // 2, W // 2)


def test_step0_is_the_unconditional_vitmatte():
    """FiLM's last layer is zero, so the condition cannot move step 0."""
    head = _head()
    assert torch.count_nonzero(head.film_mlp[-1].weight) == 0
    assert torch.count_nonzero(head.film_mlp[-1].bias) == 0
    feat, img = torch.randn(1, 8, GH, GW), torch.randn(1, 1, H, W)
    a = head(feat, img, torch.randn(6) * 10.0)
    b = head(feat, img, torch.randn(6) * -3.0)
    assert torch.allclose(a, b, atol=0, rtol=0)
    # ...and it stops being a no-op the moment the layer is not zero
    with torch.no_grad():
        head.film_mlp[-1].bias.add_(0.5)
    assert not torch.allclose(a, head(feat, img, torch.randn(6)))


def test_concat_conditioning_widens_the_convstream_input():
    head = _head(cond="concat")
    assert head.img_chans == 1 + M.CONCAT_COND_CH
    assert head.film_mlp is None and head.cond_proj is not None
    alpha = head(torch.randn(1, 8, GH, GW), torch.randn(1, 1, H, W), torch.randn(6))
    assert tuple(alpha.shape) == (1, 1, H, W)
    # the condition DOES reach the output on this row (no zero-init claim here)
    a = head(torch.zeros(1, 8, GH, GW), torch.zeros(1, 1, H, W), torch.ones(6))
    b = head(torch.zeros(1, 8, GH, GW), torch.zeros(1, 1, H, W), -torch.ones(6))
    assert not torch.allclose(a, b)


def test_bn_row_swaps_the_norm_and_gn8_is_the_default():
    assert isinstance(_head().detail.convstream.convs[0].bn, torch.nn.GroupNorm)
    bn = _head(norm="bn")
    assert isinstance(bn.detail.convstream.convs[0].bn, torch.nn.BatchNorm2d)


# --------------------------------------------------------------------------- #
# the criterion, against closed-form arithmetic
# --------------------------------------------------------------------------- #
def _pair(pred, gt):
    return {"phas": pred}, {"phas": gt}


def test_unknown_l1_uses_the_literal_262144():
    crit = M.MattingCriterion()
    pred = torch.ones(1, 1, 2, 2)
    gt = torch.zeros(1, 1, 2, 2)
    smap = torch.tensor([[[[1., 1.], [0., 0.]]]])
    got = crit.unknown_l1_loss(smap, *_pair(pred, gt))["unknown_l1_loss"]
    # l1_loss over the WHOLE tensor (mean of [1,1,0,0]) * (B*262144/sum(map))
    assert float(got) == pytest.approx(0.5 * (1 * 262144 / 2.0))
    assert M.NORM_CONST == 262144


def test_known_l1_is_the_complement_and_keeps_its_own_guard():
    crit = M.MattingCriterion()
    pred = torch.ones(1, 1, 2, 2)
    gt = torch.zeros(1, 1, 2, 2)
    smap = torch.tensor([[[[1., 1.], [0., 0.]]]])
    got = crit.known_l1_loss(smap, *_pair(pred, gt))["known_l1_loss"]
    assert float(got) == pytest.approx(0.5 * (1 * 262144 / 2.0))
    # sample_map all ones -> the complement is empty -> the original's scale = 0
    full = torch.ones(1, 1, 2, 2)
    assert float(crit.known_l1_loss(full, *_pair(pred, gt))["known_l1_loss"]) == 0.0


def test_the_two_novel_zero_guards_are_what_stops_a_nan():
    """An all-hard GT (every is_fake sample) makes both denominators 0."""
    pred = torch.rand(1, 1, 4, 4)
    gt = torch.zeros(1, 1, 4, 4)
    smap = torch.zeros(1, 1, 4, 4)
    guarded = M.MattingCriterion()
    u = guarded.unknown_l1_loss(smap, *_pair(pred, gt))["unknown_l1_loss"]
    g = guarded.loss_gradient_penalty(smap, *_pair(pred, gt))["loss_gradient_penalty"]
    assert float(u) == 0.0 and float(g) == 0.0
    raw = M.MattingCriterion(guard_unknown=False, guard_grad=False)
    ru = raw.unknown_l1_loss(smap, *_pair(pred, gt))["unknown_l1_loss"]
    rg = raw.loss_gradient_penalty(smap, *_pair(pred, gt))["loss_gradient_penalty"]
    assert not torch.isfinite(ru) and not torch.isfinite(rg)


def test_gradient_penalty_is_sobel_plus_the_0p01_sparsity_pair():
    crit = M.MattingCriterion()
    pred = torch.rand(1, 1, 6, 6)
    smap = torch.ones(1, 1, 6, 6)
    got = crit.loss_gradient_penalty(smap, *_pair(pred, pred))["loss_gradient_penalty"]
    kx = torch.tensor([[[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]])
    ky = torch.tensor([[[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]]])
    dx = F.conv2d(pred, kx, padding=1)
    dy = F.conv2d(pred, ky, padding=1)
    scale = 1 * 262144 / 36.0
    want = 0.01 * (dx.abs().mean() + dy.abs().mean()) * scale
    assert float(got) == pytest.approx(float(want), rel=1e-5)
    assert M.GRAD_SPARSITY == 0.01


def test_laplacian_loss_is_zero_on_an_exact_match_and_uses_five_levels():
    x = torch.rand(1, 1, 64, 64)
    assert float(M.laplacian_loss(x, x)) == pytest.approx(0.0, abs=1e-6)
    assert M.LAP_MAX_LEVELS == 5
    k = M.gauss_kernel()
    assert tuple(k.shape) == (1, 1, 5, 5)
    assert float(k.sum()) == pytest.approx(1.0)
    assert len(M.laplacian_pyramid(x, k, 5)) == 5


def test_crop_to_even_size_and_the_pyramid_helpers():
    odd = torch.rand(1, 1, 7, 9)
    assert tuple(M.crop_to_even_size(odd).shape[-2:]) == (6, 8)
    k = M.gauss_kernel()
    d = M.downsample(torch.rand(1, 1, 8, 8), k)
    assert tuple(d.shape[-2:]) == (4, 4)
    assert tuple(M.upsample(d, k).shape[-2:]) == (8, 8)


def test_the_four_names_and_their_order_are_the_configs():
    assert M.LOSS_NAMES == ("unknown_l1_loss", "known_l1_loss",
                            "loss_pha_laplacian", "loss_gradient_penalty")
    assert M.parse_losses("grad,unknown_l1") == ("unknown_l1_loss",
                                                 "loss_gradient_penalty")
    assert M.parse_losses(M.DEFAULT_LOSSES) == M.LOSS_NAMES
    with pytest.raises(ValueError, match="unknown --matte-losses entry"):
        M.parse_losses("iou")
    with pytest.raises(ValueError, match="selected no term"):
        M.parse_losses("")


def test_criterion_dispatch_matches_the_reference_signature_split():
    crit = M.MattingCriterion()
    pred, gt = torch.rand(1, 1, 64, 64), torch.rand(1, 1, 64, 64)
    smap = (gt > 0.4).float()
    got = crit(smap, {"phas": pred}, {"phas": gt})
    assert set(got) == set(M.LOSS_NAMES)
    assert all(torch.isfinite(v) for v in got.values())


# --------------------------------------------------------------------------- #
# GT policy: is_fake, the D-9 audit
# --------------------------------------------------------------------------- #
def _inputs(**kw):
    base = dict(
        sample_id="s1", feat=torch.randn(1, 8, GH, GW), sim=None, center=None,
        cond_h=torch.zeros(1, 3, 6), cond_mask=torch.ones(1, 3, dtype=torch.bool),
        word_ids=torch.tensor([0]), word_offsets=torch.tensor([0]),
        phi_dir=torch.zeros(GH * GW, 71), guide_hi=torch.rand(1, 1, H, W),
        gt_low=torch.rand(GH, GW), gt_hi=torch.rand(H, W), gt_partner_low=None,
        grid_h=GH, grid_w=GW, is_fake=False, family="radial",
        route_semantic=False, h_cond=torch.randn(1, 6),
        gt_pix=torch.rand(H, W), gt_pix_source="render")
    base.update(kw)
    return AmortSampleInputs(**base)


def test_is_fake_zeroes_the_target_and_all_four_terms_stay_finite():
    head = _head()
    x = _inputs(is_fake=True)
    gt = head.target_of(x, (H, W))
    assert float(gt.abs().sum()) == 0.0
    smap = ((gt > 0) & (gt < 1)).float()
    assert float(smap.sum()) == 0.0            # -> both novel guards engage
    terms = head.criterion(smap, {"phas": torch.rand(1, 1, H, W)}, {"phas": gt})
    assert set(terms) == set(M.LOSS_NAMES)
    assert all(torch.isfinite(v) for v in terms.values())
    assert head.rt.n_fake == 1


def test_the_d9_audit_falls_back_and_counts_and_stays_fallen_back():
    head = _head(audit_n=10, audit_tol=0.02)
    render = torch.zeros(H, W)
    raster = torch.full((H, W), 0.5)           # mean-abs 0.5, way over tolerance
    x = _inputs(sample_id="bad", gt_pix=render, gt_hi=raster)
    got = head.target_of(x, (H, W))
    assert float(got.mean()) == pytest.approx(0.5)
    assert head.rt.audit_fail == 1 and head.rt.audit_pass == 0
    assert head.rt.audit_worst == pytest.approx(0.5)
    assert "bad" in head.audit_failed
    # sticky: the same sample keeps the raster even after the budget is spent
    head.rt.audit_n = 999
    assert float(head.target_of(x, (H, W)).mean()) == pytest.approx(0.5)


def test_the_d9_audit_keeps_the_render_when_it_agrees():
    head = _head(audit_n=10, audit_tol=0.02)
    render = torch.zeros(H, W)
    x = _inputs(sample_id="ok", gt_pix=render, gt_hi=torch.full((H, W), 0.01))
    got = head.target_of(x, (H, W))
    assert float(got.abs().sum()) == 0.0
    assert head.rt.audit_pass == 1 and head.rt.audit_fail == 0
    assert head.facts()["gt_pix_source"] == {"render": 1}


def test_missing_pixel_gt_is_refused_not_faked():
    head = _head()
    with pytest.raises(ValueError, match="no `gt_pix`"):
        head.target_of(_inputs(gt_pix=None), (H, W))


def test_target_is_projected_with_the_gt_low_operator_at_half_res():
    head = _head(res=0.5)
    x = _inputs()
    gt = head.target_of(x, (H // 2, W // 2))
    assert tuple(gt.shape) == (1, 1, H // 2, W // 2)


# --------------------------------------------------------------------------- #
# the image source (the shared-infrastructure gap)
# --------------------------------------------------------------------------- #
def test_luma_path_normalises_with_the_luma_collapsed_constants():
    head = _head()
    x = _inputs(guide_hi=torch.full((1, 1, H, W), 0.5))
    img = head.image_of(x)
    assert tuple(img.shape) == (1, 1, H, W)
    assert float(img[0, 0, 0, 0]) == pytest.approx((0.5 - M.LUMA_MEAN) / M.LUMA_STD)
    assert head.rt.img_source == {"guide_hi": 1}


def test_rgb_path_uses_vitmattes_pixel_mean_and_std(monkeypatch):
    monkeypatch.setitem(M.SEAM, "pixgt_img", True)
    head = M.MatteHead(in_dim=8, text_dim=6, img_source="rgb")
    assert head.img_chans == 3

    class _PG:
        img = torch.full((3, H, W), 0.5)

    x = _inputs(pixgt=_PG())
    img = head.image_of(x)
    assert tuple(img.shape) == (1, 3, H, W)
    for c in range(3):
        assert float(img[0, c, 0, 0]) == pytest.approx(
            (0.5 - M.PIXEL_MEAN[c]) / M.PIXEL_STD[c])
    assert M.PIXEL_MEAN == (123.675 / 255., 116.280 / 255., 103.530 / 255.)
    assert M.PIXEL_STD == (58.395 / 255., 57.120 / 255., 57.375 / 255.)


def test_the_rgb_path_runs_end_to_end_through_the_installed_seam(monkeypatch):
    """The proposal-faithful 3-channel detail branch, on the real seam class."""
    class _Base:
        def get(self, sample, **kw):
            return type("PG", (), {})()

    class _Sample:
        sample_id = "s1"

        def image_tensor(self):
            return torch.rand(3, H, W)

    monkeypatch.setitem(M.SEAM, "pixgt_img", True)
    prov = M.make_image_provider(_Base)()
    pg = prov.get(_Sample())
    head = M.MatteHead(in_dim=8, text_dim=6, img_source="rgb")
    assert head.img_chans == 3 and head.detail.conv_chans == [3, 48, 96, 192]
    img = head.image_of(_inputs(pixgt=pg))
    alpha = head(torch.randn(1, 8, GH, GW), img, torch.randn(6))
    assert tuple(alpha.shape) == (1, 1, H, W)
    assert head.facts()["img_source_counts"] == {"pixgt_img": 1}


def test_rgb_is_refused_loudly_when_neither_source_exists(monkeypatch):
    monkeypatch.setitem(M.SEAM, "pixgt_img", False)
    monkeypatch.setattr(M, "_rgb_available", lambda: True)
    head = M.MatteHead(in_dim=8, text_dim=6, img_source="rgb")
    with pytest.raises(ValueError, match="needs the spec-5 RGB"):
        head.image_of(_inputs(pixgt=None))


def test_an_image_of_the_wrong_resolution_is_an_assertion_not_a_resize():
    head = _head()
    with pytest.raises(AssertionError, match="would not line up"):
        head.image_of(_inputs(guide_hi=torch.rand(1, 1, H + 16, W)))


def test_the_image_seam_attaches_the_sample_image():
    class _Base:
        def get(self, sample, **kw):
            return type("PG", (), {})()

    class _Sample:
        def image_tensor(self):
            return torch.zeros(3, H, W)

    prov = M.make_image_provider(_Base)()
    pg = prov.get(_Sample())
    assert tuple(pg.img.shape) == (3, H, W)


# --------------------------------------------------------------------------- #
# the detail branch's device/dtype (EPR022_MATTE_SMOKE, 2026-08-15 03:10)
# --------------------------------------------------------------------------- #
# The smoke run died in `ConvStream` with
#   RuntimeError: Input type (torch.FloatTensor) and weight type
#   (CUDABFloat16Type) should be the same
# `pixgt.img` is `WhereBSample.image_tensor()` -- CPU float32 -- and
# `amort/data.py:769` moves only `pg.alpha` to the run device, storing the PixGT
# object itself as-is.  The image is therefore the ONE input of this arm that
# arrives on the CPU, and `image_of` used to hand it to the ConvStream unmoved.
# CPU-only box: `meta` stands in for the second device, `bfloat16` for the
# weight dtype autocast produced.
class _CPUImagePG:
    """A PixGT carrying exactly what the seam attaches: CPU float32 (3,H,W)."""

    def __init__(self):
        self.img = torch.rand(3, H, W)


def test_the_detail_image_follows_the_head_onto_its_device(monkeypatch):
    monkeypatch.setitem(M.SEAM, "pixgt_img", True)
    head = M.MatteHead(in_dim=8, text_dim=6, img_source="rgb").to("meta")
    x = _inputs(pixgt=_CPUImagePG())
    assert x.pixgt.img.device == torch.device("cpu")
    img = head.image_of(x)
    assert img.device == head.io_ref.device == torch.device("meta")
    assert img.dtype == head.io_ref.dtype


def test_the_imagenet_constants_migrate_with_the_head_and_are_not_state(monkeypatch):
    monkeypatch.setitem(M.SEAM, "pixgt_img", True)
    head = M.MatteHead(in_dim=8, text_dim=6, img_source="rgb")
    bufs = dict(head.named_buffers())
    assert {"pixel_mean", "pixel_std"} <= set(bufs)
    assert torch.allclose(bufs["pixel_mean"].reshape(3), torch.tensor(M.PIXEL_MEAN))
    assert torch.allclose(bufs["pixel_std"].reshape(3), torch.tensor(M.PIXEL_STD))
    # literals, not learned state: they must not enter/leave a checkpoint
    assert not [k for k in head.state_dict() if k.startswith("pixel_")]
    # a buffer moves with `.to()`; a `torch.tensor(...)` built in forward would not
    moved = M.MatteHead(in_dim=8, text_dim=6, img_source="rgb").to("meta")
    assert moved.pixel_mean.device == torch.device("meta")


@pytest.mark.parametrize("cond", ["film", "concat"])
def test_a_cpu_float32_image_against_bf16_weights_does_not_raise(monkeypatch, cond):
    """The crash reproduced in the dtypes CPU can hold: bf16 weights, fp32 image."""
    monkeypatch.setitem(M.SEAM, "pixgt_img", True)
    head = M.MatteHead(in_dim=8, text_dim=6, img_source="rgb",
                       cond=cond).to(torch.bfloat16)
    feat = torch.randn(1, 8, GH, GW, dtype=torch.bfloat16)
    h_cond = torch.randn(6, dtype=torch.bfloat16)

    # (i) the image the arm builds itself
    img = head.image_of(_inputs(pixgt=_CPUImagePG()))
    assert img.dtype == torch.bfloat16
    alpha = head(feat, img, h_cond)
    assert tuple(alpha.shape) == (1, 1, H, W) and alpha.dtype == torch.bfloat16
    assert torch.isfinite(alpha).all()

    # (ii) a raw float32 image handed straight to forward (a caller-built one)
    alpha2 = head(feat, torch.rand(1, 3, H, W), h_cond)
    assert tuple(alpha2.shape) == (1, 1, H, W) and torch.isfinite(alpha2).all()


def test_fp32_stays_fp32_when_the_head_is_fp32(monkeypatch):
    """`--matte-precision fp32`'s no_autocast path is not demoted by the fix."""
    monkeypatch.setitem(M.SEAM, "pixgt_img", True)
    head = M.MatteHead(in_dim=8, text_dim=6, img_source="rgb")
    img = head.image_of(_inputs(pixgt=_CPUImagePG()))
    assert img.dtype == torch.float32
    alpha = head(torch.randn(1, 8, GH, GW), img, torch.randn(6))
    assert alpha.dtype == torch.float32


def test_the_luma_path_also_follows_the_head_onto_its_device():
    head = _head().to("meta")
    img = head.image_of(_inputs(guide_hi=torch.rand(1, 1, H, W)))
    assert img.device == torch.device("meta") == head.io_ref.device


# --------------------------------------------------------------------------- #
# RNG independence
# --------------------------------------------------------------------------- #
def test_the_random_control_field_does_not_touch_the_global_rng():
    torch.manual_seed(0)
    a = torch.rand(4)
    torch.manual_seed(0)
    M._rand_field("s1", (16, 16), 20260814, torch.device("cpu"))
    M._rand_field("s2", (16, 16), 20260814, torch.device("cpu"))
    b = torch.rand(4)
    assert torch.equal(a, b)
    # ...and it is deterministic per sample id
    r1 = M._rand_field("s1", (8, 8), 7, torch.device("cpu"))
    r2 = M._rand_field("s1", (8, 8), 7, torch.device("cpu"))
    r3 = M._rand_field("s2", (8, 8), 7, torch.device("cpu"))
    assert torch.equal(r1, r2) and not torch.equal(r1, r3)


def test_building_the_matte_arm_leaves_the_shared_modules_init_stream_alone():
    """`CondEncoder` is constructed before the head; its draw must not move."""
    from q3vl.whereb.amort.model import AmortModel

    torch.manual_seed(1234)
    ref = AmortModel("P3prime", in_dim=8, ch=16, n_blocks=1, cond_text_dim=6,
                     cond_out=8, with_semantic=False, use_sim_field=False)
    w_ref = ref.cond.proj[1].weight.detach().clone()
    torch.manual_seed(1234)
    got = AmortModel("MATTE", in_dim=8, cond_text_dim=6, cond_out=8,
                     arm_kwargs=dict(img_source="luma"))
    assert torch.equal(w_ref, got.cond.proj[1].weight.detach())
    # B-4: SemanticHead not built, CondEncoder frozen out of the optimizer
    assert got.sem is None and got.cond_frozen
    assert got.facts()["cond_encoder_unused"] is True


# --------------------------------------------------------------------------- #
# hooks: optimizer / scheduler / builder
# --------------------------------------------------------------------------- #
def test_optimizer_spec_is_vitmattes_adamw():
    args = argparse.Namespace(lr=5e-4, weight_decay=0.1)
    spec = M.optimizer_spec(args)
    assert (spec.type, spec.lr, spec.weight_decay) == ("adamw", 5e-4, 0.1)
    assert spec.betas == (0.9, 0.999) and spec.grouping == "dim"
    assert M.VITMATTE_LR == 5e-4 and M.VITMATTE_WD == 0.1


def test_scheduler_kwargs_carry_the_milestones_over_as_fractions():
    args = argparse.Namespace(scheduler="multistep")
    kw = M.scheduler_kwargs(args, 1200)
    assert kw["values"] == [1.0, 0.1, 0.05]
    assert kw["milestones"] == [360, 1080]          # 30% / 90% of 1200
    assert kw["warmup_steps"] == 2                  # round(1200 * 250/134687)
    assert kw["warmup_factor"] == 0.001
    assert M.VITMATTE_MAX_ITER == 134687
    assert M.VITMATTE_MILESTONES == (40406, 121218)
    # a cosine row must not inherit the warmup_factor
    assert M.scheduler_kwargs(argparse.Namespace(scheduler="cosine"), 1200) == {}
    with pytest.raises(SystemExit):
        M.scheduler_kwargs(args, 0)


def test_the_multistep_ladder_is_what_make_scheduler_produces():
    from q3vl.where.calibrate import make_scheduler

    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1.0)
    sch = make_scheduler(opt, 1200, 0.03, "multistep",
                         **M.scheduler_kwargs(argparse.Namespace(
                             scheduler="multistep"), 1200))
    seen = {}
    for step in range(1200):
        seen[step] = opt.param_groups[0]["lr"]
        sch.step()
    assert seen[0] == pytest.approx(0.001)          # warmup_factor at step 0
    assert seen[100] == pytest.approx(1.0)
    assert seen[359] == pytest.approx(1.0)
    assert seen[360] == pytest.approx(0.1)
    assert seen[1079] == pytest.approx(0.1)
    assert seen[1080] == pytest.approx(0.05)


def test_builder_kwargs_ask_for_spec5_pixel_gt_and_the_hi_rasters():
    kw = M.builder_kwargs(argparse.Namespace())
    assert kw["want_hi"] is True
    assert kw["pixgt_size"](GH, GW) == (H, W)


def test_every_flag_of_the_proposals_entry_row_exists():
    ap = argparse.ArgumentParser()
    M.add_arguments(ap)
    ns, _ = ap.parse_known_args([])
    for name in ("matte_cond", "matte_res", "matte_gt_source", "matte_losses",
                 "matte_norm", "matte_img_source", "matte_precision",
                 "matte_gt_audit", "matte_gt_audit_tol", "matte_control_seed",
                 "matte_grad_sigma", "matte_no_pix_diag"):
        assert hasattr(ns, name), name
    assert (ns.matte_cond, ns.matte_res, ns.matte_gt_source, ns.matte_norm,
            ns.matte_losses) == ("film", "1.0", "render", "gn8", M.DEFAULT_LOSSES)
    kw = M.head_kwargs_from_args(ns)
    assert kw["losses"] == M.LOSS_NAMES and kw["res"] == 1.0


def test_head_kwargs_tolerate_the_bare_entry_scripts_namespace():
    """`--arm MATTE` without the wrapper still builds the faithful default."""
    kw = M.head_kwargs_from_args(argparse.Namespace())
    assert kw["cond"] == "film" and kw["losses"] == M.LOSS_NAMES


# --------------------------------------------------------------------------- #
# MAM diagnostic columns
# --------------------------------------------------------------------------- #
def test_mam_gauss_gradient_kernel_matches_the_reference_construction():
    hx, hy, size = M.gauss_grad_kernels(1.4)
    assert size == 9 and hx.shape == (9, 9)
    assert hy == pytest.approx(hx.T)
    assert float((hx ** 2).sum()) == pytest.approx(1.0, rel=1e-5)
    # a derivative filter has zero DC response
    assert abs(float(hx.sum())) < 1e-5


def test_sad_and_grad_are_zero_on_an_exact_match():
    x = torch.rand(24, 32)
    assert M.pix_sad(x, x) == pytest.approx(0.0)
    assert M.pix_gradient_error(x, M._grad_amp(x)) == pytest.approx(0.0, abs=1e-9)
    assert M.pix_sad(torch.ones(10, 10), torch.zeros(10, 10)) == pytest.approx(0.1)


# --------------------------------------------------------------------------- #
# the pre-registered board column
# --------------------------------------------------------------------------- #
def _complete_row(**kw):
    row = {"family": "radial", "pix_gt_area_frac": 0.2, "pix_gt_source": "render"}
    for m in M._PIX_METRICS:
        row[m] = 0.5
        for c in M._PIX_CONTROLS:
            row[f"{m}_{c}"] = 0.25
    row.update(kw)
    return row


def test_criteria_columns_publishes_pix_readout_with_both_controls():
    col = M.criteria_columns([_complete_row(), _complete_row(family="band")])
    pr = col["pix_readout"]
    assert pr["n"] == 2
    for m in M._PIX_METRICS:
        assert pr[m]["median"] == pytest.approx(0.5)
        assert pr[f"{m}_center"]["median"] == pytest.approx(0.25)
        assert pr[f"{m}_random"]["median"] == pytest.approx(0.25)
    assert set(pr["by_family"]) == {"radial", "band"}
    assert pr["gt_pix_source"] == {"render": 2}
    assert "DIAGNOSTIC ONLY" in pr["note"]


def test_the_board_carries_the_run_level_guard_c(monkeypatch):
    """Guard (c) -- gt_pix provenance + the D-9 audit -- must reach metrics.json."""
    monkeypatch.setattr(M, "_LIVE_HEAD", [])
    col = M.criteria_columns([_complete_row()])["pix_readout"]
    assert col["run_guard"]["available"] is False       # no head built here
    head = _head(audit_n=4, audit_tol=0.02)
    head.target_of(_inputs(sample_id="ok", gt_pix=torch.zeros(H, W),
                           gt_hi=torch.zeros(H, W)), (H, W))
    head.target_of(_inputs(sample_id="bad", gt_pix=torch.zeros(H, W),
                           gt_hi=torch.ones(H, W)), (H, W))
    monkeypatch.setattr(M, "_LIVE_HEAD", [head])
    g = M.criteria_columns([_complete_row()])["pix_readout"]["run_guard"]
    assert g["available"] is True
    assert g["gt_pix_source_counts"] == {"render": 2}
    assert g["gt_pix_audit"]["n_pass"] == 1 and g["gt_pix_audit"]["n_fail"] == 1
    assert g["gt_pix_audit"]["n_fallback_ids"] == 1
    assert g["n_samples"] == 2 and g["n_fake"] == 0


def test_the_audit_does_not_spend_its_budget_on_foreign_instruction_draws():
    head = _head(audit_n=4)
    head.target_of(_inputs(sample_id="f", is_fake=True), (H, W))
    assert head.rt.audit_n == 0 and head.rt.n_fake == 1


def test_the_wrapper_refuses_no_pixgt():
    from q3vl.whereb.scripts import run_matte_arm as R

    with pytest.raises(SystemExit, match="incompatible with --arm MATTE"):
        R.main(["--no-pixgt", "--matte-allow-missing-base"])


def test_a_row_without_its_controls_does_not_count():
    bad = _complete_row()
    del bad["pix_soft_iou_random"]
    col = M.criteria_columns([bad])["pix_readout"]
    assert col["n"] == 0 and col["n_rows_incomplete"] == 1


def test_an_empty_pix_readout_column_refuses_to_publish():
    from q3vl.whereb.amort.evaluate import assert_criteria_ran

    board = {"criteria_columns": M.criteria_columns([])}
    with pytest.raises(AssertionError, match="pix_readout"):
        assert_criteria_ran(board, "MATTE")
    board_ok = {"criteria_columns": M.criteria_columns([_complete_row()])}
    rep = assert_criteria_ran(board_ok, "MATTE")
    assert rep["required"] == ["pix_readout"] and rep["computed"]["pix_readout"] == 1


def test_head_facts_do_not_claim_deep_supervision_branches():
    """`deep_supervision_tags` must find nothing, or the board demands columns."""
    from q3vl.whereb.amort.evaluate import deep_supervision_tags

    assert deep_supervision_tags(_head().facts()) == []


# --------------------------------------------------------------------------- #
# the whole seam, end to end on CPU
# --------------------------------------------------------------------------- #
def _model():
    from q3vl.whereb.amort.model import AmortModel

    return AmortModel("MATTE", in_dim=8, cond_text_dim=6, cond_out=8,
                      arm_kwargs=dict(img_source="luma"))


def test_forward_geo_returns_m_low_on_the_fpre_grid():
    model = _model()
    x = _inputs()
    out = model.forward_geo(x.feat, model.cond_of(x.cond_h, x.cond_mask,
                                                  x.word_ids, x.word_offsets),
                            x.phi_dir, grid_h=GH, grid_w=GW,
                            h_cond=x.h_cond, sample=x)
    assert tuple(out["m_low"].shape) == (GH, GW)
    assert tuple(out["alpha_pix"].shape) == (1, 1, H, W)
    m = out["m_low"].detach()
    assert 0.0 <= float(m.min()) and float(m.max()) <= 1.0


def test_forward_geo_survives_a_bf16_head_and_a_cpu_float32_image(monkeypatch):
    """The 2026-08-15 crash's own stack: forward_geo -> matte.forward ->
    image_of -> ConvStream, with the image arriving unmoved off the PixGT."""
    from q3vl.whereb.amort.model import AmortModel

    monkeypatch.setitem(M.SEAM, "pixgt_img", True)
    model = AmortModel("MATTE", in_dim=8, cond_text_dim=6, cond_out=8,
                       arm_kwargs=dict(img_source="rgb"))
    model.geo.to(torch.bfloat16)
    x = _inputs(feat=torch.randn(1, 8, GH, GW, dtype=torch.bfloat16),
                pixgt=_CPUImagePG())
    out = model.forward_geo(x.feat, model.cond_of(x.cond_h, x.cond_mask,
                                                  x.word_ids, x.word_offsets),
                            x.phi_dir, grid_h=GH, grid_w=GW,
                            h_cond=x.h_cond, sample=x)
    assert tuple(out["alpha_pix"].shape) == (1, 1, H, W)
    assert tuple(out["m_low"].shape) == (GH, GW)
    assert out["matte"]["img_source"] == "rgb"


def test_compute_micro_batch_emits_the_four_preregistered_loss_columns():
    from q3vl.whereb.amort.losses import LossWeights, aggregate
    from q3vl.whereb.amort.trainer import compute_micro_batch

    model = _model()
    xs = [_inputs(sample_id="a"), _inputs(sample_id="b", is_fake=True)]
    total, stats, rows = compute_micro_batch(model, xs, LossWeights())
    assert torch.isfinite(total) and total.requires_grad
    for name in ("L_unknown_l1", "L_known_l1", "L_pha_laplacian",
                 "L_gradient_penalty"):
        assert name in stats, name
    # D-8: the fake subset's four terms are logged apart
    assert "L_unknown_l1_fake" in stats and stats["n_fake"] == 1
    # the shared early-warning columns are still produced by the trainer
    assert "area_ratio_median" in stats and "std_ratio_median" in stats
    assert rows[0]["gt_pix_source"] == "render"
    total.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.geo.parameters())
    assert aggregate  # imported for the contract it documents


def test_a_term_set_mismatch_is_an_assertion_at_the_first_micro_batch():
    model = _model()
    x = _inputs()
    out = model.forward_geo(x.feat, model.cond_of(x.cond_h, x.cond_mask,
                                                  x.word_ids, x.word_offsets),
                            x.phi_dir, grid_h=GH, grid_w=GW,
                            h_cond=x.h_cond, sample=x)
    model.geo.losses = ("unknown_l1_loss",)       # head says one, criterion four
    from q3vl.whereb.amort.losses import LossWeights

    with pytest.raises(AssertionError, match="pre-registered loss column"):
        M.compute_loss(model, out, x, LossWeights())


def test_loss_subsets_drop_exactly_the_terms_they_name():
    from q3vl.whereb.amort.losses import LossWeights
    from q3vl.whereb.amort.model import AmortModel

    model = AmortModel("MATTE", in_dim=8, cond_text_dim=6, cond_out=8,
                       arm_kwargs=dict(img_source="luma",
                                       losses=M.parse_losses("unknown_l1,known_l1")))
    x = _inputs()
    out = model.forward_geo(x.feat, model.cond_of(x.cond_h, x.cond_mask,
                                                  x.word_ids, x.word_offsets),
                            x.phi_dir, grid_h=GH, grid_w=GW,
                            h_cond=x.h_cond, sample=x)
    sl = M.compute_loss(model, out, x, LossWeights())
    assert set(sl.terms) == {"unknown_l1", "known_l1"}


def test_per_sample_row_emits_every_metric_next_to_both_controls():
    model = _model()
    x = _inputs()
    out = model.forward_geo(x.feat, model.cond_of(x.cond_h, x.cond_mask,
                                                  x.word_ids, x.word_offsets),
                            x.phi_dir, grid_h=GH, grid_w=GW,
                            h_cond=x.h_cond, sample=x)
    row = M.per_sample_row(model, out, x)
    for m in M._PIX_METRICS:
        assert isinstance(row[m], float)
        for c in M._PIX_CONTROLS:
            assert isinstance(row[f"{m}_{c}"], float)
    assert (row["pix_h"], row["pix_w"]) == (H, W)
    assert row["pix_gt_source"] == "render"
    assert M.train_stats(out, x)["matte_img_source"] == "luma"


def test_the_arm_refuses_a_forward_without_the_language_condition():
    model = _model()
    x = _inputs()
    with pytest.raises(ValueError, match="reads h_cond"):
        model.forward_geo(x.feat, model.cond_of(x.cond_h, x.cond_mask, x.word_ids,
                                                x.word_offsets),
                          x.phi_dir, grid_h=GH, grid_w=GW, h_cond=None, sample=x)


def test_the_arm_refuses_a_forward_without_the_sample():
    model = _model()
    x = _inputs()
    with pytest.raises(ValueError, match="needs the AmortSampleInputs"):
        model.forward_geo(x.feat, model.cond_of(x.cond_h, x.cond_mask, x.word_ids,
                                                x.word_offsets),
                          x.phi_dir, grid_h=GH, grid_w=GW, h_cond=x.h_cond,
                          sample=None)


# --------------------------------------------------------------------------- #
# the entry script
# --------------------------------------------------------------------------- #
def test_the_wrapper_injects_the_vitmatte_recipe_only_where_the_caller_is_silent():
    from q3vl.whereb.scripts import run_matte_arm as R

    rest = ["--run-name", "x"]
    R._default(rest, "--lr", "5e-4")
    R._default(rest, "--lr", "9e-9")
    assert rest.count("--lr") == 1 and "5e-4" in rest
    rest2 = ["--lr", "3e-4"]
    R._default(rest2, "--lr", "5e-4")
    assert rest2 == ["--lr", "3e-4"]


def test_the_wrapper_rechecks_the_first_step_row():
    from q3vl.whereb.scripts import run_matte_arm as R

    good = {"step": 1, "L_unknown_l1": 1.0, "L_known_l1": 1.0,
            "L_pha_laplacian": 1.0, "L_gradient_penalty": 1.0}
    assert R.check_first_step_row(good, M.LOSS_NAMES)["ok"] is True
    bad = dict(good)
    del bad["L_gradient_penalty"]
    rep = R.check_first_step_row(bad, M.LOSS_NAMES)
    assert rep["ok"] is False and rep["missing"] == ["L_gradient_penalty"]
    assert R.check_first_step_row(None, M.LOSS_NAMES)["ok"] is False


def test_facts_record_every_decision_the_proposal_asked_to_be_recorded():
    f = _head().facts()
    for key in ("in_chans", "img_chans", "fusion_out", "res", "norm", "cond",
                "losses", "norm_const", "grad_sparsity", "lap_max_levels",
                "zero_guards", "precision", "img_source", "gt_pix_source",
                "gt_pix_audit", "n_params", "control_seed"):
        assert key in f, key
    assert f["norm_const"] == 262144 and f["lap_max_levels"] == 5
    assert f["zero_guards"] == {"unknown_l1_loss": True,
                                "loss_gradient_penalty": True,
                                "known_l1_loss": True}
    assert f["loss_weights"] == {n: 1.0 for n in M.LOSS_NAMES}
    assert math.isclose(f["gt_pix_audit"]["tol"], 0.02)
