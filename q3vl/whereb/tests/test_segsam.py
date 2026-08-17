"""EPR-018 SEGSAM: the port is the reference recipe, and it is WIRED.

CPU only (``CUDA_VISIBLE_DEVICES=""``).  Three things are pinned:

* every transcribed constant equals the reference source's value (the numbers
  in ``PROPOSAL.md`` §3.1 / §3.2, re-opened 2026-08-14);
* the structural claims hold on real tensors -- parameter counts per module,
  the 4x decoder grid, ``m_low`` through the ``gt_low`` operator, the two
  losses against hand-computed values, the zero-target behaviour, and the fact
  that building this head does not move the global RNG stream;
* the seams the registry declares actually fire: ``compute_micro_batch``
  produces ``L_bce_lisa`` / ``L_dice_lisa``, the criterion column exists and an
  empty one refuses to publish, and the entry wrapper pins the shared flags the
  recipe implies.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pytest
import torch

from q3vl.whereb.amort import arms as A
from q3vl.whereb.amort import segsam as S

GH, GW = 4, 6


@pytest.fixture(autouse=True)
def clean_options():
    """``OPTIONS`` is module-level run state; no test may leak into another.

    ``FIRST_LOSS_COLUMNS`` is the same kind of state (the first micro-batch's
    loss columns, captured in-process) and is cleared with it -- a witness left
    behind by an earlier test would make the next one's assertion pass for the
    wrong reason.
    """
    keep = dict(S.OPTIONS)
    S.OPTIONS.clear()
    S.FIRST_LOSS_COLUMNS.clear()
    yield S.OPTIONS
    S.OPTIONS.clear()
    S.FIRST_LOSS_COLUMNS.clear()
    S.OPTIONS.update(keep)


def _head(**kw):
    kw.setdefault("scratch", True)
    kw.setdefault("seed", 7)
    return S.SegSamHead(in_dim=1024, text_dim=2560, **kw)


def _sample(**kw):
    from q3vl.whereb.amort.data import AmortSampleInputs

    base = dict(
        sample_id="s1", feat=torch.randn(1, 1024, GH, GW), sim=None, center=None,
        cond_h=torch.randn(1, 3, 2560), cond_mask=torch.ones(1, 3, dtype=torch.bool),
        word_ids=torch.zeros(1, dtype=torch.long),
        word_offsets=torch.zeros(1, dtype=torch.long),
        phi_dir=torch.zeros(GH * GW, 71), guide_hi=None,
        gt_low=torch.rand(GH, GW), gt_hi=None, gt_partner_low=None,
        grid_h=GH, grid_w=GW, is_fake=False, family="radial",
        route_semantic=False, h_cond=torch.randn(1, 2560),
        gt_pix=torch.rand(4 * GH, 4 * GW), gt_pix_source="render")
    base.update(kw)
    return AmortSampleInputs(**base)


# --------------------------------------------------------------------------- #
# registry contract
# --------------------------------------------------------------------------- #
def test_the_registry_loads_this_arm_with_its_preregistered_column():
    mod = A.load_arm("SEGSAM")
    assert mod is S
    assert S.ARM == "SEGSAM" and S.CRITERIA == A.ARM_CRITERIA["SEGSAM"]
    for hook in A.REQUIRED_HOOKS:
        assert callable(getattr(S, hook))
    for hook in ("add_arguments", "head_kwargs_from_args", "optimizer_spec",
                 "scheduler_kwargs", "builder_kwargs", "per_sample_row",
                 "criteria_columns", "train_stats"):
        assert callable(A.arm_hook("SEGSAM", hook)), hook


def test_the_live_four_are_untouched():
    from q3vl.whereb.amort.model import ARMS

    assert ARMS == ("P1", "P3prime", "SHAPE3", "UNIQ")
    assert not A.is_new_arm("P3prime")


# --------------------------------------------------------------------------- #
# transcribed constants (PROPOSAL 3.1 / 3.2)
# --------------------------------------------------------------------------- #
def test_loss_and_optimiser_constants_are_the_reference_values():
    assert S.LISA_BCE_WEIGHT == 2.0            # train_ds.py:80
    assert S.LISA_DICE_WEIGHT == 0.5           # train_ds.py:79
    assert S.LISA_DICE_SCALE == 1000           # LISA.py:20
    assert S.LISA_DICE_EPS == 1e-6             # LISA.py:21
    assert S.LISA_NUM_MASKS_EPS == 1e-8        # LISA.py:38, :58
    assert S.LISA_LR == 3e-4                   # train_ds.py:77
    assert S.LISA_WEIGHT_DECAY == 0.0          # train_ds.py:275
    assert S.LISA_BETAS == (0.9, 0.95)         # train_ds.py:85-86
    assert S.LISA_GRAD_CLIP == 1.0             # train_ds.py:295
    assert S.LISA_OUT_DIM == 256               # train_ds.py:92
    assert S.LISA_WARMUP_FRAC == 100.0 / 5000.0    # train_ds.py:65-66, :285


def test_sam_structural_constants_are_build_sam_vit_h():
    assert S.SAM_PROMPT_EMBED_DIM == 256
    assert S.SAM_IMAGE_EMBEDDING_SIZE == (64, 64)
    assert S.SAM_MASK_IN_CHANS == 16
    assert (S.SAM_TRANSFORMER_DEPTH, S.SAM_TRANSFORMER_MLP_DIM,
            S.SAM_TRANSFORMER_HEADS) == (2, 2048, 8)
    assert S.SAM_NUM_MULTIMASK_OUTPUTS == 3
    assert (S.SAM_IOU_HEAD_DEPTH, S.SAM_IOU_HEAD_HIDDEN_DIM) == (3, 256)
    assert S.FINE_UPSCALE == 4                  # two ConvTranspose2d(stride 2)


def test_defaults_are_the_proposals_written_values():
    o = S.options()
    assert o["gt"] == "raster" and o["sup"] == "native"
    assert o["dice_weight"] == 0.5 and o["bce_weight"] == 2.0
    assert o["scratch"] is False and o["frozen_decoder"] is False


# --------------------------------------------------------------------------- #
# structure
# --------------------------------------------------------------------------- #
def test_parameter_counts_per_module():
    """The ported modules, counted -- not the proposal's rough estimate.

    ``text_hidden_fcs`` = 2560*2560+2560 + 2560*256+256; ``neck`` = 1024*256 +
    2*256 + 256*256*9 + 2*256 (both convs bias-free); the two SAM subtrees are
    the checkpoint's own tensor counts (4,058,340 / 6,476 incl. the PE buffer).
    """
    p = _head().param_counts()
    assert p["text_hidden_fcs"] == 2560 * 2560 + 2560 + 2560 * 256 + 256 == 7_211_776
    assert p["neck"] == 1024 * 256 + 512 + 256 * 256 * 9 + 512 == 852_992
    assert p["mask_decoder"] == 4_058_340
    assert p["prompt_encoder"] + p["buffers"] == 6_476
    assert p["trainable"] == 7_211_776 + 852_992 + 4_058_340
    assert p["prompt_encoder_trainable"] == 0        # LISA.py:82-83


def test_forward_shapes_and_the_four_times_grid():
    h = _head()
    out = h(torch.randn(1, 1024, GH, GW), torch.randn(1, 2560), grid=(GH, GW))
    assert tuple(out["logits_fine"].shape) == (1, 4 * GH, 4 * GW)
    assert tuple(out["m_low"].shape) == (GH, GW)
    assert tuple(out["iou_pred"].shape) == (1, 1)    # multimask_output = False
    assert out["n_cond_vectors"] == 1


def test_k_greater_than_one_is_the_preregistered_cat_aggregation():
    """§4: each row through the SAME ``text_hidden_fcs``, cat as K sparse tokens
    of one prompt -- one mask out, not K."""
    h = _head()
    feat = torch.randn(1, 1024, GH, GW)
    out = h(feat, torch.randn(4, 2560), grid=(GH, GW))
    assert tuple(out["logits_fine"].shape) == (1, 4 * GH, 4 * GW)
    assert out["n_cond_vectors"] == 4


def test_m_low_equals_area_resize_of_the_sigmoid():
    from q3vl.whereb.amort.pixgt import area_project

    h = _head()
    out = h(torch.randn(1, 1024, GH, GW), torch.randn(1, 2560))
    want = area_project(torch.sigmoid(out["logits_fine"][0].float()), (GH, GW))
    assert torch.allclose(out["m_low"], want, atol=0, rtol=0)


def test_the_bf16_autocast_path_survives_and_keeps_the_field_in_fp32():
    """The trainer wraps the forward in bf16 autocast on CUDA (LISA's own
    ``--precision bf16``).  This is the CPU-level stand-in: it exercises the
    dtype seams -- the empty fp32 sparse buffer cat'd with bf16 text embeds, the
    per-sample PE cast, the dense prompt cast -- and pins that ``m_low`` and the
    loss stay fp32 (dice divides by 1000 and sums 24k terms; bf16 there is not
    the reference arithmetic)."""
    h = _head()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = h(torch.randn(1, 1024, GH, GW), torch.randn(1, 2560), grid=(GH, GW))
        total, _, _ = S.lisa_mask_loss(out["logits_fine"].float(),
                                       torch.rand(1, 4 * GH, 4 * GW))
    assert out["logits_fine"].dtype is torch.bfloat16
    assert out["m_low"].dtype is torch.float32
    assert total.dtype is torch.float32 and torch.isfinite(total)
    total.backward()
    assert h.neck[0].weight.grad is not None


def test_the_head_refuses_a_grid_that_is_not_the_features():
    h = _head()
    with pytest.raises(AssertionError, match="declared grid"):
        h(torch.randn(1, 1024, GH, GW), torch.randn(1, 2560), grid=(GH + 1, GW))


def test_building_the_head_does_not_move_the_global_rng_stream():
    """N1: ``PositionEmbeddingRandom`` draws a Gaussian buffer and the four
    submodules draw their initialisations.  A run with this arm must leave every
    other draw sequence bit-identical."""
    torch.manual_seed(0)
    before = torch.randn(5)
    torch.manual_seed(0)
    _head()
    after = torch.randn(5)
    assert torch.equal(before, after)


def test_the_same_seed_builds_the_same_head():
    a, b = _head(seed=123), _head(seed=123)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)
    c = _head(seed=124)
    assert not all(torch.equal(pa, pc) for pa, pc in zip(a.parameters(), c.parameters()))


def test_frozen_decoder_ablation_releases_only_neck_and_fcs():
    h = _head(frozen_decoder=True)
    p = h.param_counts()
    assert p["mask_decoder_trainable"] == 0
    assert p["trainable"] == p["text_hidden_fcs"] + p["neck"]
    assert h.facts()["trainable_modules"] == ["neck", "text_hidden_fcs"]


# --------------------------------------------------------------------------- #
# pretrained weights: two subtrees, never the image encoder
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not Path(S.SAM_VIT_H_DEFAULT).is_file(),
                    reason="SAM ViT-H checkpoint not on this box")
def test_only_the_two_subtrees_are_loaded_and_they_match_exactly():
    h = S.SegSamHead(scratch=False, seed=7)
    r = h.weights_report
    assert r["loaded"] is True and r["path"] == S.SAM_VIT_H_DEFAULT
    assert set(r["subtrees"]) == set(S.SUBTREES)
    assert r["subtrees"]["mask_decoder"]["n_params"] == 4_058_340
    assert r["subtrees"]["prompt_encoder"]["n_params"] == 6_476
    for name in S.SUBTREES:
        assert r["subtrees"][name]["missing_keys"] == []
        assert r["subtrees"][name]["unexpected_keys"] == []
    # the 632M image encoder is seen in the file and NOT loaded
    assert r["skipped_prefixes"]["image_encoder"] > 6e8
    assert len(r["subtree_sha256"]) == 64


@pytest.mark.skipif(not Path(S.SAM_VIT_H_DEFAULT).is_file(),
                    reason="SAM ViT-H checkpoint not on this box")
def test_pretrained_and_scratch_are_different_models():
    a = S.SegSamHead(scratch=False, seed=7)
    b = S.SegSamHead(scratch=True, seed=7)
    sa, sb = a.mask_decoder.state_dict(), b.mask_decoder.state_dict()
    assert any(not torch.allclose(sa[k], sb[k]) for k in sa)
    assert b.weights_report == {"loaded": False} and b.weights_path == ""


def test_a_missing_checkpoint_names_the_url_and_the_ablation_flag():
    with pytest.raises(FileNotFoundError, match="segment_anything|--segsam-scratch"):
        S.SegSamHead(scratch=False, weights="/nonexistent/sam_vit_h.pth")


# --------------------------------------------------------------------------- #
# the two losses (LISA.py:16-59), against hand-computed values
# --------------------------------------------------------------------------- #
def test_sigmoid_ce_matches_the_closed_form():
    logits = torch.tensor([[[0.0, 2.0], [-1.0, 0.5]]])
    target = torch.tensor([[[1.0, 0.0], [1.0, 0.25]]])
    got = float(S.sigmoid_ce_loss(logits, target, num_masks=1.0))

    def bce(z, t):
        return -(t * math.log(1 / (1 + math.exp(-z)))
                 + (1 - t) * math.log(1 - 1 / (1 + math.exp(-z))))

    want = (sum(bce(z, t) for z, t in [(0.0, 1.0), (2.0, 0.0), (-1.0, 1.0),
                                       (0.5, 0.25)]) / 4.0) / (1.0 + 1e-8)
    assert got == pytest.approx(want, rel=1e-6)


def test_dice_matches_the_closed_form_including_the_1000_scale():
    logits = torch.tensor([[[0.0, 2.0], [-1.0, 0.5]]])
    target = torch.tensor([[[1.0, 0.0], [1.0, 0.25]]])
    got = float(S.dice_loss(logits, target, num_masks=1.0))
    p = [1 / (1 + math.exp(-z)) for z in (0.0, 2.0, -1.0, 0.5)]
    t = [1.0, 0.0, 1.0, 0.25]
    num = 2 * sum(pi / 1000 * ti for pi, ti in zip(p, t))
    den = sum(pi / 1000 for pi in p) + sum(ti / 1000 for ti in t)
    want = (1 - (num + 1e-6) / (den + 1e-6)) / (1.0 + 1e-8)
    assert got == pytest.approx(want, rel=1e-6)


def test_dice_on_an_all_zero_target_is_finite_and_monotone():
    """§3.1 last row: the zero-target case has no NaN and no division by zero,
    and the loss rises with the predicted foreground area."""
    zeros = torch.zeros(1, 8, 8)
    empty = S.dice_loss(torch.full((1, 8, 8), -20.0), zeros, 1.0)
    full = S.dice_loss(torch.full((1, 8, 8), 20.0), zeros, 1.0)
    for v in (empty, full):
        assert torch.isfinite(v)
    assert float(empty) == pytest.approx(0.0, abs=1e-3)
    assert float(full) == pytest.approx(1.0, abs=1e-3)
    assert float(empty) < float(full)


def test_sigmoid_ce_on_an_all_zero_target_pushes_the_logits_negative():
    z = torch.zeros(1, 4, 4, requires_grad=True)
    S.sigmoid_ce_loss(z, torch.zeros(1, 4, 4), 1.0).backward()
    assert bool((z.grad > 0).all())          # gradient descent lowers the logit


def test_the_total_is_two_bce_plus_half_dice():
    logits = torch.randn(1, 6, 6)
    target = torch.rand(1, 6, 6)
    total, bce, dice = S.lisa_mask_loss(logits, target)
    assert float(total) == pytest.approx(2.0 * float(bce) + 0.5 * float(dice), rel=1e-6)
    only_bce, _, _ = S.lisa_mask_loss(logits, target, dice_weight=0.0)
    assert float(only_bce) == pytest.approx(2.0 * float(bce), rel=1e-6)


def test_the_loss_refuses_mismatched_shapes():
    with pytest.raises(AssertionError, match="same shape"):
        S.lisa_mask_loss(torch.randn(1, 4, 4), torch.rand(1, 8, 8))


# --------------------------------------------------------------------------- #
# compute_loss: targets, fake samples, zero guard
# --------------------------------------------------------------------------- #
def _out_for(head, x):
    return {"segsam": head(x.feat, x.h_cond, grid=(x.grid_h, x.grid_w)),
            "m_low": None}


class _Model:
    def __init__(self, head):
        self.geo = head
        self.arm = "SEGSAM"
        self.is_new_arm = True
        self.training = True


def test_compute_loss_emits_the_two_preregistered_terms():
    h = _head()
    x = _sample()
    sl = S.compute_loss(_Model(h), _out_for(h, x), x, None)
    assert set(sl.terms) == {"bce_lisa", "dice_lisa"}
    assert torch.isfinite(sl.total)
    assert h.counts["n_samples"] == 1 and h.counts["gt_source_render"] == 1


def test_a_fake_sample_gets_a_zeroed_target_and_its_own_columns():
    """D-6 default: the formula is unchanged, the target is zeroed, the numbers
    are counted separately."""
    h = _head()
    x = _sample(is_fake=True, gt_pix=torch.rand(4 * GH, 4 * GW))
    sl = S.compute_loss(_Model(h), _out_for(h, x), x, None)
    assert set(sl.terms) == {"bce_lisa", "dice_lisa", "bce_lisa_fake",
                             "dice_lisa_fake"}
    assert float(sl.terms["bce_lisa"].detach()) == float(
        sl.terms["bce_lisa_fake"].detach())
    assert h.counts["n_fake"] == 1
    # the same sample with its real GT is a different number
    y = _sample(gt_pix=x.gt_pix)
    real = S.compute_loss(_Model(h), _out_for(h, y), y, None)
    assert float(real.terms["dice_lisa"].detach()) != float(
        sl.terms["dice_lisa"].detach())


def test_an_all_zero_gt_is_excluded_counted_and_not_silent(capsys):
    h = _head()
    x = _sample(gt_pix=torch.zeros(4 * GH, 4 * GW))
    sl = S.compute_loss(_Model(h), _out_for(h, x), x, None)
    assert float(sl.total.detach()) == 0.0
    assert float(sl.terms["bce_lisa"].detach()) == 0.0
    assert float(sl.stats["segsam_excluded"]) == 1.0
    assert h.counts["n_gt_allzero_excluded"] == 1
    assert "all-zero GT excluded" in capsys.readouterr().out


def test_compute_loss_without_pixel_gt_names_the_flag():
    h = _head()
    x = _sample(gt_pix=None)
    with pytest.raises(AssertionError, match="no-pixgt"):
        S.compute_loss(_Model(h), _out_for(h, x), x, None)


def test_a_mismatched_gt_grid_is_projected_with_the_gt_low_operator():
    h = _head()
    x = _sample(gt_pix=torch.rand(8 * GH, 8 * GW))
    S.compute_loss(_Model(h), _out_for(h, x), x, None)
    assert h.counts["target_area_projected"] == 1


def test_sup_cgt_interpolates_the_logits_up_instead():
    h = _head(sup="cgt")
    x = _sample(gt_pix=torch.rand(S.CGT_UPSCALE * GH, S.CGT_UPSCALE * GW))
    sl = S.compute_loss(_Model(h), _out_for(h, x), x, None)
    assert torch.isfinite(sl.total)
    assert h.counts["target_cgt_interpolated"] == 1


def test_the_gradient_reaches_the_three_trainable_modules_only():
    h = _head()
    x = _sample()
    sl = S.compute_loss(_Model(h), _out_for(h, x), x, None)
    sl.total.backward()
    assert h.text_hidden_fcs[0][0].weight.grad is not None
    assert h.neck[0].weight.grad is not None
    assert h.mask_decoder.mask_tokens.weight.grad is not None
    assert all(p.grad is None for p in h.prompt_encoder.parameters())


# --------------------------------------------------------------------------- #
# optimiser / schedule / builder hooks
# --------------------------------------------------------------------------- #
def test_optimizer_spec_is_lisas_adamw():
    spec = S.optimizer_spec(argparse.Namespace(lr=3e-4))
    assert spec.type == "adamw"
    assert spec.lr == 3e-4 and spec.weight_decay == 0.0
    assert spec.betas == (0.9, 0.95) and spec.grouping == "none"


def test_warmup_is_the_reference_fraction_at_the_matched_horizon():
    assert S.warmup_for(1200) == 24          # 1200 * 100/5000
    assert S.warmup_for(5000) == 100         # the reference horizon itself
    kw = S.scheduler_kwargs(argparse.Namespace(scheduler="linear", max_steps=1200), 1200)
    assert kw == {"warmup_steps": 24}


def test_a_non_linear_scheduler_is_refused():
    with pytest.raises(SystemExit, match="--scheduler linear"):
        S.scheduler_kwargs(argparse.Namespace(scheduler="cosine", max_steps=1200), 1200)


def test_the_schedule_is_linear_warmup_then_linear_decay_to_zero():
    from q3vl.where.calibrate import make_scheduler

    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=3e-4)
    sched = make_scheduler(opt, 1200, 0.03, "linear", warmup_steps=24)
    lrs = []
    for _ in range(1200):
        lrs.append(opt.param_groups[0]["lr"])
        sched.step()
    assert lrs[0] == pytest.approx(3e-4 / 24)
    assert lrs[23] == pytest.approx(3e-4)          # end of warmup
    assert lrs[24 + (1200 - 24) // 2] == pytest.approx(3e-4 * 0.5, rel=1e-2)
    assert lrs[-1] == pytest.approx(0.0, abs=3e-7)


def test_builder_kwargs_sizes_the_pixel_gt_grid():
    args = argparse.Namespace(pixgt_source="render", pixgt_fallback="cgt1024",
                              no_pixgt=False)
    kw = S.builder_kwargs(args)
    assert kw["pixgt_size"](32, 48) == (128, 192)      # 4x, decoder-native
    S.OPTIONS["sup"] = "cgt"
    kw = S.builder_kwargs(args)
    assert kw["pixgt_size"](32, 48) == (1024, 1536)    # the .cgt grid


def test_builder_kwargs_refuses_a_gt_source_that_contradicts_the_flag():
    S.OPTIONS["gt"] = "png"
    with pytest.raises(SystemExit, match="--segsam-gt"):
        S.builder_kwargs(argparse.Namespace(pixgt_source="render",
                                            pixgt_fallback="cgt1024",
                                            no_pixgt=False))


def test_builder_kwargs_refuses_no_pixgt():
    with pytest.raises(SystemExit, match="no-pixgt"):
        S.builder_kwargs(argparse.Namespace(pixgt_source="render",
                                            pixgt_fallback="cgt1024",
                                            no_pixgt=True))


# --------------------------------------------------------------------------- #
# the criterion column
# --------------------------------------------------------------------------- #
def test_per_sample_row_carries_the_whole_three_column_set():
    h = _head()
    x = _sample()
    row = S.per_sample_row(_Model(h), _out_for(h, x), x)
    for col in ("segsam_fine_soft_iou", "segsam_fine_topk_iou",
                "segsam_fine_boundary_f1", "segsam_fine_center_topk_iou",
                "segsam_fine_center_soft_iou", "segsam_fine_random_floor"):
        assert row[col] is not None, col
    assert (row["segsam_fine_h"], row["segsam_fine_w"]) == (4 * GH, 4 * GW)
    assert "segsam_fine_error" not in row


def test_per_sample_row_records_a_failure_instead_of_raising():
    h = _head()
    x = _sample(gt_pix=None)
    row = S.per_sample_row(_Model(h), _out_for(h, x), x)
    assert row["segsam_fine_topk_iou"] is None
    assert "no gt_pix" in row["segsam_fine_error"]


def test_criteria_columns_and_the_publication_refusal():
    from q3vl.whereb.amort.evaluate import assert_criteria_ran

    h = _head()
    rows = []
    for i in range(4):
        x = _sample(sample_id=f"s{i}")
        rows.append(S.per_sample_row(_Model(h), _out_for(h, x), x))
    cols = S.criteria_columns(rows)
    col = cols["segsam_fine"]
    assert col["n"] == 4 and col["median"] is not None
    for k in ("soft_iou", "boundary_f1_tol4", "center_prior_topk_iou",
              "random_floor", "delta_vs_center_prior"):
        assert col[k] is not None, k
    assert col["boundary_tol_cells"] == S.BOUNDARY_TOL_FINE
    assert col["gt_pix_sources"] == {"render": 4}
    board = {"criteria_columns": cols}
    assert_criteria_ran(board, "SEGSAM")               # passes

    empty = {"criteria_columns": S.criteria_columns([])}
    with pytest.raises(AssertionError, match="0 values"):
        assert_criteria_ran(empty, "SEGSAM")


def test_assert_publishable_needs_the_loss_columns_from_step_one():
    h = _head()
    rows = [S.per_sample_row(_Model(h), _out_for(h, _sample()), _sample())
            for _ in range(2)]
    board = {"criteria_columns": S.criteria_columns(rows),
             "corr_center_minus_corr_gt": {"delta": -0.1, "p": 0.001},
             "m3_disease_present": False}
    ok = S.assert_publishable(board, steps_row={"step": 1, "L_bce_lisa": 0.7,
                                                "L_dice_lisa": 0.9})
    assert ok["checks"]["loss_columns"]["L_bce_lisa"] == 0.7
    assert ok["checks"]["m3_guard"]["corr_center_minus_corr_gt"] == -0.1
    with pytest.raises(AssertionError, match="L_dice_lisa"):
        S.assert_publishable(board, steps_row={"step": 1, "L_bce_lisa": 0.7})


def test_assert_publishable_refuses_a_partial_criterion_column():
    board = {"criteria_columns": {"segsam_fine": {"n": 10, "median": 0.5}}}
    with pytest.raises(AssertionError, match="incomplete"):
        S.assert_publishable(board, steps_row={"L_bce_lisa": 0.1,
                                               "L_dice_lisa": 0.2})


# --------------------------------------------------------------------------- #
# regression, 2026-08-15: the loss ran and the assertion could not see it
#
# `EPR018_SEGSAM_SMOKE` died at quick eval with "the first row is missing
# ['L_bce_lisa', 'L_dice_lisa'] ... (columns present: [])" while row 1 of
# `steps.jsonl` carried both columns.  The row was never looked up:
# `evaluate.evaluate_arm` (`evaluate.py:660`) calls `assert_criteria_ran(board,
# arm)` with no `steps_row`, and only `run_amort_arm._finish_board` passes one.
# The fix looks the row up instead of reading "not supplied" as "not computed";
# these tests pin BOTH halves -- it must find a real row, and it must still
# fail when the columns genuinely are not there.
# --------------------------------------------------------------------------- #
def _publishable_board():
    h = _head()
    rows = [S.per_sample_row(_Model(h), _out_for(h, _sample()), _sample())
            for _ in range(2)]
    return {"criteria_columns": S.criteria_columns(rows),
            "corr_center_minus_corr_gt": {"delta": -0.1, "p": 0.001},
            "m3_disease_present": False}


def _write_steps(path: Path, *rows) -> Path:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_assert_publishable_reads_steps_jsonl_when_the_caller_passes_no_row(tmp_path):
    p = _write_steps(tmp_path / "steps.jsonl",
                     {"step": 1, "loss": 5.7, "L_bce_lisa": 2.63,
                      "L_dice_lisa": 0.92, "n": 4},
                     {"step": 2, "loss": 2.6, "L_bce_lisa": 1.19,
                      "L_dice_lisa": 0.47, "n": 4})
    rep = S.assert_publishable(_publishable_board(), steps_row=None,
                               steps_path=p)
    # the FIRST row, not the last, and the report says where it came from
    assert rep["checks"]["loss_columns"] == {"L_bce_lisa": 2.63,
                                             "L_dice_lisa": 0.92}
    assert "steps.jsonl" in rep["checks"]["loss_columns_source"]


def test_a_disk_row_without_the_columns_still_refuses_to_publish(tmp_path):
    p = _write_steps(tmp_path / "steps.jsonl",
                     {"step": 1, "loss": 5.7, "L_something_else": 1.0})
    with pytest.raises(AssertionError, match=r"columns present: \['L_something_else'\]"):
        S.assert_publishable(_publishable_board(), steps_row=None, steps_path=p)


def test_the_caller_s_row_wins_over_the_file(tmp_path):
    p = _write_steps(tmp_path / "steps.jsonl", {"step": 1, "L_bce_lisa": 2.63,
                                                "L_dice_lisa": 0.92})
    rep = S.assert_publishable(_publishable_board(),
                               steps_row={"step": 1, "L_bce_lisa": 0.7,
                                          "L_dice_lisa": 0.9}, steps_path=p)
    assert rep["checks"]["loss_columns"]["L_bce_lisa"] == 0.7
    assert rep["checks"]["loss_columns_source"] == "caller"


def test_the_in_process_witness_covers_the_unflushed_quick_eval(tmp_path):
    """`trainer.py:608` only flushes every `log_every` steps, so a quick eval
    on a non-flush step reads an empty file.  The loss columns are still known:
    `compute_loss` recorded them on the first micro-batch."""
    h = _head()
    x = _sample()
    S.compute_loss(_Model(h), _out_for(h, x), x, None)
    assert set(S.FIRST_LOSS_COLUMNS) == {"L_bce_lisa", "L_dice_lisa"}
    rep = S.assert_publishable(_publishable_board(), steps_row=None,
                               steps_path=tmp_path / "steps.jsonl")  # absent
    assert rep["checks"]["loss_columns_source"] == (
        "in-process first-micro-batch witness")
    assert rep["checks"]["loss_columns"]["L_bce_lisa"] == pytest.approx(
        S.FIRST_LOSS_COLUMNS["L_bce_lisa"])


def test_no_row_anywhere_is_a_failure_not_a_pass(tmp_path):
    assert not S.FIRST_LOSS_COLUMNS
    with pytest.raises(AssertionError, match="no first row is available"):
        S.assert_publishable(_publishable_board(), steps_row=None,
                             steps_path=tmp_path / "steps.jsonl")
    with pytest.raises(AssertionError, match="no first row is available"):
        S.assert_publishable(_publishable_board(), steps_row=None)
    # eval_only re-scores a checkpoint: no training steps, nothing to check
    rep = S.assert_publishable(_publishable_board(), steps_row=None,
                               eval_only=True)
    assert rep["checks"]["loss_columns"].startswith("skipped")


def test_the_witness_is_the_first_micro_batch_and_matches_the_stats_columns():
    """The witness and `steps.jsonl` are the same claim: `losses.aggregate`
    prefixes `L_` to the term names and the trainer splices them into the row."""
    from q3vl.whereb.amort.losses import LossWeights
    from q3vl.whereb.amort.trainer import compute_micro_batch

    S.OPTIONS["scratch"] = True
    from q3vl.whereb.amort.model import AmortModel

    model = AmortModel("SEGSAM", arm_args=argparse.Namespace(seed=7))
    _, stats, _ = compute_micro_batch(model, [_sample(), _sample(sample_id="s2")],
                                      LossWeights())
    assert {"L_bce_lisa", "L_dice_lisa"} <= set(stats)
    assert set(S.FIRST_LOSS_COLUMNS) == {"L_bce_lisa", "L_dice_lisa"}
    # a second micro-batch does not overwrite the FIRST row's record
    first = dict(S.FIRST_LOSS_COLUMNS)
    compute_micro_batch(model, [_sample(sample_id="s3")], LossWeights())
    assert S.FIRST_LOSS_COLUMNS == first


def test_the_head_declares_no_deep_supervision_branch():
    from q3vl.whereb.amort.evaluate import deep_supervision_tags

    assert deep_supervision_tags(_head().facts()) == []


# --------------------------------------------------------------------------- #
# end to end through the shared seams
# --------------------------------------------------------------------------- #
def test_the_model_builds_the_arm_and_the_micro_batch_logs_the_two_columns():
    from q3vl.whereb.amort.losses import LossWeights
    from q3vl.whereb.amort.model import AmortModel
    from q3vl.whereb.amort.trainer import compute_micro_batch

    S.OPTIONS["scratch"] = True
    model = AmortModel("SEGSAM", arm_args=argparse.Namespace(seed=7))
    assert model.is_new_arm and model.sem is None and model.cond_frozen
    assert isinstance(model.geo, S.SegSamHead)

    total, stats, rows = compute_micro_batch(model, [_sample(), _sample(sample_id="s2")],
                                             LossWeights())
    assert torch.isfinite(total)
    assert "L_bce_lisa" in stats and "L_dice_lisa" in stats
    assert stats["n"] == 2 and stats["n_fake"] == 0
    assert "area_ratio_median" in stats            # the shared early-warning cols
    assert rows[0]["gt_pix_source"] == "render"
    assert model.facts()["arm_head"]["params"]["trainable"] == 12_123_108


def test_the_head_refuses_extra_stem_channels():
    from q3vl.whereb.amort.arms import ArmContext

    h = _head()
    ctx = ArmContext(feat=torch.randn(1, 1024, GH, GW), grid_h=GH, grid_w=GW,
                     h_cond=torch.randn(1, 2560),
                     extra=torch.zeros(1, 1, GH, GW))
    with pytest.raises(AssertionError, match="no-sim-field"):
        S.forward(_Model(h), h, ctx)


def test_a_missing_h_cond_names_the_readout_builder():
    from q3vl.whereb.amort.arms import ArmContext

    h = _head()
    ctx = ArmContext(feat=torch.randn(1, 1024, GH, GW), grid_h=GH, grid_w=GW)
    with pytest.raises(ValueError, match="ReadoutBuilder"):
        S.forward(_Model(h), h, ctx)


# --------------------------------------------------------------------------- #
# the entry wrapper
# --------------------------------------------------------------------------- #
def test_the_wrapper_keeps_the_legacy_routing_row_available(monkeypatch):
    """`--newarm-legacy-routing` must not arrive with the semantic head off."""
    import q3vl.whereb.amort.evaluate as _ev
    import q3vl.whereb.scripts.run_amort_arm as base
    from q3vl.whereb.amort import arms as _arms
    from q3vl.whereb.scripts import run_segsam_arm as W

    seen: dict[str, list[str]] = {}
    monkeypatch.setattr(base, "main", lambda argv: seen.setdefault("argv", argv) and 0)
    monkeypatch.setitem(_arms.ARM_SETUP, "segsam", {})
    monkeypatch.setattr(_ev, "assert_criteria_ran", _ev.assert_criteria_ran)

    W.main(["--newarm-legacy-routing", "--max-steps", "1200"])
    argv = seen["argv"]
    assert "--newarm-legacy-routing" in argv
    assert "--no-semantic-head" not in argv
    # the two recipe pins (not B-4 routing) still stand
    assert "--no-sim-field" in argv and "--no-film" in argv
    rec = _arms.ARM_SETUP["segsam"]["pinned_shared_flags"]["--no-semantic-head"]
    assert rec["pinned"] is False and rec["reason"] == "--newarm-legacy-routing"


def test_the_wrapper_pins_the_shared_flags_the_recipe_implies(monkeypatch):
    import q3vl.whereb.scripts.run_amort_arm as base
    from q3vl.whereb.amort import arms as _arms
    from q3vl.whereb.scripts import run_segsam_arm as W

    import q3vl.whereb.amort.evaluate as _ev

    seen: dict[str, list[str]] = {}
    monkeypatch.setattr(base, "main", lambda argv: seen.setdefault("argv", argv) and 0)
    monkeypatch.setitem(_arms.ARM_SETUP, "segsam", {})
    # the wrapper swaps `assert_criteria_ran` in place; restore it so the swap
    # cannot leak into another test in this session
    monkeypatch.setattr(_ev, "assert_criteria_ran", _ev.assert_criteria_ran)

    W.main(["--max-steps", "1200"])
    argv = seen["argv"]
    for flag in ("--no-sim-field", "--no-film", "--no-semantic-head"):
        assert flag in argv
    assert argv[argv.index("--arm") + 1] == "SEGSAM"
    assert argv[argv.index("--scheduler") + 1] == "linear"
    assert argv[argv.index("--pixgt-source") + 1] == "render"
    assert argv[argv.index("--pixgt-fallback") + 1] == "cgt1024"
    rec = _arms.ARM_SETUP["segsam"]
    assert rec["options"]["gt"] == "raster"
    assert rec["scheduler"]["warmup_steps_at_1200"] == 24
    assert rec["loss"]["dice_as_target"] is True
    assert rec["pinned_shared_flags"]["--scheduler"]["pinned"] is True

    # the publication assertion is installed on the module attribute
    # `_finish_board` imports at call time, and it fires for this arm only
    board = {"criteria_columns": {"segsam_fine": {"n": 1, "median": 0.4}}}
    with pytest.raises(AssertionError, match="incomplete"):
        _ev.assert_criteria_ran(board, "SEGSAM",
                                steps_row={"L_bce_lisa": 0.1, "L_dice_lisa": 0.2})
    assert _ev.assert_criteria_ran({"criteria_columns": {}}, "P1")["required"] == []


def test_the_wrapper_translates_the_gt_flag_and_records_the_override(monkeypatch):
    import q3vl.whereb.scripts.run_amort_arm as base
    from q3vl.whereb.amort import arms as _arms
    from q3vl.whereb.scripts import run_segsam_arm as W

    import q3vl.whereb.amort.evaluate as _ev

    seen: dict[str, list[str]] = {}
    monkeypatch.setattr(base, "main", lambda argv: seen.setdefault("argv", argv) and 0)
    monkeypatch.setitem(_arms.ARM_SETUP, "segsam", {})
    # the wrapper swaps `assert_criteria_ran` in place; restore it so the swap
    # cannot leak into another test in this session
    monkeypatch.setattr(_ev, "assert_criteria_ran", _ev.assert_criteria_ran)

    W.main(["--segsam-gt", "png", "--segsam-dice-weight", "0.0",
            "--scheduler", "cosine"])
    argv = seen["argv"]
    assert argv[argv.index("--pixgt-source") + 1] == "cgt1024"
    assert S.options()["dice_weight"] == 0.0
    # the caller's own --scheduler wins, and the record says the pin did not fire
    pins = _arms.ARM_SETUP["segsam"]["pinned_shared_flags"]
    assert pins["--scheduler"]["pinned"] is False


def test_the_installed_assertion_works_at_the_evaluate_arm_call_site(monkeypatch,
                                                                    tmp_path):
    """The 2026-08-15 regression, end to end through the wrapper.

    `evaluate.evaluate_arm` calls `assert_criteria_ran(board, arm)` -- two
    positional arguments, no `steps_row` -- and writes `metrics.json` three
    lines later.  The installed assertion must find the run's `steps.jsonl`
    itself there, and must still refuse a run whose first row has no loss
    columns.
    """
    import q3vl.whereb.amort.evaluate as _ev
    import q3vl.whereb.scripts.run_amort_arm as base
    from q3vl.whereb.amort import arms as _arms
    from q3vl.whereb.scripts import run_segsam_arm as W

    monkeypatch.setattr(base, "main", lambda argv: 0)
    monkeypatch.setitem(_arms.ARM_SETUP, "segsam", {})
    monkeypatch.setattr(_ev, "assert_criteria_ran", _ev.assert_criteria_ran)

    run = "amort_SEGSAM_regression"
    W.main(["--out-root", str(tmp_path), "--run-name", run])
    steps = tmp_path / run / "steps.jsonl"

    board = _publishable_board()
    # (a) before any row exists and with no loss call in this process: refused
    with pytest.raises(AssertionError, match="no first row is available"):
        _ev.assert_criteria_ran(board, "SEGSAM")

    # (b) the failing case the smoke run actually was in: the row is on disk
    _write_steps(steps, {"step": 1, "loss": 5.73, "lr": 3e-4,
                         "L_bce_lisa": 2.635, "L_dice_lisa": 0.920, "n": 4},
                 {"step": 2, "loss": 2.63, "L_bce_lisa": 1.199,
                  "L_dice_lisa": 0.471, "n": 4})
    rep = _ev.assert_criteria_ran(board, "SEGSAM")          # no steps_row
    got = rep["segsam_publication"]["checks"]["loss_columns"]
    assert got == {"L_bce_lisa": 2.635, "L_dice_lisa": 0.920}

    # (c) the assertion is not a bypass: a first row without the columns fails
    _write_steps(steps, {"step": 1, "loss": 5.73, "n": 4})
    S.FIRST_LOSS_COLUMNS.clear()
    with pytest.raises(AssertionError, match=r"columns present: \[\]"):
        _ev.assert_criteria_ran(board, "SEGSAM")

    # (d) other arms are untouched by the swap
    assert _ev.assert_criteria_ran({"criteria_columns": {}}, "P1")["required"] == []
