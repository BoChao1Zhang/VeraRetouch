"""EPR-023 CondInst arm: the ported numbers, the seams, and the guards.

CPU only (``CUDA_VISIBLE_DEVICES=""`` is enough to run the whole file).  What is
pinned here is what the proposal writes down and what would otherwise be
invisible in a board:

* the dynamic head is **169** parameters with rel-coords and **153** without,
  and the controller emits ``169 + 2``;
* the ported functions (``aligned_bilinear`` / ``parse_dynamic_params`` /
  ``mask_heads_forward`` / ``dice_coefficient``) behave as the upstream ones;
* step 0 with a zero condition is exactly ``sigmoid(0) = 0.5`` everywhere;
* turning an ablation branch on does not move the faithful head's initialisation
  (RNG independence);
* the optimizer and schedule are SGD 0.01 / momentum 0.9 / wd 1e-4 with
  detectron2's norm-exempt grouping and milestones ``[800, 1067]`` / warmup 13;
* the pre-registered criterion column is produced and a board without it is
  refused; the first ``steps.jsonl`` row must carry ``L_dice``.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from q3vl.whereb.amort import condinst as C
from q3vl.whereb.amort.condinst import CondInstConfig

TEXT_DIM = C.TEXT_DIM


@pytest.fixture(autouse=True)
def _reset_config():
    C.set_config(None)
    yield
    C.set_config(None)


def _head(**kw) -> C.CondInstHead:
    cfg = CondInstConfig(**kw)
    C.set_config(cfg)
    return C.build_head(in_dim=16, text_dim=32, cfg=cfg)


def _sample(gh=3, gw=4, factor=4, *, fake=False, family="radial",
            geometry=None, mask_type=None, source="render", device="cpu"):
    pix = (factor * gh, factor * gw)
    gt_pix = torch.zeros(pix)
    gt_pix[:, : pix[1] // 2] = 1.0
    pg = SimpleNamespace(mask_type=mask_type, geometry=geometry, source=source,
                         alpha=gt_pix)
    return SimpleNamespace(
        sample_id="s1", is_fake=fake, family=family,
        gt_low=torch.zeros(gh, gw), gt_pix=gt_pix, gt_pix_source=source,
        pixgt=pg, grid_h=gh, grid_w=gw)


# --------------------------------------------------------------------------- #
# registry contract
# --------------------------------------------------------------------------- #
def test_the_registry_can_load_this_arm():
    from q3vl.whereb.amort import arms as A

    A._CACHE.pop("CONDINST", None)
    mod = A.load_arm("CONDINST")
    assert mod.ARM == "CONDINST"
    assert mod.CRITERIA == A.ARM_CRITERIA["CONDINST"] == ("condinst_pix_readout",)
    for hook in A.REQUIRED_HOOKS:
        assert callable(getattr(mod, hook))
    for hook in ("head_kwargs_from_args", "optimizer_spec", "scheduler_kwargs",
                 "builder_kwargs", "readout_spec", "per_sample_row",
                 "criteria_columns", "train_stats"):
        assert callable(A.arm_hook("CONDINST", hook)), hook


# --------------------------------------------------------------------------- #
# the 169 / 153 count (proposal §3 num_gen_params 复算)
# --------------------------------------------------------------------------- #
def test_dynamic_head_is_169_parameters_with_rel_coords():
    w, b, n = C.dynamic_param_nums(8, 8, 3, disable_rel_coords=False)
    assert w == [80, 64, 8] and b == [8, 8, 1]
    assert sum(w) == 152 and sum(b) == 17 and n == 169


def test_dynamic_head_is_153_parameters_without_rel_coords():
    w, b, n = C.dynamic_param_nums(8, 8, 3, disable_rel_coords=True)
    assert w == [64, 64, 8] and b == [8, 8, 1] and n == 153


def test_controller_emits_num_gen_params_plus_two_centre_columns():
    on = _head()
    off = _head(disable_rel_coords=True)
    assert on.num_gen_params == 169 and on.controller.proj.out_features == 171
    assert off.num_gen_params == 153 and off.controller.proj.out_features == 155
    assert on.facts()["controller_out"] == 171


def test_head_parameter_budget_matches_the_proposal_estimate():
    """~2.2M: 1024*128*9 + 4*128*128*9 + (128*8+8) + (2560*171+171) + BN."""
    head = C.build_head(in_dim=1024, text_dim=2560, cfg=CondInstConfig())
    n = head.n_params()["total"]
    assert 2.0e6 < n < 2.4e6, n
    assert head.n_params()["controller"] == 2560 * 171 + 171


# --------------------------------------------------------------------------- #
# the ported functions
# --------------------------------------------------------------------------- #
def test_parse_dynamic_params_splits_into_the_upstream_shapes():
    w_n, b_n, n = C.dynamic_param_nums(8, 8, 3)
    theta = torch.arange(float(n)).reshape(1, n)
    ws, bs = C.parse_dynamic_params(theta, 8, w_n, b_n)
    assert [tuple(x.shape) for x in ws] == [(8, 10, 1, 1), (8, 8, 1, 1), (1, 8, 1, 1)]
    assert [tuple(x.shape) for x in bs] == [(8,), (8,), (1,)]
    # the split is contiguous and in order: weights first, then biases
    assert float(ws[0].reshape(-1)[0]) == 0.0
    assert float(bs[-1][0]) == n - 1


def test_mask_heads_forward_is_relu_between_layers_and_none_at_the_end():
    x = torch.ones(1, 2, 1, 1)
    w = [torch.tensor([[[[-1.0]], [[0.0]]]]), torch.tensor([[[[3.0]]]])]
    b = [torch.zeros(1), torch.zeros(1)]
    # layer0 -> -1, relu -> 0, layer1 -> 0  (no relu after the last layer)
    assert float(C.mask_heads_forward(x, w, b, 1)) == 0.0
    w[0] = torch.tensor([[[[2.0]], [[0.0]]]])
    assert float(C.mask_heads_forward(x, w, b, 1)) == 6.0


@pytest.mark.parametrize("factor", [1, 2, 4])
def test_aligned_bilinear_shapes_and_constant_preservation(factor):
    x = torch.full((1, 1, 3, 5), 0.7)
    y = C.aligned_bilinear(x, factor)
    assert tuple(y.shape) == (1, 1, 3 * factor, 5 * factor)
    assert torch.allclose(y, torch.full_like(y, 0.7), atol=1e-6)


def test_normalized_locations_are_cell_centres_and_rel_coords_stay_in_pm_one():
    loc = C.normalized_locations(2, 4)
    assert tuple(loc.shape) == (2, 2, 4)
    assert torch.allclose(loc[0][0], torch.tensor([0.125, 0.375, 0.625, 0.875]))
    assert torch.allclose(loc[1][:, 0], torch.tensor([0.25, 0.75]))
    for c in (0.0, 0.5, 1.0):
        rel = torch.tensor([c, c]).reshape(2, 1, 1) - loc
        assert float(rel.min()) >= -1.0 and float(rel.max()) <= 1.0


def test_dice_matches_the_hand_computed_value():
    x = torch.tensor([[1.0, 0.0]])
    t = torch.tensor([[1.0, 1.0]])
    want = 1.0 - 2 * 1.0 / (1.0 + 2.0 + 1e-5)
    assert float(C.dice_coefficient(x, t)) == pytest.approx(want, abs=1e-6)


def test_dice_is_finite_on_an_all_zero_pair():
    """Zero protection: the 1e-5 in the denominator is the only thing between
    an empty prediction on an empty target and a 0/0."""
    z = torch.zeros(1, 16)
    v = float(C.dice_coefficient(z, z))
    assert math.isfinite(v) and v == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# forward
# --------------------------------------------------------------------------- #
def test_forward_shapes_and_the_low_grid_is_the_area_resize_of_the_pixel_grid():
    from q3vl.where.upsample import area_resize

    head = _head()
    feat = torch.randn(1, 16, 3, 5)
    out = head(feat, torch.randn(32))
    assert tuple(out["m_low"].shape) == (3, 5)
    assert tuple(out["m_pix"].shape) == (12, 20)
    assert torch.allclose(
        out["m_low"], area_resize(out["m_pix"][None, None], (3, 5))[0, 0])
    assert tuple(out["rel_coords"].shape) == (2, 3, 5)
    m = out["m_pix"].detach()
    assert float(m.min()) >= 0.0 and float(m.max()) <= 1.0


def test_step0_with_a_zero_condition_is_exactly_one_half_everywhere():
    """controller bias = 0 -> theta = 0 -> every dynamic kernel is 0 -> the
    logit is 0 -> sigmoid = 0.5.  This is the reproducible half of the
    proposal's step-0 statement."""
    head = _head()
    out = head(torch.randn(1, 16, 3, 5), torch.zeros(32))
    assert float(out["theta"].abs().max()) == 0.0
    assert torch.allclose(out["m_pix"], torch.full_like(out["m_pix"], 0.5))
    assert torch.allclose(out["m_low"], torch.full_like(out["m_low"], 0.5))
    assert torch.allclose(out["center"], torch.full_like(out["center"], 0.5))


def test_disable_rel_coords_removes_the_two_coordinate_channels():
    head = _head(disable_rel_coords=True)
    out = head(torch.randn(1, 16, 3, 5), torch.randn(32))
    assert out["rel_coords"] is None
    assert tuple(out["m_pix"].shape) == (12, 20)
    assert head.weight_nums[0] == 64


def test_k_greater_than_one_readout_rows_are_averaged_after_the_controller():
    head = _head()
    rows = torch.randn(4, 32)
    out_multi = head(torch.zeros(1, 16, 2, 2), rows)
    out_mean = head(torch.zeros(1, 16, 2, 2), rows.mean(0, keepdim=True))
    assert torch.allclose(out_multi["theta"], out_mean["theta"], atol=1e-6)


def test_query_ctrl_reads_the_span_rows_and_starts_as_the_mean_pool():
    head = _head(ctrl="query")
    rows = torch.randn(1, 5, 32)
    got = head.condition(None, rows, torch.ones(1, 5, dtype=torch.bool))
    assert tuple(got.shape) == (1, 32)
    # zero-initialised query -> uniform attention -> the span mean
    assert torch.allclose(got[0], rows[0].mean(0), atol=1e-6)


def test_query_ctrl_without_span_rows_names_the_missing_input():
    head = _head(ctrl="query")
    with pytest.raises(ValueError, match="h_where"):
        head.condition(None, None, None)


def test_gradients_reach_the_mask_branch_and_the_controller():
    head = _head()
    out = head(torch.randn(1, 16, 3, 5), torch.randn(32))
    C.dice_coefficient(out["m_pix"].reshape(1, -1),
                       torch.rand(1, 12 * 20)).mean().backward()
    assert head.mask_branch.refine[0].weight.grad.abs().sum() > 0
    assert head.controller.proj.weight.grad.abs().sum() > 0


# --------------------------------------------------------------------------- #
# initialisation / RNG independence
# --------------------------------------------------------------------------- #
def test_conv_blocks_have_no_bias_under_bn_and_the_output_conv_does():
    head = _head()
    assert head.mask_branch.refine[0].bias is None
    assert isinstance(head.mask_branch.refine[1], torch.nn.BatchNorm2d)
    out_conv = head.mask_branch.tower[-1]
    assert isinstance(out_conv, torch.nn.Conv2d) and out_conv.kernel_size == (1, 1)
    assert out_conv.bias is not None
    assert out_conv.out_channels == 8


def test_controller_init_is_normal_0p01_with_zero_bias():
    head = C.build_head(in_dim=64, text_dim=2560, cfg=CondInstConfig())
    w = head.controller.proj.weight.detach()
    assert float(head.controller.proj.bias.abs().max()) == 0.0
    assert 0.007 < float(w.std()) < 0.013


@pytest.mark.parametrize("kw", [{"geom_reg_weight": 0.05}, {"ctrl": "query"},
                                {"center_sup": True}])
def test_turning_an_ablation_branch_on_does_not_move_the_faithful_init(kw):
    base = _head()
    other = _head(**kw)
    for (na, pa), (nb, pb) in zip(base.mask_branch.named_parameters(),
                                  other.mask_branch.named_parameters()):
        assert na == nb and torch.equal(pa, pb), na
    assert torch.equal(base.controller.proj.weight, other.controller.proj.weight)


def test_build_head_does_not_consume_the_global_rng_stream():
    torch.manual_seed(7)
    a = torch.randn(3)
    torch.manual_seed(7)
    C.build_head(in_dim=16, text_dim=32, cfg=CondInstConfig())
    b = torch.randn(3)
    assert torch.equal(a, b)


def test_geom_branch_is_not_built_when_the_weight_is_zero():
    head = _head()
    assert head.geom is None and head.span_query is None
    out = head(torch.randn(1, 16, 2, 2), torch.randn(32))
    assert "geom" not in out
    assert head.n_params()["geom_reg"] == 0
    assert head.facts()["geom_reg"] is None


def test_geom_head_last_layer_is_zero_initialised():
    head = _head(geom_reg_weight=0.05)
    assert float(head.geom.fc2.weight.abs().max()) == 0.0
    assert float(head.geom.fc2.bias.abs().max()) == 0.0
    g = head.geom(torch.randn(32))
    assert set(g) == {"route", "circulargradient", "circulargradient_flipped",
                      "gradient", "gradient_flipped"}
    assert float(torch.cat(list(g.values())).abs().max()) == 0.0


# --------------------------------------------------------------------------- #
# loss
# --------------------------------------------------------------------------- #
def test_compute_loss_is_the_dice_of_the_pixel_field():
    head = _head()
    model = SimpleNamespace(geo=head)
    x = _sample()
    out = head(torch.randn(1, 16, 3, 4), torch.randn(32))
    sl = C.compute_loss(model, out, x, None)
    want = C.dice_coefficient(out["m_pix"].reshape(1, -1), x.gt_pix.reshape(1, -1)).mean()
    assert set(sl.terms) == {"dice"}
    assert float(sl.total) == pytest.approx(float(want), abs=1e-7)


def test_a_foreign_sample_contributes_exactly_zero_but_keeps_the_graph():
    """dynamic_mask_head.py:210-216 -- the no-instance dummy loss."""
    head = _head()
    x = _sample(fake=True)
    out = head(torch.randn(1, 16, 3, 4), torch.randn(32))
    sl = C.compute_loss(SimpleNamespace(geo=head), out, x, None)
    assert float(sl.total) == 0.0 and sl.total.requires_grad
    sl.total.backward()
    assert float(head.controller.proj.weight.grad.abs().sum()) == 0.0


def test_bce_ablation_swaps_the_term_name_and_the_value():
    head = _head(mask_loss="bce")
    x = _sample()
    out = head(torch.randn(1, 16, 3, 4), torch.randn(32))
    sl = C.compute_loss(SimpleNamespace(geo=head), out, x, None)
    assert set(sl.terms) == {"bce"}
    want = torch.nn.functional.binary_cross_entropy(out["m_pix"], x.gt_pix)
    assert float(sl.total) == pytest.approx(float(want), abs=1e-5)


def test_missing_pixel_gt_names_the_flag_instead_of_training_on_nothing():
    head = _head()
    x = _sample()
    x.gt_pix = None
    out = head(torch.randn(1, 16, 3, 4), torch.randn(32))
    with pytest.raises(ValueError, match="no-pixgt"):
        C.compute_loss(SimpleNamespace(geo=head), out, x, None)


def test_a_pixel_gt_on_the_wrong_grid_is_an_assertion_not_a_broadcast():
    head = _head()
    x = _sample()
    x.gt_pix = torch.zeros(6, 8)
    out = head(torch.randn(1, 16, 3, 4), torch.randn(32))
    with pytest.raises(AssertionError, match="pixgt_size"):
        C.compute_loss(SimpleNamespace(geo=head), out, x, None)


def test_geom_loss_is_zero_on_a_semantic_sample_and_counted():
    head = _head(geom_reg_weight=0.05, geom_route_loss="none")
    x = _sample(family="semantic")
    out = head(torch.randn(1, 16, 3, 4), torch.randn(32))
    sl = C.compute_loss(SimpleNamespace(geo=head), out, x, None)
    assert float(sl.terms["geom"]) == 0.0
    assert sl.stats["geom_zeroed"] == 1.0
    assert sl.stats["geom_route_target"] == C.GEOM_ROUTE_CLASSES.index("semantic")


def test_geom_loss_on_an_analytic_sample_is_l1_plus_bce_at_zero_init():
    head = _head(geom_reg_weight=0.05, geom_route_loss="none")
    geom = {"Left": 0.2, "Right": 0.6, "Top": 0.1, "Bottom": 0.5, "Angle": 0.0,
            "Feather": 50.0, "Flipped": "false"}
    x = _sample(family="radial", mask_type="circulargradient", geometry=geom)
    out = head(torch.randn(1, 16, 3, 4), torch.randn(32))
    sl = C.compute_loss(SimpleNamespace(geo=head), out, x, None)
    cont, flipped = C.geom_targets("circulargradient", geom)
    want = sum(abs(v) for v in cont) / len(cont) + math.log(2.0)   # zero-init head
    assert float(sl.terms["geom"]) == pytest.approx(want, abs=1e-5)
    assert sl.stats["geom_zeroed"] == 0.0
    assert float(sl.total) == pytest.approx(
        float(C.dice_coefficient(out["m_pix"].reshape(1, -1),
                                 x.gt_pix.reshape(1, -1)).mean()) + 0.05 * want,
        abs=1e-5)


def test_geom_targets_follow_raster_geometry_s_own_formulas():
    geom = {"Left": -1.1007, "Right": 2.0993, "Top": 0.0, "Bottom": 1.0913,
            "Angle": 45.0, "Feather": 2.0, "Flipped": "true"}
    cont, flipped = C.geom_targets("circulargradient", geom)
    assert cont[0] == pytest.approx((-1.1007 + 2.0993) / 2)
    assert cont[2] == pytest.approx(abs(2.0993 - (-1.1007)) / 2)
    assert cont[4] == pytest.approx(math.sin(2 * math.radians(45.0)))
    assert cont[5] == pytest.approx(math.cos(2 * math.radians(45.0)))
    assert cont[6] == pytest.approx(max(2.0 / 100.0, 0.05))    # the 0.05 floor
    assert flipped == 1.0
    lin, fl = C.geom_targets("gradient", {"ZeroX": 0.1, "ZeroY": 0.2,
                                          "FullX": 0.9, "FullY": 0.8})
    assert lin == [0.1, 0.2, 0.9, 0.8] and fl == 0.0
    assert C.geom_targets(None, None) is None


def test_center_supervision_is_off_by_default_and_l1_when_on():
    head = _head()
    x = _sample()
    out = head(torch.randn(1, 16, 3, 4), torch.randn(32))
    assert "center" not in C.compute_loss(SimpleNamespace(geo=head), out, x, None).terms

    head2 = _head(center_sup=True)
    out2 = head2(torch.randn(1, 16, 3, 4), torch.zeros(32))
    sl = C.compute_loss(SimpleNamespace(geo=head2), out2, x, None)
    tgt = C._centroid(x.gt_pix)
    assert float(sl.terms["center"]) == pytest.approx(
        float((out2["center"] - tgt).abs().mean()), abs=1e-6)


def test_centroid_of_an_empty_field_is_none_not_a_nan():
    assert C._centroid(torch.zeros(4, 4)) is None


# --------------------------------------------------------------------------- #
# optimizer / schedule
# --------------------------------------------------------------------------- #
def test_optimizer_spec_is_the_detectron2_recipe():
    C.set_config(CondInstConfig())
    spec = C.optimizer_spec(SimpleNamespace())
    assert (spec.type, spec.lr, spec.weight_decay) == ("sgd", 0.01, 1e-4)
    assert (spec.momentum, spec.nesterov) == (0.9, False)
    assert spec.grouping == "norm_bias" and spec.norm_weight_decay == 0.0


def test_norm_bias_grouping_exempts_batchnorm_and_decays_bias():
    from q3vl.whereb.amort.trainer import AmortTrainConfig, build_optimizer

    head = C.build_head(in_dim=16, text_dim=32, cfg=CondInstConfig())
    cfg = AmortTrainConfig(arm="CONDINST")
    opt = build_optimizer(head, cfg, C.optimizer_spec(SimpleNamespace()))
    assert isinstance(opt, torch.optim.SGD)
    decayed, exempt = opt.param_groups[0], opt.param_groups[1]
    assert decayed["weight_decay"] == 1e-4 and exempt["weight_decay"] == 0.0
    bn = head.mask_branch.refine[1]
    ids_exempt = {id(p) for p in exempt["params"]}
    assert {id(bn.weight), id(bn.bias)} <= ids_exempt
    assert id(head.controller.proj.bias) in {id(p) for p in decayed["params"]}


def test_schedule_is_the_carried_over_multistep_ladder():
    C.set_config(CondInstConfig())
    kw = C.scheduler_kwargs(SimpleNamespace(max_steps=1200), 1200)
    assert kw == {"milestones": [800, 1067], "gamma": 0.1,
                  "warmup_steps": 13, "warmup_factor": 1e-3}


def test_the_schedule_refuses_to_guess_a_step_budget():
    C.set_config(CondInstConfig())
    with pytest.raises(ValueError, match="lr-milestones"):
        C.scheduler_kwargs(SimpleNamespace(max_steps=0), 0)


def test_a_non_warmup_multistep_scheduler_is_refused():
    """Bypassing the wrapper must not silently give cosine + warmup_factor."""
    assert C.SCHEDULER_KIND == "warmup_multistep"
    C.set_config(CondInstConfig())
    with pytest.raises(SystemExit, match="--scheduler 'warmup_multistep'"):
        C.scheduler_kwargs(SimpleNamespace(max_steps=1200, scheduler="cosine"),
                           1200)
    # the arm config is checked too: `run_condinst_arm` copies `--scheduler`
    # into it, so a mismatch there is the same mixed schedule
    C.set_config(CondInstConfig(scheduler="cosine"))
    with pytest.raises(SystemExit, match="the arm config says 'cosine'"):
        C.scheduler_kwargs(SimpleNamespace(max_steps=1200,
                                           scheduler="warmup_multistep"), 1200)
    C.set_config(CondInstConfig())


def test_the_ladder_actually_reaches_the_lr(tmp_path):
    from q3vl.where.calibrate import make_scheduler

    C.set_config(CondInstConfig())
    head = C.build_head(in_dim=16, text_dim=32, cfg=CondInstConfig())
    from q3vl.whereb.amort.trainer import AmortTrainConfig, build_optimizer

    opt = build_optimizer(head, AmortTrainConfig(arm="CONDINST"),
                          C.optimizer_spec(SimpleNamespace()))
    sch = make_scheduler(opt, 1200, 0.03, "warmup_multistep",
                         **C.scheduler_kwargs(SimpleNamespace(max_steps=1200), 1200))
    lrs = []
    for _ in range(1200):
        lrs.append(opt.param_groups[0]["lr"])
        sch.step()
    assert lrs[0] == pytest.approx(0.01 * 1e-3, rel=1e-6)     # warmup factor
    assert lrs[13] == pytest.approx(0.01, rel=1e-6)           # full lr
    assert lrs[900] == pytest.approx(0.001, rel=1e-6)         # after milestone 1
    assert lrs[1100] == pytest.approx(0.0001, rel=1e-6)       # after milestone 2


def test_builder_kwargs_render_the_gt_on_the_heads_own_grid():
    C.set_config(CondInstConfig())
    assert C.builder_kwargs(SimpleNamespace())["pixgt_size"](3, 5) == (12, 20)
    C.set_config(CondInstConfig(mask_out_stride=8))
    assert C.builder_kwargs(SimpleNamespace())["pixgt_size"](3, 5) == (6, 10)


# --------------------------------------------------------------------------- #
# configuration guards
# --------------------------------------------------------------------------- #
def test_geom_regression_refuses_the_cgt_gt_source():
    with pytest.raises(ValueError, match="geometry"):
        CondInstConfig(geom_reg_weight=0.05, gt="cgt")


def test_a_stride_that_does_not_divide_sixteen_is_refused():
    with pytest.raises(ValueError, match="upsample factor"):
        CondInstConfig(mask_out_stride=3)


def test_readout_spec_maps_the_ctrl_alias():
    from q3vl.whereb.readout import ReadoutSpec

    C.set_config(CondInstConfig())
    args = SimpleNamespace(cond_readout="seg_where", readout_qtok=0, readout_nseg=1)
    assert C.readout_spec(args) == ReadoutSpec("seg_where")
    C.set_config(CondInstConfig(ctrl="pool"))
    assert C.readout_spec(args).kind == "where_span_pool"
    C.set_config(CondInstConfig(ctrl="query"))
    assert C.readout_spec(args).kind == "where_span_pool"
    args2 = SimpleNamespace(cond_readout="im_end", readout_qtok=0, readout_nseg=1)
    with pytest.raises(ValueError, match="same knob"):
        C.readout_spec(args2)


def test_qtok_readout_passes_through_untouched():
    C.set_config(CondInstConfig())
    spec = C.readout_spec(SimpleNamespace(cond_readout="qtok", readout_qtok=4,
                                          readout_nseg=1))
    assert spec.kind == "qtok" and spec.n_vectors == 4


# --------------------------------------------------------------------------- #
# criterion columns + runtime assertions
# --------------------------------------------------------------------------- #
def _rows(head, n=3, **kw):
    rows = []
    for i in range(n):
        x = _sample(**kw)
        x.sample_id = f"s{i}"
        out = head(torch.randn(1, 16, 3, 4), torch.randn(32))
        r = C.per_sample_row(SimpleNamespace(geo=head), out, x)
        r["family"] = x.family
        rows.append(r)
    return rows


def test_per_sample_row_carries_the_three_column_set_plus_the_controls():
    head = _head()
    r = _rows(head, 1)[0]
    for key in ("condinst_pix_soft_iou", "condinst_pix_topk_iou",
                "condinst_pix_boundary_f1", "condinst_pix_center_soft_iou",
                "condinst_pix_center_topk_iou", "condinst_pix_center_boundary_f1",
                "condinst_pix_random_floor", "condinst_gt_pix_source"):
        assert key in r, key
    assert r["condinst_pix_grid"] == [12, 16]
    assert 0.0 <= r["condinst_pix_topk_iou"] <= 1.0


def test_criteria_column_is_produced_and_a_board_with_it_passes_the_assertion():
    from q3vl.whereb.amort.evaluate import assert_criteria_ran

    head = _head()
    cols = C.criteria_columns(_rows(head, 3))
    col = cols["condinst_pix_readout"]
    assert col["n"] == 3 and col["soft_iou"]["n"] == 3
    assert col["gt_pix_source_counts"] == {"render": 3}
    assert col["gt_pix_fallback_n"] == 0
    rep = assert_criteria_ran({"criteria_columns": cols}, "CONDINST",
                              head_facts=head.facts(), steps_row={"L_dice": 0.4})
    assert rep["required"] == ["condinst_pix_readout"]
    assert rep["computed"]["condinst_pix_readout"] == 3


def test_an_empty_criterion_column_refuses_the_board():
    from q3vl.whereb.amort.evaluate import assert_criteria_ran

    cols = C.criteria_columns([])
    assert cols["condinst_pix_readout"]["n"] == 0
    with pytest.raises(AssertionError, match="condinst_pix_readout"):
        assert_criteria_ran({"criteria_columns": cols}, "CONDINST")


def test_the_head_promises_no_deep_supervision_branches():
    from q3vl.whereb.amort.evaluate import deep_supervision_tags

    assert deep_supervision_tags(_head().facts()) == []
    assert deep_supervision_tags(_head(geom_reg_weight=0.05).facts()) == []


def test_the_geom_route_column_exists_and_is_asserted_when_the_branch_is_on():
    head = _head(geom_reg_weight=0.05)
    geom = {"Left": 0.2, "Right": 0.6, "Top": 0.1, "Bottom": 0.5, "Angle": 10.0,
            "Feather": 50.0, "Flipped": "false"}
    rows = _rows(head, 2, family="radial", mask_type="circulargradient",
                 geometry=geom)
    cols = C.criteria_columns(rows)
    route = cols["geom_reg_route"]
    assert route["n"] == 2 and 0.0 <= route["accuracy"] <= 1.0
    assert route["by_family"]["radial"]["n"] == 2
    assert "cx" in route["component_l1_median"]
    with pytest.raises(AssertionError, match="geom_reg_route"):
        C.criteria_columns([])           # geom is on, so an empty board is refused


def test_the_first_step_witness_requires_the_registered_loss_columns():
    C.set_config(CondInstConfig())
    assert C.assert_first_step_columns({"L_dice": 0.5})["present"]
    with pytest.raises(AssertionError, match="L_dice"):
        C.assert_first_step_columns({"loss": 0.5})
    C.set_config(CondInstConfig(geom_reg_weight=0.05))
    with pytest.raises(AssertionError, match="L_geom"):
        C.assert_first_step_columns({"L_dice": 0.5})
    assert C.assert_first_step_columns({"L_dice": 0.5, "L_geom": 0.1})["present"]
    C.set_config(CondInstConfig(mask_loss="bce"))
    assert C.assert_first_step_columns({"L_bce": 0.5})["present"]


def test_the_terms_the_loss_emits_are_the_columns_the_witness_asks_for():
    """The witness and the loss must agree, or one of them is decorative."""
    from q3vl.whereb.amort.losses import aggregate

    head = _head(geom_reg_weight=0.05)
    x = _sample(family="radial", mask_type="circulargradient",
                geometry={"Left": 0.2, "Right": 0.6, "Top": 0.1, "Bottom": 0.5})
    out = head(torch.randn(1, 16, 3, 4), torch.randn(32))
    sl = C.compute_loss(SimpleNamespace(geo=head), out, x, None)
    sl.stats.update(is_fake=0.0, area_ratio=1.0, pred_std=0.1, gt_std=0.1,
                    pred_area=0.1, gt_area=0.1, has_partner=0.0)
    _, stats = aggregate([sl])
    C.assert_first_step_columns(stats)


# --------------------------------------------------------------------------- #
# the entry wrapper
# --------------------------------------------------------------------------- #
def test_wrapper_defaults_are_the_faithful_recipe():
    from q3vl.whereb.scripts import run_condinst_arm as R

    own, rest = R.build_parser().parse_known_args([])
    cfg = R.config_from(own, R._peek(rest))
    assert cfg == CondInstConfig(scheduler="warmup_multistep", seed=20260810)
    assert cfg.milestones(1200) == [800, 1067] and cfg.warmup(1200) == 13


def test_wrapper_maps_every_ablation_flag():
    from q3vl.whereb.scripts import run_condinst_arm as R

    own, rest = R.build_parser().parse_known_args(
        ["--condinst-no-rel-coords", "--condinst-mask-loss", "bce",
         "--condinst-gt", "cgt", "--condinst-ctrl", "pool",
         "--condinst-center-sup", "--optimizer", "adamw"])
    cfg = R.config_from(own, R._peek(rest))
    assert cfg.disable_rel_coords and cfg.mask_loss == "bce" and cfg.gt == "cgt"
    assert cfg.ctrl == "pool" and cfg.center_sup and cfg.optimizer == "adamw"
    assert cfg.readout_kind == "where_span_pool"


def test_wrapper_hands_the_base_entry_the_b4_command_line(tmp_path, monkeypatch):
    """The whole delegation, with the base entry stubbed out: which flags the
    run actually gets, and that the frozen record lands next to them."""
    import json

    import q3vl.whereb.amort.trainer as T
    import q3vl.whereb.scripts.run_amort_arm as base
    from q3vl.whereb.amort import arms as A
    from q3vl.whereb.scripts import run_condinst_arm as R

    seen: dict = {}
    monkeypatch.setattr(T, "aggregate", T.aggregate)          # restored on teardown
    monkeypatch.setattr(base, "main", lambda rest: (seen.setdefault("rest", rest), 0)[1])
    rc = R.main(["--run-name", "cond_t", "--out-root", str(tmp_path),
                 "--max-steps", "1200"])
    assert rc == 0
    rest = seen["rest"]
    for flag in ("--arm", "--no-semantic-head", "--no-sim-field", "--no-film",
                 "--scheduler", "--max-grad-norm", "--pixgt-source"):
        assert flag in rest, flag
    assert R._peek_value(rest, "--arm") == "CONDINST"
    assert R._peek_value(rest, "--scheduler") == "warmup_multistep"
    assert R._peek_value(rest, "--max-grad-norm") == "0"
    assert R._peek_value(rest, "--pixgt-source") == "render"
    assert "--cond-readout" not in rest              # seg = the base default
    assert A.ARM_KWARGS["cfg"] == CondInstConfig(seed=20260810)

    rec = json.loads((tmp_path / "cond_t" / "config" / "condinst_setup.json")
                     .read_text(encoding="utf-8"))
    assert rec["schedule"]["milestones"] == [800, 1067]
    assert rec["schedule"]["warmup_steps"] == 13
    assert rec["optimizer"]["lr"] == 0.01 and rec["optimizer"]["type"] == "sgd"
    assert len(rec["condinst_sha256"]) == 64 and len(rec["wrapper_sha256"]) == 64
    assert rec["config"]["geom_on"] is False
    A.ARM_KWARGS.clear()
    A.ARM_SETUP.clear()


def test_wrapper_forwards_the_cgt_gt_source_and_the_ctrl_alias(tmp_path, monkeypatch):
    import q3vl.whereb.amort.trainer as T
    import q3vl.whereb.scripts.run_amort_arm as base
    from q3vl.whereb.amort import arms as A
    from q3vl.whereb.scripts import run_condinst_arm as R

    seen: dict = {}
    monkeypatch.setattr(T, "aggregate", T.aggregate)
    monkeypatch.setattr(base, "main", lambda rest: (seen.setdefault("rest", rest), 0)[1])
    R.main(["--condinst-gt", "cgt", "--condinst-ctrl", "pool"])
    assert R._peek_value(seen["rest"], "--pixgt-source") == "cgt1024"
    assert R._peek_value(seen["rest"], "--cond-readout") == "where_span_pool"
    A.ARM_KWARGS.clear()
    A.ARM_SETUP.clear()


def test_wrapper_keeps_the_legacy_routing_row_available(tmp_path, monkeypatch):
    """`--newarm-legacy-routing` must not arrive with the semantic head off."""
    import json

    import q3vl.whereb.amort.trainer as T
    import q3vl.whereb.scripts.run_amort_arm as base
    from q3vl.whereb.amort import arms as A
    from q3vl.whereb.scripts import run_condinst_arm as R

    seen: dict = {}
    monkeypatch.setattr(T, "aggregate", T.aggregate)
    monkeypatch.setattr(base, "main", lambda rest: (seen.setdefault("rest", rest), 0)[1])
    R.main(["--newarm-legacy-routing", "--run-name", "cond_legacy",
            "--out-root", str(tmp_path), "--max-steps", "1200"])
    rest = seen["rest"]
    assert "--newarm-legacy-routing" in rest
    assert "--no-semantic-head" not in rest
    # the two non-routing flags still go in
    assert "--no-sim-field" in rest and "--no-film" in rest
    rec = json.loads((tmp_path / "cond_legacy" / "config" / "condinst_setup.json")
                     .read_text(encoding="utf-8"))
    assert "legacy" in rec["injected_flags"]["--no-semantic-head"]
    A.ARM_KWARGS.clear()
    A.ARM_SETUP.clear()


def test_wrapper_injects_the_b4_flags_but_never_overrides_the_caller():
    from q3vl.whereb.scripts import run_condinst_arm as R

    injected: dict = {}
    rest = ["--max-grad-norm", "1.0"]
    for flag, val in (("--no-semantic-head", None), ("--no-sim-field", None),
                      ("--no-film", None), ("--scheduler", "warmup_multistep"),
                      ("--max-grad-norm", "0"), ("--pixgt-source", "render")):
        R._ensure(rest, flag, val, injected)
    assert "--max-grad-norm" not in injected            # the caller's 1.0 stands
    assert rest.count("--max-grad-norm") == 1
    assert injected["--scheduler"] == "warmup_multistep"
    assert injected["--no-film"] is True
    assert R._peek_value(rest, "--pixgt-source") == "render"
