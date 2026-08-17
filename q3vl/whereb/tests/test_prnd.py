"""EPR-020 (arm ``PRND``) -- the PointRend port, on CPU.

Three groups:

* the port itself -- structure, initialisation, the two-term loss, the
  optimiser/schedule the reference specifies;
* the proposal's pre-registered pre-commit alignment tests (a) / (b) / (c);
* the wiring the campaign's red lines demand -- the criterion column refuses an
  empty board, the two publication assertions fire, the point sampler never
  touches the global RNG stream, and turning the arm off changes nothing.

Everything here runs with ``CUDA_VISIBLE_DEVICES=""``.
"""

from __future__ import annotations

import types

import pytest
import torch

from q3vl.whereb.amort import prnd

TEXT_DIM = 2560
IN_DIM = 1024


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _head(**kw):
    # coarse_dim 32: GroupNorm(32, C) is the reference's own norm
    # (defaults.py:415), so any toy head still needs C % 32 == 0
    p = dict(in_dim=8, text_dim=16, coarse_dim=32, fc_dim=12, num_fc=3,
             train_points=32, subdiv_steps=2, subdiv_points=64)
    p.update(kw)
    return prnd.PointRendHead(**p)


class _Sample:
    """The fields of ``AmortSampleInputs`` this arm reads."""

    def __init__(self, gh=3, gw=4, up=4, dim=8, tdim=16, is_fake=False,
                 pixgt=None, family="radial"):
        H, W = up * gh, up * gw
        self.sample_id = "s1"
        self.feat = torch.randn(1, dim, gh, gw)
        self.grid_h, self.grid_w = gh, gw
        self.gt_low = torch.rand(gh, gw)
        self.gt_hi = torch.rand(H, W)
        self.gt_pix = torch.rand(H, W)
        self.gt_pix_source = "maskhi512"
        self.pixgt = pixgt
        self.is_fake = is_fake
        self.family = family
        self.h_cond = torch.randn(1, tdim)
        self.meta = {}


def _ctx(x, **kw):
    from q3vl.whereb.amort.arms import ArmContext

    base = dict(feat=x.feat, grid_h=x.grid_h, grid_w=x.grid_w,
                h_cond=x.h_cond, sample=x)
    base.update(kw)
    return ArmContext(**base)


class _Model:
    """The two attributes ``forward`` / ``compute_loss`` read off the model."""

    def __init__(self, head, training=True):
        self.geo = head
        self.training = training
        self.arm = "PRND"
        self.is_new_arm = True


@pytest.fixture(autouse=True)
def _clean_options():
    """Module state is per-run in production and must be per-test here.

    ``prnd.install_publication_assert`` replaces a function on the SHARED
    ``evaluate`` module, so leaving it installed would change what
    ``assert_criteria_ran`` does for every later test in the session.
    ``FIRST_LOSS_COLUMNS`` (the first micro-batch's ``L_*`` witness) and
    ``RUN_CONTEXT`` (the run's ``steps.jsonl`` path) are the same kind of
    per-run state: a test that ran the loss would otherwise hand the next test
    a witness it never produced.
    """
    import q3vl.whereb.amort.evaluate as ev

    saved = dict(prnd.OPTIONS)
    saved_assert = ev.assert_criteria_ran
    prnd.OPTIONS.clear()
    prnd.FIRST_LOSS_COLUMNS.clear()
    prnd.RUN_CONTEXT.update(steps_path=None, eval_only=False)
    yield
    prnd.OPTIONS.clear()
    prnd.OPTIONS.update(saved)
    prnd.FIRST_LOSS_COLUMNS.clear()
    prnd.RUN_CONTEXT.update(steps_path=None, eval_only=False)
    ev.assert_criteria_ran = saved_assert


# --------------------------------------------------------------------------- #
# 1. registry contract
# --------------------------------------------------------------------------- #
def test_the_registry_can_load_this_arm():
    from q3vl.whereb.amort import arms as A

    mod = A.load_arm("PRND")
    assert mod is prnd
    assert prnd.ARM == "PRND" and prnd.CRITERIA == ("prnd_point_readout",)
    for hook in A.REQUIRED_HOOKS:
        assert callable(getattr(mod, hook))
    for hook in ("head_kwargs_from_args", "optimizer_spec", "scheduler_kwargs",
                 "builder_kwargs", "per_sample_row", "criteria_columns",
                 "train_stats", "add_arguments"):
        assert callable(A.arm_hook("PRND", hook) or getattr(mod, hook)), hook


def test_the_head_carries_no_deep_supervision_tags():
    """PointRend has no auxiliary branches; a stray key would make
    ``assert_criteria_ran`` demand ``L_*_{tag}`` columns that cannot exist."""
    from q3vl.whereb.amort.evaluate import deep_supervision_tags

    assert deep_supervision_tags(_head().facts()) == []


def test_no_st_lang_part_is_imported():
    """§1: none of ST_LANG's parts may enter this arm.  Checked on the IMPORT
    statements (prose about what is absent is allowed to name them)."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(prnd))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    for banned in ("ConvTower", "FiLM", "CondEncoder", "SemanticHead",
                   "UniQHead", "P1Head", "P3PrimeHead", "ShapeDistHead"):
        assert banned not in imported, banned
    for mod in imported:
        assert not mod.startswith(("q3vl.whereb.amort.uniq", ".uniq")), mod
        assert mod not in ("q3vl.whereb.amort.heads", ".heads"), mod


# --------------------------------------------------------------------------- #
# 2. the port: structure and initialisation
# --------------------------------------------------------------------------- #
def test_reference_defaults_are_the_published_numbers():
    d = prnd.DEFAULTS
    assert d["train_points"] == 1024        # Base-PointRend-Semantic-FPN.yaml:13
    assert d["oversample"] == 3.0           # config.py:35
    assert d["importance"] == 0.75          # config.py:38
    assert d["fc_dim"] == 256 and d["num_fc"] == 3          # yaml:10-11
    assert d["coarse_dim"] == 128           # defaults.py:411
    assert d["coarse_pred_each_layer"] is False             # yaml:17
    assert d["subdiv_points"] == 8192       # yaml:15
    assert d["subdiv_steps"] == 4           # N4
    assert d["point_loss"] == "bce"         # point_head.py:73-75, no dice
    assert (d["optimizer"], d["lr"], d["momentum"], d["wd"], d["gamma"]) == \
        ("sgd", 0.01, 0.9, 1e-4, 0.1)
    assert d["warmup_iters"] == 18 and d["milestones"] == "738,1015"
    assert prnd.GN_GROUPS == 32
    assert (prnd.M2F_MASK_WEIGHT, prnd.M2F_DICE_WEIGHT) == (5.0, 5.0)


def test_coarse_head_is_one_conv_gn_relu_plus_1x1():
    h = _head(in_dim=64, text_dim=32, coarse_dim=64)
    c = h.coarse
    assert isinstance(c.conv, torch.nn.Conv2d)
    assert c.conv.in_channels == 64 + 64 and c.conv.out_channels == 64
    assert c.conv.kernel_size == (3, 3) and c.conv.padding == (1, 1)
    assert c.conv.bias is None                 # bias = not norm (GN)
    assert isinstance(c.norm, torch.nn.GroupNorm)
    assert c.norm.num_groups == 32 and c.norm.num_channels == 64
    assert c.predictor.kernel_size == (1, 1) and c.predictor.out_channels == 1


def test_point_head_is_three_conv1d_k1_then_a_predictor():
    h = _head(in_dim=64, fc_dim=32, num_fc=3)
    ph = h.point_head
    assert len(ph.fc_layers) == 3
    assert ph.fc_layers[0].in_channels == 64 + 1      # fc_dim_in = C + num_classes
    for fc in ph.fc_layers:
        assert fc.kernel_size == (1,) and fc.out_channels == 32
    assert ph.fc_layers[1].in_channels == 32          # coarse_pred_each_layer off
    assert ph.predictor.in_channels == 32 and ph.predictor.out_channels == 1


def test_coarse_pred_each_layer_widens_every_layer():
    ph = _head(in_dim=64, fc_dim=32, coarse_pred_each_layer=True).point_head
    assert ph.fc_layers[1].in_channels == 32 + 1
    assert ph.predictor.in_channels == 32 + 1


def test_point_predictor_is_near_zero_and_coarse_predictor_is_not():
    """step-0 state, straight from the reference (point_head.py:119-121 vs
    semantic_seg.py:215).  Written down because "the coarse field is not
    constant at step 0" is a property, not a bug."""
    h = _head(in_dim=8, coarse_dim=32)
    assert float(h.point_head.predictor.weight.detach().abs().max()) < 0.01
    assert float(h.point_head.predictor.bias.detach().abs().max()) == 0.0
    assert float(h.coarse.predictor.weight.detach().abs().max()) > 0.01
    assert float(h.coarse.predictor.bias.detach().abs().max()) == 0.0
    assert float(h.coarse.conv.weight.detach().abs().max()) > 0.0


def test_step0_point_logits_are_near_zero_so_point_sigmoid_is_half():
    h = _head(in_dim=8, text_dim=16, coarse_dim=32, train_points=64)
    feat = torch.randn(1, 8, 3, 4)
    l_c = h.coarse_logits(feat, torch.randn(1, 16))
    coords = h.sample_train_points(l_c)
    l_p = h.point_logits(feat, l_c, coords)
    assert float(torch.sigmoid(l_p).detach().sub(0.5).abs().max()) < 0.05


def test_trainable_parameter_count_matches_the_proposal_arithmetic():
    """Proposal note 5: ~2.2M.  Exact: conv (1024+128)*128*9 + GN 256 +
    1x1 129 + Linear 2560*128+128 + point head 1025*256+256 + 2*(256*256+256)
    + 257."""
    h = prnd.PointRendHead(in_dim=IN_DIM, text_dim=TEXT_DIM)
    expect = ((IN_DIM + 128) * 128 * 9) + 2 * 128 + (128 + 1) \
        + (TEXT_DIM * 128 + 128) \
        + (1025 * 256 + 256) + 2 * (256 * 256 + 256) + (256 + 1)
    assert h.n_trainable() == expect
    assert 1.9e6 < expect < 2.3e6


def test_cond_projection_is_a_plain_linear_2560_to_128():
    h = prnd.PointRendHead(in_dim=IN_DIM, text_dim=TEXT_DIM)
    assert h.coarse.cond_proj.in_features == TEXT_DIM
    assert h.coarse.cond_proj.out_features == 128
    # K > 1 (the qtok / nseg readout rows): same Linear, mean over the rows
    g1 = h.coarse.cond_channels(torch.zeros(1, TEXT_DIM))
    g4 = h.coarse.cond_channels(torch.zeros(4, TEXT_DIM))
    assert g1.shape == (128,) and torch.allclose(g1, g4)


def test_the_condition_actually_reaches_the_coarse_field():
    h = _head()
    feat = torch.randn(1, 8, 3, 4)
    a = h.coarse_logits(feat, torch.randn(1, 16))
    b = h.coarse_logits(feat, torch.randn(1, 16))
    assert not torch.allclose(a, b)


# --------------------------------------------------------------------------- #
# 3. point sampling
# --------------------------------------------------------------------------- #
def test_training_sampler_returns_the_pre_registered_split():
    h = _head(train_points=100, oversample=3.0, importance=0.75)
    l_c = torch.randn(1, 1, 5, 7)
    p = h.sample_train_points(l_c)
    assert p.shape == (1, 100, 2)
    assert float(p.min()) >= 0.0 and float(p.max()) <= 1.0


def test_uncertainty_is_the_l1_distance_to_the_decision_surface():
    x = torch.tensor([[[-2.0, 0.0, 3.0]]])
    assert torch.allclose(prnd.calculate_uncertainty(x),
                          torch.tensor([[[-2.0, -0.0, -3.0]]]))
    with pytest.raises(ValueError):
        prnd.calculate_uncertainty(torch.zeros(1, 2, 3))


def test_importance_points_land_on_the_uncertain_half():
    """The first ``beta*N`` coordinates are the importance half.  With a coarse
    field whose left half has |logit| ~ 0 and whose right half is saturated,
    they must be the left half; the remaining ``(1-beta)*N`` stay uniform."""
    l_c = torch.full((1, 1, 8, 32), 20.0)
    l_c[..., :16] = 0.0
    h = _head(train_points=200, oversample=3.0, importance=0.75)
    p = h.sample_train_points(l_c)
    imp_x, rnd_x = p[0, :150, 0], p[0, 150:, 0]
    assert float((imp_x < 0.5).float().mean()) > 0.95
    assert 0.2 < float((rnd_x < 0.5).float().mean()) < 0.8
    assert rnd_x.shape == (50,)


def test_uncertainty_is_computed_on_the_sampled_logits_not_on_the_map():
    """``point_features.py:92-98``, the reference's own warning.  On a coarse
    field that alternates +1 / -1 per column, EVERY cell of the map has
    uncertainty -1, so computing on the map first would leave the importance
    half indistinguishable from a uniform draw.  Computing on the sampled
    logits instead concentrates it on the interpolated zero crossings."""
    l_c = torch.ones(1, 1, 8, 32)
    l_c[..., 1::2] = -1.0
    h = _head(train_points=200, oversample=3.0, importance=0.75)
    p = h.sample_train_points(l_c)
    v = prnd.point_sample(l_c, p, align_corners=False).reshape(-1).abs()
    imp, rnd = v[:150], v[150:]
    assert float(imp.mean()) < 0.15
    assert float(rnd.mean()) > 3 * float(imp.mean())


def test_grid_selection_returns_cell_centres():
    unc = torch.zeros(1, 1, 2, 2)
    unc[0, 0, 1, 1] = 5.0
    idx, coords = prnd.get_uncertain_point_coords_on_grid(unc, 1)
    assert int(idx[0, 0]) == 3
    assert torch.allclose(coords[0, 0], torch.tensor([0.75, 0.75]))


def test_grid_selection_is_capped_by_the_grid_size():
    _idx, coords = prnd.get_uncertain_point_coords_on_grid(
        torch.randn(1, 1, 4, 5), 8192)
    assert coords.shape == (1, 20, 2)


# --- RNG independence (proposal §3 "RNG 纪律", N1 lesson) --------------------
def test_point_sampling_never_touches_the_global_rng_stream():
    h = _head(train_points=64)
    l_c = torch.randn(1, 1, 5, 7)
    torch.manual_seed(1234)
    before = torch.rand(8)
    torch.manual_seed(1234)
    for _ in range(5):
        h.sample_train_points(l_c)
    after = torch.rand(8)
    assert torch.equal(before, after)


def test_two_heads_with_the_same_seed_draw_the_same_points():
    l_c = torch.randn(1, 1, 5, 7)
    a = _head(train_points=64, point_seed=20260810 + 314)
    b = _head(train_points=64, point_seed=20260810 + 314)
    c = _head(train_points=64, point_seed=20260810 + 315)
    assert torch.equal(a.sample_train_points(l_c), b.sample_train_points(l_c))
    assert not torch.equal(a.sample_train_points(l_c), c.sample_train_points(l_c))


def test_the_point_seed_is_cfg_seed_plus_314():
    assert prnd.POINT_SEED_OFFSET == 314
    args = types.SimpleNamespace(seed=4242)
    h = prnd.build_head(in_dim=8, text_dim=16, args=args)
    assert h.point_seed == 4242 + 314
    assert h.facts()["point_rng"]["dedicated_generator"] is True


# --------------------------------------------------------------------------- #
# 4. the pre-registered alignment tests (proposal §3 "初始化" row)
# --------------------------------------------------------------------------- #
def test_a_point_sample_at_pixel_centres_recovers_the_field_exactly():
    """(a) ``((u+0.5)/W, (v+0.5)/H)`` must reproduce ``alpha[v, u]`` bit for
    bit -- this pins BOTH the ``2x-1`` remap and ``align_corners=False``.  One
    of the two wrong = half a cell = 8 px on the 32x48 grid."""
    for (H, W) in ((5, 7), (32, 48)):
        alpha = torch.rand(H, W)
        v, u = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        coords = torch.stack([(u.reshape(-1) + 0.5) / W,
                              (v.reshape(-1) + 0.5) / H], dim=-1)[None].float()
        got = prnd.point_sample(alpha[None, None], coords,
                                align_corners=False).reshape(H, W)
        assert torch.allclose(got, alpha, atol=1e-6)


def test_a_negative_control_the_wrong_alignment_does_not_recover_it():
    alpha = torch.rand(16, 16)
    H, W = alpha.shape
    v, u = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    coords = torch.stack([(u.reshape(-1) + 0.5) / W,
                          (v.reshape(-1) + 0.5) / H], dim=-1)[None].float()
    wrong = torch.nn.functional.grid_sample(
        alpha[None, None], 2.0 * coords.reshape(1, 1, -1, 2) - 1.0,
        align_corners=True).reshape(H, W)
    assert not torch.allclose(wrong, alpha, atol=1e-3)


def _geom_sample():
    """One analytic ``circulargradient`` geometry, in ``raster_geometry``'s own
    parameter form (``dataset_build/src/construct/canonical_masks.py:98-123``)."""
    return "circulargradient", {"Top": "0.2", "Bottom": "0.7", "Left": "0.25",
                                "Right": "0.8", "Angle": "0", "Feather": "60"}


def test_b_the_pixel_gt_and_the_coarse_grid_share_one_image_domain():
    """(b) spec-5 has ``H = 16*gh`` exactly, so the two renders sample the same
    ``[0,1]^2``: the fine render restricted to every 16th sample must BE the
    coarse render.  Any pad seam or off-by-one domain would break the identity."""
    pixgt = pytest.importorskip("q3vl.whereb.amort.pixgt")
    pytest.importorskip("dataset_build.src.construct.canonical_masks")
    mt, geom = _geom_sample()
    gh, gw, up = 8, 12, 16
    fine = pixgt.render_analytic(mt, geom, up * gh, up * gw)
    coarse = pixgt.render_analytic(mt, geom, gh, gw)
    assert torch.allclose(fine[::up, ::up], coarse, atol=1e-6)


def test_b2_sampling_the_fine_render_at_coarse_centres_stays_inside_the_bound():
    """(b), literal form: ``point_sample`` of the fine render at the coarse
    cell centres, against the coarse render, within the field's own
    interpolation bound (its largest neighbour-to-neighbour step).  A whole-cell
    or axis-swapped misalignment blows straight through that bound."""
    pixgt = pytest.importorskip("q3vl.whereb.amort.pixgt")
    pytest.importorskip("dataset_build.src.construct.canonical_masks")
    mt, geom = _geom_sample()
    gh, gw, up = 32, 48, 16                       # the real spec-5 geometry
    fine = pixgt.render_analytic(mt, geom, up * gh, up * gw)
    coarse = pixgt.render_analytic(mt, geom, gh, gw)
    coords = pixgt.make_coord(gh, gw)[None]
    got = prnd.point_sample(fine[None, None], coords,
                            align_corners=False).reshape(gh, gw)
    bound = max(float((coarse[:, 1:] - coarse[:, :-1]).abs().max()),
                float((coarse[1:] - coarse[:-1]).abs().max())) + 1e-4
    assert float((got - coarse).abs().max()) <= bound
    # two controls, both of which a misaligned domain would produce
    shifted = coords.clone()
    shifted[..., 0] = shifted[..., 0] + 1.0 / gw      # one whole coarse cell
    assert float((prnd.point_sample(fine[None, None], shifted,
                                    align_corners=False).reshape(gh, gw)
                  - coarse).abs().max()) > bound
    swapped = prnd.point_sample(fine[None, None], coords.flip(-1),
                                align_corners=False).reshape(gh, gw)
    assert float((swapped - coarse).abs().max()) > bound


def test_c_subdivision_lands_exactly_on_the_pixel_grid():
    """(c) four doublings, no resize to make up the difference."""
    h = _head(in_dim=8, text_dim=16, coarse_dim=32, subdiv_steps=4,
              subdiv_points=64)
    feat = torch.randn(1, 8, 3, 4)
    l_c = h.coarse_logits(feat, torch.randn(1, 16))
    out = h.subdivision(feat, l_c)
    assert tuple(out.shape) == (1, 1, 3 * 16, 4 * 16)


def test_subdivision_reads_the_coarse_features_from_the_original_field():
    """``semantic_seg.py:122-124``: coarse features come from ``L_c``, never
    from the partially refined map."""
    import inspect

    src = inspect.getsource(prnd.PointRendHead.subdivision)
    assert "point_sample(coarse_logits, point_coords" in src


# --------------------------------------------------------------------------- #
# 5. forward / loss
# --------------------------------------------------------------------------- #
def test_forward_training_emits_points_and_a_grid_sized_m_low():
    h = _head()
    x = _Sample()
    out = prnd.forward(_Model(h, training=True), h, _ctx(x))
    assert out["m_low"].shape == (x.grid_h, x.grid_w)
    assert out["prnd"]["point_coords"].shape == (1, 32, 2)
    assert out["prnd"]["point_logits"].shape == (1, 1, 32)
    assert out["prnd"]["m_low_source"] == "coarse"


def test_forward_eval_runs_subdivision_and_projects_with_the_gt_low_operator():
    h = _head(subdiv_steps=2, subdiv_points=64)
    x = _Sample()
    out = prnd.forward(_Model(h, training=False), h, _ctx(x))
    assert out["prnd"]["m_hi"].shape == (1, 1, 4 * x.grid_h, 4 * x.grid_w)
    assert out["prnd"]["m_low_source"] == "subdivision"
    from q3vl.where.upsample import area_resize

    assert torch.allclose(
        out["m_low"],
        area_resize(out["prnd"]["m_hi"], (x.grid_h, x.grid_w))[0, 0])


def test_no_subdivision_row_deploys_the_coarse_field():
    h = _head(no_subdivision=True, subdiv_steps=2)
    x = _Sample()
    out = prnd.forward(_Model(h, training=False), h, _ctx(x))
    assert out["prnd"]["m_low_source"] == "coarse"
    assert out["prnd"]["m_hi_source"] == "coarse_up"
    assert torch.allclose(out["m_low"],
                          torch.sigmoid(out["prnd"]["coarse_logits"])[0, 0])


def test_forward_without_h_cond_names_the_missing_flag():
    h = _head()
    x = _Sample()
    with pytest.raises(ValueError, match="ReadoutBuilder"):
        prnd.forward(_Model(h), h, _ctx(x, h_cond=None))


def test_loss_is_exactly_two_bce_terms_and_no_dice():
    l_c = torch.zeros(1, 1, 2, 3)
    alpha = torch.full((8, 12), 0.25)
    l_p = torch.zeros(1, 1, 16)
    a_p = torch.full((1, 16), 0.25)
    terms = prnd.pointrend_losses(l_c, l_p, alpha, a_p, point_loss="bce")
    assert set(terms) == {"coarse", "point"}
    # BCEwithLogits(0, t) = log 2 for any t
    want = torch.log(torch.tensor(2.0))
    assert torch.allclose(terms["coarse"], want, atol=1e-6)
    assert torch.allclose(terms["point"], want, atol=1e-6)


def test_coarse_term_upsamples_before_it_scores():
    """``semantic_seg.py:255-262``: interpolate to the input resolution FIRST.
    A coarse logit of +5 over a GT that is 1 on the top half and 0 on the
    bottom must therefore be penalised on the bottom half."""
    l_c = torch.full((1, 1, 2, 2), 5.0)
    alpha = torch.zeros(8, 8)
    alpha[:4] = 1.0
    t = prnd.pointrend_losses(l_c, torch.zeros(1, 1, 4), alpha,
                             torch.zeros(1, 4))["coarse"]
    assert float(t) > 2.0


def test_pixel_gt_must_be_an_integer_isotropic_multiple_of_the_grid():
    with pytest.raises(AssertionError, match="integer, isotropic multiple"):
        prnd.pointrend_losses(torch.zeros(1, 1, 3, 4), torch.zeros(1, 1, 4),
                              torch.zeros(20, 40), torch.zeros(1, 4))


def test_m2f_row_carries_dice_with_the_5_0_weights():
    l_c = torch.zeros(1, 1, 2, 3)
    alpha = torch.zeros(8, 12)
    l_p = torch.zeros(1, 1, 16)
    a_p = torch.zeros(1, 16)
    terms = prnd.pointrend_losses(l_c, l_p, alpha, a_p, point_loss="m2f")
    assert set(terms) == {"coarse", "point", "dice"}
    ce = prnd.sigmoid_ce_loss(l_p.reshape(1, -1), a_p, 1.0)
    assert torch.allclose(terms["point"], 5.0 * ce, atol=1e-6)
    assert torch.allclose(terms["dice"], 5.0 * prnd.dice_loss(
        l_p.reshape(1, -1), a_p, 1.0), atol=1e-6)


def test_dice_on_an_empty_target_is_finite_no_zero_division():
    p = torch.full((1, 64), -8.0)
    t = torch.zeros(1, 64)
    d = prnd.dice_loss(p, t, 1.0)
    assert torch.isfinite(d) and float(d) < 0.05


def test_fake_samples_get_a_zeroed_target_and_a_finite_gradient():
    """§2.1: the conservative policy is `alpha_hi := 0`, both terms unchanged,
    nothing excluded and nothing reweighted -- and neither term divides by a GT
    area, so there is no empty-GT zero division."""
    h = _head()
    x = _Sample(is_fake=True)
    m = _Model(h, training=True)
    out = prnd.forward(m, h, _ctx(x))
    sl = prnd.compute_loss(m, out, x, None)
    assert torch.isfinite(sl.total) and sl.total.requires_grad
    sl.total.backward()
    assert h.coarse.predictor.weight.grad is not None
    alpha, src = prnd.pixel_gt(x)
    assert float(alpha.abs().max()) == 0.0 and src.endswith("+fake_zeroed")


def test_loss_terms_are_named_so_steps_jsonl_carries_L_coarse_and_L_point():
    from q3vl.whereb.amort.losses import aggregate

    h = _head()
    x = _Sample()
    m = _Model(h, training=True)
    sl = prnd.compute_loss(m, prnd.forward(m, h, _ctx(x)), x, None)
    sl.stats = {**sl.stats, "is_fake": 0.0, "pred_area": 0.5, "gt_area": 0.5,
                "area_ratio": 1.0, "pred_std": 0.1, "gt_std": 0.1,
                "has_partner": 0.0}
    _total, stats = aggregate([sl])
    assert "L_coarse" in stats and "L_point" in stats
    # the observation columns of the proposal's readout list (c)/(d)
    assert "L_diag_point_accuracy" in stats and "L_diag_point_n" in stats


def test_compute_loss_refuses_a_forward_that_ran_in_eval_mode():
    h = _head()
    x = _Sample()
    m_eval = _Model(h, training=False)
    out = prnd.forward(m_eval, h, _ctx(x))
    with pytest.raises(AssertionError, match="stuck in eval mode"):
        prnd.compute_loss(_Model(h, training=True), out, x, None)


def test_point_targets_come_from_the_closed_form_on_the_analytic_row():
    calls = {}

    class _PixGT:
        analytic = True
        mask_type = "circulargradient"
        geometry = {"x": 1}

        def points(self, coords):
            calls["n"] = int(coords.shape[0])
            return torch.full((coords.shape[0],), 0.375)

    x = _Sample(pixgt=_PixGT())
    coords = torch.rand(1, 9, 2)
    got = prnd.gt_at_points(x, x.gt_pix, coords)
    assert calls["n"] == 9 and torch.allclose(got, torch.full((1, 9), 0.375))


def test_point_targets_are_bilinear_not_nearest_on_a_raster():
    """N6: the GT is a continuous alpha, so the point labels are interpolated."""
    x = _Sample()
    alpha = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    coords = torch.tensor([[[0.5, 0.5]]])
    got = prnd.gt_at_points(x, alpha, coords)
    assert 0.4 < float(got) < 0.6


# --------------------------------------------------------------------------- #
# 6. optimiser + schedule
# --------------------------------------------------------------------------- #
def test_optimizer_spec_is_the_reference_sgd_with_norm_exempt_decay():
    spec = prnd.optimizer_spec(types.SimpleNamespace())
    assert (spec.type, spec.lr, spec.momentum, spec.weight_decay) == \
        ("sgd", 0.01, 0.9, 1e-4)
    assert spec.nesterov is False
    assert spec.grouping == "norm_bias" and spec.norm_weight_decay == 0.0


def test_row5_adamw_falls_back_to_the_campaign_default():
    prnd.OPTIONS["optimizer"] = "adamw"
    assert prnd.optimizer_spec(types.SimpleNamespace()) is None
    assert prnd.scheduler_kwargs(types.SimpleNamespace(), 1200) is None


def test_schedule_at_1200_steps_is_the_literal_pre_registered_ladder():
    kw = prnd.scheduler_kwargs(types.SimpleNamespace(), 1200)
    assert kw["milestones"] == [738, 1015]
    assert kw["warmup_steps"] == 18
    assert kw["gamma"] == 0.1
    assert kw["warmup_factor"] == 1.0 / 1000.0


def test_the_ladder_is_carried_by_fraction_to_another_horizon():
    from q3vl.where.calibrate import scale_milestones

    kw = prnd.scheduler_kwargs(types.SimpleNamespace(), 600)
    assert kw["milestones"] == scale_milestones(prnd.MILESTONE_FRACS, 600)
    assert kw["warmup_steps"] == round(1000 / 65000 * 600)


def test_explicit_milestones_are_taken_literally():
    prnd.OPTIONS["milestones"] = "10,20"
    prnd.OPTIONS["warmup_iters"] = 3
    kw = prnd.scheduler_kwargs(types.SimpleNamespace(), 600)
    assert kw["milestones"] == [10, 20] and kw["warmup_steps"] == 3


def test_the_scheduler_kwargs_are_accepted_by_make_scheduler():
    from q3vl.where.calibrate import make_scheduler

    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(2))], lr=0.01)
    sch = make_scheduler(opt, 1200, 0.03, "warmup_multistep",
                         **prnd.scheduler_kwargs(types.SimpleNamespace(), 1200))
    lrs = []
    for _ in range(1200):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sch.step()
    assert lrs[0] == pytest.approx(0.01 / 1000, rel=1e-6)   # WARMUP_FACTOR
    assert lrs[100] == pytest.approx(0.01, rel=1e-6)
    assert lrs[800] == pytest.approx(0.001, rel=1e-6)       # gamma once
    assert lrs[1100] == pytest.approx(0.0001, rel=1e-6)     # gamma twice


def test_build_optimizer_keeps_the_norm_affine_parameters_out_of_decay():
    from q3vl.whereb.amort.trainer import AmortTrainConfig, build_optimizer

    h = prnd.PointRendHead(in_dim=32, text_dim=16, coarse_dim=32)
    opt = build_optimizer(h, AmortTrainConfig(arm="PRND"),
                          prnd.optimizer_spec(types.SimpleNamespace()))
    assert isinstance(opt, torch.optim.SGD)
    decayed, undecayed = opt.param_groups[0], opt.param_groups[1]
    assert decayed["weight_decay"] == 1e-4 and undecayed["weight_decay"] == 0.0
    gn_ids = {id(p) for p in h.coarse.norm.parameters()}
    assert {id(p) for p in undecayed["params"]} == gn_ids


# --------------------------------------------------------------------------- #
# 7. builder seam + the GT flag guard
# --------------------------------------------------------------------------- #
def test_builder_kwargs_ask_for_the_16x_pixel_grid_and_want_hi():
    args = types.SimpleNamespace(pixgt_source="maskhi", no_pixgt=False)
    kw = prnd.builder_kwargs(args)
    assert kw["want_hi"] is True
    assert kw["pixgt_size"](32, 48) == (512, 768)


@pytest.mark.parametrize("gt,src", [("maskhi", "maskhi"),
                                    ("cgt1024", "cgt1024"),
                                    ("analytic", "render")])
def test_prnd_gt_maps_onto_the_shared_pixgt_source(gt, src):
    assert prnd.GT_TO_PIXGT_SOURCE[gt] == src
    prnd.OPTIONS["gt"] = gt
    prnd.builder_kwargs(types.SimpleNamespace(pixgt_source=src, no_pixgt=False))


def test_a_gt_flag_that_disagrees_with_the_provider_refuses_to_start():
    prnd.OPTIONS["gt"] = "maskhi"
    with pytest.raises(SystemExit, match="Refusing to start"):
        prnd.builder_kwargs(types.SimpleNamespace(pixgt_source="render",
                                                  no_pixgt=False))


def test_no_pixgt_refuses_to_start():
    with pytest.raises(SystemExit, match="no-pixgt"):
        prnd.builder_kwargs(types.SimpleNamespace(pixgt_source="maskhi",
                                                  no_pixgt=True))


# --------------------------------------------------------------------------- #
# 8. the pre-registered criterion column + the publication assertions
# --------------------------------------------------------------------------- #
def _full_sample(is_fake=False, gh=3, gw=4):
    """An ``AmortSampleInputs`` with everything ``compute_micro_batch`` reads."""
    from q3vl.whereb.amort.data import AmortSampleInputs

    up = 4                       # 2 ** subdiv_steps of the toy head
    return AmortSampleInputs(
        sample_id="s1", feat=torch.randn(1, 8, gh, gw), sim=None, center=None,
        cond_h=torch.randn(1, 5, 16), cond_mask=torch.ones(1, 5, dtype=torch.bool),
        word_ids=torch.tensor([0]), word_offsets=torch.tensor([0]),
        phi_dir=torch.randn(gh * gw, 71), guide_hi=None,
        gt_low=torch.rand(gh, gw), gt_hi=torch.rand(up * gh, up * gw),
        gt_partner_low=None, grid_h=gh, grid_w=gw, is_fake=is_fake,
        family="radial", route_semantic=False, h_cond=torch.randn(1, 16),
        gt_pix=torch.rand(up * gh, up * gw), gt_pix_source="maskhi512")


def _row(**kw):
    base = dict(sample_id="s", uncovered=False, is_fake=False, family="radial",
                hard_iou=0.6, grid_boundary_f1=0.4, center_prior_hard_iou=0.5,
                center_prior_boundary_f1=0.3, center_prior_soft_iou=0.45,
                random_floor=0.25, prnd_pix_soft_iou=0.55,
                prnd_transition_abs_err=0.08, prnd_m_hi_source="subdivision",
                prnd_m_low_source="subdivision", prnd_gt_pix_source="maskhi512")
    base.update(kw)
    return base


def test_the_column_carries_every_mandatory_criterion():
    cols = prnd.criteria_columns([_row(), _row(hard_iou=0.7, family="band")])
    c = cols["prnd_point_readout"]
    assert c["n"] == 2
    for k in ("topk_iou", "grid_boundary_f1", "center_prior_topk_iou",
              "random_floor", "transition_band_abs_err",
              "paired_delta_vs_center_prior", "by_family"):
        assert k in c, k
    assert set(c["by_family"]) == {"radial", "band"}
    assert c["gt_pix_sources"] == {"maskhi512": 2}


def test_a_board_with_no_pixel_column_is_refused():
    from q3vl.whereb.amort.evaluate import assert_criteria_ran

    cols = prnd.criteria_columns([_row(prnd_pix_soft_iou=None,
                                       prnd_pix_error="no pixel field")])
    assert cols["prnd_point_readout"]["n"] == 0
    with pytest.raises(AssertionError, match="cannot adjudicate"):
        assert_criteria_ran({"criteria_columns": cols}, "PRND")
    # ... and a populated one passes
    ok = prnd.criteria_columns([_row()])
    rep = assert_criteria_ran({"criteria_columns": ok}, "PRND")
    assert rep["required"] == ["prnd_point_readout"]


def test_the_column_excludes_fake_and_uncovered_rows():
    cols = prnd.criteria_columns([_row(), _row(is_fake=True),
                                  _row(uncovered=True)])
    assert cols["prnd_point_readout"]["n"] == 1


def _board(headline=True, main="generated"):
    ctx = {"n": 10}
    if headline:
        ctx["headline_normal_only"] = {"n": 7, "topk_iou_median": 0.61}
    return {"main_context": main, "contexts": {main: ctx}}


def test_publication_needs_L_coarse_and_L_point_in_the_first_step_row():
    ok = {"step": 1, "L_coarse": 0.5, "L_point": 0.4}
    rep = prnd.assert_publishable(_board(), steps_row=ok)
    assert rep["steps_columns"]["L_coarse"] == 0.5
    with pytest.raises(AssertionError, match=r"L_point"):
        prnd.assert_publishable(_board(), steps_row={"step": 1, "L_coarse": 0.5})
    with pytest.raises(AssertionError, match="no first row is available"):
        prnd.assert_publishable(_board(), steps_row=None)
    # eval_only has no training steps of its own; recorded, not asserted
    rep = prnd.assert_publishable(_board(), steps_row=None, eval_only=True)
    assert rep["steps_columns"]["skipped"].startswith("eval_only")


def test_publication_needs_headline_normal_only():
    row = {"step": 1, "L_coarse": 0.5, "L_point": 0.4}
    with pytest.raises(AssertionError, match="headline_normal_only"):
        prnd.assert_publishable(_board(headline=False), steps_row=row)


# --------------------------------------------------------------------------- #
# regression, 2026-08-15: the loss ran and the assertion could not see it
#
# `EPR020_PRND_SMOKE` died at quick eval and again at the final board with
# "no steps.jsonl row was supplied" while row 1 of `steps.jsonl` carried both
# `L_coarse` and `L_point`.  The row was never looked up: `evaluate.evaluate_arm`
# (`evaluate.py:660`) calls `assert_criteria_ran(board, arm)` with no
# `steps_row`, and only `run_amort_arm._finish_board` passes one.  The fix looks
# the row up instead of reading "not supplied" as "not computed"; these tests
# pin BOTH halves -- it must find a real row, and it must still fail when the
# columns genuinely are not there.
# --------------------------------------------------------------------------- #
def _write_steps(path, *rows):
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_assert_publishable_reads_steps_jsonl_when_the_caller_passes_no_row(tmp_path):
    p = _write_steps(tmp_path / "steps.jsonl",
                     {"step": 1, "loss": 3.1, "L_coarse": 2.57, "L_point": 0.54},
                     {"step": 2, "loss": 2.6, "L_coarse": 1.19, "L_point": 0.47})
    rep = prnd.assert_publishable(_board(), steps_row=None, steps_path=p)
    # the FIRST row, not the last, and the report says where it came from
    assert rep["steps_columns"] == {"L_coarse": 2.57, "L_point": 0.54}
    assert "steps.jsonl" in rep["steps_columns_source"]


def test_a_disk_row_without_the_columns_still_refuses_to_publish(tmp_path):
    p = _write_steps(tmp_path / "steps.jsonl",
                     {"step": 1, "loss": 3.1, "L_something_else": 1.0})
    with pytest.raises(AssertionError,
                       match=r"columns present: \['L_something_else'\]"):
        prnd.assert_publishable(_board(), steps_row=None, steps_path=p)


def test_the_callers_row_wins_over_the_file(tmp_path):
    p = _write_steps(tmp_path / "steps.jsonl",
                     {"step": 1, "L_coarse": 2.57, "L_point": 0.54})
    rep = prnd.assert_publishable(
        _board(), steps_row={"step": 1, "L_coarse": 0.7, "L_point": 0.9},
        steps_path=p)
    assert rep["steps_columns"] == {"L_coarse": 0.7, "L_point": 0.9}
    assert rep["steps_columns_source"] == "caller"


def test_the_in_process_witness_is_the_last_resort(tmp_path):
    """A quick eval can land before the trainer flushed its buffer, so an empty
    (or absent) steps.jsonl is not proof that the loss never ran."""
    prnd.FIRST_LOSS_COLUMNS.update({"L_coarse": 2.57, "L_point": 0.54})
    rep = prnd.assert_publishable(_board(), steps_row=None,
                                  steps_path=tmp_path / "steps.jsonl")
    assert rep["steps_columns"] == {"L_coarse": 2.57, "L_point": 0.54}
    assert rep["steps_columns_source"] == "in-process first-micro-batch witness"


def test_the_witness_is_filled_by_the_first_micro_batch_and_never_overwritten():
    from q3vl.whereb.amort.losses import LossWeights
    from q3vl.whereb.amort.model import AmortModel
    from q3vl.whereb.amort.trainer import compute_micro_batch

    m = AmortModel("PRND", in_dim=8, cond_text_dim=16,
                   arm_kwargs={"coarse_dim": 32, "fc_dim": 12, "num_fc": 3,
                               "train_points": 16, "subdiv_steps": 2,
                               "subdiv_points": 16})
    m.train()
    assert not prnd.FIRST_LOSS_COLUMNS
    compute_micro_batch(m, [_full_sample()], LossWeights())
    assert {"L_coarse", "L_point"} <= set(prnd.FIRST_LOSS_COLUMNS)
    first = dict(prnd.FIRST_LOSS_COLUMNS)
    compute_micro_batch(m, [_full_sample()], LossWeights())
    assert prnd.FIRST_LOSS_COLUMNS == first          # first row, not the last


def test_no_row_anywhere_raises_its_own_error(tmp_path):
    """Caller, disk and witness all empty: a distinct failure from "the loss
    ran and its columns are wrong", and still a refusal."""
    with pytest.raises(AssertionError, match="no first row is available"):
        prnd.assert_publishable(_board(), steps_row=None,
                                steps_path=tmp_path / "steps.jsonl")


# --------------------------------------------------------------------------- #
# the second hazard: `headline_normal_only` exists only when the scored subset
# contained a `winner_confidence == "normal"` row (`evaluate.py:482-483`), and a
# quick eval scores `--quick-eval-limit` samples.
# --------------------------------------------------------------------------- #
def _quick_board(conf=("low",), n=32):
    """What `evaluate_arm(quick=True)` produces: three contexts, main = gt."""
    ctx = {"n": n, "strata": {"winner_confidence":
                              {c: {"n": n} for c in conf}}}
    return {"main_context": "gt",
            "contexts": {"gt": ctx, "shuffled": {"n": n}, "antonym": {"n": n}}}


def test_quick_eval_without_a_normal_sample_records_the_skip_instead_of_raising():
    row = {"step": 1, "L_coarse": 0.5, "L_point": 0.4}
    rep = prnd.assert_publishable(_quick_board(), steps_row=row)
    hn = rep["headline_normal_only"]
    assert "quick eval" in hn["skipped"]
    assert hn["winner_confidence_strata"] == {"low": 32}
    assert hn["contexts"] == ["antonym", "gt", "shuffled"]


def test_the_published_board_still_hard_requires_headline_normal_only():
    row = {"step": 1, "L_coarse": 0.5, "L_point": 0.4}
    assert prnd.is_publication_board(_board()) is True
    assert prnd.is_publication_board(_quick_board()) is False
    with pytest.raises(AssertionError, match="headline_normal_only"):
        prnd.assert_publishable(_board(headline=False), steps_row=row)


def test_an_interim_board_that_has_normal_rows_but_no_headline_still_raises():
    """The skip is licensed by "there were no normal samples", not by "this is a
    quick eval": a quick board WITH normal rows and no key is a disagreement."""
    row = {"step": 1, "L_coarse": 0.5, "L_point": 0.4}
    with pytest.raises(AssertionError, match="headline_normal_only"):
        prnd.assert_publishable(_quick_board(conf=("low", "normal")),
                                steps_row=row)


def test_the_installed_assertion_works_at_the_evaluate_arm_call_site(monkeypatch,
                                                                    tmp_path):
    """The regression end to end through the wrapper.

    `evaluate.evaluate_arm` calls `assert_criteria_ran(board, arm)` -- two
    positional arguments, no `steps_row` -- and writes `metrics.json` three
    lines later.  The installed assertion must find the run's `steps.jsonl`
    itself there, and must still refuse a run whose first row has no loss
    columns.
    """
    import q3vl.whereb.amort.evaluate as ev
    import q3vl.whereb.scripts.run_amort_arm as base
    import q3vl.whereb.scripts.run_prnd_arm as W
    from q3vl.whereb.amort import arms as A

    monkeypatch.setattr(base, "main", lambda argv: 0)
    monkeypatch.setattr(A, "ARM_SETUP", {})
    monkeypatch.setattr(ev, "assert_criteria_ran", ev.assert_criteria_ran)

    run = "amort_PRND_regression"
    W.main(["--out-root", str(tmp_path), "--run-name", run])
    steps = tmp_path / run / "steps.jsonl"

    board = {**_board(), "criteria_columns": prnd.criteria_columns([_row()])}
    # (a) before any row exists and with no loss call in this process: refused
    with pytest.raises(AssertionError, match="no first row is available"):
        ev.assert_criteria_ran(board, "PRND")

    # (b) the failing case the smoke run actually was in: the row is on disk
    _write_steps(steps,
                 {"step": 1, "loss": 3.12, "lr": 1e-4, "L_coarse": 2.568,
                  "L_point": 0.552, "n": 4},
                 {"step": 2, "loss": 2.60, "L_coarse": 1.199, "L_point": 0.471,
                  "n": 4})
    rep = ev.assert_criteria_ran(board, "PRND")             # no steps_row
    got = rep["prnd_publication"]["steps_columns"]
    assert got == {"L_coarse": 2.568, "L_point": 0.552}

    # (c) the assertion is not a bypass: a first row without the columns fails
    _write_steps(steps, {"step": 1, "loss": 3.12, "n": 4})
    prnd.FIRST_LOSS_COLUMNS.clear()
    with pytest.raises(AssertionError, match=r"columns present: \[\]"):
        ev.assert_criteria_ran(board, "PRND")

    # (d) a quick-eval board with no normal sample survives the same call site
    quick = {**_quick_board(),
             "criteria_columns": prnd.criteria_columns([_row()])}
    _write_steps(steps, {"step": 1, "L_coarse": 2.568, "L_point": 0.552})
    rep = ev.assert_criteria_ran(quick, "PRND")
    assert "quick eval" in rep["prnd_publication"]["headline_normal_only"]["skipped"]

    # (e) other arms are untouched by the swap
    assert ev.assert_criteria_ran({"criteria_columns": {}}, "P1")["required"] == []


def test_eval_only_reaches_the_assertion_from_argv_not_only_from_the_board(
        monkeypatch, tmp_path):
    """`--eval-only` re-scores a checkpoint, so the run has no steps.jsonl of its
    own.  `_finish_board` records that in the board, but the `evaluate_arm` call
    site runs first and has not built that key yet."""
    import q3vl.whereb.amort.evaluate as ev
    import q3vl.whereb.scripts.run_amort_arm as base
    import q3vl.whereb.scripts.run_prnd_arm as W
    from q3vl.whereb.amort import arms as A

    monkeypatch.setattr(base, "main", lambda argv: 0)
    monkeypatch.setattr(A, "ARM_SETUP", {})
    monkeypatch.setattr(ev, "assert_criteria_ran", ev.assert_criteria_ran)

    W.main(["--out-root", str(tmp_path), "--run-name", "amort_PRND_evalonly",
            "--eval-only"])
    board = {**_board(), "criteria_columns": prnd.criteria_columns([_row()])}
    rep = ev.assert_criteria_ran(board, "PRND")
    assert rep["prnd_publication"]["steps_columns"]["skipped"].startswith(
        "eval_only")


def test_per_sample_row_reports_the_pixel_column_and_never_raises():
    h = _head(subdiv_steps=2, subdiv_points=64)
    x = _Sample()
    m = _Model(h, training=False)
    out = prnd.forward(m, h, _ctx(x))
    row = prnd.per_sample_row(m, out, x)
    assert row["prnd_pix_soft_iou"] is not None
    assert 0.0 <= row["prnd_pix_soft_iou"] <= 1.0
    assert row["prnd_m_hi_source"] == "subdivision"
    # a broken sample must land in a column, not in a traceback
    broken = prnd.per_sample_row(m, out, object())
    assert broken["prnd_pix_error"]


def test_per_sample_row_says_so_when_there_is_no_pixel_field():
    h = _head()
    x = _Sample()
    m = _Model(h, training=True)
    row = prnd.per_sample_row(m, prnd.forward(m, h, _ctx(x)), x)
    assert row["prnd_pix_soft_iou"] is None and row["prnd_pix_error"]


# --------------------------------------------------------------------------- #
# 9. records + "off = nothing changes"
# --------------------------------------------------------------------------- #
def test_the_loss_preregistration_record_names_dice_honestly():
    rec = prnd.loss_form()
    assert rec["dice_in_loss"] is False
    assert rec["form"] == ("L = 1.0*BCE(up16(coarse), alpha_hi) + "
                           "1.0*BCE(points, PS(alpha_hi))")
    assert rec["iou_as_field_target"] is False
    prnd.OPTIONS["point_loss"] = "m2f"
    rec = prnd.loss_form()
    assert rec["dice_in_loss"] is True and "dice" in rec["form"]


def test_setup_record_carries_every_flag_and_is_json_serialisable():
    import json

    rec = prnd.setup_record()["prnd"]
    assert set(rec["flags"]) == set(prnd.DEFAULTS)
    assert rec["pixgt_source"] == "maskhi"
    assert rec["steps_jsonl_columns"]["losses"] == ["L_coarse", "L_point"]
    json.dumps(rec, default=str)


def test_configure_records_only_the_flags_that_were_passed():
    ns = types.SimpleNamespace(**{f"prnd_{k}": None for k in prnd.DEFAULTS})
    ns.prnd_train_points = 2048
    o = prnd.configure(ns)
    assert prnd.OPTIONS == {"train_points": 2048}
    assert o["train_points"] == 2048 and o["oversample"] == 3.0


def test_the_live_arms_are_untouched_by_this_module():
    from q3vl.whereb.amort.model import ARMS, AmortModel

    assert ARMS == ("P1", "P3prime", "SHAPE3", "UNIQ")
    m = AmortModel("P1", in_dim=8, ch=8, n_blocks=1, cond_text_dim=16)
    assert m.is_new_arm is False and m.sem is not None


def test_the_model_builds_this_arm_under_the_b4_defaults():
    from q3vl.whereb.amort.model import AmortModel

    m = AmortModel("PRND", in_dim=8, cond_text_dim=16,
                   arm_kwargs={"coarse_dim": 32, "fc_dim": 12,
                               "subdiv_steps": 2, "subdiv_points": 16})
    assert m.is_new_arm and m.new_arm_defaults
    assert m.sem is None and m.cond_frozen
    assert not m.use_sim_field and not m.use_film
    assert m.facts()["arm_head"]["head"] == "PointRendHead"
    out = m.forward_geo(torch.randn(1, 8, 3, 4), None, None, grid_h=3, grid_w=4,
                        h_cond=torch.randn(1, 16))
    assert out["m_low"].shape == (3, 4)


def test_compute_micro_batch_runs_the_arm_loss_end_to_end():
    """The real trainer seam: AmortModel -> forward_geo -> the arm's loss ->
    aggregate, with the shared early-warning columns intact."""
    from q3vl.whereb.amort.losses import LossWeights
    from q3vl.whereb.amort.model import AmortModel
    from q3vl.whereb.amort.trainer import compute_micro_batch

    m = AmortModel("PRND", in_dim=8, cond_text_dim=16,
                   arm_kwargs={"coarse_dim": 32, "fc_dim": 12, "num_fc": 3,
                               "train_points": 16, "subdiv_steps": 2,
                               "subdiv_points": 16})
    m.train()
    xs = [_full_sample(), _full_sample(is_fake=True)]
    total, stats, rows = compute_micro_batch(m, xs, LossWeights())
    assert total.requires_grad and torch.isfinite(total)
    assert "L_coarse" in stats and "L_point" in stats
    assert "area_ratio_median" in stats and "std_ratio_median" in stats
    assert stats["n"] == 2 and stats["n_fake"] == 1
    # the seven-term stack did NOT run
    assert not any(k in stats for k in ("L_bce", "L_sdf", "L_area", "L_sep"))
    # the arm's own per-step diagnostics reached the row via train_stats
    assert rows[0]["prnd_point_accuracy"] is not None
    assert rows[0]["gt_pix_source"] == "maskhi512"
    total.backward()
    assert any(p.grad is not None for p in m.geo.parameters())


def test_the_publication_assertions_are_installed_in_front_of_the_shared_one():
    import q3vl.whereb.amort.evaluate as ev
    import q3vl.whereb.scripts.run_prnd_arm as W

    original = ev.assert_criteria_ran
    assert W.install_publication_assert() is True
    assert W.install_publication_assert() is False        # idempotent
    assert ev.assert_criteria_ran is not original
    assert getattr(ev.assert_criteria_ran, "_prnd_base") is original
    cols = prnd.criteria_columns([_row()])
    board = {**_board(), "criteria_columns": cols}
    rep = ev.assert_criteria_ran(
        board, "PRND", steps_row={"step": 1, "L_coarse": 0.5, "L_point": 0.4})
    assert rep["prnd_publication"]["headline_normal_only"]["n"] == 7
    # ... and it really refuses a board missing the loss columns
    with pytest.raises(AssertionError, match=r"L_point"):
        ev.assert_criteria_ran(board, "PRND", steps_row={"step": 1})
    # every other arm is untouched
    assert ev.assert_criteria_ran({"criteria_columns": {}}, "P1")["required"] == []


def test_the_assertions_are_installed_on_every_launch_path_not_just_the_wrapper():
    """``builder_kwargs`` runs on any launch path, before a single step; a guard
    that only fires under one entry script is not a runtime guard."""
    import q3vl.whereb.amort.evaluate as ev

    original = ev.assert_criteria_ran
    prnd.builder_kwargs(types.SimpleNamespace(pixgt_source="maskhi",
                                              no_pixgt=False))
    assert ev.assert_criteria_ran is not original
    with pytest.raises(AssertionError, match="no first row is available"):
        ev.assert_criteria_ran(
            {**_board(), "criteria_columns": prnd.criteria_columns([_row()])},
            "PRND")


def test_the_wrapper_pins_the_shared_flags_the_recipe_implies():
    import q3vl.whereb.scripts.run_prnd_arm as W

    pinned: dict = {}
    rest = W._pin(["--run-name", "x"], "--want-hi", None, pinned, why="w")
    assert rest[-1] == "--want-hi" and pinned["--want-hi"]["pinned"]
    rest2 = W._pin(["--want-hi"], "--want-hi", None, pinned, why="w")
    assert rest2 == ["--want-hi"] and pinned["--want-hi"]["pinned"] is False


def test_the_wrapper_composes_the_argv_it_promises(monkeypatch, tmp_path):
    """End to end over the wrapper's own job: parse --prnd-*, pin the shared
    flags the recipe implies, write the pre-registration records, delegate the
    rest verbatim."""
    import q3vl.whereb.amort.evaluate as ev
    import q3vl.whereb.scripts.run_amort_arm as base
    import q3vl.whereb.scripts.run_prnd_arm as W
    from q3vl.whereb.amort import arms as A

    seen: dict = {}

    def _fake_base(argv):
        seen["argv"] = list(argv)
        return 0

    monkeypatch.setattr(base, "main", _fake_base)
    monkeypatch.setattr(ev, "assert_criteria_ran", ev.assert_criteria_ran)
    monkeypatch.setattr(A, "ARM_SETUP", {})

    rc = W.main(["--run-name", "amort_PRND_t", "--out-root", str(tmp_path),
                 "--max-steps", "1200", "--cond-readout", "seg_where",
                 "--prnd-gt", "cgt1024", "--prnd-train-points", "2048"])
    assert rc == 0
    argv = seen["argv"]
    assert argv[:2] == ["--arm", "PRND"]
    assert "--want-hi" in argv
    assert argv[argv.index("--pixgt-source") + 1] == "cgt1024"
    assert argv[argv.index("--max-grad-norm") + 1] == "0"
    assert argv[argv.index("--scheduler") + 1] == "warmup_multistep"
    # the caller's own flags survive untouched
    assert argv[argv.index("--cond-readout") + 1] == "seg_where"
    assert argv[argv.index("--run-name") + 1] == "amort_PRND_t"

    import json

    cfg = tmp_path / "amort_PRND_t" / "config"
    rec = json.loads((cfg / "prnd_setup.json").read_text())
    assert rec["flags"]["train_points"] == 2048 and rec["flags"]["gt"] == "cgt1024"
    assert rec["schedule_at_this_horizon"]["milestones"] == [738, 1015]
    assert rec["pinned_shared_flags"]["--want-hi"]["pinned"] is True
    assert len(rec["prnd_sha256"]) == 64 and len(rec["wrapper_sha256"]) == 64
    pre = json.loads((cfg / "loss_preregistration_prnd.json").read_text())
    assert pre["dice_in_loss"] is False and pre["arm"] == "PRND"
    assert A.ARM_SETUP["prnd"]["pixgt_source"] == "cgt1024"


def test_the_wrapper_does_not_override_a_flag_the_caller_passed(monkeypatch,
                                                                tmp_path):
    import q3vl.whereb.scripts.run_amort_arm as base
    import q3vl.whereb.scripts.run_prnd_arm as W
    from q3vl.whereb.amort import arms as A

    seen: dict = {}

    def _fake_base(argv):
        seen["argv"] = list(argv)
        return 0

    monkeypatch.setattr(base, "main", _fake_base)
    monkeypatch.setattr(A, "ARM_SETUP", {})
    W.main(["--max-grad-norm", "1.0", "--scheduler", "cosine",
            "--out-root", str(tmp_path)])
    argv = seen["argv"]
    assert argv.count("--max-grad-norm") == 1
    assert argv[argv.index("--max-grad-norm") + 1] == "1.0"
    assert argv[argv.index("--scheduler") + 1] == "cosine"
    assert A.ARM_SETUP["prnd"]["pinned_shared_flags"]["--scheduler"]["pinned"] is False


def test_the_wrapper_exposes_every_flag_of_the_proposals_entry_row():
    import argparse

    ap = argparse.ArgumentParser(add_help=False)
    prnd.add_arguments(ap)
    got = {a for act in ap._actions for a in act.option_strings}
    for flag in ("--prnd-train-points", "--prnd-oversample", "--prnd-importance",
                 "--prnd-fc-dim", "--prnd-num-fc", "--prnd-coarse-dim",
                 "--prnd-coarse-pred-each-layer", "--prnd-subdiv-steps",
                 "--prnd-subdiv-points", "--prnd-point-loss", "--prnd-gt",
                 "--prnd-no-subdivision", "--prnd-optimizer", "--prnd-lr",
                 "--prnd-momentum", "--prnd-wd", "--prnd-warmup-iters",
                 "--prnd-milestones", "--prnd-gamma"):
        assert flag in got, flag
