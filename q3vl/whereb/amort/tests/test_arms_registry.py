"""The EPR-018..023 arm registry, and that it is WIRED, not just declared.

CPU only.  A stand-in arm module is installed in ``sys.modules`` so the whole
seam -- ``AmortModel.__init__`` -> ``forward_geo`` -> ``compute_micro_batch`` ->
``assert_criteria_ran`` -- is exercised without loading a 4B base.  Four things
are pinned:

* the six real modules satisfy the contract (import, ``ARM`` / ``CRITERIA``
  agree with the registry, the three required hooks are callable);
* a registered-but-absent module still fails with instructions rather than
  ``ModuleNotFoundError``;
* the pre-registered criterion column really refuses a board that carries 0
  values for it ("defined, measured, not wired" has cost this campaign three
  times and the new arms do not get a fourth);
* the B-4 口径 (all four families through the new head, SemanticHead not built,
  CondEncoder frozen) is what the model actually constructs.
"""

from __future__ import annotations

import inspect
import sys
import types

import pytest
import torch
import torch.nn as nn

from q3vl.whereb.amort import arms as A

FAKE = "FAKEARM"
FAKE_MOD = "q3vl.whereb.amort._fakearm_for_tests"
FAKE_COL = "fake_readout"


class _Head(nn.Module):
    def __init__(self, in_dim=1024, text_dim=2560, scale=1.0):
        super().__init__()
        self.proj = nn.Linear(text_dim, in_dim)
        self.scale = float(scale)

    def facts(self):
        return {"kind": "fake", "scale": self.scale}


def _make_module(**hooks):
    mod = types.ModuleType(FAKE_MOD)
    mod.ARM = FAKE
    mod.CRITERIA = (FAKE_COL,)

    def build_head(*, in_dim, text_dim, args=None, **kw):
        return _Head(in_dim, text_dim, **kw)

    def forward(model, head, ctx):
        h = ctx.require_vector(FAKE)                      # (2560,)
        w = head.proj(h)                                  # (1024,)
        s = torch.einsum("c,bchw->bhw", w, ctx.feat)[0]
        return {"m_low": torch.sigmoid(s), "fake": {"logit": s}}

    def compute_loss(model, out, x, weights):
        from q3vl.whereb.amort.losses import AmortLoss

        t = torch.nn.functional.mse_loss(out["m_low"], x.gt_low)
        return AmortLoss(total=t, terms={"fake": t})

    mod.build_head, mod.forward, mod.compute_loss = build_head, forward, compute_loss
    for k, v in hooks.items():
        setattr(mod, k, v)
    return mod


@pytest.fixture()
def fake_arm(monkeypatch):
    def install(**hooks):
        mod = _make_module(**hooks)
        monkeypatch.setitem(sys.modules, FAKE_MOD, mod)
        monkeypatch.setitem(A.ARM_MODULES, FAKE, FAKE_MOD)
        monkeypatch.setitem(A.ARM_CRITERIA, FAKE, (FAKE_COL,))
        A._CACHE.pop(FAKE, None)
        yield_mod = mod
        return yield_mod

    yield install
    A._CACHE.pop(FAKE, None)


# --- the registry -----------------------------------------------------------

def test_the_six_names_and_their_columns_are_the_proposals():
    assert A.NEW_ARMS == ("SEGSAM", "SAMDEC", "PRND", "LIIF", "MATTE", "CONDINST")
    assert A.ARM_CRITERIA == {
        "SEGSAM": ("segsam_fine",),
        "SAMDEC": ("samdec_cand",),
        "PRND": ("prnd_point_readout",),
        "LIIF": ("liif_grid_decode",),
        "MATTE": ("pix_readout",),
        "CONDINST": ("condinst_pix_readout",),
    }
    assert set(A.ARM_MODULES) == set(A.NEW_ARMS)
    assert all(A.is_new_arm(a) for a in A.NEW_ARMS)
    assert not A.is_new_arm("UNIQ")


@pytest.mark.parametrize("arm", A.NEW_ARMS)
def test_the_six_arm_modules_honour_the_registry_contract(arm):
    """The six heads exist now, so the contract is asserted forwards.

    (Until 2026-08-14 this test asserted the scaffold instead -- "the module
    does NOT import yet".  That assertion inverts the moment the arm is
    written, so it is replaced by the one the registry actually needs to
    hold every day: the module imports, its name and its pre-registered
    column agree with the table `assert_criteria_ran` reads, and the three
    required hooks are callable with the documented signature.)
    """
    mod = A.load_arm(arm)
    assert mod.__name__ == A.ARM_MODULES[arm]
    assert mod.ARM == arm
    assert tuple(mod.CRITERIA) == A.ARM_CRITERIA[arm]
    for hook in A.REQUIRED_HOOKS:
        assert callable(getattr(mod, hook, None)), f"{arm}: {hook}"
        assert A.arm_hook(arm, hook) is not None
    # `build_head` is called by AmortModel.__init__ as
    # build_head(in_dim=..., text_dim=..., args=..., **kw)
    params = inspect.signature(mod.build_head).parameters
    for kw in ("in_dim", "text_dim", "args"):
        assert kw in params, f"{arm}.build_head has no {kw!r}"
    assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()), \
        f"{arm}.build_head must accept **kw (ARM_KWARGS)"
    # nothing exported under a name the contract does not know: a hook spelled
    # `criteria_colums` is a silent no-op, which is what `arm_hook` refuses
    for name in A.OPTIONAL_HOOKS:
        fn = getattr(mod, name, None)
        if fn is not None:
            assert callable(fn), f"{arm}.{name} is not callable"
            assert A.arm_hook(arm, name) is fn


def test_an_unwritten_arm_fails_with_the_contract_not_a_traceback(monkeypatch):
    """The error path the six modules used to exercise, kept alive.

    A registered-but-absent module must say "write this file, with these
    three functions" instead of raising ``ModuleNotFoundError``.
    """
    ghost = "q3vl.whereb.amort._never_written"
    monkeypatch.setitem(A.ARM_MODULES, FAKE, ghost)
    monkeypatch.setitem(A.ARM_CRITERIA, FAKE, ("ghost_col",))
    A._CACHE.pop(FAKE, None)
    with pytest.raises(ImportError) as ei:
        A.load_arm(FAKE)
    A._CACHE.pop(FAKE, None)
    msg = str(ei.value)
    assert ghost.replace(".", "/") + ".py" in msg
    for hook in A.REQUIRED_HOOKS:
        assert hook in msg
    assert repr(("ghost_col",)) in msg


def test_missing_required_hook_is_named(fake_arm, monkeypatch):
    mod = fake_arm()
    monkeypatch.delattr(mod, "compute_loss")
    A._CACHE.pop(FAKE, None)
    with pytest.raises(ImportError, match=r"missing the required hook\(s\) \['compute_loss'\]"):
        A.load_arm(FAKE)


def test_an_arm_may_not_rename_its_own_criterion(fake_arm, monkeypatch):
    mod = fake_arm()
    monkeypatch.setattr(mod, "CRITERIA", ("something_else",))
    A._CACHE.pop(FAKE, None)
    with pytest.raises(ImportError, match="may not rename"):
        A.load_arm(FAKE)


def test_arm_hook_rejects_a_typo(fake_arm):
    fake_arm()
    assert A.arm_hook(FAKE, "criteria_columns") is None
    with pytest.raises(KeyError, match="not part of the arm contract"):
        A.arm_hook(FAKE, "criteria_colums")


def test_context_says_which_flag_is_missing():
    ctx = A.ArmContext(feat=torch.zeros(1, 4, 2, 3), grid_h=2, grid_w=3)
    with pytest.raises(ValueError, match="ReadoutBuilder"):
        ctx.require_cond("X")
    ctx.h_cond = torch.zeros(4, 2560)
    with pytest.raises(ValueError, match="one condition vector"):
        ctx.require_vector("X")


# --- the model seam ---------------------------------------------------------

def _model(fake_arm, **kw):
    from q3vl.whereb.amort.model import AmortModel

    fake_arm()
    return AmortModel(FAKE, in_dim=8, cond_text_dim=16, **kw)


def test_b4_defaults_no_semantic_head_frozen_cond_no_sim_no_film(fake_arm):
    m = _model(fake_arm)
    assert m.is_new_arm and m.new_arm_defaults
    assert m.sem is None                       # SemanticHead not constructed
    assert m.cond_frozen
    assert not any(p.requires_grad for p in m.cond.parameters())
    assert not m.use_sim_field and not m.use_film
    f = m.facts()
    assert f["cond_encoder_unused"] and f["has_semantic_head"] is False
    assert f["arm_head"]["kind"] == "fake"


def test_legacy_routing_switch_restores_the_semantic_head(fake_arm):
    m = _model(fake_arm, new_arm_defaults=False)
    assert m.new_arm_defaults is False
    assert m.sem is not None
    assert all(p.requires_grad for p in m.cond.parameters())


def test_head_kwargs_reach_build_head(fake_arm):
    m = _model(fake_arm, arm_kwargs={"scale": 2.5})
    assert m.geo.scale == 2.5


def test_head_kwargs_from_args_hook(fake_arm):
    fake_arm(head_kwargs_from_args=lambda args: {"scale": 3.5})
    from q3vl.whereb.amort.model import AmortModel

    m = AmortModel(FAKE, in_dim=8, cond_text_dim=16, arm_args=object())
    assert m.geo.scale == 3.5


def test_forward_geo_routes_to_the_arm_module(fake_arm):
    m = _model(fake_arm)
    out = m.forward_geo(torch.randn(1, 8, 3, 4), None, None, grid_h=3, grid_w=4,
                        h_cond=torch.randn(1, 16))
    assert out["m_low"].shape == (3, 4)
    assert "fake" in out


def test_forward_geo_refuses_a_head_that_returns_no_m_low(fake_arm, monkeypatch):
    from q3vl.whereb.amort.model import AmortModel

    mod = fake_arm()
    m = AmortModel(FAKE, in_dim=8, cond_text_dim=16)
    monkeypatch.setattr(mod, "forward", lambda model, head, ctx: {"nope": 1})
    with pytest.raises(AssertionError, match="returned no 'm_low'"):
        m.forward_geo(torch.randn(1, 8, 3, 4), None, None, grid_h=3, grid_w=4,
                      h_cond=torch.randn(1, 16))


def test_live_arms_are_untouched_by_the_registry():
    from q3vl.whereb.amort.model import ARMS, AmortModel

    assert ARMS == ("P1", "P3prime", "SHAPE3", "UNIQ")
    m = AmortModel("P1", in_dim=8, ch=8, n_blocks=1, cond_text_dim=16)
    assert m.is_new_arm is False and m.new_arm_defaults is False
    assert m.sem is not None and not m.cond_frozen
    assert m.use_sim_field and m.use_film


def test_unknown_arm_still_raises():
    from q3vl.whereb.amort.model import AmortModel

    with pytest.raises(ValueError, match="unknown arm"):
        AmortModel("NOPE")


# --- the trainer seam -------------------------------------------------------

class _X:
    def __init__(self, gh=3, gw=4, dim=8, tdim=16):
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
        self.is_fake = False
        self.family = "radial"
        self.route_semantic = False
        self.h_cond = torch.randn(1, tdim)
        self.gt_pix_source = "render"
        self.meta = {}


def test_compute_micro_batch_uses_the_arm_loss_and_keeps_the_warning_columns(fake_arm):
    from q3vl.whereb.amort.losses import LossWeights
    from q3vl.whereb.amort.trainer import compute_micro_batch

    m = _model(fake_arm)
    total, stats, rows = compute_micro_batch(m, [_X(), _X()], LossWeights())
    assert total.requires_grad
    assert "L_fake" in stats                       # the arm's own term
    # the pre-registered stop-and-check triggers, computed by the trainer
    assert "area_ratio_median" in stats and "std_ratio_median" in stats
    assert stats["n"] == 2 and stats["n_fake"] == 0
    assert rows[0]["gt_pix_source"] == "render"
    # the seven-term stack did NOT run
    assert not any(k.startswith("L_bce") or k.startswith("L_sdf") for k in stats)


def test_train_stats_hook_adds_columns(fake_arm):
    from q3vl.whereb.amort.losses import LossWeights
    from q3vl.whereb.amort.trainer import compute_micro_batch

    fake_arm(train_stats=lambda out, x: {"sup_cells": int(out["m_low"].numel())})
    from q3vl.whereb.amort.model import AmortModel

    m = AmortModel(FAKE, in_dim=8, cond_text_dim=16)
    _t, _s, rows = compute_micro_batch(m, [_X()], LossWeights())
    assert rows[0]["sup_cells"] == 12


def test_a_head_without_h_cond_fails_loudly(fake_arm):
    from q3vl.whereb.amort.losses import LossWeights
    from q3vl.whereb.amort.trainer import compute_micro_batch

    m = _model(fake_arm)
    x = _X()
    x.h_cond = None
    with pytest.raises(ValueError, match="ReadoutBuilder"):
        compute_micro_batch(m, [x], LossWeights())


# --- the criterion assertion ------------------------------------------------

def test_required_table_carries_all_six_arms():
    import inspect

    from q3vl.whereb.amort.evaluate import assert_criteria_ran

    src = inspect.getsource(assert_criteria_ran)
    assert "ARM_CRITERIA" in src, "the table must be built from arms.ARM_CRITERIA"
    for arm, cols in A.ARM_CRITERIA.items():
        with pytest.raises(AssertionError, match=cols[0]):
            assert_criteria_ran({"criteria_columns": {}}, arm)


@pytest.mark.parametrize("arm,col", [(a, c[0]) for a, c in A.ARM_CRITERIA.items()])
def test_n_zero_refuses_the_board(arm, col):
    from q3vl.whereb.amort.evaluate import assert_criteria_ran

    with pytest.raises(AssertionError, match="cannot adjudicate"):
        assert_criteria_ran({"criteria_columns": {col: {"n": 0}}}, arm)
    rep = assert_criteria_ran({"criteria_columns": {col: {"n": 7}}}, arm)
    assert rep["required"] == [col] and rep["computed"][col] == 7


def test_live_arms_required_table_is_unchanged():
    from q3vl.whereb.amort.evaluate import assert_criteria_ran

    assert assert_criteria_ran({"criteria_columns": {}}, "P1")["required"] == []
    with pytest.raises(AssertionError, match="shape_residual"):
        assert_criteria_ran({"criteria_columns": {}}, "SHAPE3")
    with pytest.raises(AssertionError, match="uniq_best"):
        assert_criteria_ran({"criteria_columns": {}}, "UNIQ")
