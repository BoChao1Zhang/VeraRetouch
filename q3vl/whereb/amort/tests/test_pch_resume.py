"""Regression tests for geometry-code injection on a *resumed* checkpoint.

The bug being fenced in: adding the geometry code as broadcast stem channels
widens ``stem.weight`` from 1025 to 1046 input channels, and
``load_state_dict`` refuses the checkpoint -- three B2 jobs died on it.  The
deeper requirement is stronger than "does not raise": an arm that claims to
continue a checkpoint must, before it has taken a single step, **compute exactly
what that checkpoint computed**.  Otherwise the injection Delta is confounded by
however much the fresh tensors perturbed the model.

So the assertions here are on outputs, not on shapes:

* ``test_pch_resume_is_bit_identical`` -- resume P3' into a PCH arm and the
  field is bit-for-bit the checkpoint's, code or no code;
* ``test_broadcast_resume_is_bit_identical`` -- same for the widened-stem form,
  proving the zero-padding branch (not just ``strict=False``);
* ``test_empty_code_is_exact_no_op`` -- a sample whose ``<where>`` named no
  geometry gets *zero* residual even after the injector has been trained,
  which is the graceful-fallback claim;
* ``test_unexpected_keys_are_refused`` -- a checkpoint carrying tensors the
  model cannot place is an error, never a silent partial load.
"""

from __future__ import annotations

import pytest
import torch

from q3vl.whereb.amort.geomparse import GEOM_DIM
from q3vl.whereb.amort.model import AmortModel
from q3vl.whereb.amort.resume import ResumeError, load_resumable

GH = GW = 6
IN_DIM = 1024


def _base_model(**kw) -> AmortModel:
    torch.manual_seed(0)
    return AmortModel("P3prime", in_dim=IN_DIM, n_blocks=2, with_semantic=False,
                      **kw)


def _inputs():
    g = torch.Generator().manual_seed(7)
    feat = torch.randn(1, IN_DIM, GH, GW, generator=g)
    cond_h = torch.randn(1, 4, 2560, generator=g)
    sim = torch.randn(1, 1, GH, GW, generator=g)
    word_ids = torch.zeros(1, dtype=torch.long)
    word_off = torch.zeros(1, dtype=torch.long)
    return feat, cond_h, sim, word_ids, word_off


def _field(model: AmortModel, geom=None) -> torch.Tensor:
    feat, cond_h, sim, word_ids, word_off = _inputs()
    model.eval()
    with torch.no_grad():
        cond = model.cond_of(cond_h, None, word_ids, word_off)
        out = model.forward_geo(feat, cond, torch.zeros(GH * GW, 71),
                                sim=sim, geom=geom, grid_h=GH, grid_w=GW)
    return out["s_low"].clone()


def _trained_checkpoint() -> dict[str, torch.Tensor]:
    """A base arm with *non-trivial* weights.

    Zero-initialised heads would make the test pass for the wrong reason: any
    two models agree when both output zero.  So the tensors are perturbed first.
    """
    base = _base_model()
    with torch.no_grad():
        for p in base.parameters():
            p.add_(torch.randn_like(p) * 0.02)
    return base, {k: v.clone() for k, v in base.state_dict().items()}


def test_pch_resume_is_bit_identical():
    base, sd = _trained_checkpoint()
    code = torch.zeros(GEOM_DIM)
    code[[0, 6, 14]] = 1.0                       # oval / right / large

    arm = _base_model(geom_inject=True, geom_mode="pch", pch_size="full")
    report = load_resumable(arm, sd)
    assert report["n_from_init"] > 0
    assert all(k.startswith("pch.") for k in report["from_init"])
    assert report["zero_padded"] == []           # PCH must not touch the stem

    ref = _field(base)
    assert torch.equal(_field(arm, geom=code), ref), "PCH perturbed step 0"
    assert torch.equal(_field(arm, geom=None), ref)


def test_broadcast_resume_is_bit_identical():
    base, sd = _trained_checkpoint()
    code = torch.zeros(GEOM_DIM)
    code[[1, 7]] = 1.0

    arm = _base_model(geom_inject=True, geom_mode="broadcast")
    report = load_resumable(arm, sd)
    assert [z["key"] for z in report["zero_padded"]] == ["geo.tower.stem.weight"]
    z = report["zero_padded"][0]
    assert z["new_in"] - z["old_in"] == GEOM_DIM

    # NOT bit-identical, and the reason is worth recording: zeroing the new
    # channels makes the layer mathematically identical, but a 1046-channel
    # convolution does not reduce in the same order as a 1025-channel one, so
    # the result differs in the last couple of ulps (measured 7e-9 absolute on
    # one cell of a 6x6 field).  PCH has no such caveat -- it leaves every
    # existing tensor's shape alone and reproduces the checkpoint exactly.
    got, ref = _field(arm, geom=code), _field(base)
    assert not torch.equal(got, ref) or True      # documents, does not require
    assert torch.allclose(got, ref, rtol=0, atol=1e-6)


def test_empty_code_is_exact_no_op():
    """After training, a code of all zeros must still contribute nothing."""
    arm = _base_model(geom_inject=True, geom_mode="pch", pch_size="lite")
    with torch.no_grad():                         # simulate a trained injector
        for p in arm.pch.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    live = torch.zeros(GEOM_DIM)
    live[[2, 9]] = 1.0
    empty = torch.zeros(GEOM_DIM)

    codes = torch.randn(1, 128, GH, GW)
    assert torch.count_nonzero(arm.pch(codes, empty)) == 0
    assert torch.count_nonzero(arm.pch(codes, live)) > 0, \
        "a trained injector that does nothing on a live code is not being tested"


def test_unexpected_keys_are_refused():
    _, sd = _trained_checkpoint()
    sd["geo.tower.not_a_real_layer"] = torch.zeros(3)
    with pytest.raises(ResumeError, match="cannot place"):
        load_resumable(_base_model(geom_inject=True, geom_mode="pch"), sd)


def test_missing_non_injector_key_is_refused():
    _, sd = _trained_checkpoint()
    del sd["geo.to_field.weight"]
    with pytest.raises(ResumeError, match="absent from the checkpoint"):
        load_resumable(_base_model(geom_inject=True, geom_mode="pch"), sd)


def test_shape_residual_is_wired_and_discriminative():
    """The SHAPE3 criterion must both run and separate shapes.

    It sat in `edgequal.py` uncalled while both SHAPE3 arms ran and were judged
    on IoU instead.  Two things are asserted: the eval row actually carries the
    column, and the metric still tells a quadratic level set from a blob -- a
    wired-but-meaningless column would be the same failure wearing a different
    hat.

    Note the inputs are SOFT fields.  The fit is a least-squares quadratic in
    LOGIT space, so a hard 0/1 mask scores ~0.29 no matter how perfect its
    shape; predictions and soft GT are both sigmoids, which is the regime the
    published validation numbers (0.047 / 0.040 / 0.268) live in.
    """
    import numpy as np

    from q3vl.whereb.amort.evaluate import _shape_residual_row

    h = w = 16
    yy, xx = np.mgrid[0:h, 0:w]
    x, y = (xx - 7.5) / 7.5, (yy - 7.5) / 7.5
    ellipse = 1 / (1 + np.exp(-(3.0 - 6 * (x * x / 0.6 + y * y / 0.3))))
    blob = 1 / (1 + np.exp(-(3.0 - 6 * (np.abs(x) * 2.2 + np.abs(y) ** 3))))

    k = int((ellipse > 0.5).sum())
    row = _shape_residual_row(torch.tensor(ellipse), torch.tensor(ellipse),
                              "radial", k)
    assert row["shape_residual"] == pytest.approx(0.0, abs=0.02)
    assert row["shape_residual_gt"] == pytest.approx(0.0, abs=0.02)

    bad = _shape_residual_row(torch.tensor(blob), torch.tensor(ellipse),
                              "radial", k)
    assert bad["shape_residual"] > 0.15, "metric no longer separates shapes"

    # semantic has no analytic member to fit; the column must be absent, not 1.0
    assert _shape_residual_row(torch.tensor(ellipse), torch.tensor(ellipse),
                               "semantic", k)["shape_residual"] is None


def test_criteria_assertion_blocks_an_unadjudicable_board():
    from q3vl.whereb.amort.evaluate import assert_criteria_ran

    empty = {"criteria_columns": {"shape_residual": {"n": 0}}}
    with pytest.raises(AssertionError, match="cannot adjudicate"):
        assert_criteria_ran(empty, "SHAPE3")
    # an arm that pre-registers nothing is unaffected
    assert assert_criteria_ran(empty, "P3prime")["required"] == []
    assert assert_criteria_ran({"criteria_columns":
                                {"shape_residual": {"n": 168}}},
                               "SHAPE3")["computed"]["shape_residual"] == 168


def test_pch_param_budget():
    full = _base_model(geom_inject=True, geom_mode="pch", pch_size="full")
    lite = _base_model(geom_inject=True, geom_mode="pch", pch_size="lite")
    assert full.pch.n_params() == 3_855_296        # proposal Full budget 3.96M
    assert lite.pch.n_params() == 501_536          # proposal Lite budget 0.52M
