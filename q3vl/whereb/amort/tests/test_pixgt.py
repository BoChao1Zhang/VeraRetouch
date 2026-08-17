"""``q3vl.whereb.amort.pixgt`` -- the shared pixel GT for EPR-018..023.

CPU only.  The two facts a half-cell mistake would hide (8 px on a 32x48 grid)
are pinned directly:

* :func:`eval_analytic` reproduces ``raster_geometry`` bit-for-bit on the grid
  that function itself renders (``x = j/w``, ``y = i/h``);
* :func:`sample_points` at :func:`make_coord` reproduces the raster exactly
  (``2x-1`` + ``align_corners=False``).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dataset_build.src.construct.canonical_masks import (linear_strength,
                                                         raster_geometry)
from q3vl.where.upsample import area_resize
from q3vl.whereb.amort.pixgt import (MASK_TYPE_OF_FAMILY, PixGT, PixGTProvider,
                                     area_project, audit_analytic, eval_analytic,
                                     linear_amount, make_coord, mask_type_of,
                                     render_analytic, sample_points)

RADIAL = {"Left": 0.15, "Right": 0.72, "Top": 0.1, "Bottom": 0.63,
          "Angle": 31.5, "Feather": 42.0}
BAND_FLIP = {**RADIAL, "Flipped": "true"}
LINEAR = {"ZeroX": 0.1, "ZeroY": 0.2, "FullX": 0.85, "FullY": 0.4}


def test_family_mapping():
    assert mask_type_of("radial") == "circulargradient"
    assert mask_type_of("band-1") == "circulargradient"
    assert mask_type_of("linear") == "gradient"
    assert mask_type_of("semantic") is None
    assert set(MASK_TYPE_OF_FAMILY) == {"radial", "band", "linear"}


# --- (c) analytic render ----------------------------------------------------

@pytest.mark.parametrize("geom", [RADIAL, BAND_FLIP])
def test_render_matches_raster_geometry(geom):
    a = render_analytic("circulargradient", geom, 17, 23)
    b = raster_geometry("circulargradient", geom, 17, 23)
    assert torch.equal(a, torch.from_numpy(b))


@pytest.mark.parametrize("mt,geom", [("circulargradient", RADIAL),
                                     ("circulargradient", BAND_FLIP),
                                     ("gradient", LINEAR)])
def test_eval_analytic_equals_raster_on_its_own_grid(mt, geom):
    h, w = 13, 19
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    coords = torch.from_numpy(np.stack([(xs / w).ravel(), (ys / h).ravel()], -1))
    got = eval_analytic(mt, geom, coords, amount_ref_hw=(h, w))
    want = render_analytic(mt, geom, h, w).reshape(-1)
    assert torch.allclose(got, want, atol=1e-6), float((got - want).abs().max())


def test_linear_amount_matches_the_construction_side_formula():
    h, w = 64, 96
    raw = raster_geometry("gradient", LINEAR, h, w)
    want, _eff = linear_strength(raw, 0.5)
    assert linear_amount(LINEAR, (h, w)) == pytest.approx(want, abs=1e-6)


def test_linear_render_applies_the_amount():
    h, w = 40, 60
    raw = raster_geometry("gradient", LINEAR, h, w)
    amount, eff = linear_strength(raw, 0.5)
    got = render_analytic("gradient", LINEAR, h, w)
    assert torch.allclose(got, torch.from_numpy(eff), atol=1e-6)
    # and the recomputation is deterministic, not a stored constant
    assert linear_amount(LINEAR, (h, w)) == pytest.approx(amount, abs=1e-6)


def test_linear_amount_reference_grid_is_declared_not_implied():
    """``amount`` is a mean over a grid, so the grid is part of the number.

    Measured on this geometry (2026-08-14): 0.887121 at 32x48, 0.870340 at
    128x192, 0.866248 at 512x768, 0.865569 at 1024x1536 -- a 2.4% spread between
    the coarsest and the published (.cgt, short side 1024) grid.  The resulting
    FIELD difference is 0.00274 mean-abs, i.e. an order of magnitude inside the
    0.02 pre-flight tolerance -- but it is not zero, so which grid was used is
    recorded in ``PixGTProvider.facts()['amount_ref_hw']`` rather than implied.
    """
    coarse = linear_amount(LINEAR, (32, 48))
    published = linear_amount(LINEAR, (1024, 1536))
    assert coarse > published                        # monotone, not identical
    assert abs(coarse - published) > 1e-3            # the dependence is real
    a = render_analytic("gradient", LINEAR, 128, 192, amount=published)
    b = render_analytic("gradient", LINEAR, 128, 192)   # amount from (128,192)
    assert float((a - b).abs().mean()) < 0.02        # inside the audit tolerance


def test_eval_analytic_refuses_a_linear_family_without_a_reference():
    with pytest.raises(ValueError, match="needs an amount"):
        eval_analytic("gradient", LINEAR, make_coord(4, 4))


def test_unsupported_mask_type_raises():
    with pytest.raises(ValueError, match="unsupported mask_type"):
        eval_analytic("semantic", {}, make_coord(2, 2))


# --- (d) projection / (e) point sampling ------------------------------------

def test_area_project_is_the_gt_low_operator():
    x = torch.rand(64, 96)
    assert torch.equal(area_project(x, (16, 24)),
                       area_resize(x[None, None], (16, 24))[0, 0])


def test_sample_points_at_cell_centres_is_the_identity():
    h, w = 9, 14
    f = torch.rand(h, w)
    got = sample_points(f, make_coord(h, w))
    assert torch.allclose(got, f.reshape(-1), atol=1e-6), \
        "half-cell offset: 2x-1 / align_corners=False disagree"


def test_sample_points_multichannel():
    f = torch.rand(3, 8, 11)
    got = sample_points(f, make_coord(8, 11))
    assert got.shape == (3, 88)
    assert torch.allclose(got, f.reshape(3, -1), atol=1e-6)


def test_make_coord_is_cell_centres_not_corners():
    c = make_coord(2, 2)
    assert torch.allclose(c[0], torch.tensor([0.25, 0.25]))
    assert torch.allclose(c[-1], torch.tensor([0.75, 0.75]))


def test_pixgt_points_uses_the_closed_form_when_analytic():
    a = render_analytic("circulargradient", RADIAL, 32, 48)
    pg = PixGT(alpha=a, source="render", family="radial",
               mask_type="circulargradient", geometry=RADIAL)
    coords = torch.tensor([[0.31, 0.44], [0.9, 0.05]])
    closed = pg.points(coords)
    assert closed.shape == (2,)
    raster = PixGT(alpha=a, source="cgt1024", family="radial").points(coords)
    # both are the same field; the closed form is exact, the raster is bilinear
    assert torch.allclose(closed, raster, atol=0.05)


# --- the dispatcher ---------------------------------------------------------

class _Store:
    def __init__(self, rows):
        self.rows = rows

    def row(self, candidate_id):
        return self.rows.get(candidate_id)


class _Views:
    HI = ".maskhi.png"

    def __init__(self, sids):
        self.sids = set(sids)

    def has(self, sid, suffix):
        return sid in self.sids and suffix == self.HI

    def mask_hi(self, sid):
        return torch.full((64, 96), 0.25)


class _Sample:
    def __init__(self, sid, cand):
        self.sample_id = sid
        self.meta = {"candidate_id": cand}


def _provider(**kw):
    store = _Store({
        "c-radial": {"slot_id": "radial-0", "slot_mode": "radial",
                     "geometry": RADIAL, "effective_alpha_mean": 0.3},
        "c-linear": {"slot_id": "linear-1", "slot_mode": "linear",
                     "geometry": LINEAR, "effective_alpha_mean": 0.5},
        "c-semantic": {"slot_id": "semantic-0", "slot_mode": "semantic",
                       "geometry": None, "effective_alpha_mean": 0.4},
    })
    return PixGTProvider(
        geom_store=store, maskviews=_Views({"s-radial", "s-linear", "s-semantic",
                                            "s-miss"}),
        families={"s-radial": "radial", "s-linear": "linear",
                  "s-semantic": "semantic", "s-miss": "band"},
        **kw)


def test_analytic_families_render_and_semantic_falls_back_counted():
    p = _provider()
    r = p.get(_Sample("s-radial", "c-radial"), size=(32, 48))
    assert r.source == "render" and r.alpha.shape == (32, 48)
    s = p.get(_Sample("s-semantic", "c-semantic"), size=(32, 48))
    assert s.source == "maskhi512" and s.reason == "semantic_no_geometry"
    assert s.alpha.shape == (32, 48)          # projected with area_project
    f = p.facts()
    assert f["n_render"] == 1
    assert f["counts"]["fallback_semantic_no_geometry"] == 1
    assert 0.0 < f["frac_render"] < 1.0


def test_store_miss_is_a_counted_fallback_not_a_crash():
    p = _provider()
    got = p.get(_Sample("s-miss", "nope"), size=(16, 24))
    assert got.source == "maskhi512" and got.reason == "geom_store_miss"
    assert p.facts()["counts"]["fallback_geom_store_miss"] == 1


def test_prefer_maskhi_skips_the_analytic_path_entirely():
    p = _provider(prefer="maskhi")
    got = p.get(_Sample("s-radial", "c-radial"), size=(32, 48))
    assert got.source == "maskhi512" and got.reason == "prefer_maskhi"
    assert p.facts()["n_render"] == 0


def test_linear_family_amount_is_recomputed_per_render():
    p = _provider()
    got = p.get(_Sample("s-linear", "c-linear"), size=(32, 48))
    assert got.source == "render"
    assert got.amount == pytest.approx(linear_amount(LINEAR, (32, 48)), abs=1e-6)


def test_needs_record_is_false_when_the_cgt_branch_is_unreachable():
    assert _provider().needs_record("radial") is False
    assert _provider(raster_fallback="cgt1024").needs_record("radial") is True
    assert _provider(prefer="cgt1024").needs_record("radial") is True


def test_prefer_cgt1024_downgraded_to_maskhi_is_visible_in_facts():
    """A run whose config says 1024 but was served 512 must SAY so."""
    p = _provider(prefer="cgt1024")            # no mask_resolver attached
    got = p.get(_Sample("s-radial", "c-radial"), size=(32, 48))
    assert got.source == "maskhi512"
    assert p.facts()["counts"]["downgraded_cgt1024_to_maskhi512"] == 1


def test_no_raster_at_all_raises_with_both_reasons():
    p = PixGTProvider(geom_store=None, families={"s": "semantic"})
    with pytest.raises(KeyError, match="no pixel GT raster available"):
        p.get(_Sample("s", None), size=(8, 8))


def test_analytic_path_refuses_to_guess_a_resolution():
    with pytest.raises(ValueError, match="pass size"):
        _provider().get(_Sample("s-radial", "c-radial"))


# --- the pre-flight audit ---------------------------------------------------

def test_audit_passes_when_the_raster_is_the_analytic_render():
    class _ExactViews(_Views):
        def mask_hi(self, sid):
            return render_analytic("circulargradient", RADIAL, 64, 96)

    p = _provider()
    p.maskviews = _ExactViews({"s-radial"})
    rep = audit_analytic(p, [(_Sample("s-radial", "c-radial"), None)], tol=0.02)
    assert rep.n == 1 and rep.n_pass == 1 and rep.n_fail == 0
    assert rep.worst < 1e-6


def test_audit_reports_and_does_not_raise_on_a_mismatch():
    p = _provider()                                # views return a constant 0.25
    rep = audit_analytic(p, [(_Sample("s-radial", "c-radial"), None)], tol=0.02)
    assert rep.n_fail == 1 and rep.failures[0]["sample_id"] == "s-radial"
    assert rep.to_dict()["worst_mean_abs"] > 0.02


def test_audit_counts_the_skipped_semantic_family():
    p = _provider()
    rep = audit_analytic(p, [(_Sample("s-semantic", "c-semantic"), None)])
    assert rep.n == 0 and rep.n_skipped == 1
    assert rep.skipped["semantic_no_geometry"] == 1
