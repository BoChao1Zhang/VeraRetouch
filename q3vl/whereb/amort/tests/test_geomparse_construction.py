"""AMD-8: the GT code read off the construction-side geometry parameters.

Two things are worth testing and one thing is not.  Worth testing: (a) the code
lands on the same buckets the v4a annotation template landed on -- checked
against the *real* ``responses.py`` rather than a copy of its numbers, because a
copy is exactly what would silently drift; (b) the resulting direction/extent
columns are not degenerate, which is the whole reason `region` was retired.  Not
worth testing: that a fixed vector equals a fixed vector, which is what a
snapshot test of the code would be.

The four per-family fixtures are real rows (``prod-l3/l4/l6``, read 2026-08-12),
frozen here with the ``<where>`` span the annotator actually wrote for them, so
the end-to-end claim "geometry parameters -> code -> the same words the sample's
reasoning uses" is checked on data and not on a hand-made geometry dict.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from q3vl.whereb.amort.geomparse import (GEOM_DIM, GEOM_SLOTS, describe,
                                         geom_features,
                                         geom_features_from_construction,
                                         geom_features_from_vrmeta)

_NAMES = [n for n, _ in GEOM_SLOTS]
_IDX = {n: i for i, n in enumerate(_NAMES)}
_DIR = [i for i, n in enumerate(_NAMES) if n.startswith("dir_")]
_EXT = [i for i, n in enumerate(_NAMES) if n.startswith("ext_")]

#: The slots the construction code claims.  ``dir_edge`` and
#: ``ext_soft``/``ext_hard``/``ext_partial``/``shape_oval`` are reachable from
#: the template vocabulary but deliberately not derived, so a template-vs-code
#: comparison must be scoped or it fails on boilerplate nouns ("runs off both
#: edges of the frame", "the strong side").
_CLAIMED = [_IDX[n] for n in (
    "dir_left", "dir_right", "dir_top", "dir_bottom", "dir_center",
    "dir_horizontal", "dir_vertical", "dir_diagonal",
    "ext_large", "ext_small", "ext_whole", "ext_moderate")]


# Real rows: (slot_id, geometry, before-image size, the sample's own <where>).
FIXTURES = {
    "band": dict(
        sample_id="sft_0009fa3bd6d403b7ca718eed40ac07cb",
        slot_id="band-0",
        geometry={"Angle": 117.24, "Bottom": 0.9577, "Feather": 55.0,
                  "Flipped": "true", "Left": -1.0821, "Midpoint": 50.0,
                  "Right": 2.1179, "Roundness": 0.0, "Top": 0.3917},
        size=(360.0, 540.0),
        alpha_mean=0.59180087,
        where=("subject: the seated woman; edit scope: a vertical band through "
               "the woman, extending beyond her into the foliage, ground, logs, "
               "and basket and running off both edges of the frame"),
        expect={"shape_band", "dir_vertical", "ext_moderate"},
        # what survived into the annotator's own words: the gauge did not
        where_expect={"shape_band", "dir_vertical"},
    ),
    "linear": dict(
        sample_id="sft_00018be2765b8db9a34e6ecf9c6f9211",
        slot_id="linear-0",
        geometry={"Flipped": "false", "FullX": 0.4334, "FullY": 0.4965,
                  "ZeroX": 0.8139, "ZeroY": 0.5035},
        size=(1422.0, 800.0),
        alpha_mean=0.49999994,
        where=("subject: the woman with the bed, wall, and nearby furnishings; "
               "edit scope: a horizontal linear gradient spanning the whole "
               "frame, strongest at the woman and nearby bedroom scene, fading "
               "continuously toward the opposite side"),
        expect={"shape_linear", "dir_horizontal", "dir_left", "ext_whole"},
        # the compass word became scene content ("strongest at the woman")
        where_expect={"shape_linear", "dir_horizontal", "ext_whole"},
    ),
    "radial": dict(
        sample_id="sft_0003f99148b50e292b6837a9a9b6e76e",
        slot_id="radial-0",
        geometry={"Angle": 84.78, "Bottom": 0.8809, "Feather": 85.0,
                  "Flipped": "true", "Left": -0.0726, "Midpoint": 50.0,
                  "Right": 0.8788, "Roundness": 0.0, "Top": 0.3498},
        size=(1440.0, 1614.0),
        alpha_mean=0.38210529,
        where=("subject: the woman on the city street; edit scope: a large "
               "vertical oval falloff centered on the woman, strongest over her "
               "and fading outward beyond her outline into nearby buildings, "
               "street, and railing while leaving the far corners unchanged"),
        expect={"shape_radial", "dir_vertical", "dir_center", "ext_large"},
        where_expect={"shape_radial", "dir_vertical", "dir_center", "ext_large"},
    ),
    "semantic": dict(
        sample_id="sft_0007d6899e970709b1577f9fe9b9b18a",
        slot_id="semantic-1",
        geometry=None,
        size=(2048.0, 1365.0),
        alpha_mean=0.00813279,
        where=("subject: the person standing in the lower part of the city "
               "plaza; edit scope: stays within the person"),
        expect={"shape_semantic"},
        where_expect={"shape_semantic"},
    ),
}


def _template_code(slot_mode, geometry, size):
    """The v4a hint this candidate produced, parsed back with the text parser."""
    from dataset_build.src.construct.responses import _geometry_words
    return geom_features(_geometry_words(slot_mode, geometry, size))


# -- 1. the frozen contract -------------------------------------------------

def test_geom_slots_frozen_at_21():
    """AMD-6 froze the vocabulary; three downstream clients index it by position."""
    assert GEOM_DIM == 21
    assert sum(n.startswith("shape_") for n in _NAMES) == 5
    assert sum(n.startswith("dir_") for n in _NAMES) == 9
    assert sum(n.startswith("ext_") for n in _NAMES) == 7


def test_bucket_tables_match_the_template_module():
    """Bucket edges and compass order are read off responses.py, not copied blind."""
    from dataset_build.src.construct import responses as R

    from q3vl.whereb.amort import geomparse as G

    assert [b for b, _ in R._AXIS_BUCKETS] == [b for b, _ in G._AXIS_EDGE_SLOT]
    for (_, word), (_, slot) in zip(R._AXIS_BUCKETS, G._AXIS_EDGE_SLOT):
        assert slot == f"dir_{word.split(',')[0]}"
    assert len(R._COMPASS) == len(G._COMPASS_SLOTS) == 8
    for word, slots in zip(R._COMPASS, G._COMPASS_SLOTS):
        # "lower-right corner" -> dir_bottom + dir_right, minus the noun.
        want = {"lower": "dir_bottom", "upper": "dir_top",
                "left": "dir_left", "right": "dir_right", "top": "dir_top",
                "bottom": "dir_bottom"}
        named = {want[w] for w in word.replace("-", " ").split()
                 if w in want}
        assert set(slots) == named, word
    assert R._GEOMETRY_MODES == G._GEOMETRY_FAMILIES


# -- 2. end to end, one real sample per family ------------------------------

@pytest.mark.parametrize("family", sorted(FIXTURES))
def test_family_end_to_end(family):
    f = FIXTURES[family]
    v, cont, conf = geom_features_from_construction(
        f["slot_id"], f["geometry"], f["size"], alpha_mean=f["alpha_mean"])
    assert set(describe(v)) == f["expect"], describe(v)

    # the bucket words that survived paraphrase are in the sample's own span.
    # ``where_expect`` is a subset of ``expect``: the annotator is free to drop a
    # word (this band lost its gauge), which is the 82%-not-100% the A1 arm
    # measures -- it must never *gain* one the geometry does not license.
    parsed = geom_features(f["where"])
    assert f["where_expect"] <= f["expect"]
    for slot in f["where_expect"]:
        assert parsed[_IDX[slot]] >= 0.5, f"{slot} missing from <where>: {f['where']}"

    assert conf[0] == 1.0
    if f["geometry"] is None:
        assert conf[1] == conf[2] == 0.0      # semantic declares no geometry
        assert conf[3] == 0.0                 # ... and no centre without a mask
    else:
        assert conf[1] == conf[2] == conf[3] == 1.0
        assert np.all((cont >= 0.0) & (cont <= 1.0))


@pytest.mark.parametrize("family", ["band", "linear", "radial"])
def test_family_matches_the_template_text(family):
    """The gate: code and the v4a hint are one bucket decision, two renderings."""
    f = FIXTURES[family]
    v, _, _ = geom_features_from_construction(
        f["slot_id"], f["geometry"], f["size"])
    t = _template_code(family, f["geometry"], f["size"])
    assert np.array_equal(v[_CLAIMED] >= 0.5, t[_CLAIMED] >= 0.5), (
        f"code={describe(v)} template={describe(t)}")


# -- 3. the bucket edges themselves -----------------------------------------

@pytest.mark.parametrize("width,slot", [
    (0.20, "ext_small"), (0.3499, "ext_small"),
    (0.35, "ext_moderate"), (0.5999, "ext_moderate"),
    (0.60, "ext_large"), (0.80, "ext_large"),
])
def test_band_gauge_edges(width, slot):
    """responses.py:1250 -- 0.35 / 0.6, and the q=3 middle bucket is reachable."""
    geom = {"Left": -1.6, "Right": 1.6, "Top": 0.5 - width / 2,
            "Bottom": 0.5 + width / 2, "Angle": 0.0, "Flipped": "true"}
    v, _, _ = geom_features_from_construction("band-0", geom, (100.0, 100.0))
    assert slot in describe(v)
    t = _template_code("band", geom, (100.0, 100.0))
    assert np.array_equal(v[_CLAIMED] >= 0.5, t[_CLAIMED] >= 0.5)


@pytest.mark.parametrize("area,slot", [
    (0.05, "ext_small"), (0.0999, "ext_small"),
    (0.1001, "ext_moderate"), (0.3499, "ext_moderate"),
    (0.3501, "ext_large"), (0.60, "ext_large"),
])
def test_radial_gauge_edges(area, slot):
    """responses.py:1266 -- 0.10 / 0.35 on pi*a*b, not on the bounding box."""
    half = math.sqrt(area / math.pi)          # a == b => pi*a*b == area
    geom = {"Left": 0.5 - half, "Right": 0.5 + half,
            "Top": 0.5 - half, "Bottom": 0.5 + half,
            "Angle": 0.0, "Flipped": "true"}
    v, _, _ = geom_features_from_construction("radial-0", geom, (100.0, 100.0))
    assert slot in describe(v)
    t = _template_code("radial", geom, (100.0, 100.0))
    assert np.array_equal(v[_CLAIMED] >= 0.5, t[_CLAIMED] >= 0.5)


@pytest.mark.parametrize("angle,slot", [
    (0.0, "dir_horizontal"), (22.4, "dir_horizontal"),
    (22.6, "dir_diagonal"), (67.4, "dir_diagonal"),
    (67.6, "dir_vertical"), (112.4, "dir_vertical"),
    (112.6, "dir_diagonal"), (157.4, "dir_diagonal"),
    (157.6, "dir_horizontal"),
])
def test_axis_edges_on_a_square_frame(angle, slot):
    geom = {"Left": -1.6, "Right": 1.6, "Top": 0.4, "Bottom": 0.6,
            "Angle": angle, "Flipped": "true"}
    v, _, _ = geom_features_from_construction("band-0", geom, (100.0, 100.0))
    assert slot in describe(v)


def test_aspect_ratio_moves_the_axis_bucket():
    """The reason ``size`` is not optional in practice (responses.py:1194).

    45 deg normalised on a 3:2 frame is 33.7 deg on the picture -- across the
    22.5 edge in neither case, so the test uses an angle that does cross: the
    annotator saw "horizontal" where the stored number says "diagonal".
    """
    geom = {"Left": -1.6, "Right": 1.6, "Top": 0.4, "Bottom": 0.6,
            "Angle": 30.0, "Flipped": "true"}
    square, _, _ = geom_features_from_construction("band-0", geom, (100.0, 100.0))
    wide, _, _ = geom_features_from_construction("band-0", geom, (1000.0, 100.0))
    assert "dir_diagonal" in describe(square)
    assert "dir_horizontal" in describe(wide)


def test_linear_compass_points_where_the_ramp_strengthens():
    """Zero -> Full is the direction of increasing alpha (responses.py:1233)."""
    geom = {"ZeroX": 0.1, "ZeroY": 0.5, "FullX": 0.9, "FullY": 0.5,
            "Flipped": "false"}
    v, _, _ = geom_features_from_construction("linear-0", geom, (100.0, 100.0))
    assert {"dir_right", "dir_horizontal", "ext_whole"} <= set(describe(v))
    assert "dir_left" not in describe(v)


# -- 4. non-degeneracy: the point of AMD-8 ----------------------------------

def _spread():
    """A synthetic spread with all four families and all three axis buckets."""
    out = []
    for k, angle in enumerate((5.0, 45.0, 95.0, 140.0, 170.0)):
        out.append(("band-0", {"Left": -1.6, "Right": 1.6, "Top": 0.3,
                               "Bottom": 0.3 + 0.2 + 0.1 * k, "Angle": angle,
                               "Flipped": "true"}))
        half = 0.1 + 0.06 * k
        out.append(("radial-0", {"Left": 0.5 - half, "Right": 0.5 + half,
                                 "Top": 0.4 - half, "Bottom": 0.4 + half,
                                 "Angle": angle, "Flipped": "true"}))
        theta = math.radians(angle)
        out.append(("linear-0", {"ZeroX": 0.5 - 0.4 * math.cos(theta),
                                 "ZeroY": 0.5 - 0.4 * math.sin(theta),
                                 "FullX": 0.5 + 0.4 * math.cos(theta),
                                 "FullY": 0.5 + 0.4 * math.sin(theta),
                                 "Flipped": "false"}))
        out.append(("semantic-0", None))
    return out


def test_direction_is_not_a_constant():
    """`region` gave one value 82% of the time; the point is that this does not."""
    codes = np.stack([geom_features_from_construction(s, g, (100.0, 100.0))[0]
                      for s, g in _spread()])
    lit = {n: float((codes[:, _IDX[n]] >= 0.5).mean())
           for n in _NAMES if n.startswith("dir_")}
    assert lit["dir_center"] < 0.5, lit
    assert sum(v > 0.0 for v in lit.values()) >= 5, lit


def test_extent_is_not_identically_zero():
    codes = np.stack([geom_features_from_construction(s, g, (100.0, 100.0))[0]
                      for s, g in _spread()])
    lit = (codes[:, _EXT] >= 0.5).any(axis=1)
    # every geometry family declares an extent; semantic (1 in 4) declares none
    assert lit.mean() == pytest.approx(0.75)
    assert (codes[:, _EXT] >= 0.5).sum(axis=0).max() > 0


def test_vrmeta_code_is_the_degenerate_one():
    """The comparison that motivated AMD-8, stated as an executable fact."""
    old = np.stack([geom_features_from_vrmeta("radial-0", "center")
                    for _ in range(4)])
    assert (old[:, _EXT] >= 0.5).sum() == 0
    assert describe(old[0]) == ["shape_radial", "dir_center"]


# -- 5. continuous group ----------------------------------------------------

def test_cont_from_alpha_is_exact_and_covers_semantic():
    alpha = np.zeros((100, 100), dtype=np.float32)
    alpha[10:30, 60:80] = 1.0                     # centre (0.70, 0.20), 4% cover
    v, cont, conf = geom_features_from_construction("semantic-1", None, None,
                                                    alpha=alpha)
    assert conf[3] == 1.0
    assert cont[0] == pytest.approx(0.695, abs=1e-3)
    assert cont[1] == pytest.approx(0.195, abs=1e-3)
    assert cont[2] == pytest.approx(0.04, abs=1e-6)
    assert describe(v) == ["shape_semantic"]


def test_cont_stays_in_the_declared_domain():
    """§2.1 says [0,1]; band ellipses run to Left=-1.08 in the raw parameters."""
    geom = FIXTURES["band"]["geometry"]
    _, cont, conf = geom_features_from_construction(
        "band-0", geom, FIXTURES["band"]["size"],
        alpha_mean=FIXTURES["band"]["alpha_mean"])
    assert np.all((cont >= 0.0) & (cont <= 1.0))
    assert conf[3] == 1.0


def test_unknown_family_is_all_zero_and_confesses_it():
    v, cont, conf = geom_features_from_construction("mystery-9", None, None)
    assert v.sum() == 0.0 and cont.sum() == 0.0 and conf.sum() == 0.0
