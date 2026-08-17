"""B2: parse the geometry phrase out of a ``<where>`` span into a dense vector.

Why this exists (the whole diagnostic chain in one paragraph): the geometry
*is* in the data -- `<where>` names the shape for 79% of samples and gets it
right 82% of the time (reading 2) -- and the head cannot use it: substituting
ground-truth reasoning text moves IoU by +0.0007..+0.0087, all under the 0.01
decision band (reading 1).  The suspected lesion is the pooled `<where>` hidden
-> 256-d -> FiLM pathway, which is the same shape of bottleneck that cost 8.6
IoU points and resurrected M3 on the visual side (the pooled-control arm).  So
this module turns the phrase into an explicit low-dimensional code that is
**broadcast spatially into the tower input**, never pooled.

The vocabulary is reverse-engineered from the GT text templates, then applied
unchanged to generated text -- the same anti-overfitting discipline the
type-word router used (keywords derived from GT, frozen, then applied to model
output).  Nothing here is fitted on predictions.
"""

from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = ["GEOM_SLOTS", "GEOM_DIM", "geom_features",
           "geom_features_from_vrmeta", "geom_features_from_construction",
           "shuffle_features", "describe"]

#: Ordered slots.  Multi-hot, not one-hot: "a large oval falloff toward the
#: lower left" legitimately fires shape=oval, extent=large and two directions.
GEOM_SLOTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # --- shape (the DoF `<where>` names most often, 79% coverage) ---
    ("shape_oval", ("oval", "elliptical", "circular", "round", "ellipse")),
    ("shape_band", ("band", "strip", "stripe", "belt")),
    ("shape_linear", ("linear", "ramp", "gradient", "spanning", "sweep")),
    ("shape_radial", ("radial", "falloff", "vignette", "concentric")),
    ("shape_semantic", ("within", "confined", "itself", "silhouette", "stays")),
    # --- direction ---
    ("dir_left", ("left", "leftward", "leftmost")),
    ("dir_right", ("right", "rightward", "rightmost")),
    ("dir_top", ("top", "upper", "above", "top-left", "top-right")),
    ("dir_bottom", ("bottom", "lower", "below", "beneath", "underneath")),
    ("dir_center", ("center", "centre", "centered", "centred", "middle")),
    ("dir_edge", ("edge", "edges", "border", "corner", "corners", "perimeter")),
    ("dir_horizontal", ("horizontal", "horizontally", "across")),
    ("dir_vertical", ("vertical", "vertically")),
    ("dir_diagonal", ("diagonal", "diagonally")),
    # --- extent ---
    ("ext_large", ("large", "broad", "wide", "big", "huge", "most", "majority")),
    ("ext_small", ("small", "narrow", "thin", "tight", "compact", "little")),
    ("ext_whole", ("whole", "entire", "all", "full", "overall")),
    ("ext_partial", ("half", "part", "partial", "portion", "some")),
    # the q=3 middle bucket: band "moderately wide", radial "moderate".
    # Without it both families collapsed to "neither small nor large" and
    # the parsed arm systematically under-read extent.
    ("ext_moderate", ("moderate", "moderately", "medium")),
    ("ext_soft", ("soft", "gentle", "gradual", "smooth", "subtle")),
    ("ext_hard", ("sharp", "hard", "abrupt", "crisp", "strong")),
)
GEOM_DIM = len(GEOM_SLOTS)
_INDEX = {name: i for i, (name, _) in enumerate(GEOM_SLOTS)}


#: Canonical multi-word phrases the v4a template emits, matched BEFORE single
#: words and then consumed.  Two failures made this necessary:
#:   * "moderately wide" contains "wide", so the q=3 MIDDLE bucket also fired
#:     ext_large -- the two buckets are meant to be exclusive;
#:   * the axis descriptor "diagonal, running from the upper left down to the
#:     lower right" names four directions that are not directions of the edit;
#:     it is one orientation.  Word matching fired dir_left+right+top+bottom on
#:     every diagonal sample.
_PHRASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("diagonal, running from the upper left down to the lower right",
     ("dir_diagonal",)),
    ("diagonal, running from the lower left up to the upper right",
     ("dir_diagonal",)),
    ("moderately wide", ("ext_moderate",)),
)


def _scope(text: str) -> str:
    """The ``edit scope:`` clause, or the whole span if the template is absent.

    Restricted to the scope clause because the subject clause names the object
    ("the glass vase"), not the geometry, and its nouns would fire extent words
    by accident ("the large dog").
    """
    t = (text or "").lower()
    m = re.search(r"(?:edit\s*)?scope\s*:\s*(.*?)(?:;|$)", t, flags=re.S)
    return m.group(1) if m else t


def geom_features(where_text: str) -> np.ndarray:
    """``(GEOM_DIM,)`` multi-hot in {0,1}.  Empty text -> all zeros."""
    # `[a-z]+`, NOT `[a-z\-]+`: the generator emits hyphenated compass
    # corners ("lower-right corner", "upper-left corner"), and keeping the
    # hyphen made them a single token that matched no slot at all -- every
    # corner direction was silently dropped.
    scope = _scope(where_text)
    v = np.zeros(GEOM_DIM, dtype=np.float32)
    # phrases first, then remove them so their constituent words cannot fire
    for phrase, slots in _PHRASES:
        if phrase in scope:
            for s_ in slots:
                v[_INDEX[s_]] = 1.0
            scope = scope.replace(phrase, " ")
    words = set(re.findall(r"[a-z]+", scope))
    for i, (_, keys) in enumerate(GEOM_SLOTS):
        if words & set(keys):
            v[i] = 1.0
    return v


def shuffle_features(v: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Negative control: permute the slots, preserving the number of active bits.

    Preserving the bit count matters -- a control that also changed how *much*
    signal is present would confound "the geometry was wrong" with "there was
    less input", and the registered discipline is that any gain must be shown
    against a same-capacity, same-density control.
    """
    out = np.zeros_like(v)
    on = int((v > 0.5).sum())
    if on:
        out[rng.choice(len(v), size=on, replace=False)] = 1.0
    # EPR-003 registered the conservation and asserted it nowhere, so "the
    # control was same-density" was a property of the code rather than a checked
    # fact.  It also fires -- correctly -- on a *probabilistic* code, where a
    # 0/1 resample is not a same-density control at all and the arm would need a
    # value permutation instead.
    got, want = int((out > 0.5).sum()), on
    if got != want:
        raise AssertionError(
            f"shuffle control changed the active-bit count {want} -> {got}; a "
            "same-density negative control is pre-registered, not incidental")
    return out


#: vrmeta `region` vocabulary -> direction slots (whitespace-split, so
#: "lower right" sets both).  Measured on V_where: center 305, lower 41,
#: left 25, right 16, "lower right" 6, upper 5, "lower left" 2.
_REGION_SLOT = {"center": "dir_center", "centre": "dir_center",
                "lower": "dir_bottom", "upper": "dir_top",
                "left": "dir_left", "right": "dir_right",
                "top": "dir_top", "bottom": "dir_bottom"}
_FAMILY_SLOT = {"radial": "shape_radial", "linear": "shape_linear",
                "band": "shape_band", "semantic": "shape_semantic"}


def geom_features_from_vrmeta(slot_id: str | None, region: str | None) -> np.ndarray:
    """DEPRECATED (AMD-8): the **residual** GT code, shape-only in practice.

    `.vrmeta.json` carries `slot_id` (the family the mask was generated from) and
    `region`, and nothing else about the geometry.  Only the `slot_id` half of
    that is sound:

    * `region` is **not** a direction label.  It is `_coarse_region(effective
      alpha)` in `dataset_build/src/construct/canonical_masks.py:126` -- a 3x3
      centroid bucket of the *rendered* mask, whose middle cell swallows 82% of
      V_where (measured: center 305 of 400).  The campaign data discipline
      forbids using it as a direction, so the nine direction slots this function
      fills are ~82% the single constant `dir_center`.
    * extent is absent from vrmeta entirely, so all seven extent slots are 0 and
      `capture_ext` is not measurable against this code (EPR-004 reported it
      N/A for exactly this reason).

    Kept only so EPR-001..004 stay reproducible byte-for-byte.  **New work must
    use** :func:`geom_features_from_construction`, which reads the same geometry
    parameters the v4a reasoning template read and therefore fills direction and
    extent from a source that is not degenerate.
    """
    v = np.zeros(GEOM_DIM, dtype=np.float32)
    fam = str(slot_id or "").rsplit("-", 1)[0]
    slot = _FAMILY_SLOT.get(fam)
    if slot:
        v[_INDEX[slot]] = 1.0
    for tok in re.findall(r"[a-z]+", (region or "").lower()):
        s = _REGION_SLOT.get(tok)
        if s:
            v[_INDEX[s]] = 1.0
    return v


# ---------------------------------------------------------------------------
# AMD-8: the GT code read off the construction-side geometry parameters.
#
# Every bucket edge below is the *literal* the v4a annotation template uses, so
# the code and the reasoning text that was shown to the annotator are two
# renderings of one decision.  Citations are to
# ``dataset_build/src/construct/responses.py`` (read 2026-08-12); the test
# ``q3vl/whereb/amort/tests/test_geomparse_construction.py`` imports that module
# and asserts the two stay in step, because a silent drift here would show up as
# "the extractor got worse" rather than as a bug.
#
# The geometry parameters themselves live in the databuild projection
# (``canonical_candidates.payload->'geometry'``, keyed by ``candidate_id``);
# they are NOT in ``.vrmeta.json``.  See
# ``q3vl/whereb/scripts/export_construct_geometry.py``.
# ---------------------------------------------------------------------------

#: ``responses.py:1143-1149`` ``_AXIS_BUCKETS``, mapped onto the direction slots
#: the same words hit in :func:`geom_features`.  "diagonal, running from ..." is
#: one orientation, not four corners -- the same reason ``_PHRASES`` exists.
_AXIS_EDGE_SLOT: tuple[tuple[float, str], ...] = (
    (22.5, "dir_horizontal"),
    (67.5, "dir_diagonal"),
    (112.5, "dir_vertical"),
    (157.5, "dir_diagonal"),
    (180.0, "dir_horizontal"),
)

#: ``responses.py:1151-1154`` ``_COMPASS``, in the same sector order.  Only the
#: compass *direction* is taken: the "edge"/"corner" noun that every entry ends
#: in is boilerplate (it is constant inside the linear family and carries no
#: information), so ``dir_edge`` is deliberately left unset -- see NOTES.
_COMPASS_SLOTS: tuple[tuple[str, ...], ...] = (
    ("dir_right",),                 # right edge
    ("dir_bottom", "dir_right"),    # lower-right corner
    ("dir_bottom",),                # bottom edge
    ("dir_bottom", "dir_left"),     # lower-left corner
    ("dir_left",),                  # left edge
    ("dir_top", "dir_left"),        # upper-left corner
    ("dir_top",),                   # top edge
    ("dir_top", "dir_right"),       # upper-right corner
)

#: ``responses.py:1250`` band gauge: narrow / moderately wide / broad.
_BAND_GAUGE: tuple[tuple[float, str], ...] = (
    (0.35, "ext_small"), (0.6, "ext_moderate"), (math.inf, "ext_large"))
#: ``responses.py:1266`` radial gauge: tight / moderate / large.
_RADIAL_GAUGE: tuple[tuple[float, str], ...] = (
    (0.10, "ext_small"), (0.35, "ext_moderate"), (math.inf, "ext_large"))

#: ``responses.py:1138`` -- the only slot modes that declare a geometry.
_GEOMETRY_FAMILIES = frozenset({"band", "linear", "radial"})


def _num(geometry: Mapping[str, Any], name: str, default: float = 0.0) -> float:
    """``responses.py:1164`` ``_geometry_number`` -- legacy ``"+0.42"`` tolerated."""
    try:
        return float(str(geometry.get(name, default)).lstrip("+"))
    except (TypeError, ValueError):
        return default


def _inside(geometry: Mapping[str, Any]) -> bool:
    """``responses.py:1174`` ``_applies_inside``."""
    return str(geometry.get("Flipped", "false")).strip().lower().lstrip("+") == "true"


def _ellipse(geometry: Mapping[str, Any]) -> tuple[float, float, float, float]:
    """``responses.py:1224`` -- (cx, cy, half_along, half_across), normalised."""
    left, right = _num(geometry, "Left"), _num(geometry, "Right")
    top, bottom = _num(geometry, "Top"), _num(geometry, "Bottom")
    return ((left + right) / 2.0, (top + bottom) / 2.0,
            abs(right - left) / 2.0, abs(bottom - top) / 2.0)


def _linear_vector(geometry: Mapping[str, Any]) -> tuple[float, float]:
    """``responses.py:1233`` -- direction the edit strengthens in."""
    dx = _num(geometry, "FullX", 1.0) - _num(geometry, "ZeroX")
    dy = _num(geometry, "FullY") - _num(geometry, "ZeroY")
    return (-dx, -dy) if _inside(geometry) else (dx, dy)


def _visual(dx: float, dy: float, size: tuple[float, float] | None) -> tuple[float, float]:
    """``responses.py:1194`` ``visual_vector``.

    Lightroom geometry is normalised (the frame squashed to a unit square); the
    annotator saw the pixels, where the same direction is stretched back by the
    aspect ratio.  A normalised 45 deg is 33.7 deg on a 3:2 print, which is
    enough to cross an orientation bucket edge -- so the axis slot is wrong
    without ``size``.
    """
    dx, dy = float(dx), float(dy)
    if not size:
        return dx, dy
    w, h = float(size[0]), float(size[1])
    if w <= 0.0 or h <= 0.0:
        return dx, dy
    return dx * w, dy * h


def _visual_angle(angle_degrees: float, size: tuple[float, float] | None) -> float:
    """``responses.py:1217`` ``visual_angle``."""
    theta = math.radians(float(angle_degrees))
    dx, dy = _visual(math.cos(theta), math.sin(theta), size)
    return math.degrees(math.atan2(dy, dx))


def _axis_slot(angle_degrees: float) -> str:
    """``responses.py:1179`` ``axis_bucket``, returning the slot the word hits."""
    angle = float(angle_degrees) % 180.0
    for bound, slot in _AXIS_EDGE_SLOT:
        if angle < bound:
            return slot
    return _AXIS_EDGE_SLOT[-1][1]


def _compass_slots(dx: float, dy: float) -> tuple[str, ...]:
    """``responses.py:1188`` ``compass_bucket``, returning the slots it names."""
    angle = math.degrees(math.atan2(float(dy), float(dx))) % 360.0
    return _COMPASS_SLOTS[int(((angle + 22.5) % 360.0) // 45.0)]


def _gauge_slot(table: Sequence[tuple[float, str]], value: float) -> str:
    for bound, slot in table:
        if value < bound:
            return slot
    return table[-1][1]


def _alpha_moments(alpha: np.ndarray) -> tuple[float, float, float]:
    """(cx, cy, cover) of a rendered alpha field, normalised to the frame.

    Same first moment ``canonical_masks.py:126 _coarse_region`` takes before it
    throws the numbers away for a 3x3 word -- taken here at full precision,
    which is the whole point of the q=3 continuous group.
    """
    a = np.asarray(alpha, dtype=np.float64)
    total = float(a.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("alpha has no mass")
    yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    return (float((a * xx).sum() / total) / a.shape[1],
            float((a * yy).sum() / total) / a.shape[0],
            float(a.mean()))


def geom_features_from_construction(
    slot_id: str | None,
    geometry: Mapping[str, Any] | None,
    size: tuple[float, float] | None = None,
    *,
    alpha: np.ndarray | None = None,
    alpha_mean: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The GT GeoCode, read off the construction-side geometry parameters (AMD-8).

    This is the project privilege the literature route does not have, taken at
    the *right* place: the same `geometry` dict that
    ``responses.py:_geometry_words`` turned into the annotator's edit-region
    hint, so shape, direction and extent all come from the numbers the mask was
    actually rasterised from -- not from `region`, whose 82% "center" made the
    old code shape-only (see :func:`geom_features_from_vrmeta`).

    Args:
        slot_id: ``"band-1"``, ``"radial-0"``, ... -- family before the dash.
        geometry: ``canonical_candidates.payload->'geometry'``.  ``None`` for the
            ``semantic`` family, which by construction declares no geometry
            (``responses.py:1138``).
        size: ``(width, height)`` of the **before** image the annotator saw.
            Only its ratio matters.  Omitting it silently moves samples across
            orientation bucket edges, so callers should pass it.
        alpha: the rendered ``.cgt`` field, if available.  Makes the continuous
            group exact (and defined for ``semantic``); otherwise cx/cy fall
            back to the analytic centre.
        alpha_mean: construction-side ``effective_alpha_mean``, used for `cover`
            when ``alpha`` is not passed.

    Returns:
        ``(c_disc, c_cont, conf)`` -- the §2.1 GeoCode: ``c_disc`` (GEOM_DIM,)
        multi-hot; ``c_cont`` (3,) = (cx, cy, cover) in [0,1]; ``conf`` (4,) =
        group confidence [shape, dir, extent, cont], 1.0 where the construction
        side actually says something and 0.0 where it says nothing (D-4: a
        silent group takes the null path rather than pretending to be an
        all-zero code).

    Extent slots ``ext_whole``/``ext_partial``/``ext_soft``/``ext_hard`` and the
    ``shape_oval`` slot are reachable from the template vocabulary but are not
    filled here beyond ``ext_whole`` for the linear family; see NOTES for the
    two open calls this leaves.
    """
    v = np.zeros(GEOM_DIM, dtype=np.float32)
    cont = np.zeros(3, dtype=np.float32)
    conf = np.zeros(4, dtype=np.float32)

    fam = str(slot_id or "").rsplit("-", 1)[0]
    shape_slot = _FAMILY_SLOT.get(fam)
    if shape_slot is not None:
        v[_INDEX[shape_slot]] = 1.0
        conf[0] = 1.0

    have_geom = fam in _GEOMETRY_FAMILIES and isinstance(geometry, Mapping)
    cx = cy = None
    cover = None

    if have_geom:
        assert geometry is not None
        if fam in ("band", "radial"):
            # responses.py:1244 -- one axis word for both, from the same call.
            v[_INDEX[_axis_slot(_visual_angle(_num(geometry, "Angle"), size))]] = 1.0
            cx, cy, half_along, half_across = _ellipse(geometry)
            if fam == "band":
                # responses.py:1246,1250 -- gauge on the band's full width.
                v[_INDEX[_gauge_slot(_BAND_GAUGE, 2.0 * half_across)]] = 1.0
                # The band's own coverage is its width: it runs off both edges
                # of the frame (responses.py:1253), so it sweeps that fraction.
                cover = min(max(2.0 * half_across, 0.0), 1.0)
            else:
                # responses.py:1265-1266 -- gauge on the ellipse area.
                area = math.pi * half_along * half_across
                v[_INDEX[_gauge_slot(_RADIAL_GAUGE, area)]] = 1.0
                # responses.py:1268 "centred on the subject" -- a positional
                # claim the geometry backs (radial_geom centres the ellipse on
                # the subject PCA centroid, subject_geom.py:83).
                v[_INDEX["dir_center"]] = 1.0
                cover = min(max(area, 0.0), 1.0)
        else:  # linear
            dx, dy = _visual(*_linear_vector(geometry), size)
            # responses.py:1281 -- the axis is recomputed from the vector here.
            v[_INDEX[_axis_slot(math.degrees(math.atan2(dy, dx)))]] = 1.0
            # responses.py:1285 -- which way the ramp strengthens.
            for s_ in _compass_slots(dx, dy):
                v[_INDEX[s_]] = 1.0
            # responses.py:1282 "spanning the whole frame".
            v[_INDEX["ext_whole"]] = 1.0
            cx = (_num(geometry, "ZeroX") + _num(geometry, "FullX", 1.0)) / 2.0
            cy = (_num(geometry, "ZeroY") + _num(geometry, "FullY")) / 2.0
            cover = 1.0
        conf[1] = conf[2] = 1.0

    if alpha is not None:
        try:
            cx, cy, cover = _alpha_moments(alpha)
        except ValueError:
            pass
    elif alpha_mean is not None and math.isfinite(float(alpha_mean)):
        cover = float(alpha_mean)

    if cx is not None and cy is not None and cover is not None:
        cont[:] = (min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0),
                   min(max(cover, 0.0), 1.0))
        conf[3] = 1.0
    return v, cont, conf


def describe(v: Sequence[float]) -> list[str]:
    """Active slot names -- for viz captions and per-sample audit rows."""
    return [GEOM_SLOTS[i][0] for i, x in enumerate(v) if x > 0.5]
