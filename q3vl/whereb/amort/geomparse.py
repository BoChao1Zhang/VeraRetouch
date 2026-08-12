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

import re
from typing import Sequence

import numpy as np

__all__ = ["GEOM_SLOTS", "GEOM_DIM", "geom_features",
           "geom_features_from_vrmeta", "shuffle_features", "describe"]

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
    on = int(v.sum())
    if on:
        out[rng.choice(len(v), size=on, replace=False)] = 1.0
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
    """The **ground-truth** geometry code, straight off the construction side.

    This is a project privilege the literature route does not have: `.vrmeta.json`
    carries `slot_id` (the family the mask was actually generated from) and
    `region` (the coarse placement computed from the mask itself), so the code
    needs no text parsing and carries no generation error.

    It is therefore the go/no-go **upper bound** for the whole geometry-injection
    direction: if handing the head a perfect code does not move the number, no
    better parser, decoder or readout of the same information can either.

    Extent slots stay 0 -- vrmeta does not record them -- so this code is exact
    on shape and direction and silent on extent.  Any comparison against the
    parsed code must keep that asymmetry in view.
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


def describe(v: Sequence[float]) -> list[str]:
    """Active slot names -- for viz captions and per-sample audit rows."""
    return [GEOM_SLOTS[i][0] for i, x in enumerate(v) if x > 0.5]
