"""WEVAL-1 taxonomy: the geometry is checked against shapes whose answer is known.

A classification that is only asserted against itself can drift without anyone
noticing, and every downstream statement in CLASS_REPORT.md ("small regions are
the hard ones") inherits that drift.  So each quantity is measured here on a
synthetic shape with an analytic value: a disc, a square, an annulus, two blobs,
an L, and a soft-edged blob.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from q3vl.whereb.analysis.taxonomy import (
    AREA_CLASSES, DIMENSIONS, TaxonomyConfig, classify_geometry, global_labels,
    mask_geometry, perimeter_marching_squares,
)

H = W = 256


def disc(r: float = 60.0, cy: float = 128.5, cx: float = 128.5) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W]
    return (((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r).astype(np.float32)


def square(a: int = 120) -> np.ndarray:
    m = np.zeros((H, W), np.float32)
    lo = (H - a) // 2
    m[lo:lo + a, lo:lo + a] = 1.0
    return m


# --- perimeter --------------------------------------------------------------

def test_perimeter_matches_the_analytic_circle_within_5pct():
    """A crack-following count would be 8R = +27% here; the weighted form is +4.5%."""
    p = perimeter_marching_squares(disc(60) > 0.5)
    truth = 2 * math.pi * 60
    assert abs(p - truth) / truth < 0.05
    assert p == pytest.approx(393.99, abs=0.5)          # pinned: the report quotes it


def test_perimeter_matches_the_analytic_square():
    p = perimeter_marching_squares(square(120) > 0.5)
    assert p == pytest.approx(480.0, rel=0.02)
    assert p == pytest.approx(476.0, abs=0.5)           # pinned


def test_perimeter_of_an_empty_mask_is_zero():
    assert perimeter_marching_squares(np.zeros((8, 8), bool)) == 0.0


# --- circularity ordering ---------------------------------------------------

def test_circularity_orders_disc_above_square_above_annulus():
    g_disc = mask_geometry(disc(60))
    g_sq = mask_geometry(square(120))
    ring = disc(60) - disc(30)
    g_ring = mask_geometry(ring)
    assert g_disc["circularity"] > g_sq["circularity"] > g_ring["circularity"]
    # the three numbers the module docstring and the report quote
    assert g_disc["circularity"] == pytest.approx(0.915, abs=0.01)
    assert g_sq["circularity"] == pytest.approx(0.799, abs=0.01)
    assert g_ring["circularity"] == pytest.approx(0.300, abs=0.01)


def test_a_shredded_field_is_less_circular_than_a_compact_one_of_equal_area():
    """The dimension has to measure shape, not size: equal area, different score."""
    rng = np.random.default_rng(0)
    compact = square(100)
    n = int(compact.sum())
    flat = np.zeros(H * W, np.float32)
    flat[rng.choice(H * W, size=n, replace=False)] = 1.0
    shredded = flat.reshape(H, W)
    assert mask_geometry(compact)["area_px"] == mask_geometry(shredded)["area_px"]
    assert mask_geometry(shredded)["circularity"] < 0.05
    assert mask_geometry(compact)["circularity"] > 0.5


# --- area / components / holes ---------------------------------------------

def test_area_frac_is_the_hand_computed_fraction():
    g = mask_geometry(square(128))
    assert g["area_px"] == 128 * 128
    assert g["area_frac"] == pytest.approx(128 * 128 / (H * W))
    assert g["area_frac"] == pytest.approx(0.25)


def test_two_blobs_are_two_components_and_specks_are_not():
    m = np.zeros((H, W), np.float32)
    m[20:80, 20:80] = 1.0
    m[160:220, 160:220] = 1.0
    assert mask_geometry(m)["n_components"] == 2
    m2 = square(120).copy()
    m2[5:8, 5:8] = 1.0                        # 9 px speck: below both floors
    g2 = mask_geometry(m2)
    assert g2["n_components_raw"] == 2
    assert g2["n_components"] == 1, "an anti-aliasing speck is not a second region"


def test_an_annulus_has_exactly_one_hole_and_a_disc_has_none():
    assert mask_geometry(disc(60) - disc(30))["n_holes"] == 1
    assert mask_geometry(disc(60))["n_holes"] == 0


def test_a_pinhole_is_not_a_hole():
    m = square(120).copy()
    m[100:103, 100:103] = 0.0                 # 9 px
    assert mask_geometry(m)["n_holes"] == 0


# --- position ---------------------------------------------------------------

def test_centroid_distance_is_zero_at_the_centre_and_grows_outwards():
    assert mask_geometry(disc(40))["centroid_dist_rel"] < 0.01
    off = mask_geometry(disc(30, cy=40.5, cx=40.5))
    assert off["centroid_dist_rel"] > 0.4


def test_position_uses_short_side_units_so_aspect_ratio_does_not_bias_it():
    """A square frame and a wide frame put the same relative offset in the same
    class -- the whole point of matching ``norm_coords``' convention."""
    tall = np.zeros((256, 512), np.float32)
    tall[118:138, 246:266] = 1.0              # centred
    g = mask_geometry(tall)
    assert g["centroid_dist_rel"] < 0.01
    assert classify_geometry(g)["position"] == "center"


# --- softness ---------------------------------------------------------------

def test_soft_frac_is_relative_to_the_mask_maximum():
    """A 0.64-amplitude hard-edged mask must not read as 100% soft."""
    hard = disc(60) * 0.64
    g = mask_geometry(hard)
    assert g["mask_max"] == pytest.approx(0.64)
    assert g["soft_frac_absolute"] == pytest.approx(1.0), "the absolute band saturates"
    assert g["soft_frac"] < 0.05, "the relative band sees a hard edge"
    assert classify_geometry(g)["softness"] == "hard"


def test_a_feathered_edge_reads_as_soft():
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.sqrt((yy - 128.5) ** 2 + (xx - 128.5) ** 2)
    feather = np.clip((90 - d) / 60.0, 0.0, 1.0).astype(np.float32)
    g = mask_geometry(feather)
    assert g["soft_frac"] > 0.4
    assert classify_geometry(g)["softness"] == "soft"


# --- classification ---------------------------------------------------------

@pytest.mark.parametrize("frac,want", [
    (0.02, "tiny"), (0.10, "small"), (0.30, "medium"), (0.60, "large"),
])
def test_area_classes_follow_the_configured_cuts(frac, want):
    side = int(round(math.sqrt(frac * H * W)))
    m = np.zeros((H, W), np.float32)
    m[:side, :side] = 1.0
    assert classify_geometry(mask_geometry(m))["area"] == want


def test_every_dimension_gets_a_label_and_they_are_all_known_values():
    labels = classify_geometry(mask_geometry(disc(60)))
    assert set(labels) == set(DIMENSIONS)
    assert labels["components"] == "single"
    assert labels["topology"] == "solid"
    assert labels["boundary"] == "compact"
    assert labels["area"] in AREA_CLASSES


def test_thresholds_are_configurable_without_touching_the_geometry():
    g = mask_geometry(disc(60))
    strict = TaxonomyConfig(compact_min=0.95)
    assert classify_geometry(g)["boundary"] == "compact"
    assert classify_geometry(g, strict)["boundary"] == "complex"


def test_an_empty_mask_is_flagged_rather_than_classified_as_small():
    g = mask_geometry(np.zeros((32, 32), np.float32))
    assert g["degenerate"] == "empty_mask"
    assert all(v.startswith("degenerate:") for v in classify_geometry(g).values())


def test_global_samples_get_their_own_class_on_every_dimension():
    assert set(global_labels()) == set(DIMENSIONS)
    assert set(global_labels().values()) == {"global"}
