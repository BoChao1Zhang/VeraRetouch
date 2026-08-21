from __future__ import annotations

import numpy as np
import pytest

from dataset_build.agent_loop.render import _sample_indices, delta_e_map
from dataset_build.agent_loop.visibility import (
    COMPONENT_COLUMNS, EDGE_BAND_HIGH, EDGE_BAND_LOW, REQUIRED_COLUMNS,
    SAMPLE_BUDGET_MAX, SAMPLE_PIXELS_EDGE, SAMPLE_PIXELS_IN, SAMPLE_PIXELS_OUT,
    VISIBILITY_CONTRACT, VISIBILITY_TAU, VisibilityError, assert_component_columns,
    assert_visibility_columns, delta_e_at, draw_region, lab_at, lab_components,
    random_area_mask, visibility_components, visibility_metrics,
)


def _flat_image(shape, value):
    image = np.empty((shape[0], shape[1], 3), dtype=np.float32)
    image[:] = np.asarray(value, dtype=np.float32)
    return image


def _rect_alpha(shape, top, left, height, width, value=1.0):
    alpha = np.zeros(shape, dtype=np.float32)
    alpha[top:top + height, left:left + width] = value
    return alpha


# ------------------------------------------------------------------ definitions
def test_de_in_out_contrast_follow_the_written_definition():
    before = _flat_image((64, 64), (0.40, 0.40, 0.40))
    after = before.copy()
    alpha = _rect_alpha((64, 64), 8, 8, 16, 16)
    after[8:24, 8:24] = (0.60, 0.40, 0.40)

    row = visibility_metrics(before, after, alpha, seed_key="chain-a")

    assert row["visibility_contract"] == VISIBILITY_CONTRACT
    assert row["n_in"] == 16 * 16
    assert row["n_out"] == min(64 * 64 - 16 * 16, SAMPLE_PIXELS_OUT)
    assert row["support_frac"] == pytest.approx(256 / 4096)
    assert row["de_out"] == pytest.approx(0.0, abs=1e-6)
    assert row["de_in"] > 1.0
    assert row["de_contrast"] == pytest.approx(row["de_in"] - row["de_out"])
    # binary alpha -> the weighted mean equals the unweighted one
    expected = float(delta_e_map(before[8:24, 8:24], after[8:24, 8:24]).mean())
    assert row["de_in"] == pytest.approx(expected, rel=1e-5)
    assert row["de_in_p50"] == pytest.approx(expected, rel=1e-5)
    assert row["de_in_p90"] == pytest.approx(expected, rel=1e-5)


def test_de_in_is_alpha_weighted_over_the_support():
    before = _flat_image((32, 32), (0.40, 0.40, 0.40))
    after = before.copy()
    alpha = np.zeros((32, 32), dtype=np.float32)
    alpha[0:16, :] = 1.0
    alpha[16:32, :] = 0.25
    after[0:16, :] = (0.60, 0.40, 0.40)
    after[16:32, :] = (0.45, 0.40, 0.40)

    row = visibility_metrics(before, after, alpha, seed_key="chain-w")

    strong = float(delta_e_map(before[0:16], after[0:16]).mean())
    weak = float(delta_e_map(before[16:32], after[16:32]).mean())
    weighted = (strong * 1.0 * 512 + weak * 0.25 * 512) / (1.0 * 512 + 0.25 * 512)
    assert row["n_in"] == 1024
    assert row["n_out"] == 0
    assert row["de_out"] is None
    assert row["de_contrast"] is None
    assert row["de_in"] == pytest.approx(weighted, rel=1e-5)


def test_de_out_is_nonzero_when_the_edit_leaks_outside():
    before = _flat_image((48, 48), (0.40, 0.40, 0.40))
    alpha = _rect_alpha((48, 48), 4, 4, 12, 12)
    leaked = before.copy()
    leaked[:] = (0.44, 0.40, 0.40)
    leaked[4:16, 4:16] = (0.60, 0.40, 0.40)

    clean = before.copy()
    clean[4:16, 4:16] = (0.60, 0.40, 0.40)

    leaky = visibility_metrics(before, leaked, alpha, seed_key="chain-leak")
    tight = visibility_metrics(before, clean, alpha, seed_key="chain-leak")

    assert leaky["de_out"] > 0.5
    assert tight["de_out"] == pytest.approx(0.0, abs=1e-6)
    assert tight["de_contrast"] > leaky["de_contrast"]


def test_edge_band_uses_the_preregistered_alpha_window():
    height = width = 40
    ramp = np.linspace(0.0, 1.0, width, dtype=np.float32)
    alpha = np.repeat(ramp[None, :], height, axis=0)
    before = _flat_image((height, width), (0.40, 0.40, 0.40))
    full = _flat_image((height, width), (0.70, 0.40, 0.40))
    after = before * (1.0 - alpha[..., None]) + full * alpha[..., None]

    row = visibility_metrics(before, after, alpha.astype(np.float32), seed_key="chain-e")

    in_band = int(((alpha >= EDGE_BAND_LOW) & (alpha <= EDGE_BAND_HIGH)).sum())
    assert row["n_edge"] == min(in_band, SAMPLE_PIXELS_EDGE)
    assert row["edge_de"] is not None and row["edge_de"] > 0.0
    assert row["edge_step_p95"] is not None and row["edge_step_p95"] >= 0.0
    # a binary mask owns no pixel inside [0.2, 0.8]; the edge columns come back None
    hard = _rect_alpha((height, width), 0, 0, height, width // 2)
    hard_after = np.where(hard[..., None] > 0.5, full, before)
    hard_row = visibility_metrics(before, hard_after, hard, seed_key="chain-h")
    assert hard_row["n_edge"] == 0
    assert hard_row["edge_de"] is None
    assert hard_row["edge_step_p95"] is None


# --------------------------------------------------- support-set sampling fix
def test_support_sampling_is_independent_of_support_frac():
    """The v2 defect: whole-image sampling leaves ~4096*support_frac in-mask points."""
    height = width = 512
    before = _flat_image((height, width), (0.40, 0.40, 0.40))
    after = before.copy()
    alpha = _rect_alpha((height, width), 10, 10, 115, 115)  # 13225 px = 5.05%
    after[10:125, 10:125] = (0.60, 0.40, 0.40)

    row = visibility_metrics(before, after, alpha, seed_key="chain-small")

    support = int((alpha > VISIBILITY_TAU).sum())
    assert support > SAMPLE_PIXELS_IN
    assert row["n_in"] == SAMPLE_PIXELS_IN
    assert row["n_out"] == SAMPLE_PIXELS_OUT

    legacy = _sample_indices(height * width, "0" * 32)
    legacy_in = int((alpha.reshape(-1)[legacy] > VISIBILITY_TAU).sum())
    assert legacy_in < 400          # ~4096 * 0.0505
    assert row["n_in"] > 4 * legacy_in


def test_support_sampling_returns_the_whole_pool_when_small():
    before = _flat_image((256, 256), (0.40, 0.40, 0.40))
    after = before.copy()
    alpha = _rect_alpha((256, 256), 5, 5, 20, 20)  # 400 px
    after[5:25, 5:25] = (0.60, 0.40, 0.40)
    row = visibility_metrics(before, after, alpha, seed_key="chain-tiny")
    assert row["n_in"] == 400


def test_draw_region_is_deterministic_and_budget_capped():
    pool = np.arange(10_000, dtype=np.int64)
    first = draw_region(pool, SAMPLE_PIXELS_IN, "seed-1", "in")
    again = draw_region(pool, SAMPLE_PIXELS_IN, "seed-1", "in")
    other_key = draw_region(pool, SAMPLE_PIXELS_IN, "seed-2", "in")
    other_region = draw_region(pool, SAMPLE_PIXELS_IN, "seed-1", "out")
    assert first.size == SAMPLE_PIXELS_IN
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other_key)
    assert not np.array_equal(first, other_region)
    assert np.array_equal(first, np.sort(first))
    with pytest.raises(VisibilityError):
        draw_region(pool, SAMPLE_BUDGET_MAX + 1, "seed-1", "in")


# --------------------------------------------------------------- random floor
def test_random_area_mask_is_deterministic_and_area_matched():
    first = random_area_mask((300, 200), 0.25, "chain-f")
    again = random_area_mask((300, 200), 0.25, "chain-f")
    other = random_area_mask((300, 200), 0.25, "chain-g")
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)
    assert first.mean() == pytest.approx(0.25, abs=0.01)
    assert other.mean() == pytest.approx(0.25, abs=0.01)
    assert random_area_mask((10, 10), 1.0, "x").mean() == pytest.approx(1.0)
    assert random_area_mask((10, 10), 0.0, "x").sum() == 0.0


def test_floor_columns_are_produced_next_to_the_real_mask_columns():
    before = _flat_image((128, 128), (0.40, 0.40, 0.40))
    alpha = _rect_alpha((128, 128), 20, 20, 40, 40)
    after = before.copy()
    after[:] = (0.44, 0.40, 0.40)          # a weak whole-image leak
    after[20:60, 20:60] = (0.60, 0.40, 0.40)

    row = visibility_metrics(before, after, alpha, seed_key="chain-floor")

    assert row["floor_de_in"] is not None
    assert row["floor_de_out"] is not None
    assert row["floor_de_contrast"] == pytest.approx(
        row["floor_de_in"] - row["floor_de_out"]
    )
    assert row["de_in_over_floor"] == pytest.approx(row["de_in"] / row["floor_de_in"])
    assert row["n_floor_in"] > 0 and row["n_floor_out"] > 0
    assert row["de_in"] > row["floor_de_in"]


def test_floor_ratio_is_none_when_the_random_mask_sees_no_change():
    before = _flat_image((128, 128), (0.40, 0.40, 0.40))
    after = before.copy()
    alpha = _rect_alpha((128, 128), 20, 20, 40, 40)
    after[20:60, 20:60] = (0.60, 0.40, 0.40)

    row = visibility_metrics(before, after, alpha, seed_key="chain-floor")

    assert row["floor_de_in"] == pytest.approx(0.0, abs=1e-6)
    assert row["de_in_over_floor"] is None
    assert row["de_in_minus_floor"] == pytest.approx(row["de_in"])


def test_metrics_are_reproducible_and_seed_keyed():
    rng = np.random.default_rng(7)
    before = rng.random((96, 96, 3)).astype(np.float32)
    after = np.clip(before + 0.05, 0.0, 1.0).astype(np.float32)
    alpha = np.clip(rng.random((96, 96)).astype(np.float32), 0.0, 1.0)

    first = visibility_metrics(before, after, alpha, seed_key="chain-r")
    again = visibility_metrics(before, after, alpha, seed_key="chain-r")
    assert first == again
    other = visibility_metrics(before, after, alpha, seed_key="chain-s")
    assert other["n_in"] == first["n_in"]
    assert other["floor_de_in"] != first["floor_de_in"]


# ------------------------------------------------------------------- contracts
def test_delta_e_at_matches_the_full_field():
    rng = np.random.default_rng(3)
    before = rng.random((24, 24, 3)).astype(np.float32)
    after = np.clip(before + 0.1, 0.0, 1.0).astype(np.float32)
    index = np.array([0, 5, 100, 575], dtype=np.int64)
    field = delta_e_map(before, after).reshape(-1)
    assert delta_e_at(before, after, index) == pytest.approx(field[index], rel=1e-5)
    assert delta_e_at(before, after, np.array([], dtype=np.int64)).size == 0


def test_assert_visibility_columns_catches_missing_and_wrong_contract():
    before = _flat_image((32, 32), (0.40, 0.40, 0.40))
    after = before.copy()
    after[0:8, 0:8] = (0.60, 0.40, 0.40)
    alpha = _rect_alpha((32, 32), 0, 0, 8, 8)
    row = visibility_metrics(before, after, alpha, seed_key="chain-c")
    assert REQUIRED_COLUMNS <= set(row)
    assert_visibility_columns(row)
    with pytest.raises(VisibilityError):
        assert_visibility_columns({k: v for k, v in row.items() if k != "de_contrast"})
    with pytest.raises(VisibilityError):
        assert_visibility_columns({**row, "visibility_contract": "other"})


def test_shape_and_empty_support_are_loud():
    before = _flat_image((16, 16), (0.4, 0.4, 0.4))
    after = before.copy()
    with pytest.raises(VisibilityError):
        visibility_metrics(before, after, np.zeros((16, 16), np.float32), seed_key="z")
    with pytest.raises(VisibilityError):
        visibility_metrics(before, after, np.ones((8, 8), np.float32), seed_key="z")
    with pytest.raises(VisibilityError):
        visibility_metrics(before, _flat_image((8, 8), 0.4),
                           np.ones((16, 16), np.float32), seed_key="z")


# ------------------------------------------------------- D1b Lab decomposition
def test_lab_components_decompose_delta_e_ab_exactly():
    rng = np.random.default_rng(11)
    before = rng.random((16, 16, 3)).astype(np.float32)
    after = rng.random((16, 16, 3)).astype(np.float32)
    index = np.arange(16 * 16, dtype=np.int64)
    lab_before, lab_after = lab_at(before, index), lab_at(after, index)
    parts = lab_components(lab_before, lab_after)
    de_ab = np.linalg.norm(lab_after - lab_before, axis=1)
    total = np.sqrt(parts["dL"] ** 2 + parts["dC"] ** 2 + parts["dH"] ** 2)
    assert total == pytest.approx(de_ab, rel=1e-6, abs=1e-6)


def test_lightness_only_edit_puts_everything_in_dl():
    before = _flat_image((32, 32), (0.40, 0.40, 0.40))
    after = _flat_image((32, 32), (0.60, 0.60, 0.60))
    alpha = np.ones((32, 32), dtype=np.float32)

    row = visibility_components(before, after, alpha, seed_key="chain-l")

    index = np.array([0], dtype=np.int64)
    expected = float(lab_at(after, index)[0, 0] - lab_at(before, index)[0, 0])
    assert row["dL_signed_mean"] == pytest.approx(expected, rel=1e-5)
    assert row["dL_mean"] == pytest.approx(abs(expected), rel=1e-5)
    assert row["dL_p90"] == pytest.approx(abs(expected), rel=1e-5)
    # float32 sRGB neutrals carry ~1e-3 of numerical chroma; that is the floor
    assert row["dC_mean"] == pytest.approx(0.0, abs=1e-2)
    assert row["dHue_mean"] == pytest.approx(0.0, abs=1e-2)
    assert row["dL_grad"] == pytest.approx(0.0, abs=1e-6)
    assert row["dL_step_mean"] == pytest.approx(0.0, abs=1e-6)


def test_signed_mean_records_direction_while_the_magnitude_does_not():
    before = _flat_image((32, 32), (0.40, 0.40, 0.40))
    alpha = np.ones((32, 32), dtype=np.float32)
    up = _flat_image((32, 32), (0.55, 0.55, 0.55))
    down = _flat_image((32, 32), (0.25, 0.25, 0.25))

    brighter = visibility_components(before, up, alpha, seed_key="chain-d")
    darker = visibility_components(before, down, alpha, seed_key="chain-d")

    assert brighter["dL_signed_mean"] > 0.0
    assert darker["dL_signed_mean"] < 0.0
    assert brighter["dL_mean"] > 0.0 and darker["dL_mean"] > 0.0


def test_chroma_only_edit_puts_nothing_in_dl():
    before = _flat_image((32, 32), (0.50, 0.50, 0.50))
    after = before.copy()
    after[:] = (0.55, 0.48, 0.48)
    alpha = np.ones((32, 32), dtype=np.float32)

    row = visibility_components(before, after, alpha, seed_key="chain-c2")

    assert abs(row["dC_signed_mean"]) > 1.0
    assert row["dC_mean"] == pytest.approx(abs(row["dC_signed_mean"]), rel=1e-6)
    assert abs(row["dL_signed_mean"]) < abs(row["dC_signed_mean"])


def test_dl_grad_separates_uniform_from_structured_lightness_change():
    before = _flat_image((32, 32), (0.40, 0.40, 0.40))
    alpha = np.ones((32, 32), dtype=np.float32)
    uniform = _flat_image((32, 32), (0.60, 0.60, 0.60))
    structured = before.copy()
    structured[:, 0::2] = (0.70, 0.70, 0.70)
    structured[:, 1::2] = (0.30, 0.30, 0.30)

    flat = visibility_components(before, uniform, alpha, seed_key="chain-g")
    rough = visibility_components(before, structured, alpha, seed_key="chain-g")

    assert flat["dL_grad"] == pytest.approx(0.0, abs=1e-6)
    assert rough["dL_grad"] > 10.0
    assert flat["dL_step_mean"] == pytest.approx(0.0, abs=1e-6)
    assert rough["dL_step_mean"] > 10.0


def test_highlight_subset_uses_the_before_image_lightness_over_the_support():
    before = _flat_image((32, 32), (0.30, 0.30, 0.30))
    before[0:8, :] = (0.98, 0.98, 0.98)          # L* > 85 band
    after = before.copy()
    after[0:8, :] = (1.00, 1.00, 1.00)           # only the bright band moves
    alpha = np.zeros((32, 32), dtype=np.float32)
    alpha[0:16, :] = 1.0                          # support = bright band + dark band

    row = visibility_components(before, after, alpha, seed_key="chain-h")

    assert row["n_highlight"] == 8 * 32
    assert row["highlight_frac"] == pytest.approx(0.5)
    assert row["dL_highlight_mean"] > 0.5
    assert row["dL_highlight_signed_mean"] == pytest.approx(row["dL_highlight_mean"])
    assert row["dL_mean"] < row["dL_highlight_mean"]


def test_highlight_columns_are_none_when_no_support_pixel_is_bright():
    before = _flat_image((32, 32), (0.30, 0.30, 0.30))
    after = _flat_image((32, 32), (0.45, 0.45, 0.45))
    alpha = np.ones((32, 32), dtype=np.float32)

    row = visibility_components(before, after, alpha, seed_key="chain-nh")

    assert row["n_highlight"] == 0
    assert row["highlight_frac"] == 0.0
    assert row["dL_highlight_mean"] is None
    assert row["dL_highlight_signed_mean"] is None


def test_components_share_the_support_sample_of_the_de00_family():
    rng = np.random.default_rng(5)
    before = rng.random((96, 96, 3)).astype(np.float32)
    after = np.clip(before + 0.05, 0.0, 1.0).astype(np.float32)
    alpha = np.clip(rng.random((96, 96)).astype(np.float32), 0.0, 1.0)

    base = visibility_metrics(before, after, alpha, seed_key="chain-share")
    comp = visibility_components(before, after, alpha, seed_key="chain-share")

    assert comp["n_in"] == base["n_in"] == SAMPLE_PIXELS_IN
    assert comp["support_frac"] == base["support_frac"]
    inside = draw_region(
        np.flatnonzero(alpha.reshape(-1) > VISIBILITY_TAU),
        SAMPLE_PIXELS_IN, "chain-share", "in",
    )
    parts = lab_components(lab_at(before, inside), lab_at(after, inside))
    assert comp["dL_mean"] == pytest.approx(float(np.abs(parts["dL"]).mean()), rel=1e-9)
    assert comp["dL_grad"] == pytest.approx(float(parts["dL"].std()), rel=1e-9)


def test_components_are_reproducible_and_seed_keyed():
    rng = np.random.default_rng(9)
    before = rng.random((96, 96, 3)).astype(np.float32)
    after = np.clip(before + 0.05, 0.0, 1.0).astype(np.float32)
    alpha = np.clip(rng.random((96, 96)).astype(np.float32), 0.0, 1.0)

    first = visibility_components(before, after, alpha, seed_key="chain-p")
    again = visibility_components(before, after, alpha, seed_key="chain-p")
    assert first == again
    other = visibility_components(before, after, alpha, seed_key="chain-q")
    assert other["n_in"] == first["n_in"]
    assert other["dL_mean"] != first["dL_mean"]


def test_assert_component_columns_catches_missing_and_wrong_contract():
    before = _flat_image((32, 32), (0.40, 0.40, 0.40))
    after = before.copy()
    after[0:8, 0:8] = (0.60, 0.40, 0.40)
    alpha = _rect_alpha((32, 32), 0, 0, 8, 8)
    row = visibility_components(before, after, alpha, seed_key="chain-cc")
    assert COMPONENT_COLUMNS <= set(row)
    assert_component_columns(row)
    with pytest.raises(VisibilityError):
        assert_component_columns({k: v for k, v in row.items() if k != "dL_p90"})
    with pytest.raises(VisibilityError):
        assert_component_columns({**row, "components_contract": "other"})


def test_component_shape_and_empty_support_are_loud():
    before = _flat_image((16, 16), (0.4, 0.4, 0.4))
    after = before.copy()
    with pytest.raises(VisibilityError):
        visibility_components(before, after, np.zeros((16, 16), np.float32), seed_key="z")
    with pytest.raises(VisibilityError):
        visibility_components(before, after, np.ones((8, 8), np.float32), seed_key="z")
    with pytest.raises(VisibilityError):
        visibility_components(before, _flat_image((8, 8), 0.4),
                              np.ones((16, 16), np.float32), seed_key="z")
