"""B12: source histogram evidence, LUT histogram response, and their wiring."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dataset_build.agent_loop import prompts as prompts_module
from dataset_build.agent_loop import source_histogram as histogram_module
from dataset_build.agent_loop.candidates import LutRecord
from dataset_build.agent_loop.prompts import (
    HISTOGRAM_ROW_FORMAT, prompt_registry, prompt_revision_fingerprint,
    shortlist_row_text,
)
from dataset_build.agent_loop.segment_fingerprints import (
    HISTOGRAM_AGGREGATES, HISTOGRAM_DERIVATION_REVISION, HISTOGRAM_GROUPS,
    REGISTERED_SEGMENT_FINGERPRINT_TABLES, SEGMENT_FINGERPRINT_SCHEMA,
    SEGMENT_FINGERPRINT_SCHEMA_V2, SEGMENT_FINGERPRINT_TABLE_SHA256,
    SEGMENT_FINGERPRINT_TABLE_SHA256_V2, SegmentFingerprintError,
    derive_histogram_response, load_segment_fingerprints, load_segment_histograms,
    validate_histogram_response, validate_segment_fingerprint_row,
)
from dataset_build.agent_loop.source_histogram import (
    CHROMA_BIN_EDGES, HISTOGRAM_MATCH_GATE, HISTOGRAM_SAMPLE_PIXELS, HUE_SECTOR_COUNT,
    L_BIN_COUNT, SOURCE_HISTOGRAM_CONTRACT, SourceHistogramError,
    assert_histogram_columns, histogram_match_bonus, l_bin_shares, sample_pixels,
    source_histogram, source_histogram_block, source_histogram_text,
)


def _image(seed: int = 7, height: int = 96, width: int = 128) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((height, width, 3), dtype=np.float64)


# ------------------------------------------------------------------ source side
def test_source_histogram_is_deterministic_and_sums_to_one() -> None:
    image = _image()
    first = source_histogram(image)
    second = source_histogram(image.copy())
    assert first == second
    assert first["histogram_contract"] == SOURCE_HISTOGRAM_CONTRACT
    assert first["sample_pixels"] == HISTOGRAM_SAMPLE_PIXELS
    assert len(first["l_bins"]) == L_BIN_COUNT
    assert len(first["c_bins"]) == len(CHROMA_BIN_EDGES) + 1
    assert len(first["hue_sectors"]) == HUE_SECTOR_COUNT
    assert sum(first["l_bins"]) == pytest.approx(1.0, abs=1e-5)
    assert sum(first["c_bins"]) == pytest.approx(1.0, abs=1e-5)
    # The hue sectors only cover the chromatic pixels, i.e. everything above the first
    # chroma bin, so they sum to `1 - c_bins[0]` and never to 1.
    assert sum(first["hue_sectors"]) == pytest.approx(1.0 - first["c_bins"][0], abs=1e-5)


def test_source_histogram_sampling_is_seed_free_and_budget_capped() -> None:
    image = _image(height=200, width=200)
    assert sample_pixels(image).shape == (HISTOGRAM_SAMPLE_PIXELS, 3)
    assert np.array_equal(sample_pixels(image), sample_pixels(image))
    tiny = _image(height=8, width=8)
    assert sample_pixels(tiny).shape == (64, 3)
    with pytest.raises(SourceHistogramError):
        sample_pixels(image, 0)


def test_clip_columns_read_the_pre_registered_l_thresholds() -> None:
    black = np.zeros((16, 16, 3), dtype=np.float64)
    white = np.ones((16, 16, 3), dtype=np.float64)
    assert source_histogram(black)["clip_low"] == 1.0
    assert source_histogram(black)["clip_high"] == 0.0
    assert source_histogram(white)["clip_high"] == 1.0
    assert source_histogram(white)["clip_low"] == 0.0
    assert source_histogram(black)["l_bins"][0] == 1.0
    assert source_histogram(white)["l_bins"][-1] == 1.0


def test_l_bin_shares_are_equal_width_over_zero_to_hundred() -> None:
    values = np.array([0.0, 12.4, 12.6, 50.0, 99.9, 100.0, 137.0, -3.0])
    shares = l_bin_shares(values)
    assert len(shares) == L_BIN_COUNT
    assert sum(shares) == pytest.approx(1.0)
    # -3, 0 and 12.4 fall in bin 0; 137 and 100 clamp into bin 7.
    assert shares[0] == pytest.approx(3 / 8)
    assert shares[7] == pytest.approx(3 / 8)


def test_serialized_line_is_one_line_with_the_frozen_column_layout() -> None:
    row = source_histogram(_image())
    line = source_histogram_text(row)
    assert "\n" not in line
    assert line.startswith(f"source_histogram {SOURCE_HISTOGRAM_CONTRACT}")
    groups = [part.strip() for part in line.split("|")]
    assert len(groups) == 5
    assert len(groups[1].split()) == 1 + L_BIN_COUNT
    assert len(groups[2].split()) == 1 + 2
    assert len(groups[3].split()) == 1 + len(CHROMA_BIN_EDGES) + 1
    assert len(groups[4].split()) == 1 + HUE_SECTOR_COUNT
    assert line == source_histogram_text(dict(reversed(list(row.items()))))
    block = source_histogram_block(row)
    assert block.endswith(line) and block.count("\n") == 1


def test_serialized_line_stays_inside_its_token_budget() -> None:
    line = source_histogram_text(source_histogram(_image()))
    # 20 numbers + 4 labels; ~1 token per 3 characters keeps this under ~70 tokens.
    assert len(line) < 210
    assert len(line) / 3 < 70


def test_assert_histogram_columns_is_a_real_runtime_gate() -> None:
    row = source_histogram(_image())
    assert_histogram_columns(row)
    with pytest.raises(SourceHistogramError, match="columns missing"):
        assert_histogram_columns({k: v for k, v in row.items() if k != "l_bins"})
    with pytest.raises(SourceHistogramError, match="contract mismatch"):
        assert_histogram_columns({**row, "histogram_contract": "other"})
    with pytest.raises(SourceHistogramError, match="exactly"):
        assert_histogram_columns({**row, "l_bins": row["l_bins"][:3]})
    with pytest.raises(SourceHistogramError, match="finite"):
        assert_histogram_columns({**row, "clip_low": None})


# ------------------------------------------------------------------ LUT side
def test_derive_histogram_response_is_the_bin_share_difference() -> None:
    inside = [0.5, 0.2, 0.1, 0.1, 0.05, 0.03, 0.02, 0.0]
    outside = [0.3, 0.2, 0.1, 0.2, 0.05, 0.03, 0.07, 0.05]
    payload = derive_histogram_response(inside, outside)
    assert payload["delta"] == [
        round(b - a, 6) for a, b in zip(inside, outside)
    ]
    assert payload["d_shadow"] == pytest.approx(-0.2)
    assert payload["d_mid"] == pytest.approx(0.1)
    assert payload["d_high"] == pytest.approx(0.1)
    total = sum(payload[name] for name in HISTOGRAM_AGGREGATES)
    assert total == pytest.approx(0.0, abs=1e-9)
    validate_histogram_response(payload)


def test_histogram_groups_partition_every_l_bin_exactly_once() -> None:
    seen = [index for group in HISTOGRAM_GROUPS.values() for index in group]
    assert sorted(seen) == list(range(L_BIN_COUNT))


def test_derive_histogram_response_rejects_a_wrong_bin_count() -> None:
    with pytest.raises(SegmentFingerprintError, match="exactly"):
        derive_histogram_response([0.5, 0.5], [0.5, 0.5])


def _v1_row() -> dict[str, Any]:
    return {
        "schema": SEGMENT_FINGERPRINT_SCHEMA,
        "preset_id": "p0",
        "derivation_revision": "ramp-band-lumsign-v1",
        "source_sha256": "a" * 64,
        "hsl_features_sha256": "b" * 64,
        "bands_per_segment": {"shadows": 1, "mids": 2, "highlights": 5},
        "segments": {
            name: {"dL": 1.0, "dC": 2.0, "d_hue": 3.0, "cast_a": 4.0, "cast_b": 5.0}
            for name in ("shadows", "mids", "highlights")
        },
    }


def _v2_row() -> dict[str, Any]:
    row = _v1_row()
    row["schema"] = SEGMENT_FINGERPRINT_SCHEMA_V2
    row["histogram_revision"] = HISTOGRAM_DERIVATION_REVISION
    row["probe_sha256"] = "c" * 64
    row["histogram"] = derive_histogram_response(
        [0.125] * 8, [0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15, 0.1]
    )
    return row


def test_row_validation_accepts_v1_and_v2_and_rejects_the_mixtures() -> None:
    validate_segment_fingerprint_row(_v1_row())
    validate_segment_fingerprint_row(_v2_row())
    with pytest.raises(SegmentFingerprintError, match="must not carry a histogram"):
        validate_segment_fingerprint_row({**_v1_row(), "histogram": {}})
    with pytest.raises(SegmentFingerprintError, match="histogram_revision"):
        validate_segment_fingerprint_row({**_v2_row(), "histogram_revision": "x"})
    with pytest.raises(SegmentFingerprintError, match="probe_sha256"):
        validate_segment_fingerprint_row({**_v2_row(), "probe_sha256": "short"})
    with pytest.raises(SegmentFingerprintError, match="schema must be one of"):
        validate_segment_fingerprint_row({**_v1_row(), "schema": "lut-v3"})


def _write(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ), encoding="utf-8")
    return path


def test_loaders_are_backward_compatible_with_a_v1_artifact(tmp_path: Path) -> None:
    v1 = _write(tmp_path / "v1.jsonl", [_v1_row()])
    assert set(load_segment_fingerprints(v1)) == {"p0"}
    assert load_segment_histograms(v1) == {}

    v2 = _write(tmp_path / "v2.jsonl", [_v2_row()])
    # The v1 half of a v2 file reads exactly as it did before B12.
    assert load_segment_fingerprints(v2) == load_segment_fingerprints(v1)
    histograms = load_segment_histograms(v2)
    assert set(histograms) == {"p0"}
    assert set(histograms["p0"]) == {
        "l_bins_in", "l_bins_out", "delta", *HISTOGRAM_AGGREGATES
    }

    mixed = _write(tmp_path / "mixed.jsonl", [
        _v1_row(), {**_v2_row(), "preset_id": "p1"},
    ])
    with pytest.raises(SegmentFingerprintError, match="mixed fingerprint schemas"):
        load_segment_histograms(mixed)


def test_production_v2_table_is_the_registered_one() -> None:
    """B12 item 2: the built artifact is what `prompt_registry()` claims."""
    path = Path(
        "/home/bc/data/scratch/lut_reannotate/out/segment_fingerprints.v2.jsonl"
    )
    if not path.is_file():
        pytest.skip("production v2 fingerprint artifact is not present")
    from dataset_build.agent_loop.lut_annotations import file_sha256

    assert file_sha256(path) == SEGMENT_FINGERPRINT_TABLE_SHA256_V2
    histograms = load_segment_histograms(path)
    assert len(histograms) == 4051
    for values in histograms.values():
        # The three aggregates partition a distribution difference, so they cancel
        # up to the artifact's 6-decimal rounding.
        assert sum(values[name] for name in HISTOGRAM_AGGREGATES) == \
            pytest.approx(0.0, abs=1e-5)


def test_registered_tables_carry_both_generations() -> None:
    assert REGISTERED_SEGMENT_FINGERPRINT_TABLES == (
        SEGMENT_FINGERPRINT_TABLE_SHA256, SEGMENT_FINGERPRINT_TABLE_SHA256_V2,
    )
    assert len(set(REGISTERED_SEGMENT_FINGERPRINT_TABLES)) == 2


# ------------------------------------------------------------------ retrieval bonus
def _record(**histogram: float) -> LutRecord:
    return LutRecord(
        preset_id="p0", path="/dev/null", format="lut", name="p0",
        style_major="major", style_minor="minor", scene_affinity=(), de_med=1.0,
        caption="objective", per_probe={}, hsl_features={},
        histogram_response=dict(histogram) or None,
    )


def test_histogram_match_bonus_follows_the_pre_registered_sign_convention() -> None:
    clipped_shadows = {"clip_low": 0.10, "clip_high": 0.0}
    clipped_highlights = {"clip_low": 0.0, "clip_high": 0.10}
    scale = HISTOGRAM_MATCH_GATE["delta_scale"]

    lifts = {"d_shadow": -scale, "d_mid": 0.0, "d_high": 0.0}
    darkens = {"d_shadow": scale, "d_mid": 0.0, "d_high": 0.0}
    assert histogram_match_bonus(clipped_shadows, lifts) == \
        pytest.approx(HISTOGRAM_MATCH_GATE["shadow_weight"])
    assert histogram_match_bonus(clipped_shadows, darkens) == 0.0

    tames = {"d_shadow": 0.0, "d_mid": 0.0, "d_high": -scale}
    blows = {"d_shadow": 0.0, "d_mid": 0.0, "d_high": scale}
    assert histogram_match_bonus(clipped_highlights, tames) == \
        pytest.approx(HISTOGRAM_MATCH_GATE["highlight_weight"])
    assert histogram_match_bonus(clipped_highlights, blows) == 0.0


def test_histogram_match_bonus_saturates_and_gates_on_the_clip_thresholds() -> None:
    scale = HISTOGRAM_MATCH_GATE["delta_scale"]
    lifts = {"d_shadow": -10.0 * scale, "d_mid": 0.0, "d_high": 0.0}
    assert histogram_match_bonus({"clip_low": 0.10}, lifts) == \
        pytest.approx(HISTOGRAM_MATCH_GATE["shadow_weight"])
    below = HISTOGRAM_MATCH_GATE["clip_low_min"] / 2.0
    assert histogram_match_bonus({"clip_low": below}, lifts) == 0.0
    half = {"d_shadow": -0.5 * scale, "d_mid": 0.0, "d_high": 0.0}
    assert histogram_match_bonus({"clip_low": 0.10}, half) == \
        pytest.approx(0.5 * HISTOGRAM_MATCH_GATE["shadow_weight"])


def test_histogram_term_is_exactly_zero_without_either_side() -> None:
    lifts = {"d_shadow": 1.0, "d_mid": 0.0, "d_high": -1.0}
    assert histogram_match_bonus(None, lifts) == 0.0
    assert histogram_match_bonus({"clip_low": 1.0, "clip_high": 1.0}, None) == 0.0
    assert histogram_match_bonus({}, lifts) == 0.0
    assert histogram_match_bonus({"clip_low": 1.0}, {}) == 0.0


def test_unmounted_record_carries_no_histogram_column() -> None:
    plain = _record()
    assert plain.histogram_aggregates() is None
    assert "histogram" not in plain.prompt_view()
    mounted = _record(d_shadow=-0.1, d_mid=0.05, d_high=0.05, delta=[0.0] * 8)
    assert mounted.histogram_aggregates() == {
        "d_shadow": -0.1, "d_mid": 0.05, "d_high": 0.05
    }
    view = mounted.prompt_view()
    assert set(view["histogram"]) == set(HISTOGRAM_AGGREGATES)
    line = shortlist_row_text(0, {**view, "achievable_bins": ["natural"]})
    assert line.endswith(" | -0.100 0.050 0.050")


def test_histogram_row_format_never_emits_negative_zero() -> None:
    line = shortlist_row_text(0, {
        "caption": "objective", "achievable_bins": ["natural"],
        "fingerprint": {}, "histogram": {"d_shadow": -0.0001, "d_mid": 0.0,
                                         "d_high": 0.0},
    })
    assert "-0" not in line
    assert HISTOGRAM_ROW_FORMAT == "{:.3f}"


# ------------------------------------------------------------------ registry
@pytest.mark.parametrize("key", [
    "source_histogram", "segment_fingerprint_histogram", "histogram_match_gate",
])
def test_b12_registry_keys_carry_their_live_values(key: str) -> None:
    value = prompt_registry()[key]
    assert value
    assert json.dumps(value, sort_keys=True)


def test_histogram_match_gate_enters_the_prompt_revision_chain(monkeypatch) -> None:
    before = prompt_revision_fingerprint()
    monkeypatch.setattr(
        prompts_module, "HISTOGRAM_MATCH_GATE",
        {**HISTOGRAM_MATCH_GATE, "shadow_weight": 9.0},
    )
    assert prompt_revision_fingerprint() != before


@pytest.mark.parametrize("name,probe", [
    ("L_BIN_COUNT", 9),
    ("CLIP_LOW_L", 3.0),
    ("CLIP_HIGH_L", 97.0),
    ("HUE_SECTOR_COUNT", 12),
    ("L_BIN_FORMAT", "{:.4f}"),
    ("SOURCE_HISTOGRAM_CONTRACT", "probe-contract"),
])
def test_source_histogram_constants_enter_the_prompt_revision_chain(
    monkeypatch, name: str, probe: object
) -> None:
    before = prompt_revision_fingerprint()
    monkeypatch.setattr(prompts_module, name, probe)
    assert prompt_revision_fingerprint() != before


@pytest.mark.parametrize("name,probe", [
    ("HISTOGRAM_DERIVATION_REVISION", "probe-revision"),
    ("SEGMENT_FINGERPRINT_TABLE_SHA256_V2", "d" * 64),
    ("HISTOGRAM_ROW_FORMAT", "{:.4f}"),
])
def test_lut_histogram_constants_enter_the_prompt_revision_chain(
    monkeypatch, name: str, probe: object
) -> None:
    before = prompt_revision_fingerprint()
    monkeypatch.setattr(prompts_module, name, probe)
    assert prompt_revision_fingerprint() != before


def test_source_histogram_module_has_no_agent_loop_dependency() -> None:
    """The module is a leaf: nothing in it can create an import cycle."""
    source = Path(histogram_module.__file__).read_text(encoding="utf-8")
    assert "\nfrom ." not in source and "\nimport ." not in source
