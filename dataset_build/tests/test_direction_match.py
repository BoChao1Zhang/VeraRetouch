"""B10: R6.1 segment fingerprints and R6.2 direction matching."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from dataset_build.agent_loop.candidates import CandidateError, LutCatalog
from dataset_build.agent_loop.config import CatalogConfig
from dataset_build.agent_loop.direction_match import (
    AXIS_NAMES, DirectionMatchError, DirectionVector, SegmentFingerprintTable,
    direction_from_diagnosis, direction_from_text, direction_match_score,
    direction_match_scores, fingerprint_axes, measure_direction,
    measure_tonal_weights, rank_by_direction, uniform_tonal_weights,
)
from dataset_build.agent_loop.lut_annotations import migrate_annotation
from dataset_build.agent_loop.segment_fingerprints import (
    BAND_SEGMENT_LUM_THRESHOLD, DERIVATION_REVISION, SEGMENT_FIELDS, SEGMENT_NAMES,
    SegmentFingerprintError, assign_bands, build_segment_fingerprints,
    derive_segment_fingerprint, load_segment_fingerprints,
    validate_segment_fingerprint_row,
)


BAND_NAMES = ("红", "橙", "黄", "绿", "浅绿", "蓝", "紫", "洋红")


def _hsl_features(
    *, ramp: dict[float, tuple[float, float, float, float]] | None = None,
    bands: dict[str, tuple[float, float, float]] | None = None,
) -> dict:
    """`ramp[in] = (L_in, L_out, a_out, b_out)`, `bands[name] = (sat, hue, lum)`."""
    ramp = ramp or {
        0.15: (15.0, 10.0, 1.0, 3.0), 0.30: (32.0, 30.0, 3.0, 5.0),
        0.50: (53.0, 58.0, -2.0, -6.0),
        0.70: (73.0, 75.0, -1.0, -4.0), 0.85: (87.0, 85.0, -3.0, -8.0),
    }
    bands = bands or {
        name: (float(index) - 3.5, float(index) - 4.0, float(index) * 2.0 - 7.0)
        for index, name in enumerate(BAND_NAMES)
    }
    return {
        "spec_rev": "hsl8-v1", "grid_size": 33,
        "neutral_ramp": [
            {"in": level, "L_in": values[0], "L_out": values[1],
             "a_out": values[2], "b_out": values[3]}
            for level, values in sorted(ramp.items())
        ],
        "bands": {
            name: {"d_sat_pct": values[0], "d_hue_deg": values[1],
                   "d_lum_pct": values[2], "d_sat_pct_mean": values[0],
                   "d_lum_pct_mean": values[2], "d_hue_deg_iqr": 1.0}
            for name, values in bands.items()
        },
        "summary": {
            "mid_gray_a": ramp[0.50][2], "mid_gray_b": ramp[0.50][3],
            "mid_gray_dL": ramp[0.50][1] - ramp[0.50][0], "sat_pct_mean": 0.0,
            "contrast_ratio": 1.0, "shadow_dL": ramp[0.15][1] - ramp[0.15][0],
            "highlight_dL": ramp[0.85][1] - ramp[0.85][0], "hue_rot_abs_max": 4.0,
        },
    }


def _flat_fingerprint(**overrides: float) -> dict[str, dict[str, float]]:
    base = dict.fromkeys(SEGMENT_FIELDS, 0.0)
    base.update(overrides)
    return {segment: dict(base) for segment in SEGMENT_NAMES}


# --- R6.1 derivation ----------------------------------------------------------------

def test_segment_fingerprint_follows_the_documented_ramp_and_band_recipe() -> None:
    fingerprint = derive_segment_fingerprint(_hsl_features())
    assert set(fingerprint) == set(SEGMENT_NAMES)
    assert all(set(values) == set(SEGMENT_FIELDS) for values in fingerprint.values())
    # dL / cast come from the ramp points of the segment.
    assert fingerprint["shadows"]["dL"] == pytest.approx(((10 - 15) + (30 - 32)) / 2)
    assert fingerprint["mids"]["dL"] == pytest.approx(58 - 53)
    assert fingerprint["highlights"]["dL"] == pytest.approx(((75 - 73) + (85 - 87)) / 2)
    assert fingerprint["shadows"]["cast_a"] == pytest.approx((1.0 + 3.0) / 2)
    assert fingerprint["shadows"]["cast_b"] == pytest.approx((3.0 + 5.0) / 2)
    assert fingerprint["mids"]["cast_a"] == pytest.approx(-2.0)
    assert fingerprint["highlights"]["cast_b"] == pytest.approx((-4.0 + -8.0) / 2)
    # Bands land by the sign of d_lum_pct: index 0..7 -> lum -7,-5,-3,-1,1,3,5,7.
    features = _hsl_features()
    assignment = assign_bands(features["bands"])
    assert [len(assignment[name]) for name in SEGMENT_NAMES] == [3, 2, 3]
    shadow_bands = [BAND_NAMES[i] for i in (0, 1, 2)]
    assert sorted(assignment["shadows"]) == sorted(shadow_bands)
    assert fingerprint["shadows"]["dC"] == pytest.approx(
        sum(float(i) - 3.5 for i in (0, 1, 2)) / 3
    )
    assert fingerprint["highlights"]["d_hue"] == pytest.approx(
        sum(float(i) - 4.0 for i in (5, 6, 7)) / 3
    )
    assert BAND_SEGMENT_LUM_THRESHOLD == 2.0


def test_an_empty_segment_falls_back_to_the_mean_over_all_bands() -> None:
    bands = {name: (2.0, 4.0, 10.0) for name in BAND_NAMES}
    fingerprint = derive_segment_fingerprint(_hsl_features(bands=bands))
    assignment = assign_bands(_hsl_features(bands=bands)["bands"])
    assert assignment["shadows"] == [] and assignment["mids"] == []
    assert len(assignment["highlights"]) == len(BAND_NAMES)
    for segment in SEGMENT_NAMES:
        assert fingerprint[segment]["dC"] == pytest.approx(2.0)
        assert fingerprint[segment]["d_hue"] == pytest.approx(4.0)


def test_derivation_rejects_a_truncated_ramp_or_missing_bands() -> None:
    features = _hsl_features()
    features["neutral_ramp"] = features["neutral_ramp"][:3]
    with pytest.raises(SegmentFingerprintError, match="missing input levels"):
        derive_segment_fingerprint(features)
    features = _hsl_features()
    features["bands"] = {}
    with pytest.raises(SegmentFingerprintError, match="bands"):
        derive_segment_fingerprint(features)


def _write_annotations(path: Path, count: int) -> None:
    rows = []
    for index in range(count):
        ramp = {
            0.15: (15.0, 15.0 - index, 1.0, 3.0), 0.30: (32.0, 30.0, 3.0, 5.0),
            0.50: (53.0, 53.0 + index, -2.0, -6.0),
            0.70: (73.0, 75.0, -1.0, -4.0), 0.85: (87.0, 85.0, -3.0, -8.0),
        }
        rows.append(migrate_annotation({
            "preset_id": f"p{index}", "ok": True, "name": f"preset {index}",
            "style_major": "major", "style_minor": f"minor {index}",
            "scene_affinity": "general", "caption": f"caption {index}",
            "per_probe": {}, "hsl_features": _hsl_features(ramp=ramp),
        }, float(index + 3), annotations_source_sha256="a" * 64,
            perceptual_de_source_sha256="b" * 64))
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_segment_fingerprint_artifact_is_deterministic_and_reloads(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.closed-v1.jsonl"
    _write_annotations(annotations, 5)
    output = tmp_path / "segment_fingerprints.v1.jsonl"
    first = build_segment_fingerprints(annotations, output)
    payload = output.read_bytes()
    second = build_segment_fingerprints(annotations, output)
    assert output.read_bytes() == payload
    assert first["output"]["sha256"] == second["output"]["sha256"]
    assert first["output"]["records"] == 5
    assert first["derivation_revision"] == DERIVATION_REVISION
    assert set(first["distribution"]) == {
        f"{segment}.{field}" for segment in SEGMENT_NAMES for field in SEGMENT_FIELDS
    }
    loaded = load_segment_fingerprints(output)
    assert set(loaded) == {f"p{index}" for index in range(5)}
    assert loaded["p3"] == derive_segment_fingerprint(
        json.loads(annotations.read_text(encoding="utf-8").splitlines()[3])["hsl_features"]
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["preset_id"] for row in rows] == sorted(row["preset_id"] for row in rows)
    assert {row["source_sha256"] for row in rows} == {
        first["inputs"]["annotations_sha256"]
    }


def test_a_foreign_derivation_revision_is_rejected(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.closed-v1.jsonl"
    _write_annotations(annotations, 2)
    output = tmp_path / "fingerprints.jsonl"
    build_segment_fingerprints(annotations, output)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    rows[0]["derivation_revision"] = "someone-elses-v9"
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    with pytest.raises(SegmentFingerprintError, match="derivation_revision"):
        load_segment_fingerprints(output)
    with pytest.raises(SegmentFingerprintError, match="segments"):
        validate_segment_fingerprint_row({
            "schema": "lut-segment-fingerprint-v1",
            "derivation_revision": DERIVATION_REVISION, "preset_id": "p0",
            "source_sha256": "a" * 64, "hsl_features_sha256": "b" * 64,
            "segments": {"mids": {}},
        })


# --- catalog mounting ---------------------------------------------------------------

def _catalog_fixture(root: Path, *, fingerprints: Path | None = None) -> CatalogConfig:
    bank = root / "bank"
    bank.mkdir(exist_ok=True)
    annotations = root / "annotations.closed-v1.jsonl"
    _write_annotations(annotations, 3)
    features = []
    for index in range(3):
        cube = root / f"p{index}.cube"
        cube.write_text(
            "LUT_3D_SIZE 2\n" + "".join(
                f"{red} {green} {blue}\n"
                for blue in (0.0, 1.0) for green in (0.0, 1.0) for red in (0.0, 1.0)
            ), encoding="ascii",
        )
        features.append({
            "preset_id": f"p{index}", "path": str(cube), "kind": "lut", "fmt": "cube",
        })
    (bank / "features.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in features), encoding="utf-8"
    )
    (root / "databuild.toml").write_text(
        f'[presets]\nbank_dir = "{bank}"\n', encoding="utf-8"
    )
    return CatalogConfig(
        annotations=annotations, global_major_limit=3, global_per_major_limit=7,
        local_limit=9, segment_fingerprints=fingerprints,
    )


def test_catalog_mounts_fingerprints_only_when_configured(tmp_path: Path) -> None:
    databuild = tmp_path / "databuild.toml"
    unmounted = LutCatalog.load(_catalog_fixture(tmp_path), databuild)
    assert [row.segment_fingerprint for row in unmounted.records] == [None] * 3
    assert unmounted.segment_fingerprints_mounted is False
    with pytest.raises(CandidateError, match="not mounted"):
        unmounted.segment_fingerprint_table()

    fingerprints = tmp_path / "fingerprints.jsonl"
    build_segment_fingerprints(tmp_path / "annotations.closed-v1.jsonl", fingerprints)
    mounted = LutCatalog.load(
        _catalog_fixture(tmp_path, fingerprints=fingerprints), databuild
    )
    assert mounted.segment_fingerprints_mounted is True
    assert mounted.get("p1").segment_fingerprint == load_segment_fingerprints(
        fingerprints
    )["p1"]
    # The unmounted rows are otherwise identical, so nothing else in the catalog moved.
    for before, after in zip(unmounted.records, mounted.records, strict=True):
        assert before.prompt_view() == after.prompt_view()
        assert before.fingerprint() == after.fingerprint()

    table = mounted.segment_fingerprint_table()
    assert table is mounted.segment_fingerprint_table()
    assert table.preset_ids == ("p0", "p1", "p2")
    assert table.values.shape == (3, len(SEGMENT_NAMES), len(SEGMENT_FIELDS))


def test_a_fingerprint_artifact_that_misses_a_preset_is_rejected(tmp_path: Path) -> None:
    fingerprints = tmp_path / "fingerprints.jsonl"
    config = _catalog_fixture(tmp_path, fingerprints=fingerprints)
    build_segment_fingerprints(config.annotations, fingerprints)
    kept = [
        line for line in fingerprints.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["preset_id"] != "p2"
    ]
    fingerprints.write_text("".join(line + "\n" for line in kept), encoding="utf-8")
    with pytest.raises(CandidateError, match="does not cover p2"):
        LutCatalog.load(config, tmp_path / "databuild.toml")


# --- R6.2 direction vectors ---------------------------------------------------------

def test_direction_from_text_reads_the_pre_registered_axes() -> None:
    warm = direction_from_text("整体偏黄，肤色发黄")
    assert (warm.cast_b, warm.cast_a) == (1.0, 0.0)
    assert warm.mode == "correction" and warm.origin == "keywords"
    cool = direction_from_text("a cool blue cast in the shadows")
    assert cool.cast_b == -1.0
    flat = direction_from_text("flat and muted, 对比不足")
    assert (flat.saturation, flat.contrast) == (-1.0, -1.0)
    hot = direction_from_text("过饱和，对比过强，过曝")
    assert (hot.saturation, hot.contrast, hot.lightness) == (1.0, 1.0, 1.0)
    assert direction_from_text("nothing relevant here").is_zero()
    with pytest.raises(DirectionMatchError, match="unknown direction mode"):
        direction_from_text("x", "whatever")


def test_direction_from_diagnosis_reads_the_field_of_its_mode() -> None:
    diagnosis = {
        "correction_needs": ["整体偏黄", "欠曝"],
        "enhancement_opportunities": ["可以更冷调一些"],
    }
    correction = direction_from_diagnosis(diagnosis, "correction")
    assert (correction.cast_b, correction.lightness) == (1.0, -1.0)
    assert correction.mode == "correction"
    enhancement = direction_from_diagnosis(diagnosis, "enhancement")
    assert enhancement.cast_b == -1.0 and enhancement.lightness == 0.0
    assert enhancement.mode == "enhancement"
    assert direction_from_diagnosis({}, "correction").is_zero()


def test_correction_rewards_the_opposite_cast_and_enhancement_the_same_one() -> None:
    weights = uniform_tonal_weights()
    warm_lut = _flat_fingerprint(cast_b=6.0)
    cool_lut = _flat_fingerprint(cast_b=-6.0)
    yellow_cast = direction_from_text("整体偏黄")

    warm_correction = direction_match_score(yellow_cast, warm_lut, weights)
    cool_correction = direction_match_score(yellow_cast, cool_lut, weights)
    assert warm_correction == pytest.approx(-1.0)
    assert cool_correction == pytest.approx(1.0)

    wanted_warmth = direction_from_text("可以更暖调", "enhancement")
    assert direction_match_score(wanted_warmth, warm_lut, weights) == pytest.approx(1.0)
    assert direction_match_score(wanted_warmth, cool_lut, weights) == pytest.approx(-1.0)

    # A LUT that only moves an unrelated axis is orthogonal, not opposed.
    lifter = _flat_fingerprint(dL=6.0)
    assert direction_match_score(yellow_cast, lifter, weights) == pytest.approx(0.0)
    assert direction_match_score(yellow_cast, _flat_fingerprint(), weights) == 0.0
    assert direction_match_score(DirectionVector(), warm_lut, weights) == 0.0


def test_the_contrast_axis_is_a_cross_segment_difference_gated_by_tonal_weight() -> None:
    booster = _flat_fingerprint()
    booster["shadows"]["dL"] = -6.0
    booster["highlights"]["dL"] = 6.0
    flattener = _flat_fingerprint()
    flattener["shadows"]["dL"] = 6.0
    flattener["highlights"]["dL"] = -6.0
    hazy = direction_from_text("对比不足")
    spread = uniform_tonal_weights()
    assert direction_match_score(hazy, booster, spread) == pytest.approx(1.0)
    assert direction_match_score(hazy, flattener, spread) == pytest.approx(-1.0)
    # A mask that lives entirely in the mids has no contrast to read.
    mids_only = {"shadows": 0.0, "mids": 1.0, "highlights": 0.0}
    assert direction_match_score(hazy, booster, mids_only) == 0.0
    lopsided = {"shadows": 0.01, "mids": 0.49, "highlights": 0.50}
    assert 0.0 < direction_match_score(hazy, booster, lopsided) <= 1.0


def test_tonal_weights_reweight_the_segments_and_are_validated() -> None:
    fingerprint = _flat_fingerprint()
    fingerprint["shadows"]["cast_b"] = 9.0
    fingerprint["highlights"]["cast_b"] = -9.0
    table = SegmentFingerprintTable.from_mapping({"a": fingerprint})
    shadows = fingerprint_axes(table, {"shadows": 1.0})[0, AXIS_NAMES.index("cast_b")]
    highlights = fingerprint_axes(
        table, {"highlights": 1.0}
    )[0, AXIS_NAMES.index("cast_b")]
    assert shadows == pytest.approx(9.0)
    assert highlights == pytest.approx(-9.0)
    yellow = direction_from_text("偏黄")
    assert direction_match_score(yellow, fingerprint, {"shadows": 1.0}) < 0.0
    assert direction_match_score(yellow, fingerprint, {"highlights": 1.0}) > 0.0
    for bad in ({"shadows": 0.0}, {"shadows": -1.0}, {"midtones": 1.0}):
        with pytest.raises(DirectionMatchError):
            direction_match_score(yellow, fingerprint, bad)


def test_single_and_batch_scoring_agree_and_ranking_is_deterministic() -> None:
    fingerprints = {
        "cool": _flat_fingerprint(cast_b=-6.0),
        "warm": _flat_fingerprint(cast_b=6.0),
        "twin": _flat_fingerprint(cast_b=-6.0),
        "lift": _flat_fingerprint(dL=4.0),
    }
    table = SegmentFingerprintTable.from_mapping(fingerprints)
    weights = uniform_tonal_weights()
    direction = direction_from_text("偏黄")
    batch = direction_match_scores(direction, table, weights)
    for index, preset_id in enumerate(table.preset_ids):
        assert batch[index] == pytest.approx(
            direction_match_score(direction, fingerprints[preset_id], weights)
        )
    ranked = rank_by_direction(direction, table, weights, 2)
    assert [preset_id for preset_id, _score in ranked] == ["cool", "twin"]
    assert rank_by_direction(direction, table, weights, 4)[-1][0] == "warm"
    with pytest.raises(DirectionMatchError, match="exactly"):
        SegmentFingerprintTable.from_mapping({"x": {"mids": {}}})


def test_scoring_the_whole_catalog_stays_under_ten_milliseconds() -> None:
    rng = np.random.default_rng(20260820)
    values = rng.normal(0.0, 6.0, size=(4051, len(SEGMENT_NAMES), len(SEGMENT_FIELDS)))
    table = SegmentFingerprintTable(
        tuple(f"rcp_{index:06d}" for index in range(4051)), values
    )
    direction = direction_from_text("整体偏黄，对比不足")
    weights = uniform_tonal_weights()
    direction_match_scores(direction, table, weights)
    elapsed = []
    for _ in range(5):
        start = time.perf_counter()
        scores = direction_match_scores(direction, table, weights)
        elapsed.append(time.perf_counter() - start)
    assert scores.shape == (4051,)
    assert float(np.abs(scores).max()) <= 1.0 + 1e-9
    assert min(elapsed) < 0.010


# --- R6.2 two-image measurement -----------------------------------------------------

def _gray(shape: tuple[int, int], level: float) -> np.ndarray:
    return np.full((*shape, 3), level, dtype=np.float32)


def test_measure_direction_reads_the_a_to_b_change() -> None:
    before = _gray((64, 64), 0.5)
    warmer = before.copy()
    warmer[:, :, 2] = 0.4
    measured = measure_direction(before, warmer)
    assert measured.origin == "measured" and measured.mode == "correction"
    assert measured.cast_b > 1.0
    assert measured.cast_b > 2.0 * abs(measured.cast_a)
    # A uniform mid gray has neither a shadow nor a highlight population to compare.
    assert measured.contrast == 0.0
    brighter = measure_direction(before, _gray((64, 64), 0.7))
    assert brighter.lightness > 5.0
    assert measure_direction(before, before.copy()).is_zero()
    assert measure_direction(before, warmer) == measure_direction(before, warmer)
    with pytest.raises(DirectionMatchError, match="does not match"):
        measure_direction(before, _gray((32, 32), 0.5))


def test_measure_direction_weights_by_the_mask() -> None:
    before = _gray((64, 64), 0.5)
    after = before.copy()
    after[:, :32, 2] = 0.2
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[:, :32] = 1.0
    whole_frame = measure_direction(before, after)
    inside = measure_direction(before, after, mask)
    outside = measure_direction(before, after, 1.0 - mask)
    assert inside.cast_b > whole_frame.cast_b > 0.0
    assert outside.is_zero()
    assert inside.cast_b == pytest.approx(2.0 * whole_frame.cast_b, rel=0.05)
    with pytest.raises(DirectionMatchError, match="mask shape"):
        measure_direction(before, after, np.zeros((8, 8), dtype=np.float32))


def test_measured_contrast_splits_the_frame_by_source_lightness() -> None:
    before = np.concatenate(
        [_gray((32, 64), 0.2), _gray((32, 64), 0.8)], axis=0
    )
    after = np.concatenate(
        [_gray((32, 64), 0.1), _gray((32, 64), 0.9)], axis=0
    )
    boosted = measure_direction(before, after)
    assert boosted.contrast > 5.0
    assert measure_direction(after, before).contrast < -5.0
    # Both segments moving the same way is lightness, not contrast.
    lifted = measure_direction(
        before, np.concatenate([_gray((32, 64), 0.3), _gray((32, 64), 0.9)], axis=0)
    )
    assert lifted.lightness > 0.0
    assert abs(lifted.contrast) < abs(boosted.contrast)


def test_measure_tonal_weights_reads_the_mask_region() -> None:
    image = np.concatenate(
        [_gray((32, 64), 0.2), _gray((32, 64), 0.5), _gray((32, 64), 0.8)], axis=0
    )
    weights = measure_tonal_weights(image)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert set(weights) == set(SEGMENT_NAMES)
    for name in SEGMENT_NAMES:
        assert weights[name] == pytest.approx(1 / 3, abs=0.01)
    mask = np.zeros((96, 64), dtype=np.float32)
    mask[64:, :] = 1.0
    highlights_only = measure_tonal_weights(image, mask)
    assert highlights_only["highlights"] == pytest.approx(1.0)
    empty = np.zeros((96, 64), dtype=np.float32)
    with pytest.raises(DirectionMatchError, match="eligible"):
        measure_tonal_weights(image, empty)


def test_a_measured_residual_cast_selects_the_opposite_lut() -> None:
    """R6.2 end to end: source vs global_after -> direction -> fingerprint ranking."""
    source = _gray((48, 48), 0.5)
    global_after = source.copy()
    global_after[:, :, 2] = 0.35
    residual = measure_direction(source, global_after)
    assert residual.cast_b > 0.0
    fingerprints = {
        "cooling": _flat_fingerprint(cast_b=-5.0),
        "warming": _flat_fingerprint(cast_b=5.0),
        "neutral": _flat_fingerprint(dL=1.0),
    }
    table = SegmentFingerprintTable.from_mapping(fingerprints)
    weights = measure_tonal_weights(global_after)
    ranked = rank_by_direction(residual, table, weights, 3)
    assert [preset_id for preset_id, _score in ranked] == [
        "cooling", "neutral", "warming"
    ]
    assert ranked[0][1] > 0.0 > ranked[-1][1]
