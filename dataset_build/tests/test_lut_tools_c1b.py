"""C1b fixes in the questionnaire / metric / clustering tools.

Only the pure, DB-free surfaces are covered here: the shared `winner_confidence`
accounting, the rating-CSV id column, the Spearman floor, and the preset-bank
fingerprint that now enters every metric manifest.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataset_build.tools.export_agent_loop_review import (
    winner_confidence_counts, winner_confidence_warning,
)
from dataset_build.tools.lut_render_distance import (
    SPEARMAN_MIN_PAIRS, features_inputs, features_jsonl_path, read_ratings,
)


# --------------------------------------------------------------- winner_confidence


def test_winner_confidence_counts_bucket_missing_values_as_unknown() -> None:
    counts = winner_confidence_counts(["low", "low", "normal", None, ""])
    assert counts == {"low": 2, "normal": 1, "unknown": 2}


def test_winner_confidence_warning_fires_on_low_and_unknown_only() -> None:
    """C1b items 3-5: the builders do not filter, so the warning is the whole guard."""
    assert winner_confidence_warning({"normal": 7}, filtered=False) == ""
    assert winner_confidence_warning({}, filtered=False) == ""
    low = winner_confidence_warning({"low": 3, "normal": 1}, filtered=False)
    assert "3/4" in low and "filtering is OFF" in low
    unknown = winner_confidence_warning({"unknown": 2}, filtered=True)
    assert "2/2" in unknown and "filtering is ON" in unknown


def test_every_questionnaire_builder_declares_its_filter_stance() -> None:
    from dataset_build.tools import (
        chain_quality_questionnaire, intent_quality_questionnaire,
        recheck_questionnaire,
    )

    for module in (chain_quality_questionnaire, intent_quality_questionnaire,
                   recheck_questionnaire):
        assert module.WINNER_CONFIDENCE_FILTERED is False
    # C1b item 5: a recheck bundle that drops the field cannot be audited later.
    assert "winner_confidence" in recheck_questionnaire.CARRIED_FIELDS


# ------------------------------------------------------------------- rating CSV


def _csv(path: Path, header: str, *rows: str) -> Path:
    path.write_text(header + "\n" + "".join(row + "\n" for row in rows), encoding="utf-8")
    return path


def test_read_ratings_accepts_pair_id_and_item_id(tmp_path: Path) -> None:
    """C1b item 8: the questionnaire builders emit `item_id`, not `pair_id`."""
    pair = _csv(tmp_path / "pair.csv", "pair_id,rating,notes", "p1,4,", "p2,2,x")
    item = _csv(tmp_path / "item.csv", "item_id,rating,notes", "c001,5,", "c002,,skip")
    assert read_ratings([pair]) == {"p1": 4, "p2": 2}
    assert read_ratings([item]) == {"c001": 5}
    assert read_ratings([pair, item]) == {"p1": 4, "p2": 2, "c001": 5}


def test_read_ratings_rejects_a_csv_with_neither_id_column(tmp_path: Path) -> None:
    path = _csv(tmp_path / "bad.csv", "id,rating", "x,3")
    with pytest.raises(SystemExit, match="no pair_id/item_id column"):
        read_ratings([path])


def test_read_ratings_still_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = _csv(tmp_path / "dup.csv", "item_id,rating", "c001,3", "c001,4")
    with pytest.raises(SystemExit, match="duplicate rating"):
        read_ratings([path])


# --------------------------------------------------------------------- Spearman


def test_spearman_floor_is_three_pairs() -> None:
    """C1b item 9: below the floor the calibration reports null, never (0.0, 1.0)."""
    assert SPEARMAN_MIN_PAIRS == 3
    source = Path(
        "dataset_build/tools/lut_render_distance.py"
    ).read_text(encoding="utf-8")
    assert "insufficient_pairs" in source
    assert "(0.0, 1.0)" not in source


# ------------------------------------------------------------- preset bank hash


def _databuild(tmp_path: Path, bank: Path) -> Path:
    path = tmp_path / "databuild.toml"
    path.write_text(f'[presets]\nbank_dir = "{bank}"\n', encoding="utf-8")
    return path


def test_features_inputs_fingerprints_the_preset_bank(tmp_path: Path) -> None:
    """C1b item 11: `annotations_sha256` alone is half of the catalog identity."""
    bank = tmp_path / "bank"
    bank.mkdir()
    features = bank / "features.jsonl"
    features.write_text(
        "".join(json.dumps({"preset_id": f"p{index}", "fmt": "lut"}) + "\n"
                for index in range(3)),
        encoding="utf-8",
    )
    databuild = _databuild(tmp_path, bank)
    assert features_jsonl_path(databuild) == features
    inputs = features_inputs(databuild)
    assert inputs["features_jsonl"] == str(features)
    assert len(inputs["features_jsonl_sha256"]) == 64
    assert inputs["features_jsonl_rows"] == 3

    features.write_text(
        features.read_text(encoding="utf-8")
        + json.dumps({"preset_id": "p3", "fmt": "lut"}) + "\n",
        encoding="utf-8",
    )
    assert features_inputs(databuild)["features_jsonl_sha256"] != \
        inputs["features_jsonl_sha256"]


def test_features_inputs_reports_a_missing_bank_as_null(tmp_path: Path) -> None:
    inputs = features_inputs(_databuild(tmp_path, tmp_path / "absent"))
    assert inputs["features_jsonl_sha256"] is None
    assert inputs["features_jsonl_rows"] is None


def test_every_metric_manifest_carries_the_bank_fingerprint() -> None:
    for name in ("lut_render_distance", "lut_metric_ab", "lut_metric_p1",
                 "cluster_lut_effects"):
        source = Path(f"dataset_build/tools/{name}.py").read_text(encoding="utf-8")
        assert "features_inputs(databuild)" in source, name
