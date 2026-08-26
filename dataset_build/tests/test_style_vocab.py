"""EPR-036 / EPR-036b: Style Card closed vocabulary + assertion checker.

EPR-036b freezes the vocabulary: ``DEFAULT_VOCAB`` is now ``vocab.v1.json``
(status ``frozen-v1``) and the draft stays on disk untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataset_build.tools.style_vocab_check import (
    AFFINITY_TAG_TO_SCENE, DEFAULT_ANNOTATIONS, DEFAULT_FINGERPRINTS,
    DEFAULT_VOCAB, DRAFT_VOCAB, FEATURE_FIELDS, FROZEN_AFFINITY, OPS,
    SCENE_TAXONOMY_FROZEN, TAG_FIELD_SIZE_BOUNDS, TAG_FIELDS, StyleVocabError,
    accepted_tags, annotation_quality, build_features, check_tags,
    evaluate_term, iter_terms, load_library, load_vocab, main, resolve_affinity,
    term_index, validate_vocab, vocab_sha256,
)

SIZE_BOUNDS = TAG_FIELD_SIZE_BOUNDS


@pytest.fixture(scope="module")
def vocab() -> dict:
    if not DEFAULT_VOCAB.is_file():
        pytest.skip("vocab draft artifact is not present")
    return load_vocab(DEFAULT_VOCAB)


@pytest.fixture(scope="module")
def library() -> dict:
    if not (DEFAULT_ANNOTATIONS.is_file() and DEFAULT_FINGERPRINTS.is_file()):
        pytest.skip("production annotation artifacts are not present")
    return load_library()


# ------------------------------------------------------------ vocab shape
def test_tag_field_sizes_are_inside_the_spec_bands(vocab: dict) -> None:
    for field, (low, high) in SIZE_BOUNDS.items():
        assert low <= len(vocab[field]) <= high, field


def test_semantic_affinity_is_the_frozen_nine_word_list(vocab: dict) -> None:
    """Ruling (a): still_life -> still-life, 'unknown' dropped."""
    assert tuple(vocab["semantic_affinity"]) == FROZEN_AFFINITY
    assert len(vocab["semantic_affinity"]) == 9
    assert "unknown" not in vocab["semantic_affinity"]
    assert "still_life" not in vocab["semantic_affinity"]
    assert "still-life" in vocab["semantic_affinity"]


def test_scene_taxonomy_copy_matches_the_authority_module() -> None:
    """The tool's local copy may not drift from SCENE_TAXONOMY."""
    from dataset_build.agent_loop.lut_annotations import SCENE_TAXONOMY

    assert SCENE_TAXONOMY_FROZEN == tuple(SCENE_TAXONOMY)


def test_affinity_mapping_bridges_to_the_data_word_forms(vocab: dict) -> None:
    """Ruling (a): the map lives in the checker and is mirrored in vocab meta."""
    assert AFFINITY_TAG_TO_SCENE == {"still-life": "still_life"}
    assert vocab["meta"]["affinity_tag_to_scene"] == AFFINITY_TAG_TO_SCENE
    assert resolve_affinity("still-life") == "still_life"
    assert resolve_affinity("portrait") == "portrait"
    for word in vocab["semantic_affinity"]:
        assert resolve_affinity(word) in SCENE_TAXONOMY_FROZEN
    assert evaluate_term(
        {"scene_affinity": ["still_life"]}, vocab["semantic_affinity"]["still-life"]
    )
    assert not evaluate_term(
        {"scene_affinity": ["portrait"]}, vocab["semantic_affinity"]["still-life"]
    )


def test_every_word_is_lowercase_hyphen_style(vocab: dict) -> None:
    for field, word, _term in iter_terms(vocab):
        assert word == word.lower()
        # m1: after ruling (a) every word of all four fields is hyphen-form.
        assert "_" not in word, f"{field}/{word}"
        assert word.replace("-", "").isalnum()


def test_every_word_carries_one_assertion_triple(vocab: dict) -> None:
    for field, word, term in iter_terms(vocab):
        assert set(term) >= {"field", "op", "threshold", "quantile"}
        assert term["op"] in OPS, f"{field}/{word}"


def test_words_are_unique_across_the_four_tag_fields(vocab: dict) -> None:
    index = term_index(vocab)
    assert len(index) == sum(len(vocab[field]) for field in TAG_FIELDS)


def test_validate_vocab_rejects_a_bad_op(vocab: dict) -> None:
    broken = json.loads(json.dumps(vocab))
    word = next(iter(broken["tone_tags"]))
    broken["tone_tags"][word]["op"] = "~="
    with pytest.raises(StyleVocabError, match="unknown op"):
        validate_vocab(broken)


def test_validate_vocab_rejects_a_duplicated_triple(vocab: dict) -> None:
    broken = json.loads(json.dumps(vocab))
    words = list(broken["tone_tags"])
    broken["tone_tags"][words[1]] = dict(broken["tone_tags"][words[0]])
    with pytest.raises(StyleVocabError, match="duplicates"):
        validate_vocab(broken)


def test_validate_vocab_rejects_a_missing_tag_field(vocab: dict) -> None:
    broken = {k: v for k, v in vocab.items() if k != "semantic_risks"}
    with pytest.raises(StyleVocabError, match="missing tag fields"):
        validate_vocab(broken)


# --------------------------------------------------- frozen-v1 bookkeeping
def test_vocab_is_frozen_with_a_registered_sha256(vocab: dict) -> None:
    """Criterion (c): status, ruling record, date and self-registered digest."""
    assert vocab["schema"] == "lut-style-card-vocab-v1"
    assert vocab["status"] == "frozen-v1"
    assert vocab["meta"]["frozen_at"] == "2026-08-24"
    assert set(vocab["meta"]["decisions"]) >= {
        "a_semantic_affinity", "b_semantic_risks_quantiles",
        "c_vivid_ops", "d_annotation_quality",
    }
    assert vocab["meta"]["sha256"] == vocab_sha256(vocab)
    on_disk = json.loads(DEFAULT_VOCAB.read_text(encoding="utf-8"))
    assert vocab_sha256(on_disk) == vocab["meta"]["sha256"]


def test_draft_is_kept_on_disk_and_is_not_the_default_vocab() -> None:
    assert DRAFT_VOCAB.is_file()
    assert DEFAULT_VOCAB.name == "vocab.v1.json"
    assert DRAFT_VOCAB != DEFAULT_VOCAB


def test_risk_quantiles_are_p15_p85_mild_and_p05_p95_strong(vocab: dict) -> None:
    """Ruling (b): the severity quantiles did not move."""
    for word, term in vocab["semantic_risks"].items():
        expected = ("P15", "P85") if word.endswith("-mild") else ("P05", "P95")
        assert term["quantile"] in expected, word


def test_vivid_words_compare_with_a_strict_greater_than(
    vocab: dict, library: dict
) -> None:
    """Ruling (c): '>' at the unchanged P75 threshold 5.3."""
    for word in ("vivid-blues", "vivid-greens"):
        term = vocab["palette_tags"][word]
        assert term["op"] == ">"
        assert term["threshold"] == 5.3
        assert term["quantile"] == "P75"
        # a preset sitting exactly on the point mass must now fail
        boundary = next(
            pid for pid in sorted(library)
            if library[pid][term["field"]] == term["threshold"]
        )
        assert evaluate_term(library[boundary], term) is False
        assert evaluate_term(
            library[boundary], {**term, "op": ">="}
        ) is True


# ------------------------------------------- validate_vocab negative cases
def test_validate_vocab_rejects_an_underscore_word(vocab: dict) -> None:
    """m1."""
    broken = json.loads(json.dumps(vocab))
    word = next(iter(broken["tone_tags"]))
    broken["tone_tags"]["deep_shadows"] = broken["tone_tags"].pop(word)
    with pytest.raises(StyleVocabError, match="underscore"):
        validate_vocab(broken)


def test_validate_vocab_rejects_an_undersized_tag_field(vocab: dict) -> None:
    """m2 (low side)."""
    broken = json.loads(json.dumps(vocab))
    for word in list(broken["tone_tags"])[:3]:
        del broken["tone_tags"][word]
    with pytest.raises(StyleVocabError, match="outside the frozen band"):
        validate_vocab(broken)


def test_validate_vocab_rejects_an_oversized_tag_field(vocab: dict) -> None:
    """m2 (high side)."""
    broken = json.loads(json.dumps(vocab))
    donor = next(iter(broken["palette_tags"].values()))
    for i in range(3):
        broken["palette_tags"][f"filler-word-{i}"] = {**donor, "threshold": 1000 + i}
    with pytest.raises(StyleVocabError, match="outside the frozen band"):
        validate_vocab(broken)


def test_validate_vocab_rejects_an_affinity_list_that_is_not_the_frozen_one(
    vocab: dict,
) -> None:
    """m2 (affinity is an exact list, not a range)."""
    broken = json.loads(json.dumps(vocab))
    del broken["semantic_affinity"]["product"]
    with pytest.raises(StyleVocabError, match="frozen list"):
        validate_vocab(broken)


def test_validate_vocab_rejects_unknown_as_an_affinity_scene(vocab: dict) -> None:
    """Ruling (a): 'unknown' is no longer a legal tag."""
    broken = json.loads(json.dumps(vocab))
    broken["semantic_affinity"]["product"]["threshold"] = "unknown"
    with pytest.raises(StyleVocabError, match="not a legal Style Card tag"):
        validate_vocab(broken)


def test_validate_vocab_rejects_a_scene_outside_the_taxonomy(vocab: dict) -> None:
    """Ruling (a): the mapping must land inside SCENE_TAXONOMY."""
    broken = json.loads(json.dumps(vocab))
    broken["semantic_affinity"]["product"]["threshold"] = "sunset"
    with pytest.raises(StyleVocabError, match="outside the scene taxonomy"):
        validate_vocab(broken)


def test_validate_vocab_rejects_a_non_membership_affinity_assertion(
    vocab: dict,
) -> None:
    broken = json.loads(json.dumps(vocab))
    broken["semantic_affinity"]["product"]["op"] = ">="
    with pytest.raises(StyleVocabError, match="affinity assertion must be"):
        validate_vocab(broken)


def test_validate_vocab_rejects_a_misspelled_feature_field(vocab: dict) -> None:
    """m3: the namespace whitelist catches the typo at load time."""
    broken = json.loads(json.dumps(vocab))
    word = next(iter(broken["tone_tags"]))
    broken["tone_tags"][word]["field"] = "segments.shadow.dL"
    with pytest.raises(StyleVocabError, match="unknown feature field"):
        validate_vocab(broken)


def test_feature_field_whitelist_covers_the_frozen_vocabulary(vocab: dict) -> None:
    """m3: every declared field is inside FEATURE_FIELDS, and the set is closed."""
    for _field, _word, term in iter_terms(vocab):
        assert term["field"] in FEATURE_FIELDS
    assert "scene_affinity" in FEATURE_FIELDS
    assert "segments.shadows.dL" in FEATURE_FIELDS
    assert "segments.shadow.dL" not in FEATURE_FIELDS


def test_validate_vocab_rejects_a_word_reused_across_tag_fields(vocab: dict) -> None:
    """m4."""
    broken = json.loads(json.dumps(vocab))
    word = next(iter(broken["tone_tags"]))
    broken["palette_tags"][word] = dict(broken["tone_tags"][word])
    with pytest.raises(StyleVocabError, match="duplicate word across tag fields"):
        validate_vocab(broken)


def test_validate_vocab_rejects_a_tampered_sha256(vocab: dict) -> None:
    broken = json.loads(json.dumps(vocab))
    broken["meta"]["sha256"] = "0" * 64
    with pytest.raises(StyleVocabError, match="recomputed payload sha256"):
        validate_vocab(broken)


def test_validate_vocab_rejects_a_tampered_affinity_map(vocab: dict) -> None:
    broken = json.loads(json.dumps(vocab))
    broken["meta"]["affinity_tag_to_scene"] = {"still-life": "portrait"}
    with pytest.raises(StyleVocabError, match="affinity_tag_to_scene"):
        validate_vocab(broken)


# --------------------------------------------------- assertion evaluation
def test_every_word_has_a_real_positive_and_negative_row(
    vocab: dict, library: dict
) -> None:
    """Judgement (c): one in-library pass example and one fail example per word."""
    for field, word, term in iter_terms(vocab):
        positive = term["example_pass"]
        negative = term["example_fail"]
        assert positive in library, f"{field}/{word} positive example"
        assert negative in library, f"{field}/{word} negative example"

        pass_rows = check_tags(library[positive], [word], vocab)
        assert pass_rows[0]["pass"] is True, f"{field}/{word} positive"
        assert pass_rows[0]["tag_field"] == field
        assert annotation_quality(pass_rows) == "clean"

        fail_rows = check_tags(library[negative], [word], vocab)
        assert fail_rows[0]["pass"] is False, f"{field}/{word} negative"
        assert fail_rows[0]["reason"] == "assertion-false"
        assert annotation_quality(fail_rows) == "conflicted"


def test_thresholds_match_the_declared_library_quantile(
    vocab: dict, library: dict
) -> None:
    """Judgement (b): the threshold really is that quantile of the 4051 rows."""
    import numpy as np

    assert len(library) == 4051
    for field, word, term in iter_terms(vocab):
        if term["op"] == "contains":
            continue
        pct = float(term["quantile"].lstrip("P"))
        values = np.array(
            [library[pid][term["field"]] for pid in sorted(library)], dtype=float
        )
        assert term["threshold"] == pytest.approx(
            float(np.percentile(values, pct)), abs=1e-4
        ), f"{field}/{word}"


def test_no_word_is_empty_or_saturated(vocab: dict, library: dict) -> None:
    for field, word, term in iter_terms(vocab):
        hits = sum(
            1 for features in library.values() if evaluate_term(features, term)
        )
        assert hits > 0, f"{field}/{word} is empty"
        assert hits / len(library) <= 0.95, f"{field}/{word} is saturated"


def test_strong_risk_is_nested_inside_its_mild_counterpart(
    vocab: dict, library: dict
) -> None:
    risks = vocab["semantic_risks"]
    strongs = [w for w in risks if w.endswith("-strong")]
    assert strongs
    for strong in strongs:
        mild = strong[: -len("-strong")] + "-mild"
        assert mild in risks
        for features in library.values():
            if evaluate_term(features, risks[strong]):
                assert evaluate_term(features, risks[mild]), f"{strong} vs {mild}"


# ------------------------------------------------------ checker behaviour
def test_unknown_word_fails_without_raising(vocab: dict, library: dict) -> None:
    features = library[next(iter(library))]
    rows = check_tags(features, ["not-a-real-tag"], vocab)
    assert rows == [{
        "tag": "not-a-real-tag", "tag_field": None, "assertion": None,
        "value": None, "pass": False, "reason": "unknown-word",
    }]
    assert annotation_quality(rows) == "conflicted"


def test_failed_tags_are_dropped_and_the_record_is_conflicted(
    vocab: dict, library: dict
) -> None:
    """Ruling (d): tag-level drop, record-level 'conflicted'."""
    good = next(iter(vocab["tone_tags"]))
    bad = next(w for w in vocab["tone_tags"] if w != good)
    preset = vocab["tone_tags"][good]["example_pass"]
    rows = check_tags(
        library[preset], [good, bad, "not-a-real-tag"], vocab
    )
    assert rows[0]["pass"] is True
    failed = [row["tag"] for row in rows if not row["pass"]]
    assert "not-a-real-tag" in failed
    assert accepted_tags(rows) == [row["tag"] for row in rows if row["pass"]]
    assert set(accepted_tags(rows)).isdisjoint(failed)
    assert annotation_quality(rows) == "conflicted"


def test_a_fully_passing_record_is_clean_and_keeps_every_tag(
    vocab: dict, library: dict
) -> None:
    """Ruling (d): 'clean' only when every tag passes."""
    word = next(iter(vocab["tone_tags"]))
    preset = vocab["tone_tags"][word]["example_pass"]
    rows = check_tags(library[preset], [word], vocab)
    assert accepted_tags(rows) == [word]
    assert annotation_quality(rows) == "clean"


def test_missing_feature_fails_the_tag_instead_of_crashing(vocab: dict) -> None:
    term = next(iter(vocab["tone_tags"].values()))
    rows = check_tags({"scene_affinity": []}, [
        next(iter(vocab["tone_tags"]))
    ], vocab)
    assert rows[0]["pass"] is False
    assert term["field"] in rows[0]["reason"]


def test_evaluate_term_supports_every_declared_op() -> None:
    features = {"x": 1.0, "scene_affinity": ["night"]}
    assert evaluate_term(features, {"field": "x", "op": ">=", "threshold": 1.0})
    assert evaluate_term(features, {"field": "x", "op": "<=", "threshold": 1.0})
    assert not evaluate_term(features, {"field": "x", "op": ">", "threshold": 1.0})
    assert not evaluate_term(features, {"field": "x", "op": "<", "threshold": 1.0})
    assert evaluate_term(
        features, {"field": "scene_affinity", "op": "contains", "threshold": "night"}
    )


def test_build_features_flattens_the_documented_namespace() -> None:
    annotation = {
        "preset_id": "p0",
        "scene_affinity": ["portrait"],
        "hsl_features": {
            "summary": {"contrast_ratio": 1.1, "sat_pct_mean": -8.0},
            "bands": {"蓝": {"d_sat_pct": -21.1, "d_hue_deg": -0.4}},
            "neutral_ramp": [{"in": 0.15, "L_in": 15.3, "L_out": 6.1}],
        },
    }
    fingerprint = {
        "segments": {"shadows": {"dL": -8.45, "cast_b": 5.05}},
        "histogram": {"d_shadow": -0.0229, "d_mid": 0.0, "d_high": 0.0229},
    }
    features = build_features(annotation, fingerprint)
    assert features["summary.contrast_ratio"] == 1.1
    assert features["bands.blue.d_sat_pct"] == -21.1
    assert features["ramp.p15.L_out"] == 6.1
    assert features["segments.shadows.dL"] == -8.45
    assert features["histogram.d_shadow"] == -0.0229
    assert features["scene_affinity"] == ["portrait"]


def test_build_features_rejects_a_row_without_hsl_features() -> None:
    with pytest.raises(StyleVocabError, match="hsl_features"):
        build_features({"preset_id": "p0"})


def test_cli_reports_per_tag_pass_fail(
    vocab: dict, library: dict, capsys: pytest.CaptureFixture[str]
) -> None:
    word = next(iter(vocab["tone_tags"]))
    preset_id = vocab["tone_tags"][word]["example_pass"]
    code = main([
        "--preset-id", preset_id, "--tags", f"{word},not-a-real-tag", "--json",
    ])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["preset_id"] == preset_id
    assert [row["pass"] for row in payload["tags"]] == [True, False]
    assert payload["annotation_quality"] == "conflicted"


def test_cli_rejects_an_unknown_preset_id(vocab: dict, library: dict) -> None:
    assert main(["--preset-id", "not-a-preset", "--tags", "deep-shadows"]) == 2


# ----------------------------------------------------------- artifact set
def test_deliverables_are_on_disk() -> None:
    root = Path(DEFAULT_VOCAB).parent
    if not root.is_dir():
        pytest.skip("EPR-036 artifact directory is not present")
    for name in (
        "vocab.v1-draft.json", "coverage.v1-draft.json", "coverage.v1-draft.md",
        "RESULT.md", "NOTES.md",
        "vocab.v1.json", "coverage.v1.json", "coverage.v1.md",
        "RESULT_freeze.md", "NOTES_freeze.md",
    ):
        assert (root / name).is_file(), name
