"""Style Card closed-vocabulary numeric assertion checker (EPR-036).

One tag word == one assertion triple ``(field, op, threshold)``.  A tag is
*admissible* for a preset iff the triple evaluates true on that preset's
numeric features.  Nothing here calls an LLM and nothing here mutates the
source tables.

Feature namespace (flat dotted keys, built by :func:`build_features`):

``summary.<k>``          the 8 keys of ``hsl_features.summary``
``bands.<en>.<k>``       8 HSL bands (Chinese keys romanised via ``BAND_EN``)
                         x {d_hue_deg, d_hue_deg_iqr, d_lum_pct,
                            d_lum_pct_mean, d_sat_pct, d_sat_pct_mean}
``ramp.p<NN>.<k>``       the 5 ``hsl_features.neutral_ramp`` points keyed by
                         their ``in`` value x {L_in, L_out, a_out, b_out}
``segments.<seg>.<k>``   ``segment_fingerprints.v2`` segments
                         {shadows, mids, highlights} x {dL, dC, d_hue,
                         cast_a, cast_b}
``histogram.<k>``        ``segment_fingerprints.v2`` histogram aggregates
                         {d_shadow, d_mid, d_high}
``scene_affinity``       the closed scene list of the annotation row

The enumeration of legal ``term["field"]`` paths is :data:`FEATURE_FIELDS`;
``validate_vocab`` rejects any word whose field is outside it, so a typo fails
at load time instead of failing silently on every row.

Affinity word forms: the Style Card vocabulary is all-hyphen, the data side
(``scene_affinity``, authority ``dataset_build/agent_loop/lut_annotations.py``
``SCENE_TAXONOMY``) still carries the original word forms.  The bridge is the
explicit table :data:`AFFINITY_TAG_TO_SCENE` (``still-life`` -> ``still_life``),
mirrored into ``vocab["meta"]["affinity_tag_to_scene"]``.  ``unknown`` is a
data-side scene word but **not** a legal Style Card tag (frozen-v1 ruling).

Hue-sign convention (``tools/lut_reannotate/hslfeat.py:37-51``): band centres
are red 0 / orange 30 / yellow 60 / green 120 / cyan 180 / blue 240 /
purple 270 / magenta 300 on the sRGB HLS wheel and ``d_hue_deg`` is the
wrapped ``hue_out - hue_in``.  Positive therefore means rotation toward the
next band clockwise on that list (green -> cyan, blue -> purple,
orange -> yellow); negative means the reverse (orange -> red/magenta,
blue -> cyan, green -> yellow).

CLI::

    python -m dataset_build.tools.style_vocab_check \\
        --preset-id rcp_0001f94f36ca4af3 --tags deep-shadows,cool-highlights
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

DEFAULT_ANNOTATIONS = Path(
    "/home/bc/data/scratch/lut_reannotate/out/annotations.closed-v1.jsonl"
)
DEFAULT_FINGERPRINTS = Path(
    "/home/bc/data/scratch/lut_reannotate/out/segment_fingerprints.v2.jsonl"
)
VOCAB_DIR = Path(__file__).resolve().parents[2] / "experiments/prs/EPR-036_style-vocab"
DRAFT_VOCAB = VOCAB_DIR / "vocab.v1-draft.json"
DEFAULT_VOCAB = VOCAB_DIR / "vocab.v1.json"

BAND_EN: dict[str, str] = {
    "红": "red",
    "橙": "orange",
    "黄": "yellow",
    "绿": "green",
    "浅绿": "cyan",
    "蓝": "blue",
    "紫": "purple",
    "洋红": "magenta",
}
BAND_METRICS = (
    "d_hue_deg", "d_hue_deg_iqr", "d_lum_pct",
    "d_lum_pct_mean", "d_sat_pct", "d_sat_pct_mean",
)
SUMMARY_KEYS = (
    "contrast_ratio", "highlight_dL", "hue_rot_abs_max", "mid_gray_a",
    "mid_gray_b", "mid_gray_dL", "sat_pct_mean", "shadow_dL",
)
RAMP_KEYS = ("L_in", "L_out", "a_out", "b_out")
RAMP_STOPS = ("p15", "p30", "p50", "p70", "p85")
SEGMENTS = ("shadows", "mids", "highlights")
SEGMENT_METRICS = ("dL", "dC", "d_hue", "cast_a", "cast_b")
HISTOGRAM_AGGREGATES = ("d_shadow", "d_mid", "d_high")

TAG_FIELDS = (
    "tone_tags", "palette_tags", "semantic_affinity", "semantic_risks",
)
OPS = (">=", "<=", ">", "<", "contains")

#: Every legal ``term["field"]`` path (m3: namespace whitelist).  Derived from
#: the same constants ``build_features`` writes with, so the two cannot drift.
FEATURE_FIELDS: frozenset[str] = frozenset(
    [f"summary.{key}" for key in SUMMARY_KEYS]
    + [
        f"bands.{en}.{metric}"
        for en in BAND_EN.values() for metric in BAND_METRICS
    ]
    + [
        f"ramp.{stop}.{key}"
        for stop in RAMP_STOPS for key in RAMP_KEYS
    ]
    + [
        f"segments.{seg}.{metric}"
        for seg in SEGMENTS for metric in SEGMENT_METRICS
    ]
    + [f"histogram.{key}" for key in HISTOGRAM_AGGREGATES]
    + ["scene_affinity"]
)

#: Data-side closed scene list.  Authority: ``SCENE_TAXONOMY`` in
#: ``dataset_build/agent_loop/lut_annotations.py:36`` (itself ``SCENE_WEIGHTS``
#: of ``dataset_build/src/construct/sources.py``).  Copied here so this tool
#: keeps no import edge into ``agent_loop``; a test asserts the two are equal.
SCENE_TAXONOMY_FROZEN: tuple[str, ...] = (
    "portrait", "landscape", "food", "unknown", "still_life", "architecture",
    "night", "street", "wedding", "product",
)

#: frozen-v1 ruling (a): Style Card tag word -> data-side ``scene_affinity``
#: word.  Only word forms that actually differ appear here.
AFFINITY_TAG_TO_SCENE: dict[str, str] = {"still-life": "still_life"}

#: frozen-v1 ruling (a): the 9-word ``semantic_affinity`` closed set
#: (``still_life`` -> ``still-life``, ``unknown`` dropped).
FROZEN_AFFINITY: tuple[str, ...] = (
    "portrait", "landscape", "food", "still-life", "architecture",
    "night", "street", "wedding", "product",
)

#: m2: runtime size guards for the three numeric tag fields.  ``semantic_affinity``
#: is not a range but an exact frozen list (:data:`FROZEN_AFFINITY`).
TAG_FIELD_SIZE_BOUNDS: dict[str, tuple[int, int]] = {
    "tone_tags": (12, 16),
    "palette_tags": (16, 24),
    "semantic_risks": (8, 12),
}


class StyleVocabError(RuntimeError):
    """Raised on malformed vocab, unknown tag, or missing feature."""


# --------------------------------------------------------------- features
def build_features(
    annotation: Mapping[str, Any],
    fingerprint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Flatten one annotation row (+ its fingerprint row) into dotted keys."""
    hsl = annotation.get("hsl_features")
    if not isinstance(hsl, Mapping):
        raise StyleVocabError("annotation row has no hsl_features mapping")
    out: dict[str, Any] = {}

    summary = hsl.get("summary") or {}
    for key in SUMMARY_KEYS:
        if key in summary:
            out[f"summary.{key}"] = float(summary[key])

    bands = hsl.get("bands") or {}
    for zh, en in BAND_EN.items():
        row = bands.get(zh)
        if not isinstance(row, Mapping):
            continue
        for metric in BAND_METRICS:
            if metric in row:
                out[f"bands.{en}.{metric}"] = float(row[metric])

    for point in hsl.get("neutral_ramp") or []:
        stop = f"p{round(float(point['in']) * 100):02d}"
        for key in RAMP_KEYS:
            if key in point:
                out[f"ramp.{stop}.{key}"] = float(point[key])

    if fingerprint is not None:
        segments = fingerprint.get("segments") or {}
        for seg in SEGMENTS:
            row = segments.get(seg)
            if not isinstance(row, Mapping):
                continue
            for metric in SEGMENT_METRICS:
                if metric in row:
                    out[f"segments.{seg}.{metric}"] = float(row[metric])
        histogram = fingerprint.get("histogram") or {}
        for key in HISTOGRAM_AGGREGATES:
            if key in histogram:
                out[f"histogram.{key}"] = float(histogram[key])

    # Non-numeric feature: the existing closed scene vocabulary, used by the
    # ``contains`` operator for semantic_affinity words.
    scene = annotation.get("scene_affinity")
    out["scene_affinity"] = list(scene) if isinstance(scene, list) else []
    return out


def load_library(
    annotations_path: Path = DEFAULT_ANNOTATIONS,
    fingerprints_path: Path = DEFAULT_FINGERPRINTS,
) -> dict[str, dict[str, Any]]:
    """preset_id -> flat feature dict, for the whole library."""
    fingerprints: dict[str, Any] = {}
    with fingerprints_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                fingerprints[row["preset_id"]] = row
    features: dict[str, dict[str, Any]] = {}
    with annotations_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            preset_id = row["preset_id"]
            features[preset_id] = build_features(row, fingerprints.get(preset_id))
    return features


# ------------------------------------------------------------------ vocab
def load_vocab(path: Path = DEFAULT_VOCAB) -> dict[str, Any]:
    vocab = json.loads(path.read_text(encoding="utf-8"))
    validate_vocab(vocab)
    return vocab


def iter_terms(vocab: Mapping[str, Any]) -> Iterable[tuple[str, str, dict]]:
    """Yield ``(tag_field, word, term)`` over the four closed sets."""
    for field in TAG_FIELDS:
        for word, term in (vocab.get(field) or {}).items():
            yield field, word, term


def vocab_sha256(vocab: Mapping[str, Any]) -> str:
    """sha256 of the vocab payload with ``meta.sha256`` blanked to ``null``.

    The digest has to exclude the slot it is written into, so the registered
    value is reproducible from the frozen file itself:
    ``vocab_sha256(json.load(f)) == json.load(f)["meta"]["sha256"]``.
    """
    payload = json.loads(json.dumps(vocab, ensure_ascii=False, sort_keys=False))
    if isinstance(payload.get("meta"), dict):
        payload["meta"]["sha256"] = None
    blob = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def resolve_affinity(threshold: Any, mapping: Mapping[str, str] | None = None) -> Any:
    """Style Card affinity tag word -> data-side ``scene_affinity`` word."""
    table = AFFINITY_TAG_TO_SCENE if mapping is None else mapping
    if isinstance(threshold, str):
        return table.get(threshold, threshold)
    return threshold


def validate_vocab(vocab: Mapping[str, Any]) -> None:
    """Reject a malformed vocabulary at load time.

    Checks, in order: the four tag fields exist; word forms are lowercase
    ``alnum+hyphen`` and (m1) carry no underscore outside ``semantic_affinity``,
    which itself must equal :data:`FROZEN_AFFINITY` verbatim; (m2) the three
    numeric fields sit inside :data:`TAG_FIELD_SIZE_BOUNDS`; every word carries
    a complete ``(field, op, threshold)`` triple with a known op and (m3) a
    ``field`` inside :data:`FEATURE_FIELDS`; affinity words resolve through
    :data:`AFFINITY_TAG_TO_SCENE` into :data:`SCENE_TAXONOMY_FROZEN` minus
    ``unknown``; (m4) no word repeats across the four tag fields; no two words
    share one assertion triple; a registered ``meta.sha256`` matches the payload.
    """
    missing = [field for field in TAG_FIELDS if field not in vocab]
    if missing:
        raise StyleVocabError(f"vocab is missing tag fields: {missing}")

    # m2: size guards.
    for field, (low, high) in TAG_FIELD_SIZE_BOUNDS.items():
        size = len(vocab[field] or {})
        if not low <= size <= high:
            raise StyleVocabError(
                f"{field}: size {size} outside the frozen band [{low}, {high}]"
            )
    affinity_words = tuple(vocab["semantic_affinity"] or {})
    if affinity_words != FROZEN_AFFINITY:
        raise StyleVocabError(
            f"semantic_affinity: {list(affinity_words)} != frozen list "
            f"{list(FROZEN_AFFINITY)}"
        )

    mapping = (vocab.get("meta") or {}).get("affinity_tag_to_scene")
    if mapping is not None and dict(mapping) != AFFINITY_TAG_TO_SCENE:
        raise StyleVocabError(
            f"meta.affinity_tag_to_scene {dict(mapping)} != {AFFINITY_TAG_TO_SCENE}"
        )

    for field, word, term in iter_terms(vocab):
        if word != word.lower() or not word.replace("-", "").replace("_", "").isalnum():
            raise StyleVocabError(f"{field}/{word}: word is not lowercase alnum+hyphen")
        # m1: only the data side keeps underscores; every vocabulary word is
        # hyphen-form, and semantic_affinity reaches the data side through
        # AFFINITY_TAG_TO_SCENE instead of through its own word form.
        if field != "semantic_affinity" and "_" in word:
            raise StyleVocabError(f"{field}/{word}: underscore is not a legal word form")
        for key in ("field", "op", "threshold"):
            if key not in term:
                raise StyleVocabError(f"{field}/{word}: assertion misses {key!r}")
        if term["op"] not in OPS:
            raise StyleVocabError(f"{field}/{word}: unknown op {term['op']!r}")
        # m3: field namespace whitelist.
        if term["field"] not in FEATURE_FIELDS:
            raise StyleVocabError(
                f"{field}/{word}: unknown feature field {term['field']!r}"
            )
        if field == "semantic_affinity":
            if term["field"] != "scene_affinity" or term["op"] != "contains":
                raise StyleVocabError(
                    f"{field}/{word}: affinity assertion must be "
                    f"'scene_affinity contains <scene>'"
                )
            scene = resolve_affinity(term["threshold"])
            if scene == "unknown":
                raise StyleVocabError(
                    f"{field}/{word}: 'unknown' is not a legal Style Card tag"
                )
            if scene not in SCENE_TAXONOMY_FROZEN:
                raise StyleVocabError(
                    f"{field}/{word}: {scene!r} is outside the scene taxonomy"
                )

    # m4: one word may not live in two tag fields.
    term_index(vocab)

    triples: dict[tuple, str] = {}
    for field, word, term in iter_terms(vocab):
        key = (term["field"], term["op"], term["threshold"])
        if key in triples:
            raise StyleVocabError(
                f"{field}/{word}: assertion triple duplicates {triples[key]}"
            )
        triples[key] = f"{field}/{word}"

    registered = (vocab.get("meta") or {}).get("sha256")
    if registered is not None:
        actual = vocab_sha256(vocab)
        if registered != actual:
            raise StyleVocabError(
                f"meta.sha256 {registered} != recomputed payload sha256 {actual}"
            )


def term_index(vocab: Mapping[str, Any]) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for field, word, term in iter_terms(vocab):
        if word in index:
            raise StyleVocabError(f"duplicate word across tag fields: {word}")
        index[word] = {**term, "tag_field": field}
    return index


# -------------------------------------------------------------- evaluate
def evaluate_term(features: Mapping[str, Any], term: Mapping[str, Any]) -> bool:
    """Evaluate one assertion triple against one preset's flat features.

    ``contains`` on ``scene_affinity`` resolves the tag word through
    :data:`AFFINITY_TAG_TO_SCENE` first, because the vocabulary is hyphen-form
    while the table still stores the original scene word forms.
    """
    field, op, threshold = term["field"], term["op"], term["threshold"]
    if field not in features:
        raise StyleVocabError(f"feature {field!r} absent from this preset")
    value = features[field]
    if op == "contains":
        if field == "scene_affinity":
            threshold = resolve_affinity(threshold)
        return threshold in value
    value = float(value)
    threshold = float(threshold)
    if op == ">=":
        return value >= threshold
    if op == "<=":
        return value <= threshold
    if op == ">":
        return value > threshold
    if op == "<":
        return value < threshold
    raise StyleVocabError(f"unknown op {op!r}")


def check_tags(
    features: Mapping[str, Any],
    tags: Sequence[str],
    vocab: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Per-tag pass/fail rows. Unknown words fail with ``reason='unknown-word'``.

    Granularity (frozen-v1 ruling d): the unit of judgement is the **tag**, the
    unit of labelling is the **record**.  One row per requested tag, ``pass``
    false for an unknown word, a false assertion, or a feature the preset does
    not carry.  Downstream: every failed tag is dropped from the card (see
    :func:`accepted_tags`) and, because at least one tag failed, the whole
    record is labelled ``conflicted`` by :func:`annotation_quality` and goes to
    the re-annotation queue; the card is ``clean`` only when every tag passed.
    A failed tag never silently survives, and a partially failed card never
    silently counts as clean.
    """
    index = term_index(vocab)
    rows: list[dict[str, Any]] = []
    for tag in tags:
        term = index.get(tag)
        if term is None:
            rows.append({
                "tag": tag, "tag_field": None, "assertion": None,
                "value": None, "pass": False, "reason": "unknown-word",
            })
            continue
        try:
            passed = evaluate_term(features, term)
            reason = "ok" if passed else "assertion-false"
        except StyleVocabError as exc:
            passed, reason = False, str(exc)
        rows.append({
            "tag": tag,
            "tag_field": term["tag_field"],
            "assertion": f"{term['field']} {term['op']} {term['threshold']}",
            "value": features.get(term["field"]),
            "pass": passed,
            "reason": reason,
        })
    return rows


def accepted_tags(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """The tags that survive onto the card (frozen-v1 ruling d): the passing ones."""
    return [row["tag"] for row in rows if row["pass"]]


def annotation_quality(rows: Sequence[Mapping[str, Any]]) -> str:
    """Programmatic ``annotation_quality`` (spec 3.1B, frozen-v1 ruling d).

    ``"clean"`` iff every checked tag passed; any single failed tag (unknown
    word, false assertion, or missing feature) labels the whole record
    ``"conflicted"``, i.e. it enters the re-annotation queue while its failed
    tags are dropped by :func:`accepted_tags`.
    """
    return "clean" if all(row["pass"] for row in rows) else "conflicted"


# ------------------------------------------------------------------- cli
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset-id", required=True)
    parser.add_argument("--tags", required=True, help="comma separated tag words")
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--fingerprints", type=Path, default=DEFAULT_FINGERPRINTS)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    vocab = load_vocab(args.vocab)
    library = load_library(args.annotations, args.fingerprints)
    if args.preset_id not in library:
        print(f"unknown preset_id: {args.preset_id}", file=sys.stderr)
        return 2
    features = library[args.preset_id]
    tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
    rows = check_tags(features, tags, vocab)
    payload = {
        "preset_id": args.preset_id,
        "tags": rows,
        "accepted_tags": accepted_tags(rows),
        "annotation_quality": annotation_quality(rows),
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for row in rows:
            flag = "PASS" if row["pass"] else "FAIL"
            print(f"{flag}  {row['tag']:<26} {row['assertion']}  value={row['value']}")
        print(f"annotation_quality = {payload['annotation_quality']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
