"""Unit tests for the seven-segment -> two-segment conversion (SFT spec 4.2).

Run: ``python -m pytest q3vl/tests/test_twoseg.py -q``
"""

from __future__ import annotations

import pytest

from dataset_build.src.construct.responses import (
    REASONING_FIELDS,
    _SECTION_TOKENS,
    assemble_reasoning,
)
from q3vl.data.twoseg import (
    COLOR_FIELDS,
    WHERE_FIELD,
    ReasoningRejected,
    contains_legacy_tag,
    convert,
)

BODIES = {
    "problem_lighting": "The frame is flat and underexposed.",
    "problem_global_color": "The palette is muddy overall.",
    "problem_specific_color": "The foliage green is dull.",
    "region_scope": "subject: the woman; edit scope: a diagonal band.",
    "plan_lighting": "Deepen the shadows firmly.",
    "plan_global_color": "Push the whole frame warmer.",
    "plan_specific_color": "Lift the foliage green.",
}


def _assembled(**overrides: str) -> str:
    return assemble_reasoning({**BODIES, **overrides})


def test_field_order_matches_canonical_parser():
    assert tuple(_SECTION_TOKENS) == REASONING_FIELDS
    assert WHERE_FIELD == "region_scope"
    assert COLOR_FIELDS == tuple(f for f in REASONING_FIELDS if f != WHERE_FIELD)
    # problems before plans, spatial statement excluded
    assert COLOR_FIELDS == (
        "problem_lighting", "problem_global_color", "problem_specific_color",
        "plan_lighting", "plan_global_color", "plan_specific_color",
    )


def test_round_trip_maps_bodies_verbatim():
    seg = convert(_assembled())
    assert seg.where == BODIES["region_scope"]
    assert seg.color == "\n".join(BODIES[f] for f in COLOR_FIELDS)
    assert seg.closing == ""
    assert not seg.has_closing
    assert not contains_legacy_tag(seg.where + seg.color)
    # region_scope must not be duplicated into color
    assert BODIES["region_scope"] not in seg.color


def test_closing_text_is_kept_but_never_invented():
    with_closing = _assembled() + "So the picture reads calmer."
    seg = convert(with_closing)
    assert seg.has_closing
    assert seg.color.endswith("So the picture reads calmer.")
    assert convert(_assembled()).closing == ""


@pytest.mark.parametrize("field", REASONING_FIELDS)
def test_missing_segment_is_rejected(field):
    text = _assembled()
    start, end = _SECTION_TOKENS[field]
    broken = text.replace(start, "").replace(end, "")
    with pytest.raises(ReasoningRejected) as excinfo:
        convert(broken)
    assert excinfo.value.reason in {"segment_missing", "segment_out_of_order"}


def test_unclosed_segment_is_rejected():
    text = _assembled().replace(_SECTION_TOKENS["plan_lighting"][1], "")
    with pytest.raises(ReasoningRejected) as excinfo:
        convert(text)
    assert excinfo.value.reason in {"segment_missing", "segment_out_of_order"}


def test_duplicated_segment_is_rejected():
    start, end = _SECTION_TOKENS["problem_lighting"]
    text = _assembled() + start + "again" + end
    with pytest.raises(ReasoningRejected) as excinfo:
        convert(text)
    assert excinfo.value.reason == "segment_duplicated"


def test_out_of_order_segments_are_rejected():
    parts = [
        _SECTION_TOKENS[f][0] + BODIES[f] + _SECTION_TOKENS[f][1] for f in REASONING_FIELDS
    ]
    parts[0], parts[1] = parts[1], parts[0]
    with pytest.raises(ReasoningRejected) as excinfo:
        convert("".join(parts))
    assert excinfo.value.reason == "segment_out_of_order"


def test_empty_body_is_rejected():
    with pytest.raises(ReasoningRejected) as excinfo:
        convert(_assembled(plan_global_color="   "))
    assert excinfo.value.reason == "segment_empty"


def test_stray_text_between_segments_is_rejected():
    text = _assembled().replace(
        _SECTION_TOKENS["region_scope"][1],
        _SECTION_TOKENS["region_scope"][1] + " leftover ",
    )
    with pytest.raises(ReasoningRejected) as excinfo:
        convert(text)
    assert excinfo.value.reason == "stray_text_between_segments"


def test_region_scope_copy_into_color_is_rejected():
    with pytest.raises(ReasoningRejected) as excinfo:
        convert(_assembled(plan_lighting=BODIES["region_scope"]))
    assert excinfo.value.reason == "region_scope_leaked_into_color"


def test_empty_input_is_rejected():
    with pytest.raises(ReasoningRejected) as excinfo:
        convert("")
    assert excinfo.value.reason == "reasoning_empty"
