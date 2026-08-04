"""Seven-segment -> two-segment deterministic conversion (SFT spec 4.2).

The seven legacy tags come from the canonical parser
``dataset_build/src/construct/responses.py`` (``_SECTION_TOKENS``); they are
imported rather than retyped so a change there breaks this module loudly
instead of silently producing a different dataset.

Conversion contract, quoted from spec 4.2:

* ``<where>`` = the ``region_scope`` body, verbatim;
* ``<color>`` = the other six bodies **in their original problem-then-plan
  order**, followed by the original closing text *if the source has one*;
* ``region_scope`` is never copied into ``color``;
* no closing text is invented;
* a missing / duplicated / out-of-order / unclosed segment sends the sample to
  the rejection report -- never to a guessed repair.

The parser is deliberately positional (it walks every tag occurrence in order)
rather than seven independent regex searches: independent searches cannot see a
duplicated opener or a swapped pair, which are exactly the failures spec 4.2
demands be rejected.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from dataset_build.src.construct.responses import _SECTION_TOKENS, REASONING_FIELDS

WHERE_FIELD = "region_scope"
COLOR_FIELDS: tuple[str, ...] = tuple(f for f in REASONING_FIELDS if f != WHERE_FIELD)
# The body separator inside <color>.  Six bodies that used to be delimited by
# tags need *some* delimiter once the tags are removed; a single newline is the
# minimal choice and matches ``q3vl.train.dataset.assemble_color_segment``.
COLOR_SEPARATOR = "\n"

_ORDERED_TAGS: tuple[tuple[str, str, str], ...] = tuple(
    (field, kind, tag)
    for field in REASONING_FIELDS
    for kind, tag in (("start", _SECTION_TOKENS[field][0]), ("end", _SECTION_TOKENS[field][1]))
)
_EXPECTED_SEQUENCE: tuple[tuple[str, str], ...] = tuple((f, k) for f, k, _ in _ORDERED_TAGS)
_TAG_TO_SLOT: dict[str, tuple[str, str]] = {tag: (f, k) for f, k, tag in _ORDERED_TAGS}
_ANY_TAG = re.compile("|".join(re.escape(tag) for tag in _TAG_TO_SLOT))

assert tuple(_SECTION_TOKENS) == REASONING_FIELDS, "canonical field order drifted"
assert len(_TAG_TO_SLOT) == 14, "expected 14 legacy tags"


class ReasoningRejected(ValueError):
    """The legacy reasoning string cannot be converted; reason is machine readable."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class TwoSegment:
    where: str
    color: str
    closing: str
    fields: dict[str, str]

    @property
    def has_closing(self) -> bool:
        return bool(self.closing)


def convert(reasoning: str) -> TwoSegment:
    """Convert one assembled seven-segment reasoning string into two segments."""
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ReasoningRejected("reasoning_empty")

    matches = list(_ANY_TAG.finditer(reasoning))
    observed = tuple(_TAG_TO_SLOT[m.group(0)] for m in matches)
    if observed != _EXPECTED_SEQUENCE:
        raise ReasoningRejected(*_diagnose(observed))

    bodies: dict[str, str] = {}
    for i in range(0, len(matches), 2):
        field = _EXPECTED_SEQUENCE[i][0]
        body = reasoning[matches[i].end():matches[i + 1].start()]
        if not body.strip():
            raise ReasoningRejected("segment_empty", field)
        bodies[field] = body.strip()

    prefix = reasoning[: matches[0].start()]
    if prefix.strip():
        raise ReasoningRejected("stray_text_before_first_segment", prefix.strip()[:80])
    for i in range(1, len(_EXPECTED_SEQUENCE) // 2):
        gap = reasoning[matches[2 * i - 1].end(): matches[2 * i].start()]
        if gap.strip():
            raise ReasoningRejected("stray_text_between_segments", gap.strip()[:80])

    # Anything after the last closing tag is the original closing text.  Spec 4.2
    # keeps it if it exists and forbids writing one when it does not.
    closing = reasoning[matches[-1].end():].strip()

    where = bodies[WHERE_FIELD]
    color_parts = [bodies[f] for f in COLOR_FIELDS]
    if closing:
        color_parts.append(closing)
    color = COLOR_SEPARATOR.join(color_parts)

    if where in color:
        # Not a hard rule violation on its own (the annotator may legitimately
        # repeat a phrase), but an exact whole-segment copy would mean the
        # spatial statement leaked into <color>; that one is rejected.
        for part in color_parts:
            if part == where:
                raise ReasoningRejected("region_scope_leaked_into_color", where[:80])

    return TwoSegment(where=where, color=color, closing=closing, fields=bodies)


def _diagnose(observed: tuple[tuple[str, str], ...]) -> tuple[str, str]:
    """Classify why the observed tag sequence is not the canonical one."""
    seen = [f"{f}.{k}" for f, k in observed]
    expected = [f"{f}.{k}" for f, k in _EXPECTED_SEQUENCE]
    if len(seen) != len(set(seen)):
        dupes = sorted({t for t in seen if seen.count(t) > 1})
        return "segment_duplicated", ",".join(dupes)
    missing = [t for t in expected if t not in seen]
    if missing:
        return "segment_missing", ",".join(missing)
    extra = [t for t in seen if t not in expected]
    if extra:
        return "segment_unknown_tag", ",".join(extra)
    return "segment_out_of_order", "|".join(seen)


def contains_legacy_tag(text: str) -> list[str]:
    """Legacy tags that survived into a converted string (must always be empty)."""
    return [tag for tag in _TAG_TO_SLOT if tag in text]
