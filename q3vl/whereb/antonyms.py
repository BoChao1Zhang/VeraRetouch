"""The fixed antonym table for the invariance control (A5-B1, main-agent ruling).

The control it serves
---------------------
Where-B's output is supposed to be determined by the **subject** of the
instruction, not by its colour direction.  So flipping only the colour-direction
words -- "darker" -> "brighter", "warmer" -> "cooler", "saturated" ->
"desaturated" -- while leaving the subject phrase untouched must leave the mask
essentially unchanged.  A field that moves is reading colour words it has no
business reading.  This is an **invariance** control (small ``|delta|`` is the
pass), the mirror image of the directional same-image paired difference (large
positive ``delta`` is the pass); the campaign runs both.

Why a fixed table rather than a generator
-----------------------------------------
A generated or model-proposed antonym list is a fabrication risk: nobody can
check it, and it can silently change between runs.  This table is a literal,
version-controlled constant with a digest, so a report can cite exactly which
substitution produced a number.  It covers the three axes the ruling names --
luminance, temperature, saturation -- and nothing else.

Substitution rules that matter
------------------------------
* the swap is **simultaneous** (one regex pass over an alternation), so
  ``darker -> brighter`` cannot then be re-flipped back by the ``brighter``
  rule;
* alternatives are matched **longest first**, because ``desaturated`` contains
  ``saturated`` and a naive pass would corrupt it;
* matching is case-insensitive and case-preserving, and uses word boundaries, so
  ``Brighter`` stays capitalised and ``unsaturated`` is never touched;
* the table is an **involution**: applying it twice returns the original text.
  ``tests/test_antonyms.py`` asserts that on the real corpus.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Iterable

__all__ = ["ANTONYM_AXES", "ANTONYM_PAIRS", "ANTONYM_MAP", "table_digest",
           "flip_text", "flipped_terms", "AXES"]

#: (axis -> the bidirectional pairs on that axis).  Every word appears in at
#: most one pair, which is what makes the substitution a well-defined involution.
ANTONYM_AXES: dict[str, tuple[tuple[str, str], ...]] = {
    "luminance": (
        ("darker", "brighter"),
        ("darken", "brighten"),
        ("darkened", "brightened"),
        ("darkening", "brightening"),
        ("dimmer", "lighter"),
        ("dim", "light"),
        ("shadowed", "illuminated"),
    ),
    "temperature": (
        ("warmer", "cooler"),
        ("warm", "cool"),
        ("warmth", "coolness"),
        ("warmed", "cooled"),
        ("warming", "cooling"),
    ),
    "saturation": (
        ("saturated", "desaturated"),
        ("saturate", "desaturate"),
        ("saturation", "desaturation"),
        ("vivid", "muted"),
        ("richer", "duller"),
        ("rich", "dull"),
        ("intensify", "subdue"),
        ("intensified", "subdued"),
        ("intensifying", "subduing"),
        ("vibrant", "drab"),
    ),
}

AXES: tuple[str, ...] = tuple(ANTONYM_AXES)

#: flattened, in declaration order
ANTONYM_PAIRS: tuple[tuple[str, str], ...] = tuple(
    p for axis in ANTONYM_AXES for p in ANTONYM_AXES[axis]
)

#: the simultaneous swap map, lowercase keys
ANTONYM_MAP: dict[str, str] = {}
for _a, _b in ANTONYM_PAIRS:
    ANTONYM_MAP[_a] = _b
    ANTONYM_MAP[_b] = _a

if len(ANTONYM_MAP) != 2 * len(ANTONYM_PAIRS):      # a word in two pairs
    raise AssertionError("antonym table is not one-to-one; the swap would be ambiguous")

# longest alternatives first: `desaturated` must win over `saturated`
_PATTERN = re.compile(
    r"\b(" + "|".join(sorted((re.escape(w) for w in ANTONYM_MAP), key=len, reverse=True))
    + r")\b",
    flags=re.IGNORECASE,
)


def table_digest() -> str:
    """sha256 of the canonical table, so a report can pin which table it used."""
    blob = json.dumps(ANTONYM_AXES, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _match_case(src: str, repl: str) -> str:
    if src.isupper():
        return repl.upper()
    if src[:1].isupper():
        return repl[:1].upper() + repl[1:]
    return repl


def flip_text(text: str) -> str:
    """Swap every colour-direction word; leave everything else byte-identical."""
    return _PATTERN.sub(lambda m: _match_case(m.group(0), ANTONYM_MAP[m.group(0).lower()]),
                        text)


def flipped_terms(text: str) -> list[str]:
    """The words this text would have flipped -- for the per-sample record."""
    return [m.group(0) for m in _PATTERN.finditer(text)]


def axis_of(word: str) -> str | None:
    w = word.lower()
    for axis, pairs in ANTONYM_AXES.items():
        for a, b in pairs:
            if w in (a, b):
                return axis
    return None


def axes_touched(text: str) -> list[str]:
    seen: list[str] = []
    for w in flipped_terms(text):
        ax = axis_of(w)
        if ax and ax not in seen:
            seen.append(ax)
    return seen


def coverage(texts: Iterable[str]) -> dict[str, float | int]:
    """How much of a corpus this table actually reaches (for the report)."""
    n = touched = 0
    per_axis: dict[str, int] = {a: 0 for a in AXES}
    for t in texts:
        n += 1
        axes = axes_touched(t)
        if axes:
            touched += 1
        for a in axes:
            per_axis[a] += 1
    return {"n": n, "n_flippable": touched,
            "flippable_rate": touched / n if n else 0.0,
            "per_axis": per_axis, "digest": table_digest()}
