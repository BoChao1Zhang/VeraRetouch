"""Stage A of the WP5 re-evaluation: mechanical, judge-free annotation checks.

Every check here is decidable from the text alone, so it costs no relay call and
carries no judge noise.  The blind review (``reeval_annot_blind.py``) therefore
never asks a model about structure or leakage -- it only scores what a human eye
would have to look at the images to decide.

Checked units are the current SFT rows plus, when ``--include-reannot`` is set,
both text versions of every re-annotated pair, so the A/B comparison in stage B3
can be read next to the mechanical verdict for the same two texts.

Usage:
  python -m dataset_build.tools.reeval_annot_mech --out <dir>
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

BUILD_ROOT = Path("/mnt/nfs/bc/data/builds/eval100-annotqa-20260727")
REVIEW_ROOT = Path("/var/cache/veradata/annot_review/eval100-annotqa-20260727")

# Canonical section order, mirroring construct.responses.assemble_reasoning.
# The v4 contract inserts ``region_scope`` between the problems and the plans;
# every row written before it has six sections and is read as ``legacy`` rather
# than reported as a structural failure.
SECTIONS_LEGACY = (
    ("problem", "light"),
    ("problem", "globalcolor"),
    ("problem", "specificcolor"),
    ("plan", "light"),
    ("plan", "globalcolor"),
    ("plan", "specificcolor"),
)
SECTIONS_V4 = (
    *SECTIONS_LEGACY[:3],
    ("region", "scope"),
    *SECTIONS_LEGACY[3:],
)
SECTION_TOKEN_RE = re.compile(
    r"<(problem|plan|region)_(light|globalcolor|specificcolor|scope)_(start|end)>"
)
REGION_SCOPE_SPAN_RE = re.compile(r"<region_scope_start>.*?<region_scope_end>", re.S)
GLOBAL_REGION_SCOPE = "global adjustment across the entire frame"

# Word budgets from the v4 contract (D7), anchored on the p90 of blind-review
# passes.  Counted for every unit so the distributions stay comparable, but only
# flagged for v4 text: the legacy rows were written without any budget.
INSTRUCTION_WORD_CAP = 75
INSTRUCTION_SHORT_WORD_CAP = 30

# Lightroom / Camera Raw parameter keys.  The ``2012`` suffix and the ``crs:``
# prefix are unambiguous key forms; the bare English words are not (an
# annotation is *supposed* to say "contrast"), so the bare vocabulary is only
# flagged when it is bound to a key-like shape: CamelCase compound,
# snake_case compound, or an adjacent numeric value.
CRS_WORDS = (
    "exposure", "contrast", "highlights", "shadows", "whites", "blacks",
    "clarity", "vibrance", "saturation", "temperature", "temp", "tint",
    "dehaze", "texture", "sharpness", "sharpening", "grain", "vignette",
    "luminance", "hue", "gamma", "curve", "tonecurve", "splittoning",
    "colorgrade", "whitebalance", "wb",
)
_CRS_ALT = "|".join(sorted(CRS_WORDS, key=len, reverse=True))

LEAK_PATTERNS: dict[str, re.Pattern[str]] = {
    # 1. explicit parameter-key syntax
    "param_key_crs": re.compile(r"\bcrs:[A-Za-z0-9_]+|\b[A-Za-z]+2012\b"),
    "param_key_camel": re.compile(
        r"\b(?:Exposure|Contrast|Highlights|Shadows|Whites|Blacks|Clarity|Vibrance|"
        r"Saturation|Temperature|Tint|Dehaze|Texture|Sharpness|LuminanceSmoothing|"
        r"SplitToning|ColorGrade|PostCropVignette|ToneCurve|HueAdjustment|"
        r"SaturationAdjustment|LuminanceAdjustment|GrayMixer|ParametricShadows|"
        r"ParametricHighlights|ParametricDarks|ParametricLights|IncrementalTemperature|"
        r"IncrementalTint|AutoTone|ConvertToGrayscale)(?:[A-Z][a-z]+)+\b"
    ),
    "param_key_snake": re.compile(
        rf"\b(?:{_CRS_ALT})_[a-z0-9]+\b|\b[a-z0-9]+_(?:{_CRS_ALT})\b", re.IGNORECASE
    ),
    "param_key_slider": re.compile(
        rf"\b(?:{_CRS_ALT})\s+(?:slider|value|parameter|key|setting)s?\b", re.IGNORECASE
    ),
    # 2. numeric magnitudes / deltas
    "numeric_signed": re.compile(r"(?<![\w.])[+−±-]\s?\d+(?:\.\d+)?(?![\w])"),
    "numeric_unit": re.compile(
        r"\b\d+(?:\.\d+)?\s*(?:%|EV\b|stops?\b|kelvin\b|K\b|px\b|pixels?\b|units?\b|"
        r"points?\b|degrees?\b|°)", re.IGNORECASE
    ),
    "numeric_near_param": re.compile(
        rf"\b(?:{_CRS_ALT})\b[^.;:]{{0,24}}?\b\d+(?:\.\d+)?\b|"
        rf"\b\d+(?:\.\d+)?\b[^.;:]{{0,16}}?\b(?:{_CRS_ALT})\b", re.IGNORECASE
    ),
    # 3. pipeline scores
    "score_leak": re.compile(
        r"\b(?:IAA|one[\s_-]?align|q[\s_-]?score|aesthetic score|quality score|"
        r"improvement score|source_onealign|reliab\w+ score)\b", re.IGNORECASE
    ),
    # 4. mask / geometry / alpha
    "mask_geometry_leak": re.compile(
        r"\b(?:mask(?:s|ed|ing)?|c_?gt\b|ground[\s_-]?truth|alpha(?:\s?(?:channel|map|"
        r"mean|blend))?|matte|bounding\s?box|bbox|segmentation|segment(?:ed|ation)?\s?map|"
        r"feather(?:ing|ed)?|selection\s?edge|roi\b|region\s?of\s?interest|"
        r"gradient\s?ramp|radial\s?gradient|linear\s?gradient|geometry)\b", re.IGNORECASE
    ),
    # 5. degrade / auto pipeline vocabulary
    "degrade_auto_leak": re.compile(
        r"\b(?:degrade[ds]?|degradation|synthetic(?:ally)?\s?degrad\w+|auto[\s_-]?tone|"
        r"auto[\s_-]?(?:adjust\w*|correct\w*|enhance\w*)|automated\s?pipeline)\b",
        re.IGNORECASE,
    ),
    # 6. slot / candidate / preset identifiers
    "slot_candidate_leak": re.compile(
        r"\bslot(?:s|_mode|\s?mode|\s?index|\s?id)?\b|\bcandidate\s?(?:id|\d|slot)|"
        r"\bwinner(?:\s?rank)?\b|\brank\s?\d\b|\bpreset[\s_-]?id\b|\brcp_[0-9a-f]{6,}\b|"
        r"\be18_\d+\b|\.cube\b|\.xmp\b|\bLUT\b|\bpairing[\s_-]?index\b|\bmode[\s_-]?index\b",
        re.IGNORECASE,
    ),
}

# The v4 contract (D5) lets region_scope name the edit's shape: "band", "radial
# falloff" and "linear gradient" are legal *there*.  Everything else the leak set
# bans -- above all every number -- stays banned inside region_scope too, so the
# scope text is checked with the same set minus those two alternatives.
SCOPE_LEAK_PATTERNS: dict[str, re.Pattern[str]] = {
    **LEAK_PATTERNS,
    "mask_geometry_leak": re.compile(
        r"\b(?:mask(?:s|ed|ing)?|c_?gt\b|ground[\s_-]?truth|alpha(?:\s?(?:channel|map|"
        r"mean|blend))?|matte|bounding\s?box|bbox|segmentation|segment(?:ed|ation)?\s?map|"
        r"feather(?:ing|ed)?|selection\s?edge|roi\b|region\s?of\s?interest|"
        r"gradient\s?ramp|geometry)\b", re.IGNORECASE
    ),
}

# Shape and direction vocabulary.  Legal in region_scope, banned in the
# instructions and in the six problem/plan sections: those must point at the
# affected area by naming the scene content that occupies it.
#
# ``falls off`` and ``gradient(s)`` used to sit here and were deleted rather than
# adjudicated: in a photographic annotation they are almost always the sky or the
# light itself ("the light falls off toward the corners", "the sky gradients are
# banded"), so they produced far more false alarms than catches.  The pipeline
# senses that matter are still blocked -- ``falloff`` as a noun stays below, and
# ``linear gradient`` / ``radial gradient`` / ``gradient ramp`` stay in the
# mask_geometry_leak set, which is a leak rather than a mere shape word.
GEOMETRY_WORDS = re.compile(
    r"\b(?:bands?|strips?|stripes?|ribbons?|slices?|columns?|swaths?|swathes?|"
    r"wedges?|ellipses?|elliptical|ovals?|radially?|radiating|concentric|"
    r"falloffs?|diagonal(?:ly)?|horizontal(?:ly)?|vertical(?:ly)?|"
    r"upper[\s-]?(?:left|right|portion|half|third)|"
    r"lower[\s-]?(?:left|right|portion|half|third)|"
    r"top[\s-]?(?:left|right|half|third)|bottom[\s-]?(?:left|right|half|third)|"
    r"(?:left|right)[\s-]?(?:hand[\s-]?side|half)|"
    r"left[\s-]?to[\s-]?right|corner[\s-]?to[\s-]?corner|"
    r"(?:left|right|up|down|in|out)wards?|"
    r"north[\s-]?(?:west|east)|south[\s-]?(?:west|east)|"
    r"sweeping across|from one side to the other|"
    r"crossing the (?:whole )?(?:picture|frame|image|scene)|"
    r"axis|orientation)\b",
    re.IGNORECASE,
)

# Number words, so spelling a measurement out is not a way past the digit rules.
# "half", "third" and "quarter" are deliberately absent: "a third of the frame
# wide" is vague enough to be ordinary English, and demanding it be flagged would
# also condemn "a third of the image is sky".
_NUMBER_WORDS = (
    r"zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred"
)
_SPELLED_NUMBER = rf"\b(?:{_NUMBER_WORDS})(?:[\s-](?:{_NUMBER_WORDS}))*\b"

# Geometry key names in the shapes v4b actually hands over.  ``\b`` does not fire
# on either side of an underscore, so ``axis_angle``, ``half_axis_along`` and
# ``zero_end`` all walked straight past the old word-boundary form.  The token
# below matches a snake_case identifier any component of which is a geometry key;
# the second alternative covers components that are only key-like inside such an
# identifier ("end" alone is ordinary English, "zero_end" is not).
_GEOM_KEYS = (
    "angle", "axis", "axes", "orientation", "tilt", "rotation", "coordinate",
    "coordinates", "centre", "center", "feather", "span", "extent", "radius",
    "diameter", "width", "height", "length",
)
_GEOM_KEYS_SNAKE_ONLY = ("end", "start", "along", "across", "x", "y")


def _alt(words: Iterable[str]) -> str:
    return "|".join(sorted(words, key=len, reverse=True))


_GEOM_KEY_TOKEN = (
    rf"\b(?:[A-Za-z]+_)*(?:{_alt(_GEOM_KEYS)})(?:_[A-Za-z]+)*\b"
    rf"|\b(?:[A-Za-z]+_)+(?:{_alt(_GEOM_KEYS_SNAKE_ONLY)})(?:_[A-Za-z]+)*\b"
)

# The v4b death clause: v4b hands the model the raw geometry numbers, so any
# numeric geometry surfacing in the output means the variant leaked its own hint.
NUMERIC_GEOMETRY_ECHO = re.compile(
    # an angular unit, after digits or after a spelled-out number
    rf"\b\d+(?:\.\d+)?\s*(?:°|degrees?\b|deg\b)|"
    rf"{_SPELLED_NUMBER}\s+(?:degrees?|deg)\b|"
    # a geometry key with a number close by, either order
    rf"(?:{_GEOM_KEY_TOKEN})[^.;:]{{0,24}}?\b\d+(?:\.\d+)?\b|"
    rf"\b\d+(?:\.\d+)?\b[^.;:]{{0,20}}?\b(?:of the frame|frame width|frame height|"
    rf"across the frame)\b|"
    # a percentage, digits or words, with or without the sign
    rf"(?:\b\d+(?:\.\d+)?|{_SPELLED_NUMBER})\s*(?:%|per\s?cent)|"
    # a coordinate pair, bracketed or written as "A by B"
    rf"\(\s*[-+]?\d*\.\d+\s*,\s*[-+]?\d*\.\d+\s*\)|"
    rf"\b\d+(?:\.\d+)?\s*(?:by|x|×)\s*\d+(?:\.\d+)?\b|"
    # two bare decimals inside one sentence -- the shape of the hint itself, and
    # the form every remaining escape took ("0.36 wide and reaches 1.6 along")
    rf"\d*\.\d+\b[^.;:!?]{{0,48}}?\b\d*\.\d+",
    re.IGNORECASE,
)

# Numeric forms that are ordinary English rather than a parameter leak.
NUMERIC_WHITELIST = re.compile(
    r"\b(?:19|20)\d{2}s?\b|\bone|two|three|four|five|six\b", re.IGNORECASE
)


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9'’]+", text or ""))

# Word senses that collide with pipeline vocabulary but are ordinary photographic
# or subject-matter English.  Each entry was adjudicated by reading the sentence
# it matched in this dataset; a hit whose surrounding text matches one of these
# is reported as ``benign`` instead of ``leak`` so the leak count stays honest.
BENIGN_SENSE = (
    # bird plumage, not an alpha edge
    re.compile(r"\bfeather(?:s|ed|ing)?\b(?=\s+(?:texture|detail|tone|color|pattern))", re.I),
    re.compile(r"(?<=\bfine\s)\bfeather\b", re.I),
    re.compile(r"(?<=\bnatural\s)\bfeather\b", re.I),
    re.compile(r"(?<=\bpale\s)\bfeather\b", re.I),
    re.compile(r"(?<=\bsubtle\s)\bfeather\b", re.I),
    re.compile(r"(?<=\bpeach\s)\bfeather\b", re.I),
    # matte print/tonal finish, not an alpha matte
    re.compile(r"\bmatte\b(?=\s+(?:appearance|finish|look|tonal|texture|surface))", re.I),
    # architectural geometry of the depicted scene, not mask geometry
    re.compile(r"(?:structural|building|architectural|edges,)\s+geometry\b", re.I),
    # facial "mask" markings on an animal, not a segmentation mask
    re.compile(r"\bwing,\s*mask\b|\bfacial\s+mask\b|\bmask\s+markings?\b|\beye\s+mask\b", re.I),
)


# The same adjudication, for the shape/direction vocabulary.  A geometry word is
# only banned when it is used to point at the edited area; the picture itself is
# allowed to contain vertical lines, oval tables and bands of cloud.  Each entry
# below was written against a sentence the review produced as a false alarm, so
# the list is deliberately concrete rather than a general theory of English: a
# new false alarm is meant to be adjudicated and added, not guessed at.
GEOMETRY_BENIGN = (
    # a direction qualifying a structure that is depicted, not the edit's extent.
    # The trailing "of the <thing>" is part of the match so that the thing itself
    # ("the vertical lines of the columns") is adjudicated with the phrase.
    re.compile(
        r"\b(?:horizontal|vertical|diagonal)(?:ly)?\s+"
        r"(?:lines?|edges?|beams?|bars?|slats?|blinds?|rails?|posts?|columns?|"
        r"planks?|boards?|ridges?|folds?|creases?|seams?)"
        r"(?:\s+of\s+(?:the\s+)?[A-Za-z-]+)?", re.I,
    ),
    # the camera's orientation, i.e. how the shot was framed
    re.compile(r"\b(?:camera|frame|shot|image)\s+orientation\b|"
               r"\borientation\s+of\s+the\s+(?:camera|frame|shot|image)\b", re.I),
    # the axis of a depicted object
    re.compile(r"\b(?:wheel|axle|hub|rotor|propeller|steering|clock|windmill|"
               r"turbine|spindle)\s+ax[ie]s\b", re.I),
    # an object that happens to be oval
    re.compile(r"\boval\s+(?:table|plate|mirror|rug|window|pool|pond|dish|bowl|"
               r"tray|basin|arena|track|platter|locket|brooch|face)s?\b", re.I),
    # weather and material banding in the scene
    re.compile(r"\bbands?\s+of\s+(?:cloud|mist|fog|haze|rock|stone|sand|snow|"
               r"water|trees?|foliage|fabric|hair|colou?r)", re.I),
    re.compile(r"\b(?:rock|sand|stone|grass|water|cloud|snow|bark|paint|fabric|"
               r"weed|seaweed)\s+strips?\b", re.I),
)


def benign_span(
    text: str, start: int, end: int,
    patterns: Iterable[re.Pattern[str]] = BENIGN_SENSE,
) -> bool:
    """True when a hit falls inside an adjudicated benign word sense."""
    window = text[max(0, start - 40):end + 40]
    offset = start - max(0, start - 40)
    for pattern in patterns:
        for match in pattern.finditer(window):
            if match.start() <= offset < match.end() or \
                    match.start() < offset + (end - start) <= match.end():
                return True
    return False


def geometry_words(text: str) -> list[str]:
    """Shape/direction words used to point at the edited area, lowercased."""
    return sorted({
        match.group(0).lower()
        for match in GEOMETRY_WORDS.finditer(text or "")
        if not benign_span(text, match.start(), match.end(), GEOMETRY_BENIGN)
    })

CJK_RE = re.compile(r"[㐀-䶿一-鿿぀-ヿ가-힯]")


def rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def strip_sections(reasoning: str) -> str:
    """Reasoning prose with the structural tokens removed."""
    return SECTION_TOKEN_RE.sub(" ", reasoning)


def detect_contract(reasoning: str) -> str:
    """``v4`` when the text carries a region_scope section, else ``legacy``."""
    return "v4" if "<region_scope_start>" in (reasoning or "") else "legacy"


def check_sections(reasoning: str, contract: str = "auto") -> dict[str, Any]:
    """Verify the section token skeleton and its canonical order.

    ``contract`` is ``v4`` (seven sections), ``legacy`` (the six written before the
    v4 contract) or ``auto``, which picks by looking for the region_scope token so
    a mixed corpus can be checked in one pass.
    """
    reasoning = reasoning or ""
    if contract == "auto":
        contract = detect_contract(reasoning)
    sections = SECTIONS_V4 if contract == "v4" else SECTIONS_LEGACY
    found = SECTION_TOKEN_RE.findall(reasoning)
    starts = [(kind, aspect) for kind, aspect, edge in found if edge == "start"]
    ends = [(kind, aspect) for kind, aspect, edge in found if edge == "end"]
    balanced = all(
        reasoning.count(f"<{kind}_{aspect}_start>") == 1
        and reasoning.count(f"<{kind}_{aspect}_end>") == 1
        for kind, aspect in sections
    )
    bodies = {}
    for kind, aspect in sections:
        match = re.search(
            rf"<{kind}_{aspect}_start>(.*?)<{kind}_{aspect}_end>", reasoning, re.S
        )
        bodies[f"{kind}_{aspect}"] = (match.group(1).strip() if match else "")
    return {
        "contract": contract,
        "sections_present": len(set(starts)) == len(sections)
        and len(set(ends)) == len(sections),
        "sections_ordered": starts == list(sections),
        "sections_balanced": balanced,
        "empty_sections": sorted(k for k, v in bodies.items() if len(v) < 8),
        "bodies": bodies,
    }


def leak_hits(
    text: str, patterns: Mapping[str, re.Pattern[str]] | None = None
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Return (real hits, hits dismissed as a benign word sense)."""
    hits: dict[str, list[str]] = {}
    benign: dict[str, list[str]] = {}
    for name, pattern in (patterns or LEAK_PATTERNS).items():
        real, soft = [], []
        for match in pattern.finditer(text):
            token = match.group(0)
            if name.startswith("numeric") and NUMERIC_WHITELIST.fullmatch(token.strip()):
                continue
            (soft if benign_span(text, match.start(), match.end()) else real).append(token)
        if real:
            hits[name] = sorted(set(real))[:8]
        if soft:
            benign[name] = sorted(set(soft))[:8]
    return hits, benign


def short_is_truncation(long_text: str, short_text: str) -> dict[str, Any]:
    """A distilled short instruction restates; a truncated one is a prefix."""
    def norm(value: str) -> str:
        value = unicodedata.normalize("NFKC", value or "").strip().lower()
        return re.sub(r"\s+", " ", value)

    lng, srt = norm(long_text), norm(short_text)
    prefix = 0
    for a, b in zip(lng, srt):
        if a != b:
            break
        prefix += 1
    prefix_ratio = prefix / max(len(srt), 1)
    long_words = set(re.findall(r"[a-z]+", lng))
    short_words = set(re.findall(r"[a-z]+", srt))
    novel = short_words - long_words
    return {
        "len_long": len(long_text or ""),
        "len_short": len(short_text or ""),
        "len_ratio": round(len(short_text or "") / max(len(long_text or ""), 1), 4),
        "prefix_ratio": round(prefix_ratio, 4),
        # A short that reproduces >=90% of itself as a literal prefix of the long
        # form is a cut, not a distillation.
        "is_prefix_truncation": prefix_ratio >= 0.9 and len(srt) > 20,
        "short_unterminated": bool(srt) and not srt.rstrip().endswith((".", "!", "?")),
        "short_novel_words": sorted(novel)[:8],
        "short_longer_than_long": len(short_text or "") >= len(long_text or ""),
    }


def style_universe(groups: Iterable[Mapping[str, Any]]) -> list[str]:
    names: set[str] = set()
    for group in groups:
        for candidate in group.get("candidates", []) or []:
            for key in ("style_name", "major", "minor"):
                value = str(candidate.get(key) or "").strip()
                if len(value) >= 2:
                    names.add(value)
    return sorted(names)


def check_unit(
    unit_id: str,
    sft_id: str,
    variant: str,
    task_type: str,
    instruction: str,
    instruction_short: str,
    reasoning: str,
    style_name: str | None,
    styles: list[str],
    contract: str = "auto",
) -> dict[str, Any]:
    sections = check_sections(reasoning, contract)
    contract = sections["contract"]
    bodies = sections.pop("bodies")
    scope_text = bodies.get("region_scope", "")
    # region_scope is the only place shape vocabulary is legal, so it is searched
    # with its own pattern set and kept out of the restricted prose below.
    restricted_reasoning = strip_sections(
        REGION_SCOPE_SPAN_RE.sub(" ", reasoning or "")
    )
    prose = " \n".join(
        [instruction or "", instruction_short or "", restricted_reasoning]
    )
    hits, benign = leak_hits(prose)
    scope_hits, scope_benign = leak_hits(scope_text, SCOPE_LEAK_PATTERNS)
    # Structural tokens are not leakage; only the free text is searched above.
    style_named = []
    if task_type != "style":
        # names are dominantly CJK; a bare substring test is exact here.
        style_named = sorted({name for name in styles if name and name in prose})
    style_verbatim = None
    if task_type == "style" and style_name:
        style_verbatim = style_name in (instruction or "")
    instruction_words = word_count(instruction)
    short_words = word_count(instruction_short)
    geometry_in_instruction = sorted(set(
        geometry_words(instruction or "") + geometry_words(instruction_short or "")
    ))
    geometry_in_reasoning = geometry_words(restricted_reasoning)
    numeric_geometry = sorted({
        match.group(0).strip().lower()
        for match in NUMERIC_GEOMETRY_ECHO.finditer(
            prose + " \n" + scope_text
        )
    })[:8]
    # The v4-only rules below are gated on the contract: the legacy rows were
    # written under a prompt that *asked* for the coarse region by name and set no
    # word budget, so flagging them would corrupt the WP5-comparable summary.  The
    # measurements themselves are recorded for every unit.
    is_v4 = contract == "v4"
    flags = {
        "sections_incomplete": not sections["sections_present"],
        "sections_misordered": not sections["sections_ordered"],
        "sections_unbalanced": not sections["sections_balanced"],
        "sections_empty": bool(sections["empty_sections"]),
        "region_scope_missing": is_v4 and not scope_text,
        "global_scope_not_degenerate": (
            is_v4 and task_type == "style"
            and scope_text.rstrip(".").strip().lower() != GLOBAL_REGION_SCOPE
        ),
        "style_name_missing": task_type == "style" and style_verbatim is False,
        "local_names_style": bool(style_named),
        "cjk_in_local": task_type != "style" and bool(CJK_RE.search(prose)),
        "leak_any": bool(hits) or bool(scope_hits),
        **{f"leak_{name}": True for name in hits},
        **{f"scope_leak_{name}": True for name in scope_hits},
        "geometry_word_in_instruction": is_v4 and bool(geometry_in_instruction),
        "geometry_word_in_reasoning": is_v4 and bool(geometry_in_reasoning),
        "numeric_geometry_echo": is_v4 and bool(numeric_geometry),
        "instruction_over_cap": is_v4 and instruction_words > INSTRUCTION_WORD_CAP,
        "instruction_short_over_cap": (
            is_v4 and short_words > INSTRUCTION_SHORT_WORD_CAP
        ),
    }
    short = short_is_truncation(instruction, instruction_short)
    flags["short_is_truncation"] = short["is_prefix_truncation"]
    flags["short_degenerate"] = short["short_longer_than_long"] or short["len_short"] < 20
    short["words_long"] = instruction_words
    short["words_short"] = short_words
    return {
        "unit_id": unit_id,
        "sft_id": sft_id,
        "variant": variant,
        "contract": contract,
        "task_type": task_type,
        "style_name": style_name,
        "flags": {k: v for k, v in flags.items() if v},
        "n_flags": sum(1 for v in flags.values() if v),
        "sections": sections,
        "section_lengths": {k: len(v) for k, v in bodies.items()},
        "region_scope": scope_text,
        "leak_hits": hits,
        "leak_benign": benign,
        "scope_leak_hits": scope_hits,
        "scope_leak_benign": scope_benign,
        "geometry_words": {
            "instruction": geometry_in_instruction, "reasoning": geometry_in_reasoning,
        },
        "numeric_geometry": numeric_geometry,
        "style_named": style_named,
        "short": short,
    }


def iter_units(include_reannot: bool) -> Iterator[dict[str, Any]]:
    sft = rows(BUILD_ROOT / "sft.jsonl")
    groups = rows(BUILD_ROOT / "groups.jsonl")
    styles = style_universe(groups)
    style_by_candidate = {
        candidate["candidate_id"]: candidate.get("style_name")
        for group in groups
        for candidate in group.get("candidates", []) or []
    }
    for row in sft:
        yield {
            "unit_id": f"cur::{row['sft_id']}",
            "sft_id": row["sft_id"],
            "variant": "current",
            "task_type": row["task_type"],
            "instruction": row["instruction"],
            "instruction_short": row["instruction_short"],
            "reasoning": row["reasoning"],
            "style_name": style_by_candidate.get(row.get("candidate_id")),
            "styles": styles,
        }
    if not include_reannot:
        return
    comparison = json.loads(
        (REVIEW_ROOT / "reannot_ab" / "final_comparison.json").read_text()
    )
    task_types = {row["sft_id"]: row["task_type"] for row in sft}
    style_names = {
        row["sft_id"]: style_by_candidate.get(row.get("candidate_id")) for row in sft
    }
    for pair in comparison:
        for variant in ("before", "after"):
            text = pair[variant]
            yield {
                "unit_id": f"{variant}::{pair['sft_id']}",
                "sft_id": pair["sft_id"],
                "variant": f"reannot_{variant}",
                "task_type": task_types.get(pair["sft_id"], "local"),
                "instruction": text["instruction"],
                "instruction_short": text["instruction_short"],
                "reasoning": text["reasoning"],
                "style_name": style_names.get(pair["sft_id"]),
                "styles": styles,
            }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    # Required, with no default: the default used to be the WP5 evidence
    # directory, so a bare re-run overwrote a relay-answer set that cannot be
    # rebuilt.  Naming the destination is one word and it is never the wrong one.
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--include-reannot", action="store_true", default=True)
    parser.add_argument("--no-reannot", dest="include_reannot", action="store_false")
    parser.add_argument(
        "--contract", choices=["auto", "v4", "legacy"], default="auto",
        help="section contract to check against; auto detects it per unit",
    )
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    results = [check_unit(
        unit["unit_id"], unit["sft_id"], unit["variant"], unit["task_type"],
        unit["instruction"], unit["instruction_short"], unit["reasoning"],
        unit["style_name"], unit["styles"], args.contract,
    ) for unit in iter_units(args.include_reannot)]

    out_path = args.out / "mech_flags.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    summary: dict[str, Any] = {}
    for variant in sorted({r["variant"] for r in results}):
        subset = [r for r in results if r["variant"] == variant]
        counter: Counter[str] = Counter()
        for result in subset:
            counter.update(result["flags"].keys())
        summary[variant] = {
            "n": len(subset),
            "clean": sum(1 for r in subset if not r["flags"]),
            "flags": dict(counter.most_common()),
            "by_task": dict(Counter(r["task_type"] for r in subset)),
            "by_contract": dict(Counter(r["contract"] for r in subset)),
        }
    (args.out / "mech_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    print(json.dumps({"out": str(out_path), "units": len(results), "summary": summary},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
