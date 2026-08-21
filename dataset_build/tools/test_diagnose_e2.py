"""E2: three-arm comparison of how histogram evidence is delivered to the diagnosis stage.

Arm A - histogram board PNG at native 800x1000, ``detail=high``. Reused verbatim from the
E1 run (``docs/assets/diagnose_v2_test_20260821/records.jsonl``); no call is re-issued.
Arm B - no board image at all; the same four panels are handed over as a compact text
histogram block (L* 8 bins + p1/p99 + both clipped shares, R/G/B 8 bins each + channel
means/medians, C*ab 4 bins, hue 6 sectors) and the rules cite data rows instead of panels.
Arm C - the same board PNG as arm A, but pushed through the agent-loop image contract
(longest edge 512, JPEG q85, subsampling 0, ``detail=low``).

Everything else is held fixed: same 10 sources, same v2 rule skeleton, ``gpt-5.6-terra``
at ``reasoning_effort=high``, same strict JSON schema, temperature 0.1.

This is a *test* harness: it imports the E1 harness read-only, changes no prompt revision
and writes nothing into the agent-loop stores. Output lands in
``docs/assets/diagnose_v2_test_20260821/e2/``.

Usage::

    .venv/bin/python -m dataset_build.tools.test_diagnose_e2 --limit 10
    .venv/bin/python -m dataset_build.tools.test_diagnose_e2 --score-only
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import random
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from dataset_build.agent_loop.prompts import DIAGNOSIS_SCHEMA
from dataset_build.tools.test_diagnose_v2 import (
    C_BINS, C_MAX, FIVE_FIELDS, HUE_CHROMA_MIN, HUE_SECTOR_LABELS, L_BINS,
    MODEL, REASONING_EFFORT, REPO_ROOT, RGB_BINS, V2_RULES,
    _consume, _data_url, endpoint_credentials, load_sources, panel_stats,
    preview_jpeg, render_board, schema_error, select_sources,
)

DEFAULT_SOURCES = Path("/home/bc/data/agent_loop/local-v1/sources5k.annotated.jsonl")
DEFAULT_DATABUILD = REPO_ROOT / "databuild.prod-l8-local400k-20260812.toml"
DEFAULT_ENDPOINT = "provider-c-lane-1"
E1_DIR = REPO_ROOT / "docs/assets/diagnose_v2_test_20260821"
DEFAULT_OUT = E1_DIR / "e2"

# agent_loop.prompts.IMAGE_ENCODING, mirrored here so the test harness never imports the
# live encoder: {"format": "jpeg", "longest_edge": 512, "quality": 85, "subsampling": 0}
COMPRESSED_LONGEST_EDGE = 512
COMPRESSED_QUALITY = 85
COMPRESSED_SUBSAMPLING = 0

TEXT_L_BINS = 8
TEXT_RGB_BINS = 8
TEXT_C_BINS = 4


# ------------------------------------------------------------------ arm B: text block
def _rebin(values: Sequence[float], groups: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    return [float(chunk.sum()) for chunk in np.array_split(array, groups)]


def text_histogram_block(stats: Mapping[str, Any]) -> str:
    """The four panels of the board, as text. Same numbers, no picture."""
    lines: list[str] = ["The following text histogram data was measured on the "
                        "full-resolution pixels of Image 1."]

    lines.append("")
    lines.append("[L* data rows] lightness, 8 bins over 0-100, share of the frame:")
    l_bins = _rebin(stats["l_hist"], TEXT_L_BINS)
    edges = [index * 100.0 / TEXT_L_BINS for index in range(TEXT_L_BINS + 1)]
    lines.append("  " + " | ".join(
        f"{edges[i]:.0f}-{edges[i + 1]:.0f}: {share * 100:.2f}%"
        for i, share in enumerate(l_bins)
    ))
    lines.append(f"  L* mean {stats['l_mean']:.1f}   p1 {stats['l_p1']:.1f}   "
                 f"median {stats['l_p50']:.1f}   p99 {stats['l_p99']:.1f}")
    lines.append(f"  clipped low (L*<2): {stats['clip_low'] * 100:.2f}%     "
                 f"clipped high (L*>98): {stats['clip_high'] * 100:.2f}%")

    lines.append("")
    lines.append("[RGB data rows] R/G/B channel histograms, 8 bins over 0-255, share of "
                 "the frame per channel:")
    edges = [index * 256.0 / TEXT_RGB_BINS for index in range(TEXT_RGB_BINS + 1)]
    for channel, name in enumerate("RGB"):
        bins = _rebin(stats["rgb_hist"][channel], TEXT_RGB_BINS)
        lines.append(f"  {name}: " + " | ".join(
            f"{edges[i]:.0f}-{edges[i + 1]:.0f}: {share * 100:.2f}%"
            for i, share in enumerate(bins)
        ))
    means = stats["rgb_mean"]
    medians = stats["rgb_median"]
    lines.append(
        f"  channel mean   R {means[0]:.1f}   G {means[1]:.1f}   B {means[2]:.1f}"
        f"   (R-B {means[0] - means[2]:+.1f},  G-B {means[1] - means[2]:+.1f},  "
        f"R-G {means[0] - means[1]:+.1f})"
    )
    lines.append(f"  channel median R {medians[0]:.0f}   G {medians[1]:.0f}   "
                 f"B {medians[2]:.0f}")
    lines.append("  A channel whose histogram sits further to the right than the other "
                 "two (higher mean / median) is the direction of the colour cast; the "
                 "size of the gap is its strength.")

    lines.append("")
    lines.append("[C*ab data rows] chroma (saturation), 4 bins over 0-120, share of the "
                 "frame:")
    c_bins = _rebin(stats["c_hist"], TEXT_C_BINS)
    edges = [index * C_MAX / TEXT_C_BINS for index in range(TEXT_C_BINS + 1)]
    lines.append("  " + " | ".join(
        f"{edges[i]:.0f}-{edges[i + 1]:.0f}: {share * 100:.2f}%"
        for i, share in enumerate(c_bins)
    ))
    lines.append(f"  C*ab mean {stats['c_mean']:.1f}   median {stats['c_p50']:.1f}   "
                 f"p95 {stats['c_p95']:.1f}   share C*>60: {stats['c_over60'] * 100:.2f}%")
    lines.append(f"  chromatic pixels (C*ab>={HUE_CHROMA_MIN:.0f}): "
                 f"{stats['chromatic_share'] * 100:.1f}% of the frame")

    lines.append("")
    lines.append("[hue data rows] 6 Lab hue sectors, share of the whole frame:")
    for label, share in zip(HUE_SECTOR_LABELS, stats["hue_shares"]):
        lines.append(f"  {label}: {share * 100:.1f}%")
    return "\n".join(lines)


_B_REPLACEMENTS = (
    (
        """You receive two images. Image 1 is the source photograph (a downscaled preview; \
downscaling hides subtle colour casts and mild clipping, so do not go by your eyes alone). \
Image 2 is a histogram board rendered from the full-resolution pixel data, in four panels:
Panel 1 - L* lightness histogram (x axis 0-100, with the p1 / p99 vertical lines and the \
clipped share printed at both ends): read the exposure centre of mass, tonal crowding, blocked \
shadows (pile-up at the left end) and clipped highlights (pile-up at the right end).
Panel 2 - R/G/B channel histograms drawn on top of each other: the direction in which the three \
curves are offset from one another IS the direction of the colour cast (blue channel shifted \
right overall = a blue cast), and the size of the offset is its strength. This is far more \
reliable colour-cast evidence than the naked eye.
Panel 3 - C*ab chroma (saturation) histogram: the overall saturation level and the \
over-saturated tail.
Panel 4 - hue share bars: the area share of the leading hues.""",
        """You receive one image and one block of text histogram data. Image 1 is the source \
photograph (a downscaled preview; downscaling hides subtle colour casts and mild clipping, so \
do not go by your eyes alone). Below the rules you get the text histogram data measured on the \
full-resolution pixel data, in four groups of data rows:
The L* data rows - the L* lightness histogram (8 bins over 0-100, plus the mean / p1 / median / \
p99 and the clipped share at both ends): read the exposure centre of mass, tonal crowding, \
blocked shadows (weight piled into the leftmost bins) and clipped highlights (weight piled into \
the rightmost bins).
The RGB data rows - the R/G/B channel histograms (8 bins each over 0-255, plus each channel's \
mean and median): the direction in which the three channels are offset from one another IS the \
direction of the colour cast (the blue channel sitting further right overall = a blue cast), and \
the size of the offset is its strength. This is far more reliable colour-cast evidence than the \
naked eye.
The C*ab data rows - the C*ab chroma (saturation) histogram: the overall saturation level and \
the over-saturated tail.
The hue data rows - the area share of the leading hues.""",
    ),
    ("1. Exposure and tonality - mainly Panel 1, with Image 1 as support",
     "1. Exposure and tonality - mainly the L* data rows, with Image 1 as support"),
    ("2. White balance and colour cast - mainly Panel 2: the direction and the magnitude of the \
channel offsets",
     "2. White balance and colour cast - mainly the RGB data rows: the direction and the \
magnitude of the channel offsets"),
    ("3. Colour state - Panels 3 and 4 plus Image 1",
     "3. Colour state - the C*ab and hue data rows plus Image 1"),
    ("4. Contrast and texture - Image 1 plus the shape of Panel 1",
     "4. Contrast and texture - Image 1 plus the shape of the L* data rows"),
    ("state which image you took the evidence from",
     "state whether you took the evidence from Image 1 or from the histogram data"),
    ("A histogram reading must cite its panel number.",
     "A histogram reading must cite which group of data rows it came from (the L*, RGB, "
     "C*ab or hue data rows)."),
)


def build_b_rules() -> str:
    text = V2_RULES
    for old, new in _B_REPLACEMENTS:
        if old not in text:
            raise SystemExit(f"arm-B rewrite anchor missing: {old[:60]!r}")
        text = text.replace(old, new)
    if re.search(r"[Pp]anel", text):
        raise SystemExit("arm-B rules still mention a panel")
    return text


B_RULES = build_b_rules()


# ------------------------------------------------------------------ arm C: compression
def compress_board(board_png: bytes) -> bytes:
    """The agent-loop image contract, applied to the board."""
    with Image.open(io.BytesIO(board_png)) as source:
        image = source.convert("RGB")
        if max(image.size) > COMPRESSED_LONGEST_EDGE:
            scale = COMPRESSED_LONGEST_EDGE / max(image.size)
            size = tuple(max(1, round(value * scale)) for value in image.size)
            image = image.resize(size, Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, "JPEG", quality=COMPRESSED_QUALITY, optimize=False,
                   progressive=False, subsampling=COMPRESSED_SUBSAMPLING)
    return output.getvalue()


# ------------------------------------------------------------------ transport
def call_arm(client: Any, content: list[dict[str, Any]], *, attempts: int = 3
             ) -> dict[str, Any]:
    payload = {
        "model": MODEL,
        "input": [{"role": "user", "content": content}],
        "stream": True,
        "temperature": 0.1,
        "max_output_tokens": 4096,
        "store": False,
        "reasoning": {"effort": REASONING_EFFORT},
        "text": {"format": {"type": "json_schema", "name": "source_diagnosis_v1",
                            "strict": True, "schema": DIAGNOSIS_SCHEMA}},
    }
    last_error = "unknown"
    for attempt in range(1, attempts + 1):
        try:
            result = _consume(client.responses.create(**payload))
            if result["model"] != MODEL:
                last_error = f"model_substitution:{result['model']}"
            else:
                try:
                    parsed = json.loads(result["text"])
                except json.JSONDecodeError:
                    last_error = "invalid_json"
                else:
                    error = schema_error(parsed)
                    if error is None:
                        return {"parsed": parsed, "usage": result["usage"],
                                "attempts": attempt}
                    last_error = error
        except Exception as exc:
            last_error = f"{type(exc).__name__}:{str(exc)[:160]}"
        if attempt < attempts:
            print(f"    retry {attempt + 1}/{attempts} after {last_error}", flush=True)
            # the relay answers 502 / rate_limit_exceeded in bursts; back off long
            time.sleep(min(120.0, 10.0 * attempt + random.random() * 5.0))
    raise RuntimeError(last_error)


def arm_content(arm: str, preview: bytes, board_png: bytes, stats: Mapping[str, Any]
                ) -> list[dict[str, Any]]:
    if arm == "B":
        return [
            {"type": "input_text", "text": B_RULES},
            {"type": "input_image", "image_url": _data_url(preview, "image/jpeg"),
             "detail": "low"},
            {"type": "input_text", "text": text_histogram_block(stats)},
        ]
    if arm == "C":
        return [
            {"type": "input_text", "text": V2_RULES},
            {"type": "input_image", "image_url": _data_url(preview, "image/jpeg"),
             "detail": "low"},
            {"type": "input_image",
             "image_url": _data_url(compress_board(board_png), "image/jpeg"),
             "detail": "low"},
        ]
    raise ValueError(arm)


def field_items(diagnosis: Mapping[str, Any]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for name in FIVE_FIELDS:
        for item in diagnosis.get(name) or []:
            out.append((name, str(item)))
    return out


PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w])")

# a percentage only counts as a clip reading when the wording right around it names
# the clipped end; anything else (hue shares, chroma shares, bin shares) is not one
_AFTER_LOW = re.compile(
    r"^\s*(?:of (?:the )?(?:pixels|frame)\s*)?(?:are|is|lie|lies|sits?)?\s*"
    r"(?:at or )?(?:below|under)\s*L\*?\s*[<=]*\s*2\b"
    r"|^\s*(?:clipped|clipping)\s*(?:low|at the (?:low|dark|shadow) end|in the shadows)"
    r"|^\s*(?:black|shadow|low-end|low end)[- ]?clipping"
    r"|^\s*(?:at|in) L\*\s*0-2\b", re.IGNORECASE)
_AFTER_HIGH = re.compile(
    r"^\s*(?:of (?:the )?(?:pixels|frame)\s*)?(?:are|is|lie|lies|sits?)?\s*"
    r"(?:at or )?(?:above|over|exceed(?:s|ing)?)\s*L\*?\s*[>=]*\s*98\b"
    r"|^\s*(?:clipped|clipping)\s*(?:high|highlights?|at the (?:high|bright) end)"
    r"|^\s*(?:highlight|high-end|high end|bright-end)[- ]?clipping",
    re.IGNORECASE)
_BEFORE_LOW = re.compile(
    r"(?:clipped[- ]low|clipped at the low end|low[- ]end clipping|shadow clipping|"
    r"black clipping|clipping (?:below|at) L\*?\s*[<=]*\s*2|below L\*?\s*[<=]*\s*2 (?:is|share is)|"
    r"clipped (?:in the )?shadows?)"
    r"[^%]{0,25}$", re.IGNORECASE)
_BEFORE_HIGH = re.compile(
    r"(?:clipped[- ]high|clipped highlights?|high[- ]end clipping|highlight clipping|"
    r"clipping (?:above|at) L\*?\s*[>=]*\s*98|above L\*?\s*[>=]*\s*98 (?:is|share is)|"
    r"clipped at the high end)"
    r"[^%]{0,25}$", re.IGNORECASE)


def clip_readings(diagnosis: Mapping[str, Any], stats: Mapping[str, Any]
                  ) -> list[dict[str, Any]]:
    """Percentages the answer quotes as a clipped share, with their truth."""
    truth = {"clip_low": stats["clip_low"] * 100, "clip_high": stats["clip_high"] * 100}
    out: list[dict[str, Any]] = []
    for name, item in field_items(diagnosis):
        for match in PERCENT.finditer(item):
            after = item[match.end():match.end() + 60]
            before = item[max(0, match.start() - 70):match.start()]
            # the wording that follows the number wins over the wording before it:
            # "0.00% clipped low and 0.01% clipped high" must not put both on the low end
            if _AFTER_LOW.search(after):
                target = "clip_low"
            elif _AFTER_HIGH.search(after):
                target = "clip_high"
            elif _BEFORE_LOW.search(before):
                target = "clip_low"
            elif _BEFORE_HIGH.search(before):
                target = "clip_high"
            else:
                continue
            value = float(match.group(1))
            reference = truth[target]
            if abs(reference) < 0.005:
                rel = 0.0 if value < 0.005 else float("inf")
            else:
                rel = abs(value - reference) / reference
            out.append({"field": name, "value": value, "target": target,
                        "truth": reference, "rel_err": rel, "ok": bool(rel <= 0.30),
                        "text": item})
    return out


CITE_PATTERN = re.compile(
    r"panel\s*[1-4]"
    r"|(?:L\*|RGB|R/G/B|C\*ab|hue|histogram)\s*(?:data\s*)?(?:row|rows|block|section)"
    r"|(?:the\s+)?histogram\s+data",
    re.IGNORECASE)


def field_items(diagnosis: Mapping[str, Any]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for name in FIVE_FIELDS:
        for item in diagnosis.get(name) or []:
            out.append((name, str(item)))
    return out


def extract_cast_claims(diagnosis: Mapping[str, Any]) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for name, item in field_items(diagnosis):
        if not CAST_CONTEXT.search(item):
            continue
        labels = sorted({
            label for label, patterns in CAST_LEXICON
            for pattern in patterns if re.search(pattern, item, re.IGNORECASE)
        })
        if labels:
            claims.append({"field": name, "labels": labels, "text": item})
    return claims


def _cumulative(name: str, shares: Sequence[float]) -> list[tuple[str, float]]:
    """Individual bins plus every prefix and suffix run, in percent."""
    out: list[tuple[str, float]] = []
    total = len(shares)
    for start in range(total):
        running = 0.0
        for end in range(start, total):
            running += shares[end]
            out.append((f"{name}[{start + 1}:{end + 1}]", running * 100))
    return out


def truth_numbers(stats: Mapping[str, Any]) -> list[tuple[str, float]]:
    """Every number a diagnosis is allowed to quote as a histogram reading."""
    out: list[tuple[str, float]] = [
        ("l_mean", stats["l_mean"]), ("l_p1", stats["l_p1"]),
        ("l_p50", stats["l_p50"]), ("l_p99", stats["l_p99"]),
        ("clip_low_pct", stats["clip_low"] * 100),
        ("clip_high_pct", stats["clip_high"] * 100),
        ("c_mean", stats["c_mean"]), ("c_p50", stats["c_p50"]), ("c_p95", stats["c_p95"]),
        ("c_over60_pct", stats["c_over60"] * 100),
        ("chromatic_pct", stats["chromatic_share"] * 100),
        ("non_chromatic_pct", 100.0 - stats["chromatic_share"] * 100),
    ]
    for channel, name in enumerate("RGB"):
        out.append((f"{name}_mean", stats["rgb_mean"][channel]))
        out.append((f"{name}_median", stats["rgb_median"][channel]))
    means = stats["rgb_mean"]
    out.append(("R-B", means[0] - means[2]))
    out.append(("G-B", means[1] - means[2]))
    out.append(("R-G", means[0] - means[1]))
    medians = stats["rgb_median"]
    out.append(("Rmed-Bmed", medians[0] - medians[2]))
    out.append(("Gmed-Bmed", medians[1] - medians[2]))
    out.append(("Rmed-Gmed", medians[0] - medians[1]))
    out.extend(_cumulative("hue", stats["hue_shares"]))
    out.extend(_cumulative("Lbin", _rebin(stats["l_hist"], TEXT_L_BINS)))
    out.extend(_cumulative("Cbin", _rebin(stats["c_hist"], TEXT_C_BINS)))
    for channel, name in enumerate("RGB"):
        out.extend(_cumulative(
            f"{name}bin", _rebin(stats["rgb_hist"][channel], TEXT_RGB_BINS)))
    return out


# A labelled reading names the statistic it quotes, so it can be checked against that
# one truth value. Only high-precision wordings are read; anything ambiguous is left to
# the looser `numeric_audit`.
_V = r"([+-]?\d+(?:\.\d+)?)"
_TRIPLE = re.compile(
    rf"R\s*(?:mean\s*)?(?:is\s*|=\s*|of\s*)?{_V}\s*,?\s*(?:and\s+)?"
    rf"G\s*(?:mean\s*)?(?:is\s*|=\s*|of\s*)?{_V}\s*,?\s*(?:and\s+)?"
    rf"B\s*(?:mean\s*)?(?:is\s*|=\s*|of\s*)?{_V}", re.IGNORECASE)
_SIMPLE: tuple[tuple[str, re.Pattern[str], str | None, str | None], ...] = (
    # key, pattern, required word before (within 40 chars), forbidden word before
    ("l_mean", re.compile(rf"L\*?\s*mean\s*(?:is|of|=)?\s*{_V}", re.I), "L", None),
    ("l_mean", re.compile(rf"mean\s*L\*\s*(?:is|of|=)?\s*{_V}", re.I), "L", None),
    ("l_p50", re.compile(rf"L\*?\s*median\s*(?:is|of|=)?\s*{_V}", re.I), "L", None),
    ("l_p50", re.compile(rf"median\s*(?:is|of|=)?\s*L\*\s*{_V}", re.I), "L", None),
    ("l_p1", re.compile(rf"\bp1\s*(?:is|of|=|at)?\s*L?\*?\s*{_V}", re.I), "L", None),
    ("l_p99", re.compile(rf"\bp99\s*(?:is|of|=|at)?\s*L?\*?\s*{_V}", re.I), "L", None),
    ("R-B", re.compile(rf"R\s*-\s*B\s*(?:is|of|=|difference)?\s*{_V}", re.I), None, None),
    ("G-B", re.compile(rf"G\s*-\s*B\s*(?:is|of|=|difference)?\s*{_V}", re.I), None, None),
    ("R-G", re.compile(rf"R\s*-\s*G\s*(?:is|of|=|difference)?\s*{_V}", re.I), None, None),
    ("c_mean", re.compile(rf"C\*ab\s*(?:chroma\s*)?mean\s*(?:is|of|=)?\s*{_V}", re.I),
     None, None),
    ("c_mean", re.compile(rf"mean\s*(?:C\*ab|chroma)\s*(?:is|of|=)?\s*{_V}", re.I),
     None, None),
    ("c_p95", re.compile(rf"\bp95\s*(?:is|of|=)?\s*{_V}", re.I), None, None),
)


def named_readings(diagnosis: Mapping[str, Any], stats: Mapping[str, Any]
                   ) -> list[dict[str, Any]]:
    """Readings that name their own statistic, checked against that statistic."""
    truth = dict(truth_numbers(stats))

    def entry(field: str, key: str, value: float, text: str) -> dict[str, Any]:
        reference = truth[key]
        gap = min(abs(value - reference), abs(value - abs(reference)))
        return {"field": field, "stat": key, "value": value, "truth": reference,
                "gap": gap, "ok": bool(gap <= max(0.6, abs(reference) * 0.005)),
                "text": text}

    out: list[dict[str, Any]] = []
    for name, item in field_items(diagnosis):
        seen: set[tuple[str, float]] = set()
        for match in _TRIPLE.finditer(item):
            window = item[max(0, match.start() - 90):match.start()].lower()
            # whichever of the two words sits closest to the triple wins
            median = window.rfind("median") > window.rfind("mean")
            for offset, channel in enumerate("RGB"):
                key = f"{channel}_median" if median else f"{channel}_mean"
                value = float(match.group(offset + 1))
                if (key, value) not in seen:
                    seen.add((key, value))
                    out.append(entry(name, key, value, item))
        for key, pattern, axis, _unused in _SIMPLE:
            for match in pattern.finditer(item):
                window = item[max(0, match.start() - 90):match.start()].lower()
                # a bare "p1 / p95 / p99" belongs to whichever axis was named last
                lightness = max(window.rfind("l*"), window.rfind("lightness"))
                chroma = max(window.rfind("c*ab"), window.rfind("chroma"))
                if axis == "L" and chroma > lightness:
                    continue
                if key == "c_p95" and lightness > chroma:
                    continue
                value = float(match.group(1))
                if (key, value) in seen:
                    continue
                seen.add((key, value))
                out.append(entry(name, key, value, item))
    return out


# "Panel 3", "Layer 6", "Image 1" are references, not readings
_REFERENCE_PREFIX = re.compile(r"(?:panels?|layers?|images?|figure|stage)\s*$",
                               re.IGNORECASE)


def citation_count(diagnosis: Mapping[str, Any]) -> int:
    return sum(1 for _, item in field_items(diagnosis) if CITE_PATTERN.search(item))


ARM_LABEL = {
    "A": "A - board PNG 800x1000, detail=high",
    "B": "B - text histogram, no board image",
    "C": "C - board JPEG 512/q85, detail=low",
}
ARM_SHORT = {"A": "A board(high)", "B": "B text", "C": "C board(512/q85/low)"}


def _cells(values: Any) -> str:
    if not isinstance(values, list) or not values:
        return "_(empty)_"
    out = []
    for item in values:
        text = str(item).replace("|", "\\|").replace("\n", " ")
        out.append("- " + text)
    return "<br>".join(out)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def write_report(records: Sequence[Mapping[str, Any]], out_dir: Path) -> Path:
    """Raw three-arm dump. No verdicts: the numbers and the answers, side by side."""
    by_id = {row["source_id"]: row for row in records}

    lines: list[str] = []
    lines.append("# E2 - histogram board vs equal-information text vs compressed board")
    lines.append("")
    lines.append(f"- model `{MODEL}`, `reasoning_effort={REASONING_EFFORT}`, "
                 "temperature 0.1, strict JSON, same schema as the frozen v1 batch")
    lines.append("- same 10 sources as E1 (first 10 rows of `sources5k.annotated.jsonl` "
                 "in ascending `sha1(source_id)` order), same v2 rule skeleton")
    lines.append("- every arm gets the same 512px source preview at `detail=low`")
    lines.append("")
    lines.append("| arm | image 2 | histogram evidence | rules |")
    lines.append("| --- | --- | --- | --- |")
    lines.append("| A | 800x1000 PNG board, `detail=high` | 4 rendered panels | "
                 "E1 v2 rules verbatim |")
    lines.append("| B | none | text block: L* 8 bins + mean/p1/median/p99 + both clipped "
                 "shares, R/G/B 8 bins each + channel means/medians + R-B/G-B/R-G, C*ab "
                 "4 bins + mean/median/p95/share>60 + chromatic share, hue 6 sectors | "
                 "v2 rules with the panel paragraph and every panel reference rewritten "
                 "to data rows (`prompt_arm_b.txt`) |")
    lines.append("| C | same board through the agent-loop image contract (longest edge "
                 "512, JPEG q85, subsampling 0), `detail=low` | 4 rendered panels, "
                 "compressed | E1 v2 rules verbatim |")
    lines.append("")
    lines.append("Arm A is reused verbatim from the E1 run; no A call was re-issued. "
                 "A's token counts therefore come from the E1 run, B's and C's from this "
                 "one.")
    lines.append("")

    lines.append("## Totals")
    lines.append("")
    header = "| metric | " + " | ".join(ARM_SHORT[a] for a in "ABC") + " |"
    lines.append(header)
    lines.append("| --- | --- | --- | --- |")

    def usage_of(row: Mapping[str, Any], arm: str, key: str) -> int:
        usage = (row["arms"][arm] or {}).get("usage") or {}
        return int(usage.get(key, 0) or 0)

    def parsed_of(row: Mapping[str, Any], arm: str) -> Mapping[str, Any]:
        return (row["arms"][arm] or {}).get("parsed") or {}

    def line(label: str, render: Any) -> None:
        lines.append(f"| {label} | " + " | ".join(render(arm) for arm in "ABC") + " |")

    line("schema-valid answers",
         lambda a: f"{sum(1 for r in records if parsed_of(r, a))}/{len(records)}")
    line("mean input tokens",
         lambda a: f"{_mean([usage_of(r, a, 'input_tokens') for r in records]):.1f}")
    line("mean output tokens",
         lambda a: f"{_mean([usage_of(r, a, 'output_tokens') for r in records]):.1f}")
    line("items across the five fields",
         lambda a: str(sum(len(parsed_of(r, a).get(name) or [])
                           for r in records for name in FIVE_FIELDS)))
    line("items naming a panel / data row",
         lambda a: str(sum(citation_count(parsed_of(r, a)) for r in records)))
    for mode in ("correction_led", "enhancement_led", "mixed"):
        line(f"intent_mode = {mode}", lambda a, mode=mode: str(sum(
            1 for r in records if parsed_of(r, a).get("intent_mode") == mode)))
    lines.append("")

    lines.append("| # | source_id | A in/out | B in/out | C in/out | A items | B items | "
                 "C items | A intent | B intent | C intent |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for index, record in enumerate(records, start=1):
        cells = [f"{usage_of(record, a, 'input_tokens')} / "
                 f"{usage_of(record, a, 'output_tokens')}" for a in "ABC"]
        cells += [str(sum(len(parsed_of(record, a).get(n) or []) for n in FIVE_FIELDS))
                  for a in "ABC"]
        cells += [str(parsed_of(record, a).get("intent_mode")) for a in "ABC"]
        lines.append(f"| {index} | `{record['source_id']}` | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Histogram ground truth per source")
    lines.append("")
    lines.append("Measured on the original-resolution file; these are exactly the numbers "
                 "the board prints and the text block lists.")
    lines.append("")
    lines.append("| # | source_id | scene | L* mean / p1 / median / p99 | clipped low % | "
                 "clipped high % | mean R / G / B | R-B | G-B | R-G | C*ab mean / median "
                 "/ p95 | share C*>60 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for index, record in enumerate(records, start=1):
        stats = record["stats"]
        means = stats["rgb_mean"]
        lines.append(
            f"| {index} | `{record['source_id']}` | {record['scene']} | "
            f"{stats['l_mean']:.1f} / {stats['l_p1']:.1f} / {stats['l_p50']:.1f} / "
            f"{stats['l_p99']:.1f} | {stats['clip_low'] * 100:.2f} | "
            f"{stats['clip_high'] * 100:.2f} | {means[0]:.1f} / {means[1]:.1f} / "
            f"{means[2]:.1f} | {means[0] - means[2]:+.1f} | {means[1] - means[2]:+.1f} | "
            f"{means[0] - means[1]:+.1f} | {stats['c_mean']:.1f} / {stats['c_p50']:.1f} / "
            f"{stats['c_p95']:.1f} | {stats['c_over60'] * 100:.2f}% |"
        )
    lines.append("")

    lines.append("## Per-source, three arms, five fields side by side")
    lines.append("")
    for index, record in enumerate(records, start=1):
        source_id = record["source_id"]
        stats = record["stats"]
        means = stats["rgb_mean"]
        lines.append(f"### {index}. `{source_id}` ({record['scene']})")
        lines.append("")
        lines.append(
            f'<img src="../images/{source_id}.source.jpg" width="260"> '
            f'<img src="../images/{source_id}.board.png" width="300"> '
            f'<img src="images/{source_id}.board.q85.jpg" width="300">'
        )
        lines.append("")
        lines.append("(left: source preview, all arms | middle: arm A board as sent "
                     f"({record['board_png_bytes'] // 1024} kB PNG, `detail=high`) | "
                     f"right: arm C board as sent ({record['compressed_bytes'] // 1024} "
                     "kB JPEG at 512 long edge, `detail=low`))")
        lines.append("")
        lines.append(f"Subject: {record.get('subject_description') or '-'}")
        lines.append("")
        lines.append(
            f"Truth - L* mean {stats['l_mean']:.1f}, p1 {stats['l_p1']:.1f}, median "
            f"{stats['l_p50']:.1f}, p99 {stats['l_p99']:.1f}; clipped low "
            f"{stats['clip_low'] * 100:.2f}%, clipped high {stats['clip_high'] * 100:.2f}%; "
            f"mean R {means[0]:.1f} G {means[1]:.1f} B {means[2]:.1f} "
            f"(R-B {means[0] - means[2]:+.1f}, G-B {means[1] - means[2]:+.1f}, "
            f"R-G {means[0] - means[1]:+.1f}); median R {stats['rgb_median'][0]:.0f} "
            f"G {stats['rgb_median'][1]:.0f} B {stats['rgb_median'][2]:.0f}; C*ab mean "
            f"{stats['c_mean']:.1f}, median {stats['c_p50']:.1f}, p95 {stats['c_p95']:.1f}, "
            f"share C*>60 {stats['c_over60'] * 100:.2f}%; hue sectors "
            + ", ".join(f"{label.split(' ', 1)[1]} {share * 100:.1f}%"
                        for label, share in zip(HUE_SECTOR_LABELS, stats["hue_shares"]))
        )
        lines.append("")
        modes = " | ".join(
            f"{arm}: `{parsed_of(record, arm).get('intent_mode')}`" for arm in "ABC")
        lines.append(f"intent_mode - {modes}")
        lines.append("")
        for name in FIVE_FIELDS:
            lines.append(f"**{name}**")
            lines.append("")
            lines.append("| " + " | ".join(ARM_SHORT[a] for a in "ABC") + " |")
            lines.append("| --- | --- | --- |")
            lines.append("| " + " | ".join(
                _cells(parsed_of(record, a).get(name)) for a in "ABC") + " |")
            lines.append("")

    lines.append("## Appendix - every labelled histogram reading, as quoted")
    lines.append("")
    lines.append("Raw extraction, no verdict: each row is a statistic the answer named "
                 "explicitly, the value it quoted, and the value measured on the file. "
                 "Rows where the two differ are marked with `!=` so they are easy to "
                 "find; nothing is scored.")
    lines.append("")
    for index, record in enumerate(records, start=1):
        lines.append(f"**{index}. `{record['source_id']}`**")
        lines.append("")
        lines.append("| arm | statistic | quoted | measured | |")
        lines.append("| --- | --- | --- | --- | --- |")
        for arm in "ABC":
            parsed = parsed_of(record, arm)
            if not parsed:
                continue
            for reading in named_readings(parsed, record["stats"]):
                flag = "" if reading["ok"] else "!="
                lines.append(
                    f"| {arm} | {reading['stat']} | {reading['value']:g} | "
                    f"{reading['truth']:.2f} | {flag} |"
                )
            for reading in clip_readings(parsed, record["stats"]):
                flag = "" if reading["ok"] else "!="
                lines.append(
                    f"| {arm} | {reading['target']} % | {reading['value']:g} | "
                    f"{reading['truth']:.2f} | {flag} |"
                )
        lines.append("")

    target = out_dir / "report.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--databuild-config", type=Path, default=DEFAULT_DATABUILD)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=420.0)
    parser.add_argument("--read-timeout", type=float, default=120.0)
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--dump-prompts", action="store_true")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.dump_prompts:
        (args.out / "prompt_arm_b.txt").write_text(B_RULES + "\n", encoding="utf-8")
        selected = select_sources(load_sources(args.sources), 1)
        stats = panel_stats(selected[0]["source_path"])
        (args.out / "sample_text_block.txt").write_text(
            text_histogram_block(stats) + "\n", encoding="utf-8")
        print(text_histogram_block(stats))
        return 0

    path = args.out / "records.jsonl"
    if args.score_only:
        records = [json.loads(line) for line in
                   path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        records = build_records(args)
        path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True)
                      for row in records) + "\n", encoding="utf-8")
        (args.out / "prompt_arm_b.txt").write_text(B_RULES + "\n", encoding="utf-8")

    report = write_report(records, args.out)
    ok = {arm: sum(1 for row in records if (row["arms"][arm] or {}).get("parsed"))
          for arm in ("A", "B", "C")}
    print(f"records: {path}  succeeded {ok}", flush=True)
    print(f"report: {report}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
