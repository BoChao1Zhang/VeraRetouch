"""E1: standalone A/B harness for the v2 source-diagnosis prompt.

This is a *test* harness. It imports nothing from ``dataset_build.agent_loop`` except the
frozen ``DIAGNOSIS_SCHEMA`` (read-only), writes nothing into the agent-loop artifact or
audit stores, and does not touch any prompt revision. Everything it produces lands under
``docs/assets/diagnose_v2_test_20260821/``.

Pipeline per source:

1. Compute four histogram panels on the *original-resolution* file (no downscale).
2. Render them into one deterministic 640x674 PNG board with PIL (no matplotlib).
3. Send [512px source JPEG, histogram board PNG] plus the v2 rules to ``gpt-5.6-terra``
   at ``reasoning_effort=high`` with the same strict JSON schema the frozen v1 batch used.
4. Put the v2 answer next to the frozen v1 answer of the same source in one md report.

Usage::

    .venv/bin/python -m dataset_build.tools.test_diagnose_v2 --limit 10
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import random
import re
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from skimage.color import rgb2lab

from dataset_build.agent_loop.prompts import DIAGNOSIS_SCHEMA

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCES = Path("/home/bc/data/agent_loop/local-v1/sources5k.annotated.jsonl")
DEFAULT_DATABUILD = REPO_ROOT / "databuild.prod-l8-local400k-20260812.toml"
DEFAULT_OUT = REPO_ROOT / "docs/assets/diagnose_v2_test_20260821"
DEFAULT_ENDPOINT = "provider-c-lane-1"
MODEL = "gpt-5.6-terra"
REASONING_EFFORT = "high"
PROMPT_TAG = "diagnose-v3.4-lut-multiaxis-test"

PREVIEW_LONGEST_EDGE = 512
PREVIEW_QUALITY = 85
BOARD_SIZE = (640, 604)

# Panel binning. These are display bins of the board only; they are deliberately finer
# than the frozen 8-bin `source_histogram` contract, which this harness does not touch.
L_BINS = 64
RGB_BINS = 64
C_BINS = 48
C_MAX = 120.0
HUE_SECTORS = 6
HUE_CHROMA_MIN = 10.0
CLIP_LOW_L = 2.0
CLIP_HIGH_L = 98.0

FIVE_FIELDS = (
    "correction_needs", "preserve_intent", "enhancement_opportunities",
    "forbidden_directions", "evidence",
)

HUE_SECTOR_LABELS = (
    "0-60 red / orange",
    "60-120 yellow",
    "120-180 green",
    "180-240 cyan / teal",
    "240-300 blue",
    "300-360 magenta / pink",
)
HUE_SECTOR_COLORS = (
    (220, 55, 40), (225, 175, 20), (40, 170, 60),
    (30, 170, 185), (45, 80, 220), (190, 60, 175),
)


# --------------------------------------------------------------------- v3 prompt
# LUT-specialised revision: the downstream editor is a colour lookup table, the
# histogram reading is a mandatory first phase, and the actionable fields are locked
# to the closed keyword vocabulary that `direction_match.KEYWORD_AXES` and
# `LutCatalog._score` actually parse (an item outside that vocabulary steers nothing).
V2_RULES = """You are the source-diagnosis stage of a photo-retouch data system. Downstream, \
a deterministic retrieval parses your fixed-form lines verbatim to recall candidate LUTs, an \
LLM stage re-ranks that shortlist against your evidence, and the editor applies one LUT to \
the whole frame plus a second LUT inside the subject mask. The whole chain is colour-lookup \
editing only - write nothing a LUT cannot express.

You receive two images. Image 1 is the source photograph. Image 2 is a histogram board \
rendered from the full-resolution pixel data, in four panels:
Panel 1 - L* lightness histogram (x axis 0-100, with the p1 / p99 vertical lines and the \
clipped share printed at both ends).
Panel 2 - R/G/B channel histograms drawn on top of each other: the direction in which the \
three curves are offset from one another IS the direction of the colour cast (blue channel \
shifted right overall = a blue cast), and the size of the offset is its strength. This is far \
more reliable colour-cast evidence than the naked eye.
Panel 3 - C*ab chroma (saturation) histogram: the overall saturation level and the \
over-saturated tail.
Panel 4 - hue share bars: the area share of the leading hues; the six bars are always \
drawn top-to-bottom in the fixed hue order red, yellow, green, cyan, blue, magenta.

PHASE 1 - picture reading (Image 1, always first). Judge the photograph as a professional \
photographer and colourist: the scene and its mood; the exposure discipline (are highlights \
blown where detail matters, are shadows blocked where the eye expects depth); what reads as \
deliberate style versus defect (a warm sunset's cast is a mood, not an error); the memory \
colours (skin, sky, foliage); the palette harmony and where it clashes; the light and \
atmosphere assets a grade must not destroy; and, aesthetically, the finished looks this \
image could genuinely become.

PHASE 2 - histogram reading (Image 2). Read the four panels and keep the numbers:
2a Panel 1: exposure centre of mass; p1 / p99; blocked-shadow share (left end) and \
clipped-highlight share (right end); midtone crowding.
2b Panel 2: channel-offset direction and size = colour-cast direction and strength.
2c Panel 3: saturation level (median) and the over-saturated tail (p95, share above 60).
2d Panel 4: the leading hue shares.
Every histogram figure you cite later must come from this phase with its panel number.

PHASE 3 - write the five fields. Cross-check Phase 1 against Phase 2 before writing: a \
correction or an enhancement direction stands only when the picture judgement and the \
histogram reading agree on it (a defect the histogram cannot corroborate, or a reading the \
picture explains away as style, is not a direction). Defects go to correction_needs, style \
headroom goes to enhancement_opportunities, the assets go to preserve_intent and \
forbidden_directions; evidence backs each line with both sides of that cross-check. Every \
actionable item carries this fixed retrieval line, parsed verbatim downstream:
<axis> | <scope> | <observed state> | <move>
axis - exactly one of: exposure, band_lightness, contrast, hue, cast, saturation.
scope - global, shadows, midtones, highlights, or ONE region word (subject, skin, sky, \
foliage, background); region lines are served by the masked local pass.
observed state - verbatim from this closed vocabulary; the retrieval parses these words \
and nothing else:
  cast: "warm" / "yellow cast", "cool" / "blue cast", "magenta cast", "green cast"
  exposure / band_lightness: "underexposed" / "too dark", "overexposed" / "blown"
  contrast: "low contrast" / "hazy", "harsh" / "too contrasty"
  saturation: "flat" / "muted" / "dull", "oversaturated"
  hue: "hue drift toward <red|yellow|green|cyan|blue|rose>"
move - one of: lift, deepen, compress-highlights, raise-contrast, lower-contrast, \
raise-saturation, lower-saturation, neutralise, toward-amber, toward-teal, \
hue-toward-<red|yellow|green|cyan|blue|rose>. Keep state words out of the move slot, and \
never let one field carry both a state word and its opposite - bare-word parsing cancels \
them. No numbers in these lines; every number lives in evidence.

correction_needs - the fixed-form lines for states that would be errors if left unfixed. \
Cover every defect the two phases agree on - in particular, a meaningful clipped-highlight \
or blocked-shadow share in Panel 1 that the picture confirms as lost detail IS a correction, \
not a style. If there really is nothing, leave it empty - inventing a defect makes the \
downstream stages edit against a problem that does not exist.
preserve_intent - prohibitions protecting the colour and tonal assets of Phase 1 (what \
colour, cast or tonal quality must survive the grade). LUT-scoped only.
enhancement_opportunities - 2 to 4 DIFFERENT enhancement directions ordered by payoff, for \
states that leave headroom rather than errors. Each direction is one coherent finished look \
a colourist would actually grade towards, built from 2 to 4 component moves - a real look \
is multi-dimensional, one axis alone is not a look. Item form: one short style brief, then \
" => ", then its component retrieval lines joined by " ; ", ALWAYS in this fixed dimension \
order (skip a dimension with nothing to do): 1. colour temperature (cast), 2. tone \
(exposure / band_lightness / contrast), 3. saturation, 4. stylisation (hue). E.g. \
"golden-hour garden portrait => cast | global | cool | toward-amber ; band_lightness | \
subject | too dark | lift ; saturation | foliage | muted | raise-saturation ; hue | foliage \
| hue drift toward yellow | hue-toward-green". Different means genuinely distinct looks \
between items - vary the mood, the leading dimension or the scope; never restate one look \
with the components reshuffled.
forbidden_directions - the concrete LUT moves that would damage THIS image, using the words \
"warm", "cool", "dark", "saturation" where they apply (e.g. "do not push warmer", "do not \
darken the shadows further", "do not lift saturation of skin"); you may quote histogram \
readings.
evidence - one entry per actionable line, carrying its numbers: the Phase-2 histogram \
readings (each citing its panel number) and the Phase-1 picture observation that back the \
line, in enough detail for the downstream global and local passes to calibrate strength. \
Left and right always mean the viewer's left and right in the picture as shown. Never state \
or infer the image resolution, the pixel dimensions or the file size.
intent_mode - errors dominate = correction_led, enhancement dominates = enhancement_led, \
comparable = mixed; go by what is actually in the picture.

Return only the strict JSON object."""


# --------------------------------------------------------------------- selection
def load_sources(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def select_sources(rows: Iterable[Mapping[str, Any]], limit: int) -> list[dict[str, Any]]:
    """First `limit` rows in ascending sha1(source_id) order."""
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: hashlib.sha1(str(row["source_id"]).encode("utf-8")).hexdigest(),
    )
    return ordered[:limit]


# --------------------------------------------------------------------- statistics
def _iter_chunks(array: np.ndarray, rows_per_chunk: int) -> Iterable[np.ndarray]:
    for start in range(0, array.shape[0], rows_per_chunk):
        yield array[start:start + rows_per_chunk]


def panel_stats(path: str | Path) -> dict[str, Any]:
    """Four-panel statistics of one image at its original resolution."""
    with Image.open(path) as handle:
        image = ImageOps.exif_transpose(handle).convert("RGB")
        rgb8 = np.asarray(image, dtype=np.uint8)

    height, width = rgb8.shape[:2]
    total = float(height * width)

    rgb_hist = np.zeros((3, RGB_BINS), dtype=np.float64)
    rgb_sum = np.zeros(3, dtype=np.float64)
    l_hist = np.zeros(L_BINS, dtype=np.float64)
    c_hist = np.zeros(C_BINS, dtype=np.float64)
    hue_counts = np.zeros(HUE_SECTORS, dtype=np.float64)
    clip_low = clip_high = 0.0
    l_sum = c_sum = 0.0
    l_hist_fine = np.zeros(1000, dtype=np.float64)  # percentile support on L*
    c_hist_fine = np.zeros(1200, dtype=np.float64)

    rows_per_chunk = max(1, 2_000_000 // max(1, width))
    for chunk8 in _iter_chunks(rgb8, rows_per_chunk):
        flat8 = chunk8.reshape(-1, 3)
        for channel in range(3):
            counts = np.bincount(
                (flat8[:, channel].astype(np.int64) * RGB_BINS) // 256,
                minlength=RGB_BINS,
            )
            rgb_hist[channel] += counts[:RGB_BINS]
        rgb_sum += flat8.sum(axis=0, dtype=np.float64)

        lab = rgb2lab(flat8.astype(np.float64).reshape(-1, 1, 3) / 255.0).reshape(-1, 3)
        lightness, a_star, b_star = lab[:, 0], lab[:, 1], lab[:, 2]
        chroma = np.hypot(a_star, b_star)

        l_sum += float(lightness.sum())
        c_sum += float(chroma.sum())
        clip_low += float((lightness < CLIP_LOW_L).sum())
        clip_high += float((lightness > CLIP_HIGH_L).sum())

        index = np.clip((lightness / 100.0 * L_BINS).astype(np.int64), 0, L_BINS - 1)
        l_hist += np.bincount(index, minlength=L_BINS).astype(np.float64)
        fine = np.clip((lightness * 10.0).astype(np.int64), 0, 999)
        l_hist_fine += np.bincount(fine, minlength=1000).astype(np.float64)

        c_index = np.clip((chroma / C_MAX * C_BINS).astype(np.int64), 0, C_BINS - 1)
        c_hist += np.bincount(c_index, minlength=C_BINS).astype(np.float64)
        c_fine = np.clip((chroma * 10.0).astype(np.int64), 0, 1199)
        c_hist_fine += np.bincount(c_fine, minlength=1200).astype(np.float64)

        chromatic = chroma >= HUE_CHROMA_MIN
        if bool(chromatic.any()):
            angle = np.degrees(
                np.arctan2(b_star[chromatic], a_star[chromatic])
            ) % 360.0
            sector = np.clip(
                (angle / (360.0 / HUE_SECTORS)).astype(np.int64), 0, HUE_SECTORS - 1
            )
            hue_counts += np.bincount(sector, minlength=HUE_SECTORS).astype(np.float64)

    def _percentile(hist: np.ndarray, scale: float, fraction: float) -> float:
        cumulative = np.cumsum(hist)
        target = cumulative[-1] * fraction
        return float(int(np.searchsorted(cumulative, target)) / scale)

    rgb_means = rgb_sum / total
    rgb_medians = [
        _percentile(rgb_hist[channel], RGB_BINS / 256.0, 0.5) for channel in range(3)
    ]
    return {
        "size": (int(width), int(height)),
        "pixels": int(total),
        "l_hist": (l_hist / total).tolist(),
        "l_mean": l_sum / total,
        "l_p1": _percentile(l_hist_fine, 10.0, 0.01),
        "l_p50": _percentile(l_hist_fine, 10.0, 0.50),
        "l_p99": _percentile(l_hist_fine, 10.0, 0.99),
        "clip_low": clip_low / total,
        "clip_high": clip_high / total,
        "rgb_hist": (rgb_hist / total).tolist(),
        "rgb_mean": rgb_means.tolist(),
        "rgb_median": rgb_medians,
        "c_hist": (c_hist / total).tolist(),
        "c_mean": c_sum / total,
        "c_p50": _percentile(c_hist_fine, 10.0, 0.50),
        "c_p95": _percentile(c_hist_fine, 10.0, 0.95),
        "c_over60": float(c_hist_fine[600:].sum() / total),
        "hue_shares": (hue_counts / total).tolist(),
        "chromatic_share": float(hue_counts.sum() / total),
    }


# --------------------------------------------------------------------- rendering
_FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")

# The board is drawn at `_SUPERSAMPLE` times its final size and LANCZOS-downsampled,
# so 3px curves and 10px digits stay clean at 640x690.
_SUPERSAMPLE = 2

# Pure, fully saturated channel colours; green is darkened only enough to stay
# readable on white.
RGB_CHANNEL_COLORS = ((255, 0, 0), (0, 200, 0), (0, 0, 255))
RGB_FILL_ALPHA = 46
L_BAR_COLOR = (95, 95, 95)
CHROMA_BAR_COLOR = (125, 95, 160)
CHROMA_MARK_COLOR = (35, 35, 35)
P1_COLOR = (205, 30, 30)
P99_COLOR = (30, 60, 205)
TICK_COLOR = (120, 120, 120)
PANEL_NUMBER_COLOR = (55, 55, 55)


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(_FONT_DIR / name), size)


@dataclass(frozen=True)
class _Panel:
    left: int
    top: int
    right: int
    bottom: int


def _axis(draw: ImageDraw.ImageDraw, box: _Panel, width: int = 1) -> None:
    draw.rectangle(
        [box.left, box.top, box.right, box.bottom], outline=(165, 165, 165), width=width
    )


def _points(box: _Panel, values: Sequence[float], peak: float) -> list[tuple[float, float]]:
    span_x = box.right - box.left
    span_y = box.bottom - box.top
    points = []
    for index, value in enumerate(values):
        x = box.left + span_x * index / max(1, len(values) - 1)
        y = box.bottom - span_y * min(1.0, float(value) / peak if peak > 0 else 0.0)
        points.append((x, y))
    return points


def _curve(
    draw: ImageDraw.ImageDraw, box: _Panel, values: Sequence[float], peak: float,
    color: tuple[int, int, int], width: int = 2,
) -> None:
    draw.line(_points(box, values, peak), fill=color, width=width, joint="curve")


def _bars(
    draw: ImageDraw.ImageDraw, box: _Panel, values: Sequence[float], peak: float,
    color: tuple[int, int, int],
) -> None:
    span_x = (box.right - box.left) / len(values)
    span_y = box.bottom - box.top
    for index, value in enumerate(values):
        height = span_y * min(1.0, float(value) / peak if peak > 0 else 0.0)
        x0 = box.left + span_x * index
        draw.rectangle(
            [x0, box.bottom - height, x0 + max(1.0, span_x - 1.0), box.bottom],
            fill=color,
        )


def _vline(
    draw: ImageDraw.ImageDraw, box: _Panel, x: float, color: tuple[int, int, int],
    width: int = 1, dash: int = 4,
) -> None:
    y = box.top
    while y < box.bottom:
        draw.line([(x, y), (x, min(y + dash, box.bottom))], fill=color, width=width)
        y += dash * 2


def _x_ticks(
    draw: ImageDraw.ImageDraw, box: _Panel, ticks: Sequence[int], span: float,
    font: ImageFont.FreeTypeFont, pad: int,
) -> None:
    for tick in ticks:
        x = box.left + (box.right - box.left) * tick / span
        draw.text((x, box.bottom + pad), str(tick), TICK_COLOR, font, anchor="ma")


def render_board(stats: Mapping[str, Any]) -> Image.Image:
    """Four-panel board carrying numbers only - no titles, legends or prose.

    What is printed: the panel number (1-4), the L* p1 / p99 dashed lines with their
    values and the two clipped shares at the ends of panel 1, the R/G/B channel means
    in the channel colours, the C*ab median / p95 dashed lines with their values, the
    hue-sector shares, and the x-axis ticks. Everything else is conveyed by geometry
    and colour; the panel semantics live in the prompt.
    """
    s = _SUPERSAMPLE
    size = (BOARD_SIZE[0] * s, BOARD_SIZE[1] * s)
    board = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(board)
    num_font = _font("DejaVuSans-Bold.ttf", 24 * s)
    val_font = _font("DejaVuSans-Bold.ttf", 12 * s)
    tick_font = _font("DejaVuSans.ttf", 10 * s)

    left, right = 30 * s, 625 * s
    plot_h = 134 * s
    tick_pad = 3 * s
    panel_tops = tuple(value * s for value in (12, 170, 328, 486))

    def _panel_number(index: int, top: int) -> None:
        draw.text((3 * s, top - 4 * s), str(index + 1), PANEL_NUMBER_COLOR, num_font)

    # ---- Panel 1: L* lightness -------------------------------------------------
    box = _Panel(left, panel_tops[0], right, panel_tops[0] + plot_h)
    _panel_number(0, box.top)
    _axis(draw, box, width=s)
    l_hist = stats["l_hist"]
    _bars(draw, box, l_hist, max(l_hist), L_BAR_COLOR)
    draw.text((box.left + 4 * s, box.top + 3 * s),
              f"{stats['clip_low'] * 100:.2f}%", P1_COLOR, val_font)
    draw.text((box.right - 4 * s, box.top + 3 * s),
              f"{stats['clip_high'] * 100:.2f}%", P99_COLOR, val_font, anchor="ra")
    for value, color in ((stats["l_p1"], P1_COLOR), (stats["l_p99"], P99_COLOR)):
        x = box.left + (box.right - box.left) * min(max(value, 0.0), 100.0) / 100.0
        _vline(draw, box, x, color, width=s, dash=4 * s)
        anchor = "ra" if value > 60.0 else "la"
        draw.text((x + (-4 * s if anchor == "ra" else 4 * s), box.top + 38 * s),
                  f"{value:.1f}", color, val_font, anchor=anchor)
    _x_ticks(draw, box, (0, 20, 40, 60, 80, 100), 100.0, tick_font, tick_pad)

    # ---- Panel 2: R/G/B overlay ------------------------------------------------
    box = _Panel(left, panel_tops[1], right, panel_tops[1] + plot_h)
    _panel_number(1, box.top)
    _axis(draw, box, width=s)
    rgb_hist = stats["rgb_hist"]
    peak = max(max(channel) for channel in rgb_hist)
    fills = Image.new("RGBA", size, (0, 0, 0, 0))
    fill_draw = ImageDraw.Draw(fills)
    for channel, color in enumerate(RGB_CHANNEL_COLORS):
        polygon = _points(box, rgb_hist[channel], peak)
        polygon = [(box.left, box.bottom), *polygon, (box.right, box.bottom)]
        fill_draw.polygon(polygon, fill=(*color, RGB_FILL_ALPHA))
    board.paste(Image.alpha_composite(board.convert("RGBA"), fills).convert("RGB"))
    for channel, color in enumerate(RGB_CHANNEL_COLORS):
        _curve(draw, box, rgb_hist[channel], peak, color, width=2 * s)
    _x_ticks(draw, box, (0, 64, 128, 192, 255), 255.0, tick_font, tick_pad)
    means = stats["rgb_mean"]
    x = box.left + 4 * s
    for value, color in zip(means, RGB_CHANNEL_COLORS):
        label = f"{value:.1f}"
        draw.text((x, box.top + 3 * s), label, color, val_font)
        x += draw.textlength(label, val_font) + 14 * s

    # ---- Panel 3: C*ab chroma --------------------------------------------------
    box = _Panel(left, panel_tops[2], right, panel_tops[2] + plot_h)
    _panel_number(2, box.top)
    _axis(draw, box, width=s)
    c_hist = stats["c_hist"]
    _bars(draw, box, c_hist, max(c_hist), CHROMA_BAR_COLOR)
    for value, row in ((stats["c_p50"], 3), (stats["c_p95"], 21)):
        x = box.left + (box.right - box.left) * min(max(value, 0.0), C_MAX) / C_MAX
        _vline(draw, box, x, CHROMA_MARK_COLOR, width=s, dash=4 * s)
        anchor = "ra" if value > 0.6 * C_MAX else "la"
        draw.text((x + (-4 * s if anchor == "ra" else 4 * s), box.top + row * s),
                  f"{value:.1f}", CHROMA_MARK_COLOR, val_font, anchor=anchor)
    _x_ticks(draw, box, (0, 20, 40, 60, 80, 100, 120), C_MAX, tick_font, tick_pad)

    # ---- Panel 4: hue shares ---------------------------------------------------
    top = panel_tops[3]
    _panel_number(3, top)
    bar_left, bar_right = 30 * s, 560 * s
    shares = stats["hue_shares"]
    peak = max(max(shares), 1e-6)
    for index, share in enumerate(shares):
        y0 = top + index * 18 * s
        width = (bar_right - bar_left) * share / peak
        draw.rectangle([bar_left, y0, bar_left + max(2.0 * s, width), y0 + 13 * s],
                       fill=HUE_SECTOR_COLORS[index])
        draw.text((bar_left + width + 6 * s, y0 + 1 * s), f"{share * 100:.1f}%",
                  (40, 40, 40), val_font)

    return board.resize(BOARD_SIZE, Image.Resampling.LANCZOS)


def preview_jpeg(path: str | Path) -> bytes:
    with Image.open(path) as handle:
        image = ImageOps.exif_transpose(handle).convert("RGB")
        if max(image.size) > PREVIEW_LONGEST_EDGE:
            scale = PREVIEW_LONGEST_EDGE / max(image.size)
            size = tuple(max(1, round(value * scale)) for value in image.size)
            image = image.resize(size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=PREVIEW_QUALITY, optimize=False,
                   progressive=False, subsampling=0)
    return buffer.getvalue()


# --------------------------------------------------------------------- transport
def endpoint_credentials(config_path: Path, endpoint_id: str) -> tuple[str, str]:
    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    for entry in payload.get("annotation", {}).get("external_endpoints", []):
        if str(entry.get("id")) == endpoint_id:
            return str(entry["base_url"]), str(entry["api_key"])
    raise SystemExit(f"endpoint not found in {config_path}: {endpoint_id}")


def _data_url(payload: bytes, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(payload).decode("ascii")


def _consume(stream: Any) -> dict[str, Any]:
    from openai.types.responses import (
        ResponseCompletedEvent, ResponseErrorEvent, ResponseFailedEvent,
        ResponseTextDeltaEvent,
    )

    chunks: list[str] = []
    completed: Any = None
    for event in stream:
        if isinstance(event, ResponseTextDeltaEvent):
            chunks.append(event.delta)
        elif isinstance(event, ResponseCompletedEvent):
            completed = event.response
        elif isinstance(event, ResponseFailedEvent):
            error = getattr(event.response, "error", None)
            raise RuntimeError(str(getattr(error, "code", None) or "response_failed"))
        elif isinstance(event, ResponseErrorEvent):
            raise RuntimeError(str(event.code or "response_error"))
    if completed is None:
        raise RuntimeError("responses_stream_interrupted")
    try:
        fallback = str(getattr(completed, "output_text", "") or "")
    except (TypeError, ValueError):
        fallback = ""
    # Relay injects U+200B at the head of the first delta; output_text is clean.
    text = (fallback or "".join(chunks)).lstrip("\u200b\ufeff")
    if not text:
        raise RuntimeError("responses_output_empty")
    usage = getattr(completed, "usage", None)
    details = getattr(usage, "input_tokens_details", None)
    return {
        "text": text,
        "model": str(getattr(completed, "model", "") or ""),
        "usage": {
            "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
            "cached_tokens": int(getattr(details, "cached_tokens", 0) or 0),
        },
    }


def schema_error(parsed: Any) -> str | None:
    if not isinstance(parsed, Mapping):
        return "not_an_object"
    missing = sorted(set(DIAGNOSIS_SCHEMA["required"]) - set(parsed))
    if missing:
        return f"missing_keys:{','.join(missing)}"
    for name in FIVE_FIELDS:
        value = parsed.get(name)
        if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
            return f"{name}_not_string_list"
    if parsed.get("intent_mode") not in ("correction_led", "enhancement_led", "mixed"):
        return "bad_intent_mode"
    return None


def diagnose_v2(
    client: Any, source_jpeg: bytes, board_png: bytes, *, attempts: int = 3,
) -> dict[str, Any]:
    payload = {
        "model": MODEL,
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": V2_RULES},
            {"type": "input_image", "image_url": _data_url(source_jpeg, "image/jpeg"),
             "detail": "low"},
            {"type": "input_image", "image_url": _data_url(board_png, "image/png"),
             "detail": "high"},
        ]}],
        "stream": True,
        "temperature": 0.1,
        "max_output_tokens": 4096,
        "store": False,
        "reasoning": {"effort": REASONING_EFFORT},
        "text": {"format": {
            "type": "json_schema", "name": "source_diagnosis_v1", "strict": True,
            "schema": DIAGNOSIS_SCHEMA,
        }},
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
        except Exception as exc:  # transport / relay errors
            last_error = f"{type(exc).__name__}:{str(exc)[:160]}"
        if attempt < attempts:
            print(f"    retry {attempt + 1}/{attempts} after {last_error}", flush=True)
            time.sleep(min(60.0, 2 ** (attempt - 1) + random.random() * 0.25))
    raise RuntimeError(last_error)


# --------------------------------------------------------------------- report
def _bullets(values: Any) -> str:
    if not isinstance(values, list) or not values:
        return "_(empty)_"
    return "\n".join(f"- {str(item)}" for item in values)


_PANEL_PATTERN = re.compile(r"panel\s*[1-4]", re.IGNORECASE)


def histogram_citations(diagnosis: Mapping[str, Any]) -> int:
    """Number of string entries across the five fields that cite a panel number."""
    count = 0
    for name in FIVE_FIELDS:
        for item in diagnosis.get(name) or []:
            if _PANEL_PATTERN.search(str(item)):
                count += 1
    return count


def _relative(path: Path, base: Path) -> str:
    return os.path.relpath(path, base)


def write_report(records: Sequence[Mapping[str, Any]], out_dir: Path) -> Path:
    ok = [row for row in records if row.get("v2") is not None]
    lines: list[str] = []
    lines.append("# E1 - diagnosis prompt v2 (histogram board) vs frozen v1")
    lines.append("")
    lines.append(f"- model: `{MODEL}`, `reasoning_effort={REASONING_EFFORT}`, "
                 f"strict JSON, schema identical to the frozen v1 batch")
    lines.append(f"- v2 prompt tag: `{PROMPT_TAG}` (test only; no revision was changed)")
    lines.append("- v2 input per source: 512px source preview (detail=low) + "
                 f"{BOARD_SIZE[0]}x{BOARD_SIZE[1]} histogram board (detail=high) + "
                 "v2 rules")
    lines.append("- v1 side is read verbatim from each source's frozen "
                 "`source_annotation_path` JSON")
    lines.append(f"- sources: first {len(records)} rows of "
                 "`sources5k.annotated.jsonl` in ascending `sha1(source_id)` order")
    lines.append("")
    lines.append("Harness decisions (not pre-registered by the task card):")
    lines.append("")
    lines.append(f"- the board is sent at `detail=high` and at its native "
                 f"{BOARD_SIZE[0]}x{BOARD_SIZE[1]} so the printed numbers stay legible; "
                 "the source preview keeps the frozen 512px / `detail=low` encoding of "
                 "the v1 batch")
    lines.append("- board bin counts (L*: 64, R/G/B: 64, C*ab: 48 over 0-120, hue: 6) are "
                 "display bins of this harness only; the frozen 8-bin "
                 "`source_histogram` contract is untouched")
    lines.append("- all four panels are computed on the original-resolution file "
                 "(no downscale, no pixel subsampling)")
    lines.append("- v1 token usage is the frozen `provenance.usage`; v1 saw one image and "
                 "the v1 rules, v2 sees two images and the longer v2 rules")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    if ok:
        v2_in = sum(row["v2"]["usage"]["input_tokens"] for row in ok) / len(ok)
        v2_out = sum(row["v2"]["usage"]["output_tokens"] for row in ok) / len(ok)
        v1_in = sum(row["v1_usage"].get("input_tokens", 0) for row in ok) / len(ok)
        v1_out = sum(row["v1_usage"].get("output_tokens", 0) for row in ok) / len(ok)
        cites = sum(row["v2_citations"] for row in ok)
        items = sum(
            len(row["v2"]["parsed"].get(name) or []) for row in ok for name in FIVE_FIELDS
        )
    else:
        v2_in = v2_out = v1_in = v1_out = 0.0
        cites = items = 0
    lines.append(f"- succeeded: {len(ok)}/{len(records)}")
    lines.append("")
    lines.append("| metric | v1 (frozen) | v2 (this run) |")
    lines.append("| --- | --- | --- |")
    lines.append(f"| mean input tokens | {v1_in:.1f} | {v2_in:.1f} |")
    lines.append(f"| mean output tokens | {v1_out:.1f} | {v2_out:.1f} |")
    for mode in ("correction_led", "enhancement_led", "mixed"):
        v1_count = sum(
            1 for row in ok if row["v1"].get("intent_mode") == mode
        )
        v2_count = sum(
            1 for row in ok if row["v2"]["parsed"].get("intent_mode") == mode
        )
        lines.append(f"| intent_mode = {mode} | {v1_count} | {v2_count} |")
    v1_cites = sum(histogram_citations(row["v1"]) for row in ok)
    lines.append(f"| items citing a panel number | {v1_cites} | {cites} |")
    lines.append(f"| total items across the five fields | "
                 f"{sum(len(row['v1'].get(n) or []) for row in ok for n in FIVE_FIELDS)} "
                 f"| {items} |")
    lines.append("")

    lines.append("| # | source_id | scene | v2 in/out tokens | v1 in/out tokens | "
                 "v2 panel-citing items | v2 intent_mode | v1 intent_mode |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for index, row in enumerate(records, start=1):
        if row.get("v2") is None:
            lines.append(
                f"| {index} | `{row['source_id']}` | {row['scene']} | FAILED "
                f"({row.get('error', '')}) | - | - | - | - |"
            )
            continue
        usage = row["v2"]["usage"]
        v1_usage = row["v1_usage"]
        lines.append(
            f"| {index} | `{row['source_id']}` | {row['scene']} | "
            f"{usage['input_tokens']} / {usage['output_tokens']} | "
            f"{v1_usage.get('input_tokens', 0)} / {v1_usage.get('output_tokens', 0)} | "
            f"{row['v2_citations']} | {row['v2']['parsed'].get('intent_mode')} | "
            f"{row['v1'].get('intent_mode')} |"
        )
    lines.append("")

    for index, row in enumerate(records, start=1):
        lines.append(f"## {index}. `{row['source_id']}` ({row['scene']})")
        lines.append("")
        lines.append(f"<img src=\"{row['preview_rel']}\" width=\"340\"> "
                     f"<img src=\"{row['board_rel']}\" width=\"400\">")
        lines.append("")
        lines.append(f"Subject: {row.get('subject_description') or '-'}")
        lines.append("")
        if row.get("v2") is None:
            lines.append(f"**v2 FAILED**: `{row.get('error', '')}`")
            lines.append("")
        else:
            usage = row["v2"]["usage"]
            v1_usage = row["v1_usage"]
            lines.append(
                f"token usage - v2: in {usage['input_tokens']}, out "
                f"{usage['output_tokens']} | v1: in {v1_usage.get('input_tokens', 0)}, "
                f"out {v1_usage.get('output_tokens', 0)}"
            )
            lines.append("")
            lines.append(
                f"intent_mode - v2: `{row['v2']['parsed'].get('intent_mode')}` | "
                f"v1: `{row['v1'].get('intent_mode')}`"
            )
            lines.append("")
        for name in FIVE_FIELDS:
            lines.append(f"### {name}")
            lines.append("")
            lines.append("**v2**")
            lines.append("")
            lines.append(
                _bullets((row["v2"]["parsed"].get(name)) if row.get("v2") else None)
            )
            lines.append("")
            lines.append("**v1 (frozen)**")
            lines.append("")
            lines.append(_bullets(row["v1"].get(name)))
            lines.append("")

    target = out_dir / "report.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


# --------------------------------------------------------------------- driver
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--databuild-config", type=Path, default=DEFAULT_DATABUILD)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--boards-only", action="store_true",
                        help="render the boards and skip every API call")
    parser.add_argument("--report-only", action="store_true",
                        help="rebuild report.md from an existing records.jsonl")
    args = parser.parse_args(argv)

    out_dir = args.out
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        records = [
            json.loads(line)
            for line in (out_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        print(f"report: {write_report(records, out_dir)}", flush=True)
        return 0

    selected = select_sources(load_sources(args.sources), args.limit)
    print(f"selected {len(selected)} sources", flush=True)

    client = None
    if not args.boards_only:
        from openai import OpenAI

        base_url, api_key = endpoint_credentials(args.databuild_config, args.endpoint)
        client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0,
                        timeout=args.timeout,
                        default_headers={"X-vgate-class": "agent-loop"})

    def build(index: int, source: Mapping[str, Any]) -> dict[str, Any]:
        source_id = str(source["source_id"])
        print(f"[{index}/{len(selected)}] {source_id}", flush=True)
        stats = panel_stats(source["source_path"])
        board = render_board(stats)
        board_path = image_dir / f"{source_id}.board.png"
        board.save(board_path, "PNG", optimize=True)
        preview = preview_jpeg(source["source_path"])
        preview_path = image_dir / f"{source_id}.source.jpg"
        preview_path.write_bytes(preview)

        annotation = json.loads(
            Path(str(source["source_annotation_path"])).read_text(encoding="utf-8")
        )
        record: dict[str, Any] = {
            "source_id": source_id,
            "scene": str(source.get("scene") or "unknown"),
            "subject_description": (source.get("subject") or {}).get("description"),
            "source_path": str(source["source_path"]),
            "preview_rel": _relative(preview_path, out_dir),
            "board_rel": _relative(board_path, out_dir),
            "stats": {key: value for key, value in stats.items()
                      if key not in ("l_hist", "rgb_hist", "c_hist")},
            "v1": dict(annotation.get("diagnosis") or {}),
            "v1_usage": dict((annotation.get("provenance") or {}).get("usage") or {}),
            "v2": None,
        }
        if client is not None:
            board_bytes = board_path.read_bytes()
            try:
                record["v2"] = diagnose_v2(client, preview, board_bytes,
                                           attempts=args.attempts)
            except Exception as exc:
                record["error"] = str(exc)[:300]
                print(f"    FAILED: {record['error']}", flush=True)
            else:
                record["v2_citations"] = histogram_citations(record["v2"]["parsed"])
                print(f"    [{index}] ok in={record['v2']['usage']['input_tokens']} "
                      f"out={record['v2']['usage']['output_tokens']} "
                      f"cites={record['v2_citations']}", flush=True)
        return record

    if args.concurrency > 1 and client is not None:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            records = list(pool.map(
                lambda pair: build(pair[0], pair[1]),
                list(enumerate(selected, start=1)),
            ))
    else:
        records = [build(index, source)
                   for index, source in enumerate(selected, start=1)]

    (out_dir / "records.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in records)
        + "\n", encoding="utf-8",
    )
    (out_dir / "prompt_v2.txt").write_text(V2_RULES + "\n", encoding="utf-8")
    report = write_report(records, out_dir)
    succeeded = sum(1 for row in records if row.get("v2") is not None)
    print(f"report: {report}  ({succeeded}/{len(records)} succeeded)", flush=True)
    return 0 if succeeded >= max(1, int(0.8 * len(records))) else 1


if __name__ == "__main__":
    sys.exit(main())
