"""E4: the four-panel histogram board sent alongside the source photograph.

This is the production port of the E1/E2 harness board (``boardv3``): the same
statistics on the same original-resolution pixels, rendered by the same PIL-only
code into the same 640x604 PNG. It is a leaf module - it imports nothing from
``dataset_build.agent_loop`` - so the diagnosis prompt, the annotate driver and any
offline tool can all render the identical board.

Contract:

* :func:`panel_stats` reads the file at its original resolution (no downscale, no
  pixel subsampling) and returns the four panels' numbers.
* :func:`render_board` draws them; nothing random, no time or locale input, no
  matplotlib. The same statistics therefore always produce the same pixels.
* :func:`board_png` is the one-call path used in production: file path in,
  PNG bytes out. Two calls on one file return byte-identical payloads.

The board's display bins (``L_BINS`` / ``RGB_BINS`` / ``C_BINS`` / ``HUE_SECTORS``)
are deliberately finer than the frozen 8-bin ``source_histogram`` contract, which
this module does not touch and does not read.

The board carries printed digits, so the transport must send it at its native size
without re-encoding; that is what the ``passthrough`` image encoding in ``prompts``
is for.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from skimage.color import rgb2lab

# Bump on any change to the statistics or to the drawing: the board bytes are part
# of what the diagnosis model is shown, so the revision travels with the prompt
# registry and lands in every annotation's provenance.
BOARD_REVISION = "histogram-board-v3-640x604"

BOARD_SIZE: tuple[int, int] = (640, 604)

# Panel binning. These are display bins of this board only.
L_BINS = 64
RGB_BINS = 64
C_BINS = 48
C_MAX = 120.0
HUE_SECTORS = 6
HUE_CHROMA_MIN = 10.0
CLIP_LOW_L = 2.0
CLIP_HIGH_L = 98.0

# Everything that decides the numbers on the board, in one registry-shaped dict.
BOARD_BIN_GEOMETRY: dict[str, Any] = {
    "l_bins": L_BINS,
    "rgb_bins": RGB_BINS,
    "c_bins": C_BINS,
    "c_max": C_MAX,
    "hue_sectors": HUE_SECTORS,
    "hue_chroma_min": HUE_CHROMA_MIN,
    "clip_low_l": CLIP_LOW_L,
    "clip_high_l": CLIP_HIGH_L,
}

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


def _open_image(path: str | Path) -> Image.Image:
    """Same reader discipline as ``artifacts``: plain file first, archive second."""
    try:
        from dataset_build.tools.archive_reader import open_image

        return open_image(path)
    except (ImportError, FileNotFoundError, ValueError):
        return Image.open(path)


# --------------------------------------------------------------------- statistics
def _iter_chunks(array: np.ndarray, rows_per_chunk: int) -> Iterable[np.ndarray]:
    for start in range(0, array.shape[0], rows_per_chunk):
        yield array[start:start + rows_per_chunk]


def panel_stats(path: str | Path) -> dict[str, Any]:
    """Four-panel statistics of one image at its original resolution."""
    with _open_image(path) as handle:
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
# so 3px curves and 10px digits stay clean at the final size.
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


def board_png(path: str | Path) -> bytes:
    """Render one image's board and return its PNG bytes (production entry point)."""
    board = render_board(panel_stats(path))
    output = io.BytesIO()
    board.save(output, "PNG", optimize=True)
    return output.getvalue()


__all__ = [
    "BOARD_BIN_GEOMETRY", "BOARD_REVISION", "BOARD_SIZE", "board_png", "panel_stats",
    "render_board",
]
