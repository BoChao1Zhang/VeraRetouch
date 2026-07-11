"""Build review contact sheets for subject-aware local-mask feathering.

This is an evaluation-only tool. It reads VLM ``main_subject`` annotations and
their precomputed SAM3 masks, then visualizes eight candidate masks per source:

    1 semantic + 1 radial + 3 band + 3 linear

Each tile shows the alpha over the source, the alpha itself, and a strong local
exposure preview. The candidate set deliberately spans strict inside-only and
bounded outside feather profiles. It does not import or change the production
construct policy.

Example:

    python -m dataset_build.tools.evaluate_mask_feather \
        --out-dir _mask_review/feather_eval_v1
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dataset_build.source_qa import config as qa_config


DEFAULT_CONCEPTS = ("woman", "man", "dog", "cat", "red car", "perfume bottle")
DEFAULT_OUT_DIR = "_mask_review/feather_eval_v1"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


@dataclass(frozen=True)
class SourceRecord:
    asset_id: str
    source_path: str
    concept: str
    mask_path: str
    db_area: float
    db_bbox: Any


@dataclass(frozen=True)
class FeatherProfile:
    name: str
    feather_in: float
    feather_out: float


@dataclass
class Candidate:
    index: int
    kind: str
    variant: str
    hard_mask: np.ndarray
    feather: FeatherProfile
    geometry: dict[str, Any]


FEATHER_STRICT_NARROW = FeatherProfile("strict-narrow", 0.008, 0.0)
FEATHER_STRICT_MEDIUM = FeatherProfile("strict-medium", 0.015, 0.0)
FEATHER_STRICT_WIDE = FeatherProfile("strict-wide", 0.030, 0.0)
FEATHER_BOUNDED_NARROW = FeatherProfile("bounded-narrow", 0.008, 0.0025)
FEATHER_BOUNDED_MEDIUM = FeatherProfile("bounded-medium", 0.015, 0.005)
FEATHER_BOUNDED_WIDE = FeatherProfile("bounded-wide", 0.030, 0.010)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    path = FONT_PATH.replace(".ttf", "-Bold.ttf") if bold else FONT_PATH
    try:
        return ImageFont.truetype(path, size=size)
    except OSError:
        return ImageFont.load_default()


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def feather_binary(
    hard_mask: np.ndarray,
    feather_in: float,
    feather_out: float,
) -> np.ndarray:
    """Feather a binary support using fractions of the image short edge.

    ``feather_in`` consumes transition width inside the hard support.
    ``feather_out`` allows only a bounded transition outside it. An out width
    of zero guarantees exact alpha zero at every exterior pixel.
    """
    hard = np.asarray(hard_mask, dtype=bool)
    if hard.ndim != 2 or not hard.any():
        raise ValueError("hard_mask must be a non-empty HxW mask")
    if feather_in < 0.0 or feather_out < 0.0:
        raise ValueError("feather widths must be non-negative")
    if feather_in == 0.0 and feather_out == 0.0:
        return hard.astype(np.float32)

    short = float(min(hard.shape))
    in_px = feather_in * short
    out_px = feather_out * short
    dist_in = cv2.distanceTransform(hard.astype(np.uint8), cv2.DIST_L2, 5)
    dist_out = cv2.distanceTransform((~hard).astype(np.uint8), cv2.DIST_L2, 5)

    # Pixel centers next to the support boundary lie roughly half a pixel from
    # it. This signed distance makes the transition symmetric when widths match.
    signed = np.where(
        hard,
        np.maximum(dist_in - 0.5, 0.0),
        -np.maximum(dist_out - 0.5, 0.0),
    )
    denom = max(in_px + out_px, 1e-6)
    alpha = _smoothstep((signed + out_px) / denom).astype(np.float32)

    if out_px == 0.0:
        alpha[~hard] = 0.0
    else:
        alpha[(~hard) & (dist_out >= out_px + 0.5)] = 0.0
    if in_px == 0.0:
        alpha[hard] = 1.0
    return np.clip(alpha, 0.0, 1.0)


def _significant_component_count(mask: np.ndarray) -> int:
    binary = np.asarray(mask, dtype=np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return 0
    total = int(binary.sum())
    floor = max(48, int(total * 0.03))
    return sum(int(stats[i, cv2.CC_STAT_AREA]) >= floor for i in range(1, count))


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask > 0.5, dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary.astype(bool)
    total = int(binary.sum())
    floor = max(48, int(total * 0.01))
    keep = [i for i in range(1, count) if int(stats[i, cv2.CC_STAT_AREA]) >= floor]
    if not keep:
        keep = [1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))]
    return np.isin(labels, keep)


def _load_preview(source_path: str, long_edge: int) -> np.ndarray:
    with Image.open(source_path) as im:
        image = im.convert("RGB")
        scale = min(1.0, long_edge / max(image.size))
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(size, Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.uint8)


def _load_subject_mask(mask_path: str, shape: tuple[int, int]) -> np.ndarray:
    with Image.open(mask_path) as im:
        mask = im.convert("L").resize((shape[1], shape[0]), Image.Resampling.NEAREST)
        return _clean_mask(np.asarray(mask, dtype=np.float32) / 255.0)


def _subject_geometry(mask: np.ndarray) -> dict[str, Any]:
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if xs.size < 32:
        raise ValueError("subject mask has too few pixels")
    points = np.stack([(xs + 0.5) / w, (ys + 0.5) / h], axis=1)
    center = points.mean(axis=0)
    cov = np.cov((points - center).T)
    _, vectors = np.linalg.eigh(cov)
    major = vectors[:, 1]
    if major[0] < 0:
        major = -major
    minor = np.array([-major[1], major[0]])
    major_extent = float(np.quantile(np.abs((points - center) @ major), 0.99))
    minor_extent = float(np.quantile(np.abs((points - center) @ minor), 0.99))
    angle = math.degrees(math.atan2(float(major[1]), float(major[0])))
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    return {
        "points": points,
        "center": center,
        "major": major,
        "minor": minor,
        "major_extent": major_extent,
        "minor_extent": minor_extent,
        "angle_deg": angle,
        "bbox": [float(x0), float(y0), float(x1), float(y1)],
    }


def _normalized_grid(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return (xx + 0.5) / w, (yy + 0.5) / h


def _ellipse_mask(
    shape: tuple[int, int],
    center: Sequence[float],
    angle_deg: float,
    radius_major: float,
    radius_minor: float,
) -> np.ndarray:
    xx, yy = _normalized_grid(shape)
    theta = math.radians(angle_deg)
    dx, dy = xx - center[0], yy - center[1]
    along = dx * math.cos(theta) + dy * math.sin(theta)
    across = -dx * math.sin(theta) + dy * math.cos(theta)
    return (along / max(radius_major, 1e-4)) ** 2 + (
        across / max(radius_minor, 1e-4)
    ) ** 2 <= 1.0


def _band_mask(
    shape: tuple[int, int],
    center: Sequence[float],
    angle_deg: float,
    half_width: float,
) -> np.ndarray:
    xx, yy = _normalized_grid(shape)
    theta = math.radians(angle_deg)
    normal_x, normal_y = -math.sin(theta), math.cos(theta)
    distance = (xx - center[0]) * normal_x + (yy - center[1]) * normal_y
    return np.abs(distance) <= half_width


def _linear_mask(
    shape: tuple[int, int], normal: Sequence[float], boundary: float
) -> np.ndarray:
    xx, yy = _normalized_grid(shape)
    return xx * normal[0] + yy * normal[1] >= boundary


def _rotated(vector: np.ndarray, degrees: float) -> np.ndarray:
    theta = math.radians(degrees)
    rotation = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=np.float64,
    )
    return rotation @ vector


def build_candidates(subject_mask: np.ndarray) -> list[Candidate]:
    """Return the fixed 1/1/3/3 review allocation for one subject mask."""
    sg = _subject_geometry(subject_mask)
    center = sg["center"]
    points = sg["points"]
    shape = subject_mask.shape

    radial_margin = 1.22
    radial_major = max(float(sg["major_extent"]) * radial_margin, 0.16)
    radial_minor = max(float(sg["minor_extent"]) * radial_margin, 0.11)
    radial = _ellipse_mask(
        shape, center, float(sg["angle_deg"]), radial_major, radial_minor
    )

    candidates = [
        Candidate(
            1,
            "semantic",
            "vlm-main-subject",
            subject_mask,
            FEATHER_STRICT_MEDIUM,
            {"source": "source_captions.main_subject + sam3_masks"},
        ),
        Candidate(
            2,
            "radial",
            "pca-ellipse",
            radial,
            FEATHER_BOUNDED_MEDIUM,
            {
                "center": center.tolist(),
                "angle_deg": float(sg["angle_deg"]),
                "radius_major": radial_major,
                "radius_minor": radial_minor,
                "margin": radial_margin,
            },
        ),
    ]

    band_specs = (
        (0.0, 1.05, FEATHER_STRICT_NARROW, "subject-axis-tight"),
        (14.0, 1.10, FEATHER_BOUNDED_MEDIUM, "rotated-medium"),
        (-20.0, 1.22, FEATHER_BOUNDED_WIDE, "rotated-wide"),
    )
    for offset, width_scale, feather, name in band_specs:
        angle = float(sg["angle_deg"]) + offset
        theta = math.radians(angle)
        normal = np.array([-math.sin(theta), math.cos(theta)])
        subject_half_width = float(np.quantile(np.abs((points - center) @ normal), 0.995))
        half_width = max(subject_half_width * width_scale, 0.055)
        hard = _band_mask(shape, center, angle, half_width)
        candidates.append(
            Candidate(
                len(candidates) + 1,
                "band",
                name,
                hard,
                feather,
                {
                    "center": center.tolist(),
                    "angle_deg": angle,
                    "angle_offset_deg": offset,
                    "half_width": half_width,
                    "coverage_scale": width_scale,
                },
            )
        )

    x0, y0, x1, y1 = sg["bbox"]
    sides = [
        (x0, "left", np.array([1.0, 0.0])),
        (1.0 - x1, "right", np.array([-1.0, 0.0])),
        (y0, "top", np.array([0.0, 1.0])),
        (1.0 - y1, "bottom", np.array([0.0, -1.0])),
    ]
    sides.sort(key=lambda item: item[0], reverse=True)
    linear_specs = (
        (0.04, -14.0, FEATHER_BOUNDED_NARROW, "close-subject"),
        (0.16, 0.0, FEATHER_STRICT_MEDIUM, "mid-coverage"),
        (0.32, 14.0, FEATHER_STRICT_WIDE, "wide-coverage"),
    )
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    room, side, base_normal = sides[0]
    for gap, tilt, feather, name in linear_specs:
        normal = _rotated(base_normal, tilt)
        normal /= np.linalg.norm(normal)
        subject_min = float(np.min(points @ normal))
        frame_min = float(np.min(corners @ normal))
        boundary = subject_min - gap * max(subject_min - frame_min, 0.01)
        hard = _linear_mask(shape, normal, boundary)
        candidates.append(
            Candidate(
                len(candidates) + 1,
                "linear",
                f"{name}-{side}",
                hard,
                feather,
                {
                    "empty_side": side,
                    "axis_room": float(room),
                    "normal": normal.tolist(),
                    "tilt_deg": tilt,
                    "boundary": boundary,
                    "gap_fraction": gap,
                },
            )
        )

    counts = {kind: sum(c.kind == kind for c in candidates) for kind in {c.kind for c in candidates}}
    expected = {"semantic": 1, "radial": 1, "band": 3, "linear": 3}
    if counts != expected:
        raise AssertionError(f"unexpected candidate allocation: {counts}")
    return candidates


def _srgb_to_linear(image: np.ndarray) -> np.ndarray:
    return np.where(
        image <= 0.04045,
        image / 12.92,
        ((image + 0.055) / 1.055) ** 2.4,
    )


def _linear_to_srgb(image: np.ndarray) -> np.ndarray:
    return np.where(
        image <= 0.0031308,
        image * 12.92,
        1.055 * np.maximum(image, 0.0) ** (1.0 / 2.4) - 0.055,
    )


def _exposure_preview(image: np.ndarray, alpha: np.ndarray, ev: float) -> np.ndarray:
    source = image.astype(np.float32) / 255.0
    edited = _linear_to_srgb(np.clip(_srgb_to_linear(source) * (2.0**ev), 0.0, 1.0))
    composite = source * (1.0 - alpha[..., None]) + edited * alpha[..., None]
    return np.clip(np.round(composite * 255.0), 0, 255).astype(np.uint8)


def _draw_contour(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], width: int) -> None:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(image, contours, -1, color, width, lineType=cv2.LINE_AA)


def _alpha_overlay(image: np.ndarray, alpha: np.ndarray, hard: np.ndarray) -> np.ndarray:
    out = image.astype(np.float32)
    heat = np.empty_like(out)
    heat[...] = np.array([255, 35, 145], dtype=np.float32)
    strength = (0.10 + 0.48 * alpha)[..., None] * (alpha > 0)[..., None]
    out = out * (1.0 - strength) + heat * strength
    result = np.clip(out, 0, 255).astype(np.uint8)
    _draw_contour(result, hard, (35, 235, 255), 2)
    return result


def _alpha_map(alpha: np.ndarray, hard: np.ndarray) -> np.ndarray:
    gray = np.clip(np.round(alpha * 255.0), 0, 255).astype(np.uint8)
    out = np.repeat(gray[..., None], 3, axis=2)
    spill = (alpha > 0.0) & ~hard
    out[spill, 0] = np.maximum(out[spill, 0], 180)
    out[spill, 1] = (out[spill, 1] * 0.25).astype(np.uint8)
    out[spill, 2] = (out[spill, 2] * 0.25).astype(np.uint8)
    _draw_contour(out, hard, (35, 235, 255), 2)
    contours, _ = cv2.findContours(
        hard.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if contours:
        contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
        h, w = hard.shape
        target = np.array([w * 0.5, h * 0.5])
        point = contour[int(np.argmin(np.linalg.norm(contour - target, axis=1)))]
        crop_half = max(10, int(min(h, w) * 0.09))
        x0, x1 = max(0, point[0] - crop_half), min(w, point[0] + crop_half)
        y0, y1 = max(0, point[1] - crop_half), min(h, point[1] + crop_half)
        crop = out[y0:y1, x0:x1]
        inset_w, inset_h = max(48, int(w * 0.34)), max(48, int(h * 0.34))
        if crop.size:
            inset = cv2.resize(crop, (inset_w, inset_h), interpolation=cv2.INTER_NEAREST)
            ix, iy = w - inset_w - 6, h - inset_h - 6
            cv2.rectangle(out, (ix - 2, iy - 2), (w - 4, h - 4), (255, 255, 255), 2)
            out[iy : iy + inset_h, ix : ix + inset_w] = inset
    return out


def _fit_panel(image: np.ndarray, size: tuple[int, int]) -> Image.Image:
    panel = Image.new("RGB", size, (24, 26, 30))
    im = Image.fromarray(image)
    im.thumbnail((size[0] - 12, size[1] - 12), Image.Resampling.LANCZOS)
    panel.paste(im, ((size[0] - im.width) // 2, (size[1] - im.height) // 2))
    return panel


def _candidate_tile(
    image: np.ndarray,
    candidate: Candidate,
    alpha: np.ndarray,
    ev: float,
    tile_size: tuple[int, int] = (820, 330),
) -> Image.Image:
    tile = Image.new("RGB", tile_size, (245, 246, 248))
    draw = ImageDraw.Draw(tile)
    title = (
        f"{candidate.index:02d} {candidate.kind} / {candidate.variant} | "
        f"{candidate.feather.name} | in {candidate.feather.feather_in * 100:.2f}% "
        f"out {candidate.feather.feather_out * 100:.2f}%"
    )
    draw.text((14, 10), title, fill=(17, 22, 29), font=_font(17, bold=True))
    outside = alpha[~candidate.hard_mask]
    outside_max = float(outside.max()) if outside.size else 0.0
    stats = (
        f"hard {candidate.hard_mask.mean() * 100:.1f}% | alpha mean {alpha.mean() * 100:.1f}% "
        f"| exterior max {outside_max:.3f}"
    )
    draw.text((14, 37), stats, fill=(65, 72, 82), font=_font(14))

    panel_size = (255, 235)
    panels = (
        (_alpha_overlay(image, alpha, candidate.hard_mask), "alpha overlay"),
        (_alpha_map(alpha, candidate.hard_mask), "alpha; red = bounded outside"),
        (_exposure_preview(image, alpha, ev), f"local exposure {ev:+.1f} EV"),
    )
    for i, (panel_image, label) in enumerate(panels):
        x = 14 + i * 268
        tile.paste(_fit_panel(panel_image, panel_size), (x, 69))
        draw.text((x + 4, 307), label, fill=(54, 61, 71), font=_font(13))
    return tile


def _sheet_for_source(
    record: SourceRecord,
    image: np.ndarray,
    candidates: Sequence[Candidate],
    alphas: Sequence[np.ndarray],
    ev: float,
) -> Image.Image:
    width, header, gap = 1680, 112, 14
    tile_w, tile_h = 820, 330
    sheet = Image.new("RGB", (width, header + 4 * tile_h + 5 * gap), (31, 34, 39))
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (18, 14),
        f"{record.asset_id} | VLM main_subject: {record.concept}",
        fill=(247, 249, 252),
        font=_font(26, bold=True),
    )
    draw.text(
        (18, 52),
        "cyan = hard support boundary; magenta = alpha; red in alpha panel = bounded exterior feather",
        fill=(198, 204, 214),
        font=_font(16),
    )
    draw.text(
        (18, 79),
        record.source_path,
        fill=(151, 160, 173),
        font=_font(13),
    )
    for index, (candidate, alpha) in enumerate(zip(candidates, alphas)):
        row, col = divmod(index, 2)
        x = gap + col * (tile_w + gap)
        y = header + gap + row * (tile_h + gap)
        sheet.paste(_candidate_tile(image, candidate, alpha, ev), (x, y))
    return sheet


def _overview(
    rows: Sequence[tuple[SourceRecord, np.ndarray, Sequence[Candidate], Sequence[np.ndarray], float]],
) -> Image.Image:
    label_w, cell_w, cell_h = 210, 190, 180
    header_h = 112
    width = label_w + 8 * cell_w
    height = header_h + len(rows) * cell_h
    sheet = Image.new("RGB", (width, height), (27, 30, 35))
    draw = ImageDraw.Draw(sheet)
    draw.text((16, 12), "Feather evaluation v1: final local-exposure previews", fill="white", font=_font(24, True))
    draw.text((16, 48), "1 semantic + 1 radial + 3 band + 3 linear", fill=(188, 196, 207), font=_font(15))
    if rows:
        for candidate in rows[0][2]:
            x = label_w + (candidate.index - 1) * cell_w + 8
            draw.text(
                (x, 82),
                f"{candidate.index} {candidate.kind}",
                fill=(226, 230, 237),
                font=_font(13, True),
            )

    for row_index, (record, image, candidates, alphas, ev) in enumerate(rows):
        y = header_h + row_index * cell_h
        fill = (242, 244, 247) if row_index % 2 == 0 else (229, 232, 237)
        ImageDraw.Draw(sheet).rectangle((0, y, width, y + cell_h), fill=fill)
        source_thumb = _fit_panel(image, (120, 118))
        sheet.paste(source_thumb, (8, y + 8))
        draw = ImageDraw.Draw(sheet)
        draw.text((10, y + 132), record.asset_id[:24], fill=(26, 31, 38), font=_font(12, True))
        draw.text((10, y + 151), record.concept, fill=(65, 73, 84), font=_font(13))
        for col, (candidate, alpha) in enumerate(zip(candidates, alphas)):
            preview = _exposure_preview(image, alpha, ev)
            panel = _fit_panel(preview, (cell_w - 10, 132))
            x = label_w + col * cell_w + 5
            sheet.paste(panel, (x, y + 6))
            draw.text(
                (x + 3, y + 142),
                f"in {candidate.feather.feather_in * 100:g}% / out {candidate.feather.feather_out * 100:g}%",
                fill=(49, 56, 66),
                font=_font(11),
            )
            draw.text((x + 3, y + 159), candidate.variant[:25], fill=(83, 91, 102), font=_font(10))
    return sheet


def _fetch_candidates(concepts: Sequence[str], seed: str, limit: int = 250) -> dict[str, list[SourceRecord]]:
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("psycopg is required for automatic source selection") from exc

    query = """
        SELECT a.asset_id, a.path, sc.main_subject, sm.png_path, sm.area, sm.bbox
        FROM source_captions sc
        JOIN assets a USING(asset_id)
        JOIN sam3_masks sm
          ON sm.asset_id = sc.asset_id AND sm.concept = sc.main_subject
        WHERE a.asset_type = 'image'
          AND sc.main_subject = %s
          AND sm.png_path IS NOT NULL
          AND sm.area BETWEEN 0.04 AND 0.42
          AND COALESCE(a.final_decision, a.auto_verdict, 'keep') <> 'drop'
        ORDER BY md5(a.asset_id || %s)
        LIMIT %s
    """
    selected: dict[str, list[SourceRecord]] = {}
    with psycopg.connect(qa_config.PG_DSN) as conn:
        with conn.cursor() as cursor:
            for concept in concepts:
                cursor.execute(query, (concept, seed, limit))
                selected[concept] = [
                    SourceRecord(
                        asset_id=row[0],
                        source_path=row[1],
                        concept=row[2],
                        mask_path=row[3],
                        db_area=float(row[4]),
                        db_bbox=json.loads(row[5]) if isinstance(row[5], str) else row[5],
                    )
                    for row in cursor.fetchall()
                ]
    return selected


def _choose_records(
    concepts: Sequence[str], seed: str, preview_long_edge: int
) -> list[SourceRecord]:
    pools = _fetch_candidates(concepts, seed)
    records: list[SourceRecord] = []
    for concept in concepts:
        for record in pools.get(concept, []):
            if not os.path.isfile(record.source_path) or not os.path.isfile(record.mask_path):
                continue
            try:
                image = _load_preview(record.source_path, preview_long_edge)
                mask = _load_subject_mask(record.mask_path, image.shape[:2])
            except (OSError, ValueError):
                continue
            area = float(mask.mean())
            if not (0.04 <= area <= 0.42):
                continue
            if _significant_component_count(mask) != 1:
                continue
            records.append(record)
            break
        else:
            raise RuntimeError(f"no single-component review sample found for concept {concept!r}")
    return records


def _manifest_candidate(candidate: Candidate, alpha: np.ndarray) -> dict[str, Any]:
    outside = alpha[~candidate.hard_mask]
    return {
        "index": candidate.index,
        "kind": candidate.kind,
        "variant": candidate.variant,
        "feather_profile": candidate.feather.name,
        "feather_in_short_edge_fraction": candidate.feather.feather_in,
        "feather_out_short_edge_fraction": candidate.feather.feather_out,
        "geometry": candidate.geometry,
        "hard_coverage": float(candidate.hard_mask.mean()),
        "alpha_mean": float(alpha.mean()),
        "alpha_nonzero_coverage": float((alpha > 0.0).mean()),
        "exterior_alpha_max": float(outside.max()) if outside.size else 0.0,
        "exterior_nonzero_coverage": float(((alpha > 0.0) & ~candidate.hard_mask).mean()),
    }


def generate(
    out_dir: str,
    concepts: Sequence[str],
    seed: str,
    preview_long_edge: int,
    exposure_ev: float,
) -> Path:
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = _choose_records(concepts, seed, preview_long_edge)
    overview_rows = []
    manifest: dict[str, Any] = {
        "version": "feather_eval_v1",
        "evaluation_only": True,
        "selection": {
            "annotation": "source_captions.main_subject",
            "mask": "sam3_masks.png_path",
            "single_significant_component_required": True,
            "seed": seed,
            "concepts": list(concepts),
        },
        "preview_long_edge": preview_long_edge,
        "candidate_allocation": {"semantic": 1, "radial": 1, "band": 3, "linear": 3},
        "preview_edit": {"kind": "linear-light exposure", "ev": exposure_ev},
        "sources": [],
    }

    for record in records:
        image = _load_preview(record.source_path, preview_long_edge)
        subject = _load_subject_mask(record.mask_path, image.shape[:2])
        candidates = build_candidates(subject)
        alphas = [
            feather_binary(
                candidate.hard_mask,
                candidate.feather.feather_in,
                candidate.feather.feather_out,
            )
            for candidate in candidates
        ]
        sheet = _sheet_for_source(record, image, candidates, alphas, exposure_ev)
        sheet_path = output / f"{record.asset_id}.contact.jpg"
        sheet.save(sheet_path, quality=94, subsampling=0)
        overview_rows.append((record, image, candidates, alphas, exposure_ev))
        manifest["sources"].append(
            {
                "asset_id": record.asset_id,
                "source_path": record.source_path,
                "main_subject": record.concept,
                "sam3_mask_path": record.mask_path,
                "db_mask_area": record.db_area,
                "db_mask_bbox": record.db_bbox,
                "preview_shape": list(image.shape[:2]),
                "significant_component_count": _significant_component_count(subject),
                "contact_sheet": str(sheet_path),
                "candidates": [
                    _manifest_candidate(candidate, alpha)
                    for candidate, alpha in zip(candidates, alphas)
                ],
            }
        )

    overview_path = output / "overview.jpg"
    _overview(overview_rows).save(overview_path, quality=94, subsampling=0)
    manifest["overview"] = str(overview_path)
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def _parse_concepts(values: Iterable[str]) -> list[str]:
    concepts = []
    for value in values:
        concepts.extend(part.strip() for part in value.split(",") if part.strip())
    return concepts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--concept",
        action="append",
        default=[],
        help="exact VLM main_subject; repeat or pass comma-separated values",
    )
    parser.add_argument("--seed", default="feather-eval-v1")
    parser.add_argument("--preview-long-edge", type=int, default=768)
    parser.add_argument("--exposure-ev", type=float, default=1.25)
    args = parser.parse_args()
    concepts = _parse_concepts(args.concept) or list(DEFAULT_CONCEPTS)
    manifest = generate(
        args.out_dir,
        concepts,
        args.seed,
        args.preview_long_edge,
        args.exposure_ev,
    )
    print(manifest)


if __name__ == "__main__":
    main()
