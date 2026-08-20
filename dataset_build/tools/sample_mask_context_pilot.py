"""Generate a review-only pilot for broader subject-aware local masks.

This tool deliberately does not change the canonical production mask contract.
It samples SAM-ready sources, constructs the proposed context/radial/band masks,
and writes visual review sheets plus machine-readable measurements.

Run from the repository root:

    PYTHONPATH=.:dataset_build:dataset_build/src \
      .venv-lens/bin/python -m dataset_build.tools.sample_mask_context_pilot
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from construct.canonical_masks import linear_strength, raster_geometry
from construct.sources import (
    SourceInventoryResult,
    SourceRecord,
    _precomputed_inventory,
    load_scene_metadata,
    scene_stratified_order,
)
from construct.subject_geom import linear_geom
from dataset_build.tools.archive_reader import open_image, open_rgb


BUILD_ID = "mask-protocol-pilot-v2"
SUBJECT_CACHE = Path("/home/bc/data/datasets/vera_directionA_1M/subject_cache")
POSTGRES_DSN = "postgresql://research:research@127.0.0.1:5432/research"
DEFAULT_OUT = Path("docs/assets/mask_protocol_pilot100_v2_20260818")
AREA_BUCKETS = (
    ("tiny", 0.005, 0.020),
    ("small", 0.020, 0.060),
    ("medium", 0.060, 0.150),
    ("large", 0.150, 0.850001),
)
COLORS = {
    "core": (238, 73, 73),
    "semantic": (30, 200, 220),
    "radial": (49, 196, 112),
    "band": (242, 190, 52),
    "linear": (208, 92, 227),
}
_RESAMPLING = getattr(Image, "Resampling", Image)


class PilotReject(RuntimeError):
    """A source cannot satisfy every proposed pilot range."""

    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code


@dataclass(frozen=True, slots=True)
class PilotMask:
    name: str
    mode: str
    alpha: np.ndarray
    meta: dict[str, Any]


def _seed_int(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _mask_stats(alpha: np.ndarray) -> dict[str, float]:
    alpha = np.asarray(alpha, dtype=np.float32)
    return {
        "mean": float(alpha.mean()),
        "half_area": float((alpha >= 0.5).mean()),
        "support_area": float((alpha > 0.05).mean()),
        "max": float(alpha.max()),
    }


def _subject_coverage(alpha: np.ndarray, core: np.ndarray) -> dict[str, float]:
    subject = core > 0.5
    if not subject.any():
        raise PilotReject("subject_empty", "subject has no pixels")
    values = np.asarray(alpha, dtype=np.float32)[subject]
    return {
        "subject_high_coverage": float((values >= 0.5).mean()),
        "subject_support_coverage": float((values > 0.05).mean()),
    }


def _working_views(source: SourceRecord, display_long_edge: int, work_short_edge: int):
    image = open_rgb(source.source_path)
    scale = min(1.0, display_long_edge / max(image.size))
    display_size = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    display = image.resize(display_size, _RESAMPLING.LANCZOS)

    subject = open_image(source.subject_path).convert("L")
    core_display = np.asarray(
        subject.resize(display_size, _RESAMPLING.NEAREST), dtype=np.float32
    ) / 255.0
    core_display = (core_display > 0.5).astype(np.float32)

    work_scale = min(1.0, work_short_edge / min(display_size))
    work_size = (
        max(1, round(display_size[0] * work_scale)),
        max(1, round(display_size[1] * work_scale)),
    )
    rgb_work = cv2.resize(
        np.asarray(display, dtype=np.uint8), work_size, interpolation=cv2.INTER_AREA
    )
    core_work = cv2.resize(
        core_display, work_size, interpolation=cv2.INTER_NEAREST
    ).astype(np.float32)
    core_work = (core_work > 0.5).astype(np.float32)
    area = float(core_work.mean())
    if area < 0.005 or area > 0.85:
        raise PilotReject("subject_area", f"working subject area {area:.4f}")
    return display, core_display, rgb_work, core_work


def _semantic_core(core: np.ndarray) -> PilotMask:
    alpha = (core > 0.5).astype(np.float32)
    return PilotMask(
        "semantic_core", "semantic", alpha,
        {**_mask_stats(alpha), **_subject_coverage(alpha, core), "expanded": False},
    )


def _subject_geometry(core: np.ndarray):
    ys, xs = np.nonzero(core > 0.5)
    if len(xs) < 32:
        raise PilotReject("subject_points", f"only {len(xs)} subject pixels")
    points = np.stack([xs, ys], axis=1).astype(np.float64)
    center = points.mean(axis=0)
    covariance = np.cov((points - center).T)
    _, vectors = np.linalg.eigh(covariance)
    major = vectors[:, 1]
    angle = math.atan2(major[1], major[0])
    return points, center, angle


def _ellipse_alpha(shape: tuple[int, int], center: np.ndarray, angle: float,
                   axis_a: float, axis_b: float) -> np.ndarray:
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    dx, dy = xx - center[0], yy - center[1]
    cosine, sine = math.cos(angle), math.sin(angle)
    along = dx * cosine + dy * sine
    across = -dx * sine + dy * cosine
    distance = np.sqrt((along / axis_a) ** 2 + (across / axis_b) ** 2)
    return _smoothstep((1.25 - distance) / 0.50).astype(np.float32)


def _passes_edit_gate(alpha: np.ndarray, core: np.ndarray, target: float) -> bool:
    stats = _mask_stats(alpha)
    coverage = _subject_coverage(alpha, core)
    return (
        stats["mean"] >= target
        and coverage["subject_high_coverage"] >= 0.98
        and coverage["subject_support_coverage"] >= 1.0
    )


def _radial_masks(core: np.ndarray, rng: random.Random, count: int) -> list[PilotMask]:
    points, center, pca_angle = _subject_geometry(core)
    short = float(min(core.shape))
    masks = []
    for index in range(count):
        if index == 0:
            angle = pca_angle + math.radians(rng.uniform(-12.0, 12.0))
        elif index == 1:
            angle = rng.choice((0.0, math.pi / 2.0)) + math.radians(
                rng.uniform(-15.0, 15.0)
            )
        else:
            angle = pca_angle + math.pi / 2.0 + math.radians(rng.uniform(-15.0, 15.0))
        unit = np.array([math.cos(angle), math.sin(angle)])
        normal = np.array([-math.sin(angle), math.cos(angle)])
        centered = points - center
        extent_a = float(np.quantile(np.abs(centered @ unit), 0.98))
        extent_b = float(np.quantile(np.abs(centered @ normal), 0.98))
        axis_a = max(1.18 * extent_a, 0.25 * short)
        axis_b = max(1.18 * extent_b, 0.18 * short)
        if axis_a < axis_b:
            axis_a, axis_b = axis_b, axis_a
            angle += math.pi / 2.0
        target = rng.uniform(0.46, 0.58)

        def at_scale(scale: float) -> np.ndarray:
            return _ellipse_alpha(
                core.shape, center, angle, axis_a * scale, axis_b * scale
            )

        low, high = 1.0, 1.0
        base = at_scale(high)
        while not _passes_edit_gate(base, core, target) and high < 16.0:
            high *= 1.5
            base = at_scale(high)
        if not _passes_edit_gate(base, core, target):
            raise PilotReject("radial_gate", f"radial-{index} cannot satisfy gate")
        if high > 1.0:
            for _ in range(28):
                scale = (low + high) / 2.0
                if _passes_edit_gate(at_scale(scale), core, target):
                    high = scale
                else:
                    low = scale
            base = at_scale(high)
        axis_a *= high
        axis_b *= high
        stats = _mask_stats(base)
        coverage = _subject_coverage(base, core)
        if stats["mean"] <= 0.45:
            raise PilotReject("radial_mass", f"radial-{index} mean={stats['mean']:.4f}")
        masks.append(PilotMask(
            f"radial_{index}", "radial", base,
            {
                **stats,
                **coverage,
                "target_mean": target,
                "half_major_short": axis_a / short,
                "half_minor_short": axis_b / short,
                "angle_degrees": math.degrees(angle) % 180.0,
            },
        ))
    return masks


def _band_alpha(shape: tuple[int, int], center: np.ndarray, angle: float,
                half_width: float) -> np.ndarray:
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    normal_x, normal_y = -math.sin(angle), math.cos(angle)
    distance = np.abs((xx - center[0]) * normal_x + (yy - center[1]) * normal_y)
    return _smoothstep((1.25 - distance / half_width) / 0.50).astype(np.float32)


def _band_directions(shape: tuple[int, int], rng: random.Random,
                     count: int) -> list[tuple[str, float]]:
    height, width = shape
    if width > height:
        offsets = np.linspace(-10.0, 10.0, count) if count > 1 else np.array([0.0])
        return [
            ("vertical", math.pi / 2.0 + math.radians(float(offset)))
            for offset in offsets
        ]
    diagonal = rng.choice((45.0, 135.0))
    pool = [("horizontal", 0.0), ("vertical", 90.0), ("diagonal", diagonal)]
    if count < len(pool):
        rng.shuffle(pool)
        pool = pool[:count]
    return [
        (name, math.radians(degrees + rng.uniform(-8.0, 8.0)))
        for name, degrees in pool
    ]


def _band_masks(core: np.ndarray, rng: random.Random, count: int) -> list[PilotMask]:
    _points, center, _pca_angle = _subject_geometry(core)
    short = float(min(core.shape))
    masks = []
    for index, (direction, angle) in enumerate(_band_directions(core.shape, rng, count)):
        target = rng.uniform(0.46, 0.58)
        low, high = 0.18 * short, 2.0 * max(core.shape)

        def at_width(half_width: float) -> np.ndarray:
            return _band_alpha(core.shape, center, angle, half_width)

        if _passes_edit_gate(at_width(low), core, target):
            half_width = low
        elif not _passes_edit_gate(at_width(high), core, target):
            raise PilotReject("band_gate", f"band-{index} cannot satisfy gate")
        else:
            for _ in range(28):
                middle = (low + high) / 2.0
                if _passes_edit_gate(at_width(middle), core, target):
                    high = middle
                else:
                    low = middle
            half_width = high
        alpha = at_width(half_width)
        stats = _mask_stats(alpha)
        coverage = _subject_coverage(alpha, core)
        if stats["mean"] <= 0.45:
            raise PilotReject("band_mass", f"band-{index} mean={stats['mean']:.4f}")
        full_width = 2.0 * half_width / short
        masks.append(PilotMask(
            f"band_{index}", "band", alpha,
            {
                **stats,
                **coverage,
                "target_mean": target,
                "half_width_short": half_width / short,
                "full_width_short": full_width,
                "angle_degrees": math.degrees(angle) % 180.0,
                "direction": direction,
            },
        ))
    return masks


def _linear_masks(core: np.ndarray, rng: random.Random) -> list[PilotMask]:
    ys, xs = np.nonzero(core > 0.5)
    height, width = core.shape
    bbox = (
        float(xs.min()) / width,
        float(ys.min()) / height,
        float(xs.max() + 1) / width,
        float(ys.max() + 1) / height,
    )
    rooms = {
        "left": bbox[0], "right": 1.0 - bbox[2],
        "top": bbox[1], "bottom": 1.0 - bbox[3],
    }
    sides = [side for side, room in sorted(rooms.items(), key=lambda item: -item[1])
             if room >= 0.20]
    if not sides:
        raise PilotReject("linear_room", f"rooms={rooms}")
    masks = []
    for index in range(2):
        side = sides[index % len(sides)]
        spec = linear_geom(
            bbox, rng, apply_subject_side=True, area=float(core.mean()), side=side
        )
        if spec is None:
            raise PilotReject("linear_geometry", f"side={side}")
        raw = raster_geometry("gradient", spec["geom"], height, width)
        amount, alpha = linear_strength(raw, 0.50)
        stats = _mask_stats(alpha)
        masks.append(PilotMask(
            f"linear_{index}", "linear", alpha,
            {**stats, "side": side, "amount": amount},
        ))
    return masks


def _build_masks(core: np.ndarray, seed: int) -> tuple[list[PilotMask], str]:
    rng = random.Random(seed)
    core_mask = PilotMask(
        "subject_core", "core", core,
        {**_mask_stats(core), **_subject_coverage(core, core)},
    )
    large = float(core.mean()) >= 0.15
    radial_count = 2 if large else 3
    band_count = 2 if large else 3
    masks = [core_mask]
    if large:
        masks.append(_semantic_core(core))
    masks.extend(_radial_masks(core, rng, radial_count))
    masks.extend(_band_masks(core, rng, band_count))
    masks.extend(_linear_masks(core, rng))
    slot_plan = "2 radial + 2 semantic + 2 band + 2 linear" if large \
        else "3 radial + 3 band + 2 linear (no semantic)"
    return masks, slot_plan


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _panel(image: Image.Image, mask: PilotMask | None, title: str) -> Image.Image:
    rgb = np.asarray(image, dtype=np.float32)
    if mask is not None:
        alpha = cv2.resize(
            mask.alpha, image.size, interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)
        color = np.asarray(COLORS[mask.mode], dtype=np.float32)
        opacity = (0.58 * alpha)[..., None]
        rgb = rgb * (1.0 - opacity) + color * opacity
        contours, _ = cv2.findContours(
            (alpha >= 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(rgb, contours, -1, COLORS[mask.mode], 2)
    body = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")
    has_coverage = mask is not None and "subject_high_coverage" in mask.meta
    bar_height = 70 if has_coverage else 52
    panel = Image.new("RGB", (body.width, body.height + bar_height), (20, 23, 27))
    panel.paste(body, (0, bar_height))
    draw = ImageDraw.Draw(panel)
    draw.text((9, 5), title, fill=(244, 246, 248), font=_font(14))
    if mask is not None:
        meta = mask.meta
        line = (
            f"mean {meta['mean']:.3f} | >=.5 {meta['half_area']:.3f} | "
            f">.05 {meta['support_area']:.3f}"
        )
        draw.text((9, 27), line, fill=(188, 195, 204), font=_font(12))
        if has_coverage:
            coverage = (
                f"subject >=.5 {meta['subject_high_coverage']:.3f} | "
                f">.05 {meta['subject_support_coverage']:.3f}"
            )
            draw.text((9, 47), coverage, fill=(164, 213, 178), font=_font(12))
    return panel


def _save_sheet(path: Path, image: Image.Image, source: SourceRecord,
                bucket: str, masks: list[PilotMask], slot_plan: str) -> None:
    panels = [_panel(image, None, "original")]
    panels.extend(_panel(image, mask, mask.name) for mask in masks)
    columns = 4
    rows = math.ceil(len(panels) / columns)
    header = 76
    panel_width = panels[0].width
    panel_height = max(panel.height for panel in panels)
    sheet = Image.new(
        "RGB", (columns * panel_width, header + rows * panel_height), (12, 14, 17)
    )
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (12, 8),
        f"{source.source_id} | scene={source.scene} | bucket={bucket} | "
        f"subject_area={masks[0].meta['mean']:.3f}",
        fill=(245, 247, 249), font=_font(18),
    )
    draw.text(
        (12, 33), f"slots: {slot_plan}",
        fill=(174, 181, 191), font=_font(13),
    )
    draw.text(
        (12, 53), "cyan=semantic  green=radial  yellow=band  magenta=linear",
        fill=(150, 158, 168), font=_font(12),
    )
    for index, panel in enumerate(panels):
        x = (index % columns) * panel_width
        y = header + (index // columns) * panel_height + (panel_height - panel.height)
        sheet.paste(panel, (x, y))
    sheet.save(path, quality=92)


def _quantiles(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "n": int(array.size),
        "min": float(array.min()),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def _write_index(out: Path, records: list[dict[str, Any]]) -> None:
    cards = []
    for record in records:
        filename = html.escape(record["file"])
        cards.append(
            f'<a class="card" href="{filename}"><img loading="lazy" src="{filename}">'
            f'<span>{html.escape(record["bucket"])} | '
            f'{html.escape(record["scene"])} | A={record["subject_area"]:.3f}</span></a>'
        )
    page = """<!doctype html><meta charset="utf-8"><title>Mask protocol v2 pilot 100</title>
<style>
body{margin:0;background:#111418;color:#edf0f3;font:14px sans-serif}
header{position:sticky;top:0;background:#171b20;padding:14px 20px;z-index:2}
main{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:10px;padding:10px}
.card{color:#dce1e6;text-decoration:none;background:#1b2026;border:1px solid #303740;padding:6px}
.card img{display:block;width:100%;height:auto}.card span{display:block;padding:7px 3px 2px}
</style><header><b>Mask protocol v2 pilot 100</b> - click any sheet for full resolution</header><main>"""
    page += "".join(cards) + "</main>"
    (out / "index.html").write_text(page, encoding="utf-8")


def _pilot_inventory() -> SourceInventoryResult:
    """Use the landed eligibility index without scanning the sparse relabel overlay."""
    by_id, by_path, metadata_status = load_scene_metadata(POSTGRES_DSN)
    precomputed = _precomputed_inventory(SUBJECT_CACHE, by_id, by_path)
    if precomputed is None:
        raise RuntimeError("precomputed subject inventory is unavailable")
    rows, counts = precomputed
    rows.sort(key=lambda row: row.source_id)
    return SourceInventoryResult(
        eligible=tuple(rows),
        counts=dict(sorted(counts.items())),
        scene_metadata_status=f"precomputed_archive:{metadata_status}",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--display-long-edge", type=int, default=600)
    parser.add_argument("--work-short-edge", type=int, default=224)
    parser.add_argument("--max-attempts", type=int, default=3000)
    args = parser.parse_args(argv)
    if args.count % len(AREA_BUCKETS):
        raise SystemExit(f"--count must be divisible by {len(AREA_BUCKETS)}")

    args.out.mkdir(parents=True, exist_ok=True)
    inventory = _pilot_inventory()
    quota = args.count // len(AREA_BUCKETS)
    by_bucket: dict[str, list[SourceRecord]] = {}
    for name, low, high in AREA_BUCKETS:
        rows = [row for row in inventory.eligible if low <= row.mask_area < high]
        by_bucket[name] = scene_stratified_order(rows, BUILD_ID + "-" + name, args.seed)

    accepted: list[dict[str, Any]] = []
    # C1b item 10: each accepted record costs a full mask build plus a rendered contact
    # sheet, and the bucket loop raises `RuntimeError` on a quota shortfall - which used
    # to discard every record produced up to that point, because `samples.jsonl` was
    # only written after the loop. Records are appended as they are accepted.
    samples_handle = (args.out / "samples.jsonl").open("w", encoding="utf-8")
    rejects: Counter[str] = Counter()
    attempts = 0
    for bucket, _low, _high in AREA_BUCKETS:
        made = 0
        for source in by_bucket[bucket]:
            if made >= quota:
                break
            if attempts >= args.max_attempts:
                break
            attempts += 1
            try:
                display, _core_display, _rgb_work, core_work = _working_views(
                    source, args.display_long_edge, args.work_short_edge
                )
                masks, slot_plan = _build_masks(
                    core_work,
                    _seed_int(BUILD_ID, args.seed, source.source_id),
                )
            except PilotReject as exc:
                rejects[exc.code] += 1
                continue
            except Exception as exc:  # noqa: BLE001 - preserve pilot progress
                rejects[f"unexpected:{type(exc).__name__}"] += 1
                continue
            filename = f"{len(accepted) + 1:03d}_{bucket}_{source.source_id[:16]}.jpg"
            _save_sheet(
                args.out / filename, display, source, bucket, masks, slot_plan
            )
            record = {
                "index": len(accepted) + 1,
                "file": filename,
                "source_id": source.source_id,
                "source_path": str(source.source_path),
                "scene": source.scene,
                "subject": source.subject,
                "bucket": bucket,
                "subject_area": masks[0].meta["mean"],
                "slot_plan": slot_plan,
                "masks": {mask.name: mask.meta for mask in masks},
            }
            accepted.append(record)
            samples_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            samples_handle.flush()
            made += 1
            print(json.dumps({"accepted": len(accepted), "file": filename}), flush=True)
        if made != quota:
            samples_handle.close()
            raise RuntimeError(f"bucket {bucket} produced {made}/{quota} samples")

    samples_handle.close()

    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in accepted:
        for name, meta in record["masks"].items():
            for field in (
                "mean", "half_area", "support_area",
                "subject_high_coverage", "subject_support_coverage",
            ):
                if field in meta:
                    values[name][field].append(float(meta[field]))
    summary = {
        "count": len(accepted),
        "seed": args.seed,
        "inventory_eligible": len(inventory.eligible),
        "inventory_counts": inventory.counts,
        "scene_metadata_status": inventory.scene_metadata_status,
        "attempts": attempts,
        "rejects": dict(sorted(rejects.items())),
        "by_bucket": dict(Counter(record["bucket"] for record in accepted)),
        "by_scene": dict(Counter(record["scene"] for record in accepted)),
        "metrics": {
            name: {field: _quantiles(series) for field, series in fields.items()}
            for name, fields in values.items()
        },
        "rules": {
            "semantic": "exact SAM core only when subject_area >= 0.15",
            "nonlarge_slots": "3 radial + 3 band + 2 linear",
            "large_slots": "2 radial + 2 semantic + 2 band + 2 linear",
            "radial_effective_mean": "> 0.45",
            "radial_half_axes_short_edge_min": [0.25, 0.18],
            "band_effective_mean": "> 0.45",
            "band_directions_portrait": ["horizontal", "vertical", "diagonal"],
            "band_direction_landscape": "vertical",
            "subject_high_coverage_min": 0.98,
            "subject_support_coverage": 1.0,
            "linear_effective_mean_target": 0.50,
        },
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    notes = f"""# Mask protocol v2 pilot 100

Review-only output. Production mask code was not changed.

- Samples: {len(accepted)} ({quota} per subject-area bucket)
- Seed: {args.seed}
- Source attempts: {attempts}
- Browse: `index.html`
- Per-sample measurements: `samples.jsonl`
- Aggregate measurements and rejection counts: `summary.json`

Subjects below 15% use three radial, three band, and two linear slots. Large
subjects retain the exact SAM semantic core and use the canonical 2+2+2+2 slot
mix. Semantic context expansion is disabled. Every radial and band has effective
alpha mean above 0.45, at least 98% high-strength subject coverage, and 100%
subject support. Portrait bands cover horizontal, vertical, and diagonal
directions; landscape bands are vertical. Panel measurements include both mask
area and subject coverage.
"""
    (args.out / "README.md").write_text(notes, encoding="utf-8")
    # The contact-sheet index is a viewing convenience; a failure here must not lose the
    # records and the summary that are already on disk.
    index_error = None
    try:
        _write_index(args.out, accepted)
    except Exception as exc:  # noqa: BLE001 - reported, never silent
        index_error = f"{type(exc).__name__}: {exc}"
    print(json.dumps({"out": str(args.out), "count": len(accepted),
                      "attempts": attempts, "index_error": index_error}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
