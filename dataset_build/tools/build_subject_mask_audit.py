"""Build a deterministic human-audit set for VLM-selected SAM3 subject masks.

The tool does not call a model or change production data. It samples the current
construct-eligible pool, overlays each stored ``main_subject`` mask on its source,
and writes contact sheets plus a CSV with empty human-review fields.

Run from the repository root::

    python -m dataset_build.tools.build_subject_mask_audit \
        --out _mask_review/subject_audit_v1 --n 600
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from dataset_build.source_qa import config, db


_MULTI_RE = re.compile(
    r"\b(people|persons|couple|group|crowd|family|children|men|women|"
    r"bride and groom|cars|birds|animals|flowers|trees)\b",
    re.I,
)
_PORTRAIT_RE = re.compile(
    r"\b(person|people|woman|women|man|men|girl|boy|bride|groom|couple|"
    r"child|children|baby|model|dancer|hiker|skier|surfer)\b",
    re.I,
)
_ANIMAL_RE = re.compile(
    r"\b(dog|cat|bird|horse|cow|sheep|deer|bear|fox|rabbit|pet|animal|"
    r"seagull|duck|swan|elephant|lion|tiger|monkey|fish)\w*\b",
    re.I,
)
_PRODUCT_RE = re.compile(
    r"\b(food|plate|dish|meal|cake|fruit|drink|coffee|bottle|cup|watch|"
    r"shoe|bag|camera|phone|product|jewelry|flower|car|motorcycle|bicycle)\w*\b",
    re.I,
)
_LANDSCAPE_RE = re.compile(
    r"\b(sky|skies|cloud|clouds|mountain|mountains|water|waterfall|waterfalls|"
    r"ocean|oceans|sea|seas|river|rivers|lake|lakes|forest|forests|field|fields|"
    r"beach|beaches|sand|snow|milky way|sunset|sunrise|landscape|wave|waves)\b",
    re.I,
)


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _stratum(row: dict) -> str:
    subject = str(row.get("main_subject") or "")
    scene = str(row.get("scene") or "").lower()
    area = float(row.get("mask_area") or 0.0)
    if _MULTI_RE.search(subject):
        return "multi_instance"
    if _LANDSCAPE_RE.search(subject):
        return "landscape_or_no_subject"
    if area < 0.08:
        return "small_subject"
    if area > 0.45:
        return "large_subject"
    if scene in {"portrait", "wedding"} or _PORTRAIT_RE.search(subject):
        return "portrait"
    if _ANIMAL_RE.search(subject):
        return "animal"
    if scene in {"food", "product", "still_life"} or _PRODUCT_RE.search(subject):
        return "product_or_food"
    return "general"


def _rows() -> list[dict]:
    min_iaa = getattr(
        config, "CONSTRUCT_SOURCE_IAA_MIN", config.GATE["iaa_keep_above"])
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT a.asset_id, a.path, a.scene, a.megapixels, "
            "c.main_subject, c.subjects, m.png_path AS mask_path, "
            "m.area AS mask_area, m.bbox "
            "FROM assets a "
            "JOIN source_captions c USING(asset_id) "
            "JOIN sam3_masks m ON m.asset_id=a.asset_id "
            "AND m.concept=c.main_subject "
            "WHERE a.asset_type='image' AND a.b_quality=3 "
            "AND a.iaa_mixed IS NOT NULL AND a.iaa_mixed >= %s "
            "AND a.dup_of IS NULL AND m.png_path IS NOT NULL "
            "ORDER BY a.asset_id",
            (min_iaa,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def _sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if os.path.exists(row["path"]) and os.path.exists(row["mask_path"]):
            row["stratum"] = _stratum(row)
            buckets[row["stratum"]].append(row)
    names = [
        "multi_instance",
        "landscape_or_no_subject",
        "small_subject",
        "large_subject",
        "portrait",
        "animal",
        "product_or_food",
        "general",
    ]
    per = math.ceil(n / len(names))
    selected: list[dict] = []
    used: set[str] = set()
    for name in names:
        pool = list(buckets.get(name, ()))
        rng.shuffle(pool)
        for row in pool[:per]:
            if row["asset_id"] not in used:
                selected.append(row)
                used.add(row["asset_id"])
    if len(selected) < n:
        rest = [row for row in rows if row["asset_id"] not in used
                and os.path.exists(row["path"])
                and os.path.exists(row["mask_path"])]
        rng.shuffle(rest)
        for row in rest[: n - len(selected)]:
            row["stratum"] = _stratum(row)
            selected.append(row)
    rng.shuffle(selected)
    return selected[:n]


def _load_pair(row: dict, preview_size: tuple[int, int]) -> tuple[Image.Image, np.ndarray]:
    with Image.open(row["path"]) as source_im:
        source = source_im.convert("RGB")
        source.thumbnail(preview_size, Image.Resampling.LANCZOS)
    with Image.open(row["mask_path"]) as mask_im:
        mask_preview = mask_im.convert("L").resize(
            source.size, Image.Resampling.BILINEAR)
        mask = np.asarray(mask_preview, dtype=np.float32) / 255.0
    return source, np.clip(mask, 0.0, 1.0)


def _fit(im: Image.Image, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, (20, 20, 20))
    fitted = ImageOps.contain(im, size, Image.Resampling.LANCZOS)
    x = (size[0] - fitted.width) // 2
    y = (size[1] - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def _overlay(source: Image.Image, mask: np.ndarray) -> Image.Image:
    src = np.asarray(source, dtype=np.float32)
    color = np.zeros_like(src)
    color[..., 1] = 255.0
    alpha = (mask * 0.42)[..., None]
    out = src * (1.0 - alpha) + color * alpha
    try:
        import cv2

        edge = cv2.morphologyEx(
            (mask >= 0.5).astype(np.uint8), cv2.MORPH_GRADIENT,
            np.ones((3, 3), np.uint8),
        ).astype(bool)
        out[edge] = np.asarray([255.0, 220.0, 30.0], dtype=np.float32)
    except Exception:
        pass
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8), "RGB")


def _tile(row: dict, tile_w: int, tile_h: int) -> Image.Image:
    source, mask = _load_pair(row, (tile_w, tile_h - 48))
    vis = _fit(_overlay(source, mask), (tile_w, tile_h - 48))
    tile = Image.new("RGB", (tile_w, tile_h), (12, 12, 12))
    tile.paste(vis, (0, 48))
    draw = ImageDraw.Draw(tile)
    title_font = _font(15)
    small_font = _font(12)
    subject = str(row.get("main_subject") or "")
    title = f"#{int(row['audit_index']):03d} {row['stratum']} | {subject}"[:60]
    meta = (f"scene={row.get('scene') or '-'}  area={float(row.get('mask_area') or 0):.3f}  "
            f"mp={float(row.get('megapixels') or 0):.1f}  id={row['asset_id'][-8:]}")
    draw.text((6, 4), title, font=title_font, fill=(245, 245, 245))
    draw.text((6, 27), meta, font=small_font, fill=(180, 180, 180))
    return tile


def _write_sheets(rows: list[dict], out_dir: Path, cols: int, rows_per: int,
                  tile_w: int, tile_h: int) -> list[str]:
    paths: list[str] = []
    page_size = cols * rows_per
    for page, start in enumerate(range(0, len(rows), page_size), 1):
        batch = rows[start:start + page_size]
        sheet = Image.new(
            "RGB", (cols * tile_w, rows_per * tile_h), (8, 8, 8))
        with ThreadPoolExecutor(max_workers=min(8, len(batch))) as pool:
            tiles = list(pool.map(
                lambda row: _tile(row, tile_w, tile_h), batch))
        for idx, tile in enumerate(tiles):
            sheet.paste(tile, ((idx % cols) * tile_w, (idx // cols) * tile_h))
        path = out_dir / f"sheet_{page:03d}.jpg"
        sheet.save(path, "JPEG", quality=90, subsampling=0)
        paths.append(str(path))
    return paths


def _write_manifest(rows: list[dict], out_dir: Path) -> None:
    fields = [
        "index", "asset_id", "source_path", "mask_path", "scene", "stratum",
        "main_subject", "mask_area", "megapixels", "sample_weight", "reviewer",
        "review_version", "subject_correct",
        "single_instance_correct", "mask_usable", "should_skip",
        "failure_reason", "review_confidence", "notes",
    ]
    with open(out_dir / "audit.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "index": row["audit_index"],
                "asset_id": row["asset_id"],
                "source_path": row["path"],
                "mask_path": row["mask_path"],
                "scene": row.get("scene") or "",
                "stratum": row["stratum"],
                "main_subject": row.get("main_subject") or "",
                "mask_area": row.get("mask_area") or 0.0,
                "megapixels": row.get("megapixels") or 0.0,
                "sample_weight": row.get("sample_weight") or 0.0,
                "reviewer": "",
                "review_version": "subject_union_audit_v1",
                "subject_correct": "",
                "single_instance_correct": "",
                "mask_usable": "",
                "should_skip": "",
                "failure_reason": "",
                "review_confidence": "",
                "notes": "",
            })


def _write_review_guide(out_dir: Path) -> None:
    text = """# Subject union-mask audit v1

This is a stratified audit of the existing VLM main-subject **union** masks. It
does not yet evaluate the planned numbered SAM3 instance proposals.

Allowed values:

- `subject_correct`, `single_instance_correct`, `mask_usable`, `should_skip`:
  `yes`, `no`, or `uncertain`.
- `failure_reason`: one of `wrong_subject`, `multi_instance_union`,
  `under_segmented`, `over_segmented`, `background_region`, `tiny_subject`,
  `no_localizable_subject`, `other`, or blank.
- `review_confidence`: `high`, `medium`, or `low`.

Use the numeric index shown in each tile to update the matching CSV row.
`sample_weight` is the mask-ready population count for the stratum divided by
its sampled count; weighted aggregate metrics must use this value.
"""
    (out_dir / "REVIEW_GUIDE.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="_mask_review/subject_audit_v1")
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--tile-width", type=int, default=420)
    parser.add_argument("--tile-height", type=int, default=320)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pool = _rows()
    selected = _sample(pool, args.n, args.seed)
    pool_counts = Counter(_stratum(row) for row in pool)
    sample_counts = Counter(row["stratum"] for row in selected)
    for index, row in enumerate(selected, 1):
        row["audit_index"] = index
        row["sample_weight"] = (
            pool_counts[row["stratum"]] / sample_counts[row["stratum"]])
    _write_manifest(selected, out_dir)
    _write_review_guide(out_dir)
    sheets = _write_sheets(
        selected, out_dir, args.cols, args.rows,
        args.tile_width, args.tile_height)
    summary = {
        "pool_size": len(pool),
        "sample_size": len(selected),
        "seed": args.seed,
        "strata": dict(Counter(row["stratum"] for row in selected)),
        "pool_strata": dict(pool_counts),
        "sample_weights": {
            name: pool_counts[name] / sample_counts[name]
            for name in sorted(sample_counts)
        },
        "sheets": sheets,
        "review_fields": [
            "subject_correct", "single_instance_correct", "mask_usable",
            "should_skip", "failure_reason", "notes",
        ],
        "audit_scope": "existing VLM main-subject union masks only",
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
