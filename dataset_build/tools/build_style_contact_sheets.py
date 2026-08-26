"""EPR-045 · one annotation contact sheet per LUT for the P1 Style Card pass.

Spec source (the only one): ``docs/确定性召回与 Global - Local 精排优化方案（R4 - G4 - L4）.md``
§3.1 「标注输入：contact sheet（P1）」 — every sheet carries

* the neutral ramp before / after,
* the 3 lightness x 8 hue-band colour chart before / after,
* the full-strength result,
* the normalised-strength result (global ``medium`` centre ΔE00 ≈ 5.5; a LUT that
  cannot reach it is shown at full strength and tagged ``strength_capacity: weak``),
* four fixed scenes: portrait/skin, foliage/sky, architecture/daylight, night/mixed.

and §4.2 — strength is an RGB linear blend (``render.py:147`` ``apply_global_strength``
and its local twin at ``:156``), so one full-LUT render synthesises every alpha and the
bisection never re-runs the LUT.

Numeric conventions are taken from the assets this sheet is meant to be read next to:

* the 8 hue bands and their wheel centres are ``tools/lut_reannotate/hslfeat.py:BANDS``;
  the chart's saturation/lightness levels are that module's ``SAT_LEVELS[1]`` and
  ``LUM_LEVELS``; the ramp ticks are its ``GRAY_LEVELS``.  The English band words are
  the ones already frozen in ``dataset_build/tools/build_caption_v2_pilot.py:BANDS``.
* the normalisation target / tolerance / iteration count are the pre-registered
  ``[normalize]`` block of ``configs/lut_numeric_clusters.epr035.toml``
  (``target_de00 = 5.5``, ``tolerance = 0.05``, ``bisect_iters = 60``).  The
  *calibration content* differs (see ``CALIBRATION_SPEC``), so ``alpha_hat`` here is
  not the EPR-035 alpha.

Nothing under ``dataset_build/agent_loop/`` is imported for its side effects: the LUT
grid loader is the packed bank plus ``dataset_build.lut_io.load_lut``, and the sampler
is the same ``apply_lut_cpu_oracle`` that ``render.CanonicalCpuLutRenderer`` calls.

Visualisation discipline: every panel is a literal sRGB value.  No per-image min-max,
no per-image softmax, no auto-levels anywhere in this file.
"""
from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib


# ------------------------------------------------------------------ pre-registered spec

SHEET_SCHEMA = "lut-contact-sheet-v1"
SPEC_REV = "contact-sheet-v1"

TARGET_DE00 = 5.5
TOLERANCE_DE00 = 0.05
BISECT_ITERS = 60
CALIBRATION_SPEC = "four-scene-4x128px-resize-v1"
CALIB_SIDE = 128  # per scene; 4 * 128 * 128 = 65,536 calibration pixels

BANDS: tuple[tuple[str, float, str], ...] = (
    ("红", 0.0, "red"),
    ("橙", 30.0, "orange"),
    ("黄", 60.0, "yellow"),
    ("绿", 120.0, "green"),
    ("浅绿", 180.0, "aqua"),
    ("蓝", 240.0, "blue"),
    ("紫", 270.0, "purple"),
    ("洋红", 300.0, "magenta"),
)
CHART_SAT = 0.70                       # hslfeat.SAT_LEVELS[1]
CHART_LUM = (0.35, 0.50, 0.65)         # hslfeat.LUM_LEVELS
GRAY_LEVELS = (0.15, 0.30, 0.50, 0.70, 0.85)   # hslfeat.GRAY_LEVELS
RAMP_STEPS = 228

SOURCES_JSONL = Path("/home/bc/data/agent_loop/local-v1/sources_full.jsonl")
AGENT_LOOP_CONFIG = Path("configs/agent_loop.local-v2-b12.toml")
OUT_ROOT = Path("/home/bc/data/scratch/lut_contact_sheets")

# The four fixed scenes.  Rule family: filter `scene` to the mapped word, require /
# forbid whole-word patterns on `subject.description`, then take the smallest
# sha1(source_id).  Same four images for all 4,051 sheets.
SCENE_SLOTS: tuple[dict[str, Any], ...] = (
    {
        "slot": "portrait_skin",
        "label": "portrait / skin",
        "scene": "portrait",
        "require": (r"\b(woman|man|girl|boy|person|face|portrait|model|bride|groom)\b",),
        "exclude": r"\b(night|silhouette|silhouetted|back of|away from)\b",
        "min_mask_area": 0.15,
    },
    {
        "slot": "foliage_sky",
        "label": "foliage / sky",
        "scene": "landscape",
        "require": (
            r"\b(tree|trees|forest|leaves|foliage|grass|jungle|meadow|bush|ferns?|moss|green)\b",
            r"\b(sky|sunset|sunrise|clouds?|horizon)\b",
        ),
        "exclude": r"\b(night|snow|snowy|underwater|winter|fog|foggy|mist|misty|seabed)\b",
        "min_mask_area": 0.0,
    },
    {
        "slot": "architecture_daylight",
        "label": "architecture / daylight",
        "scene": "architecture",
        "require": (
            r"\b(building|church|facade|tower|towers|temple|house|bridge|archway|cathedral|wall)\b",
            r"\b(sunlight|sunlit|daylight|sunny|clear sky|sky)\b",
        ),
        "exclude": r"\b(night|dark|neon|dusk|illuminated|mist|misty)\b",
        "min_mask_area": 0.0,
    },
    {
        "slot": "night_mixed",
        "label": "night / mixed light",
        "scene": "night",
        "require": (r"\b(illuminated|lit|neon|city|lights)\b",),
        "exclude": r"\b(milky way|starry|aurora|moon)\b",
        "min_mask_area": 0.0,
    },
)
SCENE_ORDER: tuple[str, ...] = tuple(spec["slot"] for spec in SCENE_SLOTS)

REQUIRED_PANELS: tuple[str, ...] = (
    "neutral_ramp.before", "neutral_ramp.after_full",
    "hue_chart.before", "hue_chart.after_full",
) + tuple(
    f"scene.{slot}.{kind}"
    for slot in SCENE_ORDER
    for kind in ("before", "full", "normalized")
)

SIDECAR_TOP_KEYS: tuple[str, ...] = (
    "schema", "spec_rev", "preset_id", "preset_name", "lut_path", "lut_format",
    "lut_content_hash", "alpha_hat", "dE00_reached", "dE00_full", "strength_capacity",
    "normalization", "scenes", "panels", "sheet", "params_sha256",
)
SIDECAR_SHEET_KEYS: tuple[str, ...] = ("path", "png_sha256", "png_bytes", "width", "height")
SIDECAR_NORM_KEYS: tuple[str, ...] = (
    "target_de00", "tolerance", "bisect_iters", "calibration", "calibration_pixels",
)

# canvas geometry
MARGIN = 20
CANVAS_W = 1180
HEADER_H = 72
RAMP_STRIP_H = 46
CHART_PATCH = 56
CHART_GAP = 4
CHART_ROW_LABEL_W = 46
SCENE_BOX = 340
SCENE_GAP = 10
CAPTION_H = 18
SECTION_GAP = 18
BG = (255, 255, 255)
INK = (18, 18, 18)
RULE = (176, 176, 176)


# --------------------------------------------------------------- runtime assertions

_ASSERTIONS: Counter = Counter()


def assertion_counts() -> dict[str, int]:
    return dict(_ASSERTIONS)


def reset_assertions() -> None:
    _ASSERTIONS.clear()


def bump_assertions(counts: Mapping[str, int]) -> None:
    """Fold a worker process' assertion counters into this process' totals."""
    for name, value in counts.items():
        _ASSERTIONS[name] += int(value)


def require_assertions(*names: str) -> None:
    """Fail if a pre-registered assertion was defined but never actually called."""
    missing = [name for name in names if _ASSERTIONS[name] == 0]
    if missing:
        raise SystemExit(f"pre-registered runtime assertion never ran: {missing}")


def assert_sheet_panels(panels: Sequence[Mapping[str, Any]],
                        canvas: tuple[int, int]) -> dict[str, int]:
    """Panel completeness of one composed sheet (§3.1 content contract)."""
    _ASSERTIONS["assert_sheet_panels"] += 1
    names = [str(panel.get("name") or "") for panel in panels]
    if len(names) != len(set(names)):
        raise SystemExit(f"duplicate panel name: {sorted(n for n in names if names.count(n) > 1)}")
    missing = [name for name in REQUIRED_PANELS if name not in set(names)]
    if missing:
        raise SystemExit(f"contact sheet is missing required panels: {missing}")
    extra = [name for name in names if name not in set(REQUIRED_PANELS)]
    if extra:
        raise SystemExit(f"contact sheet declares unknown panels: {extra}")
    width, height = canvas
    for panel in panels:
        box = list(panel.get("box") or [])
        if len(box) != 4:
            raise SystemExit(f"panel {panel.get('name')} has no 4-tuple box")
        x0, y0, x1, y1 = (int(v) for v in box)
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise SystemExit(f"panel {panel.get('name')} box {box} outside canvas {canvas}")
    return {"panels": len(names), "required": len(REQUIRED_PANELS)}


def validate_sidecar(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Sidecar schema check (called for every sheet, on write and on resume)."""
    _ASSERTIONS["validate_sidecar"] += 1
    missing = [key for key in SIDECAR_TOP_KEYS if key not in payload]
    if missing:
        raise SystemExit(f"sidecar missing keys: {missing}")
    if payload["schema"] != SHEET_SCHEMA:
        raise SystemExit(f"sidecar schema {payload['schema']!r} != {SHEET_SCHEMA!r}")
    if payload["spec_rev"] != SPEC_REV:
        raise SystemExit(f"sidecar spec_rev {payload['spec_rev']!r} != {SPEC_REV!r}")
    if payload["strength_capacity"] not in {"normal", "weak"}:
        raise SystemExit(f"bad strength_capacity {payload['strength_capacity']!r}")
    alpha = float(payload["alpha_hat"])
    if not 0.0 <= alpha <= 1.0:
        raise SystemExit(f"alpha_hat {alpha} outside [0, 1]")
    reached = float(payload["dE00_reached"])
    if payload["strength_capacity"] == "weak":
        if alpha != 1.0:
            raise SystemExit("weak sheet must be rendered at alpha 1.0")
        if reached >= TARGET_DE00 - TOLERANCE_DE00:
            raise SystemExit(f"weak sheet reached {reached} >= target-tolerance")
    else:
        if abs(reached - TARGET_DE00) > TOLERANCE_DE00:
            raise SystemExit(
                f"normal sheet reached {reached}, |Δ| > tolerance {TOLERANCE_DE00}"
            )
    norm_missing = [key for key in SIDECAR_NORM_KEYS if key not in payload["normalization"]]
    if norm_missing:
        raise SystemExit(f"sidecar normalization missing keys: {norm_missing}")
    sheet_missing = [key for key in SIDECAR_SHEET_KEYS if key not in payload["sheet"]]
    if sheet_missing:
        raise SystemExit(f"sidecar sheet missing keys: {sheet_missing}")
    scenes = list(payload["scenes"])
    if tuple(str(row["slot"]) for row in scenes) != SCENE_ORDER:
        raise SystemExit(f"sidecar scene order {[r.get('slot') for r in scenes]} != {SCENE_ORDER}")
    assert_sheet_panels(payload["panels"], (int(payload["sheet"]["width"]),
                                            int(payload["sheet"]["height"])))
    return {"panels": len(payload["panels"]), "scenes": len(scenes)}


# ------------------------------------------------------------------------- utilities


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    if path.is_file():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()  # pragma: no cover


# ---------------------------------------------------------------------- LUT plumbing


def apply_lut(image: np.ndarray, grid: np.ndarray,
              dmin: np.ndarray | None, dmax: np.ndarray | None) -> np.ndarray:
    """The exact sampler ``render.CanonicalCpuLutRenderer.render_full`` uses."""
    from dataset_build.src.construct.rendering import apply_lut_cpu_oracle

    return apply_lut_cpu_oracle(image, grid, domain_min=dmin, domain_max=dmax).astype(np.float32)


def apply_global_strength(before: np.ndarray, full_edit: np.ndarray,
                          strength: float) -> np.ndarray:
    """``render.py:147`` verbatim: strength is an RGB linear blend, nothing else."""
    value = float(np.clip(strength, 0.0, 1.0))
    return (before * (1.0 - value) + full_edit * value).astype(np.float32)


class LutBank:
    """Packed ``luts.npz`` first, ``.cube`` / ``.3dl`` parse as the fallback.

    Deliberately cache-free: 4,051 x 33^3 x 3 float32 grids would be 1.7 GB per worker.
    """

    def __init__(self, bank_dir: Path):
        self.bank_dir = Path(bank_dir)
        self._packed = None
        self._by_path: dict[str, tuple[str, np.ndarray, np.ndarray]] = {}
        meta_path = self.bank_dir / "luts_meta.json"
        packed_path = self.bank_dir / "luts.npz"
        if meta_path.is_file() and packed_path.is_file():
            with meta_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            self._packed = np.load(packed_path, allow_pickle=False)
            for preset_id, row in metadata.items():
                real = os.path.realpath(str(row.get("path") or ""))
                self._by_path[real] = (
                    preset_id,
                    np.asarray(row.get("dmin", (0, 0, 0)), dtype=np.float32),
                    np.asarray(row.get("dmax", (1, 1, 1)), dtype=np.float32),
                )

    def load(self, path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        from dataset_build.lut_io import load_lut

        real = os.path.realpath(path)
        packed = self._by_path.get(real)
        if packed is not None and self._packed is not None:
            preset_id, dmin, dmax = packed
            return np.asarray(self._packed[preset_id], dtype=np.float32), dmin, dmax
        return load_lut(real)


def bank_dir_of(agent_loop_config: Path) -> Path:
    with Path(agent_loop_config).open("rb") as handle:
        loop = tomllib.load(handle)
    databuild = (Path(agent_loop_config).parent / str(loop["agent_loop"]["databuild_config"]))
    with databuild.resolve().open("rb") as handle:
        build = tomllib.load(handle)
    return Path(str((build.get("presets") or {}).get("bank_dir") or ""))


def load_catalog(agent_loop_config: Path) -> list[dict[str, Any]]:
    """`preset_id`-sorted LUT rows that exist in the bank *and* carry a closed-v1 row."""
    with Path(agent_loop_config).open("rb") as handle:
        loop = tomllib.load(handle)
    annotations = Path(str(loop["catalog"]["annotations"])).expanduser()
    names: dict[str, str] = {}
    with annotations.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("ok") is not True:
                continue
            preset_id = str(row.get("preset_id") or row.get("key") or "")
            names[preset_id] = str(row.get("name") or "")
    features = bank_dir_of(agent_loop_config) / "features.jsonl"
    records: list[dict[str, Any]] = []
    with features.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("kind") or "") != "lut":
                continue
            preset_id = str(row.get("preset_id") or "")
            path = str(row.get("path") or "")
            if preset_id not in names or not Path(path).is_file():
                continue
            records.append({
                "preset_id": preset_id,
                "name": names[preset_id],
                "path": path,
                "fmt": str(row.get("fmt") or ""),
                "content_hash": str(row.get("preset_content_hash") or ""),
            })
    records.sort(key=lambda row: row["preset_id"])
    return records


# ------------------------------------------------------------------- scene selection


def select_scenes(sources_jsonl: Path,
                  slots: Sequence[Mapping[str, Any]] = SCENE_SLOTS) -> list[dict[str, Any]]:
    """Deterministic: scene word + description gates, then smallest sha1(source_id)."""
    rows: list[dict[str, Any]] = []
    with Path(sources_jsonl).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    chosen: list[dict[str, Any]] = []
    for spec in slots:
        pool: list[dict[str, Any]] = []
        for row in rows:
            if str(row.get("scene") or "") != spec["scene"]:
                continue
            subject = row.get("subject") or {}
            text = str(subject.get("description") or "").lower()
            if not all(re.search(pattern, text) for pattern in spec["require"]):
                continue
            if spec["exclude"] and re.search(spec["exclude"], text):
                continue
            if float(subject.get("mask_area") or 0.0) < float(spec["min_mask_area"]):
                continue
            if not Path(str(row.get("source_path") or "")).is_file():
                continue
            pool.append(row)
        if not pool:
            raise SystemExit(f"scene slot {spec['slot']} has an empty pool")
        pool.sort(key=lambda row: hashlib.sha1(str(row["source_id"]).encode("utf-8")).hexdigest())
        pick = pool[0]
        chosen.append({
            "slot": spec["slot"],
            "label": spec["label"],
            "scene": spec["scene"],
            "source_id": str(pick["source_id"]),
            "source_path": str(pick["source_path"]),
            "pool_size": len(pool),
            "description": str((pick.get("subject") or {}).get("description") or ""),
        })
    return chosen


def load_scene_images(scenes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Panel-resolution RGB plus the fixed calibration tile for each scene."""
    display: list[np.ndarray] = []
    calib: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []
    for row in scenes:
        path = Path(str(row["source_path"]))
        with Image.open(path) as handle:
            rgb = handle.convert("RGB")
            panel = rgb.copy()
            panel.thumbnail((SCENE_BOX, SCENE_BOX), Image.Resampling.LANCZOS)
            tile = rgb.resize((CALIB_SIDE, CALIB_SIDE), Image.Resampling.LANCZOS)
        display.append(np.asarray(panel, dtype=np.float32) / 255.0)
        calib.append(np.asarray(tile, dtype=np.float32) / 255.0)
        meta.append({
            "slot": str(row["slot"]),
            "label": str(row["label"]),
            "scene": str(row["scene"]),
            "source_id": str(row["source_id"]),
            "source_path": str(path),
            "source_sha256": sha256_file(path),
        })
    return {
        "display": display,
        "calibration": np.concatenate([tile.reshape(-1, 3) for tile in calib], axis=0),
        "meta": meta,
    }


# ------------------------------------------------------------------- normalisation


def mean_de00(before_lab: np.ndarray, after_rgb: np.ndarray) -> float:
    from skimage.color import deltaE_ciede2000, rgb2lab

    after_lab = rgb2lab(np.clip(after_rgb, 0.0, 1.0).astype(np.float64).reshape(1, -1, 3))
    return float(deltaE_ciede2000(before_lab, after_lab.reshape(-1, 3)).mean())


def solve_alpha(before_rgb: np.ndarray, full_rgb: np.ndarray,
                target: float = TARGET_DE00, tolerance: float = TOLERANCE_DE00,
                iterations: int = BISECT_ITERS) -> dict[str, Any]:
    """Bisect the RGB blend alpha onto ``mean ΔE00 == target`` on the calibration tile."""
    from skimage.color import rgb2lab

    before = np.asarray(before_rgb, dtype=np.float32).reshape(-1, 3)
    full = np.asarray(full_rgb, dtype=np.float32).reshape(-1, 3)
    before_lab = rgb2lab(before.astype(np.float64).reshape(1, -1, 3)).reshape(-1, 3)

    def de_at(alpha: float) -> float:
        return mean_de00(before_lab, apply_global_strength(before, full, alpha))

    de_full = de_at(1.0)
    if de_full < target - tolerance:
        return {
            "alpha": 1.0, "achieved": de_full, "de_full": de_full,
            "weak": True, "iterations": 0,
        }
    low, high = 0.0, 1.0
    used = 0
    for _ in range(int(iterations)):
        used += 1
        mid = (low + high) / 2.0
        value = de_at(mid)
        if value < target:
            low = mid
        else:
            high = mid
        if abs(value - target) <= tolerance:
            return {
                "alpha": mid, "achieved": value, "de_full": de_full,
                "weak": False, "iterations": used,
            }
    alpha = (low + high) / 2.0
    return {
        "alpha": alpha, "achieved": de_at(alpha), "de_full": de_full,
        "weak": False, "iterations": used,
    }


# ------------------------------------------------------------------------ chart data


def ramp_input() -> np.ndarray:
    """`RAMP_STEPS` neutral patches from 0 to 1, shape (1, N, 3)."""
    values = np.linspace(0.0, 1.0, RAMP_STEPS, dtype=np.float32)
    return np.repeat(values[None, :, None], 3, axis=2)


def chart_input() -> np.ndarray:
    """3 lightness x 8 hue bands at ``CHART_SAT``, shape (3, 8, 3)."""
    rows = []
    for lum in CHART_LUM:
        row = []
        for _cn, centre, _en in BANDS:
            row.append(list(colorsys.hls_to_rgb(centre / 360.0, lum, CHART_SAT)))
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def to_image(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(array * 255.0 + 0.5, 0, 255).astype(np.uint8), "RGB")


# ---------------------------------------------------------------------- composition


def _paste_fit(canvas: Image.Image, array: np.ndarray, x: int, y: int,
               box: int) -> tuple[int, int, int, int]:
    """Letterbox one panel-resolution render into a ``box`` x ``box`` cell."""
    image = to_image(array)
    if max(image.size) != box:
        image = image.copy()
        image.thumbnail((box, box), Image.Resampling.LANCZOS)
    off_x = x + (box - image.width) // 2
    off_y = y
    canvas.paste(image, (off_x, off_y))
    return (off_x, off_y, off_x + image.width, off_y + image.height)


def compose_sheet(record: Mapping[str, Any], scenes: Mapping[str, Any],
                  ramp_pair: tuple[np.ndarray, np.ndarray],
                  chart_pair: tuple[np.ndarray, np.ndarray],
                  scene_renders: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
                  solution: Mapping[str, Any]) -> tuple[Image.Image, list[dict[str, Any]]]:
    small = _font(12)
    tiny = _font(11)
    head = _font(19, bold=True)
    sub = _font(13)

    chart_block_w = CHART_ROW_LABEL_W + 8 * CHART_PATCH + 7 * CHART_GAP
    ramp_h = CAPTION_H + RAMP_STRIP_H + 4 + RAMP_STRIP_H + 16
    chart_h = CAPTION_H + 16 + 3 * CHART_PATCH + 2 * CHART_GAP
    scene_h = 2 * CAPTION_H + 4 * (CAPTION_H + SCENE_BOX + SCENE_GAP)
    height = (MARGIN + HEADER_H + ramp_h + SECTION_GAP + chart_h
              + SECTION_GAP + scene_h + MARGIN)
    canvas = Image.new("RGB", (CANVAS_W, height), BG)
    draw = ImageDraw.Draw(canvas)
    panels: list[dict[str, Any]] = []

    capacity = "weak" if solution["weak"] else "normal"
    norm_caption = (
        f"full strength (strength_capacity: weak, dE00={solution['de_full']:.2f})"
        if solution["weak"] else
        f"normalized alpha={solution['alpha']:.4f} (dE00={solution['achieved']:.2f})"
    )

    # The sheet carries no style word of any kind: the existing Chinese `name` /
    # `style_major` stay in the sidecar so they cannot prime the Style Card annotator.
    y = MARGIN
    draw.text((MARGIN, y), f"{record['preset_id']}", font=head, fill=INK)
    draw.text((MARGIN, y + 26),
              f"strength_capacity={capacity}   alpha_hat={solution['alpha']:.4f}   "
              f"dE00_reached={solution['achieved']:.3f}   dE00_full={solution['de_full']:.3f}   "
              f"target={TARGET_DE00} +-{TOLERANCE_DE00}   calibration={CALIBRATION_SPEC}",
              font=sub, fill=INK)
    draw.text((MARGIN, y + 46),
              f"schema={SHEET_SCHEMA}   spec_rev={SPEC_REV}   lut={Path(record['path']).name}",
              font=tiny, fill=INK)
    y += HEADER_H
    draw.line([(MARGIN, y - 6), (CANVAS_W - MARGIN, y - 6)], fill=RULE, width=1)

    # --- neutral ramp -----------------------------------------------------------
    ramp_w = CANVAS_W - 2 * MARGIN
    draw.text((MARGIN, y), "Neutral ramp  (top: before   bottom: after, full strength)",
              font=small, fill=INK)
    y += CAPTION_H
    for name, array in (("neutral_ramp.before", ramp_pair[0]),
                        ("neutral_ramp.after_full", ramp_pair[1])):
        strip = to_image(array).resize((ramp_w, RAMP_STRIP_H), Image.Resampling.NEAREST)
        canvas.paste(strip, (MARGIN, y))
        draw.rectangle([MARGIN, y, MARGIN + ramp_w - 1, y + RAMP_STRIP_H - 1],
                       outline=RULE, width=1)
        panels.append({"name": name, "box": [MARGIN, y, MARGIN + ramp_w, y + RAMP_STRIP_H]})
        y += RAMP_STRIP_H + 4
    y -= 4
    for level in GRAY_LEVELS:
        x = MARGIN + int(round(level * (ramp_w - 1)))
        draw.line([(x, y), (x, y + 4)], fill=INK, width=1)
        draw.text((x - 9, y + 5), f"{level:.2f}", font=tiny, fill=INK)
    y += 16 + SECTION_GAP

    # --- 3 lightness x 8 hue band chart ----------------------------------------
    draw.text((MARGIN, y), "Hue-band chart  (3 lightness x 8 bands, HSL sat "
                           f"{CHART_SAT:.2f})   left: before   right: after, full strength",
              font=small, fill=INK)
    y += CAPTION_H
    chart_left = MARGIN
    chart_right = MARGIN + chart_block_w + 40
    for origin, name, array in ((chart_left, "hue_chart.before", chart_pair[0]),
                                (chart_right, "hue_chart.after_full", chart_pair[1])):
        px = origin + CHART_ROW_LABEL_W
        for index, (_cn, _centre, english) in enumerate(BANDS):
            x = px + index * (CHART_PATCH + CHART_GAP)
            draw.text((x + 2, y + 2), english, font=tiny, fill=INK)
        top = y + 16
        for row_index, lum in enumerate(CHART_LUM):
            cy = top + row_index * (CHART_PATCH + CHART_GAP)
            draw.text((origin, cy + CHART_PATCH // 2 - 6), f"L{int(lum * 100)}",
                      font=tiny, fill=INK)
            for col_index in range(len(BANDS)):
                cx = px + col_index * (CHART_PATCH + CHART_GAP)
                colour = tuple(int(v) for v in np.clip(
                    array[row_index, col_index] * 255.0 + 0.5, 0, 255).astype(np.uint8))
                draw.rectangle([cx, cy, cx + CHART_PATCH - 1, cy + CHART_PATCH - 1], fill=colour)
                draw.rectangle([cx, cy, cx + CHART_PATCH - 1, cy + CHART_PATCH - 1],
                               outline=RULE, width=1)
        block_w = CHART_ROW_LABEL_W + 8 * CHART_PATCH + 7 * CHART_GAP
        block_h = 16 + 3 * CHART_PATCH + 2 * CHART_GAP
        panels.append({"name": name, "box": [origin, y, origin + block_w, y + block_h]})
    y += 16 + 3 * CHART_PATCH + 2 * CHART_GAP + SECTION_GAP

    # --- four fixed scenes ------------------------------------------------------
    scene_left = (CANVAS_W - (3 * SCENE_BOX + 2 * SCENE_GAP)) // 2
    columns = ("before", "full strength", norm_caption)
    draw.text((MARGIN, y), "Fixed scenes", font=small, fill=INK)
    y += CAPTION_H
    for column_index, title in enumerate(columns):
        draw.text((scene_left + column_index * (SCENE_BOX + SCENE_GAP), y), title,
                  font=tiny, fill=INK)
    y += CAPTION_H
    for slot_index, meta in enumerate(scenes["meta"]):
        before, full, normalized = scene_renders[slot_index]
        draw.text((scene_left, y), f"{meta['label']}   [{meta['source_id']}]",
                  font=small, fill=INK)
        row_top = y + CAPTION_H
        for column_index, (kind, array) in enumerate(
            (("before", before), ("full", full), ("normalized", normalized))
        ):
            x = scene_left + column_index * (SCENE_BOX + SCENE_GAP)
            box = _paste_fit(canvas, array, x, row_top, SCENE_BOX)
            draw.rectangle(list(box), outline=RULE, width=1)
            panels.append({"name": f"scene.{meta['slot']}.{kind}", "box": list(box)})
        y = row_top + SCENE_BOX + SCENE_GAP
    return canvas, panels


# ---------------------------------------------------------------------- one preset


def scene_rules_payload(slots: Sequence[Mapping[str, Any]] = SCENE_SLOTS) -> list[dict[str, Any]]:
    """`SCENE_SLOTS` verbatim, so ``params_sha256`` covers the selection rule itself.

    Without this the digest only pinned the four images the rule happened to pick: editing
    a require/exclude pattern that resolves to the same four images left the digest — and
    therefore the resume gate — unchanged.
    """
    return [
        {
            "slot": str(spec["slot"]),
            "label": str(spec["label"]),
            "scene": str(spec["scene"]),
            "require": [str(pattern) for pattern in spec["require"]],
            "exclude": str(spec["exclude"] or ""),
            "min_mask_area": float(spec["min_mask_area"]),
        }
        for spec in slots
    ]


def params_payload(scenes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "scene_rules": scene_rules_payload(),
        "scene_pick_order": "sha1(source_id) hex, ascending, first row wins",
        "schema": SHEET_SCHEMA,
        "spec_rev": SPEC_REV,
        "target_de00": TARGET_DE00,
        "tolerance": TOLERANCE_DE00,
        "bisect_iters": BISECT_ITERS,
        "calibration": CALIBRATION_SPEC,
        "calib_side": CALIB_SIDE,
        "bands": [[cn, centre, en] for cn, centre, en in BANDS],
        "chart_sat": CHART_SAT,
        "chart_lum": list(CHART_LUM),
        "gray_levels": list(GRAY_LEVELS),
        "ramp_steps": RAMP_STEPS,
        "geometry": {
            "canvas_w": CANVAS_W, "margin": MARGIN, "header_h": HEADER_H,
            "ramp_strip_h": RAMP_STRIP_H, "chart_patch": CHART_PATCH,
            "chart_gap": CHART_GAP, "scene_box": SCENE_BOX, "scene_gap": SCENE_GAP,
        },
        "scenes": [
            {"slot": row["slot"], "source_id": row["source_id"],
             "source_sha256": row["source_sha256"]}
            for row in scenes
        ],
    }


def build_one(record: Mapping[str, Any], bank: LutBank, scenes: Mapping[str, Any],
              params_sha: str, out_root: Path) -> dict[str, Any]:
    grid, dmin, dmax = bank.load(Path(record["path"]))

    ramp_before = ramp_input()
    ramp_after = apply_lut(ramp_before, grid, dmin, dmax)
    chart_before = chart_input()
    chart_after = apply_lut(chart_before, grid, dmin, dmax)

    calib_before = scenes["calibration"]
    calib_full = apply_lut(calib_before.reshape(1, -1, 3), grid, dmin, dmax).reshape(-1, 3)
    solution = solve_alpha(calib_before, calib_full)

    scene_renders: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for before in scenes["display"]:
        full = apply_lut(before, grid, dmin, dmax)
        normalized = apply_global_strength(before, full, float(solution["alpha"]))
        scene_renders.append((before, full, normalized))

    canvas, panels = compose_sheet(
        record, scenes, (ramp_before, ramp_after), (chart_before, chart_after),
        scene_renders, solution,
    )
    assert_sheet_panels(panels, canvas.size)

    sheets_dir = out_root / "sheets"
    sidecar_dir = out_root / "sidecar"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    png_path = sheets_dir / f"{record['preset_id']}.png"
    sidecar_path = sidecar_dir / f"{record['preset_id']}.json"
    tmp_path = png_path.with_suffix(".png.tmp")
    tmp_sidecar = sidecar_path.with_suffix(".json.tmp")
    # Both files are staged as `.tmp`, the sidecar is validated while the PNG is still a
    # `.tmp`, and only then are the two renamed: a rejected sidecar leaves no orphan PNG.
    canvas.save(tmp_path, format="PNG", optimize=False, compress_level=6)

    payload = {
        "schema": SHEET_SCHEMA,
        "spec_rev": SPEC_REV,
        "preset_id": str(record["preset_id"]),
        "preset_name": str(record.get("name") or ""),
        "lut_path": str(record["path"]),
        "lut_format": str(record.get("fmt") or ""),
        "lut_content_hash": str(record.get("content_hash") or ""),
        "alpha_hat": round(float(solution["alpha"]), 6),
        "dE00_reached": round(float(solution["achieved"]), 4),
        "dE00_full": round(float(solution["de_full"]), 4),
        "strength_capacity": "weak" if solution["weak"] else "normal",
        "normalization": {
            "target_de00": TARGET_DE00,
            "tolerance": TOLERANCE_DE00,
            "bisect_iters": BISECT_ITERS,
            "iterations_used": int(solution["iterations"]),
            "calibration": CALIBRATION_SPEC,
            "calibration_pixels": int(calib_before.shape[0]),
        },
        "scenes": list(scenes["meta"]),
        "panels": panels,
        "sheet": {
            "path": str(png_path),
            "png_sha256": sha256_file(tmp_path),
            "png_bytes": int(tmp_path.stat().st_size),
            "width": int(canvas.size[0]),
            "height": int(canvas.size[1]),
        },
        "params_sha256": params_sha,
    }
    try:
        validate_sidecar(payload)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    tmp_sidecar.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    os.replace(tmp_path, png_path)
    os.replace(tmp_sidecar, sidecar_path)
    return payload


def already_done(preset_id: str, params_sha: str, out_root: Path) -> dict[str, Any] | None:
    """Resume gate: PNG present, sidecar parses, schema valid, params + png sha match."""
    sidecar_path = out_root / "sidecar" / f"{preset_id}.json"
    png_path = out_root / "sheets" / f"{preset_id}.png"
    if not sidecar_path.is_file() or not png_path.is_file():
        return None
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        validate_sidecar(payload)
    except (Exception, SystemExit):
        # `validate_sidecar` reports schema/spec drift with SystemExit (a BaseException):
        # a stale sidecar must send this one preset back to the render queue, never abort
        # the whole batch.
        return None
    if payload.get("params_sha256") != params_sha:
        return None
    if int(payload["sheet"]["png_bytes"]) != int(png_path.stat().st_size):
        return None
    if payload["sheet"]["png_sha256"] != sha256_file(png_path):
        return None
    return payload


# --------------------------------------------------------------------- worker glue

_WORKER: dict[str, Any] = {}


def _worker_init(bank_dir: str, scene_rows: list[dict[str, Any]], params_sha: str,
                 out_root: str) -> None:
    reset_assertions()
    _WORKER["bank"] = LutBank(Path(bank_dir))
    _WORKER["scenes"] = load_scene_images(scene_rows)
    _WORKER["params_sha"] = params_sha
    _WORKER["out_root"] = Path(out_root)


def _worker_run(record: dict[str, Any]) -> dict[str, Any]:
    # the returned counters are this record's delta, never the worker's running total
    opening = assertion_counts()
    started = time.perf_counter()
    try:
        payload = build_one(record, _WORKER["bank"], _WORKER["scenes"],
                            _WORKER["params_sha"], _WORKER["out_root"])
        ok, extra = True, {
            "alpha_hat": payload["alpha_hat"],
            "dE00_reached": payload["dE00_reached"],
            "dE00_full": payload["dE00_full"],
            "strength_capacity": payload["strength_capacity"],
            "png_bytes": payload["sheet"]["png_bytes"],
        }
    except Exception as exc:  # result-before-optional-stage: one bad LUT never kills the run
        ok, extra = False, {"error": f"{type(exc).__name__}: {exc}"}
    closing = assertion_counts()
    delta = {name: closing[name] - opening.get(name, 0) for name in closing}
    return {
        "preset_id": record["preset_id"], "ok": ok,
        "seconds": time.perf_counter() - started,
        "assertions": {name: value for name, value in delta.items() if value},
        **extra,
    }


# --------------------------------------------------------------------------- driver


def write_progress(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=AGENT_LOOP_CONFIG)
    parser.add_argument("--sources", type=Path, default=SOURCES_JSONL)
    parser.add_argument("--out", type=Path, default=OUT_ROOT)
    parser.add_argument("--limit", type=int, default=0,
                        help="first N preset_ids in lexicographic order (0 = all)")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--force", action="store_true", help="ignore the resume gate")
    args = parser.parse_args(argv)

    reset_assertions()
    records = load_catalog(args.config)
    scenes = select_scenes(args.sources)
    scene_meta = load_scene_images(scenes)["meta"]
    params = params_payload(scene_meta)
    params_sha = sha256_text(canonical_json(params))
    if args.limit:
        records = records[: args.limit]

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "sheets").mkdir(exist_ok=True)
    (out_root / "sidecar").mkdir(exist_ok=True)
    write_progress(out_root / "params.json", {"params_sha256": params_sha, **params,
                                              "scene_detail": scenes})

    pending: list[dict[str, Any]] = []
    skipped = 0
    for record in records:
        if not args.force and already_done(record["preset_id"], params_sha, out_root):
            skipped += 1
            continue
        pending.append(record)
    print(f"[plan] catalog={len(records)} pending={len(pending)} resumed={skipped} "
          f"params_sha256={params_sha[:16]} workers={args.workers}", flush=True)

    progress_path = out_root / "progress.json"
    started = time.time()
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    weak = 0
    total_bytes = 0

    def absorb(result: Mapping[str, Any], fold_assertions: bool = True) -> None:
        nonlocal weak, total_bytes
        # in-process (workers<=1) runs already bumped this process' counters
        if fold_assertions:
            bump_assertions(result.get("assertions") or {})
        if result.get("ok"):
            results.append(dict(result))
            weak += int(result["strength_capacity"] == "weak")
            total_bytes += int(result["png_bytes"])
        else:
            failures.append(dict(result))

    def snapshot(done: bool) -> None:
        elapsed = time.time() - started
        write_progress(progress_path, {
            "schema": "lut-contact-sheet-progress-v1",
            "params_sha256": params_sha,
            "catalog": len(records),
            "resumed": skipped,
            "written": len(results),
            "failed": len(failures),
            "weak": weak,
            "total_png_bytes": total_bytes,
            "elapsed_seconds": round(elapsed, 2),
            "seconds_per_sheet": round(elapsed / max(len(results), 1), 4),
            "assertions": assertion_counts(),
            "failures": failures[:50],
            "done": done,
        })

    if pending:
        if args.workers <= 1:
            _worker_init(str(bank_dir_of(args.config)), scenes, params_sha, str(out_root))
            for index, record in enumerate(pending, 1):
                absorb(_worker_run(record), fold_assertions=False)
                if index % 25 == 0 or index == len(pending):
                    snapshot(False)
                    print(f"[run] {index}/{len(pending)} weak={weak} failed={len(failures)}",
                          flush=True)
        else:
            executor = ProcessPoolExecutor(
                max_workers=int(args.workers), initializer=_worker_init,
                initargs=(str(bank_dir_of(args.config)), scenes, params_sha, str(out_root)),
            )
            with executor:
                futures = [executor.submit(_worker_run, record) for record in pending]
                for index, future in enumerate(as_completed(futures), 1):
                    absorb(future.result())
                    if index % 50 == 0 or index == len(futures):
                        snapshot(False)
                        print(f"[run] {index}/{len(futures)} weak={weak} "
                              f"failed={len(failures)}", flush=True)
    # results land before any optional stage
    snapshot(True)
    require_assertions("assert_sheet_panels", "validate_sidecar")
    print(f"[done] written={len(results)} resumed={skipped} failed={len(failures)} "
          f"weak={weak} bytes={total_bytes} assertions={assertion_counts()}", flush=True)
    if results:
        alphas = sorted(row["alpha_hat"] for row in results)
        print(f"[alpha] min={alphas[0]:.4f} p50={alphas[len(alphas) // 2]:.4f} "
              f"max={alphas[-1]:.4f}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
