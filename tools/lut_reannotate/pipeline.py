#!/usr/bin/env /home/bc/miniconda3/bin/python3
"""Re-annotate the 4,051 bank LUTs from scratch (TOOL-LutReannot-1).

The previous annotation pass ran *before* the red/blue axis-order fix, so every
name, caption and per-probe line in ``vlm_names.jsonl`` describes a picture that
was never rendered.  All of it is void.  This tool rebuilds the evidence and the
labels:

    probes    fetch and cache the six fixed probe source images (long edge 768)
    render    apply every LUT to every probe on CPU, JPEG q90, resumable
    annotate  6 before/after pairs + an HSL 8-band response table -> relay VLM
    pack      LUT bodies + annotations.jsonl + MANIFEST.json into one zip

Two things are load-bearing and easy to get wrong, so both are asserted rather
than assumed:

1. **Axis order.**  ``luts.npz`` stores grids as ``[b][g][r]`` with RGB values.
   ``render --selfcheck`` re-renders sampled outputs against
   ``dataset_build.src.construct.rendering.apply_lut_cpu_oracle`` and fails the
   run on any disagreement above 1e-5.  That is the exact bug this whole
   re-annotation exists to undo; it does not get to happen twice.
2. **Model substitution.**  The relay has been measured answering a request for
   one model with another (~1/3 of calls on provider-b, 2026-07-28).  Every call
   pins ``response.model`` and a mismatch fails the record into the retry path,
   so a batch has one known annotator rather than a silent mixture.

GPU is deliberately untouched: both cards are at 98% on training, and trilinear
interpolation on a 768 px image is a numpy gather.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import queue
import random
import sys
import threading
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageOps

REPO = Path("/home/bc/VeraRetouch")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hslfeat  # noqa: E402  (sibling module, importable however this file is invoked)

BANK = Path("/home/bc/data/datasets/vera_directionA_1M/preset_bank_full")
FEATURES = BANK / "features.jsonl"
LUTS_NPZ = BANK / "luts.npz"
LUTS_META = BANK / "luts_meta.json"
WORK = Path(os.environ.get("LUT_REANNOT_WORK", "/home/bc/data/scratch/lut_reannotate"))
CONFIG_TOML = REPO / "databuild.prod-l6-local17k-20260801.toml"

# User instruction 2026-08-11 overrides the TOML's ``external_model``: the
# annotator is terra, not luna, and the pin is checked against this exact name.
MODEL = "gpt-5.6-terra"
REASONING_EFFORT = "low"
MAX_OUTPUT_TOKENS = 6000
LANE_ID = "provider-c-lane-1"

LONG_EDGE = 768
JPEG_QUALITY = 90
PROMPT_REV = "lutreannot-v1"
RENDER_REV = "cpu-trilinear-le768-q90-v1"

# Recovered 2026-08-11 from the axis-fix session transcript that wrote
# ``probe_before.json``; corroborated slot-by-slot against that file's mean
# chroma (see NOTES.md §2).  Slot order is the bank's own probe order, which is
# also the order ``features.jsonl:lab_vec`` is laid out in.
PROBES: tuple[dict[str, Any], ...] = (
    {"slot": 1, "name": "red", "cn": "红", "desc": "红/暖色主导",
     "asset_id": "src_308bb19e31eeb228",
     "path": "/home/bc/datasets/MMArt-PPR10k/global/790_2/before.jpg"},
    {"slot": 2, "name": "yellow", "cn": "黄", "desc": "黄色主导",
     "asset_id": "src_dbafb5a380332b8a",
     "path": "/home/bc/data/datasets/_scratch/TAD66K/38974475@N0530442571530.jpg"},
    {"slot": 3, "name": "green", "cn": "绿", "desc": "绿色主导",
     "asset_id": "src_fe5abed17e617006",
     "path": "/home/bc/data/datasets/_scratch/TAD66K/emanuelezallocco34195304370.jpg"},
    {"slot": 4, "name": "blue", "cn": "蓝", "desc": "蓝色主导",
     "asset_id": "src_9bb8c3b8ec140e3a",
     "path": "/home/bc/data/datasets/_scratch/TAD66K/michelblanchette11450465554.jpg"},
    {"slot": 5, "name": "skin", "cn": "肤色", "desc": "肤色/人脸",
     "asset_id": "src_8e1419bab64e14ad",
     "path": "/home/bc/data/datasets/presets_sources/quandian/quandian_000870.jpg"},
    {"slot": 6, "name": "neutral", "cn": "中性", "desc": "中性/低饱和",
     "asset_id": "src_48fda67912a38548",
     "path": "/home/bc/data/datasets/_scratch/TAD66K/oscarplaza16256301813.jpg"},
)
PROBE_CN = tuple(p["cn"] for p in PROBES)

TAXONOMY_HINT = (
    "低饱褪彩 / 复古胶片 / 暖调复古 / 青绿胶片 / 品红冷调 / 黄绿暖调 / "
    "黑白去色 / 暖调高亮 / 青蓝清冷 / 青橙暗调"
)

SYSTEM_PROMPT = (
    "你是资深调色师。任务: 给一个 LUT 预设命名并描述它的功能——把画面变成什么风格的 look, "
    "与具体图像内容无关。\n"
    "你会收到两类证据:\n"
    "(1) 6 组固定探针的 before/after 真实渲染图, 顺序固定为 红/黄/绿/蓝/肤色/中性, "
    "覆盖四个高饱和主色相 + 人像肤色 + 中性低饱和, 所以这个预设对各色相、肤色、明暗的处理都会暴露出来;\n"
    "(2) 一张【8 色相带响应表】+【中性灰响应表】, 由程序直接在 LUT 网格上采样算出, 是客观测量, 不可推翻。\n"
    "硬约束:\n"
    "1. **色温/色罩只看『中性灰响应』的 a*/b* 与『肤色』探针**。红/黄/绿/蓝四张本身就是高饱和彩色, "
    "它们的 chroma 被压缩是正常现象, 不要据此把预设叫成『冷调』或『赛博』。"
    "只有中性灰确实被推冷(b*<0)时才算冷调。\n"
    "2. **不要滥用『赛博』『霓虹』**。仅当中性/肤色无明显冷移、却有强烈增艳(八带平均 ΔSat 明显为正)时才用。\n"
    "3. **区分度是重点**: name 可以粗, 但 per_probe 六行必须各自不同, 用测量说话"
    "(例如『蓝: 压暗 -12、转青 +18°、去饱 -40%』)。"
    "同一大类的不同预设, per_probe 细节必须不同; caption 要抓住这个预设区别于同类的最显著 1-2 个处理, "
    "不得把不同预设糊成同一句。\n"
    "4. 描述 look 的功能, 不描述探针图里画的是什么东西(不要写『照片里有一辆车』)。\n"
    "5. 测量为准: 你的文字必须与响应表一致, 不臆造未测到的效果。\n"
    f"风格大类参考词(不强制沿用, 可另起): {TAXONOMY_HINT}\n"
    "strength: subtle=几乎看不出/微调, moderate=明显但自然, strong=强风格化。\n"
    "scene_affinity: portrait=更适合人像, landscape=更适合风光, general=通用。\n"
    "只输出 JSON。"
)

SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["name", "per_probe", "caption", "style_major", "style_minor",
                 "scene_affinity", "strength"],
    "properties": {
        "name": {"type": "string", "description": "≤10 字中文风格名"},
        "per_probe": {
            "type": "object",
            "additionalProperties": False,
            "required": list(PROBE_CN),
            "properties": {cn: {"type": "string"} for cn in PROBE_CN},
        },
        "caption": {"type": "string", "description": "一句话, 这个预设区别于同类的最显著处理"},
        "style_major": {"type": "string", "description": "风格大类"},
        "style_minor": {"type": "string", "description": "风格小类"},
        "scene_affinity": {"type": "string", "enum": ["portrait", "landscape", "general"]},
        "strength": {"type": "string", "enum": ["subtle", "moderate", "strong"]},
    },
}


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def apply_lut(image: np.ndarray, grid: np.ndarray,
              dmin: Sequence[float] | None = None,
              dmax: Sequence[float] | None = None) -> np.ndarray:
    """Trilinear LUT application for ``[b][g][r] -> RGB`` grids, values 0..1.

    Written independently of ``apply_lut_cpu_oracle`` and checked against it by
    ``render --selfcheck``: the axis convention is the one thing in this pipeline
    whose silent inversion produced the annotations we are now throwing away, so
    it gets an executable second opinion rather than a comment.
    """
    source = np.asarray(image, dtype=np.float32)
    cube = np.asarray(grid, dtype=np.float32)
    if cube.ndim != 4 or cube.shape[-1] != 3 or len(set(cube.shape[:3])) != 1:
        raise ValueError(f"LUT grid must be cubic BGRxRGB, got {cube.shape}")
    size = cube.shape[0]
    low = np.zeros(3, dtype=np.float32) if dmin is None else np.asarray(dmin, dtype=np.float32)
    high = np.ones(3, dtype=np.float32) if dmax is None else np.asarray(dmax, dtype=np.float32)
    span = np.where(high == low, np.float32(1.0), high - low)
    coords = np.clip((source - low) / span, 0.0, 1.0) * (size - 1)
    floor = np.floor(coords).astype(np.int64)
    ceil = np.minimum(floor + 1, size - 1)
    frac = (coords - floor).astype(np.float32)

    flat = cube.reshape(-1, 3)
    r0, g0, b0 = floor[..., 0], floor[..., 1], floor[..., 2]
    r1, g1, b1 = ceil[..., 0], ceil[..., 1], ceil[..., 2]
    fr = frac[..., 0, None]
    fg = frac[..., 1, None]
    fb = frac[..., 2, None]

    def at(bi, gi, ri):
        return flat[(bi * size + gi) * size + ri]

    c00 = at(b0, g0, r0) * (1.0 - fr) + at(b0, g0, r1) * fr
    c10 = at(b0, g1, r0) * (1.0 - fr) + at(b0, g1, r1) * fr
    c01 = at(b1, g0, r0) * (1.0 - fr) + at(b1, g0, r1) * fr
    c11 = at(b1, g1, r0) * (1.0 - fr) + at(b1, g1, r1) * fr
    lower = c00 * (1.0 - fg) + c10 * fg
    upper = c01 * (1.0 - fg) + c11 * fg
    return np.clip(lower * (1.0 - fb) + upper * fb, 0.0, 1.0).astype(np.float32)


def to_long_edge(image: Image.Image, long_edge: int = LONG_EDGE) -> Image.Image:
    scale = long_edge / max(image.width, image.height)
    if scale >= 1.0:
        return image
    return image.resize((max(1, round(image.width * scale)),
                         max(1, round(image.height * scale))), Image.LANCZOS)


# --------------------------------------------------------------------------- #
# probes
# --------------------------------------------------------------------------- #
def _archive_reader():
    """An ``ArchiveReader`` whose shard roots are rewritten onto the soft mount.

    The catalog stores ``/mnt/nfs/...`` (hard, writable).  Reads must go through
    ``/mnt/nfs-ro`` -- a hung hard mount takes the process with it, and nothing
    here has any business writing to the archive.
    """
    from dataset_build.tools.archive_reader import ArchiveReader

    class ReadOnlyRootReader(ArchiveReader):
        def locate(self, source_path):
            row = dict(super().locate(source_path))
            root = str(row["root"])
            if root == "/mnt/nfs" or root.startswith("/mnt/nfs/"):
                row["root"] = "/mnt/nfs-ro" + root[len("/mnt/nfs"):]
            return row

    return ReadOnlyRootReader()


def probe_dir() -> Path:
    return WORK / "probes"


def probe_before_png(name: str) -> Path:
    return probe_dir() / f"before_{name}.png"


def probe_before_jpg(name: str) -> Path:
    return probe_dir() / f"before_{name}.jpg"


def cmd_probes(args: argparse.Namespace) -> int:
    out = probe_dir()
    out.mkdir(parents=True, exist_ok=True)
    reader = None
    records = []
    for probe in PROBES:
        started = time.time()
        source = Path(probe["path"])
        if source.is_file():
            raw, origin = source.read_bytes(), "local"
        else:
            if reader is None:
                reader = _archive_reader()
            raw, origin = reader.read(str(source)), "nfs-archive"
        elapsed = time.time() - started
        with Image.open(io.BytesIO(raw)) as handle:
            handle.load()
            oriented = ImageOps.exif_transpose(handle).convert("RGB")
        resized = to_long_edge(oriented)
        resized.save(probe_before_png(probe["name"]), format="PNG", optimize=True)
        resized.save(probe_before_jpg(probe["name"]), format="JPEG",
                     quality=JPEG_QUALITY, subsampling=0)
        records.append({
            "slot": probe["slot"], "name": probe["name"], "cn": probe["cn"],
            "asset_id": probe["asset_id"], "source_path": str(source),
            "origin": origin, "fetch_seconds": round(elapsed, 3),
            "source_bytes": len(raw),
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "original_size": [oriented.width, oriented.height],
            "before_size": [resized.width, resized.height],
        })
        print(f"[probes] {probe['name']:<8} {origin:<12} {elapsed:6.2f}s  "
              f"{oriented.width}x{oriented.height} -> {resized.width}x{resized.height}")
    if reader is not None:
        reader.close()
    payload = {"generated_at": _now(), "long_edge": LONG_EDGE,
               "jpeg_quality": JPEG_QUALITY, "probes": records}
    (out / "probes.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[probes] wrote {out/'probes.json'}")
    return 0


def load_probe_arrays() -> dict[str, np.ndarray]:
    arrays = {}
    for probe in PROBES:
        path = probe_before_png(probe["name"])
        if not path.is_file():
            raise SystemExit(f"probe cache missing: {path} (run `pipeline.py probes` first)")
        with Image.open(path) as handle:
            handle.load()
            arrays[probe["name"]] = np.asarray(handle.convert("RGB"), dtype=np.float32) / 255.0
    return arrays


# --------------------------------------------------------------------------- #
# bank access
# --------------------------------------------------------------------------- #
def load_bank() -> list[dict[str, Any]]:
    rows = []
    with FEATURES.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("kind") == "lut":
                rows.append(row)
    rows.sort(key=lambda item: item["preset_id"])
    return rows


def select(rows: Sequence[Mapping[str, Any]], limit: int | None,
           ids: Sequence[str] | None) -> list[dict[str, Any]]:
    if ids:
        wanted = set(ids)
        chosen = [dict(row) for row in rows if row["preset_id"] in wanted]
        missing = wanted - {row["preset_id"] for row in chosen}
        if missing:
            raise SystemExit(f"unknown preset_id(s): {sorted(missing)}")
    else:
        chosen = [dict(row) for row in rows]
    return chosen[:limit] if limit else chosen


def render_dir(preset_id: str) -> Path:
    return WORK / "renders" / preset_id[4:6] / preset_id


def render_complete(preset_id: str) -> bool:
    base = render_dir(preset_id)
    return all((base / f"{p['name']}.jpg").is_file()
               and (base / f"{p['name']}.jpg").stat().st_size > 0 for p in PROBES)


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #
_WORKER: dict[str, Any] = {}


def _worker_init() -> None:
    _WORKER["npz"] = np.load(LUTS_NPZ, allow_pickle=False)
    _WORKER["meta"] = json.loads(LUTS_META.read_text())
    _WORKER["probes"] = load_probe_arrays()


def _render_one(preset_id: str) -> tuple[str, str]:
    try:
        meta = _WORKER["meta"][preset_id]
        grid = _WORKER["npz"][preset_id]
        base = render_dir(preset_id)
        base.mkdir(parents=True, exist_ok=True)
        for probe in PROBES:
            target = base / f"{probe['name']}.jpg"
            if target.is_file() and target.stat().st_size > 0:
                continue
            out = apply_lut(_WORKER["probes"][probe["name"]], grid,
                            meta.get("dmin"), meta.get("dmax"))
            pixels = np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)
            temporary = target.with_suffix(".jpg.part")
            Image.fromarray(pixels, mode="RGB").save(
                temporary, format="JPEG", quality=JPEG_QUALITY, subsampling=0)
            os.replace(temporary, target)
        return preset_id, ""
    except Exception as exc:  # noqa: BLE001 - one bad LUT must not kill the batch
        return preset_id, f"{type(exc).__name__}: {exc}"


def cmd_render(args: argparse.Namespace) -> int:
    rows = select(load_bank(), args.limit, args.ids)
    pending = [row["preset_id"] for row in rows if not render_complete(row["preset_id"])]
    print(f"[render] {len(rows)} selected, {len(rows)-len(pending)} already done, "
          f"{len(pending)} to render, workers={args.workers}")
    progress = WORK / "out" / "render_progress.json"
    progress.parent.mkdir(parents=True, exist_ok=True)
    done = failed = 0
    failures: list[tuple[str, str]] = []
    started = time.time()
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as pool:
            for index, (preset_id, error) in enumerate(
                    pool.map(_render_one, pending, chunksize=4), 1):
                if error:
                    failed += 1
                    failures.append((preset_id, error))
                else:
                    done += 1
                if index % 200 == 0 or index == len(pending):
                    elapsed = time.time() - started
                    progress.write_text(json.dumps({
                        "selected": len(rows), "pending": len(pending),
                        "done": done, "failed": failed,
                        "elapsed_s": round(elapsed, 1),
                        "rate_presets_per_s": round(index / max(elapsed, 1e-6), 2),
                    }, indent=2))
                    print(f"[render] {index}/{len(pending)} ok={done} fail={failed} "
                          f"{elapsed:.0f}s", flush=True)
    for preset_id, error in failures[:20]:
        print(f"[render] FAIL {preset_id}: {error}", file=sys.stderr)
    if args.selfcheck:
        rc = selfcheck([row["preset_id"] for row in rows], args.selfcheck_n)
        if rc:
            return rc
    return 1 if failed else 0


def selfcheck(preset_ids: Sequence[str], count: int = 3) -> int:
    """Re-render sampled probes through the reference oracle and diff."""
    sys.modules.setdefault("tomli", __import__("tomllib"))
    from dataset_build.src.construct.rendering import apply_lut_cpu_oracle

    npz = np.load(LUTS_NPZ, allow_pickle=False)
    meta = json.loads(LUTS_META.read_text())
    probes = load_probe_arrays()
    rng = random.Random(20260811)
    sample = rng.sample(list(preset_ids), min(count, len(preset_ids)))
    worst = 0.0
    print(f"[selfcheck] {len(sample)} presets x {len(PROBES)} probes vs apply_lut_cpu_oracle")
    for preset_id in sample:
        grid = npz[preset_id]
        row = meta[preset_id]
        for probe in PROBES:
            image = probes[probe["name"]]
            mine = apply_lut(image, grid, row.get("dmin"), row.get("dmax"))
            reference = apply_lut_cpu_oracle(image, grid,
                                             np.asarray(row.get("dmin"), dtype=np.float32),
                                             np.asarray(row.get("dmax"), dtype=np.float32))
            diff = float(np.abs(mine - reference).max())
            worst = max(worst, diff)
            print(f"[selfcheck] {preset_id} {probe['name']:<8} grid={grid.shape[0]:<3} "
                  f"max|diff|={diff:.3e}")
    print(f"[selfcheck] worst max|diff| = {worst:.3e}  (tolerance 1e-5)")
    if worst >= 1e-5:
        print("[selfcheck] FAILED", file=sys.stderr)
        return 2
    print("[selfcheck] PASS")
    return 0


# --------------------------------------------------------------------------- #
# annotate
# --------------------------------------------------------------------------- #
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jpeg_data_url(raw: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode()


def hsl_features(preset_id: str, npz, meta: Mapping[str, Any]) -> dict[str, Any]:
    row = meta[preset_id]
    return hslfeat.compute(apply_lut, npz[preset_id], row.get("dmin"), row.get("dmax"))


def build_content(preset_id: str, features: Mapping[str, Any],
                  before: Mapping[str, str]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = [{"type": "input_text", "text": SYSTEM_PROMPT}]
    parts.append({"type": "input_text", "text": hslfeat.render_table(features)})
    base = render_dir(preset_id)
    for index, probe in enumerate(PROBES, 1):
        after = base / f"{probe['name']}.jpg"
        if not after.is_file():
            raise FileNotFoundError(f"missing render {after}")
        parts.append({"type": "input_text",
                      "text": f"【探针 {index}/6 · {probe['cn']}({probe['desc']})】"
                              f"下面第一张是 before(原图), 第二张是 after(过这个 LUT 之后)"})
        parts.append({"type": "input_image", "image_url": before[probe["name"]]})
        parts.append({"type": "input_image", "image_url": _jpeg_data_url(after.read_bytes())})
    parts.append({"type": "input_text",
                  "text": "现在给出这个 LUT 的 JSON 标注。per_probe 六行必须逐一不同, "
                          "且与上面的响应表一致。"})
    return parts


def cmd_annotate(args: argparse.Namespace) -> int:
    from dataset_build.tools.reeval_relay import (
        JsonlStore, ModelSubstituted, call_once, load_lanes, prompt_digest,
    )

    rows = select(load_bank(), args.limit, args.ids)
    missing = [row["preset_id"] for row in rows if not render_complete(row["preset_id"])]
    if missing:
        raise SystemExit(f"{len(missing)} selected presets are not rendered, "
                         f"e.g. {missing[:5]} -- run `pipeline.py render` first")

    lanes = [lane for lane in load_lanes(Path(args.config)) if lane.lane_id == args.lane]
    if not lanes:
        raise SystemExit(f"lane {args.lane!r} not found in {args.config}")
    lane = lanes[0]

    before = {p["name"]: _jpeg_data_url(probe_before_jpg(p["name"]).read_bytes())
              for p in PROBES}

    out_dir = WORK / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    store = JsonlStore(Path(args.out or out_dir / "annotations.jsonl"))
    progress_path = out_dir / "annotate_progress.json"
    pending = [row for row in rows if row["preset_id"] not in store.done]
    print(f"[annotate] model={args.model} effort={args.effort} lane={lane.lane_id} "
          f"concurrency={args.concurrency}")
    print(f"[annotate] {len(rows)} selected, {len(rows)-len(pending)} already done, "
          f"{len(pending)} to call -> {store.path}")

    # Up front and single-threaded on purpose: an ``NpzFile`` wraps one ZipFile
    # handle, and 32 worker threads pulling grids out of it concurrently is a
    # data race, not a speedup.  The whole set costs well under a minute.
    npz = np.load(LUTS_NPZ, allow_pickle=False)
    meta = json.loads(LUTS_META.read_text())
    feature_start = time.time()
    features_by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(pending, 1):
        features_by_id[row["preset_id"]] = hsl_features(row["preset_id"], npz, meta)
        if index % 500 == 0:
            print(f"[annotate] hsl features {index}/{len(pending)}", flush=True)
    npz.close()
    if pending:
        print(f"[annotate] hsl features for {len(pending)} LUTs in "
              f"{time.time()-feature_start:.1f}s")

    work: queue.Queue = queue.Queue()
    for row in pending:
        work.put(row)
    stats = {"total": len(rows), "skipped": len(rows) - len(pending), "ok": 0,
             "failed": 0, "substituted": 0, "input_tokens": 0, "output_tokens": 0,
             "latency_sum_s": 0.0}
    lock = threading.Lock()
    started = time.time()

    def flush() -> None:
        payload = dict(stats)
        payload["elapsed_s"] = round(time.time() - started, 1)
        payload["remaining"] = len(pending) - stats["ok"] - stats["failed"]
        if stats["ok"]:
            payload["mean_latency_s"] = round(stats["latency_sum_s"] / stats["ok"], 2)
        progress_path.write_text(json.dumps(payload, indent=2))

    def worker() -> None:
        while True:
            try:
                row = work.get_nowait()
            except queue.Empty:
                return
            preset_id = row["preset_id"]
            try:
                features = features_by_id[preset_id]
                content = build_content(preset_id, features, before)
            except Exception as exc:  # noqa: BLE001 - a broken input is a failed record
                with lock:
                    stats["failed"] += 1
                    flush()
                store.append({"key": preset_id, "preset_id": preset_id, "ok": False,
                              "error": f"{type(exc).__name__}: {exc}",
                              "provenance": {"timestamp": _now(), "prompt_rev": PROMPT_REV}})
                continue
            digest = prompt_digest(content, SCHEMA)
            last_error = None
            retried: list[str] = []
            for attempt in range(args.attempts):
                call_started = time.time()
                try:
                    result, call_meta = call_once(
                        lane, args.model, args.effort, content, SCHEMA,
                        schema_name="lut_annotation", max_output_tokens=args.max_output_tokens)
                    returned = str(call_meta.get("returned_model") or "")
                    # ``call_once`` only pins a prefix.  The instruction is an exact
                    # pin, so an unannounced snapshot suffix is a substitution here.
                    if returned != args.model:
                        lane.substitutions += 1
                        raise ModelSubstituted(
                            f"requested {args.model}, relay returned {returned}")
                    latency = time.time() - call_started
                    lane.calls += 1
                    store.append({
                        "key": preset_id, "preset_id": preset_id, "ok": True,
                        **result,
                        "hsl_features": features,
                        "provenance": {
                            "model": args.model,
                            "response_model": call_meta.get("returned_model"),
                            "reasoning_effort": args.effort,
                            "lane": call_meta.get("lane"),
                            "attempt": attempt,
                            "timestamp": _now(),
                            "prompt_rev": PROMPT_REV,
                            "render_rev": RENDER_REV,
                            "hsl_spec_rev": hslfeat.SPEC_REV,
                            "probe_order": [p["cn"] for p in PROBES],
                            "input_tokens": call_meta.get("input_tokens"),
                            "output_tokens": call_meta.get("output_tokens"),
                            "latency_s": round(latency, 2),
                            "retried_errors": retried,
                            **digest,
                        },
                    })
                    with lock:
                        stats["ok"] += 1
                        stats["latency_sum_s"] += latency
                        stats["input_tokens"] += call_meta.get("input_tokens") or 0
                        stats["output_tokens"] += call_meta.get("output_tokens") or 0
                        flush()
                    break
                except ModelSubstituted as exc:
                    last_error = f"ModelSubstituted: {exc}"
                    retried.append(last_error)
                    with lock:
                        stats["substituted"] += 1
                    time.sleep(1.0 + random.random())
                except Exception as exc:  # noqa: BLE001 - transport retry
                    lane.errors += 1
                    last_error = f"{type(exc).__name__}: {str(exc)[:300]}"
                    retried.append(last_error)
                    time.sleep(2.0 * (attempt + 1) + random.random())
            else:
                store.append({"key": preset_id, "preset_id": preset_id, "ok": False,
                              "error": last_error, "retried_errors": retried,
                              "provenance": {"timestamp": _now(), "prompt_rev": PROMPT_REV,
                                             "model": args.model, **digest}})
                with lock:
                    stats["failed"] += 1
                    flush()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.concurrency)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    flush()
    elapsed = time.time() - started
    print(f"[annotate] ok={stats['ok']} failed={stats['failed']} "
          f"substituted={stats['substituted']} in {elapsed:.1f}s")
    if stats["ok"]:
        print(f"[annotate] mean latency {stats['latency_sum_s']/stats['ok']:.2f}s/record; "
              f"tokens in={stats['input_tokens']} out={stats['output_tokens']} "
              f"(mean in={stats['input_tokens']/stats['ok']:.0f} "
              f"out={stats['output_tokens']/stats['ok']:.0f})")
    print(f"[annotate] lane calls={lane.calls} errors={lane.errors} "
          f"substitutions={lane.substitutions}")
    return 1 if stats["failed"] else 0


def cmd_failures(args: argparse.Namespace) -> int:
    path = Path(args.out or WORK / "out" / "annotations.jsonl")
    ok: set[str] = set()
    bad: dict[str, str] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("ok"):
                ok.add(row["key"])
            else:
                bad[row["key"]] = str(row.get("error"))[:200]
    outstanding = {key: error for key, error in bad.items() if key not in ok}
    for key, error in sorted(outstanding.items()):
        print(f"{key}\t{error}")
    print(f"# ok={len(ok)} outstanding_failures={len(outstanding)}", file=sys.stderr)
    return 0


# --------------------------------------------------------------------------- #
# pack
# --------------------------------------------------------------------------- #
def cmd_pack(args: argparse.Namespace) -> int:
    rows = select(load_bank(), args.limit, args.ids)
    wanted = {row["preset_id"] for row in rows}
    annotations_path = Path(args.annotations or WORK / "out" / "annotations.jsonl")
    annotations: dict[str, dict] = {}
    if annotations_path.is_file():
        for line in annotations_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("ok") and row["key"] in wanted:
                annotations[row["key"]] = row
    print(f"[pack] {len(rows)} LUTs selected, {len(annotations)} annotations matched")

    used: set[str] = set()
    entries = []
    for row in rows:
        source = Path(row["path"])
        member = f"luts/{row['fmt']}/{source.name}"
        if member in used:
            member = f"luts/{row['fmt']}/{row['preset_id']}_{source.name}"
        used.add(member)
        entries.append((row, source, member))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    compression = zipfile.ZIP_STORED if args.store else zipfile.ZIP_DEFLATED
    manifest_rows = []
    total_bytes = 0
    started = time.time()
    with zipfile.ZipFile(out, "w", compression=compression, allowZip64=True) as archive:
        for index, (row, source, member) in enumerate(entries, 1):
            payload = source.read_bytes()
            total_bytes += len(payload)
            archive.writestr(member, payload)
            manifest_rows.append({
                "preset_id": row["preset_id"],
                "fmt": row["fmt"],
                "pack_id": row.get("pack_id"),
                "member": member,
                "source_path": str(source),
                "bytes": len(payload),
                "preset_content_hash": row.get("preset_content_hash"),
                "file_sha256": hashlib.sha256(payload).hexdigest(),
                "annotated": row["preset_id"] in annotations,
            })
            if index % 500 == 0:
                print(f"[pack] {index}/{len(entries)} members, "
                      f"{total_bytes/1e9:.2f} GB raw", flush=True)
        lines = "".join(json.dumps(annotations[row["preset_id"]], ensure_ascii=False) + "\n"
                        for row in rows if row["preset_id"] in annotations)
        archive.writestr("annotations.jsonl", lines.encode("utf-8"))
        probes_meta = probe_dir() / "probes.json"
        manifest = {
            "generated_at": _now(),
            "bank_dir": str(BANK),
            "n_luts": len(entries),
            "n_annotations": len(annotations),
            "fmt_counts": {fmt: sum(1 for r in manifest_rows if r["fmt"] == fmt)
                           for fmt in sorted({r["fmt"] for r in manifest_rows})},
            "raw_lut_bytes": total_bytes,
            "annotation": {
                "model": MODEL, "reasoning_effort": REASONING_EFFORT,
                "prompt_rev": PROMPT_REV, "hsl_spec_rev": hslfeat.SPEC_REV,
                "schema_sha256": hashlib.sha256(
                    json.dumps(SCHEMA, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            },
            "render": {
                "rev": RENDER_REV, "long_edge": LONG_EDGE, "jpeg_quality": JPEG_QUALITY,
                "engine": "cpu trilinear (tools/lut_reannotate/pipeline.py:apply_lut)",
                "probes": json.loads(probes_meta.read_text()) if probes_meta.is_file() else None,
            },
            "luts": manifest_rows,
        }
        archive.writestr("MANIFEST.json",
                         json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))

    with zipfile.ZipFile(out) as archive:
        names = archive.namelist()
        lut_members = [n for n in names if n.startswith("luts/")]
        with archive.open("annotations.jsonl") as handle:
            annotation_lines = sum(1 for line in handle if line.strip())
    size = out.stat().st_size
    print(f"[pack] wrote {out} ({size/1e9:.3f} GB, raw {total_bytes/1e9:.3f} GB, "
          f"{time.time()-started:.1f}s)")
    print(f"[pack] members: {len(names)} total = {len(lut_members)} luts "
          f"+ annotations.jsonl + MANIFEST.json")
    print(f"[pack] annotations.jsonl lines = {annotation_lines}")
    consistent = (len(lut_members) == len(entries) and len(names) == len(entries) + 2
                  and annotation_lines == len(annotations))
    if not consistent:
        print("[pack] MISMATCH: member count / annotation line count disagree",
              file=sys.stderr)
        return 2
    if annotation_lines != len(entries):
        print(f"[pack] NOTE: {len(entries)-annotation_lines} LUTs have no annotation "
              f"(expected only for a --limit smoke run)")
    print("[pack] consistency check PASS")
    return 0


# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    global WORK
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--work", default=None, help=f"working directory (default {WORK})")
    sub = parser.add_subparsers(dest="command", required=True)

    probes = sub.add_parser("probes", help="fetch and cache the six probe source images")
    probes.set_defaults(func=cmd_probes)

    render = sub.add_parser("render", help="render every LUT over every probe (CPU)")
    render.add_argument("--workers", type=int, default=32)
    render.add_argument("--limit", type=int, default=None)
    render.add_argument("--ids", nargs="*", default=None)
    render.add_argument("--selfcheck", action="store_true", default=True)
    render.add_argument("--no-selfcheck", dest="selfcheck", action="store_false")
    render.add_argument("--selfcheck-n", type=int, default=3)
    render.set_defaults(func=cmd_render)

    annotate = sub.add_parser("annotate", help="call the relay VLM for every rendered LUT")
    annotate.add_argument("--config", default=str(CONFIG_TOML))
    annotate.add_argument("--lane", default=LANE_ID)
    annotate.add_argument("--model", default=MODEL)
    annotate.add_argument("--effort", default=REASONING_EFFORT)
    annotate.add_argument("--max-output-tokens", type=int, default=MAX_OUTPUT_TOKENS)
    annotate.add_argument("--concurrency", type=int, default=32)
    annotate.add_argument("--attempts", type=int, default=3)
    annotate.add_argument("--limit", type=int, default=None)
    annotate.add_argument("--ids", nargs="*", default=None)
    annotate.add_argument("--out", default=None)
    annotate.set_defaults(func=cmd_annotate)

    failures = sub.add_parser("failures", help="list preset_ids that still have no ok row")
    failures.add_argument("--out", default=None)
    failures.set_defaults(func=cmd_failures)

    pack = sub.add_parser("pack", help="zip LUT bodies + annotations.jsonl + MANIFEST.json")
    pack.add_argument("--out", required=True)
    pack.add_argument("--annotations", default=None)
    pack.add_argument("--limit", type=int, default=None)
    pack.add_argument("--ids", nargs="*", default=None)
    pack.add_argument("--store", action="store_true", help="no deflate (faster, bigger)")
    pack.set_defaults(func=cmd_pack)

    args = parser.parse_args(argv)
    if args.work:
        WORK = Path(args.work)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
