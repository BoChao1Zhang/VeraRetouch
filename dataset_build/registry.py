"""
dataset_build/registry.py
=========================
The SOURCE + RECIPE registry for the VeraRetouch Direction-A, RECIPE-BASED,
region-local-heavy 1,000,000-sample dataset build.

This module turns the on-disk corpora into three flat, resumable indexes that the
downstream streams (streams.py) consume without ever touching the raw directory
trees again:

  - source_index.jsonl  : one `SourceItem` per usable 'before' image / raw.
  - recipe_index.jsonl  : one `RecipeAsset` per usable 'look' recipe (post junk
                          filter, post technical/B&W filter, post dedup-by-stem).
  - pack_index.jsonl    : one row per 全店素材 product pack (per probe_全店素材 §6),
                          for traceability + per-pack scene routing.

Design constraints (from the architect / contracts.py / config.yaml / probes):
  * CPU-ONLY. No torch / transformers / rawpy / cv2 / model weights. We `os.scandir`
    the trees, parse XMP with stdlib ElementTree (via the reused parser), and read
    the preset_dataset_v1 manifest JSONL. Nothing heavy is imported.
  * RECIPE-BASED (USER DECISION 1): we record source PATHS + recipe PATHS only; we
    never decode or render here. Width/height stay None until a later decode stage.
  * Junk filter (probe_全店素材 §4) + technical/B&W filter (probe_recipe_parsers
    §1b/§2) + dedup-by-preset-stem (G3) are applied to recipes.
  * Scene tagging + scene_affinity routing (probe_sources_budget §3): skin-friendly /
    portrait recipes -> portrait sources; landscape/cine recipes -> scenery sources.
    The preset_dataset_v1 manifest (subject_type/scene_type/style_family/skin_friendly
    + content_hash) is the catalog that drives this; filename heuristics are the
    fallback.
  * Unicode/space-heavy CJK paths (probe_全店素材 G8): never shell-glob unquoted; use
    pathlib / os.scandir.

Grounding (read):
  - dataset_build/contracts.py                      (SourceItem / RecipeAsset / Registry)
  - dataset_build/config.yaml                        (all paths + filters live here)
  - docs/plan/dataset/probe/probe_sources_budget.md  (§1 inventory, §3 scene rule)
  - docs/plan/dataset/probe/probe_全店素材.md          (§4 junk filter, §6 pack spec, G1-G8)
  - docs/plan/dataset/probe/probe_GREYSKY_raw_photogs.md (triples, raw decode, awards, korean)
  - docs/plan/dataset/probe/probe_recipe_parsers.md  (§1 XMP map, §2 cube/3dl, §3 costyle skip)

Reused on-disk code (imported lazily, only when scanning recipes):
  - presets/scripts/build_preset_dataset.parse_xmp_file (XMP -> core_settings raw strings)
  - presets/preset_dataset_v1/manifests/manifest.jsonl  (scene/style/skin catalog)

CLI:
    python -m dataset_build.registry --config dataset_build/config.yaml \
        [--out-dir <dir>] [--limit-per-corpus N] [--no-recipe-parse]
writes source_index.jsonl / recipe_index.jsonl / pack_index.jsonl under out-dir
(defaults to config out_root) and prints the counts dict.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from dataset_build.contracts import (
    RecipeAsset,
    RecipeKind,
    Registry,
    RawDecode,
    SourceItem,
)

# ---------------------------------------------------------------------------
# Static fact tables (mirror config defaults; config overrides them at runtime).
# ---------------------------------------------------------------------------

RECIPE_EXTS = {".xmp", ".lrtemplate", ".cube", ".3dl"}  # costyle DEFERRED (probe_recipe §3)
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
RAW_EXTS = {".dng", ".cr2", ".arw", ".nef", ".rw2"}

# How recipe extension maps to RecipeKind / fmt.
_FMT_TO_KIND = {
    "xmp": RecipeKind.PARAM,
    "lrtemplate": RecipeKind.PARAM,
    "cube": RecipeKind.LUT,
    "3dl": RecipeKind.LUT,
}

# Default junk-filter (probe_全店素材 §4). Overridable from config.recipes.filters.
_DEFAULT_JUNK_EXT = {
    ".mov", ".mp4", ".txt", ".doc", ".docx", ".pdf", ".zip", ".rar", ".exe",
    ".apk", ".cfg", ".url", ".wav", ".jsp", ".8bf", ".ds_store", ".free",
    ".cbf", ".mbr", ".hfs", ".pkg",
}
_DEFAULT_JUNK_PATH_RE = r"教程|安装|导入|说明|必看|二维码|广告|微信|售后|转换器|Generator|使用方法"
_DEFAULT_TECHNICAL_RE = (
    r"Slog|S-Log|SLog2|SLog3|REC709|Rec\.709|709toLog|LogtoRec|LinearTo|_to_|"
    r"Conversion|Technical|LUTCalc|Identity|Neutral"
)
_DEFAULT_MIN_SOURCE_BYTES = 512_000

# Scene-matching regexes (probe_sources_budget §3; config.scene_matching).
_DEFAULT_PORTRAIT_RE = r"portrait|人像|instagram|film.?portrait|skin|新娘|婚礼|wedding|肤|肤色|汉装|古风|写真"
_DEFAULT_SCENERY_RE = r"landscape|ocean|海洋|flower|花|cine|电影|sky|风光|sunset|日落|mountain|风景"

# preset_dataset_v1 manifest is rooted relative to this prefix (verified on disk).
_PRESET_V1_PATH_ROOT = "/home/bc/retouching/presets"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stable_id(*parts: Any, prefix: str = "") -> str:
    """sha1 over the given parts -> short stable id (contracts: 'sha1 of abs_path + size')."""
    h = hashlib.sha1(" ".join(str(p) for p in parts).encode("utf-8", "surrogatepass"))
    return f"{prefix}{h.hexdigest()[:16]}"


def _safe_stat_size(path: Path) -> Optional[int]:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _iter_files(root: Path) -> Iterator[Path]:
    """Recursive, OSError-tolerant file walk over a (possibly CJK-named) tree."""
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            yield Path(entry.path)
                    except OSError:
                        continue
        except (OSError, PermissionError):
            continue


def _ext(path: Path) -> str:
    return path.suffix.lower()


def _preset_stem(path: Path) -> str:
    """Dedup key (G3): preset stem, whitespace/case-normalized, ext stripped.
    Cross-format duplicates (same look as xmp/lrtemplate/cube/dng) collapse here."""
    stem = path.stem
    stem = re.sub(r"\s+", "", stem).lower()
    # strip common ordinal suffixes like '-1' / '_02' that distinguish format dupes weakly
    return stem


# GREYSKY collection layout: <collection>/{DNG 原片参数文件, JPG 预览文件, XMP 预设文件}/<stem>.{dng,jpg,xmp}
_GREYSKY_JPG_DIR = "JPG 预览文件"   # expert preview == the gold 'after'
_GREYSKY_XMP_DIR = "XMP 预设文件"   # the expert preset


def _greysky_siblings(dng: Path) -> Tuple[Optional[str], Optional[str]]:
    """Resolve a GREYSKY DNG's sibling expert JPG (preview = the gold 'after') and
    XMP (preset) by stem within its collection. Exact ``<stem>.ext`` first, then a
    whitespace/case-normalized scan (folder/file names carry CJK + stray spaces)."""
    col = dng.parent.parent  # <collection>/DNG 原片参数文件/x.dng -> <collection>
    target = _preset_stem(dng)

    def _find(subdir: str, ext: str) -> Optional[str]:
        d = col / subdir
        if not d.is_dir():
            return None
        exact = d / f"{dng.stem}{ext}"
        if exact.is_file():
            return str(exact)
        try:
            for f in d.iterdir():
                if f.is_file() and f.suffix.lower() == ext and _preset_stem(f) == target:
                    return str(f)
        except OSError:
            pass
        return None

    return _find(_GREYSKY_JPG_DIR, ".jpg"), _find(_GREYSKY_XMP_DIR, ".xmp")


def _load_yaml(path: str) -> Dict[str, Any]:
    """Tiny YAML loader: prefer PyYAML; the config is plain enough that we require it."""
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover - environment guard
        raise RuntimeError(
            "PyYAML is required to read config.yaml. `pip install pyyaml` "
            f"(import failed: {e})"
        )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


class SourceRegistry(Registry):
    """Concrete `Registry`: scans sources + recipes from config-declared corpora.

    Constructed from the parsed config dict (or a path to config.yaml). All paths,
    budgets, and filter regexes are read from config — never hard-coded — so the
    main process can repoint corpora without editing code.
    """

    def __init__(self, config: Dict[str, Any], *, parse_recipe_params: bool = True,
                 limit_per_corpus: Optional[int] = None) -> None:
        self.cfg = config
        self.parse_recipe_params = parse_recipe_params
        self.limit_per_corpus = limit_per_corpus

        self.sources_cfg: Dict[str, Any] = config.get("sources", {}) or {}
        self.recipes_cfg: Dict[str, Any] = config.get("recipes", {}) or {}
        self.scene_cfg: Dict[str, Any] = config.get("scene_matching", {}) or {}

        filters = (self.recipes_cfg.get("filters") or {})
        self.junk_ext = {e.lower() for e in filters.get("junk_ext", [])} or set(_DEFAULT_JUNK_EXT)
        self.junk_path_re = re.compile(
            filters.get("junk_path_re", _DEFAULT_JUNK_PATH_RE), re.IGNORECASE
        )
        self.technical_re = re.compile(
            filters.get("technical_name_re", _DEFAULT_TECHNICAL_RE), re.IGNORECASE
        )
        self.drop_bw = bool(filters.get("drop_bw", True))
        self.min_source_bytes = int(filters.get("min_source_bytes", _DEFAULT_MIN_SOURCE_BYTES))

        self.portrait_re = re.compile(
            self.scene_cfg.get("portrait_recipe_re", _DEFAULT_PORTRAIT_RE), re.IGNORECASE
        )
        self.scenery_re = re.compile(
            self.scene_cfg.get("scenery_recipe_re", _DEFAULT_SCENERY_RE), re.IGNORECASE
        )

        # Lazily built {abs_source_path -> manifest record} catalog (scene tags).
        self._manifest_by_path: Optional[Dict[str, Dict[str, Any]]] = None
        # Pack rows produced as a side effect of scanning 全店素材 recipes/sources.
        self._pack_rows: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def from_config_path(cls, path: str, **kw: Any) -> "SourceRegistry":
        return cls(_load_yaml(path), **kw)

    # ------------------------------------------------------------------ #
    # Scene helpers
    # ------------------------------------------------------------------ #

    def _scene_affinity_from_text(self, *texts: Optional[str]) -> str:
        """Soft routing: portrait | landscape | any from a recipe's name/style text."""
        blob = " ".join(t for t in texts if t)
        if self.portrait_re.search(blob):
            return "portrait"
        if self.scenery_re.search(blob):
            return "landscape"
        return self.scene_cfg.get("default", "any")

    def _load_manifest_catalog(self) -> Dict[str, Dict[str, Any]]:
        """Read preset_dataset_v1 manifest.jsonl into {abs_source_path -> record}.

        Provides scene_type / style_family / subject_type / skin_friendly /
        content_hash for any recipe whose abs path appears in the manifest. Demoted
        (scene_matching) role only — never a primary recipe source.
        """
        if self._manifest_by_path is not None:
            return self._manifest_by_path
        out: Dict[str, Dict[str, Any]] = {}
        mcfg = (self.recipes_cfg.get("preset_v1_manifest") or {})
        mpath = mcfg.get("manifest")
        if mpath and os.path.exists(mpath):
            with open(mpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    sp = rec.get("source_path")
                    if not sp:
                        continue
                    abs_p = os.path.normpath(os.path.join(_PRESET_V1_PATH_ROOT, sp))
                    out[abs_p] = rec
        self._manifest_by_path = out
        return out

    def _manifest_tags(self, abs_path: str) -> Dict[str, Any]:
        """Return scene/style/skin tags for a recipe abs path, if catalogued."""
        rec = self._load_manifest_catalog().get(os.path.normpath(abs_path))
        if not rec:
            return {}
        skin = str(rec.get("skin_friendly", "")).strip().lower()
        return {
            "scene_type": rec.get("scene_type"),
            "style_family": rec.get("style_family"),
            "subject_type": rec.get("subject_type"),
            "skin_friendly": skin in ("yes", "true", "1"),
            "content_hash": rec.get("content_hash"),
            "name_en": rec.get("name_en"),
            "is_duplicate": bool(rec.get("is_duplicate")),
        }

    # ================================================================== #
    # SOURCES
    # ================================================================== #

    def scan_sources(self) -> Iterator[SourceItem]:
        scanners = (
            self._scan_tad66k,
            self._scan_fivek,
            self._scan_greysky_sources,
            self._scan_korean,
            self._scan_quandian_sources,
            self._scan_awards,
            self._scan_mmart,
            self._scan_unsplash,
            self._scan_fivek_gold,
            self._scan_ppr10k,
        )
        for fn in scanners:
            n = 0
            for item in fn():
                yield item
                n += 1
                if self.limit_per_corpus is not None and n >= self.limit_per_corpus:
                    break

    # ---- per-corpus source scanners -------------------------------------- #

    def _emit_source(self, path: Path, corpus: str, *, scene: Optional[str] = None,
                     is_portrait_pool: bool = False, raw_decode: RawDecode = RawDecode.NONE,
                     tags: Optional[List[str]] = None,
                     meta: Optional[Dict[str, Any]] = None,
                     size: Optional[int] = None) -> SourceItem:
        ap = str(path)
        sz = size if size is not None else _safe_stat_size(path)
        return SourceItem(
            source_id=_stable_id(ap, sz, prefix="src_"),
            path=ap,
            corpus=corpus,
            raw_decode=raw_decode,
            scene=scene,
            is_portrait_pool=is_portrait_pool,
            tags=tags or [],
            bytes_size=sz,
            meta=meta or {},
        )

    def _corpus_path(self, key: str) -> Optional[Path]:
        c = self.sources_cfg.get(key) or {}
        p = c.get("path")
        return Path(p) if p else None

    def _scan_tad66k(self) -> Iterator[SourceItem]:
        """TAD66K is a zip (~66K). We register the EXTRACTED dir if present, else
        emit a single archive-pointer SourceItem so streams know to extract lazily."""
        c = self.sources_cfg.get("tad66k") or {}
        extract_to = c.get("extract_to")
        scene = c.get("scene", "any")
        if extract_to and os.path.isdir(extract_to):
            for f in _iter_files(Path(extract_to)):
                if _ext(f) in IMAGE_EXTS:
                    yield self._emit_source(f, "tad66k", scene=scene)
            return
        zip_path = c.get("path")
        if zip_path and os.path.exists(zip_path):
            yield self._emit_source(
                Path(zip_path), "tad66k", scene=scene,
                meta={"archive": True, "extract_to": extract_to,
                      "note": "zip not extracted; streams extract lazily to scratch"},
            )

    def _scan_fivek(self) -> Iterator[SourceItem]:
        """FiveK = 5000 DNGs inside a tar (probe_sources §1a). Source-only (no experts).
        Register the tar pointer; the streams extract+rawpy-decode under fivek-cleaning."""
        c = self.sources_cfg.get("fivek") or {}
        tar_path = c.get("path")
        if not tar_path or not os.path.exists(tar_path):
            return
        # If a pre-extracted dir of dng exists alongside, prefer it; else archive pointer.
        scratch = (self.cfg.get("scratch_dir") or "")
        extracted = os.path.join(scratch, "fivek_dng") if scratch else None
        if extracted and os.path.isdir(extracted):
            for f in _iter_files(Path(extracted)):
                if _ext(f) in RAW_EXTS:
                    yield self._emit_source(f, "fivek", scene=c.get("scene", "any"),
                                            raw_decode=RawDecode.RAWPY)
            return
        yield self._emit_source(
            Path(tar_path), "fivek", scene=c.get("scene", "any"),
            raw_decode=RawDecode.RAWPY,
            meta={"archive": True, "raw_glob": c.get("raw_glob"),
                  "note": "tar of 5000 dng; extract lazily, decode under fivek-cleaning"},
        )

    def _scan_greysky_sources(self) -> Iterator[SourceItem]:
        """GREYSKY DNG 'before' raws (TIER-1). Decoded via rawpy. Whole-tree search
        because DNG can be misfiled inside the XMP folder (probe_GREYSKY gotcha c)."""
        root = self._corpus_path("greysky")
        if not root or not root.is_dir():
            return
        for f in _iter_files(root):
            if self._is_junk_path(f, root):
                continue
            if _ext(f) == ".dng":
                sz = _safe_stat_size(f)
                # GREYSKY DNGs are genuine raws (multi-MB), unlike 全店素材 preset-dngs.
                if sz is not None and sz < 2_000_000:
                    continue
                jpg, xmp = _greysky_siblings(f)
                meta: Dict[str, Any] = {"collection": f.parent.parent.name}
                if jpg:  # the real expert JPG = the gold 'after' (S5 real-JPG bypass)
                    meta["expert_after_jpg"] = jpg
                if xmp:
                    meta["expert_xmp"] = xmp
                yield self._emit_source(
                    f, "greysky", scene="any", raw_decode=RawDecode.RAWPY,
                    tags=["tier1_gold"], size=sz, meta=meta,
                )

    def _scan_korean(self) -> Iterator[SourceItem]:
        """211 Korean portraits (~1312 jpg). The dir name in config may be truncated;
        glob the real one under presets/ as a fallback (probe_GREYSKY §4)."""
        root = self._corpus_path("korean_portraits")
        if root is None or not root.is_dir():
            # fallback: glob 211* under the presets dir
            presets_dir = Path(_PRESET_V1_PATH_ROOT)
            matches = sorted(presets_dir.glob("211*")) if presets_dir.is_dir() else []
            root = matches[0] if matches else None
        if root is None or not root.is_dir():
            return
        for f in _iter_files(root):
            if _ext(f) in IMAGE_EXTS and not self._is_junk_path(f, root):
                sz = _safe_stat_size(f)
                if sz is not None and sz < self.min_source_bytes:
                    continue
                yield self._emit_source(f, "korean", scene="portrait",
                                        is_portrait_pool=True, size=sz)

    def _scan_quandian_sources(self) -> Iterator[SourceItem]:
        """全店素材 SOURCE images: curated portrait pools (H033/H100/H103) + genuine
        RAW (6 CR2 + 2 ARW). Non-junk jpg/png >min_bytes; preset-dngs are NOT sources
        (probe_全店素材 G2)."""
        c = self.sources_cfg.get("quandian") or {}
        root = c.get("path")
        if not root or not os.path.isdir(root):
            return
        portrait_pools = set(c.get("portrait_pools") or [])
        rootp = Path(root)
        for f in _iter_files(rootp):
            ext = _ext(f)
            if self._is_junk_path(f, rootp):
                continue
            pack_id = self._pack_id_for(f, rootp)
            is_portrait = self._pack_is_portrait(pack_id, portrait_pools)
            if ext in IMAGE_EXTS:
                sz = _safe_stat_size(f)
                if sz is None or sz < self.min_source_bytes:
                    continue
                item = self._emit_source(
                    f, "quandian",
                    scene="portrait" if is_portrait else "any",
                    is_portrait_pool=is_portrait, size=sz,
                    meta={"pack_id": pack_id},
                )
                self._pack_add(pack_id, rootp, "source_images", str(f))
                yield item
            elif ext in (".cr2", ".arw"):
                # genuine source RAW (rare). dng excluded: those are presets (G2).
                yield self._emit_source(f, "quandian",
                                        scene="portrait" if is_portrait else "any",
                                        is_portrait_pool=is_portrait,
                                        raw_decode=RawDecode.RAWPY,
                                        meta={"pack_id": pack_id})

    def _scan_awards(self) -> Iterator[SourceItem]:
        """Award photographers: double-nested .rar/.zip (~43G). Register archive
        pointers only; streams extract lazily with 7z (probe_GREYSKY §3)."""
        root = self._corpus_path("awards")
        if not root or not root.is_dir():
            return
        for f in _iter_files(root):
            if _ext(f) in (".rar", ".zip"):
                yield self._emit_source(
                    f, "awards", scene="any",
                    meta={"archive": True, "extract": "7z_two_step",
                          "note": "double-nested; 7z x outer then inner, extract on demand"},
                )

    def _scan_mmart(self) -> Iterator[SourceItem]:
        """MMArt: 4055 before.jpg under global/<id>/. Stale grpo paths rebase to local
        (probe_sources §1a / gotcha 4). One SourceItem per unique before.jpg."""
        c = self.sources_cfg.get("mmart") or {}
        image_root = c.get("image_root")
        if not image_root or not os.path.isdir(image_root):
            return
        rootp = Path(image_root)
        for entry_dir in sorted(p for p in rootp.iterdir() if p.is_dir()):
            before = entry_dir / "before.jpg"
            if before.is_file():
                yield self._emit_source(
                    before, "mmart", scene="any",
                    meta={"mmart_id": entry_dir.name,
                          "processed": str(entry_dir / "processed.jpg"),
                          "config_xmp": str(entry_dir / "config.xmp")},
                )

    def _scan_unsplash(self) -> Iterator[SourceItem]:
        """Unsplash-lite is a URL CSV (no pixels). Register the downloaded slice dir
        if present; otherwise skip (download is deferred, off pilot critical path)."""
        c = self.sources_cfg.get("unsplash") or {}
        scratch = self.cfg.get("scratch_dir") or ""
        dl_dir = os.path.join(scratch, "unsplash") if scratch else None
        if dl_dir and os.path.isdir(dl_dir):
            for f in _iter_files(Path(dl_dir)):
                if _ext(f) in IMAGE_EXTS:
                    yield self._emit_source(f, "unsplash", scene=c.get("scene", "any"))

    def _scan_fivek_gold(self) -> Iterator[SourceItem]:
        """fivek GOLD GLOBAL pairs (S8): real before.jpg -> real expert after.jpg.

        Layout (verified): <root>/train_global/<sample_id>/{before.jpg, processed.jpg,
        config.lua, meta.json, en/user_want_{short,middle,long}/user_prompt.txt}.

        ONLY ``train_global`` is scanned. ``test_global`` (5000) is the EVAL HOLDOUT
        and is never scanned here (see config sources.fivek_gold comment). Each usable
        sample dir -> one SourceItem (corpus="fivek_gold") with the real expert JPG as
        the gold 'after' (after_source=REAL_JPG, like S5 Tier1ExpertStream). The
        old-Lightroom config.lua params are NOT in the teacher CRS2012 space -> stored
        as metadata (``fivek_params_lua`` path) ONLY, never used as teacher params.
        """
        c = self.sources_cfg.get("fivek_gold") or {}
        root = c.get("path")
        if not root or not os.path.isdir(root):
            return
        rootp = Path(root)
        for entry_dir in sorted(p for p in rootp.iterdir() if p.is_dir()):
            before = entry_dir / "before.jpg"
            processed = entry_dir / "processed.jpg"
            if not (before.is_file() and processed.is_file()):
                continue
            meta_json: Dict[str, Any] = {}
            mp = entry_dir / "meta.json"
            if mp.is_file():
                try:
                    with open(mp, "r", encoding="utf-8") as f:
                        meta_json = json.load(f)
                except (OSError, json.JSONDecodeError):
                    meta_json = {}
            instr = self._read_text(entry_dir / "en" / "user_want_middle" / "user_prompt.txt")
            instr_short = self._read_text(entry_dir / "en" / "user_want_short" / "user_prompt.txt")
            yield self._emit_source(
                before, "fivek_gold", scene="any", is_portrait_pool=False,
                tags=["gold", "fivek"],
                meta={
                    "expert_after_jpg": str(processed),
                    "expert": meta_json.get("expert", "C"),
                    "instruction": instr,
                    "instruction_short": instr_short,
                    "fivek_params_lua": str(entry_dir / "config.lua"),
                    "base_name": meta_json.get("base_name", entry_dir.name),
                    "split": "train_global",
                    "tier": "gold",
                    "expert_flag": True,
                },
            )

    @staticmethod
    def _read_text(path: Path) -> Optional[str]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except OSError:
            return None

    def _scan_ppr10k(self) -> Iterator[SourceItem]:
        """PPR10K region-local portrait PARAM sources (S4): real human masks + the
        per-source target XMP (teacher-manifold CRS2012 params).

        Layout (verified): <root>/{source/<id>.png, target_{a,b,c}/<id>.png,
        xmp/target_{a,b,c}/<id>.xmp, masks/360p/masks_360p/<orig_base>.png,
        manifests/id_map.csv}. The mask filename is keyed by ``orig_base`` (id_map.csv
        new_id -> orig_base), NOT the 4-digit <id>; pattern is
        ``masks/360p/masks_360p/<orig_base>.png`` (one human mask per source). All 3
        experts (a/b/c) are emitted per source -> ~8875 x 3 = 26,625 region-local
        items. C_GT = the real human mask (mask_quality=1.0, mask_source=PPR10K).
        """
        c = self.sources_cfg.get("ppr10k") or {}
        root = c.get("path")
        if not root or not os.path.isdir(root):
            return
        rootp = Path(root)
        mask_dir = rootp / "masks" / "360p" / "masks_360p"
        # id_map.csv: new_id -> orig_base (mask filename stem).
        id_to_base: Dict[str, str] = {}
        id_map_path = rootp / "manifests" / "id_map.csv"
        if id_map_path.is_file():
            import csv as _csv

            with open(id_map_path, "r", encoding="utf-8") as f:
                for row in _csv.DictReader(f):
                    nid = (row.get("new_id") or "").strip()
                    ob = (row.get("orig_base") or "").strip()
                    if nid and ob:
                        id_to_base[nid] = ob
        src_dir = rootp / "source"
        if not src_dir.is_dir():
            return
        for src_png in sorted(src_dir.glob("*.png")):
            sid = src_png.stem
            orig_base = id_to_base.get(sid)
            if not orig_base:
                continue
            mask_path = mask_dir / f"{orig_base}.png"
            if not mask_path.is_file():
                continue
            for e in ("a", "b", "c"):
                xmp_path = rootp / "xmp" / f"target_{e}" / f"{sid}.xmp"
                target_path = rootp / "target_{}".format(e) / f"{sid}.png"
                if not xmp_path.is_file():
                    continue
                yield SourceItem(
                    source_id=f"ppr10k_{sid}_{e}",
                    path=str(src_png),
                    corpus="ppr10k",
                    raw_decode=RawDecode.NONE,
                    scene="portrait",
                    is_portrait_pool=True,
                    tags=["ppr10k", "portrait_human_mask"],
                    bytes_size=_safe_stat_size(src_png),
                    meta={
                        "ppr10k_mask": str(mask_path),
                        "ppr10k_xmp": str(xmp_path),
                        "ppr10k_target": str(target_path),
                        "expert": e,
                        "orig_base": orig_base,
                    },
                )

    # ================================================================== #
    # RECIPES
    # ================================================================== #

    def scan_recipes(self) -> Iterator[RecipeAsset]:
        """Yield deduped, filtered RecipeAssets from 全店素材 + E18 + GREYSKY.

        Dedup-by-preset-stem (G3): the first recipe seen for a given (stem) wins; XMP
        is preferred as canonical, so we scan XMP-bearing corpora before cube banks.
        preset_dataset_v1 is NOT emitted as a recipe (scene_matching role only); it is
        consumed via the manifest catalog to tag the recipes that ARE emitted.
        """
        seen_stems: set = set()
        scanners = (
            self._scan_greysky_recipes,     # XMP, gold
            self._scan_quandian_recipes,    # xmp + lrtemplate + cube
            self._scan_e18_recipes,         # cube
        )
        for fn in scanners:
            n = 0
            for asset in fn():
                if self.cfg.get("storage", {}).get("dedup_preset_stem", True):
                    stem = _preset_stem(Path(asset.path))
                    key = (stem, asset.fmt if asset.kind == RecipeKind.LUT else "param")
                    # collapse cross-format param dupes by stem; keep LUTs distinct by stem+fmt
                    dedup_key = stem if asset.kind == RecipeKind.PARAM else key
                    if dedup_key in seen_stems:
                        continue
                    seen_stems.add(dedup_key)
                yield asset
                n += 1
                if self.limit_per_corpus is not None and n >= self.limit_per_corpus:
                    break

    # ---- recipe helpers -------------------------------------------------- #

    def _is_junk_path(self, path: Path, root: Optional[Path] = None) -> bool:
        """Junk if the ext is junk, OR if the junk path-regex matches the path
        BELOW the corpus root. We exclude the corpus root's own name from the
        path-regex test because a legitimate product folder name can collide with
        a junk token — e.g. E18's `《300+广告级LUT》` ('ad-GRADE LUT') contains
        `广告` which the tutorial/ad-link regex would otherwise flag. Junk lives in
        SUBdirs (tutorials/installers/ad-links), so testing the relative path is
        both correct and avoids that false positive."""
        if _ext(path) in self.junk_ext:
            return True
        test_str = str(path)
        if root is not None:
            try:
                test_str = str(path.relative_to(root))
            except ValueError:
                test_str = str(path)
        if self.junk_path_re.search(test_str):
            return True
        return False

    def _is_bw_xmp(self, params: Optional[Dict[str, Dict[str, float]]],
                   raw_text: Optional[str]) -> bool:
        if not self.drop_bw:
            return False
        if raw_text and 'ConvertToGrayscale="True"' in raw_text:
            return True
        return False

    def _parse_params(self, path: Path) -> Tuple[Optional[Dict[str, Dict[str, float]]], Optional[str]]:
        """Parse an XMP into VeraRetouch param dict via the reused parser.
        Returns (param_dict | None, raw_b&w_flag_text | None). Lazy import; if the
        reused parser is unavailable or parsing fails, returns (None, None)."""
        if not self.parse_recipe_params:
            return None, None
        if _ext(path) != ".xmp":
            # lrtemplate parsing is deferred to recipes.py (Lua parser); we only tag here.
            return None, None
        try:
            from dataset_build.recipes import xmp_core_settings_to_params  # type: ignore
        except Exception:
            xmp_core_settings_to_params = None  # recipes.py may not exist yet
        try:
            sys.path.insert(0, "/home/bc/retouching/presets/scripts")
            import build_preset_dataset as bpd  # type: ignore
            rec = bpd.parse_xmp_file(path)
            core = dict(rec.core_settings)
        except Exception:
            return None, None
        # Detect B&W from the raw core settings (probe_全店素材 G7).
        bw = core.get("ConvertToGrayscale", "").strip().lower() == "true"
        params = None
        if xmp_core_settings_to_params is not None:
            try:
                params = xmp_core_settings_to_params(core)
            except Exception:
                params = None
        return params, ('ConvertToGrayscale="True"' if bw else "")

    def _emit_recipe(self, path: Path, *, pack_id: Optional[str], scene_affinity: str,
                     style: Optional[str], is_bw: bool, is_technical: bool,
                     has_local_mask: bool, lut_size: Optional[int],
                     tags: Optional[List[str]] = None,
                     meta: Optional[Dict[str, Any]] = None) -> RecipeAsset:
        ext = _ext(path).lstrip(".")
        kind = _FMT_TO_KIND.get(ext, RecipeKind.PARAM)
        sz = _safe_stat_size(path)
        return RecipeAsset(
            recipe_id=_stable_id(str(path), sz, prefix="rcp_"),
            path=str(path),
            kind=kind,
            fmt=ext,
            pack_id=pack_id,
            style=style,
            scene_affinity=scene_affinity,
            is_bw=is_bw,
            is_technical=is_technical,
            has_local_mask=has_local_mask,
            lut_size=lut_size,
            tags=tags or [],
            meta=meta or {},
        )

    @staticmethod
    def _peek_cube_size(path: Path) -> Optional[int]:
        """Cheap header read for LUT_3D_SIZE / .3dl Mesh size, without a full parse."""
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for _ in range(40):
                    line = f.readline()
                    if not line:
                        break
                    m = re.match(r"\s*LUT_3D_SIZE\s+(\d+)", line, re.IGNORECASE)
                    if m:
                        return int(m.group(1))
                    m = re.match(r"\s*Mesh\s+(\d+)\s+(\d+)", line, re.IGNORECASE)
                    if m:
                        # 3dl: 17^3 grid typical; size not directly in header -> heuristic 17
                        return 17
        except OSError:
            return None
        return None

    @staticmethod
    def _xmp_has_local_mask(path: Path) -> bool:
        """True if the XMP encodes a spatial region (probe_全店素材 §2 local-mask)."""
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read(200_000)
        except OSError:
            return False
        return any(k in txt for k in (
            "MaskGroupBasedCorrections", "PaintBasedCorrections",
            "GradientBasedCorrections", "CircularGradientBasedCorrections",
        ))

    # ---- per-corpus recipe scanners -------------------------------------- #

    def _scan_greysky_recipes(self) -> Iterator[RecipeAsset]:
        root = self._corpus_path("greysky")
        if root is None:
            gc = self.recipes_cfg.get("greysky_xmp") or {}
            root = Path(gc["path"]) if gc.get("path") else None
        if root is None or not root.is_dir():
            return
        for f in _iter_files(root):
            if _ext(f) != ".xmp" or self._is_junk_path(f, root):
                continue
            params, bw_text = self._parse_params(f)
            is_bw = self._is_bw_xmp(params, bw_text)
            is_tech = bool(self.technical_re.search(f.name))
            if is_bw or is_tech:
                continue
            scene_aff = self._scene_affinity_from_text(f.name, str(f.parent))
            yield self._emit_recipe(
                f, pack_id="GREYSKY", scene_affinity=scene_aff, style=None,
                is_bw=is_bw, is_technical=is_tech,
                has_local_mask=self._xmp_has_local_mask(f), lut_size=None,
                tags=["tier1_gold"],
                meta={"params": params} if params else {},
            )

    def _scan_quandian_recipes(self) -> Iterator[RecipeAsset]:
        c = self.recipes_cfg.get("quandian") or {}
        root = c.get("path")
        if not root or not os.path.isdir(root):
            return
        rootp = Path(root)
        for f in _iter_files(rootp):
            ext = _ext(f)
            if ext not in RECIPE_EXTS or self._is_junk_path(f, rootp):
                continue
            pack_id = self._pack_id_for(f, rootp)
            mtags = self._manifest_tags(str(f))
            is_tech = bool(self.technical_re.search(str(f)))
            scene_aff = self._scene_affinity_from_text(
                mtags.get("scene_type"), mtags.get("subject_type"),
                mtags.get("style_family"), f.name, str(f.parent),
            )
            if mtags.get("skin_friendly"):
                scene_aff = "portrait"
            lut_size = None
            has_local = False
            is_bw = False
            params = None
            if ext in (".xmp",):
                params, bw_text = self._parse_params(f)
                is_bw = self._is_bw_xmp(params, bw_text)
                has_local = self._xmp_has_local_mask(f)
            elif ext in (".cube", ".3dl"):
                lut_size = self._peek_cube_size(f)
            if is_bw:
                continue
            tags = []
            if mtags.get("is_duplicate"):
                tags.append("manifest_dup")
            asset = self._emit_recipe(
                f, pack_id=pack_id, scene_affinity=scene_aff,
                style=mtags.get("style_family") or mtags.get("name_en"),
                is_bw=is_bw, is_technical=is_tech, has_local_mask=has_local,
                lut_size=lut_size, tags=tags,
                meta={"manifest": mtags or None, "params": params} if (mtags or params) else {},
            )
            self._pack_add(pack_id, rootp, "recipes", str(f), ext=ext, asset=asset)
            yield asset

    def _scan_e18_recipes(self) -> Iterator[RecipeAsset]:
        c = self.recipes_cfg.get("e18_luts") or {}
        root = c.get("path")
        if not root or not os.path.isdir(root):
            return
        rootp = Path(root)
        for f in _iter_files(rootp):
            ext = _ext(f)
            if ext not in (".cube", ".3dl") or self._is_junk_path(f, rootp):
                continue
            is_tech = bool(self.technical_re.search(f.name))
            if is_tech:
                continue
            scene_aff = self._scene_affinity_from_text(f.name, str(f.parent))
            yield self._emit_recipe(
                f, pack_id="E18", scene_affinity=scene_aff, style=None,
                is_bw=False, is_technical=is_tech, has_local_mask=False,
                lut_size=self._peek_cube_size(f), tags=["cinematic_lut"],
            )

    # ------------------------------------------------------------------ #
    # Pack-index bookkeeping (probe_全店素材 §6)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _pack_id_for(path: Path, root: Path) -> Optional[str]:
        """Top-level pack folder under 全店素材 (e.g. 'H010 ...') -> 'H010'."""
        try:
            rel = path.relative_to(root)
        except ValueError:
            return None
        if not rel.parts:
            return None
        top = rel.parts[0]
        m = re.match(r"(H\d{2,3}(?:-\d+)?)", top)
        return m.group(1) if m else top

    @staticmethod
    def _pack_is_portrait(pack_id: Optional[str], portrait_pools: set) -> bool:
        if not pack_id:
            return False
        base = pack_id.split("-")[0]
        return base in portrait_pools or pack_id in portrait_pools

    def _pack_add(self, pack_id: Optional[str], root: Path, bucket: str, abs_path: str,
                  *, ext: Optional[str] = None, asset: Optional[RecipeAsset] = None) -> None:
        if not pack_id:
            return
        row = self._pack_rows.setdefault(pack_id, {
            "pack_id": pack_id,
            "abs_path": str(root / pack_id) if (root / pack_id).exists() else None,
            "recipes": {"xmp": [], "lrtemplate": [], "cube": [], "3dl": []},
            "recipe_counts": {"xmp": 0, "cube": 0, "3dl": 0, "lrtemplate": 0,
                              "local_mask": 0, "bw": 0, "technical": 0},
            "source_images": [],
        })
        if bucket == "source_images":
            if len(row["source_images"]) < 50:  # cap stored list; counts are exact below
                row["source_images"].append(abs_path)
            row.setdefault("source_count", 0)
            row["source_count"] += 1
        elif bucket == "recipes" and ext:
            k = ext.lstrip(".")
            if k in row["recipes"] and len(row["recipes"][k]) < 200:
                row["recipes"][k].append(abs_path)
            if k in row["recipe_counts"]:
                row["recipe_counts"][k] += 1
            if asset is not None:
                if asset.has_local_mask:
                    row["recipe_counts"]["local_mask"] += 1
                if asset.is_technical:
                    row["recipe_counts"]["technical"] += 1

    # ================================================================== #
    # WRITE + LOAD
    # ================================================================== #

    def write_indexes(self, out_dir: str) -> Dict[str, int]:
        """Scan everything and write the three indexes atomically. Returns counts."""
        os.makedirs(out_dir, exist_ok=True)
        storage = self.cfg.get("storage", {}) or {}
        src_name = storage.get("source_index", "source_index.jsonl")
        rcp_name = storage.get("recipe_index", "recipe_index.jsonl")
        pack_name = storage.get("pack_index", "pack_index.jsonl")

        counts = {"sources": 0, "recipes": 0, "packs": 0,
                  "recipes_param": 0, "recipes_lut": 0,
                  "recipes_portrait": 0, "recipes_landscape": 0,
                  "recipes_local_mask": 0, "sources_portrait_pool": 0}

        # Recipes first so pack rows (built during quandian source+recipe scan) are
        # populated; sources are scanned in the same pass below.
        rcp_path = os.path.join(out_dir, rcp_name)
        with _AtomicWriter(rcp_path) as w:
            for asset in self.scan_recipes():
                w.write(json.dumps(_asdict_jsonable(asset), ensure_ascii=False) + "\n")
                counts["recipes"] += 1
                if asset.kind == RecipeKind.PARAM:
                    counts["recipes_param"] += 1
                else:
                    counts["recipes_lut"] += 1
                if asset.scene_affinity == "portrait":
                    counts["recipes_portrait"] += 1
                elif asset.scene_affinity == "landscape":
                    counts["recipes_landscape"] += 1
                if asset.has_local_mask:
                    counts["recipes_local_mask"] += 1

        src_path = os.path.join(out_dir, src_name)
        with _AtomicWriter(src_path) as w:
            for item in self.scan_sources():
                w.write(json.dumps(_asdict_jsonable(item), ensure_ascii=False) + "\n")
                counts["sources"] += 1
                if item.is_portrait_pool:
                    counts["sources_portrait_pool"] += 1

        pack_path = os.path.join(out_dir, pack_name)
        with _AtomicWriter(pack_path) as w:
            for row in self._pack_rows.values():
                w.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts["packs"] += 1

        return counts

    def write_greysky_index(self, out_dir: str, name: str = "greysky_index.jsonl") -> Dict[str, int]:
        """Write ONLY the GREYSKY sources — with the DNG->expert-JPG/XMP pairing — to
        a SEPARATE index file. Deliberately does NOT touch the shared
        source_index.jsonl that a live build is reading (isolation; the S5 stream
        reads this file when present, see run.load_plan_inputs)."""
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, name)
        n = paired = 0
        with _AtomicWriter(path) as w:
            for item in self._scan_greysky_sources():
                w.write(json.dumps(_asdict_jsonable(item), ensure_ascii=False) + "\n")
                n += 1
                if (item.meta or {}).get("expert_after_jpg"):
                    paired += 1
        return {"greysky_sources": n, "with_expert_jpg": paired, "path": path}

    def load_sources(self, index_path: str,
                     corpus: Optional[str] = None,
                     scene: Optional[str] = None) -> List[SourceItem]:
        out: List[SourceItem] = []
        for rec in _read_jsonl(index_path):
            if corpus is not None and rec.get("corpus") != corpus:
                continue
            if scene is not None and rec.get("scene") not in (scene, "any", None):
                continue
            out.append(_source_from_dict(rec))
        return out

    def load_recipes(self, index_path: str,
                     kind: Optional[RecipeKind] = None,
                     scene_affinity: Optional[str] = None) -> List[RecipeAsset]:
        kind_val = kind.value if isinstance(kind, RecipeKind) else kind
        out: List[RecipeAsset] = []
        for rec in _read_jsonl(index_path):
            if kind_val is not None and rec.get("kind") != kind_val:
                continue
            if scene_affinity is not None:
                aff = rec.get("scene_affinity")
                # 'any' recipes match any requested affinity (soft routing).
                if aff not in (scene_affinity, "any", None):
                    continue
            out.append(_recipe_from_dict(rec))
        return out


# ---------------------------------------------------------------------------
# (De)serialization helpers — keep enums as their .value, dataclasses as dicts.
# ---------------------------------------------------------------------------


def _asdict_jsonable(obj: Any) -> Dict[str, Any]:
    d = dataclasses.asdict(obj)
    for k, v in list(d.items()):
        if hasattr(v, "value"):  # Enum
            d[k] = v.value
    return d


def _read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _source_from_dict(rec: Dict[str, Any]) -> SourceItem:
    return SourceItem(
        source_id=rec["source_id"],
        path=rec["path"],
        corpus=rec["corpus"],
        raw_decode=RawDecode(rec.get("raw_decode", "none")),
        width=rec.get("width"),
        height=rec.get("height"),
        scene=rec.get("scene"),
        is_portrait_pool=rec.get("is_portrait_pool", False),
        tags=rec.get("tags", []),
        bytes_size=rec.get("bytes_size"),
        meta=rec.get("meta", {}),
    )


def _recipe_from_dict(rec: Dict[str, Any]) -> RecipeAsset:
    return RecipeAsset(
        recipe_id=rec["recipe_id"],
        path=rec["path"],
        kind=RecipeKind(rec["kind"]),
        fmt=rec["fmt"],
        pack_id=rec.get("pack_id"),
        style=rec.get("style"),
        scene_affinity=rec.get("scene_affinity"),
        is_bw=rec.get("is_bw", False),
        is_technical=rec.get("is_technical", False),
        has_local_mask=rec.get("has_local_mask", False),
        lut_size=rec.get("lut_size"),
        tags=rec.get("tags", []),
        meta=rec.get("meta", {}),
    )


class _AtomicWriter:
    """Context manager: write to <path>.tmp then fsync+rename to <path>
    (config.storage.atomic_write)."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.tmp = path + ".tmp"
        self._fh = None

    def __enter__(self) -> "_AtomicWriter":
        self._fh = open(self.tmp, "w", encoding="utf-8")
        return self

    def write(self, s: str) -> None:
        assert self._fh is not None
        self._fh.write(s)

    def __exit__(self, exc_type, exc, tb) -> None:
        assert self._fh is not None
        if exc_type is None:
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._fh.close()
            os.replace(self.tmp, self.path)
        else:
            self._fh.close()
            try:
                os.remove(self.tmp)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Scan on-disk corpora into the source/recipe/pack indexes.")
    p.add_argument("--config", default="dataset_build/config.yaml", help="Path to config.yaml")
    p.add_argument("--out-dir", default=None,
                   help="Output dir for the indexes (defaults to config out_root)")
    p.add_argument("--limit-per-corpus", type=int, default=None,
                   help="Cap items scanned per corpus (smoke test).")
    p.add_argument("--no-recipe-parse", action="store_true",
                   help="Skip XMP->params parsing (faster scan; params filled later).")
    p.add_argument("--greysky-index", action="store_true",
                   help="Write ONLY greysky_index.jsonl (DNG->expert JPG/XMP pairing for S5 "
                        "real-JPG); does not touch the shared source_index.jsonl.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    cfg = _load_yaml(args.config)
    out_dir = args.out_dir or cfg.get("out_root")
    if not out_dir:
        print("ERROR: no --out-dir and no out_root in config", file=sys.stderr)
        return 2
    reg = SourceRegistry(
        cfg,
        parse_recipe_params=not args.no_recipe_parse,
        limit_per_corpus=args.limit_per_corpus,
    )
    if getattr(args, "greysky_index", False):
        # Isolated: write only greysky_index.jsonl (DNG->expert JPG/XMP pairing for
        # S5 real-JPG). Never rewrites the shared source_index.jsonl a live run reads.
        counts = reg.write_greysky_index(out_dir)
        print(json.dumps({"out_dir": out_dir, "greysky": counts}, ensure_ascii=False, indent=2))
        return 0
    counts = reg.write_indexes(out_dir)
    print(json.dumps({"out_dir": out_dir, "counts": counts}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
