"""
dataset_build/run.py
====================
Sharded, RESUMABLE orchestrator that wires registry -> streams -> pack for the
VeraRetouch Direction-A, RECIPE-BASED 1,000,000 dataset build.

It is the data-generation entrypoint named in DATASET_BUILD_PLAN.md §9
(``python -m dataset_build.run`` / wired by ``run_build``). Flags:

  --config PATH        config.yaml (default dataset_build/config.yaml)
  --stream S1[,S2,...] which stream(s) to build (default: all enabled)
  --pilot [N]          pilot mode; per-stream budgets from config.pilot
                       (or scale the full budgets to total N if given)
  --full               full 1M run (config.budget)
  --dry-run            use CPU MockModels (NO weights, NO vLLM); validates the
                       full DAG + sharding + C_GT writing end-to-end
  --shard i/n          partition the per-stream work into n slices, build slice i
  --resume             skip sample_ids already committed to shards
  --out-suffix STR     append to out_root (e.g. "_pilot")
  --limit N            hard cap on samples per stream (debug)

CRITICAL (subagent rule): this module LOADS NO WEIGHTS at import time and the
``--dry-run`` path loads none at all. Real model loading happens only when the
main process passes neither ``--dry-run`` nor a stubbed config, and even then the
heavy classes (VeraParamRenderer / Sam3ConceptMasker / QwenVLCleaner /
RecipeParserImpl / Registry) are imported LAZILY inside ``_build_real_models``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import shutil
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# yaml is the only top-level third-party dep; degrade gracefully if absent.
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from dataset_build.contracts import (
    DegradeSpec,
    PARAM_KEYS,
    RawDecode,
    RecipeAsset,
    RecipeKind,
    Sample,
    SourceItem,
    StreamId,
)
from dataset_build.pack import ShardWriter, validate_manifest_index
from dataset_build.streams import (
    BuildContext,
    QAGate,
    _PlanInputs,
    make_stream,
)

DEFAULT_CONFIG = str(Path(__file__).with_name("config.yaml"))


def _mem_avail_mb() -> int:
    """Available system RAM in MB (for progress logging / OOM diagnosis)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return -1


# ===========================================================================
# Config loading.
# ===========================================================================


def load_config(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("pyyaml is required to load config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_out_root(config: Dict[str, Any], suffix: str = "") -> str:
    root = config.get("out_root", "./vera_directionA_1M")
    return root + suffix if suffix else root


# ===========================================================================
# Mock models for --dry-run (CPU only, no weights, satisfy the duck types).
# ===========================================================================


class MockRenderer:
    """Returns a deterministic synthetic 'after' (tiny solid-color tile) so the
    DAG runs with no GPU. Never used for real data."""

    def render(self, image_paths, param_dicts, batch_size=4, chunk=262144, max_new_tokens=256):
        import numpy as np

        outs = []
        for _ in image_paths:
            outs.append(np.full((8, 8, 3), 128, dtype="uint8"))
        return outs


class MockMasker:
    """Returns a centered soft blob mask of a fixed small size (native_size if
    given), enough to exercise composite + C_GT + coverage gate."""

    def _blob(self, native):
        import numpy as np

        H, W = native if native else (64, 64)
        yy, xx = np.mgrid[0:H, 0:W].astype("float32")
        cy, cx = H * 0.5, W * 0.5
        r = 0.3 * max(H, W)
        d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        return np.clip(1.0 - d / (r + 1e-6), 0.0, 1.0).astype("float32")

    def mask(self, image, concept, native_size=None, soft=True, reduce="max", min_score=0.3):
        return self._blob(native_size)

    def masks(self, image, concepts, **kw):
        native = kw.get("native_size")
        return {c: self._blob(native) for c in concepts}

    def cgt_aspect_stack(self, image, concept_map):
        import numpy as np

        return np.stack([self._blob(None) for _ in range(3)], axis=0)


class MockCleaner:
    """Synthesizes plausible annotations + passing judge scores offline."""

    def gen_instruction(self, image_path, scene_meta=None):
        scene = (scene_meta or {}).get("scene", "photo")
        return {
            "instruction_long": f"Enhance this {scene} with a balanced, natural edit.",
            "instruction_short": f"balanced {scene} edit",
            "lang": "en",
        }

    def reason_params(self, image_path, instruction):
        return {"think": "Adjust exposure and color for balance.", "answer": {}}

    def verify(self, before_path, after_path, params):
        return {
            "look_match": True,
            "param_sane": True,
            "processed_ok": True,
            "score": 0.85,
            "reason": "mock pass",
        }

    def tag_scene_region(self, image_path, instruction):
        return {
            "scene": "any",
            "style": "balanced",
            "region_local": True,
            "sam3_concepts": ["the main subject"],
            "groundingdino_prompt": "the main subject",
            "masksubtype_hint": 1,
        }


class MockParser:
    """Returns identity params / a minimal degrade spec; no file parsing."""

    def xmp_to_params(self, xmp_path):
        return {k: {"value": 0.0} for k in PARAM_KEYS}

    def lrtemplate_to_params(self, path):
        return {k: {"value": 0.0} for k in PARAM_KEYS}

    def sample_degrade_spec(self, aspects, seed, region_local):
        # a small, valid Gaussian-op spec touching one L key for non-trivial magnitude
        op = {"Exposure2012": 25.0} if "L" in aspects else {"Vibrance": 20.0}
        return DegradeSpec(
            mode="gaussian_op",
            op_params=op,
            sigma_profile="aether_tab8",
            aspects=list(aspects),
            forward=False,
            seed=seed,
        )

    def load_cube(self, path):
        from dataset_build.recipes import identity_lut

        return identity_lut(2), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)


class MockLutApplier:
    def apply_lut(self, img, lut, domain_min=(0, 0, 0), domain_max=(1, 1, 1)):
        return img


def build_mock_context(config: Dict[str, Any], writer: ShardWriter) -> BuildContext:
    return BuildContext(
        config=config,
        renderer=MockRenderer(),
        masker=MockMasker(),
        cleaner=MockCleaner(),
        parser=MockParser(),
        lut_applier=MockLutApplier(),
        cgt_writer=writer,
    )


# ===========================================================================
# Real model + index loading (LAZY; main process only).
# ===========================================================================


def _stream_needs_renderer(stream_id: StreamId, config: Dict[str, Any]) -> bool:
    """Whether this stream can consume the VeraRetouch teacher renderer."""
    if stream_id in (StreamId.S1_DEGRADE_LOCAL, StreamId.S7_DEGRADE_GLOBAL):
        qa = config.get("qa", {}) or {}
        return _stream_verify_enabled(stream_id, qa) and _verify_sample_rate(qa) > 0.0
    if stream_id == StreamId.S5_GREYSKY_GLOBAL:
        # Most GREYSKY rows carry real expert JPGs and skip teacher render, but keep
        # the renderer available for rows that only have an XMP.
        return True
    if stream_id in (
        StreamId.S2_RECIPE_LOCAL,
        StreamId.S3_MMART_LOCAL,
        StreamId.S4_PPR10K_LOCAL,
        StreamId.S6_RECIPE_GLOBAL,
    ):
        return True
    return False


def _stream_needs_masker(stream_id: StreamId, config: Dict[str, Any]) -> bool:
    if stream_id in (StreamId.S2_RECIPE_LOCAL, StreamId.S3_MMART_LOCAL, StreamId.S4_PPR10K_LOCAL):
        return True
    return False


def _stream_needs_parser(stream_id: StreamId, config: Dict[str, Any]) -> bool:
    if stream_id in (
        StreamId.S1_DEGRADE_LOCAL,
        StreamId.S2_RECIPE_LOCAL,
        StreamId.S4_PPR10K_LOCAL,
        StreamId.S5_GREYSKY_GLOBAL,
        StreamId.S6_RECIPE_GLOBAL,
        StreamId.S7_DEGRADE_GLOBAL,
    ):
        return True
    return False


def _streams_need_cleaner(stream_ids: Sequence[StreamId], config: Dict[str, Any]) -> bool:
    qa = config.get("qa", {}) or {}
    if bool(qa.get("enable_verify", False)) and _verify_sample_rate(qa) > 0.0:
        # VLM verify still needs the cleaner, but S1/S7's optional build-time
        # render is a teacher preview only; their real z* PSNR gate is Stage-0.
        if any(s not in (StreamId.S1_DEGRADE_LOCAL, StreamId.S7_DEGRADE_GLOBAL) for s in stream_ids):
            return True
    mode = str(qa.get("annotation_mode", "vlm")).strip().lower()
    if mode in {"template", "offline", "deterministic", "none"}:
        return False
    # S1/S7 are synthetic inverse-degradation streams with known target
    # aspect/scope/params. Generic VLM instruction generation would see the clean
    # source image, not the synthetic degradation, so it can reduce label fidelity.
    return any(s not in (StreamId.S1_DEGRADE_LOCAL, StreamId.S7_DEGRADE_GLOBAL) for s in stream_ids)


def _stream_verify_enabled(stream_id: StreamId, qa_cfg: Dict[str, Any]) -> bool:
    if not bool(qa_cfg.get("enable_verify", False)):
        return False
    if stream_id in (StreamId.S1_DEGRADE_LOCAL, StreamId.S7_DEGRADE_GLOBAL):
        return bool(qa_cfg.get("verify_degrade_streams", False))
    return True


def _verify_sample_rate(qa_cfg: Dict[str, Any]) -> float:
    return float(qa_cfg.get("verify_sample_rate", 1.0) or 0.0)


def _build_real_models(
    config: Dict[str, Any],
    stream_ids: Optional[Sequence[StreamId]] = None,
) -> Dict[str, Any]:
    """Lazily import + construct the heavy collaborators. NEVER called by
    subagents / dry-run. Each import is guarded so a missing sibling module
    yields ``None`` for that collaborator (the streams degrade gracefully)."""
    out: Dict[str, Any] = {
        "renderer": None,
        "masker": None,
        "cleaner": None,
        "parser": None,
        "lut_applier": None,
        "cached_tagger": None,
    }

    stream_ids = list(stream_ids or list(StreamId))
    need_renderer = any(_stream_needs_renderer(s, config) for s in stream_ids)
    need_masker = any(_stream_needs_masker(s, config) for s in stream_ids)
    need_cleaner = _streams_need_cleaner(stream_ids, config)
    need_parser = any(_stream_needs_parser(s, config) for s in stream_ids)

    # Concrete sibling class names (read from the as-written modules, 2026-06-01):
    #   render.VeraRetouchRenderer / masking.Sam3Masker /
    #   vlm_clean.QwenVLCleaner / recipes.DiskRecipeParser+DiskLutApplier.
    vr = (config.get("models", {}) or {}).get("veraretouch", {}) or {}
    if need_renderer:
        try:
            from dataset_build.render import VeraRetouchRenderer  # type: ignore

            kw: Dict[str, Any] = {"dtype": vr.get("dtype", "bfloat16"),
                                  "max_new_tokens": int(vr.get("max_new_tokens", 256)),
                                  "greedy": bool(vr.get("greedy", True)),
                                  "num_workers": int(vr.get("num_workers", 0))}
            if vr.get("model_path"):
                kw["model_path"] = vr["model_path"]
            if vr.get("config_add_path"):
                kw["config_add_path"] = vr["config_add_path"]
            out["renderer"] = VeraRetouchRenderer(**kw)
        except Exception as e:  # pragma: no cover - depends on sibling module
            print(f"[run] renderer unavailable: {e}", file=sys.stderr)
    else:
        print("[run] renderer skipped: selected streams do not need teacher render", file=sys.stderr)

    if need_masker:
        try:
            sam = (config.get("models", {}) or {}).get("sam3", {}) or {}
            if sam.get("use_cache") and sam.get("cache_dir"):
                # Base-env safe: read precomputed SAM3 PNGs (no transformers-5.2 import),
                # so the orchestrator can co-exist with the renderer (llava). Wave-2 path.
                from dataset_build.mask_cache import CachedMasker

                out["masker"] = CachedMasker(sam["cache_dir"])
                print(f"[run] masker: CachedMasker({sam['cache_dir']})", file=sys.stderr)
            else:
                from dataset_build.masking import Sam3Masker  # type: ignore

                mkw: Dict[str, Any] = {
                    "score_threshold": float(sam.get("score_threshold", 0.3)),
                    "mask_threshold": float(sam.get("mask_threshold", 0.5)),
                    "concept_map": (config.get("cgt", {}) or {}).get("concept_map"),
                }
                if sam.get("model_dir"):
                    mkw["model_dir"] = sam["model_dir"]
                out["masker"] = Sam3Masker(**mkw)
        except Exception as e:  # pragma: no cover
            print(f"[run] masker unavailable: {e}", file=sys.stderr)
    else:
        print("[run] masker skipped: selected streams do not need SAM3/CachedMasker", file=sys.stderr)

    # Optional PRECOMPUTED source-tag/aesthetic cache (tag_precompute.py). DEFAULT-OFF:
    # only built when tag_cache.use_cache:true; otherwise stays None (no behavior change).
    tc = (config.get("tag_cache", {}) or {})
    if tc.get("use_cache") and tc.get("cache_dir"):
        try:
            from dataset_build.tag_cache import CachedTagger

            out["cached_tagger"] = CachedTagger(tc["cache_dir"])
            print(f"[run] tagger: CachedTagger({tc['cache_dir']})", file=sys.stderr)
        except Exception as e:  # pragma: no cover
            print(f"[run] cached tagger unavailable: {e}", file=sys.stderr)

    if need_cleaner:
        try:
            from dataset_build.vlm_clean import QwenVLCleaner  # type: ignore

            # QwenVLCleaner.from_config reads the vllm: + qa: blocks for us.
            if hasattr(QwenVLCleaner, "from_config"):
                out["cleaner"] = QwenVLCleaner.from_config(config)
            else:
                v = config.get("vllm", {}) or {}
                out["cleaner"] = QwenVLCleaner(
                    base_url=v.get("base_url", "http://localhost:8001/v1"),
                    api_key=v.get("api_key", "EMPTY"),
                    model=v.get("served_model_name", "qwen3-vl-8b"),
                )
        except Exception as e:  # pragma: no cover
            print(f"[run] cleaner unavailable: {e}", file=sys.stderr)
    else:
        print("[run] cleaner skipped: selected streams/config do not need VLM cleaner", file=sys.stderr)

    if need_parser:
        try:
            from dataset_build.recipes import DiskLutApplier, DiskRecipeParser  # type: ignore

            sigma = (config.get("degrade", {}) or {}).get("sigma_profile", "aether_tab8")
            out["parser"] = DiskRecipeParser(sigma_profile=sigma)
            out["lut_applier"] = DiskLutApplier()
        except Exception as e:  # pragma: no cover
            print(f"[run] parser unavailable: {e}", file=sys.stderr)
    else:
        print("[run] parser skipped: selected streams do not need recipe parsing", file=sys.stderr)

    return out


# ===========================================================================
# Index loading (registry outputs) -> per-stream _PlanInputs.
# ===========================================================================


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def _row_to_source(d: Dict[str, Any]) -> SourceItem:
    return SourceItem(
        source_id=d["source_id"],
        path=d["path"],
        corpus=d.get("corpus", "any"),
        raw_decode=RawDecode(d.get("raw_decode", "none")),
        width=d.get("width"),
        height=d.get("height"),
        scene=d.get("scene"),
        is_portrait_pool=bool(d.get("is_portrait_pool", False)),
        tags=list(d.get("tags", []) or []),
        bytes_size=d.get("bytes_size"),
        meta=d.get("meta", {}) or {},
    )


def _row_to_recipe(d: Dict[str, Any]) -> RecipeAsset:
    return RecipeAsset(
        recipe_id=d["recipe_id"],
        path=d["path"],
        kind=RecipeKind(d.get("kind", "param")),
        fmt=d.get("fmt", "xmp"),
        pack_id=d.get("pack_id"),
        style=d.get("style"),
        scene_affinity=d.get("scene_affinity"),
        is_bw=bool(d.get("is_bw", False)),
        is_technical=bool(d.get("is_technical", False)),
        has_local_mask=bool(d.get("has_local_mask", False)),
        lut_size=d.get("lut_size"),
        tags=list(d.get("tags", []) or []),
        meta=d.get("meta", {}) or {},
    )


# corpus pools per stream (DATASET_BUILD_PLAN §2 construction column).
# NOTE: the legacy "fivek" tar corpus (5000 source-only DNGs) still feeds S1/S7 as
# raw 'before' inputs; the NEW gold pair corpus "fivek_gold" (real before->expert
# after) is a DISTINCT name and feeds ONLY S8 -> no collision.
_STREAM_SOURCE_CORPORA: Dict[StreamId, Sequence[str]] = {
    StreamId.S1_DEGRADE_LOCAL: ("tad66k", "fivek", "awards", "korean", "unsplash", "quandian"),
    StreamId.S2_RECIPE_LOCAL: ("tad66k", "awards", "korean", "quandian", "unsplash"),
    StreamId.S3_MMART_LOCAL: ("mmart",),
    StreamId.S4_PPR10K_LOCAL: ("ppr10k",),
    StreamId.S5_GREYSKY_GLOBAL: ("greysky",),
    StreamId.S6_RECIPE_GLOBAL: ("tad66k", "awards", "korean", "quandian", "unsplash"),
    StreamId.S7_DEGRADE_GLOBAL: ("tad66k", "fivek", "awards", "korean", "unsplash", "quandian"),
    StreamId.S8_FIVEK_GLOBAL: ("fivek_gold",),
}
# Streams that draw from the SHARED recipe_index pool. S4 is NOT here: it pairs each
# ppr10k source with its OWN per-source target XMP (source.meta['ppr10k_xmp']), and
# S8 is gold real-JPG (no recipe pool, no LUT, no teacher params).
_RECIPE_STREAMS = {
    StreamId.S2_RECIPE_LOCAL,
    StreamId.S6_RECIPE_GLOBAL,
    StreamId.S5_GREYSKY_GLOBAL,
}


def load_plan_inputs(
    config: Dict[str, Any],
    out_root: str,
    stream_id: StreamId,
    synthetic: bool = False,
) -> _PlanInputs:
    """Resolve the per-stream source/recipe/mmart pools from the registry indexes
    under ``out_root``. In ``--dry-run`` (synthetic) we fabricate tiny in-memory
    pools so the pipeline runs without a registry pass."""
    if synthetic:
        return _synthetic_inputs(stream_id)

    # Registry indexes are build-global: read them from the BASE out_root (where
    # registry.py writes), NOT the per-run suffixed output dir (which only isolates
    # shards/manifests). Otherwise a --out-suffix run finds no sources.
    root = Path(config.get("out_root", out_root))
    storage = config.get("storage", {}) or {}
    sources = [_row_to_source(r) for r in _read_jsonl(root / storage.get("source_index", "source_index.jsonl"))]
    recipes = [_row_to_recipe(r) for r in _read_jsonl(root / storage.get("recipe_index", "recipe_index.jsonl"))]

    corpora = set(_STREAM_SOURCE_CORPORA.get(stream_id, ()))
    src_pool = [s for s in sources if s.corpus in corpora] if corpora else sources

    # S5: prefer the isolated greysky_index.jsonl (carries the DNG->expert JPG/XMP
    # pairing for the real-JPG bypass) when a re-scan has produced it; falls back to
    # the shared source_index greysky rows otherwise.
    if stream_id == StreamId.S5_GREYSKY_GLOBAL:
        gpath = root / "greysky_index.jsonl"
        if gpath.exists():
            gsrc = [_row_to_source(r) for r in _read_jsonl(gpath)]
            if gsrc:
                src_pool = gsrc

    rec_pool: List[RecipeAsset] = []
    if stream_id in _RECIPE_STREAMS:
        rec_pool = recipes
        if stream_id == StreamId.S5_GREYSKY_GLOBAL:
            rec_pool = [r for r in recipes if (r.pack_id or "").startswith("GREYSKY")]

    mmart_records: List[Dict[str, Any]] = []
    if stream_id == StreamId.S3_MMART_LOCAL:
        mmart_records = _read_jsonl(root / "mmart_records.jsonl")

    # Clean-before-build (Phase 3): drop any source/recipe not cleared by
    # source_qa's per-asset verdict (read-only PG predicate). Opt-in
    # (config.cleaning.enabled, default off) and fail-closed (PG error -> HALT),
    # so the build never consumes an un-cleaned asset.
    from dataset_build import cleaning
    src_pool = cleaning.filter_sources(config, stream_id.value, src_pool)
    rec_pool = cleaning.filter_recipes(config, stream_id.value, rec_pool)

    return _PlanInputs(sources=src_pool, recipes=rec_pool, mmart_records=mmart_records)


def _synthetic_inputs(stream_id: StreamId) -> _PlanInputs:
    """A handful of fake sources/recipes/records for --dry-run."""
    srcs = [
        SourceItem(
            source_id=f"src{i:04d}",
            path=f"/tmp/synthetic/source_{i:04d}.jpg",
            corpus="tad66k",
            scene=("portrait" if i % 2 else "landscape"),
            width=64,
            height=64,
        )
        for i in range(64)
    ]
    recs = [
        RecipeAsset(
            recipe_id=f"rec{i:04d}",
            path=f"/tmp/synthetic/look_{i:04d}.xmp",
            kind=(RecipeKind.LUT if i % 3 == 0 else RecipeKind.PARAM),
            fmt=("cube" if i % 3 == 0 else "xmp"),
            pack_id=("GREYSKY/FILM" if i % 5 == 0 else f"H{i:03d}"),
            scene_affinity=("portrait" if i % 2 else "any"),
            lut_size=(33 if i % 3 == 0 else None),
        )
        for i in range(32)
    ]
    mmart = [
        {
            "id": f"mm{i:04d}",
            "source_id": f"src{i:04d}",
            "before_path": f"/tmp/synthetic/source_{i:04d}.jpg",
            "instruction": "Brighten the subject and warm the skin tones.",
            "think": "The subject is underexposed; lift exposure and add warmth.",
            "answer": {"Exposure2012": {"value": 30.0}, "IncrementalTemperature": {"value": 10.0}},
            "masksubtype_hint": 1,
            "scene": "portrait",
        }
        for i in range(16)
    ]
    return _PlanInputs(sources=srcs, recipes=recs, mmart_records=mmart)


# ===========================================================================
# Sharded work partitioning (--shard i/n).
# ===========================================================================


def _parse_shard_flag(flag: Optional[str]) -> Tuple[int, int]:
    if not flag:
        return 0, 1
    try:
        i, n = flag.split("/")
        i, n = int(i), int(n)
        if not (0 <= i < n and n >= 1):
            raise ValueError
        return i, n
    except Exception:
        raise SystemExit(f"--shard must be 'i/n' with 0<=i<n (got {flag!r})")


# ===========================================================================
# Budgets.
# ===========================================================================


def resolve_budgets(
    config: Dict[str, Any], pilot: bool, pilot_total: Optional[int], full: bool
) -> Dict[StreamId, int]:
    budget_cfg = config.get("budget", {}) or {}
    pilot_cfg = config.get("pilot", {}) or {}
    out: Dict[StreamId, int] = {}

    if pilot and pilot_total is None:
        per = pilot_cfg.get("streams", {}) or {}
        for sid in StreamId:
            out[sid] = int(per.get(sid.value, 0))
        return out

    streams = budget_cfg.get("streams", {}) or {}
    full_total = int(budget_cfg.get("total", 1_000_000))
    scale = (pilot_total / full_total) if (pilot and pilot_total) else 1.0
    for sid in StreamId:
        b = streams.get(sid.value, {}) or {}
        out[sid] = int(round(int(b.get("budget", 0)) * scale))
    return out


def apply_s4_fallback(
    budgets: Dict[StreamId, int], config: Dict[str, Any]
) -> Dict[StreamId, int]:
    """If PPR10K masks are unavailable, fold S4's budget into the fallback stream
    (config: budget.s4_fallback_stream; gotcha 3)."""
    ppr = (config.get("models", {}) or {}).get("ppr10k_masks", {}) or {}
    if ppr.get("available", False):
        return budgets
    s4 = budgets.get(StreamId.S4_PPR10K_LOCAL, 0)
    if not s4:
        return budgets
    fb = (config.get("budget", {}) or {}).get("s4_fallback_stream", "S1")
    try:
        fb_id = StreamId(fb)
    except ValueError:
        fb_id = StreamId.S1_DEGRADE_LOCAL
    budgets = dict(budgets)
    budgets[fb_id] = budgets.get(fb_id, 0) + s4
    budgets[StreamId.S4_PPR10K_LOCAL] = 0
    print(f"[run] PPR10K masks unavailable -> folding S4 ({s4}) into {fb_id.value}")
    return budgets


# ===========================================================================
# Per-stream runner.
# ===========================================================================


def run_stream(
    stream_id: StreamId,
    budget: int,
    config: Dict[str, Any],
    ctx: BuildContext,
    inputs: _PlanInputs,
    writer: ShardWriter,
    gate: QAGate,
    shard_i: int = 0,
    shard_n: int = 1,
    resume: bool = False,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Build one stream: plan -> (slice by --shard) -> build_one -> gate -> write.
    Resumable via the writer's committed ids. Returns per-stream counts."""
    if budget <= 0:
        return {"stream": stream_id.value, "planned": 0, "accepted": 0, "rejected": 0}

    stream = make_stream(stream_id, ctx, inputs, gate)
    if inputs.recipes:
        ctx.recipe_assets.update({r.recipe_id: r for r in inputs.recipes})
    seed = int(config.get("seed", 1234))
    done = set(writer.done_ids()) if resume else set()

    accepted = 0
    rejected = 0
    planned = 0
    t0 = time.time()

    # We iterate plan() directly so we can apply --shard slicing + reject logging
    # (Stream.run hides rejects; here we want them in rejects.jsonl).
    from dataset_build.streams import _det_rng, make_sample_id
    from dataset_build.contracts import CgtRef

    # Concurrency model (the batched, overlapped pipeline):
    #   main thread : plan (serial -> deterministic sid) -> prep params -> ONE
    #                 batched GPU render per batch (render_kw.batch_size) under
    #                 ctx.render_lock -> submit each item to the clean pool ->
    #                 drain completed futures and WRITE (ShardWriter on main only).
    #   clean pool  : build_one(precomputed_after=...) = C_GT + annotate + verify
    #                 + gate, GPU-free, vLLM-HTTP-bound, conc workers.
    # While the main thread blocks on the GPU render of batch K+1, the pool runs
    # clean+verify for batch K -> render overlaps clean and the GPU stays busy.
    vllm_cfg = (config.get("vllm", {}) or {})
    conc = max(1, int(vllm_cfg.get("concurrency", 16)))
    render_kw = ctx.render_kw
    batch_n = max(1, int(render_kw.get("batch_size", 8)))
    qa_cfg = config.get("qa", {}) or {}
    max_source_pixels = int(
        ((config.get("sources", {}) or {}).get("max_source_pixels")
         or vllm_cfg.get("max_image_pixels")
         or 80_000_000)
    )

    # Render the teacher 'after' ONLY when a downstream consumer exists. For normal
    # streams this means VLM verify. For S1/S7, verify_degrade_streams is an
    # optional teacher-degraded preview only; Stage-0 owns the real z* PSNR gate.
    verify_rate = _verify_sample_rate(qa_cfg)
    is_degrade_stream = stream_id in (StreamId.S1_DEGRADE_LOCAL, StreamId.S7_DEGRADE_GLOBAL)
    vlm_verify_needed = (
        (not is_degrade_stream)
        and ctx.cleaner is not None
        and _stream_verify_enabled(stream_id, qa_cfg)
        and verify_rate > 0.0
    )
    degrade_preview_needed = (
        is_degrade_stream
        and _stream_verify_enabled(stream_id, qa_cfg)
        and verify_rate > 0.0
    )
    after_needed = vlm_verify_needed or degrade_preview_needed
    render_needed = after_needed and ctx.renderer is not None

    # Per-run scratch for TRANSIENT verify JPEGs (deleted right after the judge
    # reads them). Swept at START so a prior kill -9 can't leak native-res afters.
    if vlm_verify_needed:
        # Keyed by shard so concurrent shard processes never sweep each other's
        # in-flight temps (sample_ids are globally unique, so files never collide).
        vtmp = os.path.join(config.get("scratch_dir", "/tmp"), "verify_tmp",
                            f"{stream_id.value}_s{shard_i}of{shard_n}")
        shutil.rmtree(vtmp, ignore_errors=True)
        os.makedirs(vtmp, exist_ok=True)
        ctx.verify_tmp_dir = vtmp
    else:
        ctx.verify_tmp_dir = None

    def _planned_items():
        """Yield (sid, source, recipe, region_local) for this shard's not-yet-done
        plan positions (deterministic ids => idempotent --resume)."""
        pos = -1
        for source, recipe, region_local in stream.plan(budget, seed):
            pos += 1
            if pos % shard_n != shard_i:
                continue
            key = recipe.source_recipe_id or recipe.kind.value
            sid = make_sample_id(stream_id, source.source_id, key, pos)
            if sid in done:
                continue
            yield sid, source, recipe, region_local

    def _continue(item, after):
        """Clean-pool task: C_GT + annotate + verify + gate (GPU-free)."""
        sid, source, recipe, region_local = item
        try:
            s = stream.build_one(source, recipe, region_local, sid, "pending",
                                 precomputed_after=after)
        except Exception as e:  # isolate per-sample failures (never kill the run)
            print(f"[run] build_one error {sid}: {e}", file=sys.stderr)
            s = None
        return item, s

    def _write(item, sample):
        nonlocal accepted, rejected
        sid, source, recipe, region_local = item
        if sample is None:
            rejected += 1
            rej = Sample(
                sample_id=sid, stream=stream_id, shard="rejected",
                source_path=source.path, raw_decode=source.raw_decode,
                recipe=recipe, region_local=region_local, c_gt=CgtRef(),
            )
            rej.quality.rejected_reason = "rejected"
            writer.log_reject(rej)
        else:
            writer.write(sample)
            done.add(sample.sample_id)
            accepted += 1

    def _reject_planned_item(item, reason: str) -> None:
        nonlocal rejected
        sid, source, recipe, region_local = item
        rejected += 1
        rej = Sample(
            sample_id=sid, stream=stream_id, shard="rejected",
            source_path=source.path, raw_decode=source.raw_decode,
            recipe=recipe, region_local=region_local, c_gt=CgtRef(),
        )
        rej.quality.rejected_reason = reason
        writer.log_reject(rej)
        done.add(sid)

    def _source_too_large(item) -> Optional[str]:
        _sid, source, _recipe, _region_local = item
        if not max_source_pixels:
            return None
        w = int(source.width or 0)
        h = int(source.height or 0)
        raw_decode = source.raw_decode
        is_plain_image = (
            raw_decode == RawDecode.NONE
            or str(raw_decode) == RawDecode.NONE.value
            or getattr(raw_decode, "value", None) == RawDecode.NONE.value
        )
        if (not w or not h) and is_plain_image:
            try:
                # Header-only size probe. PIL.Image.open() reads metadata lazily;
                # it does not allocate the full pixel buffer unless we load/convert.
                from PIL import Image

                with Image.open(source.path) as im:
                    w, h = im.size
                    source.width = w
                    source.height = h
            except Exception:
                w, h = 0, 0
        if w and h:
            pixels = int(w) * int(h)
            if pixels > max_source_pixels:
                return f"source_pixels>{max_source_pixels}({pixels})"
        return None

    def _needs_after_for_verify(item) -> bool:
        if not after_needed:
            return False
        if verify_rate >= 1.0:
            return True
        sid = item[0]
        return _det_rng(seed, sid, "verify").random() < verify_rate

    def _after_for_item(item, params):
        sid, source, recipe, region_local = item
        if params is not None:
            return None
        if recipe.kind == RecipeKind.LUT and _needs_after_for_verify(item):
            try:
                after = stream._apply_lut_after(source, recipe)
                if after is None:
                    return None
                # Downscale BEFORE the after enters the bounded pending queue, exactly
                # like the teacher path (_render_after_batch). _apply_lut_after returns a
                # native-res array; holding max_outstanding of those blows up RAM. The
                # after is only consumed by the (downscaled) judge, never stored as C_GT,
                # so downscaling is safe and build==train is preserved.
                cap = int(qa_cfg.get("verify_downscale_longedge", 768))
                return stream._downscale_rgb(after, cap)
            except Exception as e:  # noqa: BLE001
                print(f"[run] LUT after failed {sid}: {e}", file=sys.stderr)
        return None

    items = _planned_items()
    stop = False
    _last_log = t0                        # throttled progress+RAM logging (OOM/crash diagnosis)
    pending: deque = deque()              # FIFO of (item, future) preserving plan order
    # In-flight cap. Big enough to keep the clean pool busy, but bounded so held
    # afters don't exhaust RAM. conc (=64) saturates the pool without a 2x overhang;
    # afters are downscaled before they enter the queue (see _render_after_batch).
    configured_max_outstanding = vllm_cfg.get("max_outstanding_samples")
    if configured_max_outstanding is None:
        configured_max_outstanding = qa_cfg.get("max_outstanding_samples")
    if configured_max_outstanding is None:
        max_outstanding = max(conc, batch_n * 2)
    else:
        max_outstanding = int(configured_max_outstanding)
    max_outstanding = max(1, max_outstanding)

    with ThreadPoolExecutor(max_workers=conc) as ex:
        def drain(block: bool) -> None:
            """Write EVERY completed future wherever it sits in the deque (so one
            slow head future can't stall later-done writes / freed slots). Relative
            order is preserved for both written and not-yet-done items. On block we
            wait the oldest in-flight future first so a slot always frees."""
            nonlocal stop
            if not pending:
                return
            if block:
                pending[0][1].result()        # wait oldest -> guarantees a freed slot
            keep: deque = deque()
            while pending:
                item, fut = pending.popleft()
                if fut.done():
                    _it, sample = fut.result()
                    _write(item, sample)
                    if limit and accepted >= limit:
                        stop = True
                        keep.extend(pending)
                        pending.clear()
                        break
                else:
                    keep.append((item, fut))
            pending.extend(keep)

        while not stop:
            batch = list(itertools.islice(items, batch_n))
            if not batch:
                break
            planned += len(batch)
            keep = []
            for it in batch:
                too_large = _source_too_large(it)
                if too_large:
                    _reject_planned_item(it, too_large)
                else:
                    keep.append(it)
            batch = keep
            if not batch:
                continue
            sources = [it[1] for it in batch]
            if after_needed:
                # ONE batched GPU render for the whole batch (the dominant cost; the
                # 768-token autoregression is batched here). Blocks the main thread on
                # CUDA while the pool cleans the previous batch.
                params_list = [
                    (
                        stream._params_for_render(it[2])
                        if (render_needed and _needs_after_for_verify(it))
                        else None
                    )
                    for it in batch
                ]
                afters = (
                    stream._render_after_batch(sources, params_list)
                    if render_needed
                    else [None] * len(batch)
                )
                for idx, (it, params) in enumerate(zip(batch, params_list)):
                    if afters[idx] is None:
                        afters[idx] = _after_for_item(it, params)
            else:
                afters = [None] * len(batch)  # verify off -> after unused -> skip render
            for it, after in zip(batch, afters):
                while len(pending) >= max_outstanding and not stop:
                    drain(block=True)
                if stop:
                    break
                pending.append((it, ex.submit(_continue, it, after)))
                # bound outstanding work, then opportunistically write what's ready.
                drain(block=False)

            # throttled progress + RAM heartbeat (diagnose silent deaths/OOM creep).
            now = time.time()
            if now - _last_log >= 20:
                _last_log = now
                dt_ = now - t0
                print(f"[run] {stream_id.value} progress planned={planned} accepted={accepted} "
                      f"rejected={rejected} inflight={len(pending)} "
                      f"rate={round(accepted/dt_,2) if dt_>0 else 0}/s ram_avail_mb={_mem_avail_mb()}",
                      flush=True)

        # flush remaining in-flight work (unless --limit already stopped us).
        while pending and not stop:
            drain(block=True)

    writer.flush()
    dt = time.time() - t0
    return {
        "stream": stream_id.value,
        "planned": planned,
        "accepted": accepted,
        "rejected": rejected,
        "seconds": round(dt, 2),
        "rate_per_s": round(accepted / dt, 2) if dt > 0 else None,
    }


# ===========================================================================
# CLI.
# ===========================================================================


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="dataset_build.run",
        description="Sharded, resumable orchestrator for the VeraRetouch Direction-A dataset.",
    )
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--stream", default=None, help="comma list S1,S2,... (default: all enabled)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--pilot", nargs="?", const=-1, type=int, default=None,
                      help="pilot mode; optional N to scale full budgets to total N")
    mode.add_argument("--full", action="store_true", help="full 1M run")
    p.add_argument("--dry-run", action="store_true", help="CPU MockModels; no weights/vLLM")
    p.add_argument("--shard", default=None, help="partition work: i/n")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out-suffix", default="")
    p.add_argument("--limit", type=int, default=None, help="cap accepted samples/stream (debug)")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    out_root = resolve_out_root(config, args.out_suffix)

    pilot = args.pilot is not None
    pilot_total = args.pilot if (pilot and args.pilot and args.pilot > 0) else None
    full = bool(args.full)
    if not pilot and not full:
        # default to pilot if neither given (subagent-safe smoke run)
        pilot = True

    shard_i, shard_n = _parse_shard_flag(args.shard)
    if shard_n > 1:
        storage_cfg = dict(config.get("storage", {}) or {})
        storage_cfg.setdefault("shard_prefix", f"shard_w{shard_i}of{shard_n}")
        config["storage"] = storage_cfg

    # which streams
    if args.stream:
        try:
            stream_ids = [StreamId(s.strip()) for s in args.stream.split(",") if s.strip()]
        except ValueError as e:
            raise SystemExit(f"bad --stream: {e}")
    else:
        stream_ids = list(StreamId)

    budgets = resolve_budgets(config, pilot, pilot_total, full)
    budgets = apply_s4_fallback(budgets, config)

    writer = ShardWriter(out_root, config)
    if args.resume:
        manifest_report = validate_manifest_index(out_root, config)
        if not manifest_report.get("ok", False):
            preview = manifest_report.get("issues", [])[:5]
            raise SystemExit(
                "[run] --resume manifest invariant check failed: "
                + json.dumps(preview, ensure_ascii=False, sort_keys=True)
            )
    gate = QAGate(config)

    if args.dry_run:
        ctx = build_mock_context(config, writer)
        models = {"renderer": ctx.renderer, "masker": ctx.masker,
                  "cleaner": ctx.cleaner, "parser": ctx.parser,
                  "lut_applier": ctx.lut_applier}
    else:
        models = _build_real_models(config, stream_ids)
        # Decoupled core facade: render_lock+renderer relocate into core.render;
        # cleaner/masker become the delegating core.vllm/core.sam3 wrappers (which
        # forward verbatim, so the build is byte-identical). gpu=None keeps render
        # serialized by render_lock alone, exactly as before. Rollback = drop the
        # `core=` kwarg (the inline path is the byte-identical fallback).
        from dataset_build.core import build_core
        from dataset_build.core.gpu_compute import GpuCompute

        # VERA_DISABLE_CORE=1 pins the inline (pre-core) path — the byte-identical
        # fallback and the Phase-2 rollback switch.
        disable_core = os.environ.get("VERA_DISABLE_CORE") == "1"
        # Shared per-card lease. The shard runs under CUDA_VISIBLE_DEVICES=<i>, so
        # in-process the renderer sees cuda:0. Uncontended in the build (renderer
        # is the only GPU tenant) -> byte-identical; establishes the
        # lease->render_lock contract for render/IQA/SAM3 co-tenancy.
        gpu = GpuCompute(["cuda:0"])
        core = None if disable_core else build_core(
            config,
            renderer=models["renderer"],
            masker=models["masker"],
            cleaner=models["cleaner"],
            gpu=gpu,
        )
        ctx = BuildContext(
            config=config,
            renderer=models["renderer"],          # kept for the render_needed gate + fallback
            masker=(core.sam3 if core else models["masker"]),     # delegating wrapper when core on
            cleaner=(core.vllm if core else models["cleaner"]),   # delegating wrapper when core on
            parser=models["parser"],
            lut_applier=models.get("lut_applier"),
            cgt_writer=writer,
            cached_tagger=models.get("cached_tagger"),
            core=core,
        )

    print(f"[run] out_root={out_root} streams={[s.value for s in stream_ids]} "
          f"pilot={pilot} full={full} dry_run={args.dry_run} shard={shard_i}/{shard_n} "
          f"resume={args.resume}")

    results: List[Dict[str, Any]] = []
    for sid in stream_ids:
        b = budgets.get(sid, 0)
        if b <= 0:
            continue
        inputs = load_plan_inputs(config, out_root, sid, synthetic=args.dry_run)
        res = run_stream(
            sid, b, config, ctx, inputs, writer, gate,
            shard_i=shard_i, shard_n=shard_n, resume=args.resume, limit=args.limit,
        )
        results.append(res)
        print(f"[run] {sid.value}: {res}")

    stats = writer.stats()
    print(f"[run] DONE stats={json.dumps(stats)}")

    rl_frac = stats.get("region_local_fraction", 0.0)
    floor = float(config.get("region_local_floor", 0.50))
    if stats.get("committed", 0) > 0 and rl_frac + 1e-9 < floor:
        print(f"[run] WARNING region-local fraction {rl_frac:.3f} < floor {floor}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
