"""Build-side reader for the source-image TAG + AESTHETIC precompute cache.

Mirrors ``mask_cache.CachedMasker``: the precompute driver (``tag_precompute.py``)
writes ``<cache_dir>/<path_key(source_path)>/tags.json`` per UNIQUE source image,
and the base-env build orchestrator consumes it through ``CachedTagger`` INSTEAD
of issuing the inline per-sample VLM ``tag_scene_region`` call.

``tags_for(source_path)`` returns the SAME dict shape that
``vlm_clean.QwenVLCleaner.tag_scene_region`` returns
(``scene / style / region_local / sam3_concepts / groundingdino_prompt /
masksubtype_hint``) PLUS aesthetic fields (``aesthetic_vlm``,
``aesthetic_model``, ``aesthetic``) so it is a drop-in for the inline tagger.
Returns ``None`` on a cache miss (the caller then falls through to the live VLM /
offline path — no behavior change).

The path_key is REUSED from ``mask_cache`` so the tag cache and the SAM3 mask
cache share identical keys/sharding (disjoint + consistent shards).
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

from dataset_build.mask_cache import path_key

_VALID_SCENES = (
    "portrait", "landscape", "street", "food", "product",
    "wedding", "night", "architecture", "still_life", "any",
)


def _as_path(image: Any) -> str:
    if isinstance(image, str):
        return image
    return getattr(image, "filename", None) or str(image)


def _coerce(d: dict) -> dict:
    """Normalize a cached tags.json blob into the tag_scene_region shape + aesthetic."""
    scene = str(d.get("scene", "")).strip().lower()
    if scene not in _VALID_SCENES:
        scene = "any"
    concepts = d.get("sam3_concepts", [])
    if not isinstance(concepts, list):
        concepts = [str(concepts)] if concepts else []
    concepts = [str(c).strip() for c in concepts if str(c).strip()]
    try:
        msub = int(d.get("masksubtype_hint", 0) or 0)
    except (TypeError, ValueError):
        msub = 0

    def _f(v: Any) -> Optional[float]:
        try:
            return None if v is None else float(v)
        except (TypeError, ValueError):
            return None

    aesthetic_vlm = _f(d.get("aesthetic_vlm"))
    aesthetic_model = _f(d.get("aesthetic_model"))
    # Combined: prefer the dedicated model score, else the VLM score.
    aesthetic = aesthetic_model if aesthetic_model is not None else aesthetic_vlm
    if d.get("aesthetic") is not None:
        aesthetic = _f(d.get("aesthetic"))
    return {
        "scene": scene,
        "style": str(d.get("style", "")).strip(),
        "region_local": bool(d.get("region_local", False)),
        "sam3_concepts": concepts,
        "groundingdino_prompt": str(d.get("groundingdino_prompt", "")).strip(),
        "masksubtype_hint": msub,
        "aesthetic_vlm": aesthetic_vlm,
        "aesthetic_model": aesthetic_model,
        "aesthetic": aesthetic,
    }


class CachedTagger:
    """Reads precomputed per-source ``tags.json``. Drop-in for the inline tagger.

    Missing key/file -> ``tags_for`` returns ``None`` so the caller keeps its
    existing live/offline behavior (graceful, no behavior change on miss)."""

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir

    def _tags_path(self, source_path: Any) -> str:
        return os.path.join(self.cache_dir, path_key(_as_path(source_path)), "tags.json")

    def tags_for(self, source_path: Any) -> Optional[dict]:
        p = self._tags_path(source_path)
        if not os.path.exists(p):
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            return None
        if not isinstance(d, dict):
            return None
        return _coerce(d)
