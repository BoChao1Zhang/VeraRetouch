"""Cache-backed concept masker.

SAM3 (transformers 5.2, env monetgpt_sam3) cannot share a process with the
VeraRetouch renderer (llava, base env). So SAM3 C_GT masks are PRECOMPUTED by
``sam3_precompute.py`` into a PNG cache, and the base-env orchestrator consumes
them through ``CachedMasker`` — a duck-typed ConceptMasker (``.masks`` /
``.mask``) that loads PNGs and never imports SAM3.

Cache layout (keys shared with the writer so they always agree):
    <cache_dir>/<path_key(source_path)>/<concept_slug(concept)>.png   # 8-bit L, [0,1]
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


def path_key(path: Any) -> str:
    """Stable 16-hex key from a source path (writer & reader must match)."""
    return hashlib.sha1(os.fspath(path).encode("utf-8")).hexdigest()[:16]


def concept_slug(concept: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(concept).strip().lower()).strip("_")
    return s or "concept"


def _as_path(image: Any) -> str:
    if isinstance(image, str):
        return image
    return getattr(image, "filename", None) or str(image)


def _load_png(path: str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    try:
        from PIL import Image

        a = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        return a
    except Exception:
        return None


def _resize(mask: np.ndarray, native_size: Optional[Tuple[int, int]]) -> np.ndarray:
    if native_size is None:
        return mask
    H, W = native_size
    if mask.shape == (H, W):
        return mask
    try:
        import cv2

        return cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    except Exception:
        from PIL import Image

        im = Image.fromarray((np.clip(mask, 0.0, 1.0) * 255).astype("uint8")).resize((W, H))
        return np.asarray(im, dtype=np.float32) / 255.0


class CachedMasker:
    """Reads precomputed SAM3 concept PNGs. Missing concepts are simply omitted
    from ``.masks`` -> streams' ``np.maximum.reduce`` over an empty list yields no
    mask -> the sample gracefully degrades to global (DATASET §5: C_GT optional)."""

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir

    def mask(self, image: Any, concept: str, native_size: Optional[Tuple[int, int]] = None, **kw) -> np.ndarray:
        p = os.path.join(self.cache_dir, path_key(_as_path(image)), concept_slug(concept) + ".png")
        m = _load_png(p)
        if m is None:
            H, W = native_size if native_size else (288, 288)
            return np.zeros((H, W), dtype=np.float32)
        return _resize(m, native_size)

    def masks(self, image: Any, concepts: Sequence[str], native_size: Optional[Tuple[int, int]] = None, **kw) -> Dict[str, np.ndarray]:
        base = os.path.join(self.cache_dir, path_key(_as_path(image)))
        out: Dict[str, np.ndarray] = {}
        for c in concepts:
            m = _load_png(os.path.join(base, concept_slug(c) + ".png"))
            if m is not None:
                out[c] = _resize(m, native_size)
        return out
