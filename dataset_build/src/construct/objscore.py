"""Artimuse + Charm mixed IAA scores for rendered candidates.

Historically this module returned MUSIQ + CLIP-IQA+. The construct pipeline now
uses the same IAA signal as source cleaning: ArtiMuse score, Charm score, and a
weighted 0..100 ``iaa_mixed`` score. The public ``score(path)`` entry point is
kept so callers do not need to know how the scorer is hosted.
"""
from __future__ import annotations

import os
import threading
from typing import Any, Optional

from dataset_build.source_qa import config
from dataset_build.source_qa.iaa import MixedIAARunner, blend_scores

_DEV = os.environ.get("CONSTRUCT_IAA_DEVICE", config.IAA_DEVICE)
_LOCK = threading.Lock()
_RUNNER: Optional[MixedIAARunner] = None
_CACHE: dict[str, dict[str, Optional[float]]] = {}


def _runner() -> MixedIAARunner:
    global _RUNNER
    if _RUNNER is None:
        _RUNNER = MixedIAARunner(device=_DEV)
    return _RUNNER


def score(path: str) -> dict[str, Optional[float]]:
    """Return {iaa_mixed, artimuse, charm} for an image, cached per path."""
    if path in _CACHE:
        return _CACHE[path]
    try:
        with _LOCK:
            raw = _runner().score_path(path)
        out = {k: raw.get(k) for k in ("iaa_mixed", "artimuse", "charm")}
    except Exception:  # noqa: BLE001 - a bad render must not kill the batch
        out = {"iaa_mixed": None, "artimuse": None, "charm": None}
    _CACHE[path] = out
    return out


def mixed_value(obj: Any) -> Optional[float]:
    """Extract a 0..100 mixed IAA value, accepting old tuple callers too."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        v = obj.get("iaa_mixed")
        if v is not None:
            return float(v)
        return blend_scores(
            obj.get("artimuse"),
            obj.get("charm"),
            config.IAA_ARTIMUSE_WEIGHT,
            config.IAA_CHARM_WEIGHT,
        )
    if isinstance(obj, (tuple, list)):
        if len(obj) >= 3 and obj[0] is not None:
            return float(obj[0])
        if len(obj) >= 2 and obj[0] is not None and obj[1] is not None:
            # Backward compatibility for the old (musiq, clipiqa) tuple.
            return max(0.0, min(100.0, 0.5 * float(obj[0]) + 50.0 * float(obj[1])))
    return None
