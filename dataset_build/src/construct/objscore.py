"""Objective, learned no-reference quality scores for renders (musiq + clipiqa+, pyiqa on GPU0).

These are continuous, objective perceptual-quality predictors — complementary to the VLM's near-binary
aesthetic judgment (corr only ~0.2-0.33), so blending them in genuinely separates renders by quality
instead of just reshaping a bimodal VLM histogram. GPU0 is otherwise idle (vLLM is on GPU1).

  musiq   : MUSIQ perceptual quality, ~[0,100]
  clipiqa : CLIP-IQA+ quality/aesthetic, ~[0,1]
"""
from __future__ import annotations

import os
import threading

_DEV = os.environ.get("CONSTRUCT_IQA_DEVICE", "cuda:0")
_LOCK = threading.Lock()      # pyiqa GPU calls are serialized (agent scores from a thread pool)
_M = {}
_CACHE = {}                   # path -> (musiq, clipiqa); sources are re-scored across their 8 candidates


def _models():
    if not _M:
        import pyiqa
        _M["musiq"] = pyiqa.create_metric("musiq", device=_DEV).eval()
        _M["clipiqa"] = pyiqa.create_metric("clipiqa+", device=_DEV).eval()
    return _M


def score(path: str):
    """(musiq, clipiqa) for an image, or (None, None) on failure. Cached per path."""
    if path in _CACHE:
        return _CACHE[path]
    try:
        with _LOCK:
            m = _models()
            musiq = float(m["musiq"](path).item())
            clip = float(m["clipiqa"](path).item())
        _CACHE[path] = (musiq, clip)
    except Exception:  # noqa: BLE001 - a bad render must not kill the batch
        _CACHE[path] = (None, None)
    return _CACHE[path]
