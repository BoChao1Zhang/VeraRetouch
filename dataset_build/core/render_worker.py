"""In-process teacher-render worker for the core layer.

``RenderClient`` owns the (non-thread-safe) GPU teacher renderer and the
``render_lock`` that serializes it — relocated verbatim from
``streams.Stream._render_after_batch`` / ``_render_after`` so the build's
existing main-thread overlap loop keeps calling it unchanged through the core
facade. The render logic (None-param filtering, single batched ``render()`` under
the lock, 768 long-edge downscale, scatter back to input order) is byte-for-byte
the same as before; only its *owner* moved. With ``gpu=None`` (the build path)
the only serialization is ``render_lock``, exactly as today.

``downscale_rgb`` is the single canonical downscale implementation; the old
``streams.Stream._downscale_rgb`` now delegates here so both code paths produce
identical pixels.
"""

from __future__ import annotations

import sys
import threading
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Sequence


def downscale_rgb(arr: Any, longedge: int) -> Any:
    """Long-edge downscale of an RGB uint8 array (best-effort; returns input on
    failure or if already small). Canonical impl — keep byte-identical to the
    historical ``streams.Stream._downscale_rgb``."""
    try:
        import cv2
        import numpy as np

        a = np.asarray(arr)
        h, w = a.shape[:2]
        m = max(h, w)
        if not longedge or m <= longedge:
            return a
        s = longedge / float(m)
        return cv2.resize(a, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                          interpolation=cv2.INTER_AREA)
    except Exception:
        return arr


class RenderClient:
    """Owns the teacher renderer + its serializing lock (in-process, 1 worker)."""

    def __init__(
        self,
        renderer: Optional[Any],
        render_kw: Dict[str, int],
        render_lock: Optional[Any] = None,
        gpu: Optional[Any] = None,
        device: str = "cuda:0",
    ) -> None:
        self.renderer = renderer
        self.render_kw = dict(render_kw or {})
        # threading.Lock (NOT RLock): the GPU teacher is non-reentrant.
        self.render_lock = render_lock if render_lock is not None else threading.Lock()
        self.gpu = gpu
        self.device = device

    def _lease(self):
        # Strict order: GPU lease (if any) OUTSIDE render_lock. None => no lease.
        return self.gpu.lease(self.device) if self.gpu is not None else nullcontext()

    def render_batch(
        self,
        paths: Sequence[str],
        params_list: Sequence[Optional[Dict[str, Dict[str, float]]]],
        downscale_longedge: int,
        log_prefix: str = "render",
    ) -> List[Optional[Any]]:
        """Batched teacher render: ONE ``renderer.render()`` for the whole batch.

        Returns one np.uint8 HxWx3 RGB (or None) per input, IN INPUT ORDER. Items
        whose params is None never reach the GPU and come back None. The single
        ``render()`` is held under the GPU lease (if any) then ``render_lock``.
        """
        n = len(paths)
        results: List[Optional[Any]] = [None] * n
        if self.renderer is None:
            return results
        idxs: List[int] = []
        rpaths: List[str] = []
        pdicts: List[Dict[str, Dict[str, float]]] = []
        for i, (pth, p) in enumerate(zip(paths, params_list)):
            if p is None:
                continue
            idxs.append(i)
            rpaths.append(pth)
            pdicts.append(p)
        if not rpaths:
            return results
        try:
            with self._lease():
                with self.render_lock:
                    outs = self.renderer.render(rpaths, pdicts, **self.render_kw)
        except Exception as e:  # noqa: BLE001 - isolate a whole-batch teacher failure
            print(f"[{log_prefix}] batch render failed ({len(rpaths)}): {e}", file=sys.stderr)
            return results
        cap = int(downscale_longedge)
        for k, slot in enumerate(idxs):
            out = outs[k] if (k < len(outs)) else None
            results[slot] = downscale_rgb(out, cap) if out is not None else None
        return results

    def render_one(
        self,
        path: str,
        params: Dict[str, Dict[str, float]],
        log_prefix: str = "render",
    ) -> Optional[Any]:
        """Render the GLOBAL 'after' for one item (no downscale). None on failure."""
        if self.renderer is None:
            return None
        try:
            with self._lease():
                with self.render_lock:
                    outs = self.renderer.render([path], [params], **self.render_kw)
        except Exception as e:  # noqa: BLE001 - isolate per-sample teacher failures
            print(f"[{log_prefix}] render failed for {path}: {e}", file=sys.stderr)
            return None
        return outs[0] if outs else None
