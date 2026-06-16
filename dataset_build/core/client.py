"""Core client handles: vllm / sam3 / iqa / lr.

Thin wrappers around the existing heavy collaborators so the business keeps its
exact algorithms while consuming a uniform ``core.*`` surface:

  * ``VLLMClient`` wraps a ``QwenVLCleaner`` (which routes through the vGate
    broker with an ``X-vgate-class`` priority). High-level methods
    (``gen_instruction`` / ``reason_params`` / ``verify`` / ``tag_*``) delegate
    verbatim via ``__getattr__``; ``submit`` is the low-level OpenAI passthrough
    (prio-aware, image-aware, bounded 429 retry) for callers that build their
    own messages.
  * ``Sam3Client`` reads precomputed masks from a ``CachedMasker`` and, on a
    cache miss, optionally falls back to a live ``Sam3Masker`` under a GPU lease
    (serialized against the renderer).
  * ``IqaClient`` is QA-internal NR-IQA (pyiqa) under a GPU lease, with a
    lease-amortized batch path.
  * ``LrClient`` is a thin handle over the source_qa LR submit path.

``RenderClient`` lives in ``core.render_worker`` and is re-exported by the facade.
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Sequence, Tuple

# OpenAI 429 is surfaced as RateLimitError; import defensively (openai is only
# present in request-issuing envs).
try:  # pragma: no cover - import shape depends on env
    from openai import RateLimitError as _RateLimitError  # type: ignore
    from openai import APIError as _APIError  # type: ignore
    _RETRYABLE: Tuple[type, ...] = (_RateLimitError, _APIError)
except Exception:  # pragma: no cover
    _RETRYABLE = (Exception,)


class VLLMClient:
    """Broker-routed VLM handle. Delegates high-level methods to the cleaner."""

    def __init__(self, cleaner: Any) -> None:
        self._cleaner = cleaner

    def __getattr__(self, name: str) -> Any:
        # Only reached for names not found on the instance/class -> delegate.
        return getattr(self._cleaner, name)

    @property
    def cleaner(self) -> Any:
        return self._cleaner

    def submit(
        self,
        messages: Sequence[Dict[str, Any]],
        images: Optional[Sequence[str]] = None,
        json_mode: bool = False,
        prio: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """Low-level chat passthrough to the broker via the cleaner's client.

        ``images`` (local file paths) are encoded with the cleaner's bounded-JPEG
        ``_img_to_data_uri`` and appended as ``image_url`` parts to the last user
        message (or a new one). ``prio`` sets ``X-vgate-class`` (defaults to the
        cleaner's class). A bounded retry absorbs broker ``429``/transient errors,
        honoring ``Retry-After`` when present.
        """
        msgs: List[Dict[str, Any]] = [dict(m) for m in messages]
        if images:
            img_parts = [{"type": "image_url",
                          "image_url": {"url": self._cleaner._img_to_data_uri(p)}}
                         for p in images]
            target = next((m for m in reversed(msgs) if m.get("role") == "user"), None)
            if target is None:
                target = {"role": "user", "content": []}
                msgs.append(target)
            content = target.get("content")
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            elif not isinstance(content, list):
                content = []
            target["content"] = list(content) + img_parts

        kwargs: Dict[str, Any] = dict(
            model=self._cleaner.model,
            messages=msgs,
            temperature=self._cleaner.temperature,
            max_tokens=self._cleaner.max_tokens,
            extra_headers={"X-vgate-class": prio or getattr(self._cleaner, "vgate_class", "build-annotate")},
            timeout=timeout or self._cleaner.timeout,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        client = self._cleaner._get_client()
        last: Optional[Exception] = None
        for attempt in range(max(1, max_retries)):
            try:
                resp = client.chat.completions.create(**kwargs)
                msg = resp.choices[0].message
                return {"content": msg.content or "", "usage": getattr(resp, "usage", None)}
            except _RETRYABLE as e:  # 429 / transient upstream
                last = e
                if attempt >= max_retries - 1:
                    break
                time.sleep(_retry_after(e, default=2.0 * (attempt + 1)))
        raise RuntimeError(f"core.vllm.submit failed after {max_retries} attempts: {last}")


def _retry_after(exc: Exception, default: float) -> float:
    """Best-effort Retry-After (seconds) from an OpenAI/HTTP error, else default."""
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None)
    if headers is not None:
        try:
            val = headers.get("retry-after") or headers.get("Retry-After")
            if val is not None:
                return float(val)
        except Exception:
            pass
    return float(default)


class Sam3Client:
    """Mask handle: precomputed CachedMasker read + optional live SAM3 fallback.

    On a cache miss for some concepts, and when a live ``Sam3Masker`` is wired,
    it computes the missing masks live under a GPU lease (serialized against the
    renderer per the lease->render_lock contract).
    """

    def __init__(
        self,
        cached_masker: Optional[Any] = None,
        live_masker: Optional[Any] = None,
        gpu: Optional[Any] = None,
        device: str = "cuda:0",
    ) -> None:
        self._cached = cached_masker
        self._live = live_masker
        self._gpu = gpu
        self._device = device

    def __getattr__(self, name: str) -> Any:
        # Delegate unknown attrs to the cached masker for back-compat.
        if self._cached is not None:
            return getattr(self._cached, name)
        raise AttributeError(name)

    @property
    def masker(self) -> Any:
        return self._cached

    def _lease(self):
        return self._gpu.lease(self._device) if self._gpu is not None else nullcontext()

    def masks(self, image: Any, concepts: Sequence[str],
              native_size: Optional[Any] = None, **kw) -> Dict[str, Any]:
        concepts = list(concepts)
        res: Dict[str, Any] = {}
        if self._cached is not None:
            res = dict(self._cached.masks(image, concepts, native_size=native_size, **kw) or {})
        missing = [c for c in concepts if res.get(c) is None]
        if missing and self._live is not None:
            with self._lease():
                live = self._live.masks(image, missing, native_size=native_size, **kw) or {}
            for c, v in live.items():
                if v is not None:
                    res[c] = v
        return res


class IqaClient:
    """QA-internal NR-IQA handle (pyiqa) under a GPU lease."""

    def __init__(self, scorer: Any, gpu: Optional[Any] = None,
                 device: str = "cuda:0", batch_size: int = 16) -> None:
        self._scorer = scorer
        self._gpu = gpu
        self._device = device
        self._batch = max(1, int(batch_size))

    def _lease(self):
        return self._gpu.lease(self._device) if self._gpu is not None else nullcontext()

    def score(self, path: str, want_face: bool = False) -> Dict[str, Any]:
        with self._lease():
            return self._scorer.score_path(path, want_face=want_face)

    def score_batch(self, items: Sequence[Tuple[str, bool]]) -> List[Dict[str, Any]]:
        """Score (path, want_face) items, amortizing one GPU lease over each
        chunk of up to ``batch_size`` paths (the renderer/SAM3 can't preempt
        mid-chunk)."""
        out: List[Dict[str, Any]] = []
        items = list(items)
        for i in range(0, len(items), self._batch):
            chunk = items[i:i + self._batch]
            with self._lease():
                for path, want_face in chunk:
                    out.append(self._scorer.score_path(path, want_face=want_face))
        return out


class LrClient:
    """Thin handle over the source_qa LR (Lightroom farm) submit path.

    ``submit_fn`` is the existing durable-queue submit callable; the build does
    not use this (LR is a QA-stage path). Kept for facade completeness.
    """

    def __init__(self, submit_fn: Optional[Any] = None) -> None:
        self._submit_fn = submit_fn

    def submit(self, *args, **kw) -> Any:
        if self._submit_fn is None:
            raise RuntimeError("LrClient has no submit_fn configured")
        return self._submit_fn(*args, **kw)
