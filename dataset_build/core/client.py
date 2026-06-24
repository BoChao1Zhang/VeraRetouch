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

import threading
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
    """LR (Lightroom farm) resource handle — the core's ``core.lr``.

    Real renders go to the durable ``render_jobs`` queue + ``lrc_task_server``
    farm via ``source_qa.lr_render`` (lazy-imported so the core does not hard-
    depend on source_qa). A bounded semaphore is the **LR pool admission**: the
    true number of concurrent renders submitted to the farm, independent of any
    caller's thread-pool size — size it to the online eligible LrC clients (1:1 =
    no server-side queue). The ``render_jobs`` idempotency/caching and job
    bookkeeping stay with the caller (preset_qa), which already owns that state;
    this handle only owns the *transport + concurrency cap*. Structured failure
    dicts from ``lr_render`` ({ok, error_code, retryable, ...}) pass through
    verbatim. The build does not use LR (it is a QA-stage path).
    """

    def __init__(
        self,
        max_concurrency: Optional[int] = None,
        render_fn: Optional[Any] = None,
        submit_and_wait_fn: Optional[Any] = None,
        health_fn: Optional[Any] = None,
    ) -> None:
        self._render_fn = render_fn
        self._submit_and_wait_fn = submit_and_wait_fn
        self._health_fn = health_fn
        cap = int(max_concurrency) if max_concurrency else 0
        self._sem: Optional[threading.BoundedSemaphore] = (
            threading.BoundedSemaphore(cap) if cap > 0 else None)

    @staticmethod
    def _lr() -> Any:
        from dataset_build.source_qa import lr_render  # lazy: source_qa-only dep
        return lr_render

    def _gate(self):
        return self._sem if self._sem is not None else nullcontext()

    def render(self, recipe_path: str, fmt: str, photo_path: str) -> Dict[str, Any]:
        """Convert preset -> config.lua, submit to the farm, return the after.

        Returns ``lr_render.render_via_lr``'s structured dict
        (``{ok, after_path, ...}`` or ``{ok: False, error_code, retryable, ...}``).
        Bounded by the LR pool semaphore (held only for the network render)."""
        fn = self._render_fn or self._lr().render_via_lr
        with self._gate():
            return fn(recipe_path, fmt, photo_path)

    def submit_and_wait(self, photo_path: str, lua_path: str) -> Dict[str, Any]:
        """Low-level: submit a pre-built config.lua + probe, long-poll the after."""
        fn = self._submit_and_wait_fn or self._lr().submit_and_wait
        with self._gate():
            return fn(photo_path, lua_path)

    def health(self) -> bool:
        fn = self._health_fn or self._lr().lr_health
        try:
            return bool(fn())
        except Exception:
            return False

    # Back-compat alias: ``submit`` == ``render``.
    def submit(self, recipe_path: str, fmt: str, photo_path: str) -> Dict[str, Any]:
        return self.render(recipe_path, fmt, photo_path)
