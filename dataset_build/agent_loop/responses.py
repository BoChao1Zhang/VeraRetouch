"""Narrow OpenAI Responses adapter with usage and provider-cache telemetry."""
from __future__ import annotations

import base64
import io
import json
import random
import threading
import time
from contextlib import nullcontext
from typing import Any, Callable, Mapping

from PIL import Image

from dataset_build.responses_events import is_official_response_event

from .api_cache import CachedResult, ExactResponseCache, RequestSpec, retryable_exception
from .artifacts import ArtifactStore
from .config import EndpointConfig
from .prompts import semantic_error


class TransportError(RuntimeError):
    def __init__(self, code: str, *, retry_after: float | None = None):
        super().__init__(code)
        self.code = code
        self.retry_after = retry_after


class ModelSubstituted(RuntimeError):
    pass


_HOSTED_UNSUPPORTED_SCHEMA_KEYWORDS = frozenset({"uniqueItems"})


def _hosted_schema(node: Any) -> Any:
    if isinstance(node, Mapping):
        return {
            key: _hosted_schema(value) for key, value in node.items()
            if key not in _HOSTED_UNSUPPORTED_SCHEMA_KEYWORDS
        }
    if isinstance(node, list):
        return [_hosted_schema(value) for value in node]
    return node


def _usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if isinstance(usage, Mapping):
        input_details = usage.get("input_tokens_details") or {}
        return {
            "input_tokens": int(usage.get("input_tokens", 0) or 0),
            "output_tokens": int(usage.get("output_tokens", 0) or 0),
            "cached_tokens": int(input_details.get("cached_tokens", 0) or 0),
            "cache_write_tokens": int(input_details.get("cache_write_tokens", 0) or 0),
        }
    input_details = getattr(usage, "input_tokens_details", None)
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "cached_tokens": int(getattr(input_details, "cached_tokens", 0) or 0),
        "cache_write_tokens": int(getattr(input_details, "cache_write_tokens", 0) or 0),
    }


def consume_stream(stream: Any) -> dict[str, Any]:
    from openai.types.responses import (
        ResponseCompletedEvent, ResponseErrorEvent, ResponseFailedEvent,
        ResponseTextDeltaEvent,
    )

    chunks: list[str] = []
    completed: Any = None
    relay_telemetry_events = 0
    context = stream if hasattr(stream, "__enter__") else nullcontext(stream)
    with context as events:
        for event in events:
            if not is_official_response_event(event):
                # The zzone relay's codex-backed channels inject benign telemetry
                # events (observed: type='codex.rate_limits', carrying plan/usage
                # numbers) into every stream. They are skipped and counted; any
                # other unofficial event still fails closed as a malformed stream.
                if str(getattr(event, "type", "") or "").startswith("codex."):
                    relay_telemetry_events += 1
                    continue
                raise TransportError("untyped_responses_event")
            if isinstance(event, ResponseTextDeltaEvent):
                chunks.append(event.delta)
            elif isinstance(event, ResponseCompletedEvent):
                completed = event.response
            elif isinstance(event, ResponseFailedEvent):
                error = getattr(event.response, "error", None)
                raise TransportError(str(getattr(error, "code", None) or "response_failed"))
            elif isinstance(event, ResponseErrorEvent):
                raise TransportError(str(event.code or "response_error"))
    if completed is None:
        raise TransportError("responses_stream_interrupted")
    if getattr(completed, "status", None) not in (None, "completed"):
        raise TransportError("responses_not_completed")
    try:
        # This relay sometimes completes with a text item whose content is null;
        # the SDK property then raises instead of returning "". That is an empty
        # output, i.e. the retryable case below, not a caller-side TypeError.
        fallback = str(getattr(completed, "output_text", "") or "")
    except (TypeError, ValueError):
        fallback = ""
    # This relay injects a zero-width space (U+200B) at the head of the first
    # text delta (keep-alive / anti-buffering), so joined deltas are not valid
    # JSON while completed.output_text is clean. Prefer output_text; fall back
    # to the deltas only when it is empty, and strip any leading zero-width
    # marks either way (strict json_schema output must start with "{").
    text = (fallback or "".join(chunks)).lstrip("\u200b\ufeff")
    if not text:
        raise TransportError("responses_output_empty")
    raw = completed.model_dump(mode="json") if hasattr(completed, "model_dump") else {}
    return {
        "text": text,
        "model": str(getattr(completed, "model", "") or ""),
        "response_id": str(getattr(completed, "id", "") or ""),
        "usage": _usage(completed),
        "raw_response": raw,
        "relay_telemetry_events": relay_telemetry_events,
    }


def _message_text(response: Any) -> str:
    """Collect text parts from ``response.output`` message items."""
    parts: list[str] = []
    for item in getattr(response, "output", None) or []:
        if isinstance(item, Mapping):
            item_type, content = item.get("type"), item.get("content")
        else:
            item_type, content = getattr(item, "type", None), getattr(item, "content", None)
        if item_type != "message":
            continue
        for part in content or []:
            value = part.get("text") if isinstance(part, Mapping) else getattr(part, "text", None)
            if isinstance(value, str):
                parts.append(value)
    return "".join(parts)


def consume_response(response: Any) -> dict[str, Any]:
    """Non-streaming counterpart of :func:`consume_stream` (same result shape)."""
    if getattr(response, "status", None) not in (None, "completed"):
        raise TransportError("responses_not_completed")
    try:
        # This relay sometimes completes with a text item whose content is null;
        # the SDK property then raises instead of returning "". That is an empty
        # output, i.e. the retryable case below, not a caller-side TypeError.
        text = str(getattr(response, "output_text", "") or "")
    except (TypeError, ValueError):
        text = ""
    # Some responses carry empty output_text while the message content parts do
    # hold the text; strip leading zero-width marks either way (same discipline
    # as the streaming path: strict json_schema output must start with "{").
    text = (text or _message_text(response)).lstrip("\u200b\ufeff")
    if not text:
        raise TransportError("responses_output_empty")
    raw = response.model_dump(mode="json") if hasattr(response, "model_dump") else {}
    return {
        "text": text,
        "model": str(getattr(response, "model", "") or ""),
        "response_id": str(getattr(response, "id", "") or ""),
        "usage": _usage(response),
        "raw_response": raw,
    }


class ResponsesAdapter:
    def __init__(
        self, endpoint: EndpointConfig, artifacts: ArtifactStore,
        *, client_factory: Any | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.artifacts = artifacts
        self._client_factory = client_factory
        self._local = threading.local()

    def _client(self) -> Any:
        client = getattr(self._local, "client", None)
        if client is None:
            if self._client_factory is not None:
                client = self._client_factory(self.endpoint)
            else:
                from openai import OpenAI

                client = OpenAI(
                    api_key=self.endpoint.api_key,
                    base_url=self.endpoint.base_url,
                    max_retries=0,
                    timeout=self.endpoint.timeout_seconds,
                    default_headers={"X-vgate-class": "agent-loop"},
                )
            self._local.client = client
        return client

    def _content(self, item: Mapping[str, Any]) -> dict[str, Any]:
        if item.get("type") != "input_image":
            return dict(item)
        payload = self.artifacts.read_bytes(str(item["artifact_sha256"]))
        encoding = dict(item.get("encoding") or {})
        if encoding.get("passthrough"):
            # E4: the histogram board prints digits; re-encoding it to JPEG or
            # downscaling it would smear them, so its stored bytes go out as they are.
            rendered = {
                "type": "input_image",
                "image_url": f"data:{item.get('media_type') or 'image/png'};base64,"
                + base64.b64encode(payload).decode("ascii"),
                "detail": str(item.get("detail") or "high"),
            }
            if "prompt_cache_breakpoint" in item:
                rendered["prompt_cache_breakpoint"] = dict(item["prompt_cache_breakpoint"])
            return rendered
        longest_edge = int(encoding.get("longest_edge", 512))
        quality = int(encoding.get("quality", 85))
        subsampling = int(encoding.get("subsampling", 2))
        with Image.open(io.BytesIO(payload)) as source:
            image = source.convert("RGB")
            if max(image.size) > longest_edge:
                scale = longest_edge / max(image.size)
                size = tuple(max(1, round(value * scale)) for value in image.size)
                image = image.resize(size, getattr(Image, "Resampling", Image).LANCZOS)
            output = io.BytesIO()
            image.save(
                output, "JPEG", quality=quality, optimize=False, progressive=False,
                subsampling=subsampling,
            )
        rendered = {
            "type": "input_image",
            "image_url": "data:image/jpeg;base64," + base64.b64encode(
                output.getvalue()
            ).decode("ascii"),
            "detail": str(item.get("detail") or "low"),
        }
        if "prompt_cache_breakpoint" in item:
            rendered["prompt_cache_breakpoint"] = dict(item["prompt_cache_breakpoint"])
        return rendered

    def send_once(self, spec: RequestSpec, _attempt: int) -> dict[str, Any]:
        request = spec.canonical
        behavior = request["behavior"]
        payload: dict[str, Any] = {
            "model": request["model"],
            "input": [
                {"role": message["role"],
                 "content": [self._content(item) for item in message["content"]]}
                for message in request["input"]
            ],
            "stream": True,
            "temperature": behavior["temperature"],
            "max_output_tokens": behavior["max_output_tokens"],
            "store": behavior["store"],
            "prompt_cache_key": spec.prompt_cache_key,
            "text": {"format": {
                "type": "json_schema", "name": request["schema_name"], "strict": True,
                "schema": _hosted_schema(request["schema"])
                if self.endpoint.provider_kind == "hosted" else request["schema"],
            }},
        }
        if behavior.get("reasoning_effort") is not None:
            payload["reasoning"] = {"effort": behavior["reasoning_effort"]}
        if behavior.get("reasoning_effort") is None:
            payload["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        try:
            result = consume_stream(self._client().responses.create(**payload))
        except Exception as exc:
            if isinstance(exc, (TransportError, ModelSubstituted)):
                raise
            response = getattr(exc, "response", None)
            headers = getattr(response, "headers", {}) or {}
            retry_value = None
            if hasattr(headers, "get"):
                retry_value = headers.get("retry-after") or headers.get("Retry-After")
            try:
                retry_after = float(retry_value) if retry_value is not None else None
            except (TypeError, ValueError):
                retry_after = None
            status = getattr(exc, "status_code", None)
            body = getattr(exc, "body", None)
            provider_code = None
            provider_param = None
            if isinstance(body, Mapping):
                error = body.get("error") if isinstance(body.get("error"), Mapping) else body
                provider_code = error.get("code") or error.get("type")
                provider_param = error.get("param")
            detail = str(provider_code or type(exc).__name__)
            if provider_param:
                detail += f":{provider_param}"
            code = f"{status}:{detail}" if status else detail
            raise TransportError(code, retry_after=retry_after) from exc
        if result["model"] != request["model"]:
            raise ModelSubstituted("provider_returned_different_model")
        try:
            result["parsed"] = json.loads(result["text"])
        except json.JSONDecodeError:
            result["parsed"] = None
        return result


class CachedResponsesClient:
    def __init__(self, adapter: ResponsesAdapter, cache: ExactResponseCache) -> None:
        self.adapter = adapter
        self.cache = cache

    def request(
        self, spec: RequestSpec,
        validate_extra: Callable[[dict[str, Any]], str | None] | None = None,
        *, bypass_exact_cache: bool = False,
    ) -> CachedResult:
        stage = str(spec.canonical["stage"])

        def validate(response: dict[str, Any]) -> str | None:
            if response.get("model") != spec.canonical["model"]:
                return "model_substitution"
            parsed = response.get("parsed")
            if parsed is None:
                return "invalid_json"
            error = semantic_error(stage, parsed)
            return error if error is not None or validate_extra is None else validate_extra(parsed)

        if bypass_exact_cache:
            from .api_cache import CacheExhausted

            attempts = max(1, self.adapter.endpoint.attempts)
            last_error = "invalid_response"
            for attempt in range(1, attempts + 1):
                retry_after = None
                try:
                    response = self.adapter.send_once(spec, attempt)
                    error = validate(response)
                    if error is None:
                        return CachedResult(
                            spec.request_hash, response,
                            dict(response.get("usage") or {}), False, attempt,
                        )
                    last_error = error
                except TransportError as exc:
                    last_error = exc.code
                    if not retryable_exception(exc):
                        raise CacheExhausted(last_error, attempt) from exc
                    retry_after = exc.retry_after
                if attempt < attempts:
                    delay = float(retry_after) if retry_after is not None else \
                        2 ** (attempt - 1) + random.random() * 0.25
                    time.sleep(min(max(delay, 0.0), 60.0))
            raise CacheExhausted(last_error, attempts)

        return self.cache.execute(
            spec, self.adapter.send_once, validate, attempts=self.adapter.endpoint.attempts,
            wait_timeout=self.adapter.endpoint.timeout_seconds * self.adapter.endpoint.attempts + 60,
        )


__all__ = [
    "CachedResponsesClient", "ModelSubstituted", "ResponsesAdapter", "TransportError",
    "consume_response", "consume_stream",
]
