"""Official-SDK Responses transport for retained local VLM utilities."""
from __future__ import annotations

import threading
import base64
import io
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from dataset_build.responses_events import is_official_response_event


@dataclass(frozen=True, slots=True)
class ResponsesText:
    text: str
    attempt: int
    model: str | None = None


class ResponsesVlmError(RuntimeError):
    """Sanitized terminal transport failure after bounded attempts."""

    def __init__(self, error_type: str, attempts: int):
        super().__init__(error_type)
        self.error_type = error_type
        self.attempts = attempts


class ModelSubstituted(RuntimeError):
    """The relay answered as a model other than the one that was requested."""


_LOCAL = threading.local()


def encode_image_data_url(
    path: str, *, longest_edge: int = 768, quality: int = 88
) -> str:
    from PIL import Image, ImageOps

    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        if max(image.size) > longest_edge:
            scale = longest_edge / max(image.size)
            size = tuple(max(1, int(round(value * scale))) for value in image.size)
            resampling = getattr(Image, "Resampling", Image)
            image = image.resize(size, resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")


# Hosted strict structured-output implements a narrower JSON Schema subset than
# local guided decoding.  ``uniqueItems`` is the keyword this campaign hit: with
# it, gpt-5.6-terra answers every request with a Cloudflare 502 (origin error,
# not a 400 naming the field) — bisected 2026-08-12 on the sam3_subject_selection
# schema, where dropping that one keyword turns a 100% failure into a 100% pass.
# No guarantee is lost: every caller re-validates the parsed object anyway.
_HOSTED_UNSUPPORTED_SCHEMA_KEYWORDS = frozenset({"uniqueItems"})


def _hosted_schema(node: Any) -> Any:
    if isinstance(node, Mapping):
        return {
            key: _hosted_schema(value) for key, value in node.items()
            if key not in _HOSTED_UNSUPPORTED_SCHEMA_KEYWORDS
        }
    if isinstance(node, list):
        return [_hosted_schema(item) for item in node]
    return node


def _client(base_url: str, api_key: str, timeout: float, vgate_class: str) -> Any:
    clients = getattr(_LOCAL, "clients", None)
    if clients is None:
        clients = {}
        _LOCAL.clients = clients
    key = (base_url.rstrip("/"), api_key, float(timeout), vgate_class)
    if key not in clients:
        from openai import OpenAI

        clients[key] = OpenAI(
            api_key=api_key,
            base_url=key[0],
            max_retries=0,
            timeout=timeout,
            default_headers={"X-vgate-class": vgate_class},
        )
    return clients[key]


def consume_text(stream: Any) -> str:
    """Consume only official typed streaming events through response.completed."""
    return consume_response(stream)[0]


def consume_response(stream: Any) -> tuple[str, str | None]:
    """As ``consume_text``, but also return the model id the relay answered as.

    External relays have substituted models before, so the caller needs the
    echoed id to police it; local vLLM answers with whatever name it was served
    under and its callers simply ignore the second element.
    """
    from openai.types.responses import (
        ResponseCompletedEvent,
        ResponseErrorEvent,
        ResponseFailedEvent,
        ResponseTextDeltaEvent,
    )

    chunks: list[str] = []
    completed: Any = None
    context = stream if hasattr(stream, "__enter__") else nullcontext(stream)
    with context as events:
        for event in events:
            if not is_official_response_event(event):
                raise RuntimeError("untyped_responses_event")
            if isinstance(event, ResponseTextDeltaEvent):
                chunks.append(event.delta)
            elif isinstance(event, ResponseCompletedEvent):
                completed = event.response
            elif isinstance(event, ResponseFailedEvent):
                error = getattr(event.response, "error", None)
                raise RuntimeError(str(getattr(error, "code", None) or "response_failed"))
            elif isinstance(event, ResponseErrorEvent):
                raise RuntimeError(str(event.code or "response_error"))
    if completed is None:
        raise RuntimeError("responses_stream_interrupted")
    if getattr(completed, "status", None) not in (None, "completed"):
        raise RuntimeError("responses_not_completed")
    raw = "".join(chunks) or str(getattr(completed, "output_text", "") or "")
    if not raw:
        raise RuntimeError("responses_output_empty")
    return raw, str(getattr(completed, "model", "") or "") or None


def request_text(
    *,
    base_url: str,
    api_key: str,
    model: str,
    content: Sequence[Mapping[str, Any]],
    schema_name: str,
    schema: Mapping[str, Any],
    timeout: float,
    temperature: float = 0.1,
    max_output_tokens: int = 320,
    attempts: int = 3,
    vgate_class: str = "qa-judge",
    client_factory: Callable[[str, str, float, str], Any] | None = None,
    reasoning_effort: str | None = None,
    expect_model: str | None = None,
) -> ResponsesText:
    """Issue one strict Responses request with bounded transport retries.

    ``reasoning_effort`` switches the payload from the local-vLLM shape to the
    external-relay shape: the thinking switch is a vLLM chat-template kwarg that
    a hosted reasoning model rejects, and the effort knob is the reverse.
    ``expect_model`` makes a substituted model a retryable failure instead of a
    silently mislabelled row.
    """
    payload = {
        "model": model,
        "input": [{"role": "user", "content": [dict(item) for item in content]}],
        "stream": True,
        "max_output_tokens": int(max_output_tokens),
        "temperature": float(temperature),
        "text": {"format": {
            "type": "json_schema",
            "name": schema_name,
            "strict": True,
            "schema": dict(schema) if reasoning_effort is None else _hosted_schema(schema),
        }},
    }
    if reasoning_effort is None:
        payload["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    else:
        payload["reasoning"] = {"effort": str(reasoning_effort)}
    factory = client_factory or _client
    attempts = max(1, int(attempts))
    last_error = "transport_error"
    for attempt in range(1, attempts + 1):
        try:
            client = factory(base_url, api_key, timeout, vgate_class)
            text, returned = consume_response(client.responses.create(**payload))
            if expect_model is not None and returned != expect_model:
                raise ModelSubstituted(
                    f"requested {expect_model!r} but the relay answered as {returned!r}")
            return ResponsesText(text, attempt, returned)
        except Exception as exc:  # noqa: BLE001 - retry boundary is deliberately broad
            last_error = type(exc).__name__
    raise ResponsesVlmError(last_error, attempts)


__all__ = [
    "ModelSubstituted", "ResponsesText", "ResponsesVlmError", "consume_response",
    "consume_text", "encode_image_data_url", "request_text",
]
