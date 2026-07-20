"""Canonical OpenAI Responses annotation transport and durable queue drain."""
from __future__ import annotations

import base64
import email.utils
import importlib.metadata
import io
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Protocol

from PIL import Image, ImageOps

from dataset_build.responses_events import (
    is_official_response_event,
    validate_response_event_surface,
)

from .config import (
    AnnotationConfig,
    ExternalEndpointConfig,
    LocalAnnotationConfig,
    redact_text,
    uri_secrets,
)
from .state import ArtifactStore, stable_id


PINNED_OPENAI_VERSION = "2.46.0"
ANNOTATION_FIELDS = (
    "problem_lighting",
    "plan_lighting",
    "problem_global_color",
    "plan_global_color",
    "problem_specific_color",
    "plan_specific_color",
    "instruction_long",
    "instruction_short",
)
REASONING_FIELDS = ANNOTATION_FIELDS[:6]
QUOTA_CODES = frozenset({
    "insufficient_quota",
    "billing_hard_limit_reached",
    "billing_not_active",
    "quota_exceeded",
})

ANNOTATION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        **{
            name: {"type": "string", "minLength": 8}
            for name in REASONING_FIELDS
        },
        "instruction_long": {"type": "string", "minLength": 8},
        "instruction_short": {"type": "string", "minLength": 4},
    },
    "required": list(ANNOTATION_FIELDS),
    "additionalProperties": False,
}

_ASPECT_TOKENS = {
    "lighting": (
        "<problem_light_start>", "<problem_light_end>",
        "<plan_light_start>", "<plan_light_end>",
    ),
    "global_color": (
        "<problem_globalcolor_start>", "<problem_globalcolor_end>",
        "<plan_globalcolor_start>", "<plan_globalcolor_end>",
    ),
    "specific_color": (
        "<problem_specificcolor_start>", "<problem_specificcolor_end>",
        "<plan_specificcolor_start>", "<plan_specificcolor_end>",
    ),
}

_SYSTEM_PROMPT = (
    "You write image-retouching training annotations. Compare the first image "
    "(before) with the second image (after) and return only the requested JSON. "
    "All prose must be English, except that a supplied non-English style name must "
    "be copied verbatim. Describe visible photographic problems and practical plans; "
    "do not mention measurements, masks, preset IDs, or implementation details."
)


class AnnotationError(RuntimeError):
    """A classified annotation failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool,
        quota: bool = False,
        retry_after: float | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.quota = quota
        self.retry_after = retry_after
        self.status_code = status_code


class ClientFactory(Protocol):
    def __call__(
        self, endpoint: ExternalEndpointConfig | LocalAnnotationConfig, route: str
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class PreparedTask:
    prompt: str
    before_data_url: str
    after_data_url: str


@dataclass(frozen=True, slots=True)
class StreamResult:
    fields: dict[str, str]
    returned_model: str | None
    usage: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AttemptResult:
    fields: dict[str, str]
    route: str
    endpoint_id: str
    returned_model: str | None
    usage: dict[str, Any]
    attempt: int
    round: int


@dataclass(slots=True)
class _RelayState:
    config: ExternalEndpointConfig
    inflight: int = 0
    removed: bool = False


def preflight_openai_sdk() -> None:
    """Fail before mutation unless the pinned SDK exposes typed Responses streaming."""
    try:
        version = importlib.metadata.version("openai")
        from openai import OpenAI
        from openai.types.responses import (
            ResponseCompletedEvent,
            ResponseErrorEvent,
            ResponseFailedEvent,
            ResponseTextDeltaEvent,
        )
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise AnnotationError(
            "openai_sdk_missing",
            f"openai=={PINNED_OPENAI_VERSION} is required",
            retryable=False,
        ) from exc
    if version != PINNED_OPENAI_VERSION:
        raise AnnotationError(
            "openai_sdk_version",
            f"openai=={PINNED_OPENAI_VERSION} is required, found {version}",
            retryable=False,
        )
    client = OpenAI(api_key="preflight", base_url="http://127.0.0.1/v1", max_retries=0)
    if not callable(getattr(client.responses, "create", None)):
        raise AnnotationError(
            "responses_sdk_unavailable", "OpenAI Responses.create is unavailable", retryable=False
        )
    for event_type in (
        ResponseTextDeltaEvent,
        ResponseCompletedEvent,
        ResponseFailedEvent,
        ResponseErrorEvent,
    ):
        if not hasattr(event_type, "model_fields"):
            raise AnnotationError(
                "responses_sdk_unavailable", "typed Responses stream events are unavailable",
                retryable=False,
            )
    try:
        validate_response_event_surface()
    except RuntimeError as exc:
        raise AnnotationError(
            "responses_sdk_unavailable", str(exc), retryable=False
        ) from exc


def assemble_reasoning(parts: Mapping[str, str]) -> str:
    aspects = ("lighting", "global_color", "specific_color")
    tokens = tuple(_ASPECT_TOKENS[name] for name in aspects)
    problem = (
        str(parts["problem_lighting"]),
        str(parts["problem_global_color"]),
        str(parts["problem_specific_color"]),
    )
    plan = (
        str(parts["plan_lighting"]),
        str(parts["plan_global_color"]),
        str(parts["plan_specific_color"]),
    )
    return "".join(row[0] + value + row[1] for row, value in zip(tokens, problem)) + \
        "".join(row[2] + value + row[3] for row, value in zip(tokens, plan))


def parse_annotation_json(raw: str) -> dict[str, str]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AnnotationError(
            "schema_failed", "response was not strict JSON", retryable=False
        ) from exc
    if not isinstance(value, dict) or set(value) != set(ANNOTATION_FIELDS):
        raise AnnotationError(
            "schema_failed", "response keys did not match the annotation schema", retryable=False
        )
    parsed: dict[str, str] = {}
    for name in ANNOTATION_FIELDS:
        item = value[name]
        minimum = 4 if name == "instruction_short" else 8
        if not isinstance(item, str) or len(item.strip()) < minimum:
            raise AnnotationError(
                "schema_failed", f"response field {name} violated minLength", retryable=False
            )
        parsed[name] = item.strip()
    return parsed


def encode_image_data_url(
    path: str | Path, *, longest_edge: int = 768, quality: int = 90
) -> str:
    source = Path(path)
    try:
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            scale = min(1.0, longest_edge / max(image.size))
            if scale < 1.0:
                size = (
                    max(1, round(image.width * scale)),
                    max(1, round(image.height * scale)),
                )
                resampling = getattr(Image, "Resampling", Image)
                image = image.resize(size, resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=quality, optimize=True)
    except (OSError, ValueError) as exc:
        raise AnnotationError(
            "annotation_image_invalid", f"cannot encode annotation image: {source}",
            retryable=False,
        ) from exc
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + encoded


def _objective_hints(candidate: Mapping[str, Any]) -> str:
    hints = candidate.get("objective_hints")
    required = ("brightness", "warmth", "chroma", "contrast")
    if not isinstance(hints, Mapping) or any(name not in hints for name in required):
        raise AnnotationError(
            "annotation_task_invalid",
            "candidate objective_hints must contain brightness, warmth, chroma, and contrast",
            retryable=False,
        )
    return "; ".join(f"{name}: {hints[name]}" for name in required)


def _subject_name(value: Any) -> str:
    if isinstance(value, Mapping):
        for key in ("name", "subject_name", "label", "class_name"):
            if str(value.get(key) or "").strip():
                return str(value[key]).strip()
        return "subject"
    return str(value or "subject").strip() or "subject"


def build_prompt(task: Mapping[str, Any]) -> str:
    group = task.get("group")
    candidate = task.get("candidate")
    if not isinstance(group, Mapping) or not isinstance(candidate, Mapping):
        raise AnnotationError(
            "annotation_task_invalid", "annotation task is missing group/candidate", retryable=False
        )
    hints = _objective_hints(candidate)
    mode = str(group.get("render_mode") or "").lower()
    if mode == "global":
        style_name = str(candidate.get("style_name") or "").strip()
        if not style_name:
            raise AnnotationError(
                "annotation_task_invalid", "global candidate is missing style_name",
                retryable=False,
            )
        task_clause = (
            f'This is a global style task. Both instructions must name the style "{style_name}" '
            "verbatim and describe how to reproduce the visible overall look."
        )
    elif mode == "local":
        subject = _subject_name(candidate.get("subject") or group.get("subject"))
        region = str(candidate.get("region") or "").strip()
        if not region:
            raise AnnotationError(
                "annotation_task_invalid", "local candidate is missing coarse region",
                retryable=False,
            )
        task_clause = (
            f'This is a local task affecting the subject "{subject}" in the {region}. '
            "Describe that subject/region without naming a preset or style."
        )
    else:
        raise AnnotationError(
            "annotation_task_invalid", "render_mode must be local or global", retryable=False
        )
    return (
        "The images are ordered before, then after. " + task_clause + "\n"
        "Objective direction hints computed from the true edit support: " + hints + ".\n"
        "Fill all eight schema fields. The six problem/plan fields must be substantive."
    )


def prepare_task(task: Mapping[str, Any], config: AnnotationConfig) -> PreparedTask:
    group = task.get("group")
    candidate = task.get("candidate")
    if not isinstance(group, Mapping) or not isinstance(candidate, Mapping):
        raise AnnotationError(
            "annotation_task_invalid", "annotation task is missing group/candidate", retryable=False
        )
    before = group.get("source_path")
    after = candidate.get("after_path")
    if not before or not after:
        raise AnnotationError(
            "annotation_task_invalid", "annotation task is missing before/after paths",
            retryable=False,
        )
    return PreparedTask(
        prompt=build_prompt(task),
        before_data_url=encode_image_data_url(
            str(before), longest_edge=config.image_long_edge,
            quality=config.image_jpeg_quality,
        ),
        after_data_url=encode_image_data_url(
            str(after), longest_edge=config.image_long_edge,
            quality=config.image_jpeg_quality,
        ),
    )


def request_payload(
    prepared: PreparedTask,
    config: AnnotationConfig,
    *,
    route: str,
) -> dict[str, Any]:
    content = [
        {"type": "input_text", "text": prepared.prompt},
        {"type": "input_image", "image_url": prepared.before_data_url},
        {"type": "input_image", "image_url": prepared.after_data_url},
    ]
    payload: dict[str, Any] = {
        "instructions": _SYSTEM_PROMPT,
        "input": [{"role": "user", "content": content}],
        "stream": True,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "veraretouch_annotation",
                "strict": True,
                "schema": ANNOTATION_JSON_SCHEMA,
            }
        },
    }
    if route == "external":
        payload.update({
            "model": config.external_model,
            "reasoning": {"effort": config.external_reasoning_effort},
            "max_output_tokens": config.external_max_output_tokens,
        })
    elif route == "local":
        payload.update({
            "model": config.local.model,
            "temperature": config.local.temperature,
            "max_output_tokens": config.local.max_output_tokens,
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": config.local.enable_thinking,
                }
            },
        })
    else:
        raise ValueError(f"unsupported annotation route: {route}")
    return payload


def _response_usage(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        result = dump(mode="json", exclude_none=True)
        return dict(result) if isinstance(result, Mapping) else {}
    return {}


def _consume_stream(stream: Any) -> StreamResult:
    from openai.types.responses import (
        ResponseCompletedEvent,
        ResponseErrorEvent,
        ResponseFailedEvent,
        ResponseTextDeltaEvent,
    )

    chunks: list[str] = []
    completed: Any = None
    context = stream if hasattr(stream, "__enter__") else _null_context(stream)
    with context as events:
        for event in events:
            if not is_official_response_event(event):
                raise AnnotationError(
                    "untyped_stream_event", "Responses stream returned an untyped event",
                    retryable=True,
                )
            if isinstance(event, ResponseTextDeltaEvent):
                chunks.append(event.delta)
            elif isinstance(event, ResponseCompletedEvent):
                completed = event.response
            elif isinstance(event, ResponseFailedEvent):
                error = getattr(event.response, "error", None)
                code = str(getattr(error, "code", None) or "response_failed")
                message = str(getattr(error, "message", None) or "Responses stream failed")
                raise AnnotationError(
                    code, message, retryable=True, quota=code in QUOTA_CODES
                )
            elif isinstance(event, ResponseErrorEvent):
                code = str(event.code or "response_error")
                raise AnnotationError(
                    code, event.message, retryable=True, quota=code in QUOTA_CODES
                )
    if completed is None:
        raise AnnotationError(
            "stream_interrupted", "Responses stream ended before response.completed",
            retryable=True,
        )
    status = getattr(completed, "status", None)
    if status not in (None, "completed"):
        raise AnnotationError(
            "response_not_completed", f"Responses status was {status}", retryable=True
        )
    raw = "".join(chunks)
    if not raw:
        raw = str(getattr(completed, "output_text", "") or "")
    fields = parse_annotation_json(raw)
    return StreamResult(
        fields=fields,
        returned_model=str(getattr(completed, "model", "") or "") or None,
        usage=_response_usage(getattr(completed, "usage", None)),
    )


@contextmanager
def _null_context(value: Any) -> Iterator[Any]:
    yield value


def _retry_after(headers: Any) -> float | None:
    if not headers:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        try:
            parsed = email.utils.parsedate_to_datetime(str(raw))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def _error_code(exc: BaseException) -> str:
    direct = getattr(exc, "code", None)
    if direct:
        return str(direct)
    body = getattr(exc, "body", None)
    if isinstance(body, Mapping):
        nested = body.get("error")
        if isinstance(nested, Mapping) and nested.get("code"):
            return str(nested["code"])
        if body.get("code"):
            return str(body["code"])
    return ""


def classify_exception(exc: BaseException) -> AnnotationError:
    if isinstance(exc, AnnotationError):
        return exc
    status = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    retry_after = _retry_after(getattr(response, "headers", None))
    code = _error_code(exc)
    quota = code in QUOTA_CODES
    message = str(exc) or type(exc).__name__
    if quota:
        return AnnotationError(
            code, message, retryable=True, quota=True, retry_after=retry_after,
            status_code=status,
        )
    if status == 429:
        return AnnotationError(
            code or "rate_limited", message, retryable=True, retry_after=retry_after,
            status_code=status,
        )
    if isinstance(status, int) and status >= 500:
        return AnnotationError(
            code or "upstream_5xx", message, retryable=True, retry_after=retry_after,
            status_code=status,
        )
    if isinstance(status, int) and 400 <= status < 500:
        return AnnotationError(
            code or "request_4xx", message, retryable=False, status_code=status
        )
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)) or type(exc).__name__ in {
        "APIConnectionError", "APITimeoutError",
    }:
        return AnnotationError("network_error", message, retryable=True)
    return AnnotationError("transport_error", message, retryable=True)


class ExternalRelayPool:
    """Thread-safe least-inflight pool with round-robin tie breaking."""

    def __init__(
        self,
        endpoints: tuple[ExternalEndpointConfig, ...],
        *,
        removed_ids: set[str] | None = None,
        exhausted: bool = False,
    ) -> None:
        self._states = [_RelayState(endpoint) for endpoint in endpoints]
        removed = removed_ids or set()
        for state in self._states:
            state.removed = exhausted or state.config.id in removed
        self._condition = threading.Condition(threading.Lock())
        self._cursor = 0

    @property
    def exhausted(self) -> bool:
        with self._condition:
            return not any(not state.removed for state in self._states)

    @property
    def active_ids(self) -> tuple[str, ...]:
        with self._condition:
            return tuple(state.config.id for state in self._states if not state.removed)

    def remove(self, endpoint_id: str) -> bool:
        """Remove one endpoint and return whether the pool is now exhausted."""
        with self._condition:
            for state in self._states:
                if state.config.id == endpoint_id:
                    state.removed = True
                    break
            self._condition.notify_all()
            return not any(not state.removed for state in self._states)

    @contextmanager
    def lease(self) -> Iterator[ExternalEndpointConfig | None]:
        state = self._acquire()
        if state is None:
            yield None
            return
        try:
            yield state.config
        finally:
            with self._condition:
                state.inflight -= 1
                self._condition.notify_all()

    def _acquire(self) -> _RelayState | None:
        with self._condition:
            while True:
                active = [state for state in self._states if not state.removed]
                if not active:
                    return None
                available = [
                    state for state in active
                    if state.inflight < state.config.concurrency
                ]
                if not available:
                    self._condition.wait()
                    continue
                minimum = min(state.inflight for state in available)
                tied = {id(state) for state in available if state.inflight == minimum}
                selected: _RelayState | None = None
                for offset in range(len(self._states)):
                    index = (self._cursor + offset) % len(self._states)
                    if id(self._states[index]) in tied:
                        selected = self._states[index]
                        self._cursor = (index + 1) % len(self._states)
                        break
                assert selected is not None
                selected.inflight += 1
                return selected


class ResponsesAnnotator:
    """Official-SDK transport plus durable three-round annotation drain."""

    def __init__(
        self,
        config: AnnotationConfig,
        store: ArtifactStore,
        *,
        client_factory: ClientFactory | None = None,
        sleep: Callable[[float], None] = time.sleep,
        random_value: Callable[[], float] = random.random,
    ) -> None:
        self.config = config
        self.store = store
        self._client_factory = client_factory or self._default_client
        self._sleep = sleep
        self._random_value = random_value
        self._client_lock = threading.Lock()
        self._pool_state_lock = threading.Lock()
        self._clients: dict[str, Any] = {}
        self._secrets = tuple(
            secret
            for endpoint in config.external_endpoints
            for secret in (endpoint.api_key, *uri_secrets(endpoint.base_url))
        ) + (config.local.api_key, *uri_secrets(config.local.base_url))
        removed = {
            str(row.get("endpoint_id"))
            for row in store.failures
            if row.get("error_code") == "external_endpoint_exhausted"
        }
        self.pool = ExternalRelayPool(
            config.external_endpoints,
            removed_ids=removed,
            exhausted=store.external_pool_exhausted(),
        )

    @staticmethod
    def _default_client(
        endpoint: ExternalEndpointConfig | LocalAnnotationConfig, route: str
    ) -> Any:
        from openai import OpenAI

        return OpenAI(
            api_key=endpoint.api_key,
            base_url=endpoint.base_url,
            max_retries=0,
            timeout=180.0,
            default_headers={"X-vgate-class": "build-annotate"},
        )

    def _client(self, endpoint: ExternalEndpointConfig | LocalAnnotationConfig, route: str) -> Any:
        key = endpoint.id if isinstance(endpoint, ExternalEndpointConfig) else "local"
        with self._client_lock:
            if key not in self._clients:
                self._clients[key] = self._client_factory(endpoint, route)
            return self._clients[key]

    def _request(
        self,
        prepared: PreparedTask,
        endpoint: ExternalEndpointConfig | LocalAnnotationConfig,
        route: str,
    ) -> StreamResult:
        client = self._client(endpoint, route)
        payload = request_payload(prepared, self.config, route=route)
        try:
            stream = client.responses.create(**payload)
            return _consume_stream(stream)
        except BaseException as exc:
            raise classify_exception(exc) from exc

    def _failure(
        self,
        task: Mapping[str, Any],
        *,
        event_type: str,
        error_code: str,
        message: str,
        round_number: int,
        attempt: int,
        retryable: bool,
        terminal: bool,
        endpoint_id: str | None = None,
        durable: bool = False,
    ) -> bool:
        task_id = str(task["task_id"])
        event_id = stable_id(
            "failure", self.store.build_id, task_id, event_type, error_code,
            round_number, attempt, endpoint_id or "",
        )
        return self.store.append_failure({
            "build_id": self.store.build_id,
            "event_id": event_id,
            "event_type": event_type,
            "stage": "annotation",
            "task_id": task_id,
            "group_id": task.get("group_id"),
            "candidate_id": task.get("candidate_id"),
            "round": round_number,
            "attempt": attempt,
            "retryable": retryable,
            "error_code": error_code,
            "message": redact_text(message, self._secrets),
            "endpoint_id": endpoint_id,
            "terminal": terminal,
        }, durable=durable)

    def _persist_quota_removal(
        self,
        task: Mapping[str, Any],
        endpoint_id: str,
        round_number: int,
        attempt: int,
        exhausted: bool,
    ) -> None:
        with self._pool_state_lock:
            endpoint_recorded = any(
                row.get("error_code") == "external_endpoint_exhausted"
                and row.get("endpoint_id") == endpoint_id
                for row in self.store.failures
            )
            if not endpoint_recorded:
                self._failure(
                    task,
                    event_type="pool_state",
                    error_code="external_endpoint_exhausted",
                    message=f"external endpoint {endpoint_id} exhausted its permanent quota",
                    round_number=round_number,
                    attempt=attempt,
                    retryable=True,
                    terminal=False,
                    endpoint_id=endpoint_id,
                    durable=True,
                )
            if exhausted and not self.store.external_pool_exhausted():
                self._failure(
                task,
                event_type="pool_state",
                error_code="external_pool_exhausted",
                message="all external annotation endpoints exhausted permanent quota",
                round_number=round_number,
                attempt=attempt,
                retryable=True,
                terminal=False,
                endpoint_id=endpoint_id,
                durable=True,
                )

    def _append_sft(self, task: Mapping[str, Any], result: AttemptResult) -> None:
        group = task["group"]
        candidate = task["candidate"]
        mode = str(group["render_mode"])
        qa = dict(candidate.get("qa") or {})
        qa["annotation"] = {
            "route": result.route,
            "endpoint_id": result.endpoint_id,
            "returned_model": result.returned_model,
            "usage": result.usage,
            "attempt": result.attempt,
            "round": result.round,
            "status": "completed",
        }
        local: dict[str, Any] | None = None
        if mode == "local":
            local = {
                "slot_mode": candidate.get("slot_mode"),
                "mode_index": candidate.get("mode_index"),
                "mask_id": candidate.get("mask_id"),
                "C_GT": candidate.get("cgt_path"),
                "subject": candidate.get("subject") or group.get("subject"),
                "region": candidate.get("region"),
                "raw_alpha_mean": candidate.get("raw_alpha_mean"),
                "amount": candidate.get("amount"),
                "effective_alpha_mean": candidate.get("effective_alpha_mean"),
                "pairing_index": candidate.get("pairing_index"),
            }
        task_id = str(task["task_id"])
        self.store.append_sft({
            "build_id": self.store.build_id,
            "sft_id": stable_id("sft", task_id),
            "annotation_task_id": task_id,
            "group_id": task["group_id"],
            "candidate_id": task["candidate_id"],
            "winner_rank": task["winner_rank"],
            "I_in": group["source_path"],
            "I_tar": candidate["after_path"],
            "recipe": candidate.get("recipe") or candidate.get("preset_id"),
            "local": local,
            "task_type": "local" if mode == "local" else "style",
            "instruction": result.fields["instruction_long"],
            "instruction_short": result.fields["instruction_short"],
            "reasoning": assemble_reasoning(result.fields),
            "annot_src": (
                f"responses:external:{result.endpoint_id}"
                if result.route == "external" else "responses:local"
            ),
            "qa": qa,
        })
        self.store.checkpoint()

    def _attempt_count(self, task_id: str, round_number: int) -> int:
        return sum(
            1 for row in self.store.failures
            if row.get("task_id") == task_id
            and row.get("stage") == "annotation"
            and row.get("event_type") == "attempt"
            and row.get("round") == round_number
        )

    def _round_done(self, task_id: str, round_number: int) -> bool:
        return any(
            row.get("task_id") == task_id
            and row.get("event_type") == "round_exhausted"
            and row.get("round") == round_number
            for row in self.store.failures
        )

    def _next_round(self, task_id: str) -> int | None:
        if self.store.has_terminal_failure(task_id):
            return None
        for round_number in range(1, self.config.queue_rounds + 1):
            if not self._round_done(task_id, round_number):
                return round_number
        return None

    def run_round(self, task: Mapping[str, Any], round_number: int) -> str:
        task_id = str(task["task_id"])
        attempt = self._attempt_count(task_id, round_number)
        try:
            prepared = prepare_task(task, self.config)
        except AnnotationError as failure:
            self._failure(
                task, event_type="terminal", error_code=failure.code,
                message=str(failure), round_number=round_number, attempt=attempt,
                retryable=False, terminal=True, durable=True,
            )
            return "terminal"

        while attempt < self.config.transport_attempts_per_round:
            route = "local" if self.pool.exhausted else "external"
            endpoint: ExternalEndpointConfig | LocalAnnotationConfig | None
            lease = self.pool.lease() if route == "external" else _null_context(self.config.local)
            with lease as endpoint:
                if endpoint is None:
                    continue
                attempt += 1
                endpoint_id = endpoint.id if isinstance(endpoint, ExternalEndpointConfig) else "local"
                try:
                    stream_result = self._request(prepared, endpoint, route)
                except AnnotationError as failure:
                    if failure.quota and route == "external":
                        exhausted = self.pool.remove(endpoint_id)
                        self._persist_quota_removal(
                            task, endpoint_id, round_number, attempt, exhausted
                        )
                    self._failure(
                        task,
                        event_type="attempt",
                        error_code=failure.code,
                        message=str(failure),
                        round_number=round_number,
                        attempt=attempt,
                        retryable=failure.retryable,
                        terminal=False,
                        endpoint_id=endpoint_id,
                        durable=True,
                    )
                    if not failure.retryable:
                        self._failure(
                            task,
                            event_type="terminal",
                            error_code=failure.code,
                            message=str(failure),
                            round_number=round_number,
                            attempt=attempt,
                            retryable=False,
                            terminal=True,
                            endpoint_id=endpoint_id,
                            durable=True,
                        )
                        return "terminal"
                    if attempt < self.config.transport_attempts_per_round and not failure.quota:
                        delay = failure.retry_after
                        if delay is None:
                            delay = min(60.0, 2.0 ** (attempt - 1)) * (
                                0.75 + 0.5 * self._random_value()
                            )
                        self._sleep(delay)
                    continue
            result = AttemptResult(
                fields=stream_result.fields,
                route=route,
                endpoint_id=endpoint_id,
                returned_model=stream_result.returned_model,
                usage=stream_result.usage,
                attempt=attempt,
                round=round_number,
            )
            self._append_sft(task, result)
            return "completed"
        return "retryable_exhausted"

    def drain(self, *, max_workers: int | None = None) -> dict[str, int]:
        workers = max_workers or max(
            1, sum(endpoint.concurrency for endpoint in self.config.external_endpoints)
        )
        counts = {"completed": 0, "terminal": 0, "transport_failed": 0}
        for round_number in range(1, self.config.queue_rounds + 1):
            tasks = [
                task for task in self.store.pending_annotation_tasks()
                if self._next_round(str(task["task_id"])) == round_number
            ]
            if not tasks:
                continue
            with ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
                future_to_task = {
                    executor.submit(self.run_round, task, round_number): task
                    for task in tasks
                }
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        status = future.result()
                    except BaseException as exc:
                        failure = classify_exception(exc)
                        self._failure(
                            task,
                            event_type="terminal",
                            error_code="annotation_worker_failed",
                            message=str(failure),
                            round_number=round_number,
                            attempt=self._attempt_count(str(task["task_id"]), round_number),
                            retryable=False,
                            terminal=True,
                            durable=True,
                        )
                        counts["terminal"] += 1
                        continue
                    if status == "completed":
                        counts["completed"] += 1
                    elif status == "terminal":
                        counts["terminal"] += 1
                    elif round_number < self.config.queue_rounds:
                        self._failure(
                            task,
                            event_type="round_exhausted",
                            error_code="annotation_round_exhausted",
                            message="annotation transport attempts exhausted for queue round",
                            round_number=round_number,
                            attempt=self.config.transport_attempts_per_round,
                            retryable=True,
                            terminal=False,
                            durable=True,
                        )
                    else:
                        self._failure(
                            task,
                            event_type="terminal",
                            error_code="transport_failed",
                            message="annotation transport failed after all durable rounds",
                            round_number=round_number,
                            attempt=self.config.transport_attempts_per_round,
                            retryable=False,
                            terminal=True,
                            durable=True,
                        )
                        counts["transport_failed"] += 1
        self.store.checkpoint()
        counts["pending"] = len(self.store.pending_annotation_tasks())
        return counts
