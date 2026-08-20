"""Canonical request hashing and durable singleflight response caching."""
from __future__ import annotations

import hashlib
import json
import random
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from .persistence import AuditStore


_FORBIDDEN_HASH_KEYS = frozenset({
    "api_key", "timeout", "timeout_seconds", "trace_id", "request_id",
    "worker_id", "lease_id", "created_at", "retry_at",
})


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _assert_safe_manifest(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in _FORBIDDEN_HASH_KEYS:
                raise ValueError(f"non-behavioral or secret field in canonical request: {key}")
            _assert_safe_manifest(child)
    elif isinstance(value, list):
        for child in value:
            _assert_safe_manifest(child)


@dataclass(frozen=True, slots=True)
class RequestSpec:
    canonical: dict[str, Any]
    prompt_cache_key: str

    def __post_init__(self) -> None:
        _assert_safe_manifest(self.canonical)

    @property
    def request_hash(self) -> str:
        return hashlib.sha256(canonical_json(self.canonical).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CachedResult:
    request_hash: str
    response: dict[str, Any]
    usage: dict[str, Any]
    cache_hit: bool
    attempt_no: int


class CacheExhausted(RuntimeError):
    def __init__(self, error_type: str, attempts: int):
        super().__init__(error_type)
        self.error_type = error_type
        self.attempts = attempts


def retryable_exception(exc: BaseException) -> bool:
    code = str(getattr(exc, "code", "") or "")
    status_text = code.split(":", 1)[0]
    if status_text.isdigit():
        status = int(status_text)
        return status in {408, 409, 425, 429} or status >= 500
    if type(exc).__name__ in {"ModelSubstituted", "TypeError", "ValueError"}:
        return False
    return True


class ExactResponseCache:
    def __init__(self, store: AuditStore, *, lease_seconds: int = 180) -> None:
        self.store = store
        self.lease_seconds = max(1, int(lease_seconds))

    def execute(
        self,
        spec: RequestSpec,
        send_once: Callable[[RequestSpec, int], dict[str, Any]],
        validate: Callable[[dict[str, Any]], str | None],
        *,
        attempts: int = 3,
        wait_timeout: float | None = None,
    ) -> CachedResult:
        owner = uuid.uuid4().hex
        deadline = time.monotonic() + (
            float(wait_timeout) if wait_timeout is not None else self.lease_seconds * 3
        )
        recorded_wait = False
        while True:
            claim = self.store.acquire_request(
                spec.request_hash, spec.canonical, owner, self.lease_seconds
            )
            if claim["action"] == "resolved":
                self.store.record_cache_event(spec.request_hash, "hit")
                return CachedResult(
                    spec.request_hash, dict(claim["response"]), dict(claim.get("usage") or {}),
                    True, int(claim.get("attempt_no") or 0),
                )
            if claim["action"] == "owner":
                self.store.record_cache_event(spec.request_hash, "miss")
                break
            if not recorded_wait:
                self.store.record_cache_event(spec.request_hash, "wait")
                recorded_wait = True
            if time.monotonic() >= deadline:
                raise TimeoutError("exact_cache_wait_timeout")
            remaining = max(0.01, float(claim["lease_expires_at"]) - time.time())
            time.sleep(min(0.1, remaining))

        attempts = max(1, int(attempts))
        last_error = "invalid_response"
        for index in range(1, attempts + 1):
            retry_after = None
            try:
                response = send_once(spec, index)
                error = validate(response)
                usage = dict(response.get("usage") or {})
                if error is None:
                    attempt_no = self.store.record_attempt(
                        spec.request_hash, owner, response=response, usage=usage,
                        validation={"valid": True}, valid=True,
                    )
                    return CachedResult(
                        spec.request_hash, response, usage, False, attempt_no
                    )
                last_error = error
                self.store.record_attempt(
                    spec.request_hash, owner, response=response, usage=usage,
                    validation={"valid": False, "reason": error}, valid=False,
                    error_type="schema_or_semantic_validation",
                )
            except Exception as exc:  # retry boundary deliberately contains provider errors
                last_error = str(getattr(exc, "code", None) or type(exc).__name__)
                self.store.record_attempt(
                    spec.request_hash, owner, response=None, usage=None,
                    validation={"valid": False, "reason": last_error}, valid=False,
                    error_type=last_error,
                )
                if not retryable_exception(exc):
                    self.store.release_request(spec.request_hash, owner)
                    raise CacheExhausted(last_error, index) from exc
                retry_after = getattr(exc, "retry_after", None)
            if index < attempts:
                delay = float(retry_after) if retry_after is not None else \
                    2 ** (index - 1) + random.random() * 0.25
                time.sleep(min(max(delay, 0.0), 60.0))
        self.store.release_request(spec.request_hash, owner)
        raise CacheExhausted(last_error, attempts)


def prefix_cache_key(
    *, model_family: str, prompt_revision: str, stage: str, prefix: Any
) -> str:
    identity = {
        "model_family": model_family,
        "prompt_revision": prompt_revision,
        "stage": stage,
        "prefix": prefix,
    }
    digest = hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()[:40]
    stage_tag = hashlib.sha256(stage.encode("utf-8")).hexdigest()[:8]
    return f"vr:{stage_tag}:{digest}"


__all__ = [
    "CacheExhausted", "CachedResult", "ExactResponseCache", "RequestSpec",
    "canonical_json", "prefix_cache_key", "retryable_exception",
]
