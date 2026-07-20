"""Pinned OpenAI SDK event-type validation shared by Responses consumers."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, get_args


def _typed_event_classes(surface: Any) -> tuple[type[Any], ...]:
    if isinstance(surface, type):
        return (surface,)
    return tuple(
        event_class
        for argument in get_args(surface)
        for event_class in _typed_event_classes(argument)
    )


@lru_cache(maxsize=1)
def _event_contract() -> dict[type[Any], frozenset[str]]:
    from openai.types.responses.response_stream_event import ResponseStreamEvent

    classes = _typed_event_classes(ResponseStreamEvent)
    contract: dict[type[Any], frozenset[str]] = {}
    for event_class in classes:
        annotation = getattr(event_class, "__annotations__", {}).get("type")
        if annotation is None:
            field = getattr(event_class, "model_fields", {}).get("type")
            annotation = getattr(field, "annotation", None)
        values = frozenset(value for value in get_args(annotation) if isinstance(value, str))
        if values:
            contract[event_class] = values
    if not contract:
        raise RuntimeError("pinned OpenAI SDK exposes no typed Responses events")
    return contract


def is_official_response_event(event: Any) -> bool:
    """Return true only for a pinned-SDK event with its declared discriminator."""
    for event_class, event_types in _event_contract().items():
        if isinstance(event, event_class):
            return getattr(event, "type", None) in event_types
    return False


def validate_response_event_surface() -> None:
    """Force SDK union introspection during preflight."""
    _event_contract()


__all__ = ["is_official_response_event", "validate_response_event_surface"]
