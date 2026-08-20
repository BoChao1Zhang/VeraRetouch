"""Terra concurrency, per-source serialization, and prefix-affinity ordering."""
from __future__ import annotations

import contextlib
import hashlib
import heapq
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence


_STAGE_PRIORITY = {
    "local_repair": 0,
    "local_propose": 1,
    "global_propose": 2,
    "diagnose": 3,
    "preflight": 4,
}


@dataclass(order=True, slots=True)
class _Waiter:
    priority: tuple[int, int, int]
    ticket: int
    prefix_key: str


class TerraLimiter:
    def __init__(
        self, target: int = 16, *, campaign_id: str = "", lane_id: str = "",
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if not 1 <= target <= 16:
            raise ValueError("Terra target must be in [1, 16]")
        self.target = target
        self.lane_id = lane_id
        self.rate_limited = 0
        self.recovered = 0
        self.effective = target
        self.in_flight = 0
        self._success_window = 0
        self._condition = threading.Condition()
        self._source_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._queue: list[_Waiter] = []
        self._ticket = 0
        self._last_prefix = ""
        self.events: list[dict[str, Any]] = []
        self.campaign_id = campaign_id
        self._event_sink = event_sink

    def _emit(self, row: dict[str, Any]) -> None:
        row["lane_id"] = self.lane_id
        self.events.append(row)
        if self._event_sink is not None:
            self._event_sink(row)

    @contextlib.contextmanager
    def slot(
        self, source_id: str, stage: str, prefix_key: str
    ) -> Iterator[None]:
        source_lock = self._source_locks[source_id]
        with source_lock:
            with self._condition:
                ticket = self._ticket
                self._ticket += 1
                affinity = 0 if prefix_key == self._last_prefix else 1
                waiter = _Waiter(
                    (_STAGE_PRIORITY.get(stage, 9), affinity, ticket), ticket, prefix_key
                )
                heapq.heappush(self._queue, waiter)
                while self._queue[0].ticket != ticket or self.in_flight >= self.effective:
                    self._condition.wait()
                heapq.heappop(self._queue)
                self.in_flight += 1
                self._last_prefix = prefix_key
                self._emit({
                    "at": time.time(), "event": "start", "stage": stage,
                    "in_flight": self.in_flight, "effective_limit": self.effective,
                    "campaign_id": self.campaign_id, "source_id": source_id,
                    "prefix_key": prefix_key,
                })
            try:
                yield
            finally:
                with self._condition:
                    self.in_flight -= 1
                    self._emit({
                        "at": time.time(), "event": "finish", "stage": stage,
                        "in_flight": self.in_flight, "effective_limit": self.effective,
                        "campaign_id": self.campaign_id, "source_id": source_id,
                        "prefix_key": prefix_key,
                    })
                    self._condition.notify_all()

    def observe(self, error_type: str | None) -> None:
        lowered = error_type and (
            "429" in error_type or "ratelimit" in error_type.lower()
            or "overload" in error_type.lower()
        )
        with self._condition:
            if lowered:
                self.effective = max(1, self.effective - 1)
                self._success_window = 0
                self.rate_limited += 1
            elif error_type is None:
                self._success_window += 1
                if self._success_window >= 20 and self.effective < self.target:
                    self.effective += 1
                    self._success_window = 0
                    self.recovered += 1
            self._condition.notify_all()

    def metrics(self) -> dict[str, Any]:
        with self._condition:
            return {
                "lane_id": self.lane_id,
                "target": self.target, "effective": self.effective,
                "in_flight": self.in_flight, "rate_limited": self.rate_limited,
                "recovered": self.recovered, "events": list(self.events),
            }


def route_lane_index(key: str, lane_count: int) -> int:
    """Deterministic source -> lane assignment.

    Routing is per source (never per request): the endpoint identity is part of
    the canonical request manifest, so a source that moved between lanes would
    produce a different request hash and lose its exact-cache/resume entries.
    """
    if lane_count < 1:
        raise ValueError("lane_count must be positive")
    if lane_count == 1:
        return 0
    digest = hashlib.sha256(f"terra-lane|{key}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % lane_count


@dataclass(frozen=True, slots=True)
class TerraLane:
    """One provider key: its own client and its own concurrency limiter."""

    index: int
    identity: str
    endpoint: Any
    client: Any
    limiter: TerraLimiter


class TerraRouter:
    """Fixed per-source routing across independent Terra lanes."""

    def __init__(self, lanes: Sequence[TerraLane]) -> None:
        rows = tuple(lanes)
        if not rows:
            raise ValueError("Terra router requires at least one lane")
        if len({lane.identity for lane in rows}) != len(rows):
            raise ValueError("Terra lane identities must be distinct")
        self.lanes = rows

    def __len__(self) -> int:
        return len(self.lanes)

    def lane_index_for(self, key: str) -> int:
        return route_lane_index(str(key), len(self.lanes))

    def lane_for(self, key: str) -> TerraLane:
        return self.lanes[self.lane_index_for(key)]

    @property
    def total_concurrency(self) -> int:
        return sum(lane.limiter.target for lane in self.lanes)

    def metrics(self) -> dict[str, Any]:
        return {
            "lane_count": len(self.lanes),
            "total_concurrency": self.total_concurrency,
            "lanes": [
                {"index": lane.index, "identity": lane.identity,
                 **{key: value for key, value in lane.limiter.metrics().items()
                    if key != "events"}}
                for lane in self.lanes
            ],
        }


__all__ = ["TerraLane", "TerraLimiter", "TerraRouter", "route_lane_index"]
