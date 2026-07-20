"""vLLM replica discovery for the broker.

Polls a fixed list of candidate host ports for an OpenAI-compatible
``GET /v1/models`` endpoint. A replica counts as *live* iff it is reachable
**and** serves the configured model name (canonical default:
``qwen3_5-35b-a3b``).

Heterogeneous replicas (a port that serves a *different* model name) are
isolated — kept out of the live set and logged loudly — because least-outstanding
routing across mismatched models would silently corrupt the build. The registry
tolerates discovering only a subset of candidates (today only ``reason_g1:8002``
is up); the live set simply shrinks and the global budget scales with it.

Pure Python + httpx; loads no torch and never touches a GPU.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger("vgate.discovery")


@dataclass
class Replica:
    """Live state of one candidate vLLM replica (one host port)."""

    port: int
    alive: bool = False
    served_name: Optional[str] = None       # model id last reported by /v1/models
    isolated: bool = False                   # reachable but serves a foreign model
    last_ok: float = 0.0                     # monotonic time of last successful probe
    last_error: Optional[str] = None
    consecutive_failures: int = 0

    @property
    def base_url(self) -> str:
        # Replicas are local host-port maps of the in-container vLLM :8000.
        return f"http://127.0.0.1:{self.port}"


class ReplicaRegistry:
    """Discovers and tracks the set of routable vLLM replicas.

    The registry only *describes* replicas (alive / served-name / errors). The
    broker owns inflight accounting and admission, reading ``live_ports()`` each
    time it routes. Mutation happens only inside ``poll_once`` (driven from the
    broker's single asyncio loop), so reads from request handlers are consistent
    between awaits without an extra lock.
    """

    def __init__(
        self,
        ports: List[int],
        expected_served_name: Optional[str] = None,
        probe_timeout: float = 4.0,
    ) -> None:
        if not ports:
            raise ValueError("ReplicaRegistry needs at least one candidate port")
        self.expected_served_name = expected_served_name
        self.probe_timeout = float(probe_timeout)
        self.replicas: Dict[int, Replica] = {p: Replica(port=p) for p in ports}
        self._announced_ready = False

    # -- discovery ---------------------------------------------------------
    async def poll_once(self, client: httpx.AsyncClient) -> None:
        """Probe every candidate port once and refresh its ``Replica`` state."""
        for port, rep in self.replicas.items():
            await self._probe(client, rep)
        self._assert_consistency()

    async def _probe(self, client: httpx.AsyncClient, rep: Replica) -> None:
        url = f"{rep.base_url}/v1/models"
        try:
            resp = await client.get(url, timeout=self.probe_timeout)
            resp.raise_for_status()
            served = _first_model_id(resp.json())
        except Exception as exc:  # unreachable / not-ready / bad payload
            was_alive = rep.alive
            rep.alive = False
            rep.consecutive_failures += 1
            rep.last_error = f"{type(exc).__name__}: {exc}"
            if was_alive:
                logger.warning("replica :%d went DOWN (%s)", rep.port, rep.last_error)
            return

        rep.served_name = served
        rep.last_error = None
        rep.last_ok = time.monotonic()
        rep.consecutive_failures = 0

        expected = self.expected_served_name
        if expected is not None and served != expected:
            if not rep.isolated:
                logger.error(
                    "replica :%d serves '%s' but expected '%s' -> ISOLATED "
                    "(refusing to route across heterogeneous models)",
                    rep.port, served, expected,
                )
            rep.isolated = True
            rep.alive = False
            return

        rep.isolated = False
        if not rep.alive:
            logger.info("replica :%d is LIVE (served='%s')", rep.port, served)
        rep.alive = True

    def _assert_consistency(self) -> None:
        """Warn if reachable replicas disagree on served name.

        ``expected_served_name`` is authoritative when set; this is a belt-and-
        suspenders check that surfaces a misconfigured launcher even when the
        expected name itself is wrong.
        """
        names = {r.served_name for r in self.replicas.values() if r.served_name}
        if len(names) > 1:
            logger.warning("heterogeneous served names across replicas: %s", sorted(names))
        if self.expected_served_name is None and len(names) == 1:
            # Auto-adopt the single observed name as canonical going forward.
            self.expected_served_name = next(iter(names))
            logger.info("adopted served name '%s' as canonical", self.expected_served_name)

    # -- queries (read by the broker on the hot path) ----------------------
    def live_ports(self) -> List[int]:
        return [p for p, r in self.replicas.items() if r.alive]

    def snapshot(self) -> List[dict]:
        return [
            {
                "port": r.port,
                "alive": r.alive,
                "isolated": r.isolated,
                "served_name": r.served_name,
                "last_error": r.last_error,
                "consecutive_failures": r.consecutive_failures,
                "seconds_since_ok": (round(time.monotonic() - r.last_ok, 1) if r.last_ok else None),
            }
            for r in self.replicas.values()
        ]


def _first_model_id(payload: object) -> Optional[str]:
    """Extract the served model id from an OpenAI ``/v1/models`` response."""
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list) and data and isinstance(data[0], dict):
            mid = data[0].get("id")
            return str(mid) if mid is not None else None
    return None
