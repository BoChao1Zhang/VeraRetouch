"""Streaming Responses proxy over discovered local vLLM replicas.

vGate supplies replica discovery, least-outstanding routing, bounded admission,
and weighted queue classes. It exposes only ``/v1/responses`` for generation;
the canonical producer configures its URL in ``[annotation.local]``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .discovery import ReplicaRegistry

logger = logging.getLogger("vgate.app")

# --- priority classes -------------------------------------------------------
# Weighted-fair shares; higher weight == larger slice of admission slots.
CLASS_WEIGHTS: Dict[str, float] = {
    "build-annotate": 4.0,
    "qa-judge": 2.0,
    "tag": 1.0,
}
CLASSES = list(CLASS_WEIGHTS)

# Hop-by-hop / connection-management headers we must not forward verbatim.
_DROP_REQ_HEADERS = {"host", "content-length", "connection", "keep-alive",
                     "transfer-encoding", "upgrade", "x-vgate-class"}
_DROP_RESP_HEADERS = {"content-length", "transfer-encoding", "connection",
                      "keep-alive"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class BrokerConfig:
    ports: List[int]
    served_name: Optional[str]
    per_replica_cap: int = _env_int("VGATE_REPLICA_CAP", 32)
    poll_interval: float = _env_float("VGATE_POLL_INTERVAL", 10.0)
    probe_timeout: float = _env_float("VGATE_PROBE_TIMEOUT", 4.0)
    max_queue_wait: float = _env_float("VGATE_MAX_QUEUE_WAIT", 300.0)
    starve_boost: float = _env_float("VGATE_STARVE_BOOST", 30.0)
    retry_after: int = _env_int("VGATE_RETRY_AFTER", 3)
    upstream_read_timeout: float = _env_float("VGATE_UPSTREAM_READ_TIMEOUT", 600.0)
    default_class: str = os.environ.get("VGATE_DEFAULT_CLASS", "build-annotate")


class Backpressure(Exception):
    """Queue wait exceeded; surfaces as 429 + Retry-After."""


class NoReplica(Exception):
    """No live replica to route to; surfaces as 503 + Retry-After."""


@dataclass
class _Waiter:
    cls: str
    enqueued_at: float


class Broker:
    """Admission + least-outstanding routing over the discovered replica set."""

    def __init__(self, cfg: BrokerConfig, registry: ReplicaRegistry) -> None:
        self.cfg = cfg
        self.registry = registry
        self.lock = asyncio.Lock()
        self.cond = asyncio.Condition(self.lock)
        # Inflight is keyed by port (survives replicas coming/going); both the
        # global count and per-replica counts are guarded by ``self.lock``.
        self.inflight: Dict[int, int] = {}
        self.total_inflight: int = 0
        self.queues: Dict[str, "list[_Waiter]"] = {c: [] for c in CLASSES}
        self.vtime: Dict[str, float] = {c: 0.0 for c in CLASSES}
        self._rr: int = 0  # round-robin cursor for inflight ties
        self.client: Optional[httpx.AsyncClient] = None

    # -- admission ---------------------------------------------------------
    async def admit(self, cls: str) -> int:
        """Block until a global slot is free and this waiter wins its class.

        Returns the chosen replica port. Raises :class:`Backpressure` if the
        queue wait exceeds ``max_queue_wait`` with replicas present, or
        :class:`NoReplica` if no live replica appears within that window.
        """
        deadline = time.monotonic() + self.cfg.max_queue_wait
        waiter = _Waiter(cls=cls, enqueued_at=time.monotonic())
        async with self.cond:
            self._enqueue(waiter)
            try:
                while True:
                    port = self._try_admit(waiter)
                    if port is not None:
                        return port
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise (NoReplica() if not self.registry.live_ports()
                               else Backpressure())
                    try:
                        await asyncio.wait_for(self.cond.wait(), timeout=remaining)
                    except asyncio.TimeoutError:
                        raise (NoReplica() if not self.registry.live_ports()
                               else Backpressure())
            finally:
                self._discard(waiter)

    def _enqueue(self, waiter: _Waiter) -> None:
        q = self.queues[waiter.cls]
        if not q:
            # On (re)activating an idle class, pull its virtual clock up to the
            # busiest active class so it can't hoard slots after a long idle.
            active = [self.vtime[c] for c in CLASSES if self.queues[c]]
            if active:
                self.vtime[waiter.cls] = max(self.vtime[waiter.cls], min(active))
        q.append(waiter)

    def _discard(self, waiter: _Waiter) -> None:
        q = self.queues[waiter.cls]
        try:
            q.remove(waiter)
        except ValueError:
            pass

    def _try_admit(self, waiter: _Waiter) -> Optional[int]:
        """Return a port if ``waiter`` may proceed now, else ``None``."""
        live = self.registry.live_ports()
        if not live:
            return None
        budget = self.cfg.per_replica_cap * len(live)
        if self.total_inflight >= budget:
            return None

        winner = self._select_winner()
        if winner is not waiter:
            return None

        port = self._pick_replica(live)
        self.queues[waiter.cls].remove(waiter)
        self.inflight[port] = self.inflight.get(port, 0) + 1
        self.total_inflight += 1
        self.vtime[waiter.cls] += 1.0 / CLASS_WEIGHTS[waiter.cls]
        return port

    def _select_winner(self) -> Optional[_Waiter]:
        """Pick the next waiter to admit across all class queues.

        Anti-starvation first: any waiter queued longer than ``starve_boost``
        wins (oldest such waiter). Otherwise weighted-fair: the front waiter of
        the class with the smallest virtual time (service / weight).
        """
        now = time.monotonic()
        oldest: Optional[_Waiter] = None
        for c in CLASSES:
            if self.queues[c]:
                head = self.queues[c][0]
                if oldest is None or head.enqueued_at < oldest.enqueued_at:
                    oldest = head
        if oldest is not None and (now - oldest.enqueued_at) >= self.cfg.starve_boost:
            return oldest

        best_cls: Optional[str] = None
        for c in CLASSES:
            if not self.queues[c]:
                continue
            if best_cls is None or self.vtime[c] < self.vtime[best_cls]:
                best_cls = c
        return self.queues[best_cls][0] if best_cls is not None else None

    def _pick_replica(self, live: List[int]) -> int:
        """Least-outstanding among live replicas; round-robin on ties."""
        min_inflight = min(self.inflight.get(p, 0) for p in live)
        candidates = [p for p in live if self.inflight.get(p, 0) == min_inflight]
        if len(candidates) == 1:
            return candidates[0]
        choice = candidates[self._rr % len(candidates)]
        self._rr += 1
        return choice

    async def release(self, port: int) -> None:
        async with self.cond:
            if self.inflight.get(port, 0) > 0:
                self.inflight[port] -= 1
                self.total_inflight -= 1
            self.cond.notify_all()

    async def notify_budget_changed(self) -> None:
        async with self.cond:
            self.cond.notify_all()

    # -- observability -----------------------------------------------------
    def status(self) -> dict:
        live = self.registry.live_ports()
        return {
            "replicas": self.registry.snapshot(),
            "live_ports": live,
            "per_replica_cap": self.cfg.per_replica_cap,
            "budget": self.cfg.per_replica_cap * len(live),
            "total_inflight": self.total_inflight,
            "inflight_by_port": dict(self.inflight),
            "queue_depth": {c: len(self.queues[c]) for c in CLASSES},
            "vtime": {c: round(self.vtime[c], 3) for c in CLASSES},
        }


# --- request handling -------------------------------------------------------
def _class_of(request: Request, default_class: str) -> str:
    cls = request.headers.get("x-vgate-class", "").strip()
    return cls if cls in CLASS_WEIGHTS else default_class


def _forward_request_headers(request: Request) -> Dict[str, str]:
    return {k: v for k, v in request.headers.items()
            if k.lower() not in _DROP_REQ_HEADERS}


def _forward_response_headers(resp: httpx.Response) -> Dict[str, str]:
    return {k: v for k, v in resp.headers.items()
            if k.lower() not in _DROP_RESP_HEADERS}


async def _proxy_generative(request: Request, subpath: str) -> StreamingResponse:
    """Admit, route, and stream-proxy a generative ``/v1`` request."""
    broker: Broker = request.app.state.broker
    cls = _class_of(request, broker.cfg.default_class)
    body = await request.body()

    try:
        port = await broker.admit(cls)
    except Backpressure:
        return JSONResponse(
            {"error": {"message": "broker over capacity, retry shortly",
                       "type": "overloaded", "code": "vgate_backpressure"}},
            status_code=429, headers={"Retry-After": str(broker.cfg.retry_after)},
        )
    except NoReplica:
        return JSONResponse(
            {"error": {"message": "no live vLLM replica", "type": "unavailable",
                       "code": "vgate_no_replica"}},
            status_code=503, headers={"Retry-After": str(broker.cfg.retry_after)},
        )

    url = f"http://127.0.0.1:{port}/v1/{subpath}"
    headers = _forward_request_headers(request)
    client = broker.client
    assert client is not None

    try:
        upstream_req = client.build_request("POST", url, content=body, headers=headers)
        resp = await client.send(upstream_req, stream=True)
    except Exception as exc:
        await broker.release(port)
        logger.warning("upstream :%d error for /v1/%s: %s", port, subpath, exc)
        return JSONResponse(
            {"error": {"message": f"upstream error: {exc}", "type": "upstream",
                       "code": "vgate_upstream"}},
            status_code=502, headers={"Retry-After": str(broker.cfg.retry_after)},
        )

    released = {"done": False}

    async def _release_once() -> None:
        if not released["done"]:
            released["done"] = True
            await broker.release(port)

    async def _stream():
        try:
            async for chunk in resp.aiter_raw():
                yield chunk
        finally:
            await resp.aclose()
            await _release_once()

    out_headers = _forward_response_headers(resp)
    out_headers["X-Vgate-Replica"] = str(port)
    return StreamingResponse(
        _stream(),
        status_code=resp.status_code,
        headers=out_headers,
        media_type=resp.headers.get("content-type"),
    )


def build_app(cfg: BrokerConfig) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # One client for both probing (short per-call timeout) and proxying
        # (long read timeout for generation). Pool sized to the max budget.
        max_conn = cfg.per_replica_cap * max(1, len(cfg.ports)) + 32
        timeout = httpx.Timeout(
            connect=10.0, read=cfg.upstream_read_timeout, write=120.0, pool=cfg.max_queue_wait + 60.0,
        )
        client = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(max_connections=max_conn, max_keepalive_connections=max_conn),
        )
        registry = ReplicaRegistry(cfg.ports, cfg.served_name, cfg.probe_timeout)
        broker = Broker(cfg, registry)
        broker.client = client
        app.state.broker = broker

        # Prime discovery once so /v1/models works the instant we serve.
        try:
            await registry.poll_once(client)
        except Exception as exc:  # pragma: no cover - first poll is best-effort
            logger.warning("initial replica poll failed: %s", exc)

        async def _poll_loop():
            while True:
                await asyncio.sleep(cfg.poll_interval)
                try:
                    await registry.poll_once(client)
                except Exception as exc:
                    logger.warning("replica poll error: %s", exc)
                await broker.notify_budget_changed()

        poller = asyncio.create_task(_poll_loop())
        logger.info("vgate up: ports=%s served='%s' cap=%d -> budget=%d",
                    cfg.ports, cfg.served_name, cfg.per_replica_cap,
                    cfg.per_replica_cap * len(registry.live_ports()))
        try:
            yield
        finally:
            poller.cancel()
            await client.aclose()

    app = FastAPI(title="vGate vLLM broker", version="0.1.0", lifespan=lifespan)

    @app.post("/v1/responses")
    async def responses(request: Request):
        return await _proxy_generative(request, "responses")

    @app.get("/v1/models")
    async def models(request: Request):
        broker: Broker = request.app.state.broker
        live = broker.registry.live_ports()
        if not live:
            return JSONResponse(
                {"error": {"message": "no live vLLM replica", "type": "unavailable"}},
                status_code=503, headers={"Retry-After": str(broker.cfg.retry_after)},
            )
        try:
            r = await broker.client.get(f"http://127.0.0.1:{live[0]}/v1/models",
                                        timeout=broker.cfg.probe_timeout)
            return JSONResponse(r.json(), status_code=r.status_code)
        except Exception as exc:
            return JSONResponse(
                {"error": {"message": f"upstream error: {exc}", "type": "upstream"}},
                status_code=502,
            )

    @app.get("/")
    @app.get("/status")
    async def status(request: Request):
        return JSONResponse(request.app.state.broker.status())

    @app.get("/healthz")
    async def healthz(request: Request):
        broker: Broker = request.app.state.broker
        ok = bool(broker.registry.live_ports())
        return JSONResponse({"status": "ok" if ok else "degraded",
                             "live_replicas": broker.registry.live_ports()},
                            status_code=200 if ok else 503)

    return app


def _parse_ports(spec: str) -> List[int]:
    return [int(p) for p in spec.replace(",", " ").split() if p.strip()]


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(prog="dataset_build.core.broker.app",
                                description="vGate streaming Responses broker")
    p.add_argument("--host", default=os.environ.get("VGATE_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=_env_int("VGATE_PORT", 8003))
    p.add_argument("--replica-ports", default=os.environ.get("VGATE_REPLICA_PORTS", "8001,8002"),
                   help="comma/space-separated candidate vLLM host ports")
    p.add_argument("--served-name", default=os.environ.get("VGATE_SERVED_NAME", "qwen3_5-35b-a3b"),
                   help="expected served model id; replicas serving anything else are isolated")
    p.add_argument("--cap", type=int, default=_env_int("VGATE_REPLICA_CAP", 32),
                   help="per-replica admission cap; global budget = cap * live replicas")
    p.add_argument("--log-level", default=os.environ.get("VGATE_LOG_LEVEL", "info"))
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    served = args.served_name.strip() or None
    cfg = BrokerConfig(
        ports=_parse_ports(args.replica_ports),
        served_name=served,
        per_replica_cap=args.cap,
    )
    app = build_app(cfg)
    logger.info("starting vgate on %s:%d (replicas=%s, served='%s', cap=%d)",
                args.host, args.port, cfg.ports, served, cfg.per_replica_cap)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
