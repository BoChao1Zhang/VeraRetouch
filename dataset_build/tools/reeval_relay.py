"""Shared relay transport for the WP5 annotation/IAA re-evaluation tools.

Three things here are not in ``review_annot_quality.py`` and are the reason this
module exists:

1. **Streaming.**  The relay sits behind a proxy with a 120 s read timeout, so a
   non-streamed ``xhigh`` request returns HTTP 524 rather than an answer.  The
   production annotator already streams; the review path did not.
2. **Model pinning.**  ``provider-b-lane-2`` silently answers a
   ``gpt-5.6-sol`` request with ``gpt-5.5`` for a fraction of calls (measured
   ~1/3 on 2026-07-28) and that substituted model also violates the strict JSON
   schema.  Every call here checks ``response.model`` and retries elsewhere on a
   mismatch, so a run has one known judge instead of an uncontrolled mixture.
3. **Resume.**  Results append to JSONL keyed by a caller-supplied record key;
   a rerun skips keys already on disk, so a partial batch is never re-billed and
   never double-counted.

Concurrency is bounded per lane (default 2), which is the shape the relay was
observed to tolerate.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import queue
import random
import threading
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from PIL import Image

CONFIG = Path("/home/bc/VeraRetouch/databuild.eval100.toml")


class ModelSubstituted(RuntimeError):
    """The relay answered with a model other than the one requested."""


@dataclass
class Lane:
    lane_id: str
    client: Any
    calls: int = 0
    substitutions: int = 0
    errors: int = 0


def load_lanes(config: Path = CONFIG, timeout: float = 600.0) -> list[Lane]:
    """Build one client per configured endpoint.  The config is only ever read."""
    from openai import OpenAI

    cfg = tomllib.load(config.open("rb"))
    lanes = []
    for index, endpoint in enumerate(cfg["annotation"]["external_endpoints"]):
        lanes.append(Lane(
            lane_id=str(endpoint.get("id") or f"lane-{index}"),
            client=OpenAI(
                base_url=endpoint["base_url"], api_key=endpoint["api_key"],
                max_retries=0, timeout=timeout,
            ),
        ))
    return lanes


_IMAGE_CACHE: dict[tuple[str, int, int], str] = {}
_IMAGE_LOCK = threading.Lock()


def data_url(path: Path | str, long_edge: int = 1024, quality: int = 92) -> str:
    """JPEG data URI at review resolution.

    The first-round review shrank to 512 px / q85, well under the 1024 px short
    edge / q95 the pipeline actually renders, which made small regional edits
    unjudgeable.  1024 px long edge / q92 keeps the judge's evidence close to the
    artefact under test.
    """
    key = (str(path), long_edge, quality)
    with _IMAGE_LOCK:
        cached = _IMAGE_CACHE.get(key)
    if cached is not None:
        return cached
    with Image.open(path) as image:
        image.load()
        rgb = image.convert("RGB")
    scale = long_edge / max(rgb.width, rgb.height)
    if scale < 1:
        rgb = rgb.resize((round(rgb.width * scale), round(rgb.height * scale)), Image.LANCZOS)
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=quality)
    url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
    with _IMAGE_LOCK:
        _IMAGE_CACHE[key] = url
    return url


def data_url_bytes(raw: bytes, long_edge: int = 1024, quality: int = 92) -> str:
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        rgb = image.convert("RGB")
    scale = long_edge / max(rgb.width, rgb.height)
    if scale < 1:
        rgb = rgb.resize((round(rgb.width * scale), round(rgb.height * scale)), Image.LANCZOS)
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def call_once(
    lane: Lane, model: str, effort: str, content: list[dict], schema: Mapping[str, Any],
    schema_name: str = "review", max_output_tokens: int = 6000,
) -> tuple[dict, dict]:
    """One streamed, schema-constrained, model-pinned call."""
    stream = lane.client.responses.create(
        model=model,
        input=[{"role": "user", "content": content}],
        reasoning={"effort": effort},
        max_output_tokens=max_output_tokens,
        stream=True,
        text={"format": {"type": "json_schema", "name": schema_name,
                         "strict": True, "schema": dict(schema)}},
    )
    final = None
    for event in stream:
        if getattr(event, "type", None) == "response.completed":
            final = event.response
    if final is None:
        raise RuntimeError("stream ended before response.completed")
    if getattr(final, "status", None) != "completed":
        raise RuntimeError(f"status={getattr(final, 'status', None)}")
    returned = str(getattr(final, "model", "") or "")
    if not returned.startswith(model):
        lane.substitutions += 1
        raise ModelSubstituted(f"requested {model}, relay returned {returned}")
    usage = getattr(final, "usage", None)
    meta = {
        "lane": lane.lane_id,
        "returned_model": returned,
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
    }
    return json.loads(final.output_text), meta


def prompt_digest(
    content: Sequence[Mapping[str, Any]], schema: Mapping[str, Any]
) -> dict[str, str]:
    """Fingerprint of the rubric text and the schema an answer was produced under.

    Recorded on every result row so a later reading of the log can tell whether
    two answers were judged by the same instructions, without keeping a copy of
    the prompt.  Only the text parts are hashed: the images are already pinned by
    the record key and hashing megabytes of base64 per call buys nothing.
    """
    rubric = "\n".join(
        str(part.get("text") or "")
        for part in content if part.get("type") == "input_text"
    )
    schema_text = json.dumps(schema, sort_keys=True, ensure_ascii=False)
    return {
        "rubric_sha256": hashlib.sha256(rubric.encode("utf-8")).hexdigest(),
        "schema_sha256": hashlib.sha256(schema_text.encode("utf-8")).hexdigest(),
    }


@dataclass
class Task:
    key: str
    build: Callable[[], tuple[list[dict], Mapping[str, Any]]]
    record: dict = field(default_factory=dict)


class JsonlStore:
    """Append-only result log that makes a rerun idempotent."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.done: set[str] = set()
        if self.path.is_file():
            for line in self.path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("ok"):
                    self.done.add(row["key"])

    def append(self, row: Mapping[str, Any]) -> None:
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()


def run_tasks(
    tasks: Sequence[Task], lanes: Sequence[Lane], store: JsonlStore, *,
    model: str, effort: str, per_lane: int = 2, attempts: int = 3,
    schema_name: str = "review", max_output_tokens: int = 6000,
    validate: Callable[[dict], dict] | None = None,
    progress: Path | None = None,
) -> dict[str, int]:
    """Drive tasks over the lanes, at most ``per_lane`` in flight per lane.

    A task that exhausts its attempts is logged with ``ok: false`` and does not
    stop the batch: a partial answer set with a known hole is more useful than a
    run that dies at 90%.
    """
    pending = [task for task in tasks if task.key not in store.done]
    work: queue.Queue[Task | None] = queue.Queue()
    for task in pending:
        work.put(task)
    stats = {"total": len(tasks), "skipped": len(tasks) - len(pending),
             "ok": 0, "failed": 0, "substituted": 0}
    stats_lock = threading.Lock()
    started = time.time()

    def write_progress() -> None:
        if progress is None:
            return
        payload = dict(stats)
        payload["elapsed_s"] = round(time.time() - started, 1)
        payload["remaining"] = len(pending) - stats["ok"] - stats["failed"]
        progress.write_text(json.dumps(payload, indent=2))

    def worker(lane: Lane) -> None:
        while True:
            try:
                task = work.get_nowait()
            except queue.Empty:
                return
            content, schema = task.build()
            digest = prompt_digest(content, schema)
            last_error = None
            for attempt in range(attempts):
                # rotate lanes across attempts so a substituting lane cannot
                # capture a task
                use = lane if attempt == 0 else lanes[(lanes.index(lane) + attempt) % len(lanes)]
                try:
                    result, meta = call_once(
                        use, model, effort, content, schema,
                        schema_name=schema_name, max_output_tokens=max_output_tokens,
                    )
                    if validate is not None:
                        result = validate(result)
                    use.calls += 1
                    store.append({"key": task.key, "ok": True, "attempt": attempt,
                                  **task.record, "result": result,
                                  "meta": {**meta, **digest}})
                    with stats_lock:
                        stats["ok"] += 1
                        write_progress()
                    break
                except ModelSubstituted as exc:
                    last_error = f"ModelSubstituted: {exc}"
                    with stats_lock:
                        stats["substituted"] += 1
                    time.sleep(1.0 + random.random())
                except Exception as exc:  # noqa: BLE001 - transport/validation retry
                    use.errors += 1
                    last_error = f"{type(exc).__name__}: {str(exc)[:300]}"
                    time.sleep(2.0 * (attempt + 1) + random.random())
            else:
                store.append({"key": task.key, "ok": False, **task.record,
                              "error": last_error, "meta": dict(digest)})
                with stats_lock:
                    stats["failed"] += 1
                    write_progress()

    threads = [threading.Thread(target=worker, args=(lane,), daemon=True)
               for lane in lanes for _ in range(per_lane)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    write_progress()
    stats["lanes"] = {lane.lane_id: {"calls": lane.calls, "substitutions": lane.substitutions,
                                     "errors": lane.errors} for lane in lanes}
    return stats


def read_results(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    # last write wins for a key
    latest: dict[str, dict] = {}
    for row in out:
        if row.get("ok") or row["key"] not in latest:
            latest[row["key"]] = row
    return list(latest.values())
