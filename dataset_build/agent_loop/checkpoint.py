"""LangGraph checkpointer lifecycle helpers."""
from __future__ import annotations

import hashlib
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from langgraph.checkpoint.base import BaseCheckpointSaver

from .config import AgentLoopConfig


_AGENT_RELATIONS = frozenset({
    "agent_source_run", "agent_branch", "api_request", "api_attempt",
    "api_cache_event", "api_request_context", "render_record",
    "render_calibration", "validation_record", "provider_preflight",
    "scheduler_event", "artifact_record", "checkpoints", "checkpoint_blobs",
    "checkpoint_writes", "checkpoint_migrations",
})


def validate_postgres_storage(config: AgentLoopConfig) -> dict[str, Any]:
    if config.checkpoint.backend != "postgres":
        return {}
    import psycopg

    required = str(config.checkpoint.required_tablespace or "")
    with psycopg.connect(config.checkpoint_dsn(), autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT current_schema(), current_setting('default_tablespace'), "
                "pg_tablespace_location(oid) FROM pg_tablespace WHERE spcname=%s",
                (required,),
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError(f"required PostgreSQL tablespace is missing: {required}")
            schema = str(row[0] or "")
            active_tablespace = str(row[1] or "")
            location = str(row[2] or "")
            if not schema or schema == "public":
                raise RuntimeError("agent loop PostgreSQL DSN must set a non-public search_path")
            if active_tablespace != required:
                raise RuntimeError(
                    f"agent loop PostgreSQL DSN must set default_tablespace={required}"
                )
            if not location:
                raise RuntimeError("agent loop PostgreSQL tablespace must have a dedicated location")
            cur.execute(
                "SELECT c.relname, COALESCE(t.spcname, dt.spcname) "
                "FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
                "LEFT JOIN pg_tablespace t ON t.oid=c.reltablespace "
                "JOIN pg_database d ON d.datname=current_database() "
                "JOIN pg_tablespace dt ON dt.oid=d.dattablespace "
                "WHERE n.nspname=current_schema() AND c.relkind IN ('r','i','S')"
            )
            misplaced = sorted(
                name for name, tablespace in cur.fetchall()
                if str(name) in _AGENT_RELATIONS and str(tablespace) != required
            )
    if misplaced:
        raise RuntimeError(
            f"agent loop PostgreSQL relations are outside {required}: {', '.join(misplaced)}"
        )
    try:
        free_bytes = shutil.disk_usage(location).free
    except OSError as exc:
        raise RuntimeError(
            f"cannot inspect PostgreSQL tablespace capacity: {location}"
        ) from exc
    minimum = config.checkpoint.minimum_free_bytes
    if free_bytes < minimum:
        raise RuntimeError(
            f"PostgreSQL tablespace free bytes {free_bytes} below required {minimum}"
        )
    return {
        "schema": schema, "tablespace": required, "location": location,
        "free_bytes": free_bytes, "minimum_free_bytes": minimum,
    }


class ShardedCheckpointSaver(BaseCheckpointSaver):
    """Route threads across independent saver locks backed by one shared database."""

    def __init__(self, savers: Sequence[Any]) -> None:
        if not savers:
            raise ValueError("at least one checkpoint saver is required")
        self._savers = tuple(savers)
        super().__init__(serde=self._savers[0].serde)

    @property
    def config_specs(self) -> list[Any]:
        return list(self._savers[0].config_specs)

    def _for_thread(self, thread_id: str) -> Any:
        digest = hashlib.sha256(thread_id.encode("utf-8")).digest()
        return self._savers[int.from_bytes(digest[:8], "big") % len(self._savers)]

    def _for_config(self, config: dict[str, Any]) -> Any:
        configurable = config.get("configurable") or {}
        thread_id = str(configurable.get("thread_id") or "")
        if not thread_id:
            raise ValueError("checkpoint config lacks thread_id")
        return self._for_thread(thread_id)

    def get_tuple(self, config):
        return self._for_config(config).get_tuple(config)

    def list(self, config, *, filter=None, before=None, limit=None):
        saver = self._for_config(config) if config is not None else self._savers[0]
        yield from saver.list(config, filter=filter, before=before, limit=limit)

    def put(self, config, checkpoint, metadata, new_versions):
        return self._for_config(config).put(config, checkpoint, metadata, new_versions)

    def put_writes(self, config, writes, task_id, task_path=""):
        return self._for_config(config).put_writes(config, writes, task_id, task_path)

    def delete_thread(self, thread_id):
        return self._for_thread(str(thread_id)).delete_thread(thread_id)

    def delete_for_runs(self, run_ids):
        return self._savers[0].delete_for_runs(run_ids)

    def copy_thread(self, source_thread_id, target_thread_id):
        return self._for_thread(str(source_thread_id)).copy_thread(
            source_thread_id, target_thread_id
        )

    def prune(self, thread_ids, *, strategy="keep_latest"):
        return self._savers[0].prune(thread_ids, strategy=strategy)

    def get_delta_channel_history(self, *, config, channels):
        return self._for_config(config).get_delta_channel_history(
            config=config, channels=channels
        )

    def get_next_version(self, current, channel):
        return self._savers[0].get_next_version(current, channel)


@contextmanager
def open_checkpointer(config: AgentLoopConfig) -> Iterator[Any]:
    if config.checkpoint.backend == "postgres":
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        validate_postgres_storage(config)
        with ConnectionPool(
            conninfo=config.checkpoint_dsn(), min_size=4, max_size=32,
            kwargs={"autocommit": True, "prepare_threshold": 0,
                    "row_factory": dict_row},
        ) as pool:
            savers = tuple(PostgresSaver(pool) for _ in range(32))
            savers[0].setup()
            validate_postgres_storage(config)
            yield ShardedCheckpointSaver(savers)
    else:
        from langgraph.checkpoint.sqlite import SqliteSaver

        path = Path(config.checkpoint_dsn())
        path.parent.mkdir(parents=True, exist_ok=True)
        with SqliteSaver.from_conn_string(str(path)) as saver:
            yield saver


__all__ = ["ShardedCheckpointSaver", "open_checkpointer", "validate_postgres_storage"]
