"""Business audit persistence, separate from LangGraph checkpoints."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _loads(value: Any) -> Any:
    if value is None or isinstance(value, (dict, list)):
        return value
    return json.loads(str(value))


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _proposal_audit_values(row: dict[str, Any]) -> tuple[Any, ...]:
    """Audit-only retrieval columns: scorer agreement and local/global direction cosine."""
    cosine = row.get("direction_cosine")
    return (
        str(row["branch_id"]), str(row["campaign_id"]), str(row["source_sha256"]),
        str(row["level"]), str(row["preset_id"]),
        _optional_int(row.get("scorer_top1")), _optional_int(row.get("scorer_top3")),
        _optional_int(row.get("scorer_top1_raw")),
        None if cosine is None else float(cosine), time.time(),
    )


class AuditStore(ABC):
    @abstractmethod
    def setup(self) -> None: ...

    @abstractmethod
    def acquire_request(
        self,
        request_hash: str,
        canonical_request: dict[str, Any],
        owner_id: str,
        lease_seconds: int,
    ) -> dict[str, Any]: ...

    @abstractmethod
    def record_attempt(
        self,
        request_hash: str,
        owner_id: str,
        *,
        response: dict[str, Any] | None,
        usage: dict[str, Any] | None,
        validation: dict[str, Any],
        valid: bool,
        error_type: str | None = None,
    ) -> int: ...

    @abstractmethod
    def get_request(self, request_hash: str) -> dict[str, Any] | None: ...

    @abstractmethod
    def record_cache_event(self, request_hash: str, event: str) -> None: ...

    @abstractmethod
    def release_request(self, request_hash: str, owner_id: str) -> None: ...

    @abstractmethod
    def record_request_context(
        self, request_hash: str, campaign_id: str, source_sha256: str, stage: str,
        cache_hit: bool,
    ) -> None: ...

    @abstractmethod
    def record_artifact(self, ref: Any, retention: str) -> None: ...

    @abstractmethod
    def record_source_start(self, row: dict[str, Any]) -> None: ...

    @abstractmethod
    def record_source_finish(
        self, campaign_id: str, source_sha256: str, prompt_revision: str,
        status: str, counts: dict[str, Any], manifest: dict[str, Any] | None,
    ) -> None: ...

    @abstractmethod
    def record_branch(self, row: dict[str, Any]) -> None: ...

    @abstractmethod
    def record_proposal_audit(self, row: dict[str, Any]) -> None: ...

    @abstractmethod
    def record_render(self, row: dict[str, Any]) -> None: ...

    @abstractmethod
    def get_render(self, render_hash: str) -> dict[str, Any] | None: ...

    @abstractmethod
    def record_render_calibration(self, calibration_hash: str, render_hash: str) -> None: ...

    @abstractmethod
    def get_render_calibration(self, calibration_hash: str) -> str | None: ...

    @abstractmethod
    def record_validation(self, row: dict[str, Any]) -> None: ...

    @abstractmethod
    def record_preflight(self, key: str, row: dict[str, Any], passed: bool) -> None: ...

    @abstractmethod
    def get_preflight(self, key: str) -> dict[str, Any] | None: ...

    @abstractmethod
    def record_scheduler_event(self, row: dict[str, Any]) -> None: ...

    @abstractmethod
    def mark_artifact_purged(self, sha256: str) -> bool: ...

    @abstractmethod
    def source_status(self, campaign_id: str | None = None) -> list[dict[str, Any]]: ...

    @abstractmethod
    def export_tables(self) -> dict[str, list[dict[str, Any]]]: ...

    @abstractmethod
    def artifacts_with_retention(self, retention: str) -> list[dict[str, Any]]: ...

    @abstractmethod
    def database_storage(self) -> dict[str, int]: ...


_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_source_run (
 campaign_id TEXT NOT NULL, source_sha256 TEXT NOT NULL, prompt_revision TEXT NOT NULL,
 thread_id TEXT NOT NULL, source_id TEXT NOT NULL, status TEXT NOT NULL,
 started_at REAL NOT NULL, updated_at REAL NOT NULL, counts_json TEXT NOT NULL,
 manifest_json TEXT, PRIMARY KEY(campaign_id, source_sha256, prompt_revision)
);
CREATE TABLE IF NOT EXISTS agent_branch (
 branch_id TEXT NOT NULL, campaign_id TEXT NOT NULL, source_sha256 TEXT NOT NULL,
 parent_id TEXT, level TEXT NOT NULL, status TEXT NOT NULL, proposal_json TEXT NOT NULL,
 result_json TEXT NOT NULL, created_at REAL NOT NULL,
 PRIMARY KEY(campaign_id, branch_id)
);
CREATE TABLE IF NOT EXISTS proposal_audit (
 branch_id TEXT NOT NULL, campaign_id TEXT NOT NULL, source_sha256 TEXT NOT NULL,
 level TEXT NOT NULL, preset_id TEXT NOT NULL, scorer_top1 INTEGER,
 scorer_top3 INTEGER, scorer_top1_raw INTEGER, direction_cosine REAL,
 created_at REAL NOT NULL,
 PRIMARY KEY(campaign_id, branch_id)
);
CREATE TABLE IF NOT EXISTS api_request (
 request_hash TEXT PRIMARY KEY, canonical_request_json TEXT NOT NULL,
 status TEXT NOT NULL, owner_id TEXT, lease_expires_at REAL,
 resolved_attempt_id INTEGER, response_json TEXT, usage_json TEXT,
 created_at REAL NOT NULL, updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS api_attempt (
 request_hash TEXT NOT NULL, attempt_no INTEGER NOT NULL, started_at REAL NOT NULL,
 finished_at REAL NOT NULL, response_json TEXT, usage_json TEXT,
 validation_json TEXT NOT NULL, valid INTEGER NOT NULL, error_type TEXT,
 PRIMARY KEY(request_hash, attempt_no)
);
CREATE TABLE IF NOT EXISTS api_cache_event (
 event_id TEXT PRIMARY KEY, request_hash TEXT NOT NULL, event TEXT NOT NULL, created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS api_request_context (
 request_hash TEXT NOT NULL, campaign_id TEXT NOT NULL, source_sha256 TEXT NOT NULL,
 stage TEXT NOT NULL, cache_hit INTEGER NOT NULL, created_at REAL NOT NULL,
 PRIMARY KEY(request_hash,campaign_id,source_sha256,stage)
);
CREATE TABLE IF NOT EXISTS render_record (
 render_hash TEXT PRIMARY KEY, source_sha256 TEXT NOT NULL, branch_id TEXT NOT NULL,
 stage TEXT NOT NULL, input_json TEXT NOT NULL, parameters_json TEXT NOT NULL,
 metrics_json TEXT NOT NULL, artifact_json TEXT, status TEXT NOT NULL,
 created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS render_calibration (
 calibration_hash TEXT PRIMARY KEY, render_hash TEXT NOT NULL, created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS validation_record (
 validation_id TEXT PRIMARY KEY, source_sha256 TEXT NOT NULL, branch_id TEXT NOT NULL,
 request_hash TEXT, passed INTEGER NOT NULL, defects_json TEXT NOT NULL,
 raw_json TEXT NOT NULL, created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS provider_preflight (
 preflight_key TEXT PRIMARY KEY, passed INTEGER NOT NULL, result_json TEXT NOT NULL,
 created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS scheduler_event (
 event_id TEXT PRIMARY KEY, campaign_id TEXT NOT NULL, source_id TEXT NOT NULL,
 stage TEXT NOT NULL, prefix_key TEXT NOT NULL, event TEXT NOT NULL,
 in_flight INTEGER NOT NULL, effective_limit INTEGER NOT NULL, created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS artifact_record (
 sha256 TEXT PRIMARY KEY, uri TEXT NOT NULL, media_type TEXT NOT NULL, size INTEGER NOT NULL,
 retention TEXT NOT NULL, created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_source_status ON agent_source_run(campaign_id,status);
CREATE INDEX IF NOT EXISTS idx_branch_source ON agent_branch(campaign_id,source_sha256);
CREATE INDEX IF NOT EXISTS idx_attempt_request ON api_attempt(request_hash);
"""


# B6 blocker 7: `branch_id` alone used to be the primary key of both tables, so the
# same source re-run under a second campaign silently overwrote the first campaign's
# rows. The key is `(campaign_id, branch_id)`; `setup()` migrates legacy tables once
# and is a no-op when the key already matches.
_CAMPAIGN_PK_TABLES: tuple[str, ...] = ("agent_branch", "proposal_audit")
_CAMPAIGN_PK_COLUMNS: tuple[str, ...] = ("campaign_id", "branch_id")


def _sqlite_table_ddl(table: str) -> str:
    marker = f"CREATE TABLE IF NOT EXISTS {table} ("
    start = _SQLITE_SCHEMA.index(marker)
    return _SQLITE_SCHEMA[start:_SQLITE_SCHEMA.index(");", start) + 1]


class SQLiteAuditStore(AuditStore):
    def __init__(self, path: str | os.PathLike[str]) -> None:
        value = str(path)
        self._keeper: sqlite3.Connection | None = None
        if value == ":memory:":
            value = f"file:agent-loop-{uuid.uuid4().hex}?mode=memory&cache=shared"
            self._uri = True
            self._keeper = sqlite3.connect(value, uri=True, check_same_thread=False)
        else:
            target = Path(value).expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            value = str(target)
            self._uri = False
        self.path = value
        self._local = threading.local()

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(
                self.path, timeout=30.0, isolation_level=None,
                check_same_thread=False, uri=self._uri,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return conn

    def setup(self) -> None:
        conn = self._conn()
        conn.executescript(_SQLITE_SCHEMA)
        self._migrate_campaign_primary_key(conn)
        # Rebuilding a table drops the indexes that hung off it; the schema script
        # is `IF NOT EXISTS` throughout, so replaying it restores them.
        conn.executescript(_SQLITE_SCHEMA)

    def _migrate_campaign_primary_key(self, conn: sqlite3.Connection) -> None:
        for table in _CAMPAIGN_PK_TABLES:  # closed internal tuple, not user input
            info = conn.execute(f"PRAGMA table_info({table})").fetchall()
            if {row["name"] for row in info if int(row["pk"]) > 0} \
                    == set(_CAMPAIGN_PK_COLUMNS):
                continue
            columns = ",".join(str(row["name"]) for row in info)
            ddl = _sqlite_table_ddl(table).replace(
                f"IF NOT EXISTS {table} (", f"IF NOT EXISTS {table}_pkmig (", 1
            )
            conn.executescript(
                f"{ddl};\n"
                f"INSERT INTO {table}_pkmig({columns}) SELECT {columns} FROM {table};\n"
                f"DROP TABLE {table};\n"
                f"ALTER TABLE {table}_pkmig RENAME TO {table};"
            )

    def acquire_request(
        self, request_hash: str, canonical_request: dict[str, Any], owner_id: str,
        lease_seconds: int,
    ) -> dict[str, Any]:
        conn = self._conn()
        now = time.time()
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                "INSERT OR IGNORE INTO api_request VALUES(?,?,?,?,?,?,?,?,?,?)",
                (request_hash, _json(canonical_request), "pending", None, None,
                 None, None, None, now, now),
            )
            row = conn.execute(
                "SELECT * FROM api_request WHERE request_hash=?", (request_hash,)
            ).fetchone()
            assert row is not None
            if row["canonical_request_json"] != _json(canonical_request):
                raise RuntimeError("request_hash_collision")
            if row["status"] == "resolved":
                result = {
                    "action": "resolved", "response": _loads(row["response_json"]),
                    "usage": _loads(row["usage_json"]),
                    "attempt_no": row["resolved_attempt_id"],
                }
            elif row["owner_id"] == owner_id or float(row["lease_expires_at"] or 0) <= now:
                conn.execute(
                    "UPDATE api_request SET owner_id=?,lease_expires_at=?,updated_at=? "
                    "WHERE request_hash=?",
                    (owner_id, now + lease_seconds, now, request_hash),
                )
                result = {"action": "owner"}
            else:
                result = {
                    "action": "wait",
                    "lease_expires_at": float(row["lease_expires_at"] or now),
                }
            conn.execute("COMMIT")
            return result
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def record_attempt(
        self, request_hash: str, owner_id: str, *, response: dict[str, Any] | None,
        usage: dict[str, Any] | None, validation: dict[str, Any], valid: bool,
        error_type: str | None = None,
    ) -> int:
        conn = self._conn()
        now = time.time()
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                "SELECT owner_id,status FROM api_request WHERE request_hash=?",
                (request_hash,),
            ).fetchone()
            if row is None or (row["owner_id"] != owner_id and row["status"] != "resolved"):
                raise RuntimeError("request_lease_not_owned")
            attempt_no = int(conn.execute(
                "SELECT COALESCE(MAX(attempt_no),0)+1 FROM api_attempt WHERE request_hash=?",
                (request_hash,),
            ).fetchone()[0])
            conn.execute(
                "INSERT INTO api_attempt VALUES(?,?,?,?,?,?,?,?,?)",
                (request_hash, attempt_no, now, now, _json(response) if response else None,
                 _json(usage) if usage else None, _json(validation), int(valid), error_type),
            )
            if valid:
                conn.execute(
                    "UPDATE api_request SET status='resolved',resolved_attempt_id=?,"
                    "response_json=?,usage_json=?,owner_id=NULL,lease_expires_at=NULL,updated_at=? "
                    "WHERE request_hash=?",
                    (attempt_no, _json(response), _json(usage or {}), now, request_hash),
                )
            else:
                conn.execute(
                    "UPDATE api_request SET updated_at=? WHERE request_hash=?", (now, request_hash)
                )
            conn.execute("COMMIT")
            return attempt_no
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def get_request(self, request_hash: str) -> dict[str, Any] | None:
        row = self._conn().execute(
            "SELECT * FROM api_request WHERE request_hash=?", (request_hash,)
        ).fetchone()
        return _sqlite_row(row) if row else None

    def record_cache_event(self, request_hash: str, event: str) -> None:
        self._conn().execute(
            "INSERT INTO api_cache_event VALUES(?,?,?,?)",
            (uuid.uuid4().hex, request_hash, event, time.time()),
        )

    def release_request(self, request_hash: str, owner_id: str) -> None:
        self._conn().execute(
            "UPDATE api_request SET owner_id=NULL,lease_expires_at=0,updated_at=? "
            "WHERE request_hash=? AND owner_id=? AND status='pending'",
            (time.time(), request_hash, owner_id),
        )

    def record_request_context(
        self, request_hash: str, campaign_id: str, source_sha256: str, stage: str,
        cache_hit: bool,
    ) -> None:
        self._conn().execute(
            "INSERT INTO api_request_context VALUES(?,?,?,?,?,?) ON CONFLICT(" 
            "request_hash,campaign_id,source_sha256,stage) DO UPDATE SET "
            "cache_hit=MIN(api_request_context.cache_hit,excluded.cache_hit)",
            (request_hash, campaign_id, source_sha256, stage, int(cache_hit), time.time()),
        )

    def record_artifact(self, ref: Any, retention: str) -> None:
        self._conn().execute(
            "INSERT INTO artifact_record VALUES(?,?,?,?,?,?) ON CONFLICT(sha256) DO UPDATE "
            "SET retention=CASE WHEN artifact_record.retention='accepted' THEN 'accepted' "
            "ELSE excluded.retention END",
            (ref.sha256, ref.uri, ref.media_type, ref.size, retention, time.time()),
        )

    def record_source_start(self, row: dict[str, Any]) -> None:
        now = time.time()
        self._conn().execute(
            "INSERT INTO agent_source_run VALUES(?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(campaign_id,source_sha256,prompt_revision) DO UPDATE SET "
            "thread_id=excluded.thread_id,source_id=excluded.source_id,updated_at=excluded.updated_at",
            (row["campaign_id"], row["source_sha256"], row["prompt_revision"],
             row["thread_id"], row["source_id"], "running", now, now, "{}", None),
        )

    def record_source_finish(
        self, campaign_id: str, source_sha256: str, prompt_revision: str,
        status: str, counts: dict[str, Any], manifest: dict[str, Any] | None,
    ) -> None:
        self._conn().execute(
            "UPDATE agent_source_run SET status=?,updated_at=?,counts_json=?,manifest_json=? "
            "WHERE campaign_id=? AND source_sha256=? AND prompt_revision=?",
            (status, time.time(), _json(counts), _json(manifest) if manifest else None,
             campaign_id, source_sha256, prompt_revision),
        )

    def record_branch(self, row: dict[str, Any]) -> None:
        self._conn().execute(
            "INSERT INTO agent_branch VALUES(?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(campaign_id,branch_id) DO UPDATE "
            "SET status=excluded.status,result_json=excluded.result_json",
            (row["branch_id"], row["campaign_id"], row["source_sha256"],
             row.get("parent_id"), row["level"], row["status"],
             _json(row.get("proposal", {})), _json(row.get("result", {})), time.time()),
        )

    def record_proposal_audit(self, row: dict[str, Any]) -> None:
        self._conn().execute(
            "INSERT INTO proposal_audit VALUES(?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(campaign_id,branch_id) DO UPDATE "
            "SET scorer_top1=excluded.scorer_top1,"
            "scorer_top3=excluded.scorer_top3,"
            "scorer_top1_raw=excluded.scorer_top1_raw,"
            "direction_cosine=excluded.direction_cosine",
            _proposal_audit_values(row),
        )

    def record_render(self, row: dict[str, Any]) -> None:
        self._conn().execute(
            "INSERT INTO render_record VALUES(?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(render_hash) DO NOTHING",
            (row["render_hash"], row["source_sha256"], row["branch_id"], row["stage"],
             _json(row["input"]), _json(row["parameters"]), _json(row["metrics"]),
             _json(row.get("artifact")) if row.get("artifact") else None,
             row["status"], time.time()),
        )

    def get_render(self, render_hash: str) -> dict[str, Any] | None:
        row = self._conn().execute(
            "SELECT * FROM render_record WHERE render_hash=?", (render_hash,)
        ).fetchone()
        return _sqlite_row(row) if row else None

    def record_render_calibration(self, calibration_hash: str, render_hash: str) -> None:
        self._conn().execute(
            "INSERT INTO render_calibration VALUES(?,?,?) ON CONFLICT(calibration_hash) "
            "DO UPDATE SET render_hash=excluded.render_hash",
            (calibration_hash, render_hash, time.time()),
        )

    def get_render_calibration(self, calibration_hash: str) -> str | None:
        row = self._conn().execute(
            "SELECT render_hash FROM render_calibration WHERE calibration_hash=?",
            (calibration_hash,),
        ).fetchone()
        return str(row[0]) if row else None

    def record_validation(self, row: dict[str, Any]) -> None:
        self._conn().execute(
            "INSERT INTO validation_record VALUES(?,?,?,?,?,?,?,?) "
            "ON CONFLICT(validation_id) DO UPDATE SET passed=excluded.passed,"
            "defects_json=excluded.defects_json,raw_json=excluded.raw_json",
            (row["validation_id"], row["source_sha256"], row["branch_id"],
             row.get("request_hash"), int(row["passed"]), _json(row.get("defects", [])),
             _json(row.get("raw", {})), time.time()),
        )

    def record_preflight(self, key: str, row: dict[str, Any], passed: bool) -> None:
        self._conn().execute(
            "INSERT INTO provider_preflight VALUES(?,?,?,?) ON CONFLICT(preflight_key) "
            "DO UPDATE SET passed=excluded.passed,result_json=excluded.result_json,"
            "created_at=excluded.created_at",
            (key, int(passed), _json(row), time.time()),
        )

    def get_preflight(self, key: str) -> dict[str, Any] | None:
        row = self._conn().execute(
            "SELECT passed,result_json,created_at FROM provider_preflight WHERE preflight_key=?",
            (key,),
        ).fetchone()
        if not row:
            return None
        return {"passed": bool(row[0]), "result": _loads(row[1]), "created_at": row[2]}

    def record_scheduler_event(self, row: dict[str, Any]) -> None:
        self._conn().execute(
            "INSERT INTO scheduler_event VALUES(?,?,?,?,?,?,?,?,?)",
            (uuid.uuid4().hex, row["campaign_id"], row["source_id"], row["stage"],
             row["prefix_key"], row["event"], int(row["in_flight"]),
             int(row["effective_limit"]), float(row["at"])),
        )

    def mark_artifact_purged(self, sha256: str) -> bool:
        """Claim a quarantined blob for deletion; True only when this call won it.

        C1b item 12: the retention column is the reference check. `record_artifact`
        makes `accepted` sticky, so any blob that a committed row anywhere in the
        campaign still points at is `accepted` and this conditional UPDATE matches
        nothing. Callers unlink only on True.
        """
        cursor = self._conn().execute(
            "UPDATE artifact_record SET retention='purged' WHERE sha256=? AND retention='quarantine'",
            (sha256,),
        )
        return bool(cursor.rowcount)

    def source_status(self, campaign_id: str | None = None) -> list[dict[str, Any]]:
        if campaign_id:
            rows = self._conn().execute(
                "SELECT * FROM agent_source_run WHERE campaign_id=? ORDER BY started_at",
                (campaign_id,),
            ).fetchall()
        else:
            rows = self._conn().execute(
                "SELECT * FROM agent_source_run ORDER BY started_at"
            ).fetchall()
        return [_sqlite_row(row) for row in rows]

    def export_tables(self) -> dict[str, list[dict[str, Any]]]:
        tables = (
            "agent_source_run", "agent_branch", "proposal_audit", "api_request",
            "api_attempt", "api_cache_event", "api_request_context",
            "render_record", "render_calibration", "validation_record",
            "provider_preflight", "scheduler_event", "artifact_record",
        )
        return {
            table: [_sqlite_row(row) for row in self._conn().execute(
                f"SELECT * FROM {table}"  # table names are a closed internal tuple
            ).fetchall()]
            for table in tables
        }

    def artifacts_with_retention(self, retention: str) -> list[dict[str, Any]]:
        rows = self._conn().execute(
            "SELECT * FROM artifact_record WHERE retention=? ORDER BY sha256",
            (retention,),
        ).fetchall()
        return [_sqlite_row(row) for row in rows]

    def database_storage(self) -> dict[str, int]:
        conn = self._conn()
        page_count = int(conn.execute("PRAGMA page_count").fetchone()[0])
        page_size = int(conn.execute("PRAGMA page_size").fetchone()[0])
        return {"bytes": page_count * page_size}


def _sqlite_row(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    for key in tuple(result):
        if key.endswith("_json") and result[key] is not None:
            result[key] = _loads(result[key])
    return result


_POSTGRES_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_source_run (
 campaign_id TEXT NOT NULL, source_sha256 TEXT NOT NULL, prompt_revision TEXT NOT NULL,
 thread_id TEXT NOT NULL, source_id TEXT NOT NULL, status TEXT NOT NULL,
 started_at DOUBLE PRECISION NOT NULL, updated_at DOUBLE PRECISION NOT NULL,
 counts_json TEXT NOT NULL, manifest_json TEXT,
 PRIMARY KEY(campaign_id, source_sha256, prompt_revision)
);
CREATE TABLE IF NOT EXISTS agent_branch (
 branch_id TEXT NOT NULL, campaign_id TEXT NOT NULL, source_sha256 TEXT NOT NULL,
 parent_id TEXT, level TEXT NOT NULL, status TEXT NOT NULL, proposal_json TEXT NOT NULL,
 result_json TEXT NOT NULL, created_at DOUBLE PRECISION NOT NULL,
 PRIMARY KEY(campaign_id, branch_id)
);
CREATE TABLE IF NOT EXISTS proposal_audit (
 branch_id TEXT NOT NULL, campaign_id TEXT NOT NULL, source_sha256 TEXT NOT NULL,
 level TEXT NOT NULL, preset_id TEXT NOT NULL, scorer_top1 INTEGER,
 scorer_top3 INTEGER, scorer_top1_raw INTEGER, direction_cosine DOUBLE PRECISION,
 created_at DOUBLE PRECISION NOT NULL,
 PRIMARY KEY(campaign_id, branch_id)
);
CREATE TABLE IF NOT EXISTS api_request (
 request_hash TEXT PRIMARY KEY, canonical_request_json TEXT NOT NULL,
 status TEXT NOT NULL, owner_id TEXT, lease_expires_at DOUBLE PRECISION,
 resolved_attempt_id BIGINT, response_json TEXT, usage_json TEXT,
 created_at DOUBLE PRECISION NOT NULL, updated_at DOUBLE PRECISION NOT NULL
);
CREATE TABLE IF NOT EXISTS api_attempt (
 request_hash TEXT NOT NULL, attempt_no BIGINT NOT NULL,
 started_at DOUBLE PRECISION NOT NULL, finished_at DOUBLE PRECISION NOT NULL,
 response_json TEXT, usage_json TEXT, validation_json TEXT NOT NULL,
 valid INTEGER NOT NULL, error_type TEXT,
 PRIMARY KEY(request_hash, attempt_no)
);
CREATE TABLE IF NOT EXISTS api_cache_event (
 event_id TEXT PRIMARY KEY, request_hash TEXT NOT NULL, event TEXT NOT NULL,
 created_at DOUBLE PRECISION NOT NULL
);
CREATE TABLE IF NOT EXISTS api_request_context (
 request_hash TEXT NOT NULL, campaign_id TEXT NOT NULL, source_sha256 TEXT NOT NULL,
 stage TEXT NOT NULL, cache_hit INTEGER NOT NULL, created_at DOUBLE PRECISION NOT NULL,
 PRIMARY KEY(request_hash,campaign_id,source_sha256,stage)
);
CREATE TABLE IF NOT EXISTS render_record (
 render_hash TEXT PRIMARY KEY, source_sha256 TEXT NOT NULL, branch_id TEXT NOT NULL,
 stage TEXT NOT NULL, input_json TEXT NOT NULL, parameters_json TEXT NOT NULL,
 metrics_json TEXT NOT NULL, artifact_json TEXT, status TEXT NOT NULL,
 created_at DOUBLE PRECISION NOT NULL
);
CREATE TABLE IF NOT EXISTS render_calibration (
 calibration_hash TEXT PRIMARY KEY, render_hash TEXT NOT NULL,
 created_at DOUBLE PRECISION NOT NULL
);
CREATE TABLE IF NOT EXISTS validation_record (
 validation_id TEXT PRIMARY KEY, source_sha256 TEXT NOT NULL, branch_id TEXT NOT NULL,
 request_hash TEXT, passed INTEGER NOT NULL, defects_json TEXT NOT NULL,
 raw_json TEXT NOT NULL, created_at DOUBLE PRECISION NOT NULL
);
CREATE TABLE IF NOT EXISTS provider_preflight (
 preflight_key TEXT PRIMARY KEY, passed INTEGER NOT NULL, result_json TEXT NOT NULL,
 created_at DOUBLE PRECISION NOT NULL
);
CREATE TABLE IF NOT EXISTS scheduler_event (
 event_id TEXT PRIMARY KEY, campaign_id TEXT NOT NULL, source_id TEXT NOT NULL,
 stage TEXT NOT NULL, prefix_key TEXT NOT NULL, event TEXT NOT NULL,
 in_flight INTEGER NOT NULL, effective_limit INTEGER NOT NULL,
 created_at DOUBLE PRECISION NOT NULL
);
CREATE TABLE IF NOT EXISTS artifact_record (
 sha256 TEXT PRIMARY KEY, uri TEXT NOT NULL, media_type TEXT NOT NULL, size BIGINT NOT NULL,
 retention TEXT NOT NULL, created_at DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_source_status ON agent_source_run(campaign_id,status);
CREATE INDEX IF NOT EXISTS idx_branch_source ON agent_branch(campaign_id,source_sha256);
CREATE INDEX IF NOT EXISTS idx_attempt_request ON api_attempt(request_hash);
"""


class PostgresAuditStore(AuditStore):
    """PostgreSQL implementation with transactional row leases."""

    def __init__(self, dsn: str) -> None:
        from psycopg_pool import ConnectionPool

        self._pool = ConnectionPool(conninfo=dsn, min_size=1, max_size=20, open=False)
        self._pool.open()

    def setup(self) -> None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT current_schema()")
                current_schema = str(cur.fetchone()[0] or "")
                if current_schema == "public" or not current_schema:
                    raise RuntimeError(
                        "agent loop PostgreSQL DSN must set a non-public search_path"
                    )
                for statement in _POSTGRES_SCHEMA.split(";"):
                    if statement.strip():
                        cur.execute(statement)
                cur.execute(
                    "ALTER TABLE api_request ALTER COLUMN lease_expires_at "
                    "TYPE DOUBLE PRECISION USING lease_expires_at::DOUBLE PRECISION"
                )
                self._migrate_campaign_primary_key(cur)
            conn.commit()

    @staticmethod
    def _migrate_campaign_primary_key(cur: Any) -> None:
        for table in _CAMPAIGN_PK_TABLES:  # closed internal tuple, not user input
            cur.execute(
                "SELECT c.conname, ARRAY(SELECT a.attname FROM unnest(c.conkey) AS k "
                "JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = k) "
                "FROM pg_constraint c "
                "WHERE c.conrelid = to_regclass(%s) AND c.contype = 'p'",
                (table,),
            )
            row = cur.fetchone()
            if row is None or set(row[1]) == set(_CAMPAIGN_PK_COLUMNS):
                continue
            cur.execute(f'ALTER TABLE {table} DROP CONSTRAINT "{row[0]}"')
            cur.execute(
                f"ALTER TABLE {table} ADD PRIMARY KEY "
                f"({', '.join(_CAMPAIGN_PK_COLUMNS)})"
            )

    def acquire_request(
        self, request_hash: str, canonical_request: dict[str, Any], owner_id: str,
        lease_seconds: int,
    ) -> dict[str, Any]:
        now = time.time()
        manifest = _json(canonical_request)
        with self._pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            cur.execute(
                "INSERT INTO api_request(request_hash,canonical_request_json,status,created_at,updated_at) "
                "VALUES(%s,%s,'pending',%s,%s) ON CONFLICT(request_hash) DO NOTHING",
                (request_hash, manifest, now, now),
            )
            cur.execute("SELECT * FROM api_request WHERE request_hash=%s FOR UPDATE", (request_hash,))
            names = [column.name for column in cur.description]
            row = dict(zip(names, cur.fetchone()))
            if row["canonical_request_json"] != manifest:
                raise RuntimeError("request_hash_collision")
            if row["status"] == "resolved":
                return {"action": "resolved", "response": _loads(row["response_json"]),
                        "usage": _loads(row["usage_json"]),
                        "attempt_no": row["resolved_attempt_id"]}
            if row["owner_id"] == owner_id or float(row["lease_expires_at"] or 0) <= now:
                cur.execute(
                    "UPDATE api_request SET owner_id=%s,lease_expires_at=%s,updated_at=%s "
                    "WHERE request_hash=%s", (owner_id, now + lease_seconds, now, request_hash),
                )
                return {"action": "owner"}
            return {"action": "wait", "lease_expires_at": float(row["lease_expires_at"])}

    def record_attempt(
        self, request_hash: str, owner_id: str, *, response: dict[str, Any] | None,
        usage: dict[str, Any] | None, validation: dict[str, Any], valid: bool,
        error_type: str | None = None,
    ) -> int:
        now = time.time()
        with self._pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            cur.execute(
                "SELECT owner_id,status FROM api_request WHERE request_hash=%s FOR UPDATE",
                (request_hash,),
            )
            row = cur.fetchone()
            if row is None or (row[0] != owner_id and row[1] != "resolved"):
                raise RuntimeError("request_lease_not_owned")
            cur.execute(
                "SELECT COALESCE(MAX(attempt_no),0)+1 FROM api_attempt WHERE request_hash=%s",
                (request_hash,),
            )
            attempt_no = int(cur.fetchone()[0])
            cur.execute(
                "INSERT INTO api_attempt VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (request_hash, attempt_no, now, now, _json(response) if response else None,
                 _json(usage) if usage else None, _json(validation), int(valid), error_type),
            )
            if valid:
                cur.execute(
                    "UPDATE api_request SET status='resolved',resolved_attempt_id=%s,"
                    "response_json=%s,usage_json=%s,owner_id=NULL,lease_expires_at=NULL,updated_at=%s "
                    "WHERE request_hash=%s",
                    (attempt_no, _json(response), _json(usage or {}), now, request_hash),
                )
            else:
                cur.execute("UPDATE api_request SET updated_at=%s WHERE request_hash=%s",
                            (now, request_hash))
            return attempt_no

    def _execute(self, sql: str, values: Iterable[Any] = (), *, fetch: bool = False):
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(sql, tuple(values))
            rows = cur.fetchall() if fetch else []
            names = [column.name for column in cur.description] if fetch else []
            conn.commit()
        return [dict(zip(names, row)) for row in rows]

    def get_request(self, request_hash: str) -> dict[str, Any] | None:
        rows = self._execute("SELECT * FROM api_request WHERE request_hash=%s",
                             (request_hash,), fetch=True)
        return _postgres_row(rows[0]) if rows else None

    def record_cache_event(self, request_hash: str, event: str) -> None:
        self._execute(
            "INSERT INTO api_cache_event VALUES(%s,%s,%s,%s)",
            (uuid.uuid4().hex, request_hash, event, time.time()),
        )

    def release_request(self, request_hash: str, owner_id: str) -> None:
        self._execute(
            "UPDATE api_request SET owner_id=NULL,lease_expires_at=0,updated_at=%s "
            "WHERE request_hash=%s AND owner_id=%s AND status='pending'",
            (time.time(), request_hash, owner_id),
        )

    def record_request_context(
        self, request_hash: str, campaign_id: str, source_sha256: str, stage: str,
        cache_hit: bool,
    ) -> None:
        self._execute(
            "INSERT INTO api_request_context VALUES(%s,%s,%s,%s,%s,%s) ON CONFLICT(" 
            "request_hash,campaign_id,source_sha256,stage) DO UPDATE SET "
            "cache_hit=LEAST(api_request_context.cache_hit,excluded.cache_hit)",
            (request_hash, campaign_id, source_sha256, stage, int(cache_hit), time.time()),
        )

    def record_artifact(self, ref: Any, retention: str) -> None:
        self._execute(
            "INSERT INTO artifact_record VALUES(%s,%s,%s,%s,%s,%s) ON CONFLICT(sha256) "
            "DO UPDATE SET retention=CASE WHEN artifact_record.retention='accepted' THEN "
            "'accepted' ELSE excluded.retention END",
            (ref.sha256, ref.uri, ref.media_type, ref.size, retention, time.time()),
        )

    def record_source_start(self, row: dict[str, Any]) -> None:
        now = time.time()
        self._execute(
            "INSERT INTO agent_source_run VALUES(%s,%s,%s,%s,%s,'running',%s,%s,'{}',NULL) "
            "ON CONFLICT(campaign_id,source_sha256,prompt_revision) DO UPDATE SET "
            "thread_id=excluded.thread_id,source_id=excluded.source_id,updated_at=excluded.updated_at",
            (row["campaign_id"], row["source_sha256"], row["prompt_revision"],
             row["thread_id"], row["source_id"], now, now),
        )

    def record_source_finish(self, campaign_id: str, source_sha256: str,
                             prompt_revision: str, status: str, counts: dict[str, Any],
                             manifest: dict[str, Any] | None) -> None:
        self._execute(
            "UPDATE agent_source_run SET status=%s,updated_at=%s,counts_json=%s,manifest_json=%s "
            "WHERE campaign_id=%s AND source_sha256=%s AND prompt_revision=%s",
            (status, time.time(), _json(counts), _json(manifest) if manifest else None,
             campaign_id, source_sha256, prompt_revision),
        )

    def record_branch(self, row: dict[str, Any]) -> None:
        self._execute(
            "INSERT INTO agent_branch VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s) "
            "ON CONFLICT(campaign_id,branch_id) DO UPDATE "
            "SET status=excluded.status,result_json=excluded.result_json",
            (row["branch_id"], row["campaign_id"], row["source_sha256"], row.get("parent_id"),
             row["level"], row["status"], _json(row.get("proposal", {})),
             _json(row.get("result", {})), time.time()),
        )

    def record_proposal_audit(self, row: dict[str, Any]) -> None:
        self._execute(
            "INSERT INTO proposal_audit VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
            "ON CONFLICT(campaign_id,branch_id) DO UPDATE "
            "SET scorer_top1=excluded.scorer_top1,"
            "scorer_top3=excluded.scorer_top3,"
            "scorer_top1_raw=excluded.scorer_top1_raw,"
            "direction_cosine=excluded.direction_cosine",
            _proposal_audit_values(row),
        )

    def record_render(self, row: dict[str, Any]) -> None:
        self._execute(
            "INSERT INTO render_record VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
            "ON CONFLICT(render_hash) DO NOTHING",
            (row["render_hash"], row["source_sha256"], row["branch_id"], row["stage"],
             _json(row["input"]), _json(row["parameters"]), _json(row["metrics"]),
             _json(row.get("artifact")) if row.get("artifact") else None,
             row["status"], time.time()),
        )

    def get_render(self, render_hash: str) -> dict[str, Any] | None:
        rows = self._execute("SELECT * FROM render_record WHERE render_hash=%s",
                             (render_hash,), fetch=True)
        if not rows:
            return None
        row = rows[0]
        for key in tuple(row):
            if key.endswith("_json") and row[key] is not None:
                row[key] = _loads(row[key])
        return row

    def record_render_calibration(self, calibration_hash: str, render_hash: str) -> None:
        self._execute(
            "INSERT INTO render_calibration VALUES(%s,%s,%s) ON CONFLICT(calibration_hash) "
            "DO UPDATE SET render_hash=excluded.render_hash",
            (calibration_hash, render_hash, time.time()),
        )

    def get_render_calibration(self, calibration_hash: str) -> str | None:
        rows = self._execute(
            "SELECT render_hash FROM render_calibration WHERE calibration_hash=%s",
            (calibration_hash,), fetch=True,
        )
        return str(rows[0]["render_hash"]) if rows else None

    def record_validation(self, row: dict[str, Any]) -> None:
        self._execute(
            "INSERT INTO validation_record VALUES(%s,%s,%s,%s,%s,%s,%s,%s) "
            "ON CONFLICT(validation_id) DO UPDATE SET passed=excluded.passed,"
            "defects_json=excluded.defects_json,raw_json=excluded.raw_json",
            (row["validation_id"], row["source_sha256"], row["branch_id"],
             row.get("request_hash"), int(row["passed"]), _json(row.get("defects", [])),
             _json(row.get("raw", {})), time.time()),
        )

    def record_preflight(self, key: str, row: dict[str, Any], passed: bool) -> None:
        self._execute(
            "INSERT INTO provider_preflight VALUES(%s,%s,%s,%s) ON CONFLICT(preflight_key) "
            "DO UPDATE SET passed=excluded.passed,result_json=excluded.result_json,"
            "created_at=excluded.created_at", (key, int(passed), _json(row), time.time()),
        )

    def get_preflight(self, key: str) -> dict[str, Any] | None:
        rows = self._execute("SELECT * FROM provider_preflight WHERE preflight_key=%s",
                             (key,), fetch=True)
        if not rows:
            return None
        return {"passed": bool(rows[0]["passed"]), "result": _loads(rows[0]["result_json"]),
                "created_at": rows[0]["created_at"]}

    def record_scheduler_event(self, row: dict[str, Any]) -> None:
        self._execute(
            "INSERT INTO scheduler_event VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (uuid.uuid4().hex, row["campaign_id"], row["source_id"], row["stage"],
             row["prefix_key"], row["event"], int(row["in_flight"]),
             int(row["effective_limit"]), float(row["at"])),
        )

    def mark_artifact_purged(self, sha256: str) -> bool:
        """Claim a quarantined blob for deletion; True only when this call won it.

        C1b item 12: `RETURNING` makes the claim and its answer one statement, so two
        concurrent passes over the same deduplicated blob cannot both unlink it.
        """
        rows = self._execute(
            "UPDATE artifact_record SET retention='purged' WHERE sha256=%s "
            "AND retention='quarantine' RETURNING sha256", (sha256,), fetch=True,
        )
        return bool(rows)

    def source_status(self, campaign_id: str | None = None) -> list[dict[str, Any]]:
        if campaign_id:
            rows = self._execute(
                "SELECT * FROM agent_source_run WHERE campaign_id=%s ORDER BY started_at",
                (campaign_id,), fetch=True)
        else:
            rows = self._execute("SELECT * FROM agent_source_run ORDER BY started_at", fetch=True)
        return [_postgres_row(row) for row in rows]

    def export_tables(self) -> dict[str, list[dict[str, Any]]]:
        tables = (
            "agent_source_run", "agent_branch", "proposal_audit", "api_request",
            "api_attempt", "api_cache_event", "api_request_context",
            "render_record", "render_calibration", "validation_record",
            "provider_preflight", "scheduler_event", "artifact_record",
        )
        return {
            table: [_postgres_row(row) for row in self._execute(
                f"SELECT * FROM {table}", fetch=True
            )]
            for table in tables
        }

    def artifacts_with_retention(self, retention: str) -> list[dict[str, Any]]:
        return [
            _postgres_row(row) for row in self._execute(
                "SELECT * FROM artifact_record WHERE retention=%s ORDER BY sha256",
                (retention,), fetch=True,
            )
        ]

    def database_storage(self) -> dict[str, int]:
        rows = self._execute(
            "SELECT COALESCE(SUM(pg_total_relation_size(c.oid)),0) AS bytes "
            "FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname=current_schema() AND c.relkind IN ('r','p','m')",
            fetch=True,
        )
        return {"bytes": int(rows[0]["bytes"] if rows else 0)}


def _postgres_row(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    for key in tuple(result):
        if key.endswith("_json") and result[key] is not None:
            result[key] = _loads(result[key])
    return result


def make_audit_store(backend: str, location: str) -> AuditStore:
    store: AuditStore
    if backend == "sqlite":
        store = SQLiteAuditStore(location)
    elif backend == "postgres":
        store = PostgresAuditStore(location)
    else:
        raise ValueError(f"unsupported audit backend: {backend}")
    store.setup()
    return store


__all__ = [
    "AuditStore", "PostgresAuditStore", "SQLiteAuditStore", "make_audit_store",
]
