"""Idempotent PostgreSQL projection of canonical JSON artifacts for the viewer."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from .config import redact_text, uri_secrets
from .state import ArtifactStore


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS canonical_builds (
    build_id TEXT PRIMARY KEY,
    schema_version INTEGER NOT NULL,
    phase TEXT NOT NULL,
    status TEXT NOT NULL,
    manifest JSONB NOT NULL,
    projected_at TIMESTAMPTZ NOT NULL
);
CREATE TABLE IF NOT EXISTS canonical_groups (
    group_id TEXT PRIMARY KEY,
    build_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    source_path TEXT NOT NULL,
    render_mode TEXT NOT NULL,
    preset_filter TEXT,
    major TEXT,
    scene TEXT,
    winner_ids JSONB NOT NULL,
    payload JSONB NOT NULL,
    projected_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS canonical_groups_build_mode_idx
    ON canonical_groups(build_id, render_mode);
CREATE INDEX IF NOT EXISTS canonical_groups_source_idx
    ON canonical_groups(source_id);
CREATE TABLE IF NOT EXISTS canonical_candidates (
    candidate_id TEXT PRIMARY KEY,
    group_id TEXT NOT NULL,
    build_id TEXT NOT NULL,
    slot_index INTEGER NOT NULL,
    slot_id TEXT,
    preset_id TEXT,
    preset_format TEXT,
    major TEXT,
    minor TEXT,
    after_path TEXT NOT NULL,
    render_engine TEXT,
    mask_id TEXT,
    cgt_path TEXT,
    region TEXT,
    winner BOOLEAN NOT NULL,
    rank INTEGER,
    qa JSONB NOT NULL,
    payload JSONB NOT NULL,
    projected_at TIMESTAMPTZ NOT NULL,
    UNIQUE(group_id, slot_index)
);
CREATE INDEX IF NOT EXISTS canonical_candidates_group_idx
    ON canonical_candidates(group_id, slot_index);
CREATE INDEX IF NOT EXISTS canonical_candidates_filter_idx
    ON canonical_candidates(build_id, preset_format, major, minor, winner);
CREATE TABLE IF NOT EXISTS canonical_sft (
    sft_id TEXT PRIMARY KEY,
    build_id TEXT NOT NULL,
    annotation_task_id TEXT NOT NULL UNIQUE,
    group_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    winner_rank INTEGER NOT NULL,
    input_path TEXT NOT NULL,
    target_path TEXT NOT NULL,
    task_type TEXT NOT NULL,
    annot_src TEXT NOT NULL,
    instruction TEXT NOT NULL,
    instruction_short TEXT NOT NULL,
    reasoning TEXT NOT NULL,
    qa JSONB NOT NULL,
    payload JSONB NOT NULL,
    projected_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS canonical_sft_group_idx ON canonical_sft(group_id, winner_rank);
CREATE TABLE IF NOT EXISTS canonical_failures (
    event_id TEXT PRIMARY KEY,
    build_id TEXT NOT NULL,
    event_type TEXT,
    stage TEXT,
    task_id TEXT,
    source_id TEXT,
    group_id TEXT,
    candidate_id TEXT,
    round_number INTEGER,
    attempt INTEGER,
    retryable BOOLEAN NOT NULL,
    error_code TEXT NOT NULL,
    message TEXT NOT NULL,
    endpoint_id TEXT,
    terminal BOOLEAN NOT NULL,
    payload JSONB NOT NULL,
    projected_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS canonical_failures_filter_idx
    ON canonical_failures(build_id, stage, error_code, terminal);
CREATE INDEX IF NOT EXISTS canonical_failures_task_idx ON canonical_failures(task_id);
"""


@dataclass(frozen=True, slots=True)
class ProjectionResult:
    ok: bool
    counts: dict[str, int]
    error: str | None = None


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _optional_identifier(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _execute_many(cursor: Any, sql: str, rows: list[tuple[Any, ...]]) -> None:
    if rows:
        cursor.executemany(sql, rows)


def _delete_stale(
    cursor: Any,
    table: str,
    id_column: str,
    build_id: str,
    current_ids: list[str],
) -> None:
    cursor.execute(
        f"DELETE FROM {table} WHERE build_id=%s "
        f"AND NOT ({id_column}=ANY(%s::text[]))",
        (build_id, current_ids),
    )


def project_artifacts(
    store: ArtifactStore,
    manifest: Mapping[str, Any],
    postgres_dsn: str,
    *,
    connect_fn: Callable[[str], Any] | None = None,
    now: Callable[[], datetime] | None = None,
) -> ProjectionResult:
    """Rebuild the viewer projection; errors never mutate authoritative artifacts."""
    if connect_fn is None:
        try:
            import psycopg
        except ImportError as exc:
            return ProjectionResult(False, {}, "postgres_driver_unavailable")
        connect_fn = psycopg.connect
    projected_at = (now or (lambda: datetime.now(timezone.utc)))()
    counts = {
        "builds": 1,
        "groups": len(store.groups),
        "candidates": sum(len(row.get("candidates") or []) for row in store.groups.values()),
        "sft": len(store.sft),
        "failures": len(store.failures),
    }
    connection = None
    try:
        connection = connect_fn(postgres_dsn)
        cursor = connection.cursor()
        cursor.execute(SCHEMA_SQL)
        cursor.execute(
            """INSERT INTO canonical_builds
               (build_id,schema_version,phase,status,manifest,projected_at)
               VALUES (%s,%s,%s,%s,%s::jsonb,%s)
               ON CONFLICT (build_id) DO UPDATE SET
                 schema_version=EXCLUDED.schema_version, phase=EXCLUDED.phase,
                 status=EXCLUDED.status, manifest=EXCLUDED.manifest,
                 projected_at=EXCLUDED.projected_at""",
            (
                store.build_id, int(manifest.get("schema_version", 1)),
                str(manifest.get("phase") or "projection"),
                str(manifest.get("status") or "running"), _json(manifest), projected_at,
            ),
        )
        group_rows: list[tuple[Any, ...]] = []
        candidate_rows: list[tuple[Any, ...]] = []
        for group in store.groups.values():
            winner_ids = list(group.get("winner_ids") or [])
            winners = set(winner_ids)
            group_rows.append((
                group["group_id"], group["build_id"], group["source_id"],
                group["source_path"], group["render_mode"], group.get("preset_filter"),
                group.get("major"), group.get("scene"), _json(winner_ids),
                _json(group), projected_at,
            ))
            for slot_index, candidate in enumerate(group.get("candidates") or []):
                candidate_rows.append((
                    candidate["candidate_id"], group["group_id"], group["build_id"],
                    slot_index, candidate.get("slot_id"), candidate.get("preset_id"),
                    candidate.get("format"), candidate.get("major"), candidate.get("minor"),
                    candidate["after_path"], candidate.get("render_engine"),
                    candidate.get("mask_id"), candidate.get("cgt_path"),
                    candidate.get("region"), candidate["candidate_id"] in winners,
                    candidate.get("rank"), _json(candidate.get("qa") or {}),
                    _json(candidate), projected_at,
                ))
        _execute_many(cursor, """INSERT INTO canonical_groups
            (group_id,build_id,source_id,source_path,render_mode,preset_filter,major,scene,
             winner_ids,payload,projected_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s)
            ON CONFLICT (group_id) DO UPDATE SET
             build_id=EXCLUDED.build_id,source_id=EXCLUDED.source_id,
             source_path=EXCLUDED.source_path,render_mode=EXCLUDED.render_mode,
             preset_filter=EXCLUDED.preset_filter,major=EXCLUDED.major,scene=EXCLUDED.scene,
             winner_ids=EXCLUDED.winner_ids,payload=EXCLUDED.payload,
             projected_at=EXCLUDED.projected_at""", group_rows)
        _execute_many(cursor, """INSERT INTO canonical_candidates
            (candidate_id,group_id,build_id,slot_index,slot_id,preset_id,preset_format,
             major,minor,after_path,render_engine,mask_id,cgt_path,region,winner,rank,qa,
             payload,projected_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s)
            ON CONFLICT (candidate_id) DO UPDATE SET
             group_id=EXCLUDED.group_id,build_id=EXCLUDED.build_id,
             slot_index=EXCLUDED.slot_index,slot_id=EXCLUDED.slot_id,
             preset_id=EXCLUDED.preset_id,preset_format=EXCLUDED.preset_format,
             major=EXCLUDED.major,minor=EXCLUDED.minor,after_path=EXCLUDED.after_path,
             render_engine=EXCLUDED.render_engine,mask_id=EXCLUDED.mask_id,
             cgt_path=EXCLUDED.cgt_path,region=EXCLUDED.region,winner=EXCLUDED.winner,
             rank=EXCLUDED.rank,qa=EXCLUDED.qa,payload=EXCLUDED.payload,
             projected_at=EXCLUDED.projected_at""", candidate_rows)
        sft_rows = [(
            row["sft_id"], row["build_id"], row["annotation_task_id"], row["group_id"],
            row["candidate_id"], row["winner_rank"], row["I_in"], row["I_tar"],
            row["task_type"], row["annot_src"], row["instruction"],
            row["instruction_short"], row["reasoning"], _json(row.get("qa") or {}),
            _json(row), projected_at,
        ) for row in store.sft.values()]
        _execute_many(cursor, """INSERT INTO canonical_sft
            (sft_id,build_id,annotation_task_id,group_id,candidate_id,winner_rank,
             input_path,target_path,task_type,annot_src,instruction,instruction_short,
             reasoning,qa,payload,projected_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s)
            ON CONFLICT (sft_id) DO UPDATE SET
             build_id=EXCLUDED.build_id,annotation_task_id=EXCLUDED.annotation_task_id,
             group_id=EXCLUDED.group_id,candidate_id=EXCLUDED.candidate_id,
             winner_rank=EXCLUDED.winner_rank,input_path=EXCLUDED.input_path,
             target_path=EXCLUDED.target_path,task_type=EXCLUDED.task_type,
             annot_src=EXCLUDED.annot_src,instruction=EXCLUDED.instruction,
             instruction_short=EXCLUDED.instruction_short,reasoning=EXCLUDED.reasoning,
             qa=EXCLUDED.qa,payload=EXCLUDED.payload,projected_at=EXCLUDED.projected_at""",
            sft_rows)
        failure_rows = [(
            row["event_id"], row["build_id"], row.get("event_type"), row.get("stage"),
            row.get("task_id"), _optional_identifier(row.get("source_id")),
            _optional_identifier(row.get("group_id")),
            _optional_identifier(row.get("candidate_id")), row.get("round"), row.get("attempt"),
            bool(row.get("retryable")), str(row.get("error_code") or "unknown"),
            str(row.get("message") or ""), row.get("endpoint_id"), bool(row.get("terminal")),
            _json(row), projected_at,
        ) for row in store.failures]
        _execute_many(cursor, """INSERT INTO canonical_failures
            (event_id,build_id,event_type,stage,task_id,source_id,group_id,candidate_id,
             round_number,attempt,retryable,error_code,message,endpoint_id,terminal,payload,
             projected_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s)
            ON CONFLICT (event_id) DO UPDATE SET
             build_id=EXCLUDED.build_id,event_type=EXCLUDED.event_type,stage=EXCLUDED.stage,
             task_id=EXCLUDED.task_id,source_id=EXCLUDED.source_id,
             group_id=EXCLUDED.group_id,candidate_id=EXCLUDED.candidate_id,
             round_number=EXCLUDED.round_number,attempt=EXCLUDED.attempt,
             retryable=EXCLUDED.retryable,error_code=EXCLUDED.error_code,
             message=EXCLUDED.message,endpoint_id=EXCLUDED.endpoint_id,
             terminal=EXCLUDED.terminal,payload=EXCLUDED.payload,
             projected_at=EXCLUDED.projected_at""", failure_rows)
        _delete_stale(
            cursor, "canonical_candidates", "candidate_id", store.build_id,
            [row[0] for row in candidate_rows],
        )
        _delete_stale(
            cursor, "canonical_sft", "sft_id", store.build_id,
            [row[0] for row in sft_rows],
        )
        _delete_stale(
            cursor, "canonical_failures", "event_id", store.build_id,
            [row[0] for row in failure_rows],
        )
        _delete_stale(
            cursor, "canonical_groups", "group_id", store.build_id,
            [row[0] for row in group_rows],
        )
        connection.commit()
        cursor.close()
        return ProjectionResult(True, counts)
    except Exception as exc:  # noqa: BLE001 - projection is explicitly non-authoritative
        if connection is not None:
            try:
                connection.rollback()
            except Exception:
                pass
        return ProjectionResult(
            False, counts,
            redact_text(f"{type(exc).__name__}: {exc}", (postgres_dsn, *uri_secrets(postgres_dsn))),
        )
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
