"""Read-only access to canonical databuild projections and JSONL artifacts."""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


QUEUE_STATES = ("complete", "pending", "failed", "none")
FAILURE_STATES = ("terminal", "retryable", "recorded", "none")
WINNER_FILTERS = ("any", "yes", "no", "top1", "top2")


_SAFE_WINNER_IDS_SQL = (
    "(CASE WHEN jsonb_typeof(g.winner_ids)='array' "
    "THEN g.winner_ids ELSE '[]'::jsonb END)"
)
_SAFE_CANDIDATES_SQL = (
    "(CASE WHEN jsonb_typeof(g.payload->'candidates')='array' "
    "THEN g.payload->'candidates' ELSE '[]'::jsonb END)"
)
_CANONICAL_GROUP_SQL = (
    "(SELECT count(*) FROM canonical_candidates cg WHERE cg.group_id=g.group_id)=8 "
    "AND NOT EXISTS (SELECT 1 FROM canonical_candidates cg "
    "WHERE cg.group_id=g.group_id AND (cg.candidate_id='' "
    "OR jsonb_typeof(cg.payload)<>'object' "
    "OR cg.payload->>'candidate_id' IS DISTINCT FROM cg.candidate_id)) "
    "AND jsonb_typeof(g.payload)='object' "
    "AND g.payload->>'group_id'=g.group_id "
    "AND g.payload->>'build_id'=g.build_id "
    "AND jsonb_typeof(g.payload->'candidates')='array' "
    f"AND jsonb_array_length({_SAFE_CANDIDATES_SQL})=8 "
    f"AND NOT EXISTS (SELECT 1 FROM jsonb_array_elements({_SAFE_CANDIDATES_SQL}) "
    "AS candidate(value) WHERE jsonb_typeof(candidate.value)<>'object' "
    "OR jsonb_typeof(candidate.value->'candidate_id')<>'string' "
    "OR candidate.value->>'candidate_id'='') "
    "AND (SELECT count(DISTINCT candidate.value->>'candidate_id') FROM "
    f"jsonb_array_elements({_SAFE_CANDIDATES_SQL}) AS candidate(value))=8 "
    f"AND NOT EXISTS (SELECT 1 FROM jsonb_array_elements({_SAFE_CANDIDATES_SQL}) "
    "AS candidate(value) WHERE NOT EXISTS (SELECT 1 FROM canonical_candidates pc "
    "WHERE pc.group_id=g.group_id "
    "AND pc.candidate_id=candidate.value->>'candidate_id' "
    "AND pc.payload=candidate.value)) "
    "AND jsonb_typeof(g.winner_ids)='array' "
    "AND g.payload->'winner_ids'=g.winner_ids "
    f"AND jsonb_array_length({_SAFE_WINNER_IDS_SQL})<=2 "
    f"AND NOT EXISTS (SELECT 1 FROM jsonb_array_elements({_SAFE_WINNER_IDS_SQL}) "
    "AS typed(value) WHERE jsonb_typeof(typed.value)<>'string' "
    "OR typed.value='\"\"'::jsonb) "
    f"AND (SELECT count(*) FROM jsonb_array_elements_text({_SAFE_WINNER_IDS_SQL}))="
    "(SELECT count(DISTINCT winner.value) FROM "
    f"jsonb_array_elements_text({_SAFE_WINNER_IDS_SQL}) AS winner(value)) "
    f"AND NOT EXISTS (SELECT 1 FROM jsonb_array_elements_text({_SAFE_WINNER_IDS_SQL}) "
    "AS winner(candidate_id) WHERE NOT EXISTS (SELECT 1 FROM canonical_candidates wc "
    "WHERE wc.group_id=g.group_id AND wc.candidate_id=winner.candidate_id))"
)
_GROUP_ORDER_SQL = (
    "COALESCE(g.payload #>> '{stage_timestamps,qa_completed_at}',"
    "g.payload #>> '{stage_timestamps,render_completed_at}','') DESC,"
    "g.group_id DESC"
)


class StoreUnavailable(RuntimeError):
    """The preferred viewer store cannot currently answer a request."""


@dataclass(frozen=True, slots=True)
class GroupFilters:
    build_id: str | None = None
    mode: str | None = None
    preset_format: str | None = None
    major: str | None = None
    minor: str | None = None
    queue_state: str | None = None
    failure_state: str | None = None
    winner: str = "any"


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def _sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return []
        return list(parsed) if isinstance(parsed, list) else []
    return list(value) if isinstance(value, Sequence) else []


def _candidate_rows(group: Mapping[str, Any]) -> list[dict[str, Any]]:
    winners = set(_sequence(group.get("winner_ids")))
    rows: list[dict[str, Any]] = []
    for index, value in enumerate(_sequence(group.get("candidates"))):
        candidate = _mapping(value)
        candidate.setdefault("slot_index", index)
        candidate.setdefault("format", candidate.get("preset_format"))
        candidate["winner"] = candidate.get("candidate_id") in winners
        rows.append(candidate)
    rows.sort(key=lambda row: (int(row.get("slot_index") or 0), str(row.get("candidate_id") or "")))
    return rows


def canonical_group_error(
    group: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]] | None = None,
) -> str | None:
    rows = list(candidates) if candidates is not None else _candidate_rows(group)
    if len(rows) != 8:
        return "group does not contain exactly eight candidates"
    candidate_ids = [row.get("candidate_id") for row in rows]
    if any(not isinstance(candidate_id, str) or not candidate_id for candidate_id in candidate_ids):
        return "candidate IDs must be non-empty strings"
    if len(set(candidate_ids)) != 8:
        return "candidate IDs must be distinct"
    winner_ids = group.get("winner_ids")
    if not isinstance(winner_ids, list):
        return "winner_ids must be a list"
    if len(winner_ids) > 2 or any(
        not isinstance(winner_id, str) or not winner_id for winner_id in winner_ids
    ):
        return "winner_ids must contain at most two non-empty strings"
    if len(set(winner_ids)) != len(winner_ids):
        return "winner_ids must be distinct"
    if not set(winner_ids).issubset(candidate_ids):
        return "winner_ids must reference group candidates"
    return None


def _related_failures(
    group: Mapping[str, Any], failures: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    group_id = group.get("group_id")
    source_id = group.get("source_id")
    candidate_ids = {row.get("candidate_id") for row in _candidate_rows(group)}
    related: list[dict[str, Any]] = []
    for value in failures:
        row = dict(value)
        if (
            (row.get("group_id") and row.get("group_id") == group_id)
            or (row.get("candidate_id") and row.get("candidate_id") in candidate_ids)
            or (
                not row.get("group_id")
                and not row.get("candidate_id")
                and row.get("source_id")
                and row.get("source_id") == source_id
            )
        ):
            related.append(row)
    return related


def _failure_state(failures: Sequence[Mapping[str, Any]]) -> str:
    if any(bool(row.get("terminal")) for row in failures):
        return "terminal"
    if any(bool(row.get("retryable")) for row in failures):
        return "retryable"
    if failures:
        return "recorded"
    return "none"


def _queue_state(
    group: Mapping[str, Any],
    sft_rows: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
) -> str:
    winners = {str(value) for value in _sequence(group.get("winner_ids")) if value}
    if not winners:
        return "none"
    completed = {str(row.get("candidate_id")) for row in sft_rows if row.get("candidate_id")}
    if winners.issubset(completed):
        return "complete"
    failed = {
        str(row.get("candidate_id"))
        for row in failures
        if row.get("stage") == "annotation" and row.get("terminal") and row.get("candidate_id")
    }
    if winners & failed:
        return "failed"
    return "pending"


def _summary(
    group: Mapping[str, Any],
    sft_rows: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    candidates = _candidate_rows(group)
    winner_ids = _sequence(group.get("winner_ids"))
    timestamps = _mapping(group.get("stage_timestamps"))
    return {
        "group_id": group.get("group_id"),
        "build_id": group.get("build_id"),
        "source_id": group.get("source_id"),
        "source_path": group.get("source_path"),
        "scene": group.get("scene"),
        "render_mode": group.get("render_mode"),
        "preset_filter": group.get("preset_filter"),
        "major": group.get("major") or _mapping(group.get("coverage")).get("major"),
        "winner_ids": winner_ids,
        "winner_count": len(winner_ids),
        "candidate_count": len(candidates),
        "sft_count": len(sft_rows),
        "queue_state": _queue_state(group, sft_rows, failures),
        "failure_state": _failure_state(failures),
        "failure_count": len(failures),
        "completed_at": timestamps.get("qa_completed_at") or timestamps.get("render_completed_at"),
        "candidates": [
            {
                "candidate_id": row.get("candidate_id"),
                "slot_index": row.get("slot_index"),
                "after_path": row.get("after_path"),
                "format": row.get("format"),
                "winner": bool(row.get("winner")),
                "rank": row.get("rank"),
            }
            for row in candidates
        ],
    }


def _candidate_matches(candidate: Mapping[str, Any], filters: GroupFilters) -> bool:
    if filters.preset_format and candidate.get("format") != filters.preset_format:
        return False
    if filters.major and candidate.get("major") != filters.major:
        return False
    if filters.minor and candidate.get("minor") != filters.minor:
        return False
    if filters.winner == "yes" and not candidate.get("winner"):
        return False
    if filters.winner == "no" and candidate.get("winner"):
        return False
    if filters.winner == "top1" and not (candidate.get("winner") and candidate.get("rank") == 1):
        return False
    if filters.winner == "top2" and not (candidate.get("winner") and candidate.get("rank") == 2):
        return False
    return True


def _group_matches(
    group: Mapping[str, Any],
    sft_rows: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    filters: GroupFilters,
) -> bool:
    if filters.build_id and group.get("build_id") != filters.build_id:
        return False
    if filters.mode and group.get("render_mode") != filters.mode:
        return False
    if filters.queue_state and _queue_state(group, sft_rows, failures) != filters.queue_state:
        return False
    if filters.failure_state and _failure_state(failures) != filters.failure_state:
        return False
    candidates = _candidate_rows(group)
    has_candidate_filter = bool(
        filters.preset_format or filters.major or filters.minor or filters.winner != "any"
    )
    return not has_candidate_filter or any(_candidate_matches(row, filters) for row in candidates)


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    if not path.is_file():
        return [], 0
    records: list[dict[str, Any]] = []
    malformed = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except (TypeError, ValueError):
                malformed += 1
                continue
            if isinstance(value, Mapping):
                records.append(dict(value))
            else:
                malformed += 1
    return records, malformed


@dataclass(slots=True)
class _BuildData:
    directory: Path
    manifest: dict[str, Any]
    groups: dict[str, dict[str, Any]]
    sft_by_group: dict[str, list[dict[str, Any]]]
    failures: list[dict[str, Any]]
    malformed_records: int


class JsonlStore:
    """Small/local read-only fallback over canonical artifact directories."""

    def __init__(self, roots: Sequence[str | os.PathLike[str]]) -> None:
        self.roots = tuple(Path(root).expanduser().resolve() for root in roots)
        self._lock = threading.RLock()
        self._signature: tuple[tuple[str, int, int], ...] | None = None
        self._builds: dict[str, _BuildData] = {}

    def _directories(self) -> list[Path]:
        found: dict[str, Path] = {}
        for root in self.roots:
            if (root / "manifest.json").is_file():
                found[str(root)] = root
            if not root.is_dir():
                continue
            try:
                children = tuple(root.iterdir())
            except OSError:
                continue
            for child in children:
                if child.is_dir() and (child / "manifest.json").is_file():
                    found[str(child.resolve())] = child.resolve()
        return sorted(found.values(), key=lambda value: str(value))

    def _current_signature(self, directories: Sequence[Path]) -> tuple[tuple[str, int, int], ...]:
        values: list[tuple[str, int, int]] = []
        for directory in directories:
            for name in ("manifest.json", "groups.jsonl", "sft.jsonl", "failures.jsonl"):
                path = directory / name
                try:
                    stat = path.stat()
                except OSError:
                    values.append((str(path), -1, -1))
                else:
                    values.append((str(path), stat.st_mtime_ns, stat.st_size))
        return tuple(values)

    def _refresh(self) -> None:
        directories = self._directories()
        signature = self._current_signature(directories)
        with self._lock:
            if signature == self._signature:
                return
            builds: dict[str, _BuildData] = {}
            for directory in directories:
                try:
                    manifest_value = json.loads((directory / "manifest.json").read_text("utf-8"))
                except (OSError, TypeError, ValueError):
                    continue
                manifest = _mapping(manifest_value)
                build_id = str(manifest.get("build_id") or directory.name)
                group_rows, group_bad = _read_jsonl(directory / "groups.jsonl")
                sft_rows, sft_bad = _read_jsonl(directory / "sft.jsonl")
                failures, failure_bad = _read_jsonl(directory / "failures.jsonl")
                groups: dict[str, dict[str, Any]] = {}
                for row in group_rows:
                    group_id = row.get("group_id")
                    if (
                        not isinstance(group_id, str)
                        or not group_id
                        or row.get("build_id") != build_id
                        or canonical_group_error(row) is not None
                        or group_id in groups
                    ):
                        group_bad += 1
                        continue
                    groups[group_id] = row
                sft_by_group: dict[str, list[dict[str, Any]]] = {}
                seen_sft: dict[str, dict[str, Any]] = {}
                for row in sft_rows:
                    if row.get("sft_id") and row.get("build_id") == build_id:
                        seen_sft[str(row["sft_id"])] = row
                for row in seen_sft.values():
                    if row.get("group_id"):
                        sft_by_group.setdefault(str(row["group_id"]), []).append(row)
                for rows in sft_by_group.values():
                    rows.sort(key=lambda row: (int(row.get("winner_rank") or 0), str(row.get("sft_id"))))
                seen_failures: dict[str, dict[str, Any]] = {}
                for index, row in enumerate(failures):
                    if row.get("build_id") != build_id:
                        continue
                    event_id = str(row.get("event_id") or f"missing-event-{index}")
                    seen_failures[event_id] = row
                builds[build_id] = _BuildData(
                    directory=directory,
                    manifest=manifest,
                    groups=groups,
                    sft_by_group=sft_by_group,
                    failures=list(seen_failures.values()),
                    malformed_records=group_bad + sft_bad + failure_bad,
                )
            self._builds = builds
            self._signature = signature

    def builds(self) -> list[dict[str, Any]]:
        self._refresh()
        rows: list[dict[str, Any]] = []
        for build_id, data in self._builds.items():
            manifest = data.manifest
            rows.append({
                "build_id": build_id,
                "schema_version": manifest.get("schema_version"),
                "phase": manifest.get("phase"),
                "status": manifest.get("status"),
                "updated_at": manifest.get("updated_at") or manifest.get("ended_at"),
                "targets": _mapping(manifest.get("targets")),
                "completed": _mapping(manifest.get("completed")),
                "malformed_records": data.malformed_records,
                "artifact_root": str(data.directory),
            })
        rows.sort(key=lambda row: (str(row.get("updated_at") or ""), str(row["build_id"])), reverse=True)
        return rows

    def _group_context(
        self, data: _BuildData, group: Mapping[str, Any]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        sft = list(data.sft_by_group.get(str(group.get("group_id")), ()))
        failures = _related_failures(group, data.failures)
        return sft, failures

    def groups(
        self, filters: GroupFilters, page: int, page_size: int
    ) -> dict[str, Any]:
        self._refresh()
        matches: list[dict[str, Any]] = []
        for build_id, data in self._builds.items():
            if filters.build_id and build_id != filters.build_id:
                continue
            for group in reversed(tuple(data.groups.values())):
                sft, failures = self._group_context(data, group)
                if _group_matches(group, sft, failures, filters):
                    matches.append(_summary(group, sft, failures))
        matches.sort(
            key=lambda row: (str(row.get("completed_at") or ""), str(row.get("group_id") or "")),
            reverse=True,
        )
        offset = (page - 1) * page_size
        return {
            "total": len(matches),
            "page": page,
            "page_size": page_size,
            "items": matches[offset:offset + page_size],
        }

    def group(self, group_id: str) -> dict[str, Any] | None:
        self._refresh()
        for data in self._builds.values():
            group = data.groups.get(group_id)
            if group is None:
                continue
            sft, failures = self._group_context(data, group)
            normalized = dict(group)
            normalized["queue_state"] = _queue_state(group, sft, failures)
            normalized["failure_state"] = _failure_state(failures)
            return {
                "group": normalized,
                "candidates": _candidate_rows(group),
                "sft": sft,
                "failures": failures,
            }
        return None

    def facets(self, build_id: str | None = None) -> dict[str, Any]:
        self._refresh()
        modes: set[str] = set()
        formats: set[str] = set()
        majors: set[str] = set()
        minors: set[str] = set()
        for current_id, data in self._builds.items():
            if build_id and current_id != build_id:
                continue
            for group in data.groups.values():
                if group.get("render_mode"):
                    modes.add(str(group["render_mode"]))
                for candidate in _candidate_rows(group):
                    if candidate.get("format"):
                        formats.add(str(candidate["format"]))
                    if candidate.get("major"):
                        majors.add(str(candidate["major"]))
                    if candidate.get("minor"):
                        minors.add(str(candidate["minor"]))
        return {
            "builds": [row["build_id"] for row in self.builds()],
            "modes": sorted(modes),
            "formats": sorted(formats),
            "majors": sorted(majors),
            "minors": sorted(minors),
            "queue_states": list(QUEUE_STATES),
            "failure_states": list(FAILURE_STATES),
            "winner_filters": list(WINNER_FILTERS),
        }

    def health(self) -> dict[str, Any]:
        builds = self.builds()
        return {
            "ok": True,
            "source": "jsonl",
            "builds": len(builds),
            "groups": sum(len(data.groups) for data in self._builds.values()),
            "malformed_records": sum(data.malformed_records for data in self._builds.values()),
            "roots": [str(root) for root in self.roots],
        }


class PostgresStore:
    """Paged reader for the rebuildable canonical PostgreSQL projection."""

    def __init__(self, dsn: str, connect_fn: Callable[[str], Any] | None = None) -> None:
        self._dsn = dsn
        self._connect_fn = connect_fn

    def _connect(self) -> Any:
        if self._connect_fn is not None:
            try:
                return self._connect_fn(self._dsn)
            except Exception as exc:
                raise StoreUnavailable("postgres_unavailable") from exc
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise StoreUnavailable("postgres_driver_unavailable") from exc
        try:
            return psycopg.connect(self._dsn, row_factory=dict_row)
        except Exception as exc:  # Do not expose a credential-bearing DSN or server message.
            raise StoreUnavailable("postgres_unavailable") from exc

    @staticmethod
    def _rows(cursor: Any) -> list[dict[str, Any]]:
        return [dict(row) for row in cursor.fetchall()]

    def builds(self) -> list[dict[str, Any]]:
        connection = self._connect()
        try:
            rows = self._rows(connection.execute(
                "SELECT build_id,schema_version,phase,status,manifest,projected_at "
                "FROM canonical_builds ORDER BY projected_at DESC,build_id"
            ))
        except Exception as exc:
            raise StoreUnavailable("postgres_query_failed") from exc
        finally:
            connection.close()
        return [{
            "build_id": row.get("build_id"),
            "schema_version": row.get("schema_version"),
            "phase": row.get("phase"),
            "status": row.get("status"),
            "updated_at": str(row.get("projected_at") or ""),
            "targets": _mapping(_mapping(row.get("manifest")).get("targets")),
            "completed": _mapping(_mapping(row.get("manifest")).get("completed")),
            "malformed_records": 0,
            "artifact_root": None,
        } for row in rows]

    @staticmethod
    def _where(filters: GroupFilters) -> tuple[str, list[Any]]:
        clauses = [_CANONICAL_GROUP_SQL]
        args: list[Any] = []
        if filters.build_id:
            clauses.append("g.build_id=%s")
            args.append(filters.build_id)
        if filters.mode:
            clauses.append("g.render_mode=%s")
            args.append(filters.mode)
        candidate_clauses: list[str] = []
        if filters.preset_format:
            candidate_clauses.append("fc.preset_format=%s")
            args.append(filters.preset_format)
        if filters.major:
            candidate_clauses.append("fc.major=%s")
            args.append(filters.major)
        if filters.minor:
            candidate_clauses.append("fc.minor=%s")
            args.append(filters.minor)
        if filters.winner == "yes":
            candidate_clauses.append("fc.winner=TRUE")
        elif filters.winner == "no":
            candidate_clauses.append("fc.winner=FALSE")
        elif filters.winner == "top1":
            candidate_clauses.extend(("fc.winner=TRUE", "fc.rank=1"))
        elif filters.winner == "top2":
            candidate_clauses.extend(("fc.winner=TRUE", "fc.rank=2"))
        if candidate_clauses:
            clauses.append(
                "EXISTS (SELECT 1 FROM canonical_candidates fc WHERE fc.group_id=g.group_id AND "
                + " AND ".join(candidate_clauses) + ")"
            )
        related_failure = (
            "(ff.group_id=g.group_id OR (ff.candidate_id IS NOT NULL AND EXISTS "
            "(SELECT 1 FROM canonical_candidates rc WHERE rc.group_id=g.group_id "
            "AND rc.candidate_id=ff.candidate_id)) OR (ff.group_id IS NULL AND "
            "ff.candidate_id IS NULL AND ff.source_id=g.source_id))"
        )
        missing_winner = (
            f"EXISTS (SELECT 1 FROM jsonb_array_elements_text({_SAFE_WINNER_IDS_SQL}) "
            "AS winner(candidate_id) WHERE NOT EXISTS (SELECT 1 FROM canonical_sft qs "
            "WHERE qs.group_id=g.group_id AND qs.candidate_id=winner.candidate_id))"
        )
        failed_winner = (
            "EXISTS (SELECT 1 FROM canonical_failures ff WHERE ff.stage='annotation' "
            "AND ff.terminal=TRUE AND ff.candidate_id IN "
            f"(SELECT jsonb_array_elements_text({_SAFE_WINNER_IDS_SQL})))"
        )
        if filters.queue_state == "complete":
            clauses.append(
                f"jsonb_array_length({_SAFE_WINNER_IDS_SQL})>0 AND NOT " + missing_winner
            )
        elif filters.queue_state == "pending":
            clauses.append(
                f"jsonb_array_length({_SAFE_WINNER_IDS_SQL})>0 AND " + missing_winner
                + " AND NOT " + failed_winner
            )
        elif filters.queue_state == "failed":
            clauses.append(
                f"jsonb_array_length({_SAFE_WINNER_IDS_SQL})>0 AND " + missing_winner
                + " AND " + failed_winner
            )
        elif filters.queue_state == "none":
            clauses.append(f"jsonb_array_length({_SAFE_WINNER_IDS_SQL})=0")
        if filters.failure_state == "terminal":
            clauses.append(
                "EXISTS (SELECT 1 FROM canonical_failures ff WHERE " + related_failure
                + " AND ff.terminal=TRUE)"
            )
        elif filters.failure_state == "retryable":
            clauses.append(
                "EXISTS (SELECT 1 FROM canonical_failures ff WHERE " + related_failure
                + " AND ff.retryable=TRUE AND ff.terminal=FALSE)"
            )
        elif filters.failure_state == "recorded":
            clauses.append(
                "EXISTS (SELECT 1 FROM canonical_failures ff WHERE " + related_failure + ")"
            )
        elif filters.failure_state == "none":
            clauses.append(
                "NOT EXISTS (SELECT 1 FROM canonical_failures ff WHERE " + related_failure + ")"
            )
        return " AND ".join(clauses), args

    def groups(self, filters: GroupFilters, page: int, page_size: int) -> dict[str, Any]:
        where, args = self._where(filters)
        connection = self._connect()
        try:
            total = connection.execute(
                f"SELECT count(*) AS n FROM canonical_groups g WHERE {where}", args
            ).fetchone()["n"]
            rows = self._rows(connection.execute(
                f"""SELECT g.payload FROM canonical_groups g WHERE {where}
                    ORDER BY {_GROUP_ORDER_SQL} LIMIT %s OFFSET %s""",
                [*args, page_size, (page - 1) * page_size],
            ))
            groups = [_mapping(row.get("payload")) for row in rows]
            group_ids = [str(group.get("group_id")) for group in groups]
            source_ids = [str(group.get("source_id")) for group in groups]
            candidate_ids = [
                str(candidate.get("candidate_id"))
                for group in groups
                for candidate in _candidate_rows(group)
                if candidate.get("candidate_id")
            ]
            sft_by_group: dict[str, list[dict[str, Any]]] = {}
            related_failures: list[dict[str, Any]] = []
            if group_ids:
                for row in self._rows(connection.execute(
                    "SELECT group_id,payload FROM canonical_sft "
                    "WHERE group_id=ANY(%s) ORDER BY group_id,winner_rank", [group_ids]
                )):
                    sft_by_group.setdefault(str(row.get("group_id")), []).append(
                        _mapping(row.get("payload"))
                    )
                for row in self._rows(connection.execute(
                    "SELECT payload FROM canonical_failures WHERE group_id=ANY(%s) "
                    "OR candidate_id=ANY(%s) OR (group_id IS NULL AND candidate_id IS NULL "
                    "AND source_id=ANY(%s)) ORDER BY projected_at,event_id",
                    [group_ids, candidate_ids, source_ids],
                )):
                    related_failures.append(_mapping(row.get("payload")))
            items: list[dict[str, Any]] = []
            for group in groups:
                group_id = str(group.get("group_id"))
                failures = _related_failures(group, related_failures)
                items.append(_summary(group, sft_by_group.get(group_id, ()), failures))
        except StoreUnavailable:
            raise
        except Exception as exc:
            raise StoreUnavailable("postgres_query_failed") from exc
        finally:
            connection.close()
        return {"total": int(total), "page": page, "page_size": page_size, "items": items}

    def _group_with_connection(
        self, connection: Any, group_id: str, group: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        if group is None:
            row = connection.execute(
                "SELECT payload FROM canonical_groups WHERE group_id=%s", [group_id]
            ).fetchone()
            if not row:
                return {}
            group = _mapping(row["payload"])
        candidate_rows = self._rows(connection.execute(
            "SELECT payload,winner,slot_index,rank FROM canonical_candidates "
            "WHERE group_id=%s ORDER BY slot_index", [group_id]
        ))
        candidates: list[dict[str, Any]] = []
        for row in candidate_rows:
            candidate = _mapping(row.get("payload"))
            candidate["winner"] = bool(row.get("winner"))
            candidate["slot_index"] = row.get("slot_index")
            candidate["rank"] = row.get("rank")
            candidates.append(candidate)
        if canonical_group_error(group, candidates) is not None:
            return {}
        sft = [
            _mapping(row.get("payload"))
            for row in self._rows(connection.execute(
                "SELECT payload FROM canonical_sft WHERE group_id=%s ORDER BY winner_rank", [group_id]
            ))
        ]
        source_id = group.get("source_id")
        candidate_ids = [candidate["candidate_id"] for candidate in candidates]
        failure_rows = self._rows(connection.execute(
            "SELECT payload FROM canonical_failures WHERE group_id=%s OR candidate_id=ANY(%s) "
            "OR (group_id IS NULL AND candidate_id IS NULL AND source_id=%s) "
            "ORDER BY projected_at,event_id",
            [group_id, candidate_ids, source_id],
        ))
        group_with_candidates = {**group, "candidates": candidates}
        failures = _related_failures(
            group_with_candidates,
            [_mapping(row.get("payload")) for row in failure_rows],
        )
        normalized = dict(group)
        normalized["queue_state"] = _queue_state(group, sft, failures)
        normalized["failure_state"] = _failure_state(failures)
        return {"group": normalized, "candidates": candidates, "sft": sft, "failures": failures}

    def group(self, group_id: str) -> dict[str, Any] | None:
        connection = self._connect()
        try:
            detail = self._group_with_connection(connection, group_id)
        except Exception as exc:
            raise StoreUnavailable("postgres_query_failed") from exc
        finally:
            connection.close()
        return detail or None

    def facets(self, build_id: str | None = None) -> dict[str, Any]:
        connection = self._connect()
        try:
            build_args: list[Any] = []
            clauses = [_CANONICAL_GROUP_SQL]
            if build_id:
                clauses.append("g.build_id=%s")
                build_args = [build_id]
            group_where = " WHERE " + " AND ".join(clauses)
            modes = [row["value"] for row in self._rows(connection.execute(
                "SELECT DISTINCT g.render_mode AS value FROM canonical_groups g" + group_where
                + " ORDER BY value", build_args
            )) if row.get("value")]
            values = self._rows(connection.execute(
                "SELECT DISTINCT fc.preset_format,fc.major,fc.minor "
                "FROM canonical_candidates fc JOIN canonical_groups g "
                "ON g.group_id=fc.group_id" + group_where, build_args
            ))
        except Exception as exc:
            raise StoreUnavailable("postgres_query_failed") from exc
        finally:
            connection.close()
        return {
            "builds": [row["build_id"] for row in self.builds()],
            "modes": modes,
            "formats": sorted({str(row["preset_format"]) for row in values if row.get("preset_format")}),
            "majors": sorted({str(row["major"]) for row in values if row.get("major")}),
            "minors": sorted({str(row["minor"]) for row in values if row.get("minor")}),
            "queue_states": list(QUEUE_STATES),
            "failure_states": list(FAILURE_STATES),
            "winner_filters": list(WINNER_FILTERS),
        }

    def health(self) -> dict[str, Any]:
        builds = self.builds()
        connection = self._connect()
        try:
            groups = connection.execute(
                f"SELECT count(*) AS n FROM canonical_groups g WHERE {_CANONICAL_GROUP_SQL}"
            ).fetchone()["n"]
        except Exception as exc:
            raise StoreUnavailable("postgres_query_failed") from exc
        finally:
            connection.close()
        return {
            "ok": True,
            "source": "postgres",
            "builds": len(builds),
            "groups": int(groups),
        }


class ViewerRepository:
    """Use PostgreSQL when available, otherwise the authoritative JSONL artifacts."""

    def __init__(
        self,
        build_roots: Sequence[str | os.PathLike[str]],
        postgres_dsn: str | None = None,
        *,
        postgres_connect_fn: Callable[[str], Any] | None = None,
    ) -> None:
        self.jsonl = JsonlStore(build_roots)
        self.postgres = (
            PostgresStore(postgres_dsn, postgres_connect_fn) if postgres_dsn else None
        )
        self._last_source = "jsonl"
        self._last_fallback_reason: str | None = None
        self._lock = threading.Lock()

    def _call(self, name: str, *args: Any) -> Any:
        if self.postgres is not None:
            try:
                result = getattr(self.postgres, name)(*args)
            except StoreUnavailable as exc:
                with self._lock:
                    self._last_source = "jsonl"
                    self._last_fallback_reason = str(exc)
            else:
                with self._lock:
                    self._last_source = "postgres"
                    self._last_fallback_reason = None
                return result
        return getattr(self.jsonl, name)(*args)

    def builds(self) -> list[dict[str, Any]]:
        return self._call("builds")

    def groups(self, filters: GroupFilters, page: int, page_size: int) -> dict[str, Any]:
        return self._call("groups", filters, page, page_size)

    def group(self, group_id: str) -> dict[str, Any] | None:
        return self._call("group", group_id)

    def facets(self, build_id: str | None = None) -> dict[str, Any]:
        return self._call("facets", build_id)

    def health(self) -> dict[str, Any]:
        if self.postgres is not None:
            try:
                result = self.postgres.health()
            except StoreUnavailable as exc:
                with self._lock:
                    self._last_source = "jsonl"
                    self._last_fallback_reason = str(exc)
            else:
                with self._lock:
                    self._last_source = "postgres"
                    self._last_fallback_reason = None
                return result
        result = self.jsonl.health()
        with self._lock:
            reason = self._last_fallback_reason
        if reason:
            result["fallback_reason"] = reason
        return result


__all__ = [
    "FAILURE_STATES",
    "GroupFilters",
    "JsonlStore",
    "PostgresStore",
    "QUEUE_STATES",
    "StoreUnavailable",
    "ViewerRepository",
    "WINNER_FILTERS",
    "canonical_group_error",
]
