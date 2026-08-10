from __future__ import annotations

import io
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from fastapi import HTTPException
from PIL import Image

from databuild_viewer.backend import app as backend
from databuild_viewer.backend import repository as repository_module
from databuild_viewer.backend.repository import (
    GroupFilters,
    JsonlStore,
    PostgresStore,
    StoreUnavailable,
    ViewerRepository,
)


def _jsonl(path: Path, rows: list[dict], *, torn_tail: bool = False) -> None:
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    if torn_tail:
        text += '{"event_id":"torn"'
    path.write_text(text, encoding="utf-8")


class CanonicalViewerBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.build = self.root / "build-a"
        self.build.mkdir()
        (self.build / "manifest.json").write_text(json.dumps({
            "schema_version": 1,
            "build_id": "build-a",
            "phase": "complete",
            "status": "complete_with_failures",
            "updated_at": "2026-07-20T12:00:00+00:00",
            "targets": {"groups": 3, "local": 1, "global": 2},
            "completed": {"groups": 3, "sft": 2},
        }), encoding="utf-8")
        self.source = self.root / "source.jpg"
        Image.new("RGB", (32, 24), (120, 130, 140)).save(self.source)
        self.cgt = self.root / "mask.png"
        Image.new("L", (32, 24), 128).save(self.cgt)

        local_candidates = [self._candidate("local", index, "xmp" if index < 4 else "lut")
                            for index in range(8)]
        global_candidates = [self._candidate("global", index, "lrtemplate") for index in range(8)]
        no_winner_candidates = [self._candidate("none", index, "lut") for index in range(8)]
        groups = [
            self._group("local", "local", local_candidates, ["local-c0", "local-c1"]),
            self._group("global", "global", global_candidates, ["global-c0"]),
            self._group("none", "global", no_winner_candidates, []),
        ]
        _jsonl(self.build / "groups.jsonl", groups)
        _jsonl(self.build / "sft.jsonl", [
            self._sft("local", 1),
            self._sft("global", 1),
        ])
        _jsonl(self.build / "failures.jsonl", [{
            "event_id": "annotation-failed",
            "build_id": "build-a",
            "event_type": "annotation_terminal",
            "stage": "annotation",
            "group_id": "local-group",
            "candidate_id": "local-c1",
            "retryable": False,
            "terminal": True,
            "error_code": "schema_failed",
            "message": "invalid structured output",
        }], torn_tail=True)
        backend.configure(
            build_roots=(self.root,),
            image_roots=(self.root,),
            cache_root=self.root / "viewer-cache",
            catalog_path=self.root / "catalog.sqlite3",
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_postgres_queue_filters_match_winner_candidate_ids(self) -> None:
        complete, _ = PostgresStore._where(GroupFilters(queue_state="complete"))
        pending, _ = PostgresStore._where(GroupFilters(queue_state="pending"))
        failed, _ = PostgresStore._where(GroupFilters(queue_state="failed"))
        for clause in (complete, pending, failed):
            self.assertIn("CASE WHEN jsonb_typeof(g.winner_ids)='array'", clause)
            self.assertIn("CASE WHEN jsonb_typeof(g.payload->'candidates')='array'", clause)
            self.assertNotIn("jsonb_array_elements_text(g.winner_ids)", clause)
            self.assertNotIn("jsonb_array_length(g.winner_ids)", clause)
            self.assertIn("canonical_candidates cg", clause)
            self.assertIn("count(*) FROM canonical_candidates", clause)
            self.assertIn("count(DISTINCT cg.candidate_id)", clause)
            self.assertIn("cg.build_id=g.build_id", clause)
            self.assertIn("pc.payload=candidate.value", clause)
            self.assertIn("g.payload->'winner_ids'=g.winner_ids", clause)
            self.assertIn("WITH ORDINALITY", clause)
            self.assertIn("qs.build_id=g.build_id", clause)
            self.assertIn("qs.winner_rank=winner.winner_rank", clause)
        for clause in (complete, pending, failed):
            self.assertIn("qs.candidate_id=winner.candidate_id", clause)
        for clause in (pending, failed):
            self.assertIn("ff.candidate_id IN", clause)
            self.assertIn("ff.build_id=g.build_id", clause)

        terminal, _ = PostgresStore._where(GroupFilters(failure_state="terminal"))
        self.assertIn("ff.build_id=g.build_id", terminal)
        self.assertIn("rc.candidate_id=ff.candidate_id", terminal)
        self.assertIn("ff.candidate_id IS NULL AND ff.source_id=g.source_id", terminal)

    def test_postgres_queries_preserve_jsonl_order_and_canonical_facets(self) -> None:
        queries: list[tuple[str, list | None]] = []

        class Cursor:
            def __init__(self, *, one=None, rows=()):
                self.one = one
                self.rows = list(rows)

            def fetchone(self):
                return self.one

            def fetchall(self):
                return self.rows

        class Connection:
            def execute(self, sql, args=None):
                queries.append((sql, args))
                if "count(*) AS n" in sql:
                    return Cursor(one={"n": 0})
                return Cursor()

            def close(self):
                pass

        store = PostgresStore("postgresql://ignored", lambda _dsn: Connection())
        store.groups(GroupFilters(), 1, 20)
        payload_query = next(sql for sql, _args in queries if "SELECT g.payload" in sql)
        self.assertIn("g.payload #>> '{stage_timestamps,qa_completed_at}'", payload_query)
        self.assertIn("g.payload #>> '{stage_timestamps,render_completed_at}'", payload_query)
        self.assertIn("g.group_id DESC", payload_query)
        self.assertNotIn("ORDER BY g.projected_at", payload_query)

        queries.clear()
        facets = store.facets("build-a")
        self.assertEqual([], facets["modes"])
        mode_query = next(sql for sql, _args in queries if "render_mode AS value" in sql)
        candidate_query = next(sql for sql, _args in queries if "fc.preset_format" in sql)
        for sql in (mode_query, candidate_query):
            self.assertIn("jsonb_typeof(g.winner_ids)='array'", sql)
            self.assertIn("g.build_id=%s", sql)
        self.assertIn("JOIN canonical_groups g", candidate_query)

    def test_postgres_detail_scopes_candidates_and_sft_to_group_build(self) -> None:
        group = self._group(
            "global",
            "global",
            [self._candidate("global", index, "lut") for index in range(8)],
            ["global-c0"],
        )
        candidates = group["candidates"]
        sft = self._sft("global", 1)
        queries: list[tuple[str, list | None]] = []

        class Cursor:
            def __init__(self, *, one=None, rows=()):
                self.one = one
                self.rows = list(rows)

            def fetchone(self):
                return self.one

            def fetchall(self):
                return self.rows

        class Connection:
            def execute(self, sql, args=None):
                queries.append((sql, args))
                if sql.startswith("SELECT g.payload FROM canonical_groups"):
                    return Cursor(one={"payload": group})
                if sql.startswith("SELECT payload,winner,slot_index,rank"):
                    return Cursor(rows=[{
                        "payload": candidate,
                        "winner": candidate["candidate_id"] in group["winner_ids"],
                        "slot_index": index,
                        "rank": candidate["rank"],
                    } for index, candidate in enumerate(candidates)])
                if sql.startswith("SELECT payload FROM canonical_sft"):
                    return Cursor(rows=[{"payload": sft}])
                if sql.startswith("SELECT payload FROM canonical_failures"):
                    return Cursor()
                raise AssertionError(sql)

            def close(self):
                pass

        detail = PostgresStore(
            "postgresql://ignored", lambda _dsn: Connection()
        ).group("global-group", "build-a")
        self.assertIsNotNone(detail)
        assert detail is not None
        self.assertEqual("complete", detail["group"]["queue_state"])
        candidate_sql, candidate_args = next(
            (sql, args) for sql, args in queries
            if sql.startswith("SELECT payload,winner,slot_index,rank")
        )
        sft_sql, sft_args = next(
            (sql, args) for sql, args in queries
            if sql.startswith("SELECT payload FROM canonical_sft")
        )
        for sql, args in ((candidate_sql, candidate_args), (sft_sql, sft_args)):
            self.assertIn("build_id=%s", sql)
            self.assertEqual(["global-group", "build-a"], args)

    def test_postgres_health_reports_canonical_group_count(self) -> None:
        class Cursor:
            def __init__(self, *, one=None, rows=()):
                self.one = one
                self.rows = list(rows)

            def fetchone(self):
                return self.one

            def fetchall(self):
                return self.rows

        class Connection:
            def execute(self, sql, _args=None):
                if "FROM canonical_builds" in sql:
                    return Cursor(rows=[])
                if "count(*) AS n" in sql:
                    return Cursor(one={"n": 7})
                raise AssertionError(sql)

            def close(self):
                pass

        health = PostgresStore(
            "postgresql://ignored", lambda _dsn: Connection()
        ).health()
        self.assertEqual(7, health["groups"])

    def _candidate(self, prefix: str, index: int, preset_format: str) -> dict:
        candidate = {
            "candidate_id": f"{prefix}-c{index}",
            "slot_id": f"slot-{index}",
            "slot_index": index,
            "preset_id": f"preset-{prefix}-{index}",
            "format": preset_format,
            "major": "portrait" if index < 4 else "film",
            "minor": f"minor-{index % 2}",
            "after_path": str(self.source),
            "render_engine": "cuda",
            "visibility": {"visible_de": 3.1 + index, "visible_fraction": 0.7},
            "qa": {"onealign": 0.7, "q": 0.8, "reliable": True, "veto": False},
            "rank": index + 1,
        }
        if prefix == "local":
            candidate.update({
                "slot_mode": "semantic" if index in (2, 3) else "radial",
                "mask_id": "shared-semantic" if index in (2, 3) else f"mask-{index}",
                "cgt_path": str(self.cgt),
                "subject": {"name": "person"},
                "region": "center",
                "amount": 0.8,
                "effective_alpha_mean": 0.42,
            })
        return candidate

    def _group(self, prefix: str, mode: str, candidates: list[dict], winners: list[str]) -> dict:
        return {
            "schema_version": 1,
            "build_id": "build-a",
            "group_id": f"{prefix}-group",
            "source_id": f"{prefix}-source",
            "source_path": str(self.source),
            "scene": "studio",
            "subject": {"name": "person"} if mode == "local" else None,
            "render_mode": mode,
            "preset_filter": "all",
            "major": "portrait",
            "candidates": candidates,
            "winner_ids": winners,
            "winner_ranks": list(range(1, len(winners) + 1)),
            "stage_timestamps": {"qa_completed_at": f"2026-07-20T10:0{len(winners)}:00+00:00"},
        }

    def _sft(self, prefix: str, rank: int) -> dict:
        return {
            "build_id": "build-a",
            "sft_id": f"{prefix}-sft-{rank}",
            "annotation_task_id": f"{prefix}-task-{rank}",
            "group_id": f"{prefix}-group",
            "candidate_id": f"{prefix}-c{rank - 1}",
            "winner_rank": rank,
            "I_in": str(self.source),
            "I_tar": str(self.source),
            "recipe": {"preset_id": f"preset-{prefix}-{rank - 1}"},
            "local": {"C_GT": str(self.cgt)} if prefix == "local" else None,
            "task_type": "local" if prefix == "local" else "style",
            "instruction": "A deliberately long final instruction for layout verification.",
            "instruction_short": "Refine the portrait.",
            "reasoning": "<problem_lighting>Flat light</problem_lighting>",
            "annot_src": "responses:local",
            "qa": {"annotation": {"status": "completed"}},
        }

    def test_jsonl_store_returns_canonical_detail_and_facets(self) -> None:
        store = JsonlStore((self.root,))
        self.assertEqual(["build-a"], [row["build_id"] for row in store.builds()])
        detail = store.group("local-group")
        self.assertIsNotNone(detail)
        assert detail is not None
        self.assertEqual(8, len(detail["candidates"]))
        self.assertEqual(1, len(detail["sft"]))
        self.assertEqual("failed", detail["group"]["queue_state"])
        self.assertTrue(detail["candidates"][0]["winner"])
        self.assertEqual("shared-semantic", detail["candidates"][2]["mask_id"])
        facets = store.facets("build-a")
        self.assertEqual(["global", "local"], facets["modes"])
        self.assertEqual(["lrtemplate", "lut", "xmp"], facets["formats"])

    def test_jsonl_full_ledger_reads_manifests_then_only_selected_build(self) -> None:
        second = self.root / "build-b"
        second.mkdir()
        (second / "manifest.json").write_text(json.dumps({
            "build_id": "build-b", "status": "complete", "completed": {"groups": 999},
        }), encoding="utf-8")
        for name in ("groups.jsonl", "sft.jsonl", "failures.jsonl"):
            (second / name).write_text("", encoding="utf-8")
        calls: list[Path] = []
        original = repository_module._read_jsonl

        def tracked(path):
            calls.append(Path(path))
            return original(path)

        store = JsonlStore((self.root,))
        with mock.patch.object(repository_module, "_read_jsonl", side_effect=tracked):
            self.assertEqual(2, len(store.builds()))
            health = store.health()
            self.assertEqual(1002, health["groups"])
            self.assertEqual([], calls)
            with self.assertRaisesRegex(StoreUnavailable, "build_id_required"):
                store.groups(GroupFilters(), 1, 20)
            self.assertEqual(3, store.groups(GroupFilters(build_id="build-a"), 1, 20)["total"])
        self.assertTrue(calls)
        self.assertTrue(all(path.parent == self.build for path in calls))
        self.assertEqual("build-a", store._active_id)

    def test_jsonl_build_switch_is_serialized_and_releases_old_active(self) -> None:
        builds: dict[str, Path] = {}
        for build_id in ("build-b", "build-c"):
            directory = self.root / build_id
            directory.mkdir()
            (directory / "manifest.json").write_text(json.dumps({
                "build_id": build_id, "status": "complete", "completed": {"groups": 0},
            }), encoding="utf-8")
            for name in ("groups.jsonl", "sft.jsonl", "failures.jsonl"):
                (directory / name).write_text("", encoding="utf-8")
            builds[build_id] = directory

        store = JsonlStore((self.root,))
        self.assertIsNotNone(store.group("local-group", "build-a"))
        self.assertEqual("build-a", store._active_id)
        original = repository_module._read_jsonl
        parser_lock = threading.Lock()
        active_parsers = 0
        max_parsers = 0
        b_entered = threading.Event()
        b_release = threading.Event()
        c_entered = threading.Event()
        errors: list[BaseException] = []
        results: list[str] = []

        def tracked(path):
            nonlocal active_parsers, max_parsers
            path = Path(path)
            with parser_lock:
                active_parsers += 1
                max_parsers = max(max_parsers, active_parsers)
            try:
                if path.parent in builds.values():
                    self.assertIsNone(store._active)
                    self.assertIsNone(store._active_id)
                if path.parent == builds["build-b"] and path.name == "groups.jsonl":
                    b_entered.set()
                    self.assertTrue(b_release.wait(2))
                if path.parent == builds["build-c"] and path.name == "groups.jsonl":
                    c_entered.set()
                time.sleep(0.01)
                return original(path)
            finally:
                with parser_lock:
                    active_parsers -= 1

        def load(build_id: str) -> None:
            try:
                store.groups(GroupFilters(build_id=build_id), 1, 20)
                results.append(build_id)
            except BaseException as exc:
                errors.append(exc)

        with mock.patch.object(repository_module, "_read_jsonl", side_effect=tracked):
            thread_b = threading.Thread(target=load, args=("build-b",))
            thread_b.start()
            self.assertTrue(b_entered.wait(1))
            thread_c = threading.Thread(target=load, args=("build-c",))
            thread_c.start()
            self.assertFalse(c_entered.wait(0.05))
            b_release.set()
            thread_b.join(2)
            thread_c.join(2)

        self.assertFalse(thread_b.is_alive())
        self.assertFalse(thread_c.is_alive())
        self.assertEqual([], errors)
        self.assertEqual(["build-b", "build-c"], results)
        self.assertEqual(1, max_parsers)
        self.assertEqual("build-c", store._active_id)
        self.assertEqual(builds["build-c"], store._active.directory)

    def test_jsonl_load_failure_leaves_no_active_build(self) -> None:
        second = self.root / "build-b"
        second.mkdir()
        (second / "manifest.json").write_text(json.dumps({"build_id": "build-b"}), encoding="utf-8")
        for name in ("groups.jsonl", "sft.jsonl", "failures.jsonl"):
            (second / name).write_text("", encoding="utf-8")
        store = JsonlStore((self.root,))
        self.assertIsNotNone(store.group("local-group", "build-a"))
        with mock.patch.object(repository_module, "_read_jsonl", side_effect=OSError("read failed")):
            with self.assertRaisesRegex(OSError, "read failed"):
                store.groups(GroupFilters(build_id="build-b"), 1, 20)
        self.assertIsNone(store._active)
        self.assertIsNone(store._active_id)

    def test_duplicate_build_ids_fail_closed_and_recover(self) -> None:
        duplicates = []
        for name in ("duplicate-a", "duplicate-b"):
            directory = self.root / name
            directory.mkdir()
            (directory / "manifest.json").write_text(
                json.dumps({"build_id": "duplicate", "completed": {"groups": 0}}),
                encoding="utf-8",
            )
            duplicates.append(directory)
        store = JsonlStore((self.root,))
        with self.assertRaisesRegex(StoreUnavailable, "duplicate_build_id:duplicate") as failure:
            store.builds()
        self.assertNotIn(str(duplicates[0]), str(failure.exception))
        self.assertEqual({}, store._manifests)
        self.assertIsNone(store._active)
        with self.assertRaisesRegex(StoreUnavailable, "duplicate_build_id:duplicate"):
            ViewerRepository((self.root,)).health()

        (duplicates[1] / "manifest.json").write_text(
            json.dumps({"build_id": "duplicate-b", "completed": {"groups": 0}}),
            encoding="utf-8",
        )
        build_ids = [row["build_id"] for row in store.builds()]
        self.assertEqual(1, build_ids.count("duplicate"))
        self.assertIn("duplicate-b", build_ids)

        overlapping = JsonlStore((self.root, self.build))
        overlap_ids = [row["build_id"] for row in overlapping.builds()]
        self.assertEqual(1, overlap_ids.count("build-a"))

    def test_all_required_filters_are_applied(self) -> None:
        store = JsonlStore((self.root,))
        cases = (
            (GroupFilters(build_id="build-a", mode="local"), "local-group"),
            (GroupFilters(build_id="build-a", preset_format="lrtemplate"), "global-group"),
            (GroupFilters(build_id="build-a", major="film"), "local-group"),
            (GroupFilters(build_id="build-a", minor="minor-1"), "local-group"),
            (GroupFilters(build_id="build-a", queue_state="failed"), "local-group"),
            (GroupFilters(build_id="build-a", queue_state="complete"), "global-group"),
            (GroupFilters(build_id="build-a", failure_state="terminal"), "local-group"),
            (GroupFilters(build_id="build-a", winner="top2"), "local-group"),
            (GroupFilters(build_id="build-a", winner="no"), "none-group"),
        )
        for filters, expected in cases:
            with self.subTest(filters=filters):
                page = store.groups(filters, 1, 20)
                self.assertIn(expected, [row["group_id"] for row in page["items"]])

    def test_malformed_torn_tail_is_reported_without_mutation(self) -> None:
        store = JsonlStore((self.root,))
        self.assertEqual(0, store.health()["malformed_records"])
        store.group("local-group", "build-a")
        health = store.health()
        self.assertEqual(1, health["malformed_records"])
        self.assertEqual('{"event_id":"torn"', (self.build / "failures.jsonl").read_text("utf-8").splitlines()[-1])

    def test_malformed_group_is_counted_and_never_exposed(self) -> None:
        rows = [json.loads(line) for line in (self.build / "groups.jsonl").read_text("utf-8").splitlines()]
        malformed = self._group(
            "malformed",
            "global",
            [self._candidate("malformed", index, "lut") for index in range(7)],
            ["malformed-c0"],
        )
        _jsonl(self.build / "groups.jsonl", [*rows, malformed])

        store = JsonlStore((self.root,))
        self.assertIsNone(store.group("malformed-group"))
        self.assertEqual(3, store.groups(GroupFilters(build_id="build-a"), 1, 20)["total"])
        self.assertEqual(2, store.health()["malformed_records"])

    def test_nonwinner_annotation_failure_does_not_fail_queue(self) -> None:
        rows = [json.loads(line) for line in (self.build / "groups.jsonl").read_text("utf-8").splitlines()]
        pending = self._group(
            "pending",
            "global",
            [self._candidate("pending", index, "lut") for index in range(8)],
            ["pending-c0"],
        )
        _jsonl(self.build / "groups.jsonl", [*rows, pending])
        _jsonl(self.build / "failures.jsonl", [{
            "event_id": "pending-nonwinner-failed",
            "build_id": "build-a",
            "stage": "annotation",
            "group_id": "pending-group",
            "candidate_id": "pending-c7",
            "retryable": False,
            "terminal": True,
            "error_code": "schema_failed",
        }])

        detail = JsonlStore((self.root,)).group("pending-group")
        self.assertIsNotNone(detail)
        assert detail is not None
        self.assertEqual("pending", detail["group"]["queue_state"])
        self.assertEqual("terminal", detail["group"]["failure_state"])

    def test_invalid_extra_sft_cannot_mark_queue_complete(self) -> None:
        group_rows = [
            json.loads(line)
            for line in (self.build / "groups.jsonl").read_text("utf-8").splitlines()
        ]
        sft_rows = [
            json.loads(line)
            for line in (self.build / "sft.jsonl").read_text("utf-8").splitlines()
        ]
        group = self._group(
            "badqueue",
            "global",
            [self._candidate("badqueue", index, "lut") for index in range(8)],
            ["badqueue-c0"],
        )
        valid = self._sft("badqueue", 1)
        extra = {
            **valid,
            "sft_id": "badqueue-sft-extra",
            "annotation_task_id": "badqueue-task-extra",
            "candidate_id": "badqueue-c7",
        }
        _jsonl(self.build / "groups.jsonl", [*group_rows, group])
        _jsonl(self.build / "sft.jsonl", [*sft_rows, valid, extra])

        detail = JsonlStore((self.root,)).group("badqueue-group")
        self.assertIsNotNone(detail)
        assert detail is not None
        self.assertEqual("pending", detail["group"]["queue_state"])

    def test_postgres_failure_falls_back_without_exposing_dsn(self) -> None:
        secret_dsn = "postgresql://viewer:secret@127.0.0.1/viewer"

        def fail(_dsn: str):
            raise RuntimeError(secret_dsn)

        repo = ViewerRepository((self.root,), secret_dsn, postgres_connect_fn=fail)
        health = repo.health()
        self.assertEqual("jsonl", health["source"])
        self.assertEqual("postgres_unavailable", health["fallback_reason"])
        self.assertNotIn("secret", json.dumps(health))

    def test_api_has_inspection_and_bounded_prepare_routes_but_no_review_mutation(self) -> None:
        paths = {route.path for route in backend.app.routes}
        self.assertIn("/api/groups", paths)
        self.assertIn("/api/groups/{group_id}", paths)
        self.assertIn("/api/groups/{group_id}/prepare", paths)
        self.assertNotIn("/api/review", paths)
        self.assertFalse(any("dpo" in path.lower() for path in paths))
        self.assertEqual(3, backend.api_groups(
            build_id="build-a",
            mode=None,
            preset_format=None,
            major=None,
            minor=None,
            queue_state=None,
            failure_state=None,
            winner="any",
            page=1,
            page_size=50,
        )["total"])
        self.assertEqual(8, len(backend.api_group("global-group", "build-a")["candidates"]))

    def test_image_path_commonpath_and_type_guards(self) -> None:
        self.assertEqual(self.source.resolve(), backend._safe_path(str(self.source)))
        retired = self.root / "retired" / "candidate.jpg"
        self.assertEqual(retired.resolve(), backend._authorized_image_path(str(retired)))
        sibling = Path(self.temp.name + "-sibling")
        sibling.mkdir()
        try:
            outside = sibling / "outside.jpg"
            Image.new("RGB", (4, 4)).save(outside)
            with self.assertRaises(HTTPException) as denied:
                backend._safe_path(str(outside))
            self.assertEqual(403, denied.exception.status_code)
            with self.assertRaises(HTTPException) as traversal:
                backend._authorized_image_path(str(self.root / ".." / sibling.name / "outside.jpg"))
            self.assertEqual(403, traversal.exception.status_code)
            text = self.root / "not-image.txt"
            text.write_text("x", encoding="utf-8")
            with self.assertRaises(HTTPException) as unsupported:
                backend._safe_path(str(text))
            self.assertEqual(415, unsupported.exception.status_code)
            rejected = (
                "~/data/image.jpg",
                "relative/image.jpg",
                str(self.root / "folder" / ".." / "source.jpg"),
                "//home/bc/data/image.jpg",
            )
            for value in rejected:
                with self.subTest(value=value), self.assertRaises(HTTPException) as invalid:
                    backend._authorized_image_path(value)
                self.assertEqual(403, invalid.exception.status_code)
            with self.assertRaises(HTTPException) as nul:
                backend._authorized_image_path(str(self.root / "bad.jpg") + "\x00")
            self.assertEqual(400, nul.exception.status_code)
        finally:
            outside.unlink()
            sibling.rmdir()

    def test_startup_matrix_uses_durable_nfs_default_and_explicit_overrides(self) -> None:
        captures: list[dict] = []
        config = self.root / "viewer.toml"
        explicit = self.root / "explicit-build"
        explicit.mkdir()
        env_root = self.root / "env-ledger"
        env_root.mkdir()

        def capture(**kwargs):
            captures.append(kwargs)

        cases = (
            ([], (backend.DEFAULT_NFS_LEDGER_ROOT,), None),
            (["--config", str(config)], (backend.DEFAULT_NFS_LEDGER_ROOT,), "postgresql://fixture"),
            (["--config", str(config), "--build-root", str(explicit)], (explicit.resolve(),), "postgresql://fixture"),
            (["--nfs-ledger", str(explicit)], (explicit.resolve(),), None),
        )
        with mock.patch.object(
            backend, "_load_databuild_config", return_value=(Path("/mnt/ramstage/retired"), "postgresql://fixture")
        ), mock.patch.object(backend, "configure", side_effect=capture), mock.patch.object(
            backend, "selfcheck", return_value={
                "source": "jsonl", "builds": 0, "groups": 0, "malformed_records": 0,
            }
        ):
            for argv, expected_roots, expected_dsn in cases:
                self.assertEqual(0, backend.main([*argv, "--selfcheck"]))
                self.assertEqual(expected_roots, tuple(captures[-1]["build_roots"]))
                self.assertEqual(expected_dsn, captures[-1]["postgres_dsn"])
            with mock.patch.object(backend, "_build_roots", (env_root.resolve(),)):
                self.assertEqual(0, backend.main(["--selfcheck"]))
                self.assertEqual((env_root.resolve(),), tuple(captures[-1]["build_roots"]))

    def test_prepare_api_deduplicates_assets_and_returns_versioned_status(self) -> None:
        calls: list[tuple[str, tuple[str, ...], bool]] = []

        class StubMaterializer:
            db_path = self.root / "catalog.sqlite3"

            @staticmethod
            def cache_status():
                return {"root": "/tmp/cache", "bytes": 0, "max_bytes": 1000}

            @staticmethod
            def prepare(group_id, paths, *, build_id, retry=False):
                calls.append((group_id, build_id, tuple(paths), retry))
                return {
                    "schema_version": 1,
                    "group_id": group_id,
                    "state": "queued",
                    "files": {"done": 0, "total": len(paths)},
                    "bytes": {"done": 0, "total": 0},
                    "current_item": None,
                    "message": None,
                    "updated_at": "now",
                }

            @staticmethod
            def status(group_id, *, build_id):
                return {
                    "schema_version": 1,
                    "group_id": group_id,
                    "state": "materializing",
                    "files": {"done": 2, "total": 4},
                    "bytes": {"done": 20, "total": 40},
                    "current_item": str(self.source),
                    "message": None,
                    "updated_at": "now",
                }

            @staticmethod
            def read_cached(_path):
                return None

        with mock.patch.object(backend, "materializer", StubMaterializer()):
            queued = backend.api_prepare_group("local-group", "build-a")
            status = backend.api_prepare_status("local-group", "build-a")
        self.assertEqual(1, queued["schema_version"])
        self.assertEqual("materializing", status["state"])
        self.assertEqual(1, len(calls))
        group_id, build_id, paths, retry = calls[0]
        self.assertEqual("local-group", group_id)
        self.assertEqual("build-a", build_id)
        self.assertFalse(retry)
        self.assertEqual(len(paths), len(set(paths)))
        self.assertIn(str(self.source), paths)
        self.assertIn(str(self.cgt), paths)

    def test_prepare_and_image_share_canonical_in_root_alias_key(self) -> None:
        inside = self.root / "inside"
        inside.mkdir()
        alias = self.root / "alias"
        alias.symlink_to(inside, target_is_directory=True)
        aliased_path = alias / "archived.png"
        canonical_path = inside / "archived.png"
        buffer = io.BytesIO()
        Image.new("RGB", (9, 7), (12, 34, 56)).save(buffer, format="PNG")
        payload = buffer.getvalue()
        prepared: list[tuple[str, ...]] = []
        reads: list[str] = []

        class StubRepository:
            @staticmethod
            def group(group_id, build_id):
                self.assertEqual(("alias-group", "build-a"), (group_id, build_id))
                return {"group": {"source_path": str(aliased_path)}, "candidates": [], "sft": []}

        class StubMaterializer:
            @staticmethod
            def prepare(_group_id, paths, *, build_id, retry=False):
                self.assertEqual("build-a", build_id)
                self.assertFalse(retry)
                prepared.append(tuple(paths))
                return {"schema_version": 1, "state": "queued"}

            @staticmethod
            def read_cached(path):
                reads.append(path)
                return payload

        with mock.patch.object(backend, "repository", StubRepository()), mock.patch.object(
            backend, "materializer", StubMaterializer()
        ):
            backend.api_prepare_group("alias-group", "build-a")
            full = backend.image(str(aliased_path), full=True)
            thumb = backend.image(str(aliased_path), w=64)

        canonical = str(canonical_path.resolve(strict=False))
        self.assertEqual([(canonical,)], prepared)
        self.assertEqual([canonical, canonical], reads)
        self.assertEqual("image/png", full.media_type)
        self.assertEqual("image/jpeg", thumb.media_type)
        self.assertGreater(len(thumb.body), 0)

    def test_archived_image_is_cache_only_after_authorization_and_decodes(self) -> None:
        archived = self.root / "retired" / "candidate.png"
        buffer = io.BytesIO()
        Image.new("RGB", (9, 7), (12, 34, 56)).save(buffer, format="PNG")
        payload = buffer.getvalue()
        with mock.patch.object(backend.materializer, "read_cached", return_value=None):
            with self.assertRaises(HTTPException) as pending:
                backend.image(str(archived), w=64)
        self.assertEqual(409, pending.exception.status_code)

        with mock.patch.object(backend.materializer, "read_cached", return_value=payload):
            response = backend.image(str(archived), w=64)
        self.assertEqual("image/jpeg", response.media_type)
        self.assertGreater(len(response.body), 0)

        outside = Path(self.temp.name + "-outside") / "secret.png"
        with mock.patch.object(backend.materializer, "read_cached") as denied_read:
            with self.assertRaises(HTTPException) as denied:
                backend.image(str(outside))
        self.assertEqual(403, denied.exception.status_code)
        denied_read.assert_not_called()

    def test_selfcheck_is_read_only_and_accepts_empty_store(self) -> None:
        result = backend.selfcheck()
        self.assertTrue(result["ok"])
        self.assertEqual(1, result["builds"])
        empty = self.root / "empty"
        empty.mkdir()
        backend.configure(build_roots=(empty,), image_roots=(empty,))
        self.assertEqual(0, backend.selfcheck()["builds"])

    def test_selfcheck_validates_winners_for_every_exposed_group(self) -> None:
        candidates = [self._candidate("invalid", index, "lut") for index in range(8)]
        valid = self._group("valid", "global", candidates, ["invalid-c0"])
        invalid = {**valid, "group_id": "invalid-group", "winner_ids": ["missing"]}

        class StubRepository:
            @staticmethod
            def health():
                return {"ok": True, "source": "stub", "groups": 2}

            @staticmethod
            def builds():
                return [{"build_id": "build-a"}]

            @staticmethod
            def groups(_filters, _page, _page_size):
                return {
                    "total": 2,
                    "items": [{"group_id": "valid-group"}, {"group_id": "invalid-group"}],
                }

            @staticmethod
            def group(group_id, build_id):
                self.assertEqual("build-a", build_id)
                group = valid if group_id == "valid-group" else invalid
                return {"group": group, "candidates": candidates, "sft": [], "failures": []}

        with mock.patch.object(backend, "repository", StubRepository()):
            with self.assertRaisesRegex(RuntimeError, "winner_ids must reference"):
                backend.selfcheck()

    def test_sft_invariant_requires_distinct_one_based_winner_mapping(self) -> None:
        group = {"build_id": "build", "group_id": "group", "winner_ids": ["c0", "c1"]}
        valid = [
            {"build_id": "build", "group_id": "group", "candidate_id": "c0", "winner_rank": 1},
            {"build_id": "build", "group_id": "group", "candidate_id": "c1", "winner_rank": 2},
        ]
        self.assertIsNone(backend.canonical_sft_error(group, valid))
        invalid = (
            ([*valid, {"candidate_id": "c2", "winner_rank": 3}], "more than two"),
            ([{**valid[0], "winner_rank": True}], "one-based"),
            ([{**valid[0], "winner_rank": 2}], "winner_ids order"),
            ([{**valid[0], "build_id": "other"}], "build_id does not match"),
            ([{**valid[0], "group_id": "other"}], "group_id does not match"),
            ([
                valid[0],
                {**valid[1], "candidate_id": "c0"},
            ], "candidate_ids must be distinct"),
            ([
                valid[0],
                {**valid[1], "winner_rank": 1},
            ], "winner_ranks must be distinct"),
        )
        for rows, message in invalid:
            with self.subTest(message=message):
                self.assertIn(message, backend.canonical_sft_error(group, rows) or "")

    def test_selfcheck_reaches_later_pages_and_validates_sft(self) -> None:
        candidates = [self._candidate("page", index, "lut") for index in range(8)]
        group = self._group("page", "global", candidates, ["page-c0"])
        calls: list[int] = []

        class StubRepository:
            @staticmethod
            def health():
                return {"ok": True, "source": "stub", "groups": 201}

            @staticmethod
            def builds():
                return [{"build_id": "build-a"}]

            @staticmethod
            def groups(_filters, page, _page_size):
                calls.append(page)
                if page == 1:
                    return {
                        "total": 201,
                        "items": [{"group_id": f"page-{index}"} for index in range(200)],
                    }
                return {"total": 201, "items": [{"group_id": "page-late"}]}

            @staticmethod
            def group(group_id, build_id):
                self.assertEqual("build-a", build_id)
                sft = [] if group_id != "page-late" else [
                    {
                        "build_id": "build-a",
                        "group_id": group_id,
                        "candidate_id": "page-c0",
                        "winner_rank": 2,
                    }
                ]
                return {
                    "group": {**group, "group_id": group_id},
                    "candidates": candidates,
                    "sft": sft,
                    "failures": [],
                }

        with mock.patch.object(backend, "repository", StubRepository()):
            with self.assertRaisesRegex(RuntimeError, "valid one-based winner rank"):
                backend.selfcheck()
        self.assertEqual([1, 2], calls)


if __name__ == "__main__":
    unittest.main()
