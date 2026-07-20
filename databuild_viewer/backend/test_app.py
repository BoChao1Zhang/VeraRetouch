from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi import HTTPException
from PIL import Image

from databuild_viewer.backend import app as backend
from databuild_viewer.backend.repository import (
    GroupFilters,
    JsonlStore,
    PostgresStore,
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
        backend.configure(build_roots=(self.root,), image_roots=(self.root,))

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
            self.assertIn("pc.payload=candidate.value", clause)
            self.assertIn("g.payload->'winner_ids'=g.winner_ids", clause)
        for clause in (complete, pending):
            self.assertIn("qs.candidate_id=winner.candidate_id", clause)
        for clause in (pending, failed):
            self.assertIn("ff.candidate_id IN", clause)

        terminal, _ = PostgresStore._where(GroupFilters(failure_state="terminal"))
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

    def test_postgres_failure_falls_back_without_exposing_dsn(self) -> None:
        secret_dsn = "postgresql://viewer:secret@127.0.0.1/viewer"

        def fail(_dsn: str):
            raise RuntimeError(secret_dsn)

        repo = ViewerRepository((self.root,), secret_dsn, postgres_connect_fn=fail)
        health = repo.health()
        self.assertEqual("jsonl", health["source"])
        self.assertEqual("postgres_unavailable", health["fallback_reason"])
        self.assertNotIn("secret", json.dumps(health))

    def test_api_is_read_only_and_has_no_dpo_or_review_routes(self) -> None:
        paths = {route.path for route in backend.app.routes}
        self.assertIn("/api/groups", paths)
        self.assertIn("/api/groups/{group_id}", paths)
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
        self.assertEqual(8, len(backend.api_group("global-group")["candidates"]))

    def test_image_path_commonpath_and_type_guards(self) -> None:
        self.assertEqual(self.source.resolve(), backend._safe_path(str(self.source)))
        sibling = Path(self.temp.name + "-sibling")
        sibling.mkdir()
        try:
            outside = sibling / "outside.jpg"
            Image.new("RGB", (4, 4)).save(outside)
            with self.assertRaises(HTTPException) as denied:
                backend._safe_path(str(outside))
            self.assertEqual(403, denied.exception.status_code)
            text = self.root / "not-image.txt"
            text.write_text("x", encoding="utf-8")
            with self.assertRaises(HTTPException) as unsupported:
                backend._safe_path(str(text))
            self.assertEqual(415, unsupported.exception.status_code)
        finally:
            outside.unlink()
            sibling.rmdir()

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
            def group(group_id):
                group = valid if group_id == "valid-group" else invalid
                return {"group": group, "candidates": candidates, "sft": [], "failures": []}

        with mock.patch.object(backend, "repository", StubRepository()):
            with self.assertRaisesRegex(RuntimeError, "winner_ids must reference"):
                backend.selfcheck()


if __name__ == "__main__":
    unittest.main()
