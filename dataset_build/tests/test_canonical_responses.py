from __future__ import annotations

import asyncio
import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from PIL import Image

from construct.config import (
    AnnotationConfig,
    ExternalEndpointConfig,
    LocalAnnotationConfig,
)
from construct.responses import (
    ANNOTATION_FIELDS,
    ANNOTATION_JSON_SCHEMA,
    AnnotationError,
    ExternalRelayPool,
    ResponsesAnnotator,
    _consume_stream,
    preflight_openai_sdk,
    prepare_task,
    request_payload,
)
from construct.state import ArtifactStore, stable_id
from dataset_build.core.responses_vlm import consume_text
from dataset_build.source_qa.caption_subjects import (
    _SCHEMA as CAPTION_SUBJECTS_SCHEMA,
    _one as caption_one,
    _parse as parse_caption_subjects,
)
from dataset_build.tools.eval_subject_instance_selector import (
    SUBJECT_LABEL_SCHEMA,
    SUBJECT_SELECTION_SCHEMA,
    _call_selector,
    _call_subject_label,
    _parse_selection,
    _parse_subject_label,
)


def valid_fields() -> dict[str, str]:
    return {
        "problem_lighting": "The lighting looks too flat.",
        "plan_lighting": "Build clearer tonal separation.",
        "problem_global_color": "The overall color feels muted.",
        "plan_global_color": "Restore a balanced overall palette.",
        "problem_specific_color": "The main subject lacks color definition.",
        "plan_specific_color": "Refine the subject colors selectively.",
        "instruction_long": "Apply the Style Name look with balanced tone and color.",
        "instruction_short": "Apply the Style Name look.",
    }


def completed_events(fields: dict[str, str] | None = None, *, model: str = "returned-model"):
    from openai.types.responses import ResponseCompletedEvent, ResponseTextDeltaEvent

    raw = json.dumps(fields or valid_fields())
    response = SimpleNamespace(
        status="completed",
        model=model,
        usage={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
    )
    return [
        ResponseTextDeltaEvent.model_construct(
            content_index=0, delta=raw[: len(raw) // 2], item_id="msg",
            logprobs=[], output_index=0, sequence_number=1,
            type="response.output_text.delta",
        ),
        ResponseTextDeltaEvent.model_construct(
            content_index=0, delta=raw[len(raw) // 2 :], item_id="msg",
            logprobs=[], output_index=0, sequence_number=2,
            type="response.output_text.delta",
        ),
        ResponseCompletedEvent.model_construct(
            response=response, sequence_number=3, type="response.completed"
        ),
    ]


class FakeStream:
    def __init__(self, events):
        self.events = events
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.closed = True

    def __iter__(self):
        return iter(self.events)


class FakeHttpError(Exception):
    def __init__(self, status_code: int, code: str, *, retry_after: str | None = None):
        super().__init__(f"HTTP {status_code}: {code}")
        self.status_code = status_code
        self.body = {"error": {"code": code}}
        self.response = SimpleNamespace(
            headers={} if retry_after is None else {"Retry-After": retry_after}
        )


class FakeResponses:
    def __init__(self, actions, *, before_call=None):
        self.actions = list(actions)
        self.calls: list[dict[str, Any]] = []
        self.before_call = before_call

    def create(self, **kwargs):
        if self.before_call:
            self.before_call()
        self.calls.append(kwargs)
        if not self.actions:
            raise AssertionError("unexpected Responses.create call")
        action = self.actions.pop(0)
        if isinstance(action, BaseException):
            raise action
        return FakeStream(action)


class FakeClient:
    def __init__(self, responses):
        self.responses = responses


class ResponseFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.before = self.root / "before.jpg"
        self.after = self.root / "after.jpg"
        Image.new("RGB", (1200, 600), (240, 20, 20)).save(self.before)
        Image.new("RGB", (600, 1200), (20, 220, 20)).save(self.after)
        self.config = AnnotationConfig(
            external_model="external-model",
            image_long_edge=768,
            image_jpeg_quality=90,
            external_reasoning_effort="medium",
            external_max_output_tokens=6000,
            transport_attempts_per_round=4,
            queue_rounds=3,
            external_endpoints=(
                ExternalEndpointConfig("relay-a", "https://a.example/v1", "key-a", 2),
                ExternalEndpointConfig("relay-b", "https://b.example/v1", "key-b", 2),
            ),
            local=LocalAnnotationConfig(
                "http://127.0.0.1:8003/v1", "EMPTY", "qwen3_5-35b-a3b",
                0.2, False, 2048,
            ),
        )

    def tearDown(self):
        self.tmp.cleanup()

    def task(self, *, source: str = "source-1", mode: str = "global") -> dict[str, Any]:
        group_id = stable_id("group", "build", source, mode, 0)
        candidates = []
        for index in range(8):
            candidates.append({
                "candidate_id": stable_id("candidate", group_id, index),
                "preset_id": f"preset-{index}",
                "style_name": "Style Name",
                "after_path": str(self.after),
                "objective_hints": {
                    "brightness": "brighter",
                    "warmth": "warmer",
                    "chroma": "richer",
                    "contrast": "higher",
                },
                "qa": {"q": 0.8, "rank": index + 1},
                "mask_id": f"mask-{index}",
                "cgt_path": str(self.root / f"cgt-{index}.png"),
                "subject": {"name": "person", "instance_id": "hidden"},
                "region": "center-left",
            })
        group = {
            "build_id": "build",
            "group_id": group_id,
            "source_id": source,
            "source_path": str(self.before),
            "render_mode": mode,
            "subject": {"name": "person"},
            "candidates": candidates,
            "winner_ids": [candidates[0]["candidate_id"]],
        }
        return {
            "task_id": stable_id("annotation", group_id, candidates[0]["candidate_id"], 1),
            "group_id": group_id,
            "candidate_id": candidates[0]["candidate_id"],
            "winner_rank": 1,
            "group": group,
            "candidate": candidates[0],
        }

    @staticmethod
    def factory(clients):
        def create(endpoint, route):
            key = endpoint.id if hasattr(endpoint, "id") else "local"
            return clients[key]
        return create


class RequestShapeTests(ResponseFixture):
    def test_pinned_sdk_and_strict_request_shapes(self):
        preflight_openai_sdk()
        prepared = prepare_task(self.task(), self.config)
        external = request_payload(prepared, self.config, route="external")
        self.assertEqual(external["model"], "external-model")
        self.assertEqual(external["reasoning"], {"effort": "medium"})
        self.assertEqual(external["max_output_tokens"], 6000)
        fmt = external["text"]["format"]
        self.assertTrue(fmt["strict"])
        self.assertEqual(fmt["schema"], ANNOTATION_JSON_SCHEMA)
        self.assertEqual(fmt["schema"]["required"], list(ANNOTATION_FIELDS))
        self.assertFalse(fmt["schema"]["additionalProperties"])

        content = external["input"][0]["content"]
        images = [row["image_url"] for row in content if row["type"] == "input_image"]
        self.assertEqual(len(images), 2)
        decoded = []
        for data_url in images:
            image = Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1])))
            self.assertEqual(image.format, "JPEG")
            self.assertEqual(max(image.size), 768)
            decoded.append(image.convert("RGB").resize((1, 1)).getpixel((0, 0)))
        self.assertGreater(decoded[0][0], decoded[0][1])
        self.assertGreater(decoded[1][1], decoded[1][0])

        local = request_payload(prepared, self.config, route="local")
        self.assertEqual(local["model"], "qwen3_5-35b-a3b")
        self.assertEqual(local["temperature"], 0.2)
        self.assertEqual(local["max_output_tokens"], 2048)
        self.assertEqual(
            local["extra_body"], {"chat_template_kwargs": {"enable_thinking": False}}
        )

    def test_local_prompt_exposes_only_subject_region_and_objective_hints(self):
        task = self.task(mode="local")
        prompt = prepare_task(task, self.config).prompt
        self.assertIn("person", prompt)
        self.assertIn("center-left", prompt)
        self.assertNotIn("mask-0", prompt)
        self.assertNotIn("instance_id", prompt)
        self.assertNotIn("preset-0", prompt)


class StreamTests(unittest.TestCase):
    def test_official_typed_completed_event_is_required(self):
        result = _consume_stream(FakeStream(completed_events()))
        self.assertEqual(result.fields, valid_fields())
        self.assertEqual(result.returned_model, "returned-model")
        self.assertEqual(result.usage["total_tokens"], 30)

        from openai.types.responses import ResponseErrorEvent, ResponseFailedEvent

        with self.assertRaisesRegex(AnnotationError, "before response.completed"):
            _consume_stream(FakeStream(completed_events()[:-1]))
        failed = ResponseFailedEvent.model_construct(
            response=SimpleNamespace(
                error=SimpleNamespace(code="server_error", message="failed")
            ),
            sequence_number=1,
            type="response.failed",
        )
        with self.assertRaisesRegex(AnnotationError, "failed"):
            _consume_stream(FakeStream([failed]))
        errored = ResponseErrorEvent.model_construct(
            code="insufficient_quota", message="quota", param=None,
            sequence_number=1, type="error",
        )
        with self.assertRaises(AnnotationError) as caught:
            _consume_stream(FakeStream([errored]))
        self.assertTrue(caught.exception.quota)

    def test_lifecycle_events_are_typed_and_discriminators_are_enforced(self):
        from openai.types.responses import ResponseCreatedEvent, ResponseTextDeltaEvent

        created = ResponseCreatedEvent.model_construct(
            response=SimpleNamespace(status="in_progress"),
            sequence_number=0,
            type="response.created",
        )
        result = _consume_stream(FakeStream([created, *completed_events()]))
        self.assertEqual(result.fields, valid_fields())

        spoofed = ResponseTextDeltaEvent.model_construct(
            content_index=0, delta="{}", item_id="msg", logprobs=[], output_index=0,
            sequence_number=1, type="response.completed",
        )
        with self.assertRaisesRegex(AnnotationError, "untyped event"):
            _consume_stream(FakeStream([spoofed]))
        with self.assertRaisesRegex(RuntimeError, "untyped_responses_event"):
            consume_text(FakeStream([spoofed]))
        with self.assertRaisesRegex(AnnotationError, "untyped event"):
            _consume_stream(FakeStream([SimpleNamespace(type="response.created")]))

    def test_malformed_schema_is_terminal(self):
        bad = valid_fields()
        bad.pop("instruction_short")
        with self.assertRaises(AnnotationError) as caught:
            _consume_stream(FakeStream(completed_events(bad)))
        self.assertEqual(caught.exception.code, "schema_failed")
        self.assertFalse(caught.exception.retryable)


class PoolTests(ResponseFixture):
    def test_least_inflight_and_round_robin_ties(self):
        pool = ExternalRelayPool(self.config.external_endpoints)
        with pool.lease() as first:
            with pool.lease() as second:
                with pool.lease() as third:
                    self.assertEqual(
                        [first.id, second.id, third.id],
                        ["relay-a", "relay-b", "relay-a"],
                    )
        with pool.lease() as fourth:
            self.assertEqual(fourth.id, "relay-b")


class DurableDrainTests(ResponseFixture):
    def test_completed_stream_writes_training_shape_and_tokens(self):
        with ArtifactStore(self.root / "out", "build", fsync_every=1) as store:
            task = self.task()
            store.append_group(task["group"])
            relay_a = FakeResponses([completed_events()])
            clients = {
                "relay-a": FakeClient(relay_a),
                "relay-b": FakeClient(FakeResponses([])),
                "local": FakeClient(FakeResponses([])),
            }
            annotator = ResponsesAnnotator(
                self.config, store, client_factory=self.factory(clients), sleep=lambda _: None
            )
            result = annotator.drain(max_workers=1)
            self.assertEqual(result["completed"], 1)
            row = next(iter(store.sft.values()))
            self.assertEqual(
                set(("I_in", "I_tar", "recipe", "local", "task_type", "instruction",
                     "instruction_short", "reasoning", "annot_src", "qa")) - set(row),
                set(),
            )
            self.assertIn("<problem_light_start>", row["reasoning"])
            self.assertIn("<plan_specificcolor_end>", row["reasoning"])
            self.assertEqual(row["annot_src"], "responses:external:relay-a")
            self.assertEqual(relay_a.calls[0]["max_output_tokens"], 6000)

    def test_local_sft_preserves_canonical_mask_metadata(self):
        with ArtifactStore(self.root / "local-shape", "build", fsync_every=1) as store:
            task = self.task(mode="local")
            candidate = task["group"]["candidates"][0]
            candidate.update({
                "slot_mode": "linear", "mode_index": 1, "pairing_index": 6,
                "raw_alpha_mean": 0.75, "amount": 2.0 / 3.0,
                "effective_alpha_mean": 0.5,
            })
            task["candidate"] = candidate
            store.append_group(task["group"])
            clients = {
                "relay-a": FakeClient(FakeResponses([completed_events()])),
                "relay-b": FakeClient(FakeResponses([])),
                "local": FakeClient(FakeResponses([])),
            }
            ResponsesAnnotator(
                self.config, store, client_factory=self.factory(clients), sleep=lambda _: None
            ).drain(max_workers=1)
            local = next(iter(store.sft.values()))["local"]
            self.assertEqual(local["slot_mode"], "linear")
            self.assertEqual(local["pairing_index"], 6)
            self.assertEqual(local["raw_alpha_mean"], 0.75)
            self.assertAlmostEqual(local["amount"], 2.0 / 3.0)
            self.assertEqual(local["effective_alpha_mean"], 0.5)

    def test_429_and_5xx_retry_pool_wide_and_respect_retry_after(self):
        sleeps = []
        with ArtifactStore(self.root / "retry", "build", fsync_every=1) as store:
            task = self.task()
            store.append_group(task["group"])
            a = FakeResponses([
                FakeHttpError(429, "rate_limit", retry_after="7"),
                completed_events(),
            ])
            b = FakeResponses([FakeHttpError(503, "overloaded")])
            clients = {
                "relay-a": FakeClient(a), "relay-b": FakeClient(b),
                "local": FakeClient(FakeResponses([])),
            }
            result = ResponsesAnnotator(
                self.config, store, client_factory=self.factory(clients),
                sleep=sleeps.append, random_value=lambda: 0.5,
            ).drain(max_workers=1)
            self.assertEqual(result["completed"], 1)
            self.assertEqual(sleeps, [7.0, 2.0])
            self.assertEqual(len(a.calls) + len(b.calls), 3)
            self.assertFalse(store.external_pool_exhausted())

    def test_nonretryable_4xx_and_schema_failure_are_terminal(self):
        for name, action, code in (
            ("bad-request", FakeHttpError(400, "invalid_request"), "invalid_request"),
            ("schema", completed_events({**valid_fields(), "extra": "not allowed"}),
             "schema_failed"),
        ):
            with self.subTest(name=name), ArtifactStore(
                self.root / name, "build", fsync_every=1
            ) as store:
                task = self.task(source=name)
                store.append_group(task["group"])
                clients = {
                    "relay-a": FakeClient(FakeResponses([action])),
                    "relay-b": FakeClient(FakeResponses([])),
                    "local": FakeClient(FakeResponses([])),
                }
                result = ResponsesAnnotator(
                    self.config, store, client_factory=self.factory(clients),
                    sleep=lambda _: None,
                ).drain(max_workers=1)
                self.assertEqual(result["terminal"], 1)
                self.assertEqual(len(store.sft), 0)
                self.assertTrue(any(
                    row.get("terminal") and row.get("error_code") == code
                    for row in store.failures
                ))

    def test_dual_quota_exhaustion_latches_before_same_round_local_reroute(self):
        out = self.root / "quota"
        with ArtifactStore(out, "build", fsync_every=1) as store:
            task = self.task()
            store.append_group(task["group"])

            def assert_latched():
                self.assertTrue(store.external_pool_exhausted())

            clients = {
                "relay-a": FakeClient(FakeResponses([
                    FakeHttpError(429, "insufficient_quota")
                ])),
                "relay-b": FakeClient(FakeResponses([
                    FakeHttpError(429, "billing_hard_limit_reached")
                ])),
                "local": FakeClient(FakeResponses(
                    [completed_events(model="qwen3_5-35b-a3b")], before_call=assert_latched
                )),
            }
            result = ResponsesAnnotator(
                self.config, store, client_factory=self.factory(clients), sleep=lambda _: None
            ).drain(max_workers=1)
            self.assertEqual(result["completed"], 1)
            self.assertTrue(store.external_pool_exhausted())
            row = next(iter(store.sft.values()))
            self.assertEqual(row["annot_src"], "responses:local")
            second = self.task(source="source-2")
            store.append_group(second["group"])

        with ArtifactStore(out, "build", fsync_every=1) as resumed:
            local = FakeResponses([completed_events(model="qwen3_5-35b-a3b")])
            clients = {
                "relay-a": FakeClient(FakeResponses([])),
                "relay-b": FakeClient(FakeResponses([])),
                "local": FakeClient(local),
            }
            result = ResponsesAnnotator(
                self.config, resumed, client_factory=self.factory(clients), sleep=lambda _: None
            ).drain(max_workers=1)
            self.assertEqual(result["completed"], 1)
            self.assertEqual(len(local.calls), 1)
            self.assertEqual(len(clients["relay-a"].responses.calls), 0)
            self.assertEqual(len(clients["relay-b"].responses.calls), 0)

    def test_three_round_transport_budget_is_durable_on_resume(self):
        out = self.root / "transport"
        actions_a = [ConnectionError("offline-a") for _ in range(6)]
        actions_b = [ConnectionError("offline-b") for _ in range(6)]
        with ArtifactStore(out, "build", fsync_every=1) as store:
            task = self.task()
            store.append_group(task["group"])
            clients = {
                "relay-a": FakeClient(FakeResponses(actions_a)),
                "relay-b": FakeClient(FakeResponses(actions_b)),
                "local": FakeClient(FakeResponses([])),
            }
            result = ResponsesAnnotator(
                self.config, store, client_factory=self.factory(clients),
                sleep=lambda _: None, random_value=lambda: 0.5,
            ).drain(max_workers=1)
            self.assertEqual(result["transport_failed"], 1)
            attempts = [row for row in store.failures if row.get("event_type") == "attempt"]
            self.assertEqual(len(attempts), 12)
            self.assertEqual({row["round"] for row in attempts}, {1, 2, 3})

        with ArtifactStore(out, "build", fsync_every=1) as resumed:
            never = FakeResponses([])
            clients = {
                "relay-a": FakeClient(never), "relay-b": FakeClient(never),
                "local": FakeClient(never),
            }
            result = ResponsesAnnotator(
                self.config, resumed, client_factory=self.factory(clients),
                sleep=lambda _: None,
            ).drain(max_workers=1)
            self.assertEqual(result["pending"], 0)
            self.assertEqual(len(never.calls), 0)


class Sam3ResponsesTransportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / "source.jpg"
        self.overlay = self.root / "overlay.jpg"
        Image.new("RGB", (80, 60), (60, 80, 100)).save(self.source)
        Image.new("RGB", (80, 60), (180, 100, 40)).save(self.overlay)

    def tearDown(self):
        self.tmp.cleanup()

    def test_subject_label_uses_typed_streaming_responses_and_strict_schema(self):
        fields = {
            "has_localizable_subject": True,
            "subject_scope": "single",
            "sam_prompt": "woman",
            "expected_count": 1,
            "description": "woman near the center",
            "visual_center": [0.48, 0.55],
            "confidence": 0.93,
            "reason_code": "clear_primary",
        }
        responses = FakeResponses([completed_events(fields)])
        with patch(
            "dataset_build.core.responses_vlm._client",
            return_value=FakeClient(responses),
        ):
            asset_id, variant, result = _call_subject_label(
                {"asset_id": "source-1", "source_path": str(self.source)},
                "a", "http://127.0.0.1:8003/v1", "qwen3_5-35b-a3b", 30,
                api_key="local-secret",
            )
        self.assertEqual((asset_id, variant), ("source-1", "a"))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["sam_prompt"], "woman")
        payload = responses.calls[0]
        self.assertTrue(payload["stream"])
        self.assertNotIn("messages", payload)
        self.assertEqual(payload["max_output_tokens"], 320)
        self.assertEqual(payload["extra_body"]["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(payload["text"]["format"]["schema"], SUBJECT_LABEL_SCHEMA)
        self.assertTrue(payload["text"]["format"]["strict"])
        self.assertEqual(
            [item["type"] for item in payload["input"][0]["content"]],
            ["input_image", "input_text"],
        )
        self.assertNotIn("local-secret", json.dumps(payload))

    def test_selector_uses_two_images_and_maps_stable_ids(self):
        fields = {
            "decision": "select",
            "instance_ids": [2, 1],
            "confidence": 0.88,
            "subject": "woman",
            "reason_code": "clear_primary",
        }
        responses = FakeResponses([completed_events(fields)])
        record = {
            "asset_id": "source-1",
            "source_path": str(self.source),
            "main_subject": "woman",
            "subject_annotation": {"a": {
                "subject_scope": "single",
                "description": "woman near center",
                "visual_center": [0.5, 0.5],
            }},
            "variants": {"a": {
                "overlay_path": str(self.overlay),
                "display_to_stable": {"1": 7, "2": 3},
            }},
            "proposals": [
                {"stable_id": 3, "mask_sha256": "sha-three"},
                {"stable_id": 7, "mask_sha256": "sha-seven"},
            ],
        }
        with patch(
            "dataset_build.core.responses_vlm._client",
            return_value=FakeClient(responses),
        ):
            _, _, result = _call_selector(
                record, "a", "http://127.0.0.1:8003/v1", "qwen3_5-35b-a3b", 30,
            )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["stable_instance_ids"], [3, 7])
        self.assertEqual(result["mask_sha256_list"], ["sha-seven", "sha-three"])
        payload = responses.calls[0]
        self.assertEqual(payload["text"]["format"]["schema"], SUBJECT_SELECTION_SCHEMA)
        self.assertEqual(
            [item["type"] for item in payload["input"][0]["content"]],
            ["input_image", "input_image", "input_text"],
        )

    def test_untyped_stream_failure_is_sanitized_after_three_attempts(self):
        responses = FakeResponses([[{"type": "response.output_text.delta"}]] * 3)
        with patch(
            "dataset_build.core.responses_vlm._client",
            return_value=FakeClient(responses),
        ):
            _, _, result = _call_subject_label(
                {"asset_id": "source-1", "source_path": str(self.source)},
                "a", "http://user:password@127.0.0.1:8003/v1", "model", 30,
                api_key="local-secret",
            )
        self.assertEqual(result["status"], "transport_error")
        self.assertEqual(result["error"], "RuntimeError")
        self.assertEqual(len(responses.calls), 3)
        self.assertNotIn("secret", json.dumps(result))
        self.assertNotIn("password", json.dumps(result))

    def test_subject_and_selector_parsers_reject_repaired_values(self):
        label = {
            "has_localizable_subject": True,
            "subject_scope": "single",
            "sam_prompt": "woman",
            "expected_count": 1,
            "description": "woman near center",
            "visual_center": [0.5, 0.5],
            "confidence": 0.9,
            "reason_code": "clear_primary",
        }
        self.assertEqual(_parse_subject_label(json.dumps(label))["status"], "ok")
        invalid_labels = []
        for key, value in (
            ("sam_prompt", " Woman "),
            ("expected_count", "1"),
            ("description", "woman near center "),
            ("visual_center", [1.2, 0.5]),
            ("confidence", True),
        ):
            invalid_labels.append({**label, key: value})
        invalid_labels.append({**label, "extra": "field"})
        invalid_labels.append({**label, "has_localizable_subject": False,
                               "visual_center": [0.5, 0.5]})
        for fields in invalid_labels:
            with self.subTest(label=fields):
                self.assertEqual(
                    _parse_subject_label(json.dumps(fields))["status"], "parse_error"
                )

        selection = {
            "decision": "select", "instance_ids": [1], "confidence": 0.8,
            "subject": "woman", "reason_code": "clear_primary",
        }
        self.assertEqual(_parse_selection(json.dumps(selection), {1, 2})["status"], "ok")
        invalid_selections = (
            {**selection, "instance_ids": [0]},
            {**selection, "instance_ids": [1, 1]},
            {**selection, "instance_ids": [True]},
            {**selection, "instance_ids": []},
            {**selection, "subject": " woman"},
            {**selection, "confidence": 1.2},
            {**selection, "decision": "no_subject", "instance_ids": [1]},
        )
        for fields in invalid_selections:
            with self.subTest(selection=fields):
                self.assertEqual(
                    _parse_selection(json.dumps(fields), {1, 2})["status"], "parse_error"
                )
        self.assertEqual(
            _parse_selection(json.dumps({**selection, "instance_ids": [3]}), {1, 2})["status"],
            "invalid_id",
        )


class CaptionResponsesTransportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / "source.jpg"
        Image.new("RGB", (1600, 800), (40, 90, 150)).save(self.source)

    def tearDown(self):
        self.tmp.cleanup()

    def test_caption_uses_typed_streaming_responses_and_strict_schema(self):
        fields = {
            "caption": "一名女子站在柔和冷色调的窗边。",
            "subjects": [
                {"en": "woman", "cn": "女子", "main": True, "area": 0.35},
            ],
        }
        responses = FakeResponses([completed_events(fields)])
        with patch(
            "dataset_build.core.responses_vlm._client",
            return_value=FakeClient(responses),
        ):
            result = caption_one({"path": str(self.source)})

        self.assertEqual(result, fields)
        payload = responses.calls[0]
        self.assertTrue(payload["stream"])
        self.assertNotIn("messages", payload)
        self.assertEqual(payload["max_output_tokens"], 1100)
        self.assertEqual(payload["text"]["format"]["schema"], CAPTION_SUBJECTS_SCHEMA)
        self.assertTrue(payload["text"]["format"]["strict"])
        content = payload["input"][0]["content"]
        self.assertEqual([item["type"] for item in content], ["input_image", "input_text"])
        image = Image.open(io.BytesIO(base64.b64decode(
            content[0]["image_url"].split(",", 1)[1]
        )))
        self.assertEqual(image.format, "JPEG")
        self.assertEqual(max(image.size), 768)

    def test_caption_parser_rejects_repaired_or_non_object_json(self):
        self.assertIsNone(parse_caption_subjects("```json\n{}\n```"))
        self.assertIsNone(parse_caption_subjects('[{"caption": "not an object"}]'))
        valid = {
            "caption": "一名女子站在窗边。",
            "subjects": [{"en": "woman", "cn": "女子", "main": True, "area": 0.35}],
        }
        self.assertEqual(parse_caption_subjects(json.dumps(valid)), valid)
        invalid = (
            {**valid, "caption": "    "},
            {**valid, "caption": " 一名女子站在窗边。"},
            {**valid, "subjects": [{**valid["subjects"][0], "en": "Woman"}]},
            {**valid, "subjects": [{**valid["subjects"][0], "cn": " 女子"}]},
            {**valid, "subjects": [{**valid["subjects"][0], "main": 1}]},
            {**valid, "subjects": [{**valid["subjects"][0], "area": 1.2}]},
            {**valid, "subjects": [valid["subjects"][0], {**valid["subjects"][0]}]},
        )
        for fields in invalid:
            with self.subTest(fields=fields):
                self.assertIsNone(parse_caption_subjects(json.dumps(fields)))


class BrokerResponsesTests(unittest.IsolatedAsyncioTestCase):
    async def test_responses_route_streams_and_releases_after_termination(self):
        from dataset_build.core.broker.app import BrokerConfig, _proxy_generative, build_app

        released = []

        class UpstreamResponse:
            status_code = 200
            headers = {"content-type": "text/event-stream"}

            async def aiter_raw(self):
                yield b"data: first\n\n"
                yield b"data: second\n\n"

            async def aclose(self):
                return None

        class Client:
            def __init__(self):
                self.url = None

            def build_request(self, method, url, **_kwargs):
                self.url = url
                return object()

            async def send(self, _request, *, stream):
                self.assert_stream = stream
                return UpstreamResponse()

        client = Client()

        class Broker:
            cfg = SimpleNamespace(default_class="build-annotate", retry_after=3)

            async def admit(self, cls):
                self.cls = cls
                return 8123

            async def release(self, port):
                released.append(port)

        broker = Broker()
        broker.client = client

        async def body():
            return b'{"model":"qwen"}'

        request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(broker=broker)),
            headers={"x-vgate-class": "build-annotate"},
            body=body,
        )
        response = await _proxy_generative(request, "responses")
        self.assertEqual(client.url, "http://127.0.0.1:8123/v1/responses")
        self.assertEqual(released, [])
        iterator = response.body_iterator.__aiter__()
        self.assertEqual(await iterator.__anext__(), b"data: first\n\n")
        self.assertEqual(released, [])
        self.assertEqual(await iterator.__anext__(), b"data: second\n\n")
        self.assertEqual(released, [])
        with self.assertRaises(StopAsyncIteration):
            await iterator.__anext__()
        self.assertEqual(released, [8123])

        cfg = BrokerConfig(ports=[8001], served_name="qwen3_5-35b-a3b")
        routes = {route.path for route in build_app(cfg).routes}
        self.assertIn("/v1/responses", routes)
        self.assertNotIn("/v1/chat/completions", routes)
        self.assertNotIn("/v1/completions", routes)


if __name__ == "__main__":
    unittest.main()
