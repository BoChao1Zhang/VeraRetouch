from __future__ import annotations

import asyncio
import base64
import dataclasses
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
    MODEL_ATTEMPT_LIMIT,
    PROSE_ATTEMPT_LIMIT,
    PROSE_VIOLATION,
    SCHEMA_ATTEMPT_LIMIT,
    UPSTREAM_ATTEMPT_LIMIT,
    AnnotationError,
    ExternalRelayPool,
    ResponsesAnnotator,
    _consume_stream,
    build_prompt,
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
        "region_scope": "global adjustment across the entire frame",
        "instruction_long": "Apply the Style Name look with balanced tone and color.",
        "instruction_short": "Apply the Style Name look.",
    }


def completed_events(fields: dict[str, str] | None = None, *, model: str = "external-model"):
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
    def __init__(
        self,
        status_code: int,
        code: str,
        *,
        retry_after: str | None = None,
        body: Any | None = None,
    ):
        super().__init__(f"HTTP {status_code}: {code}")
        self.status_code = status_code
        self.body = {"error": {"code": code}} if body is None else body
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
            external_reasoning_effort="low",
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
        self.assertEqual(external["reasoning"], {"effort": "low"})
        self.assertEqual(external["max_output_tokens"], 6000)
        # Diversity knobs both production lanes accept; they are prompt-layer
        # constants, so they never reach the durable effective config.
        self.assertEqual(external["temperature"], 1.2)
        self.assertEqual(external["top_p"], 1.0)
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

    def test_geometry_hint_is_read_in_the_aspect_ratio_of_the_before_image(self):
        # The before image is 1200x600, so this band's stored 71.72 deg -- vertical
        # in the normalised frame the geometry is written in -- is a 56 deg
        # diagonal on the picture the model is handed.  prepare_task is what wires
        # the size through; build_prompt on its own has no image to measure.
        task = self.task(mode="local")
        task["candidate"].update(
            slot_mode="band",
            geometry={
                "Angle": 71.72, "Top": 0.2604, "Bottom": 0.8947, "Left": -0.3017,
                "Right": 0.8213, "Feather": 55.0, "Flipped": "true",
            },
        )
        diagonal = "straight band, diagonal, running from the upper left down to the lower right"
        prompt = prepare_task(task, self.config).prompt
        self.assertIn(diagonal, prompt)
        self.assertNotIn("straight band, vertical", prompt)
        self.assertIn("straight band, vertical", build_prompt(task))


class ObjectiveHintWordingTests(ResponseFixture):
    """v5 measured-direction table (WP15c), and the v4.1 licence rules it keeps.

    The fresh100 review's eight blind fails all *followed* their hints, so the
    v4.1 revision was about what the model is licensed to say, not about making
    it obey harder; WP14 then measured the anchoring effect that makes that the
    only safe posture (a wrong hint drives luna from 55.8% to 5.8%).  v5 keeps
    every one of those licence rules and changes the numbers behind them to the
    WP15ab ROC winners, plus a per-surface block that is the only place a
    localised colour claim is allowed to come from.
    """

    def prompt_for(self, hints: dict[str, Any]) -> str:
        task = self.task(mode="local")
        task["candidate"]["objective_hints"] = hints
        return build_prompt(task)

    @staticmethod
    def hints(surfaces: Any = None, **deltas: float) -> dict[str, Any]:
        labels = {
            "brightness": ("brighter", "darker"),
            "warmth": ("warmer", "cooler"),
            "hue_gm": ("shifted toward magenta/red", "shifted toward green"),
            "chroma": ("richer", "more muted"),
            "contrast": ("higher contrast", "lower contrast"),
        }
        built: dict[str, Any] = {
            name: {
                "delta": deltas.get(name, 0.0),
                "direction": labels[name][0 if deltas.get(name, 0.0) > 0 else 1],
            }
            for name in labels
        }
        built["surfaces"] = list(surfaces or [])
        return built

    @staticmethod
    def surface(name: str, area: float, d_C: float, **extra: Any) -> dict[str, Any]:
        row = {
            "name": name, "area": area, "d_L": 0.0, "d_a": 0.0, "d_b": 0.0,
            "d_C": d_C, "direction": "richer" if d_C > 0 else "more muted",
            "low_confidence": False,
        }
        row.update(extra)
        return row

    def test_the_table_is_three_labelled_blocks_in_a_fixed_order(self):
        # The block order is load bearing: the veto has to be read before the
        # per-surface licence that narrows it, and the reading rule last.
        prompt = self.prompt_for(self.hints(brightness=6.0))
        headings = [
            "MEASURED EDIT DIRECTIONS (read off the image pair, not estimated).",
            "OVERALL, across the whole edit region:",
            "WHICH COLOURS MOVED, named by the colour they had in the BEFORE image",
            "HOW TO USE:",
        ]
        positions = []
        for heading in headings:
            self.assertIn(heading, prompt)
            positions.append(prompt.index(heading))
        self.assertEqual(positions, sorted(positions))

    def test_all_five_axes_speak_even_when_they_have_nothing_to_say(self):
        # v4a's silence was an omission, which reads as licence.  Every axis
        # gets a line, and a silent one says so in words.
        prompt = self.prompt_for(self.hints(brightness=6.0))
        overall = prompt.split("OVERALL, across the whole edit region:\n")[1]
        overall = overall.split("\n\nWHICH COLOURS MOVED")[0]
        self.assertEqual(
            [line.strip().split(":")[0] for line in overall.splitlines()],
            ["brightness", "contrast", "saturation", "warmth", "green/magenta"],
        )
        self.assertEqual(overall.count(
            "the shift is subtle and may not be visible -- do not state a "
            "direction either way"), 4)

    def test_each_axis_dead_band_is_the_wp15_operating_point(self):
        # brightness 1.0 and saturation 2.0 survived the ROC unchanged; warmth
        # tightened to 1.2 because the axis is now a projection on the high-alpha
        # core, contrast widened to 1.3 because dropping clipped pixels raises a
        # real move, and green/magenta enters at 5.0.
        just_under = self.prompt_for(self.hints(
            brightness=0.99, contrast=1.29, chroma=1.99, warmth=1.19, hue_gm=4.99))
        for axis in ("brightness", "contrast", "saturation", "warmth", "green/magenta"):
            self.assertIn(f"{axis}: the shift is subtle", just_under)

        just_over = self.prompt_for(self.hints(
            brightness=1.0, contrast=1.3, chroma=2.0, warmth=1.2, hue_gm=5.0))
        self.assertIn("brightness: moderately brighter", just_over)
        self.assertIn("contrast: moderately higher contrast", just_over)
        self.assertIn("saturation: moderately richer", just_over)
        self.assertIn("warmth: moderately warmer", just_over)
        self.assertIn("green/magenta: strongly shifted toward magenta/red", just_over)

    def test_every_asserted_direction_is_anchored_against_its_opposite(self):
        prompt = self.prompt_for(self.hints(
            brightness=6.0, contrast=-6.0, warmth=-6.0, chroma=6.0, hue_gm=-6.0))
        for phrase in (
            "brightness: strongly brighter -- do not describe it as darker",
            "contrast: strongly lower contrast -- do not describe it as higher contrast",
            "warmth: strongly cooler -- do not describe it as warmer",
            "saturation: strongly richer -- do not describe it as more muted",
            "green/magenta: strongly shifted toward green -- do not describe it as "
            "shifted toward magenta/red",
        ):
            self.assertIn(phrase, prompt)

    def test_a_sub_band_axis_is_refused_rather_than_called_mixed(self):
        # v4a said "near-neutral or visually mixed" and said nothing about what
        # to do with that, which the model read as licence to pick a direction.
        prompt = self.prompt_for(self.hints(warmth=0.2, chroma=0.2))
        self.assertNotIn("near-neutral or visually mixed", prompt)
        self.assertIn(
            "warmth: the shift is subtle and may not be visible -- do not state a "
            "direction either way", prompt,
        )

    def test_the_surface_block_carries_the_gated_rows_and_says_so_when_empty(self):
        empty = self.prompt_for(self.hints(brightness=6.0))
        self.assertIn(
            "none -- no single colour moved enough on its own to be named", empty)

        listed = self.prompt_for(self.hints(
            chroma=-3.0,
            surfaces=[self.surface("red", 0.18, -9.2, d_L=1.4),
                      self.surface("blue", 0.09, 7.1)],
        ))
        self.assertIn(
            "the red areas (18% of the region, saturation -9.2, lightness +1.4): "
            "clearly more muted -- do not describe it as richer", listed)
        self.assertIn("the blue areas (9% of the region", listed)
        self.assertNotIn(
            "none -- no single colour moved enough on its own to be named", listed)

    def test_a_surface_that_contradicts_the_region_is_demoted_not_dropped(self):
        # The whole-region figure is the one the ROC operating point was fitted
        # on, so it keeps authority; dropping the dissenting surface outright is
        # how v4.1 lost the lipstick case (region mean -0.46, lipstick -10.73).
        prompt = self.prompt_for(self.hints(
            chroma=3.0,
            surfaces=[self.surface("red", 0.12, -10.7, low_confidence=True)],
        ))
        self.assertIn("the red areas (12% of the region", prompt)
        self.assertIn(
            "[uncertain: the whole-region saturation moves the other way, and it is "
            "the one that wins]", prompt)

    def test_hints_are_framed_as_a_veto_and_the_chroma_override_is_gone(self):
        prompt = self.prompt_for(self.hints(brightness=6.0, chroma=-6.0))
        self.assertIn("HOW TO USE:", prompt)
        self.assertIn("Treat the OVERALL lines as a veto, not a script", prompt)
        self.assertIn("never assert a direction they rule out", prompt)
        self.assertIn("A colour listed above is where you are allowed to be specific",
                      prompt)
        self.assertIn("call the effect mixed rather than claiming every object "
                      "changes uniformly", prompt)
        self.assertEqual(
            [line.strip()[:2] for line in prompt.splitlines()
             if line.startswith("  1.") or line.startswith("  2.")
             or line.startswith("  3.")],
            ["1.", "2.", "3."],
        )
        # v4a told the model to drop the warm/cool hint whenever chroma fell,
        # which cancelled the one anchor the colour wording had.
        self.assertNotIn("avoid asserting a warm/cool shift unless", prompt)
        self.assertNotIn("prioritize that visible desaturation", prompt)

    def test_a_pre_wp15c_journal_row_still_renders(self):
        # Backward compatibility is a hard requirement: reannotation reads rows
        # written before this change, and they have four axes and no table.
        prompt = self.prompt_for({
            "brightness": {"delta": 6.0, "direction": "brighter"},
            "warmth": {"delta": 0.4, "direction": "warmer"},
            "chroma": {"delta": -6.0, "direction": "more muted"},
            "contrast": {"delta": -6.0, "direction": "lower"},
        })
        self.assertIn("brightness: strongly brighter -- do not describe it as darker",
                      prompt)
        # The legacy bare tonal word still finds its anchor.
        self.assertIn("contrast: strongly lower -- do not describe it as higher", prompt)
        self.assertNotIn("green/magenta", prompt)
        self.assertIn(
            "none -- no single colour moved enough on its own to be named", prompt)

    def test_legacy_string_hints_still_pass_straight_through(self):
        # Imported rows carry the direction word with no delta to threshold.
        prompt = build_prompt(self.task(mode="local"))
        self.assertIn("brightness: brighter", prompt)
        self.assertIn("saturation: richer", prompt)


class StreamTests(unittest.TestCase):
    def test_known_relay_rate_limit_control_event_is_ignored(self):
        control = SimpleNamespace(type="codex.rate_limits", rate_limits={}, credits=None)
        result = _consume_stream(FakeStream([control, *completed_events()]))
        self.assertEqual(result.fields, valid_fields())

    def test_official_typed_completed_event_is_required(self):
        result = _consume_stream(FakeStream(completed_events()))
        self.assertEqual(result.fields, valid_fields())
        self.assertEqual(result.returned_model, "external-model")
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

    def test_malformed_schema_is_retryable(self):
        bad = valid_fields()
        bad.pop("instruction_short")
        with self.assertRaises(AnnotationError) as caught:
            _consume_stream(FakeStream(completed_events(bad)))
        self.assertEqual(caught.exception.code, "schema_failed")
        # The draw is retryable; the annotator bounds how many are taken.
        self.assertTrue(caught.exception.retryable)


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

    def test_five_key_lanes_round_robin_and_remove_independently(self):
        lanes = tuple(
            ExternalEndpointConfig(
                f"provider-{provider}-lane-{lane}",
                f"https://{provider}.example/v1",
                f"key-{provider}-{lane}",
                1,
            )
            for provider, lane in (("a", 1), ("b", 1), ("b", 2), ("c", 1), ("c", 2))
        )
        pool = ExternalRelayPool(lanes)
        leases = []
        try:
            for _ in lanes:
                lease = pool.lease()
                leases.append(lease)
                endpoint = lease.__enter__()
                self.assertEqual(endpoint.id, lanes[len(leases) - 1].id)
        finally:
            for lease in reversed(leases):
                lease.__exit__(None, None, None)

        self.assertFalse(pool.remove("provider-b-lane-1"))
        seen = set()
        for _ in range(8):
            with pool.lease() as endpoint:
                seen.add(endpoint.id)
        self.assertNotIn("provider-b-lane-1", seen)
        self.assertTrue({lane.id for lane in lanes[2:]}.issubset(seen))

    def test_avoid_reroutes_away_from_the_lane_the_pool_would_have_picked(self):
        # Three lanes, loaded so that the ordinary least-inflight/round-robin
        # choice is relay-a and relay-b is already at capacity.  A substituted
        # draw asks for anything but relay-a, and the only place left is relay-c.
        lanes = (
            ExternalEndpointConfig("relay-a", "https://a.example/v1", "key-a", 2),
            ExternalEndpointConfig("relay-b", "https://b.example/v1", "key-b", 1),
            ExternalEndpointConfig("relay-c", "https://c.example/v1", "key-c", 2),
        )
        pool = ExternalRelayPool(lanes)
        held = []
        for expected in ("relay-a", "relay-b", "relay-c"):
            lease = pool.lease()
            held.append(lease)
            self.assertEqual(lease.__enter__().id, expected)
        # relay-a goes idle again; relay-b stays saturated, relay-c stays busy.
        held[0].__exit__(None, None, None)
        try:
            with pool.lease() as default_choice:
                self.assertEqual(default_choice.id, "relay-a")
            with pool.lease(avoid="relay-a") as rerouted:
                self.assertEqual(rerouted.id, "relay-c")
        finally:
            for lease in reversed(held[1:]):
                lease.__exit__(None, None, None)

    def test_avoid_is_ignored_when_it_would_leave_no_lane_at_all(self):
        pool = ExternalRelayPool(self.config.external_endpoints[:1])
        with pool.lease(avoid="relay-a") as only_lane:
            self.assertEqual(only_lane.id, "relay-a")

    def test_avoid_takes_a_set_so_two_spoiled_lanes_are_both_skipped(self):
        # Debt C-2: avoid used to hold one id, so a task spoiled by relay-a and
        # then by relay-b walked straight back onto relay-a on its third draw.
        lanes = (
            ExternalEndpointConfig("relay-a", "https://a.example/v1", "key-a", 2),
            ExternalEndpointConfig("relay-b", "https://b.example/v1", "key-b", 2),
            ExternalEndpointConfig("relay-c", "https://c.example/v1", "key-c", 2),
        )
        pool = ExternalRelayPool(lanes)
        for _ in range(4):
            with pool.lease(avoid={"relay-a", "relay-b"}) as endpoint:
                self.assertEqual(endpoint.id, "relay-c")
        # Excluding everything still yields a lane rather than stalling.
        with pool.lease(avoid={"relay-a", "relay-b", "relay-c"}) as fallback:
            self.assertIn(fallback.id, {"relay-a", "relay-b", "relay-c"})


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
                     "instruction_short", "reasoning", "annot_src", "annot_model",
                     "qa")) - set(row),
                set(),
            )
            self.assertIn("<problem_light_start>", row["reasoning"])
            self.assertIn("<plan_specificcolor_end>", row["reasoning"])
            self.assertEqual(row["annot_src"], "responses:external:relay-a")
            self.assertEqual(row["annot_model"], "external-model")
            self.assertEqual(row["qa"]["annotation"]["returned_model"], "external-model")
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

    def _bad_schema_events(self):
        return completed_events({**valid_fields(), "extra": "not allowed"})

    def test_nonretryable_4xx_and_exhausted_bad_draws_are_terminal(self):
        # 4xx is terminal on sight; a bad draw (schema violation or substituted
        # model) is terminal only after its bounded number of draws is spent.
        for name, actions, code in (
            ("bad-request", [FakeHttpError(400, "invalid_request")], "invalid_request"),
            ("schema", [self._bad_schema_events() for _ in range(SCHEMA_ATTEMPT_LIMIT)],
             "schema_failed"),
            ("model", [completed_events(model="gpt-5.5")
                       for _ in range(MODEL_ATTEMPT_LIMIT)],
             "model_substituted"),
        ):
            with self.subTest(name=name), ArtifactStore(
                self.root / name, "build", fsync_every=1
            ) as store:
                task = self.task(source=name)
                store.append_group(task["group"])
                # The lanes alternate, so the draws are split across both relays.
                clients = {
                    "relay-a": FakeClient(FakeResponses(actions[0::2])),
                    "relay-b": FakeClient(FakeResponses(actions[1::2])),
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
                self.assertEqual(
                    sum(
                        1 for row in store.failures
                        if row.get("event_type") == "attempt"
                        and row.get("error_code") == code
                    ),
                    len(actions),
                )

    def test_schema_failure_retries_within_its_bound(self):
        sleeps: list[float] = []
        with ArtifactStore(self.root / "schema-retry", "build", fsync_every=1) as store:
            task = self.task(source="schema-retry")
            store.append_group(task["group"])
            clients = {
                "relay-a": FakeClient(FakeResponses([self._bad_schema_events()])),
                "relay-b": FakeClient(FakeResponses([completed_events()])),
                "local": FakeClient(FakeResponses([])),
            }
            result = ResponsesAnnotator(
                self.config, store, client_factory=self.factory(clients),
                sleep=sleeps.append,
            ).drain(max_workers=1)
            self.assertEqual(result["completed"], 1)
            self.assertEqual(result["terminal"], 0)
            # One failed draw is journalled; the second one succeeded, so the SFT
            # row carries attempt 2 and no terminal event exists for the task.
            attempts = [
                row for row in store.failures
                if row.get("event_type") == "attempt"
                and row.get("task_id") == task["task_id"]
            ]
            self.assertEqual([row["error_code"] for row in attempts], ["schema_failed"])
            self.assertEqual([row["attempt"] for row in attempts], [1])
            self.assertTrue(all(row["retryable"] for row in attempts))
            self.assertFalse(any(row.get("terminal") for row in store.failures))
            row = next(iter(store.sft.values()))
            self.assertEqual(row["qa"]["annotation"]["attempt"], 2)
            self.assertEqual(row["annot_src"], "responses:external:relay-b")
            # A bad draw is not congestion, so it is redrawn without backoff.
            self.assertEqual(sleeps, [])

    @staticmethod
    def _after_reference_events():
        """A parseable answer whose problem section points at the after image."""
        return completed_events({
            **valid_fields(),
            "problem_lighting": "The light is softer than the finished image.",
        })

    def test_prose_violation_is_an_immediate_bad_draw(self):
        # The relay answered, the JSON parsed and the model was the one asked
        # for; only the prose broke a v5.1 ban, so the draw is discarded and
        # taken again without backoff, exactly like a schema violation.  It is
        # deliberately not a lane rotation -- both lanes serve the same model, so
        # the text says nothing about where it came from -- which
        # ``ProseViolationTests`` pins on ``_LANE_ROTATING_CODES`` directly.
        sleeps: list[float] = []
        with ArtifactStore(self.root / "prose-retry", "build", fsync_every=1) as store:
            task = self.task(source="prose-retry")
            store.append_group(task["group"])
            clients = {
                "relay-a": FakeClient(FakeResponses([self._after_reference_events()])),
                "relay-b": FakeClient(FakeResponses([completed_events()])),
                "local": FakeClient(FakeResponses([])),
            }
            result = ResponsesAnnotator(
                self.config, store, client_factory=self.factory(clients),
                sleep=sleeps.append,
            ).drain(max_workers=1)
            self.assertEqual((result["completed"], result["terminal"]), (1, 0))
            attempts = [
                row for row in store.failures
                if row.get("event_type") == "attempt"
                and row.get("task_id") == task["task_id"]
            ]
            self.assertEqual(
                [(row["error_code"], row["endpoint_id"], row["retryable"])
                 for row in attempts],
                [(PROSE_VIOLATION, "relay-a", True)],
            )
            self.assertIn("finished image", attempts[0]["message"])
            self.assertFalse(any(row.get("terminal") for row in store.failures))
            row = next(iter(store.sft.values()))
            self.assertEqual(row["qa"]["annotation"]["attempt"], 2)
            self.assertEqual(sleeps, [])

    def test_prose_violation_terminates_once_its_draws_are_spent(self):
        with ArtifactStore(self.root / "prose-bound", "build", fsync_every=1) as store:
            task = self.task(source="prose-bound")
            store.append_group(task["group"])
            actions = [
                self._after_reference_events() for _ in range(PROSE_ATTEMPT_LIMIT)
            ]
            clients = {
                "relay-a": FakeClient(FakeResponses(actions[0::2])),
                "relay-b": FakeClient(FakeResponses(actions[1::2])),
                "local": FakeClient(FakeResponses([])),
            }
            result = ResponsesAnnotator(
                self.config, store, client_factory=self.factory(clients),
                sleep=lambda _: None,
            ).drain(max_workers=1)
            self.assertEqual((result["completed"], result["terminal"]), (0, 1))
            self.assertEqual(len(store.sft), 0)
            self.assertEqual(
                sum(1 for row in store.failures
                    if row.get("event_type") == "attempt"
                    and row.get("error_code") == PROSE_VIOLATION),
                PROSE_ATTEMPT_LIMIT,
            )
            self.assertTrue(any(
                row.get("terminal") and row.get("error_code") == PROSE_VIOLATION
                for row in store.failures
            ))

    def test_substituted_model_is_redrawn_on_another_lane(self):
        sleeps: list[float] = []
        with ArtifactStore(self.root / "model-retry", "build", fsync_every=1) as store:
            task = self.task(source="model-retry")
            store.append_group(task["group"])
            # relay-a answers as a different model, exactly as provider-b-lane-2
            # was observed doing; the redraw must leave that lane.
            clients = {
                "relay-a": FakeClient(FakeResponses([completed_events(model="gpt-5.5")])),
                "relay-b": FakeClient(FakeResponses([completed_events()])),
                "local": FakeClient(FakeResponses([])),
            }
            result = ResponsesAnnotator(
                self.config, store, client_factory=self.factory(clients),
                sleep=sleeps.append,
            ).drain(max_workers=1)
            self.assertEqual((result["completed"], result["terminal"]), (1, 0))
            attempts = [
                row for row in store.failures
                if row.get("event_type") == "attempt"
                and row.get("task_id") == task["task_id"]
            ]
            self.assertEqual(
                [(row["error_code"], row["attempt"], row["endpoint_id"], row["retryable"])
                 for row in attempts],
                [("model_substituted", 1, "relay-a", True)],
            )
            self.assertIn("gpt-5.5", attempts[0]["message"])
            self.assertFalse(any(row.get("terminal") for row in store.failures))
            row = next(iter(store.sft.values()))
            self.assertEqual(row["qa"]["annotation"]["attempt"], 2)
            self.assertEqual(row["annot_src"], "responses:external:relay-b")
            self.assertEqual(row["annot_model"], "external-model")
            # A substituted answer is a bad draw, not congestion.
            self.assertEqual(sleeps, [])

    def test_a_substituted_draw_leaves_the_lane_the_pool_keeps_choosing(self):
        # With two idle lanes the round-robin cursor already alternates, so the
        # rotation is invisible.  Here relay-a is the lane the pool prefers on
        # every draw -- it is the only idle one -- and relay-b is saturated, so
        # only the explicit avoid can move the redraw onto relay-c.
        config = dataclasses.replace(self.config, external_endpoints=(
            ExternalEndpointConfig("relay-a", "https://a.example/v1", "key-a", 2),
            ExternalEndpointConfig("relay-b", "https://b.example/v1", "key-b", 1),
            ExternalEndpointConfig("relay-c", "https://c.example/v1", "key-c", 2),
        ))
        with ArtifactStore(self.root / "avoid-drain", "build", fsync_every=1) as store:
            task = self.task(source="avoid-drain")
            store.append_group(task["group"])
            relay_a = FakeResponses([completed_events(model="gpt-5.5")] * MODEL_ATTEMPT_LIMIT)
            relay_b = FakeResponses([])
            relay_c = FakeResponses([completed_events()])
            annotator = ResponsesAnnotator(
                config, store,
                client_factory=self.factory({
                    "relay-a": FakeClient(relay_a), "relay-b": FakeClient(relay_b),
                    "relay-c": FakeClient(relay_c), "local": FakeClient(FakeResponses([])),
                }),
                sleep=lambda _: None,
            )
            held = []
            for expected in ("relay-a", "relay-b", "relay-c"):
                lease = annotator.pool.lease()
                held.append(lease)
                self.assertEqual(lease.__enter__().id, expected)
            held[0].__exit__(None, None, None)
            try:
                status = annotator.run_round(task, 1)
            finally:
                for lease in reversed(held[1:]):
                    lease.__exit__(None, None, None)

            self.assertEqual(status, "completed")
            self.assertEqual(
                (len(relay_a.calls), len(relay_b.calls), len(relay_c.calls)), (1, 0, 1)
            )
            attempts = [r for r in store.failures if r.get("event_type") == "attempt"]
            self.assertEqual(
                [(r["error_code"], r["endpoint_id"]) for r in attempts],
                [("model_substituted", "relay-a")],
            )
            self.assertEqual(
                next(iter(store.sft.values()))["annot_src"],
                "responses:external:relay-c",
            )

    def test_a_schema_failure_also_leaves_the_lane_that_produced_it(self):
        # fresh100 put all 36 of its schema failures on one lane and none on the
        # other, so a redraw that can land back on the same lane spends the whole
        # bounded budget where the answer cannot come from.  Same staging as the
        # substituted-model case: relay-a is the lane the pool keeps choosing.
        config = dataclasses.replace(self.config, external_endpoints=(
            ExternalEndpointConfig("relay-a", "https://a.example/v1", "key-a", 2),
            ExternalEndpointConfig("relay-b", "https://b.example/v1", "key-b", 1),
            ExternalEndpointConfig("relay-c", "https://c.example/v1", "key-c", 2),
        ))
        with ArtifactStore(self.root / "schema-avoid", "build", fsync_every=1) as store:
            task = self.task(source="schema-avoid")
            store.append_group(task["group"])
            relay_a = FakeResponses([self._bad_schema_events()] * SCHEMA_ATTEMPT_LIMIT)
            relay_b = FakeResponses([])
            relay_c = FakeResponses([completed_events()])
            annotator = ResponsesAnnotator(
                config, store,
                client_factory=self.factory({
                    "relay-a": FakeClient(relay_a), "relay-b": FakeClient(relay_b),
                    "relay-c": FakeClient(relay_c), "local": FakeClient(FakeResponses([])),
                }),
                sleep=lambda _: None,
            )
            held = []
            for expected in ("relay-a", "relay-b", "relay-c"):
                lease = annotator.pool.lease()
                held.append(lease)
                self.assertEqual(lease.__enter__().id, expected)
            held[0].__exit__(None, None, None)
            try:
                status = annotator.run_round(task, 1)
            finally:
                for lease in reversed(held[1:]):
                    lease.__exit__(None, None, None)

            self.assertEqual(status, "completed")
            self.assertEqual(
                (len(relay_a.calls), len(relay_b.calls), len(relay_c.calls)), (1, 0, 1)
            )
            attempts = [r for r in store.failures if r.get("event_type") == "attempt"]
            self.assertEqual(
                [(r["error_code"], r["endpoint_id"]) for r in attempts],
                [("schema_failed", "relay-a")],
            )
            self.assertEqual(
                next(iter(store.sft.values()))["annot_src"],
                "responses:external:relay-c",
            )

    def test_a_spoiled_lane_stays_avoided_across_an_unrelated_failure(self):
        # Debt C-2 at the drain level.  relay-a substitutes the model, relay-b
        # then 500s, and the round-robin cursor would hand the third draw back to
        # relay-a.  The avoid set is not cleared by the intervening 5xx, which is
        # a different lane's problem, so the third draw stays off relay-a.
        with ArtifactStore(self.root / "avoid-sticky", "build", fsync_every=1) as store:
            task = self.task(source="avoid-sticky")
            store.append_group(task["group"])
            relay_a = FakeResponses([
                completed_events(model="gpt-5.5"), completed_events(),
            ])
            relay_b = FakeResponses([
                FakeHttpError(503, "overloaded"), completed_events(),
            ])
            annotator = ResponsesAnnotator(
                self.config, store,
                client_factory=self.factory({
                    "relay-a": FakeClient(relay_a), "relay-b": FakeClient(relay_b),
                    "local": FakeClient(FakeResponses([])),
                }),
                sleep=lambda _: None, random_value=lambda: 0.5,
            )
            self.assertEqual(annotator.run_round(task, 1), "completed")
            self.assertEqual((len(relay_a.calls), len(relay_b.calls)), (1, 2))
            self.assertEqual(
                next(iter(store.sft.values()))["annot_src"],
                "responses:external:relay-b",
            )

    def test_upstream_error_4xx_is_retried_instead_of_killing_the_task(self):
        # provider-b-lane-1 wraps a transient upstream fault as 400 with
        # {"type": "upstream_error"}; WP7 lost 12 tasks to the blanket 4xx rule.
        sleeps: list[float] = []
        with ArtifactStore(self.root / "upstream", "build", fsync_every=1) as store:
            task = self.task(source="upstream")
            store.append_group(task["group"])
            clients = {
                "relay-a": FakeClient(FakeResponses([
                    FakeHttpError(400, "upstream", body={"type": "upstream_error"}),
                ])),
                "relay-b": FakeClient(FakeResponses([completed_events()])),
                "local": FakeClient(FakeResponses([])),
            }
            result = ResponsesAnnotator(
                self.config, store, client_factory=self.factory(clients),
                sleep=sleeps.append,
            ).drain(max_workers=1)
            self.assertEqual((result["completed"], result["terminal"]), (1, 0))
            attempts = [
                r for r in store.failures if r.get("event_type") == "attempt"
            ]
            self.assertEqual(
                [(r["error_code"], r["retryable"]) for r in attempts],
                [("upstream_error", True)],
            )
            self.assertFalse(any(r.get("terminal") for r in store.failures))
            # Bounded like a bad draw, so the redraw is immediate.
            self.assertEqual(sleeps, [])

    def test_upstream_error_is_bounded_and_ordinary_4xx_stays_terminal(self):
        for name, error, code, terminal_after in (
            ("upstream-bound",
             FakeHttpError(400, "upstream", body={"type": "upstream_error"}),
             "upstream_error", UPSTREAM_ATTEMPT_LIMIT),
            ("upstream-nested",
             FakeHttpError(400, "x", body={"error": {"type": "upstream_error"}}),
             "upstream_error", UPSTREAM_ATTEMPT_LIMIT),
            ("plain-4xx", FakeHttpError(400, "invalid_request"), "invalid_request", 1),
        ):
            with self.subTest(name=name), ArtifactStore(
                self.root / name, "build", fsync_every=1
            ) as store:
                task = self.task(source=name)
                store.append_group(task["group"])
                actions = [error] * terminal_after
                clients = {
                    "relay-a": FakeClient(FakeResponses(actions[0::2])),
                    "relay-b": FakeClient(FakeResponses(actions[1::2])),
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
                self.assertEqual(
                    sum(
                        1 for row in store.failures
                        if row.get("event_type") == "attempt"
                        and row.get("error_code") == code
                    ),
                    terminal_after,
                )

    def test_alternating_bad_draws_spend_two_budgets_that_never_borrow(self):
        # sub, schema, sub, schema, sub.  Neither code reaches its own bound of 3
        # before the fifth draw, so that is where the task dies -- and it dies as
        # model_substituted, the code that got there first.  One shared budget
        # would have killed a recoverable task on the third draw instead.
        script = FakeResponses([
            completed_events(model="gpt-5.5"),
            self._bad_schema_events(),
            completed_events(model="gpt-5.5"),
            self._bad_schema_events(),
            completed_events(model="gpt-5.5"),
        ])
        with ArtifactStore(self.root / "alternating", "build", fsync_every=1) as store:
            task = self.task(source="alternating")
            store.append_group(task["group"])
            # Both lanes read from one script, so the sequence the task meets is
            # the sequence above whichever lane the pool hands out.
            result = ResponsesAnnotator(
                self.config, store,
                client_factory=self.factory({
                    "relay-a": FakeClient(script),
                    "relay-b": FakeClient(script),
                    "local": FakeClient(FakeResponses([])),
                }),
                sleep=lambda _: None,
            ).drain(max_workers=1)

            self.assertEqual((result["completed"], result["terminal"]), (0, 1))
            self.assertEqual(len(script.calls), 5)
            codes = [
                row["error_code"] for row in store.failures
                if row.get("event_type") == "attempt"
                and row.get("task_id") == task["task_id"]
            ]
            self.assertEqual(codes, [
                "model_substituted", "schema_failed", "model_substituted",
                "schema_failed", "model_substituted",
            ])
            self.assertEqual(codes.count("model_substituted"), MODEL_ATTEMPT_LIMIT)
            self.assertEqual(codes.count("schema_failed"), SCHEMA_ATTEMPT_LIMIT - 1)
            terminal = [row for row in store.failures if row.get("terminal")]
            # Round 1 spends its four transport attempts; the fifth draw is the
            # first of round 2 and is the one that ends the task.
            self.assertEqual(
                [(row["error_code"], row["round"], row["attempt"]) for row in terminal],
                [("model_substituted", 2, 1)],
            )
            self.assertEqual(len(store.sft), 0)

    def test_local_route_is_not_policed_for_a_substituted_model(self):
        # vLLM answers with the name it was served under, which need not be the id
        # the config asks for.  Local is the single rescue draw after the pool is
        # exhausted and has no other lane to rotate to, so a prefix verdict there
        # would throw away a perfectly good answer and the task with it.
        config = dataclasses.replace(self.config, queue_rounds=1)
        with ArtifactStore(self.root / "local-model", "build", fsync_every=1) as store:
            task = self.task()
            store.append_group(task["group"])
            local = FakeResponses([completed_events(model="Qwen/Qwen3.5-35B-A3B")])
            result = ResponsesAnnotator(
                config, store,
                client_factory=self.factory({
                    "relay-a": FakeClient(FakeResponses([
                        FakeHttpError(429, "insufficient_quota")
                    ])),
                    "relay-b": FakeClient(FakeResponses([
                        FakeHttpError(429, "billing_hard_limit_reached")
                    ])),
                    "local": FakeClient(local),
                }),
                sleep=lambda _: None,
            ).drain(max_workers=1)

            self.assertEqual((result["completed"], result["terminal"]), (1, 0))
            self.assertEqual(len(local.calls), 1)
            self.assertFalse(any(
                row.get("error_code") == "model_substituted" for row in store.failures
            ))
            row = next(iter(store.sft.values()))
            self.assertEqual(row["annot_src"], "responses:local")
            self.assertEqual(row["annot_model"], "Qwen/Qwen3.5-35B-A3B")

    def test_substitution_budget_is_rebuilt_from_the_durable_journal(self):
        # The bound is per task across rounds and resumes: the draws spent before
        # a restart are counted from failures.jsonl, not forgotten.
        config = dataclasses.replace(self.config, transport_attempts_per_round=2)
        out = self.root / "model-resume"
        task = self.task(source="model-resume")
        substituted = lambda: completed_events(model="gpt-5.5")  # noqa: E731
        with ArtifactStore(out, "build", fsync_every=1) as store:
            store.append_group(task["group"])
            status = ResponsesAnnotator(
                config, store,
                client_factory=self.factory({
                    "relay-a": FakeClient(FakeResponses([substituted()])),
                    "relay-b": FakeClient(FakeResponses([substituted()])),
                    "local": FakeClient(FakeResponses([])),
                }),
                sleep=lambda _: None,
            ).run_round(task, 1)
            self.assertEqual(status, "retryable_exhausted")

        with ArtifactStore(out, "build", fsync_every=1) as resumed:
            relay_a = FakeResponses([substituted()])
            relay_b = FakeResponses([substituted()])
            status = ResponsesAnnotator(
                config, resumed,
                client_factory=self.factory({
                    "relay-a": FakeClient(relay_a),
                    "relay-b": FakeClient(relay_b),
                    "local": FakeClient(FakeResponses([])),
                }),
                sleep=lambda _: None,
            ).run_round(task, 2)
            # The third draw overall, not the third of this round, is terminal.
            self.assertEqual(status, "terminal")
            self.assertEqual(len(relay_a.calls) + len(relay_b.calls), 1)
            self.assertTrue(any(
                row.get("terminal") and row.get("error_code") == "model_substituted"
                for row in resumed.failures
            ))

    def test_legacy_sft_row_without_annot_model_is_still_complete(self):
        # Every landed build predates the audit field.  A resume must read those
        # rows as completed work instead of re-annotating them.
        out = self.root / "legacy-sft"
        task = self.task(source="legacy")
        with ArtifactStore(out, "build", fsync_every=1) as store:
            store.append_group(task["group"])
            store.append_sft({
                "build_id": "build",
                "sft_id": stable_id("sft", task["task_id"]),
                "annotation_task_id": task["task_id"],
                "group_id": task["group_id"],
                "candidate_id": task["candidate_id"],
                "winner_rank": 1,
                "I_in": str(self.before),
                "I_tar": str(self.after),
                "recipe": "preset-0",
                "local": None,
                "task_type": "style",
                "instruction": "Apply the Style Name look.",
                "instruction_short": "Apply the look.",
                "reasoning": "<problem_light_start>flat<problem_light_end>",
                "annot_src": "responses:external:relay-a",
                "qa": {"annotation": {"status": "completed"}},
            })
        with ArtifactStore(out, "build", fsync_every=1) as resumed:
            row = next(iter(resumed.sft.values()))
            self.assertNotIn("annot_model", row)
            self.assertIsNone(row.get("annot_model"))
            self.assertIn(task["task_id"], resumed.completed_annotation_tasks())
            # Empty fakes raise on any call, so a redraw would fail the test.
            clients = {
                "relay-a": FakeClient(FakeResponses([])),
                "relay-b": FakeClient(FakeResponses([])),
                "local": FakeClient(FakeResponses([])),
            }
            result = ResponsesAnnotator(
                self.config, resumed, client_factory=self.factory(clients),
                sleep=lambda _: None,
            ).drain(max_workers=1)
            self.assertEqual(
                result,
                {"completed": 0, "terminal": 0, "transport_failed": 0, "pending": 0},
            )

    def test_external_only_quota_exhaustion_never_calls_local(self):
        self.config = dataclasses.replace(self.config, local_fallback=False)
        with ArtifactStore(self.root / "external-only", "build", fsync_every=1) as store:
            task = self.task()
            store.append_group(task["group"])
            local = FakeResponses([])
            clients = {
                "relay-a": FakeClient(FakeResponses([
                    FakeHttpError(429, "insufficient_quota")
                ])),
                "relay-b": FakeClient(FakeResponses([
                    FakeHttpError(429, "billing_hard_limit_reached")
                ])),
                "local": FakeClient(local),
            }
            result = ResponsesAnnotator(
                self.config, store, client_factory=self.factory(clients), sleep=lambda _: None
            ).drain(max_workers=1)
            self.assertEqual(result["transport_failed"], 1)
            self.assertEqual(len(local.calls), 0)
            self.assertTrue(store.external_pool_exhausted())
            skips = [
                row for row in store.failures
                if row.get("error_code") == "external_pool_exhausted_skip"
            ]
            self.assertEqual([row["round"] for row in skips], [1, 2, 3])
            self.assertTrue(all(
                row["event_type"] == "skipped"
                and row["stage"] == "annotation"
                and not row["terminal"]
                and not row["retryable"]
                for row in skips
            ))
            self.assertTrue(any(
                row.get("terminal") and row.get("error_code") == "transport_failed"
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

    def test_final_quota_latch_survives_crash_before_endpoint_marker(self):
        out = self.root / "quota-crash"
        with ArtifactStore(out, "build", fsync_every=1) as store:
            task = self.task()
            store.append_group(task["group"])
            clients = {
                "relay-a": FakeClient(FakeResponses([
                    FakeHttpError(429, "insufficient_quota")
                ])),
                "relay-b": FakeClient(FakeResponses([
                    FakeHttpError(429, "billing_hard_limit_reached")
                ])),
                "local": FakeClient(FakeResponses([])),
            }
            annotator = ResponsesAnnotator(
                self.config, store, client_factory=self.factory(clients), sleep=lambda _: None
            )
            original_append = store.append_failure

            def crash_after_pool_latch(record, *, durable=False):
                appended = original_append(record, durable=durable)
                if record.get("error_code") == "external_pool_exhausted":
                    raise KeyboardInterrupt("crash after durable pool latch")
                return appended

            store.append_failure = crash_after_pool_latch
            with self.assertRaisesRegex(KeyboardInterrupt, "durable pool latch"):
                annotator.run_round(task, 1)
            self.assertTrue(store.external_pool_exhausted())
            self.assertFalse(any(
                row.get("error_code") == "external_endpoint_exhausted"
                and row.get("endpoint_id") == "relay-b"
                for row in store.failures
            ))

        with ArtifactStore(out, "build", fsync_every=1) as resumed:
            local = FakeResponses([completed_events(model="qwen3_5-35b-a3b")])
            relay_a = FakeResponses([])
            relay_b = FakeResponses([])
            result = ResponsesAnnotator(
                self.config,
                resumed,
                client_factory=self.factory({
                    "relay-a": FakeClient(relay_a),
                    "relay-b": FakeClient(relay_b),
                    "local": FakeClient(local),
                }),
                sleep=lambda _: None,
            ).drain(max_workers=1)
            self.assertEqual(result["completed"], 1)
            self.assertTrue(resumed.external_pool_exhausted())
            self.assertEqual(len(local.calls), 1)
            self.assertEqual(len(relay_a.calls) + len(relay_b.calls), 0)

    def test_last_round_last_attempt_exhaustion_still_reroutes_to_local(self):
        with ArtifactStore(self.root / "last-attempt", "build", fsync_every=1) as store:
            task = self.task()
            store.append_group(task["group"])

            def assert_latched():
                self.assertTrue(store.external_pool_exhausted())

            # A sequential drain leases relay-a/relay-b alternately, so each relay serves
            # two attempts per round: the pool only dies on the final attempt of the
            # final durable round, leaving no later round to reroute the task locally.
            relay_a = FakeResponses(
                [ConnectionError("offline-a")] * 5
                + [FakeHttpError(429, "insufficient_quota")]
            )
            relay_b = FakeResponses(
                [ConnectionError("offline-b")] * 5
                + [FakeHttpError(429, "billing_hard_limit_reached")]
            )
            local = FakeResponses(
                [completed_events(model="qwen3_5-35b-a3b")], before_call=assert_latched
            )
            result = ResponsesAnnotator(
                self.config, store,
                client_factory=self.factory({
                    "relay-a": FakeClient(relay_a),
                    "relay-b": FakeClient(relay_b),
                    "local": FakeClient(local),
                }),
                sleep=lambda _: None, random_value=lambda: 0.5,
            ).drain(max_workers=1)

            self.assertEqual(result["completed"], 1)
            self.assertEqual(result["transport_failed"], 0)
            self.assertEqual(result["pending"], 0)
            self.assertEqual(
                (len(relay_a.calls), len(relay_b.calls), len(local.calls)), (6, 6, 1)
            )
            row = next(iter(store.sft.values()))
            self.assertEqual(row["annot_src"], "responses:local")
            self.assertEqual(
                {key: row["qa"]["annotation"][key] for key in ("route", "round", "attempt")},
                {"route": "local", "round": 3, "attempt": 5},
            )
            attempts = [r for r in store.failures if r.get("event_type") == "attempt"]
            self.assertEqual(len(attempts), 12)
            self.assertEqual({r["round"] for r in attempts}, {1, 2, 3})
            latch = [
                r for r in store.failures
                if r.get("error_code") == "external_pool_exhausted"
            ]
            self.assertEqual([(r["round"], r["attempt"]) for r in latch], [(3, 4)])
            self.assertFalse(any(r.get("terminal") for r in store.failures))

    def test_final_round_local_rescue_is_granted_exactly_once_across_resume(self):
        # queue_rounds=1 makes round 1 the final durable round, so the owed local
        # attempt is due in the same round that discovers exhaustion.
        config = dataclasses.replace(self.config, queue_rounds=1)
        out = self.root / "rescue-crash"
        with ArtifactStore(out, "build", fsync_every=1) as store:
            task = self.task()
            store.append_group(task["group"])
            local = FakeResponses([])
            annotator = ResponsesAnnotator(
                config, store,
                client_factory=self.factory({
                    "relay-a": FakeClient(FakeResponses([
                        ConnectionError("offline-a"),
                        FakeHttpError(429, "insufficient_quota"),
                    ])),
                    "relay-b": FakeClient(FakeResponses([
                        ConnectionError("offline-b"),
                        FakeHttpError(429, "billing_hard_limit_reached"),
                    ])),
                    "local": FakeClient(local),
                }),
                sleep=lambda _: None,
            )
            original_append = store.append_failure

            def crash_before_local_rescue(record, *, durable=False):
                appended = original_append(record, durable=durable)
                if record.get("event_type") == "attempt" and record.get("attempt") == 4:
                    raise KeyboardInterrupt("crash before the owed local attempt")
                return appended

            store.append_failure = crash_before_local_rescue
            with self.assertRaisesRegex(KeyboardInterrupt, "owed local attempt"):
                annotator.run_round(task, 1)
            self.assertTrue(store.external_pool_exhausted())
            self.assertEqual(len(local.calls), 0)
            self.assertEqual(len(store.sft), 0)

        with ArtifactStore(out, "build", fsync_every=1) as resumed:
            local = FakeResponses([completed_events(model="qwen3_5-35b-a3b")])
            relay_a = FakeResponses([])
            relay_b = FakeResponses([])
            clients = {
                "relay-a": FakeClient(relay_a),
                "relay-b": FakeClient(relay_b),
                "local": FakeClient(local),
            }
            result = ResponsesAnnotator(
                config, resumed, client_factory=self.factory(clients), sleep=lambda _: None
            ).drain(max_workers=1)
            self.assertEqual(result["completed"], 1)
            self.assertEqual(result["pending"], 0)
            self.assertEqual(len(local.calls), 1)
            self.assertEqual(len(relay_a.calls) + len(relay_b.calls), 0)
            self.assertEqual(len(resumed.sft), 1)
            annotation = next(iter(resumed.sft.values()))["qa"]["annotation"]
            self.assertEqual((annotation["route"], annotation["attempt"]), ("local", 5))

            again = ResponsesAnnotator(
                config, resumed, client_factory=self.factory(clients), sleep=lambda _: None
            ).drain(max_workers=1)
            self.assertEqual(again["completed"], 0)
            self.assertEqual(len(local.calls), 1)
            self.assertEqual(len(resumed.sft), 1)

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
