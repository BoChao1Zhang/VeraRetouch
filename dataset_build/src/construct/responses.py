"""Canonical OpenAI Responses annotation transport and durable queue drain."""
from __future__ import annotations

import base64
import email.utils
import importlib.metadata
import io
import json
import math
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Protocol

from PIL import Image

from dataset_build.responses_events import (
    is_official_response_event,
    validate_response_event_surface,
)
from dataset_build.tools.archive_reader import open_rgb

from .config import (
    AnnotationConfig,
    ExternalEndpointConfig,
    LocalAnnotationConfig,
    redact_text,
    uri_secrets,
)
from .state import ArtifactStore, stable_id


PINNED_OPENAI_VERSION = "2.46.0"
# v4 annotation contract: the six problem/plan sections plus an explicit
# ``region_scope`` declaration between them.  The band slot fails most often
# because its mask is a strip that crosses the whole frame through the subject,
# and a model that is never told so collapses the region to the subject alone.
# ``region_scope`` is the one field where the edit's shape may be stated outright.
# The order is the emission order of the assembled ``reasoning`` string (three
# problems, the scope declaration, three plans, then the two instructions), and a
# strict json_schema generates its properties in exactly this order.  An
# autoregressive model therefore declares how far the edit reaches *before* it
# writes the plans, so every plan is conditioned on the scope rather than the
# other way round.  ``REASONING_FIELDS`` is the first seven, and its order must
# stay identical to ``_SECTION_TOKENS``.
ANNOTATION_FIELDS = (
    "problem_lighting",
    "problem_global_color",
    "problem_specific_color",
    "region_scope",
    "plan_lighting",
    "plan_global_color",
    "plan_specific_color",
    "instruction_long",
    "instruction_short",
)
REASONING_FIELDS = ANNOTATION_FIELDS[:7]
QUOTA_CODES = frozenset({
    "insufficient_quota",
    "billing_hard_limit_reached",
    "billing_not_active",
    "quota_exceeded",
})
# Telemetry some compatible relays prepend outside the Responses event union. It
# carries no response content, so it is skipped rather than rejected as untyped.
IGNORED_STREAM_EVENT_TYPES = frozenset({"codex.rate_limits"})

ANNOTATION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        **{
            name: {"type": "string", "minLength": 8}
            for name in REASONING_FIELDS
        },
        "instruction_long": {"type": "string", "minLength": 8},
        "instruction_short": {"type": "string", "minLength": 4},
    },
    "required": list(ANNOTATION_FIELDS),
    "additionalProperties": False,
}

# A schema violation is the model sampling badly, not the relay refusing the
# request: the same task on the same lane usually parses on the next draw, and
# treating the first one as terminal silently dropped about one winner in twenty.
# It stays bounded so a task the model genuinely cannot answer still terminates
# rather than burning the whole per-round transport budget.  Counted over the
# task's durable attempt events, so a resume inherits what it already spent.
SCHEMA_FAILED = "schema_failed"
SCHEMA_ATTEMPT_LIMIT = 3

# ``provider-b-lane-2`` answers a fraction of requests with a model other than the
# one asked for (~21% measured on 2026-07-28 for gpt-5.6-sol), and the substituted
# model also ignores the strict schema.  An unchecked response makes the dataset an
# unlabelled mixture of two annotators, so the draw is discarded and redrawn on
# another lane.  Bounded exactly like a schema violation, and counted per code so
# neither bad-draw budget can consume the other.
MODEL_SUBSTITUTED = "model_substituted"
MODEL_ATTEMPT_LIMIT = 3

# ``provider-b-lane-1`` wraps a transient upstream fault as `400 {"type":
# "upstream_error"}`.  The blanket "4xx is the client's fault, so it is terminal"
# rule is right for every other 4xx and wrong for this one: WP7 pass 1 lost 12
# tasks to it and they all succeeded on an unchanged replay, which is the
# signature of an upstream blip rather than a malformed request.  It is bounded
# like a bad draw so a genuinely broken request still terminates instead of
# burning the round, and it is *not* a lane rotation: the fault is upstream of
# the lane, so moving lanes buys nothing.
UPSTREAM_ERROR = "upstream_error"
UPSTREAM_ATTEMPT_LIMIT = 3

# Bad draws: the relay answered, but the answer is unusable.  These are redrawn
# immediately (no backoff) and bounded per task per code over durable attempts.
_BAD_DRAW_LIMITS = {
    SCHEMA_FAILED: SCHEMA_ATTEMPT_LIMIT,
    MODEL_SUBSTITUTED: MODEL_ATTEMPT_LIMIT,
    UPSTREAM_ERROR: UPSTREAM_ATTEMPT_LIMIT,
}

# Codes that mean "this lane produced it, ask a different one".  A schema
# violation belongs here on measurement, not principle: all 36 of fresh100's
# schema failures came from provider-b-lane-2 and none from lane-1, so a redraw
# that can land back on the same lane spends the whole budget in the one place
# that cannot answer -- which is how that build lost 5 of 80 rows for good.
_LANE_ROTATING_CODES = frozenset({MODEL_SUBSTITUTED, SCHEMA_FAILED})

# Sampling knobs sent with every external draw.  Deliberately module constants
# rather than ``[annotation]`` keys: the durable ``effective_config`` stays
# byte-identical, so every existing build — eval100 above all — still resumes.
# Both production lanes accept them on ``/v1/responses`` (probed 2026-07-29:
# HTTP 200, requested model returned).  The gateway unmarshals ``temperature``
# into a float64 (a string 500s with a Go unmarshal error) but does not range
# check it (3.0 was accepted), so treat "accepted" as structural, not as proof
# the upstream honours it.  Lane rotation, not sampling, is what actually makes
# a redraw a different draw.
EXTERNAL_SAMPLING: dict[str, Any] = {"temperature": 1.2, "top_p": 1.0}

# Emission order of the assembled ``reasoning`` string, declared once: three
# problems, the region-scope declaration, then three plans.
_SECTION_TOKENS: dict[str, tuple[str, str]] = {
    "problem_lighting": ("<problem_light_start>", "<problem_light_end>"),
    "problem_global_color": ("<problem_globalcolor_start>", "<problem_globalcolor_end>"),
    "problem_specific_color": (
        "<problem_specificcolor_start>", "<problem_specificcolor_end>",
    ),
    "region_scope": ("<region_scope_start>", "<region_scope_end>"),
    "plan_lighting": ("<plan_light_start>", "<plan_light_end>"),
    "plan_global_color": ("<plan_globalcolor_start>", "<plan_globalcolor_end>"),
    "plan_specific_color": ("<plan_specificcolor_start>", "<plan_specificcolor_end>"),
}
# The schema's generation order and the reasoning emission order are the same
# list; a divergence would silently reorder the training text.
assert tuple(_SECTION_TOKENS) == REASONING_FIELDS

# The degenerate region_scope for an edit that has no region: kept verbatim so
# the structure is identical for every task type and a checker can recognise it.
GLOBAL_REGION_SCOPE = "global adjustment across the entire frame"

# Word budgets anchored on the p90 of the samples that passed blind review.  They
# are prompt guidance, not schema: a long instruction is a quality flag for the
# mechanical checker, never a reason to burn a redraw on an otherwise good answer.
INSTRUCTION_WORD_AIM = 60
INSTRUCTION_WORD_CAP = 75
INSTRUCTION_SHORT_WORD_AIM = 22
INSTRUCTION_SHORT_WORD_CAP = 30

_SYSTEM_PROMPT = (
    "You write image-retouching training annotations. Compare the first image "
    "(before) with the second image (after) and return only the requested JSON. "
    "All prose must be English, except that a supplied non-English style name must "
    "be copied verbatim. Describe visible photographic problems and practical plans; "
    "do not mention measurements, masks, preset IDs, or implementation details.\n"
    "region_scope is the one field that states the edited area outright. Write it "
    "as a single line with two parts separated by a semicolon: first identify the "
    "subject the way a viewer would ('subject: the woman in the meadow'), then "
    "declare how far the edit reaches ('edit scope: a diagonal band through the "
    "subject, extending across the background from lower-left to upper-right'). "
    "Plain shape nouns (band, strip, radial falloff, linear gradient), direction "
    "words (horizontal, vertical, diagonal, lower-left, upper-right) and reach "
    "statements ('extends beyond the subject into the background') belong here and "
    f"nowhere else. For an edit that covers the whole picture write exactly: "
    f"{GLOBAL_REGION_SCOPE}\n"
    "In every field, including region_scope, never write a number or a measured "
    "quantity (angles, widths, fractions, percentages, opacities), never name an "
    "editor parameter or a numeric change, never mention a mask, alpha, feathering, "
    "a selection, PCA, a preset or slot identifier, and never quote a quality or "
    "aesthetic score.\n"
    "In the problem and plan fields, never use a shape or a direction to say "
    "which area the edit covers: point at it by naming the scene content that "
    "occupies it. Shape and direction words are still fine there when they "
    "describe the picture itself -- the vertical lines of a building, a bank of "
    "cloud, light that falls off toward the corners -- because that is "
    "describing what is depicted, not pointing at the edited region.\n"
    f"instruction_long is the user's own request: aim for about "
    f"{INSTRUCTION_WORD_AIM} words and never exceed {INSTRUCTION_WORD_CAP}. "
    f"instruction_short condenses it: aim for about {INSTRUCTION_SHORT_WORD_AIM} "
    f"words and never exceed {INSTRUCTION_SHORT_WORD_CAP}. Neither instruction may "
    "use shape or direction vocabulary, name a preset, or state a measurement; both "
    "identify the area purely by what is depicted there."
)


class AnnotationError(RuntimeError):
    """A classified annotation failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool,
        quota: bool = False,
        retry_after: float | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.quota = quota
        self.retry_after = retry_after
        self.status_code = status_code


class ClientFactory(Protocol):
    def __call__(
        self, endpoint: ExternalEndpointConfig | LocalAnnotationConfig, route: str
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class PreparedTask:
    prompt: str
    before_data_url: str
    after_data_url: str


@dataclass(frozen=True, slots=True)
class StreamResult:
    fields: dict[str, str]
    returned_model: str | None
    usage: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AttemptResult:
    fields: dict[str, str]
    route: str
    endpoint_id: str
    returned_model: str | None
    usage: dict[str, Any]
    attempt: int
    round: int


@dataclass(slots=True)
class _RelayState:
    config: ExternalEndpointConfig
    inflight: int = 0
    removed: bool = False


def preflight_openai_sdk() -> None:
    """Fail before mutation unless the pinned SDK exposes typed Responses streaming."""
    try:
        version = importlib.metadata.version("openai")
        from openai import OpenAI
        from openai.types.responses import (
            ResponseCompletedEvent,
            ResponseErrorEvent,
            ResponseFailedEvent,
            ResponseTextDeltaEvent,
        )
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise AnnotationError(
            "openai_sdk_missing",
            f"openai=={PINNED_OPENAI_VERSION} is required",
            retryable=False,
        ) from exc
    if version != PINNED_OPENAI_VERSION:
        raise AnnotationError(
            "openai_sdk_version",
            f"openai=={PINNED_OPENAI_VERSION} is required, found {version}",
            retryable=False,
        )
    client = OpenAI(api_key="preflight", base_url="http://127.0.0.1/v1", max_retries=0)
    if not callable(getattr(client.responses, "create", None)):
        raise AnnotationError(
            "responses_sdk_unavailable", "OpenAI Responses.create is unavailable", retryable=False
        )
    for event_type in (
        ResponseTextDeltaEvent,
        ResponseCompletedEvent,
        ResponseFailedEvent,
        ResponseErrorEvent,
    ):
        if not hasattr(event_type, "model_fields"):
            raise AnnotationError(
                "responses_sdk_unavailable", "typed Responses stream events are unavailable",
                retryable=False,
            )
    try:
        validate_response_event_surface()
    except RuntimeError as exc:
        raise AnnotationError(
            "responses_sdk_unavailable", str(exc), retryable=False
        ) from exc


def assemble_reasoning(parts: Mapping[str, str]) -> str:
    return "".join(
        start + str(parts[name]) + end
        for name, (start, end) in _SECTION_TOKENS.items()
    )


def split_reasoning(reasoning: str) -> dict[str, str]:
    """Read an assembled ``reasoning`` string back into its named sections.

    Sections that are absent are simply missing from the result, so the six-section
    rows written before the v4 contract -- every existing eval100 row -- still read
    back without raising.  ``"region_scope" in split_reasoning(text)`` is the
    variant test.
    """
    sections: dict[str, str] = {}
    for name, (start, end) in _SECTION_TOKENS.items():
        match = re.search(
            f"{re.escape(start)}(.*?){re.escape(end)}", reasoning or "", re.S
        )
        if match is not None:
            sections[name] = match.group(1).strip()
    return sections


def parse_annotation_json(raw: str) -> dict[str, str]:
    """Parse one annotation response, or raise a bounded-retryable schema failure.

    ``SCHEMA_ATTEMPT_LIMIT`` is enforced by the annotator, which is the only place
    that can see how many draws this task has already had.
    """
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AnnotationError(
            SCHEMA_FAILED, "response was not strict JSON", retryable=True
        ) from exc
    if not isinstance(value, dict) or set(value) != set(ANNOTATION_FIELDS):
        raise AnnotationError(
            SCHEMA_FAILED, "response keys did not match the annotation schema",
            retryable=True,
        )
    parsed: dict[str, str] = {}
    for name in ANNOTATION_FIELDS:
        item = value[name]
        minimum = 4 if name == "instruction_short" else 8
        if not isinstance(item, str) or len(item.strip()) < minimum:
            raise AnnotationError(
                SCHEMA_FAILED, f"response field {name} violated minLength",
                retryable=True,
            )
        parsed[name] = item.strip()
    return parsed


def encode_image_data_url(
    path: str | Path, *, longest_edge: int = 768, quality: int = 90
) -> str:
    return _encode_image(path, longest_edge=longest_edge, quality=quality)[0]


def _encode_image(
    path: str | Path, *, longest_edge: int = 768, quality: int = 90
) -> tuple[str, tuple[int, int]]:
    """The data URL plus the orientation-corrected size the model is shown.

    The size is the one the annotator perceives, so it is taken after the EXIF
    transpose and before the aspect-preserving downscale.
    """
    source = Path(path)
    try:
        image = open_rgb(source)
        displayed = image.size
        scale = min(1.0, longest_edge / max(image.size))
        if scale < 1.0:
            size = (
                max(1, round(image.width * scale)),
                max(1, round(image.height * scale)),
            )
            resampling = getattr(Image, "Resampling", Image)
            image = image.resize(size, resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
    except (OSError, ValueError) as exc:
        raise AnnotationError(
            "annotation_image_invalid", f"cannot encode annotation image: {source}",
            retryable=False,
        ) from exc
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + encoded, displayed


# --- v4.1 objective colour-direction hints ---------------------------------
# A revision of the v4a wording, not a new arm: ``PROMPT_VARIANT`` stays "v4a".
#
# What the fresh100 out-of-sample review (2026-07-29) actually showed.  All eight
# of its blind fails transcribed these hints *faithfully* and the reviewer
# disputed the picture anyway, so "the model ignored the hint" was never the
# failure.  Rebuilding all eight from the archived pixels also cleared the
# obvious suspect: the deltas are already alpha-weighted over the mask support
# (``agent.py`` hands ``mask.effective_alpha`` to ``objective_edit_hints``), and
# recomputing them that way reproduced every stored figure to within 0.1.  There
# is no whole-frame dilution here to fix.
#
# What is wrong is the *definition*.  A first-moment spatial mean is not the
# quantity a viewer names.  Three mechanisms, each measured on that panel:
#   * chroma averages over every masked pixel, most of them near-neutral, so a
#     desaturation confined to the few coloured objects vanishes (one fail: mean
#     -0.5, reviewer "chiefly a strong desaturation, especially the woman's red
#     lipstick") or even inverts (another: mean +1.2 -> "moderately richer" ->
#     the text said "richer color" -> reviewer "visibly paler, less saturated").
#   * lightness steals saturation: +6 L* against +1 C* reads as paler, not richer.
#   * clipped highlights drag mean b* and mean C* down while the surviving
#     midtones intensify, so "strongly cooler and muted" described a frame the
#     reviewer called warmer and more saturated.
# Reweighting was tried and rejected on evidence: alpha * max(C_before, C_after)
# repairs two of those fails, softens two, and inverts one the reviewer had
# already scored 5/5.  No reweighting reconciles the two, so v4.1 changes what
# the model is told to do with the figure instead of changing the figure.
#
# Consequence for the wording below: a direction is asserted only when it clears
# a dead band, and when it is asserted it is anchored against its opposite; below
# the band the model is told not to call a direction at all rather than being
# handed the old bare "near-neutral or visually mixed", which read as licence.
_TONAL_DEAD_BAND = 1.0
# b* and C* span far less of a real photograph's range than L* does, and the
# colour axes are the disputed ones.  2.0 was chosen on the fresh100 panel: it
# suppresses exactly the two direction claims the reviewer rejected outright
# (warmth +1.03 written as "warmer" against a visible cyan/green shift; chroma
# +1.22 written as "richer" against a visibly paler result) and costs only some
# precision on two samples the reviewer had already passed.
_COLOUR_DEAD_BAND = 2.0
#
# --- v5 (WP15c): the dead bands are the ROC operating points ----------------
#
# WP15ab re-derived every one of these on the fresh200 n=160 Direction-List
# panel at a 0.90 target precision, so they are no longer a judgement call about
# wording -- they are where the winning metric stops being able to tell the
# judge's verdict apart from noise.  Sources: winning_spec.json ("cut") and
# winning_report.json in /var/cache/veradata/annot_review/wp15_metric_roc/.
#
# They stay module constants and out of the TOML for the same reason the metric
# definitions in visibility.py do: a band is half of what a recorded direction
# means, and a "semantically neutral" config key for it would let two builds
# disagree about what "warmer" was allowed to mean.  Edit here, restart.
_DEAD_BANDS: dict[str, float] = {
    # Unchanged by the ROC: full.d_L at 1.0 already holds 0.94 precision on the
    # holdout (AUC 0.951) and no candidate beat it enough to justify a move.
    "brightness": _TONAL_DEAD_BAND,
    # Unchanged: full.d_C at 2.0, holdout precision 0.93.
    "chroma": _COLOUR_DEAD_BAND,
    # Tightened 2.0 -> 1.2.  The axis changed underneath it: warmth is now the
    # 70 deg projection on the high-alpha core rather than whole-support b*, and
    # the projection's scale is smaller, so the old band silenced the axis
    # almost entirely (9 assertions in 160 rows).
    "warmth": 1.2,
    # New axis, no v4.1 predecessor.  full.d_a is only worth speaking at all
    # well past its noise floor: the WP15b sweep puts the 0.90-precision point
    # at 5.0, and WP14 measured luna as unable to see a* shifts at any of the
    # amplitudes it was probed with, so a narrower band would be pure anchoring.
    "hue_gm": 5.0,
    # Widened 1.0 -> 1.3.  Dropping the clipped pixels raises the magnitude of
    # a genuine contrast move, so the band moves with it.
    "contrast": 1.3,
}
_HINT_OPPOSITE = {
    "brighter": "darker", "darker": "brighter",
    "warmer": "cooler", "cooler": "warmer",
    "richer": "more muted", "more muted": "richer",
    "higher contrast": "lower contrast", "lower contrast": "higher contrast",
    "shifted toward magenta/red": "shifted toward green",
    "shifted toward green": "shifted toward magenta/red",
    # Journals written before WP15c stored the bare tonal words for contrast.
    "higher": "lower", "lower": "higher",
}
_HINT_UNCERTAIN = (
    "the shift is subtle and may not be visible -- do not state a direction either way"
)
# Journal key -> the name the annotator sees.  ``chroma`` is called saturation
# in the prompt because that is the word the Direction List labels use, and
# ``hue_gm`` has no one-word name that is not a colour claim in itself.
_OVERALL_AXES: tuple[tuple[str, str], ...] = (
    ("brightness", "brightness"),
    ("contrast", "contrast"),
    ("saturation", "chroma"),
    ("warmth", "warmth"),
    ("green/magenta", "hue_gm"),
)
_HOW_TO_USE = (
    "Treat the OVERALL lines as a veto, not a script: never assert a direction "
    "they rule out, and never assert one they decline to call.",
    "A listed colour surface is where you are allowed to be specific: name the "
    "scene content carrying that colour and describe what happened to it there, "
    "rather than restating the overall figure. A surface marked low confidence "
    "may be mentioned only as a possibility, and the OVERALL line wins.",
    "With no surface listed, stay at the level the OVERALL lines support: call "
    "the effect mixed rather than claiming every object changes uniformly, and "
    "do not invent a localised colour claim to fill the gap.",
)


def _hint_phrase(name: str, delta: float, direction: str) -> str:
    """One axis, discretised: an anchored direction or an explicit refusal."""
    magnitude = abs(delta)
    if magnitude < _DEAD_BANDS.get(name, _TONAL_DEAD_BAND):
        return _HINT_UNCERTAIN
    strength = "moderately" if magnitude < 4.0 else "strongly"
    opposite = _HINT_OPPOSITE.get(direction)
    if opposite is None:
        return f"{strength} {direction}"
    return f"{strength} {direction} -- do not describe it as {opposite}"


def _axis_line(label: str, name: str, hint: Any) -> str:
    """One OVERALL row, or the explicit silence row."""
    if isinstance(hint, str) and hint.strip():
        # Imported rows carry the direction word with no delta to threshold.
        return f"  {label}: {hint.strip()}"
    if not isinstance(hint, Mapping):
        raise AnnotationError(
            "annotation_task_invalid", f"candidate {name} objective hint is invalid",
            retryable=False,
        )
    try:
        delta = float(hint["delta"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AnnotationError(
            "annotation_task_invalid", f"candidate {name} objective delta is invalid",
            retryable=False,
        ) from exc
    direction = str(hint.get("direction") or "").strip()
    if not direction:
        raise AnnotationError(
            "annotation_task_invalid", f"candidate {name} objective direction is invalid",
            retryable=False,
        )
    return f"  {label}: {_hint_phrase(name, delta, direction)}"


def _surface_line(row: Any) -> str | None:
    """One BY COLOUR SURFACE row; the gate already ran in visibility.py."""
    if not isinstance(row, Mapping):
        return None
    name = str(row.get("name") or "").strip()
    direction = str(row.get("direction") or "").strip()
    try:
        area = float(row.get("area"))
        d_c = float(row.get("d_C"))
        d_l = float(row.get("d_L"))
    except (TypeError, ValueError):
        return None
    if not name or not direction:
        return None
    opposite = _HINT_OPPOSITE.get(direction)
    phrase = f"clearly {direction}" if opposite is None else \
        f"clearly {direction} -- do not describe it as {opposite}"
    line = (f"  {name} surfaces ({round(area * 100)}% of the region, "
            f"chroma {d_c:+.1f}, lightness {d_l:+.1f}): {phrase}")
    if row.get("low_confidence"):
        line += (" [low confidence: the whole-region saturation figure moves the "
                 "other way, and it is the one that wins]")
    return line


def _objective_hints(candidate: Mapping[str, Any]) -> str:
    """The v5 three-block measured-direction table (WP15ab §5).

    Block one states all five axes, silence included, so that "nothing was said
    about warmth" cannot be read as "warmth is free to describe".  Block two is
    the only place a localised colour claim is licensed.  Block three is the
    reading rule, moved in here from the surrounding prompt so that the licence
    travels with the numbers it applies to.
    """
    hints = candidate.get("objective_hints")
    required = ("brightness", "warmth", "chroma", "contrast")
    if not isinstance(hints, Mapping) or any(name not in hints for name in required):
        raise AnnotationError(
            "annotation_task_invalid",
            "candidate objective_hints must contain brightness, warmth, chroma, and contrast",
            retryable=False,
        )
    lines = ["MEASURED EDIT DIRECTIONS (computed from the pixels, not estimated).", ""]
    lines.append("OVERALL, across the whole edit region:")
    for label, name in _OVERALL_AXES:
        # ``hue_gm`` and ``surfaces`` postdate WP15c; a journal written before it
        # simply has four axes and no table, and must still render.
        if name not in hints:
            continue
        lines.append(_axis_line(label, name, hints[name]))

    surfaces = hints.get("surfaces")
    surface_lines = [line for line in
                     (_surface_line(row) for row in (surfaces or []))
                     if line] if isinstance(surfaces, (list, tuple)) else []
    lines.append("")
    lines.append(
        "BY COLOUR SURFACE, using the colours of the BEFORE image (a surface is "
        "listed only when it covers enough of the region and its chroma moved far "
        "enough to be seen):"
    )
    if surface_lines:
        lines.extend(surface_lines)
    else:
        lines.append("  none -- no single colour surface carries a visible chroma change")

    lines.append("")
    lines.append("HOW TO USE:")
    for index, rule in enumerate(_HOW_TO_USE, start=1):
        lines.append(f"  {index}. {rule}")
    return "\n".join(lines)


# --- v4 edit-region geometry hints -----------------------------------------
# Which shape the annotator is told about, and how:
#   "v4a" -- discretised natural-language descriptors (orientation bucket, size
#            bucket, reach statement)
#   "v4b" -- the raw geometry numbers, same reach statement and same wording rule
# Deliberately a module constant and not an ``[annotation]`` key: the durable
# ``effective_config`` must stay byte-identical or eval100 stops resuming, so a
# variant switch is "edit this line, restart" -- exactly how prompt-v2 became v3.
PROMPT_VARIANT = "v4a"

# Only these slots have a geometry to declare.  ``semantic`` already carries the
# subject and is confined to it; global/style edits cover the frame.
_GEOMETRY_MODES = frozenset({"band", "linear", "radial"})

# Orientation of an undirected axis in image coordinates (x right, y *down*), so
# 45 deg runs upper-left to lower-right and 135 deg runs lower-left to upper-right.
# Bucket edges sit halfway between the four named directions.
_AXIS_BUCKETS: tuple[tuple[float, str], ...] = (
    (22.5, "horizontal"),
    (67.5, "diagonal, running from the upper left down to the lower right"),
    (112.5, "vertical"),
    (157.5, "diagonal, running from the lower left up to the upper right"),
    (180.0, "horizontal"),
)
# Where a directed vector points, in the same coordinates, in 45 deg sectors.
_COMPASS = (
    "right edge", "lower-right corner", "bottom edge", "lower-left corner",
    "left edge", "upper-left corner", "top edge", "upper-right corner",
)

_GEOMETRY_WORDING = (
    "Declare that subject and that reach in region_scope. Everywhere else -- both "
    "instructions and all six problem/plan fields -- point at the same area by "
    "naming the scene content that lies along it, never by its shape, its "
    "direction, or any number from this hint."
)


def _geometry_number(
    geometry: Mapping[str, Any], name: str, default: float = 0.0
) -> float:
    """One geometry value, tolerating the ``"+0.42"`` string form of legacy rows."""
    try:
        return float(str(geometry.get(name, default)).lstrip("+"))
    except (TypeError, ValueError):
        return default


def _applies_inside(geometry: Mapping[str, Any]) -> bool:
    """``Flipped`` inverts the rasterised alpha, so true means "edit inside"."""
    return str(geometry.get("Flipped", "false")).strip().lower().lstrip("+") == "true"


def axis_bucket(angle_degrees: float) -> str:
    """Coarse orientation of an undirected axis given in image coordinates."""
    angle = float(angle_degrees) % 180.0
    for bound, name in _AXIS_BUCKETS:
        if angle < bound:
            return name
    return _AXIS_BUCKETS[-1][1]


def compass_bucket(dx: float, dy: float) -> str:
    """Which frame edge or corner a vector points at, in image coordinates."""
    angle = math.degrees(math.atan2(float(dy), float(dx))) % 360.0
    return _COMPASS[int(((angle + 22.5) % 360.0) // 45.0)]


def visual_vector(
    dx: float, dy: float, size: tuple[float, float] | None
) -> tuple[float, float]:
    """A normalised-coordinate direction as it lies on the displayed picture.

    Lightroom geometry lives in normalised coordinates (x = column / width,
    y = row / height), i.e. in a frame squashed to a unit square.  The annotator
    looks at the pixels instead, where the same direction is stretched back by the
    aspect ratio.  A 3:2 photo turns a normalised 45 deg into 33.7 deg on screen,
    which is enough to move it into a different orientation bucket, so every
    direction handed to the model is converted here first.  ``size`` is the
    ``(width, height)`` of the *before* image; ``None`` (or a degenerate size)
    falls back to treating the frame as square, which is the identity.
    """
    dx, dy = float(dx), float(dy)
    if not size:
        return dx, dy
    width, height = float(size[0]), float(size[1])
    if width <= 0.0 or height <= 0.0:
        return dx, dy
    return dx * width, dy * height


def visual_angle(angle_degrees: float, size: tuple[float, float] | None) -> float:
    """``angle_degrees`` measured in normalised coordinates, seen on the picture."""
    theta = math.radians(float(angle_degrees))
    dx, dy = visual_vector(math.cos(theta), math.sin(theta), size)
    return math.degrees(math.atan2(dy, dx))


def _ellipse(geometry: Mapping[str, Any]) -> tuple[float, float, float, float]:
    left, right = _geometry_number(geometry, "Left"), _geometry_number(geometry, "Right")
    top, bottom = _geometry_number(geometry, "Top"), _geometry_number(geometry, "Bottom")
    return (
        (left + right) / 2.0, (top + bottom) / 2.0,
        abs(right - left) / 2.0, abs(bottom - top) / 2.0,
    )


def _linear_vector(geometry: Mapping[str, Any]) -> tuple[float, float]:
    """Direction the edit strengthens in, after ``Flipped`` is applied."""
    dx = _geometry_number(geometry, "FullX", 1.0) - _geometry_number(geometry, "ZeroX")
    dy = _geometry_number(geometry, "FullY") - _geometry_number(geometry, "ZeroY")
    return (-dx, -dy) if _applies_inside(geometry) else (dx, dy)


def _geometry_words(
    slot_mode: str, geometry: Mapping[str, Any], size: tuple[float, float] | None
) -> str:
    inside = _applies_inside(geometry)
    axis = axis_bucket(visual_angle(_geometry_number(geometry, "Angle"), size))
    if slot_mode == "band":
        width = 2.0 * _ellipse(geometry)[3]
        # Edges moved from 0.25/0.50 on the eval100 distribution: only 7 of 140
        # bands fell under 0.25, so "narrow" was a near-dead bucket while the bulk
        # of ordinary bands piled into "broad".
        gauge = "narrow" if width < 0.35 else "moderately wide" if width < 0.6 else "broad"
        shape = (
            f"a {gauge} straight band, {axis}"
            ", that passes through the subject and runs off both edges of the frame"
        )
        reach = (
            "so it covers background on both sides of the subject as well, and the "
            "subject is not the extent of the edit; scene content outside the band "
            "is unchanged"
        ) if inside else (
            "and it is everything outside that band which changes, so most of the "
            "background is affected and the subject is not"
        )
    elif slot_mode == "radial":
        _, _, half_along, half_across = _ellipse(geometry)
        coverage = math.pi * half_along * half_across
        gauge = "tight" if coverage < 0.10 else "moderate" if coverage < 0.35 else "large"
        shape = (
            f"a {gauge} oval falloff centred on the subject, its long axis {axis}"
        )
        reach = (
            "strongest on the subject and fading outward past its outline into the "
            "surrounding scene, so the nearby background changes too; the far "
            "corners of the frame stay unchanged"
        ) if inside else (
            "weakest on the subject and strongest away from it, so it is the "
            "surrounding background that changes"
        )
    else:
        dx, dy = visual_vector(*_linear_vector(geometry), size)
        shape = (
            f"a {axis_bucket(math.degrees(math.atan2(dy, dx)))} linear gradient "
            "spanning the whole frame"
        )
        reach = (
            f"strongest toward the {compass_bucket(dx, dy)}, where the subject sits, "
            "and fading continuously to nothing toward the opposite side, so the "
            "background on the strong side changes as much as the subject does"
        )
    return f"the edited area is {shape}, {reach}."


# v4b's numbers are self-consistent only if each one says which frame it is in.
# Positions and extents stay normalised -- a point at (0.26, 0.58) is the same
# point of the picture whatever the aspect ratio, so normalised is exactly what a
# "fraction of the frame" means.  An angle is not: 45 deg in normalised
# coordinates is 33.7 deg on a 3:2 print, and the model reads a degree figure as
# the tilt it can see.  So ``axis_angle`` is converted to the visual angle and the
# preamble says which convention each kind of number follows.
_V4B_FRAME_NOTE = (
    "the edited area is defined by these frame coordinates: x runs left to right "
    "and y runs top to bottom, each spanning 0 to 1, so every position and every "
    "extent below is a fraction of the frame's own width or height"
)
_V4B_ANGLE_NOTE = (
    "; axis_angle is the one exception, measured on the picture as you actually "
    "see it, in degrees clockwise from horizontal"
)


def _geometry_numbers(
    slot_mode: str, geometry: Mapping[str, Any], size: tuple[float, float] | None
) -> str:
    flipped = _applies_inside(geometry)
    if slot_mode == "linear":
        body = (
            "shape=linear_gradient; "
            f"zero_end=({_geometry_number(geometry, 'ZeroX'):.3f}, "
            f"{_geometry_number(geometry, 'ZeroY'):.3f}); "
            f"full_end=({_geometry_number(geometry, 'FullX', 1.0):.3f}, "
            f"{_geometry_number(geometry, 'FullY'):.3f}); "
            f"flipped={'true' if flipped else 'false'}; strength runs from "
            + ("full at the zero end to none at the full end" if flipped
               else "none at the zero end to full at the full end")
        )
        preamble = _V4B_FRAME_NOTE
    else:
        cx, cy, half_along, half_across = _ellipse(geometry)
        angle = visual_angle(_geometry_number(geometry, "Angle"), size)
        body = (
            f"shape={'band' if slot_mode == 'band' else 'ellipse'}; "
            f"axis_angle={angle % 180.0:.2f} degrees; "
            f"center=({cx:.3f}, {cy:.3f}); half_axis_along={half_along:.3f}; "
            f"half_axis_across={half_across:.3f}; "
            f"feather={_geometry_number(geometry, 'Feather', 50.0):.0f}; the edit "
            f"applies {'inside' if flipped else 'outside'} this shape"
        )
        preamble = _V4B_FRAME_NOTE + _V4B_ANGLE_NOTE
    return (
        preamble + ": " + body + ". It reaches past the "
        "subject and into the surrounding background."
    )


def edit_geometry_hint(
    candidate: Mapping[str, Any],
    *,
    variant: str | None = None,
    size: tuple[float, float] | None = None,
) -> str | None:
    """The edit-region geometry clause for one candidate, or ``None``.

    ``None`` means there is no geometry to declare: ``semantic`` slots are confined
    to the subject the prompt already names, and global/style tasks cover the frame.

    ``size`` is the ``(width, height)`` of the before image the model is shown; it
    turns the stored normalised angles into the angles that are visible there.
    Omitting it treats the frame as square, which is what every square image gets
    anyway and the only honest fallback when the size cannot be read.
    """
    slot_mode = str(candidate.get("slot_mode") or "").strip().lower()
    geometry = candidate.get("geometry")
    if slot_mode not in _GEOMETRY_MODES or not isinstance(geometry, Mapping):
        return None
    chosen = str(variant or PROMPT_VARIANT).strip().lower()
    body = (
        _geometry_numbers(slot_mode, geometry, size) if chosen == "v4b"
        else _geometry_words(slot_mode, geometry, size)
    )
    return "Edit-region geometry, for your understanding only: " + body + " " + \
        _GEOMETRY_WORDING


def _subject_name(value: Any) -> str:
    if isinstance(value, Mapping):
        for key in ("name", "subject_name", "label", "class_name"):
            if str(value.get(key) or "").strip():
                return str(value[key]).strip()
        return "subject"
    return str(value or "subject").strip() or "subject"


def build_prompt(
    task: Mapping[str, Any], *, before_size: tuple[float, float] | None = None
) -> str:
    group = task.get("group")
    candidate = task.get("candidate")
    if not isinstance(group, Mapping) or not isinstance(candidate, Mapping):
        raise AnnotationError(
            "annotation_task_invalid", "annotation task is missing group/candidate", retryable=False
        )
    hints = _objective_hints(candidate)
    mode = str(group.get("render_mode") or "").lower()
    geometry_hint: str | None = None
    if mode == "global":
        style_name = str(candidate.get("style_name") or "").strip()
        if not style_name:
            raise AnnotationError(
                "annotation_task_invalid", "global candidate is missing style_name",
                retryable=False,
            )
        task_clause = (
            f'This is a global style task. Both instructions must name the style "{style_name}" '
            "verbatim and describe how to reproduce the visible overall look. "
            f'The edit covers the whole picture, so region_scope is exactly "{GLOBAL_REGION_SCOPE}".'
        )
    elif mode == "local":
        subject = _subject_name(candidate.get("subject") or group.get("subject"))
        region = str(candidate.get("region") or "").strip()
        slot_mode = str(candidate.get("slot_mode") or "").strip().lower()
        if not region:
            raise AnnotationError(
                "annotation_task_invalid", "local candidate is missing coarse region",
                retryable=False,
            )
        if slot_mode in _GEOMETRY_MODES:
            task_clause = (
                f"This is a local task with a regional edit affecting visible image content around the "
                f'{region}. The edited area may include "{subject}" and neighboring scene '
                "elements. Name the coarse image region and the actual scene content that "
                "changes. Do not claim the edit is confined to that subject or that nearby "
                "background stays unchanged. Do not name a preset, style, or selection mechanism."
            )
            geometry_hint = edit_geometry_hint(candidate, size=before_size)
        else:
            # ``semantic`` is confined to the subject; legacy imports predate the
            # canonical slot-mode vocabulary and their subject metadata remains
            # authoritative for reannotation.  Neither has a geometry to declare.
            task_clause = (
                f'This is a local task affecting the subject "{subject}" in the {region}. '
                "Describe that subject/region without naming a preset or style. In "
                "region_scope, identify that subject and state that the edit stays "
                "within it."
            )
    else:
        raise AnnotationError(
            "annotation_task_invalid", "render_mode must be local or global", retryable=False
        )
    return (
        "The images are ordered before, then after. " + task_clause + "\n"
        + (geometry_hint + "\n" if geometry_hint else "")
        + hints + "\n"
        "Fill all nine schema fields. The six problem/plan fields must be substantive, and "
        "region_scope must identify the subject and declare how far the edit reaches."
    )


def prepare_task(task: Mapping[str, Any], config: AnnotationConfig) -> PreparedTask:
    group = task.get("group")
    candidate = task.get("candidate")
    if not isinstance(group, Mapping) or not isinstance(candidate, Mapping):
        raise AnnotationError(
            "annotation_task_invalid", "annotation task is missing group/candidate", retryable=False
        )
    before = group.get("source_path")
    after = candidate.get("after_path")
    if not before or not after:
        raise AnnotationError(
            "annotation_task_invalid", "annotation task is missing before/after paths",
            retryable=False,
        )
    # The before image is encoded first so its displayed size can steer the
    # geometry hint: the orientation buckets are only correct once the stored
    # normalised angles are read back in the aspect ratio the model sees.
    before_url, before_size = _encode_image(
        str(before), longest_edge=config.image_long_edge,
        quality=config.image_jpeg_quality,
    )
    return PreparedTask(
        prompt=build_prompt(task, before_size=before_size),
        before_data_url=before_url,
        after_data_url=encode_image_data_url(
            str(after), longest_edge=config.image_long_edge,
            quality=config.image_jpeg_quality,
        ),
    )


def request_payload(
    prepared: PreparedTask,
    config: AnnotationConfig,
    *,
    route: str,
) -> dict[str, Any]:
    content = [
        {"type": "input_text", "text": prepared.prompt},
        {"type": "input_image", "image_url": prepared.before_data_url},
        {"type": "input_image", "image_url": prepared.after_data_url},
    ]
    payload: dict[str, Any] = {
        "instructions": _SYSTEM_PROMPT,
        "input": [{"role": "user", "content": content}],
        "stream": True,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "veraretouch_annotation",
                "strict": True,
                "schema": ANNOTATION_JSON_SCHEMA,
            }
        },
    }
    if route == "external":
        payload.update({
            "model": config.external_model,
            "reasoning": {"effort": config.external_reasoning_effort},
            "max_output_tokens": config.external_max_output_tokens,
            **EXTERNAL_SAMPLING,
        })
    elif route == "local":
        payload.update({
            "model": config.local.model,
            "temperature": config.local.temperature,
            "max_output_tokens": config.local.max_output_tokens,
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": config.local.enable_thinking,
                }
            },
        })
    else:
        raise ValueError(f"unsupported annotation route: {route}")
    return payload


def _response_usage(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        result = dump(mode="json", exclude_none=True)
        return dict(result) if isinstance(result, Mapping) else {}
    return {}


def _consume_stream(stream: Any) -> StreamResult:
    from openai.types.responses import (
        ResponseCompletedEvent,
        ResponseErrorEvent,
        ResponseFailedEvent,
        ResponseTextDeltaEvent,
    )

    chunks: list[str] = []
    completed: Any = None
    context = stream if hasattr(stream, "__enter__") else _null_context(stream)
    with context as events:
        for event in events:
            if getattr(event, "type", None) in IGNORED_STREAM_EVENT_TYPES:
                continue
            if not is_official_response_event(event):
                raise AnnotationError(
                    "untyped_stream_event", "Responses stream returned an untyped event",
                    retryable=True,
                )
            if isinstance(event, ResponseTextDeltaEvent):
                chunks.append(event.delta)
            elif isinstance(event, ResponseCompletedEvent):
                completed = event.response
            elif isinstance(event, ResponseFailedEvent):
                error = getattr(event.response, "error", None)
                code = str(getattr(error, "code", None) or "response_failed")
                message = str(getattr(error, "message", None) or "Responses stream failed")
                raise AnnotationError(
                    code, message, retryable=True, quota=code in QUOTA_CODES
                )
            elif isinstance(event, ResponseErrorEvent):
                code = str(event.code or "response_error")
                raise AnnotationError(
                    code, event.message, retryable=True, quota=code in QUOTA_CODES
                )
    if completed is None:
        raise AnnotationError(
            "stream_interrupted", "Responses stream ended before response.completed",
            retryable=True,
        )
    status = getattr(completed, "status", None)
    if status not in (None, "completed"):
        raise AnnotationError(
            "response_not_completed", f"Responses status was {status}", retryable=True
        )
    raw = "".join(chunks)
    if not raw:
        raw = str(getattr(completed, "output_text", "") or "")
    fields = parse_annotation_json(raw)
    return StreamResult(
        fields=fields,
        returned_model=str(getattr(completed, "model", "") or "") or None,
        usage=_response_usage(getattr(completed, "usage", None)),
    )


@contextmanager
def _null_context(value: Any) -> Iterator[Any]:
    yield value


def _retry_after(headers: Any) -> float | None:
    if not headers:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        try:
            parsed = email.utils.parsedate_to_datetime(str(raw))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def _error_code(exc: BaseException) -> str:
    direct = getattr(exc, "code", None)
    if direct:
        return str(direct)
    body = getattr(exc, "body", None)
    if isinstance(body, Mapping):
        nested = body.get("error")
        if isinstance(nested, Mapping) and nested.get("code"):
            return str(nested["code"])
        if body.get("code"):
            return str(body["code"])
    return ""


def _is_upstream_error(exc: BaseException) -> bool:
    """Does this 4xx body say the *upstream* failed rather than the request?

    The relay reports it as a top-level ``type``; the SDK's own error shape puts
    the same discriminator under ``error``, and some gateways fill ``code``
    instead.  All three are read so the classification does not depend on which
    envelope a given lane happens to use.
    """
    body = getattr(exc, "body", None)
    candidates = [body]
    if isinstance(body, Mapping):
        candidates.append(body.get("error"))
    for entry in candidates:
        if not isinstance(entry, Mapping):
            continue
        if any(str(entry.get(key) or "") == UPSTREAM_ERROR for key in ("type", "code")):
            return True
    return _error_code(exc) == UPSTREAM_ERROR


def classify_exception(exc: BaseException) -> AnnotationError:
    if isinstance(exc, AnnotationError):
        return exc
    status = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    retry_after = _retry_after(getattr(response, "headers", None))
    code = _error_code(exc)
    quota = code in QUOTA_CODES
    message = str(exc) or type(exc).__name__
    if quota:
        return AnnotationError(
            code, message, retryable=True, quota=True, retry_after=retry_after,
            status_code=status,
        )
    if status == 429:
        return AnnotationError(
            code or "rate_limited", message, retryable=True, retry_after=retry_after,
            status_code=status,
        )
    if isinstance(status, int) and status >= 500:
        return AnnotationError(
            code or "upstream_5xx", message, retryable=True, retry_after=retry_after,
            status_code=status,
        )
    if isinstance(status, int) and 400 <= status < 500:
        if _is_upstream_error(exc):
            return AnnotationError(
                UPSTREAM_ERROR, message, retryable=True, status_code=status
            )
        return AnnotationError(
            code or "request_4xx", message, retryable=False, status_code=status
        )
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)) or type(exc).__name__ in {
        "APIConnectionError", "APITimeoutError",
    }:
        return AnnotationError("network_error", message, retryable=True)
    return AnnotationError("transport_error", message, retryable=True)


class ExternalRelayPool:
    """Thread-safe least-inflight pool with round-robin tie breaking."""

    def __init__(
        self,
        endpoints: tuple[ExternalEndpointConfig, ...],
        *,
        removed_ids: set[str] | None = None,
        exhausted: bool = False,
    ) -> None:
        self._states = [_RelayState(endpoint) for endpoint in endpoints]
        removed = removed_ids or set()
        for state in self._states:
            state.removed = exhausted or state.config.id in removed
        self._condition = threading.Condition(threading.Lock())
        self._cursor = 0

    @property
    def exhausted(self) -> bool:
        with self._condition:
            return not any(not state.removed for state in self._states)

    @property
    def active_ids(self) -> tuple[str, ...]:
        with self._condition:
            return tuple(state.config.id for state in self._states if not state.removed)

    def remove(self, endpoint_id: str) -> bool:
        """Remove one endpoint and return whether the pool is now exhausted."""
        with self._condition:
            for state in self._states:
                if state.config.id == endpoint_id:
                    state.removed = True
                    break
            self._condition.notify_all()
            return not any(not state.removed for state in self._states)

    @contextmanager
    def lease(
        self, *, avoid: str | Iterable[str] | None = None
    ) -> Iterator[ExternalEndpointConfig | None]:
        state = self._acquire(avoid)
        if state is None:
            yield None
            return
        try:
            yield state.config
        finally:
            with self._condition:
                state.inflight -= 1
                self._condition.notify_all()

    def _acquire(
        self, avoid: str | Iterable[str] | None = None
    ) -> _RelayState | None:
        # A set rather than the single id this used to take: a task that has been
        # spoiled by two different lanes must be able to skip both, otherwise the
        # third draw walks straight back onto the first one.
        if avoid is None:
            excluded: frozenset[str] = frozenset()
        elif isinstance(avoid, str):
            excluded = frozenset({avoid})
        else:
            excluded = frozenset(str(entry) for entry in avoid)
        with self._condition:
            while True:
                active = [state for state in self._states if not state.removed]
                if not active:
                    return None
                # A lane that just spoiled the draw is skipped so it cannot
                # capture the same task every time.  When the excluded set covers
                # every surviving lane there is nowhere else to go, and one more
                # draw there beats failing the task outright.
                candidates = [
                    state for state in active if state.config.id not in excluded
                ] or active
                available = [
                    state for state in candidates
                    if state.inflight < state.config.concurrency
                ]
                if not available:
                    self._condition.wait()
                    continue
                minimum = min(state.inflight for state in available)
                tied = {id(state) for state in available if state.inflight == minimum}
                selected: _RelayState | None = None
                for offset in range(len(self._states)):
                    index = (self._cursor + offset) % len(self._states)
                    if id(self._states[index]) in tied:
                        selected = self._states[index]
                        self._cursor = (index + 1) % len(self._states)
                        break
                assert selected is not None
                selected.inflight += 1
                return selected


class ResponsesAnnotator:
    """Official-SDK transport plus durable three-round annotation drain."""

    def __init__(
        self,
        config: AnnotationConfig,
        store: ArtifactStore,
        *,
        client_factory: ClientFactory | None = None,
        sleep: Callable[[float], None] = time.sleep,
        random_value: Callable[[], float] = random.random,
    ) -> None:
        self.config = config
        self.store = store
        self._client_factory = client_factory or self._default_client
        self._sleep = sleep
        self._random_value = random_value
        self._client_lock = threading.Lock()
        self._pool_state_lock = threading.Lock()
        self._clients: dict[str, Any] = {}
        self._secrets = tuple(
            secret
            for endpoint in config.external_endpoints
            for secret in (endpoint.api_key, *uri_secrets(endpoint.base_url))
        ) + (config.local.api_key, *uri_secrets(config.local.base_url))
        removed = {
            str(row.get("endpoint_id"))
            for row in store.failures
            if row.get("error_code") == "external_endpoint_exhausted"
        }
        self.pool = ExternalRelayPool(
            config.external_endpoints,
            removed_ids=removed,
            exhausted=store.external_pool_exhausted(),
        )

    @staticmethod
    def _default_client(
        endpoint: ExternalEndpointConfig | LocalAnnotationConfig, route: str
    ) -> Any:
        from openai import OpenAI

        return OpenAI(
            api_key=endpoint.api_key,
            base_url=endpoint.base_url,
            max_retries=0,
            timeout=180.0,
            default_headers={"X-vgate-class": "build-annotate"},
        )

    def _client(self, endpoint: ExternalEndpointConfig | LocalAnnotationConfig, route: str) -> Any:
        key = endpoint.id if isinstance(endpoint, ExternalEndpointConfig) else "local"
        with self._client_lock:
            if key not in self._clients:
                self._clients[key] = self._client_factory(endpoint, route)
            return self._clients[key]

    def _request(
        self,
        prepared: PreparedTask,
        endpoint: ExternalEndpointConfig | LocalAnnotationConfig,
        route: str,
    ) -> StreamResult:
        client = self._client(endpoint, route)
        payload = request_payload(prepared, self.config, route=route)
        try:
            stream = client.responses.create(**payload)
            result = _consume_stream(stream)
            # Both production lanes echo the requested id verbatim (probed
            # 2026-07-29: "gpt-5.6-luna"), and every substitution seen so far
            # answered with an unrelated family (gpt-5.5 for a gpt-5.6-* ask).  A
            # prefix therefore accepts a dated variant of the model asked for and
            # rejects a different one.  A relay that echoes nothing proves nothing,
            # so it is left to the schema check rather than blamed for substitution.
            # Only the external lanes are policed.  Local vLLM answers with the name
            # it was served under, which need not be the id the config asks for, and
            # there a false verdict is lethal rather than merely wasteful: local has
            # no second lane to rotate to and it is reached as the one rescue draw
            # after the pool is exhausted, so it would burn the task's last chance on
            # an answer that was perfectly good.
            requested = str(payload["model"])
            returned = result.returned_model
            if route == "external" and returned and not returned.startswith(requested):
                raise AnnotationError(
                    MODEL_SUBSTITUTED,
                    f"requested model {requested} but the relay answered as {returned}",
                    retryable=True,
                )
            return result
        except BaseException as exc:
            raise classify_exception(exc) from exc

    def _failure(
        self,
        task: Mapping[str, Any],
        *,
        event_type: str,
        error_code: str,
        message: str,
        round_number: int,
        attempt: int,
        retryable: bool,
        terminal: bool,
        endpoint_id: str | None = None,
        durable: bool = False,
    ) -> bool:
        task_id = str(task["task_id"])
        event_id = stable_id(
            "failure", self.store.build_id, task_id, event_type, error_code,
            round_number, attempt, endpoint_id or "",
        )
        return self.store.append_failure({
            "build_id": self.store.build_id,
            "event_id": event_id,
            "event_type": event_type,
            "stage": "annotation",
            "task_id": task_id,
            "group_id": task.get("group_id"),
            "candidate_id": task.get("candidate_id"),
            "round": round_number,
            "attempt": attempt,
            "retryable": retryable,
            "error_code": error_code,
            "message": redact_text(message, self._secrets),
            "endpoint_id": endpoint_id,
            "terminal": terminal,
        }, durable=durable)

    def _persist_quota_removal(
        self,
        task: Mapping[str, Any],
        endpoint_id: str,
        round_number: int,
        attempt: int,
        exhausted: bool,
    ) -> None:
        with self._pool_state_lock:
            endpoint_recorded = any(
                row.get("error_code") == "external_endpoint_exhausted"
                and row.get("endpoint_id") == endpoint_id
                for row in self.store.failures
            )
            if exhausted and not self.store.external_pool_exhausted():
                self._failure(
                    task,
                    event_type="pool_state",
                    error_code="external_pool_exhausted",
                    message="all external annotation endpoints exhausted permanent quota",
                    round_number=round_number,
                    attempt=attempt,
                    retryable=True,
                    terminal=False,
                    endpoint_id=endpoint_id,
                    durable=True,
                )
            if not endpoint_recorded:
                self._failure(
                    task,
                    event_type="pool_state",
                    error_code="external_endpoint_exhausted",
                    message=f"external endpoint {endpoint_id} exhausted its permanent quota",
                    round_number=round_number,
                    attempt=attempt,
                    retryable=True,
                    terminal=False,
                    endpoint_id=endpoint_id,
                    durable=True,
                )

    def _append_sft(self, task: Mapping[str, Any], result: AttemptResult) -> None:
        group = task["group"]
        candidate = task["candidate"]
        mode = str(group["render_mode"])
        qa = dict(candidate.get("qa") or {})
        qa["annotation"] = {
            "route": result.route,
            "endpoint_id": result.endpoint_id,
            "returned_model": result.returned_model,
            "usage": result.usage,
            "attempt": result.attempt,
            "round": result.round,
            "status": "completed",
        }
        local: dict[str, Any] | None = None
        if mode == "local":
            local = {
                "slot_mode": candidate.get("slot_mode"),
                "mode_index": candidate.get("mode_index"),
                "mask_id": candidate.get("mask_id"),
                "C_GT": candidate.get("cgt_path"),
                "subject": candidate.get("subject") or group.get("subject"),
                "region": candidate.get("region"),
                "raw_alpha_mean": candidate.get("raw_alpha_mean"),
                "amount": candidate.get("amount"),
                "effective_alpha_mean": candidate.get("effective_alpha_mean"),
                "pairing_index": candidate.get("pairing_index"),
            }
        task_id = str(task["task_id"])
        self.store.append_sft({
            "build_id": self.store.build_id,
            "sft_id": stable_id("sft", task_id),
            "annotation_task_id": task_id,
            "group_id": task["group_id"],
            "candidate_id": task["candidate_id"],
            "winner_rank": task["winner_rank"],
            # How well the ranking separated this winner from rank2 when it was
            # chosen: "normal", or "low" for a winner inside the resolution limit
            # WP5 measured.  An audit/filter field like ``annot_model`` — rows
            # written before it simply lack it, and every reader uses ``get``.
            "winner_confidence": task.get("winner_confidence"),
            "I_in": group["source_path"],
            "I_tar": candidate["after_path"],
            "recipe": candidate.get("recipe") or candidate.get("preset_id"),
            "local": local,
            "task_type": "local" if mode == "local" else "style",
            "instruction": result.fields["instruction_long"],
            "instruction_short": result.fields["instruction_short"],
            "reasoning": assemble_reasoning(result.fields),
            "annot_src": (
                f"responses:external:{result.endpoint_id}"
                if result.route == "external" else "responses:local"
            ),
            # The model that actually answered, checked against the requested one
            # before this row existed.  Purely an audit field: rows written before
            # it simply do not carry it, and every reader uses ``get``.
            "annot_model": result.returned_model,
            "qa": qa,
        })
        self.store.checkpoint()

    def _attempt_count(self, task_id: str, round_number: int) -> int:
        return sum(
            1 for row in self.store.failures
            if row.get("task_id") == task_id
            and row.get("stage") == "annotation"
            and row.get("event_type") == "attempt"
            and row.get("round") == round_number
        )

    def _draw_attempt_count(self, task_id: str, code: str) -> int:
        """Bad draws of one kind this task has already spent, over every round."""
        return sum(
            1 for row in self.store.failures
            if row.get("task_id") == task_id
            and row.get("stage") == "annotation"
            and row.get("event_type") == "attempt"
            and row.get("error_code") == code
        )

    def _local_attempted(self, task_id: str, round_number: int) -> bool:
        return any(
            row.get("task_id") == task_id
            and row.get("stage") == "annotation"
            and row.get("event_type") == "attempt"
            and row.get("round") == round_number
            and row.get("endpoint_id") == "local"
            for row in self.store.failures
        )

    def _local_rescue_due(self, task_id: str, round_number: int) -> bool:
        """Report whether a pool-exhausted task still owes one local attempt.

        A task that meets external pool exhaustion stays pending and is rerouted to
        local vLLM: inside the current round while an attempt remains, otherwise at the
        start of its next durable round. In the final round there is no next round, so
        the owed local attempt is granted beyond the per-round transport budget instead
        of letting the task reach ``transport_failed`` without ever trying local. The
        predicate is rebuilt from persisted attempt events, so resume grants the same
        single extra attempt exactly once.
        """
        return (
            self.config.local_fallback
            and round_number >= self.config.queue_rounds
            and self.pool.exhausted
            and not self._local_attempted(task_id, round_number)
        )

    def _round_done(self, task_id: str, round_number: int) -> bool:
        return any(
            row.get("task_id") == task_id
            and row.get("event_type") == "round_exhausted"
            and row.get("round") == round_number
            for row in self.store.failures
        )

    def _next_round(self, task_id: str) -> int | None:
        if self.store.has_terminal_failure(task_id):
            return None
        for round_number in range(1, self.config.queue_rounds + 1):
            if not self._round_done(task_id, round_number):
                return round_number
        return None

    def run_round(self, task: Mapping[str, Any], round_number: int) -> str:
        task_id = str(task["task_id"])
        attempt = self._attempt_count(task_id, round_number)
        # Every lane that has already spoiled a draw of this task, so the redraw
        # leaves all of them.  It accumulates and is never cleared inside the
        # round: a lane that answered with the wrong model or unparseable JSON is
        # still the wrong place to go after some *other* lane times out.
        avoid_endpoints: set[str] = set()
        try:
            prepared = prepare_task(task, self.config)
        except AnnotationError as failure:
            self._failure(
                task, event_type="terminal", error_code=failure.code,
                message=str(failure), round_number=round_number, attempt=attempt,
                retryable=False, terminal=True, durable=True,
            )
            return "terminal"

        while True:
            if self.pool.exhausted and not self.config.local_fallback:
                self._failure(
                    task,
                    event_type="skipped",
                    error_code="external_pool_exhausted_skip",
                    message="external pool is exhausted and local fallback is disabled",
                    round_number=round_number,
                    attempt=attempt,
                    retryable=False,
                    terminal=False,
                    durable=True,
                )
                return "retryable_exhausted"
            if attempt >= self.config.transport_attempts_per_round \
                    and not self._local_rescue_due(task_id, round_number):
                return "retryable_exhausted"
            route = "local" if self.pool.exhausted else "external"
            endpoint: ExternalEndpointConfig | LocalAnnotationConfig | None
            lease = (
                self.pool.lease(avoid=avoid_endpoints) if route == "external"
                else _null_context(self.config.local)
            )
            with lease as endpoint:
                if endpoint is None:
                    continue
                attempt += 1
                endpoint_id = endpoint.id if isinstance(endpoint, ExternalEndpointConfig) else "local"
                try:
                    stream_result = self._request(prepared, endpoint, route)
                except AnnotationError as failure:
                    if failure.quota and route == "external":
                        exhausted = self.pool.remove(endpoint_id)
                        self._persist_quota_removal(
                            task, endpoint_id, round_number, attempt, exhausted
                        )
                    self._failure(
                        task,
                        event_type="attempt",
                        error_code=failure.code,
                        message=str(failure),
                        round_number=round_number,
                        attempt=attempt,
                        retryable=failure.retryable,
                        terminal=False,
                        endpoint_id=endpoint_id,
                        durable=True,
                    )
                    # The attempt event above is already durable, so this count
                    # includes the draw that just failed and survives a resume.
                    draw_limit = _BAD_DRAW_LIMITS.get(failure.code)
                    spent_draws = (
                        draw_limit is not None
                        and self._draw_attempt_count(task_id, failure.code) >= draw_limit
                    )
                    if failure.code in _LANE_ROTATING_CODES and route == "external":
                        avoid_endpoints.add(endpoint_id)
                    if not failure.retryable or spent_draws:
                        self._failure(
                            task,
                            event_type="terminal",
                            error_code=failure.code,
                            message=str(failure),
                            round_number=round_number,
                            attempt=attempt,
                            retryable=False,
                            terminal=True,
                            endpoint_id=endpoint_id,
                            durable=True,
                        )
                        return "terminal"
                    # A bad draw is not congestion, so the next one is taken
                    # immediately; backoff still governs every transport failure it
                    # was written for.
                    if attempt < self.config.transport_attempts_per_round \
                            and not failure.quota and draw_limit is None:
                        delay = failure.retry_after
                        if delay is None:
                            delay = min(60.0, 2.0 ** (attempt - 1)) * (
                                0.75 + 0.5 * self._random_value()
                            )
                        self._sleep(delay)
                    continue
            result = AttemptResult(
                fields=stream_result.fields,
                route=route,
                endpoint_id=endpoint_id,
                returned_model=stream_result.returned_model,
                usage=stream_result.usage,
                attempt=attempt,
                round=round_number,
            )
            self._append_sft(task, result)
            return "completed"

    def drain(self, *, max_workers: int | None = None) -> dict[str, int]:
        workers = max_workers or max(
            1, sum(endpoint.concurrency for endpoint in self.config.external_endpoints)
        )
        counts = {"completed": 0, "terminal": 0, "transport_failed": 0}
        for round_number in range(1, self.config.queue_rounds + 1):
            tasks = [
                task for task in self.store.pending_annotation_tasks()
                if self._next_round(str(task["task_id"])) == round_number
            ]
            if not tasks:
                continue
            with ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
                future_to_task = {
                    executor.submit(self.run_round, task, round_number): task
                    for task in tasks
                }
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        status = future.result()
                    except BaseException as exc:
                        failure = classify_exception(exc)
                        self._failure(
                            task,
                            event_type="terminal",
                            error_code="annotation_worker_failed",
                            message=str(failure),
                            round_number=round_number,
                            attempt=self._attempt_count(str(task["task_id"]), round_number),
                            retryable=False,
                            terminal=True,
                            durable=True,
                        )
                        counts["terminal"] += 1
                        continue
                    if status == "completed":
                        counts["completed"] += 1
                    elif status == "terminal":
                        counts["terminal"] += 1
                    elif round_number < self.config.queue_rounds:
                        self._failure(
                            task,
                            event_type="round_exhausted",
                            error_code="annotation_round_exhausted",
                            message="annotation transport attempts exhausted for queue round",
                            round_number=round_number,
                            attempt=self.config.transport_attempts_per_round,
                            retryable=True,
                            terminal=False,
                            durable=True,
                        )
                    else:
                        self._failure(
                            task,
                            event_type="terminal",
                            error_code="transport_failed",
                            message="annotation transport failed after all durable rounds",
                            round_number=round_number,
                            attempt=self.config.transport_attempts_per_round,
                            retryable=False,
                            terminal=True,
                            durable=True,
                        )
                        counts["transport_failed"] += 1
        self.store.checkpoint()
        counts["pending"] = len(self.store.pending_annotation_tasks())
        return counts
