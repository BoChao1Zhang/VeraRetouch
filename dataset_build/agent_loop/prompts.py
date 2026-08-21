"""Immutable prompt families, strict schemas, and canonical serialization."""
from __future__ import annotations

import json
import hashlib
from typing import Any, Mapping, Sequence

from .api_cache import RequestSpec, prefix_cache_key
from .artifacts import ArtifactRef
from .config import EndpointConfig
from .direction_match import (
    AXIS_SCALES, AXIS_WEIGHTS, KEYWORD_AXES, MEASURE_SAMPLE_PIXELS,
)
from .models import (
    ACTIVE_LOCAL_INTENTS, BAND_GEOMETRY_GATE, FINGERPRINT_FIELDS, FINGERPRINT_FORMATS,
    GLOBAL_DELTA_E_TARGETS, INTENT_BIN_QUOTA, INTENT_DIRECTION_GATE,
    INTENT_FINGERPRINT_GATE, INTENT_PACKET_PRIORITY,
    INTENT_ROLE_DOMAINS, INTENT_SEGMENT_GATE,
    LOCAL_DELTA_E_LADDERS, LOCAL_INTENT_LADDER, LOCAL_INTENTS,
    LOCAL_ONLINE_RETRIEVAL, LOCAL_PACKET_ROW_LIMIT, LOCAL_VISIBILITY_FLOOR,
    MASK_REACH_GATE, REASON_CODES, SUBJECT_HEADROOM_GATE, intent_packet_order,
)
from .segment_fingerprints import (
    BAND_SEGMENT_LUM_THRESHOLD, HISTOGRAM_AGGREGATES, HISTOGRAM_DERIVATION_REVISION,
    HISTOGRAM_GROUPS, SEGMENT_FINGERPRINT_TABLE_SHA256,
    SEGMENT_FINGERPRINT_TABLE_SHA256_V2,
)
from .source_histogram import (
    CHROMA_BIN_EDGES, CLIP_FORMAT, CLIP_HIGH_L, CLIP_LOW_L, C_BIN_FORMAT,
    HISTOGRAM_MATCH_GATE, HISTOGRAM_SAMPLE_PIXELS, HUE_CHROMA_MIN, HUE_FORMAT,
    HUE_SECTOR_COUNT, L_BIN_COUNT, L_BIN_FORMAT, SOURCE_HISTOGRAM_CONTRACT,
    SOURCE_HISTOGRAM_HEADER, source_histogram_block,
)


PROMPT_REVISION = "local-agent-v1"
# v4 = ZWSP 文本决议(output_text 优先 + 前导零宽剥离);传输为流式——非流式经实测在该
# 中转站不命中 prompt cache(1/4 vs 流式 4/4),`consume_response` 留作回退路径。
ADAPTER_REVISION = "responses-auto-prefix-cache-v4"
IMAGE_ENCODING = {
    "format": "jpeg", "longest_edge": 512, "quality": 85, "subsampling": 0,
}
CANDIDATE_SERIALIZATION_REVISION = "lut-intent-v7.1-optB"
MASK_SUMMARY_REVISION = "mask-summary-v3-role"
# The frozen 5k offline diagnosis batch was produced under `diagnose-v1`; it stays
# valid (its records carry no prompt revision and are validated by schema only).
# This revision only governs diagnosis batches produced from now on.
DIAGNOSE_PROMPT_REVISION = "diagnose-v2-viewer-orientation"

# B12 item 3: the last column group of every row, present only when the catalog mounts
# a v2 fingerprint table. `{:.3f}` keeps the three numbers at ~4 tokens per row.
HISTOGRAM_ROW_FORMAT = "{:.3f}"
HISTOGRAM_SHORTLIST_NOTE = (
    "When a row ends with three more numbers they are that LUT's tone-histogram "
    "response, measured on a fixed probe pixel set: d_shadow d_mid d_high are the "
    "change in the share of pixels landing in L* bins "
    f"{list(HISTOGRAM_GROUPS['d_shadow'])}, {list(HISTOGRAM_GROUPS['d_mid'])} and "
    f"{list(HISTOGRAM_GROUPS['d_high'])} of the {L_BIN_COUNT} equal-width L* bins. "
    "A positive number means the LUT moves more pixels into that tonal group, a "
    "negative one that it moves pixels out of it; the three always add up to zero."
)

SHORTLIST_COLUMNS = (
    "id | achievable_bins | " + " ".join(FINGERPRINT_FIELDS) + " | caption"
)
SHORTLIST_HEADER = (
    "LUT shortlist table. One LUT per row, columns separated by ' | ':\n"
    f"{SHORTLIST_COLUMNS}\n"
    "id is the row_index you return. achievable_bins lists the strength bins this LUT "
    "can reach on this image; you may only return a bin listed on that row. "
    "dL, shadow_dL and highlight_dL are Lab L* shifts at mid gray, shadow and highlight. "
    "contrast is the tone-response contrast ratio (1.000 = unchanged). "
    "cast_hue is the mid-gray color-cast angle in degrees and cast_mag its Lab chroma "
    "magnitude. dSat is the mean saturation shift in percent. hue_rot is the largest "
    "absolute band hue rotation in degrees. caption is the frozen objective description.\n"
    + HISTOGRAM_SHORTLIST_NOTE
)
# B8 item 4: the local table adds one trailing column after caption.
LOCAL_SHORTLIST_NOTE = (
    "Local rows carry one extra trailing column after caption: the CIEDE2000 color "
    "difference this LUT reaches inside the assigned mask at full strength. A larger "
    "number means more room for a visible local change on that mask."
)


def _zero_safe(template: str, value: Any) -> str:
    """Format one number; a rounded-to-zero negative prints without its sign."""
    text = template.format(float(value))
    return text.lstrip("-") if float(text) == 0.0 else text


def _fingerprint_number(name: str, value: float) -> str:
    """Format one fingerprint number; a rounded-to-zero negative prints as `0`."""
    return _zero_safe(FINGERPRINT_FORMATS[name], value)


def shortlist_row_text(row_index: int, row: Mapping[str, Any]) -> str:
    fingerprint = row.get("fingerprint") or {}
    numbers = " ".join(
        _fingerprint_number(name, float(fingerprint.get(name, 0.0)))
        for name in FINGERPRINT_FIELDS
    )
    bins = ",".join(str(item) for item in (row.get("achievable_bins") or []))
    caption = " ".join(str(row.get("caption") or "").split())
    text = f"{row_index} | {bins} | {numbers} | {caption}"
    # B8 item 4: local rows carry the mask-conditioned reachable dE as a trailing
    # column. Global rows have no mask and never carry it.
    reach = row.get("mask_reach_de")
    if reach is not None:
        text += f" | {float(reach):.2f}"
    # B12 item 3: `d_shadow d_mid d_high` close every row of a v2-mounted catalog. A
    # v1 mount carries no `histogram` key and the row is byte-identical to the B11 line.
    histogram = row.get("histogram")
    if histogram:
        text += " | " + " ".join(
            _zero_safe(HISTOGRAM_ROW_FORMAT, histogram.get(name, 0.0))
            for name in HISTOGRAM_AGGREGATES
        )
    return text


def shortlist_rows_text(rows: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(
        shortlist_row_text(index, row) for index, row in enumerate(rows)
    )


def global_shortlist_text(shortlist: Mapping[str, Sequence[Mapping[str, Any]]]) -> str:
    blocks = [SHORTLIST_HEADER]
    for major, rows in shortlist.items():
        blocks.append(f"[major] {major}\n{shortlist_rows_text(rows)}")
    return "\n".join(blocks)


def local_shortlist_text(rows: Sequence[Mapping[str, Any]]) -> str:
    return SHORTLIST_HEADER + "\n" + LOCAL_SHORTLIST_NOTE + \
        "\n[local candidate rows]\n" + shortlist_rows_text(rows)


def _object(properties: dict[str, Any], required: Sequence[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object", "properties": properties,
        "required": list(required or properties), "additionalProperties": False,
    }


def _array(items: dict[str, Any], minimum: int = 0, maximum: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"type": "array", "items": items, "minItems": minimum}
    if maximum is not None:
        result["maxItems"] = maximum
    return result


DIAGNOSIS_SCHEMA = _object({
    "correction_needs": _array({"type": "string"}, 0, 4),
    "preserve_intent": _array({"type": "string"}, 1, 5),
    "enhancement_opportunities": _array({"type": "string"}, 2, 4),
    "forbidden_directions": _array({"type": "string"}, 0, 5),
    "evidence": _array({"type": "string"}, 1, 8),
    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    "intent_mode": {
        "type": "string", "enum": ["correction_led", "enhancement_led", "mixed"]
    },
})

REASON_CODES_SCHEMA = _array({"type": "string", "enum": list(REASON_CODES)}, 1, 3)

GLOBAL_PROPOSAL_SCHEMA = _object({
    "row_index": {"type": "integer", "minimum": 0},
    "bin": {"type": "string", "enum": ["natural", "medium", "bold"]},
    "reason_codes": REASON_CODES_SCHEMA,
})
GLOBAL_BATCH_SCHEMA = _object({
    "major": {"type": "string"},
    "proposals": _array(GLOBAL_PROPOSAL_SCHEMA, 1, 6),
})

LOCAL_PROPOSAL_SCHEMA = _object({
    "row_index": {"type": "integer", "minimum": 0},
    "bin": {"type": "string", "enum": ["subtle", "natural", "strong"]},
    "mask_id": {"type": "string"},
    "intent": {"type": "string", "enum": list(LOCAL_INTENTS)},
    "reason_codes": REASON_CODES_SCHEMA,
})
LOCAL_BATCH_SCHEMA = _object({"proposals": _array(LOCAL_PROPOSAL_SCHEMA, 1, 3)})

DEFECT_SCHEMA = _object({
    "defect_code": {"type": "string", "enum": [
        "subject_damage", "key_color_error", "new_color_cast", "local_boundary",
        "halo", "dirty_edge", "discontinuous_transition", "exposure_clip",
        "banding", "render_failure", "local_invisible", "region_mismatch",
    ]},
    "location": {"type": "string"},
    "evidence": {"type": "string"},
    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
})
VALIDATION_SCHEMA = _object({
    "passed": {"type": "boolean"},
    "defects": _array(DEFECT_SCHEMA, 0, 8),
})

PREFLIGHT_SCHEMA = _object({"ok": {"type": "boolean"}, "tail": {"type": "string"}})

_DIAGNOSE_RULES = """You are the source-diagnosis stage of a photo-retouch data system.
Report only visible facts. Never invent a defect to satisfy a quota. Separate correction needs,
preservation constraints, enhancement opportunities, forbidden directions, and concrete evidence.
In evidence, left and right always mean the viewer's left and right in the picture as shown, never
the depicted subject's own left and right. Never state or infer the image resolution, pixel
dimensions, megapixel count or file size, and never call the picture low-resolution or upscaled.
Enhancement opportunities must contain two to four useful items. Return only the strict JSON object."""

_GLOBAL_RULES = """You are the whole-image (global) stage of a LUT-retouch chain. The shortlist below
was recalled deterministically from the frozen diagnosis; your job is the fine ranking: pick the LUTs
that actually realize the diagnosed directions on this image.
Read the diagnosis first. correction_needs items and the components of each enhancement_opportunities
item are fixed retrieval lines '<axis> | <scope> | <observed state> | <move>'; each enhancement item
is one coherent look, 'style brief => component lines' in the fixed dimension order colour
temperature, tone, saturation, stylisation; evidence carries the numbers behind every line, and the
source_histogram line summarizes the untouched photo.
Choose one style major from the offered majors, then choose up to six materially distinct whole-image
edits from that major's rows. Serve the diagnosis: corrections first, then the looks - cover
different diagnosed directions rather than variants of one look. Match a row to a direction through
its measured columns (dL, shadow_dL, highlight_dL, contrast, cast_hue and cast_mag, dSat, hue_rot,
and the d_shadow d_mid d_high trio) against the direction's axis, scope and move. Choose the bin from
that row's achievable_bins by the evidence numbers: a strongly evidenced deviation justifies a
stronger bin, while an asset named in preserve_intent or forbidden_directions caps the strength.
Never pick a row whose caption realizes a forbidden direction.
Return each pick as the row id (row_index) plus a bin. Every proposal must use a different
row_index; if the chosen major has fewer eligible rows than the requested maximum, return fewer
proposals and never repeat a row. Include both correction and enhancement capacity when the image
supports both. Honor task.min_proposals by choosing a major with enough rows, and use at least two
bins when returning multiple proposals. Give one to three reason_codes per proposal from the fixed
vocabulary. Return only the strict JSON object, with no rationale or extra prose."""

_LOCAL_INTENT_GUIDE = """Local intent vocabulary. Every assigned mask carries a role and the intents
offered on it, and every offered intent lists the row ids you may pick for it:
cast_correction_local - undo the color cast that is still left inside the masked region after the
whole-image edit; compare the two images to see what that leftover cast is.
contrast_boost - deepen the tonal contrast inside the masked region: its shadows go down and its
highlights go up.
luminance_pop - brighten the masked subject region so it separates from its surroundings.
sat_boost - deepen the colors of the masked subject region while its brightness stays put.
hue_shift - move the color cast of the masked subject region while brightness and saturation stay put.
background_control - quiet the background: no brightening, no added color, no new cast.
zonal_contrast - darken the background so the subject reads as the brighter zone."""

_LOCAL_RULES = """Two images are shown, in this order: the first is the untouched source photo, the
second is the real rendered whole-image (global) result that was produced from it. Read the pair as a
before/after: what the global edit already fixed, what it left behind, and what is still open inside a
region. Every local edit you pick is applied on top of the second image, never on the first.
Select up to three local edits;
you may not answer that no local edit is needed. Use only assigned mask IDs and local candidate rows
that are not excluded; return each pick as the row id (row_index), a bin from that row's
achievable_bins, one assigned mask_id, and one intent offered on that mask. The row id must be one of
the row ids listed for that mask and intent. Every proposal must use a different mask, and when you
return two or more proposals they must cover at least two different intents. Give one to three
reason_codes per proposal from the fixed vocabulary. Return only the strict JSON object, with no
rationale or extra prose."""

_VALIDATE_RULES = """Validate exactly one source -> global -> final chain. Reject only concrete hard
defects: subject damage, key-color error, new cast, boundary/halo/dirty edge/discontinuity, severe clip,
banding/render failure, invisible local change, or assigned-region mismatch. Do not rank aesthetics or
judge whether it looks premium. Every defect needs code, location, visible evidence, and confidence."""


# C1b item 1: the explicit key list of `prompt_registry()`. The structural test asserts
# the built registry's key set equals this tuple, so adding or deleting a registry key
# is a deliberate two-file edit and can never be an accident.
PROMPT_REGISTRY_KEYS: tuple[str, ...] = (
    "active_local_intents",
    "adapter_revision",
    "axis_scales",
    "axis_weights",
    "background_family_counts",
    "background_role_gate",
    "band_geometry_gate",
    "band_segment_lum_threshold",
    "candidate_serialization",
    "diagnose_prompt_revision",
    "fingerprint_fields",
    "fingerprint_formats",
    "global_bin_quota",
    "global_delta_e_targets",
    "histogram_match_gate",
    "image_encoding",
    "intent_direction_gate",
    "intent_fingerprint_gate",
    "intent_packet_order",
    "intent_packet_priority",
    "intent_role_domains",
    "intent_segment_gate",
    "keyword_axes",
    "local_delta_e_ladders",
    "local_intent_ladder",
    "local_intents",
    "local_online_retrieval",
    "local_packet_contract",
    "local_shortlist_note",
    "local_visibility_floor",
    "mask_reach_gate",
    "mask_summary",
    "measure_sample_pixels",
    "prompt_revision",
    "reason_codes",
    "role_packet_target",
    "rules",
    "schemas",
    "segment_fingerprint_histogram",
    "segment_fingerprint_table_sha256",
    "shortlist_header",
    "source_histogram",
    "subject_band_gate",
    "subject_headroom_gate",
)


def prompt_registry() -> dict[str, Any]:
    """Every constant whose value changes what the model is asked or offered.

    `prompt_revision_fingerprint()` is the canonical SHA-256 of this dict.
    """
    # Local import: the mask-supply gates live next to the mask builder, and
    # `candidates` never imports `prompts`, so this cannot cycle.
    from .candidates import (
        BACKGROUND_FAMILY_COUNTS, BACKGROUND_ROLE_GATE, GLOBAL_BIN_QUOTA,
        ROLE_PACKET_TARGET, SUBJECT_BAND_GATE,
    )

    registry = {
        "prompt_revision": PROMPT_REVISION,
        "adapter_revision": ADAPTER_REVISION,
        "image_encoding": IMAGE_ENCODING,
        "candidate_serialization": CANDIDATE_SERIALIZATION_REVISION,
        "shortlist_header": SHORTLIST_HEADER,
        "fingerprint_fields": list(FINGERPRINT_FIELDS),
        "fingerprint_formats": dict(FINGERPRINT_FORMATS),
        "reason_codes": list(REASON_CODES),
        "mask_summary": MASK_SUMMARY_REVISION,
        "diagnose_prompt_revision": DIAGNOSE_PROMPT_REVISION,
        "local_intents": list(LOCAL_INTENTS),
        "active_local_intents": list(ACTIVE_LOCAL_INTENTS),
        # B11 items 1-3 (R7.1/R7.3): the mounted fingerprint table, the online retrieval
        # budget, the role domains, the two new intent domains and the packet order all
        # decide which rows exist and in which order the model sees them.
        "segment_fingerprint_table_sha256": SEGMENT_FINGERPRINT_TABLE_SHA256,
        "local_online_retrieval": dict(LOCAL_ONLINE_RETRIEVAL),
        "intent_role_domains": {
            intent: list(roles) for intent, roles in INTENT_ROLE_DOMAINS.items()
        },
        "intent_segment_gate": dict(INTENT_SEGMENT_GATE),
        "intent_direction_gate": dict(INTENT_DIRECTION_GATE),
        "intent_packet_priority": dict(INTENT_PACKET_PRIORITY),
        "intent_packet_order": list(intent_packet_order(ACTIVE_LOCAL_INTENTS)),
        # B6 item 10: the packet row budget and the per-bin floor shape every local
        # shortlist the model sees, so they belong in the revision fingerprint.
        "local_packet_contract": {
            "row_limit": LOCAL_PACKET_ROW_LIMIT, "bin_quota": INTENT_BIN_QUOTA,
        },
        # B7 item 1: the per-intent strength ladder decides which bins each shortlist
        # row advertises, so it belongs in the revision fingerprint.
        "local_intent_ladder": dict(LOCAL_INTENT_LADDER),
        "local_delta_e_ladders": {
            name: {bin_: list(band) for bin_, band in ladder.items()}
            for name, ladder in LOCAL_DELTA_E_LADDERS.items()
        },
        # B8 items 1-4: the transition floor, the mask-supply gates and the
        # mask-conditioned reach gate all decide which rows and masks exist at all.
        "local_shortlist_note": LOCAL_SHORTLIST_NOTE,
        "local_visibility_floor": LOCAL_VISIBILITY_FLOOR,
        "band_geometry_gate": dict(BAND_GEOMETRY_GATE),
        "mask_reach_gate": dict(MASK_REACH_GATE),
        "background_role_gate": dict(BACKGROUND_ROLE_GATE),
        "subject_band_gate": dict(SUBJECT_BAND_GATE),
        # C1b item 1: eleven constants that decide which rows / masks / bins exist or
        # what the calibrator is asked to hit, but had never been registered.
        # `INTENT_FINGERPRINT_GATE` + `INTENT_SEGMENT_GATE` + `INTENT_DIRECTION_GATE`
        # are the per-intent candidate domains; `GLOBAL_DELTA_E_TARGETS` is the global
        # strength ladder (the local one was already registered);
        # `SUBJECT_HEADROOM_GATE` gates the false-white soft cap and the posterior clip
        # guard; `ROLE_PACKET_TARGET` / `BACKGROUND_FAMILY_COUNTS` / `GLOBAL_BIN_QUOTA`
        # shape the mask and shortlist mix; `AXIS_SCALES` / `AXIS_WEIGHTS` /
        # `KEYWORD_AXES` / `MEASURE_SAMPLE_PIXELS` define the R6.2 direction prefilter
        # that picks the local rows; `BAND_SEGMENT_LUM_THRESHOLD` decides which bands
        # enter a segment of the mounted fingerprint table.
        "intent_fingerprint_gate": dict(INTENT_FINGERPRINT_GATE),
        "global_delta_e_targets": {
            bin_: list(band) for bin_, band in GLOBAL_DELTA_E_TARGETS.items()
        },
        "subject_headroom_gate": dict(SUBJECT_HEADROOM_GATE),
        "role_packet_target": dict(ROLE_PACKET_TARGET),
        "background_family_counts": dict(BACKGROUND_FAMILY_COUNTS),
        "global_bin_quota": GLOBAL_BIN_QUOTA,
        "axis_scales": dict(AXIS_SCALES),
        "axis_weights": dict(AXIS_WEIGHTS),
        "keyword_axes": [
            [list(aliases), axis, sign] for aliases, axis, sign in KEYWORD_AXES
        ],
        "measure_sample_pixels": MEASURE_SAMPLE_PIXELS,
        "band_segment_lum_threshold": BAND_SEGMENT_LUM_THRESHOLD,
        # B12 items 1-3. `source_histogram` is everything that decides the bytes of the
        # one histogram line the global round now sees (bin geometry + the four fixed
        # number formats + the frozen header text). `segment_fingerprint_histogram` is
        # the LUT-side derivation: which table the v2 columns came from, how the eight
        # bins are grouped, and the serialization width of the three shortlist numbers.
        # `histogram_match_gate` is the pre-registered retrieval bonus, sign convention
        # included, so a weight change is a different prompt revision.
        "source_histogram": {
            "contract": SOURCE_HISTOGRAM_CONTRACT,
            "sample_pixels": HISTOGRAM_SAMPLE_PIXELS,
            "l_bin_count": L_BIN_COUNT,
            "clip_low_l": CLIP_LOW_L,
            "clip_high_l": CLIP_HIGH_L,
            "chroma_bin_edges": list(CHROMA_BIN_EDGES),
            "hue_sector_count": HUE_SECTOR_COUNT,
            "hue_chroma_min": HUE_CHROMA_MIN,
            "formats": {
                "l_bin": L_BIN_FORMAT, "clip": CLIP_FORMAT,
                "c_bin": C_BIN_FORMAT, "hue": HUE_FORMAT,
            },
            "header": SOURCE_HISTOGRAM_HEADER,
        },
        "segment_fingerprint_histogram": {
            "derivation_revision": HISTOGRAM_DERIVATION_REVISION,
            "table_sha256_v2": SEGMENT_FINGERPRINT_TABLE_SHA256_V2,
            "groups": {
                name: list(group) for name, group in HISTOGRAM_GROUPS.items()
            },
            "aggregates": list(HISTOGRAM_AGGREGATES),
            "row_format": HISTOGRAM_ROW_FORMAT,
            "note": HISTOGRAM_SHORTLIST_NOTE,
        },
        "histogram_match_gate": dict(HISTOGRAM_MATCH_GATE),
        "rules": {
            "diagnose": _DIAGNOSE_RULES, "global": _GLOBAL_RULES,
            "local": _LOCAL_RULES, "local_intent_guide": _LOCAL_INTENT_GUIDE,
            "validate": _VALIDATE_RULES,
        },
        "schemas": {
            "diagnose": DIAGNOSIS_SCHEMA, "global": GLOBAL_BATCH_SCHEMA,
            "local": LOCAL_BATCH_SCHEMA, "validate": VALIDATION_SCHEMA,
            "preflight": PREFLIGHT_SCHEMA,
        },
    }
    return registry


def prompt_revision_fingerprint() -> str:
    payload = json.dumps(
        prompt_registry(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _text(value: Any) -> dict[str, Any]:
    return {"type": "input_text", "text": value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )}


def _image(ref: ArtifactRef | Mapping[str, Any], *, detail: str = "low") -> dict[str, Any]:
    value = ref.to_dict() if isinstance(ref, ArtifactRef) else dict(ref)
    return {
        "type": "input_image", "artifact_sha256": value["sha256"],
        "media_type": value["media_type"], "detail": detail,
        "encoding": dict(IMAGE_ENCODING),
    }


def _request(
    *, endpoint: EndpointConfig, stage: str, schema_name: str, schema: dict[str, Any],
    prefix_content: list[dict[str, Any]], tail_content: list[dict[str, Any]],
) -> RequestSpec:
    if not prefix_content or prefix_content[0].get("type") != "input_text":
        raise ValueError("prompt prefix must begin with developer text")
    developer = {"role": "developer", "content": [dict(prefix_content[0])]}
    stable_user = [dict(item) for item in prefix_content[1:]]
    inputs = [
        developer,
        {"role": "user", "content": stable_user + tail_content},
    ]
    behavior = {
        "temperature": endpoint.temperature,
        "reasoning_effort": endpoint.reasoning_effort,
        "max_output_tokens": endpoint.max_output_tokens,
        "store": False,
    }
    cache_key = prefix_cache_key(
        model_family=endpoint.model, prompt_revision=PROMPT_REVISION,
        stage=stage,
        prefix={
            "input": [developer, {"role": "user", "content": stable_user}],
            "schema": schema, "behavior": behavior,
        },
    )
    canonical = {
        "endpoint_identity": endpoint.identity,
        "model": endpoint.model,
        "prompt_revision": PROMPT_REVISION,
        "adapter_revision": ADAPTER_REVISION,
        "stage": stage,
        "input": inputs,
        "schema_name": schema_name,
        "schema": schema,
        "behavior": behavior,
        "prompt_cache_key": cache_key,
    }
    return RequestSpec(canonical=canonical, prompt_cache_key=cache_key)


def diagnosis_request(endpoint: EndpointConfig, source: Mapping[str, Any]) -> RequestSpec:
    return _request(
        endpoint=endpoint, stage="diagnose", schema_name="source_diagnosis_v1",
        schema=DIAGNOSIS_SCHEMA, prefix_content=[_text(_DIAGNOSE_RULES)],
        tail_content=[_image(source), _text("Diagnose this source image.")],
    )


def global_request(
    endpoint: EndpointConfig, source: Mapping[str, Any], diagnosis: Mapping[str, Any],
    shortlist: Mapping[str, Any], *, min_proposals: int = 1, max_proposals: int,
    source_histogram: Mapping[str, Any] | None = None,
) -> RequestSpec:
    """B12 item 1: the frozen source-histogram line sits in the stable prefix, in the
    slot right after the frozen diagnosis and right before the shortlist table."""
    if not 1 <= min_proposals <= max_proposals <= 6:
        raise ValueError("global proposal bounds must satisfy 1 <= min <= max <= 6")
    prefix = [
        _text(_GLOBAL_RULES), _image(source), _text({"diagnosis": diagnosis}),
    ]
    if source_histogram is not None:
        prefix.append(_text(source_histogram_block(source_histogram)))
    prefix.append(_text(global_shortlist_text(shortlist)))
    return _request(
        endpoint=endpoint, stage="global_propose", schema_name="global_batch_v1",
        schema=GLOBAL_BATCH_SCHEMA,
        prefix_content=prefix,
        tail_content=[_text({
            "task": {
                "min_proposals": min_proposals,
                "max_proposals": max_proposals,
                "choose_one_major": True,
                "offered_majors": list(shortlist),
            },
        })],
    )


def assigned_mask_views(
    masks: Sequence[Mapping[str, Any]], packets: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Mask summaries plus the intent packets offered on each mask."""
    views = []
    for mask in masks:
        mask_id = str(mask["mask_id"])
        offered = [
            {"intent": str(packet["intent"]),
             "row_indices": [int(index) for index in packet["row_indices"]]}
            for packet in packets if str(packet["mask_id"]) == mask_id
        ]
        views.append({**dict(mask), "offered_intents": offered})
    return views


def local_request(
    endpoint: EndpointConfig, source: Mapping[str, Any],
    global_after: Mapping[str, Any], diagnosis: Mapping[str, Any],
    global_decision: Mapping[str, Any], render_summary: Mapping[str, Any],
    lut_shortlist: Sequence[Mapping[str, Any]], masks: Sequence[Mapping[str, Any]],
    *, max_proposals: int, packets: Sequence[Mapping[str, Any]] = (),
    excluded_row_indices: Sequence[int] = (),
    repair: Mapping[str, Any] | None = None,
) -> RequestSpec:
    """B11 item 4 (R7.2): the local round sees source -> global_after, in that order.

    The source image is per-source content, so it sits in the stable prefix together
    with the rules, the intent guide and the frozen diagnosis; every global sibling and
    every repair attempt of one source therefore shares that prefix byte for byte. The
    per-branch shortlist stays last in the prefix and `global_after` stays in the tail,
    which keeps the two images in source -> global_after order inside the user turn.
    """
    stable = [
        _text(_LOCAL_RULES), _text(_LOCAL_INTENT_GUIDE), _text({"diagnosis": diagnosis}),
        _image(source), _text(local_shortlist_text(lut_shortlist)),
    ]
    tail: list[dict[str, Any]] = [
        _image(global_after),
        _text({"global_decision": global_decision, "actual_render": render_summary}),
        _text({
            "excluded_row_indices": sorted({int(item) for item in excluded_row_indices}),
            "assigned_masks": assigned_mask_views(masks, packets),
            "task": {"max_proposals": max_proposals},
        }),
    ]
    if repair is not None:
        tail.append(_text({"repair": repair}))
    return _request(
        endpoint=endpoint, stage="local_propose", schema_name="local_batch_v1",
        schema=LOCAL_BATCH_SCHEMA, prefix_content=stable, tail_content=tail,
    )


def validation_request(
    endpoint: EndpointConfig, source: Mapping[str, Any], global_after: Mapping[str, Any],
    final_after: Mapping[str, Any], assignment: Mapping[str, Any],
) -> RequestSpec:
    return _request(
        endpoint=endpoint, stage="chain_verify", schema_name="chain_validation_v1",
        schema=VALIDATION_SCHEMA,
        prefix_content=[
            _text(_VALIDATE_RULES), _image(source), _image(global_after),
        ],
        tail_content=[
            _image(final_after),
            _text({"assigned_region": dict(assignment),
                   "task": "Validate this one final result against the assigned region."}),
        ],
    )


def preflight_request(
    endpoint: EndpointConfig, image: Mapping[str, Any], reference: Sequence[Mapping[str, Any]],
    tail: str,
) -> RequestSpec:
    rules = (
        "Provider capability fixture for the local-retouch pipeline. Inspect the image and the "
        "representative objective color-response records, then return ok=true and copy the tail."
    )
    return _request(
        endpoint=endpoint, stage="provider_preflight", schema_name="provider_preflight_v1",
        schema=PREFLIGHT_SCHEMA,
        prefix_content=[_text(rules), _image(image), _text({"objective_lut_reference": list(reference)})],
        tail_content=[_text({"tail": tail})],
    )


def semantic_error(stage: str, parsed: Any) -> str | None:
    schema = {
        "diagnose": DIAGNOSIS_SCHEMA,
        "global_propose": GLOBAL_BATCH_SCHEMA,
        "local_propose": LOCAL_BATCH_SCHEMA,
        "chain_verify": VALIDATION_SCHEMA,
        "provider_preflight": PREFLIGHT_SCHEMA,
    }.get(stage)
    if schema is not None:
        mechanical = schema_error(schema, parsed)
        if mechanical is not None:
            return mechanical
    if not isinstance(parsed, dict):
        return "root_not_object"
    if stage == "diagnose":
        required = set(DIAGNOSIS_SCHEMA["required"])
        if set(parsed) != required:
            return "diagnosis_keys"
        if not 2 <= len(parsed.get("enhancement_opportunities", [])) <= 4:
            return "enhancement_count"
    elif stage == "global_propose":
        proposals = parsed.get("proposals")
        if not isinstance(proposals, list) or not 1 <= len(proposals) <= 6:
            return "global_count"
        indices = [row.get("row_index") for row in proposals if isinstance(row, dict)]
        if len(set(indices)) != len(indices):
            return "global_not_distinct"
        if _duplicate_reason_codes(proposals):
            return "reason_codes_not_distinct"
    elif stage == "local_propose":
        proposals = parsed.get("proposals")
        if not isinstance(proposals, list) or not 1 <= len(proposals) <= 3:
            return "local_count"
        pairs = [(row.get("row_index"), row.get("mask_id")) for row in proposals
                 if isinstance(row, dict)]
        if len(set(pairs)) != len(pairs):
            return "local_not_distinct"
        # Contract C3: the submitted sibling set must cover at least two intents.
        intents = {row.get("intent") for row in proposals if isinstance(row, dict)}
        if len(proposals) >= 2 and len(intents) < 2:
            return "local_intent_diversity"
        if _duplicate_reason_codes(proposals):
            return "reason_codes_not_distinct"
    elif stage == "chain_verify":
        passed = parsed.get("passed")
        defects = parsed.get("defects")
        if not isinstance(passed, bool) or not isinstance(defects, list):
            return "validation_shape"
        if passed == bool(defects):
            return "validation_pass_defect_conflict"
    return None


def _duplicate_reason_codes(proposals: Sequence[Any]) -> bool:
    for row in proposals:
        if not isinstance(row, dict):
            continue
        codes = row.get("reason_codes")
        if isinstance(codes, list) and len(set(codes)) != len(codes):
            return True
    return False


def schema_error(schema: Mapping[str, Any], value: Any, path: str = "$") -> str | None:
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, dict):
            return f"{path}:object"
        required = set(schema.get("required") or ())
        if not required.issubset(value):
            return f"{path}:required"
        properties = schema.get("properties") or {}
        if schema.get("additionalProperties") is False and set(value).difference(properties):
            return f"{path}:additional"
        for key, child in properties.items():
            if key in value:
                error = schema_error(child, value[key], f"{path}.{key}")
                if error:
                    return error
    elif kind == "array":
        if not isinstance(value, list):
            return f"{path}:array"
        if len(value) < int(schema.get("minItems", 0)):
            return f"{path}:minItems"
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            return f"{path}:maxItems"
        for index, item in enumerate(value):
            error = schema_error(schema["items"], item, f"{path}[{index}]")
            if error:
                return error
    elif kind == "string":
        if not isinstance(value, str) or not value.strip():
            return f"{path}:string"
    elif kind in {"number", "integer"}:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return f"{path}:{kind}"
        if kind == "integer" and not isinstance(value, int) \
                and float(value) != int(value):
            return f"{path}:integer"
        if "minimum" in schema and float(value) < float(schema["minimum"]):
            return f"{path}:minimum"
        if "maximum" in schema and float(value) > float(schema["maximum"]):
            return f"{path}:maximum"
    elif kind == "boolean" and not isinstance(value, bool):
        return f"{path}:boolean"
    if "enum" in schema and value not in schema["enum"]:
        return f"{path}:enum"
    return None


__all__ = [
    "ADAPTER_REVISION", "CANDIDATE_SERIALIZATION_REVISION", "DIAGNOSE_PROMPT_REVISION",
    "DIAGNOSIS_SCHEMA",
    "GLOBAL_BATCH_SCHEMA", "HISTOGRAM_ROW_FORMAT", "HISTOGRAM_SHORTLIST_NOTE",
    "IMAGE_ENCODING", "MASK_SUMMARY_REVISION",
    "LOCAL_BATCH_SCHEMA", "LOCAL_SHORTLIST_NOTE", "PREFLIGHT_SCHEMA",
    "PROMPT_REGISTRY_KEYS", "PROMPT_REVISION", "REASON_CODES", "prompt_registry",
    "SHORTLIST_COLUMNS", "SHORTLIST_HEADER", "VALIDATION_SCHEMA", "assigned_mask_views",
    "diagnosis_request", "global_request", "global_shortlist_text", "local_request",
    "local_shortlist_text", "preflight_request", "prompt_revision_fingerprint",
    "schema_error", "semantic_error", "shortlist_row_text", "shortlist_rows_text",
    "validation_request",
]
