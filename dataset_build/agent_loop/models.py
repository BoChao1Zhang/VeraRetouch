"""Serializable state contracts shared by graph nodes."""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, Sequence, TypedDict


StrengthBin = Literal["natural", "medium", "bold"]
IntentMode = Literal["correction_led", "enhancement_led", "mixed"]

GLOBAL_DELTA_E_TARGETS: dict[str, tuple[float, float, bool]] = {
    "natural": (3.0, 4.5, False),
    "medium": (4.5, 6.5, False),
    "bold": (6.5, 8.5, True),
}
# B6 recalibration (200-chain intent questionnaire, user-approved 2026-08-20) moved
# the whole local ladder up by one bin width. B7 (156-row v2 questionnaire,
# user-approved 2026-08-20) splits that single ladder in two and looks it up per
# intent: the intents that gained under the B6 band keep it (`high`), the intents
# that lost under it fall back to the pre-B6 band (`low`).
# B8 item 1 (R5.1, user-approved 2026-08-20) puts a hard transition floor of 3.5 under
# every active intent: the `low` ladder's [2.5, 3.5) subtle band is the only band that
# sat below it, and its lower edge is raised to 3.5 (band width kept at 1.0) instead of
# deleting the bin, so all three bin names stay valid for every intent and the
# `LOCAL_BIN_DOWNSHIFT` chain of the false-white soft cap keeps a landing bin. Side
# effect: on the `low` ladder `subtle` and `natural` now name the same band.
# B11 item 5 (R7.4, user-approved 2026-08-20) raises that floor 3.5 -> 4.0 the same way:
# every band whose lower edge sat at 3.5 moves to 4.0 and keeps its upper edge, so the
# `subtle` band is [4.0, 4.5) on both ladders and no bin name disappears.
LOCAL_DELTA_E_LADDERS: dict[str, dict[str, tuple[float, float, bool]]] = {
    "high": {
        "subtle": (4.0, 4.5, False),
        "natural": (4.5, 5.5, False),
        "strong": (5.5, 6.5, True),
    },
    "low": {
        "subtle": (4.0, 4.5, False),
        "natural": (4.0, 4.5, False),
        "strong": (4.5, 5.5, True),
    },
}
# B8 item 1: minimum calibrated mask ΔE any local leaf may be rendered at. A leaf whose
# calibration cannot reach it is rejected (`local_visibility_floor`), never kept weak.
# B11 item 5 (R7.4): 3.5 -> 4.0. R5.2: replaced by the V floor once R3 lands.
LOCAL_VISIBILITY_FLOOR: float = 4.0
LOCAL_INTENT_LADDER: dict[str, str] = {
    "luminance_pop": "low",
    "hue_shift": "low",
    "sat_boost": "high",
    "background_control": "high",
    "zonal_contrast": "high",
    # B11 item 3 (R7.3). Both new intents start on the conservative `low` ladder; no
    # data exists to place them on `high` (see NOTES_optB_B11).
    "contrast_boost": "low",
    "cast_correction_local": "low",
    # Disabled intents keep a ladder so already frozen audit rows still resolve.
    "highlight_rescue": "high",
    "warm_cool_split": "high",
}
LOCAL_DELTA_E_TARGETS: dict[str, dict[str, tuple[float, float, bool]]] = {
    intent: LOCAL_DELTA_E_LADDERS[ladder]
    for intent, ladder in LOCAL_INTENT_LADDER.items()
}
# B7 item 2 (luminance_pop false-white soft cap). `p99_luma` is the subject-region
# 99th percentile of the max channel measured on global_after by
# `subject_highlight_headroom`. Above this level the whole target ladder of the
# capped intent is shifted down one bin for that chain; `subtle` has no lower bin
# and therefore keeps its band while still being audited as capped.
LUMA_SOFT_CAP: dict[str, float] = {"p99_luma_max": 0.96}
LUMA_SOFT_CAP_INTENTS: frozenset[str] = frozenset({"luminance_pop"})
LOCAL_BIN_DOWNSHIFT: dict[str, str] = {
    "strong": "natural", "natural": "subtle", "subtle": "subtle",
}

# Frozen eight-number LUT effect fingerprint (decisions doc section 2.1) and its
# serialization precision. Order is load-bearing: it is the shortlist column order.
FINGERPRINT_FIELDS: tuple[str, ...] = (
    "dL", "contrast", "shadow_dL", "highlight_dL", "cast_hue", "cast_mag",
    "dSat", "hue_rot",
)
FINGERPRINT_FORMATS: dict[str, str] = {
    "dL": "{:.1f}", "contrast": "{:.3f}", "shadow_dL": "{:.1f}",
    "highlight_dL": "{:.1f}", "cast_hue": "{:.0f}", "cast_mag": "{:.1f}",
    "dSat": "{:.1f}", "hue_rot": "{:.1f}",
}
# Local intent vocabulary v1 (DECISIONS_agent_loop_local_intent_optB_20260819 section 3).
# Intent is a candidate-packet generation parameter, never a post-hoc label.
LOCAL_INTENTS: tuple[str, ...] = (
    "luminance_pop", "sat_boost", "hue_shift", "zonal_contrast",
    "warm_cool_split", "highlight_rescue", "background_control",
    # B11 item 3 (R7.3, 2026-08-20).
    "contrast_boost", "cast_correction_local",
)
# B11 item 3: an intent may now serve more than one mask role. The two new intents are
# defined on "the masked region" without reference to what is inside it, so they are
# offered on both roles; every v1 intent keeps its single role.
INTENT_ROLE_DOMAINS: dict[str, tuple[str, ...]] = {
    "luminance_pop": ("subject",), "sat_boost": ("subject",),
    "hue_shift": ("subject",), "highlight_rescue": ("subject",),
    "zonal_contrast": ("background",), "warm_cool_split": ("background",),
    "background_control": ("background",),
    "contrast_boost": ("subject", "background"),
    "cast_correction_local": ("subject", "background"),
}
# Primary role, kept for the frozen audit rows and every single-role reader.
INTENT_ROLES: dict[str, str] = {
    intent: roles[0] for intent, roles in INTENT_ROLE_DOMAINS.items()
}


def intent_serves_role(intent: str, role: str) -> bool:
    """B11 item 3: role match against the (possibly multi-role) intent domain."""
    if intent not in INTENT_ROLE_DOMAINS:
        raise KeyError(f"unknown local intent: {intent}")
    return role in INTENT_ROLE_DOMAINS[intent]


# B6: `warm_cool_split` is disabled as a generation parameter (user-approved
# 2026-08-20). B7 adds `highlight_rescue` on the same terms (user-approved
# 2026-08-20). Both stay in `LOCAL_INTENTS` and `INTENT_ROLES` so already frozen
# audit rows keep resolving, but they never enter a candidate packet again.
DISABLED_LOCAL_INTENTS: frozenset[str] = frozenset(
    {"warm_cool_split", "highlight_rescue"}
)
ACTIVE_LOCAL_INTENTS: tuple[str, ...] = tuple(
    intent for intent in LOCAL_INTENTS if intent not in DISABLED_LOCAL_INTENTS
)


def assert_local_visibility_floor() -> None:
    """Runtime assertion of B8 item 1: no active intent may target below the floor."""
    for intent in ACTIVE_LOCAL_INTENTS:
        for strength_bin, (low, _high, _inclusive) in \
                LOCAL_DELTA_E_TARGETS[intent].items():
            if float(low) < LOCAL_VISIBILITY_FLOOR:
                raise ValueError(
                    f"{intent}/{strength_bin} targets {low} below the local "
                    f"visibility floor {LOCAL_VISIBILITY_FLOOR}"
                )


assert_local_visibility_floor()
# v1 simplification of the two paired intents. One chain carries exactly one local
# edit (the graph fans local proposals out as siblings on global_after, never as a
# chain), so `subject + background` pairs cannot be expressed in a single leaf.
# Both paired intents are realized on the background mask only; the subject half is
# left to the subject intents. Recorded on every packet, proposal and leaf.
INTENT_V1_VARIANTS: dict[str, str] = {
    "zonal_contrast": "background_darken",
    "warm_cool_split": "background_cool",
}
# Pre-registered fingerprint direction domains per intent (candidate side).
INTENT_FINGERPRINT_GATE: dict[str, float] = {
    "dL_positive_min": 0.5,
    "dL_negative_max": -0.5,
    "dL_nonpositive_max": 0.0,
    "dL_small_abs_max": 3.0,
    "dSat_positive_min": 3.0,
    "dSat_small_abs_max": 15.0,
    "dSat_nonpositive_max": 0.0,
    "cast_mag_mid_min": 1.5,
    "cast_mag_mid_max": 12.0,
    "cast_mag_small_max": 3.0,
    # B6: tightened from -0.5 (user-approved 2026-08-20).
    "highlight_dL_negative_max": -2.0,
    "cast_b_cool_max": -1.0,
}
# B11 item 3 (R7.3). `contrast_boost` reads the *segmented* fingerprint (B10/R6.1), not
# the eight-number one: the LUT must darken the shadows segment and lift the highlights
# segment of the neutral ramp. Pre-registered initial values, units are Lab L*.
INTENT_SEGMENT_GATE: dict[str, float] = {
    "contrast_shadow_dL_max": -1.0, "contrast_highlight_dL_min": 1.0,
}
# B11 item 3 (R7.3). `cast_correction_local` reads the R6.2 direction match against the
# *measured* residual of source -> global_after inside the mask, in correction mode, so a
# LUT that points against that residual scores high. Pre-registered initial value; the
# score is a cosine in [-1, 1].
INTENT_DIRECTION_GATE: dict[str, float] = {"cast_correction_match_min": 0.5}
# B11 item 3 (R7.3). Packet order inside one mask: the two new intents and
# `zonal_contrast` come first. This is a sort key, not a quota - nothing is dropped for
# missing the cut, the packets are only emitted (and therefore row-indexed) in this
# order. Ties fall back to the `ACTIVE_LOCAL_INTENTS` order.
INTENT_PACKET_PRIORITY: dict[str, int] = {
    "cast_correction_local": 0, "contrast_boost": 1, "zonal_contrast": 2,
}
INTENT_PACKET_PRIORITY_DEFAULT: int = 3
# B11 item 2 (R7.1). Online retrieval budget: the direction prefilter keeps this many
# presets of the whole catalog per (mask, intent) before the mask-conditioned reach
# probe runs on the survivors.
LOCAL_ONLINE_RETRIEVAL: dict[str, int] = {"prefilter_top_k": 50}
# Pre-registered subject highlight-headroom numbers (B2). `near_clip_level` is the
# 250/255 max-channel level; `subject_clip_regression_max` is the posterior gate on
# newly clipped subject pixels of final_after relative to global_after.
SUBJECT_HEADROOM_GATE: dict[str, float] = {
    "near_clip_level": 250.0 / 255.0,
    "near_clip_fraction_max": 0.005,
    "p99_luma_max": 0.98,
    "sat_mean_max": 0.55,
    "subject_clip_regression_max": 0.005,
}
# Per-packet LUT row budget and the per-strength-bin floor inside one packet. Both
# enter `prompt_revision_fingerprint()` (B6 item 10).
LOCAL_PACKET_ROW_LIMIT = 4
INTENT_BIN_QUOTA = 1
# B8 item 2 (R4.1, pre-registered initial values). Every band-family mask slot, in both
# roles, must be at least `min_width_short` of the image short edge wide on its narrow
# side and no longer than `aspect_max` times that width. Both numbers are measured on
# every contiguous slab of the full-resolution alpha >= 0.5 support, so the complement
# bands used by the background role are gated slab by slab.
BAND_GEOMETRY_GATE: dict[str, float] = {"min_width_short": 0.18, "aspect_max": 4.0}
# B8 item 4 (R1). Mask-conditioned reachability of one (mask, LUT) pair: sample at most
# `sample_pixels` pixels of the mask support (alpha > `alpha_min`), render at full local
# strength and take the alpha-weighted mean CIEDE2000. A LUT whose mask-conditioned
# reach is below `reach_de_min` cannot be calibrated to the visibility floor on that
# mask, so it never enters the packet.
MASK_REACH_GATE: dict[str, float] = {
    "sample_pixels": 1024.0, "alpha_min": 0.05,
    "reach_de_min": LOCAL_VISIBILITY_FLOOR,
}

# Closed reason vocabulary (decisions doc section 2.4); free-text rationale is gone.
REASON_CODES: tuple[str, ...] = (
    "cast_correction", "exposure_fit", "scene_mood_fit", "skin_safe",
    "palette_harmony", "tonal_contrast", "diversity_pick", "forbidden_clear",
)


class SourceInput(TypedDict, total=False):
    source_id: str
    source_path: str
    subject_path: str
    pass_index: int
    scene: str
    subject: dict[str, Any]
    source_annotation: dict[str, Any]
    source_annotation_path: str


class MainState(SourceInput, total=False):
    source_sha256: str
    subject_sha256: str
    thread_id: str
    source_artifact: dict[str, Any]
    source_render_artifact: dict[str, Any]
    palette: dict[str, Any]
    masks: list[dict[str, Any]]
    mask_diagnostics: list[dict[str, Any]]
    source_annotation_artifact: dict[str, Any]
    diagnosis: dict[str, Any]
    preset_reach: dict[str, Any]
    global_shortlist: dict[str, list[dict[str, Any]]]
    global_shortlist_artifact: dict[str, Any]
    selected_major: str
    global_proposals: list[dict[str, Any]]
    branches: Annotated[list[dict[str, Any]], operator.add]
    committed_leaves: list[dict[str, Any]]
    terminal_status: str
    reject_reasons: list[str]
    tree_artifact: dict[str, Any]


class GlobalBranchState(TypedDict, total=False):
    source_id: str
    source_sha256: str
    thread_id: str
    source_artifact: dict[str, Any]
    source_render_artifact: dict[str, Any]
    diagnosis: dict[str, Any]
    preset_reach: dict[str, Any]
    palette: dict[str, Any]
    scene: str
    masks: list[dict[str, Any]]
    selected_major: str
    global_shortlist: dict[str, list[dict[str, Any]]]
    global_proposal: dict[str, Any]
    global_render: dict[str, Any]
    local_shortlist: list[dict[str, Any]]
    local_shortlist_deficits: list[dict[str, Any]]
    local_packet: list[dict[str, Any]]
    local_intent_packets: list[dict[str, Any]]
    local_packet_notes: list[dict[str, Any]]
    local_retrieval: list[dict[str, Any]]
    role_packet_note: dict[str, Any]
    intent_supply: int
    subject_headroom: dict[str, Any]
    unused_masks: list[dict[str, Any]]
    local_proposals: list[dict[str, Any]]
    leaves: Annotated[list[dict[str, Any]], operator.add]
    repair_count: int
    branch_status: str
    branch_reject_reason: str
    branches: list[dict[str, Any]]


class LocalLeafInput(TypedDict, total=False):
    source_id: str
    source_sha256: str
    thread_id: str
    source_artifact: dict[str, Any]
    source_render_artifact: dict[str, Any]
    diagnosis: dict[str, Any]
    selected_major: str
    global_proposal: dict[str, Any]
    global_render: dict[str, Any]
    local_proposal: dict[str, Any]
    subject_headroom: dict[str, Any]
    repair_count: int
    leaves: list[dict[str, Any]]


class BranchOutput(TypedDict):
    branches: list[dict[str, Any]]


def target_center(strength_bin: str) -> float:
    low, high, _inclusive = GLOBAL_DELTA_E_TARGETS[strength_bin]
    return (low + high) / 2.0


def intent_packet_order(intents: Sequence[str] = ACTIVE_LOCAL_INTENTS) -> tuple[str, ...]:
    """B11 item 3 (R7.3): the packet emission order of one mask."""
    order = {intent: index for index, intent in enumerate(intents)}
    return tuple(sorted(intents, key=lambda intent: (
        INTENT_PACKET_PRIORITY.get(intent, INTENT_PACKET_PRIORITY_DEFAULT),
        order[intent],
    )))


def local_ladder(intent: str) -> dict[str, tuple[float, float, bool]]:
    """The strength ladder of one local intent (B7 item 1)."""
    if intent not in LOCAL_DELTA_E_TARGETS:
        raise KeyError(f"unknown local intent: {intent}")
    return LOCAL_DELTA_E_TARGETS[intent]


def luma_capped(intent: str, subject_p99_luma: float | None) -> bool:
    """B7 item 2 trigger: capped intent on a subject that is already near white."""
    if intent not in LUMA_SOFT_CAP_INTENTS or subject_p99_luma is None:
        return False
    return float(subject_p99_luma) > LUMA_SOFT_CAP["p99_luma_max"]


def resolve_local_target(
    intent: str, strength_bin: str, subject_p99_luma: float | None = None,
) -> tuple[float, float, bool, str, bool]:
    """(low, high, inclusive, effective_bin, luma_capped) for one local render."""
    ladder = local_ladder(intent)
    if strength_bin not in ladder:
        raise KeyError(f"unknown local strength bin: {strength_bin}")
    capped = luma_capped(intent, subject_p99_luma)
    effective = LOCAL_BIN_DOWNSHIFT[strength_bin] if capped else strength_bin
    low, high, inclusive = ladder[effective]
    return low, high, inclusive, effective, capped


def local_target_center(
    intent: str, strength_bin: str, *, capped: bool = False
) -> float:
    ladder = local_ladder(intent)
    effective = LOCAL_BIN_DOWNSHIFT[strength_bin] if capped else strength_bin
    low, high, _inclusive = ladder[effective]
    return (low + high) / 2.0


__all__ = [
    "ACTIVE_LOCAL_INTENTS", "BAND_GEOMETRY_GATE", "BranchOutput",
    "DISABLED_LOCAL_INTENTS",
    "FINGERPRINT_FIELDS", "FINGERPRINT_FORMATS",
    "GLOBAL_DELTA_E_TARGETS", "GlobalBranchState", "INTENT_BIN_QUOTA",
    "INTENT_DIRECTION_GATE", "INTENT_FINGERPRINT_GATE", "INTENT_PACKET_PRIORITY",
    "INTENT_PACKET_PRIORITY_DEFAULT", "INTENT_ROLE_DOMAINS", "INTENT_SEGMENT_GATE",
    "INTENT_ROLES", "INTENT_V1_VARIANTS", "IntentMode", "LOCAL_BIN_DOWNSHIFT",
    "LOCAL_DELTA_E_LADDERS", "LOCAL_DELTA_E_TARGETS", "LOCAL_INTENT_LADDER",
    "LOCAL_INTENTS", "LOCAL_ONLINE_RETRIEVAL", "LOCAL_PACKET_ROW_LIMIT",
    "LOCAL_VISIBILITY_FLOOR",
    "LUMA_SOFT_CAP", "LUMA_SOFT_CAP_INTENTS", "LocalLeafInput", "MASK_REACH_GATE",
    "MainState",
    "REASON_CODES", "SUBJECT_HEADROOM_GATE", "SourceInput", "StrengthBin",
    "assert_local_visibility_floor", "intent_packet_order", "intent_serves_role",
    "local_ladder", "local_target_center", "luma_capped", "resolve_local_target",
    "target_center",
]
