"""Frozen constants for the Qwen3-VL-4B Arm B base SFT.

Every number here is quoted from ``docs/QWEN3_VL_BASE_SFT_SPEC_2026-08-04.md``.
Nothing in this module may be overridden by a config file: these are the
pre-registered structural decisions of the experiment.
"""

from __future__ import annotations

# --- spec 4.3: the four original special tokens, in registration order ------
WHERE_OPEN = "<where>"
WHERE_CLOSE = "</where>"
COLOR_OPEN = "<color>"
COLOR_CLOSE = "</color>"

# --- v2seg (2026-08-14): two trailing readout tokens ------------------------
# Emitted AFTER </color> and BEFORE <|im_end|>, one per reasoning segment, so
# each carries a hidden state that has already attended to the whole segment.
# They MUST stay at the END of SPECIAL_TOKENS: registration order fixes the ids,
# and q3vl/whereb/attnread.py hard-codes <where>=151669 / </where>=151670.
# Appending keeps 151669..151672 and gives the new pair 151673 / 151674.
SEG_WHERE_TOK = "<seg_where>"
SEG_COLOR_TOK = "<seg_color>"

SPECIAL_TOKENS: tuple[str, ...] = (
    WHERE_OPEN, WHERE_CLOSE, COLOR_OPEN, COLOR_CLOSE,  # ids 151669..151672
    SEG_WHERE_TOK, SEG_COLOR_TOK,                      # v2seg, ids 151673/151674
)

# --- spec 4.2: the seven legacy reasoning tags that must NOT survive --------
# Source of truth for the tag strings: dataset_build/src/construct/responses.py
LEGACY_SEGMENT_TAGS: tuple[str, ...] = (
    "<problem_light_start>", "<problem_light_end>",
    "<problem_globalcolor_start>", "<problem_globalcolor_end>",
    "<problem_specificcolor_start>", "<problem_specificcolor_end>",
    "<region_scope_start>", "<region_scope_end>",
    "<plan_light_start>", "<plan_light_end>",
    "<plan_globalcolor_start>", "<plan_globalcolor_end>",
    "<plan_specificcolor_start>", "<plan_specificcolor_end>",
)

# canonical parser field names (spec 4.2 last paragraph)
CANONICAL_WHERE_FIELD = "region_scope"
CANONICAL_COLOR_FIELDS: tuple[str, ...] = (
    "problem_lighting",
    "problem_global_color",
    "problem_specific_color",
    "plan_lighting",
    "plan_global_color",
    "plan_specific_color",
)

# --- spec 5: image preprocessing contract ----------------------------------
IMAGE_SHORT_SIDE = 512
IMAGE_LONG_SIDE_MAX = 2048
# patch_size(16) * spatial_merge_size(2); verified against the local
# config.json vision_config of Qwen3-VL-4B-Instruct.
IMAGE_ALIGN_FACTOR = 32
IMAGE_MAX_ASPECT_RATIO = 4.0
# Rounding the long side to a multiple of 32 while the short side is pinned at
# 512 can distort the aspect ratio by at most 16/512 = 3.125%.
IMAGE_ASPECT_TOLERANCE = 0.032

# --- spec 6: sequence length ----------------------------------------------
MODEL_MAX_LENGTH = 2048

# --- spec 7: optimiser / distributed --------------------------------------
GLOBAL_BATCH_SIZE = 32

# --- label / diagnostic encoding ------------------------------------------
IGNORE_INDEX = -100
SEG_IGNORE = 0  # prompt, template, image placeholders
SEG_WHERE = 1  # <where> ... </where>
SEG_COLOR = 2  # <color> ... </color>
SEG_EOS = 3  # <|im_end|>\n
SEG_SEGWHERE = 4  # v2seg: the single <seg_where> readout token
SEG_SEGCOLOR = 5  # v2seg: the single <seg_color> readout token
SEGMENT_NAMES = {
    SEG_WHERE: "where",
    SEG_COLOR: "color",
    SEG_EOS: "eos",
    SEG_SEGWHERE: "segwhere",
    SEG_SEGCOLOR: "segcolor",
}

# --- spec 2.2: Arm B trainable boundary inside model.visual ----------------
# NOTE (spec 2.2 warning): merger and deepstack mergers live *inside*
# model.visual. A blanket model.visual.requires_grad_(False) would freeze them.
VISUAL_TRAINABLE_PREFIXES: tuple[str, ...] = (
    "model.visual.merger.",
    "model.visual.deepstack_merger_list.",
)
VISUAL_ROOT_PREFIX = "model.visual."
