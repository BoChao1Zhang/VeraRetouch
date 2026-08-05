"""Frozen constants for Stage-Where-B (MetaCanvas -> global basis parameters).

Every number here is quoted from
``docs/METACANVAS_WHERE_WHAT_FINAL_EXPERIMENT_PROTOCOL_2026-08-04.md`` (section
cited per line) or from a repo/model fact that was re-verified on 2026-08-05.
Anything the protocol does *not* pin down is marked ``# DECISION`` and is also
listed in
``experiments/Q3VL_metacanvas_where_what_20260804/where_b/NOTES.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Re-exported so Where-A and Where-B cannot drift apart.
from q3vl.where.config import (  # noqa: F401
    FPRE_DIM,
    PHI_DIR_DIM,
    S_SCALE,
    SEM_DIM,
    CBAND_M,
    LOCAL_BUILDS,
    MODEL_DIR,
    PhiConfig,
    UpsampleConfig,
)
from q3vl.where.config import BASIS_DIR as WHERE_A_BASIS_DIR  # noqa: F401
from q3vl.where.config import MASKVIEW_DIR as WHERE_A_MASKVIEW_DIR  # noqa: F401
from q3vl.where.config import ORACLE_DIR as WHERE_A_ORACLE_DIR  # noqa: F401
from q3vl.where.config import SPLIT_DIR  # noqa: F401

#: v2 adds the ``<color>`` segment alongside the ``<where>`` one (amendment A-4:
#: Stage-What adopts the same 50/50 teacher/generated context split as Where-B,
#: and the Base SFT model emits both segments in a single generation anyway).
#: **Every v1 field keeps its exact name and its exact meaning** -- the v1 keys
#: still describe the ``<where>`` segment -- so a v1 consumer reading a v2
#: record sees no change.  ``q3vl/whereb/tests/test_gencontext.py`` asserts that.
SCHEMA_GENCTX = "q3vl.where_b.genwhere/2"
SCHEMA_GENCTX_V1 = "q3vl.where_b.genwhere/1"
#: the field set a v1 consumer is entitled to find in any record
SCHEMA_GENCTX_V1_FIELDS: tuple[str, ...] = (
    "schema_version", "sample_id", "split", "build", "render_mode",
    "winner_confidence", "checkpoint", "generated_ids", "generated_text",
    "n_generated_tokens", "where_ids", "where_text", "format_failure",
    "truncated", "stop_reason", "starts_with_where_open", "gen",
)

GLOBAL_BUILDS = ("g1", "g2", "g3", "g4")

# --- model facts (verified 2026-08-05 against the local config.json) --------
# text_config.hidden_size of Qwen3-VL-4B-Instruct.
TEXT_HIDDEN = 2560
# text_config.num_hidden_layers; hidden_states has L+1 entries.
TEXT_LAYERS = 36

# --- protocol 5.1: the connector ------------------------------------------
CONNECTOR_DIM = 512
CONNECTOR_BLOCKS = 6
CONNECTOR_HEADS = 8
CONNECTOR_FFN = 2048
# DECISION: the protocol fixes width/blocks/heads/FFN and the *order*
# self-attn -> xattn(H_where) -> xattn(F_pre) -> FFN, and says the cross-attn
# residual gates are zero-initialised.  It says nothing about self-attn/FFN
# gates, so those are plain residuals (no gate), which is the MetaCanvas /
# LLaMA-Adapter convention: only the newly injected modality is gated.
GATE_SHAPE = "scalar"          # DECISION: one learnable scalar per cross-attn
ATTN_DROPOUT = 0.0             # DECISION: 1 epoch, no dropout anywhere
# 2D Fourier position encoding: 2 coords x N bands x {sin, cos}
POS_BANDS = 32
POS_MAX_FREQ = 8.0             # DECISION: bands are log-spaced in [1, 8] cycles

# --- protocol 5.2: the four MetaCanvas structures --------------------------
# ``streams``: independent (query bank + input projections + connector) copies.
# ``pools``:   independent attention pools + heads (1 = joint, 2 = w / rho).
STRUCTURES: dict[str, dict[str, int | str]] = {
    "MC8-Joint":       {"canvas": 8,  "streams": 1, "pools": 1},
    "MC16-Joint":      {"canvas": 16, "streams": 1, "pools": 1},
    "MC16-SplitHead":  {"canvas": 16, "streams": 1, "pools": 2},
    "MC16-DualCanvas": {"canvas": 16, "streams": 2, "pools": 2},
}
STRUCTURE_IDS = tuple(STRUCTURES)

# --- protocol 4.3 / 5.3: readouts and the 8 main arms ----------------------
READOUTS = ("band", "cband12")
ARMS: dict[str, tuple[str, str]] = {
    "W01": ("MC8-Joint", "band"),
    "W02": ("MC8-Joint", "cband12"),
    "W03": ("MC16-Joint", "band"),
    "W04": ("MC16-Joint", "cband12"),
    "W05": ("MC16-SplitHead", "band"),
    "W06": ("MC16-SplitHead", "cband12"),
    "W07": ("MC16-DualCanvas", "band"),
    "W08": ("MC16-DualCanvas", "cband12"),
}
ARM_IDS = tuple(ARMS)

# --- protocol 5.4: context -------------------------------------------------
CONTEXT_MODES = ("gt", "generated", "null", "shuffled")
TRAIN_CONTEXT_MODES = ("gt", "generated")
TEACHER_FRACTION = 0.5          # protocol 5.4: batch is fixed 50/50
# Fixed token boundary used when a generated segment has no closing tag.
# Measured on V_where+V_what+T_final (2711 records, ``tokens.where``):
# local max 79, global max 30, p99 60.  81 = 79 body + <where> + </where>, so
# 96 never truncates a GT segment and is a hard bound for a runaway generation.
WHERE_CONTEXT_MAX_TOKENS = 96
# Same rule for the <color> segment (amendment A-4).  Measured on the same 2711
# records (``tokens.color``): min 108, p50 178, p95 244, p99 283, **max 324**;
# 324 + <color> + </color> = 326, so 384 never truncates a GT colour segment.
COLOR_CONTEXT_MAX_TOKENS = 384
# Generation budget.  A two-segment continuation is where + colour + 4 tags:
# measured ``tokens.where + tokens.color`` max 332, p99 296 -> 336 with tags.
# 512 leaves headroom and matches the Base SFT spec's gen_diag_max_new_tokens.
GEN_MAX_NEW_TOKENS = 512
# What the where-only pipeline used before amendment A-4; kept for reference.
GEN_MAX_NEW_TOKENS_WHERE_ONLY = 128
#: generation modes of scripts/make_generated_context.py
GENCTX_MODES = ("two_segment", "forced_color")
# DECISION: shuffle partners come from the same source image and the same
# render mode ("same image, same local level", protocol 5.4).  Measured on
# V_where: 4.5% of local samples have no partner under this rule (23% under the
# stricter (source_image_id, build) reading), and those are reported as
# uncovered rather than paired across images.
SHUFFLE_GROUP_KEYS = ("source_image_id", "render_mode")

# --- protocol 5.1 / 14.8: the H_where extraction contract ------------------
# Ruling D-B2 lives in q3vl/whereb/contracts.py, NOT here: it is a *cross-stage*
# contract and Stage-What has to read H_color out of the same place (review nit
# N1).  These are re-exports so existing call sites keep working.
from .contracts import (  # noqa: E402
    SEGMENT_HIDDEN_FINAL_NORM as WHERE_HIDDEN_FINAL_NORM,
    SEGMENT_HIDDEN_LAYER as WHERE_HIDDEN_LAYER,
    SEGMENT_HIDDEN_RULING,  # noqa: F401
)

# --- protocol 5.5: the loss ------------------------------------------------
MASK_IOU_W = 1.00
MASK_BCE_W = 0.25
MASK_BF1_W = 0.10
# DECISION: soft-IoU is the min/max form (PLAN v2 L205 / E2 / Where-A
# ``soft_iou_minmax``), so the Where-B loss, the Where-B gate and the Where-A
# oracle ceiling are all the *same* number and the "relative to oracle" gate is
# meaningful.
SOFT_IOU_KIND = "minmax"
# Boundary-F1 surrogate of Bokhovkin & Burnaev, arXiv:1905.07852 (verified
# 2026-08-05, see NOTES V-B6): boundary map = pool(1-y, 3) - (1-y); tolerance
# map = pool(boundary, theta); P = sum(b_pd * b_gt_ext)/sum(b_pd);
# R = sum(b_gt * b_pd_ext)/sum(b_gt); loss = 1 - 2PR/(P+R).
BOUNDARY_KERNEL = 3             # boundary extraction window (paper's theta0)
BOUNDARY_TOL_PX = 3             # "3px" tolerance -> pooling radius 3
CURVE_Z_LO, CURVE_Z_HI, CURVE_Z_N = -3.0, 3.0, 257   # protocol 5.5
HUBER_DELTA = 1.0               # DECISION: torch default for Huber
# two-段 schedule (protocol 5.5)
STAGE1_FRACTION = 0.30
STAGE1_WEIGHTS = {"s": 1.00, "curve": 1.00, "dir": 0.10}
STAGE2_WEIGHTS = {"s": 0.25, "curve": 0.25, "dir": 0.05}
MASK_WEIGHT = 1.00              # constant in both stages
# RULING D-B16 (2026-08-05, review blocker B5): the oracle auxiliaries are
# normalised by the number of samples that *have* an oracle, not by the batch
# size.  Averaging them over the whole batch would multiply the protocol's
# nominal weights by the local fraction of the split (train is 47.45% local),
# so stage-1's "1.00 L_s + 1.00 L_curve" would in fact run at ~0.47 -- closer to
# stage-2's 0.25 than to stage-1, destroying the very contrast stage 1 exists
# for.  ``"batch"`` restores the old behaviour and is kept only so the effect
# can be reproduced.
AUX_DENOMINATOR = "with_oracle"          # "with_oracle" | "batch"
# DECISION: L_mask is evaluated at the spec-5 image resolution, reached through
# the one sanctioned guided upsample of the scalar s (protocol 4.2).  A 3px
# boundary tolerance is meaningless on the 32x48 F_pre grid, and the gate in 5.6
# is stated against the ``.cgt`` mask.  Low-resolution mask metrics are still
# logged next to it.
MASK_LOSS_SPACE = "hi"

# --- protocol 10.3: optimisation ------------------------------------------
LEARNING_RATE = 2.0e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03
SCHEDULER = "cosine"
MAX_GRAD_NORM = 1.0
PRECISION = "bf16"
EFFECTIVE_BATCH = 32
EPOCHS = 1.0
EVAL_STEPS = 500
SAVE_STEPS = 500
# DECISION: 10.3 does not repeat 10.4's "no weight decay on bias / LayerNorm";
# the standard convention is applied and the flag is explicit.
NO_DECAY_ON_BIAS_NORM = True
SEED = 20260804

# --- protocol 5.6 + amendment A-5: gates ------------------------------------
# A-5 (2026-08-05, user red line in CLAUDE.md) revises the CRITERIA only:
#   - the `AUC_target >= 0.80` row is DELETED.  AUC is banned campaign-wide as a
#     spatial-field criterion: a fixed "the main subject" phrase scored 0.907
#     against AUC_target 0.523; a zero-parameter centre prior scored 0.836 and
#     beat all six RO-9c readouts; and MCQ-L's three conditions moved AUC by
#     0.003 while soft-IoU moved by 0.10 and PSNR_in by 2.2 dB.
#   - the "3px boundary F1 / oracle" row becomes GRID-level boundary F1.  The
#     pixel-level 3px version mostly measures boundary *length*: a random top-k
#     field scores 0.0394 on it against a centre prior's 0.0327.
#   - a CENTRE-PRIOR BASELINE column is added, on the same support and the same
#     matched-area top-k rule, with a paired delta and a p-value.  Any claim that
#     the field located the subject has to clear this, not just clear zero.
# The §5.5 LOSS IS UNTOUCHED (protocol 9.5/10.4 forbid changing a loss after the
# fact); `boundary_f1_loss` keeps its pixel-level 3px form.  A-5 moves criteria.
GATES: tuple[tuple[str, str, float], ...] = (
    ("local_soft_iou_median", ">=", 0.75),
    ("soft_iou_vs_oracle_ratio", ">=", 0.85),
    ("local_soft_iou_p10", ">=", 0.55),
    ("grid_boundary_f1_vs_oracle_ratio", ">=", 0.75),
    ("center_prior_delta_hard_iou", ">", 0.0),
    ("center_prior_delta_hard_iou_p", "<=", 0.05),
    ("instruction_shuffle_iou_drop", ">=", 0.20),
    ("s_std_ratio_median", ">=", 0.60),
    ("global_soft_iou", ">=", 0.98),
    ("gt_generated_iou_gap", "<=", 0.05),
)
#: thresholding rule for every spatial field, no exceptions (red line)
TOPK_RULE = "match_gt_area"
#: tolerance for the grid-level boundary F1, in GRID CELLS (not pixels)
GRID_BOUNDARY_TOL_CELLS = 1
#: the three negative controls instruction-conditionality must carry (A-5).
#: `shuffled` is the protocol 5.4 control; the other two are added by A-5 and
#: are produced by the same swap machinery with a different replacement text.
INSTRUCTION_NEGATIVE_CONTROLS = ("shuffled", "irrelevant_words", "fixed_phrase")
# lexicographic order; ``True`` = larger is better
SELECTION_ORDER: tuple[tuple[str, bool], ...] = (
    ("local_soft_iou_median", True),
    ("grid_boundary_f1", True),
    ("local_soft_iou_p10", True),
    ("n_trainable_params", False),
    ("peak_memory_gib", False),
    ("latency_ms", False),
)
GATE_FAILED_TAG = "WHERE-GATE-FAILED"

# --- data ------------------------------------------------------------------
# DECISION: Where-B trains on the full protocol 2.1 data range (g1-g4 + l1-l6).
# The 5.6 gate "global mask soft-IoU >= 0.98" cannot be met by a model that has
# never seen a global sample.  Global samples have no ``.cgt`` mask and no
# Where-A oracle latent: their GT mask is all-ones and the oracle auxiliaries
# are masked out for them (L_mask still applies).
INCLUDE_GLOBAL = True
# MAIN-AGENT RULING (task card item 7): winner_confidence=low is kept, in line
# with Base SFT and Where-A.  It is always reported as its own stratum.
EXCLUDE_WINNER_CONFIDENCE_LOW = False
BASIS_ARM = "BA-3-Joint"        # protocol 4.4: the frozen projector for Where-B

WHERE_B_ROOT = Path("/mnt/nfs/bc/data/datasets/where_b-20260805")
GENCTX_DIR = WHERE_B_ROOT / "genwhere"
GENCTX_SHARD_BYTES = 1 * 1024**3
RUN_ROOT = Path("/home/bc/data/runs/where_b")
REPORT_DIR = Path(
    "/home/bc/VeraRetouch/experiments/Q3VL_metacanvas_where_what_20260804/where_b"
)
SFT_CHECKPOINT = Path("/home/bc/data/runs/q3vl_base_sft_20260804/checkpoint-4976")

# strata every report must break out (task card item 7 + reviewer request)
STRATA_KEYS = ("upscaled", "winner_confidence", "build", "render_mode")


@dataclass(frozen=True)
class ConnectorConfig:
    dim: int = CONNECTOR_DIM
    n_blocks: int = CONNECTOR_BLOCKS
    n_heads: int = CONNECTOR_HEADS
    ffn: int = CONNECTOR_FFN
    dropout: float = ATTN_DROPOUT
    pos_bands: int = POS_BANDS
    pos_max_freq: float = POS_MAX_FREQ
    text_dim: int = TEXT_HIDDEN
    vision_dim: int = FPRE_DIM


@dataclass(frozen=True)
class ArmConfig:
    """Everything that defines one of the eight main arms."""

    arm: str = "W01"
    seed: int = SEED
    connector: ConnectorConfig = field(default_factory=ConnectorConfig)
    phi: PhiConfig = field(default_factory=PhiConfig)
    upsample: UpsampleConfig = field(default_factory=UpsampleConfig)
    basis_arm: str = BASIS_ARM
    mask_loss_space: str = MASK_LOSS_SPACE

    @property
    def structure(self) -> str:
        return ARMS[self.arm][0]

    @property
    def readout(self) -> str:
        return ARMS[self.arm][1]

    @property
    def canvas(self) -> int:
        return int(STRUCTURES[self.structure]["canvas"])

    @property
    def n_streams(self) -> int:
        return int(STRUCTURES[self.structure]["streams"])

    @property
    def n_pools(self) -> int:
        return int(STRUCTURES[self.structure]["pools"])


@dataclass(frozen=True)
class TrainConfig:
    arm: str = "W01"
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    warmup_ratio: float = WARMUP_RATIO
    scheduler: str = SCHEDULER
    max_grad_norm: float = MAX_GRAD_NORM
    precision: str = PRECISION
    effective_batch: int = EFFECTIVE_BATCH
    micro_batch: int = 4                 # probed on GPU, then GAS is derived
    epochs: float = EPOCHS
    eval_steps: int = EVAL_STEPS
    save_steps: int = SAVE_STEPS
    seed: int = SEED
    no_decay_on_bias_norm: bool = NO_DECAY_ON_BIAS_NORM
    teacher_fraction: float = TEACHER_FRACTION

    def grad_accum(self) -> int:
        if self.effective_batch % self.micro_batch:
            raise ValueError(
                f"effective batch {self.effective_batch} is not a multiple of "
                f"micro batch {self.micro_batch} (protocol 10.3 fixes the former)"
            )
        return self.effective_batch // self.micro_batch


def arm_config(arm: str, **kw) -> ArmConfig:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; protocol 5.3 defines {ARM_IDS}")
    return ArmConfig(arm=arm, **kw)
