"""Frozen constants for Stage-What (independent ``Q_color`` -> 48-Gaussian LUT).

Every number here is quoted from
``docs/METACANVAS_WHERE_WHAT_FINAL_EXPERIMENT_PROTOCOL_2026-08-04.md`` (section
cited per line), from a repo fact re-verified on 2026-08-05, or from a repo
precedent whose provenance is named.  Anything the protocol does *not* pin down
is marked ``# DECISION`` and is also listed in
``experiments/Q3VL_metacanvas_where_what_20260804/what/NOTES.md``.

The hidden-state convention is **imported, never re-declared** (ruling D-B2):
``H_color`` has to be read out of exactly the same place as Where-B's
``H_where``, or the two stages' language conditioning is not comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --- cross-stage contract (ruling D-B2) -------------------------------------
# q3vl/whereb/tests/test_contracts.py scans the whole package for a
# re-declaration; these names must arrive by import.
from q3vl.whereb.contracts import (  # noqa: F401
    SEGMENT_HIDDEN_FINAL_NORM as COLOR_HIDDEN_FINAL_NORM,
    SEGMENT_HIDDEN_LAYER as COLOR_HIDDEN_LAYER,
    SEGMENT_HIDDEN_RULING,
)

# Re-exported so Where and What cannot drift apart on geometry/units.
from q3vl.where.config import (  # noqa: F401
    FPRE_DIM,
    LOCAL_BUILDS,
    MODEL_DIR,
    PHI_DIR_DIM,
    PhiConfig,
    S_SCALE,
    SPLIT_DIR,
    UpsampleConfig,
)
from q3vl.whereb.config import (  # noqa: F401
    CONNECTOR_DIM,
    GLOBAL_BUILDS,
    TEXT_HIDDEN,
    ConnectorConfig,
)

SCHEMA_GTLUT = "q3vl.what.gtlut/1"
SCHEMA_ZGT = "q3vl.what.zgt/1"

# ===========================================================================
# amendment A-4 -- teacher / generated <color> context
# ===========================================================================
# Protocol 5.4 gave Stage-Where a 50/50 teacher/generated discipline; the
# original Stage-What implementation was 100% teacher-forced and never said so
# (REVIEW-impl-What NF-1).  A-4 puts What on the same footing:
#
#   training   50% GT <color> hidden / 50% Base-SFT-generated <color> hidden
#   evaluation GT and generated reported separately
#   selection  the generated-context board is the main one
#   controls   C01/C02 generate with no <where> prompt and a forced <color> prefix
CONTEXT_GT = "gt"
CONTEXT_GENERATED = "generated"
CONTEXT_MODES = (CONTEXT_GT, CONTEXT_GENERATED)
#: the board V_what selection reads (amendment A-4 item 3)
SELECTION_CONTEXT = CONTEXT_GENERATED
TEACHER_FRACTION = 0.5

# The producer owns these three, so Stage-What **imports** them rather than
# re-declaring them: the boundary, the mode vocabulary and the schema string all
# have to be byte-identical on both sides or a consumer assertion becomes a
# consumer bug.  (Both sides independently measured the same 384: Stage-What from
# 3,745 sampled records -- min 108 / p50 178 / p95 246 / p99 285 / max 324 --
# 384 = max + the two tags + ~18% margin, the margin Where-B used for its 96
# against a measured max of 79.)  ``gt_color_context`` *raises* rather than
# truncating, so a corpus sample past the boundary surfaces immediately instead
# of silently shortening a teacher context; full-corpus verification is ``WT-J9``.
from q3vl.whereb.config import (  # noqa: E402
    COLOR_CONTEXT_MAX_TOKENS,
    GENCTX_MODES,
    SCHEMA_GENCTX as SCHEMA_COLOR_GENCTX,
)
from q3vl.whereb.config import GEN_MAX_NEW_TOKENS as GEN_COLOR_MAX_NEW_TOKENS  # noqa: E402

#: How an arm's generated context is produced.  Values are the producer's
#: (``q3vl.whereb.config.GENCTX_MODES``); the names are Stage-What's, because
#: ``with_where_prefix`` reads better at a call site than ``two_segment``.
GENCTX_MODE_WITH_WHERE = "two_segment"     # prompt -> <where>..</where><color>..
GENCTX_MODE_FORCED_COLOR = "forced_color"  # prompt (no <where>) + <color> forced
if set(GENCTX_MODES) != {GENCTX_MODE_WITH_WHERE, GENCTX_MODE_FORCED_COLOR}:
    raise AssertionError(
        f"the generated-context mode vocabulary diverged: producer publishes "
        f"{GENCTX_MODES}, Stage-What consumes "
        f"{(GENCTX_MODE_WITH_WHERE, GENCTX_MODE_FORCED_COLOR)}"
    )

# ===========================================================================
# protocol 7.1 -- Q_color and the continuous style code
# ===========================================================================
N_COLOR_QUERIES = 16            # "16 learnable color queries"
M_COLOR_DIM = 512               # "M_color in R^[16 x 512]"
Z_STYLE_DIM = 1024              # "z_style in R^1024"
# DECISION: the protocol fixes the *output* shapes of the colour connector and
# says the queries "only cross-attend H_color".  It does not fix the depth, so
# the Where connector's shape (protocol 5.1: 6 pre-norm blocks, 8 heads, FFN
# 2048) is reused verbatim minus the F_pre cross-attention, which the protocol
# forbids here.  Same depth on both sides keeps "Where vs What query bank" a
# comparison of what they read, not of how deep they are.
COLOR_CONNECTOR_BLOCKS = 6
COLOR_CONNECTOR_HEADS = 8
COLOR_CONNECTOR_FFN = 2048
# DECISION: MLP(AttentionPool(M_color)) -> 1024.  One hidden layer at 2 * 512.
Z_STYLE_MLP_HIDDEN = 1024

# --- the frozen functional target encoder (protocol 7.1) --------------------
# "u(T) = flatten(T(x) - x), x a fixed 17^3 RGB grid;
#  z_gt = L2Norm(SRHT_1024(u(T) - mean_train_u))"
ZGT_GRID = 17                   # 17^3 = 4913 colours -> u has 14739 entries
ZGT_DIM = 1024
# DECISION: "a fixed public seed" without a value.  The campaign seed is used
# and published here so any reader can rebuild the identical projection.
SRHT_SEED = 20260804
# Fast Walsh-Hadamard needs a power of two >= 14739.
SRHT_PAD = 16384

# ===========================================================================
# protocol 7.2 -- Gaussian-aligned visual pooling
# ===========================================================================
V_PROJ_DIM = 256                # "V: 1024 -> 256"
# v_i = [RGB(3), Lab(3), V(F_pre)(256)] + log(sum a_i) + valid bit
POOL_FEATURE_DIM = 3 + 3 + V_PROJ_DIM + 2
POOL_EPS = 1e-6
# DECISION: a slot counts as having "no valid pixel" when its total aligned
# mass is below this.  Below it the slot falls back to the masked global pool
# (protocol 7.2 "rather than producing NaN") and its valid bit goes to 0.
POOL_VALID_TAU = 1e-6
# DECISION: pooling runs on the F_pre grid (H/16 x W/16).  ``F_pre(p)`` is only
# defined there; I_in is area-downsampled to the same grid and m_pred is the
# low-resolution readout m_low = R(s_low; rho).  Sampling natural query colours
# (protocol 9.1) still happens at full image resolution with the guided-upsampled
# m_hi -- the two uses have different requirements and the difference is
# recorded rather than blurred.
POOL_SPACE = "fpre_grid"
# DECISION: Lab entering *any* tensor in this stage is the dimensionless form
# (L/100, a/128, b/128).  The 229x gradient accident of A0 came from absolute
# Lab; there is no reason to keep the raw units anywhere.
LAB_L_SCALE = 100.0
LAB_AB_SCALE = 128.0
# max chroma of the normalised Lab space: sqrt((128/128)^2 + (128/128)^2)
CHROMA_NORM = 2.0 ** 0.5

# ===========================================================================
# protocol 7.3 -- the shared 48-slot Transformer backend
# ===========================================================================
N_SLOTS = 48
BACKEND_DIM = 512
BACKEND_HEADS = 8
BACKEND_FFN = 2048
SEED_BLOCKS = 2                 # "2 seed Transformer blocks"
REFINE_BLOCKS = 4               # "4 refinement Transformer blocks"
# DECISION: the protocol says the block "receives the projected v_i and WC
# tokens" and uses "zero-init gated residual" without saying which residuals are
# gated.  Following protocol 5.1's convention (only the newly injected modality
# is gated) every *injection* branch is gated -- cross-attn(M_color),
# cross-attn(WC tokens) and the per-slot v_i injection -- while self-attention
# and the FFN are plain residuals.  The stated purpose ("stop the visual or
# Where branch overpowering the colour semantics at the start of training") is
# satisfied by construction.
GATE_SHAPE = "scalar"
BACKEND_DROPOUT = 0.0           # DECISION: 1 epoch, no dropout anywhere
# DECISION: "the independent decoder head receives [h_i, P(z_style)]".  P is a
# linear projection of z_style; 128 keeps the head input at 640.
Z_STYLE_HEAD_PROJ = 128
# DECISION: the decoder head is Linear -> GELU -> Linear with the second Linear
# zero-initialised, so every arm starts at exactly T(x) = x (protocol 7.6's
# identity-centred parameterisation is only an identity if the raw outputs are
# zero).  The bottleneck is the knob protocol 7.5 names for equalising FG/SB.
FG_HEAD_BOTTLENECK = 128
# SB's bottleneck is *solved* so that the two arms' totals differ by <= 2%; see
# q3vl.what.generator.solve_sb_bottleneck.  None = solve at construction.
SB_HEAD_BOTTLENECK: int | None = None
PARAM_MATCH_TOLERANCE = 0.02    # protocol 7.5: "no more than 2%"

# ===========================================================================
# protocol 7.4 / 7.5 -- the two generators
# ===========================================================================
# mu 3 + Cholesky 6 + opacity 1 + existence 1 + M 9 + b 3 = 23
N_PRIM_FG = 23
# SB generates only opacity + existence + M + b = 14 per primitive
N_PRIM_SB = 14
N_GLOBAL = 12                   # global G (9) + b (3)
N_GEOMETRY = 9                  # mu 3 + Cholesky 6 (provisional / shared)
FG_TOTAL_PER_SAMPLE = N_SLOTS * N_PRIM_FG + N_GLOBAL     # 1116, protocol 7.4
SB_TOTAL_PER_SAMPLE = N_SLOTS * N_PRIM_SB + N_GLOBAL     # 684
GENERATORS = ("FG48", "SB48")

# "fixed 4 x 4 x 3 RGB anchors" (protocol 7.5).  Cell centres, not cell corners:
# mu is a sigmoid, so an anchor at 0.0 or 1.0 has no finite pre-image.
ANCHOR_GRID = (4, 4, 3)         # R x G x B -> 48
assert ANCHOR_GRID[0] * ANCHOR_GRID[1] * ANCHOR_GRID[2] == N_SLOTS

# ===========================================================================
# protocol 7.6 -- LUT function parameterisation
# ===========================================================================
# DECISION (and the one place this file departs from the protocol's *literal*
# formula).  Protocol 7.6 writes
#     T_pred(x) = clamp((I + dG) x + b_g + sum_i q_i(x) (M_i x + b_i), 0, 1)
# and says "global and local affines both use identity-centred residual
# parameterisation".  Those two statements are jointly impossible: q_i are
# normalised (sum_i q_i = 1), so with M_i = I + dM_i and dG = dM = b = 0 the
# formula gives T(x) = x + x = 2x -- exactly the failure the campaign red line
# names ("global affine G initialised to 0, not I, otherwise f(x) = 2x").
# The resolution that keeps both the red line and an identity at initialisation
# is: the *local* affines are identity-centred (their normalised mixture already
# contributes the I x term) and the *global* affine is a pure residual with
# G initialised to 0:
#     T_pred(x) = clamp(dG x + b_g + sum_i q_i(x) (M_i x + b_i), 0, 1),  M_i = I + dM_i
# which is also, term for term, the CI-pinned renderer of
# ``model/glut_repro/model_rdg.py::render`` (GLUT Eq.1-3 + the PLAN 1.1 gate).
# "identity_centered" reproduces the literal reading and exists only so the 2x
# claim can be demonstrated rather than asserted.
GLOBAL_AFFINE_MODE = "residual_zero"        # "residual_zero" | "identity_centered"

# DECISION: protocol 7.6 says "the covariance is built from a Cholesky factor
# with a softplus diagonal", while the campaign red line says a bounded sigmoid
# and forbids a *bare exp*.  softplus is not a bare exp, so the protocol's word
# is followed, with a strictly positive floor so the factor cannot approach
# singular: diag = SIGMA_LO + softplus(raw + softplus_inv(SIGMA_INIT - SIGMA_LO)).
# ``bounded_sigmoid`` is the red line's alternative (RD-G's verified
# 0.02 + 0.48 * sigmoid) and is one constant away.  Listed for the main agent.
SIGMA_PARAM = "softplus_floor"              # "softplus_floor" | "bounded_sigmoid"
SIGMA_LO = 0.02                             # RD-G precedent (PLAN 1.4)
SIGMA_SPAN = 0.48                           # only used by "bounded_sigmoid"
SIGMA_INIT = 0.20                           # value at raw = 0 in both modes
CHOL_OFF_SCALE = 0.1                        # RD-G precedent
AFFINE_SCALE = 0.1                          # M = I + 0.1 z, b = 0.1 z (RD-G)
GLOBAL_SCALE = 0.1                          # dG = 0.1 z, b_g = 0.1 z (RD-G)
# DECISION: opacity/existence biases from RD-G, whose identity CI depends on the
# gate being a *common* factor at init (it cancels in the normaliser).
OPACITY_BIAS = -2.0                         # sigmoid(z - 2) ~= 0.12
EXISTENCE_BIAS = 4.0                        # sigmoid(z + 4) ~= 0.982
MIXTURE_EPS = 1e-6                          # RD-G / GLUT eps in the normaliser

BAKE_SIZE = 33                              # protocol 7.6 / 9.4: 33^3
BAKE_CHUNK_POINTS = 8192                    # DECISION: lattice chunk, memory only

# ===========================================================================
# protocol 6 -- the four Where -> What interfaces
# ===========================================================================
# ``mask_pool``: a_i(p) = m_pred(p) * N(...) instead of a_i(p) = N(...).
# ``tokens``:    which WC tokens the backend cross-attends.
WC_INTERFACES: dict[str, dict[str, object]] = {
    "WC-0": {"tokens": ("global_vis",),
             "mask_pool": False, "where_prefix": True},
    "WC-1": {"tokens": ("global_vis", "f_roi", "f_bg"),
             "mask_pool": True, "where_prefix": True},
    "WC-2": {"tokens": ("global_vis", "z_where", "w", "rho"),
             "mask_pool": False, "where_prefix": True},
    "WC-3": {"tokens": ("global_vis", "f_roi", "f_bg", "z_where", "w", "rho"),
             "mask_pool": True, "where_prefix": True},
    # protocol 8.2 controls -- not interfaces, but they live in the same slot
    "NOWHERE": {"tokens": ("global_vis",),
                "mask_pool": False, "where_prefix": False},
    "ORACLE": {"tokens": ("global_vis", "f_roi", "f_bg", "z_where", "w", "rho"),
               "mask_pool": True, "where_prefix": True},
}
WC_IDS = ("WC-0", "WC-1", "WC-2", "WC-3")
#: WC token kinds that need the Where model's dense outputs
WC_DENSE_TOKENS = ("f_roi", "f_bg")
#: WC token kinds that need the Where model's latent outputs
WC_LATENT_TOKENS = ("z_where", "w", "rho")

# ===========================================================================
# protocol 8 -- the 12 arms
# ===========================================================================
ARMS: dict[str, tuple[str, str, str]] = {
    # arm : (WC interface, generator, where source)
    "T01": ("WC-0", "FG48", "predicted"),
    "T02": ("WC-1", "FG48", "predicted"),
    "T03": ("WC-2", "FG48", "predicted"),
    "T04": ("WC-3", "FG48", "predicted"),
    "T05": ("WC-0", "SB48", "predicted"),
    "T06": ("WC-1", "SB48", "predicted"),
    "T07": ("WC-2", "SB48", "predicted"),
    "T08": ("WC-3", "SB48", "predicted"),
    "C01": ("NOWHERE", "FG48", "none"),
    "C02": ("NOWHERE", "SB48", "none"),
    "C03": ("ORACLE", "FG48", "oracle"),
    "C04": ("ORACLE", "SB48", "oracle"),
}
ARM_IDS = tuple(ARMS)
MAIN_ARM_IDS = tuple(a for a in ARMS if a.startswith("T"))
CONTROL_ARM_IDS = tuple(a for a in ARMS if a.startswith("C"))
#: protocol 8.2: "oracle inputs exist only in the ceiling control and may not
#: enter the main board".
CEILING_ARM_IDS = ("C03", "C04")


def genctx_mode_of(arm: str) -> str:
    """Amendment A-4 item 4: which generation an arm's ``<color>`` context is.

    ``C01``/``C02`` drop the ``<where>`` prefix from the model's sequence, so
    their *generated* context has to be generated the same way -- from a prompt
    with no ``<where>`` and a forced ``<color>`` open tag.  Replaying a context
    that was generated *after* a ``<where>`` span would put the where reasoning
    back into the control arm through the token ids, which is precisely what the
    strict no-where control exists to exclude.
    """
    return (GENCTX_MODE_FORCED_COLOR if not WC_INTERFACES[ARMS[arm][0]]["where_prefix"]
            else GENCTX_MODE_WITH_WHERE)

# ===========================================================================
# protocol 9 -- the loss, symbol by symbol
# ===========================================================================
N_QUERY_UNIFORM = 1024
N_QUERY_NATURAL = 1024
N_QUERY_TOTAL = N_QUERY_UNIFORM + N_QUERY_NATURAL
# DECISION: "a fixed stratified uniform-RGB sampler" without a stratification.
# 8^3 = 512 strata x 2 jittered points = 1024, isotropic in the three channels,
# drawn once from a fixed seed and shared by every sample and every arm.
UNIFORM_STRATA = 8
UNIFORM_PER_STRATUM = 2
UNIFORM_SEED = 20260804
assert UNIFORM_STRATA ** 3 * UNIFORM_PER_STRATUM == N_QUERY_UNIFORM

CHARBONNIER_EPS = 1e-3          # DECISION: the standard value
HUBER_DELTA = 1.0               # DECISION: torch default (as in Where-B)

# protocol 9.5, verbatim
W_FUNC = 1.00
W_HC = 10.00
W_SPARSE = 0.001
W_STYLE_COS = 0.05
W_STYLE_DIST = 0.05
W_VARCOV = 0.02
W_BAKE = 0.10
LOSS_WEIGHTS = {
    "L_func": W_FUNC, "L_hc": W_HC, "R_sparse": W_SPARSE,
    "L_style_cos": W_STYLE_COS, "L_style_dist": W_STYLE_DIST,
    "L_varcov": W_VARCOV, "L_bake": W_BAKE,
}

# DECISION: single-GPU-per-arm is the protocol 11 schedule, so protocol 9.3's
# "stop-gradient FIFO statistics queue" is the operative branch.  Size, push
# rule and warm-up are fixed here for every arm.
STYLE_QUEUE_SIZE = 256          # 8 effective batches of 32
STYLE_QUEUE_MIN = 32            # below this L_var/L_cov are reported as 0
VAR_TARGET = 1.0                # L_var = mean max(0, 1 - std_d)
VAR_EPS = 1e-4                  # std = sqrt(var + eps), VICReg convention

# protocol 9.2: the mandatory gradient-norm ratio log.  Computing it needs two
# extra backward passes, so it runs on a schedule, always including step 0.
GRAD_RATIO_EVERY = 50

# ===========================================================================
# protocol 10.4 -- optimisation
# ===========================================================================
QUERY_CONNECTOR_BACKEND_LR = 1.0e-4
DECODER_HEAD_LR = 1.0e-4
SHARED_GEOMETRY_LR = 5.0e-5     # SB48 only
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03
SCHEDULER = "cosine"
MAX_GRAD_NORM = 1.0
PRECISION = "bf16"
EFFECTIVE_BATCH = 32
EPOCHS = 1.0
EVAL_STEPS = 500
SAVE_STEPS = 500
KEEP_LAST_CHECKPOINTS = 3
PROTECTED_EPOCHS = (0.5, 1.0)
SEED = 20260804

# ===========================================================================
# NF-2 ruling -- in-loop evaluation (route (a) + offline complement)
# ===========================================================================
# Protocol 10.4 fixes ``eval_steps: 500``.  Running it on the whole of V_what
# twice (both contexts) with a full image render would cost more than the 500
# training steps it interrupts, so the in-loop pass is a **fixed deterministic
# subset**, LUT-function metrics only.  The complete V_what with image metrics is
# the offline job (``scripts/evaluate_what.py``) and is what selection reads.
EVAL_SUBSET_SIZE = 256
EVAL_SUBSET_SEED = 20260804
#: mask-area strata edges (fraction of frame).  Used only when a mask source is
#: available; the subset manifest records whether it was.
MASK_AREA_BINS = (0.05, 0.20, 0.50)
#: The online proxy for protocol 12.4's primary key.  **Not** the same
#: measurement -- no image is rendered in the loop -- but the same direction:
#: CIEDE2000 on local samples, generated context, smaller is better.  It decides
#: which checkpoint *files survive* the rolling deletion; the offline board
#: decides which checkpoint *wins*.  See NOTES D-W12.
ONLINE_SELECTION_KEY = "local_lut_de00_median"

# ===========================================================================
# protocol 12 -- gates and selection
# ===========================================================================
# (metric key, comparison, threshold) -- analytic vs 33^3 tetrahedral readback
BAKE_GATE: tuple[tuple[str, str, float], ...] = (
    ("bake_mae_mean", "<=", 1e-4),
    ("bake_err_p99", "<=", 5e-4),
    ("bake_non_finite", "<=", 0.0),
)
FINITE_GATE: tuple[tuple[str, str, float], ...] = (
    ("lut_non_finite", "<=", 0.0),
)
# protocol 12.4 step 1: "instruction-shuffle positive dependence".  The shuffled
# context must make the LUT *worse*; a pre-registered minimum margin keeps
# "0.001 worse" from counting as a dependency.
# DECISION: 0.25 dE00 on the local V_what board, the same order as Where-B's
# pre-registered 0.20 soft-IoU drop.
INSTRUCTION_SHUFFLE_MIN_DELTA = 0.25

# lexicographic; True = larger is better (protocol 12.4)
SELECTION_ORDER: tuple[tuple[str, bool], ...] = (
    ("local_image_de00_median", False),
    ("lut_de00_p90", False),
    ("boundary_de00_median", False),
    ("n_trainable_params", False),
    ("latency_ms", False),
)
GATE_FAILED_TAG = "WHAT-GATE-FAILED"

# strata every report must break out (protocol 12.2)
STRATA_KEYS = ("build", "render_mode", "winner_confidence", "upscaled",
               "mask_area_bin", "l_level_bin")

# ===========================================================================
# data
# ===========================================================================
# DECISION: the same population discipline as Where-B (main-agent ruling on
# task-card item 7): winner_confidence=low stays in, always as its own stratum.
EXCLUDE_WINNER_CONFIDENCE_LOW = False
INCLUDE_GLOBAL = True

# The GT LUT function.  VERIFIED 2026-08-05 (see NOTES V-W1): every sft2seg
# record carries ``preset_path``, a real ``.cube`` (98.5%) or ``.3dl`` (1.5%)
# file under /home/bc/data/datasets/recipes, and ``lut_id`` -> ``preset_path``
# is 1:1 (481 sampled ids, 0 conflicts, 481/481 present).  ``T_gt`` is that
# table; there is no need to re-render anything.
RECIPE_ROOTS = (Path("/home/bc/data/datasets/recipes"),)
# DECISION: the interpolator that *defines* T_gt.  The dataset's I_tar was
# rendered with trilinear grid_sample (align_corners=True, border padding,
# clamp 0..1) -- dataset_build/core/render_backend.py::_render_cube_gpu and its
# CPU oracle -- so trilinear is what makes L_func's target and the final-image
# target the same function.  "tetrahedral" is the delivery-side interpolator and
# is always *reported* next to it (protocol 12.1) but is not the training GT.
GT_LUT_INTERP = "trilinear"     # "trilinear" | "tetrahedral"

#: published by the extended ``q3vl.whereb.scripts.make_generated_context``
#: (WB-IMPL); one directory per (split, mode).
COLOR_GENCTX_ROOT = Path("/mnt/nfs/bc/data/datasets/where_b-20260805/genwhere")

WHAT_ROOT = Path("/mnt/nfs/bc/data/datasets/what-20260805")
GTLUT_DIR = WHAT_ROOT / "gtluts"        # <lut_id>.lut.npy + <lut_id>.lutmeta.json
ZGT_DIR = WHAT_ROOT / "zgt"             # zgt_center.npz + per-lut z_gt
GTLUT_SHARD_BYTES = 1 * 1024**3
RUN_ROOT = Path("/home/bc/data/runs/what")
REPORT_DIR = Path(
    "/home/bc/VeraRetouch/experiments/Q3VL_metacanvas_where_what_20260804/what"
)
SFT_CHECKPOINT = Path("/home/bc/data/runs/q3vl_base_sft_20260804/checkpoint-4976")
#: filled in once Where-B has selected and frozen one checkpoint (protocol 5.6);
#: every What arm uses this same file.
WHERE_CHECKPOINT: Path | None = None


# ===========================================================================
# config objects
# ===========================================================================

@dataclass(frozen=True)
class ColorConnectorConfig:
    dim: int = M_COLOR_DIM
    n_queries: int = N_COLOR_QUERIES
    n_blocks: int = COLOR_CONNECTOR_BLOCKS
    n_heads: int = COLOR_CONNECTOR_HEADS
    ffn: int = COLOR_CONNECTOR_FFN
    dropout: float = BACKEND_DROPOUT
    text_dim: int = TEXT_HIDDEN
    z_style_dim: int = Z_STYLE_DIM
    z_style_hidden: int = Z_STYLE_MLP_HIDDEN


@dataclass(frozen=True)
class BackendConfig:
    dim: int = BACKEND_DIM
    n_slots: int = N_SLOTS
    n_heads: int = BACKEND_HEADS
    ffn: int = BACKEND_FFN
    seed_blocks: int = SEED_BLOCKS
    refine_blocks: int = REFINE_BLOCKS
    dropout: float = BACKEND_DROPOUT
    z_style_dim: int = Z_STYLE_DIM
    z_style_head_proj: int = Z_STYLE_HEAD_PROJ
    v_dim: int = POOL_FEATURE_DIM
    vision_dim: int = FPRE_DIM


@dataclass(frozen=True)
class LutConfig:
    """Everything that changes the numeric value of ``T_pred``."""

    n_slots: int = N_SLOTS
    sigma_param: str = SIGMA_PARAM
    sigma_lo: float = SIGMA_LO
    sigma_span: float = SIGMA_SPAN
    sigma_init: float = SIGMA_INIT
    chol_off_scale: float = CHOL_OFF_SCALE
    affine_scale: float = AFFINE_SCALE
    global_scale: float = GLOBAL_SCALE
    opacity_bias: float = OPACITY_BIAS
    existence_bias: float = EXISTENCE_BIAS
    mixture_eps: float = MIXTURE_EPS
    global_affine_mode: str = GLOBAL_AFFINE_MODE
    anchor_grid: tuple[int, int, int] = ANCHOR_GRID
    clamp_output: bool = True


@dataclass(frozen=True)
class ArmConfig:
    """Everything that defines one of the twelve Stage-What arms."""

    arm: str = "T01"
    seed: int = SEED
    color: ColorConnectorConfig = field(default_factory=ColorConnectorConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    lut: LutConfig = field(default_factory=LutConfig)
    upsample: UpsampleConfig = field(default_factory=UpsampleConfig)
    phi: PhiConfig = field(default_factory=PhiConfig)
    #: Where-B readout of the frozen Where checkpoint; needed to size the rho
    #: token projection.  Set from the checkpoint at load time.
    where_readout: str = "cband12"
    fg_bottleneck: int = FG_HEAD_BOTTLENECK
    sb_bottleneck: int | None = SB_HEAD_BOTTLENECK

    @property
    def wc(self) -> str:
        return ARMS[self.arm][0]

    @property
    def generator(self) -> str:
        return ARMS[self.arm][1]

    @property
    def where_source(self) -> str:
        return ARMS[self.arm][2]

    @property
    def mask_pool(self) -> bool:
        return bool(WC_INTERFACES[self.wc]["mask_pool"])

    @property
    def where_prefix(self) -> bool:
        return bool(WC_INTERFACES[self.wc]["where_prefix"])

    @property
    def wc_tokens(self) -> tuple[str, ...]:
        return tuple(WC_INTERFACES[self.wc]["tokens"])           # type: ignore[arg-type]

    @property
    def is_ceiling(self) -> bool:
        return self.arm in CEILING_ARM_IDS

    @property
    def genctx_mode(self) -> str:
        return genctx_mode_of(self.arm)


@dataclass(frozen=True)
class TrainConfig:
    arm: str = "T01"
    backend_lr: float = QUERY_CONNECTOR_BACKEND_LR
    head_lr: float = DECODER_HEAD_LR
    geometry_lr: float = SHARED_GEOMETRY_LR
    weight_decay: float = WEIGHT_DECAY
    warmup_ratio: float = WARMUP_RATIO
    scheduler: str = SCHEDULER
    max_grad_norm: float = MAX_GRAD_NORM
    precision: str = PRECISION
    effective_batch: int = EFFECTIVE_BATCH
    micro_batch: int = 4                     # probed on GPU, then GAS is derived
    epochs: float = EPOCHS
    eval_steps: int = EVAL_STEPS
    save_steps: int = SAVE_STEPS
    #: ``None`` disables rolling deletion entirely (9 checkpoints x ~0.4 GiB per
    #: arm is an affordable alternative to relying on the protections).
    keep_last: int | None = KEEP_LAST_CHECKPOINTS
    seed: int = SEED
    grad_ratio_every: int = GRAD_RATIO_EVERY
    style_queue_size: int = STYLE_QUEUE_SIZE
    teacher_fraction: float = TEACHER_FRACTION      # amendment A-4
    eval_subset_size: int = EVAL_SUBSET_SIZE        # NF-2: in-loop eval subset
    eval_micro_batch: int = 4
    #: the key ``best()`` ranks by.  The online proxy, not the offline primary.
    selection_key: str = ONLINE_SELECTION_KEY

    def grad_accum(self) -> int:
        if self.effective_batch % self.micro_batch:
            raise ValueError(
                f"effective batch {self.effective_batch} is not a multiple of "
                f"micro batch {self.micro_batch} (protocol 10.4 fixes the former)"
            )
        return self.effective_batch // self.micro_batch


def arm_config(arm: str, **kw) -> ArmConfig:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; protocol 8 defines {ARM_IDS}")
    return ArmConfig(arm=arm, **kw)
