"""Frozen constants for Stage-Where-A basis calibration.

Every number here is quoted from
``docs/METACANVAS_WHERE_WHAT_FINAL_EXPERIMENT_PROTOCOL_2026-08-04.md`` (cited
per line) or from a repo fact that was re-verified on 2026-08-05; nothing is
inferred from a naming convention.  Anything that the protocol does *not* pin
down is marked ``# DECISION`` and is also listed in
``experiments/Q3VL_metacanvas_where_what_20260804/where_a/NOTES.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Re-exported so the Where code and the trainer cannot drift apart.
from q3vl.train.constants import (  # noqa: F401
    IMAGE_ALIGN_FACTOR,
    IMAGE_LONG_SIDE_MAX,
    IMAGE_SHORT_SIDE,
)

SCHEMA_ORACLE = "q3vl.where_a.oracle/1"
SCHEMA_MASKVIEW = "q3vl.where_a.maskview/1"
SCHEMA_BASIS = "q3vl.where_a.basis/1"

# --- protocol 4.1: merger-pre feature --------------------------------------
# vision_config.hidden_size of Qwen3-VL-4B-Instruct (verified in config.json).
FPRE_DIM = 1024
# vision_config.patch_size; F_pre lives on an H/16 x W/16 grid.
VISION_PATCH = 16
# vision_config.spatial_merge_size; the pre-merger token order is
# (grid_h//2, grid_w//2, 2, 2) -- verified against
# transformers 4.57.1 Qwen2VLImageProcessorFast.permute(0,1,4,7,5,8,3,2,6,9)
# and Qwen3VLVisionModel.fast_pos_embed_interpolate.
SPATIAL_MERGE = 2

# --- protocol 4.2: Phi-64 ---------------------------------------------------
SEM_DIM = 64
GEO_NAMES = ("x", "y", "P2x", "P2y", "xy")
RANGE_NAMES = ("L", "S")
SEM_NAMES = tuple(f"e{i}" for i in range(1, SEM_DIM + 1))
PHI_DIR_NAMES = GEO_NAMES + RANGE_NAMES + SEM_NAMES
PHI_DIR_DIM = len(PHI_DIR_NAMES)  # 71
assert PHI_DIR_DIM == 71

# residualisation design block: [1, geo5, L, S] -> 8 columns
RESID_BLOCK_DIM = 1 + len(GEO_NAMES) + len(RANGE_NAMES)
STD_EPS = 1e-6          # DECISION: E2 precedent (e2lib.standardize)
LSTSQ_RCOND = None      # numpy/torch default

# --- protocol 4.2: s_low ----------------------------------------------------
S_SCALE = 3.0           # s = 3 * tanh(q/3)
# CLAUDE.md "s cache consumption contract": both sides must state the domain.
# PRODUCER: `s_low = 3*tanh(q/3)` lives in the OPEN interval (-3, 3) by
# construction -- there is no image, latent or projector for which it does not.
# CONSUMER: every readout is only defined/評価 on this domain; the CBand12
# centre grid is exactly linspace(-3, 3, 12), so a value outside it sits beyond
# the last centre.  The one operation that can leave the domain is the guided
# upsample (a *linear* extrapolation with the high-res guide), see
# `UpsampleConfig.clamp_domain` and REVIEW-impl-WhereA B-4.
S_DOMAIN = (-S_SCALE, S_SCALE)

# --- protocol 4.3: R-Band ---------------------------------------------------
BAND_K_LO, BAND_K_HI = 1.0, 40.0        # protocol: k in [1, 40]
# DECISION: h>0 is the only constraint in the protocol.  We reuse the E2
# bounded-sigmoid parameterisation (h in (0.02, 2.50)) instead of a softplus:
# it keeps h>0, stays inside the s in (-3,3) range where a half-width beyond
# 2.5 is already a full pass-band, and matches experiments/E2_basis_fit_20260803.
BAND_H_LO, BAND_H_HI = 0.02, 2.50

# --- protocol 4.3: R-CBand12 ------------------------------------------------
CBAND_M = 12
CBAND_MU_LO, CBAND_MU_HI = -3.0, 3.0    # fixed centres, linspace(-3, 3, 12)
CBAND_SIG_LO, CBAND_SIG_HI = 0.025, 0.30
CBAND_EPS = 1e-9                        # E2 precedent; only used by the "eps" mode
# REVIEW-impl-WhereA B-4: the literal `sum(c_i g_i) / (sum(g_i) + eps)` collapses
# to *exactly zero* wherever every Gaussian underflows the eps -- with sigma at
# its 0.025 lower bound and centres 6/11 = 0.545 apart that already happens
# halfway between two centres (g ~ 6e-27 << 1e-9), and everywhere beyond the end
# centres.  The fit never pays for it, because the low-res fit only ever sees
# z in (-3, 3) near the centres; the guided upsample then pushes z past 3.3 at
# strong edges and the mask silently drops to 0 there.
# "logsumexp" is the eps -> 0 limit of the same formula, evaluated as a softmax
# over log g_i: mathematically identical in the well-conditioned regime (asserted
# to 1e-9 in tests) and free of the collapse.  o_i enters through logsigmoid, so
# an o_i that underflows to 0.0 cannot produce -inf/NaN either.
CBAND_NORMALIZATION = "logsumexp"       # "logsumexp" | "eps" (legacy, for comparison)

# --- protocol 4.2: guided upsample -----------------------------------------
# DECISION (D5): neither radius nor eps is pinned by the protocol.  radius is
# expressed on the *low-res* grid so it is resolution independent.
# These are PROVISIONAL: REVIEW-impl-WhereA N-16 showed the E2 default (r=32,
# eps=1e-3) came from a *full-resolution, per-basis-channel* filter, i.e. exactly
# the order protocol 4.2 now forbids, so it is not transferable.  The values
# below must be re-fixed from the S2 sweep (`scripts/sweep_upsample.py`) before
# any arm runs.
GUIDED_RADIUS_LOW = 2
GUIDED_EPS = 1e-3
GUIDED_PARAMS_PROVISIONAL = True        # flipped to False when S2 fixes D5

# --- protocol 10.2: Where-A optimisation -----------------------------------
PROJECTOR_LR = 1.0e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03
SCHEDULER = "cosine"
PRECISION = "bf16_forward_fp32_oracle_fit"
EPOCHS = 1.0
MAX_GRAD_NORM = 1.0      # DECISION: not pinned for Where-A; 10.3/10.4 use 1.0.

# --- oracle fit -------------------------------------------------------------
FIT_DTYPE = "float64"                   # protocol 10.2
FIT_MAX_ITER = 120                      # E2 precedent
FIT_TOL_GRAD = 1e-9                     # fixed tolerance (protocol 10.2)
FIT_TOL_CHANGE = 1e-11
# E2 precedent.  Seeds = lsq + radial + n_random + 1 near-degenerate = n_random+3;
# R-Band doubles that (both polarities), so n_random=6 -> 9 seeds -> 18 starts for
# band and 9 for cband12.  (REVIEW-impl-WhereA N-10: the old comment said 8.)
FIT_N_RANDOM = 6
FIT_HISTORY = 20
FIT_LINE_SEARCH = "strong_wolfe"
# A fit is *rejected* (never silently replaced by a zero vector, protocol 10.2)
# when any of these hold.
FIT_REJECT_LOSS = 0.90                  # DECISION: loss = 1 - softIoU_minmax
FIT_REJECT_ALPHA = 1e-6                 # collapsed direction (s is constant)

# D2, re-adjudicated by the main agent on 2026-08-05 (REVIEW-impl-WhereA B-6):
# inner and outer objective are now THE SAME.  The "IoU must not be an
# optimisation target" red line belongs to the previous campaign's context --
# protocol 5.5 makes soft-IoU the Where main loss outright -- while keeping two
# different objectives silently broke the envelope-theorem argument that D3
# (per-batch inner fit) rests on: with inner f = 1-softIoU and outer g = MSE,
# dg/d(latent) != 0 at the inner optimum, so the dropped implicit term
# (dg/dlatent)(dlatent*/dB) is O(1), not O(inner residual).
FIT_OBJECTIVE = "soft_iou_minmax"
CALIB_OBJECTIVE = "soft_iou_minmax"

# --- protocol 4.4: the four arms -------------------------------------------
ARMS = ("BA-0-Fixed", "BA-1-Band", "BA-2-CBand12", "BA-3-Joint")
ARM_READOUTS: dict[str, tuple[str, ...]] = {
    "BA-0-Fixed": (),                      # projector not trained at all
    "BA-1-Band": ("band",),
    "BA-2-CBand12": ("cband12",),
    "BA-3-Joint": ("band", "cband12"),     # pre-registered main arm
}
ARM_TRAINS_PROJECTOR = {a: bool(ARM_READOUTS[a]) for a in ARMS}
JOINT_READOUT_WEIGHTS = {"band": 0.5, "cband12": 0.5}   # DECISION: equal
PROJECTOR_SEED = 20260804                # seeded orthogonal init (BA-0 + all)

# --- data -------------------------------------------------------------------
SFT2SEG_ROOT = Path("/mnt/nfs/bc/data/datasets/sft2seg-20260804")
SPLIT_DIR = SFT2SEG_ROOT / "splits"
BUILD_DATASET_ROOT = Path("/mnt/nfs/bc/data/datasets/sft")
LOCAL_BUILDS = ("l1", "l2", "l3", "l4", "l5", "l6")
MASK_SUFFIX = ".cgt.png"
# D1, adjudicated by the main agent on 2026-08-05 (REVIEW-impl-WhereA section 10
# concurred): `winner_confidence` ranks *which candidate won*, while Where-A's
# label is *where the candidate region is* -- orthogonal.  So calibration
# TRAINING includes `low` (75,544 local train instead of 42,752), while the
# headline V_where oracle numbers are reported on `normal` only, with `low` as
# its own stratum.  The relaxation is Where-A-only: it must not flow into
# Where-B / What evaluation GT.
EXCLUDE_WINNER_CONFIDENCE_LOW = False       # training population
HEADLINE_WINNER_CONFIDENCE = ("normal",)    # reporting population
# DECISION: degenerate masks (constant everywhere) are counted and reported but
# not dropped -- dropping them would change the population the ceiling is
# measured on.
DROP_DEGENERATE_MASKS = False
DEGENERATE_STD = 1e-4

# durable outputs (protocol 2.3: durable data on NFS, scratch is rebuildable)
WHERE_A_ROOT = Path("/mnt/nfs/bc/data/datasets/where_a-20260805")
MASKVIEW_DIR = WHERE_A_ROOT / "maskviews"
ORACLE_DIR = WHERE_A_ROOT / "oracle"
BASIS_DIR = WHERE_A_ROOT / "basis"
MASKVIEW_SHARD_BYTES = 1 * 1024**3
ORACLE_SHARD_BYTES = 1 * 1024**3

REPORT_DIR = Path(
    "/home/bc/VeraRetouch/experiments/Q3VL_metacanvas_where_what_20260804/where_a"
)
MODEL_DIR = Path("/home/bc/data/models/Qwen3-VL-4B-Instruct")
# The Base SFT run freezes patch_embed, pos_embed and all 24 vision blocks
# (q3vl/train/freeze.py), so F_pre is *identical* under the base weights and
# under any SFT checkpoint.  preflight item WA-P4b asserts this once the
# checkpoints exist.
SFT_CHECKPOINTS = (
    Path("/home/bc/data/runs/q3vl_base_sft_20260804/checkpoint-2488"),
    Path("/home/bc/data/runs/q3vl_base_sft_20260804/checkpoint-4976"),
)


@dataclass(frozen=True)
class PhiConfig:
    """Everything that changes the numeric value of ``phi_dir``."""

    sem_dim: int = SEM_DIM
    std_eps: float = STD_EPS
    # short side spans [-1, 1]; the long side spans [-AR, AR] so a pixel is
    # square in feature space (protocol 4.1 "true aspect ratio, never squashed").
    coord_mode: str = "short_side_unit"
    luma: str = "rec709"                  # E2 precedent
    saturation: str = "hsv"               # E2 precedent
    residualize: bool = True
    standardize_range: bool = True


@dataclass(frozen=True)
class UpsampleConfig:
    radius_low: int = GUIDED_RADIUS_LOW
    eps: float = GUIDED_EPS
    guide: str = "luma"                   # DECISION: scalar luma guide
    # The guided filter is an affine extrapolation from the high-res guide, so
    # it can (and measurably does) push s past +-3 at strong edges even though
    # `3*tanh` made the low-res field live strictly inside (-3, 3).  Values past
    # the last CBand centre are not a meaningful part of the s axis: the readout
    # is only defined there by extrapolation, and the un-normalised mixture
    # underflows.  Clamping restores the producer's own invariant, is a global
    # constant operation (never per-image -- that is a red line), and the
    # out-of-domain fraction is always reported rather than swallowed.
    clamp_domain: bool = True
    domain: tuple[float, float] = S_DOMAIN


@dataclass(frozen=True)
class FitConfig:
    objective: str = FIT_OBJECTIVE
    max_iter: int = FIT_MAX_ITER
    n_random: int = FIT_N_RANDOM
    tol_grad: float = FIT_TOL_GRAD
    tol_change: float = FIT_TOL_CHANGE
    history_size: int = FIT_HISTORY
    line_search_fn: str = FIT_LINE_SEARCH
    dtype: str = FIT_DTYPE
    reject_loss: float = FIT_REJECT_LOSS
    reject_alpha: float = FIT_REJECT_ALPHA
    seed: int = 0


@dataclass(frozen=True)
class CalibConfig:
    """One arm's calibration settings.

    ``objective`` is used for BOTH the inner L-BFGS latent fit and the outer
    AdamW step on ``B`` -- see the D2 note above; splitting them breaks the
    bilevel argument.
    """

    arm: str = "BA-3-Joint"
    projector_lr: float = PROJECTOR_LR
    weight_decay: float = WEIGHT_DECAY
    warmup_ratio: float = WARMUP_RATIO
    scheduler: str = SCHEDULER
    epochs: float = EPOCHS
    max_grad_norm: float = MAX_GRAD_NORM
    objective: str = CALIB_OBJECTIVE
    batch_size: int = 8
    inner_fit: FitConfig = field(default_factory=lambda: FitConfig(max_iter=40, n_random=2))
    phi: PhiConfig = field(default_factory=PhiConfig)
    upsample: UpsampleConfig = field(default_factory=UpsampleConfig)
    seed: int = PROJECTOR_SEED

    def readouts(self) -> tuple[str, ...]:
        return ARM_READOUTS[self.arm]
