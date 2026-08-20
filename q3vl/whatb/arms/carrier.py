"""EPR-024 -- the instruction-conditioned CGLUT carrier arm (``CARRIER``).

The one structural change (``EPR-024:11-14``): CGLUT's condition, a per-LUT
learnable lookup ``e_l in R^64``, is replaced by the frozen VLM's ``<seg_color>``
(id 151674) last-layer hidden put through one projection
``pi = Linear(LayerNorm(2560) -> d)``.  **Everything else is transcribed
verbatim** -- the 3-layer shared encoder + five parameter heads (App A.2), the
GLUT Eq.1-5 forward with the official demo's numerics, the loss ladder
``L_rec + 10 L_hc + 0.001 R_sparse``, the 128^3 colour sampling, Adam + cosine
1e-3 over 40 epochs, the 8192-colour step, the 0.1x learning-rate group and the
epoch-5-to-20 10%->40% hard-example mining.

Nothing here re-implements a shared object.  The carrier is
:mod:`q3vl.whatb.glut`, the generator :mod:`q3vl.whatb.generator`, CIELab/dE00
:mod:`q3vl.whatb.colorimetry`, the LUT operator :mod:`q3vl.whatb.lutdata`, the
board :mod:`q3vl.whatb.criteria`, the publication gate
:mod:`q3vl.whatb.publish`, and the degeneracy guard
:mod:`q3vl.whatb.degeneracy`.  This module is the arm: the wiring, the four
losses, the mining, the evaluation protocol and the run record.

Frozen block, honoured item by item
-----------------------------------
======================================  =================================
train ``normal``-only ``n = 93934``     :data:`q3vl.whatb.splits.TRAIN_NORMAL_N`
``B = 32`` x ``Q = 256`` = 8192/step    :data:`q3vl.whatb.queries.COLORS_PER_STEP`
2936 steps/epoch, 117,440 total         :meth:`CarrierConfig.total_steps`
``--clamp`` default ``two``             :class:`q3vl.whatb.glut.GlutCarrier`
headline ``Î=(1-a)I + a f̂(I)``          :func:`q3vl.whatb.criteria.compose_hat`
the twelve pre-registered keys          :data:`q3vl.whatb.criteria.PREREGISTERED_KEYS`
own colour-span encoder + assertion     :mod:`q3vl.whatb.colorspan`
======================================  =================================

The three where-side failures this arm is structurally closed against
---------------------------------------------------------------------
1. **Assertion contract.**  Publication goes through
   :func:`q3vl.whatb.publish.assert_publishable`, which fetches the first
   ``steps.jsonl`` row itself (caller -> disk -> in-process witness) and raises
   *different* exceptions for "nobody handed me a row" and "the row has no loss
   columns".  :func:`train_step` calls ``record_step_witness`` on step 0, so the
   third tier is always populated even before the trainer flushes.
2. **device/dtype.**  Every tensor entering a forward is moved with
   ``.to(device=ref.device, dtype=ref.dtype)``; the two constant query grids are
   ``register_buffer(..., persistent=False)`` on the model, and there is no bare
   ``torch.tensor(...)`` in any ``forward``.
3. **Constant fields.**  :func:`quick_eval` calls
   :func:`~q3vl.whatb.degeneracy.assert_transform_not_degenerate` at the FIRST
   quick eval; a flat / identity / sample-invariant transform leaves via
   ``SystemExit(2)`` with the three measured numbers printed.  The thresholds
   travel into ``run_setup.json``.

Every criterion is computed on the device the tensors are already on -- no
``.cpu()`` anywhere on a metric path (the where side moved an IoU by 0.296 that
way, and a colour metric would hide it completely).
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from q3vl.whatb import criteria as C
from q3vl.whatb.colorimetry import chroma_hue, srgb_to_lab
from q3vl.whatb.degeneracy import (
    DegeneracyThresholds,
    assert_transform_not_degenerate,
)
from q3vl.whatb.gate import identity_gate
from q3vl.whatb.generator import (
    SEG_COLOR_HIDDEN_DIM,
    CGLUTGenerator,
    SegColorProjection,
    generator_param_count,
)
from q3vl.whatb.glut import (
    EPS,
    CLAMP_FLAG_CHOICES,
    GlutAux,
    GlutCarrier,
    GlutParams,
    n_params_glut,
)
from q3vl.whatb.guards import record_step_witness
from q3vl.whatb.lutdata import BANK_DIR, LutBank, apply_lut_volume, mix_alpha
from q3vl.whatb.publish import assert_publishable, step_columns_for
from q3vl.whatb.queries import (
    BATCH_SAMPLES,
    COLORS_PER_STEP,
    DEFAULT_SEED,
    QUERIES_PER_SAMPLE,
    QuerySampler,
    image_histogram_colors,
    mining_ratio,
    select_hard,
    uniform_grid,
)
from q3vl.whatb.readout import WHATB_READOUT_KINDS, ReplyPlan, WhatReadoutSpec, verify_plan
from q3vl.whatb.evaldata import (   # the ONE image / GT-alpha loader
    SHORT_SIDE,
    SampleStore,
    read_member as _read_member,
    resize_short_side as _resize_short_side,
)
from q3vl.whatb.splits import (
    DATA_CHOICES,
    DATASET_ROOT,
    IndexRow,
    active_dataset_version,
    ro_path,
)
from q3vl.whatb.zcache import (            # the ONE z cache (HANDOFF 步骤 0-7)
    CONTROL_TAGS,
    ZCACHE_FIELDS,
    ZCache,
    ZCacheDir,
    SyntheticZCache,
    write_z_cache,
)

__all__ = [
    "ARM",
    "ARM_NAME",
    "AXES",
    "BASE_CHECKPOINT",
    "BATCH_SPLITS",
    "BATCH_SPLIT_COLORS",
    "FROZEN_BATCH_SPLITS",
    "CarrierConfig",
    "CarrierModel",
    "CarrierLoss",
    "compute_loss",
    "hue_chroma_loss",
    "opacity_entropy",
    "build_optimizer",
    "build_scheduler",
    "mine_colors",
    "params_slice",
    "image_loss_terms",
    "train_step",
    "ZCache",
    "write_z_cache",
    "EvalSample",
    "SampleStore",
    "bake_transform_volume",
    "library_mean_volume",
    "quick_eval",
    "evaluate_samples",
    "interpolation_columns",
    "build_arm_board",
    "loss_preregistration",
    "run_setup_record",
    "add_arguments",
    "config_from_args",
]

# --------------------------------------------------------------------------- #
# 0. identity + the frozen numbers
# --------------------------------------------------------------------------- #
#: the board's arm name; ``criteria.ARM_AXES`` keys on it
ARM = "EPR-024"
#: the queue / CLI short name
ARM_NAME = "CARRIER"
#: P1 arm -> required table = the twelve keys + interp_grid/path_len/mono_rate/oob_rate
AXES: tuple[str, ...] = ("P1",)

BASE_CHECKPOINT = "/home/bc/data/runs/q3vl_base_sft_v2seg_20260814/checkpoint-4976"

#: ``--batch-split`` -> ``(B samples, Q colours per sample)``.
#:
#: ``32x256`` is the frozen value and stays the default; ``64x128`` is EPR-024's
#: own ablation row and is NOT step-matched to the main board (EPR-024:882).
#: Both carry ``B * Q == COLORS_PER_STEP == 8192`` and are the only two entries
#: step-matched to the EPR-024..029 board.
#:
#: The entries under "capacity rows" were opened up on 2026-08-16 (EPR-030): the
#: product ``B * Q`` is no longer pinned to 8192, it is **recorded** per entry in
#: :data:`BATCH_SPLIT_COLORS` and written into ``run_setup.json``.  A run on one
#: of these rows is on its own horizon (``steps_per_epoch = ceil(n / B)`` moves
#: with ``B``) and is not comparable, step for step, with anything measured on
#: the frozen pair.
#:
#: Every capacity row below was measured on one H100-95GiB, 20 training steps,
#: ``--backbone qdec --qdec-dim 256 --head-init zero --qdec-head-lr-scale 0.01
#: --data v2seg+l8 --loss l0``; the ``max_memory_reserved`` / s-per-step table is
#: in ``experiments/prs/EPR-030_shared-query-backbone/PROPOSAL.md``.  No entry
#: here OOMed under that configuration.
BATCH_SPLITS: dict[str, tuple[int, int]] = {
    # -- the frozen pair: B * Q == 8192 --------------------------------------
    "32x256": (BATCH_SAMPLES, QUERIES_PER_SAMPLE),
    "64x128": (64, 128),
    # -- EPR-030 capacity rows: B * Q > 8192 ---------------------------------
    "32x2048": (32, 2048),            # 65,536 colours/step
    "64x4096": (64, 4096),            # 262,144
    "128x4096": (128, 4096),          # 524,288
    "128x8192": (128, 8192),          # 1,048,576
    "256x8192": (256, 8192),          # 2,097,152
    "256x16384": (256, 16384),        # 4,194,304
    "256x24576": (256, 24576),        # 6,291,456
    "512x12288": (512, 12288),        # 6,291,456
    "256x32768": (256, 32768),        # 8,388,608
    "512x16384": (512, 16384),        # 8,388,608
    "1024x8192": (1024, 8192),        # 8,388,608
    "512x24576": (512, 24576),        # 12,582,912
    "512x28672": (512, 28672),        # 14,680,064
    "512x32768": (512, 32768),        # 16,777,216
    "1024x16384": (1024, 16384),      # 16,777,216
}

#: the two splits that carry exactly ``COLORS_PER_STEP`` colours per step
FROZEN_BATCH_SPLITS: tuple[str, ...] = ("32x256", "64x128")

#: every entry's own colour budget ``B * Q``.  Recorded, not asserted equal to a
#: constant -- that assertion is what pinned the step to 8192.
BATCH_SPLIT_COLORS: dict[str, int] = {name: b * q
                                      for name, (b, q) in BATCH_SPLITS.items()}

# the name is the factoring, so a typo cannot silently run a different batch
assert all(name == f"{b}x{q}" for name, (b, q) in BATCH_SPLITS.items())
# the frozen pair is still exactly CGLUT's own colour batch -- unchanged
assert all(BATCH_SPLIT_COLORS[name] == COLORS_PER_STEP
           for name in FROZEN_BATCH_SPLITS)
assert BATCH_SPLITS["32x256"] == (32, 256) and BATCH_SPLITS["64x128"] == (64, 128)

EPOCHS = 40                    # GLUT App A.1, CGLUT row
LAMBDA_HC = 10.0               # App A.1
LAMBDA_SPARSE = 0.001          # App A.1
BASE_LR = 1e-3                 # App A.1 "starting from 10^-3"
PROJ_LR_SCALE = 0.1            # App A.1 "0.1x the base rate" for the condition side
MAX_GRAD_NORM = 1.0            # NOT in the paper; repo convention, declared as a deviation
ADAM_BETAS = (0.9, 0.999)      # NOT in the paper; PyTorch defaults, declared
HC_EPS = 1e-3                  # frozen block, NOVEL numeric
MINING_START_EPOCH, MINING_END_EPOCH = 5, 20
MINING_R_START, MINING_R_END = 0.10, 0.40
GRID_N_HEADLINE = 17           # X_grid of the function-value column
GRID_N_FLOOR = 9               # the 9^3 grid the pre-registered floors were measured on
HIST_BITS, HIST_TOPK = 5, 4096  # X_img
INTERP_ALPHAS: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)   # GLUT App B.3 grid
INTERP_K = 20                  # IP-B path resolution
STRENGTH_U: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)         # §4.G

#: control tag -> (per-sample E key, per-sample M key) understood by build_board
CONTROL_ROW_KEYS: dict[str, tuple[str, str]] = {
    "shuffle": ("E_N1_shuffle", "M_N1_shuffle"),
    "irrelevant": ("E_N2_irrelevant", "M_N2_irrelevant"),
    "const": ("E_N3_const", "M_N3_const"),
}


# --------------------------------------------------------------------------- #
# 1. configuration (the §3.6-⑤ flag surface, frozen into run_setup.json)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CarrierConfig:
    """Every ``--flag`` of EPR-024 §3.6-⑤ plus the numbers they imply."""

    readout: str = "seg_color"
    readout_qtok: int = 0
    cond_dim: int = 64
    n_gauss: int = 48
    gen_width: int = 128
    loss_level: int = 3
    lambda_img: float = 0.0
    clamp: str = "two"
    batch_split: str = "32x256"
    hc_eps: float = HC_EPS
    hc_mask: bool = True
    context: str = "generated"
    cond_zero: bool = False
    cond_trainmean: bool = False
    lut_resample: str = "none"
    mining: bool = True
    # -- optimisation (App A.1; every deviation is named in run_setup) --
    base_lr: float = BASE_LR
    proj_lr_scale: float = PROJ_LR_SCALE
    epochs: int = EPOCHS
    max_grad_norm: float = MAX_GRAD_NORM
    seed: int = DEFAULT_SEED
    precision: str = "auto"
    #: which training corpora the population is drawn from (``--data``).  The
    #: default is the frozen sft2seg split alone; an arm that unions a second
    #: source re-declares the field's default and passes ``train_n`` measured.
    data: str = "v2seg"
    #: the active index口径's declared sft2seg train normal-only n
    #: (``q3vl.whatb.splits.DATASET_VERSIONS``); a runner overrides it with the
    #: population it measured.  Not a literal: the number is口径-dependent
    #: (v20260804 = 93934, cut-p45 = 80269).
    train_n: int = field(
        default_factory=lambda: active_dataset_version().train_normal_n)
    max_steps: int | None = None
    # -- evaluation --
    lib_size: int = 1137
    n_repeats: int = 8
    bake_grid: int = 65
    interp_pairs: int = 120
    strength_column: bool = True
    interp_mix_point: str = "post_pi"

    def __post_init__(self) -> None:
        if self.readout not in WHATB_READOUT_KINDS:
            raise ValueError(f"--readout {self.readout!r} not in {WHATB_READOUT_KINDS}")
        if self.clamp not in CLAMP_FLAG_CHOICES:
            raise ValueError(f"--clamp must be one of {CLAMP_FLAG_CHOICES}, got {self.clamp!r}")
        if self.batch_split not in BATCH_SPLITS:
            raise ValueError(f"--batch-split must be one of {sorted(BATCH_SPLITS)}")
        if self.loss_level not in (1, 2, 3, 4):
            raise ValueError(f"--loss-level must be 1..4, got {self.loss_level}")
        if self.context not in ("teacher", "generated"):
            raise ValueError(f"--context must be teacher|generated, got {self.context!r}")
        if self.lut_resample != "none":
            raise ValueError(
                "--lut-resample only implements 'none' (ruling 11.1-3): resampling "
                "changes y and the function-space target stops matching the "
                "dataset's own generation law")
        if self.cond_zero and self.cond_trainmean:
            raise ValueError("--cond-zero and --cond-trainmean are two different ablation rows")
        if self.precision not in ("auto", "bf16", "fp32"):
            raise ValueError(f"--precision must be auto|bf16|fp32, got {self.precision!r}")
        if self.interp_mix_point not in ("post_pi", "pre_pi"):
            raise ValueError("--interp-mix-point must be post_pi|pre_pi")
        if self.data not in DATA_CHOICES:
            raise ValueError(f"--data must be one of {DATA_CHOICES}, got {self.data!r}")

    # -- derived --
    @property
    def batch_samples(self) -> int:
        return BATCH_SPLITS[self.batch_split][0]

    @property
    def queries_per_sample(self) -> int:
        return BATCH_SPLITS[self.batch_split][1]

    @property
    def colors_per_step(self) -> int:
        return self.batch_samples * self.queries_per_sample

    @property
    def steps_per_epoch(self) -> int:
        """``ceil(93934 / B)`` -- one epoch is one pass over the train samples."""
        return int(math.ceil(self.train_n / self.batch_samples))

    @property
    def total_steps(self) -> int:
        """2936 x 40 = 117,440 on the frozen split (U4's common horizon)."""
        return int(self.max_steps) if self.max_steps else self.steps_per_epoch * self.epochs

    @property
    def lambda_hc(self) -> float:
        return LAMBDA_HC if self.loss_level >= 2 else 0.0

    @property
    def lambda_sparse(self) -> float:
        return LAMBDA_SPARSE if self.loss_level >= 3 else 0.0

    @property
    def lambda_img_effective(self) -> float:
        return float(self.lambda_img) if self.loss_level >= 4 else 0.0

    @property
    def step_columns(self) -> tuple[str, ...]:
        return step_columns_for(self.loss_level)

    def epoch_of(self, step: int) -> float:
        return float(step) / float(max(1, self.steps_per_epoch))

    def as_dict(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d.update({
            "batch_samples": self.batch_samples,
            "queries_per_sample": self.queries_per_sample,
            "colors_per_step": self.colors_per_step,
            "steps_per_epoch": self.steps_per_epoch,
            "total_steps": self.total_steps,
            "lambda_hc": self.lambda_hc,
            "lambda_sparse": self.lambda_sparse,
            "lambda_img_effective": self.lambda_img_effective,
            "adam_betas": list(ADAM_BETAS),
            "step_columns": list(self.step_columns),
        })
        return d


def add_arguments(ap) -> None:
    """Register the EPR-024 §3.6-⑤ flag surface on an ``argparse`` parser."""
    g = ap.add_argument_group("EPR-024 carrier arm")
    g.add_argument("--readout", default="seg_color", choices=list(WHATB_READOUT_KINDS))
    g.add_argument("--readout-qtok", type=int, default=0)
    g.add_argument("--cond-dim", type=int, default=64)
    g.add_argument("--n-gauss", type=int, default=48)
    g.add_argument("--gen-width", type=int, default=128, choices=[128, 64])
    g.add_argument("--loss-level", type=int, default=3, choices=[1, 2, 3, 4])
    g.add_argument("--lambda-img", type=float, default=0.0)
    g.add_argument("--clamp", default="two", choices=list(CLAMP_FLAG_CHOICES))
    g.add_argument("--batch-split", default="32x256", choices=sorted(BATCH_SPLITS))
    g.add_argument("--hc-eps", type=float, default=HC_EPS)
    g.add_argument("--hc-mask", dest="hc_mask", action="store_true", default=True)
    g.add_argument("--no-hc-mask", dest="hc_mask", action="store_false")
    g.add_argument("--context", default="generated", choices=["teacher", "generated"])
    g.add_argument("--cond-zero", action="store_true")
    g.add_argument("--cond-trainmean", action="store_true")
    g.add_argument("--lut-resample", default="none", choices=["none", "33"])
    g.add_argument("--mining", dest="mining", action="store_true", default=True)
    g.add_argument("--no-mining", dest="mining", action="store_false")
    g.add_argument("--epochs", type=int, default=EPOCHS)
    g.add_argument("--max-steps", type=int, default=None)
    g.add_argument("--base-lr", type=float, default=BASE_LR)
    g.add_argument("--proj-lr-scale", type=float, default=PROJ_LR_SCALE)
    g.add_argument("--max-grad-norm", type=float, default=MAX_GRAD_NORM)
    g.add_argument("--seed", type=int, default=DEFAULT_SEED)
    g.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp32"])
    g.add_argument("--lib-size", type=int, default=1137)
    g.add_argument("--bake-grid", type=int, default=65)
    g.add_argument("--interp-pairs", type=int, default=120)
    g.add_argument("--no-strength-column", dest="strength_column",
                   action="store_false", default=True)


def config_from_args(args) -> CarrierConfig:
    """``argparse.Namespace`` -> :class:`CarrierConfig` (no silent defaults)."""
    return CarrierConfig(
        readout=args.readout, readout_qtok=args.readout_qtok,
        cond_dim=args.cond_dim, n_gauss=args.n_gauss, gen_width=args.gen_width,
        loss_level=args.loss_level, lambda_img=args.lambda_img, clamp=args.clamp,
        batch_split=args.batch_split, hc_eps=args.hc_eps, hc_mask=args.hc_mask,
        context=args.context, cond_zero=args.cond_zero,
        cond_trainmean=args.cond_trainmean, lut_resample=args.lut_resample,
        mining=args.mining, base_lr=args.base_lr,
        proj_lr_scale=args.proj_lr_scale, epochs=args.epochs,
        max_grad_norm=args.max_grad_norm, seed=args.seed,
        precision=args.precision, max_steps=args.max_steps,
        lib_size=args.lib_size, bake_grid=args.bake_grid,
        interp_pairs=args.interp_pairs, strength_column=args.strength_column,
    )


# --------------------------------------------------------------------------- #
# 2. the model:  z -> pi -> generator -> GLUT carrier
# --------------------------------------------------------------------------- #
class CarrierModel(nn.Module):
    """``pi`` + CGLUT generator + the shared GLUT carrier.

    ``pi`` stands exactly where CGLUT's ``E[l]`` lookup stood; the generator and
    the carrier are the shared modules, unmodified.  Initialisation is PyTorch's
    default everywhere (``EPR-024:556``, ruling 11.1-1): **step 0 is not the
    identity** for this arm, and :meth:`step0_maxabs_f_minus_id` is the witness
    column that says so on the board.

    The two constant query grids live in non-persistent buffers so no forward
    ever calls ``torch.tensor(...)`` and every evaluation grid is already on the
    model's device.
    """

    def __init__(self, cfg: CarrierConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.pi = SegColorProjection(in_dim=SEG_COLOR_HIDDEN_DIM, cond_dim=cfg.cond_dim)
        self.generator = CGLUTGenerator(
            cond_dim=cfg.cond_dim, hidden=cfg.gen_width, n_gauss=cfg.n_gauss,
            mode="full",            # D4 Full Generation: all 22N+12 generated
            m_residual=False,       # D6: no identity anchoring
            zero_init_last=False,   # D6: PyTorch default init
        )
        self.carrier = GlutCarrier(clamp=cfg.clamp, residual=True, eps=EPS)
        self.register_buffer("query_grid17", uniform_grid(GRID_N_HEADLINE), persistent=False)
        self.register_buffer("query_grid9", uniform_grid(GRID_N_FLOOR), persistent=False)
        self.register_buffer("train_mean_z", torch.zeros(SEG_COLOR_HIDDEN_DIM), persistent=False)

    # -- bookkeeping --
    @property
    def device(self) -> torch.device:
        return self.pi.proj.weight.device

    @property
    def param_dtype(self) -> torch.dtype:
        return self.pi.proj.weight.dtype

    def set_train_mean_z(self, z_mean: Tensor) -> None:
        """Install ``z̄_train`` for the ``--cond-trainmean`` ablation column."""
        self.train_mean_z.copy_(z_mean.to(device=self.train_mean_z.device,
                                          dtype=self.train_mean_z.dtype))

    @property
    def config(self) -> dict[str, Any]:
        n_pi = sum(p.numel() for p in self.pi.parameters())
        n_gen = sum(p.numel() for p in self.generator.parameters())
        return {
            "pi": {"in_dim": SEG_COLOR_HIDDEN_DIM, "cond_dim": self.cfg.cond_dim,
                   "n_params": n_pi,
                   "form": "LayerNorm(2560) + Linear(2560 -> d)"},
            "generator": self.generator.config,
            "carrier": self.carrier.config,
            "n_params_pi": n_pi,
            "n_params_generator": n_gen,
            "n_params_total": n_pi + n_gen,
            "n_params_generator_closed_form": generator_param_count(
                cond_dim=self.cfg.cond_dim, hidden=self.cfg.gen_width,
                n_gauss=self.cfg.n_gauss, mode="full"),
            "theta_dim": n_params_glut(self.cfg.n_gauss),
            "init": "pytorch default everywhere (EPR-024:556 / ruling 11.1-1); "
                    "step 0 is NOT the identity",
        }

    # -- forward --
    def condition(self, z: Tensor) -> Tensor:
        """``z (B, 2560) -> u (B, d)``, with the two condition ablations applied.

        ``--cond-zero`` feeds ``z = 0`` and ``--cond-trainmean`` feeds
        ``z = z̄_train``: the in-model counterparts of B1 (§4.D).
        """
        ref = self.pi.proj.weight
        zc = z.to(device=ref.device, dtype=ref.dtype)
        if self.cfg.cond_zero:
            zc = torch.zeros_like(zc)
        elif self.cfg.cond_trainmean:
            zc = self.train_mean_z.to(device=ref.device, dtype=ref.dtype).expand_as(zc)
        return self.pi(zc)

    def params_from_condition(self, u: Tensor) -> GlutParams:
        return self.generator(u)

    def params_for(self, z: Tensor) -> GlutParams:
        return self.generator(self.condition(z))

    def forward(self, z: Tensor, x: Tensor, *, return_aux: bool = False,
                clamp: str | None = None) -> Tensor | tuple[Tensor, GlutAux]:
        """``(B,2560) x (B,Q,3) -> (B,Q,3)``.  ``clamp=None`` uses ``--clamp``."""
        params = self.params_for(z)
        return self.carrier(x, params, return_aux=return_aux, clamp=clamp)

    def transform_grid(self, z: Tensor, x: Tensor, *, clamp: str | None = None,
                       return_aux: bool = False):
        """Evaluate on a shared ``(P,3)`` query set for a batch of conditions."""
        params = self.params_for(z)
        return self.carrier(x, params, return_aux=return_aux, clamp=clamp)

    def transform_image(self, z: Tensor, img: Tensor, *, clamp: str | None = None,
                        point_chunk: int | None = None) -> Tensor:
        """``f̂(I)`` for one ``(3,H,W)`` sRGB image; returns ``(3,H,W)``.

        The queries are flattened to ``(P, 3)`` on purpose: the carrier reads a
        two-axis ``x`` as "one query set shared by the batch", while a three-axis
        ``(H, W, 3)`` would be read as ``H`` batch elements.
        """
        if img.dim() != 3 or img.shape[0] != 3:
            raise ValueError(f"img must be (3,H,W), got {tuple(img.shape)}")
        params = self.params_for(z.reshape(1, -1))
        h, w = int(img.shape[1]), int(img.shape[2])
        x = img.permute(1, 2, 0).reshape(-1, 3).to(device=params.device)
        out = self.carrier(x, params, clamp=clamp, point_chunk=point_chunk)
        return out[0].reshape(h, w, 3).permute(2, 0, 1).to(dtype=img.dtype)

    @torch.no_grad()
    def step0_maxabs_f_minus_id(self, z: Tensor) -> float:
        """``max |f_theta(x) - x|`` on the 17^3 grid -- the init-档 witness."""
        x = self.query_grid17
        y = self.transform_grid(z.reshape(-1, SEG_COLOR_HIDDEN_DIM), x)
        return float((y - x.unsqueeze(0)).abs().max())


# --------------------------------------------------------------------------- #
# 3. the loss ladder (GLUT Eq.6-8 + the NOVEL fourth level)
# --------------------------------------------------------------------------- #
@dataclass
class CarrierLoss:
    """Total plus every named part, and the pre-registered step columns."""

    total: Tensor
    l_rec: Tensor
    l_hc: Tensor
    l_sparse: Tensor
    l_img: Tensor | None
    n_hc_masked: int
    n_colors: int

    def row(self, *, n_luts_in_batch: int, mining_ratio_value: float,
            extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """The ``steps.jsonl`` row.  Column names are the §4.H pre-registration."""
        row: dict[str, Any] = {
            # detached on purpose: a step row must not keep the graph alive
            "L_rec": float(self.l_rec.detach()),
            "L_hc": float(self.l_hc.detach()),
            "L_sparse": float(self.l_sparse.detach()),
            "n_colors": int(self.n_colors),
            "n_luts_in_batch": int(n_luts_in_batch),
            "mining_ratio": float(mining_ratio_value),
            "n_hc_masked": int(self.n_hc_masked),
            "loss": float(self.total.detach()),
        }
        if self.l_img is not None:
            row["L_img"] = float(self.l_img.detach())
        if extra:
            row.update(dict(extra))
        return row


def reconstruction_loss(f: Tensor, y: Tensor) -> Tensor:
    """GLUT Eq.6 ``L_rec = || ŷ - y ||_1`` (mean over colours and channels)."""
    if f.shape != y.shape:
        raise ValueError(f"prediction {tuple(f.shape)} != target {tuple(y.shape)}")
    return (f - y.to(device=f.device, dtype=f.dtype)).abs().mean()


def hue_chroma_loss(f: Tensor, y: Tensor, *, eps_c: float = HC_EPS,
                    mask: bool = True) -> tuple[Tensor, Tensor]:
    """GLUT Eq.7 ``L_hc = C (1 - <ĥ, h>)``, returns ``(loss, n_masked)``.

    ``C`` and ``h`` are the **target**'s chroma and unit hue; ``ĥ`` is the
    prediction's.  The frozen block's ``C -> 0`` handling: both hues divide by
    ``max(C, eps_c)`` and the whole term is multiplied by the hard mask
    ``1[C >= eps_c]`` computed on the target chroma -- the target's ``C`` is the
    weight in Eq.7, so it is the one that makes the term ``0/0``.  A prediction
    with ``Ĉ -> 0`` is well defined (``<ĥ,h> -> 0``, the term tends to ``C``) and
    is deliberately not masked.  ``--no-hc-mask`` is the ablation row: hues still
    divide by ``max(C, eps_c)``, the mask is simply not applied.

    ``n_masked`` counts the query points the mask removed; it is the frozen
    block's ``n_hc_masked`` step column.
    """
    lab_hat = srgb_to_lab(f)
    lab_y = srgb_to_lab(y.to(device=f.device, dtype=f.dtype))
    c_t, h_t, valid = chroma_hue(lab_y, eps_c)
    _, h_p, _ = chroma_hue(lab_hat, eps_c)
    term = c_t * (1.0 - (h_p * h_t).sum(dim=-1))
    n_masked = (~valid).sum()
    if mask:
        term = term * valid.to(term.dtype)
    return term.mean(), n_masked


def opacity_entropy(opacity: Tensor, *, eps: float = EPS) -> Tensor:
    """GLUT Eq.8 ``R_sparse`` -- the binary entropy of the opacities, negated."""
    o = opacity
    return -(o * torch.log(o + eps) + (1.0 - o) * torch.log(1.0 - o + eps)).mean()


def image_l1(i_hat: Tensor, i_star: Tensor) -> Tensor:
    """The NOVEL fourth level ``L_img = || Î - I* ||_1`` (differentiable compose)."""
    return (i_hat - i_star.to(device=i_hat.device, dtype=i_hat.dtype)).abs().mean()


def compute_loss(f: Tensor, y: Tensor, aux: GlutAux, cfg: CarrierConfig, *,
                 i_hat: Tensor | None = None, i_star: Tensor | None = None
                 ) -> CarrierLoss:
    """The ladder of §3.2, level by level.  Weights are the paper's."""
    l_rec = reconstruction_loss(f, y)
    if cfg.loss_level >= 2:
        l_hc, n_masked = hue_chroma_loss(f, y, eps_c=cfg.hc_eps, mask=cfg.hc_mask)
    else:
        l_hc = f.new_zeros(())
        n_masked = f.new_zeros((), dtype=torch.long)
    l_sparse = (opacity_entropy(aux.opacity, eps=EPS) if cfg.loss_level >= 3
                else f.new_zeros(()))
    total = l_rec + cfg.lambda_hc * l_hc + cfg.lambda_sparse * l_sparse
    l_img: Tensor | None = None
    if cfg.loss_level >= 4:
        if i_hat is None or i_star is None:
            raise ValueError(
                "--loss-level 4 asks for L_img but the batch carried no images; "
                "the composed Î and the dataset's I* are both required "
                "(EPR-024 §3.2 ④)")
        l_img = image_l1(i_hat, i_star)
        total = total + cfg.lambda_img_effective * l_img
    return CarrierLoss(total=total, l_rec=l_rec, l_hc=l_hc, l_sparse=l_sparse,
                       l_img=l_img, n_hc_masked=int(n_masked),
                       n_colors=int(f.shape[0] * f.shape[1]))


def loss_preregistration(cfg: CarrierConfig) -> dict[str, Any]:
    """``config/loss_preregistration.json`` for this arm -- what was promised."""
    terms = [
        {"name": "L_rec", "weight": 1.0, "form": "|| f_theta(x) - L_l(x) ||_1",
         "source": "GLUT Eq.6", "active": True},
        {"name": "L_hc", "weight": cfg.lambda_hc,
         "form": "C * (1 - <h_hat, h>) in CIELab, C = sqrt(a^2+b^2)",
         "source": "GLUT Eq.7 + App A.1 (lambda_hc = 10)",
         "c_to_zero": {"h": "(a,b)/max(C, eps_c)", "eps_c": cfg.hc_eps,
                       "hard_mask": cfg.hc_mask, "masked_on": "target chroma",
                       "logged_as": "n_hc_masked", "status": "NOVEL numeric"},
         "active": cfg.loss_level >= 2},
        {"name": "R_sparse", "weight": cfg.lambda_sparse,
         "form": "-(1/N) sum_i [o log(o+eps) + (1-o) log(1-o+eps)], eps = 1e-6",
         "source": "GLUT Eq.8 + App A.1 (lambda_sparse = 0.001)",
         "active": cfg.loss_level >= 3},
        {"name": "L_img", "weight": cfg.lambda_img_effective,
         "form": "|| Î - I* ||_1, Î = (1-a) I + a f_theta(I), GT alpha",
         "source": "NOVEL addition (GLUT uses natural images for evaluation only)",
         "active": cfg.loss_level >= 4},
    ]
    return {
        "arm": ARM, "arm_name": ARM_NAME, "loss_level": cfg.loss_level,
        "total": "L_rec + 10*L_hc + 0.001*R_sparse (+ lambda_img*L_img at level 4)",
        "terms": terms,
        "supervision_space": "function values y = L_l(x); the LUT is evaluated "
                             "with the generator's own operator (rendering.py:390-405)",
        "step_columns": list(cfg.step_columns),
        "not_used": ["AUC (banned)", "IoU (banned)", "val loss for selection (banned)",
                     "per-image min-max / softmax normalisation (banned)"],
    }


# --------------------------------------------------------------------------- #
# 4. optimiser + schedule (App A.1, verbatim)
# --------------------------------------------------------------------------- #
def build_optimizer(model: CarrierModel, cfg: CarrierConfig) -> torch.optim.Adam:
    """Adam, two groups: the generator at ``1e-3``, ``pi`` at ``0.1x``.

    App A.1 gives the 0.1x rate to "the style embeddings and shared geometry
    parameters"; this arm has no shared geometry (Full Generation), so the 0.1x
    group is exactly ``pi`` -- the parameter that stands where the style
    embedding stood.  That mapping is NOVEL and is recorded as such.
    ``betas`` are PyTorch's defaults: the paper does not give them.
    """
    groups = [
        {"params": list(model.generator.parameters()), "lr": float(cfg.base_lr),
         "name": "generator"},
        {"params": list(model.pi.parameters()),
         "lr": float(cfg.base_lr) * float(cfg.proj_lr_scale), "name": "pi"},
    ]
    return torch.optim.Adam(groups, lr=float(cfg.base_lr), betas=ADAM_BETAS)


def build_scheduler(opt: torch.optim.Optimizer, cfg: CarrierConfig):
    """Cosine annealing over the whole run (App A.1: "over the entire training
    duration"), one schedule per group -- each keeps its own base lr."""
    return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.total_steps)


# --------------------------------------------------------------------------- #
# 5. targets, mining and one training step
# --------------------------------------------------------------------------- #
def lut_targets(bank: LutBank, lut_ids: Sequence[str], x: Tensor) -> Tensor:
    """``y[b] = L_{lut_ids[b]}(x[b])`` for ``x`` of shape ``(B, Q, 3)``."""
    if x.dim() != 3 or x.shape[-1] != 3:
        raise ValueError(f"x must be (B, Q, 3), got {tuple(x.shape)}")
    if len(lut_ids) != x.shape[0]:
        raise ValueError(f"{len(lut_ids)} lut ids for {x.shape[0]} samples")
    return torch.stack([bank.apply(x[b], lut_ids[b]) for b in range(x.shape[0])], dim=0)


def mine_colors(model: CarrierModel, cfg: CarrierConfig, *, z: Tensor,
                lut_ids: Sequence[str], bank: LutBank, sampler: QuerySampler,
                epoch: float, device: Any = "cpu"
                ) -> tuple[Tensor, Tensor, float, int]:
    """Ruling 11.1-4: within-batch top-``r`` colour resampling, no cross-step state.

    A probe batch of ``B x Q`` uniform colours is scored under ``no_grad``; the
    ``r`` fraction with the largest per-colour L1 is kept and the rest of the
    step is filled with fresh uniform colours, so the step still carries exactly
    ``B x Q = 8192`` colours and ``r * 8192`` of them are hard ones.

    The selection is **per sample** (``round(r*Q)`` hard colours out of each
    sample's own probe).  A single global top-``r`` over the flattened 8192 would
    give a different number of colours to different samples and break the frozen
    ``(B, Q) = (32, 256)`` rectangle; the count of hard colours is identical
    either way, and the ``topk`` stays on the device in both.

    Returns ``(x, y, ratio, n_hard)``.
    """
    b, q = int(z.shape[0]), cfg.queries_per_sample
    x_probe = sampler.sample(b, q, device=device)
    if not cfg.mining:
        return x_probe, lut_targets(bank, lut_ids, x_probe), 0.0, 0
    ratio = mining_ratio(epoch, start_epoch=MINING_START_EPOCH,
                         end_epoch=MINING_END_EPOCH,
                         r_start=MINING_R_START, r_end=MINING_R_END)
    k = int(round(ratio * q))
    if k <= 0:
        return x_probe, lut_targets(bank, lut_ids, x_probe), ratio, 0
    y_probe = lut_targets(bank, lut_ids, x_probe)
    with torch.no_grad():
        f_probe = model(z, x_probe)
        err = (f_probe - y_probe).abs().mean(dim=-1)          # (B, Q), on device
    idx = torch.stack([select_hard(err[i], ratio)[:k] for i in range(b)], dim=0)
    gather = idx.unsqueeze(-1).expand(-1, -1, 3)
    x_hard = torch.gather(x_probe, 1, gather)
    y_hard = torch.gather(y_probe, 1, gather)
    x_fresh = sampler.sample(b, q - k, device=device)
    y_fresh = lut_targets(bank, lut_ids, x_fresh)
    return (torch.cat([x_hard, x_fresh], dim=1),
            torch.cat([y_hard, y_fresh], dim=1), ratio, int(k * b))


def params_slice(params: GlutParams, i: int) -> GlutParams:
    """One sample's ``GlutParams`` (batch axis kept at 1), gradient intact.

    Needed by the fourth loss level: the images of a batch have different aspect
    ratios, so ``Î`` is composed one sample at a time rather than on a stacked
    ``(B,3,H,W)`` that does not exist.
    """
    return GlutParams(*(getattr(params, k)[i:i + 1] for k in _PARAM_FIELDS))


_PARAM_FIELDS: tuple[str, ...] = ("mu", "chol_diag", "chol_off", "opacity_logit",
                                  "m_local", "b_local", "g_matrix", "g_bias")


def image_loss_terms(model: CarrierModel, params: GlutParams, bank: LutBank,
                     images: Sequence[Tensor], alphas: Sequence[Tensor | float],
                     lut_ids: Sequence[str]) -> tuple[Tensor, Tensor]:
    """``(Î, I*)`` flattened over a batch of differently-shaped images.

    Both sides are built with the frozen formation (``compose_hat`` / the data
    law), so ``L_img`` is measured on exactly the quantity the headline is.
    """
    device = model.device
    hats, stars = [], []
    for b, img in enumerate(images):
        im = img.to(device=device)
        a = alphas[b] if isinstance(alphas[b], float) else alphas[b].to(device=device)
        h, w = int(im.shape[1]), int(im.shape[2])
        x = im.permute(1, 2, 0).reshape(-1, 3)
        f_img = model.carrier(x, params_slice(params, b))[0].reshape(h, w, 3).permute(2, 0, 1)
        hats.append(mix_alpha(im, f_img, a).reshape(-1))
        stars.append(bank.f_star_image(im, a, lut_ids[b]).reshape(-1))
    return torch.cat(hats), torch.cat(stars)


def _autocast(cfg: CarrierConfig, device: torch.device):
    """bf16 autocast on CUDA only.  The carrier disables autocast for its own
    maths (Eq.1's determinant and ``exp(logpdf)`` stay fp32), so this only ever
    covers the generator MLP."""
    want_bf16 = cfg.precision == "bf16" or (cfg.precision == "auto" and device.type == "cuda")
    if want_bf16 and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def train_step(model: CarrierModel, cfg: CarrierConfig, opt: torch.optim.Optimizer,
               scheduler, *, step: int, z: Tensor, lut_ids: Sequence[str],
               bank: LutBank, sampler: QuerySampler,
               images: Sequence[Tensor] | None = None,
               alphas: Sequence[Tensor | float] | None = None
               ) -> dict[str, Any]:
    """One optimisation step; returns the ``steps.jsonl`` row.

    On ``step == 0`` the row is also handed to
    :func:`~q3vl.whatb.guards.record_step_witness` -- the third tier of the
    publication gate's row lookup, so a quick eval that lands before the trainer
    flushes ``steps.jsonl`` still finds the loss columns instead of reporting a
    plumbing failure as "the loss never ran".
    """
    device = model.device
    model.train()
    epoch = cfg.epoch_of(step)
    z = z.to(device=device)
    x, y, ratio, n_hard = mine_colors(model, cfg, z=z, lut_ids=lut_ids, bank=bank,
                                      sampler=sampler, epoch=epoch, device=device)
    with _autocast(cfg, device):
        params = model.params_for(z)
    f, aux = model.carrier(x, params, return_aux=True)
    i_hat = i_star = None
    if cfg.loss_level >= 4:
        if images is None or alphas is None:
            raise ValueError("--loss-level 4 needs images and GT alpha in the batch")
        i_hat, i_star = image_loss_terms(model, params, bank, images, alphas, lut_ids)
    loss = compute_loss(f, y, aux, cfg, i_hat=i_hat, i_star=i_star)

    opt.zero_grad(set_to_none=True)
    loss.total.backward()
    gnorm = float(torch.nn.utils.clip_grad_norm_(
        [p for g in opt.param_groups for p in g["params"]], cfg.max_grad_norm)
        if cfg.max_grad_norm > 0 else 0.0)
    opt.step()
    if scheduler is not None:
        scheduler.step()

    row = loss.row(n_luts_in_batch=len({str(l) for l in lut_ids}),
                   mining_ratio_value=ratio,
                   extra={"step": int(step), "epoch": epoch,
                          "n_hard_colors": int(n_hard), "gnorm": gnorm,
                          "lr_generator": float(opt.param_groups[0]["lr"]),
                          "lr_pi": float(opt.param_groups[1]["lr"]),
                          "degenerate_precision": int(aux.degenerate_precision.sum())})
    if step == 0:
        record_step_witness(row)
    return row


# --------------------------------------------------------------------------- #
# 6. the z cache -- ONE implementation, in q3vl/whatb/zcache.py
# --------------------------------------------------------------------------- #
# EPR-024 wrote the first reader/writer here and the other five arms each grew
# their own; the frozen block says 共同依赖只写一份, so the implementation moved
# to :mod:`q3vl.whatb.zcache` and this module re-exports it.  Nothing about the
# on-disk contract or the three start-up assertions changed -- the class is the
# same object, not a copy.


# --------------------------------------------------------------------------- #
# 7. evaluation data (images at short side 512 + GT alpha)
# --------------------------------------------------------------------------- #
@dataclass
class EvalSample:
    """Everything one evaluation row needs, already on the target device."""

    sample_id: str
    task_type: str
    winner_confidence: str
    lut_id: str
    source_image_id: str
    minor: str | None
    image: Tensor                       # (3, H, W) sRGB in [0,1], short side 512
    alpha: Tensor | float               # (1, H, W) GT alpha, or 1.0 for style
    z: Tensor                           # (2560,)
    z_controls: dict[str, Tensor] = field(default_factory=dict)

    @property
    def alpha_mean(self) -> float:
        return 1.0 if isinstance(self.alpha, float) else float(self.alpha.mean())


# --------------------------------------------------------------------------- #
# 8. the B1 baked volume
# --------------------------------------------------------------------------- #
def bake_transform_volume(values: Tensor, n: int) -> Tensor:
    """``(n^3, 3)`` values on :func:`uniform_grid` ``-> (1, 3, n, n, n)`` volume.

    ``uniform_grid`` enumerates ``(r, g, b)`` with ``r`` slowest, so the reshaped
    tensor is indexed ``[r, g, b]`` while the bank's storage -- and therefore
    :func:`~q3vl.whatb.lutdata.apply_lut_volume` -- is ``[b, g, r]``.  The
    ``permute(2, 1, 0, 3)`` is that transposition and nothing else; the unit test
    bakes a real LUT at its own grid size and checks the round trip is exact.
    """
    if values.dim() != 2 or values.shape[-1] != 3 or values.shape[0] != n ** 3:
        raise ValueError(f"expected ({n ** 3}, 3) values, got {tuple(values.shape)}")
    grid = values.reshape(n, n, n, 3).permute(2, 1, 0, 3).contiguous()   # [b, g, r]
    return grid.permute(3, 0, 1, 2).unsqueeze(0).contiguous()


def library_mean_volume(bank: LutBank, lut_ids: Sequence[str], n: int, *,
                        device: Any = "cpu", chunk: int = 64) -> tuple[Tensor, Tensor]:
    """B1's ``L̄(x) = mean_l L_l(x)``, baked once onto an ``n^3`` grid.

    Returns ``(volume, mean_values_on_grid)``.  Baking is what makes B1 affordable
    at headline resolution (the exact point-wise mean would be
    ``|Lib| x 327k`` LUT evaluations per image); the caller reports the residual
    against the exact mean on the 17^3 query grid, which is the number that says
    how much the bake cost.
    """
    x = uniform_grid(n, device=device)
    acc = torch.zeros_like(x)
    for i in range(0, len(lut_ids), chunk):
        for lid in lut_ids[i:i + chunk]:
            acc += bank.apply(x, lid)
    mean = acc / float(max(1, len(lut_ids)))
    return bake_transform_volume(mean, n), mean


# --------------------------------------------------------------------------- #
# 9. quick eval -- the degeneracy gate lives here
# --------------------------------------------------------------------------- #
@torch.no_grad()
def quick_eval(model: CarrierModel, cfg: CarrierConfig, *, z: Tensor,
               lut_ids: Sequence[str], bank: LutBank, step: int,
               thresholds: DegeneracyThresholds | None = None,
               first: bool = True, exit_process: bool = True) -> dict[str, Any]:
    """Function-space quick eval + the three degeneracy assertions.

    The gate runs at the **first** quick eval, not at step 0: a zero-initialised
    head is legitimately the identity at step 0, but a head that is still flat,
    still the identity, or still sample-invariant once it has trained is the
    where-side PRND/CONDINST failure (1200 steps and 2.6 GPU-hours before anyone
    noticed the field was constant).  Any of the three exits with
    ``SystemExit(2)`` and the measured numbers printed.
    """
    model.eval()
    x = model.query_grid17
    zz = z.to(device=model.device)
    f = model.transform_grid(zz, x)
    if first:
        assert_transform_not_degenerate(
            f, x, thresholds=thresholds or DegeneracyThresholds(),
            where=f"quick_eval@step{step}", exit_process=exit_process,
            extra={"arm": ARM, "n_gauss": cfg.n_gauss, "cond_dim": cfg.cond_dim,
                   "clamp": cfg.clamp})
    y = torch.stack([bank.apply(x, lid) for lid in lut_ids], dim=0)
    grid_de00 = torch.stack([C.function_distance(f[i], y[i]) for i in range(f.shape[0])])
    l1 = (f - y).abs().mean()
    return {"step": int(step), "n": int(f.shape[0]),
            "grid_de00_mean": float(grid_de00.mean()),
            "grid_de00_p50": float(grid_de00.median()),
            "l1_grid": float(l1),
            "maxabs_f_minus_id": float((f - x.unsqueeze(0)).abs().max()),
            "std_over_queries": float(f.std(dim=1, unbiased=False).mean()),
            "std_over_samples": float(f.std(dim=0, unbiased=False).mean())
            if f.shape[0] > 1 else None}


# --------------------------------------------------------------------------- #
# 10. the full evaluation -> per-sample rows
# --------------------------------------------------------------------------- #
def _headline_error(img: Tensor, alpha: Tensor | float, f_img: Tensor,
                    i_star: Tensor) -> float:
    """``mean dE00(Î, I*)`` for one prediction of ``f̂(I)`` (frozen formation)."""
    return float(C.image_delta_e00(C.compose_hat(img, alpha, f_img), i_star))


@torch.no_grad()
def evaluate_samples(model: CarrierModel, cfg: CarrierConfig,
                     samples: Sequence[EvalSample], *, bank: LutBank,
                     lib: C.LibraryValues, lib_mean_volume: Tensor,
                     bucket_pools: Mapping[str, Sequence[str]] | None = None,
                     seed: int = DEFAULT_SEED,
                     point_chunk: int | None = None) -> list[dict[str, Any]]:
    """One row per sample, carrying every column ``build_board`` knows.

    Columns produced here (all on the samples' device, no ``.cpu()``):

    ``E_arm``                     headline dE00(Î, I*)
    ``E_B0_identity``             f̂ = id
    ``E_B1_libmean``              f̂ = the baked library mean
    ``E_B2_librandom_repeats``    R uniform draws from Lib_tr
    ``E_B3_bucket_retrieval_repeats``  R draws from the sample's own minor bucket
    ``E_B4_oracle``               arg min over Lib_tr on the 9^3 / dE76 protocol
    ``E_B6_libfill``              nearest OTHER library LUT (V_what column)
    ``E_N{1,2,3}_*`` / ``M_N{1,2,3}_*``   the three negative controls
    ``grid_error`` / ``img_error`` / ``unseen_color_error``
    ``loc_in`` / ``loc_band`` / ``loc_out``  (side columns, GT alpha)
    """
    model.eval()
    device = model.device
    grid17 = model.query_grid17
    grid9 = model.query_grid9
    ids = [s.sample_id for s in samples]

    draws_b2 = C.library_random_draw(lib.lut_ids, len(samples),
                                     repeats=cfg.n_repeats, seed=seed)
    draws_b3: list[list[str | None]] | None = None
    if bucket_pools is not None:
        draws_b3 = C.bucket_draw([s.minor or "" for s in samples], bucket_pools,
                                 repeats=cfg.n_repeats, seed=seed + 1)

    # B4 / B6 selection: 9^3 grid, dE76 -- the protocol the floors were measured in
    targets9 = {s.sample_id: bank.apply(grid9, s.lut_id) for s in samples}
    oracle = C.oracle_lut_ids(lib, targets9, metric="de76", exclude_self=False)
    # B6 is keyed by LUT, not by sample: `exclude_self` drops the target's own row
    # from the library, and it can only find that row if the key IS the lut_id.
    # (On T_lut_unseen no key is in Lib_tr, so B6 there is B4 by construction --
    # which is why §4.1 leaves the B6 cell blank for the two test splits.)
    fill = C.oracle_lut_ids(lib, {s.lut_id: targets9[s.sample_id] for s in samples},
                            metric="de76", exclude_self=True)

    rows: list[dict[str, Any]] = []
    for i, s in enumerate(samples):
        img = s.image.to(device=device)
        alpha = s.alpha if isinstance(s.alpha, float) else s.alpha.to(device=device)
        i_star = bank.f_star_image(img, alpha, s.lut_id)
        f_arm = model.transform_image(s.z, img, point_chunk=point_chunk)
        i_hat = C.compose_hat(img, alpha, f_arm)

        row: dict[str, Any] = {
            "sample_id": s.sample_id, "winner_confidence": s.winner_confidence,
            "task_type": s.task_type, "lut_id": s.lut_id,
            "source_image_id": s.source_image_id, "minor": s.minor,
            "alpha_mean": s.alpha_mean, "lut_size": bank.size(s.lut_id),
            "E_arm": float(C.image_delta_e00(i_hat, i_star)),
            "E_B0_identity": float(C.image_delta_e00(img, i_star)),
            "E_B1_libmean": _headline_error(
                img, alpha,
                apply_lut_volume(lib_mean_volume.to(device=device),
                                 img.permute(1, 2, 0)).permute(2, 0, 1), i_star),
            "E_B4_oracle": _headline_error(
                img, alpha, bank.apply_image(img, oracle[s.sample_id][0]), i_star),
            "E_B6_libfill": _headline_error(
                img, alpha, bank.apply_image(img, fill[s.lut_id][0]), i_star),
            "B4_lut_id": oracle[s.sample_id][0],
            "B4_select_de76_9grid": oracle[s.sample_id][1],
        }
        row["E_B2_librandom_repeats"] = [
            _headline_error(img, alpha, bank.apply_image(img, draws_b2[r][i]), i_star)
            for r in range(cfg.n_repeats)]
        if draws_b3 is not None:
            vals = [_headline_error(img, alpha, bank.apply_image(img, lid), i_star)
                    for r in range(cfg.n_repeats)
                    if (lid := draws_b3[r][i]) is not None]
            row["E_B3_bucket_retrieval_repeats"] = vals or None
            row["B3_bucket_missing"] = int(cfg.n_repeats - len(vals))

        # -- function-value columns --
        f_grid = model.transform_grid(s.z.reshape(1, -1), grid17)[0]
        y_grid = bank.apply(grid17, s.lut_id)
        row["grid_error"] = float(C.function_distance(f_grid, y_grid))
        colors, weights = image_histogram_colors(img, bits=HIST_BITS, top_k=HIST_TOPK)
        f_hist = model.transform_grid(s.z.reshape(1, -1), colors)[0]
        row["img_error"] = float(C.function_distance(f_hist, bank.apply(colors, s.lut_id),
                                                     weights))
        unseen = _heldout_colors(cfg, i, device=device)
        f_unseen = model.transform_grid(s.z.reshape(1, -1), unseen)[0]
        row["unseen_color_error"] = float(
            C.function_distance(f_unseen, bank.apply(unseen, s.lut_id)))

        # -- locality (side columns; GT alpha at short side 512) --
        if not isinstance(alpha, float):
            row.update(C.locality_errors(i_hat, i_star, img, alpha))

        # -- the three negative controls --
        for tag, (e_key, m_key) in CONTROL_ROW_KEYS.items():
            z_ctrl = s.z_controls.get(tag)
            if z_ctrl is None:
                continue
            f_ctrl_img = model.transform_image(z_ctrl, img, point_chunk=point_chunk)
            row[e_key] = _headline_error(img, alpha, f_ctrl_img, i_star)
            f_ctrl_grid = model.transform_grid(z_ctrl.reshape(1, -1), grid17)[0]
            row[m_key] = float(C.function_distance(f_grid, f_ctrl_grid))

        if cfg.strength_column:
            row.update(_strength_columns(model, s, bank, grid17))
        rows.append(row)
    return rows


def _heldout_colors(cfg: CarrierConfig, i: int, *, device: Any = "cpu",
                    n: int = 4913) -> Tensor:
    """A deterministic draw from the complement of the 128^3 training colours."""
    sampler = QuerySampler(seed=cfg.seed + 9001 + i, q=n)
    return sampler.sample(1, n, device=device, heldout=True)[0]


@torch.no_grad()
def _strength_columns(model: CarrierModel, s: EvalSample, bank: LutBank,
                      grid: Tensor) -> dict[str, Any]:
    """§4.G with the synthetic target ``y_u = (1-u) x + u L_l(x)``.

    ``theta`` does not depend on ``u`` in this arm, so the evaluated family is
    the identity gate applied to the arm's own transform,
    ``f̂_u = x + u (f̂(x) - x)`` (:func:`q3vl.whatb.gate.identity_gate`).  That is
    an evaluation-time construction, not a model input -- EPR-027 is the arm that
    makes ``u`` an input -- and it is reported as a diagnostic, never as a
    pre-registered column.
    """
    f = model.transform_grid(s.z.reshape(1, -1), grid)[0]
    y_lut = bank.apply(grid, s.lut_id)
    errs, mags = [], []
    for u in STRENGTH_U:
        f_u = identity_gate(grid, f, u)
        y_u = grid * (1.0 - u) + y_lut * u
        errs.append(float(C.function_distance(f_u, y_u)))
        mags.append(float((f_u - grid).abs().mean()))
    diffs = np.diff(np.asarray(mags))
    mono = float((diffs > 0).mean()) if diffs.size else float("nan")
    order = np.argsort(np.argsort(np.asarray(mags)))
    u_rank = np.argsort(np.argsort(np.asarray(STRENGTH_U)))
    denom = len(STRENGTH_U) * (len(STRENGTH_U) ** 2 - 1)
    spearman = 1.0 - 6.0 * float(((order - u_rank) ** 2).sum()) / denom
    return {"strength_de00_mean": float(np.mean(errs)),
            "strength_mono_rate": mono, "strength_mono_rate_random_floor": 0.5,
            "strength_spearman": spearman, "strength_spearman_random_floor": 0.0,
            "strength_de00_by_u": errs}


# --------------------------------------------------------------------------- #
# 11. the interpolation columns (P1's four required keys)
# --------------------------------------------------------------------------- #
def same_source_pairs(samples: Sequence[EvalSample], *, limit: int | None = None,
                      seed: int = DEFAULT_SEED) -> list[tuple[int, int]]:
    """IP-A's pairs: two samples of the same source image with different LUTs."""
    by_source: dict[str, list[int]] = {}
    for i, s in enumerate(samples):
        by_source.setdefault(s.source_image_id, []).append(i)
    rng = random.Random(seed)
    pairs: list[tuple[int, int]] = []
    for src in sorted(by_source):
        idx = by_source[src]
        cand = [(a, b) for k, a in enumerate(idx) for b in idx[k + 1:]
                if samples[a].lut_id != samples[b].lut_id]
        if cand:
            pairs.append(rng.choice(cand))
    rng.shuffle(pairs)
    return pairs if limit is None else pairs[:int(limit)]


@torch.no_grad()
def interpolation_columns(model: CarrierModel, cfg: CarrierConfig,
                          samples: Sequence[EvalSample], *, bank: LutBank,
                          lib: C.LibraryValues | None = None,
                          limit: int | None = None) -> dict[str, Any]:
    """IP-A + IP-B -> the four required P1 columns plus their trivial floors.

    IP-A (with GT): ``alpha in {0,.2,.4,.6,.8,1}`` (GLUT App B.3's grid), the
    condition mixed at ``pi``'s output (EPR-026's ``post_pi`` default -- ``pi``
    stands where ``e_l`` stood and CGLUT mixes ``e``), GT = the function-space
    linear blend of the two LUTs on the 17^3 grid.  The trivial ``output mixing``
    column ``(1-a) f̂_a + a f̂_b`` is emitted next to it: §4.F says an
    interpolation claim without that column is void.

    IP-B (no GT): the six path quantities at ``K = 20`` from
    :func:`q3vl.whatb.criteria.path_quantities`, with ``oob_rate`` taken from the
    **pre-clamp** value as §4.F defines it and the degenerate-weight rate
    ``Pr[sum_j p_j o_j < tau]`` from the carrier's own aux.
    """
    grid = model.query_grid17
    pairs = same_source_pairs(samples, limit=limit if limit is not None else cfg.interp_pairs,
                              seed=cfg.seed)
    if not pairs:
        raise ValueError(
            "IP-A needs at least one source with two samples carrying different "
            "lut_ids; V_what normal-only has 120 of them, so an empty list means "
            "the sample list, not the protocol, is wrong")

    def mixed_params(za: Tensor, zb: Tensor, a: float) -> GlutParams:
        if cfg.interp_mix_point == "post_pi":
            ua = model.condition(za.reshape(1, -1))
            ub = model.condition(zb.reshape(1, -1))
            return model.params_from_condition((1.0 - a) * ua + a * ub)
        z = (1.0 - a) * za.reshape(1, -1) + a * zb.reshape(1, -1)
        return model.params_for(z)

    ip_a, ip_mix, ip_end = [], [], []
    path_rows: list[dict[str, Any]] = []
    degen_rates: list[float] = []
    for ia, ib in pairs:
        sa, sb = samples[ia], samples[ib]
        la = bank.apply(grid, sa.lut_id)
        lb = bank.apply(grid, sb.lut_id)
        fa = model.transform_grid(sa.z.reshape(1, -1), grid)[0]
        fb = model.transform_grid(sb.z.reshape(1, -1), grid)[0]
        errs, mixes = [], []
        for a in INTERP_ALPHAS:
            f_cond = model.carrier(grid, mixed_params(sa.z, sb.z, a))[0]
            gt = (1.0 - a) * la + a * lb
            errs.append(float(C.function_distance(f_cond, gt)))
            mixes.append(float(C.function_distance((1.0 - a) * fa + a * fb, gt)))
        ip_a.append(float(np.mean(errs)))
        ip_mix.append(float(np.mean(mixes)))
        ip_end.append(0.5 * (errs[0] + errs[-1]))

        path, pre = [], []
        for k in range(INTERP_K + 1):
            p = mixed_params(sa.z, sb.z, k / INTERP_K)
            y, aux = model.carrier(grid, p, return_aux=True)
            path.append(y[0])
            pre.append(aux.pre_clamp[0])
            degen_rates.append(float(aux.degenerate_weight_mask(1e-3).to(y.dtype).mean()))
        pq = C.path_quantities(torch.stack(path))
        pre_t = torch.stack(pre)
        pq["oob_rate"] = float(((pre_t < 0.0) | (pre_t > 1.0)).any(dim=-1).to(pre_t.dtype).mean())
        pq["oob_rate_source"] = "pre-clamp (§4.F: A(alpha) is measured before the clamp)"
        path_rows.append(pq)

    def col(key: str) -> dict[str, Any]:
        return C.describe([r.get(key) for r in path_rows])

    return {
        "interp_grid": {**C.describe(ip_a), "n": len(ip_a),
                        "quantity": "mean over alpha of dE00(f^cond_alpha, "
                                    "(1-a)L_a + a L_b) on the 17^3 grid",
                        "trivial_output_mixing": C.describe(ip_mix),
                        "endpoints": C.describe(ip_end),
                        "alphas": list(INTERP_ALPHAS),
                        "mix_point": cfg.interp_mix_point,
                        "external_reference": {
                            "CGLUT-32L Full (PSNR, GLUT App B.3 Table 7)":
                                [48.67, 35.44, 31.16, 31.33, 34.64, 47.95],
                            "CGLUT-32L Shared Geo. (PSNR)":
                                [47.36, 38.46, 34.67, 34.47, 37.60, 46.18]}},
        "path_len": {**col("path_len"), "chord": col("chord"), "rho": col("rho"),
                     "sigma_bar": col("sigma_bar"), "jump_max": col("jump_max"),
                     "k_steps": INTERP_K,
                     "note": "jump_max = K * max_k delta_k, no percentile trimming"},
        "mono_rate": {**col("mono_rate"), "random_floor": 0.5,
                      "readout": "CIELab mean b* along alpha"},
        "oob_rate": {**col("oob_rate"),
                     "degenerate_weight_rate": float(np.mean(degen_rates)) if degen_rates else None,
                     "tau": 1e-3,
                     "source": "pre-clamp values (§4.F)"},
    }


# --------------------------------------------------------------------------- #
# 12. the board
# --------------------------------------------------------------------------- #
def build_arm_board(rows: Sequence[Mapping[str, Any]], *, split: str,
                    extra_columns: Mapping[str, Mapping[str, Any]],
                    published: bool = True, seed: int = DEFAULT_SEED,
                    facts: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """``criteria.build_board`` + the arm's own bookkeeping.

    ``extra_columns`` must carry the four P1 keys (``interp_grid`` / ``path_len``
    / ``mono_rate`` / ``oob_rate``): they are pre-registered and
    ``assert_criteria_ran`` refuses a board without them.
    """
    board = C.build_board(rows, arm=ARM, split=split, extra_columns=extra_columns,
                          seed=seed)
    board["published"] = bool(published)
    board["arm_name"] = ARM_NAME
    if facts:
        board["facts"] = dict(facts)
    return board


def publish_board(board: Mapping[str, Any], cfg: CarrierConfig, *,
                  steps_path: Any = None, steps_row: Mapping[str, Any] | None = None,
                  eval_only: bool = False) -> dict[str, Any]:
    """The single publication gate every ``metrics.json`` write goes through."""
    return assert_publishable(board, ARM, steps_row=steps_row, steps_path=steps_path,
                              eval_only=eval_only, loss_level=cfg.loss_level,
                              axes=AXES)


# --------------------------------------------------------------------------- #
# 13. the run record
# --------------------------------------------------------------------------- #
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_setup_record(cfg: CarrierConfig, model: CarrierModel, *,
                     checkpoint: str = BASE_CHECKPOINT,
                     colorspan_check: Mapping[str, Any] | None = None,
                     zcaches: Mapping[str, Any] | None = None,
                     split_facts: Mapping[str, Any] | None = None,
                     bank_facts: Mapping[str, Any] | None = None,
                     thresholds: DegeneracyThresholds | None = None,
                     step0_witness: float | None = None,
                     extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Everything ``run_setup.json`` has to carry, including the source sha256s.

    Source hashes are frozen here because the campaign rule is *no source edits
    after the process starts*; the two files that define this arm are hashed by
    content, not by mtime.
    """
    here = Path(__file__).resolve()
    runner = here.parent.parent / "scripts" / "run_carrier_arm.py"
    rec: dict[str, Any] = {
        "arm": ARM, "arm_name": ARM_NAME, "axes": list(AXES),
        "checkpoint": checkpoint,
        "config": cfg.as_dict(),
        "model": model.config,
        "readout": WhatReadoutSpec(kind=cfg.readout, qtok=cfg.readout_qtok).to_dict(),
        "frozen_block": {
            "data": cfg.data,
            "train_normal_only_n": cfg.train_n,
            "batch": f"B={cfg.batch_samples} x Q={cfg.queries_per_sample} = "
                     f"{cfg.colors_per_step} colours/step",
            # the six numbers a horizon is a function of, written down explicitly
            # (the product is no longer pinned to 8192, so it is recorded)
            "batch_split": cfg.batch_split,
            "batch_samples": cfg.batch_samples,
            "queries_per_sample": cfg.queries_per_sample,
            "colors_per_step": cfg.colors_per_step,
            "colours_per_step": cfg.colors_per_step,
            "batch_split_step_matched_to_epr024": cfg.batch_split in FROZEN_BATCH_SPLITS,
            "steps_per_epoch": cfg.steps_per_epoch,
            "total_steps": cfg.total_steps,
            "clamp": cfg.clamp,
            "headline_formation": "Î = (1-a) ⊙ I + a ⊙ f̂(I)",
            "preregistered_keys": list(C.PREREGISTERED_KEYS),
            "colorspan": "q3vl.whatb.colorspan (own implementation + tokenizer assertion)",
        },
        "optimizer": {
            "name": "Adam", "betas": list(ADAM_BETAS),
            "base_lr": cfg.base_lr, "pi_lr": cfg.base_lr * cfg.proj_lr_scale,
            "scheduler": "CosineAnnealingLR", "t_max": cfg.total_steps,
            "max_grad_norm": cfg.max_grad_norm,
            "deviations_declared": [
                "Adam betas are PyTorch defaults (the paper does not give them)",
                "max_grad_norm = 1.0 is a repository convention, not in the paper",
                "the 0.1x group is pi (this arm has no shared geometry) -- NOVEL mapping",
            ],
        },
        "mining": {"enabled": cfg.mining, "granularity": "colour query point",
                   "scope": "per sample (round(r*Q) of each sample's own probe)",
                   "epochs": [MINING_START_EPOCH, MINING_END_EPOCH],
                   "ratio": [MINING_R_START, MINING_R_END]},
        "degeneracy_thresholds": (thresholds or DegeneracyThresholds()).as_dict(),
        "source_sha256": {
            "carrier.py": _sha256(here),
            "run_carrier_arm.py": _sha256(runner) if runner.is_file() else None,
            # shared layer this arm consumes (the z cache and the image / GT-alpha
            # loader used to live inside carrier.py and were covered by its hash)
            **{n: _sha256(here.parent.parent / n)
               for n in ("zcache.py", "evaldata.py", "colorimetry.py")
               if (here.parent.parent / n).is_file()},
        },
    }
    if colorspan_check is not None:
        rec["colorspan_check"] = dict(colorspan_check)
    if zcaches is not None:
        rec["z_caches"] = dict(zcaches)
    if split_facts is not None:
        rec["split_facts"] = dict(split_facts)
    if bank_facts is not None:
        rec["lut_bank"] = dict(bank_facts)
    if step0_witness is not None:
        rec["step0_maxabs_f_minus_id"] = float(step0_witness)
    if extra:
        rec.update(dict(extra))
    return rec


def open_bank(cfg: CarrierConfig, bank_dir: str | Path = BANK_DIR) -> LutBank:
    """The LUT bank at this run's resample setting (``none``, ruling 11.1-3)."""
    return LutBank(bank_dir, resample=cfg.lut_resample)
