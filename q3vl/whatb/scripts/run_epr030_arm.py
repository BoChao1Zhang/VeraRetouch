"""EPR-030 entry point: the shared query backbone + the plain-L1 loss.

    python -m q3vl.whatb.scripts.run_epr030_arm \\
        --run-name whatb_EPR030_main \\
        --data v2seg+l8 \\
        --zcache-root /home/bc/data/runs/whatb/zcache_v2seg \\
        --zcache-root-l8 /home/bc/data/runs/whatb/zcache_l8 \\
        --backbone qdec --qdec-dim 512 --qdec-layers 6 --qdec-heads 8 \\
        --qdec-mem-rows 1 --qdec-act gelu --head-init bias \\
        --clamp two --clamp-grad st --loss l0 \\
        --batch-split 32x256 --context generated

Three changes, one arm
----------------------
1. **backbone**: ``z -> pi -> 3-layer MLP -> 5 heads`` becomes ``N+1 queries ->
   L x (cross-attention -> FFN) -> one shared 22-d head + one 12-d head``
   (:mod:`q3vl.whatb.qdecoder`).  ``--backbone mlp`` runs EPR-024's generator
   *unchanged* as the control row, so the backbone is the only variable on that
   line.
2. **loss**: ``L_rec + 10 L_hc + 0.001 R_sparse`` becomes the single term
   ``L = mean |f(x) - L_l(x)|`` (:mod:`q3vl.whatb.losses_l0`).  The three
   optional terms exist and are off; each is a published ablation row.
3. **data**: ``--data v2seg+l8`` unions the L8 normal rows onto the frozen
   sft2seg train split.  Two z caches, one condition
   (:class:`q3vl.whatb.zcache.MultiZCache`); every member goes through its own
   ``assert_belongs_to``.  ``--data v2seg`` is the paired ablation row and is
   compared at a **fixed step count**, because the two settings have different
   epoch lengths.

Everything else is imported, not re-written: the data pipeline, the z-cache
assertions, the colour-span assertion, mining, quick eval + the degeneracy
guard, the criteria board, the publication gate and checkpoint selection all
come from ``run_carrier_arm.main(argv, arm=<this module>)``.  This file supplies
the arm-specific symbols that runner looks up.

The training population is **measured**, never written down:
:func:`q3vl.whatb.splits.train_normal_rows` counts each source and asserts it
against that source's own on-disk declaration (sft2seg against the frozen
block's ``TRAIN_NORMAL_N``, L8 against ``l8_train.manifest.report.json``), and
the runner re-asserts ``steps_per_epoch == ceil(n / B)`` against the population
it actually loaded.

Run-time assertions this arm adds (the campaign has paid for each one)
---------------------------------------------------------------------
* ``--loss l0`` asserts ``lambda_hc == lambda_sparse == lambda_mono == 0``
  *before* the first step (:func:`~q3vl.whatb.losses_l0.assert_l0_pure`);
* the first quick eval asserts the L0 function was actually called
  (:func:`~q3vl.whatb.losses_l0.assert_l0_ran`, a process counter) -- "defined
  but never wired" has happened three times;
* the decoder asserts, with a real forward, that step 0 is the identity map and
  that every parameter equals the ``SharedGeometry`` init
  (:meth:`~q3vl.whatb.qdecoder.GlutQueryDecoder.assert_step0_identity`);
* the shared degeneracy guard still runs at the **first quick eval**
  (one epoch in on the full horizon), thresholds untouched.
"""

from __future__ import annotations

import contextlib
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from q3vl.whatb import criteria as C
from q3vl.whatb import losses_l0 as L0
from q3vl.whatb import splits as S
from q3vl.whatb.arms import carrier as A
from q3vl.whatb.glut import CLAMP_GRAD_CHOICES, EPS, GlutCarrier, GlutParams
from q3vl.whatb.guards import measure_degeneracy, record_step_witness
from q3vl.whatb.publish import assert_publishable
from q3vl.whatb.qdecoder import (
    QDEC_ACTIVATIONS,
    QDEC_HEAD_INITS,
    GlutQueryDecoder,
    qdecoder_param_count,
)
from q3vl.whatb.scripts import run_carrier_arm as R

# --------------------------------------------------------------------------- #
# 0. identity
# --------------------------------------------------------------------------- #
#: the board's arm name
ARM = "EPR-030"
#: the queue / CLI short name
ARM_NAME = "QDEC"
#: P1: the required criteria table is the twelve frozen keys + the four
#: interpolation columns.  (Passed to ``required_criteria`` explicitly, so no
#: entry has to be added to the shared ``criteria.ARM_AXES`` table.)
AXES: tuple[str, ...] = ("P1",)

RUNNER_PROG = "run_epr030_arm"
RUNNER_DESC = "EPR-030 shared query backbone (N+1-query decoder) + L0 (plain L1)"

BACKBONES: tuple[str, ...] = ("qdec", "mlp")

#: the arm trains on both corpora; ``--data v2seg`` is the paired ablation row
DEFAULT_DATA = "v2seg+l8"

# defaults = StatLUT's published decoder shape (arXiv:2607.08227 appendix)
QDEC_DIM, QDEC_LAYERS, QDEC_HEADS, QDEC_MEM_ROWS = 512, 6, 8, 1
#: the 0.1x group of GLUT App A.1, mapped onto the query prior (EPR-029 §3.5)
QDEC_PRIOR_LR_SCALE = 0.1
#: ... and onto the output head, whose **bias is the shared geometry** under
#: Bias-HyperInit.  Measured basis for the default (see
#: ``GlutQueryDecoder.param_groups``): at ``1.0`` the run reaches NaN by step 80;
#: at ``0.1`` L_rec sits at 0.157-0.168 through step 160.  ``1.0`` is an ablation row.
QDEC_HEAD_LR_SCALE = 0.1
DIAG_EVERY = 200

# -- symbols run_carrier_arm.main() looks up on the arm module and that this
# -- arm does NOT change: re-exported so the runner finds exactly one copy.
BASE_CHECKPOINT = A.BASE_CHECKPOINT
BANK_DIR = A.BANK_DIR
SHORT_SIDE = A.SHORT_SIDE
CONTROL_TAGS = A.CONTROL_TAGS
ZCache = A.ZCache
SampleStore = A.SampleStore
EvalSample = A.EvalSample
evaluate_samples = A.evaluate_samples
interpolation_columns = A.interpolation_columns
library_mean_volume = A.library_mean_volume
build_scheduler = A.build_scheduler
open_bank = A.open_bank


# --------------------------------------------------------------------------- #
# 1. configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Epr030Config(A.CarrierConfig):
    """EPR-024's config plus the backbone, loss and data knobs of this arm.

    Inheriting rather than re-declaring keeps the frozen block in one place:
    ``B x Q = 32 x 256``, ``--clamp two``, the headline formation and the twelve
    pre-registered keys all come from the parent and are not restated here.

    ``train_n`` is **not** a default in this class.  It is measured by
    :func:`q3vl.whatb.splits.train_normal_n` from the sources ``--data`` names
    (each counted against its own on-disk declaration) and passed in by
    :func:`config_from_args`; ``steps_per_epoch`` and ``total_steps`` follow from
    it, and the runner re-asserts both against the population it actually loads.
    """

    #: ``v2seg`` = the frozen sft2seg train split alone; ``v2seg+l8`` adds the
    #: L8 normal rows.  Default is the union -- ``v2seg`` is the paired
    #: ``E030_L8OUT`` ablation row.
    data: str = "v2seg+l8"
    backbone: str = "qdec"
    qdec_dim: int = QDEC_DIM
    qdec_layers: int = QDEC_LAYERS
    qdec_heads: int = QDEC_HEADS
    qdec_self_attn: bool = False
    qdec_mem_rows: int = QDEC_MEM_ROWS
    qdec_act: str = "gelu"
    qdec_prior_lr_scale: float = QDEC_PRIOR_LR_SCALE
    qdec_head_lr_scale: float = QDEC_HEAD_LR_SCALE
    head_init: str = "bias"
    g_residual: bool = False
    clamp_grad: str = "st"
    loss: str = "l0"
    l0_lambda_hc: float = 0.0
    l0_lambda_sparse: float = 0.0
    l0_lambda_mono: float = 0.0
    hc_cnorm: bool = True
    mono_grid: int = 9
    diag_every: int = DIAG_EVERY

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.backbone not in BACKBONES:
            raise ValueError(f"--backbone must be one of {BACKBONES}, got {self.backbone!r}")
        if self.qdec_act not in QDEC_ACTIVATIONS:
            raise ValueError(f"--qdec-act must be one of {QDEC_ACTIVATIONS}")
        if self.head_init not in QDEC_HEAD_INITS:
            raise ValueError(f"--head-init must be one of {QDEC_HEAD_INITS}")
        if self.clamp_grad not in CLAMP_GRAD_CHOICES:
            raise ValueError(f"--clamp-grad must be one of {CLAMP_GRAD_CHOICES}")
        if self.loss not in L0.LOSS_CHOICES:
            raise ValueError(f"--loss must be one of {L0.LOSS_CHOICES}")
        if self.loss_level != 1:
            raise ValueError(
                "EPR-030 fixes the GLUT loss ladder at level 1 (the L1 term); the "
                "recipe is chosen with --loss and the additive rows with "
                "--lambda-hc / --lambda-sparse / --lambda-mono")

    # the arm's own weights replace the ladder's -- the parent derives these
    # from --loss-level, which EPR-030 pins to 1.
    @property
    def lambda_hc(self) -> float:
        return float(self.l0_lambda_hc)

    @property
    def lambda_sparse(self) -> float:
        return float(self.l0_lambda_sparse)

    @property
    def lambda_mono(self) -> float:
        return float(self.l0_lambda_mono)

    @property
    def l0_weights(self) -> L0.L0Weights:
        return L0.L0Weights(lambda_hc=self.lambda_hc,
                            lambda_sparse=self.lambda_sparse,
                            lambda_mono=self.lambda_mono,
                            hc_cnorm=self.hc_cnorm, hc_eps=self.hc_eps,
                            hc_mask=self.hc_mask, mono_grid=self.mono_grid)

    def as_dict(self) -> dict[str, Any]:
        d = super().as_dict()
        d.update({"arm": ARM, "arm_name": ARM_NAME,
                  "lambda_mono": self.lambda_mono,
                  "l0_weights": self.l0_weights.as_dict()})
        return d


def add_arguments(ap) -> None:
    """EPR-024's flag surface plus this arm's; ``--loss-level`` is pinned to 1."""
    A.add_arguments(ap)
    for action in ap._actions:          # the ladder is not this arm's variable
        if action.dest == "loss_level":
            action.default, action.choices = 1, [1]
            action.help = ("pinned to 1 for EPR-030; the recipe is --loss and the "
                           "additive rows are --lambda-hc / --lambda-sparse / --lambda-mono")
    g = ap.add_argument_group("EPR-030 shared query backbone")
    g.add_argument("--data", default=DEFAULT_DATA, choices=list(S.DATA_CHOICES),
                   help="training sources.  v2seg = the frozen sft2seg train "
                        "split (normal-only); v2seg+l8 adds the L8 normal rows. "
                        "n, steps/epoch and total_steps are measured from this, "
                        "never written down")
    g.add_argument("--backbone", default="qdec", choices=list(BACKBONES),
                   help="qdec = the N+1-query decoder; mlp = EPR-024's generator (control row)")
    g.add_argument("--qdec-dim", type=int, default=QDEC_DIM)
    g.add_argument("--qdec-layers", type=int, default=QDEC_LAYERS)
    g.add_argument("--qdec-heads", type=int, default=QDEC_HEADS)
    g.add_argument("--qdec-self-attn", dest="qdec_self_attn", action="store_true",
                   default=False)
    g.add_argument("--qdec-mem-rows", type=int, default=QDEC_MEM_ROWS)
    g.add_argument("--qdec-act", default="gelu", choices=list(QDEC_ACTIVATIONS))
    g.add_argument("--qdec-prior-lr-scale", type=float, default=QDEC_PRIOR_LR_SCALE)
    g.add_argument("--qdec-head-lr-scale", type=float, default=QDEC_HEAD_LR_SCALE,
                   help="lr multiplier of the output head (its bias IS the shared "
                        "geometry under Bias-HyperInit); 1.0 is the ablation row")
    g.add_argument("--head-init", default="bias", choices=list(QDEC_HEAD_INITS))
    g.add_argument("--g-residual", dest="g_residual", action="store_true", default=False,
                   help="G = I + dG instead of G = dG; step 0 is then f = 2x, not the "
                        "identity, and the step-0 assertion is switched off (NOTES 1)")
    g.add_argument("--clamp-grad", default="st", choices=list(CLAMP_GRAD_CHOICES))
    g.add_argument("--loss", default="l0", choices=list(L0.LOSS_CHOICES))
    g.add_argument("--lambda-hc", dest="l0_lambda_hc", type=float, default=0.0)
    g.add_argument("--lambda-sparse", dest="l0_lambda_sparse", type=float, default=0.0)
    g.add_argument("--lambda-mono", dest="l0_lambda_mono", type=float, default=0.0)
    g.add_argument("--hc-cnorm", dest="hc_cnorm", action="store_true", default=True)
    g.add_argument("--no-hc-cnorm", dest="hc_cnorm", action="store_false")
    g.add_argument("--mono-grid", type=int, default=9)
    g.add_argument("--diag-every", type=int, default=DIAG_EVERY)


def config_from_args(args) -> Epr030Config:
    """``argparse.Namespace`` -> :class:`Epr030Config` (no silent defaults).

    ``train_n`` is measured here, from the sources ``--data`` names, so that
    ``steps_per_epoch`` / ``total_steps`` are a function of the data on disk.
    """
    return Epr030Config(
        data=args.data, train_n=S.train_normal_n(args.data,
                                                 split=args.train_split),
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
        backbone=args.backbone, qdec_dim=args.qdec_dim, qdec_layers=args.qdec_layers,
        qdec_heads=args.qdec_heads, qdec_self_attn=args.qdec_self_attn,
        qdec_mem_rows=args.qdec_mem_rows, qdec_act=args.qdec_act,
        qdec_prior_lr_scale=args.qdec_prior_lr_scale,
        qdec_head_lr_scale=args.qdec_head_lr_scale, head_init=args.head_init,
        g_residual=args.g_residual, clamp_grad=args.clamp_grad, loss=args.loss,
        l0_lambda_hc=args.l0_lambda_hc, l0_lambda_sparse=args.l0_lambda_sparse,
        l0_lambda_mono=args.l0_lambda_mono, hc_cnorm=args.hc_cnorm,
        mono_grid=args.mono_grid, diag_every=args.diag_every,
    )


# --------------------------------------------------------------------------- #
# 2. the model
# --------------------------------------------------------------------------- #
class Epr030Model(A.CarrierModel):
    """EPR-024's model with the backbone swapped and the clamp gradient switched.

    ``--backbone qdec`` drops ``pi`` and the MLP generator entirely (they are set
    to ``None``, so they hold no parameters and appear in no optimiser group) and
    routes ``z`` through :class:`~q3vl.whatb.qdecoder.GlutQueryDecoder`.
    ``--backbone mlp`` leaves EPR-024's ``pi`` + ``CGLUTGenerator`` exactly as
    they are, so that row differs from the main row in **one** thing.

    The carrier is rebuilt with ``clamp_grad`` because the clamp's backward is
    one of this arm's variables; the forward is bit-identical either way.
    """

    def __init__(self, cfg: Epr030Config) -> None:
        super().__init__(cfg)
        self.carrier = GlutCarrier(clamp=cfg.clamp, residual=True, eps=EPS,
                                   clamp_grad=cfg.clamp_grad)
        self.qdec: GlutQueryDecoder | None = None
        if cfg.backbone == "qdec":
            self.qdec = GlutQueryDecoder(
                n_gauss=cfg.n_gauss, dim=cfg.qdec_dim, layers=cfg.qdec_layers,
                heads=cfg.qdec_heads, self_attn=cfg.qdec_self_attn,
                mem_rows=cfg.qdec_mem_rows, activation=cfg.qdec_act,
                head_init=cfg.head_init, g_residual=cfg.g_residual, eps=EPS)
            self.pi = None
            self.generator = None

    # -- bookkeeping --
    @property
    def device(self) -> torch.device:
        if self.qdec is not None:
            return self.qdec.head_gauss.weight.device
        return super().device

    @property
    def param_dtype(self) -> torch.dtype:
        if self.qdec is not None:
            return self.qdec.head_gauss.weight.dtype
        return super().param_dtype

    @property
    def config(self) -> dict[str, Any]:
        if self.qdec is None:
            cfg = dict(super().config)
            cfg["backbone"] = "mlp"
            cfg["carrier"] = self.carrier.config
            return cfg
        n = sum(p.numel() for p in self.qdec.parameters())
        return {
            "backbone": "qdec",
            "generator": self.qdec.config,
            "carrier": self.carrier.config,
            "n_params_pi": 0,
            "n_params_generator": int(n),
            "n_params_total": int(n),
            "theta_dim": self.qdec.theta_dim,
            "init": "Bias-HyperInit: head weight 0, head bias = the SharedGeometry "
                    "init; step 0 IS the identity (asserted, not assumed)",
        }

    # -- forward --
    def condition(self, z: Tensor) -> Tensor:
        """``z -> u``: the memory rows for ``qdec``, ``pi(z)`` for ``mlp``.

        The two condition ablations (``--cond-zero`` / ``--cond-trainmean``) are
        applied to ``z`` before either projection, exactly as in EPR-024.
        """
        if self.qdec is None:
            return super().condition(z)
        ref = self.qdec.mem_proj[0].weight
        zc = z.to(device=ref.device, dtype=ref.dtype)
        if self.cfg.cond_zero:
            zc = torch.zeros_like(zc)
        elif self.cfg.cond_trainmean:
            zc = self.train_mean_z.to(device=ref.device, dtype=ref.dtype).expand_as(zc)
        return self.qdec.memory(zc)

    def params_from_condition(self, u: Tensor) -> GlutParams:
        if self.qdec is None:
            return super().params_from_condition(u)
        return self.qdec.decode(u)

    def params_for(self, z: Tensor) -> GlutParams:
        return self.params_from_condition(self.condition(z))

    def param_groups(self, cfg: Epr030Config) -> list[dict[str, Any]]:
        """Adam groups.  ``qdec``: query prior **and output head** at ``0.1x``
        (measured basis in :meth:`~q3vl.whatb.qdecoder.GlutQueryDecoder.param_groups`);
        ``mlp``: EPR-024's two groups, unchanged."""
        if self.qdec is not None:
            return self.qdec.param_groups(cfg.base_lr,
                                          prior_lr_scale=cfg.qdec_prior_lr_scale,
                                          head_lr_scale=cfg.qdec_head_lr_scale)
        return [
            {"params": list(self.generator.parameters()), "lr": float(cfg.base_lr),
             "name": "generator"},
            {"params": list(self.pi.parameters()),
             "lr": float(cfg.base_lr) * float(cfg.proj_lr_scale), "name": "pi"},
        ]

    @torch.no_grad()
    def assert_step0(self) -> dict[str, float]:
        """The decoder's own step-0 witness, or the MLP row's recorded absence."""
        if self.qdec is None:
            return {"backbone": "mlp", "step0_identity_asserted": 0.0}
        if self.cfg.g_residual:
            return {"backbone": "qdec", "step0_identity_asserted": 0.0,
                    "reason": "--g-residual: G = I + dG makes step 0 f = 2x (NOTES 1)"}
        rep = self.qdec.assert_step0_identity()
        rep["step0_identity_asserted"] = 1.0
        return rep


#: the name ``run_carrier_arm.main`` looks the model class up under (it asks the
#: arm module for ``CarrierModel``); EPR-030's class is the one it gets.
CarrierModel = Epr030Model
#: same for the config type, so ``isinstance`` checks in the runner keep working
CarrierConfig = Epr030Config


def build_optimizer(model: Epr030Model, cfg: Epr030Config) -> torch.optim.Adam:
    """Adam with the arm's groups; betas are PyTorch's (the paper gives none)."""
    return torch.optim.Adam(model.param_groups(cfg), lr=float(cfg.base_lr),
                            betas=A.ADAM_BETAS)


# --------------------------------------------------------------------------- #
# 3. the loss + one training step
# --------------------------------------------------------------------------- #
def loss_preregistration(cfg: Epr030Config) -> dict[str, Any]:
    """``config/loss_preregistration.json`` -- what this run promised to optimise."""
    w = cfg.l0_weights
    terms = [
        {"name": "L_rec", "weight": 1.0, "form": "mean | f_theta(x) - L_l(x) |",
         "source": "NILUT Eq.(6) / fit.py:77-78 ('# more stable than L2'); "
                   "model/glut_repro/train_rdg.py loss_main (code fact only)",
         "active": True},
        {"name": "L_hc", "weight": w.lambda_hc,
         "form": ("mean( (C / C.detach().mean()) * (1 - <h_hat, h>) )" if w.hc_cnorm
                  else "mean( C * (1 - <h_hat, h>) )"),
         "source": "CLUT-Net utils/losses.py:24-25 (direction term at weight 1); "
                   "the C normalisation is this project's measured configuration",
         "c_to_zero": {"h": "(a,b)/max(C, eps_c)", "eps_c": w.hc_eps,
                       "hard_mask": w.hc_mask, "masked_on": "target chroma",
                       "logged_as": "n_hc_masked"},
         "active": w.lambda_hc != 0.0},
        {"name": "R_sparse", "weight": w.lambda_sparse,
         "form": "-(1/N) sum_i [o log(o+eps) + (1-o) log(1-o+eps)], eps = 1e-6",
         "source": "GLUT Eq.8, at the family-stable regulariser weight 1e-4 "
                   "(3DLUT / AdaInt / SepLUT sparse_factor)",
         "active": w.lambda_sparse != 0.0},
        {"name": "L_mono", "weight": w.lambda_mono,
         "form": f"mean(relu(v[i] - v[i+1])) over the 3 axes of a {w.mono_grid}^3 grid",
         "source": "3DLUT models.py:340-359 mn_cons at monotonicity_factor = 10",
         "active": w.lambda_mono != 0.0},
    ]
    return {
        "arm": ARM, "arm_name": ARM_NAME, "loss": cfg.loss,
        "total": "L = mean|f(x) - L_l(x)|  (+ lambda_hc L_hc + lambda_sparse R_sparse "
                 "+ lambda_mono L_mono, all 0 on the main row)",
        "terms": terms,
        "weights": w.as_dict(),
        "supervision_space": "function values y = L_l(x); B=32 samples x Q=256 colours",
        "step_columns": list(cfg.step_columns),
        "runtime_assertions": [
            "assert_l0_pure(lambda_hc, lambda_sparse, lambda_mono) before step 0",
            "assert_l0_ran() at the first quick eval (call counter > 0)",
            "GlutQueryDecoder.assert_step0_identity() at construction",
        ],
        "not_used": ["AUC (banned)", "IoU (banned)", "val loss for selection (banned)",
                     "per-image min-max / softmax normalisation (banned)"],
    }


@torch.no_grad()
def _diagnostics(model: Epr030Model, cfg: Epr030Config, *, z: Tensor,
                 x: Tensor, params: GlutParams, aux, acts: Sequence[Mapping[str, float]]
                 ) -> dict[str, Any]:
    """The four low-cost columns this failure was missing (task card D).

    * per-layer near-zero rate of the FFN **pre-activation** (the quantity that
      went 0.469 -> 1.0000 on the MLP head nobody was watching);
    * the fraction of query points the two clamps saturate -- separately for the
      global branch (``Gx + g``) and for the final clamp;
    * ``cross_std`` / ``point_std`` / ``identity_dev`` from the shared
      :func:`~q3vl.whatb.guards.measure_degeneracy`, on the 9^3 grid so the
      three numbers are comparable across steps and across arms.
    """
    out: dict[str, Any] = {}
    for i, a in enumerate(acts):
        out[f"act_nearzero_l{i}"] = float(a["near_zero_rate"])
        out[f"act_dead_l{i}"] = float(a["post_act_zero_rate"])
    if acts:
        out["act_nearzero_max"] = max(float(a["near_zero_rate"]) for a in acts)
        out["act_dead_max"] = max(float(a["post_act_zero_rate"]) for a in acts)
    if aux is not None:
        out["clamp_sat_final"] = float(aux.oob_mask().to(torch.float32).mean())
    glob = torch.einsum("bij,bpj->bpi", params.g_matrix.to(x.dtype), x) \
        + params.g_bias.to(x.dtype).unsqueeze(1)
    out["clamp_sat_global"] = float(((glob < 0.0) | (glob > 1.0)).any(dim=-1)
                                    .to(torch.float32).mean())
    grid = model.query_grid9
    f = model.transform_grid(z, grid)
    rep = measure_degeneracy(f, grid)
    out.update({"point_std": rep.point_std, "identity_dev": rep.identity_dev,
                "cross_std": rep.cross_std})
    return out


def train_step(model: Epr030Model, cfg: Epr030Config, opt: torch.optim.Optimizer,
               scheduler, *, step: int, z: Tensor, lut_ids: Sequence[str],
               bank, sampler, images: Sequence[Tensor] | None = None,
               alphas: Sequence[Tensor | float] | None = None) -> dict[str, Any]:
    """One optimisation step under ``L0``; returns the ``steps.jsonl`` row.

    Mining, the autocast policy and the row contract are EPR-024's, imported.
    The differences are the loss (:func:`q3vl.whatb.losses_l0.l0_losses`) and the
    diagnostics block, which runs every ``--diag-every`` steps and at step 0.
    """
    if images is not None or alphas is not None:
        raise ValueError("EPR-030 supervises function values only; --loss-level 4 "
                         "(the image term) is another EPR's variable")
    device = model.device
    model.train()
    epoch = cfg.epoch_of(step)
    z = z.to(device=device)
    x, y, ratio, n_hard = A.mine_colors(model, cfg, z=z, lut_ids=lut_ids, bank=bank,
                                        sampler=sampler, epoch=epoch, device=device)
    diag_due = cfg.diag_every > 0 and (step == 0 or (step + 1) % cfg.diag_every == 0)
    acts: list[dict[str, float]] = []
    with (model.qdec.capture_activations() if (diag_due and model.qdec is not None)
          else contextlib.nullcontext()) as sink:
        with A._autocast(cfg, device):
            params = model.params_for(z)
        if sink is not None:
            acts = list(sink)
    f, aux = model.carrier(x, params, return_aux=True)

    grid_values = None
    if cfg.lambda_mono != 0.0:
        grid_values = model.carrier(_mono_grid(cfg, device, f.dtype), params)
    loss = L0.l0_losses(f, y, opacity=aux.opacity, grid_values=grid_values,
                        weights=cfg.l0_weights)

    extra: dict[str, Any] = {"step": int(step), "epoch": epoch,
                             "n_hard_colors": int(n_hard),
                             "degenerate_precision": int(aux.degenerate_precision.sum())}
    if diag_due:
        extra["grad_shares"] = L0.term_grad_norm_shares(
            loss.terms, [p for g in opt.param_groups for p in g["params"]])
        extra.update(_diagnostics(model, cfg, z=z, x=x, params=params, aux=aux,
                                  acts=acts))

    opt.zero_grad(set_to_none=True)
    loss.total.backward()
    gnorm = float(torch.nn.utils.clip_grad_norm_(
        [p for g in opt.param_groups for p in g["params"]], cfg.max_grad_norm)
        if cfg.max_grad_norm > 0 else 0.0)
    opt.step()
    if scheduler is not None:
        scheduler.step()

    extra["gnorm"] = gnorm
    for g in opt.param_groups:
        extra[f"lr_{g.get('name', 'group')}"] = float(g["lr"])
    row = loss.row(n_luts_in_batch=len({str(l) for l in lut_ids}),
                   mining_ratio_value=ratio, extra=extra)
    if step == 0:
        record_step_witness(row)
    return row


def _mono_grid(cfg: Epr030Config, device, dtype) -> Tensor:
    """The uniform ``n^3`` grid the monotonicity hinge is measured on."""
    t = torch.linspace(0.0, 1.0, int(cfg.mono_grid), device=device, dtype=dtype)
    return torch.stack(torch.meshgrid(t, t, t, indexing="ij"), dim=-1).reshape(-1, 3)


def quick_eval(model: Epr030Model, cfg: Epr030Config, *, z: Tensor,
               lut_ids: Sequence[str], bank, step: int, first: bool = True,
               **kw) -> dict[str, Any]:
    """EPR-024's quick eval (degeneracy guard included) + the L0 counter check.

    ``step < 0`` is the runner's ``--eval-only`` tail call: there was no training
    loop in this process, so the counter is not asserted there (it would report a
    training failure for a run that did not train).
    """
    row = A.quick_eval(model, cfg, z=z, lut_ids=lut_ids, bank=bank, step=step,
                       first=first, **kw)
    if first and step >= 0:
        row["l0_calls"] = L0.assert_l0_ran(where=f"quick_eval@step{step}")
    return row


# --------------------------------------------------------------------------- #
# 4. board + record
# --------------------------------------------------------------------------- #
def build_arm_board(rows: Sequence[Mapping[str, Any]], *, split: str,
                    extra_columns: Mapping[str, Mapping[str, Any]],
                    published: bool = True, seed: int = A.DEFAULT_SEED,
                    facts: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """``criteria.build_board`` under this arm's name.  Columns are unchanged."""
    board = C.build_board(rows, arm=ARM, split=split, extra_columns=extra_columns,
                          seed=seed)
    board["published"] = bool(published)
    board["arm_name"] = ARM_NAME
    if facts:
        board["facts"] = dict(facts)
    return board


def publish_board(board: Mapping[str, Any], cfg: Epr030Config, *,
                  steps_path: Any = None, steps_row: Mapping[str, Any] | None = None,
                  eval_only: bool = False) -> dict[str, Any]:
    """The one publication gate.  ``axes`` is passed explicitly (P1)."""
    return assert_publishable(board, ARM, steps_row=steps_row, steps_path=steps_path,
                              eval_only=eval_only, loss_level=1, axes=AXES)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_setup_record(cfg: Epr030Config, model: Epr030Model, **kw) -> dict[str, Any]:
    """EPR-024's record, re-stamped for this arm and with the new sources hashed."""
    rec = A.run_setup_record(cfg, model, **kw)
    rec["arm"], rec["arm_name"], rec["axes"] = ARM, ARM_NAME, list(AXES)
    here = Path(__file__).resolve()
    pkg = here.parent.parent
    rec["source_sha256"].update({
        n: _sha256(pkg / n) for n in ("qdecoder.py", "losses_l0.py", "glut.py")
        if (pkg / n).is_file()})
    rec["source_sha256"]["run_epr030_arm.py"] = _sha256(here)
    rec["frozen_block"]["clamp_grad"] = cfg.clamp_grad
    rec["epr030"] = {
        "data": cfg.data,
        "data_sources": list(S.DATA_SOURCES[cfg.data]),
        "train_normal_n_measured": cfg.train_n,
        "steps_per_epoch": cfg.steps_per_epoch,
        "total_steps": cfg.total_steps,
        "backbone": cfg.backbone,
        "qdec": (model.qdec.config if model.qdec is not None else None),
        "qdec_param_count_closed_form": qdecoder_param_count(
            n_gauss=cfg.n_gauss, dim=cfg.qdec_dim, layers=cfg.qdec_layers,
            mem_rows=cfg.qdec_mem_rows, self_attn=cfg.qdec_self_attn),
        "loss": cfg.loss,
        "weights": cfg.l0_weights.as_dict(),
        "clamp_grad": cfg.clamp_grad,
        "diag_every": cfg.diag_every,
        "step0": model.assert_step0(),
        # UNCONDITIONAL.  The former ``if cfg.l0_weights.pure_l1`` guard was the
        # exact complement of the condition ``assert_l0_pure`` raises on, so the
        # assertion could not fire for any input -- "defined but not wired", the
        # fourth time.  ``loss`` is not an experimental axis of this arm, so a
        # non-zero ``--lambda-hc/--lambda-sparse/--lambda-mono`` stops the
        # process here, before step 0.  The flags stay (default 0.0).
        "l0_purity_check": L0.assert_l0_pure(
            lambda_hc=cfg.lambda_hc, lambda_sparse=cfg.lambda_sparse,
            lambda_mono=cfg.lambda_mono, loss=cfg.loss),
        "optimizer_groups": [
            {"name": g.get("name"), "lr": g["lr"],
             "n_params": int(sum(p.numel() for p in g["params"]))}
            for g in model.param_groups(cfg)],
    }
    return rec


# --------------------------------------------------------------------------- #
# 5. entry point
# --------------------------------------------------------------------------- #
def build_parser():
    """The full CLI of this arm (EPR-024's surface + section 1's flags)."""
    return R.build_parser(sys.modules[__name__])


def main(argv: list[str] | None = None) -> int:
    """``run_carrier_arm.main`` driven by this module -- one pipeline, two arms."""
    return R.main(argv, arm=sys.modules[__name__])


if __name__ == "__main__":
    raise SystemExit(main())
