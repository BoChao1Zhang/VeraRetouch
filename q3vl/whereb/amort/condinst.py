"""EPR-023: CondInst's conditional-convolution mask head, ported verbatim.

Reference: *Conditional Convolutions for Instance Segmentation* (arXiv 2003.05664,
Tian / Shen / Chen), official implementation ``aim-uofa/AdelaiDet``.  Every
structural number in this file is quoted from the proposal
``experiments/prs/EPR-023_relcoord-dynamic-head/PROPOSAL.md``, which in turn
quotes the upstream files opened on 2026-08-14:

* ``adet/modeling/condinst/mask_branch.py:36-51``  -- the shared mask branch
* ``adet/modeling/condinst/dynamic_mask_head.py``  -- ``dice_coefficient`` L51-59,
  ``parse_dynamic_params`` L62-87, the ``weight_nums``/``bias_nums`` construction
  L114-131, ``mask_heads_forward`` L135-153, the rel-coord block L169-179, the
  ``aligned_bilinear`` call L190-196, ``mask_scores = sigmoid`` L221, the dice
  mask loss L245-249 and the no-instance dummy loss L210-216
* ``adet/modeling/condinst/condinst.py:103-108``  -- the controller and its init
* ``adet/utils/comm.py:23-45 / 48-61``            -- ``aligned_bilinear`` /
  ``compute_locations``
* ``adet/layers/conv_with_kaiming_uniform.py:23-47`` -- ``conv_block``
* ``adet/config/defaults.py`` + ``detectron2/config/defaults.py`` -- every
  hyper-parameter default and the whole optimizer recipe

What is NOT upstream (each one is listed in the proposal's NOVEL table and is
recorded in ``facts()``):

* single-scale input: the frozen base exposes one ``F_pre`` (stride 16), so there
  is one refine conv and no FPN sum (``mask_branch.py:70-84`` has no counterpart);
* the controller is a ``Linear(2560, 169 + 2)`` on the language condition
  ``h_cond`` instead of a ``Conv2d`` read at an FCOS positive location -- one
  image, one mask, no positions to select from;
* the instance centre is the controller's own 2-dim ``sigmoid`` output rather
  than the assigned grid location, and the rel-coords are normalised by the
  frame (cell centres ``(i + 0.5)/n``) rather than by a per-level
  ``sizes_of_interest``;
* the upsample factor is ``16 / 4 = 4`` instead of ``8 / 4 = 2`` (same operator,
  same output stride, different input stride);
* the FCOS / BoxInst / semantic-auxiliary losses are removed (no detector);
* the geometry-parameter regression head is the batch's mandated ablation row
  (``--geom-reg-weight``); CondInst has no such head.

``dice`` stays the optimisation target: the campaign's "no dice/IoU in a loss"
red line was lifted for this batch of faithful ports by the user on 2026-08-14.

Nothing here is imported by the live arms: the module is reached only through
``q3vl.whereb.amort.arms`` when ``--arm CONDINST`` is passed.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from q3vl.where.upsample import area_resize
from q3vl.whereb.fields import no_autocast

__all__ = [
    "ARM", "CRITERIA", "CondInstConfig", "CondInstHead", "MaskBranch",
    "Controller", "GeomRegHead", "SpanQueryReadout",
    "conv_block", "aligned_bilinear", "normalized_locations",
    "dynamic_param_nums", "parse_dynamic_params", "mask_heads_forward",
    "dice_coefficient", "geom_targets", "GEOM_LAYOUT",
    "config", "set_config", "assert_first_step_columns", "loss_preregistration",
    "build_head", "forward", "compute_loss", "head_kwargs_from_args",
    "optimizer_spec", "scheduler_kwargs", "builder_kwargs", "readout_spec",
    "per_sample_row", "criteria_columns", "train_stats",
]

#: registry name and pre-registered criterion column (``arms.ARM_CRITERIA``)
ARM = "CONDINST"
CRITERIA = ("condinst_pix_readout",)

# --------------------------------------------------------------------------- #
# frozen constants (proposal §3 "新增超参与建议默认值")
# --------------------------------------------------------------------------- #
MASK_BRANCH_CHANNELS = 128          # MASK_BRANCH.CHANNELS      defaults.py:246
MASK_BRANCH_NUM_CONVS = 4           # MASK_BRANCH.NUM_CONVS     defaults.py:248
MASK_BRANCH_NORM = "BN"             # MASK_BRANCH.NORM          defaults.py:247
MASK_BRANCH_OUT_CHANNELS = 8        # MASK_BRANCH.OUT_CHANNELS  defaults.py:244
MASK_HEAD_CHANNELS = 8              # MASK_HEAD.CHANNELS        defaults.py:238
MASK_HEAD_NUM_LAYERS = 3            # MASK_HEAD.NUM_LAYERS      defaults.py:239
DISABLE_REL_COORDS = False          # MASK_HEAD.DISABLE_REL_COORDS defaults.py:241
MASK_OUT_STRIDE = 4                 # MASK_OUT_STRIDE           defaults.py:228
#: ``F_pre`` is H/16 (``q3vl/where/fpre.py``); CondInst's mask feature is p3 =
#: stride 8.  The output stride is kept at 4, so the factor is 4 and not 2.
FEAT_STRIDE = 16
DICE_EPS = 1e-5                     # dynamic_mask_head.py:51-59
CONTROLLER_INIT_STD = 0.01          # condinst.py:107-108
TEXT_DIM = 2560                     # contracts.py:28-40

# optimizer / schedule: Base-CondInst.yaml + detectron2/config/defaults.py
BASE_LR = 0.01                      # SOLVER.BASE_LR
MOMENTUM = 0.9                      # defaults.py:534
NESTEROV = False                    # defaults.py:536
WEIGHT_DECAY = 1e-4                 # defaults.py:538
WEIGHT_DECAY_NORM = 0.0             # defaults.py:541  (bias follows :577 = 1e-4)
LR_GAMMA = 0.1                      # defaults.py:543
WARMUP_FACTOR = 1.0 / 1000.0        # defaults.py:549
#: SOLVER.LR_SCHEDULER_NAME "WarmupMultiStepLR" (defaults.py:531).  The kind is
#: pre-registered, not a knob: `scheduler_kwargs` refuses anything else, because
#: `make_scheduler` ignores `milestones`/`gamma` under `cosine` while still
#: honouring `warmup_factor` -- a mixed schedule that no board can show.
SCHEDULER_KIND = "warmup_multistep"
#: (60000, 80000) / MAX_ITER 90000, carried over as fractions (proposal §3):
#: ``round(1200 * 60000/90000) = 800``, ``round(1200 * 80000/90000) = 1067``
MILESTONE_FRACS = (60000.0 / 90000.0, 80000.0 / 90000.0)
#: WARMUP_ITERS 1000 / 90000 -> ``round(1200 * 1000/90000) = 13``
WARMUP_FRAC = 1000.0 / 90000.0
#: the proposal writes the 1200-step values dead; the fractions above must
#: reproduce them exactly or the schedule is not the one that was registered
MILESTONES_AT_1200 = [800, 1067]
WARMUP_AT_1200 = 13

#: ``--geom-reg-weight`` output layout (proposal "消融行必含项①").  Continuous
#: columns are L1, ``flipped`` is BCE, ``route`` is a 3-way logit.
GEOM_ROUTE_CLASSES = ("circulargradient", "gradient", "semantic")
GEOM_LAYOUT: dict[str, tuple[str, ...]] = {
    "circulargradient": ("cx", "cy", "rx", "ry", "sin2a", "cos2a", "feather"),
    "gradient": ("zero_x", "zero_y", "full_x", "full_y"),
}
GEOM_N_OUT = (len(GEOM_ROUTE_CLASSES)
              + len(GEOM_LAYOUT["circulargradient"]) + 1      # + flipped
              + len(GEOM_LAYOUT["gradient"]) + 1)             # + flipped


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CondInstConfig:
    """Every ``--condinst-*`` flag, frozen into one record for the run setup.

    The defaults ARE the faithful recipe: constructing ``CondInstConfig()`` and
    running it is the main arm, and each ablation row is exactly one field away.
    """

    # -- head ---------------------------------------------------------------
    mask_out_stride: int = MASK_OUT_STRIDE
    head_channels: int = MASK_HEAD_CHANNELS
    head_layers: int = MASK_HEAD_NUM_LAYERS
    disable_rel_coords: bool = DISABLE_REL_COORDS
    mask_branch_channels: int = MASK_BRANCH_CHANNELS
    mask_branch_num_convs: int = MASK_BRANCH_NUM_CONVS
    mask_branch_out_channels: int = MASK_BRANCH_OUT_CHANNELS
    mask_branch_norm: str = MASK_BRANCH_NORM
    #: ``seg`` = h_cond (<seg_where>), ``pool`` = <where> span mean-pool,
    #: ``query`` = a learnable query cross-attending the <where> span rows
    ctrl: str = "seg"
    # -- supervision --------------------------------------------------------
    gt: str = "raster"                 # raster | cgt   (--condinst-gt)
    mask_loss: str = "dice"            # dice | bce     (--condinst-mask-loss)
    center_sup: bool = False           # --condinst-center-sup
    center_weight: float = 0.05        # NOVEL: no upstream value exists
    geom_reg_weight: float = 0.0       # --geom-reg-weight (0 = branch not built)
    geom_hidden: int = 256
    #: NOVEL: the proposal specifies a 3-way route logit and a route-accuracy
    #: readout column but no loss for it.  ``ce`` supervises it (weight 1.0
    #: inside ``L_geom``, i.e. ``geom_reg_weight`` overall); ``none`` leaves it
    #: unsupervised, which makes the accuracy column read the zero-init argmax.
    geom_route_loss: str = "ce"
    #: D-1: the angle columns of near-isotropic samples are noise construction
    #: side.  The proposal's conservative default is NOT to mask them, and to
    #: record that they were not masked.
    geom_angle_mask_ratio: float = 0.0
    # -- optimizer / schedule ----------------------------------------------
    optimizer: str = "sgd"
    lr: float = BASE_LR
    weight_decay: float = WEIGHT_DECAY
    weight_decay_norm: float = WEIGHT_DECAY_NORM
    momentum: float = MOMENTUM
    nesterov: bool = NESTEROV
    scheduler: str = SCHEDULER_KIND
    lr_milestones: str = ""            # "" = derive from the step budget
    lr_gamma: float = LR_GAMMA
    warmup_iters: int = -1             # <0 = derive from the step budget
    warmup_factor: float = WARMUP_FACTOR
    seed: int = 20260810

    def __post_init__(self) -> None:
        if self.ctrl not in ("seg", "pool", "query"):
            raise ValueError(f"--condinst-ctrl must be seg|pool|query, got {self.ctrl!r}")
        if self.gt not in ("raster", "cgt"):
            raise ValueError(f"--condinst-gt must be raster|cgt, got {self.gt!r}")
        if self.mask_loss not in ("dice", "bce"):
            raise ValueError(
                f"--condinst-mask-loss must be dice|bce, got {self.mask_loss!r}")
        if self.geom_route_loss not in ("ce", "none"):
            raise ValueError(
                f"--condinst-geom-route-loss must be ce|none, got "
                f"{self.geom_route_loss!r}")
        if self.optimizer not in ("sgd", "adamw", "adam"):
            raise ValueError(f"--optimizer must be sgd|adamw|adam, got {self.optimizer!r}")
        if self.head_layers < 2:
            raise ValueError(
                "the dynamic head needs at least an input and an output layer "
                f"(dynamic_mask_head.py:114-131), got {self.head_layers}")
        if FEAT_STRIDE % int(self.mask_out_stride):
            raise ValueError(
                f"--condinst-mask-out-stride {self.mask_out_stride} does not "
                f"divide the F_pre stride {FEAT_STRIDE}; the upsample factor "
                "(mask_feat_stride / mask_out_stride, dynamic_mask_head.py:190-196) "
                "would not be an integer")
        if self.geom_reg_weight < 0:
            raise ValueError("--geom-reg-weight must be >= 0")
        if self.geom_reg_weight > 0 and self.gt == "cgt":
            # the analytic `geometry` dict only reaches the head through the
            # PixGT render path; with `--condinst-gt cgt` there is none, and the
            # regression would silently train on an all-zero target
            raise ValueError(
                "--geom-reg-weight > 0 needs --condinst-gt raster: the "
                "regression targets are the construction-side `geometry` dict, "
                "which reaches the head through the analytic PixGT path only "
                "(pixgt.PixGT.geometry). Ablation rows ① and ④ are separate rows.")

    # -- derived -----------------------------------------------------------
    @property
    def up_factor(self) -> int:
        """``mask_feat_stride / mask_out_stride`` (dynamic_mask_head.py:190-196)."""
        return FEAT_STRIDE // int(self.mask_out_stride)

    @property
    def geom_on(self) -> bool:
        return float(self.geom_reg_weight) > 0.0

    @property
    def readout_kind(self) -> str:
        """The ``q3vl.whereb.readout`` kind this ``ctrl`` implies, if any."""
        return {"pool": "where_span_pool", "query": "where_span_pool"}.get(self.ctrl, "")

    def milestones(self, total_steps: int) -> list[int]:
        if self.lr_milestones:
            return [int(x) for x in str(self.lr_milestones).replace(" ", "").split(",") if x]
        from q3vl.where.calibrate import scale_milestones

        if int(total_steps) <= 0:
            raise ValueError(
                "the CondInst milestones are carried over from (60000, 80000) / "
                "MAX_ITER 90000 as fractions, so the step budget must be known; "
                "pass --max-steps (default 1200) or --lr-milestones 800,1067")
        ms = scale_milestones(MILESTONE_FRACS, int(total_steps))
        if int(total_steps) == 1200 and ms != MILESTONES_AT_1200:
            raise AssertionError(
                f"the 1200-step milestones must be {MILESTONES_AT_1200} "
                f"(proposal §3 optimizer table), got {ms}")
        return ms

    def warmup(self, total_steps: int) -> int:
        if int(self.warmup_iters) >= 0:
            return int(self.warmup_iters)
        if int(total_steps) <= 0:
            raise ValueError(
                "the warmup length is 1000/90000 of the budget; pass --max-steps "
                "or --warmup-iters 13")
        w = int(round(WARMUP_FRAC * int(total_steps)))
        if int(total_steps) == 1200 and w != WARMUP_AT_1200:
            raise AssertionError(
                f"the 1200-step warmup must be {WARMUP_AT_1200} steps, got {w}")
        return w

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(up_factor=self.up_factor, geom_on=self.geom_on,
                 feat_stride=FEAT_STRIDE,
                 readout_kind_from_ctrl=self.readout_kind or None)
        return d


#: module-level configuration seam, mirroring ``uniq4.VARIANT4``: the wrapper
#: ``run_condinst_arm.py`` sets it before delegating to ``run_amort_arm.main``,
#: because the optional hooks below are handed ``run_amort_arm``'s namespace and
#: not the arm's own.  Unset = the faithful defaults, so ``--arm CONDINST`` on
#: the base entry runs the registered recipe rather than something undefined.
_CONFIG: CondInstConfig | None = None


def set_config(cfg: CondInstConfig | None) -> None:
    global _CONFIG
    _CONFIG = cfg


def config() -> CondInstConfig:
    return _CONFIG if _CONFIG is not None else CondInstConfig()


def loss_preregistration(args: Any = None) -> dict[str, Any]:
    """``arms.RUN_REQUIRED_HOOKS``: what ``loss_preregistration.json`` says.

    ``run_amort_arm``'s shared record is the live arms' seven-term ST_LANG
    stack with ``dice_as_target: false``.  For this arm dice IS the target
    (``condinst.py:161-165`` sums the mask loss in with no coefficient), so the
    shared record would be false twice over.
    """
    c = config()
    dice = str(c.mask_loss) == "dice"
    form = f"L = 1.0 * {c.mask_loss}(m_pix, gt_pix)"
    if c.center_sup:
        form += f" + {float(c.center_weight)} * |center_pred - centroid(gt)|"
    if float(c.geom_reg_weight) > 0:
        form += f" + {float(c.geom_reg_weight)} * L_geom"
    return {
        "arm": ARM,
        "form": form,
        "mask_loss": ("1 - 2*sum(x*t)/(sum(x^2) + sum(t^2) + "
                      f"{DICE_EPS})   [dynamic_mask_head.py:51-59]") if dice
                     else "BCEWithLogits on the same logit (ablation row)",
        "coefficient": "1.0 -- condinst.py:161-165 sums the mask loss in with "
                       "no coefficient",
        "center_sup": (f"{float(c.center_weight)} * L1 on the predicted "
                       "centroid") if c.center_sup else "off",
        "geom_reg": (f"{float(c.geom_reg_weight)} * (L1 on the normalised "
                     "geometry parameters + route CE)") if float(c.geom_reg_weight) > 0
                    else "off",
        "is_fake": "the no-instance branch of dynamic_mask_head.py:210-216: a "
                   "zero-valued term built from the tensors that would have "
                   "carried the gradient",
        "seven_term_stack": "not entered (trainer.py:244-251)",
        "iou_as_field_target": False,
        "iou_as_selection_target": False,
        "dice_as_target": dice,
        "dice_red_line_waiver": ("the campaign's 'dice is never a mask target' "
                                 "line (losses.py:9-11) is waived for the "
                                 "EPR-018..023 batch by the user's 2026-08-14 "
                                 "instruction; CondInst's mask loss IS dice")
        if dice else None,
    }


# --------------------------------------------------------------------------- #
# verbatim ports
# --------------------------------------------------------------------------- #
def conv_block(in_channels: int, out_channels: int, kernel_size: int = 3,
               norm: str | None = MASK_BRANCH_NORM) -> nn.Sequential:
    """``conv_with_kaiming_uniform.py:23-47``: conv -> norm -> ReLU(inplace).

    ``bias=(norm is None)``; the weight is ``kaiming_uniform_(a=1)`` and the bias,
    when there is one, is zeroed.
    """
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1,
                     padding=(kernel_size - 1) // 2, bias=(norm is None))
    nn.init.kaiming_uniform_(conv.weight, a=1)
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)
    layers: list[nn.Module] = [conv]
    if norm:
        if norm == "BN":
            layers.append(nn.BatchNorm2d(out_channels))
        elif norm == "GN":
            layers.append(nn.GroupNorm(32, out_channels))
        else:
            raise ValueError(f"unsupported MASK_BRANCH.NORM {norm!r}")
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def aligned_bilinear(tensor: torch.Tensor, factor: int) -> torch.Tensor:
    """``adet/utils/comm.py:23-45``, transcribed line for line."""
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(tensor, size=(oh, ow), mode="bilinear", align_corners=True)
    tensor = F.pad(tensor, pad=(factor // 2, 0, factor // 2, 0), mode="replicate")

    return tensor[:, :, :oh - 1, :ow - 1]


def normalized_locations(h: int, w: int, *, device=None,
                         dtype=torch.float32) -> torch.Tensor:
    """``(2, h, w)`` cell-centre coordinates as ``[x; y]`` in ``[0, 1]``.

    ``compute_locations`` (``comm.py:48-61``) puts a location at
    ``arange(0, n*stride, stride) + stride // 2``, i.e. the centre of each cell
    in pixel units.  Divided by the extent that is ``(i + 0.5)/n`` -- the same
    rewrite the proposal registers, and the same cell-centre convention
    ``pixgt.make_coord`` uses.
    """
    xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) / float(w)
    ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) / float(h)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], dim=0)


def dynamic_param_nums(in_channels: int, channels: int, num_layers: int,
                       disable_rel_coords: bool = False
                       ) -> tuple[list[int], list[int], int]:
    """``dynamic_mask_head.py:114-131``.

    With the defaults (``in_channels = 8``, ``channels = 8``, ``num_layers = 3``,
    rel-coords ON) this is ``[80, 64, 8]`` / ``[8, 8, 1]`` -> **169**; with
    ``DISABLE_REL_COORDS`` the first layer becomes ``8*8`` -> **153**.
    """
    in_ch = int(in_channels) + (0 if disable_rel_coords else 2)
    weight_nums: list[int] = []
    bias_nums: list[int] = []
    for i in range(int(num_layers)):
        if i == 0:
            weight_nums.append(in_ch * channels)
            bias_nums.append(channels)
        elif i == num_layers - 1:
            weight_nums.append(channels * 1)
            bias_nums.append(1)
        else:
            weight_nums.append(channels * channels)
            bias_nums.append(channels)
    return weight_nums, bias_nums, sum(weight_nums) + sum(bias_nums)


def parse_dynamic_params(params: torch.Tensor, channels: int,
                         weight_nums: Sequence[int], bias_nums: Sequence[int]
                         ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """``dynamic_mask_head.py:62-87``, transcribed line for line."""
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)

    num_insts = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(torch.split_with_sizes(
        params, list(weight_nums) + list(bias_nums), dim=1))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        if l < num_layers - 1:
            weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
        else:
            weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts)

    return weight_splits, bias_splits


def mask_heads_forward(features: torch.Tensor, weights: Sequence[torch.Tensor],
                       biases: Sequence[torch.Tensor], num_insts: int) -> torch.Tensor:
    """``dynamic_mask_head.py:135-153``, transcribed line for line."""
    assert features.dim() == 4
    n_layers = len(weights)
    x = features
    for i, (w, b) in enumerate(zip(weights, biases)):
        x = F.conv2d(x, w, bias=b, stride=1, padding=0, groups=num_insts)
        if i < n_layers - 1:
            x = F.relu(x)
    return x


def dice_coefficient(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """``dynamic_mask_head.py:51-59``, transcribed line for line.

    ``1 - 2 * sum(x*t) / (sum(x^2) + sum(t^2) + 1e-5)``.  The soft ``.cgt`` alpha
    goes in unchanged: the formula is defined for any target in [0, 1] and the
    proposal registers it unmodified.
    """
    eps = DICE_EPS
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1.0 - (2 * intersection / union)
    return loss


# --------------------------------------------------------------------------- #
# modules
# --------------------------------------------------------------------------- #
class MaskBranch(nn.Module):
    """``mask_branch.py:36-51``, single scale.

    One ``conv_block`` refine (there is exactly one input feature, so there is
    nothing to sum: ``mask_branch.py:70-84`` has no counterpart here), then
    ``num_convs`` 3x3 ``conv_block``s, then a bare ``Conv2d(channels, 8, 1)``
    with PyTorch's default initialisation (``mask_branch.py:48-50``).
    """

    def __init__(self, in_dim: int = 1024, channels: int = MASK_BRANCH_CHANNELS,
                 num_convs: int = MASK_BRANCH_NUM_CONVS,
                 out_channels: int = MASK_BRANCH_OUT_CHANNELS,
                 norm: str = MASK_BRANCH_NORM):
        super().__init__()
        self.in_dim = int(in_dim)
        self.channels = int(channels)
        self.num_convs = int(num_convs)
        self.out_channels = int(out_channels)
        self.norm = norm
        self.refine = conv_block(in_dim, channels, 3, norm)
        tower: list[nn.Module] = [conv_block(channels, channels, 3, norm)
                                  for _ in range(int(num_convs))]
        tower.append(nn.Conv2d(channels, max(int(out_channels), 1), 1))
        self.tower = nn.Sequential(*tower)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.tower(self.refine(feat))


class Controller(nn.Module):
    """``condinst.py:103-108`` with the conv replaced by a linear on ``h_cond``.

    Upstream reads ``Conv2d(in_channels, num_gen_params, 3, padding=1)`` at the
    FCOS positive location of the instance.  One image / one mask means there is
    no location to select, so the nearest object is the same single affine map
    applied to the language condition; the initialisation is unchanged
    (``normal_(std=0.01)``, ``constant_(bias, 0)``).

    ``+2`` outputs carry the instance centre (NOVEL, proposal §2): upstream takes
    it from the assigned grid cell, which does not exist here.
    """

    def __init__(self, text_dim: int = TEXT_DIM, num_gen_params: int = 169,
                 n_center: int = 2):
        super().__init__()
        self.text_dim = int(text_dim)
        self.num_gen_params = int(num_gen_params)
        self.n_center = int(n_center)
        self.proj = nn.Linear(int(text_dim), int(num_gen_params) + int(n_center))
        torch.nn.init.normal_(self.proj.weight, std=CONTROLLER_INIT_STD)
        torch.nn.init.constant_(self.proj.bias, 0)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(K, text_dim)`` -> ``(num_gen_params,)`` theta and ``(2,)`` centre.

        K > 1 (the ``qtok`` / ``nseg`` readout rows) is the pre-registered NOVEL
        aggregation of proposal §4: **each row through the same Linear, then the
        mean over rows** -- the head still emits one theta and one centre, and
        the 169-parameter form is unchanged.
        """
        if h.dim() == 1:
            h = h.unsqueeze(0)
        out = self.proj(h).mean(dim=0)
        theta = out[: self.num_gen_params]
        center = torch.sigmoid(out[self.num_gen_params:])
        return theta, center


class SpanQueryReadout(nn.Module):
    """``--condinst-ctrl query`` (proposal §4 row ⑩): one learnable query vector
    cross-attending the ``<where>`` span rows.

    Single head, no projections -- the proposal specifies "query learnable,
    key/value = the span's T rows", and adding K/V projections would be a second
    unregistered change.  The query is **zero-initialised**, so at step 0 the
    attention is uniform and the readout is exactly the span mean-pool of row ⑥;
    the two rows therefore start from the same vector and differ only by what
    training does to the query.
    """

    def __init__(self, dim: int = TEXT_DIM):
        super().__init__()
        self.dim = int(dim)
        self.query = nn.Parameter(torch.zeros(int(dim)))

    def forward(self, rows: torch.Tensor, mask: torch.Tensor | None = None
                ) -> torch.Tensor:
        if rows.dim() == 3:
            rows = rows[0]
        if rows.dim() != 2:
            raise ValueError(f"expected (T, D) span rows, got {tuple(rows.shape)}")
        scores = rows @ self.query / math.sqrt(float(self.dim))
        if mask is not None:
            m = mask.reshape(-1).bool()
            if m.numel() == scores.numel():
                scores = scores.masked_fill(~m, float("-inf"))
        attn = torch.softmax(scores, dim=0)
        return attn @ rows


class GeomRegHead(nn.Module):
    """Ablation row ① : the geometry-parameter regression branch.

    ``Linear(2560, 256) + ReLU + Linear(256, D)`` with the **last layer zero
    initialised** (campaign zero-init discipline), reading the controller's own
    input vector.  ``D`` = 3 route logits + (7 continuous + 1 flipped) for
    ``circulargradient`` + (4 continuous + 1 flipped) for ``gradient``.

    Coordinate columns carry no sigmoid/clamp: measured construction-side values
    fall outside [0, 1] (a band sample has ``Left = -1.1007``).
    """

    def __init__(self, text_dim: int = TEXT_DIM, hidden: int = 256):
        super().__init__()
        self.text_dim = int(text_dim)
        self.hidden = int(hidden)
        self.fc1 = nn.Linear(int(text_dim), int(hidden))
        self.fc2 = nn.Linear(int(hidden), GEOM_N_OUT)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        n_r = len(GEOM_ROUTE_CLASSES)
        n_c = len(GEOM_LAYOUT["circulargradient"])
        self._slices = {
            "route": slice(0, n_r),
            "circulargradient": slice(n_r, n_r + n_c),
            "circulargradient_flipped": slice(n_r + n_c, n_r + n_c + 1),
            "gradient": slice(n_r + n_c + 1, n_r + n_c + 1 + len(GEOM_LAYOUT["gradient"])),
            "gradient_flipped": slice(GEOM_N_OUT - 1, GEOM_N_OUT),
        }

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        if h.dim() == 2:                       # K > 1 readouts: mean over rows
            h = h.mean(dim=0)
        y = self.fc2(F.relu(self.fc1(h)))
        return {name: y[sl] for name, sl in self._slices.items()}


class CondInstHead(nn.Module):
    """mask branch + controller + the dynamic 1x1 head (+ the optional branches).

    ``forward(feat, h_cond)`` returns ``m_low`` on the ``(gh, gw)`` criterion
    grid, ``m_pix`` on the head's native ``stride-4`` grid, and everything the
    loss and the diagnostics read.
    """

    def __init__(self, in_dim: int = 1024, text_dim: int = TEXT_DIM,
                 cfg: CondInstConfig | None = None):
        super().__init__()
        self.cfg = cfg or CondInstConfig()
        c = self.cfg
        self.in_dim = int(in_dim)
        self.text_dim = int(text_dim)
        # -- ORDER MATTERS: the two faithful modules are constructed first, so
        # turning an ablation branch on cannot shift the RNG stream the main
        # head is initialised from (tests/test_condinst.py pins this).
        self.mask_branch = MaskBranch(in_dim, c.mask_branch_channels,
                                      c.mask_branch_num_convs,
                                      c.mask_branch_out_channels,
                                      c.mask_branch_norm)
        self.weight_nums, self.bias_nums, self.num_gen_params = dynamic_param_nums(
            c.mask_branch_out_channels, c.head_channels, c.head_layers,
            c.disable_rel_coords)
        self.controller = Controller(text_dim, self.num_gen_params, n_center=2)
        self.span_query = SpanQueryReadout(text_dim) if c.ctrl == "query" else None
        self.geom = GeomRegHead(text_dim, c.geom_hidden) if c.geom_on else None

    # -- the head ----------------------------------------------------------
    def condition(self, h_cond: torch.Tensor | None = None,
                  h_where: torch.Tensor | None = None,
                  h_mask: torch.Tensor | None = None) -> torch.Tensor:
        """``(K, text_dim)`` rows the controller consumes, per ``--condinst-ctrl``."""
        if self.span_query is not None:
            if h_where is None:
                raise ValueError(
                    "--condinst-ctrl query reads the <where> span ROWS "
                    "(ArmContext.h_where) and the batch carried none")
            return self.span_query(h_where, h_mask).reshape(1, -1)
        if h_cond is None:
            raise ValueError(
                "CONDINST reads h_cond (the readout of q3vl.whereb.readout); the "
                "batch builder supplied none")
        return h_cond.reshape(-1, self.text_dim)

    def forward(self, feat: torch.Tensor, h_cond: torch.Tensor | None = None, *,
                h_where: torch.Tensor | None = None,
                h_mask: torch.Tensor | None = None) -> dict[str, Any]:
        if feat.dim() != 4:
            raise ValueError(f"expected (1, C, gh, gw) features, got {tuple(feat.shape)}")
        c = self.cfg
        gh, gw = int(feat.shape[-2]), int(feat.shape[-1])
        rows = self.condition(h_cond, h_where, h_mask)

        mask_feat = self.mask_branch(feat)                       # (1, 8, gh, gw)
        theta, center = self.controller(rows)                    # (169,), (2,)

        if c.disable_rel_coords:
            head_inputs = mask_feat
            rel = None
        else:
            loc = normalized_locations(gh, gw, device=mask_feat.device,
                                       dtype=mask_feat.dtype)    # (2, gh, gw)
            rel = center.reshape(2, 1, 1) - loc                  # (2, gh, gw)
            head_inputs = torch.cat([rel.unsqueeze(0), mask_feat], dim=1)

        weights, biases = parse_dynamic_params(
            theta.reshape(1, -1), c.head_channels, self.weight_nums, self.bias_nums)
        logit_low = mask_heads_forward(head_inputs, weights, biases, num_insts=1)
        logit = aligned_bilinear(logit_low, c.up_factor)          # (1,1,4gh,4gw)
        m_pix = torch.sigmoid(logit)[0, 0]                        # dynamic_mask_head.py:221
        m_low = area_resize(m_pix[None, None], (gh, gw))[0, 0]    # gt_low's operator

        out: dict[str, Any] = {
            "m_low": m_low, "m_pix": m_pix, "logit": logit[0, 0],
            "logit_low": logit_low[0, 0], "mask_feat": mask_feat,
            "theta": theta, "center": center, "rel_coords": rel,
            "grid": (gh, gw), "pix_grid": tuple(m_pix.shape),
        }
        if self.geom is not None:
            out["geom"] = self.geom(rows)
        return out

    # -- record ------------------------------------------------------------
    def n_params(self) -> dict[str, int]:
        def n(m) -> int:
            return 0 if m is None else int(sum(p.numel() for p in m.parameters()))

        return {"mask_branch": n(self.mask_branch), "controller": n(self.controller),
                "span_query": n(self.span_query), "geom_reg": n(self.geom),
                "total": int(sum(p.numel() for p in self.parameters())),
                "trainable": int(sum(p.numel() for p in self.parameters()
                                     if p.requires_grad))}

    def facts(self) -> dict[str, Any]:
        """Goes into ``model.facts()["arm_head"]`` and the run setup.

        Deliberately carries no ``n_stages`` / ``n_refine_layers`` / ``aux_groups``
        key: ``evaluate.deep_supervision_tags`` reads those, and this head has no
        intermediate supervision to promise.
        """
        c = self.cfg
        return {
            "arm": ARM,
            "reference": "CondInst (arXiv 2003.05664) / aim-uofa/AdelaiDet",
            "config": c.to_dict(),
            "weight_nums": list(self.weight_nums),
            "bias_nums": list(self.bias_nums),
            "num_gen_params": int(self.num_gen_params),
            "controller_out": int(self.num_gen_params + 2),
            "rel_coords": not c.disable_rel_coords,
            "upsample": {"operator": "aligned_bilinear (comm.py:23-45)",
                         "factor": c.up_factor,
                         "in_stride": FEAT_STRIDE,
                         "out_stride": int(c.mask_out_stride)},
            "readout_effective": ("xattn_over_where_span" if c.ctrl == "query"
                                  else c.readout_kind or "from --cond-readout"),
            "mask_loss": c.mask_loss,
            "dice_eps": DICE_EPS,
            "norm_caveat": (
                "MASK_BRANCH.NORM = BN is kept verbatim, but this campaign's "
                "forward runs ONE SAMPLE AT A TIME (trainer.compute_micro_batch "
                "-- the H/16 grid varies per image), so the BatchNorm statistics "
                "are computed over H*W of a single image instead of over "
                "IMS_PER_BATCH 16 images. Structural consequence of the "
                "per-sample loop, recorded rather than silently absorbed."),
            "geom_reg": (None if self.geom is None else {
                "weight": float(c.geom_reg_weight),
                "hidden": int(c.geom_hidden),
                "n_out": GEOM_N_OUT,
                "route_classes": list(GEOM_ROUTE_CLASSES),
                "route_loss": c.geom_route_loss,
                "layout": {k: list(v) for k, v in GEOM_LAYOUT.items()},
                "angle_mask_enabled": bool(c.geom_angle_mask_ratio > 0),
                "angle_mask_note": (
                    "D-1 unresolved: the near-isotropic angle columns are NOT "
                    "masked (conservative default); recorded, not silent"),
                "last_layer_zero_init": True,
            }),
            "center_supervision": (None if not c.center_sup else
                                   {"weight": float(c.center_weight),
                                    "target": "gt_pix centroid (NOVEL)"}),
            "params": self.n_params(),
            "step0": (
                "controller weight ~ N(0, 0.01^2), bias 0 -> theta = 0 exactly at "
                "h_cond = 0, dynamic head output 0, sigmoid = 0.5; with a real "
                "h_cond theta is small but non-zero (proposal §3 初始化)"),
        }


# --------------------------------------------------------------------------- #
# geometry-regression targets (ablation row ①)
# --------------------------------------------------------------------------- #
def _gvalue(geometry: Any, name: str, default: float = 0.0) -> float:
    """``canonical_masks.py:92-96``'s own parser -- same lstrip('+'), same
    fallback -- so the regression target is the number the renderer read."""
    try:
        return float(str(dict(geometry).get(name, default)).lstrip("+"))
    except (TypeError, ValueError):
        return default


def _gflipped(geometry: Any) -> float:
    return 1.0 if str(dict(geometry).get("Flipped", "false")).lower().lstrip("+") == "true" \
        else 0.0


def geom_targets(mask_type: str | None, geometry: Any
                 ) -> tuple[list[float], float] | None:
    """``(continuous targets, flipped)`` for one sample, or ``None``.

    Keys and formulas are ``raster_geometry``'s own (``canonical_masks.py:98-122``):
    centre and radii from ``Left/Right/Top/Bottom``, the angle as
    ``sin(2 theta) / cos(2 theta)`` (+-180 deg is the same ellipse), ``Feather``
    after the ``max(f/100, 0.05)`` the renderer applies, and the four
    ``Zero*/Full*`` corners for the ``gradient`` family.  Coordinates are already
    normalised image fractions (``x = xx/width``) and are NOT rescaled.
    """
    if not mask_type or geometry is None:
        return None
    if mask_type == "circulargradient":
        left, right = _gvalue(geometry, "Left"), _gvalue(geometry, "Right")
        top, bottom = _gvalue(geometry, "Top"), _gvalue(geometry, "Bottom")
        angle = math.radians(_gvalue(geometry, "Angle"))
        return ([(left + right) / 2.0, (top + bottom) / 2.0,
                 abs(right - left) / 2.0, abs(bottom - top) / 2.0,
                 math.sin(2 * angle), math.cos(2 * angle),
                 max(_gvalue(geometry, "Feather", 50.0) / 100.0, 0.05)],
                _gflipped(geometry))
    if mask_type == "gradient":
        return ([_gvalue(geometry, "ZeroX"), _gvalue(geometry, "ZeroY"),
                 _gvalue(geometry, "FullX", 1.0), _gvalue(geometry, "FullY")],
                _gflipped(geometry))
    return None


def _route_target(family: str, mask_type: str | None) -> int:
    from .pixgt import mask_type_of

    mt = mask_type or mask_type_of(family)
    if mt in GEOM_ROUTE_CLASSES:
        return GEOM_ROUTE_CLASSES.index(mt)
    return GEOM_ROUTE_CLASSES.index("semantic")


def _geom_loss(head: CondInstHead, out: dict[str, Any], x: Any
               ) -> tuple[torch.Tensor, dict[str, Any]]:
    """``L_geom`` for one sample plus its guard/observation columns.

    * continuous columns: L1
    * ``Flipped``: BCE-with-logits
    * route: 3-way cross-entropy (**NOVEL** -- the proposal registers the route
      logit and its accuracy column but no loss for it; ``--condinst-geom-route-loss
      none`` turns it off and the accuracy column then reads the zero-init argmax)
    * ``semantic`` (and any store miss): the PARAMETER terms are zeroed and the
      sample is counted; the route term still applies, because "semantic" is one
      of the three route classes and is the only label such a sample has.
    """
    g = out["geom"]
    cfg = head.cfg
    dev = g["route"].device
    pg = getattr(x, "pixgt", None)
    mask_type = getattr(pg, "mask_type", None)
    geometry = getattr(pg, "geometry", None)
    tgt = geom_targets(mask_type, geometry)

    zero = g["route"].sum() * 0.0
    total = zero
    stats: dict[str, Any] = {"geom_route_target": _route_target(x.family, mask_type),
                             "geom_has_params": bool(tgt is not None),
                             "geom_mask_type": mask_type or "",
                             "geom_family": x.family}

    if cfg.geom_route_loss == "ce":
        t = torch.tensor([stats["geom_route_target"]], device=dev, dtype=torch.long)
        total = total + F.cross_entropy(g["route"].reshape(1, -1), t)
    stats["geom_route_pred"] = int(g["route"].detach().argmax())

    if tgt is None:
        stats["geom_zeroed"] = 1.0
        return total, stats
    stats["geom_zeroed"] = 0.0

    cont, flipped = tgt
    pred = g[mask_type]
    y = torch.tensor(cont, device=dev, dtype=pred.dtype)
    per_col = (pred - y).abs()
    if cfg.geom_angle_mask_ratio > 0 and mask_type == "circulargradient":
        names = GEOM_LAYOUT[mask_type]
        rx, ry = abs(cont[names.index("rx")]), abs(cont[names.index("ry")])
        ratio = max(rx, ry) / max(min(rx, ry), 1e-6)
        if ratio < float(cfg.geom_angle_mask_ratio):
            keep = torch.ones_like(per_col)
            for nm in ("sin2a", "cos2a"):
                keep[names.index(nm)] = 0.0
            per_col = per_col * keep
            stats["geom_angle_masked"] = 1.0
    total = total + per_col.mean()

    fl_logit = g[f"{mask_type}_flipped"].reshape(())
    fl_t = torch.tensor(float(flipped), device=dev, dtype=fl_logit.dtype)
    total = total + F.binary_cross_entropy_with_logits(fl_logit, fl_t)
    stats["geom_flipped_correct"] = float((fl_logit.detach() > 0).float() == fl_t)
    for nm, v in zip(GEOM_LAYOUT[mask_type], per_col.detach().tolist()):
        stats[f"geom_l1_{nm}"] = float(v)
    return total, stats


def _centroid(field: torch.Tensor) -> torch.Tensor | None:
    """Normalised ``(x, y)`` centre of mass, or ``None`` for an empty field."""
    h, w = int(field.shape[-2]), int(field.shape[-1])
    m = field.clamp_min(0)
    s = m.sum()
    if float(s) <= 1e-8:
        return None
    loc = normalized_locations(h, w, device=field.device, dtype=field.dtype)
    return torch.stack([(m * loc[0]).sum() / s, (m * loc[1]).sum() / s])


# --------------------------------------------------------------------------- #
# runtime assertions
# --------------------------------------------------------------------------- #
def assert_first_step_columns(stats: dict[str, Any], cfg: CondInstConfig | None = None
                              ) -> dict[str, Any]:
    """The proposal's first-row witness: ``steps.jsonl`` line 1 must carry
    ``L_dice`` (``L_bce`` under the loss ablation) and, with the auxiliary head
    on, ``L_geom``.

    Called on the FIRST aggregated micro-batch by the wrapper's seam, i.e. on the
    columns that are actually written, not on the intention to write them.
    """
    cfg = cfg or config()
    want = [f"L_{cfg.mask_loss}"] + (["L_geom"] if cfg.geom_on else [])
    missing = [k for k in want if k not in stats]
    if missing:
        raise AssertionError(
            f"arm {ARM}: the first training step logged no {missing} column "
            f"(present: {sorted(k for k in stats if k.startswith('L_'))}). The "
            "loss the proposal registers never reached steps.jsonl.")
    return {"checked": want, "present": True}


# --------------------------------------------------------------------------- #
# arm hooks (q3vl/whereb/amort/arms.py)
# --------------------------------------------------------------------------- #
def build_head(*, in_dim: int, text_dim: int, args: Any = None,
               cfg: CondInstConfig | None = None, **kw) -> nn.Module:
    if kw:
        raise TypeError(f"unexpected head kwargs {sorted(kw)}")
    cfg = cfg or config()
    set_config(cfg)
    # Deterministic head init WITHOUT consuming the global stream: reseeding in
    # place would shift every later draw (the trainer's sampling order among
    # them) by an amount that depends on this head's parameter count, i.e. on
    # which ablation row is running.
    state = torch.get_rng_state()
    try:
        torch.manual_seed(int(cfg.seed))
        head = CondInstHead(in_dim=in_dim, text_dim=text_dim, cfg=cfg)
    finally:
        torch.set_rng_state(state)
    return head


def head_kwargs_from_args(args: Any) -> dict[str, Any]:
    """The wrapper puts the parsed config in ``ARM_KWARGS``; this only fills in
    the default when the base entry is used directly."""
    return {"cfg": config()}


def forward(model, head: CondInstHead, ctx) -> dict[str, Any]:
    """``AmortModel.forward_geo`` -> the head.  fp32 (D-3: AMP.ENABLED False)."""
    h_cond = None
    if head.cfg.ctrl != "query":
        h_cond = ctx.require_cond(ARM)
    feat = ctx.feat
    with no_autocast(feat.device.type):
        out = head(feat.float(),
                   None if h_cond is None else h_cond.float(),
                   h_where=(None if ctx.h_where is None else ctx.h_where.float()),
                   h_mask=ctx.h_mask)
    out["n_cond_vectors"] = 1 if h_cond is None else int(h_cond.shape[0])
    return out


def compute_loss(model, out: dict[str, Any], x, weights):
    """``L = dice(m_pix, gt_pix)`` (+ the ablation terms), one sample.

    ``condinst.py:161-165`` sums the mask loss into the total with **no
    coefficient**, so the weight here is 1.0.  A foreign sample takes the
    no-instance branch of ``dynamic_mask_head.py:210-216``: a zero-valued term
    built from the tensors that would have carried the gradient.
    """
    from .losses import AmortLoss

    head = model.geo
    cfg = head.cfg
    terms: dict[str, torch.Tensor] = {}
    stats: dict[str, Any] = {"gt_pix_source": getattr(x, "gt_pix_source", "")}
    m_pix = out["m_pix"]
    dummy = out["mask_feat"].sum() * 0 + out["theta"].sum() * 0

    if x.is_fake:
        terms[cfg.mask_loss] = dummy
        total = dummy
        stats["condinst_fake"] = 1.0
    else:
        gt = getattr(x, "gt_pix", None)
        if gt is None:
            raise ValueError(
                f"{x.sample_id}: arm {ARM} supervises on the stride-"
                f"{cfg.mask_out_stride} pixel GT and the batch carried none; the "
                "run must not pass --no-pixgt (pixgt.PixGTProvider is wired by "
                "run_amort_arm for every EPR-018..023 arm)")
        if tuple(gt.shape) != tuple(m_pix.shape):
            raise AssertionError(
                f"{x.sample_id}: gt_pix is {tuple(gt.shape)} but the head emits "
                f"{tuple(m_pix.shape)}; the builder's pixgt_size and the head's "
                "upsample factor disagree")
        if cfg.gt == "cgt" and stats["gt_pix_source"] == "render":
            raise AssertionError(
                f"{x.sample_id}: --condinst-gt cgt but the GT came from the "
                "analytic renderer; run_condinst_arm must forward "
                "--pixgt-source cgt1024")
        gt = gt.to(m_pix.dtype)
        if cfg.mask_loss == "dice":
            loss = dice_coefficient(m_pix.reshape(1, -1), gt.reshape(1, -1)).mean()
        else:
            # sigmoid(logit) == m_pix, so this is BCE on the same quantity, in
            # the numerically stable form
            loss = F.binary_cross_entropy_with_logits(out["logit"], gt)
        terms[cfg.mask_loss] = loss
        total = loss
        stats["condinst_fake"] = 0.0

    if cfg.center_sup and not x.is_fake:
        tgt = _centroid(x.gt_pix.to(m_pix.dtype))
        c_term = ((out["center"] - tgt).abs().mean() if tgt is not None
                  else out["center"].sum() * 0)
        terms["center"] = c_term
        total = total + float(cfg.center_weight) * c_term

    if head.geom is not None:
        if x.is_fake:
            terms["geom"] = out["geom"]["route"].sum() * 0
        else:
            g_total, g_stats = _geom_loss(head, out, x)
            terms["geom"] = g_total
            total = total + float(cfg.geom_reg_weight) * g_total
            stats.update(g_stats)

    return AmortLoss(total=total, terms=terms, stats=stats)


def optimizer_spec(args: Any):
    """SGD 0.01 / momentum 0.9 / wd 1e-4, detectron2's norm-exempt grouping."""
    from .trainer import OptimizerSpec

    c = config()
    return OptimizerSpec(type=c.optimizer, lr=float(c.lr),
                         weight_decay=float(c.weight_decay),
                         momentum=float(c.momentum), nesterov=bool(c.nesterov),
                         grouping=("norm_bias" if c.optimizer == "sgd" else "dim"),
                         norm_weight_decay=float(c.weight_decay_norm))


def scheduler_kwargs(args: Any, total_steps: int) -> dict[str, Any]:
    """detectron2 ``WarmupMultiStepLR``: linear warmup ``1/1000 -> 1``, then the
    ``gamma = 0.1`` ladder at the carried-over milestones.

    The *kind* lives on ``--scheduler``, which this hook cannot set, so it is
    asserted here (same guard as ``segsam.py:1160-1166``): under ``cosine`` the
    ``milestones`` / ``gamma`` returned below are ignored while
    ``warmup_factor`` is not, so a CONDINST run started around
    ``run_condinst_arm.py`` (which pins the kind) would silently train on
    cosine-plus-detectron2-warmup -- a schedule nothing pre-registers, on a
    board that looks identical.
    """
    c = config()
    for name, kind in (("--scheduler", str(getattr(args, "scheduler",
                                                   SCHEDULER_KIND)
                                            or SCHEDULER_KIND)),
                       ("the arm config", str(c.scheduler))):
        if kind != SCHEDULER_KIND:
            raise SystemExit(
                f"arm CONDINST needs --scheduler {SCHEDULER_KIND!r} "
                "(detectron2 WarmupMultiStepLR, defaults.py:531/543/549) but "
                f"{name} says {kind!r}.  run_condinst_arm.py pins it; do not "
                "pass --scheduler by hand.")
    total = int(total_steps or getattr(args, "max_steps", 0) or 0)
    return {"milestones": c.milestones(total), "gamma": float(c.lr_gamma),
            "warmup_steps": c.warmup(total), "warmup_factor": float(c.warmup_factor)}


def builder_kwargs(args: Any) -> dict[str, Any]:
    """The pixel GT is rendered on the head's own output grid."""
    f = config().up_factor
    return {"pixgt_size": (lambda gh, gw: (f * gh, f * gw))}


def readout_spec(args: Any):
    """``--condinst-ctrl`` is an alias of ``--cond-readout`` (proposal 改哪里 ⑨)."""
    from q3vl.whereb.readout import ReadoutSpec

    c = config()
    kind = getattr(args, "cond_readout", "seg_where")
    qtok = int(getattr(args, "readout_qtok", 0) or 0)
    nseg = int(getattr(args, "readout_nseg", 1) or 1)
    if c.readout_kind:
        if kind not in ("seg_where", c.readout_kind):
            raise ValueError(
                f"--condinst-ctrl {c.ctrl} maps to --cond-readout "
                f"{c.readout_kind}, but --cond-readout {kind} was also passed; "
                "they are the same knob, so pass one of them")
        kind, qtok, nseg = c.readout_kind, 0, 1
    return ReadoutSpec(kind=kind, qtok=qtok, nseg=nseg)


# --------------------------------------------------------------------------- #
# evaluation columns
# --------------------------------------------------------------------------- #
def per_sample_row(model, out: dict[str, Any], x) -> dict[str, Any]:
    """The pre-registered ``condinst_pix_readout`` measurements, per sample.

    Measured on the head's **native stride-4 field** ``m_pix`` against ``gt_pix``
    at the same resolution -- the three-column set the criteria section fixes:
    soft-IoU (min/max), matched-area top-k IoU, grid boundary F1 (tol = 1 cell),
    the centre-prior column on the same support, and the random top-k floor.
    No per-image min-max anywhere: the fields go in as they come out.
    """
    from q3vl.whereb.metrics import (grid_boundary_f1, gt_area_k, hard_iou,
                                     soft_iou_value, topk_mask)

    from .evaluate import center_prior_unit, random_floor

    gt = getattr(x, "gt_pix", None)
    if gt is None or "m_pix" not in out:
        return {}
    m = out["m_pix"].detach().float()
    g = gt.detach().float()
    if tuple(m.shape) != tuple(g.shape):
        return {"condinst_pix_shape_mismatch": True}
    ph, pw = int(m.shape[-2]), int(m.shape[-1])
    k = gt_area_k(g)
    m_k, g_k = topk_mask(m, k), topk_mask(g, k)
    cp = center_prior_unit(ph, pw, device=m.device)
    cp_k = topk_mask(cp, k)
    row = {
        "condinst_pix_soft_iou": soft_iou_value(m, g),
        "condinst_pix_topk_iou": hard_iou(m_k, g_k),
        "condinst_pix_boundary_f1": grid_boundary_f1(m_k, g_k, tol_cells=1),
        "condinst_pix_center_soft_iou": soft_iou_value(cp, g),
        "condinst_pix_center_topk_iou": hard_iou(cp_k, g_k),
        "condinst_pix_center_boundary_f1": grid_boundary_f1(cp_k, g_k, tol_cells=1),
        "condinst_pix_random_floor": random_floor(float((g > 0.5).float().mean())),
        "condinst_pix_grid": [ph, pw],
        "condinst_gt_pix_source": getattr(x, "gt_pix_source", ""),
        "condinst_center_x": float(out["center"][0]),
        "condinst_center_y": float(out["center"][1]),
    }
    if "geom" in out:
        g_out = out["geom"]
        pred = int(g_out["route"].detach().argmax())
        pg = getattr(x, "pixgt", None)
        target = _route_target(x.family, getattr(pg, "mask_type", None))
        row.update(condinst_geom_route_pred=GEOM_ROUTE_CLASSES[pred],
                   condinst_geom_route_target=GEOM_ROUTE_CLASSES[target],
                   condinst_geom_route_correct=bool(pred == target))
        tgt = geom_targets(getattr(pg, "mask_type", None), getattr(pg, "geometry", None))
        if tgt is not None:
            mt = pg.mask_type
            cont, flipped = tgt
            err = (g_out[mt].detach().float()
                   - torch.tensor(cont, device=m.device)).abs().tolist()
            for nm, v in zip(GEOM_LAYOUT[mt], err):
                row[f"condinst_geom_l1_{nm}"] = float(v)
            row["condinst_geom_flipped_correct"] = bool(
                float(g_out[f"{mt}_flipped"].detach() > 0) == float(flipped))
    return row


def _agg_col(vals: Sequence[float]) -> dict[str, Any]:
    v = [float(x) for x in vals if x is not None and np.isfinite(float(x))]
    if not v:
        return {"n": 0}
    return {"n": len(v), "median": float(np.median(v)), "mean": float(np.mean(v)),
            "min": float(np.min(v)), "max": float(np.max(v))}


def criteria_columns(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """``board["criteria_columns"]`` entries.

    ``assert_criteria_ran`` refuses a board whose ``condinst_pix_readout`` carries
    ``n = 0``; the auxiliary-head column is asserted here, because
    ``arms.ARM_CRITERIA`` is a frozen table and cannot depend on a flag.
    """
    live = [r for r in rows if r.get("condinst_pix_topk_iou") is not None]
    sources: dict[str, int] = {}
    for r in live:
        s = str(r.get("condinst_gt_pix_source") or "")
        sources[s] = sources.get(s, 0) + 1
    n_render = sources.get("render", 0)
    col: dict[str, Any] = {
        **_agg_col([r["condinst_pix_topk_iou"] for r in live]),
        "soft_iou": _agg_col([r["condinst_pix_soft_iou"] for r in live]),
        "boundary_f1": _agg_col([r["condinst_pix_boundary_f1"] for r in live]),
        "center_prior_topk_iou": _agg_col(
            [r["condinst_pix_center_topk_iou"] for r in live]),
        "center_prior_soft_iou": _agg_col(
            [r["condinst_pix_center_soft_iou"] for r in live]),
        "center_prior_boundary_f1": _agg_col(
            [r["condinst_pix_center_boundary_f1"] for r in live]),
        "random_floor": _agg_col([r["condinst_pix_random_floor"] for r in live]),
        "gt_pix_source_counts": dict(sorted(sources.items())),
        "gt_pix_fallback_n": len(live) - n_render,
        "gt_pix_fallback_frac": ((len(live) - n_render) / len(live)) if live else None,
        "note": ("matched-area top-k IoU on the head's native stride-4 field "
                 "m_pix against gt_pix at the same resolution, with the min/max "
                 "soft-IoU, the tol=1 grid boundary F1, the centre prior on the "
                 "same support and the a/(2-a) random floor beside it "
                 "(EPR-023 §3 改哪里 ⑦)"),
    }
    out: dict[str, Any] = {"condinst_pix_readout": col}

    routed = [r for r in rows if r.get("condinst_geom_route_correct") is not None]
    if routed:
        by_family: dict[str, Any] = {}
        for fam in ("radial", "band", "linear", "semantic"):
            sub = [r for r in routed if r.get("family") == fam]
            by_family[fam] = {
                "n": len(sub),
                "accuracy": (float(np.mean([r["condinst_geom_route_correct"]
                                            for r in sub])) if sub else None)}
        flip = [r["condinst_geom_flipped_correct"] for r in routed
                if r.get("condinst_geom_flipped_correct") is not None]
        l1_cols = sorted({k for r in routed for k in r
                          if k.startswith("condinst_geom_l1_")})
        out["geom_reg_route"] = {
            "n": len(routed),
            "accuracy": float(np.mean([r["condinst_geom_route_correct"]
                                       for r in routed])),
            "by_family": by_family,
            "flipped_accuracy": (float(np.mean(flip)) if flip else None),
            "component_l1_median": {
                c.replace("condinst_geom_l1_", ""):
                    _agg_col([r[c] for r in routed if c in r]).get("median")
                for c in l1_cols},
            "route_loss": config().geom_route_loss,
            "angle_mask_enabled": bool(config().geom_angle_mask_ratio > 0),
            "note": ("argmax of the auxiliary head's 3-way route logit vs the "
                     "construction-side family; replaces the cls head the "
                     "faithful recipe does not have (EPR-023 消融行①)"),
        }
    if config().geom_on and not out.get("geom_reg_route", {}).get("n"):
        raise AssertionError(
            f"arm {ARM} runs with --geom-reg-weight {config().geom_reg_weight} "
            "and the board carries 0 values for 'geom_reg_route'; refusing to "
            "publish an ablation row that cannot adjudicate itself")
    return out


def train_stats(out: dict[str, Any], x) -> dict[str, Any]:
    """Extra per-sample columns on the trainer's own rows."""
    row = {"condinst_center_x": float(out["center"][0].detach()),
           "condinst_center_y": float(out["center"][1].detach()),
           "condinst_pix_grid": list(out.get("pix_grid", ())),
           "condinst_gt_pix_source": getattr(x, "gt_pix_source", "")}
    return row
