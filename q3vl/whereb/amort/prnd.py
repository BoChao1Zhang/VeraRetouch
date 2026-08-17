"""EPR-020 -- PointRend point-sampled segmentation head (arm ``PRND``).

A faithful port of PointRend (arXiv 1912.08193) onto the frozen Qwen3-VL
``F_pre`` grid: coarse head + point head + training-time point sampling +
inference-time subdivision.  Every structural constant, initialisation, loss
form, optimiser and schedule value is the reference's own; the eight places
where the reference has nothing to copy are marked ``NOVEL N1..N8`` with the
proposal's reason, and nothing else deviates.

Reference files, pulled with ``curl`` from the GitHub ``main`` raw endpoints on
2026-08-14 and checked line by line (``nl -ba``):

    d2 = facebookresearch/detectron2
    d2 projects/PointRend/point_rend/point_features.py   19-42, 63-116, 119-143
    d2 projects/PointRend/point_rend/point_head.py       73-75, 94-129
    d2 projects/PointRend/point_rend/mask_head.py        43-49, 118
    d2 projects/PointRend/point_rend/semantic_seg.py     68-135  (subdivision)
    d2 detectron2/modeling/meta_arch/semantic_seg.py     193-215, 255-266
    d2 detectron2/config/defaults.py                     411-416, 526-551
    d2 projects/PointRend/point_rend/config.py           32-48
    d2 .../SemanticSegmentation/Base-PointRend-Semantic-FPN.yaml   4-17
    d2 .../pointrend_semantic_R_101_FPN_1x_cityscapes.yaml         10, 16-19
    fvcore fvcore/nn/weight_init.py                      8-39
    m2f  mask2former/modeling/criterion.py               21-40, 48-65, 73-87
    m2f  configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml 24-25

Nothing from ST_LANG enters this file: no ``ConvTower`` / ``FiLM`` /
``CondEncoder`` / UNIQ query stack import, by construction (§1 of the proposal).

Loss (main arm, §2.1) -- two terms, both weight 1.0, no dice / IoU / SDF::

    L = BCEwithLogits(up16(L_c), alpha_hi).mean()
      + BCEwithLogits(l_p, PS(alpha_hi, P)).mean()

Ablation row 2 (``--prnd-point-loss m2f``) replaces the point term with
Mask2Former's ``5.0 * sigmoid_CE + 5.0 * dice`` -- that row **does** contain
dice, by the user's 2026-08-14 instruction to port the reference recipes
verbatim; it is recorded as such in ``facts()``.

RNG discipline (proposal §3 "初始化" row, N1 lesson): the training point sampler
draws from a **dedicated** ``torch.Generator`` seeded ``cfg.seed + 314``.  It
never touches the global stream -- one extra global draw per step would shift
the sample order and the fake coin flips of every step after it and silently
break the U4 step-matched comparison with the control rows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from q3vl.where.upsample import area_resize
from q3vl.whereb.fields import no_autocast

from .losses import AmortLoss

__all__ = [
    "ARM", "CRITERIA", "DEFAULTS", "OPTIONS",
    "point_sample", "calculate_uncertainty",
    "get_uncertain_point_coords_with_randomness",
    "get_uncertain_point_coords_on_grid",
    "c2_msra_fill", "c2_xavier_fill",
    "CoarseHead", "StandardPointHead", "PointRendHead",
    "pointrend_losses", "sigmoid_ce_loss", "dice_loss",
    "build_head", "forward", "compute_loss",
    "add_arguments", "configure", "options", "setup_record",
    "loss_form", "loss_preregistration",
    "head_kwargs_from_args", "optimizer_spec", "scheduler_kwargs",
    "builder_kwargs", "per_sample_row", "criteria_columns", "train_stats",
    "assert_publishable", "install_publication_assert", "RUN_CONTEXT",
    "FIRST_LOSS_COLUMNS", "first_step_row", "resolve_steps_row",
    "is_publication_board",
]

# --------------------------------------------------------------------------- #
# registry contract (q3vl/whereb/amort/arms.py)
# --------------------------------------------------------------------------- #
ARM = "PRND"
CRITERIA = ("prnd_point_readout",)

#: ``SOLVER.STEPS (40000, 55000) / MAX_ITER 65000`` and ``WARMUP_ITERS 1000``
#: (cityscapes yaml :17-18, defaults.py:550) carried over as FRACTIONS -- the
#: campaign runs 1200 steps (U4).  At 1200 these round to 738 / 1015 / 18,
#: which are the literal defaults written in the proposal's entry row.
MILESTONE_FRACS: tuple[float, float] = (40000.0 / 65000.0, 55000.0 / 65000.0)
WARMUP_FRAC: float = 1000.0 / 65000.0
REFERENCE_MAX_ITER = 65000
#: the horizon those literal defaults were computed for
PREREGISTERED_TOTAL_STEPS = 1200

#: ``--prnd-*`` defaults = the reference values (§3.1 table).  Not a config
#: object: the hooks below are called with three *different* namespaces by the
#: entry script, so the arm keeps one module-level record instead (the seam
#: ``uniq4.VARIANT4`` already uses, covered by the sha256 source freeze).
DEFAULTS: dict[str, Any] = {
    "train_points": 1024,          # Base-PointRend-Semantic-FPN.yaml:13
    "oversample": 3.0,             # config.py:35
    "importance": 0.75,            # config.py:38
    "fc_dim": 256,                 # config.py:43 / yaml:10
    "num_fc": 3,                   # config.py:44 / yaml:11
    "coarse_dim": 128,             # defaults.py:411 (SEM_SEG_HEAD.CONVS_DIM)
    "coarse_pred_each_layer": False,   # yaml:17 (semantic config)
    "subdiv_steps": 4,             # N4: stride 16 -> pixel GT needs 4 doublings
    "subdiv_points": 8192,         # yaml:15
    "point_loss": "bce",           # point_head.py:73-75 (no dice)
    "gt": "maskhi",                # spec-5 .maskhi, short side 512
    "no_subdivision": False,
    "optimizer": "sgd",            # d2 solver/build.py:139
    "lr": 0.01,                    # cityscapes yaml:16
    "momentum": 0.9,               # defaults.py:534
    "wd": 1e-4,                    # defaults.py:538
    "warmup_iters": 18,            # 1000/65000 of 1200
    "milestones": "738,1015",      # (40000, 55000)/65000 of 1200
    "gamma": 0.1,                  # defaults.py:543
}

#: filled by ``run_prnd_arm.py`` before it delegates; empty = every default
OPTIONS: dict[str, Any] = {}

#: ``--prnd-gt`` -> the shared ``--pixgt-source`` value it requires
GT_TO_PIXGT_SOURCE: dict[str, str] = {
    "maskhi": "maskhi", "cgt1024": "cgt1024", "analytic": "render",
}

#: Mask2Former ablation row 2 weights (maskformer2_R50_bs16_50ep.yaml:24-25)
M2F_MASK_WEIGHT = 5.0
M2F_DICE_WEIGHT = 5.0

#: PointRend's own norm: ``get_norm("GN", c)`` == ``nn.GroupNorm(32, c)``
#: (defaults.py:415; detectron2/layers/batch_norm.py:189)
GN_GROUPS = 32

#: the point sampler's dedicated stream (proposal §3 "RNG 纪律")
POINT_SEED_OFFSET = 314


def options() -> dict[str, Any]:
    """The effective ``--prnd-*`` record: defaults overlaid with ``OPTIONS``."""
    return {**DEFAULTS, **OPTIONS}


def configure(ns: Any) -> dict[str, Any]:
    """``argparse`` namespace (``--prnd-*``) -> ``OPTIONS``; returns the record."""
    for k in DEFAULTS:
        v = getattr(ns, f"prnd_{k}", None)
        if v is not None:
            OPTIONS[k] = v
    return options()


def parse_milestones(spec: Any) -> list[int]:
    if isinstance(spec, (list, tuple)):
        return sorted(int(x) for x in spec)
    return sorted(int(x) for x in str(spec).replace(" ", "").split(",") if x)


# --------------------------------------------------------------------------- #
# fvcore weight init (fvcore/nn/weight_init.py:8-39), copied verbatim
# --------------------------------------------------------------------------- #
def c2_xavier_fill(module: nn.Module) -> None:
    nn.init.kaiming_uniform_(module.weight, a=1)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


def c2_msra_fill(module: nn.Module) -> None:
    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


# --------------------------------------------------------------------------- #
# point_features.py -- ported verbatim, with the RNG made explicit
# --------------------------------------------------------------------------- #
def point_sample(input: torch.Tensor, point_coords: torch.Tensor,
                 **kwargs) -> torch.Tensor:
    """``point_features.py:19-42``.

    ``input`` ``(N, C, H, W)``, ``point_coords`` ``(N, P, 2)`` in ``[0,1]^2``
    given as ``(x, y)``; returns ``(N, C, P)``.  The ``2x - 1`` remap and
    ``align_corners=False`` at every call site are the whole alignment contract:
    getting either wrong shifts the field by half a cell, which on the 32x48
    grid is 8 px.  ``padding_mode`` is left at ``grid_sample``'s default
    (``"zeros"``) -- the reference passes nothing either.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


def calculate_uncertainty(logits: torch.Tensor) -> torch.Tensor:
    """``mask_head.py:43-49`` -- the class-agnostic branch: ``-|logit|``.

    N5: the L1 distance to the 0.5 decision surface.  The GT here is a
    continuous alpha (``canonical_masks.py:120``'s ``a^2(3-2a)`` transition
    band), so ``sigmoid(logit) = 0.5`` need not coincide with the GT's
    ``alpha = 0.5`` iso-contour.  The formula is NOT changed for that; the
    reference's own class-agnostic uncertainty is used as written.
    """
    if logits.shape[1] != 1:
        raise ValueError(
            f"the class-agnostic uncertainty takes a single channel, got "
            f"{logits.shape[1]}")
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


def get_uncertain_point_coords_with_randomness(
    coarse_logits: torch.Tensor,
    uncertainty_func,
    num_points: int,
    oversample_ratio: float,
    importance_sample_ratio: float,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """``point_features.py:63-116``, verbatim but with an explicit ``generator``.

    Order is not interchangeable (the reference's own comment, L92-98): the
    uncertainty is computed on the **sampled point logits**, never on the coarse
    map before sampling.  A point interpolated between two coarse cells holding
    -1 and +1 has logit 0 and therefore uncertainty 0; computing on the map
    first would record it as -1 and the importance sampling would look at the
    wrong places.

    ``generator`` is the only deviation and it is a discipline requirement, not
    a recipe change: the draws must not come out of the global RNG stream.
    """
    if oversample_ratio < 1:
        raise ValueError("oversample_ratio must be >= 1")
    if not (0.0 <= importance_sample_ratio <= 1.0):
        raise ValueError("importance_sample_ratio must be in [0, 1]")
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2,
                              device=coarse_logits.device,
                              dtype=coarse_logits.dtype, generator=generator)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long,
                                       device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2
    )
    if num_random_points > 0:
        point_coords = torch.cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 2,
                           device=coarse_logits.device,
                           dtype=coarse_logits.dtype, generator=generator),
            ],
            dim=1,
        )
    return point_coords


def get_uncertain_point_coords_on_grid(uncertainty_map: torch.Tensor,
                                       num_points: int):
    """``point_features.py:119-143``, verbatim."""
    R, _, H, W = uncertainty_map.shape
    h_step = 1.0 / float(H)
    w_step = 1.0 / float(W)

    num_points = min(H * W, num_points)
    point_indices = torch.topk(uncertainty_map.view(R, H * W), k=num_points,
                               dim=1)[1]
    point_coords = torch.zeros(R, num_points, 2, dtype=torch.float,
                               device=uncertainty_map.device)
    point_coords[:, :, 0] = w_step / 2.0 + (point_indices % W).to(torch.float) * w_step
    point_coords[:, :, 1] = h_step / 2.0 + (point_indices // W).to(torch.float) * h_step
    return point_indices, point_coords


# --------------------------------------------------------------------------- #
# heads
# --------------------------------------------------------------------------- #
class CoarseHead(nn.Module):
    """``SemSegFPNHead`` at a single scale (``semantic_seg.py:193-215``).

    N3: ``head_length = max(1, log2(stride) - log2(common_stride))``.  The coarse
    field is produced on ``F_pre`` itself, so ``stride == common_stride == 16``
    and the reference formula gives ``max(1, 0) = 1`` conv and **no** upsample
    branch (that branch is gated on ``stride != common_stride``).  This is their
    formula evaluated here, not a structure chosen here.

    N2 (the only language seam): ``h_cond`` -> ``Linear(2560 -> coarse_dim)``
    -> broadcast to ``coarse_dim`` channels -> concatenated to ``F_pre`` on the
    channel axis **before** the 3x3 conv.  Not a dot product, not FiLM.  The
    point head sees the instruction only through ``coarse_features``, which is
    PointRend's own information flow (``point_head.py:124``).
    """

    def __init__(self, in_dim: int = 1024, text_dim: int = 2560,
                 coarse_dim: int = 128, num_classes: int = 1,
                 gn_groups: int = GN_GROUPS):
        super().__init__()
        self.in_dim = int(in_dim)
        self.text_dim = int(text_dim)
        self.coarse_dim = int(coarse_dim)
        self.num_classes = int(num_classes)
        # N2 projection.  `c2_xavier_fill` is what PointRend uses for its FC
        # layers (mask_head.py:118).
        self.cond_proj = nn.Linear(self.text_dim, self.coarse_dim)
        c2_xavier_fill(self.cond_proj)
        # `bias=not norm` (semantic_seg.py:202) -- norm is "GN", so bias=False
        self.conv = nn.Conv2d(self.in_dim + self.coarse_dim, self.coarse_dim,
                              kernel_size=3, stride=1, padding=1, bias=False)
        self.norm = nn.GroupNorm(int(gn_groups), self.coarse_dim)
        c2_msra_fill(self.conv)
        self.predictor = nn.Conv2d(self.coarse_dim, self.num_classes,
                                   kernel_size=1, stride=1, padding=0)
        # NOT zero-initialised: PointRend msra-fills the predictor
        # (semantic_seg.py:215), so step 0 is not a constant field.  Recorded in
        # run_setup so it is not mistaken for a bug later.
        c2_msra_fill(self.predictor)

    def cond_channels(self, h_cond: torch.Tensor) -> torch.Tensor:
        """``(K, text_dim)`` -> ``(coarse_dim,)``.

        ``K == 1`` is the main arm (a single ``<seg_where>`` row).  For the
        readout-ablation rows with ``K > 1`` (``qtok`` K_q = 4 / 8, ``nseg``
        K = 2 / 4) the proposal's pre-registered NOVEL default applies: every
        row goes through the SAME ``Linear`` and the results are averaged.
        """
        if h_cond.dim() == 1:
            h_cond = h_cond[None]
        if h_cond.dim() != 2:
            raise ValueError(f"h_cond must be (K, D), got {tuple(h_cond.shape)}")
        return self.cond_proj(h_cond).mean(dim=0)

    def forward(self, feat: torch.Tensor, h_cond: torch.Tensor) -> torch.Tensor:
        """``(1, in_dim, gh, gw)`` + ``(K, text_dim)`` -> ``(1, 1, gh, gw)``."""
        if feat.dim() != 4:
            raise ValueError(f"feat must be (N, C, H, W), got {tuple(feat.shape)}")
        g = self.cond_channels(h_cond)                       # (coarse_dim,)
        n, _, h, w = feat.shape
        gmap = g.reshape(1, self.coarse_dim, 1, 1).expand(n, self.coarse_dim, h, w)
        x = torch.cat([feat, gmap], dim=1)
        z = F.relu(self.norm(self.conv(x)))
        return self.predictor(z)


class StandardPointHead(nn.Module):
    """``point_head.py:94-129``, verbatim.

    N1: PointRend reads ``input_shape.channels`` (``point_head.py:101``), so the
    1024-channel single-scale ``F_pre`` goes in unprojected -- the reference
    interface takes the channel count as given and no projection layer is added.
    """

    def __init__(self, input_channels: int = 1024, num_classes: int = 1,
                 fc_dim: int = 256, num_fc: int = 3,
                 coarse_pred_each_layer: bool = False,
                 cls_agnostic_mask: bool = True):
        super().__init__()
        self.coarse_pred_each_layer = bool(coarse_pred_each_layer)
        fc_dim_in = int(input_channels) + int(num_classes)
        self.fc_layers = []
        for k in range(int(num_fc)):
            fc = nn.Conv1d(fc_dim_in, int(fc_dim), kernel_size=1, stride=1,
                           padding=0, bias=True)
            self.add_module("fc{}".format(k + 1), fc)
            self.fc_layers.append(fc)
            fc_dim_in = int(fc_dim)
            fc_dim_in += int(num_classes) if self.coarse_pred_each_layer else 0

        num_mask_classes = 1 if cls_agnostic_mask else int(num_classes)
        self.predictor = nn.Conv1d(fc_dim_in, num_mask_classes, kernel_size=1,
                                   stride=1, padding=0)

        for layer in self.fc_layers:
            c2_msra_fill(layer)
        # normal distribution initialisation for the mask prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    def forward(self, fine_grained_features: torch.Tensor,
                coarse_features: torch.Tensor) -> torch.Tensor:
        x = torch.cat((fine_grained_features, coarse_features), dim=1)
        for layer in self.fc_layers:
            x = F.relu(layer(x))
            if self.coarse_pred_each_layer:
                x = torch.cat((x, coarse_features), dim=1)
        return self.predictor(x)


class PointRendHead(nn.Module):
    """Coarse head + point head + sampling + subdivision, for one sample."""

    def __init__(
        self,
        *,
        in_dim: int = 1024,
        text_dim: int = 2560,
        coarse_dim: int = 128,
        num_classes: int = 1,
        fc_dim: int = 256,
        num_fc: int = 3,
        coarse_pred_each_layer: bool = False,
        train_points: int = 1024,
        oversample: float = 3.0,
        importance: float = 0.75,
        subdiv_steps: int = 4,
        subdiv_points: int = 8192,
        no_subdivision: bool = False,
        point_loss: str = "bce",
        gt: str = "maskhi",
        point_seed: int = 20260810 + POINT_SEED_OFFSET,
    ):
        super().__init__()
        if point_loss not in ("bce", "m2f"):
            raise ValueError(f"--prnd-point-loss must be bce|m2f, got {point_loss!r}")
        if gt not in GT_TO_PIXGT_SOURCE:
            raise ValueError(
                f"--prnd-gt must be one of {sorted(GT_TO_PIXGT_SOURCE)}, got {gt!r}")
        self.in_dim = int(in_dim)
        self.text_dim = int(text_dim)
        self.num_classes = int(num_classes)
        self.train_points = int(train_points)
        self.oversample = float(oversample)
        self.importance = float(importance)
        self.subdiv_steps = int(subdiv_steps)
        self.subdiv_points = int(subdiv_points)
        self.no_subdivision = bool(no_subdivision)
        self.point_loss = str(point_loss)
        self.gt = str(gt)
        self.point_seed = int(point_seed)

        self.coarse = CoarseHead(in_dim=in_dim, text_dim=text_dim,
                                 coarse_dim=coarse_dim, num_classes=num_classes)
        self.point_head = StandardPointHead(
            input_channels=in_dim, num_classes=num_classes, fc_dim=fc_dim,
            num_fc=num_fc, coarse_pred_each_layer=coarse_pred_each_layer,
            cls_agnostic_mask=True)
        #: one dedicated generator per device; NEVER the global stream
        self._generators: dict[str, torch.Generator] = {}
        self.n_point_draws = 0

    # -- RNG ---------------------------------------------------------------
    def generator(self, device: torch.device) -> torch.Generator:
        key = f"{device.type}:{device.index if device.index is not None else 0}"
        g = self._generators.get(key)
        if g is None:
            g = torch.Generator(device=device)
            g.manual_seed(self.point_seed)
            self._generators[key] = g
        return g

    def reset_point_rng(self) -> None:
        """Re-seed every point generator (tests; never called by the loop)."""
        for g in self._generators.values():
            g.manual_seed(self.point_seed)

    # -- the three pieces ---------------------------------------------------
    def coarse_logits(self, feat: torch.Tensor, h_cond: torch.Tensor) -> torch.Tensor:
        return self.coarse(feat, h_cond)

    def sample_train_points(self, coarse_logits: torch.Tensor) -> torch.Tensor:
        """``(1, N, 2)`` training coordinates, under ``no_grad`` as in the
        reference (``semantic_seg.py:74-81``)."""
        with torch.no_grad():
            p = get_uncertain_point_coords_with_randomness(
                coarse_logits, calculate_uncertainty, self.train_points,
                self.oversample, self.importance,
                generator=self.generator(coarse_logits.device))
        self.n_point_draws += 1
        return p

    def point_logits(self, feat: torch.Tensor, coarse_logits: torch.Tensor,
                     point_coords: torch.Tensor) -> torch.Tensor:
        """``semantic_seg.py:82-91`` -- fine-grained from the feature map, coarse
        from the **original** coarse logits, concatenated by the point head."""
        fine = point_sample(feat, point_coords, align_corners=False)
        coarse = point_sample(coarse_logits, point_coords, align_corners=False)
        return self.point_head(fine, coarse)

    def subdivision(self, feat: torch.Tensor, coarse_logits: torch.Tensor,
                    steps: int | None = None) -> torch.Tensor:
        """``semantic_seg.py:106-134``.  ``(1, 1, gh, gw)`` -> ``(1, 1, H, W)``
        with ``H = gh * 2**steps``.  The coarse features are always sampled from
        the ORIGINAL ``coarse_logits``, never from the partially refined map."""
        s = self.subdiv_steps if steps is None else int(steps)
        logits = coarse_logits.clone()
        for _ in range(s):
            logits = F.interpolate(logits, scale_factor=2, mode="bilinear",
                                   align_corners=False)
            uncertainty_map = calculate_uncertainty(logits)
            point_indices, point_coords = get_uncertain_point_coords_on_grid(
                uncertainty_map, self.subdiv_points)
            fine = point_sample(feat, point_coords, align_corners=False)
            coarse = point_sample(coarse_logits, point_coords, align_corners=False)
            pl = self.point_head(fine, coarse)
            N, C, H, W = logits.shape
            point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
            logits = (
                logits.reshape(N, C, H * W)
                .scatter_(2, point_indices, pl)
                .view(N, C, H, W)
            )
        return logits

    # -- one sample ---------------------------------------------------------
    def forward(self, feat: torch.Tensor, h_cond: torch.Tensor, *,
                training: bool) -> dict[str, Any]:
        """``feat`` ``(1, in_dim, gh, gw)`` float32, ``h_cond`` ``(K, text_dim)``."""
        gh, gw = int(feat.shape[-2]), int(feat.shape[-1])
        l_c = self.coarse_logits(feat, h_cond)                 # (1, 1, gh, gw)
        out: dict[str, Any] = {"coarse_logits": l_c, "grid": (gh, gw)}
        if training:
            # No subdivision while training: the reference runs it only in
            # inference (semantic_seg.py:106).  `m_low` here therefore comes
            # from the coarse field and feeds only the shared shape assertion
            # and the early-warning columns -- recorded as `m_low_source`.
            coords = self.sample_train_points(l_c)
            out["point_coords"] = coords
            out["point_logits"] = self.point_logits(feat, l_c, coords)
            out["m_low"] = torch.sigmoid(l_c)[0, 0]
            out["m_low_source"] = "coarse"
            out["m_hi"] = None
            out["m_hi_source"] = "none"
            return out
        if self.no_subdivision:
            # Ablation row 4: the point head is training-only; the deployed
            # field is the coarse one (`m_low = sigmoid(L_c)`).  The pixel-level
            # diagnostic column still needs a pixel field, and the reference's
            # own dense convention supplies one without inventing anything:
            # `up(L_c)` to the input resolution (semantic_seg.py:257-262).
            up = 2 ** self.subdiv_steps
            m_hi = torch.sigmoid(F.interpolate(
                l_c, scale_factor=up, mode="bilinear", align_corners=False))
            out["m_low"] = torch.sigmoid(l_c)[0, 0]
            out["m_low_source"] = "coarse"
            out["m_hi"] = m_hi
            out["m_hi_source"] = "coarse_up"
            return out
        logits_hi = self.subdivision(feat, l_c)
        m_hi = torch.sigmoid(logits_hi)                        # (1, 1, H, W)
        # `m_low` uses the SAME operator `gt_low` does (data.py:685-686 ->
        # q3vl/where/upsample.py:54-62), so the headline criterion keeps
        # measuring the two with one ruler.
        out["m_low"] = area_resize(m_hi, (gh, gw))[0, 0]
        out["m_low_source"] = "subdivision"
        out["m_hi"] = m_hi
        out["m_hi_source"] = "subdivision"
        out["logits_hi"] = logits_hi
        return out

    # -- reporting ----------------------------------------------------------
    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def facts(self) -> dict[str, Any]:
        # NOTE: deliberately carries none of `n_stages` / `n_refine_layers` /
        # `aux_groups`, so `evaluate.deep_supervision_tags` stays empty for this
        # arm -- PointRend has no auxiliary supervision branches.
        return {
            "head": "PointRendHead",
            "reference": "PointRend (arXiv 1912.08193), detectron2 PointRend project",
            "in_dim": self.in_dim,
            "text_dim": self.text_dim,
            "coarse_dim": self.coarse.coarse_dim,
            "gn_groups": GN_GROUPS,
            "num_classes": self.num_classes,
            "fc_dim": int(self.point_head.fc_layers[0].out_channels),
            "num_fc": len(self.point_head.fc_layers),
            "coarse_pred_each_layer": bool(self.point_head.coarse_pred_each_layer),
            "train_num_points": self.train_points,
            "oversample_ratio": self.oversample,
            "importance_sample_ratio": self.importance,
            "subdivision_steps": self.subdiv_steps,
            "subdivision_num_points": self.subdiv_points,
            "no_subdivision": self.no_subdivision,
            "point_loss": self.point_loss,
            "gt_source_flag": self.gt,
            "point_rng": {
                "seed": self.point_seed,
                "offset_from_cfg_seed": POINT_SEED_OFFSET,
                "dedicated_generator": True,
                "note": ("the training point draws never touch the global RNG "
                         "stream; a stray global draw per step would shift the "
                         "sample order and the fake coin flips of every later "
                         "step and void the U4 step-matched comparison"),
            },
            "coarse_predictor_init": "c2_msra_fill (NOT zero) -- step 0 coarse "
                                     "field is not a constant field, by the "
                                     "reference (semantic_seg.py:215)",
            "point_predictor_init": "N(0, 0.001^2), bias 0 -- step 0 point "
                                    "logits ~ 0, sigmoid ~ 0.5 "
                                    "(point_head.py:119-121)",
            "n_trainable_params": self.n_trainable(),
            "n_point_draws": self.n_point_draws,
        }


# --------------------------------------------------------------------------- #
# losses
# --------------------------------------------------------------------------- #
def sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor,
                    num_masks: float) -> torch.Tensor:
    """m2f ``criterion.py:48-65`` (ablation row 2 only)."""
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.mean(1).sum() / num_masks


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor,
              num_masks: float) -> torch.Tensor:
    """m2f ``criterion.py:21-40`` (ablation row 2 only).

    The ``+1`` smoothing on both numerator and denominator is the reference's;
    it is also why an all-zero target (the ``is_fake`` policy) cannot divide by
    zero here.
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def pointrend_losses(
    coarse_logits: torch.Tensor,      # (1, 1, gh, gw)
    point_logits: torch.Tensor,       # (1, 1, N)
    alpha_hi: torch.Tensor,           # (H, W) in [0, 1]
    point_targets: torch.Tensor,      # (1, N) or (1, 1, N) in [0, 1]
    *,
    point_loss: str = "bce",
    loss_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """The two-term criterion of §2.1.  Returns ``{"coarse", "point"[, "dice"]}``.

    ``L_coarse`` follows ``SemSegFPNHead.losses`` (``semantic_seg.py:255-266``):
    upsample the prediction to the input resolution FIRST, then take the loss
    with ``reduction="mean"``, times ``LOSS_WEIGHT`` (1.0, ``defaults.py:416``).
    The binary head substitutes ``binary_cross_entropy_with_logits`` for their
    multi-class ``cross_entropy`` -- which is the reference's own binary form
    (``point_head.py:73-75``), not a new loss.

    Neither term divides by a GT area, so an empty (``is_fake``) target is
    finite and differentiable rather than a zero-division.
    """
    if alpha_hi.dim() != 2:
        raise ValueError(f"alpha_hi must be (H, W), got {tuple(alpha_hi.shape)}")
    gh, gw = int(coarse_logits.shape[-2]), int(coarse_logits.shape[-1])
    H, W = int(alpha_hi.shape[-2]), int(alpha_hi.shape[-1])
    if H % gh or W % gw or (H // gh) != (W // gw):
        raise AssertionError(
            f"the pixel GT {H}x{W} is not an integer, isotropic multiple of the "
            f"coarse grid {gh}x{gw}: spec-5 guarantees H = 16*gh and W = 16*gw "
            "(q3vl/where/fpre.py:45-49), so a mismatch here means the two fields "
            "no longer share one [0,1]^2 image domain")
    up = H // gh
    dense = F.interpolate(coarse_logits.float(), scale_factor=up, mode="bilinear",
                          align_corners=False)
    l_coarse = F.binary_cross_entropy_with_logits(
        dense.reshape(-1), alpha_hi.reshape(-1).float(), reduction="mean"
    ) * float(loss_weight)

    pl = point_logits.reshape(1, -1)
    pt = point_targets.reshape(1, -1).float()
    terms = {"coarse": l_coarse}
    if point_loss == "bce":
        # point_head.py:73-75 -- BCE with logits, reduction="mean", NO dice
        terms["point"] = F.binary_cross_entropy_with_logits(
            pl, pt, reduction="mean")
    else:
        # ablation row 2: Mask2Former's point recipe, dice included by the
        # user's 2026-08-14 instruction to port the reference verbatim
        terms["point"] = M2F_MASK_WEIGHT * sigmoid_ce_loss(pl, pt, 1.0)
        terms["dice"] = M2F_DICE_WEIGHT * dice_loss(pl, pt, 1.0)
    return terms


# --------------------------------------------------------------------------- #
# GT plumbing
# --------------------------------------------------------------------------- #
def pixel_gt(x: Any) -> tuple[torch.Tensor, str]:
    """``(alpha_hi, source)`` for one sample.

    Prefers the :mod:`q3vl.whereb.amort.pixgt` provider's field (whose source is
    the ``--prnd-gt`` flag, carried through ``--pixgt-source``); falls back to
    the builder's ``gt_hi`` (the published ``.maskhi``) and says so.
    ``is_fake`` (foreign instruction, p = 0.15) zeroes the whole field -- the
    pre-registered conservative policy of §2.1, counted, never silent.
    """
    a = getattr(x, "gt_pix", None)
    src = str(getattr(x, "gt_pix_source", "") or "")
    if a is None:
        a = getattr(x, "gt_hi", None)
        src = "gt_hi_maskhi512"
    if a is None:
        raise ValueError(
            "arm PRND supervises at pixel resolution and this sample carries "
            "neither gt_pix nor gt_hi.  Pass --want-hi (and do not pass "
            "--no-pixgt) -- run_prnd_arm.py does both.")
    a = a.float()
    if a.dim() == 4:
        a = a[0, 0]
    elif a.dim() == 3:
        a = a[0]
    if getattr(x, "is_fake", False):
        a = torch.zeros_like(a)
        src = f"{src}+fake_zeroed"
    return a, src


def gt_at_points(x: Any, alpha_hi: torch.Tensor,
                 coords: torch.Tensor) -> torch.Tensor:
    """``(1, N)`` point labels at ``coords`` ``(1, N, 2)``.

    N6: bilinear, not ``mode="nearest"``.  The reference's semantic config uses
    nearest because its GT is an integer class id map (``semantic_seg.py:96``);
    here the GT is a continuous alpha and the point sigmoid CE takes float
    targets directly, which is Mask2Former's口径 for the same mechanism
    (``criterion.py:171-175``).

    On the ``analytic`` GT row the closed form is evaluated at the sampled
    coordinates instead of interpolating a raster -- that is the whole point of
    that row; the provider's :meth:`PixGT.points` owns that dispatch.
    """
    if getattr(x, "is_fake", False):
        return torch.zeros(1, coords.shape[1], device=coords.device,
                           dtype=alpha_hi.dtype)
    pg = getattr(x, "pixgt", None)
    if pg is not None and getattr(pg, "analytic", False) \
            and getattr(pg, "mask_type", None) and getattr(pg, "geometry", None):
        return pg.points(coords[0]).reshape(1, -1).to(alpha_hi.dtype)
    return point_sample(alpha_hi[None, None], coords,
                        align_corners=False).reshape(1, -1)


# --------------------------------------------------------------------------- #
# registry hooks
# --------------------------------------------------------------------------- #
def build_head(*, in_dim: int, text_dim: int, args: Any = None, **kw):
    o = options()
    seed = int(getattr(args, "seed", 20260810) or 20260810)
    params = dict(
        in_dim=in_dim, text_dim=text_dim,
        coarse_dim=int(o["coarse_dim"]),
        fc_dim=int(o["fc_dim"]), num_fc=int(o["num_fc"]),
        coarse_pred_each_layer=bool(o["coarse_pred_each_layer"]),
        train_points=int(o["train_points"]), oversample=float(o["oversample"]),
        importance=float(o["importance"]),
        subdiv_steps=int(o["subdiv_steps"]), subdiv_points=int(o["subdiv_points"]),
        no_subdivision=bool(o["no_subdivision"]), point_loss=str(o["point_loss"]),
        gt=str(o["gt"]), point_seed=seed + POINT_SEED_OFFSET,
    )
    params.update(kw)
    return PointRendHead(**params)


def head_kwargs_from_args(args: Any) -> dict[str, Any]:
    """Nothing extra: :func:`build_head` reads :func:`options` itself, so the
    three different namespaces the entry script hands the hooks cannot disagree
    about what this arm was configured with."""
    return {}


def forward(model, head, ctx) -> dict[str, Any]:
    """Registry hook -- one sample through the head."""
    h_cond = ctx.require_cond(ARM)                       # (K, 2560)
    feat = ctx.feat
    if ctx.grid_h and ctx.grid_w:
        if (int(feat.shape[-2]), int(feat.shape[-1])) != (int(ctx.grid_h),
                                                          int(ctx.grid_w)):
            raise AssertionError(
                f"feat grid {tuple(feat.shape[-2:])} != declared grid "
                f"{(ctx.grid_h, ctx.grid_w)}")
    # fp32, autocast off: the point sampling and the two losses are computed in
    # full precision, exactly as the live arms do for their field stage
    # (model.py:289) and as AMP.ENABLED=False in the reference (defaults.py:595).
    with no_autocast(feat.device.type):
        res = head(feat.float(), h_cond.float(), training=bool(model.training))
    out: dict[str, Any] = {"m_low": res["m_low"], "prnd": res, "params": {}}
    if res.get("m_hi") is not None:
        out["m_hi"] = res["m_hi"]
    return out


#: The ``L_*`` columns this arm's loss emitted on its FIRST training
#: micro-batch, captured in-process.
#:
#: ``steps.jsonl`` is the artifact the arm pre-registers, but the trainer's
#: handle is block-buffered and only flushed every ``log_every`` steps
#: (``trainer.py:607-608``), so a quick eval that lands on a non-flush step can
#: read an EMPTY file -- indistinguishable, from disk alone, from "the loss
#: never ran".  This witness is the same claim one link earlier in the chain:
#: ``losses.aggregate`` prefixes ``L_`` to exactly these keys
#: (``losses.py:541-542``) and the trainer splices its output into the row
#: verbatim (``trainer.py:600-605``), so a column here is a column there.
FIRST_LOSS_COLUMNS: dict[str, float] = {}


def _witness(terms: dict[str, Any]) -> None:
    """Record the first micro-batch's ``L_*`` columns.  ``terms`` unchanged."""
    if not FIRST_LOSS_COLUMNS:
        FIRST_LOSS_COLUMNS.update({f"L_{k}": float(v.detach())
                                   for k, v in terms.items()})


def compute_loss(model, out: dict[str, Any], x, weights) -> AmortLoss:
    """Registry hook -- the two-term PointRend criterion for one sample."""
    res = out["prnd"]
    l_c = res["coarse_logits"]
    coords = res.get("point_coords")
    l_p = res.get("point_logits")
    if coords is None or l_p is None:
        raise AssertionError(
            "arm PRND: compute_loss was handed a forward that produced no "
            "training points -- the head emits them only while model.training "
            "is True, so this is the 'stuck in eval mode' failure, not a "
            "configuration the loss can absorb")
    head = model.geo
    with no_autocast(l_c.device.type):
        alpha_hi, gt_src = pixel_gt(x)
        alpha_hi = alpha_hi.to(l_c.device)
        a_p = gt_at_points(x, alpha_hi, coords)
        terms = pointrend_losses(l_c, l_p, alpha_hi, a_p,
                                 point_loss=head.point_loss)
        total = terms["coarse"] + terms["point"]
        if "dice" in terms:
            total = total + terms["dice"]
        with torch.no_grad():
            # `point/accuracy` (point_head.py:67-71).  The reference's GT is
            # binary; the continuous alpha here is compared at its own 0.5
            # iso-level, which is the same decision surface the logit sign is.
            acc = ((l_p.reshape(-1) > 0.0) == (a_p.reshape(-1) > 0.5)).float().mean()
            n_pts = float(l_p.reshape(-1).numel())
            n_imp = float(int(head.importance * head.train_points))
    # These are NOT loss terms.  `aggregate` (losses.py:539-542) is the only
    # per-step channel into steps.jsonl and it aggregates `terms` only, so the
    # pre-registered observation columns (c)/(d) ride there and are labelled
    # `diag_*` to keep them distinguishable from `L_coarse` / `L_point`.
    diag = {
        "diag_point_accuracy": acc.detach(),
        "diag_point_n": torch.as_tensor(n_pts, device=l_c.device),
        "diag_point_n_importance": torch.as_tensor(n_imp, device=l_c.device),
        "diag_point_gt_mean": a_p.detach().float().mean(),
    }
    res["diag"] = {"point_accuracy": float(acc), "point_n": n_pts,
                   "point_n_importance": n_imp, "gt_pix_source": gt_src}
    _witness({**terms, **diag})
    return AmortLoss(total=total, terms={**terms, **diag},
                     stats={"prnd_point_accuracy": float(acc),
                            "prnd_point_n": n_pts})


def train_stats(out: dict[str, Any], x) -> dict[str, Any]:
    d = (out.get("prnd", {}) or {}).get("diag", {}) or {}
    if not d:
        return {}
    return {"prnd_point_accuracy": d.get("point_accuracy"),
            "prnd_point_n": d.get("point_n"),
            "prnd_gt_pix_source": d.get("gt_pix_source")}


def optimizer_spec(args: Any):
    """SGD 0.01 / momentum 0.9 / wd 1e-4 with detectron2's norm-exempt grouping.

    ``--prnd-optimizer adamw`` (ablation row 5) returns ``None`` = the campaign's
    own AdamW 3e-4 / wd 0.01 / dim>1 grouping, i.e. exactly the row the proposal
    pre-registers as "swap the optimiser back and change nothing else".
    """
    from .trainer import OptimizerSpec

    o = options()
    if str(o["optimizer"]) != "sgd":
        return None
    return OptimizerSpec(
        type="sgd",
        lr=float(o["lr"]),
        weight_decay=float(o["wd"]),
        momentum=float(o["momentum"]),
        nesterov=False,                 # defaults.py:536
        grouping="norm_bias",           # WEIGHT_DECAY_NORM applies to norms only
        norm_weight_decay=0.0,          # defaults.py:541
    )


def schedule_for(total_steps: int) -> dict[str, Any]:
    """``WarmupMultiStepLR`` carried onto a ``total_steps`` horizon.

    The literal defaults (warmup 18, milestones 738/1015) are the reference's
    fractions evaluated at the pre-registered 1200-step horizon; if a run uses a
    different horizon and has not overridden the flags, the same fractions are
    re-evaluated there rather than silently keeping absolute steps that no
    longer mean what they meant.
    """
    from q3vl.where.calibrate import scale_milestones

    o = options()
    total = int(total_steps or PREREGISTERED_TOTAL_STEPS)
    ms_default = "milestones" not in OPTIONS or \
        str(OPTIONS.get("milestones")) == str(DEFAULTS["milestones"])
    wu_default = "warmup_iters" not in OPTIONS or \
        int(OPTIONS.get("warmup_iters")) == int(DEFAULTS["warmup_iters"])
    if ms_default and total != PREREGISTERED_TOTAL_STEPS:
        milestones = scale_milestones(MILESTONE_FRACS, total)
        ms_origin = f"fractions {MILESTONE_FRACS} of {total}"
    else:
        milestones = parse_milestones(o["milestones"])
        ms_origin = ("literal --prnd-milestones" if not ms_default
                     else f"literal default at the pre-registered {total} steps")
    if wu_default and total != PREREGISTERED_TOTAL_STEPS:
        warmup = max(1, int(round(WARMUP_FRAC * total)))
        wu_origin = f"fraction {WARMUP_FRAC:.6f} of {total}"
    else:
        warmup = int(o["warmup_iters"])
        wu_origin = ("literal --prnd-warmup-iters" if not wu_default
                     else f"literal default at the pre-registered {total} steps")
    return {"milestones": milestones, "gamma": float(o["gamma"]),
            "warmup_steps": warmup,
            "warmup_factor": 1.0 / 1000.0,     # defaults.py:549, "linear"
            "_milestones_origin": ms_origin, "_warmup_origin": wu_origin,
            "_total_steps": total}


def scheduler_kwargs(args: Any, total_steps: int) -> dict[str, Any] | None:
    if str(options()["optimizer"]) != "sgd":
        return None                      # row 5: the campaign cosine, untouched
    kw = schedule_for(total_steps)
    return {k: v for k, v in kw.items() if not k.startswith("_")}


def builder_kwargs(args: Any) -> dict[str, Any]:
    """``want_hi`` + the pixel-GT resolution, and the ``--prnd-gt`` guard.

    ``pixgt_size`` is ``(16*gh, 16*gw)`` -- the spec-5 pixel grid, and the
    resolution 4 subdivision doublings land on exactly (``2**4 == 16``).  The
    shared default is 4x, which would make the coarse dense loss compare a 16x
    upsample against a 4x GT.
    """
    o = options()
    want = GT_TO_PIXGT_SOURCE[str(o["gt"])]
    got = str(getattr(args, "pixgt_source", want))
    if got != want:
        raise SystemExit(
            f"--prnd-gt {o['gt']!r} needs --pixgt-source {want!r} but the run "
            f"was given {got!r}.  Refusing to start: the run record would say "
            f"{o['gt']!r} while the supervision came from {got!r}.  "
            "run_prnd_arm.py sets this for you; do not pass --pixgt-source by "
            "hand.")
    if getattr(args, "no_pixgt", False):
        raise SystemExit(
            "arm PRND supervises at pixel resolution through the pixgt "
            "provider; --no-pixgt would leave --prnd-gt unenforced.")
    # the two extra publication assertions of §3 -- installed here so they hold
    # on every launch path, not only under run_prnd_arm.py.  `args` is the
    # parsed run_amort_arm namespace, so this is also where the assertion learns
    # the run directory it must read steps.jsonl from and whether the run is an
    # `--eval-only` re-score: the `evaluate.evaluate_arm` call site passes
    # neither, and reading "not passed" as "not computed" is what took the
    # 2026-08-15 smoke down.
    root = getattr(args, "out_root", None)
    steps_path = None
    if root is not None:
        run_name = getattr(args, "run_name", None) or \
            f"amort_{getattr(args, 'arm', ARM)}"           # run_amort_arm.py:537
        steps_path = Path(root) / run_name / "steps.jsonl"
    # `None` (the attribute is absent) leaves whatever the wrapper already
    # recorded alone; a real `--eval-only` value overwrites it.
    eval_only = getattr(args, "eval_only", None)
    if install_publication_assert(steps_path=steps_path,
                                  eval_only=(None if eval_only is None
                                             else bool(eval_only))):
        print("PRND: publication assertions installed (steps.jsonl first row "
              "must carry L_coarse / L_point; the published board must carry "
              f"headline_normal_only); steps.jsonl = {steps_path}", flush=True)
    up = 2 ** int(o["subdiv_steps"])
    return {"want_hi": True,
            "pixgt_size": (lambda gh, gw, _u=up: (_u * gh, _u * gw))}


# --------------------------------------------------------------------------- #
# evaluation: the pre-registered criterion column
# --------------------------------------------------------------------------- #
def per_sample_row(model, out: dict[str, Any], x) -> dict[str, Any]:
    """The arm's per-sample diagnostic columns (evaluator seam).

    Failure here must never take the board down (D-20 lesson 6) and must never
    vanish silently either, so an error is recorded in its own column.
    """
    res = out.get("prnd", {}) or {}
    row: dict[str, Any] = {
        "prnd_m_low_source": res.get("m_low_source"),
        "prnd_m_hi_source": res.get("m_hi_source"),
        "prnd_gt_pix_source": str(getattr(x, "gt_pix_source", "") or ""),
        "prnd_pix_soft_iou": None,
        "prnd_transition_abs_err": None,
    }
    m_hi = res.get("m_hi")
    if m_hi is None:
        row["prnd_pix_error"] = "no pixel field (m_hi is None)"
        return row
    try:
        from q3vl.whereb.metrics import soft_iou_value

        alpha, _src = pixel_gt(x)
        alpha = alpha.detach().to(m_hi.device)
        m_hi = m_hi.detach()
        pred = m_hi.reshape(alpha.shape) if m_hi.numel() == alpha.numel() \
            else m_hi[0, 0]
        if pred.shape != alpha.shape:
            row["prnd_pix_error"] = (f"pixel field {tuple(pred.shape)} vs GT "
                                     f"{tuple(alpha.shape)}")
            return row
        row["prnd_pix_soft_iou"] = soft_iou_value(pred, alpha)
        band = (alpha > 0.05) & (alpha < 0.95)
        if bool(band.any()):
            row["prnd_transition_abs_err"] = float(
                (pred[band] - alpha[band]).abs().mean())
        row["prnd_pix_gt_area_frac"] = float((alpha > 0.5).float().mean())
    except Exception as exc:                                   # noqa: BLE001
        row["prnd_pix_error"] = repr(exc)
    return row


def criteria_columns(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """``prnd_point_readout`` -- the pre-registered column (§3.2).

    Built from the SAME rows the headline is built from.  Its ``n`` is the count
    of pixel-level soft-IoU values; ``assert_criteria_ran`` refuses a board with
    ``n = 0``, which is the pre-registered "no pixel field, no publication".

    Contents, all of them mandatory by the criteria red line:
    pixel soft-IoU (min/max form) + matched-area top-k IoU + GRID boundary F1
    (tol = 1 cell; the pixel-level 3px form is banned) + the centre-prior column
    on the same support + the ``a/(2-a)`` random top-k floor, plus the paired
    delta against the centre prior with its sign-flip permutation p.
    """
    from .evaluate import _agg
    from q3vl.whereb.metrics import paired_delta

    live = [r for r in rows if not r.get("uncovered") and not r.get("is_fake")]
    pix = [r for r in live if r.get("prnd_pix_soft_iou") is not None]
    have_grid = [r for r in live if r.get("hard_iou") is not None
                 and r.get("center_prior_hard_iou") is not None]

    def _counts(key: str) -> dict[str, int]:
        c: dict[str, int] = {}
        for r in live:
            v = str(r.get(key) or "")
            if v:
                c[v] = c.get(v, 0) + 1
        return dict(sorted(c.items()))

    col: dict[str, Any] = {
        **_agg(r["prnd_pix_soft_iou"] for r in pix),
        "quantity": "pixel-resolution soft-IoU (min/max) of the subdivision "
                    "field against the pixel GT",
        "n_rows": len(live),
        "n_pix_errors": sum(1 for r in live if r.get("prnd_pix_error")),
        "transition_band_abs_err": _agg(r.get("prnd_transition_abs_err")
                                        for r in live),
        "topk_iou": _agg(r["hard_iou"] for r in have_grid),
        "grid_boundary_f1": _agg(r["grid_boundary_f1"] for r in have_grid),
        "center_prior_topk_iou": _agg(r["center_prior_hard_iou"]
                                      for r in have_grid),
        "center_prior_boundary_f1": _agg(r["center_prior_boundary_f1"]
                                         for r in have_grid),
        "center_prior_soft_iou": _agg(r["center_prior_soft_iou"]
                                      for r in have_grid),
        "random_floor": _agg(r["random_floor"] for r in have_grid),
        "m_hi_sources": _counts("prnd_m_hi_source"),
        "m_low_sources": _counts("prnd_m_low_source"),
        "gt_pix_sources": _counts("prnd_gt_pix_source"),
        "note": ("EPR-020 pre-registered column.  AUC is banned; the pixel-level "
                 "3px boundary F1 is banned as a criterion, so the boundary "
                 "column here is the GRID-level tol=1 form on matched-area "
                 "top-k masks."),
    }
    if have_grid:
        col["paired_delta_vs_center_prior"] = {
            **paired_delta([r["hard_iou"] for r in have_grid],
                           [r["center_prior_hard_iou"] for r in have_grid]),
            "n_pairs": len(have_grid)}
        col["paired_delta_vs_random_floor"] = {
            **paired_delta([r["hard_iou"] for r in have_grid],
                           [r["random_floor"] for r in have_grid]),
            "n_pairs": len(have_grid)}
    by_family: dict[str, Any] = {}
    for fam in sorted({str(r.get("family")) for r in live}):
        sub = [r for r in pix if str(r.get("family")) == fam]
        gsub = [r for r in have_grid if str(r.get("family")) == fam]
        by_family[fam] = {"n": len(gsub),
                          "pix_soft_iou": _agg(r["prnd_pix_soft_iou"] for r in sub),
                          "topk_iou": _agg(r["hard_iou"] for r in gsub),
                          "grid_boundary_f1": _agg(r["grid_boundary_f1"]
                                                   for r in gsub)}
    col["by_family"] = by_family
    return {"prnd_point_readout": col}


# --------------------------------------------------------------------------- #
# the two extra runtime assertions of §3 (entry row, items (i) and (ii))
# --------------------------------------------------------------------------- #
#: Run-level facts the assertion needs and the shared call site does not pass.
#:
#: ``evaluate.evaluate_arm`` (``evaluate.py:660``) calls
#: ``assert_criteria_ran(board, arm)`` -- two positional arguments and nothing
#: else -- so the run directory and ``--eval-only`` have to reach the assertion
#: some other way.  :func:`install_publication_assert` fills this in from
#: whichever caller knows them (``builder_kwargs`` has the parsed ``args``; the
#: entry wrapper has its own argv peek), and the installed closure reads it at
#: call time so a later, better-informed install still lands.
RUN_CONTEXT: dict[str, Any] = {"steps_path": None, "eval_only": False}


def first_step_row(path: Any) -> dict[str, Any] | None:
    """The FIRST row of a ``steps.jsonl``, or ``None`` when there is none yet.

    Same reader as ``run_amort_arm._first_step_row``; duplicated here because
    the assertion has to work at the ``evaluate_arm`` call site too, and that
    one imports nothing from the entry script.
    """
    try:
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    return json.loads(line)
    except (OSError, ValueError):
        return None
    return None


def resolve_steps_row(steps_row: Any = None, *, steps_path: Any = None
                      ) -> tuple[dict[str, Any] | None, str]:
    """``(row, source)`` -- the first-step row to check, most authoritative first.

    There are TWO call sites of ``evaluate.assert_criteria_ran`` and they do not
    agree on their arguments, which is what killed the 2026-08-15 PRND smoke:

    * ``run_amort_arm._finish_board`` (``run_amort_arm.py:1045-1053``) reads
      ``steps.jsonl`` itself and passes the row -> source ``caller``;
    * ``evaluate.evaluate_arm`` (``evaluate.py:660``) passes ``board`` and
      ``arm`` and NOTHING ELSE -- and it is the call site that gates
      ``metrics.json`` (written three lines below it) and the one every quick
      eval goes through, so the check has to work there.  It reads the artifact
      off disk instead.

    Third fallback: :data:`FIRST_LOSS_COLUMNS`, for a quick eval that lands
    before the trainer flushed its write buffer.  ``(None, "unavailable")`` --
    nothing from the caller, nothing on disk and no loss call in this process --
    is a failure, not a pass; :func:`assert_publishable` raises on it.
    """
    if steps_row is not None:
        return dict(steps_row), "caller"
    if steps_path is not None:
        row = first_step_row(steps_path)
        if row is not None:
            return row, f"steps.jsonl ({steps_path})"
    if FIRST_LOSS_COLUMNS:
        return dict(FIRST_LOSS_COLUMNS), "in-process first-micro-batch witness"
    return None, "unavailable"


def is_publication_board(board: dict[str, Any]) -> bool:
    """Is this the board that gets published, or an interim one?

    The published board is the full six-context one built by
    ``run_amort_arm`` at ``run_amort_arm.py:1099`` / ``:1160`` and written to
    ``eval_final/metrics.json``; it is the only one that carries the
    ``generated`` context.  ``evaluate_arm(quick=True)`` fixes the contexts to
    ``("gt", "shuffled", "antonym")`` (``evaluate.py:519-520``) and the holdout
    call passes ``("gt", "shuffled")`` (``run_amort_arm.py:1173-1175``);
    neither is published and neither is a subset the reporting convention is
    defined on.
    """
    return "generated" in (board.get("contexts") or {})


def assert_publishable(board: dict[str, Any], *, steps_row: Any = None,
                       eval_only: bool = False, steps_path: Any = None
                       ) -> dict[str, Any]:
    """Refuse a PRND board that cannot show its own experiment ran.

    (i) the FIRST row of ``steps.jsonl`` must already carry ``L_coarse`` and
        ``L_point``.  First row, not last: a term that is missing is missing
        from step one, and "defined but not wired" has cost this campaign three
        times.  The row comes from :func:`resolve_steps_row`, so the check runs
        at BOTH call sites of ``evaluate.assert_criteria_ran`` -- passing no row
        is not a way through, only a different place to look.
    (ii) the board must carry ``headline_normal_only`` -- the reporting
        convention is normal-only and the pooled figure may not stand in for it.
        Hard on the published board.  On an interim board (quick eval, holdout)
        the key exists only when that subset happened to contain a
        ``winner_confidence == "normal"`` row (``evaluate.py:482-483``), so its
        absence is recorded together with the subset's own
        ``winner_confidence`` histogram -- and an interim board that DOES carry
        normal rows without the key still raises, because then the two disagree.

    Wired in front of :func:`q3vl.whereb.amort.evaluate.assert_criteria_ran`,
    i.e. before ``metrics.json`` is written at either call site.
    """
    rep: dict[str, Any] = {"arm": ARM}
    want = ("L_coarse", "L_point")
    if eval_only:
        rep["steps_columns"] = {"skipped": "eval_only (no training steps)"}
    else:
        row, source = resolve_steps_row(steps_row, steps_path=steps_path)
        if row is None:
            raise AssertionError(
                f"arm PRND pre-registers {list(want)} in steps.jsonl and no "
                "first row is available anywhere: the caller passed none, "
                f"{steps_path or '<no steps.jsonl path given>'} carries no "
                "row, and this arm's loss has not run in this process.  "
                "Refusing to publish a board that cannot show its loss ran.")
        missing = [c for c in want if c not in row]
        if missing:
            raise AssertionError(
                f"arm PRND: the first steps.jsonl row [{source}] carries no "
                f"{missing}; the two-term criterion never reached the loss "
                f"(columns present: "
                f"{sorted(k for k in row if k.startswith('L_'))})")
        rep["steps_columns"] = {c: row[c] for c in sorted(row)
                                if c.startswith("L_")}
        rep["steps_columns_source"] = source

    main = board.get("main_context")
    ctx = (board.get("contexts", {}) or {}).get(main, {}) or {}
    hn = ctx.get("headline_normal_only")
    if hn:
        rep["headline_normal_only"] = {
            "n": hn.get("n"), "topk_iou_median": hn.get("topk_iou_median")}
        return rep
    conf = ((ctx.get("strata") or {}).get("winner_confidence") or {})
    n_normal = int((conf.get("normal") or {}).get("n", 0))
    if is_publication_board(board) or n_normal:
        raise AssertionError(
            f"arm PRND: the board's main context {main!r} carries no "
            "'headline_normal_only'; the headline convention is normal-only "
            "(winner_confidence == 'normal') and the pooled figure may not be "
            f"published in its place (mixing costs ~0.031).  "
            f"winner_confidence strata: "
            f"{ {k: v.get('n') for k, v in conf.items()} }")
    rep["headline_normal_only"] = {
        "skipped": "interim board (not the published six-context board) whose "
                   "sample carries no winner_confidence=='normal' row -- a "
                   "quick eval takes --quick-eval-limit samples and the key is "
                   "only built when that subset has normal rows "
                   "(evaluate.py:482-483); the published board still asserts it",
        "contexts": sorted(board.get("contexts") or {}),
        "winner_confidence_strata": {k: v.get("n") for k, v in conf.items()},
    }
    return rep


def install_publication_assert(*, steps_path: Any = None,
                               eval_only: bool | None = None) -> bool:
    """Put :func:`assert_publishable` in front of the shared
    ``evaluate.assert_criteria_ran``.  Returns whether it installed.

    ``run_amort_arm._finish_board`` does ``from ... import assert_criteria_ran``
    INSIDE the function, so replacing the module attribute reaches it -- and it
    runs before ``metrics.json`` is written, which is what "refuse to publish"
    has to mean.  ``evaluate.py`` is a shared file with no per-arm hook for
    these two checks, so this is the seam (the same one ``run_uniq4b_arm.py``
    uses for its swaps).

    Called from :func:`builder_kwargs` rather than only from the entry wrapper:
    a runtime assertion that depends on which script the operator typed is not a
    runtime assertion.  ``builder_kwargs`` is the one hook every launch path
    runs, it runs before a single step, and it is handed the parsed ``args`` --
    which is where ``steps_path`` / ``eval_only`` come from.  Idempotent, and a
    no-op for every other arm; ``steps_path`` and ``eval_only`` are recorded
    whenever they are given, so a second, better-informed caller still lands.
    """
    import q3vl.whereb.amort.evaluate as _ev

    if steps_path is not None:
        RUN_CONTEXT["steps_path"] = steps_path
    if eval_only is not None:
        RUN_CONTEXT["eval_only"] = bool(eval_only)

    base = _ev.assert_criteria_ran
    if getattr(base, "_prnd_wrapped", False):
        return False

    def _assert_with_prnd(board, arm, *, head_facts=None, steps_row=None):
        report = base(board, arm, head_facts=head_facts, steps_row=steps_row)
        if arm == ARM:
            # `--eval-only` re-scores a checkpoint without training, so this run
            # has no steps.jsonl of its own.  `_finish_board` says so in the
            # board; the `evaluate_arm` call site has not built that key yet,
            # which is why the argv fact is carried in RUN_CONTEXT as well.
            skipped = bool((board.get("deep_supervision_check") or {}).get("skipped")
                           or board.get("eval_only")
                           or RUN_CONTEXT.get("eval_only"))
            report["prnd_publication"] = assert_publishable(
                board, steps_row=steps_row, eval_only=skipped,
                steps_path=RUN_CONTEXT.get("steps_path"))
        return report

    _assert_with_prnd._prnd_wrapped = True          # type: ignore[attr-defined]
    _assert_with_prnd._prnd_base = base             # type: ignore[attr-defined]
    _ev.assert_criteria_ran = _assert_with_prnd
    return True


# --------------------------------------------------------------------------- #
# entry-script surface
# --------------------------------------------------------------------------- #
def add_arguments(ap) -> None:
    """Every ``--prnd-*`` flag of the proposal's entry row; defaults = §3.1."""
    g = ap.add_argument_group("EPR-020 PointRend (--arm PRND)")
    g.add_argument("--prnd-train-points", type=int, default=None,
                   help=f"TRAIN_NUM_POINTS (default {DEFAULTS['train_points']}; "
                        "ablation row 1: 196 / 2048; row 2: 12544)")
    g.add_argument("--prnd-oversample", type=float, default=None,
                   help=f"OVERSAMPLE_RATIO k (default {DEFAULTS['oversample']})")
    g.add_argument("--prnd-importance", type=float, default=None,
                   help=f"IMPORTANCE_SAMPLE_RATIO beta (default "
                        f"{DEFAULTS['importance']})")
    g.add_argument("--prnd-fc-dim", type=int, default=None,
                   help=f"POINT_HEAD.FC_DIM (default {DEFAULTS['fc_dim']})")
    g.add_argument("--prnd-num-fc", type=int, default=None,
                   help=f"POINT_HEAD.NUM_FC (default {DEFAULTS['num_fc']})")
    g.add_argument("--prnd-coarse-dim", type=int, default=None,
                   help=f"SEM_SEG_HEAD.CONVS_DIM (default "
                        f"{DEFAULTS['coarse_dim']})")
    g.add_argument("--prnd-coarse-pred-each-layer", action="store_true",
                   default=None,
                   help="COARSE_PRED_EACH_LAYER (default OFF = the semantic "
                        "config's False; the repo default True is the instance "
                        "config's)")
    g.add_argument("--prnd-subdiv-steps", type=int, default=None,
                   help=f"SUBDIVISION_STEPS (default {DEFAULTS['subdiv_steps']} "
                        "= stride 16 -> the pixel GT)")
    g.add_argument("--prnd-subdiv-points", type=int, default=None,
                   help=f"SUBDIVISION_NUM_POINTS (default "
                        f"{DEFAULTS['subdiv_points']})")
    g.add_argument("--prnd-point-loss", default=None, choices=["bce", "m2f"],
                   help="bce = PointRend's own point loss (no dice); m2f = "
                        "ablation row 2, 5.0*sigmoid_CE + 5.0*dice")
    g.add_argument("--prnd-gt", default=None,
                   choices=sorted(GT_TO_PIXGT_SOURCE),
                   help=f"pixel-GT source (default {DEFAULTS['gt']!r}); sets "
                        "--pixgt-source accordingly")
    g.add_argument("--prnd-no-subdivision", action="store_true", default=None,
                   help="ablation row 4: inference emits the coarse field only")
    g.add_argument("--prnd-optimizer", default=None, choices=["sgd", "adamw"],
                   help="sgd = the reference recipe; adamw = ablation row 5 "
                        "(the campaign's AdamW 3e-4 / wd 0.01 / cosine)")
    g.add_argument("--prnd-lr", type=float, default=None,
                   help=f"SOLVER.BASE_LR (default {DEFAULTS['lr']})")
    g.add_argument("--prnd-momentum", type=float, default=None,
                   help=f"SOLVER.MOMENTUM (default {DEFAULTS['momentum']})")
    g.add_argument("--prnd-wd", type=float, default=None,
                   help=f"SOLVER.WEIGHT_DECAY (default {DEFAULTS['wd']}; "
                        "WEIGHT_DECAY_NORM = 0.0 is applied by the grouping)")
    g.add_argument("--prnd-warmup-iters", type=int, default=None,
                   help=f"WARMUP_ITERS carried to 1200 steps (default "
                        f"{DEFAULTS['warmup_iters']})")
    g.add_argument("--prnd-milestones", default=None,
                   help=f"SOLVER.STEPS carried to 1200 steps (default "
                        f"{DEFAULTS['milestones']!r})")
    g.add_argument("--prnd-gamma", type=float, default=None,
                   help=f"SOLVER.GAMMA (default {DEFAULTS['gamma']})")


def loss_form() -> dict[str, Any]:
    """The ``loss_preregistration`` record for this arm."""
    o = options()
    if str(o["point_loss"]) == "bce":
        form = ("L = 1.0*BCE(up16(coarse), alpha_hi) + "
                "1.0*BCE(points, PS(alpha_hi))")
        dice = False
        note = ("PointRend's semantic recipe is exactly two terms: the coarse "
                "head's own dense loss and one point loss.  No dice, no IoU, "
                "no SDF/area/sep/fake/cls/sel.")
    else:
        form = ("L = 1.0*BCE(up16(coarse), alpha_hi) + "
                "5.0*sigmoid_CE(points, PS(alpha_hi)) + "
                "5.0*dice(points, PS(alpha_hi))")
        dice = True
        note = ("ablation row 2: the Mask2Former point recipe, ported verbatim "
                "per the user's 2026-08-14 instruction; this row DOES contain "
                "dice.")
    return {"arm": ARM, "form": form, "dice_in_loss": dice,
            "dice_as_target": dice,
            "iou_as_field_target": False, "iou_as_selection_target": False,
            "point_loss": str(o["point_loss"]), "note": note}


def loss_preregistration(args: Any = None) -> dict[str, Any]:
    """``arms.RUN_REQUIRED_HOOKS``: what ``loss_preregistration.json`` says.

    The same dict the wrapper writes to ``loss_preregistration_prnd.json``, so
    the shared file and the per-arm file cannot drift apart.
    """
    return loss_form()


def setup_record() -> dict[str, Any]:
    """What ``run_setup.json`` carries under ``new_arm`` for this arm."""
    o = options()
    return {
        "prnd": {
            "flags": o,
            "milestone_fracs": list(MILESTONE_FRACS),
            "warmup_frac": WARMUP_FRAC,
            "reference_max_iter": REFERENCE_MAX_ITER,
            "schedule": schedule_for(PREREGISTERED_TOTAL_STEPS),
            "pixgt_source": GT_TO_PIXGT_SOURCE[str(o["gt"])],
            "point_seed_offset": POINT_SEED_OFFSET,
            "loss_preregistration": loss_form(),
            "steps_jsonl_columns": {
                "losses": ["L_coarse", "L_point"] +
                          (["L_dice"] if str(o["point_loss"]) == "m2f" else []),
                "diagnostics_not_losses": [
                    "L_diag_point_accuracy", "L_diag_point_n",
                    "L_diag_point_n_importance", "L_diag_point_gt_mean"],
                "note": ("aggregate() prefixes every term with 'L_'; the "
                         "'diag_' entries are observation columns "
                         "(point/accuracy, point counts), NOT loss terms.  The "
                         "optimised quantity is L_coarse + L_point"
                         " (+ L_dice on row 2)."),
            },
        }
    }
