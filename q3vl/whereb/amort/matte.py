"""EPR-022 MATTE: ViTMatte's ``Detail_Capture`` + ``MattingCriterion``, ported
verbatim onto the frozen Qwen3-VL ``F_pre`` grid with a language condition.

The whole arm is one file behind the :mod:`q3vl.whereb.amort.arms` contract, so
no shared file is edited.  What is ported, and from where (raw files downloaded
2026-08-14 from ``hustvl/ViTMatte`` main and re-read line by line while writing
this module):

``modeling/decoder/detail_capture.py``
    ``Basic_Conv3x3`` L5-26, ``ConvStream`` L28-57, ``Fusion_Block`` L59-76,
    ``Matting_Head`` L78-98, ``Detail_Capture`` L100-139.  Channel tables,
    strides, ``padding=1``, ``bias=False``, ``ReLU(True)``, ``scale_factor=2``
    bilinear / ``align_corners=False``, the concat order ``[D, F_up]``, the
    ``D{len-i-1}`` take order and the trailing ``sigmoid`` (L138) are the
    original's, unchanged.
``modeling/criterion/matting_criterion.py``
    ``loss_gradient_penalty`` L13-36 (Sobel kernels L21/L26, the ``0.01``
    sparsity pair L33-34), ``loss_pha_laplacian`` L38-42, ``unknown_l1_loss``
    L44-50, ``known_l1_loss`` L52-63 (its ``sum == 0 -> scale = 0`` guard
    L56-59), ``laplacian_loss`` L77-84 (``max_levels=5``, ``2**level``,
    ``/max_levels``) and the pyramid helpers L86-132.  The literal ``262144``
    (L18/L46/L59) is kept as a literal.
``configs/common/model.py``
    the four-loss list L38, ``pixel_mean``/``pixel_std`` L40-41.
``engine/mattingtrainer.py``
    ``losses = sum(loss_dict.values())`` L34 -> the four terms are added at
    weight 1.0 each.
``configs/ViTMatte_S_100ep.py`` + ``configs/common/optimizer.py`` +
detectron2 ``configs/common/optim.py``
    ``lr = 5e-4``, ``wd = 0.1``, ``betas = (0.9, 0.999)``,
    ``weight_decay_norm = 0.0``, MultiStep ``[1.0, 0.1, 0.05]`` at 30% / 90% of
    the horizon, ``warmup_factor = 0.001``, ``warmup_length = 250/134687``.
    Carried over as FRACTIONS of this campaign's 1200-step horizon
    (:func:`scheduler_kwargs`); the backbone's 0.65 layer-wise lr decay does not
    apply because the base is frozen.

The five deviations, each one flagged NOVEL in the proposal's port table and
each one recorded in :meth:`MatteHead.facts`:

(a) ``BatchNorm2d -> GroupNorm(8, C)`` -- the pipeline is per-sample (13 grid
    shapes in 400 samples, ``trainer.py:90-97``), so a batch statistic is a
    single-sample statistic.  ``--matte-norm bn`` is the ablation row.
(b) ``img_chans 4 -> 3``: there is no trimap, so the fourth channel is dropped
    and ``conv_chans`` becomes ``[3, 48, 96, 192]`` (last fusion in = 64+3 = 67).
(c) the language condition replaces the trimap: FiLM on every fusion output,
    ``Linear(2560,256) + ReLU + Linear(256, 2*sum(fusion_out))``, last layer
    zero-initialised so step 0 is the unconditional ViTMatte exactly.
    ``--matte-cond concat`` is the ablation row (project to 16 channels, tile,
    concatenate onto the ConvStream input).
(d) ``sample_map``: ``0 < gt_pix < 1`` instead of ``trimap == 0.5``.
(e) the ``sum == 0 -> scale = 0`` guard of ``known_l1_loss`` L56-59 is given to
    ``unknown_l1_loss`` AND to ``loss_gradient_penalty`` as well, because a
    foreign-instruction sample's GT is identically zero here and the original
    data has no such case.

NOTES -- decisions taken on the proposal's written conservative default, not
silently invented (each is in ``facts()`` and in ``config/matte_setup.json``):

1. D-1 warmup: the PROPORTIONAL reading, ``round(1200 * 250/134687) = 2`` steps.
2. D-2: the literal ``262144``, not ``H*W``.
3. D-3: batch 32 (the campaign's ``effective_batch``), not ViTMatte's 16.
4. D-4: the repo's ``max_grad_norm = 1.0`` -- ViTMatte does not *declare*
   ``clip_gradients``, which is not the same as declaring it off.
5. D-8: ``is_fake`` samples get ``gt_pix = 0`` and go through all four terms;
   their four terms are ALSO logged separately as ``L_*_fake``.
6. D-9: the analytic re-render spot check is ``mean-abs <= 0.02`` over the first
   ``--matte-gt-audit`` (200) analytic samples, comparing the re-render against
   the published ``.cgt``-derived raster the builder already carries as
   ``gt_hi``; a sample over tolerance falls back to that raster **for the rest
   of the run** and is counted.
7. D-10: the FiLM form above.  D-12: ``<seg_color>`` is not consumed.
8. Precision: ``--matte-precision autocast`` (the repo's ambient bf16, which is
   what the proposal's port table asks for); ``fp32`` wraps head + criterion in
   ``no_autocast`` and is the alternative row.

CONTRACT GAP (reported, not worked around in a shared file): the proposal's
§3 ④ asks for an ``img_pix`` field on ``AmortSampleInputs``; the shared
infrastructure landed ``gt_pix`` but not ``img_pix``, so the 3-channel image the
ConvStream is supposed to read is not on the sample.  Two paths are provided and
the one in force is recorded per sample:

* ``rgb`` -- ``x.img_pix`` when the field exists, else ``x.pixgt.img``, which
  :func:`install_image_seam` attaches by subclassing ``PixGTProvider`` from the
  arm's own entry script (the seam idiom ``run_uniq4b_arm.py:90-91`` already
  uses).  This is the proposal-faithful path and costs no extra IO: the sample's
  PIL image is already decoded and ``image_tensor()`` is a numpy copy.
* ``luma`` -- ``x.guide_hi``, the Rec.709 luma the builder computes anyway under
  ``want_hi=True``.  ``img_chans`` becomes 1.  This is a DEGRADED path; it exists
  only so ``--arm MATTE`` through the bare entry script fails soft-and-counted
  rather than not running at all.

``--matte-img-source auto`` (default) picks ``rgb`` when it is available and
``luma`` otherwise, prints which, and records it.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from q3vl.where.config import VISION_PATCH
from q3vl.where.upsample import area_resize
from q3vl.whereb.fields import no_autocast
from q3vl.whereb.metrics import gt_area_k, percentile, soft_iou_value, topk_mask

__all__ = [
    "ARM", "CRITERIA",
    "Basic_Conv3x3", "ConvStream", "Fusion_Block", "Matting_Head",
    "Detail_Capture", "MattingCriterion", "MatteHead",
    "laplacian_loss", "gauss_kernel", "gauss_grad_kernels", "pix_sad",
    "pix_gradient_error", "install_image_seam",
    "add_arguments", "head_kwargs_from_args", "build_head", "forward",
    "compute_loss", "optimizer_spec", "scheduler_kwargs", "builder_kwargs",
    "per_sample_row", "criteria_columns", "train_stats",
    "loss_preregistration",
]

#: registry name and the pre-registered criterion column (``arms.ARM_CRITERIA``)
ARM = "MATTE"
CRITERIA = ("pix_readout",)

# --------------------------------------------------------------------------- #
# literals, all quoted from the reference files
# --------------------------------------------------------------------------- #
#: ``configs/common/model.py:40-41``
PIXEL_MEAN: tuple[float, float, float] = (123.675 / 255., 116.280 / 255., 103.530 / 255.)
PIXEL_STD: tuple[float, float, float] = (58.395 / 255., 57.120 / 255., 57.375 / 255.)
#: ``q3vl/where/upsample.py:69`` -- the weights the builder's luma already uses
LUMA_WEIGHTS: tuple[float, float, float] = (0.2126, 0.7152, 0.0722)
#: NOVEL, only used on the degraded 1-channel path: the same ImageNet constants
#: collapsed through the luma weights, so the single channel is standardised the
#: way the three channels would have been.
LUMA_MEAN: float = float(sum(w * m for w, m in zip(LUMA_WEIGHTS, PIXEL_MEAN)))
LUMA_STD: float = float(sum(w * s for w, s in zip(LUMA_WEIGHTS, PIXEL_STD)))

#: ``matting_criterion.py:18, 46, 59`` -- the literal, kept literal (D-2)
NORM_CONST = 262144
#: ``matting_criterion.py:33-34``
GRAD_SPARSITY = 0.01
#: ``matting_criterion.py:77``
LAP_MAX_LEVELS = 5

#: ``configs/common/model.py:38`` -- names AND order
LOSS_NAMES: tuple[str, ...] = ("unknown_l1_loss", "known_l1_loss",
                               "loss_pha_laplacian", "loss_gradient_penalty")
#: what a ``--matte-losses`` entry may be spelled as
LOSS_ALIASES: dict[str, str] = {
    "unknown_l1": "unknown_l1_loss", "unknown_l1_loss": "unknown_l1_loss",
    "known_l1": "known_l1_loss", "known_l1_loss": "known_l1_loss",
    "lap": "loss_pha_laplacian", "pha_laplacian": "loss_pha_laplacian",
    "loss_pha_laplacian": "loss_pha_laplacian",
    "grad": "loss_gradient_penalty", "gradient_penalty": "loss_gradient_penalty",
    "loss_gradient_penalty": "loss_gradient_penalty",
}
#: ``steps.jsonl`` column stem of each term (``aggregate`` prefixes ``L_``)
TERM_OF: dict[str, str] = {"unknown_l1_loss": "unknown_l1",
                           "known_l1_loss": "known_l1",
                           "loss_pha_laplacian": "pha_laplacian",
                           "loss_gradient_penalty": "gradient_penalty"}
DEFAULT_LOSSES = "unknown_l1,known_l1,lap,grad"

#: ``ViTMatte_S_100ep.py:8, 11-13, 15`` and ``configs/common/scheduler.py:11``
VITMATTE_MAX_ITER = int(43100 / 16 / 2 * 100)          # 134687
VITMATTE_MILESTONES = (int(43100 / 16 / 2 * 30), int(43100 / 16 / 2 * 90))
VITMATTE_VALUES: tuple[float, ...] = (1.0, 0.1, 0.05)
VITMATTE_WARMUP_ITERS = 250
VITMATTE_WARMUP_FACTOR = 0.001
VITMATTE_LR = 5e-4
VITMATTE_WD = 0.1                                       # detectron2 optim.py:27
VITMATTE_BETAS = (0.9, 0.999)                           # detectron2 optim.py:26

#: ``detail_capture.py:35, 109``
CONVSTREAM_OUT: tuple[int, ...] = (48, 96, 192)
FUSION_OUT: tuple[int, ...] = (256, 128, 64, 32)
#: NOVEL (proposal §3 ⑤(c) / D-10): the concat conditioning's channel budget
CONCAT_COND_CH = 16
#: NOVEL: the FiLM MLP's hidden width (proposal D-10 (iii))
FILM_HIDDEN = 256

#: set by :func:`install_image_seam`; the head reads it at construction time so
#: ``img_chans`` is decided once, loudly, and recorded.
SEAM: dict[str, Any] = {"pixgt_img": False}


# --------------------------------------------------------------------------- #
# detail_capture.py, ported
# --------------------------------------------------------------------------- #
def _norm2d(kind: str, ch: int) -> nn.Module:
    """``detail_capture.py`` L18 / L90's ``BatchNorm2d``, or the GN8 stand-in."""
    if kind == "gn8":
        if ch % 8:
            raise ValueError(f"GroupNorm(8, {ch}) needs a multiple of 8")
        return nn.GroupNorm(8, ch)
    if kind == "bn":
        return nn.BatchNorm2d(ch)
    raise ValueError(f"unknown --matte-norm {kind!r}; expected gn8 or bn")


class Basic_Conv3x3(nn.Module):
    """``detail_capture.py:5-26``, with the norm layer selectable (NOVEL (a))."""

    def __init__(self, in_chans: int, out_chans: int, stride: int = 2,
                 padding: int = 1, norm: str = "gn8"):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, 3, stride, padding, bias=False)
        self.bn = _norm2d(norm, out_chans)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvStream(nn.Module):
    """``detail_capture.py:28-57``.  ``D0`` is the input tensor itself (L51)."""

    def __init__(self, in_chans: int = 4, out_chans: Sequence[int] = CONVSTREAM_OUT,
                 norm: str = "gn8"):
        super().__init__()
        self.convs = nn.ModuleList()
        self.conv_chans = list(out_chans)
        self.conv_chans.insert(0, in_chans)
        for i in range(len(self.conv_chans) - 1):
            self.convs.append(Basic_Conv3x3(self.conv_chans[i],
                                            self.conv_chans[i + 1], norm=norm))

    def forward(self, x):
        out_dict = {"D0": x}
        for i in range(len(self.convs)):
            x = self.convs[i](x)
            out_dict["D" + str(i + 1)] = x
        return out_dict


class Fusion_Block(nn.Module):
    """``detail_capture.py:59-76``: bilinear x2, ``cat([D, F_up])``, Conv3x3 s1."""

    def __init__(self, in_chans: int, out_chans: int, norm: str = "gn8"):
        super().__init__()
        self.conv = Basic_Conv3x3(in_chans, out_chans, stride=1, padding=1,
                                  norm=norm)

    def forward(self, x, D):
        F_up = F.interpolate(x, scale_factor=2, mode="bilinear",
                             align_corners=False)
        out = torch.cat([D, F_up], dim=1)
        out = self.conv(out)
        return out


class Matting_Head(nn.Module):
    """``detail_capture.py:78-98``.  Both convs keep their default ``bias=True``."""

    def __init__(self, in_chans: int = 32, mid_chans: int = 16, norm: str = "gn8"):
        super().__init__()
        self.matting_convs = nn.Sequential(
            nn.Conv2d(in_chans, mid_chans, 3, 1, 1),
            _norm2d(norm, mid_chans),
            nn.ReLU(True),
            nn.Conv2d(mid_chans, 1, 1, 1, 0),
        )

    def forward(self, x):
        return self.matting_convs(x)


class Detail_Capture(nn.Module):
    """``detail_capture.py:100-139``.

    ``in_chans`` is the backbone's channel count -- the original's own knob
    (``ViTMatte_B_100ep.py:9`` sets it to 768 for the B backbone), so 1024 for
    ``F_pre`` is that knob, not an adaptation.  ``img_chans`` is 3 (NOVEL (b))
    or 1 on the degraded luma path, plus :data:`CONCAT_COND_CH` under
    ``cond="concat"``.

    ``n_fusion < len(convstream_out) + 1`` is the ``--matte-res 0.5`` row: the
    block chain simply stops one level early, so the output is at 1/2 and the
    matting head takes ``fusion_out[n_fusion - 1]`` channels.  The take order
    ``D{n_conv - i}`` reduces to the original's ``D{len(blks) - i - 1}`` exactly
    when the chain is full.
    """

    def __init__(self, in_chans: int = 384, img_chans: int = 4,
                 convstream_out: Sequence[int] = CONVSTREAM_OUT,
                 fusion_out: Sequence[int] = FUSION_OUT,
                 norm: str = "gn8"):
        super().__init__()
        fusion_out = list(fusion_out)
        convstream_out = list(convstream_out)
        if not 1 <= len(fusion_out) <= len(convstream_out) + 1:
            raise ValueError(
                f"fusion_out has {len(fusion_out)} entries; the ConvStream has "
                f"{len(convstream_out)} levels, so 1..{len(convstream_out) + 1} "
                "fusion blocks are defined")
        self.convstream = ConvStream(in_chans=img_chans, out_chans=convstream_out,
                                     norm=norm)
        self.conv_chans = self.convstream.conv_chans
        self.n_conv = len(convstream_out)

        self.fusion_blks = nn.ModuleList()
        self.fus_channs = fusion_out.copy()
        self.fus_channs.insert(0, in_chans)
        for i in range(len(self.fus_channs) - 1):
            self.fusion_blks.append(Fusion_Block(
                in_chans=self.fus_channs[i] + self.conv_chans[-(i + 1)],
                out_chans=self.fus_channs[i + 1], norm=norm))

        self.matting_head = Matting_Head(in_chans=fusion_out[-1], norm=norm)
        self.fusion_out = tuple(fusion_out)

    def forward(self, features, images, film=None):
        """``film`` is the NOVEL (c) hook: a callable ``(i, x) -> x``."""
        detail_features = self.convstream(images)
        for i in range(len(self.fusion_blks)):
            d_name_ = "D" + str(self.n_conv - i)
            features = self.fusion_blks[i](features, detail_features[d_name_])
            if film is not None:
                features = film(i, features)
        phas = torch.sigmoid(self.matting_head(features))
        return {"phas": phas}


# --------------------------------------------------------------------------- #
# matting_criterion.py, ported
# --------------------------------------------------------------------------- #
def gauss_kernel(device="cpu", dtype=torch.float32) -> torch.Tensor:
    """``matting_criterion.py:98-106``: ``[1,4,6,4,1]`` outer product / 256."""
    kernel = torch.tensor([[1., 4., 6., 4., 1.],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]], device=device, dtype=dtype)
    kernel = kernel / 256
    return kernel[None, None, :, :]


def gauss_convolution(img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """``matting_criterion.py:108-114``."""
    B, C, H, W = img.shape
    img = img.reshape(B * C, 1, H, W)
    img = F.pad(img, (2, 2, 2, 2), mode="reflect")
    img = F.conv2d(img, kernel)
    return img.reshape(B, C, H, W)


def downsample(img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """``matting_criterion.py:116-119``."""
    img = gauss_convolution(img, kernel)
    return img[:, :, ::2, ::2]


def upsample(img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """``matting_criterion.py:121-126``."""
    B, C, H, W = img.shape
    out = torch.zeros((B, C, H * 2, W * 2), device=img.device, dtype=img.dtype)
    out[:, :, ::2, ::2] = img * 4
    return gauss_convolution(out, kernel)


def crop_to_even_size(img: torch.Tensor) -> torch.Tensor:
    """``matting_criterion.py:128-132``."""
    H, W = img.shape[2:]
    H = H - H % 2
    W = W - W % 2
    return img[:, :, :H, :W]


def laplacian_pyramid(img: torch.Tensor, kernel: torch.Tensor,
                      max_levels: int) -> list[torch.Tensor]:
    """``matting_criterion.py:86-96``."""
    current = img
    pyramid = []
    for _ in range(max_levels):
        current = crop_to_even_size(current)
        down = downsample(current, kernel)
        up = upsample(down, kernel)
        pyramid.append(current - up)
        current = down
    return pyramid


def laplacian_loss(pred: torch.Tensor, true: torch.Tensor,
                   max_levels: int = LAP_MAX_LEVELS) -> torch.Tensor:
    """``matting_criterion.py:77-84``: ``2**level`` weights, ``/ max_levels``."""
    kernel = gauss_kernel(device=pred.device, dtype=pred.dtype)
    pred_pyramid = laplacian_pyramid(pred, kernel, max_levels)
    true_pyramid = laplacian_pyramid(true, kernel, max_levels)
    loss = 0
    for level in range(max_levels):
        loss = loss + (2 ** level) * F.l1_loss(pred_pyramid[level],
                                               true_pyramid[level])
    return loss / max_levels


class MattingCriterion(nn.Module):
    """``matting_criterion.py:5-73``, with the two extra zero guards of NOVEL (e).

    ``guard_unknown`` / ``guard_grad`` are the guards; both default on and both
    are exactly ``known_l1_loss``'s own ``if torch.sum(...) == 0: scale = 0``
    (L56-59).  Turning them off reproduces the original's division by zero and
    exists only so a test can show the guard is doing something.
    """

    def __init__(self, losses: Sequence[str] = LOSS_NAMES, *,
                 guard_unknown: bool = True, guard_grad: bool = True):
        super().__init__()
        bad = [k for k in losses if k not in LOSS_NAMES]
        if bad:
            raise ValueError(f"unknown loss(es) {bad}; expected {LOSS_NAMES}")
        self.losses = tuple(losses)
        self.guard_unknown = bool(guard_unknown)
        self.guard_grad = bool(guard_grad)

    # -- the four terms ----------------------------------------------------
    def loss_gradient_penalty(self, sample_map, preds, targets):
        """``matting_criterion.py:13-36`` + the NOVEL (e) guard on ``scale``."""
        preds = preds["phas"]
        targets = targets["phas"]

        # sample_map for unknown area
        s = torch.sum(sample_map)
        if self.guard_grad and float(s) == 0:
            # NOVEL (e): the original has no guard here, and an all-hard GT
            # (every foreign-instruction sample) makes the denominator 0.
            scale = 0
        else:
            scale = sample_map.shape[0] * NORM_CONST / s

        # gradient in x
        sobel_x_kernel = torch.tensor([[[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]],
                                      dtype=preds.dtype, device=preds.device)
        delta_pred_x = F.conv2d(preds, weight=sobel_x_kernel, padding=1)
        delta_gt_x = F.conv2d(targets, weight=sobel_x_kernel, padding=1)

        # gradient in y
        sobel_y_kernel = torch.tensor([[[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]]],
                                      dtype=preds.dtype, device=preds.device)
        delta_pred_y = F.conv2d(preds, weight=sobel_y_kernel, padding=1)
        delta_gt_y = F.conv2d(targets, weight=sobel_y_kernel, padding=1)

        loss = (F.l1_loss(delta_pred_x * sample_map, delta_gt_x * sample_map) * scale
                + F.l1_loss(delta_pred_y * sample_map, delta_gt_y * sample_map) * scale
                + GRAD_SPARSITY * torch.mean(torch.abs(delta_pred_x * sample_map)) * scale
                + GRAD_SPARSITY * torch.mean(torch.abs(delta_pred_y * sample_map)) * scale)
        return {"loss_gradient_penalty": loss}

    def loss_pha_laplacian(self, preds, targets):
        """``matting_criterion.py:38-42``.

        The 5-level pyramid reflect-pads by 2 at every level, so the smallest
        side must survive four halvings and still be > 2: ``min(H, W) >= 48``.
        ViTMatte's 512x512 crop and this campaign's 512-short-side spec-5 image
        both satisfy it; a smaller input would otherwise die inside ``F.pad``
        with a message about dimension 3.
        """
        h, w = preds["phas"].shape[-2:]
        if min(int(h), int(w)) < 3 * 2 ** (LAP_MAX_LEVELS - 1):
            raise ValueError(
                f"loss_pha_laplacian needs min(H, W) >= "
                f"{3 * 2 ** (LAP_MAX_LEVELS - 1)} for its {LAP_MAX_LEVELS}-level "
                f"pyramid; got {(int(h), int(w))}")
        return {"loss_pha_laplacian": laplacian_loss(preds["phas"], targets["phas"])}

    def unknown_l1_loss(self, sample_map, preds, targets):
        """``matting_criterion.py:44-50`` + the NOVEL (e) guard on ``scale``."""
        s = torch.sum(sample_map)
        if self.guard_unknown and float(s) == 0:
            scale = 0
        else:
            scale = sample_map.shape[0] * NORM_CONST / s
        loss = F.l1_loss(preds["phas"] * sample_map,
                         targets["phas"] * sample_map) * scale
        return {"unknown_l1_loss": loss}

    def known_l1_loss(self, sample_map, preds, targets):
        """``matting_criterion.py:52-63``, verbatim including its own guard."""
        new_sample_map = torch.zeros_like(sample_map)
        new_sample_map[sample_map == 0] = 1
        if torch.sum(new_sample_map) == 0:
            scale = 0
        else:
            scale = new_sample_map.shape[0] * NORM_CONST / torch.sum(new_sample_map)
        loss = F.l1_loss(preds["phas"] * new_sample_map,
                         targets["phas"] * new_sample_map) * scale
        return {"known_l1_loss": loss}

    def forward(self, sample_map, preds, targets):
        """``matting_criterion.py:66-73``: three terms take the map, laplacian
        does not."""
        losses: dict[str, torch.Tensor] = {}
        for k in self.losses:
            if k in ("unknown_l1_loss", "known_l1_loss", "loss_gradient_penalty"):
                losses.update(getattr(self, k)(sample_map, preds, targets))
            else:
                losses.update(getattr(self, k)(preds, targets))
        return losses


# --------------------------------------------------------------------------- #
# MAM diagnostic columns (口径 only; never a loss, never the headline)
# --------------------------------------------------------------------------- #
def gauss_grad_kernels(sigma: float = 1.4) -> tuple[np.ndarray, np.ndarray, int]:
    """``Matting-Anything/evaluation/metrics.py:27-47`` (``genGaussKernel``).

    ``q`` is a parameter of the original signature that its body never reads;
    it is therefore not a parameter here.
    """
    eps = 1e-2
    hsize = int(np.ceil(sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * eps))))
    size = 2 * hsize + 1
    u = np.arange(size, dtype=np.float64) - hsize
    g = np.exp(-np.power(u, 2) / (2 * np.power(sigma, 2))) / (sigma * np.sqrt(2 * np.pi))
    dg = -u * g / np.power(sigma, 2)
    hx = np.outer(g, dg).astype(np.float32)            # hx[i,j] = gauss(u)*dgauss(v)
    hx = hx / np.sqrt(np.sum(np.power(np.abs(hx), 2)))
    hy = hx.transpose(1, 0).copy()
    return hx, hy, size


_GRAD_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor, int]] = {}


def _grad_filters(sigma: float, device, dtype):
    key = (round(float(sigma), 6), str(device), str(dtype))
    got = _GRAD_CACHE.get(key)
    if got is None:
        hx, hy, size = gauss_grad_kernels(sigma)
        # `ImageFilter` (metrics.py:56-64) is an nn.Conv2d, i.e. correlation, and
        # the caller pre-flips the kernel (metrics.py:104-105) to make it a true
        # convolution.  Same two steps here.
        kx = torch.from_numpy(hx[::-1, ::-1].copy()).to(device=device, dtype=dtype)
        ky = torch.from_numpy(hy[::-1, ::-1].copy()).to(device=device, dtype=dtype)
        got = (kx[None, None], ky[None, None], size)
        _GRAD_CACHE[key] = got
    return got


def _grad_amp(field: torch.Tensor, sigma: float = 1.4) -> torch.Tensor:
    """``sqrt(fx^2 + fy^2)`` of ``(H, W)`` under the MAM gaussian derivative."""
    x = field.reshape(1, 1, *field.shape[-2:]).float()
    kx, ky, size = _grad_filters(sigma, x.device, x.dtype)
    pad = size // 2
    xp = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    gx = F.conv2d(xp, kx)
    gy = F.conv2d(xp, ky)
    return (gx.pow(2) + gy.pow(2)).sqrt()[0, 0]


def pix_gradient_error(pred: torch.Tensor, gt_amp: torch.Tensor,
                       sigma: float = 1.4) -> float:
    """``BatchGradient`` (``metrics.py:190-203``) with an all-ones mask.

    MAM divides by ``mask.sum() + 1``; with the full-frame mask that is
    ``H*W + 1``.  Inputs are already in [0,1] (MAM's ``/255.`` normalisation).
    """
    amp = _grad_amp(pred, sigma)
    err = (amp - gt_amp).pow(2)
    return float(err.sum() / (err.numel() + 1.0))


def pix_sad(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """``BatchSAD`` (``metrics.py:169-174``): ``sum|d| / 1000``, full frame.

    Scales linearly with frame area and is NOT comparable with the unknown-area
    SAD published in the matting literature; it is a within-board column only.
    """
    return float((pred.float() - gt.float()).abs().sum() / 1000.0)


# --------------------------------------------------------------------------- #
# the head
# --------------------------------------------------------------------------- #
@dataclass
class _Counters:
    img_source: dict[str, int]
    gt_pix_source: dict[str, int]
    audit_n: int = 0
    audit_pass: int = 0
    audit_fail: int = 0
    audit_worst: float = 0.0
    audit_skipped: dict[str, int] = None            # type: ignore[assignment]
    n_fake: int = 0
    n_samples: int = 0
    n_empty_sample_map: int = 0

    def __post_init__(self) -> None:
        if self.audit_skipped is None:
            self.audit_skipped = {}


class MatteHead(nn.Module):
    """``Detail_Capture`` + the language condition + the run-time bookkeeping.

    Everything that is not ViTMatte lives here rather than inside the ported
    classes, so the port can be read against the raw file line by line.
    """

    def __init__(self, *, in_dim: int = 1024, text_dim: int = 2560,
                 cond: str = "film", res: float = 1.0, norm: str = "gn8",
                 losses: Sequence[str] = LOSS_NAMES,
                 img_source: str = "auto", precision: str = "autocast",
                 audit_n: int = 200, audit_tol: float = 0.02,
                 control_seed: int = 20260814, grad_sigma: float = 1.4,
                 pix_diag: bool = True,
                 guard_unknown: bool = True, guard_grad: bool = True):
        super().__init__()
        if cond not in ("film", "concat"):
            raise ValueError(f"unknown --matte-cond {cond!r}; expected film|concat")
        if precision not in ("autocast", "fp32"):
            raise ValueError(
                f"unknown --matte-precision {precision!r}; expected autocast|fp32")
        n_fusion = {1.0: 4, 0.5: 3}.get(round(float(res), 3))
        if n_fusion is None:
            raise ValueError(f"unknown --matte-res {res!r}; expected 1.0 or 0.5")

        self.in_dim = int(in_dim)
        self.text_dim = int(text_dim)
        self.cond_kind = cond
        self.res = float(res)
        self.norm_kind = norm
        self.losses = tuple(losses)
        self.precision = precision
        self.audit_budget = int(audit_n)
        self.audit_tol = float(audit_tol)
        self.control_seed = int(control_seed)
        self.grad_sigma = float(grad_sigma)
        self.pix_diag = bool(pix_diag)

        self.img_source = _resolve_img_source(img_source)
        self.img_source_flag = img_source
        base_img_chans = 3 if self.img_source == "rgb" else 1
        img_chans = base_img_chans + (CONCAT_COND_CH if cond == "concat" else 0)
        self.base_img_chans = base_img_chans
        self.img_chans = img_chans

        fusion_out = list(FUSION_OUT[:n_fusion])
        self.detail = Detail_Capture(in_chans=self.in_dim, img_chans=img_chans,
                                     convstream_out=CONVSTREAM_OUT,
                                     fusion_out=fusion_out, norm=norm)
        #: the pixel stride of the head's output relative to the image
        self.out_stride = 2 ** (len(CONVSTREAM_OUT) + 1 - n_fusion)

        if cond == "film":
            self.film_mlp = nn.Sequential(
                nn.Linear(self.text_dim, FILM_HIDDEN), nn.ReLU(),
                nn.Linear(FILM_HIDDEN, 2 * sum(fusion_out)))
            # zero-initialised last layer: step 0 is x*(1+0)+0, i.e. the
            # unconditional ViTMatte exactly (proposal "初始化" row / D-10)
            nn.init.zeros_(self.film_mlp[-1].weight)
            nn.init.zeros_(self.film_mlp[-1].bias)
            self.cond_proj = None
        else:
            self.film_mlp = None
            self.cond_proj = nn.Linear(self.text_dim, CONCAT_COND_CH)

        # `vitmatte.py:63`'s ImageNet constants, held as BUFFERS rather than
        # rebuilt per call: a buffer follows `.to(device)` / `.to(dtype)` with
        # the rest of the head, so the normalisation can never be computed on a
        # different device from the ConvStream weights.  `persistent=False`
        # keeps them out of `state_dict` (they are literals, not state).
        self.register_buffer("pixel_mean",
                             torch.tensor(PIXEL_MEAN,
                                          dtype=torch.float32).reshape(1, 3, 1, 1),
                             persistent=False)
        self.register_buffer("pixel_std",
                             torch.tensor(PIXEL_STD,
                                          dtype=torch.float32).reshape(1, 3, 1, 1),
                             persistent=False)

        self.criterion = MattingCriterion(self.losses,
                                          guard_unknown=guard_unknown,
                                          guard_grad=guard_grad)
        self.rt = _Counters(img_source={}, gt_pix_source={})
        #: sample ids whose analytic re-render failed the D-9 spot check; they
        #: keep using the published raster for the rest of the run
        self.audit_failed: set[str] = set()
        self._film_slices = _film_slices(fusion_out)
        self.fusion_out = tuple(fusion_out)

    # -- conditioning ------------------------------------------------------
    def _film_fn(self, h_cond: torch.Tensor):
        if self.film_mlp is None:
            return None
        w = self.film_mlp[0].weight
        gb = self.film_mlp(h_cond.reshape(1, -1).to(device=w.device,
                                                    dtype=w.dtype))[0]

        def apply(i: int, x: torch.Tensor) -> torch.Tensor:
            g0, g1, b0, b1 = self._film_slices[i]
            gamma = gb[g0:g1].reshape(1, -1, 1, 1).to(x.dtype)
            beta = gb[b0:b1].reshape(1, -1, 1, 1).to(x.dtype)
            return x * (1.0 + gamma) + beta

        return apply

    # -- the image ---------------------------------------------------------
    @property
    def io_ref(self) -> torch.Tensor:
        """The weight every detail-branch input is aligned to.

        The image is the ONE input this arm reads straight off the dataset
        (``AmortSampleInputs.pixgt.img`` is the decoded PIL tensor, which
        ``amort/data.py:769`` never moves -- only ``pg.alpha`` is), so the
        device/dtype has to be taken from the module, not assumed.
        """
        return self.detail.convstream.convs[0].conv.weight

    def image_of(self, x: Any) -> torch.Tensor:
        """``(1, img_chans_before_cond, H, W)``, normalised, on the head's device.

        Raises rather than substituting a different resolution or a zero tensor:
        a detail branch fed the wrong picture is invisible in every column.
        """
        ref = self.io_ref
        got = None
        src = ""
        if self.img_source == "rgb":
            got = getattr(x, "img_pix", None)
            src = "img_pix"
            if got is None:
                pg = getattr(x, "pixgt", None)
                got = getattr(pg, "img", None) if pg is not None else None
                src = "pixgt_img"
            if got is None:
                raise ValueError(
                    f"{x.sample_id}: --matte-img-source rgb needs the spec-5 RGB "
                    "image.  AmortSampleInputs carries no `img_pix` (shared "
                    "infrastructure gap) and no image was attached to the PixGT; "
                    "launch through q3vl/whereb/scripts/run_matte_arm.py, which "
                    "installs matte.install_image_seam(), or pass "
                    "--matte-img-source luma to run the degraded 1-channel path.")
            img = got.reshape(1, 3, *got.shape[-2:]).to(device=ref.device,
                                                        dtype=torch.float32)
            img = (img - self.pixel_mean.to(torch.float32)) \
                / self.pixel_std.to(torch.float32)       # vitmatte.py:63
        else:
            got = getattr(x, "guide_hi", None)
            src = "guide_hi"
            if got is None:
                raise ValueError(
                    f"{x.sample_id}: --matte-img-source luma reads `guide_hi`, "
                    "which the builder only computes with want_hi=True.  "
                    "matte.builder_kwargs() sets it; a builder constructed "
                    "without it cannot serve this arm.")
            img = got.reshape(1, 1, *got.shape[-2:]).to(device=ref.device,
                                                        dtype=torch.float32)
            img = (img - LUMA_MEAN) / LUMA_STD
        # the normalisation is done in fp32 and handed over in the head's own
        # parameter dtype: under the ambient bf16 autocast that is fp32 and the
        # conv casts it, and a head explicitly cast to bf16 gets bf16.
        img = img.to(dtype=ref.dtype)
        self.rt.img_source[src] = self.rt.img_source.get(src, 0) + 1
        want = (x.grid_h * VISION_PATCH, x.grid_w * VISION_PATCH)
        if tuple(img.shape[-2:]) != want:
            raise AssertionError(
                f"{x.sample_id}: detail-branch image is {tuple(img.shape[-2:])} "
                f"but the F_pre grid {(x.grid_h, x.grid_w)} implies {want} "
                f"(stride {VISION_PATCH}); the x2 fusion chain would not line up")
        return img

    # -- forward -----------------------------------------------------------
    def forward(self, feat: torch.Tensor, img: torch.Tensor,
                h_cond: torch.Tensor) -> torch.Tensor:
        """``(1, 1024, gh, gw)`` + ``(1, C, H, W)`` + ``(2560,)`` -> alpha."""
        ref = self.io_ref
        if img.device != ref.device or img.dtype != ref.dtype:
            # a caller that built the image itself (or a future `img_pix`
            # field) must not be able to hand the ConvStream a CPU tensor.
            # A no-op in every live configuration: the head's parameters are
            # fp32 and `image_of` already returns fp32 on the head's device,
            # so the ambient bf16 autocast -- not this line -- is what decides
            # the conv's compute dtype (and `--matte-precision fp32` passes
            # fp32 into fp32 weights under `no_autocast`).
            img = img.to(device=ref.device, dtype=ref.dtype)
        if self.cond_proj is not None:
            w = self.cond_proj.weight
            c = self.cond_proj(h_cond.reshape(1, -1).to(device=w.device,
                                                        dtype=w.dtype))
            c = c.reshape(1, CONCAT_COND_CH, 1, 1).expand(
                1, CONCAT_COND_CH, img.shape[-2], img.shape[-1]).to(img.dtype)
            img = torch.cat([img, c], dim=1)
        film = self._film_fn(h_cond)
        out = self.detail(feat, img, film=film)
        return out["phas"]

    # -- the pixel GT ------------------------------------------------------
    def target_of(self, x: Any, size: tuple[int, int]) -> torch.Tensor:
        """``(1, 1, h, w)`` supervision target, with the D-9 audit and D-8 rule.

        ``size`` is the head's own output size, so ``--matte-res 0.5`` projects
        the GT down with the ``gt_low`` operator rather than upsampling alpha.
        """
        gt = getattr(x, "gt_pix", None)
        if gt is None:
            raise ValueError(
                f"{x.sample_id}: the MATTE criterion is pixel-resolution and the "
                "builder supplied no `gt_pix`.  Drop --no-pixgt (the provider is "
                "built by run_amort_arm when --arm is one of the new arms).")
        src = str(getattr(x, "gt_pix_source", "") or "unknown")
        self.rt.gt_pix_source[src] = self.rt.gt_pix_source.get(src, 0) + 1
        self.rt.n_samples += 1

        if not bool(getattr(x, "is_fake", False)):
            # a foreign-instruction draw discards the GT below, so auditing it
            # would spend the n=200 budget on samples the run never trains on
            gt = self._audited(x, gt, src)
        if bool(getattr(x, "is_fake", False)):
            # D-8, written down: a foreign instruction has no GT here, so the
            # target is identically zero and the four terms are computed on it.
            gt = torch.zeros_like(gt)
            self.rt.n_fake += 1
        gt = gt.reshape(1, 1, *gt.shape[-2:]).float()
        if tuple(gt.shape[-2:]) != tuple(size):
            gt = area_resize(gt, (int(size[0]), int(size[1])))
        return gt

    def _audited(self, x: Any, gt: torch.Tensor, src: str) -> torch.Tensor:
        """D-9: spot-check the analytic re-render, fall back and COUNT."""
        sid = str(x.sample_id)
        raster = getattr(x, "gt_hi", None)
        if sid in self.audit_failed:
            return raster.float() if raster is not None else gt
        if src != "render":
            return gt
        if self.rt.audit_n >= self.audit_budget:
            return gt
        if raster is None:
            self.rt.audit_skipped["no_gt_hi"] = \
                self.rt.audit_skipped.get("no_gt_hi", 0) + 1
            return gt
        r = raster.float()
        if tuple(r.shape[-2:]) != tuple(gt.shape[-2:]):
            self.rt.audit_skipped["shape_mismatch"] = \
                self.rt.audit_skipped.get("shape_mismatch", 0) + 1
            return gt
        d = float((gt.float() - r).abs().mean())
        self.rt.audit_n += 1
        self.rt.audit_worst = max(self.rt.audit_worst, d)
        if d <= self.audit_tol:
            self.rt.audit_pass += 1
            return gt
        self.rt.audit_fail += 1
        self.audit_failed.add(sid)
        return r

    # -- record ------------------------------------------------------------
    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def facts(self) -> dict[str, Any]:
        return {
            "arm": ARM,
            "port": "hustvl/ViTMatte @ main (raw files read 2026-08-14)",
            "in_chans": self.in_dim, "img_chans": self.img_chans,
            "base_img_chans": self.base_img_chans,
            "convstream_out": list(CONVSTREAM_OUT),
            "fusion_out": list(self.fusion_out),
            "res": self.res, "out_stride": self.out_stride,
            "norm": self.norm_kind, "cond": self.cond_kind,
            "film_hidden": FILM_HIDDEN if self.cond_kind == "film" else None,
            "film_zero_init_last_layer": self.cond_kind == "film",
            "concat_cond_ch": CONCAT_COND_CH if self.cond_kind == "concat" else None,
            "losses": list(self.losses),
            "loss_weights": {n: 1.0 for n in self.losses},
            "norm_const": NORM_CONST,
            "grad_sparsity": GRAD_SPARSITY,
            "lap_max_levels": LAP_MAX_LEVELS,
            "zero_guards": {"unknown_l1_loss": self.criterion.guard_unknown,
                            "loss_gradient_penalty": self.criterion.guard_grad,
                            "known_l1_loss": True},
            "precision": self.precision,
            "img_source_flag": self.img_source_flag,
            "img_source": self.img_source,
            "img_source_counts": dict(sorted(self.rt.img_source.items())),
            "gt_pix_source": dict(sorted(self.rt.gt_pix_source.items())),
            "gt_pix_audit": {"tol": self.audit_tol, "budget": self.audit_budget,
                             "n": self.rt.audit_n, "n_pass": self.rt.audit_pass,
                             "n_fail": self.rt.audit_fail,
                             "worst_mean_abs": self.rt.audit_worst,
                             "n_fallback_ids": len(self.audit_failed),
                             "skipped": dict(sorted(self.rt.audit_skipped.items()))},
            "n_samples": self.rt.n_samples, "n_fake": self.rt.n_fake,
            "n_empty_sample_map": self.rt.n_empty_sample_map,
            "n_params": self.n_params(),
            "n_params_detail": int(sum(p.numel() for p in self.detail.parameters())),
            "grad_sigma": self.grad_sigma,
            "control_seed": self.control_seed,
            "seam_pixgt_img": bool(SEAM.get("pixgt_img")),
        }


def _film_slices(fusion_out: Sequence[int]) -> list[tuple[int, int, int, int]]:
    """Per-block ``(gamma_lo, gamma_hi, beta_lo, beta_hi)`` into the flat MLP out."""
    out, off = [], 0
    for c in fusion_out:
        out.append((off, off + c, off + c, off + 2 * c))
        off += 2 * c
    return out


def _resolve_img_source(flag: str) -> str:
    """``auto|rgb|luma`` -> ``rgb|luma``, deciding ONCE and saying so."""
    if flag not in ("auto", "rgb", "luma"):
        raise ValueError(
            f"unknown --matte-img-source {flag!r}; expected auto|rgb|luma")
    have_rgb = _rgb_available()
    if flag == "rgb":
        if not have_rgb:
            raise ValueError(
                "--matte-img-source rgb: neither AmortSampleInputs.img_pix "
                "(shared infrastructure gap, EPR-022 §3 ④) nor the PixGT image "
                "seam is available.  Launch through run_matte_arm.py, or accept "
                "the degraded --matte-img-source luma path.")
        return "rgb"
    if flag == "luma":
        return "luma"
    return "rgb" if have_rgb else "luma"


def _rgb_available() -> bool:
    if SEAM.get("pixgt_img"):
        return True
    try:
        from .data import AmortSampleInputs

        return "img_pix" in getattr(AmortSampleInputs, "__dataclass_fields__", {})
    except Exception:                                    # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# the image seam (installed by run_matte_arm.py, never on import)
# --------------------------------------------------------------------------- #
def make_image_provider(base_cls):
    """``PixGTProvider`` subclass that also carries the sample's RGB image."""

    class ImagePixGTProvider(base_cls):                   # type: ignore[misc,valid-type]
        """Attach ``PixGT.img`` -- the spec-5 RGB the ConvStream reads.

        ``PixGTProvider.get`` is handed the ``WhereBSample`` itself
        (``amort/data.py:733``), whose PIL image is already decoded, so
        ``image_tensor()`` is a numpy copy and not a second JPEG decode.
        """

        def get(self, sample, **kw):
            pg = super().get(sample, **kw)
            fn = getattr(sample, "image_tensor", None)
            pg.img = fn() if callable(fn) else None
            return pg

    return ImagePixGTProvider


def install_image_seam() -> dict[str, Any]:
    """Swap ``pixgt.PixGTProvider`` for the image-carrying subclass.

    Called by ``run_matte_arm.py`` BEFORE ``run_amort_arm.main`` -- the entry
    script resolves the name at call time, so the subclass is what it builds.
    Same seam idiom as ``run_uniq4b_arm.py:90-91``; never done on import, so a
    plain ``--arm MATTE`` through the bare entry script is unaffected.
    """
    from . import pixgt as _pixgt

    if not SEAM.get("pixgt_img"):
        SEAM["base_provider"] = _pixgt.PixGTProvider
        _pixgt.PixGTProvider = make_image_provider(_pixgt.PixGTProvider)
        SEAM["pixgt_img"] = True
    return {"pixgt_img": True,
            "provider": _pixgt.PixGTProvider.__name__}


# --------------------------------------------------------------------------- #
# arm hooks
# --------------------------------------------------------------------------- #
def add_arguments(ap) -> None:
    """The proposal's entry row, one flag per line.  Used by ``run_matte_arm``."""
    ap.add_argument("--matte-cond", default="film", choices=["film", "concat"],
                    help="conditioning form (NOVEL (c)); film = per-fusion "
                         "affine with a zero-initialised last layer")
    ap.add_argument("--matte-res", default="1.0", choices=["1.0", "0.5"],
                    help="1.0 = 4 fusion blocks to full resolution (ViTMatte); "
                         "0.5 = 3 blocks, head at 1/2 (NOVEL)")
    ap.add_argument("--matte-gt-source", default="render", choices=["render", "cgt"],
                    help="TRAINING pixel GT only: render = analytic re-render of "
                         "the three geometric families; cgt = every family from "
                         ".cgt.png.  The headline GT (gt_low) is untouched.")
    ap.add_argument("--matte-losses", default=DEFAULT_LOSSES,
                    help="comma list from unknown_l1,known_l1,lap,grad "
                         "(configs/common/model.py:38 order); all four = the "
                         "faithful recipe")
    ap.add_argument("--matte-norm", default="gn8", choices=["gn8", "bn"],
                    help="gn8 = GroupNorm(8,C) (NOVEL (a)); bn = the original "
                         "BatchNorm2d, the ablation row")
    ap.add_argument("--matte-img-source", default="auto",
                    choices=["auto", "rgb", "luma"],
                    help="detail-branch input.  rgb = the spec-5 RGB (needs the "
                         "PixGT image seam or an img_pix field); luma = the "
                         "1-channel guide_hi, a DEGRADED fallback")
    ap.add_argument("--matte-precision", default="autocast",
                    choices=["autocast", "fp32"],
                    help="autocast = the repo's ambient bf16 (the port table's "
                         "reading); fp32 = head+criterion under no_autocast")
    ap.add_argument("--matte-gt-audit", type=int, default=200,
                    help="D-9: how many analytic-render samples to spot-check "
                         "against the published raster (0 = off)")
    ap.add_argument("--matte-gt-audit-tol", type=float, default=0.02,
                    help="D-9 tolerance, mean-abs")
    ap.add_argument("--matte-control-seed", type=int, default=20260814,
                    help="seed of the random top-k control field; its generator "
                         "is private, so the control never perturbs training RNG")
    ap.add_argument("--matte-grad-sigma", type=float, default=1.4,
                    help="MAM Grad column sigma (evaluation/metrics.py:91)")
    ap.add_argument("--matte-no-pix-diag", action="store_true",
                    help="skip the pixel diagnostic columns; the board then "
                         "cannot publish (pix_readout would be n=0)")


def parse_losses(spec: str | Iterable[str]) -> tuple[str, ...]:
    """``"unknown_l1,known_l1"`` -> the canonical names, in the reference order."""
    if isinstance(spec, str):
        parts = [p.strip() for p in spec.split(",") if p.strip()]
    else:
        parts = [str(p).strip() for p in spec if str(p).strip()]
    got = []
    for p in parts:
        name = LOSS_ALIASES.get(p)
        if name is None:
            raise ValueError(
                f"unknown --matte-losses entry {p!r}; expected a subset of "
                f"{sorted(set(LOSS_ALIASES))}")
        if name not in got:
            got.append(name)
    if not got:
        raise ValueError("--matte-losses selected no term at all")
    return tuple(n for n in LOSS_NAMES if n in got)


def head_kwargs_from_args(args) -> dict[str, Any]:
    def g(name, default):
        return getattr(args, name, default)

    return dict(
        cond=g("matte_cond", "film"),
        res=float(g("matte_res", 1.0)),
        norm=g("matte_norm", "gn8"),
        losses=parse_losses(g("matte_losses", DEFAULT_LOSSES)),
        img_source=g("matte_img_source", "auto"),
        precision=g("matte_precision", "autocast"),
        audit_n=int(g("matte_gt_audit", 200)),
        audit_tol=float(g("matte_gt_audit_tol", 0.02)),
        control_seed=int(g("matte_control_seed", 20260814)),
        grad_sigma=float(g("matte_grad_sigma", 1.4)),
        pix_diag=not bool(g("matte_no_pix_diag", False)),
    )


#: the head built for this process.  ``criteria_columns`` is handed only the
#: per-sample rows, but the pre-registered guard (c) is a property of the RUN
#: (``gt_pix`` source counts + the D-9 audit) and has to reach ``metrics.json``,
#: which is the artefact a reviewer reads.  One arm per process, set once.
_LIVE_HEAD: list[Any] = []


def loss_preregistration(args=None) -> dict[str, Any]:
    """``arms.RUN_REQUIRED_HOOKS``: what ``loss_preregistration.json`` says.

    ``run_amort_arm``'s shared record is the live arms' seven-term ST_LANG
    stack, which this arm never enters (``trainer.py:244-251``).  ViTMatte
    sums its selected terms with **equal weight and no coefficients**
    (``mattingtrainer.py:34``).
    """
    if _LIVE_HEAD:
        losses = tuple(_LIVE_HEAD[0].losses)
    else:
        losses = parse_losses(getattr(args, "matte_losses", DEFAULT_LOSSES))
    return {
        "arm": ARM,
        "form": "L = " + " + ".join(f"1.0 * {TERM_OF[n]}" for n in losses)
                + "   [mattingtrainer.py:34, equal weight, no coefficients]",
        "terms": {
            "unknown_l1": "L1 inside sample_map, scaled by "
                          "N*NORM_CONST/sum(map)   [matting_criterion.py:44-50]",
            "known_l1": "L1 outside sample_map, same scaling"
                        "   [matting_criterion.py:52-63]",
            "pha_laplacian": f"{LAP_MAX_LEVELS}-level Laplacian pyramid L1, "
                             "2**level weighting   [matting_criterion.py:38-42, "
                             "utils L77-84]",
            "gradient_penalty": "Sobel-x/y L1 inside sample_map + "
                                f"{GRAD_SPARSITY} * sparsity of the predicted "
                                "gradients   [matting_criterion.py:13-36]",
        },
        "selected": [TERM_OF[n] for n in losses],
        "sample_map": "NOVEL (d): the GT's own soft band (0 < g < 1) stands in "
                      "for ViTMatte's trimap == 0.5 unknown region",
        "is_fake": "D-8: the four terms are ALSO logged as *_fake columns and "
                   "are NOT added to the total a second time",
        "seven_term_stack": "not entered (trainer.py:244-251)",
        "iou_as_field_target": False,
        "iou_as_selection_target": False,
        "dice_as_target": False,
    }


def build_head(*, in_dim: int, text_dim: int, args=None, **kw) -> nn.Module:
    head = MatteHead(in_dim=in_dim, text_dim=text_dim, **kw)
    _LIVE_HEAD.clear()
    _LIVE_HEAD.append(head)
    print(f"MATTE head: in_chans={head.in_dim} img_chans={head.img_chans} "
          f"({head.img_source}) cond={head.cond_kind} res={head.res} "
          f"norm={head.norm_kind} losses={list(head.losses)} "
          f"params={head.n_params()}", flush=True)
    return head


def forward(model, head, ctx) -> dict[str, Any]:
    """``arms.ArmContext`` -> ``{"m_low", "alpha_pix", ...}``."""
    x = ctx.sample
    if x is None:
        raise ValueError(
            "arm MATTE needs the AmortSampleInputs (it reads gt_pix and the "
            "spec-5 image); forward_geo was called without sample=")
    h_cond = ctx.require_vector(ARM)                     # (2560,)
    img = head.image_of(x)
    feat = ctx.feat
    if head.precision == "fp32":
        with no_autocast(feat.device.type):
            alpha = head(feat.float(), img.float(), h_cond.float())
    else:
        alpha = head(feat, img, h_cond)
    m_low = area_resize(alpha.float(), (int(ctx.grid_h), int(ctx.grid_w)))[0, 0]
    return {"m_low": m_low, "alpha_pix": alpha, "params": {},
            "matte": {"img_source": head.img_source,
                      "gt_pix_source": str(getattr(x, "gt_pix_source", "")),
                      "out_hw": tuple(int(v) for v in alpha.shape[-2:])}}


def compute_loss(model, out: dict[str, Any], x, weights):
    """The four ported terms, equal weight, summed (``mattingtrainer.py:34``)."""
    from .losses import AmortLoss

    head = model.geo
    alpha = out["alpha_pix"]
    if alpha.dim() != 4 or alpha.shape[:2] != (1, 1):
        raise AssertionError(
            f"{x.sample_id}: alpha_pix is {tuple(alpha.shape)}, expected (1,1,H,W)")
    ctxman = (no_autocast(alpha.device.type) if head.precision == "fp32"
              else _null_ctx())
    with ctxman:
        gt = head.target_of(x, tuple(alpha.shape[-2:])).to(alpha.dtype)
        # NOVEL (d): the GT's own soft band stands in for `trimap == 0.5`
        sample_map = ((gt > 0) & (gt < 1)).to(gt.dtype)
        n_unknown = float(sample_map.sum())
        if n_unknown == 0:
            head.rt.n_empty_sample_map += 1
        terms_raw = head.criterion(sample_map, {"phas": alpha}, {"phas": gt})

    _assert_terms(head, terms_raw, x)
    total = None
    terms: dict[str, torch.Tensor] = {}
    for name in head.losses:
        v = terms_raw[name]
        total = v if total is None else total + v         # equal weight, summed
        terms[TERM_OF[name]] = v
    if bool(getattr(x, "is_fake", False)):
        # D-8: the fake subset's four terms, counted separately in steps.jsonl
        # (`aggregate` turns every `terms` key into an `L_*` column and averages
        # over the samples that carry it).  They are NOT added to `total` again.
        for name in head.losses:
            terms[f"{TERM_OF[name]}_fake"] = terms_raw[name].detach()
    stats = {"matte_unknown_frac": n_unknown / max(1.0, float(sample_map.numel())),
             "matte_alpha_mean": float(alpha.detach().float().mean()),
             "matte_gt_mean": float(gt.detach().float().mean())}
    return AmortLoss(total=total, terms=terms, stats=stats)


class _null_ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _assert_terms(head, terms_raw: dict[str, torch.Tensor], x) -> None:
    """Runtime assertion (b): the configured terms are the terms produced.

    The proposal asks that the first ``steps.jsonl`` row carry all four ``L_*``
    columns.  ``aggregate`` builds those columns from exactly this dict, so the
    check is made HERE, on every micro-batch including the first, rather than
    read back out of a file after the run (``run_matte_arm.py`` also re-checks
    the written first row, which is the proposal's literal form).
    """
    want = set(head.losses)
    got = set(terms_raw)
    if got != want:
        raise AssertionError(
            f"{x.sample_id}: MATTE criterion produced {sorted(got)} but is "
            f"configured for {sorted(want)}; the L_* columns of steps.jsonl are "
            "built from this dict, so a mismatch is a missing pre-registered "
            "loss column")
    for name, v in terms_raw.items():
        if not torch.is_tensor(v):
            raise AssertionError(f"term {name} is not a tensor: {type(v)}")


def optimizer_spec(args):
    """AdamW, ViTMatte's values, recorded explicitly in ``run_setup.json``.

    ``grouping="dim"`` is the campaign's split and coincides with detectron2's
    ``weight_decay_norm = 0.0`` (``common/optim.py:23``): every norm weight/bias
    is 1-D and therefore lands in the zero-decay group.  The backbone's 0.65
    layer-wise decay (``configs/common/optimizer.py:25``) does not apply -- the
    base is frozen, and ``get_vit_lr_decay_rate`` returns 1.0 for every
    non-backbone parameter anyway (L15-21).
    """
    from .trainer import OptimizerSpec

    return OptimizerSpec(type="adamw",
                         lr=float(getattr(args, "lr", VITMATTE_LR)),
                         weight_decay=float(getattr(args, "weight_decay",
                                                    VITMATTE_WD)),
                         betas=VITMATTE_BETAS, grouping="dim")


def scheduler_kwargs(args, total_steps: int) -> dict[str, Any]:
    """MultiStep ``[1.0, 0.1, 0.05]`` at 30% / 90%, warmup 250/134687 (D-1).

    Every number is carried over as a FRACTION of ViTMatte's own horizon, so the
    schedule has the reference's shape on this campaign's 1200-step horizon.
    Returns ``{}`` unless the run is actually on a multistep kind, so the D-5
    alternative arm (repo optimiser + cosine) is not silently given a
    ``warmup_factor`` the cosine branch would also honour.
    """
    from q3vl.where.calibrate import scale_milestones

    kind = str(getattr(args, "scheduler", "cosine"))
    if kind not in ("multistep", "warmup_multistep"):
        return {}
    n = int(total_steps or 0)
    if n <= 0:
        raise SystemExit(
            "--scheduler multistep needs an explicit horizon: pass --max-steps "
            "(the proposal's 档 is 1200).  The milestones are 30%/90% of it and "
            "cannot be placed against an unknown total.")
    fracs = [m / VITMATTE_MAX_ITER for m in VITMATTE_MILESTONES]
    return {"values": list(VITMATTE_VALUES),
            "milestones": scale_milestones(fracs, n),
            "warmup_steps": int(round(n * VITMATTE_WARMUP_ITERS / VITMATTE_MAX_ITER)),
            "warmup_factor": VITMATTE_WARMUP_FACTOR}


def builder_kwargs(args) -> dict[str, Any]:
    """Pixel GT at spec-5 (the image grid), and ``gt_hi``/``guide_hi`` carried.

    ``want_hi=True`` buys three things at once: the published raster the D-9
    audit compares against, the raster a failed sample falls back to, and the
    luma the degraded image path reads.
    """
    return {"want_hi": True,
            "pixgt_size": (lambda gh, gw: (VISION_PATCH * int(gh),
                                           VISION_PATCH * int(gw)))}


# --------------------------------------------------------------------------- #
# the pre-registered board column
# --------------------------------------------------------------------------- #
def _rand_field(sample_id: str, shape: tuple[int, int], seed: int,
                device) -> torch.Tensor:
    """A uniform field from a PRIVATE generator.

    Not ``torch.rand``: the diagnostic control must not consume the global RNG,
    or the arm's own initialisation / dropout stream depends on how many
    diagnostic columns were switched on.
    """
    g = torch.Generator()
    g.manual_seed(int((seed ^ zlib.crc32(str(sample_id).encode())) & 0x7FFFFFFF))
    return torch.rand(shape, generator=g).to(device)


def per_sample_row(model, out: dict[str, Any], x) -> dict[str, Any]:
    """The pixel diagnostic columns.  Diagnostics only -- never the headline.

    Every column is emitted three times: the prediction, the centre prior, and a
    random top-k field of the matched area.  The proposal's rule is that the
    control numbers come out first and the column is enabled only if they did,
    so :func:`criteria_columns` counts a row only when all three are present.
    """
    head = model.geo
    if not getattr(head, "pix_diag", True):
        return {}
    alpha = out.get("alpha_pix")
    gt_pix = getattr(x, "gt_pix", None)
    if alpha is None or gt_pix is None:
        return {}
    a = alpha.detach()[0, 0].float()
    gt = gt_pix.detach().float()
    if tuple(gt.shape[-2:]) != tuple(a.shape[-2:]):
        gt = area_resize(gt.reshape(1, 1, *gt.shape[-2:]),
                         tuple(int(v) for v in a.shape[-2:]))[0, 0]
    h, w = int(a.shape[-2]), int(a.shape[-1])

    from .evaluate import center_prior_unit

    cp = center_prior_unit(h, w, device=a.device).float()
    k = gt_area_k(gt)
    rnd = topk_mask(_rand_field(x.sample_id, (h, w), head.control_seed, a.device), k)

    gt_amp = _grad_amp(gt, head.grad_sigma)
    row = {
        "pix_h": h, "pix_w": w,
        "pix_gt_source": str(getattr(x, "gt_pix_source", "") or ""),
        "pix_gt_area_frac": float((gt > 0.5).float().mean()),
        "pix_soft_iou": soft_iou_value(a, gt),
        "pix_soft_iou_center": soft_iou_value(cp, gt),
        "pix_soft_iou_random": soft_iou_value(rnd, gt),
        "pix_sad": pix_sad(a, gt),
        "pix_sad_center": pix_sad(cp, gt),
        "pix_sad_random": pix_sad(rnd, gt),
        "pix_grad": pix_gradient_error(a, gt_amp, head.grad_sigma),
        "pix_grad_center": pix_gradient_error(cp, gt_amp, head.grad_sigma),
        "pix_grad_random": pix_gradient_error(rnd, gt_amp, head.grad_sigma),
        "pix_alpha_mean": float(a.mean()),
        "pix_alpha_std": float(a.std()),
    }
    return row


_PIX_METRICS = ("pix_soft_iou", "pix_sad", "pix_grad")
_PIX_CONTROLS = ("center", "random")


def _agg(xs: Iterable[float]) -> dict[str, Any]:
    v = [float(t) for t in xs if t is not None and math.isfinite(float(t))]
    if not v:
        return {"n": 0}
    return {"n": len(v), "mean": float(np.mean(v)), "median": float(np.median(v)),
            "p10": percentile(v, 0.10), "p25": percentile(v, 0.25),
            "p75": percentile(v, 0.75), "p90": percentile(v, 0.90),
            "min": float(np.min(v)), "max": float(np.max(v))}


def _block(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(rows)}
    for m in _PIX_METRICS:
        out[m] = _agg(r.get(m) for r in rows)
        for c in _PIX_CONTROLS:
            out[f"{m}_{c}"] = _agg(r.get(f"{m}_{c}") for r in rows)
    return out


def criteria_columns(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """``board["criteria_columns"]["pix_readout"]``.

    ``assert_criteria_ran`` (``evaluate.py:331`` via ``arms.ARM_CRITERIA``)
    refuses a MATTE board whose ``n`` here is 0, which is the proposal's "n=0
    拒出板".  A row counts only if it carries the metric AND both controls --
    "对照数字先出、列才启用".
    """
    from .evaluate import AREA_BINS

    def complete(r: dict[str, Any]) -> bool:
        for m in _PIX_METRICS:
            if r.get(m) is None:
                return False
            for c in _PIX_CONTROLS:
                if r.get(f"{m}_{c}") is None:
                    return False
        return True

    live = [r for r in rows if complete(r)]
    col: dict[str, Any] = {
        **_block(live),
        "n_rows_seen": len(rows),
        "n_rows_incomplete": len(rows) - len(live),
        "by_family": {fam: _block([r for r in live if r.get("family") == fam])
                      for fam in sorted({str(r.get("family")) for r in live})},
        "by_area": {f"{lo:.2f}-{hi:.2f}": _block(
            [r for r in live if lo <= float(r.get("pix_gt_area_frac") or 0.0) < hi])
            for lo, hi in AREA_BINS},
        "gt_pix_source": _counts(r.get("pix_gt_source") for r in live),
        # pre-registered guard (c), carried into the board so it survives the
        # run: how much of the supervision really was the analytic target the
        # run claims, and what the D-9 spot check found.
        **_run_guard(),
        "note": ("pixel-resolution diagnostics at the head's own output "
                 "resolution: soft-IoU (min/max form), MAM SAD (full frame, "
                 "sum|d|/1000) and MAM Grad (sigma=1.4).  Each is reported "
                 "beside a centre-prior field and a matched-area random top-k "
                 "field.  DIAGNOSTIC ONLY: not the headline, not a loss, not a "
                 "checkpoint-selection input.  SAD scales with frame area and "
                 "is not comparable with published unknown-area SAD."),
    }
    return {"pix_readout": col}


def _run_guard() -> dict[str, Any]:
    """The run-level half of guard (c): GT provenance + the D-9 audit result.

    Empty (and explicitly so) when no head was built through
    :func:`build_head` -- a board that cannot show the guard says it cannot,
    rather than omitting the key and reading as a pass.
    """
    if not _LIVE_HEAD:
        return {"run_guard": {"available": False,
                              "reason": "no MatteHead was built in this process"}}
    f = _LIVE_HEAD[0].facts()
    return {"run_guard": {
        "available": True,
        "gt_pix_source_counts": f["gt_pix_source"],
        "gt_pix_audit": f["gt_pix_audit"],
        "img_source": f["img_source"],
        "img_source_counts": f["img_source_counts"],
        "n_samples": f["n_samples"], "n_fake": f["n_fake"],
        "n_empty_sample_map": f["n_empty_sample_map"],
        "note": ("counts accumulate over BOTH the training and the evaluation "
                 "builders of this process; gt_pix_audit is the D-9 spot check "
                 "(mean-abs vs the published .cgt-derived raster) and "
                 "n_fail == n_fallback_ids samples were routed back to it"),
    }}


def _counts(xs: Iterable[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in xs:
        key = str(v or "unknown")
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def train_stats(out: dict[str, Any], x) -> dict[str, Any]:
    """Extra per-sample training-row columns (the trainer's ``rows`` list)."""
    mt = out.get("matte") or {}
    return {"matte_img_source": mt.get("img_source"),
            "matte_out_hw": list(mt.get("out_hw") or ()),
            "matte_gt_pix_source": mt.get("gt_pix_source")}
