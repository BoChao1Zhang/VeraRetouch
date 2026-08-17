"""EPR-021 -- LIIF implicit-coordinate head (arm ``LIIF``).

A faithful port of **Learning Continuous Image Representation with Local
Implicit Image Function** (arXiv 2012.09161, ``github.com/yinboc/liif``) onto
the frozen Qwen3-VL feature grid.  The mask stops being "one field on the
``(gh, gw)`` grid" and becomes a function of a continuous coordinate: given
``x``, take the neighbouring deep features, the relative coordinate and the
query cell size, and one shared MLP returns the alpha at ``x``.

Every structural number below is quoted from the reference implementation
(files downloaded 2026-08-14 from the ``main`` branch and checked with
``nl -ba``); the line numbers in the comments are that file's::

    models/liif.py      L13-18 three switches, L22-29 in_dim, L46-48 unfold,
                        L50-59 local ensemble constants, L61-63 make_coord,
                        L69-83 four-corner nearest sample + rel_coord,
                        L86-90 cell decode, L92-105 area weighting
                        (L100-102: the two diagonal swaps)
    models/mlp.py       L9-18  Linear + ReLU, no Norm, no dropout
    utils.py            L102-117 make_coord: [-1,1], CELL CENTRES, (row, col)
    utils.py            L91-96  make_optimizer -> Adam(lr), wd = torch default 0
    train_liif.py       L91/L110 loss = nn.L1Loss() and nothing else,
                        L114-116 no clipping / no warmup / no autocast,
                        L83/L157-158 MultiStepLR
    test.py             L16-29 batched_predict, L73 pred.clamp_(0, 1)
    configs/train-div2k/train_edsr-baseline-liif.yaml
                        L12 scale_max 4, L14 sample_q 2304, L15 batch 16,
                        L33-35 gt sub 0.5 / div 0.5, L44-48 mlp [256]*4,
                        L50-53 adam 1e-4, L54-57 milestones [200,400,600,800]
                        gamma 0.5 over epoch_max 1000
    datasets/wrappers.py L62-68 uniform sampling without replacement,
                        L70-72 cell = (2/H_out, 2/W_out), L91-98/L107 s~U(1,4)
    models/edsr.py      L124/L169 edsr-baseline out_dim = n_feats = 64

Deviations from the reference recipe (each one is a row of the proposal's
"忠实移植配方表"; nothing else deviates):

* row 7  encoder -> a 1x1 Conv 1024->64 adapter on the frozen ``F_pre``;
* row 8  ``imnet`` input gains the 64-d language condition (LIIF is
  unconditional): 576 + 2 + 2 + 64 = 644;
* row 9  ``out_dim`` 1 (alpha) instead of 3 (RGB);
* rows 14/15 no LR crop, no augmentation;
* row 17 the MultiStep milestones are carried over as FRACTIONS of the run
  (20/40/60/80%), i.e. [240, 480, 720, 960] at the campaign's 1200 steps;
* row 18 effective batch 32 (the campaign's step-matching protocol);
* row 20 the encoder is FROZEN -- only the adapter, the condition projection
  and ``imnet`` train (protocol 3, ``q3vl/whereb/hiddens.py:161-163``);
* row 21 checkpoint selection is the quick-eval hard gate +
  ``local_soft_iou_median`` (never a val loss / PSNR --
  ``q3vl/whereb/amort/trainer.py:492-506``).

The head plugs into the six-arm registry (``q3vl/whereb/amort/arms.py``) and
touches no shared file.
"""

from __future__ import annotations

import contextlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from q3vl.where.upsample import area_resize
from q3vl.whereb.fields import no_autocast

from .losses import AmortLoss, bce_soft

__all__ = [
    "ARM", "CRITERIA", "DEFAULTS", "CFG", "config", "loss_preregistration",
    "SCHEDULER_KIND",
    "make_coord", "MLP", "GeomRegHead", "LIIFHead",
    "build_head", "forward", "compute_loss",
    "head_kwargs_from_args", "optimizer_spec", "scheduler_kwargs",
    "builder_kwargs", "per_sample_row", "criteria_columns", "train_stats",
    "add_arguments", "assert_first_step_row",
]

#: registry name and the pre-registered criterion column (arms.py:76, 88)
ARM = "LIIF"
CRITERIA = ("liif_grid_decode",)

#: ``train_liif.py:83/157-158`` MultiStepLR.  The kind is pre-registered, not a
#: knob: ``scheduler_kwargs`` refuses anything else, because ``make_scheduler``
#: ignores ``milestones``/``gamma`` under ``cosine`` while still honouring
#: ``warmup_steps`` -- a mixed schedule no board can show.
SCHEDULER_KIND = "multistep"

#: the geometry-regression ablation registers a SECOND column.  It cannot go
#: into ``CRITERIA`` -- ``arms.load_arm`` requires ``CRITERIA ==
#: ARM_CRITERIA["LIIF"]`` exactly, and ``ARM_CRITERIA`` is static while this
#: column is conditional on ``--geom-reg-weight > 0``.  It is therefore
#: produced by :func:`criteria_columns` and asserted there, by this module.
GEOM_CRITERION = "geom_reg_route"


# --------------------------------------------------------------------------- #
# configuration: the wrapper's flags reach the hooks through here
# --------------------------------------------------------------------------- #
#: every default is the reference value (or the proposal's written-down
#: conservative default where LIIF has nothing to copy)
DEFAULTS: dict[str, Any] = {
    # -- LIIF proper -------------------------------------------------------
    "sample_q": 2304,                       # yaml L14
    "hidden": (256, 256, 256, 256),         # yaml L48
    "feat_dim": 64,                         # edsr.py L169 (0 = no adapter)
    "cond_dim": 64,                         # NOVEL (LIIF is unconditional)
    "scale_min": 1.0,                       # wrappers.py L91-98
    "scale_max": 4.0,                       # yaml L12
    "local_ensemble": True,                 # liif.py L13-18
    "feat_unfold": True,
    "cell_decode": True,
    "eval_bsize": 65536,                    # test.py L16-29
    "loss": "l1",                           # train_liif.py L91
    "lr": 1e-4,                             # yaml L50-53
    "gt": "closed",                         # closed | cgt (= published raster)
    # -- pixel diagnostic columns (proposal 判据段, not in the headline) ----
    "pix_diag": True,
    # -- ablation row (1): geometry-parameter regression head, default OFF --
    "geom_reg_weight": 0.0,                 # 0 = the branch is NOT constructed
    "geom_reg_type_weight": 1.0,            # BCE on the mask_type logit
    "geom_angle_mask_ratio": 1.35,          # 待决策 (c) conservative default
    "geom_band_rx": 1.2,                    # 待决策 (d) conservative default
    # -- provenance / bookkeeping -----------------------------------------
    "pixgt_scale": 16,                      # (gh,gw) -> (16gh,16gw) = short 512
    "amount_ref": 256,                      # 待决策 (f): k x k reference grid

    "seed": 20260810,
    "run_dir": None,
}

#: filled by ``q3vl/whereb/scripts/run_liifhead_arm.py`` before it delegates to
#: ``run_amort_arm.main``.  The optional hooks below are handed the BASE
#: parser's namespace (``run_amort_arm.py:832-834`` passes ``args``, not
#: ``arms.ARM_ARGS``), so the arm's own flags have to travel out of band --
#: same seam ``uniq4.VARIANT4`` uses, and the sha256 freeze covers it.
CFG: dict[str, Any] = {}


def config() -> dict[str, Any]:
    """The resolved configuration: defaults with the wrapper's overrides on top."""
    return {**DEFAULTS, **CFG}


def loss_preregistration(args: Any = None) -> dict[str, Any]:
    """``arms.RUN_REQUIRED_HOOKS``: what ``loss_preregistration.json`` says.

    ``run_amort_arm``'s shared record is the live arms' seven-term ST_LANG
    stack, which this arm never enters (``trainer.py:244-251``).  LIIF's
    reference recipe is ONE term.
    """
    c = config()
    l1 = str(c["loss"]) == "l1"
    gw = float(c["geom_reg_weight"])
    form = ("L = mean |y_hat - (2g - 1)|   [train_liif.py:91/110, nn.L1Loss]"
            if l1 else
            "L = BCE_soft(sigmoid(y_hat), g)   [ablation ⑥, replaces L1]")
    if gw > 0:
        form += f" + {gw} * L_geom"
    return {
        "arm": ARM,
        "form": form,
        "points": (f"{int(c['sample_q'])} query coordinates sampled uniformly "
                   "without replacement per sample; the loss lives on the "
                   "points, never on the grid read-out "
                   "[datasets/wrappers.py:62-68]"),
        "target": ("the pixel GT sampled at the same coordinates "
                   f"(gt = {c['gt']!r}, pixgt_scale {int(c['pixgt_scale'])}x)"),
        "geom_reg": (f"L1 on the normalised geometry parameters + "
                     f"{float(c['geom_reg_type_weight'])} * BCE on the "
                     "mask_type logit; analytic families only, semantic and "
                     "store misses excluded from the denominator")
        if gw > 0 else "off (geom_reg_weight = 0, the branch is not built)",
        "terms": ["L_l1_pts" if l1 else "L_bce_pts"] + (["L_geom"] if gw > 0 else []),
        "seven_term_stack": "not entered (trainer.py:244-251)",
        "iou_as_field_target": False,
        "iou_as_selection_target": False,
        "dice_as_target": False,
        "pixel_diagnostic": ("soft-IoU / band-MAE at the pixel grid are "
                             "DIAGNOSTIC columns, never a loss term"),
    }


# --------------------------------------------------------------------------- #
# LIIF primitives
# --------------------------------------------------------------------------- #
def make_coord(shape: Sequence[int], *, flatten: bool = True, device=None,
               dtype=torch.float32) -> torch.Tensor:
    """``utils.py:102-117`` with ``ranges=None``: cell centres in ``[-1, 1]``.

    Order is ``(row, col)`` -- LIIF flips to ``(x, y)`` only immediately before
    ``grid_sample`` (``liif.py:74/78``).  ``r = (v1 - v0) / (2n)`` and
    ``seq = v0 + r + 2r * arange(n)`` are copied literally; the cell-centre
    convention is what makes the 4x read-out land on the sub-cells of the
    ``area_resize`` the criterion GT uses.
    """
    coord_seqs = []
    for n in shape:
        v0, v1 = -1, 1
        r = (v1 - v0) / (2 * int(n))
        seq = v0 + r + (2 * r) * torch.arange(int(n), device=device, dtype=dtype)
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing="ij"), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


class MLP(nn.Module):
    """``models/mlp.py:9-18`` -- Linear + ReLU, **no Norm, no dropout**.

    Initialisation is PyTorch's default on purpose: the reference does not
    initialise ``nn.Linear`` at all, so neither does this (proposal §3
    "初始化（step0 是什么）").
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_list: Sequence[int]):
        super().__init__()
        layers: list[nn.Module] = []
        lastv = int(in_dim)
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, int(hidden)))
            layers.append(nn.ReLU())
            lastv = int(hidden)
        layers.append(nn.Linear(lastv, int(out_dim)))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape[:-1]
        x = self.layers(x.reshape(-1, x.shape[-1]))
        return x.view(*shape, -1)


# --------------------------------------------------------------------------- #
# ablation row (1): geometry-parameter regression head
# --------------------------------------------------------------------------- #
#: continuous columns per ``mask_type``.  Coordinates stay in
#: ``raster_geometry``'s own [0,1] normalisation (``canonical_masks.py:99-100``)
#: and are NOT squashed: ``band``'s ``rx`` is the constant 1.6
#: (``subject_geom.py:22``), i.e. outside [0,1], so a sigmoid/clamp on the
#: regression output would make the band family unrepresentable.
CIRC_COLS: tuple[str, ...] = ("cx", "cy", "rx", "ry", "sin2a", "cos2a", "feather")
GRAD_COLS: tuple[str, ...] = ("zx", "zy", "fx", "fy")


class GeomRegHead(nn.Module):
    """``c -> MLP(64 -> 64 -> P)``, ReLU, **last layer zero-initialised**.

    ``P`` is split into two groups by ``mask_type`` (proposal 消融行 ①): the
    ``circulargradient`` group (7 continuous + 1 ``Flipped`` logit) and the
    ``gradient`` group (4 continuous + 1 ``Flipped`` logit), plus the single
    ``gradient`` vs ``circulargradient`` routing logit that replaces the cls
    head the faithful LIIF recipe does not have.

    Zero-initialising the three output layers is the reason this branch can be
    switched on without perturbing step 0 of the main loss.
    """

    def __init__(self, cond_dim: int = 64, hidden: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(int(cond_dim), int(hidden)), nn.ReLU())
        self.circ = nn.Linear(int(hidden), len(CIRC_COLS) + 1)
        self.grad = nn.Linear(int(hidden), len(GRAD_COLS) + 1)
        self.type_logit = nn.Linear(int(hidden), 1)
        for lin in (self.circ, self.grad, self.type_logit):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, c: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(c.reshape(1, -1))
        circ = self.circ(h)[0]
        grad = self.grad(h)[0]
        return {
            "circ_cont": circ[: len(CIRC_COLS)],
            "circ_flip": circ[len(CIRC_COLS)],
            "grad_cont": grad[: len(GRAD_COLS)],
            "grad_flip": grad[len(GRAD_COLS)],
            "type_logit": self.type_logit(h)[0, 0],
        }


def _gvalue(geometry: Mapping[str, Any], name: str, default: float = 0.0) -> float:
    """``canonical_masks.py:90-94``'s own parser (values arrive as strings)."""
    try:
        return float(str(geometry.get(name, default)).lstrip("+"))
    except (TypeError, ValueError):
        return float(default)


def geom_targets(mask_type: str, geometry: Mapping[str, Any], *,
                 angle_mask_ratio: float = 1.35) -> dict[str, Any]:
    """The regression targets of one sample, in ``raster_geometry``'s units.

    ``Angle`` is encoded as ``sin(2θ) / cos(2θ)`` because θ and θ+180° give the
    same ellipse (``canonical_masks.py:107-108``), and both columns are MASKED
    when the ellipse is near-isotropic: the construction side draws the angle
    from ``U(-90, 90)`` when ``natural_elongation < 1.15``
    (``subject_geom.py:99-100``), and ``MIN_ELONG = 1.35`` pins every such
    sample at ``rx/ry == 1.35`` (L97-98), so ``rx/ry <= 1.35 + 1e-3`` is the
    only separable proxy the stored parameters allow.  Masking is deliberately
    conservative (it also masks the 1.15-1.35 band, whose angle carries a ±18°
    jitter) -- proposal 待决策 (c).
    """
    if mask_type == "circulargradient":
        left, right = _gvalue(geometry, "Left"), _gvalue(geometry, "Right")
        top, bottom = _gvalue(geometry, "Top"), _gvalue(geometry, "Bottom")
        cx, cy = (left + right) / 2.0, (top + bottom) / 2.0
        rx = max(abs(right - left) / 2.0, 1e-3)
        ry = max(abs(bottom - top) / 2.0, 1e-3)
        theta = math.radians(_gvalue(geometry, "Angle"))
        feather = max(_gvalue(geometry, "Feather", 50.0) / 100.0, 0.05)
        ratio = max(rx, ry) / max(min(rx, ry), 1e-9)
        angle_ok = bool(ratio > float(angle_mask_ratio) + 1e-3)
        vals = [cx, cy, rx, ry, math.sin(2 * theta), math.cos(2 * theta), feather]
        mask = [1.0, 1.0, 1.0, 1.0, float(angle_ok), float(angle_ok), 1.0]
        return {"mask_type": mask_type, "cols": CIRC_COLS, "values": vals,
                "col_mask": mask, "angle_masked": not angle_ok,
                "flipped": _flipped(geometry), "rx": rx, "ry": ry}
    if mask_type == "gradient":
        vals = [_gvalue(geometry, "ZeroX"), _gvalue(geometry, "ZeroY"),
                _gvalue(geometry, "FullX", 1.0), _gvalue(geometry, "FullY")]
        return {"mask_type": mask_type, "cols": GRAD_COLS, "values": vals,
                "col_mask": [1.0] * len(GRAD_COLS), "angle_masked": False,
                "flipped": _flipped(geometry), "rx": None, "ry": None}
    raise ValueError(f"unsupported mask_type {mask_type!r}")


def _flipped(geometry: Mapping[str, Any]) -> float:
    return float(str(geometry.get("Flipped", "false")).lower().lstrip("+") == "true")


# --------------------------------------------------------------------------- #
# the head
# --------------------------------------------------------------------------- #
class LIIFHead(nn.Module):
    """``F_pre`` + ``h_cond`` -> alpha at any continuous coordinate.

    Trainable, and nothing else is (the VLM is frozen by protocol 3)::

        1x1 Conv 1024 -> 64                       65,600
        LayerNorm(2560) + Linear(2560 -> 64)     169,024
        imnet 644 -> 256 -> 256 -> 256 -> 256 -> 1   362,753
                                              --------------
                                                 597,377
    """

    def __init__(
        self,
        *,
        in_dim: int = 1024,
        text_dim: int = 2560,
        feat_dim: int = 64,
        cond_dim: int = 64,
        hidden: Sequence[int] = (256, 256, 256, 256),
        out_dim: int = 1,
        local_ensemble: bool = True,
        feat_unfold: bool = True,
        cell_decode: bool = True,
        sample_q: int = 2304,
        scale_min: float = 1.0,
        scale_max: float = 4.0,
        eval_bsize: int = 65536,
        loss: str = "l1",
        pix_diag: bool = True,
        geom_reg_weight: float = 0.0,
        geom_reg_type_weight: float = 1.0,
        geom_angle_mask_ratio: float = 1.35,
        geom_band_rx: float = 1.2,
        seed: int = 20260810,
    ):
        super().__init__()
        if loss not in ("l1", "bce"):
            raise ValueError(f"--liif-loss must be l1|bce, got {loss!r}")
        self.in_dim = int(in_dim)
        self.text_dim = int(text_dim)
        self.feat_dim = int(feat_dim)
        self.cond_dim = int(cond_dim)
        self.hidden = tuple(int(h) for h in hidden)
        self.local_ensemble = bool(local_ensemble)
        self.feat_unfold = bool(feat_unfold)
        self.cell_decode = bool(cell_decode)
        self.sample_q = int(sample_q)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.eval_bsize = int(eval_bsize)
        self.loss_kind = str(loss)
        self.pix_diag = bool(pix_diag)
        self.geom_reg_weight = float(geom_reg_weight)
        self.geom_reg_type_weight = float(geom_reg_type_weight)
        self.geom_angle_mask_ratio = float(geom_angle_mask_ratio)
        self.geom_band_rx = float(geom_band_rx)
        self.seed = int(seed)

        # row 7: the encoder LIIF trains is frozen here, so the only thing that
        # can adapt 1024 -> the reference's 64 is this 1x1 conv.  feat_dim = 0
        # is ablation row (7): F_pre goes in raw.
        self.adapter = (nn.Conv2d(self.in_dim, self.feat_dim, 1)
                        if self.feat_dim > 0 else None)
        ch = self.feat_dim if self.feat_dim > 0 else self.in_dim

        # NOVEL: LIIF has no conditioning at all.  One LayerNorm + Linear, and
        # the 64-d result is concatenated to EVERY point's imnet input.
        self.cond_proj = (nn.Sequential(nn.LayerNorm(self.text_dim),
                                        nn.Linear(self.text_dim, self.cond_dim))
                          if self.cond_dim > 0 else None)

        # liif.py:22-29, plus the condition
        imnet_in_dim = ch * (9 if self.feat_unfold else 1)
        imnet_in_dim += 2                                   # attach coord
        if self.cell_decode:
            imnet_in_dim += 2
        imnet_in_dim += self.cond_dim
        self.imnet_in_dim = int(imnet_in_dim)
        self.imnet = MLP(self.imnet_in_dim, int(out_dim), self.hidden)

        self.geom = (GeomRegHead(self.cond_dim) if self.geom_reg_weight > 0
                     else None)

        # Point sampling draws from its OWN generator, seeded once: the draw
        # must not consume the global torch stream (which would make every
        # other stochastic component of the run depend on how many points this
        # head happened to sample) and must be reproducible from the seed alone.
        self._gen = torch.Generator()
        self._gen.manual_seed(self.seed)
        # counters -> facts(); every fallback is counted, never silent
        self.counts: dict[str, int] = {}
        self._loss_keys_asserted = False

    # -- bookkeeping -------------------------------------------------------
    def _count(self, key: str, n: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + int(n)

    @property
    def loss_term_key(self) -> str:
        """``l1_pts`` for the faithful recipe, ``bce_pts`` for ablation ⑥."""
        return "l1_pts" if self.loss_kind == "l1" else "bce_pts"

    def expected_loss_columns(self) -> list[str]:
        cols = [f"L_{self.loss_term_key}"]
        if self.geom is not None:
            cols.append("L_geom")
        return cols

    def n_params(self) -> dict[str, int]:
        def _n(m) -> int:
            return 0 if m is None else int(sum(p.numel() for p in m.parameters()))

        return {"adapter": _n(self.adapter), "cond_proj": _n(self.cond_proj),
                "imnet": _n(self.imnet), "geom_reg": _n(self.geom),
                "total": int(sum(p.numel() for p in self.parameters()))}

    def facts(self) -> dict[str, Any]:
        return {
            "arm": ARM,
            "criterion": CRITERIA[0],
            "in_dim": self.in_dim, "feat_dim": self.feat_dim,
            "cond_dim": self.cond_dim, "text_dim": self.text_dim,
            "imnet_in_dim": self.imnet_in_dim, "hidden": list(self.hidden),
            "local_ensemble": self.local_ensemble,
            "feat_unfold": self.feat_unfold,
            "cell_decode": self.cell_decode,
            "sample_q": self.sample_q,
            "scale": [self.scale_min, self.scale_max],
            "eval_scale": self.scale_max,
            "eval_bsize": self.eval_bsize,
            "loss": self.loss_kind, "loss_term": self.loss_term_key,
            "activation": ("affine(0.5y+0.5)" if self.loss_kind == "l1"
                           else "sigmoid"),
            "pix_diag": self.pix_diag,
            "geom_reg": (None if self.geom is None else {
                "weight": self.geom_reg_weight,
                "type_weight": self.geom_reg_type_weight,
                "angle_mask_ratio": self.geom_angle_mask_ratio,
                "band_rx_threshold": self.geom_band_rx,
                "circ_cols": list(CIRC_COLS), "grad_cols": list(GRAD_COLS),
                "criterion": GEOM_CRITERION}),
            "params": self.n_params(),
            "seed": self.seed,
            "counts": dict(sorted(self.counts.items())),
            # LIIF trains its encoder (train_liif.py:77-78); this arm does not.
            "encoder_trained": False,
        }

    # -- conditioning ------------------------------------------------------
    def cond_vector(self, h_cond: torch.Tensor | None, device=None) -> torch.Tensor:
        """``(K, 2560) -> (cond_dim,)``.

        ``K > 1`` (``--readout-nseg`` / ``--readout-qtok``) has no reference:
        the pre-registered NOVEL default is "each row through the SAME
        LayerNorm+Linear, then the mean", so ``imnet``'s input width stays 644.
        """
        if self.cond_proj is None:
            return torch.zeros(0, device=device)
        if h_cond is None:
            self._count("null_cond_zero")
            return torch.zeros(self.cond_dim, device=device)
        h = h_cond.reshape(-1, self.text_dim).to(dtype=torch.float32)
        if device is not None:
            h = h.to(device)
        if h.shape[0] > 1:
            self._count(f"cond_rows_{h.shape[0]}")
        return self.cond_proj(h).mean(dim=0)

    # -- the LIIF query ----------------------------------------------------
    def prepare(self, feat: torch.Tensor) -> dict[str, Any]:
        """``gen_feat`` (liif.py:33-35) + the unfold of L46-48, done once.

        ``test.py:16-29`` re-runs the unfold per chunk; hoisting it is
        numerically identical and is the only reason a 512x768 diagnostic
        decode is affordable.
        """
        if feat.dim() != 4 or feat.shape[0] != 1:
            raise ValueError(f"expected (1, C, gh, gw) features, got "
                             f"{tuple(feat.shape)}")
        f = self.adapter(feat) if self.adapter is not None else feat
        if self.feat_unfold:
            f = F.unfold(f, 3, padding=1).view(
                f.shape[0], f.shape[1] * 9, f.shape[2], f.shape[3])
        feat_coord = (make_coord(f.shape[-2:], flatten=False, device=f.device,
                                 dtype=f.dtype)
                      .permute(2, 0, 1).unsqueeze(0)
                      .expand(f.shape[0], 2, *f.shape[-2:]))
        return {"feat": f, "feat_coord": feat_coord,
                "h": int(feat.shape[-2]), "w": int(feat.shape[-1])}

    def query(self, prep: dict[str, Any], coord: torch.Tensor,
              cell: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """``liif.py:37-106`` verbatim, with ``c`` appended to every point.

        ``coord`` / ``cell`` are ``(1, Q, 2)`` in ``[-1, 1]``, ``(row, col)``.
        Returns ``(1, Q, out_dim)`` -- the RAW decoder output, exactly what
        LIIF's L1 is charged on (no sigmoid, no clamp; the clamp belongs to the
        inference path, ``test.py:73``).
        """
        feat = prep["feat"]
        feat_coord = prep["feat_coord"]
        if self.local_ensemble:
            vx_lst, vy_lst, eps_shift = [-1, 1], [-1, 1], 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1]) -- liif.py:57-59
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        preds, areas = [], []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode="nearest", align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode="nearest", align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                if self.cond_dim:
                    inp = torch.cat(
                        [inp, c.reshape(1, 1, -1).expand(inp.shape[0],
                                                         inp.shape[1], -1)],
                        dim=-1)

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            # liif.py:100-102 -- the two diagonal swaps.  Dropping them weights
            # each corner by its OWN area instead of the opposite one, i.e. the
            # bilinear weights come out inverted.
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret

    def batched_query(self, prep: dict[str, Any], coord: torch.Tensor,
                      cell: torch.Tensor, c: torch.Tensor,
                      bsize: int | None = None) -> torch.Tensor:
        """``test.py:16-29``'s ``batched_predict`` -- chunked, same result."""
        n = coord.shape[1]
        bsize = int(bsize or self.eval_bsize)
        if n <= bsize:
            return self.query(prep, coord, cell, c)
        out = []
        ql = 0
        while ql < n:
            qr = min(ql + bsize, n)
            out.append(self.query(prep, coord[:, ql:qr, :], cell[:, ql:qr, :], c))
            ql = qr
        return torch.cat(out, dim=1)

    def alpha_of(self, y: torch.Tensor) -> torch.Tensor:
        """Decoder output -> alpha in [0, 1].

        ``l1``: the inverse of the reference's target normalisation
        (``yaml:33-35`` ``sub 0.5 / div 0.5``) followed by ``test.py:73``'s
        ``clamp_(0, 1)``.  ``bce`` (ablation ⑥) reads the same output as a
        logit.
        """
        if self.loss_kind == "bce":
            return torch.sigmoid(y)
        return torch.clamp(0.5 * y + 0.5, 0.0, 1.0)

    def decode_grid(self, prep: dict[str, Any], c: torch.Tensor,
                    qh: int, qw: int, *, bsize: int | None = None
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode the cell centres of a ``(qh, qw)`` query grid.

        Returns ``(alpha, y_raw)``, both ``(qh, qw)``.  ``cell = (2/qh, 2/qw)``
        (``wrappers.py:70-72``).
        """
        dev = prep["feat"].device
        dt = prep["feat"].dtype
        coord = make_coord((qh, qw), device=dev, dtype=dt).unsqueeze(0)
        cell = torch.ones_like(coord)
        cell[:, :, 0] *= 2 / int(qh)
        cell[:, :, 1] *= 2 / int(qw)
        y = self.batched_query(prep, coord, cell, c, bsize)[0, :, 0]
        return self.alpha_of(y).view(int(qh), int(qw)), y.view(int(qh), int(qw))

    # -- training-time point sampling --------------------------------------
    def draw_scale(self) -> float:
        """``s ~ U(scale_min, scale_max)`` (``wrappers.py:107``), own RNG."""
        u = torch.rand((), generator=self._gen).item()
        return self.scale_min + (self.scale_max - self.scale_min) * u

    def sample_points(self, gh: int, gw: int) -> dict[str, Any]:
        """``s ~ U(1,4)``, then ``sample_q`` cell centres of ``(gh*s, gw*s)``
        drawn uniformly WITHOUT replacement (``wrappers.py:62-68``).

        DEVIATION, forced and counted: LIIF's HR crop is 48s x 48s >= 2304
        cells, so ``np.random.choice(..., replace=False)`` always has enough
        points.  Here the query grid is ``round(gh*s) x round(gw*s)``, which at
        ``s = 1`` is 32x48 = 1536 < 2304.  Sampling is therefore
        ``min(sample_q, qh*qw)`` -- taking every cell once rather than raising
        or silently sampling with replacement.  ``n_pts`` is logged per step.
        """
        s = self.draw_scale()
        qh = max(1, int(round(gh * s)))
        qw = max(1, int(round(gw * s)))
        n = qh * qw
        q = min(int(self.sample_q), n)
        if q < self.sample_q:
            self._count("sample_q_short")
        idx = torch.randperm(n, generator=self._gen)[:q]
        coord = make_coord((qh, qw))[idx]              # CPU, (q, 2) (row, col)
        cell = torch.ones_like(coord)
        cell[:, 0] *= 2 / qh
        cell[:, 1] *= 2 / qw
        return {"s": float(s), "qh": qh, "qw": qw, "n_pts": int(q),
                "coord": coord, "cell": cell}

    def gt_points(self, sample: Any, coord: torch.Tensor) -> torch.Tensor:
        """GT alpha at ``coord`` (``(q,2)``, ``(row,col)`` in ``[-1,1]``, CPU).

        Closed form for the three analytic families, bilinear on the published
        raster otherwise -- both through :class:`q3vl.whereb.amort.pixgt.PixGT`,
        whose ``points()`` takes ``(x, y)`` cell-centre coordinates in [0,1].

        ``is_fake`` (foreign instruction, p = 0.15, ``losses.py:74``):
        ``g(x_p) = 0`` exactly, so the L1 target is the finite constant -1
        (proposal §2, written-down policy; counted, not silent).
        """
        q = coord.shape[0]
        if bool(getattr(sample, "is_fake", False)):
            self._count("fake_zero_gt")
            return torch.zeros(q, dtype=torch.float32)
        pg = getattr(sample, "pixgt", None)
        if pg is None:
            raise ValueError(
                "arm LIIF supervises on point-sampled pixel GT and the batch "
                "builder supplied no PixGT.  Drop --no-pixgt (run_amort_arm "
                "builds a PixGTProvider for every EPR-018..023 arm).")
        xy = torch.stack([(coord[:, 1] + 1.0) / 2.0,
                          (coord[:, 0] + 1.0) / 2.0], dim=-1)
        self._count(f"gt_{pg.source}")
        return pg.points(xy).reshape(-1).to(torch.float32)


# --------------------------------------------------------------------------- #
# arms.py hooks
# --------------------------------------------------------------------------- #
def build_head(*, in_dim: int, text_dim: int, args: Any = None, **kw) -> nn.Module:
    cfg = {**config(), **kw}
    return LIIFHead(
        in_dim=int(in_dim), text_dim=int(text_dim),
        feat_dim=int(cfg["feat_dim"]), cond_dim=int(cfg["cond_dim"]),
        hidden=tuple(cfg["hidden"]),
        local_ensemble=bool(cfg["local_ensemble"]),
        feat_unfold=bool(cfg["feat_unfold"]),
        cell_decode=bool(cfg["cell_decode"]),
        sample_q=int(cfg["sample_q"]),
        scale_min=float(cfg["scale_min"]), scale_max=float(cfg["scale_max"]),
        eval_bsize=int(cfg["eval_bsize"]), loss=str(cfg["loss"]),
        pix_diag=bool(cfg["pix_diag"]),
        geom_reg_weight=float(cfg["geom_reg_weight"]),
        geom_reg_type_weight=float(cfg["geom_reg_type_weight"]),
        geom_angle_mask_ratio=float(cfg["geom_angle_mask_ratio"]),
        geom_band_rx=float(cfg["geom_band_rx"]),
        seed=int(getattr(args, "seed", cfg["seed"]) if args is not None
                 else cfg["seed"]),
    )


def head_kwargs_from_args(args: Any) -> dict[str, Any]:
    """``build_head`` kwargs from the wrapper's config (``ARM_KWARGS`` wins)."""
    return {}


def optimizer_spec(args: Any):
    """``utils.py:91-96`` + ``yaml:50-53``: ``Adam(params, lr=1e-4)``.

    No weight decay (torch's default 0), no parameter groups, no warmup, no
    gradient clipping (``train_liif.py:114-116``; the wrapper passes
    ``--max-grad-norm 0``, which the trainer reads as "clipping off").
    """
    from .trainer import OptimizerSpec

    return OptimizerSpec(type="adam", lr=float(config()["lr"]),
                         weight_decay=0.0, grouping="none")


def scheduler_kwargs(args: Any, total_steps: int) -> dict[str, Any]:
    """``MultiStepLR([200,400,600,800], gamma 0.5)`` over ``epoch_max 1000``.

    Carried over as fractions (20/40/60/80%), which at the campaign's 1200
    steps is ``[240, 480, 720, 960]``.  ``warmup_steps=1`` is how
    ``make_scheduler`` spells "no warmup": its warmup floor is 1 step and
    ``lr_lambda(0) = (0+1)/1 = 1.0``, so the LR starts at 1e-4 exactly as
    ``train_liif.py`` does.  The wrapper passes ``--scheduler multistep``.
    """
    from q3vl.where.calibrate import scale_milestones

    # The *kind* lives on ``--scheduler``, which this hook cannot set, so it is
    # asserted here (same guard as ``segsam.py:1160-1166``).  Under ``cosine``
    # ``make_scheduler`` ignores the ``milestones`` / ``gamma`` below while
    # still honouring ``warmup_steps``: a run started around
    # ``run_liifhead_arm.py`` (which pins the kind) would train on
    # cosine-with-a-1-step-warmup and publish a board that looks like the
    # MultiStep recipe.
    kind = str(getattr(args, "scheduler", SCHEDULER_KIND) or SCHEDULER_KIND)
    if kind != SCHEDULER_KIND:
        raise SystemExit(
            f"arm LIIF needs --scheduler {SCHEDULER_KIND!r} (MultiStepLR "
            "[240,480,720,960] gamma 0.5, train_liif.py:83/157-158) but the "
            f"run was given {kind!r}.  run_liifhead_arm.py pins it; do not "
            "pass --scheduler by hand.")
    if int(total_steps) <= 0:
        raise ValueError(
            "arm LIIF needs an explicit step horizon to place its MultiStep "
            "milestones on (--max-steps; the campaign's step-matched value is "
            "1200).  Refusing to run an unmatched schedule.")
    return {"milestones": scale_milestones((0.2, 0.4, 0.6, 0.8), int(total_steps)),
            "gamma": 0.5, "warmup_steps": 1}


def builder_kwargs(args: Any) -> dict[str, Any]:
    """Pixel GT at the spec-5 pixel grid instead of the 4x default.

    ``AmortBatchBuilder``'s default is ``(4gh, 4gw)`` (``data.py:423``).  This
    arm asks for ``(16gh, 16gw)`` = short side 512, for two reasons the
    proposal states: the semantic family's point GT is a BILINEAR read of the
    published ``.maskhi.png`` at its own resolution (§3 ③), and the pixel
    diagnostic columns are defined at 512 (判据段).
    """
    k = int(config()["pixgt_scale"])
    return {"pixgt_size": (lambda gh, gw: (k * int(gh), k * int(gw)))}


def forward(model, head: LIIFHead, ctx) -> dict[str, Any]:
    """``AmortModel.forward_geo`` -> the arm (``arms.py:29-32``).

    Always returns ``m_low``: the 4x grid decode projected back with
    ``area_resize`` -- the SAME operator ``gt_low`` is built with
    (``data.py:685-686``), so the criterion measures both with one ruler.
    In training it additionally returns the sampled points and their GT.
    """
    x = ctx.sample
    feat = ctx.feat
    dev = feat.device
    training = bool(head.training)
    with no_autocast(dev.type):
        h_cond = ctx.h_cond
        if h_cond is None and getattr(x, "readout", None) is None:
            # no readout builder at all = a configuration error, not a miss
            ctx.require_cond(ARM)
        c = head.cond_vector(h_cond, device=dev)
        prep = head.prepare(feat.float())
        gh, gw = int(ctx.grid_h), int(ctx.grid_w)
        if (prep["h"], prep["w"]) != (gh, gw):
            raise AssertionError(
                f"feature grid {(prep['h'], prep['w'])} != declared grid "
                f"{(gh, gw)}")
        out: dict[str, Any] = {}

        if training:
            pts = head.sample_points(gh, gw)
            g = head.gt_points(x, pts["coord"]).to(dev)
            coord = pts["coord"].to(device=dev, dtype=prep["feat"].dtype)
            cell = pts["cell"].to(device=dev, dtype=prep["feat"].dtype)
            out["y_pts"] = head.query(prep, coord.unsqueeze(0),
                                      cell.unsqueeze(0), c)[0, :, 0]
            out["g_pts"] = g
            out["liif_train"] = {"s": pts["s"], "qh": pts["qh"], "qw": pts["qw"],
                                 "n_pts": pts["n_pts"],
                                 "gt_pix_source": getattr(x, "gt_pix_source", "")}

        # the read-out that the board is scored on.  In training it is a
        # diagnostic only (the loss lives on the points), so it is built under
        # no_grad -- keeping it in the graph would double the head's memory for
        # a tensor nothing differentiates.
        qh = max(1, int(round(gh * head.scale_max)))
        qw = max(1, int(round(gw * head.scale_max)))
        with torch.no_grad() if training else contextlib.nullcontext():
            alpha, y_raw = head.decode_grid(prep, c, qh, qw)
            m_low = area_resize(alpha[None, None], (gh, gw))[0, 0]
            s_low = area_resize(y_raw[None, None], (gh, gw))[0, 0]
        out["m_low"] = m_low
        out["s_low"] = s_low
        out["params"] = {}
        out["liif_decode"] = {"scale": float(head.scale_max), "qh": qh, "qw": qw,
                              "cell": [2.0 / qh, 2.0 / qw],
                              "grid_h": gh, "grid_w": gw,
                              "projection": "area_resize",
                              "activation": ("sigmoid" if head.loss_kind == "bce"
                                             else "clamp(0.5y+0.5,0,1)")}

        if head.geom is not None:
            out["geom_pred"] = head.geom(c)
            out["geom_target"] = _geom_target_of(head, x)

        if (not training) and head.pix_diag:
            # 「结果落盘先于可选阶段」: the full-frame 512-grid decode is a
            # DIAGNOSTIC (never the headline -- the board is scored on `m_low`,
            # which is already in `out`).  An OOM or a malformed `gt_pix` inside
            # it must cost the six `pix_*` columns of ONE row, not the eval row
            # and not the run.  Same shape as `segsam.py:1231-1256` /
            # `prnd.py:1093-1099`: the failure is recorded on the row and
            # counted, never swallowed.
            try:
                out["liif_pix"] = _pixel_diagnostic(head, prep, c, x)
            except Exception as exc:  # noqa: BLE001 -- recorded, never silent
                out["liif_pix"] = None
                out["liif_pix_error"] = f"{type(exc).__name__}: {exc}"
                head._count("pix_diag_error")
                print(f"WARN arm LIIF: pixel diagnostic failed on "
                      f"{getattr(x, 'sample_id', '?')}: {out['liif_pix_error']} "
                      "-- pix_* columns are None for this row, the headline "
                      "read-out (m_low) is unaffected", flush=True)
    return out


def _geom_target_of(head: LIIFHead, x: Any) -> dict[str, Any] | None:
    """Targets from the sample's own ``PixGT`` (which carries the sqlite row).

    ``PixGTProvider.geometry_of`` already resolved ``ConstructGeomStore``
    (``data.py:96-143``) for the analytic families; ``semantic`` and store
    misses have no parameters and are excluded from ``L_geom``'s denominator
    (proposal 消融行 ①), counted rather than zero-filled.
    """
    pg = getattr(x, "pixgt", None)
    if pg is None or not getattr(pg, "mask_type", None) or pg.geometry is None:
        head._count("geom_target_missing")
        return None
    if bool(getattr(x, "is_fake", False)):
        head._count("geom_target_fake_skipped")
        return None
    head._count("geom_target")
    return geom_targets(pg.mask_type, pg.geometry,
                        angle_mask_ratio=head.geom_angle_mask_ratio)


def _pixel_diagnostic(head: LIIFHead, prep: dict[str, Any], c: torch.Tensor,
                      x: Any) -> dict[str, Any] | None:
    """The two pixel-level diagnostic columns (never in the headline).

    Full-frame chunked decode at the pixel GT's own grid (short side 512),
    then soft-IoU(minmax) against it and the MAE inside the GT transition band
    ``0.05 < g < 0.95``.  Each column is reported next to a centre prior and a
    floor on the SAME support.  No AUC, no per-image min-max normalisation --
    the numbers are computed on the raw field.
    """
    gt = getattr(x, "gt_pix", None)
    if gt is None:
        head._count("pix_diag_no_gt")
        return None
    from q3vl.whereb.amort.evaluate import center_prior_unit, random_floor
    from q3vl.whereb.metrics import soft_iou_value

    gt = gt.detach().float().to(prep["feat"].device)
    ph, pw = int(gt.shape[-2]), int(gt.shape[-1])
    alpha, _ = head.decode_grid(prep, c, ph, pw)
    cp = center_prior_unit(ph, pw, device=alpha.device)
    band = (gt > 0.05) & (gt < 0.95)
    n_band = int(band.sum())
    if n_band:
        mae = float((alpha[band] - gt[band]).abs().mean())
        mae_cp = float((cp[band] - gt[band]).abs().mean())
        # a U(0,1) prediction's expected |u - g| = (g^2 + (1-g)^2) / 2
        g = gt[band]
        mae_floor = float(((g ** 2 + (1.0 - g) ** 2) / 2.0).mean())
    else:
        mae = mae_cp = mae_floor = None
    head._count("pix_diag")
    return {
        "h": ph, "w": pw,
        "gt_source": getattr(x, "gt_pix_source", ""),
        "soft_iou": soft_iou_value(alpha, gt),
        "soft_iou_center": soft_iou_value(cp, gt),
        "soft_iou_floor": random_floor(float((gt > 0.5).float().mean())),
        "band_mae": mae, "band_mae_center": mae_cp, "band_mae_floor": mae_floor,
        "n_band": n_band,
    }


def compute_loss(model, out: dict[str, Any], x, weights) -> AmortLoss:
    """``train_liif.py:91/110``: ``L = mean |y_hat - (2g - 1)|``, and nothing else.

    The seven-term stack is not entered (``trainer.py:244-251`` routes new arms
    here instead): the reference has one loss and mixing another term in would
    make the port a different recipe.  Ablation ⑥ swaps L1 for a soft BCE on
    ``sigma(y)``.  Ablation ① adds ``geom_reg_weight * L_geom`` and nothing
    else changes.
    """
    head = model.geo
    y = out["y_pts"]
    g = out["g_pts"].to(y.dtype)
    if head.loss_kind == "l1":
        term = (y - (2.0 * g - 1.0)).abs().mean()
    else:
        term = bce_soft(torch.sigmoid(y), g)
    key = head.loss_term_key
    terms: dict[str, torch.Tensor] = {key: term}
    total = term
    # split columns so the foreign-instruction policy is auditable per step
    with torch.no_grad():
        if bool(getattr(x, "is_fake", False)):
            terms[f"{key}_fake"] = term.detach()
        else:
            terms[f"{key}_real"] = term.detach()

    stats: dict[str, float] = {
        "liif_n_pts": float(out.get("liif_train", {}).get("n_pts", 0)),
        "liif_scale": float(out.get("liif_train", {}).get("s", 0.0)),
    }

    if head.geom is not None:
        gl, gstats = _geom_loss(head, out)
        terms["geom"] = gl
        total = total + float(head.geom_reg_weight) * gl
        stats.update(gstats)

    if not head._loss_keys_asserted:
        # runtime assertion #1 (proposal §3 ⑦(a)): the pre-registered loss
        # columns must exist on the FIRST micro-batch, not "somewhere later".
        want = {k[2:] for k in head.expected_loss_columns()}
        missing = sorted(want - set(terms))
        if missing:
            raise AssertionError(
                f"arm LIIF pre-registers the loss column(s) "
                f"{sorted('L_' + m for m in missing)} and the first "
                f"micro-batch produced {sorted('L_' + t for t in terms)}")
        head._loss_keys_asserted = True

    return AmortLoss(total=total, terms=terms, stats=stats)


def _geom_loss(head: LIIFHead, out: dict[str, Any]
               ) -> tuple[torch.Tensor, dict[str, float]]:
    """``L_geom = mean_j |p_hat_j - p_j| + BCE(sigma(f_hat), Flipped)``.

    Plus the ``gradient`` vs ``circulargradient`` routing logit, weighted by
    ``geom_reg_type_weight`` (default 1.0).  That term is NOT in the proposal's
    formula block, which lists only the continuous columns and ``Flipped``;
    without it the pre-registered routing-accuracy column would report an
    untrained zero-initialised constant, i.e. exactly the vacuous column the
    campaign's runtime assertions exist to prevent.  Set
    ``--geom-reg-type-weight 0`` for the formula as literally written.
    """
    pred = out["geom_pred"]
    tgt = out.get("geom_target")
    dev = pred["type_logit"].device
    if tgt is None:
        # semantic / store miss: excluded from the denominator, contributes
        # exactly zero gradient (and is counted in facts()).
        zero = pred["type_logit"] * 0.0
        return zero, {"geom_n": 0.0}

    mt = tgt["mask_type"]
    cont_pred = pred["circ_cont"] if mt == "circulargradient" else pred["grad_cont"]
    flip_pred = pred["circ_flip"] if mt == "circulargradient" else pred["grad_flip"]
    vals = torch.tensor(tgt["values"], dtype=cont_pred.dtype, device=dev)
    mask = torch.tensor(tgt["col_mask"], dtype=cont_pred.dtype, device=dev)
    denom = mask.sum().clamp_min(1.0)
    l_cont = ((cont_pred - vals).abs() * mask).sum() / denom
    l_flip = F.binary_cross_entropy_with_logits(
        flip_pred.reshape(()), torch.tensor(float(tgt["flipped"]),
                                            dtype=flip_pred.dtype, device=dev))
    l = l_cont + l_flip
    is_grad = 1.0 if mt == "gradient" else 0.0
    if head.geom_reg_type_weight:
        l = l + float(head.geom_reg_type_weight) * F.binary_cross_entropy_with_logits(
            pred["type_logit"].reshape(()),
            torch.tensor(is_grad, dtype=pred["type_logit"].dtype, device=dev))
    stats = {
        "geom_n": 1.0,
        "geom_l1": float(l_cont.detach()),
        "geom_type_correct": float((float(pred["type_logit"].detach()) > 0.0)
                                   == (is_grad > 0.5)),
        "geom_angle_masked": float(bool(tgt["angle_masked"])),
    }
    return l, stats


def train_stats(out: dict[str, Any], x) -> dict[str, Any]:
    """Extra per-sample columns on the trainer's per-step rows."""
    t = dict(out.get("liif_train", {}) or {})
    row = {"liif_scale": t.get("s"), "liif_qh": t.get("qh"),
           "liif_qw": t.get("qw"), "liif_n_pts": t.get("n_pts"),
           "gt_pix_source": t.get("gt_pix_source", "")}
    if "geom_target" in out:
        tg = out["geom_target"]
        row["geom_mask_type"] = None if tg is None else tg["mask_type"]
    return row


# --------------------------------------------------------------------------- #
# evaluation columns
# --------------------------------------------------------------------------- #
def per_sample_row(model, out: dict[str, Any], x) -> dict[str, Any]:
    """The arm's own per-sample columns (``evaluate.py:194-206``)."""
    dec = out.get("liif_decode") or {}
    row: dict[str, Any] = {
        # the pre-registered criterion: this row's m_low really was produced by
        # the LIIF decoder on the 4x query grid and area_resize'd back
        "liif_grid_decode": 1.0 if dec else 0.0,
        "liif_decode_scale": dec.get("scale"),
        "liif_decode_qh": dec.get("qh"), "liif_decode_qw": dec.get("qw"),
        "liif_decode_projection": dec.get("projection"),
        "liif_decode_activation": dec.get("activation"),
        "liif_gt_pix_source": getattr(x, "gt_pix_source", ""),
    }
    pix = out.get("liif_pix")
    if pix:
        row.update({
            "pix_soft_iou_512": pix["soft_iou"],
            "pix_soft_iou_512_center": pix["soft_iou_center"],
            "pix_soft_iou_512_floor": pix["soft_iou_floor"],
            "pix_band_mae_512": pix["band_mae"],
            "pix_band_mae_512_center": pix["band_mae_center"],
            "pix_band_mae_512_floor": pix["band_mae_floor"],
            "pix_grid": [pix["h"], pix["w"]],
        })
    if out.get("liif_pix_error"):
        # the diagnostic raised; the row still exists and still carries
        # `liif_grid_decode` (the pre-registered criterion)
        row["pix_diag_error"] = out["liif_pix_error"]
    gp, gt = out.get("geom_pred"), out.get("geom_target")
    if gp is not None:
        pred_grad = bool(float(gp["type_logit"].detach()) > 0.0)
        rx_hat = float(gp["circ_cont"].detach()[CIRC_COLS.index("rx")])
        head = model.geo
        pred_label = ("linear" if pred_grad
                      else ("band" if rx_hat >= head.geom_band_rx else "radial"))
        row.update({"geom_reg_route": 1.0, "geom_pred_label": pred_label,
                    "geom_pred_rx": rx_hat,
                    "geom_pred_is_gradient": float(pred_grad)})
        if gt is not None:
            row["geom_gt_mask_type"] = gt["mask_type"]
            row["geom_type_correct"] = float(
                pred_grad == (gt["mask_type"] == "gradient"))
            row["geom_angle_masked"] = float(bool(gt["angle_masked"]))
    return row


def _agg(vals: Sequence[float]) -> dict[str, Any]:
    v = [float(x) for x in vals if x is not None and np.isfinite(float(x))]
    if not v:
        return {"n": 0, "median": None, "mean": None}
    return {"n": len(v), "median": float(np.median(v)), "mean": float(np.mean(v)),
            "p25": float(np.percentile(v, 25)), "p75": float(np.percentile(v, 75))}


def criteria_columns(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """``board["criteria_columns"]`` entries, built from the headline's rows.

    ``assert_criteria_ran`` (``evaluate.py:313-368``) refuses to publish a board
    whose ``liif_grid_decode`` column carries ``n = 0``.  The conditional
    ``geom_reg_route`` column cannot be registered in the static
    ``ARM_CRITERIA`` table, so it is asserted here instead -- same rule, same
    "n = 0 cannot publish" outcome.
    """
    rows = [dict(r) for r in rows]
    dec = [r for r in rows if r.get("liif_grid_decode") is not None]
    n_dec = int(sum(float(r["liif_grid_decode"]) for r in dec))
    scales = sorted({r.get("liif_decode_scale") for r in dec
                     if r.get("liif_decode_scale") is not None})
    cols: dict[str, Any] = {
        "liif_grid_decode": {
            "n": n_dec,
            "n_rows": len(rows),
            "frac": (n_dec / len(rows)) if rows else None,
            "scales": scales,
            "projection": sorted({r.get("liif_decode_projection") for r in dec
                                  if r.get("liif_decode_projection")}),
            "activation": sorted({r.get("liif_decode_activation") for r in dec
                                  if r.get("liif_decode_activation")}),
            "gt_pix_source": _hist(rows, "liif_gt_pix_source"),
            "note": ("m_low was decoded by the LIIF head on the "
                     "scale x (gh, gw) query grid's cell centres and projected "
                     "back with area_resize -- the gt_low operator "
                     "(data.py:685-686)"),
        },
    }
    pix = [r for r in rows if r.get("pix_soft_iou_512") is not None]
    err = [r.get("pix_diag_error") for r in rows if r.get("pix_diag_error")]
    if err:
        # counted and named, never silent: a diagnostic that failed on some
        # rows changes the `n` of the diagnostic column only
        cols["liif_pix_diag_errors"] = {
            "n": len(err), "n_rows": len(rows),
            "kinds": sorted({str(e).split(":", 1)[0] for e in err}),
            "first": err[0],
            "note": ("the pixel diagnostic raised on these rows; their "
                     "pix_* columns are absent.  The headline read-out "
                     "(m_low / liif_grid_decode) is not affected."),
        }
    if pix:
        cols["liif_pix_diag"] = {
            "n": len(pix),
            "pix_soft_iou_512": _agg([r["pix_soft_iou_512"] for r in pix]),
            "pix_soft_iou_512_center": _agg(
                [r["pix_soft_iou_512_center"] for r in pix]),
            "pix_soft_iou_512_floor": _agg(
                [r["pix_soft_iou_512_floor"] for r in pix]),
            "pix_band_mae_512": _agg([r.get("pix_band_mae_512") for r in pix]),
            "pix_band_mae_512_center": _agg(
                [r.get("pix_band_mae_512_center") for r in pix]),
            "pix_band_mae_512_floor": _agg(
                [r.get("pix_band_mae_512_floor") for r in pix]),
            "note": ("diagnostic only, never the headline; soft-IoU is the "
                     "minmax form on the raw field (no per-image min-max "
                     "normalisation, no AUC)"),
        }
    geo = [r for r in rows if r.get("geom_reg_route") is not None]
    if geo:
        cols[GEOM_CRITERION] = _geom_column(geo)
        if int(cols[GEOM_CRITERION]["n"]) == 0:
            raise AssertionError(
                "arm LIIF ran with --geom-reg-weight > 0, which pre-registers "
                f"'{GEOM_CRITERION}', and the board carries 0 values for it")
    _assert_steps_columns()
    return cols


def _hist(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        v = str(r.get(key, "") or "")
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))


def _geom_column(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Routing accuracy + the band/radial/linear confusion (消融行 ① 观察列)."""
    typed = [r for r in rows if r.get("geom_type_correct") is not None]
    labels = ("band", "radial", "linear")
    conf = {f"gt_{g}__pred_{p}": 0 for g in labels for p in labels}
    n_conf = 0
    for r in rows:
        gt_fam = str(r.get("family", "") or "").split("-")[0]
        pred = r.get("geom_pred_label")
        if gt_fam in labels and pred in labels:
            conf[f"gt_{gt_fam}__pred_{pred}"] += 1
            n_conf += 1
    return {
        "n": len(rows),
        "mask_type_accuracy": (float(np.mean([r["geom_type_correct"]
                                              for r in typed]))
                               if typed else None),
        "n_mask_type": len(typed),
        "band_rx_threshold": float(config()["geom_band_rx"]),
        "pred_rx": _agg([r.get("geom_pred_rx") for r in rows]),
        "angle_masked_frac": (float(np.mean([r.get("geom_angle_masked", 0.0)
                                             for r in rows]))),
        "confusion": conf,
        "n_confusion": n_conf,
        "note": ("layer (i) = the gradient vs circulargradient logit; layer "
                 "(ii) = band iff predicted rx >= threshold "
                 "(subject_geom.py:22 BAND_AXIS_LEN = 1.6 vs radial's "
                 "extent_a * margin, margin <= 1.60)"),
    }


# --------------------------------------------------------------------------- #
# runtime assertion #2: the pre-registered loss columns reached steps.jsonl
# --------------------------------------------------------------------------- #
def assert_first_step_row(row: Mapping[str, Any], *, expected: Sequence[str]
                          ) -> dict[str, Any]:
    """The first ``steps.jsonl`` row must already carry every ``L_*`` column.

    First row, not last: a loss term that is missing is missing from step one,
    and finding out at the end costs the whole arm.
    """
    have = sorted(k for k in row if str(k).startswith("L_"))
    missing = [c for c in expected if c not in row]
    if missing:
        raise AssertionError(
            f"arm LIIF pre-registers {list(expected)} in steps.jsonl and the "
            f"first row carries {have}; the configured loss never reached the "
            "optimiser")
    return {"checked": list(expected), "columns": have, "step": row.get("step")}


def _assert_steps_columns() -> dict[str, Any] | None:
    """Same check, run from inside the board build (i.e. at the first quick
    eval, ~step 500) when the wrapper recorded the run directory."""
    run_dir = config().get("run_dir")
    expected = config().get("expected_loss_columns")
    if not run_dir or not expected:
        return None
    p = Path(run_dir) / "steps.jsonl"
    if not p.exists():
        return None
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                return assert_first_step_row(json.loads(line), expected=expected)
    return None


# --------------------------------------------------------------------------- #
# flags (the wrapper owns the parsing; this keeps one definition of the set)
# --------------------------------------------------------------------------- #
def add_arguments(ap) -> None:
    """Register ``--liif-*`` / ``--geom-reg-*`` on an ``ArgumentParser``."""
    ap.add_argument("--liif-sample-q", type=int, default=DEFAULTS["sample_q"],
                    help="points per sample (yaml L14: 2304)")
    ap.add_argument("--liif-hidden", default="256,256,256,256",
                    help="imnet hidden widths (yaml L48)")
    ap.add_argument("--liif-feat-dim", type=int, default=DEFAULTS["feat_dim"],
                    help="1x1 adapter width (edsr.py L169: 64); 0 = ablation "
                         "row 7, F_pre goes in raw")
    ap.add_argument("--liif-cond-dim", type=int, default=DEFAULTS["cond_dim"],
                    help="width of the language condition concatenated per "
                         "point (NOVEL: LIIF is unconditional)")
    ap.add_argument("--liif-scale-max", type=float, default=DEFAULTS["scale_max"],
                    help="s ~ U(1, scale_max) (yaml L12); ALSO the read-out "
                         "grid: the board is decoded at scale_max x (gh, gw)")
    ap.add_argument("--liif-no-local-ensemble", action="store_true",
                    help="ablation ②: liif.py L54-55's [0] branch")
    ap.add_argument("--liif-no-feat-unfold", action="store_true",
                    help="ablation ⑧: drop the 3x3 unfold (liif.py L46-48)")
    ap.add_argument("--liif-no-cell-decode", action="store_true",
                    help="ablation ⑨: drop rel_cell (liif.py L86-90)")
    ap.add_argument("--liif-gt", default=DEFAULTS["gt"], choices=["closed", "cgt"],
                    help="closed = analytic re-render / closed-form point "
                         "evaluation; cgt = ablation ⑤, every family read "
                         "bilinearly off the published .maskhi.png")
    ap.add_argument("--liif-loss", default=DEFAULTS["loss"], choices=["l1", "bce"],
                    help="l1 = train_liif.py L91; bce = ablation ⑥")
    ap.add_argument("--liif-eval-bsize", type=int, default=DEFAULTS["eval_bsize"],
                    help="chunk size of the decode (test.py L16-29)")
    ap.add_argument("--liif-no-pix-diag", action="store_true",
                    help="skip the 512-grid diagnostic columns (they cost a "
                         "full-frame decode per evaluated sample)")
    ap.add_argument("--liif-pixgt-scale", type=int, default=DEFAULTS["pixgt_scale"],
                    help="pixel-GT grid = k x (gh, gw); 16 = the spec-5 pixel "
                         "grid, i.e. .maskhi's own short-side-512 resolution")
    ap.add_argument("--liif-amount-ref", type=int, default=DEFAULTS["amount_ref"],
                    help="k x k reference grid the linear family's `amount` "
                         "raw_mean is measured on (待决策 (f) writes 256); "
                         "0 = the shared default, i.e. the rendered grid itself")
    ap.add_argument("--geom-reg-weight", type=float,
                    default=DEFAULTS["geom_reg_weight"],
                    help="ablation ①: 0.0 = the branch is not constructed")
    ap.add_argument("--geom-reg-type-weight", type=float,
                    default=DEFAULTS["geom_reg_type_weight"],
                    help="weight of the gradient/circulargradient routing BCE "
                         "inside L_geom (0 = the formula as literally written)")
    ap.add_argument("--geom-reg-angle-mask-ratio", type=float,
                    default=DEFAULTS["geom_angle_mask_ratio"],
                    help="mask the angle columns when rx/ry <= this "
                         "(subject_geom.py MIN_ELONG = 1.35; 待决策 (c))")
    ap.add_argument("--geom-reg-band-rx", type=float,
                    default=DEFAULTS["geom_band_rx"],
                    help="predicted rx above which a circulargradient is read "
                         "as band (待决策 (d))")


def cfg_from_args(own) -> dict[str, Any]:
    """The wrapper's parsed namespace -> :data:`CFG`."""
    hidden = tuple(int(h) for h in str(own.liif_hidden).split(",") if h.strip())
    if not hidden:
        raise ValueError("--liif-hidden must list at least one width")
    return {
        "sample_q": int(own.liif_sample_q),
        "hidden": hidden,
        "feat_dim": int(own.liif_feat_dim),
        "cond_dim": int(own.liif_cond_dim),
        "scale_min": float(DEFAULTS["scale_min"]),
        "scale_max": float(own.liif_scale_max),
        "local_ensemble": not bool(own.liif_no_local_ensemble),
        "feat_unfold": not bool(own.liif_no_feat_unfold),
        "cell_decode": not bool(own.liif_no_cell_decode),
        "gt": str(own.liif_gt),
        "loss": str(own.liif_loss),
        "eval_bsize": int(own.liif_eval_bsize),
        "pix_diag": not bool(own.liif_no_pix_diag),
        "pixgt_scale": int(own.liif_pixgt_scale),
        "amount_ref": int(own.liif_amount_ref),
        "geom_reg_weight": float(own.geom_reg_weight),
        "geom_reg_type_weight": float(own.geom_reg_type_weight),
        "geom_angle_mask_ratio": float(own.geom_reg_angle_mask_ratio),
        "geom_band_rx": float(own.geom_reg_band_rx),
    }
