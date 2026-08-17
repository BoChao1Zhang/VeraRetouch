"""EPR-018: LISA's ``[SEG]``-token -> SAM mask decoder, on frozen Qwen3-VL features.

Registry arm ``SEGSAM`` (``q3vl/whereb/amort/arms.py``).  Everything structural,
every loss constant and every optimiser constant is transcribed from the
reference sources, whose raw files were re-opened on 2026-08-14 (URLs in
:data:`SOURCES`); the three places this repository cannot reproduce LISA are
marked **NOVEL** in the code, with the reason, exactly as the proposal lists
them (``experiments/prs/EPR-018_seg-token-sam-decoder/PROPOSAL.md`` §1.2).

The port, end to end
--------------------
``F_pre (1,1024,gh,gw)``  -> neck (PixelLM ``image_feature_neck`` shape)
                          -> ``image_embeddings (1,256,gh,gw)``
``h_cond (K,2560)``       -> ``text_hidden_fcs`` (LISA.py:91-98, verbatim)
                          -> ``text_embeds (1,K,256)``
                          -> ``prompt_encoder(points=None, boxes=None,
                             masks=None, text_embeds=...)``  (LISA.py:275-280)
``mask_decoder(multimask_output=False)`` -> ``raw (1, 4gh, 4gw)``
``m_low = area_resize(sigmoid(raw), (gh, gw))``  -- the ``gt_low`` operator, so
the frozen headline criterion measures this arm with the same ruler as every
published board (``q3vl/whereb/amort/data.py:685-686``).

Loss: ``L = 2.0 * sigmoid_ce + 0.5 * dice`` (LISA.py:16-59, 321-335; weights
``train_ds.py:79-80``).  The campaign's "dice is never a mask target" red line
is waived for this batch by the user's 2026-08-14 instruction, and the waiver
is written into every run through :func:`loss_form`.

Three NOVEL adaptations (proposal §1.2, nothing else deviates)
--------------------------------------------------------------
1. **image encoder**: SAM's ViT-H is replaced by the campaign's frozen
   Qwen3-VL ``F_pre`` + a PixelLM-shaped neck (``PixelLM.py:175-191``), because
   the campaign freezes the VLM and does not add a second 632M image tower.
2. **conditioning vector**: LISA's generated ``[SEG]`` row becomes the v2seg
   supervised ``<seg_where>`` row, read by :mod:`q3vl.whereb.readout`.
3. **text CE**: removed -- this arm has no text-generation branch, so LISA's
   ``ce_loss`` (and the LoRA / lm_head / embed_tokens it serves) has no
   counterpart.

Plus two mechanical consequences of (1), both recorded in :func:`facts`:
``image_pe`` is computed per sample as ``pe_layer((gh, gw))`` instead of the
fixed 64x64 ``get_dense_pe()``, and the ``no_mask_embed`` dense prompt is
expanded to ``(gh, gw)``.  ``postprocess_masks`` is dropped on the main arm
(there is no 1024 letterbox to undo) and returns as the ``--segsam-sup cgt``
ablation row.

Pre-registered violation, declared not hidden
---------------------------------------------
``PositionEmbeddingRandom`` is a coordinate/position encoding, which DELTA §5.6
forbids (``q3vl/whereb/amort/heads.py:15-21``).  This arm keeps it -- removing
it would not be a port -- and is judged on the per-board execution line
``corr_center_minus_corr_gt`` (``evaluate.py:413-416``): that column above 0 is
the centre-prior disease and kills this structure.

NOTES (2026-08-14, decided conservatively, not silently)
--------------------------------------------------------
* D-2 (SAM checkpoint location) had no proposal default.  The asset pass
  downloaded the official ``vit_h`` checkpoint to :data:`SAM_VIT_H_DEFAULT`, so
  that path is the default here; a missing file raises with the official URL
  rather than training a scratch decoder that looks like the pretrained one.
* D-5 / D-6 / D-7 / D-8 follow the proposal's written defaults: supervision on
  the decoder-native 4x grid, ``is_fake`` samples supervised against an
  all-zero target with the unchanged formula, ``<seg_color>`` unconsumed.
* Per-step counters (GT source, all-zero exclusions) cannot reach
  ``steps.jsonl``: ``losses.aggregate`` forwards only ``uniq_*`` stats and the
  trainer discards the per-sample rows.  They are carried in ``facts()``
  (-> ``run_setup.json`` and every checkpoint) and in the board's criterion
  column instead.  See the report's contract-gap note.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from q3vl.whereb.fields import no_autocast

from .losses import AmortLoss

__all__ = [
    "ARM", "CRITERIA", "DEFAULTS", "OPTIONS", "SOURCES",
    "LayerNorm2d", "MLPBlock", "MLP", "Attention", "TwoWayAttentionBlock",
    "TwoWayTransformer", "PositionEmbeddingRandom", "PromptEncoder",
    "MaskDecoder", "build_neck", "SegSamHead",
    "sigmoid_ce_loss", "dice_loss", "lisa_mask_loss",
    "load_sam_subtrees", "build_head", "forward", "compute_loss",
    "add_arguments", "configure", "options", "setup_record", "loss_form",
    "loss_preregistration",
    "head_kwargs_from_args", "optimizer_spec", "scheduler_kwargs",
    "builder_kwargs", "per_sample_row", "criteria_columns", "train_stats",
    "assert_publishable", "FIRST_LOSS_COLUMNS", "first_step_row",
    "resolve_steps_row",
]

# --------------------------------------------------------------------------- #
# registry contract
# --------------------------------------------------------------------------- #
ARM = "SEGSAM"
CRITERIA = ("segsam_fine",)

#: raw files re-opened 2026-08-14; every constant below cites one of them
SOURCES: dict[str, str] = {
    "LISA": "https://raw.githubusercontent.com/dvlab-research/LISA/main/model/LISA.py",
    "train_ds": "https://raw.githubusercontent.com/dvlab-research/LISA/main/train_ds.py",
    "mask_decoder": "https://raw.githubusercontent.com/dvlab-research/LISA/main/"
                    "model/segment_anything/modeling/mask_decoder.py",
    "prompt_encoder": "https://raw.githubusercontent.com/dvlab-research/LISA/main/"
                      "model/segment_anything/modeling/prompt_encoder.py",
    "transformer": "https://raw.githubusercontent.com/dvlab-research/LISA/main/"
                   "model/segment_anything/modeling/transformer.py",
    "common": "https://raw.githubusercontent.com/dvlab-research/LISA/main/"
              "model/segment_anything/modeling/common.py",
    "build_sam": "https://raw.githubusercontent.com/facebookresearch/"
                 "segment-anything/main/segment_anything/build_sam.py",
    "PixelLM": "https://raw.githubusercontent.com/MaverickRen/PixelLM/main/"
               "model/PixelLM.py",
}

# -- SAM structural constants (build_sam.py:62-98) --------------------------
SAM_PROMPT_EMBED_DIM = 256
SAM_IMAGE_SIZE = 1024
SAM_PATCH = 16
SAM_IMAGE_EMBEDDING_SIZE = (SAM_IMAGE_SIZE // SAM_PATCH, SAM_IMAGE_SIZE // SAM_PATCH)
SAM_MASK_IN_CHANS = 16
SAM_TRANSFORMER_DEPTH = 2
SAM_TRANSFORMER_MLP_DIM = 2048
SAM_TRANSFORMER_HEADS = 8
SAM_NUM_MULTIMASK_OUTPUTS = 3
SAM_IOU_HEAD_DEPTH = 3
SAM_IOU_HEAD_HIDDEN_DIM = 256
#: two ConvTranspose2d(stride 2) in ``output_upscaling`` (mask_decoder.py:53-63)
FINE_UPSCALE = 4

# -- LISA loss / optimiser constants ----------------------------------------
LISA_OUT_DIM = 256                 # train_ds.py:92 (--out_dim)
LISA_BCE_WEIGHT = 2.0              # train_ds.py:80
LISA_DICE_WEIGHT = 0.5             # train_ds.py:79
LISA_DICE_SCALE = 1000             # LISA.py:20
LISA_DICE_EPS = 1e-6               # LISA.py:21
LISA_NUM_MASKS_EPS = 1e-8          # LISA.py:38, :58
LISA_LR = 3e-4                     # train_ds.py:77
LISA_WEIGHT_DECAY = 0.0            # train_ds.py:275
LISA_BETAS = (0.9, 0.95)           # train_ds.py:85-86, :276
LISA_GRAD_CLIP = 1.0               # train_ds.py:295
#: ``warmup_num_steps 100`` of ``epochs 10 * steps_per_epoch 500 = 5000``
LISA_WARMUP_FRAC = 100.0 / 5000.0
LISA_REFERENCE_TOTAL_STEPS = 5000
#: the campaign's step-matched horizon (U4; run_amort_arm.py:110)
PREREGISTERED_TOTAL_STEPS = 1200

#: official ``vit_h`` checkpoint (segment-anything README L112), downloaded here
SAM_VIT_H_DEFAULT = "/home/bc/data/models/sam/sam_vit_h_4b8939.pth"
SAM_VIT_H_URL = ("https://dl.fbaipublicfiles.com/segment_anything/"
                 "sam_vit_h_4b8939.pth")
#: the only two subtrees this arm reads; ``image_encoder.*`` (632M) is not
SUBTREES: tuple[str, ...] = ("mask_decoder", "prompt_encoder")

#: ``.cgt`` short side 1024 over the ``F_pre`` grid: spec-5 renders the short
#: side at 512 and ``F_pre`` is ``H/16 x W/16`` (``q3vl/where/fpre.py:1-22``),
#: so the ``.cgt`` grid is exactly 32x the token grid.
CGT_UPSCALE = 32
SUP_UPSCALE: dict[str, int] = {"native": FINE_UPSCALE, "cgt": CGT_UPSCALE}

#: boundary-F1 tolerance on the fine grid: 1 coarse cell = 4 fine cells, so the
#: fine diagnostic keeps the same physical tolerance as the headline's
#: ``tol_cells = 1`` (``q3vl/whereb/config.py:265``).
BOUNDARY_TOL_FINE = 4

#: ``--segsam-gt`` -> the shared ``--pixgt-source`` / ``--pixgt-fallback`` pair.
#: ``raster``: analytic ``raster_geometry`` for radial/band/linear, ``.cgt``
#: (short side 1024, area-resized) for semantic and for any geometry miss --
#: proposal §3 (6).  ``png``: all four families from ``.cgt``.
GT_TO_PIXGT: dict[str, tuple[str, str]] = {
    "raster": ("render", "cgt1024"),
    "png": ("cgt1024", "cgt1024"),
}

#: ``--segsam-*`` defaults = the proposal's written values.  Module-level (the
#: seam ``uniq4.VARIANT4`` / ``prnd.OPTIONS`` already use) because the entry
#: script hands the hooks three different argparse namespaces; the sha256
#: source freeze covers this record.
DEFAULTS: dict[str, Any] = {
    "weights": SAM_VIT_H_DEFAULT,   # D-2, see the NOTES in the module docstring
    "scratch": False,               # ablation (4): no pretrained decoder
    "frozen_decoder": False,        # ablation (3): train_mask_decoder = False
    "dice_weight": LISA_DICE_WEIGHT,  # ablation (2): 0.0
    "bce_weight": LISA_BCE_WEIGHT,  # not a flag: LISA's 2.0, fixed
    "gt": "raster",                 # ablation (1): "png"
    "sup": "native",                # ablation (5): "cgt"
}

#: filled by ``run_segsam_arm.py`` before it delegates; empty = every default
OPTIONS: dict[str, Any] = {}


def options() -> dict[str, Any]:
    """The effective ``--segsam-*`` record: defaults overlaid with ``OPTIONS``."""
    return {**DEFAULTS, **OPTIONS}


def configure(ns: Any) -> dict[str, Any]:
    """``argparse`` namespace (``--segsam-*``) -> ``OPTIONS``; returns the record."""
    for k in DEFAULTS:
        v = getattr(ns, f"segsam_{k}", None)
        if v is not None:
            OPTIONS[k] = v
    return options()


# --------------------------------------------------------------------------- #
# vendored SAM modules (structure copied; see SOURCES)
# --------------------------------------------------------------------------- #
class LayerNorm2d(nn.Module):
    """``common.py:31-43``, verbatim."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class MLPBlock(nn.Module):
    """``common.py:13-26``, verbatim."""

    def __init__(self, embedding_dim: int, mlp_dim: int,
                 act: Type[nn.Module] = nn.GELU) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class MLP(nn.Module):
    """``mask_decoder.py:169-191``, verbatim."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int, sigmoid_output: bool = False) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class Attention(nn.Module):
    """``transformer.py:185-242``, verbatim."""

    def __init__(self, embedding_dim: int, num_heads: int,
                 downsample_rate: int = 1) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)

    def _recombine_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)
        out = attn @ v
        out = self._recombine_heads(out)
        return self.out_proj(out)


class TwoWayAttentionBlock(nn.Module):
    """``transformer.py:109-182``, verbatim."""

    def __init__(self, embedding_dim: int, num_heads: int, mlp_dim: int = 2048,
                 activation: Type[nn.Module] = nn.ReLU,
                 attention_downsample_rate: int = 2,
                 skip_first_layer_pe: bool = False) -> None:
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self, queries, keys, query_pe, key_pe):
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)
        return queries, keys


class TwoWayTransformer(nn.Module):
    """``transformer.py:16-106``, verbatim."""

    def __init__(self, depth: int, embedding_dim: int, num_heads: int,
                 mlp_dim: int, activation: Type[nn.Module] = nn.ReLU,
                 attention_downsample_rate: int = 2) -> None:
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(TwoWayAttentionBlock(
                embedding_dim=embedding_dim, num_heads=num_heads,
                mlp_dim=mlp_dim, activation=activation,
                attention_downsample_rate=attention_downsample_rate,
                skip_first_layer_pe=(i == 0)))
        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(self, image_embedding, image_pe, point_embedding):
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        queries = point_embedding
        keys = image_embedding
        for layer in self.layers:
            queries, keys = layer(queries=queries, keys=keys,
                                  query_pe=point_embedding, key_pe=image_pe)
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)
        return queries, keys


class PositionEmbeddingRandom(nn.Module):
    """``prompt_encoder.py:189-238``, verbatim.

    DELTA §5.6 violation, declared: this is a coordinate encoding.  The arm is
    judged on ``corr_center_minus_corr_gt`` (see the module docstring).
    """

    def __init__(self, num_pos_feats: int = 64, scale: float | None = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer("positional_encoding_gaussian_matrix",
                             scale * torch.randn((2, num_pos_feats)))

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        coords = 2 * coords - 1
        if coords.dtype != self.positional_encoding_gaussian_matrix.dtype:
            coords = coords.to(self.positional_encoding_gaussian_matrix.dtype)
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: tuple[int, int]) -> torch.Tensor:
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device,
                          dtype=self.positional_encoding_gaussian_matrix.dtype)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)

    def forward_with_coords(self, coords_input: torch.Tensor,
                            image_size: tuple[int, int]) -> torch.Tensor:
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))


class PromptEncoder(nn.Module):
    """``prompt_encoder.py:16-186``.

    One deviation from verbatim, and it is the NOVEL item the proposal lists:
    :meth:`forward` takes an optional ``embedding_size`` so the ``no_mask_embed``
    dense prompt can be expanded to **this sample's** ``(gh, gw)`` instead of
    SAM's fixed 64x64 (the same reason ``image_pe`` is recomputed per sample).
    Everything else -- module names, shapes, initialisation -- is unchanged, so
    the checkpoint's ``prompt_encoder.*`` subtree loads ``strict=True``.
    """

    def __init__(self, embed_dim: int, image_embedding_size: tuple[int, int],
                 input_image_size: tuple[int, int], mask_in_chans: int,
                 activation: Type[nn.Module] = nn.GELU) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.num_point_embeddings: int = 4
        point_embeddings = [nn.Embedding(1, embed_dim)
                            for _ in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)
        self.mask_input_size = (4 * image_embedding_size[0],
                                4 * image_embedding_size[1])
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def dense_pe_for(self, size: tuple[int, int]) -> torch.Tensor:
        """NOVEL: ``get_dense_pe()`` at an arbitrary grid (proposal §3 (4))."""
        return self.pe_layer((int(size[0]), int(size[1]))).unsqueeze(0)

    def _embed_points(self, points, labels, pad: bool):
        points = points + 0.5
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding

    def _embed_boxes(self, boxes):
        boxes = boxes + 0.5
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks):
        return self.mask_downscaling(masks)

    def _get_batch_size(self, points, boxes, masks, text_embeds) -> int:
        if points is not None:
            return points[0].shape[0]
        if boxes is not None:
            return boxes.shape[0]
        if masks is not None:
            return masks.shape[0]
        if text_embeds is not None:
            return text_embeds.shape[0]
        return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(self, points, boxes, masks, text_embeds,
                embedding_size: tuple[int, int] | None = None):
        bs = self._get_batch_size(points, boxes, masks, text_embeds)
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim),
                                        device=self._get_device())
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if text_embeds is not None:
            sparse_embeddings = torch.cat([sparse_embeddings, text_embeds], dim=1)

        size = tuple(embedding_size or self.image_embedding_size)
        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, int(size[0]), int(size[1]))
        return sparse_embeddings, dense_embeddings


class MaskDecoder(nn.Module):
    """``mask_decoder.py:16-164``, verbatim."""

    def __init__(self, *, transformer_dim: int, transformer: nn.Module,
                 num_multimask_outputs: int = 3,
                 activation: Type[nn.Module] = nn.GELU,
                 iou_head_depth: int = 3, iou_head_hidden_dim: int = 256) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4,
                               kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8,
                               kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
             for _ in range(self.num_mask_tokens)])
        self.iou_prediction_head = MLP(transformer_dim, iou_head_hidden_dim,
                                       self.num_mask_tokens, iou_head_depth)

    def forward(self, image_embeddings, image_pe, sparse_prompt_embeddings,
                dense_prompt_embeddings, multimask_output: bool):
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings, image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings)
        mask_slice = slice(1, None) if multimask_output else slice(0, 1)
        return masks[:, mask_slice, :, :], iou_pred[:, mask_slice]

    def predict_masks(self, image_embeddings, image_pe, sparse_prompt_embeddings,
                      dense_prompt_embeddings):
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1:(1 + self.num_mask_tokens), :]

        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: list[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(
            b, self.num_mask_tokens, h, w)
        iou_pred = self.iou_prediction_head(iou_token_out)
        return masks, iou_pred


def build_neck(in_dim: int, out_chans: int = SAM_PROMPT_EMBED_DIM) -> nn.Sequential:
    """PixelLM ``image_feature_neck`` (``PixelLM.py:175-191``), shape for shape.

    NOVEL only in ``in_dim``: PixelLM feeds it the VLM hidden size, this arm
    feeds it ``F_pre``'s 1024 channels.
    """
    return nn.Sequential(
        nn.Conv2d(in_dim, out_chans, kernel_size=1, bias=False),
        LayerNorm2d(out_chans),
        nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
        LayerNorm2d(out_chans),
    )


# --------------------------------------------------------------------------- #
# pretrained weights: two subtrees, never the image encoder
# --------------------------------------------------------------------------- #
def load_sam_subtrees(prompt_encoder: nn.Module, mask_decoder: nn.Module,
                      path: str) -> dict[str, Any]:
    """Load ``prompt_encoder.*`` and ``mask_decoder.*`` from a SAM checkpoint.

    ``image_encoder.*`` (632M parameters) is never materialised.  Missing or
    unexpected keys are a hard failure -- a decoder that silently kept half its
    random initialisation would be an unlabelled fourth ablation row -- and the
    full key report goes into :func:`SegSamHead.facts`.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"SAM ViT-H checkpoint not found: {path}\n"
            f"Download the official one ({SAM_VIT_H_URL}) or pass "
            "--segsam-scratch to run the no-pretrained-weights ablation row "
            "(proposal §4 row 4) -- those are different experiments.")
    try:
        sd = torch.load(str(p), map_location="cpu", mmap=True, weights_only=True)
    except (RuntimeError, TypeError):        # not a zipfile-format checkpoint
        sd = torch.load(str(p), map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd and "image_encoder.pos_embed" not in sd:
        sd = sd["state_dict"]

    report: dict[str, Any] = {"path": str(p), "bytes": p.stat().st_size,
                              "subtrees": {}, "skipped_prefixes": {}}
    hasher = hashlib.sha256()
    for name, module in (("prompt_encoder", prompt_encoder),
                         ("mask_decoder", mask_decoder)):
        prefix = name + "."
        sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        if not sub:
            raise KeyError(
                f"{path} carries no {prefix}* tensors; this is not a SAM "
                f"checkpoint (top-level prefixes: "
                f"{sorted({k.split('.')[0] for k in sd})})")
        for k in sorted(sub):
            hasher.update(k.encode())
            hasher.update(np.ascontiguousarray(
                sub[k].detach().to(torch.float32).cpu().numpy()).tobytes())
        res = module.load_state_dict(sub, strict=False)
        missing, unexpected = list(res.missing_keys), list(res.unexpected_keys)
        report["subtrees"][name] = {
            "n_tensors": len(sub),
            "n_params": int(sum(int(v.numel()) for v in sub.values())),
            "loaded_keys": sorted(sub),
            "missing_keys": missing, "unexpected_keys": unexpected,
        }
        if missing or unexpected:
            raise RuntimeError(
                f"SAM {name} subtree did not match this module: missing="
                f"{missing} unexpected={unexpected}.  Refusing to train a "
                "partially-initialised decoder under the pretrained arm's name.")
    other = sorted({k.split(".")[0] for k in sd} - set(SUBTREES))
    report["skipped_prefixes"] = {
        pre: int(sum(int(v.numel()) for k, v in sd.items()
                     if k.startswith(pre + ".")))
        for pre in other}
    report["subtree_sha256"] = hasher.hexdigest()
    return report


# --------------------------------------------------------------------------- #
# the head
# --------------------------------------------------------------------------- #
class SegSamHead(nn.Module):
    """LISA's segmentation branch, on ``F_pre`` + ``h_cond``.

    ``forward`` returns the decoder-native fine logits and the ``(gh, gw)``
    mask the frozen criterion path consumes.
    """

    def __init__(self, in_dim: int = 1024, text_dim: int = 2560, *,
                 out_dim: int = LISA_OUT_DIM,
                 weights: str = SAM_VIT_H_DEFAULT,
                 scratch: bool = False,
                 frozen_decoder: bool = False,
                 dice_weight: float = LISA_DICE_WEIGHT,
                 bce_weight: float = LISA_BCE_WEIGHT,
                 gt: str = "raster",
                 sup: str = "native",
                 seed: int = 20260810) -> None:
        super().__init__()
        if gt not in GT_TO_PIXGT:
            raise ValueError(f"--segsam-gt must be one of {sorted(GT_TO_PIXGT)}, got {gt!r}")
        if sup not in SUP_UPSCALE:
            raise ValueError(f"--segsam-sup must be one of {sorted(SUP_UPSCALE)}, got {sup!r}")
        self.in_dim = int(in_dim)
        self.text_dim = int(text_dim)
        self.out_dim = int(out_dim)
        self.dice_weight = float(dice_weight)
        self.bce_weight = float(bce_weight)
        self.gt = str(gt)
        self.sup = str(sup)
        self.scratch = bool(scratch)
        self.frozen_decoder = bool(frozen_decoder)
        self.seed = int(seed)
        #: per-run counters; `aggregate` cannot carry them to steps.jsonl, so
        #: they ride in facts() (run_setup.json + every checkpoint) instead
        self.counts: dict[str, int] = {}

        # RNG independence (N1): the head owns four randomly initialised
        # submodules, one of which (`PositionEmbeddingRandom`) draws a Gaussian
        # buffer.  Building them inside a forked stream keeps this arm from
        # shifting the global draw sequence -- so the batch sampler, the fake
        # coin and every other arm's initialisation are bit-identical to a run
        # without this head -- and makes the head's own initialisation a
        # function of `seed` alone.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            #: LISA.py:91-98, verbatim (in_dim = config.hidden_size = 2560)
            self.text_hidden_fcs = nn.ModuleList([nn.Sequential(
                nn.Linear(self.text_dim, self.text_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.text_dim, self.out_dim),
                nn.Dropout(0.0),
            )])
            self.neck = build_neck(self.in_dim, self.out_dim)
            self.prompt_encoder = PromptEncoder(
                embed_dim=SAM_PROMPT_EMBED_DIM,
                image_embedding_size=SAM_IMAGE_EMBEDDING_SIZE,
                input_image_size=(SAM_IMAGE_SIZE, SAM_IMAGE_SIZE),
                mask_in_chans=SAM_MASK_IN_CHANS)
            self.mask_decoder = MaskDecoder(
                num_multimask_outputs=SAM_NUM_MULTIMASK_OUTPUTS,
                transformer=TwoWayTransformer(
                    depth=SAM_TRANSFORMER_DEPTH,
                    embedding_dim=SAM_PROMPT_EMBED_DIM,
                    mlp_dim=SAM_TRANSFORMER_MLP_DIM,
                    num_heads=SAM_TRANSFORMER_HEADS),
                transformer_dim=SAM_PROMPT_EMBED_DIM,
                iou_head_depth=SAM_IOU_HEAD_DEPTH,
                iou_head_hidden_dim=SAM_IOU_HEAD_HIDDEN_DIM)

        self.weights_path = "" if self.scratch else str(weights)
        self.weights_report: dict[str, Any] = {"loaded": False}
        if not self.scratch:
            self.weights_report = {"loaded": True,
                                   **load_sam_subtrees(self.prompt_encoder,
                                                       self.mask_decoder,
                                                       self.weights_path)}

        # LISA.py:82-87: SAM frozen, only `mask_decoder` released when
        # train_mask_decoder=True (train_ds.py:96, default True).
        for p in self.prompt_encoder.parameters():
            p.requires_grad_(False)
        for p in self.mask_decoder.parameters():
            p.requires_grad_(not self.frozen_decoder)

    # -- one sample ---------------------------------------------------------
    def forward(self, feat: torch.Tensor, h_cond: torch.Tensor,
                grid: tuple[int, int] | None = None) -> dict[str, Any]:
        """``feat (1,in_dim,gh,gw)`` + ``h_cond (K, text_dim)`` -> logits/mask."""
        if feat.dim() != 4 or feat.shape[0] != 1:
            raise ValueError(f"expected feat (1,C,gh,gw), got {tuple(feat.shape)}")
        if h_cond.dim() != 2 or h_cond.shape[-1] != self.text_dim:
            raise ValueError(
                f"expected h_cond (K,{self.text_dim}), got {tuple(h_cond.shape)}")
        gh, gw = int(feat.shape[-2]), int(feat.shape[-1])
        if grid is not None and tuple(grid) not in ((0, 0), (gh, gw)):
            raise AssertionError(
                f"feat grid {(gh, gw)} != declared grid {tuple(grid)}")

        image_embeddings = self.neck(feat)                       # (1,256,gh,gw)
        # LISA.py:249-250 + :275-280.  K > 1 (the readout ablation rows) is the
        # proposal's pre-registered NOVEL default: each row through the SAME
        # text_hidden_fcs, then cat as K sparse tokens of ONE prompt
        # (prompt_encoder.py:176-177's own operation).  K = 1 is LISA exactly.
        text_embeds = self.text_hidden_fcs[0](h_cond).unsqueeze(0)   # (1,K,256)
        sparse, dense = self.prompt_encoder(
            points=None, boxes=None, masks=None, text_embeds=text_embeds,
            embedding_size=(gh, gw))
        sparse = sparse.to(text_embeds.dtype)                     # LISA.py:281
        image_pe = self.prompt_encoder.dense_pe_for((gh, gw))     # NOVEL, §3 (4)
        low_res_masks, iou_pred = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe.to(image_embeddings.dtype),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense.to(image_embeddings.dtype),
            multimask_output=False)                               # LISA.py:269
        logits_fine = low_res_masks[:, 0]                         # (1,4gh,4gw)

        # the criterion path's field: sigmoid, then the `gt_low` operator.  No
        # per-image normalisation anywhere (red line) -- area_resize is a fixed
        # linear projection.
        from .pixgt import area_project

        with no_autocast(feat.device.type):
            m_low = area_project(torch.sigmoid(logits_fine[0].float()), (gh, gw))
        return {"logits_fine": logits_fine, "m_low": m_low,
                "iou_pred": iou_pred, "grid": (gh, gw),
                "fine_grid": (int(logits_fine.shape[-2]), int(logits_fine.shape[-1])),
                "n_cond_vectors": int(h_cond.shape[0])}

    # -- bookkeeping --------------------------------------------------------
    def count(self, key: str, n: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + int(n)

    def param_counts(self) -> dict[str, int]:
        def n(mod: nn.Module) -> int:
            return int(sum(p.numel() for p in mod.parameters()))

        def t(mod: nn.Module) -> int:
            return int(sum(p.numel() for p in mod.parameters() if p.requires_grad))

        mods = {"text_hidden_fcs": self.text_hidden_fcs, "neck": self.neck,
                "prompt_encoder": self.prompt_encoder,
                "mask_decoder": self.mask_decoder}
        out = {k: n(m) for k, m in mods.items()}
        out.update({f"{k}_trainable": t(m) for k, m in mods.items()})
        out["total"] = int(sum(p.numel() for p in self.parameters()))
        out["trainable"] = int(sum(p.numel() for p in self.parameters()
                                   if p.requires_grad))
        out["frozen"] = out["total"] - out["trainable"]
        out["buffers"] = int(sum(b.numel() for b in self.buffers()))
        return out

    def facts(self) -> dict[str, Any]:
        """Goes into ``run_setup.json`` (``model.facts()['arm_head']``) and into
        every checkpoint.  Deliberately carries no ``n_stages`` /
        ``n_refine_layers`` / ``aux_groups`` key: this arm has no
        deep-supervision branch, and ``evaluate.deep_supervision_tags`` reads
        exactly those names."""
        return {
            "arm": ARM,
            "options": {"weights": self.weights_path, "scratch": self.scratch,
                        "frozen_decoder": self.frozen_decoder,
                        "dice_weight": self.dice_weight,
                        "bce_weight": self.bce_weight,
                        "gt": self.gt, "sup": self.sup, "seed": self.seed},
            "params": self.param_counts(),
            "trainable_modules": sorted(
                {n.split(".")[0] for n, p in self.named_parameters()
                 if p.requires_grad}),
            "frozen_modules": sorted(
                {n.split(".")[0] for n, p in self.named_parameters()
                 if not p.requires_grad}),
            "weights": self.weights_report,
            "structure": {
                "text_hidden_fcs": f"Linear({self.text_dim},{self.text_dim}) + "
                                   f"ReLU + Linear({self.text_dim},{self.out_dim})"
                                   " + Dropout(0.0)  [LISA.py:91-98]",
                "neck": f"Conv1x1({self.in_dim}->{self.out_dim},bias=False) + "
                        "LN2d + Conv3x3(bias=False) + LN2d  [PixelLM.py:175-191]",
                "mask_decoder": "transformer_dim 256 / TwoWayTransformer(depth 2,"
                                " mlp 2048, heads 8) / num_multimask_outputs 3 /"
                                " iou_head 3x256  [build_sam.py:87-98]",
                "multimask_output": False,
                "image_pe": "NOVEL: pe_layer((gh,gw)) per sample, not the fixed"
                            " 64x64 get_dense_pe()",
                "dense_prompt": "no_mask_embed expanded to (gh,gw)  NOVEL size",
                "fine_upscale": FINE_UPSCALE,
                "sup_upscale": SUP_UPSCALE[self.sup],
            },
            "loss": loss_form(self.bce_weight, self.dice_weight),
            "prereg_violation": {
                "rule": "DELTA 5.6 forbids coordinate / position channels "
                        "(heads.py:15-21)",
                "kept": "PositionEmbeddingRandom (prompt_encoder.py:189-238)",
                "execution_line": "corr_center_minus_corr_gt > 0 kills this "
                                  "structure (evaluate.py:413-416)",
            },
            "counts": dict(sorted(self.counts.items())),
            "sources": dict(SOURCES),
        }


# --------------------------------------------------------------------------- #
# loss: LISA.py:16-59 verbatim, LISA.py:321-335 composition
# --------------------------------------------------------------------------- #
def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float,
              scale: float = LISA_DICE_SCALE, eps: float = LISA_DICE_EPS) -> torch.Tensor:
    """``LISA.py:16-38``, verbatim (``scale`` 1000, ``eps`` 1e-6).

    Zero protection, checked in the tests: on ``targets == 0`` the numerator is
    exactly 0 and the loss is ``1 - eps/(sum(p/scale) + eps)`` -- finite for
    every prediction, 0 when the prediction is empty, ~1 when it is not.
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    return loss.sum() / (num_masks + LISA_NUM_MASKS_EPS)


def sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor,
                    num_masks: float) -> torch.Tensor:
    """``LISA.py:42-59``, verbatim."""
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.flatten(1, 2).mean(1).sum() / (num_masks + LISA_NUM_MASKS_EPS)


def lisa_mask_loss(logits_fine: torch.Tensor, gt_fine: torch.Tensor,
                   num_masks: float = 1.0, *,
                   bce_weight: float = LISA_BCE_WEIGHT,
                   dice_weight: float = LISA_DICE_WEIGHT
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(total, bce, dice)`` for one sample's ``(1, H, W)`` logits and target.

    LISA sums ``sigmoid_ce(pred_i, gt_i, n_i) * n_i`` over the batch and divides
    by the total mask count (``LISA.py:321-332``).  This arm has exactly one
    mask per sample, so the per-sample call takes ``num_masks = 1`` and the
    division by the batch's mask count is the micro-batch mean
    ``losses.aggregate`` already performs -- the same quantity, assembled by the
    trainer that owns the batch.
    """
    if logits_fine.shape != gt_fine.shape:
        raise AssertionError(
            f"logits {tuple(logits_fine.shape)} and target {tuple(gt_fine.shape)}"
            " must have the same shape")
    bce = sigmoid_ce_loss(logits_fine, gt_fine, num_masks)
    dice = dice_loss(logits_fine, gt_fine, num_masks)
    total = float(bce_weight) * bce + float(dice_weight) * dice
    return total, bce, dice


def loss_form(bce_weight: float = LISA_BCE_WEIGHT,
              dice_weight: float = LISA_DICE_WEIGHT) -> dict[str, Any]:
    """The pre-registration record of this arm's criterion.

    ``run_amort_arm`` writes one ``loss_preregistration.json`` with the live
    arms' seven-term form and ``dice_as_target: false`` hard-coded.  That record
    is FALSE for this arm, and the file has no per-arm hook, so the wrapper
    writes this one beside it.
    """
    return {
        "total": f"{bce_weight} * sigmoid_ce + {dice_weight} * dice",
        "sigmoid_ce": "F.binary_cross_entropy_with_logits(reduction='none')"
                      ".flatten(1,2).mean(1).sum() / (num_masks + 1e-8)"
                      "   [LISA.py:42-59]",
        "dice": "inputs.sigmoid().flatten(1,2); numerator = 2*(p/1000*t).sum(-1);"
                " denominator = (p/1000).sum(-1) + (t/1000).sum(-1);"
                " 1 - (num+1e-6)/(den+1e-6); .sum()/(num_masks+1e-8)"
                "   [LISA.py:16-38]",
        "weights_source": "train_ds.py:79-80 (dice 0.5, bce 2.0)",
        "ce_loss": "REMOVED (NOVEL): no text-generation branch, base frozen"
                   "   [LISA.py:307-308]",
        "num_masks": "1 per sample; the micro-batch mean in losses.aggregate is"
                     " LISA's division by the batch mask count",
        "target": "soft alpha in [0,1] (this repo's GT), formula unchanged"
                  "   [proposal 3.1 'GT value range']",
        "is_fake": "target zeroed, same formula, counted (proposal D-6 default)",
        "iou_as_field_target": False,
        "iou_as_selection_target": False,
        "dice_as_target": True,
        "dice_red_line_waiver": "the campaign's 'dice is never a mask target' "
                                "line (losses.py:9-11) is waived for the "
                                "EPR-018..023 batch by the user's 2026-08-14 "
                                "instruction; LISA's 0.5 is kept verbatim",
    }


def loss_preregistration(args: Any = None) -> dict[str, Any]:
    """``arms.RUN_REQUIRED_HOOKS``: what ``loss_preregistration.json`` says.

    Same content as the ``loss_preregistration_segsam.json`` the wrapper
    writes beside it -- one function, so the two files cannot disagree.
    """
    o = options()
    return {"arm": ARM,
            **loss_form(float(o["bce_weight"]), float(o["dice_weight"]))}


# --------------------------------------------------------------------------- #
# registry hooks
# --------------------------------------------------------------------------- #
def build_head(*, in_dim: int, text_dim: int, args: Any = None, **kw):
    o = options()
    params = dict(
        in_dim=in_dim, text_dim=text_dim, out_dim=LISA_OUT_DIM,
        weights=str(o["weights"]), scratch=bool(o["scratch"]),
        frozen_decoder=bool(o["frozen_decoder"]),
        dice_weight=float(o["dice_weight"]), bce_weight=float(o["bce_weight"]),
        gt=str(o["gt"]), sup=str(o["sup"]),
        seed=int(getattr(args, "seed", 20260810) or 20260810),
    )
    params.update(kw)
    return SegSamHead(**params)


def head_kwargs_from_args(args: Any) -> dict[str, Any]:
    """Nothing extra: :func:`build_head` reads :func:`options` itself, so the
    three namespaces the entry script hands the hooks cannot disagree about what
    this arm was configured with."""
    return {}


def forward(model, head, ctx) -> dict[str, Any]:
    """Registry hook -- one sample through the head."""
    if ctx.extra is not None:
        raise AssertionError(
            "arm SEGSAM takes no extra stem channels: LISA's decoder sees the "
            "image embedding and the text prompt and nothing else.  Pass "
            "--no-sim-field (and drop --center-prior-channel / --geom-inject) "
            "-- run_segsam_arm.py does this for you.")
    h_cond = ctx.require_cond(ARM)                       # (K, 2560)
    res = head(ctx.feat, h_cond, grid=(ctx.grid_h, ctx.grid_w))
    return {"m_low": res["m_low"], "segsam": res, "params": {}}


def _target_for(x, head, logits_fine: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, str]:
    """``(logits_at_supervision, target, note)`` for one sample.

    ``sup=native``: the decoder's own ``4x(gh,gw)`` grid; the pixel GT provider
    is configured to render there, and anything else is projected with the
    ``gt_low`` operator rather than silently accepted.
    ``sup=cgt``: LISA's ``postprocess_masks`` chain -- the logits are
    interpolated up to the ``.cgt`` grid (bilinear, ``align_corners=False``,
    ``LISA.py:289-293``) and the loss is taken there.
    """
    from .pixgt import area_project

    gt = getattr(x, "gt_pix", None)
    if gt is None:
        raise AssertionError(
            f"{getattr(x, 'sample_id', '?')}: arm SEGSAM supervises at pixel "
            "resolution and the batch carried no gt_pix.  --no-pixgt is not "
            "compatible with this arm (run_segsam_arm.py refuses it).")
    gt = gt.to(logits_fine.device).float()
    if gt.dim() != 2:
        raise AssertionError(f"gt_pix must be (H,W), got {tuple(gt.shape)}")
    if head.sup == "cgt":
        if tuple(gt.shape) == tuple(logits_fine.shape[-2:]):
            return logits_fine, gt[None], "cgt_same_grid"
        up = F.interpolate(logits_fine[None].float(), size=tuple(gt.shape),
                           mode="bilinear", align_corners=False)[0]
        return up, gt[None], "cgt_interpolated"
    if tuple(gt.shape) != tuple(logits_fine.shape[-2:]):
        gt = area_project(gt, (int(logits_fine.shape[-2]), int(logits_fine.shape[-1])))
        return logits_fine, gt[None], "area_projected"
    return logits_fine, gt[None], "native"


#: The ``L_*`` columns this arm's loss emitted on its FIRST training
#: micro-batch, captured in-process.
#:
#: ``steps.jsonl`` is the artifact the arm pre-registers, but the trainer's
#: handle is block-buffered and only flushed every ``log_every`` steps
#: (``trainer.py:608``), so a quick eval that lands on a non-flush step reads an
#: EMPTY file -- indistinguishable, from disk alone, from "the loss never ran".
#: This witness is the same claim one link earlier in the chain:
#: ``losses.aggregate`` prefixes ``L_`` to exactly these keys
#: (``losses.py:542``) and the trainer splices its output into the row verbatim
#: (``trainer.py:606``), so a column here is a column there.
FIRST_LOSS_COLUMNS: dict[str, float] = {}


def _witness(terms: dict[str, Any]) -> dict[str, Any]:
    """Record the first micro-batch's ``L_*`` columns; ``terms`` unchanged."""
    if not FIRST_LOSS_COLUMNS:
        FIRST_LOSS_COLUMNS.update({f"L_{k}": float(v.detach())
                                   for k, v in terms.items()})
    return terms


def compute_loss(model, out: dict[str, Any], x, weights) -> AmortLoss:
    """Registry hook -- ``2.0 * sigmoid_ce + 0.5 * dice`` for one sample."""
    head = getattr(model, "geo", None)
    res = out.get("segsam")
    if res is None or head is None:
        raise AssertionError(
            "arm SEGSAM: compute_loss was handed a forward with no 'segsam' "
            "record; the loss and the field would then be measuring different "
            "things")
    logits_fine = res["logits_fine"]
    with no_autocast(logits_fine.device.type):
        logits, target, note = _target_for(x, head, logits_fine)
        logits = logits.float()
        is_fake = bool(getattr(x, "is_fake", False))
        if is_fake:
            # D-6 default: foreign instructions get an all-zero target and the
            # unchanged formula.  LISA never trains on an empty GT, so this is
            # NOVEL and pre-registered; the numbers are counted separately.
            target = torch.zeros_like(target)
        head.count("n_samples")
        head.count(f"gt_source_{getattr(x, 'gt_pix_source', '') or 'unset'}")
        head.count(f"target_{note}")
        if is_fake:
            head.count("n_fake")

        if (not is_fake) and float(target.max()) <= 0.0:
            # Degenerate GT: counted, printed once in a while, and excluded
            # from the gradient rather than silently trained against zeros
            # (which is the fake-sample target and would blur the two).
            head.count("n_gt_allzero_excluded")
            n = head.counts["n_gt_allzero_excluded"]
            if n <= 5 or n % 100 == 0:
                print(f"SEGSAM all-zero GT excluded ({n} so far): "
                      f"{getattr(x, 'sample_id', '?')} family="
                      f"{getattr(x, 'family', '?')} source="
                      f"{getattr(x, 'gt_pix_source', '')}", flush=True)
            zero = logits.sum() * 0.0
            return AmortLoss(total=zero,
                             terms=_witness({"bce_lisa": zero,
                                             "dice_lisa": zero}),
                             stats={"segsam_excluded": 1.0,
                                    "segsam_is_fake": float(is_fake)})

        total, bce, dice = lisa_mask_loss(
            logits, target, num_masks=1.0,
            bce_weight=head.bce_weight, dice_weight=head.dice_weight)

    terms = {"bce_lisa": bce, "dice_lisa": dice}
    if is_fake:
        # `aggregate` averages each term over the samples that carry it, so the
        # duplicated names give the fake subset's own two columns in
        # steps.jsonl next to `n_fake` (proposal 3.1, last row).
        terms["bce_lisa_fake"] = bce
        terms["dice_lisa_fake"] = dice
    stats = {"segsam_is_fake": float(is_fake), "segsam_excluded": 0.0,
             "segsam_fine_h": float(logits.shape[-2]),
             "segsam_fine_w": float(logits.shape[-1])}
    return AmortLoss(total=total, terms=_witness(terms), stats=stats)


def train_stats(out: dict[str, Any], x) -> dict[str, Any]:
    """Per-sample training columns (trainer seam).

    ``compute_micro_batch`` builds these rows and the run loop then discards
    them -- only ``losses.aggregate``'s output reaches ``steps.jsonl``, and it
    forwards no arm-specific stat.  The hook is implemented anyway (it is part
    of the contract and costs nothing); the counters that matter travel in
    ``facts()`` and in the board's criterion column.
    """
    res = out.get("segsam") or {}
    return {"segsam_fine_grid": res.get("fine_grid"),
            "segsam_n_cond_vectors": res.get("n_cond_vectors"),
            "segsam_gt_pix_source": str(getattr(x, "gt_pix_source", "") or "")}


def optimizer_spec(args: Any):
    """AdamW, ``wd 0.0``, ``betas (0.9, 0.95)``, one parameter group.

    DeepSpeed's AdamW (``train_ds.py:271-278``) applies one decay to everything;
    with ``weight_decay = 0`` all three groupings coincide numerically, and
    ``"none"`` is the one that says so in the run record.
    """
    from .trainer import OptimizerSpec

    lr = float(getattr(args, "lr", LISA_LR) or LISA_LR)
    if abs(lr - LISA_LR) > 1e-12:
        print(f"SEGSAM NOTE: --lr {lr} deviates from LISA's {LISA_LR} "
              "(train_ds.py:77); the run record keeps both.", flush=True)
    return OptimizerSpec(type="adamw", lr=lr, weight_decay=LISA_WEIGHT_DECAY,
                         betas=LISA_BETAS, grouping="none")


def warmup_for(total_steps: int) -> int:
    """``warmup_num_steps 100`` of 5000 carried over as a fraction (U4)."""
    total = int(total_steps or PREREGISTERED_TOTAL_STEPS)
    return max(1, int(round(total * LISA_WARMUP_FRAC)))


def scheduler_kwargs(args: Any, total_steps: int) -> dict[str, Any]:
    """DeepSpeed ``WarmupDecayLR``: linear warmup, then linear decay to 0.

    The *kind* lives on ``cfg.scheduler`` (``--scheduler``), which this hook
    cannot set, so it is asserted here instead: a SEGSAM run on the campaign
    cosine would be a different recipe with an identical-looking board.

    The warmup is a fraction of the run's declared horizon (``--max-steps``,
    1200 by default); the decay is over the trainer's own ``total_steps``,
    which equals it whenever the data allows the full horizon -- 42752 train
    samples at batch 32 is 1336 steps per epoch, so it does at this档.
    """
    kind = str(getattr(args, "scheduler", "cosine"))
    if kind != "linear":
        raise SystemExit(
            f"arm SEGSAM needs --scheduler linear (DeepSpeed WarmupDecayLR, "
            f"train_ds.py:279-287) but the run was given {kind!r}.  "
            "run_segsam_arm.py pins it; do not pass --scheduler by hand.")
    total = int(total_steps or getattr(args, "max_steps", 0) or 0)
    return {"warmup_steps": warmup_for(total)}


def builder_kwargs(args: Any) -> dict[str, Any]:
    """The pixel-GT resolution, and the ``--segsam-gt`` guard.

    ``pixgt_size`` is the resolution the loss is taken at: ``4x(gh,gw)`` on the
    main arm (the decoder's native grid) and ``32x(gh,gw)`` under
    ``--segsam-sup cgt`` (the ``.cgt`` grid: spec-5 short side 512, ``F_pre``
    = H/16, so ``.cgt``'s short side 1024 is 32 token cells).
    """
    o = options()
    want_src, want_fb = GT_TO_PIXGT[str(o["gt"])]
    got_src = str(getattr(args, "pixgt_source", want_src))
    got_fb = str(getattr(args, "pixgt_fallback", want_fb))
    if got_src != want_src or got_fb != want_fb:
        raise SystemExit(
            f"--segsam-gt {o['gt']!r} needs --pixgt-source {want_src!r} "
            f"--pixgt-fallback {want_fb!r} but the run was given "
            f"{got_src!r}/{got_fb!r}.  Refusing to start: the run record would "
            f"say {o['gt']!r} while the supervision came from somewhere else.  "
            "run_segsam_arm.py sets both for you.")
    if getattr(args, "no_pixgt", False):
        raise SystemExit(
            "arm SEGSAM supervises at pixel resolution through the pixgt "
            "provider; --no-pixgt would leave it with no target at all.")
    up = SUP_UPSCALE[str(o["sup"])]
    return {"pixgt_size": (lambda gh, gw, _u=up: (_u * gh, _u * gw))}


# --------------------------------------------------------------------------- #
# evaluation: the pre-registered criterion column
# --------------------------------------------------------------------------- #
def per_sample_row(model, out: dict[str, Any], x) -> dict[str, Any]:
    """Per-sample fine-grid diagnostics (evaluator seam).

    The headline stays on ``m_low`` and the frozen path; this column says what
    the decoder's own ``4x`` field looks like, with the same three-column set
    the criteria red line demands (soft-IoU + boundary F1 + centre prior) and
    the ``a/(2-a)`` random floor beside it.

    A failure here must not take the board down (D-20 lesson 6) and must not
    vanish either, so it lands in its own ``segsam_fine_error`` column -- and a
    board where every row errored has ``n = 0`` and cannot publish.
    """
    from q3vl.whereb.metrics import (gt_area_k, grid_boundary_f1, hard_iou,
                                     soft_iou_value, topk_mask)

    from .evaluate import center_prior_unit, random_floor
    from .pixgt import area_project

    res = out.get("segsam") or {}
    row: dict[str, Any] = {
        "segsam_gt_pix_source": str(getattr(x, "gt_pix_source", "") or ""),
        "segsam_n_cond_vectors": res.get("n_cond_vectors"),
        "segsam_fine_topk_iou": None,
    }
    logits = res.get("logits_fine")
    gt_pix = getattr(x, "gt_pix", None)
    if logits is None or gt_pix is None:
        row["segsam_fine_error"] = ("no logits_fine" if logits is None
                                    else "no gt_pix in the batch")
        return row
    try:
        pred = torch.sigmoid(logits[0].detach().float())
        fh, fw = int(pred.shape[-2]), int(pred.shape[-1])
        gt = gt_pix.detach().to(pred.device).float()
        if tuple(gt.shape) != (fh, fw):
            gt = area_project(gt, (fh, fw))
        k = gt_area_k(gt)
        cp = center_prior_unit(fh, fw, device=pred.device)
        pred_k, gt_k, cp_k = topk_mask(pred, k), topk_mask(gt, k), topk_mask(cp, k)
        area = float((gt > 0.5).float().mean())
        row.update({
            "segsam_fine_h": fh, "segsam_fine_w": fw,
            "segsam_fine_soft_iou": soft_iou_value(pred, gt),
            "segsam_fine_topk_iou": hard_iou(pred_k, gt_k),
            "segsam_fine_boundary_f1": grid_boundary_f1(
                pred_k, gt_k, tol_cells=BOUNDARY_TOL_FINE),
            "segsam_fine_center_soft_iou": soft_iou_value(cp, gt),
            "segsam_fine_center_topk_iou": hard_iou(cp_k, gt_k),
            "segsam_fine_center_boundary_f1": grid_boundary_f1(
                cp_k, gt_k, tol_cells=BOUNDARY_TOL_FINE),
            "segsam_fine_random_floor": random_floor(area),
            "segsam_fine_gt_area_frac": area,
            "segsam_fine_pred_area_frac": float(pred.mean()),
        })
    except Exception as exc:  # noqa: BLE001 -- recorded, never silent
        row["segsam_fine_error"] = f"{type(exc).__name__}: {exc}"
    return row


def criteria_columns(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """``segsam_fine`` -- the pre-registered column (proposal §3 (9)).

    Built from the SAME rows the headline is built from.  ``n`` is the count of
    fine-grid top-k IoU values; ``assert_criteria_ran`` refuses a board with
    ``n = 0``, which is this arm's "no fine field, no publication".
    """
    from q3vl.whereb.metrics import paired_delta

    from .evaluate import _agg

    live = [r for r in rows if r.get("segsam_fine_topk_iou") is not None]
    topk = [float(r["segsam_fine_topk_iou"]) for r in live]
    center = [float(r["segsam_fine_center_topk_iou"]) for r in live]
    floor = [float(r["segsam_fine_random_floor"]) for r in live]
    src: dict[str, int] = {}
    for r in rows:
        key = str(r.get("segsam_gt_pix_source") or "unset")
        src[key] = src.get(key, 0) + 1
    col: dict[str, Any] = {
        **_agg(topk),
        "soft_iou": _agg(r["segsam_fine_soft_iou"] for r in live),
        "boundary_f1_tol4": _agg(r["segsam_fine_boundary_f1"] for r in live),
        "center_prior_topk_iou": _agg(center),
        "center_prior_soft_iou": _agg(r["segsam_fine_center_soft_iou"] for r in live),
        "center_prior_boundary_f1": _agg(
            r["segsam_fine_center_boundary_f1"] for r in live),
        "random_floor": _agg(floor),
        "gt_area_frac": _agg(r["segsam_fine_gt_area_frac"] for r in live),
        "pred_area_frac": _agg(r["segsam_fine_pred_area_frac"] for r in live),
        "gt_pix_sources": dict(sorted(src.items())),
        "n_errors": sum(1 for r in rows if r.get("segsam_fine_error")),
        "grid": "decoder-native 4x(gh,gw)",
        "boundary_tol_cells": BOUNDARY_TOL_FINE,
        "note": ("the decoder's own fine field, matched-area top-k; the "
                 "headline stays on m_low and the frozen path.  Three-column "
                 "set: soft-IoU (min/max) + grid boundary F1 (tol 4 fine cells "
                 "= 1 coarse cell) + the centre prior on the same support, with "
                 "the a/(2-a) random floor beside it."),
    }
    if live:
        col["delta_vs_center_prior"] = paired_delta(topk, center)
        col["delta_vs_random_floor"] = paired_delta(topk, floor)
    return {"segsam_fine": col}


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
    agree on their arguments, which is what broke the 2026-08-15 smoke run:

    * ``run_amort_arm._finish_board`` (``run_amort_arm.py:1045-1053``) reads
      ``steps.jsonl`` itself and passes the row -> source ``caller``;
    * ``evaluate.evaluate_arm`` (``evaluate.py:660``) passes ``board`` and
      ``arm`` and NOTHING ELSE -- and it is the call site that gates
      ``metrics.json`` (written three lines below it), so the check has to work
      there.  It reads the artifact off disk instead.

    Third fallback: :data:`FIRST_LOSS_COLUMNS`, for a quick eval that lands
    before the trainer flushed its write buffer.  ``(None, "unavailable")`` --
    nothing on disk and no loss call in this process -- is a failure, not a
    pass; :func:`assert_publishable` raises on it.
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


def assert_publishable(board: dict[str, Any], *, steps_row: Any = None,
                       eval_only: bool = False, steps_path: Any = None
                       ) -> dict[str, Any]:
    """The arm's own runtime assertions, run before ``metrics.json`` is written.

    1. ``steps.jsonl``'s FIRST row carries ``L_bce_lisa`` and ``L_dice_lisa``
       (proposal §3 (9)): a run whose loss columns are missing from step one is
       a run whose loss is not the ported one.  The row comes from
       :func:`resolve_steps_row`, so the check runs at BOTH call sites of
       ``evaluate.assert_criteria_ran`` -- passing no row is not a way through.
    2. the criterion column carries the whole three-column set, not just the
       headline number -- a partially computed column is the "defined but not
       wired" failure with extra steps.
    """
    report: dict[str, Any] = {"arm": ARM, "checks": {}}
    want_cols = ("L_bce_lisa", "L_dice_lisa")
    if eval_only:
        report["checks"]["loss_columns"] = "skipped (eval_only: no training steps)"
    else:
        row, source = resolve_steps_row(steps_row, steps_path=steps_path)
        if row is None:
            raise AssertionError(
                f"arm SEGSAM pre-registers {list(want_cols)} in steps.jsonl and "
                "no first row is available anywhere: the caller passed none, "
                f"{steps_path or '<no steps.jsonl path given>'} carries no row, "
                "and this arm's loss has not run in this process.  Refusing to "
                "publish a board that cannot show its loss ran.")
        missing = [c for c in want_cols if c not in row]
        if missing:
            raise AssertionError(
                f"arm SEGSAM pre-registers {list(want_cols)} in steps.jsonl and "
                f"the first row [{source}] is missing {missing}; the ported "
                f"loss did not run (columns present: "
                f"{sorted(k for k in row if k.startswith('L_'))})")
        report["checks"]["loss_columns"] = {c: row[c] for c in want_cols}
        report["checks"]["loss_columns_source"] = source

    col = (board.get("criteria_columns") or {}).get("segsam_fine") or {}
    need = ("soft_iou", "boundary_f1_tol4", "center_prior_topk_iou",
            "random_floor", "delta_vs_center_prior")
    absent = [c for c in need if col.get(c) is None]
    if int(col.get("n", 0)) <= 0 or absent:
        raise AssertionError(
            f"arm SEGSAM's criterion column is incomplete: n={col.get('n', 0)}, "
            f"missing {absent}.  The three-column set (soft-IoU + boundary F1 + "
            "centre prior) and the random floor are not optional.")
    report["checks"]["criterion_column"] = {
        "n": col["n"], "median": col.get("median"),
        "center_prior_median": (col.get("center_prior_topk_iou") or {}).get("median"),
        "random_floor_median": (col.get("random_floor") or {}).get("median"),
        "delta_vs_center_prior": col.get("delta_vs_center_prior"),
    }
    g = board.get("corr_center_minus_corr_gt") or {}
    report["checks"]["m3_guard"] = {
        "corr_center_minus_corr_gt": g.get("delta"), "p": g.get("p"),
        "m3_disease_present": board.get("m3_disease_present"),
        "note": "DELTA 5.6 execution line for this arm's PositionEmbeddingRandom",
    }
    return report


# --------------------------------------------------------------------------- #
# entry-script helpers
# --------------------------------------------------------------------------- #
def add_arguments(ap) -> None:
    """The ``--segsam-*`` flag surface (proposal, entry row).

    ``default=None`` throughout so :func:`configure` can tell "not passed" from
    "passed the default value", and :data:`DEFAULTS` stays the single source of
    truth for what the arm ran with.
    """
    ap.add_argument("--segsam-weights", dest="segsam_weights", default=None,
                    help=f"SAM ViT-H checkpoint; only mask_decoder.*/"
                         f"prompt_encoder.* are read (default {SAM_VIT_H_DEFAULT})")
    ap.add_argument("--segsam-scratch", dest="segsam_scratch",
                    action="store_true", default=None,
                    help="ablation (4): same structure, no pretrained weights")
    ap.add_argument("--segsam-frozen-decoder", dest="segsam_frozen_decoder",
                    action="store_true", default=None,
                    help="ablation (3): train_mask_decoder=False -- only neck "
                         "and text_hidden_fcs train")
    ap.add_argument("--segsam-dice-weight", dest="segsam_dice_weight",
                    type=float, default=None,
                    help=f"ablation (2): {LISA_DICE_WEIGHT} = LISA, 0.0 = BCE only")
    ap.add_argument("--segsam-gt", dest="segsam_gt", default=None,
                    choices=sorted(GT_TO_PIXGT),
                    help="ablation (1): raster = analytic re-render (semantic "
                         "falls back to .cgt and is counted); png = all four "
                         "families from .cgt")
    ap.add_argument("--segsam-sup", dest="segsam_sup", default=None,
                    choices=sorted(SUP_UPSCALE),
                    help="ablation (5): native = the decoder's 4x grid; cgt = "
                         "interpolate to the .cgt grid first (LISA's "
                         "postprocess_masks chain)")


def setup_record() -> dict[str, Any]:
    """The arm's pre-registration record, merged into ``run_setup.json``."""
    o = options()
    src, fb = GT_TO_PIXGT[str(o["gt"])]
    return {"segsam": {
        "options": o,
        "pixgt": {"source": src, "fallback": fb,
                  "size": f"{SUP_UPSCALE[str(o['sup'])]}x(gh,gw)"},
        "loss": loss_form(float(o["bce_weight"]), float(o["dice_weight"])),
        "optimizer": {"type": "adamw", "lr": LISA_LR,
                      "weight_decay": LISA_WEIGHT_DECAY,
                      "betas": list(LISA_BETAS), "grouping": "none",
                      "grad_clip": LISA_GRAD_CLIP,
                      "source": "train_ds.py:271-278, :85-86, :295"},
        "scheduler": {"kind": "linear",
                      "warmup_steps_at_1200": warmup_for(PREREGISTERED_TOTAL_STEPS),
                      "reference": f"WarmupDecayLR warmup 100 / "
                                   f"{LISA_REFERENCE_TOTAL_STEPS} steps "
                                   f"(train_ds.py:279-287), carried over as a "
                                   f"fraction (U4 step matching)"},
        "novel": [
            "image encoder -> frozen Qwen3-VL F_pre + PixelLM-shaped neck",
            "[SEG] -> the v2seg supervised <seg_where> row (readout.py)",
            "text CE / LoRA / lm_head / embed_tokens removed",
            "image_pe and the dense prompt sized per sample, not 64x64",
            "postprocess_masks dropped on the main arm (--segsam-sup cgt row)",
            "is_fake samples: zeroed target, unchanged formula (D-6 default)",
        ],
        "prereg_violation": "PositionEmbeddingRandom is a coordinate encoding "
                            "(DELTA 5.6); execution line = "
                            "corr_center_minus_corr_gt",
        "criterion_column": CRITERIA[0],
        "sources": dict(SOURCES),
    }}
