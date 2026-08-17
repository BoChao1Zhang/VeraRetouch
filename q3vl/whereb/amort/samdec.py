"""EPR-019 SAMDEC: SAM's mask decoder, ported verbatim and trained FROM SCRATCH.

The arm module of the EPR-018..023 registry (``q3vl/whereb/amort/arms.py``):
``build_head`` / ``forward`` / ``compute_loss`` plus the optional hooks.  Nothing
outside this file, ``q3vl/whereb/scripts/run_samdec_arm.py`` and
``q3vl/whereb/tests/test_samdec.py`` is touched.

What is ported, and from where
------------------------------
Every structural line below is the upstream file, opened 2026-08-14 by ``curl``
on ``raw.githubusercontent.com`` and read in full:

``segment_anything/modeling/mask_decoder.py``
    ``MaskDecoder`` (L16-164): ``iou_token = Embedding(1, 256)``,
    ``num_mask_tokens = num_multimask_outputs + 1 = 4``, the two-stage
    ``output_upscaling`` (ConvT 256->64, LayerNorm2d, GELU, ConvT 64->32, GELU),
    four per-token hypernetwork ``MLP(256, 256, 32, 3)``, the
    ``MLP(256, 256, 4, 3)`` IoU head, ``multimask_output -> slice(1, None)``,
    ``tokens = cat([iou_token, mask_tokens, sparse_prompt])``,
    ``masks = hyper_in @ upscaled_embedding.view(b, c, h*w)``.  ``MLP`` (L169-191).
``segment_anything/modeling/transformer.py``
    ``TwoWayTransformer`` / ``TwoWayAttentionBlock`` / ``Attention`` (whole file),
    including ``skip_first_layer_pe=(i == 0)``, the trailing
    ``final_attn_token_to_image`` + ``norm_final_attn``, and the 2x q/k/v
    downsample on every cross-attention.
``segment_anything/modeling/common.py``
    ``MLPBlock`` and ``LayerNorm2d`` (channel-wise ConvNeXt/detectron2 form,
    eps 1e-6 -- NOT ``GroupNorm(1, C)``).
``segment_anything/modeling/prompt_encoder.py``
    ``PositionEmbeddingRandom`` (L~240 in the current file: buffer
    ``scale * randn(2, num_pos_feats)``, ``2*coords - 1``, ``2*pi``,
    ``cat([sin, cos])``, ``y_embed = cumsum - 0.5`` normalised by ``h``) and
    ``no_mask_embed`` broadcast as the dense prompt.
``segment_anything/modeling/image_encoder.py``
    the neck: ``Conv2d(embed_dim, 256, 1, bias=False) + LayerNorm2d(256) +
    Conv2d(256, 256, 3, padding=1, bias=False) + LayerNorm2d(256)``.
``segment_anything/build_sam.py``
    the hyper-parameters: ``prompt_embed_dim = 256``, ``TwoWayTransformer(depth=2,
    embedding_dim=256, mlp_dim=2048, num_heads=8)``, ``iou_head_depth=3``,
    ``iou_head_hidden_dim=256``; ``build_sam_vit_l``'s ``encoder_embed_dim =
    1024`` is the same 1024 this campaign's ``F_pre`` carries, so the neck's
    input width is copied rather than adapted.
``facebookresearch/sam2 training/loss_fns.py``
    the loss constants SAM v1's paper leaves to its (unreleased) training code:
    focal ``alpha = 0.25`` / ``gamma = 2`` with a spatial ``mean``, dice with
    ``+1`` in numerator and denominator, IoU-head ``mse_loss``, and the
    lowest-``20*focal + 1*dice`` winner-takes-all whose IoU term is taken at the
    *same* index (comment: "to be consistent w/ SAM").

Three deviations, each pre-registered in the proposal and each recorded in
``facts()`` -- they are NOT silent:

1. the image encoder is the frozen Qwen3-VL, not a trainable MAE-initialised ViT
   (the campaign-wide constraint; SAM's own decoder is from scratch either way);
2. ``prompt_proj = Linear(2560, 256)`` maps the ``<seg_where>`` hidden onto one
   sparse prompt token -- SAM has no text prompt encoder to copy (NOVEL,
   proposal NOTES 6);
3. the ablation ``--no-samdec-iou-head`` removes ``iou_token`` /
   ``iou_prediction_head`` from the ported class, which upstream always builds.

Nothing here loads a SAM checkpoint.  Loading pre-trained weights is EPR-018's
variable; this arm's whole point is the same structure trained from zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from q3vl.where.upsample import area_resize

__all__ = [
    "ARM", "CRITERIA", "VARIANT", "SAM_SOURCES",
    "LayerNorm2d", "MLPBlock", "MLP", "Attention", "TwoWayAttentionBlock",
    "TwoWayTransformer", "MaskDecoder", "PositionEmbeddingRandom",
    "SAMDecHead", "SamLossConfig", "sam_mask_loss", "assert_steps_row",
    "add_arguments", "variant_from_args", "set_variant", "config_of",
    "loss_config", "loss_preregistration",
    "build_head", "forward", "compute_loss", "head_kwargs_from_args",
    "optimizer_spec", "scheduler_kwargs", "builder_kwargs", "readout_spec",
    "per_sample_row", "criteria_columns", "train_stats",
]

#: registry name and the pre-registered criterion column
#: (``q3vl/whereb/amort/arms.py`` ``ARM_CRITERIA["SAMDEC"]``).  A board whose
#: ``criteria_columns["samdec_cand"]`` carries ``n = 0`` cannot publish.
ARM = "SAMDEC"
CRITERIA: tuple[str, ...] = ("samdec_cand",)

#: the raw files this port was written against (2026-08-14).  Recorded in
#: ``facts()`` so a board says which upstream revision it copied.
SAM_SOURCES: tuple[str, ...] = (
    "https://raw.githubusercontent.com/facebookresearch/segment-anything/main/"
    "segment_anything/modeling/mask_decoder.py",
    "https://raw.githubusercontent.com/facebookresearch/segment-anything/main/"
    "segment_anything/modeling/transformer.py",
    "https://raw.githubusercontent.com/facebookresearch/segment-anything/main/"
    "segment_anything/modeling/common.py",
    "https://raw.githubusercontent.com/facebookresearch/segment-anything/main/"
    "segment_anything/modeling/prompt_encoder.py",
    "https://raw.githubusercontent.com/facebookresearch/segment-anything/main/"
    "segment_anything/modeling/image_encoder.py",
    "https://raw.githubusercontent.com/facebookresearch/segment-anything/main/"
    "segment_anything/build_sam.py",
    "https://raw.githubusercontent.com/facebookresearch/sam2/main/training/loss_fns.py",
    "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/"
    "sam2.1_training/sam2.1_hiera_b%2B_MOSE_finetune.yaml",
    "https://ar5iv.labs.arxiv.org/html/2304.02643",
)

#: The pre-registered configuration: **every number here is SAM's own** (paper
#: appendix A + the two official code sources above).  The wrapper
#: ``run_samdec_arm.py`` overwrites entries from its ``--samdec-*`` flags; the
#: flags exist so ``run_setup.json`` carries a record of the values, not because
#: they are meant to be tuned (proposal §3.3, entry row).
VARIANT: dict[str, Any] = {
    "multimask": 3,          # multimask_output=True -> slice(1, None)
    "loss": "focal_dice",    # focal_dice | focal | bce_dice | bce
    "gt": "png",             # png -> .cgt.png @1024 | raster -> analytic re-render
    "iou_head": True,
    "prompt": "one",         # one sparse token from h_cond | span (ablation 10)
    "lr": 8e-4,              # SAM §A "The initial learning rate ... is 8e-4"
    "wd": 0.1,               # SAM §A "we set weight decay (wd) to 0.1"
    "sched": "sam_step",     # linear warmup 250/90k, x0.1 @ 60k and @ 86666
    "focal_alpha": 0.25,     # sam2 training/loss_fns.py sigmoid_focal_loss
    "focal_gamma": 2.0,      # ditto
    "w_focal": 20.0,         # SAM §A "20:1 ratio of focal loss to dice loss"
    "w_dice": 1.0,
    "w_iou": 1.0,            # SAM §A "constant scaling factor of 1.0"
    "seed": 20260810,
}

#: SAM §A's schedule, as fractions of its own 90k horizon.  Carried over to this
#: campaign's 1200 steps by :func:`q3vl.where.calibrate.scale_milestones`
#: (proposal §3.2: warmup 3 steps, x0.1 at 800 and 1156) -- the shape of the
#: schedule is preserved, the absolute counts are not, which is the one
#: adaptation the step-matching discipline (U4) forces.
SAM_TOTAL_ITERS = 90000
SAM_WARMUP_ITERS = 250
SAM_MILESTONE_FRACS: tuple[float, float] = (60000 / 90000, 86666 / 90000)
SAM_GAMMA = 0.1

#: the GT-side threshold of the IoU head's regression target.  SAM2 uses
#: ``targets > 0``; this campaign's GT is a **soft alpha**, where ``> 0`` would
#: count the whole feather band as foreground.  0.5 is the threshold every
#: criterion already uses (``q3vl/whereb/metrics.py:142-144`` ``gt_area_k``),
#: so the head regresses the same quantity the board reports.  NOVEL, proposal
#: §3.1 last-but-one row.
IOU_GT_THRESHOLD = 0.5

#: the loss columns ``steps.jsonl`` must carry from its FIRST row on.
#: ``losses.aggregate`` (``q3vl/whereb/amort/losses.py:539-542``) prefixes every
#: ``AmortLoss.terms`` key with ``L_``, so the proposal's ``sup_cells`` witness
#: is spelled ``L_sup_cells`` on the board.  Both spellings are accepted by
#: :func:`assert_steps_row`; the module emits the prefixed one.
STEP_WITNESS_COLUMNS: tuple[str, ...] = ("L_focal", "L_dice", "L_iouhead")
SUP_CELLS_COLUMNS: tuple[str, ...] = ("L_sup_cells", "sup_cells")


# --------------------------------------------------------------------------- #
# ported SAM modules  (common.py)
# --------------------------------------------------------------------------- #
class LayerNorm2d(nn.Module):
    """``segment_anything/modeling/common.py`` -- channel-wise, eps 1e-6.

    Deliberately not ``nn.GroupNorm(1, C)``: that normalises over channels *and*
    space, this normalises per spatial position over channels only.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MLPBlock(nn.Module):
    """``common.py`` ``MLPBlock``.  The transformer passes ``act=nn.ReLU``."""

    def __init__(self, embedding_dim: int, mlp_dim: int,
                 act: type[nn.Module] = nn.GELU) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class MLP(nn.Module):
    """``mask_decoder.py`` ``MLP``: ``num_layers`` Linears, ReLU between them.

    Same shape as this repo's ``uniq.SelMLP``; deliberately re-declared rather
    than imported, so that ``--arm SAMDEC`` pulls in none of the ST_LANG module
    graph (proposal §3.3 ①).
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int, sigmoid_output: bool = False) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.sigmoid_output = sigmoid_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


# --------------------------------------------------------------------------- #
# ported SAM modules  (transformer.py)
# --------------------------------------------------------------------------- #
class Attention(nn.Module):
    """``transformer.py`` ``Attention``: q/k/v reduced by ``downsample_rate``."""

    def __init__(self, embedding_dim: int, num_heads: int,
                 downsample_rate: int = 1) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        if self.internal_dim % num_heads:
            raise ValueError("num_heads must divide embedding_dim.")
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

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                v: torch.Tensor) -> torch.Tensor:
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
    """``transformer.py`` ``TwoWayAttentionBlock`` -- four sub-layers, in order."""

    def __init__(self, embedding_dim: int, num_heads: int, mlp_dim: int = 2048,
                 activation: type[nn.Module] = nn.ReLU,
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

    def forward(self, queries: torch.Tensor, keys: torch.Tensor,
                query_pe: torch.Tensor,
                key_pe: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
    """``transformer.py`` ``TwoWayTransformer``; built with depth 2 / dim 256 /
    8 heads / mlp 2048 (``build_sam.py``)."""

    def __init__(self, depth: int, embedding_dim: int, num_heads: int,
                 mlp_dim: int, activation: type[nn.Module] = nn.ReLU,
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

    def forward(self, image_embedding: torch.Tensor, image_pe: torch.Tensor,
                point_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


# --------------------------------------------------------------------------- #
# ported SAM modules  (mask_decoder.py / prompt_encoder.py)
# --------------------------------------------------------------------------- #
class MaskDecoder(nn.Module):
    """``mask_decoder.py`` ``MaskDecoder``, one deviation.

    ``iou_head=False`` (the pre-registered ablation ⑤,
    ``--no-samdec-iou-head``) does not build ``iou_token`` /
    ``iou_prediction_head`` at all: the token stack starts at the mask tokens and
    ``iou_pred`` comes back as an all-zero, gradient-free tensor so the caller's
    shapes do not change.  Upstream always builds both; this branch is the
    ablation and is recorded as such in ``SAMDecHead.facts()``.
    """

    def __init__(self, *, transformer_dim: int, transformer: nn.Module,
                 num_multimask_outputs: int = 3,
                 activation: type[nn.Module] = nn.GELU,
                 iou_head_depth: int = 3, iou_head_hidden_dim: int = 256,
                 iou_head: bool = True) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs
        self.use_iou_head = bool(iou_head)

        self.iou_token = nn.Embedding(1, transformer_dim) if iou_head else None
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
        self.output_hypernetworks_mlps = nn.ModuleList([
            MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
            for _ in range(self.num_mask_tokens)])
        self.iou_prediction_head = (
            MLP(transformer_dim, iou_head_hidden_dim, self.num_mask_tokens,
                iou_head_depth) if iou_head else None)

    def forward(self, image_embeddings: torch.Tensor, image_pe: torch.Tensor,
                sparse_prompt_embeddings: torch.Tensor,
                dense_prompt_embeddings: torch.Tensor,
                multimask_output: bool) -> tuple[torch.Tensor, torch.Tensor]:
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings, image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings)
        mask_slice = slice(1, None) if multimask_output else slice(0, 1)
        return masks[:, mask_slice, :, :], iou_pred[:, mask_slice]

    def predict_masks(self, image_embeddings: torch.Tensor,
                      image_pe: torch.Tensor,
                      sparse_prompt_embeddings: torch.Tensor,
                      dense_prompt_embeddings: torch.Tensor,
                      ) -> tuple[torch.Tensor, torch.Tensor]:
        parts = ([self.iou_token.weight] if self.use_iou_head else [])
        parts.append(self.mask_tokens.weight)
        output_tokens = torch.cat(parts, dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        hs, src = self.transformer(src, pos_src, tokens)
        first = 1 if self.use_iou_head else 0
        iou_token_out = hs[:, 0, :] if self.use_iou_head else None
        mask_tokens_out = hs[:, first:(first + self.num_mask_tokens), :]

        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: list[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(
            b, self.num_mask_tokens, h, w)

        if self.use_iou_head:
            iou_pred = self.iou_prediction_head(iou_token_out)
        else:
            iou_pred = masks.new_zeros((b, self.num_mask_tokens))
        return masks, iou_pred


class PositionEmbeddingRandom(nn.Module):
    """``prompt_encoder.py`` ``PositionEmbeddingRandom``: a NON-trainable buffer.

    Pre-registration note (proposal §3.3 "初始化"): this is a **coordinate
    basis**, which DELTA S5.6 forbids.  It is carried as a *declared* violation,
    exactly as EPR-011's Fourier arm was, and its execution line is the
    ``corr_center_minus_corr_gt`` column every board already computes
    (``q3vl/whereb/amort/evaluate.py:465-467``): if the centre-prior pathology
    comes back positive there, S5.6 stands and this structure is dead.
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
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device,
                          dtype=self.positional_encoding_gaussian_matrix.dtype)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)


# --------------------------------------------------------------------------- #
# the head
# --------------------------------------------------------------------------- #
class SAMDecHead(nn.Module):
    """SAM's neck + prompt + decoder over the frozen ``F_pre``, from scratch.

    ``forward`` returns ``(logits, iou_pred)`` for ONE sample: ``logits`` is
    ``(n_candidates, 4*gh, 4*gw)`` -- the decoder's native 4x output, which is
    the point of the arm -- and ``iou_pred`` is ``(n_candidates,)``.

    Initialisation is PyTorch's default everywhere, because SAM's own decoder
    carries no explicit ``init`` call.  It is drawn inside a forked RNG so that
    (a) the head is reproducible from ``seed`` regardless of how much RNG the
    surrounding run consumed before it, and (b) constructing it does not shift
    the global stream that the trainer's sample order and the negative controls
    draw from.  Neither property changes any distribution; both make the arm's
    board reproducible.
    """

    def __init__(self, *, in_dim: int = 1024, text_dim: int = 2560,
                 transformer_dim: int = 256, multimask: int = 3,
                 iou_head: bool = True, prompt: str = "one",
                 seed: int = 20260810) -> None:
        super().__init__()
        if int(multimask) not in (1, 3):
            raise ValueError(
                f"--samdec-multimask must be 3 (multimask_output=True, "
                f"mask_decoder.py slice(1, None)) or 1 (slice(0, 1)); got {multimask}")
        if prompt not in ("one", "span"):
            raise ValueError(
                f"--samdec-prompt must be 'one' (h_cond -> 1 sparse token) or "
                f"'span' (the <where> span's T tokens, ablation ⑩); got {prompt!r}")
        self.in_dim = int(in_dim)
        self.text_dim = int(text_dim)
        self.transformer_dim = int(transformer_dim)
        self.multimask = int(multimask)
        self.prompt = str(prompt)
        self.seed = int(seed)
        self.use_iou_head = bool(iou_head)
        #: counted, never assumed: how many sparse prompt tokens the head was
        #: actually handed (SAM's own prompts are "rarely greater than 20").
        self.prompt_token_hist: dict[int, int] = {}

        devices = (list(range(torch.cuda.device_count()))
                   if torch.cuda.is_available() else [])
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.seed)
            c = self.transformer_dim
            # image_encoder.py:88-104 -- SAM's own neck, input width copied from
            # build_sam_vit_l's encoder_embed_dim = 1024 = F_pre's width.
            self.neck = nn.Sequential(
                nn.Conv2d(self.in_dim, c, kernel_size=1, bias=False),
                LayerNorm2d(c),
                nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
                LayerNorm2d(c),
            )
            # NOVEL (proposal NOTES 6): SAM's sparse prompt is "one 256-d token";
            # a single linear map is the minimal faithful analogue of that.
            self.prompt_proj = nn.Linear(self.text_dim, c)
            self.pe_layer = PositionEmbeddingRandom(c // 2)
            self.no_mask_embed = nn.Embedding(1, c)
            self.decoder = MaskDecoder(
                transformer_dim=c,
                transformer=TwoWayTransformer(depth=2, embedding_dim=c,
                                              num_heads=8, mlp_dim=2048),
                num_multimask_outputs=3, activation=nn.GELU,
                iou_head_depth=3, iou_head_hidden_dim=256,
                iou_head=self.use_iou_head)

    # -- properties --------------------------------------------------------
    @property
    def n_candidates(self) -> int:
        return 3 if self.multimask == 3 else 1

    # -- forward -----------------------------------------------------------
    def forward(self, feat: torch.Tensor, h_cond: torch.Tensor,
                grid_h: int, grid_w: int) -> tuple[torch.Tensor, torch.Tensor]:
        if feat.dim() != 4 or feat.shape[0] != 1:
            raise ValueError(f"expected (1, {self.in_dim}, gh, gw), got "
                             f"{tuple(feat.shape)}")
        if feat.shape[1] != self.in_dim:
            raise ValueError(f"feat has {feat.shape[1]} channels, head was built "
                             f"for {self.in_dim}")
        if (grid_h, grid_w) != tuple(feat.shape[-2:]):
            raise ValueError(f"grid {(grid_h, grid_w)} disagrees with feat "
                             f"{tuple(feat.shape[-2:])}")
        if h_cond.dim() != 2 or h_cond.shape[-1] != self.text_dim:
            raise ValueError(f"expected (T, {self.text_dim}) condition rows, got "
                             f"{tuple(h_cond.shape)}")
        w = self.prompt_proj.weight
        img = self.neck(feat.to(w.dtype))
        pe = self.pe_layer((grid_h, grid_w)).unsqueeze(0).to(img.dtype)
        spr = self.prompt_proj(h_cond.to(w.dtype)).unsqueeze(0)   # (1, T, 256)
        n_tok = int(spr.shape[1])
        self.prompt_token_hist[n_tok] = self.prompt_token_hist.get(n_tok, 0) + 1
        dns = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            1, -1, grid_h, grid_w)
        logits, iou = self.decoder(
            image_embeddings=img, image_pe=pe, sparse_prompt_embeddings=spr,
            dense_prompt_embeddings=dns.to(img.dtype),
            multimask_output=(self.multimask == 3))
        logits, iou = logits[0], iou[0]
        if logits.shape != (self.n_candidates, 4 * grid_h, 4 * grid_w):
            raise AssertionError(
                f"decoder returned {tuple(logits.shape)}; the ported "
                f"output_upscaling is 4x, so {self.n_candidates} x "
                f"{(4 * grid_h, 4 * grid_w)} was expected")
        return logits, iou

    # -- record ------------------------------------------------------------
    def facts(self) -> dict[str, Any]:
        groups: dict[str, int] = {}
        for name, p in self.named_parameters():
            groups[name.split(".")[0]] = groups.get(name.split(".")[0], 0) + p.numel()
        return {
            "arm": ARM,
            "port": "segment-anything mask_decoder + transformer + neck",
            "pretrained_weights_loaded": False,
            "in_dim": self.in_dim, "text_dim": self.text_dim,
            "transformer_dim": self.transformer_dim,
            "transformer": {"depth": 2, "num_heads": 8, "mlp_dim": 2048,
                            "attention_downsample_rate": 2},
            "num_mask_tokens": self.decoder.num_mask_tokens,
            "multimask_output": self.multimask == 3,
            "n_candidates": self.n_candidates,
            "iou_head": self.use_iou_head,
            "prompt": self.prompt,
            "upscale_factor": 4,
            "seed": self.seed,
            "n_params": sum(p.numel() for p in self.parameters()),
            "n_trainable": sum(p.numel() for p in self.parameters()
                               if p.requires_grad),
            "params_by_group": groups,
            "pe_is_buffer": not any(
                p is self.pe_layer.positional_encoding_gaussian_matrix
                for p in self.parameters()),
            "prompt_token_hist": dict(sorted(self.prompt_token_hist.items())),
            "s5_6_declared_violation": "PositionEmbeddingRandom (coordinate basis)",
            "sources": list(SAM_SOURCES),
        }


# --------------------------------------------------------------------------- #
# the loss  (SAM §A + sam2/training/loss_fns.py)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SamLossConfig:
    """The loss constants, all of them SAM's own.  See :data:`VARIANT`."""

    kind: str = "focal_dice"        # focal_dice | focal | bce_dice | bce
    alpha: float = 0.25
    gamma: float = 2.0
    w_focal: float = 20.0
    w_dice: float = 1.0
    w_iou: float = 1.0
    iou_head: bool = True

    def __post_init__(self) -> None:
        if self.kind not in ("focal_dice", "focal", "bce_dice", "bce"):
            raise ValueError(
                f"--samdec-loss must be focal_dice|focal|bce_dice|bce, got {self.kind!r}")

    @property
    def use_focal(self) -> bool:
        return self.kind in ("focal_dice", "focal")

    @property
    def use_dice(self) -> bool:
        return self.kind in ("focal_dice", "bce_dice")

    def to_dict(self) -> dict[str, Any]:
        return {"loss": self.kind, "focal_alpha": self.alpha,
                "focal_gamma": self.gamma, "w_focal": self.w_focal,
                "w_dice": self.w_dice if self.use_dice else 0.0,
                "w_iou": self.w_iou if self.iou_head else 0.0,
                "iou_head": self.iou_head,
                "iou_gt_threshold": IOU_GT_THRESHOLD}


def _mask_terms(logits: torch.Tensor, target: torch.Tensor,
                cfg: SamLossConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """``(mask_k, dice_k)`` per candidate; both ``(K,)``.

    ``mask_k`` is ``sigmoid_focal_loss`` (``sam2 training/loss_fns.py:52-90``:
    ``alpha=0.25``, ``gamma=2``, spatial ``mean``) or, for the
    ``--samdec-loss bce*`` ablation rows, the same spatial mean of plain
    ``BCEWithLogits``.  ``dice_k`` is ``loss_fns.py:20-49`` with the ``+1``
    smoothing that makes the empty-GT (``is_fake``) case defined.
    """
    z = logits.flatten(1)                       # (K, N)
    t = target.reshape(1, -1).expand_as(z)      # soft alpha in [0, 1]
    ce = F.binary_cross_entropy_with_logits(z, t, reduction="none")
    if cfg.use_focal:
        p = torch.sigmoid(z)
        p_t = p * t + (1 - p) * (1 - t)
        loss = ce * ((1 - p_t) ** cfg.gamma)
        if cfg.alpha >= 0:
            alpha_t = cfg.alpha * t + (1 - cfg.alpha) * (1 - t)
            loss = alpha_t * loss
        mask_k = loss.mean(-1)
    else:
        mask_k = ce.mean(-1)
    if cfg.use_dice:
        p = torch.sigmoid(z)
        numerator = 2 * (p * t).sum(-1)
        denominator = p.sum(-1) + t.sum(-1)
        dice_k = 1 - (numerator + 1) / (denominator + 1)
    else:
        dice_k = torch.zeros_like(mask_k)
    return mask_k, dice_k


def sam_mask_loss(logits: torch.Tensor, iou_pred: torch.Tensor | None,
                  target: torch.Tensor, cfg: SamLossConfig | None = None
                  ) -> dict[str, Any]:
    """SAM's full mask criterion for ONE sample.  §3.1 of the proposal, verbatim.

    ``logits`` ``(K, H, W)``; ``target`` ``(H, W)`` soft alpha in [0, 1];
    ``iou_pred`` ``(K,)`` or None.  Returns the terms, the winner index and the
    no-grad diagnostics::

        k*    = argmin_k [ w_focal * focal_k + w_dice * dice_k ]
        L     = w_focal * focal_k* + w_dice * dice_k* + w_iou * (iou_pred[k*] - IoU_k*)^2

    with ``IoU_k*`` the **constant** IoU of the thresholded prediction against
    ``target > 0.5``, denominator clamped at 1 (so an empty GT is defined and its
    target is 0).  Only the winner's terms carry gradient -- SAM §A: "only
    backpropagate from the lowest loss" -- and the IoU term is taken at the same
    index (``loss_fns.py:277-282``, "to be consistent w/ SAM").
    """
    cfg = cfg or SamLossConfig()
    if logits.dim() != 3:
        raise ValueError(f"expected (K, H, W) logits, got {tuple(logits.shape)}")
    if target.shape != logits.shape[-2:]:
        raise ValueError(
            f"target is {tuple(target.shape)} but the decoder produced "
            f"{tuple(logits.shape[-2:])}; the pixel GT must be projected to the "
            "decoder's own 4x grid (pixgt_size = (4*gh, 4*gw))")
    z = logits.float()
    t = target.float().clamp(0.0, 1.0)

    mask_k, dice_k = _mask_terms(z, t, cfg)
    combo = cfg.w_focal * mask_k + (cfg.w_dice * dice_k if cfg.use_dice else 0.0)
    k = int(torch.argmin(combo.detach()))

    with torch.no_grad():
        pred = (z.flatten(1) > 0)
        gt = (t.reshape(1, -1) > IOU_GT_THRESHOLD).expand_as(pred)
        inter = (pred & gt).sum(-1).float()
        union = (pred | gt).sum(-1).float()
        ious = inter / torch.clamp(union, min=1.0)

    total = cfg.w_focal * mask_k[k]
    if cfg.use_dice:
        total = total + cfg.w_dice * dice_k[k]
    if cfg.iou_head and iou_pred is not None:
        l_iou = F.mse_loss(iou_pred.float()[k], ious[k])
        total = total + cfg.w_iou * l_iou
    else:
        l_iou = z.new_zeros(())
    return {
        "total": total,
        "focal": mask_k[k].detach(),
        "dice": dice_k[k].detach(),
        "iouhead": l_iou.detach(),
        "sel": k,
        "mask_per_candidate": mask_k.detach(),
        "dice_per_candidate": dice_k.detach(),
        "iou_true": ious,
        "sup_cells": int(z.shape[-1] * z.shape[-2]),
    }


def assert_steps_row(row: Any) -> dict[str, Any]:
    """The pre-registered training witness: the FIRST ``steps.jsonl`` row carries
    ``L_focal`` / ``L_dice`` / ``L_iouhead`` and the ``sup_cells`` count.

    ``losses.aggregate`` prefixes ``AmortLoss.terms`` keys with ``L_``, so this
    arm's cell-count witness lands as ``L_sup_cells``; both spellings are
    accepted, one of them must be there.  Called by ``run_samdec_arm.py`` after
    the run and by :func:`compute_loss` on every micro-batch -- "defined but not
    wired" has cost this campaign three times.
    """
    r = dict(row or {})
    missing = [c for c in STEP_WITNESS_COLUMNS if c not in r]
    if not any(c in r for c in SUP_CELLS_COLUMNS):
        missing.append("/".join(SUP_CELLS_COLUMNS))
    if missing:
        raise AssertionError(
            f"arm {ARM} pre-registers the training witness columns "
            f"{list(STEP_WITNESS_COLUMNS) + [SUP_CELLS_COLUMNS[0]]} and the row "
            f"is missing {missing}; columns present: {sorted(r)}")
    sup = next(float(r[c]) for c in SUP_CELLS_COLUMNS if c in r)
    if sup <= 0:
        raise AssertionError(
            f"arm {ARM}: sup_cells = {sup}; the 4x supervision grid cannot be empty")
    return {"L_focal": r["L_focal"], "L_dice": r["L_dice"],
            "L_iouhead": r["L_iouhead"], "sup_cells": sup}


# --------------------------------------------------------------------------- #
# flags / configuration
# --------------------------------------------------------------------------- #
def add_arguments(ap) -> None:
    """Register the ``--samdec-*`` flags.  Single source of truth: the wrapper
    ``run_samdec_arm.py`` calls exactly this, so the flag surface and the
    defaults cannot drift from :data:`VARIANT`."""
    ap.add_argument("--samdec-multimask", type=int, default=VARIANT["multimask"],
                    choices=[3, 1],
                    help="3 = multimask_output=True (mask_decoder.py slice(1,None), "
                         "the SAM default); 1 = slice(0,1), ablation ③")
    ap.add_argument("--samdec-loss", default=VARIANT["loss"],
                    choices=["focal_dice", "focal", "bce_dice", "bce"],
                    help="focal_dice = SAM's 20:1 recipe; focal = drop dice "
                         "(ablation ①); bce_dice = focal -> BCE (ablation ②)")
    ap.add_argument("--samdec-gt", default=VARIANT["gt"], choices=["png", "raster"],
                    help="png = .cgt.png area-downsampled to (4gh,4gw) for all "
                         "four families; raster = analytic re-render for the "
                         "three geometric families (ablation ④)")
    ap.add_argument("--samdec-iou-head", dest="samdec_iou_head",
                    action="store_true", default=VARIANT["iou_head"],
                    help="SAM's IoU token + prediction head (default ON)")
    ap.add_argument("--no-samdec-iou-head", dest="samdec_iou_head",
                    action="store_false",
                    help="ablation ⑤: no iou_token / iou_prediction_head, no "
                         "L_iou, inference takes candidate 0 instead of argmax")
    ap.add_argument("--samdec-prompt", default=VARIANT["prompt"],
                    choices=["one", "span"],
                    help="one = h_cond -> 1 sparse token (SAM's point-prompt "
                         "analogue); span = the T <where>-span rows each through "
                         "the same prompt_proj (ablation ⑩, needs "
                         "--cond-readout where_span_pool)")
    ap.add_argument("--samdec-lr", type=float, default=VARIANT["lr"],
                    help="SAM §A: 8e-4 after warmup (NOT batch-rescaled; NOTES 1)")
    ap.add_argument("--samdec-wd", type=float, default=VARIANT["wd"],
                    help="SAM §A: weight decay 0.1")
    ap.add_argument("--samdec-sched", default=VARIANT["sched"],
                    choices=["sam_step", "cosine"],
                    help="sam_step = SAM §A's step-wise decay carried over "
                         "proportionally (warmup 250/90k, x0.1 at 60k and 86666)")
    ap.add_argument("--samdec-focal-alpha", type=float,
                    default=VARIANT["focal_alpha"])
    ap.add_argument("--samdec-focal-gamma", type=float,
                    default=VARIANT["focal_gamma"])
    ap.add_argument("--samdec-w-focal", type=float, default=VARIANT["w_focal"])
    ap.add_argument("--samdec-w-dice", type=float, default=VARIANT["w_dice"])
    ap.add_argument("--samdec-w-iou", type=float, default=VARIANT["w_iou"])


#: ``--samdec-*`` dest -> :data:`VARIANT` key
_FLAG_KEYS: dict[str, str] = {
    "samdec_multimask": "multimask", "samdec_loss": "loss", "samdec_gt": "gt",
    "samdec_iou_head": "iou_head", "samdec_prompt": "prompt",
    "samdec_lr": "lr", "samdec_wd": "wd", "samdec_sched": "sched",
    "samdec_focal_alpha": "focal_alpha", "samdec_focal_gamma": "focal_gamma",
    "samdec_w_focal": "w_focal", "samdec_w_dice": "w_dice",
    "samdec_w_iou": "w_iou",
}


def variant_from_args(args) -> dict[str, Any]:
    """The ``--samdec-*`` values present on ``args``, as :data:`VARIANT` keys."""
    out: dict[str, Any] = {}
    for dest, key in _FLAG_KEYS.items():
        if hasattr(args, dest):
            out[key] = getattr(args, dest)
    return out


def set_variant(**kw: Any) -> dict[str, Any]:
    """Wrapper seam (mirrors ``uniq4.VARIANT4``): the entry script fills this in
    before delegating to ``run_amort_arm.main``."""
    unknown = sorted(set(kw) - set(VARIANT))
    if unknown:
        raise KeyError(f"unknown SAMDEC variant key(s) {unknown}; known: "
                       f"{sorted(VARIANT)}")
    VARIANT.update(kw)
    return dict(VARIANT)


def config_of(args=None) -> dict[str, Any]:
    """:data:`VARIANT`, overridden by any ``--samdec-*`` value carried on ``args``.

    ``args`` is whatever the caller has: the entry script's own namespace
    (``run_amort_arm``'s, which does not carry the ``--samdec-*`` flags) or the
    wrapper's.  ``seed`` follows the run's ``--seed`` when there is one, so the
    head's initialisation is tied to the run record rather than to a constant
    that a re-run could silently diverge from.
    """
    cfg = dict(VARIANT)
    if args is not None:
        cfg.update(variant_from_args(args))
        if getattr(args, "seed", None) is not None:
            cfg["seed"] = int(args.seed)
    return cfg


def loss_config(args=None) -> SamLossConfig:
    c = config_of(args)
    return SamLossConfig(kind=str(c["loss"]), alpha=float(c["focal_alpha"]),
                         gamma=float(c["focal_gamma"]),
                         w_focal=float(c["w_focal"]), w_dice=float(c["w_dice"]),
                         w_iou=float(c["w_iou"]), iou_head=bool(c["iou_head"]))


def loss_preregistration(args=None) -> dict[str, Any]:
    """``arms.RUN_REQUIRED_HOOKS``: what ``loss_preregistration.json`` says.

    ``run_amort_arm``'s own record is the live arms' seven-term ST_LANG stack
    with ``dice_as_target: false``.  Both halves are false here: this arm never
    enters that stack (``trainer.py:244-251``) and SAM's criterion carries
    ``w_dice = 1.0`` on the winner.
    """
    c = loss_config(args)
    d = c.to_dict()
    terms = [f"{c.w_focal} * {'focal' if c.use_focal else 'bce'}"]
    if c.use_dice:
        terms.append(f"{c.w_dice} * dice")
    form = " + ".join(terms) + "   (winner k* only)"
    if c.iou_head:
        form += f" + {c.w_iou} * MSE(iou_pred[k*], IoU_k*)"
    return {
        "arm": ARM,
        "form": f"L = {form}",
        "winner": "k* = argmin_k [w_focal*focal_k + w_dice*dice_k]; SAM §A "
                  "'only backpropagate from the lowest loss'",
        "focal": "sigmoid_focal_loss(alpha=0.25, gamma=2, spatial mean)"
                 "   [sam2 training/loss_fns.py:52-90]",
        "dice": "1 - (2*inter + 1)/(sum(p) + sum(t) + 1)"
                "   [sam2 training/loss_fns.py:20-49]",
        "iou_head": ("MSE against the CONSTANT (no_grad) IoU of the "
                     "thresholded candidate vs GT > "
                     f"{IOU_GT_THRESHOLD}; SAM's IoU-prediction head, not an "
                     "objective on the field") if c.iou_head else False,
        "weights": d,
        "seven_term_stack": "not entered (trainer.py:244-251 routes new arms "
                            "to this module's compute_loss)",
        # the two IoU facts are split for the same reason run_amort_arm splits
        # them: a single flag would be a false record the moment the IoU head
        # is on
        "iou_as_field_target": False,
        "iou_as_selection_target": bool(c.iou_head),
        "dice_as_target": bool(c.use_dice),
        "dice_red_line_waiver": ("the campaign's 'dice is never a mask target' "
                                 "line (losses.py:9-11) is waived for the "
                                 "EPR-018..023 batch by the user's 2026-08-14 "
                                 "instruction; SAM's 1.0 is kept verbatim")
        if c.use_dice else None,
    }


# --------------------------------------------------------------------------- #
# the arm hooks  (q3vl/whereb/amort/arms.py)
# --------------------------------------------------------------------------- #
def build_head(*, in_dim: int, text_dim: int, args=None, **kw) -> nn.Module:
    c = config_of(args)
    params = dict(in_dim=in_dim, text_dim=text_dim,
                  multimask=int(c["multimask"]), iou_head=bool(c["iou_head"]),
                  prompt=str(c["prompt"]), seed=int(c["seed"]))
    params.update(kw)
    return SAMDecHead(**params)


def head_kwargs_from_args(args) -> dict[str, Any]:
    c = config_of(args)
    return {"multimask": int(c["multimask"]), "iou_head": bool(c["iou_head"]),
            "prompt": str(c["prompt"]), "seed": int(c["seed"])}


def readout_spec(args):
    """Pass ``--cond-readout`` through, and refuse the one incoherent pairing.

    ``--samdec-prompt span`` projects the rows of the reply span the encoder was
    fed.  That span is the ``<where>`` span only under
    ``--cond-readout where_span_pool`` (``q3vl/whereb/readout.py:305-311``);
    under ``seg_where`` it also contains the colour span and the seg token, so
    the "T tokens of <where>" ablation would silently be projecting something
    else.
    """
    from q3vl.whereb.readout import ReadoutSpec

    spec = ReadoutSpec(kind=getattr(args, "cond_readout", "seg_where"),
                       qtok=int(getattr(args, "readout_qtok", 0) or 0),
                       nseg=int(getattr(args, "readout_nseg", 1) or 1))
    if str(config_of(args)["prompt"]) == "span" and spec.kind != "where_span_pool":
        raise SystemExit(
            "--samdec-prompt span (ablation ⑩) reads the rows of the <where> "
            f"span, but --cond-readout is {spec.kind!r}, whose reply span also "
            "carries the <color> span / seg tokens.  Pass "
            "--cond-readout where_span_pool with it (proposal §4 row ⑩).")
    return spec


def builder_kwargs(args) -> dict[str, Any]:
    """The pixel GT is produced at the decoder's own resolution, ``(4gh, 4gw)``.

    Same ``area_resize`` as ``gt_low`` (``q3vl/whereb/amort/data.py:685-686``,
    ``q3vl/where/upsample.py:54-62``), so the supervision grid and the criterion
    grid are two projections of one raster rather than two different rulers.
    """
    return {"pixgt_size": (lambda gh, gw: (4 * int(gh), 4 * int(gw)))}


def optimizer_spec(args):
    """SAM §A: AdamW, betas (0.9, 0.999), lr 8e-4 after warmup, wd 0.1.

    ``betas=None`` -> torch's default ``(0.9, 0.999)``, i.e. the paper's values,
    passed by *not* passing them (``trainer.build_optimizer``).  The decay
    grouping stays this repo's ``dim > 1`` split: SAM's training code was never
    released, so whether it exempted norms/biases cannot be checked, and
    inventing an exemption would be a number with no source (NOTES 3).
    """
    from .trainer import OptimizerSpec

    c = config_of(args)
    return OptimizerSpec(type="adamw", lr=float(c["lr"]),
                         weight_decay=float(c["wd"]), betas=None,
                         grouping="dim")


def scheduler_kwargs(args, total_steps: int) -> dict[str, Any]:
    """SAM §A's schedule, shape-preserved onto ``total_steps``.

    250/90000 warmup -> 3 steps at 1200; x0.1 at 60000/90000 and 86666/90000 ->
    steps 800 and 1156.  The kind itself is ``multistep``
    (``q3vl/where/calibrate.py:129-131``), whose body after the linear warmup is
    ``gamma ** #(milestones passed)`` -- exactly SAM's step-wise decay.  The
    entry script passes ``--scheduler multistep`` for ``--samdec-sched sam_step``.
    """
    from q3vl.where.calibrate import scale_milestones

    c = config_of(args)
    if str(c["sched"]) != "sam_step":
        return {}
    t = max(1, int(total_steps))
    warmup = max(1, int(round(t * SAM_WARMUP_ITERS / SAM_TOTAL_ITERS)))
    return {"warmup_steps": warmup,
            "milestones": scale_milestones(SAM_MILESTONE_FRACS, t),
            "gamma": SAM_GAMMA}


def _condition_rows(head: SAMDecHead, ctx) -> torch.Tensor:
    """``(T, 2560)``: the sparse-prompt source, per ``--samdec-prompt``.

    ``one`` -> the readout rows themselves (K = 1 for every single-position
    readout; K > 1 -- the ``qtok`` / ``nseg`` ablation rows -- each get their own
    sparse token through the same ``prompt_proj``, which is the NOVEL default
    §4 pre-registers).  ``span`` -> every row of the ``<where>`` span.
    """
    if head.prompt == "span":
        h = ctx.h_where
        if h is None:
            raise ValueError(
                f"arm {ARM} --samdec-prompt span needs the reply-span hiddens "
                "(ArmContext.h_where) and the builder supplied none")
        kind = (getattr(ctx.sample, "readout", None) or {}).get("kind")
        if kind not in (None, "where_span_pool"):
            raise AssertionError(
                f"--samdec-prompt span expects the <where> span as the reply "
                f"(readout where_span_pool); this sample's plan is {kind!r}")
        rows = h[0] if h.dim() == 3 else h
        if rows.shape[0] == 0:
            raise ValueError(f"arm {ARM}: empty <where> span, no sparse prompt")
        return rows
    return ctx.require_cond(ARM)


def forward(model, head: SAMDecHead, ctx) -> dict[str, Any]:
    """``AmortModel.forward_geo`` -> the SAM decoder -> ``m_low`` on ``(gh, gw)``.

    Inference selection is SAM's own and GT-free: ``j* = argmax(iou_pred)``
    (candidate 0 when the IoU head is ablated away).  The selected candidate's
    ``sigmoid`` is projected back to the criterion grid with ``area_resize`` --
    the same operator and the same call shape ``gt_low`` uses -- so every column
    downstream of ``m_low`` is the frozen ruler, unchanged.
    """
    rows = _condition_rows(head, ctx)
    logits, iou = head(ctx.feat, rows, int(ctx.grid_h), int(ctx.grid_w))
    j = 0 if not head.use_iou_head else int(torch.argmax(iou.detach()))
    # float32 from here on: `m_low` is what every criterion column is computed
    # from, and "着色归着色、算数归算数" -- the arithmetic does not run in the
    # autocast dtype just because the decoder did.
    m_4x = torch.sigmoid(logits[j].float())
    m_low = area_resize(m_4x[None, None], (int(ctx.grid_h), int(ctx.grid_w)))[0, 0]
    return {
        "m_low": m_low,
        "m_4x": m_4x,
        "samdec": {"logits": logits, "iou_pred": iou, "sel": j,
                   "n_cand": int(logits.shape[0]),
                   "n_prompt_tokens": int(rows.shape[0])},
    }


def _target_of(x, shape: tuple[int, int]) -> torch.Tensor:
    """The pixel GT for one sample, at the decoder's grid.

    ``is_fake`` (foreign instruction, p = 0.15) is an **all-zero** target: the
    criterion is "this image contains nothing the instruction asks for".  Both
    SAM terms are defined there -- dice's ``+1`` smoothing and the IoU
    denominator's ``clamp(min=1)`` -- which is why the fake stream stays in the
    loss rather than being filtered out (proposal §3.1 last row / NOTES 4).
    """
    gt = getattr(x, "gt_pix", None)
    if gt is None:
        raise ValueError(
            f"arm {ARM} supervises on the pixel GT and this sample carries none. "
            "Build the run with a PixGTProvider (do NOT pass --no-pixgt); the "
            "grid is set by this arm's builder_kwargs -> pixgt_size (4gh, 4gw).")
    gt = gt.float()
    if tuple(gt.shape[-2:]) != tuple(shape):
        raise AssertionError(
            f"{getattr(x, 'sample_id', '?')}: pixel GT is {tuple(gt.shape[-2:])} "
            f"but the decoder produced {tuple(shape)}; pixgt_size must be "
            "(4*gh, 4*gw)")
    if bool(getattr(x, "is_fake", False)):
        return torch.zeros_like(gt)
    return gt


def compute_loss(model, out: dict[str, Any], x, weights):
    """SAM's criterion, and the runtime assertion that it produced its columns.

    The seven-term ST_LANG stack is not entered at all (``trainer.py:244-251``
    routes new arms here instead): SAM's recipe is ``20*focal + 1*dice`` on the
    lowest-loss candidate plus ``1.0 * MSE`` on the IoU head, and adding a term
    the reference does not have would make this a different recipe under the
    same name.
    """
    sd = out.get("samdec")
    if not sd:
        raise AssertionError(
            f"arm {ARM}: forward returned no 'samdec' block, so the WTA loss has "
            "no candidates to choose between")
    head = getattr(model, "geo", None)
    cfg = loss_config(None)
    if isinstance(head, SAMDecHead):
        cfg = SamLossConfig(kind=cfg.kind, alpha=cfg.alpha, gamma=cfg.gamma,
                            w_focal=cfg.w_focal, w_dice=cfg.w_dice,
                            w_iou=cfg.w_iou, iou_head=head.use_iou_head)
        if int(sd["n_cand"]) != head.n_candidates:
            raise AssertionError(
                f"arm {ARM}: head is configured for {head.n_candidates} "
                f"candidate(s) but this forward returned {sd['n_cand']}")
    logits = sd["logits"]
    target = _target_of(x, tuple(logits.shape[-2:]))
    res = sam_mask_loss(logits, sd.get("iou_pred"), target, cfg)

    from .losses import AmortLoss

    total = res["total"]
    terms = {
        "focal": res["focal"],
        "dice": res["dice"],
        "iouhead": res["iouhead"],
        # the witness column: `aggregate` publishes it as `L_sup_cells`, which is
        # how a board shows the supervision really ran on the 4x grid (16x the
        # cells the live arms use) rather than on (gh, gw).
        "sup_cells": total.detach().new_tensor(float(res["sup_cells"])),
    }
    stats = {
        "samdec_sel": float(res["sel"]),
        "samdec_n_cand": float(logits.shape[0]),
        "samdec_iou_true": float(res["iou_true"][res["sel"]]),
        "samdec_sup_cells": float(res["sup_cells"]),
    }
    if sd.get("iou_pred") is not None and head is not None and cfg.iou_head:
        stats["samdec_iou_pred"] = float(sd["iou_pred"][res["sel"]].detach())
    # runtime assertion, every micro-batch including the first: the four
    # pre-registered witness columns are produced by THIS call, not merely
    # declared in a docstring.
    assert_steps_row({f"L_{k}": float(v) for k, v in terms.items()})
    return AmortLoss(total=total, terms=terms, stats=stats)


def train_stats(out: dict[str, Any], x) -> dict[str, Any]:
    """Extra per-sample training columns (``trainer.py:272-274``)."""
    sd = out.get("samdec") or {}
    return {"samdec_sel": int(sd.get("sel", -1)),
            "samdec_n_cand": int(sd.get("n_cand", 0)),
            "samdec_n_prompt_tokens": int(sd.get("n_prompt_tokens", 0)),
            "samdec_sup_cells": int(
                getattr(x, "gt_pix", torch.empty(0)).numel()),
            "samdec_gt_pix_source": getattr(x, "gt_pix_source", "")}


# --------------------------------------------------------------------------- #
# evaluation columns
# --------------------------------------------------------------------------- #
def per_sample_row(model, out: dict[str, Any], x) -> dict[str, Any]:
    """The arm's diagnostics, on the SAME grid and the SAME top-k rule the
    headline uses (``metrics.gt_area_k`` / ``topk_mask`` / ``hard_iou``).

    ``samdec_best_of_3`` is the achievable ceiling of the three candidates and
    ``samdec_iou_mae`` is how far the IoU head's score is from the candidate's
    real matched-area top-k IoU -- i.e. how much the GT-free selection leaves on
    the table, and whether the head's ranking means anything.
    """
    sd = out.get("samdec")
    if not sd:
        return {}
    from q3vl.whereb.metrics import gt_area_k, hard_iou, topk_mask

    logits = sd["logits"].detach()
    gh, gw = int(x.grid_h), int(x.grid_w)
    gt = x.gt_low.detach().float()
    k = gt_area_k(gt)
    gt_k = topk_mask(gt, k)
    ious: list[float] = []
    for i in range(int(logits.shape[0])):
        m = area_resize(torch.sigmoid(logits[i].float())[None, None], (gh, gw))[0, 0]
        ious.append(hard_iou(topk_mask(m, k), gt_k))
    sel = int(sd["sel"])
    best = int(np.argmax(ious)) if ious else 0
    iou_pred = sd.get("iou_pred")
    pred_score = (float(iou_pred.detach()[sel]) if iou_pred is not None else None)
    return {
        "samdec_sel": sel,
        "samdec_n_cand": int(logits.shape[0]),
        "samdec_cand_ious": [float(v) for v in ious],
        "samdec_sel_iou": float(ious[sel]),
        "samdec_best_of_3": float(ious[best]),
        "samdec_best_cand": best,
        "samdec_sel_is_best": bool(best == sel),
        "samdec_iou_pred": pred_score,
        "samdec_iou_mae": (None if pred_score is None
                           else abs(pred_score - float(ious[sel]))),
        "samdec_n_prompt_tokens": int(sd.get("n_prompt_tokens", 0)),
    }


def criteria_columns(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """``board["criteria_columns"]["samdec_cand"]`` -- the pre-registered column.

    ``assert_criteria_ran`` (``q3vl/whereb/amort/evaluate.py:331-372``) refuses to
    publish a board whose ``n`` here is 0, so an arm that stopped emitting its
    candidate diagnostics cannot be read as if it had.
    """
    from .evaluate import _agg

    live = [r for r in rows if r.get("samdec_best_of_3") is not None]
    mae = [r["samdec_iou_mae"] for r in live if r.get("samdec_iou_mae") is not None]
    sel_hist: dict[str, int] = {}
    for r in live:
        key = str(int(r.get("samdec_sel", -1)))
        sel_hist[key] = sel_hist.get(key, 0) + 1
    block: dict[str, Any] = {
        **_agg(r["samdec_best_of_3"] for r in live),
        "selected_iou_median": (float(np.median([r["samdec_sel_iou"] for r in live]))
                                if live else None),
        "sel_is_best_frac": (float(np.mean([bool(r["samdec_sel_is_best"])
                                            for r in live])) if live else None),
        "iou_mae": _agg(mae),
        "sel_hist": dict(sorted(sel_hist.items())),
        "n_cand": (int(live[0].get("samdec_n_cand", 0)) if live else 0),
        "prompt_tokens": _agg(r.get("samdec_n_prompt_tokens") for r in live),
        "note": ("best-of-K matched-area top-k IoU of the 4x candidates after "
                 "area_resize to (gh, gw), against the deployed candidate "
                 "argmax(iou_pred); iou_mae is |iou_pred - real top-k IoU| of "
                 "the selected candidate (EPR-019 §3.3 ⑥)"),
    }
    return {"samdec_cand": block}
