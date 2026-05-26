from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .residual_lut import apply_lut_4d, identity_lut_4d


SemanticMode = Literal["normal", "zero", "shuffle"]
OutputMode = Literal["lut_only", "terminal_residual"]
AlphaSource = Literal["multiscale", "hidden"]


@dataclass
class VeraSALUTOutput:
    image: torch.Tensor
    lut_image: torch.Tensor
    lut: torch.Tensor
    alpha: torch.Tensor
    context: torch.Tensor
    semantic_tokens: torch.Tensor | None
    exec_tokens: torch.Tensor | None
    affinity_maps: tuple[torch.Tensor | None, ...]
    gate: torch.Tensor | None


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        act: bool = True,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.GroupNorm(_group_count(out_channels), out_channels),
        ]
        if act:
            layers.append(nn.SiLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _as_feature_map(
    x: torch.Tensor,
    *,
    expected_channels: int | None = None,
    spatial_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    if x.ndim == 4:
        if expected_channels is None:
            return x
        if x.shape[1] == expected_channels:
            return x
        if x.shape[-1] == expected_channels:
            return x.permute(0, 3, 1, 2).contiguous()
        raise ValueError(
            f"expected a feature map with {expected_channels} channels, got {tuple(x.shape)}"
        )

    if x.ndim != 3:
        raise ValueError(
            "vision_spatial must be channel-first map, channel-last map, or tokens; "
            f"got {tuple(x.shape)}"
        )

    batch, tokens, channels = x.shape
    if expected_channels is not None and channels != expected_channels:
        raise ValueError(f"expected token dim {expected_channels}, got {channels}")

    if spatial_size is None:
        side = int(math.sqrt(tokens))
        if side * side != tokens:
            raise ValueError(
                "token-form vision_spatial must have a square token count unless "
                "vision_spatial_size is provided"
            )
        height = width = side
    else:
        height, width = spatial_size
        if height * width != tokens:
            raise ValueError(
                f"vision_spatial_size {spatial_size} is incompatible with {tokens} tokens"
            )

    return x.transpose(1, 2).reshape(batch, channels, height, width).contiguous()


def _image_stack(i_in: torch.Tensor, i_base: torch.Tensor | None) -> torch.Tensor:
    if i_base is None:
        i_base = i_in
    if i_in.shape != i_base.shape:
        raise ValueError(f"i_in and i_base shapes must match, got {tuple(i_in.shape)} and {tuple(i_base.shape)}")
    if i_in.ndim != 4 or i_in.shape[1] != 3:
        raise ValueError(f"images must have shape (B, 3, H, W), got {tuple(i_in.shape)}")
    return torch.cat((i_in, i_base, i_base - i_in), dim=1)


def _apply_semantic_mode(tokens: torch.Tensor | None, mode: SemanticMode) -> torch.Tensor | None:
    if tokens is None:
        return None
    if mode == "normal":
        return tokens
    if mode == "zero":
        return torch.zeros_like(tokens)
    if mode == "shuffle":
        if tokens.shape[0] < 2:
            return tokens
        return tokens.roll(shifts=1, dims=0)
    raise ValueError(f"unsupported semantic mode: {mode}")


class RetouchSemanticBankFusion(nn.Module):
    """Fuse selected LLM-layer retouch token states into L/GC/SC semantic tokens."""

    def __init__(self, hidden_dim: int, out_dim: int, *, layer_count: int = 4, token_count: int = 3):
        super().__init__()
        if layer_count < 1:
            raise ValueError("layer_count must be >= 1")
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.layer_count = layer_count
        self.token_count = token_count
        self.project = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, out_dim))
        self.layer_logits = nn.Parameter(torch.zeros(token_count, layer_count))

    def forward(self, hidden_bank: torch.Tensor) -> torch.Tensor:
        if hidden_bank.ndim == 3:
            hidden_bank = hidden_bank.unsqueeze(1)
        if hidden_bank.ndim != 4:
            raise ValueError(
                "hidden_bank must be (B, T, D) or (B, L, T, D), "
                f"got {tuple(hidden_bank.shape)}"
            )

        _, layers, tokens, hidden_dim = hidden_bank.shape
        if tokens != self.token_count:
            raise ValueError(f"expected {self.token_count} retouch tokens, got {tokens}")
        if hidden_dim != self.hidden_dim:
            raise ValueError(f"expected hidden dim {self.hidden_dim}, got {hidden_dim}")

        projected = self.project(hidden_bank)
        if layers == 1:
            return projected[:, 0]
        if layers != self.layer_count:
            raise ValueError(f"expected {self.layer_count} hidden layers, got {layers}")

        weights = F.softmax(self.layer_logits, dim=1)
        return torch.einsum("bltc,tl->btc", projected, weights)


class ContentVisionPyramid(nn.Module):
    def __init__(
        self,
        *,
        image_channels: int = 9,
        feature_channels: int = 128,
        vision_channels: int | None,
        use_vision: bool,
    ):
        super().__init__()
        if use_vision and vision_channels is None:
            raise ValueError("vision_channels is required when use_vision=True")
        self.use_vision = use_vision
        self.vision_channels = vision_channels

        self.image_stem = nn.Sequential(
            ConvGNAct(image_channels, feature_channels // 2, stride=2),
            ConvGNAct(feature_channels // 2, feature_channels, stride=2),
        )
        self.image_down_8 = ConvGNAct(feature_channels, feature_channels, stride=2)
        self.image_down_16 = ConvGNAct(feature_channels, feature_channels, stride=2)

        if use_vision:
            self.vision_proj = ConvGNAct(vision_channels, feature_channels, kernel_size=1, padding=0)
        else:
            self.vision_proj = None

        fuse_in = feature_channels * (2 if use_vision else 1)
        self.fuse_blocks = nn.ModuleList(
            [ConvGNAct(fuse_in, feature_channels) for _ in range(3)]
        )

    def forward(
        self,
        image_stack: torch.Tensor,
        *,
        vision_spatial: torch.Tensor | None = None,
        vision_spatial_size: tuple[int, int] | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor] | None]:
        f4 = self.image_stem(image_stack)
        f8 = self.image_down_8(f4)
        f16 = self.image_down_16(f8)
        image_features = [f4, f8, f16]

        if not self.use_vision:
            return [
                block(feat)
                for block, feat in zip(self.fuse_blocks, image_features)
            ], None

        if vision_spatial is None:
            raise ValueError("vision_spatial is required when use_vision=True")
        assert self.vision_proj is not None
        vision_map = _as_feature_map(
            vision_spatial,
            expected_channels=self.vision_channels,
            spatial_size=vision_spatial_size,
        )
        vision_map = self.vision_proj(vision_map)
        vision_features = [
            F.interpolate(vision_map, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            for feat in image_features
        ]
        fused = [
            block(torch.cat((img_feat, vis_feat), dim=1))
            for block, img_feat, vis_feat in zip(self.fuse_blocks, image_features, vision_features)
        ]
        return fused, vision_features


class SemanticStylePyramid(nn.Module):
    def __init__(self, *, feature_channels: int = 128, token_count: int = 3, scale_count: int = 3):
        super().__init__()
        self.feature_channels = feature_channels
        self.token_count = token_count
        self.film = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(feature_channels),
                    nn.Linear(feature_channels, feature_channels * 2),
                )
                for _ in range(scale_count)
            ]
        )
        self.token_proj = nn.ModuleList(
            [nn.Linear(feature_channels, feature_channels, bias=False) for _ in range(scale_count)]
        )
        self.feature_proj = nn.ModuleList(
            [nn.Conv2d(feature_channels, feature_channels, kernel_size=1, bias=False) for _ in range(scale_count)]
        )
        self.affinity_fuse = nn.ModuleList(
            [ConvGNAct(token_count, feature_channels) for _ in range(scale_count)]
        )

    def forward(
        self,
        base_features: list[torch.Tensor],
        semantic_tokens: torch.Tensor,
        exec_tokens: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], tuple[torch.Tensor, ...]]:
        if semantic_tokens.ndim != 3 or semantic_tokens.shape[1] != self.token_count:
            raise ValueError(
                f"semantic_tokens must be (B, {self.token_count}, C), got {tuple(semantic_tokens.shape)}"
            )

        pooled = semantic_tokens.mean(dim=1)
        if exec_tokens is not None:
            pooled = pooled + exec_tokens.mean(dim=1)

        semantic_features: list[torch.Tensor] = []
        affinity_maps: list[torch.Tensor] = []
        for idx, feat in enumerate(base_features):
            scale, bias = self.film[idx](pooled).chunk(2, dim=-1)
            scale = scale[:, :, None, None]
            bias = bias[:, :, None, None]

            mean = feat.mean(dim=(2, 3), keepdim=True)
            std = feat.var(dim=(2, 3), unbiased=False, keepdim=True).add(1e-6).sqrt()
            styled = (feat - mean) / std
            styled = styled * (1.0 + scale) + bias

            token_proj = F.normalize(self.token_proj[idx](semantic_tokens), dim=-1)
            feature_proj = self.feature_proj[idx](styled)
            bsz, channels, height, width = feature_proj.shape
            feature_tokens = F.normalize(feature_proj.flatten(2).transpose(1, 2), dim=-1)
            affinity = torch.einsum("btc,bnc->btn", token_proj, feature_tokens)
            affinity = affinity.reshape(bsz, self.token_count, height, width)

            styled = styled + self.affinity_fuse[idx](affinity)
            semantic_features.append(styled)
            affinity_maps.append(affinity)

        return semantic_features, tuple(affinity_maps)


class SpatialCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        feature_channels: int = 128,
        attention_channels: int = 128,
        num_heads: int = 4,
        token_count: int = 3,
        max_attention_tokens: int = 256,
    ):
        super().__init__()
        if attention_channels % num_heads != 0:
            raise ValueError("attention_channels must be divisible by num_heads")
        self.max_attention_tokens = max_attention_tokens
        self.token_count = token_count
        self.q = nn.Conv2d(feature_channels, attention_channels, kernel_size=1)
        self.k = nn.Conv2d(feature_channels, attention_channels, kernel_size=1)
        self.v = nn.Conv2d(feature_channels, attention_channels, kernel_size=1)
        self.attention = nn.MultiheadAttention(attention_channels, num_heads, batch_first=True)
        self.attn_out = ConvGNAct(attention_channels, feature_channels, kernel_size=1, padding=0)
        self.gate = nn.Sequential(
            nn.Conv2d(feature_channels * 2 + token_count, feature_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=1),
        )
        self.fixed = ConvGNAct(feature_channels, feature_channels)
        self.refine = ConvGNAct(feature_channels, feature_channels)

    def forward(
        self,
        content: torch.Tensor,
        style: torch.Tensor,
        affinity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if content.shape[-2:] != style.shape[-2:]:
            style = F.interpolate(style, size=content.shape[-2:], mode="bilinear", align_corners=False)

        height, width = content.shape[-2:]
        attn_size = self._attention_size(height, width)
        q_in = F.interpolate(content, size=attn_size, mode="bilinear", align_corners=False)
        k_in = F.interpolate(style, size=attn_size, mode="bilinear", align_corners=False)

        q = self.q(q_in).flatten(2).transpose(1, 2)
        k = self.k(k_in).flatten(2).transpose(1, 2)
        v = self.v(k_in).flatten(2).transpose(1, 2)
        attended, _ = self.attention(q, k, v, need_weights=False)

        bsz = content.shape[0]
        attn_height, attn_width = attn_size
        attended = attended.transpose(1, 2).reshape(bsz, -1, attn_height, attn_width)
        attended = F.interpolate(attended, size=(height, width), mode="bilinear", align_corners=False)
        attended = self.attn_out(attended)

        if affinity is None:
            affinity = content.new_zeros(bsz, self.token_count, height, width)
        elif affinity.shape[-2:] != (height, width):
            affinity = F.interpolate(affinity, size=(height, width), mode="bilinear", align_corners=False)

        gate = torch.sigmoid(self.gate(torch.cat((content, attended, affinity), dim=1)))
        fused = content + self.fixed(content) * gate + attended
        return self.refine(fused)

    def _attention_size(self, height: int, width: int) -> tuple[int, int]:
        tokens = height * width
        if tokens <= self.max_attention_tokens:
            return height, width
        scale = math.sqrt(float(self.max_attention_tokens) / float(tokens))
        return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


class ContextMapHead(nn.Module):
    def __init__(self, *, feature_channels: int = 128, scale_count: int = 3):
        super().__init__()
        self.fuse = nn.Sequential(
            ConvGNAct(feature_channels * scale_count, feature_channels),
            ConvGNAct(feature_channels, feature_channels),
            nn.Conv2d(feature_channels, 1, kernel_size=1),
        )

    def forward(self, features: list[torch.Tensor], output_size: tuple[int, int]) -> torch.Tensor:
        target_size = features[0].shape[-2:]
        fused = torch.cat(
            [
                feat if feat.shape[-2:] == target_size else F.interpolate(
                    feat, size=target_size, mode="bilinear", align_corners=False
                )
                for feat in features
            ],
            dim=1,
        )
        context = torch.sigmoid(self.fuse(fused))
        return F.interpolate(context, size=output_size, mode="bilinear", align_corners=False)


class MultiScaleLUTWeightGenerator(nn.Module):
    def __init__(
        self,
        *,
        feature_channels: int = 128,
        basis_count: int = 64,
        scale_count: int = 3,
        pool_size: int = 3,
        use_exec_tokens: bool = True,
    ):
        super().__init__()
        self.pool_size = pool_size
        self.use_exec_tokens = use_exec_tokens
        self.refine = nn.ModuleList([ConvGNAct(feature_channels, feature_channels) for _ in range(scale_count)])
        input_dim = scale_count * feature_channels * pool_size * pool_size
        if use_exec_tokens:
            input_dim += feature_channels
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, feature_channels * 2),
            nn.SiLU(inplace=True),
            nn.Linear(feature_channels * 2, basis_count),
        )

    def forward(
        self,
        features: list[torch.Tensor],
        *,
        exec_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled = []
        for block, feat in zip(self.refine, features):
            item = F.adaptive_avg_pool2d(block(feat), (self.pool_size, self.pool_size))
            pooled.append(item.flatten(start_dim=1))

        flat = torch.cat(pooled, dim=1)
        if self.use_exec_tokens:
            if exec_tokens is None:
                flat = torch.cat((flat, flat.new_zeros(flat.shape[0], features[0].shape[1])), dim=1)
            else:
                flat = torch.cat((flat, exec_tokens.mean(dim=1)), dim=1)

        return F.softmax(self.classifier(flat), dim=1)


class VeraSALUT(nn.Module):
    """VeraRetouch feature-conditioned SA-LUT executor.

    The default forward path is the MVE main experiment from plan/model_pipeline:
    lookup is applied to I_in with a semantic context coordinate, producing a
    LUT-only paired retouch result. Set output_mode="terminal_residual" for the
    later VeraRetouch renderer residual integration experiment.
    """

    def __init__(
        self,
        *,
        vision_channels: int | None,
        hidden_dim: int | None,
        exec_dim: int | None = None,
        image_channels: int = 9,
        feature_channels: int = 128,
        grid_size: int = 17,
        context_size: int = 2,
        basis_count: int = 64,
        rho: float = 0.15,
        output_mode: OutputMode = "lut_only",
        alpha_source: AlphaSource = "multiscale",
        use_vision: bool = True,
        use_semantic_context: bool = True,
        use_semantic_alpha: bool = True,
        use_exec_alpha: bool = True,
        semantic_layers: int = 4,
        semantic_token_count: int = 3,
        attention_channels: int | None = None,
        attention_heads: int = 4,
        max_attention_tokens: int = 256,
        gamma_init: float = -4.0,
    ):
        super().__init__()
        if output_mode not in ("lut_only", "terminal_residual"):
            raise ValueError("output_mode must be 'lut_only' or 'terminal_residual'")
        if alpha_source not in ("multiscale", "hidden"):
            raise ValueError("alpha_source must be 'multiscale' or 'hidden'")
        if grid_size < 2:
            raise ValueError("grid_size must be >= 2")
        if context_size < 2:
            raise ValueError("context_size must be >= 2")
        requires_semantic = use_semantic_context or use_semantic_alpha or alpha_source == "hidden"
        if requires_semantic and hidden_dim is None:
            raise ValueError("hidden_dim is required when semantic conditioning is enabled")

        self.grid_size = grid_size
        self.context_size = context_size
        self.basis_count = basis_count
        self.rho = float(rho)
        self.output_mode = output_mode
        self.alpha_source = alpha_source
        self.use_vision = use_vision
        self.use_semantic_context = use_semantic_context
        self.use_semantic_alpha = use_semantic_alpha
        self.use_exec_alpha = use_exec_alpha and exec_dim is not None
        self.semantic_token_count = semantic_token_count
        self.feature_channels = feature_channels
        self.exec_dim = exec_dim

        self.content = ContentVisionPyramid(
            image_channels=image_channels,
            feature_channels=feature_channels,
            vision_channels=vision_channels,
            use_vision=use_vision,
        )

        self.semantic_bank = (
            RetouchSemanticBankFusion(
                hidden_dim,
                feature_channels,
                layer_count=semantic_layers,
                token_count=semantic_token_count,
            )
            if hidden_dim is not None
            else None
        )
        self.exec_project = (
            nn.Sequential(nn.LayerNorm(exec_dim), nn.Linear(exec_dim, feature_channels))
            if exec_dim is not None
            else None
        )
        self.semantic_style = SemanticStylePyramid(
            feature_channels=feature_channels,
            token_count=semantic_token_count,
        )

        if attention_channels is None:
            attention_channels = feature_channels
        self.cross_attention = nn.ModuleList(
            [
                SpatialCrossAttentionBlock(
                    feature_channels=feature_channels,
                    attention_channels=attention_channels,
                    num_heads=attention_heads,
                    token_count=semantic_token_count,
                    max_attention_tokens=max_attention_tokens,
                )
                for _ in range(3)
            ]
        )
        self.context_head = ContextMapHead(feature_channels=feature_channels)
        self.alpha_head = MultiScaleLUTWeightGenerator(
            feature_channels=feature_channels,
            basis_count=basis_count,
            use_exec_tokens=self.use_exec_alpha,
        )
        self.hidden_alpha_head = nn.Sequential(
            nn.LayerNorm(feature_channels * semantic_token_count),
            nn.Linear(feature_channels * semantic_token_count, feature_channels * 2),
            nn.SiLU(inplace=True),
            nn.Linear(feature_channels * 2, basis_count),
        )

        self.delta_basis = nn.Parameter(
            torch.empty(basis_count, grid_size, grid_size, grid_size, context_size, 3)
        )
        nn.init.normal_(self.delta_basis, mean=0.0, std=1e-4)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init))) if output_mode == "terminal_residual" else None

    @classmethod
    def from_ablation(cls, variant: Literal["E1", "E2", "E3", "E4"], **kwargs) -> "VeraSALUT":
        variant = variant.upper()  # type: ignore[assignment]
        if variant == "E1":
            kwargs.update(
                use_vision=False,
                use_semantic_context=False,
                use_semantic_alpha=False,
                use_exec_alpha=False,
            )
        elif variant == "E2":
            kwargs.update(
                use_vision=True,
                use_semantic_context=False,
                use_semantic_alpha=False,
                use_exec_alpha=False,
            )
        elif variant == "E3":
            kwargs.update(
                use_vision=True,
                use_semantic_context=True,
                use_semantic_alpha=False,
                use_exec_alpha=False,
            )
        elif variant == "E4":
            kwargs.update(use_vision=True, use_semantic_context=True, use_semantic_alpha=True)
        else:
            raise ValueError(f"unknown VeraSA-LUT ablation variant: {variant}")
        return cls(**kwargs)

    def forward(
        self,
        i_in: torch.Tensor,
        *,
        i_base: torch.Tensor | None = None,
        vision_spatial: torch.Tensor | None = None,
        hidden_bank: torch.Tensor | None = None,
        z_exec: torch.Tensor | None = None,
        vision_spatial_size: tuple[int, int] | None = None,
        semantic_mode: SemanticMode = "normal",
        return_aux: bool = False,
    ) -> torch.Tensor | VeraSALUTOutput:
        image_stack = _image_stack(i_in, i_base)
        content_features, vision_features = self.content(
            image_stack,
            vision_spatial=vision_spatial,
            vision_spatial_size=vision_spatial_size,
        )

        semantic_tokens = self._semantic_tokens(hidden_bank, semantic_mode)
        exec_tokens = self._exec_tokens(z_exec, semantic_mode)

        if self.use_semantic_context:
            if semantic_tokens is None:
                raise ValueError("hidden_bank is required when use_semantic_context=True")
            semantic_base = vision_features if vision_features is not None else content_features
            style_features, affinity_maps = self.semantic_style(
                semantic_base,
                semantic_tokens,
                exec_tokens=exec_tokens,
            )
        else:
            style_features = vision_features if vision_features is not None else content_features
            affinity_maps = tuple(None for _ in range(len(content_features)))

        attended_features = [
            block(content, style, affinity)
            for block, content, style, affinity in zip(
                self.cross_attention,
                content_features,
                style_features,
                affinity_maps,
            )
        ]
        context = self.context_head(attended_features, output_size=i_in.shape[-2:])

        alpha_features = attended_features if self.use_semantic_alpha else content_features
        alpha_exec_tokens = exec_tokens if self.use_semantic_alpha else None
        alpha = self._predict_alpha(
            alpha_features,
            semantic_tokens=semantic_tokens,
            exec_tokens=alpha_exec_tokens,
        )
        lut = self._mix_lut(alpha, device=i_in.device, dtype=i_in.dtype)

        lookup_image = i_in
        gate = None
        if self.output_mode == "terminal_residual":
            if i_base is None:
                raise ValueError("i_base is required for output_mode='terminal_residual'")
            lookup_image = i_base

        lut_image = apply_lut_4d(lut, lookup_image, context)
        if self.output_mode == "terminal_residual":
            assert self.gamma is not None
            gate = torch.sigmoid(self.gamma).to(dtype=i_in.dtype)
            image = (i_base + gate * (lut_image - i_base)).clamp(0.0, 1.0)
        else:
            image = lut_image.clamp(0.0, 1.0)

        if not return_aux:
            return image
        return VeraSALUTOutput(
            image=image,
            lut_image=lut_image,
            lut=lut,
            alpha=alpha,
            context=context,
            semantic_tokens=semantic_tokens,
            exec_tokens=exec_tokens,
            affinity_maps=affinity_maps,
            gate=gate,
        )

    def _semantic_tokens(self, hidden_bank: torch.Tensor | None, semantic_mode: SemanticMode) -> torch.Tensor | None:
        if hidden_bank is None:
            return None
        if self.semantic_bank is None:
            raise ValueError("hidden_bank was provided but this model has no semantic_bank")
        return _apply_semantic_mode(self.semantic_bank(hidden_bank), semantic_mode)

    def _exec_tokens(self, z_exec: torch.Tensor | None, semantic_mode: SemanticMode) -> torch.Tensor | None:
        if z_exec is None:
            return None
        if self.exec_project is None or self.exec_dim is None:
            raise ValueError("z_exec was provided but exec_dim was not configured")
        if z_exec.ndim == 2:
            if z_exec.shape[-1] == self.semantic_token_count * self.exec_dim:
                z_exec = z_exec.reshape(z_exec.shape[0], self.semantic_token_count, self.exec_dim)
            elif z_exec.shape[-1] == self.exec_dim:
                z_exec = z_exec[:, None, :].expand(-1, self.semantic_token_count, -1)
            else:
                raise ValueError(f"unexpected flattened z_exec shape {tuple(z_exec.shape)}")
        if z_exec.ndim != 3:
            raise ValueError(f"z_exec must be (B, T, D) or flattened, got {tuple(z_exec.shape)}")
        if z_exec.shape[1] != self.semantic_token_count or z_exec.shape[2] != self.exec_dim:
            raise ValueError(
                f"expected z_exec shape (B, {self.semantic_token_count}, {self.exec_dim}), "
                f"got {tuple(z_exec.shape)}"
            )
        return _apply_semantic_mode(self.exec_project(z_exec), semantic_mode)

    def _predict_alpha(
        self,
        features: list[torch.Tensor],
        *,
        semantic_tokens: torch.Tensor | None,
        exec_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.alpha_source == "hidden":
            if semantic_tokens is None:
                raise ValueError("hidden alpha source requires hidden_bank")
            logits = self.hidden_alpha_head(semantic_tokens.flatten(start_dim=1))
            return F.softmax(logits, dim=1)
        return self.alpha_head(features, exec_tokens=exec_tokens)

    def _mix_lut(self, alpha: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        basis = torch.tanh(self.delta_basis).to(device=device, dtype=dtype)
        alpha = alpha.to(device=device, dtype=dtype)
        delta = torch.einsum("bk,kijlmc->bijlmc", alpha, basis)
        identity = identity_lut_4d(
            self.grid_size,
            self.context_size,
            device=device,
            dtype=dtype,
        ).unsqueeze(0)
        return identity + self.rho * delta


def context_tv(context: torch.Tensor) -> torch.Tensor:
    if context.ndim != 4 or context.shape[1] != 1:
        raise ValueError(f"context must have shape (B, 1, H, W), got {tuple(context.shape)}")
    return context.diff(dim=2).abs().mean() + context.diff(dim=3).abs().mean()


def lut_monotonicity(lut: torch.Tensor) -> torch.Tensor:
    if lut.ndim != 6:
        raise ValueError(f"expected a 4D LUT with shape (B, G, G, G, C, 3), got {tuple(lut.shape)}")
    penalty = lut.new_tensor(0.0)
    for dim in (1, 2, 3):
        penalty = penalty + F.relu(-lut.diff(dim=dim)).mean()
    return penalty
