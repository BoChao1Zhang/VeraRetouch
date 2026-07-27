"""OneAlign aesthetic scorer retained for canonical candidate ranking."""
from __future__ import annotations

import os

import sys
import threading
from typing import Any, Optional

from PIL import Image, ImageOps

from . import config


def _decode_rgb(path: str) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def _prepare_qalign_transformers_compat(modeling_llama: Any) -> None:
    """Expose public Llama symbols expected by Q-Align's legacy wildcard import."""
    setattr(
        modeling_llama,
        "__all__",
        [name for name in dir(modeling_llama) if not name.startswith("_")],
    )


def _install_qalign_rope_compat(modeling_llama2: Any, torch: Any) -> None:
    """Restore the transformers 4.37 RoPE surface used by Q-Align."""

    class LegacyLlamaRotaryEmbedding(torch.nn.Module):
        def __init__(
            self, dim: int, max_position_embeddings: int = 2048,
            base: float = 10000, device: Any = None,
        ) -> None:
            super().__init__()
            self.dim = dim
            self.max_position_embeddings = max_position_embeddings
            self.base = base
            inv_freq = 1.0 / (
                self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self._set_cos_sin_cache(
                max_position_embeddings, self.inv_freq.device, torch.get_default_dtype()
            )

        def _set_cos_sin_cache(self, seq_len: int, device: Any, dtype: Any) -> None:
            self.max_seq_len_cached = seq_len
            positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            frequencies = torch.outer(positions, self.inv_freq)
            embedding = torch.cat((frequencies, frequencies), dim=-1)
            self.register_buffer(
                "cos_cached", embedding.cos().to(dtype), persistent=False
            )
            self.register_buffer(
                "sin_cached", embedding.sin().to(dtype), persistent=False
            )

        def forward(self, value: Any, seq_len: int | None = None) -> tuple[Any, Any]:
            length = int(seq_len if seq_len is not None else value.shape[-2])
            if length > self.max_seq_len_cached:
                self._set_cos_sin_cache(length, value.device, value.dtype)
            return (
                self.cos_cached[:length].to(dtype=value.dtype),
                self.sin_cached[:length].to(dtype=value.dtype),
            )

    class LegacyLlamaLinearScalingRotaryEmbedding(LegacyLlamaRotaryEmbedding):
        def __init__(self, *args: Any, scaling_factor: float = 1.0, **kwargs: Any) -> None:
            self.scaling_factor = scaling_factor
            super().__init__(*args, **kwargs)

        def _set_cos_sin_cache(self, seq_len: int, device: Any, dtype: Any) -> None:
            self.max_seq_len_cached = seq_len
            positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            positions = positions / self.scaling_factor
            frequencies = torch.outer(positions, self.inv_freq)
            embedding = torch.cat((frequencies, frequencies), dim=-1)
            self.register_buffer(
                "cos_cached", embedding.cos().to(dtype), persistent=False
            )
            self.register_buffer(
                "sin_cached", embedding.sin().to(dtype), persistent=False
            )

    class LegacyLlamaDynamicNTKScalingRotaryEmbedding(LegacyLlamaRotaryEmbedding):
        def __init__(self, *args: Any, scaling_factor: float = 1.0, **kwargs: Any) -> None:
            self.scaling_factor = scaling_factor
            super().__init__(*args, **kwargs)

        def _set_cos_sin_cache(self, seq_len: int, device: Any, dtype: Any) -> None:
            self.max_seq_len_cached = seq_len
            if seq_len > self.max_position_embeddings:
                base = self.base * (
                    (self.scaling_factor * seq_len / self.max_position_embeddings)
                    - (self.scaling_factor - 1)
                ) ** (self.dim / (self.dim - 2))
                inv_freq = 1.0 / (
                    base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
                )
                self.register_buffer("inv_freq", inv_freq, persistent=False)
            positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            frequencies = torch.outer(positions, self.inv_freq)
            embedding = torch.cat((frequencies, frequencies), dim=-1)
            self.register_buffer(
                "cos_cached", embedding.cos().to(dtype), persistent=False
            )
            self.register_buffer(
                "sin_cached", embedding.sin().to(dtype), persistent=False
            )

    def apply_rotary_pos_emb(
        query: Any, key: Any, cos: Any, sin: Any,
        position_ids: Any, unsqueeze_dim: int = 1,
    ) -> tuple[Any, Any]:
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        return (
            (query * cos) + (modeling_llama2.rotate_half(query) * sin),
            (key * cos) + (modeling_llama2.rotate_half(key) * sin),
        )

    modeling_llama2.LlamaRotaryEmbedding = LegacyLlamaRotaryEmbedding
    modeling_llama2.LlamaLinearScalingRotaryEmbedding = (
        LegacyLlamaLinearScalingRotaryEmbedding
    )
    modeling_llama2.LlamaDynamicNTKScalingRotaryEmbedding = (
        LegacyLlamaDynamicNTKScalingRotaryEmbedding
    )
    modeling_llama2.apply_rotary_pos_emb = apply_rotary_pos_emb


def _prepare_qalign_config_compat(config_class: Any) -> None:
    """Supply recent Llama defaults absent from Q-Align's vendored config."""
    if not hasattr(config_class, "mlp_bias"):
        config_class.mlp_bias = False


def _prepare_qalign_model_compat(model: Any) -> None:
    """Restore legacy attention-selection flags consumed by Q-Align's forward."""
    implementation = str(getattr(model.config, "_attn_implementation", "eager"))
    model._use_flash_attention_2 = implementation == "flash_attention_2"
    model._use_sdpa = implementation == "sdpa"


def _redirect_qalign_assets(modeling_mplug_owl2: Any, model_path: str) -> None:
    """Redirect Q-Align's hard-coded Hub tokenizer/processor to local assets."""

    def proxy(loader: Any) -> Any:
        class LocalLoader:
            @staticmethod
            def from_pretrained(path: str, *args: Any, **kwargs: Any) -> Any:
                resolved = model_path if path == "q-future/one-align" else path
                return loader.from_pretrained(resolved, *args, **kwargs)

        return LocalLoader

    modeling_mplug_owl2.AutoTokenizer = proxy(modeling_mplug_owl2.AutoTokenizer)
    modeling_mplug_owl2.CLIPImageProcessor = proxy(
        modeling_mplug_owl2.CLIPImageProcessor
    )


class OneAlignRunner:
    """Lazy OneAlign runner returning scores on a 0..100 scale."""

    # 路径可用环境变量覆盖；默认权重放在 SSD 缓存上（机械盘上加载 15 GiB 权重要
    # 3~4 分钟，SSD 约 30 秒，而 IAA 会起多个实例反复加载）。
    REPO_DIR = os.environ.get("VERA_QALIGN_REPO", "/home/bc/code/iaa_models/Q-Align")
    MODEL_PATH = os.environ.get(
        "VERA_ONEALIGN_MODEL", "/var/cache/veradata/models/OneAlign"
    )

    def __init__(self, device: Optional[str] = None):
        self.device = device or config.IAA_DEVICE
        self.scorer = None
        self._torch = None
        self._lock = threading.Lock()

    def load(self) -> None:
        if self.scorer is not None:
            return
        import torch
        import transformers

        version = tuple(int(value) for value in transformers.__version__.split(".")[:2])
        if version < (4, 37):
            raise RuntimeError(
                f"transformers=={transformers.__version__} is too old for OneAlign"
            )
        if self.REPO_DIR not in sys.path:
            sys.path.insert(0, self.REPO_DIR)
        import transformers.models.llama.modeling_llama as modeling_llama
        import transformers.pytorch_utils as pytorch_utils

        # Q-Align imports the legacy Llama module with `import *`; recent
        # transformers narrows __all__ even though these symbols still exist.
        _prepare_qalign_transformers_compat(modeling_llama)
        if not hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):
            def find_pruneable(heads, n_heads, head_size, already_pruned_heads):
                heads = set(heads) - already_pruned_heads
                mask = torch.ones(n_heads, head_size)
                for head in heads:
                    shifted = head - sum(1 if previous < head else 0 for previous in already_pruned_heads)
                    mask[shifted] = 0
                mask = mask.view(-1).contiguous().eq(1)
                return heads, torch.arange(len(mask), dtype=torch.long)[mask].long()

            pytorch_utils.find_pruneable_heads_and_indices = find_pruneable
        from q_align.evaluate.scorer import QAlignAestheticScorer
        from q_align.model.configuration_mplug_owl2 import MPLUGOwl2Config
        import q_align.model.modeling_llama2 as modeling_llama2
        import q_align.model.modeling_mplug_owl2 as modeling_mplug_owl2
        from transformers.modeling_attn_mask_utils import (
            _prepare_4d_causal_attention_mask_for_sdpa,
        )

        if not hasattr(modeling_llama2, "_prepare_4d_causal_attention_mask_for_sdpa"):
            modeling_llama2._prepare_4d_causal_attention_mask_for_sdpa = (
                _prepare_4d_causal_attention_mask_for_sdpa
            )
        _install_qalign_rope_compat(modeling_llama2, torch)
        _prepare_qalign_config_compat(MPLUGOwl2Config)
        _redirect_qalign_assets(modeling_mplug_owl2, self.MODEL_PATH)
        self._torch = torch
        self.scorer = QAlignAestheticScorer(
            pretrained=self.MODEL_PATH, device=self.device
        ).eval()
        _prepare_qalign_model_compat(self.scorer.model.model)

    def _score_pils(self, images: list[Any]) -> list[float]:
        self.load()
        with self._lock, self._torch.inference_mode():
            raw = self.scorer(images).detach().float().cpu().tolist()
        return [max(0.0, min(100.0, float(score) * 100.0)) for score in raw]

    def score_path(self, path: str) -> dict[str, Optional[float]]:
        score = self._score_pils([_decode_rgb(path)])[0]
        return {"iaa_mixed": score, "onealign": score}

    def preprocess_path(self, path: str) -> dict[str, Any]:
        return {"pil": _decode_rgb(path)}

    def score_batch_pre(
        self, items: list[dict[str, Any]]
    ) -> list[dict[str, Optional[float]]]:
        output = [{"iaa_mixed": None, "onealign": None} for _ in items]
        indexes = [index for index, item in enumerate(items) if item.get("pil") is not None]
        if not indexes:
            return output
        scores = self._score_pils([items[index]["pil"] for index in indexes])
        for index, score in zip(indexes, scores):
            output[index] = {"iaa_mixed": score, "onealign": score}
        return output


__all__ = ["OneAlignRunner"]
