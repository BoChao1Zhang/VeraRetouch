"""OneAlign aesthetic scorer retained for canonical candidate ranking."""
from __future__ import annotations

import sys
import threading
from typing import Any, Optional

from PIL import Image, ImageOps

from . import config


def _decode_rgb(path: str) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


class OneAlignRunner:
    """Lazy OneAlign runner returning scores on a 0..100 scale."""

    REPO_DIR = "/home/bc/code/iaa_models/Q-Align"
    MODEL_PATH = "/home/bc/data/models/OneAlign"

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
        import transformers.pytorch_utils as pytorch_utils

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
        import q_align.model.modeling_llama2 as modeling_llama2
        from transformers.models.llama.modeling_llama import (
            _prepare_4d_causal_attention_mask_for_sdpa,
        )

        if not hasattr(modeling_llama2, "_prepare_4d_causal_attention_mask_for_sdpa"):
            modeling_llama2._prepare_4d_causal_attention_mask_for_sdpa = (
                _prepare_4d_causal_attention_mask_for_sdpa
            )
        self._torch = torch
        self.scorer = QAlignAestheticScorer(
            pretrained=self.MODEL_PATH, device=self.device
        ).eval()

    def _score_pils(self, images: list[Any]) -> list[float]:
        self.load()
        with self._lock, self._torch.inference_mode():
            raw = self.scorer(images).detach().float().cpu().tolist()
        return [max(0.0, min(100.0, float(score) * 100.0)) for score in raw]

    def score_path(self, path: str) -> dict[str, Optional[float]]:
        try:
            score = self._score_pils([_decode_rgb(path)])[0]
        except Exception:  # one decode/score failure is an unreliable candidate
            return {"iaa_mixed": None, "onealign": None}
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
        try:
            scores = self._score_pils([items[index]["pil"] for index in indexes])
        except Exception:
            return output
        for index, score in zip(indexes, scores):
            output[index] = {"iaa_mixed": score, "onealign": score}
        return output


__all__ = ["OneAlignRunner"]
