"""Lazy instance-level SAM3 adapter retained for canonical batch relabeling."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np


class Sam3Masker:
    """Load SAM3 once and expose native-resolution text-prompt masks."""

    def __init__(
        self,
        model_dir: str = "/home/bc/data/models",
        device: str = "cuda",
        dtype: str = "bfloat16",
        score_threshold: float = 0.3,
        mask_threshold: float = 0.5,
    ) -> None:
        self.model_dir = model_dir
        self.device = device
        self.dtype = dtype
        self.score_threshold = float(score_threshold)
        self.mask_threshold = float(mask_threshold)
        self._det = None
        self._proc = None
        self._torch = None

    def _ensure_loaded(self) -> None:
        if self._det is not None:
            return
        import torch
        from transformers import Sam3Processor, Sam3VideoModel

        dtype = getattr(torch, self.dtype) if isinstance(self.dtype, str) else self.dtype
        video_model = Sam3VideoModel.from_pretrained(
            self.model_dir, local_files_only=True, dtype=dtype
        ).eval()
        if self.device and self.device != "cpu":
            video_model = video_model.to(self.device)
        self._torch = torch
        self._det = video_model.detector_model
        self._proc = Sam3Processor.from_pretrained(self.model_dir, local_files_only=True)

    @staticmethod
    def _as_pil(image: Any):
        from PIL import Image, ImageOps

        if isinstance(image, (str, Path)):
            source = Image.open(str(image))
            result = ImageOps.exif_transpose(source).convert("RGB")
        elif isinstance(image, np.ndarray):
            result = Image.fromarray(np.ascontiguousarray(image).astype(np.uint8), "RGB")
        elif hasattr(image, "convert"):
            result = image.convert("RGB")
        else:
            raise TypeError(f"unsupported image type: {type(image)}")
        return result, (result.height, result.width)

    def mask(
        self,
        image: Any,
        prompt: str,
        native_size: Optional[tuple[int, int]] = None,
        *,
        soft: bool = True,
        reduce: str = "max",
        min_score: float = 0.3,
    ) -> np.ndarray:
        self._ensure_loaded()
        torch = self._torch
        source, decoded_size = self._as_pil(image)
        height, width = native_size or decoded_size
        inputs = self._proc(images=source, text=prompt, return_tensors="pt").to(
            self._det.device
        )
        with torch.inference_mode():
            outputs = self._det(**inputs)
        result = self._proc.post_process_instance_segmentation(
            outputs,
            threshold=float(min(min_score, self.score_threshold)),
            mask_threshold=self.mask_threshold,
            target_sizes=[(height, width)],
        )[0]
        masks = result.get("masks")
        scores = result.get("scores")
        if masks is None or len(masks) == 0:
            return np.zeros((height, width), dtype=np.float32)
        values = (
            masks.detach().to(torch.float32).cpu().numpy()
            if hasattr(masks, "detach")
            else np.asarray(masks, dtype=np.float32)
        )
        if values.ndim == 2:
            values = values[None]
        if scores is not None:
            scores_array = (
                scores.detach().to(torch.float32).cpu().numpy()
                if hasattr(scores, "detach")
                else np.asarray(scores)
            )
            keep = scores_array >= float(min_score)
            if not keep.any():
                return np.zeros((height, width), dtype=np.float32)
            values = values[keep]
        combined = values.sum(axis=0) if reduce == "sum" else values.max(axis=0)
        combined = np.clip(combined, 0.0, 1.0).astype(np.float32)
        return combined if soft else (combined >= self.mask_threshold).astype(np.float32)

    def masks(
        self, image: Any, prompts: Sequence[str], **kwargs: Any
    ) -> dict[str, np.ndarray]:
        source, native_size = self._as_pil(image)
        kwargs.setdefault("native_size", native_size)
        return {prompt: self.mask(source, prompt, **kwargs) for prompt in prompts}


__all__ = ["Sam3Masker"]
