"""Dedicated OBJECTIVE aesthetic scorer (CLIP ViT-L/14 + LAION linear head).

This is the model-based aesthetic signal for the SOURCE-IMAGE TAG + AESTHETIC
precompute pass (``tag_precompute.py``). It is intentionally self-contained and
GRACEFUL: if the CLIP weights or the LAION MLP checkpoint are missing/unloadable
(the download may not have finished), it disables itself, logs ONCE, and
``score()`` returns ``None`` rather than raising — the precompute driver then
falls back to the VLM aesthetic only.

Weights (verified on disk 2026-06-02):
  - clip_dir = /home/bc/data/models/clip-vit-large-patch14   (CLIPModel + CLIPProcessor)
  - mlp_path = /home/bc/data/models/laion_aesthetic_sac_logos_ava1_l14_linearMSE.pth

The LAION 'improved-aesthetic-predictor' head arch MUST match the checkpoint
keys exactly. The checkpoint stores ``layers.{0,2,4,6,7}`` Linear weights with
``layers.{1,3,5}`` dropouts, i.e. ``self.layers = nn.Sequential(...)`` of:
    Linear(768,1024), Dropout(0.2), Linear(1024,128), Dropout(0.2),
    Linear(128,64), Dropout(0.1), Linear(64,16), Linear(16,1)
The CLIP image embedding is L2-normalized before the MLP; the output is a
single ~[1,10] float (the standard LAION aesthetic scale).
"""
from __future__ import annotations

import os
import sys
import threading
from typing import Any, List, Optional, Sequence

_LOGGED_ONCE = False


def _log_once(msg: str) -> None:
    global _LOGGED_ONCE
    if not _LOGGED_ONCE:
        print(f"[aesthetic] {msg}", file=sys.stderr, flush=True)
        _LOGGED_ONCE = True


def _build_mlp(input_size: int = 768):
    """The LAION linearMSE head; keys land under ``layers.*`` (matches ckpt)."""
    import torch.nn as nn

    class _AestheticMLP(nn.Module):
        def __init__(self, in_dim: int):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(in_dim, 1024),   # layers.0
                nn.Dropout(0.2),           # layers.1
                nn.Linear(1024, 128),      # layers.2
                nn.Dropout(0.2),           # layers.3
                nn.Linear(128, 64),        # layers.4
                nn.Dropout(0.1),           # layers.5
                nn.Linear(64, 16),         # layers.6
                nn.Linear(16, 1),          # layers.7
            )

        def forward(self, x):
            return self.layers(x)

    return _AestheticMLP(input_size)


class AestheticScorer:
    """CLIP ViT-L/14 image embedding -> LAION linear head -> ~[1,10] float.

    GRACEFUL: any load failure sets ``self.enabled=False`` and ``score()`` /
    ``score_batch()`` return ``None`` (logged once). Never raises on construction
    or scoring so the precompute run cannot crash on a missing/partial download.
    Uses cuda if available else cpu; always ``eval()`` + ``no_grad``.
    """

    def __init__(
        self,
        clip_dir: str = "/home/bc/data/models/clip-vit-large-patch14",
        mlp_path: str = "/home/bc/data/models/laion_aesthetic_sac_logos_ava1_l14_linearMSE.pth",
        device: Optional[str] = None,
    ) -> None:
        self.clip_dir = clip_dir
        self.mlp_path = mlp_path
        self.enabled = False
        self.device = "cpu"
        self._model = None
        self._processor = None
        self._mlp = None
        self._lock = threading.Lock()  # CLIP forward is not safe across threads

        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            if not (clip_dir and os.path.isdir(clip_dir)):
                _log_once(f"disabled: CLIP dir missing/unreadable: {clip_dir}")
                return
            if not (mlp_path and os.path.isfile(mlp_path)):
                _log_once(f"disabled: LAION MLP checkpoint missing: {mlp_path}")
                return

            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

            model = CLIPModel.from_pretrained(clip_dir)
            processor = CLIPProcessor.from_pretrained(clip_dir)
            model = model.to(self.device).eval()

            sd = torch.load(mlp_path, map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            mlp = _build_mlp(768)
            missing, unexpected = mlp.load_state_dict(sd, strict=False)
            if missing or unexpected:
                _log_once(
                    f"MLP load_state_dict mismatch missing={list(missing)} "
                    f"unexpected={list(unexpected)}"
                )
            mlp = mlp.to(self.device).eval()

            self._model = model
            self._processor = processor
            self._mlp = mlp
            self.enabled = True
        except Exception as e:  # any import/load failure -> disabled, never raise
            _log_once(f"disabled (load failed): {type(e).__name__}: {e}")
            self.enabled = False

    # ---- internals ------------------------------------------------------
    @staticmethod
    def _to_pil(image: Any):
        from PIL import Image

        if isinstance(image, Image.Image):
            return image.convert("RGB")
        # treat as path
        with Image.open(image) as im:
            return im.convert("RGB")

    def _scores_for_pils(self, pils: List[Any]) -> Optional[List[float]]:
        if not self.enabled or self._model is None:
            return None
        try:
            import torch

            with self._lock, torch.no_grad():
                inputs = self._processor(images=pils, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self.device)
                feats = self._model.get_image_features(pixel_values=pixel_values)
                feats = feats.float()
                feats = feats / feats.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
                out = self._mlp(feats).squeeze(-1)  # [N]
                return [float(v) for v in out.detach().cpu().tolist()]
        except Exception as e:  # per-call failure -> None (caller falls back)
            _log_once(f"score failed: {type(e).__name__}: {e}")
            return None

    # ---- public API -----------------------------------------------------
    def score(self, image: Any) -> Optional[float]:
        """Return a single ~[1,10] aesthetic float, or None if disabled/failed.

        ``image`` is a PIL.Image or a path. Decode failure on one image -> None
        (never raises)."""
        if not self.enabled:
            return None
        try:
            pil = self._to_pil(image)
        except Exception as e:
            _log_once(f"decode failed: {type(e).__name__}: {e}")
            return None
        res = self._scores_for_pils([pil])
        return res[0] if res else None

    def score_batch(self, images: Sequence[Any]) -> List[Optional[float]]:
        """Score a list of PIL.Image/paths. Returns list aligned with input;
        entries that fail to decode become None. All-None if disabled."""
        if not self.enabled:
            return [None] * len(images)
        pils: List[Any] = []
        idx_map: List[int] = []
        out: List[Optional[float]] = [None] * len(images)
        for i, im in enumerate(images):
            try:
                pils.append(self._to_pil(im))
                idx_map.append(i)
            except Exception:
                continue
        if not pils:
            return out
        scores = self._scores_for_pils(pils)
        if scores is None:
            return out
        for j, i in enumerate(idx_map):
            out[i] = scores[j]
        return out


class Tad66kAestheticLabels:
    """Best-effort reader of TAD66K aesthetic labels (filename -> mean score).

    Optional cross-coverage signal. Reads ``train.csv``/``test.csv`` under
    ``csv_dir`` if present (columns include an image filename + a score). If the
    files are absent or unparseable it stays empty and ``score_for(name)``
    returns None — never raises.
    """

    _SCORE_COLS = ("score", "mean", "label", "aesthetic", "mos")
    _NAME_COLS = ("image", "img", "filename", "name", "path")

    def __init__(self, csv_dir: str):
        self.csv_dir = csv_dir
        self.labels: dict = {}
        try:
            self._load()
        except Exception as e:  # best-effort
            _log_once(f"TAD66K labels load failed: {type(e).__name__}: {e}")

    def _load(self) -> None:
        import csv

        for fn in ("train.csv", "test.csv", "val.csv"):
            p = os.path.join(self.csv_dir, fn)
            if not os.path.isfile(p):
                continue
            with open(p, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fields = [c.lower() for c in (reader.fieldnames or [])]
                name_c = next((c for c in (reader.fieldnames or []) if c.lower() in self._NAME_COLS), None)
                score_c = next((c for c in (reader.fieldnames or []) if c.lower() in self._SCORE_COLS), None)
                if name_c is None or score_c is None:
                    continue
                for row in reader:
                    try:
                        key = os.path.basename(str(row[name_c]).strip())
                        self.labels[key] = float(row[score_c])
                    except (KeyError, TypeError, ValueError):
                        continue

    def score_for(self, image_path_or_name: str) -> Optional[float]:
        if not self.labels:
            return None
        return self.labels.get(os.path.basename(str(image_path_or_name).strip()))
