#!/usr/bin/env python3
"""Run ArtiMuse scores on Photographer-IAA inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score Photographer-IAA with ArtiMuse.")
    parser.add_argument("--input", type=Path, required=True, help="generic_image_score.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="Prediction JSONL.")
    parser.add_argument("--model-path", type=Path, default=Path("/home/bc/data/models/ArtiMuse"))
    parser.add_argument("--repo-dir", type=Path, default=Path("/home/bc/code/iaa_models/ArtiMuse"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--use-flash-attn", action="store_true")
    return parser.parse_args()


def load_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def install_timm_droppath_stub() -> None:
    import importlib.machinery
    import types

    class DropPath(torch.nn.Module):
        def __init__(self, drop_prob: float = 0.0) -> None:
            super().__init__()
            self.drop_prob = drop_prob

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.drop_prob == 0.0 or not self.training:
                return x
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            return x.div(keep_prob) * random_tensor

    timm_module = types.ModuleType("timm")
    models_module = types.ModuleType("timm.models")
    layers_module = types.ModuleType("timm.models.layers")
    timm_module.__spec__ = importlib.machinery.ModuleSpec("timm", loader=None)
    models_module.__spec__ = importlib.machinery.ModuleSpec("timm.models", loader=None)
    layers_module.__spec__ = importlib.machinery.ModuleSpec("timm.models.layers", loader=None)
    layers_module.DropPath = DropPath
    models_module.layers = layers_module
    timm_module.models = models_module
    sys.modules["timm"] = timm_module
    sys.modules["timm.models"] = models_module
    sys.modules["timm.models.layers"] = layers_module


def install_torchvision_stub() -> None:
    import enum
    import importlib.machinery
    import types

    class InterpolationMode(enum.Enum):
        NEAREST = 0
        NEAREST_EXACT = 0
        BILINEAR = 2
        BICUBIC = 3
        BOX = 4
        HAMMING = 5
        LANCZOS = 1

    torchvision_module = types.ModuleType("torchvision")
    transforms_module = types.ModuleType("torchvision.transforms")
    transforms_v2_module = types.ModuleType("torchvision.transforms.v2")
    transforms_v2_functional_module = types.ModuleType("torchvision.transforms.v2.functional")
    torchvision_module.__spec__ = importlib.machinery.ModuleSpec("torchvision", loader=None)
    transforms_module.__spec__ = importlib.machinery.ModuleSpec("torchvision.transforms", loader=None)
    transforms_v2_module.__spec__ = importlib.machinery.ModuleSpec("torchvision.transforms.v2", loader=None)
    transforms_v2_functional_module.__spec__ = importlib.machinery.ModuleSpec(
        "torchvision.transforms.v2.functional", loader=None
    )
    transforms_module.InterpolationMode = InterpolationMode
    transforms_v2_module.functional = transforms_v2_functional_module
    torchvision_module.transforms = transforms_module
    sys.modules["torchvision"] = torchvision_module
    sys.modules["torchvision.transforms"] = transforms_module
    sys.modules["torchvision.transforms.v2"] = transforms_v2_module
    sys.modules["torchvision.transforms.v2.functional"] = transforms_v2_functional_module


def seen_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                seen.add(str(json.loads(line)["benchmark_id"]))
            except (KeyError, json.JSONDecodeError):
                continue
    return seen


def load_image(image_file: str, device: str, input_size: int = 448) -> torch.Tensor:
    image = Image.open(image_file).convert("RGB")
    image = image.resize((input_size, input_size), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0).to(torch.bfloat16).to(device)


def main() -> None:
    args = parse_args()
    install_timm_droppath_stub()
    install_torchvision_stub()
    sys.path.insert(0, str(args.repo_dir / "src"))
    sys.path.insert(0, str(args.repo_dir / "src" / "artimuse"))
    from artimuse.internvl.model.internvl_chat.modeling_artimuse import InternVLChatModel
    from transformers import AutoTokenizer

    rows = load_rows(args.input.expanduser(), args.limit)
    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    done = seen_ids(args.output.expanduser()) if args.resume else set()
    mode = "a" if args.resume else "w"

    model = InternVLChatModel.from_pretrained(
        str(args.model_path.expanduser()),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=args.use_flash_attn,
    ).eval().to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path.expanduser()), trust_remote_code=True, use_fast=False
    )
    generation_config = {
        "max_new_tokens": 8192,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
    }
    with args.output.expanduser().open(mode, encoding="utf-8") as out:
        for row in tqdm(rows, desc="ArtiMuse"):
            benchmark_id = str(row["benchmark_id"])
            if benchmark_id in done:
                continue
            try:
                pixel_values = load_image(row["image_path"], args.device)
                with torch.inference_mode():
                    score = float(model.score(args.device, tokenizer, pixel_values, dict(generation_config)))
                pred = max(0.0, min(100.0, score))
                payload = {
                    "benchmark_id": benchmark_id,
                    "pred_score_0_100": pred,
                    "raw_score": score,
                    "model": "ArtiMuse",
                }
            except Exception as exc:
                payload = {
                    "benchmark_id": benchmark_id,
                    "error": repr(exc),
                    "model": "ArtiMuse",
                }
            out.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
