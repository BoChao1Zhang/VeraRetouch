#!/usr/bin/env python3
"""Run HumanAesExpert scores on Photographer-IAA inputs."""

from __future__ import annotations

import argparse
import json
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
    parser = argparse.ArgumentParser(description="Score Photographer-IAA with HumanAesExpert.")
    parser.add_argument("--input", type=Path, required=True, help="generic_image_score.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="Prediction JSONL.")
    parser.add_argument("--model-path", type=Path, default=Path("/home/bc/data/models/HumanAesExpert-8B"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--method", choices=["score", "metavoter"], default="metavoter")
    parser.add_argument("--max-num", type=int, default=12, help="Maximum dynamic image tiles.")
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


def install_timm_droppath_stub() -> None:
    import importlib.machinery
    import sys
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
    import sys
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


def transform_image(image: Image.Image, input_size: int) -> torch.Tensor:
    image = image.convert("RGB").resize((input_size, input_size), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False,
) -> list[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    }
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image(image_file: str, input_size: int = 448, max_num: int = 12) -> torch.Tensor:
    image = Image.open(image_file).convert("RGB")
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform_image(image, input_size) for image in images]
    return torch.stack(pixel_values)


def normalize_score(raw: float) -> float:
    if raw <= 1.5:
        return raw * 100.0
    if raw <= 5.0:
        return (raw - 1.0) / 4.0 * 100.0
    return raw


def main() -> None:
    args = parse_args()
    install_timm_droppath_stub()
    install_torchvision_stub()
    from transformers import AutoModel, AutoTokenizer

    rows = load_rows(args.input.expanduser(), args.limit)
    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    done = seen_ids(args.output.expanduser()) if args.resume else set()
    mode = "a" if args.resume else "w"

    model = AutoModel.from_pretrained(
        str(args.model_path.expanduser()),
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_flash_attn=args.use_flash_attn,
        trust_remote_code=True,
    ).eval().to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path.expanduser()), trust_remote_code=True, use_fast=False
    )
    question = "<image>\nRate the aesthetics of this human picture."

    with args.output.expanduser().open(mode, encoding="utf-8") as out:
        for row in tqdm(rows, desc="HumanAesExpert"):
            benchmark_id = str(row["benchmark_id"])
            if benchmark_id in done:
                continue
            try:
                pixel_values = load_image(row["image_path"], max_num=args.max_num).to(torch.float16).to(args.device)
                with torch.inference_mode():
                    if args.method == "metavoter":
                        raw = float(model.run_metavoter(tokenizer, pixel_values))
                    else:
                        raw = float(model.score(tokenizer, pixel_values, question))
                pred = max(0.0, min(100.0, normalize_score(raw)))
                payload = {
                    "benchmark_id": benchmark_id,
                    "pred_score_0_100": pred,
                    "raw_score": raw,
                    "method": args.method,
                    "model": "HumanAesExpert-8B",
                }
            except Exception as exc:
                payload = {
                    "benchmark_id": benchmark_id,
                    "error": repr(exc),
                    "method": args.method,
                    "model": "HumanAesExpert-8B",
                }
            out.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
