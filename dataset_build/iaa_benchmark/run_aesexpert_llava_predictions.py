#!/usr/bin/env python3
"""Run AesExpert LLaVA-v1.5 predictions on Photographer-IAA inputs.

This is a native LLaVA fallback for the AesExpert checkpoint. The public
AesExpert model uses the original LLaVA `llava_llama` config, which the local
vLLM image does not serve directly.
"""

from __future__ import annotations

import argparse
import enum
import importlib.machinery
import json
import re
import sys
import types
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

DEFAULT_PROMPT = (
    "Rate the overall photographic aesthetic quality of this image on a 0 to 100 scale. "
    "Answer with only one number."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score Photographer-IAA with AesExpert/LLaVA.")
    parser.add_argument("--input", type=Path, required=True, help="generic_image_score.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="Prediction JSONL.")
    parser.add_argument("--model-path", type=Path, default=Path("/home/bc/data/models/AesExpert-LLaVA-7B"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def install_torchvision_stub() -> None:
    """Avoid importing a broken torchvision build for text/vision model setup."""

    class InterpolationMode(enum.Enum):
        NEAREST = 0
        NEAREST_EXACT = 0
        BILINEAR = 2
        BICUBIC = 3
        BOX = 4
        HAMMING = 5
        LANCZOS = 1

    modules: dict[str, types.ModuleType] = {}
    for name in (
        "torchvision",
        "torchvision.transforms",
        "torchvision.transforms.functional",
        "torchvision.transforms.v2",
        "torchvision.transforms.v2.functional",
    ):
        module = types.ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        modules[name] = module
        sys.modules[name] = module

    modules["torchvision"].transforms = modules["torchvision.transforms"]
    modules["torchvision.transforms"].InterpolationMode = InterpolationMode
    modules["torchvision.transforms.v2"].functional = modules["torchvision.transforms.v2.functional"]


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


def parse_score(text: str) -> float | None:
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text)
    if not match:
        lowered = text.lower()
        qualitative_scores = (
            (("very low", "very unattractive", "terrible", "awful"), 5.0),
            (("excellent", "exceptional", "outstanding", "very high", "very attractive"), 95.0),
            (("high", "beautiful", "great", "good", "appealing", "attractive"), 80.0),
            (("average", "medium", "moderate", "fair", "ordinary"), 50.0),
            (("low", "poor", "bad", "unappealing", "inferior", "unattractive"), 20.0),
        )
        for words, score in qualitative_scores:
            if any(re.search(rf"\b{re.escape(word)}\b", lowered) for word in words):
                return score
        return None
    score = float(match.group(0))
    if score <= 5.0:
        score = (score - 1.0) / 4.0 * 100.0
    return max(0.0, min(100.0, score))


def main() -> None:
    args = parse_args()
    install_torchvision_stub()
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
    from transformers import AutoTokenizer

    rows = load_rows(args.input.expanduser(), args.limit)
    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = seen_ids(output_path) if args.resume else set()
    mode = "a" if args.resume else "w"

    model_path = str(args.model_path.expanduser())
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    ).eval().to(args.device)

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=args.device, dtype=torch.float16)
    image_processor = vision_tower.image_processor

    if getattr(model.config, "mm_use_im_patch_token", True):
        tokenizer.add_tokens(["<im_patch>"], special_tokens=True)
    if getattr(model.config, "mm_use_im_start_end", False):
        tokenizer.add_tokens(["<im_start>", "<im_end>"], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    with output_path.open(mode, encoding="utf-8") as out:
        for row in tqdm(rows, desc="AesExpert-LLaVA"):
            benchmark_id = str(row["benchmark_id"])
            if benchmark_id in done:
                continue
            try:
                image = Image.open(row["image_path"]).convert("RGB")
                image_tensor = process_images([image], image_processor, model.config).to(
                    device=args.device, dtype=torch.float16
                )
                conv = conv_templates["llava_v1"].copy()
                conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{DEFAULT_PROMPT}")
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer_image_token(
                    prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                ).unsqueeze(0).to(args.device)
                generation_kwargs: dict[str, Any] = {
                    "do_sample": args.temperature > 0,
                    "max_new_tokens": args.max_new_tokens,
                    "use_cache": True,
                }
                if args.temperature > 0:
                    generation_kwargs["temperature"] = args.temperature
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor,
                        **generation_kwargs,
                    )
                generated = output_ids[0]
                if generated.shape[0] > input_ids.shape[1]:
                    generated = generated[input_ids.shape[1] :]
                response = tokenizer.decode(generated, skip_special_tokens=True).strip()
                pred = parse_score(response)
                if pred is None:
                    payload = {
                        "benchmark_id": benchmark_id,
                        "error": f"no numeric score in response: {response!r}",
                        "model": "AesExpert-LLaVA-7B",
                        "raw_response": response,
                    }
                else:
                    payload = {
                        "benchmark_id": benchmark_id,
                        "pred_score_0_100": pred,
                        "model": "AesExpert-LLaVA-7B",
                        "raw_response": response,
                    }
            except Exception as exc:
                payload = {
                    "benchmark_id": benchmark_id,
                    "error": repr(exc),
                    "model": "AesExpert-LLaVA-7B",
                }
            out.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
