#!/usr/bin/env python3
"""Run Q-Align/OneAlign aesthetic scores on Photographer-IAA inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score Photographer-IAA with OneAlign.")
    parser.add_argument("--input", type=Path, required=True, help="generic_image_score.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="Prediction JSONL.")
    parser.add_argument("--model-path", type=Path, default=Path("/home/bc/data/models/OneAlign"))
    parser.add_argument("--repo-dir", type=Path, default=Path("/home/bc/code/iaa_models/Q-Align"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--prompt-template",
        choices=("quickstart", "paper-iaa"),
        default="quickstart",
        help="quickstart uses QAlignAestheticScorer; paper-iaa mirrors q_align/evaluate/iaa_eval.py.",
    )
    return parser.parse_args()


def patch_transformers_pruning() -> None:
    import transformers.pytorch_utils as pytorch_utils

    if hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):
        return

    def find_pruneable_heads_and_indices(
        heads: list[int] | set[int],
        n_heads: int,
        head_size: int,
        already_pruned_heads: set[int],
    ) -> tuple[set[int], torch.LongTensor]:
        heads = set(heads) - already_pruned_heads
        mask = torch.ones(n_heads, head_size)
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask), dtype=torch.long)[mask].long()
        return heads, index

    pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices


def patch_qalign_attention_mask() -> None:
    import q_align.model.modeling_llama2 as modeling_llama2
    from transformers.models.llama.modeling_llama import _prepare_4d_causal_attention_mask_for_sdpa

    if not hasattr(modeling_llama2, "_prepare_4d_causal_attention_mask_for_sdpa"):
        modeling_llama2._prepare_4d_causal_attention_mask_for_sdpa = _prepare_4d_causal_attention_mask_for_sdpa


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


def batched(rows: list[dict[str, Any]], batch_size: int):
    for i in range(0, len(rows), batch_size):
        yield rows[i : i + batch_size]


class PaperIAAScorer(torch.nn.Module):
    """OneAlign scorer matching q_align/evaluate/iaa_eval.py prompt and score mapping."""

    def __init__(self, pretrained: str, device: str) -> None:
        super().__init__()
        from q_align.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from q_align.conversation import conv_templates
        from q_align.mm_utils import get_model_name_from_path, tokenizer_image_token
        from q_align.model.builder import load_pretrained_model

        model_name = get_model_name_from_path(pretrained)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            pretrained, None, model_name, device=device
        )

        inp = "How would you rate the aesthetics of this image?"
        conv = conv_templates["mplug_owl2"].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt() + " The aesthetics of the image is"

        self.preferential_ids_ = [
            id_[1] for id_ in tokenizer(["excellent", "good", "fair", "poor", "bad"])["input_ids"]
        ]
        self.weight_tensor = torch.tensor([1, 0.75, 0.5, 0.25, 0.0], dtype=torch.float16).to(model.device)
        self.model = model
        self.image_processor = image_processor
        self.input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(model.device)

    @staticmethod
    def expand2square(pil_img: Image.Image, background_color: tuple[int, int, int]) -> Image.Image:
        width, height = pil_img.size
        if width == height:
            return pil_img
        if width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

    def forward(self, images: list[Image.Image]) -> torch.Tensor:
        images = [
            self.expand2square(img, tuple(int(x * 255) for x in self.image_processor.image_mean))
            for img in images
        ]
        image_tensor = (
            self.image_processor.preprocess(images, return_tensors="pt")["pixel_values"]
            .half()
            .to(self.model.device)
        )
        output_logits = self.model(
            self.input_ids.repeat(image_tensor.shape[0], 1),
            images=image_tensor,
        )["logits"][:, -1, self.preferential_ids_]
        return torch.softmax(output_logits, -1) @ self.weight_tensor


def main() -> None:
    args = parse_args()
    patch_transformers_pruning()
    sys.path.insert(0, str(args.repo_dir.expanduser()))
    from q_align.evaluate.scorer import QAlignAestheticScorer

    patch_qalign_attention_mask()

    rows = load_rows(args.input.expanduser(), args.limit)
    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    done = seen_ids(args.output.expanduser()) if args.resume else set()
    rows = [row for row in rows if str(row["benchmark_id"]) not in done]
    mode = "a" if args.resume else "w"

    if args.prompt_template == "paper-iaa":
        scorer = PaperIAAScorer(pretrained=str(args.model_path.expanduser()), device=args.device).eval()
    else:
        scorer = QAlignAestheticScorer(pretrained=str(args.model_path.expanduser()), device=args.device).eval()

    with args.output.expanduser().open(mode, encoding="utf-8") as out:
        for batch in tqdm(list(batched(rows, args.batch_size)), desc="OneAlign"):
            ok_rows = []
            images = []
            for row in batch:
                try:
                    images.append(Image.open(row["image_path"]).convert("RGB"))
                    ok_rows.append(row)
                except Exception as exc:
                    payload = {
                        "benchmark_id": str(row["benchmark_id"]),
                        "error": repr(exc),
                        "model": "OneAlign",
                    }
                    out.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            if not ok_rows:
                out.flush()
                continue
            try:
                with torch.inference_mode():
                    scores = scorer(images).detach().float().cpu().tolist()
                for row, score in zip(ok_rows, scores):
                    pred = max(0.0, min(100.0, float(score) * 100.0))
                    payload = {
                        "benchmark_id": str(row["benchmark_id"]),
                        "pred_score_0_100": pred,
                        "raw_score_0_1": float(score),
                        "model": "OneAlign",
                        "prompt_template": args.prompt_template,
                    }
                    out.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            except Exception as exc:
                for row in ok_rows:
                    payload = {
                        "benchmark_id": str(row["benchmark_id"]),
                        "error": repr(exc),
                        "model": "OneAlign",
                    }
                    out.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
