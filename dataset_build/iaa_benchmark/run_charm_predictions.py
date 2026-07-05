#!/usr/bin/env python3
"""Run Charm PARA scores on Photographer-IAA inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score Photographer-IAA with Charm.")
    parser.add_argument("--input", type=Path, required=True, help="generic_image_score.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="Prediction JSONL.")
    parser.add_argument("--checkpoint", type=Path, default=Path("/home/bc/data/models/Charm/para_charm.pth"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--patch-selection", default="frequency")
    parser.add_argument("--training-dataset", default="para")
    parser.add_argument("--backbone", default="facebook/dinov2-small")
    parser.add_argument("--batch-size", type=int, default=1, help="Reserved; Charm package predicts one image at a time.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
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
                row = json.loads(line)
                if "pred_score_0_100" in row:
                    seen.add(str(row["benchmark_id"]))
            except (KeyError, json.JSONDecodeError):
                continue
    return seen


def para_1_5_to_0_100(score: float) -> float:
    return max(0.0, min(100.0, (score - 1.0) / 4.0 * 100.0))


def score_to_0_100(score: float, training_dataset: str) -> float:
    if training_dataset == "para":
        return para_1_5_to_0_100(score)
    if training_dataset in {"ava", "tad66k"}:
        return max(0.0, min(100.0, (score - 1.0) / 9.0 * 100.0))
    return max(0.0, min(100.0, score * 100.0))


def patch_charm_tokenizer_for_py313(charm_tokenizer_cls) -> None:
    """Replace a locals()-mutation block that is unreliable on Python 3.13."""

    def high_res_preserve_ms(self, image, mask=None):
        image = self.pad_or_crop(image, self.lcm(self.scaled_patchsizes))
        if mask is not None:
            mask = self.pad_or_crop(mask, self.lcm(self.scaled_patchsizes))
            if mask.size()[1:] != image.size()[1:]:
                raise ValueError("Image size and mask size do not match.")

        patch_sizes = [
            x + (self.patch_size - x % self.patch_size) if x % self.patch_size != 0 else x
            for x in self.scaled_patchsizes
        ]
        patch_strides = [
            x + (self.patch_size - x % self.patch_size) if x % self.patch_size != 0 else x
            for x in self.scaled_patchsizes
        ]

        image_patches = self.image_to_patches(image, patch_sizes[-1], patch_strides[-1])
        importance = self.calculate_importance(
            self.patch_selection_strategy,
            image_patches,
            patch_sizes[-1],
            patch_strides[-1],
            mask,
        )

        n_patch_per_col = image.size()[-1] // patch_sizes[-1]
        n_patch_per_row = image.size()[-2] // patch_sizes[-1]
        ratio = 1 / self.num_scales
        n_patches = int((self.initial_hidden_size * ratio) / ((2 ** (self.num_scales - 1)) ** 2))

        selected: dict[int, list[int]] = {}
        patches_by_scale: dict[int, list[torch.Tensor]] = {}
        masks_by_scale: dict[int, list[int]] = {}

        high_scale = self.num_scales - 1
        selected[high_scale] = self.patch_selection(
            self.patch_selection_strategy,
            importance,
            n_patches,
            high_scale,
            range(len(image_patches)),
        )

        patches_by_scale[high_scale] = []
        for index in sorted(selected[high_scale]):
            patches_by_scale[high_scale].extend(
                self.image_to_patches(image_patches[index], self.patch_size, self.patch_stride)
            )
        masks_by_scale[high_scale] = [high_scale] * len(patches_by_scale[high_scale])

        remaining_patches = range(len(image_patches))
        intermediate_patches: list[torch.Tensor] = []
        intermediate_masks: list[int] = []
        selected_intermediate: list[int] = []
        for scale in range(self.num_scales):
            if scale == 0 or scale == high_scale:
                continue
            remaining_patches = list(set(remaining_patches) - set(selected[high_scale]))
            selected[scale] = self.patch_selection(
                self.patch_selection_strategy,
                importance,
                n_patches,
                scale,
                remaining_patches,
            )
            patch_count = 0
            for index in sorted(selected[scale]):
                resized = F.interpolate(
                    image_patches[index].unsqueeze(0),
                    size=(self.scaled_patchsizes[scale], self.scaled_patchsizes[scale]),
                    mode="bicubic",
                ).squeeze(0)
                patches = self.image_to_patches(resized, self.patch_size, self.patch_stride)
                intermediate_patches.extend(patches)
                patch_count += len(patches)
            masks_by_scale[scale] = [scale] * patch_count
            intermediate_masks.extend(masks_by_scale[scale])
            selected_intermediate.extend(selected[scale])

        selected_all = selected_intermediate + selected[high_scale]
        selected[0] = [x for x in range(0, len(image_patches)) if x not in selected_all]

        remaining_final: list[torch.Tensor] = []
        low_mask_count = 0
        for index in sorted(selected[0]):
            resized = F.interpolate(
                image_patches[index].unsqueeze(0),
                size=(self.scaled_patchsizes[0], self.scaled_patchsizes[0]),
                mode="bicubic",
            ).squeeze(0)
            patches = self.image_to_patches(resized, self.patch_size, self.patch_stride)
            remaining_final.extend(patches)
            low_mask_count += len(patches)
        masks_by_scale[0] = [0] * low_mask_count

        final = remaining_final + intermediate_patches + patches_by_scale[high_scale]
        mask_ms = masks_by_scale[0] + intermediate_masks + masks_by_scale[high_scale]
        final_tensor = torch.stack(final)

        masks = []
        for scale in range(self.num_scales):
            p = patch_sizes[scale] // self.patch_size
            binary_mask = self.create_binary_mask(
                (3, p * n_patch_per_row, p * n_patch_per_col),
                p,
                selected[scale],
            )
            masks.append(binary_mask)

        pos_embeds = self.prepare_pos_embed_ms(masks, self.pos_embed.shape[-1]).squeeze(0)
        final_tensor = torch.cat(
            (torch.zeros(1, final_tensor.shape[1], final_tensor.shape[2], final_tensor.shape[3]), final_tensor),
            dim=0,
        )
        mask_ms.insert(0, 0)

        if final_tensor.shape[0] != pos_embeds.shape[0]:
            raise ValueError("Pos embedding length doesn't match the tokens length.")

        if self.without_pad_or_dropping:
            return final_tensor, pos_embeds, torch.Tensor(mask_ms)
        if final_tensor.shape[0] < self.hidden_size:
            input_tensor = self.padding(final_tensor, self.hidden_size)
            pos_embeds = self.padding(pos_embeds.unsqueeze(-1).unsqueeze(-1), self.hidden_size).squeeze(-1).squeeze(-1)
            padded_area = self.hidden_size - final_tensor.shape[0]
            mask = torch.Tensor(mask_ms + [9] * padded_area)
        elif final_tensor.shape[0] > self.hidden_size:
            input_tensor, pos_embeds, mask = self.random_drop(final_tensor, pos_embeds, torch.Tensor(mask_ms))
        else:
            input_tensor = final_tensor
            mask = torch.Tensor(mask_ms)
        return input_tensor, pos_embeds, mask

    charm_tokenizer_cls.highResPreserve_ms = high_res_preserve_ms


def main() -> None:
    args = parse_args()
    from Charm_tokenizer.ImageProcessor import Charm_Tokenizer
    import Charm_tokenizer.Backbone as charm_backbone

    patch_charm_tokenizer_for_py313(Charm_Tokenizer)

    checkpoint = str(args.checkpoint.expanduser())
    charm_backbone.hf_hub_download = lambda repo_id, filename: checkpoint

    tokenizer = Charm_Tokenizer(
        patch_selection=args.patch_selection,
        training_dataset=args.training_dataset,
        backbone=args.backbone,
        without_pad_or_dropping=True,
    )
    scorer = charm_backbone.backbone(training_dataset=args.training_dataset, device=args.device)
    scorer.model = scorer.model.to(args.device).eval()

    rows = load_rows(args.input.expanduser(), args.limit)
    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    done = seen_ids(args.output.expanduser()) if args.resume else set()
    mode = "a" if args.resume else "w"

    with args.output.expanduser().open(mode, encoding="utf-8") as out:
        for row in tqdm(rows, desc="Charm"):
            benchmark_id = str(row["benchmark_id"])
            if benchmark_id in done:
                continue
            try:
                tokens, pos_embed, mask_token = tokenizer.preprocess(row["image_path"])
                with torch.inference_mode():
                    prediction = scorer.model(
                        tokens.unsqueeze(0).to(args.device),
                        pos_embed.unsqueeze(0).to(args.device),
                        mask_token.unsqueeze(0).to(args.device),
                    )
                    raw = float(scorer.mean_score(prediction)[0])
                payload = {
                    "benchmark_id": benchmark_id,
                    "pred_score_0_100": score_to_0_100(raw, args.training_dataset),
                    "raw_score": raw,
                    "model": f"Charm-{args.training_dataset}",
                    "patch_selection": args.patch_selection,
                    "training_dataset": args.training_dataset,
                    "backbone": args.backbone,
                }
            except Exception as exc:
                payload = {
                    "benchmark_id": benchmark_id,
                    "error": repr(exc),
                    "model": f"Charm-{args.training_dataset}",
                    "patch_selection": args.patch_selection,
                    "training_dataset": args.training_dataset,
                    "backbone": args.backbone,
                }
            out.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
