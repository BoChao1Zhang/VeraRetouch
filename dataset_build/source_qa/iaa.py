"""Artimuse + Charm image aesthetic assessment for source QA and construct.

Scores are normalized to 0..100 and blended as ``iaa_mixed``. The runner is
lazy-loading and process-local: importing this module never loads GPU weights.

Run:
    python -m dataset_build.source_qa.iaa --limit 100 --device cuda:0
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.machinery
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile

from . import config, db

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _clamp_0_100(v: float) -> float:
    return max(0.0, min(100.0, float(v)))


def score_to_0_100(score: float, training_dataset: str) -> float:
    """Normalize Charm raw scores to the common 0..100 aesthetic scale."""
    if training_dataset == "para":
        return _clamp_0_100((score - 1.0) / 4.0 * 100.0)
    if training_dataset in {"ava", "tad66k"}:
        return _clamp_0_100((score - 1.0) / 9.0 * 100.0)
    return _clamp_0_100(score * 100.0)


def blend_scores(artimuse: Optional[float], charm: Optional[float],
                 artimuse_weight: float, charm_weight: float) -> Optional[float]:
    vals = []
    if artimuse is not None:
        vals.append((max(0.0, float(artimuse_weight)), float(artimuse)))
    if charm is not None:
        vals.append((max(0.0, float(charm_weight)), float(charm)))
    denom = sum(w for w, _ in vals)
    if not vals or denom <= 0:
        return None
    return _clamp_0_100(sum(w * v for w, v in vals) / denom)


@contextlib.contextmanager
def _temporary_artimuse_compat_modules():
    """Provide the exact minimal modules ArtiMuse needs, then restore sys.modules.

    The local Charm package needs real ``torchvision.transforms``. ArtiMuse only
    needs ``timm.models.layers.DropPath`` during class import, so a temporary
    shim avoids version drift without poisoning the rest of the process.
    """
    import enum
    import types

    names = [
        "timm", "timm.models", "timm.models.layers",
        "torchvision", "torchvision.transforms",
        "torchvision.transforms.v2", "torchvision.transforms.v2.functional",
    ]
    sentinel = object()
    saved = {name: sys.modules.get(name, sentinel) for name in names}

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

    class InterpolationMode(enum.Enum):
        NEAREST = 0
        NEAREST_EXACT = 0
        BILINEAR = 2
        BICUBIC = 3
        BOX = 4
        HAMMING = 5
        LANCZOS = 1

    timm_module = types.ModuleType("timm")
    models_module = types.ModuleType("timm.models")
    layers_module = types.ModuleType("timm.models.layers")
    timm_module.__spec__ = importlib.machinery.ModuleSpec("timm", loader=None)
    models_module.__spec__ = importlib.machinery.ModuleSpec("timm.models", loader=None)
    layers_module.__spec__ = importlib.machinery.ModuleSpec("timm.models.layers", loader=None)
    layers_module.DropPath = DropPath
    models_module.layers = layers_module
    timm_module.models = models_module

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

    sys.modules.update({
        "timm": timm_module,
        "timm.models": models_module,
        "timm.models.layers": layers_module,
        "torchvision": torchvision_module,
        "torchvision.transforms": transforms_module,
        "torchvision.transforms.v2": transforms_v2_module,
        "torchvision.transforms.v2.functional": transforms_v2_functional_module,
    })
    try:
        yield
    finally:
        for name, module in saved.items():
            if module is sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _artimuse_image(image_file: str, device: str, input_size: int = 448) -> torch.Tensor:
    image = Image.open(image_file).convert("RGB")
    image = image.resize((input_size, input_size), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0).to(torch.bfloat16).to(device)


class ArtiMuseScorer:
    def __init__(self, model_path: str, repo_dir: str, device: str,
                 use_flash_attn: bool = False):
        self.model_path = Path(model_path).expanduser()
        self.repo_dir = Path(repo_dir).expanduser()
        self.device = device
        self.use_flash_attn = use_flash_attn
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self.model is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(f"ArtiMuse model path not found: {self.model_path}")
        if not self.repo_dir.exists():
            raise FileNotFoundError(f"ArtiMuse repo path not found: {self.repo_dir}")
        src = str(self.repo_dir / "src")
        art = str(self.repo_dir / "src" / "artimuse")
        for p in (art, src):
            if p not in sys.path:
                sys.path.insert(0, p)
        with _temporary_artimuse_compat_modules():
            from artimuse.internvl.model.internvl_chat.modeling_artimuse import InternVLChatModel
            from transformers import AutoTokenizer

            self.model = InternVLChatModel.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_flash_attn=self.use_flash_attn,
            ).eval().to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path), trust_remote_code=True, use_fast=False
            )
        self.generation_config = {
            "max_new_tokens": 8192,
            "do_sample": False,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

    def score_path(self, path: str) -> float:
        self._load()
        pixel_values = _artimuse_image(path, self.device)
        with self._lock, torch.inference_mode():
            score = float(self.model.score(
                self.device, self.tokenizer, pixel_values, dict(self.generation_config)
            ))
        return _clamp_0_100(score)


def patch_charm_tokenizer_for_py313(charm_tokenizer_cls) -> None:
    """Replace Charm's locals()-mutation block, which is unreliable on Python 3.13."""

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
            self.patch_selection_strategy, importance, n_patches, high_scale, range(len(image_patches))
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
                self.patch_selection_strategy, importance, n_patches, scale, remaining_patches
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
                (3, p * n_patch_per_row, p * n_patch_per_col), p, selected[scale]
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


class CharmScorer:
    def __init__(self, checkpoint: str, device: str, patch_selection: str,
                 training_dataset: str, backbone: str, model_dir: str):
        self.checkpoint = Path(checkpoint).expanduser()
        self.model_dir = Path(model_dir).expanduser()
        self.device = device
        self.patch_selection = patch_selection
        self.training_dataset = training_dataset
        self.backbone = backbone
        self.tokenizer = None
        self.scorer = None
        self._lock = threading.Lock()

    def _local_hf_download(self, repo_id: str, filename: str, *args, **kwargs) -> str:
        if filename == self.checkpoint.name and self.checkpoint.exists():
            return str(self.checkpoint)
        local = self.model_dir / filename
        if local.exists():
            return str(local)
        raise FileNotFoundError(f"Charm local file not found: {local}")

    def _load(self) -> None:
        if self.scorer is not None:
            return
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"Charm checkpoint not found: {self.checkpoint}")
        from Charm_tokenizer.ImageProcessor import Charm_Tokenizer
        import Charm_tokenizer.Backbone as charm_backbone
        import Charm_tokenizer.ImageProcessor as charm_image_processor

        patch_charm_tokenizer_for_py313(Charm_Tokenizer)
        charm_backbone.hf_hub_download = self._local_hf_download
        charm_image_processor.hf_hub_download = self._local_hf_download
        self.tokenizer = Charm_Tokenizer(
            patch_selection=self.patch_selection,
            training_dataset=self.training_dataset,
            backbone=self.backbone,
            without_pad_or_dropping=True,
        )
        self.scorer = charm_backbone.backbone(training_dataset=self.training_dataset, device=self.device)
        self.scorer.model = self.scorer.model.to(self.device).eval()

    def score_path(self, path: str) -> float:
        self._load()
        tokens, pos_embed, mask_token = self.tokenizer.preprocess(path)
        with self._lock, torch.inference_mode():
            prediction = self.scorer.model(
                tokens.unsqueeze(0).to(self.device),
                pos_embed.unsqueeze(0).to(self.device),
                mask_token.unsqueeze(0).to(self.device),
            )
            raw = float(self.scorer.mean_score(prediction)[0])
        return score_to_0_100(raw, self.training_dataset)


class MixedIAARunner:
    """Lazy Artimuse + Charm runner returning denormalized 0..100 scores."""

    def __init__(self, device: Optional[str] = None, artimuse_weight: Optional[float] = None,
                 charm_weight: Optional[float] = None, enable_artimuse: bool = True,
                 enable_charm: bool = True):
        self.device = device or config.IAA_DEVICE
        self.artimuse_weight = float(
            config.IAA_ARTIMUSE_WEIGHT if artimuse_weight is None else artimuse_weight
        )
        self.charm_weight = float(config.IAA_CHARM_WEIGHT if charm_weight is None else charm_weight)
        self.enable_artimuse = enable_artimuse
        self.enable_charm = enable_charm
        self._artimuse: Optional[ArtiMuseScorer] = None
        self._charm: Optional[CharmScorer] = None

    @property
    def artimuse(self) -> ArtiMuseScorer:
        if self._artimuse is None:
            self._artimuse = ArtiMuseScorer(
                model_path=config.IAA_ARTIMUSE_MODEL_PATH,
                repo_dir=config.IAA_ARTIMUSE_REPO_DIR,
                device=self.device,
                use_flash_attn=config.IAA_ARTIMUSE_USE_FLASH_ATTN,
            )
        return self._artimuse

    @property
    def charm(self) -> CharmScorer:
        if self._charm is None:
            self._charm = CharmScorer(
                checkpoint=config.IAA_CHARM_CHECKPOINT,
                device=self.device,
                patch_selection=config.IAA_CHARM_PATCH_SELECTION,
                training_dataset=config.IAA_CHARM_TRAINING_DATASET,
                backbone=config.IAA_CHARM_BACKBONE,
                model_dir=config.IAA_CHARM_MODEL_DIR,
            )
        return self._charm

    def score_path(self, path: str) -> Dict[str, Optional[float]]:
        out: Dict[str, Optional[float]] = {}
        art = charm = None
        if self.enable_artimuse:
            try:
                art = self.artimuse.score_path(path)
                out["artimuse"] = art
            except Exception as exc:  # noqa: BLE001 - one bad scorer must not kill the batch
                out["artimuse"] = None
                out["artimuse_error"] = f"{type(exc).__name__}: {str(exc)[:240]}"
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        if self.enable_charm:
            try:
                charm = self.charm.score_path(path)
                out["charm"] = charm
            except Exception as exc:  # noqa: BLE001
                out["charm"] = None
                out["charm_error"] = f"{type(exc).__name__}: {str(exc)[:240]}"
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        out["iaa_mixed"] = blend_scores(art, charm, self.artimuse_weight, self.charm_weight)
        return out


_COL = {"artimuse": "artimuse_score", "charm": "charm_score", "iaa_mixed": "iaa_mixed"}


def _as_db_scores(scores: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    return {k: v for k, v in scores.items() if k in _COL}


def run(limit: Optional[int] = None, corpus: Optional[str] = None,
        device: Optional[str] = None, resume: bool = True,
        verdict: Optional[str] = None, order: str = "asc", bq3: bool = False) -> dict:
    # ponytail: 双进程并行用 asc+desc 两端夹击（resume 的 NOT EXISTS 保证不重扫），
    # 比加 shard 参数省事；若需 >2 进程再上真分片

    runner = MixedIAARunner(device=device or config.IAA_DEVICE)
    conn = db.connect()
    run_id = db.start_run(conn, "iaa", {
        "device": device or config.IAA_DEVICE,
        "corpus": corpus,
        "verdict": verdict,
        "artimuse_weight": runner.artimuse_weight,
        "charm_weight": runner.charm_weight,
        "resume": resume,
    })
    where = ["a.asset_type='image'", "a.dup_of IS NULL"]
    params: list[Any] = []
    if resume:
        where.append("NOT EXISTS (SELECT 1 FROM iqa_scores s WHERE s.asset_id=a.asset_id AND s.metric='iaa_mixed')")
    if corpus:
        where.append("a.corpus=?")
        params.append(corpus)
    if verdict:
        # 优先灌 construct 会用到的池（keep 10.2 万），全量 13.5 万 ~1s/张跑不完整夜
        vs = [v.strip() for v in verdict.split(",") if v.strip()]
        where.append(f"a.auto_verdict IN ({','.join('?' * len(vs))})")
        params.extend(vs)
    if bq3:
        where.append("a.b_quality=3")   # construct 源池硬条件，不浪费算力在不可用行
    sql = (f"SELECT a.asset_id, a.path FROM assets a WHERE {' AND '.join(where)} "
           f"ORDER BY a.asset_id{' DESC' if order == 'desc' else ''}")
    if limit:
        sql += f" LIMIT {int(limit)}"
    todo = conn.execute(sql, params).fetchall()
    print(f"[iaa] {len(todo)} images to score with Artimuse+Charm", file=sys.stderr)

    n_ok = n_err = 0
    for i, r in enumerate(todo):
        aid, path = r["asset_id"], r["path"]
        try:
            scores = runner.score_path(path)
            db_scores = _as_db_scores(scores)
            db.add_scores(conn, aid, db_scores, run_id=run_id, model_version=config.IAA_MODEL_VERSION)
            denorm = {_COL[m]: v for m, v in db_scores.items() if v is not None}
            if db_scores.get("iaa_mixed") is not None:
                denorm["aesthetic"] = db_scores["iaa_mixed"]
            denorm["status"] = "iaa_done"
            db.update_asset_fields(conn, aid, **denorm)
            db.log_event(conn, aid, "iaa", "ok", scores, run_id)
            n_ok += 1
        except Exception as exc:  # noqa: BLE001
            db.log_event(conn, aid, "iaa", "error", {"err": str(exc)[:300]}, run_id)
            n_err += 1
        if (i + 1) % 50 == 0:
            conn.commit()
            print(f"[iaa] {i+1}/{len(todo)} ok={n_ok} err={n_err}", file=sys.stderr)
    conn.commit()
    db.finish_run(conn, run_id, {"ok": n_ok, "err": n_err})
    conn.close()
    return {"ok": n_ok, "err": n_err, "run_id": run_id}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--verdict", default=None, help="按 auto_verdict 过滤，如 keep 或 keep,review")
    ap.add_argument("--order", default="asc", choices=("asc", "desc"), help="扫描方向（双进程夹击用）")
    ap.add_argument("--bq3", action="store_true", help="只扫 b_quality=3（construct 可用池）")
    args = ap.parse_args()
    print(json.dumps(run(
        limit=args.limit,
        corpus=args.corpus,
        device=args.device,
        resume=not args.no_resume,
        verdict=args.verdict,
        order=args.order,
        bq3=args.bq3,
    ), ensure_ascii=False))


if __name__ == "__main__":
    main()
