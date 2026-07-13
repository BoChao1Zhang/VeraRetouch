"""Artimuse + Charm image aesthetic assessment for source QA and construct.

Scores are normalized to 0..100 and blended as ``iaa_mixed``. The runner is
lazy-loading and process-local: importing this module never loads GPU weights.

环境注意（2026-07-05）: base conda 的 transformers==4.36 没有 Qwen2*，ArtiMuse
(InternVL-8B/Qwen2) 会 ImportError —— 必须用隔离 venv 跑本模块：
    /home/bc/.venvs/iaa437/bin/python -m dataset_build.source_qa.iaa ...
(venv 继承 base 的 torch/psycopg，仅覆盖 transformers==4.37.2 + peft==0.10.0)

Run (推荐 batch 流水线；--batch 1 回退旧逐张路径):
    /home/bc/.venvs/iaa437/bin/python -m dataset_build.source_qa.iaa \
        --device cuda:0 --verdict keep --bq3 --batch 16
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.machinery
import json
import os
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def _decode_rgb(image_file: str) -> Image.Image:
    """一次解码给两个模型共用（batch 流水线里省一次 JPEG decode）。"""
    return Image.open(image_file).convert("RGB")


def _artimuse_tensor_cpu(image: Image.Image, input_size: int = 448) -> torch.Tensor:
    """(1,3,448,448) bf16 CPU 张量；与旧 _artimuse_image 逐位一致，只是不上卡。"""
    image = image.resize((input_size, input_size), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0).to(torch.bfloat16)


def _artimuse_image(image_file: str, device: str, input_size: int = 448) -> torch.Tensor:
    return _artimuse_tensor_cpu(_decode_rgb(image_file), input_size).to(device)


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
        # fail-fast 守卫（2026-07-05 事故）：base conda 的 transformers 4.36 没有
        # Qwen2*，ArtiMuse 只会每张图静默 ImportError、把 iaa_mixed 污染成纯 Charm。
        # 版本不对必须当场炸，提示切 venv，绝不允许带病打分。
        import transformers
        ver = tuple(int(x) for x in transformers.__version__.split(".")[:2])
        if ver < (4, 37):
            import sys as _sys
            raise RuntimeError(
                f"transformers=={transformers.__version__} 无 Qwen2 支持，ArtiMuse 起不来；"
                "请用隔离 venv 运行: /home/bc/.venvs/iaa437/bin/python -m dataset_build.source_qa.iaa ...\n"
                f"  [diag] exe={_sys.executable}\n"
                f"  [diag] transformers={transformers.__file__}\n"
                f"  [diag] sys.path[:6]={_sys.path[:6]}"
            )
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
        self._build_batch_inputs()

    # ArtiMuse 官方 score() 里的原始提示词（batch 路径复用同一 prompt，保证分数一致）
    _SCORE_QUESTION = """
        Rate the aesthetics score of the image in 0-100.
        In the output format, numbers are replaced by 2 corresponding letters,
        and the mapping relationship is:
        score 0 to 25: 0-aa, 1-ab, 2-ac, 3-ad, ... , 25-az,
        score 26 to 50: 26-ca, 27-cb, 28-cc, 29-cd, ..., 50-cy,
        score 51 to 75: 51-da, 52-db, 53-dc, 54-dd, ..., 75-dy,
        score 76 to 100: 76-ea, 77-eb, 78-ec, 79-ed, ..., 100-ey.

        The answer only outputs 2 corresponding letters.
        """

    def _build_batch_inputs(self) -> None:
        """预构建定长 prompt（所有图完全一致），batch 前向零 tokenize 开销。

        官方 score() 是一次 LM 前向取末位 101 个分数 token 的 softmax 期望，
        没有自回归生成 —— 因此 448px 定长输入 + 同一 prompt 天然可 batch。
        """
        from internvl.conversation import get_conv_template
        from internvl.model.internvl_chat.aes_tokens import AESTHETICS_TOKEN_LIST

        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        template = get_conv_template(self.model.template)
        template.system_message = self.model.system_message
        template.append_message(template.roles[0], "<image>\n" + self._SCORE_QUESTION)
        template.append_message(template.roles[1], None)
        query = template.get_prompt().replace(
            "<image>", "<img>" + "<IMG_CONTEXT>" * self.model.num_image_token + "</img>", 1
        )
        inputs = self.tokenizer(query, return_tensors="pt")
        self._prompt_ids = inputs["input_ids"].to(self.device)
        self._prompt_mask = inputs["attention_mask"].to(self.device)
        self._pref_ids = torch.tensor(
            [self.tokenizer.convert_tokens_to_ids(w) for w in AESTHETICS_TOKEN_LIST],
            device=self.device,
        )
        self._score_weight = torch.arange(101, dtype=torch.float32, device=self.device)

    def score_pixel_batch(self, pixel_values: torch.Tensor) -> List[float]:
        """(B,3,448,448) bf16 → B 个 0..100 分。

        与官方 score() 同一前向（generate_logits 单次 LM 前向，无自回归），
        softmax 期望也保持官方的 bf16 数值路径（fp32 softmax 会引入 ~0.7 的
        系统性分差，见 A/B），只是把 B 张图拼一个 batch。"""
        self._load()
        bsz = int(pixel_values.shape[0])
        with self._lock, torch.inference_mode():
            out = self.model.generate_logits(
                pixel_values=pixel_values.to(self.device),
                input_ids=self._prompt_ids.expand(bsz, -1),
                attention_mask=self._prompt_mask.expand(bsz, -1),
            )
            logits = out.logits[:, -1, self._pref_ids].detach()
            scores = torch.softmax(logits, -1) @ self._score_weight.to(logits.dtype)
        return [_clamp_0_100(s) for s in scores.tolist()]

    def score_path(self, path: str) -> float:
        self._load()
        pixel_values = _artimuse_image(path, self.device)
        with self._lock, torch.inference_mode():
            score = float(self.model.score(
                self.device, self.tokenizer, pixel_values, dict(self.generation_config)
            ))
        return _clamp_0_100(score)


def _grid_patches(image: torch.Tensor, ps: int) -> Optional[torch.Tensor]:
    """image_to_patches(stride==ps 且 H,W 整除) 的向量化等价：行优先网格 (N,C,ps,ps)。"""
    c, h, w = image.shape
    if h % ps or w % ps:
        return None
    return (image.reshape(c, h // ps, ps, w // ps, ps)
            .permute(1, 3, 0, 2, 4).reshape(-1, c, ps, ps))


def _subdivide_patches(patches: torch.Tensor, sub: int) -> torch.Tensor:
    """(N,C,P,P) → 每个 patch 按行优先切成 (P/sub)² 个 sub 块，与官方逐块
    image_to_patches 的顺序一致 → (N*(P/sub)², C, sub, sub)。"""
    n, c, p, _ = patches.shape
    k = p // sub
    return (patches.reshape(n, c, k, sub, k, sub)
            .permute(0, 2, 4, 1, 3, 5).reshape(-1, c, sub, sub))


def _frequency_importance(patches: torch.Tensor) -> list:
    """calculate_frequency 的向量化等价：整批 fft2 一次算完。

    官方逐 patch 算 20*log10|fftshift(fft2)| 的均值再稳定升序排序；均值对
    fftshift（纯排列）不变，故省去 shift。返回与官方相同的 [(metric, idx)]。"""
    mag = 20.0 * torch.log10(torch.abs(torch.fft.fft2(patches.float())))
    metrics = mag.mean(dim=(1, 2, 3))
    order = torch.argsort(metrics, stable=True)
    return [(metrics[i], i) for i in order.tolist()]


def _binary_mask_fast(n_rows: int, n_cols: int, p: int, selected) -> torch.Tensor:
    """create_binary_mask 的向量化等价：patch 网格散点 + p 倍块扩展。"""
    grid = torch.zeros(n_rows * n_cols, dtype=torch.bool)
    if len(selected):
        grid[torch.as_tensor(sorted(selected), dtype=torch.long)] = True
    grid = grid.reshape(n_rows, n_cols)
    return grid.repeat_interleave(p, dim=0).repeat_interleave(p, dim=1)


def patch_charm_tokenizer_for_py313(charm_tokenizer_cls) -> None:
    """Replace Charm's locals()-mutation block, which is unreliable on Python 3.13.

    2026-07-05 同时做了吞吐向量化：官方实现对 30k 级 patch 逐个 Python 循环做
    切块/FFT/interpolate/建 mask，大图单张预处理数秒~数百秒（随 CPU 争抢恶化）。
    这里把热路径换成整批张量操作；50 张 A/B 实测输出与旧实现**逐位相同**
    （tokens maxΔ=0，pos_embed/mask 全等），2048px 图预处理 1.1-1.4s → 0.08-0.1s。"""

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
        patches_t = _grid_patches(image, patch_sizes[-1])
        if patches_t is None:   # 尺寸不整除（非常规配置）→ 官方慢路径兜底
            image_patches = self.image_to_patches(image, patch_sizes[-1], patch_strides[-1])
            patches_t = torch.stack(image_patches)
        image_patches = patches_t   # (N,3,P,P) 张量支持 len()/下标，后续官方调用兼容
        if self.patch_selection_strategy == "frequency":
            importance = _frequency_importance(patches_t)
        else:
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

        high_scale = self.num_scales - 1
        selected[high_scale] = self.patch_selection(
            self.patch_selection_strategy, importance, n_patches, high_scale, range(len(image_patches))
        )

        # 高分辨率 scale：整批 gather + 网格细分（官方是逐 patch image_to_patches 循环）
        hi_idx = sorted(selected[high_scale])
        hi_tokens = _subdivide_patches(
            patches_t[torch.as_tensor(hi_idx, dtype=torch.long)], self.patch_size)
        masks_hi = [high_scale] * hi_tokens.shape[0]

        remaining_patches = range(len(image_patches))
        intermediate_chunks: list[torch.Tensor] = []
        intermediate_masks: list[int] = []
        selected_intermediate: list[int] = []
        for scale in range(self.num_scales):
            if scale == 0 or scale == high_scale:
                continue
            remaining_patches = list(set(remaining_patches) - set(selected[high_scale]))
            selected[scale] = self.patch_selection(
                self.patch_selection_strategy, importance, n_patches, scale, remaining_patches
            )
            mid_idx = sorted(selected[scale])
            if mid_idx:
                resized = F.interpolate(
                    patches_t[torch.as_tensor(mid_idx, dtype=torch.long)],
                    size=(self.scaled_patchsizes[scale], self.scaled_patchsizes[scale]),
                    mode="bicubic",
                )
                chunk = _subdivide_patches(resized, self.patch_size)
                intermediate_chunks.append(chunk)
                intermediate_masks.extend([scale] * chunk.shape[0])
            selected_intermediate.extend(selected[scale])

        selected_all = set(selected_intermediate) | set(selected[high_scale])
        selected[0] = [x for x in range(0, len(image_patches)) if x not in selected_all]

        # 低分辨率 scale（占 patch 总数 ~98%）：官方逐 patch interpolate 是单张大图
        # 80s+ 的元凶；bicubic 对 batch 维独立，整批一次结果逐位相同。
        low_idx = sorted(selected[0])
        if low_idx:
            resized = F.interpolate(
                patches_t[torch.as_tensor(low_idx, dtype=torch.long)],
                size=(self.scaled_patchsizes[0], self.scaled_patchsizes[0]),
                mode="bicubic",
            )
            low_tokens = _subdivide_patches(resized, self.patch_size)
        else:
            low_tokens = patches_t.new_zeros((0, patches_t.shape[1], self.patch_size, self.patch_size))
        masks_low = [0] * low_tokens.shape[0]

        final_tensor = torch.cat([low_tokens, *intermediate_chunks, hi_tokens], dim=0)
        mask_ms = masks_low + intermediate_masks + masks_hi

        masks = []
        for scale in range(self.num_scales):
            p = patch_sizes[scale] // self.patch_size
            masks.append(_binary_mask_fast(n_patch_per_row, n_patch_per_col, p, selected[scale]))

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
                 training_dataset: str, backbone: str, model_dir: str,
                 max_longedge: int = 0):
        self.checkpoint = Path(checkpoint).expanduser()
        self.model_dir = Path(model_dir).expanduser()
        self.device = device
        self.patch_selection = patch_selection
        self.training_dataset = training_dataset
        self.backbone = backbone
        # >0 时长边超限先降采样再切 patch。官方对 para/spaq 本来就做 1024 上限，
        # ava 不做——但 AVA 训练分布本身是小图，6K px 输入既 OOD 又让 token 数
        # 失控（30k patch → 8k tokens → 前向 2.2s + 11GB 显存尖峰）。
        self.max_longedge = int(max_longedge or 0)
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

    def preprocess_pil(self, image: Image.Image):
        """CPU 重活（frequency patch selection），供线程池预取；不碰 GPU。

        复刻官方 Charm_Tokenizer.preprocess 的 RGB 路径：ava/aadb 等不做 1024
        降采样（只有 para/spaq 有），ToTensor 与 asarray/255 逐位一致。
        """
        self._load()
        if self.training_dataset in ("para", "spaq"):
            w, h = image.size
            if max(w, h) > 1024:
                ratio = 1024.0 / max(w, h)
                image = image.resize((int(w * ratio), int(h * ratio)))
        elif self.max_longedge and max(image.size) > self.max_longedge:
            w, h = image.size
            ratio = self.max_longedge / max(w, h)
            image = image.resize(
                (max(1, int(w * ratio)), max(1, int(h * ratio))),
                Image.Resampling.BICUBIC,
            )
        tensor = torch.from_numpy(
            np.asarray(image, dtype=np.float32) / 255.0
        ).permute(2, 0, 1)
        return self.tokenizer.preprare_patches(tensor)

    def score_pre(self, pre) -> float:
        """GPU 前向 + 归一化（输入为 preprocess_pil/tokenizer.preprocess 的产物）。"""
        tokens, pos_embed, mask_token = pre
        with self._lock, torch.inference_mode():
            prediction = self.scorer.model(
                tokens.unsqueeze(0).to(self.device),
                pos_embed.unsqueeze(0).to(self.device),
                mask_token.unsqueeze(0).to(self.device),
            )
            raw = float(self.scorer.mean_score(prediction)[0])
        return score_to_0_100(raw, self.training_dataset)

    def score_path(self, path: str) -> float:
        # 走 preprocess_pil（含 max_longedge），保证逐张/批量两条路径分数一致
        self._load()
        return self.score_pre(self.preprocess_pil(_decode_rgb(path)))


class OneAlignRunner:
    """Q-Align/OneAlign 美学打分 runner（2026-07-13 起 construct 生产后端）。

    demo100 人工审阅结论：OneAlign 的组内排序最贴人审美（对照 ArtiMuse+Charm/
    AesExpert/HumanAesExpert）。接口与 MixedIAARunner 同形（load/score_path/
    preprocess_path/score_batch_pre），输出 {"iaa_mixed": 0..100, "onealign": 同值}
    —— 沿用 iaa_mixed 键名保持 qa/objscore 下游零改动。
    注意：OneAlign-q 分布整体低于 ArtiMuse 体系（demo100 中位 0.516 vs 0.577），
    TAU_SFT 需配 0.50（tier.py）。
    """

    REPO_DIR = "/home/bc/code/iaa_models/Q-Align"
    MODEL_PATH = "/home/bc/data/models/OneAlign"

    def __init__(self, device: Optional[str] = None):
        self.device = device or config.IAA_DEVICE
        self.scorer = None
        self._lock = threading.Lock()

    def load(self) -> None:
        if self.scorer is not None:
            return
        import transformers
        ver = tuple(int(x) for x in transformers.__version__.split(".")[:2])
        if ver < (4, 37):
            import sys as _sys
            raise RuntimeError(
                f"transformers=={transformers.__version__} 过旧，OneAlign(LLaVA) 起不来；"
                "请用隔离 venv: /home/bc/.venvs/iaa437/bin/python\n"
                f"  [diag] exe={_sys.executable}")
        if self.REPO_DIR not in sys.path:
            sys.path.insert(0, self.REPO_DIR)
        # transformers>=4.37 兼容 patch（同 iaa_benchmark/run_onealign_predictions.py）
        import torch.nn as _nn  # noqa: F401 - q_align import 链需要 torch 先就位
        import transformers.pytorch_utils as pytorch_utils
        if not hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):
            def _fphi(heads, n_heads, head_size, already_pruned_heads):
                heads = set(heads) - already_pruned_heads
                mask = torch.ones(n_heads, head_size)
                for head in heads:
                    head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                    mask[head] = 0
                mask = mask.view(-1).contiguous().eq(1)
                index = torch.arange(len(mask), dtype=torch.long)[mask].long()
                return heads, index
            pytorch_utils.find_pruneable_heads_and_indices = _fphi
        from q_align.evaluate.scorer import QAlignAestheticScorer
        import q_align.model.modeling_llama2 as modeling_llama2
        from transformers.models.llama.modeling_llama import (
            _prepare_4d_causal_attention_mask_for_sdpa)
        if not hasattr(modeling_llama2, "_prepare_4d_causal_attention_mask_for_sdpa"):
            modeling_llama2._prepare_4d_causal_attention_mask_for_sdpa = \
                _prepare_4d_causal_attention_mask_for_sdpa
        self.scorer = QAlignAestheticScorer(
            pretrained=self.MODEL_PATH, device=self.device).eval()

    def _score_pils(self, images: List[Any]) -> List[float]:
        self.load()
        with self._lock, torch.inference_mode():
            raw = self.scorer(images).detach().float().cpu().tolist()
        return [max(0.0, min(100.0, float(s) * 100.0)) for s in raw]

    def score_path(self, path: str) -> Dict[str, Optional[float]]:
        try:
            s = self._score_pils([_decode_rgb(path)])[0]
            return {"iaa_mixed": s, "onealign": s}
        except Exception:  # noqa: BLE001 - 单张失败交上游按 missing 处理
            return {"iaa_mixed": None, "onealign": None}

    def preprocess_path(self, path: str) -> Dict[str, Any]:
        return {"pil": _decode_rgb(path)}

    def score_batch_pre(self, items: List[Dict[str, Any]]) -> List[Dict[str, Optional[float]]]:
        outs: List[Dict[str, Optional[float]]] = [{"iaa_mixed": None, "onealign": None}
                                                  for _ in items]
        idx = [i for i, it in enumerate(items) if it.get("pil") is not None]
        if idx:
            try:
                ss = self._score_pils([items[i]["pil"] for i in idx])
                for i, s in zip(idx, ss):
                    outs[i] = {"iaa_mixed": s, "onealign": s}
            except Exception:  # noqa: BLE001
                pass
        return outs


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
                max_longedge=getattr(config, "IAA_CHARM_MAX_LONGEDGE", 0),
            )
        return self._charm

    def load(self) -> None:
        """急加载两个模型；起不来立刻炸。

        2026-07-05 事故教训: score_path 的静默降级让 ArtiMuse ImportError
        (transformers 4.36 无 Qwen2) 连续 202 张全部落成纯 Charm 分还不报警，
        且每张图重复走一遍失败 import + empty_cache 把吞吐也拖死。"""
        if self.enable_artimuse:
            self.artimuse._load()
        if self.enable_charm:
            self.charm._load()

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

    # ---- batch 流水线（--batch N）：CPU 预取 + ArtiMuse 单前向 batch ----------

    def preprocess_path(self, path: str) -> Dict[str, Any]:
        """线程池里跑的 CPU 部分：一次解码 + 两个模型的预处理。"""
        item: Dict[str, Any] = {}
        image = _decode_rgb(path)
        if self.enable_artimuse:
            item["art_pv"] = _artimuse_tensor_cpu(image)
        if self.enable_charm:
            try:
                item["charm_pre"] = self.charm.preprocess_pil(image)
            except Exception as exc:  # noqa: BLE001 - charm 预处理挂了不拖累 artimuse
                item["charm_error"] = f"{type(exc).__name__}: {str(exc)[:240]}"
        return item

    def score_batch_pre(self, items: List[Dict[str, Any]]) -> List[Dict[str, Optional[float]]]:
        """对 preprocess_path 的产物打分；返回与 score_path 相同结构的 dict 列表。"""
        outs: List[Dict[str, Optional[float]]] = [{} for _ in items]
        if self.enable_artimuse:
            idx = [i for i, it in enumerate(items) if it.get("art_pv") is not None]
            if idx:
                try:
                    scores = self.artimuse.score_pixel_batch(
                        torch.cat([items[i]["art_pv"] for i in idx]))
                    for i, s in zip(idx, scores):
                        outs[i]["artimuse"] = s
                except Exception:  # noqa: BLE001 - batch 挂了逐张兜底，只废掉真正的坏图
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    for i in idx:
                        try:
                            outs[i]["artimuse"] = self.artimuse.score_pixel_batch(
                                items[i]["art_pv"])[0]
                        except Exception as exc:  # noqa: BLE001
                            outs[i]["artimuse"] = None
                            outs[i]["artimuse_error"] = f"{type(exc).__name__}: {str(exc)[:240]}"
        if self.enable_charm:
            for i, it in enumerate(items):
                if it.get("charm_error"):
                    outs[i]["charm"] = None
                    outs[i]["charm_error"] = it["charm_error"]
                    continue
                try:
                    outs[i]["charm"] = self.charm.score_pre(it["charm_pre"])
                except Exception as exc:  # noqa: BLE001
                    outs[i]["charm"] = None
                    outs[i]["charm_error"] = f"{type(exc).__name__}: {str(exc)[:240]}"
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        for o in outs:
            o["iaa_mixed"] = blend_scores(
                o.get("artimuse"), o.get("charm"), self.artimuse_weight, self.charm_weight)
        return outs


_COL = {"artimuse": "artimuse_score", "charm": "charm_score", "iaa_mixed": "iaa_mixed"}


def _as_db_scores(scores: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    return {k: v for k, v in scores.items() if k in _COL}


def run(limit: Optional[int] = None, corpus: Optional[str] = None,
        device: Optional[str] = None, resume: bool = True,
        verdict: Optional[str] = None, order: str = "asc", bq3: bool = False,
        batch: int = 1, workers: int = 6) -> dict:
    # ponytail: 双进程并行用 asc+desc 两端夹击（resume 的 NOT EXISTS 保证不重扫），
    # 比加 shard 参数省事；若需 >2 进程再上真分片

    runner = MixedIAARunner(device=device or config.IAA_DEVICE)
    runner.load()   # 急加载：模型环境坏了必须在写任何分数前炸掉（防纯 Charm 静默污染）
    conn = db.connect()
    run_id = db.start_run(conn, "iaa", {
        "device": device or config.IAA_DEVICE,
        "corpus": corpus,
        "verdict": verdict,
        "artimuse_weight": runner.artimuse_weight,
        "charm_weight": runner.charm_weight,
        "resume": resume,
        "batch": batch,
        "workers": workers,
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
    print(f"[iaa] {len(todo)} images to score with Artimuse+Charm "
          f"(batch={batch} workers={workers})", file=sys.stderr)

    t_start = time.monotonic()
    n_ok = n_err = 0

    def _write_one(aid: str, scores: Dict[str, Optional[float]]) -> None:
        # DB 回填逻辑与旧逐张路径逐字一致（iqa_scores + assets 冗余列 + 事件）
        nonlocal n_ok
        db_scores = _as_db_scores(scores)
        db.add_scores(conn, aid, db_scores, run_id=run_id, model_version=config.IAA_MODEL_VERSION)
        denorm = {_COL[m]: v for m, v in db_scores.items() if v is not None}
        if db_scores.get("iaa_mixed") is not None:
            denorm["aesthetic"] = db_scores["iaa_mixed"]
        denorm["status"] = "iaa_done"
        db.update_asset_fields(conn, aid, **denorm)
        db.log_event(conn, aid, "iaa", "ok", scores, run_id)
        n_ok += 1

    def _progress(done: int, every: int = 50) -> None:
        if done and done % every < max(1, batch):
            conn.commit()
            rate = done / max(1e-9, time.monotonic() - t_start) * 60.0
            print(f"[iaa] {done}/{len(todo)} ok={n_ok} err={n_err} {rate:.1f}/min",
                  file=sys.stderr)

    if batch <= 1:
        # 旧逐张路径（保留；--batch 1）
        for i, r in enumerate(todo):
            aid, path = r["asset_id"], r["path"]
            try:
                scores = runner.score_path(path)
                _write_one(aid, scores)
            except Exception as exc:  # noqa: BLE001
                db.log_event(conn, aid, "iaa", "error", {"err": str(exc)[:300]}, run_id)
                n_err += 1
            if (i + 1) % 50 == 0:
                conn.commit()
                print(f"[iaa] {i+1}/{len(todo)} ok={n_ok} err={n_err}", file=sys.stderr)
    else:
        # batch 流水线：线程池预取 CPU 预处理（解码 + Charm patch selection），
        # 主线程消费并做 GPU 前向（ArtiMuse 整批一次、Charm 逐张小前向），
        # GPU 不再等 IO/CPU。深度 2*batch 保证消费一批时下一批已在路上。
        done = 0
        buf: List[Any] = []

        def _flush() -> None:
            nonlocal done, n_err
            if not buf:
                return
            rows_, items = zip(*buf)
            outs = runner.score_batch_pre(list(items))
            for r, scores in zip(rows_, outs):
                _write_one(r["asset_id"], scores)
            done += len(buf)
            buf.clear()
            _progress(done)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs: deque = deque()
            it = iter(todo)
            depth = batch * 2
            for r in it:
                futs.append((r, ex.submit(runner.preprocess_path, r["path"])))
                if len(futs) >= depth:
                    break
            while futs:
                r, f = futs.popleft()
                nxt = next(it, None)
                if nxt is not None:
                    futs.append((nxt, ex.submit(runner.preprocess_path, nxt["path"])))
                try:
                    buf.append((r, f.result()))
                except Exception as exc:  # noqa: BLE001 - 解码/预处理失败按旧语义记单张 error
                    db.log_event(conn, r["asset_id"], "iaa", "error",
                                 {"err": str(exc)[:300]}, run_id)
                    n_err += 1
                    done += 1
                if len(buf) >= batch:
                    _flush()
            _flush()

    conn.commit()
    elapsed = time.monotonic() - t_start
    rate = (n_ok + n_err) / max(1e-9, elapsed) * 60.0
    print(f"[iaa] finished ok={n_ok} err={n_err} in {elapsed:.0f}s ({rate:.1f}/min)",
          file=sys.stderr)
    db.finish_run(conn, run_id, {"ok": n_ok, "err": n_err,
                                 "elapsed_s": round(elapsed, 1),
                                 "per_min": round(rate, 2)})
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
    ap.add_argument("--batch", type=int, default=1,
                    help="ArtiMuse batch 大小；>1 走预取流水线，1=旧逐张路径")
    ap.add_argument("--workers", type=int, default=6, help="CPU 预处理线程数（batch>1 时生效）")
    args = ap.parse_args()
    print(json.dumps(run(
        limit=args.limit,
        corpus=args.corpus,
        device=args.device,
        resume=not args.no_resume,
        verdict=args.verdict,
        order=args.order,
        bq3=args.bq3,
        batch=args.batch,
        workers=args.workers,
    ), ensure_ascii=False))


if __name__ == "__main__":
    main()
