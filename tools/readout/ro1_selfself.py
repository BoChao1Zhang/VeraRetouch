"""RO-1 · 零训练 self-self 稠密读出（EXPERIMENTS_v3 §2.1 RO-1 / PLAN §2.2 D-0+D-1+D-3）。

三个官方算子跑在**同一份 OpenAI CLIP ViT-B/16 权重、同一份预处理**上（见
`clip_naclip/VENDOR.md`），因此 AUC 可横向比较：

    sclip     = arch 'vanilla' + attn 'csa'       （末层保留残差与 FFN）
    naclip    = arch 'reduced' + attn 'naclip'    （kkᵀ + 高斯邻域偏置，std=5）
    clearclip = arch 'reduced' + attn 'clearclip' （qqᵀ，丢残差丢 FFN）
    vanilla   = arch 'vanilla' + attn 'vanilla'   （原始 CLIP，对照基线）

**红线**：s 场一律不做逐图归一化（min-max / softmax / 分位数）。本模块唯一的输出口径是
patch 特征与文本嵌入的**余弦相似度**（全局固定口径，跨图可比）。归一化方式是臂的属性，
留给 RO-X2 专门 A/B。

后处理阶梯（D-0 伪影三件套 + D-3 guided filter），每一步单独可开关：
  A1 高范数 outlier 剔除 + 4 邻域插值（norm > median + 3·MAD）
  A2 patch 网格周期性伪影诊断（2D 功率谱），阳性时可选陷波
  A3 测试时寄存器（training-free registers, arXiv 2506.08010）
  A4 guided filter 上采样（kornia.filters.guided_blur，复用 tools/scache/upsample.py）
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clip_naclip import clip as clip_pkg                       # noqa: E402
from clip_naclip.imagenet_template import openai_imagenet_template  # noqa: E402

# 官方权重（sha256 见 clip_naclip/VENDOR.md）
DEFAULT_WEIGHTS = "/home/bc/data/models/openai_clip/ViT-B-16.pt"

# OpenAI CLIP 预处理常量（clip_naclip/clip.py::_transform）
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# 算子定义：官方配置逐条核实，出处见 VENDOR.md
OPERATORS: dict[str, dict] = {
    # SCLIP@3608360 clip_segmentor.py:63 `encode_image(img, return_all=True, csa=True)`
    # + clip/model.py:249-251（末层 x = x + custom_attn; x = x + mlp）
    "sclip": {"arch": "vanilla", "attn": "csa", "std": 0.0},
    # NACLIP@0cac3a6 naclip.py:21 默认 arch='reduced', attn_strategy='naclip', gaussian_std=5.
    # README:54 主结果命令 `bash test_all.sh reduced naclip 5 on {gpu} {log}`
    "naclip": {"arch": "reduced", "attn": "naclip", "std": 5.0},
    # ClearCLIP@ad68a40 demo.py（model_type='ClearCLIP', ignore_residual=True）
    # + open_clip/transformer.py:520-528, 614-616
    "clearclip": {"arch": "reduced", "attn": "clearclip", "std": 0.0},
    # 未改造的 CLIP 末层（对照基线，非三算子之一）
    "vanilla": {"arch": "vanilla", "attn": "vanilla", "std": 0.0},
}


# ---------------------------------------------------------------- 模型


def load_clip(weights: str = DEFAULT_WEIGHTS, device: str = "cuda"):
    """加载 OpenAI CLIP ViT-B/16（非 JIT），fp32。"""
    model, _ = clip_pkg.load(weights, device=device, jit=False)
    model = model.float().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def set_operator(model, op: str) -> None:
    cfg = OPERATORS[op]
    model.visual.set_params(cfg["arch"], cfg["attn"], cfg["std"])


# ---------------------------------------------------------------- 文本侧


@torch.no_grad()
def encode_text_bank(model, phrases: Sequence[str], device: str = "cuda",
                     use_templates: bool = True, batch: int = 256) -> torch.Tensor:
    """80 模板集成的文本嵌入（NACLIP@0cac3a6 naclip.py:30-42 的逐字口径）。

    每个短语 → 80 条 openai_imagenet_template → encode_text → L2 归一 → 均值 → 再 L2 归一。
    返回 (n_phrase, dim)，float32。
    """
    out = []
    for ph in phrases:
        prompts = [t(ph) for t in openai_imagenet_template] if use_templates else [ph]
        feats = []
        for i in range(0, len(prompts), batch):
            tok = clip_pkg.tokenize(prompts[i:i + batch], truncate=True).to(device)
            f = model.encode_text(tok).float()
            feats.append(f / f.norm(dim=-1, keepdim=True))
        f = torch.cat(feats, dim=0).mean(dim=0)
        out.append(f / f.norm())
    return torch.stack(out, dim=0)


# ---------------------------------------------------------------- 图像侧


def preprocess_image(img: Image.Image, short_side: int, patch: int = 16,
                     device: str = "cuda") -> tuple[torch.Tensor, tuple[int, int]]:
    """短边缩到 short_side（保持长宽比），两边裁到 patch 的整数倍。

    与 SCLIP/NACLIP 的 mmseg `Resize(scale=(2048, 336), keep_ratio=True)`（ClearCLIP 448）
    同语义：只按短边缩放、不改长宽比、不做 center crop（整图前向，voc21 的 slide_crop=0 档）。
    """
    w, h = img.size
    scale = short_side / min(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    nw, nh = max(patch, nw // patch * patch), max(patch, nh // patch * patch)
    im = img.convert("RGB").resize((nw, nh), Image.Resampling.BICUBIC)
    a = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)
    a = a.permute(2, 0, 1)
    mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
    std = torch.tensor(CLIP_STD).view(3, 1, 1)
    a = (a - mean) / std
    return a.unsqueeze(0).to(device), (nh // patch, nw // patch)


@dataclass
class DenseFeat:
    """一次 ViT 前向的稠密产物。"""
    feat: torch.Tensor           # (Hp, Wp, D) 已 L2 归一的投影后 patch 特征
    hidden_norm: np.ndarray      # (Hp, Wp) **末层之前** hidden 的 L2 范数（D-0 outlier 判据；
                                 #          三算子共享，因为只有最后一个 block 被改造）
    grid: tuple[int, int]
    extra_norm: np.ndarray | None = None   # 追加寄存器 token 的范数（末层之前口径）
    last_norm: np.ndarray | None = None    # (Hp, Wp) ln_post 前（末层之后）hidden 范数


@torch.no_grad()
def dense_features(model, img_t: torch.Tensor, grid: tuple[int, int],
                   extra_tokens: int = 0) -> DenseFeat:
    """整图前向 → 逐 patch 的（L2 归一）CLIP 嵌入。

    口径与三个官方 segmentor 一致：`encode_image(..., return_all=True)` → 丢 CLS →
    `feat /= feat.norm(dim=-1)`（NACLIP naclip.py:58-60 / SCLIP clip_segmentor.py:63-65 /
    ClearCLIP clearclip_segmentor.py:123-124）。
    """
    feats = model.encode_image(img_t, return_all=True, extra_tokens=extra_tokens).float()
    pre = model.visual._prelast_hidden.float()         # (1, L, width) 末层之前
    last = model.visual._last_hidden.float()           # (1, L, width) ln_post 之前
    n = grid[0] * grid[1]
    patch_feat = feats[0, 1:1 + n]
    patch_feat = patch_feat / patch_feat.norm(dim=-1, keepdim=True)
    hid_norm = pre[0, 1:1 + n].norm(dim=-1).cpu().numpy().reshape(grid)
    last_norm = last[0, 1:1 + n].norm(dim=-1).cpu().numpy().reshape(grid)
    extra_norm = None
    if extra_tokens > 0:
        extra_norm = pre[0, 1 + n:].norm(dim=-1).cpu().numpy()
    return DenseFeat(patch_feat.reshape(grid[0], grid[1], -1), hid_norm, grid,
                     extra_norm, last_norm)


def similarity_field(df: DenseFeat, text_emb: torch.Tensor) -> np.ndarray:
    """s = cos(patch_feat, text_emb)。**无任何逐图归一化**（红线）。"""
    s = (df.feat @ text_emb.to(df.feat.device).float())
    return s.detach().cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------- D-0 伪影三件套


def outlier_mask_mad(hidden_norm: np.ndarray, k: float = 3.0) -> np.ndarray:
    """norm > median + k·MAD 的高范数 outlier patch（PLAN §2.2 D-0 第二件）。"""
    v = hidden_norm.ravel()
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    if mad <= 0:
        return np.zeros_like(hidden_norm, dtype=bool)
    return hidden_norm > med + k * mad


def interpolate_outliers(s: np.ndarray, outlier: np.ndarray) -> np.ndarray:
    """把 outlier 位置换成 4 邻域（非 outlier）均值；无可用邻域则退回全图中位数。"""
    if not outlier.any():
        return s
    out = s.copy()
    valid = ~outlier
    pad_s = np.pad(s, 1, mode="edge")
    pad_v = np.pad(valid.astype(np.float32), 1, mode="edge")
    acc = np.zeros_like(s, dtype=np.float64)
    cnt = np.zeros_like(s, dtype=np.float64)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        sl = pad_s[1 + dy:1 + dy + s.shape[0], 1 + dx:1 + dx + s.shape[1]]
        vv = pad_v[1 + dy:1 + dy + s.shape[0], 1 + dx:1 + dx + s.shape[1]]
        acc += sl * vv
        cnt += vv
    fill = np.where(cnt > 0, acc / np.maximum(cnt, 1e-9), float(np.median(s[valid])))
    out[outlier] = fill[outlier]
    return out.astype(np.float32)


def grid_periodicity(s: np.ndarray) -> dict:
    """patch 网格周期性伪影诊断（D-0 第三件的「16px 周期功率谱检查」）。

    s 已在 patch 网格上，原图 16px 周期 = patch 网格上的 **1 patch 周期 = Nyquist**。
    指标 = 最高频（棋盘格）功率占总功率的比例，以及它相对同环带中位功率的倍数。
    """
    a = s - s.mean()
    if a.std() == 0:
        return {"nyquist_power_frac": 0.0, "nyquist_ratio": 1.0}
    P = np.abs(np.fft.fft2(a)) ** 2
    total = float(P.sum()) + 1e-12
    h, w = P.shape
    # 棋盘格分量 = (h//2, w//2)
    nyq = float(P[h // 2, w // 2])
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    r = np.sqrt(fy ** 2 + fx ** 2)
    band = (r > 0.40) & (r <= 0.51)
    med = float(np.median(P[band])) if band.any() else 1.0
    return {"nyquist_power_frac": nyq / total,
            "nyquist_ratio": nyq / (med + 1e-12)}


def notch_nyquist(s: np.ndarray) -> np.ndarray:
    """陷波：置零棋盘格（Nyquist）分量后逆变换。仅在 A2 诊断阳性时使用。"""
    Fm = np.fft.fft2(s)
    h, w = Fm.shape
    Fm[h // 2, w // 2] = 0
    if h % 2 == 0 and w % 2 == 0:
        Fm[h // 2, 0] = 0
        Fm[0, w // 2] = 0
    return np.real(np.fft.ifft2(Fm)).astype(np.float32)


# ---- A3 测试时寄存器（arXiv 2506.08010，官方实现 test-time-registers@860df43）


@dataclass
class RegisterNeurons:
    """{layer: [neuron, ...]}，以及发现过程的元数据。"""
    neurons: dict[int, list[int]]
    meta: dict = field(default_factory=dict)


def _mlp_gelu_modules(model):
    """官方 ClipHookManager.neuron_activation_component = resblocks[l].mlp.gelu。"""
    return [blk.mlp.gelu for blk in model.visual.transformer.resblocks]


@torch.no_grad()
def find_register_neurons(model, images: Iterable[torch.Tensor], grids: Iterable[tuple[int, int]],
                          register_norm_threshold: float = 30.0, top_k: int = 10,
                          highest_layer: int = 5, apply_sparsity_filter: bool = True,
                          sparsity_frac_threshold: float = 0.5,
                          sparsity_activation_threshold: float = 0.5) -> RegisterNeurons:
    """复刻 test-time-registers@860df43 `shared/algorithms.py::find_register_neurons`。

    超参取官方 `configs/openai_clip_base.yaml`（ViT-B-16 / openai）：
    register_norm_threshold=30, top_k=10, highest_layer=5, detect_outliers_layer=-1。
    检测在**未改造**的 CLIP（arch/attn = vanilla）上做。
    """
    set_operator(model, "vanilla")
    gelus = _mlp_gelu_modules(model)
    n_layers = len(gelus)
    acts: list[torch.Tensor] = []
    handles = [m.register_forward_hook(
        lambda mod, inp, out, store=acts: store.append(out.detach())) for m in gelus]
    scores, n_used = None, 0
    try:
        for img_t, grid in zip(images, grids):
            acts.clear()
            model.encode_image(img_t, return_all=True)
            hid = model.visual._last_hidden[0].float()          # (L, width) 末层输出
            norms = hid.norm(dim=-1)
            loc = torch.where(norms > register_norm_threshold)[0]
            if loc.numel() == 0:
                continue
            if scores is None:
                scores = torch.zeros((n_layers, acts[0].shape[-1]), device=hid.device)
            for l in range(n_layers):
                a = acts[l][:, 0, :].float()                    # LND -> (L, n_neurons)
                reg = a[loc].abs()
                if apply_sparsity_filter:
                    sparse = (a < sparsity_activation_threshold).sum(0) >= \
                        sparsity_frac_threshold * a.shape[0]
                    if not sparse.any():
                        continue
                    scores[l] += reg.mean(0) * sparse.float()
                else:
                    scores[l] += reg.mean(0)
            n_used += 1
    finally:
        for h in handles:
            h.remove()
    if scores is None or n_used == 0:
        return RegisterNeurons({}, {"n_images_with_outliers": 0,
                                    "register_norm_threshold": register_norm_threshold})
    scores /= n_used
    n_neu = scores.shape[1]
    flat = scores.flatten()
    order = torch.argsort(flat, descending=True)
    picked: dict[int, list[int]] = {}
    ranked = []
    for idx in order.tolist():
        layer, neuron = idx // n_neu, idx % n_neu
        if layer > highest_layer:
            continue
        ranked.append((layer, neuron, float(flat[idx])))
        if len(ranked) >= top_k:
            break
    for layer, neuron, _ in ranked:
        picked.setdefault(layer, []).append(neuron)
    return RegisterNeurons(picked, {
        "n_images_with_outliers": n_used,
        "register_norm_threshold": register_norm_threshold,
        "top_k": top_k, "highest_layer": highest_layer,
        "apply_sparsity_filter": apply_sparsity_filter,
        "ranked": ranked,
    })


class RegisterIntervention:
    """把 register neuron 的激活搬到追加的寄存器 token 上（官方 `activate_on_registers`）。

    逐字对应 test-time-registers@860df43 `shared/hook_fn.py::activate_on_registers`，
    normal_values='zero'、scale=1.0、num_registers=1（官方 notebook 默认），
    仅索引顺序按本 fork 的 LND 布局改写。
    """

    def __init__(self, model, reg: RegisterNeurons, num_registers: int = 1,
                 scale: float = 1.0, normal_values: str = "zero"):
        self.model, self.reg = model, reg
        self.num_registers, self.scale, self.normal_values = num_registers, scale, normal_values
        self._handles: list = []

    def __enter__(self):
        gelus = _mlp_gelu_modules(self.model)
        for layer, neurons in self.reg.neurons.items():
            idx = torch.tensor(neurons, dtype=torch.long)

            def hook(mod, inp, out, idx=idx):
                nr = self.num_registers
                sel = out[:, 0, :][:, idx]                       # (L, n_sel)  LND, bsz=1
                pos_max, neg_max = sel.max(), sel.min()
                sign_max = pos_max if abs(float(pos_max)) > abs(float(neg_max)) else neg_max
                out[-nr:, 0, :][:, idx] = self.scale * sign_max
                if self.normal_values == "zero":
                    out[1:-nr, 0, :][:, idx] = 0
                return out

            self._handles.append(gelus[layer].register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False


# ---------------------------------------------------------------- 评分口径


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC = 归一化 Mann-Whitney U（与 G1 `analyze_g1.py::roc_auc` 同一实现）。"""
    from scipy.stats import rankdata

    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels).ravel().astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def mask_to_grid(mask: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    """软掩膜 → 目标网格的**面积均值**（与 scache/oracle.py 的 BOX 面积加权同语义）。"""
    t = torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32))[None, None]
    return F.adaptive_avg_pool2d(t, grid)[0, 0].numpy()


def bilinear_to(s: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    t = torch.from_numpy(np.ascontiguousarray(s, dtype=np.float32))[None, None]
    return F.interpolate(t, size=hw, mode="bilinear", align_corners=False)[0, 0].numpy()
