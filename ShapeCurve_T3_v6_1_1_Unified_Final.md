# 课题三 ShapeCurve — v6.1.1 最终统一实施方案

> **版本**: v6.1.1-unified-final
> **日期**: 2026-05-24
> **核心范式**: Hand-coded main controls + frozen semantic residual dictionary + small free low-rank tail + latent Gaussian continuous policy
> **统一原则**: 优先成熟库 (SciPy / pyexiv2 / colour-science / piq / kornia / tensorly / sklearn / einops), 不自写已验证的算法
> **来源整合**: v6.1-final (战略) + GPT-5.5 v6.1.1 §18-29 (实施补全) + Claude Complete Implementation (项目结构 + VLM Hidden + Day-by-Day)

---

## 0. 文档地位

本文档是 ShapeCurve / T3 课题的**唯一活跃实施文档**, 工程师拿这份文档即可从 Day 0 启动到 P7 投稿.

历史版本:
- v1-v5: 已废 (toy / 27-D JSON / FFHQR / free CPD)
- v6: 架构对, 数据口径错, 代码 8 处 bug
- v6.1: 修正数据栈 + 代码 8 处 (战略层完整, 实施细节不足)
- **v6.1.1: 补全 §19-29 实施细节 + 库优先原则 (current)**

---

## 1. 库依赖清单 (优先成熟库, 不重复造轮子)

```python
# requirements.txt
# Core
torch>=2.3.0
torchvision>=0.18.0
einops>=0.8.0                     # 张量操作

# Transformers / VLM
transformers>=4.45.0
peft>=0.12.0
accelerate>=0.34.0

# RL frameworks
trl>=0.10.0                       # P1 custom GRPOTrainer

# Color science (替代自写 ΔE2000 / RGB↔Lab/HSV)
colour-science>=0.4.4
scikit-image>=0.24.0
kornia>=0.7.0                     # GPU-batched color ops + histograms

# Image quality (替代自写 LPIPS / NIQE / MUSIQ)
piq>=0.8.0                        # batch LPIPS, NIQE, MUSIQ, ΔE, PSNR
lpips>=0.1.4                      # fallback

# Numerical / decomposition (替代自写 CPD / sparse coding / B-spline)
scipy>=1.13.0                     # BSpline, gaussian_filter
tensorly>=0.8.1                   # CPD (Phase 2)
scikit-learn>=1.5.0               # SparseCoder, OMP, KMeans
numpy>=1.26.0

# Data / config / logging
pyyaml>=6.0
pyexiv2>=2.12                     # MMArt XMP parsing (代替 regex)
xxhash>=3.5.0                     # fast file hashing (代替 sha1, 10x)
wandb>=0.18.0
hydra-core>=1.3.0

# Visualization
matplotlib>=3.9.0
pillow>=10.4.0
plotly>=5.24.0
```

---

## 2. 最终决策路线

```text
Input image + instruction
    ↓
VLM backbone hidden state at selected layer L (default L=18)
    ↓
64-D latent Gaussian policy z ~ N(μ_z(h), σ_z)
    ↓
FiLM-modulated action decoder
    ↓
Main controls: curve + HSL + WB
    + frozen semantic dictionary residual coefficients (16 atoms)
    + small free CPD/B-spline tail (R_free=4, K=10)
    ↓
Differentiable global retouch renderer (sRGB gamma-encoded)
    ↓
SFT / render loss / continuous-action GRPO / real-pair fine-tuning
```

**最终数据口径**:
- 核心 unique base: FiveK 5,000 + PPR10K 11,161 = 16,161
- 排除主训: FFHQR
- 不算独立 base: MMArt-PPR10K rows / instruction variants
- 必须新增: Tier B off-manifold dense-teacher synthetic

**最终裁决标准** (MVP 必须先证明):
1. continuous policy gradient 有效
2. hidden state 对图像/指令条件有信息
3. hybrid renderer 在 Tier B + Tier C 上接近 D4 dense-LUT controlled upper bound
4. semantic dictionary 能解释真实/非同构 residual

---

## 3. 综合修正表

| # | 修正 | 来源 |
| :- | :- | :- |
| **数据修正 (v6 → v6.1)** | | |
| 1 | FiveK + PPR10K = 16,161 unique base, 排除 FFHQR | v6 → v6.1 |
| 2 | MMArt-PPR10K 仅 instruction/config resource | v6 → v6.1 |
| 3 | Tier B off-manifold dense-teacher 必须新增 | v6 → v6.1 |
| 4 | PPR10K 必须 group-level split | v6 → v6.1 |
| 5 | real pair 不强制 GT action L1 | v6 → v6.1 |
| **代码修正 (v6 → v6.1)** | | |
| 6 | curve endpoint 改 learnable y_min/y_max | v6 → v6.1 |
| 7 | softclip 改 identity-preserving smooth clamp | v6 → v6.1 |
| 8 | residual/main energy 用 delta vs identity | v6 → v6.1 |
| 9 | free-tail gate 改有界 g_max × sigmoid | v6 → v6.1 |
| 10 | dictionary sparsity 用阈值 \|a_j\| < 0.03 | v6 → v6.1 |
| 11 | P3 reward 归一到 1.0 | v6 → v6.1 |
| 12 | 三类标签 gt_action / pseudo_action / no_action_label | v6 → v6.1 |
| 13 | JarvisArt/MMArt 仅 external reference | v6 → v6.1 |
| **实施补全 (v6.1 → v6.1.1)** | | |
| 14 | CPD/free-tail reconstructed-grid loss 公式 | §9 |
| 15 | ZPolicy/FiLMDecoder/Renderer 模块化代码骨架 | §6 |
| 16 | 16 semantic atoms 具体生成器 (uses kornia) | §11 |
| 17 | Tier B 5 family dense-teacher 生成器 | §12 |
| 18 | MMArt hash audit + global-only filter (pyexiv2) | §13 |
| 19 | Reward 12 项 τ 表 + 归一化 (uses piq) | §10 |
| 20 | Layer probe protocol + 显著性阈值 + bootstrap CI | §16 |
| 21 | D4 dense-LUT basis-LUT head 完整 spec | §17 |
| 22 | P0 reward sanity 4-candidate canary test | §18 |
| 23 | Manifest 版本控制 + leakage assert | §19 |
| 24 | Inference workflow + AMP + sRGB 锁定 | §20 |

---

## 4. 五个可证伪科学问题

| Q | 问题 | 评估数据 | 通过 | 否决 |
| :- | :- | :- | :- | :- |
| Q1 | Hybrid renderer ≈ dense LUT upper bound? | **Tier B + Tier C** | T3-full vs D4: PSNR 差 ≤1.0 dB, ΔE2000 差 ≤1.5 | 差 >2.0 dB → pivot 为 explanation layer |
| Q2 | shape/smoothness prior 有用? | Tier A + Tier B | 单调违反 ↓≥30%, PSNR 损失 ≤0.3, banding ↓ | 降级为 hard constraint |
| Q3 | latent Gaussian continuous policy 稳定? | P0/P1 RL runs | entropy 不 collapse, seed PSNR std ≤0.3 dB | 改 DAPO/STD gate |
| Q4 | semantic dictionary 覆盖真实 residual? | Tier B + Tier C | dict explained ratio: Tier B ≥70%, Tier C ≥60% | Phase 2 扩展 |
| Q5 | VLM hidden 携带图像/指令信息? | hidden probe | true h vs shuffled: R²↑0.20, ΔE↓15%, F1↑10pt, 95% CI 不跨 0 | 撤回主张 |

**硬要求**: Q1/Q4 不能只在 Tier A 上裁决 (Tier A 是 self-target 循环).

---

## 5. 项目目录结构

```text
shape_curve_t3/
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── data_tiers.yaml
│   ├── atoms_phase1.yaml
│   ├── archetypes.yaml
│   ├── rewards_p0.yaml      # τ 表见 §10
│   ├── rewards_p3.yaml
│   ├── train_p0.yaml
│   ├── train_p1_sft.yaml
│   ├── train_p1_grpo.yaml
│   └── mvp0_tests.yaml
├── shape_curve/
│   ├── constants.py                # 全局超参集中
│   ├── data/
│   │   ├── audit_mmart.py          # pyexiv2-based XMP audit
│   │   ├── manifest.py             # versioned + leakage assert
│   │   ├── splits.py               # FiveK + PPR10K group split
│   │   ├── color_stats.py          # kornia-based 64-D histograms
│   │   └── ...
│   ├── atoms/
│   │   ├── helpers.py              # luma, hsv_via_kornia, hue_mask, soft_range
│   │   ├── generators.py           # 16 atoms inline
│   │   ├── build_dictionary.py
│   │   └── visualize.py            # atom cards
│   ├── archetypes/
│   │   ├── samplers.py             # 10-12 archetype distributions
│   │   └── generate_tier_a.py
│   ├── tier_b/
│   │   ├── teacher_families.py     # 5 families
│   │   └── generate_tier_b.py
│   ├── renderer/
│   │   ├── bspline.py              # scipy.interpolate.BSpline
│   │   ├── identity_lut.py
│   │   ├── curve_ops.py            # Hermite interp
│   │   ├── hsl_ops.py              # kornia.color
│   │   ├── wb_ops.py
│   │   ├── softclip.py             # identity-preserving
│   │   ├── trilinear.py            # F.grid_sample
│   │   └── render.py
│   ├── model/
│   │   ├── hidden_extractor.py     # <RET_ACTION> token hidden
│   │   ├── z_policy.py
│   │   ├── film_decoder.py
│   │   ├── action_dataclass.py     # ShapeCurveAction
│   │   └── full_model.py
│   ├── train/
│   │   ├── minimal_grpo.py         # P0
│   │   ├── trl_grpo_trainer.py     # P1
│   │   ├── reward.py               # τ-based composite reward (uses piq)
│   │   └── losses.py
│   └── eval/
│       ├── paired.py               # piq-based PSNR/LPIPS/ΔE/NIQE/MUSIQ
│       ├── layer_probe.py
│       ├── intervention.py
│       ├── reward_sanity.py        # 4-candidate canary
│       └── external_baselines.py
├── scripts/
│   ├── day0_mmart_audit.sh
│   ├── day1_build_atoms.sh
│   └── ...
├── tests/
│   ├── test_renderer_gradcheck.py
│   ├── test_atoms_normalization.py
│   ├── test_softclip_identity.py
│   ├── test_grpo_credit_assignment.py
│   ├── test_data_split_no_leak.py
│   └── ...
├── notebooks/
└── reports/
```

---

## 6. 模型架构

### 6.1 全局常量 (`constants.py`)

```python
from typing import Final

G_LUT: Final[int] = 33
M_ATOMS: Final[int] = 16
R_FREE: Final[int] = 4
K_SPLINE: Final[int] = 10
Z_DIM: Final[int] = 64

VLM_NAME: Final[str] = "Gyh68/VeraRetouch"
VLM_HIDDEN_DIM: Final[int] = 1024
VLM_DEFAULT_LAYER: Final[int] = 18
VLM_NUM_LAYERS: Final[int] = 24

BASE_DIM: Final[int] = 512
G_MAX_TAIL: Final[float] = 0.08
HSL_SCALE: Final[float] = 0.5
WB_SCALE: Final[tuple] = (0.3, 0.2)
BLACK_LIFT_RANGE: Final[float] = 0.12
WHITE_DROP_RANGE: Final[float] = 0.12
TAIL_COLOR_SCALE: Final[float] = 0.1

SOFTCLIP_BETA: Final[float] = 20.0
DICT_SPARSITY_THRESHOLD: Final[float] = 0.03

GRPO_GROUP_SIZE: Final[int] = 4
GRPO_EPS_CLIP: Final[float] = 0.2
GRPO_KL_BETA: Final[float] = 0.04
LR_LORA: Final[float] = 1e-5
LR_DECODER: Final[float] = 1e-4
LR_Z_POLICY: Final[float] = 5e-5

LOG_STD_INIT: Final[float] = -2.5
LOG_STD_MIN: Final[float] = -5.0
LOG_STD_MAX: Final[float] = -1.0
SIGMA_FREEZE_STEPS: Final[int] = 500

RHO_DEFAULT = {
    "warm_highlight_color": 0.035, "cool_shadow_color": 0.035,
    "skin_hue_stabilizer": 0.030, "cyan_shadow_twist": 0.045,
    "blue_sky_luma_chroma": 0.035, "foliage_saturation_luma": 0.035,
    "magenta_green_axis": 0.040, "yellow_blue_axis": 0.040,
    "shadow_matte_desat": 0.035, "highlight_rolloff_chroma": 0.030,
    "cross_process_green": 0.055, "cross_process_orange": 0.055,
    "vibrance_low_sat": 0.040, "saturation_compress": 0.040,
    "sepia_tint": 0.045, "cyanotype_tint": 0.045,
}
```

### 6.2 Action dataclass

```python
# model/action_dataclass.py
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class ShapeCurveAction:
    """所有 action 参数. shape 假设 batch dim B."""
    curves: torch.Tensor      # [B, 3, 8]      monotonic 8-knot per channel
    hsl: torch.Tensor         # [B, 8, 3]      8 bands × (h, s, l) shifts
    wb: torch.Tensor          # [B, 2]         Temp, Tint
    dict_coef: torch.Tensor   # [B, M=16]      ∈ [-1, 1]
    tail_gate: torch.Tensor   # [B, R=4]       ∈ [0, g_max]
    tail_color: torch.Tensor  # [B, R, 3]
    tail_alpha: torch.Tensor  # [B, R, K=10]   B-spline coeff for u
    tail_beta: torch.Tensor   # [B, R, K]      B-spline coeff for v
    tail_gamma: torch.Tensor  # [B, R, K]      B-spline coeff for w
    z: Optional[torch.Tensor] = None
    logp_z: Optional[torch.Tensor] = None
```

### 6.3 ZPolicy (Gaussian latent)

```python
# model/z_policy.py
import math
import torch
import torch.nn as nn
from torch.distributions import Normal

from shape_curve.constants import Z_DIM, LOG_STD_INIT, LOG_STD_MIN, LOG_STD_MAX


class ZPolicy(nn.Module):
    def __init__(self, h_dim: int, z_dim: int = Z_DIM,
                 init_log_std: float = LOG_STD_INIT):
        super().__init__()
        self.mu = nn.Sequential(
            nn.LayerNorm(h_dim),
            nn.Linear(h_dim, 512), nn.GELU(),
            nn.Linear(512, z_dim),
        )
        self.log_std = nn.Parameter(torch.full((z_dim,), float(init_log_std)))
        self.z_dim = z_dim

    def std_multiplier(self, step: int, total_steps: int) -> float:
        if total_steps <= 0:
            return 1.0
        x = min(max(step / total_steps, 0.0), 1.0)
        return 0.3 + 0.7 * 0.5 * (1.0 + math.cos(math.pi * x))

    def forward(self, h, step=0, total_steps=1, deterministic=False):
        mu = self.mu(h)
        m = self.std_multiplier(step, total_steps)
        log_std_eff = torch.clamp(self.log_std + math.log(m),
                                  min=LOG_STD_MIN, max=LOG_STD_MAX)
        std = torch.exp(log_std_eff).expand_as(mu)
        if deterministic:
            z = mu
        else:
            z = mu + std * torch.randn_like(mu)
        logp = Normal(mu, std).log_prob(z.detach()).sum(dim=-1)
        return z, logp, mu, std

    def log_prob_at(self, h, z_old, step, total_steps):
        """重新计算当前 policy 在 z_old 上的 log_prob (用于 GRPO update)."""
        mu = self.mu(h)
        m = self.std_multiplier(step, total_steps)
        log_std_eff = torch.clamp(self.log_std + math.log(m),
                                  min=LOG_STD_MIN, max=LOG_STD_MAX)
        std = torch.exp(log_std_eff).expand_as(mu)
        return Normal(mu, std).log_prob(z_old.detach()).sum(dim=-1)
```

### 6.4 FiLM Action Decoder

```python
# model/film_decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from shape_curve.constants import (
    Z_DIM, BASE_DIM, M_ATOMS, R_FREE, K_SPLINE, G_MAX_TAIL,
    BLACK_LIFT_RANGE, WHITE_DROP_RANGE, HSL_SCALE, WB_SCALE, TAIL_COLOR_SCALE,
)
from shape_curve.model.action_dataclass import ShapeCurveAction


class FiLMActionDecoder(nn.Module):
    def __init__(self, h_dim, z_dim=Z_DIM, hidden=BASE_DIM,
                 dict_atoms=M_ATOMS, r_free=R_FREE, k_basis=K_SPLINE,
                 g_max=G_MAX_TAIL):
        super().__init__()
        self.dict_atoms = dict_atoms
        self.r_free = r_free
        self.k_basis = k_basis
        self.g_max = g_max

        self.h_proj = nn.Sequential(
            nn.LayerNorm(h_dim),
            nn.Linear(h_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        # FiLM: γ, β predicted from z; last layer zero-init
        self.z_mod = nn.Sequential(
            nn.LayerNorm(z_dim),
            nn.Linear(z_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 2 * hidden),
        )
        nn.init.zeros_(self.z_mod[-1].weight)
        nn.init.zeros_(self.z_mod[-1].bias)

        # Heads
        self.curve_head = nn.Linear(hidden, 3 * 9)        # 7 Δy + 2 endpoints / channel
        self.hsl_head = nn.Linear(hidden, 24)
        self.wb_head = nn.Linear(hidden, 2)
        self.dict_head = nn.Linear(hidden, dict_atoms)
        per_rank = 1 + 3 + 3 * k_basis
        self.tail_head = nn.Linear(hidden, r_free * per_rank)

        self.register_buffer('wb_scale', torch.tensor(WB_SCALE))

    def _decode_curves(self, raw):
        """raw: [B, 27] → [B, 3, 8] monotonic with learnable endpoints."""
        raw = rearrange(raw, 'b (c k) -> b c k', c=3, k=9)
        raw_delta, raw_black, raw_white = raw[..., :7], raw[..., 7], raw[..., 8]
        dy = F.softplus(raw_delta) + 1e-4
        interior = torch.cumsum(dy, dim=-1)
        interior = interior / (interior[..., -1:] + 1e-6)
        y = torch.cat([torch.zeros_like(interior[..., :1]), interior], dim=-1)
        y_min = torch.sigmoid(raw_black)[..., None] * BLACK_LIFT_RANGE
        y_max = 1.0 - torch.sigmoid(raw_white)[..., None] * WHITE_DROP_RANGE
        return y_min + y * (y_max - y_min)

    def forward(self, h, z) -> ShapeCurveAction:
        base = self.h_proj(h)
        gamma, beta = self.z_mod(z).chunk(2, dim=-1)
        fused = base * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * beta

        curves = self._decode_curves(self.curve_head(fused))
        hsl = torch.tanh(self.hsl_head(fused)).view(-1, 8, 3) * HSL_SCALE
        wb = torch.tanh(self.wb_head(fused)) * self.wb_scale
        dict_coef = torch.tanh(self.dict_head(fused))

        B = fused.shape[0]
        per_rank = 1 + 3 + 3 * self.k_basis
        tail = self.tail_head(fused).view(B, self.r_free, per_rank)
        gate = self.g_max * torch.sigmoid(tail[..., 0])
        color = torch.tanh(tail[..., 1:4]) * TAIL_COLOR_SCALE
        coeff = tail[..., 4:]
        alpha, beta_c, gamma_c = torch.split(coeff, self.k_basis, dim=-1)
        return ShapeCurveAction(curves, hsl, wb, dict_coef,
                                gate, color, alpha, beta_c, gamma_c, z=z)
```

### 6.5 VLM Hidden Extractor

```python
# model/hidden_extractor.py
import torch
import torch.nn as nn
from shape_curve.constants import VLM_DEFAULT_LAYER

SPECIAL_TOKEN = "<RET_ACTION>"


class VLMHiddenExtractor(nn.Module):
    def __init__(self, vlm_model, vlm_processor, layer=VLM_DEFAULT_LAYER):
        super().__init__()
        self.vlm = vlm_model
        self.processor = vlm_processor
        self.layer = layer
        if SPECIAL_TOKEN not in self.processor.tokenizer.get_vocab():
            self.processor.tokenizer.add_special_tokens(
                {"additional_special_tokens": [SPECIAL_TOKEN]}
            )
            self.vlm.resize_token_embeddings(len(self.processor.tokenizer))

    def forward(self, image, instruction):
        prompts = [
            f"<image>\nInstruction: {instr}\nPredict retouch action: {SPECIAL_TOKEN}"
            for instr in instruction
        ]
        inputs = self.processor(images=image, text=prompts,
                                return_tensors="pt", padding=True).to(image.device)
        with torch.no_grad():
            outputs = self.vlm(**inputs, output_hidden_states=True, return_dict=True)

        h_at_layer = outputs.hidden_states[self.layer]
        retouch_id = self.processor.tokenizer.convert_tokens_to_ids(SPECIAL_TOKEN)
        image_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")
        input_ids = inputs['input_ids']

        h_action_token, h_image_pool = [], []
        for b in range(input_ids.shape[0]):
            ret_pos = (input_ids[b] == retouch_id).nonzero(as_tuple=True)[0]
            h_action_token.append(h_at_layer[b, ret_pos[-1]])
            img_pos = (input_ids[b] == image_id).nonzero(as_tuple=True)[0]
            if len(img_pos) > 0:
                h_image_pool.append(h_at_layer[b, img_pos].mean(dim=0))
            else:
                h_image_pool.append(torch.zeros_like(h_action_token[-1]))

        return torch.stack(h_action_token), torch.stack(h_image_pool)
```

### 6.6 Color stats (using kornia)

```python
# data/color_stats.py
import torch
import kornia.color as KC


def compute_color_stats(image: torch.Tensor) -> torch.Tensor:
    """64-D Lab/HSV/luminance histogram (kornia 替代自写).
    image: (B, 3, H, W) sRGB ∈ [0, 1] → (B, 64).
    """
    # Luminance via Rec.709 weights
    lum = KC.rgb_to_grayscale(image,
        rgb_weights=torch.tensor([0.2126, 0.7152, 0.0722]))
    lum_hist = batched_hist(lum.flatten(1), bins=16, range=(0, 1))

    # HSV via kornia
    hsv = KC.rgb_to_hsv(image)
    hue_hist = batched_hist(hsv[:, 0].flatten(1), bins=16, range=(0, 2*3.14159))
    sat_hist = batched_hist(hsv[:, 1].flatten(1), bins=8, range=(0, 1))

    # Lab via kornia
    lab = KC.rgb_to_lab(image)
    a_hist = batched_hist(((lab[:, 1] + 128) / 256).flatten(1), bins=8, range=(0, 1))
    b_hist = batched_hist(((lab[:, 2] + 128) / 256).flatten(1), bins=8, range=(0, 1))

    feats = torch.cat([lum_hist, hue_hist, sat_hist, a_hist, b_hist], dim=-1)
    return torch.nn.functional.pad(feats, (0, 64 - feats.shape[-1]))


def batched_hist(x, bins, range):
    """Soft batched histogram."""
    centers = torch.linspace(range[0], range[1], bins, device=x.device)
    bandwidth = (range[1] - range[0]) / bins
    diff = x.unsqueeze(-1) - centers
    weights = torch.exp(-diff.pow(2) / (bandwidth ** 2 / 4))
    hist = weights.sum(dim=1)
    return hist / hist.sum(dim=-1, keepdim=True).clamp(min=1e-6)
```

---

## 7. 渲染器实现

### 7.1 B-spline basis (使用 SciPy)

```python
# renderer/bspline.py
import numpy as np
import torch
from scipy.interpolate import BSpline


_CACHE = {}


def cubic_bspline_basis(N: int = 33, K: int = 10) -> torch.Tensor:
    """scipy.interpolate.BSpline 提供正确的 cubic B-spline basis."""
    key = (N, K)
    if key in _CACHE:
        return _CACHE[key]

    n_interior = max(K - 4, 0)
    interior_knots = np.linspace(0, 1, n_interior + 2)[1:-1] if n_interior > 0 else []
    knots = np.concatenate([np.zeros(4), interior_knots, np.ones(4)])
    x = np.linspace(0, 1, N)
    B_np = np.zeros((N, K), dtype=np.float32)
    for k in range(K):
        c = np.zeros(K + 4)
        c[k] = 1.0
        spline = BSpline(knots, c, 3)
        B_np[:, k] = spline(x)
    B = torch.from_numpy(B_np)
    _CACHE[key] = B
    return B
```

### 7.2 Identity LUT

```python
# renderer/identity_lut.py
import torch


def build_identity_lut(G: int = 33) -> torch.Tensor:
    """Identity 3D LUT: output_c = input_c. Shape: (G, G, G, 3)."""
    coords = torch.linspace(0, 1, G)
    r, g, b = torch.meshgrid(coords, coords, coords, indexing='ij')
    return torch.stack([r, g, b], dim=-1)
```

### 7.3 Identity-preserving softclip

```python
# renderer/softclip.py
import torch
import torch.nn.functional as F


def softclip_identity(x: torch.Tensor, beta: float = 20.0) -> torch.Tensor:
    """近似 identity 的 smooth clamp to [0, 1].
    For x ∈ [0, 1]: output ≈ x (1e-9 内)
    For x < 0:      asymptote to 0
    For x > 1:      asymptote to 1
    """
    return x + F.softplus(-beta * x) / beta - F.softplus(beta * (x - 1.0)) / beta


def gamut_penalty(lut_pre: torch.Tensor) -> torch.Tensor:
    over = F.relu(lut_pre - 1.0).pow(2).mean()
    under = F.relu(-lut_pre).pow(2).mean()
    return over + under
```

### 7.4 HSL operations (使用 kornia)

```python
# renderer/hsl_ops.py
import torch
import kornia.color as KC


HUE_CENTERS_DEG = torch.tensor([0., 30., 60., 120., 180., 220., 280., 320.])


def apply_hsl_to_lut(lut: torch.Tensor, hsl_24: torch.Tensor) -> torch.Tensor:
    """lut: (B, G, G, G, 3) sRGB → 修改后的 LUT.
    hsl_24: (B, 8, 3) 每个 band (Δh, Δs, Δl).

    使用 kornia.color.rgb_to_hsv (GPU-batched, 替代自写).
    """
    B = lut.shape[0]
    G = lut.shape[1]
    rgb_flat = lut.reshape(B, -1, 3).permute(0, 2, 1).unsqueeze(-1)
    hsv = KC.rgb_to_hsv(rgb_flat)
    h, s, v = hsv[:, 0].squeeze(-1), hsv[:, 1].squeeze(-1), hsv[:, 2].squeeze(-1)
    h_deg = h * 360.0 / (2 * 3.14159)

    band_weights = torch.zeros(B, h_deg.shape[1], 8, device=lut.device)
    for b_idx, center in enumerate(HUE_CENTERS_DEG):
        circ_dist = torch.min((h_deg - center).abs(),
                              360 - (h_deg - center).abs())
        band_weights[:, :, b_idx] = torch.exp(-circ_dist.pow(2) / (2 * 25 ** 2))
    band_weights = band_weights / band_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)

    delta_h = (band_weights * hsl_24[:, :, 0].unsqueeze(1)).sum(-1)
    delta_s = (band_weights * hsl_24[:, :, 1].unsqueeze(1)).sum(-1)
    delta_v = (band_weights * hsl_24[:, :, 2].unsqueeze(1)).sum(-1)

    h_new = (h + delta_h * (2 * 3.14159 / 360)) % (2 * 3.14159)
    s_new = (s + delta_s).clamp(0, 1)
    v_new = (v + delta_v).clamp(0, 1)
    hsv_new = torch.stack([h_new, s_new, v_new], dim=1).unsqueeze(-1)
    rgb_new = KC.hsv_to_rgb(hsv_new).squeeze(-1).permute(0, 2, 1)
    return rgb_new.reshape(B, G, G, G, 3)
```

### 7.5 Trilinear apply (using `F.grid_sample`)

```python
# renderer/trilinear.py
import torch
import torch.nn.functional as F


def trilinear_apply(lut: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    """lut: (B, 3, G, G, G), image: (B, 3, H, W) ∈ [0, 1] → (B, 3, H, W)."""
    coords = image.permute(0, 2, 3, 1) * 2 - 1            # → [-1, 1]
    coords_3d = coords.unsqueeze(1)
    coords_reordered = coords_3d[..., [2, 1, 0]]
    out = F.grid_sample(lut, coords_reordered, mode='bilinear',
                        padding_mode='border', align_corners=True)
    return out.squeeze(2)
```

### 7.6 Main render

```python
# renderer/render.py
import torch
from einops import einsum

from shape_curve.renderer.bspline import cubic_bspline_basis
from shape_curve.renderer.identity_lut import build_identity_lut
from shape_curve.renderer.softclip import softclip_identity, gamut_penalty
from shape_curve.renderer.trilinear import trilinear_apply
from shape_curve.constants import G_LUT, K_SPLINE


def hybrid_render(input_image, action, dictionary_atoms, rho_per_atom, training=True):
    """v6.1.1 hybrid renderer.
    action: ShapeCurveAction dataclass
    dictionary_atoms: (M=16, G, G, G, 3) frozen
    rho_per_atom: (M,) frozen
    """
    G = G_LUT
    device = input_image.device
    B_spline = cubic_bspline_basis(G, K_SPLINE).to(device)
    identity = build_identity_lut(G).to(device)

    # 1. Main LUT (curve → HSL → WB)
    lut_main = build_curve_lut(action.curves)
    lut_main = apply_hsl_to_lut(lut_main, action.hsl)
    lut_main = apply_wb_to_lut(lut_main, action.wb)

    # 2. Dictionary LUT
    scaled_coef = action.dict_coef * rho_per_atom[None, :]
    lut_dict = einsum(scaled_coef, dictionary_atoms,
                      'b m, m g h k c -> b g h k c')

    # 3. Free tail (B-spline factors)
    u = einsum(B_spline, action.tail_alpha, 'g k, b r k -> b r g')
    v = einsum(B_spline, action.tail_beta, 'g k, b r k -> b r g')
    w = einsum(B_spline, action.tail_gamma, 'g k, b r k -> b r g')
    u_n = u / (u.norm(dim=-1, keepdim=True) + 1e-6)
    v_n = v / (v.norm(dim=-1, keepdim=True) + 1e-6)
    w_n = w / (w.norm(dim=-1, keepdim=True) + 1e-6)
    lut_tail = einsum(action.tail_gate, action.tail_color, u_n, v_n, w_n,
                      'b r, b r c, b r i, b r j, b r k -> b i j k c')

    # 4. Compose + soft gamut
    lut_pre = lut_main + lut_dict + lut_tail
    r_gamut = gamut_penalty(lut_pre)

    if training:
        lut_final = softclip_identity(lut_pre)
    else:
        lut_final = torch.clamp(lut_pre, 0, 1)

    # Reshape: (B, G, G, G, 3) → (B, 3, G, G, G)
    lut_final_chw = lut_final.permute(0, 4, 1, 2, 3)
    rendered = trilinear_apply(lut_final_chw, input_image)

    # v6.1.1 energy calc (delta vs identity)
    main_delta = lut_main - identity[None]
    residual_delta = lut_dict + lut_tail
    main_energy = main_delta.pow(2).mean(dim=(1, 2, 3, 4)).sqrt()
    residual_energy = residual_delta.pow(2).mean(dim=(1, 2, 3, 4)).sqrt()
    residual_ratio = residual_energy / (main_energy + residual_energy + 1e-6)

    return rendered, dict(
        r_gamut=r_gamut,
        main_energy=main_energy.mean(),
        residual_energy=residual_energy.mean(),
        residual_ratio=residual_ratio.mean(),
        clipping_ratio=((lut_pre > 1.0) | (lut_pre < 0)).float().mean(),
        dict_sparsity=(action.dict_coef.abs() < 0.03).float().mean(),
        free_tail_gate_p95=action.tail_gate.flatten().quantile(0.95),
        lut_main=lut_main, lut_dict=lut_dict, lut_tail=lut_tail, lut_final=lut_final,
    )
```

---

## 8. 数据栈

### 8.1 五层 tier 回顾

| Tier | 数量 (P1) | label_type | 用途 |
| :- | :- | :- | :- |
| A renderer-aligned synthetic | 80-120K | `gt_action` | SFT + sanity |
| **B off-manifold dense-teacher** | 20-50K | `teacher_lut` | **Q1/Q4 公平测** |
| C real paired retouch | ~58K | `no_action_label` 或 `pseudo_action` | 真实闭环 |
| D MMArt-PPR10K resource | ~20-50K (post audit) | instruction + 部分 pair | instruction 多样性 |
| E external eval | ~1.5K | held-out | 主表 |

### 8.2 License manifest

| 数据 | Code License | Dataset Agreement |
| :- | :- | :- |
| MIT-Adobe FiveK | -- | **Research-only** |
| PPR10K | Apache-2.0 (code) | **Non-commercial research** |
| MMArt-PPR10K | Apache-2.0 | + 继承 PPR10K 图像权利 |
| FFHQR | -- | CC BY-NC-SA 4.0 (**排除主训**) |
| MMArt-Bench | Apache-2.0 | -- |

### 8.3 Splits

**FiveK**: 4000 train / 500 val / 500 test (by image_id stable hash)
**PPR10K**: 1356 train groups / 165 val groups / 165 test groups (group-level split, MUST assert no leakage)
**MMArt-PPR10K**: inherit PPR10K split by hash; standalone samples use hash-based split

---

## 9. Loss 设计 (含 CPD/free-tail reconstructed-grid)

### 9.1 三类标签 + 可用 loss

| label_type | 来源 | 可用监督 | 禁止 |
| :- | :- | :- | :- |
| `gt_action` | Tier A | action loss + render loss + grid loss | 无 |
| `teacher_lut` | Tier B | dense/grid render loss; 可做 inverse fitting pseudo-action | raw factor L1 |
| `no_action_label` | Tier C | render / perceptual / color loss | action L1 |
| `pseudo_action` | Tier C/D offline-fitted | 低权重 pseudo-action loss + render loss | 当作 GT |

### 9.2 明确禁止 raw CPD factor L1

free-tail components 有 permutation / scaling / sign ambiguity. 主损失必须在 **reconstructed grid** 上:

```text
不要: ||α_pred - α_gt||₁ + ||β_pred - β_gt||₁ + ...
采用: ||reconstruct_tail(pred) - reconstruct_tail(gt)||₁
```

### 9.3 Tier A 完整 SFT loss

```python
loss_weights_tier_a = {
    "main_action": 1.00,
    "dict_coef":   0.50,
    "tail_grid":   0.30,    # reconstructed grid L1, NOT factor L1
    "render":      1.00,
    "color":       0.50,
    "smooth":      0.05,
    "range":       0.10,
}
```

前 500 step gradient-norm audit: 如 tail_grid 对 decoder grad norm > main_action × 3, 降权或冻结 tail head warmup main/dict.

### 9.4 Tier B/C main_action_fitted 明确含义

```text
LUT_main_fitted = build_main_lut(curve, hsl, wb)  # identity → curve → HSL → WB
residual_gt = LUT_teacher_or_fitted_target - LUT_main_fitted
```

Tier C 若只有 image pair, 先离线 inverse fitting → `pseudo_action` (低权重).

### 9.5 Loss 代码 (uses piq + scipy)

```python
# train/losses.py
import torch
import torch.nn.functional as F
import piq


def charbonnier(x, eps=1e-3):
    return torch.sqrt(x * x + eps * eps).mean()


def render_loss(pred, target):
    """piq.LPIPS 替代自写 perceptual."""
    char = charbonnier(pred - target)
    lpips_val = piq.LPIPS()(pred.clamp(0, 1), target.clamp(0, 1))
    return char + 0.5 * lpips_val


def color_loss(pred, target):
    """ΔE2000 via piq."""
    delta_e = piq.delta_e(pred.clamp(0, 1), target.clamp(0, 1), method="2000")
    return delta_e.mean()


def tv3d_lut(lut):
    return (
        (lut[:, 1:] - lut[:, :-1]).abs().mean() +
        (lut[:, :, 1:] - lut[:, :, :-1]).abs().mean() +
        (lut[:, :, :, 1:] - lut[:, :, :, :-1]).abs().mean()
    )


def smoothness2_3d(lut):
    """Second-order differences (banding penalty)."""
    terms = []
    for dim in [1, 2, 3]:
        n = lut.shape[dim] - 2
        a = lut.narrow(dim, 2, n)
        b = lut.narrow(dim, 1, n)
        c = lut.narrow(dim, 0, n)
        terms.append((a - 2 * b + c).abs().mean())
    return sum(terms)


def tail_grid_loss(tail_pred_grid, tail_gt_grid):
    return F.smooth_l1_loss(tail_pred_grid, tail_gt_grid, beta=0.01)
```

---

## 10. Reward 配方 + τ 表

### 10.1 P0/P1 paired reward

```text
R_total = 0.45·R_ref + 0.20·R_hist_content + 0.25·R_constraints + 0.10·R_action_prior
```

### 10.2 12 项 τ 初值表

| 项 | 记号 | τ 初值 | 说明 |
| :- | :- | :-: | :- |
| Charbonnier RGB L1 | d_l1 | 0.030 | image [0,1] |
| ΔE2000 mean (`piq.delta_e`) | d_de | 6.0 | initial 不太尖锐 |
| ΔE2000 p95 | d_de95 | 12.0 | stress |
| LPIPS (`piq.LPIPS`) | d_lpips | 0.150 | perceptual |
| luminance hist L1 | d_lhist | 0.120 | 64 bins, kornia |
| hue/sat hist L1 | d_hhist | 0.150 | HSV 2D hist |
| edge consistency | d_edge | 0.080 | Sobel L1 |
| gamut penalty | p_gamut | 0.005 | LUT_pre 越界 |
| clipping ratio | p_clip | 0.010 | >1% 即惩罚 |
| banding score | p_band | 0.020 | 二阶差分 |
| residual energy | p_res | 0.050 | RMS |
| free-tail gate | p_gate | 0.030 | mean gate |

### 10.3 Reward 代码 (uses piq)

```python
# train/reward.py
import torch
import piq


def exp_reward(distance, tau):
    return torch.exp(-distance / tau)


def neg_penalty(penalty, tau, max_abs=2.0):
    return -torch.clamp(penalty / tau, 0.0, max_abs)


def compute_p0_reward(metrics):
    R_ref = (
        0.35 * exp_reward(metrics['l1'], 0.030) +
        0.35 * exp_reward(metrics['delta_e'], 6.0) +
        0.30 * exp_reward(metrics['lpips'], 0.150)
    )
    R_hist_content = (
        0.40 * exp_reward(metrics['lum_hist'], 0.120) +
        0.35 * exp_reward(metrics['hsv_hist'], 0.150) +
        0.25 * exp_reward(metrics['edge'], 0.080)
    )
    R_constraints = (
        0.30 * neg_penalty(metrics['gamut'], 0.005) +
        0.25 * neg_penalty(metrics['clip_ratio'], 0.010) +
        0.25 * neg_penalty(metrics['banding'], 0.020) +
        0.20 * neg_penalty(metrics['curve_violation'], 0.001)
    )
    R_action_prior = (
        0.40 * neg_penalty(metrics['free_tail_gate'], 0.030) +
        0.30 * neg_penalty(metrics['residual_energy'], 0.050) +
        0.30 * metrics['residual_efficiency']
    )
    return 0.45 * R_ref + 0.20 * R_hist_content + 0.25 * R_constraints + 0.10 * R_action_prior
```

### 10.4 P3 instruction reward (sum=1.0)

```text
0.30·R_ref + 0.15·R_hist + 0.10·R_no_ref (Q-Align via piq) +
0.15·R_constraints + 0.10·R_action_prior + 0.20·R_edit_reward (EditScore Qwen3-VL-4B)
= 1.00 ✓
```

### 10.5 Residual efficiency

```text
R_residual_efficiency = (Q_full - Q_main_only) - λ × residual_energy

residual 显著提升质量 → 净 reward 高
residual 无效 → 净 reward 负
```

---

## 11. 16 Semantic Atoms 生成器

### 11.1 Helpers (`atoms/helpers.py`)

```python
import torch
import torch.nn.functional as F
import kornia.color as KC


def rgb_grid(G=33, device='cpu'):
    x = torch.linspace(0, 1, G, device=device)
    r, g, b = torch.meshgrid(x, x, x, indexing='ij')
    return torch.stack([r, g, b], dim=-1)


def luma(rgb):
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def hsv_via_kornia(rgb):
    """kornia 替代自写 HSV."""
    rgb_chw = rgb.permute(3, 0, 1, 2).unsqueeze(0)
    hsv_chw = KC.rgb_to_hsv(rgb_chw).squeeze(0)
    hsv = hsv_chw.permute(1, 2, 3, 0)
    return hsv[..., 0] / (2 * 3.14159), hsv[..., 1], hsv[..., 2]


def soft_range(x, lo, hi, sharp=40.0):
    return torch.sigmoid(sharp * (x - lo)) * torch.sigmoid(sharp * (hi - x))


def hue_mask(h, center_deg, width_deg=30.0):
    c = center_deg / 360.0
    w = width_deg / 360.0
    d = torch.minimum((h - c).abs(), 1.0 - (h - c).abs())
    return torch.exp(-0.5 * (d / (w / 2.355 + 1e-6)) ** 2)


def smooth3d(field, passes=2):
    """3D Gaussian smoothing via conv3d."""
    x = field.permute(3, 0, 1, 2).unsqueeze(0)
    kernel = torch.tensor([1., 2., 1.], device=field.device, dtype=field.dtype)
    kernel = kernel / kernel.sum()
    for _ in range(passes):
        for dim in (2, 3, 4):
            shape = [1, 1, 1, 1, 1]; shape[dim] = 3
            k = kernel.view(shape).repeat(3, 1, 1, 1, 1).to(field.device)
            pad = [0, 0, 0, 0, 0, 0]
            pad[2 * (4 - dim)] = 1; pad[2 * (4 - dim) + 1] = 1
            x = F.conv3d(F.pad(x, pad, mode='replicate'), k, groups=3)
    return x.squeeze(0).permute(1, 2, 3, 0)


def normalize_atom(D, eps=1e-6):
    """p95-amplitude normalization → D_unit."""
    D = D - D.mean(dim=(0, 1, 2), keepdim=True) * 0.25
    D = smooth3d(D, passes=2)
    amp = torch.quantile(D.abs().amax(dim=-1).flatten(), 0.95)
    return D / (amp + eps)
```

### 11.2 16 Atoms 定义 (`atoms/generators.py`)

```python
import torch
from shape_curve.atoms.helpers import (
    rgb_grid, luma, hsv_via_kornia, soft_range, hue_mask, normalize_atom
)
from shape_curve.constants import RHO_DEFAULT, G_LUT


ATOM_SPECS = [
    ('warm_highlight_color', 0.035),
    ('cool_shadow_color', 0.035),
    ('skin_hue_stabilizer', 0.030),
    ('cyan_shadow_twist', 0.045),
    ('blue_sky_luma_chroma', 0.035),
    ('foliage_saturation_luma', 0.035),
    ('magenta_green_axis', 0.040),
    ('yellow_blue_axis', 0.040),
    ('shadow_matte_desat', 0.035),
    ('highlight_rolloff_chroma', 0.030),
    ('cross_process_green', 0.055),
    ('cross_process_orange', 0.055),
    ('vibrance_low_sat', 0.040),
    ('saturation_compress', 0.040),
    ('sepia_tint', 0.045),
    ('cyanotype_tint', 0.045),
]


def make_atom(name: str, G: int = G_LUT, device='cpu') -> torch.Tensor:
    rgb = rgb_grid(G, device)
    Y = luma(rgb)
    h, s, v = hsv_via_kornia(rgb)
    gray = Y[..., None].expand_as(rgb)

    if name == 'warm_highlight_color':
        m = soft_range(Y, 0.55, 1.00) * (0.5 + 0.5 * s)
        D = m[..., None] * torch.tensor([1.0, 0.35, -0.45], device=device)
    elif name == 'cool_shadow_color':
        m = soft_range(Y, 0.00, 0.45)
        D = m[..., None] * torch.tensor([-0.55, -0.10, 1.0], device=device)
    elif name == 'skin_hue_stabilizer':
        m = (hue_mask(h, 28, 32) * soft_range(s, 0.12, 0.85)
             * soft_range(Y, 0.25, 0.90))
        target = torch.tensor([1.0, 0.56, 0.38], device=device)
        target = target / target.norm()
        current = rgb / (rgb.norm(dim=-1, keepdim=True) + 1e-6)
        D = m[..., None] * (target - current) * 0.8
    elif name == 'cyan_shadow_twist':
        m = soft_range(Y, 0.00, 0.50) * hue_mask(h, 200, 70)
        D = m[..., None] * torch.tensor([-0.35, 0.45, 0.65], device=device)
    elif name == 'blue_sky_luma_chroma':
        m = (hue_mask(h, 210, 45) * soft_range(s, 0.15, 0.95)
             * soft_range(Y, 0.35, 1.0))
        D = m[..., None] * torch.tensor([0.05, 0.10, 0.55], device=device)
    elif name == 'foliage_saturation_luma':
        m = (hue_mask(h, 105, 50) * soft_range(s, 0.12, 0.9)
             * soft_range(Y, 0.20, 0.85))
        D = m[..., None] * torch.tensor([-0.10, 0.45, -0.12], device=device)
    elif name == 'magenta_green_axis':
        m = soft_range(s, 0.05, 1.0)
        D = m[..., None] * torch.tensor([0.65, -0.75, 0.65], device=device)
    elif name == 'yellow_blue_axis':
        m = soft_range(s, 0.05, 1.0)
        D = m[..., None] * torch.tensor([0.55, 0.45, -0.75], device=device)
    elif name == 'shadow_matte_desat':
        m = soft_range(Y, 0.00, 0.35)
        D = (m[..., None] * (gray - rgb) * 1.3
             + m[..., None] * torch.tensor([0.10, 0.08, 0.06], device=device))
    elif name == 'highlight_rolloff_chroma':
        m = soft_range(Y, 0.70, 1.00)
        D = (m[..., None] * (gray - rgb) * 0.8
             + m[..., None] * torch.tensor([-0.05, -0.04, -0.02], device=device))
    elif name == 'cross_process_green':
        m = soft_range(Y, 0.15, 0.90) * (0.4 + 0.6 * s)
        D = m[..., None] * torch.tensor([-0.25, 0.65, -0.10], device=device)
    elif name == 'cross_process_orange':
        m = soft_range(Y, 0.20, 1.00) * (0.4 + 0.6 * s)
        D = m[..., None] * torch.tensor([0.65, 0.28, -0.35], device=device)
    elif name == 'vibrance_low_sat':
        m = soft_range(s, 0.00, 0.35) * soft_range(Y, 0.15, 0.95)
        D = m[..., None] * (rgb - gray) * 1.8
    elif name == 'saturation_compress':
        m = soft_range(s, 0.60, 1.00)
        D = m[..., None] * (gray - rgb) * 1.2
    elif name == 'sepia_tint':
        m = soft_range(Y, 0.05, 0.95)
        sepia = torch.stack([Y * 1.10, Y * 0.88, Y * 0.55], dim=-1).clamp(0, 1)
        D = m[..., None] * (sepia - rgb)
    elif name == 'cyanotype_tint':
        m = soft_range(Y, 0.05, 0.95)
        cyan = torch.stack([Y * 0.45, Y * 0.80, Y * 1.10], dim=-1).clamp(0, 1)
        D = m[..., None] * (cyan - rgb)
    else:
        raise ValueError(f'unknown atom: {name}')

    return normalize_atom(D)


def build_phase1_dictionary(G=33, device='cpu'):
    atoms, rho, names = [], [], []
    for name, _ in ATOM_SPECS:
        atoms.append(make_atom(name, G, device))
        rho.append(RHO_DEFAULT[name])
        names.append(name)
    dictionary = torch.stack(atoms, dim=0)
    rho = torch.tensor(rho, dtype=dictionary.dtype, device=device)
    return dictionary, rho, names
```

### 11.3 Atom visualization cards

每个 atom 必须输出 card 含: ColorChecker before/after, grayscale ramp, hue wheel, 2-4 自然图 +1.0 / -1.0 对比, p95 amplitude / mean residual / smoothness score.

---

## 12. Tier B Dense-Teacher 生成器

### 12.1 5 family + 采样比例

```yaml
tier_b_family_mix:
  smooth_random:       0.30
  matrix_plus_smooth:  0.20
  hue_twist:           0.20
  color_isolation:     0.15
  sat_compress:        0.15
```

### 12.2 实现 (`tier_b/teacher_families.py`)

```python
import torch
import torch.nn.functional as F
from shape_curve.atoms.helpers import (
    rgb_grid, luma, hsv_via_kornia, hue_mask, soft_range
)


def identity_lut(G=33, device='cpu'):
    return rgb_grid(G, device)


def smooth_random_residual(G=33, control=5, amp=0.06, device='cpu', seed=None):
    """低分辨率噪声 + trilinear upsample."""
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(1, 3, control, control, control,
                            device=device, generator=gen)
    else:
        noise = torch.randn(1, 3, control, control, control, device=device)
    noise = noise - noise.mean(dim=(2, 3, 4), keepdim=True)
    up = F.interpolate(noise, size=(G, G, G), mode='trilinear', align_corners=True)
    up = up[0].permute(1, 2, 3, 0)

    rgb = identity_lut(G, device)
    boundary = torch.minimum(rgb, 1 - rgb).amin(dim=-1, keepdim=True)
    boundary_weight = (0.25 + boundary).clamp(0.15, 1.0)
    return amp * boundary_weight * up / (up.abs().amax() + 1e-6)


def random_color_matrix(device='cpu', seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return torch.eye(3, device=device) + 0.08 * torch.randn(3, 3, device=device)


def apply_matrix_lut(lut, A):
    return torch.einsum('...c,dc->...d', lut, A).clamp(-0.2, 1.2)


def generate_dense_teacher_lut(family='smooth_random', seed=None, G=33, device='cpu'):
    lut = identity_lut(G, device)

    if family == 'smooth_random':
        lut = lut + smooth_random_residual(G, 5, 0.06, device, seed)
    elif family == 'matrix_plus_smooth':
        lut = apply_matrix_lut(lut, random_color_matrix(device, seed))
        lut = lut + smooth_random_residual(G, 4, 0.035, device, seed)
    elif family == 'hue_twist':
        h, s, v = hsv_via_kornia(lut)
        twist = hue_mask(h, 190, 70) * soft_range(s, 0.2, 1.0)
        lut = lut + twist[..., None] * torch.tensor([-0.06, 0.05, 0.08], device=device)
    elif family == 'color_isolation':
        h, s, v = hsv_via_kornia(lut)
        keep = hue_mask(h, 25, 35)
        gray = luma(lut)[..., None].expand_as(lut)
        lut = lut * keep[..., None] + (0.65 * gray + 0.35 * lut) * (1 - keep[..., None])
    elif family == 'sat_compress':
        Y = luma(lut)[..., None]
        h, s, v = hsv_via_kornia(lut)
        gray = Y.expand_as(lut)
        m = soft_range(s, 0.55, 1.0)
        lut = lut + m[..., None] * (gray - lut) * 0.25
    else:
        raise ValueError(family)

    return lut.clamp(0, 1)
```

### 12.3 Holdout 严格规则

```text
train 中 color_isolation 仅保留 {red, orange, cyan}
holdout 使用 {foliage_green, blue, magenta}

train 中 smooth_random / matrix_plus_smooth 在 ±0.08 magnitude
holdout 使用 ±0.12 stress magnitude
```

---

## 13. MMArt Hash Audit (使用 pyexiv2)

```python
# data/audit_mmart.py
from pathlib import Path
import json
import xxhash
import pyexiv2
from tqdm import tqdm


def xxhash_file(path: Path) -> str:
    """xxhash 比 sha1 快 10×, 仍可作 fingerprint."""
    if path is None or not path.exists():
        return None
    h = xxhash.xxh64()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


GLOBAL_XMP_KEYS = {
    'crs:Temperature', 'crs:Tint', 'crs:Exposure2012', 'crs:Contrast2012',
    'crs:Highlights2012', 'crs:Shadows2012', 'crs:Whites2012', 'crs:Blacks2012',
    'crs:Vibrance', 'crs:Saturation',
    'crs:ToneCurvePV2012', 'crs:ToneCurvePV2012Red',
    'crs:ToneCurvePV2012Green', 'crs:ToneCurvePV2012Blue',
    # HSL adjustments
    'crs:HueAdjustmentRed', 'crs:HueAdjustmentOrange', 'crs:HueAdjustmentYellow',
    'crs:HueAdjustmentGreen', 'crs:HueAdjustmentAqua', 'crs:HueAdjustmentBlue',
    'crs:HueAdjustmentPurple', 'crs:HueAdjustmentMagenta',
    'crs:SaturationAdjustmentRed', 'crs:SaturationAdjustmentOrange',
    'crs:SaturationAdjustmentYellow', 'crs:SaturationAdjustmentGreen',
    'crs:SaturationAdjustmentAqua', 'crs:SaturationAdjustmentBlue',
    'crs:SaturationAdjustmentPurple', 'crs:SaturationAdjustmentMagenta',
    'crs:LuminanceAdjustmentRed', 'crs:LuminanceAdjustmentOrange',
    'crs:LuminanceAdjustmentYellow', 'crs:LuminanceAdjustmentGreen',
    'crs:LuminanceAdjustmentAqua', 'crs:LuminanceAdjustmentBlue',
    'crs:LuminanceAdjustmentPurple', 'crs:LuminanceAdjustmentMagenta',
    # Split toning
    'crs:SplitToningShadowHue', 'crs:SplitToningShadowSaturation',
    'crs:SplitToningHighlightHue', 'crs:SplitToningHighlightSaturation',
    'crs:SplitToningBalance',
    # Color grading
    'crs:ColorGradeShadowHue', 'crs:ColorGradeShadowSat',
    'crs:ColorGradeMidtoneHue', 'crs:ColorGradeMidtoneSat',
    'crs:ColorGradeHighlightHue', 'crs:ColorGradeHighlightSat',
    'crs:ColorGradeBlending', 'crs:ColorGradeGlobalHue',
    'crs:ColorGradeGlobalSat', 'crs:ColorGradeGlobalLum',
    # Calibration
    'crs:CameraCalibrationRedHue', 'crs:CameraCalibrationRedSaturation',
    'crs:CameraCalibrationGreenHue', 'crs:CameraCalibrationGreenSaturation',
    'crs:CameraCalibrationBlueHue', 'crs:CameraCalibrationBlueSaturation',
}

LOCAL_XMP_PREFIXES = (
    'crs:CircularGradientBasedCorrections', 'crs:GradientBasedCorrections',
    'crs:PaintBasedCorrections', 'crs:RetouchAreas',
    'crs:LuminanceSmoothing', 'crs:ColorNoiseReduction',
    'crs:Clarity', 'crs:Texture', 'crs:Dehaze',
    'crs:Sharpness', 'crs:Sharpening', 'crs:PostCropVignette',
    'crs:GrainAmount', 'crs:LensProfile',
    'crs:DefringePurple', 'crs:DefringeGreen',
)


def parse_xmp_global_only(xmp_path: Path):
    """pyexiv2 proper XMP parsing (替代 regex, 正确处理 RDF/namespace)."""
    if not xmp_path.exists():
        return False, {'reason': 'missing_xmp'}
    try:
        with pyexiv2.Image(str(xmp_path)) as img:
            xmp = img.read_xmp()
    except Exception as e:
        return False, {'reason': f'parse_error:{e}'}

    parsed_global = {}
    found_local = []
    for key, val in xmp.items():
        for prefix in LOCAL_XMP_PREFIXES:
            if key.startswith(prefix):
                if (isinstance(val, str)
                    and val.strip() not in ('', '0', '0.000000')) or \
                   (isinstance(val, (list, dict)) and len(val) > 0):
                    found_local.append(key)
        if key in GLOBAL_XMP_KEYS:
            parsed_global[key] = val

    return len(found_local) == 0, {
        'parsed_global': parsed_global,
        'rejected_local_keys': found_local,
    }


def audit_mmart(mmart_path, ppr_train_hashes, ppr_test_hashes,
                output_manifest_jsonl, output_report_md):
    """Walk MMArt folders, hash-dedupe, global-only filter."""
    folders = sorted(Path(mmart_path).glob('*/'))
    counts = {'total': len(folders), 'incomplete': 0, 'local_rejected': 0,
              'global_accepted': 0, 'overlap_ppr_train': 0, 'overlap_ppr_test': 0}

    manifest = []
    for folder in tqdm(folders, desc='MMArt audit'):
        before = folder / 'before.jpg'
        processed = folder / 'processed.jpg'
        xmp = folder / 'config.xmp'
        if not (before.exists() and processed.exists() and xmp.exists()):
            counts['incomplete'] += 1
            continue

        before_h = xxhash_file(before)
        if before_h in ppr_test_hashes:
            counts['overlap_ppr_test'] += 1
            continue
        if before_h in ppr_train_hashes:
            counts['overlap_ppr_train'] += 1

        ok, info = parse_xmp_global_only(xmp)
        if not ok:
            counts['local_rejected'] += 1
            continue
        counts['global_accepted'] += 1

        for variant in ('user_want_short', 'user_want_middle', 'user_want_long'):
            v_path = folder / f'{variant}.txt'
            if v_path.exists():
                manifest.append({
                    'sample_id': f'{folder.name}_{variant}',
                    'source_dataset': 'mmart_ppr10k',
                    'source_image_id': folder.name,
                    'before_xxh': before_h,
                    'processed_xxh': xxhash_file(processed),
                    'config_xmp_xxh': xxhash_file(xmp),
                    'instruction': v_path.read_text(),
                    'instruction_length': variant.replace('user_want_', ''),
                    'parsed_xmp_global': info['parsed_global'],
                    'label_type': 'no_action_label',
                    'license_group': 'mmart_audit',
                })

    Path(output_manifest_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(output_manifest_jsonl, 'w') as f:
        for entry in manifest:
            f.write(json.dumps(entry) + '\n')

    report = f"""# MMArt-PPR10K Audit Report
Total folders: {counts['total']}
Incomplete: {counts['incomplete']}
Rejected (local ops): {counts['local_rejected']}
Global-only accepted: {counts['global_accepted']}
Overlap PPR train: {counts['overlap_ppr_train']}
Overlap PPR test: {counts['overlap_ppr_test']} (excluded)

Manifest entries: {len(manifest)}
Effective Tier D pairs: {counts['global_accepted']}
"""
    Path(output_report_md).parent.mkdir(parents=True, exist_ok=True)
    Path(output_report_md).write_text(report)
    return counts
```

**P0a 算力**: `~1 CPU-day, 0 H100-h` (pyexiv2 仍 CPU 密集, 多进程加速 ~4 小时).

---

## 14. 训练栈 + Minimal GRPO Loop

### 14.1 阶段化训练

| 阶段 | 目标 | 训练对象 | 数据 |
| :- | :- | :- | :- |
| S0 | renderer / decoder sanity | decoder + renderer | Tier A small |
| S1 | deterministic SFT | action decoder + LoRA | Tier A |
| S2 | off-manifold expression | decoder + D4 | Tier B |
| S3 | continuous RL | latent Gaussian policy | Tier A+B |
| S4 | real-pair fine-tune | decoder + LoRA | Tier C + filtered Tier D |
| S5 | instruction reward | LoRA + policy head | Tier C/D/E + reward model |

### 14.2 Trainer 选择

```text
P0:    pure PyTorch minimal continuous-GRPO loop  (验证数学闭环)
P1:    TRL/custom Trainer                          (正式 ablation)
P2:    verl 或 distributed actor                   (视吞吐)
辅助:  ms-swift 仅 SFT / token-DAPO baseline
```

### 14.3 P0 Minimal GRPO loop

```python
# train/minimal_grpo.py
"""P0 minimal continuous-GRPO with v6.1.1 corrections.

关键:
1. Rollout 与 Update 严格分离 (no_grad 包 rollout)
2. z_old.detach() 防止 reparameterized pathwise gradient 与 PG 混淆
3. Differentiable render loss 单独路径 (用 fresh reparameterized z)
4. gradient clipping max_norm=1.0
"""
import math
import torch
import torch.nn as nn
from torch.distributions import Normal

from shape_curve.train.reward import compute_p0_reward
from shape_curve.train.losses import render_loss


def train_step(model, batch, optimizer, step, total_steps, config):
    image = batch['input_image'].cuda()
    target = batch['target_image'].cuda()
    instruction = batch['instruction']
    gt_action = batch.get('gt_action', None)

    group_size = config['group_size']
    eps_clip = config['eps_clip']

    # ═══ Rollout (no grad) ═══
    with torch.no_grad():
        rollouts = []
        for g in range(group_size):
            rendered, action, aux = model(image, instruction, mode='rl_sample',
                                          step=step, total_steps=total_steps)
            log_prob = Normal(action.mu, action.std).log_prob(action.z).sum(-1)
            metrics = extract_metrics(rendered, target, action, aux)
            reward = compute_p0_reward(metrics)
            rollouts.append({
                'z': action.z.detach(),
                'log_prob': log_prob.detach(),
                'reward': reward,
            })
        rewards = torch.stack([r['reward'] for r in rollouts])
        advantage = (rewards - rewards.mean(0)) / (rewards.std(0) + 1e-6)

    # ═══ Update phase ═══
    optimizer.zero_grad()
    loss_grpo = 0.0
    for g in range(group_size):
        h = model.extract_h(image, instruction)
        mu_new, std_new, _, _ = model.z_policy(h, step=step, total_steps=total_steps)
        log_prob_new = Normal(mu_new, std_new).log_prob(rollouts[g]['z']).sum(-1)
        ratio = torch.exp(log_prob_new - rollouts[g]['log_prob'])
        clipped = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip)
        loss_g = -torch.min(ratio * advantage[g], clipped * advantage[g]).mean()
        loss_grpo += loss_g / group_size

    # Differentiable render loss path (fresh reparameterized z)
    rendered, action, aux = model(image, instruction, mode='rl_sample',
                                   step=step, total_steps=total_steps)
    loss_render = config['lambda_render'] * render_loss(rendered, target)
    loss_gamut = config['lambda_gamut'] * aux['r_gamut']

    loss_action = torch.tensor(0.0, device=image.device)
    if gt_action is not None:
        loss_action = config['lambda_action'] * action_l1_loss(action, gt_action)

    loss_total = loss_grpo + loss_render + loss_gamut + loss_action
    loss_total.backward()
    nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                              max_norm=1.0)
    optimizer.step()

    return {
        'loss_total': loss_total.item(),
        'loss_grpo': loss_grpo.item(),
        'reward_mean': rewards.mean().item(),
        'reward_std': rewards.std().item(),
        'residual_ratio': aux['residual_ratio'].item(),
        'clipping_ratio': aux['clipping_ratio'].item(),
        'dict_sparsity': aux['dict_sparsity'].item(),
    }
```

---

## 15. MVP-0 测试计划

| Test | 目标 | 数据 | 通过 |
| :- | :- | :- | :- |
| **A** SFT/renderer sanity | decoder+renderer 能学 Tier A | 16-64 images × Tier A | action MSE ↓≥70%, render ΔE/L1 ↓, monotonic≈0 |
| **B** token-only GRPO negative | 证明 token-only GRPO 不够 | 同 A, special token 固定 | reward 无 ↑, error 不变 |
| **C-1** head-only RL sanity | policy head 可优化 | 16-64 conditions, VLM frozen | reward ↑, error ↓ |
| **C-2** hidden informativeness probe | hidden 携带 condition | Tier A 8K train | true h vs shuffled: R²↑0.20, ΔE↓15%, F1↑10pt, 95% CI 不跨 0 |
| **C-3** conditional RL toy | credit assignment + conditioning | 32 conflicting conditions | true h reward > shuffled, warm/cool 区分 ≥80% |
| **C-4** LoRA unfreeze | LoRA 进一步适配 | C-1 之后 +500 step | held-out ΔE ↓≥0.5 dB |
| **D** Tier B off-manifold probe | T3 在非自生成 target 上 | 5K Tier B | T3-full vs D4: PSNR ≥A2-0.5, dict ratio ≥50% |

---

## 16. Layer Probe Protocol

### 16.1 数据 + 任务

```text
samples: Tier A 8K train / 1K val
archetypes: 10-12 core + 3 stress
candidate_layers: 8, 12, 14, 16, 18, 20, 22

probe tasks:
  1. action regression       → R², MSE, render ΔE
  2. archetype classification → accuracy, macro-F1
  3. parameter regression    → Spearman ρ
  4. shuffled control        → gap to true hidden
```

### 16.2 通过阈值

```text
true_hidden vs constant_hidden:
  action R² 提升 ≥ 0.20
  render mean ΔE2000 ↓ ≥ 15%

true_hidden vs image_shuffled:
  render mean ΔE2000 ↓ ≥ 10%

true_hidden vs instruction_shuffled:
  archetype macro-F1 ↑ ≥ 10 points

bootstrap: 95% CI of improvement 不跨 0
```

最终选择: 优先 validation rendered ΔE/LPIPS 最好的层; 若差距 < 2%, 选更早层减少 next-token bias. 24-layer backbone 默认 layer 18.

---

## 17. D4 Dense-LUT Controlled Upper Bound

### 17.1 不使用外部 zero-shot 作 D4

VeraRetouch / JarvisArt / AceTone 可作 external baselines, 不作 controlled upper bound.

### 17.2 D4 配置

```yaml
D4_dense_basis_lut:
  same_backbone: VeraRetouch-0.6B + LoRA (与 T3 同)
  same_z_dim: 64
  same_training_data: Tier A + Tier B
  same_training_steps: ~12K
  grid_size: 33
  K_dense: 64
  basis_type: smooth_random_frozen   # 离线 PCA from Tier B residuals
  coeff_range: tanh × 0.08
```

### 17.3 Architecture

```text
h_pooled + z → FiLM decoder → dense_coef ∈ R^{64}
                              → LUT_dense_residual = Σ_k b_k × B_k_dense (frozen basis)
                              → LUT_final = LUT_main + LUT_dense_residual
```

### 17.4 D4 loss

```text
L_D4 = render_loss + color_loss + TV3D(LUT_dense) + gamut_penalty
```

D4 不做 dictionary sparsity, 不做 free-tail efficiency. 是表达力上界, 不是 interpretation model.

### 17.5 Q1 通过门

```text
T3-full vs D4 on Tier B/C:
  PSNR gap ≤ 1.0 dB
  mean ΔE2000 gap ≤ 1.5
  LPIPS gap ≤ 0.03
```

---

## 18. P0 Reward Sanity Check: 4-Candidate Canary

### 18.1 目的

正式 GRPO 前确认 reward 不反向鼓励 overedit / wrong hue / clipping / banding.

### 18.2 构造

对 100 个 (input, instruction, target) 生成 4 candidate:

```text
A target_near:           target action + small noise
B underedit:             0.3 × target action
C overedit:              1.8 × target action
D wrong_hue_artifact:    hue direction flipped + clipping stress
```

### 18.3 通过门

```text
reward(A) > reward(B) in ≥ 75% samples
reward(A) > reward(C) in ≥ 75% samples
reward(A) > reward(D) in ≥ 90% samples
reward(D) bottom-ranked in ≥ 80% samples
```

不通过 → 调 τ, constraints, residual efficiency, 再进入 GRPO.

---

## 19. Manifest 版本控制 + Leakage Assert

### 19.1 文件组织

```text
data_manifests/
  manifest_v001_raw_sources.jsonl
  manifest_v002_splits.jsonl
  manifest_v003_tier_a_20k.jsonl
  manifest_v004_tier_b_5k.jsonl
  manifest_v005_mmart_audit.jsonl
  manifest_v006_train_mix_p1.jsonl
  manifest_v006_train_mix_p1.sha256
```

### 19.2 Schema (dataclass)

```python
from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class ManifestEntry:
    sample_id: str
    manifest_version: str
    created_at: str
    generator_git_commit: str
    config_sha256: str
    source_dataset: Literal["fivek", "ppr10k", "mmart_ppr10k", "procedural"]
    source_image_id: str
    source_group_id: Optional[str] = None
    before_path: str = ""
    before_xxh: str = ""
    target_path: Optional[str] = None
    target_xxh: Optional[str] = None
    teacher_lut_path: Optional[str] = None
    teacher_lut_xxh: Optional[str] = None
    config_xmp_path: Optional[str] = None
    config_xmp_xxh: Optional[str] = None
    split: Literal["train", "val", "test"] = "train"
    image_domain: Literal["general", "portrait", "instruction_global"] = "general"
    label_type: Literal["gt_action", "pseudo_action", "teacher_lut", "no_action_label"] = "no_action_label"
    synthetic_tier: Optional[Literal["A", "B"]] = None
    archetype_id: Optional[str] = None
    teacher_family: Optional[str] = None
    random_seed: Optional[int] = None
    license_group: Literal["fivek_research", "ppr_nc_research",
                            "procedural_public", "mmart_audit"] = "ppr_nc_research"
    allowed_use: Literal["research_only", "procedural_public"] = "research_only"
    color_space: str = "srgb_gamma_encoded"
    notes: str = ""
```

### 19.3 Leakage assert

```python
def assert_no_leakage(rows):
    by_hash = {}
    for r in rows:
        for k in ['before_xxh', 'target_xxh', 'teacher_lut_xxh', 'config_xmp_xxh']:
            h = r.get(k)
            if not h:
                continue
            by_hash.setdefault((k, h), set()).add(r['split'])
    leaks = {k: v for k, v in by_hash.items() if len(v) > 1}
    if leaks:
        raise RuntimeError(f'split leakage detected: {list(leaks.items())[:10]}')
```

---

## 20. Inference + AMP + 色彩空间

### 20.1 Inference workflow

```text
Input image + instruction
  → resize/normalize to training resolution
  → prompt with <RET_ACTION>
  → extract selected-layer hidden (default L=18)
  → z = μ_z deterministic (single) 或 μ_z + τ·σ·ε (multi-candidate)
  → decode action via FiLMActionDecoder
  → build LUT_pre
  → clamp LUT_pre to [0,1] (eval mode)
  → trilinear apply (原分辨率)
  → export:
       rendered image (PNG/JPEG)
       final 33³ LUT (CUBE format)
       action JSON (curve + HSL + WB + dict_coef + tail)
       dictionary coefficient bar plot
       atom contribution visualization
```

候选采样: τ_sample ∈ {0.3, 0.5, 0.8} → 3 candidates

### 20.2 AMP / DeepSpeed / FSDP

P0:
```yaml
precision: fp32_or_bf16
trainer: pure_pytorch
vlm: frozen
batch_size: small
```

P1:
```yaml
precision: bf16
optimizer: AdamW
lr_decoder: 5e-4
lr_lora: 1e-4
lr_log_std: 5e-5
clip_grad_norm: 1.0
device: 1-2 GPUs
activation_checkpointing: optional
```

P2/P3:
```yaml
distributed:
  deepspeed: zero2 first
  fsdp: only if memory pressure
  vllm_rollout: DISABLED for continuous hidden-state action
```

### 20.3 色彩空间锁定

```text
working space: sRGB gamma-encoded RGB ∈ [0, 1]
LUT domain:    sRGB gamma-encoded RGB ∈ [0, 1]
operations:    global 3D LUT over sRGB grid
```

禁止: LogC / S-Log / V-Log / Cineon / DaVinci Wide Gamut / 未知 DOMAIN_MIN/MAX .cube / legal-range video LUT.

---

## 21. Phase 2 Dictionary Expansion (tensorly + sklearn)

```python
# 仅 Q4 未达标或 D4 gap 明显时执行
import numpy as np
import torch
from sklearn.decomposition import SparseCoder
from tensorly.decomposition import parafac
import tensorly as tl


def phase2_expand_dictionary(tier_b_holdout, tier_c_holdout,
                              phase1_dict, rho_phase1):
    """学 GT residual gap, 不是 model prediction error."""
    gaps = []
    
    for sample in tier_b_holdout + tier_c_holdout:
        main_action_fitted = fit_main_to_target(sample.input, sample.target)
        lut_main_fitted = build_main_lut(main_action_fitted)
        residual_gt = sample.target_lut - lut_main_fitted

        # sklearn SparseCoder
        flat_residual = residual_gt.reshape(-1).numpy()
        coder = SparseCoder(
            dictionary=phase1_dict.reshape(16, -1).numpy(),
            transform_algorithm='lasso_lars',
            transform_alpha=0.01,
        )
        a_phase1 = coder.transform(flat_residual.reshape(1, -1))[0]

        recon = sum(rho_phase1[j] * a_phase1[j] * phase1_dict[j] for j in range(16))
        gap = residual_gt - recon
        gaps.append(gap.numpy())

    # tensorly CPD
    gap_tensor = tl.tensor(np.stack(gaps, axis=0))
    weights, factors = parafac(gap_tensor, rank=8, normalize_factors=True)
    new_atoms = reconstruct_atoms_from_cpd(weights, factors)

    # 接受门
    return [atom for atom in new_atoms if accept_gate(atom, phase1_dict, gaps)]


def accept_gate(atom, phase1_dict, gaps):
    return (
        check_p95_de2000_drop(atom, phase1_dict, gaps) >= 0.10
        and check_D4_gap_shrink(atom) >= 0.10
        and check_atom_usage(atom) >= 0.01
        and max_cosine(atom, phase1_dict) <= 0.85
        and is_namable(atom)
    )
```

---

## 22. 时间线与算力

| 阶段 | 周期 | 主要产出 | 算力 |
| :- | :- | :- | :- |
| P0a | 0.5 周 | MMArt hash audit + manifest | ~1 CPU-day, 0 H100-h |
| P0b | 1.5 周 | 16 atoms + Tier A 20K + renderer | ~30 H100-h |
| P0c | 1 周 | Tier B 5K + visualization cards | ~20 H100-h |
| P0d | 1.5 周 | decoder + z policy + minimal GRPO + MVP tests | ~80 H100-h |
| P1a | 0.5 周 | Tier A SFT checkpoint | ~50 H100-h |
| P1b | 2 周 | 12 controlled runs (hybrid/no-shape/dict-only/D4 × seeds) | ~600 H100-h |
| P2 | 2 周 | ablation M/R/K/layer | ~400 H100-h |
| P3 | 2 周 | Tier C real-pair fine-tune + reward variants | ~300 H100-h |
| P4 | 2 周 | Tier E external eval | ~100 H100-h |
| P5 | 1 周 | stability + Phase 2 atoms | ~150 H100-h |
| P6 | 1 周 | data scaling | ~150 H100-h |
| P7 | 1 周 | intervention/user study/writing | ~30 H100-h |

```text
MVP 裁决 P0-P2:    ≈ 1,185 H100-h ≈ 49 H100-days
完整路径 P0-P7:    ≈ 1,915 H100-h ≈ 80 H100-days
```

---

## 23. 立即执行清单: 前 8 工作日

| Day | 任务 | 交付物 |
| :-: | :- | :- |
| 0 | MMArt-PPR10K hash audit (pyexiv2 + multiprocessing) + license manifest | `manifest_audit.md` |
| 1 | PPR10K group split assert + FiveK split + license manifest | `splits/*.json` |
| 1-2 | 16 semantic atoms generator (kornia helpers) + visualization cards | `atoms_phase1.yaml`, atom cards PNG |
| 2-3 | 10-12 archetype distributions + holdout family + Tier A 20K | `archetypes.yaml` + 20K JSONL |
| 3 | Tier B dense-teacher (5 families) + 5K Tier B | `generate_tier_b.py` + 5K JSONL |
| 4 | Renderer v6.1.1 (scipy BSpline + kornia HSV + softclip_identity + energy fix) | renderer unit tests pass |
| 5 | Data pipeline 端到端 + manifest leakage assert + 5 sample sanity | data sanity report |
| 5-6 | ZPolicy + FiLMActionDecoder + ShapeCurveAction + VLMHiddenExtractor | forward sanity |
| 6-7 | Minimal continuous-GRPO loop + 4-candidate reward sanity | Test A 启动 |
| 7-8 | MVP-0 full tests (A/B/C-1/C-2/C-3/C-4/D) + layer probe | `p0_decision_report.md` |

---

## 24. 风险与 Kill 标准

| 风险 | 概率 | 影响 | 应急 |
| :- | :-: | :-: | :- |
| T3-full 与 D4 差距 > 2 dB | 中 | 致命 | pivot 为 D4 explanation layer |
| dictionary explained ratio < 50% | 中 | 高 | Phase 2 expansion |
| hidden probe true h ≈ shuffled h | 低-中 | 致命 | 撤回 VLM-conditioned 主张 |
| continuous GRPO 不稳定 | 中 | 高 | 回退 SFT + differentiable render |
| MMArt audit 后 global-only 少 | 中 | 中 | Tier D 降级 instruction paraphrase |
| PPR10K split 泄漏 | 低 | 高 | 强制 group-level assert |
| FiveK/PPR10K license 限制 | 中 | 中 | 仅研究模型, 公开 generator |
| free tail 吞噬 main | 中 | 中 | g_max bound + residual efficiency reward |
| EditScore 对 subtle retouching 不对齐 | 高 | 中 | P3 才启用, 权重 ≤0.20 |
| Tier A 自生成闭环 | 已规避 | 高 | Q1/Q4 只用 Tier B+C |

### 硬 Kill 标准

```text
1. MVP-0 Test C-2 失败:    T3 主线 kill
2. P1 Q1+Q4 都否决:        pivot 为 explanation layer
3. P4 Tier E 全面失败:      投稿降级, 主打 interpretability
```

---

## 25. 论文表述边界

建议:
> *We propose an interpretable global retouching action space that decomposes edits into hand-coded main controls, a frozen semantic residual dictionary, and a small low-rank residual tail. The model is trained on research-license paired retouching data and reproducible procedural synthetic tiers. Controlled ablations compare the proposed hybrid renderer against a dense-LUT upper bound under the same backbone, data, and budget.*

不要写:
- T3 beats JarvisArt / VeraRetouch / AceTone universally
- T3 solves all Lightroom retouching tools
- The 16 CPD components have stable semantics
- MMArt-PPR10K provides 50K independent base images
- FFHQR is a suitable general retouch training set

---

## 26. 最终 Go / No-Go

```text
Go:
  FiveK + PPR10K core data
  Tier A/B/C/D/E data tiers
  C-hybrid dictionary: 16 frozen semantic atoms + optional Phase 2
  latent Gaussian policy z_dim=64
  FiLM decoder
  B-spline K=10 free tail (scipy.interpolate.BSpline)
  P0 minimal loop before TRL
  库优先: piq / kornia / colour-science / pyexiv2 / scipy / tensorly / sklearn / einops

No-Go unless fixed:
  FFHQR main training
  MMArt rows counted as base images
  Q1/Q4 on Tier A only
  real pairs forced action L1
  direct 1682-D Gaussian policy
  free CPD components framed as stable semantic brushes
  self-written ΔE2000 / LPIPS / B-spline / XMP parser (用对应库)
```

---

## 27. 资料核实依据

- MIT-Adobe FiveK: https://data.csail.mit.edu/graphics/fivek/ (research-only)
- FiveK HF mirror (5000 DNG + 25000 expert): https://huggingface.co/datasets/yuukicammy/MIT-Adobe-FiveK
- FiveK License: https://data.csail.mit.edu/graphics/fivek/legal/LicenseAdobe.txt
- PPR10K: https://github.com/csjliang/PPR10K (Apache code + non-commercial dataset)
- PPR10K paper (1681 groups): https://openaccess.thecvf.com/content/CVPR2021/papers/Liang_PPR10K_*.pdf
- MMArt-PPR10K: https://huggingface.co/datasets/JarvisArt/MMArt-PPR10k
- FFHQR (CC BY-NC-SA): https://github.com/skylab-tech/ffhqr-dataset
- Qwen2.5-0.5B (24 layers): https://huggingface.co/Qwen/Qwen2.5-0.5B
- TRL GRPOTrainer: https://huggingface.co/docs/trl/en/grpo_trainer
- FoveateR Gaussian continuous policy: arXiv 2604.21079
- VeraRetouch backbone: arXiv 2604.27375, HF Gyh68/VeraRetouch
- piq (Image Quality): https://github.com/photosynthesis-team/piq
- kornia.color: https://kornia.readthedocs.io/en/latest/color.html
- colour-science: https://www.colour-science.org/
- tensorly: http://tensorly.org/
- pyexiv2: https://github.com/LeoHsiao1/pyexiv2

---

## 附录: 三份文档统一映射表

| v6.1.1-unified 节 | v6.1-final | GPT-5.5 v6.1.1 | Claude Complete |
| :- | :- | :- | :- |
| §0-4 战略 | §0-2, §17 | -- | -- |
| §5 项目结构 | -- | §12 | Part 1 |
| §6 模型 (ZPolicy/FiLM/HiddenExt) | §3.1-3.5 | §20.1-20.3 | §7 |
| §7 渲染器 | §4 | §20.4-20.5 | §6 |
| §8 数据栈 | §5 | -- | Part 2 |
| §9 Loss (含 reconstructed-grid) | §9 | **§19 新增** | §10 |
| §10 Reward + τ 表 | §9 | **§24 新增** | -- |
| §11 16 atoms | §3.5.2 | **§21 完整代码** | Part 3 |
| §12 Tier B 5 family | §5.3 | **§22 完整代码** | Part 5 |
| §13 MMArt audit | §5.3 Tier D | **§23 完整代码** | Part 2 |
| §14 训练栈 | §7 | §20.5 | Part 8 |
| §15 MVP-0 | §8 | -- | Part 11 |
| §16 Layer probe | -- | **§25 新增** | -- |
| §17 D4 dense-LUT | §10 | **§26 新增** | -- |
| §18 Reward sanity canary | -- | **§27 新增** | -- |
| §19 Manifest 版本控制 | -- | **§28 新增** | -- |
| §20 Inference + AMP + sRGB | -- | **§29 新增** | -- |
| §21 Phase 2 (sklearn+tensorly) | §11 | -- | -- |
| §22 时间线 | §13 | §13.1 (P0a 修正) | -- |
| §23 立即执行清单 | §14 | §29.5 | -- |
| §24 风险 | §15 | -- | -- |
| §25 论文边界 | §16 | -- | -- |
| §26 Go/No-Go | §17 | -- | -- |
| §27 资料核实 | §18 | §30 | §16 |
