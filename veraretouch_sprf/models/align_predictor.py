#!/usr/bin/env python3
# 源自 experiments/prs/EPR-051_masked-restore-production/stage0/sprf/align_predictor.py @ 2026-09-06（原件未入 git；原 sha256 见 veraretouch_sprf/PROVENANCE.md），逐字复制/仅改导入与路径解析（EPR-052）。
"""EPR-051 · stage0 · sprf · T-ALIGN：predicted-LUT-latent（contract=predicted_lut）。

把 C-LUT 的 **oracle 查表**换成**预测**：训练一个 predictor 从图像证据回归 128 维
编辑 latent，其余管线与 C-LUT 逐键相同 —— 得到第一个**可部署口径**的数字。

对齐目标（「hidden state 对齐 LUT 特征」的第一步）
--------------------------------------------------
C-LUT 的编辑 latent 是 `edit_enc(逆表描述子)`，而逆表描述子只是 `lut_id` 的函数
（L_m 是池属性，与图像/β/s 无关）。所以**目标 latent 库也只是 lut_id 的函数**：
`target[j] = edit_enc_frozen(inv_table[j])`，4051 支 + 1 个 null 行 = 4052×128，
一次算完即可，不需要逐样本生成。

注入方式（**不改任何共享源码**）
--------------------------------
`SprfModel.edit_descriptor` 在 `inv_lut` 档是 `inv_table[src]`，随后 `StageConditioner`
把 `edit_enc(descriptor)` 与 `h` 拼接进 FiLM 头。要塞进**预测**的 latent，只需在
**实例**上做两件事（类定义与文件都不动，所以在跑的 C-LUT final 不受影响）：

    model.edit_descriptor = lambda src: lat_flat[src]   # (B,) long -> (B,128)
    model.cond.edit_enc   = nn.Identity()               # latent 直连 FiLM 拼接

`edits[b, k] = b*K + k` 索引进本 batch 展平后的预测 latent。实测梯度可回流到
predictor（feasibility：12/12 行拿到非零梯度）。

预测端只吃**推理时拿得到**的证据：冻结 SigLIP2 pooled `c` + 退化图 `y` 的直方图
（含 β 加权）。**不碰 lut_id、不碰 x（clean）、不碰逆表** —— 所以 `predicted_lut`
是可部署契约，与 `oracle_lut` 分列、禁混读。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# 直方图格与 (b,g,r) 展平序，与 precompute_inv_lut.output_grid / edit_cond 一致。
HIST_STATS = 8


def hist_bins(grid: int) -> int:
    return int(grid) ** 3


def feature_dim(grid: int) -> int:
    """每阶段的手工特征维度：β 加权直方图 + 全局直方图 + 8 个标量。"""
    return 2 * hist_bins(grid) + HIST_STATS


def _bin_index(c: torch.Tensor, grid: int) -> torch.Tensor:
    q = (c.clamp(0.0, 1.0) * int(grid)).floor().long().clamp_(0, int(grid) - 1)
    r, g, b = q[:, 0], q[:, 1], q[:, 2]
    return (b * int(grid) + g) * int(grid) + r


def hist_features(y_px: torch.Tensor, alphas_px: torch.Tensor,
                  grid: int) -> torch.Tensor:
    """`y_px` (P,3) 退化图像素、`alphas_px` (K,P) 逐阶段 β -> (K, feature_dim)。

    R0 式手工直方图（X-Probe 里手工直方图携带的 LUT 证据是 SigLIP 的 ~2 倍），
    这里加了**逐阶段 β 加权**那一路：第 m 阶段真正作用过的像素分布，才是关于
    L_m 的证据；全局那一路给整图色彩上下文，两路都只用 `y`。

    计数按 `n_bins` 归一（均匀分布时均值为 1），量级与 c 可比。
    """
    n = hist_bins(grid)
    p = y_px.shape[0]
    idx = _bin_index(y_px, grid)
    ones = torch.ones(p, dtype=torch.float32)
    hg = torch.zeros(n, dtype=torch.float32).index_add_(0, idx, ones) * (n / max(p, 1))
    gmean = y_px.mean(0)
    out = []
    for k in range(alphas_px.shape[0]):
        w = alphas_px[k].to(torch.float32)
        s = float(w.sum())
        hb = torch.zeros(n, dtype=torch.float32).index_add_(0, idx, w)
        hb = hb * (n / s) if s > 0 else hb
        wm = ((y_px * w.unsqueeze(-1)).sum(0) / s) if s > 0 else torch.zeros(3)
        stats = torch.cat([wm, gmean,
                           torch.tensor([w.mean(), (w > 0).float().mean()])])
        out.append(torch.cat([hb, hg, stats]))
    return torch.stack(out)


class LatentPredictor(nn.Module):
    """`(c, 手工特征, 阶段号) -> 128 维编辑 latent`。

    两路各自投影后相加（而不是先 cat 再一个大 Linear）：两路量纲/稀疏度差很多，
    分开投影让各自的尺度独立，参数也少一半。阶段号走 embedding —— 同一张图的
    第 m 阶段与第 m' 阶段要预测**不同**的 LUT，阶段身份必须显式进去。
    """

    def __init__(self, c_dim: int, feat_dim: int, n_steps: int, latent: int,
                 hidden: int, layers: int):
        super().__init__()
        if int(layers) < 2:
            raise SystemExit(f"align_predictor: layers = {layers} 必须 >= 2")
        self.proj_c = nn.Linear(int(c_dim), int(hidden))
        self.proj_f = nn.Linear(int(feat_dim), int(hidden))
        self.emb_stage = nn.Embedding(int(n_steps), int(hidden))
        mid: list[nn.Module] = []
        for _ in range(int(layers) - 1):
            mid += [nn.Linear(int(hidden), int(hidden)), nn.SiLU()]
        self.trunk = nn.Sequential(*mid)
        self.head = nn.Linear(int(hidden), int(latent))
        self.n_steps, self.latent = int(n_steps), int(latent)

    def forward(self, c: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """`c` (B, c_dim)、`feat` (B, K, feat_dim) -> (B, K, latent)。"""
        b, k, _ = feat.shape
        if k != self.n_steps:
            raise SystemExit(f"align_predictor: 阶段数 {k} != {self.n_steps}")
        m = torch.arange(k, device=feat.device).view(1, k).expand(b, k)
        h = (self.proj_c(c).unsqueeze(1) + self.proj_f(feat) + self.emb_stage(m))
        return self.head(self.trunk(F.silu(h)))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


@torch.no_grad()
def build_target_latents(edit_enc: nn.Module, inv_table: torch.Tensor,
                         chunk: int = 256) -> torch.Tensor:
    """`edit_enc_frozen(inv_table[j])` 逐行算出对齐目标库 (N_lut+1, latent)。

    `inv_table` 末行是 `load_inv_table` 追加的全零 null 行（Δ_edit_null 用），
    它的 latent 一并算出来，这样 null 控制在预测臂上仍是同一个东西。
    """
    outs = []
    for i in range(0, inv_table.shape[0], int(chunk)):
        outs.append(edit_enc(inv_table[i:i + int(chunk)]))
    return torch.cat(outs)
