#!/usr/bin/env python3
# 源自 experiments/prs/EPR-051_masked-restore-production/stage0/sprf/stage_backend_g4d.py @ 2026-09-06（原件未入 git；原 sha256 见 veraretouch_sprf/PROVENANCE.md），逐字复制/仅改导入与路径解析（EPR-052）。
"""EPR-051 · stage0 · arm=sprf · T5 · B-g4d：把逐阶段 action 换成 G4D 载体。

消融的问题：SPRF 的收益到底来自「阶段分解 + 逐阶段求逆」这个**结构**，还是来自
「逐像素 MLP」这个 **action 表示**？B-g4d 保持 SPRF 的一切（同一条真实折线、同一套
β 场、同一逆序 Euler、同一批损失与判据），只把 action 的产生方式换成 EPR-028 的
G4D 载体：

    θ_m ∈ R^1308  ←  新头（线性）从 StageConditioner 的 (c + stage_type/stage_id
                     + depth + s 嵌入) 生成，**逐阶段一份**
    ĥ            =  Glut4DCarrier(z, s, θ_m) − z

为什么是 `C(z) − z`：SPRF 的更新式是 `z ← z + β_m·ĥ`，代进去就是
`z ← (1-β)·z + β·C(z)` —— 正是渲染律 `mix_alpha(z, C(z), β)` 本身。
所以 C 的语义是「β=1 时的完全反演色」，与 h* = r^(m-1) − L_m(r^(m-1)) 的口径一致
（`r^m + β·h* = r^(m-1)`）。

初始化：末层权重全零、bias 抄 `G4DGenerator._init_heads` 的恒等 init，于是
θ_m 是常量恒等参数 ⇒ `C(z) ≈ z` ⇒ `ĥ ≈ 0` ⇒ step-0 ≈ 恒等。
**注意是「≈」不是「=」**：g4d 的恒等 init 只保证 `sum_i w_i ≈ 1`
（`G4DGenerator` 自己的 docstring：`f(x, s) ~= x`），所以 B-g4d 的 A1 判据必须是
**两界**（绝对值界 + 8bit 量化级界），不能是 `torch.equal`。见 train_sprf 的 A1。

`g4d.py` 一行不动：本文件只 import 它的类与常量。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

from veraretouch_sprf import _paths as _P  # EPR-052：替代原 sys.path 注入块
_P.ensure_sys_path()
REPO = _P.REPO

from q3vl.whatb.arms.g4d import (                      # noqa: E402
    MU_S_INIT_RANGE,
    OPACITY_LOGIT_INIT,
    SIGMA_RGB_INIT,
    TAU_RAW_INIT,
    G4DConfig,
    G4DParams,
    Glut4DCarrier,
    n_params_g4d,
)
from q3vl.whatb.glut import softplus_inverse, uniform_grid_positions  # noqa: E402

from veraretouch_sprf.data.stage_targets import die                          # noqa: E402

ACTION_BACKENDS = ("mlp", "g4d")


def theta_layout(n: int, conditional: bool) -> list[tuple[str, tuple[int, ...]]]:
    """扁平 θ 的切分表。A3/N=48 合计 1308，与 `n_params_g4d` 对表断言。"""
    lay = [("mu_x", (n, 3)), ("chol_diag", (n, 3)), ("chol_off", (n, 3)),
           ("opacity_logit", (n,)), ("m_local", (n, 3, 3)), ("b_local", (n, 3)),
           ("g_matrix", (3, 3)), ("g_bias", (3,))]
    if conditional:
        lay += [("beta", (n, 3)), ("mu_s", (n,)), ("tau_raw", (n,))]
    return lay


def identity_theta(cfg: G4DConfig) -> torch.Tensor:
    """`_init_heads` 的恒等 init，摊平成一个 θ 向量（作为末层 bias）。"""
    n = cfg.n_gauss
    gen = torch.Generator().manual_seed(int(cfg.init_seed))
    parts = {
        "mu_x": uniform_grid_positions(n),
        "chol_diag": torch.full((n, 3), float(softplus_inverse(SIGMA_RGB_INIT))),
        "chol_off": torch.zeros(n, 3),
        "opacity_logit": torch.full((n,), float(OPACITY_LOGIT_INIT)),
        "m_local": torch.eye(3).reshape(1, 3, 3).expand(n, 3, 3).contiguous(),
        "b_local": torch.zeros(n, 3),
        "g_matrix": torch.zeros(3, 3),
        "g_bias": torch.zeros(3),
    }
    if cfg.conditional:
        lo, hi = MU_S_INIT_RANGE
        parts["beta"] = torch.zeros(n, 3)
        parts["mu_s"] = torch.rand(n, generator=gen) * (hi - lo) + lo
        parts["tau_raw"] = torch.full((n,), float(TAU_RAW_INIT))
    flat = torch.cat([parts[k].reshape(-1)
                      for k, _ in theta_layout(n, cfg.conditional)])
    want = n_params_g4d(n, cfg.mode)
    if flat.numel() != want:
        die(f"identity_theta 长度 {flat.numel()} != n_params_g4d {want}")
    return flat


def unpack_theta(flat: torch.Tensor, cfg: G4DConfig) -> G4DParams:
    """`(B, 1308) -> G4DParams`，切分顺序与 `theta_layout` 一致。"""
    n = cfg.n_gauss
    b = flat.shape[0]
    out, off = {}, 0
    for name, shape in theta_layout(n, cfg.conditional):
        size = 1
        for d in shape:
            size *= d
        out[name] = flat[:, off:off + size].reshape(b, *shape)
        off += size
    if off != flat.shape[1]:
        die(f"unpack_theta 只用了 {off}/{flat.shape[1]} 个数")
    return G4DParams(mode=cfg.mode, **out)


class G4DActionHead(nn.Module):
    """StageConditioner 的隐状态 -> θ_m -> Glut4DCarrier -> ĥ = C(z) − z。"""

    def __init__(self, hidden: int, g4d_cfg: G4DConfig, point_chunk: int):
        super().__init__()
        self.cfg = g4d_cfg
        self.n_theta = n_params_g4d(g4d_cfg.n_gauss, g4d_cfg.mode)
        self.theta = nn.Linear(int(hidden), self.n_theta)
        nn.init.zeros_(self.theta.weight)              # 末层零权重
        with torch.no_grad():                          # bias = 恒等 init
            self.theta.bias.copy_(identity_theta(g4d_cfg))
        self.carrier = Glut4DCarrier(g4d_cfg)
        self.point_chunk = int(point_chunk)
        self.invalid_theta_count = 0

    def forward(self, cond_h: torch.Tensor, z: torch.Tensor,
                s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`cond_h` (B,H)、`z` (B,P,3)、`s` (B,) -> `(h_hat (B,P,3), theta (B,1308))`。

        非有限的 θ 计数并**停机**（不静默）：全零/异常参数必须计数并排除。
        """
        theta = self.theta(cond_h)
        bad = int((~torch.isfinite(theta)).any(dim=-1).sum())
        if bad:
            self.invalid_theta_count += bad
            die(f"G4D θ 出现非有限值：本批 {bad} 个样本（累计 "
                f"{self.invalid_theta_count}）—— 全零/异常参数必须计数并排除，禁静默")
        params = unpack_theta(theta, self.cfg)
        c = self.carrier(z, s, params, point_chunk=self.point_chunk)
        return c - z, theta
