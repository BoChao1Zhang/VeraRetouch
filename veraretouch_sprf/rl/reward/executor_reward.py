"""执行器奖励骨架（EPR-052）：key + 生成文本 → 读出 → 注入 BK-FULL → rollout → linf8。

全部数值路径复用 eval 线原件（不另写口径）：
  读出       eval/dump_readout.readout_from_generated（adapter s2 + span_pool）
  执行器装载 models/bk_load.load_bk_model（run_args → 指纹校验 → ckpt）
  注入       eval/eval_vlmadapt.main 内的做法：edit_descriptor := identity、cond.edit_enc := nn.Identity、
             contract := predicted_lut，edits = lat.flip(0)[None]（slot→chain），SS.rollout_with_metrics
  指标       data/train_stage0.linf8(x0, x̂)
当前为 **stub**：函数签名与数据流已定，主体 `NotImplementedError`，待调研确认奖励形式后接线。
导入测试可通过（不在模块级加载任何模型）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class ExecutorContext:
    """一次进程内常驻的执行器上下文（模型 + 逆表 + 目标表 + 特征缓存 + 样本索引）。"""
    device: str
    model: Any                      # bk_load.load_bk_model 返回的 BkSprfModel（eval、无梯度）
    real_edit_enc: Any              # model.bk.core.edit_enc（注入时被 Identity 替换，A-inj 用它复原）
    inv: Any                        # edit_cond.InvLutSource
    cfg: dict                       # run_args["config"]
    n_steps: int
    samples: list = field(default_factory=list)   # T0.load_shards 的样本行
    blobs: dict = field(default_factory=dict)
    feats: dict = field(default_factory=dict)     # pooled SigLIP c，按 T0.feature_key
    targets: torch.Tensor | None = None           # AP.build_target_latents(edit_enc, inv_table)，行号序
    n_pix: int = 100000


def load_executor_context(backbone_run: str | Path, ckpt: str = "ckpt_last.pt",
                          feats_cache: str | Path = "", heldout_ids: str | Path = "",
                          device: str = "cuda:0", n_pix: int = 100000) -> ExecutorContext:
    """镜像 eval/eval_vlmadapt.main 的装载段（bk_load → ST.install_compact_row → load_shards → bind_build_config
    → feature cache → build_target_latents）。待接线。"""
    raise NotImplementedError("EPR-052 骨架：待调研确认后接线（复用 eval_vlmadapt.main 113–215 行的装载序列）")


@torch.no_grad()
def readout_latents(vlm, proc, enc, gen_ids: list[int], device: str, dtype, stage_ids: list[int],
                    include_stage_token: bool = False) -> tuple[torch.Tensor, list]:
    """-> ((6,128) slot 序读出向量, missing stage tokens)。直接转调 dump_readout.readout_from_generated。"""
    from veraretouch_sprf.eval import dump_readout as D
    z, missing = D.readout_from_generated(vlm, proc, enc, gen_ids, device, dtype,
                                          include_stage_token, stage_ids)
    return z, missing


@torch.no_grad()
def inject_and_rollout(ctx: ExecutorContext, key: str, lat_slot: torch.Tensor) -> torch.Tensor:
    """把 (6,128) slot 序向量注入 BK-FULL 并 rollout，返回 x̂ (1,P,3)。

    注入约定（与 eval_vlmadapt 逐字同）：edit_descriptor := identity；cond.edit_enc := nn.Identity；
    edits = lat_slot.flip(0)[None]（chain 序）；A-inj：同一向量走 oracle_lut 路径须逐位相等。待接线。"""
    raise NotImplementedError("EPR-052 骨架：待接线（eval_vlmadapt.run_roll / _batch_of）")


def linf8(ctx: ExecutorContext, key: str, x_hat: torch.Tensor) -> float:
    """T0.linf8(x0_pixels, x_hat) —— 与 heldout headline 同一函数。待接线。"""
    raise NotImplementedError("EPR-052 骨架：待接线（train_stage0.linf8）")


def executor_reward(ctx: ExecutorContext, key: str, lat_slot: torch.Tensor, missing: list,
                    mode: str = "neg_linf8", missing_penalty: float | None = None) -> dict:
    """-> {reward, linf8, n_missing}. mode ∈ {neg_linf8, identity_gain}（待调研确认）。"""
    if missing and missing_penalty is not None:
        return dict(reward=float(missing_penalty), linf8=None, n_missing=len(missing))
    x_hat = inject_and_rollout(ctx, key, lat_slot)
    e = linf8(ctx, key, x_hat)
    if mode == "neg_linf8":
        r = -e
    elif mode == "identity_gain":
        raise NotImplementedError("identity_gain 需要 identity 行的 linf8（T0.linf8(y, x0)），待接线")
    else:
        raise ValueError(mode)
    return dict(reward=float(r), linf8=float(e), n_missing=len(missing or []))
