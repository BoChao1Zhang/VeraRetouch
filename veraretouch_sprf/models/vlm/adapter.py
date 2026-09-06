# 源自 experiments/prs/EPR-051_masked-restore-production/stage0/sprf/vlmsft/train_q3vl_adapt3.py @ 2026-09-06（节选：class Adapter / readout_z_and_hidden / readout_z，逐字复制，仅补导入；原 sha256 见 PROVENANCE.md）。EPR-052。
"""S2F-B 读出 + adapter 定义（节选自 train_q3vl_adapt3.py；train/train_vlm_adapt.py 内仍保留同一份原定义）。

Adapter：z_m (2560) -> 128 维 LUT latent，六槽共享权重 + 槽嵌入（零初始化 ⇒ step-0 恒等）。
readout_z：Qwen3VLModel.last_hidden_state（已过 final norm）在第 m 段正文 token 上 span mean-pool。
dump_readout.py 原从历史文件 train_q3vl_adapt.py 导入 Adapter；两处定义逐字相同（已 diff 核对）。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from veraretouch_sprf.models.vlm import q3vl_common as Q
from veraretouch_sprf.data import q3vl_text as T


class Adapter(nn.Module):
    """z_m (2560) -> 128-d LUT latent, weights shared across the 6 slots."""

    def __init__(self, in_dim=2560, hidden=768, out_dim=128, n_slots=6):
        super().__init__()
        self.slot = nn.Embedding(n_slots, in_dim)
        nn.init.zeros_(self.slot.weight)          # identity at step 0
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim))

    def forward(self, z):                          # (B, 6, in_dim)
        k = torch.arange(z.shape[1], device=z.device)
        return self.net(z + self.slot(k).unsqueeze(0))


def readout_z_and_hidden(mm, batch, device, dtype):
    """Same contract as readout_z but also returns the hidden states, so the CE
    term costs no extra forward."""
    out = mm(input_ids=batch["input_ids"].to(device),
             attention_mask=batch["attention_mask"].to(device),
             pixel_values=batch["pixel_values"].to(device, dtype),
             image_grid_thw=batch["image_grid_thw"].to(device), return_dict=True)
    hidden = out.last_hidden_state.float()
    if hidden.shape[-1] != Q.Z_DIM:
        Q.die(f"h is {hidden.shape[-1]}-dim, contract says {Q.Z_DIM}")
    z = torch.stack([T.span_pool(hidden[i], batch["spans"][i])
                     for i in range(hidden.shape[0])], dim=0)
    return z, hidden


def readout_z(mm, batch, device, dtype):
    """(B, 6, 2560) fp32 span-pooled conditions, gradient intact.

    `mm` is the Qwen3VLModel.  Its `last_hidden_state` is the last decoder block's
    output ALREADY through the text tower's final norm, so this reproduces the
    EPR-033 contract (SEGMENT_HIDDEN_LAYER=-1, SEGMENT_HIDDEN_FINAL_NORM=True)
    exactly -- and NO further norm may be applied here.

    Unlike the forward-hook route this is an ordinary graph node, so gradient
    checkpointing is safe (A-9 is hook-specific), and lm_head is never invoked,
    so full-sequence vocab logits are never materialised.
    """
    out = mm(input_ids=batch["input_ids"].to(device),
             attention_mask=batch["attention_mask"].to(device),
             pixel_values=batch["pixel_values"].to(device, dtype),
             image_grid_thw=batch["image_grid_thw"].to(device),
             return_dict=True)
    hidden = out.last_hidden_state.float()
    if hidden.shape[-1] != Q.Z_DIM:
        Q.die(f"h is {hidden.shape[-1]}-dim, contract says {Q.Z_DIM}")
    return torch.stack([T.span_pool(hidden[i], batch["spans"][i])
                        for i in range(hidden.shape[0])], dim=0)
