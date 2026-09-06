"""读出向量对预建目标 e* 的余弦奖励（EPR-052 骨架，可直接用）。

e*：`data/build_targets_bkfull.py` 用 BK-FULL edit_enc 预建（train/val/heldout），文件 target_latents_bkfull.pt，
    形状约定与 train_vlm_adapt 相同：按 key 取 (6, 128)，**slot 序**（m=1..6 恢复序；chain k = 6 − m）。
读出向量：`eval/dump_readout.readout_from_generated` 的输出 (6, 128)，同为 slot 序 —— 余弦在同一序下比，
不做 flip（flip 只在注入求解器时做）。
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F


def load_target_table(path: str | Path):
    """-> torch.load 的原始 blob（含 keys / latents 等；字段名以 build_targets_bkfull 落盘为准）。"""
    return torch.load(Path(path), map_location="cpu")


def target_for_key(blob, key: str) -> torch.Tensor:
    """按 key 取 (6,128) slot 序目标。字段名按 build_targets_bkfull.py 的落盘格式（待接线时核对）。"""
    if "by_key" in blob:
        return blob["by_key"][key]
    idx = blob["keys"].index(key)
    return blob["latents"][idx]


def cosine_reward(lat_pred: torch.Tensor, e_star: torch.Tensor,
                  missing: list | None = None, missing_penalty: float = -1.0) -> dict:
    """lat_pred / e_star: (6,128) slot 序。-> {reward, per_stage (6,), n_missing}.

    缺阶段（missing 非空，读出为 None/零向量）的阶段余弦记 missing_penalty（待调研确认）。
    """
    lat_pred = lat_pred.float(); e_star = e_star.float()
    cs = F.cosine_similarity(lat_pred, e_star, dim=-1)          # (6,)
    n_missing = 0
    if missing:
        for m in missing:
            i = int(str(m).strip("<>").rsplit("_", 1)[-1]) - 1 if not isinstance(m, int) else int(m)
            if 0 <= i < cs.numel():
                cs[i] = missing_penalty
                n_missing += 1
    return dict(reward=float(cs.mean()), per_stage=cs, n_missing=n_missing)
