"""veraretouch_sprf 路径解析（EPR-052 扶正时新增；替代原 sprf 各文件顶部的 sys.path 注入块）。

原件（experiments/prs/EPR-051.../stage0/sprf/*.py）用 `REPO = Path("/home/bc/VeraRetouch")` +
`sys.path.insert` 把仓库根 / dataset_build/tools / stage0 / sprf 塞进 sys.path，再裸 import 同目录模块。
主线包改为包内绝对导入；仍需仓库根（`q3vl.*`）与 `dataset_build/tools`（`epr050_build_degradation`）
在 sys.path 上，由 `ensure_sys_path()` 负责（包 `__init__` 导入时调用一次）。

`src(name)`：按原文件名找源文件（供各入口的 provenance sha256 记录 / K4 钉死表用）；
先找主线包内的同名文件（含改名映射），找不到再回落到 legacy sprf 目录（未扶正的历史文件）。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent
REPO = Path(os.environ.get("VR_REPO", PKG.parent)).resolve()
TOOLS = REPO / "dataset_build" / "tools"
EPR051 = REPO / "experiments" / "prs" / "EPR-051_masked-restore-production"
STAGE0 = EPR051 / "stage0"
SPRF_LEGACY = STAGE0 / "sprf"
VLMSFT_LEGACY = SPRF_LEGACY / "vlmsft"

# 改名的文件：legacy 名 -> 主线包相对路径
RENAMED = {
    "train_q3vl_sft3.py": "train/train_vlm_sft.py",
    "train_q3vl_adapt3.py": "train/train_vlm_adapt.py",
    "batch_eval_bk.py": "eval/eval_decoder.py",
    "epr051_heldout_split.py": "scripts/heldout_split.py",
    "epr051_merge_gencache.py": "scripts/merge_gencache.py",
}


def ensure_sys_path() -> None:
    for p in (str(REPO), str(TOOLS)):
        if p not in sys.path:
            sys.path.insert(0, p)


def _pkg_index() -> dict[str, Path]:
    idx: dict[str, Path] = {}
    for sub in ("data", "models", "models/vlm", "solver", "train", "eval", "eval/probes", "scripts"):
        d = PKG / sub
        if d.is_dir():
            for f in d.glob("*.py"):
                idx.setdefault(f.name, f)
    for old, new in RENAMED.items():
        p = PKG / new
        if p.exists():
            idx[old] = p
    return idx


def src(name: str) -> Path:
    """原文件名 -> 主线包内文件；不在包内的历史文件回落到 legacy 目录。"""
    p = _pkg_index().get(name)
    if p is not None:
        return p
    for d in (SPRF_LEGACY, VLMSFT_LEGACY, STAGE0):
        q = d / name
        if q.exists():
            return q
    raise FileNotFoundError(f"src({name!r}): 主线包与 legacy 目录均无此文件")
