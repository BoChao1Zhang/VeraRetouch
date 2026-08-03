"""s-cache 读写 API（INF-5）。

约定（EXPERIMENTS_v3 §INF-5 + 任务卡 T5）：

    <root>/<arm>/<img_id>__<instr_hash>.npy        # float16, 默认 32x32（可配原生分辨率）
    <root>/<arm>/<img_id>__<instr_hash>.meta.json  # 层号 / 归一化参数 / 生成时间 / arm 版本 / 溯源

- 读出臂只管往自己的 arm 目录写；渲染臂只读缓存目录 → 两臂完全解耦，
  交叉组合 = 换一个 arm 目录名。
- meta 必含字段: img_id, instr_hash, shape, dtype, layer, norm, created_at, arm, arm_version。
  norm 约定为 dict，例如 {"kind": "linear", "domain": [0, 1]}；layer 为读出层号（oracle 为 None）。

零第三方依赖（numpy 之外），不依赖 torch。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence, Tuple, Union

import numpy as np

DEFAULT_RESOLUTION = 32          # 默认 32x32（EXPERIMENTS_v3: 约 2KB/图）
NO_INSTR = "noinstr"             # 无指令条目的 instr_hash 占位
_KEY_SEP = "__"
_SAFE_RE = re.compile(r"[^A-Za-z0-9_.\-]")


def instr_hash(instruction: Optional[str], n: int = 12) -> str:
    """指令文本 -> 稳定短哈希（md5 hex 前 n 位）；空/None -> 'noinstr'。"""
    if instruction is None or instruction == "":
        return NO_INSTR
    return hashlib.md5(instruction.encode("utf-8")).hexdigest()[:n]


def cache_key(img_id: str, ihash: str) -> str:
    if _KEY_SEP in img_id:
        raise ValueError(f"img_id 不得含 '{_KEY_SEP}': {img_id!r}")
    img_id = _SAFE_RE.sub("-", img_id)
    ihash = _SAFE_RE.sub("-", ihash)
    return f"{img_id}{_KEY_SEP}{ihash}"


def split_key(key: str) -> Tuple[str, str]:
    img_id, sep, ihash = key.rpartition(_KEY_SEP)
    if not sep or not img_id:
        raise ValueError(f"非法缓存键: {key!r}")
    return img_id, ihash


@dataclass
class SEntry:
    """一条 s 缓存：s 场数组 + 元数据。"""

    img_id: str
    instr_hash: str
    s: np.ndarray                      # float16, (H, W)
    meta: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return cache_key(self.img_id, self.instr_hash)


class SCache:
    """单臂 s 缓存目录的读写句柄。

    Parameters
    ----------
    root : 缓存根目录（含各 arm 子目录）。不提供默认值——落位是数据治理决策。
    arm  : 臂名（如 'oracle', 'ro1-l17'）。
    resolution : int | (H, W) | None。None = 原生分辨率（不校验形状，逐条目自记 shape）。
    arm_version : 写入 meta 的 arm 版本串。
    """

    def __init__(
        self,
        root: Union[str, Path],
        arm: str,
        resolution: Union[int, Tuple[int, int], None] = DEFAULT_RESOLUTION,
        arm_version: str = "v0",
    ) -> None:
        if not arm or "/" in arm:
            raise ValueError(f"非法 arm 名: {arm!r}")
        self.root = Path(root)
        self.arm = arm
        self.dir = self.root / arm
        if resolution is None:
            self.resolution: Optional[Tuple[int, int]] = None
        elif isinstance(resolution, int):
            self.resolution = (resolution, resolution)
        else:
            self.resolution = (int(resolution[0]), int(resolution[1]))
        self.arm_version = arm_version

    # ---------- 路径 ----------

    def paths_for(self, img_id: str, ihash: str) -> Tuple[Path, Path]:
        key = cache_key(img_id, ihash)
        return self.dir / f"{key}.npy", self.dir / f"{key}.meta.json"

    def exists(self, img_id: str, ihash: str) -> bool:
        npy, meta = self.paths_for(img_id, ihash)
        return npy.exists() and meta.exists()

    # ---------- 写 ----------

    def write(
        self,
        img_id: str,
        ihash: str,
        s: np.ndarray,
        *,
        layer: Optional[int] = None,
        norm: Optional[dict] = None,
        extra_meta: Optional[dict] = None,
        overwrite: bool = True,
    ) -> Path:
        """写一条缓存（npy + meta.json，均原子落盘）。

        s: 二维数组，将转 float16。resolution 非 None 时校验形状。
        """
        s = np.asarray(s)
        if s.ndim != 2:
            raise ValueError(f"s 场须为二维 (H,W)，得到 shape={s.shape}")
        if self.resolution is not None and tuple(s.shape) != self.resolution:
            raise ValueError(f"s 形状 {s.shape} != 本臂约定 {self.resolution}")
        if not np.isfinite(s).all():
            raise ValueError(f"s 含 NaN/Inf: {img_id}__{ihash}")
        s16 = s.astype(np.float16)

        npy_path, meta_path = self.paths_for(img_id, ihash)
        if not overwrite and npy_path.exists():
            raise FileExistsError(str(npy_path))
        self.dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "img_id": img_id,
            "instr_hash": ihash,
            "shape": list(s16.shape),
            "dtype": "float16",
            "layer": layer,
            "norm": norm if norm is not None else {"kind": "linear", "domain": [0.0, 1.0]},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "arm": self.arm,
            "arm_version": self.arm_version,
        }
        if extra_meta:
            for k in extra_meta:
                if k in meta:
                    raise ValueError(f"extra_meta 不得覆盖必含字段: {k}")
            meta.update(extra_meta)

        self._atomic_write(npy_path, lambda f: np.save(f, s16))
        self._atomic_write(
            meta_path,
            lambda f: f.write(json.dumps(meta, ensure_ascii=False, indent=1).encode("utf-8")),
        )
        return npy_path

    @staticmethod
    def _atomic_write(path: Path, writer) -> None:
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=path.suffix)
        try:
            with os.fdopen(fd, "wb") as f:
                writer(f)
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    # ---------- 读 ----------

    def read(self, img_id: str, ihash: str) -> SEntry:
        npy_path, meta_path = self.paths_for(img_id, ihash)
        s = np.load(npy_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return SEntry(img_id=img_id, instr_hash=ihash, s=s, meta=meta)

    def keys(self) -> Iterator[Tuple[str, str]]:
        """迭代 (img_id, instr_hash)，按文件名排序（确定序）。"""
        if not self.dir.is_dir():
            return
        for p in sorted(self.dir.glob("*.npy")):
            yield split_key(p.stem)

    def __len__(self) -> int:
        return sum(1 for _ in self.keys())

    def iter_entries(self) -> Iterator[SEntry]:
        """逐条迭代全部缓存条目（确定序）。"""
        for img_id, ihash in self.keys():
            yield self.read(img_id, ihash)

    def iter_batches(self, batch_size: int) -> Iterator[Sequence[SEntry]]:
        """批量迭代：yield SEntry 列表（末批可短）。"""
        if batch_size <= 0:
            raise ValueError("batch_size 须为正")
        buf: list[SEntry] = []
        for e in self.iter_entries():
            buf.append(e)
            if len(buf) == batch_size:
                yield buf
                buf = []
        if buf:
            yield buf

    def read_stack(self, keys: Iterable[Tuple[str, str]]) -> np.ndarray:
        """按给定键序读出并堆叠为 (N, H, W) float16（要求同形状）。"""
        arrs = [self.read(i, h).s for i, h in keys]
        return np.stack(arrs, axis=0)
