"""G2: 按索引行取 (I_in, I_tar, s) 三元组，像素严格对齐。

关键点（已核实 tools/bgr_check/common.py 的生产复刻）：生产渲染输入 =
`preprocess_source_bytes(源图字节, short_edge=1024)`（EXIF 校正 + LANCZOS 缩到短边
1024），归档 after `.jpg` 与 `.cgt.png` 均在该分辨率。因此 I_in **必须**从 img 银行
源图按同一管线重建，不能用 shard 里的 `.in.jpg`（那是 VLM 预览图，尺寸任意，
与 after 不同分辨率 → 逐像素不可比）。
"""

from __future__ import annotations

import io
import sys
from dataclasses import dataclass

import numpy as np
from PIL import Image

sys.path.insert(0, "/home/bc/VeraRetouch/tools/bgr_check")
import common as C  # noqa: E402

SHORT_EDGE = 1024


def _read_ref(ref: dict) -> bytes:
    if ref.get("kind") == "local":
        with open(ref["path"], "rb") as f:
            return f.read()
    tar = ref.get("tar")
    return C.read_member(str(tar), int(ref["offset_data"]), int(ref["length"]))


def load_source_rgb(row: dict) -> np.ndarray:
    """源图 → 生产渲染输入，float32 (H,W,3) ∈ [0,1]。"""
    return C.preprocess_source_bytes(_read_ref(row["source_ref"]), SHORT_EDGE)


def load_after_rgb(row: dict) -> np.ndarray:
    img = Image.open(io.BytesIO(_read_ref(row["tar_ref"]))).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def load_mask(row: dict) -> np.ndarray | None:
    if not row.get("mask_ref"):
        return None
    img = Image.open(io.BytesIO(_read_ref(row["mask_ref"])))
    if img.mode != "L":
        img = img.convert("L")
    return np.asarray(img, dtype=np.float32) / 255.0


@dataclass
class Triplet:
    x: np.ndarray          # (P,3) float32 [0,1]
    y: np.ndarray          # (P,3) float32 [0,1]
    s: np.ndarray          # (P,)  float32 [0,1]
    hw: tuple[int, int]
    stride: int
    id: str


def load_triplet(row: dict, max_pixels: int | None = None,
                 donor_mask: np.ndarray | None = None) -> Triplet | None:
    """尺寸不一致直接返回 None（记为 skip，不做插值对齐——插值会污染天花板）。

    donor_mask: 供 D-SFT-G 对照档使用的移植掩膜（(h,w) float32），会最近邻
    重采样到本图尺寸；此时忽略 row 自带 mask_ref。
    """
    x = load_source_rgb(row)
    y = load_after_rgb(row)
    if x.shape[:2] != y.shape[:2]:
        return None
    if donor_mask is not None:
        m = _resize_nn(donor_mask, x.shape[:2])
    else:
        m = load_mask(row)
        if m is None:
            m = np.ones(x.shape[:2], dtype=np.float32)
        elif m.shape[:2] != x.shape[:2]:
            return None

    stride = 1
    h, w = x.shape[:2]
    if max_pixels is not None and h * w > max_pixels:
        stride = int(np.ceil(np.sqrt(h * w / max_pixels)))
        x, y, m = x[::stride, ::stride], y[::stride, ::stride], m[::stride, ::stride]
    return Triplet(
        x=np.ascontiguousarray(x.reshape(-1, 3)),
        y=np.ascontiguousarray(y.reshape(-1, 3)),
        s=np.ascontiguousarray(m.reshape(-1)),
        hw=(h, w), stride=stride, id=row["candidate_id"])


def _resize_nn(m: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """最近邻重采样（掩膜移植用；不引入新的中间灰度）。"""
    h, w = hw
    yi = (np.arange(h) * (m.shape[0] / h)).astype(np.int64).clip(0, m.shape[0] - 1)
    xi = (np.arange(w) * (m.shape[1] / w)).astype(np.int64).clip(0, m.shape[1] - 1)
    return m[yi][:, xi]
