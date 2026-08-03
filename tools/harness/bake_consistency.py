"""INF-1 烘焙一致性评测器 (wave-1.5 F2, 清 REVIEW-impl-wave1 blocker B2).

规格 (EXPERIMENTS_v3 §1 INF-1「烘焙一致性 (四面体插值回读)」; CLAUDE.md 红线
「烘焙一致性从第一天当一等指标」):
- 接口: render_fn(rgb) -> rgb, 任意可微渲染器的逐点求值
  ((...,3) float RGB ∈ [0,1] -> 同形状).
- 流程: 均匀 33³ 格点采样渲染 -> 组装 (33,33,33,3) 3D LUT (T2 canonical 约定,
  index [r,g,b]) -> colour 四面体插值回读 (复用 tools/cube 的应用器
  apply_lut_tetrahedral —— 即 hald.py 的 GT 路应用器, 不重写).
- 评测面: 128³ 留出色 {1,3,...,255}³ (全通道奇数 => 与 GLUT 训练色 {0,2,...,254}³
  严格互斥, 全部属于 T2 hald eval 留出集), 排成 1024x2048x3; 外加可选自然图.
- 指标: 直渲 vs 烘焙回读的逐像素 ΔE00 (skimage rgb2lab D65 -> deltaE_ciede2000,
  与 metrics.delta_e00 同链) 的 p50/p90/p99 (附 mean/max); PSNR 差 (round×255 口径,
  metrics.psnr_full(direct, baked)); 最大逐通道绝对误差.
- 判据 (常量写死): 每个评测面 ΔE00 p99 < 0.5 为过; 总判 passed = 全部评测面过.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_CUBE = str(_HERE.parent / "cube")
if _CUBE not in sys.path:
    sys.path.append(_CUBE)  # append 而非 insert: 避免 cube/selfcheck.py 等同名遮蔽

import metrics
# 跨目录运行时 import（上方 sys.path.append）；静态解析由仓库根 pyrightconfig.json
# 的 executionEnvironments(tools/harness → extraPaths tools/cube) 统一解决（F7-3）。
from cubelib import apply_lut_tetrahedral, identity_table

DE00_P99_PASS = 0.5  # 判据常量 (写死): ΔE00 p99 < 0.5 为过
GRID_SIZE = 33  # 烘焙格密度 (INF-1 规格 33³)
HELDOUT_SHAPE = (1024, 2048, 3)  # 128³ = 2,097,152 留出色

RenderFn = Callable[[np.ndarray], np.ndarray]


def held_out_image() -> np.ndarray:
    """128³ 留出色 {1,3,...,255}³ 排成 1024x2048x3 float32, R 最快 (对齐 hald 约定)."""
    b, g, r = np.mgrid[1:256:2, 1:256:2, 1:256:2]
    colors = np.stack([r, g, b], axis=-1).reshape(-1, 3)
    return (colors.astype(np.float32) / 255.0).reshape(HELDOUT_SHAPE)


def bake_lut(render_fn: RenderFn, grid_size: int = GRID_SIZE) -> np.ndarray:
    """均匀 grid_size³ 格点采样 render_fn -> (S,S,S,3) float32 LUT.

    逐点采样保持 identity_table 的 canonical [r,g,b] 索引约定:
    lut[i,j,k] = render(identity[i,j,k]).
    """
    grid = identity_table(grid_size).astype(np.float64)
    lut = np.asarray(render_fn(grid), dtype=np.float64)
    if lut.shape != grid.shape:
        raise ValueError(f"render_fn output shape {lut.shape} != grid {grid.shape}")
    return lut.astype(np.float32)


def _surface_stats(direct: np.ndarray, baked: np.ndarray) -> dict:
    """单评测面: ΔE00 分位 + PSNR 差 + 最大逐通道绝对误差 + 判据."""
    a, b = metrics.to_float01(direct), metrics.to_float01(baked)
    de = np.asarray(deltaE_ciede2000(rgb2lab(a), rgb2lab(b)), dtype=np.float64)
    p50, p90, p99 = (float(np.percentile(de, q)) for q in (50.0, 90.0, 99.0))
    return {
        "de00_p50": p50,
        "de00_p90": p90,
        "de00_p99": p99,
        "de00_mean": float(de.mean()),
        "de00_max": float(de.max()),
        "max_abs_err": float(np.max(np.abs(a - b))),
        "psnr_direct_vs_baked": metrics.psnr_full(direct, baked),
        "passed": bool(p99 < DE00_P99_PASS),
    }


def bake_consistency(
    render_fn: RenderFn,
    natural_images: Sequence[np.ndarray] | None = None,
    grid_size: int = GRID_SIZE,
) -> dict:
    """烘焙一致性评测 (INF-1 一等指标). 返回结构见模块 docstring.

    render_fn 被调用于: grid_size³ 格点 (烘焙), 128³ 留出色与每张自然图 (直渲参照).
    """
    lut = bake_lut(render_fn, grid_size)
    held = held_out_image()
    direct = np.asarray(render_fn(held.astype(np.float64)), dtype=np.float64)
    baked = apply_lut_tetrahedral(lut, held)
    result: dict = {
        "grid_size": grid_size,
        "n_heldout": HELDOUT_SHAPE[0] * HELDOUT_SHAPE[1],
        "de00_p99_threshold": DE00_P99_PASS,
        "heldout": _surface_stats(direct, baked),
        "natural": [],
    }
    for i, img in enumerate(natural_images or []):
        im = metrics.to_float01(img)
        if im.ndim != 3 or im.shape[-1] != 3:
            raise ValueError(f"natural image {i}: expect HxWx3, got {im.shape}")
        d = np.asarray(render_fn(im), dtype=np.float64)
        b_img = apply_lut_tetrahedral(lut, im.astype(np.float32))
        st = _surface_stats(d, b_img)
        st["id"] = i
        result["natural"].append(st)
    result["passed"] = bool(
        result["heldout"]["passed"] and all(s["passed"] for s in result["natural"])
    )
    return result
