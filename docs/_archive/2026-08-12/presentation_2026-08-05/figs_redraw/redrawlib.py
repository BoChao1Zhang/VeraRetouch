"""汇报配图重绘公用件 —— 空间场可视化纪律（CLAUDE.md「空间场可视化纪律」）的实现。

四条硬规则，本文件是它们唯一的落点：

1. **色标只取有效格**。`expand2square` 的补边格吃掉 RO-9b 场 52% 的（relu）质量、
   57% 的源全局 argmax 落在补边格；逐图 min-max 的分母被补边支配 ⇒ 有效区里的结构被
   压平。`valid_norm()` 的统计量只取 `valid16` 为真的格。
2. **补边格画白，不填补**。平滑用 `smooth_masked()`（补边格设 NaN，既不参与邻域统计、
   也不被邻域填回）；出图时凡"最近格是补边格"的像素一律涂白（`paint_pad_white=True`）。
   注：补边格在原图坐标里**不全在画面外**——覆盖率 <0.5 的边缘格被判无效但仍有部分
   像素落在画面内，所以必须显式涂白，不能指望裁剪替我们做掉。
3. **叠图用 `grid_to_img` 严格逆映射，禁止直接 resize**。6-09 原脚本把 16 格直接
   resize 到**未 pad** 的图上，而网格覆盖的是 pad 后的方形 ⇒ 约 1/3 的系统性错位。
4. **需要对比的两个场共用同一把色标**（`valid_norm()` 一次吃进多个场）。

⚑ 着色归着色、算数归算数：本文件只负责着色；任何进入正文/NOTES 的数字都由调用方
   在**未归一化的原始场**上算（`metrics.py` 口径，纯秩次，不经过这里的任何归一化）。
"""
from __future__ import annotations

import sqlite3  # noqa: F401  isort:skip  ⚑ 必须早于科学栈（战役 bug R6）
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "tools" / "readout"))

GRID = 16
SHORT_SIDE = 512          # 与模型输入同一条路径：短边 512 LANCZOS（不 pad）
BLEND = 0.74              # 热力图与**去饱和**底图的混合系数
CMAP = "inferno"


# --------------------------------------------------------------------------- 图像
def open_short512(img_path: str):
    """与模型输入同一条路径：短边 512 LANCZOS（不 pad，保留原始长宽比）。"""
    from PIL import Image

    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    sc = SHORT_SIDE / min(w, h)
    return im.resize((int(round(w * sc)), int(round(h * sc))), Image.Resampling.LANCZOS)


def luma_valid(img_path: str):
    """从源图重算 (luma16, valid16)，对齐 expand2square pad 语义。

    旧批次 stacks 里的 luma16 是 center-crop 口径（错），一律以此为准
    （analyze_g1.luma_valid_from_image 同款）。"""
    from ro9_gl_attention import luma_to_grid

    im = open_short512(img_path)
    return luma_to_grid(np.asarray(im.convert("L"), dtype=np.float32) / 255.0)


# --------------------------------------------------------------------------- 几何
def _geom(img):
    """(side, n, top, left)：expand2square + 16 格网格的几何量。"""
    w, h = img.size
    side = max(w, h)
    n = side // GRID
    top = (side - h) // 2 if w >= h else 0
    left = (side - w) // 2 if h > w else 0
    return side, n, top, left


def grid_to_img(m16: np.ndarray, img) -> np.ndarray:
    """(16,16) → 与 `img` 逐像素对齐的 (h,w)。**`luma_to_grid` 的严格逆映射。**

    网格 → n*16 方形（BICUBIC）→ 补到 side → 裁掉 pad → resize 回 img 尺寸。
    ⚠️ 禁止用 `Image.resize(img.size)` 直接从 16 格拉到未 pad 的图（6-09 的错位来源）。
    """
    from PIL import Image

    w, h = img.size
    side, n, top, left = _geom(img)
    big = np.asarray(Image.fromarray(m16.astype(np.float32)).resize(
        (n * GRID, n * GRID), Image.BICUBIC), dtype=np.float32)
    sq = np.full((side, side), float(np.nanmin(big)), dtype=np.float32)
    sq[: n * GRID, : n * GRID] = big                     # luma_to_grid 的截尾同款
    crop = sq[top:top + h, left:left + w]
    return np.asarray(Image.fromarray(crop).resize((w, h), Image.BICUBIC), dtype=np.float32)


def cell_index_map(img):
    """(h,w) 整数图：每个像素归属的 16×16 格线性下标；截尾残带记 -1。

    与 `grid_to_img` **同一条几何**，但走最近格（不插值）——专给"哪些像素属于补边格"
    这个 0/1 判断用，避免 BICUBIC 把补边格的值糊进有效区。
    """
    w, h = img.size
    side, n, top, left = _geom(img)
    ys = np.arange(h) + top
    xs = np.arange(w) + left
    gy = np.where(ys < n * GRID, ys // max(n, 1), -1)
    gx = np.where(xs < n * GRID, xs // max(n, 1), -1)
    gy = np.clip(gy, -1, GRID - 1)
    gx = np.clip(gx, -1, GRID - 1)
    idx = gy[:, None] * GRID + gx[None, :]
    idx[(gy[:, None] < 0) | (gx[None, :] < 0)] = -1
    return idx


# --------------------------------------------------------------------------- 着色
def smooth_masked(m: np.ndarray, keep: np.ndarray, k: int = 1) -> np.ndarray:
    """3×3 均值平滑，只用 `keep` 格参与邻域统计（补边格既不参与、也不被填补）。

    逐字复用 `RO9c/diag_raw_colormap.smooth_masked`。"""
    a = m.astype(np.float64).copy()
    a[~keep] = np.nan
    pad = np.pad(a, k, constant_values=np.nan)
    acc, cnt = np.zeros_like(a), np.zeros_like(a)
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            w = pad[k + dy:k + dy + a.shape[0], k + dx:k + dx + a.shape[1]]
            ok = np.isfinite(w)
            acc += np.where(ok, w, 0.0)
            cnt += ok
    with np.errstate(all="ignore"):
        out = acc / np.maximum(cnt, 1)
    out[cnt == 0] = np.nan
    return out


def valid_norm(fields, valid: np.ndarray, smooth: bool = True):
    """多个场**共用一把色标**：先各自 masked-smooth，再在**有效格**上取联合 min/max。

    返回 (list[已平滑场], lo, hi)。**不做 relu**——本项目的 s 场是 pre-softmax logit，
    实测 [−13.4, +1.3]、>0 的格不足 4%，relu 会把整张图压成空白（G3/RO-9c 已踩过）。
    """
    sm = [smooth_masked(f, valid, 1) if smooth else np.where(valid, f, np.nan)
          for f in fields]
    stat = np.concatenate([s[valid].ravel() for s in sm])
    lo, hi = float(np.nanmin(stat)), float(np.nanmax(stat))
    if not np.isfinite(hi - lo) or hi - lo < 1e-12:
        hi = lo + 1e-12
    return sm, lo, hi


def _pad_pixels(valid: np.ndarray, img) -> np.ndarray:
    """(h,w) bool：该像素最近的 16×16 格是补边格（或落在截尾残带）。"""
    idx = cell_index_map(img)
    vflat = valid.ravel()
    return (idx < 0) | ~vflat[np.clip(idx, 0, GRID * GRID - 1)]


def _gray_rgb(img) -> np.ndarray:
    """去饱和底图：热力图叠上去之后，画面里的**颜色只来自场**，不来自照片本身。"""
    g = np.asarray(img.convert("L"), dtype=np.float32)
    return np.repeat((60.0 + g * 0.62)[..., None], 3, axis=2)


def heat_over_image(field_sm: np.ndarray, valid: np.ndarray, img, lo: float, hi: float,
                    cmap: str = CMAP, blend: float = BLEND):
    """已平滑场 → 叠在去饱和底图上的 RGB uint8。补边格所属像素涂白。"""
    import matplotlib.cm as cm
    from PIL import Image

    a = grid_to_img(np.nan_to_num(field_sm, nan=lo).astype(np.float32), img)
    a = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    heat = cm.get_cmap(cmap)(a)[..., :3] * 255.0
    out = _gray_rgb(img) * (1.0 - blend) + heat * blend
    out[_pad_pixels(valid, img)] = 255.0                # 补边格 → 白，不填补
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def region_over_image(region16: np.ndarray, valid: np.ndarray, img, dim: float = 0.34):
    """目标区域 / 真值掩膜叠图：区域内保留原色，区域外压暗去色。补边格涂白。

    ⚑ 不画轮廓线（评审 2026-08-05：热力图上不叠加任何线）。
    """
    from PIL import Image

    m = np.clip(grid_to_img(np.clip(region16, 0, 1).astype(np.float32), img),
                0.0, 1.0)[..., None]
    base = np.asarray(img.convert("RGB"), dtype=np.float32)
    g = np.asarray(img.convert("L"), dtype=np.float32)[..., None]
    out = (g * dim) * (1.0 - m) + base * m
    out[_pad_pixels(valid, img)] = 255.0
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def plain_image(img, valid: np.ndarray = None):
    """原图列。给了 valid 就一并把补边格涂白，保证五列的可视范围完全一致。"""
    from PIL import Image

    out = np.asarray(img.convert("RGB"), dtype=np.float32).copy()
    if valid is not None:
        out[_pad_pixels(valid, img)] = 255.0
    return Image.fromarray(out.astype(np.uint8))


# --------------------------------------------------------------------------- 版式
def cjk_font(size: int):
    from PIL import ImageFont

    for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
              "/home/bc/.local/share/fonts/windows/NotoSansSC-VF.ttf",
              "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"):
        if Path(p).is_file():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def row_figure(tiles, titles, out_path: Path, short_side: int = 460,
               cmap: str = CMAP, gap: int = 14, margin: int = 12,
               title_h: int = 62, bar_w: int = 40):
    """单行联图：列标题一行 + N 张等高瓦片 + 一根无刻度色标条。**图内无其他文字。**

    每格短边 ≥ `short_side`（默认 460 ≥ 判据的 400）。
    """
    import matplotlib.cm as cm
    from PIL import Image, ImageDraw

    fitted = []
    for t in tiles:
        w, h = t.size
        sc = short_side / min(w, h)
        fitted.append(t.resize((int(round(w * sc)), int(round(h * sc))),
                               Image.Resampling.LANCZOS))
    H = max(t.size[1] for t in fitted)
    xs, x = [], margin
    for t in fitted:
        xs.append(x)
        x += t.size[0] + gap
    bar_x = x + 10
    W = bar_x + bar_w + 34 + margin
    panel = Image.new("RGB", (W, title_h + H + margin), (255, 255, 255))
    d = ImageDraw.Draw(panel)
    font = cjk_font(38)
    for t, xx, tt in zip(fitted, xs, titles):
        panel.paste(t, (xx, title_h))
        try:
            bb = d.textbbox((0, 0), tt, font=font)
            tw = bb[2] - bb[0]
        except Exception:
            tw = 20 * len(tt)
        d.text((xx + max(0, (t.size[0] - tw) // 2), title_h - 52), tt,
               fill=(20, 20, 20), font=font)
    # 色标条（无刻度、无数字；低在下、高在上）
    grad = np.linspace(1.0, 0.0, H)[:, None].repeat(bar_w, axis=1)
    bar = (cm.get_cmap(cmap)(grad)[..., :3] * 255).astype(np.uint8)
    panel.paste(Image.fromarray(bar), (bar_x, title_h))
    d.rectangle([bar_x, title_h, bar_x + bar_w - 1, title_h + H - 1], outline=(120, 120, 120))
    f2 = cjk_font(26)
    d.text((bar_x + bar_w + 6, title_h), "高", fill=(20, 20, 20), font=f2)
    d.text((bar_x + bar_w + 6, title_h + H - 30), "低", fill=(20, 20, 20), font=f2)
    panel.save(out_path)
    return panel
