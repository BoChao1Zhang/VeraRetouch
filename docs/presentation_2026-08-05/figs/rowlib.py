"""汇报单行配图公用库（2026-08-05，版式已按评审第二轮要求收紧）。

评审方硬约束：
  * **一页一图，一张图最多 1 行**；
  * **画面里只有图像内容和一行简短列标题**——不放图注、不放副标题、不放 source id、
    不放格内数值、不叠加真值轮廓线。所有说明改写进 `ROW_FIGURE_MAP.md`。

设计要点：
  * 每格短边 >= `CELL`（默认 440 px），列标题字号按投影距离设定；
  * 真值单独占一列，绝不往热力图 / 渲染图上叠线。

⚑ 本库只做版式，不碰任何实验数字：裁剪档逐像素搬运原图，重绘档只从已落盘的
   `features20.npz` / stacks 读数，不重跑推理、不重新归一化。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT_DIR = HERE.parent / "assets_row"

CELL = 440              # 每格短边（>= 400 的硬要求）
GAP = 8                 # 列间距
MARGIN = 20

_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
_FONT_BOLD_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    *_FONT_CANDIDATES,
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for path in (_FONT_BOLD_CANDIDATES if bold else _FONT_CANDIDATES):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# 文本排布
# ---------------------------------------------------------------------------
def wrap(draw: ImageDraw.ImageDraw, text: str, f: ImageFont.FreeTypeFont,
         max_w: int) -> list[str]:
    """按可用像素宽折行；中英混排逐字符累加（中文没有空格可依）。"""
    out: list[str] = []
    for para in text.split("\n"):
        if not para:
            out.append("")
            continue
        line = ""
        for ch in para:
            probe = line + ch
            if draw.textlength(probe, font=f) > max_w and line:
                out.append(line)
                line = ch
            else:
                line = probe
        out.append(line)
    return out


def _text_block_h(draw, text, f, max_w, leading):
    return len(wrap(draw, text, f, max_w)) * leading


# ---------------------------------------------------------------------------
# 单行版式
# ---------------------------------------------------------------------------
def compose_row(tiles: list[Image.Image],
                col_titles: list[str],
                out_name: str,
                cell: int = CELL,
                out_dir: Path | None = None) -> Path:
    """把 N 个格子拼成一张单行图：**只有图像内容 + 一行简短列标题**。

    tiles      : N 张 PIL 图，短边会被缩放到 `cell`（保持长宽比）
    col_titles : N 个简短中文列名（画在每格正上方，单行）

    图注 / source id / 逐格指标一律不进图，改写进 ROW_FIGURE_MAP.md。
    """
    assert len(tiles) == len(col_titles), "列名数量必须与格子数量一致"
    n = len(tiles)

    scaled: list[Image.Image] = []
    for t in tiles:
        w, h = t.size
        sc = cell / min(w, h)
        scaled.append(t.convert("RGB").resize(
            (max(1, int(round(w * sc))), max(1, int(round(h * sc)))),
            Image.Resampling.LANCZOS))
    col_w = max(im.size[0] for im in scaled)
    img_h = max(im.size[1] for im in scaled)

    f_col = font(27, bold=True)
    head_h = 44
    width = n * col_w + (n - 1) * GAP + 2 * MARGIN
    height = MARGIN + head_h + img_h + MARGIN

    panel = Image.new("RGB", (width, height), (255, 255, 255))
    d = ImageDraw.Draw(panel)
    for i, name in enumerate(col_titles):
        d.text((MARGIN + i * (col_w + GAP) + 2, MARGIN), name,
               font=f_col, fill=(20, 20, 20))
    y = MARGIN + head_h
    for i, im in enumerate(scaled):
        x0 = MARGIN + i * (col_w + GAP)
        panel.paste(im, (x0 + (col_w - im.size[0]) // 2,
                         y + (img_h - im.size[1]) // 2))
        d.rectangle([x0, y, x0 + col_w - 1, y + img_h - 1],
                    outline=(210, 210, 210), width=1)

    dest = (out_dir or OUT_DIR)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / out_name
    panel.save(path)
    return path


def save_plain(src: Image.Image, out_name: str, target_w: int = 2600,
               out_dir: Path | None = None) -> Path:
    """纯图表（散点 / 曲线）拆出来单独成页：原样搬运，只做放大，不加任何文字。"""
    w, h = src.size
    sc = max(1.0, target_w / w)
    body = src.convert("RGB").resize((int(round(w * sc)), int(round(h * sc))),
                                     Image.Resampling.LANCZOS)
    dest = (out_dir or OUT_DIR)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / out_name
    body.save(path)
    return path


# ---------------------------------------------------------------------------
# 从已落盘大图裁行（次选路径：脚本需 GPU 重跑时用）
# ---------------------------------------------------------------------------
def crop_cells(src: Path | str, *, n_cols: int, col_w: int,
               y0: int, y1: int, x0: int = 0,
               cols: list[int] | None = None) -> list[Image.Image]:
    """按等宽列切出一行的若干格。`cols` 给定时只取这些列（0-based）。"""
    im = Image.open(src).convert("RGB")
    idx = cols if cols is not None else list(range(n_cols))
    return [im.crop((x0 + c * col_w, y0, x0 + (c + 1) * col_w, y1)) for c in idx]


def crop_box(src: Path | str, box: tuple[int, int, int, int]) -> Image.Image:
    return Image.open(src).convert("RGB").crop(box)


def trim_px(im: Image.Image, tol: int = 248, min_px: int = 10) -> Image.Image:
    """去掉四周只剩坐标轴框线的近白边。

    与 `trim` 的差别：按**非白像素个数**判定，不按比例——matplotlib 的轴框线会在每一行
    留下 1–2 个非白像素，比例法会因此判不出来。只用于确知外面是纯白轴留白的图。
    """
    a = np.asarray(im.convert("L"), dtype=np.float32)
    rows = (a < tol).sum(1) >= min_px
    cols = (a < tol).sum(0) >= min_px
    if not rows.any() or not cols.any():
        return im
    y0, y1 = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]))
    x0, x1 = int(np.argmax(cols)), int(len(cols) - np.argmax(cols[::-1]))
    return im.crop((x0, y0, x1, y1))


def trim(im: Image.Image, tol: int = 246, frac: float = 0.995) -> Image.Image:
    """去掉四周整行/整列的近白边（matplotlib 轴框留白），不动内容像素。"""
    a = np.asarray(im.convert("L"), dtype=np.float32)
    rows = (a > tol).mean(1) < frac
    cols = (a > tol).mean(0) < frac
    if not rows.any() or not cols.any():
        return im
    y0, y1 = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]))
    x0, x1 = int(np.argmax(cols)), int(len(cols) - np.argmax(cols[::-1]))
    return im.crop((x0, y0, x1, y1))


# ---------------------------------------------------------------------------
# 与 visualize_final.py 逐字一致的着色（重绘档用，禁止另立一套色标）
# ---------------------------------------------------------------------------
def mask_image(a: np.ndarray) -> Image.Image:
    g = np.clip(np.asarray(a, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(np.repeat(g[..., None], 3, axis=-1), "RGB")


def signed_rgb(a: np.ndarray, limit: float | None = None) -> np.ndarray:
    v = np.asarray(a, dtype=np.float32)
    if limit is None:
        limit = float(np.percentile(np.abs(v), 98))
    limit = max(float(limit), 1e-6)
    s = np.clip(v / limit, -1.0, 1.0)
    white = np.full((*s.shape, 3), 245.0, dtype=np.float32)
    neg = np.asarray([35.0, 92.0, 190.0], dtype=np.float32)
    pos = np.asarray([205.0, 45.0, 55.0], dtype=np.float32)
    st = np.abs(s)[..., None]
    end = np.where((s >= 0)[..., None], pos, neg)
    return np.clip(white * (1.0 - st) + end * st, 0, 255).astype(np.uint8)


def signed_image(a: np.ndarray, limit: float | None = None) -> Image.Image:
    return Image.fromarray(signed_rgb(a, limit), "RGB")


def up_nearest(im: Image.Image, size: int) -> Image.Image:
    """16x16 这类低分辨率场必须最近邻放大：禁止插值造出不存在的边界。"""
    return im.resize((size, size), Image.Resampling.NEAREST)
