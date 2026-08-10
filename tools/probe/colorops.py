"""PR-1/PR-3 探针数据构造：色彩算子 + 已知扰动 + C5 像素统计基线特征。

全部公式**已在线核实**（见 experiments/PR13_probe_whatwhere_20260803/NOTES.md §一）：

- sRGB EOTF / 线性 RGB↔CIE XYZ 矩阵：IEC 61966-2-1（Wikipedia sRGB 条目核实，
  正矩阵 4 位、逆矩阵 2003 修订 7 位）。
- CIE D 系日光轨迹 x_D(T)（4000–7000 K 与 7000–25000 K 两段三次式）与
  y_D = −3.000 x_D² + 2.870 x_D − 0.275：Wikipedia Standard illuminant 条目核实。
- CIELAB：D65 白点 (0.95047, 1.0, 1.08883)，标准 f(t) 分段。

设计纪律：
- 所有色彩变换在**线性光域**施加（gamma/对比度算子除外，见 A4 注释），
  变换后 clip 到 [0,1] 并量化 uint8（模拟真实图像），clip 比例随样本记账。
- WB 增益做**亮度归一**（除以 0.2126g_r+0.7152g_g+0.0722g_b），
  使 A1/A2 与 A3（曝光）在标签层面解耦。
"""
from __future__ import annotations

import numpy as np

# --- sRGB <-> linear (IEC 61966-2-1) --------------------------------------
SRGB_THR_ENC = 0.04045
SRGB_THR_LIN = 0.0031308

# linear sRGB -> CIE XYZ (D65)
M_RGB2XYZ = np.array([
    [0.4124, 0.3576, 0.1805],
    [0.2126, 0.7152, 0.0722],
    [0.0193, 0.1192, 0.9505],
], dtype=np.float64)
# CIE XYZ -> linear sRGB (2003 amendment, 7 decimals)
M_XYZ2RGB = np.array([
    [+3.2406255, -1.5372080, -0.4986286],
    [-0.9689307, +1.8757561, +0.0415175],
    [+0.0557101, -0.2040211, +1.0569959],
], dtype=np.float64)
LUMA_W = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
D65_XYZ = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """float32 计算（相对精度 1e-7，远细于 8bit 量化步 1/255），比 float64 快约 2×。

    uint8 输入按 /255 自动归一（sRGB 编码值域 [0,1]）。
    """
    x = np.asarray(x)
    x = x.astype(np.float32) / np.float32(255.0) if x.dtype == np.uint8 \
        else x.astype(np.float32)
    return np.where(x <= SRGB_THR_ENC, x / np.float32(12.92),
                    ((x + np.float32(0.055)) / np.float32(1.055)) ** np.float32(2.4))


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    return np.where(x <= SRGB_THR_LIN, np.float32(12.92) * x,
                    np.float32(1.055) * x ** np.float32(1 / 2.4) - np.float32(0.055))


# --- CIE D-series daylight locus ------------------------------------------
def cct_to_xy(T: float) -> tuple[float, float]:
    """CIE D 系日光轨迹色度 (x_D, y_D)。有效域 4000–25000 K（核实见 NOTES §一.3）。"""
    if not (4000.0 <= T <= 25000.0):
        raise ValueError(f"CCT {T} 超出 D 系日光轨迹有效域 [4000, 25000] K")
    t3 = 1e3 / T
    if T <= 7000.0:
        x = 0.244063 + 0.09911 * t3 + 2.9678 * t3 ** 2 - 4.6070 * t3 ** 3
    else:
        x = 0.237040 + 0.24748 * t3 + 1.9018 * t3 ** 2 - 2.0064 * t3 ** 3
    y = -3.000 * x * x + 2.870 * x - 0.275
    return float(x), float(y)


def xy_to_XYZ(x: float, y: float, Y: float = 1.0) -> np.ndarray:
    return np.array([Y * x / y, Y, Y * (1.0 - x - y) / y], dtype=np.float64)


def wb_gains(T_from: float, T_to: float) -> np.ndarray:
    """von Kries 对角增益（线性 sRGB 域），亮度归一（中性灰亮度不变）。"""
    g = (M_XYZ2RGB @ xy_to_XYZ(*cct_to_xy(T_to))) / (
        M_XYZ2RGB @ xy_to_XYZ(*cct_to_xy(T_from)))
    return g / float(LUMA_W @ g)


# --- CIELAB ----------------------------------------------------------------
def _lab_f(t: np.ndarray) -> np.ndarray:
    d = 6.0 / 29.0
    return np.where(t > d ** 3, np.cbrt(np.abs(t)) * np.sign(t),
                    t / (3 * d * d) + 4.0 / 29.0)


def linear_rgb_to_lab(rgb_lin: np.ndarray) -> np.ndarray:
    """(...,3) 线性 sRGB -> (...,3) CIELAB（D65）。"""
    xyz = np.asarray(rgb_lin, dtype=np.float32) @ M_RGB2XYZ.T.astype(np.float32)
    f = _lab_f(xyz / D65_XYZ)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def srgb_u8_to_lab(img_u8: np.ndarray) -> np.ndarray:
    return linear_rgb_to_lab(srgb_to_linear(img_u8))


# --- 扰动算子（返回 uint8 图 + 记账 dict）----------------------------------
BASE_CCT = 6500.0
BASE_MIRED = 1e6 / BASE_CCT


def _finish(lin: np.ndarray) -> tuple[np.ndarray, float]:
    """线性域 -> uint8 sRGB；返回 (img_u8, clip_frac)。"""
    clip = float(np.mean((lin < 0.0) | (lin > 1.0)))  # 记账：越界像素比例
    enc = linear_to_srgb(lin)
    return (np.clip(np.rint(enc * 255.0), 0, 255).astype(np.uint8), clip)


def apply_cct(img_u8: np.ndarray, mired_shift: float,
              base_cct: float = BASE_CCT) -> tuple[np.ndarray, dict]:
    """A1 / A6：沿日光轨迹的白平衡变换。mired_shift = 1e6/T_to − 1e6/T_from。"""
    T_to = 1e6 / (1e6 / base_cct + mired_shift)
    g = wb_gains(base_cct, T_to)
    lin = srgb_to_linear(img_u8) * g.astype(np.float32)
    out, clip = _finish(lin)
    return out, {"mired_shift": float(mired_shift), "cct_to": float(T_to),
                 "gains": g.tolist(), "clip_frac": clip}


def apply_wb_tint(img_u8: np.ndarray, delta: float) -> tuple[np.ndarray, dict]:
    """A2：PLAN §3 表的 diag(1+δ, 1, 1−δ)（线性光域）。"""
    g = np.array([1.0 + delta, 1.0, 1.0 - delta], dtype=np.float64)
    lin = srgb_to_linear(img_u8) * g.astype(np.float32)
    out, clip = _finish(lin)
    return out, {"delta": float(delta), "gains": g.tolist(), "clip_frac": clip}


def apply_ev(img_u8: np.ndarray, stops: float) -> tuple[np.ndarray, dict]:
    """A3：线性光域标量缩放 2^stops。"""
    lin = srgb_to_linear(img_u8) * np.float32(2.0 ** stops)
    out, clip = _finish(lin)
    return out, {"stops": float(stops), "clip_frac": clip}


def apply_gamma(img_u8: np.ndarray, gamma: float) -> tuple[np.ndarray, dict]:
    """A4：**编码域**（sRGB 值域）tone-curve gamma —— 摄影软件的"对比度/gamma"控件语义。

    标签用 log2(1/gamma)（正 = 对比更硬）。在编码域施加是刻意的：PLAN §3 的
    tone curve 档就是编码域曲线；线性域 gamma 会退化成近似曝光，和 A3 混淆。
    """
    enc = np.clip(img_u8.astype(np.float32) / np.float32(255.0), 0.0, 1.0) ** np.float32(gamma)
    out = np.clip(np.rint(enc * 255.0), 0, 255).astype(np.uint8)
    return out, {"gamma": float(gamma), "label": float(np.log2(1.0 / gamma)),
                 "clip_frac": 0.0}


# --- 区域色度读数（A5 用）--------------------------------------------------
def region_mean_lab(img_u8: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """mask: (H,W) float [0,1] 软掩膜（与 img 同分辨率）。返回区域加权 Lab 均值。

    先在**线性光域**做加权平均再转 Lab（物理正确：光是线性可加的）。
    """
    w = np.asarray(mask, dtype=np.float64)
    if w.sum() <= 1e-6:
        return np.array([np.nan, np.nan, np.nan])
    lin = srgb_to_linear(img_u8)
    mean_lin = (lin * w[..., None]).sum(axis=(0, 1)) / w.sum()
    return linear_rgb_to_lab(mean_lin[None, :])[0]


# --- C5：像素统计上界基线特征 ---------------------------------------------
_PCTS = (1, 5, 25, 50, 75, 95, 99)
MAX_PIX = 60000     # 全局统计的像素抽样上限（无偏，且把 C5 成本从 ~1.5s 降到 ~0.15s）


def _subsample(img_u8: np.ndarray) -> np.ndarray:
    """确定性等距抽样到 ≤MAX_PIX 像素（全局矩/分位的无偏估计）。"""
    f = img_u8.reshape(-1, 3)
    if f.shape[0] <= MAX_PIX:
        return f
    step = int(np.ceil(f.shape[0] / MAX_PIX))
    return f[::step]


def pixel_stats(img_u8: np.ndarray) -> np.ndarray:
    """全局像素统计（**无空间信息**）——PLAN §3 的 C5「像素统计上界基线」。

    含：sRGB/线性/Lab 三域逐通道 (mean, std, 7 分位) + gray-world 估计
    + white-patch(p99) + Shades-of-Gray(p=6) + 色度直方图（ab 平面 8×8）。
    共 3*(3*9) + 3 + 3 + 3 + 64 = 154 维。
    """
    enc = _subsample(img_u8).astype(np.float32) / np.float32(255.0)
    lin = srgb_to_linear(enc)
    lab = linear_rgb_to_lab(lin)
    feats: list[np.ndarray] = []
    for f in (enc, lin, lab):
        feats.append(f.mean(axis=0))
        feats.append(f.std(axis=0))
        feats.append(np.percentile(f, _PCTS, axis=0).ravel())
    flin = lin
    gw = flin.mean(axis=0)
    feats.append(gw / (gw.sum() + 1e-12))                       # gray-world
    wp = np.percentile(flin, 99, axis=0)
    feats.append(wp / (wp.sum() + 1e-12))                       # white-patch
    sog = (flin ** 6).mean(axis=0) ** (1 / 6)
    feats.append(sog / (sog.sum() + 1e-12))                     # shades-of-gray
    ab = lab.reshape(-1, 3)[:, 1:]
    h, _, _ = np.histogram2d(ab[:, 0], ab[:, 1], bins=8,
                             range=[[-60, 60], [-60, 60]])
    feats.append((h / max(ab.shape[0], 1)).ravel())
    return np.concatenate([np.atleast_1d(f) for f in feats]).astype(np.float32)


def pixel_stats_thumb(img_u8: np.ndarray, grid: int = 8,
                      base: np.ndarray | None = None) -> np.ndarray:
    """C5+ 强化基线（稳健性检查用）：像素统计 ⊕ grid×grid 缩略图 Lab。

    **不是**预注册 C5——C5 按 PLAN §3 定义为「像素统计」（全局矩/分位），
    缩略图带空间信息，作为「更强基线也打不过 VLM 吗」的附加对照单独报。
    `base` 传入已算好的 `pixel_stats(img_u8)` 可免去重复计算。
    """
    from PIL import Image
    im = Image.fromarray(img_u8).resize((grid, grid), Image.Resampling.BOX)
    lab = srgb_u8_to_lab(np.asarray(im))
    p = pixel_stats(img_u8) if base is None else base
    return np.concatenate([p, lab.ravel().astype(np.float32)])
