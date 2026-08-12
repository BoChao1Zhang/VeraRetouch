"""其余多行联图 → 单行图（裁剪档 + 从已落盘 npz 重绘档），全部零 GPU。

版式遵守评审第二轮要求：**画面里只有图像内容 + 一行简短列标题**，
不放图注、不放 source id、不放格内数值、不叠加真值轮廓线。
逐图的 source id / level / 指标写进 `row_facts_crops.json`，供 ROW_FIGURE_MAP.md 引用。

覆盖的原图（选行规则一律「按原图自上而下的行序取前 N 行」，不看效果）：
  P13a  compare20_p01.png                     裁行（5 行 × 7 列 → 3 张 1×7）
  P13b  best_where_w14_metaquery.png          裁行（6 行 × 5 列 → 3 张 1×5）
  P14a  basis_vlm14_full/features20.npz       重绘（14 基 → 几何 8 + 语义 6 两张单行）
  P14b  basis_geo_range8_full/features20.npz  重绘（4 行 × 8 列 → 2 张 1×8）
  P15a  ring_evidence.png                     上下拆分（散点 / 样例各成一页）
  P15b  constrained_axis_evidence.png         上下拆分
  P15c  success_semantic.png                  裁行（3 行 × 3 列 → 2 张 1×3）
  P15d  failure_semantic.png                  裁行
  P64a  basis_geo_range8_full                 重绘（5 行 × 8 列 → 2 张 1×8）
  P64b  basis_vlm14_full                      重绘
  APXA2 E1b failure_r48.png                   裁行（3 行 × 3 列 → 2 张 1×3）

用法：/home/bc/miniconda3/bin/python make_row_crops.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

import rowlib

Image.MAX_IMAGE_PIXELS = None
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
EXP = REPO / "experiments"
FACTS: list[dict] = []


def note(fig: str, **kw) -> None:
    FACTS.append({"fig": fig, **kw})


# ---------------------------------------------------------------------------
# P13a · MCQ config A/B 对照（dense 头的边缘退化）
# 版式常量逐字取自 compare_ab_visualizations.py：width=128, title_h=24,
# header_h=20, panel=128 ⇒ row_h=172，7 列
# ---------------------------------------------------------------------------
def p13a() -> None:
    src = EXP / "MCQ_full_local_l1l6_20260804/viz/config_ab_compare/compare20_p01.png"
    cols = ["输入", "目标", "A 最终渲染", "B 最终渲染",
            "真值区域", "A 预测区域", "B 预测区域"]
    W, TITLE, HEAD, PANEL = 128, 24, 20, 128
    row_h = TITLE + HEAD + PANEL
    big = Image.open(src).convert("RGB")
    for r in range(3):
        y0 = r * row_h + TITLE + HEAD
        tiles = [big.crop((c * W, y0, (c + 1) * W, y0 + PANEL)) for c in range(7)]
        name = f"P13a_dense_edge_degradation_row{r + 1}.png"
        rowlib.compose_row(tiles, cols, name)
        note(name, source_png=str(src.relative_to(REPO)),
             rule=f"按原图行序取第 {r + 1} 行（共 5 行）",
             native_px="每格 128×128（模型渲染分辨率），放大到 440 不引入新信息")
        print("wrote", name)


# ---------------------------------------------------------------------------
# P13b · MetaQuery where 读出（同图两条指令几乎不动）
# ---------------------------------------------------------------------------
def p13b() -> None:
    src = EXP / "MCQ_e2e_whatwhere_20260803/viz/best_where_w14_metaquery.png"
    cols = ["输入图像", "目标区域", "同图另一候选区", "预测 mask", "预测 mask 叠加原图"]
    ys = [(199, 603), (681, 1085), (1162, 1566)]
    xs = [(13, 539), (608, 1012), (1141, 1545), (1675, 2079), (2147, 2674)]
    big = Image.open(src).convert("RGB")
    for r, (y0, y1) in enumerate(ys):
        tiles = [rowlib.trim_px(big.crop((x0, y0, x1, y1))) for x0, x1 in xs]
        name = f"P13b_metaquery_not_instruction_conditioned_row{r + 1}.png"
        rowlib.compose_row(tiles, cols, name)
        note(name, source_png=str(src.relative_to(REPO)),
             rule=f"按原图行序取第 {r + 1} 行（共 6 行）",
             pairing="第 1、2 行是同一张图的 rega / regb 两条指令，可直接看「切指令时预测几乎不动」")
        print("wrote", name)


# ---------------------------------------------------------------------------
# P14a / P14b / P64a / P64b · 从已落盘 features20.npz 重绘
# 着色沿用 visualize_final.py 的 mask_image / signed_image，不另立色标
# ---------------------------------------------------------------------------
BASIS_NAMES_8 = ["1", "x", "y", "P2(x)", "P2(y)", "xy", "L", "S"]
BASIS_NAMES_14 = BASIS_NAMES_8 + [f"e{i}" for i in range(1, 7)]


def _basis_tile(v16: np.ndarray, size: int = 440) -> Image.Image:
    """单张基函数图：2/98 分位拉伸后灰度（与 basis_montage 逐字一致），最近邻放大。"""
    lo, hi = np.percentile(v16, [2, 98])
    g = np.clip((v16 - lo) / max(hi - lo, 1e-8), 0, 1)
    return rowlib.up_nearest(rowlib.mask_image(g), size)


def p14a() -> None:
    npz = EXP / "MCQ_basis_where_l1l6_20260804/viz/basis_vlm14_full/features20.npz"
    z = np.load(npz)
    basis = z["spatial_basis"][0].reshape(16, 16, -1).astype(np.float32)
    uid = str(z["uids"][0])
    groups = [("row1", list(range(8)), "几何基 8 维（解析图案 + 图像亮度/饱和度）"),
              ("row2", list(range(8, 14)), "VLM 语义基 e1..e6（图像相关）")]
    for tag, idx, desc in groups:
        tiles = [_basis_tile(basis[..., i]) for i in idx]
        cols = [f"{i:02d} {BASIS_NAMES_14[i]}" for i in idx]
        name = f"P14a_basis_functions_14_{tag}.png"
        rowlib.compose_row(tiles, cols, name)
        note(name, source_npz=str(npz.relative_to(REPO)), uid=uid, group=desc,
             rule="原图是 14 基的 4×4 图鉴；按基的定义顺序拆成「几何 8 + 语义 6」两张单行",
             native_px="每格 16×16 原生分辨率，最近邻放大（不插值，不造边界）")
        print("wrote", name)


def p14b() -> None:
    npz = EXP / "MCQ_basis_where_l1l6_20260804/viz/basis_geo_range8_full/features20.npz"
    z = np.load(npz)
    basis = z["spatial_basis"].reshape(len(z["uids"]), 16, 16, -1).astype(np.float32)
    coeff = z["spatial_coeff"].astype(np.float32)
    for r in range(2):
        contrib = basis[r] * coeff[r][None, None, :]
        scale = max(float(np.percentile(np.abs(contrib), 98)), 1e-6)
        tiles = [rowlib.up_nearest(rowlib.signed_image(contrib[..., i], scale), 440)
                 for i in range(8)]
        cols = [f"{BASIS_NAMES_8[i]} × w" for i in range(8)]
        name = f"P14b_basis_weighted_contrib_row{r + 1}.png"
        rowlib.compose_row(tiles, cols, name)
        note(name, source_npz=str(npz.relative_to(REPO)), uid=str(z["uids"][r]),
             rule=f"按冻结清单顺序取第 {r + 1} 个样本（共 20 个）",
             coefficients={BASIS_NAMES_8[i]: round(float(coeff[r][i]), 3)
                           for i in range(8)},
             colour="同一行内共享色标（±%.4f，取 |贡献| 的 98 分位）" % scale)
        print("wrote", name)


def _where_row_tiles(vdir: Path, npz, r: int, names: list[str]):
    """where20 的一行：输入取自同目录 gallery20 的 128px 原图，其余从 npz 原生重绘。"""
    gallery = Image.open(vdir / "gallery20_p01.png").convert("RGB")
    W, TITLE, HEAD, PANEL = 128, 24, 20, 128
    y0 = r * (TITLE + HEAD + PANEL) + TITLE + HEAD
    src_tile = gallery.crop((0, y0, W, y0 + PANEL))

    gt = npz["gt_mask"][r].astype(np.float32)
    pm = npz["pred_mask"][r].astype(np.float32)
    basis = npz["spatial_basis"][r].reshape(16, 16, -1).astype(np.float32)
    coeff = npz["spatial_coeff"][r].astype(np.float32)
    contrib = basis * coeff[None, None, :]
    order = np.argsort(-np.abs(coeff))[:2]
    cscale = max(float(np.percentile(np.abs(contrib[..., order]), 98)), 1e-6)

    tiles = [src_tile,
             rowlib.mask_image(gt),
             rowlib.mask_image(pm),
             rowlib.signed_image(pm - gt, 1.0),
             rowlib.signed_image(npz["latent_s"][r].astype(np.float32), 3.0),
             rowlib.mask_image(npz["renderer_s"][r].astype(np.float32)),
             *[rowlib.up_nearest(rowlib.signed_image(contrib[..., i], cscale), 440)
               for i in order]]
    cols = ["输入", "真值 mask", "预测 mask", "mask 差", "latent s", "renderer s",
            *[f"{names[i]} × {coeff[i]:+.2f}" for i in order]]
    return tiles, cols, order, coeff


def p64(tag: str, subdir: str, names: list[str]) -> None:
    vdir = EXP / "MCQ_basis_where_l1l6_20260804/viz" / subdir
    z = np.load(vdir / "features20.npz")
    for r in range(2):
        tiles, cols, order, coeff = _where_row_tiles(vdir, z, r, names)
        name = f"{tag}_row{r + 1}.png"
        rowlib.compose_row(tiles, cols, name)
        note(name, source_npz=str((vdir / "features20.npz").relative_to(REPO)),
             uid=str(z["uids"][r]),
             soft_iou=round(float(z["soft_iou"][r]), 4),
             rule=f"按冻结清单顺序取第 {r + 1} 个样本（共 20 个）",
             top_basis=[f"{names[i]} w={float(coeff[i]):+.3f}" for i in order])
        print("wrote", name)


# ---------------------------------------------------------------------------
# P15a / P15b · 上散点 + 下样例行的复合版式 → 拆成两张
# ---------------------------------------------------------------------------
def split_scatter(src: Path, stem: str, y_split: int,
                  sample_cols: list[tuple[int, int]], sample_y: tuple[int, int],
                  col_titles: list[str], rule: str) -> None:
    big = Image.open(src).convert("RGB")
    rowlib.save_plain(big.crop((0, 0, big.size[0], y_split)), f"{stem}_row1.png")
    note(f"{stem}_row1.png", source_png=str(src.relative_to(REPO)),
         part="原复合图的上半：散点 / 中位线，纯图表，原样搬运未改内容")
    print("wrote", f"{stem}_row1.png")
    tiles = [rowlib.trim(big.crop((x0, sample_y[0], x1, sample_y[1])))
             for x0, x1 in sample_cols]
    rowlib.compose_row(tiles, col_titles, f"{stem}_row2.png")
    note(f"{stem}_row2.png", source_png=str(src.relative_to(REPO)),
         part="原复合图的下半：样例行", rule=rule)
    print("wrote", f"{stem}_row2.png")


# ---------------------------------------------------------------------------
# P15c / P15d / APX-A2 · 3 行 × 3 列 → 逐行裁
# ---------------------------------------------------------------------------
def crop_rows(src: Path, stem: str, ys: list[tuple[int, int]],
              xs: list[tuple[int, int]], cols: list[str],
              n: int, extra: dict | None = None) -> None:
    big = Image.open(src).convert("RGB")
    for r in range(n):
        y0, y1 = ys[r]
        tiles = [rowlib.trim(big.crop((x0, y0, x1, y1))) for x0, x1 in xs]
        name = f"{stem}_row{r + 1}.png"
        rowlib.compose_row(tiles, cols, name)
        note(name, source_png=str(src.relative_to(REPO)),
             rule=f"按原图行序取第 {r + 1} 行（共 {len(ys)} 行）", **(extra or {}))
        print("wrote", name)


def main() -> int:
    p13a()
    p13b()
    p14a()
    p14b()
    p64("P64a_geo8_detail", "basis_geo_range8_full", BASIS_NAMES_8)
    p64("P64b_vlm14_detail", "basis_vlm14_full", BASIS_NAMES_14)

    E2 = EXP / "E2_basis_fit_20260803/viz"
    split_scatter(
        E2 / "ring_evidence.png", "P15a_ring_monotone_vs_bandpass",
        y_split=548,
        sample_cols=[(98, 450), (473, 825), (848, 1200), (1223, 1575)],
        sample_y=(581, 932),
        col_titles=["目标环形掩膜", "带通读出拟合", "单调读出拟合", "拟合出的 s 场"],
        rule="原图下半就是 4 列样例，原样拆出，未改内容")
    split_scatter(
        E2 / "constrained_axis_evidence.png", "P15b_constrained_axis",
        y_split=678,
        sample_cols=[(98, 445), (468, 815), (838, 1185), (1208, 1555), (1578, 1925)],
        sample_y=(713, 1059),
        col_titles=["目标环形掩膜", "受约束 s 轴", "带通读出", "单高斯", "单调读出"],
        rule="原图下半就是 5 列样例，原样拆出，未改内容")

    crop_rows(E2 / "success_semantic.png", "P15c_semantic_fit",
              ys=[(99, 534), (593, 922), (982, 1416)],
              xs=[(24, 518), (541, 1035), (1057, 1550)],
              cols=["图像", "目标掩膜", "拟合结果"], n=2,
              extra={"note": "第 1、2 行为单调读出档，第 3 行为带通读出档"})
    crop_rows(E2 / "failure_semantic.png", "P15d_semantic_failure",
              ys=[(131, 458), (522, 966), (1031, 1358)],
              xs=[(24, 518), (541, 1035), (1057, 1550)],
              cols=["图像", "目标掩膜", "拟合结果"], n=2,
              extra={"note": "语义族尾部失败案例：多个互不相连的小实例 + 细碎边界"})
    crop_rows(EXP / "E1b_svd_20260803/viz/failure_r48.png", "APX_e1b_lowrank_failure",
              ys=[(112, 396), (467, 751), (822, 1107)],
              xs=[(24, 592), (615, 1183), (1207, 1790)],
              cols=["原始 LUT（hald 预览）", "秩 48 重建", "ΔE00 图（含色标）"], n=2)

    (HERE / "row_facts_crops.json").write_text(
        json.dumps(FACTS, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
