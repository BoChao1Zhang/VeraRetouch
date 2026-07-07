"""Subject-aware two-mode mask geometry (Mask v3).

用户定稿的原则（2026-07-06）：
  - 主体质心/占幅只决定覆盖范围，不决定 mask 类型：有主体时径向/线性都可用。
  - 径向 = PCA 长椭圆完整覆盖主体（外扩 + 最小伸长率）；线性 = 分区，主体完整落在一侧。
  - preset 由 GLOBAL 流程选（本模块只出几何），应用在 mask 内或 mask 外（50/50）。
  - 无明确主体：只用线性对画面二分，不用径向。
  - 线性过渡带必须软（画面自然第一）。
  - 主体来源：预计算 SAM3 cache 的 regions.json + concept mask PNG，按背景词表过滤。

输出 dict 与 mask_synth.sample_geom 同形：{mask_type, what, geom, ...}，
geom 为 LR XMP 几何（CircularGradient: Top/Left/Bottom/Right/Angle/Feather/Flipped；
Gradient: ZeroX/ZeroY/FullX/FullY），可直接进 synth_local_xmp / cgt_raster。

CLI:  python -m construct.subject_geom verify --src-dir D --mask-dir M [--iaa]
      （用外部目录的源图+主体mask 预览两模式几何并可选 IAA 对比，不依赖 LR 农场）
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from typing import Any, Dict, Optional

import numpy as np

SAM3_CACHE = os.environ.get(
    "CONSTRUCT_SAM3_CACHE", "/home/bc/data/datasets/vera_directionA_1M/sam3_cache")

# 背景概念：山川河流天地墙面等环境类不作主体（人物/动物/物件可以）
_BACKGROUND_RE = re.compile(
    r"(sky|cloud|mountain|hill|water|sea|ocean|river|lake|pond|ground|soil|dirt|grass|"
    r"lawn|field|meadow|forest|wood|tree|bush|road|street|path|pavement|wall|floor|"
    r"ceiling|beach|sand|snow|horizon|background|landscape|terrain|cliff|rock_face|"
    r"foliage|vegetation)", re.I)

AREA_LO, AREA_HI = 0.02, 0.55
MIN_BBOX_FILL = 0.20      # mask 面积 / bbox 面积，过滤碎片化 mask
MIN_LINEAR_ROOM = 0.20    # 线性分区要求主体对侧至少留这么多画幅
MARGIN_MAX, MARGIN_MIN = 1.60, 1.02   # 外扩系数上/下限（小主体↔大主体）
AREA_SMALL, AREA_LARGE = 0.04, 0.45   # 对应插值的占比端点
MIN_ELONG = 1.35
RADIAL_FEATHERS = (55.0, 70.0, 85.0)   # LR Feather；径向也偏软
LINEAR_RAMP = (0.22, 0.40)             # 线性过渡带宽度（画幅比例）：必须软
ANGLE_JITTER = 18.0                    # 径向/束状轴向抖动（度）
LINE_ANGLE_JITTER = 25.0               # 线性分割线倾角抖动（度）
BAND_MAX_WIDTH = 0.85                  # 束宽占幅上限，超了三分区退化
BAND_AXIS_LEN = 1.6                    # 束沿轴半长（穿出画幅 → 软边条带）
RADIAL_MIN_B = 0.12                    # 径向短半轴绝对下限（小主体的光晕区不至于过窄）
BAND_MIN_W = 0.12                      # 束半宽绝对下限


# --------------------------------------------------------------------------- #
# 主体挑选
# --------------------------------------------------------------------------- #
def pick_subject(regions: Dict[str, Dict[str, Any]]) -> Optional[str]:
    """从 regions.json 里挑最大合格主体概念；无 → None（走线性二分）。"""
    best, best_area = None, 0.0
    for concept, r in regions.items():
        try:
            area = float(r["area"])
            x0, y0, x1, y1 = r["bbox"]
        except (KeyError, TypeError, ValueError):
            continue
        if not (AREA_LO <= area <= AREA_HI):
            continue
        if _BACKGROUND_RE.search(concept):
            continue
        bbox_area = max((x1 - x0) * (y1 - y0), 1e-6)
        if area / bbox_area < MIN_BBOX_FILL:
            continue
        if area > best_area:
            best, best_area = concept, area
    return best


# --------------------------------------------------------------------------- #
# 几何构造（归一化坐标，LR XMP 语义与 mask_synth.cgt_raster 对齐）
# --------------------------------------------------------------------------- #
def adaptive_margin(area: float) -> float:
    """外扩系数随主体占比自适应：小主体外扩大（上限 MARGIN_MAX），
    大主体刚好覆盖（MARGIN_MIN）。占比对数插值。"""
    a = min(max(float(area), 1e-4), 1.0)
    t = (math.log(a) - math.log(AREA_SMALL)) / (math.log(AREA_LARGE) - math.log(AREA_SMALL))
    return MARGIN_MAX + (MARGIN_MIN - MARGIN_MAX) * min(max(t, 0.0), 1.0)


def _size_factor(area: float) -> float:
    """1=小主体（外扩最大），0=大主体（刚覆盖）。"""
    m = adaptive_margin(area)
    return (m - MARGIN_MIN) / (MARGIN_MAX - MARGIN_MIN)


def _mask_pca(mask01: np.ndarray):
    """归一化坐标 PCA：返回 (center, major_vec, a_extent, b_extent, area_frac)；点太少 → None。"""
    h, w = mask01.shape
    ys, xs = np.nonzero(mask01 > 0.5)
    if xs.size < 32:
        return None
    pts = np.stack([xs / w, ys / h], 1).astype(np.float64)
    c = pts.mean(0)
    cov = np.cov((pts - c).T)
    _, evecs = np.linalg.eigh(cov)
    major, minor = evecs[:, 1], evecs[:, 0]
    # 98 分位投影半径：覆盖主体主干，不被伸出的单肢/杂点撑爆
    a = float(np.quantile(np.abs((pts - c) @ major), 0.98))
    b = float(np.quantile(np.abs((pts - c) @ minor), 0.98))
    return c, major, a, b, float(xs.size / (h * w))


def _ellipse_geom(c, a, b, angle, feather, apply_inside) -> dict:
    return {"Top": round(c[1] - b, 4), "Bottom": round(c[1] + b, 4),
            "Left": round(c[0] - a, 4), "Right": round(c[0] + a, 4),
            "Angle": round(angle, 2), "Feather": feather,
            "Roundness": 0.0, "Midpoint": 50.0,
            "Flipped": "true" if apply_inside else "false"}


def radial_geom(mask01: np.ndarray, rng: random.Random, apply_inside: bool) -> Optional[dict]:
    """PCA 主轴长椭圆完整覆盖主体，轴向带抖动（近各向同性时角度自由采样）。"""
    p = _mask_pca(mask01)
    if p is None:
        return None
    c, major, a, b, area = p
    natural_elong = a / max(b, 1e-6)
    m = adaptive_margin(area)   # 大主体刚覆盖，小主体外扩大
    a *= m
    b *= m
    b = max(b, RADIAL_MIN_B)   # 绝对下限：相对外扩在小主体端失效
    if natural_elong < MIN_ELONG or a < b * MIN_ELONG:
        a = b * MIN_ELONG
    # 不设占幅守卫：大主体就用刚覆盖的椭圆（apply_outside 即薄环带编辑，合法样本）
    if natural_elong < 1.15:     # 近各向同性：PCA 角是噪声，角度自由采样
        angle = rng.uniform(-90.0, 90.0)
    else:
        angle = math.degrees(math.atan2(major[1], major[0])) \
            + rng.uniform(-ANGLE_JITTER, ANGLE_JITTER)
    geom = _ellipse_geom(c, a, b, angle, float(rng.choice(RADIAL_FEATHERS)), apply_inside)
    return {"mask_type": "circulargradient", "what": "Mask/CircularGradient",
            "geom": geom, "_mode": "radial",
            "_apply": "inside" if apply_inside else "outside"}


def band_geom(mask01: np.ndarray, rng: random.Random, apply_inside: bool) -> Optional[dict]:
    """束状三分区：只含主体的软边条带，画面分为 侧|束|侧。
    实现 = 长轴穿出画幅的椭圆（LR 单 mask 可表达）；束方向 = 主体 PCA 主轴
    （60%）或 横/竖 ± 抖动（40%），支持任意倾角。Flipped=true → 束内(主体区)。"""
    p = _mask_pca(mask01)
    if p is None:
        return None
    c, major, _, b, area = p
    width = b * adaptive_margin(area) * 1.05   # 束半宽 = 主体短轴自适应外扩
    width = max(width, BAND_MIN_W)
    if 2 * width > BAND_MAX_WIDTH:       # 束太宽，三分区退化
        return None
    if rng.random() < 0.6:
        angle = math.degrees(math.atan2(major[1], major[0])) \
            + rng.uniform(-ANGLE_JITTER, ANGLE_JITTER)
    else:
        angle = rng.choice((0.0, 90.0)) + rng.uniform(-ANGLE_JITTER, ANGLE_JITTER)
    geom = _ellipse_geom(c, BAND_AXIS_LEN, width, angle,
                         float(rng.choice(RADIAL_FEATHERS)), apply_inside)
    return {"mask_type": "circulargradient", "what": "Mask/CircularGradient",
            "geom": geom, "_mode": "band",
            "_apply": "inside" if apply_inside else "outside"}


def _gradient_geom(pc, base_axis_deg: float, half: float, toward_high: bool,
                   rng: random.Random) -> dict:
    """过 pc、沿 base_axis（含倾角抖动）的软 ramp。toward_high → Full 在高坐标侧。"""
    theta = math.radians(base_axis_deg
                         + rng.uniform(-LINE_ANGLE_JITTER, LINE_ANGLE_JITTER))
    d = (math.cos(theta), math.sin(theta))
    s = 1.0 if toward_high else -1.0
    return {"ZeroX": round(pc[0] - s * half * d[0], 4),
            "ZeroY": round(pc[1] - s * half * d[1], 4),
            "FullX": round(pc[0] + s * half * d[0], 4),
            "FullY": round(pc[1] + s * half * d[1], 4), "Flipped": "false"}


def linear_geom(bbox, rng: random.Random, apply_subject_side: bool,
                area: Optional[float] = None) -> Optional[dict]:
    """线性侧分区：主体完整落在一侧，分割线允许倾斜。软过渡（宽 ramp）。
    分割线离主体的距离随主体占比自适应：大主体贴边，小主体推远。"""
    x0, y0, x1, y1 = bbox
    rooms = {"left": x0, "right": 1 - x1, "top": y0, "bottom": 1 - y1}
    side, room = max(rooms.items(), key=lambda kv: kv[1])  # 对侧空间最大的方向
    if room < MIN_LINEAR_ROOM:
        return None
    wdt = min(rng.uniform(*LINEAR_RAMP), room * 0.9)
    edge = {"left": x0, "right": x1, "top": y0, "bottom": y1}[side]
    sgn = -1 if side in ("left", "top") else 1     # 从主体边缘往空侧偏
    gap = 0.15 + 0.35 * _size_factor(area) if area is not None else 0.35
    center = edge + sgn * (gap * room)             # 分割线放主体边缘外的空区里
    horiz = side in ("left", "right")
    pc = (center, 0.5) if horiz else (0.5, center)
    subj_first = sgn > 0                            # 主体在低坐标侧
    # Full(=1, 应用侧)落在主体侧 ⇔ apply_subject_side；真值表化简为 XOR
    toward_high = bool(subj_first) ^ bool(apply_subject_side)
    geom = _gradient_geom(pc, 0.0 if horiz else 90.0, wdt / 2, toward_high, rng)
    return {"mask_type": "gradient", "what": "Mask/Gradient", "geom": geom,
            "_mode": "linear", "_side": side,
            "_apply": "subject_side" if apply_subject_side else "env_side"}


def bisect_geom(rng: random.Random) -> dict:
    """无主体：软过渡线性二分（位置 1/3、1/2、2/3，方向水平/垂直 + 倾角抖动）。"""
    pos = rng.choice((1 / 3, 0.5, 2 / 3)) + rng.uniform(-0.04, 0.04)
    wdt = rng.uniform(*LINEAR_RAMP)
    horiz = rng.random() < 0.5
    pc = (pos, 0.5) if horiz else (0.5, pos)
    geom = _gradient_geom(pc, 0.0 if horiz else 90.0, wdt / 2,
                          rng.random() < 0.5, rng)
    return {"mask_type": "gradient", "what": "Mask/Gradient", "geom": geom,
            "_mode": "linear_bisect", "_apply": "one_side"}


def semantic_alpha(mask01: np.ndarray, apply_inside: bool = True) -> np.ndarray:
    """语义 mask local edit 的 α：SAM3 主体 mask + 占比自适应软羽化。
    LR XMP 表达不了任意语义 mask，此路线走 numpy 合成（databuild Route 2 / gpu_render）。
    羽化 σ 随主体占比缩放：小主体细边，大主体宽过渡。"""
    import cv2

    m = np.asarray(mask01, dtype=np.float32)
    short = min(m.shape)
    area = float(m.mean())
    sigma = short * (0.008 + 0.017 * math.sqrt(max(area, 1e-4)))
    k = max(3, int(round(sigma * 6)) | 1)
    a = np.clip(cv2.GaussianBlur(m, (k, k), sigma), 0.0, 1.0)
    return a if apply_inside else 1.0 - a


# --------------------------------------------------------------------------- #
# 策略入口
# --------------------------------------------------------------------------- #
def sample_from_subject(regions: Dict[str, Dict[str, Any]],
                        mask01: Optional[np.ndarray],
                        rng: random.Random) -> Optional[dict]:
    """有 regions（+可选主体 mask 数组）时的两模式采样；返回 None 表示无法出几何。"""
    concept = pick_subject(regions)
    if concept is None:
        return dict(bisect_geom(rng), _subject=None)
    r = regions[concept]
    apply_inside = rng.random() < 0.5
    # 模式权重：径向 0.4 / 束状 0.3 / 线性侧分区 0.3；失败沿序退阶
    u = rng.random()
    order = (["radial", "band", "linear"] if u < 0.4 else
             ["band", "radial", "linear"] if u < 0.7 else
             ["linear", "band", "radial"])
    g = None
    for mode in order:
        if mode == "radial" and mask01 is not None:
            g = radial_geom(mask01, rng, apply_inside)
        elif mode == "band" and mask01 is not None:
            g = band_geom(mask01, rng, apply_inside)
        elif mode == "linear":
            g = linear_geom(r["bbox"], rng, apply_subject_side=apply_inside,
                            area=r.get("area"))
        if g is not None:
            break
    if g is None:
        return None
    g["_subject"] = {"concept": concept, "area": r["area"], "bbox": r["bbox"]}
    return g


def sample_for_image(image_path: str, rng: random.Random,
                     cache_dir: str = SAM3_CACHE) -> Optional[dict]:
    """从预计算 SAM3 cache 读 regions + 主体 mask，产出主体感知几何。
    cache 缺失 → None（调用方 fallback 到 mask_synth.sample_geom）。"""
    from dataset_build.mask_cache import CachedMasker

    cm = CachedMasker(cache_dir)
    regions = cm.regions(image_path)
    if not regions:
        return None
    concept = pick_subject(regions)
    mask01 = cm.mask(image_path, concept) if concept else None
    if mask01 is not None and not mask01.any():
        mask01 = None
    return sample_from_subject(regions, mask01, rng)


# --------------------------------------------------------------------------- #
# verify：外部源图+主体 mask 目录 → 几何预览 + 可选 IAA 对比（不走 LR 农场）
# --------------------------------------------------------------------------- #
def _raster(mask_type: str, geom: dict, h: int, w: int) -> np.ndarray:
    from .mask_synth import cgt_raster
    return cgt_raster(mask_type, geom, h, w)


def verify(src_dir: str, mask_dir: str, out_dir: str, n: int = 0, iaa: bool = False) -> None:
    import cv2
    from PIL import Image

    from dataset_build.mask_cache import compute_regions

    os.makedirs(out_dir, exist_ok=True)
    ids = sorted(p[:-4] for p in os.listdir(src_dir) if p.endswith(".png"))
    if n:
        ids = ids[:n]
    class_paths = {"subject": [], "no_subject": []}
    for iid in ids:
        rng = random.Random(iid)
        img = np.asarray(Image.open(os.path.join(src_dir, iid + ".png")).convert("RGB"),
                         np.float32) / 255
        h, w = img.shape[:2]
        mp = os.path.join(mask_dir, iid + ".png")
        m = cv2.imread(mp, 0) if os.path.exists(mp) else None
        mask01 = (m > 127).astype(np.float32) if m is not None and m.max() > 0 else None
        regions = compute_regions({"subject": mask01}) if mask01 is not None else {}
        g = sample_from_subject(regions, mask01, rng)
        if g is None:
            continue
        alpha = _raster(g["mask_type"], g["geom"], h, w)
        ev = rng.choice((0.85, -0.85))
        edited = np.clip(img * (2.0 ** ev), 0, 1)
        out = img * (1 - alpha[..., None]) + edited * alpha[..., None]
        # 三联：源图(mask绿+几何红线) | α 图 | 预览
        vis = (img * 255).astype(np.uint8).copy()
        if mask01 is not None:
            cnts, _ = cv2.findContours(mask01.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (60, 220, 60), 3)
        red = np.zeros_like(vis); red[..., 0] = 255
        vis = (vis * (1 - 0.35 * alpha[..., None])
               + red * 0.35 * alpha[..., None]).astype(np.uint8)
        strip = np.concatenate([
            vis, np.repeat((alpha * 255).astype(np.uint8)[..., None], 3, 2),
            (out * 255).astype(np.uint8)], axis=1)
        tag = f'{g["_mode"]}_{g["_apply"]}_ev{ev:+.2f}'
        cls = "subject" if g.get("_subject") else "no_subject"
        prev = os.path.join(out_dir, f"{iid}.{cls}.{tag}.jpg")
        Image.fromarray(strip).save(prev, quality=90)
        outp = os.path.join(out_dir, f"{iid}.out.png")
        Image.fromarray((out * 255).astype(np.uint8)).save(outp)
        class_paths[cls].append(outp)
        print(iid, cls, tag, flush=True)

    if iaa and any(class_paths.values()):
        from . import objscore
        print("\n=== IAA (iaa_mixed 0-100) 主体明显 vs 不明显 ===")
        for cls, paths in class_paths.items():
            if not paths:
                continue
            scores = [v["iaa_mixed"] for v in objscore.score_many(paths).values()
                      if v.get("iaa_mixed") is not None]
            if scores:
                q = np.percentile(scores, [25, 50, 75])
                print(f"{cls}: n={len(scores)} mean={np.mean(scores):.2f} "
                      f"std={np.std(scores):.2f} q25/50/75={q[0]:.1f}/{q[1]:.1f}/{q[2]:.1f}")
        json.dump({k: v for k, v in class_paths.items()},
                  open(os.path.join(out_dir, "class_paths.json"), "w"), indent=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--src-dir", required=True)
    v.add_argument("--mask-dir", required=True)
    v.add_argument("--out-dir", default="/tmp/subject_geom_verify")
    v.add_argument("--n", type=int, default=0)
    v.add_argument("--iaa", action="store_true")
    a = ap.parse_args()
    if a.cmd == "verify":
        verify(a.src_dir, a.mask_dir, a.out_dir, a.n, a.iaa)


if __name__ == "__main__":
    main()
