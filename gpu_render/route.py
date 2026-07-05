"""preset 渲染分流：高 ΔE / 不可靠 preset → Lightroom 农场，其余 → 本地 GPU 渲染。

架构（per-preset 残差 LUT 之后）：preset 库固定，每个 preset 只需一次性农场渲
8 张探针 GT → `residual.py --fit` 拟合专属残差 → LOO ΔE 即该 preset 的诚实保真。
分流按 **post-residual 实测 ΔE**（fidelity_residual.json 优先，退回 fidelity.json）。
新 preset 入库流程 = 渲探针(8 张,~7s 农场时间) → 拟残差 → 实测分流，不再靠预测。

启发式仅用于「未渲探针」的预筛（mask 类直接 farm）：
  1. local-mask preset                          → farm
  2. 有未覆盖键（本地算子无对应实现）           → farm
  3. 相机 profile 非 Adobe Standard/None        → farm（色彩科学复现不了）
  4. 否则                                       → local（待渲探针后按实测复核）

产物：
  - routing.jsonl: 每个 preset {preset_id, route, reason, measured_de?}
  - WORST.md: 按 ΔE 降序的最差 preset 表（含路由）+ 分流比例 + 预期混合保真

用法: python -m gpu_render.route [--threshold 6.0]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # VeraRetouch 根（gpu_render 包父目录）
from gpu_render.sweeps import CALIB_ROOT

MASK_KEYS = ("MaskGroupBasedCorrections", "CircularGradientBasedCorrections",
             "GradientBasedCorrections", "PaintBasedCorrections")
# 本地算子覆盖的 crs 键前缀（能可靠渲染的）
COVERED_PREFIX = (
    "Exposure2012", "Contrast2012", "Highlights2012", "Shadows2012", "Whites2012", "Blacks2012",
    "Parametric", "ToneCurvePV2012", "Clarity2012", "Texture", "Dehaze",
    "RedHue", "RedSaturation", "GreenHue", "GreenSaturation", "BlueHue", "BlueSaturation", "ShadowTint",
    "HueAdjustment", "SaturationAdjustment", "LuminanceAdjustment", "ConvertToGrayscale", "GrayMixer",
    "SplitToning", "ColorGrade", "Vibrance", "Saturation", "Sharpness", "Sharpen", "LuminanceSmoothing",
    "LuminanceNoiseReduction",   # Detail/Contrast 子键与 LuminanceSmoothing 同族消费（replay._SKIP_PREFIXES）
    "ColorNoiseReduction", "Grain", "PostCropVignette", "VignetteAmount", "VignetteMidpoint",
    "IncrementalTemperature", "IncrementalTint", "Temperature", "Tint", "WhiteBalance",
)
# 明确忽略的键（几何/元数据/无像素效果）——不算未覆盖
IGNORE_PREFIX = (
    "ProcessVersion", "Version", "PresetType", "HasSettings", "Enable", "Supports", "Camera",
    "Crop", "Perspective", "Lens", "Chromatic", "AutoLateral", "Defringe", "Orientation",
    "Look", "UUID", "Name", "SortName", "ShortName", "Group", "Cluster", "Description",
    "Copyright", "ContactInfo", "ToneCurveName", "RequiresRGB", "GrainSeed", "OverrideLook",
    "AutoWhiteVersion", "AutoTone", "AutoGrayscale", "AutoBrightness", "AutoContrast",
    "ParametricShadowSplit", "ParametricMidtoneSplit", "ParametricHighlightSplit", "id",
    "internalName", "type",
    "Amount", "Stubbed",   # lrtemplate 元数据键（replay 同名忽略），非像素效果
)


def _attrs(path: str, fmt: str) -> tuple[dict, str]:
    txt = open(path, encoding="utf-8", errors="ignore").read()
    if any(k in txt for k in MASK_KEYS):
        return {}, "mask"
    if fmt == "xmp" or path.endswith(".xmp"):
        attrs = dict(re.findall(r'crs:([A-Za-z0-9]+)="([^"]*)"', txt))
    else:
        attrs = dict((k, v) for k, v in re.findall(r'\b([A-Za-z][A-Za-z0-9]+)\s*=\s*"?([^",}\s]+)', txt))
    prof = attrs.get("CameraProfile", "")
    return attrs, prof


def _nonzero(v: str) -> bool:
    try:
        return abs(float(str(v).lstrip("+"))) > 1e-6
    except (TypeError, ValueError):
        return str(v).lower() in ("true",)


def route_preset(path: str, fmt: str, measured_de=None, threshold: float = 6.0) -> dict:
    attrs, prof = _attrs(path, fmt)
    if prof == "mask":
        return {"route": "farm", "reason": "local-mask preset"}
    # 已测 preset：实测 ΔE 是最直接真值，优先按它分流。
    # 本地渲染出的 ΔE 已经反映了未覆盖键/profile 的真实影响——若 ΔE 低说明那些键无关紧要。
    if measured_de is not None:
        if measured_de > threshold:
            return {"route": "farm", "reason": f"measured ΔE {measured_de:.1f} > {threshold}"}
        return {"route": "local", "reason": f"measured ΔE {measured_de:.1f} ≤ {threshold}"}
    # 未测(生产中的新 preset)：用启发式预测哪些会渲坏。
    uncovered = []
    for k, v in attrs.items():
        if not _nonzero(v):
            continue
        if any(k.startswith(p) for p in IGNORE_PREFIX):
            continue
        if not any(k.startswith(p) for p in COVERED_PREFIX):
            uncovered.append(k)
    if uncovered:
        return {"route": "farm", "reason": f"uncovered keys: {uncovered[:4]}"}
    # profile 判定大小写不敏感；Embedded/Default 对 JPEG 源等价于不套 raw profile
    if prof and prof.strip().lower() not in (
            "adobe standard", "adobe color", "adobe", "embedded", "default color", "", "none"):
        return {"route": "farm", "reason": f"exotic profile: {prof}"}
    return {"route": "local", "reason": "well-covered (heuristic)"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=6.0, help="ΔE 超此值走农场")
    a = ap.parse_args()
    presets = [json.loads(l) for l in open(os.path.join(CALIB_ROOT, "preset_test", "presets.jsonl"))]
    # 实测 ΔE：残差 LOO（residual_eval.json，图像泛化的诚实指标）优先于
    # fidelity_residual.json（8 探针全参与拟合，偏乐观），再退回 pre-residual fidelity.json
    de_by_id = {}
    for name in ("fidelity.json", "fidelity_residual.json"):
        fp = os.path.join(CALIB_ROOT, "preset_test", name)
        if not os.path.exists(fp):
            continue
        for r in json.load(open(fp))["presets"]:
            if r["rows"]:
                de_by_id[r["preset_id"]] = sum(x["de_mean"] for x in r["rows"]) / len(r["rows"])
    rp = os.path.join(CALIB_ROOT, "preset_test", "residual_eval.json")
    if os.path.exists(rp):
        for r in json.load(open(rp))["presets"]:
            if r.get("de_after") is not None:
                de_by_id[r["preset_id"]] = r["de_after"]

    rows = []
    for pr in presets:
        de = de_by_id.get(pr["preset_id"])
        dec = route_preset(pr["path"], pr["fmt"], de, a.threshold)
        rows.append({"preset_id": pr["preset_id"], "fmt": pr["fmt"], "measured_de": de, **dec})

    out = os.path.join(CALIB_ROOT, "preset_test", "routing.jsonl")
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    farm = [r for r in rows if r["route"] == "farm"]
    local = [r for r in rows if r["route"] == "local"]
    local_des = [r["measured_de"] for r in local if r["measured_de"] is not None]
    import numpy as np
    # 混合保真：farm 视为 ΔE≈0（就是 LR 本身），local 保留实测
    blended = [0.0] * len(farm) + local_des
    md = f"""# preset 渲染分流 (阈值 ΔE>{a.threshold} → farm)

## 分流比例
- **本地 GPU**: {len(local)}/{len(rows)} ({len(local)/len(rows)*100:.0f}%)
- **Lightroom 农场**: {len(farm)}/{len(rows)} ({len(farm)/len(rows)*100:.0f}%)

## 混合保真 (farm 部分 = LR 本身 ΔE≈0)
- 全本地中位 ΔE: {np.median(local_des+[0]):.2f}  →  分流后混合中位 ΔE: **{np.median(blended):.2f}**
- 分流后达标(ΔE≤3): **{np.mean([b<=3 for b in blended])*100:.0f}%**

## 走农场的 preset (按 ΔE 降序)
| preset_id | fmt | ΔE | 原因 |
|---|---|---:|---|
"""
    for r in sorted(farm, key=lambda x: -(x["measured_de"] or 99)):
        de = f"{r['measured_de']:.1f}" if r["measured_de"] is not None else "-"
        md += f"| {r['preset_id'][:24]} | {r['fmt']} | {de} | {r['reason']} |\n"
    mdp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "WORST.md")
    open(mdp, "w").write(md)
    print(f"local={len(local)} farm={len(farm)}  混合中位ΔE={np.median(blended):.2f}  "
          f"达标(≤3)={np.mean([b<=3 for b in blended])*100:.0f}%")
    print(f"-> {out}\n-> {mdp}")


if __name__ == "__main__":
    main()
