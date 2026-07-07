"""GPU 批处理 preset 渲染管线（torch, cuda:1）——仿 replay.py 的 LR 管线顺序，
但用 gpu.REGISTRY 的 torch 算子，一次把 B 张图(BCHW)锁步过整条链。

设计（详见 gpu/DESIGN.md）：
  - 管线阶段/次序/键消费逻辑与 tools/lr_calib/replay.replay() 逐行一致（golden 参考不变），
    仅把 numpy `out (H,W,3)` 换成 torch `out (B,3,H,W) on cuda:1`，算子来自 gpu.REGISTRY。
  - gpu 算子约定：fn(img_bchw, ctx) -> 同形张量，ctx = {"label","attrs","elements","fit"}，
    单 ctx 的标量参数广播到整个 batch。故「单 preset 多图」= 一次 kernel 摊平 batch（并行关键）。
  - 「多 preset 单图 / 混合」：按 preset 分组，组内同形 stack 成 batch，逐组过链（gpu_replay）。
  - 覆盖：gpu.REGISTRY 的 65 个算子全部 GPU 原生跑（含 15 个 HSL 稀有色带，见 gpu/hsl.py
    的 gimp_stable 移植）。fallback 机制保留以防未来新增未覆盖算子：fallback="cpu" 时
    逐样本回退 numpy apply_scalar_op（忠实复现）；fallback="skip" 时跳过（吞吐 benchmark 用）。

只依赖 gpu.REGISTRY 与 gpu_render.replay/sweeps/local_apply 的只读接口，不改本体/其他组模块。
"""
from __future__ import annotations

import os

os.environ.setdefault("MONETGPT_TORCH_DEVICE", "cuda:1")
# cpu_apply 回退（如触发）需 numpy 后端（避免 non_gimp 路由到 cuda:0）
os.environ.setdefault("MONETGPT_NON_GIMP_BACKEND", "numpy")

import numpy as np
import torch

from gpu_render.gpu import REGISTRY as GPU_REG
from gpu_render.replay import apply_scalar_op, _f, _HSL_COLORS8
from gpu_render.local_apply import load_fit, FITS_DIR
from gpu_render.sweeps import OPS

DEVICE = os.environ.get("MONETGPT_TORCH_DEVICE", "cuda:1")

# gpu.REGISTRY 的算子无需 CPU 回退（fits 校正已折进 gpu op 内）。
# 15 个 HSL 稀有色带已在 gpu/hsl.py 原生化（rare_hsl_parity de_mean≈0），本列表现为空；
# 保留符号（scratch/gpu_profile.py 引用）与 cpu_apply/skip 回退机制以防未来未覆盖算子。
GPU_MISSING_HSL: tuple = ()


# ---------------------------------------------------------------------------
# numpy<->torch batch 转换
# ---------------------------------------------------------------------------
def to_batch(imgs_hwc, device: str = DEVICE) -> torch.Tensor:
    """[H,W,3] float01 numpy 列表（同形）-> (B,3,H,W) float32 on device。"""
    if isinstance(imgs_hwc, np.ndarray) and imgs_hwc.ndim == 3:
        imgs_hwc = [imgs_hwc]
    arr = np.stack([np.ascontiguousarray(np.asarray(x, np.float32).transpose(2, 0, 1))
                    for x in imgs_hwc])
    return torch.from_numpy(arr).to(device, dtype=torch.float32)


def to_hwc_list(out_bchw: torch.Tensor):
    a = out_bchw.detach().float().cpu().numpy()
    return [np.ascontiguousarray(a[b].transpose(1, 2, 0)) for b in range(a.shape[0])]


def _apply_cfg_np(cfg, img):
    from gpu_render.image_ops.non_gimp_ops import apply_non_gimp_config
    return apply_non_gimp_config(cfg, img, 255.0)


# ---------------------------------------------------------------------------
# 单 preset 锁步过链（作用于整个 batch tensor）
# ---------------------------------------------------------------------------
def replay_batch(out: torch.Tensor, preset: dict, fits_dir: str = FITS_DIR,
                 fallback: str = "cpu", enable=None) -> tuple:
    """把一个 preset 施加到整个 batch。out: (B,3,H,W) on cuda:1。
    返回 (out, info)；info = {"consumed": set, "fallback_ops": [...], "skipped_ops": [...]}。
    fallback: "cpu"=未覆盖算子逐样本回退 numpy（忠实）；"skip"=跳过（吞吐用）。
    stage 次序/键消费与 replay.replay() 一致。"""
    attrs, curves = preset["attrs"], preset["curves"]
    consumed: set = set()
    fb_ops: list = []
    sk_ops: list = []

    def _on(stage):
        return enable is None or stage in enable

    def reg(op, ctx_attrs, label="replay"):
        nonlocal out
        fn = GPU_REG.get(op)
        if fn is None:
            return False
        ctx = {"label": label, "attrs": ctx_attrs, "elements": "",
               "fit": load_fit(op, fits_dir)}
        out = fn(out, ctx).clamp(0.0, 1.0)
        return True

    def cpu_apply(op, lr_v):
        """逐样本 numpy apply_scalar_op（忠实复现 replay 的 fits 标量路径）。"""
        nonlocal out
        arr = out.detach().cpu().numpy()
        fit = load_fit(op, fits_dir)
        res = np.empty_like(arr)
        for b in range(arr.shape[0]):
            hwc = np.ascontiguousarray(arr[b].transpose(1, 2, 0))
            o = apply_scalar_op(hwc, op, lr_v, fit, _apply_cfg_np)
            res[b] = np.asarray(o, np.float32).transpose(2, 0, 1)
        out = torch.from_numpy(res).to(out.device, torch.float32).clamp(0.0, 1.0)

    def scalar(op, lr_v):
        # gpu.REGISTRY 优先（fits 校正已内建）；否则 CPU 回退 / 跳过
        if reg(op, {OPS[op]["crs"]: lr_v}, label=f"{lr_v:+g}"):
            return
        if fallback == "cpu":
            cpu_apply(op, lr_v)
            fb_ops.append(op)
        else:
            sk_ops.append(op)

    _SCALAR_KEYS = {
        "Exposure2012": "Exposure", "Contrast2012": "Contrast",
        "Highlights2012": "Highlights", "Whites2012": "Whites",
        "Blacks2012": "Blacks", "Shadows2012": "Shadows",
    }

    # 1. 白平衡
    if _on("wb"):
        if _f(attrs, "IncrementalTemperature"):
            scalar("Temperature", _f(attrs, "IncrementalTemperature")); consumed.add("IncrementalTemperature")
        if _f(attrs, "IncrementalTint"):
            scalar("Tint", _f(attrs, "IncrementalTint")); consumed.add("IncrementalTint")
        if attrs.get("Temperature") not in (None, "", "0") or attrs.get("Tint") not in (None, "", "0"):
            if reg("AbsTemperature", {"Temperature": attrs.get("Temperature", ""),
                                      "Tint": attrs.get("Tint", ""),
                                      "WhiteBalance": attrs.get("WhiteBalance", "Custom")}):
                consumed.update(k for k in ("Temperature", "Tint") if k in attrs)
    # 2. 基础 tone
    if _on("tone"):
        for key in ("Exposure2012", "Contrast2012", "Highlights2012", "Shadows2012",
                    "Whites2012", "Blacks2012"):
            v = _f(attrs, key)
            if v:
                scalar(_SCALAR_KEYS[key], v); consumed.add(key)
    # 3. 参数曲线
    if _on("pcurve"):
        for pk in ("ParametricShadows", "ParametricDarks", "ParametricLights", "ParametricHighlights"):
            v = _f(attrs, pk)
            if v and reg(pk, {pk: v}):
                consumed.add(pk)
    # 4. 点曲线
    if _on("curve") and curves:
        els = "".join(
            f"<crs:{ck}><rdf:Seq>" + "".join(f"<rdf:li>{int(x)}, {int(y)}</rdf:li>" for x, y in pts)
            + f"</rdf:Seq></crs:{ck}>" for ck, pts in curves.items())
        fn = GPU_REG.get("ToneCurve")
        if fn is not None:
            ctx = {"label": "replay", "attrs": {}, "elements": els,
                   "fit": load_fit("ToneCurve", fits_dir)}
            out = fn(out, ctx).clamp(0.0, 1.0)
            consumed.update(curves)
    # 5. 局部对比/雾
    if _on("localcon"):
        for key, op in (("Clarity2012", "Clarity"), ("Texture", "Texture")):
            v = _f(attrs, key)
            if v:
                scalar(op, v); consumed.add(key)
        if _f(attrs, "Dehaze") and reg("Dehaze", {"Dehaze": _f(attrs, "Dehaze")}):
            consumed.add("Dehaze")
    # 6. 校准面板
    if _on("calib"):
        calib = {k: _f(attrs, k) for k in ("RedHue", "RedSaturation", "GreenHue", "GreenSaturation",
                                           "BlueHue", "BlueSaturation", "ShadowTint") if _f(attrs, k)}
        for k, v in calib.items():
            if reg(f"Calib{k}", {k: v}):
                consumed.add(k)
    # 7. 黑白 or HSL
    if _on("hslbw"):
        if str(attrs.get("ConvertToGrayscale", "")).lower() in ("true", "1"):
            gm = {k: _f(attrs, k) for k in attrs if k.startswith("GrayMixer") and _f(attrs, k)}
            if reg("BlackWhite", {"ConvertToGrayscale": "True", **{k: str(int(v)) for k, v in gm.items()}}):
                consumed.add("ConvertToGrayscale"); consumed.update(gm)
        else:
            for pfx in ("HueAdjustment", "SaturationAdjustment", "LuminanceAdjustment"):
                for c in _HSL_COLORS8:
                    k = f"{pfx}{c}"
                    v = _f(attrs, k)
                    if not v:
                        continue
                    if reg(k, {k: v}):
                        consumed.add(k)
                    elif k in OPS:
                        if fallback == "cpu":
                            cpu_apply(k, v); fb_ops.append(k)
                        else:
                            sk_ops.append(k)
                        consumed.add(k)
    # 8. 分离色调 / ColorGrade（一次核心调用消费全部键，随后 break）
    if _on("splittone") and (
            any(_f(attrs, k) for k in attrs if "Saturation" in k and k.startswith(("SplitToning", "ColorGrade")))
            or any(_f(attrs, k) for k in attrs if k.startswith("ColorGrade") and k.endswith("Lum"))):
        ctx_st = {k: attrs[k] for k in attrs if k.startswith(("SplitToning", "ColorGrade"))}
        for op in ("SplitToneShadow", "SplitToneHighlight", "SplitToneBalance",
                   "ColorGradeMid", "ColorGradeGlobal", "ColorGradeLum", "ColorGradeBlending"):
            if reg(op, ctx_st):
                consumed.update(ctx_st)
                break
    # 9. 饱和族
    if _on("sat"):
        for key in ("Vibrance", "Saturation"):
            v = _f(attrs, key)
            if v:
                scalar(key, v); consumed.add(key)
    # 10. 细节
    if _on("detail"):
        if _f(attrs, "Sharpness"):
            sp = {k: attrs[k] for k in ("Sharpness", "SharpenRadius", "SharpenDetail",
                                        "SharpenEdgeMasking") if k in attrs}
            if reg("Sharpness", sp):
                consumed.update(sp)
            else:
                scalar("Sharpness", _f(attrs, "Sharpness")); consumed.add("Sharpness")
        if _f(attrs, "LuminanceSmoothing"):
            scalar("LuminanceNoiseReduction", _f(attrs, "LuminanceSmoothing"))
            consumed.add("LuminanceSmoothing")
        if _f(attrs, "ColorNoiseReduction") and reg("ColorNR", {"ColorNoiseReduction": _f(attrs, "ColorNoiseReduction")}):
            consumed.add("ColorNoiseReduction")
    # 11. 颗粒
    if _on("grain") and _f(attrs, "GrainAmount") and reg("Grain", {
            "GrainAmount": _f(attrs, "GrainAmount"),
            "GrainSize": _f(attrs, "GrainSize", 25), "GrainFrequency": _f(attrs, "GrainFrequency", 50)}):
        consumed.update(k for k in ("GrainAmount", "GrainSize", "GrainFrequency") if k in attrs)
    # 12. 暗角
    if _on("vignette"):
        vig = {k: attrs[k] for k in attrs if k.startswith("PostCropVignette") or k == "VignetteAmount"}
        if any(_f(attrs, k) for k in ("PostCropVignetteAmount", "VignetteAmount")):
            if reg("Vignette", vig):
                consumed.update(vig)
    # 13. 局部修正（MaskGroupBasedCorrections / 语义 α）——与 replay.replay 一致
    if _on("local") and preset.get("locals"):
        from gpu_render.gpu.local_gpu import apply_locals_batch
        out, lfb = apply_locals_batch(out, preset["locals"], fits_dir, fallback)
        fb_ops.extend(lfb)

    return out, {"consumed": consumed, "fallback_ops": fb_ops, "skipped_ops": sk_ops}


# ---------------------------------------------------------------------------
# 高层 API：混合 (源图, preset) → 批量输出（按 preset+分辨率分组锁步）
# ---------------------------------------------------------------------------
def gpu_replay(imgs, presets, fits_dir: str = FITS_DIR, fallback: str = "cpu",
               device: str = DEVICE) -> list:
    """imgs: HWC float01 numpy 列表（或单张）。presets: 单个已解析 preset（施于全部图）
    或与 imgs 等长的 preset 列表（逐图）。返回与 imgs 对齐的 HWC numpy 输出列表。

    实现：按 (preset 身份, H, W) 分组；组内 stack 成 batch 一次过链；输出按原序还原。
    - 单 preset 多图 → 少数组（同形归一批），一次 kernel 摊平 batch。
    - 多 preset 单图 / 混合 → 组内可能只 1 张，但仍走 batch 路径，正确性一致。"""
    if isinstance(imgs, np.ndarray) and imgs.ndim == 3:
        imgs = [imgs]
    n = len(imgs)
    if isinstance(presets, dict):
        presets = [presets] * n
    assert len(presets) == n, "presets 数量须与 imgs 一致或为单个 preset"

    groups: dict = {}
    for i, (im, pr) in enumerate(zip(imgs, presets)):
        key = (id(pr), im.shape)
        groups.setdefault(key, {"preset": pr, "idx": [], "imgs": []})
        groups[key]["idx"].append(i)
        groups[key]["imgs"].append(im)

    out_list: list = [None] * n
    with torch.no_grad():
        for g in groups.values():
            batch = to_batch(g["imgs"], device)
            res, _ = replay_batch(batch, g["preset"], fits_dir, fallback)
            for i, hwc in zip(g["idx"], to_hwc_list(res)):
                out_list[i] = hwc
            del batch, res
    torch.cuda.empty_cache()
    return out_list
