"""P · 端到端训练后指令的作用 —— 真实渲染效果对比图（替换原柱状图 P9/P10a）。

出图：docs/presentation_2026-08-05/figs/P_condition_render_compare.png
缓存：docs/presentation_2026-08-05/figs/_cache_P_condition_render_compare.npz

三个 arm 的 condition_mode 语义（出处 train.py:113-135 `condition_inputs`）：
  config_a                 condition_mode="full"          图 + 本样本真实指令
  condition_fixed_shuffle  condition_mode="fixed_shuffle" 图 + **别的样本的真实指令**
                                                          （确定性错位，data.py:109-121）
  condition_image_only     condition_mode="image_only"    图 + **对所有图逐字相同的中性句**
                                                          NEUTRAL_INSTRUCTION（train.py:115）

⚠ 目录名与语义不一致，图上一律按代码语义标注，不按目录名直译。

选源规则（effect-blind，脚本内不得按效果挑图）：
  沿用 config_a 已冻结的固定 20 例可视化清单
  = LocalDataset(pool="test", limit=20) → data.stratified_limit（按 uid 的 sha256 排序，
    层内先保证 source 不重复，层间配额均分），与本图的输出效果完全无关；
  再按 (level, source_id, uid) 排序，每个层级取第 1 例 —— L1..L6 各 1 例，共 6 行。
  该清单已在 2026-08-04 落盘于 viz/config_a/visualization.json，脚本启动时断言一致。

红线遵守：
  - 输入严格 = I_in + instruction，`I_tar` / `.cgt` 只作评测真值与图中对照列，绝不入模型；
  - Δ 图（输出 − 输入）用**全图共用的单一常量增益**（由目标列的 p90 定标），禁逐图 min-max；
  - 不做任何训练、不写 runs/ 下任何文件。

运行：
  /home/bc/VeraRetouch/.venv-lens/bin/python <本文件>            # 有缓存则只重画
  /home/bc/VeraRetouch/.venv-lens/bin/python <本文件> --refresh  # 重跑三臂推理
"""
from __future__ import annotations

import argparse
import json
import sqlite3  # noqa: F401  (repo-wide guard: sqlite3 must import before torch)
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
EXP = REPO / "experiments/MCQ_full_local_l1l6_20260804"
RUNS = EXP / "runs"
MANIFEST = "/mnt/nfs/bc/data/datasets/derived/metacanvas-local-l1l6-v2-20260804"
ANCHORS = REPO / "experiments/RDG_transformer_20260803/runs/ceiling_kmeans/anchors.npy"
FROZEN_UIDS = EXP / "viz/config_a/visualization.json"

CACHE = HERE / "_cache_P_condition_render_compare.npz"
CACHE_JSON = HERE / "_cache_P_condition_render_compare.json"
OUT = HERE / "P_condition_render_compare.png"

VIZ_LIMIT = 20          # 与 viz/config_a 的冻结清单一致
N_ROWS = 6              # 每个层级 1 例，L1..L6

# (arm 目录, 图上的中文短标签, 图上的语义副标题)
ARMS = [
    ("config_a", "真实指令",
     "图 + 本样本真实指令"),
    ("condition_fixed_shuffle", "别的样本的指令",
     "图 + 别的样本的指令"),
    ("condition_image_only", "全体相同中性句",
     "图 + 逐字相同的中性句"),
]

FONT_REG = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


# --------------------------------------------------------------------------
# 阶段一：推理（三臂 × 同一批固定源）
# --------------------------------------------------------------------------
def run_inference(device: str) -> None:
    import torch

    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(EXP))
    import data as DD
    from local_model import LocalCanvasVLM
    from legacy_model import LegacyLocalCanvasVLM
    from train import (condition_inputs, make_loader, render_full,
                       spatial_fields, to_device, NEUTRAL_INSTRUCTION)
    from model.glut_repro.model_rdg import delta_e00

    frozen = json.loads(FROZEN_UIDS.read_text(encoding="utf-8"))["uids"]
    anchors = torch.from_numpy(np.load(ANCHORS))
    dev = torch.device(device)
    torch.cuda.set_device(dev.index or 0)

    store: dict[str, np.ndarray] = {}
    meta: dict = {"arms": {}, "neutral_instruction": NEUTRAL_INSTRUCTION,
                  "manifest": MANIFEST, "pool": "test", "limit": VIZ_LIMIT}
    shared_done = False

    for arm, _, _ in ARMS:
        run_dir = RUNS / arm
        ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
        cfg = ck["args"]
        mode = cfg.get("condition_mode", "full")
        taps = tuple(int(v) for v in cfg["taps"].split(","))
        is_legacy = "renderer" not in cfg
        if is_legacy:
            model = LegacyLocalCanvasVLM(
                dim=int(cfg["dim"]), lora_r=int(cfg["lora_r"]), taps=taps,
                anchors=anchors, n_gauss=int(cfg["n_gauss"]),
                attn_impl=cfg["attn_impl"])
        else:
            model = LocalCanvasVLM(
                dim=int(cfg["dim"]), lora_r=int(cfg["lora_r"]), taps=taps,
                anchors=anchors, n_gauss=int(cfg["n_gauss"]),
                attn_impl=cfg["attn_impl"], renderer=cfg["renderer"],
                spatial_readout=cfg["spatial_readout"],
                param_readout=cfg["param_readout"], interaction=cfg["interaction"])
        model = model.to(dev).eval()
        loaded = model.load_state_dict(ck["state"], strict=False)
        if loaded.unexpected_keys:
            raise RuntimeError(f"{arm}: unexpected checkpoint keys "
                               f"{loaded.unexpected_keys[:5]}")
        print(f"[{arm}] legacy={is_legacy} mode={mode} step={ck['step']} "
              f"missing={len(loaded.missing_keys)} (frozen base params)", flush=True)

        dataset = DD.LocalDataset(MANIFEST, "test", model.image_processor,
                                  limit=VIZ_LIMIT, verify_hash=True)
        uids = [row["uid"] for row in dataset.rows]
        if uids != frozen:
            raise RuntimeError(f"{arm}: 固定清单漂移，与 viz/config_a 不一致")
        loader = make_loader(dataset, batch=2, workers=2)

        preds, masks, ious, de00s, used_instr = [], [], [], [], []
        with torch.no_grad():
            for raw in loader:
                batch = to_device(raw, dev)
                instructions, images = condition_inputs(batch, mode)
                # 红线断言：条件输入里绝不含目标图 / GT 掩膜
                assert images.data_ptr() == batch["image"].data_ptr() or mode == "instruction_only"
                # LegacyLocalCanvasVLM 复用基类 forward，签名里没有 source_rgb
                # （该参数只有 basis_vlm14 读出会用到，dense 读出用不上）。
                kw = {} if is_legacy else {"source_rgb": batch["source_rgb"]}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    params, _ = model(instructions, images, batch["image_size"], **kw)
                params.setdefault("renderer", "gaussian4d")
                pred_mask, pred_s = spatial_fields(params, batch["mask"].shape[-2:])
                pred_mask, pred_s = pred_mask.float(), pred_s.float()
                pred = render_full(params, batch["source_rgb"], pred_s)
                tgt = batch["target_rgb"]
                de = delta_e00(pred.reshape(pred.shape[0], -1, 3),
                               tgt.reshape(tgt.shape[0], -1, 3)).mean(1)
                pm = pred_mask.cpu().numpy()
                gm = batch["mask"].float().cpu().numpy()
                for i in range(pred.shape[0]):
                    den = np.maximum(pm[i], gm[i]).sum() + 1e-8
                    ious.append(float(np.minimum(pm[i], gm[i]).sum() / den))
                    de00s.append(float(de[i]))
                preds.append(pred.float().cpu().numpy())
                masks.append(pm)
                used_instr.extend(instructions)
                if not shared_done:
                    store.setdefault("source", []).append(
                        batch["source_rgb"].cpu().numpy())
                    store.setdefault("target", []).append(
                        batch["target_rgb"].cpu().numpy())
                    store.setdefault("gt_mask", []).append(gm)
                    meta.setdefault("uid", []).extend(raw["uid"])
                    meta.setdefault("level", []).extend(raw["level"])
                    meta.setdefault("instruction_real", []).extend(raw["instruction"])
                    meta.setdefault("instruction_shuffle", []).extend(
                        raw["instruction_fixed_shuffle"])
        shared_done = True
        store[f"{arm}__pred"] = np.concatenate(preds)
        store[f"{arm}__mask"] = np.concatenate(masks)
        store[f"{arm}__iou"] = np.asarray(ious, dtype=np.float32)
        store[f"{arm}__de00"] = np.asarray(de00s, dtype=np.float32)
        meta["arms"][arm] = {
            "condition_mode": mode, "step": int(ck["step"]),
            "is_legacy_module_naming": bool(is_legacy),
            "instruction_used": used_instr,
            "config_digest": ck.get("config_digest"),
            "manifest_digest": ck.get("manifest_digest"),
        }
        del model, ck
        torch.cuda.empty_cache()

    for key in ("source", "target", "gt_mask"):
        store[key] = np.concatenate(store[key])
    np.savez_compressed(CACHE, **{k: np.asarray(v, dtype=np.float32)
                                  for k, v in store.items()})
    CACHE_JSON.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
    print(f"[cached] {CACHE}", flush=True)


# --------------------------------------------------------------------------
# 阶段二：出图
# --------------------------------------------------------------------------
def font(path: str, size: int):
    from PIL import ImageFont
    return ImageFont.truetype(path, size=size)


def rgb_img(arr: np.ndarray):
    from PIL import Image
    return Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), "RGB")


def gray_img(arr: np.ndarray):
    from PIL import Image
    g = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(np.repeat(g[..., None], 3, axis=-1), "RGB")


def delta_img(out: np.ndarray, src: np.ndarray, gain: float):
    """Δ = 输出 − 输入，用**全图共用的单一增益**居中到中性灰。禁逐图归一化。"""
    return rgb_img(np.clip(0.5 + gain * (out - src), 0.0, 1.0))


def truncate(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def build_figure() -> dict:
    from PIL import Image, ImageDraw

    z = np.load(CACHE)
    meta = json.loads(CACHE_JSON.read_text(encoding="utf-8"))
    uids, levels = meta["uid"], meta["level"]

    # ---- 选源：每个层级取第 1 例（清单本身已按 level, source_id, uid 排序）----
    picked: list[int] = []
    for level in sorted(set(levels)):
        picked.append(next(i for i, lv in enumerate(levels) if lv == level))
    picked = picked[:N_ROWS]
    if len(picked) != N_ROWS:
        raise RuntimeError(f"选源规则未取满 {N_ROWS} 行：{picked}")

    source, target, gt_mask = z["source"], z["target"], z["gt_mask"]

    # ---- Δ 图的全局增益：由**目标列**的 90 分位数确定，三臂/全图共用一个常量 ----
    # 目标 Δ 的分布极偏（掩膜外几乎不动，中位仅 0.004），用高分位定标会让 Δ 图整体发灰、
    # 看不出臂间差异；这里让目标列的 p90 落在 0.40，即最强的约 10% 像素饱和，值在图注声明。
    ref = np.abs(target[picked] - source[picked])
    gain = float(np.round(0.40 / max(np.percentile(ref, 90), 1e-6), 1))
    sat = 0.5 / gain

    # ---- 全 20 例的聚合统计（写进图注，避免只靠展示的 6 行说话）----
    n_all = len(uids)
    var_all = np.stack([[float(np.var(z[f"{a}__pred"][i] - source[i])
                               / max(np.var(target[i] - source[i]), 1e-12))
                         for a, _, _ in ARMS] for i in range(n_all)])
    win_all = int((var_all[:, 0] > var_all[:, 1:].max(1)).sum())
    win_shown = int((var_all[picked, 0] > var_all[picked, 1:].max(1)).sum())
    counter_shown = [f"{levels[i]}" for i in picked
                     if var_all[i, 0] <= var_all[i, 1:].max()]
    mask_corr = float(np.median([
        np.corrcoef(z[f"{ARMS[0][0]}__mask"][i].ravel(),
                    z[f"{a}__mask"][i].ravel())[0, 1]
        for a, _, _ in ARMS[1:] for i in range(n_all)]))

    # ---- 尺寸（Δ 图保持方形，不做纵向压扁；指标写在它右侧）----
    M, LW, T, DT, RT, GAP = 30, 82, 196, 90, 38, 14
    DS = DT + 6
    row_h = RT + T + DS + GAP
    n_col = 8
    width = M + LW + n_col * T + M
    head_h, title_h, cap_h = 74, 116, 268
    height = M + title_h + head_h + N_ROWS * row_h + cap_h

    canvas = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(canvas)
    f_title = font(FONT_BOLD, 30)
    f_sub = font(FONT_REG, 18)
    f_head = font(FONT_BOLD, 15)
    f_head2 = font(FONT_REG, 12)
    f_ann = font(FONT_BOLD, 12)
    f_ann2 = font(FONT_REG, 12)
    f_tag = font(FONT_BOLD, 11)
    f_row = font(FONT_REG, 12)
    f_lab = font(FONT_BOLD, 17)
    f_cap = font(FONT_REG, 13)

    BLUE, ORANGE, GREY = "#2a6ebb", "#b8791f", "#8a8f94"
    col_x = [M + LW + i * T for i in range(n_col)]

    # ---- 标题 ----
    y = M
    d.text((M, y), "端到端训练后，指令到底起了什么作用 —— 三档条件输入的真实渲染结果",
           font=f_title, fill=(15, 15, 15))
    y += 40
    d.text((M, y),
           "同一批固定源、同一套训练配方（37,370 条训练池 / 6000 step / 同 seed），"
           "唯一变量 = 喂进模型的条件输入",
           font=f_sub, fill=(68, 68, 68))
    y += 28
    d.text((M, y),
           "看点：第 6–8 列 —— 三档预测掩膜位置几乎一样；每格下方的 Δ 图（输出 − 输入，同一放大倍数）"
           "—— 只有真实指令那一档有明显色块，另两档接近平灰。指令没改「改哪」，改的是「改多狠」。",
           font=f_sub, fill=(150, 20, 32))
    y = M + title_h

    # ---- 列头 ----
    heads = [
        ("输入 I_in", "模型唯一看到的图", (40, 40, 40)),
        ("目标 I_tar", "真值，不入模型", (40, 40, 40)),
        (f"{ARMS[0][1]} → 渲染", ARMS[0][2], BLUE),
        (f"{ARMS[1][1]} → 渲染", ARMS[1][2], ORANGE),
        (f"{ARMS[2][1]} → 渲染", ARMS[2][2], GREY),
        ("GT 掩膜 .cgt", "真值，不入模型", (40, 40, 40)),
        (f"{ARMS[0][1]} → 预测掩膜", ARMS[0][2], BLUE),
        (f"{ARMS[2][1]} → 预测掩膜", ARMS[2][2], GREY),
    ]
    for i, (h1, h2, color) in enumerate(heads):
        d.text((col_x[i] + 5, y + 6), h1, font=f_head, fill=color)
        d.text((col_x[i] + 5, y + 28), h2, font=f_head2, fill=(110, 110, 110))
    # 三个渲染列 / 两个掩膜列的分组底线
    d.line((col_x[2], y + 66, col_x[4] + T - 6, y + 66), fill=BLUE, width=2)
    d.line((col_x[6], y + 66, col_x[7] + T - 6, y + 66), fill=(120, 120, 120), width=2)
    y += head_h

    rows_out = []
    for r, idx in enumerate(picked):
        y0 = y + r * row_h
        # 行内指令原文（截断），让人看清「打乱指令」确实是另一条真实指令
        d.text((M, y0 + 2),
               f'{levels[idx]} · {uids[idx].split("_")[-1][:10]} · '
               f'真实指令：“{truncate(meta["instruction_real"][idx], 96)}”',
               font=f_row, fill=(60, 60, 60))
        d.text((M, y0 + 18),
               f'　　　　　　　　　　　　 打乱指令：'
               f'“{truncate(meta["instruction_shuffle"][idx], 92)}”',
               font=f_row, fill=(150, 110, 40))
        ytile = y0 + RT
        d.text((M + 4, ytile + 8), levels[idx], font=f_lab, fill=(30, 30, 30))
        d.text((M + 4, ytile + 30), f"#{r + 1:02d}", font=f_row, fill=(120, 120, 120))

        panels = [
            ("img", source[idx], None, None),
            ("img", target[idx], target[idx], ["目标幅度", "（参考基准）"]),
        ]
        row_metrics = {}
        for arm, short, _ in ARMS:
            pred = z[f"{arm}__pred"][idx]
            iou = float(z[f"{arm}__iou"][idx])
            de = float(z[f"{arm}__de00"][idx])
            vr = float(np.var(pred - source[idx]) /
                       max(np.var(target[idx] - source[idx]), 1e-12))
            row_metrics[arm] = {"soft_iou": iou, "de00": de, "var_ratio": vr}
            panels.append(("img", pred, pred,
                           [f"soft-IoU {iou:.3f}", f"ΔE00 {de:.2f}",
                            f"Δ 幅度比 {vr:.2f}"]))
        panels.append(("mask", gt_mask[idx], None, None))
        for arm in (ARMS[0][0], ARMS[2][0]):
            panels.append(("mask", z[f"{arm}__mask"][idx], None,
                           [f"soft-IoU {row_metrics[arm]['soft_iou']:.3f}"]))

        for c, (kind, arr, delta_src, ann) in enumerate(panels):
            x = col_x[c]
            img = rgb_img(arr) if kind == "img" else gray_img(arr)
            canvas.paste(img.resize((T - 6, T - 6), Image.Resampling.LANCZOS),
                         (x + 3, ytile))
            d.rectangle((x + 3, ytile, x + T - 4, ytile + T - 7),
                        outline=(200, 200, 200), width=1)
            ystrip = ytile + T - 2
            text_x = x + 5
            if delta_src is not None:
                canvas.paste(
                    delta_img(delta_src, source[idx], gain).resize(
                        (DT, DT), Image.Resampling.LANCZOS), (x + 3, ystrip))
                d.rectangle((x + 3, ystrip, x + 2 + DT, ystrip + DT - 1),
                            outline=(185, 185, 185), width=1)
                d.text((x + 6, ystrip + DT - 17), f"Δ ×{gain:g}",
                       font=f_tag, fill=(20, 20, 20),
                       stroke_width=2, stroke_fill=(255, 255, 255))
                text_x = x + DT + 9
            if ann:
                color = (25, 25, 25) if c != 1 else (110, 110, 110)
                for k, line in enumerate(ann):
                    d.text((text_x, ystrip + 4 + k * 19), line,
                           font=f_ann if k == 0 else f_ann2, fill=color)
        rows_out.append({"uid": uids[idx], "level": levels[idx], **row_metrics})

    # ---- 图注 ----
    ycap = y + N_ROWS * row_h + 8
    per_arm = json.loads((RUNS / ARMS[0][0] / "metrics.json").read_text())
    agg = []
    for arm, short, sem in ARMS:
        sel = json.loads((RUNS / arm / "metrics.json").read_text())["final"]["select"]
        agg.append((arm, short, sem, sel))
    cap_lines = [
        ("样本量与选源：图中 6 个源来自 config_a 于 2026-08-04 已冻结的 20 例可视化清单"
         "（LocalDataset(pool=\"test\", limit=20) → stratified_limit，按 uid 的 sha256 排序、"
         "层内先保证 source 不重复）；本图按 L1–L6 每层取该清单中的第 1 例，共 6 行。"
         "该清单与本图的输出效果无关，脚本内无任何按效果挑图的分支。"),
        ("三档条件语义（train.py:113-135 condition_inputs，⚠ 目录名不直观，按代码语义读）："
         f"「{ARMS[0][1]}」= condition_mode=\"full\"，喂本样本真实指令（runs/config_a）；"
         f"「{ARMS[1][1]}」= \"fixed_shuffle\"，喂另一个样本的真实指令"
         "（data.py:109-121 确定性错位，跨 source 且跨 preset，逐样本不同但都是错的；"
         "runs/condition_fixed_shuffle）；"
         f"「{ARMS[2][1]}」= \"image_only\"，对所有图逐字相同的"
         f" “{meta['neutral_instruction']}”（train.py:115；runs/condition_image_only）。"),
        ("怎么读这张图：① 第 6–8 列（GT 掩膜 / 真实指令预测掩膜 / 中性句预测掩膜）三档位置几乎重合，"
         f"20 例上三档预测掩膜的逐图相关中位为 {mask_corr:.3f} —— 这正是三档掩膜 AUC 几乎相同"
         "（0.9469 / 0.9499 / 0.9481）的原因，也是本项目全实验弃用 AUC 作空间场判据的原因；"
         "② 第 2–5 列下方的 Δ 图（= 0.5 + "
         f"{gain:g}×(输出 − 输入)，全图共用同一常量增益、无逐图归一化，中性灰 = 完全没改，"
         f"|Δ| ≥ {sat:.2f} 处饱和）则分得开：真实指令一档的色块明显更接近目标列，"
         "另两档接近平灰、明显保守。"),
        ("反例照登，不做筛选：按上述规则选出的 " + str(N_ROWS) + " 行里，"
         f"真实指令一档的「Δ 幅度比」在 {win_shown}/{N_ROWS} 行为三档最高；"
         f"冻结清单全部 {n_all} 例上则为 {win_all}/{n_all} 例。"
         + (f"本图 {'、'.join(counter_shown)} 行方向相反"
            f"（另有 {n_all - win_all - len(counter_shown)} 例未展示者同样相反）"
            "—— 打乱指令一档的幅度反而更大，属于该模型「未把指令用足」的表现，"
            "按规则保留在图中，不做替换。" if counter_shown else "")),
        ("整池数字（runs/<arm>/metrics.json → .final.select，n = "
         f"{agg[0][3]['n']}，三档同一批样本）：" +
         "；".join(f"{s[1]} soft-IoU {s[3]['soft_iou_p50']:.3f}・"
                   f"mask 内 PSNR {s[3]['psnr_in_p50']:.2f} dB・"
                   f"ΔE00 {s[3]['de00_p50']:.3f}・Δ 方差比 {s[3]['var_ratio']:.3f}"
                   for s in agg) + "。"),
        ("口径：推理输入严格 = 原图 + 一条指令，目标图与 .cgt 掩膜仅用于本图的对照列与指标计算，"
         "从不进入模型（脚本内有 assert）。选中 step = " +
         "/".join(str(meta["arms"][a]["step"]) for a, _, _ in ARMS) +
         "；manifest digest 三档一致 = "
         f"{per_arm['manifest_digest'][:16]}…；本图只做推理，未训练、未改动任何实验产物。"),
    ]
    yy = ycap
    for line in cap_lines:
        for wrapped in wrap_cjk(line, 118):
            d.text((M, yy), wrapped, font=f_cap, fill=(55, 55, 55))
            yy += 19
        yy += 5

    canvas.save(OUT, dpi=(150, 150))
    return {"out": str(OUT), "gain": gain, "rows": rows_out,
            "picked_uids": [uids[i] for i in picked]}


def wrap_cjk(text: str, width: int) -> list[str]:
    """按显示宽度（CJK 记 2，ASCII 记 1）折行。"""
    lines, cur, w = [], "", 0
    for ch in text:
        cw = 2 if ord(ch) > 0x2E80 else 1
        if w + cw > width * 2 and ch == " ":
            lines.append(cur)
            cur, w = "", 0
            continue
        cur += ch
        w += cw
        if w >= width * 2:
            lines.append(cur)
            cur, w = "", 0
    if cur:
        lines.append(cur)
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="重跑三臂推理")
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    if args.refresh or not CACHE.exists():
        run_inference(args.device)
    info = build_figure()
    print(f"[written] {info['out']}  Δ图增益={info['gain']}")
    for row in info["rows"]:
        line = f"  {row['level']} {row['uid'][-10:]}"
        for arm, short, _ in ARMS:
            m = row[arm]
            line += (f" | {short}: IoU {m['soft_iou']:.3f} "
                     f"ΔE00 {m['de00']:.2f} var {m['var_ratio']:.3f}")
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
