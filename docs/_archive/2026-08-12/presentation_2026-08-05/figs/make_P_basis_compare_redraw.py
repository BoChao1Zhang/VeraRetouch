"""P · 「模型能否读出好的基系数」对照图 —— 重绘版（替换 MCQ_basis viz/*/where20_p01.png）。

出图：docs/presentation_2026-08-05/figs/P_basis_compare_redraw.png
缓存：docs/presentation_2026-08-05/figs/_cache_P_basis_redraw_picks.json
      docs/presentation_2026-08-05/figs/_cache_P_basis_redraw.npz

为什么重绘（评审反馈「这个图肉眼读不懂区别」）：
  旧图两臂的 where 场各自逐图 min-max 着色 ⇒ 低频结构被各自归一化拉成同一个样子，
  读者看不出「加了语义基反而更差」。本图三处改动直击该问题：
    (a) 两臂都画**未经任何逐图归一化的概率场** sigmoid(mask_logits)，色标固定 [0,1]，
        因此同一行、乃至全图所有行共享同一把尺；
    (b) 加 **VLM14 − Geo8 差分列**（发散色图、零值居中、全图共用一个对称上限）；
    (c) 加 **raw attention 参照列**（冻结 VLM 的原始注意力，RO-9c 已落盘），
        给出「语义定位信息本来长什么样」的上下文。

坐标对齐（CLAUDE.md「空间场可视化纪律」：低分辨率场叠回原图禁直接 resize）：
  两条 16×16 网格的**前向路径不同**，因此逆映射也不同，本脚本分别实现、不混用：
    · Geo8 / VLM14 的 mask_logits —— 前向 = source_rgb 先被 `uint8_rgb` **等比例压成
      128×128 方形**（data.py:74），再 `F.interpolate(..., 16, mode="area")` 取基
      （local_model.py:205-216）；训练时 mask_logits 也是 bilinear 升采样回该 128×128
      方形与 `.cgt` 比对。⇒ 该网格逐格覆盖原图宽/高的 1/16，逆映射 = 逐轴等比例展开，
      本脚本 `grid_to_img_stretch()`，与训练用的 `F.interpolate(bilinear)` 同一算子。
    · raw attention —— 前向 = `process_images_` 的 **expand2square 黑边 pad 成方形**再
      切 256 token（llava/mm_utils.py:186）。⇒ 网格覆盖的是 pad 后的方形，逆映射必须
      先还原 pad、再裁掉黑边，本脚本 `grid_to_img_pad()` = RO-9c `make_figs.grid_to_img`
      逐行照抄（`luma_to_grid` 的严格逆）。**直接 resize 会有约 1/3 的系统性错位。**

色标口径：
  · 第 3 列 raw attention —— 画**注意力密度** f / mean(f[valid])（1.0 = 均匀），
    分母**只取有效格**；pad 格既不参与色标也不被填补，且已被 `grid_to_img_pad` 裁出画面，
    另在每格右下角给 16×16 缩略图，pad 格画白并打叉。**非逐图 min-max。**
  · 第 4/5 列 —— 固定绝对色标 [0,1]，无任何逐图归一化，两臂共用。
  · 第 6 列 —— 发散色图，零值居中，上限 = 全部 6 行 |Δ| 的最大值（全图共用一个数）。
  · 着色归着色、算数归算数：图上所有数字（soft-IoU / IoU@k / 系数）一律取自**未归一化的
    原始场**，不经过上面任何一条着色路径。

选源规则（effect-blind，预注册，脚本内不得按效果挑图）：
  见 `pick_sources()` docstring 与产出的 `_cache_P_basis_redraw_picks.json`。

红线遵守：
  · 只做推理，不训练、不写 experiments/ 下任何文件（RO-9c 目录只读）；
  · `I_tar` / `.cgt` 绝不入模型，只作对照列与评分真值；
  · 不使用任何 AUC；空间判据用 soft-IoU + 面积匹配 top-k 的 hard IoU。

运行：
  /home/bc/VeraRetouch/.venv-lens/bin/python <本文件>              # 有缓存则只重画
  /home/bc/VeraRetouch/.venv-lens/bin/python <本文件> --refresh    # 重选源 + 重跑推理
"""
from __future__ import annotations

import argparse
import io
import json
import sqlite3  # noqa: F401  (repo-wide guard: sqlite3 must import before torch)
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
EXP = REPO / "experiments/MCQ_full_local_l1l6_20260804"
BASIS = REPO / "experiments/MCQ_basis_where_l1l6_20260804"
RO9C = REPO / "experiments/RO9c_subject_repro_20260805"
MANIFEST = Path("/mnt/nfs/bc/data/datasets/derived/metacanvas-local-l1l6-v2-20260804")
ANCHORS = REPO / "experiments/RDG_transformer_20260803/runs/ceiling_kmeans/anchors.npy"

PICKS = HERE / "_cache_P_basis_redraw_picks.json"
CACHE = HERE / "_cache_P_basis_redraw.npz"
CACHE_JSON = HERE / "_cache_P_basis_redraw.json"
POP = HERE / "_cache_P_basis_redraw_pop.npz"
OUT = HERE / "P_basis_compare_redraw.png"

GRID = 16
DISP = 448                      # 联图用短边像素（IoU@k 也在该分辨率上算）
DISP_SINGLE = 760               # 单行 PPT 图用短边像素（每格渲染短边 ≥ 600 px）
N_SINGLE = 4                    # 单行 PPT 图张数（按 GT 形状复杂度前 N 名，effect-blind）
LEVELS = ("L1", "L2", "L3", "L4", "L5", "L6")

ARMS = [("basis_geo_range8_full", "Geo8", "8 个解析基（1,x,y,P2x,P2y,xy,L,S）"),
        ("basis_vlm14_full", "VLM14", "8 解析基 + 6 个 VLM 语义基")]

# RO-9c 预注册主口径（analyze_ro9c.py:49-51）：cond=auto（零指令）、token=GC、全 24 层、raw
RO9C_COND = "auto"
RO9C_TOKEN_IDX = 1              # TOKEN_TAGS = ("light", "colortemp", "colormixer")
RO9C_TOKEN_NAME = "<retouch_color&temp>"

FONT_REG = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


# ==========================================================================
# 通用：tar 读取 / 逆映射 / 判据
# ==========================================================================
def read_ref(ref: dict) -> bytes:
    if ref.get("kind") != "indexed_tar":
        raise RuntimeError(f"non-indexed ref: {ref.get('kind')}")
    with open(ref["tar"], "rb") as handle:
        handle.seek(int(ref["offset_data"]))
        raw = handle.read(int(ref["length"]))
    if len(raw) != int(ref["length"]):
        raise IOError(f"short read for {ref['logical_path']}")
    return raw


def disp_size(width: int, height: int, short: int = DISP) -> tuple[int, int]:
    scale = short / min(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def grid_to_img_stretch(m16: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """16×16（覆盖**等比压扁的方形**）→ (h, w)。逐轴等比例展开 = 训练侧升采样的逆。

    训练侧：`F.interpolate(mask_logits[1,1,16,16], size=(128,128), mode="bilinear",
    align_corners=False)`（train.py:186-189）。这里换成 PIL BILINEAR 到显示尺寸，
    两者都是同一条「网格逐格覆盖画幅 1/16」的映射，只是目标分辨率不同。
    """
    width, height = size
    return np.asarray(Image.fromarray(m16.astype(np.float32)).resize(
        (width, height), Image.Resampling.BILINEAR), dtype=np.float32)


def grid_to_img_pad(m16: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """16×16（覆盖 **expand2square pad 后的方形**）→ (h, w)。

    逐行照抄 `experiments/RO9c_subject_repro_20260805/make_figs.py:grid_to_img`
    （= `tools/readout/ro9_gl_attention.py:luma_to_grid` 的严格逆）。禁直接 resize。
    """
    width, height = size
    side = max(width, height)
    n = side // GRID
    big = np.asarray(Image.fromarray(m16.astype(np.float32)).resize(
        (n * GRID, n * GRID), Image.Resampling.BICUBIC), dtype=np.float32)
    square = np.full((side, side), float(big.min()), dtype=np.float32)
    square[: n * GRID, : n * GRID] = big              # luma_to_grid 的截尾同款
    top = (side - height) // 2 if width >= height else 0
    left = (side - width) // 2 if height > width else 0
    crop = square[top:top + height, left:left + width]
    return np.asarray(Image.fromarray(crop).resize((width, height),
                                                   Image.Resampling.BILINEAR),
                      dtype=np.float32)


def soft_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """与 MCQ 实验逐字相同的口径（visualize_final.py:335-340）。pred 必须是概率。"""
    inter = float((pred * gt).sum())
    union = float((pred + gt - pred * gt).sum())
    return inter / max(union, 1e-8)


def iou_at_k(score: np.ndarray, gt_binary: np.ndarray) -> float:
    """面积匹配 top-k 的 hard IoU（CLAUDE.md「AUC 全实验禁用」替代判据表第 1 行）。

    k = GT 正类像素数；**不逐场调阈值**，只按分数排序取前 k。
    """
    k = int(gt_binary.sum())
    if k == 0 or k >= gt_binary.size:
        return float("nan")
    flat = score.reshape(-1)
    idx = np.argpartition(-flat, k - 1)[:k]
    top = np.zeros(flat.shape, dtype=bool)
    top[idx] = True
    top = top.reshape(gt_binary.shape)
    inter = float((top & gt_binary).sum())
    union = float((top | gt_binary).sum())
    return inter / max(union, 1e-8)


# ==========================================================================
# 阶段 0：选源（零 GPU，effect-blind）
# ==========================================================================
def pick_sources() -> dict:
    """预注册选源规则 —— 只用 GT 侧与数据纪律属性，**不看任何一臂的预测**。

    ① 全集：metacanvas-local-l1l6-v2-20260804 的 `val` 池（两臂 train-pool=fit、
       eval-pools=select，val 从未进过训练也未进过 checkpoint 选择）；
       且 source_id ∈ RO-9c 已冻结的 214 源批（`config/g1_region_opp.SNAPSHOT.json`），
       且 `run/stacks/<source_id>__auto.npz` 存在 —— 保证 raw attention 列有真实数据。
       该 manifest 已在构建期过滤 winner_confidence=normal（manifest.json:filters）。
    ② 断言：RO-9c 源图缓存与 metacanvas 源图**长宽比一致**（否则 pad 几何不同、
       raw attention 无法对齐）。169 源实测 0 例不一致。
    ③ GT（`.cgt` 候选区域掩膜，128×128，阈值 0.5）过滤：
       面积占比 ∈ [0.05, 0.45]，且最大连通分量 ≥ 掩膜面积的 40%（排除纯散点）。
    ④ 排序键 = **形状复杂度** = 周长 / (2·sqrt(π·面积))（各向同性圆 = 1.0），降序。
       任务卡要求「优先选清晰剪影 / 多连通区域」——这类源最能暴露低频几何拟合不上轮廓。
    ⑤ 逐层级 L1→L6 各取第 1 名；并列按 uid 字典序；已被前面层级用过的 source 跳过。
    """
    from scipy import ndimage

    region = json.loads((RO9C / "config/g1_region_opp.SNAPSHOT.json").read_text())
    by_id = {row["img_id"]: row for row in region}
    have = {p.name.split("__")[0]
            for p in (RO9C / "run/stacks").glob(f"*__{RO9C_COND}.npz")}
    universe = set(by_id) & have

    rows = []
    for line in open(MANIFEST / "records.jsonl", encoding="utf-8"):
        record = json.loads(line)
        if record["pool"] == "val" and record["source_id"] in universe:
            rows.append(record)
    if not rows:
        raise RuntimeError("val ∩ RO-9c 交集为空")

    # ② 长宽比断言
    aspect, mismatched = {}, []
    for record in rows:
        sid = record["source_id"]
        if sid in aspect:
            continue
        src = Image.open(io.BytesIO(read_ref(record["source_ref"])))
        g1 = Image.open(by_id[sid]["img_path"])
        aspect[sid] = (src.size, g1.size)
        if abs(src.size[0] / src.size[1] - g1.size[0] / g1.size[1]) > 0.01:
            mismatched.append(sid)
    if mismatched:
        raise RuntimeError(f"aspect mismatch, raw-attention 无法对齐: {mismatched[:5]}")

    scored = []
    for record in rows:
        mask = Image.open(io.BytesIO(read_ref(record["mask_ref"]))).convert("L")
        m128 = np.asarray(mask.resize((128, 128), Image.Resampling.BILINEAR),
                          dtype=np.float32) / 255.0
        binary = m128 >= 0.5
        area = int(binary.sum())
        if area == 0:
            continue
        edge = np.zeros_like(binary)
        edge[:-1, :] |= binary[:-1, :] ^ binary[1:, :]
        edge[:, :-1] |= binary[:, :-1] ^ binary[:, 1:]
        perimeter = (edge.sum() + binary[0, :].sum() + binary[-1, :].sum()
                     + binary[:, 0].sum() + binary[:, -1].sum())
        labels, n = ndimage.label(binary)
        sizes = ndimage.sum(binary, labels, range(1, n + 1)) if n else np.asarray([0])
        scored.append({
            "uid": record["uid"], "source_id": record["source_id"],
            "level": record["level"], "preset_id": record.get("preset_id"),
            "instruction": record["instruction"],
            "gt_area_frac": area / binary.size,
            "gt_components": int(n),
            "gt_largest_frac": float(sizes.max() / area),
            "shape_complexity": float(perimeter / (2 * np.sqrt(np.pi * area))),
        })

    picked, used = [], set()
    for level in LEVELS:
        pool = [row for row in scored
                if row["level"] == level
                and 0.05 <= row["gt_area_frac"] <= 0.45
                and row["gt_largest_frac"] >= 0.40
                and row["source_id"] not in used]
        pool.sort(key=lambda row: (-row["shape_complexity"], row["uid"]))
        if not pool:
            raise RuntimeError(f"{level} 无合格源")
        picked.append(pool[0])
        used.add(pool[0]["source_id"])

    ro9c_blind = set(json.loads(
        (RO9C / "config/figure_picks.json").read_text())["picked"])
    payload = {
        "schema": "p_basis_redraw_picks_v1",
        "rule": pick_sources.__doc__,
        "manifest": str(MANIFEST),
        "pool": "val",
        "ro9c_batch": str(RO9C / "config/g1_region_opp.SNAPSHOT.json"),
        "n_candidate_records": len(rows),
        "n_candidate_sources": len({row["source_id"] for row in rows}),
        "aspect_mismatch": len(mismatched),
        "ro9c_effect_blind_overlap": sorted(
            {row["source_id"] for row in picked} & ro9c_blind),
        "picked": picked,
        "candidates": scored,
    }
    PICKS.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                     encoding="utf-8")
    for row in picked:
        print(f"[pick] {row['level']} {row['source_id']} cx="
              f"{row['shape_complexity']:.2f} area={row['gt_area_frac']:.3f} "
              f"ncc={row['gt_components']}", flush=True)
    print(f"[pick] RO-9c effect-blind 交集: {payload['ro9c_effect_blind_overlap']}",
          flush=True)
    return payload


# ==========================================================================
# 阶段 1：推理（GPU；两臂 × 同一批 6 源）
# ==========================================================================
def run_inference(device: str) -> None:
    import torch
    import torch.nn.functional as F

    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(EXP))
    import data as DD
    from local_model import LocalCanvasVLM
    from train import condition_inputs, make_loader, to_device

    picks = json.loads(PICKS.read_text(encoding="utf-8"))
    wanted = [row["uid"] for row in picks["picked"]]
    anchors = torch.from_numpy(np.load(ANCHORS))
    dev = torch.device(device)
    torch.cuda.set_device(dev.index or 0)

    store: dict[str, np.ndarray] = {}
    meta: dict = {"arms": {}, "uids": wanted, "device": device,
                  "manifest": str(MANIFEST), "pool": "val"}
    shared_done = False

    for arm, short, _ in ARMS:
        run_dir = BASIS / "runs" / arm
        ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
        cfg = ck["args"]
        if cfg.get("condition_mode", "full") != "full":
            raise RuntimeError(f"{arm}: 期望 condition_mode=full，实测 {cfg}")
        taps = tuple(int(v) for v in cfg["taps"].split(","))
        model = LocalCanvasVLM(
            dim=int(cfg["dim"]), lora_r=int(cfg["lora_r"]), taps=taps,
            anchors=anchors, n_gauss=int(cfg["n_gauss"]),
            attn_impl=cfg["attn_impl"], renderer=cfg["renderer"],
            spatial_readout=cfg["spatial_readout"],
            param_readout=cfg["param_readout"], interaction=cfg["interaction"])
        model = model.to(dev).eval()
        loaded = model.load_state_dict(ck["state"], strict=False)
        if loaded.unexpected_keys:
            raise RuntimeError(f"{arm}: unexpected keys {loaded.unexpected_keys[:5]}")
        print(f"[{arm}] readout={cfg['spatial_readout']} step={ck['step']}", flush=True)

        dataset = DD.LocalDataset(MANIFEST, "val", model.image_processor,
                                  limit=0, verify_hash=True)
        index = {row["uid"]: row for row in dataset.rows}
        missing = [uid for uid in wanted if uid not in index]
        if missing:
            raise RuntimeError(f"val 池缺 uid: {missing}")
        dataset.rows = [index[uid] for uid in wanted]        # 冻结顺序 = picks 顺序
        loader = make_loader(dataset, batch=3, workers=2)

        logits, coeffs, bases, uids = [], [], [], []
        gts, sources = [], []
        with torch.no_grad():
            for raw in loader:
                batch = to_device(raw, dev)
                instructions, images = condition_inputs(batch, "full")
                # 红线断言：条件输入 = I_in + instruction，绝无 I_tar / .cgt
                assert images.data_ptr() == batch["image"].data_ptr()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    params, _ = model(instructions, images, batch["image_size"],
                                      source_rgb=batch["source_rgb"])
                logits.append(params["mask_logits"].float().cpu().numpy())
                key14 = "spatial_coeff14" if "spatial_coeff14" in params else None
                if key14:
                    coeffs.append(params[key14].float().cpu().numpy())
                    bases.append(params["spatial_basis14"].float().cpu().numpy())
                else:
                    coeffs.append(params["spatial_coeff8"].float().cpu().numpy())
                    bases.append(params["spatial_basis8"].float().cpu().numpy())
                uids.extend(raw["uid"])
                if not shared_done:
                    gts.append(batch["mask"].float().cpu().numpy())
                    sources.append(batch["source_rgb"].float().cpu().numpy())
                _ = F
        if uids != wanted:
            raise RuntimeError(f"{arm}: uid 顺序漂移 {uids}")
        store[f"{short}_logits16"] = np.concatenate(logits).astype(np.float32)
        store[f"{short}_coeff"] = np.concatenate(coeffs).astype(np.float32)
        store[f"{short}_basis"] = np.concatenate(bases).astype(np.float32)
        meta["arms"][short] = {"run_dir": str(run_dir), "step": int(ck["step"]),
                               "spatial_readout": cfg["spatial_readout"],
                               "config_name": cfg["config_name"]}
        if not shared_done:
            store["gt128"] = np.concatenate(gts).astype(np.float32)
            store["src128"] = np.concatenate(sources).astype(np.float32)
            shared_done = True
        del model
        torch.cuda.empty_cache()

    np.savez_compressed(CACHE, uids=np.asarray(wanted), **store)
    CACHE_JSON.write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n",
                          encoding="utf-8")
    print(f"[infer] -> {CACHE}", flush=True)


# ==========================================================================
# 阶段 1b：候选全集上的总体趋势（GPU；防止 6 例被当成总体）
# ==========================================================================
def run_population(device: str) -> None:
    """在**本图的候选全集**（val ∩ RO-9c，539 条）上跑两臂，出 soft-IoU 的配对差分。

    动机：6 行是「GT 形状复杂度最高」的极端子样本，方向未必等于总体。
    冻结的总体参照见 `runs/*/metrics.json` 与 VISUALIZATION_REPORT（完整 select，
    4225 条：Geo8 soft-IoU 0.528 vs VLM14 0.503）。本阶段补上同一候选全集的口径。
    """
    import torch

    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(EXP))
    import data as DD
    from local_model import LocalCanvasVLM
    from train import condition_inputs, make_loader, to_device

    picks = json.loads(PICKS.read_text(encoding="utf-8"))
    cand = sorted(picks["candidates"], key=lambda row: row["uid"])
    wanted = [row["uid"] for row in cand]
    anchors = torch.from_numpy(np.load(ANCHORS))
    dev = torch.device(device)
    torch.cuda.set_device(dev.index or 0)

    out: dict[str, np.ndarray] = {}
    for arm, short, _ in ARMS:
        ck = torch.load(BASIS / "runs" / arm / "best.pt", map_location="cpu",
                        weights_only=False)
        cfg = ck["args"]
        taps = tuple(int(v) for v in cfg["taps"].split(","))
        model = LocalCanvasVLM(
            dim=int(cfg["dim"]), lora_r=int(cfg["lora_r"]), taps=taps,
            anchors=anchors, n_gauss=int(cfg["n_gauss"]),
            attn_impl=cfg["attn_impl"], renderer=cfg["renderer"],
            spatial_readout=cfg["spatial_readout"],
            param_readout=cfg["param_readout"], interaction=cfg["interaction"])
        model = model.to(dev).eval()
        loaded = model.load_state_dict(ck["state"], strict=False)
        if loaded.unexpected_keys:
            raise RuntimeError(f"{arm}: unexpected keys {loaded.unexpected_keys[:5]}")
        dataset = DD.LocalDataset(MANIFEST, "val", model.image_processor,
                                  limit=0, verify_hash=True)
        index = {row["uid"]: row for row in dataset.rows}
        dataset.rows = [index[uid] for uid in wanted]
        loader = make_loader(dataset, batch=8, workers=6)

        ious, seen = [], []
        with torch.no_grad():
            for step, raw in enumerate(loader):
                batch = to_device(raw, dev)
                instructions, images = condition_inputs(batch, "full")
                assert images.data_ptr() == batch["image"].data_ptr()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    params, _ = model(instructions, images, batch["image_size"],
                                      source_rgb=batch["source_rgb"])
                logits = params["mask_logits"].float().cpu().numpy()
                gt = batch["mask"].float().cpu().numpy()
                for i in range(logits.shape[0]):
                    prob = sigmoid(grid_to_img_stretch(logits[i], (128, 128)))
                    ious.append(soft_iou(prob, gt[i]))
                seen.extend(raw["uid"])
                if step % 10 == 0:
                    print(f"[pop:{short}] {len(ious)}/{len(wanted)}", flush=True)
        if seen != wanted:
            raise RuntimeError(f"{arm}: population uid 顺序漂移")
        out[f"{short}_soft_iou"] = np.asarray(ious, dtype=np.float32)
        del model
        torch.cuda.empty_cache()

    np.savez_compressed(
        POP, uids=np.asarray(wanted),
        complexity=np.asarray([row["shape_complexity"] for row in cand], np.float32),
        area=np.asarray([row["gt_area_frac"] for row in cand], np.float32),
        components=np.asarray([row["gt_components"] for row in cand], np.int32),
        **out)
    delta = out["VLM14_soft_iou"] - out["Geo8_soft_iou"]
    print(f"[pop] n={len(delta)} median Geo8={np.median(out['Geo8_soft_iou']):.4f} "
          f"VLM14={np.median(out['VLM14_soft_iou']):.4f} "
          f"median Δ={np.median(delta):+.4f} win={float((delta > 0).mean()):.3f}",
          flush=True)


# ==========================================================================
# 阶段 2：出图（零 GPU）
# ==========================================================================
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


_CJK_PUNCT = "，。；：、（）【】「」“”《》·—…％⚑※"


def wrap_cjk(text: str, budget: float = 99.0) -> str:
    """中英混排的手工折行：CJK 记 1 个宽度单位、ASCII 记 0.52、空格 0.30。

    matplotlib 的 `wrap=True` 按空格切词，中文没有空格会整段溢出画布，故自己折。
    """
    lines: list[str] = []
    for para in text.split("\n"):
        tokens, buf = [], ""
        for ch in para:
            if ord(ch) > 0x2E80 or ch in _CJK_PUNCT:
                if buf:
                    tokens.append(buf)
                    buf = ""
                tokens.append(ch)
            elif ch == " ":
                if buf:
                    tokens.append(buf)
                    buf = ""
                tokens.append(" ")
            else:
                buf += ch
        if buf:
            tokens.append(buf)

        def width(tok: str) -> float:
            if tok == " ":
                return 0.30
            if len(tok) == 1 and (ord(tok) > 0x2E80 or tok in _CJK_PUNCT):
                return 1.0
            return 0.52 * len(tok)

        cur: list[str] = []
        used = 0.0
        for tok in tokens:
            w = width(tok)
            if used + w > budget and cur:
                lines.append("".join(cur).rstrip())
                cur, used = [], 0.0
                if tok == " ":
                    continue
            cur.append(tok)
            used += w
        lines.append("".join(cur).rstrip())
    return "\n".join(lines)


def load_row_assets(picks: dict, short: int = DISP) -> list[dict]:
    """按 uid 取回原始长宽比的输入图 / 全分辨率 GT / RO-9c raw attention。"""
    region = json.loads((RO9C / "config/g1_region_opp.SNAPSHOT.json").read_text())
    by_id = {row["img_id"]: row for row in region}
    index = {}
    for line in open(MANIFEST / "records.jsonl", encoding="utf-8"):
        record = json.loads(line)
        index[record["uid"]] = record

    rows = []
    for pick in picks["picked"]:
        record = index[pick["uid"]]
        source = Image.open(io.BytesIO(read_ref(record["source_ref"]))).convert("RGB")
        from PIL import ImageOps
        source = ImageOps.exif_transpose(source)                 # = data.preprocess_source
        size = disp_size(*source.size, short=short)
        img = np.asarray(source.resize(size, Image.Resampling.LANCZOS),
                         dtype=np.float32) / 255.0
        gt_full = Image.open(io.BytesIO(read_ref(record["mask_ref"]))).convert("L")
        gt = np.asarray(gt_full.resize(size, Image.Resampling.BILINEAR),
                        dtype=np.float32) / 255.0

        with np.load(RO9C / "run/stacks" /
                     f"{pick['source_id']}__{RO9C_COND}.npz") as z:
            raw_stack = z["raw"][RO9C_TOKEN_IDX]                 # (24,16,16)
            valid16 = z["valid16"]
            attn_meta = json.loads(str(z["meta"]))
        attn16 = raw_stack.mean(0).astype(np.float32)            # 全 24 层均值 = RO-9c 主口径
        rows.append({**pick, "img": img, "gt": gt, "size": size,
                     "attn16": attn16, "valid16": valid16,
                     "attn_meta": attn_meta,
                     "instruction_used": record["instruction"],
                     "g1_path": by_id[pick["source_id"]]["img_path"]})
    return rows


def assemble_fields(rows: list[dict], cache: dict) -> None:
    """把两臂的 16×16 logits 与 RO-9c raw attention 摊到显示分辨率，并算判据。

    ⚑ 所有进入判据的数字都取自**未归一化的原始场**；着色用的密度 / 概率另算。
    """
    for i, row in enumerate(rows):
        size = row["size"]
        for short in ("Geo8", "VLM14"):
            logits16 = cache[f"{short}_logits16"][i]
            row[f"{short}_p"] = sigmoid(grid_to_img_stretch(logits16, size))
            row[f"{short}_p16"] = sigmoid(logits16)
            row[f"{short}_coeff"] = cache[f"{short}_coeff"][i]
            # soft-IoU：与实验逐字相同的 128×128 方形口径
            p128 = sigmoid(grid_to_img_stretch(logits16, (128, 128)))
            row[f"{short}_softiou"] = soft_iou(p128, cache["gt128"][i])
        # raw attention：注意力密度（分母只取有效格），pad 格不填补
        valid = row["valid16"]
        attn = row["attn16"]
        row["attn_density16"] = attn / max(float(attn[valid].mean()), 1e-12)
        row["pad_cells"] = int((~valid).sum())
        row["pad_mass"] = float(attn[~valid].sum() / attn.sum()) if (~valid).any() else 0.0
        # 显示用：pad 格置为有效格最小值后做严格逆映射，pad 区域随后被裁出画面
        shown = row["attn_density16"].copy()
        shown[~valid] = float(row["attn_density16"][valid].min())
        row["attn_disp"] = grid_to_img_pad(shown, size)
        # 判据：显示分辨率下的面积匹配 top-k IoU（三列同一口径、同一 GT）
        gtb = row["gt"] >= 0.5
        row["attn_iouk"] = iou_at_k(grid_to_img_pad(attn, size), gtb)
        for short in ("Geo8", "VLM14"):
            row[f"{short}_iouk"] = iou_at_k(row[f"{short}_p"], gtb)
        row["diff"] = row["VLM14_p"] - row["Geo8_p"]


def load_cache(picks: dict) -> dict:
    with np.load(CACHE) as z:
        cache = {k: z[k] for k in z.files}
    uids = [str(u) for u in cache["uids"]]
    if uids != [row["uid"] for row in picks["picked"]]:
        raise RuntimeError("缓存与选源清单不一致，请 --refresh")
    return cache


def pop_summary() -> str:
    """候选全集上的总体方向 —— 每张单页图都带一行，避免单张幻灯片被当成总体结论。"""
    if not POP.exists():
        return ""
    from scipy.stats import wilcoxon

    with np.load(POP) as z:
        pg, pv, pcx = z["Geo8_soft_iou"], z["VLM14_soft_iou"], z["complexity"]
    delta = pv - pg
    edges = np.quantile(pcx, [0.0, 0.25, 0.50, 0.75, 1.0])
    edges[-1] += 1e-6
    idx = np.clip(np.digitize(pcx, edges[1:-1]), 0, 3)
    dm = [float(np.median(delta[idx == b])) for b in range(4)]
    wr = [float((delta[idx == b] > 0).mean()) for b in range(4)]
    stat = wilcoxon(pv, pg)
    return (
        f"总体方向（候选全集 n={len(pg)}，同一 effect-blind 全集）："
        f"soft-IoU 中位 Geo8 {np.median(pg):.3f} / VLM14 {np.median(pv):.3f}，"
        f"配对 Δ={np.median(delta):+.4f}，VLM14 胜率 {float((delta>0).mean())*100:.0f}%，"
        f"Wilcoxon p={stat.pvalue:.1e} ⇒ 平均而言 Geo8 更好（与已冻结的完整 select 4225 条"
        "：Geo8 0.528 > VLM14 0.503 一致）。但按 GT 形状复杂度四分位分桶，配对 Δ 依次为 "
        + " / ".join(f"{v:+.3f}" for v in dm)
        + "（胜率 " + " / ".join(f"{v*100:.0f}%" for v in wr)
        + "）—— 语义基只在轮廓最复杂的那一档上转正，"
        "本页这类源正落在该档，故单页的方向不能当作总体结论。")


def _mark_gt(ax, gt: np.ndarray, halo: str = "white", lw: float = 2.6) -> None:
    """GT 轮廓：红虚线 + 描边（红色在 inferno 的亮区上对比度不足，故加 halo）。"""
    import matplotlib.patheffects as pe

    cs = ax.contour(gt, levels=[0.5], colors=["#ff2626"], linewidths=lw,
                    linestyles=["dashed"])
    for coll in (cs.collections if hasattr(cs, "collections") else [cs]):
        coll.set_path_effects([pe.Stroke(linewidth=lw + 2.2, foreground=halo),
                               pe.Normal()])


def draw_single(picks: dict, clean: bool = False) -> list[Path]:
    """PPT 版式：**一页一图、一图一行**，6 列。

    选源（effect-blind，不看任何一臂预测）：在 `pick_sources()` 已冻结的 L1–L6 六例中，
    再按同一把排序键 —— **GT 形状复杂度** —— 取前 N_SINGLE 名。

    `clean=True`（`--clean`，评审 2026-08-05 要求「只读原始 attention 图本身」）：
    画面里只留热力图 + 一行短列标题 + 色标条。删除大标题/副标题、列标题下的小字、
    底部全部说明段落、格内数值标注、以及叠在第 3–6 列上的红虚线 GT 轮廓
    （真值已单独占第 2 列）。**保留补边格的白色打叉**（那是数据本身）。
    数据、选源、色标口径与非 clean 版逐字相同 —— 本开关只动画面元素。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    font_manager.fontManager.addfont(FONT_REG)
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    plt.rcParams["axes.unicode_minus"] = False

    cache = load_cache(picks)
    rows = load_row_assets(picks, short=DISP_SINGLE)
    assemble_fields(rows, cache)

    order = sorted(range(len(rows)),
                   key=lambda i: (-rows[i]["shape_complexity"], rows[i]["uid"]))
    chosen = [rows[i] for i in order[:N_SINGLE]]
    dlim = max(float(np.abs(row["diff"]).max()) for row in chosen)   # 4 张共用一个上限
    pop_line = "" if clean else pop_summary()

    seq = plt.get_cmap("inferno")
    div = LinearSegmentedColormap.from_list(
        "bwr_soft", ["#1f4e9c", "#7fa8dc", "#f5f5f5", "#e8a09a", "#a4201f"])

    heads = [
        ("输入 $I_{in}$", "#222222"),
        ("GT 掩膜 (.cgt)", "#222222"),
        (f"raw attention 参照\n冻结 VLM · {RO9C_TOKEN_NAME} · 24 层均值", "#7d3c98"),
        ("Geo8 预测场\n8 个解析基", "#1a5276"),
        ("VLM14 预测场\n8 解析基 + 6 个 VLM 语义基", "#922b21"),
        ("差分 VLM14 − Geo8", "#154360"),
    ]
    if clean:                       # 一行短列标题，无小字说明
        heads = [("输入", "#222222"), ("GT 掩膜", "#222222"),
                 ("raw attention", "#7d3c98"), ("Geo8", "#1a5276"),
                 ("VLM14", "#922b21"), ("差分", "#154360")]

    written: list[Path] = []
    for row in chosen:
        gt = row["gt"]
        ncol = 6
        fig_w = 23.0
        left, right, wspace = 0.026, 0.990, 0.028
        cellw = fig_w * (right - left) / (ncol + (ncol - 1) * wspace)
        img_h = cellw * row["size"][1] / row["size"][0]
        if clean:
            title_h, head_h, cbar_h, cap_h = 0.0, 0.62, 0.62, 0.0
            ratios = [head_h, img_h, cbar_h]
            fig_h = head_h + img_h + cbar_h + 0.20
        else:
            title_h, head_h, cbar_h, cap_h = 0.92, 1.02, 0.98, 2.30
            ratios = [head_h, img_h, cbar_h, cap_h]
            fig_h = title_h + head_h + img_h + cbar_h + cap_h + 0.55
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=170)
        gs = fig.add_gridspec(
            len(ratios), ncol, height_ratios=ratios,
            left=left, right=right, wspace=wspace, hspace=0.06,
            top=1.0 - (0.06 if clean else title_h) / fig_h, bottom=0.014)

        if not clean:
            fig.text(left, 1.0 - 0.20 / fig_h,
                     "能否读出好的基系数？ Geo8（8 个解析基） vs VLM14"
                     "（8 解析基 + 6 个 VLM 语义基）",
                     fontsize=25, fontweight="bold", va="top", color="#111111")
            fig.text(left, 1.0 - 0.60 / fig_h,
                     f"{row['level']} · {row['source_id']} · uid …{row['uid'][-12:]} · "
                     f"GT 形状复杂度 {row['shape_complexity']:.2f}（圆=1.0）· "
                     f"GT 面积 {row['gt_area_frac']*100:.0f}% · 连通块 "
                     f"{row['gt_components']} · "
                     f"数据池 val（两臂训练与 ckpt 选择全程未见）",
                     fontsize=14.5, va="top", color="#444444")

        for c, (head, colour) in enumerate(heads):
            ax = fig.add_subplot(gs[0, c])
            ax.axis("off")
            ax.text(0.5, 0.22 if clean else 0.30, head, ha="center", va="center",
                    fontsize=21.0 if clean else 17.5,
                    fontweight="bold", color=colour, linespacing=1.45)

        axes = [fig.add_subplot(gs[1, c]) for c in range(ncol)]
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
                spine.set_color("#999999")

        axes[0].imshow(row["img"])
        axes[1].imshow(row["img"] * 0.35 + 0.65)
        axes[1].imshow(np.dstack([gt, gt * 0.25, gt * 0.25, gt * 0.85]))
        _mark_gt(axes[1], gt, halo="#333333", lw=2.4)

        im_attn = axes[2].imshow(row["attn_disp"], cmap=seq, vmin=0.0, vmax=3.0)
        if not clean:
            _mark_gt(axes[2], gt)
            axes[2].text(0.018, 0.978,
                         f"IoU@k = {row['attn_iouk']:.3f}\nsoft-IoU 不适用",
                         transform=axes[2].transAxes, fontsize=15.5, va="top",
                         color="white", fontweight="bold", linespacing=1.35,
                         bbox=dict(fc="#000000cc", ec="none", pad=2.8))
            axes[2].text(0.018, 0.022,
                         f"补边格 {row['pad_cells']}/256 · 占注意力质量 "
                         f"{row['pad_mass']*100:.0f}%\n"
                         "（色标不含补边格 · 已被逆映射裁出画面 · 未填补）",
                         transform=axes[2].transAxes, fontsize=11.5, va="bottom",
                         color="#ffe680", linespacing=1.35,
                         bbox=dict(fc="#000000cc", ec="none", pad=2.2))
        iw = 0.215
        ih = iw * row["size"][0] / row["size"][1]
        inset = axes[2].inset_axes([0.985 - iw, 0.975 - ih, iw, ih])
        shown16 = np.where(row["valid16"], row["attn_density16"], np.nan)
        inset.imshow(np.ma.masked_invalid(shown16), cmap=seq, vmin=0.0, vmax=3.0)
        ys, xs = np.nonzero(~row["valid16"])
        for y, x in zip(ys, xs):
            inset.add_patch(Rectangle((x - .5, y - .5), 1, 1, fc="white", ec="none"))
            inset.plot([x - .5, x + .5], [y - .5, y + .5], color="#c0392b", lw=1.0)
            inset.plot([x - .5, x + .5], [y + .5, y - .5], color="#c0392b", lw=1.0)
        inset.set_xticks([])
        inset.set_yticks([])
        inset.set_xlim(-.5, GRID - .5)
        inset.set_ylim(GRID - .5, -.5)
        for spine in inset.spines.values():
            spine.set_color("#ffe680")
            spine.set_linewidth(1.4)
        if not clean:
            inset.set_xlabel("16×16 原始网格\n白+叉 = 补边格", fontsize=10.2,
                             color="#ffe680", labelpad=2.5, linespacing=1.25)

        im_pred = None
        for c, short in ((3, "Geo8"), (4, "VLM14")):
            im_pred = axes[c].imshow(row[f"{short}_p"], cmap=seq, vmin=0.0, vmax=1.0)
            if not clean:
                _mark_gt(axes[c], gt)
                axes[c].text(0.018, 0.978,
                             f"soft-IoU = {row[f'{short}_softiou']:.3f}\n"
                             f"IoU@k = {row[f'{short}_iouk']:.3f}",
                             transform=axes[c].transAxes, fontsize=15.5, va="top",
                             color="white", fontweight="bold", linespacing=1.35,
                             bbox=dict(fc="#000000cc", ec="none", pad=2.8))

        im_diff = axes[5].imshow(row["diff"], cmap=div, vmin=-dlim, vmax=dlim)
        delta = row["VLM14_softiou"] - row["Geo8_softiou"]
        if not clean:
            _mark_gt(axes[5], gt, halo="#333333", lw=2.4)
            axes[5].text(0.018, 0.978, f"Δ soft-IoU = {delta:+.3f}\n"
                         f"|Δ|>0.2 的像素 "
                         f"{float((np.abs(row['diff'])>0.2).mean())*100:.0f}%",
                         transform=axes[5].transAxes, fontsize=15, va="top",
                         fontweight="bold", linespacing=1.35,
                         color="#7b1010" if delta < 0 else "#0b4d0b",
                         bbox=dict(fc="#ffffffdd", ec="none", pad=2.8))

        bar_y, bar_hh = (0.46, 0.22) if clean else (0.60, 0.15)
        bar1 = fig.add_subplot(gs[2, 2])
        bar1.axis("off")
        cax1 = bar1.inset_axes([0.05, bar_y, 0.90, bar_hh])
        fig.colorbar(im_attn, cax=cax1, orientation="horizontal", ticks=[0, 1, 2, 3])
        cax1.tick_params(labelsize=13 if clean else 11)

        bar2 = fig.add_subplot(gs[2, 3:5])
        bar2.axis("off")
        cax2 = bar2.inset_axes([0.24, bar_y, 0.52, bar_hh])
        fig.colorbar(im_pred, cax=cax2, orientation="horizontal",
                     ticks=[0, 0.25, 0.5, 0.75, 1.0])
        cax2.tick_params(labelsize=13 if clean else 11)

        bar3 = fig.add_subplot(gs[2, 5])
        bar3.axis("off")
        cax3 = bar3.inset_axes([0.05, bar_y, 0.90, bar_hh])
        fig.colorbar(im_diff, cax=cax3, orientation="horizontal",
                     ticks=[-dlim, 0, dlim], format="%+.2f")
        cax3.tick_params(labelsize=13 if clean else 11)

        if not clean:
            cax1.set_title("注意力密度 f / mean(f[有效格])，1.0 = 均匀\n"
                           "分母只取有效格 · ≥3 截断 · 非逐图 min-max",
                           fontsize=11.5, pad=4, linespacing=1.35)
            cax2.set_title("预测区域概率 sigmoid(mask_logits) · 固定绝对色标 [0,1]\n"
                           "两臂共用同一把尺 · 无任何逐图归一化",
                           fontsize=11.5, pad=4, linespacing=1.35)
            cax3.set_title(f"VLM14 − Geo8 · 零值居中\n"
                           f"{N_SINGLE} 张单页图共用上限 ±{dlim:.2f}",
                           fontsize=11.5, pad=4, linespacing=1.35)

        cap = fig.add_subplot(gs[3, 0:6]) if not clean else None
        if cap is not None:
            cap.axis("off")
            cap.text(
                0.0, 1.06,
                wrap_cjk(
            "选源规则（effect-blind，预注册，脚本内不看任何一臂预测）：metacanvas-local-l1l6-v2 的 val 池"
            "（两臂 train-pool=fit / eval-pools=select，val 全程未进训练、未进 ckpt 选择）∩ RO-9c 已冻结的 214 源批"
            f"（长宽比一致断言 0 例不符），得 {picks['n_candidate_records']} 条候选 / "
            f"{picks['n_candidate_sources']} 源；再要求 GT(.cgt) 面积 ∈[5%,45%] 且最大连通块 ≥40%，"
            "按 GT 形状复杂度（周长 / 2√(π·面积)，圆=1）降序，L1–L6 各取第 1 名、源不重复，"
            f"再取其中复杂度前 {N_SINGLE} 名各出一页。全过程只用 GT 与数据纪律属性。\n"
            "坐标对齐（两条 16×16 网格前向路径不同，逆映射分别做、不混用）：Geo8/VLM14 的网格覆盖"
            "『等比压扁成方形的画幅』（解析基取自 source_rgb 的 area 下采样，训练时也按此与 .cgt 比对）"
            "⇒ 逆映射 = 逐轴等比例展开，与训练侧 bilinear 升采样同一算子；raw attention 的网格覆盖"
            "『expand2square 黑边 pad 后的方形』⇒ 逆映射 = 还原 pad → 裁掉黑边（RO-9c make_figs.grid_to_img，"
            "luma_to_grid 的严格逆）。直接 resize 会把 25–38% 的画幅盖上补边格的值，本图未使用。\n"
            "色标口径：第 3 列 = 注意力密度，分母只取有效格，补边格既不参与色标也不填补（缩略图里画白打叉）；"
            "第 4/5 列 = 固定绝对 [0,1]，两臂共用同一把尺；第 6 列 = 发散色图零值居中、多页共用上限。"
            "着色归着色、算数归算数：图上所有数字取自未归一化的原始场。soft-IoU = 实验原式（128×128 方形，概率场）；"
            "IoU@k = 面积匹配 top-k 的 hard IoU（显示分辨率，三列同一 GT、同一规则，禁逐场调阈值）。全图未使用任何 AUC。\n"
            + (pop_line + "\n" if pop_line else "") +
            "读图要点：① 红虚线 = GT 轮廓，直接看两臂贴不贴合；② 第 4/5 列共用同一把固定色标，形状差异肉眼可比"
            "（旧图两臂各自逐图归一化，正是差异被抹掉的原因）；③ 第 6 列蓝 = 语义基把概率往下压、红 = 往上抬，"
            "看它压/抬的位置落在 GT 轮廓内还是外；④ 第 3 列表明冻结 VLM 的原始注意力本身也不贴合轮廓，"
            "它是主体先验而非指令 grounding（RO-9c 主判据已判 FAIL）。",
                    budget=141.0),
                fontsize=11.2, va="top", linespacing=1.55, color="#1a1a1a")

        out = HERE / (f"P_basis_single_{row['level']}_"
                      f"{row['source_id'].replace('_a', 'a')[:14]}.png")
        fig.savefig(out, dpi=170, facecolor="white")
        plt.close(fig)
        written.append(out)
        px = int(cellw * 170)
        print(f"[single] {out.name}  每格 {px}px 宽 · soft-IoU geo="
              f"{row['Geo8_softiou']:.3f} vlm={row['VLM14_softiou']:.3f} "
              f"(Δ={delta:+.3f}) · IoU@k attn={row['attn_iouk']:.3f} "
              f"geo={row['Geo8_iouk']:.3f} vlm={row['VLM14_iouk']:.3f}", flush=True)

    (HERE / "P_basis_single_index.json").write_text(json.dumps({
        "schema": "p_basis_single_index_v1",
        "rule": draw_single.__doc__,
        "diff_shared_limit": dlim,
        "files": [{"path": str(p), "level": r["level"], "source_id": r["source_id"],
                   "uid": r["uid"], "shape_complexity": r["shape_complexity"],
                   "geo8_soft_iou": r["Geo8_softiou"],
                   "vlm14_soft_iou": r["VLM14_softiou"],
                   "delta_soft_iou": r["VLM14_softiou"] - r["Geo8_softiou"],
                   "raw_attn_iou_at_k": r["attn_iouk"],
                   "geo8_iou_at_k": r["Geo8_iouk"], "vlm14_iou_at_k": r["VLM14_iouk"],
                   "pad_cells": r["pad_cells"], "pad_mass_frac": r["pad_mass"]}
                  for p, r in zip(written, chosen)],
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return written


def draw(picks: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    font_manager.fontManager.addfont(FONT_REG)
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    plt.rcParams["axes.unicode_minus"] = False

    cache = load_cache(picks)
    meta = json.loads(CACHE_JSON.read_text(encoding="utf-8"))
    rows = load_row_assets(picks)
    assemble_fields(rows, cache)
    dlim = max(float(np.abs(row["diff"]).max()) for row in rows)

    # ---- 版式 -------------------------------------------------------------
    ncol, nrow = 6, len(rows)
    fig_w = 22.0
    cellw = (fig_w * 0.976) / (ncol + (ncol - 1) * 0.035)
    aspects = [row["size"][1] / row["size"][0] for row in rows]
    head_h, cbar_h, cap_h, gap = 1.05, 0.90, 3.65, 0.20
    body = [cellw * a for a in aspects]
    fig_h = head_h + sum(body) + cbar_h + cap_h + gap * (nrow + 2)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=130)
    gs = fig.add_gridspec(nrow + 3, ncol,
                          height_ratios=[head_h] + body + [cbar_h, cap_h],
                          left=0.030, right=0.988, top=0.988, bottom=0.010,
                          wspace=0.035, hspace=gap / (sum(body) / nrow))

    fig.text(0.030, 0.9965,
             "能否读出好的基系数？ Geo8（8 个解析基） vs VLM14（8 解析基 + 6 个 VLM 语义基）"
             "  ——  同一 checkpoint、同一批源、同一把色标",
             fontsize=17.5, fontweight="bold", va="top", color="#111111")

    heads = ["输入 $I_{in}$", "GT 掩膜 (.cgt)",
             f"raw attention 参照\n冻结 VLM · {RO9C_TOKEN_NAME} · 24 层均值",
             "Geo8 预测场\n8 解析基", "VLM14 预测场\n8 解析基 + 6 语义基",
             "差分  VLM14 − Geo8"]
    hcol = ["#222222", "#222222", "#7d3c98", "#1a5276", "#922b21", "#154360"]
    for c, (head, colour) in enumerate(zip(heads, hcol)):
        ax = fig.add_subplot(gs[0, c])
        ax.axis("off")
        ax.text(0.5, 0.18, head, ha="center", va="center", fontsize=15.5,
                fontweight="bold", color=colour, linespacing=1.45)

    seq = plt.get_cmap("inferno")
    div = LinearSegmentedColormap.from_list(
        "bwr_soft", ["#1f4e9c", "#7fa8dc", "#f5f5f5", "#e8a09a", "#a4201f"])

    im_attn = im_pred = im_diff = None
    for r, row in enumerate(rows):
        gt = row["gt"]
        axes = [fig.add_subplot(gs[r + 1, c]) for c in range(ncol)]
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.6)
                spine.set_color("#999999")

        axes[0].imshow(row["img"])
        axes[0].set_ylabel(f"{row['level']}\n{row['source_id'][:12]}",
                           fontsize=10.8, fontweight="bold", labelpad=4)
        axes[0].text(0.015, 0.975,
                     f"形状复杂度 {row['shape_complexity']:.2f} · "
                     f"GT 面积 {row['gt_area_frac']*100:.0f}% · "
                     f"连通块 {row['gt_components']}",
                     transform=axes[0].transAxes, fontsize=9.2, va="top",
                     color="white",
                     bbox=dict(fc="#000000cc", ec="none", pad=1.8))

        axes[1].imshow(row["img"] * 0.35 + 0.65)
        axes[1].imshow(np.dstack([gt, gt * 0.25, gt * 0.25, gt * 0.85]))
        axes[1].contour(gt, levels=[0.5], colors=["#e02020"],
                        linewidths=1.5, linestyles="--")

        im_attn = axes[2].imshow(row["attn_disp"], cmap=seq, vmin=0.0, vmax=3.0)
        axes[2].contour(gt, levels=[0.5], colors=["#39ff14"],
                        linewidths=1.6, linestyles="--")
        axes[2].text(0.015, 0.975,
                     f"IoU@k={row['attn_iouk']:.3f}   soft-IoU 不适用",
                     transform=axes[2].transAxes, fontsize=10.2, va="top",
                     color="white", bbox=dict(fc="#000000cc", ec="none", pad=1.8))
        axes[2].text(0.015, 0.025,
                     f"pad 格 {row['pad_cells']}/256 · 占注意力质量 "
                     f"{row['pad_mass']*100:.0f}%（已裁出画面、未填补）",
                     transform=axes[2].transAxes, fontsize=8.6, va="bottom",
                     color="#ffe680", bbox=dict(fc="#000000cc", ec="none", pad=1.6))
        # pad 格显式标注：右下角 16×16 缩略图，pad 格画白并打叉
        inset = axes[2].inset_axes([0.735, 0.045, 0.245, 0.245 *
                                    row["size"][0] / row["size"][1]])
        shown16 = np.where(row["valid16"], row["attn_density16"], np.nan)
        inset.imshow(np.ma.masked_invalid(shown16), cmap=seq, vmin=0.0, vmax=3.0)
        ys, xs = np.nonzero(~row["valid16"])
        for y, x in zip(ys, xs):
            inset.add_patch(Rectangle((x - .5, y - .5), 1, 1, fc="white", ec="none"))
            inset.plot([x - .5, x + .5], [y - .5, y + .5], color="#c0392b", lw=0.7)
            inset.plot([x - .5, x + .5], [y + .5, y - .5], color="#c0392b", lw=0.7)
        inset.set_xticks([])
        inset.set_yticks([])
        inset.set_xlim(-.5, GRID - .5)
        inset.set_ylim(GRID - .5, -.5)
        for spine in inset.spines.values():
            spine.set_color("#ffe680")
            spine.set_linewidth(1.0)

        for c, short in ((3, "Geo8"), (4, "VLM14")):
            im_pred = axes[c].imshow(row[f"{short}_p"], cmap=seq, vmin=0.0, vmax=1.0)
            axes[c].contour(gt, levels=[0.5], colors=["#39ff14"],
                            linewidths=1.6, linestyles="--")
            axes[c].text(0.015, 0.975,
                         f"soft-IoU={row[f'{short}_softiou']:.3f}   "
                         f"IoU@k={row[f'{short}_iouk']:.3f}",
                         transform=axes[c].transAxes, fontsize=10.6, va="top",
                         color="white", fontweight="bold",
                         bbox=dict(fc="#000000cc", ec="none", pad=1.8))

        im_diff = axes[5].imshow(row["diff"], cmap=div, vmin=-dlim, vmax=dlim)
        axes[5].contour(gt, levels=[0.5], colors=["#111111"],
                        linewidths=1.5, linestyles="--")
        delta = row["VLM14_softiou"] - row["Geo8_softiou"]
        axes[5].text(0.015, 0.975,
                     f"Δ soft-IoU = {delta:+.3f}",
                     transform=axes[5].transAxes, fontsize=10.6, va="top",
                     fontweight="bold",
                     color="#7b1010" if delta < 0 else "#0b4d0b",
                     bbox=dict(fc="#ffffffcc", ec="none", pad=1.8))
        axes[5].text(0.015, 0.025,
                     f"|Δ|>0.2 的像素 {float((np.abs(row['diff'])>0.2).mean())*100:.0f}%",
                     transform=axes[5].transAxes, fontsize=8.8, va="bottom",
                     color="#333333", bbox=dict(fc="#ffffffcc", ec="none", pad=1.6))

    # ---- 色标条（各列正下方） ---------------------------------------------
    bar1 = fig.add_subplot(gs[nrow + 1, 2])
    bar1.axis("off")
    cax1 = bar1.inset_axes([0.06, 0.62, 0.88, 0.13])
    fig.colorbar(im_attn, cax=cax1, orientation="horizontal", ticks=[0, 1, 2, 3])
    cax1.set_title("注意力密度  f / mean(f[有效格])，1.0 = 均匀\n"
                   "分母只取有效格 · ≥3 截断 · 非逐图 min-max",
                   fontsize=9.6, pad=4, linespacing=1.4)

    bar2 = fig.add_subplot(gs[nrow + 1, 3:5])
    bar2.axis("off")
    cax2 = bar2.inset_axes([0.26, 0.62, 0.48, 0.13])
    fig.colorbar(im_pred, cax=cax2, orientation="horizontal",
                 ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cax2.set_title("预测区域概率 sigmoid(mask_logits)\n"
                   "固定绝对色标 [0,1] · 两臂共用 · 无任何逐图归一化",
                   fontsize=9.6, pad=4, linespacing=1.4)

    bar3 = fig.add_subplot(gs[nrow + 1, 5])
    bar3.axis("off")
    cax3 = bar3.inset_axes([0.06, 0.62, 0.88, 0.13])
    fig.colorbar(im_diff, cax=cax3, orientation="horizontal",
                 ticks=[-dlim, 0, dlim], format="%+.2f")
    cax3.set_title("VLM14 − Geo8\n零值居中 · 全图共用一个对称上限",
                   fontsize=9.6, pad=4, linespacing=1.4)

    geo = np.stack([row["Geo8_softiou"] for row in rows])
    vlm = np.stack([row["VLM14_softiou"] for row in rows])
    geok = np.stack([row["Geo8_iouk"] for row in rows])
    vlmk = np.stack([row["VLM14_iouk"] for row in rows])
    attnk = np.stack([row["attn_iouk"] for row in rows])
    blind = picks.get("ro9c_effect_blind_overlap") or []

    # ---- 总体趋势面板（候选全集，防止 6 例被当成总体） ---------------------
    pop_txt = "（未生成 _cache_P_basis_redraw_pop.npz，跳过总体面板）"
    if POP.exists():
        from scipy.stats import wilcoxon

        with np.load(POP) as z:
            pop = {k: z[k] for k in z.files}
        pg, pv = pop["Geo8_soft_iou"], pop["VLM14_soft_iou"]
        pcx = pop["complexity"]
        delta = pv - pg
        edges = np.quantile(pcx, [0.0, 0.25, 0.50, 0.75, 1.0])
        edges[-1] += 1e-6
        idx = np.clip(np.digitize(pcx, edges[1:-1]), 0, 3)
        stat = wilcoxon(pv, pg)

        hold_p = fig.add_subplot(gs[nrow + 2, 0])
        hold_d = fig.add_subplot(gs[nrow + 2, 1])
        hold_p.axis("off")
        hold_d.axis("off")
        pax = hold_p.inset_axes([0.20, 0.20, 0.78, 0.66])
        dax = hold_d.inset_axes([0.26, 0.20, 0.72, 0.66])
        centres = np.arange(4)
        gm = [float(np.median(pg[idx == b])) for b in range(4)]
        vm = [float(np.median(pv[idx == b])) for b in range(4)]
        dm = [float(np.median(delta[idx == b])) for b in range(4)]
        wr = [float((delta[idx == b] > 0).mean()) for b in range(4)]
        nb = [int((idx == b).sum()) for b in range(4)]
        pax.plot(centres, gm, "o-", color="#1a5276", lw=2.0, label="Geo8")
        pax.plot(centres, vm, "s-", color="#922b21", lw=2.0, label="VLM14")
        pax.set_xticks(centres)
        pax.set_xticklabels([f"[{edges[b]:.2f},{edges[b+1]:.2f})\nn={nb[b]}"
                             for b in range(4)], fontsize=8.2)
        pax.set_xlabel("GT 形状复杂度 四分位（→ 越右轮廓越复杂）", fontsize=9.2)
        pax.set_ylabel("soft-IoU 中位", fontsize=9.2)
        pax.set_title(f"候选全集 n={len(pg)}（val ∩ RO-9c）", fontsize=10.2)
        pax.legend(fontsize=9, loc="lower left")
        pax.grid(alpha=0.25, lw=0.6)
        pax.tick_params(labelsize=8.4)
        pax.axvspan(2.5, 3.5, color="#f0c000", alpha=0.20, zorder=0)
        lo, hi = pax.get_ylim()
        pax.set_ylim(lo, hi + 0.18 * (hi - lo))
        pax.text(3.0, hi + 0.09 * (hi - lo), "上图 6 例\n全在此桶", ha="center",
                 va="center", fontsize=8.6, color="#8a6d00", fontweight="bold")

        dax.bar(centres, dm, color=["#922b21" if v > 0 else "#1a5276" for v in dm],
                width=0.62)
        dax.axhline(0, color="black", lw=1.0)
        dax.set_xticks(centres)
        dax.set_xticklabels([f"胜率\n{wr[b]*100:.0f}%" for b in range(4)],
                            fontsize=8.4)
        dax.set_xlabel("同上四分位", fontsize=9.2)
        dax.set_ylabel("配对 Δ soft-IoU 中位\n(VLM14 − Geo8)", fontsize=9.2)
        dax.set_title(f"总体配对 Δ={np.median(delta):+.4f} · 胜率 "
                      f"{float((delta>0).mean())*100:.0f}% · Wilcoxon p={stat.pvalue:.1e}",
                      fontsize=9.4, pad=10)
        dax.grid(alpha=0.25, lw=0.6, axis="y")
        dax.tick_params(labelsize=8.4)

        pop_txt = (
            f"总体（候选全集 n={len(pg)}）：soft-IoU 中位 Geo8 {np.median(pg):.3f} / "
            f"VLM14 {np.median(pv):.3f}，配对 Δ={np.median(delta):+.4f}，"
            f"VLM14 胜率 {float((delta>0).mean())*100:.0f}%，Wilcoxon p={stat.pvalue:.1e}；"
            "按 GT 形状复杂度四分位分桶，Δ 依次为 "
            + " / ".join(f"{v:+.3f}" for v in dm) + "（左=最简单，右=最复杂）。")

    caption = fig.add_subplot(gs[nrow + 2, 2:6])
    caption.axis("off")
    caption.text(
        0.0, 1.02,
        wrap_cjk(
        "选源规则（effect-blind，预注册，脚本内不看任何一臂预测）："
        "metacanvas-local-l1l6-v2 的 val 池（两臂 train-pool=fit / eval-pools=select，"
        "val 全程未进训练、未进 ckpt 选择）∩ RO-9c 已冻结的 214 源批"
        f"（长宽比一致断言 0 例不符），得 {picks['n_candidate_records']} 条候选 / "
        f"{picks['n_candidate_sources']} 源；再要求 GT(.cgt) 面积 ∈[5%,45%] 且最大连通块 ≥40%，"
        "按 GT 形状复杂度（周长 / 2√(π·面积)，圆=1）降序，L1–L6 各取第 1 名、源不重复。"
        f"与 RO-9c effect-blind 清单的交集：{blind if blind else '无'}。\n"
        "坐标对齐（两条 16×16 网格前向路径不同，逆映射分别做、不混用）："
        "Geo8/VLM14 的网格覆盖『等比压扁成方形的画幅』（解析基取自 source_rgb 的 area 下采样，"
        "训练时也按此与 .cgt 比对）⇒ 逆映射 = 逐轴等比例展开，与训练侧 bilinear 升采样同一算子；"
        "raw attention 的网格覆盖『expand2square 黑边 pad 后的方形』⇒ 逆映射 = 还原 pad → 裁掉黑边"
        "（RO-9c make_figs.grid_to_img，luma_to_grid 的严格逆）。直接 resize 有约 1/3 的系统性错位，本图未使用。\n"
        "色标口径：第 3 列 = 注意力密度（分母只取有效格；pad 格既不参与色标也不填补，已被逆映射裁出画面，"
        "另在缩略图里画白打叉）；第 4/5 列 = 固定绝对 [0,1]，两臂共用；第 6 列 = 发散色图零值居中、全图共用上限。"
        "着色归着色、算数归算数：图上所有数字取自未归一化的原始场。"
        "soft-IoU = 实验原式（128×128 方形，概率场）；IoU@k = 面积匹配 top-k 的 hard IoU"
        "（显示分辨率，三列同一 GT、同一规则，禁逐场调阈值）。全图未使用任何 AUC。\n"
        f"※ 这 6 例是「GT 形状复杂度最高」的极端子样本，方向不等于总体：本页 6 例里 VLM14 在 "
        f"{int((vlm > geo).sum())}/6 例上 soft-IoU 更高（中位 Geo8 {np.median(geo):.3f} / "
        f"VLM14 {np.median(vlm):.3f}），而已冻结的完整 select（4225 条）是 "
        "Geo8 0.528 > VLM14 0.503。" + pop_txt + "\n"
        "读图要点：① 绿/红虚线 = GT 轮廓，直接看两臂贴不贴合；"
        "② 第 4/5 列共用同一把固定色标，形状差异肉眼可比（旧图各自归一化正是差异被抹掉的原因）；"
        "③ 第 6 列蓝 = 语义基把概率往下压、红 = 往上抬 —— 看它压/抬的位置是否落在 GT 轮廓内外；"
        "④ 第 3 列表明冻结 VLM 的原始注意力本身也不贴合轮廓（IoU@k 中位 "
        f"{np.median(attnk):.3f}，低于两臂的 {np.median(geok):.3f} / {np.median(vlmk):.3f}），"
        "它是主体先验而非指令 grounding（RO-9c 已判 FAIL）。"),
        fontsize=9.7, va="top", linespacing=1.58, color="#1a1a1a", wrap=False)

    fig.savefig(OUT, dpi=130, facecolor="white")
    print(f"[draw] -> {OUT}", flush=True)

    summary = {
        "rows": [{
            "level": row["level"], "uid": row["uid"], "source_id": row["source_id"],
            "shape_complexity": row["shape_complexity"],
            "gt_area_frac": row["gt_area_frac"], "gt_components": row["gt_components"],
            "geo8_soft_iou": row["Geo8_softiou"], "vlm14_soft_iou": row["VLM14_softiou"],
            "geo8_iou_at_k": row["Geo8_iouk"], "vlm14_iou_at_k": row["VLM14_iouk"],
            "raw_attn_iou_at_k": row["attn_iouk"],
            "pad_cells": row["pad_cells"], "pad_mass_frac": row["pad_mass"],
            "geo8_coeff": row["Geo8_coeff"].tolist(),
            "vlm14_coeff": row["VLM14_coeff"].tolist(),
            "diff_abs_max": float(np.abs(row["diff"]).max()),
        } for row in rows],
        "diff_shared_limit": dlim,
        "arms": meta["arms"],
        "ro9c": {"cond": RO9C_COND, "token": RO9C_TOKEN_NAME,
                 "layers": "0-23 mean", "readout": "raw (no common-mode removal)"},
    }
    (HERE / "P_basis_compare_redraw.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for row in summary["rows"]:
        print(f"  {row['level']} {row['source_id'][:20]:22s} "
              f"soft-IoU geo={row['geo8_soft_iou']:.3f} vlm={row['vlm14_soft_iou']:.3f} "
              f"| IoU@k attn={row['raw_attn_iou_at_k']:.3f} "
              f"geo={row['geo8_iou_at_k']:.3f} vlm={row['vlm14_iou_at_k']:.3f} "
              f"| pad {row['pad_cells']}/256 mass={row['pad_mass_frac']:.2f}", flush=True)


# ==========================================================================
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                        choices=("all", "picks", "infer", "pop", "draw", "single"))
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--clean", action="store_true",
                        help="单页图只留热力图 + 一行短列标题 + 色标条："
                             "去大标题/副标题、列标题小字、底部说明段落、格内数值、"
                             "以及第 3–6 列上的 GT 轮廓线（保留补边格白色打叉）。"
                             "不加该开关即复现原版式。")
    args = parser.parse_args()

    if args.stage in ("all", "picks") and (args.refresh or not PICKS.exists()):
        pick_sources()
    picks = json.loads(PICKS.read_text(encoding="utf-8"))
    if args.stage in ("all", "infer") and (args.refresh or not CACHE.exists()):
        run_inference(args.device)
    if args.stage in ("all", "pop") and (args.refresh or not POP.exists()):
        run_population(args.device)
    if args.stage in ("all", "single"):
        draw_single(picks, clean=args.clean)
    if args.stage in ("all", "draw"):
        draw(picks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
