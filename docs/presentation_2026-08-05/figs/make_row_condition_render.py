"""图 6 的单行版：端到端训练后三档条件输入的真实渲染结果。

原图 `assets/P53a_condition_render_compare.png` 是「6 行（L1..L6 各 1 例）× 8 列」的联图，
且每格下方挂着 Δ 图与三行数字。评审第二轮要求：一图一行、画面里只有图像内容与一行列标题。
故拆成三张互不重叠的单行图：

  row1  L2 源 · 8 列渲染 / 掩膜   —— 支撑「指令没改『改哪』」
  row2  L2 源 · 5 列 Δ 图         —— 支撑「改的是『改多狠』」（Δ 是图像内容，不是标注）
  row3  L6 源 · 5 列 Δ 图         —— 报告 §5.3 已声明的方向相反反例，照登

选源规则（写在 ROW_FIGURE_MAP.md，不进图）：L2 取冻结清单里该层首例；L6 是 6 行里
唯一一行「打乱指令的 Δ 幅度反而更大」，属预先声明的反例，规则是反例必须照登。

⚑ 零 GPU：只读 `_cache_P_condition_render_compare.npz`（三臂推理结果已落盘），
   Δ 图沿用同一常量增益公式（全图共用一个常量），禁逐图归一化。

用法：/home/bc/miniconda3/bin/python make_row_condition_render.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import rowlib
from make_P_condition_render_compare import (ARMS, CACHE, CACHE_JSON, RUNS,
                                             delta_img, gray_img, rgb_img)

HERE = Path(__file__).resolve().parent
CELL = 440

COLS_RENDER = ["输入", "目标", "真实指令 → 渲染", "别的样本的指令 → 渲染",
               "全体相同中性句 → 渲染", "真值掩膜",
               "真实指令 → 预测掩膜", "中性句 → 预测掩膜"]
COLS_DELTA = ["输入", "目标 Δ", "真实指令 Δ", "别的样本的指令 Δ", "全体相同中性句 Δ"]


def main() -> int:
    z = np.load(CACHE)
    meta = json.loads(CACHE_JSON.read_text(encoding="utf-8"))
    levels = meta["level"]
    uids = meta["uid"]
    source, target, gt_mask = z["source"], z["target"], z["gt_mask"]

    picked_all = [next(i for i, lv in enumerate(levels) if lv == L)
                  for L in sorted(set(levels))][:6]
    gain = float(np.round(0.40 / max(np.percentile(
        np.abs(target[picked_all] - source[picked_all]), 90), 1e-6), 1))

    n_all = len(uids)
    var_all = np.stack([[float(np.var(z[f"{a}__pred"][i] - source[i])
                               / max(np.var(target[i] - source[i]), 1e-12))
                         for a, _, _ in ARMS] for i in range(n_all)])
    win_all = int((var_all[:, 0] > var_all[:, 1:].max(1)).sum())
    mask_corr = float(np.median([
        np.corrcoef(z[f"{ARMS[0][0]}__mask"][i].ravel(),
                    z[f"{a}__mask"][i].ravel())[0, 1]
        for a, _, _ in ARMS[1:] for i in range(n_all)]))

    def idx_of(level):
        return next(i for i, lv in enumerate(levels) if lv == level)

    def render_row(idx) -> list:
        return [rgb_img(source[idx]), rgb_img(target[idx]),
                *[rgb_img(z[f"{a}__pred"][idx]) for a, _, _ in ARMS],
                gray_img(gt_mask[idx]),
                gray_img(z[f"{ARMS[0][0]}__mask"][idx]),
                gray_img(z[f"{ARMS[2][0]}__mask"][idx])]

    def delta_row(idx) -> list:
        return [rgb_img(source[idx]),
                delta_img(target[idx], source[idx], gain),
                *[delta_img(z[f"{a}__pred"][idx], source[idx], gain)
                  for a, _, _ in ARMS]]

    plan = [("P53a_condition_render_compare_row1.png", "L2", render_row, COLS_RENDER),
            ("P53a_condition_render_compare_row2.png", "L2", delta_row, COLS_DELTA),
            ("P53a_condition_render_compare_row3.png", "L6", delta_row, COLS_DELTA)]

    facts = []
    for name, level, builder, cols in plan:
        idx = idx_of(level)
        print("wrote", rowlib.compose_row(builder(idx), cols, name, cell=CELL))
        vr = {a: float(np.var(z[f"{a}__pred"][idx] - source[idx])
                       / max(np.var(target[idx] - source[idx]), 1e-12))
              for a, _, _ in ARMS}
        facts.append({
            "fig": name, "level": level, "uid": uids[idx],
            "delta_gain": gain,
            "soft_iou": {a: round(float(z[f"{a}__iou"][idx]), 4) for a, _, _ in ARMS},
            "de00": {a: round(float(z[f"{a}__de00"][idx]), 3) for a, _, _ in ARMS},
            "delta_var_ratio": {a: round(vr[a], 4) for a in vr},
            "real_instruction_strongest": bool(
                vr[ARMS[0][0]] > max(vr[a] for a, _, _ in ARMS[1:])),
            "instruction_real": meta["instruction_real"][idx],
            "instruction_shuffle": meta["instruction_shuffle"][idx],
        })

    agg = {s[1]: json.loads((RUNS / s[0] / "metrics.json").read_text())["final"]["select"]
           for s in ARMS}
    (HERE / "row_facts_condition_render.json").write_text(
        json.dumps({"rows": facts,
                    "pool_level": {k: {kk: v[kk] for kk in
                                       ("n", "soft_iou_p50", "psnr_in_p50",
                                        "de00_p50", "var_ratio") if kk in v}
                                   for k, v in agg.items()},
                    "mask_corr_median_20": round(mask_corr, 4),
                    "real_instruction_strongest_count": f"{win_all}/{n_all}",
                    "neutral_instruction": meta["neutral_instruction"]},
                   ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
