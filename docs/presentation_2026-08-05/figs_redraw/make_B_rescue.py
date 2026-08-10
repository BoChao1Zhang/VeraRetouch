"""B 组重绘 —— 汇报**图 5(a)(b)(c)**「修正前后的读出场对比」。

替换：`experiments/RO9b_readout_fix_20260803/viz/success_0*.png` /
`failure_instrblind_00_*.png`（原图 `imshow` 的 vmin/vmax 取全体格 ⇒ 分母被
`expand2square` 补边格支配：修正前的场里 **52% 的（relu）质量在补边格、57.5% 的源
全局 argmax 落在补边格**，有效区只占到整段色标的 66%）。

⚑ **场的来源换了，必须知会**：原图用的是 `RO9b/config/fields_final.npz` 的 `s_fin_*`，
   那份对应 **AUC_target 准则**终配置 `{pre, top3, wsum, a1, zlayer}`，定位质量中位
   AUC **0.7278**——**不是**汇报正文写的 0.93。正文的 0.93 出自 REPORT §2a **受限档**
   （禁学习式层加权，主叙事）`{qq, top3, best1, aff, zlayer=False}`，AUC **0.9200**，
   但该配置的逐源场当时**没有落盘**。本目录 `recompute_ro9b_restricted.py` 用已落盘的
   逐头栈**零 GPU 重算**了它（同 SEED、同 5 折 source-level CV、同 out-of-fold 纪律），
   实测复现 AUC **0.9200**、AUC_target **0.5042**，与 REPORT §2a 逐位一致。
   **本组图用重算出来的 §2a 场**，这样图与正文的 0.93 才对得上。

版式（评审 2026-08-05 定稿）：单行五列、只有一行短列标题和一根无刻度色标条，
**图内无图注、无轮廓线、无格内数字**。

色标口径（写进 REDRAW_NOTES，不写进图）：
  第 4/5 列（修正后 · 自己的指令 / 对立指令）**共用一把色标** —— 同一个算子、同一个
  值域，这一对必须共标，否则各自归一化会把"两条指令几乎一样"抹掉。
  第 3 列（修正前）是**另一个算子**（RO-9 原样 pre/head-mean/L8–15，值域 [−13.4, +1.3]，
  与修正后的 [−2.5, +1.9] 差一个数量级），强行共标只会把它压成纯色，故单独取自己的
  有效格色标；两列之间只比**形状**，不比数值。
"""
from __future__ import annotations

import sqlite3  # noqa: F401  isort:skip
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

import metrics as MT  # noqa: E402
import picks as PK  # noqa: E402
import redrawlib as RL  # noqa: E402

TITLES = ["输入", "真值", "修正前", "修正后", "修正后·对立指令"]


def short_id(iid: str) -> str:
    return iid if not iid.startswith("src_") else "src_" + iid[4:12]


def main() -> None:
    z = np.load(HERE / "fields_restricted_auc.npz", allow_pickle=True)
    ids = [str(x) for x in z["img_id"]]
    picked, pick_meta = PK.pick_sources()

    hf = json.loads(str(z["headline_fix"]))
    hb = json.loads(str(z["headline_base"]))
    notes = {
        "replaces": "BIWEEKLY_REPORT 图 5(a)(b)(c) / assets/P52a_,P52b_,P52c_*.png",
        "source_rule": pick_meta,
        "field_before": {"cfg": json.loads(str(z["cfg_base"])),
                         "desc": "RO-9 原样：pre / head-mean / L8–15 / D-0（无任何拟合）"},
        "field_after": {"cfg": json.loads(str(z["cfg_fix"])),
                        "desc": "RO-9b REPORT §2a 受限档终配置（主叙事）；本目录零 GPU 重算，"
                                "5 折 source-level CV out-of-fold",
                        "layers_per_fold": json.loads(str(z["layers_per_fold"]))},
        "reproduction_check": {
            "recomputed_auc_after": hf["auc"], "report_2a_auc": 0.9200006817329652,
            "recomputed_auc_target_after": hf["auc_target_bg"],
            "report_2a_auc_target": 0.5041547435615232,
            "recomputed_auc_before": hb["auc"], "report_S0_auc": 0.6607084611587151,
            "note": "⚠️ AUC 仅用于与 REPORT 已落盘数字对表（判据不完整，结论限于排序）；"
                    "判据一律看下面的 soft-IoU / grid 边界 F1 / 中心先验列"},
        "colour_scale": "第 4、5 列共用一把色标（同算子同值域）；第 3 列因算子与值域不同"
                        "单独取自己的有效格色标，与第 4 列只比形状不比数值",
        "figures": [],
    }

    for iid in picked:
        i = ids.index(iid)
        valid, m_soft = z["valid16"][i], z["mask16"][i]
        img = RL.open_short512(str(z["img_path"][i]))
        s_before, s_a, s_b = z["s_base_a"][i], z["s_fix_a"][i], z["s_fix_b"][i]

        (nbef,), lo0, hi0 = RL.valid_norm([s_before], valid)      # 修正前：自己的色标
        (na, nb), lo1, hi1 = RL.valid_norm([s_a, s_b], valid)     # 修正后一对：共标
        tiles = [RL.plain_image(img, valid),
                 RL.region_over_image(m_soft, valid, img),
                 RL.heat_over_image(nbef, valid, img, lo0, hi0),
                 RL.heat_over_image(na, valid, img, lo1, hi1),
                 RL.heat_over_image(nb, valid, img, lo1, hi1)]
        out = HERE / f"B_rescue_{short_id(iid)}.png"
        RL.row_figure(tiles, TITLES, out)

        # ---- 算数（未归一化原始场；面积匹配 top-k，纯秩次；着色完全不参与）----
        d = np.abs(np.clip((na - lo1) / (hi1 - lo1), 0, 1)
                   - np.clip((nb - lo1) / (hi1 - lo1), 0, 1))[valid]
        notes["figures"].append({
            "file": out.name, "img_id": iid, "pool": str(z["pool"][i]),
            "region_b_kind": str(z["region_b_kind"][i]),
            "instr_a": str(z["instr_a"][i]), "instr_b": str(z["instr_b"][i]),
            "pad_cells": int((~valid).sum()),
            "before": MT.score(s_before, m_soft, valid),
            "after": MT.score(s_a, m_soft, valid),
            "after_opposite_instr": MT.score(s_b, m_soft, valid),
            "center_prior": MT.score(MT.CENTER_PRIOR, m_soft, valid),
            "rho_after_a_vs_b": MT.pearson_valid(s_a, s_b, valid),
            "rho_before_a_vs_b": MT.pearson_valid(s_before, z["s_base_b"][i], valid),
            "d_norm_max_after": float(np.nanmax(d)),
            "s_range_before_all": [float(s_before.min()), float(s_before.max())],
            "s_range_before_valid": [float(s_before[valid].min()),
                                     float(s_before[valid].max())],
            "s_range_after_valid": [float(min(s_a[valid].min(), s_b[valid].min())),
                                    float(max(s_a[valid].max(), s_b[valid].max()))],
        })
        f = notes["figures"][-1]
        print(f"wrote {out.name}  soft-IoU {f['before']['soft_iou']:.3f}→"
              f"{f['after']['soft_iou']:.3f} (中心先验 {f['center_prior']['soft_iou']:.3f}) "
              f"| bF1 {f['before']['bf1_grid']:.3f}→{f['after']['bf1_grid']:.3f} "
              f"| ρ(A,B)后 {f['rho_after_a_vs_b']:.4f}", flush=True)

    # ---- 全批统计（212 源），供 REDRAW_NOTES / 正文引用 ----
    from scipy.stats import wilcoxon

    acc = {k: [] for k in ("before_iou", "after_iou", "afterB_iou", "center_iou",
                           "before_bf1", "after_bf1", "center_bf1", "rho_after",
                           "rho_before")}
    for i in range(len(ids)):
        v, ms = z["valid16"][i], z["mask16"][i]
        for tag, f in (("before", z["s_base_a"][i]), ("after", z["s_fix_a"][i]),
                       ("afterB", z["s_fix_b"][i]), ("center", MT.CENTER_PRIOR)):
            sc = MT.score(f, ms, v)
            acc[f"{tag}_iou"].append(sc["soft_iou"])
            if tag != "afterB":
                acc[f"{tag}_bf1"].append(sc["bf1_grid"])
        acc["rho_after"].append(MT.pearson_valid(z["s_fix_a"][i], z["s_fix_b"][i], v))
        acc["rho_before"].append(MT.pearson_valid(z["s_base_a"][i], z["s_base_b"][i], v))

    def pair(a, b):
        d = np.array(a, float) - np.array(b, float)
        d = d[np.isfinite(d)]
        return {"median_delta": float(np.median(d)), "wilcoxon_p": float(wilcoxon(d).pvalue),
                "winrate": float((d > 0).mean()), "n": int(len(d))}

    notes["batch"] = {
        "n": len(ids),
        "soft_iou_median": {k: float(np.nanmedian(acc[f"{k}_iou"]))
                            for k in ("before", "after", "afterB", "center")},
        "bf1_grid_median": {k: float(np.nanmedian(acc[f"{k}_bf1"]))
                            for k in ("before", "after", "center")},
        "paired_after_minus_before_soft_iou": pair(acc["after_iou"], acc["before_iou"]),
        "paired_after_minus_center_soft_iou": pair(acc["after_iou"], acc["center_iou"]),
        "paired_after_minus_before_bf1": pair(acc["after_bf1"], acc["before_bf1"]),
        "paired_after_minus_center_bf1": pair(acc["after_bf1"], acc["center_bf1"]),
        "rho_opposite_instr_median": {"before": float(np.nanmedian(acc["rho_before"])),
                                      "after": float(np.nanmedian(acc["rho_after"]))},
        "n_rho_below_0.30": {
            "before": int(np.nansum(np.array(acc["rho_before"]) < 0.30)),
            "after": int(np.nansum(np.array(acc["rho_after"]) < 0.30))},
    }
    (HERE / "B_rescue_meta.json").write_text(
        json.dumps(notes, indent=1, ensure_ascii=False, default=float))
    b = notes["batch"]
    print("\n批中位 soft-IoU:", {k: round(v, 4) for k, v in b["soft_iou_median"].items()})
    print("批中位 bF1_grid:", {k: round(v, 4) for k, v in b["bf1_grid_median"].items()})
    print("ρ(指令A, 指令B) 中位:", {k: round(v, 4)
                                     for k, v in b["rho_opposite_instr_median"].items()},
          "| ρ<0.30 的源:", b["n_rho_below_0.30"])


if __name__ == "__main__":
    main()
