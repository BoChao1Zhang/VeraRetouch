"""P5a（RO-9c 补件 D「四个场并排」）的单行版，一页一个源。

原图 `experiments/RO9c_subject_repro_20260805/viz/diag_four_fields.png` 是
「多行 effect-blind 源 × 6 列」的联图，且每格上叠了 top-k 预测实线 + GT 虚线 + 指标文字。
按评审第二轮要求，本脚本重绘为单行图：**只有图像内容 + 一行简短列标题**，
不叠任何轮廓线、不写格内数字；真值单独占最后一列。

着色规则逐字沿用 `diag_four_fields.field_tile`：只取有效格的 min-max、**不做 relu**
（否则全负的中心先验场会被压成空白），四个场同一条规则；判据数字不经过着色，
逐源指标写进 `row_facts_four_fields.json`。

⚑ 零 GPU：只读 RO-9c 落盘 stacks 与 per_source_four_field.json。

用法：/home/bc/miniconda3/bin/python make_row_four_fields.py [--rows 2]
"""
from __future__ import annotations

import sqlite3  # noqa: F401  isort:skip
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RO9C = REPO / "experiments" / "RO9c_subject_repro_20260805"
G1DIR = REPO / "experiments" / "G1_s_identifiability_20260803"
G1BDIR = REPO / "experiments" / "G1b_difflmm_20260803"
for p in (REPO / "tools" / "readout", G1DIR, RO9C, HERE):
    sys.path.insert(0, str(p))

from analyze_ro9c import GRID, PRIMARY_COND, load_stacks           # noqa: E402
import make_figs as MF                                             # noqa: E402
from diag_four_fields import FIELD_ORDER, build_fields             # noqa: E402
from diag_raw_colormap import smooth_masked                        # noqa: E402

import rowlib                                                      # noqa: E402

COLS = ["原图", "中心先验场（不看图像）", "raw L", "raw GC", "共模消除 GC", "真值掩膜"]


def plain_field_tile(f, valid, img):
    """只取有效格的 min-max（不 relu），叠回原图；无任何轮廓线、无文字。"""
    from PIL import Image
    m = smooth_masked(f.astype(np.float64), valid, 1)
    stat = m[valid]
    lo, hi = float(np.nanmin(stat)), float(np.nanmax(stat))
    mm = np.clip((np.nan_to_num(m, nan=lo) - lo) / (hi - lo + 1e-8), 0, 1)
    a = MF.grid_to_img(mm.astype(np.float32), img)
    return Image.blend(img.convert("RGB"),
                       Image.fromarray(MF.jet(np.clip(a, 0, 1))), 0.55)


def gt_tile(img, mask16):
    from PIL import Image
    m = MF.grid_to_img(mask16.astype(np.float32), img)
    base = np.asarray(img.convert("RGB"), dtype=np.float32)
    red = base.copy()
    red[..., 0] = np.minimum(255, red[..., 0] + 110)
    a = np.clip(m, 0, 1)[..., None] * 0.55
    return Image.fromarray((base * (1 - a) + red * a).astype(np.uint8))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2)
    args = ap.parse_args()

    recs = load_stacks([RO9C / "run"])
    reg_by_id = {r["img_id"]: r for r in json.loads(
        (RO9C / "config" / "g1_region_opp.SNAPSHOT.json").read_text())}
    picks_cfg = json.loads((RO9C / "config" / "four_field_picks.json").read_text())
    per = json.loads((RO9C / "per_source_four_field.json").read_text())["per_source"]
    with np.load(G1BDIR / "config" / "subject_masks16.npz") as z:
        masks = {k: z[k] for k in z.files}
    lv = {}
    with np.load(G1BDIR / "config" / "luma_valid16.npz") as z:
        for iid in {k.rsplit("|", 1)[0] for k in z.files}:
            lv[iid] = (z[f"{iid}|l"], z[f"{iid}|v"])

    yy, xx = np.mgrid[0:GRID, 0:GRID]
    center = -np.sqrt((yy - (GRID - 1) / 2.0) ** 2 + (xx - (GRID - 1) / 2.0) ** 2)

    facts, n = [], 0
    for iid in picks_cfg["picked"]:
        if n >= args.rows:
            break
        rec = recs.get(f"{iid}__{PRIMARY_COND}")
        if rec is None or iid not in per:
            continue
        n += 1
        img = MF.open_512(reg_by_id[iid]["img_path"])
        valid = lv[iid][1]
        fl = build_fields(rec, lv[iid][0], center)
        tiles = [img] + [plain_field_tile(fl[f], valid, img) for f in FIELD_ORDER] \
            + [gt_tile(img, masks[iid])]
        name = f"P5a_four_fields_row{n}.png"
        rowlib.compose_row(tiles, COLS, name, cell=512)
        facts.append({"fig": name, "src": iid,
                      "subject_area": round(float(reg_by_id[iid]["subject_area"]), 3),
                      "per_field": {f: {"auc": round(per[iid][f]["auc"], 4),
                                        "soft_iou": round(per[iid][f]["soft_iou"], 4),
                                        "bf1": round(per[iid][f]["bf1"], 4)}
                                    for f in FIELD_ORDER}})
        print("wrote", name)

    (HERE / "row_facts_four_fields.json").write_text(
        json.dumps({"picks_rule": picks_cfg["rule"], "rows": facts},
                   ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
