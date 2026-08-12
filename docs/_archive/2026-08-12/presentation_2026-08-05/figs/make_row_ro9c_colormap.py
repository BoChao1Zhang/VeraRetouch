"""图 2(b) / 图 2(b)-L 的单行版：RO-9c 七列着色消融，每源一页。

原图 `experiments/RO9c_subject_repro_20260805/viz/diag_raw_colormap.png`
是「5 行 effect-blind 源 × 7 列」的联图。评审要求一页一图、一图一行，故本脚本
按 `config/figure_picks.json` 的**清单顺序**逐源出单行图（不按效果挑）。

⚑ 只读 RO-9c 的落盘 stacks 与配置，零推理、零 GPU；着色逻辑直接调用
   `diag_raw_colormap.py` 的同名实现（`smooth_masked` + `make_figs.grid_to_img`），
   **不另立色标**。判据数字（AUC）用未归一化原始场重算，与原图一致。

用法：/home/bc/miniconda3/bin/python make_row_ro9c_colormap.py [--rows 3]
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

from analyze_g1 import roc_auc                                    # noqa: E402
from analyze_ro9c import GRID, PRIMARY_COND, TOKEN_TAGS, field, load_stacks  # noqa: E402
import make_figs as MF                                            # noqa: E402
from diag_raw_colormap import smooth_masked                       # noqa: E402

import rowlib                                                     # noqa: E402

COLS = [
    "原图",
    "原始着色（逐字原版）",
    "仅修正网格对齐",
    "仅色标取有效格",
    "秩次着色",
    "共模消除档",
    "真值掩膜",
]
TOKEN = {
    "colortemp": ("GC", "<retouch_color&temp>", "P3c_seven_col_ablation_GC"),
    "light": ("L", "<retouch_light>", "P3d_seven_col_ablation_L"),
}


def _jet_tile(m16, img, valid=None, aligned=True):
    from PIL import Image
    v = np.maximum(m16, 0.0)
    m = smooth_masked(v, valid, 1) if valid is not None else MF.smooth(v, 1)
    stat = m[valid] if valid is not None else m
    lo, hi = float(np.nanmin(stat)), float(np.nanmax(stat))
    mm = np.clip((np.nan_to_num(m, nan=lo) - lo) / (hi - lo + 1e-8), 0, 1)
    if aligned:
        heat = Image.fromarray(MF.jet(np.clip(
            MF.grid_to_img(mm.astype(np.float32), img), 0, 1)))
    else:
        heat = Image.fromarray(MF.jet(mm)).resize(img.size, Image.Resampling.BICUBIC)
    return Image.blend(img.convert("RGB"), heat, 0.55)


def _rank_tile(f16, valid, img):
    from PIL import Image
    from scipy.stats import rankdata
    r = np.zeros(GRID * GRID, dtype=np.float64)
    sel = valid.ravel()
    r[sel] = (rankdata(f16.ravel()[sel]) - 1) / max(1, sel.sum() - 1)
    a = MF.grid_to_img(r.reshape(GRID, GRID).astype(np.float32), img)
    return Image.blend(img.convert("RGB"),
                       Image.fromarray(MF.jet(np.clip(a, 0, 1))), 0.55)


def _gt_tile(img, mask16):
    from PIL import Image
    m = MF.grid_to_img(mask16.astype(np.float32), img)
    base = np.asarray(img.convert("RGB"), dtype=np.float32)
    red = base.copy()
    red[..., 0] = np.minimum(255, red[..., 0] + 110)
    a = np.clip(m, 0, 1)[..., None] * 0.55
    return Image.fromarray((base * (1 - a) + red * a).astype(np.uint8))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=3, help="每个 token 出几张单行图")
    args = ap.parse_args()

    recs = load_stacks([RO9C / "run"])
    region = json.loads((RO9C / "config" / "g1_region_opp.SNAPSHOT.json").read_text())
    reg_by_id = {r["img_id"]: r for r in region}
    picks_cfg = json.loads((RO9C / "config" / "figure_picks.json").read_text())
    picks = picks_cfg["picked"]
    with np.load(G1BDIR / "config" / "subject_masks16.npz") as z:
        masks = {k: z[k] for k in z.files}
    lv = {}
    with np.load(G1BDIR / "config" / "luma_valid16.npz") as z:
        for iid in {k.rsplit("|", 1)[0] for k in z.files}:
            lv[iid] = (z[f"{iid}|l"], z[f"{iid}|v"])

    written, facts = [], []
    for tag, (short, tokname, stem) in TOKEN.items():
        ti = TOKEN_TAGS.index(tag)
        n = 0
        for iid in picks:
            if n >= args.rows:
                break
            rec = recs.get(f"{iid}__{PRIMARY_COND}")
            if rec is None:
                continue
            n += 1
            img = MF.open_512(reg_by_id[iid]["img_path"])
            valid = lv[iid][1]
            gt = (masks[iid] >= 0.5) & valid
            fr, fm = field(rec, "raw", ti), field(rec, "m1", ti)
            sel = valid.ravel()
            a_raw = roc_auc(fr.ravel()[sel], gt.ravel()[sel])
            a_m1 = roc_auc(fm.ravel()[sel], gt.ravel()[sel])
            padmass = float(fr[~valid].sum() / fr.sum()) if (~valid).any() else 0.0
            in_pad = not bool(valid.ravel()[int(np.argmax(fr))])

            tiles = [img,
                     _jet_tile(fr, img, None, aligned=False),
                     _jet_tile(fr, img, None, aligned=True),
                     _jet_tile(fr, img, valid, aligned=True),
                     _rank_tile(fr, valid, img),
                     _jet_tile(fm, img, valid, aligned=True),
                     _gt_tile(img, masks[iid])]
            facts.append({"fig": f"{stem}_row{n}.png", "token": short,
                          "src": iid,
                          "subject_area": round(float(
                              reg_by_id[iid]["subject_area"]), 3),
                          "pad_mass_share": round(padmass, 4),
                          "argmax_in_pad": in_pad,
                          "auc_raw": round(float(a_raw), 4),
                          "auc_m1": round(float(a_m1), 4)})
            written.append(rowlib.compose_row(
                tiles, COLS, f"{stem}_row{n}.png", cell=512))
    (HERE / "row_facts_ro9c.json").write_text(
        json.dumps({"picks_rule": picks_cfg, "rows": facts},
                   ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for p in written:
        print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
