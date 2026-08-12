"""图 9 的单行版：同一层同一个头，只换 query token 的逐案例对照，一页一个源。

原图 `assets/P11b_query_position_cases.png` 是「5 行 effect-blind 源 × 4 列」的联图。
本脚本按 RO-9c 已冻结的 `config/figure_picks.json` **清单顺序**逐源出 1×4 单行图。

⚑ 零 GPU：只读 /var/cache/veradata/ro3_stacks_20260803 的落盘注意力栈。
   场的提取、D-0 修复、AUC、掩膜全部复用 make_P10_query_position.py 的同一实现，
   不另立一套；着色只用有效格定对称色标（禁逐图 min-max 让黑边格当分母）。

用法：/home/bc/miniconda3/bin/python make_row_query_position.py [--rows 3]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import rowlib
from make_P10_query_position import (G1, HEAD, LAYER, MASK_BIN_THR, MODE, RO9C,
                                     diff_auc_for, mpl)
from analyze_g1 import SubjectMaskBank  # noqa: E402

COLS = ["原图",
        "special token 当 query",
        "指令文本 token 当 query",
        "真值掩膜"]

CELL = 512
GRID = 16


def _cmap():
    import matplotlib
    matplotlib.use("Agg")
    try:
        return matplotlib.colormaps["RdBu_r"]
    except AttributeError:                       # 老版 matplotlib
        from matplotlib import cm
        return cm.get_cmap("RdBu_r")


_CMAP = _cmap()


def _grid_tile(rgb16: np.ndarray, valid: np.ndarray):
    """16×16 场 → 裁掉 expand2square 补边格后最近邻放大；不插值、不叠线。

    补边格不是数据（是预处理产生的纯黑填充），直接裁出视野；裁剪后长宽比与原图一致。
    """
    from PIL import Image
    ys, xs = np.where(valid)
    a = rgb16[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8)
    h, w = a.shape[:2]
    sc = CELL / min(h, w)
    return Image.fromarray(a, "RGB").resize(
        (int(round(w * sc)), int(round(h * sc))), Image.Resampling.NEAREST)


def render_row(iid, meta, g, k: int) -> Path:
    """直接用 PIL 画 1×4（最近邻放大，无插值、无白边、无叠线、无格内数字）。"""
    from PIL import Image

    src = Image.open(meta["img_path"]).convert("RGB")
    tiles = [src]
    for key in ("gl", "instr"):
        f = np.where(g["valid"], g["diff"][key], np.nan)
        lim = float(np.nanmax(np.abs(f))) if np.isfinite(f).any() else 1.0
        norm = np.clip((np.nan_to_num(g["diff"][key], nan=0.0) / (2 * lim)) + 0.5, 0, 1)
        rgb = (_CMAP(norm)[..., :3] * 255).astype(np.uint8)
        tiles.append(_grid_tile(rgb, g["valid"]))
    gray = np.clip(g["m16"], 0, 1)
    tiles.append(_grid_tile(
        (np.repeat(gray[..., None], 3, axis=-1) * 255).astype(np.uint8), g["valid"]))
    return rowlib.compose_row(tiles, COLS,
                              f"P11b_query_position_cases_row{k}.png", cell=440)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=3)
    args = ap.parse_args()

    bank = SubjectMaskBank()
    picks_cfg = json.loads((RO9C / "config/figure_picks.json").read_text())
    picks = picks_cfg["picked"]
    region = {r["img_id"]: r for r in
              json.loads((G1 / "config/g1_region_opp.json").read_text())}

    k, facts = 0, []
    for iid in picks:
        if k >= args.rows:
            break
        got = diff_auc_for(iid, bank)
        if got is None:
            print(f"  [skip] {iid}")
            continue
        k += 1
        facts.append({"fig": f"P11b_query_position_cases_row{k}.png", "src": iid,
                      "pool": region[iid]["pool"],
                      "subject_area": round(float(region[iid]["subject_area"]), 3),
                      "region_b_kind": region[iid]["region_b_kind"],
                      "auc_special_token_query": round(float(got["auc"]["gl"]), 4),
                      "auc_instruction_token_query": round(float(got["auc"]["instr"]), 4)})
        print("wrote", render_row(iid, region[iid], got, k))
    (Path(__file__).resolve().parent / "row_facts_query_position.json").write_text(
        json.dumps({"picks_rule": picks_cfg, "layer": LAYER, "head": HEAD,
                    "mode": MODE, "rows": facts},
                   ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
