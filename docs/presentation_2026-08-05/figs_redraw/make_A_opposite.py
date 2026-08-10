"""A 组重绘 —— 汇报**图 4**「同图对立指令下的注意力场」。

替换：`experiments/G1_s_identifiability_20260803/viz/region_FAIL_rho1.00_maxrho_*.png`
（原图用 `imshow(s_canon, vmin/vmax 取全体格)`：分母被 `expand2square` 补边格支配 ——
实测补边格占 16×16 的中位 **37.5%**，90% 的源全局最小值、54% 的源全局最大值落在补边格，
有效区只占到整段色标的 **65.9%** ⇒ 有效区内的结构被压平，评审「看不到任何信息」。）

数据：G1 区域对立批 214 组（有效 210）「同图 + 两条指向相反区域的指令」，
`reg_a` 点名主体、`reg_b` 点名主体的补集（背景或空间对侧），
场 = canonical s（GL=`<retouch_light>`，L8–15 head-mean 平均），直接读 `run_regfull/stacks`。

版式（评审 2026-08-05 定稿）：单行五列、只有一行短列标题和一根无刻度色标条，
**图内无图注、无轮廓线、无格内数字**。全部口径与数字写在 `REDRAW_NOTES.md`。

选源（effect-blind，不看任何读出结果）：取 RO-9c 已冻结的
`RO9c_subject_repro_20260805/config/figure_picks.json` 的**三个来源池各一张**
（awards / unsplash / ppr10k，即冻结列表的前三项，不重排、不筛选）。
"""
from __future__ import annotations

import sqlite3  # noqa: F401  isort:skip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
G1 = REPO / "experiments" / "G1_s_identifiability_20260803"
G1B = REPO / "experiments" / "G1b_difflmm_20260803"
RO9C = REPO / "experiments" / "RO9c_subject_repro_20260805"

import metrics as MT  # noqa: E402
import picks as PK  # noqa: E402
import redrawlib as RL  # noqa: E402

TITLES = ["输入", "指令A 的场", "指令B 的场", "指令A 的目标", "指令B 的目标"]

# build_region_opp.py 的 COMPLEMENT 表 → 几何区域（原图坐标的行/列半区）
SPATIAL = {
    "lower": ("rows", 0.0, 0.5),          # reg_b = "the upper part of the image"
    "upper": ("rows", 0.5, 1.0),          # reg_b = "the lower part of the image"
    "left": ("cols", 0.5, 1.0),           # reg_b = "the right side of the image"
    "right": ("cols", 0.0, 0.5),          # reg_b = "the left side of the image"
    "lower left": ("quad", 0.0, 0.5, 0.5, 1.0),    # upper right
    "lower right": ("quad", 0.0, 0.5, 0.0, 0.5),   # upper left
    "upper left": ("quad", 0.5, 1.0, 0.5, 1.0),    # lower right
    "upper right": ("quad", 0.5, 1.0, 0.0, 0.5),   # lower left
}


def instr_hash(s: str, n: int = 12) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:n]


def spatial_region16(subject_region: str, img) -> np.ndarray:
    """reg_b 的空间对侧区域 → 16×16 覆盖率（走 `luma_to_grid` 同一条 pad 语义）。"""
    from ro9_gl_attention import luma_to_grid

    w, h = img.size
    m = np.zeros((h, w), dtype=np.float32)
    spec = SPATIAL[subject_region]
    if spec[0] == "rows":
        m[int(spec[1] * h):int(spec[2] * h), :] = 1.0
    elif spec[0] == "cols":
        m[:, int(spec[1] * w):int(spec[2] * w)] = 1.0
    else:
        m[int(spec[1] * h):int(spec[2] * h), int(spec[3] * w):int(spec[4] * w)] = 1.0
    return luma_to_grid(m)[0]


def short_id(iid: str) -> str:
    return iid if not iid.startswith("src_") else "src_" + iid[4:12]


def main() -> None:
    sys.path.insert(0, str(REPO / "tools" / "readout"))
    region = json.loads((G1 / "config" / "g1_region_opp.json").read_text())
    by_id = {r["img_id"]: r for r in region}
    with np.load(G1B / "config" / "subject_masks16.npz") as z:
        masks = {k: z[k] for k in z.files}
    _ = RO9C

    picked, pick_meta = PK.pick_sources()
    for iid in picked:                       # 前置断言：三份产物齐全
        assert iid in masks, iid
        for t in ("reg_a", "reg_b"):
            p = G1 / "run_regfull" / "stacks" / \
                f"{iid}__{instr_hash(by_id[iid]['instructions'][t])}.npz"
            assert p.is_file(), p

    notes = {"replaces": "BIWEEKLY_REPORT 图 4 / assets/P8a_G1_two_opposite_instructions.png",
             "source_rule": pick_meta,
             "field": "canonical s（GL=<retouch_light>，L8–15 head-mean 平均），"
                      "G1/run_regfull/stacks 直接读，未做 relu",
             "colour": "两个场共用一把色标；统计量只取 valid 格；补边格白；"
                       "叠图走 grid_to_img 严格逆映射",
             "picked": [], "figures": []}

    for iid in picked:
        r = by_id[iid]
        img = RL.open_short512(r["img_path"])
        _, valid = RL.luma_valid(r["img_path"])
        sa = np.load(G1 / "run_regfull" / "stacks" /
                     f"{iid}__{instr_hash(r['instructions']['reg_a'])}.npz"
                     )["s_canon"].astype(np.float32)
        sb = np.load(G1 / "run_regfull" / "stacks" /
                     f"{iid}__{instr_hash(r['instructions']['reg_b'])}.npz"
                     )["s_canon"].astype(np.float32)
        subj = masks[iid].astype(np.float32)
        tgt_a = subj
        if r["region_b_kind"] == "background":
            tgt_b = np.clip(1.0 - subj, 0, 1) * valid
        else:
            tgt_b = spatial_region16(r["subject_region"], img)

        # ⚑ 两个场共用一把色标（否则各自归一化会把"它们几乎一样"这件事抹掉）
        (na, nb), lo, hi = RL.valid_norm([sa, sb], valid)
        tiles = [RL.plain_image(img, valid),
                 RL.heat_over_image(na, valid, img, lo, hi),
                 RL.heat_over_image(nb, valid, img, lo, hi),
                 RL.region_over_image(tgt_a, valid, img),
                 RL.region_over_image(tgt_b, valid, img)]
        out = HERE / f"A_opposite_{short_id(iid)}.png"
        RL.row_figure(tiles, TITLES, out)

        # ---- 算数（未归一化原始场，纯秩次口径；与着色完全无关）----
        ma = MT.score(sa, tgt_a, valid)
        mb = MT.score(sb, tgt_b, valid)
        mba = MT.score(sb, tgt_a, valid)
        cp_a = MT.score(MT.CENTER_PRIOR, tgt_a, valid)
        ra, rb = MT.rank_pct(sa, valid), MT.rank_pct(sb, valid)
        (za, zb), _lo, _hi = RL.valid_norm([sa, sb], valid)
        d = np.abs(np.clip((za - lo) / (hi - lo), 0, 1)
                   - np.clip((zb - lo) / (hi - lo), 0, 1))[valid]
        notes["figures"].append({
            "file": out.name, "img_id": iid, "pool": r["pool"],
            "region_b_kind": r["region_b_kind"], "subject_region": r["subject_region"],
            "subject_area": r["subject_area"], "winner_confidence": r["winner_confidence"],
            "instr_a": r["instructions"]["reg_a"], "instr_b": r["instructions"]["reg_b"],
            "rho_valid": MT.pearson_valid(sa, sb, valid),
            "d_rank_p50": float(np.nanmedian(np.abs(ra - rb)[valid])),
            "d_norm_max": float(np.nanmax(d)), "d_norm_p50": float(np.nanmedian(d)),
            "pad_cells": int((~valid).sum()),
            "s_range_all": [float(min(sa.min(), sb.min())), float(max(sa.max(), sb.max()))],
            "s_range_valid": [float(min(sa[valid].min(), sb[valid].min())),
                              float(max(sa[valid].max(), sb[valid].max()))],
            "A_on_targetA": ma, "B_on_targetB": mb, "B_on_targetA": mba,
            "center_prior_on_targetA": cp_a,
        })
        notes["picked"].append(iid)
        print(f"wrote {out.name}  rho={notes['figures'][-1]['rho_valid']:.4f}  "
              f"Δnorm_max={notes['figures'][-1]['d_norm_max']:.3f}", flush=True)

    (HERE / "A_opposite_meta.json").write_text(
        json.dumps(notes, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
