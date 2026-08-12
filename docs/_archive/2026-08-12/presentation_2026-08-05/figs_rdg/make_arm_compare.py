"""汇报 §7.2「参数生成端的结构选择」·跨臂渲染效果对比图（单行版式）。

三组图，只读 RDG_transformer_20260803 的产物：

  组 A  输入 | 目标 | MLP | Transformer                 —— 跨范式
  组 B  输入 | 目标 | MLP | 加宽 MLP                     —— 参数量翻 20 倍无用
  组 C  输入 | 目标 | Transformer-小[/中]/大             —— 容量阶梯

六臂共享 tokenizer 与 GLUT 渲染核心，唯一变量是中间的生成器；
参数由 128x128 的 (I_in, after) 对产生（模型的真实输入不变），
显示时把逐像素算子 f_theta 施加到高分辨率原图上——f_theta 只看颜色
（红线：逐像素算子禁 (x,y)/邻域/MLP/排序），所以换分辨率不改变任何数值。

选源规则（effect-blind，三档，见 NOTES.md §2）：

  profile "mid"   第一轮组 A 用。val_lut 池 harness 同一批 1024 行，按
                  `lut_de_ident` 升序取第 25/50/75 百分位最近秩。
  profile "high"  第二轮组 B / 组 C 用。两级纯数据键：① `lut_de_ident` 上尾 10%；
                  ② 池内按**图像空间编辑幅度**降序取前 3。
                  **已知系统性偏置**：编辑幅度上尾被"去色/黑白"类 LUT 占据
                  （把饱和色抹成灰的逐像素 dE00 必然大），第二轮三张里有两张
                  目标是纯黑白，无法体现"调色"。
  profile "color" 第三轮（2026-08-05）组 A/B/C 共用。在 "high" 的两级键之间插入
                  一道**目标图色彩性门**，只看 I_in / I_tar，不涉及任何被测臂输出：
                    c_tar  = 目标图逐像素 CIELab chroma sqrt(a*^2+b*^2) 的均值
                    c_res  = 目标图 chroma **离散度** = 逐像素 (a*,b*) 到该图
                             (a*,b*) 质心的平均距离
                    edit   = 128 缓存对上 输入<->目标 的逐像素 dE00 中位数
                  门：c_tar >= 12.0 且 c_res >= 12.0 且 edit >= 9.0。
                  c_res 是关键的一列：纯黑白与"单色调覆盖"(sepia / split-tone /
                  整体单一色)都会让它塌到 0，而"某个大面积单一物体很艳"(如红裙)
                  它仍然很高——所以它排掉的是"没有配色可看"，不是"不够艳"。
                  三个量都只用 imgs_in / imgs_after，排序里没有任何一臂的误差。

只读：runs/*/best.pt、cache/*。只写：本目录。
"""
from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

sys.path.insert(0, "/home/bc/VeraRetouch")
from model.glut_repro import data_rdg as D                        # noqa: E402
from model.glut_repro.model_rdg import (                          # noqa: E402
    RDGModel, delta_e00, render, srgb_to_lab,
)

EXP = "/home/bc/VeraRetouch/experiments/RDG_transformer_20260803"
OUT = "/home/bc/VeraRetouch/docs/presentation_2026-08-05/figs_rdg"

ARMS = ["mlp", "mlp_wide", "gtiny", "glite", "gbase"]

# 组名 -> (列, 列标题)
GROUPS = {
    "A_paradigm": (["__in__", "__gt__", "mlp", "gbase"],
                   ["输入", "目标", "MLP", "Transformer"]),
    "B_mlpwide": (["__in__", "__gt__", "mlp", "mlp_wide"],
                  ["输入", "目标", "MLP", "加宽 MLP"]),
    "C_ladder": (["__in__", "__gt__", "gtiny", "glite", "gbase"],
                 ["输入", "目标", "Transformer-小", "Transformer-中",
                  "Transformer-大"]),
}
# 组 C 的两档版本（--c-mode 2）：跳过中间档，只留 小 vs 大
C_TWO = (["__in__", "__gt__", "gtiny", "gbase"],
         ["输入", "目标", "Transformer-小", "Transformer-大"])

PCTS = (25, 50, 75)          # profile "mid"
HIGH_TAIL_FRAC = 0.10        # "high"/"color" 第一级：lut_de_ident 上尾比例

# profile "color" 的目标图色彩性门（阈值为整池常量，非逐图；见模块 docstring）
C_TAR_MIN = 12.0             # 目标图平均 chroma 下限
C_RES_MIN = 12.0             # 目标图 chroma 离散度下限（排除去色 + 单色调覆盖）
EDIT_MIN = 9.0               # 编辑幅度下限（排除"目标与输入几乎没变"）

# 第三轮组 A 保留的那一张（主 agent 指定，不重选）
A_KEEP_PCT = 50              # profile "mid" 的 p50 == row 326221

# 内容侧排除（非结果侧）。规则第 1 名 326774 的画面是人体裸露，不适合放进
# 汇报投影；该判断只看输入/目标缩略图，在任何一臂被前向之前做出，不看结果。
# 写在这里是为了让选源仍然完全确定、可复算、可撤销（--no-content-exclude）。
CONTENT_EXCLUDE = {326774: "画面为人体裸露，不宜进汇报投影"}


# --------------------------------------------------------------------- 版式
def cjk_font(size: int):
    for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
              "/home/bc/.local/share/fonts/windows/NotoSansSC-VF.ttf",
              "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"):
        if os.path.isfile(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def row_figure(tiles, titles, out_path, short_side=512, gap=16, margin=14,
               title_h=68, font_px=40):
    """单行联图：一行列标题 + N 张等高瓦片。图内无其他文字、无边框、无格内数值。"""
    fitted = []
    for t in tiles:
        w, h = t.size
        sc = short_side / min(w, h)
        fitted.append(t.resize((int(round(w * sc)), int(round(h * sc))),
                               Image.Resampling.LANCZOS))
    H = max(t.size[1] for t in fitted)
    xs, x = [], margin
    for t in fitted:
        xs.append(x)
        x += t.size[0] + gap
    W = x - gap + margin
    panel = Image.new("RGB", (W, title_h + H + margin), (255, 255, 255))
    d = ImageDraw.Draw(panel)
    font = cjk_font(font_px)
    for t, xx, tt in zip(fitted, xs, titles):
        panel.paste(t, (xx, title_h))
        try:
            bb = d.textbbox((0, 0), tt, font=font)
            tw, top = bb[2] - bb[0], bb[1]
        except Exception:
            tw, top = font_px * len(tt), 0
        d.text((xx + max(0, (t.size[0] - tw) // 2), title_h - font_px - 14 - top),
               tt, fill=(20, 20, 20), font=font)
    panel.save(out_path)
    return panel


# ----------------------------------------------------------------- 高清原图
def _read_ref(ref):
    if ref[0] == "local":
        with open(ref[1], "rb") as f:
            return f.read()
    with open(ref[1], "rb") as f:
        f.seek(ref[2])
        return f.read(ref[3])


def hires(ref, short_side):
    """原始 JPEG -> exif_transpose -> 面积平均降采样到短边 short_side（保持宽高比）。

    面积平均（Image.BOX）与 build_cache.py 的重采样口径一致，颜色统计无偏。
    """
    im = Image.open(io.BytesIO(_read_ref(ref)))
    im = ImageOps.exif_transpose(im).convert("RGB")
    w, h = im.size
    sc = short_side / min(w, h)
    if sc < 1.0:
        im = im.resize((max(1, int(round(w * sc))), max(1, int(round(h * sc)))),
                       Image.Resampling.BOX)
    return im


def apply_f(p, arr, dev, chunk=131072):
    """f_theta 逐像素施加到 (H,W,3) float32 [0,1] 上 -> uint8 (H,W,3)。"""
    H, W, _ = arr.shape
    x = torch.from_numpy(arr.reshape(1, -1, 3)).to(dev)
    outs = []
    for i in range(0, x.shape[1], chunk):
        outs.append(render(p, x[:, i:i + chunk]).clamp(0, 1))
    y = torch.cat(outs, 1).reshape(H, W, 3)
    return y, (y.cpu().numpy() * 255.0 + 0.5).astype(np.uint8)


# ------------------------------------------------------------------- 选源
def load_ident(cache, eval_n):
    """(rows, lut_de_ident)。该量是纯数据属性；断言各臂逐样本一致，
    防止误把某一臂的结果当成选源依据。"""
    rows = cache.idx_of("val_lut")[:eval_n]
    ident = np.load(os.path.join(EXP, "runs/gbase/per_sample.npz"))["lut_de_ident"]
    for a in ("mlp", "mlp_wide", "gtiny", "glite"):
        f = os.path.join(EXP, f"runs/{a}/per_sample.npz")
        if os.path.isfile(f):
            other = np.load(f)["lut_de_ident"]
            assert np.allclose(ident, other), f"lut_de_ident differs on {a}"
    assert len(ident) == len(rows)
    return rows, ident


def edit_magnitude(cache, rows, dev, bs=64):
    """图像空间编辑幅度：128 缓存对上 输入↔目标 的逐像素 dE00 中位数。
    纯数据属性（只用 in / after，不含任何模型输出）。"""
    out = np.zeros(len(rows), dtype=np.float32)
    for i in range(0, len(rows), bs):
        rr = rows[i:i + bs]
        a = torch.from_numpy(
            np.asarray(cache.after[rr], dtype=np.float32) / 255.0).to(dev)
        s = torch.from_numpy(
            np.asarray(cache.imgin[cache.group[rr]],
                       dtype=np.float32) / 255.0).to(dev)
        out[i:i + len(rr)] = delta_e00(s, a).reshape(len(rr), -1) \
            .median(dim=1).values.cpu().numpy()
    return out


def chroma_stats(cache, rows, dev, bs=64):
    """目标图色彩性，两列，都是纯数据属性（只用 imgs_after）：

      c_tar  逐像素 chroma sqrt(a*^2+b*^2) 的均值
      c_res  逐像素 (a*,b*) 到该图 (a*,b*) 质心的平均距离（chroma 离散度）

    c_res 才是"有没有配色可看"的那一列：纯黑白 -> 0；单一色调覆盖（sepia /
    split-tone / 整体偏一个色）质心被推远、残差塌掉 -> 也接近 0；
    而"某个大面积物体很艳"（红裙）残差仍然很大。
    """
    ct = np.zeros(len(rows), dtype=np.float32)
    cr = np.zeros(len(rows), dtype=np.float32)
    for i in range(0, len(rows), bs):
        rr = rows[i:i + bs]
        a = torch.from_numpy(
            np.asarray(cache.after[rr], dtype=np.float32) / 255.0).to(dev)
        lab = srgb_to_lab(a).reshape(len(rr), -1, 3)
        aa, bb = lab[..., 1], lab[..., 2]
        ct[i:i + len(rr)] = torch.sqrt(aa * aa + bb * bb).mean(1).cpu().numpy()
        da = aa - aa.mean(1, keepdim=True)
        db = bb - bb.mean(1, keepdim=True)
        cr[i:i + len(rr)] = torch.sqrt(da * da + db * db).mean(1).cpu().numpy()
    return ct, cr


def _dedup_take(cache, rows, order, n_pick, start=0, exclude=()):
    """沿 order 前进取 n_pick 行，LUT 与源组互不重复。"""
    picks, seen_p, seen_g = [], set(), set()
    for k in range(start, len(order)):
        j = int(order[k])
        r = int(rows[j])
        pn, gp = cache.presets[int(cache.preset[r])], int(cache.group[r])
        if pn in seen_p or gp in seen_g:
            continue
        seen_p.add(pn)
        seen_g.add(gp)
        if r in exclude:                       # 去重口径照常吃掉它，只是不入选
            continue
        picks.append({"row": r, "j": j, "preset": pn, "group": gp,
                      "conf": int(cache.conf[r])})
        if len(picks) == n_pick:
            break
    return picks


def pick_mid(cache, rows, ident, n_pick=3):
    """profile "mid"（组 A，2026-08-05 第二轮未改动）：ident 升序 p25/p50/p75。"""
    order = np.argsort(ident, kind="stable")
    picks, seen_p, seen_g = [], set(), set()
    for q in PCTS[:n_pick]:
        k0 = int(round(q / 100.0 * (len(rows) - 1)))
        for k in range(k0, len(rows)):          # 撞重则沿确定性方向前进
            j = int(order[k])
            r = int(rows[j])
            pn, gp = cache.presets[int(cache.preset[r])], int(cache.group[r])
            if pn in seen_p or gp in seen_g:
                continue
            seen_p.add(pn)
            seen_g.add(gp)
            picks.append({"profile": "mid", "pct": q, "rank": k, "row": r,
                          "preset": pn, "group": gp,
                          "ident_de00": float(ident[j])})
            break
    return picks


def pick_high(cache, rows, ident, edit, n_pick=3):
    """profile "high"（第二轮组 B / C）：ident 上尾 10% 内，按编辑幅度降序取前 3。"""
    order = np.argsort(ident, kind="stable")
    tail = order[int(np.ceil((1.0 - HIGH_TAIL_FRAC) * len(rows))):]
    # 降序 edit；同值以 row 号升序打破 -> 完全确定
    tail_sorted = np.array(sorted(tail, key=lambda j: (-float(edit[j]),
                                                       int(rows[j]))))
    out = []
    for p in _dedup_take(cache, rows, tail_sorted, n_pick):
        j = p.pop("j")
        out.append({"profile": "high",
                    "rank_edit_in_tail": len(out),
                    "ident_pct": round(float((ident <= ident[j]).mean()) * 100, 2),
                    "edit_pct": round(float((edit <= edit[j]).mean()) * 100, 2),
                    "ident_de00": float(ident[j]),
                    "edit_de00_128": float(edit[j]), **p})
    return out


def pick_color(cache, rows, ident, edit, c_tar, c_res, n_pick=3,
               content_exclude=True):
    """profile "color"（第三轮，A/B/C 共用）。

    ① 第一级键不变：`lut_de_ident` 上尾 10%（锁住 LUT 本身够狠）。
    ② **新增目标图色彩性门**（只看 I_in / I_tar）：
         c_tar >= C_TAR_MIN and c_res >= C_RES_MIN and edit >= EDIT_MIN
    ③ 第三级键不变：门内按编辑幅度降序，同值以 row 号升序打破，
       去重「LUT 互不相同、源组互不相同」，取前 n_pick。

    返回 (picks, stats)；stats 记录过滤掉多少候选、剩余池多大，写进 meta.json。
    """
    order = np.argsort(ident, kind="stable")
    tail = order[int(np.ceil((1.0 - HIGH_TAIL_FRAC) * len(rows))):]
    gate = [int(j) for j in tail
            if c_tar[j] >= C_TAR_MIN and c_res[j] >= C_RES_MIN
            and edit[j] >= EDIT_MIN]
    col_ok = np.array([c_tar[j] >= C_TAR_MIN and c_res[j] >= C_RES_MIN
                       for j in tail])
    ed_ok = np.array([edit[j] >= EDIT_MIN for j in tail])
    n_lut = len({cache.presets[int(cache.preset[int(rows[j])])] for j in gate})
    n_src = len({int(cache.group[int(rows[j])]) for j in gate})
    stats = {"tail_frac": HIGH_TAIL_FRAC, "tail_n": int(len(tail)),
             "gate": {"c_tar_min": C_TAR_MIN, "c_res_min": C_RES_MIN,
                      "edit_min": EDIT_MIN},
             "n_after_gate": len(gate),
             "n_dropped": int(len(tail)) - len(gate),
             # 非互斥计数 + 互斥归因，两套都给
             "n_dropped_by": {
                 "c_tar_lt_min": int(sum(1 for j in tail
                                         if c_tar[j] < C_TAR_MIN)),
                 "c_res_lt_min": int(sum(1 for j in tail
                                         if c_res[j] < C_RES_MIN)),
                 "edit_lt_min": int(sum(1 for j in tail
                                        if edit[j] < EDIT_MIN)),
                 "color_only": int((~col_ok & ed_ok).sum()),
                 "edit_only": int((col_ok & ~ed_ok).sum()),
                 "both": int((~col_ok & ~ed_ok).sum())},
             # 上尾里目标是纯黑白 / 近单色的比例（就是本轮要修的那个偏置）
             "tail_bw_share": {
                 "c_res_lt_1": int(sum(1 for j in tail if c_res[j] < 1.0)),
                 "c_res_lt_5": int(sum(1 for j in tail if c_res[j] < 5.0)),
                 "pool_c_res_lt_1": int((c_res < 1.0).sum()),
                 "pool_n": int(len(rows))},
             # 去重后真正能取几张：LUT 与源组都要互不相同，短板是 LUT
             "n_distinct_lut_after_gate": n_lut,
             "n_distinct_src_after_gate": n_src,
             "dedup_capacity": min(n_lut, n_src),
             "content_excluded": ({str(k): v
                                   for k, v in CONTENT_EXCLUDE.items()}
                                  if content_exclude else {})}
    ordered = np.array(sorted(gate, key=lambda j: (-float(edit[j]),
                                                   int(rows[j]))))
    ex = set(CONTENT_EXCLUDE) if content_exclude else set()
    out = []
    for p in _dedup_take(cache, rows, ordered, n_pick, exclude=ex):
        j = p.pop("j")
        out.append({"profile": "color",
                    "rank_edit_in_gate": len(out),
                    "ident_pct": round(float((ident <= ident[j]).mean()) * 100, 2),
                    "edit_pct": round(float((edit <= edit[j]).mean()) * 100, 2),
                    "c_tar_pct": round(float((c_tar <= c_tar[j]).mean()) * 100, 2),
                    "c_res_pct": round(float((c_res <= c_res[j]).mean()) * 100, 2),
                    "ident_de00": float(ident[j]),
                    "edit_de00_128": float(edit[j]),
                    "c_tar": float(c_tar[j]), "c_res": float(c_res[j]), **p})
    assert len(out) == n_pick, (
        f"色彩性门过后可选样本不足：只凑出 {len(out)}/{n_pick}。"
        f"门内 {len(gate)} 行、去重容量 {min(n_lut, n_src)}（不同 LUT {n_lut} 个 /"
        f"不同源组 {n_src} 个），内容排除 {len(ex)} 行。"
        f"要么放宽 HIGH_TAIL_FRAC，要么下调 C_TAR_MIN/C_RES_MIN。")
    return out, stats


# -------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mem-frac", type=float, default=0.12)
    ap.add_argument("--short", type=int, default=512)
    ap.add_argument("--eval-n", type=int, default=1024)
    ap.add_argument("--cache", default=os.path.join(EXP, "cache"))
    ap.add_argument("--groups", default="A_paradigm,B_mlpwide,C_ladder",
                    help="要出图的组，逗号分隔")
    ap.add_argument("--c-mode", type=int, default=3, choices=(2, 3),
                    help="组 C 的档数：3=小/中/大，2=小/大")
    ap.add_argument("--round", type=int, default=3, choices=(2, 3),
                    help="选源轮次：3=色彩性门（默认），2=第二轮 mid+high")
    ap.add_argument("--no-content-exclude", action="store_true",
                    help="撤销 CONTENT_EXCLUDE（把规则第 1 名放回来）")
    ap.add_argument("--meta", default="meta.json")
    args = ap.parse_args()
    dev = args.device
    if dev.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.mem_frac,
                                                   int(dev.split(":")[1]))
    torch.set_grad_enabled(False)
    os.makedirs(OUT, exist_ok=True)

    groups = dict(GROUPS)
    if args.c_mode == 2:
        groups["C_ladder"] = C_TWO
    want = [g.strip() for g in args.groups.split(",") if g.strip()]
    for g in want:
        assert g in groups, f"unknown group {g}"

    cache = D.RenderCache(args.cache)
    plan = pickle.load(open(os.path.join(EXP, "cache/plan.pkl"), "rb"))
    rows_all, ident = load_ident(cache, args.eval_n)

    # ---- 选源 + 每个样本进哪几组 -------------------------------------------
    edit = edit_magnitude(cache, rows_all, dev)
    picks, sel_stats = [], {}
    if args.round == 3:
        c_tar, c_res = chroma_stats(cache, rows_all, dev)
        col, sel_stats = pick_color(
            cache, rows_all, ident, edit, c_tar, c_res,
            content_exclude=not args.no_content_exclude)
        # 组 A 保留主 agent 指定的那一张（profile "mid" 的 p50），另加 color 前两名
        keep = [p for p in pick_mid(cache, rows_all, ident)
                if p["pct"] == A_KEEP_PCT]
        for p in keep:
            p["groups"] = ["A_paradigm"]
            p["note"] = "主 agent 指定保留，未参与本轮重选"
        for k, p in enumerate(col):
            p["groups"] = (["A_paradigm"] if k < 2 else []) \
                + ["B_mlpwide", "C_ladder"]
        picks = keep + col
    else:                                   # 第二轮口径，保留可复现
        c_tar = c_res = None
        for p in pick_mid(cache, rows_all, ident):
            p["groups"] = ["A_paradigm"]
            picks.append(p)
        for p in pick_high(cache, rows_all, ident, edit):
            p["groups"] = ["B_mlpwide", "C_ladder"]
            picks.append(p)
    for p in picks:                         # 只出被点名的那几组
        p["groups"] = [g for g in p["groups"] if g in want]
    picks = [p for p in picks if p["groups"]]
    print("[picks]", json.dumps(picks, ensure_ascii=False), flush=True)
    print("[select_stats]", json.dumps(sel_stats, ensure_ascii=False),
          flush=True)

    models, counts = {}, {}
    for arm in ARMS:
        ck = torch.load(os.path.join(EXP, "runs", arm, "best.pt"),
                        map_location=dev, weights_only=False)
        a = ck["args"]
        assert a["arm"] == arm, f"{arm}: checkpoint arm mismatch {a['arm']}"
        anchors = torch.from_numpy(np.load(a["anchors"])) \
            if os.path.isfile(a["anchors"]) else None
        m = RDGModel(a["arm"], a["n_gauss"], anchors=anchors, free=a["free"])
        m.load_state_dict(ck["model"], strict=True)
        m = m.to(dev).eval()
        models[arm] = m
        counts[arm] = {"step": int(ck["step"]), **m.n_params()}
        print(f"[load] {arm} step={ck['step']} params={m.n_params()}", flush=True)

    PQ = (0, 25, 50, 75, 90, 95, 99, 100)
    pool = {"ident_pctl": {str(q): float(np.percentile(ident, q)) for q in PQ},
            "edit_pctl": {str(q): float(np.percentile(edit, q)) for q in PQ},
            "spearman_ident_edit": float(np.corrcoef(
                np.argsort(np.argsort(ident)),
                np.argsort(np.argsort(edit)))[0, 1])}
    if c_tar is not None:
        pool["c_tar_pctl"] = {str(q): float(np.percentile(c_tar, q))
                              for q in PQ}
        pool["c_res_pctl"] = {str(q): float(np.percentile(c_res, q))
                              for q in PQ}
        # 门槛本身在整池里的分位（判据可复算）
        pool["gate_pct_in_pool"] = {
            "c_tar_min": round(float((c_tar < C_TAR_MIN).mean()) * 100, 2),
            "c_res_min": round(float((c_res < C_RES_MIN).mean()) * 100, 2),
            "edit_min": round(float((edit < EDIT_MIN).mean()) * 100, 2)}
    meta = {"round": args.round, "picks": picks, "select_stats": sel_stats,
            "counts": counts, "short_side": args.short,
            "eval_n": args.eval_n, "c_mode": args.c_mode,
            "groups_emitted": want, "pool_stats": pool, "samples": []}

    for pk in picks:
        row = pk["row"]
        gid = int(cache.group[row])
        mine = [g for g in want if g in pk["groups"]]
        # 模型输入：严格用 128 缓存对（与训练/评测完全一致）
        img, src128, pid = D.collate_to_gpu(
            [D.Loader(cache, np.array([row])).__getitem__(0)], dev)
        # 显示用高清
        in_im = hires(plan["in"][gid], args.short)
        gt_im = hires(plan["after"][row], args.short)
        if gt_im.size != in_im.size:
            gt_im = gt_im.resize(in_im.size, Image.Resampling.BOX)
        in_arr = np.asarray(in_im, dtype=np.float32) / 255.0
        gt_arr = np.asarray(gt_im, dtype=np.float32) / 255.0
        gt_t = torch.from_numpy(gt_arr).to(dev)
        in_t = torch.from_numpy(in_arr).to(dev)

        tiles = {"__in__": in_im, "__gt__": gt_im}
        rec = {**pk, "size": list(in_im.size), "arms": {}}
        rendered = {}
        xs = D.eval_colors(1, 16384, 777, dev)
        tab = D.load_lut_bank([pk["preset"]], dev)
        yt = D.tri_lookup(tab, xs)
        rec["identity_cube_de00"] = float(delta_e00(xs, yt).mean())
        for arm in ARMS:
            p = models[arm].params_from_image(img)[0][0]
            y, u8 = apply_f(p, in_arr, dev)
            tiles[arm] = Image.fromarray(u8)
            rendered[arm] = y
            cube_de = float(delta_e00(render(p, xs).clamp(0, 1), yt).mean())
            img_de = delta_e00(y, gt_t).reshape(-1)
            rec["arms"][arm] = {
                "cube_de00": cube_de,
                "img_de00_p50": float(img_de.median()),
                "img_de00_mean": float(img_de.mean()),
                # 这一臂把图从输入挪走了多远（对比"编辑幅度"看是否只是复制输入）
                "moved_from_input_p50": float(
                    delta_e00(y, in_t).reshape(-1).median()),
            }
            del p
        del tab
        # 尺度参照：这次编辑本身在图像空间有多大
        rec["edit_size_img_de00_p50"] = float(
            delta_e00(in_t, gt_t).reshape(-1).median())
        # 臂间两两差（图像空间），用来给"肉眼看不看得出"一个数字背书
        pairs = [("mlp", "mlp_wide"), ("mlp", "gbase"), ("gtiny", "glite"),
                 ("glite", "gbase"), ("gtiny", "gbase")]
        rec["pair_img_de00_p50"] = {
            f"{a}|{b}": float(
                delta_e00(rendered[a], rendered[b]).reshape(-1).median())
            for a, b in pairs}
        rec["pair_img_de00_p90"] = {
            f"{a}|{b}": float(
                delta_e00(rendered[a], rendered[b]).reshape(-1)
                .quantile(0.9))
            for a, b in pairs}
        del rendered, in_t
        print(f"[row {row}] " + json.dumps(rec["arms"]), flush=True)

        for gname in mine:
            cols, titles = groups[gname]
            path = os.path.join(OUT, f"{gname}_{row}.png")
            pan = row_figure([tiles[c] for c in cols], titles, path,
                             short_side=args.short)
            print(f"[fig] {path} {pan.size}", flush=True)
        rec["figs"] = {g: f"{g}_{row}.png" for g in mine}
        meta["samples"].append(rec)

    with open(os.path.join(OUT, args.meta), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print("[done]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
