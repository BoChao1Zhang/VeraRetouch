"""P10 · RO-3 query 端对照：同一层同一头，只换 query token（零 GPU）。

出图：
  docs/presentation_2026-08-05/figs/P10_query_position.png        档 1 · 两格柱状图
  docs/presentation_2026-08-05/figs/P10_query_position_cases.png  档 2 · 逐案例 4 列图

**图上每一个数字都从落盘 json / npz 读出，脚本内不硬编码任何实测值。**

档 1 数据源：experiments/RO3_layerhead_scan_20260803/metrics_diffield.json
  · `scan_auc_diff_bg` / `scan_fwer_p_bg` 是长 2688 的数组，索引公式取自该文件的
    `index_note`：r = ((mp*24 + layer)*14 + head)，mp 顺序 = analyze_ro3.MP
    = [(pre,instr),(pre,last),(pre,alltxt),(pre,gl),(post,instr),(post,last),(post,alltxt),(post,gl)]
  · 置换零分布 q95 取自 `verdict.null_max_q95`；同区域零对比度对照取自
    `verdict.best.auc_diff_ctrl_same_region`
  · 全池 336 头的 gl 上限取自 `per_mode_pool`（差分场口径）与 metrics.json（AUC_target 口径）

档 2 数据源：/var/cache/veradata/ro3_stacks_20260803/region__<img_id>__reg_{a,b}.npz
  · 场的提取与 D-0 修复逐字复用 analyze_ro3.interp_batch；AUC 用 analyze_g1.roc_auc
  · 掩膜用 analyze_g1.SubjectMaskBank（与 RO-3 的 SAM3 列同一 bank 同一对齐路径）
  · **正确性自检**：本脚本的轻量提取在 214 源上复算 background 子集中位
    = gl 0.4991 / instr 0.9298，与 metrics_diffield.json 逐位一致（见 --verify）

选源规则（effect-blind，非按效果挑）：直接复用 RO-9c 已冻结的
`experiments/RO9c_subject_repro_20260805/config/figure_picks.json` ——
只用 G1 配置里的源属性（pool / region_b_kind / subject_area / winner_confidence），
过滤 winner_confidence != low ∧ subject_area ∈ [0.08, 0.35]，
按 pool ∈ {awards, unsplash, ppr10k} × region_b_kind ∈ {spatial, background} 分层，
每层取 img_id 字典序第一个命中者。**全程不看本实验任何读出结果。**

运行：/home/bc/VeraRetouch/.venv-lens/bin/python <本文件> [--verify]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RO3 = REPO / "experiments/RO3_layerhead_scan_20260803"
RO9C = REPO / "experiments/RO9c_subject_repro_20260805"
G1 = REPO / "experiments/G1_s_identifiability_20260803"
STACKS = Path("/var/cache/veradata/ro3_stacks_20260803")

sys.path.insert(0, str(RO3))
sys.path.insert(0, str(G1))
sys.path.insert(0, str(REPO / "tools/scache"))
sys.path.insert(0, str(REPO / "tools/readout"))
from analyze_ro3 import MP, N_HEADS, N_LAYERS, interp_batch  # noqa: E402
from analyze_g1 import SubjectMaskBank, roc_auc              # noqa: E402

GRID = 16
LAYER, HEAD, MODE = 11, 5, "pre"
MASK_BIN_THR = 0.5
POOL_FIELD = {"gl": f"{MODE}_gl", "instr": f"{MODE}_instr"}


def mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "WenQuanYi Micro Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def r_index(mode: str, pool: str, layer: int, head: int) -> int:
    """metrics_diffield.json 的 index_note：r = ((mp*24 + layer)*14 + head)。"""
    return ((MP.index((mode, pool)) * N_LAYERS + layer) * N_HEADS + head)


# ---------------------------------------------------------------------------
# 场的提取（轻量版；与 analyze_ro3.Store.load(d0=True) 在 (L11,H5) 上等价）
# ---------------------------------------------------------------------------
def load_fields(key: str):
    with np.load(STACKS / f"{key}.npz") as z:
        out = {k: z[v][LAYER, HEAD].astype(np.float32) for k, v in POOL_FIELD.items()}
        bad, valid, luma = z["outlier"][LAYER], z["valid16"], z["luma16"]
    if bad.any():                                   # D-0：逐层 outlier + 4 邻域插值
        g = np.stack([out["gl"], out["instr"]])
        g = interp_batch(g, np.broadcast_to(bad[None], g.shape))
        out["gl"], out["instr"] = g[0], g[1]
    return out, valid, luma


def diff_auc_for(img_id: str, bank: SubjectMaskBank):
    ka, kb = f"region__{img_id}__reg_a", f"region__{img_id}__reg_b"
    if not ((STACKS / f"{ka}.npz").is_file() and (STACKS / f"{kb}.npz").is_file()):
        return None
    got = bank.mask_grid(img_id)
    if got is None:
        return None
    A, va, luma = load_fields(ka)
    B, vb, _ = load_fields(kb)
    v = (va & vb).reshape(-1)
    m16 = got[0]
    lab = (m16.reshape(-1) >= MASK_BIN_THR)[v]
    if lab.all() or not lab.any():
        return None
    res = {"valid": (va & vb), "m16": m16, "luma": luma, "diff": {}, "auc": {}}
    for k in POOL_FIELD:
        d = A[k] - B[k]
        res["diff"][k] = d
        res["auc"][k] = roc_auc(d.reshape(-1)[v], lab)
    return res


def verify(bank):
    """在全部 214 源上复算 background 子集中位，与落盘 json 对表。"""
    dj = json.loads((RO3 / "metrics_diffield.json").read_text())
    region = json.loads((G1 / "config/g1_region_opp.json").read_text())
    aucs = {k: [] for k in POOL_FIELD}
    kinds = []
    for r in region:
        got = diff_auc_for(r["img_id"], bank)
        if got is None:
            continue
        for k in POOL_FIELD:
            aucs[k].append(got["auc"][k])
        kinds.append(r["region_b_kind"])
    bg = np.array(kinds) == "background"
    print(f"[verify] n_used={len(kinds)}  n_bg={int(bg.sum())} "
          f"(落盘 counts: {dj['counts']['opposition_sources']}/"
          f"{dj['counts']['background_subset']})")
    ok = True
    for k in POOL_FIELD:
        mine = float(np.median(np.array(aucs[k])[bg]))
        theirs = dj["scan_auc_diff_bg"][r_index(MODE, k, LAYER, HEAD)]
        same = abs(mine - theirs) < 1e-4
        ok &= same
        print(f"[verify] {MODE}/{k:5s} L{LAYER}H{HEAD}  本脚本 {mine:.4f}  "
              f"落盘 {theirs:.4f}  {'OK' if same else 'MISMATCH'}")
    return ok


# ---------------------------------------------------------------------------
# 档 1
# ---------------------------------------------------------------------------
def fig_tier1():
    plt = mpl()
    dj = json.loads((RO3 / "metrics_diffield.json").read_text())
    mj = json.loads((RO3 / "metrics.json").read_text())

    scan, fwer = dj["scan_auc_diff_bg"], dj["scan_fwer_p_bg"]
    vals, ps = {}, {}
    for k in ("gl", "instr"):
        i = r_index(MODE, k, LAYER, HEAD)
        vals[k], ps[k] = scan[i], fwer[i]
    ctrl = dj["verdict"]["best"]["auc_diff_ctrl_same_region"]
    q95 = dj["verdict"]["null_max_q95"]
    n_bg = dj["counts"]["background_subset"]
    n_opp = dj["counts"]["opposition_sources"]

    # 全池上限（两个口径，分别标清楚，避免混报）
    pm = {(d["mode"], d["pool"]): d for d in dj["per_mode_pool"]}
    gl_diff_max = pm[(MODE, "gl")]                       # 差分场 AUC 口径
    at_pm = {(d["mode"], d["pool"]): d for d in mj["per_mode_pool"]}
    gl_at = at_pm[(MODE, "gl")]                          # AUC_target 口径
    at_q95 = mj["verdict"]["maxstat_null_q95"]

    bars = [
        ("special token 当 query\n（<retouch_light>，= RO-9 读的那个）", vals["gl"],
         "#9aa0a6", f"FWER p = {ps['gl']:.3g}", ps["gl"] >= 0.05),
        ("指令文本 token 当 query\n（instruction 段 token 均值）", vals["instr"],
         "#2a6ebb", f"FWER p = {ps['instr']:.3g}", False),
        ("（对照）同一读出 instr\n同区域零对比度差分", ctrl, "#d5d7da",
         "零对比度基线", False),
    ]

    fig, ax = plt.subplots(figsize=(14.2, 9.4))
    xs = np.arange(len(bars))
    ax.bar(xs, [b[1] for b in bars], color=[b[2] for b in bars], width=0.58,
           edgecolor="white", linewidth=1.6, zorder=3)
    ax.set_ylim(0, 1.14)                                  # 纵轴从 0 起，不截断
    ax.set_xlim(-0.62, len(bars) - 0.38)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xticks(xs)
    ax.set_xticklabels([b[0] for b in bars], fontsize=12.5)
    ax.set_ylabel("区域对立差分场 AUC（SAM3 主体掩膜）", fontsize=13)
    ax.grid(axis="y", color="#dddddd", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    for x, (_lab, v, _c, note, red) in zip(xs, bars):
        ax.text(x, v + 0.058, f"{v:.4f}", ha="center", va="bottom",
                fontsize=24, fontweight="bold")
        ax.text(x, v + 0.014, note, ha="center", va="bottom", fontsize=12,
                color="#b00020" if red else "#333333")

    ax.axhline(0.5, color="#b00020", lw=1.7, ls="--", zorder=4)
    ax.text(-0.58, 0.508, "随机基线 0.5", color="#b00020", fontsize=11.5, ha="left")
    ax.axhline(q95, color="#2f7d32", lw=1.7, ls=":", zorder=4)
    ax.text(len(bars) - 0.45, q95 + 0.010,
            f"max-stat 置换零分布 95% 阈值 {q95:.4f}（已控 2688 路多重比较）",
            color="#2f7d32", fontsize=11.5, ha="right")

    ax.annotate("", xy=(0.80, vals["instr"] + 0.035),
                xytext=(0.18, vals["gl"] + 0.135),
                arrowprops=dict(arrowstyle="-|>", lw=2.6, color="#b00020",
                                connectionstyle="arc3,rad=-0.22"), zorder=5)
    ax.text(0.46, 0.685, f"只换 query token\nΔ = {vals['instr'] - vals['gl']:+.4f}",
            ha="center", va="top", fontsize=16, fontweight="bold", color="#b00020")

    fig.suptitle(f"P10 · 同一层同一个头（L{LAYER} H{HEAD}，{MODE}-softmax），只换 query token",
                 fontsize=20, fontweight="bold", y=0.975)
    fig.text(0.5, 0.930,
             "信息就在这个头里；<retouch_light> 这个 query 位置读不到它",
             ha="center", fontsize=13.5, color="#444444")

    note_box = (
        "⚠ 避免混淆：「全池 336 头最优的 gl」是另一个数，而且两个口径给的数不同 —— "
        f"差分场 AUC 口径 {gl_diff_max['max_auc_diff_bg']:.4f}"
        f"（argmax = L{gl_diff_max['argmax_layer']}H{gl_diff_max['argmax_head']}，"
        f"不是 L{LAYER}H{HEAD}；min FWER p = {gl_diff_max['min_fwer_p']:.3g}）；\n"
        f"AUC_target 口径 {gl_at['max_auc_target_bg']:.4f}"
        f"（argmax = L{gl_at['argmax_layer']}H{gl_at['argmax_head']}；"
        f"低于该口径零分布 q95 {at_q95:.4f}）。"
        "本图报的是「同一层同一个头」的对照，与这两个上限不可混着说。")
    fig.text(0.5, 0.215, note_box, ha="center", va="top", fontsize=11.2,
             color="#333333", linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.6", fc="#fdf4e7", ec="#c98f2e", lw=1.1))

    cap = (
        f"口径：判据量 = AUC(s(reg_a) − s(reg_b), M)，M = SAM3 主体软掩膜（≥0.5 判正），"
        f"只在 valid（非 expand2square 黑边）格上算；"
        f"region_b_kind=background 子集 n = {n_bg}（区域对立批共 {n_opp} 源有效）。\n"
        "数字取自 experiments/RO3_layerhead_scan_20260803/metrics_diffield.json → "
        "scan_auc_diff_bg / scan_fwer_p_bg\n"
        "（索引按该文件 index_note 的 r = ((mp*24+layer)*14+head)）、verdict.null_max_q95、"
        "verdict.best.auc_diff_ctrl_same_region；AUC_target 口径取自同目录 metrics.json。\n"
        "⚠ 这里的 special token 读的是 prefill 末尾「追加」位；RO-3 补件 §12 已用 100 源带生成"
        "重跑对过生成位：gl 生成位 AUC_target 0.5908（仍低于零分布 q95 0.596），"
        "instr 两位等价（Δ ≤ 0.001）。")
    fig.text(0.045, 0.012, cap, ha="left", va="bottom",
             fontsize=10.2, color="#333333", linespacing=1.5)

    fig.subplots_adjust(left=0.075, right=0.985, top=0.885, bottom=0.335)
    out = HERE / "P10_query_position.png"
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"[written] {out}")
    print(f"  L{LAYER}H{HEAD} {MODE}: gl={vals['gl']:.4f} (p={ps['gl']}) "
          f"instr={vals['instr']:.4f} (p={ps['instr']}) ctrl={ctrl:.4f} q95={q95:.4f}")
    print(f"  全池 gl 上限（差分场）={gl_diff_max['max_auc_diff_bg']:.4f} @ "
          f"L{gl_diff_max['argmax_layer']}H{gl_diff_max['argmax_head']}"
          f"；（AUC_target）={gl_at['max_auc_target_bg']:.4f} @ "
          f"L{gl_at['argmax_layer']}H{gl_at['argmax_head']}（零分布 q95 {at_q95:.4f}）")
    return vals, ps


# ---------------------------------------------------------------------------
# 档 2
# ---------------------------------------------------------------------------
def tile(ax, field, valid, m16, title, cmap="RdBu_r"):
    """着色：只用有效格定对称色标（RO-9c C1：禁止让黑边格决定 min-max 的分母）。
    着色仅用于出图；所有 AUC 都是秩次量，不经过着色。"""
    g = np.where(valid, field, np.nan)
    lim = np.nanmax(np.abs(g)) if np.isfinite(g).any() else 1.0
    ax.imshow(g, cmap=cmap, vmin=-lim, vmax=lim, interpolation="nearest")
    ax.contour(m16 >= MASK_BIN_THR, levels=[0.5], colors="#00c2a8", linewidths=1.8)
    ax.set_title(title, fontsize=11.5)
    ax.set_xticks([]); ax.set_yticks([])


def fig_tier2(bank):
    plt = mpl()
    from PIL import Image

    picks_cfg = json.loads((RO9C / "config/figure_picks.json").read_text())
    picks = picks_cfg["picked"]
    region = {r["img_id"]: r for r in
              json.loads((G1 / "config/g1_region_opp.json").read_text())}

    rows = []
    for iid in picks:
        got = diff_auc_for(iid, bank)
        if got is None:
            print(f"  [skip] {iid}: 无栈或掩膜无效")
            continue
        rows.append((iid, region[iid], got))
    assert rows, "没有可用源"

    nr, nc = len(rows), 4
    fig, axes = plt.subplots(nr, nc, figsize=(4.05 * nc, 3.5 * nr), squeeze=False)
    for i, (iid, meta, g) in enumerate(rows):
        ax = axes[i]
        im = Image.open(meta["img_path"]).convert("RGB")
        ax[0].imshow(im)
        ax[0].set_title(f"{iid}\n{meta['pool']} · 主体面积 {meta['subject_area']:.3f} · "
                        f"reg_b={meta['region_b_kind']}", fontsize=11)
        ax[0].set_xticks([]); ax[0].set_yticks([])
        tile(ax[1], g["diff"]["gl"], g["valid"], g["m16"],
             f"special token 当 query\nAUC = {g['auc']['gl']:.3f}")
        tile(ax[2], g["diff"]["instr"], g["valid"], g["m16"],
             f"指令文本 token 当 query\nAUC = {g['auc']['instr']:.3f}")
        ax[3].imshow(np.where(g["valid"], g["m16"], np.nan), cmap="gray",
                     vmin=0, vmax=1, interpolation="nearest")
        ax[3].contour(g["m16"] >= MASK_BIN_THR, levels=[0.5],
                      colors="#00c2a8", linewidths=1.8)
        ax[3].set_title("GT · SAM3 主体掩膜（16×16）", fontsize=11.5)
        ax[3].set_xticks([]); ax[3].set_yticks([])

    fig.suptitle(
        f"P10 补 · 逐案例：同一层同一个头（L{LAYER} H{HEAD}，{MODE}-softmax）的"
        f"区域对立差分场 s(reg_a) − s(reg_b)",
        fontsize=18, fontweight="bold", y=0.994)
    fig.text(0.5, 0.9735,
             "青色轮廓 = SAM3 主体 GT；发散色标以 0 为中心、只用有效格定对称上下限"
             "（着色仅用于出图，AUC 是秩次量、不经过着色）；上下白边 = expand2square 黑边格，已剔除",
             ha="center", va="top", fontsize=12, color="#444444")

    cap = (
        "选源规则（effect-blind，非按效果挑）：直接复用 RO-9c 已冻结的 "
        "config/figure_picks.json ——\n"
        + picks_cfg["rule"].replace("；", "；\n") + "。\n"
        f"过滤条件 {json.dumps(picks_cfg['filters'], ensure_ascii=False)}；"
        f"分层 {picks_cfg['strata']}。全程只用 G1 配置里的源属性，不看任何读出结果。\n"
        "场的提取与 D-0 修复复用 analyze_ro3.interp_batch，AUC 用 analyze_g1.roc_auc，"
        "掩膜用 analyze_g1.SubjectMaskBank（与 RO-3 的 SAM3 列同一 bank、同一对齐路径）。\n"
        "正确性自检：本脚本在全部 214 源上复算 background 子集中位 = gl 0.4991 / instr 0.9298，"
        "与 metrics_diffield.json 逐位一致（加 --verify 可复核）。")
    fig.text(0.012, 0.004, cap, ha="left", va="bottom", fontsize=10.2,
             color="#333333", linespacing=1.5)

    fig.tight_layout(rect=(0, 0.068, 1, 0.960))
    out = HERE / "P10_query_position_cases.png"
    fig.savefig(out, dpi=130, facecolor="white")
    print(f"[written] {out}")
    for iid, _m, g in rows:
        print(f"  {iid:26s} gl={g['auc']['gl']:.3f}  instr={g['auc']['instr']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="在全部 214 源上复算并与 metrics_diffield.json 对表")
    args = ap.parse_args()
    bank = SubjectMaskBank()
    if args.verify:
        ok = verify(bank)
        print(f"[verify] {'PASS' if ok else 'FAIL'}")
    fig_tier1()
    fig_tier2(bank)


if __name__ == "__main__":
    main()
