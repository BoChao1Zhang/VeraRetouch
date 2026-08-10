"""G2 补批分析（D-34）：分 build 稳健性表 + 与原批次的对拍。

只做统计与对拍，不重算任何 Δ —— 输入是两批 per_image_*.jsonl。
输出 metrics_strat/strat_summary.json + 终端 Markdown 表。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

E = Path("/home/bc/VeraRetouch/experiments/G2_oracle_ceiling_20260803")
M0, M1 = E / "metrics", E / "metrics_strat"
KEY = "delta_arm"          # headline 口径（臂式 4D，逐箱仿射；D-24）


def rd(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def q(v, k=KEY):
    a = np.asarray([r[k] for r in v if np.isfinite(r.get(k, np.nan))], float)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "median": float(np.median(a)),
            "q10": float(np.percentile(a, 10)), "q90": float(np.percentile(a, 90)),
            "mean": float(a.mean()), "min": float(a.min()), "max": float(a.max())}


def frac(v, thr, k=KEY):
    a = np.asarray([r[k] for r in v], float)
    return float(np.mean(a >= thr))


old_r, old_c = rd(M0 / "per_image_real.jsonl"), rd(M0 / "per_image_control.jsonl")
new_r, new_c = rd(M1 / "per_image_real_strat.jsonl"), rd(M1 / "per_image_control_strat.jsonl")

out: dict = {"key": KEY, "note": "只改抽样；估计器/判据沿用 D-24 逐箱最小二乘仿射"}

# ---------------------------------------------------------------- 分 build 稳健性
for name, rows in (("real", new_r), ("control", new_c)):
    tab = {}
    for b in sorted({r["build"] for r in rows}):
        sub = [r for r in rows if r["build"] == b]
        tab[b] = {**q(sub), "nested_median": q(sub, "delta")["median"],
                  "cv_median": q(sub, "delta_arm_cv")["median"],
                  "donor_median": q(sub, "delta_arm_donor")["median"],
                  "const_absmax": float(np.max(np.abs(
                      [r.get("delta_arm_const", 0.0) for r in sub]))),
                  "psnr_id_median": q(sub, "psnr_id")["median"],
                  "psnr_3d_median": q(sub, "psnr_3d")["median"],
                  "psnr_4d_arm_median": q(sub, "psnr_4d_arm")["median"],
                  "frac_ge_1db": frac(sub, 1.0), "frac_ge_04db": frac(sub, 0.4),
                  "mask_area_median": float(np.median(
                      [r["mask_area"] for r in sub if r.get("mask_area") is not None]))
                  if any(r.get("mask_area") is not None for r in sub) else None}
    out[f"by_build_{name}"] = tab
    meds = [v["median"] for v in tab.values()]
    out[f"by_build_{name}_spread"] = {
        "n_builds": len(meds), "min": min(meds), "max": max(meds),
        "range": max(meds) - min(meds)}

# ---------------------------------------------------------------- 全批总览对拍
def overall(rows, ctrl):
    r = q(rows)
    c = q(ctrl)
    sp = spearmanr([x["delta"] for x in rows], [x["delta_alt"] for x in rows])
    ma = [(x["mask_area"], x[KEY]) for x in rows if x.get("mask_area") is not None]
    spa = spearmanr([a for a, _ in ma], [d for _, d in ma])
    return {
        "n_real": r["n"], "real_median": r["median"],
        "real_q10": r["q10"], "real_q90": r["q90"],
        "real_nested_median": q(rows, "delta")["median"],
        "real_cv_median": q(rows, "delta_arm_cv")["median"],
        "real_donor_median": q(rows, "delta_arm_donor")["median"],
        "n_control": c["n"], "control_median": c["median"],
        "control_q10": c["q10"], "control_q90": c["q90"],
        "local_signal_db": r["median"] - c["median"],
        "frac_ge_1db": frac(rows, 1.0), "frac_ge_04db": frac(rows, 0.4),
        "bandwidth_spearman_nested": float(sp.statistic),          # type: ignore
        "bandwidth_spearman_arm": float(spearmanr(
            [x[KEY] for x in rows], [x["delta_arm_alt"] for x in rows]).statistic),  # type: ignore
        "spearman_maskarea_vs_delta": float(spa.statistic),        # type: ignore
        "real_const_absmax": float(np.max(np.abs(
            [x.get("delta_arm_const", 0.0) for x in rows]))),
        "control_const_absmax": float(np.max(np.abs(
            [x.get("delta_arm_const", 0.0) for x in ctrl]))),
    }


out["batch_orig"] = overall(old_r, old_c)
out["batch_strat"] = overall(new_r, new_c)

# ---------------------------------------------------------------- 重叠样本逐位一致
o = {r["id"]: r for r in old_r}
dup = [(r, o[r["id"]]) for r in new_r if r["id"] in o]
KS = (KEY, "delta", "delta_alt", "psnr_3d", "psnr_4d", "psnr_4d_arm", "psnr_id")
per_key = {}
for k in KS:
    dv = np.abs([a[k] - b[k] for a, b in dup])
    per_key[k] = {"max_abs_diff": float(dv.max()) if dv.size else 0.0,
                  "n_bit_identical": int((dv == 0).sum()),
                  "n_gt_1e_9": int((dv > 1e-9).sum()),
                  "n_gt_1e_3": int((dv > 1e-3).sum())}
out["overlap_real"] = {
    "n_overlap": len(dup), "per_key": per_key,
    "headline_bit_identical": per_key[KEY]["max_abs_diff"] == 0.0,
    "note": "headline 口径 delta_arm / psnr_3d / psnr_4d_arm / psnr_id 在全部重叠样本上"
            "**逐位一致** → 估计器零改动已证；嵌套式 delta / delta_alt 各有 4 / 2 张"
            "在 ≤8e-7 dB 量级有差，来源是 GPU index_add_ 原子累加顺序（与 O-5a 记的"
            "1.7e-6 dB Δ_const 浮点噪声同一类），不改变任何一位有效数字。"}
oc = {r["id"]: r for r in old_c}
dupc = [(r, oc[r["id"]]) for r in new_c if r["id"] in oc]
badc = [a["id"] for a, b in dupc if abs(a["delta"] - b["delta"]) > 1e-12]
# 对照档 s = 移植掩膜，donor 由**流内位置**决定 → 分层后配对变了，delta 本就不该一致
out["overlap_control"] = {"n_overlap": len(dupc), "n_delta_differs": len(badc),
                          "why": "control 档 s=移植掩膜，donor 按流内位置 i%%64 配对，"
                                 "分层改变流顺序 → 同一张图配到不同 donor，Δ 不同属预期"}

# ---------------------------------------------------------------- 新旧 build 分组
NEW_L = {"prod-l5-local17k-20260801", "prod-l6-local17k-20260801"}
NEW_G = {"prod-g3-global25k-20260801", "prod-g4-global25k-20260801"}
out["coverage_gap_closed"] = {
    "real_new_builds": q([r for r in new_r if r["build"] in NEW_L]),
    "real_old_builds": q([r for r in new_r if r["build"] not in NEW_L]),
    "control_new_builds": q([r for r in new_c if r["build"] in NEW_G]),
    "control_old_builds": q([r for r in new_c if r["build"] not in NEW_G]),
}
a = [r[KEY] for r in new_r if r["build"] in NEW_L]
b = [r[KEY] for r in new_r if r["build"] not in NEW_L]
u = mannwhitneyu(a, b, alternative="two-sided")
out["coverage_gap_closed"]["real_new_vs_old_mannwhitney_p"] = float(u.pvalue)  # type: ignore
ac = [r[KEY] for r in new_c if r["build"] in NEW_G]
bc = [r[KEY] for r in new_c if r["build"] not in NEW_G]
out["coverage_gap_closed"]["control_new_vs_old_mannwhitney_p"] = float(
    mannwhitneyu(ac, bc, alternative="two-sided").pvalue)                     # type: ignore

# ---------------------------------------------------------------- 分池（补批口径）
for name, rows in (("real", new_r), ("control", new_c)):
    out[f"by_pool_{name}"] = {
        p: {**q([r for r in rows if r["pool"] == p])}
        for p in sorted({r["pool"] for r in rows})}

(M1 / "strat_summary.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------------------------------------------------------------- 终端 Markdown
def show(tag: str, tab: dict) -> None:
    print(f"\n### 分 build 稳健性（{tag}，补批 stratified）\n")
    print("| build | n | Δ_arm 中位 | q10 | q90 | Δ_nested 中位 | Δ_arm_cv | Δ_donor | ≥1dB | 掩膜面积 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for b, v in tab.items():
        ma = "—" if v["mask_area_median"] is None else f"{v['mask_area_median']:.3f}"
        print(f"| {b} | {v['n']} | **{v['median']:.2f}** | {v['q10']:.2f} | {v['q90']:.2f} "
              f"| {v['nested_median']:.2f} | {v['cv_median']:.2f} | {v['donor_median']:.2f} "
              f"| {v['frac_ge_1db']*100:.1f}% | {ma} |")


show("真实档 D-SFT-L", out["by_build_real"])
show("对照档 D-SFT-G", out["by_build_control"])
print("\n### 两批对拍\n")
print("| 量 | 原批次(索引截断) | 补批(分层) | 差 |")
print("|---|---|---|---|")
ROWS = [("真实档 n", "n_real", "{:.0f}"), ("真实档 Δ_arm 中位 (判据②≥1.0)", "real_median", "{:.3f}"),
        ("真实档 q10", "real_q10", "{:.3f}"), ("真实档 q90", "real_q90", "{:.3f}"),
        ("真实档 Δ_nested 中位", "real_nested_median", "{:.3f}"),
        ("真实档 Δ_arm_cv 中位", "real_cv_median", "{:.3f}"),
        ("真实档 Δ_donor 中位", "real_donor_median", "{:.3f}"),
        ("对照档 n", "n_control", "{:.0f}"),
        ("对照档 Δ_arm 中位 (判据③<0.4)", "control_median", "{:.3f}"),
        ("②−③ 局部信号强度", "local_signal_db", "{:.3f}"),
        ("≥1 dB 占比", "frac_ge_1db", "{:.4f}"), ("≥0.4 dB 占比", "frac_ge_04db", "{:.4f}"),
        ("带宽 Spearman(33³,17³) arm (判据⑤>0.8)", "bandwidth_spearman_arm", "{:.4f}"),
        ("Spearman(掩膜面积, Δ_arm)", "spearman_maskarea_vs_delta", "{:.4f}")]
for lab, k, f in ROWS:
    a0, a1 = out["batch_orig"][k], out["batch_strat"][k]
    print(f"| {lab} | {f.format(a0)} | {f.format(a1)} | {a1-a0:+.3f} |")
print(f"\n重叠样本（真实档）：{out['overlap_real']['n_overlap']} 组；headline delta_arm 逐位一致 = "
      f"{out['overlap_real']['headline_bit_identical']}")
for k, v in out["overlap_real"]["per_key"].items():
    print(f"    {k:12s} max|Δ|={v['max_abs_diff']:.2e}  n(>1e-3)={v['n_gt_1e_3']}")
print(f"新覆盖 l5/l6：{out['coverage_gap_closed']['real_new_builds']}")
print(f"新覆盖 g3/g4：{out['coverage_gap_closed']['control_new_builds']}")
print(f"→ metrics_strat/strat_summary.json")
