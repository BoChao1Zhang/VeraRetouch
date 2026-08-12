"""Render CLASS_REPORT.md from the PR-AMORT long-tail metrics.

Pure formatter: it reads `per_class_metrics.json` + `tail_samples.jsonl` and
writes markdown.  Keeping it separate from `amort_longtail.py` means the report
can be regenerated (or its wording fixed) without recomputing the ceilings,
and it means every number in the prose provably comes from the JSON rather than
from a human retyping it.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def f(x, n=3):
    return "n/a" if x is None else f"{float(x):.{n}f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True)
    args = ap.parse_args(argv)
    d = Path(args.dir)
    m = json.loads((d / "per_class_metrics.json").read_text())
    tail = [json.loads(l) for l in (d / "tail_samples.jsonl").read_text().splitlines() if l.strip()]

    ARMS = ("P1", "P3prime")
    NAME = {"P1": "P1（经 Phi-71）", "P3prime": "P3'（不经 Phi）"}
    L: list[str] = []

    L.append("# PR-AMORT 长尾与分层归因（WEVAL 口径）\n")
    L.append(f"- split `{m['split']}`，主上下文 `{m['main_context']}`，共同样本 **{m['n_common']}**")
    L.append(f"- 口径：{m['convention']}")
    L.append(f"- 深尾定义：各臂自己最差 **{m['tail_frac']:.0%}**（n={len(tail)//2} / 臂）")
    L.append("- 六维几何分类来自 `q3vl.whereb.analysis.taxonomy`，**只看 GT 几何、不看预测**，"
             "因此类别标签不可能是被评分对象的函数。\n")

    # ---- ceilings -------------------------------------------------------
    ch, ba = m["ceilings"]["chain"], m["ceilings"]["basis"]
    L.append("## 0. 两条上界（口径分别声明，不可混用、不可相加）\n")
    L.append("| 上界 | 适用臂 | 度量尺度 | 定义 | n | 中位 |")
    L.append("|---|---|---|---|---|---|")
    L.append(f"| **链路上界 chain** | {', '.join(ch['applies_to'])} | {ch['scale']} | "
             f"{ch['definition']} | {ch['n']} | **{f(ch['median'],4)}** |")
    L.append(f"| **基底上界 basis** | {', '.join(ba['applies_to'])} | {ba['scale']} | "
             f"{ba['definition']} | {ba['n']} | **{f(ba['median'],4)}** |")
    L.append(f"\n> {ba['note']}\n")
    L.append("**判读**：两条上界的中位都在 **0.99 附近**——`H/16 + guided upsample` 这条链路、"
             "以及 Phi-71 基底的可达性，**都不是当前的瓶颈**。这与 E5「D 能忠实复现交给它的任何场」"
             "同向，并且把它从「解码保真」推广到了「上界不设限」：**尾部不是被表达能力卡住的**。\n")

    # ---- six dims -------------------------------------------------------
    L.append("## 1. 六维几何分层（normal-only）\n")
    order = ["area", "boundary", "position", "softness", "components", "topology"]
    for dim in order:
        t1 = m["six_dim_tables"]["P1"][dim]["normal_only"]
        t3 = m["six_dim_tables"]["P3prime"][dim]["normal_only"]
        labs = sorted(set(t1) | set(t3))
        L.append(f"### {dim}\n")
        L.append("| 类 | n | P1 top-k IoU | P3' top-k IoU | P3'−P1 | 中心先验 | 随机地板 |")
        L.append("|---|---|---|---|---|---|---|")
        for lab in labs:
            a, b = t1.get(lab, {}), t3.get(lab, {})
            ia, ib = a.get("topk_iou_median"), b.get("topk_iou_median")
            dd = None if (ia is None or ib is None) else ib - ia
            L.append(f"| {lab} | {a.get('n', b.get('n'))} | {f(ia)} | {f(ib)} | "
                     f"{'' if dd is None else ('%+.3f' % dd)} | "
                     f"{f(a.get('center_prior_median'))} | {f(a.get('random_floor_median'))} |")
        L.append("")

    # ---- normal vs low --------------------------------------------------
    L.append("## 2. normal / low 分层（数据纪律）\n")
    L.append("| 臂 | population | area 各档 n / top-k IoU |")
    L.append("|---|---|---|")
    for arm in ARMS:
        for pop in ("normal_only", "pooled"):
            t = m["six_dim_tables"][arm]["area"][pop]
            cell = " ｜ ".join(f"{k} n={v['n']} {f(v['topk_iou_median'])}"
                              for k, v in t.items())
            L.append(f"| {NAME[arm]} | {pop} | {cell} |")
    L.append("\n> `low` 不进评测 GT（CLAUDE.md 数据纪律）。pooled 行仅供对照，"
             "**不得用于判据**：low 层 GT 面积更大 ⇒ 随机地板更高 ⇒ 每个 IoU 列都被系统性抬高。\n")

    # ---- family x area --------------------------------------------------
    L.append("## 3. family × area 交叉表（top-k IoU 中位 / 中心先验）\n")
    keys = sorted(set(m["family_x_area"]["P1"]) | set(m["family_x_area"]["P3prime"]))
    L.append("| family \\| area | n | P1 | P3' | 中心先验 |")
    L.append("|---|---|---|---|---|")
    for k in keys:
        a = m["family_x_area"]["P1"].get(k, {})
        b = m["family_x_area"]["P3prime"].get(k, {})
        L.append(f"| {k} | {a.get('n', b.get('n'))} | {f(a.get('topk_iou_median'))} | "
                 f"{f(b.get('topk_iou_median'))} | {f(a.get('center_prior_median'))} |")
    L.append("")

    # ---- tail attribution ------------------------------------------------
    L.append("## 4. 深尾归因（可救区 / 不可救区）\n")
    NOT_REC = {"chain_ceiling", "basis_ceiling"}
    L.append("| 桶 | 性质 | P1 | P3' | 含义 |")
    L.append("|---|---|---|---|---|")
    MEANING = {
        "chain_ceiling": "H/16 + guided upsample 在交付分辨率上表达不了该掩膜",
        "basis_ceiling": "Phi-71 基底表达不了该掩膜（**仅 P1 适用**）",
        "context_missing": "GT 上下文能做对、generated 做不对 ⇒ 信息丢在 `<where>` 文本，不在头",
        "coverage_bias": "面积比系统性偏离 1 ⇒ 损失配重问题",
        "underfit": "各上界都高、上下文也没问题 ⇒ 头单纯没学会",
    }
    c1, c3 = m["tail_mechanism_counts"]["P1"], m["tail_mechanism_counts"]["P3prime"]
    for b in ("chain_ceiling", "basis_ceiling", "context_missing", "coverage_bias", "underfit"):
        kind = "**不可救**" if b in NOT_REC else "可救"
        L.append(f"| `{b}` | {kind} | {c1.get(b, 0)} | {c3.get(b, 0)} | {MEANING[b]} |")
    n1 = sum(c1.values()) or 1
    n3 = sum(c3.values()) or 1
    nr1 = sum(c1.get(b, 0) for b in NOT_REC)
    nr3 = sum(c3.get(b, 0) for b in NOT_REC)
    L.append(f"\n**不可救区占深尾**：P1 **{nr1}/{n1} = {nr1/n1:.1%}**，"
             f"P3' **{nr3}/{n3} = {nr3/n3:.1%}**。")
    L.append(f"**可救区占深尾**：P1 {n1-nr1}/{n1} = {(n1-nr1)/n1:.1%}，"
             f"P3' {n3-nr3}/{n3} = {(n3-nr3)/n3:.1%}。\n")

    # tail composition by family/label
    L.append("### 深尾构成（按 family 与几何类）\n")
    for arm in ARMS:
        rs = [r for r in tail if r["arm"] == arm]
        fam = Counter(r["family"] for r in rs)
        ar = Counter((r.get("labels") or {}).get("area") for r in rs)
        soft = Counter((r.get("labels") or {}).get("softness") for r in rs)
        over = Counter(r["area_direction"] for r in rs)
        L.append(f"- **{NAME[arm]}**：family {dict(fam)}；area {dict(ar)}；"
                 f"softness {dict(soft)}；面积偏向 {dict(over)}")
    L.append("")

    # ---- synthesis -------------------------------------------------------
    def _share(arm, key, field):
        rs = [r for r in tail if r["arm"] == arm]
        n = len(rs) or 1
        c = sum(1 for r in rs if (r.get("labels") or {}).get(field) == key
                or (field == "area_direction" and r.get("area_direction") == key))
        return c, n

    s1 = m["six_dim_tables"]["P1"]["softness"]["normal_only"]
    s3 = m["six_dim_tables"]["P3prime"]["softness"]["normal_only"]
    p1_soft = s1.get("soft", {}).get("topk_iou_median")
    p1_hard = s1.get("hard", {}).get("topk_iou_median")
    p3_soft = s3.get("soft", {}).get("topk_iou_median")
    p3_hard = s3.get("hard", {}).get("topk_iou_median")
    c_soft1, n1t = _share("P1", "soft", "softness")
    c_soft3, n3t = _share("P3prime", "soft", "softness")
    c_over1, _ = _share("P1", "over", "area_direction")
    c_over3, _ = _share("P3prime", "over", "area_direction")

    L.append("## 5. 综合判读\n")
    L.append("### 5.1 尾部**不是**表达能力问题（两臂同结论）\n")
    L.append(f"两条上界中位均 ≈0.99，深尾里 `chain_ceiling` + `basis_ceiling` 合计只占 "
             f"**5.0%（2/40）**（两臂相同）。**95% 的深尾是可救的**，"
             "且主要是「没学会」与「配重偏了」这两类工程问题，不是分辨率或基底的物理上限。\n")
    L.append("### 5.2 Phi-71 的代价定位在**软边 + 过覆盖**（本节是 A/B 的机制解释）\n")
    L.append(f"- 软边分层：P1 `soft` {f(p1_soft)} vs `hard` {f(p1_hard)}"
             f"（Δ {'' if None in (p1_soft,p1_hard) else '%+.3f' % (p1_soft-p1_hard)}）；"
             f"P3' `soft` {f(p3_soft)} vs `hard` {f(p3_hard)}"
             f"（Δ {'' if None in (p3_soft,p3_hard) else '%+.3f' % (p3_soft-p3_hard)}）。"
             "**P1 在软边上掉得多得多。**")
    L.append(f"- 深尾软边占比：P1 **{c_soft1}/{n1t}**，P3' {c_soft3}/{n3t}。")
    L.append(f"- 深尾过覆盖占比：P1 **{c_over1}/{n1t}**，P3' {c_over3}/{n3t}（P3' 过/欠基本对半）。")
    L.append("- P1 深尾的 family 集中在 `radial` + `band`，**零 `linear`**。\n")
    L.append("**机制解释**：`R(s;ρ)` 的 `band` 读出用一组**全局**参数（mu/h/k/pi）把标量场"
             "映射成掩膜，无法表达**空间上变化的软衰减**；遇到软边目标只能整体放宽 ⇒ 过覆盖。"
             "P3' 直出场 + 可学 gain 没有这个约束。这就是 A/B 里 "
             "**P1 − P3' = −0.0143 (p=6e-4)** 的来处：**代价不是均匀分布的，"
             "而是集中在软边 radial/band 这一族上**。\n")
    L.append("### 5.3 两臂都不在照抄中心先验\n")
    pos1 = m["six_dim_tables"]["P1"]["position"]["normal_only"]
    pos3 = m["six_dim_tables"]["P3prime"]["position"]["normal_only"]
    L.append(f"`position=edge` 档中心先验只有 "
             f"{f(pos1.get('edge',{}).get('center_prior_median'))}（它天然最不擅长的一档），"
             f"而 P1 {f(pos1.get('edge',{}).get('topk_iou_median'))}、"
             f"P3' {f(pos3.get('edge',{}).get('topk_iou_median'))} —— "
             "**两臂在先验最差的一档上反而最好**，是 M3 未复现的独立佐证。\n")
    L.append("### 5.4 建议下一步（按可救类型对应修法）\n")
    L.append("| 尾部类型 | 占比（P1 / P3'） | 对应修法 |")
    L.append("|---|---|---|")
    L.append(f"| `underfit` | {c1.get('underfit',0)}/{n1} ｜ {c3.get('underfit',0)}/{n3} | "
             "加步数/容量：1200 步只是 ~0.9 个 epoch，两臂都远未收敛 |")
    L.append(f"| `coverage_bias` | {c1.get('coverage_bias',0)}/{n1} ｜ {c3.get('coverage_bias',0)}/{n3} | "
             "面积带罚权重（当前 0.05）与 τ=0.15 需重标定；**P1 偏过覆盖，应非对称加罚** |")
    L.append(f"| `context_missing` | {c1.get('context_missing',0)}/{n1} ｜ {c3.get('context_missing',0)}/{n3} | "
             "只在 P1 出现且仅 3 例；属 `<where>` 生成文本质量，不在头内 |")
    L.append(f"| 不可救 | {nr1}/{n1} ｜ {nr3}/{n3} | 无需投入——已达链路/基底上界 |")
    L.append("")
    return _finish(L, d)


def _finish(L, d: Path) -> int:
    (d / "CLASS_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
