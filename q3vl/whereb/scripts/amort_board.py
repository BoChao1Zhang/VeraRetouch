"""Build the PR-AMORT criteria table (and the P1 vs P3' A/B) from the boards.

Kept separate from the training entrypoint so the table can be regenerated from
artefacts alone, without a GPU and without re-running an arm -- which is also
what makes it usable by the result reviewer, who by protocol reads only the
delivery folder.

Every row carries the mandatory comparison columns (centre prior, random floor)
and the E3 falsification pair.  Numbers are quoted in the published convention:
**matched-area top-k IoU**.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

# Reference numbers in the REPORTING convention: matched-area top-k IoU,
# **normal-only** (winner_confidence == "normal").  Review U7: the pooled
# figures every earlier card quotes include 44% `low` samples, which must not
# enter evaluation GT and which inflate every IoU column at once (larger GT
# area -> higher random floor).  Recomputed 2026-08-11 from the raw masks and
# from W01's own per_sample.jsonl; the pooled column is kept only so the two
# can never be silently swapped.
PUBLISHED = {
    "random_floor": 0.2254,
    "center_prior": 0.4853,
    "center_prior_bf1": 0.2276,
    "W01_generated": 0.4874,
    "W01_gt": 0.5109,
    "oracle_ceiling": None,          # not yet recomputed normal-only
}
PUBLISHED_POOLED = {
    "random_floor": 0.2582, "center_prior": 0.5088, "center_prior_bf1": 0.1994,
    "W01_generated": 0.5333, "W01_gt": 0.5500, "oracle_ceiling": 0.9737,
    "prior_projection_best": 0.5163,
}


def _f(x: Any, n: int = 4) -> str:
    return "n/a" if x is None else f"{float(x):.{n}f}"


def _p(d: Any) -> Any:
    """p-value under either key.

    ``q3vl.whereb.metrics.paired_delta`` returns ``p_value`` (10,000 sign-flip
    permutations, add-one corrected so it can never be exactly 0); the local
    :func:`paired` helper returns ``p``.  Reading only one of the two silently
    printed "n/a" for every p-value on the board -- i.e. dropped the entire
    significance column while still showing the deltas.
    """
    if not isinstance(d, dict):
        return None
    return d.get("p_value", d.get("p"))


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def _rows(per_sample: Path, mode: str, normal_only: bool = True) -> list[dict]:
    """Rows for one context.  `normal_only` keeps the A/B on the reporting
    convention -- `low` samples must not enter evaluation GT (U7)."""
    out = []
    with per_sample.open() as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("mode") != mode or r.get("uncovered") or r.get("is_fake"):
                continue
            if normal_only and r.get("winner_confidence") != "normal":
                continue
            out.append(r)
    return out


def paired(a: list[float], b: list[float], n_perm: int = 10000,
           seed: int = 0) -> dict[str, float]:
    """Paired difference with a permutation p-value (sign-flip)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {"delta": float("nan"), "p": float("nan"), "n": 0}
    obs = float(np.median(d))
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, d.size))
    null = np.median(signs * d[None, :], axis=1)
    p = float((np.abs(null) >= abs(obs)).mean())
    return {"delta": obs, "p": p, "n": int(d.size)}


def arm_table(board: dict, main: str | None = None) -> str:
    main = main or board.get("main_context", "generated")
    s = board["contexts"][main]
    # The measured columns and the reference columns MUST share one convention.
    # `summarise_rows` reports pooled aggregates plus a `headline_normal_only`
    # block; PUBLISHED holds normal-only numbers.  Reading the pooled aggregate
    # against them silently compares 400 samples (44% of which are `low`, whose
    # larger GT area lifts every IoU column) with a 224-sample reference -- the
    # mixed-convention error U7 raised, running in the opposite direction.
    hn = s.get("headline_normal_only") or {}
    L = []
    L.append(f"### 主表（context = `{main}`，口径 = matched-area top-k IoU，"
             f"**population = normal-only n={hn.get('n', '?')}**；"
             f"pooled 并列于末行）\n")
    L.append("| 列 | 实测 | 预注册/参照 | 判读 |")
    L.append("|---|---|---|---|")
    topk = hn.get("topk_iou_median", s.get("topk_iou", {}).get("median"))
    cp = hn.get("center_prior_median", s.get("center_prior_topk_iou", {}).get("median"))
    rf = hn.get("random_floor_median", s.get("random_floor", {}).get("median"))
    L.append(f"| **top-k IoU 中位** | **{_f(topk)}** | W01 {PUBLISHED['W01_generated']} / "
             f"中心先验 {PUBLISHED['center_prior']} / 地板 {PUBLISHED['random_floor']} | "
             f"{'超过 W01' if topk and topk > PUBLISHED['W01_generated'] else ('超过中心先验' if topk and topk > PUBLISHED['center_prior'] else '未过中心先验')} |")
    L.append(f"| 中心先验列（本次实测） | {_f(cp)} | normal-only {PUBLISHED['center_prior']} | 复算一致性 |")
    L.append(f"| 随机地板 a/(2−a) | {_f(rf)} | normal-only {PUBLISHED['random_floor']} | — |")
    L.append(f"| grid 边界 F1 | {_f(hn.get('boundary_f1_median', s.get('grid_boundary_f1', {}).get('median')))} | "
             f"中心先验 {PUBLISHED['center_prior_bf1']} | 形状贴合 |")
    L.append(f"| 裸 soft-IoU（并列，非主列） | {_f(s.get('soft_iou_raw', {}).get('median'))} | — | 口径不同勿混用 |")
    d = hn.get("delta_vs_center_prior") or s.get("delta_vs_center_prior", {})
    L.append(f"| **vs 中心先验 配对 Δ** | **{_f(d.get('delta'))}** (p={_f(_p(d), 4)}) | "
             f"W01 normal-only 仅 +0.0017 | {'显著优于 W01 病灶' if d.get('delta') and d['delta'] > 0.0017 else '未超过 W01 病灶水平'} |")
    d2 = s.get("delta_vs_random_floor", {})
    L.append(f"| vs 随机地板 配对 Δ | {_f(d2.get('delta'))} (p={_f(_p(d2), 4)}) | >0 | — |")
    L.append(f"| 面积比中位 | {_f(s.get('area_ratio', {}).get('median'))} | 门 [0.8, 1.5] | — |")
    L.append(f"| std 比中位 | {_f(s.get('std_ratio', {}).get('median'))} | 早警 <0.5 | — |")

    m3 = hn.get("corr_center_minus_corr_gt") or s.get("corr_center_minus_corr_gt", {})
    L.append(f"| **corr(中心先验) − corr(GT)** | **{_f(m3.get('delta'))}** (p={_f(_p(m3), 4)}) | "
             f"**<0 才可晋级** | {'**M3 同病，不得晋级**' if s.get('m3_disease_present') else 'M3 未复现，可晋级'} |")

    sw = board.get("swap_subject_delta", {})
    L.append(f"| 换主体配对 Δ | {_f(sw.get('delta'))} (p={_f(_p(sw), 4)}, n={sw.get('n_pairs')}) | 门 >0；W01 +0.145 | 条件性 |")
    an = board.get("antonym_invariance", {})
    L.append(f"| antonym 不变性 \\|Δ\\| 中位 | {_f(an.get('median_abs_delta'))} | 门 ≤0.05（只报不训） | "
             f"{'PASS' if an.get('within_threshold') else 'FAIL'} |")
    for neg in ("fixed_phrase", "irrelevant_words"):
        k = f"delta_vs_{neg}"
        if k in board:
            L.append(f"| vs {neg} 配对 Δ | {_f(board[k].get('delta'))} (p={_f(_p(board[k]), 4)}) | W01 +0.23 | 负控制 |")

    L.append(f"| *(pooled 并列，n={s.get('n')})* | *{_f(s.get('topk_iou', {}).get('median'))}* | "
             f"*中心先验 {_f(s.get('center_prior_topk_iou', {}).get('median'))} / "
             f"地板 {_f(s.get('random_floor', {}).get('median'))}* | *含 44% low，勿用于判据* |")
    g = board.get("gate", {})
    L.append(f"\n**硬门**：面积比 {'PASS' if g.get('area_ratio_ok') else 'FAIL'} ｜ "
             f"换主体 Δ>0 {'PASS' if g.get('swap_delta_ok') else 'FAIL'} ｜ "
             f"antonym {'PASS' if g.get('antonym_ok') else 'FAIL'} ⇒ "
             f"**{'PASS' if g.get('pass') else 'FAIL'}**\n")

    st = s.get("strata", {})
    if st.get("area"):
        L.append("### area 分层（≥0.45 档地板已 0.527，占比约一半，单列不作主张）\n")
        L.append("| area | n | top-k IoU | 中心先验 | 随机地板 | 边界 F1 |")
        L.append("|---|---|---|---|---|---|")
        for k in sorted(st["area"]):
            v = st["area"][k]
            L.append(f"| {k} | {v['n']} | {_f(v['topk_iou_median'])} | {_f(v['center_prior_median'])} | "
                     f"{_f(v['random_floor_median'])} | {_f(v['boundary_f1_median'])} |")
        L.append("")
    if st.get("family"):
        L.append("### mask_type 分层\n")
        L.append("| family | n | top-k IoU | 中心先验 | 随机地板 | 边界 F1 |")
        L.append("|---|---|---|---|---|---|")
        for k in sorted(st["family"]):
            v = st["family"][k]
            L.append(f"| {k} | {v['n']} | {_f(v['topk_iou_median'])} | {_f(v['center_prior_median'])} | "
                     f"{_f(v['random_floor_median'])} | {_f(v['boundary_f1_median'])} |")
        L.append("")
    ms = board.get("m_sem")
    if ms:
        L.append("### m_sem（语义子集，一等交付）\n")
        L.append(f"- n = {ms['n']}；top-k IoU 中位 **{_f(ms['topk_iou_median'])}**；"
                 f"边界 F1 {_f(ms['boundary_f1_median'])}；"
                 f"中心先验 {_f(ms['center_prior_median'])}；地板 {_f(ms['random_floor_median'])}")
        L.append(f"- 路由到语义头的比例：{_f(ms.get('routed_to_semantic_head'), 3)}\n")
    r = board.get("routing", {})
    if r.get("confusion"):
        L.append("### 路由实测（family × head）\n")
        heads = sorted({h for v in r["confusion"].values() for h in v})
        L.append("| family | " + " | ".join(heads) + " |")
        L.append("|---" * (len(heads) + 1) + "|")
        for fam in sorted(r["confusion"]):
            L.append(f"| {fam} | " + " | ".join(str(r["confusion"][fam].get(h, 0))
                                                for h in heads) + " |")
        L.append("")
    return "\n".join(L)


def ab_table(b1: dict, b3: dict, d1: Path, d3: Path, main: str = "generated") -> str:
    """The A/B the pair exists for: is Phi-71 an asset or a liability?"""
    r1 = {r["sample_id"]: r for r in _rows(d1 / "per_sample.jsonl", main)}
    r3 = {r["sample_id"]: r for r in _rows(d3 / "per_sample.jsonl", main)}
    ids = sorted(set(r1) & set(r3))
    a = [r1[i]["hard_iou"] for i in ids]
    b = [r3[i]["hard_iou"] for i in ids]
    st = paired(a, b)
    L = ["## A/B：Phi-71 中间层是资产还是负债\n",
         f"同图配对，n = {st['n']}（**normal-only**），context = `{main}`，"
         f"口径 = matched-area top-k IoU\n",
         "| 臂 | top-k IoU 中位 | 硬门 | M3 证伪列 |", "|---|---|---|---|"]
    for name, bd in (("P1（经 Phi-71）", b1), ("P3'（不经 Phi）", b3)):
        s = bd["contexts"].get(main, {})
        hn = s.get("headline_normal_only") or {}
        L.append(f"| {name} | {_f(hn.get('topk_iou_median', s.get('topk_iou', {}).get('median')))} | "
                 f"{'PASS' if bd.get('gate', {}).get('pass') else 'FAIL'} | "
                 f"{'同病' if s.get('m3_disease_present') else '未复现'} |")
    L.append(f"\n**配对差分 P1 − P3' = {_f(st['delta'])}（p = {_f(_p(st), 4)}）**\n")
    if st["p"] > 0.05:
        v = ("差分不显著 ⇒ **Phi-71 在前馈链路里既不帮也不拖**。"
             "这本身是可交付结论，且直接决定 What 侧是否继续维护 (w*, ρ*) 这条 latent 接口。")
    elif st["delta"] > 0:
        v = "P1 显著更高 ⇒ **Phi-71 是资产**，解析基底为前馈头提供了有用的归纳偏置。"
    else:
        v = "P3' 显著更高 ⇒ **Phi-71 是负债**，把它留在前馈链路里要付代价。"
    L.append(f"**判读**：{v}\n")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p1", default="/home/bc/data/runs/where_b/amort_P1_20260810/eval_final")
    ap.add_argument("--p3", default="/home/bc/data/runs/where_b/amort_P3prime_20260810/eval_final")
    ap.add_argument("--main", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    parts: list[str] = []
    boards: dict[str, dict] = {}
    for tag, d in (("P1", Path(args.p1)), ("P3prime", Path(args.p3))):
        f = d / "metrics.json"
        if not f.exists():
            parts.append(f"## {tag}\n\n(no board at {f})\n")
            continue
        boards[tag] = _load(f)
        parts.append(f"## {tag}\n\n" + arm_table(boards[tag], args.main))
    if len(boards) == 2:
        main_ctx = args.main or boards["P1"].get("main_context", "generated")
        parts.append(ab_table(boards["P1"], boards["P3prime"],
                              Path(args.p1), Path(args.p3), main_ctx))
    text = "\n".join(parts)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
