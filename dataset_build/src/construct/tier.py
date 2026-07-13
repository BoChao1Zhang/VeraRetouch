"""Tier / materialize the SFT + DPO data contract from a QA'd candidate group.

Contract (design):
  SFT  = {I_in=source, I_tar=rendered jpg, recipe={preset_id,kind,path,fmt,content_hash},
          local=None (global; Mask-v2 fills it for local samples), instruction, reasoning,
          qa={merit_score, rank}}
  DPO  = {I_in, chosen={...}, rejected={...}, margin}  — same source, a clearly-better vs worse pair.

Selection:
  - reliable & not-vetoed candidates, ranked by merit_score desc.
  - SFT: top-2 with merit_score >= TAU_SFT.
  - DPO: chosen = the SFT top; rejected = a vetoed render, else the lowest-merit reliable one,
    if the margin >= MARGIN_DPO.

instruction/reasoning：winner 选定**之后**（pair 先于 instruction 生成，顺序不变），若
config qa.vlm_annotate.enabled（默认 true）则由 annotate.annotate_winner（单次 35B 合并
调用 + 防泄露 guard）生成；vLLM 不可用/guard 重试耗尽时回退到本文件的模板——模板已修掉
两处泄露：auto/param 任务不点名 GT 风格名、local 任务的 instruction 不含由 GT local_params
翻译来的方向词（宁可朴素不可泄露）。q ∈ qa.verify_band 的 winner 额外走 vlm_clean.verify
（before+after 双图），结果只进 qa 字段、不进训练文本。

CLI:  python -m construct.tier summary /path/to/r4_pilot   # threshold distribution from groups.jsonl
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from typing import List, Tuple

from . import annotate
from .bank import load_captions

# OneAlign 后端（2026-07-13 默认）q 分布整体偏低（demo100 中位 0.516 vs 旧体系
# 0.577），TAU 配 0.50（≈SFT 1.6/组）；切回 artimuse_charm 时用 CONSTRUCT_TAU_SFT=0.55。
TAU_SFT = float(os.environ.get("CONSTRUCT_TAU_SFT", "0.50"))
MARGIN_DPO = 0.15  # min q gap for a chosen/rejected DPO pair
_CAPS = None


def _caps():
    global _CAPS
    if _CAPS is None:
        _CAPS = load_captions()
    return _CAPS


_MERIT_CN = {"CLEAN": "画面干净通透", "WBQUALITY": "整体色调讨喜", "MEMORYCOLOR": "记忆色可信",
             "CONTRAST": "明暗对比自然", "VIVID": "鲜艳度恰当", "SUBJECTPOP": "主体更突出",
             "MOOD": "氛围契合题材", "BETTER": "整体优于原图"}


def _local_edit_phrase(lp: dict) -> str:
    """Plain-language local edit from the mask's Local* params (drives the local instruction)."""
    v = []
    e = lp.get("LocalExposure2012", 0)
    if e > 0.05: v.append("提亮")
    elif e < -0.05: v.append("压暗")
    if lp.get("LocalContrast2012", 0) > 5: v.append("加强对比")
    if lp.get("LocalSaturation", 0) > 5: v.append("增艳")
    elif lp.get("LocalSaturation", 0) < -5: v.append("降饱和")
    if lp.get("LocalHighlights2012", 0) < -5: v.append("收高光")
    if lp.get("LocalShadows2012", 0) > 5: v.append("提阴影")
    if lp.get("LocalClarity2012", 0) > 5: v.append("增清晰")
    return "、".join(v) or "做局部调整"


def _merit_phrase(qa: dict) -> str:
    hits = (qa or {}).get("merit_hits") or []
    return "、".join(_MERIT_CN[h] for h in hits if h in _MERIT_CN) or "整体观感提升"


def _tpl_rng(key: str):
    import random
    return random.Random(int.from_bytes(hashlib.sha1(key.encode()).digest()[:4], "big"))


# 回退模板的种子化变体（v2 去模板化：annotate 失败回退时也不产生单一前缀）。全部无泄露。
_STYLE_TPLS = ("请把这张照片调成「{n}」的风格。",
               "想要「{n}」的感觉，帮我把这张照片调过去。",
               "帮我按「{n}」的风格处理一下这张。",
               "这张想试试「{n}」风格，麻烦调一下。")
_AUTO_TPLS = ("原片看着有点平，请帮我调得更好看、更耐看。",
              "帮我优化下这张的整体观感，让它更舒服耐看。",
              "总觉得原片差点意思，请帮我调得更有质感。",
              "麻烦调一下整体的色调和光感，让它更讨喜。")
_LOCAL_TPLS = ("{r}那一块看着和整体不太搭，帮我处理一下。",
               "{r}这部分帮我弄得舒服自然一点。",
               "照片里{r}区域的观感差点意思，请调整优化一下。",
               "请针对{r}区域做些调整，让画面更协调耐看。")


def _instruction(cap: dict, task_type: str = "style", seed_key: str = "") -> str:
    """回退模板。防泄露：只有 style 任务（风格名=任务输入）才点名 GT 风格名。"""
    rng = _tpl_rng(seed_key or "tpl")
    if task_type == "style":
        name = (cap or {}).get("vlm_name") or "电影感调色"
        return rng.choice(_STYLE_TPLS).format(n=name)
    return rng.choice(_AUTO_TPLS)   # auto/param：不点名风格


def _reasoning(cap: dict, qa: dict, is_portrait: bool, task_type: str = "style") -> str:
    # use vlm_function (natural usage/scene description) — NOT vlm_caption (leaks technical ΔL/Δb*
    # metrics, unnatural as a think target). 回退模板：auto/param 不出现风格名（防泄露）。
    fn = (cap or {}).get("vlm_function") or ""
    merits = _merit_phrase(qa)
    if task_type == "style":
        name = (cap or {}).get("vlm_name") or "该风格"
        head = f"这张{'人像' if is_portrait else ''}照片适合「{name}」"
    else:
        head = f"这张{'人像' if is_portrait else ''}照片适合这样的调色处理"
    return f"{head}:{fn} 应用后{merits}。" if fn else f"{head},应用后{merits}。"


def _task_type(src: str, c: dict) -> str:
    """按候选确定 task_type。local（Route 1）=local；global 一律 style
    （2026-07-12 重构：每组锁定一个风格大类后，同源多组的 winner 风格各异，
    无风格条件的 auto/param 会成为同图多峰监督——auto 样本改由 FiveK/PPR10K
    专家 GT 通路提供，preset 通路 instruction 全部点名风格）。
    Route 2（sam3 区域×LUT）已删除（2026-07-13 用户定案：local 只保留 Route 1）。"""
    return "local" if c.get("local") else "style"


def _recipe_of(c: dict) -> dict:
    lc = c.get("local")
    if lc:
        if not lc.get("local_params"):
            return {"kind": "local_preset", "mask_type": lc["mask_type"],
                    "geom": lc["geom"], "blend_mode": lc.get("blend_mode", "exact"),
                    "base_preset_id": lc.get("base_preset_id"),
                    "base_preset_path": lc.get("base_preset_path"),
                    "base_preset_content_hash": lc.get("base_preset_content_hash")}
        return {"kind": "local_param", "mask_type": lc["mask_type"], "geom": lc["geom"],
                "local_params": lc["local_params"], "base_preset_id": lc.get("base_preset_id")}
    return {"preset_id": c["preset_id"], "kind": c["kind"], "path": c["preset_path"],
            "fmt": c.get("fmt"), "content_hash": c.get("content_hash")}


def make_sft_record(group: dict, c: dict, rank: int, caps: dict) -> dict:
    """单条 SFT 记录构造（模板/VLM 标注/verify 全链）。build 与 retier 离线回收共用。"""
    src = group["source"]; isp = group.get("is_portrait", False)
    lc = c.get("local")
    task = _task_type(src, c)
    region = lparams = None
    seed_key = f"{src}|{c.get('preset_id')}"
    if lc:
        cap = caps.get(lc.get("base_preset_id"), {})
        # 回退模板防泄露：instruction 不含 GT 方向词（指令即答案）；方向词只进 reasoning。
        instr = _tpl_rng(seed_key).choice(_LOCAL_TPLS).format(r=lc["region"])
        if lc.get("local_params"):
            edit = _local_edit_phrase(lc["local_params"])
            reason = (f"对{lc['region']}区域局部{edit}，" + _merit_phrase(c["qa"]) + "。")
            lparams = lc["local_params"]
        else:
            name = cap.get("vlm_name") or "所选调色"
            reason = (f"只对{lc['region']}区域应用「{name}」，画面其余部分保持不变，"
                      + _merit_phrase(c["qa"]) + "。")
        local = {"mask_unit_id": lc["mask_unit_id"], "geom": lc["geom"], "C_GT": lc["cgt_path"],
                 "base_preset_id": lc.get("base_preset_id"),
                 "blend_mode": lc.get("blend_mode", "exact")}
        region = lc.get("region")
    else:
        cap = caps.get(c["preset_id"], {})
        instr = _instruction(cap, task, seed_key)
        reason = _reasoning(cap, c["qa"], isp, task)
        local = None
    # --- VLM 标注（单次合并调用）：成功则替换模板；失败回退上面的无泄露模板 ---
    # v2：img caption 锚定主体/场景 + 质量指标（source_iaa/after_iaa/q）供 reasoning 引用
    instr_short, annot_src = None, "template"
    annotate._bump("winners")
    if annotate.enabled():
        try:
            metrics = {k: v for k, v in {
                "source_iaa": group.get("source_iaa") or c["qa"].get("source_iaa"),
                "after_iaa": c["qa"].get("iaa_mixed"),
                "q": c["qa"].get("q")}.items() if isinstance(v, (int, float))}
            ann = annotate.annotate_winner(
                c["after_path"], task, cap, local_region=region,
                local_params=lparams, source_path=src,
                img_caption=annotate.source_caption(src), metrics=metrics)
            instr, instr_short = ann["instruction_long"], ann["instruction_short"]
            reason, annot_src = ann["reasoning"], "vlm"
        except Exception as e:  # noqa: BLE001 - 标注失败必须回退模板而非丢样本
            annotate._bump("annotate_fallback")
            print(f"[annotate-fallback] {os.path.basename(c['after_path'])[:24]} "
                  f"{task}: {type(e).__name__}: {str(e)[:80]}")
    rec = {"I_in": src, "I_tar": c["after_path"], "recipe": _recipe_of(c), "local": local,
           "task_type": task, "instruction": instr, "instruction_short": instr_short,
           "reasoning": reason, "annot_src": annot_src,
           "qa": {"q": c["qa"].get("q"), "merit_score": c["qa"].get("merit_score"), "rank": rank,
                  "merit_hits": c["qa"].get("merit_hits"), "facets": c["qa"].get("facets")}}
    # --- verify 条件化：仅 q ∈ qa.verify_band 的 winner 走 before+after 双图 verify；
    #     结果进 qa 字段（诊断信号），绝不进训练文本。 ---
    if annotate.enabled() and annotate.should_verify(c["qa"].get("q")):
        v = annotate.verify_winner(src, c["after_path"], instr)
        if v:
            rec["qa"]["verify"] = v
    return rec


def build(group: dict) -> Tuple[List[dict], List[dict]]:
    src = group["source"]
    caps = _caps()
    cands = [c for c in group["candidates"] if c.get("qa")]
    _q = lambda c: c["qa"].get("q", -99)
    # SFT pool = reliable, non-veto, ranked by graded q. Unreliable are NOT discarded (per design):
    # their q is already deducted -> they fall out of SFT naturally and can serve as DPO rejecteds.
    keep = sorted([c for c in cands if c["qa"].get("reliable") and not c["qa"].get("veto")],
                  key=lambda c: -_q(c))
    # rejected 只收「真差」（veto 或可靠低分）。打分失败（reliable=False 且非 veto，
    # q=0.0 是 missing_iaa 占位）不是负样本——它只是没被评过，进 DPO 会教模型
    # 讨厌一张可能不差的渲染（review 2026-07-13 确认曾实际发生）。
    rejects = sorted([c for c in cands if c["qa"].get("veto") or c["qa"].get("reliable")],
                     key=_q)   # worst-q first (veto / reliable low merit)

    sft = [make_sft_record(group, c, rank, caps) for rank, c in
           enumerate(k for k in keep if _q(k) >= TAU_SFT) if rank < 2]

    dpo = []
    if keep and rejects:
        chosen = keep[0]
        for rej in rejects:
            if rej["after_path"] == chosen["after_path"]:
                continue
            margin = _q(chosen) - _q(rej)
            if rej["qa"].get("veto") or margin >= MARGIN_DPO:
                dpo.append({"I_in": src,
                            "chosen": {"I_tar": chosen["after_path"], "recipe": _recipe_of(chosen),
                                       "q": _q(chosen), "merit_score": chosen["qa"].get("merit_score")},
                            "rejected": {"I_tar": rej["after_path"], "recipe": _recipe_of(rej),
                                         "q": _q(rej), "merit_score": rej["qa"].get("merit_score"),
                                         "veto": rej["qa"].get("veto")},
                            "margin": round(margin, 3)})
                break  # one clean pair per source for now
    return sft, dpo


def summarize(groups: list, sft: list, dpo: list) -> dict:
    import numpy as np
    n = len(groups)
    allc = [c for g in groups for c in g["candidates"] if c.get("qa")]
    qs = [c["qa"].get("q") for c in allc if c["qa"].get("q") is not None and not c["qa"].get("veto")]
    rel = sum(1 for c in allc if c["qa"].get("reliable"))
    tot = sum(len(g["candidates"]) for g in groups)
    veto = sum(1 for c in allc if c["qa"].get("veto"))
    pct = (lambda arr, p: round(float(np.percentile(arr, p)), 3))
    return {
        "n_sources": n, "candidates": tot, "reliable_frac": round(rel / max(tot, 1), 3),
        "veto_frac": round(veto / max(tot, 1), 3),
        "sources_with_sft": len({s["I_in"] for s in sft}), "sft_records": len(sft),
        "sources_with_dpo": len({d["I_in"] for d in dpo}), "dpo_pairs": len(dpo),
        "q_dist_nonveto": {f"p{p}": pct(qs, p) for p in (5, 10, 25, 50, 75, 90, 95)} if qs else {},
        "q_mean": round(float(np.mean(qs)), 3) if qs else None,
        "q_std": round(float(np.std(qs)), 3) if qs else None,
        "TAU_SFT": TAU_SFT, "MARGIN_DPO": MARGIN_DPO,
        "annot_src": {k: sum(1 for s in sft if s.get("annot_src") == k)
                      for k in ("vlm", "template")},
        "vlm_annotate": annotate.stats(),   # 单样本调用预算口径（annotate/verify 次数）
    }


def main() -> None:
    if len(sys.argv) >= 3 and sys.argv[1] == "summary":
        d = sys.argv[2]
        groups = [json.loads(l) for l in open(os.path.join(d, "groups.jsonl")) if l.strip()]
        sft = [json.loads(l) for l in open(os.path.join(d, "sft.jsonl")) if l.strip()]
        dpo = [json.loads(l) for l in open(os.path.join(d, "dpo.jsonl")) if l.strip()]
        print(json.dumps(summarize(groups, sft, dpo), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
