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

ponytail: instruction/reasoning are v1 TEMPLATES from the preset's VLM caption + which QA merit
dims fired — flagged for refinement (these are the SFT think targets; a prior audit found think
quality matters). Swap in a VLM-written instruction/reasoning pass before full production.

CLI:  python -m construct.tier summary /path/to/r4_pilot   # threshold distribution from groups.jsonl
"""
from __future__ import annotations

import json
import os
import sys
from typing import List, Tuple

from .bank import load_captions

TAU_SFT = 0.62     # graded composite q to qualify as an SFT target (q in [-1,1], ~0.5 = neutral)
MARGIN_DPO = 0.22  # min q gap for a chosen/rejected DPO pair
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


def _instruction(cap: dict) -> str:
    name = (cap or {}).get("vlm_name") or "电影感调色"
    return f"请把这张照片调成「{name}」的风格。"   # v1 template — refine w/ a VLM instruction pass


def _reasoning(cap: dict, qa: dict, is_portrait: bool) -> str:
    # use vlm_function (natural usage/scene description) — NOT vlm_caption (leaks technical ΔL/Δb*
    # metrics, unnatural as a think target). v1 template; refine with a VLM reasoning pass.
    name = (cap or {}).get("vlm_name") or "该风格"
    fn = (cap or {}).get("vlm_function") or ""
    merits = _merit_phrase(qa)
    head = f"这张{'人像' if is_portrait else ''}照片适合「{name}」"
    return f"{head}:{fn} 应用后{merits}。" if fn else f"{head},应用后{merits}。"


def build(group: dict) -> Tuple[List[dict], List[dict]]:
    src = group["source"]; isp = group["is_portrait"]
    caps = _caps()
    cands = [c for c in group["candidates"] if c.get("qa")]
    _q = lambda c: c["qa"].get("q", -99)
    # SFT pool = reliable, non-veto, ranked by graded q. Unreliable are NOT discarded (per design):
    # their q is already deducted -> they fall out of SFT naturally and can serve as DPO rejecteds.
    keep = sorted([c for c in cands if c["qa"].get("reliable") and not c["qa"].get("veto")],
                  key=lambda c: -_q(c))
    rejects = sorted(cands, key=_q)   # worst-q first (veto / unreliable / low merit)

    def recipe(c):
        lc = c.get("local")
        if lc and lc.get("route") == "sam3":
            return {"kind": "lut_in_sam3", "base_preset_id": lc.get("base_preset_id"),
                    "base_preset_path": lc.get("base_preset_path"), "concept": lc["concept"]}
        if lc:
            return {"kind": "local_param", "mask_type": lc["mask_type"], "geom": lc["geom"],
                    "local_params": lc["local_params"], "base_preset_id": lc.get("base_preset_id")}
        return {"preset_id": c["preset_id"], "kind": c["kind"], "path": c["preset_path"],
                "fmt": c.get("fmt"), "content_hash": c.get("content_hash")}

    def sft_record(c, rank):
        lc = c.get("local")
        if lc and lc.get("route") == "sam3":
            cap = caps.get(lc.get("base_preset_id"), {})
            name = cap.get("vlm_name") or "所选调色"
            instr = f"请把这张照片的{lc['concept_cn']}调成「{name}」的风格。"
            reason = f"只对{lc['concept_cn']}局部应用「{name}」，" + _merit_phrase(c["qa"]) + "。"
            local = {"mask_unit_id": lc["mask_unit_id"], "concept": lc["concept"],
                     "C_GT": lc["cgt_path"], "base_preset_id": lc.get("base_preset_id")}
        elif lc:
            edit = _local_edit_phrase(lc.get("local_params") or {})
            instr = f"请把这张照片{lc['region']}区域{edit}。"
            reason = (f"对{lc['region']}区域局部{edit}，" + _merit_phrase(c["qa"]) + "。")
            local = {"mask_unit_id": lc["mask_unit_id"], "geom": lc["geom"], "C_GT": lc["cgt_path"],
                     "base_preset_id": lc.get("base_preset_id")}
        else:
            cap = caps.get(c["preset_id"], {})
            instr, reason, local = _instruction(cap), _reasoning(cap, c["qa"], isp), None
        return {"I_in": src, "I_tar": c["after_path"], "recipe": recipe(c), "local": local,
                "instruction": instr, "reasoning": reason,
                "qa": {"q": c["qa"].get("q"), "merit_score": c["qa"].get("merit_score"), "rank": rank,
                       "merit_hits": c["qa"].get("merit_hits"), "facets": c["qa"].get("facets")}}

    sft = [sft_record(c, rank) for rank, c in
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
                            "chosen": {"I_tar": chosen["after_path"], "recipe": recipe(chosen),
                                       "q": _q(chosen), "merit_score": chosen["qa"].get("merit_score")},
                            "rejected": {"I_tar": rej["after_path"], "recipe": recipe(rej),
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
        "note": "instruction/reasoning are v1 templates — refine with a VLM pass before full build",
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
