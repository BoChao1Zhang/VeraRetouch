"""Construct QA — OneAlign IAA 组内排序 + 确定性极端失败 veto。

qa_rank：源图打一次分作参照，候选 q = 0.72·绝对分 + 0.28·对源提升
（objscore 后端见 CONSTRUCT_IAA_BACKEND，2026-07-13 起默认 onealign）；
det_extreme（死白/死黑）在打分前直接 veto。12 维 VLM 问卷与 Bradley-Terry
pairwise 已随 IAA 主线定案删除（历史见 git；设计文档
docs/source_qa/VARIANT_AESTHETIC_QA_DESIGN_2026-06-24.md）。

_uri/_VLLM_GATE 为 annotate（35B instruction 生成）共享的 vLLM 工具。
CLI smoke: python -m construct.qa
"""
from __future__ import annotations

import base64
import io
import math
import threading
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFile, ImageOps

from dataset_build.source_qa import config
from . import objscore

ImageFile.LOAD_TRUNCATED_IMAGES = True  # render downloads occasionally land a few bytes short

# Global vLLM admission（annotate 消费）：aggregate fan-out 必须尊重引擎并发上限。
_VLLM_GATE = threading.BoundedSemaphore(int(getattr(config, "VLLM_CONCURRENCY", 16)))

_W_IAA_ABS, _W_IAA_IMPR = 0.72, 0.28   # qa_rank: absolute IAA / improvement over source

_DET = {"hi": 0.45, "luma_hi": 205.0, "lo": 0.30, "luma_lo": 32.0, "cf_abs": 150.0, "cf_rel": 2.8}


def _uri(path: str, longedge: int = 768) -> str:
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(path); im.load()
    im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((longedge, longedge))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _obj_detail(obj, key: str) -> Optional[float]:
    if isinstance(obj, dict) and obj.get(key) is not None:
        return float(obj[key])
    return None


def _stats(path: str):
    im = Image.open(path).convert("RGB"); im.thumbnail((256, 256))
    a = np.asarray(im, "float32"); R, G, B = a[..., 0], a[..., 1], a[..., 2]
    rg = R - G; yb = 0.5 * (R + G) - B
    cf = float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))
    luma = 0.299 * R + 0.587 * G + 0.114 * B
    return {"cf": cf, "hi": float((a.max(-1) > 250).mean()), "lo": float((luma < 8).mean()),
            "luma": float(luma.mean()), "lstd": float(luma.std())}


def _det_flags(after_stats, src_cf: float) -> set:
    """Moderate flags — used to CORROBORATE the VLM veto (hard-drop only when both agree; xval
    showed metric-alone over-fires on HILIGHT and under-fires on SAT vs the VLM)."""
    f = set()
    if after_stats["hi"] > _DET["hi"] or after_stats["luma"] > _DET["luma_hi"]:
        f.add("HILIGHTCLIP")
    if after_stats["lo"] > _DET["lo"] or after_stats["luma"] < _DET["luma_lo"]:
        f.add("SHADOWCRUSH")
    if after_stats["cf"] > max(_DET["cf_abs"], _DET["cf_rel"] * src_cf):
        f.add("SATCLIP")
    return f


def _det_extreme(after_stats) -> bool:
    """Severe, unambiguous technical breakage — trustworthy to hard-drop ALONE regardless of VLM."""
    return (after_stats["hi"] > 0.70 or after_stats["luma"] > 218
            or after_stats["lo"] > 0.45 or after_stats["luma"] < 25)


def _iaa_rank_q(iaa: Optional[float], src_iaa: Optional[float]) -> tuple[float, float]:
    if iaa is None:
        return 0.0, 0.5
    abs_q = max(0.0, min(1.0, iaa / 100.0))
    if src_iaa is None:
        impr = 0.5
    else:
        impr = 0.5 + 0.5 * math.tanh(2.0 * ((iaa - src_iaa) / 100.0))
    q = _W_IAA_ABS * abs_q + _W_IAA_IMPR * impr
    return max(0.0, min(1.0, q)), impr


def qa_rank(source_path: str, variants: List[Tuple[str, str]], scene: Optional[str] = None,
            is_portrait: bool = False) -> dict:
    """Rank rendered global/local candidates by Artimuse+Charm mixed IAA.

    The source photo is scored once as the reference. Candidate q blends absolute
    after quality with improvement over source, while deterministic extreme
    exposure/color failures still veto unusable renders before tiering.
    """
    if not variants:
        return {"ranking": [], "scores": {}}
    src_stats = _stats(source_path)
    src_cf = src_stats["cf"]
    # 注意：不要用 score_many 批路径——生产实测批内打分被系统性压低 12-19 分
    # （单独重打 60-69 vs 批内记录 44-50，2026-07-06；根因待查，回填场景的 A/B 未覆盖
    # 本混合内容批），导致 SFT 产率 1.7→0.11/组。逐张 ~0.13s 足够快。
    src_obj = objscore.score(source_path)
    src_iaa = objscore.mixed_value(src_obj)

    scores = {}
    for lab, path in variants:
        st = _stats(path)
        det = sorted(_det_flags(st, src_cf))
        if _det_extreme(st):
            scores[lab] = {
                "veto": True, "reliable": True, "q": 0.0, "merit_score": 0.0,
                "merit_hits": [], "is_bw": st["cf"] < 8.0, "det": det,
                "why": "det_extreme", "source_iaa": src_iaa,
            }
            continue
        obj = objscore.score(path)
        iaa = objscore.mixed_value(obj)
        q, impr = _iaa_rank_q(iaa, src_iaa)
        reliable = iaa is not None
        scores[lab] = {
            "veto": False,
            "reliable": reliable,
            "q": round(q, 4),
            "merit_score": round(q, 4),
            "merit_hits": [],
            "is_bw": st["cf"] < 8.0,
            "det": det,
            "iaa_mixed": None if iaa is None else round(float(iaa), 3),
            "artimuse": _obj_detail(obj, "artimuse"),
            "charm": _obj_detail(obj, "charm"),
            "source_iaa": None if src_iaa is None else round(float(src_iaa), 3),
            "iaa_impr": round(impr, 4),
            "beat_source": bool(iaa is not None and src_iaa is not None and iaa > src_iaa),
            "qa_mode": objscore.BACKEND,
            "why": "" if reliable else "missing_iaa",
        }
    ranking = sorted(scores, key=lambda l: (scores[l]["veto"], -scores[l]["q"]))
    return {"ranking": ranking, "scores": scores, "qa_fail_frac": 0.0,
            "source_iaa": None if src_iaa is None else round(float(src_iaa), 3),
            "qa_mode": objscore.BACKEND}


def _smoke() -> None:
    import json
    from dataset_build.source_qa import db
    from . import render
    feats = [json.loads(l) for _, l in zip(range(4), open(
        "/home/bc/data/datasets/vera_directionA_1M/preset_bank_full/features.jsonl"))]
    conn = db.connect()
    src = conn.execute("SELECT path FROM assets WHERE asset_type='image' AND b_quality=3 "
                       "AND dup_of IS NULL ORDER BY asset_id LIMIT 1").fetchall()[0]["path"]
    conn.close()
    variants = []
    for f in feats:
        r = render.render_preset(f["path"], f["kind"], f.get("fmt"), src)
        if r.get("ok"):
            variants.append((f["preset_id"], r["after_path"]))
    out = qa_rank(src, variants)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    assert out["scores"], "qa returned nothing"
    print("qa._smoke OK")


if __name__ == "__main__":
    _smoke()
