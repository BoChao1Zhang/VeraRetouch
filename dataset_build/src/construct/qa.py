"""Source-conditioned aesthetic QA — 12-dim binary F⊕R questionnaire.

Final gate for the 50k set. SOURCE (第一张) + variant (第二张, source with ONE color-grade
preset applied). Reuses the project's anti-bias method: shuffled position codes (model never
sees dim/polarity), F⊕R per dimension, anchor/honesty/gray traps, rule cleaner. Design +
adversarial critique in docs/source_qa/VARIANT_AESTHETIC_QA_DESIGN_2026-06-24.md.

12 dims = 6 VETO (broken⇒unusable) + 6 MERIT (quality points). Critique fixes applied:
  - contradiction = both-1 ONLY (both-0 = honest "clean but plain" middle, PASSES) — the
    twice-burned lesson; counting veto both-0 as a contradiction biases toward loud edits.
  - veto = F=0 ∧ R=1 (defect asserted AND quality denied), not R=1 alone.
  - hard-drop only when a deterministic CV metric corroborates (clip/crush/extreme-sat) OR
    ≥2 VLM veto dims fire; single uncorroborated VLM veto only penalizes merit (FPR control).
  - R claims are distinct symptoms (not ¬F) so both-1 is a real contradiction signal.
  - chroma dims skipped on B&W variants; face dims skipped when no person.
  - ARTIFACT/banding/halo are NOT VLM dims (35B can't see at 2 small images) → deterministic.

vLLM = config.VLLM_* (2 images/prompt). CLI smoke: python -m construct.qa
"""
from __future__ import annotations

import base64
import io
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import numpy as np
import requests
from PIL import Image, ImageFile, ImageOps

from dataset_build.source_qa import config

ImageFile.LOAD_TRUNCATED_IMAGES = True  # render downloads occasionally land a few bytes short

QA_WORKERS = 6
_SCRAMBLE_SEED = 20260624

# dim: (key, type, chroma, portrait, F_question[1=good], R_question[1=defect])
# Each question = 【explicit scope: which region/attribute】 + ONE concrete OBSERVABLE fact (not a
# holistic "professional/premium/intentional" verdict — the 35B can't answer those reliably).
_DIMS = [
    # --- VETO: concrete technical breakage, scoped, CV-corroborated ---
    ("HILIGHTCLIP", "veto", False, False,
     "【第二张最亮的区域(如天空、亮墙、皮肤反光)】里还能看到纹理或明暗变化吗？",
     "【第二张最亮的区域】是否出现成片纯白、完全看不到细节的死白块？"),
    ("SHADOWCRUSH", "veto", False, False,
     "【第二张的暗部/阴影区域】里还能看清物体的细节吗？",
     "【第二张的暗部】是否糊成纯黑一片、或出现一块块的色阶台阶，看不清细节？"),
    ("SATCLIP", "veto", True, False,
     "【第二张里最鲜艳的那块颜色】内部还能看出深浅和明暗层次吗？",
     "【第二张里某一块鲜艳颜色】是否糊成一整块实色、看不出深浅(或边缘出现溢色出血)？"),
    ("SKINTONE", "veto", True, True,
     "【第二张里人物的皮肤】看起来是正常健康人的肤色吗？",
     "【第二张里人物的皮肤】是否明显发橙、发黄、发红或发青，不像正常人的肤色？"),
    # --- MERIT: concrete observables, each clearly scoped ---
    ("CLEAN", "merit", False, False,
     "【第二张整体画面】干净通透吗(没有灰蒙蒙、脏雾的感觉)？",
     "【第二张整体画面】是否像蒙了一层灰或雾、发闷发脏不通透？"),
    ("WBQUALITY", "merit", True, False,
     "【第二张的整体色调/偏色】看起来舒服、像刻意设计的风格(暖调或冷调都协调)吗？",
     "【第二张的整体色调】是否别扭、像白平衡出错的脏偏色(整体发黄绿/发洋红等不讨喜)？"),
    ("MEMORYCOLOR", "merit", True, False,
     "【第二张里天空、草木、皮肤这些「我们知道大概是什么颜色」的东西】颜色还算可信吗？",
     "【第二张里天空/草木/皮肤】是否被染成明显不真实的怪颜色(天发青绿、草发黄、肤发橙品红)？"),
    ("CONTRAST", "merit", False, False,
     "【第二张的明暗对比】看起来自然舒适吗？",
     "【第二张的明暗对比】是否过强显得刺眼生硬、或过平显得灰扁？"),
    ("VIVID", "merit", True, False,
     "【第二张整体的鲜艳程度】恰到好处吗？",
     "【第二张整体颜色】是否浓艳到刺眼、过分饱和？"),
    ("SUBJECTPOP", "merit", False, True,
     "相比第一张，【第二张里的人物主体】是否更从背景中凸显、更有立体感？",
     "相比第一张，【第二张里的主体】是否变得更平、和背景更糊在一起？"),
    ("MOOD", "merit", False, False,
     "【第二张营造出的情绪氛围】和这张照片拍的内容/场景相配吗？",
     "【第二张的情绪氛围】和画面拍的内容明显矛盾吗(如把温馨暖场弄成阴冷)？"),
    ("BETTER", "merit", False, False,
     "总体看，【第二张】是不是比第一张更好看、更想多看一眼？",
     "总体看，【第二张】是不是其实不如第一张(更平庸/更脏/更难看)？"),
]
_CONTROLS = [
    ("ANCHOR", "anchor", "我看到了两张图：第一张原图、第二张是对它调色后的成片，同一画面只是颜色不同。", 1),
    ("HONPOS", "honesty_pos", "第二张是一张有可见画面内容的照片(不是空白/纯黑/纯色块)。", 1),
    ("HONNEG", "honesty_neg", "第二张画面完全空白、没有任何可见内容。", 0),
    ("GRAYTRAP", "trap_gray", "第二张是一张有明显色彩的彩色照片(不是黑白/单色)。", None),  # expected from measured cf
]

_QTEXT = {}            # audit_id -> question
for k, _t, _c, _p, fq, rq in _DIMS:
    _QTEXT[k + "_F"] = fq
    _QTEXT[k + "_R"] = rq
for k, kind, q, _e in _CONTROLS:
    _QTEXT[k] = q
_DIM_BY_KEY = {k: (t, chroma, p) for k, t, chroma, p, _f, _r in _DIMS}
# 2-phase: phase-1 = veto dims + all controls (~12 codes); phase-2 = merit dims (~16 codes),
# only run on veto survivors. Smaller per-call code count => far fewer parse-format failures.
_VETO_ITEMS = [(k + pol, k, pol[1:]) for k, t, c, p, f, r in _DIMS if t == "veto" for pol in ("_F", "_R")] \
    + [(k, k, kind) for k, kind, q, e in _CONTROLS]
_MERIT_ITEMS = [(k + pol, k, pol[1:]) for k, t, c, p, f, r in _DIMS if t == "merit" for pol in ("_F", "_R")]


def _pos_map(items, salt=0):
    """assign shuffled 2-digit codes to items (model sees only codes). deterministic per (items,salt)."""
    codes = [f"{i:02d}" for i in range(1, len(items) + 1)]
    random.Random(_SCRAMBLE_SEED + salt).shuffle(codes)
    return {codes[i]: it for i, it in enumerate(items)}  # code -> (audit_id, claim_key, polarity)


def _uri(path: str, longedge: int = 768) -> str:
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(path); im.load()
    im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((longedge, longedge))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _stats(path: str):
    im = Image.open(path).convert("RGB"); im.thumbnail((256, 256))
    a = np.asarray(im, "float32"); R, G, B = a[..., 0], a[..., 1], a[..., 2]
    rg = R - G; yb = 0.5 * (R + G) - B
    cf = float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))
    luma = 0.299 * R + 0.587 * G + 0.114 * B
    return {"cf": cf, "hi": float((a.max(-1) > 250).mean()), "lo": float((luma < 8).mean()),
            "luma": float(luma.mean())}


# Data-driven thresholds — calibrated on the preset-clean preview distribution (construct.calibrate
# dist, n=4000): clean renders reach hi-clip p90=0.16/p99=0.76, luma p99.5=203, lo-crush p99.9=0.35,
# cf p99.5=152. Probe renders skew bright/saturated, so these are CONSERVATIVE upper bounds — they
# flag only CLEARLY broken source renders; the VLM HILIGHT/SHADOW/SAT dims catch subtler cases.
# (Hand-picked hi>0.15 was wrong: it would false-flag ~10% of clean renders.)
_DET = {"hi": 0.45, "luma_hi": 205.0, "lo": 0.30, "luma_lo": 32.0, "cf_abs": 150.0, "cf_rel": 2.8}


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


_SYS = (
    "你是资深商业修图质检员。第一张是【原图】，第二张是对原图套用某调色预设后的真实成片。\n"
    "判断标准：自然、克制、高级、耐看才是好成片；不要因为更鲜艳/更亮/对比更强本身就给 yes——"
    "过饱和溢色、肤色失真、死白死黑、廉价滤镜感都是缺陷。请独立看两张图，逐题作答。\n"
    "下面每个『两位编号』对应一个 yes/no 判断，1=yes，0=no：\n{items}\n"
    "只输出一个 JSON 对象：键是两位编号(字符串)，值是 0 或 1，必须且只包含上面列出的全部编号，"
    '例如 {{"07":1,"02":0,"11":1}}。不要输出 JSON 以外的任何文字。'
)


def _call(suri: str, auri: str, prompt: str, temp: float, retries: int = 3) -> Optional[str]:
    payload = {"model": config.VLLM_MODEL, "max_tokens": 300, "temperature": temp,
               "messages": [{"role": "user", "content": [
                   {"type": "image_url", "image_url": {"url": suri}},
                   {"type": "image_url", "image_url": {"url": auri}},
                   {"type": "text", "text": prompt}]}]}
    if not config.VLLM_ENABLE_THINKING:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    headers = {"Authorization": f"Bearer {config.VLLM_API_KEY}", "X-vgate-class": "qa-judge"}
    for attempt in range(retries):
        try:
            r = requests.post(config.VLLM_BASE_URL + "/chat/completions", json=payload,
                              headers=headers, timeout=180)
            if r.status_code == 429 and attempt < retries - 1:
                time.sleep(float(r.headers.get("Retry-After", 2 * (attempt + 1)))); continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception:  # noqa: BLE001
            if attempt == retries - 1:
                return None
            time.sleep(2 * (attempt + 1))
    return None


def _active(key: str, is_portrait: bool, is_bw: bool) -> bool:
    t, chroma, p = _DIM_BY_KEY[key]
    if p and not is_portrait:
        return False
    if chroma and is_bw:
        return False
    return True


def _parse_json_bits(raw: str, expected: set):
    """Parse a {code: 0/1} JSON object. Models emit JSON far more reliably than packed 3-char codes
    (the old format lost/dup/invented codes). Returns (code->bit, ok, why)."""
    import re as _re
    import json_repair
    raw = _re.sub(r"<think>.*?</think>", "", raw or "", flags=_re.S)
    i = raw.find("{")
    if i < 0:
        return {}, False, "nojson"
    try:
        obj = json_repair.loads(raw[i:])
    except Exception:  # noqa: BLE001
        return {}, False, "jsonerr"
    if not isinstance(obj, dict):
        return {}, False, "notdict"
    bits = {}
    for k, v in obj.items():
        k = str(k).strip()
        try:
            b = int(v)
        except Exception:  # noqa: BLE001
            continue
        if k in expected and b in (0, 1):
            bits[k] = b
    if expected - set(bits):
        return bits, False, "missing"
    return bits, True, ""


def _run_phase(suri: str, auri: str, items: list, salt: int):
    """One VLM call over `items`. Returns (ans audit_id->bit, ok, why). Reask at temp 0 on parse fail."""
    pos_map = _pos_map(items, salt)
    prompt = _SYS.format(items="\n".join(f"{c}：{_QTEXT[pos_map[c][0]]}" for c in sorted(pos_map)))
    last = "empty"
    for temp in (0.1, 0.0):
        raw = _call(suri, auri, prompt, temp) or ""
        bit, ok, reason = _parse_json_bits(raw, set(pos_map))
        if ok:
            return {pos_map[c][0]: b for c, b in bit.items()}, True, ""
        last = f"parse:{reason}"
    return {}, False, last


def _contra(ans: dict, keys: list) -> int:
    return sum(1 for k in keys if ans.get(k + "_F") == 1 and ans.get(k + "_R") == 1)  # both-1 only


def qa_score_pair(suri: str, after_path: str, is_portrait: bool, src_cf: float) -> dict:
    try:
        return _qa_score_pair(suri, after_path, is_portrait, src_cf)
    except Exception as e:  # noqa: BLE001 - one corrupt render must not kill the batch
        return {"reliable": False, "why": f"exception:{type(e).__name__}", "veto": True, "q": -99}


def _qa_score_pair(suri: str, after_path: str, is_portrait: bool, src_cf: float) -> dict:
    """2-phase: phase-1 veto dims + controls; if not vetoed, phase-2 merit dims. Smaller calls
    (~12 / ~16 codes) => far better parse reliability than one 28-code call."""
    st = _stats(after_path)
    is_bw = st["cf"] < 8.0
    det = _det_flags(st, src_cf)
    auri = _uri(after_path)
    veto_keys = [k for k, t, c, p, f, r in _DIMS if t == "veto" and _active(k, is_portrait, is_bw)]
    merit_keys = [k for k, t, c, p, f, r in _DIMS if t == "merit" and _active(k, is_portrait, is_bw)]
    bad = lambda why, veto=True: {"reliable": False, "why": why, "veto": veto, "q": -99,
                                  "det": sorted(det), "is_bw": is_bw}

    # phase 1: veto dims + controls
    ans1, ok1, why1 = _run_phase(suri, auri, _VETO_ITEMS, salt=1)
    if not ok1:
        return bad(why1)
    if (ans1.get("ANCHOR") != 1 or ans1.get("HONPOS") != 1 or ans1.get("HONNEG") != 0
            or ans1.get("GRAYTRAP") != (0 if is_bw else 1)):
        return bad("control")
    if _contra(ans1, veto_keys) > 1:
        return bad("contra:veto")
    vlm_veto = [k for k in veto_keys if ans1.get(k + "_F") == 0 and ans1.get(k + "_R") == 1]
    corroborated = [k for k in vlm_veto if k in det]
    # cross-validation: hard-drop only when metric ∧ VLM AGREE, or >=2 VLM vetoes, or severe metric
    # extreme. metric-alone / vlm-alone are uncertain (xval showed low agreement) -> don't hard-drop.
    hard_veto = bool(corroborated) or len(vlm_veto) >= 2 or _det_extreme(st)
    base = {"reliable": True, "vlm_veto": vlm_veto, "det": sorted(det), "is_bw": is_bw}
    if hard_veto:
        return {**base, "veto": True, "q": -3.0, "why": "veto", "merit_score": 0.0}

    # phase 2: merit dims (survivors only)
    ans2, ok2, why2 = _run_phase(suri, auri, _MERIT_ITEMS, salt=2)
    if not ok2:
        return bad(f"merit_{why2}", veto=False)
    if _contra(ans2, merit_keys) > 1:
        return bad("contra:merit", veto=False)
    merit_hits = [k for k in merit_keys if ans2.get(k + "_F") == 1 and ans2.get(k + "_R") == 0]
    merit_score = len(merit_hits) / max(len(merit_keys), 1)
    soft_pen = 0.34 if vlm_veto else 0.0
    return {**base, "veto": False, "q": round(merit_score - soft_pen, 3), "why": "",
            "merit_score": round(merit_score, 3), "merit_hits": merit_hits, "soft_veto": bool(vlm_veto)}


def qa_rank(source_path: str, variants: List[Tuple[str, str]], scene: Optional[str] = None,
            is_portrait: bool = False) -> dict:
    if not variants:
        return {"ranking": [], "scores": {}}
    suri = _uri(source_path); src_cf = _stats(source_path)["cf"]
    with ThreadPoolExecutor(max_workers=QA_WORKERS) as ex:
        res = list(ex.map(lambda v: qa_score_pair(suri, v[1], is_portrait, src_cf), variants))
    scores = {lab: r for (lab, _), r in zip(variants, res)}
    ranking = sorted(scores, key=lambda l: (scores[l]["veto"], not scores[l].get("reliable", False),
                                            -scores[l]["q"]))
    return {"ranking": ranking, "scores": scores}


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
    print(f"n_dims={len(_DIMS)} phase1_codes={len(_VETO_ITEMS)} phase2_codes={len(_MERIT_ITEMS)}  qa._smoke OK")


if __name__ == "__main__":
    _smoke()
