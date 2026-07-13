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
import hashlib
import io
import math
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import numpy as np
import requests

from . import objscore

# --- objective quality combination ------------------------------------------- #
# The VLM questionnaire is retained as a diagnostic fallback, but construct ranking is now led by
# the same mixed IAA signal used in source cleaning: Artimuse + Charm absolute quality and the
# candidate's improvement over the source photo.
_W_VLM, _W_OQUAL, _W_OIMPR = 0.25, 0.50, 0.25   # legacy pair scorer: VLM composite / IAA quality / IAA gain
_W_IAA_ABS, _W_IAA_IMPR = 0.72, 0.28            # qa_rank: absolute IAA / improvement over source


def _objective(after_obj, src_obj, polish: float):
    """(iaa_quality, iaa_improvement) in [0,1]; falls back to pixel polish if IAA is missing."""
    iaa_a = objscore.mixed_value(after_obj)
    if iaa_a is None:
        return polish, 0.5
    obj_q = max(0.0, min(1.0, iaa_a / 100.0))
    iaa_s = objscore.mixed_value(src_obj)
    if iaa_s is not None:
        d = (iaa_a - iaa_s) / 100.0   # aesthetic gain vs the source photo
        obj_i = 0.5 + 0.5 * math.tanh(2.0 * d)
    else:
        obj_i = 0.5
    return obj_q, obj_i


def _obj_detail(obj, key: str) -> Optional[float]:
    if isinstance(obj, dict) and obj.get(key) is not None:
        return float(obj[key])
    return None
from PIL import Image, ImageFile, ImageOps

from dataset_build.source_qa import config

ImageFile.LOAD_TRUNCATED_IMAGES = True  # render downloads occasionally land a few bytes short

QA_WORKERS = 6
# Global vLLM admission: per-source pools are 6 wide, but with many sources in
# flight the aggregate fan-out must still respect the engine cap — oversubscribed
# threads block here cheaply instead of triggering vGate 429 storms.
_VLLM_GATE = threading.BoundedSemaphore(int(getattr(config, "VLLM_CONCURRENCY", 16)))
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
            "luma": float(luma.mean()), "lstd": float(luma.std())}


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
    "判断标准：自然、克制、高级、耐看才是好成片。若第二张确实更通透/色调更讨喜/层次更好/更耐看，"
    "请如实给 yes，不要一味挑剔；但也不要因为单纯更鲜艳/更亮/对比更强就给 yes——"
    "过饱和溢色、肤色失真、死白死黑、廉价滤镜感才是缺陷。请独立看两张图，按事实逐题作答。\n"
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
            with _VLLM_GATE:
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


# --------------------------------------------------------------------------- #
# graded composite scoring (extends source_qa.clean_aes: keeps its gate order + both-1 contradiction
# + merit-as-soft-signal, but turns the discrete merit_frac into a GRADED, logically-gated composite
# so scores spread instead of collapsing to 0/1, and unreliable => deduction not a hard zero).
# --------------------------------------------------------------------------- #
_W = {"overall": 0.30, "tone": 0.22, "color": 0.22, "mood": 0.13, "pop": 0.13}  # facet weights (Σ=1)
_GATE_FLOOR = 0.40        # a defect-gated facet keeps this fraction (logical coupling, not annihilation)
_MONO_DISCOUNT = 0.10     # B&W is a heavy desaturation; small discount kills "B&W always wins"
_PEN_SOFTVETO = 0.06      # per uncorroborated VLM veto dim
_PEN_CONTRA = 0.07        # per both-1 contradiction beyond tolerance
_CONTRA_TOL = 1           # borrowed from source_qa AES_CONTRA_TOL: tolerate 1 contradiction
_NEUTRAL = 0.5            # honest-middle baseline
_DED_PARSE = 0.32         # unreliable (phase unparseable) deduction from neutral
_DED_CONTROL = 0.40       # judge failed sanity controls -> heavier deduction (but NOT -99/drop)


def _graded(ans: dict, key: str) -> float:
    """F⊕R -> 3-level signal. 1.0 = good asserted & no defect; 0.0 = defect asserted & not-good;
    0.5 = honest middle (both-0) OR contradiction (both-1). This is the gradient fix."""
    f, r = ans.get(key + "_F"), ans.get(key + "_R")
    if f == 1 and r == 0:
        return 1.0
    if f == 0 and r == 1:
        return 0.0
    return 0.5


def _vhealth(ans: dict, key: str, det: set) -> float:
    """veto-dim health in [0,1] (1=clean). Only an ASSERTED defect (R=1∧F=0) pulls it down — an
    ambiguous both-0 ('no defect confirmed') is healthy, so the gate doesn't punish fine renders the
    judge was merely unsure about. A corroborating det metric forces it low."""
    f, r = ans.get(key + "_F"), ans.get(key + "_R")
    if r != 1:
        h = 1.0                       # no defect asserted -> healthy
    elif f == 1:
        h = 0.5                       # both-1 contradiction
    else:
        h = 0.0                       # defect asserted, quality denied
    if key in det:
        h = min(h, 0.2)
    return h




def _polish(st: dict) -> float:
    """Continuous pixel-quality prior in [0,1] (clip-free, well-exposed, moderate contrast/saturation).
    Varies per-image even when the VLM answers identically -> de-spikes the rails, fills the middle."""
    bell = lambda v, c, w: max(0.0, 1.0 - abs(v - c) / w)
    clip_ok = max(0.0, 1.0 - (st["hi"] + st["lo"]) / 0.20)
    expo_ok = bell(st["luma"], 118.0, 95.0)            # mid exposure
    contrast_ok = bell(st["lstd"], 52.0, 46.0)         # punchy but not flat / not crushed
    sat_ok = bell(st["cf"], 45.0, 55.0)                # moderate colorfulness (gray & garish both off)
    return (clip_ok + expo_ok + contrast_ok + sat_ok) / 4


def _composite(ans1: dict, ans2: dict, is_portrait: bool, is_bw: bool, det: set):
    """Logically-gated facet composite in [0,1]. tone gated by clip/crush health, color gated by
    satclip/skintone health (can't be 'good color' if oversaturated). B&W color=neutral (no shrink)."""
    g = lambda k: _graded(ans2, k)
    gate = lambda h: _GATE_FLOOR + (1.0 - _GATE_FLOOR) * h
    tone_h = (_vhealth(ans1, "HILIGHTCLIP", det) + _vhealth(ans1, "SHADOWCRUSH", det)) / 2
    tone = (g("CLEAN") + g("CONTRAST")) / 2 * gate(tone_h)
    if is_bw:
        color = _NEUTRAL                                  # neutral, NOT skipped -> no fraction inflation
    else:
        ch = [_vhealth(ans1, "SATCLIP", det)] + ([_vhealth(ans1, "SKINTONE", det)] if is_portrait else [])
        color = (g("WBQUALITY") + g("MEMORYCOLOR") + g("VIVID")) / 3 * gate(sum(ch) / len(ch))
    pop = g("SUBJECTPOP") if is_portrait else _NEUTRAL
    mood, overall = g("MOOD"), g("BETTER")
    comp = (_W["overall"] * overall + _W["tone"] * tone + _W["color"] * color
            + _W["mood"] * mood + _W["pop"] * pop)
    facets = {k: round(v, 3) for k, v in
              {"overall": overall, "tone": tone, "color": color, "mood": mood, "pop": pop}.items()}
    return comp, facets


def qa_score_pair(suri: str, after_path: str, is_portrait: bool, src_cf: float,
                  src_obj=(None, None)) -> dict:
    try:
        return _qa_score_pair(suri, after_path, is_portrait, src_cf, src_obj)
    except Exception as e:  # noqa: BLE001 - one corrupt render must not kill the batch
        return {"reliable": False, "why": f"exception:{type(e).__name__}", "veto": False,
                "q": round(_NEUTRAL - _DED_CONTROL, 3), "merit_score": 0.0, "merit_hits": []}


def _qa_score_pair(suri: str, after_path: str, is_portrait: bool, src_cf: float,
                   src_obj=(None, None)) -> dict:
    """2-phase graded QA. phase-1 veto dims + controls; phase-2 merit dims on survivors. Score = a
    logically-gated facet composite (gradient), NOT a hit-fraction. Unreliable answers DEDUCT from a
    neutral baseline instead of becoming a hard zero/drop (mirrors source_qa's merit-soft policy)."""
    st = _stats(after_path)
    is_bw = st["cf"] < 8.0
    det = _det_flags(st, src_cf)
    auri = _uri(after_path)
    veto_keys = [k for k, t, c, p, f, r in _DIMS if t == "veto" and _active(k, is_portrait, is_bw)]
    merit_keys = [k for k, t, c, p, f, r in _DIMS if t == "merit" and _active(k, is_portrait, is_bw)]
    pol = _polish(st)
    after_obj = objscore.score(after_path)
    oq, oi = _objective(after_obj, src_obj, pol)   # Artimuse+Charm mixed IAA signals
    base = {"det": sorted(det), "is_bw": is_bw, "obj_q": round(oq, 3), "obj_impr": round(oi, 3),
            "iaa_mixed": _obj_detail(after_obj, "iaa_mixed"),
            "artimuse": _obj_detail(after_obj, "artimuse"),
            "charm": _obj_detail(after_obj, "charm")}
    # unreliable VLM answers => fall back to the OBJECTIVE score (minus a confidence discount) rather
    # than a flat deduction: still a real, separable quality estimate, just trusted less.
    def ded(why, d, veto=False):
        q = max(0.0, 0.62 * oq + 0.38 * oi - 0.5 * d)
        return {**base, "reliable": False, "veto": veto, "why": why, "merit_hits": [],
                "merit_score": round(oq, 3), "q": round(q, 3)}

    # phase 1: veto dims + controls
    ans1, ok1, why1 = _run_phase(suri, auri, _VETO_ITEMS, salt=1)
    if not ok1:
        return ded("p1_" + why1, _DED_PARSE)
    if (ans1.get("ANCHOR") != 1 or ans1.get("HONPOS") != 1 or ans1.get("HONNEG") != 0
            or ans1.get("GRAYTRAP") != (0 if is_bw else 1)):
        return ded("control", _DED_CONTROL)
    contra1 = _contra(ans1, veto_keys)
    vlm_veto = [k for k in veto_keys if ans1.get(k + "_F") == 0 and ans1.get(k + "_R") == 1]
    corroborated = [k for k in vlm_veto if k in det]
    # hard veto only when metric ∧ VLM agree, or >=2 VLM vetoes, or severe metric extreme. Graded by
    # severity (not a flat -3) so even rejects keep a gradient.
    if corroborated or len(vlm_veto) >= 2 or _det_extreme(st):
        sev = len(corroborated) + 0.5 * max(0, len(vlm_veto) - 2) + (1 if _det_extreme(st) else 0)
        q = round(max(-1.0, -0.25 - 0.25 * sev), 3)
        return {**base, "reliable": True, "veto": True, "vlm_veto": vlm_veto, "why": "veto",
                "merit_score": 0.0, "merit_hits": [], "q": q}

    # phase 2: merit dims (survivors only)
    ans2, ok2, why2 = _run_phase(suri, auri, _MERIT_ITEMS, salt=2)
    if not ok2:
        return ded("p2_" + why2, _DED_PARSE)
    contra2 = _contra(ans2, merit_keys)
    comp, facets = _composite(ans1, ans2, is_portrait, is_bw, det)
    merit_hits = [k for k in merit_keys if _graded(ans2, k) >= 1.0]
    contra_excess = max(0, contra1 + contra2 - _CONTRA_TOL)
    pen = (_MONO_DISCOUNT * is_bw + _PEN_SOFTVETO * len(vlm_veto) + _PEN_CONTRA * contra_excess)
    # IAA-led composite: VLM aesthetic judgment + mixed IAA quality + improvement-over-source.
    q = _W_VLM * comp + _W_OQUAL * oq + _W_OIMPR * oi - pen
    q = round(max(-1.0, min(1.0, q)), 3)
    return {**base, "reliable": contra_excess == 0, "veto": False, "vlm_veto": vlm_veto,
            "facets": facets, "polish": round(pol, 3), "merit_score": round(comp, 3),
            "merit_hits": merit_hits, "soft_veto": bool(vlm_veto), "q": q,
            "why": "" if contra_excess == 0 else "contra"}


# --------------------------------------------------------------------------- #
# Comparative QA: judge the SAME source's renders pairwise (VLMs are far more reliable at "A vs B"
# than at absolute binary scoring), with the source itself as a fixed anchor. Bradley-Terry turns the
# win matrix into a continuous strength; q = P(candidate beats source) = p_i/(p_i+p_src) — objective,
# source-anchored, no ties, genuinely separable. det-extreme defects are vetoed before the tournament.
# --------------------------------------------------------------------------- #
_CMP_SYS = (
    "你是资深商业修图评审。下面是同一张照片的两个版本：【图A】和【图B】。\n"
    "判断哪一张作为最终成片更好——更自然耐看、色调更讨喜、明暗层次与通透度更好，且没有"
    "过曝死白/死黑、过饱和溢色、肤色失真、廉价滤镜感等缺陷。不要因为单纯更亮/更艳就判它好。\n"
    "只输出一个字符：A（图A更好）、B（图B更好）、或 T（确实难分伯仲）。不要解释。"
)


def _compare(uri_a: str, uri_b: str) -> Optional[float]:
    """Pairwise: returns A's score in {1.0 win, 0.5 tie, 0.0 loss}, or None on call
    failure. A failed call must NOT read as a tie: during a vLLM outage every pair
    would silently become 0.5, BT strengths collapse to 1.0, and all candidates get
    q~0.5 with reliable=True — garbage SFT with no error signal anywhere."""
    out = _call(uri_a, uri_b, _CMP_SYS, temp=0.0)
    if not out:
        return None
    for ch in out.strip().upper():
        if ch == "A":
            return 1.0
        if ch == "B":
            return 0.0
        if ch == "T":
            return 0.5
    return 0.5


def _bt(items: list, results: list, iters: int = 300) -> dict:
    """Bradley-Terry strengths from results=[(a,b,score_a)]. +0.5 prior win vs a strength-1 phantom
    regularizes undefeated/winless items toward the field mean (=1)."""
    wins = {i: 0.5 for i in items}
    n: dict = {}
    for a, b, sa in results:
        wins[a] += sa; wins[b] += (1 - sa)
        n[(a, b)] = n.get((a, b), 0) + 1; n[(b, a)] = n.get((b, a), 0) + 1
    p = {i: 1.0 for i in items}
    for _ in range(iters):
        nv = {}
        for i in items:
            denom = 1.0 / (p[i] + 1.0)                       # phantom anchor: 1 game vs strength 1
            for j in items:
                if j != i and n.get((i, j)):
                    denom += n[(i, j)] / (p[i] + p[j])
            nv[i] = wins[i] / denom if denom > 0 else p[i]
        m = sum(nv.values()) / len(nv)
        p = {i: max(1e-6, v / m) for i, v in nv.items()}
    return p


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
    print(f"n_dims={len(_DIMS)} phase1_codes={len(_VETO_ITEMS)} phase2_codes={len(_MERIT_ITEMS)}  qa._smoke OK")


if __name__ == "__main__":
    _smoke()
