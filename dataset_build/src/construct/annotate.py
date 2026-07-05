"""construct.annotate — winner 样本的 VLM instruct/reasoning 标注（单次合并调用 + 防泄露）。

接线位置：tier.build 选出 winner **之后**（pair 选定先于 instruction 生成，顺序不变），
对每条 SFT 记录调用一次 annotate_winner，把模板 instruction/reasoning 替换为 35B 看图
写出的版本；vLLM 不可用/防泄露 guard 屡次命中时抛 AnnotateError，tier 回退到无泄露模板。

单次合并调用：一次 vLLM 请求同时产出 reasoning + instruction（合并 vlm_clean.gen_instruction
与 reason_params 的目标）。模型看到【原图, 成片】两张图（<= limit_mm_per_prompt_image=2），
输出一个 JSON：{"reasoning": ..., "instruction_long": ..., "instruction_short": ...}。

防泄露规则（prompt 层 + 代码 guard 层，按 task_type）：
  style   : instruction 可含风格名 vlm_name（风格名本来就是任务输入）；reasoning 可用 vlm_function。
  auto/param: prompt 不注入 preset 名/vlm_name/vlm_caption；模型只看图逆向编辑意图写用户口吻
            指令；vlm_function 仅作为 reasoning 的措辞参考，风格名在全部文本中被 guard 禁止。
  local   : instruction 只描述区域 + 用户意图层面的诉求；由 GT local_params 翻译来的方向词
            （提亮/压暗/加强对比…）在 instruction 中被 guard 禁止（reasoning 里允许）。
  degrade : degrade_info 非空（退化链，未来接入）。instruction = 用户抱怨式，从图观察；
            reasoning 可引用退化类型与修复方向，但 instruction 禁止具体参数值/参数名。
  全任务  : vlm_caption 的 Δ 技术量（ΔL/Δa*/Δb*/ΔE…）与 caption 长片段在**任何**训练文本中
            都被 guard 禁止 —— 不只靠 prompt，返回前用代码断言过滤，命中即带反馈重试，
            重试耗尽则抛错降级模板。

调用预算打点：STATS 记录 annotate/verify 实际调用次数，冒烟脚本据此打印单样本口径
（目标：annotate ~1 次/winner + verify 条件带内 ~0.15 次/winner）。
"""
from __future__ import annotations

import json
import re
import threading
from typing import List, Optional

import requests

from dataset_build.source_qa import config
from dataset_build.vlm_clean import loads_lenient

# --------------------------------------------------------------------------- #
# 配置（dataset_build/config.yaml 的 qa.vlm_annotate / qa.verify_band 段）
# --------------------------------------------------------------------------- #
_CFG_PATH = None   # 懒加载 + 可注入（测试用）
_CFG = None
_CFG_LOCK = threading.Lock()

_DEFAULTS = {
    "enabled": True,          # qa.vlm_annotate.enabled
    "style_frac": 0.5,        # global 样本里 style 任务（instruction 点名风格）的比例
    "max_leak_retries": 2,    # 防泄露 guard 命中后的带反馈重试次数
    "max_fail_streak": 5,     # vLLM 连续失败 N 次后熔断，本进程回退模板
    "max_tokens": 900,
    "verify_band": (0.45, 0.70),
}


def _load_cfg() -> dict:
    """读 dataset_build/config.yaml 的 qa 段一次；缺失/读失败一律用默认值（不阻塞流水线）。"""
    global _CFG
    with _CFG_LOCK:
        if _CFG is not None:
            return _CFG
        cfg = dict(_DEFAULTS)
        try:
            import os
            import yaml
            path = _CFG_PATH or os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "config.yaml")
            qa = (yaml.safe_load(open(path)) or {}).get("qa") or {}
            va = qa.get("vlm_annotate") or {}
            for k in ("enabled", "style_frac", "max_leak_retries", "max_fail_streak", "max_tokens"):
                if k in va:
                    cfg[k] = va[k]
            band = qa.get("verify_band")
            if isinstance(band, (list, tuple)) and len(band) == 2:
                cfg["verify_band"] = (float(band[0]), float(band[1]))
        except Exception:  # noqa: BLE001 - 配置读不到就用默认，宁可朴素不可中断
            pass
        _CFG = cfg
        return _CFG


def enabled() -> bool:
    return bool(_load_cfg()["enabled"])


def style_frac() -> float:
    return float(_load_cfg()["style_frac"])


def verify_band() -> tuple:
    return tuple(_load_cfg()["verify_band"])


# --------------------------------------------------------------------------- #
# 调用预算打点 + vLLM 熔断
# --------------------------------------------------------------------------- #
STATS = {"winners": 0, "annotate_calls": 0, "annotate_ok": 0, "annotate_fallback": 0,
         "leak_retries": 0, "verify_calls": 0}
_STATS_LOCK = threading.Lock()
_FAIL_STREAK = 0          # 连续 transport 失败次数（成功清零；泄露重试不计入）


def _bump(key: str, n: int = 1) -> None:
    with _STATS_LOCK:
        STATS[key] = STATS.get(key, 0) + n


def stats() -> dict:
    with _STATS_LOCK:
        s = dict(STATS)
    w = max(1, s.get("winners", 0))
    s["annotate_calls_per_winner"] = round(s["annotate_calls"] / w, 3)
    s["verify_calls_per_winner"] = round(s["verify_calls"] / w, 3)
    return s


class AnnotateError(RuntimeError):
    """标注失败（vLLM 不可用 / 解析失败 / 防泄露 guard 重试耗尽）→ tier 回退无泄露模板。"""


def _circuit_open() -> bool:
    return _FAIL_STREAK >= int(_load_cfg()["max_fail_streak"])


# --------------------------------------------------------------------------- #
# 防泄露 guard（代码层，返回前断言过滤 —— 不只靠 prompt）
# --------------------------------------------------------------------------- #
# vlm_caption 的 Δ 技术量指纹：Δb*<-5、ΔL、ΔE、Δa*>0、"Delta E" 等。任何训练文本禁止。
_DELTA_RE = re.compile(r"[Δ∆]\s*[ELabCHelabch]?\*?|\bDelta\s*[ELab]|\b[Lab]\*\s*[<>=+-]", re.I)
# Lightroom 参数名（instruction 不允许出现参数级术语；reasoning 用自然语言即可，一并禁止内部键名）
_LR_KEY_RE = re.compile(
    r"\b(Exposure2012|Contrast2012|Highlights2012|Shadows2012|Whites2012|Blacks2012|"
    r"IncrementalTemperature|Temperature|Tint|Vibrance|Saturation|Clarity2012|"
    r"Local[A-Za-z]+|HueAdjustment[A-Za-z]+|SaturationAdjustment[A-Za-z]+|"
    r"LuminanceAdjustment[A-Za-z]+)\b")
# 具体参数值（degrade 任务的 instruction 禁止）：+30、-0.5、2 档、1.5EV…
_NUM_VAL_RE = re.compile(r"[+-]\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?\s*(?:档|stops?|ev)\b", re.I)
# local 任务 instruction 的通用技术方向句式（不止 GT 词表）：「提亮/压暗/增加清晰度」这类
# 动词+参数量的指令 = 把答案方向写进指令（冒烟发现 GT 未含曝光项时"提亮"会漏网）。
# 用户口吻的观察（"太暗/有点闷/更通透"）不受限。
_LOCAL_DIR_RE = re.compile(
    r"[提调加]亮|[压调]暗|加深|"
    r"(?:加强|增加|提高|降低|加大|减小|增|提|降)\s*(?:对比|饱和|清晰|锐)|"
    r"[收压降]高光|[提拉]阴影|锐化|去雾")

# GT local_params -> 方向词（与 tier._local_edit_phrase 同一映射）。local 任务的
# instruction 若出现由 GT 翻译来的这些词 = 指令即答案，guard 禁止。
_DIR_WORDS = [
    ("LocalExposure2012", lambda v: v > 0.05, ("提亮", "调亮", "加亮")),
    ("LocalExposure2012", lambda v: v < -0.05, ("压暗", "调暗", "加深")),
    ("LocalContrast2012", lambda v: v > 5, ("加强对比", "增加对比", "提高对比")),
    ("LocalSaturation", lambda v: v > 5, ("增艳", "提高饱和", "增加饱和")),
    ("LocalSaturation", lambda v: v < -5, ("降饱和", "降低饱和", "去饱和")),
    ("LocalHighlights2012", lambda v: v < -5, ("收高光", "压高光", "降高光")),
    ("LocalShadows2012", lambda v: v > 5, ("提阴影", "拉阴影", "提亮阴影")),
    ("LocalClarity2012", lambda v: v > 5, ("增清晰", "提高清晰", "加清晰")),
]


def gt_direction_words(local_params: Optional[dict]) -> List[str]:
    """由 GT local_params 推出的方向词表（local 任务 instruction 的禁用词）。"""
    lp = local_params or {}
    out: List[str] = []
    for key, cond, words in _DIR_WORDS:
        v = lp.get(key, 0)
        try:
            if cond(float(v)):
                out.extend(words)
        except (TypeError, ValueError):
            continue
    return out


def _caption_fragments(caption: str) -> List[str]:
    """vlm_caption 切成 >=8 字的短句片段；任一片段原样出现在输出 = caption 泄露。"""
    if not caption:
        return []
    frags = re.split(r"[，。；、,;()（）\s]+", caption)
    return [f for f in frags if len(f) >= 8]


def find_leaks(ann: dict, task_type: str, preset_meta: Optional[dict],
               banned_dirs: Optional[List[str]] = None) -> List[str]:
    """返回违规列表（空 = 干净）。instruction = long+short；reasoning 单独一档。"""
    meta = preset_meta or {}
    name = (meta.get("vlm_name") or "").strip()
    caption = meta.get("vlm_caption") or ""
    instr = f'{ann.get("instruction_long", "")}\n{ann.get("instruction_short", "")}'
    all_text = f'{instr}\n{ann.get("reasoning", "")}'
    v: List[str] = []
    # 全任务：Δ 技术量 / caption 片段 / LR 内部参数键名 —— 任何训练文本禁止
    if _DELTA_RE.search(all_text):
        v.append("delta_metric")
    for frag in _caption_fragments(caption):
        if frag in all_text:
            v.append(f"caption_fragment:{frag[:12]}")
    if _LR_KEY_RE.search(all_text):
        v.append("lr_param_key")
    # 风格名：auto/param/degrade 全文本禁止；local 仅 instruction 禁止；style 允许且必须出现
    if name:
        if task_type in ("auto", "param", "degrade") and name in all_text:
            v.append("style_name")
        elif task_type == "local" and name in instr:
            v.append("style_name_in_instruction")
        elif task_type == "style" and name not in ann.get("instruction_long", ""):
            v.append("style_name_missing")   # style 任务的 instruction 必须点名风格（任务输入）
    # local：GT 方向词 + 通用技术方向句式禁止出现在 instruction（reasoning 允许）
    for w in (banned_dirs or []):
        if w in instr:
            v.append(f"gt_direction:{w}")
    if task_type == "local":
        m = _LOCAL_DIR_RE.search(instr)
        if m:
            v.append(f"tech_direction:{m.group(0)}")
    # degrade：instruction 禁止具体参数值
    if task_type == "degrade" and _NUM_VAL_RE.search(instr):
        v.append("param_value")
    return v


# --------------------------------------------------------------------------- #
# prompt（单次合并调用：reasoning + instruction 同产）
# --------------------------------------------------------------------------- #
_SYS = (
    "你是资深商业修图师，同时为修图训练数据写标注。你会看到两张图：第一张是【原图】，"
    "第二张是对原图完成某次修图后的【成片】。\n"
    "请产出两样东西：\n"
    "1. reasoning：修图师接到用户请求后的思考过程——先指出原图上可见的画面证据"
    "（主体、光线、色彩、氛围各自的现状），再解释这次修图往哪个方向调整、成片达到了什么效果。"
    "必须引用图上看得到的证据，泛泛之词（如『提升了质感』『更有高级感』而无画面依据）不合格。\n"
    "2. instruction：模拟真实用户拿着原图向修图师提出的修图请求（自然中文、用户口吻、祈使句），"
    "该请求的理想结果正是第二张成片——请求内容必须与两张图的实际视觉差异一致。\n"
    "只输出一个 JSON 对象，不要输出 JSON 以外的任何文字：\n"
    '{"reasoning": "<3-5句中文>", "instruction_long": "<2-3句用户口吻请求>", '
    '"instruction_short": "<不超过15个字的简短请求>"}'
)


def _task_clause(task_type: str, preset_meta: Optional[dict], degrade_info: Optional[dict],
                 local_region: Optional[str]) -> str:
    meta = preset_meta or {}
    name = (meta.get("vlm_name") or "").strip()
    fn = (meta.get("vlm_function") or "").strip()
    if task_type == "style":
        region = f"，且只应用于画面中的【{local_region}】区域" if local_region else ""
        clause = (
            f"任务类型：风格化调色{region}。用户明确想要「{name or '这种'}」风格，"
            f"instruction 必须点名风格名称「{name}」（例如『请把这张调成「{name}」的风格』），"
            "并可补充 1-2 句与画面相符的期望。\n"
            "reasoning 里解释：为什么这张照片适合该风格、成片上哪些可见变化体现了它。")
        if fn:
            clause += f"\n该风格的适用场景参考（可为 reasoning 提供措辞启发）：{fn}"
        return clause
    if task_type in ("auto", "param"):
        clause = (
            "任务类型：自动优化。用户不知道任何风格/预设/滤镜的名字——instruction 里严禁出现"
            "风格名、预设名、滤镜名、任何 Lightroom 参数名或数值；只用普通用户的话描述想要的"
            "效果（例如『帮我把这张修得更通透、氛围更好』），且描述必须与成片的实际视觉变化一致。\n"
            "reasoning 用你自己的话、基于两张图的可见差异解释这次修图为什么这样调"
            "（同样不得出现任何风格名/预设名）。")
        if fn:
            clause += ("\n背景参考（仅可作为 reasoning 的措辞启发，"
                       f"严禁原样写入 instruction）：这类处理{fn}")
        return clause
    if task_type == "local":
        return (
            f"任务类型：局部修图。这次修图只作用于画面中的【{local_region or '目标'}】区域，"
            "画面其余部分基本不变。\n"
            "instruction 只描述该区域与用户意图层面的诉求——由你对比两张图判断该区域的问题"
            "（例如『这个区域看起来有点闷，帮我处理一下』『希望人物从背景里更突出』）。"
            "instruction 里严禁出现技术性的调整方向指令（如提亮/压暗/加对比/增饱和/收高光等）"
            "和任何风格名。\n"
            "reasoning 里才允许写方向性的调整描述：该区域朝什么方向调、为什么这样调。")
    if task_type == "degrade":
        kinds = ""
        if isinstance(degrade_info, dict):
            kinds = "、".join(str(k) for k in (degrade_info.get("kinds")
                                               or degrade_info.get("aspects") or [])) \
                or str(degrade_info.get("kind") or "")
        return (
            "任务类型：坏图修复。第一张原图存在画质/色彩问题，第二张是修复后的成片。\n"
            "instruction 写成用户抱怨式（例如『这张照片发灰又偏色，帮我修一下』）——抱怨的内容"
            "必须是第一张图上肉眼可见的问题，严禁出现任何具体参数值、参数名或技术术语。\n"
            f"reasoning 可以引用退化类型（{kinds or '按图判断'}）与对应的修复方向。")
    raise AnnotateError(f"unknown task_type: {task_type}")


# --------------------------------------------------------------------------- #
# vLLM 调用（HTTP；复用 qa 的图片编码 + 全局 admission gate；解析用 vlm_clean.loads_lenient）
# --------------------------------------------------------------------------- #
def _post(user_text: str, images: List[str], retries: int = 3) -> str:
    """一次 chat/completions。transport 级重试 + 429 退避；耗尽抛 AnnotateError。"""
    from . import qa as _qa   # 局部 import 避免环
    import time
    content = [{"type": "image_url", "image_url": {"url": _qa._uri(p)}} for p in images]
    content.append({"type": "text", "text": user_text})
    payload = {"model": config.VLLM_MODEL, "max_tokens": int(_load_cfg()["max_tokens"]),
               "temperature": 0.3,
               "messages": [{"role": "system", "content": _SYS},
                            {"role": "user", "content": content}]}
    if not config.VLLM_ENABLE_THINKING:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    headers = {"Authorization": f"Bearer {config.VLLM_API_KEY}",
               "X-vgate-class": "build-annotate"}
    last = "empty"
    for attempt in range(retries):
        try:
            with _qa._VLLM_GATE:
                r = requests.post(config.VLLM_BASE_URL + "/chat/completions", json=payload,
                                  headers=headers, timeout=180)
            if r.status_code == 429 and attempt < retries - 1:
                time.sleep(float(r.headers.get("Retry-After", 2 * (attempt + 1))))
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"] or ""
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {str(e)[:120]}"
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
    raise AnnotateError(f"vllm_unavailable: {last}")


def _normalize(raw: str) -> Optional[dict]:
    d = loads_lenient(re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S))
    if not isinstance(d, dict):
        return None
    out = {k: str(d.get(k, "")).strip()
           for k in ("reasoning", "instruction_long", "instruction_short")}
    if not (out["reasoning"] and out["instruction_long"]):
        return None
    if not out["instruction_short"]:
        out["instruction_short"] = out["instruction_long"][:15]
    return out


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def annotate_winner(image_path: str, task_type: str, preset_meta: Optional[dict],
                    degrade_info: Optional[dict] = None, local_region: Optional[str] = None,
                    local_params: Optional[dict] = None,
                    source_path: Optional[str] = None) -> dict:
    """winner 样本 -> {"instruction_long", "instruction_short", "reasoning"}。

    image_path = winner 成片；source_path = 原图（给出时模型看 before+after 双图，单次调用）。
    失败（vLLM 不可用 / 熔断 / 防泄露重试耗尽）抛 AnnotateError，调用方回退无泄露模板。
    """
    global _FAIL_STREAK
    if degrade_info:
        task_type = "degrade"
    if task_type == "style" and not ((preset_meta or {}).get("vlm_name") or "").strip():
        task_type = "auto"   # 没有可点名的风格名（caption 缺失）-> 按自动优化处理，避免空「」
    if not enabled():
        raise AnnotateError("disabled")
    if _circuit_open():
        raise AnnotateError(f"circuit_open (fail_streak={_FAIL_STREAK})")

    banned_dirs = gt_direction_words(local_params) if task_type == "local" else []
    clause = _task_clause(task_type, preset_meta, degrade_info, local_region)
    images = [p for p in (source_path, image_path) if p]
    base_text = ("第一张是【原图】，第二张是【成片】。\n" if len(images) == 2
                 else "这张图是修图后的【成片】（原图未提供，请从成片逆推）。\n") + clause

    feedback = ""
    max_leak = int(_load_cfg()["max_leak_retries"])
    for attempt in range(1 + max_leak):
        _bump("annotate_calls")
        try:
            raw = _post(base_text + feedback, images)
        except AnnotateError:
            with _STATS_LOCK:
                _FAIL_STREAK += 1               # transport 失败计入熔断
            raise
        with _STATS_LOCK:
            _FAIL_STREAK = 0                    # 服务在线：熔断计数清零
        ann = _normalize(raw)
        if ann is None:
            leaks = ["parse_failed"]
        else:
            leaks = find_leaks(ann, task_type, preset_meta, banned_dirs)
        if not leaks:
            _bump("annotate_ok")
            return ann
        if attempt < max_leak:
            _bump("leak_retries")
            banned_terms = [t for t in ([((preset_meta or {}).get("vlm_name") or "")]
                                        if task_type != "style" else []) + banned_dirs if t]
            feedback = ("\n\n【重写要求】上一次输出违反了标注规范（" + "、".join(leaks[:4]) + "）。"
                        "请重新输出同样的 JSON，务必：不出现技术量/参数名/Δ 指标；"
                        + (f"以下词语严禁出现在 instruction 中：{'、'.join(banned_terms)}；"
                           if banned_terms else "")
                        + ("instruction 不得出现提亮/压暗/加对比/增饱和/锐化等任何技术方向词，"
                           "只写区域观感与用户诉求（方向性描述放 reasoning）；"
                           if task_type == "local" else "")
                        + ("style 任务的 instruction 必须点名风格名称；" if task_type == "style" else "")
                        + "instruction 保持用户口吻。")
    raise AnnotateError(f"leak_guard_exhausted: {leaks}")


# --------------------------------------------------------------------------- #
# verify 条件化（q ∈ qa.verify_band 的 winner 才走 vlm_clean.verify；结果只进 qa 字段）
# --------------------------------------------------------------------------- #
_CLEANER = None
_CLEANER_LOCK = threading.Lock()


def _cleaner():
    global _CLEANER
    with _CLEANER_LOCK:
        if _CLEANER is None:
            from dataset_build.vlm_clean import QwenVLCleaner
            _CLEANER = QwenVLCleaner(
                base_url=config.VLLM_BASE_URL, model=config.VLLM_MODEL,
                max_tokens=512, image_longedge=config.VLLM_IMAGE_LONGEDGE,
                vgate_class="build-annotate")
        return _CLEANER


def should_verify(q: Optional[float]) -> bool:
    if q is None:
        return False
    lo, hi = verify_band()
    return lo <= float(q) <= hi


def verify_winner(before_path: str, after_path: str, instruction: str) -> Optional[dict]:
    """before+after 双图 QA（vlm_clean.verify）。失败返回 None；结果进 qa 字段，不进训练文本。"""
    _bump("verify_calls")
    try:
        v = _cleaner().verify(before_path, after_path, {"_instruction": instruction})
        return v or None
    except Exception:  # noqa: BLE001 - verify 是旁路信号，失败不阻塞 tier
        return None


# --------------------------------------------------------------------------- #
# 冒烟：python -m construct.annotate <after.jpg> [source.jpg]
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    after = sys.argv[1]
    src = sys.argv[2] if len(sys.argv) > 2 else None
    meta = {"vlm_name": "青绿冷调复古",
            "vlm_caption": "中性灰与肤色被显著推冷并注入绿色调（Δb*<-5, Δa*<0）",
            "vlm_function": "适合需要营造清冷、疏离或独特胶片质感的风景与静物摄影。"}
    for tt in ("style", "auto"):
        out = annotate_winner(after, tt, meta, source_path=src)
        print(f"--- {tt} ---\n" + json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(stats(), ensure_ascii=False))
