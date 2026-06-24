"""图像 QA 运行器（流程1 IMQ + 流程2 AES 的真实 vLLM 调用 + 清洗）。

把 config 的反作弊 POS_MAP/system prompt 渲染成模型只见两位位置码的提示，单次多模态
调用 35B(thinking off, temp 0.1)，输出位置码串，交 qa_clean 的纯规则清洗器裁定。
preset 流程在 preset_qa.py（需真实 LR 渲染对）。
"""
from __future__ import annotations

import base64
import io
import threading
from typing import Optional

from . import config as C
from . import qa_clean as Q

_LOCAL = threading.local()


def _session():
    if not hasattr(_LOCAL, "s"):
        import requests
        _LOCAL.s = requests.Session()
    return _LOCAL.s


def img_data_uri(path: str, longedge: Optional[int] = None) -> str:
    from PIL import Image, ImageOps
    Image.MAX_IMAGE_PIXELS = None
    longedge = longedge or C.VLLM_IMAGE_LONGEDGE
    im = Image.open(path)
    im.load()
    # 送审图始终用 EXIF 旋正后的显示朝向(与 musiq 等一致)；HAS_EXIF_FIXED 仅管 width/height 真值列
    im = ImageOps.exif_transpose(im) or im
    im = im.convert("RGB")
    w, h = im.size
    if max(w, h) > longedge:
        s = longedge / max(w, h)
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def build_prompt(kind: str, asset: dict) -> tuple:
    """返回 (prompt_text, pos_map)。kind∈{'IMQ','aes'}。"""
    if kind == "IMQ":
        pos_map = C.imq_pos_map(bool(asset.get("is_portrait_pool")))
        tmpl, qtext = C.IMQ_SYSTEM_PROMPT, C.IMQ_QTEXT
    else:
        mff = asset.get("max_face_frac")
        has_face = (mff is not None and mff > C.AES_FACE_MIN)
        pos_map = C.aes_pos_map(has_face)
        tmpl, qtext = C.AES_SYSTEM_PROMPT, C.AES_QTEXT
    items = C.render_items_block(pos_map, qtext)
    return tmpl.format(items=items), pos_map


def call_vllm(data_uri: str, prompt: str, temperature: float = 0.1,
              max_tokens: int = 220, max_retries: int = 3) -> str:
    payload = {
        "model": C.VLLM_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_uri}},
            {"type": "text", "text": prompt},
        ]}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if not C.VLLM_ENABLE_THINKING:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    headers = {"Authorization": f"Bearer {C.VLLM_API_KEY}", "X-vgate-class": "qa-judge"}
    last = ""
    for _ in range(max_retries):
        try:
            r = _session().post(C.VLLM_BASE_URL + "/chat/completions",
                                json=payload, headers=headers, timeout=120)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            last = str(e)
    raise RuntimeError(f"vllm call failed: {last}")


def run_imq(asset: dict, data_uri: Optional[str] = None) -> dict:
    """跑 IMQ；不可信按 IMQ_REASK_MAX 重问一次(重洗题序由 reask 时再做)。"""
    uri = data_uri or img_data_uri(asset["path"])
    prompt, _ = build_prompt("IMQ", asset)
    raw = call_vllm(uri, prompt, temperature=0.1)
    out = Q.clean_imq(raw, asset)
    out["reask_count"] = 0
    if not out["reliable"] and C.IMQ_REASK_MAX >= 1:
        raw2 = call_vllm(uri, prompt, temperature=0.0)
        out2 = Q.clean_imq(raw2, asset)
        out2["reask_count"] = 1
        out2["raw_first"] = raw
        out = out2
    return out


def run_aes(asset: dict, data_uri: Optional[str] = None) -> dict:
    uri = data_uri or img_data_uri(asset["path"])
    prompt, _ = build_prompt("aes", asset)
    raw = call_vllm(uri, prompt, temperature=C.AES_TEMPERATURE)
    out = Q.clean_aes(raw, asset)
    out["reask_count"] = 0
    if not out["reliable"] and C.AES_RETRY >= 1:
        raw2 = call_vllm(uri, prompt, temperature=0.0)
        out2 = Q.clean_aes(raw2, asset)
        out2["reask_count"] = 1
        out2["raw_first"] = raw
        out = out2
    return out
