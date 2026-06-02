"""
dataset_build/vlm_clean.py
==========================
Offline Qwen3-VL-8B-Instruct cleaner / annotator over the *served* vLLM
OpenAI-compatible endpoint.

This module is the construction-time VLM layer for the VeraRetouch Direction-A,
RECIPE-BASED, region-local-heavy 1,000,000-sample dataset build. It talks to the
endpoint over **HTTP only** (no local model weights, no server launch here — the
main process launches the vLLM container per docs/plan/dataset/probe/probe_vllm_clean.md §1).

What it provides
----------------
  * ``QwenVLCleaner`` — concrete implementation of ``contracts.Cleaner`` plus the
    low-level chat plumbing (base64 data-URI images, JSON-mode, retry/backoff).
  * The 4 prompt templates (probe_vllm_clean §3):
        (a) gen_instruction       — instruction generation from an image
        (b) reason_params         — <think> reasoning + <answer> VeraRetouch params
        (c) verify                — before/after + MMArt processed-jpg QA check
        (d) tag_scene_region      — scene + SAM3 region-concept tagging
  * Parsers normalizing any Lightroom-style ``<answer>`` (json OR single-quoted
    python-dict) into the canonical VeraRetouch param dict
    ``{key: {"value": <raw_LR_number>}}`` keeping ONLY ``contracts.PARAM_KEYS``.
  * ``quality_gate(sample) -> (accept, scores)`` — the config-driven accept/reject
    heuristic over the QA scores (mirrors streams.QAGate but operates on a Sample
    and returns the computed ``QualityScores`` so callers can persist them).

Grounding (read before editing):
  - docs/plan/dataset/probe/probe_vllm_clean.md   (serve cmd, 4 templates, gotchas)
  - dataset_build/contracts.py                    (Cleaner ABC, Sample, PARAM_KEYS,
                                                    QualityScores schema)
  - dataset_build/config.yaml                     (vllm:, qa:, models:)
  - data/infer_dataset.py get_organized_dict L229-296 (the 38-key 3-aspect schema)
  - /home/bc/datasets/MMArt-PPR10k/grpo_dataset.json   (the verbatim system persona)

CRITICAL caveats encoded here (MEMORY + probe §5):
  (a) MMArt before/processed/xmp may NOT correspond -> template (c) processed_ok gate;
      MMArt stale image paths are rebased to /home/bc/datasets/MMArt-PPR10k/global/<id>/.
  (b) param-mode is GLOBAL/uniform -> local-mask keys (MaskGroupBasedCorrections,
      Local*, CorrectionMasks) are STRIPPED from the renderer params but surfaced as a
      region hint (parse_answer returns them under ``local_hints``).
  (c) MMArt <answer> JSON is single-quoted python-dict -> ast.literal_eval fallback.
  (d) Lightroom-native answer keys (Temperature, RedHue, RedSaturation, ...) are
      ALIASED into the VeraRetouch PARAM_KEYS (IncrementalTemperature, HueAdjustmentRed,
      SaturationAdjustmentRed, ...) before bucketing.

NOTE: imports are intentionally light (stdlib + optional ``openai``/``tenacity``/
``yaml``). ``openai`` is only required when actually issuing a request, so this file
``py_compile``s and imports on any env even where the client lib is absent.
"""

from __future__ import annotations

import argparse
import ast
import base64
import json
import os
import random
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from dataset_build.contracts import (
    COLORMIXER_KEYS,
    COLORTEMP_KEYS,
    LIGHT_KEYS,
    PARAM_KEYS,
    Cleaner,
    QualityScores,
    Sample,
    StreamId,
)

# ---------------------------------------------------------------------------
# 0. Constants: the verbatim MMArt persona + the LR-native -> VeraRetouch alias map.
# ---------------------------------------------------------------------------

# The base persona is loaded verbatim from any grpo_dataset.json record's ["system"]
# field at runtime (see QwenVLCleaner._load_persona). This embedded copy is the
# fallback used if the grpo json is unavailable; it is functionally equivalent to the
# probed system string (probe_vllm_clean §3).
_FALLBACK_PERSONA = (
    "You are an expert in Adobe Lightroom. Your task is to translate natural language "
    "descriptions of image edits, provided in either English or Chinese, into a structured "
    "JSON format representing Lightroom parameter adjustments. This includes both global "
    "settings and local adjustments using masks.\n\n"
    "# Output Format Requirements & Key Rules\n"
    "1. Valid JSON Only.\n"
    "2. English Parameter Names: parameter names (keys) must remain the standard English "
    "Lightroom identifiers (e.g., Exposure2012, Temperature, MaskSubType). Do not translate keys.\n"
    "3. Modified Parameters Only: include only the parameters you are adjusting.\n\n"
    "The assistant first thinks about the reasoning process in the mind and then provides the "
    "user with the answer. The reasoning process and answer are enclosed within <think> </think> "
    "and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> "
    "<answer> answer here </answer>."
)

# Lightroom-native answer key  ->  VeraRetouch PARAM_KEYS canonical key.
# MMArt / Qwen emit LR-native names; VeraRetouch's get_organized_dict (L229-296) keys are
# the 2012/Incremental*/HueAdjustment*/SaturationAdjustment*/LuminanceAdjustment* family.
_HSL_BANDS = ("Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta")
_KEY_ALIASES: Dict[str, str] = {
    # Color & temperature: LR "Temperature"/"Tint" are absolute Kelvin in As-Shot WB;
    # VeraRetouch consumes the *incremental* relative units (the /100 LR slider units).
    "Temperature": "IncrementalTemperature",
    "Tint": "IncrementalTint",
    "IncrementalTemperature2012": "IncrementalTemperature",
    "IncrementalTint2012": "IncrementalTint",
    # Bare (process-2010) light keys -> 2012 equivalents.
    "Exposure": "Exposure2012",
    "Contrast": "Contrast2012",
    "Highlights": "Highlights2012",
    "Shadows": "Shadows2012",
    "Whites": "Whites2012",
    "Blacks": "Blacks2012",
}
# Per-band HSL: LR uses <Band>Hue / <Band>Saturation / <Band>Luminance.
for _b in _HSL_BANDS:
    _KEY_ALIASES[f"{_b}Hue"] = f"HueAdjustment{_b}"
    _KEY_ALIASES[f"{_b}Saturation"] = f"SaturationAdjustment{_b}"
    _KEY_ALIASES[f"{_b}Luminance"] = f"LuminanceAdjustment{_b}"

# Keys that signal a LOCAL/mask edit (param-mode is GLOBAL -> strip but keep as hint).
_LOCAL_KEY_RE = re.compile(
    r"^(Local|Mask|Correction)|MaskGroupBasedCorrections|CorrectionMasks|"
    r"MaskSubType|MaskSubCategoryID|CircularGradient|PaintBased|Gradient",
    re.IGNORECASE,
)

_PARAM_KEY_SET = set(PARAM_KEYS)
_VALID_SCENES = (
    "portrait", "landscape", "street", "food", "product",
    "wedding", "night", "architecture", "still_life", "any",
)

# The MMArt grpo json (for loading the verbatim persona + path-rebasing helpers).
_MMART_GRPO_JSON = "/home/bc/datasets/MMArt-PPR10k/grpo_dataset.json"
_MMART_GLOBAL_ROOT = "/home/bc/datasets/MMArt-PPR10k/global"


# ---------------------------------------------------------------------------
# 1. Tiny retry/backoff (use tenacity if installed, else a stdlib decorator).
# ---------------------------------------------------------------------------

def _retry(max_attempts: int = 3, base: float = 0.8, cap: float = 8.0):
    """Decorator: retry on any Exception with exponential backoff + jitter.

    Uses ``tenacity`` if importable (probe asks for it) so behaviour matches the
    rest of the build; otherwise falls back to a stdlib implementation so this
    module has no hard third-party retry dependency.
    """
    try:  # pragma: no cover - depends on env
        import tenacity  # type: ignore

        return tenacity.retry(
            stop=tenacity.stop_after_attempt(max_attempts),
            wait=tenacity.wait_random_exponential(multiplier=base, max=cap),
            reraise=True,
        )
    except Exception:
        def deco(fn: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                last: Optional[BaseException] = None
                for attempt in range(max_attempts):
                    try:
                        return fn(*args, **kwargs)
                    except Exception as exc:  # noqa: BLE001
                        last = exc
                        if attempt == max_attempts - 1:
                            break
                        sleep = min(cap, base * (2 ** attempt)) * (0.5 + random.random())
                        time.sleep(sleep)
                assert last is not None
                raise last
            return wrapper
        return deco


# ---------------------------------------------------------------------------
# 2. Parsing helpers (module-level so streams/pack can reuse them without a client).
# ---------------------------------------------------------------------------

def loads_lenient(text: str) -> Optional[dict]:
    """Parse a JSON-ish blob into a dict.

    Order: direct ``json.loads`` -> fenced ```json``` block -> first balanced
    ``{...}`` substring -> ``ast.literal_eval`` (handles MMArt single-quoted
    python-dict ``<answer>`` — probe §5 gotcha 3). Returns None on total failure.
    """
    if text is None:
        return None
    s = text.strip()
    if not s:
        return None

    # 1) direct
    try:
        d = json.loads(s)
        if isinstance(d, dict):
            return d
    except Exception:
        pass

    # 2) fenced ```json ... ```
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.S | re.I)
    candidates: List[str] = []
    if fence:
        candidates.append(fence.group(1))

    # 3) first balanced {...} substring
    start = s.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(s)):
            c = s[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(s[start:i + 1])
                    break

    for cand in candidates:
        try:
            d = json.loads(cand)
            if isinstance(d, dict):
                return d
        except Exception:
            pass
        # 4) ast literal_eval fallback (single-quoted python dict)
        try:
            d = ast.literal_eval(cand)
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    return None


def split_think_answer(text: str) -> Tuple[str, Optional[dict], dict]:
    """Split a ``<think>...</think><answer>...</answer>`` envelope.

    Returns ``(think_str, answer_param_dict_or_None, extras)`` where
    ``answer_param_dict`` is already normalized to the VeraRetouch
    ``{key:{"value":raw}}`` schema (PARAM_KEYS only) and ``extras`` carries
    ``{"local_hints": {...}, "dropped": [...]}`` from ``parse_answer``.

    Robust to: missing tags (treats whole string as the answer blob), and to the
    answer being a bare JSON / single-quoted dict.
    """
    think = ""
    m_think = re.search(r"<think>(.*?)</think>", text, flags=re.S | re.I)
    if m_think:
        think = m_think.group(1).strip()

    m_ans = re.search(r"<answer>(.*?)</answer>", text, flags=re.S | re.I)
    ans_blob = m_ans.group(1).strip() if m_ans else text
    raw = loads_lenient(ans_blob)
    if raw is None:
        return think, None, {"local_hints": {}, "dropped": []}
    params, extras = parse_answer(raw)
    return think, params, extras


def _coerce_number(v: Any) -> Optional[float]:
    """LR slider value -> float. Handles '+30', '30', 30, '-0.2', 'As Shot'->None."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().lstrip("+")
        try:
            return float(s)
        except ValueError:
            return None
    return None


def parse_answer(raw: dict) -> Tuple[Dict[str, Dict[str, float]], dict]:
    """Normalize a raw Lightroom-style answer dict into the VeraRetouch param dict.

    Steps:
      1. Alias LR-native keys -> VeraRetouch PARAM_KEYS (``_KEY_ALIASES``).
      2. Keep only keys in ``PARAM_KEYS`` with numeric values -> {"value": raw_LR}.
         (RAW Lightroom units; the renderer's get_organized_dict divides by 100.)
      3. Strip local-mask keys (param-mode is GLOBAL) but record them in
         ``extras["local_hints"]`` as a region hint for template (d) / masking.

    Returns ``(params, extras)``; ``params`` may be empty (a valid no-op edit).
    """
    params: Dict[str, Dict[str, float]] = {}
    local_hints: Dict[str, Any] = {}
    dropped: List[str] = []

    for k, v in raw.items():
        if not isinstance(k, str):
            continue
        # local / mask structures -> hint, never sent to the global renderer
        if _LOCAL_KEY_RE.match(k) or isinstance(v, (list, dict)):
            local_hints[k] = v
            continue
        canon = _KEY_ALIASES.get(k, k)
        if canon not in _PARAM_KEY_SET:
            dropped.append(k)
            continue
        num = _coerce_number(v)
        if num is None:
            dropped.append(k)
            continue
        # last-writer-wins if both alias and canonical appear
        params[canon] = {"value": num}

    return params, {"local_hints": local_hints, "dropped": dropped}


def aspect_buckets(params: Dict[str, Dict[str, float]]) -> Dict[str, List[str]]:
    """Group a param dict into the 3 Direction-A aspects (L / GC / SC).

    Mirrors the get_organized_dict bucketing (L229-296). Used by streams to set
    ``DegradeSpec.aspects`` / per-aspect magnitudes and by quality_gate sanity.
    """
    out: Dict[str, List[str]] = {"L": [], "GC": [], "SC": []}
    for k in params:
        if k in LIGHT_KEYS:
            out["L"].append(k)
        elif k in COLORTEMP_KEYS:
            out["GC"].append(k)
        elif k in COLORMIXER_KEYS:
            out["SC"].append(k)
    return out


def rebase_mmart_path(stale_path: str) -> str:
    """Rebase a stale MMArt image path to the real on-disk root (probe §5 gotcha 2).

    grpo_dataset.json references e.g.
      /home/bc/retouching/JarvisArt/datasets/MMArt-PPR10k/.../<id>/before.jpg
    which does NOT exist; the files live under
      /home/bc/datasets/MMArt-PPR10k/global/<id>/{before,processed}.jpg
    We keep the trailing ``<id>/<file>`` (last two path components) and re-root it.
    """
    p = Path(stale_path)
    parts = p.parts
    if len(parts) >= 2:
        tail = Path(parts[-2]) / parts[-1]
    else:
        tail = Path(p.name)
    return str(Path(_MMART_GLOBAL_ROOT) / tail)


# ---------------------------------------------------------------------------
# 3. Heuristic param-sanity (cheap, no VLM) — used by quality_gate.
# ---------------------------------------------------------------------------

# Plausible RAW LR-unit ranges per aspect family (sliders are ~[-100,100]; exposure
# in stops is small). Anything outside -> not param_sane.
_RANGE_LIGHT = (-100.0, 100.0)
_RANGE_EXPOSURE = (-5.0, 5.0)      # Exposure2012 is in stops, not /100 slider units
_RANGE_COLORTEMP = (-100.0, 100.0)
_RANGE_HSL = (-100.0, 100.0)


def heuristic_param_sane(params: Dict[str, Dict[str, float]]) -> bool:
    """Return True iff all param values sit in plausible raw-LR ranges."""
    if not params:
        return True
    for k, v in params.items():
        val = v.get("value") if isinstance(v, dict) else None
        if val is None:
            return False
        if k == "Exposure2012":
            lo, hi = _RANGE_EXPOSURE
        elif k in LIGHT_KEYS:
            lo, hi = _RANGE_LIGHT
        elif k in COLORTEMP_KEYS:
            lo, hi = _RANGE_COLORTEMP
        else:
            lo, hi = _RANGE_HSL
        if not (lo <= float(val) <= hi):
            return False
    return True


# ---------------------------------------------------------------------------
# 4. The cleaner client.
# ---------------------------------------------------------------------------

class QwenVLCleaner(Cleaner):
    """Offline Qwen3-VL-8B cleaner/annotator over the served vLLM endpoint.

    HTTP only; loads no weights. Construct from config.yaml (``from_config``) or
    explicit kwargs. ``base_url`` MUST point at the running vLLM container
    (main process launches it; this class never spawns it).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8001/v1",
        api_key: str = "EMPTY",
        model: str = "qwen3-vl-8b",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        timeout: float = 120.0,
        max_retries: int = 3,
        persona: Optional[str] = None,
        image_longedge: int = 0,
        image_encode_concurrency: int = 2,
        max_image_pixels: int = 80_000_000,
        image_cache_entries: int = 512,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.persona = persona or self._load_persona()
        # Downscale long-edge for images sent to the judge (0 = native). Instruction/
        # judge tasks don't need native res, and vision-encode/prefill of native-res
        # images (3-4 calls/sample) is the clean-stage bottleneck. Downscaling cuts
        # vision tokens ~quadratically with no data-content change.
        self.image_longedge = int(image_longedge or 0)
        self.max_image_pixels = int(max_image_pixels or 0)
        self.image_cache_entries = max(0, int(image_cache_entries or 0))
        self._image_uri_cache: "OrderedDict[Tuple[str, int, int, int, int], str]" = OrderedDict()
        self._image_cache_lock = threading.RLock()
        enc_n = max(1, int(image_encode_concurrency or 1))
        self._image_encode_sem = threading.BoundedSemaphore(enc_n)
        self._client = None  # lazy OpenAI client

    # ---- construction helpers -------------------------------------------
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "QwenVLCleaner":
        """Build from a parsed config.yaml dict (reads the ``vllm:`` + ``qa:`` blocks)."""
        v = config.get("vllm", {})
        qa = config.get("qa", {})
        return cls(
            base_url=v.get("base_url", "http://localhost:8001/v1"),
            api_key=v.get("api_key", "EMPTY"),
            model=v.get("served_model_name", "qwen3-vl-8b"),
            temperature=float(v.get("temperature", 0.2)),
            max_tokens=int(v.get("max_tokens", 2048)),
            max_retries=int(qa.get("max_retries", 3)),
            image_longedge=int(v.get("image_longedge", 0)),
            image_encode_concurrency=int(v.get("image_encode_concurrency", 2)),
            max_image_pixels=int(v.get("max_image_pixels", 80_000_000)),
            image_cache_entries=int(v.get("image_cache_entries", 512)),
        )

    @staticmethod
    def _load_persona(grpo_json: str = _MMART_GRPO_JSON) -> str:
        """Load the verbatim MMArt system persona from grpo_dataset.json record 0.

        Falls back to the embedded ``_FALLBACK_PERSONA`` if the file is missing
        (so this works on machines without the MMArt dataset mounted). Reads only
        the first record cheaply by scanning to the first ``"system"`` value.
        """
        try:
            with open(grpo_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and data:
                sys = data[0].get("system")
                if isinstance(sys, str) and sys.strip():
                    return sys.strip()
        except Exception:
            pass
        return _FALLBACK_PERSONA

    # ---- low-level chat plumbing ----------------------------------------
    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI  # lazy: not needed for import/parse-only use
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "openai client not installed; needed only to issue requests. "
                    "Run in an env with `openai` (e.g. the vllm env)."
                ) from exc
            self._client = OpenAI(
                base_url=self.base_url, api_key=self.api_key, timeout=self.timeout
            )
        return self._client

    def _img_to_data_uri(self, path: str) -> str:
        """Encode a local image file to a base64 ``data:`` URI.

        Images sent to the VLM are always encoded as bounded JPEG thumbnails.
        This is deliberate: some source pools contain print-resolution frame/PSD
        assets, and PIL's ``convert("RGB")`` on many such files concurrently can
        allocate tens of GB. ``thumbnail`` asks Pillow to reduce during decode
        where the format supports it, and the semaphore caps concurrent decodes.
        """
        longedge = self.image_longedge or 768
        max_pixels = self.max_image_pixels
        cache_key = self._image_cache_key(path, longedge, max_pixels)
        if cache_key is not None:
            cached = self._image_cache_get(cache_key)
            if cached is not None:
                return cached
        with self._image_encode_sem:
            try:
                import io
                from PIL import Image  # lazy; only needed when sending images

                with Image.open(path) as im:
                    w, h = im.size
                    if max_pixels and (w * h) > max_pixels:
                        raise ValueError(
                            f"image too large for VLM encode: {w}x{h} "
                            f"({w*h} px > {max_pixels}) path={path}"
                        )
                    im.thumbnail((longedge, longedge), Image.LANCZOS)
                    im = im.convert("RGB")
                    buf = io.BytesIO()
                    im.save(buf, format="JPEG", quality=90)
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                uri = f"data:image/jpeg;base64,{b64}"
                if cache_key is not None:
                    self._image_cache_put(cache_key, uri)
                return uri
            except Exception as exc:
                raise RuntimeError(f"failed to safely encode VLM image {path}: {exc}") from exc

    def _image_cache_key(
        self, path: str, longedge: int, max_pixels: int
    ) -> Optional[Tuple[str, int, int, int, int]]:
        if self.image_cache_entries <= 0:
            return None
        try:
            st = os.stat(path)
        except OSError:
            return None
        return (os.path.abspath(path), int(longedge), int(max_pixels), int(st.st_mtime_ns), int(st.st_size))

    def _image_cache_get(self, key: Tuple[str, int, int, int, int]) -> Optional[str]:
        with self._image_cache_lock:
            value = self._image_uri_cache.get(key)
            if value is not None:
                self._image_uri_cache.move_to_end(key)
            return value

    def _image_cache_put(self, key: Tuple[str, int, int, int, int], value: str) -> None:
        with self._image_cache_lock:
            self._image_uri_cache[key] = value
            self._image_uri_cache.move_to_end(key)
            while len(self._image_uri_cache) > self.image_cache_entries:
                self._image_uri_cache.popitem(last=False)

    def _chat(
        self,
        system: str,
        user_text: str,
        images: Optional[Sequence[str]] = None,
        json_mode: bool = False,
    ) -> str:
        """One chat-completions call. ``images`` = local file paths (-> data URIs)."""

        @_retry(max_attempts=self.max_retries)
        def _call() -> str:
            client = self._get_client()
            content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
            for img in (images or []):
                content.append(
                    {"type": "image_url",
                     "image_url": {"url": self._img_to_data_uri(img)}}
                )
            kwargs: Dict[str, Any] = dict(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": content},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""

        return _call()

    def _chat_json(
        self, system: str, user_text: str,
        images: Optional[Sequence[str]] = None,
    ) -> dict:
        """``_chat`` with JSON-mode + lenient parse; returns {} on hard failure."""
        out = self._chat(system, user_text, images=images, json_mode=True)
        d = loads_lenient(out)
        return d if isinstance(d, dict) else {}

    # ---- (a) instruction generation -------------------------------------
    def gen_instruction(
        self, image_path: str, scene_meta: Optional[dict] = None
    ) -> dict:
        sm = scene_meta or {}
        preset_clause = ""
        preset_name = sm.get("preset_name") or sm.get("style")
        if preset_name:
            preset_clause = (
                f", which an expert retouched using the preset '{preset_name}' "
                f"(scene: {sm.get('scene', 'unknown')}, style: {sm.get('style', 'unknown')})."
            )
        system = self.persona + "\n\nYou are reverse-engineering the photographer's intent."
        user_text = (
            f"Look at this photograph{preset_clause}.\n"
            "Describe, as a natural editing request a user would type, the retouch this "
            "image needs/received. Cover three aspects when relevant:\n"
            "  1. LIGHTING (exposure, contrast, highlights/shadows, tone),\n"
            "  2. GLOBAL COLOR & TEMPERATURE (warmth, tint, overall vibrance/saturation),\n"
            "  3. SPECIFIC COLOR (named hues e.g. sky-blue, skin-orange, foliage-green).\n"
            "Output JSON ONLY:\n"
            '{"instruction_long": "<rich, specific, ~2-3 sentences>", '
            '"instruction_short": "<terse user-style, <=12 words>", "lang": "en"}'
        )
        d = self._chat_json(system, user_text, images=[image_path])
        return {
            "instruction_long": str(d.get("instruction_long", "")).strip(),
            "instruction_short": str(d.get("instruction_short", "")).strip(),
            "lang": str(d.get("lang", "en")).strip() or "en",
        }

    # ---- (b) <think> + <answer> params ----------------------------------
    def reason_params(self, image_path: str, instruction: str) -> dict:
        # System = MMArt persona VERBATIM (already specifies the <think>/<answer> envelope).
        system = self.persona
        user_text = f"<image>{instruction}"
        # Not JSON-mode: the assistant emits the <think>...</think><answer>...</answer>
        # envelope, which we parse by regex then json/ast.
        raw = self._chat(system, user_text, images=[image_path], json_mode=False)
        think, params, extras = split_think_answer(raw)
        return {
            "think": think,
            "answer": params or {},
            "local_hints": extras.get("local_hints", {}),
            "dropped": extras.get("dropped", []),
        }

    # ---- (c) verify (before/after + MMArt processed check) --------------
    def verify(self, before_path: str, after_path: str, params: dict) -> dict:
        system = self.persona + "\n\nYou are a strict QA reviewer for a retouching dataset."
        try:
            params_json = json.dumps(params, ensure_ascii=False)
        except Exception:
            params_json = str(params)
        instruction = ""
        if isinstance(params, dict):
            instruction = str(params.get("_instruction", "")) if "_instruction" in params else ""
        user_text = (
            'Image 1 = ORIGINAL ("before"). Image 2 = RETOUCHED RESULT ("after").\n'
            f"The after was produced by these Lightroom params: {params_json}.\n"
            f'The stated edit intent was: "{instruction}".\n'
            "Judge:\n"
            "  - look_match: does the after visually realize the intent vs the before?\n"
            "  - param_sane: are the param values plausible (no extreme/contradictory "
            "values, within typical Lightroom ranges)?\n"
            "  - processed_ok: does the after look like a genuine expert retouch of the SAME "
            "scene as before (NOT a different image, NOT a corrupt/over-baked render)?\n"
            "Output JSON ONLY:\n"
            '{"look_match": bool, "param_sane": bool, "processed_ok": bool, '
            '"score": <0.0-1.0>, "reason": "<one sentence>"}'
        )
        d = self._chat_json(system, user_text, images=[before_path, after_path])
        if not d:
            # judge/parse failure: signal "not measured" so the caller leaves the
            # quality fields None (gate stays open) instead of auto-rejecting a good
            # sample on a transient flake. (Coercing {} -> all-False + 0.0 would
            # silently delete it.)
            return {}
        return {
            "look_match": bool(d.get("look_match", False)),
            "param_sane": bool(d.get("param_sane", False)),
            "processed_ok": bool(d.get("processed_ok", False)),
            "score": _clamp01(d.get("score", 0.0)),
            "reason": str(d.get("reason", "")).strip(),
        }

    # ---- (d) scene + region-concept tagging -----------------------------
    def tag_scene_region(self, image_path: str, instruction: str) -> dict:
        system = self.persona + "\n\nYou also localize which region the edit targets."
        user_text = (
            f'<image> Edit intent: "{instruction}".\n'
            "Tag this photo:\n"
            "  - scene: one of {portrait, landscape, street, food, product, wedding, "
            "night, architecture, still_life}\n"
            "  - style: short label (e.g. 'warm vintage film', 'clean bright commercial')\n"
            "  - region_local: true if the edit primarily targets a SPECIFIC region/object, "
            "false if it is a global look.\n"
            "  - sam3_concepts: list of open-vocabulary noun phrases naming the target "
            'region(s) for SAM3 text-prompting (e.g. ["sky"], ["person\'s face","hair"]).\n'
            "  - groundingdino_prompt: a single GroundingDINO text prompt, period-separated "
            '(e.g. "sky . person").\n'
            "  - masksubtype_hint: MMArt MaskSubType int (1=Subject, 2=Sky, 3=Person; 0 if "
            "none/global).\n"
            "Output JSON ONLY with exactly those keys."
        )
        d = self._chat_json(system, user_text, images=[image_path])
        scene = str(d.get("scene", "")).strip().lower()
        if scene not in _VALID_SCENES:
            scene = "any"
        concepts = d.get("sam3_concepts", [])
        if not isinstance(concepts, list):
            concepts = [str(concepts)] if concepts else []
        concepts = [str(c).strip() for c in concepts if str(c).strip()]
        try:
            msub = int(d.get("masksubtype_hint", 0) or 0)
        except (TypeError, ValueError):
            msub = 0
        return {
            "scene": scene,
            "style": str(d.get("style", "")).strip(),
            "region_local": bool(d.get("region_local", False)),
            "sam3_concepts": concepts,
            "groundingdino_prompt": str(d.get("groundingdino_prompt", "")).strip(),
            "masksubtype_hint": msub,
        }


def _clamp01(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, f))


# ---------------------------------------------------------------------------
# 5. Quality gate (config-driven accept/reject over a Sample's QA scores).
# ---------------------------------------------------------------------------

# Default thresholds mirror config.yaml:qa (so callers may run without a config dict).
_DEFAULT_QA = {
    "mllm_score_min": 0.70,
    "require_look_match": True,
    "require_param_sane": True,
    "mmart_require_processed_ok": True,
    "mask_quality_min": 0.30,
    "histsim_min": 0.50,
    "er_recon_psnr_min": 25.0,
}


def quality_gate(
    sample: Sample,
    qa_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, QualityScores]:
    """Accept/reject a Sample on its computed QA scores (data_construction_dossier §8).

    Returns ``(accept, scores)`` where ``scores`` is the (possibly mutated)
    ``sample.quality`` with ``rejected_reason`` set on the first failed gate. This
    is the heuristic used by ``streams.QAGate.accept`` but exposed here so the
    vLLM-cleaning stage can gate + log rejects right after annotation.

    Gates (each applied only when the corresponding score is present / enabled):
      1. mllm_score      >= mllm_score_min
      2. look_match      truthy        (if require_look_match)
      3. param_sane      truthy        (if require_param_sane)
      4. processed_ok    truthy        (MMArt only, if mmart_require_processed_ok)
      5. mask_quality    >= mask_quality_min   (region-local samples only)
      6. histsim         >= histsim_min        (if present)
      7. er_recon_psnr   >= er_recon_psnr_min  (if present; Track-A floor)
    """
    cfg = {**_DEFAULT_QA, **(qa_cfg or {})}
    q = sample.quality

    def reject(reason: str) -> Tuple[bool, QualityScores]:
        q.rejected_reason = reason
        return False, q

    # 1. MLLM judge score
    if q.mllm_score is not None and q.mllm_score < cfg["mllm_score_min"]:
        return reject(f"mllm_score<{cfg['mllm_score_min']}({q.mllm_score:.3f})")

    # 2/3. look_match && param_sane
    if cfg["require_look_match"] and q.look_match is not None and not q.look_match:
        return reject("look_match=false")
    if cfg["require_param_sane"] and q.param_sane is not None and not q.param_sane:
        return reject("param_sane=false")

    # 4. MMArt processed.jpg sanity (caveat a) — only for MMArt-sourced pixels.
    is_mmart = (
        sample.stream == StreamId.S3_MMART_LOCAL
        or sample.scene_meta.tags and "mmart" in sample.scene_meta.tags
        or sample.meta.get("corpus") == "mmart"
    )
    if (
        cfg["mmart_require_processed_ok"]
        and is_mmart
        and q.processed_ok is not None
        and not q.processed_ok
    ):
        return reject("mmart_processed_ok=false")

    # 5. mask quality (region-local only)
    if (
        sample.region_local
        and q.mask_quality is not None
        and q.mask_quality < cfg["mask_quality_min"]
    ):
        return reject(f"mask_quality<{cfg['mask_quality_min']}({q.mask_quality:.3f})")

    # 6. color-direction agreement
    if q.histsim is not None and q.histsim < cfg["histsim_min"]:
        return reject(f"histsim<{cfg['histsim_min']}({q.histsim:.3f})")

    # 7. Track-A round-trip PSNR floor
    if q.er_recon_psnr is not None and q.er_recon_psnr < cfg["er_recon_psnr_min"]:
        return reject(f"er_recon_psnr<{cfg['er_recon_psnr_min']}({q.er_recon_psnr:.2f})")

    q.rejected_reason = None
    return True, q


# ---------------------------------------------------------------------------
# 6. Thin CLI (smoke-test a single image against the served endpoint).
#    NEVER launches a server; talks HTTP to an already-running one.
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML required to read config.yaml") from exc
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Offline Qwen3-VL cleaner smoke-test (HTTP to a running vLLM endpoint)."
    )
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    ap.add_argument("--image", help="image path for templates (a)/(b)/(d)")
    ap.add_argument("--after", help="after image path for template (c)")
    ap.add_argument(
        "--task", choices=["a", "b", "c", "d", "all"], default="a",
        help="a=gen_instruction b=reason_params c=verify d=tag_scene_region",
    )
    ap.add_argument("--instruction", default="make this look warm and cinematic")
    args = ap.parse_args(argv)

    cfg = _load_yaml(args.config) if os.path.exists(args.config) else {}
    cleaner = QwenVLCleaner.from_config(cfg)

    if not args.image:
        ap.error("--image is required")

    out: Dict[str, Any] = {}
    if args.task in ("a", "all"):
        out["a_gen_instruction"] = cleaner.gen_instruction(args.image)
    if args.task in ("b", "all"):
        instr = out.get("a_gen_instruction", {}).get("instruction_long") or args.instruction
        out["b_reason_params"] = cleaner.reason_params(args.image, instr)
    if args.task in ("c", "all"):
        if not args.after:
            ap.error("--after is required for task c")
        out["c_verify"] = cleaner.verify(args.image, args.after, {"_instruction": args.instruction})
    if args.task in ("d", "all"):
        out["d_tag_scene_region"] = cleaner.tag_scene_region(args.image, args.instruction)

    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
