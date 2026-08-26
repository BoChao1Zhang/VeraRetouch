"""C-cap pilot: re-write 10 LUT captions under the v2 caption spec and diff them.

What this does
--------------
1. Draw ``--size`` rows with ``random.Random(--seed).sample(ok_rows, size)`` from
   ``annotations.closed-v1.jsonl`` (file order, ``ok`` rows only).
2. For each LUT, pick the *significant* HSL bands off the measured per-band numbers
   (see ``SIGNIFICANT_GATE`` below), rank them and keep the top ``TOP_BANDS``.
3. Ask ``gpt-5.6-terra`` (lane-1 of the databuild credentials TOML, non-streaming,
   reasoning effort ``low``, temperature 0.3, Responses API strict ``json_schema``)
   for ``{"style", "hue_description", "skin_note"}``.
4. Run the pre-registered **sign validator** on the answer.  A violation is fed back
   verbatim and the LUT is re-generated at most ``MAX_REGENERATIONS`` times; a run that
   still fails is written out as ``FAILED_VALIDATION`` with its last answer intact.
5. Emit a JSON ledger plus a side-by-side markdown report (old Chinese caption vs the
   new English one).

Pre-registered parameters
-------------------------
* ``SIGNIFICANT_GATE = 5.0`` -- a band is significant iff ``|d_sat_pct| >= 5`` or
  ``|d_hue_deg| >= 5`` or ``|d_lum_pct| >= 5``.
* ``TOP_BANDS = 4`` -- significant bands are ordered by
  ``max(|d_sat_pct|, |d_hue_deg|, |d_lum_pct|)`` descending, ties broken by the band's
  hue-wheel order, and the first four are handed to the model.
* Direction word <-> number sign table (``LEXICON``):

  ===================================  =============================================
  word / phrase                        required number
  ===================================  =============================================
  ``richer`` / ``more saturated``      ``d_sat_pct >= +5``
  ``muted`` / ``desaturated``          ``d_sat_pct <= -5``
  ``brightened`` / ``lighter`` /       ``d_lum_pct >= +5``
  ``lifted``
  ``darkened`` / ``deeper`` /          ``d_lum_pct <= -5``
  ``deepened``
  ``toward <colour>``                  ``|d_hue_deg| >= 5`` and ``<colour>`` is the
                                       hue-wheel neighbour on the side that
                                       ``sign(d_hue_deg)`` points to
  ===================================  =============================================

  ``d_hue_deg`` is a signed rotation on the sRGB hue wheel, so ``+`` moves a band
  toward the next band by increasing hue angle (red -> orange -> yellow -> green ->
  aqua -> blue -> purple -> magenta -> red) and ``-`` moves it the other way.
* ``hue_description`` must be a list of ``;``-separated clauses; each clause carries
  exactly one band subject (no cross-band wording, so no averaging is expressible) and
  at least one direction word, and that subject must be one of the bands that were
  handed to the model.
* ``skin_note`` is checked against the ``红`` and ``橙`` bands only (the two skin-bearing
  bands of the 8-band spec).  **Assumption, pre-registered here:** a direction word in
  ``skin_note`` passes iff at least one of ``红`` / ``橙`` clears the 5-unit gate on that
  quantity and *neither* of them clears it with the opposite sign.  ``skin_note`` may be
  the empty string.
* No digit may appear in any of the three slots (the spec forbids numeric prefixes).

Usage::

    .venv/bin/python -m dataset_build.tools.build_caption_v2_pilot \\
        --out-dir docs/assets/caption_v2_pilot_20260824
"""
from __future__ import annotations

import argparse
import json
import random
import re
import tomllib
from pathlib import Path
from typing import Any

from openai import OpenAI

# --- fixed inputs -------------------------------------------------------------------
ANNOTATIONS = Path("/home/bc/data/scratch/lut_reannotate/out/annotations.closed-v1.jsonl")
FINGERPRINTS_V2 = Path(
    "/home/bc/data/scratch/lut_reannotate/out/segment_fingerprints.v2.jsonl"
)
CREDENTIALS = Path("/home/bc/VeraRetouch/databuild.prod-l8-local400k-20260812.toml")
LANE = "provider-c-lane-1"

# --- pre-registered constants -------------------------------------------------------
SAMPLE_SEED = 20260824
SAMPLE_SIZE = 10
SIGNIFICANT_GATE = 5.0
TOP_BANDS = 4
MAX_REGENERATIONS = 2
STYLE_MIN_WORDS = 3
STYLE_MAX_WORDS = 5

# --- transport ----------------------------------------------------------------------
REASONING_EFFORT = "low"
TEMPERATURE = 0.3
MAX_OUTPUT_TOKENS = 2048
TIMEOUT_SECONDS = 180.0
ATTEMPTS = 3

# --- band vocabulary ----------------------------------------------------------------
# (chinese band name, hue-wheel centre in degrees, english band word) in wheel order.
BANDS: tuple[tuple[str, float, str], ...] = (
    ("红", 0.0, "red"),
    ("橙", 30.0, "orange"),
    ("黄", 60.0, "yellow"),
    ("绿", 120.0, "green"),
    ("浅绿", 180.0, "aqua"),
    ("蓝", 240.0, "blue"),
    ("紫", 270.0, "purple"),
    ("洋红", 300.0, "magenta"),
)
CN_TO_EN = {cn: en for cn, _deg, en in BANDS}
EN_ORDER: tuple[str, ...] = tuple(en for _cn, _deg, en in BANDS)
EN_TO_CN = {en: cn for cn, _deg, en in BANDS}
# accepted synonyms, folded onto the canonical band word before any check runs
ALIASES: dict[str, str] = {
    "reds": "red", "oranges": "orange", "yellows": "yellow", "greens": "green",
    "aquas": "aqua", "blues": "blue", "purples": "purple", "magentas": "magenta",
    "cyan": "aqua", "cyans": "aqua", "teal": "aqua", "teals": "aqua",
    "violet": "purple", "violets": "purple", "pink": "magenta", "pinks": "magenta",
}
SKIN_BANDS: tuple[str, ...] = ("红", "橙")

# --- direction lexicon --------------------------------------------------------------
# phrase -> (quantity key, required sign)
LEXICON: dict[str, tuple[str, int]] = {
    "more saturated": ("d_sat_pct", +1),
    "richer": ("d_sat_pct", +1),
    "desaturated": ("d_sat_pct", -1),
    "muted": ("d_sat_pct", -1),
    "brightened": ("d_lum_pct", +1),
    "lighter": ("d_lum_pct", +1),
    "lifted": ("d_lum_pct", +1),
    "darkened": ("d_lum_pct", -1),
    "deepened": ("d_lum_pct", -1),
    "deeper": ("d_lum_pct", -1),
}
# comparative / delta words that duplicate a lexicon meaning without being checkable.
# Their presence is a violation: every directional claim must come from ``LEXICON``.
OUT_OF_VOCAB: tuple[str, ...] = (
    "warmer", "cooler", "brighter", "darker", "dimmer", "duller", "paler",
    "stronger", "weaker", "vivid", "vibrant", "punchier", "punchy", "boosted",
    "reduced", "increased", "decreased", "intensified", "softened", "faded",
    "dulled", "neutralized", "neutralised", "crushed", "lowered", "raised",
)
OUT_OF_VOCAB_RE = re.compile(r"\b(" + "|".join(OUT_OF_VOCAB) + r")\b")
BARE_SATURATED_RE = re.compile(r"(?<!more )\bsaturated\b")
BAND_WORD_RE = re.compile(
    r"\b(" + "|".join(sorted(set(EN_ORDER) | set(ALIASES), key=len, reverse=True)) + r")\b"
)
TOWARD_RE = re.compile(
    r"\b(?:shifted\s+|pushed\s+|rotated\s+|pulled\s+)?towards?\s+(?:the\s+)?("
    + "|".join(sorted(set(EN_ORDER) | set(ALIASES), key=len, reverse=True))
    + r")\b"
)
DIGIT_RE = re.compile(r"\d")


def neighbour(band_cn: str, sign: int) -> str:
    """The hue-wheel neighbour of ``band_cn`` on the ``sign`` side, as an english word."""
    index = EN_ORDER.index(CN_TO_EN[band_cn])
    return EN_ORDER[(index + (1 if sign > 0 else -1)) % len(EN_ORDER)]


def canonical(word: str) -> str:
    return ALIASES.get(word.lower(), word.lower())


# --- band selection -----------------------------------------------------------------
def band_magnitude(row: dict[str, Any]) -> float:
    return max(
        abs(float(row["d_sat_pct"])),
        abs(float(row["d_hue_deg"])),
        abs(float(row["d_lum_pct"])),
    )


def significant_bands(bands: dict[str, Any]) -> list[str]:
    """Chinese band names above the gate, ranked, capped at ``TOP_BANDS``."""
    wheel = {cn: index for index, (cn, _deg, _en) in enumerate(BANDS)}
    picked = [cn for cn, _deg, _en in BANDS
              if cn in bands and band_magnitude(bands[cn]) >= SIGNIFICANT_GATE]
    picked.sort(key=lambda cn: (-band_magnitude(bands[cn]), wheel[cn]))
    return picked[:TOP_BANDS]


# --- validator ----------------------------------------------------------------------
def _check_word(
    phrase: str, quantity: str, sign: int, numbers: dict[str, Any], label: str
) -> str | None:
    value = float(numbers[quantity])
    if sign > 0 and value >= SIGNIFICANT_GATE:
        return None
    if sign < 0 and value <= -SIGNIFICANT_GATE:
        return None
    return (
        f"{label}: the word '{phrase}' claims {quantity} "
        f"{'>= +' if sign > 0 else '<= -'}{SIGNIFICANT_GATE:g}, but the measured "
        f"{quantity} is {value:+.1f}."
    )


def _clause_violations(
    clause: str, bands: dict[str, Any], allowed: list[str]
) -> list[str]:
    text = clause.strip().lower()
    if not text:
        return []
    targets = [canonical(match.group(1)) for match in TOWARD_RE.finditer(text)]
    stripped = TOWARD_RE.sub(" ", text)
    subjects = {canonical(m.group(1)) for m in BAND_WORD_RE.finditer(stripped)}
    if len(subjects) != 1:
        return [
            f"clause '{clause.strip()}': needs exactly one band subject, found "
            f"{sorted(subjects) if subjects else 'none'}."
        ]
    subject_en = subjects.pop()
    subject_cn = EN_TO_CN[subject_en]
    if subject_cn not in allowed:
        return [
            f"clause '{clause.strip()}': band '{subject_en}' was not in the table you "
            f"were given; only {[CN_TO_EN[cn] for cn in allowed]} may be described."
        ]
    numbers = bands[subject_cn]
    problems: list[str] = []
    found = 0
    for phrase, (quantity, sign) in LEXICON.items():
        if re.search(r"\b" + re.escape(phrase) + r"\b", stripped):
            found += 1
            issue = _check_word(phrase, quantity, sign, numbers, f"band {subject_en}")
            if issue is not None:
                problems.append(issue)
    for target in targets:
        found += 1
        rotation = float(numbers["d_hue_deg"])
        if abs(rotation) < SIGNIFICANT_GATE:
            problems.append(
                f"band {subject_en}: 'toward {target}' claims a hue rotation, but the "
                f"measured d_hue_deg is {rotation:+.1f} (below "
                f"{SIGNIFICANT_GATE:g} degrees)."
            )
            continue
        wanted = neighbour(subject_cn, 1 if rotation > 0 else -1)
        if target != wanted:
            problems.append(
                f"band {subject_en}: d_hue_deg is {rotation:+.1f}, so the only allowed "
                f"hue target is '{wanted}', not '{target}'."
            )
    if found == 0:
        problems.append(
            f"clause '{clause.strip()}': carries no direction word from the allowed list."
        )
    return problems


def _skin_violations(note: str, bands: dict[str, Any]) -> list[str]:
    text = note.strip().lower()
    if not text:
        return []
    problems: list[str] = []
    present = [cn for cn in SKIN_BANDS if cn in bands]
    for phrase, (quantity, sign) in LEXICON.items():
        if not re.search(r"\b" + re.escape(phrase) + r"\b", text):
            continue
        values = [float(bands[cn][quantity]) for cn in present]
        supports = any(v >= SIGNIFICANT_GATE if sign > 0 else v <= -SIGNIFICANT_GATE
                       for v in values)
        contradicts = any(v <= -SIGNIFICANT_GATE if sign > 0 else v >= SIGNIFICANT_GATE
                          for v in values)
        if not supports or contradicts:
            problems.append(
                f"skin_note: '{phrase}' claims {quantity} "
                f"{'>= +' if sign > 0 else '<= -'}{SIGNIFICANT_GATE:g}, but the skin "
                f"bands measure "
                + ", ".join(
                    f"{CN_TO_EN[cn]} {float(bands[cn][quantity]):+.1f}" for cn in present
                )
                + "."
            )
    for match in TOWARD_RE.finditer(text):
        target = canonical(match.group(1))
        allowed_targets = set()
        for cn in present:
            rotation = float(bands[cn]["d_hue_deg"])
            if abs(rotation) >= SIGNIFICANT_GATE:
                allowed_targets.add(neighbour(cn, 1 if rotation > 0 else -1))
        if target not in allowed_targets:
            problems.append(
                f"skin_note: 'toward {target}' is not supported; the skin bands measure "
                + ", ".join(
                    f"{CN_TO_EN[cn]} d_hue_deg {float(bands[cn]['d_hue_deg']):+.1f}"
                    for cn in present
                )
                + f", allowing only {sorted(allowed_targets) or 'no hue target'}."
            )
    return problems


def _vocabulary_violations(slot: str, text: str) -> list[str]:
    """Directional wording outside ``LEXICON`` cannot be sign-checked, so it is banned."""
    lowered = text.lower()
    problems = [
        f"{slot}: '{match.group(1)}' is not in the allowed direction-word list."
        for match in OUT_OF_VOCAB_RE.finditer(lowered)
    ]
    if BARE_SATURATED_RE.search(lowered):
        problems.append(
            f"{slot}: bare 'saturated' is not allowed; use 'more saturated' or "
            "'desaturated'."
        )
    return problems


def validate(answer: dict[str, str], bands: dict[str, Any], allowed: list[str]) -> list[str]:
    problems: list[str] = []
    style = str(answer.get("style", "")).strip()
    hue = str(answer.get("hue_description", "")).strip()
    skin = str(answer.get("skin_note", "")).strip()

    for slot, value in (("style", style), ("hue_description", hue), ("skin_note", skin)):
        if DIGIT_RE.search(value):
            problems.append(f"{slot}: contains a digit; the spec forbids numbers.")
    words = [w for w in re.split(r"[\s]+", style.rstrip(".")) if w]
    if not STYLE_MIN_WORDS <= len(words) <= STYLE_MAX_WORDS:
        problems.append(
            f"style: has {len(words)} words, must be "
            f"{STYLE_MIN_WORDS}-{STYLE_MAX_WORDS}."
        )
    if not hue:
        problems.append("hue_description: empty.")
    problems.extend(_vocabulary_violations("hue_description", hue))
    problems.extend(_vocabulary_violations("skin_note", skin))
    for clause in re.split(r"[;.]", hue):
        problems.extend(_clause_violations(clause, bands, allowed))
    problems.extend(_skin_violations(skin, bands))
    return problems


# --- prompt -------------------------------------------------------------------------
SCHEMA_NAME = "lut_caption_v2"
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "style": {"type": "string"},
        "hue_description": {"type": "string"},
        "skin_note": {"type": "string"},
    },
    "required": ["style", "hue_description", "skin_note"],
    "additionalProperties": False,
}

INSTRUCTION = """\
You are writing an English caption for one colour LUT. The caption has two slots plus an
optional third; write them exactly to spec, in English, and use no digits anywhere.

`style` -- 3 to 5 words, an abstract description of the stylistic change (mood / look).
No numbers, no percentages, no mention of exposure, white balance or overall saturation
levels.

`hue_description` -- an objective per-band description built ONLY from the measured
table below. Hard rules:
  * Write one or more clauses separated by "; ".
  * Each clause describes exactly ONE band and must name that band once (red, orange,
    yellow, green, aqua, blue, purple, magenta). Never merge or average two bands into
    one clause. Never mention a band that is absent from the table.
  * Keep every band's own sign. Do not smooth signs across bands.
  * Each clause must carry at least one direction word, and only these are allowed:
      saturation up   : "richer" | "more saturated"      (needs d_sat_pct >= +5)
      saturation down : "muted" | "desaturated"          (needs d_sat_pct <= -5)
      lightness up    : "brightened" | "lighter" | "lifted"   (needs d_lum_pct >= +5)
      lightness down  : "darkened" | "deepened" | "deeper"    (needs d_lum_pct <= -5)
      hue rotation    : "shifted toward <colour>"        (needs |d_hue_deg| >= 5)
  * "shifted toward <colour>" must name the hue-wheel neighbour on the side the SIGN of
    d_hue_deg points to. The wheel order is red -> orange -> yellow -> green -> aqua ->
    blue -> purple -> magenta -> red. A POSITIVE d_hue_deg moves forward along that
    order; a NEGATIVE d_hue_deg moves backward.
  * Use a direction word only when its number clears the threshold above.
  * No other comparative wording is accepted anywhere in `hue_description` or
    `skin_note`: words such as "warmer", "cooler", "brighter", "darker", "vivid",
    "boosted", "reduced", "softened", "faded" or a bare "saturated" are rejected.

`skin_note` -- one short clause about skin only, or the empty string if there is nothing
supported to say. It may use the same direction words, and they must be consistent with
the red and orange band numbers in the table.

Do not restate brightness, colour temperature or overall saturation as numbers or as a
numeric prefix; those are read elsewhere.
"""


def build_prompt(record: dict[str, Any], allowed: list[str]) -> str:
    features = record["hsl_features"]
    bands = features["bands"]
    lines = [INSTRUCTION, "", "=== measured bands (only these may be described) ==="]
    lines.append("| band | d_sat_pct | d_hue_deg | d_lum_pct |")
    lines.append("| --- | --- | --- | --- |")
    for cn in allowed:
        row = bands[cn]
        lines.append(
            f"| {CN_TO_EN[cn]} | {float(row['d_sat_pct']):+.1f} | "
            f"{float(row['d_hue_deg']):+.2f} | {float(row['d_lum_pct']):+.1f} |"
        )
    summary = features["summary"]
    lines.append("")
    lines.append("=== global summary (context only, do not restate as numbers) ===")
    lines.append(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    lines.append("")
    lines.append("=== neutral ramp (context only) ===")
    lines.append(json.dumps(features["neutral_ramp"], ensure_ascii=False))
    skin = str(record.get("per_probe", {}).get("肤色", "") or "")
    if skin:
        lines.append("")
        lines.append("=== skin probe, verbatim from the existing annotation ===")
        lines.append(skin)
    return "\n".join(lines)


# --- provider -----------------------------------------------------------------------
def lane_credentials(path: Path, identity: str) -> tuple[str, str, str]:
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    annotation = data["annotation"]
    for row in annotation["external_endpoints"]:
        if str(row.get("id")) == identity:
            return (
                str(row["base_url"]).rstrip("/"), str(row["api_key"]),
                str(annotation["external_model"]),
            )
    raise KeyError(f"endpoint {identity!r} is absent from {path}")


def ask(client: OpenAI, model: str, prompt: str) -> dict[str, Any]:
    last: Exception | None = None
    for _attempt in range(ATTEMPTS):
        try:
            response = client.responses.create(
                model=model,
                input=[{"role": "user", "content": [
                    {"type": "input_text", "text": prompt}
                ]}],
                stream=False,
                temperature=TEMPERATURE,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                store=False,
                reasoning={"effort": REASONING_EFFORT},
                text={"format": {
                    "type": "json_schema", "name": SCHEMA_NAME,
                    "strict": True, "schema": SCHEMA,
                }},
                timeout=TIMEOUT_SECONDS,
            )
            text = str(getattr(response, "output_text", "") or "")
            if not text:
                raise ValueError("responses_output_empty")
            usage = getattr(response, "usage", None)
            return {
                "parsed": json.loads(text),
                "usage": {
                    "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
                    "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
                },
            }
        except Exception as exc:  # retried below; the last one is re-raised
            last = exc
    raise RuntimeError(f"caption request failed after {ATTEMPTS} attempts: {last}")


def compose(answer: dict[str, str]) -> str:
    style = str(answer.get("style", "")).strip().rstrip(".")
    hue = str(answer.get("hue_description", "")).strip().rstrip(".")
    skin = str(answer.get("skin_note", "")).strip().rstrip(".")
    text = f"{style}. {hue}."
    if skin:
        text += f" {skin}."
    return text


# --- markdown -----------------------------------------------------------------------
def _cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def markdown(payload: dict[str, Any]) -> str:
    out: list[str] = []
    out.append("# C-cap 新 caption 方案试标(10 个 LUT)\n")
    out.append("## 预注册参数\n")
    out.append(f"- 抽样:`random.Random({payload['seed']}).sample(ok_rows, "
               f"{payload['size']})`,总体 = `{ANNOTATIONS}` 里 `ok=true` 的 "
               f"{payload['n_ok']} 行(文件序)。")
    out.append(f"- 显著 band 判据:`|d_sat_pct| >= {SIGNIFICANT_GATE:g}` 或 "
               f"`|d_hue_deg| >= {SIGNIFICANT_GATE:g}` 或 "
               f"`|d_lum_pct| >= {SIGNIFICANT_GATE:g}`;"
               f"按 `max(|三量|)` 降序(同值按色轮序)取前 {TOP_BANDS} 个。")
    out.append("- 方向词↔数字符号映射表:")
    out.append("")
    out.append("| 方向词 | 要求 |")
    out.append("| --- | --- |")
    grouped: dict[tuple[str, int], list[str]] = {}
    for phrase, key in LEXICON.items():
        grouped.setdefault(key, []).append(phrase)
    for (quantity, sign), phrases in grouped.items():
        gate = f"{'>= +' if sign > 0 else '<= -'}{SIGNIFICANT_GATE:g}"
        out.append(f"| {' / '.join(sorted(phrases))} | `{quantity} {gate}` |")
    out.append(f"| `shifted toward <colour>` | `|d_hue_deg| >= {SIGNIFICANT_GATE:g}`,"
               "且 `<colour>` = 该 band 在色轮序 "
               "red→orange→yellow→green→aqua→blue→purple→magenta→red 上,由 "
               "`sign(d_hue_deg)` 指向的那一侧邻居(正号前进,负号后退) |")
    out.append("")
    out.append("- 结构约束:`hue_description` 用 `; ` 分句,每句恰好一个 band 主语且至少一个"
               "方向词,主语必须在下发表里(因此不存在跨 band 平均的写法);三个槽位均禁出现数字;"
               f"`style` 限 {STYLE_MIN_WORDS}-{STYLE_MAX_WORDS} 词。")
    out.append("- 词表封闭:`hue_description` / `skin_note` 里出现表外比较级方向词"
               f"({'、'.join(OUT_OF_VOCAB)},以及裸 `saturated`)即判违规——"
               "表外词无法对数字符号校验。")
    out.append("- `skin_note` 校验口径(本次预注册假设):只对 `红`/`橙` 两个含肤色 band 检查;"
               "一个方向词通过的条件是「两者中至少一个在该量上过 5 的门且方向同号」且"
               "「两者都没有以相反符号过门」;允许为空串。")
    out.append(f"- 生成:endpoint `{payload['lane']}` / model `{payload['model']}` / "
               f"effort `{REASONING_EFFORT}` / temperature {TEMPERATURE} / "
               f"transport `nonstream` / strict `json_schema`。")
    out.append(f"- 校验不过时带违规说明重生成,最多 {MAX_REGENERATIONS} 次;仍不过标 "
               "`FAILED_VALIDATION` 并原样展示。")
    out.append("")
    out.append("## token 成本实测\n")
    cost = payload["cost"]
    out.append("| 量 | 值 |")
    out.append("| --- | --- |")
    out.append(f"| 调用次数(含重生成) | {cost['calls']} |")
    out.append(f"| input tokens 合计 | {cost['input_tokens']} |")
    out.append(f"| output tokens 合计 | {cost['output_tokens']} |")
    out.append(f"| in+out 合计 | {cost['total_tokens']} |")
    out.append(f"| 单 LUT 平均 input | {cost['input_per_lut']:.1f} |")
    out.append(f"| 单 LUT 平均 output | {cost['output_per_lut']:.1f} |")
    out.append(f"| 单 LUT 平均 in+out | {cost['total_per_lut']:.1f} |")
    out.append("")
    out.append("(prompt 本体每次约 3.3k 字符;上表 input_tokens 为 endpoint 回报的原始 "
               "`usage.input_tokens`,未做任何折算。)\n")
    out.append("## 待决策(本次未拍板,按上面的预注册口径原样执行)\n")
    out.append("1. `toward <colour>` 的邻居规则只按 `sign(d_hue_deg)` 取相邻 band,与旋转"
               "幅度无关。本批 #1 `浅绿` 的 `d_hue_deg = -155.49`,按规则判 `green`;若改按"
               "「落点色相角」判,`180 - 155.49 = 24.5` 度落在 `orange` 附近。两种口径在"
               "大角度上不一致,需要定口径。")
    out.append("2. 同一 LUT 的 `d_sat_pct` 接近 -100(#1 为 -89 ~ -95)时,band 已近无彩,"
               "`d_hue_deg` 是否仍应进 caption,未定。")
    out.append("3. `skin_note` 目前只对 `红`/`橙` 两 band 做符号校验;`per_probe.肤色` 原文"
               "里的内容(如「中间调偏冷」)无对应数字,不在校验范围内。")
    out.append("")
    out.append("## 校验结果汇总\n")
    out.append("| # | preset_id | 生成轮数 | 重生成次数 | 结果 |")
    out.append("| --- | --- | --- | --- | --- |")
    for index, row in enumerate(payload["results"], 1):
        out.append(
            f"| {index} | `{row['preset_id']}` | {row['rounds']} | "
            f"{row['regenerations']} | {row['status']} |"
        )
    out.append("")
    passed = sum(1 for r in payload["results"] if r["status"] == "PASS")
    out.append(f"通过 {passed}/{len(payload['results'])};重生成总次数 "
               f"{sum(r['regenerations'] for r in payload['results'])}。\n")
    out.append("## 逐 LUT\n")
    for index, row in enumerate(payload["results"], 1):
        out.append(f"### {index}. `{row['preset_id']}`(name `{row['name']}`)\n")
        out.append("显著 band 数表(下发给模型的表):\n")
        out.append("| band | d_sat_pct | d_hue_deg | d_lum_pct |")
        out.append("| --- | --- | --- | --- |")
        for band in row["bands_table"]:
            out.append(
                f"| {band['band_en']} ({band['band_cn']}) | {band['d_sat_pct']:+.1f} | "
                f"{band['d_hue_deg']:+.2f} | {band['d_lum_pct']:+.1f} |"
            )
        out.append("")
        fingerprint = row["fingerprint_v2"]
        if fingerprint is None:
            out.append("v2 指纹:缺该 preset_id。\n")
        else:
            out.append("v2 指纹三元组:\n")
            out.append("| d_shadow | d_mid | d_high |")
            out.append("| --- | --- | --- |")
            out.append(
                f"| {fingerprint['d_shadow']:+.6f} | {fingerprint['d_mid']:+.6f} | "
                f"{fingerprint['d_high']:+.6f} |"
            )
            out.append("")
        out.append("旧 / 新 caption 并排:\n")
        out.append("| | 原文 |")
        out.append("| --- | --- |")
        out.append(f"| 旧(中文) | {_cell(row['old_caption'])} |")
        out.append(f"| 新(英文) | {_cell(row['new_caption'])} |")
        out.append("")
        out.append("新 caption 三槽原文:\n")
        out.append("| 槽 | 原文 |")
        out.append("| --- | --- |")
        for slot in ("style", "hue_description", "skin_note"):
            out.append(f"| `{slot}` | {_cell(row['answer'].get(slot, ''))} |")
        out.append("")
        out.append(f"校验:{row['rounds']} 轮,结果 {row['status']}。\n")
        for attempt in row["attempts"]:
            flag = "通过" if not attempt["violations"] else "不通过"
            out.append(f"- 第 {attempt['round']} 轮 {flag}"
                       + ("" if not attempt["violations"] else ":"))
            for problem in attempt["violations"]:
                out.append(f"  - {problem}")
        out.append("")
    return "\n".join(out)


# --- driver -------------------------------------------------------------------------
def load_fingerprints(path: Path) -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            histogram = row.get("histogram") or {}
            table[str(row["preset_id"])] = {
                "d_shadow": float(histogram.get("d_shadow", 0.0)),
                "d_mid": float(histogram.get("d_mid", 0.0)),
                "d_high": float(histogram.get("d_high", 0.0)),
            }
    return table


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
    parser.add_argument("--size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    parser.add_argument("--fingerprints", type=Path, default=FINGERPRINTS_V2)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    with args.annotations.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("ok"):
                rows.append(record)
    sample = random.Random(args.seed).sample(rows, min(args.size, len(rows)))
    fingerprints = load_fingerprints(args.fingerprints)

    base_url, api_key, model = lane_credentials(CREDENTIALS, LANE)
    client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0)

    results: list[dict[str, Any]] = []
    calls = 0
    input_tokens = 0
    output_tokens = 0
    for record in sample:
        bands = record["hsl_features"]["bands"]
        allowed = significant_bands(bands)
        prompt = build_prompt(record, allowed)
        attempts: list[dict[str, Any]] = []
        answer: dict[str, str] = {}
        violations: list[str] = []
        for round_index in range(MAX_REGENERATIONS + 1):
            message = prompt
            if violations:
                message = (
                    prompt
                    + "\n\n=== your previous answer failed the sign check ===\n"
                    + json.dumps(answer, ensure_ascii=False)
                    + "\n\n=== violations, fix every one of them ===\n"
                    + "\n".join(f"- {problem}" for problem in violations)
                )
            reply = ask(client, model, message)
            calls += 1
            input_tokens += reply["usage"]["input_tokens"]
            output_tokens += reply["usage"]["output_tokens"]
            answer = {key: str(reply["parsed"].get(key, "") or "")
                      for key in ("style", "hue_description", "skin_note")}
            violations = validate(answer, bands, allowed)
            attempts.append({
                "round": round_index + 1,
                "answer": dict(answer),
                "violations": list(violations),
                "usage": reply["usage"],
            })
            if not violations:
                break
        results.append({
            "preset_id": str(record["preset_id"]),
            "name": str(record.get("name", "")),
            "old_caption": str(record.get("caption", "")),
            "answer": answer,
            "new_caption": compose(answer),
            "status": "PASS" if not violations else "FAILED_VALIDATION",
            "rounds": len(attempts),
            "regenerations": len(attempts) - 1,
            "attempts": attempts,
            "bands_table": [
                {
                    "band_cn": cn, "band_en": CN_TO_EN[cn],
                    "d_sat_pct": float(bands[cn]["d_sat_pct"]),
                    "d_hue_deg": float(bands[cn]["d_hue_deg"]),
                    "d_lum_pct": float(bands[cn]["d_lum_pct"]),
                }
                for cn in allowed
            ],
            "fingerprint_v2": fingerprints.get(str(record["preset_id"])),
            "skin_probe": str(record.get("per_probe", {}).get("肤色", "") or ""),
        })

    size = len(results)
    payload = {
        "seed": args.seed, "size": size, "n_ok": len(rows),
        "lane": LANE, "model": model,
        "reasoning_effort": REASONING_EFFORT, "temperature": TEMPERATURE,
        "significant_gate": SIGNIFICANT_GATE, "top_bands": TOP_BANDS,
        "max_regenerations": MAX_REGENERATIONS,
        "lexicon": {phrase: list(key) for phrase, key in LEXICON.items()},
        "cost": {
            "calls": calls,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_per_lut": input_tokens / size if size else 0.0,
            "output_per_lut": output_tokens / size if size else 0.0,
            "total_per_lut": (input_tokens + output_tokens) / size if size else 0.0,
        },
        "n_pass": sum(1 for row in results if row["status"] == "PASS"),
        "n_regenerations": sum(row["regenerations"] for row in results),
        "results": results,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "pilot.json").write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=1) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "report.md").write_text(markdown(payload), encoding="utf-8")
    print(json.dumps({
        "size": size, "n_pass": payload["n_pass"],
        "n_regenerations": payload["n_regenerations"],
        "cost": payload["cost"], "out_dir": str(args.out_dir),
    }, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
