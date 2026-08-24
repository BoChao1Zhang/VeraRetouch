"""G2 step 3: ask the production model to read one LUT shortlist row, then score it.

Ten proposals are drawn from the step-2 ledger with `random.Random(SAMPLE_SEED)`. Each
one gets a single-turn request on the production endpoint (lane-1 of the databuild
credentials TOML, `gpt-5.6-terra`, non-streaming, reasoning effort `low`, Responses API
with a strict `json_schema` text format - the same transport settings the g30 run used)
carrying the verbatim production `SHORTLIST_HEADER` block plus that LUT's verbatim row.

The model returns `per_column` (its own words for every number on the row),
`expected_visual_effect`, and a `directions` object of five closed-vocabulary slots.
Only `directions` is scored; `per_column` is reproduced verbatim in the report.

Pre-registered mapping from the row numbers to the expected slot value
---------------------------------------------------------------------
Thresholds are the step-2 `NEAR_ZERO` constants (`dL` 0.5 L*, `dSat` 0.5 percent,
`cast_mag` 0.5 Lab chroma, `d_shadow` / `d_high` 0.005 pixel share).

* ``lightness``  : ``dL >  +0.5`` -> ``up``   ; ``dL <  -0.5`` -> ``down``   ; else ``none``
* ``saturation`` : ``dSat > +0.5`` -> ``up``  ; ``dSat < -0.5`` -> ``down``  ; else ``none``
* ``cast``       : ``cast_mag < 0.5`` -> ``none``; otherwise the Lab hue angle
  ``cast_hue`` falls in one of four quadrants centred on the Lab axes
  (``h=0`` is ``+a``, ``h=90`` is ``+b``, ``h=180`` is ``-a``, ``h=270`` is ``-b``),
  with the cuts at 45 / 135 / 225 / 315 degrees::

      [315, 45)  -> magenta   (+a)
      [ 45,135)  -> warm      (+b, yellow)
      [135,225)  -> green     (-a)
      [225,315)  -> cool      (-b, blue)

* ``shadows``    : ``d_shadow`` is an output-minus-input **share** of the three darkest
  L* bins, so lifting the shadows moves pixels *out* of them:
  ``d_shadow < -0.005`` -> ``lift`` ; ``d_shadow > +0.005`` -> ``deepen`` ; else ``none``
* ``highlights`` : ``d_high`` is the same quantity on the two brightest bins, and
  lifting the highlights moves pixels *into* them:
  ``d_high > +0.005`` -> ``lift`` ; ``d_high < -0.005`` -> ``deepen`` ; else ``none``

A second, supplementary reference is scored and reported separately without replacing
the primary one: the row's own ``shadow_dL`` / ``highlight_dL`` L* shifts, under
``> +0.5 -> lift`` / ``< -0.5 -> deepen`` / ``else none``.

Usage::

    .venv/bin/python -m dataset_build.tools.g30_fingerprint_probe \\
        --ledger docs/assets/g30_fingerprint_check_20260824/ledger.json \\
        --out    docs/assets/g30_fingerprint_check_20260824/probe.json
"""
from __future__ import annotations

import argparse
import json
import random
import tomllib
from pathlib import Path
from typing import Any

from openai import OpenAI

from dataset_build.tools.g30_fingerprint_ledger import NEAR_ZERO

SAMPLE_SEED = 20260824
SAMPLE_SIZE = 10
CREDENTIALS = Path("/home/bc/VeraRetouch/databuild.prod-l8-local400k-20260812.toml")
LANE = "provider-c-lane-1"
REASONING_EFFORT = "low"
TEMPERATURE = 0.1
MAX_OUTPUT_TOKENS = 2048
TIMEOUT_SECONDS = 180.0
ATTEMPTS = 3

PER_COLUMN_KEYS: tuple[str, ...] = (
    "dL", "contrast", "shadow_dL", "highlight_dL", "cast_hue", "cast_mag",
    "dSat", "hue_rot", "d_shadow", "d_mid", "d_high",
)
DIRECTION_SLOTS: dict[str, tuple[str, ...]] = {
    "lightness": ("up", "down", "none"),
    "saturation": ("up", "down", "none"),
    "cast": ("warm", "cool", "magenta", "green", "none"),
    "shadows": ("lift", "deepen", "none"),
    "highlights": ("lift", "deepen", "none"),
}
SCHEMA_NAME = "lut_row_reading"
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "per_column": {
            "type": "object",
            "properties": {key: {"type": "string"} for key in PER_COLUMN_KEYS},
            "required": list(PER_COLUMN_KEYS),
            "additionalProperties": False,
        },
        "expected_visual_effect": {"type": "string"},
        "directions": {
            "type": "object",
            "properties": {
                slot: {"type": "string", "enum": list(values)}
                for slot, values in DIRECTION_SLOTS.items()
            },
            "required": list(DIRECTION_SLOTS),
            "additionalProperties": False,
        },
    },
    "required": ["per_column", "expected_visual_effect", "directions"],
    "additionalProperties": False,
}
INSTRUCTION = (
    "Below is the verbatim shortlist header of a production LUT retrieval prompt, "
    "followed by one verbatim row from that table. Read the row.\n"
    "1. In `per_column`, say in one sentence what each named number on that row tells "
    "you about this LUT.\n"
    "2. In `expected_visual_effect`, describe what applying this LUT at full strength "
    "would do to a photograph.\n"
    "3. In `directions`, commit to one closed-vocabulary value per slot: lightness "
    "up/down/none, saturation up/down/none, cast warm/cool/magenta/green/none, "
    "shadows lift/deepen/none, highlights lift/deepen/none.\n"
    "Answer only from the numbers on the row and the header that defines them."
)


# --- pre-registered grading ---------------------------------------------------------
def expected_directions(row: dict[str, Any]) -> dict[str, str]:
    """The slot values implied by the row numbers under the mapping in the docstring."""
    def tri(value: float, gate: float, positive: str, negative: str) -> str:
        if value > gate:
            return positive
        if value < -gate:
            return negative
        return "none"

    cast = "none"
    if abs(float(row["cast_mag"])) >= NEAR_ZERO["cast_a"]:
        hue = float(row["cast_hue"]) % 360.0
        if hue < 45.0 or hue >= 315.0:
            cast = "magenta"
        elif hue < 135.0:
            cast = "warm"
        elif hue < 225.0:
            cast = "green"
        else:
            cast = "cool"
    return {
        "lightness": tri(float(row["dL"]), NEAR_ZERO["dL"], "up", "down"),
        "saturation": tri(float(row["dSat"]), NEAR_ZERO["dSat"], "up", "down"),
        "cast": cast,
        "shadows": tri(
            -float(row.get("d_shadow", 0.0)), NEAR_ZERO["d_shadow"], "lift", "deepen"
        ),
        "highlights": tri(
            float(row.get("d_high", 0.0)), NEAR_ZERO["d_high"], "lift", "deepen"
        ),
    }


def expected_segment_dl(row: dict[str, Any]) -> dict[str, str]:
    """Supplementary reference: the row's own shadow_dL / highlight_dL L* shifts."""
    def tri(value: float) -> str:
        if value > NEAR_ZERO["dL"]:
            return "lift"
        if value < -NEAR_ZERO["dL"]:
            return "deepen"
        return "none"

    return {
        "shadows": tri(float(row["shadow_dL"])),
        "highlights": tri(float(row["highlight_dL"])),
    }


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
                "model": str(getattr(response, "model", "") or ""),
                "usage": {
                    "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
                    "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
                },
            }
        except Exception as exc:  # retried below; the last one is re-raised
            last = exc
    raise RuntimeError(f"probe failed after {ATTEMPTS} attempts: {last}")


# --- markdown -----------------------------------------------------------------------
def _cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    slots = list(DIRECTION_SLOTS)
    lines.append(
        f"抽样:`random.Random({payload['seed']})` 从第 2 步入表提案里抽 "
        f"{payload['size']} 条。endpoint `{payload['lane']}` / model "
        f"`{payload['model']}` / effort `{payload['reasoning_effort']}` / "
        f"temperature {payload['temperature']} / transport `{payload['transport']}`。\n"
    )
    lines.append("### 对错矩阵(√ = 与预注册映射同值)\n")
    lines.append(
        "| # | source_id | preset_id | " + " | ".join(slots) + " | 正确槽数 |"
    )
    lines.append("| " + " --- |" * (len(slots) + 4))
    for index, row in enumerate(payload["results"], 1):
        marks = [
            ("√" if row["correct"][slot] else "×")
            + f" {row['given'][slot]}/{row['expected'][slot]}"
            for slot in slots
        ]
        lines.append("| " + " | ".join(
            [str(index), _cell(row["source_id"]), _cell(row["preset_id"]), *marks,
             str(sum(int(row["correct"][slot]) for slot in slots))]
        ) + " |")
    lines.append("")
    lines.append("单元格写法:`√/× 模型答/预注册映射值`。\n")
    lines.append("### 逐槽正确率\n")
    lines.append("| 槽 | n | 正确 | 正确率 |")
    lines.append("| --- | --- | --- | --- |")
    for slot in slots:
        cell = payload["per_slot"][slot]
        lines.append(
            f"| {slot} | {cell['n']} | {cell['n_correct']} | "
            + ("" if cell["rate"] is None else f"{cell['rate']:.4f}") + " |"
        )
    lines.append("")
    lines.append("### 抽到的 10 行里,预注册映射值本身的分布\n")
    lines.append("| 槽 | 取值分布 |")
    lines.append("| --- | --- |")
    for slot in slots:
        tally: dict[str, int] = {}
        for row in payload["results"]:
            key = row["expected"][slot]
            tally[key] = tally.get(key, 0) + 1
        lines.append(
            f"| {slot} | "
            + ", ".join(f"{key} {tally[key]}" for key in sorted(tally)) + " |"
        )
    lines.append("")
    lines.append(
        "补充参照(不替换上表):shadows / highlights 改用该行自己的 "
        "`shadow_dL` / `highlight_dL`(阈值 ±0.5 L*)判分。\n"
    )
    lines.append("| 槽 | n | 正确 | 正确率 |")
    lines.append("| --- | --- | --- | --- |")
    for slot in ("shadows", "highlights"):
        cell = payload["per_slot_segment_dl_reference"][slot]
        lines.append(
            f"| {slot} | {cell['n']} | {cell['n_correct']} | "
            + ("" if cell["rate"] is None else f"{cell['rate']:.4f}") + " |"
        )
    lines.append("")
    lines.append("### 10 份 `per_column` 解释原文\n")
    for index, row in enumerate(payload["results"], 1):
        lines.append(
            f"#### {index}. `{row['preset_id']}` (source `{row['source_id']}`, "
            f"row {row['row_index']})\n"
        )
        lines.append("行(逐字):\n")
        lines.append("```\n" + row["row_text"] + "\n```\n")
        lines.append("| 列 | 模型原文 |")
        lines.append("| --- | --- |")
        for key in PER_COLUMN_KEYS:
            lines.append(f"| {key} | {_cell(row['per_column'].get(key, ''))} |")
        lines.append("")
        lines.append("`expected_visual_effect`:\n")
        lines.append("> " + row["expected_visual_effect"].replace("\n", " ") + "\n")
    return "\n".join(lines)


# --- driver -------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--md", type=Path, default=None)
    parser.add_argument("--size", type=int, default=SAMPLE_SIZE)
    args = parser.parse_args()

    ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
    entries = ledger["entries"]
    sample = random.Random(SAMPLE_SEED).sample(entries, min(args.size, len(entries)))

    base_url, api_key, model = lane_credentials(CREDENTIALS, LANE)
    client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0)

    results = []
    for entry in sample:
        row = entry["nominal"]
        prompt = (
            INSTRUCTION + "\n\n=== shortlist header (verbatim) ===\n"
            + entry["shortlist_header"]
            + "\n\n=== the row (verbatim) ===\n" + row["row_text"] + "\n"
        )
        answer = ask(client, model, prompt)
        parsed = answer["parsed"]
        given = {slot: str(parsed["directions"][slot]) for slot in DIRECTION_SLOTS}
        expected = expected_directions(row)
        segment = expected_segment_dl(row)
        results.append({
            "source_id": entry["source_id"], "preset_id": entry["preset_id"],
            "branch_id": entry["branch_id"], "row_index": entry["row_index"],
            "row_text": row["row_text"],
            "numbers": {key: row[key] for key in PER_COLUMN_KEYS if key in row},
            "expected": expected, "given": given,
            "correct": {slot: given[slot] == expected[slot] for slot in DIRECTION_SLOTS},
            "expected_segment_dl": segment,
            "correct_segment_dl": {
                slot: given[slot] == segment[slot] for slot in segment
            },
            "per_column": parsed["per_column"],
            "expected_visual_effect": parsed["expected_visual_effect"],
            "usage": answer["usage"], "model": answer["model"],
        })

    per_slot = {
        slot: {
            "n": len(results),
            "n_correct": sum(int(row["correct"][slot]) for row in results),
            "rate": round(
                sum(int(row["correct"][slot]) for row in results) / len(results), 4
            ) if results else None,
        }
        for slot in DIRECTION_SLOTS
    }
    per_slot_segment = {
        slot: {
            "n": len(results),
            "n_correct": sum(int(row["correct_segment_dl"][slot]) for row in results),
            "rate": round(
                sum(int(row["correct_segment_dl"][slot]) for row in results)
                / len(results), 4
            ) if results else None,
        }
        for slot in ("shadows", "highlights")
    }
    payload = {
        "seed": SAMPLE_SEED, "size": len(results), "model": model, "lane": LANE,
        "reasoning_effort": REASONING_EFFORT, "temperature": TEMPERATURE,
        "transport": "nonstream", "near_zero": NEAR_ZERO,
        "per_slot": per_slot,
        "per_slot_segment_dl_reference": per_slot_segment,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=1) + "\n",
        encoding="utf-8",
    )
    if args.md is not None:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(markdown(payload), encoding="utf-8")
    print(json.dumps({
        "size": len(results), "per_slot": per_slot,
        "per_slot_segment_dl_reference": per_slot_segment, "out": str(args.out),
    }, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
