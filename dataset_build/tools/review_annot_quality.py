"""Per-sample annotation-quality review via relay gpt-5.6-sol (xhigh).

Usage:
  python -m dataset_build.tools.review_annot_quality --groups g001_xxx,g002_yyy
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import time
import tomllib
from pathlib import Path

from PIL import Image

REVIEW_ROOT = Path("/var/cache/veradata/annot_review/eval100-annotqa-20260727")
CONFIG = Path("/home/bc/VeraRetouch/databuild.eval100.toml")
DIMENSIONS = ("consistency", "reasoning", "leakage", "language", "usability")

RUBRIC = """You are auditing training annotations for a photo-retouching SFT dataset.
Given BEFORE image, AFTER image (the edit result), optionally the C_GT mask (white =
edited region, local tasks only), and the ANNOTATION (ONLY instruction, instruction_short, reasoning) plus a CONTEXT block
(task_type/style_name/subject/region/slot_mode — dataset metadata for your reference,
NOT part of the annotation; never count context fields as leakage),
score each dimension 1-5 (5 best) with a one-sentence reason:
1. consistency: instruction matches the actual before->after change (direction, region,
   magnitude). style task MUST name the style (Chinese name verbatim is fine); local task
   must NOT name any preset/style and its region wording must match the C_GT location.
2. reasoning: the token sections are present and ordered (light, global_color,
   specific_color problems, then region_scope if the sample has one, then the three
   plans); problems describe real BEFORE issues; plans match
   the applied edit; directions must not be inverted.
3. leakage: judge ONLY the annotation text (instruction/instruction_short/reasoning):
   no LR parameter keys, numeric deltas, IAA/q scores, mask/geometry/alpha, degrade/auto,
   slot info inside those three fields. 5 = no leak, 1 = severe leak.
4. language: natural English (style name exception), user-like instruction, long/short
   consistent, short is a distillation not a truncation.
5. usability: overall fitness as an SFT training sample (5 pass / 3 borderline / 1 fail).
Return strict JSON only."""

SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "object",
            "properties": {d: {"type": "integer", "minimum": 1, "maximum": 5} for d in DIMENSIONS},
            "required": list(DIMENSIONS),
            "additionalProperties": False,
        },
        "reasons": {
            "type": "object",
            "properties": {d: {"type": "string", "minLength": 8} for d in DIMENSIONS},
            "required": list(DIMENSIONS),
            "additionalProperties": False,
        },
        "verdict": {"type": "string", "enum": ["pass", "borderline", "fail"]},
    },
    "required": ["scores", "reasons", "verdict"],
    "additionalProperties": False,
}


def normalize_result(result: dict) -> dict:
    """Normalize the alternate per-dimension shape emitted by some relay lanes."""
    if "scores" in result and "reasons" in result:
        normalized = result
    elif all(
        isinstance(result.get(dimension), dict)
        and "score" in result[dimension]
        and "reason" in result[dimension]
        for dimension in DIMENSIONS
    ):
        scores = {dimension: result[dimension]["score"] for dimension in DIMENSIONS}
        reasons = {dimension: result[dimension]["reason"] for dimension in DIMENSIONS}
        usability = scores["usability"]
        verdict = "pass" if usability >= 4 else "borderline" if usability == 3 else "fail"
        normalized = {"scores": scores, "reasons": reasons, "verdict": verdict}
    else:
        raise ValueError("review response did not match a supported score schema")

    if set(normalized.get("scores", {})) != set(DIMENSIONS):
        raise ValueError("review response score dimensions are incomplete")
    if set(normalized.get("reasons", {})) != set(DIMENSIONS):
        raise ValueError("review response reason dimensions are incomplete")
    if any(
        not isinstance(score, int) or isinstance(score, bool) or not 1 <= score <= 5
        for score in normalized["scores"].values()
    ):
        raise ValueError("review response scores must be integers from 1 to 5")
    if any(not isinstance(reason, str) or len(reason) < 8 for reason in normalized["reasons"].values()):
        raise ValueError("review response reasons are invalid")
    if normalized.get("verdict") not in {"pass", "borderline", "fail"}:
        raise ValueError("review response verdict is invalid")
    return normalized


def data_url(path: Path, long_edge: int = 512, quality: int = 85) -> str:
    with Image.open(path) as image:
        image.load()
        rgb = image.convert("RGB")
    scale = long_edge / max(rgb.width, rgb.height)
    if scale < 1:
        rgb = rgb.resize((round(rgb.width * scale), round(rgb.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def clients() -> list:
    from openai import OpenAI

    cfg = tomllib.load(CONFIG.open("rb"))
    return [
        OpenAI(base_url=ep["base_url"], api_key=ep["api_key"], max_retries=0, timeout=300)
        for ep in cfg["annotation"]["external_endpoints"]
    ]


def review_sample(client, model: str, effort: str, gdir: Path, sample: dict) -> dict:
    rank = sample.get("rank")
    annotation = {k: sample.get(k) for k in ("instruction", "instruction_short", "reasoning")}
    context = {k: sample.get(k) for k in ("task_type", "style_name", "subject", "region", "slot_mode")}
    content = [{"type": "input_text", "text": RUBRIC + "\nANNOTATION:\n"
                + json.dumps(annotation, ensure_ascii=False)
                + "\nCONTEXT (not part of the annotation):\n"
                + json.dumps(context, ensure_ascii=False)
                + "\nImages order: BEFORE, AFTER" }]
    content.append({"type": "input_image", "image_url": data_url(gdir / "before.jpg")})
    content.append({"type": "input_image", "image_url": data_url(gdir / f"after_rank{rank}.jpg")})
    cgt = gdir / f"cgt_rank{rank}.png"
    if cgt.is_file():
        content[0]["text"] += ", C_GT"
        content.append({"type": "input_image", "image_url": data_url(cgt)})
    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": content}],
        reasoning={"effort": effort},
        max_output_tokens=4000,
        text={"format": {"type": "json_schema", "name": "review", "strict": True,
                         "schema": SCHEMA}},
    )
    if response.status != "completed":
        raise RuntimeError(f"status={response.status}")
    return normalize_result(json.loads(response.output_text))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", required=True)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--effort", default="xhigh")
    parser.add_argument("--review-root", type=Path, default=REVIEW_ROOT)
    args = parser.parse_args()
    pool = clients()
    call = 0
    for name in args.groups.split(","):
        gdir = args.review_root / name
        entry = json.loads((gdir / "annot.json").read_text())
        results = []
        for sample in entry["samples"]:
            last = None
            for attempt in range(3):
                client = pool[(call + attempt) % len(pool)]
                try:
                    result = review_sample(client, args.model, args.effort, gdir, sample)
                    break
                except Exception as exc:  # noqa: BLE001 - transport retry
                    last = f"{type(exc).__name__}: {exc}"
                    time.sleep(2 * (attempt + 1))
            else:
                result = {"error": last}
            call += 1
            results.append({"sft_id": sample["sft_id"], "rank": sample.get("rank"), **result})
        scored = [r for r in results if "scores" in r]
        worst = min(scored, key=lambda r: (sum(r["scores"].values()), r["scores"]["consistency"]))["sft_id"] if scored else None
        verdict = {"group_id": entry["group_id"], "render_mode": entry.get("render_mode"),
                   "samples": results, "worst_sft_id": worst}
        (gdir / "verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2))
        if worst:
            wdir = args.review_root / "worst" / name
            wdir.mkdir(parents=True, exist_ok=True)
            worst_sample = next(s for s in entry["samples"] if s["sft_id"] == worst)
            rank = worst_sample.get("rank")
            for src in ("before.jpg", f"after_rank{rank}.jpg", f"cgt_rank{rank}.png"):
                if (gdir / src).is_file():
                    (wdir / src).write_bytes((gdir / src).read_bytes())
            (wdir / "sample.json").write_text(json.dumps(worst_sample, ensure_ascii=False, indent=2))
            (wdir / "verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2))
        print(json.dumps({"group": name, "samples": len(results),
                          "errors": sum(1 for r in results if "error" in r),
                          "worst": worst}))


if __name__ == "__main__":
    main()
