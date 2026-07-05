#!/usr/bin/env python3
"""Call a local vLLM OpenAI-compatible vision endpoint for Photographer-IAA."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import re
import time
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score Photographer-IAA through vLLM OpenAI API.")
    parser.add_argument("--input", type=Path, required=True, help="vllm_requests.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="Prediction JSONL.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--sleep", type=float, default=0.0)
    return parser.parse_args()


def load_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def seen_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                seen.add(str(json.loads(line)["benchmark_id"]))
            except (KeyError, json.JSONDecodeError):
                continue
    return seen


def image_data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as fh:
        encoded = base64.b64encode(fh.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def parse_score(text: str) -> float | None:
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text)
    if not match:
        return None
    value = float(match.group(0))
    if value <= 1.5:
        value *= 100.0
    elif value <= 5.0:
        value = (value - 1.0) / 4.0 * 100.0
    return max(0.0, min(100.0, value))


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input.expanduser(), args.limit)
    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    done = seen_ids(args.output.expanduser()) if args.resume else set()
    mode = "a" if args.resume else "w"
    url = args.base_url.rstrip("/") + "/chat/completions"

    with args.output.expanduser().open(mode, encoding="utf-8") as out:
        for row in tqdm(rows, desc=f"vLLM {args.model}"):
            benchmark_id = str(row["benchmark_id"])
            if benchmark_id in done:
                continue
            try:
                payload = {
                    "model": args.model,
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": row["prompt"]},
                                {"type": "image_url", "image_url": {"url": image_data_url(row["image_path"])}},
                            ],
                        }
                    ],
                }
                response = requests.post(url, json=payload, timeout=args.timeout)
                response.raise_for_status()
                data = response.json()
                text = data["choices"][0]["message"]["content"]
                score = parse_score(text)
                result = {
                    "benchmark_id": benchmark_id,
                    "raw_response": text,
                    "model": args.model,
                }
                if score is None:
                    result["error"] = "no_numeric_score"
                else:
                    result["pred_score_0_100"] = score
            except Exception as exc:
                result = {
                    "benchmark_id": benchmark_id,
                    "error": repr(exc),
                    "model": args.model,
                }
            out.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            out.flush()
            if args.sleep:
                time.sleep(args.sleep)


if __name__ == "__main__":
    main()
