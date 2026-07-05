#!/usr/bin/env python3
"""Summarize Photographer-IAA metric JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Photographer-IAA metrics.")
    parser.add_argument("metrics", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    args = parse_args()
    rows = []
    for path in args.metrics:
        with path.expanduser().open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        overall = data.get("overall", {})
        coverage = data.get("coverage", {})
        rows.append(
            {
                "model": data.get("model_name", path.stem),
                "n": overall.get("n"),
                "coverage": coverage.get("matched_rows"),
                "plcc": overall.get("plcc"),
                "srcc": overall.get("srcc"),
                "krcc": overall.get("krcc"),
                "rmse": overall.get("rmse"),
                "mae": overall.get("mae"),
                "bias": overall.get("bias"),
                "pred_mean": overall.get("pred_mean"),
            }
        )

    headers = ["model", "n", "coverage", "plcc", "srcc", "krcc", "rmse", "mae", "bias", "pred_mean"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(header)) for header in headers) + " |")
    text = "\n".join(lines) + "\n"
    if args.output:
        args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.output.expanduser().write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
