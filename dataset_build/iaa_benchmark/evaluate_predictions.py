#!/usr/bin/env python3
"""Evaluate model predictions against Photographer-IAA metadata."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score IAA model predictions on Photographer-IAA.")
    parser.add_argument("--metadata", type=Path, required=True, help="Benchmark metadata.csv.")
    parser.add_argument("--predictions", type=Path, required=True, help="Prediction CSV or JSONL.")
    parser.add_argument("--output", type=Path, required=True, help="Output metrics JSON.")
    parser.add_argument("--model-name", required=True, help="Model name for the report.")
    parser.add_argument(
        "--prediction-key",
        default="pred_score_0_100",
        help="Prediction field in the prediction file.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def read_predictions(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    return read_csv(path)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def ranks(values: list[float]) -> list[float]:
    order = sorted(enumerate(values), key=lambda item: item[1])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and order[j][1] == order[i][1]:
            j += 1
        rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            out[order[k][0]] = rank
        i = j
    return out


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    return pearson(ranks(xs), ranks(ys))


def kendall_tau_b(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    concordant = discordant = ties_x = ties_y = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = (xs[i] > xs[j]) - (xs[i] < xs[j])
            dy = (ys[i] > ys[j]) - (ys[i] < ys[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    den = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    if den == 0:
        return None
    return (concordant - discordant) / den


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    preds = [float(row["pred"]) for row in rows]
    gts = [float(row["gt"]) for row in rows]
    errors = [p - g for p, g in zip(preds, gts)]
    abs_errors = [abs(x) for x in errors]
    sq_errors = [x * x for x in errors]
    return {
        "n": len(rows),
        "plcc": pearson(preds, gts),
        "srcc": spearman(preds, gts),
        "krcc": kendall_tau_b(preds, gts),
        "mae": sum(abs_errors) / len(abs_errors),
        "rmse": math.sqrt(sum(sq_errors) / len(sq_errors)),
        "bias": sum(errors) / len(errors),
        "pred_min": min(preds),
        "pred_mean": sum(preds) / len(preds),
        "pred_max": max(preds),
        "gt_min": min(gts),
        "gt_mean": sum(gts) / len(gts),
        "gt_max": max(gts),
    }


def score(metadata: list[dict[str, str]], predictions: list[dict[str, Any]], prediction_key: str) -> dict[str, Any]:
    by_id = {row["benchmark_id"]: row for row in metadata}
    merged: list[dict[str, Any]] = []
    missing_prediction = []
    unknown_prediction = []
    seen = set()

    for pred in predictions:
        benchmark_id = str(pred.get("benchmark_id", ""))
        if benchmark_id not in by_id:
            unknown_prediction.append(benchmark_id)
            continue
        try:
            pred_score = float(pred[prediction_key])
        except (KeyError, TypeError, ValueError):
            continue
        meta = by_id[benchmark_id]
        merged.append(
            {
                "benchmark_id": benchmark_id,
                "category": meta["category"],
                "primary_mode": meta["primary_mode"],
                "gt": float(meta["gt_aesthetic_mean_0_100"]),
                "pred": pred_score,
            }
        )
        seen.add(benchmark_id)

    for benchmark_id in by_id:
        if benchmark_id not in seen:
            missing_prediction.append(benchmark_id)

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in merged:
        by_category[row["category"]].append(row)
        by_mode[row["primary_mode"]].append(row)

    return {
        "overall": metrics(merged),
        "by_category": {k: metrics(v) for k, v in sorted(by_category.items())},
        "by_mode": {k: metrics(v) for k, v in sorted(by_mode.items())},
        "coverage": {
            "metadata_rows": len(metadata),
            "prediction_rows": len(predictions),
            "matched_rows": len(merged),
            "missing_predictions": len(missing_prediction),
            "unknown_predictions": len(unknown_prediction),
            "missing_prediction_examples": missing_prediction[:20],
            "unknown_prediction_examples": unknown_prediction[:20],
        },
    }


def main() -> None:
    args = parse_args()
    metadata = read_csv(args.metadata.expanduser())
    predictions = read_predictions(args.predictions.expanduser())
    report = score(metadata, predictions, args.prediction_key)
    report["model_name"] = args.model_name
    report["metadata"] = str(args.metadata)
    report["predictions"] = str(args.predictions)
    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    with args.output.expanduser().open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps(report["overall"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
