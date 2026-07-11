#!/usr/bin/env python3
"""Mix two prediction JSONLs on shared benchmark_id: w_a * A + w_b * B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_scores(path: Path) -> dict[str, float]:
    scores = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            score = row.get("pred_score_0_100")
            if score is not None:
                scores[str(row["benchmark_id"])] = float(score)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-a", type=Path, required=True)
    parser.add_argument("--pred-b", type=Path, required=True)
    parser.add_argument("--weight-a", type=float, default=0.75)
    parser.add_argument("--weight-b", type=float, default=0.25)
    parser.add_argument("--model-name", default="mixed")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    a, b = load_scores(args.pred_a), load_scores(args.pred_b)
    shared = sorted(set(a) & set(b))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for bid in shared:
            out.write(json.dumps({
                "benchmark_id": bid,
                "pred_score_0_100": args.weight_a * a[bid] + args.weight_b * b[bid],
                "score_a": a[bid],
                "score_b": b[bid],
                "weight_a": args.weight_a,
                "weight_b": args.weight_b,
                "model": args.model_name,
            }, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"a={len(a)} b={len(b)} mixed={len(shared)} -> {args.output}")


if __name__ == "__main__":
    main()
