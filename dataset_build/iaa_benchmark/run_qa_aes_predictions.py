#!/usr/bin/env python3
"""QA-AES questionnaire baseline: run source_qa AES questionnaire on benchmark images.

pred_score_0_100 = merit_frac * 100, only for reliable records.
Mapping rule: questionnaire runs directly on metadata.csv image paths, keyed by
benchmark_id — no DB join. max_face_frac is unavailable for benchmark images, so
every image uses the no-face AES questionnaire variant.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "dataset_build"))

from source_qa import qa_runner  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-path-column", default="image_path")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.metadata.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[: args.limit]

    done = set()
    if args.resume and args.output.exists():
        with args.output.open("r", encoding="utf-8") as fh:
            done = {json.loads(l)["benchmark_id"] for l in fh if l.strip()}
    rows = [r for r in rows if r["benchmark_id"] not in done]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def work(row: dict) -> dict:
        payload = {"benchmark_id": row["benchmark_id"], "model": "QA-AES-merit-frac"}
        try:
            out = qa_runner.run_aes({"path": row[args.image_path_column]})
            payload.update(
                reliable=bool(out["reliable"]),
                reason=out["reason"],
                merit_count=out["merit_count"],
                merit_n=out["merit_n"],
                merit_frac=out["merit_frac"],
                reask_count=out.get("reask_count"),
            )
            if out["reliable"] and out["merit_frac"] is not None:
                payload["pred_score_0_100"] = out["merit_frac"] * 100.0
        except Exception as exc:  # noqa: BLE001
            payload["error"] = repr(exc)
        return payload

    n_ok = 0
    with args.output.open("a" if args.resume else "w", encoding="utf-8") as out_fh:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(work, row) for row in rows]
            for i, fut in enumerate(as_completed(futures), 1):
                payload = fut.result()
                with lock:
                    out_fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
                    out_fh.flush()
                if "pred_score_0_100" in payload:
                    n_ok += 1
                if i % 100 == 0:
                    print(f"{i}/{len(rows)} scored={n_ok}", flush=True)
    print(f"done: {len(rows)} processed, {n_ok} scored -> {args.output}")


if __name__ == "__main__":
    main()
