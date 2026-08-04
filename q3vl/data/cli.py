"""``python -m q3vl.data.cli <stage>`` -- every stage is idempotent and separate.

    scan | geometry | convert | plan | images | records | manifest | verify | all
"""

from __future__ import annotations

import argparse
import json
import sys

from . import config as C
from . import pipeline, verify


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="q3vl.data")
    parser.add_argument("stage", choices=[
        "scan", "geometry", "convert", "plan", "images", "records",
        "manifest", "verify", "all"])
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--drop-low-confidence", action="store_true",
                        help="also filter winner_confidence=low (off by default: "
                             "the frozen split authority includes those rows)")
    parser.add_argument("--quick", action="store_true",
                        help="verify: skip the full tar re-walk")
    parser.add_argument("--out", default=None, help="write the stage result as JSON")
    args = parser.parse_args(argv)

    C.WORK_DIR.mkdir(parents=True, exist_ok=True)
    stages = {
        "scan": lambda: pipeline.stage_scan(),
        "geometry": lambda: pipeline.stage_geometry(workers=args.workers),
        "convert": lambda: pipeline.stage_convert(),
        "plan": lambda: pipeline.stage_plan(drop_low_confidence=args.drop_low_confidence),
        "images": lambda: pipeline.stage_images(workers=args.workers),
        "records": lambda: pipeline.stage_records(),
        "manifest": lambda: pipeline.stage_manifest(),
        "verify": lambda: verify.run_all(quick=args.quick),
    }
    if args.stage == "all":
        result = {name: fn() for name, fn in stages.items()}
    else:
        result = stages[args.stage]()

    blob = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(blob)
    else:
        print(blob[:20000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
