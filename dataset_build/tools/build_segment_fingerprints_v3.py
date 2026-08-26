"""EPR-043: derive the v3 segment-fingerprint artifact (v2 + reliability mask).

v3 keeps every v2 field byte for byte and adds two groups per row:
`segments_strict` (measured cells, `null` everywhere else) and `reliability`
(the origin tag of each cell). No fallback number is ever stored in a cell that a
consumer could read as a measurement -- the R4-G4-L4 spec §4.3 red line.

The measured histogram group of v2 is copied, not recomputed, so this script needs no
LUT rendering: it reads the frozen v2 table plus the closed-v1 annotations it was
derived from and is deterministic per (v2 SHA, annotations SHA).

Usage::

    python -m dataset_build.tools.build_segment_fingerprints_v3 \
        --out /home/bc/data/scratch/lut_reannotate/out/segment_fingerprints.v3.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_build.agent_loop.segment_fingerprints import (  # noqa: E402
    DEFAULT_ANNOTATIONS, DEFAULT_FINGERPRINTS_V2, DEFAULT_FINGERPRINTS_V3,
    REGISTERED_SEGMENT_FINGERPRINT_TABLES, build_segment_fingerprints_v3,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Derive the v3 segment fingerprints")
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--fingerprints-v2", type=Path, default=DEFAULT_FINGERPRINTS_V2)
    parser.add_argument("--out", type=Path, default=DEFAULT_FINGERPRINTS_V3)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--no-backup", action="store_true",
        help="overwrite an existing output instead of moving it to .bak first",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = build_segment_fingerprints_v3(
        args.annotations, args.fingerprints_v2, args.out,
        report=args.report, backup=not args.no_backup,
    )
    sha = report["output"]["sha256"]
    print(json.dumps({
        "output": report["output"],
        "assertions": report["assertions"],
        "reliability": report["reliability"],
        "registered": sha in REGISTERED_SEGMENT_FINGERPRINT_TABLES,
    }, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
