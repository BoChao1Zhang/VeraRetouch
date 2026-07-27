#!/usr/bin/env python3
"""Pack and verify cold datasets as indexed, uncompressed USTAR shards."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import time
from pathlib import Path

from dataset_build.tools.indexed_tar import (
    DEFAULT_PREFETCH_BYTES,
    DEFAULT_PREFETCH_FILES,
    DEFAULT_READ_WORKERS,
    DEFAULT_SHARD_SIZE,
    IndexedTarError,
    build_indexed_tar,
    verify_dataset,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    pack = subcommands.add_parser("pack", help="build and atomically publish a shard dataset")
    inputs = pack.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--source", type=Path, help="archive this directory tree as-is")
    inputs.add_argument(
        "--plan",
        type=Path,
        help="archive the files named by this plan JSONL, using its logical paths",
    )
    pack.add_argument("--output", type=Path, required=True)
    pack.add_argument(
        "--shard-size-gib",
        type=float,
        default=DEFAULT_SHARD_SIZE / 1024**3,
        help="target shard size in GiB; the final shard may be smaller (default: 2)",
    )
    pack.add_argument("--read-workers", type=int, default=DEFAULT_READ_WORKERS)
    pack.add_argument("--prefetch-files", type=int, default=DEFAULT_PREFETCH_FILES)
    pack.add_argument("--prefetch-mib", type=int, default=DEFAULT_PREFETCH_BYTES // 1024**2)
    verify = subcommands.add_parser("verify", help="fully verify a published shard dataset")
    verify.add_argument("--dataset", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    last_staging: str | None = None
    try:
        if args.command == "verify":
            print(json.dumps(verify_dataset(args.dataset), sort_keys=True))
            return 0

        if not math.isfinite(args.shard_size_gib) or args.shard_size_gib <= 0:
            raise IndexedTarError("--shard-size-gib must be finite and positive")
        shard_size = int(args.shard_size_gib * 1024**3)
        started = time.monotonic()
        last_report = 0.0

        def report(state) -> None:
            nonlocal last_report, last_staging
            last_staging = str(state["staging"])
            now = time.monotonic()
            if state["members"] == 1 or now - last_report >= 10:
                elapsed = max(now - started, 1e-6)
                mib = state["payload_bytes"] / 1024**2
                print(
                    f"[pack] files={state['members']} payload={mib:.1f} MiB "
                    f"rate={mib / elapsed:.1f} MiB/s shard={state['shard']}",
                    file=sys.stderr,
                    flush=True,
                )
                last_report = now

        manifest = build_indexed_tar(
            args.source,
            args.output,
            plan=args.plan,
            shard_size_bytes=shard_size,
            read_workers=args.read_workers,
            prefetch_files=args.prefetch_files,
            prefetch_bytes=args.prefetch_mib * 1024**2,
            progress=report,
        )
        print(json.dumps({
            "dataset": str(args.output),
            "dataset_id": manifest["dataset_id"],
            "members": manifest["member_count"],
            "samples": manifest["sample_count"],
            "payload_bytes": manifest["payload_bytes"],
            "shards": manifest["shard_count"],
            "status": manifest["status"],
        }, sort_keys=True))
        return 0
    except KeyboardInterrupt:
        detail = f"; unpublished staging retained at {last_staging}" if last_staging else ""
        print(f"interrupted{detail}", file=sys.stderr)
        return 130
    except (IndexedTarError, OSError, sqlite3.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
