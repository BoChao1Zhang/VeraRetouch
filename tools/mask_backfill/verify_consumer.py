"""Assert that backfilled cache entries pass the real consumer gate.

The producer writes ``subject.json`` / ``subject.png``; whether the build can
*use* them is decided by ``construct.sources._inspect_cache_dir``, which re-reads
``meta["source_path"]`` through the archive-aware ``path_exists`` / ``read_bytes``.
That is exactly the check a prefetch-path leak would fail — an entry whose
``source_path`` recorded the scratch copy looks fine until scratch is cleared.
So the gate is run here verbatim rather than reimplemented.

``read_bytes`` would resolve an archived source by reading its shard off the
*hard* NFS mount, which this campaign must never touch.  The prefetch buffer is
therefore pointed at first, and any entry whose source is neither local nor
buffered is reported as ``skipped_would_touch_nfs`` instead of being inspected.

Usage::

    python tools/mask_backfill/verify_consumer.py --cache-root <dir> [--limit 0]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _path in (_REPO, os.path.join(_REPO, "dataset_build", "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from construct.sources import _inspect_cache_dir  # noqa: E402
from dataset_build.tools.archive_reader import (  # noqa: E402
    prefetch_name, set_prefetch_dir,
)

DEFAULT_BUFFER = "/home/bc/data/scratch/mask_backfill/buffer"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--buffer-dir", default=DEFAULT_BUFFER)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--show", type=int, default=3,
                        help="print this many eligible records in full")
    args = parser.parse_args()

    buffer_dir = Path(args.buffer_dir)
    set_prefetch_dir(buffer_dir if buffer_dir.is_dir() else None)

    entries = sorted(
        entry for entry in Path(args.cache_root).iterdir()
        if entry.is_dir() and not entry.name.startswith("_")
    )
    if args.limit:
        entries = entries[:args.limit]

    reasons: collections.Counter[str] = collections.Counter()
    shown = 0
    for entry in entries:
        try:
            meta = json.loads((entry / "subject.json").read_text(encoding="utf-8"))
        except OSError:
            reasons["missing_subject_json"] += 1
            continue
        source = str(meta.get("source_path") or "")
        reachable = (os.path.exists(source)
                     or (buffer_dir / prefetch_name(source)).exists())
        if not reachable:
            reasons["skipped_would_touch_nfs"] += 1
            print(f"{entry.name}: source neither local nor buffered: {source}")
            continue
        record, reason = _inspect_cache_dir(entry, {}, {})
        reasons[reason] += 1
        if record is not None and shown < args.show:
            shown += 1
            print(f"{entry.name}: {reason}\n  " + json.dumps({
                "source_id": record.source_id,
                "source_path": os.fspath(record.source_path),
                "cache_dir": os.fspath(record.cache_dir),
                "scene": record.scene,
                "mask_area": record.mask_area,
                "subject": record.subject,
            }, ensure_ascii=False))
        elif record is None and reason not in {"subject_not_ready"}:
            print(f"{entry.name}: REJECTED {reason} source={source}")
    print(json.dumps({"entries": len(entries), **dict(sorted(reasons.items()))},
                     ensure_ascii=False))
    return 0 if reasons.get("eligible") else 1


if __name__ == "__main__":
    sys.exit(main())
