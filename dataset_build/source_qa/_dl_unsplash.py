"""One-off: multithreaded download of the Unsplash-Lite research dataset images.

The zip ships only CSV metadata (URLs, no pixels). We read photos.csv000 (TSV)
and download each photo_image_url at a bounded size to a scratch dir the registry
can scan. Resumable (skips existing non-empty files).

Run: python -m dataset_build.source_qa._dl_unsplash [--workers 32] [--width 2048] [--limit N]
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

ZIP = "/home/bc/data/datasets/unsplash-research-dataset-lite-latest.zip"
DST = "/home/bc/data/datasets/_scratch/unsplash"
_LOCAL = threading.local()


def _session():
    if not hasattr(_LOCAL, "s"):
        import requests
        _LOCAL.s = requests.Session()
    return _LOCAL.s


def _rows():
    data = subprocess.run(["7z", "e", "-so", ZIP, "photos.csv000"],
                          capture_output=True).stdout.decode("utf-8", "ignore")
    return list(csv.DictReader(io.StringIO(data), delimiter="\t"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--width", type=int, default=2048)
    ap.add_argument("--quality", type=int, default=80)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(DST, exist_ok=True)

    rows = _rows()
    if args.limit:
        rows = rows[:args.limit]
    print(f"[unsplash] {len(rows)} photos to fetch -> {DST}", file=sys.stderr)
    counts = {"ok": 0, "skip": 0, "err": 0}
    lock = threading.Lock()

    def dl(row):
        pid = row.get("photo_id")
        url = row.get("photo_image_url")
        if not pid or not url:
            return "err"
        out = os.path.join(DST, f"{pid}.jpg")
        if os.path.exists(out) and os.path.getsize(out) > 0:
            return "skip"
        full = f"{url}?fm=jpg&w={args.width}&q={args.quality}&fit=max"
        try:
            r = _session().get(full, timeout=40)
            r.raise_for_status()
            tmp = out + ".part"
            with open(tmp, "wb") as f:
                f.write(r.content)
            os.replace(tmp, out)
            return "ok"
        except Exception:
            return "err"

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, res in enumerate(ex.map(dl, rows)):
            with loc:
                counts[res] += 1
            if (i + 1) % 500 == 0:
                print(f"[unsplash] {i+1}/{len(rows)} {counts}", file=sys.stderr)
    print(f"[unsplash] DONE {counts}")


if __name__ == "__main__":
    main()
