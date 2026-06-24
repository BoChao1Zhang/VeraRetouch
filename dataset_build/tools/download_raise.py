#!/usr/bin/env python3
"""Multithreaded downloader for the RAISE-6k dataset.

RAISE ships only a CSV of per-image direct download URLs (NEF RAW + TIFF). This
tool reads that CSV (from the zipped ``RAISE_6k.csv.zip`` or a plain ``.csv``),
then fans out a ThreadPoolExecutor to fetch every file. Downloads are resumable
and idempotent: a finished file is skipped, a partial ``.part`` is restarted,
and each fetch retries with backoff before giving up.

Files keep their original RAISE id (e.g. ``r000da54ft.NEF``); the sequential
``RAISE-6k_NNNNNN`` renaming is done later by ``migrate_datasets.py`` so that
all corpora share one renaming/manifest convention.

Usage:
    python -m dataset_build.tools.download_raise --limit 20            # smoke test
    python -m dataset_build.tools.download_raise                        # full NEF pull (~110GB)
    python dataset_build/tools/download_raise.py --workers 12

The download host (193.205.194.113) is an old Apache server; keep workers
modest (8-16) to stay polite and avoid throttling.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import requests

DEFAULT_CSV = "/home/bc/data/datasets/RAISE_6k.csv.zip"
DEFAULT_OUT = "/home/bc/data/datasets/RAISE-6k/raw"
CHUNK = 1 << 20  # 1 MiB streaming chunks
# Column names in the RAISE CSV header that hold the direct-download URLs.
URL_COLUMN = {"nef": "NEF", "tiff": "TIFF"}
EXT = {"nef": ".NEF", "tiff": ".TIF"}


@dataclass
class Item:
    file_id: str
    url: str
    dest: str


def _read_csv_bytes(csv_path: str) -> str:
    """Return the CSV text from a .zip (first .csv member) or a plain .csv."""
    if csv_path.lower().endswith(".zip"):
        with zipfile.ZipFile(csv_path) as zf:
            members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not members:
                raise SystemExit(f"no .csv inside {csv_path}: {zf.namelist()}")
            with zf.open(members[0]) as fh:
                return io.TextIOWrapper(fh, encoding="utf-8", errors="replace").read()
    with open(csv_path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def parse_items(csv_path: str, out_dir: str, fmt: str, limit: int | None) -> list[Item]:
    col = URL_COLUMN[fmt]
    ext = EXT[fmt]
    text = _read_csv_bytes(csv_path)
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None or col not in reader.fieldnames:
        raise SystemExit(f"CSV has no '{col}' column; columns={reader.fieldnames}")
    items: list[Item] = []
    for row in reader:
        url = (row.get(col) or "").strip()
        file_id = (row.get("File") or "").strip()
        if not url:
            continue
        if not file_id:
            # Fall back to the URL basename without extension.
            file_id = os.path.splitext(os.path.basename(url))[0]
        items.append(Item(file_id=file_id, url=url, dest=os.path.join(out_dir, file_id + ext)))
        if limit and len(items) >= limit:
            break
    return items


def _remote_size(session: requests.Session, url: str, timeout: float) -> int | None:
    """Best-effort Content-Length via HEAD; None if the server won't say."""
    try:
        r = session.head(url, timeout=timeout, allow_redirects=True)
        if r.ok:
            cl = r.headers.get("Content-Length")
            if cl is not None:
                return int(cl)
    except requests.RequestException:
        pass
    return None


def _already_done(dest: str, expected: int | None) -> bool:
    if not os.path.exists(dest):
        return False
    size = os.path.getsize(dest)
    if size <= 0:
        return False
    if expected is not None:
        return size == expected
    return True  # exists and non-empty, no remote size to compare


def download_one(item: Item, retries: int, timeout: float, verify_size: bool) -> tuple[str, str, int]:
    """Returns (file_id, status, bytes). status in {done, skip, fail}."""
    session = requests.Session()
    session.headers["User-Agent"] = "VeraRetouch-RAISE-downloader/1.0"
    expected = _remote_size(session, item.url, timeout) if verify_size else None

    if _already_done(item.dest, expected):
        return (item.file_id, "skip", os.path.getsize(item.dest))

    part = item.dest + ".part"
    last_err = ""
    for attempt in range(1, retries + 1):
        try:
            with session.get(item.url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                written = 0
                with open(part, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=CHUNK):
                        if chunk:
                            fh.write(chunk)
                            written += len(chunk)
            if expected is not None and written != expected:
                raise IOError(f"size mismatch: got {written} want {expected}")
            if written <= 0:
                raise IOError("empty download")
            os.replace(part, item.dest)
            return (item.file_id, "done", written)
        except (requests.RequestException, IOError, OSError) as exc:
            last_err = str(exc)
            try:
                if os.path.exists(part):
                    os.remove(part)
            except OSError:
                pass
            if attempt < retries:
                time.sleep(min(2 ** attempt, 30))
    return (item.file_id, f"fail:{last_err}", 0)


def _check_free_space(out_dir: str, min_free_gb: float) -> None:
    st = os.statvfs(out_dir)
    free_gb = st.f_bavail * st.f_frsize / (1 << 30)
    if free_gb < min_free_gb:
        raise SystemExit(
            f"insufficient free space at {out_dir}: {free_gb:.0f}G free, "
            f"need >= {min_free_gb:.0f}G (use --min-free-gb to override)"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Multithreaded RAISE-6k downloader (NEF RAW by default).")
    ap.add_argument("--csv", default=DEFAULT_CSV, help="RAISE CSV (.zip or .csv) with NEF/TIFF URL columns")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output directory for downloaded files")
    ap.add_argument("--format", choices=["nef", "tiff"], default="nef", help="which column to fetch")
    ap.add_argument("--workers", type=int, default=12, help="concurrent download threads (keep 8-16)")
    ap.add_argument("--limit", type=int, default=0, help="only fetch the first N items (0 = all)")
    ap.add_argument("--retries", type=int, default=4, help="retry attempts per file")
    ap.add_argument("--timeout", type=float, default=60.0, help="per-request timeout seconds")
    ap.add_argument("--verify-size", action="store_true",
                    help="HEAD each URL and verify Content-Length (slower, stricter resume)")
    ap.add_argument("--min-free-gb", type=float, default=150.0, help="abort if free space below this")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    _check_free_space(args.out, args.min_free_gb)

    items = parse_items(args.csv, args.out, args.format, args.limit or None)
    if not items:
        raise SystemExit("no items parsed from CSV")
    print(f"[raise] {len(items)} {args.format.upper()} files -> {args.out} "
          f"(workers={args.workers}, retries={args.retries})", flush=True)

    counts = {"done": 0, "skip": 0, "fail": 0}
    total_bytes = 0
    failures: list[str] = []
    lock = threading.Lock()
    t0 = time.time()
    n = len(items)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download_one, it, args.retries, args.timeout, args.verify_size): it
                for it in items}
        completed = 0
        for fut in as_completed(futs):
            file_id, status, nbytes = fut.result()
            completed += 1
            with lock:
                key = "fail" if status.startswith("fail") else status
                counts[key] += 1
                total_bytes += nbytes
                if key == "fail":
                    failures.append(f"{file_id}: {status}")
            if completed % 50 == 0 or completed == n:
                elapsed = time.time() - t0
                gb = total_bytes / (1 << 30)
                print(f"[raise] {completed}/{n}  done={counts['done']} skip={counts['skip']} "
                      f"fail={counts['fail']}  {gb:.1f}GB  {elapsed:.0f}s", flush=True)

    print(f"[raise] FINISHED done={counts['done']} skip={counts['skip']} fail={counts['fail']} "
          f"total={total_bytes/(1<<30):.1f}GB in {time.time()-t0:.0f}s", flush=True)
    if failures:
        print(f"[raise] {len(failures)} failures (first 20):", flush=True)
        for line in failures[:20]:
            print(f"  - {line}", flush=True)
        print("[raise] re-run the same command to retry failed/partial files (resumable).", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
