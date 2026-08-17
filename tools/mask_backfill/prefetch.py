"""Stream the archived backfill sources onto local scratch, in shard order.

The 19,699 archived rows of ``backfill_pool.jsonl`` live in 149 uncompressed tar
shards on NFS.  Reading them in pool order is 19,699 random 1.6 MB reads over a
1 GbE link; reading them in ``(root, shard, offset)`` order turns the same work
into one forward pass per shard, which is the whole reason this step exists as a
separate pass instead of being folded into the SAM3 producer.

Two hard constraints shape the implementation:

* **Only ``/mnt/nfs-ro`` may be touched.**  The catalog records every archive
  root under the read-write mount ``/mnt/nfs``, which is a *hard* mount — one
  stall there is unrecoverable.  ``ReadOnlyArchiveReader`` rewrites the root onto
  the soft read-only mount before any descriptor is opened, so no code path in
  this tool can reach the hard mount.
* **A dead NFS must stop the run, not be retried into the ground.**  A soft mount
  surfaces an outage as an exception per read, so a consecutive-failure ceiling
  aborts instead of burning through the remaining pool.

Two names are written for each fetched image, hardlinked to one inode:

* ``<scratch>/<path_key>.<ext>`` — the ``read_path`` the pool row already
  declares, what the SAM3 producer opens.
* ``<scratch>/../buffer/<sha256(source_path)>`` — the layout
  ``archive_reader.set_prefetch_dir()`` expects, so *archive-aware* consumers
  (notably ``construct.sources._inspect_cache_dir``, which re-decodes the source
  through ``read_bytes``) can also be satisfied from scratch instead of falling
  through to the hard mount.  Hardlinks, so the second name costs no bytes.

Usage::

    python tools/mask_backfill/prefetch.py --pool backfill_pool.jsonl [--limit 12]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from dataset_build.tools.archive_reader import ArchiveReader, prefetch_name  # noqa: E402

NFS_RW = "/mnt/nfs/"
NFS_RO = "/mnt/nfs-ro/"


def ro_root(root: str) -> str:
    root = str(root)
    return NFS_RO + root[len(NFS_RW):] if root.startswith(NFS_RW) else root


class ReadOnlyArchiveReader(ArchiveReader):
    """``ArchiveReader`` pinned to the soft read-only mount.

    Only the descriptor factory needs overriding: it is the single place the
    catalog's root string turns into an ``open()``, so rewriting it here covers
    both the manifest read and the shard ``pread``.
    """

    def _shard_descriptor(self, root: str, shard: str) -> int:
        return super()._shard_descriptor(ro_root(root), shard)


def _pool_rows(pool: str) -> list[dict]:
    with open(pool, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _is_done(path: str) -> bool:
    try:
        return os.path.getsize(path) > 0
    except OSError:
        return False


def _link_buffer(rows: list[dict], buffer_dir: Path | None) -> int:
    """Give every fetched file its ``archive_reader`` prefetch-buffer name.

    Separate from the fetch so a resumed run repairs a buffer that was disabled,
    cleared, or interrupted, instead of only linking what this pass downloaded.
    """
    if buffer_dir is None:
        return 0
    linked = 0
    for row in rows:
        if not _is_done(row["read_path"]):
            continue
        link = buffer_dir / prefetch_name(row["source_path"])
        if link.exists():
            continue
        os.link(row["read_path"], link)
        linked += 1
    return linked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--buffer-dir", default="",
                        help="hardlink buffer for archive_reader.set_prefetch_dir "
                             "(default <scratch>/../buffer)")
    parser.add_argument("--no-buffer", action="store_true",
                        help="skip the archive_reader-shaped hardlink buffer")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--failures", default="",
                        help="jsonl of unreadable rows; default <scratch>/../prefetch_failures.jsonl")
    parser.add_argument("--max-consecutive-failures", type=int, default=20)
    parser.add_argument("--no-verify-checksum", action="store_true",
                        help="skip the catalog sha256 check (it costs ~1s/GB)")
    args = parser.parse_args()

    rows = [row for row in _pool_rows(args.pool) if row.get("archive")]
    todo = [row for row in rows if not _is_done(row["read_path"])]
    if args.limit:
        todo = todo[:args.limit]
    # Physical order: one forward pass per shard instead of 19,699 random seeks.
    todo.sort(key=lambda row: (row["archive"]["root"], row["archive"]["shard"],
                               row["archive"]["offset"]))
    scratch = Path(todo[0]["read_path"]).parent if todo else Path(
        rows[0]["read_path"]).parent
    scratch.mkdir(parents=True, exist_ok=True)
    buffer_dir = None if args.no_buffer else Path(
        args.buffer_dir or scratch.parent / "buffer")
    if buffer_dir is not None:
        buffer_dir.mkdir(parents=True, exist_ok=True)
    failures_path = Path(args.failures or scratch.parent / "prefetch_failures.jsonl")

    print(f"pool={len(rows)} archived, already local={len(rows) - len(todo) if not args.limit else 'n/a'}, "
          f"todo={len(todo)}, workers={args.workers}, scratch={scratch}", flush=True)
    if not todo:
        print(json.dumps({"todo": 0, "linked": _link_buffer(rows, buffer_dir)}))
        return 0

    reader = ReadOnlyArchiveReader(verify_checksum=not args.no_verify_checksum)
    lock = threading.Lock()
    state = {"done": 0, "bytes": 0, "failed": 0, "streak": 0, "abort": False}
    started = time.perf_counter()
    failures: list[dict] = []

    def fetch(row: dict) -> None:
        if state["abort"]:
            return
        read_path = row["read_path"]
        try:
            payload = reader.read(row["source_path"])
            tmp = read_path + ".tmp"
            with open(tmp, "wb") as handle:
                handle.write(payload)
            os.replace(tmp, read_path)
            size = len(payload)
            failure = None
        except Exception as error:  # noqa: BLE001 - a bad member must not stop the pass
            size = 0
            failure = {"path_key": row["path_key"], "source_path": row["source_path"],
                       "error": f"{type(error).__name__}: {str(error)[:200]}"}
        with lock:
            if failure is None:
                state["done"] += 1
                state["bytes"] += size
                state["streak"] = 0
            else:
                state["failed"] += 1
                state["streak"] += 1
                failures.append(failure)
                if state["streak"] >= args.max_consecutive_failures:
                    # A soft mount reports an outage one read at a time; without
                    # this the tool would "finish" with 19,699 failures.
                    state["abort"] = True
                    print(f"[abort] {state['streak']} consecutive read failures — "
                          "check /mnt/nfs-ro before retrying", file=sys.stderr, flush=True)
            total = state["done"] + state["failed"]
            if total % args.progress_every == 0 or total == len(todo):
                elapsed = time.perf_counter() - started
                rate = state["done"] / max(elapsed, 1e-6)
                mbps = state["bytes"] / 1e6 / max(elapsed, 1e-6)
                eta = (len(todo) - total) / max(rate, 1e-6) / 60
                print(f"[{total}/{len(todo)}] ok={state['done']} fail={state['failed']} "
                      f"{rate:.1f} img/s {mbps:.1f} MB/s eta={eta:.1f}min", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(fetch, todo))
    reader.close()

    elapsed = time.perf_counter() - started
    if failures:
        with failures_path.open("a", encoding="utf-8") as handle:
            for item in failures:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps({
        "todo": len(todo), "ok": state["done"], "failed": state["failed"],
        "buffer_linked": _link_buffer(rows, buffer_dir),
        "aborted": state["abort"], "bytes": state["bytes"],
        "seconds": round(elapsed, 1),
        "img_per_s": round(state["done"] / max(elapsed, 1e-6), 2),
        "MB_per_s": round(state["bytes"] / 1e6 / max(elapsed, 1e-6), 1),
        "failures_file": os.fspath(failures_path) if failures else None,
    }, ensure_ascii=False))
    return 1 if state["abort"] or failures else 0


if __name__ == "__main__":
    sys.exit(main())
