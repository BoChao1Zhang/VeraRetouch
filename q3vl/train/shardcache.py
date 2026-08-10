"""Populate the local read cache that :mod:`q3vl.train.shards` resolves against.

Why (PERF-1, 2026-08-10).  Where-B walks a shuffled 1-epoch permutation, so
every member read is a cold random read against NFS.  Measured on this box:

===================================  ==============  ==========
read                                 p50 latency     throughput
===================================  ==============  ==========
nfs-ro, random 152 KiB, 1 thread     17.0 ms         53 reads/s
local ext4, page cache dropped        6.5 ms         145 reads/s
local ext4, page cache warm            0.02 ms       45k reads/s
nfs-ro, sequential copy               --             106 MB/s
===================================  ==============  ==========

Assembling one micro-batch of 8 cost 627 ms, of which 488 ms (78%) was NFS
round-trip latency and under 17 ms was actually moving bytes.  The corpus a
Where-B arm consumes is 27.5 GiB; copying it once, sequentially, costs about
four and a half minutes and is amortised over an eight-hour arm.

What this module guarantees:

* **content, not paths** -- each shard is verified against the ``tar_sha256``
  the producing job published in its own ``manifest.json``; a shard whose
  manifest has no digest is copied but recorded as ``digest=None`` so the
  cache manifest never claims a check it did not do;
* **atomic entries** -- copy to ``<name>.partial``, fsync, ``os.replace``; a
  killed warm job leaves no half-file that a reader could mistake for a shard;
* **a capacity bound** -- ``--max-bytes`` plus a free-space floor, checked
  before each copy, so the cache can never fill the disk that also holds the
  run directory and the checkpoints;
* **rebuildability** -- everything here is a byte-for-byte copy of a published,
  immutable dataset, i.e. scratch in the sense of METACANVAS 2.3.  Deleting the
  cache root costs one warm job and nothing else.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .shards import (
    CACHE_MANIFEST_NAME,
    DEFAULT_SHARD_CACHE,
    SHARD_CACHE_ENV,
    cache_key_for,
    reset_shard_cache,
    rewrite_read_path,
    shard_cache_facts,
)

__all__ = ["CacheEntry", "plan_dataset", "plan_paths", "warm", "verify",
           "preload", "cache_root_from_env", "FREE_SPACE_FLOOR_BYTES"]

#: never let the cache take the disk below this (checkpoints + run dirs live here)
FREE_SPACE_FLOOR_BYTES = 100 * 1024 ** 3
#: default ceiling on the cache itself
DEFAULT_MAX_BYTES = 400 * 1024 ** 3
_COPY_CHUNK = 8 * 1024 * 1024


def _log(*args) -> None:
    """Unbuffered progress.  A warm job is launched with nohup under D-20, and
    ``tail`` on a log that only flushes at exit cannot tell "running" from
    "wedged" -- which is exactly the confusion D-20 exists to prevent."""
    print(*args, flush=True)


def cache_root_from_env() -> Path:
    raw = os.environ.get(SHARD_CACHE_ENV)
    return Path(raw.strip()) if raw and raw.strip() else DEFAULT_SHARD_CACHE


@dataclass(frozen=True)
class CacheEntry:
    """One shard tar to mirror locally."""

    source: Path                 # absolute path under an export prefix
    key: str                     # cache-relative key
    bytes: int                   # size the producer published
    sha256: str | None           # producer's digest, when it published one

    def local(self, root: Path) -> Path:
        return root / self.key


def _iter_manifest_shards(root: Path) -> Iterable[dict[str, Any]]:
    manifest = Path(rewrite_read_path(root)) / "manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("status") not in (None, "complete"):
        raise RuntimeError(
            f"{manifest}: status={data.get('status')!r}; refusing to cache a "
            "dataset that was not published atomically"
        )
    return data.get("shards") or []


def plan_dataset(root: str | os.PathLike) -> list[CacheEntry]:
    """Every ``shards/*.tar`` of one published indexed-tar dataset.

    The index JSONL and the sqlite catalog are deliberately **not** cached: they
    are read once at startup (12.5 s for all five of Where-B's sources) and
    caching them would put a second copy of the authority next to the data.
    """
    root = Path(root)
    out: list[CacheEntry] = []
    for shard in _iter_manifest_shards(root):
        rel = shard.get("tar") or f"shards/{shard['shard_id']}.tar"
        src = root / rel
        key = cache_key_for(src)
        if key is None:
            raise ValueError(
                f"{src} is not under a cacheable export prefix; only paths on the "
                "NFS export can be cached (the key must identify the source)"
            )
        out.append(CacheEntry(source=src, key=key,
                              bytes=int(shard["tar_bytes"]),
                              sha256=shard.get("tar_sha256")))
    return out


def plan_paths(paths: Sequence[str | os.PathLike]) -> list[CacheEntry]:
    """Ad-hoc entries for shard tars that are not part of a published dataset."""
    out = []
    for p in paths:
        src = Path(p)
        key = cache_key_for(src)
        if key is None:
            raise ValueError(f"{src} is not under a cacheable export prefix")
        out.append(CacheEntry(source=src, key=key,
                              bytes=os.stat(rewrite_read_path(src)).st_size,
                              sha256=None))
    return out


def _load_manifest(root: Path) -> dict[str, Any]:
    try:
        return json.loads((root / CACHE_MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"schema_version": "q3vl.shard_cache/1", "entries": {}}


def _write_manifest(root: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    tmp = root / (CACHE_MANIFEST_NAME + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, root / CACHE_MANIFEST_NAME)


def _copy_verified(src: Path, dst: Path) -> tuple[int, str]:
    """Stream ``src`` -> ``dst.partial`` -> ``dst``, returning (bytes, sha256)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    partial = dst.with_name(dst.name + ".partial")
    digest = hashlib.sha256()
    n = 0
    with open(rewrite_read_path(src), "rb") as fin, open(partial, "wb") as fout:
        while True:
            chunk = fin.read(_COPY_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
            fout.write(chunk)
            n += len(chunk)
        fout.flush()
        os.fsync(fout.fileno())
    os.replace(partial, dst)
    return n, digest.hexdigest()


def warm(
    entries: Sequence[CacheEntry],
    root: Path | None = None,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    free_floor: int = FREE_SPACE_FLOOR_BYTES,
    force: bool = False,
    log=_log,
) -> dict[str, Any]:
    """Copy anything missing, verify it, and publish the cache manifest.

    An entry already present at the right size (and, when the producer published
    one, the right digest) is skipped, so re-running the job after an interrupted
    copy is cheap and safe.
    """
    root = Path(root or cache_root_from_env())
    root.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(root)
    have: dict[str, Any] = manifest.setdefault("entries", {})

    planned = sum(e.bytes for e in entries)
    log(f"[cache] root={root}  {len(entries)} shards, {planned / 2**30:.2f} GiB planned")
    if planned > max_bytes:
        raise RuntimeError(
            f"plan is {planned / 2**30:.1f} GiB but --max-bytes is "
            f"{max_bytes / 2**30:.1f} GiB; raise the cap deliberately or cache "
            "fewer splits"
        )

    stats = {"copied": 0, "skipped": 0, "copied_bytes": 0, "seconds": 0.0,
             "digest_checked": 0, "digest_absent": 0}
    t0 = time.time()
    for i, e in enumerate(entries, 1):
        dst = e.local(root)
        rec = have.get(e.key)
        if not force and rec and rec.get("bytes") == e.bytes:
            try:
                if dst.stat().st_size == e.bytes:
                    stats["skipped"] += 1
                    continue
            except OSError:
                pass
        free = shutil.disk_usage(root).free
        if free - e.bytes < free_floor:
            raise RuntimeError(
                f"{dst}: {free / 2**30:.1f} GiB free, copying {e.bytes / 2**30:.1f} "
                f"GiB would break the {free_floor / 2**30:.0f} GiB floor. The cache "
                "must never crowd out the run directory."
            )
        t1 = time.time()
        n, got = _copy_verified(e.source, dst)
        dt = max(time.time() - t1, 1e-9)
        if n != e.bytes:
            dst.unlink(missing_ok=True)
            raise RuntimeError(f"{e.source}: copied {n} bytes, manifest says {e.bytes}")
        if e.sha256:
            if got != e.sha256:
                dst.unlink(missing_ok=True)
                raise RuntimeError(
                    f"{e.source}: sha256 {got} != published {e.sha256}; the copy is "
                    "not the shard the producer published, refusing to cache it"
                )
            stats["digest_checked"] += 1
        else:
            stats["digest_absent"] += 1
        have[e.key] = {"bytes": n, "sha256": got, "source": str(e.source),
                       "published_sha256": e.sha256}
        _write_manifest(root, manifest)
        stats["copied"] += 1
        stats["copied_bytes"] += n
        log(f"[cache] {i}/{len(entries)} {e.key}  {n / 2**20:.0f} MiB "
            f"in {dt:.1f}s ({n / dt / 1e6:.0f} MB/s)"
            + ("  sha256 ok" if e.sha256 else "  sha256 recorded (producer published none)"))
    _write_manifest(root, manifest)
    stats["seconds"] = round(time.time() - t0, 1)
    reset_shard_cache()
    stats["facts"] = shard_cache_facts()
    log(f"[cache] done: copied {stats['copied']} "
        f"({stats['copied_bytes'] / 2**30:.2f} GiB), skipped {stats['skipped']}, "
        f"{stats['seconds']}s")
    return stats


def preload(root: Path | None = None, *, log=_log) -> dict[str, Any]:
    """Read every cached shard once so its pages are resident before an arm starts.

    Copying already leaves the pages in the page cache, so a fresh warm job needs
    no preload -- but a *skipped* warm job (the common case: the cache is already
    complete) touches nothing, and by then the pages may be long gone.  The gap
    is worth 100 ms per micro-batch: cached shards read at 0.02 ms per member
    from RAM and 6.5 ms from this ext4 volume.  27.5 GiB against 113 GiB of free
    memory fits with room to spare, and the kernel evicts it first under
    pressure, so this can only cost time, never correctness.
    """
    root = Path(root or cache_root_from_env())
    manifest = _load_manifest(root)
    t0 = time.time()
    total = 0
    for key in manifest.get("entries", {}):
        p = root / key
        try:
            with p.open("rb") as fh:
                while True:
                    chunk = fh.read(_COPY_CHUNK)
                    if not chunk:
                        break
                    total += len(chunk)
        except OSError as exc:
            log(f"[cache] preload skipped {key}: {type(exc).__name__}: {exc}")
    dt = max(time.time() - t0, 1e-9)
    out = {"root": str(root), "bytes": total, "seconds": round(dt, 1),
           "mb_per_s": round(total / dt / 1e6, 1)}
    log(f"[cache] preload {total / 2**30:.2f} GiB in {dt:.1f}s "
        f"({total / dt / 1e6:.0f} MB/s)")
    return out


def verify(root: Path | None = None, *, deep: bool = False, log=_log) -> dict[str, Any]:
    """Check the cache against its own manifest.  ``deep`` re-hashes every shard."""
    root = Path(root or cache_root_from_env())
    manifest = _load_manifest(root)
    bad: list[dict[str, Any]] = []
    ok = 0
    for key, rec in manifest.get("entries", {}).items():
        p = root / key
        try:
            size = p.stat().st_size
        except OSError as exc:
            bad.append({"key": key, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if size != rec["bytes"]:
            bad.append({"key": key, "error": f"size {size} != {rec['bytes']}"})
            continue
        if deep:
            h = hashlib.sha256()
            with p.open("rb") as fh:
                for chunk in iter(lambda: fh.read(_COPY_CHUNK), b""):
                    h.update(chunk)
            if h.hexdigest() != rec["sha256"]:
                bad.append({"key": key, "error": "sha256 mismatch"})
                continue
        ok += 1
    out = {"root": str(root), "n_ok": ok, "n_bad": len(bad), "bad": bad, "deep": deep}
    log(json.dumps(out, indent=2))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", action="append", default=[],
                    help="published indexed-tar root (repeatable)")
    ap.add_argument("--path", action="append", default=[],
                    help="individual shard tar (repeatable)")
    ap.add_argument("--root", default=None, help=f"cache root (default ${SHARD_CACHE_ENV})")
    ap.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    ap.add_argument("--free-floor", type=int, default=FREE_SPACE_FLOOR_BYTES)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--preload", action="store_true",
                    help="read the cache once so its pages are resident")
    ap.add_argument("--deep", action="store_true", help="with --verify: re-hash")
    args = ap.parse_args(argv)

    if args.verify:
        out = verify(Path(args.root) if args.root else None, deep=args.deep)
        return 0 if out["n_bad"] == 0 else 1

    if args.preload:
        preload(Path(args.root) if args.root else None)
        return 0

    entries: list[CacheEntry] = []
    for d in args.dataset:
        entries += plan_dataset(d)
    entries += plan_paths(args.path)
    if not entries:
        ap.error("nothing to do: pass --dataset and/or --path")
    if args.plan_only:
        print(json.dumps([{"key": e.key, "bytes": e.bytes, "sha256": e.sha256}
                          for e in entries], indent=2))
        print(f"total {sum(e.bytes for e in entries) / 2**30:.2f} GiB")
        return 0
    warm(entries, Path(args.root) if args.root else None,
         max_bytes=args.max_bytes, free_floor=args.free_floor, force=args.force)
    return 0


if __name__ == "__main__":                                     # pragma: no cover
    raise SystemExit(main())
