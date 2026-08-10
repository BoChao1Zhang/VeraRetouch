"""Reads of the published artefacts this analysis needs, and nothing else.

Two sources:

* Where-A's ``maskviews/<split>`` publication -- the GT mask, in both the
  ``F_pre``-grid (``.masklow.npy``) and spec-5 (``.maskhi.png``) views;
* the split index of ``sft2seg`` -- the input image ``I_in`` and the record, for
  the long-tail panels.

**Why this module exists instead of ``q3vl.whereb.stores`` / ``q3vl.train.shards``**
(decision recorded in NOTES.md): those two modules were mid-rewrite in the
working tree while WEVAL-1 was implemented (PERF-1, fd pooling + local shard
cache), and their read path raised ``TypeError: 'function' object is not
subscriptable`` on every member.  An analysis tool that has to run against a live
training campaign cannot take a hard dependency on a file another task is
actively editing.  The byte-level read is delegated to
``q3vl.data.shardio.read_member`` -- the same function ``PublishedStore`` used
before the rewrite, checksum verification included -- and index *parsing* still
goes through ``q3vl.train.shards.ShardIndex`` (pure JSON, no IO).  Nothing here
re-implements a format.

Reads are pointed at ``/mnt/nfs-ro`` (soft, v3: EIO after ~15 s instead of a
D-state hang) via ``rewrite_read_path``; nothing here ever writes to NFS.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any, Iterator

import numpy as np

__all__ = ["PublishedIndex", "MaskViews", "SplitImages", "read_bytes"]


def read_bytes(root: Path, shard: str, offset: int, length: int,
               sha256: str | None = None) -> bytes:
    """One member of a published dataset (``root/shards/<shard>.tar``)."""
    from q3vl.data.shardio import read_member

    return read_member(Path(root), shard, int(offset), int(length), sha256)


class PublishedIndex:
    """``(sample_id, suffix) -> index row`` for a published indexed-tar dataset."""

    def __init__(self, root: str | Path, *, verify: bool = True):
        self.root = Path(root)
        manifest = self.root / "manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(
                f"{self.root} is not a published indexed-tar dataset (no "
                "manifest.json); publication is atomic, so a missing manifest "
                "means the producing job has not finished"
            )
        self.manifest = json.loads(manifest.read_text())
        if self.manifest.get("status") != "complete":
            raise RuntimeError(
                f"{self.root}: manifest status={self.manifest.get('status')!r}"
            )
        self.verify = verify
        self.rows: dict[tuple[str, str], dict[str, Any]] = {}
        for p in sorted((self.root / "indexes").glob("shard-*.idx.jsonl")):
            with p.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        r = json.loads(line)
                        self.rows[(r["sample_id"], r["suffix"])] = r

    def has(self, sample_id: str, suffix: str) -> bool:
        return (sample_id, suffix) in self.rows

    def ids(self, suffix: str) -> list[str]:
        return sorted(sid for sid, suf in self.rows if suf == suffix)

    def read(self, sample_id: str, suffix: str) -> bytes:
        try:
            r = self.rows[(sample_id, suffix)]
        except KeyError:
            raise KeyError(f"{sample_id}{suffix} not in {self.root}") from None
        return read_bytes(self.root, r["shard"], r["offset_data"], r["length"],
                          r.get("sha256") if self.verify else None)

    def facts(self) -> dict[str, Any]:
        return {"root": str(self.root), "status": self.manifest.get("status"),
                "sample_count": self.manifest.get("sample_count"),
                "index_rows": len(self.rows)}


class MaskViews(PublishedIndex):
    """Where-A's published GT mask views."""

    LOW = ".masklow.npy"
    HI = ".maskhi.png"
    META = ".maskmeta.json"

    def mask_low(self, sample_id: str) -> np.ndarray:
        a = np.load(io.BytesIO(self.read(sample_id, self.LOW)), allow_pickle=False)
        return a.astype(np.float32)

    def mask_hi(self, sample_id: str) -> np.ndarray:
        from PIL import Image

        with Image.open(io.BytesIO(self.read(sample_id, self.HI))) as im:
            return np.asarray(im.convert("L"), dtype=np.float32) / 255.0

    def meta(self, sample_id: str) -> dict[str, Any]:
        return json.loads(self.read(sample_id, self.META).decode("utf-8"))

    def iter_masks(self, sample_ids: list[str]) -> Iterator[tuple[str, np.ndarray]]:
        for sid in sample_ids:
            if self.has(sid, self.HI):
                yield sid, self.mask_hi(sid)


class SplitImages:
    """``sample_id -> (I_in bytes, record dict)`` from a split's shard index."""

    def __init__(self, index_path: str | Path, *, verify: bool = True):
        from q3vl.train.shards import ShardIndex

        self.index = ShardIndex.load(Path(index_path))
        self.verify = verify
        self.by_id = {s.sample_id: s for s in self.index.samples}

    def _read(self, ref) -> bytes:
        from q3vl.train.shards import rewrite_read_path

        path = Path(rewrite_read_path(ref.shard))
        fd = os.open(path, os.O_RDONLY)
        try:
            data = os.pread(fd, ref.length, ref.offset)
        finally:
            os.close(fd)
        if len(data) != ref.length:
            raise RuntimeError(f"short read of {ref.shard}:{ref.member}")
        if self.verify and ref.checksum:
            import hashlib

            got = hashlib.sha256(data).hexdigest()
            if got != ref.checksum:
                raise RuntimeError(f"checksum mismatch for {ref.shard}:{ref.member}")
        return data

    def image_bytes(self, sample_id: str) -> bytes:
        return self._read(self.by_id[sample_id].members["image"])

    def record(self, sample_id: str) -> dict[str, Any]:
        return json.loads(self._read(self.by_id[sample_id].members["record"]).decode("utf-8"))

    def image_array(self, sample_id: str) -> np.ndarray:
        """``(H, W, 3)`` float in [0, 1], sized exactly as the eval saw it.

        ``prepare_image`` is the campaign's single sizing path (spec-5); it is
        reused rather than re-derived so a panel's pixel grid is the same one the
        ``.maskhi`` view and the F_pre grid were built on.
        """
        from q3vl.train.imageproc import prepare_image

        image, _geom = prepare_image(self.image_bytes(sample_id))
        return np.asarray(image, dtype=np.uint8).astype(np.float32) / 255.0
