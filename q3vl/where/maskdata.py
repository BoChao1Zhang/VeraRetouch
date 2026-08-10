"""Where-A GT masks: locate, load and project the per-candidate region mask.

Provenance, established on 2026-08-05 by reading the published data (not by
naming convention):

* ``sft2seg-20260804`` records carry ``image.origin`` -- ``root`` (the original
  build batch directory), ``shard``, ``offset``, ``length``, ``sha256`` -- and
  ``source_sample_id``, the build's own sample id;
* every local build batch (``prod-l{1..6}-local17k-*``) publishes exactly one
  ``<source_sample_id>.cgt.png`` member per sample, next to ``.in.*`` (I_in) and
  ``.jpg`` (I_tar).  It is mode-``L`` uint8, soft-edged, at the build's render
  resolution (short side 1024, the raw aspect ratio);
* that member is located through the batch's own
  ``indexes/catalog.sqlite3`` (``members(sample_id, suffix)`` index) or, as a
  fallback, its ``indexes/shard-*.idx.jsonl``.

Data discipline (CLAUDE.md): ``C_GT (.cgt.png)`` is the per-candidate region
mask (soft, single channel) -- it is *not* a colour ground truth.  That is
exactly the Where-A label.
"""

from __future__ import annotations

import io
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from PIL import Image

from q3vl.train.shards import MemberRef, ShardIndex, ShardStore, rewrite_read_path

from .config import (
    DEGENERATE_STD,
    EXCLUDE_WINNER_CONFIDENCE_LOW,
    LOCAL_BUILDS,
    MASK_SUFFIX,
    SPLIT_DIR,
)
from .fpre import grid_from_geometry

__all__ = ["MaskRef", "MaskResolver", "MaskViewStore", "mask_views", "eligibility",
           "iter_split_records", "split_index_path"]

_ASPECT_TOL = 0.02


def split_index_path(split: str) -> Path:
    return SPLIT_DIR / f"{split}.index.jsonl"


@dataclass
class MaskRef:
    sample_id: str
    source_sample_id: str
    build: str
    build_id: str
    batch: str
    root: str
    member: MemberRef

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "source_sample_id": self.source_sample_id,
            "build": self.build,
            "build_id": self.build_id,
            "batch": self.batch,
            "root": self.root,
            "member": self.member.to_dict(),
        }


class MaskLookupError(RuntimeError):
    pass


class MaskResolver:
    """Locate and read a per-candidate build member, caching one catalog per batch.

    ``suffix`` defaults to the ``.cgt.png`` mask this class was written for.  The
    other per-candidate members sit in the same shard behind the same catalog
    (``.in.*`` = ``I_in``, ``.jpg`` = ``I_tar``), so protocol 12.2's target image
    is the same lookup with a different suffix -- see
    :meth:`q3vl.what.data.WhatDataset.load_target_image`.
    """

    def __init__(self, verify: str = "checksum", suffix: str = MASK_SUFFIX):
        self.verify = verify
        self.suffix = suffix
        self._cats: dict[str, sqlite3.Connection | None] = {}
        self._jsonl: dict[str, dict[str, tuple[str, str, int, int, str]]] = {}
        self._store = ShardStore("/", verify=verify)
        self.n_catalog_hits = 0
        self.n_jsonl_hits = 0

    # -- locating ----------------------------------------------------------
    def _catalog(self, root: str) -> sqlite3.Connection | None:
        if root not in self._cats:
            # W-B1: `root` is `image.origin.root`, an absolute build path baked
            # into the records -- on the hard mount, where this `exists()` is
            # itself a blocking stat.  The tar reads below go through
            # ShardStore, which rewrites on its own.
            path = Path(rewrite_read_path(root)) / "indexes" / "catalog.sqlite3"
            self._cats[root] = (
                sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
                if path.exists() else None
            )
        return self._cats[root]

    def _jsonl_table(self, root: str) -> dict[str, tuple[str, str, int, int, str]]:
        table = self._jsonl.get(root)
        if table is None:
            table = {}
            idx = Path(rewrite_read_path(root)) / "indexes"      # W-B1, see _catalog
            for p in sorted(idx.glob("shard-*.idx.jsonl")):
                with p.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        if row.get("suffix") == self.suffix:
                            table[row["sample_id"]] = (
                                row["shard"], row["member"], int(row["offset_data"]),
                                int(row["length"]), row.get("sha256"),
                            )
            self._jsonl[root] = table
        return table

    def resolve(self, record: dict[str, Any]) -> MaskRef:
        origin = (record.get("image") or {}).get("origin") or {}
        root = origin.get("root")
        src = record.get("source_sample_id")
        if not root or not src:
            raise MaskLookupError(
                f"{record.get('sample_id')}: record has no image.origin.root / "
                f"source_sample_id, cannot reach the build's {self.suffix}"
            )
        row = None
        cat = self._catalog(root)
        if cat is not None:
            row = cat.execute(
                "select shard, member, offset_data, size, sha256 from members "
                "where sample_id=? and suffix=?",
                (src, self.suffix),
            ).fetchone()
            if row is not None:
                self.n_catalog_hits += 1
        if row is None:
            row = self._jsonl_table(root).get(src)
            if row is not None:
                self.n_jsonl_hits += 1
        if row is None:
            raise MaskLookupError(
                f"{record.get('sample_id')}: no {self.suffix} for {src} under {root}")
        shard, member, offset, size, sha = row
        return MaskRef(
            sample_id=record["sample_id"],
            source_sample_id=src,
            build=record.get("build", ""),
            build_id=record.get("build_id", ""),
            batch=record.get("batch", ""),
            root=root,
            member=MemberRef(
                shard=str(Path(root) / "shards" / f"{shard}.tar"),
                member=member, offset=int(offset), length=int(size),
                size=int(size), checksum=f"sha256:{sha}" if sha else None,
            ),
        )

    # -- reading -----------------------------------------------------------
    def read_bytes(self, ref: MaskRef) -> bytes:
        """The member's raw bytes, checksum-verified.  Suffix-agnostic."""
        return self._store.read(ref.member, verify=self.verify)

    def load(self, ref: MaskRef) -> np.ndarray:
        """``(h, w)`` float32 in [0, 1]."""
        data = self._store.read(ref.member, verify=self.verify)
        with Image.open(io.BytesIO(data)) as im:
            if im.mode != "L":
                raise MaskLookupError(
                    f"{ref.sample_id}: {self.suffix} is mode {im.mode!r}, expected 'L' "
                    "(single-channel soft mask)"
                )
            arr = np.asarray(im, dtype=np.uint8)
        return arr.astype(np.float32) / 255.0

    def close(self) -> None:
        for c in self._cats.values():
            if c is not None:
                c.close()
        self._cats.clear()
        self._store.close()


# --- projection to the two views used by Where-A ----------------------------

def mask_views(
    mask: np.ndarray, out_h: int, out_w: int, grid_h: int | None = None,
    grid_w: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(mask_hi (out_h,out_w), mask_low (grid_h,grid_w))``, both in [0,1].

    ``mask_hi`` is the GT mask on the spec-5 image grid (the resolution the
    model actually sees); ``mask_low`` is its area average on the ``F_pre``
    grid, which is where the oracle fit runs.
    """
    if grid_h is None or grid_w is None:
        grid_h, grid_w = grid_from_geometry(out_h, out_w)
    t = torch.from_numpy(np.ascontiguousarray(mask)).to(torch.float32)[None, None]
    hi = torch.nn.functional.interpolate(t, size=(out_h, out_w), mode="area") \
        if (t.shape[-2] >= out_h and t.shape[-1] >= out_w) else \
        torch.nn.functional.interpolate(t, size=(out_h, out_w), mode="bilinear", align_corners=False)
    low = torch.nn.functional.interpolate(hi, size=(grid_h, grid_w), mode="area")
    return hi[0, 0].clamp(0, 1), low[0, 0].clamp(0, 1)


class MaskViewStore:
    """Reader for the mask views published by ``scripts/extract_maskviews.py``.

    Without this, that job had no consumer at all and the online pipeline
    re-decoded every 1024-short-side ``.cgt.png`` on every pass
    (REVIEW-impl-WhereA N-3).  Four arms x 1 epoch = four decodes of the same
    PNG; with the store it is one decode, ever.

    Falls back cleanly: ``get`` returns ``None`` for a sample the shards do not
    carry, and :class:`~q3vl.where.pipeline.WhereADataSource` then resolves it
    from the build as before.
    """

    def __init__(self, root: str | Path, verify: str = "checksum"):
        self.root = Path(root)
        self.verify = verify
        self._store = ShardStore(self.root / "shards", verify=verify)
        self._members: dict[str, dict[str, MemberRef]] = {}
        self.n_hits = 0
        self.n_misses = 0
        for p in sorted((self.root / "indexes").glob("shard-*.idx.jsonl")):
            with p.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    ref = MemberRef(
                        shard=str(self.root / "shards" / f"{row['shard']}.tar"),
                        member=row["member"], offset=int(row["offset_data"]),
                        length=int(row["length"]), size=int(row["size"]),
                        checksum=f"sha256:{row['sha256']}",
                    )
                    self._members.setdefault(row["sample_id"], {})[row["suffix"]] = ref
        if not self._members:
            raise MaskLookupError(f"no maskview index rows under {self.root}")

    def __contains__(self, sample_id: str) -> bool:
        return sample_id in self._members

    def __len__(self) -> int:
        return len(self._members)

    def get(self, sample_id: str) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]] | None:
        """``(mask_hi (H,W), mask_low (gh,gw), meta)`` or ``None`` if absent."""
        members = self._members.get(sample_id)
        if not members or ".masklow.npy" not in members:
            self.n_misses += 1
            return None
        import io as _io

        low = np.load(_io.BytesIO(self._store.read(members[".masklow.npy"])),
                      allow_pickle=False).astype(np.float32)
        with Image.open(_io.BytesIO(self._store.read(members[".maskhi.png"]))) as im:
            hi = np.asarray(im, dtype=np.uint8).astype(np.float32) / 255.0
        meta = json.loads(self._store.read(members[".maskmeta.json"]).decode("utf-8"))
        self.n_hits += 1
        return (torch.from_numpy(hi), torch.from_numpy(low), meta)

    def close(self) -> None:
        self._store.close()


def eligibility(
    record: dict[str, Any],
    *,
    exclude_low: bool = EXCLUDE_WINNER_CONFIDENCE_LOW,
) -> tuple[bool, str]:
    """Protocol 4.4: Where-A uses only the GT masks of local ``l1-l6``."""
    build = record.get("build")
    if build not in LOCAL_BUILDS:
        return False, f"build_{build}_not_local"
    if record.get("render_mode") not in (None, "local"):
        return False, f"render_mode_{record.get('render_mode')}"
    if exclude_low and record.get("winner_confidence") == "low":
        return False, "winner_confidence_low"
    img = record.get("image") or {}
    if not (img.get("origin") or {}).get("root"):
        return False, "no_origin_locator"
    if not img.get("out_h") or not img.get("out_w"):
        return False, "no_geometry"
    return True, "ok"


def aspect_check(record: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    """The mask must carry the same aspect ratio as the EXIF-oriented image."""
    img = record["image"]
    oh, ow = img.get("oriented_h"), img.get("oriented_w")
    mh, mw = mask.shape
    a_img = ow / oh if oh else float("nan")
    a_mask = mw / mh
    rel = abs(a_mask - a_img) / a_img if a_img == a_img and a_img > 0 else float("inf")
    return {
        "aspect_image": a_img, "aspect_mask": a_mask, "rel_error": rel,
        "ok": rel <= _ASPECT_TOL, "mask_h": mh, "mask_w": mw,
        "exif_orientation": img.get("exif_orientation"),
    }


def mask_stats(mask_low: torch.Tensor) -> dict[str, Any]:
    v = mask_low.reshape(-1)
    return {
        "mean": float(v.mean()), "std": float(v.std(unbiased=False)),
        "min": float(v.min()), "max": float(v.max()),
        "frac_soft": float(((v > 0.04) & (v < 0.96)).float().mean()),
        "degenerate": bool(float(v.std(unbiased=False)) < DEGENERATE_STD),
    }


# --- iteration --------------------------------------------------------------

def iter_split_records(
    split: str,
    *,
    local_only: bool = True,
    limit: int | None = None,
    verify: str = "checksum",
    store: ShardStore | None = None,
) -> Iterator[tuple[Any, dict[str, Any]]]:
    """Yield ``(SampleRef, record)`` from a frozen split index."""
    index = ShardIndex.load(split_index_path(split))
    st = store or ShardStore("/", verify=verify)
    n = 0
    for ref in index.samples:
        if local_only and ref.meta.get("build") not in LOCAL_BUILDS:
            continue
        rec = json.loads(st.read(ref.members["record"]).decode("utf-8"))
        yield ref, rec
        n += 1
        if limit is not None and n >= limit:
            break
