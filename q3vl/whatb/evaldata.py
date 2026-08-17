"""Images and GT alpha at the frozen headline resolution -- ONE loader.

``SampleStore`` was written inside EPR-024's arm module; EPR-026 then declared a
"contract gap" and skipped its own headline rather than import another arm, and
EPR-028 / EPR-029 grew private ``_read_member`` copies.  The frozen block says
共同依赖只写一份, so the loader lives here and every arm imports it.

Frozen口径 (unchanged from the EPR-024 implementation, byte for byte):

* images come out of ``sft2seg-20260804`` by ``(shard, offset, length)`` and are
  ``area_resize``-d to short side 512 -- the one sanctioned resize operator;
* GT alpha comes from ``where_a-20260805/maskviews/<split>``'s ``.maskhi.png``
  (mode ``L``, already short side 512);
* a ``style`` sample has no mask member and ``alpha == 1`` everywhere -- the
  dataset's own convention, not an imputation.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .splits import DATASET_ROOT, IndexRow, ro_path

__all__ = ["SHORT_SIDE", "SampleStore", "read_member", "resize_short_side"]

#: headline resolution (frozen block)
SHORT_SIDE = 512


def read_member(shard: str | Path, offset: int, length: int) -> bytes:
    with ro_path(shard).open("rb") as fh:
        fh.seek(int(offset))
        blob = fh.read(int(length))
    if len(blob) != int(length):
        raise IOError(f"short read of {shard}:{offset}")
    return blob


def resize_short_side(img: Tensor, short_side: int) -> Tensor:
    """``area_resize`` to the given short side, aspect preserved (frozen block)."""
    from q3vl.where.upsample import area_resize            # the one sanctioned operator

    _, h, w = img.shape
    if min(h, w) == short_side:
        return img
    scale = short_side / float(min(h, w))
    size = (max(1, int(round(h * scale))), max(1, int(round(w * scale))))
    return area_resize(img.unsqueeze(0), size)[0]


class SampleStore:
    """Images and GT alpha for one split, read from the soft mount.

    Images come straight out of ``sft2seg-20260804`` by ``(shard, offset,
    length)``; GT alpha comes from ``where_a-20260805/maskviews/<split>`` whose
    ``indexes/shard-*.idx.jsonl`` gives the same three coordinates for the
    ``.maskhi.png`` member (mode ``L``, short side 512 already).  ``style``
    samples have no mask member and ``alpha == 1`` everywhere -- that is the
    dataset's own convention, not an imputation.
    """

    def __init__(self, split: str, *, root: str | Path = DATASET_ROOT,
                 mask_root: str | Path = "/mnt/nfs-ro/bc/data/datasets/where_a-20260805/maskviews",
                 short_side: int = SHORT_SIDE) -> None:
        self.split = split
        self.root = Path(root)
        self.mask_dir = Path(mask_root) / split
        self.short_side = int(short_side)
        self._mask_index: dict[str, tuple[Path, int, int]] | None = None

    # -- alpha --
    def _load_mask_index(self) -> dict[str, tuple[Path, int, int]]:
        if self._mask_index is not None:
            return self._mask_index
        idx: dict[str, tuple[Path, int, int]] = {}
        idx_dir = self.mask_dir / "indexes"
        if idx_dir.is_dir():
            for p in sorted(idx_dir.glob("shard-*.idx.jsonl")):
                with p.open(encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        if str(rec.get("suffix")) != ".maskhi.png":
                            continue
                        shard = self.mask_dir / "shards" / f"{rec['shard']}.tar"
                        idx[str(rec["sample_id"])] = (shard, int(rec["offset"]),
                                                      int(rec["length"]))
        self._mask_index = idx
        return idx

    def alpha(self, sample_id: str, task_type: str, *, device: Any = "cpu",
              dtype: torch.dtype = torch.float32) -> Tensor | float:
        if task_type == "style":
            return 1.0
        entry = self._load_mask_index().get(sample_id)
        if entry is None:
            raise KeyError(
                f"{sample_id}: no .maskhi.png in {self.mask_dir}; every local "
                "sample has one (manifest sample_count == the split's local count)")
        from PIL import Image

        blob = read_member(*entry)
        with Image.open(io.BytesIO(blob)) as im:
            if im.mode != "L":
                raise ValueError(f"{sample_id}: maskhi is mode {im.mode!r}, expected 'L'")
            arr = np.asarray(im, dtype=np.uint8)
        a = torch.from_numpy(arr.astype(np.float32) / 255.0).to(device=device, dtype=dtype)
        return a.unsqueeze(0)

    # -- image --
    def image(self, row: IndexRow, *, device: Any = "cpu",
              dtype: torch.dtype = torch.float32) -> Tensor:
        from PIL import Image

        member = row.raw["members"]["image"]
        blob = read_member(member["shard"], member["offset"], member["length"])
        with Image.open(io.BytesIO(blob)) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        img = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1)
        return resize_short_side(img.to(device=device, dtype=dtype), self.short_side)

    def load(self, row: IndexRow, *, device: Any = "cpu",
             dtype: torch.dtype = torch.float32) -> tuple[Tensor, Tensor | float]:
        img = self.image(row, device=device, dtype=dtype)
        a = self.alpha(row.sample_id, row.task_type, device=device, dtype=dtype)
        if isinstance(a, Tensor) and a.shape[-2:] != img.shape[-2:]:
            from q3vl.where.upsample import area_resize

            a = area_resize(a.unsqueeze(0), (img.shape[-2], img.shape[-1]))[0]
        return img, a
