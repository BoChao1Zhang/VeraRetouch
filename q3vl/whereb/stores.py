"""Readers for the published indexed-tar derivatives Where-B consumes.

Three producers, one format (protocol 2.3):

* ``q3vl.where.packing.pack_oracle``     -> ``<sample_id>.oracle.json``   (Where-A)
* ``q3vl.where.packing.pack_maskviews``  -> ``<sample_id>.masklow.npy`` / ``.maskhi.png``
* ``q3vl.whereb.gencontext``             -> ``<sample_id>.genwhere.json`` (this stage)

The oracle schema is not re-declared here; it is read back through
:meth:`q3vl.where.basis.Latent.from_dict`, so a change on the Where-A side
surfaces as a loud failure in :class:`OracleStore` rather than as a silently
mis-parsed ``w*``.  ``tests/test_stores.py`` builds a real published shard set
with the Where-A packer and reads it back through these classes, which is the
interop test the task card asks for.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import torch
from PIL import Image

from dataset_build.tools.indexed_tar import IndexedTarError
from q3vl.train.shards import BoundedBytesCache, SliceReader
from q3vl.where.basis import Latent

__all__ = ["PublishedStore", "OracleStore", "MaskViewStore", "GenContextStore"]

#: default in-process byte budget for member payloads.  Zero disables it; the
#: point is only to let a prefetch thread pay for a read the training thread is
#: about to make (PERF-1), so it needs to hold a few hundred samples, not a
#: split.  Raw ``bytes`` are cached and every accessor re-parses them, so no
#: caller can hand another caller a mutable object.
DEFAULT_BLOB_CACHE_BYTES = 64 * 1024 ** 2


class PublishedStore:
    """Random access to a published dataset by ``(sample_id, suffix)``.

    PERF-1: reads go through a :class:`~q3vl.train.shards.SliceReader`, which
    keeps one fd per shard per thread and resolves the local shard cache.  The
    previous ``os.open``/``os.close`` per member made OPEN+CLOSE 72% of the NFS
    client's RPC mix for two arms; the bytes read are identical either way.
    """

    def __init__(self, root: str | Path, *, verify: bool = True,
                 cache_bytes: int = 0):
        self.root = Path(root)
        if not (self.root / "manifest.json").exists():
            raise FileNotFoundError(
                f"{self.root} is not a published indexed-tar dataset "
                "(no manifest.json). Publication is atomic, so a missing "
                "manifest means the producing job has not finished."
            )
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("status") != "complete":
            raise RuntimeError(
                f"{self.root}: manifest status={self.manifest.get('status')!r}, "
                "refusing to read a dataset that was not published atomically"
            )
        self.verify = verify
        self.reader = SliceReader()
        self.blobs = BoundedBytesCache(cache_bytes) if cache_bytes > 0 else None
        self.rows: dict[tuple[str, str], dict[str, Any]] = {}
        for p in sorted((self.root / "indexes").glob("shard-*.idx.jsonl")):
            with p.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        r = json.loads(line)
                        self.rows[(r["sample_id"], r["suffix"])] = r

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def sample_ids(self) -> set[str]:
        return {sid for sid, _ in self.rows}

    def has(self, sample_id: str, suffix: str) -> bool:
        return (sample_id, suffix) in self.rows

    def read(self, sample_id: str, suffix: str) -> bytes:
        try:
            r = self.rows[(sample_id, suffix)]
        except KeyError:
            raise KeyError(f"{sample_id}{suffix} not in {self.root}") from None
        key = (sample_id, suffix)
        if self.blobs is not None:
            hit = self.blobs.get(key)
            if hit is not None:
                return hit
        shard, offset, length = r["shard"], int(r["offset_data"]), int(r["length"])
        data = self.reader.read(self.root / "shards" / f"{shard}.tar", offset, length)
        if len(data) != length:
            raise IndexedTarError(
                f"short read of {shard}:{offset} ({len(data)} != {length})")
        if self.verify and r.get("sha256") is not None:
            got = hashlib.sha256(data).hexdigest()
            if got != r["sha256"]:
                raise IndexedTarError(f"checksum mismatch for {shard}:{offset}")
        if self.blobs is not None:
            self.blobs.put(key, data)
        return data

    def read_json(self, sample_id: str, suffix: str) -> dict[str, Any]:
        return json.loads(self.read(sample_id, suffix).decode("utf-8"))

    def prime(self, sample_id: str, suffix: str) -> bool:
        """Pay for one member read now (prefetch thread) so a later read is free.

        Returns False for a member this store does not carry -- a global sample
        has no oracle latent and asking for one is not an error here; the caller
        that actually needs it still raises.
        """
        if not self.has(sample_id, suffix):
            return False
        self.read(sample_id, suffix)
        return True

    def close(self) -> None:
        self.reader.close()

    def facts(self) -> dict[str, Any]:
        out = {
            "root": str(self.root),
            "status": self.manifest.get("status"),
            "member_count": self.manifest.get("member_count"),
            "sample_count": self.manifest.get("sample_count"),
            "shard_count": self.manifest.get("shard_count"),
            "index_rows": len(self.rows),
        }
        if self.blobs is not None:
            out["blob_cache"] = self.blobs.facts()
        return out


class OracleStore(PublishedStore):
    """Where-A per-image oracle latents (``w*, rho*``) for both readouts."""

    SUFFIX = ".oracle.json"

    def payload(self, sample_id: str) -> dict[str, Any]:
        return self.read_json(sample_id, self.SUFFIX)

    def latent(
        self, sample_id: str, readout: str, *, require_ok: bool = True,
        dtype=torch.float32,
    ) -> Latent | None:
        """``None`` when the sample has no usable fit -- never a zero vector.

        Protocol 10.2 is explicit that a rejected fit goes into the rejection
        report instead of being replaced by zeros; Where-B honours that by
        masking the oracle auxiliaries off for those samples (``L_mask`` still
        applies) rather than training against a fabricated target.
        """
        fits = self.payload(sample_id).get("fits") or {}
        fit = fits.get(readout)
        if fit is None:
            return None
        if require_ok and fit.get("status") != "ok":
            return None
        lat = fit.get("latent")
        if lat is None:
            return None
        return Latent.from_dict(lat, dtype=dtype)

    def status(self, sample_id: str, readout: str) -> str | None:
        fits = self.payload(sample_id).get("fits") or {}
        fit = fits.get(readout)
        return None if fit is None else fit.get("status")

    def coverage(self, sample_ids: Iterable[str], readout: str) -> dict[str, Any]:
        ids = list(sample_ids)
        present = [s for s in ids if self.has(s, self.SUFFIX)]
        ok = [s for s in present if self.status(s, readout) == "ok"]
        return {
            "n_requested": len(ids), "n_present": len(present), "n_ok": len(ok),
            "present_rate": len(present) / len(ids) if ids else None,
            "ok_rate": len(ok) / len(ids) if ids else None,
        }


class MaskViewStore(PublishedStore):
    """Where-A GT mask views: the ``F_pre``-grid and spec-5-grid projections."""

    LOW = ".masklow.npy"
    HI = ".maskhi.png"
    META = ".maskmeta.json"

    def mask_low(self, sample_id: str) -> torch.Tensor:
        a = np.load(io.BytesIO(self.read(sample_id, self.LOW)), allow_pickle=False)
        return torch.from_numpy(a.astype(np.float32))

    def mask_hi(self, sample_id: str) -> torch.Tensor:
        with Image.open(io.BytesIO(self.read(sample_id, self.HI))) as im:
            a = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
        return torch.from_numpy(a)

    def meta(self, sample_id: str) -> dict[str, Any]:
        return self.read_json(sample_id, self.META)


class GenContextStore(PublishedStore):
    """This stage's cached generated reasoning spans (see :mod:`gencontext`).

    Schema v2 (amendment A-4) carries the ``<color>`` segment next to the
    ``<where>`` one.  Every v1 field kept its name *and* its meaning, so
    :meth:`where_ids` -- the only thing Where-B reads -- behaves identically on
    v1 and v2 records.
    """

    SUFFIX = ".genwhere.json"

    def record(self, sample_id: str) -> dict[str, Any]:
        return self.read_json(sample_id, self.SUFFIX)

    def iter_records(self) -> Iterator[dict[str, Any]]:
        for sid, suf in sorted(self.rows):
            if suf == self.SUFFIX:
                yield self.read_json(sid, suf)

    # -- per-segment accessors ---------------------------------------------
    def where_ids(self, sample_id: str) -> list[int]:
        """The ``<where>`` span.  Present in v1 and v2 alike."""
        return list(self.record(sample_id)["where_ids"])

    def color_ids(self, sample_id: str) -> list[int]:
        """The ``<color>`` span.  Raises on a v1 record rather than guessing."""
        rec = self.record(sample_id)
        if "color_ids" not in rec:
            raise KeyError(
                f"{sample_id}: record is schema {rec.get('schema_version')!r}, which "
                "predates the <color> segment (amendment A-4). Re-run "
                "scripts/make_generated_context.py for this split."
            )
        return list(rec["color_ids"])

    def has_color(self) -> bool:
        for r in self.iter_records():
            return "color_ids" in r
        return False

    def summary(self) -> dict[str, Any]:
        from .gencontext import summarise_records

        records = list(self.iter_records())
        return {**summarise_records(records), **self.facts()}
