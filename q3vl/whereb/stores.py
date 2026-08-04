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

import io
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import torch
from PIL import Image

from q3vl.data.shardio import read_member
from q3vl.where.basis import Latent

__all__ = ["PublishedStore", "OracleStore", "MaskViewStore", "GenContextStore"]


class PublishedStore:
    """Random access to a published dataset by ``(sample_id, suffix)``."""

    def __init__(self, root: str | Path, *, verify: bool = True):
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
        return read_member(
            self.root, r["shard"], int(r["offset_data"]), int(r["length"]),
            r.get("sha256") if self.verify else None,
        )

    def read_json(self, sample_id: str, suffix: str) -> dict[str, Any]:
        return json.loads(self.read(sample_id, suffix).decode("utf-8"))

    def facts(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "status": self.manifest.get("status"),
            "member_count": self.manifest.get("member_count"),
            "sample_count": self.manifest.get("sample_count"),
            "shard_count": self.manifest.get("shard_count"),
            "index_rows": len(self.rows),
        }


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
    """This stage's cached generated ``<where>`` spans (see :mod:`gencontext`)."""

    SUFFIX = ".genwhere.json"

    def record(self, sample_id: str) -> dict[str, Any]:
        return self.read_json(sample_id, self.SUFFIX)

    def iter_records(self) -> Iterator[dict[str, Any]]:
        for sid, suf in sorted(self.rows):
            if suf == self.SUFFIX:
                yield self.read_json(sid, suf)

    def summary(self) -> dict[str, Any]:
        n = fail = trunc = 0
        reasons: dict[str, int] = {}
        for r in self.iter_records():
            n += 1
            fail += int(r.get("format_failure", False))
            trunc += int(r.get("truncated", False))
            k = str(r.get("stop_reason"))
            reasons[k] = reasons.get(k, 0) + 1
        return {
            "n": n,
            "format_failure_rate": fail / n if n else None,
            "truncation_rate": trunc / n if n else None,
            "stop_reasons": dict(sorted(reasons.items())),
            **self.facts(),
        }
