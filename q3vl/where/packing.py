"""Protocol 2.3 -- publish Where-A derivatives as indexed tar shards.

    "Newly produced oracle latents, basis metadata, continuous LUT codes or
     visualisation indexes are dataset derivatives too and must obey the
     cold/hot tiering and the indexed-tar-shard contract."

The format itself (uncompressed USTAR, 1--4 GiB shards, one index row per member
with shard/member/offset/length/size/checksum, schema version, per-shard digest,
sample count, staged-then-atomically-renamed publication, verified SQLite
catalog) is provided by :func:`q3vl.data.shardio.build_from_memory`; this module
only decides *what* goes in and keeps a sample's members adjacent so a
sequential reader sees them together.

Per-sample members
------------------
``<sample_id>.masklow.npy``   float16 ``(grid_h, grid_w)`` GT mask on the F_pre grid
``<sample_id>.maskhi.png``    uint8 ``(out_h, out_w)`` GT mask on the spec-5 grid
``<sample_id>.maskmeta.json`` geometry, provenance locator, mask statistics
``<sample_id>.oracle.json``   both readouts' ``(w*, rho*)``, fit report, metrics

The projector ``B`` itself is one artifact, not a per-sample derivative, so it is
written as plain files with a digest (see :func:`write_basis`).
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

from q3vl.data.shardio import build_from_memory, read_member

from .config import (
    MASKVIEW_SHARD_BYTES,
    ORACLE_SHARD_BYTES,
    SCHEMA_BASIS,
    SCHEMA_MASKVIEW,
    SCHEMA_ORACLE,
)

__all__ = ["maskview_payloads", "oracle_payloads", "pack_maskviews", "pack_oracle",
           "write_basis", "verify_published", "load_masklow", "json_bytes"]

PRODUCER = "q3vl.where.packing/1"


def json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"


def _npy_bytes(a: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, a, allow_pickle=False)
    return buf.getvalue()


def load_masklow(data: bytes) -> np.ndarray:
    return np.load(io.BytesIO(data), allow_pickle=False)


def _png_bytes(a: np.ndarray) -> bytes:
    if a.dtype != np.uint8:
        a = np.clip(np.rint(a * 255.0), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a, mode="L").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def maskview_payloads(
    sample_id: str,
    mask_low: torch.Tensor | np.ndarray,
    mask_hi: torch.Tensor | np.ndarray,
    meta: dict[str, Any],
) -> Iterator[tuple[str, bytes]]:
    low = mask_low.detach().cpu().numpy() if isinstance(mask_low, torch.Tensor) else mask_low
    hi = mask_hi.detach().cpu().numpy() if isinstance(mask_hi, torch.Tensor) else mask_hi
    payload = dict(meta)
    payload["schema_version"] = SCHEMA_MASKVIEW
    payload["sample_id"] = sample_id
    payload["mask_low_shape"] = list(low.shape)
    payload["mask_hi_shape"] = list(hi.shape)
    yield f"{sample_id}.masklow.npy", _npy_bytes(low.astype(np.float16))
    yield f"{sample_id}.maskhi.png", _png_bytes(np.asarray(hi, dtype=np.float32))
    yield f"{sample_id}.maskmeta.json", json_bytes(payload)


def oracle_payloads(
    sample_id: str, fits: dict[str, dict[str, Any]], meta: dict[str, Any]
) -> Iterator[tuple[str, bytes]]:
    payload = {
        "schema_version": SCHEMA_ORACLE,
        "sample_id": sample_id,
        "meta": meta,
        "fits": fits,
    }
    yield f"{sample_id}.oracle.json", json_bytes(payload)


def pack_maskviews(rows: Iterable[tuple[str, Any, Any, dict[str, Any]]], out_root: Path,
                   source_label: str = "where_a.maskviews") -> dict[str, Any]:
    def gen():
        for sid, low, hi, meta in rows:
            yield from maskview_payloads(sid, low, hi, meta)

    return build_from_memory(
        gen(), Path(out_root), shard_size_bytes=MASKVIEW_SHARD_BYTES,
        producer=PRODUCER, source_label=source_label,
    )


def pack_oracle(rows: Iterable[tuple[str, dict[str, Any], dict[str, Any]]], out_root: Path,
                source_label: str = "where_a.oracle") -> dict[str, Any]:
    def gen():
        for sid, fits, meta in rows:
            yield from oracle_payloads(sid, fits, meta)

    return build_from_memory(
        gen(), Path(out_root), shard_size_bytes=ORACLE_SHARD_BYTES,
        producer=PRODUCER, source_label=source_label,
    )


def write_basis(out_dir: Path, arm: str, projector, extra: dict[str, Any]) -> dict[str, Any]:
    """Publish the calibrated ``B`` plus everything needed to reproduce it."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    w = projector.weight.detach().to(torch.float32).cpu().numpy()
    wb = _npy_bytes(w)
    (out_dir / "B.npy").write_bytes(wb)
    meta = {
        "schema_version": SCHEMA_BASIS,
        "arm": arm,
        "shape": list(w.shape),
        "sha256": hashlib.sha256(wb).hexdigest(),
        "projector": projector.facts(),
        **extra,
    }
    (out_dir / "basis.json").write_bytes(json_bytes(meta))
    return meta


def verify_published(root: Path, n_random: int = 32, seed: int = 0) -> dict[str, Any]:
    """Random-read + checksum verification of a published shard set (2.3)."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    rows: list[dict[str, Any]] = []
    for p in sorted((root / "indexes").glob("shard-*.idx.jsonl")):
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(rows), size=min(n_random, len(rows)), replace=False)
    bad = []
    for i in pick:
        r = rows[int(i)]
        data = read_member(root, r["shard"], r["offset_data"], r["length"])
        if hashlib.sha256(data).hexdigest() != r["sha256"]:
            bad.append(r["member"])
    return {
        "root": str(root),
        "status": manifest.get("status"),
        "member_count": manifest.get("member_count"),
        "sample_count": manifest.get("sample_count"),
        "shard_count": manifest.get("shard_count"),
        "index_rows": len(rows),
        "n_random_checked": int(len(pick)),
        "checksum_failures": bad,
        "ok": not bad and manifest.get("member_count") == len(rows),
    }
