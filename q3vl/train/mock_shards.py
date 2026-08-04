"""Build tiny indexed tar shards for unit tests and dry runs.

The real shards come from the parallel S0-DATA task. This module produces the
same *shape* of artefact -- uncompressed tar, JSONL index carrying
``shard/member/offset/length/size/checksum``, and a terminal manifest -- so the
reader, collator and loss mask can be exercised end to end before the real data
lands. Member offsets are read back from the written tar rather than predicted,
so the index is correct regardless of tar header format.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import tarfile
from pathlib import Path
from typing import Any

from PIL import Image

SCHEMA_VERSION = "sft2seg/mock-1"


def _png_bytes(h: int, w: int, seed: int) -> bytes:
    rng = random.Random(seed)
    img = Image.new("RGB", (w, h))
    img.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(h * w)])
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def build_mock_shard(
    out_dir: str | Path,
    n_samples: int = 8,
    split: str = "train",
    shard_name: str = "shard-00000.tar",
    sizes: list[tuple[int, int]] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Write ``out_dir/{shards/<shard>, index/<split>.jsonl, manifest.json}``."""
    out_dir = Path(out_dir)
    shard_dir = out_dir / "shards"
    index_dir = out_dir / "index"
    shard_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    sizes = sizes or [(512, 512), (512, 768), (640, 512), (512, 2048)]

    payloads: dict[str, bytes] = {}
    records: dict[str, dict[str, Any]] = {}
    for i in range(n_samples):
        sid = f"{split}-{i:05d}"
        h, w = sizes[i % len(sizes)]
        payloads[f"{sid}.png"] = _png_bytes(h, w, seed + i)
        rec = {
            "sample_id": sid,
            "split": split,
            "instruction": f"Warm the light and lift the subject in frame {i}.",
            "where": f"The edit covers the subject in the {'left' if i % 2 else 'right'} half.",
            "color": (
                "The scene reads flat and cool.\n"
                "The global cast leans blue.\n"
                "Skin tones look drained.\n"
                "Lift exposure a touch.\n"
                "Warm the global balance.\n"
                "Recover the skin warmth."
            ),
            "build": "g1" if i % 2 == 0 else "l1",
            "source_image_id": f"src-{i:05d}",
        }
        records[f"{sid}.json"] = rec
        payloads[f"{sid}.json"] = json.dumps(rec, ensure_ascii=False).encode("utf-8")

    shard_path = shard_dir / shard_name
    with tarfile.open(shard_path, "w", format=tarfile.USTAR_FORMAT) as tar:
        for name, blob in payloads.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(blob)
            tar.addfile(info, io.BytesIO(blob))

    # read offsets back from the finished tar -- never predict them
    offsets: dict[str, tuple[int, int]] = {}
    with tarfile.open(shard_path, "r") as tar:
        for member in tar.getmembers():
            offsets[member.name] = (member.offset_data, member.size)

    rows = []
    for i in range(n_samples):
        sid = f"{split}-{i:05d}"
        members = {}
        for role, name in (("image", f"{sid}.png"), ("record", f"{sid}.json")):
            off, size = offsets[name]
            blob = payloads[name]
            assert size == len(blob)
            members[role] = {
                "shard": shard_name,
                "member": name,
                "offset": off,
                "length": size,
                "size": size,
                "checksum": "sha256:" + hashlib.sha256(blob).hexdigest(),
            }
        rows.append({"sample_id": sid, "split": split, "members": members})

    index_path = index_dir / f"{split}.jsonl"
    with index_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    shard_digest = hashlib.sha256(shard_path.read_bytes()).hexdigest()
    manifest_path = out_dir / "manifest.json"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "digest": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "shards": [{"name": shard_name, "sha256": shard_digest, "n_members": len(payloads)}],
        "counts": {split: {"n_effective": n_samples}},
    }
    if manifest_path.exists():
        prev = json.loads(manifest_path.read_text(encoding="utf-8"))
        prev.setdefault("counts", {}).update(manifest["counts"])
        prev.setdefault("shards", []).extend(manifest["shards"])
        manifest = prev
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "root": str(out_dir),
        "shard": str(shard_path),
        "shard_root": str(shard_dir),
        "index": str(index_path),
        "manifest": str(manifest_path),
        "n_samples": n_samples,
        "records": records,
    }


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sft2seg-mock"
    info = build_mock_shard(target, n_samples=8, split="train")
    build_mock_shard(target, n_samples=4, split="eval", shard_name="shard-eval-00000.tar", seed=100)
    print(json.dumps({k: v for k, v in info.items() if k != "records"}, indent=2))
