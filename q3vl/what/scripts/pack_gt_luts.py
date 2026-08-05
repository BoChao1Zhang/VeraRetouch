"""Publish the GT LUT tables as indexed tar shards (protocol 2.3).

    "Newly produced oracle latents, basis metadata, continuous LUT codes or
     visualisation indexes are dataset derivatives too and must obey the
     cold/hot tiering and the indexed-tar-shard contract."

``T_gt`` is not something this campaign renders: every ``sft2seg`` record already
carries ``preset_path``, a real ``.cube`` (98.5%) or ``.3dl`` (1.5%) file under
``/home/bc/data/datasets/recipes``, and ``lut_id -> preset_path`` is 1:1
(verified 2026-08-05 on 481 sampled ids across train / V_what / T_lut_unseen:
0 conflicts, 481/481 present).  What this job does is turn ~3.4k text files
totalling 8.8 GiB into one published, checksummed, randomly readable binary set,
so that a training step does not re-parse a 32768-line text file.

Per-LUT members
---------------
``<lut_id>.lut.npy``       float32 ``(S, S, S, 3)`` indexed ``[r, g, b]``
``<lut_id>.lutmeta.json``  size, domain, source path, source digest, parser version

float32, not float16: the ``.cube`` files carry six decimals and the protocol
12.1 bake gate is stated at 1e-4 RGB.  float16's resolution at 1.0 is ~1e-3, so a
float16 store would put the gate below the noise floor of its own ground truth.

NOT EXECUTED.  Scheduling is the main agent's call; the GPUs are busy with Base
SFT and this job is IO-bound on NFS.

Usage
-----
    python -m q3vl.what.scripts.pack_gt_luts --splits train V_where V_what \
        T_final T_lut_unseen --out /mnt/nfs/bc/data/datasets/what-20260805/gtluts
"""

from __future__ import annotations

# --- environment guard: sqlite3 must be imported BEFORE torch ---------------
# Verified 2026-08-05 in the campaign env (/home/bc/envs/q3vl_sft):
#   import torch; import sqlite3  -> ImportError, libstdc++ CXXABI_1.3.15 not found
#   import sqlite3; import torch  -> fine
# torch loads a libstdc++ that shadows the one `_sqlite3`'s dependency chain
# (libicui18n) needs, so any process that touches torch first can never open a
# published shard afterwards -- `q3vl.data.shardio` imports sqlite3, and every
# store in this campaign goes through it.  Importing it first costs nothing and
# inoculates the whole process.  This is campaign-wide, not Stage-What specific:
# `q3vl.whereb.stores` sits on the same chain (see NOTES R6).
import sqlite3  # noqa: F401  (import order is the point)

import argparse
import hashlib
import io
import json
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from q3vl.data.shardio import build_from_memory
from q3vl.what.config import GTLUT_DIR, GTLUT_SHARD_BYTES, SCHEMA_GTLUT
from q3vl.what.data import WhatDataset
from q3vl.what.lut import load_gt_table, table_digest

PRODUCER = "q3vl.what.scripts.pack_gt_luts/1"
SPLITS = ("train", "V_where", "V_what", "T_final", "T_lut_unseen")


def json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"


def npy_bytes(a: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, a, allow_pickle=False)
    return buf.getvalue()


def collect_lut_paths(splits=SPLITS, verify: str = "checksum"
                      ) -> tuple[dict[str, str], dict[str, Any]]:
    """``lut_id -> preset_path`` over every split, with a conflict report.

    A ``lut_id`` that maps to two different files is a data bug, not something to
    resolve by picking one: it would mean two different functions share an id and
    every ``T_lut_unseen`` claim built on that id is void.
    """
    mapping: dict[str, str] = {}
    conflicts: list[dict[str, str]] = []
    per_split: dict[str, int] = {}
    for split in splits:
        ds = WhatDataset(split, need_mask=False, verify=verify)
        seen = set()
        for i in range(len(ds)):
            rec = ds.record(i)
            lid, path = rec["lut_id"], rec["preset_path"]
            seen.add(lid)
            if lid in mapping and mapping[lid] != path:
                conflicts.append({"lut_id": lid, "a": mapping[lid], "b": path,
                                  "split": split})
            mapping[lid] = path
        per_split[split] = len(seen)
    return mapping, {"n_lut": len(mapping), "per_split": per_split,
                     "conflicts": conflicts}


def payloads(lut_id: str, path: str) -> Iterator[tuple[str, bytes]]:
    tbl = load_gt_table(path, lut_id)
    arr = tbl.table.numpy().astype(np.float32)
    raw = Path(path).read_bytes()
    meta = {
        "schema_version": SCHEMA_GTLUT,
        "lut_id": lut_id,
        "size": int(tbl.size),
        "domain_min": [float(v) for v in tbl.domain_min],
        "domain_max": [float(v) for v in tbl.domain_max],
        "source": str(path),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_bytes": len(raw),
        "table_digest": table_digest(arr),
        "layout": "table[r, g, b, c], float32, domain-normalised coordinates",
        "parser": "dataset_build.lut_io.load_lut (grid[b,g,r]) transposed once",
        "producer": PRODUCER,
    }
    yield f"{lut_id}.lut.npy", npy_bytes(arr)
    yield f"{lut_id}.lutmeta.json", json_bytes(meta)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--out", default=str(GTLUT_DIR))
    ap.add_argument("--shard-bytes", type=int, default=GTLUT_SHARD_BYTES)
    ap.add_argument("--verify", default="checksum")
    ap.add_argument("--dry-run", action="store_true",
                    help="collect and report the mapping, write nothing")
    args = ap.parse_args()

    t0 = time.time()
    mapping, report = collect_lut_paths(tuple(args.splits), args.verify)
    missing = sorted(k for k, v in mapping.items() if not Path(v).exists())
    report["n_missing"] = len(missing)
    report["missing"] = missing[:32]
    report["elapsed_collect_s"] = round(time.time() - t0, 1)
    print(json.dumps(report, ensure_ascii=False, indent=1), flush=True)
    if report["conflicts"]:
        raise SystemExit("lut_id -> preset_path is not 1:1; refusing to publish")
    if missing:
        raise SystemExit(f"{len(missing)} lut_id do not resolve to a file")
    if args.dry_run:
        return 0

    def gen():
        for lid in sorted(mapping):
            yield from payloads(lid, mapping[lid])

    out = build_from_memory(gen(), Path(args.out), shard_size_bytes=args.shard_bytes,
                            producer=PRODUCER, source_label="what.gtluts")
    summary = {**report, "publish": out, "elapsed_s": round(time.time() - t0, 1)}
    Path(args.out, "pack_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == "__main__":                                       # pragma: no cover
    raise SystemExit(main())
