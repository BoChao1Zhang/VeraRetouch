#!/usr/bin/env python
"""Run ONE shard of ``q3vl.whereb.scripts.make_generated_context``.

Why this exists
---------------
The producer has ``--limit`` but no ``--shard`` / ``--offset``, so as written the
only way to use both H100s is to give each card a different *split* -- and the
split that matters (``train``, 159,215 samples) is one indivisible job of many
hours.  This wrapper adds sharding **without touching a single file under
``q3vl/``**: it slices ``WhereBDataset.refs`` after the package's own sanctioned
factory (:func:`q3vl.whereb.data.open_dataset`) has built the dataset, then hands
control to the producer's own ``main()``.  Everything downstream of the slice --
prompt construction, generation, record building, publication -- is the author's
code, unmodified.

Three things the wrapper *forces* rather than trusts the caller with:

``--out-root``
    is suffixed with ``shard<i>of<n>``.  Publication is atomic and refuses an
    existing directory, so two shards pointed at one root would make the second
    one die -- after hours of generation.

``--report-dir``
    is suffixed the same way.  The producer writes ``genctx_<leaf>.json`` into
    it, and ``<leaf>`` does not carry the shard number: eight shards sharing one
    report dir would **silently overwrite each other's report** and leave a file
    that looks complete but describes 1/8 of the work.  Same failure class as
    the Stage-What R5 incident (a default ``--out`` that quietly replaced a
    delivered report).

``--limit``
    is applied to the **whole split first** and only then sharded, so
    ``--limit 64 --num-shards 8`` means "64 samples of the split, 8 per shard",
    never "64 per shard".  (``open_dataset`` truncates before we slice.)

Partitions
----------
``interleave`` (default) takes ``refs[i::n]``: every shard gets the same mix of
builds, image sizes and prompt lengths, so the two cards finish within a few
minutes of each other.  ``contiguous`` takes an equal block; it is kept only
because it makes a resumed/redone shard reproducible from the index alone.
Both are exhaustive and disjoint by construction, asserted in
``dryrun/test_shard_and_merge.py``.

The shards publish independently and are merged by ``merge_genctx.py``; a shard
that dies costs one shard, not the whole split.  That matters here: the producer
holds every record in memory and publishes once at the very end, so a job killed
at 95% produces nothing at all.

Usage (never run directly for a real job -- use ``genctx_dual.sh``):
    python genctx_shard.py --split train --shard 0 --num-shards 8 \
        --out-root  /mnt/nfs/.../genwhere/_shards/two_segment/train \
        --report-dir <deliverable>/genctx_jobs/reports/two_segment/train \
        --checkpoint /home/bc/data/runs/q3vl_base_sft_20260804/checkpoint-4976 \
        --batch-size 64
"""

from __future__ import annotations

# The campaign-wide R6 guard: sqlite3 must be imported before torch, and this is
# an entry point, so it belongs here (see the producer's own module docstring).
import sqlite3  # noqa: F401  (import order is the point)

import argparse
import json
import sys
from pathlib import Path

# `python <abs path>/genctx_shard.py` puts *this* directory on sys.path, not the
# repo, and `cd /home/bc/VeraRetouch` does not help (only `-m` adds the cwd).
# Without this the wrapper dies on `import q3vl` after the caller has already
# been told the job started.  Caught by dryrun/test_shard_and_merge.py.
_REPO = Path(__file__).resolve().parents[4]
if (_REPO / "q3vl").is_dir() and str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

__all__ = ["shard_indices", "shard_tag", "build_inner_argv"]

#: arguments this wrapper owns; passing them through would defeat the point
OWNED = ("--split", "--out-root", "--report-dir")


def shard_indices(n_total: int, shard: int, num_shards: int, mode: str) -> list[int]:
    """Indices of ``shard`` in a ``num_shards``-way partition of ``range(n_total)``.

    Exhaustive and pairwise disjoint for both modes; sizes differ by at most 1.
    """
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if not 0 <= shard < num_shards:
        raise ValueError(f"shard {shard} out of range for num_shards={num_shards}")
    if n_total < 0:
        raise ValueError(f"n_total must be >= 0, got {n_total}")
    if mode == "interleave":
        return list(range(shard, n_total, num_shards))
    if mode == "contiguous":
        lo = (n_total * shard) // num_shards
        hi = (n_total * (shard + 1)) // num_shards
        return list(range(lo, hi))
    raise ValueError(f"unknown shard mode {mode!r}")


def shard_tag(shard: int, num_shards: int) -> str:
    return f"shard{shard:02d}of{num_shards:02d}"


def build_inner_argv(args, rest: list[str]) -> list[str]:
    """The argv the producer's ``main()`` will parse."""
    bad = [a for a in rest if a.split("=", 1)[0] in OWNED]
    if bad:
        raise SystemExit(
            f"{bad} is owned by genctx_shard.py and must not be passed through; "
            "give --out-root / --report-dir / --split to the wrapper instead"
        )
    tag = shard_tag(args.shard, args.num_shards)
    return [
        "--split", args.split,
        "--out-root", str(Path(args.out_root) / tag),
        "--report-dir", str(Path(args.report_dir) / tag),
        *rest,
    ]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="every other flag is passed straight through to "
               "q3vl.whereb.scripts.make_generated_context (--checkpoint, "
               "--batch-size, --max-new-tokens, --forced-color-prefix, --limit, ...)",
    )
    ap.add_argument("--split", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--shard-mode", default="interleave",
                    choices=("interleave", "contiguous"))
    ap.add_argument("--out-root", required=True,
                    help="parent of the shard roots; 'shard<i>of<n>' is appended")
    ap.add_argument("--report-dir", required=True,
                    help="parent of the shard reports; 'shard<i>of<n>' is appended")
    ap.add_argument("--plan-only", action="store_true",
                    help="print the partition and the inner argv, touch no GPU")
    args, rest = ap.parse_known_args()

    inner_argv = build_inner_argv(args, rest)
    tag = shard_tag(args.shard, args.num_shards)
    if args.plan_only:
        print(json.dumps({"shard": args.shard, "num_shards": args.num_shards,
                          "shard_mode": args.shard_mode, "tag": tag,
                          "inner_argv": inner_argv}, indent=2), flush=True)
        return 0

    # imported late so --help / --plan-only cost neither torch nor the GPU
    from q3vl.whereb.scripts import make_generated_context as producer

    real_open_dataset = producer.open_dataset

    def sharded_open_dataset(split, **kwargs):
        ds, info = real_open_dataset(split, **kwargs)
        n_full = len(ds.refs)
        idx = shard_indices(n_full, args.shard, args.num_shards, args.shard_mode)
        ds.refs = [ds.refs[i] for i in idx]
        info.update({
            "shard": args.shard,
            "num_shards": args.num_shards,
            "shard_mode": args.shard_mode,
            "shard_tag": tag,
            "n_samples_before_shard": n_full,
            "n_samples": len(ds.refs),
            "shard_first_sample_id": ds.refs[0].sample_id if ds.refs else None,
            "shard_last_sample_id": ds.refs[-1].sample_id if ds.refs else None,
        })
        if not ds.refs:
            raise SystemExit(
                f"{tag} of split {split!r} is empty ({n_full} samples, "
                f"{args.num_shards} shards); refusing to publish an empty dataset"
            )
        print(json.dumps({"shard_plan": {k: info[k] for k in (
            "shard", "num_shards", "shard_mode", "shard_tag",
            "n_samples_before_shard", "n_samples",
            "shard_first_sample_id", "shard_last_sample_id")}}), flush=True)
        return ds, info

    producer.open_dataset = sharded_open_dataset
    sys.argv = ["make_generated_context", *inner_argv]
    return producer.main()


if __name__ == "__main__":
    raise SystemExit(main())
