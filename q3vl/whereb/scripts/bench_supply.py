#!/usr/bin/env python
"""Mock training loop: the real data supply, a fake forward (PERF-1).

Answers one question and only that one -- *how many seconds per optimizer step
does an arm spend getting data into the training thread* -- so the effect of the
local shard cache and of the prefetch threads can be measured without touching a
running arm and without a GPU.

It walks the **real** ``BalancedContextSampler`` order for the arm's own seed,
builds samples through the **real** ``WhereBDataset`` and reads the same oracle
and generated-context members ``BatchBuilder`` would, then throws the result
away.  ``--start-batch`` picks a window of the permutation, so pointing it at
the tail measures cold reads even while another arm is running.

    python -m q3vl.whereb.scripts.bench_supply --n-batches 12 --prefetch 0
    python -m q3vl.whereb.scripts.bench_supply --n-batches 12 --prefetch 6

Per-stage timings are only collected with ``--prefetch 0``: with workers running
they would be wall-clock overlapping and the shares would not add up, which is
exactly the kind of number that gets quoted out of context later.
"""

from __future__ import annotations

import sqlite3  # noqa: F401  (import order guard, see run_where_b)

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

from q3vl.train.shards import ShardStore, shard_cache_facts
from q3vl.whereb.config import (
    BASIS_ARM,
    GENCTX_DIR,
    ORACLE_NAMESPACE,
    SEED,
    WHERE_A_MASKVIEW_DIR,
    WHERE_A_ORACLE_DIR,
)
from q3vl.whereb.context import BalancedContextSampler, GENERATED
from q3vl.whereb.data import WhereBDataset
from q3vl.whereb.prefetch import SamplePrefetcher
from q3vl.whereb.stores import (
    DEFAULT_BLOB_CACHE_BYTES,
    GenContextStore,
    MaskViewStore,
    OracleStore,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--start-batch", type=int, default=19000)
    ap.add_argument("--n-batches", type=int, default=12)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--prefetch", type=int, default=0)
    ap.add_argument("--readout", default="band")
    ap.add_argument("--verify", default="checksum")
    ap.add_argument("--torch-threads", type=int, default=4)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    torch.set_num_threads(args.torch_threads)
    t = time.perf_counter
    t0 = t()
    mv = MaskViewStore(Path(WHERE_A_MASKVIEW_DIR) / args.split)
    oracle = OracleStore(Path(WHERE_A_ORACLE_DIR) / BASIS_ARM / ORACLE_NAMESPACE
                         / args.split, cache_bytes=DEFAULT_BLOB_CACHE_BYTES)
    genctx = GenContextStore(Path(GENCTX_DIR) / args.split,
                             cache_bytes=DEFAULT_BLOB_CACHE_BYTES)
    store = ShardStore("/", verify=args.verify)
    ds = WhereBDataset(args.split, store=store, maskviews=mv,
                       need_mask=True, verify=args.verify)
    setup_s = t() - t0
    print(f"[setup] {len(ds)} samples, {setup_s:.1f}s, "
          f"shard_cache={json.dumps(shard_cache_facts())}", flush=True)

    sampler = BalancedContextSampler(len(ds), args.micro_batch, seed=SEED,
                                     teacher_fraction=0.5)
    batches = list(sampler)
    lo = max(0, min(args.start_batch, len(batches) - args.n_batches))
    window = batches[lo:lo + args.n_batches]

    stages: dict[str, list[float]] = defaultdict(list)

    def warm(sample, mode: str) -> None:
        a = t()
        if mode == GENERATED:
            genctx.prime(sample.sample_id, GenContextStore.SUFFIX)
        stages["read_genctx"].append(t() - a)
        a = t()
        if not sample.is_global:
            oracle.prime(sample.sample_id, OracleStore.SUFFIX)
        stages["read_oracle"].append(t() - a)

    supply = SamplePrefetcher(ds, window, workers=args.prefetch, warm=warm)
    wall0 = t()
    n_samples = 0
    for micro, samples in supply:
        n_samples += len(samples)
    wall = t() - wall0

    per_micro = wall / max(1, len(window))
    out = {
        "split": args.split, "start_batch": lo, "n_batches": len(window),
        "micro_batch": args.micro_batch, "grad_accum": args.grad_accum,
        "prefetch_workers": args.prefetch, "verify": args.verify,
        "setup_s": round(setup_s, 2),
        "wall_s": round(wall, 3),
        "ms_per_micro_batch": round(per_micro * 1e3, 1),
        "ms_per_sample": round(per_micro / args.micro_batch * 1e3, 2),
        "s_per_optimizer_step": round(per_micro * args.grad_accum, 3),
        "shard_cache": shard_cache_facts(),
        "genctx_blob_cache": genctx.blobs.facts() if genctx.blobs else None,
        "oracle_blob_cache": oracle.blobs.facts() if oracle.blobs else None,
    }
    if args.prefetch <= 0 and stages:
        out["stages"] = {k: {"n": len(v), "total_s": round(sum(v), 3),
                             "mean_ms": round(sum(v) / len(v) * 1e3, 3),
                             "share": round(sum(v) / wall, 4)}
                         for k, v in sorted(stages.items(), key=lambda kv: -sum(kv[1]))}
    print(json.dumps(out, indent=2), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":                                     # pragma: no cover
    raise SystemExit(main())
