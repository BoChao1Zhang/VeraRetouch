# 源自 experiments/prs/EPR-051_masked-restore-production/stage0/sprf/vlmsft/stage_assets2.py @ 2026-09-06（原件未入 git；原 sha256 见 veraretouch_sprf/PROVENANCE.md），逐字复制/仅改导入与路径解析（EPR-052）。
"""EPR-051 / SFT+ADAPT -- stage the student's ONLY input (y) + per-stage lut ids.

For every key in the frozen snapshot:
  * copy the degraded image y out of the shard's indexed tar (or loose assets/
    dir) to local disk, so training never re-reads /mnt/nfs-ro per epoch;
  * pull the journal row and record the per-stage chain facts.

Naming trap this tool pins down (measured, epr051_cot100k.py:273-289):
  plan["assets"]["current"] = the DEGRADED image  (`<id>.after.png`)  = y
  plan["assets"]["target"]  = the CLEAN image     (`<id>.src.png`)    = x0
The annotation prompt showed CURRENT=y first and TARGET=x0 second, so the CoT
is a RESTORATION recipe y -> x0.  Only `current` is staged here: x0 must never
reach the student.

Index order trap (measured, epr051_cot_pilot.py:308 `order = reversed(...)`):
  CoT step p (1..6, restoration order) <-> chain position k = 6 - p (0-based,
  degradation order, the order row["luts"] / row["steps"] use).
  Both are written out per key so no downstream file has to re-derive it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PLAN = Path("/home/bc/data/builds/epr051_cot100k/select/plan.jsonl")


def die(m):
    print(f"FATAL: {m}", flush=True)
    raise SystemExit(2)


class TarPool:
    """os.pread per shard tar; no seek races across threads."""

    def __init__(self):
        self.fds: dict[str, int] = {}

    def read(self, tar: str, off: int, size: int) -> bytes:
        fd = self.fds.get(tar)
        if fd is None:
            fd = os.open(tar, os.O_RDONLY)
            self.fds[tar] = fd
        return os.pread(fd, size, off)


def journal_row(plan: dict) -> dict:
    with open(Path(plan["dir"]) / "pairs.jsonl", "rb") as fh:
        fh.seek(plan["off"])
        row = json.loads(fh.read(plan["len"]))
    if row["id"] != plan["id"]:
        die(f"{plan['key']}: journal offset points at {row['id']}")
    return row


def safe_name(key: str) -> str:
    return key.replace("|", "__")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="/home/bc/data/runs/epr051_vlmsft/snap_sft2")
    ap.add_argument("--out", default="/home/bc/data/runs/epr051_vlmsft/assets_y2")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    snap = Path(args.snapshot)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    keys = [json.loads(l)["key"] for l in open(snap / "records.jsonl")]
    if args.limit:
        keys = keys[: args.limit]
    want = set(keys)

    plan = {}
    for line in open(PLAN):
        r = json.loads(line)
        if r["key"] in want:
            plan[r["key"]] = r
    missing = want - set(plan)
    if missing:
        die(f"{len(missing)} snapshot keys absent from select/plan.jsonl, e.g. {sorted(missing)[:3]}")
    print(f"[plan] matched {len(plan)} keys", flush=True)

    pools = {}
    index: dict[str, dict] = {}
    lock_err = []

    def work(key: str):
        p = plan[key]
        tid = __import__("threading").get_ident()
        pool = pools.setdefault(tid, TarPool())
        dst = out / f"{safe_name(key)}.png"
        spec = p["assets"]
        try:
            if spec["mode"] == "tar":
                off, size = spec["current"]
                blob = pool.read(spec["tar"], int(off), int(size))
            else:
                blob = Path(spec["current"]).read_bytes()
            if not dst.exists() or dst.stat().st_size != len(blob):
                tmp = dst.with_suffix(".png.tmp")
                tmp.write_bytes(blob)
                tmp.replace(dst)
            row = journal_row(p)
            steps = row["steps"]
            if len(steps) != 6:
                die(f"{key}: journal has {len(steps)} steps, expected 6")
            return key, dict(
                png=str(dst), bytes=len(blob),
                sha256=hashlib.sha256(blob).hexdigest(),
                shard=p["shard"], dir=p["dir"], off=p["off"], len=p["len"],
                asset=p["asset"],
                chain=[dict(k=k, kind=s["kind"], lut=s["lut"], grid=row["grids"][k])
                       for k, s in enumerate(steps)],
                # CoT step p (1-based) -> chain index k
                cot_step_to_chain_k={str(pp): 6 - pp for pp in range(1, 7)},
                calib_s=float(row["calib"]["s"]),
            )
        except Exception as exc:
            lock_err.append((key, f"{type(exc).__name__}: {exc}"))
            return key, None

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (key, rec) in enumerate(ex.map(work, keys)):
            if rec is not None:
                index[key] = rec
            if (i + 1) % 1000 == 0:
                print(f"  staged {i+1}/{len(keys)}", flush=True)

    if lock_err:
        die(f"{len(lock_err)} failures, e.g. {lock_err[:3]}")

    kinds = {}
    for r in index.values():
        kinds[tuple(c["kind"] for c in r["chain"])] = kinds.get(
            tuple(c["kind"] for c in r["chain"]), 0) + 1
    payload = dict(n=len(index), out_dir=str(out),
                   chain_kind_orders={" ".join(k): v for k, v in kinds.items()},
                   index=index)
    ip = snap / "assets_index.json"
    ip.write_text(json.dumps(payload))
    print(f"[done] n={len(index)} index={ip}")
    print("[chain kind orders]", json.dumps(payload["chain_kind_orders"], indent=1))
    tot = sum(r["bytes"] for r in index.values())
    print(f"[bytes] {tot/2**30:.2f} GiB")


if __name__ == "__main__":
    main()
