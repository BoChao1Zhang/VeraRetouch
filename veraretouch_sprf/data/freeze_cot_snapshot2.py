# 源自 experiments/prs/EPR-051_masked-restore-production/stage0/sprf/vlmsft/freeze_cot_snapshot2.py @ 2026-09-06（原件未入 git；原 sha256 见 veraretouch_sprf/PROVENANCE.md），逐字复制/仅改导入与路径解析（EPR-052）。
"""EPR-051 / SFT+ADAPT -- freeze a settled snapshot of the CoT-100k annotation run.

The annotation drivers are still appending to out/part-*.jsonl.  This tool takes
each part file's byte prefix up to the LAST newline (so a half-written record is
never read), records that prefix's length + sha256, and writes one immutable
records.jsonl.  Later growth cannot enter this snapshot: re-running with the
frozen manifest re-reads exactly the same byte prefixes.

Filters: ok AND parsed AND depth==6 AND len(answer.cot)==6.  Dedup by key
(first occurrence in (part, line) order wins).

Assertions (hard, die on failure):
  H1  snapshot ids  n  heldout ids (snapshot_newdata_v3.heldout_ids.json) == 0
  H2  every kept record has exactly 6 cot steps with step index 1..6
  H3  no duplicate keys survive
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from veraretouch_sprf import _paths as _P  # EPR-052：替代原 sys.path 注入块
_P.ensure_sys_path()
STAGE0 = _P.STAGE0
HELDOUT = STAGE0 / "snapshot_newdata_v3.heldout_ids.json"
SRC = Path("/home/bc/data/builds/epr051_cot100k")


def die(msg: str) -> None:
    print(f"FATAL: {msg}", file=sys.stderr)
    raise SystemExit(2)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def val_bucket(key: str, salt: str) -> int:
    """sha1(salt + ':' + key) -> 0..999 bucket."""
    d = hashlib.sha1(f"{salt}:{key}".encode()).hexdigest()
    return int(d[:6], 16) % 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/bc/data/runs/epr051_vlmsft/snap_sft2")
    ap.add_argument("--s1-dir", default="/home/bc/data/runs/epr051_vlmsft/snap_sft1")
    ap.add_argument("--val-salt", default="epr051-vlmsft-val-v1")
    ap.add_argument("--val-permille", type=int, default=50)  # 5%
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    held = set(json.loads(HELDOUT.read_text()))
    parts = sorted((SRC / "out").glob("part-*.jsonl"))
    if not parts:
        die("no part files")

    part_manifest = []
    kept: dict[str, dict] = {}
    n_lines = n_bad_json = n_filtered = n_dup = 0

    for p in parts:
        raw = p.read_bytes()
        cut = raw.rfind(b"\n")
        prefix = raw[: cut + 1] if cut >= 0 else b""
        nl = 0
        nk = 0
        for line in prefix.split(b"\n"):
            if not line.strip():
                continue
            nl += 1
            n_lines += 1
            try:
                r = json.loads(line)
            except Exception:
                n_bad_json += 1
                continue
            ans = r.get("answer") or {}
            cot = ans.get("cot") or []
            if not (r.get("ok") and r.get("parsed") and r.get("depth") == 6 and len(cot) == 6):
                n_filtered += 1
                continue
            if [c.get("step") for c in cot] != [1, 2, 3, 4, 5, 6]:
                n_filtered += 1
                continue
            k = r["key"]
            if k in kept:
                n_dup += 1
                continue
            kept[k] = r
            nk += 1
        part_manifest.append(dict(part=p.name, frozen_bytes=len(prefix), n_lines=nl,
                                  n_kept=nk, sha256=sha256_bytes(prefix)))

    keys = sorted(kept)
    ids = {kept[k]["id"] for k in keys}
    inter = sorted(ids & held)
    if inter:
        die(f"H1 heldout contamination: {len(inter)} ids, e.g. {inter[:5]}")

    rec_path = out / "records.jsonl"
    with open(rec_path, "w") as fh:
        for k in keys:
            fh.write(json.dumps(kept[k], ensure_ascii=False, sort_keys=True) + "\n")

    val = [k for k in keys if val_bucket(k, args.val_salt) < args.val_permille]
    train = [k for k in keys if val_bucket(k, args.val_salt) >= args.val_permille]
    (out / "split_val_keys.json").write_text(json.dumps(val, indent=0))
    (out / "split_train_keys.json").write_text(json.dumps(train, indent=0))

    s1keys = set()
    s1p = Path(args.s1_dir) / "records.jsonl"
    if s1p.exists():
        s1keys = {json.loads(l)["key"] for l in open(s1p)}
    kset = set(keys)
    freeze = dict(
        name="epr051-vlmsft-cot-snapshot-2",
        frozen_at=__import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
        source_dir=str(SRC),
        parts=part_manifest,
        n_lines_read=n_lines,
        n_bad_json=n_bad_json,
        n_filtered_out=n_filtered,
        n_dup_keys=n_dup,
        n_records=len(keys),
        records_file=str(rec_path),
        records_sha256=sha256_file(rec_path),
        keys_sha256=hashlib.sha256(json.dumps(keys).encode()).hexdigest(),
        heldout_ids_file=str(HELDOUT),
        heldout_ids_sha256=sha256_file(HELDOUT),
        heldout_intersection=0,
        split=dict(salt=args.val_salt, permille=args.val_permille,
                   rule="int(sha1(salt+':'+key)[:6],16)%1000 < permille -> val",
                   n_train=len(train), n_val=len(val)),
        selection_rule_sha256=sha256_file(SRC / "select" / "rule.json"),
        prompt_freeze_sha256=sha256_file(SRC / "prompt_freeze.json"),
        run_args_sha256=sha256_file(SRC / "run_args.json"),
        tool_sha256=sha256_file(Path(__file__)),
        s1=dict(dir=args.s1_dir, n_s1=len(s1keys),
                s1_subset_of_s2=bool(s1keys <= kset),
                n_s1_in_s2=len(s1keys & kset), n_s1_not_in_s2=len(s1keys - kset),
                n_new_vs_s1=len(kset - s1keys)),
    )
    (out / "FREEZE.json").write_text(json.dumps(freeze, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in freeze.items() if k != "parts"}, indent=1))
    print(f"parts={len(part_manifest)} n_records={len(keys)} train/val={len(train)}/{len(val)}")


if __name__ == "__main__":
    main()
