# 源自 experiments/prs/EPR-051_masked-restore-production/stage0/sprf/vlmsft/epr051_heldout_split.py @ 2026-09-06（原件未入 git；原 sha256 见 veraretouch_sprf/PROVENANCE.md），逐字复制/仅改导入与路径解析（EPR-052）。
"""EPR-051 headline: split the frozen heldout-d6 key list into two card-halves.

The order of heldout_d6_keys.json is the frozen order and is NEVER re-sorted
here; half A is the first ceil(n/2), half B the remainder, so the union is the
original list and the concatenation A+B reproduces it exactly.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path


def sha_list(keys) -> str:
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", default="/home/bc/data/runs/epr051_vlmsft/heldout_d6_keys.json")
    ap.add_argument("--out-dir", default="/home/bc/data/runs/epr051_vlmsft/heldout_d6_split")
    ap.add_argument("--expect-n", type=int, default=1464)
    args = ap.parse_args()

    keys = json.loads(Path(args.keys).read_text())
    if len(keys) != args.expect_n:
        raise SystemExit(f"FATAL: {len(keys)} keys != expected {args.expect_n}")
    if len(set(keys)) != len(keys):
        raise SystemExit("FATAL: source key list contains duplicates")

    cut = (len(keys) + 1) // 2
    a, b = keys[:cut], keys[cut:]
    assert a + b == keys, "split is not order-preserving"
    assert not (set(a) & set(b)), "halves overlap"

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "half_a.json").write_text(json.dumps(a))
    (out / "half_b.json").write_text(json.dumps(b))
    man = dict(source=args.keys, n_total=len(keys), source_sha256=sha_list(keys),
               half_a=dict(n=len(a), sha256=sha_list(a), file=str(out / "half_a.json")),
               half_b=dict(n=len(b), sha256=sha_list(b), file=str(out / "half_b.json")),
               order="frozen order of the source list; A+B == source")
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print("SPLIT " + json.dumps({k: man[k] for k in ("n_total", "source_sha256")}))
    print(f"  half_a n={len(a)} sha={man['half_a']['sha256'][:12]}")
    print(f"  half_b n={len(b)} sha={man['half_b']['sha256'][:12]}")


if __name__ == "__main__":
    main()
