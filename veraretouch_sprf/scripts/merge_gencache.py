# 源自 experiments/prs/EPR-051_masked-restore-production/stage0/sprf/vlmsft/epr051_merge_gencache.py @ 2026-09-06（原件未入 git；原 sha256 见 veraretouch_sprf/PROVENANCE.md），逐字复制/仅改导入与路径解析（EPR-052）。
"""EPR-051 headline: merge the two card-halves' generation caches into one.

Generation is split across two cards; readout and the executor evaluation then
run in ONE process over the merged cache, so the latents all come from a single
model load and a single ordering convention.

Assertions (all fatal):
  key union == the frozen source list   (set equality, not a subset check)
  no key produced by both halves
  no key whose two halves disagree      (cannot happen given disjointness, but
                                         checked so a re-run with overlapping
                                         halves cannot silently pick a winner)
  max_new_tokens / contract agree across halves
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch


def die(m):
    raise SystemExit(f"FATAL: {m}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--half", action="append", required=True,
                    help="gencache_<tag>.pt from one card; pass twice")
    ap.add_argument("--manifest",
                    default="/home/bc/data/runs/epr051_vlmsft/heldout_d6_split/manifest.json")
    ap.add_argument("--out", required=True, help="merged gencache_<tag>.pt")
    args = ap.parse_args()

    man = json.loads(Path(args.manifest).read_text())
    want = json.loads(Path(man["source"]).read_text())
    if len(want) != man["n_total"]:
        die(f"manifest n_total {man['n_total']} != source list {len(want)}")

    merged, seen_meta, per_half = {}, {}, []
    for h in args.half:
        blob = torch.load(h, map_location="cpu")
        gc = blob["gen_cache"]
        dup = set(gc) & set(merged)
        if dup:
            die(f"{h}: {len(dup)} keys already produced by another half, "
                f"e.g. {sorted(dup)[:3]} -- halves must be disjoint")
        for k in ("contract", "max_new_tokens"):
            if k in seen_meta and seen_meta[k] != blob.get(k):
                die(f"{k} disagrees across halves: {seen_meta[k]} vs {blob.get(k)}")
            seen_meta[k] = blob.get(k)
        per_half.append(dict(file=h, n=len(gc), gen_batch=blob.get("gen_batch"),
                             max_new_tokens=blob.get("max_new_tokens")))
        merged.update(gc)

    if set(merged) != set(want):
        miss, extra = set(want) - set(merged), set(merged) - set(want)
        die(f"merged key set != frozen source: missing {len(miss)} "
            f"(e.g. {sorted(miss)[:3]}), extra {len(extra)} "
            f"(e.g. {sorted(extra)[:3]})")
    if len(merged) != len(want):
        die(f"merged {len(merged)} != {len(want)}")

    out = dict(gen_cache=merged, contract=seen_meta.get("contract"),
               max_new_tokens=seen_meta.get("max_new_tokens"),
               gen_batch=per_half[0].get("gen_batch"),
               tag=Path(args.out).stem.replace("gencache_", ""),
               merged_from=per_half, partial=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print("MERGE_OK " + json.dumps(dict(
        n=len(merged), halves=per_half, out=args.out,
        contract=seen_meta.get("contract"),
        max_new_tokens=seen_meta.get("max_new_tokens"))))


if __name__ == "__main__":
    main()
