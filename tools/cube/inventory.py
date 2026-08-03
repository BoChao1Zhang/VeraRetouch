#!/usr/bin/env python3
"""T2 inventory: reconcile production preset usage against the on-disk .cube corpus.

Collects every distinct `candidates[].preset_path` (and `recipe.preset_path`)
from all groups.jsonl in the journal archive, checks readability, reconciles
against the expected 3,522 (DATA_ASSIGNMENT §2 D-CUBE), scans the recipe roots
for the full on-disk .cube corpus, and emits the action-item-E difference lists
plus the D-CUBE manifest.

Outputs (to --out-dir):
  inventory_summary.json   headline numbers
  dcube_manifest.jsonl     one row per on-disk .cube: id, path, bucket, md5,
                           used_in_prod, builds, preset_ids, major/minor
  used_presets.txt         3,522 production preset paths
  unused_presets.txt       on-disk cubes never used in production (difference pool)
  missing_files.txt        preset paths referenced by journals but unreadable
  dup_content.json         md5 -> paths for byte-identical cube files

Usage:
  python3 inventory.py [--journal-dir DIR] [--recipe-root DIR ...] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cubelib import JOURNAL_ARCHIVE, RECIPE_ROOTS, md5_file, preset_slug

EXPECTED_USED = 3522


def collect_used(journal_dir: str):
    """Scan all groups.jsonl; return per-preset usage info keyed by path."""
    used = {}
    mismatches = []
    files = sorted(glob.glob(os.path.join(journal_dir, "*", "groups.jsonl")))
    if not files:
        raise SystemExit(f"no groups.jsonl found under {journal_dir}")
    for f in files:
        build = os.path.basename(os.path.dirname(f))
        with open(f) as fh:
            for line in fh:
                g = json.loads(line)
                for c in g.get("candidates", []):
                    p = c.get("preset_path")
                    rp = (c.get("recipe") or {}).get("preset_path")
                    if p != rp:
                        mismatches.append((build, c.get("candidate_id"), p, rp))
                    path = p or rp
                    if not path:
                        continue
                    rec = used.setdefault(path, {
                        "preset_ids": set(), "builds": set(),
                        "majors": set(), "minors": set(), "n_candidates": 0,
                    })
                    rec["n_candidates"] += 1
                    rec["builds"].add(build)
                    if c.get("preset_id"):
                        rec["preset_ids"].add(c["preset_id"])
                    if c.get("major"):
                        rec["majors"].add(c["major"])
                    if c.get("minor"):
                        rec["minors"].add(c["minor"])
    return used, mismatches


LUT_EXTS = (".cube", ".3dl")


def scan_disk(recipe_roots):
    disk = []
    for root in recipe_roots:
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in sorted(filenames):
                if fn.lower().endswith(LUT_EXTS):
                    disk.append(os.path.join(dirpath, fn))
    return sorted(disk)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--journal-dir", default=JOURNAL_ARCHIVE)
    ap.add_argument("--recipe-root", action="append", default=None,
                    help="repeatable; default: " + ", ".join(RECIPE_ROOTS))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--no-md5", action="store_true",
                    help="skip content hashing (faster smoke run)")
    args = ap.parse_args()
    roots = args.recipe_root or RECIPE_ROOTS
    os.makedirs(args.out_dir, exist_ok=True)

    used, mismatches = collect_used(args.journal_dir)
    disk = scan_disk(roots)
    disk_set = set(disk)
    used_set = set(used)

    missing = sorted(p for p in used_set if not os.path.isfile(p))
    off_corpus = sorted(used_set - disk_set - set(missing))  # used, readable, outside roots
    unused = sorted(disk_set - used_set)

    md5s = {}
    if not args.no_md5:
        for p in disk + off_corpus:
            try:
                md5s[p] = md5_file(p)
            except OSError as e:
                md5s[p] = f"ERROR:{e}"
    by_md5 = collections.defaultdict(list)
    for p, h in md5s.items():
        by_md5[h].append(p)
    dups = {h: ps for h, ps in by_md5.items() if len(ps) > 1}
    used_md5 = {md5s[p] for p in used_set if p in md5s}

    manifest_path = os.path.join(args.out_dir, "dcube_manifest.jsonl")
    with open(manifest_path, "w") as mf:
        for p in disk + off_corpus:
            u = used.get(p)
            row = {
                "id": preset_slug(p),
                "path": p,
                "format": os.path.splitext(p)[1].lstrip(".").lower(),
                "bucket": os.path.basename(os.path.dirname(p)),
                "md5": md5s.get(p),
                "used_in_prod": u is not None,
                "n_candidates": u["n_candidates"] if u else 0,
                "builds": sorted(u["builds"]) if u else [],
                "preset_ids": sorted(u["preset_ids"]) if u else [],
                "majors": sorted(u["majors"]) if u else [],
                "minors": sorted(u["minors"]) if u else [],
                "dup_of_used": (not u) and md5s.get(p) in used_md5,
            }
            mf.write(json.dumps(row, ensure_ascii=False) + "\n")

    def dump(name, lines):
        with open(os.path.join(args.out_dir, name), "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

    dump("used_presets.txt", sorted(used_set))
    dump("unused_presets.txt", unused)
    dump("missing_files.txt", missing)
    with open(os.path.join(args.out_dir, "dup_content.json"), "w") as f:
        json.dump(dups, f, indent=1)

    per_bucket_used = collections.Counter(
        os.path.basename(os.path.dirname(p)) for p in used_set)
    per_bucket_disk = collections.Counter(
        os.path.basename(os.path.dirname(p)) for p in disk)
    fmt_used = collections.Counter(
        os.path.splitext(p)[1].lstrip(".").lower() for p in used_set)
    fmt_disk = collections.Counter(
        os.path.splitext(p)[1].lstrip(".").lower() for p in disk)
    summary = {
        "expected_used": EXPECTED_USED,
        "distinct_used_preset_paths": len(used_set),
        "reconciles_with_expected": len(used_set) == EXPECTED_USED,
        "preset_path_vs_recipe_mismatches": len(mismatches),
        "used_missing_on_disk": len(missing),
        "used_outside_recipe_roots": len(off_corpus),
        "disk_cube_files": len(disk),
        "unused_on_disk": len(unused),
        "unused_dup_of_used_content": sum(
            1 for p in unused if md5s.get(p) in used_md5),
        "duplicate_content_groups": len(dups),
        "per_bucket_used": dict(per_bucket_used),
        "per_bucket_disk": dict(per_bucket_disk),
        "per_format_used": dict(fmt_used),
        "per_format_disk": dict(fmt_disk),
        "deficit_to_4000": max(0, 4000 - len(used_set)),
    }
    with open(os.path.join(args.out_dir, "inventory_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
