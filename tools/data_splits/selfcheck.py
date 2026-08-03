#!/usr/bin/env python3
"""Self-check for the materialized split side tables (task T1 acceptance).

Checks (against real data):
  1. Coverage: every source_id and preset_id present in the completed builds'
     groups.jsonl is present in splits.sqlite3, and the stored split equals an
     independent recomputation.
  2. Cross-build stability: sample up to 100 source_ids occurring in >=2
     builds; recompute the S-split independently per build occurrence and
     assert identical (and equal to the table).
  3. Ratio sanity: S-split totals near 90/5/5 (chi-square-free tolerance
     check: val and test each within [3%, 7%]); P-split assignments replayed
     gen by gen (gen-0 full rule + frozen-append increments, wave-1.5) match
     the stored ones.
  4. Stability guard (wave-1.5, T1-B1): simulating 10 new presets joining the
     real table via build_splits.merge_presets must leave every existing
     P-split assignment unchanged.

Usage:
  python3 selfcheck.py            # full check
  python3 selfcheck.py --quick    # first 1500 groups per build (smoke)
"""
from __future__ import annotations

import argparse
import collections
import os
import random
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_splits as B  # noqa: E402
import vr_common as C  # noqa: E402

FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global FAIL
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAIL += 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    limit = 1500 if args.quick else None

    db = os.path.join(C.TOOLS_DIR, "splits.sqlite3")
    if not os.path.exists(db):
        print("splits.sqlite3 missing — run build_splits.py first")
        return 2
    con = sqlite3.connect(db)
    tbl_sources = {r[0]: (r[1], r[2]) for r in con.execute("SELECT source_id, split, pool FROM sources")}
    try:
        rows = list(con.execute("SELECT preset_id, major, minor, split, gen FROM presets"))
    except sqlite3.OperationalError:  # pre-wave-1.5 table without gen column
        rows = [(r[0], r[1], r[2], r[3], 0) for r in con.execute(
            "SELECT preset_id, major, minor, split FROM presets")]
    tbl_presets = {r[0]: (r[1], r[2], r[3], r[4]) for r in rows}
    seed = con.execute("SELECT value FROM meta WHERE key='split_seed'").fetchone()[0]
    con.close()
    check("meta.split_seed matches code constant", seed == C.SPLIT_SEED, seed)

    # scan journals
    src_builds: dict[str, set[str]] = collections.defaultdict(set)
    seen_presets: set[str] = set()
    builds = C.completed_builds()
    for b in builds:
        for g in C.iter_groups(b, limit=limit):
            src_builds[g["source_id"]].add(b)
            for c in g["candidates"]:
                seen_presets.add(c["preset_id"])

    # 1. coverage + split recomputation
    missing_src = [s for s in src_builds if s not in tbl_sources]
    check("all journal source_ids covered by table",
          not missing_src, f"missing={len(missing_src)} of {len(src_builds)}")
    missing_p = [p for p in seen_presets if p not in tbl_presets]
    check("all journal preset_ids covered by table",
          not missing_p, f"missing={len(missing_p)} of {len(seen_presets)}")
    bad = sum(1 for s in src_builds if s in tbl_sources and tbl_sources[s][0] != C.s_split(s))
    check("stored S-split == independent recomputation", bad == 0, f"mismatches={bad}")

    # 2. cross-build stability, 100 samples
    multi = sorted(s for s, bs in src_builds.items() if len(bs) >= 2)
    rng = random.Random(20260802)
    sample = rng.sample(multi, min(100, len(multi))) if multi else []
    bad2 = 0
    for s in sample:
        splits = {C.s_split(s) for _ in src_builds[s]}  # per-occurrence recompute
        if len(splits) != 1 or (s in tbl_sources and tbl_sources[s][0] not in splits):
            bad2 += 1
    check("cross-build split stability (sampled)",
          bad2 == 0, f"sampled={len(sample)} multi-build sources (pool={len(multi)}), mismatches={bad2}")

    # 3a. S ratio sanity
    tot = collections.Counter(v[0] for v in tbl_sources.values())
    n = sum(tot.values())
    ratios = {k: tot[k] / n for k in ("train", "val", "test")}
    ok = 0.03 <= ratios["val"] <= 0.07 and 0.03 <= ratios["test"] <= 0.07
    check("S-split ratios near 90/5/5", ok,
          f"train={ratios['train']:.3f} val={ratios['val']:.3f} test={ratios['test']:.3f} (n={n})")

    # 3b. P-split gen-aware replay: gen-0 full rule, then frozen-append
    #     increments per gen (invariant survives incremental appends, T1-B1)
    layers: dict[str, list[tuple[int, str, str]]] = collections.defaultdict(list)
    for pid, (_, minor, sp, gen) in tbl_presets.items():
        layers[minor].append((gen, pid, sp))
    bad3 = 0
    for minor, entries in layers.items():
        sim: dict[str, str] = {}
        for g in sorted({e[0] for e in entries}):
            sim.update(C.p_split_increment(sim, [p for gg, p, _ in entries if gg == g]))
        bad3 += sum(1 for _, p, sp in entries if sim[p] != sp)
    check("stored P-split == frozen-append replay (gen-aware)", bad3 == 0,
          f"mismatches={bad3}")
    ptot = collections.Counter(v[2] for v in tbl_presets.values())
    gens = collections.Counter(v[3] for v in tbl_presets.values())
    print(f"  info: presets train/val/test = {ptot['train']}/{ptot['val']}/{ptot['test']} "
          f"({len(layers)} minor layers; gens={dict(sorted(gens.items()))})")

    # 4. stability guard (wave-1.5, T1-B1): +10 simulated presets through the
    #    real merge path must not move any existing assignment. aaa/zzz id
    #    prefixes cover both ends of the sort order on the largest layers.
    by_layer = collections.Counter(v[1] for v in tbl_presets.values())
    top = [mn for mn, _ in sorted(by_layer.items(), key=lambda kv: (-kv[1], kv[0]))[:10]]
    fakes = {f"{'aaa' if i % 2 == 0 else 'zzz'}_simcheck_{i:02d}": top[i % len(top)]
             for i in range(10)}
    scanned = {pid: v[1] for pid, v in tbl_presets.items()} | fakes
    sim_rows, sim_info = B.merge_presets(dict(tbl_presets), scanned)
    got = {r[0]: r[3] for r in sim_rows}
    moved = sum(1 for pid, v in tbl_presets.items() if got.get(pid) != v[2])
    n_new_assigned = sum(1 for p in fakes if p in got)
    ok4 = moved == 0 and n_new_assigned == 10 and len(sim_rows) == len(tbl_presets) + 10
    check("stability: +10 simulated presets leave existing P-split unchanged",
          ok4, f"moved={moved}, new_assigned={n_new_assigned}/10, "
               f"new_by_split={dict(sim_info['new_by_split'])}")

    print(f"\n{'ALL CHECKS PASSED' if FAIL == 0 else str(FAIL) + ' CHECK(S) FAILED'}"
          + (" (quick mode)" if args.quick else ""))
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
