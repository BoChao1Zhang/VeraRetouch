#!/usr/bin/env python3
"""Materialize the S/P split side tables (DATA_ASSIGNMENT §1, action item C).

Scans groups.jsonl of every completed build (journal archive), collects all
source_id and candidates[].preset_id (with taxonomy minor), and writes:

  tools/data_splits/splits.sqlite3
      sources(source_id TEXT PK, split TEXT, pool TEXT)
      presets(preset_id TEXT PK, major TEXT, minor TEXT, split TEXT, gen INT)
      meta(key TEXT PK, value TEXT)
  tools/data_splits/splits_sources.csv
  tools/data_splits/splits_presets.csv

and stats into experiments/tooling-wave1/data_splits/:
  stats_sources_pool_split.csv, stats_presets_minor_split.csv,
  build_splits_summary.json

Rerun semantics (wave-1.5, fixes REVIEW-impl-wave1 T1-B1):
  * Default = INCREMENTAL APPEND. If splits.sqlite3 exists, every stored
    source_id / preset_id KEEPS its stored split (frozen); only new ids get
    assigned. New presets take only the incremental per-layer quota
    (vr_common.p_split_increment) and are tagged with the next `gen` number.
    Entries present in the table but absent from the scan are kept (never
    deleted). A seed mismatch between meta and code is a hard failure.
  * A FULL REBUILD (recompute everything from scratch, gens reset to 0)
    requires an explicit --force and prints an old->new diff summary
    (per-split migration counts) for sources and presets before writing.

S-split: sha1("verasplit-v1:"+source_id) hex[:8] as int mod 100;
0-89 train / 90-94 val / 95-99 test (pure function of source_id => identical
across builds and reruns by construction).
P-split gen 0: stratified by minor; within a layer preset_ids sorted
ascending, tail 5% test, previous 5% val (round-half-up), rest train.
P-split gen >= 1 (incremental): existing assignments frozen; new ids sorted
ascending fill only (target quota - already held) val/test slots.

Usage:
  python3 build_splits.py                 # incremental append (default)
  python3 build_splits.py --force         # full rebuild + old->new diff summary
  python3 build_splits.py --limit 2000    # small-sample smoke run (writes to
                                          # *.sample.* files, sqlite skipped)
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vr_common as C  # noqa: E402

DB_NAME = "splits.sqlite3"

SCHEMA = (
    "CREATE TABLE sources(source_id TEXT PRIMARY KEY, split TEXT NOT NULL,"
    " pool TEXT NOT NULL);"
    "CREATE TABLE presets(preset_id TEXT PRIMARY KEY, major TEXT NOT NULL,"
    " minor TEXT NOT NULL, split TEXT NOT NULL,"
    " gen INTEGER NOT NULL DEFAULT 0);"
    "CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);"
)


def scan(limit: int | None):
    sources: dict[str, str] = {}           # source_id -> pool
    pool_conflicts = collections.Counter()
    preset_minor: dict[str, str] = {}      # preset_id -> minor
    minor_conflicts = collections.Counter()
    builds = C.completed_builds()
    per_build_groups = {}
    for b in builds:
        n = 0
        for g in C.iter_groups(b, limit=limit):
            n += 1
            sid = g["source_id"]
            pool = C.pool_of(g.get("source_path", ""))
            prev = sources.get(sid)
            if prev is None:
                sources[sid] = pool
            elif prev != pool:
                pool_conflicts[(prev, pool)] += 1
            for c in g["candidates"]:
                pid = c["preset_id"]
                mn = c["minor"]
                pm = preset_minor.get(pid)
                if pm is None:
                    preset_minor[pid] = mn
                elif pm != mn:
                    minor_conflicts[(pid, pm, mn)] += 1
        per_build_groups[b] = n
    return builds, per_build_groups, sources, preset_minor, pool_conflicts, minor_conflicts


def load_existing(db_path: str) -> tuple[dict[str, tuple[str, str]],
                                         dict[str, tuple[str, str, str, int]],
                                         str | None]:
    """Load frozen assignments from an existing side table.

    Migrates pre-wave-1.5 tables in place (adds presets.gen, default 0).
    Returns (sources: sid -> (split, pool),
             presets: pid -> (major, minor, split, gen),
             stored split_seed or None).
    """
    con = sqlite3.connect(db_path)
    cols = [r[1] for r in con.execute("PRAGMA table_info(presets)")]
    if "gen" not in cols:
        con.execute("ALTER TABLE presets ADD COLUMN gen INTEGER NOT NULL DEFAULT 0")
        con.commit()
    srcs = {r[0]: (r[1], r[2]) for r in con.execute(
        "SELECT source_id, split, pool FROM sources")}
    pres = {r[0]: (r[1], r[2], r[3], r[4]) for r in con.execute(
        "SELECT preset_id, major, minor, split, gen FROM presets")}
    row = con.execute("SELECT value FROM meta WHERE key='split_seed'").fetchone()
    con.close()
    return srcs, pres, (row[0] if row else None)


def merge_presets(old_presets: dict[str, tuple[str, str, str, int]],
                  scanned_minor: dict[str, str],
                  ) -> tuple[list[tuple[str, str, str, str, int]], dict]:
    """Frozen-append merge of preset assignments (T1-B1 fix).

    old_presets: pid -> (major, minor, split, gen). Returned untouched (frozen;
    on a minor conflict the stored minor wins and the conflict is counted).
    scanned_minor: pid -> minor from the current journal scan.
    New pids get gen = max(existing gens) + 1 and, per minor layer, fill only
    the incremental quota via vr_common.p_split_increment. With
    old_presets == {} this reduces exactly to the frozen gen-0 rule
    (p_split_layer semantics).
    Returns (rows sorted by preset_id, info dict).
    """
    new_pids = {pid: mn for pid, mn in scanned_minor.items()
                if pid not in old_presets}
    frozen_minor_conflicts = sum(
        1 for pid, (_, mn, _, _) in old_presets.items()
        if pid in scanned_minor and scanned_minor[pid] != mn)
    gen_new = max((v[3] for v in old_presets.values()), default=-1) + 1
    layers_old: dict[str, dict[str, str]] = collections.defaultdict(dict)
    for pid, (_, mn, sp, _) in old_presets.items():
        layers_old[mn][pid] = sp
    layers_new: dict[str, list[str]] = collections.defaultdict(list)
    for pid, mn in new_pids.items():
        layers_new[mn].append(pid)
    assign_new: dict[str, str] = {}
    for mn, pids in layers_new.items():
        assign_new.update(C.p_split_increment(layers_old.get(mn, {}), pids))
    rows = [(pid, mj, mn, sp, g)
            for pid, (mj, mn, sp, g) in old_presets.items()]
    rows += [(pid, mn.rsplit("_", 1)[0], mn, assign_new[pid], gen_new)
             for pid, mn in new_pids.items()]
    rows.sort()
    info = {
        "n_frozen": len(old_presets),
        "n_new": len(new_pids),
        "new_by_split": collections.Counter(assign_new.values()),
        "frozen_minor_conflicts": frozen_minor_conflicts,
        "gen_new": gen_new,
    }
    return rows, info


def print_diff(kind: str, old: dict[str, str], new: dict[str, str]) -> dict[str, int]:
    """Print per-split migration counts between two assignment maps."""
    mig: collections.Counter[str] = collections.Counter()
    for k, o in old.items():
        n = new.get(k)
        if n is None:
            mig[f"{o}->REMOVED"] += 1
        elif n != o:
            mig[f"{o}->{n}"] += 1
    for k, n in new.items():
        if k not in old:
            mig[f"ADDED->{n}"] += 1
    unchanged = sum(1 for k, o in old.items() if new.get(k) == o)
    detail = (", ".join(f"{k}={v}" for k, v in sorted(mig.items()))
              if mig else "no migrations")
    print(f"  {kind}: unchanged={unchanged}; {detail}")
    return dict(mig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="max groups per build (smoke run; skips sqlite)")
    ap.add_argument("--force", action="store_true",
                    help="full table rebuild (discards frozen assignments, resets"
                         " gen to 0); prints old->new diff summary. Default is"
                         " incremental append.")
    args = ap.parse_args()
    smoke = args.limit is not None

    builds, per_build_groups, sources, preset_minor, pool_conf, minor_conf = scan(args.limit)

    tools_dir, rep_dir = C.TOOLS_DIR, C.REPORT_DIR
    db_path = os.path.join(tools_dir, DB_NAME)

    old_sources: dict[str, tuple[str, str]] = {}
    old_presets: dict[str, tuple[str, str, str, int]] = {}
    mode = "fresh"
    if not smoke and os.path.exists(db_path):
        old_sources, old_presets, old_seed = load_existing(db_path)
        if old_seed != C.SPLIT_SEED:
            print(f"FATAL: existing table split_seed={old_seed!r} != code"
                  f" {C.SPLIT_SEED!r}; refusing to touch it", file=sys.stderr)
            return 2
        mode = "rebuild" if args.force else "incremental"

    # ---- S-split (existing rows frozen in incremental mode) ----
    if mode == "incremental":
        new_src_rows = sorted(
            (sid, C.s_split(sid), pool)
            for sid, pool in sources.items() if sid not in old_sources)
        src_rows = sorted(
            [(sid, sp, pool) for sid, (sp, pool) in old_sources.items()]
            + new_src_rows)
    else:
        new_src_rows = []
        src_rows = [(sid, C.s_split(sid), pool)
                    for sid, pool in sorted(sources.items())]

    # ---- P-split (frozen-append merge; empty old == full gen-0 derivation) ----
    preset_rows, pinfo = merge_presets(
        old_presets if mode == "incremental" else {}, preset_minor)
    new_preset_rows = ([r for r in preset_rows if r[0] not in old_presets]
                       if mode == "incremental" else [])

    # ---- rerun guard output ----
    if mode == "rebuild":
        print("== --force rebuild: old -> new diff summary ==")
        print_diff("sources", {k: v[0] for k, v in old_sources.items()},
                   {r[0]: r[1] for r in src_rows})
        print_diff("presets", {k: v[2] for k, v in old_presets.items()},
                   {r[0]: r[3] for r in preset_rows})
    elif mode == "incremental":
        table_only_src = sum(1 for s in old_sources if s not in sources)
        table_only_pre = sum(1 for p in old_presets if p not in preset_minor)
        print(f"== incremental append: {len(old_sources)} sources /"
              f" {len(old_presets)} presets frozen;"
              f" +{len(new_src_rows)} sources"
              f" {dict(collections.Counter(r[1] for r in new_src_rows))},"
              f" +{len(new_preset_rows)} presets {dict(pinfo['new_by_split'])}"
              f" (gen {pinfo['gen_new']});"
              f" table-only: {table_only_src} sources / {table_only_pre} presets ==")

    # ---- stats ----
    pool_split = collections.Counter((pool, sp) for _, sp, pool in src_rows)
    minor_split = collections.Counter((mn, sp) for _, _, mn, sp, _ in preset_rows)
    s_totals = collections.Counter(sp for _, sp, _ in src_rows)
    p_totals = collections.Counter(sp for _, _, _, sp, _ in preset_rows)
    n_layers = len({mn for _, _, mn, _, _ in preset_rows})

    suffix = ".sample" if smoke else ""
    os.makedirs(rep_dir, exist_ok=True)

    # ---- sqlite (full runs only) ----
    if not smoke:
        meta_updates = {
            "split_seed": C.SPLIT_SEED,
            "s_split_rule": "sha1(seed:source_id)[:8] as int %100; 0-89 train/90-94 val/95-99 test",
            "p_split_rule": ("gen0: stratify by minor; sort preset_id asc; tail"
                             " floor(n*0.05+0.5) test, then val, rest train."
                             " gen>=1: frozen-append, new ids fill only the"
                             " incremental quota (vr_common.p_split_increment)"),
            "builds": json.dumps(builds),
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "generator": "tools/data_splits/build_splits.py",
            "mode_last_run": mode,
        }
        if mode == "incremental":
            con = sqlite3.connect(db_path)
            con.executemany("INSERT INTO sources VALUES (?,?,?)", new_src_rows)
            con.executemany("INSERT INTO presets VALUES (?,?,?,?,?)", new_preset_rows)
            con.executemany("INSERT OR REPLACE INTO meta VALUES (?,?)",
                            sorted(meta_updates.items()))
            con.commit()
            con.close()
        else:
            if os.path.exists(db_path):
                os.remove(db_path)  # only reachable with explicit --force
            con = sqlite3.connect(db_path)
            con.executescript(SCHEMA)
            con.executemany("INSERT INTO sources VALUES (?,?,?)", src_rows)
            con.executemany("INSERT INTO presets VALUES (?,?,?,?,?)", preset_rows)
            con.executemany("INSERT INTO meta VALUES (?,?)",
                            sorted(meta_updates.items()))
            con.commit()
            con.close()

    # ---- CSVs ----
    with open(os.path.join(tools_dir, f"splits_sources{suffix}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_id", "split", "pool"])
        w.writerows(src_rows)
    with open(os.path.join(tools_dir, f"splits_presets{suffix}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["preset_id", "major", "minor", "split", "gen"])
        w.writerows(preset_rows)

    # ---- stats files ----
    pools = sorted({p for p, _ in pool_split})
    with open(os.path.join(rep_dir, f"stats_sources_pool_split{suffix}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pool", "train", "val", "test", "total"])
        for p in pools:
            tr, va, te = (pool_split.get((p, s), 0) for s in ("train", "val", "test"))
            w.writerow([p, tr, va, te, tr + va + te])
        w.writerow(["TOTAL", s_totals["train"], s_totals["val"], s_totals["test"], len(src_rows)])
    minors = sorted({m for m, _ in minor_split})
    with open(os.path.join(rep_dir, f"stats_presets_minor_split{suffix}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["minor", "train", "val", "test", "total"])
        for m in minors:
            tr, va, te = (minor_split.get((m, s), 0) for s in ("train", "val", "test"))
            w.writerow([m, tr, va, te, tr + va + te])
        w.writerow(["TOTAL", p_totals["train"], p_totals["val"], p_totals["test"], len(preset_rows)])

    summary = {
        "mode": mode,
        "builds": per_build_groups,
        "n_sources": len(src_rows),
        "n_presets": len(preset_rows),
        "s_split_totals": dict(s_totals),
        "p_split_totals": dict(p_totals),
        "n_minor_layers": n_layers,
        "source_pool_conflicts": sum(pool_conf.values()),
        "preset_minor_conflicts": sum(minor_conf.values()),
        "preset_minor_conflict_examples": [list(k) for k in list(minor_conf)[:10]],
        "smoke_limit": args.limit,
        "incremental": ({
            "frozen_sources": len(old_sources),
            "frozen_presets": len(old_presets),
            "new_sources": len(new_src_rows),
            "new_presets": pinfo["n_new"],
            "new_presets_by_split": dict(pinfo["new_by_split"]),
            "gen_new": pinfo["gen_new"],
            "frozen_minor_conflicts": pinfo["frozen_minor_conflicts"],
        } if mode == "incremental" else None),
    }
    with open(os.path.join(rep_dir, f"build_splits_summary{suffix}.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
