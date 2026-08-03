"""S-split resolution for the construct generator.

Authority (wave-1.5, review B3 fix): the frozen T1 side-table
``tools/data_splits/splits.sqlite3`` is the ONLY valid S-split source
(DATA_ASSIGNMENT §1.1: experiments read the side-table, ad-hoc splits are
forbidden). ``make_splitter()`` loads it by default.

The legacy inline sha1 rule (``s_split_v0``) is **deprecated**: it is NOT the
frozen T1 rule (T1: ``int(sha1("verasplit-v1:"+sid).hexdigest()[:8],16)%100``)
and agrees with the frozen table on only ~81% of real source_ids (wave-1
review, T4-B3). It survives solely as a fallback for environments where the
side-table file does not exist; whenever the fallback is about to start while
the side-table IS present, a 1000-source consistency audit runs and any
mismatch refuses startup.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import sqlite3
from pathlib import Path

DEFAULT_SPLIT_DB = (
    Path(__file__).resolve().parents[1] / "data_splits" / "splits.sqlite3"
)
FALLBACK_AUDIT_N = 1000

# DEPRECATED (review T4-B3) — pre-T1 stand-in rule, ~81% consistent with the
# frozen T1 table. Never use while the side-table exists; kept only so an
# environment without the table can still smoke-test the generator.
SPLIT_RULE = (
    "s_split_v0(DEPRECATED, not the frozen T1 rule): "
    "int(sha1(utf8(source_id)).hexdigest,16)%100 "
    "-> [0,89]=train,[90,94]=val,[95,99]=test"
)


def split_bucket(source_id: str) -> int:
    """DEPRECATED inline bucket (see SPLIT_RULE). Fallback/audit use only."""
    return int(hashlib.sha1(source_id.encode("utf-8")).hexdigest(), 16) % 100


def split_of(source_id: str) -> str:
    """DEPRECATED inline rule (see SPLIT_RULE). Fallback/audit use only."""
    b = split_bucket(source_id)
    if b < 90:
        return "train"
    if b < 95:
        return "val"
    return "test"


def load_split_table(path: str | Path) -> dict[str, str]:
    """Load a split side-table: .sqlite3/.sqlite/.db, .csv, or jsonl."""
    path = Path(path)
    table: dict[str, str] = {}
    suffix = path.suffix.lower()
    if suffix in (".sqlite3", ".sqlite", ".db"):
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            for sid, split in con.execute("SELECT source_id, split FROM sources"):
                table[sid] = split
        finally:
            con.close()
    elif suffix == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                table[row["source_id"]] = row["split"]
    else:  # jsonl
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                table[row["source_id"]] = row["split"]
    if not table:
        raise RuntimeError(f"split table {path} loaded 0 rows")
    return table


def _audit_inline_vs_table(table: dict[str, str],
                           n: int = FALLBACK_AUDIT_N) -> list[str]:
    """Sample n sources from the table; return ids where inline rule differs."""
    ids = sorted(table)
    rnd = random.Random("verasplit-fallback-audit")
    picks = rnd.sample(ids, min(n, len(ids)))
    return [sid for sid in picks if split_of(sid) != table[sid]]


def _inline_fallback():
    """Deprecated inline rule, guarded: refuse if the T1 side-table exists
    and disagrees with the inline rule on any of 1000 sampled sources."""
    if DEFAULT_SPLIT_DB.exists():
        table = load_split_table(DEFAULT_SPLIT_DB)
        bad = _audit_inline_vs_table(table)
        if bad:
            raise RuntimeError(
                f"inline S-split fallback REFUSED: T1 side-table "
                f"{DEFAULT_SPLIT_DB} exists and disagrees with the inline rule "
                f"on {len(bad)}/{min(FALLBACK_AUDIT_N, len(table))} sampled "
                f"sources (e.g. {bad[:3]}). Use the side-table."
            )
    print(f"[splits] WARNING: T1 side-table {DEFAULT_SPLIT_DB} not found; "
          f"using DEPRECATED inline rule (smoke-test only)", flush=True)
    return split_of, SPLIT_RULE + " [fallback: side-table absent]"


def make_splitter(split_table: dict[str, str] | None = None,
                  split_table_path: str | Path | None = None,
                  *, use_inline: bool = False):
    """Return (fn(source_id)->split, rule_string).

    Default (no args): load the frozen T1 side-table at DEFAULT_SPLIT_DB.
    Sources absent from the table map to "unknown" (excluded downstream).
    The inline rule is used only via the guarded fallback (side-table file
    absent, or explicit ``use_inline=True`` which still runs the guard).
    """
    if use_inline:
        return _inline_fallback()
    if split_table is not None:
        return (lambda sid: split_table.get(sid, "unknown")), "external_split_table"
    if split_table_path is None and DEFAULT_SPLIT_DB.exists():
        split_table_path = DEFAULT_SPLIT_DB
    if split_table_path is not None:
        path = Path(split_table_path)
        table = load_split_table(path)
        rule = f"t1_side_table:{path.name}(rows={len(table)})"
        return (lambda sid: table.get(sid, "unknown")), rule
    return _inline_fallback()
