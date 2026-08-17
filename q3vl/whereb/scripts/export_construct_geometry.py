#!/usr/bin/env python
"""Export the construction-side geometry parameters into a local sqlite sidecar.

Why a sidecar and not a live query: the geometry the v4a reasoning template read
lives in the databuild projection (``canonical_candidates.payload->'geometry'``,
keyed by ``candidate_id``) and **not** in ``.vrmeta.json`` -- the published
per-sample member carries ``slot_id`` and ``region`` and nothing else (checked
2026-08-12 against ``prod-l3/l4/l6``).  Postgres is reachable only from the
databuild environment, and a training loop cannot open a connection per sample,
so AMD-8's GT code needs the parameters materialised once, on local disk.

Run it with the databuild interpreter (``psycopg`` lives there, not in
``q3vl_sft``)::

    /home/bc/envs/databuild/bin/python -m q3vl.whereb.scripts.export_construct_geometry \\
        --out /home/bc/data/runs/where_b/construct_geometry.sqlite3

The table is keyed by ``candidate_id``, which is what the q3vl record carries
(``record["candidate_id"]``), so the consumer needs no build-id join.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

DEFAULT_DSN = "postgresql://research:research@127.0.0.1:5432/research"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS candidate_geometry (
    candidate_id TEXT PRIMARY KEY,
    build_id TEXT NOT NULL,
    slot_id TEXT,
    slot_mode TEXT,
    region TEXT,
    raw_alpha_mean REAL,
    effective_alpha_mean REAL,
    amount REAL,
    geometry TEXT
);
"""

_QUERY = """
select candidate_id,
       build_id,
       slot_id,
       payload->>'slot_mode',
       region,
       payload->>'raw_alpha_mean',
       payload->>'effective_alpha_mean',
       payload->>'amount',
       payload->'geometry'
  from canonical_candidates
 where build_id like %s
"""


def _float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default=DEFAULT_DSN)
    ap.add_argument("--build-like", default="prod-l%",
                    help="SQL LIKE over build_id; local builds by default")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import psycopg  # databuild env only

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(out)
    db.executescript(_SCHEMA)

    n = n_geom = 0
    t0 = time.time()
    with psycopg.connect(args.dsn, connect_timeout=10) as conn:
        with conn.cursor(name="geomexport") as cur:   # server-side, 800k rows
            cur.itersize = 5000
            cur.execute(_QUERY, (args.build_like,))
            batch = []
            for row in cur:
                geom = row[8]
                if geom is not None:
                    n_geom += 1
                batch.append((
                    row[0], row[1], row[2], row[3], row[4],
                    _float(row[5]), _float(row[6]), _float(row[7]),
                    None if geom is None else json.dumps(geom, sort_keys=True),
                ))
                if len(batch) >= 5000:
                    db.executemany("INSERT OR REPLACE INTO candidate_geometry "
                                   "VALUES (?,?,?,?,?,?,?,?,?)", batch)
                    n += len(batch)
                    batch.clear()
                    print(f"  {n} rows ({time.time()-t0:.0f}s)", flush=True)
            if batch:
                db.executemany("INSERT OR REPLACE INTO candidate_geometry "
                               "VALUES (?,?,?,?,?,?,?,?,?)", batch)
                n += len(batch)
    db.commit()
    by_mode = dict(db.execute(
        "select slot_mode, count(*) from candidate_geometry group by 1").fetchall())
    db.close()
    print(json.dumps({"rows": n, "rows_with_geometry": n_geom,
                      "by_slot_mode": by_mode, "out": str(out),
                      "seconds": round(time.time() - t0, 1)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
