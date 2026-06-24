"""Postgres provenance for the construct pipeline — every intermediate result is traceable.

The construct agent renders many candidates per source, QA-scores each, then tiers a few into
SFT/DPO. Previously only the final jsonl survived; this persists the WHOLE chain to Postgres so any
record traces back to its source, preset, render, and full QA verdict:

  construct_groups      one row / source processed in a run
  construct_candidates  one row / rendered+QA'd candidate (preset, after_path, full QA dict, role)
  construct_sft         one row / SFT record (links to its candidate via cand_id)
  construct_dpo         one row / DPO pair (chosen/rejected after_path link to candidates)

Join key across stages = after_path (the rendered I_tar jpg, unique per candidate). A run is the
existing `runs` row (db.start_run kind='construct_<route>'); run_id ties everything together.
"""
from __future__ import annotations

import hashlib
import os
import uuid

from psycopg.types.json import Jsonb

from dataset_build.source_qa import db

# every rendered variant is one of these engines (derived from candidate.kind)
_ENGINE = {"param": "lr_farm", "lut": "lut_trilinear",
           "local_from_preset": "lr_farm", "lut_in_sam3": "lut_composite"}


def _sha256(path: str) -> tuple:
    """(sha256_hex, size_bytes) of a file, or (None, None) if unreadable."""
    try:
        h = hashlib.sha256()
        sz = 0
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk); sz += len(chunk)
        return h.hexdigest(), sz
    except OSError:
        return None, None

SCHEMA = """
CREATE TABLE IF NOT EXISTS construct_groups (
    group_id        TEXT PRIMARY KEY,
    run_id          TEXT,
    route           TEXT,                 -- global | geom | sam3
    source_path     TEXT,
    source_asset_id TEXT,
    is_portrait     BOOLEAN,
    n_candidates    INTEGER,
    created_at      TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE IF NOT EXISTS construct_candidates (
    cand_id      TEXT PRIMARY KEY,
    group_id     TEXT,
    run_id       TEXT,
    route        TEXT,
    source_path  TEXT,
    preset_id    TEXT,                    -- preset_id (global) or mask_unit_id (local)
    kind         TEXT,
    fmt          TEXT,
    preset_path  TEXT,
    content_hash TEXT,
    after_path   TEXT,                    -- rendered I_tar (cross-stage join key)
    reliable     BOOLEAN,
    veto         BOOLEAN,
    merit_score  REAL,
    merit_hits   JSONB,
    qa           JSONB,                   -- full QA verdict (vlm_veto, det, why, q, controls...)
    local        JSONB,                   -- mask geom/concept/cgt/base_preset, NULL for global
    role         TEXT,                    -- sft | dpo_chosen | dpo_rejected | NULL
    rank         INTEGER,
    created_at   TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE IF NOT EXISTS construct_sft (
    sft_id      TEXT PRIMARY KEY,
    run_id      TEXT,
    group_id    TEXT,
    cand_id     TEXT,                     -- FK -> construct_candidates
    source_path TEXT,
    i_tar       TEXT,
    recipe      JSONB,
    local       JSONB,
    instruction TEXT,
    reasoning   TEXT,
    qa          JSONB,
    created_at  TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE IF NOT EXISTS construct_dpo (
    dpo_id        TEXT PRIMARY KEY,
    run_id        TEXT,
    group_id      TEXT,
    source_path   TEXT,
    chosen        JSONB,
    rejected      JSONB,
    margin        REAL,
    chosen_after  TEXT,                   -- after_path of chosen candidate
    rejected_after TEXT,                  -- after_path of rejected candidate
    created_at    TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE IF NOT EXISTS construct_renders (
    render_id       TEXT PRIMARY KEY,
    run_id          TEXT,
    group_id        TEXT,
    route           TEXT,
    source_path     TEXT,
    source_asset_id TEXT,
    preset_id       TEXT,
    kind            TEXT,
    fmt             TEXT,
    engine          TEXT,                -- lr_farm | lut_trilinear | lut_composite
    after_path      TEXT,                -- the saved render jpg on disk
    sha256          TEXT,                -- content hash of the render bytes (dedup / integrity)
    size_bytes      BIGINT,
    created_at      TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_crender_run    ON construct_renders (run_id);
CREATE INDEX IF NOT EXISTS ix_crender_sha    ON construct_renders (sha256);
CREATE INDEX IF NOT EXISTS ix_crender_after  ON construct_renders (after_path);
CREATE INDEX IF NOT EXISTS ix_crender_src    ON construct_renders (source_path);
CREATE INDEX IF NOT EXISTS ix_cgroups_run ON construct_groups (run_id);
CREATE INDEX IF NOT EXISTS ix_cgroups_src ON construct_groups (source_path);
CREATE INDEX IF NOT EXISTS ix_ccand_run  ON construct_candidates (run_id);
CREATE INDEX IF NOT EXISTS ix_ccand_grp  ON construct_candidates (group_id);
CREATE INDEX IF NOT EXISTS ix_ccand_src  ON construct_candidates (source_path);
CREATE INDEX IF NOT EXISTS ix_ccand_after ON construct_candidates (after_path);
CREATE INDEX IF NOT EXISTS ix_csft_run   ON construct_sft (run_id);
CREATE INDEX IF NOT EXISTS ix_cdpo_run   ON construct_dpo (run_id);
"""


def init(conn) -> None:
    for stmt in (s.strip() for s in SCHEMA.split(";")):
        if stmt:
            conn.execute(stmt)
    conn.commit()


def persist_run(run_id: str, groups: list, sft: list, dpo: list, route: str) -> dict:
    """Write a full construct run to Postgres. after_path links candidate -> sft/dpo."""
    conn = db.connect()
    init(conn)
    sft_by_tar = {s["I_tar"]: s for s in sft}
    chosen_tar = {d["chosen"]["I_tar"] for d in dpo}
    reject_tar = {d["rejected"]["I_tar"] for d in dpo}
    src2gid: dict = {}
    grows, crows, srows, drows, rrows = [], [], [], [], []

    for g in groups:
        gid = uuid.uuid4().hex
        src2gid[g["source"]] = gid
        grows.append((gid, run_id, route, g["source"], g.get("source_asset_id"),
                      g.get("is_portrait"), len(g["candidates"])))
        for c in g["candidates"]:
            cid = uuid.uuid4().hex
            ap = c.get("after_path")
            qa = c.get("qa") or {}
            if ap and os.path.exists(ap):   # register every saved render with content hash + size
                sha, sz = _sha256(ap)
                rrows.append((uuid.uuid4().hex, run_id, gid, route, g["source"],
                              g.get("source_asset_id"), c.get("preset_id"), c.get("kind"),
                              c.get("fmt"), _ENGINE.get(c.get("kind"), c.get("kind")), ap, sha, sz))
            if ap in sft_by_tar:
                role, rank = "sft", (sft_by_tar[ap].get("qa") or {}).get("rank")
            elif ap in chosen_tar:
                role, rank = "dpo_chosen", None
            elif ap in reject_tar:
                role, rank = "dpo_rejected", None
            else:
                role, rank = None, None
            crows.append((cid, gid, run_id, route, g["source"], c.get("preset_id"), c.get("kind"),
                          c.get("fmt"), c.get("preset_path"), c.get("content_hash"), ap,
                          qa.get("reliable"), qa.get("veto"), qa.get("merit_score"),
                          Jsonb(qa.get("merit_hits")), Jsonb(qa), Jsonb(c.get("local")), role, rank))
            if ap in sft_by_tar:
                s = sft_by_tar[ap]
                srows.append((uuid.uuid4().hex, run_id, gid, cid, g["source"], s["I_tar"],
                              Jsonb(s["recipe"]), Jsonb(s.get("local")), s["instruction"],
                              s["reasoning"], Jsonb(s.get("qa"))))
    for d in dpo:
        drows.append((uuid.uuid4().hex, run_id, src2gid.get(d["I_in"]), d["I_in"],
                      Jsonb(d["chosen"]), Jsonb(d["rejected"]), d.get("margin"),
                      d["chosen"]["I_tar"], d["rejected"]["I_tar"]))

    def _w():
        if grows:
            conn.executemany("INSERT INTO construct_groups VALUES (?,?,?,?,?,?,?,now()) "
                             "ON CONFLICT (group_id) DO NOTHING", grows)
        if crows:
            conn.executemany("INSERT INTO construct_candidates VALUES "
                             "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,now()) "
                             "ON CONFLICT (cand_id) DO NOTHING", crows)
        if srows:
            conn.executemany("INSERT INTO construct_sft VALUES (?,?,?,?,?,?,?,?,?,?,?,now()) "
                             "ON CONFLICT (sft_id) DO NOTHING", srows)
        if drows:
            conn.executemany("INSERT INTO construct_dpo VALUES (?,?,?,?,?,?,?,?,?,now()) "
                             "ON CONFLICT (dpo_id) DO NOTHING", drows)
        if rrows:
            conn.executemany("INSERT INTO construct_renders VALUES "
                             "(?,?,?,?,?,?,?,?,?,?,?,?,?,now()) "
                             "ON CONFLICT (render_id) DO NOTHING", rrows)

    ok = db.write_retry(conn, _w)
    conn.close()
    return {"ok": ok, "groups": len(grows), "candidates": len(crows),
            "sft": len(srows), "dpo": len(drows), "renders": len(rrows)}
