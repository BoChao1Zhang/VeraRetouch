"""Clean-before-build: read-only PG cleaning predicate (Phase 3).

The build must never consume an un-cleaned asset. source_qa writes a per-asset
verdict into the ``assets`` table (``auto_verdict`` ∈ keep|drop|review|
needs_local_render, plus ``dup_of`` for dedup cluster membership); this module
lets the build planner consume **only** cleared assets, keyed by asset id
(``assets.asset_id`` == build ``SourceItem.source_id`` / ``RecipeAsset.recipe_id``,
per source_qa ingest).

Design constraints (v2 §7, §10 open-decision 3):
  * **Opt-in.** Disabled unless ``config.cleaning.enabled`` is true — default
    builds are unaffected. Activating it (and the apply closure that populates
    the verdicts) is the operator's go-live decision.
  * **Conservative.** An asset is cleared iff
    ``auto_verdict ∈ keep_verdicts`` (default keep / needs_local_render) and,
    when ``require_dup_head``, it is a dedup cluster head (``dup_of IS NULL``).
    Anything absent / drop / review / null is excluded.
  * **Fail-closed.** Any PG/connection error raises ``SystemExit`` (HALT) — the
    build never falls open to consuming unclean assets.
  * **Read-only.** Opens its own autocommit SELECT connection; never writes.

By default the predicate applies to ``image`` sources only (the preset pool is
QA'd separately and currently has very few cleared rows); set
``cleaning.apply_to`` to include ``preset`` to gate recipes too.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Set

# Process-level cache: asset_type -> cleared id set (one query per type per run).
_CLEARED_CACHE: Dict[str, Set[str]] = {}


def cleaning_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return (config.get("cleaning", {}) or {})


def is_enabled(config: Dict[str, Any]) -> bool:
    return bool(cleaning_config(config).get("enabled", False))


def apply_to(config: Dict[str, Any]) -> List[str]:
    cc = cleaning_config(config)
    val = cc.get("apply_to", ["image"])
    return list(val) if isinstance(val, (list, tuple)) else [str(val)]


def _dsn(config: Dict[str, Any]) -> str:
    dsn = cleaning_config(config).get("pg_dsn")
    if dsn:
        return str(dsn)
    # Fall back to the source_qa default (env SOURCE_QA_PG_DSN or the local DB).
    from dataset_build.source_qa import config as sqc
    return sqc.PG_DSN


def load_cleared_ids(config: Dict[str, Any], asset_type: str) -> Set[str]:
    """Return the set of cleared ``asset_id`` for ``asset_type`` (cached).

    Fail-closed: a PG/connection/query error raises ``SystemExit`` so the build
    HALTS rather than silently consuming un-cleaned assets.
    """
    if asset_type in _CLEARED_CACHE:
        return _CLEARED_CACHE[asset_type]

    cc = cleaning_config(config)
    keep = list(cc.get("keep_verdicts", ["keep", "needs_local_render"]))
    require_head = bool(cc.get("require_dup_head", True))
    if not keep:
        raise SystemExit("[clean-before-build] FAIL-CLOSED: cleaning.keep_verdicts is empty")

    placeholders = ", ".join(["%s"] * len(keep))
    sql = (f"SELECT asset_id FROM assets "
           f"WHERE asset_type = %s AND auto_verdict IN ({placeholders})")
    params: List[Any] = [asset_type, *keep]
    if require_head:
        sql += " AND dup_of IS NULL"

    try:
        import psycopg
        with psycopg.connect(_dsn(config), autocommit=True, connect_timeout=10) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                ids = {row[0] for row in cur.fetchall()}
    except SystemExit:
        raise
    except Exception as e:  # connection refused / table missing / auth / etc.
        raise SystemExit(
            f"[clean-before-build] FAIL-CLOSED: PG cleaning predicate failed for "
            f"asset_type={asset_type!r}: {type(e).__name__}: {e}"
        ) from e

    _CLEARED_CACHE[asset_type] = ids
    return ids


def filter_sources(config: Dict[str, Any], stream_value: str, sources: List[Any]) -> List[Any]:
    """Keep only cleared image sources (no-op unless enabled + 'image' in apply_to)."""
    if not is_enabled(config) or "image" not in apply_to(config):
        return sources
    cleared = load_cleared_ids(config, "image")
    kept = [s for s in sources if getattr(s, "source_id", None) in cleared]
    print(f"[clean-before-build] {stream_value} sources {len(sources)} -> {len(kept)} "
          f"(cleared image assets={len(cleared)})", file=sys.stderr)
    return kept


def filter_recipes(config: Dict[str, Any], stream_value: str, recipes: List[Any]) -> List[Any]:
    """Keep only cleared preset recipes (no-op unless enabled + 'preset' in apply_to)."""
    if not recipes or not is_enabled(config) or "preset" not in apply_to(config):
        return recipes
    cleared = load_cleared_ids(config, "preset")
    kept = [r for r in recipes if getattr(r, "recipe_id", None) in cleared]
    print(f"[clean-before-build] {stream_value} recipes {len(recipes)} -> {len(kept)} "
          f"(cleared preset assets={len(cleared)})", file=sys.stderr)
    return kept
