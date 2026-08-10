"""Split -> per-sample class labels, with an on-disk cache.

The geometry pass is the only part of the tool that touches NFS in bulk (400
``.maskhi.png`` + 400 ``.maskmeta.json`` members for ``V_where``, measured at
~15 s cold through ``/mnt/nfs-ro``).  It is a pure function of the published
masks -- the same split gives the same labels for every arm and every step -- so
it is cached to JSON and reused across all eight arms rather than re-read once
per report.

The cache stores the **raw geometry**, not the class labels: a threshold change
is then a re-classification, not a re-read.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .taxonomy import (
    DIMENSIONS, GEOMETRY_QUANTILE_KEYS, TaxonomyConfig, classify_geometry,
    global_labels, mask_geometry,
)

__all__ = ["EXTRA_DIMENSIONS", "scan_geometry", "load_geometry_cache",
           "labels_from_geometry", "geometry_quantiles", "extra_labels"]

#: strata that are read off the record / eval rows rather than computed from the
#: mask.  ``region`` is the databuild's own coarse position label -- an
#: independent, human-authored cross-check on the geometric ``position``
#: dimension; the other three are the strata the eval already carries.
EXTRA_DIMENSIONS: tuple[str, ...] = (
    "region", "winner_confidence", "upscaled", "build", "active_primitive_bucket",
)


def scan_geometry(
    sample_ids: Sequence[str],
    *,
    maskview_root: str | Path,
    cfg: TaxonomyConfig | None = None,
    verify: bool = False,
    with_meta: bool = True,
    progress: int = 100,
    log=print,
) -> dict[str, Any]:
    """Read every GT mask once and return ``{sample_id: geometry}`` + provenance.

    ``verify=False`` skips the per-member sha256: this is a read-only analysis of
    an already-published dataset over a soft mount, and the checksum costs a
    second pass over every byte.  Set it True when the publication itself is
    under suspicion.
    """
    import time

    from .pubio import MaskViews

    cfg = cfg or TaxonomyConfig()
    mv = MaskViews(maskview_root, verify=verify)
    geometry: dict[str, Any] = {}
    extra: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    t0 = time.time()
    for i, sid in enumerate(sample_ids):
        if not mv.has(sid, mv.HI):
            missing.append(sid)
            continue
        g = mask_geometry(mv.mask_hi(sid), cfg)
        geometry[sid] = g
        if with_meta and mv.has(sid, mv.META):
            m = mv.meta(sid)
            extra[sid] = {
                "region": m.get("region"),
                "winner_confidence": m.get("winner_confidence"),
                "upscaled": m.get("upscaled"),
                "build": m.get("build"),
                "where_a_frac_soft": (m.get("mask_stats") or {}).get("frac_soft"),
                "where_a_degenerate": (m.get("mask_stats") or {}).get("degenerate"),
            }
        if progress and i % progress == 0 and log:
            log(f"  geometry {i}/{len(sample_ids)}  {time.time() - t0:.1f}s")
    return {
        "geometry": geometry,
        "extra": extra,
        "missing": missing,
        "facts": {**mv.facts(), "n_requested": len(sample_ids),
                  "n_read": len(geometry), "n_missing": len(missing),
                  "verify_checksums": verify,
                  "seconds": round(time.time() - t0, 2)},
        "taxonomy": cfg.to_dict(),
    }


def load_geometry_cache(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def labels_from_geometry(geometry: Mapping[str, Mapping[str, Any]],
                         cfg: TaxonomyConfig | None = None,
                         ) -> dict[str, dict[str, str]]:
    cfg = cfg or TaxonomyConfig()
    return {sid: classify_geometry(dict(g), cfg) for sid, g in geometry.items()}


def extra_labels(rows: Iterable[Mapping[str, Any]],
                 extra: Mapping[str, Mapping[str, Any]],
                 ) -> dict[str, dict[str, str]]:
    """Merge the record-side strata into a ``{sample_id: {dim: class}}`` map.

    ``active_primitive_bucket`` is read off the eval row when it is there.  It is
    **not** there today: ``q3vl.whereb.metrics`` defines and tests
    ``active_primitive_count`` and ``config.EXTRA_STRATA_KEYS`` lists the bucket,
    but ``evaluate.evaluate_context`` never writes it into a per-sample row, so
    every published board's ``strata.active_primitive_bucket`` has exactly one
    cell: ``{"None": n}``.  Reported here as ``n/a-not-emitted`` rather than
    silently dropped, and filled in for real when ``--checkpoint`` re-runs the
    fields (that is the only place the predicted ``rho`` exists).
    """
    out: dict[str, dict[str, str]] = {}
    for r in rows:
        sid = str(r.get("sample_id"))
        e = extra.get(sid, {})
        bucket = r.get("active_primitive_bucket")
        out[sid] = {
            "region": str(e.get("region", r.get("region", "unknown"))),
            "winner_confidence": str(r.get("winner_confidence",
                                           e.get("winner_confidence"))),
            "upscaled": str(r.get("upscaled", e.get("upscaled"))),
            "build": str(r.get("build", e.get("build"))),
            "active_primitive_bucket": (str(bucket) if bucket is not None
                                        else "n/a-not-emitted"),
        }
    return out


def merge_labels(*maps: Mapping[str, Mapping[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for m in maps:
        for sid, labels in m.items():
            out.setdefault(sid, {}).update(labels)
    return out


def geometry_quantiles(geometry: Mapping[str, Mapping[str, Any]],
                       keys: Sequence[str] = GEOMETRY_QUANTILE_KEYS,
                       ) -> dict[str, dict[str, float]]:
    import numpy as np

    out: dict[str, dict[str, float]] = {}
    for k in keys:
        vals = [float(g[k]) for g in geometry.values() if g.get(k) is not None]
        if not vals:
            continue
        a = np.asarray(vals, dtype=float)
        p10, p50, p90 = (float(x) for x in np.percentile(a, [10, 50, 90]))
        out[k] = {"n": len(vals), "min": float(a.min()), "p10": p10, "p50": p50,
                  "p90": p90, "max": float(a.max())}
    return out


def dimension_activity(labels: Mapping[str, Mapping[str, str]],
                       dimensions: Sequence[str] = DIMENSIONS,
                       ) -> dict[str, dict[str, Any]]:
    """Which dimensions this split actually exercises.

    A dimension where every sample lands in one class is *inactive* on this
    split; reporting its table as if it were a comparison would invent a contrast
    that the data does not contain (V_where: 0 of 400 ``.cgt`` masks have a hole).
    """
    out: dict[str, dict[str, Any]] = {}
    for dim in dimensions:
        counts: dict[str, int] = {}
        for lab in labels.values():
            counts[str(lab.get(dim))] = counts.get(str(lab.get(dim)), 0) + 1
        out[dim] = {"n_classes": len(counts), "counts": counts,
                    "active": len(counts) > 1}
    return out


def global_label_map(sample_ids: Iterable[str]) -> dict[str, dict[str, str]]:
    return {sid: global_labels() for sid in sample_ids}
