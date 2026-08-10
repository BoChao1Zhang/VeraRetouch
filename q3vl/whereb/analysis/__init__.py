"""WEVAL-1: per-class evaluation and long-tail bottleneck analysis for Where.

The eight Where-B arms all publish the same two artefacts -- ``metrics.json`` and
``per_sample.jsonl`` -- and both answer "how good is this arm overall".  Neither
answers the question a training decision actually needs: **which kind of region
is it failing on, and which stage of the pipeline is responsible**.  This package
is that second reader.  It is arm-agnostic: everything it needs is the eval
directory of any checkpoint plus the split's published GT masks.

Three stages, each usable on its own:

1. :mod:`~q3vl.whereb.analysis.taxonomy` -- a geometry-only classification of the
   GT mask (area / connectivity / topology / boundary complexity / position /
   edge softness).  Nothing in it looks at a prediction, so a class label can
   never be a function of the thing being scored.
2. :mod:`~q3vl.whereb.analysis.tables` -- per-class metric tables.  Every table
   is produced by :func:`q3vl.whereb.metrics.summarise`, the same aggregator the
   main board uses, so a per-class number and the headline number cannot drift.
   Single-dimension strata only: the full cross of six dimensions would be 216
   cells over 400 local samples.
3. :mod:`~q3vl.whereb.analysis.attribution` -- per-sample failure mechanisms for
   the long tail, plus the share of the tail each mechanism explains.

Red lines this package inherits (CLAUDE.md 2026-08-05) and where they are
enforced:

* **no AUC, in any form** -- the metric columns come from ``summarise`` and
  nothing here computes a ranking statistic; ``tests/test_analysis_tables.py``
  asserts no output key matches ``auc``.
* **top-k matching the GT area** is the only thresholding rule -- inherited by
  construction, since the hard-IoU / boundary-F1 columns are the ones the eval
  already computed under that rule.
* **the centre-prior column travels with every table** -- ``summarise`` emits it,
  and :mod:`report` refuses to render a table without it.
* **no per-image min-max colouring** -- :mod:`panels` colours masks on a fixed
  ``(0, 1)`` scale and ``s`` on the fixed ``(-S_SCALE, +S_SCALE)`` scale that
  ``s = 3 tanh(q/3)`` guarantees, so no figure has a per-image denominator at
  all.  Overlays go through ``viz.grid_to_img``'s exact inverse map.
* **colouring is not arithmetic** -- every number in ``per_class_metrics.json``
  and ``tail_samples.jsonl`` comes from the raw fields; :mod:`panels` returns no
  criterion.

Nothing is imported eagerly (same reason as :mod:`q3vl.whereb`: campaign bug R6,
sqlite3 must be importable after this package, so the package chain must not drag
torch in before an entry point's own guard runs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

_LAZY = {
    "TaxonomyConfig": "taxonomy",
    "DIMENSIONS": "taxonomy",
    "mask_geometry": "taxonomy",
    "classify_geometry": "taxonomy",
    "AnalysisThresholds": "attribution",
    "MECHANISMS": "attribution",
    "attribute_sample": "attribution",
    "mechanism_summary": "attribution",
    "per_class_tables": "tables",
    "load_per_sample": "tables",
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(f".{module}", __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:  # pragma: no cover
    from .attribution import (  # noqa: F401
        MECHANISMS, AnalysisThresholds, attribute_sample, mechanism_summary,
    )
    from .tables import load_per_sample, per_class_tables  # noqa: F401
    from .taxonomy import (  # noqa: F401
        DIMENSIONS, TaxonomyConfig, classify_geometry, mask_geometry,
    )
