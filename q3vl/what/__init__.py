"""Stage-What: independent ``Q_color`` -> continuous ``z_style`` -> 48-Gaussian LUT.

Implements sections 6-10.4, 12 and 14 (items 8b/9/12/13/14) of
``docs/METACANVAS_WHERE_WHAT_FINAL_EXPERIMENT_PROTOCOL_2026-08-04.md``.

The hidden-state convention is imported from ``q3vl.whereb.contracts`` (ruling
D-B2) and is deliberately not re-declared anywhere in this package.
"""

from __future__ import annotations

__all__ = ["config", "srht", "lut", "gaussians", "colorspace", "pooling",
           "attention", "color", "wc", "backend", "generator", "queries",
           "model", "losses", "metrics", "data", "hiddens", "trainer",
           "evaluate", "preflight"]
