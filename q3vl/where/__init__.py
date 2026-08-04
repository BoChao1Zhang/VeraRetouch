"""Stage-Where-A: high-resolution basis calibration.

Implements section 4 of
``docs/METACANVAS_WHERE_WHAT_FINAL_EXPERIMENT_PROTOCOL_2026-08-04.md``:
``F_pre`` extraction (4.1), the 71-dim ``phi_dir`` (4.2), the two readouts
``R-Band`` / ``R-CBand12`` (4.3), the per-image multi-start L-BFGS oracle fit
and the four basis-calibration arms ``BA-0..BA-3`` (4.4).
"""

from __future__ import annotations

__all__ = [
    "basis", "calibrate", "config", "fpre", "maskdata", "oracle",
    "packing", "phi", "pipeline", "projector", "readout", "upsample",
]
