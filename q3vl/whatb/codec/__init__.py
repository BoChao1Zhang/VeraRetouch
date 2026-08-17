"""EPR-031 WhatCodec: the canonical LUT function code and the O0 carrier table.

Spec: ``experiments/prs/EPR-031_whatcodec-evidence-code-slotdecoder/PROPOSAL.md``
(§3.1 canonical code, §5's O0 row, §7's C0 / C1 gates).  The carrier itself is
**not** this package's: every primitive, loss and parameter count comes from
:mod:`q3vl.whatb.arms.g4d` (EPR-028 R1), which this package only imports.

Two modules, one per gate:

``lutcode``
    C0 -- the whitened PCA of the ``17^3`` residual of the train LUT pool, its
    ``9^3`` control against the repository's existing ledger, and the purely
    geometric ``code_recon_de00``.

``tables``
    C1 -- ``Theta in R^{n_lut x D_theta}``: one independent, directly learnable
    carrier parameter set per ``lut_id``.  No generator, no VLM, no conditional
    read-out; the O0 row of the four-level capacity decomposition (§5).
"""

from __future__ import annotations

__all__ = ["lutcode", "tables"]
