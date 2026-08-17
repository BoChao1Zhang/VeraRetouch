"""RATTN hotfix (EPR-011 wave-5) -- NEW FILE; uniq3.py is frozen under the
running RCODE arm.

Bug: the attention maps ride concatenated onto ``sim``, and ``_extra()``
feeds the SAME widened sim to both heads -- the geometry tower was widened
(extra_ch=5) but the SemanticHead stem stayed at 1024+1=1025 and crashed on
1029 channels (observed: ``weight [96, 1025] ... got 1029``).

Fix, chosen on experimental-design grounds rather than convenience: the
semantic path must stay byte-identical across arms (it is held fixed in
every EPR-011 comparison), so it sees ONLY the original similarity channel;
the attention channels are a geometry-path conditioning source.
"""

from __future__ import annotations

from .uniq3 import AmortModelV3

__all__ = ["make_v3_attn_fixed"]


def make_v3_attn_fixed(n_attn_layers: int):
    class V3AttnFixed(AmortModelV3):
        def __init__(self, arm: str = "P1", **kw):
            kw.setdefault("attn_extra_ch", n_attn_layers)
            super().__init__(arm, **kw)

        def forward_sem(self, feat, cond, *, sim=None, center=None,
                        geom=None, guide_hi=None):
            # semantic path held fixed across arms: original sim channel only
            sim1 = None if sim is None else sim[:, :1]
            return super().forward_sem(feat, cond, sim=sim1, center=center,
                                       geom=None, guide_hi=guide_hi)

    return V3AttnFixed
