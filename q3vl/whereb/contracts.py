"""Cross-stage contracts.  **Stage-What must import from here, not re-declare.**

Nit N1 of ``docs/reviews/REVIEW-impl-WhereB.md``: ruling D-B2 fixes *one* hidden
-state convention for the whole campaign -- ``H_where`` for Stage-Where-B and
``H_color`` for Stage-What have to be read out of the same place, or the two
stages' language conditioning is not comparable and nobody would notice.  While
the constants lived in ``q3vl/whereb/config.py`` a future Stage-What could
silently declare its own.  They live here instead, in a module that belongs to
neither stage.

VERIFIED 2026-08-05 on transformers 4.57.1 with a real (toy-sized)
``Qwen3VLForConditionalGeneration``:

* ``out.hidden_states`` has ``num_hidden_layers + 1`` entries;
* ``hidden_states[-1]`` is the last decoder layer output **before** the final
  RMSNorm: ``lm_head(norm(hidden_states[-1])) == logits`` while
  ``lm_head(hidden_states[-1]) != logits``.

Ruling D-B2 therefore has to say which of the two it means, and it says
post-norm -- the state ``lm_head`` sees, whose scale is normalised and which the
SFT loss actually shaped.

``q3vl/whereb/tests/test_hiddens.py`` pins the transformers behaviour;
``test_contracts.py`` pins that nothing re-declares these names.
"""

from __future__ import annotations

#: index into ``out.hidden_states`` (or into ``language_model.layers``)
SEGMENT_HIDDEN_LAYER = -1
#: apply the final RMSNorm, i.e. read the lm_head input space (ruling D-B2)
SEGMENT_HIDDEN_FINAL_NORM = True

#: the ruling these two constants encode, quoted for anyone who greps for it
SEGMENT_HIDDEN_RULING = (
    "D-B2 (2026-08-05): H_where and H_color are both read as "
    "norm(hidden_states[-1]) -- the post-final-RMSNorm state.  Stage-What must "
    "import SEGMENT_HIDDEN_LAYER / SEGMENT_HIDDEN_FINAL_NORM from "
    "q3vl.whereb.contracts rather than declaring its own."
)

__all__ = [
    "SEGMENT_HIDDEN_LAYER",
    "SEGMENT_HIDDEN_FINAL_NORM",
    "SEGMENT_HIDDEN_RULING",
]
