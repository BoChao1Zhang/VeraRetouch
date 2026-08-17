"""The ``<seg_color>`` conditional readout for Stage-What-B.

    z = norm(hidden_states[-1]) at the <seg_color> (id 151674) position, (2560,)

Two things this module is careful about.

**1. It does not re-declare the hidden-state contract.**  ``SEGMENT_HIDDEN_LAYER``
/ ``SEGMENT_HIDDEN_FINAL_NORM`` are imported from ``q3vl.whereb.contracts``,
whose ``SEGMENT_HIDDEN_RULING`` (``contracts.py:34-40``) says in so many words
that Stage-What must import them rather than declare its own.

**2. It does not touch the contamination edge.**  ``q3vl/whereb/readout.py``
carries the mechanism (``ReplyPlan`` / ``build_reply`` / ``verify_plan`` /
``readout_hidden``) and whatb reuses it verbatim -- but its
``ReadoutBuilder.color_ids_from_text`` (``readout.py:478-486``) imports
``q3vl.what.context`` on line 484.  whatb therefore brings its own builder,
which takes the colour span from :mod:`q3vl.whatb.colorspan`.  Nothing here ever
calls ``ReadoutBuilder``.

The two new kinds (HANDOFF §八·8.4)::

    seg_color        seq = w + c + [<seg_where>, <seg_color>]
                     idx = len(seq) - 1 ; expected_ids = (<seg_color>,)
    color_span_pool  seq = w + c ; start/end cover the whole colour span,
                     pool = True   (shape of the where_span_pool branch,
                     whereb/readout.py:305-312)

They are built **here** instead of by editing ``whereb/readout.py``: EPR-018..023
are running right now and their source sha256 is frozen in their run_setup, so
the six live arms' module is left byte-identical.  ``whereb/readout.py``'s own
six kinds are reached by delegation, so there is still exactly one implementation
of each kind (NOTES 1).

The remaining four kinds of the ``--readout`` ablation group
(``color_close`` / ``im_end`` / ``seg_where`` / ``qtok``) are served by
delegating to ``whereb.readout.build_reply``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from q3vl.train.constants import SEG_COLOR_TOK, SEG_WHERE_TOK
from q3vl.whereb.contracts import (  # noqa: F401 -- re-export, never re-declare
    SEGMENT_HIDDEN_FINAL_NORM,
    SEGMENT_HIDDEN_LAYER,
    SEGMENT_HIDDEN_RULING,
)
from q3vl.whereb.readout import (
    ReadoutSpec as _WhereReadoutSpec,
    ReplyPlan,
    SegTokenIds,
    build_reply as _whereb_build_reply,
    readout_hidden,
    readout_vector,
    verify_plan,
)

from .colorspan import encode_color_span

__all__ = [
    "WHATB_READOUT_KINDS",
    "WHATB_NEEDS_V2SEG",
    "WhatReadoutSpec",
    "SegTokenIds",
    "ReplyPlan",
    "build_reply",
    "verify_plan",
    "readout_hidden",
    "readout_vector",
    "WhatReadoutBuilder",
    "SEGMENT_HIDDEN_LAYER",
    "SEGMENT_HIDDEN_FINAL_NORM",
]

#: EPR-024 §3.6-⑤ ``--readout`` choice list; default ``seg_color``.
WHATB_READOUT_KINDS: tuple[str, ...] = (
    "seg_color", "color_span_pool", "color_close", "im_end", "seg_where", "qtok",
)

#: kinds that only exist on a v2seg base (the seg tokens are supervised there).
WHATB_NEEDS_V2SEG: frozenset[str] = frozenset({"seg_color", "seg_where"})

#: the two kinds whatb builds itself; the rest delegate to whereb.
_OWN_KINDS: frozenset[str] = frozenset({"seg_color", "color_span_pool"})


@dataclass(frozen=True)
class WhatReadoutSpec:
    """``--readout`` / ``--readout-qtok`` as one frozen record for run_setup."""

    kind: str = "seg_color"
    qtok: int = 0

    def __post_init__(self) -> None:
        if self.kind not in WHATB_READOUT_KINDS:
            raise ValueError(
                f"unknown --readout {self.kind!r}; expected one of "
                f"{WHATB_READOUT_KINDS}")
        if self.kind == "qtok":
            if int(self.qtok) < 1:
                raise ValueError("--readout qtok needs --readout-qtok K >= 1")
        elif int(self.qtok):
            raise ValueError(
                "--readout-qtok is only read under --readout qtok, got "
                f"kind={self.kind!r} with qtok={self.qtok}")

    @property
    def n_vectors(self) -> int:
        return int(self.qtok) if self.kind == "qtok" else 1

    @property
    def needs_color(self) -> bool:
        """Every whatb kind consumes the colour span -- that is the point."""
        return True

    def to_dict(self) -> dict[str, Any]:
        return {"readout": self.kind, "readout_qtok": int(self.qtok),
                "n_vectors": self.n_vectors,
                "needs_v2seg": self.kind in WHATB_NEEDS_V2SEG,
                "hidden_layer": SEGMENT_HIDDEN_LAYER,
                "hidden_final_norm": SEGMENT_HIDDEN_FINAL_NORM}

    def _where_spec(self) -> _WhereReadoutSpec:
        return _WhereReadoutSpec(kind=self.kind, qtok=int(self.qtok))


def build_reply(
    tags: SegTokenIds,
    spec: WhatReadoutSpec,
    *,
    where_ids: Sequence[int],
    color_ids: Sequence[int],
    source: str = "teacher",
    on_missing_tag: str = "last",
) -> ReplyPlan:
    """Assemble the reply span and record the readout index (no later search).

    The reply always carries the **complete** reply the SFT was trained to emit
    up to the readout position: where span + colour span + the seg token(s).
    ``<seg_color>`` sits after ``<seg_where>`` in the v2seg template
    (``q3vl/train/constants.py:26-28`` registration order; the same tail the
    ``im_end`` branch builds at ``whereb/readout.py:346``).
    """
    if on_missing_tag not in ("last", "raise"):
        raise ValueError(
            f"on_missing_tag must be 'last' or 'raise', got {on_missing_tag!r}")
    if spec.kind not in _OWN_KINDS:
        return _whereb_build_reply(
            tags, spec._where_spec(), where_ids=where_ids, color_ids=color_ids,
            source=source, on_missing_tag=on_missing_tag)

    w = [int(t) for t in where_ids]
    c = [int(t) for t in color_ids]
    flags: dict[str, Any] = {}
    if not w:
        flags["null_context"] = True
    if not c:
        raise ValueError(
            f"--readout {spec.kind} needs a non-empty <color> span; got none.  "
            "The whole point of the what-side readout is the colour half of the "
            "reply -- an empty span would silently read the where half.")

    if spec.kind == "color_span_pool":
        seq = w + c
        return ReplyPlan(token_ids=seq, start=len(w), end=len(w) + len(c),
                         pool=True, kind=spec.kind, source=source,
                         expected_ids=(), flags=flags)

    # seg_color
    if not tags.has_seg:
        raise ValueError(
            "--readout seg_color needs a v2seg base: <seg_color> is not in this "
            "tokenizer's vocabulary.  The base for this campaign is "
            "q3vl_base_sft_v2seg_20260814/checkpoint-4976.")
    if c[-1] != tags.color_close:
        if on_missing_tag == "raise":
            raise ValueError(
                "--readout seg_color: the <color> span does not end in "
                f"</color> (id {tags.color_close}); last id is {c[-1]}.  This is "
                "the generated format-failure case; pass on_missing_tag='last' "
                "to accept it and have it counted.")
        flags["missing_tag"] = "<color>"
    seq = w + c + [int(tags.seg_where), int(tags.seg_color)]
    idx = len(seq) - 1
    return ReplyPlan(token_ids=seq, start=idx, end=idx + 1, pool=False,
                     kind=spec.kind, source=source,
                     expected_ids=(int(tags.seg_color),), flags=flags)


class WhatReadoutBuilder:
    """Per-run object: ``(where span, colour text) -> ReplyPlan``.

    Wiring::

        ro = WhatReadoutBuilder(tokenizer, WhatReadoutSpec("seg_color"))
        plan = ro.plan_for(sample_id=sid, where_ids=ctx.token_ids,
                           color_text=rec["color"])         # teacher
        ...                                                  # encode the reply
        z = readout_vector(h_reply, plan)                    # (2560,)

    ``facts()`` goes into ``run_setup.json``.  It carries the colour-span
    start-up record too, so a run whose token ids were never checked is visible
    on the artifact rather than only in a log line.
    """

    def __init__(self, tokenizer, spec: WhatReadoutSpec | None = None, *,
                 on_missing_tag: str = "last", control_color: str = "own"):
        self.tokenizer = tokenizer
        self.spec = spec or WhatReadoutSpec()
        self.tags = SegTokenIds.from_tokenizer(tokenizer)
        self.on_missing_tag = on_missing_tag
        if control_color not in ("own", "regenerated"):
            raise ValueError("control_color must be 'own' or 'regenerated'")
        #: NOTES: under N1/N2/N3 the instruction changes and the reasoning is
        #: **re-generated** (frozen block ⑥), so the colour span of a control is
        #: the control's own generated span -- ``"regenerated"``.  ``"own"`` (the
        #: default for teacher forcing) keeps the sample's GT colour text.
        self.control_color = control_color
        self.counts: dict[str, int] = {}
        self.colorspan_check: dict[str, Any] | None = None
        if self.spec.kind in WHATB_NEEDS_V2SEG and not self.tags.has_seg:
            raise ValueError(
                f"--readout {self.spec.kind} needs a v2seg base; this tokenizer "
                "has no <seg_color> token")

    # -- colour span (own implementation, no q3vl.what import) --------------
    def color_ids_from_text(self, color_text: str) -> list[int]:
        return encode_color_span(self.tokenizer, color_text)

    def run_colorspan_assertion(self, texts: Sequence[str], **kw) -> dict[str, Any]:
        """Frozen-block start-up assertion; result is kept for ``facts()``."""
        from .colorspan import assert_color_span_encoding

        self.colorspan_check = assert_color_span_encoding(
            self.tokenizer, texts, **kw)
        return self.colorspan_check

    # -- the plan -----------------------------------------------------------
    def plan_for(self, *, sample_id: str, where_ids: Sequence[int],
                 color_text: str | None = None,
                 color_ids: Sequence[int] | None = None,
                 source: str = "teacher",
                 control_tag: str = "none") -> ReplyPlan:
        if color_ids is not None:
            c = [int(t) for t in color_ids]
        elif color_text is not None:
            c = self.color_ids_from_text(color_text)
        else:
            raise ValueError(
                f"{sample_id}: --readout {self.spec.kind} needs the <color> "
                "span; pass color_ids= (generated / control: the regenerated "
                "reasoning's colour ids) or color_text= (teacher: rec['color'])")
        plan = build_reply(self.tags, self.spec, where_ids=where_ids,
                           color_ids=c, source=source,
                           on_missing_tag=self.on_missing_tag)
        verify_plan(plan)
        self._count(f"source_{source}")
        self._count(f"control_{control_tag}")
        for k in plan.flags:
            self._count(f"flag_{k}")
        return plan

    def _count(self, key: str) -> None:
        self.counts[key] = self.counts.get(key, 0) + 1

    def facts(self) -> dict[str, Any]:
        return {**self.spec.to_dict(),
                "tokens": self.tags.to_dict(),
                "seg_where_tok": SEG_WHERE_TOK, "seg_color_tok": SEG_COLOR_TOK,
                "on_missing_tag": self.on_missing_tag,
                "control_color": self.control_color,
                "colorspan_check": self.colorspan_check,
                "counts": dict(sorted(self.counts.items()))}
