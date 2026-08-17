"""``h_cond`` readout: WHERE in the assistant reply the language condition is taken from.

Shared infrastructure for the EPR-018..023 batch (SEGSAM / SAMDEC / PRND / LIIF /
MATTE / CONDINST).  Every one of those arms consumes **one** language vector

    h_cond = norm(hidden_states[-1])[<position>]      # (2560,)

and every one of them pre-registers the same readout-ablation group.  This module
is the single place that decides *which position* and *which token sequence*, so
the six arms cannot drift apart on the one axis they are supposed to share.

Why this is a new module and not an edit of :mod:`q3vl.whereb.hiddens`
----------------------------------------------------------------------
:meth:`q3vl.whereb.hiddens.FrozenVLM.encode` already returns
``h_where = hidden[i, n_p : n_p + n_w]`` -- the rows of the *reply* span, in reply
order (``hiddens.py:170`` builds ``prompt_ids + where_ids``, ``hiddens.py:234``
slices).  So a different readout is entirely a question of

1. what goes into ``EncodeItem.where_ids`` (the reply span), and
2. which **row index inside that span** is read.

Both are computed here, up front, and travel together as a :class:`ReplyPlan`.
``hiddens.py`` is not touched at all, so the four live arms (P1 / P3prime /
SHAPE3 / UNIQ) keep their exact ``h_where`` path, bit for bit.

The readout index is *recorded at concatenation time* -- never searched for in
the encoded sequence afterwards (proposal §1.2.1 (a)).  :func:`verify_plan` then
re-checks that the recorded index really carries the token it claims, which is
the runtime assertion for "the readout flag is wired".

The six kinds
-------------
====================  =========================================================
``seg_where``         v2seg's supervised ``<seg_where>`` (id 151673), single row.
                      Reply fed = ``where span + color span + <seg_where>``.
``where_span_pool``   mean-pool of the whole ``<where>...</where>`` span
                      (both tag tokens included).  Reply fed = where span only.
``where_close``       the ``</where>`` row (id 151670).  Reply = where span.
``color_close``       the ``</color>`` row (id 151672).  Reply = where + color.
``im_end``            the ``<|im_end|>`` row (id 151645).  Reply = where + color
                      + the template's trailing seg tokens + ``<|im_end|>``.
``qtok``              K learnable query tokens appended AFTER the full reply;
                      their last-layer hiddens are the readout.  Mechanism =
                      vocabulary extension + embedding forward hook, reused
                      verbatim from ``q3vl/whereb/amort/uniq4.py:76-113,148``.
====================  =========================================================

Teacher and generated context both go through the same builder: they differ only
in which token ids the ``<where>`` / ``<color>`` spans carry, which is exactly
the property ``hiddens.py``'s docstring protects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch

from q3vl.train.constants import (
    COLOR_CLOSE,
    COLOR_OPEN,
    SEG_COLOR_TOK,
    SEG_WHERE_TOK,
    WHERE_CLOSE,
    WHERE_OPEN,
)

from .contracts import SEGMENT_HIDDEN_FINAL_NORM, SEGMENT_HIDDEN_LAYER  # noqa: F401

__all__ = [
    "READOUT_KINDS",
    "READOUT_NEEDS_V2SEG",
    "IM_END_TOKEN",
    "ReadoutSpec",
    "SegTokenIds",
    "ReplyPlan",
    "ReadoutBuilder",
    "build_reply",
    "verify_plan",
    "readout_hidden",
    "readout_vector",
    "make_readout_vlm",
    "set_readout_grad",
    "gt_color_text",
    "color_texts_of",
]

#: the ``--readout`` choice list every arm in the EPR-018..023 batch exposes
READOUT_KINDS: tuple[str, ...] = (
    "seg_where", "where_span_pool", "where_close", "color_close", "im_end", "qtok",
)

#: which kinds only exist on a v2seg base (the two seg tokens are supervised
#: there and nowhere else).  ``im_end`` is listed as *not* needing v2seg: it
#: exists on checkpoint-4976 too, just with a shorter trailing template.
READOUT_NEEDS_V2SEG: frozenset[str] = frozenset({"seg_where"})

IM_END_TOKEN = "<|im_end|>"

#: literal ids on the Qwen3-VL-4B v2seg tokenizer, used only to cross-check what
#: the tokenizer hands back (``q3vl/train/constants.py:11-14,22-23``;
#: ``q3vl/whereb/attnread.py:62-64``).  Never used *instead* of the tokenizer.
KNOWN_IDS: dict[str, int] = {
    WHERE_OPEN: 151669, WHERE_CLOSE: 151670,
    COLOR_OPEN: 151671, COLOR_CLOSE: 151672,
    SEG_WHERE_TOK: 151673, SEG_COLOR_TOK: 151674,
    IM_END_TOKEN: 151645,
}


# --------------------------------------------------------------------------- #
# spec + token ids
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ReadoutSpec:
    """The three ``--readout*`` flags, as one frozen record for ``run_setup.json``.

    ``kind`` is the ``--readout`` value; ``qtok`` is ``--readout-qtok`` (K_q,
    only read when ``kind == "qtok"``); ``nseg`` is ``--readout-nseg`` (K, only
    read when ``kind == "seg_where"``; K > 1 needs an SFT variant that supervises
    K seg tokens and is a placeholder here).
    """

    kind: str = "seg_where"
    qtok: int = 0
    nseg: int = 1

    def __post_init__(self) -> None:
        if self.kind not in READOUT_KINDS:
            raise ValueError(
                f"unknown --readout {self.kind!r}; expected one of {READOUT_KINDS}")
        if self.kind == "qtok":
            if int(self.qtok) < 1:
                raise ValueError(
                    "--readout qtok needs --readout-qtok K >= 1 (the ablation "
                    "rows are K = 1 / 4 / 8)")
        elif int(self.qtok):
            raise ValueError(
                f"--readout-qtok is only read under --readout qtok, got "
                f"kind={self.kind!r} with qtok={self.qtok}")
        if self.kind == "seg_where":
            if int(self.nseg) < 1:
                raise ValueError("--readout-nseg must be >= 1")
        elif int(self.nseg) != 1:
            raise ValueError(
                f"--readout-nseg is only read under --readout seg_where, got "
                f"kind={self.kind!r} with nseg={self.nseg}")

    @property
    def n_vectors(self) -> int:
        """How many (2560,) rows :func:`readout_hidden` returns."""
        if self.kind == "qtok":
            return int(self.qtok)
        if self.kind == "seg_where":
            return int(self.nseg)
        return 1

    def to_dict(self) -> dict[str, Any]:
        return {"readout": self.kind, "readout_qtok": int(self.qtok),
                "readout_nseg": int(self.nseg), "n_vectors": self.n_vectors,
                "needs_v2seg": self.kind in READOUT_NEEDS_V2SEG}


@dataclass(frozen=True)
class SegTokenIds:
    """The tag ids this module needs, resolved once from a tokenizer.

    ``has_seg`` is False on the pre-v2seg base (checkpoint-4976): the two seg
    tokens are simply not in its vocabulary, so ``--readout seg_where`` cannot be
    served and says so instead of silently reading some other row.
    """

    where_open: int
    where_close: int
    color_open: int
    color_close: int
    im_end: int
    seg_where: int | None = None
    seg_color: int | None = None

    @property
    def has_seg(self) -> bool:
        return self.seg_where is not None and self.seg_color is not None

    @classmethod
    def from_tokenizer(cls, tokenizer) -> "SegTokenIds":
        def one(tok: str, required: bool = True) -> int | None:
            ids = tokenizer(tok, add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                if required:
                    raise ValueError(f"{tok!r} is not a single token: {list(ids)}")
                return None
            return int(ids[0])

        return cls(
            where_open=one(WHERE_OPEN), where_close=one(WHERE_CLOSE),
            color_open=one(COLOR_OPEN), color_close=one(COLOR_CLOSE),
            im_end=one(IM_END_TOKEN),
            seg_where=one(SEG_WHERE_TOK, required=False),
            seg_color=one(SEG_COLOR_TOK, required=False),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"where_open_id": self.where_open, "where_close_id": self.where_close,
                "color_open_id": self.color_open, "color_close_id": self.color_close,
                "im_end_id": self.im_end, "seg_where_id": self.seg_where,
                "seg_color_id": self.seg_color, "has_seg_tokens": self.has_seg}


# --------------------------------------------------------------------------- #
# the plan
# --------------------------------------------------------------------------- #
@dataclass
class ReplyPlan:
    """One sample's reply span **and** the row index the condition is read from.

    ``token_ids`` goes straight into ``EncodeItem.where_ids``; ``start``/``end``
    index into the rows of ``EncodeResult.h_where`` (same order, same length --
    ``hiddens.py:234`` slices ``hidden[n_p : n_p + n_w]``).

    For ``kind == "qtok"`` the K query ids are appended by
    :class:`~q3vl.whereb.amort.uniq4.QueryTokVLM` inside ``encode``, *after* this
    span; ``start``/``end`` therefore point past ``len(token_ids)`` and
    ``n_appended`` records how many rows the wrapper adds.
    """

    token_ids: list[int]
    start: int
    end: int
    pool: bool
    kind: str
    source: str = "teacher"
    expected_ids: tuple[int, ...] = ()
    n_appended: int = 0
    flags: dict[str, Any] = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        """Rows of ``h_where`` the readout consumes (before pooling)."""
        return self.end - self.start

    @property
    def expected_len(self) -> int:
        """Length ``h_where`` must have for this plan to be applicable."""
        return len(self.token_ids) + self.n_appended

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "source": self.source, "start": self.start,
                "end": self.end, "pool": self.pool,
                "n_tokens": len(self.token_ids), "n_appended": self.n_appended,
                "expected_ids": list(self.expected_ids),
                "flags": dict(self.flags)}


def build_reply(
    tags: SegTokenIds,
    spec: ReadoutSpec,
    *,
    where_ids: Sequence[int],
    color_ids: Sequence[int] = (),
    source: str = "teacher",
    on_missing_tag: str = "last",
) -> ReplyPlan:
    """Assemble the reply span and record the readout index.  No searching later.

    ``where_ids`` = the ``<where>...</where>`` span (teacher: ``encode_where_span``;
    generated: the cached ``where_ids`` of the genwhere record; negative controls:
    the control's own span).  ``color_ids`` = the ``<color>...</color>`` span,
    needed by ``color_close`` / ``im_end`` / ``seg_where`` / ``qtok``.

    ``on_missing_tag`` governs the *format-failure* case only (a generated span
    that never closed, ``q3vl/whereb/context.py:extract_segment`` stop_reason
    ``no_close_tag``): ``"last"`` reads the last row of the span and sets
    ``flags["missing_tag"]`` so the caller can count it; ``"raise"`` refuses.
    Never a silent fallback either way.
    """
    if on_missing_tag not in ("last", "raise"):
        raise ValueError(f"on_missing_tag must be 'last' or 'raise', got {on_missing_tag!r}")
    w = [int(t) for t in where_ids]
    c = [int(t) for t in color_ids]
    flags: dict[str, Any] = {}
    if not w:
        flags["null_context"] = True

    def _tail_row(seq: list[int], want: int, label: str) -> int:
        """Index of the last row, asserting (or flagging) that it is ``want``."""
        if not seq:
            raise ValueError(
                f"--readout {spec.kind} needs a non-empty {label} span; got none. "
                "The null context carries no tokens, so this readout is undefined "
                "for it -- pick a kind whose span is always present, or exclude "
                "the null context from this arm.")
        idx = len(seq) - 1
        if seq[idx] != want:
            if on_missing_tag == "raise":
                raise ValueError(
                    f"--readout {spec.kind}: the {label} span does not end in the "
                    f"expected tag (id {want}); last id is {seq[idx]}.  This is the "
                    "generated format-failure case; pass on_missing_tag='last' to "
                    "read the last row and count it.")
            flags["missing_tag"] = label
        return idx

    if spec.kind == "where_span_pool":
        if not w:
            raise ValueError(
                "--readout where_span_pool needs a non-empty <where> span "
                "(the null context has none)")
        return ReplyPlan(token_ids=list(w), start=0, end=len(w), pool=True,
                         kind=spec.kind, source=source, expected_ids=(), flags=flags)

    if spec.kind == "where_close":
        idx = _tail_row(w, tags.where_close, "<where>")
        return ReplyPlan(token_ids=list(w), start=idx, end=idx + 1, pool=False,
                         kind=spec.kind, source=source,
                         expected_ids=(tags.where_close,), flags=flags)

    if spec.kind == "color_close":
        seq = w + c
        idx = _tail_row(c, tags.color_close, "<color>")
        idx += len(w)
        return ReplyPlan(token_ids=seq, start=idx, end=idx + 1, pool=False,
                         kind=spec.kind, source=source,
                         expected_ids=(tags.color_close,), flags=flags)

    if spec.kind == "seg_where":
        if not tags.has_seg:
            raise ValueError(
                "--readout seg_where needs a v2seg base: <seg_where> is not in "
                "this tokenizer's vocabulary.  checkpoint-4976 predates the two "
                "seg tokens; use --readout im_end / where_close / color_close "
                "there (proposal D-7).")
        if int(spec.nseg) != 1:
            raise NotImplementedError(
                f"--readout-nseg {spec.nseg}: v2seg supervises exactly ONE "
                "<seg_where>.  K > 1 needs a corresponding SFT variant and is a "
                "placeholder row in the proposals (not scheduled).")
        seq = w + c + [int(tags.seg_where)]
        idx = len(seq) - 1
        return ReplyPlan(token_ids=seq, start=idx, end=idx + 1, pool=False,
                         kind=spec.kind, source=source,
                         expected_ids=(int(tags.seg_where),), flags=flags)

    if spec.kind == "im_end":
        tail = ([int(tags.seg_where), int(tags.seg_color)] if tags.has_seg else [])
        seq = w + c + tail + [tags.im_end]
        idx = len(seq) - 1
        flags["template"] = "v2seg" if tags.has_seg else "v1"
        return ReplyPlan(token_ids=seq, start=idx, end=idx + 1, pool=False,
                         kind=spec.kind, source=source,
                         expected_ids=(tags.im_end,), flags=flags)

    # qtok: the full reasoning reply, query ids appended by the VLM wrapper
    seq = list(w) + list(c)
    if tags.has_seg:
        seq.append(int(tags.seg_where))
    k = int(spec.qtok)
    flags["template"] = "v2seg" if tags.has_seg else "v1"
    return ReplyPlan(token_ids=seq, start=len(seq), end=len(seq) + k, pool=False,
                     kind=spec.kind, source=source, expected_ids=(), n_appended=k,
                     flags=flags)


def verify_plan(plan: ReplyPlan) -> None:
    """Runtime assertion: the recorded index really carries the token it claims.

    "Defined but not wired" has cost this campaign three times.  A readout flag
    that silently reads the wrong row produces a perfectly plausible board, so
    this runs on every plan, not on a sample of them.
    """
    if plan.start < 0 or plan.end <= plan.start:
        raise AssertionError(f"readout plan has empty slice {plan.start}:{plan.end}")
    if plan.n_appended:
        if plan.start != len(plan.token_ids):
            raise AssertionError(
                f"qtok readout must start right after the reply span "
                f"({len(plan.token_ids)}), got {plan.start}")
        if plan.end != plan.start + plan.n_appended:
            raise AssertionError("qtok readout slice does not cover the K query rows")
        return
    if plan.end > len(plan.token_ids):
        raise AssertionError(
            f"readout slice {plan.start}:{plan.end} runs past the reply span "
            f"({len(plan.token_ids)} tokens)")
    if not plan.expected_ids:
        return
    got = tuple(plan.token_ids[plan.start:plan.end])
    if got != tuple(plan.expected_ids) and "missing_tag" not in plan.flags:
        raise AssertionError(
            f"readout {plan.kind!r} points at token ids {got}, expected "
            f"{tuple(plan.expected_ids)} -- the flag is not wired to the position "
            "it names")


def readout_hidden(h_where: torch.Tensor, plan: ReplyPlan) -> torch.Tensor:
    """``(T, 2560)`` reply hiddens -> ``(K, 2560)`` condition rows.

    ``K`` is 1 for every single-position kind and for ``where_span_pool`` (whose
    rows are mean-pooled), ``K_q`` for ``qtok``.  The length assertion is the
    other half of the wiring check: a plan built for one reply and applied to a
    different one is a bug that no metric would show.
    """
    if h_where.dim() != 2:
        raise ValueError(f"expected (T, D) hiddens, got {tuple(h_where.shape)}")
    if h_where.shape[0] != plan.expected_len:
        raise AssertionError(
            f"readout plan expects {plan.expected_len} reply rows "
            f"({len(plan.token_ids)} tokens + {plan.n_appended} appended), the "
            f"encoder returned {h_where.shape[0]}")
    rows = h_where[plan.start:plan.end]
    if plan.pool:
        return rows.mean(dim=0, keepdim=True)
    return rows


def readout_vector(h_where: torch.Tensor, plan: ReplyPlan) -> torch.Tensor:
    """:func:`readout_hidden` for the single-vector kinds -> ``(2560,)``."""
    rows = readout_hidden(h_where, plan)
    if rows.shape[0] != 1:
        raise ValueError(
            f"readout {plan.kind!r} yields {rows.shape[0]} vectors; use "
            "readout_hidden() and say how the head aggregates them")
    return rows[0]


# --------------------------------------------------------------------------- #
# builder: sample + context -> plan
# --------------------------------------------------------------------------- #
class ReadoutBuilder:
    """Per-run object a batch builder holds: context -> ``(where_ids, plan)``.

    Wiring, in the arm's ``AmortBatchBuilder`` subclass / branch::

        ro = ReadoutBuilder(tokenizer, spec)                 # once per run
        plan = ro.plan_for(sample_id=s.sample_id, where_ctx=ctx,
                           color_text=<GT colour text>,      # teacher
                           color_ids=rec["color_ids"])       # generated
        item = EncodeItem(sample_id=..., image=..., prompt_ids=...,
                          where_ids=plan.token_ids)
        ...
        h_cond = readout_vector(enc.h_where, plan)           # (2560,)

    ``facts()`` goes into ``run_setup.json`` next to the source sha256s.
    """

    def __init__(
        self,
        tokenizer,
        spec: ReadoutSpec | None = None,
        *,
        on_missing_tag: str = "last",
        control_color: str = "own",
    ):
        self.tokenizer = tokenizer
        self.spec = spec or ReadoutSpec()
        self.tags = SegTokenIds.from_tokenizer(tokenizer)
        self.on_missing_tag = on_missing_tag
        if control_color not in ("own", "none"):
            raise ValueError("control_color must be 'own' or 'none'")
        #: NOTES (not silently decided): the negative controls swap the
        #: instruction and the ``<where>`` span.  The proposals do not say what
        #: the ``<color>`` span should be under a control.  ``"own"`` (default)
        #: keeps the sample's own GT colour span so the reply keeps its shape and
        #: only the where half is swapped; ``"none"`` drops the colour span.
        self.control_color = control_color
        self.counts: dict[str, int] = {}
        if self.spec.kind in READOUT_NEEDS_V2SEG and not self.tags.has_seg:
            raise ValueError(
                f"--readout {self.spec.kind} needs a v2seg base; this tokenizer "
                "has no <seg_where> token (see proposal D-7 for the pre-v2seg "
                "start-up options)")

    # -- colour span -------------------------------------------------------
    def needs_color(self) -> bool:
        return self.spec.kind in ("color_close", "im_end", "seg_where", "qtok")

    def color_ids_from_text(self, color_text: str) -> list[int]:
        """``<color>{text}</color>`` -> ids, tokenised as the SFT collator does.

        Delegates to Stage-What's own encoder so the two stages cannot tokenise
        the same span differently (``q3vl/what/context.py:92-95``).
        """
        from q3vl.what.context import encode_color_span

        return list(encode_color_span(self.tokenizer, color_text))

    # -- the plan ----------------------------------------------------------
    def plan_for(
        self,
        *,
        sample_id: str,
        where_ctx: Any,
        color_text: str | None = None,
        color_ids: Sequence[int] | None = None,
    ) -> ReplyPlan:
        """One :class:`ReplyPlan`.  ``where_ctx`` is a
        :class:`q3vl.whereb.context.WhereContext` (any mode)."""
        mode = getattr(where_ctx, "mode", "gt")
        w = list(getattr(where_ctx, "token_ids", ()) or ())
        c: list[int] = []
        if self.needs_color():
            is_control = mode not in ("gt", "generated")
            if is_control and self.control_color == "none":
                c = []
            elif color_ids is not None:
                c = [int(t) for t in color_ids]
            elif color_text is not None:
                c = self.color_ids_from_text(color_text)
            else:
                raise ValueError(
                    f"{sample_id}: --readout {self.spec.kind} needs the <color> "
                    "span; pass color_ids= (generated: the genwhere record's "
                    "'color_ids') or color_text= (teacher: the record's 'color' "
                    "field, see readout.gt_color_text)")
        src = "generated" if mode == "generated" else "teacher"
        plan = build_reply(self.tags, self.spec, where_ids=w, color_ids=c,
                           source=src, on_missing_tag=self.on_missing_tag)
        verify_plan(plan)
        self._count(f"mode_{mode}")
        self._count(f"source_{src}")
        for k in plan.flags:
            self._count(f"flag_{k}")
        return plan

    def _count(self, key: str) -> None:
        self.counts[key] = self.counts.get(key, 0) + 1

    def facts(self) -> dict[str, Any]:
        return {**self.spec.to_dict(), "tokens": self.tags.to_dict(),
                "on_missing_tag": self.on_missing_tag,
                "control_color": self.control_color,
                "counts": dict(sorted(self.counts.items()))}


def gt_color_text(record: Mapping[str, Any] | Any) -> str:
    """The GT ``<color>`` body of one dataset record.

    ``WhereBSample`` does not carry it (``q3vl/whereb/data.py:71-75`` keeps
    ``META_KEYS`` only, and ``color`` is not one of them) while Stage-What reads
    it as ``rec["color"]`` (``q3vl/what/data.py:279``).  Arms that need the
    teacher colour span therefore read it from the raw record --
    ``WhereBDataset.record(i)`` -- and pass it to
    :meth:`ReadoutBuilder.plan_for`.
    """
    if isinstance(record, Mapping):
        val = record.get("color")
    else:
        val = getattr(record, "color_text", None)
        if val is None:
            val = getattr(record, "color", None)
    if not isinstance(val, str):
        raise KeyError(
            "record carries no 'color' field; the teacher colour span cannot be "
            "synthesised without it (Stage-What reads rec['color'], "
            "q3vl/what/data.py:279)")
    return val


def color_texts_of(ds, indices: Sequence[int] | None = None,
                   workers: int = 16) -> dict[str, str]:
    """``sample_id -> GT <color> body`` for a split, read once up front.

    ``WhereBSample`` drops the field (``META_KEYS``), so the teacher colour span
    has to come from the raw records; doing it once here costs one pass and
    keeps the training loop from paying a shard read per sample.  Shape mirrors
    ``q3vl.whereb.amort.data.family_labels``.
    """
    from concurrent.futures import ThreadPoolExecutor

    idx = list(range(len(ds))) if indices is None else list(indices)

    def one(i: int) -> tuple[str, str] | None:
        try:
            rec = ds.record(i)
        except Exception:  # noqa: BLE001 -- a missing record is counted, not fatal
            return None
        sid = rec.get("sample_id") or ds.refs[i].sample_id
        val = rec.get("color")
        return (str(sid), str(val)) if isinstance(val, str) else None

    out: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        for got in ex.map(one, idx):
            if got is not None:
                out[got[0]] = got[1]
    return out


# --------------------------------------------------------------------------- #
# the qtok VLM wrapper
# --------------------------------------------------------------------------- #
def make_readout_vlm(model, processor, device="cuda", *, spec: ReadoutSpec | None = None,
                     **kw):
    """``FrozenVLM`` for every kind except ``qtok``, ``QueryTokVLM`` for that one.

    ``qtok`` reuses ``q3vl/whereb/amort/uniq4.py``'s mechanism verbatim
    (vocabulary extension + embedding forward hook + trainable fp-native query
    embeddings, ``uniq4.py:76-113``; the ids are appended to ``where_ids`` at
    ``uniq4.py:148``).  ``lora=False`` -- the base stays frozen; only the K query
    embeddings are trainable, and gradients reach them through the VLM forward,
    which is why that class turns gradient checkpointing on.

    The wrapper's train/eval gate is ``_grad_on``; the arm's model must flip it
    (see :func:`set_readout_grad`) or the query embeddings never get a gradient.
    """
    spec = spec or ReadoutSpec()
    if spec.kind != "qtok":
        from .hiddens import FrozenVLM

        return FrozenVLM(model, processor, device, **kw)
    from .amort.uniq4 import QueryTokVLM

    return QueryTokVLM(model, processor, device, n_qtok=int(spec.qtok), lora=False, **kw)


def set_readout_grad(vlm, flag: bool) -> None:
    """Flip ``QueryTokVLM._grad_on``; a no-op for a plain ``FrozenVLM``.

    Call it from the arm model's ``train()`` / ``eval()``, the way
    ``AmortModelV4`` does.  Without it the ``qtok`` rows are produced under
    ``no_grad`` and the "learnable query" ablation trains nothing while looking
    exactly like a run that did.
    """
    if hasattr(vlm, "_grad_on"):
        vlm._grad_on = bool(flag)
