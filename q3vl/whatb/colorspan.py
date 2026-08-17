"""``<color>{text}</color>`` token ids -- this package's OWN implementation.

Frozen-block item 7 (six proposals, byte-identical): the ``<seg_color>`` readout
needs the colour span in the reply, and the only existing helper that builds it
(``q3vl/whereb/readout.py:478-486``) reaches ``q3vl.what.context`` on line 484 --
the contamination tree.  So whatb carries its own encoder plus a start-up
assertion that pins it, token for token, against the checkpoint's tokenizer.

The encoder is deliberately **piecewise** (open tag id + body ids + close tag
id).  A one-line ``tok(f"<color>{t}</color>")`` would make the assertion
tautological; the piecewise form is a real claim about BPE boundaries at the two
special tokens, and the assertion is what proves it holds on this tokenizer.

Start-up assertion (frozen block, verbatim)::

    ids_self = <this module>(tok, t)
    ids_ref  = tok(f"<color>{t}</color>", add_special_tokens=False).input_ids
    assert len(ids_self) == len(ids_ref)                   # length first
    assert all(a == b for a, b in zip(ids_self, ids_ref))  # then token by token

Any inequality is an ``AssertionError`` and refuses to start training; the
number of texts sampled and the number of mismatches go into ``run_setup.json``
(:func:`assert_color_span_encoding` returns exactly that record).
"""

from __future__ import annotations

import random
from typing import Any, Sequence

from q3vl.train.constants import COLOR_CLOSE, COLOR_OPEN

__all__ = [
    "COLOR_CLOSE",
    "COLOR_OPEN",
    "ColorSpanMismatch",
    "DEFAULT_ASSERT_SAMPLES",
    "DEFAULT_ASSERT_SEED",
    "color_tag_ids",
    "encode_color_span",
    "assert_color_span_encoding",
]

#: frozen block: "对本 split 随机抽 **256** 条样本的 `color` 文本"
DEFAULT_ASSERT_SAMPLES = 256
#: repository-wide seed (EPR-024 §3.4 "seed | 20260810")
DEFAULT_ASSERT_SEED = 20260810


class ColorSpanMismatch(AssertionError):
    """This module's ids differ from the tokenizer's on the joined string."""


def color_tag_ids(tokenizer: Any) -> tuple[int, int]:
    """``(<color>, </color>)`` ids, asserting each is a single token.

    Resolved from the tokenizer, never hard-coded: ``q3vl/whereb/readout.py:110``
    keeps the literal table (151671 / 151672) only as a cross-check and says so.
    """
    out: list[int] = []
    for tag in (COLOR_OPEN, COLOR_CLOSE):
        ids = list(tokenizer(tag, add_special_tokens=False)["input_ids"])
        if len(ids) != 1:
            raise ValueError(
                f"{tag!r} is not a single token on this tokenizer ({ids}); the "
                "colour span cannot be built piecewise.  This tokenizer is not "
                "the v2seg one (q3vl/train/constants.py:11-14 registers the four "
                "tags as special tokens).")
        out.append(int(ids[0]))
    return out[0], out[1]


def encode_color_span(tokenizer: Any, text: str) -> list[int]:
    """``<color>`` + body + ``</color>`` as token ids.

    ``text`` is the record's ``color`` field (the GT colour reasoning body), with
    no tags of its own.  Body tokenisation uses ``add_special_tokens=False`` so
    no BOS/EOS is spliced into the middle of a reply.
    """
    if not isinstance(text, str):
        raise TypeError(f"colour text must be str, got {type(text).__name__}")
    open_id, close_id = color_tag_ids(tokenizer)
    body = list(tokenizer(text, add_special_tokens=False)["input_ids"])
    return [open_id, *(int(t) for t in body), close_id]


def reference_color_span(tokenizer: Any, text: str) -> list[int]:
    """The frozen block's reference form ``tok(f"<color>{t}</color>")``."""
    return [int(t) for t in
            tokenizer(f"{COLOR_OPEN}{text}{COLOR_CLOSE}",
                      add_special_tokens=False)["input_ids"]]


def assert_color_span_encoding(
    tokenizer: Any,
    texts: Sequence[str],
    *,
    n_sample: int = DEFAULT_ASSERT_SAMPLES,
    seed: int = DEFAULT_ASSERT_SEED,
    tokenizer_path: str | None = None,
) -> dict[str, Any]:
    """The frozen-block start-up assertion.  Returns the ``run_setup`` record.

    Called **before** the dataloader is built, on colour texts drawn from the
    split that is about to be trained/evaluated on.  ``n_sample`` texts are drawn
    without replacement by a private :class:`random.Random` -- the global stream
    is not touched, so inserting this check does not shift any other draw in the
    run.

    Raises :class:`ColorSpanMismatch` (an ``AssertionError``) on the first text
    whose length or whose token ids differ, quoting the differing index.
    """
    pool = [t for t in texts if isinstance(t, str)]
    if not pool:
        raise ValueError(
            "no colour texts to assert on; the start-up check may not be "
            "skipped by handing it an empty list (frozen block item 7)")
    rng = random.Random(seed)
    k = min(int(n_sample), len(pool))
    picked = rng.sample(pool, k)

    n_mismatch = 0
    for i, text in enumerate(picked):
        ids_self = encode_color_span(tokenizer, text)
        ids_ref = reference_color_span(tokenizer, text)
        if len(ids_self) != len(ids_ref):
            n_mismatch += 1
            raise ColorSpanMismatch(
                f"colour span #{i}: own encoder gives {len(ids_self)} ids, the "
                f"tokenizer gives {len(ids_ref)} on the joined string; "
                f"text[:80]={text[:80]!r}")
        for j, (a, b) in enumerate(zip(ids_self, ids_ref)):
            if a != b:
                n_mismatch += 1
                raise ColorSpanMismatch(
                    f"colour span #{i}: token {j} is {a} in this module and {b} "
                    f"in the tokenizer's joined encoding; text[:80]={text[:80]!r}")

    open_id, close_id = color_tag_ids(tokenizer)
    return {
        "check": "colorspan_vs_tokenizer",
        "n_texts_available": len(pool),
        "n_sampled": k,
        "n_mismatch": n_mismatch,
        "seed": int(seed),
        "color_open_id": open_id,
        "color_close_id": close_id,
        "tokenizer": tokenizer_path,
        "implementation": "q3vl.whatb.colorspan.encode_color_span (piecewise)",
        "imports_contaminated_tree": False,
    }
