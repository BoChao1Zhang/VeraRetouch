"""Fixtures for the whatb suite: a deterministic fake tokenizer, no GPU.

The fake tokenizer mirrors ``q3vl/whereb/tests/conftest.py``: the six special
tokens get fixed low ids in registration order, everything else is whitespace /
tag split.  It is deliberately *not* a stand-in for the real BPE -- the two tests
that must hold on the real tokenizer (the colour-span byte equality and the
``<seg_color>`` id) read ``checkpoint-4976`` and skip when it is not mounted.
"""

from __future__ import annotations

import re

import pytest

SPECIALS = ("<where>", "</where>", "<color>", "</color>", "<seg_where>",
            "<seg_color>", "<|im_end|>")


class FakeTokenizer:
    """Whitespace + tag tokeniser with stable ids."""

    def __init__(self, *, with_seg: bool = True) -> None:
        toks = SPECIALS if with_seg else tuple(
            t for t in SPECIALS if not t.startswith("<seg_"))
        self._vocab: dict[str, int] = {t: i + 10 for i, t in enumerate(toks)}
        self._inv: dict[int, str] = {v: k for k, v in self._vocab.items()}
        self._next = 100
        self.eos_token_id = 2
        self.with_seg = with_seg

    def _id(self, tok: str) -> int:
        if tok not in self._vocab:
            self._vocab[tok] = self._next
            self._inv[self._next] = tok
            self._next += 1
        return self._vocab[tok]

    def __call__(self, text: str, add_special_tokens: bool = False
                 ) -> dict[str, list[int]]:
        toks = re.findall(r"<\|?/?\w+\|?>|[^<\s]+", text)
        if not self.with_seg:
            toks = [t for t in toks if not t.startswith("<seg_")]
        return {"input_ids": [self._id(t) for t in toks]}

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        out = [self._inv.get(int(i), "?") for i in ids]
        if skip_special_tokens:
            out = [t for t in out if t not in SPECIALS]
        return " ".join(out)


@pytest.fixture()
def tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


@pytest.fixture()
def tokenizer_no_seg() -> FakeTokenizer:
    return FakeTokenizer(with_seg=False)
