"""The colour span, and the start-up assertion that pins it to the tokenizer."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from q3vl.whatb.colorspan import (
    ColorSpanMismatch,
    assert_color_span_encoding,
    color_tag_ids,
    encode_color_span,
    reference_color_span,
)

CHECKPOINT = Path("/home/bc/data/runs/q3vl_base_sft_v2seg_20260814/checkpoint-4976")
SPLITS = Path("/mnt/nfs-ro/bc/data/datasets/sft2seg-20260804/splits")


def test_encode_is_open_body_close(tokenizer):
    ids = encode_color_span(tokenizer, "make it warmer")
    o, c = color_tag_ids(tokenizer)
    assert ids[0] == o and ids[-1] == c and len(ids) == 5


def test_matches_the_reference_form(tokenizer):
    for text in ("a b c", "warmer, softer greens", "single"):
        assert encode_color_span(tokenizer, text) == \
            reference_color_span(tokenizer, text)


def test_assertion_returns_the_run_setup_record(tokenizer):
    rec = assert_color_span_encoding(tokenizer, [f"text {i}" for i in range(40)],
                                     n_sample=16)
    assert rec["n_sampled"] == 16 and rec["n_mismatch"] == 0
    assert rec["imports_contaminated_tree"] is False
    assert rec["color_open_id"] == color_tag_ids(tokenizer)[0]


def test_assertion_raises_on_a_disagreeing_tokenizer(tokenizer):
    class Broken(type(tokenizer)):
        def __call__(self, text, add_special_tokens=False):
            out = super().__call__(text, add_special_tokens)
            if text.startswith("<color>") and text.endswith("</color>"):
                out["input_ids"] = out["input_ids"][:-1]   # joined form differs
            return out

    with pytest.raises(ColorSpanMismatch, match="ids"):
        assert_color_span_encoding(Broken(), ["a b c"] * 4, n_sample=4)


def test_assertion_refuses_an_empty_pool(tokenizer):
    with pytest.raises(ValueError, match="empty"):
        assert_color_span_encoding(tokenizer, [], n_sample=4)


def test_no_contaminated_import_reaches_the_module():
    """No whatb module may import the contaminated trees, even transitively."""
    import re

    import q3vl.whatb.criteria  # noqa: F401
    import q3vl.whatb.lutdata  # noqa: F401
    import q3vl.whatb.readout  # noqa: F401

    assert not [m for m in sys.modules
                if m.startswith(("q3vl.what.", "model.glut_repro", "gpu_render"))]
    banned = re.compile(
        r"^\s*(from|import)\s+(q3vl\.what\.|model\.glut_repro|gpu_render)",
        re.MULTILINE)
    for path in sorted(Path(q3vl.whatb.criteria.__file__).parent.glob("*.py")):
        assert not banned.search(path.read_text()), path


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="v2seg checkpoint not mounted")
def test_real_tokenizer_byte_equality_on_real_colour_texts():
    """The frozen-block assertion itself, on the real base and real texts."""
    import itertools
    import json

    from transformers import AutoTokenizer

    from q3vl.whatb.splits import read_record

    tok = AutoTokenizer.from_pretrained(str(CHECKPOINT))
    assert tok("<color>", add_special_tokens=False)["input_ids"] == [151671]
    assert tok("</color>", add_special_tokens=False)["input_ids"] == [151672]
    assert tok("<seg_color>", add_special_tokens=False)["input_ids"] == [151674]

    if not (SPLITS / "V_what.index.jsonl").exists():
        pytest.skip("dataset splits not mounted")
    texts = []
    with (SPLITS / "V_what.index.jsonl").open() as fh:
        for line in itertools.islice(fh, 48):
            texts.append(read_record(json.loads(line))["color"])
    rec = assert_color_span_encoding(tok, texts, n_sample=48,
                                     tokenizer_path=str(CHECKPOINT))
    assert rec["n_sampled"] == 48 and rec["n_mismatch"] == 0
