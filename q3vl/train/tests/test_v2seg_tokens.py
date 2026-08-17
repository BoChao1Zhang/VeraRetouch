"""v2seg (2026-08-14) target-template tests: two trailing readout tokens.

CPU only -- loads the tokenizer/processor, never the model weights.

    /home/bc/envs/q3vl_sft/bin/python -m pytest q3vl/train/tests/test_v2seg_tokens.py -q

What is pinned here:
  * ``SPECIAL_TOKENS`` is the 6-tuple and the two new literals are APPENDED, so
    ``<where>``..``</color>`` keep 151669..151672 (q3vl/whereb/attnread.py hard
    codes 151669/151670) and the readouts take 151673/151674;
  * the assistant target is
    ``<where>..</where><color>..</color><seg_where><seg_color><|im_end|>\\n``;
  * both readout tokens are supervised and carry SEG_SEGWHERE / SEG_SEGCOLOR;
  * piecewise tokenisation still equals whole-string tokenisation.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from q3vl.train.collator import IM_END, Sft2SegCollator  # noqa: E402
from q3vl.train.constants import (  # noqa: E402
    COLOR_CLOSE, COLOR_OPEN, IGNORE_INDEX, SEG_COLOR, SEG_COLOR_TOK, SEG_EOS,
    SEG_IGNORE, SEG_SEGCOLOR, SEG_SEGWHERE, SEG_WHERE, SEG_WHERE_TOK,
    SPECIAL_TOKENS, WHERE_CLOSE, WHERE_OPEN,
)
from q3vl.train.tokens import register_special_tokens, verify_single_token  # noqa: E402

MODEL_PATH = os.environ.get("Q3VL_MODEL_PATH", "/home/bc/data/models/Qwen3-VL-4B-Instruct")

#: registration order fixes these; a change here breaks q3vl/whereb/attnread.py.
EXPECTED_IDS = {
    WHERE_OPEN: 151669,
    WHERE_CLOSE: 151670,
    COLOR_OPEN: 151671,
    COLOR_CLOSE: 151672,
    SEG_WHERE_TOK: 151673,
    SEG_COLOR_TOK: 151674,
}


@pytest.fixture(scope="module")
def processor():
    pytest.importorskip("transformers")
    if not Path(MODEL_PATH).exists():
        pytest.skip(f"model not available at {MODEL_PATH}")
    from transformers import AutoProcessor

    p = AutoProcessor.from_pretrained(MODEL_PATH)
    register_special_tokens(p.tokenizer)
    return p


@pytest.fixture(scope="module")
def sample():
    """A synthetic sample: real image geometry, no shard/tar plumbing."""
    pytest.importorskip("PIL")
    from PIL import Image

    from q3vl.train.dataset import Sft2SegSample
    from q3vl.train.imageproc import prepare_image

    image, geom = prepare_image(Image.new("RGB", (768, 512), (120, 90, 60)))
    return Sft2SegSample(
        sample_id="v2seg-synth-0",
        image=image,
        geometry=geom,
        instruction="Warm the light and lift the subject.",
        where_text="The edit covers the standing subject in the left half.",
        color_text="The scene reads flat and cool.\nWarm the global balance.",
        meta={},
    )


class TestSpecialTokenTuple:
    def test_appended_not_inserted(self):
        assert SPECIAL_TOKENS == (
            WHERE_OPEN, WHERE_CLOSE, COLOR_OPEN, COLOR_CLOSE, SEG_WHERE_TOK, SEG_COLOR_TOK,
        )

    def test_six_single_tokens_with_expected_ids(self, processor):
        ids = verify_single_token(processor.tokenizer)
        assert ids == EXPECTED_IDS
        assert len(set(ids.values())) == 6
        # the readouts must not eat into the old ids
        assert ids[WHERE_OPEN] == 151669 and ids[WHERE_CLOSE] == 151670

    def test_registration_is_idempotent(self, processor):
        before = len(processor.tokenizer)
        register_special_tokens(processor.tokenizer)
        assert len(processor.tokenizer) == before == 151675


class TestV2SegTemplate:
    def test_target_text(self):
        assert Sft2SegCollator.build_target_text("W", "C") == (
            f"{WHERE_OPEN}W{WHERE_CLOSE}{COLOR_OPEN}C{COLOR_CLOSE}"
            f"{SEG_WHERE_TOK}{SEG_COLOR_TOK}"
        )

    def test_tail_ids_labels_and_segments(self, processor, sample):
        col = Sft2SegCollator(processor)
        enc = col.encode_one(sample)
        ids, labels, segs = enc["input_ids"], enc["labels"], enc["segment_ids"]

        eos_ids = col._ids(f"{IM_END}\n")
        n_eos = len(eos_ids)
        tail = ids[-(n_eos + 2):]
        assert tail == [EXPECTED_IDS[SEG_WHERE_TOK], EXPECTED_IDS[SEG_COLOR_TOK]] + eos_ids

        # readout tokens are supervised (labels copy the ids, never IGNORE)
        i_sw, i_sc = len(ids) - n_eos - 2, len(ids) - n_eos - 1
        assert labels[i_sw] == EXPECTED_IDS[SEG_WHERE_TOK]
        assert labels[i_sc] == EXPECTED_IDS[SEG_COLOR_TOK]
        assert segs[i_sw] == SEG_SEGWHERE
        assert segs[i_sc] == SEG_SEGCOLOR
        assert enc["n_seg_tail_tokens"] == 2

        # exactly one of each segment id, in template order
        assert segs.count(SEG_SEGWHERE) == 1 and segs.count(SEG_SEGCOLOR) == 1
        last_color = max(i for i, s in enumerate(segs) if s == SEG_COLOR)
        first_eos = min(i for i, s in enumerate(segs) if s == SEG_EOS)
        assert last_color < i_sw < i_sc < first_eos
        # and the prompt is still fully masked
        assert all(
            s == SEG_IGNORE for s, l in zip(segs, labels) if l == IGNORE_INDEX
        )
        assert segs.count(SEG_WHERE) > 0 and segs.count(SEG_COLOR) > 0

    def test_readouts_supervised_even_without_eos_supervision(self, processor, sample):
        enc = Sft2SegCollator(processor, supervise_eos=False).encode_one(sample)
        segs = enc["segment_ids"]
        assert segs.count(SEG_EOS) == 0
        assert segs.count(SEG_SEGWHERE) == 1 and segs.count(SEG_SEGCOLOR) == 1

    def test_concat_equivalence(self, processor, sample):
        col = Sft2SegCollator(processor)
        info = col.check_concat_equivalence(sample)
        assert info["equal"], info

    def test_each_special_token_appears_once_and_is_supervised(self, processor, sample):
        col = Sft2SegCollator(processor)
        enc = col.encode_one(sample)
        ids, labels = enc["input_ids"], enc["labels"]
        for tok, tid in EXPECTED_IDS.items():
            pos = [i for i, t in enumerate(ids) if t == tid]
            assert len(pos) == 1, f"{tok} appears {len(pos)} times"
            assert labels[pos[0]] == tid, f"{tok} is not supervised"


class TestRuntimeAssertion:
    """`assert_v2seg_template` is the "defined but not wired" guard."""

    def test_passes_on_the_real_template(self, processor, sample):
        from q3vl.train.train_sft import assert_v2seg_template

        col = Sft2SegCollator(processor)
        info = assert_v2seg_template(col, sample, verify_single_token(processor.tokenizer))
        assert info["seg_where_pos"] and info["seg_color_pos"]
        assert info["tail_ids"][:2] == [
            EXPECTED_IDS[SEG_WHERE_TOK], EXPECTED_IDS[SEG_COLOR_TOK],
        ]

    def test_fails_when_the_tail_is_missing(self, processor, sample):
        from q3vl.train.train_sft import assert_v2seg_template

        col = Sft2SegCollator(processor)
        real = col.encode_one(sample)
        n = len(real["input_ids"])

        class _Pre2Seg(Sft2SegCollator):
            """Reproduces the pre-v2seg encoding: no readout tokens at all."""

            def encode_one(self, s):
                enc = super().encode_one(s)
                keep = [
                    i for i, sg in enumerate(enc["segment_ids"])
                    if sg not in (SEG_SEGWHERE, SEG_SEGCOLOR)
                ]
                return {
                    **enc,
                    "input_ids": [enc["input_ids"][i] for i in keep],
                    "labels": [enc["labels"][i] for i in keep],
                    "segment_ids": [enc["segment_ids"][i] for i in keep],
                }

        stale = _Pre2Seg(processor)
        assert len(stale.encode_one(sample)["input_ids"]) == n - 2
        with pytest.raises(SystemExit, match="v2seg template assertion FAILED"):
            assert_v2seg_template(stale, sample, verify_single_token(processor.tokenizer))


class TestDiagnosticsRegistry:
    def test_new_segments_registered(self):
        from q3vl.train.diagnostics import _SEGMENTS

        assert dict(_SEGMENTS)[SEG_SEGWHERE] == "segwhere"
        assert dict(_SEGMENTS)[SEG_SEGCOLOR] == "segcolor"

    def test_metric_names(self):
        import torch

        from q3vl.train.diagnostics import SegmentAccumulator

        acc = SegmentAccumulator()
        acc.update(
            torch.tensor([1.0, 2.0]),
            torch.tensor([True, False]),
            torch.tensor([SEG_SEGWHERE, SEG_SEGCOLOR]),
        )
        m = acc.metrics(prefix="train_seg_")
        assert m["train_seg_segwhere_acc"] == 1.0
        assert m["train_seg_segcolor_acc"] == 0.0

    def test_parse_two_segment_tolerates_the_tail(self):
        from q3vl.train.diagnostics import parse_two_segment

        p = parse_two_segment(
            f"{WHERE_OPEN}the sky{WHERE_CLOSE}{COLOR_OPEN}warm it{COLOR_CLOSE}"
            f"{SEG_WHERE_TOK}{SEG_COLOR_TOK}"
        )
        assert p["tags_complete"] and p["order_ok"] and p["seg_tail_ok"]
        assert p["where_body"] == "the sky" and p["color_body"] == "warm it"
        assert p["n_where_open"] == 1 and p["n_color_open"] == 1
        # missing tail is reported, not crashed on
        assert not parse_two_segment(
            f"{WHERE_OPEN}a{WHERE_CLOSE}{COLOR_OPEN}b{COLOR_CLOSE}"
        )["seg_tail_ok"]


class TestSingleCardVariant:
    def _args(self, tmp_path, **over):
        from q3vl.train.args import SFTTrainingArguments

        # use_cpu keeps TrainingArguments.__post_init__ from probing the GPU for
        # bf16 support, so this file stays runnable with CUDA_VISIBLE_DEVICES=""
        # (it must never touch a card). It is not a spec-frozen field.
        over.setdefault("use_cpu", True)
        return SFTTrainingArguments(output_dir=str(tmp_path), **over)

    def test_default_still_requires_two_gpus(self, tmp_path):
        from q3vl.train.args import validate_frozen_hyperparameters

        with pytest.raises(ValueError, match="world_size"):
            validate_frozen_hyperparameters(self._args(tmp_path), world_size=1)

    def test_declared_variant_accepts_one_gpu_4x8(self, tmp_path):
        from q3vl.train.args import validate_frozen_hyperparameters

        args = self._args(
            tmp_path,
            single_card_variant=True,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
        )
        info = validate_frozen_hyperparameters(args, world_size=1)
        assert info["spec_violations"] == []
        assert info["effective_global_batch"] == 32
        assert info["single_card_variant"] is True

    def test_variant_still_pins_global_batch_and_world_size(self, tmp_path):
        from q3vl.train.args import validate_frozen_hyperparameters

        args = self._args(
            tmp_path,
            single_card_variant=True,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
        )
        with pytest.raises(ValueError, match="global batch"):
            validate_frozen_hyperparameters(args, world_size=1)

        args2 = self._args(
            tmp_path,
            single_card_variant=True,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
        )
        with pytest.raises(ValueError, match="world_size"):
            validate_frozen_hyperparameters(args2, world_size=2)

    def test_variant_does_not_relax_frozen_hparams(self, tmp_path):
        from q3vl.train.args import validate_frozen_hyperparameters

        args = self._args(
            tmp_path,
            single_card_variant=True,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            learning_rate=2e-5,
        )
        with pytest.raises(ValueError, match="learning_rate"):
            validate_frozen_hyperparameters(args, world_size=1)


class TestV2SegConfigFile:
    def test_config_matches_the_variant_contract(self):
        yaml = pytest.importorskip("yaml")
        base_p = Path(__file__).resolve().parents[1] / "configs" / "sft_base.yaml"
        v2_p = Path(__file__).resolve().parents[1] / "configs" / "sft_base_v2seg.yaml"
        base = yaml.safe_load(base_p.read_text())
        v2 = yaml.safe_load(v2_p.read_text())

        assert v2["training"]["single_card_variant"] is True
        assert v2["training"]["per_device_train_batch_size"] == 4
        assert v2["training"]["gradient_accumulation_steps"] == 8
        assert v2["training"]["output_dir"] == "/home/bc/data/runs/q3vl_base_sft_v2seg_20260814"
        assert v2["training"]["output_dir"] != base["training"]["output_dir"]

        # everything else is byte-identical to the 2026-08-04 config
        assert v2["model"] == base["model"]
        assert v2["data"] == base["data"]
        changed = {"single_card_variant", "gradient_accumulation_steps", "output_dir"}
        assert set(base["training"]) - set(v2["training"]) == set()
        assert set(v2["training"]) - set(base["training"]) == {"single_card_variant"}
        for k, want in base["training"].items():
            if k in changed:
                continue
            assert v2["training"][k] == want, k
