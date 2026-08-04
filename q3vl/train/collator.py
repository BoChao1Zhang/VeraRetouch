"""Prompt assembly, image-placeholder expansion and loss masking (spec 4.3-4.4, 5, 6).

Sequence layout, built by concatenating *separately tokenised* pieces so the
label boundaries are exact rather than recovered by string search:

    [ chat template system+user turn, with <|image_pad|> expanded to N tokens ]
    [ <|im_start|>assistant\\n                                                 ]   <- ignored
    [ <where> ... </where>                                                     ]   <- supervised, seg=1
    [ <color> ... </color>                                                     ]   <- supervised, seg=2
    [ <|im_end|>\\n                                                            ]   <- supervised, seg=3

Everything before the first ``<where>`` token carries ``IGNORE_INDEX``: system,
user text, chat template scaffolding and every image placeholder token.

The trailing ``<|im_end|>`` is supervised by default. Spec 4.4 names the two
segments explicitly and is silent on the stop token; leaving it unsupervised is
a well-known way to get a model that never terminates, so it is included and
tracked as its own diagnostic segment (``supervise_eos`` turns it off).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import torch

from .constants import (
    COLOR_CLOSE, COLOR_OPEN, IGNORE_INDEX, MODEL_MAX_LENGTH,
    SEG_COLOR, SEG_EOS, SEG_IGNORE, SEG_WHERE, WHERE_CLOSE, WHERE_OPEN,
)
from .imageproc import assert_grid_matches

DEFAULT_SYSTEM_PROMPT: str | None = None
IM_END = "<|im_end|>"


class SequenceTooLong(RuntimeError):
    def __init__(self, sample_id: str, length: int, limit: int):
        super().__init__(
            f"sample {sample_id}: tokenised length {length} > model_max_length {limit}. "
            f"Spec 6 requires such samples to be filtered upstream into the rejection "
            f"report; truncating an assistant target is forbidden."
        )
        self.sample_id = sample_id
        self.length = length
        self.limit = limit


@dataclass
class CollatorStats:
    n_samples: int = 0
    n_visual_tokens: Counter = field(default_factory=Counter)
    seq_lengths: list[int] = field(default_factory=list)
    where_lengths: list[int] = field(default_factory=list)
    color_lengths: list[int] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        def q(xs, p):
            if not xs:
                return None
            s = sorted(xs)
            return s[min(len(s) - 1, int(p * (len(s) - 1)))]

        return {
            "n_samples": self.n_samples,
            "seq_len": {
                "min": min(self.seq_lengths, default=None),
                "p50": q(self.seq_lengths, 0.50),
                "p95": q(self.seq_lengths, 0.95),
                "max": max(self.seq_lengths, default=None),
            },
            "where_tokens": {"p50": q(self.where_lengths, 0.5), "max": max(self.where_lengths, default=None)},
            "color_tokens": {"p50": q(self.color_lengths, 0.5), "max": max(self.color_lengths, default=None)},
            "visual_token_hist": dict(sorted(self.n_visual_tokens.items())),
        }


class Sft2SegCollator:
    """Turns :class:`~q3vl.train.dataset.Sft2SegSample` batches into model inputs."""

    def __init__(
        self,
        processor,
        max_length: int = MODEL_MAX_LENGTH,
        system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
        supervise_eos: bool = True,
        collect_stats: bool = False,
        assert_geometry: bool = True,
    ):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt
        self.supervise_eos = supervise_eos
        self.assert_geometry = assert_geometry
        self.stats = CollatorStats() if collect_stats else None

        self.image_token = processor.image_token
        self.merge_length = processor.image_processor.merge_size ** 2
        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.tokenizer.eos_token_id

    # -- text ---------------------------------------------------------------
    def build_prompt_text(self, instruction: str) -> str:
        content = [{"type": "image"}, {"type": "text", "text": instruction}]
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": content})
        return self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

    @staticmethod
    def build_target_text(where_text: str, color_text: str) -> str:
        return (
            f"{WHERE_OPEN}{where_text}{WHERE_CLOSE}"
            f"{COLOR_OPEN}{color_text}{COLOR_CLOSE}"
        )

    def _expand_image_placeholder(self, prompt_text: str, n_visual_tokens: int) -> str:
        if prompt_text.count(self.image_token) != 1:
            raise RuntimeError(
                f"expected exactly one {self.image_token} in the prompt, "
                f"found {prompt_text.count(self.image_token)}"
            )
        return prompt_text.replace(self.image_token, self.image_token * n_visual_tokens)

    def _ids(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def encode_one(self, sample) -> dict[str, Any]:
        """Tokenise one sample into ids / labels / segment ids (no padding)."""
        geom = sample.geometry
        prompt_text = self.build_prompt_text(sample.instruction)
        prompt_ids = self._ids(self._expand_image_placeholder(prompt_text, geom.n_visual_tokens))

        where_ids = self._ids(f"{WHERE_OPEN}{sample.where_text}{WHERE_CLOSE}")
        color_ids = self._ids(f"{COLOR_OPEN}{sample.color_text}{COLOR_CLOSE}")
        eos_ids = self._ids(f"{IM_END}\n")

        input_ids = prompt_ids + where_ids + color_ids + eos_ids
        labels = (
            [IGNORE_INDEX] * len(prompt_ids)
            + where_ids
            + color_ids
            + (eos_ids if self.supervise_eos else [IGNORE_INDEX] * len(eos_ids))
        )
        segments = (
            [SEG_IGNORE] * len(prompt_ids)
            + [SEG_WHERE] * len(where_ids)
            + [SEG_COLOR] * len(color_ids)
            + [SEG_EOS if self.supervise_eos else SEG_IGNORE] * len(eos_ids)
        )

        if len(input_ids) > self.max_length:
            raise SequenceTooLong(sample.sample_id, len(input_ids), self.max_length)

        n_img = sum(1 for i in prompt_ids if i == self.processor.image_token_id)
        if n_img != geom.n_visual_tokens:
            raise RuntimeError(
                f"sample {sample.sample_id}: {n_img} image placeholder tokens in the prompt "
                f"but geometry says {geom.n_visual_tokens}"
            )
        if self.stats is not None:
            self.stats.n_samples += 1
            self.stats.n_visual_tokens[geom.n_visual_tokens] += 1
            self.stats.seq_lengths.append(len(input_ids))
            self.stats.where_lengths.append(len(where_ids))
            self.stats.color_lengths.append(len(color_ids))
        return {
            "input_ids": input_ids,
            "labels": labels,
            "segment_ids": segments,
            "n_prompt_tokens": len(prompt_ids),
            "n_where_tokens": len(where_ids),
            "n_color_tokens": len(color_ids),
        }

    # -- batch --------------------------------------------------------------
    def __call__(self, samples: list) -> dict[str, torch.Tensor]:
        encoded = [self.encode_one(s) for s in samples]
        max_len = max(len(e["input_ids"]) for e in encoded)

        input_ids, labels, attn, segs = [], [], [], []
        for e in encoded:
            pad = max_len - len(e["input_ids"])
            input_ids.append(e["input_ids"] + [self.pad_token_id] * pad)
            labels.append(e["labels"] + [IGNORE_INDEX] * pad)
            attn.append([1] * len(e["input_ids"]) + [0] * pad)
            segs.append(e["segment_ids"] + [SEG_IGNORE] * pad)

        images = [s.image for s in samples]
        image_inputs = self.processor.image_processor(
            images=images, do_resize=False, return_tensors="pt"
        )
        if self.assert_geometry:
            for s, grid in zip(samples, image_inputs["image_grid_thw"]):
                assert_grid_matches(s.geometry, grid)

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "pixel_values": image_inputs["pixel_values"],
            "image_grid_thw": image_inputs["image_grid_thw"],
            "segment_ids": torch.tensor(segs, dtype=torch.long),
        }
        return batch

    # -- integrity check used by preflight ----------------------------------
    def check_concat_equivalence(self, sample) -> dict[str, Any]:
        """Piecewise tokenisation must equal whole-string tokenisation.

        BPE can merge across a naive concatenation boundary; because every
        boundary here is either a special token or the chat template's newline
        this should hold exactly, and the preflight asserts it rather than
        assuming it.
        """
        geom = sample.geometry
        prompt_text = self._expand_image_placeholder(
            self.build_prompt_text(sample.instruction), geom.n_visual_tokens
        )
        target_text = self.build_target_text(sample.where_text, sample.color_text) + f"{IM_END}\n"
        whole = self._ids(prompt_text + target_text)
        piecewise = self.encode_one(sample)["input_ids"]
        ok = whole == piecewise
        return {
            "equal": ok,
            "len_whole": len(whole),
            "len_piecewise": len(piecewise),
            "first_divergence": next(
                (i for i, (a, b) in enumerate(zip(whole, piecewise)) if a != b), None
            ),
        }
