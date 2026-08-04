"""Pass 3: the full-sequence token length of every candidate sample (spec 6).

The length must be the length of *the sequence the trainer will actually build*,
so this module does not re-implement prompt assembly: it imports
``q3vl.train.collator.Sft2SegCollator`` and calls the same methods the collator
calls.  If the training side changes its template, these numbers change with it.

    total = |prompt with the image placeholder expanded to N visual tokens|
          + |<where>...</where>| + |<color>...</color>| + |<|im_end|>\\n|

The prompt is tokenised once per *distinct instruction* with a single
``<|image_pad|>`` and the visual count is added arithmetically, which is exact
because ``<|image_pad|>`` is a special token: repeating it cannot create or
destroy a BPE merge at either boundary.  ``verify_identity`` asserts that
equivalence on real samples rather than assuming it.
"""

from __future__ import annotations

from typing import Any, Iterable

from q3vl.train.collator import IM_END, Sft2SegCollator
from q3vl.train.constants import COLOR_CLOSE, COLOR_OPEN, WHERE_CLOSE, WHERE_OPEN
from q3vl.train.tokens import register_special_tokens

BATCH = 512


def load_processor(model_dir: str):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_dir)
    ids = register_special_tokens(processor.tokenizer)
    return processor, ids


class LengthCalculator:
    def __init__(self, processor) -> None:
        self.collator = Sft2SegCollator(processor, collect_stats=False)
        self.tokenizer = processor.tokenizer
        self.eos_len = len(self._ids(f"{IM_END}\n"))
        # cost of the template + one <|image_pad|>; the pad token is subtracted
        # so callers add the real visual token count.
        self._empty_prompt_len = len(self._ids(self.collator.build_prompt_text(""))) - 1

    def _ids(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def _batch_len(self, texts: list[str]) -> list[int]:
        enc = self.tokenizer(texts, add_special_tokens=False)["input_ids"]
        return [len(x) for x in enc]

    @property
    def template_overhead(self) -> int:
        """Tokens contributed by the chat template alone (no image, no text)."""
        return self._empty_prompt_len + self.eos_len

    def measure(self, rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = list(rows)
        out: list[dict[str, Any]] = []
        for start in range(0, len(rows), BATCH):
            chunk = rows[start:start + BATCH]
            prompts = [self.collator.build_prompt_text(r["instruction"]) for r in chunk]
            wheres = [f"{WHERE_OPEN}{r['where']}{WHERE_CLOSE}" for r in chunk]
            colors = [f"{COLOR_OPEN}{r['color']}{COLOR_CLOSE}" for r in chunk]
            p_len = self._batch_len(prompts)
            w_len = self._batch_len(wheres)
            c_len = self._batch_len(colors)
            for row, pl, wl, cl in zip(chunk, p_len, w_len, c_len):
                prompt_tokens = pl - 1 + row["vision_tokens"]
                out.append({
                    "sft_id": row["sft_id"],
                    "prompt_tokens": prompt_tokens,
                    "instruction_tokens": pl - 1 - self._empty_prompt_len,
                    "where_tokens": wl,
                    "color_tokens": cl,
                    "eos_tokens": self.eos_len,
                    "vision_tokens": row["vision_tokens"],
                    "total_tokens": prompt_tokens + wl + cl + self.eos_len,
                })
        return out

    def verify_identity(self, row: dict[str, Any]) -> dict[str, Any]:
        """Expanded-placeholder tokenisation == arithmetic shortcut, per sample."""
        text = self.collator.build_prompt_text(row["instruction"])
        expanded = self.collator._expand_image_placeholder(text, row["vision_tokens"])
        exact = len(self._ids(expanded))
        shortcut = len(self._ids(text)) - 1 + row["vision_tokens"]
        target = (len(self._ids(f"{WHERE_OPEN}{row['where']}{WHERE_CLOSE}"))
                  + len(self._ids(f"{COLOR_OPEN}{row['color']}{COLOR_CLOSE}"))
                  + self.eos_len)
        whole = len(self._ids(
            expanded + f"{WHERE_OPEN}{row['where']}{WHERE_CLOSE}"
            f"{COLOR_OPEN}{row['color']}{COLOR_CLOSE}{IM_END}\n"
        ))
        return {
            "sft_id": row["sft_id"],
            "prompt_exact": exact,
            "prompt_shortcut": shortcut,
            "prompt_equal": exact == shortcut,
            "whole_string": whole,
            "piecewise": exact + target,
            "concat_equal": whole == exact + target,
        }
