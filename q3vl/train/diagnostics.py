"""Auxiliary diagnostics (spec 4.4 last bullet, spec 8.3).

Nothing here contributes to the optimised loss. The Where/Color token losses
are recorded only, exactly as spec 4.4 requires ("不改变总 loss 权重").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from .constants import (
    COLOR_CLOSE, COLOR_OPEN, IGNORE_INDEX, LEGACY_SEGMENT_TAGS,
    SEG_COLOR, SEG_EOS, SEG_WHERE, WHERE_CLOSE, WHERE_OPEN,
)

_SEGMENTS = ((SEG_WHERE, "where"), (SEG_COLOR, "color"), (SEG_EOS, "eos"))


@dataclass
class SegmentAccumulator:
    """Sums of token CE and correct-token counts, per diagnostic segment."""

    loss_sum: dict[str, float] = field(default_factory=dict)
    token_count: dict[str, int] = field(default_factory=dict)
    correct: dict[str, int] = field(default_factory=dict)
    n_batches: int = 0

    def reset(self) -> None:
        self.loss_sum.clear()
        self.token_count.clear()
        self.correct.clear()
        self.n_batches = 0

    def update(self, per_token_loss, correct_mask, shifted_segments) -> None:
        self.n_batches += 1
        for seg_id, name in _SEGMENTS:
            mask = shifted_segments == seg_id
            n = int(mask.sum())
            if n == 0:
                continue
            self.loss_sum[name] = self.loss_sum.get(name, 0.0) + float(per_token_loss[mask].sum())
            self.token_count[name] = self.token_count.get(name, 0) + n
            self.correct[name] = self.correct.get(name, 0) + int(correct_mask[mask].sum())

    def metrics(self, prefix: str = "") -> dict[str, float]:
        out: dict[str, float] = {}
        total_loss, total_tok, total_correct = 0.0, 0, 0
        for _, name in _SEGMENTS:
            n = self.token_count.get(name, 0)
            if n == 0:
                continue
            out[f"{prefix}{name}_loss"] = self.loss_sum[name] / n
            out[f"{prefix}{name}_acc"] = self.correct[name] / n
            out[f"{prefix}{name}_tokens"] = float(n)
            total_loss += self.loss_sum[name]
            total_tok += n
            total_correct += self.correct[name]
        if total_tok:
            out[f"{prefix}assistant_loss"] = total_loss / total_tok
            out[f"{prefix}assistant_acc"] = total_correct / total_tok
        return out


@torch.no_grad()
def segment_token_stats(logits: torch.Tensor, labels: torch.Tensor, segment_ids: torch.Tensor):
    """Per-token CE and correctness on the causal-shifted positions.

    ``logits[:, :-1]`` predicts ``labels[:, 1:]``; ``segment_ids`` is aligned
    with ``labels``, so it is shifted the same way.
    """
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    shift_segments = segment_ids[:, 1:]
    valid = shift_labels != IGNORE_INDEX

    flat_logits = shift_logits[valid].float()
    flat_labels = shift_labels[valid]
    if flat_labels.numel() == 0:
        empty = torch.zeros(0, device=logits.device)
        return empty, empty.bool(), shift_segments[valid]
    per_token = F.cross_entropy(flat_logits, flat_labels, reduction="none")
    correct = flat_logits.argmax(dim=-1) == flat_labels
    return per_token, correct, shift_segments[valid]


# --- spec 8.3 generation diagnostics --------------------------------------

_LEGACY_RE = re.compile("|".join(re.escape(t) for t in LEGACY_SEGMENT_TAGS))


def parse_two_segment(text: str) -> dict[str, Any]:
    """Structural check on one generated assistant string."""
    idx = {t: text.find(t) for t in (WHERE_OPEN, WHERE_CLOSE, COLOR_OPEN, COLOR_CLOSE)}
    present = {t: i >= 0 for t, i in idx.items()}
    complete = all(present.values())
    order_ok = complete and (
        idx[WHERE_OPEN] < idx[WHERE_CLOSE] < idx[COLOR_OPEN] < idx[COLOR_CLOSE]
    )
    where_body = color_body = ""
    if order_ok:
        where_body = text[idx[WHERE_OPEN] + len(WHERE_OPEN): idx[WHERE_CLOSE]].strip()
        color_body = text[idx[COLOR_OPEN] + len(COLOR_OPEN): idx[COLOR_CLOSE]].strip()
    return {
        "tags_complete": complete,
        "order_ok": order_ok,
        "where_nonempty": bool(where_body),
        "color_nonempty": bool(color_body),
        "where_in_color": bool(where_body) and bool(color_body) and where_body in color_body,
        "legacy_tag_leak": bool(_LEGACY_RE.search(text)),
        "n_where_open": text.count(WHERE_OPEN),
        "n_color_open": text.count(COLOR_OPEN),
        "where_body": where_body,
        "color_body": color_body,
    }


def aggregate_generation_diagnostics(parsed: list[dict[str, Any]], prefix: str = "eval_gen_") -> dict[str, float]:
    n = len(parsed)
    if n == 0:
        return {}
    def rate(key: str) -> float:
        return sum(1 for p in parsed if p[key]) / n
    return {
        f"{prefix}n": float(n),
        f"{prefix}tag_completeness": rate("tags_complete"),
        f"{prefix}order_accuracy": rate("order_ok"),
        f"{prefix}where_nonempty_rate": rate("where_nonempty"),
        f"{prefix}color_nonempty_rate": rate("color_nonempty"),
        f"{prefix}where_copied_into_color_rate": rate("where_in_color"),
        f"{prefix}legacy_tag_leak_rate": rate("legacy_tag_leak"),
        f"{prefix}duplicate_where_rate": sum(1 for p in parsed if p["n_where_open"] > 1) / n,
        f"{prefix}duplicate_color_rate": sum(1 for p in parsed if p["n_color_open"] > 1) / n,
    }
