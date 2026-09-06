# 源自 experiments/prs/EPR-051_masked-restore-production/stage0/sprf/vlmsft/q3vl_text.py @ 2026-09-06（原件未入 git；原 sha256 见 veraretouch_sprf/PROVENANCE.md），逐字复制/仅改导入与路径解析（EPR-052）。
"""EPR-051 / SFT+ADAPT -- Qwen3-VL prompt/target construction and segment spans.

The 6 span boundaries are built BY CONSTRUCTION (segment bodies tokenised one at
a time and their lengths accumulated), never by searching a flat id list for the
stage tokens.  `assert_span_parity` then checks the concatenation reproduces the
one-shot tokenisation of the same text, so a BPE merge across a boundary cannot
silently shift a span (EPR-033's A_reply discipline, over integers).

`include_stage_token` (config, default False) decides whether the closing
<vr_stage_m> position is inside span m.  Excluded by default: the coordinator's
ruling allows either, and the span is meant to be the段内容 itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
from veraretouch_sprf import _paths as _P  # EPR-052：替代原 sys.path 注入块
_P.ensure_sys_path()

import torch  # noqa: E402

from veraretouch_sprf.data import cot_text as C  # noqa: E402
from veraretouch_sprf.models.vlm.q3vl_common import IGNORE_INDEX, die  # noqa: E402

IM_START, IM_END = "<|im_start|>", "<|im_end|>"


def prompt_text(instruction: str) -> str:
    """conv = one user turn carrying the image, then the assistant header.

    Written literally rather than through apply_chat_template so the exact
    string is frozen in this file and hashed with it.  It matches the shipped
    Qwen3-VL chat template for a single image-bearing user turn with no system
    message and add_generation_prompt=True (chat_template.json).
    """
    return (f"{IM_START}user\n"
            f"<|vision_start|><|image_pad|><|vision_end|>{instruction}{IM_END}\n"
            f"{IM_START}assistant\n")


def encode_prompt(processor, image, instruction: str):
    """-> dict(input_ids (1,P), pixel_values, image_grid_thw)."""
    return processor(text=[prompt_text(instruction)], images=[image],
                     do_resize=False, return_tensors="pt")


def encode_target(tokenizer, record: dict, include_stage_token: bool = False):
    """-> (target_ids list, spans list[(s,e)] relative to the target start).

    target = seg_1 <vr_stage_1> "\\n" seg_2 <vr_stage_2> "\\n" ... seg_6
             <vr_stage_6> <|im_end|>
    which is exactly cot_text.target_text(record) + IM_END.
    """
    segs = C.target_segments(record)
    ids: list[int] = []
    spans: list[tuple[int, int]] = []
    for m, seg in enumerate(segs, 1):
        body = tokenizer(seg, add_special_tokens=False).input_ids
        tok = tokenizer(C.STAGE_TOKENS[m - 1], add_special_tokens=False).input_ids
        if len(tok) != 1:
            die(f"stage token {m} is not atomic: {tok}")
        s = len(ids)
        ids.extend(body)
        e = len(ids)
        ids.extend(tok)
        if include_stage_token:
            e = len(ids)
        if e <= s:
            die(f"segment {m} produced an empty span")
        spans.append((s, e))
        if m < 6:
            ids.extend(tokenizer("\n", add_special_tokens=False).input_ids)
    ids.extend(tokenizer(IM_END, add_special_tokens=False).input_ids)
    return ids, spans


def assert_span_parity(tokenizer, record: dict, ids: list[int]) -> None:
    """The piecewise ids must equal the one-shot tokenisation of the same text."""
    whole = tokenizer(C.target_text(record) + IM_END,
                      add_special_tokens=False).input_ids
    if list(ids) != list(whole):
        die(f"A_span FAILED: piecewise target ({len(ids)} tok) != one-shot "
            f"({len(whole)} tok); a BPE merge crossed a segment boundary")


def build_example(processor, image, instruction: str, record: dict,
                  include_stage_token: bool = False, check_parity: bool = True):
    """-> dict(input_ids, labels, spans, pixel_values, image_grid_thw, n_prompt)."""
    tk = processor.tokenizer
    enc = encode_prompt(processor, image, instruction)
    p_ids = enc["input_ids"][0].tolist()
    t_ids, spans = encode_target(tk, record, include_stage_token)
    if check_parity:
        assert_span_parity(tk, record, t_ids)
    n_p = len(p_ids)
    input_ids = p_ids + t_ids
    labels = [IGNORE_INDEX] * n_p + list(t_ids)
    abs_spans = [(n_p + s, n_p + e) for s, e in spans]
    return dict(input_ids=torch.tensor(input_ids),
                labels=torch.tensor(labels),
                spans=abs_spans, n_prompt=n_p,
                pixel_values=enc["pixel_values"],
                image_grid_thw=enc["image_grid_thw"])


def spans_from_generated(tokenizer, seq_ids, stage_ids, include_stage_token=False):
    """R2: locate the 6 segment spans inside a GENERATED id sequence.

    Span m runs from just after stage token m-1 (or 0) up to stage token m.
    A stage token that never appears, or appears out of order, is reported --
    never repaired.
    """
    seq = list(seq_ids)
    pos = []
    missing = []
    cur = 0
    for m, sid in enumerate(stage_ids, 1):
        try:
            i = seq.index(sid, cur)
        except ValueError:
            missing.append(m)
            pos.append(None)
            continue
        pos.append(i)
        cur = i + 1
    spans = []
    prev_end = 0
    for m, i in enumerate(pos, 1):
        if i is None:
            spans.append(None)
            continue
        s, e = prev_end, (i + 1 if include_stage_token else i)
        if e <= s:
            spans.append(None)
            missing.append(m)
        else:
            spans.append((s, e))
        prev_end = i + 1
    return spans, missing


def span_pool(hidden: torch.Tensor, spans) -> torch.Tensor:
    """(T, H) fp32 hidden -> (6, H) mean over each span; None span -> zeros."""
    out = []
    for sp in spans:
        if sp is None:
            out.append(torch.zeros(hidden.shape[-1], device=hidden.device,
                                   dtype=hidden.dtype))
        else:
            s, e = sp
            out.append(hidden[s:e].mean(dim=0))
    return torch.stack(out)
