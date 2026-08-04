"""Arm B parameter freezing (spec 2.2).

The only structural boundary pre-registered by this experiment:

    frozen    : the 24 vision transformer blocks + vision patch embedding
                (+ the vision position embedding, which belongs to the vision
                tower and not to any merger -- see NOTES.md)
    trainable : model.visual.merger, model.visual.deepstack_merger_list (3),
                the whole language model, and the (tied) token embeddings.

This is *not* LoRA/PEFT: every parameter in the trainable set is optimised.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn

from .constants import VISUAL_ROOT_PREFIX, VISUAL_TRAINABLE_PREFIXES


class FreezeBoundaryError(RuntimeError):
    """Raised when the observed module tree does not match the Arm B contract."""


def is_trainable_param(name: str) -> bool:
    """Arm B rule, expressed once, by parameter name."""
    if name.startswith(VISUAL_ROOT_PREFIX):
        return name.startswith(VISUAL_TRAINABLE_PREFIXES)
    # language model, token embeddings, final norm, (tied) lm_head
    return True


def full_numel(param) -> int:
    """Element count that survives ZeRO-3 partitioning.

    ``from_pretrained`` runs inside ``deepspeed.zero.Init`` whenever a ZeRO-3
    config is active, which is the case for the real two-GPU launch. Every
    parameter is then partitioned and *freed*: ``param.numel()`` is ``0`` and
    ``param.shape`` is ``torch.Size([0])`` until the parameter is gathered, while
    ``ds_numel``/``ds_shape`` carry the real values. Counting ``numel()`` here
    made the whole spec 9.2/9.3 audit report zeros under the real launch -- and
    silently pass, because "vision blocks have 0 trainable params" is true when
    everything is 0. Measured on 2xH100: ``sum(p.numel())=0`` vs
    ``sum(p.ds_numel)=4,437,815,808``.
    """
    n = getattr(param, "ds_numel", None)
    if n is not None:
        return int(n)
    return int(param.numel())


def full_shape(param) -> tuple[int, ...]:
    """``param.shape`` unless ZeRO-3 has partitioned it away (then ``ds_shape``)."""
    ds = getattr(param, "ds_shape", None)
    if ds is not None:
        return tuple(int(x) for x in ds)
    return tuple(int(x) for x in param.shape)


def _group_key(name: str) -> str:
    return re.sub(r"\.\d+\.", ".{i}.", name)


@dataclass
class FreezeReport:
    trainable_params: int = 0
    frozen_params: int = 0
    trainable_names: list[str] = field(default_factory=list)
    frozen_names: list[str] = field(default_factory=list)
    trainable_groups: "OrderedDict[str, dict[str, int]]" = field(default_factory=OrderedDict)
    frozen_groups: "OrderedDict[str, dict[str, int]]" = field(default_factory=OrderedDict)
    subtree_counts: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def total_params(self) -> int:
        return self.trainable_params + self.frozen_params

    @property
    def trainable_ratio(self) -> float:
        return self.trainable_params / max(1, self.total_params)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_params": self.total_params,
            "trainable_params": self.trainable_params,
            "frozen_params": self.frozen_params,
            "trainable_ratio": self.trainable_ratio,
            "trainable_groups": self.trainable_groups,
            "frozen_groups": self.frozen_groups,
            "subtree_counts": self.subtree_counts,
            "n_trainable_tensors": len(self.trainable_names),
            "n_frozen_tensors": len(self.frozen_names),
        }


# subtrees we always report separately so the merger/vision split is auditable
_SUBTREES = (
    ("vision_blocks", "model.visual.blocks."),
    ("vision_patch_embed", "model.visual.patch_embed."),
    ("vision_pos_embed", "model.visual.pos_embed"),
    ("vision_merger", "model.visual.merger."),
    ("vision_deepstack_mergers", "model.visual.deepstack_merger_list."),
    ("language_embed_tokens", "model.language_model.embed_tokens"),
    ("language_layers", "model.language_model.layers."),
    ("language_norm", "model.language_model.norm"),
    ("lm_head", "lm_head."),
)


def apply_arm_b_freeze(model: nn.Module) -> FreezeReport:
    """Set ``requires_grad`` per the Arm B contract and return an audit report."""
    report = FreezeReport()
    subtree: dict[str, dict[str, int]] = {
        name: {"params": 0, "tensors": 0, "trainable_params": 0, "frozen_params": 0}
        for name, _ in _SUBTREES
    }

    for name, param in model.named_parameters():
        trainable = is_trainable_param(name)
        param.requires_grad_(trainable)
        n = full_numel(param)
        bucket = report.trainable_groups if trainable else report.frozen_groups
        key = _group_key(name)
        entry = bucket.setdefault(key, {"tensors": 0, "params": 0})
        entry["tensors"] += 1
        entry["params"] += n
        if trainable:
            report.trainable_params += n
            report.trainable_names.append(name)
        else:
            report.frozen_params += n
            report.frozen_names.append(name)
        for sub_name, prefix in _SUBTREES:
            if name.startswith(prefix):
                s = subtree[sub_name]
                s["params"] += n
                s["tensors"] += 1
                s["trainable_params" if trainable else "frozen_params"] += n
                break

    report.subtree_counts = subtree
    assert_freeze_boundary(model, report)
    return report


def assert_freeze_boundary(model: nn.Module, report: FreezeReport | None = None) -> None:
    """Structural assertions that catch a silently-changed module tree."""
    names = [n for n, _ in model.named_parameters()]

    n_blocks = len({n.split(".")[3] for n in names if n.startswith("model.visual.blocks.")})
    if n_blocks != 24:
        raise FreezeBoundaryError(f"expected 24 vision blocks, found {n_blocks}")

    n_deepstack = len(
        {n.split(".")[3] for n in names if n.startswith("model.visual.deepstack_merger_list.")}
    )
    if n_deepstack != 3:
        raise FreezeBoundaryError(f"expected 3 deepstack mergers, found {n_deepstack}")

    if not any(n.startswith("model.visual.merger.") for n in names):
        raise FreezeBoundaryError("main model.visual.merger parameters not found")

    for name, param in model.named_parameters():
        want = is_trainable_param(name)
        if param.requires_grad != want:
            raise FreezeBoundaryError(
                f"requires_grad mismatch on {name}: got {param.requires_grad}, want {want}"
            )

    if report is not None:
        st = report.subtree_counts
        # A report full of zeros satisfies every "must be 0" assertion below, so
        # the positive checks come first. This is exactly what happened under
        # ZeRO-3 before `full_numel` existed: all counts were 0 and the audit
        # passed while proving nothing.
        if report.total_params <= 0:
            raise FreezeBoundaryError(
                "freeze report counted 0 parameters; under ZeRO-3 `param.numel()` is the "
                "local shard size (0 until gathered) -- counts must come from `ds_numel`"
            )
        for sub, field_name in (
            ("vision_blocks", "frozen_params"),
            ("vision_patch_embed", "frozen_params"),
            ("vision_merger", "trainable_params"),
            ("vision_deepstack_mergers", "trainable_params"),
            ("language_layers", "trainable_params"),
            ("language_embed_tokens", "trainable_params"),
        ):
            if st[sub][field_name] <= 0:
                raise FreezeBoundaryError(
                    f"subtree {sub!r} reports {field_name}=0; the Arm B boundary cannot be "
                    f"verified from a vacuous count"
                )
        if st["vision_blocks"]["trainable_params"] != 0:
            raise FreezeBoundaryError("vision blocks must be fully frozen")
        if st["vision_patch_embed"]["trainable_params"] != 0:
            raise FreezeBoundaryError("vision patch embedding must be frozen")
        if st["vision_merger"]["frozen_params"] != 0:
            raise FreezeBoundaryError("main merger must be fully trainable")
        if st["vision_deepstack_mergers"]["frozen_params"] != 0:
            raise FreezeBoundaryError("deepstack mergers must be fully trainable")
        if st["language_layers"]["frozen_params"] != 0:
            raise FreezeBoundaryError("language layers must be fully trainable")
        if st["language_embed_tokens"]["frozen_params"] != 0:
            raise FreezeBoundaryError("token embeddings must be trainable (new special tokens)")


def format_freeze_table(report: FreezeReport) -> str:
    """Human-readable frozen/trainable listing for the preflight report."""
    lines = []
    lines.append(f"total params      : {report.total_params:,}")
    lines.append(
        f"trainable params  : {report.trainable_params:,} "
        f"({100 * report.trainable_ratio:.4f}%)"
    )
    lines.append(
        f"frozen params     : {report.frozen_params:,} "
        f"({100 * (1 - report.trainable_ratio):.4f}%)"
    )
    lines.append("")
    lines.append("| subtree | tensors | params | trainable | frozen |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, _ in _SUBTREES:
        s = report.subtree_counts[name]
        if s["tensors"] == 0:
            continue
        lines.append(
            f"| `{name}` | {s['tensors']} | {s['params']:,} | "
            f"{s['trainable_params']:,} | {s['frozen_params']:,} |"
        )
    lines.append("")
    lines.append("TRAINABLE parameter groups:")
    for key, v in report.trainable_groups.items():
        lines.append(f"  + {key}  [x{v['tensors']}]  {v['params']:,}")
    lines.append("FROZEN parameter groups:")
    for key, v in report.frozen_groups.items():
        lines.append(f"  - {key}  [x{v['tensors']}]  {v['params']:,}")
    return "\n".join(lines)
