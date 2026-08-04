"""The one forward contract that produces ``F_pre`` and ``H_where``.

Protocol 5.1 says ``Q_where`` reads "all ``<where>...</where>`` token hidden
states" and the merger-pre ``F_pre``; protocol 2.3 says not to run a second
visual forward if the model is running anyway.  Both come out of **one** frozen
forward here.

Two facts verified on 2026-08-05 against the installed transformers 4.57.1 and
the local Qwen3-VL-4B-Instruct (see NOTES V-B1/V-B2):

1. ``out.hidden_states`` has ``num_hidden_layers + 1`` entries and
   ``hidden_states[-1]`` is the last decoder layer output **before** the final
   RMSNorm: ``lm_head(norm(hidden_states[-1])) == logits`` while
   ``lm_head(hidden_states[-1]) != logits``.  ``H_where`` therefore has to say
   which of the two it means; this module takes the post-norm state (the one the
   lm_head sees) and :data:`q3vl.whereb.config.WHERE_HIDDEN_FINAL_NORM` records
   that choice.
2. Attention is causal, so the hidden states at ``<where>`` positions are
   **bit-identical** whether or not a ``<color>`` segment follows.  That is the
   numerical half of the protocol 14.8 proof that ``Q_where`` cannot read
   ``H_color``; the structural half is that no signature in this package
   accepts it.

Teacher and generated context go through *this same function*.  They differ only
in the token ids of the ``<where>`` span, which is what makes "the generated
hidden states use the same layer/position contract as the teacher ones" a
statement that cannot quietly become false.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import torch

from q3vl.train.constants import IGNORE_INDEX  # noqa: F401  (documents the label space)
from q3vl.where.fpre import FPreHook

from .config import TEXT_HIDDEN, WHERE_HIDDEN_FINAL_NORM, WHERE_HIDDEN_LAYER

__all__ = ["EncodeItem", "EncodeResult", "LastLayerHook", "FrozenVLM",
           "resolve_visual", "resolve_language_model"]


def resolve_visual(model: torch.nn.Module) -> torch.nn.Module:
    """``Qwen3VLForConditionalGeneration.model.visual`` (verified attribute path)."""
    for path in (("model", "visual"), ("visual",)):
        obj: Any = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "blocks"):
            return obj
    raise AttributeError(
        "cannot find the vision tower: expected model.model.visual with .blocks"
    )


def resolve_language_model(model: torch.nn.Module) -> torch.nn.Module:
    """``...model.language_model`` (has ``.layers`` and the final ``.norm``)."""
    for path in (("model", "language_model"), ("language_model",), ("model",)):
        obj: Any = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "layers") and hasattr(obj, "norm"):
            return obj
    raise AttributeError(
        "cannot find the text decoder: expected model.model.language_model with "
        ".layers and .norm"
    )


class LastLayerHook:
    """Captures one decoder layer's output without materialising all 37.

    ``output_hidden_states=True`` allocates ``(L+1, B, T, 2560)``; at batch 4 and
    850 tokens that is ~0.6 GiB of bf16 per forward, all but one slice of it
    dead.  ``tests/test_hiddens.py`` asserts this hook reproduces
    ``out.hidden_states[layer]`` exactly.
    """

    def __init__(self, lm: torch.nn.Module, layer: int = WHERE_HIDDEN_LAYER):
        self.lm = lm
        self.layer_index = layer
        self.layer = lm.layers[layer]
        self.captured: torch.Tensor | None = None
        self._handle = None

    def _fn(self, _m, _i, output):
        self.captured = output[0] if isinstance(output, tuple) else output

    @contextmanager
    def attached(self) -> Iterator["LastLayerHook"]:
        self.captured = None
        self._handle = self.layer.register_forward_hook(self._fn)
        try:
            yield self
        finally:
            self._handle.remove()
            self._handle = None


@dataclass
class EncodeItem:
    sample_id: str
    image: Any                      # PIL image, already spec-5 sized
    prompt_ids: Sequence[int]
    where_ids: Sequence[int] = ()   # empty = null context


@dataclass
class EncodeResult:
    sample_id: str
    f_pre: torch.Tensor             # (grid_h, grid_w, 1024)
    h_where: torch.Tensor           # (T, 2560); T may be 0 for the null context
    grid_h: int
    grid_w: int
    n_prompt_tokens: int
    meta: dict[str, Any] = field(default_factory=dict)


class FrozenVLM:
    """One frozen forward -> ``F_pre`` and ``H_where``.  Never builds gradients."""

    def __init__(
        self,
        model: torch.nn.Module,
        processor,
        device: str | torch.device = "cuda",
        *,
        layer: int = WHERE_HIDDEN_LAYER,
        final_norm: bool = WHERE_HIDDEN_FINAL_NORM,
        use_hook: bool = True,
    ):
        self.model = model
        self.processor = processor
        self.device = torch.device(device)
        self.layer = layer
        self.final_norm = final_norm
        self.use_hook = use_hook
        self.visual = resolve_visual(model)
        self.lm = resolve_language_model(model)
        self.pad_id = processor.tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = processor.tokenizer.eos_token_id
        for p in model.parameters():           # protocol 3: the VLM is frozen
            p.requires_grad_(False)
        model.eval()

    # -- the contract -------------------------------------------------------
    @torch.no_grad()
    def encode(self, items: Sequence[EncodeItem]) -> list[EncodeResult]:
        if not items:
            return []
        tok_lists = [list(it.prompt_ids) + list(it.where_ids) for it in items]
        max_len = max(len(t) for t in tok_lists)
        input_ids = torch.full((len(items), max_len), self.pad_id, dtype=torch.long)
        attn = torch.zeros((len(items), max_len), dtype=torch.long)
        for i, t in enumerate(tok_lists):
            input_ids[i, : len(t)] = torch.tensor(t, dtype=torch.long)
            attn[i, : len(t)] = 1

        img = self.processor.image_processor(
            images=[it.image for it in items], do_resize=False, return_tensors="pt"
        )
        grid_thw = img["image_grid_thw"]
        dtype = next(self.model.parameters()).dtype

        fhook = FPreHook(self.visual)
        lhook = LastLayerHook(self.lm, self.layer) if self.use_hook else None
        kwargs: dict[str, Any] = {}
        if lhook is None:
            kwargs["output_hidden_states"] = True

        with fhook.attached():
            if lhook is not None:
                with lhook.attached():
                    out = self.model(
                        input_ids=input_ids.to(self.device),
                        attention_mask=attn.to(self.device),
                        pixel_values=img["pixel_values"].to(self.device, dtype),
                        image_grid_thw=grid_thw.to(self.device),
                        **kwargs,
                    )
                hidden = lhook.captured
            else:
                out = self.model(
                    input_ids=input_ids.to(self.device),
                    attention_mask=attn.to(self.device),
                    pixel_values=img["pixel_values"].to(self.device, dtype),
                    image_grid_thw=grid_thw.to(self.device),
                    **kwargs,
                )
                hidden = out.hidden_states[self.layer]
        del out
        if hidden is None:
            raise RuntimeError("no decoder hidden states captured")
        if self.final_norm:
            hidden = self.lm.norm(hidden)
        if hidden.shape[-1] != TEXT_HIDDEN:
            raise AssertionError(
                f"H_where is {hidden.shape[-1]}-dim, config.json says {TEXT_HIDDEN}"
            )

        fpre = fhook.split(grid_thw)
        results = []
        for i, it in enumerate(items):
            n_p, n_w = len(it.prompt_ids), len(it.where_ids)
            g = fpre[i]
            results.append(EncodeResult(
                sample_id=it.sample_id,
                f_pre=g.float(),
                h_where=hidden[i, n_p:n_p + n_w].float(),
                grid_h=int(g.shape[0]), grid_w=int(g.shape[1]),
                n_prompt_tokens=n_p,
                meta={"n_where_tokens": n_w, "seq_len": n_p + n_w},
            ))
        return results

    # -- generation (offline job) ------------------------------------------
    @torch.no_grad()
    def generate_where(
        self,
        items: Sequence[EncodeItem],
        max_new_tokens: int,
        eos_token_id: int | None = None,
    ) -> list[list[int]]:
        """Greedy continuation of the prompts.  Left padding, so every sample's
        generation starts at the same position and the KV cache stays valid."""
        if not items:
            return []
        prompts = [list(it.prompt_ids) for it in items]
        max_len = max(len(p) for p in prompts)
        input_ids = torch.full((len(items), max_len), self.pad_id, dtype=torch.long)
        attn = torch.zeros((len(items), max_len), dtype=torch.long)
        for i, p in enumerate(prompts):
            input_ids[i, max_len - len(p):] = torch.tensor(p, dtype=torch.long)
            attn[i, max_len - len(p):] = 1
        img = self.processor.image_processor(
            images=[it.image for it in items], do_resize=False, return_tensors="pt"
        )
        dtype = next(self.model.parameters()).dtype
        gen = self.model.generate(
            input_ids=input_ids.to(self.device),
            attention_mask=attn.to(self.device),
            pixel_values=img["pixel_values"].to(self.device, dtype),
            image_grid_thw=img["image_grid_thw"].to(self.device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=eos_token_id,
        )
        new = gen[:, max_len:]
        return [[int(t) for t in row.tolist()] for row in new]

    def facts(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "final_norm": self.final_norm,
            "use_hook": self.use_hook,
            "text_hidden": TEXT_HIDDEN,
            "n_text_layers": len(self.lm.layers),
            "n_vision_blocks": len(self.visual.blocks),
            "dtype": str(next(self.model.parameters()).dtype),
            "device": str(self.device),
            "any_param_requires_grad": any(p.requires_grad for p in self.model.parameters()),
        }
