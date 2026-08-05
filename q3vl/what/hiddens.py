"""One frozen VLM forward -> ``F_pre``, ``H_where`` and ``H_color``.

Protocol 7.1 needs ``H_color`` -- all ``<color>...</color>`` token hidden states
-- and protocol 2.3 forbids a second visual forward when the model is running
anyway.  Both come out of a single frozen forward here, reusing Where-B's
verified hooks (:class:`q3vl.whereb.hiddens.LastLayerHook`,
:class:`q3vl.where.fpre.FPreHook`) rather than re-deriving them.

The layer and the final-norm decision are **imported** from
``q3vl.whereb.contracts`` (ruling D-B2).  If Stage-What read ``H_color`` at a
different depth or on the other side of the final RMSNorm than Stage-Where reads
``H_where``, the two stages' language conditioning would not be comparable and
nothing downstream would notice.

Two arm-dependent facts about the sequence, both of them protocol 8.2:

* ``where_prefix=True`` (every ``WC-*`` arm) builds
  ``prompt + <where>...</where> + <color>...</color>``: causal attention lets the
  ``<color>`` positions see the ``<where>`` text, which is exactly the residual
  the strict no-where control exists to bound;
* ``where_prefix=False`` (``C01``/``C02``) builds ``prompt + <color>...</color>``
  with no ``<where>`` span at all.

That single boolean is the whole difference between ``T01`` and ``C01``, and it
lives here, at the sequence, where it is auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from q3vl.where.fpre import FPreHook
from q3vl.whereb.hiddens import LastLayerHook, resolve_language_model, resolve_visual

from .config import COLOR_HIDDEN_FINAL_NORM, COLOR_HIDDEN_LAYER, TEXT_HIDDEN

__all__ = ["ColorEncodeItem", "ColorEncodeResult", "WhatVLM"]


@dataclass
class ColorEncodeItem:
    sample_id: str
    image: Any                        # PIL image, already spec-5 sized
    prompt_ids: Sequence[int]
    where_ids: Sequence[int] = ()     # empty when the arm has no <where> prefix
    color_ids: Sequence[int] = ()


@dataclass
class ColorEncodeResult:
    sample_id: str
    f_pre: torch.Tensor               # (grid_h, grid_w, 1024)
    h_color: torch.Tensor             # (T_color, 2560)
    h_where: torch.Tensor             # (T_where, 2560); may be empty
    grid_h: int
    grid_w: int
    meta: dict[str, Any] = field(default_factory=dict)


class WhatVLM:
    """Frozen Qwen3-VL: one forward, three outputs, no gradients ever."""

    def __init__(self, model: torch.nn.Module, processor,
                 device: str | torch.device = "cuda", *,
                 layer: int = COLOR_HIDDEN_LAYER,
                 final_norm: bool = COLOR_HIDDEN_FINAL_NORM,
                 use_hook: bool = True):
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
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()

    @torch.no_grad()
    def encode(self, items: Sequence[ColorEncodeItem]) -> list[ColorEncodeResult]:
        if not items:
            return []
        toks = [list(it.prompt_ids) + list(it.where_ids) + list(it.color_ids)
                for it in items]
        max_len = max(len(t) for t in toks)
        input_ids = torch.full((len(items), max_len), self.pad_id, dtype=torch.long)
        attn = torch.zeros((len(items), max_len), dtype=torch.long)
        for i, t in enumerate(toks):
            input_ids[i, : len(t)] = torch.tensor(t, dtype=torch.long)
            attn[i, : len(t)] = 1

        img = self.processor.image_processor(
            images=[it.image for it in items], do_resize=False, return_tensors="pt")
        grid_thw = img["image_grid_thw"]
        dtype = next(self.model.parameters()).dtype

        fhook = FPreHook(self.visual)
        lhook = LastLayerHook(self.lm, self.layer) if self.use_hook else None
        kwargs: dict[str, Any] = {} if lhook is not None else {"output_hidden_states": True}
        with fhook.attached():
            if lhook is not None:
                with lhook.attached():
                    out = self.model(
                        input_ids=input_ids.to(self.device),
                        attention_mask=attn.to(self.device),
                        pixel_values=img["pixel_values"].to(self.device, dtype),
                        image_grid_thw=grid_thw.to(self.device), **kwargs)
                hidden = lhook.captured
            else:
                out = self.model(
                    input_ids=input_ids.to(self.device),
                    attention_mask=attn.to(self.device),
                    pixel_values=img["pixel_values"].to(self.device, dtype),
                    image_grid_thw=grid_thw.to(self.device), **kwargs)
                hidden = out.hidden_states[self.layer]
        del out
        if hidden is None:
            raise RuntimeError("no decoder hidden states captured")
        if self.final_norm:
            hidden = self.lm.norm(hidden)
        if hidden.shape[-1] != TEXT_HIDDEN:
            raise AssertionError(
                f"H_color is {hidden.shape[-1]}-dim, config.json says {TEXT_HIDDEN}")

        fpre = fhook.split(grid_thw)
        results = []
        for i, it in enumerate(items):
            n_p, n_w, n_c = len(it.prompt_ids), len(it.where_ids), len(it.color_ids)
            g = fpre[i]
            results.append(ColorEncodeResult(
                sample_id=it.sample_id, f_pre=g.float(),
                h_where=hidden[i, n_p:n_p + n_w].float(),
                h_color=hidden[i, n_p + n_w:n_p + n_w + n_c].float(),
                grid_h=int(g.shape[0]), grid_w=int(g.shape[1]),
                meta={"n_where_tokens": n_w, "n_color_tokens": n_c,
                      "seq_len": n_p + n_w + n_c, "where_prefix": n_w > 0},
            ))
        return results

    def facts(self) -> dict[str, Any]:
        return {
            "layer": self.layer, "final_norm": self.final_norm,
            "use_hook": self.use_hook, "text_hidden": TEXT_HIDDEN,
            "n_text_layers": len(self.lm.layers),
            "n_vision_blocks": len(self.visual.blocks),
            "dtype": str(next(self.model.parameters()).dtype),
            "device": str(self.device),
            "any_param_requires_grad": any(p.requires_grad
                                           for p in self.model.parameters()),
        }
