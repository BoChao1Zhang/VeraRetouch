"""UNIQ wave-5: readout-source comparison + LoRA arms (EPR-011) -- NEW FILE.

Freeze contract (unchanged): running/queued arms import the amort package as
it stands; everything here is a new module wired by ``run_uniq3_arm.py`` via
documented module-attribute seams.

Two axes, four arms, all single-factor against an existing reference:

READOUT axis (vs arm B = hidden-sequence readout, the incumbent):
* ``code``  -- the parsed 21-dim geometry code (builder's ``_geom_of``, which
  follows the CONDITIONING text, so shuffle/foreign controls stay honest) is
  embedded as three group tokens (shape/dir/ext, the PCH §2.2 grouping) and
  becomes the queries' cross-attention K/V.  Text-token K/V is OFF.  This is
  the "A1/discrete readout" leg of the three-way comparison.
* ``attn``  -- F-LMM-style routing readout: per-sample attention maps from
  the SAME frozen forward (selected layers, where-rows x image-cols, mean
  over heads and where tokens) enter as extra dense channels next to the
  similarity field.  Text-token K/V is OFF.  Batch-of-1 encode with
  ``output_attentions=True`` (eager is already pinned arm-wide), sliced
  immediately -- the B1 "逐层即切即弃" intent without the 60GB cache.
  DISCLOSED crudeness: mean over heads/where-tokens, 4 layers; a null
  result convicts THIS readout form, not the attention route in general.

LoRA axis (breaks the frozen-VLM premise on purpose; the review's missing
diagnostic "瓶颈在接口还是在冻结"):
* ``TrainableVLM`` bypasses ``FrozenVLM.encode``'s ``@torch.no_grad`` via
  ``__wrapped__`` (the decorator keeps the original), injects PEFT LoRA into
  language q/k/v/o and visual attn qkv/proj, enables gradient checkpointing,
  and re-exposes the adapter parameters through the AmortModel so the frozen
  trainer's ``model.parameters()`` optimiser sees them.  builder.build keeps
  ``feat``/``cond_h``/``sim`` outside its no_grad block (verified by reading
  data.py:667 -- only sem64/phi are guarded), so gradients reach the
  adapters without touching a single frozen file.
  DISCLOSED: the similarity-field norm is fitted at step 0 (LoRA B=0 ==
  base model) and NOT refitted as the backbone drifts; drift is part of
  what the arm is allowed to adapt to.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from q3vl.whereb.hiddens import EncodeItem, EncodeResult, FrozenVLM

from .model import AmortModel as _AmortModelBase
from .uniq import UniQHead

__all__ = ["VARIANT3", "UniQ3Head", "AmortModelV3", "TrainableVLM",
           "AttnCaptureVLM", "make_attn_builder"]

#: process-wide switches, set by run_uniq3_arm before anything is built
VARIANT3: dict[str, Any] = {"readout": "hidden", "lora": False,
                            "attn_layers": (8, 16, 24, 35), "lora_r": 16}

#: (start, end) slot ranges of the 21-dim code -- the AMD-6 grouping
_CODE_GROUPS = ((0, 5), (5, 14), (14, 21))


class UniQ3Head(UniQHead):
    """UniQHead whose cross-attention K/V source is switchable.

    ``readout='hidden'`` reproduces the base head exactly.  ``'code'`` swaps
    the K/V for three embedded code-group tokens (+ learnable null token when
    the code is empty).  ``'attn'`` removes token K/V entirely -- conditioning
    arrives as dense channels through ``extra_ch`` instead.
    """

    def __init__(self, *args, readout: str = "hidden", **kw):
        super().__init__(*args, **kw)
        if readout not in ("hidden", "code", "attn"):
            raise ValueError(f"unknown readout {readout!r}")
        self.readout = readout
        ch = self.queries.shape[1]
        if readout == "code":
            self.code_embed = nn.Parameter(torch.randn(21, ch) * 0.02)
            self.code_norm = nn.LayerNorm(ch)
            self.code_null = nn.Parameter(torch.zeros(1, ch))

    def _kv_tokens(self, h_where, h_mask, geom):
        if self.readout == "hidden":
            kv = self.text_proj(h_where.float())
            pad = None
            if h_mask is not None:
                pad = h_mask.reshape(1, -1) < 0.5
                if bool(pad.all()):
                    pad = None
            return kv, pad
        if self.readout == "code":
            if geom is not None and float(geom.abs().sum()) > 0:
                g = geom.reshape(-1).float().to(self.code_embed.device)
                toks = []
                for a, b in _CODE_GROUPS:
                    w = g[a:b]
                    toks.append(self.code_norm(
                        (w.unsqueeze(-1) * self.code_embed[a:b]).sum(0)
                        / w.sum().clamp_min(1.0)))
                return torch.stack(toks).unsqueeze(0), None
            return self.code_null.unsqueeze(0), None
        return None, None                                     # 'attn': no K/V

    def forward(self, feat, extra, cond, h_where, h_mask,
                geom=None) -> dict[str, Any]:
        codes = self.tower(feat, extra, cond)
        gh, gw = codes.shape[-2:]
        pix = codes[0].reshape(codes.shape[1], gh * gw)
        if self.fourier_bands:
            four = self._fourier(gh, gw, codes.device, pix.dtype)
            pix = torch.cat([pix, four.reshape(four.shape[0], gh * gw)], dim=0)

        kv, pad = self._kv_tokens(h_where, h_mask, geom)
        qb = self.queries.unsqueeze(0)
        if kv is not None:
            att, _ = self.xattn(qb, kv, kv, key_padding_mask=pad,
                                need_weights=False)
            q = qb + att
        else:
            q = qb
        q = (q + self.ffn(self.q_norm(q)))[0]

        wb = self.to_mask(q)
        raw = wb[:, :-1] @ pix + wb[:, -1:]
        from q3vl.where.config import S_SCALE

        s_all = (S_SCALE * torch.tanh(raw / S_SCALE)).reshape(
            self.n_queries, gh, gw)
        return {"s_all": s_all, "cls_logits": self.cls(q),
                "sel_logits": self.sel(q).reshape(-1)}

    def facts(self) -> dict[str, Any]:
        return {**super().facts(), "readout": self.readout}


class AmortModelV3(_AmortModelBase):
    """UNIQ arm builds a UniQ3Head; geometry codes bypass the channel path.

    ``geom_inject`` is intercepted: the BUILDER may produce codes (the entry
    passes the flag through to it), but the model consumes them as K/V tokens
    only -- the broadcast-channel and PCH paths stay off, so ``_extra`` never
    widens the stem.  ``attn_extra_ch`` widens the stem instead for the attn
    readout, whose maps ride concatenated onto ``sim``.
    """

    def __init__(self, arm: str = "P1", **kw):
        self._v3_geom_kv = bool(kw.pop("geom_inject", False))
        attn_extra = int(kw.pop("attn_extra_ch", 0))
        super().__init__(arm, geom_inject=False, **kw)
        self._attn_extra = attn_extra
        if arm != "UNIQ":
            return
        film_dim = self.cond.out_dim if self.use_film else None
        extra_ch = (int(self.use_sim_field)
                    + int(self.use_center_prior_channel) + attn_extra)
        self.geo = UniQ3Head(
            kw.get("ch", 128), kw.get("in_dim", 1024), extra_ch,
            kw.get("n_blocks", 6), film_dim,
            text_dim=kw.get("cond_text_dim", 2560),
            n_queries=kw.get("uniq_k", 4),
            fourier_bands=kw.get("uniq_fourier_bands", 0),
            fourier_scale=kw.get("uniq_fourier_scale", 1.0),
            seed=kw.get("seed", 0),
            readout=VARIANT3["readout"])
        self._maybe_adopt_lora()

    def _maybe_adopt_lora(self):
        """Expose the injected adapters to the frozen trainer's optimiser."""
        vlm = VARIANT3.get("vlm_ref")
        if VARIANT3.get("lora") and vlm is not None:
            ps = [p for n, p in vlm.model.named_parameters() if "lora_" in n]
            if not ps:
                raise RuntimeError("lora=True but no lora_ parameters found")
            self.vlm_lora = nn.ParameterList(ps)

    def train(self, mode: bool = True):
        vlm = VARIANT3.get("vlm_ref")
        if vlm is not None and hasattr(vlm, "_grad_on"):
            vlm._grad_on = bool(mode)      # trainer.train()/eval() drives it
        return super().train(mode)

    def forward_geo(self, feat, cond, phi_dir, *, sim=None, center=None,
                    geom=None, valid_grid=None, guide_hi=None,
                    grid_h=0, grid_w=0, h_where=None, h_mask=None,
                    h_cond=None, sample=None):
        # `h_cond` / `sample` are the EPR-018..023 arguments the shared callers
        # now always pass; this arm ignores them (its conditioning is the K
        # query-token rows of `h_where`).  Accepting and dropping them keeps the
        # signature compatible without changing a single value it computes.
        if self.arm != "UNIQ":
            if self.arm == "P3prime" and not hasattr(self, "_lora_adopted"):
                self._lora_adopted = True
            return super().forward_geo(
                feat, cond, phi_dir, sim=sim, center=center, geom=None,
                valid_grid=valid_grid, guide_hi=guide_hi, grid_h=grid_h,
                grid_w=grid_w, h_where=h_where, h_mask=h_mask)
        from q3vl.whereb.fields import no_autocast

        extra = self._extra(sim, center, None)
        with no_autocast(feat.device.type):
            u = self.geo(feat.float(),
                         None if extra is None else extra.float(),
                         cond.float(), h_where, h_mask,
                         geom=geom if self._v3_geom_kv else None)
            sel = int(u["sel_logits"].detach().argmax())
            s_low = u["s_all"][sel]
            out = {"m_low": self.geo.mask_of(s_low), "s_low": s_low,
                   "params": {}, "uniq": u, "uniq_selected": sel}
            if guide_hi is not None:
                s_hi = F.interpolate(s_low.reshape(1, 1, *s_low.shape[-2:]),
                                     size=guide_hi.shape[-2:], mode="bilinear",
                                     align_corners=False)
                out["s_hi"] = s_hi
                out["m_hi"] = self.geo.mask_of(s_hi)
        return out


class AmortModelVLora(_AmortModelBase):
    """Non-UNIQ arms (e.g. P3prime) + LoRA adapter exposure."""

    def __init__(self, arm: str = "P1", **kw):
        super().__init__(arm, **kw)
        vlm = VARIANT3.get("vlm_ref")
        if VARIANT3.get("lora") and vlm is not None:
            ps = [p for n, p in vlm.model.named_parameters() if "lora_" in n]
            if not ps:
                raise RuntimeError("lora=True but no lora_ parameters found")
            self.vlm_lora = nn.ParameterList(ps)

    def train(self, mode: bool = True):
        vlm = VARIANT3.get("vlm_ref")
        if vlm is not None and hasattr(vlm, "_grad_on"):
            vlm._grad_on = bool(mode)
        return super().train(mode)


#: the genuine encode body, minus its @torch.no_grad decorator
_ENCODE_NOGRADLESS = FrozenVLM.encode.__wrapped__


class TrainableVLM(FrozenVLM):
    """FrozenVLM whose forward keeps the graph and carries LoRA adapters."""

    def __init__(self, model, processor, device="cuda", *, lora_r: int = 16,
                 lora_alpha: int = 32, **kw):
        super().__init__(model, processor, device, **kw)   # freezes everything
        from peft import LoraConfig, inject_adapter_in_model

        cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0, bias="none",
            target_modules=r".*\.(q_proj|k_proj|v_proj|o_proj)$"
                           r"|.*visual.*\.attn\.(qkv|proj)$")
        inject_adapter_in_model(cfg, model)
        n = 0
        for name, p in model.named_parameters():
            if "lora_" in name:
                # bf16 storage by user ruling (no fp32 master copies); the
                # compute path is bf16 autocast either way
                p.requires_grad_(True)
                n += p.numel()
        if not n:
            raise RuntimeError("LoRA injection matched no modules")
        self.n_lora_params = n
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        # checkpointing is a no-op on eval-mode HF modules; dropout is 0.0
        # arm-wide so train() changes recomputation, not behaviour
        model.train()

    #: flipped by the model's train()/eval() via the seam (canary lesson:
    #: the frozen entry's norm-fitting loop calls `.numpy()` on encode
    #: outputs BEFORE training starts, and a grad-carrying tensor refuses)
    _grad_on: bool = False

    def encode(self, items):
        if self._grad_on and torch.is_grad_enabled():
            return _ENCODE_NOGRADLESS(self, items)
        with torch.no_grad():
            return _ENCODE_NOGRADLESS(self, items)

    def facts(self):
        return {**super().facts(), "lora": True,
                "n_lora_params": self.n_lora_params,
                "gradient_checkpointing": True}


class AttnCaptureVLM(FrozenVLM):
    """FrozenVLM + per-sample attention-map capture (batch-of-1 encode).

    ``output_attentions=True`` under the pinned eager kernel; selected layers
    are sliced to (where-rows x image-cols), meaned over heads and where
    tokens, and stashed by sample_id for the builder to append as channels.
    """

    def __init__(self, *a, attn_layers=(8, 16, 24, 35), **kw):
        super().__init__(*a, **kw)
        self.attn_layers = tuple(attn_layers)
        self._maps: dict[str, torch.Tensor] = {}
        tok = self.processor.tokenizer
        self.image_pad_id = tok.convert_tokens_to_ids("<|image_pad|>")
        if self.image_pad_id is None or self.image_pad_id < 0:
            raise RuntimeError("cannot resolve <|image_pad|> token id")

    @torch.no_grad()
    def encode(self, items):
        results = []
        for it in items:
            res = super().encode([it])[0]
            self._maps[it.sample_id] = self._attn_map(it, res)
            results.append(res)
        return results

    @torch.no_grad()
    def _attn_map(self, it: EncodeItem, res: EncodeResult) -> torch.Tensor:
        ids = list(it.prompt_ids) + list(it.where_ids)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        attn_mask = torch.ones_like(input_ids)
        img = self.processor.image_processor(images=[it.image], do_resize=False,
                                             return_tensors="pt")
        dtype = next(self.model.parameters()).dtype
        out = self.model(input_ids=input_ids, attention_mask=attn_mask,
                         pixel_values=img["pixel_values"].to(self.device, dtype),
                         image_grid_thw=img["image_grid_thw"].to(self.device),
                         output_attentions=True)
        img_cols = (input_ids[0] == self.image_pad_id).nonzero().reshape(-1)
        gh2, gw2 = res.grid_h // 2, res.grid_w // 2
        if img_cols.numel() != gh2 * gw2:
            raise AssertionError(
                f"{it.sample_id}: {img_cols.numel()} image tokens != merger "
                f"grid {gh2}x{gw2}")
        n_p, n_w = res.n_prompt_tokens, len(it.where_ids)
        maps = []
        for li in self.attn_layers:
            a = out.attentions[li][0]                     # (H, S, S)
            if n_w:
                sl = a[:, n_p:n_p + n_w, :][..., img_cols]  # (H, T_w, N_img)
                maps.append(sl.float().mean(dim=(0, 1)))
            else:                                          # null context
                maps.append(torch.zeros(img_cols.numel(), device=a.device))
        del out
        m = torch.stack(maps).reshape(len(self.attn_layers), gh2, gw2)
        return F.interpolate(m.unsqueeze(0), size=(res.grid_h, res.grid_w),
                             mode="bilinear", align_corners=False)[0]

    def pop_map(self, sample_id: str) -> torch.Tensor:
        return self._maps.pop(sample_id)

    def facts(self):
        return {**super().facts(), "attn_capture_layers": list(self.attn_layers)}


def make_attn_builder(base_cls):
    """Builder subclass: append the captured attention maps onto ``sim``."""

    class AttnBuilder(base_cls):
        def build(self, samples, modes):
            out = super().build(samples, modes)
            for x in out:
                amap = self.vlm.pop_map(x.sample_id).to(self.device)
                amap = amap.unsqueeze(0)                   # (1, L, gh, gw)
                x.sim = amap if x.sim is None else torch.cat(
                    [x.sim, amap], dim=1)
            return out

    return AttnBuilder
