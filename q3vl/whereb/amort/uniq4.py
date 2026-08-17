"""UNIQ wave-6: in-context query tokens (MetaQueries/LISA axis) -- NEW FILE.

Completes the 2x2 readout factorial the user asked for:

                      | frozen VLM          | VLM + LoRA
  external bridge     | arms B/D (done)     | LORA_UNIQK8 (queued)
  in-context q-tokens | MQ (this file)      | ST (this file, = multi
                      |                     |  special-token finetune)

Mechanism: K new token ids are appended after the reasoning span; their
fp32 embeddings are trainable (written into the embedding output by a
forward hook, so the resized rows themselves stay frozen) and their
last-layer hiddens BECOME the unified head's query states -- the VLM's own
36 layers do the aggregation (MetaQueries arXiv:2504.06256 "effective even
when the MLLM backbone remains frozen"; with LoRA this is the LISA/PixelLM
writer-side form, arXiv:2308.00692/2312.02228).

Gradients must flow through the VLM forward even when its weights are
frozen (the query embeddings are inputs), so encode always uses the
``__wrapped__`` no-grad bypass plus gradient checkpointing -- identical
memory/runtime plumbing to the wave-5 LoRA arms; eval stays graph-free
under ``evaluate_context``'s ``@torch.no_grad``.

Death-list check: W01/W02's MetaCanvas death is "query -> pooled -> 71-dim
regression" (M3 + S3).  Here queries feed the dot-product mask head with
WTA, no 71-dim code exists, and the M3 execution column is armed on every
board -- this is a boundary-mapping test, not a resurrection.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
import torch.nn as nn

from q3vl.where.config import S_SCALE
from q3vl.whereb.hiddens import FrozenVLM

from .model import AmortModel as _AmortModelBase
from .uniq import UniQHead
from .uniq3 import _ENCODE_NOGRADLESS

__all__ = ["VARIANT4", "QueryTokVLM", "UniQ4Head", "AmortModelV4",
           "RefineLayer"]

#: ``head_cls`` / ``head_kwargs`` are the wrapper seam every EPR-01x entry
#: script writes through: ``None`` / ``{}`` = the frozen ST_LANG configuration.
VARIANT4: dict[str, Any] = {"n_qtok": 8, "lora": False, "lora_r": 16,
                            "vlm_ref": None, "head_cls": None,
                            "head_kwargs": {}}


class QueryTokVLM(FrozenVLM):
    """FrozenVLM + K appended query tokens with trainable fp32 embeddings.

    ``encode`` rewrites each item's ``where_ids`` to append the K query ids;
    the frozen slicing in the original body then returns ``h_where`` with the
    K query hiddens as its LAST K rows.  The model side (AmortModelV4) strips
    them before the pooled-FiLM path and hands them to the head as queries.
    """

    def __init__(self, model, processor, device="cuda", *, n_qtok: int = 8,
                 lora: bool = False, lora_r: int = 16, aux_groups: int = 0,
                 aux_seed: int = 20260813, **kw):
        super().__init__(model, processor, device, **kw)
        self.n_qtok = int(n_qtok)
        #: EPR-016: m extra groups of K query ids, appended AFTER the formal
        #: ones and spliced in only on the training (grad) encode branch.
        self.aux_groups = int(aux_groups)
        self.aux_seed = int(aux_seed)
        n_aux = self.n_qtok * self.aux_groups
        emb = model.get_input_embeddings()
        base_vocab = emb.weight.shape[0]
        model.resize_token_embeddings(base_vocab + self.n_qtok + n_aux)
        emb = model.get_input_embeddings()
        emb.weight.requires_grad_(False)
        self.q_ids = tuple(range(base_vocab, base_vocab + self.n_qtok))
        self.aux_ids = tuple(range(base_vocab + self.n_qtok,
                                   base_vocab + self.n_qtok + n_aux))
        self.base_vocab = base_vocab
        with torch.no_grad():
            mean = emb.weight[:base_vocab].float().mean(0)
        g = torch.Generator(device="cpu").manual_seed(20260812)
        init = mean.cpu() + 0.02 * torch.randn(self.n_qtok, mean.shape[0],
                                               generator=g)
        #: bf16 storage by user ruling (no fp32 master copies); the hook
        #: writes it into the embedding output in the output's own dtype
        self.q_embed = nn.Parameter(
            init.to(self.device, dtype=next(model.parameters()).dtype))
        self.aux_embed = None
        if n_aux:
            #: same recipe as the formal group, own fixed seed (recorded in
            #: the run config); a SEPARATE Parameter so every baseline
            #: state_dict key keeps its shape and stays loadable.
            ga = torch.Generator(device="cpu").manual_seed(self.aux_seed)
            init_a = mean.cpu() + 0.02 * torch.randn(n_aux, mean.shape[0],
                                                     generator=ga)
            self.aux_embed = nn.Parameter(
                init_a.to(self.device, dtype=next(model.parameters()).dtype))

        def _hook(module, inp, out):
            ids = inp[0]
            mask = ids >= base_vocab
            if bool(mask.any()):
                out = out.clone()
                tab = (self.q_embed if self.aux_embed is None
                       else torch.cat([self.q_embed, self.aux_embed], dim=0))
                out[mask] = tab[ids[mask] - base_vocab].to(out.dtype)
            return out

        emb.register_forward_hook(_hook)

        self.lora = bool(lora)
        self.n_lora_params = 0
        if self.lora:
            from peft import LoraConfig, inject_adapter_in_model

            cfg = LoraConfig(
                r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.0, bias="none",
                target_modules=r".*\.(q_proj|k_proj|v_proj|o_proj)$"
                               r"|.*visual.*\.attn\.(qkv|proj)$")
            inject_adapter_in_model(cfg, model)
            for name, p in model.named_parameters():
                if "lora_" in name:
                    p.requires_grad_(True)   # bf16 storage (user ruling)
                    self.n_lora_params += p.numel()
            if not self.n_lora_params:
                raise RuntimeError("LoRA injection matched no modules")
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        model.train()          # checkpointing gate; dropout is 0.0 arm-wide

    #: flipped by AmortModelV4.train() (canary lesson: the frozen entry's
    #: norm-fitting loop calls `.numpy()` on encode outputs before training)
    _grad_on: bool = False

    def encode(self, items):
        train = bool(self._grad_on and torch.is_grad_enabled())
        ids = self.q_ids + self.aux_ids if (train and self.aux_ids) \
            else self.q_ids
        items2 = [replace(it, where_ids=tuple(it.where_ids) + ids)
                  for it in items]
        if train:
            return _ENCODE_NOGRADLESS(self, items2)
        with torch.no_grad():
            return _ENCODE_NOGRADLESS(self, items2)

    def facts(self):
        return {**super().facts(), "n_qtok": self.n_qtok, "lora": self.lora,
                "n_lora_params": self.n_lora_params,
                "q_ids": [self.q_ids[0], self.q_ids[-1]],
                "aux_groups": self.aux_groups,
                "aux_seed": self.aux_seed if self.aux_ids else None,
                "gradient_checkpointing": True}


class RefineLayer(nn.Module):
    """One Mask2Former decoder layer (EPR-013), pre-norm and zero-gated.

    Order is M2F's own: masked cross-attention (the predicted field of the
    previous step restricts what each query may read), then query self-
    attention, then FFN -- ``mask2former_transformer_decoder.py`` L399-416.
    Deviations from the M2F COCO default, both deliberate:

    * **pre-norm** (M2F ships ``forward_pre`` L52-62/L112-124/L169-173 but
      defaults ``PRE_NORM: False``).  With post-norm ``LayerNorm(q + 0) != q``,
      so no zero-init can make the layer an identity and step-0 equivalence
      with the baseline would be impossible.
    * **no memory positional encoding** (M2F L404 passes ``pos=pos[level]``).
      DELTA S5.6 forbids coordinate channels reaching this head.

    Zero-init: both attention ``out_proj``s and the FFN's last Linear, exactly
    the discipline ``uniq.py`` L112-118 already uses -- so a freshly built
    stack of these layers is the identity map and ``s_all`` is bit-identical
    to the no-refine baseline at step 0.
    """

    def __init__(self, ch: int, n_heads: int = 8, ffn_mult: int = 2,
                 self_attn: bool = True):
        super().__init__()
        self.norm_x = nn.LayerNorm(ch)
        self.xattn = nn.MultiheadAttention(ch, n_heads, batch_first=True)
        nn.init.zeros_(self.xattn.out_proj.weight)
        nn.init.zeros_(self.xattn.out_proj.bias)
        self.sattn = None
        if self_attn:
            self.norm_s = nn.LayerNorm(ch)
            self.sattn = nn.MultiheadAttention(ch, n_heads, batch_first=True)
            nn.init.zeros_(self.sattn.out_proj.weight)
            nn.init.zeros_(self.sattn.out_proj.bias)
        self.norm_f = nn.LayerNorm(ch)
        self.ffn = nn.Sequential(nn.Linear(ch, ffn_mult * ch), nn.GELU(),
                                 nn.Linear(ffn_mult * ch, ch))
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, q, mem, attn_mask, query_pos):
        h = self.norm_x(q)
        att, _ = self.xattn(h + query_pos, mem, mem, attn_mask=attn_mask,
                            need_weights=False)
        q = q + att
        if self.sattn is not None:
            h = self.norm_s(q)
            hp = h + query_pos
            att, _ = self.sattn(hp, hp, h, need_weights=False)
            q = q + att
        return q + self.ffn(self.norm_f(q))


class UniQ4Head(UniQHead):
    """Query states come from the VLM (last K rows of h_where), not from
    learnable-base + cross-attention.  Everything downstream is unchanged.

    Two default-off extensions live here:

    * ``n_refine_layers > 0`` (EPR-013) inserts masked cross-attention
      refinement between ``q_proj_in`` and ``to_mask``;
    * ``aux_groups > 0`` (EPR-016) makes the head read ``(1+m)*K`` query rows
      while TRAINING and the formal K rows at inference.

    Both are inert at their defaults: no module is constructed, no branch is
    entered, and the forward is the frozen ST_LANG path.
    """

    def __init__(self, *args, text_dim: int = 2560,
                 n_refine_layers: int = 0, refine_heads: int = 8,
                 refine_self_attn: bool = True, refine_thr_raw: bool = False,
                 refine_aux_loss: bool = True, refine_anneal: bool = False,
                 aux_groups: int = 0, aux_lambda: float = 1.0, **kw):
        super().__init__(*args, text_dim=text_dim, **kw)
        ch = self.queries.shape[1]
        self.q_proj_in = nn.Sequential(nn.LayerNorm(text_dim),
                                       nn.Linear(text_dim, ch))
        # -- EPR-016 ---------------------------------------------------------
        self.aux_groups = int(aux_groups)
        self.aux_lambda = float(aux_lambda)
        # -- EPR-013 (constructed LAST, so no existing module's init RNG moves)
        self.n_refine_layers = int(n_refine_layers)
        self.refine_heads = int(refine_heads)
        self.refine_thr_raw = bool(refine_thr_raw)
        self.refine_aux_loss = bool(refine_aux_loss)
        self.refine_anneal = bool(refine_anneal)
        self.refine = None
        if self.n_refine_layers:
            g = torch.Generator(device="cpu").manual_seed(self.fourier_seed + 13)
            self.query_pos = nn.Parameter(
                torch.randn(self.n_queries, ch, generator=g) * 0.02)
            self.refine = nn.ModuleList([
                RefineLayer(ch, self.refine_heads, self_attn=refine_self_attn)
                for _ in range(self.n_refine_layers)])
            #: EoMT mask-annealing state (eomt.py L33); all-ones = always
            #: masked, which is the Mask2Former behaviour.
            self.register_buffer("attn_mask_probs",
                                 torch.ones(self.n_refine_layers),
                                 persistent=False)

    # -- rows ----------------------------------------------------------------
    def n_rows(self) -> int:
        """How many query rows this forward consumes (EPR-016 aware)."""
        if self.aux_groups and self.training:
            return self.n_queries * (1 + self.aux_groups)
        return self.n_queries

    def _query_states(self, h_where, h_mask):        # overrides parent form
        rows = self.n_rows()
        qh = h_where[0, -rows:, :].float()                # (rows, 2560)
        q = self.q_proj_in(qh)
        q = q + self.ffn(self.q_norm(q.unsqueeze(0)))[0]
        return q

    # -- pieces --------------------------------------------------------------
    def _pixels(self, feat, extra, cond):
        codes = self.tower(feat, extra, cond)
        gh, gw = codes.shape[-2:]
        pix = codes[0].reshape(codes.shape[1], gh * gw)
        code_kv = pix
        if self.fourier_bands:
            four = self._fourier(gh, gw, codes.device, pix.dtype)
            pix = torch.cat([pix, four.reshape(four.shape[0], gh * gw)], dim=0)
        return pix, code_kv, gh, gw

    def _field(self, q, pix, head=None):
        """``(rows, N)`` s-field from query states -- the shared dot product."""
        wb = (self.to_mask if head is None else head)(q)
        raw = wb[:, :-1] @ pix + wb[:, -1:]
        return S_SCALE * torch.tanh(raw / S_SCALE)

    def _run_refine(self, q, pix, code_kv, gh, gw, aux):
        rows = q.shape[0]
        mem = code_kv.transpose(0, 1).unsqueeze(0)            # (1, N, ch)
        qpos = self.query_pos
        if rows != qpos.shape[0]:                             # EPR-016 groups
            qpos = qpos.repeat(rows // qpos.shape[0], 1)
        qpos = qpos.unsqueeze(0).to(q.dtype)
        x = q.unsqueeze(0)
        for i, layer in enumerate(self.refine):
            s_prev = self._field(x[0], pix)                   # (rows, N)
            if self.refine_aux_loss:
                aux.append({"s_all": s_prev.reshape(rows, gh, gw),
                            "weight": 1.0, "tag": f"ref{i}"})
            with torch.no_grad():
                d = s_prev.detach()
                # gain > 0 makes these the same predicate; both code paths are
                # kept because the ablation row compares them literally.
                block = (d <= 0) if self.refine_thr_raw \
                    else (self.mask_of(d) < 0.5)
                # M2F L398: a fully masked row would softmax over all -inf and
                # emit NaN.  p=0.15 foreign samples drive EVERY field negative,
                # so this is the common case here, not a corner case.
                block[block.all(dim=-1)] = False
                # training-state gated: annealing draws from the global RNG and
                # randomly UNMASKS rows, so letting it run under `eval()` would
                # make evaluation stochastic and would move the RNG stream that
                # the training loop's sampling depends on (N1 lesson).  EoMT
                # anneals during training only.
                if self.refine_anneal and self.training:
                    p = float(self.attn_mask_probs[i])
                    if p < 1.0:                               # EoMT L71-82
                        off = torch.rand(rows, device=block.device) >= p
                        block[off] = False
            am = block.unsqueeze(0).expand(self.refine_heads, rows,
                                           block.shape[-1])
            x = layer(x, mem, am, qpos)
        return x[0], aux

    def set_anneal_progress(self, step: int, total: int) -> None:
        """EoMT ``mask_annealing`` (lightning_module.py L199-224).

        Layer ``i`` starts annealing at ``(i+1)/(L+1)*T`` and is fully open at
        ``(i+2)/(L+1)*T`` -- the proportional generalisation of the published
        4-block COCO schedule.  A no-op unless ``refine_anneal`` is on.
        """
        if not (self.refine_anneal and self.refine is not None) or total <= 0:
            return
        n = self.n_refine_layers
        for i in range(n):
            start = (i + 1) / (n + 1) * total
            end = (i + 2) / (n + 1) * total
            if step < start:
                p = 1.0
            elif step >= end:
                p = 0.0
            else:
                p = (1.0 - (step - start) / max(end - start, 1e-9)) ** 0.9
            self.attn_mask_probs[i] = float(p)

    def _post_query(self, q, pix, gh, gw, aux):
        """Hook for iterative heads (EPR-014 ``UniQ5Head``).  Identity here."""
        return q, aux, None

    def forward(self, feat, extra, cond, h_where, h_mask):
        pix, code_kv, gh, gw = self._pixels(feat, extra, cond)
        q = self._query_states(h_where, h_mask)
        aux: list[dict[str, Any]] = []
        if self.refine is not None:
            q, aux = self._run_refine(q, pix, code_kv, gh, gw, aux)
        q, aux, head = self._post_query(q, pix, gh, gw, aux)
        rows = q.shape[0]
        s_all = self._field(q, pix, head).reshape(rows, gh, gw)
        out: dict[str, Any] = {
            "s_all": s_all, "cls_logits": self.cls(q),
            #: the selection protocol is defined on the FORMAL group only
            "sel_logits": self.sel(q[:self.n_queries]).reshape(-1)}
        if aux:
            out["aux_supervision"] = aux
        if self.aux_groups and self.training:
            out["aux_groups"] = self.aux_groups
            out["aux_lambda"] = self.aux_lambda
        return out

    def facts(self):
        return {**super().facts(),
                "n_refine_layers": self.n_refine_layers,
                "refine_self_attn": bool(
                    self.refine is not None and self.refine[0].sattn is not None),
                "refine_thr": "raw>0" if self.refine_thr_raw else "sigmoid<0.5",
                "refine_aux_loss": self.refine_aux_loss,
                "refine_anneal": self.refine_anneal,
                "aux_groups": self.aux_groups,
                "aux_lambda": self.aux_lambda}


class AmortModelV4(_AmortModelBase):
    """Strips the K query-token hiddens from the pooled path and hands them
    to the head; exposes q_embed (+ LoRA) to the frozen trainer's optimiser."""

    def __init__(self, arm: str = "UNIQ", **kw):
        super().__init__(arm, **kw)
        self._k = int(VARIANT4["n_qtok"])
        hkw = dict(VARIANT4.get("head_kwargs") or {})
        #: EPR-016: how many EXTRA query groups the training forward carries
        self._aux = int(hkw.get("aux_groups", 0))
        if arm == "UNIQ":
            film_dim = self.cond.out_dim if self.use_film else None
            extra_ch = (int(self.use_sim_field)
                        + int(self.use_center_prior_channel))
            head_cls = VARIANT4.get("head_cls") or UniQ4Head
            self.geo = head_cls(
                kw.get("ch", 128), kw.get("in_dim", 1024), extra_ch,
                kw.get("n_blocks", 6), film_dim,
                text_dim=kw.get("cond_text_dim", 2560),
                n_queries=self._k, seed=kw.get("seed", 0),
                iou_head=bool(kw.get("uniq_iou_head", False)),
                sel_stability=float(kw.get("uniq_sel_stability", 0.0)),
                **hkw)
        vlm = VARIANT4.get("vlm_ref")
        if vlm is not None:
            ps = [vlm.q_embed]
            if getattr(vlm, "aux_embed", None) is not None:
                ps.append(vlm.aux_embed)
            ps += [p for n, p in vlm.model.named_parameters() if "lora_" in n]
            self.vlm_extra = nn.ParameterList(ps)

    def _query_rows(self) -> int:
        """Query-token rows h_where carries in the CURRENT mode (EPR-016)."""
        if self._aux and self.training:
            return self._k * (1 + self._aux)
        return self._k

    def train(self, mode: bool = True):
        vlm = VARIANT4.get("vlm_ref")
        if vlm is not None and hasattr(vlm, "_grad_on"):
            vlm._grad_on = bool(mode)
        return super().train(mode)

    def cond_of(self, h_where, h_mask, word_ids, word_offsets):
        # the pooled-FiLM path must not see the query tokens, or the pooled
        # vector becomes a function of what the queries absorbed -- a leak
        # between the two conditioning pathways
        k = self._query_rows()
        return super().cond_of(h_where[:, :-k, :] if h_where.shape[1] > k
                               else h_where[:, :0, :],
                               None, word_ids, word_offsets)

    def forward_geo(self, feat, cond, phi_dir, *, sim=None, center=None,
                    geom=None, valid_grid=None, guide_hi=None,
                    grid_h=0, grid_w=0, h_where=None, h_mask=None,
                    h_cond=None, sample=None):
        # `h_cond` / `sample` are the EPR-018..023 arguments the shared callers
        # now always pass; this arm ignores them (its conditioning is the K
        # query-token rows of `h_where`).  Accepting and dropping them keeps the
        # signature compatible without changing a single value it computes.
        if self.arm != "UNIQ":
            return super().forward_geo(
                feat, cond, phi_dir, sim=sim, center=center, geom=None,
                valid_grid=valid_grid, guide_hi=guide_hi, grid_h=grid_h,
                grid_w=grid_w, h_where=h_where, h_mask=h_mask)
        import torch.nn.functional as F

        from q3vl.whereb.fields import no_autocast

        if h_where is None or h_where.shape[1] < self._query_rows():
            raise ValueError("UNIQ4 needs h_where carrying the K query-token "
                             "hiddens as its last rows (QueryTokVLM seam)")
        extra = self._extra(sim, center, None)
        with no_autocast(feat.device.type):
            u = self.geo(feat.float(),
                         None if extra is None else extra.float(),
                         cond.float(), h_where, h_mask)
            sel = self.geo.select_index(u)
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
