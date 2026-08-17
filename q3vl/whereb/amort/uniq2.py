"""UNIQ wave-3 variants (EPR-011 arms P / PIXQ / FQ) -- NEW FILE ONLY.

Freeze contract (user ruling 2026-08-12): arms A-E were queued against the
amort package as it stood, and files a queued/running payload will import must
not change.  Everything here therefore lives in NEW modules and is wired in
via **documented module-attribute seams** installed by
``scripts/run_uniq2_arm.py`` before it delegates to the frozen entry:

    q3vl.whereb.amort.model.AmortModel   -> AmortModelV2   (subclass)
    q3vl.whereb.amort.losses.uniq_wta_loss -> dispatching_wta (wraps original)

Both frozen call sites bind these names at CALL time (imports inside
functions), which is what makes the seam sufficient without editing them.

The three variants, each single-factor on top of arm B (K=4, no Fourier):

* ``presence``  -- SAM 3 (arXiv:2511.16719, verified): "Recognition and
  localization are decoupled with a presence head, which boosts detection
  accuracy."  Here: one scalar ``p = sigmoid(Linear(mean(query states)))``
  multiplies every query mask.  No new loss term is needed: the existing
  empty-mask term on foreign instructions pushes ``p`` toward 0 and the BCE on
  real samples pushes it toward 1, so the scalar path specialises in "is
  anything referred to" and the field specialises in "where".
* ``pix_attn``  -- Mask2Former (arXiv:2112.01527, verified) query-to-pixel
  cross-attention, in its unmasked one-layer form; zero-initialised output
  projection so step 0 is identical to the base head.
* ``family_queries`` -- OMG-Seg (arXiv:2401.10229, verified) task-specific
  queries: query k is DEDICATED to family k (winner forced to the
  construction-side family label instead of free WTA argmin).

Statefulness note (the one ugly corner, priced consciously): ``mask_of`` is
called by the frozen trainer/eval as a bare ``model.geo.mask_of`` with no
sample context, so the presence gate rides on ``self._presence`` set by the
immediately preceding ``forward``.  The whole package runs strictly
per-sample (B=1 loop), which is the contract that makes this safe; a
``mask_of`` call before any ``forward`` raises rather than guessing.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from q3vl.where.config import S_SCALE

from . import losses as _losses
from .model import AmortModel as _AmortModelBase
from .uniq import UNIQ_FAMILIES, UniQHead

__all__ = ["VARIANT", "UniQ2Head", "AmortModelV2", "dispatching_wta"]

#: process-wide variant switches; exactly one arm variant runs per process.
#: Set by run_uniq2_arm BEFORE any model is built; recorded in
#: config/uniq2_setup.json and in model.facts().
VARIANT: dict[str, bool] = {"presence": False, "pix_attn": False,
                            "family_queries": False}

#: the genuine original, captured at import time -- the seam replaces the
#: module attribute, never this reference
_ORIG_WTA = _losses.uniq_wta_loss


class UniQ2Head(UniQHead):
    """UniQHead + (presence | pix_attn | family_queries), all zero-init clean.

    ``s_all`` is exactly 0 at step 0 for every variant (``to_mask`` zero-init,
    pix-attn out-proj zero-init).  The presence gate starts at sigmoid(2)
    ~= 0.88 rather than 1.0 -- these arms train from scratch, so strict
    step-0 mask equality with the base head is not load-bearing; the bias
    init just keeps early BCE off the p ~= 0.5 saddle.
    """

    def __init__(self, *args, presence: bool = False, pix_attn: bool = False,
                 family_queries: bool = False, **kw):
        super().__init__(*args, **kw)
        ch = self.queries.shape[1]
        self.presence_on = bool(presence)
        self.pix_attn_on = bool(pix_attn)
        self.family_queries = bool(family_queries)
        self._presence: torch.Tensor | None = None
        if self.family_queries and self.n_queries != len(UNIQ_FAMILIES):
            raise ValueError(
                f"family_queries dedicates query k to family k and therefore "
                f"needs n_queries == {len(UNIQ_FAMILIES)}, got {self.n_queries}")
        if self.presence_on:
            self.presence = nn.Linear(ch, 1)
            nn.init.zeros_(self.presence.weight)
            with torch.no_grad():
                self.presence.bias.fill_(2.0)
        if self.pix_attn_on:
            self.q_norm2 = nn.LayerNorm(ch)
            self.pix_xattn = nn.MultiheadAttention(ch, num_heads=4,
                                                   batch_first=True)
            nn.init.zeros_(self.pix_xattn.out_proj.weight)
            nn.init.zeros_(self.pix_xattn.out_proj.bias)
        if self.family_queries:
            g = torch.Generator(device="cpu").manual_seed(self.fourier_seed + 1)
            self.fam_embed = nn.Parameter(
                torch.randn(len(UNIQ_FAMILIES), ch, generator=g) * 0.02)

    def forward(self, feat: torch.Tensor, extra: torch.Tensor | None,
                cond: torch.Tensor | None, h_where: torch.Tensor,
                h_mask: torch.Tensor | None) -> dict[str, Any]:
        codes = self.tower(feat, extra, cond)                   # (1, ch, gh, gw)
        gh, gw = codes.shape[-2:]
        pix = codes[0].reshape(codes.shape[1], gh * gw)
        if self.fourier_bands:
            four = self._fourier(gh, gw, codes.device, pix.dtype)
            pix = torch.cat([pix, four.reshape(four.shape[0], gh * gw)], dim=0)

        kv = self.text_proj(h_where.float())
        pad = None
        if h_mask is not None:
            pad = h_mask.reshape(1, -1) < 0.5
            if bool(pad.all()):
                pad = None
        base = self.queries
        if self.family_queries:
            base = base + self.fam_embed
        qb = base.unsqueeze(0)
        att, _ = self.xattn(qb, kv, kv, key_padding_mask=pad, need_weights=False)
        q = qb + att
        q = q + self.ffn(self.q_norm(q))
        if self.pix_attn_on:
            # pixel tokens are the raw tower codes -- the Fourier channels are
            # an OUTPUT basis, not content, and have no business as keys
            ptok = codes[0].reshape(codes.shape[1], -1).transpose(0, 1).unsqueeze(0)
            att2, _ = self.pix_xattn(self.q_norm2(q), ptok, ptok,
                                     need_weights=False)
            q = q + att2
        q = q[0]                                                # (K, ch)

        wb = self.to_mask(q)
        raw = wb[:, :-1] @ pix + wb[:, -1:]
        s_all = (S_SCALE * torch.tanh(raw / S_SCALE)).reshape(
            self.n_queries, gh, gw)
        if self.presence_on:
            self._presence = torch.sigmoid(
                self.presence(q.mean(0, keepdim=True)))[0, 0]
        return {"s_all": s_all, "cls_logits": self.cls(q),
                "sel_logits": self.sel(q).reshape(-1)}

    def mask_of(self, s_field: torch.Tensor) -> torch.Tensor:
        m = torch.sigmoid(self.gain * s_field)
        if self.presence_on:
            if self._presence is None:
                raise RuntimeError(
                    "presence gate consumed before any forward set it; the "
                    "per-sample forward->loss contract was broken")
            m = self._presence * m
        return m

    def facts(self) -> dict[str, Any]:
        return {**super().facts(), "variant": {
            "presence": self.presence_on, "pix_attn": self.pix_attn_on,
            "family_queries": self.family_queries}}


class AmortModelV2(_AmortModelBase):
    """AmortModel whose UNIQ arm builds a :class:`UniQ2Head` per ``VARIANT``.

    Same constructor signature; installed over the frozen entry's name via the
    run_uniq2_arm seam.  Non-UNIQ arms are untouched.
    """

    def __init__(self, arm: str = "P1", **kw):
        super().__init__(arm, **kw)
        if arm != "UNIQ":
            return
        film_dim = self.cond.out_dim if self.use_film else None
        extra_ch = (int(self.use_sim_field)
                    + int(self.use_center_prior_channel) + self.geom_dim)
        self.geo = UniQ2Head(
            kw.get("ch", 128), kw.get("in_dim", 1024), extra_ch,
            kw.get("n_blocks", 6), film_dim,
            text_dim=kw.get("cond_text_dim", 2560),
            n_queries=kw.get("uniq_k", 4),
            fourier_bands=kw.get("uniq_fourier_bands", 0),
            fourier_scale=kw.get("uniq_fourier_scale", 1.0),
            seed=kw.get("seed", 0),
            presence=VARIANT["presence"], pix_attn=VARIANT["pix_attn"],
            family_queries=VARIANT["family_queries"])


def dispatching_wta(s_all, mask_of, gt, w, *, valid=None, phi_sdf=None,
                    gt_partner=None, is_fake=False, structural=False,
                    family="", cls_logits=None, sel_logits=None):
    """Seam target for ``losses.uniq_wta_loss``.

    family_queries off (or fake / unlabelled family) -> byte-identical to the
    original.  On -> the winner is FORCED to the construction-side family
    index (OMG-Seg task-specific queries), so query k only ever trains on
    family k; cls/sel supervision targets follow the forced winner.
    """
    if (not VARIANT["family_queries"]) or is_fake or family not in UNIQ_FAMILIES:
        return _ORIG_WTA(s_all, mask_of, gt, w, valid=valid, phi_sdf=phi_sdf,
                         gt_partner=gt_partner, is_fake=is_fake,
                         structural=structural, family=family,
                         cls_logits=cls_logits, sel_logits=sel_logits)
    j = UNIQ_FAMILIES.index(family)
    per_j = _losses.amort_sample_loss(
        mask_of(s_all[j]), gt, w, valid=valid, phi_sdf=phi_sdf,
        gt_partner=gt_partner, is_fake=False, structural=structural)
    total = per_j.total
    terms = dict(per_j.terms)
    stats = dict(per_j.stats)
    if cls_logits is not None and w.uniq_cls:
        tgt = torch.tensor([j], device=cls_logits.device)
        terms["uniq_cls"] = F.cross_entropy(cls_logits[j:j + 1], tgt)
        total = total + w.uniq_cls * terms["uniq_cls"]
    if sel_logits is not None and w.uniq_sel:
        tgt = torch.tensor([j], device=sel_logits.device)
        terms["uniq_sel"] = F.cross_entropy(sel_logits.reshape(1, -1), tgt)
        total = total + w.uniq_sel * terms["uniq_sel"]
    with torch.no_grad():
        stats["uniq_winner"] = float(j)
        if sel_logits is not None:
            stats["uniq_sel_correct"] = float(int(sel_logits.argmax()) == j)
    return _losses.AmortLoss(total=total, terms=terms, stats=stats)
