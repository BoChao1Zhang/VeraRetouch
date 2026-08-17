"""K-Net style gated kernel-update stages on top of UNIQ (EPR-014) -- NEW FILE.

Stage 0 is the existing head verbatim: ``q`` from the VLM's query-token rows,
one dot product against the tower code, tanh-limited.  Each additional stage
then does what K-Net's ``KernelUpdateHead`` does -- binarise the previous
fields, aggregate the pixel code they select into a per-query group feature,
fuse it with the old query through the two sigmoid gates of ``KernelUpdator``,
let the queries talk to each other through a MultiheadAttention, and re-emit
fields with the SAME ``to_mask`` (K-Net's ``conv_kernel_size=1`` makes its
per-kernel ``F.conv2d`` and this dot product the same operation).

Reference (opened 2026-08-13, commit 5e50ee58957dce972f51096804ff69171c2f072e
of github.com/ZwwWayne/K-Net):

* ``knet/kernel_updator.py`` L36-42 (the four Linears), L46-49 (the four
  LayerNorms), L56-94 (the gate arithmetic reproduced term by term below);
* ``knet/det/kernel_update_head.py`` L185-197 (sigmoid, ``> hard_mask_thr``,
  ``einsum('bnhw,bchw->bnc')``), L102-103 + L206-216 (kernel-to-kernel
  attention then FFN), L228/L246-259 (``fc_mask`` + per-kernel conv);
* ``knet/det/kernel_iter_head.py`` L177-231 (per-stage full loss, L181 no
  detach between stages), L114-116 (``recursive=False`` -> per-stage params);
* ``configs/det/_base_/models/knet_s3_r50_fpn.py`` L1-3 (num_stages=3,
  conv_kernel_size=1), L69 (``stage_loss_weights=[1]*3``), L90-97 (updator
  in=feat=out width).

Two NOVEL deviations, both recorded in the proposal's initialisation row and
in ``experiments/prs/IMPL_NOTES_epr012-016.md``:

1. **residual + zero-initialised exit** instead of K-Net's replacing update
   (K-Net xavier-inits, ``kernel_update_head.py`` L153-170).  Required by the
   task card: with ``--uniq5-stages S > 0`` the step-0 fields must still be
   the baseline's.
2. **pre-norm** on the attention/FFN sub-layers rather than K-Net's post-norm
   (``obj_feat = self.attention_norm(...)``).  Post-norm makes
   ``LayerNorm(q + 0) != q``, which would defeat (1); this is the same reason
   EPR-013's refine layer is pre-norm.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .uniq4 import UniQ4Head

__all__ = ["KernelUpdate", "UniQ5Head"]


class KernelUpdate(nn.Module):
    """One K-Net kernel-update stage (``KernelUpdator`` + attention + FFN)."""

    def __init__(self, ch: int = 128, n_heads: int = 8,
                 query_attn: bool = True, ffn_mult: int = 2):
        super().__init__()
        # KernelUpdator L36-42
        self.dynamic_layer = nn.Linear(ch, 2 * ch)
        self.input_layer = nn.Linear(ch, 2 * ch)
        self.input_gate = nn.Linear(ch, ch)
        self.update_gate = nn.Linear(ch, ch)
        # both gates start neutral: LayerNorm(0) = 0 -> sigmoid(0) = 0.5
        for g in (self.input_gate, self.update_gate):
            nn.init.zeros_(g.weight)
            nn.init.zeros_(g.bias)
        # KernelUpdator L46-49: one LayerNorm per gate and per fused branch
        self.input_norm_in = nn.LayerNorm(ch)    # -> input_gate
        self.norm_in = nn.LayerNorm(ch)          # -> update_gate
        self.norm_out = nn.LayerNorm(ch)         # -> param_out
        self.input_norm_out = nn.LayerNorm(ch)   # -> input_out
        self.fc_layer = nn.Linear(ch, ch)
        self.fc_norm = nn.LayerNorm(ch)
        #: NOVEL: zero-initialised residual exit (see module docstring)
        self.out_proj = nn.Linear(ch, ch)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        # kernel_update_head.py L102-103 / L206-216
        self.attn = None
        if query_attn:
            self.norm_a = nn.LayerNorm(ch)
            self.attn = nn.MultiheadAttention(ch, n_heads, batch_first=True)
            nn.init.zeros_(self.attn.out_proj.weight)
            nn.init.zeros_(self.attn.out_proj.bias)
        self.norm_ffn = nn.LayerNorm(ch)
        self.ffn = nn.Sequential(nn.Linear(ch, ffn_mult * ch), nn.GELU(),
                                 nn.Linear(ffn_mult * ch, ch))
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, q: torch.Tensor, code: torch.Tensor,
                feedback: torch.Tensor) -> torch.Tensor:
        """``q`` (rows, ch), ``code`` (ch, N), ``feedback`` (rows, N) in [0,1].

        The feedback is K-Net's binarised mask; ``einsum('kn,cn->kc')`` is the
        same **sum** aggregation as its ``einsum('bnhw,bchw->bnc')`` -- no
        normalisation, so an all-zero field (foreign-instruction samples, and
        every field at step 0) contributes exactly the zero vector instead of
        dividing by zero.
        """
        ch = q.shape[-1]
        x_feat = feedback @ code.transpose(0, 1)                # (rows, ch)
        p_in, p_out = self.dynamic_layer(x_feat).split(ch, dim=-1)
        i_in, i_out = self.input_layer(q).split(ch, dim=-1)
        gate_feats = i_in * p_in                                # KU L70
        ig = torch.sigmoid(self.input_norm_in(self.input_gate(gate_feats)))
        ug = torch.sigmoid(self.norm_in(self.update_gate(gate_feats)))
        feats = (ug * self.norm_out(p_out)
                 + ig * self.input_norm_out(i_out))            # KU L87-88
        upd = F.relu(self.fc_norm(self.fc_layer(feats)))              # KU L90-92
        q = q + self.out_proj(upd)                              # NOVEL residual
        if self.attn is not None:
            h = self.norm_a(q).unsqueeze(0)
            att, _ = self.attn(h, h, h, need_weights=False)
            q = q + att[0]
        return q + self.ffn(self.norm_ffn(q))


class UniQ5Head(UniQ4Head):
    """``UniQ4Head`` + ``n_stages`` kernel-update rounds.

    ``n_stages == 0`` constructs nothing and leaves the forward on the frozen
    ST_LANG path.  With stages on, ``s_all`` / ``cls_logits`` / ``sel_logits``
    all refer to the LAST stage (so ``forward_geo`` and ``evaluate`` need no
    change at all) and stages ``0 .. S-1`` are handed to the trainer through
    ``aux_supervision`` for K-Net's per-stage full-loss supervision.
    """

    def __init__(self, *args, n_stages: int = 0, hard_thr: float = 0.5,
                 soft_feedback: bool = False, per_stage_to_mask: bool = False,
                 stage_query_attn: bool = True, stage_heads: int = 8,
                 stage_supervision: bool = True,
                 stage_loss_weights: list[float] | None = None, **kw):
        super().__init__(*args, **kw)
        ch = self.queries.shape[1]
        self.n_stages = int(n_stages)
        self.hard_thr = float(hard_thr)
        self.soft_feedback = bool(soft_feedback)
        self.per_stage_to_mask = bool(per_stage_to_mask)
        self.stage_supervision = bool(stage_supervision)
        self.stage_loss_weights = [float(x) for x in (
            stage_loss_weights or [1.0] * self.n_stages)]
        self.stages = nn.ModuleList()
        self.stage_to_mask = nn.ModuleList()
        if self.n_stages:
            if len(self.stage_loss_weights) != self.n_stages:
                raise ValueError("stage_loss_weights must have n_stages entries")
            for _ in range(self.n_stages):
                self.stages.append(KernelUpdate(ch, stage_heads,
                                                query_attn=stage_query_attn))
            if self.per_stage_to_mask:          # ablation: unshared read-out
                pix_dim = self.to_mask.out_features - 1
                for _ in range(self.n_stages):
                    lin = nn.Linear(ch, pix_dim + 1)
                    nn.init.zeros_(lin.weight)
                    nn.init.zeros_(lin.bias)
                    self.stage_to_mask.append(lin)

    def _post_query(self, q, pix, gh, gw, aux):
        if not self.n_stages:
            return q, aux, None
        ch = self.queries.shape[1]
        code = pix[:ch]                                   # tower code, (ch, N)
        rows = q.shape[0]
        head = None
        for i in range(self.n_stages):
            s_prev = self._field(q, pix, head)            # (rows, N)
            if self.stage_supervision:
                aux.append({
                    "s_all": s_prev.reshape(rows, gh, gw),
                    "cls_logits": self.cls(q),
                    "sel_logits": self.sel(q[:self.n_queries]).reshape(-1),
                    "weight": self.stage_loss_weights[i], "tag": f"st{i}"})
            fb = self.mask_of(s_prev)
            if not self.soft_feedback:
                # KUH L192-194; the hard threshold carries no gradient, which
                # is exactly why K-Net does not detach between stages (L181).
                fb = (fb > self.hard_thr).to(fb.dtype)
            q = self.stages[i](q, code, fb)
            head = self.stage_to_mask[i] if self.per_stage_to_mask else None
        return q, aux, head

    def facts(self) -> dict[str, Any]:
        return {**super().facts(), "n_stages": self.n_stages,
                "hard_mask_thr": self.hard_thr,
                "stage_feedback": "soft" if self.soft_feedback else "hard",
                "per_stage_to_mask": self.per_stage_to_mask,
                "stage_supervision": self.stage_supervision,
                "stage_loss_weights": self.stage_loss_weights,
                "stage_query_attn": bool(
                    self.n_stages and self.stages[0].attn is not None)}
