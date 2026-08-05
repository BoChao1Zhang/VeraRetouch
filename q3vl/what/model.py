"""Stage-What assembled: ``Q_color`` -> ``z_style`` -> 48 Gaussians -> ``T_pred``.

    frozen Qwen3-VL + frozen Where + independent Q_color
      -> continuous z_style in R^1024
      -> 48-Gaussian LUT function T_pred
      -> standard 33^3 LUT                                     (protocol 0)

Protocol 14.9 -- "prove that no main arm's input contains ``I_tar``, a GT mask, a
GT LUT or an oracle latent" -- is enforced by the signature, not by a runtime
scan: :meth:`WhatModel.forward` accepts five tensors plus one
:class:`~q3vl.what.wc.WhereSignals`, and ``WhereSignals`` has no field that could
hold any of those four things.  The GT LUT and the GT mask appear only in
:mod:`q3vl.what.losses` and :mod:`q3vl.what.metrics`, downstream of the model;
``I_tar`` appears in neither -- protocol 9.5, "``I_tar`` does not enter
``L_what``" -- and only the final image metric of protocol 12.2 ever loads it.

The two ceiling arms ``C03``/``C04`` are the one exception the protocol itself
carves out: their ``WhereSignals`` carries the GT mask and the Where-A oracle
latent.  They are constructed through the same class, they are tagged
``is_ceiling``, and :func:`q3vl.what.evaluate.main_board` refuses to rank them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from .backend import SlotBackend  # noqa: F401  (documents the shared backend)
from .color import ColorStack
from .config import ArmConfig, BAKE_SIZE
from .gaussians import bake, parameter_report, render
from .pooling import VisionProjector, aligned_pool
from .wc import WCEncoder, WhereSignals
from .generator import build_generator

__all__ = ["MODEL_INPUT_KEYS", "WhatOutput", "WhatModel"]

#: the *only* keys a batch may hand to :meth:`WhatModel.forward`
MODEL_INPUT_KEYS = ("h_color", "h_color_mask", "f_pre", "f_pre_mask", "rgb_low",
                    "where")


@dataclass
class WhatOutput:
    params: dict[str, torch.Tensor]        # the constrained renderer parameters
    z_style: torch.Tensor                  # (B, 1024)
    m_color: torch.Tensor                  # (B, 16, 512)
    slots: torch.Tensor                    # (B, 48, 512) final slot features
    extra: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.z_style.shape[0])

    def render(self, x: torch.Tensor, cfg) -> torch.Tensor:
        return render(self.params, x, cfg)

    def bake(self, cfg, size: int = BAKE_SIZE) -> torch.Tensor:
        return bake(self.params, cfg, size)

    def report(self) -> dict[str, Any]:
        return parameter_report(self.params)


class WhatModel(nn.Module):
    """One Stage-What arm.  The VLM and the Where checkpoint are *not* part of it."""

    def __init__(self, cfg: ArmConfig):
        super().__init__()
        self.cfg = cfg
        self.arm = cfg.arm
        self.color = ColorStack(cfg.color, seed=cfg.seed)
        self.vision = VisionProjector()
        self.wc = WCEncoder(cfg, dim=cfg.backend.dim, seed=cfg.seed + 11)
        self.generator = build_generator(cfg)
        #: review nit N-3: the pooling diagnostics cost one device sync each and
        #: are only read on logging steps.  The trainer flips this per step.
        self.collect_pool_stats = True

    # -- forward ------------------------------------------------------------
    def forward(
        self,
        h_color: torch.Tensor,                   # (B, T, 2560)
        h_color_mask: torch.Tensor | None,        # (B, T) bool
        f_pre: torch.Tensor,                      # (B, P, 1024)
        f_pre_mask: torch.Tensor | None,          # (B, P) bool
        rgb_low: torch.Tensor,                    # (B, P, 3) I_in on the F_pre grid
        where: WhereSignals,
    ) -> WhatOutput:
        m_color, z_style = self.color(h_color, h_color_mask)
        wc_tokens, wc_mask = self.wc(f_pre, f_pre_mask, where)
        v_feat = self.vision(f_pre)
        m_pred = where.m_low if self.cfg.mask_pool else None

        def pool_fn(geom):
            return aligned_pool(geom, rgb_low, v_feat, m_pred=m_pred,
                                valid=f_pre_mask,
                                collect_stats=self.collect_pool_stats)

        # M_color is 16 dense query outputs: every one of them is valid, so the
        # backend's cross-attention needs no key mask.  The *language* mask
        # (h_color_mask) was already consumed inside the colour connector and
        # must not be reused here -- its length is T, not 16.
        params, extra = self.generator(
            z_style, m_color, None, wc_tokens, wc_mask, pool_fn,
        )
        return WhatOutput(
            params=params, z_style=z_style, m_color=m_color,
            slots=extra.pop("slots"), extra=extra,
        )

    # -- reporting ----------------------------------------------------------
    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def gate_values(self) -> dict[str, Any]:
        return {"color": self.color.connector.gate_values(),
                "backend": self.generator.backend.gate_values()}

    def facts(self) -> dict[str, Any]:
        groups: dict[str, int] = {}
        for name, p in self.named_parameters():
            if p.requires_grad:
                top = name.split(".")[0]
                groups[top] = groups.get(top, 0) + p.numel()
        return {
            "arm": self.arm,
            "wc": self.cfg.wc,
            "generator": self.cfg.generator,
            "where_source": self.cfg.where_source,
            "is_ceiling": self.cfg.is_ceiling,
            "mask_pool": self.cfg.mask_pool,
            "where_prefix": self.cfg.where_prefix,
            "n_trainable_params": self.n_trainable(),
            "params_by_group": groups,
            "color": self.color.facts(),
            "wc_encoder": self.wc.facts(),
            "generator_facts": self.generator.facts(),
            "sigma_param": self.cfg.lut.sigma_param,
            "global_affine_mode": self.cfg.lut.global_affine_mode,
        }
