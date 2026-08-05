"""Protocol 6 -- the four Where-to-What interfaces, and the two controls.

    F_roi   = sum_p m_pred(p) F_pre(p) / (sum_p m_pred(p) + eps)
    F_bg    = sum_p (1-m_pred(p)) F_pre(p) / (sum_p (1-m_pred(p)) + eps)
    z_where = AttentionPool(Q_axis, Q_readout)

| ID   | what Stage-What is given                     | question                                  |
|------|----------------------------------------------|-------------------------------------------|
| WC-0 | H_color + global visual pooling; no Where out | how far does colour reasoning get alone   |
| WC-1 | WC-0 + m_pred, F_roi, F_bg                    | is dense region pooling enough            |
| WC-2 | WC-0 + z_where, w, rho                        | is the global spatial latent better       |
| WC-3 | WC-1 + WC-2                                   | are dense and latent complementary        |

Everything a Stage-What arm may learn about Where passes through this module, so
the arm difference is exactly the token set built here plus one boolean
(``mask_pool``).  ``WC-0`` deliberately does **not** get ``m_pred``: the mask is
a Where output, and protocol 6 gives ``WC-0`` "no explicit Where output".  What
separates ``WC-0`` (``T01``/``T05``) from the strict no-where control
(``C01``/``C02``) is only the ``<where>`` prefix in the language sequence, which
the causal hidden states of ``<color>`` can still see -- exactly the residual
protocol 6's last paragraph warns about and protocol 8.2 controls for.

``C03``/``C04`` replace the predicted mask with the GT mask and the predicted
``w, rho`` with the Where-A oracle latent.  There is no query state to pool in
that case, so ``z_where`` is produced by :class:`OracleLatentEncoder` from the
oracle latent itself; this is a ceiling control that never enters the main board
(protocol 8.2), and the substitution is recorded in the arm facts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from q3vl.whereb.heads import rho_numel

from .attention import AttentionPool
from .config import (
    BACKEND_DIM,
    FPRE_DIM,
    PHI_DIR_DIM,
    WC_INTERFACES,
    ArmConfig,
)
from .pooling import global_visual_pool, roi_bg_pool

__all__ = ["WhereSignals", "WCEncoder", "OracleLatentEncoder", "wc_token_names"]

#: (w0, w_dir, alpha)
W_VECTOR_DIM = 1 + PHI_DIR_DIM + 1


def wc_token_names(arm_cfg: ArmConfig) -> tuple[str, ...]:
    return arm_cfg.wc_tokens


@dataclass
class WhereSignals:
    """Everything the frozen Where checkpoint hands to Stage-What.

    Deliberately *not* a superset of the Where output: there is no field here for
    ``I_tar``, for the GT LUT or for anything else protocol 14.9 forbids, and
    :meth:`assert_allowed` checks the arm is only given what its interface names.
    """

    m_low: torch.Tensor | None = None        # (B, P) predicted mask, F_pre grid
    #: one tensor per sample, NOT stacked: each image has its own spec-5 grid, so
    #: they cannot share a batch axis (review nit N-4).
    m_hi: list[torch.Tensor] | None = None   # [(H_i, W_i)] for the natural sampler
    canvas_axis: torch.Tensor | None = None  # (B, Nq, 512) Q_axis final state
    canvas_rho: torch.Tensor | None = None   # (B, Nq, 512) Q_readout final state
    w_vec: torch.Tensor | None = None        # (B, 73) = [w0, w_dir, alpha]
    rho_vec: torch.Tensor | None = None      # (B, n_rho) raw readout parameters
    source: str = "predicted"                # predicted | oracle | none
    meta: dict[str, Any] = field(default_factory=dict)

    def assert_allowed(self, arm_cfg: ArmConfig) -> None:
        need_dense = arm_cfg.mask_pool or any(
            t in ("f_roi", "f_bg") for t in arm_cfg.wc_tokens)
        need_latent = any(t in ("z_where", "w", "rho") for t in arm_cfg.wc_tokens)
        if need_dense and self.m_low is None:
            raise ValueError(f"{arm_cfg.arm} ({arm_cfg.wc}) needs m_pred but got none")
        if need_latent and self.w_vec is None:
            raise ValueError(f"{arm_cfg.arm} ({arm_cfg.wc}) needs w/rho but got none")
        if not need_dense and not need_latent and arm_cfg.where_source == "none":
            return
        if arm_cfg.where_source == "none" and (need_dense or need_latent):
            raise AssertionError(
                f"{arm_cfg.arm} is a strict no-where control but its interface "
                f"{arm_cfg.wc} asks for Where outputs"
            )


class OracleLatentEncoder(nn.Module):
    """``(w*, rho*) -> a 512-d stand-in for ``z_where`` (ceiling control only).

    ``z_where`` is defined in protocol 6 as a pool over the MetaCanvas query
    states, which do not exist when the latent comes from the Where-A L-BFGS fit.
    Rather than quietly feeding the *predicted* query states into an arm labelled
    "oracle" -- which would make the ceiling not a ceiling -- ``C03``/``C04`` get
    an encoding of the oracle latent itself.
    """

    def __init__(self, n_rho: int, dim: int = BACKEND_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(W_VECTOR_DIM + n_rho, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, w_vec: torch.Tensor, rho_vec: torch.Tensor) -> torch.Tensor:
        return self.norm(self.mlp(torch.cat([w_vec, rho_vec], dim=-1)))


class WCEncoder(nn.Module):
    """Builds one arm's WC token set -> ``(B, T, 512)`` plus its mask.

    Each token kind gets its own projection and its own learnable type embedding,
    so the backend can tell "this is the ROI pool" from "this is the background
    pool"; without the type embedding the two are the same linear map of two
    different vectors and the ablation between ``WC-1`` and ``WC-0`` would be
    confounded by the backend having to guess.
    """

    KINDS = ("global_vis", "f_roi", "f_bg", "z_where", "w", "rho")

    def __init__(self, arm_cfg: ArmConfig, dim: int = BACKEND_DIM, seed: int = 0):
        super().__init__()
        self.arm_cfg = arm_cfg
        self.dim = dim
        self.kinds = tuple(arm_cfg.wc_tokens)
        unknown = set(self.kinds) - set(self.KINDS)
        if unknown:
            raise ValueError(f"unknown WC token kinds {sorted(unknown)}")
        self.n_rho = rho_numel(arm_cfg.where_readout)

        self.proj = nn.ModuleDict()
        for k in self.kinds:
            if k in ("global_vis", "f_roi", "f_bg"):
                self.proj[k] = nn.Linear(FPRE_DIM, dim)
            elif k == "z_where":
                self.proj[k] = nn.Linear(dim, dim)
            elif k == "w":
                self.proj[k] = nn.Linear(W_VECTOR_DIM, dim)
            elif k == "rho":
                self.proj[k] = nn.Linear(self.n_rho, dim)
        self.type_emb = nn.Parameter(torch.zeros(len(self.kinds), dim))
        self.norm = nn.LayerNorm(dim)
        self.where_pool = (
            AttentionPool(dim, 8, seed=seed) if "z_where" in self.kinds else None
        )
        self.oracle_enc = (
            OracleLatentEncoder(self.n_rho, dim)
            if ("z_where" in self.kinds and arm_cfg.where_source == "oracle") else None
        )

    # -- forward ------------------------------------------------------------
    def forward(self, f_pre: torch.Tensor, valid: torch.Tensor | None,
                signals: WhereSignals) -> tuple[torch.Tensor, torch.Tensor]:
        signals.assert_allowed(self.arm_cfg)
        toks: list[torch.Tensor] = []
        for i, k in enumerate(self.kinds):
            toks.append(self._token(k, f_pre, valid, signals) + self.type_emb[i])
        tokens = self.norm(torch.stack(toks, dim=1))
        mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        return tokens, mask

    def _token(self, kind: str, f_pre: torch.Tensor, valid, s: WhereSignals
               ) -> torch.Tensor:
        if kind == "global_vis":
            return self.proj[kind](global_visual_pool(f_pre, valid))
        if kind in ("f_roi", "f_bg"):
            roi, bg = roi_bg_pool(f_pre, s.m_low, valid)
            return self.proj[kind](roi if kind == "f_roi" else bg)
        if kind == "z_where":
            if self.oracle_enc is not None:
                return self.oracle_enc(s.w_vec, s.rho_vec)
            if s.canvas_axis is None or s.canvas_rho is None:
                raise ValueError(
                    "z_where needs the Where model's Q_axis/Q_readout states; "
                    "an arm that cannot supply them must not request this token"
                )
            canvases = torch.cat([s.canvas_axis, s.canvas_rho], dim=1)
            return self.proj[kind](self.where_pool(canvases))
        if kind == "w":
            return self.proj[kind](s.w_vec)
        if kind == "rho":
            return self.proj[kind](s.rho_vec)
        raise ValueError(f"unknown WC token kind {kind!r}")

    # -- reporting ----------------------------------------------------------
    def facts(self) -> dict[str, Any]:
        return {
            "wc": self.arm_cfg.wc,
            "tokens": list(self.kinds),
            "n_tokens": len(self.kinds),
            "mask_pool": self.arm_cfg.mask_pool,
            "where_prefix": self.arm_cfg.where_prefix,
            "where_source": self.arm_cfg.where_source,
            "z_where_from": ("oracle_latent" if self.oracle_enc is not None
                             else ("query_states" if "z_where" in self.kinds else None)),
            "n_rho": self.n_rho,
            "n_params": sum(p.numel() for p in self.parameters()),
        }


def interface_table() -> list[dict[str, Any]]:
    """Protocol 6's table as data, for the report and the config matrix."""
    return [
        {"wc": k, "tokens": list(v["tokens"]), "mask_pool": v["mask_pool"],
         "where_prefix": v["where_prefix"]}
        for k, v in WC_INTERFACES.items()
    ]
