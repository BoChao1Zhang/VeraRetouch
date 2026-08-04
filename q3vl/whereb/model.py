"""Protocol 5.2 -- the four MetaCanvas structures, assembled into one module.

| ID                | query layout                       | parameter readout                     |
|-------------------|------------------------------------|---------------------------------------|
| `MC8-Joint`       | 8x8, 64 queries                    | one attention pool + joint head       |
| `MC16-Joint`      | 16x16, 256 queries                 | same, tests canvas resolution         |
| `MC16-SplitHead`  | 16x16, shared canvas               | separate pools + heads for `w`, `rho` |
| `MC16-DualCanvas` | two independent 16x16 bank/streams | one predicts `w`, one predicts `rho`  |

The dual-canvas arm shares *only the frozen VLM* (protocol 5.2), so its two
streams own separate query banks, separate ``H_where``/``F_pre`` projections,
separate position encoders and separate blocks.

Protocol 14.9 ("no main arm's input contains ``I_tar``, GT mask, GT LUT or
oracle latent") is enforced structurally: :meth:`WhereBModel.forward` accepts
five tensors and none of them can be any of those things.  The mask and the
oracle latent only ever appear in :mod:`q3vl.whereb.losses`, downstream of the
model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from q3vl.where.basis import Latent, alpha_of, w_dir_of
from q3vl.where.readout import param_shapes

from .config import ArmConfig, STRUCTURES
from .connector import ConnectorStream
from .heads import LatentHeads
from .qwhere import MetaCanvasQueryBank

__all__ = ["WhereBOutput", "WhereBModel", "MODEL_INPUT_KEYS", "parameter_table"]

#: the *only* keys a batch may hand to :meth:`WhereBModel.forward`
MODEL_INPUT_KEYS = ("f_pre", "f_pre_pos", "f_pre_mask", "h_where", "h_where_mask")


@dataclass
class WhereBOutput:
    w0: torch.Tensor              # (B,)
    w_raw: torch.Tensor           # (B, 71)
    alpha_raw: torch.Tensor       # (B,)
    rho: dict[str, torch.Tensor]  # readout raw params, batched
    canvas_axis: torch.Tensor     # (B, Nq, C) -- Q_axis for protocol 6
    canvas_rho: torch.Tensor      # (B, Nq, C) -- Q_readout for protocol 6
    readout: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def w_dir(self) -> torch.Tensor:
        return w_dir_of(self.w_raw)

    @property
    def alpha(self) -> torch.Tensor:
        return alpha_of(self.alpha_raw)

    def __len__(self) -> int:
        return int(self.w0.shape[0])

    def select(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "w0": self.w0[i], "w_raw": self.w_raw[i], "alpha_raw": self.alpha_raw[i],
            **{k: v[i] for k, v in self.rho.items()},
        }

    def latent(self, i: int) -> Latent:
        """One sample's prediction as a Where-A :class:`Latent` (for eval/packing)."""
        return Latent(
            self.readout, self.w0[i].detach(), self.alpha_raw[i].detach(),
            self.w_raw[i].detach(),
            {k: self.rho[k][i].detach() for k in param_shapes(self.readout)},
        )


class WhereBModel(nn.Module):
    """``Q_where`` + connector + pools + heads.  The VLM is *not* part of this."""

    def __init__(self, cfg: ArmConfig):
        super().__init__()
        self.cfg = cfg
        self.arm = cfg.arm
        self.structure = cfg.structure
        self.readout = cfg.readout
        self.canvas = cfg.canvas
        self.n_streams = cfg.n_streams
        ccfg = cfg.connector

        self.banks = nn.ModuleList(
            MetaCanvasQueryBank(cfg.canvas, ccfg.dim, seed=cfg.seed + 10 * i)
            for i in range(self.n_streams)
        )
        self.streams = nn.ModuleList(ConnectorStream(ccfg) for _ in range(self.n_streams))
        self.heads = LatentHeads(
            ccfg.dim, ccfg.n_heads, cfg.readout, cfg.n_pools, seed=cfg.seed + 7
        )

    # -- forward ------------------------------------------------------------
    def forward(
        self,
        f_pre: torch.Tensor,                        # (B, P, 1024)
        f_pre_pos: torch.Tensor,                    # (B, P, 2) true aspect ratio
        f_pre_mask: torch.Tensor | None,            # (B, P) bool
        h_where: torch.Tensor,                      # (B, T, 2560)
        h_where_mask: torch.Tensor | None,          # (B, T) bool
    ) -> WhereBOutput:
        b = f_pre.shape[0]
        canvases = []
        for bank, stream in zip(self.banks, self.streams):
            q = bank(b, stream.pos)
            canvases.append(
                stream(q, h_where, h_where_mask, f_pre, f_pre_pos, f_pre_mask)
            )
        canvas_axis = canvases[0]
        canvas_rho = canvases[1] if len(canvases) > 1 else canvases[0]
        w0, w_raw, alpha_raw, rho = self.heads(canvas_axis, canvas_rho)
        return WhereBOutput(
            w0=w0, w_raw=w_raw, alpha_raw=alpha_raw, rho=rho,
            canvas_axis=canvas_axis, canvas_rho=canvas_rho, readout=self.readout,
        )

    # -- reporting ----------------------------------------------------------
    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def gate_values(self) -> dict[str, Any]:
        return {f"stream{i}": s.gate_values() for i, s in enumerate(self.streams)}

    def facts(self) -> dict[str, Any]:
        groups: dict[str, int] = {}
        for name, p in self.named_parameters():
            top = name.split(".")[0]
            groups[top] = groups.get(top, 0) + p.numel()
        return {
            "arm": self.arm,
            "structure": self.structure,
            "readout": self.readout,
            "canvas": self.canvas,
            "n_queries": self.canvas * self.canvas,
            "n_streams": self.n_streams,
            "n_pools": self.cfg.n_pools,
            "n_trainable_params": self.n_trainable(),
            "params_by_group": groups,
            "connector": {
                "dim": self.cfg.connector.dim,
                "blocks": self.cfg.connector.n_blocks,
                "heads": self.cfg.connector.n_heads,
                "ffn": self.cfg.connector.ffn,
            },
        }


def parameter_table(arms: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    """Trainable-parameter inventory for the structures (protocol 13.1)."""
    from .config import ARMS, arm_config

    arms = arms or tuple(ARMS)
    rows = []
    for a in arms:
        m = WhereBModel(arm_config(a))
        f = m.facts()
        rows.append({
            "arm": a, "structure": f["structure"], "readout": f["readout"],
            "n_queries": f["n_queries"], "n_streams": f["n_streams"],
            "n_pools": f["n_pools"], "n_trainable_params": f["n_trainable_params"],
            "params_by_group": f["params_by_group"],
        })
    return rows


def structure_of(arm: str) -> dict[str, int | str]:
    from .config import ARMS

    return STRUCTURES[ARMS[arm][0]]
