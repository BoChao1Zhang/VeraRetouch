"""Protocol 7.4 / 7.5 -- ``WHAT-FG48`` and ``WHAT-SB48``.

``WHAT-FG48`` (Full Generation) emits every parameter of all 48 Gaussians per
sample::

    mu 3 | Cholesky 6 | opacity 1 | existence 1 | M 9 | b 3   = 23 / primitive
    global G, b                                               = 12
    48 * 23 + 12 = 1116 parameters per image

    "The 2 seed blocks first predict a provisional ``mu, Sigma`` for the aligned
     pooling; the 4 later blocks then jointly refine all parameters.  The
     gradient can pass through the pooling back to the geometry."

``WHAT-SB48`` (Shared Backend/Geometry) shares 48 trainable ``mu_i, Sigma_i``
across samples -- initialised from the fixed ``4 x 4 x 3`` RGB anchors and
optimised over all training LUTs at ``shared_geometry_lr`` -- and emits only::

    opacity, existence, M, b   = 14 / primitive
    global G, b                = 12

    "The Transformer depth, width and query count of FG48 and SB48 are the same.
     By adjusting the decoder head bottleneck, the two arms' total trainable
     parameters differ by no more than 2%."

That last sentence is implemented, not asserted: :func:`solve_sb_bottleneck`
picks SB's bottleneck from FG's realised parameter count, and
``tests/test_generator.py`` checks the realised totals of all four (WC, generator)
pairs against the 2% bound.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import torch
import torch.nn as nn

from .backend import SlotBackend, SlotHeads, n_head_params
from .config import (
    ArmConfig,
    BackendConfig,
    FG_TOTAL_PER_SAMPLE,
    LutConfig,
    N_GEOMETRY,
    N_GLOBAL,
    N_PRIM_FG,
    N_PRIM_SB,
    N_SLOTS,
    PARAM_MATCH_TOLERANCE,
    SB_TOTAL_PER_SAMPLE,
)
from .gaussians import (
    GeometryBank,
    PRIM_LAYOUT_FG,
    PRIM_LAYOUT_SB,
    anchor_points,
    decode_geometry,
    decode_global,
    decode_primitives,
)

__all__ = ["ProvisionalGeometryHead", "FG48", "SB48", "build_generator",
           "generator_param_counts", "solve_sb_bottleneck", "parameter_matrix"]

PoolFn = Callable[[dict[str, torch.Tensor]], torch.Tensor]


class ProvisionalGeometryHead(nn.Module):
    """Seed-stage ``h_i -> (mu, Cholesky)``, one independent map per slot.

    Zero-initialised, so the provisional geometry at step 0 *is* the anchor grid
    with ``sigma = sigma_init``: the first aligned pooling is a well-conditioned
    partition of the RGB cube rather than 48 coincident blobs.
    """

    def __init__(self, dim: int, n_slots: int = N_SLOTS, n_out: int = N_GEOMETRY):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_slots, dim, n_out))
        self.b = nn.Parameter(torch.zeros(n_slots, n_out))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bni,nio->bno", h, self.w) + self.b

    def n_params(self) -> int:
        return self.w.numel() + self.b.numel()


class _BaseGenerator(nn.Module):
    """Shared plumbing: one backend, one head stack, one decode contract."""

    n_prim_out: int = 0
    per_sample_params: int = 0

    def __init__(self, cfg: BackendConfig, lut: LutConfig, bottleneck: int,
                 seed: int = 0):
        super().__init__()
        self.cfg = cfg
        self.lut = lut
        self.bottleneck = int(bottleneck)
        self.backend = SlotBackend(cfg, seed=seed)
        self.heads = SlotHeads(cfg, self.n_prim_out, self.bottleneck,
                               n_global=N_GLOBAL, seed=seed + 5)
        self.register_buffer("anchors", anchor_points(lut.anchor_grid),
                             persistent=True)

    # -- to be provided by the two arms ------------------------------------
    def seed_geometry(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def decode(self, z_prim: torch.Tensor, z_glob: torch.Tensor,
               seed_geom: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    # -- the protocol 7.3 pipeline -----------------------------------------
    def forward(self, style: torch.Tensor, m_color: torch.Tensor,
                m_color_mask: torch.Tensor | None, wc: torch.Tensor,
                wc_mask: torch.Tensor | None, pool_fn: PoolFn
                ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        h = self.backend.run_seed(style, m_color, m_color_mask, wc, wc_mask)
        geom = self.seed_geometry(h)
        v, pool_stats = pool_fn(geom)
        h = self.backend.run_refine(h, v, style, m_color, m_color_mask, wc, wc_mask)
        z_prim, z_glob = self.heads(h, style)
        params = self.decode(z_prim, z_glob, geom)
        params.update(decode_global(z_glob, self.lut))
        return params, {"pool": pool_stats, "slots": h, "z_prim": z_prim,
                        "z_glob": z_glob, "seed_geometry": geom}

    # -- reporting ----------------------------------------------------------
    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def facts(self) -> dict[str, Any]:
        groups: dict[str, int] = {}
        for name, p in self.named_parameters():
            if p.requires_grad:
                top = name.split(".")[0]
                groups[top] = groups.get(top, 0) + p.numel()
        return {
            "kind": type(self).__name__,
            "n_slots": self.cfg.n_slots,
            "bottleneck": self.bottleneck,
            "n_out_per_slot": self.n_prim_out,
            "per_sample_params": self.per_sample_params,
            "n_trainable_params": self.n_trainable(),
            "params_by_group": groups,
            "backend": {"dim": self.cfg.dim, "heads": self.cfg.n_heads,
                        "ffn": self.cfg.ffn, "seed_blocks": self.cfg.seed_blocks,
                        "refine_blocks": self.cfg.refine_blocks},
            "heads": self.heads.facts(),
        }


class FG48(_BaseGenerator):
    """Protocol 7.4: every parameter of all 48 Gaussians, per sample."""

    n_prim_out = N_PRIM_FG
    per_sample_params = FG_TOTAL_PER_SAMPLE

    def __init__(self, cfg: BackendConfig, lut: LutConfig, bottleneck: int,
                 seed: int = 0):
        super().__init__(cfg, lut, bottleneck, seed)
        self.provisional = ProvisionalGeometryHead(cfg.dim, cfg.n_slots, N_GEOMETRY)

    def seed_geometry(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        return decode_geometry(self.provisional(h), self.anchors, self.lut)

    def decode(self, z_prim, z_glob, seed_geom):
        # the refined geometry replaces the provisional one; the provisional one
        # still carries gradient through the pooling (protocol 7.4)
        return decode_primitives(z_prim, self.lut, PRIM_LAYOUT_FG,
                                 anchors=self.anchors)


class SB48(_BaseGenerator):
    """Protocol 7.5: shared geometry, per-sample activation and colour payload."""

    n_prim_out = N_PRIM_SB
    per_sample_params = SB_TOTAL_PER_SAMPLE

    def __init__(self, cfg: BackendConfig, lut: LutConfig, bottleneck: int,
                 seed: int = 0):
        super().__init__(cfg, lut, bottleneck, seed)
        self.geometry = GeometryBank(lut)

    def seed_geometry(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.geometry(h.shape[0])

    def decode(self, z_prim, z_glob, seed_geom):
        return decode_primitives(z_prim, self.lut, PRIM_LAYOUT_SB,
                                 geometry=seed_geom)


# --- parameter budget -------------------------------------------------------

def generator_param_counts(cfg: BackendConfig, lut: LutConfig,
                           fg_bottleneck: int, sb_bottleneck: int) -> dict[str, int]:
    """Realised counts, obtained by constructing the modules (no arithmetic)."""
    fg = FG48(cfg, lut, fg_bottleneck)
    sb = SB48(cfg, lut, sb_bottleneck)
    return {"FG48": fg.n_trainable(), "SB48": sb.n_trainable()}


def _sb_delta_params(cfg: BackendConfig, bottleneck: int) -> int:
    """SB's heads + shared geometry, as a function of the bottleneck."""
    in_dim = cfg.dim + cfg.z_style_head_proj
    heads = n_head_params(in_dim, bottleneck, N_PRIM_SB, cfg.n_slots)
    glob = in_dim * bottleneck + bottleneck + bottleneck * N_GLOBAL + N_GLOBAL
    geometry = cfg.n_slots * N_GEOMETRY
    return heads + glob + geometry


def _fg_delta_params(cfg: BackendConfig, bottleneck: int) -> int:
    in_dim = cfg.dim + cfg.z_style_head_proj
    heads = n_head_params(in_dim, bottleneck, N_PRIM_FG, cfg.n_slots)
    glob = in_dim * bottleneck + bottleneck + bottleneck * N_GLOBAL + N_GLOBAL
    provisional = cfg.n_slots * (cfg.dim * N_GEOMETRY + N_GEOMETRY)
    return heads + glob + provisional


def solve_sb_bottleneck(cfg: BackendConfig, fg_bottleneck: int,
                        lo: int = 8, hi: int = 4096) -> int:
    """The SB bottleneck that brings the two totals closest together.

    Only the generator-side difference is solved for: both arms carry the same
    backend, the same colour stack and the same WC encoder, so equalising the
    part that differs equalises the whole model.  The search is over integers
    because the bottleneck is a layer width, not a continuous knob.
    """
    target = _fg_delta_params(cfg, fg_bottleneck)
    best, best_err = lo, math.inf
    for b in range(lo, hi + 1):
        value = _sb_delta_params(cfg, b)
        err = abs(value - target)
        if err < best_err:
            best, best_err = b, err
        elif value > target:
            break            # _sb_delta_params is strictly increasing in b
    return best


def build_generator(arm_cfg: ArmConfig, seed: int | None = None) -> _BaseGenerator:
    cfg, lut = arm_cfg.backend, arm_cfg.lut
    s = arm_cfg.seed if seed is None else seed
    if arm_cfg.generator == "FG48":
        return FG48(cfg, lut, arm_cfg.fg_bottleneck, seed=s)
    if arm_cfg.generator == "SB48":
        b = arm_cfg.sb_bottleneck or solve_sb_bottleneck(cfg, arm_cfg.fg_bottleneck)
        return SB48(cfg, lut, b, seed=s)
    raise ValueError(f"unknown generator {arm_cfg.generator!r}")


def parameter_matrix(arms: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    """Trainable-parameter inventory for the twelve arms (protocol 13.1)."""
    from .config import ARM_IDS, arm_config
    from .model import WhatModel

    rows = []
    for a in (arms or ARM_IDS):
        m = WhatModel(arm_config(a))
        f = m.facts()
        rows.append({
            "arm": a, "wc": f["wc"], "generator": f["generator"],
            "where_source": f["where_source"],
            "n_trainable_params": f["n_trainable_params"],
            "params_by_group": f["params_by_group"],
            "per_sample_params": f["generator"]["per_sample_params"]
            if isinstance(f["generator"], dict) else None,
        })
    return rows


def match_report(cfg: BackendConfig, lut: LutConfig, fg_bottleneck: int,
                 sb_bottleneck: int | None = None) -> dict[str, Any]:
    """Protocol 7.5's <=2% statement as a measured number."""
    sb_b = sb_bottleneck or solve_sb_bottleneck(cfg, fg_bottleneck)
    counts = generator_param_counts(cfg, lut, fg_bottleneck, sb_b)
    hi = max(counts.values())
    rel = abs(counts["FG48"] - counts["SB48"]) / hi
    return {
        "fg_bottleneck": fg_bottleneck, "sb_bottleneck": sb_b,
        "fg_params": counts["FG48"], "sb_params": counts["SB48"],
        "abs_diff": abs(counts["FG48"] - counts["SB48"]),
        "rel_diff": rel, "tolerance": PARAM_MATCH_TOLERANCE,
        "within_tolerance": rel <= PARAM_MATCH_TOLERANCE,
    }
