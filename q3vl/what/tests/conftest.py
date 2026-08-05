"""Shared fixtures: mock Where signals, mock LUT targets and a fake batch builder.

The mock sample is not noise.  Each one owns a **known** LUT -- an affine
``T(x) = clamp(A x + c)`` baked onto a real table -- and its ``H_color`` is a
fixed linear embedding of that LUT's own parameters.  So "the loss goes down" in
the mock closed loop means the ``H_color -> z_style -> 48 Gaussians -> T_pred``
path actually carries signal, not that the model learned the dataset mean.

Everything here is deliberately built out of the *real* modules: the real
``Batch``, the real ``WhereSignals``, the real query sampler and the real
``GtLutTable``.  A fixture that invents its own tensor layout tests the fixture.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

import pytest
import torch

from q3vl.what.config import (
    ArmConfig,
    BackendConfig,
    ColorConnectorConfig,
    CONTEXT_GENERATED,
    CONTEXT_GT,
    N_QUERY_NATURAL,
    TEXT_HIDDEN,
    arm_config,
)
from q3vl.what.context import ColorContext
from q3vl.what.data import Batch
from q3vl.what.lut import GtLutTable, lattice_points
from q3vl.what.queries import natural_query_points, query_kind_index, uniform_query_points
from q3vl.what.srht import encode_z_gt, identity_grid, pairwise_rms, u_of_table
from q3vl.what.wc import WhereSignals

GRID_H, GRID_W = 4, 5
N_PATCH = GRID_H * GRID_W


def small_arm(arm: str = "T01", **kw) -> ArmConfig:
    """A structurally faithful but CPU-sized arm.

    Every *structural* property the protocol pins -- 48 slots, 16 colour queries,
    2 seed + 4 refinement blocks, the parameter layout, the WC token set -- is
    unchanged.  Only the widths shrink, so a CPU test exercises the same code
    paths in seconds instead of minutes.
    """
    base = arm_config(arm, **kw)
    color = ColorConnectorConfig(dim=64, n_heads=4, ffn=128, n_blocks=2,
                                 z_style_dim=64, z_style_hidden=64)
    backend = BackendConfig(dim=64, n_heads=4, ffn=128, z_style_dim=64,
                            z_style_head_proj=16)
    return replace(base, color=color, backend=backend, fg_bottleneck=16,
                   sb_bottleneck=None)


@dataclass
class MockSample:
    sample_id: str
    lut_id: str
    table: GtLutTable
    h_color: torch.Tensor
    #: amendment A-4: what the model would read if it had to generate its own
    #: <color> reasoning -- the same signal, degraded.
    h_color_generated: torch.Tensor
    f_pre: torch.Tensor
    rgb_low: torch.Tensor
    m_low: torch.Tensor
    m_hi: torch.Tensor
    image: torch.Tensor
    is_global: bool = False


def affine_table(seed: int, size: int = 17) -> tuple[GtLutTable, torch.Tensor]:
    """A real table whose function is a known affine map, plus its parameters."""
    g = torch.Generator().manual_seed(seed)
    A = torch.eye(3) + 0.25 * torch.randn(3, 3, generator=g)
    c = 0.10 * torch.randn(3, generator=g)
    pts = lattice_points(size)
    vals = (pts @ A.T + c).clamp(0, 1)
    tbl = GtLutTable(
        lut_id=f"mock_{seed}", table=vals.reshape(size, size, size, 3),
        domain_min=torch.zeros(3), domain_max=torch.ones(3), source="mock")
    return tbl, torch.cat([A.reshape(-1), c])


def make_mock_sample(i: int, *, text_dim: int = TEXT_HIDDEN, text_len: int = 6,
                     is_global: bool = False, embed_seed: int = 7) -> MockSample:
    g = torch.Generator().manual_seed(1000 + i)
    tbl, theta = affine_table(i)
    # H_color is a fixed linear embedding of the LUT's own parameters, so the
    # colour path carries recoverable information about the target function.
    eg = torch.Generator().manual_seed(embed_seed)
    E = torch.randn(theta.numel(), text_dim, generator=eg) / theta.numel() ** 0.5
    h_color = (theta @ E).unsqueeze(0).repeat(text_len, 1)
    h_color = h_color + 0.01 * torch.randn(text_len, text_dim, generator=g)
    # the generated context carries the same information, less cleanly
    h_color_generated = h_color + 0.05 * torch.randn(text_len, text_dim, generator=g)

    image = torch.rand(3, GRID_H * 8, GRID_W * 8, generator=g)
    rgb_low = torch.rand(N_PATCH, 3, generator=g)
    m_low = torch.rand(N_PATCH, generator=g).clamp(0.05, 1.0)
    m_hi = torch.rand(image.shape[-2:], generator=g).clamp(0.05, 1.0)
    return MockSample(
        sample_id=f"mock_{i:04d}", lut_id=tbl.lut_id, table=tbl, h_color=h_color,
        h_color_generated=h_color_generated,
        f_pre=torch.randn(N_PATCH, 1024, generator=g), rgb_low=rgb_low,
        m_low=m_low, m_hi=m_hi, image=image, is_global=is_global)


_D_FUNC_SCALE: float | None = None


def mock_d_func_scale(n: int = 16) -> float:
    """Amendment A-2's ``C`` over the mock LUT set, by the production closed form."""
    global _D_FUNC_SCALE
    if _D_FUNC_SCALE is None:
        grid = identity_grid()
        total = torch.zeros(grid.numel(), dtype=torch.float64)
        sum_sq = 0.0
        for i in range(n):
            u = u_of_table(affine_table(i)[0].apply(grid, "trilinear")).double()
            total += u
            sum_sq += float(u.pow(2).sum())
        _D_FUNC_SCALE = pairwise_rms(n, total, sum_sq)
    return _D_FUNC_SCALE


class MockDataset:
    def __init__(self, n: int = 8):
        self.samples = [make_mock_sample(i) for i in range(n)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> MockSample:
        return self.samples[i]


class MockColorGenCtx:
    """Stand-in for the published generated-``<color>`` store (amendment A-4).

    It satisfies the two things the trainer and the builder facts ask of a real
    :class:`~q3vl.what.stores.ColorGenContextStore` -- that it exists and that it
    reports its mode -- without needing published shards.  The *contract* of the
    real store (schema version, mode agreement, coverage) is tested directly in
    ``test_a4_color_context.py`` against a fake published payload.
    """

    def __init__(self, mode: str):
        self.mode = mode

    def summary(self) -> dict[str, Any]:
        return {"kind": "MockColorGenCtx", "mode": self.mode, "n": 0}


class MockBuilder:
    """Assembles real :class:`Batch` objects from :class:`MockSample`s.

    Amendment A-4: the generated context is modelled as the teacher embedding
    plus noise -- "the model's own colour reasoning is an imperfect version of
    the GT one".  That keeps the 50/50 plumbing, the per-context loss split and
    the ratio assertions on a real code path, while the *store* contract is
    tested separately against a fake published record.
    """

    def __init__(self, cfg: ArmConfig, seed: int = 0, gt_interp: str = "trilinear"):
        from q3vl.what.srht import SRHT

        self.cfg = cfg
        self.seed = seed
        self.gt_interp = gt_interp
        self._x_uniform = uniform_query_points()
        self._kind = query_kind_index()
        self._center = torch.zeros(17 ** 3 * 3)
        # ``z_style`` and ``z_gt`` must live in the same space (protocol 7.1); the
        # production arms use 1024 for both.  The reduced test config shrinks
        # ``z_style``, so the SRHT's output width has to follow it -- which is
        # itself the assertion that the two dimensions are one dimension.
        self._srht = SRHT(17 ** 3 * 3, cfg.color.z_style_dim)
        # amendment A-2's C, computed over the mock "train" LUT set with the same
        # closed form the real job uses.  A constant, never a per-batch statistic.
        self.d_func_scale = mock_d_func_scale()
        self.color_genctx = MockColorGenCtx(cfg.genctx_mode)

    def build(self, samples: Sequence[MockSample],
              modes: Sequence[str] | None = None) -> Batch:
        b = len(samples)
        modes = [CONTEXT_GT] * b if modes is None else list(modes)
        if len(modes) != b:
            raise ValueError("samples and context modes must align")
        contexts = [
            ColorContext(mode=m, token_ids=[1, 2, 3], provenance=s.sample_id,
                         stop_reason="closed",
                         genctx_mode=(self.cfg.genctx_mode
                                      if m == CONTEXT_GENERATED else ""))
            for s, m in zip(samples, modes)
        ]
        h_color = torch.stack([
            s.h_color if m == CONTEXT_GT else s.h_color_generated
            for s, m in zip(samples, modes)
        ])
        inputs: dict[str, Any] = {
            "h_color": h_color,
            "h_color_mask": torch.ones(b, h_color.shape[1], dtype=torch.bool),
            "f_pre": torch.stack([s.f_pre for s in samples]),
            "f_pre_mask": torch.ones(b, N_PATCH, dtype=torch.bool),
            "rgb_low": torch.stack([s.rgb_low for s in samples]),
            "where": self._signals(samples),
        }
        targets = []
        for i, s in enumerate(samples):
            nat = natural_query_points(s.image, N_QUERY_NATURAL,
                                       weights=None if s.is_global else s.m_hi,
                                       seed=self.seed + i)
            x = torch.cat([self._x_uniform, nat], dim=0)
            with torch.no_grad():
                t_gt = s.table.apply(x, self.gt_interp)
                grid_vals = s.table.apply(identity_grid(), self.gt_interp)
                u_gt = u_of_table(grid_vals)
                z_gt = encode_z_gt(grid_vals, self._center, self._srht)
            targets.append({"sample_id": s.sample_id, "lut_id": s.lut_id, "x": x,
                            "t_gt": t_gt, "z_gt": z_gt, "u_gt": u_gt,
                            "query_kind": self._kind,
                            "is_global": s.is_global,
                            "natural_weighting": ("global_uniform" if s.is_global
                                                  else "frozen_m_pred"),
                            "meta": {"render_mode": "global" if s.is_global else "local",
                                     "build": "l1"}})
        batch = Batch(inputs=inputs, targets=targets,
                      sample_ids=[s.sample_id for s in samples],
                      meta=[t["meta"] for t in targets], contexts=contexts)
        batch.check_inputs()
        return batch

    def _signals(self, samples: Sequence[MockSample]) -> WhereSignals:
        if self.cfg.where_source == "none":
            return WhereSignals(source="none")
        b = len(samples)
        n_rho = 36 if self.cfg.where_readout == "cband12" else 4
        g = torch.Generator().manual_seed(self.seed + 99)
        return WhereSignals(
            m_low=torch.stack([s.m_low for s in samples]),
            m_hi=[s.m_hi for s in samples],
            canvas_axis=torch.randn(b, 16, self.cfg.backend.dim, generator=g),
            canvas_rho=torch.randn(b, 16, self.cfg.backend.dim, generator=g),
            w_vec=torch.randn(b, 73, generator=g),
            rho_vec=torch.randn(b, n_rho, generator=g),
            source=self.cfg.where_source,
        )

    def facts(self) -> dict[str, Any]:
        return {"kind": "MockBuilder", "gt_interp": self.gt_interp,
                "d_func_scale": self.d_func_scale,
                "genctx_mode": self.cfg.genctx_mode,
                "color_genctx": self.color_genctx.summary()}


@pytest.fixture()
def mock_dataset() -> MockDataset:
    return MockDataset(8)


@pytest.fixture()
def t01_cfg() -> ArmConfig:
    return small_arm("T01")


@pytest.fixture()
def t08_cfg() -> ArmConfig:
    return small_arm("T08")
