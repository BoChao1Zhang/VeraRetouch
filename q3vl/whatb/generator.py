"""CGLUT conditional generator: ``R^d -> Theta``, plus the affine-only variant.

Structure is GLUT App A.2 verbatim (via ``docs/HANDOFF_whatb_2026-08-15.md``
section 6.3 and ``EPR-024:405-424``): a **shared 3-layer encoder, 128 hidden
units** ("small" = 64), then one head per parameter group --

======================  ======  ==========  ================================
head                    layers  output      source
======================  ======  ==========  ================================
``head_mu``             2       ``3N``      App A.2, the one output the paper names
``head_cov``            2       ``6N``      "similar structure with adjusted dims"
``head_opacity``        2       ``N``       idem
``head_color``          **3**   ``12N``     App A.2 names the local colour head as 3-layer
``head_global``         2       ``12``      App A.2 fixes 12 = 9 + 3
======================  ======  ==========  ================================

ReLU between every pair of layers, nowhere else.  The "adjusted dimensions" of
rows 2/3/4 are an inference from the section 3.1 parameter list -- and the
inference is *checked*, not asserted: :func:`generator_param_count` reproduces
the paper's Table 2 counts (98K / 338K) to the digit once the projection is
swapped back for CGLUT's ``E in R^{225x64}`` lookup (14,400).  See
``tests/test_generator.py::test_paper_table2_param_count``.

The condition
-------------
This project does **not** learn a per-LUT lookup ``e_l``.  The condition is a
frozen-VLM read-out: ``z = norm(hidden_states[-1])`` at ``<seg_color>``
(id 151674), shape ``(2560,)``, projected by
:class:`SegColorProjection` = ``LayerNorm(2560) + Linear(2560 -> d)``.  That
projection stands exactly where ``e_l`` stood; ``d = 64`` is CGLUT's ``D``.
The generator itself only ever sees the ``d``-vector, so an arm may feed it a
condition from anywhere (interpolated, zeroed, train-mean, ...) without the
generator knowing.

Two modes
---------
``"full"``          Full Generation -- all five heads, ``22N + 12`` generated.
                    Init: PyTorch default everywhere (EPR-024:556 ruling 11.1-1).
``"affine_only"``   EPR-025.  ``{mu, Sigma, o}`` become condition-independent
                    ``nn.Parameter`` tables (:class:`SharedGeometry`); only
                    ``{M_i, b_i, G, g}`` (``12N + 12``) are generated, the last
                    layer of both surviving heads is zero-initialised and
                    ``M_i = I + dM_i``, so step 0 is exactly the identity and
                    ``f`` is **linear in the generated parameters** -- the
                    premise of proposition 1.

Discipline
----------
Constants (the grid init, the ``3x3`` identity used by the residual ``M``) are
``register_buffer(..., persistent=False)``; there is no ``torch.tensor(...)``
inside any ``forward``.  Every incoming condition is moved onto the module's
own ``(device, dtype)`` explicitly.  Nothing here imports ``q3vl.what``.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from q3vl.whatb.glut import (
    EPS,
    GlutParams,
    glut_geometry,
    softplus_inverse,
    uniform_grid_positions,
)

__all__ = [
    "GeneratorMode",
    "SegColorProjection",
    "SharedGeometry",
    "CGLUTGenerator",
    "generator_param_count",
    "SEG_COLOR_HIDDEN_DIM",
]

#: Qwen3-VL-4B v2seg last-layer hidden width at ``<seg_color>``.
SEG_COLOR_HIDDEN_DIM: int = 2560

GeneratorMode = Literal["full", "affine_only"]


def _mlp(sizes: list[int]) -> nn.Sequential:
    """``Linear -> ReLU -> ... -> Linear`` (no activation after the last layer)."""
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def generator_param_count(*, cond_dim: int, hidden: int, n_gauss: int, mode: GeneratorMode = "full") -> int:
    """Closed-form count of the encoder + heads (``pi`` excluded).

    ``(d=64, H=64,  N=32, full)`` -> 83,980   (+ 14,400 lookup = 98,380 ~ 98K)
    ``(d=64, H=128, N=64, full)`` -> 323,596  (+ 14,400 lookup = 337,996 ~ 338K)
    ``(d=64, H=128, N=48, full)`` -> 278,188
    ``(d=64, H=128, N=48, affine_only)`` -> 166,732
    """
    d, h, n = int(cond_dim), int(hidden), int(n_gauss)
    lin = lambda i, o: i * o + o
    enc = lin(d, h) + 2 * lin(h, h)
    color = lin(h, h) + lin(h, h) + lin(h, 12 * n)
    glob = lin(h, h) + lin(h, 12)
    total = enc + color + glob
    if mode == "full":
        total += lin(h, h) + lin(h, 3 * n)   # mu
        total += lin(h, h) + lin(h, 6 * n)   # cov
        total += lin(h, h) + lin(h, 1 * n)   # opacity
    elif mode != "affine_only":
        raise ValueError(f"unknown mode {mode!r}")
    return total


class SegColorProjection(nn.Module):
    """``pi = Linear(LayerNorm(2560) -> d)`` -- the single structural change of EPR-024.

    ``forward(z)``: ``(B, 2560) -> (B, d)``; ``z`` is cast onto the module's own
    ``(device, dtype)`` first, so a cached fp32 ``z`` meeting a bf16 module (or
    the reverse) is a cast, never a crash.
    """

    def __init__(self, *, in_dim: int = SEG_COLOR_HIDDEN_DIM, cond_dim: int = 64) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.cond_dim = int(cond_dim)
        self.norm = nn.LayerNorm(self.in_dim)
        self.proj = nn.Linear(self.in_dim, self.cond_dim)

    def forward(self, z: Tensor) -> Tensor:  # noqa: D102
        if z.shape[-1] != self.in_dim:
            raise ValueError(f"pi expects (..., {self.in_dim}), got {tuple(z.shape)}")
        ref = self.proj.weight
        return self.proj(self.norm(z.to(device=ref.device, dtype=ref.dtype)))


class SharedGeometry(nn.Module):
    """Condition-independent ``{mu, Cholesky, opacity logit}`` -- EPR-025:372-375.

    Four ``nn.Parameter`` tables, ``(N,3) / (N,3) / (N,3) / (N,)``, initialised
    per GLUT App A.1 plus the two values EPR-025 marks NOVEL::

        mu         uniform grid in [0,1]^3 (N=48 -> 4x4x3)
        chol_diag  softplus^-1(0.15) = -1.8212...  (isotropic sigma = 0.15)
        chol_off   0                               (NOVEL)
        opa_logit  +4.0  -> sigmoid = 0.98201      (NOVEL; App A.1 has o = 1.0,
                                                    unreachable through sigmoid)

    ``forward()`` **takes no condition**.  That is the type-level guarantee the
    sharing is really wired: an affine-only run cannot accidentally make the
    geometry condition-dependent without changing this signature.
    """

    def __init__(
        self,
        n_gauss: int,
        *,
        sigma: float = 0.15,
        opacity_logit: float = 4.0,
        eps: float = EPS,
    ) -> None:
        super().__init__()
        n = int(n_gauss)
        self.n_gauss = n
        self.eps = float(eps)
        self.init_sigma = float(sigma)
        self.init_opacity_logit = float(opacity_logit)
        self.mu = nn.Parameter(uniform_grid_positions(n))
        self.chol_diag = nn.Parameter(torch.full((n, 3), softplus_inverse(sigma)))
        self.chol_off = nn.Parameter(torch.zeros(n, 3))
        self.opacity_logit = nn.Parameter(torch.full((n,), float(opacity_logit)))
        self.register_buffer("eye3", torch.eye(3), persistent=False)

    def extra_repr(self) -> str:
        return f"n_gauss={self.n_gauss}, sigma={self.init_sigma}, opacity_logit={self.init_opacity_logit}"

    def forward(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """``(precision (N,3,3), logdet (N,), opacity (N,), degenerate (N,))``."""
        return glut_geometry(
            self.chol_diag, self.chol_off, self.opacity_logit,
            eps=self.eps, eye=self.eye3.to(dtype=self.chol_diag.dtype),
        )

    def parameters_list(self) -> list[nn.Parameter]:
        """The 0.1x-lr group (App A.1's "shared geometry" rule; ``o`` in it is NOVEL)."""
        return [self.mu, self.chol_diag, self.chol_off, self.opacity_logit]


class CGLUTGenerator(nn.Module):
    """``G_theta: R^d -> Theta``.  ``forward(u) -> GlutParams`` with batch ``B``.

    Parameters
    ----------
    cond_dim
        ``d``.  CGLUT's ``D = 64``; EPR-024 contrasts 256 / 32.
    hidden
        ``H``.  128 = "large", 64 = "small" (App A.2).
    n_gauss
        ``N``.  48 for this campaign, 32 forced alongside (paper value).
    mode
        ``"full"`` or ``"affine_only"`` (see the module docstring).
    m_residual
        ``M_i = I + dM_i``.  Default: ``False`` for ``full`` (EPR-024 takes
        PyTorch default init and no identity anchoring), ``True`` for
        ``affine_only`` (EPR-025:388).
    zero_init_last
        Zero the last ``Linear`` of every head.  Default: ``False`` for
        ``full``, ``True`` for ``affine_only`` (StatLUT section 3.2 form).
    shared_sigma, shared_opacity_logit
        Only for ``affine_only``: the :class:`SharedGeometry` init values.

    Shapes: ``u`` is ``(B, d)``; the returned :class:`GlutParams` has ``mu``
    ``(B, N, 3)``, ``m_local`` ``(B, N, 3, 3)``, ``g_matrix`` ``(B, 3, 3)`` ...
    In ``affine_only`` the geometry fields are the shared tables broadcast to
    ``B`` (so downstream code needs no mode switch), and
    :meth:`shared_geometry_terms` hands the caller the *once-per-step*
    ``(precision, logdet, opacity, degenerate)`` for reuse.
    """

    def __init__(
        self,
        *,
        cond_dim: int = 64,
        hidden: int = 128,
        n_gauss: int = 48,
        mode: GeneratorMode = "full",
        m_residual: bool | None = None,
        zero_init_last: bool | None = None,
        shared_sigma: float = 0.15,
        shared_opacity_logit: float = 4.0,
        eps: float = EPS,
    ) -> None:
        super().__init__()
        if mode not in ("full", "affine_only"):
            raise ValueError(f"unknown mode {mode!r}")
        d, h, n = int(cond_dim), int(hidden), int(n_gauss)
        self.cond_dim, self.hidden, self.n_gauss, self.mode = d, h, n, mode
        self.m_residual = (mode == "affine_only") if m_residual is None else bool(m_residual)
        self.zero_init_last = (mode == "affine_only") if zero_init_last is None else bool(zero_init_last)
        self.eps = float(eps)

        self.encoder = _mlp([d, h, h, h])
        self.encoder.append(nn.ReLU())  # App A.2: ReLU after every encoder layer
        self.head_color = _mlp([h, h, h, 12 * n])   # 3 layers (named in App A.2)
        self.head_global = _mlp([h, h, 12])         # 2 layers, fixed width 12

        if mode == "full":
            self.head_mu = _mlp([h, h, 3 * n])
            self.head_cov = _mlp([h, h, 6 * n])
            self.head_opacity = _mlp([h, h, 1 * n])
            self.shared_geometry = None
        else:
            self.head_mu = self.head_cov = self.head_opacity = None
            self.shared_geometry = SharedGeometry(
                n, sigma=shared_sigma, opacity_logit=shared_opacity_logit, eps=eps
            )

        self.register_buffer("eye3", torch.eye(3), persistent=False)
        if self.zero_init_last:
            for head in (self.head_color, self.head_global, self.head_mu, self.head_cov, self.head_opacity):
                if head is None:
                    continue
                last = head[-1]
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    # ---- bookkeeping ----
    def extra_repr(self) -> str:
        return (
            f"cond_dim={self.cond_dim}, hidden={self.hidden}, n_gauss={self.n_gauss}, "
            f"mode={self.mode}, m_residual={self.m_residual}, zero_init_last={self.zero_init_last}"
        )

    @property
    def theta_dim(self) -> int:
        """Size of the *generated* slice of ``Theta``: ``22N+12`` / ``12N+12``."""
        n = self.n_gauss
        return 22 * n + 12 if self.mode == "full" else 12 * n + 12

    @property
    def config(self) -> dict[str, object]:
        """What ``run_setup.json`` records for the generator."""
        cfg: dict[str, object] = {
            "cond_dim": self.cond_dim,
            "hidden": self.hidden,
            "n_gauss": self.n_gauss,
            "mode": self.mode,
            "m_residual": self.m_residual,
            "zero_init_last": self.zero_init_last,
            "theta_dim": self.theta_dim,
            "n_params": sum(p.numel() for p in self.parameters()),
        }
        if self.shared_geometry is not None:
            cfg["shared_sigma"] = self.shared_geometry.init_sigma
            cfg["shared_opacity_logit"] = self.shared_geometry.init_opacity_logit
        return cfg

    def param_groups(self, base_lr: float, *, geometry_lr_scale: float = 0.1) -> list[dict[str, object]]:
        """Optimiser groups.  App A.1: shared geometry trains at 0.1x base lr.

        (Folding ``o`` into that group is EPR-025's NOVEL choice -- GLUT's own
        Shared Geometry keeps ``o`` condition-generated.  ``--shared-geom-lr-scale 1.0``
        is that arm's ablation row 5.)
        """
        if self.shared_geometry is None:
            return [{"params": list(self.parameters()), "lr": float(base_lr), "name": "generator"}]
        geo = self.shared_geometry.parameters_list()
        geo_ids = {id(p) for p in geo}
        rest = [p for p in self.parameters() if id(p) not in geo_ids]
        return [
            {"params": rest, "lr": float(base_lr), "name": "generator"},
            {"params": geo, "lr": float(base_lr) * float(geometry_lr_scale), "name": "shared_geometry"},
        ]

    def shared_geometry_terms(self) -> tuple[Tensor, Tensor, Tensor, Tensor] | None:
        """``(precision, logdet, opacity, degenerate)`` once per step, or ``None``.

        ``affine_only`` only.  Pass it to ``glut_forward(..., _geometry=...)``
        after ``unsqueeze(0).expand(B, ...)`` to skip the per-sample recompute;
        the fact that this is *possible* is proposition 1's premise.
        """
        return None if self.shared_geometry is None else self.shared_geometry()

    # ---- forward ----
    def encode(self, u: Tensor) -> Tensor:
        """``(B, d) -> (B, H)`` -- the shared encoder."""
        if u.dim() != 2 or u.shape[-1] != self.cond_dim:
            raise ValueError(f"condition must be (B, {self.cond_dim}), got {tuple(u.shape)}")
        ref = self.head_global[0].weight
        return self.encoder(u.to(device=ref.device, dtype=ref.dtype))

    def forward(self, u: Tensor) -> GlutParams:  # noqa: D102
        h = self.encode(u)
        b, n = h.shape[0], self.n_gauss
        eye = self.eye3.to(device=h.device, dtype=h.dtype)

        color = self.head_color(h).reshape(b, n, 12)
        m_local = color[..., :9].reshape(b, n, 3, 3)
        if self.m_residual:
            m_local = eye + m_local
        b_local = color[..., 9:]
        glob = self.head_global(h)
        g_matrix = glob[:, :9].reshape(b, 3, 3)
        g_bias = glob[:, 9:]

        if self.mode == "full":
            mu = self.head_mu(h).reshape(b, n, 3)
            chol = self.head_cov(h).reshape(b, n, 6)
            chol_diag, chol_off = chol[..., :3], chol[..., 3:]
            opacity_logit = self.head_opacity(h).reshape(b, n)
        else:
            sg = self.shared_geometry
            assert sg is not None
            mu = sg.mu.to(device=h.device, dtype=h.dtype).unsqueeze(0).expand(b, n, 3)
            chol_diag = sg.chol_diag.to(device=h.device, dtype=h.dtype).unsqueeze(0).expand(b, n, 3)
            chol_off = sg.chol_off.to(device=h.device, dtype=h.dtype).unsqueeze(0).expand(b, n, 3)
            opacity_logit = sg.opacity_logit.to(device=h.device, dtype=h.dtype).unsqueeze(0).expand(b, n)

        return GlutParams(
            mu=mu,
            chol_diag=chol_diag,
            chol_off=chol_off,
            opacity_logit=opacity_logit,
            m_local=m_local,
            b_local=b_local,
            g_matrix=g_matrix,
            g_bias=g_bias,
        )
