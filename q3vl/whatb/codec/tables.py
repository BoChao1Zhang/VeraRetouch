"""C1 -- **O0**: the direct carrier table (EPR-031 §5's O0 row).

``Theta in R^{n_lut x D_theta}``.  One independent, directly learnable carrier
parameter set per ``lut_id``: **no generator, no VLM, no conditional read-out**.
O0 answers exactly one question -- can an ``N = 48`` carrier represent the LUT
pool at all -- so anything that could stand between a LUT and its parameters is
absent by construction, and ``E_O0`` is the floor the other three levels
(``Delta_decoder`` / ``Delta_code`` / ``Delta_readout``) are differenced against.

Everything about the carrier itself comes from :mod:`q3vl.whatb.arms.g4d`
(EPR-028 R1): :class:`~q3vl.whatb.arms.g4d.Glut4DCarrier` is the forward,
:class:`~q3vl.whatb.arms.g4d.G4DParams` the container,
:func:`~q3vl.whatb.arms.g4d.n_params_g4d` the width, ``total_loss`` /
``r_line`` / ``target_4d`` the objective.  Nothing is re-derived here; a second
copy of a 27-per-primitive layout is a drift waiting to happen.

Layout.  ``Theta`` is a single ``nn.Parameter`` of shape ``(n_lut, D_theta)`` and
:data:`THETA_FIELDS` slices it, so ``sum(p.numel())`` is *exactly*
``n_lut * n_params_g4d(N, mode)`` -- the property the unit test pins.  A row is
reached with ``theta[idx]``, whose backward writes into that row alone, so two
``lut_id`` cannot share a gradient (the second pinned property).

Sampling (EPR-028 R1 §8.1).  A step is ``L LUTs x Q colours x 4 s-anchors`` with
``s in {0, 1, u, 1-u}``, ``u ~ U(0,1)`` per colour; hard mining runs on **colour
groups**, never single anchors, so an endpoint is never dropped alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from q3vl.whatb import queries
from q3vl.whatb.arms import g4d
from q3vl.whatb.glut import softplus_inverse, uniform_grid_positions

__all__ = [
    "ARM",
    "LEVEL",
    "S_ANCHORS",
    "STAGES",
    "STAGE_CHOICES",
    "CARRIER_CHOICES",
    "THETA_FIELDS",
    "DirectCarrierTable",
    "StageSpec",
    "assert_anchor_structure",
    "initial_theta",
    "telemetry_row",
    "paired_anchor_batch",
    "resolve_stage",
    "r_line_from_anchors",
    "select_hard_colour_groups",
    "stage_facts",
    "theta_fields",
]

ARM: str = "EPR-031"
#: which row of §5's four-level decomposition this module implements
LEVEL: str = "O0"

#: EPR-028 R1 §8.1: every colour carries exactly four ``s`` anchors
S_ANCHORS: int = 4

#: EPR-031 §7's C1 ladder.  ``n_lut_pool = None`` means "the whole train pool".
#: colours per step = ``b_luts * q_colors * S_ANCHORS``:
#: S1 1x2048x4 = 8,192 / S2 32x512x4 = 65,536 / S3 256x2048x4 = 2,097,152.
STAGES: dict[str, dict[str, Any]] = {
    "S1": {"n_lut_pool": 1, "b_luts": 1, "q_colors": 2048,
           "total_steps": 2000, "mining": False,
           "note": "single-LUT overfit, no mining"},
    "S2": {"n_lut_pool": 32, "b_luts": 32, "q_colors": 512,
           "total_steps": 4000, "mining": True,
           "note": "32-LUT pool, fixed seeded draw"},
    "S3": {"n_lut_pool": None, "b_luts": 256, "q_colors": 2048,
           "total_steps": 18760, "mining": True,
           "note": "the whole train pool, 2,097,152 colours/step"},
}
STAGE_CHOICES: tuple[str, ...] = tuple(STAGES)

#: the two carrier settings this EPR runs (EPR-028 R1 §3).  O0 changes the
#: *conditioner* (to nothing at all); the carrier is imported unchanged.
CARRIER_CHOICES: tuple[str, ...] = ("A1", "A3")


# --------------------------------------------------------------------------- #
# 1. the flat parameter layout
# --------------------------------------------------------------------------- #
def theta_fields(mode: str, n_gauss: int) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """``((name, per-row shape), ...)`` in the flat ``Theta`` row's own order.

    A0/A1 carry the eight 3D fields (22 per primitive + 12 global); A2/A3 add
    ``beta`` / ``mu_s`` / ``tau_raw`` (27 per primitive + 12).  The widths are
    **checked** against :func:`q3vl.whatb.arms.g4d.n_params_g4d`, so a layout
    that drifts from the carrier's own count raises here rather than producing a
    table with the wrong number of learnable parameters.
    """
    if mode not in g4d.G4D_MODES:
        raise ValueError(f"unknown mode {mode!r}; expected {g4d.G4D_MODES}")
    n = int(n_gauss)
    fields: list[tuple[str, tuple[int, ...]]] = [
        ("mu_x", (n, 3)), ("chol_diag", (n, 3)), ("chol_off", (n, 3)),
        ("opacity_logit", (n,)), ("m_local", (n, 3, 3)), ("b_local", (n, 3)),
    ]
    if mode in g4d.CONDITIONAL_MODES:
        fields += [("beta", (n, 3)), ("mu_s", (n,)), ("tau_raw", (n,))]
    fields += [("g_matrix", (3, 3)), ("g_bias", (3,))]
    total = sum(math.prod(shape) for _, shape in fields)
    want = g4d.n_params_g4d(n, mode)
    if total != want:
        raise AssertionError(
            f"the flat Theta layout for {mode} N={n} is {total} wide but "
            f"n_params_g4d says {want}; the table would not be the carrier's "
            "parameter set")
    return tuple(fields)


#: the layout used by :class:`DirectCarrierTable`, exposed for the unit test
THETA_FIELDS = theta_fields


def initial_theta(mode: str, n_gauss: int, *, init_seed: int = 20260810,
                  dtype: torch.dtype = torch.float32) -> Tensor:
    """The ``(D_theta,)`` initial row -- EPR-028 R1 §8.4, verbatim.

    ``opacity_logit = 0`` (sigmoid 0.5), ``L_C`` diag ``= 0.15`` through the
    softplus inverse and off-diagonal 0, ``beta = 0``, ``mu_s ~ U(-0.1, 1.1)``,
    ``tau`` at ``sqrt(0.2) = 0.4472`` through its raw parameterisation,
    ``mu_x`` on the uniform RGB grid, ``M = I``, ``b = 0``, ``G = 0``, ``g = 0``
    -- which is ``f(x, s) ~= x`` at step 0.

    **Every row of the table starts from this same vector** (§5's O0 row is about
    capacity, not about a per-LUT initialisation advantage); ``mu_s`` is drawn
    once, from a private generator seeded by ``init_seed``, and shared.
    """
    n = int(n_gauss)
    gen = torch.Generator().manual_seed(int(init_seed))
    lo, hi = g4d.MU_S_INIT_RANGE
    values: dict[str, Tensor] = {
        "mu_x": uniform_grid_positions(n, dtype=dtype),
        "chol_diag": torch.full((n, 3), softplus_inverse(g4d.SIGMA_RGB_INIT),
                                dtype=dtype),
        "chol_off": torch.zeros(n, 3, dtype=dtype),
        "opacity_logit": torch.full((n,), g4d.OPACITY_LOGIT_INIT, dtype=dtype),
        "m_local": torch.eye(3, dtype=dtype).expand(n, 3, 3).contiguous(),
        "b_local": torch.zeros(n, 3, dtype=dtype),
        "beta": torch.zeros(n, 3, dtype=dtype),
        "mu_s": (torch.rand(n, generator=gen, dtype=torch.float32) * (hi - lo)
                 + lo).to(dtype),
        "tau_raw": torch.full((n,), g4d.TAU_RAW_INIT, dtype=dtype),
        "g_matrix": torch.zeros(3, 3, dtype=dtype),
        "g_bias": torch.zeros(3, dtype=dtype),
    }
    return torch.cat([values[name].reshape(-1)
                      for name, _ in theta_fields(mode, n)], dim=0)


# --------------------------------------------------------------------------- #
# 2. the table
# --------------------------------------------------------------------------- #
class DirectCarrierTable(nn.Module):
    """``Theta in R^{n_lut x D_theta}`` -- O0's whole model.

    ``lut_ids`` fixes the row order (the table is addressed by ``lut_id``, never
    by a batch position).  :meth:`params_for` slices a batch of rows into a
    :class:`~q3vl.whatb.arms.g4d.G4DParams`; no reshape leaves the row, so the
    gradient of a step reaches exactly the rows that step sampled.
    """

    def __init__(self, lut_ids: Sequence[str], cfg: g4d.G4DConfig, *,
                 dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        ids = [str(i) for i in lut_ids]
        if len(set(ids)) != len(ids):
            raise ValueError(
                "lut_ids carries duplicates; two rows for one LUT would make "
                "'which row did this step update' unanswerable")
        if not ids:
            raise ValueError("the O0 table needs at least one lut_id")
        self.cfg = cfg
        self.lut_ids: list[str] = ids
        self._index: dict[str, int] = {lid: i for i, lid in enumerate(ids)}
        self.fields = theta_fields(cfg.mode, cfg.n_gauss)
        row = initial_theta(cfg.mode, cfg.n_gauss, init_seed=cfg.init_seed,
                            dtype=dtype)
        self.theta = nn.Parameter(row.unsqueeze(0).repeat(len(ids), 1).clone())

    # -- bookkeeping -------------------------------------------------------- #
    @property
    def n_lut(self) -> int:
        return len(self.lut_ids)

    @property
    def theta_dim(self) -> int:
        return int(self.theta.shape[1])

    def index_of(self, lut_ids: Sequence[str], *, device: Any = None) -> Tensor:
        try:
            idx = [self._index[str(i)] for i in lut_ids]
        except KeyError as exc:                       # pragma: no cover - guard
            raise KeyError(
                f"{exc.args[0]!r} has no row in this O0 table ({self.n_lut} "
                "rows); the table is addressed by lut_id, so a LUT outside the "
                "stage's pool cannot be trained by accident") from None
        return torch.as_tensor(idx, dtype=torch.long,
                               device=device or self.theta.device)

    def config(self) -> dict[str, Any]:
        return {"arm": ARM, "level": LEVEL, **self.cfg.as_dict(),
                "n_lut": self.n_lut, "theta_dim": self.theta_dim,
                "n_params_total": sum(p.numel() for p in self.parameters()),
                "n_params_expected": self.n_lut * g4d.n_params_g4d(
                    self.cfg.n_gauss, self.cfg.mode),
                "conditioner": "none (direct table, no generator, no VLM)",
                "field_order": [name for name, _ in self.fields]}

    # -- forward ------------------------------------------------------------ #
    def rows(self, index: Tensor) -> Tensor:
        """``(B, D_theta)`` -- the sampled rows, gradient-isolated per row."""
        return self.theta[index.to(self.theta.device)]

    def params_from_rows(self, rows: Tensor) -> g4d.G4DParams:
        """Slice ``(B, D_theta)`` into the carrier's own parameter container."""
        b = int(rows.shape[0])
        out: dict[str, Tensor] = {}
        at = 0
        for name, shape in self.fields:
            width = 1
            for s in shape:
                width *= int(s)
            out[name] = rows[:, at:at + width].reshape(b, *shape)
            at += width
        if at != rows.shape[1]:
            raise AssertionError(
                f"the layout consumed {at} of {int(rows.shape[1])} columns")
        return g4d.G4DParams(mode=self.cfg.mode, **out)

    def params_for(self, lut_ids: Sequence[str]) -> g4d.G4DParams:
        return self.params_from_rows(self.rows(self.index_of(lut_ids)))

    def forward(self, lut_ids: Sequence[str]) -> g4d.G4DParams:
        return self.params_for(lut_ids)

    def param_groups(self, base_lr: float) -> list[dict[str, Any]]:
        """One group: there is nothing else to learn (that is the O0 row)."""
        return [{"params": [self.theta], "lr": float(base_lr), "name": "theta"}]


# --------------------------------------------------------------------------- #
# 3. the stage ladder
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StageSpec:
    """One resolved row of :data:`STAGES`, with its derived counts."""

    stage: str
    n_lut_pool: int
    b_luts: int
    q_colors: int
    total_steps: int
    mining: bool
    note: str

    @property
    def s_anchors(self) -> int:
        return S_ANCHORS

    @property
    def colors_per_step(self) -> int:
        """``L * Q * 4`` -- the ``(x, s)`` pair count of one step."""
        return int(self.b_luts) * int(self.q_colors) * S_ANCHORS

    @property
    def distinct_colors_per_step(self) -> int:
        return int(self.b_luts) * int(self.q_colors)

    def as_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "n_lut_pool": self.n_lut_pool,
                "b_luts": self.b_luts, "q_colors": self.q_colors,
                "s_anchors": S_ANCHORS,
                "colors_per_step": self.colors_per_step,
                "distinct_colors_per_step": self.distinct_colors_per_step,
                "total_steps": self.total_steps, "mining": self.mining,
                "note": self.note,
                "s_anchor_set": "{0, 1, u, 1-u}, u ~ U(0,1)"}


def resolve_stage(stage: str, *, n_train_luts: int) -> StageSpec:
    """:data:`STAGES` -> :class:`StageSpec`; ``None`` pool means the whole pool."""
    if stage not in STAGES:
        raise ValueError(f"--stage must be one of {STAGE_CHOICES}; got {stage!r}")
    row = STAGES[stage]
    pool = row["n_lut_pool"]
    return StageSpec(stage=stage,
                     n_lut_pool=int(n_train_luts) if pool is None else int(pool),
                     b_luts=int(row["b_luts"]), q_colors=int(row["q_colors"]),
                     total_steps=int(row["total_steps"]),
                     mining=bool(row["mining"]), note=str(row["note"]))


def stage_facts(spec: StageSpec, *, n_train_luts: int) -> dict[str, Any]:
    return {**spec.as_dict(), "n_train_luts_measured": int(n_train_luts),
            "table": "q3vl/whatb/codec/tables.py STAGES (EPR-031 §7-C1)"}


# --------------------------------------------------------------------------- #
# 4. the sampler (EPR-028 R1 §8.1)
# --------------------------------------------------------------------------- #
def paired_anchor_batch(sampler: queries.QuerySampler, b: int, q: int, *,
                        device: Any = "cpu",
                        dtype: torch.dtype = torch.float32
                        ) -> tuple[Tensor, Tensor]:
    """``(x (B, Q*4, 3), s (B, Q*4))`` -- four anchors per colour.

    Each colour draws one ``u ~ U(0,1)`` and is evaluated at
    ``s in {0, 1, u, 1-u}``: the identity endpoint, the LUT endpoint and two
    complementary interior points on the **same** colour.  The anchor axis is the
    fastest-varying one, so ``view(B, Q, 4)`` recovers the colour groups.

    The draw comes from the sampler's private generator -- adding a draw must not
    shift any other random decision in the run.
    """
    colors = sampler.sample(int(b), int(q), device=device, dtype=dtype)
    u = torch.rand(int(b), int(q), generator=sampler.generator, dtype=dtype
                   ).to(device)
    zeros = torch.zeros_like(u)
    s = torch.stack([zeros, zeros + 1.0, u, 1.0 - u], dim=-1)      # (B, Q, 4)
    x = colors.unsqueeze(2).expand(int(b), int(q), S_ANCHORS, 3
                                   ).reshape(int(b), int(q) * S_ANCHORS, 3)
    return x.contiguous(), s.reshape(int(b), int(q) * S_ANCHORS).contiguous()


def select_hard_colour_groups(err: Tensor, ratio: float, b: int, q: int) -> Tensor:
    """Mining on whole colour groups, never on single anchors.

    ``err`` is the per-pair error ``(B, Q*4)``.  It is reduced over the anchor
    axis **first** and :func:`q3vl.whatb.queries.select_hard` then runs on the
    ``(B*Q,)`` colour errors, so a selected colour keeps all four of its anchors
    and an unselected one keeps none.  Selecting on the flat ``(B, Q*4)`` would
    split the group and silently drop ``s = 0`` / ``s = 1`` endpoints.
    """
    group = err.reshape(int(b), int(q), S_ANCHORS).mean(dim=-1).reshape(-1)
    keep = torch.zeros_like(group, dtype=torch.bool)
    idx = queries.select_hard(group, float(ratio))
    if idx.numel():
        keep[idx] = True
    return keep.reshape(int(b), int(q), 1).expand(int(b), int(q), S_ANCHORS
                                                  ).reshape(int(b),
                                                            int(q) * S_ANCHORS)


def r_line_from_anchors(y_hat: Tensor, s: Tensor, b: int, q: int) -> Tensor:
    """``R_line`` at zero extra forward cost (EPR-028 R1 §4.4).

    ``f(x,0)`` and ``f(x,1)`` are anchors 0 and 1 of the same colour group, so
    the chord is already in the batch; the two interior anchors each contribute
    one residual and the two are averaged.
    """
    y = y_hat.reshape(int(b), int(q), S_ANCHORS, 3)
    ss = s.reshape(int(b), int(q), S_ANCHORS)
    f0, f1 = y[:, :, 0, :], y[:, :, 1, :]
    return 0.5 * (g4d.r_line(y[:, :, 2, :], f0, f1, ss[:, :, 2])
                  + g4d.r_line(y[:, :, 3, :], f0, f1, ss[:, :, 3]))


def assert_anchor_structure(s: Tensor, b: int, q: int, *, atol: float = 1e-6
                            ) -> dict[str, Any]:
    """Run-time check of §8.1's structure -- a pre-registered shape with a check.

    Every colour has exactly four anchors, two of which are the exact endpoints
    ``0`` and ``1`` and two of which sum to ``1``.
    """
    ss = s.reshape(int(b), int(q), S_ANCHORS)
    if not bool((ss[..., 0] == 0).all()) or not bool((ss[..., 1] == 1).all()):
        raise AssertionError(
            "the paired-anchor batch lost an endpoint: anchors 0 and 1 must be "
            "exactly s = 0 and s = 1 on every colour (R1 §8.1)")
    comp = (ss[..., 2] + ss[..., 3] - 1.0).abs().max()
    if float(comp) > float(atol):
        raise AssertionError(
            f"the two interior anchors are not complementary (max |u + (1-u) - "
            f"1| = {float(comp)}); R1 §8.1 requires s in {{0, 1, u, 1-u}}")
    return {"n_pairs_s": int(ss.numel()), "s_anchors": S_ANCHORS,
            "max_complement_error": float(comp)}


def telemetry_row(aux: g4d.G4DAux, *, weight_underflow_tau: float | None = None
                  ) -> dict[str, float]:
    """:meth:`g4d.G4DAux.columns` as plain floats, for ``steps.jsonl``."""
    kw: Mapping[str, Any] = ({} if weight_underflow_tau is None
                             else {"weight_underflow_tau": weight_underflow_tau})
    return {k: float(v.detach() if torch.is_tensor(v) else v)
            for k, v in aux.columns(**kw).items()}
