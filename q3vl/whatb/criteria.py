"""HANDOFF §四 in code: headline, the twelve pre-registered keys, B0-B6, N1-N3.

Three rules this module is built around.

**The banned list is enforced by absence** (§4.I, ``EPR-024:763-764``): there is
no AUC function here (in any form), no per-image min-max or softmax
normalisation, no percentile-trimmed (PPL-style) mean, no pooled (low-mixed)
headline, no IoU.  A column that does not exist cannot be reported by accident.
``tests/test_criteria.py`` greps this module's ``def`` lines for those names.

**Everything is measured on the tensor's device.**  No metric path calls
``.cpu()``; the where side moved an IoU by 0.296 by comparing a CUDA top-k with
a CPU one, and the same class of bug on a colour metric would be invisible.
Per-sample scalars are turned into Python floats only *after* the reduction, for
the board.

**Every claim is a paired delta.**  §4.H: 10,000-resample bootstrap 95% CI plus a
Wilcoxon signed-rank p, always against the same samples.  Absolute values live in
the appendix half of a column, never on their own.

Board shape (what an arm writes to ``metrics.json``)::

    {
      "arm": "EPR-024",
      "split": "V_what",
      "contexts": {"style": {...}, "local": {...},
                   "all": {"headline_normal_only": {...},
                           "headline_predalpha": {...}}},
      "criteria_columns": {"headline_normal_only": {"n": 567, ...},
                           "B0_identity": {...}, ..., "N3_const_M": {...}},
      "n_low_excluded": 330,
    }

``.contexts.all.headline_normal_only`` is the one number checkpoint selection may
read (never val loss, never the pooled figure).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .colorimetry import delta_e00, delta_e00_srgb, delta_e76, srgb_to_lab
from .lutdata import mix_alpha

__all__ = [
    "PREREGISTERED_KEYS",
    "REQUIRED_COMMON",
    "REQUIRED_P1",
    "REQUIRED_P2P3",
    "ARM_AXES",
    "required_criteria",
    "CriterionNotComputed",
    "compose_hat",
    "image_delta_e00",
    "function_distance",
    "locality_errors",
    "describe",
    "paired_stats",
    "LibraryValues",
    "bucket_draw",
    "library_random_draw",
    "oracle_lut_ids",
    "control_columns",
    "path_quantities",
    "build_board",
    "assert_criteria_ran",
]

# --------------------------------------------------------------------------- #
# 0. the pre-registered key table (frozen block item 6)
# --------------------------------------------------------------------------- #
#: the twelve frozen key names, in the frozen order.  A spelling that differs
#: from this tuple makes the column *silently absent* while
#: ``assert_criteria_ran`` still passes -- which is exactly why they are frozen.
PREREGISTERED_KEYS: tuple[str, ...] = (
    "headline_normal_only",
    "B0_identity", "B1_libmean", "B2_librandom", "B3_bucket_retrieval", "B4_oracle",
    "N1_shuffle_delta", "N1_shuffle_M",
    "N2_irrelevant_delta", "N2_irrelevant_M",
    "N3_const_delta", "N3_const_M",
)

#: required of EVERY arm (``REQUIRED_EPR024``, HANDOFF §4.H verbatim).  §4.H's
#: prose omits B3 from the "所有 arm 必含" sentence while the code block lists
#: it; the code block wins -- it is the same twelve keys as the frozen table.
REQUIRED_COMMON: frozenset[str] = frozenset(PREREGISTERED_KEYS)

#: P1 arms add the interpolation columns
REQUIRED_P1: frozenset[str] = frozenset(
    {"interp_grid", "path_len", "mono_rate", "oob_rate"})

#: P2 / P3 arms add the locality and field-consumption columns
REQUIRED_P2P3: frozenset[str] = frozenset(
    {"loc_in", "loc_band", "loc_out", "field_const", "field_shuffle", "field_gt"})

#: which question each arm answers (HANDOFF §五 index table).  EPR-027 spans P1
#: and P2, so its required table is the union -- the proposal says so verbatim.
ARM_AXES: dict[str, tuple[str, ...]] = {
    "EPR-024": ("P1",),
    "EPR-025": ("P1",),
    "EPR-026": ("P1",),
    "EPR-027": ("P1", "P2"),
    "EPR-028": ("P2",),
    "EPR-029": ("P3",),
}


class CriterionNotComputed(AssertionError):
    """A pre-registered criterion has no values on the board."""


def required_criteria(arm: str, axes: Sequence[str] | None = None) -> list[str]:
    """The keys ``arm`` may not publish a board without.

    ``axes`` overrides the table for an arm that is not one of the six (a smoke
    run, or a new arm being wired); passing an unknown arm without axes is an
    error rather than an empty requirement -- an empty required table is the
    "defined but not wired" failure this whole mechanism exists to stop.
    """
    if axes is None:
        if arm not in ARM_AXES:
            raise KeyError(
                f"arm {arm!r} is not in ARM_AXES {sorted(ARM_AXES)}; pass "
                "axes=('P1',) explicitly rather than publishing a board with an "
                "empty required table")
        axes = ARM_AXES[arm]
    req = set(REQUIRED_COMMON)
    for ax in axes:
        if ax == "P1":
            req |= REQUIRED_P1
        elif ax in ("P2", "P3"):
            req |= REQUIRED_P2P3
        else:
            raise ValueError(f"unknown axis {ax!r}; expected P1 / P2 / P3")
    return sorted(req)


# --------------------------------------------------------------------------- #
# 1. per-sample primitives (device-side)
# --------------------------------------------------------------------------- #
def compose_hat(img: torch.Tensor, alpha: torch.Tensor | float,
                f_img: torch.Tensor) -> torch.Tensor:
    """The frozen headline image formation ``Î = (1-a) ⊙ I + a ⊙ f̂(I)``.

    One implementation, shared with the data law (``lutdata.mix_alpha``): the
    cross-arm paired delta is only defined if every arm composes the same way.
    ``alpha`` broadcasts against ``img``; ``style`` samples pass ``1.0``.
    """
    return mix_alpha(img, f_img, alpha)


def image_delta_e00(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean dE00 over pixels of two ``(3, H, W)`` (or ``(..., 3)``) sRGB images.

    Returns a 0-dim tensor on the inputs' device.
    """
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
    if a.dim() == 3 and a.shape[0] == 3:
        a = a.permute(1, 2, 0)
        b = b.permute(1, 2, 0)
    return delta_e00_srgb(a, b).mean()


def function_distance(f_a: torch.Tensor, f_b: torch.Tensor,
                      weights: torch.Tensor | None = None,
                      *, metric: str = "de00") -> torch.Tensor:
    """``D_X(f_a, f_b) = sum_x h_x dE(f_a(x), f_b(x))`` on query values.

    ``f_a`` / ``f_b`` are ``(N, 3)`` sRGB function *values* on the same query
    colours.  ``weights`` are the ``h_x`` (``X_img``: the 5-bit histogram
    frequencies); ``None`` means the uniform ``X_grid`` measure, i.e. the mean.
    ``metric`` is ``"de00"`` (headline) or ``"de76"`` (the protocol the
    pre-registered baseline floors were measured in).
    """
    lab_a, lab_b = srgb_to_lab(f_a), srgb_to_lab(f_b)
    d = delta_e00(lab_a, lab_b) if metric == "de00" else delta_e76(lab_a, lab_b)
    if weights is None:
        return d.mean()
    w = weights.to(device=d.device, dtype=d.dtype)
    return (w * d).sum() / w.sum().clamp_min(1e-12)


def locality_errors(i_hat: torch.Tensor, i_star: torch.Tensor,
                    img: torch.Tensor, alpha: torch.Tensor,
                    *, hi: float = 0.9, lo: float = 0.05) -> dict[str, Any]:
    """§4.E three strata, on **GT alpha at short side 512** (frozen block).

    ``E_in`` over ``a >= 0.9``, ``E_band`` over ``0.05 < a < 0.9`` (both against
    ``I*``), ``E_out`` over ``a <= 0.05`` against the **input** ``I``.  Each
    stratum reports its pixel count: ``E_out`` is constructively 0 for the
    mask-blend family, so the three columns are only interpretable together.
    """
    if alpha.dim() == 3 and alpha.shape[0] == 1:
        alpha = alpha[0]
    a = alpha.to(device=i_hat.device, dtype=i_hat.dtype)
    if i_hat.dim() == 3 and i_hat.shape[0] == 3:
        i_hat, i_star, img = (t.permute(1, 2, 0) for t in (i_hat, i_star, img))
    d_star = delta_e00_srgb(i_hat, i_star)
    d_in = delta_e00_srgb(i_hat, img)
    out: dict[str, Any] = {}
    for name, mask, dmap in (("loc_in", a >= hi, d_star),
                             ("loc_band", (a > lo) & (a < hi), d_star),
                             ("loc_out", a <= lo, d_in)):
        n = int(mask.sum())
        out[name] = float(dmap[mask].mean()) if n else None
        out[f"{name}_n_pixels"] = n
    return out


# --------------------------------------------------------------------------- #
# 2. statistics (§4.H)
# --------------------------------------------------------------------------- #
def describe(values: Sequence[float | None]) -> dict[str, Any]:
    """``n / mean / std / p10 / p50 / p90`` of a column, ``None`` dropped."""
    xs = np.asarray([float(v) for v in values if v is not None], dtype=np.float64)
    if xs.size == 0:
        return {"n": 0, "mean": None, "std": None, "p10": None, "p50": None,
                "p90": None}
    return {"n": int(xs.size), "mean": float(xs.mean()),
            "std": float(xs.std(ddof=1)) if xs.size > 1 else 0.0,
            "p10": float(np.percentile(xs, 10)),
            "p50": float(np.percentile(xs, 50)),
            "p90": float(np.percentile(xs, 90))}


def paired_stats(a: Sequence[float | None], b: Sequence[float | None], *,
                 n_boot: int = 10000, seed: int = 20260810) -> dict[str, Any]:
    """Paired ``mean(a - b)`` with a bootstrap 95% CI and a Wilcoxon p (§4.H).

    ``a`` is the arm, ``b`` the comparator, sample-aligned; pairs where either
    side is ``None`` are dropped and counted.  The bootstrap uses its own
    :class:`numpy.random.Generator` -- the global stream is untouched, so adding
    a column cannot move another column's numbers.
    """
    pairs = [(float(x), float(y)) for x, y in zip(a, b)
             if x is not None and y is not None]
    n_dropped = max(len(a), len(b)) - len(pairs)
    if not pairs:
        return {"n": 0, "n_dropped": n_dropped, "delta": None, "ci95": None,
                "p_wilcoxon": None, "n_boot": int(n_boot),
                "test": "wilcoxon_signed_rank"}
    diffs = np.asarray([x - y for x, y in pairs], dtype=np.float64)
    delta = float(diffs.mean())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(int(n_boot), diffs.size))
    boots = diffs[idx].mean(axis=1)
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))

    p: float | None
    if np.allclose(diffs, 0.0):
        p = 1.0
    else:
        from scipy.stats import wilcoxon

        p = float(wilcoxon(diffs, alternative="two-sided",
                           zero_method="wilcox").pvalue)
    return {"n": len(pairs), "n_dropped": n_dropped, "delta": delta,
            "ci95": list(ci), "p_wilcoxon": p, "n_boot": int(n_boot),
            "test": "wilcoxon_signed_rank + percentile bootstrap"}


# --------------------------------------------------------------------------- #
# 3. the retrieval baselines (§4.C)
# --------------------------------------------------------------------------- #
@dataclass
class LibraryValues:
    """``Lib_tr`` evaluated once on a query grid; every B-column reads it.

    ``values`` is ``(n_lut, N, 3)`` and ``lut_ids`` names its rows.  ``lab`` is
    the CIELab of the same tensor, cached because B4 / B6 need pairwise
    distances over the whole library.
    """

    lut_ids: tuple[str, ...]
    x: torch.Tensor
    values: torch.Tensor
    _lab: torch.Tensor | None = field(default=None, repr=False)

    @classmethod
    def build(cls, bank, lut_ids: Sequence[str], x: torch.Tensor) -> "LibraryValues":
        return cls(tuple(lut_ids), x, bank.evaluate_library(list(lut_ids), x))

    @property
    def lab(self) -> torch.Tensor:
        if self._lab is None:
            self._lab = srgb_to_lab(self.values)
        return self._lab

    @property
    def index(self) -> dict[str, int]:
        return {lid: i for i, lid in enumerate(self.lut_ids)}

    def mean_transform(self) -> torch.Tensor:
        """B1: the point-wise library mean ``L̄(x)`` -- still a valid mapping."""
        return self.values.mean(dim=0)

    def distance_to(self, target: torch.Tensor, *, metric: str = "de76"
                    ) -> torch.Tensor:
        """``(n_lut,)`` distance from every library LUT to ``target`` ``(N, 3)``.

        Default ``de76`` on the query grid: the protocol the pre-registered
        floors (B0 32.79 / B1 25.33 / B2 35.37 / B4 9.93 / B6 10.17) were
        measured with -- 9^3 grid, Lab L2, mean over colours.
        """
        lab_t = srgb_to_lab(target)
        d = (delta_e76(self.lab, lab_t[None]) if metric == "de76"
             else delta_e00(self.lab, lab_t[None]))
        return d.mean(dim=1)


def oracle_lut_ids(lib: LibraryValues, targets: Mapping[str, torch.Tensor], *,
                   metric: str = "de76", exclude_self: bool = False
                   ) -> dict[str, tuple[str, float]]:
    """B4 / B6: ``arg min_l D(L_l, target)`` per target, on the query grid.

    ``targets`` maps a sample (or lut) id to that sample's GT transform values
    ``(N, 3)``.  ``exclude_self`` makes it B6 (nearest **other** library LUT).
    The argmin runs on the device; only the resulting index leaves it.
    """
    idx = lib.index
    out: dict[str, tuple[str, float]] = {}
    for key, target in targets.items():
        d = lib.distance_to(target, metric=metric)
        if exclude_self and key in idx:
            d = d.clone()
            d[idx[key]] = torch.inf
        j = int(torch.argmin(d))
        out[key] = (lib.lut_ids[j], float(d[j]))
    return out


def library_random_draw(lut_ids: Sequence[str], n: int, *, repeats: int = 8,
                        seed: int = 20260810) -> list[list[str]]:
    """B2: ``repeats`` independent uniform draws from ``Lib_tr`` per sample."""
    rng = np.random.default_rng(seed)
    pool = list(lut_ids)
    return [[pool[int(i)] for i in rng.integers(0, len(pool), size=n)]
            for _ in range(int(repeats))]


def bucket_draw(buckets: Sequence[str], pools: Mapping[str, Sequence[str]], *,
                repeats: int = 8, seed: int = 20260810
                ) -> list[list[str | None]]:
    """B3: per sample, a uniform draw from its own ``minor`` bucket's train pool.

    ``buckets[i]`` is the evaluation sample's record ``minor``; ``pools`` is
    :func:`q3vl.whatb.splits.bucket_pools` over the train records.  A bucket with
    no train pool yields ``None`` for that sample (counted, never silently
    replaced by a global draw) -- measured on V_what / T_lut_unseen normal-only,
    every bucket is covered, so a ``None`` means the label pipeline changed.

    This is a **bucket-level lower bound**, not exact retrieval: 1-of-77 buckets
    cannot give a ``lut_id`` argmax.  The ceiling of retrieval is B4, never B3.
    """
    rng = np.random.default_rng(seed)
    out: list[list[str | None]] = []
    for _ in range(int(repeats)):
        row: list[str | None] = []
        for bucket in buckets:
            pool = list(pools.get(str(bucket), ()))
            row.append(pool[int(rng.integers(0, len(pool)))] if pool else None)
        out.append(row)
    return out


# --------------------------------------------------------------------------- #
# 4. negative controls (§4.D)
# --------------------------------------------------------------------------- #
def control_columns(e_true: Sequence[float | None], e_ctrl: Sequence[float | None],
                    m_values: Sequence[float | None], *, name: str,
                    seed: int = 20260810) -> dict[str, dict[str, Any]]:
    """The two columns a negative control must publish, both or neither.

    ``<name>_delta`` = paired ``H(ctrl) - H(true)``; ``<name>_M`` = the mean
    function-space distance ``D_Xgrid(f_true, f_ctrl)``.  Reporting only the
    delta cannot tell "ignores the instruction" (delta ~ 0, M ~ 0) from "moves
    but not usefully"; reporting only M can be inflated by parameter noise.
    """
    return {f"{name}_delta": {**paired_stats(e_ctrl, e_true, seed=seed),
                              "quantity": "H(ctrl) - H(true), paired"},
            f"{name}_M": {**describe(m_values),
                          "quantity": "mean D_Xgrid(f_true, f_ctrl) [dE00]"}}


# --------------------------------------------------------------------------- #
# 5. interpolation-path quantities (§二·2.4 / §4.F-B)
# --------------------------------------------------------------------------- #
def path_quantities(f_path: torch.Tensor, *, weights: torch.Tensor | None = None,
                    readout: str | None = "b_star") -> dict[str, Any]:
    """The six path quantities of one interpolation, each with its trivial floor.

    ``f_path`` is ``(K+1, N, 3)``: the transform's values on the query set at
    ``alpha_k = k / K``.  Returns ``path_len`` (L), ``chord`` (L0), ``rho``,
    ``sigma_bar``, ``jump_max`` (``K * max_k delta_k``, **no percentile
    trimming** -- StyleGAN's official PPL trims 1%/99% and hides exactly this),
    ``mono_rate`` (sign agreement of the CIELab mean-b* readout along alpha, with
    its 0.5 random floor) and ``oob_rate``.
    """
    if f_path.dim() != 3 or f_path.shape[-1] != 3:
        raise ValueError(f"f_path must be (K+1, N, 3), got {tuple(f_path.shape)}")
    k_steps = f_path.shape[0] - 1
    if k_steps < 1:
        raise ValueError("f_path needs at least two alpha points")
    deltas = torch.stack([
        function_distance(f_path[i], f_path[i + 1], weights)
        for i in range(k_steps)])
    path_len = deltas.sum()
    chord = function_distance(f_path[0], f_path[-1], weights)
    mean_step = path_len / k_steps
    sigma_bar = (deltas.std(unbiased=False) / mean_step.clamp_min(1e-12))
    jump_max = k_steps * deltas.max()

    lab = srgb_to_lab(f_path)                       # (K+1, N, 3)
    r = lab[..., 2].mean(dim=1) if readout == "b_star" else lab[..., 0].mean(dim=1)
    dr = r[1:] - r[:-1]
    signs = torch.sign(dr)
    dominant = torch.sign(signs.sum()) if signs.sum() != 0 else torch.ones_like(signs[0])
    mono = (signs == dominant).to(dr.dtype).mean()

    oob = ((f_path < 0.0) | (f_path > 1.0)).any(dim=-1).to(f_path.dtype).mean()
    return {
        "path_len": float(path_len), "chord": float(chord),
        "rho": (float(path_len / chord) if float(chord) > 0 else None),
        "sigma_bar": float(sigma_bar), "jump_max": float(jump_max),
        "mono_rate": float(mono), "mono_rate_random_floor": 0.5,
        "oob_rate": float(oob), "k_steps": int(k_steps),
        "note": "jump_max is K*max_k delta_k, no percentile trimming",
    }


# --------------------------------------------------------------------------- #
# 6. the board
# --------------------------------------------------------------------------- #
#: per-sample row keys the board understands (anything else is ignored)
_BASELINE_KEYS = ("B0_identity", "B1_libmean", "B2_librandom",
                  "B3_bucket_retrieval", "B4_oracle", "B6_libfill")
_CONTROLS = (("N1_shuffle", "E_N1_shuffle", "M_N1_shuffle"),
             ("N2_irrelevant", "E_N2_irrelevant", "M_N2_irrelevant"),
             ("N3_const", "E_N3_const", "M_N3_const"))
_SIDE_COLUMNS = ("grid_error", "img_error", "unseen_color_error",
                 "loc_in", "loc_band", "loc_out",
                 "field_gt", "field_const", "field_shuffle",
                 "headline_predalpha")


def _headline_block(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"headline_normal_only": describe([r.get("E_arm") for r in rows]),
            "headline_predalpha": describe(
                [r.get("headline_predalpha") for r in rows]),
            "n": len(rows)}


def build_board(rows: Sequence[Mapping[str, Any]], *, arm: str, split: str,
                extra_columns: Mapping[str, Mapping[str, Any]] | None = None,
                seed: int = 20260810) -> dict[str, Any]:
    """Per-sample rows -> the board an arm publishes.

    Rows carry ``sample_id``, ``winner_confidence``, ``task_type`` and the
    per-sample scalars (``E_arm``, ``E_<baseline>``, ``E_N*``/``M_N*``, the side
    columns).  ``low`` rows are dropped from every column and counted: the
    reporting convention is normal-only and mixing ``low`` in under-counts the
    headline by about 0.031.

    ``extra_columns`` is how an arm registers a criterion this module does not
    own (``interp_grid`` / ``path_len`` / ``mono_rate`` / ``oob_rate`` from the
    interpolation protocol, for instance); each entry must carry an ``n``.
    """
    normal = [r for r in rows if r.get("winner_confidence") == "normal"]
    n_low = len(rows) - len(normal)
    style = [r for r in normal if r.get("task_type") == "style"]
    local = [r for r in normal if r.get("task_type") != "style"]

    arm_e = [r.get("E_arm") for r in normal]
    columns: dict[str, dict[str, Any]] = {
        "headline_normal_only": {**describe(arm_e),
                                 "quantity": "mean dE00(Î, I*) [GT alpha, "
                                             "short side 512]"}}

    for key in _BASELINE_KEYS:
        vals = [r.get(f"E_{key}") for r in normal]
        if key in ("B2_librandom", "B3_bucket_retrieval"):
            reps = [r.get(f"E_{key}_repeats") for r in normal]
            if any(v is not None for v in reps):
                columns[key] = _repeat_column(arm_e, vals, reps, seed=seed)
                continue
        if all(v is None for v in vals):
            continue
        columns[key] = {**describe(vals),
                        "paired_delta_arm_minus_baseline":
                            paired_stats(arm_e, vals, seed=seed)}

    for name, e_key, m_key in _CONTROLS:
        e_ctrl = [r.get(e_key) for r in normal]
        m_vals = [r.get(m_key) for r in normal]
        if all(v is None for v in e_ctrl) and all(v is None for v in m_vals):
            continue
        columns.update(control_columns(arm_e, e_ctrl, m_vals, name=name,
                                       seed=seed))

    for key in _SIDE_COLUMNS:
        vals = [r.get(key) for r in normal]
        if any(v is not None for v in vals):
            columns[key] = describe(vals)

    for key, col in dict(extra_columns or {}).items():
        col = dict(col)
        if "n" not in col:
            raise ValueError(
                f"extra column {key!r} carries no 'n'; assert_criteria_ran "
                "checks n > 0 and a column without one can never be asserted")
        columns[key] = col

    return {
        "arm": arm, "split": split,
        "n_rows": len(rows), "n_normal": len(normal), "n_low_excluded": n_low,
        "contexts": {"style": _headline_block(style),
                     "local": _headline_block(local),
                     "all": _headline_block(normal)},
        "criteria_columns": columns,
        "reporting": ("normal-only; the pooled (low-mixed) headline is not "
                      "computed by this module (§4.I)"),
    }


def _repeat_column(arm_e: Sequence[float | None],
                   mean_vals: Sequence[float | None],
                   repeats: Sequence[Sequence[float] | None], *, seed: int
                   ) -> dict[str, Any]:
    """B2 / B3: R draws per sample -> mean +- std over draws, paired on the mean."""
    per_rep: list[float] = []
    r_max = max((len(x) for x in repeats if x), default=0)
    for j in range(r_max):
        vals = [x[j] for x in repeats if x is not None and j < len(x)]
        if vals:
            per_rep.append(float(np.mean(vals)))
    means = [float(np.mean(x)) if x else None for x in repeats]
    if all(v is None for v in means):
        means = list(mean_vals)
    return {**describe(means),
            "n_repeats": r_max,
            "repeat_means": per_rep,
            "repeat_mean": float(np.mean(per_rep)) if per_rep else None,
            "repeat_std": float(np.std(per_rep, ddof=1)) if len(per_rep) > 1 else 0.0,
            "paired_delta_arm_minus_baseline": paired_stats(arm_e, means,
                                                            seed=seed)}


# --------------------------------------------------------------------------- #
# 7. the runtime assertion
# --------------------------------------------------------------------------- #
def assert_criteria_ran(board: Mapping[str, Any], arm: str, *,
                        axes: Sequence[str] | None = None,
                        required: Sequence[str] | None = None
                        ) -> dict[str, Any]:
    """Every pre-registered criterion must have been computed with ``n > 0``.

    "Defined but not wired" has cost this campaign five times (three on the
    what/where criteria, twice on the where arms last week).  This runs before
    ``metrics.json`` is written -- see :func:`q3vl.whatb.publish.assert_publishable`
    for the call that also checks the training side.
    """
    req = list(required) if required is not None else required_criteria(arm, axes)
    cols = dict(board.get("criteria_columns") or {})
    report: dict[str, Any] = {"arm": arm, "required": req, "computed": {}}
    for name in req:
        n = int((cols.get(name) or {}).get("n", 0) or 0)
        report["computed"][name] = n
        if n == 0:
            raise CriterionNotComputed(
                f"arm {arm} pre-registers {name!r} and the board carries 0 "
                f"values for it (columns present: {sorted(cols)}); refusing to "
                "publish a board that cannot adjudicate its own experiment")
    hn = ((board.get("contexts") or {}).get("all") or {}).get(
        "headline_normal_only") or {}
    report["headline_normal_only_n"] = int(hn.get("n", 0) or 0)
    if "headline_normal_only" in req and report["headline_normal_only_n"] == 0:
        raise CriterionNotComputed(
            f"arm {arm}: the column headline_normal_only is required but "
            ".contexts.all.headline_normal_only carries no rows; checkpoint "
            "selection reads exactly that key and nothing else")
    # Whether an *interim* board may be missing the key at all is a publication
    # question, not a criteria one: q3vl.whatb.publish.assert_publishable knows
    # if this is the published board (evaluate.py:483-484 -- the key exists only
    # when the subset happened to contain a normal row).
    return report


