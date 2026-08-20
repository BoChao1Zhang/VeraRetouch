"""IDGATE (EPR-027): the identity-anchored strength gate, outside the GLUT forward.

    f_u(x, p) = x + u(p) * (f_theta(x) - x)          <- the ONE structural change
    y_u(x)    = (1 - u) * x + u * L_l(x)             <- the synthetic strength target

``theta`` does not depend on ``u``: the gate is a zero-parameter fused multiply-add
sitting *outside* ``glut_forward``, and the generator / projection are byte-identical
to the EPR-024 carrier.  The spec for every number in this file is
``experiments/prs/EPR-027_identity-anchored-strength-gate/PROPOSAL.md``; the shared
layer it builds on is ``q3vl/whatb/{glut,generator,gate,guards,criteria,publish,
lutdata,queries,colorimetry}.py`` (one implementation each, six arms).

The five identities (EPR-027:345-356) -- arithmetic, never results
-------------------------------------------------------------------
=====  ==========================================  ==================================
G1     ``f_u - y_u = u (f_theta - L_l)``           ``L_rec(u) = u L_rec(1)``; the
                                                    balancing rows 1' / 1'' exist
                                                    because of this
G1'    ``C(y_u) != u C(L_l)`` (Lab is non-linear)  row 1'' re-weights ``L_hc`` with
                                                    ``C(y_u)``; ``chroma_weight_src``
                                                    and ``mean_C_weight`` are logged
                                                    per step as the witness
G2     ``||f_u - x|| = u ||f_theta - x||``         the u-monotonicity rate is 1.000 by
                                                    construction -> report section G(b)(c)
                                                    against **lambda**, print the u row
G3     ``u = 0 => f_u = x, L_rec = L_hc = 0``      those draws contribute no gradient to
                                                    theta; ``u0_frac`` counts them
G4     ``u(p) = a(p), f_theta = L_l => f_u = F*``  proposition 3; ``E_out`` is then 0 by
                                                    construction
G5     ``u in [0,1]`` with clamp BEFORE the gate    the out-of-gamut column degenerates
       -> convex combination of in-gamut points     to 0; the main arm clamps AFTER
=====  ==========================================  ==================================

Frozen calibration this module refuses to drift from (eight items, cross-arm block):
train normal-only n = 93934; B = 32 samples x Q = 256 colours = 8192 colours/step;
2936 steps/epoch; 117,440 steps (40 epochs); ``--clamp two``; the headline formation
``I_hat = (1-a) I + a f_hat(I)``; the twelve pre-registered criterion keys; and a
colour-span encoder that never imports ``q3vl.what`` (``q3vl/whatb/colorspan.py``).

Three failures the where side paid for last week, closed structurally here
--------------------------------------------------------------------------
1. *Assertion contract*: the board-time check fetches its own first-step row three
   tiers deep (:func:`q3vl.whatb.guards.resolve_first_step_row`) and "nobody handed me
   a row" raises a different exception from "the row has no loss columns".
2. *device/dtype*: every tensor entering a forward is placed with
   ``.to(device=ref.device, dtype=ref.dtype)``; there is no bare ``torch.tensor(...)``
   in any forward, and the only constants are submodule buffers.
3. *Constant fields*: the first quick eval runs
   :func:`q3vl.whatb.guards.assert_transform_not_degenerate` on ``f_theta`` -- flat
   across query colours, or the identity, or identical across samples, and the process
   leaves with ``SystemExit(2)``.
Plus: every criterion is computed on the tensor's own device (no ``.cpu()`` round trip
before a comparison -- that moved a where-side IoU by 0.296).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from q3vl.whatb import criteria as _criteria
from q3vl.whatb import publish as _publish
from q3vl.whatb import queries as _queries
from q3vl.whatb.caliber import (
    DATA_CHOICES,
    FROZEN_BATCH_SPLITS,
    FROZEN_TRAIN_NORMAL_N,
    assert_steps_per_epoch,
    default_train_normal_n as _active_train_normal_n,
    effective_lambda_hc,
    effective_lambda_sparse,
    pure_l1_record,
)
from q3vl.whatb.colorimetry import chroma_hue, delta_e00, srgb_to_lab
from q3vl.whatb.gate import IdentityGate, identity_gate
from q3vl.whatb.generator import CGLUTGenerator, SegColorProjection, SEG_COLOR_HIDDEN_DIM
from q3vl.whatb.glut import EPS, GlutParams, glut_forward
from q3vl.whatb.guards import (
    DegeneracyThresholds,
    assert_transform_not_degenerate,
    record_step_witness,
)
from q3vl.whatb.lutdata import apply_lut_volume, mix_alpha

__all__ = [
    "ARM",
    "EPR",
    "AXES",
    "ARM_CRITERIA",
    "REQUIRED_CRITERIA",
    "U_SOURCES",
    "U_SOURCE_RESOLUTION",
    "GATE_CLAMP_CHOICES",
    "U_DISTS",
    "LAMBDA_SIGNS",
    "NULL_PROMPTS",
    "FIELD_SRCS",
    "DEFAULT_U_EVAL",
    "STRENGTH_U",
    "EXTRAP_U",
    "INTERP_ALPHAS",
    "IdGateConfig",
    "IdGateArm",
    "GateOutput",
    "LossTerms",
    "UDraw",
    "sample_u",
    "lambda_sign",
    "effective_lambda_u",
    "strength_target",
    "glut_loss",
    "gate_probe_loss",
    "mine_hard_colors",
    "build_optimizer",
    "build_scheduler",
    "gate_identity_check",
    "assert_gate_identity",
    "first_quick_eval_guard",
    "train_step",
    "step_columns",
    "loss_preregistration",
    "assert_frozen_organisation",
    "EvalSample",
    "GridSpec",
    "LibraryContext",
    "evaluate_sample",
    "strength_columns",
    "degeneracy_columns",
    "interpolation_columns",
    "extra_criteria_columns",
    "build_arm_board",
    "publish_arm_board",
]

# --------------------------------------------------------------------------- #
# 0. identity of the arm
# --------------------------------------------------------------------------- #
ARM = "IDGATE"
EPR = "EPR-027"
#: EPR-027 spans P1 (the strength axis) and P2 (the spatial axis); the required
#: criteria table is the **union** (EPR-027:620).
AXES: tuple[str, ...] = ("P1", "P2")

#: the four columns only this arm has (EPR-027:635-639, verbatim key names)
ARM_CRITERIA: tuple[str, ...] = (
    "gate_identity_check",   # u = 1 -> max|f_u - f_theta| == 0 (the gate is wired)
    "gate_u_hist",           # the u values training actually used, n > 0
    "strength_dE_u",         # dE00(f_hat_u, y_u), u in {0,.25,.5,.75,1}
    "dlib_u",                # nearest library distance at u in {-0.5,0,...,2}
)

#: the full required table = the twelve frozen keys + P1 + P2/P3 + this arm's four.
REQUIRED_CRITERIA: tuple[str, ...] = tuple(
    sorted(set(_criteria.required_criteria(EPR, AXES)) | set(ARM_CRITERIA))
)

# --------------------------------------------------------------------------- #
# 1. the flags (EPR-027:462), with the proposal's conservative defaults
# --------------------------------------------------------------------------- #
#: ``--gate-u-source``.  Training is fixed to ``sample``; the others are the
#: evaluation ladder of section 4.1 (rows 2 / 2' / 3-a / 3-b / 4) and section 4.3.
U_SOURCES: tuple[str, ...] = ("sample", "gt_alpha", "lambda", "zhead", "const", "shuffle")
#: ``--gate-clamp``.  ``after`` (default) keeps the out-of-gamut column alive; with
#: ``before`` identity G5 makes it 0 for every ``u`` in [0,1].
GATE_CLAMP_CHOICES: tuple[str, ...] = ("after", "before")
#: ``--gate-u-dist``.  ``excl_band`` = ``U([0,1] \\ (0, 0.1))``, Baumann Algorithm 1's
#: "exclude the near-zero band" shape (ablation row 6); ``p_end`` is not applied there,
#: which is why that row's ``u0_frac`` is 0 by construction.
U_DISTS: tuple[str, ...] = ("uniform01", "excl_band")
#: ``--gate-lambda-sign``: ``random`` is ``learn_delta.py:85``'s random sign, consumed
#: as ``u = clamp(lambda, 0, 1)``.
LAMBDA_SIGNS: tuple[str, ...] = ("fixed", "random")
#: ``--gate-null-prompt``.  ``keep`` is the proposal's NOVEL default; ``cliptone`` is
#: CLIPtone section 4's literal neutral source description.
NULL_PROMPTS: dict[str, str] = {
    "keep": "Please keep the colors of this photo unchanged.",
    "cliptone": "normal photo",
}
#: ``--gate-field-src``: the four rows of the locality table (section 4.3).
FIELD_SRCS: tuple[str, ...] = ("gt", "pred", "const", "shuffle")
#: ``--gate-u-eval`` default (EPR-027:462): the extrapolation points included.
DEFAULT_U_EVAL: tuple[float, ...] = (-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
#: criterion section G(a): dE00(f_hat_u, y_u) is reported at exactly these u.
STRENGTH_U: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
#: criterion section G(d): the out-of-[0,1] extrapolation points.
EXTRAP_U: tuple[float, ...] = (-0.5, 1.5, 2.0)
#: criterion section F / IP-A: GLUT App B.3's own alpha grid.
INTERP_ALPHAS: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
#: section 4.2 reports ``d_lib`` at these u (EPR-027:700 header).
DLIB_U: tuple[float, ...] = (-0.5, 0.0, 1.0, 2.0)

#: frozen block items 1-3: the numbers a published run must show.  These are
#: the ORIGINAL index口径 (v20260804); the active口径 may differ and the run
#: records its own measured n next to them.
FROZEN_TRAIN_N = FROZEN_TRAIN_NORMAL_N
FROZEN_BATCH_SAMPLES = 32
FROZEN_QUERIES = 256
FROZEN_COLORS_PER_STEP = 8192
FROZEN_STEPS_PER_EPOCH = 2936
FROZEN_EPOCHS = 40
FROZEN_TOTAL_STEPS = 117_440


# --------------------------------------------------------------------------- #
# 2. configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class IdGateConfig:
    """Every knob of the arm, exactly as ``run_setup.json`` will record it.

    Defaults are the proposal's main arm.  Where the proposal marks a value NOVEL
    (``gate_p_end``, ``gate_zhead_weight``, the neutral prompt) the default is the
    conservative one it writes down, never a new invention.
    """

    # ---- carrier / generator (unchanged from EPR-024) ----
    n_gauss: int = 48
    cond_dim: int = 64
    gen_width: int = 128
    clamp: str = "two"
    residual: bool = True
    in_dim: int = SEG_COLOR_HIDDEN_DIM

    # ---- the gate (this arm's only structural change) ----
    gate: bool = True
    gate_u_source: str = "sample"
    gate_clamp: str = "after"
    gate_p_end: float = 0.2          # NOVEL (EPR-027:426), no source
    gate_ku: int = 4                 # Baumann appendix / learn_delta.yaml scale_batch_size
    gate_u_dist: str = "uniform01"
    gate_lambda_sign: str = "fixed"
    gate_null_prompt: str = "keep"
    gate_zhead: str = "none"         # "linear" = row 3-b's probe
    gate_zhead_weight: float = 0.1   # NOVEL (EPR-027:428)
    gate_field_src: str = "gt"
    gate_u_eval: tuple[float, ...] = DEFAULT_U_EVAL
    #: EPR-027:461 spells the wiring assertion as ``max|f_u - f_theta| == 0``.
    #: That equality does not hold in floating point for ``x + u (y - x)``: with
    #: ``x = 0.9`` and ``y = 1e-7`` (both reachable -- ``y`` is the pre-clamp GLUT
    #: value) the subtraction rounds to ``-0.9`` and the sum returns ``0.0``, so the
    #: residual is ``|y|``.  Measured on a fresh model it is about 3e-8, a quarter of
    #: ``eps(float32)``.  ``None`` (default) therefore compares against
    #: ``4 * finfo(dtype).eps`` and **records both** the floor and the measured value
    #: on the board; passing ``0.0`` reproduces the proposal's literal assertion and
    #: is expected to fail on some batches.  See NOTES in the return report.
    gate_identity_atol: float | None = None
    gate_identity_ulps: int = 4

    # ---- loss (GLUT Eq.6-8, weights from GLUT section 4.1) ----
    lambda_hc: float = 10.0
    lambda_sparse: float = 0.001
    hc_eps_c: float = 1e-3
    hc_mask: bool = True
    loss_level: int = 3
    #: ablation row 1': ``L_rec`` weight x0.5, the analytic half of identity G1.
    l_rec_scale: float = 1.0
    #: ablation row 1'': the ``L_hc`` chroma weight switches to ``C(y_u)`` while the
    #: target stays ``L_l(x)`` -- identity G1' says this has no closed-form balance.
    chroma_weight_src: str = "target"     # "target" | "y_u"

    # ---- optimiser (GLUT section 4.1 + App A.1) ----
    lr: float = 1e-3
    pi_lr_scale: float = 0.1
    adam_betas: tuple[float, float] = (0.9, 0.999)   # PyTorch default, logged as a deviation
    weight_decay: float = 0.0
    grad_clip: float | None = None                    # EPR-027:449 "not added"
    epochs: int = FROZEN_EPOCHS
    batch_samples: int = FROZEN_BATCH_SAMPLES
    queries: int = FROZEN_QUERIES
    #: MEASURED from ``--data`` by the runner (never a literal on a real run)
    #: the active index口径's declared train normal-only n; a runner overrides
    #: it with the measured population
    train_n: int = field(default_factory=_active_train_normal_n)
    #: which training corpora the population is drawn from (``--data``)
    data: str = "v2seg"
    seed: int = 20260810
    mining: bool = True
    precision: str = "bf16"

    # ---- evaluation ----
    field_resolution: int = 512       # short side; frozen block, the where-field rule
    grid_n: int = 17                  # X_grid of section 4.B
    floor_grid_n: int = 9             # the grid the pre-registered floors were measured on
    b1_grid_n: int = 33               # sampling grid the library-mean transform is baked on
    hist_bits: int = 5
    hist_top_k: int = 4096
    n_repeats: int = 8                # B2 / B3 draws
    n_boot: int = 10000
    degeneracy: DegeneracyThresholds = field(default_factory=DegeneracyThresholds)

    def __post_init__(self) -> None:
        _one_of("clamp", self.clamp, ("two", "one"))
        _one_of("gate_u_source", self.gate_u_source, U_SOURCES)
        _one_of("gate_clamp", self.gate_clamp, GATE_CLAMP_CHOICES)
        _one_of("gate_u_dist", self.gate_u_dist, U_DISTS)
        _one_of("gate_lambda_sign", self.gate_lambda_sign, LAMBDA_SIGNS)
        _one_of("gate_null_prompt", self.gate_null_prompt, tuple(NULL_PROMPTS))
        _one_of("gate_zhead", self.gate_zhead, ("none", "linear"))
        _one_of("gate_field_src", self.gate_field_src, FIELD_SRCS)
        _one_of("chroma_weight_src", self.chroma_weight_src, ("target", "y_u"))
        if not 0.0 <= self.gate_p_end <= 1.0:
            raise ValueError(f"--gate-p-end must be in [0,1], got {self.gate_p_end}")
        if self.gate_ku < 1:
            raise ValueError(f"--gate-ku must be >= 1, got {self.gate_ku}")
        if self.loss_level not in (1, 2, 3, 4):
            raise ValueError(f"--loss-level must be 1..4, got {self.loss_level}")
        if self.loss_level == 4:
            raise NotImplementedError(
                "--loss-level 4 adds L_img, which EPR-027 does not pre-register; "
                "the arm's loss is L_rec + 10 L_hc + 0.001 R_sparse (+ w_gate L_gate)")
        if self.gate_identity_atol is not None and self.gate_identity_atol < 0:
            raise ValueError("--gate-identity-atol must be >= 0 or unset")
        if not self.gate and self.gate_zhead != "none":
            raise ValueError("--no-gate with --gate-zhead linear: the probe would "
                             "train against a gate that is not in the graph")

    # ---- derived, frozen-block arithmetic ----
    @property
    def colors_per_step(self) -> int:
        return int(self.batch_samples) * int(self.queries)

    @property
    def batch_split(self) -> str:
        """``"BxQ"`` -- the row of the ONE shared table (``BATCH_SPLITS``)."""
        return f"{int(self.batch_samples)}x{int(self.queries)}"

    @property
    def step_matched_to_epr024(self) -> bool:
        return self.batch_split in FROZEN_BATCH_SPLITS

    @property
    def lambda_hc_effective(self) -> float:
        """``arms/carrier.py:347``'s ladder rule (``--loss-level 1`` -> 0)."""
        return effective_lambda_hc(self.lambda_hc, self.loss_level)

    @property
    def lambda_sparse_effective(self) -> float:
        """``arms/carrier.py:351``'s ladder rule (``--loss-level 1/2`` -> 0)."""
        return effective_lambda_sparse(self.lambda_sparse, self.loss_level)

    @property
    def steps_per_epoch(self) -> int:
        return int(math.ceil(self.train_n / self.batch_samples))

    @property
    def total_steps(self) -> int:
        return self.steps_per_epoch * int(self.epochs)

    @property
    def n_pairs_per_step(self) -> int:
        """``8192 * K_u`` -- NOTES 12: the extra u draws enter the batch, not the
        step count, so the optimiser step count matches EPR-024 bit for bit (U4)."""
        return self.colors_per_step * (self.gate_ku if self.gate else 1)

    @property
    def k_u(self) -> int:
        """Draws of ``u`` per sample; 1 when the gate is off (row 1 = EPR-024)."""
        return int(self.gate_ku) if self.gate else 1

    @property
    def null_prompt(self) -> str:
        return NULL_PROMPTS[self.gate_null_prompt]

    def identity_atol(self, dtype: torch.dtype) -> tuple[float, str]:
        """``(atol, source)`` for the ``u = 1`` wiring assertion at this dtype."""
        if self.gate_identity_atol is not None:
            return float(self.gate_identity_atol), "flag"
        return (float(self.gate_identity_ulps) * float(torch.finfo(dtype).eps),
                f"{self.gate_identity_ulps}*eps({dtype})")

    def to_dict(self) -> dict[str, Any]:
        out = {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}
        out["degeneracy"] = self.degeneracy.as_dict()
        out.update({
            "arm": ARM, "epr": EPR, "axes": list(AXES),
            "colors_per_step": self.colors_per_step,
            "colours_per_step": self.colors_per_step,
            "batch_split": self.batch_split,
            "batch_split_step_matched_to_epr024": self.step_matched_to_epr024,
            "steps_per_epoch": self.steps_per_epoch,
            "total_steps": self.total_steps,
            "lambda_hc_effective": self.lambda_hc_effective,
            "lambda_sparse_effective": self.lambda_sparse_effective,
            "lambda_mono_effective": 0.0,
            "n_pairs_per_step": self.n_pairs_per_step,
            "null_prompt_text": self.null_prompt,
            "required_criteria": list(REQUIRED_CRITERIA),
            "identities": ["G1", "G1'", "G2", "G3", "G4", "G5"],
        })
        return out


def _one_of(name: str, value: Any, allowed: Sequence[Any]) -> None:
    if value not in allowed:
        raise ValueError(f"{name}={value!r} is not one of {tuple(allowed)}")


def assert_frozen_organisation(cfg: IdGateConfig) -> dict[str, Any]:
    """The batch organisation's **internal** arithmetic, checked at start-up.

    Until 2026-08-16 this refused any run that was not ``32 x 256 = 8192`` /
    ``ceil(93934/32) = 2936`` / ``117,440``.  EPR-030 opened both the colour
    budget (``arms/carrier.py`` ``BATCH_SPLITS``, 17 rows) and the corpus
    (``--data v2seg+l8``), so the EPR-024 numbers are now **recorded** next to
    the run's own -- ``step_matched_to_epr024`` says whether a cross-arm paired
    delta against the EPR-024 board is step-matched at all.

    What still raises, because it is arithmetic and not a caliber:

    * ``colors_per_step == B * Q``;
    * ``steps_per_epoch == ceil(train_n / B)`` (the shared
      :func:`q3vl.whatb.caliber.assert_steps_per_epoch`);
    * ``total_steps == steps_per_epoch * epochs``.
    """
    facts = {
        "data": cfg.data,
        "train_n": cfg.train_n,
        "batch_split": cfg.batch_split,
        "batch_samples": cfg.batch_samples,
        "queries": cfg.queries,
        "colors_per_step": cfg.colors_per_step,
        "colours_per_step": cfg.colors_per_step,
        "steps_per_epoch": cfg.steps_per_epoch,
        "epochs": cfg.epochs,
        "total_steps": cfg.total_steps,
        "n_pairs_per_step": cfg.n_pairs_per_step,
        "k_u": cfg.k_u,
    }
    if cfg.colors_per_step != cfg.batch_samples * cfg.queries:   # pragma: no cover
        raise ValueError("colors_per_step is not B * Q")
    assert_steps_per_epoch(cfg.steps_per_epoch, n_train=cfg.train_n,
                           batch_samples=cfg.batch_samples,
                           where="idgate steps_per_epoch")
    if cfg.total_steps != cfg.steps_per_epoch * cfg.epochs:
        raise ValueError(
            f"total_steps {cfg.total_steps} != steps_per_epoch "
            f"{cfg.steps_per_epoch} * epochs {cfg.epochs}")
    epr024 = {
        "train_n": FROZEN_TRAIN_N,
        "batch_samples": FROZEN_BATCH_SAMPLES,
        "queries": FROZEN_QUERIES,
        "colors_per_step": FROZEN_COLORS_PER_STEP,
        "steps_per_epoch": FROZEN_STEPS_PER_EPOCH,
        "epochs": FROZEN_EPOCHS,
        "total_steps": FROZEN_TOTAL_STEPS,
    }
    facts["epr024_frozen"] = epr024
    facts["differs_from_epr024"] = {
        k: [facts[k], v] for k, v in epr024.items() if facts[k] != v}
    facts["batch_split_step_matched_to_epr024"] = cfg.step_matched_to_epr024
    facts["step_matched_to_epr024"] = not facts["differs_from_epr024"]
    return facts


# --------------------------------------------------------------------------- #
# 3. the u sampler (EPR-027:426 / :459-2)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class UDraw:
    """One step's ``u`` draw plus the columns ``steps.jsonl`` must carry."""

    u: Tensor              # (B, K_u)
    n_forced: int
    n_zero: int
    n_one: int

    @property
    def stats(self) -> dict[str, Any]:
        n = int(self.u.numel())
        return {
            "u_mean": float(self.u.mean()) if n else None,
            "u0_frac": (self.n_zero / n) if n else None,
            "u1_frac": (self.n_one / n) if n else None,
            "u_forced_frac": (self.n_forced / n) if n else None,
            "n_u": n,
        }


def sample_u(
    batch: int,
    k_u: int,
    *,
    generator: torch.Generator,
    p_end: float = 0.2,
    dist: str = "uniform01",
    device: Any = "cpu",
    dtype: torch.dtype = torch.float32,
) -> UDraw:
    """``(B, K_u)`` strengths from a **private** generator.

    ``uniform01``  ``u ~ U(0,1)``, then with probability ``p_end`` the draw is forced
                   to an endpoint, half to 0 and half to 1 (the proposal's wording).
                   Identity G3 makes the ``u = 0`` half a zero-gradient draw for
                   ``theta``, which is why ``u0_frac`` is a logged column and not an
                   implementation detail.
    ``excl_band``  ``u ~ U([0,1] \\ (0, 0.1))`` = ``0.1 + 0.9 U(0,1)`` (Baumann
                   Algorithm 1's shape).  ``p_end`` is **not** applied: ablation row 6
                   pre-registers ``u0_frac = 0`` for this row.

    The generator is this arm's own :class:`torch.Generator` -- the global torch
    stream is never touched, so adding or removing the u draw cannot move any other
    random decision in the run.  Draws happen on the generator's device and are moved
    afterwards, so CPU and CUDA runs are bit-identical.
    """
    _one_of("gate_u_dist", dist, U_DISTS)
    shape = (int(batch), int(k_u))
    base = torch.rand(shape, generator=generator, dtype=dtype)
    if dist == "excl_band":
        u = 0.1 + 0.9 * base
        return UDraw(u=u.to(device=device), n_forced=0, n_zero=0, n_one=0)

    u = base
    n_forced = n_zero = n_one = 0
    if p_end > 0.0:
        pick = torch.rand(shape, generator=generator, dtype=dtype)
        side = torch.rand(shape, generator=generator, dtype=dtype)
        forced = pick < float(p_end)
        to_one = forced & (side < 0.5)
        to_zero = forced & ~to_one
        u = torch.where(to_zero, torch.zeros_like(u), u)
        u = torch.where(to_one, torch.ones_like(u), u)
        n_forced, n_zero, n_one = int(forced.sum()), int(to_zero.sum()), int(to_one.sum())
    return UDraw(u=u.to(device=device), n_forced=n_forced, n_zero=n_zero, n_one=n_one)


def strength_target(x: Tensor, lut_values: Tensor, u: Tensor | float) -> Tensor:
    """``y_u(x) = (1-u) x + u L_l(x)`` -- the same mixer as the data law.

    Deliberately :func:`q3vl.whatb.lutdata.mix_alpha` (``rendering.py:311-313``,
    endpoint snapping included) rather than a second ``(1-a)*x + a*y``: the strength
    target and the dataset's own GT differ only in what ``u`` is, and one of the two
    drifting from the other would be invisible in the headline.
    """
    return mix_alpha(x, lut_values.to(device=x.device, dtype=x.dtype), u)


# --------------------------------------------------------------------------- #
# 4. the model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GateOutput:
    """What one gated forward hands back.  Every tensor is on the input's device."""

    y: Tensor                     # final, clamped output   (B, *S, 3)
    y_pre_clamp: Tensor           # the same value BEFORE the final clamp -> A(u)
    y_raw: Tensor                 # f_theta(x) as fed to the gate (pre-clamp under
                                  # --gate-clamp after, clamped under before)
    u: Tensor | None              # the strength actually used, broadcastable to y
    opacity: Tensor               # (B, N) sigmoid(logit) -- R_sparse reads it
    influence_sum: Tensor | None  # (B, P) sum_j p_j o_j BEFORE +eps (degenerate-weight rate)
    degenerate_precision: Tensor  # (B, N) bool, demo :496 fallback fired

    def oob_mask(self, lo: float = 0.0, hi: float = 1.0) -> Tensor:
        """``(B, *S)`` bool: the gated value left the gamut before the final clamp."""
        return ((self.y_pre_clamp < lo) | (self.y_pre_clamp > hi)).any(dim=-1)


class IdGateArm(nn.Module):
    """``pi -> CGLUT generator -> GLUT forward -> identity gate -> clamp``.

    Trainable: ``pi`` (LayerNorm(2560) + Linear(2560, d), 169,024), the generator
    (278,188 at d=64 / H=128 / N=48, full generation) and -- only in row 3-b -- a
    2,561-parameter linear probe whose weight is zero-initialised so ``u`` starts at
    ``sigmoid(0) = 0.5``.  Frozen: the whole Qwen3-VL base, which never enters this
    module; ``z`` arrives as a cached ``(B, 2560)`` read-out.

    The gate itself has **no parameters** and ``u`` has **no default** anywhere on the
    path: a gate that can be called without a strength is a gate that can be wired and
    do nothing, and the board could not tell.
    """

    def __init__(self, cfg: IdGateConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.pi = SegColorProjection(in_dim=cfg.in_dim, cond_dim=cfg.cond_dim)
        self.generator = CGLUTGenerator(
            cond_dim=cfg.cond_dim, hidden=cfg.gen_width, n_gauss=cfg.n_gauss,
            mode="full",
        )
        self.gate = IdentityGate(clamp_after=(cfg.gate_clamp == "after"))
        if cfg.gate_zhead == "linear":
            probe = nn.Linear(cfg.in_dim, 1)
            nn.init.zeros_(probe.weight)   # StatLUT section 3.2 / Baumann base.py:52
            nn.init.zeros_(probe.bias)
            self.u_probe: nn.Linear | None = probe
        else:
            self.u_probe = None

    # ---- bookkeeping -----------------------------------------------------
    def extra_repr(self) -> str:
        return (f"arm={ARM}, gate={self.cfg.gate}, gate_clamp={self.cfg.gate_clamp}, "
                f"clamp={self.cfg.clamp}, n_gauss={self.cfg.n_gauss}")

    @property
    def param_counts(self) -> dict[str, int]:
        counts = {
            "pi": sum(p.numel() for p in self.pi.parameters()),
            "generator": sum(p.numel() for p in self.generator.parameters()),
            "gate": 0,
            "u_probe": 0 if self.u_probe is None else
            sum(p.numel() for p in self.u_probe.parameters()),
        }
        counts["total"] = sum(counts.values())
        counts["theta_dim"] = self.generator.theta_dim
        return counts

    @property
    def config(self) -> dict[str, Any]:
        return {**self.cfg.to_dict(),
                "generator": self.generator.config,
                "gate_module": self.gate.config,
                "param_counts": self.param_counts,
                "carrier": {"clamp": self.cfg.clamp, "residual": self.cfg.residual,
                            "eps": EPS,
                            "final_clamp_position": ("after the gate"
                                                     if self._final_clamp_after
                                                     else "inside the carrier")}}

    @property
    def _final_clamp_after(self) -> bool:
        """Does the final ``clamp(.,0,1)`` sit after the gate?

        ``--gate-clamp after`` (the main arm) moves GLUT Eq.5's terminal clamp out
        past the gate; the demo's *global-branch* pre-clamp (``:574-579``, the second
        half of ``--clamp two``) stays inside the carrier, so the two knobs remain
        independent exactly as EPR-027:447 says.  Implemented by reading
        ``GlutAux.pre_clamp`` rather than by switching the carrier to ``clamp="none"``,
        which would drop the global pre-clamp as well.
        """
        return bool(self.cfg.gate) and self.cfg.gate_clamp == "after"

    # ---- condition path --------------------------------------------------
    def z_lambda(self, z: Tensor, z_null: Tensor | None = None,
                 lam: float | Tensor = 1.0) -> Tensor:
        """``z_lambda = z_null + lambda (z - z_null)`` (CLIPtone section 4's direction).

        ``lam == 1`` returns ``z`` **bit for bit** (``z_null + 1*(z - z_null)`` is not
        used in that case), so the lambda path is an exact bypass in the main arm.
        """
        if z_null is None:
            return z
        zn = z_null.to(device=z.device, dtype=z.dtype)
        if isinstance(lam, (int, float)):
            if float(lam) == 1.0:
                return z
            return zn + float(lam) * (z - zn)
        lam_t = lam.to(device=z.device, dtype=z.dtype)
        while lam_t.dim() < z.dim():
            lam_t = lam_t.unsqueeze(-1)
        return zn + lam_t * (z - zn)

    def theta(self, z: Tensor, z_null: Tensor | None = None,
              lam: float | Tensor = 1.0) -> GlutParams:
        """``theta = G(pi(z_lambda))``.  ``theta`` never sees ``u``."""
        return self.generator(self.pi(self.z_lambda(z, z_null, lam)))

    def predict_u(self, z: Tensor, z_null: Tensor | None = None,
                  lam: float | Tensor = 1.0) -> Tensor:
        """Row 3-b: ``u = sigmoid(w^T z_lambda + b)``, ``(B, 1)``."""
        if self.u_probe is None:
            raise ValueError(
                "--gate-zhead none: there is no probe to read u from.  Row 3-a takes "
                "u = clamp(lambda, 0, 1); row 3-b needs --gate-zhead linear.")
        zl = self.z_lambda(z, z_null, lam)
        ref = self.u_probe.weight
        return torch.sigmoid(self.u_probe(zl.to(device=ref.device, dtype=ref.dtype)))

    # ---- the forward -----------------------------------------------------
    def apply_transform(
        self,
        x: Tensor,
        params: GlutParams,
        *,
        u: Tensor | float | None,
        point_chunk: int | None = None,
        need_influence: bool = False,
    ) -> GateOutput:
        """``x -> f_theta -> gate -> clamp`` for query colours ``(B, P, 3)`` / ``(P, 3)``.

        ``u`` is required whenever the gate is on: scalar, ``(B,1)``, ``(B,P,1)`` or a
        per-pixel field.  Chunked over points so the ``(B, P, N)`` auxiliary the
        pre-clamp value comes from never allocates a whole image at once; chunking is
        numerically inert (points are independent).
        """
        if self.cfg.gate and u is None:
            raise TypeError(
                "the gate is on and u was not given.  EPR-027:459 makes u an explicit "
                "formal parameter with no default precisely so 'the gate is wired but "
                "does nothing' cannot happen silently.")

        want_pre = self._final_clamp_after or need_influence
        xb = x if x.dim() >= 3 else x.unsqueeze(0)
        lead = max(int(xb.shape[0]), params.batch_size)
        n_pts = int(xb.reshape(int(xb.shape[0]), -1, 3).shape[1])
        chunk = int(point_chunk) if point_chunk else max(1, (1 << 22) // max(1, params.n_gauss))

        ys: list[Tensor] = []
        infl: list[Tensor] = []
        opacity = degen = None
        flat = xb.reshape(int(xb.shape[0]), -1, 3)
        for start in range(0, max(n_pts, 1), chunk):
            xs = flat[:, start:start + chunk, :]
            if xs.shape[1] == 0:
                continue
            if want_pre:
                out, aux = glut_forward(xs, params, clamp=self.cfg.clamp,
                                        residual=self.cfg.residual, return_aux=True)
                raw = aux.pre_clamp if self._final_clamp_after else out
                if need_influence:
                    infl.append(aux.influence_sum)
                opacity, degen = aux.opacity, aux.degenerate_precision
            else:
                raw = glut_forward(xs, params, clamp=self.cfg.clamp,
                                   residual=self.cfg.residual)
            ys.append(raw)
        y_raw = torch.cat(ys, dim=1) if ys else flat.new_zeros((lead, 0, 3))
        if opacity is None:
            _, _, opacity, degen = _geometry_of(params)

        spatial = tuple(xb.shape[1:-1])
        y_raw = y_raw.reshape(lead, *spatial, 3)
        x_ref = xb.to(device=y_raw.device, dtype=y_raw.dtype)
        if x_ref.shape[0] != lead:
            x_ref = x_ref.expand(lead, *x_ref.shape[1:])

        if not self.cfg.gate:
            y_pre = y = y_raw
            u_used = None
        else:
            u_used = _as_u_tensor(u, y_raw)
            y_pre = identity_gate(x_ref, y_raw, u_used, clamp=False)
            y = y_pre.clamp(0.0, 1.0) if self._final_clamp_after else y_pre
        influence = torch.cat(infl, dim=1).reshape(lead, *spatial) if infl else None
        return GateOutput(y=y, y_pre_clamp=y_pre, y_raw=y_raw, u=u_used,
                          opacity=opacity, influence_sum=influence,
                          degenerate_precision=degen)

    def forward(
        self,
        x: Tensor,
        z: Tensor,
        *,
        z_null: Tensor | None = None,
        u: Tensor | float | None = None,
        lam: float | Tensor = 1.0,
        point_chunk: int | None = None,
        need_influence: bool = False,
    ) -> GateOutput:
        """The pseudo-code of EPR-027:320-334, one call."""
        params = self.theta(z, z_null, lam)
        if self.cfg.gate and u is None and self.cfg.gate_zhead == "linear":
            u = self.predict_u(z, z_null, lam).unsqueeze(-1)   # (B,1,1) over (B,P,3)
        return self.apply_transform(x, params, u=u, point_chunk=point_chunk,
                                    need_influence=need_influence)

    def apply_image(self, img: Tensor, params: GlutParams, *,
                    u: Tensor | float | None, point_chunk: int = 1 << 16) -> GateOutput:
        """``(3,H,W)`` or ``(B,3,H,W)`` sRGB in [0,1] -> the same, gated.

        ``u`` may be a per-pixel field ``(H,W)`` / ``(B,1,H,W)``: that is row 4, the
        spatial axis.  The image is transposed to ``(B,H,W,3)`` first so the gate's
        broadcast is over the trailing channel axis.
        """
        single = img.dim() == 3
        x = (img.unsqueeze(0) if single else img).permute(0, 2, 3, 1)
        if isinstance(u, Tensor):
            u = _field_to_hwc(u, x)
        out = self.apply_transform(x, params, u=u, point_chunk=point_chunk)
        back = lambda t: (t.permute(0, 3, 1, 2)[0] if single else t.permute(0, 3, 1, 2))
        return GateOutput(y=back(out.y), y_pre_clamp=back(out.y_pre_clamp),
                          y_raw=back(out.y_raw), u=out.u, opacity=out.opacity,
                          influence_sum=out.influence_sum,
                          degenerate_precision=out.degenerate_precision)

    # ---- optimiser groups -------------------------------------------------
    def param_groups(self, base_lr: float | None = None) -> list[dict[str, Any]]:
        """CGLUT App A.1: the condition-side parameters train at 0.1x base lr.

        ``pi`` stands where CGLUT's learnable ``e_l`` stood, so the 0.1x group is
        ``pi`` (and the row 3-b probe, the other condition-side linear piece); the
        generator takes the full 1e-3.  Ablation row 11 is ``--pi-lr-scale 1.0``.
        """
        lr = float(self.cfg.lr if base_lr is None else base_lr)
        slow = list(self.pi.parameters())
        if self.u_probe is not None:
            slow += list(self.u_probe.parameters())
        groups = [
            {"params": list(self.generator.parameters()), "lr": lr, "name": "generator"},
            {"params": slow, "lr": lr * float(self.cfg.pi_lr_scale), "name": "condition_side"},
        ]
        return [g for g in groups if g["params"]]


def _geometry_of(params: GlutParams) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    from q3vl.whatb.glut import glut_geometry

    return glut_geometry(params.chol_diag, params.chol_off, params.opacity_logit)


def _as_u_tensor(u: Tensor | float, ref: Tensor) -> Tensor:
    """``u`` onto ``ref``'s (device, dtype), padded on the RIGHT to ``ref``'s rank.

    Right-padding is the only reading that keeps the leading axes meaning what they
    say: ``(B,)`` and ``(B,1)`` are per-sample strengths, ``(B,P)`` is per-query, and
    all three become ``(B, ..., 1)`` against ``(B, P, 3)``.  Left-aligned broadcasting
    would silently pair a per-sample ``u`` with the query axis whenever ``B == P``.
    """
    if isinstance(u, Tensor):
        ut = u.to(device=ref.device, dtype=ref.dtype)
        while 0 < ut.dim() < ref.dim():
            ut = ut.unsqueeze(-1)
        return ut
    return torch.as_tensor(float(u), device=ref.device, dtype=ref.dtype)


def _field_to_hwc(u: Tensor, x: Tensor) -> Tensor:
    """A spatial field in any of the usual layouts -> ``(B,H,W,1)`` against ``(B,H,W,3)``."""
    b, h, w = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    ut = u.to(device=x.device, dtype=x.dtype)
    if ut.dim() == 2:                      # (H,W)
        ut = ut.reshape(1, h, w, 1)
    elif ut.dim() == 3:                    # (B,H,W) or (1,H,W)
        ut = ut.reshape(-1, h, w, 1)
    elif ut.dim() == 4 and ut.shape[1] == 1:   # (B,1,H,W)
        ut = ut.permute(0, 2, 3, 1)
    if ut.shape[0] not in (1, b):
        raise ValueError(f"field batch {tuple(ut.shape)} does not match image batch {b}")
    return ut


# --------------------------------------------------------------------------- #
# 5. the loss (GLUT Eq.6-8, weights from section 4.1; only the sampling changed)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LossTerms:
    """The three GLUT terms plus the columns the first ``steps.jsonl`` row needs."""

    total: Tensor
    l_rec: Tensor
    l_hc: Tensor
    l_sparse: Tensor
    n_hc_masked: int
    n_hc_points: int
    mean_c_weight: float
    chroma_weight_src: str

    def as_row(self) -> dict[str, Any]:
        d = lambda t: float(t.detach()) if isinstance(t, Tensor) else float(t)
        return {"L_rec": d(self.l_rec), "L_hc": d(self.l_hc),
                "L_sparse": d(self.l_sparse), "L_total": d(self.total),
                "n_hc_masked": int(self.n_hc_masked),
                "n_hc_points": int(self.n_hc_points),
                "mean_C_weight": self.mean_c_weight,
                "chroma_weight_src": self.chroma_weight_src}


def glut_loss(
    y_hat: Tensor,
    y_target: Tensor,
    opacity: Tensor,
    *,
    lambda_hc: float = 10.0,
    lambda_sparse: float = 0.001,
    eps_c: float = 1e-3,
    hc_mask: bool = True,
    l_rec_scale: float = 1.0,
    chroma_ref: Tensor | None = None,
    eps: float = EPS,
) -> LossTerms:
    """``L = s*||y_hat - y||_1 + 10*L_hc + 0.001*R_sparse`` (GLUT Eq.6-8).

    ``L_hc = C (1 - <h_hat, h>)`` with ``h = (a,b)/max(C, eps_c)`` and the frozen hard
    mask ``1[C >= eps_c]`` (the cross-arm block's NOVEL numeric; ``n_hc_masked`` is
    logged every step).  ``chroma_ref`` decouples the **weight** ``C`` from the target
    used for the hue: identity G1' says ``C(y_u)`` is not ``u C(L_l)``, so ablation row
    1'' re-weights with ``C(y_u)`` while the target stays ``L_l(x)``.  ``l_rec_scale``
    is row 1's other half (``x0.5``).

    ``srgb_to_lab`` comes from :mod:`q3vl.whatb.colorimetry` -- the criteria side reads
    the same function, so the training objective and the board cannot drift apart.
    """
    if y_hat.shape != y_target.shape:
        raise ValueError(f"y_hat {tuple(y_hat.shape)} != y {tuple(y_target.shape)}")
    y_t = y_target.to(device=y_hat.device, dtype=y_hat.dtype)
    l_rec = (y_hat - y_t).abs().mean()

    lab_hat, lab_t = srgb_to_lab(y_hat), srgb_to_lab(y_t)
    _, h_hat, _ = chroma_hue(lab_hat, eps_c)
    c_t, h_t, valid = chroma_hue(lab_t, eps_c)
    if chroma_ref is None:
        c_w = c_t
    else:
        c_w, _, _ = chroma_hue(srgb_to_lab(chroma_ref.to(device=y_hat.device,
                                                         dtype=y_hat.dtype)), eps_c)
    term = c_w * (1.0 - (h_hat * h_t).sum(dim=-1))
    n_points = int(valid.numel())
    if hc_mask:
        keep = valid.to(term.dtype)
        n_valid = keep.sum()
        l_hc = (term * keep).sum() / n_valid.clamp_min(1.0)
        n_masked = n_points - int(n_valid)
        mean_c = float((c_w * keep).sum() / n_valid.clamp_min(1.0))
    else:
        l_hc = term.mean()
        n_masked = 0
        mean_c = float(c_w.mean())

    o = opacity.to(device=y_hat.device, dtype=y_hat.dtype)
    ent = o * torch.log(o + eps) + (1.0 - o) * torch.log(1.0 - o + eps)
    l_sparse = -ent.mean(dim=-1).mean()          # -(1/N) sum_i [...], averaged over B

    total = float(l_rec_scale) * l_rec + float(lambda_hc) * l_hc \
        + float(lambda_sparse) * l_sparse
    return LossTerms(total=total, l_rec=l_rec, l_hc=l_hc, l_sparse=l_sparse,
                     n_hc_masked=n_masked, n_hc_points=n_points, mean_c_weight=mean_c,
                     chroma_weight_src=("target" if chroma_ref is None else "y_u"))


def gate_probe_loss(u_pred: Tensor, u_true: Tensor) -> Tensor:
    """Row 3-b's NOVEL term: ``L_gate = |sigmoid(w^T z_{lambda=u} + b) - u|`` (L1).

    No source exists for this one (EPR-027:428 marks it NOVEL and its weight 0.1);
    ``u`` is known at training time so the minimal form is a regression.
    """
    return (u_pred.reshape(-1) - u_true.to(device=u_pred.device,
                                           dtype=u_pred.dtype).reshape(-1)).abs().mean()


# --------------------------------------------------------------------------- #
# 6. hard-sample mining (GLUT App A.1, ruling 11.1-4 / EPR-024 section 3.3)
# --------------------------------------------------------------------------- #
def mine_hard_colors(
    model: "IdGateArm",
    params: GlutParams,
    x: Tensor,
    lut_values: Tensor,
    u: Tensor,
    *,
    ratio: float,
    sampler: _queries.QuerySampler,
    device: Any = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, int]:
    """Within-batch top-r resampling of colour queries; no cross-step state.

    Ruling 11.1-4: a ``no_grad`` forward on the uniformly drawn colours, per-colour
    L1 against ``y_u`` (averaged over this sample's ``K_u`` strengths), the top
    ``r*Q`` of them kept and ``(1-r)*Q`` fresh uniform colours drawn to refill -- so
    the step still carries exactly ``B*Q = 8192`` colours and the step count stays
    matched with the other five arms.  The ``topk`` runs on the device the errors are
    on (CPU and CUDA break ties differently).

    Returns ``(x_mined (B,Q,3), n_hard_colors)``.
    """
    b, q = int(x.shape[0]), int(x.shape[1])
    k_hard = int(round(float(ratio) * q))
    k_hard = max(0, min(k_hard, q))
    if k_hard == 0:
        return x, 0
    with torch.no_grad():
        out = model.apply_transform(x, params, u=_u_for_queries(u, q) if model.cfg.gate
                                    else None)
        y_u = strength_target(x.unsqueeze(1), lut_values.unsqueeze(1),
                              u.unsqueeze(-1).unsqueeze(-1))            # (B,K,Q,3)
        err = (out.y.unsqueeze(1) - y_u).abs().mean(dim=(1, 3))          # (B,Q)
    idx = torch.stack([_queries.select_hard(err[i], ratio)[:k_hard] for i in range(b)])
    hard = torch.gather(x, 1, idx.unsqueeze(-1).expand(b, k_hard, 3))
    fresh = sampler.sample(b, q - k_hard, device=device, dtype=dtype)
    return torch.cat([hard, fresh.to(device=x.device, dtype=x.dtype)], dim=1), b * k_hard


def _u_for_queries(u: Tensor, q: int) -> Tensor:
    """``(B, K)`` strengths -> ``(B, 1, 1)`` mean is NOT used; K is folded by the caller.

    For the mining pre-pass the transform is evaluated once per colour, so the gate
    gets the sample's mean strength; the *error* is still averaged over all K targets.
    Using the mean here only decides which colours are hard, never a loss value.
    """
    return u.mean(dim=1).reshape(-1, 1, 1)


# --------------------------------------------------------------------------- #
# 7. optimiser and schedule (GLUT section 4.1)
# --------------------------------------------------------------------------- #
def build_optimizer(model: IdGateArm, cfg: IdGateConfig | None = None
                    ) -> torch.optim.Optimizer:
    """Adam with the two lr groups.  betas / weight decay are PyTorch defaults and are
    recorded as deviations (GLUT gives neither)."""
    cfg = cfg or model.cfg
    return torch.optim.Adam(model.param_groups(cfg.lr), lr=cfg.lr,
                            betas=tuple(cfg.adam_betas), weight_decay=cfg.weight_decay)


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int
                    ) -> torch.optim.lr_scheduler.LRScheduler:
    """Cosine annealing from 1e-3 over the whole run (GLUT section 4.1, verbatim)."""
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(total_steps))


# --------------------------------------------------------------------------- #
# 8. run-time assertions
# --------------------------------------------------------------------------- #
def gate_identity_check(model: IdGateArm, x: Tensor, params: GlutParams
                        ) -> dict[str, Any]:
    """``u = 1`` must reproduce ``f_theta`` exactly (EPR-027:461).

    This is the first wiring assertion: with ``u = 1`` the gate is ``x + 1*(y-x)`` and
    the output must equal the un-gated transform.  Reported as a criterion column
    (``gate_identity_check``) so "the gate was in the graph" is on the board, not only
    in a log line.
    """
    with torch.no_grad():
        gated = model.apply_transform(x, params, u=1.0)
        # The un-gated reference is the carrier called exactly as row 1 (--no-gate)
        # calls it: clamped inside.  With u = 1 the gated path is clamp(x + 1*(pre-x))
        # = clamp(pre), so the two must agree bit for bit.
        plain = glut_forward(x, params, clamp=model.cfg.clamp,
                             residual=model.cfg.residual)
        diff = (gated.y - plain).abs().max()
        diff_pre = (gated.y_pre_clamp - gated.y_raw).abs().max()
    atol, src = model.cfg.identity_atol(gated.y.dtype)
    return {"n": int(x.reshape(-1, 3).shape[0]), "max_abs": float(diff),
            "max_abs_pre_clamp": float(diff_pre),
            "u": 1.0, "atol": atol, "atol_source": src,
            "passed": bool(float(diff) <= atol),
            "quantity": "max|f_{u=1} - f_theta| (EPR-027:461 writes '== 0'; the "
                        "float-arithmetic floor is recorded in atol/atol_source)"}


class GateNotWired(AssertionError):
    """``u = 1`` did not reproduce ``f_theta``: the gate is not the identity at u=1."""


def assert_gate_identity(model: IdGateArm, x: Tensor, params: GlutParams
                         ) -> dict[str, Any]:
    """:func:`gate_identity_check` as a hard gate.  Off when ``--no-gate``."""
    if not model.cfg.gate:
        return {"n": 0, "skipped": "--no-gate (row 1 = EPR-024, no gate in the graph)"}
    rep = gate_identity_check(model, x, params)
    if not rep["passed"]:
        raise GateNotWired(
            f"max|f_(u=1) - f_theta| = {rep['max_abs']:.3e} > atol "
            f"{rep['atol']:.3e} ({rep['atol_source']}).  With u = 1 the gate is "
            "x + 1*(f-x) and must return f; a residual above the float-arithmetic "
            "floor means the gate is not sitting where the proposal puts it, or u "
            "never reached it.")
    return rep


def first_quick_eval_guard(
    model: IdGateArm,
    z: Tensor,
    x: Tensor,
    *,
    where: str = "quick_eval",
    exit_process: bool = True,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The three degeneracy checks at the FIRST quick eval, plus the gate assertion.

    Measured on ``f_theta`` (``u = 1``, the full transform) rather than on the gated
    output: with a small ``u`` a perfectly healthy transform looks flat, and the point
    of the guard is the head, not the strength.  Any of "flat across query colours" /
    "is the identity" / "identical across samples" leaves via ``SystemExit(2)`` --
    the where side burned 2.6 GPU-hours on a constant field for want of exactly this.
    """
    params = model.theta(z)
    with torch.no_grad():
        y = model.apply_transform(x, params, u=1.0 if model.cfg.gate else None).y
    xb = x if x.dim() == 3 else x.unsqueeze(0).expand(y.shape[0], -1, -1)
    report = assert_transform_not_degenerate(
        y, xb, thresholds=model.cfg.degeneracy, where=where,
        exit_process=exit_process,
        extra={"arm": ARM, "gate": model.cfg.gate, **dict(extra or {})})
    gate_rep = assert_gate_identity(model, x, params)
    return {"degeneracy": report.as_dict(), "gate_identity_check": gate_rep}


# --------------------------------------------------------------------------- #
# 9. one training step
# --------------------------------------------------------------------------- #
def step_columns(cfg: IdGateConfig) -> tuple[str, ...]:
    """The columns the FIRST row of ``steps.jsonl`` promises (frozen block + this arm)."""
    extra: list[str] = ["u_mean", "u0_frac", "n_pairs_per_step", "n_hard_colors",
                        "chroma_weight_src", "mean_C_weight"]
    if cfg.gate_zhead == "linear":
        extra += ["L_gate", "gate_u_mae"]
    return _publish.step_columns_for(cfg.loss_level, extra=extra)


def train_step(
    model: IdGateArm,
    *,
    z: Tensor,
    lut_values_fn,
    lut_ids: Sequence[str],
    x: Tensor,
    u_draw: UDraw,
    epoch: float,
    sampler: _queries.QuerySampler,
    z_null: Tensor | None = None,
    mining_ratio: float | None = None,
) -> tuple[Tensor, dict[str, Any]]:
    """One optimisation step's forward + loss + the ``steps.jsonl`` row.

    ``x`` is ``(B, Q, 3)`` query colours, ``u_draw.u`` is ``(B, K_u)``, and
    ``lut_values_fn(x)`` returns ``L_l(x)`` for the batch's LUTs, ``(B, Q, 3)`` -- the
    caller owns the bank so this function stays pure.  The pair count
    ``B*Q*K_u`` enters the batch, never the step count (NOTES 12).

    Returns ``(total_loss, row)``; the caller does ``backward`` / ``step`` and appends
    ``row`` to ``steps.jsonl``.
    """
    cfg = model.cfg
    b, q = int(x.shape[0]), int(x.shape[1])
    u = u_draw.u.to(device=x.device, dtype=x.dtype)
    params = model.theta(z, z_null, lam=1.0)

    n_hard = 0
    if cfg.mining and mining_ratio:
        lv0 = lut_values_fn(x)
        x, n_hard = mine_hard_colors(model, params, x, lv0, u, ratio=float(mining_ratio),
                                     sampler=sampler, device=x.device, dtype=x.dtype)

    lut_values = lut_values_fn(x)                                   # (B,Q,3)
    if cfg.gate:
        xk = x.unsqueeze(1).expand(b, u.shape[1], q, 3).reshape(b, -1, 3)
        uk = u.unsqueeze(-1).expand(b, u.shape[1], q).reshape(b, -1, 1)
        lvk = lut_values.unsqueeze(1).expand(b, u.shape[1], q, 3).reshape(b, -1, 3)
        out = model.apply_transform(xk, params, u=uk)
        target = strength_target(xk, lvk, uk)
        chroma_ref = target if cfg.chroma_weight_src == "y_u" else None
    else:
        xk = x
        out = model.apply_transform(x, params, u=None)
        target = lut_values.to(device=x.device, dtype=x.dtype)
        # row 1'': no gate, but the L_hc chroma weight is taken from y_u anyway (the
        # u draw is used for the weight only and never reaches the gate).
        chroma_ref = None
        if cfg.chroma_weight_src == "y_u":
            u_w = u.mean(dim=1).reshape(b, 1, 1)
            chroma_ref = strength_target(x, lut_values, u_w)

    # the ladder gates the two optional weights (carrier.py:347/:351); at the
    # default --loss-level 3 both are cfg's own value, bit for bit.
    terms = glut_loss(out.y, target, out.opacity,
                      lambda_hc=cfg.lambda_hc_effective,
                      lambda_sparse=cfg.lambda_sparse_effective, eps_c=cfg.hc_eps_c,
                      hc_mask=cfg.hc_mask, l_rec_scale=cfg.l_rec_scale,
                      chroma_ref=chroma_ref)
    total = terms.total

    # The frozen column vocabulary is {gt_lut, y_u} (EPR-027:352).  With the gate on the
    # loss target *is* y_u, so the chroma weight is C(y_u) whether or not chroma_ref was
    # passed separately; only row 1 (no gate, target = L_l) weights with C(L_l).
    weight_src = "y_u" if (cfg.gate or cfg.chroma_weight_src == "y_u") else "gt_lut"

    row: dict[str, Any] = {
        **terms.as_row(),
        "chroma_weight_src": weight_src,
        "n_colors": b * q,
        "n_luts_in_batch": len(set(map(str, lut_ids))),
        "mining_ratio": float(mining_ratio or 0.0),
        "n_hard_colors": int(n_hard),
        "n_pairs_per_step": b * q * (u.shape[1] if cfg.gate else 1),
        "epoch": float(epoch),
        "n_degenerate_precision": int(out.degenerate_precision.sum()),
        **u_draw.stats,
    }
    if cfg.gate_zhead == "linear":
        # the probe reads z_{lambda = u}: one lambda per (sample, u) draw
        u_flat = u.reshape(-1)
        z_rep = z.repeat_interleave(u.shape[1], dim=0)
        zn_rep = None if z_null is None else z_null.repeat_interleave(u.shape[1], dim=0)
        u_pred = model.predict_u(z_rep, zn_rep, lam=u_flat)
        l_gate = gate_probe_loss(u_pred, u_flat)
        total = total + float(cfg.gate_zhead_weight) * l_gate
        row["L_gate"] = float(l_gate.detach())
        row["gate_u_mae"] = float(l_gate.detach())
        row["L_total"] = float(total.detach())
    return total, row


# --------------------------------------------------------------------------- #
# 10. evaluation: per-sample rows, the arm's own columns, the board
# --------------------------------------------------------------------------- #
@dataclass
class GridSpec:
    """The three query sets a board needs, built once per run on one device."""

    grid: Tensor          # (17^3, 3) X_grid  (section 4.B)
    floor_grid: Tensor    # (9^3, 3)  the grid the pre-registered floors used
    heldout: Tensor       # unseen colours: the odd 8-bit levels (GLUT App A.1)

    @classmethod
    def build(cls, cfg: IdGateConfig, *, device: Any = "cpu",
              dtype: torch.dtype = torch.float32,
              n_heldout: int = 4096, seed: int | None = None) -> "GridSpec":
        sampler = _queries.QuerySampler(seed=cfg.seed if seed is None else seed)
        return cls(grid=_queries.uniform_grid(cfg.grid_n, dtype=dtype, device=device),
                   floor_grid=_queries.uniform_grid(cfg.floor_grid_n, dtype=dtype,
                                                    device=device),
                   heldout=sampler.sample_heldout(1, n_heldout, device=device,
                                                  dtype=dtype)[0])


@dataclass
class LibraryContext:
    """``Lib_tr`` evaluated once, plus everything the B columns need.

    ``mean_volume`` is the library-mean transform ``L_bar`` baked onto a ``b1_grid_n``
    cube so it can be applied to an image with the same trilinear operator the data law
    uses; ``L_bar`` is not a bank LUT, so this is the only way to apply it to pixels
    without evaluating 1137 LUTs per image.  The sampling grid is recorded on the board.
    """

    values: _criteria.LibraryValues
    bucket_pools: Mapping[str, Sequence[str]]
    mean_volume: Tensor | None = None
    grid_n: int = 33

    @classmethod
    def build(cls, bank, lut_ids: Sequence[str], grid: Tensor, *,
              bucket_pools: Mapping[str, Sequence[str]] | None = None,
              mean_grid_n: int | None = None) -> "LibraryContext":
        vals = _criteria.LibraryValues.build(bank, list(lut_ids), grid)
        ctx = cls(values=vals, bucket_pools=dict(bucket_pools or {}))
        if mean_grid_n:
            q = lut_grid_queries(mean_grid_n, device=grid.device, dtype=grid.dtype)
            mean_vals = bank.evaluate_library(list(lut_ids), q).mean(dim=0)
            ctx.mean_volume = volume_from_values(mean_vals, mean_grid_n)
            ctx.grid_n = int(mean_grid_n)
        return ctx


def lut_grid_queries(n: int, *, device: Any = "cpu",
                     dtype: torch.dtype = torch.float32) -> Tensor:
    """``(n^3, 3)`` sRGB queries ordered so ``reshape(n,n,n,3)`` is ``grid[b, g, r]``.

    That is the bank's own storage order (``rendering.py:77-109``), so the values a
    transform takes on this query set can be packed straight into a volume and consumed
    by :func:`q3vl.whatb.lutdata.apply_lut_volume`.
    """
    ax = torch.linspace(0.0, 1.0, int(n), device=device, dtype=dtype)
    b, g, r = torch.meshgrid(ax, ax, ax, indexing="ij")
    return torch.stack((r.reshape(-1), g.reshape(-1), b.reshape(-1)), dim=-1)


def volume_from_values(values: Tensor, n: int) -> Tensor:
    """``(n^3, 3)`` values on :func:`lut_grid_queries` -> ``(1, 3, n, n, n)`` volume."""
    return values.reshape(int(n), int(n), int(n), 3).permute(3, 0, 1, 2)[None].contiguous()


@dataclass
class EvalSample:
    """One evaluation sample.  Tensors are handed in by the runner already on device.

    ``image`` is the sample's input image at short side 512 (``(3,H,W)`` sRGB in
    [0,1]); ``alpha`` is the **GT** field at the same resolution (``(H,W)``; ``style``
    samples pass ones).  ``z`` is the cached ``<seg_color>`` read-out and ``z_ctrl``
    the three negative controls' read-outs (their reasoning re-generated, never
    teacher-forced -- criterion section D).
    """

    sample_id: str
    winner_confidence: str
    task_type: str
    lut_id: str
    image: Tensor
    alpha: Tensor
    z: Tensor
    z_null: Tensor | None = None
    z_ctrl: Mapping[str, Tensor] = field(default_factory=dict)
    alpha_pred: Tensor | None = None
    alpha_shuffle: Tensor | None = None
    minor: str | None = None
    source_image_id: str | None = None
    lam: float = 1.0


#: what each ``--gate-u-source`` means at evaluation time, recorded on every row.
U_SOURCE_RESOLUTION: dict[str, str] = {
    "sample": "gt_alpha (the training-time draw has no evaluation meaning; this is "
              "ladder row 2, and row 2' --gate-u-source lambda must be published "
              "beside it -- EPR-027:649-652)",
    "gt_alpha": "row 2: the GT field drives the gate AND the frozen composition",
    "lambda": "row 2'/3-a: u = clamp(lambda, 0, 1), a per-image scalar",
    "zhead": "row 3-b: u = sigmoid(w^T z_lambda + b)",
    "const": "the sample's mean alpha as a constant field",
    "shuffle": "another sample's field (section 4.3 row 4)",
}


def _u_for_source(model: IdGateArm, sample: EvalSample, source: str,
                  *, field_src: str | None = None) -> Tensor | float | None:
    """Resolve ``u`` for one evaluation row from the pre-registered ladder."""
    cfg = model.cfg
    if not cfg.gate:
        return None
    if source in ("gt_alpha", "sample"):
        return _field_for(sample, field_src or "gt")
    if source == "const":
        return float(sample.alpha.mean())
    if source == "shuffle":
        return _field_for(sample, "shuffle")
    if source == "lambda":
        return effective_lambda_u(sample, cfg)
    if source == "zhead":
        return model.predict_u(sample.z.unsqueeze(0)).reshape(())
    raise ValueError(f"unknown u source {source!r}")


def lambda_sign(sample_id: str, *, seed: int) -> float:
    """+1 / -1 per sample, deterministic and RNG-free (ablation row 9).

    ``learn_delta.py:85`` randomises the sign of the sampled scale at *training* time;
    row 9 carries that shape over to the lambda consumption ``u = clamp(lambda, 0, 1)``,
    where a negative draw simply lands on ``u = 0``.  Derived from a hash of the sample
    id so re-running the evaluation cannot change which samples were flipped.
    """
    h = hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()
    return 1.0 if (h[0] & 1) == 0 else -1.0


def effective_lambda_u(sample: EvalSample, cfg: IdGateConfig) -> float:
    """``u = clamp(lambda, 0, 1)``, with row 9's optional random sign."""
    lam = float(sample.lam)
    if cfg.gate_lambda_sign == "random":
        lam = lam * lambda_sign(sample.sample_id, seed=cfg.seed)
    return min(max(lam, 0.0), 1.0)


def _field_for(sample: EvalSample, src: str) -> Tensor:
    _one_of("gate_field_src", src, FIELD_SRCS)
    if src == "gt":
        return sample.alpha
    if src == "const":
        return torch.full_like(sample.alpha, float(sample.alpha.mean()))
    if src == "pred":
        if sample.alpha_pred is None:
            raise ValueError(f"{sample.sample_id}: --gate-field-src pred needs the "
                             "where arm's m_pix, resampled with q3vl/where/upsample.py "
                             "area_resize to the headline resolution")
        return sample.alpha_pred
    if sample.alpha_shuffle is None:
        raise ValueError(f"{sample.sample_id}: --gate-field-src shuffle needs another "
                         "sample's alpha at the same resolution")
    return sample.alpha_shuffle


def _headline_error(model: IdGateArm, sample: EvalSample, params: GlutParams,
                    u: Tensor | float | None, i_star: Tensor,
                    alpha_compose: Tensor) -> float:
    """``mean dE00(I_hat, I*)`` with the frozen formation ``I_hat=(1-a)I+a f_hat(I)``.

    The formation is frozen for every arm and every row, including row 4 where the
    gate has already consumed a field: that double use of ``alpha`` is real, it is
    called out in EPR-027:649-652, and the answer there is to publish the
    ``--gate-u-source lambda`` row beside it -- not to change the formation.
    """
    f_img = model.apply_image(sample.image, params, u=u).y
    i_hat = _criteria.compose_hat(sample.image, alpha_compose.unsqueeze(0), f_img)
    return float(_criteria.image_delta_e00(i_hat, i_star))


def _image_error_of_transform(sample: EvalSample, f_img: Tensor, i_star: Tensor,
                              alpha: Tensor) -> float:
    i_hat = _criteria.compose_hat(sample.image, alpha.unsqueeze(0), f_img)
    return float(_criteria.image_delta_e00(i_hat, i_star))


@torch.no_grad()
def evaluate_sample(
    model: IdGateArm,
    sample: EvalSample,
    *,
    bank,
    grids: GridSpec,
    library: LibraryContext | None = None,
    rng: np.random.Generator | None = None,
    n_repeats: int | None = None,
    with_locality: bool = True,
    with_fields: bool = True,
) -> dict[str, Any]:
    """One per-sample row of the board (criteria section B / C / D / E).

    Everything is computed on the sample's device; only the final reductions become
    Python floats.  The row keys are exactly the ones
    :func:`q3vl.whatb.criteria.build_board` reads.
    """
    cfg = model.cfg
    rng = rng or np.random.default_rng(cfg.seed)
    n_rep = int(cfg.n_repeats if n_repeats is None else n_repeats)
    dev, dt = sample.image.device, sample.image.dtype
    alpha = sample.alpha.to(device=dev, dtype=dt)
    i_star = bank.f_star_image(sample.image, alpha.unsqueeze(0), sample.lut_id)

    params = model.theta(sample.z.unsqueeze(0), None if sample.z_null is None
                         else sample.z_null.unsqueeze(0), lam=sample.lam)
    u_main = _u_for_source(model, sample, cfg.gate_u_source,
                           field_src=cfg.gate_field_src)
    # The function-value columns live on colour queries, where a *field* has no
    # pixel to attach to; a field collapses to its mean strength there and the
    # scalar used is written into the row so the two families of columns are not
    # silently on different u.
    u_scalar = (float(u_main.mean()) if isinstance(u_main, Tensor) else u_main)

    row: dict[str, Any] = {
        "sample_id": sample.sample_id,
        "winner_confidence": sample.winner_confidence,
        "task_type": sample.task_type,
        "lut_id": sample.lut_id,
        "u_source": cfg.gate_u_source,
        "u_source_resolved": U_SOURCE_RESOLUTION[cfg.gate_u_source],
        "field_src": cfg.gate_field_src,
        "u_scalar": u_scalar,
        # EPR-027:459-6 calls the per-sample column `gate_u`; it is the same number
        "gate_u": u_scalar,
    }

    f_img = model.apply_image(sample.image, params, u=u_main).y
    i_hat = _criteria.compose_hat(sample.image, alpha.unsqueeze(0), f_img)
    row["E_arm"] = float(_criteria.image_delta_e00(i_hat, i_star))

    # ---- function-value columns (section 4.B) ----
    f_grid = model.apply_transform(grids.grid, params, u=u_scalar).y[0]
    l_grid = bank.apply(grids.grid, sample.lut_id)
    row["grid_error"] = float(_criteria.function_distance(f_grid, l_grid))
    colors, weights = _queries.image_histogram_colors(sample.image, bits=cfg.hist_bits,
                                                      top_k=cfg.hist_top_k)
    f_hist = model.apply_transform(colors, params, u=u_scalar).y[0]
    row["img_error"] = float(_criteria.function_distance(
        f_hist, bank.apply(colors, sample.lut_id), weights))
    f_unseen = model.apply_transform(grids.heldout, params, u=u_scalar).y[0]
    row["unseen_color_error"] = float(_criteria.function_distance(
        f_unseen, bank.apply(grids.heldout, sample.lut_id)))

    # ---- trivial baselines (section 4.C), all paired on this sample ----
    row["E_B0_identity"] = float(_criteria.image_delta_e00(
        _criteria.compose_hat(sample.image, alpha.unsqueeze(0), sample.image), i_star))
    if library is not None:
        if library.mean_volume is not None:
            f_mean = apply_lut_volume(library.mean_volume.to(device=dev),
                                      sample.image.permute(1, 2, 0)).permute(2, 0, 1)
            row["E_B1_libmean"] = _image_error_of_transform(sample, f_mean, i_star, alpha)
        ids = list(library.values.lut_ids)
        draws = [ids[int(rng.integers(0, len(ids)))] for _ in range(n_rep)]
        row["E_B2_librandom_repeats"] = [
            _image_error_of_transform(sample, bank.apply_image(sample.image, lid),
                                      i_star, alpha) for lid in draws]
        row["E_B2_librandom"] = float(np.mean(row["E_B2_librandom_repeats"]))
        pool = list(library.bucket_pools.get(str(sample.minor), ())) if sample.minor else []
        if pool:
            bdraw = [pool[int(rng.integers(0, len(pool)))] for _ in range(n_rep)]
            row["E_B3_bucket_retrieval_repeats"] = [
                _image_error_of_transform(sample, bank.apply_image(sample.image, lid),
                                          i_star, alpha) for lid in bdraw]
            row["E_B3_bucket_retrieval"] = float(
                np.mean(row["E_B3_bucket_retrieval_repeats"]))
        target = bank.apply(library.values.x, sample.lut_id)
        d = library.values.distance_to(target, metric="de76")
        j = int(torch.argmin(d))
        row["E_B4_oracle"] = _image_error_of_transform(
            sample, bank.apply_image(sample.image, library.values.lut_ids[j]),
            i_star, alpha)
        row["B4_oracle_lut_id"] = library.values.lut_ids[j]
        if sample.lut_id in library.values.index:
            d2 = d.clone()
            d2[library.values.index[sample.lut_id]] = torch.inf
            j2 = int(torch.argmin(d2))
            row["E_B6_libfill"] = _image_error_of_transform(
                sample, bank.apply_image(sample.image, library.values.lut_ids[j2]),
                i_star, alpha)

    # ---- negative controls (section 4.D) ----
    for name, key in (("N1_shuffle", "N1_shuffle"), ("N2_irrelevant", "N2_irrelevant"),
                      ("N3_const", "N3_const")):
        z_c = sample.z_ctrl.get(key)
        if z_c is None:
            continue
        p_c = model.theta(z_c.unsqueeze(0).to(device=dev, dtype=dt))
        f_c = model.apply_image(sample.image, p_c, u=u_main).y
        row[f"E_{name}"] = _image_error_of_transform(sample, f_c, i_star, alpha)
        f_cg = model.apply_transform(grids.grid, p_c, u=u_scalar).y[0]
        row[f"M_{name}"] = float(_criteria.function_distance(f_grid, f_cg))

    # ---- locality (section 4.E) ----
    if with_locality:
        row.update(_criteria.locality_errors(i_hat, i_star, sample.image, alpha))
    if with_fields and cfg.gate:
        for src, key in (("gt", "field_gt"), ("const", "field_const"),
                         ("shuffle", "field_shuffle")):
            try:
                u_f = _field_for(sample, src)
            except ValueError:
                continue
            row[key] = _headline_error(model, sample, params, u_f, i_star, alpha)
        if sample.alpha_pred is not None:
            row["field_pred"] = _headline_error(model, sample, params,
                                                _field_for(sample, "pred"), i_star, alpha)
    if sample.alpha_pred is not None:
        a_pred = sample.alpha_pred.to(device=dev, dtype=dt)
        row["headline_predalpha"] = float(_criteria.image_delta_e00(
            _criteria.compose_hat(sample.image, a_pred.unsqueeze(0), f_img), i_star))
    return row


# --------------------------------------------------------------------------- #
# 11. the arm's own columns (section 4.2 / section G, and section F for P1)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def strength_columns(
    model: IdGateArm,
    samples: Sequence[EvalSample],
    *,
    bank,
    grid: Tensor,
    library: LibraryContext | None = None,
    u_points: Sequence[float] = STRENGTH_U,
    extrap: Sequence[float] = EXTRAP_U,
    dlib_u: Sequence[float] = DLIB_U,
    seed: int = 20260810,
) -> dict[str, dict[str, Any]]:
    """Criterion section G on the function-value grid.

    (a) ``dE00(f_hat_u, y_u)`` at ``u in {0,.25,.5,.75,1}``;
    (b) the magnitude monotonicity rate and (c) Spearman -- **against lambda**, with
        the "against u = 1.000 by construction" row (identity G2) printed beside them
        and the 0.5 / permutation floors attached;
    (d) the out-of-gamut rate and ``d_lib`` at the extrapolated ``u``.
    """
    dev, dt = grid.device, grid.dtype
    per_u: dict[float, list[float]] = {float(u): [] for u in u_points}
    oob: dict[float, list[float]] = {float(u): [] for u in extrap}
    dlib: dict[float, list[float]] = {float(u): [] for u in dlib_u}
    mono_lam: list[float] = []
    spear_lam: list[float] = []
    spear_floor: list[float] = []
    lam_points = tuple(float(u) for u in u_points)

    for s in samples:
        params = model.theta(s.z.unsqueeze(0).to(device=dev, dtype=dt),
                             None if s.z_null is None else
                             s.z_null.unsqueeze(0).to(device=dev, dtype=dt))
        lut_vals = bank.apply(grid, s.lut_id)
        for u in u_points:
            out = model.apply_transform(grid, params, u=float(u) if model.cfg.gate else None)
            y_u = strength_target(grid, lut_vals, float(u))
            per_u[float(u)].append(float(_criteria.function_distance(out.y[0], y_u)))
        for u in extrap:
            out = model.apply_transform(grid, params,
                                        u=float(u) if model.cfg.gate else None)
            oob[float(u)].append(float(out.oob_mask().to(dt).mean()))
        if library is not None:
            # d_lib is measured on the LIBRARY's own query set, not on X_grid:
            # LibraryValues holds Lib_tr evaluated once on `values.x` (the arm builds
            # it on `GridSpec.floor_grid`, 9^3 -- the grid the pre-registered floors
            # were measured with, criteria.py:303-310), while `grid` here is X_grid
            # (17^3).  Feeding a 17^3 transform to `distance_to` raised
            # "size of tensor a (729) must match ... b (4913)" in colorimetry.py:193
            # and took the board down before it could be written; the unit tests miss
            # it because tiny_cfg sets grid_n == floor_grid_n == 4.  The B4 site above
            # already uses `library.values.x` -- this is the same idiom, and EPR-026
            # states the rule outright (interpc.py:1408 refuses a library whose X is
            # not the path's X: "d_lib is min_l D_X(f_alpha, L_l) on the SAME X").
            # The other way to make the shapes agree -- rebuild Lib_tr on X_grid --
            # would put d_lib on a different query set from B0..B4/B6, so it is not
            # taken here; if EPR-027 wants d_lib on 17^3 that is a ruling, not a fix.
            lib_x = library.values.x.to(device=dev, dtype=dt)
            for u in dlib_u:
                out = model.apply_transform(lib_x, params,
                                            u=float(u) if model.cfg.gate else None)
                d = library.values.distance_to(out.y[0], metric="de00")
                dlib[float(u)].append(float(d.min()))
        # (b)/(c) against lambda: u = clamp(lambda, 0, 1) is the row 3-a consumption
        mags = []
        for lam in lam_points:
            p_l = model.theta(s.z.unsqueeze(0).to(device=dev, dtype=dt),
                              None if s.z_null is None else
                              s.z_null.unsqueeze(0).to(device=dev, dtype=dt), lam=lam)
            u_l = min(max(lam, 0.0), 1.0) if model.cfg.gate else None
            y = model.apply_transform(grid, p_l, u=u_l).y[0]
            mags.append(float((y - grid).norm(dim=-1).mean()))
        mono_lam.append(_mono_rate(mags))
        rho, floor = _spearman_with_floor(mags, list(lam_points), seed=seed)
        spear_lam.append(rho)
        spear_floor.append(floor)

    cols: dict[str, dict[str, Any]] = {
        "strength_dE_u": {
            **_criteria.describe([v for vals in per_u.values() for v in vals]),
            "per_u": {str(k): _criteria.describe(v) for k, v in per_u.items()},
            "quantity": "dE00(f_hat_u, y_u) on X_grid, u in " + str(tuple(u_points)),
        },
        "strength_mono_rate_lambda": {
            **_criteria.describe(mono_lam), "random_floor": 0.5,
            "mono_rate_vs_u": 1.000,
            "note": "vs u the rate is 1.000 BY CONSTRUCTION (identity G2); the "
                    "reported rate is against lambda",
        },
        "strength_spearman_lambda": {
            **_criteria.describe(spear_lam),
            "permutation_floor": _criteria.describe(spear_floor),
            "spearman_vs_u": 1.000,
            "note": "vs u the correlation is 1.000 by construction (identity G2)",
        },
        "oob_rate_u": {
            **_criteria.describe([v for vals in oob.values() for v in vals]),
            "per_u": {str(k): _criteria.describe(v) for k, v in oob.items()},
            "degenerate_when": "--gate-clamp before with u in [0,1] (identity G5)",
        },
    }
    if library is not None:
        cols["dlib_u"] = {
            **_criteria.describe([v for vals in dlib.values() for v in vals]),
            "per_u": {str(k): _criteria.describe(v) for k, v in dlib.items()},
            "quantity": "min_l D_Xgrid(f_hat_u, L_l) [dE00]",
        }
    return cols


@torch.no_grad()
def degeneracy_columns(model: IdGateArm, samples: Sequence[EvalSample], *,
                       grid: Tensor, tau: float = 1e-3) -> dict[str, dict[str, Any]]:
    """``Pr_x[sum_j p_j o_j < tau]`` and the demo's precision fallback rate.

    HANDOFF section 2.3 makes the degenerate-weight rate a **column every board must
    carry**: it is proposition 2's ``delta(x) = eps / (sum_j p_j o_j + eps)``, the part
    of the colour cube no Gaussian covers.  It is 0.000 under the App A.1 grid
    initialisation and non-zero only when the geometry has drifted, so a non-zero value
    is a statement about the head, not noise.  ``degenerate_precision_rate`` counts the
    demo ``:496`` fallback (``|det Sigma| < eps`` -> identity precision), which also
    cuts the gradient to that Gaussian's shape parameters.
    """
    rates: list[float] = []
    prec: list[float] = []
    for s in samples:
        params = model.theta(s.z.unsqueeze(0).to(device=grid.device, dtype=grid.dtype))
        out = model.apply_transform(grid, params, u=1.0 if model.cfg.gate else None,
                                    need_influence=True)
        if out.influence_sum is not None:
            rates.append(float((out.influence_sum < tau).to(grid.dtype).mean()))
        prec.append(float(out.degenerate_precision.to(grid.dtype).mean()))
    return {
        "degenerate_weight_rate": {**_criteria.describe(rates), "tau": float(tau),
                                   "quantity": "Pr_x[sum_j p_j o_j < tau] on X_grid "
                                               "(proposition 2's delta)"},
        "degenerate_precision_rate": {**_criteria.describe(prec),
                                      "quantity": "fraction of Gaussians hitting the "
                                                  "demo :496 |det| < eps fallback"},
    }


def _mono_rate(values: Sequence[float]) -> float:
    """``Pr[m_{k+1} > m_k]`` -- the section G(b) rate, floor 0.5."""
    if len(values) < 2:
        return float("nan")
    steps = [1.0 if values[i + 1] > values[i] else 0.0 for i in range(len(values) - 1)]
    return float(np.mean(steps))


def _spearman_with_floor(values: Sequence[float], drivers: Sequence[float], *,
                         seed: int = 20260810, n_perm: int = 200) -> tuple[float, float]:
    """Spearman rho plus its random-permutation floor (mean |rho| over permutations)."""
    v = np.asarray(values, dtype=np.float64)
    d = np.asarray(drivers, dtype=np.float64)
    if v.size < 2 or np.allclose(v, v[0]):
        return 0.0, 0.0
    rank = lambda a: np.argsort(np.argsort(a)).astype(np.float64)
    corr = lambda a, b: float(np.corrcoef(rank(a), rank(b))[0, 1])
    rho = corr(v, d)
    rng = np.random.default_rng(seed)
    floor = float(np.mean([abs(corr(rng.permutation(v), d)) for _ in range(n_perm)]))
    return rho, floor


@torch.no_grad()
def interpolation_columns(
    model: IdGateArm,
    pairs: Sequence[tuple[EvalSample, EvalSample]],
    *,
    bank,
    grid: Tensor,
    alphas: Sequence[float] = INTERP_ALPHAS,
    k_path: int = 20,
) -> dict[str, dict[str, Any]]:
    """Criterion section F: IP-A (``interp_grid`` + the trivial output-mix column) and
    IP-B (the six path quantities).

    The condition path is CGLUT's own mixing convention -- ``(1-a) u_a + a u_b`` at the
    generator's input (post ``pi``) -- and the GT is the function-space blend
    ``(1-a) L_a + a L_b``.  The **output-mix** column ``(1-a) f_a + a f_b`` is reported
    beside it because it beats conditional interpolation by construction and a section F
    claim without it is not interpretable (section 4.F failure modes).
    """
    dev, dt = grid.device, grid.dtype
    blend_err: list[float] = []
    outmix_err: list[float] = []
    endpoint_err: list[float] = []
    path_stats: list[dict[str, Any]] = []
    oob_pre: list[float] = []

    for sa, sb in pairs:
        ua = model.pi(sa.z.unsqueeze(0).to(device=dev, dtype=dt))
        ub = model.pi(sb.z.unsqueeze(0).to(device=dev, dtype=dt))
        la = bank.apply(grid, sa.lut_id)
        lb = bank.apply(grid, sb.lut_id)
        fa = model.apply_transform(grid, model.generator(ua),
                                   u=1.0 if model.cfg.gate else None).y[0]
        fb = model.apply_transform(grid, model.generator(ub),
                                   u=1.0 if model.cfg.gate else None).y[0]
        endpoint_err.append(0.5 * (float(_criteria.function_distance(fa, la))
                                   + float(_criteria.function_distance(fb, lb))))
        for a in alphas:
            f_cond = model.apply_transform(grid, model.generator((1 - a) * ua + a * ub),
                                           u=1.0 if model.cfg.gate else None).y[0]
            gt = (1 - a) * la + a * lb
            blend_err.append(float(_criteria.function_distance(f_cond, gt)))
            outmix_err.append(float(_criteria.function_distance((1 - a) * fa + a * fb, gt)))
        path, pre = [], []
        for k in range(int(k_path) + 1):
            a = k / float(k_path)
            out = model.apply_transform(grid, model.generator((1 - a) * ua + a * ub),
                                        u=1.0 if model.cfg.gate else None)
            path.append(out.y[0])
            pre.append(float(out.oob_mask().to(dt).mean()))
        path_stats.append(_criteria.path_quantities(torch.stack(path)))
        oob_pre.append(float(np.mean(pre)))

    take = lambda key: [p[key] for p in path_stats if p.get(key) is not None]
    return {
        "interp_grid": {**_criteria.describe(blend_err),
                        "output_mix_trivial": _criteria.describe(outmix_err),
                        "endpoint": _criteria.describe(endpoint_err),
                        "alphas": list(alphas),
                        "external_reference_glut_app_b3_psnr": {
                            "CGLUT-32L Full": [48.67, 35.44, 31.16, 31.33, 34.64, 47.95],
                            "CGLUT-32L Shared Geo.": [47.36, 38.46, 34.67, 34.47,
                                                      37.60, 46.18]},
                        "quantity": "dE00(f^cond_alpha, (1-a)L_a + a L_b) on X_grid"},
        "path_len": {**_criteria.describe(take("path_len")),
                     "chord": _criteria.describe(take("chord")),
                     "rho": _criteria.describe(take("rho")),
                     "sigma_bar": _criteria.describe(take("sigma_bar")),
                     "jump_max": _criteria.describe(take("jump_max")),
                     "k_steps": int(k_path),
                     "note": "jump_max is K*max_k delta_k, no percentile trimming"},
        "mono_rate": {**_criteria.describe(take("mono_rate")),
                      "random_floor": 0.5},
        "oob_rate": {**_criteria.describe(oob_pre),
                     "quantity": "Pr_x[f_alpha(x) outside [0,1]^3] BEFORE the final "
                                 "clamp (criterion section 2.4)"},
    }


def extra_criteria_columns(
    *,
    gate_identity: Mapping[str, Any],
    u_histogram: Mapping[str, Any],
    strength: Mapping[str, Mapping[str, Any]],
    interpolation: Mapping[str, Mapping[str, Any]] | None = None,
    degeneracy: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Assemble the columns :func:`q3vl.whatb.criteria.build_board` cannot own.

    Every entry carries an ``n``: ``assert_criteria_ran`` checks ``n > 0`` and a column
    without one can never be asserted (that is the "defined but not wired" failure).
    """
    cols: dict[str, dict[str, Any]] = {
        "gate_identity_check": dict(gate_identity),
        "gate_u_hist": dict(u_histogram),
    }
    cols.update({k: dict(v) for k, v in strength.items()})
    if interpolation:
        cols.update({k: dict(v) for k, v in interpolation.items()})
    if degeneracy:
        cols.update({k: dict(v) for k, v in degeneracy.items()})
    missing = [k for k, v in cols.items() if "n" not in v]
    if missing:
        raise ValueError(f"columns without an 'n': {missing}")
    return cols


def u_histogram(u_values: Sequence[float] | Tensor, *, bins: int = 10) -> dict[str, Any]:
    """``gate_u_hist``: what the training loop actually fed the gate."""
    arr = (u_values.detach().reshape(-1).to("cpu").numpy() if isinstance(u_values, Tensor)
           else np.asarray(list(u_values), dtype=np.float64))
    if arr.size == 0:
        return {"n": 0, "bins": [], "counts": [], "mean": None}
    counts, edges = np.histogram(arr, bins=int(bins), range=(min(0.0, float(arr.min())),
                                                            max(1.0, float(arr.max()))))
    return {"n": int(arr.size), "bins": [float(e) for e in edges],
            "counts": [int(c) for c in counts], "mean": float(arr.mean()),
            "frac_zero": float((arr == 0.0).mean()), "frac_one": float((arr == 1.0).mean()),
            "quantity": "u values consumed by the gate during training"}


# --------------------------------------------------------------------------- #
# 12. board + publication
# --------------------------------------------------------------------------- #
def build_arm_board(rows: Sequence[Mapping[str, Any]], *, split: str,
                    extra_columns: Mapping[str, Mapping[str, Any]],
                    cfg: IdGateConfig, quick: bool = False,
                    published: bool | None = None) -> dict[str, Any]:
    """``criteria.build_board`` plus the identity footnotes EPR-027:643-652 requires.

    The footnotes are printed with the numbers, never instead of them: three of this
    arm's columns are 1.000 / 0 by arithmetic and a reader who does not see that beside
    the value will read a construction as a result.
    """
    board = _criteria.build_board(rows, arm=EPR, split=split,
                                  extra_columns=extra_columns, seed=cfg.seed)
    board["arm_name"] = ARM
    board["quick"] = bool(quick)
    if published is not None:
        board["published"] = bool(published)
    board["identity_footnotes"] = {
        "G2": "section G(b)(c) against u is 1.000 BY CONSTRUCTION; the reported "
              "monotonicity/Spearman are against lambda",
        "G5": "the out-of-gamut column is 0 by construction under --gate-clamp before "
              f"with u in [0,1] (this run: --gate-clamp {cfg.gate_clamp})",
        "G4": "E_out is 0 by construction wherever u(p) = alpha(p) and alpha = 0",
        "G1": "L_rec(u) = u L_rec(1); rows 1' and 1'' are the balancing controls",
        "headline_alpha_double_use":
            "with --gate-u-source gt_alpha the same alpha is used inside the gate and "
            "in the frozen headline formation; the --gate-u-source lambda row must be "
            "published beside it (EPR-027:649-652)",
    }
    board["config"] = cfg.to_dict()
    return board


def publish_arm_board(board: Mapping[str, Any], *, cfg: IdGateConfig,
                      steps_row: Mapping[str, Any] | None = None,
                      steps_path: Any = None, eval_only: bool = False,
                      waived: Sequence[str] = ()) -> dict[str, Any]:
    """The single publication gate: training ran, criteria ran, headline exists.

    Fetches its own first-step row three tiers deep, so a caller that does not pass one
    is not mistaken for a loss that never ran (the SEGSAM / PRND failure).

    ``waived`` drops named keys from the required table.  It is empty for every gated
    row -- the main arm's required table is :data:`REQUIRED_CRITERIA` verbatim -- and
    the caller (``scripts/run_idgate_arm.py``) is the one that decides which keys a 档
    structurally cannot compute, records them on the board as ``criteria_waived``, and
    is answerable for each name.
    """
    extra = [c for c in step_columns(cfg) if c not in _publish.FIRST_STEP_COLUMNS]
    return _publish.assert_publishable(
        board, EPR, steps_row=steps_row, steps_path=steps_path, eval_only=eval_only,
        loss_level=cfg.loss_level, extra_step_columns=extra, axes=AXES,
        required=[k for k in REQUIRED_CRITERIA if k not in set(waived)])


# --------------------------------------------------------------------------- #
# 13. the pre-registration record
# --------------------------------------------------------------------------- #
def loss_preregistration(cfg: IdGateConfig) -> dict[str, Any]:
    """``config/loss_preregistration.json`` -- what this run promises to optimise.

    Includes the terms explicitly **not** added, because "we did not add it" is only
    checkable if it was written down (EPR-027:431-434).
    """
    terms = [
        {"name": "L_rec", "form": "||y_hat - y_u||_1", "weight": cfg.l_rec_scale,
         "source": "GLUT Eq.6", "note": "y_u = (1-u)x + u L_l(x); identity G1 makes "
                                        "E_u[L_rec(u)] = 0.5 L_rec(1)"},
        {"name": "L_hc", "form": "C (1 - <h_hat, h>)",
         "weight": cfg.lambda_hc_effective,
         "source": "GLUT Eq.7 (weight = target chroma), section 4.1 lambda_hc = 10",
         "chroma_weight_src": cfg.chroma_weight_src,
         "c_to_zero": f"h = (a,b)/max(C, {cfg.hc_eps_c}) and a hard mask "
                      f"1[C >= {cfg.hc_eps_c}] (frozen block, NOVEL numeric); "
                      "n_hc_masked is logged every step"},
        {"name": "R_sparse",
         "form": "-(1/N) sum_i [o log(o+eps) + (1-o) log(1-o+eps)]",
         "weight": cfg.lambda_sparse_effective,
         "source": "GLUT Eq.8, eps = 1e-6"},
    ]
    if cfg.gate_zhead == "linear":
        terms.append({"name": "L_gate", "form": "|sigmoid(w^T z_{lambda=u} + b) - u|",
                      "weight": cfg.gate_zhead_weight,
                      "source": "NOVEL (EPR-027:411-416); no original to copy"})
    return {
        "arm": ARM, "epr": EPR,
        "total": "L_rec + 10 L_hc + 0.001 R_sparse"
                 + (" + w_gate L_gate" if cfg.gate_zhead == "linear" else ""),
        "loss_ladder": pure_l1_record(
            loss_level=cfg.loss_level, lambda_hc=cfg.lambda_hc_effective,
            lambda_sparse=cfg.lambda_sparse_effective),
        "terms": terms,
        "not_added": [
            {"name": "L_interval", "source": "CLIPtone section 5 Eq.5 "
                                             "(lambda_interval = 0.5, alpha = 0.7)",
             "reason": "regularises AdaInt sampling coordinates; this arm has no AdaInt"},
            {"name": "4D TV / monotonicity", "source": "4D LUT",
             "reason": "u is not a fourth LUT index axis; theta has no context axis"},
            {"name": "L_img", "source": "--loss-level 4",
             "reason": "not pre-registered for this arm"},
        ],
        "optimizer": {"name": "Adam", "lr": cfg.lr,
                      "betas": list(cfg.adam_betas),
                      "weight_decay": cfg.weight_decay,
                      "grad_clip": cfg.grad_clip,
                      "schedule": "cosine annealing from 1e-3 over the whole run",
                      "groups": {"generator": cfg.lr,
                                 "condition_side (pi + u probe)": cfg.lr * cfg.pi_lr_scale},
                      "deviations": ["betas / weight decay / clipping are PyTorch "
                                     "defaults; GLUT section 4.1 gives none"]},
        "u_sampling": {"dist": cfg.gate_u_dist, "p_end": cfg.gate_p_end,
                       "k_u": cfg.gate_ku, "novel": ["p_end"],
                       "reference": "Baumann Algorithm 1 samples "
                                    "lambda ~ U([-5,5] \\ (-0.1,0.1)) and four values "
                                    "per step (learn_delta.yaml scale_batch_size: 4)"},
        "mining": {"enabled": cfg.mining, "schedule": "epoch 5->20, ratio 0.10->0.40",
                   "granularity": "within-batch top-r colour resampling, no cross-step "
                                  "state (ruling 11.1-4)"},
        "checkpoint_selection": ".contexts.all.headline_normal_only (never val loss)",
        "banned": ["AUC in any form", "pooled (low-mixed) headline",
                   "per-image min-max / softmax", "percentile-trimmed means",
                   "IoU as an objective", "cross-step-count comparison"],
    }


def record_first_step(row: Mapping[str, Any]) -> None:
    """Tier 3 of the first-step contract: the in-process witness."""
    record_step_witness(row)
