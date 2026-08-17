"""EPR-026 INTERPC -- interpolation-consistency supervision on top of EPR-024.

The arm's **only** change to the EPR-024 baseline is one extra term in the total
loss (``EPR-026:5-8``)::

    u_a = pi(z_a) ,  u_b = pi(z_b) ,  u_alpha = (1 - alpha) u_a + alpha u_b
    L_interp = E_{(a,b), alpha, x} || f_{G(u_alpha)}(x)
                                     - ((1 - alpha) L_a(x) + alpha L_b(x)) ||_1
    L        = L_GLUT + lambda_int * L_interp ,   alpha ~ Beta(beta, beta)

with ``(a, b)`` two training samples that share a ``source_image_id`` and carry
different ``lut_id``.  ``lambda_int = 0`` is EPR-024 bit-for-bit: the pair stream
is not even constructed, so the fit stream's sample distribution is untouched
(``EPR-026:429-433``).

Everything else -- the generator, the read-out, the supervision space, the
optimiser, the step count, the data pipeline -- is EPR-024's, taken through the
shared modules of ``q3vl/whatb/`` rather than re-implemented here.  Nothing in
this file reads or imports ``q3vl/what/``, ``model/glut_repro/`` or ``gpu_render/``.

Frozen coordinates this module carries (cross-arm frozen block, six proposals
byte-identical; ``docs/HANDOFF_whatb_2026-08-15.md`` section 3)::

    train normal-only n = 93934      B = 32 samples x Q = 256 colours = 8192/step
    2936 steps/epoch                 40 epochs = 117,440 steps
    clamp = "two" (demo double clamp)
    headline image formation  I_hat = (1-a) I + a f_hat(I)
    twelve pre-registered criterion keys; P1 adds
        {interp_grid, path_len, mono_rate, oob_rate}
    colour span encoded by q3vl.whatb.colorspan + the tokenizer assertion

Three failures the where side paid for last week, closed structurally here
(task-card "硬要求"):

1. **assertion contract** -- :func:`assert_interp_step_columns` fetches its own
   first-step row three tiers deep (caller -> ``steps.jsonl`` -> in-process
   witness) via :mod:`q3vl.whatb.guards`; "nobody handed me a row"
   (``StepsRowUnavailable``) and "the row has no loss columns"
   (``LossColumnsMissing``) are different exceptions, and the
   ``lambda_int == 0`` baseline row additionally has to be *clean*
   (:class:`BaselineRowNotClean`).
2. **device / dtype** -- every tensor entering a forward is moved with
   ``.to(device=ref.device, dtype=ref.dtype)``; the only constants
   (the 17^3 evaluation grid, the critic's anchor grid) are
   ``register_buffer(..., persistent=False)``.  There is no bare
   ``torch.tensor(...)`` inside any ``forward``.
3. **constant-field / degenerate solutions** -- :func:`quick_eval_guard` runs
   ``q3vl.whatb.degeneracy.assert_transform_not_degenerate`` at the FIRST quick
   eval and exits the process on a flat / identity / sample-invariant
   transform; :func:`assert_publishable_interpc` refuses to publish a board if
   that guard never ran.

and (4) every criterion is computed on the tensor's own device -- no ``.cpu()``
in any metric path (CPU and CUDA break ``topk`` ties differently; the where side
moved an IoU by 0.296 that way).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from q3vl.whatb.caliber import (
    DATA_CHOICES,
    FROZEN_BATCH_SPLITS,
    effective_lambda_hc,
    effective_lambda_sparse,
)
from q3vl.whatb.colorimetry import chroma_hue, srgb_to_lab
from q3vl.whatb.criteria import (
    LibraryValues,
    describe,
    function_distance,
    path_quantities,
)
from q3vl.whatb.degeneracy import (
    DegeneracyCheckNotRun as _SharedDegeneracyCheckNotRun,
    DegeneracyThresholds,
    assert_transform_not_degenerate,
    clear_degeneracy_check as _clear_degeneracy_check,
    degeneracy_check_ran as _shared_degeneracy_check_ran,
)
from q3vl.whatb.generator import CGLUTGenerator, SegColorProjection
from q3vl.whatb.glut import EPS, GlutAux, GlutCarrier, GlutParams
from q3vl.whatb.guards import (
    LossColumnsMissing,       # noqa: F401 -- re-exported for the runner's except clause
    StepsRowUnavailable,      # noqa: F401 -- idem; the two are deliberately distinct
    assert_first_step_columns,
    record_step_witness,
)
from q3vl.whatb.publish import assert_publishable, step_columns_for
from q3vl.whatb.queries import (
    BATCH_SAMPLES,
    COLORS_PER_STEP,
    QUERIES_PER_SAMPLE,
    mining_ratio,
    uniform_grid,
)
from q3vl.whatb.readout import WHATB_READOUT_KINDS

__all__ = [
    "ARM",
    "AXES",
    "INTERP_ALPHA_MODES",
    "INTERP_WHERE_CHOICES",
    "INTERP_DIST_CHOICES",
    "INTERP_TARGET_CHOICES",
    "INTERP_RAMP_CHOICES",
    "INTERP_STAGE_CHOICES",
    "ALTERNATE_CHOICES",
    "IPA_ALPHA_GRID",
    "IPB_K",
    "InterpcConfig",
    "PairIndex",
    "AlphaSampler",
    "InterpcArm",
    "EmaTeacher",
    "AcaiCritic",
    "FitBatch",
    "PairBatch",
    "glut_loss_terms",
    "interp_distance",
    "interp_weight_at",
    "mine_step",
    "train_step",
    "build_optimizer",
    "build_scheduler",
    "quick_eval_guard",
    "degeneracy_check_ran",
    "reset_degeneracy_check",
    "BaselineRowNotClean",
    "DegeneracyCheckNotRun",
    "InterpPathSourceMismatch",
    "ContextCacheMismatch",
    "LossColumnsMissing",
    "StepsRowUnavailable",
    "acai_critic_loss",
    "sigmoid_rampup",
    "GLUT_TABLE7_REFERENCE",
    "assert_interp_step_columns",
    "assert_interp_path_source",
    "assert_context_caches",
    "assert_publishable_interpc",
    "ipa_pair_errors",
    "ipb_pair_path",
    "interp_extra_columns",
    "run_setup_block",
]

#: board / ``ARM_AXES`` key.  ``q3vl.whatb.criteria.ARM_AXES["EPR-026"] == ("P1",)``.
ARM = "EPR-026"
AXES: tuple[str, ...] = ("P1",)

#: ``--interp-alpha``.  ``beta0.5`` = ``Beta(0.5, 0.5)`` (EPR-026:414, the ICT
#: search grid subset), ``uniform`` = ``Beta(1,1)`` = ``U(0,1)`` (ablation 3-a),
#: ``endpoints`` = draw from ``{0, 1}`` only (ablation 3-b).
INTERP_ALPHA_MODES: tuple[str, ...] = ("beta0.5", "uniform", "endpoints")
#: ``--interp-where``: interpolate at the generator entry (CGLUT section 3.2's
#: ``e^alpha``) or on the raw 2560-d ``z`` before ``pi`` (ablation 5).  ``pi``
#: contains a LayerNorm, so the two are **not** the same map.
INTERP_WHERE_CHOICES: tuple[str, ...] = ("post_pi", "pre_pi")
#: ``--interp-dist``: L1 (GLUT Eq.6 / NILUT ``fit.py:78``) or MSE (ICT Eq.1).
INTERP_DIST_CHOICES: tuple[str, ...] = ("l1", "mse")
#: ``--interp-target``: the GT LUT mixture (NILUT appendix) or an EMA teacher
#: (ICT Eq.1's ``Mix_lambda(f_theta'(u_j), f_theta'(u_k))``).
INTERP_TARGET_CHOICES: tuple[str, ...] = ("gt_mix", "ema_teacher")
#: ``--interp-ramp``: constant weight (main arm) or ICT's sigmoid ramp-up.
INTERP_RAMP_CHOICES: tuple[str, ...] = ("const", "sigmoid")
#: ``--interp-stage``: joint training (main arm) or NILUT's two-stage fine-tune.
INTERP_STAGE_CHOICES: tuple[str, ...] = ("joint", "finetune")
#: mutually exclusive alternate rows of EPR-026 section 4.3.
ALTERNATE_CHOICES: tuple[str, ...] = ("none", "acai", "jacobian")

#: IP-A alpha grid -- GLUT App B.3 Table 7's own grid (EPR-026:542).
IPA_ALPHA_GRID: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
#: IP-B path resolution ``alpha_k = k / K`` (EPR-026:574, formalisation section 1.3).
IPB_K = 20

_TOTAL_STEPS = 117_440
_STEPS_PER_EPOCH = 2936
_TRAIN_NORMAL_N = 93_934


# --------------------------------------------------------------------------- #
# 0. configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class InterpcConfig:
    """Every flag of ``EPR-026:463`` plus the EPR-024 values it inherits.

    Defaults are the *frozen* / *proposal-written conservative* values, never a
    convenience: ``interp_weight = 0.0`` is the EPR-024 baseline row (section
    4.1 row 0) and the main arm is ``--interp-weight 1``.
    """

    # ---- EPR-024 structure (taken verbatim, EPR-024:610 flag table) --------
    readout: str = "seg_color"
    readout_qtok: int = 0
    cond_dim: int = 64
    n_gauss: int = 48
    gen_width: int = 128
    gen_mode: str = "full"
    loss_level: int = 3
    lambda_img: float = 0.0
    clamp: str = "two"
    residual: bool = True
    batch_samples: int = BATCH_SAMPLES
    queries_per_sample: int = QUERIES_PER_SAMPLE
    #: training corpora (``--data``) and the population MEASURED from them
    data: str = "v2seg"
    train_n: int = _TRAIN_NORMAL_N
    hc_eps: float = 1e-3
    hc_mask: bool = True
    lambda_hc: float = 10.0
    lambda_sparse: float = 0.001
    eps: float = EPS
    context: str = "generated"
    lut_resample: str = "none"
    mining: bool = True

    # ---- optimiser (EPR-024:546-560 = GLUT section 4.1 / App A.1) ---------
    optimizer: str = "adam"
    lr: float = 1e-3
    pi_lr_scale: float = 0.1          # pi stands where ``e_l`` stood -> 1e-4
    geometry_lr_scale: float = 0.1    # App A.1, only bites in affine_only
    scheduler: str = "cosine"
    max_grad_norm: float = 1.0
    epochs: int = 40
    total_steps: int = _TOTAL_STEPS
    steps_per_epoch: int = _STEPS_PER_EPOCH
    seed: int = 20260810
    amp_dtype: str = "bfloat16"       # bf16 forward, fp32 master weights
    mining_start_epoch: int = 5
    mining_end_epoch: int = 20
    mining_r_start: float = 0.10
    mining_r_end: float = 0.40

    # ---- EPR-026's own flags (EPR-026:463) --------------------------------
    interp_weight: float = 0.0
    interp_alpha: str = "beta0.5"
    interp_p_end: float = 1.0 / 3.0
    interp_where: str = "post_pi"
    interp_dist: str = "l1"
    interp_target: str = "gt_mix"
    interp_ema_decay: float = 0.999
    interp_ramp: str = "const"
    interp_ramp_fraction: float = 0.25   # ICT section 3.3: tops out at 1/4 of training
    interp_stage: str = "joint"
    interp_finetune_start: int | None = None
    interp_pairs_per_step: int | None = None   # None -> = batch_samples (1:1)
    interp_hc: bool = False
    alternate: str = "none"
    acai_lambda: float = 0.5          # acai.py:151 ``advweight``
    acai_gamma: float = 0.2           # acai.py:153 ``reg``
    jac_decay: float = 0.01           # train_smooth_diffusion.py:306
    jac_lambda: float = 1.0           # train_smooth_diffusion.py:221 / train.sh

    # ---- criteria knobs ----------------------------------------------------
    degenerate_weight_tau: float = 1e-3
    eval_grid_n: int = 17
    ipb_k: int = IPB_K
    degeneracy_point_std: float = 1e-3
    degeneracy_identity_dev: float = 1e-3
    degeneracy_cross_std: float = 1e-4

    def __post_init__(self) -> None:
        _one_of("--interp-alpha", self.interp_alpha, INTERP_ALPHA_MODES)
        _one_of("--interp-where", self.interp_where, INTERP_WHERE_CHOICES)
        _one_of("--interp-dist", self.interp_dist, INTERP_DIST_CHOICES)
        _one_of("--interp-target", self.interp_target, INTERP_TARGET_CHOICES)
        _one_of("--interp-ramp", self.interp_ramp, INTERP_RAMP_CHOICES)
        _one_of("--interp-stage", self.interp_stage, INTERP_STAGE_CHOICES)
        _one_of("--interp-alternate", self.alternate, ALTERNATE_CHOICES)
        _one_of("--clamp", self.clamp, ("two", "one"))
        _one_of("--readout", self.readout, WHATB_READOUT_KINDS)
        _one_of("--context", self.context, ("teacher", "generated"))
        _one_of("--gen-mode", self.gen_mode, ("full", "affine_only"))
        if self.lut_resample != "none":
            raise ValueError(
                "--lut-resample only implements 'none' (ruling 11.1-3): "
                "resampling changes y and the function-space target stops "
                "matching the dataset's own generation law")
        if self.interp_weight < 0:
            raise ValueError("--interp-weight must be >= 0")
        if not 0.0 <= self.interp_p_end <= 1.0:
            raise ValueError("--interp-p-end is a probability")
        if self.loss_level not in (1, 2, 3, 4):
            raise ValueError("--loss-level must be 1..4")
        if self.data not in DATA_CHOICES:
            raise ValueError(f"--data must be one of {DATA_CHOICES}, got {self.data!r}")
        # ``B * Q == 8192`` is no longer asserted: EPR-030 opened the colour
        # budget on 2026-08-16 (``arms/carrier.py`` BATCH_SPLITS / the
        # ``batch_split_step_matched_to_epr024`` column).  The product is
        # RECORDED per run instead, and a run outside
        # :data:`FROZEN_BATCH_SPLITS` is not step-matched to the EPR-024 board.
        if self.interp_stage == "finetune" and self.interp_finetune_start is None:
            raise ValueError(
                "--interp-stage finetune needs --interp-finetune-start STEP.  "
                "EPR-026 ablation 7 quotes NILUT's appendix ('once the CNILUT is "
                "trained, we can further fine-tune it to perform blending') but "
                "neither the appendix nor the proposal fixes when 'trained' ends, "
                "and inventing a step here would be a silent ruling (NOTES 4).")
        if self.alternate != "none" and self.interp_weight > 0:
            raise ValueError(
                "EPR-026 section 4.3: the alternate rows (a) ACAI and (b) Jacobian "
                "are mutually exclusive with the interpolation-consistency term "
                "('互斥，与 §4.2 的插值一致性项不同时上'); pass --interp-weight 0.")

    # -- derived -------------------------------------------------------------
    @property
    def pairs_per_step(self) -> int:
        """``P``.  NOVEL, no source to copy; conservative default = 1:1 with the
        fit stream (``EPR-026:435-436``)."""
        return int(self.batch_samples if self.interp_pairs_per_step is None
                   else self.interp_pairs_per_step)

    @property
    def interp_enabled(self) -> bool:
        return self.interp_weight > 0.0

    @property
    def colors_per_step(self) -> int:
        return int(self.batch_samples) * int(self.queries_per_sample)

    @property
    def batch_split(self) -> str:
        """``"BxQ"`` -- the name of the row of the ONE shared table this is.

        Derived, not stored: ``arms/carrier.py`` asserts ``name == f"{b}x{q}"``
        for every entry of ``BATCH_SPLITS``, so the name and the pair cannot
        drift apart.
        """
        return f"{int(self.batch_samples)}x{int(self.queries_per_sample)}"

    @property
    def step_matched_to_epr024(self) -> bool:
        return self.batch_split in FROZEN_BATCH_SPLITS

    @property
    def lambda_hc_effective(self) -> float:
        """``arms/carrier.py:347``; the ladder already gates the term below."""
        return effective_lambda_hc(self.lambda_hc, self.loss_level)

    @property
    def lambda_sparse_effective(self) -> float:
        """``arms/carrier.py:351``."""
        return effective_lambda_sparse(self.lambda_sparse, self.loss_level)

    @property
    def beta(self) -> float | None:
        """``Beta(beta, beta)``: 0.5 for ``beta0.5``, 1.0 for ``uniform``."""
        return {"beta0.5": 0.5, "uniform": 1.0}.get(self.interp_alpha)

    @property
    def degeneracy_thresholds(self) -> DegeneracyThresholds:
        return DegeneracyThresholds(point_std=self.degeneracy_point_std,
                                    identity_dev=self.degeneracy_identity_dev,
                                    cross_std=self.degeneracy_cross_std)

    def step_columns(self) -> tuple[str, ...]:
        """First-``steps.jsonl``-row columns this configuration promises.

        Frozen seven, ``+ L_img`` at level 4, ``+ L_interp`` when the arm's own
        variable is on.  Runtime assertion (i) of ``EPR-026:464``: the baseline
        row (``lambda_int = 0``) must **not** carry ``L_interp`` -- both sides
        are asserted, so "the term was silently off" and "the term was silently
        on" are both visible on the artefact.
        """
        extra = ["L_glut"]
        if self.interp_enabled:
            extra.append("L_interp")
        if self.alternate == "acai":
            extra.append("L_acai")
        if self.alternate == "jacobian":
            extra.append("L_jac")
        return step_columns_for(self.loss_level, extra=extra)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update({"arm": ARM, "axes": list(AXES),
                  "pairs_per_step": self.pairs_per_step,
                  "beta": self.beta,
                  "colors_per_step": self.colors_per_step,
                  "colours_per_step": self.colors_per_step,
                  "batch_split_step_matched_to_epr024": self.step_matched_to_epr024,
                  # MEASURED from --data, not the frozen literal
                  "train_normal_n": self.train_n,
                  "lambda_hc_effective": self.lambda_hc_effective,
                  "lambda_sparse_effective": self.lambda_sparse_effective,
                  "lambda_mono_effective": 0.0,
                  "step_columns": list(self.step_columns()),
                  "degeneracy_thresholds": self.degeneracy_thresholds.as_dict()})
        return d


def _one_of(flag: str, value: Any, choices: Sequence[Any]) -> None:
    if value not in choices:
        raise ValueError(f"{flag} must be one of {tuple(choices)}, got {value!r}")


# --------------------------------------------------------------------------- #
# 1. the pairing index (EPR-026:460 item 1)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PairIndex:
    """``source_image_id -> [(sample_id, lut_id)]`` for groups with >= 2 LUTs.

    Built from the split index alone, normal-only.  Measured on this repository
    (2026-08-15, re-counted by this module's own definition):

    ==============  =========  ==============  ==========  =============
    split           sources    with >= 2 LUT   pairs       median / max
    ==============  =========  ==============  ==========  =============
    train normal      22740          19925       226507      4 / 18
    V_what normal       138            120         1311      4 / 12
    T_lut_unseen        157             67          137      2 / 6
    ==============  =========  ==============  ==========  =============

    A "pair" is an unordered pair of **samples** of one source whose ``lut_id``
    differ -- the count the proposal quotes (226507 / 1311 / 137) is exactly
    this, which is how the definition is pinned rather than assumed.
    """

    sources: tuple[str, ...]
    samples: tuple[tuple[tuple[str, str], ...], ...]   # per source: (sample_id, lut_id)
    split: str = ""

    # -- construction --------------------------------------------------------
    @classmethod
    def build(cls, rows: Iterable[Any], *, split: str = "",
              normal_only: bool = True) -> "PairIndex":
        """From ``q3vl.whatb.splits.IndexRow`` objects (or plain mappings)."""
        groups: dict[str, list[tuple[str, str]]] = {}
        for row in rows:
            get = row.get if isinstance(row, Mapping) else (lambda k, _r=row: getattr(_r, k))
            if normal_only and str(get("winner_confidence")) != "normal":
                continue
            groups.setdefault(str(get("source_image_id")), []).append(
                (str(get("sample_id")), str(get("lut_id"))))
        keep = [(s, tuple(v)) for s, v in sorted(groups.items())
                if len({lid for _, lid in v}) >= 2]
        return cls(tuple(s for s, _ in keep), tuple(v for _, v in keep), split)

    # -- facts ---------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.sources)

    @property
    def n_pairs(self) -> int:
        """Unordered sample pairs with different ``lut_id``, over all sources."""
        total = 0
        for members in self.samples:
            m = len(members)
            counts: dict[str, int] = {}
            for _, lid in members:
                counts[lid] = counts.get(lid, 0) + 1
            total += m * (m - 1) // 2 - sum(k * (k - 1) // 2 for k in counts.values())
        return total

    def all_pairs(self, source_idx: int) -> list[tuple[int, int]]:
        """Every valid ``(i, j)`` inside one source -- the enumeration fallback."""
        members = self.samples[source_idx]
        return [(i, j) for i in range(len(members)) for j in range(i + 1, len(members))
                if members[i][1] != members[j][1]]

    def sample_ids(self) -> list[str]:
        return [sid for members in self.samples for sid, _ in members]

    @property
    def sha256(self) -> str:
        """Content hash of the index, recorded in ``run_setup.json``."""
        h = hashlib.sha256()
        h.update(self.split.encode())
        for src, members in zip(self.sources, self.samples):
            h.update(src.encode())
            for sid, lid in members:
                h.update(b"\0" + sid.encode() + b"\0" + lid.encode())
        return h.hexdigest()

    def facts(self) -> dict[str, Any]:
        sizes = [len(m) for m in self.samples]
        return {"split": self.split, "n_sources_with_pairs": len(self.sources),
                "n_pairs": self.n_pairs,
                "group_size_median": float(np.median(sizes)) if sizes else 0.0,
                "group_size_max": int(max(sizes)) if sizes else 0,
                "n_samples_in_pairs": sum(sizes),
                "sha256": self.sha256,
                "definition": ("unordered pairs of samples of one "
                               "source_image_id with different lut_id, "
                               "winner_confidence == normal")}

    def to_json(self) -> str:
        return json.dumps({"split": self.split,
                           "sources": {s: [list(t) for t in m]
                                       for s, m in zip(self.sources, self.samples)}},
                          ensure_ascii=False, sort_keys=True)

    # -- sampling ------------------------------------------------------------
    def draw(self, n: int, rng: np.random.Generator
             ) -> list[tuple[str, tuple[str, str], tuple[str, str]]]:
        """``n`` draws: uniform source (with replacement), then a uniform pair.

        ``EPR-026:427-428`` verbatim.  The within-source draw is rejection
        sampling on "same lut_id"; after eight rejections it falls back to the
        source's explicit pair enumeration, so a source whose members are almost
        all one LUT cannot bias the draw or loop.
        """
        if not self.sources:
            raise ValueError("pair index is empty; no source has two LUTs")
        out: list[tuple[str, tuple[str, str], tuple[str, str]]] = []
        for s_idx in rng.integers(0, len(self.sources), size=int(n)):
            members = self.samples[int(s_idx)]
            i = j = 0
            for _ in range(8):
                i, j = (int(v) for v in rng.choice(len(members), size=2, replace=False))
                if members[i][1] != members[j][1]:
                    break
            else:
                pairs = self.all_pairs(int(s_idx))
                i, j = pairs[int(rng.integers(0, len(pairs)))]
            out.append((self.sources[int(s_idx)], members[i], members[j]))
        return out


# --------------------------------------------------------------------------- #
# 2. alpha sampling (EPR-026:396, :415, :428)
# --------------------------------------------------------------------------- #
class AlphaSampler:
    """``alpha ~ Beta(beta, beta)``, with probability ``p_end`` replaced by an
    endpoint drawn uniformly from ``{0, 1}``.

    ``p_end = 1/3`` is **NOVEL** (``EPR-026:415``): no source prescribes it; the
    stated reason is that 2 of GLUT App B.3's 6 evaluation alphas are endpoints.
    ``p_end = 0`` is ablation 3-c (the ICT / NILUT reading).

    RNG discipline: a private :class:`numpy.random.Generator`, never the global
    stream, so adding the pair stream cannot move the fit stream's draws.
    """

    def __init__(self, cfg: InterpcConfig, *, seed: int | None = None) -> None:
        self.mode = cfg.interp_alpha
        self.beta = cfg.beta
        self.p_end = float(cfg.interp_p_end)
        self.seed = int(cfg.seed if seed is None else seed)
        self.rng = np.random.default_rng(self.seed)
        self.n_drawn = 0
        self.n_endpoints = 0

    def sample_numpy(self, n: int) -> np.ndarray:
        if self.mode == "endpoints":
            a = self.rng.integers(0, 2, size=int(n)).astype(np.float64)
        else:
            assert self.beta is not None
            a = self.rng.beta(self.beta, self.beta, size=int(n))
            if self.p_end > 0:
                force = self.rng.random(int(n)) < self.p_end
                a = np.where(force, self.rng.integers(0, 2, size=int(n)).astype(np.float64), a)
        self.n_drawn += int(n)
        self.n_endpoints += int(np.count_nonzero((a == 0.0) | (a == 1.0)))
        return a

    def sample(self, n: int, *, device: Any = "cpu",
               dtype: torch.dtype = torch.float32) -> Tensor:
        """``(n, 1)`` on the requested device -- the shape the mixer broadcasts."""
        a = torch.from_numpy(self.sample_numpy(n).astype(np.float64))
        return a.to(device=device, dtype=dtype).unsqueeze(-1)

    def facts(self) -> dict[str, Any]:
        return {"mode": self.mode, "beta": self.beta, "p_end": self.p_end,
                "seed": self.seed, "n_drawn": self.n_drawn,
                "n_endpoints": self.n_endpoints,
                "rng": "private numpy Generator (global stream untouched)"}


# --------------------------------------------------------------------------- #
# 3. the model
# --------------------------------------------------------------------------- #
class InterpcArm(nn.Module):
    """``pi`` + CGLUT generator + the shared GLUT carrier.  No new parameters.

    ``EPR-026:369``: the arm adds **0** trainable parameters to EPR-024 -- it
    adds a loss term.  The two interpolation entry points live here only so the
    training side and the evaluation side cannot disagree about what
    ``f^cond_alpha`` means (runtime assertion (iv) of ``EPR-026:464``).
    """

    def __init__(self, cfg: InterpcConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.pi = SegColorProjection(cond_dim=cfg.cond_dim)
        self.generator = CGLUTGenerator(cond_dim=cfg.cond_dim, hidden=cfg.gen_width,
                                        n_gauss=cfg.n_gauss, mode=cfg.gen_mode,
                                        eps=cfg.eps)
        self.carrier = GlutCarrier(clamp=cfg.clamp, residual=cfg.residual, eps=cfg.eps)
        # the only constants -- non-persistent buffers, never a forward-time
        # ``torch.tensor(...)`` (pitfall 2).
        self.register_buffer("grid_eval",
                             uniform_grid(cfg.eval_grid_n), persistent=False)

    # -- conditions ----------------------------------------------------------
    def condition(self, z: Tensor) -> Tensor:
        """``u = pi(z)``; ``z`` is cast onto ``pi``'s (device, dtype) inside."""
        return self.pi(z)

    def mix_condition(self, z_a: Tensor, z_b: Tensor, alpha: Tensor) -> Tensor:
        """``u_alpha`` for the configured ``--interp-where``.

        ``post_pi`` (default, CGLUT section 3.2's ``e^alpha_{l1l2}``)::

            u_alpha = (1 - alpha) pi(z_a) + alpha pi(z_b)

        ``pre_pi`` (ablation 5) interpolates the raw 2560-d read-out first.
        ``pi`` carries a LayerNorm, so the two are different maps -- that is the
        whole content of the ablation.
        """
        ref = self.pi.proj.weight
        a = alpha.to(device=ref.device, dtype=ref.dtype)
        if a.dim() == 1:
            a = a.unsqueeze(-1)
        if self.cfg.interp_where == "pre_pi":
            za = z_a.to(device=ref.device, dtype=ref.dtype)
            zb = z_b.to(device=ref.device, dtype=ref.dtype)
            return self.condition((1.0 - a) * za + a * zb)
        return (1.0 - a) * self.condition(z_a) + a * self.condition(z_b)

    # -- transforms ----------------------------------------------------------
    def params_from_u(self, u: Tensor) -> GlutParams:
        return self.generator(u)

    def forward(self, z: Tensor, x: Tensor, *, return_aux: bool = False,
                clamp: str | None = None):
        """``f_hat(x)`` for the read-out ``z``.  ``x`` is ``(B, P, 3)`` or ``(P, 3)``."""
        return self.transform_from_u(self.condition(z), x, return_aux=return_aux,
                                     clamp=clamp)

    def transform_from_u(self, u: Tensor, x: Tensor, *, return_aux: bool = False,
                         clamp: str | None = None):
        params = self.params_from_u(u)
        ref = params.mu
        xr = x.to(device=ref.device, dtype=ref.dtype)
        return self.carrier(xr, params, clamp=clamp, return_aux=return_aux)

    def transform_pair(self, z_a: Tensor, z_b: Tensor, alpha: Tensor, x: Tensor, *,
                       return_aux: bool = False, clamp: str | None = None):
        """``f^cond_alpha`` -- the one definition both sides use."""
        return self.transform_from_u(self.mix_condition(z_a, z_b, alpha), x,
                                     return_aux=return_aux, clamp=clamp)

    # -- witnesses -----------------------------------------------------------
    @torch.no_grad()
    def step0_witness(self, z: Tensor) -> dict[str, float]:
        """``step0_maxabs_f_minus_id`` on the 17^3 grid (``EPR-024:612``).

        EPR-024/026 take PyTorch default init, so this is **not** expected to be
        0; it is the run-time proof that the initialisation档 is the one the
        board claims.
        """
        x = self.grid_eval
        y = self.forward(z, x)
        d = (y - x.to(device=y.device, dtype=y.dtype)).abs()
        return {"step0_maxabs_f_minus_id": float(d.max()),
                "step0_mean_abs_f_minus_id": float(d.mean()),
                "grid_n": int(self.cfg.eval_grid_n)}

    def param_groups(self) -> list[dict[str, Any]]:
        """``pi`` at ``0.1x`` base lr (EPR-024:551, NOVEL mapping of App A.1's
        "lower learning rate to the style embeddings"), generator at base lr,
        shared geometry (affine-only only) at ``0.1x`` as well."""
        cfg = self.cfg
        groups = self.generator.param_groups(cfg.lr,
                                             geometry_lr_scale=cfg.geometry_lr_scale)
        groups.append({"params": list(self.pi.parameters()),
                       "lr": cfg.lr * cfg.pi_lr_scale, "name": "pi"})
        return groups

    def facts(self) -> dict[str, Any]:
        return {"arm": ARM,
                "generator": self.generator.config,
                "carrier": self.carrier.config,
                "n_params_pi": sum(p.numel() for p in self.pi.parameters()),
                "n_params_generator": sum(p.numel() for p in self.generator.parameters()),
                "n_params_total": sum(p.numel() for p in self.parameters()),
                "n_params_added_vs_epr024": 0}


class EmaTeacher:
    """``theta'`` of ICT Eq.1 -- ablation 8 only (``decay = 0.999``, ICT section 3.3).

    A detached copy of the arm; ``update`` is the standard
    ``theta' <- d theta' + (1-d) theta`` over parameters **and** buffers.
    """

    def __init__(self, arm: InterpcArm, decay: float = 0.999) -> None:
        import copy

        self.decay = float(decay)
        self.model = copy.deepcopy(arm).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.n_updates = 0

    @torch.no_grad()
    def update(self, arm: InterpcArm) -> None:
        d = self.decay
        for t, s in zip(self.model.parameters(), arm.parameters()):
            t.mul_(d).add_(s.detach().to(device=t.device, dtype=t.dtype), alpha=1.0 - d)
        for t, s in zip(self.model.buffers(), arm.buffers()):
            t.copy_(s.detach().to(device=t.device, dtype=t.dtype))
        self.n_updates += 1

    @torch.no_grad()
    def transform(self, z: Tensor, x: Tensor) -> Tensor:
        return self.model(z, x)


class AcaiCritic(nn.Module):
    """Alternate row (a): ``d_omega`` regressing ``alpha`` from a function value vector.

    ACAI's critic reads the *decoded* interpolate; here the decoded object is a
    colour transform, so the critic reads ``f(x)`` on a fixed anchor grid
    (default 9^3 = 729 colours -> 2187 inputs).  The anchor grid, the width and
    the depth are **NOVEL** (ACAI's critic is a conv net over images and has no
    counterpart here); the four numbers that ARE quoted -- ``alpha`` clipped to
    ``[0, 0.5]`` (``acai.py:62-63``), ``lambda = 0.5`` (``:151``),
    ``gamma = 0.2`` (``:153``) and the two loss forms (Eq.1 / Eq.2) -- are taken
    verbatim.
    """

    def __init__(self, *, grid_n: int = 9, hidden: int = 256) -> None:
        super().__init__()
        self.grid_n = int(grid_n)
        self.register_buffer("anchor", uniform_grid(self.grid_n), persistent=False)
        d_in = self.anchor.shape[0] * 3
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, values: Tensor) -> Tensor:
        """``(B, n_anchor, 3) -> (B,)``."""
        ref = self.net[0].weight
        v = values.to(device=ref.device, dtype=ref.dtype)
        return self.net(v.reshape(v.shape[0], -1)).squeeze(-1)


# --------------------------------------------------------------------------- #
# 4. batches
# --------------------------------------------------------------------------- #
@dataclass
class FitBatch:
    """The EPR-024 fit stream of one step.

    ``z`` ``(B, 2560)`` read-outs, ``x`` ``(B, Q, 3)`` query colours,
    ``target`` ``(B, Q, 3)`` = ``L_l(x)`` evaluated with the generator's own
    operator (``lutdata.LutBank.apply``), ``lut_ids`` for ``n_luts_in_batch``.
    ``image`` / ``alpha`` / ``i_star`` are only read at ``--loss-level 4``.
    """

    z: Tensor
    x: Tensor
    target: Tensor
    lut_ids: tuple[str, ...] = ()
    sample_ids: tuple[str, ...] = ()
    image: Tensor | None = None
    alpha: Tensor | float | None = None
    i_star: Tensor | None = None

    def __post_init__(self) -> None:
        if self.x.shape != self.target.shape:
            raise ValueError(f"x {tuple(self.x.shape)} != target "
                             f"{tuple(self.target.shape)}")
        if self.z.shape[0] != self.x.shape[0]:
            raise ValueError(f"z batch {self.z.shape[0]} != x batch {self.x.shape[0]}")

    @property
    def n_colors(self) -> int:
        return int(self.x.shape[0] * self.x.shape[1])


@dataclass
class PairBatch:
    """The interpolation stream: ``P`` pairs, the fit stream's colours.

    ``x`` **must** be the fit stream's colours (``EPR-026:355-357``: sampling
    the interpolation term outside the training colour set would void GLUT App
    A.1's unseen-colour evaluation column).  ``values_a`` / ``values_b`` are
    ``L_a(x)`` / ``L_b(x)`` on those same colours.
    """

    z_a: Tensor
    z_b: Tensor
    x: Tensor
    values_a: Tensor
    values_b: Tensor
    alpha: Tensor
    source_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        p = self.z_a.shape[0]
        for name in ("z_b", "x", "values_a", "values_b", "alpha"):
            t = getattr(self, name)
            if t.shape[0] != p:
                raise ValueError(f"PairBatch.{name} has batch {t.shape[0]}, expected {p}")
        if self.alpha.dim() == 1:
            self.alpha = self.alpha.unsqueeze(-1)

    @property
    def n_pairs(self) -> int:
        return int(self.z_a.shape[0])


# --------------------------------------------------------------------------- #
# 5. losses
# --------------------------------------------------------------------------- #
def glut_loss_terms(y: Tensor, target: Tensor, aux: GlutAux, cfg: InterpcConfig,
                    *, image_terms: tuple[Tensor, Tensor] | None = None
                    ) -> dict[str, Tensor | int]:
    """GLUT Eq.6-8 with the frozen ``L_hc`` protection.  Device-side throughout.

    ``L_rec = ||y_hat - y||_1`` (Eq.6, mean over colours and channels, the form
    NILUT's ``fit.py:78`` also uses);
    ``L_hc  = C (1 - <h_hat, h>)`` in CIELab with ``C`` the **target** chroma and
    both hues normalised by ``max(C, eps_C)``, the whole term multiplied by the
    hard mask ``1[C >= eps_C]``, ``eps_C = 1e-3`` (frozen block item 8; the
    masked count is returned as ``n_hc_masked``);
    ``R_sparse = -(1/N) sum_i [o log(o+eps) + (1-o) log(1-o+eps)]`` (Eq.8).

    Loss levels (EPR-024 section 3.2): 1 = ``L_rec``; 2 = ``+ 10 L_hc``;
    3 = ``+ 0.001 R_sparse`` (default); 4 = ``+ lambda_img L_img`` with
    ``image_terms = (I_hat, I_star)``.
    """
    tgt = target.to(device=y.device, dtype=y.dtype)
    l_rec = (y - tgt).abs().mean()

    lab_hat, lab_tgt = srgb_to_lab(y), srgb_to_lab(tgt)
    _, h_hat, _ = chroma_hue(lab_hat, cfg.hc_eps)
    c_tgt, h_tgt, valid = chroma_hue(lab_tgt, cfg.hc_eps)
    cos = (h_hat * h_tgt).sum(dim=-1)
    per_point = c_tgt * (1.0 - cos)
    if cfg.hc_mask:
        keep = valid.to(dtype=per_point.dtype)
        l_hc = (per_point * keep).sum() / keep.sum().clamp_min(1.0)
        n_masked = int((~valid).sum())
    else:                       # EPR-024's six-arm shared ablation --no-hc-mask
        l_hc = per_point.mean()
        n_masked = 0

    o = aux.opacity
    r_sparse = -(o * torch.log(o + cfg.eps)
                 + (1.0 - o) * torch.log(1.0 - o + cfg.eps)).mean()

    total = l_rec
    if cfg.loss_level >= 2:
        total = total + cfg.lambda_hc * l_hc
    if cfg.loss_level >= 3:
        total = total + cfg.lambda_sparse * r_sparse
    out: dict[str, Tensor | int] = {"L_rec": l_rec, "L_hc": l_hc,
                                    "L_sparse": r_sparse, "L_glut": total,
                                    "n_hc_masked": n_masked}
    if cfg.loss_level >= 4:
        if image_terms is None:
            raise ValueError(
                "--loss-level 4 asks for L_img = ||I_hat - I*||_1 with GT alpha "
                "(EPR-024 section 3.2 level 4) and no image pair was handed in; "
                "the image loader is not part of this arm.")
        i_hat, i_star = image_terms
        l_img = (i_hat - i_star.to(device=i_hat.device, dtype=i_hat.dtype)).abs().mean()
        out["L_img"] = l_img
        out["L_glut"] = out["L_glut"] + cfg.lambda_img * l_img
    return out


def interp_distance(y: Tensor, target: Tensor, dist: str = "l1") -> Tensor:
    """``||.||_1`` (main arm, GLUT Eq.6 family) or MSE (ICT Eq.1's ``ell``)."""
    tgt = target.to(device=y.device, dtype=y.dtype)
    if dist == "l1":
        return (y - tgt).abs().mean()
    if dist == "mse":
        return ((y - tgt) ** 2).mean()
    raise ValueError(f"--interp-dist must be one of {INTERP_DIST_CHOICES}, got {dist!r}")


def sigmoid_rampup(step: int, length: float) -> float:
    """``exp(-5 (1 - t)^2)``, ``t = clip(step / length, 0, 1)``.

    Verbatim from the ICT authors' own repository --
    ``https://raw.githubusercontent.com/vikasverma1077/ICT/master/mean_teacher/ramps.py``
    ``sigmoid_rampup`` (fetched 2026-08-15)::

        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

    ICT section 3.3 ramps the consistency coefficient to its maximum at 1/4 of
    the total epochs; that fraction is :attr:`InterpcConfig.interp_ramp_fraction`.
    """
    if length <= 0:
        return 1.0
    phase = 1.0 - min(max(float(step) / float(length), 0.0), 1.0)
    return float(math.exp(-5.0 * phase * phase))


def interp_weight_at(step: int, cfg: InterpcConfig) -> float:
    """``lambda_int(step)`` under ``--interp-ramp`` and ``--interp-stage``."""
    if not cfg.interp_enabled:
        return 0.0
    if cfg.interp_stage == "finetune":
        start = int(cfg.interp_finetune_start or 0)
        if step < start:
            return 0.0
    w = float(cfg.interp_weight)
    if cfg.interp_ramp == "sigmoid":
        w *= sigmoid_rampup(step, cfg.interp_ramp_fraction * cfg.total_steps)
    return w


# --------------------------------------------------------------------------- #
# 6. hard-example mining (ruling 11.1-4)
# --------------------------------------------------------------------------- #
def _log(value: Tensor | float) -> float:
    """A python float for ``steps.jsonl`` -- detached, so logging never warns
    about (or worse, retains) a graph."""
    return float(value.detach()) if isinstance(value, Tensor) else float(value)


def _broadcast_alpha(alpha: Tensor, ref: Tensor) -> Tensor:
    """``(P, 1)`` -> ``(P, 1, ..., 1)`` on ``ref``'s (device, dtype).

    The condition mixer wants ``(P, 1)`` against ``(P, d)``; the function-value
    mixer wants ``(P, 1, 1)`` against ``(P, Q, 3)``.  One helper so a shape
    mismatch cannot silently broadcast the wrong axis.
    """
    a = alpha.to(device=ref.device, dtype=ref.dtype)
    if a.dim() == 1:
        a = a.unsqueeze(-1)
    while a.dim() < ref.dim():
        a = a.unsqueeze(-1)
    if a.dim() != ref.dim() or a.shape[0] not in (1, ref.shape[0]):
        raise ValueError(f"alpha {tuple(alpha.shape)} does not broadcast against "
                         f"{tuple(ref.shape)}")
    return a


def _mining_ratio_for(cfg: InterpcConfig, step: int, given: float | None) -> float:
    """``r`` for this step: the caller's value, else GLUT App A.1's schedule.

    ``--no-mining`` pins it at 0 so the logged column matches what happened
    rather than what the schedule would have asked for.
    """
    if given is not None:
        return float(given)
    if not cfg.mining:
        return 0.0
    epoch = float(step) / float(max(cfg.steps_per_epoch, 1))
    return mining_ratio(epoch, start_epoch=cfg.mining_start_epoch,
                        end_epoch=cfg.mining_end_epoch,
                        r_start=cfg.mining_r_start, r_end=cfg.mining_r_end)


@torch.no_grad()
def mine_step(arm: InterpcArm, z: Tensor, x_pool: Tensor, target_pool: Tensor,
              x_fresh: Tensor, target_fresh: Tensor, ratio: float
              ) -> tuple[Tensor, Tensor, int]:
    """Within-batch top-``r`` resampling of the colour queries, no cross-step state.

    Ruling 11.1-4: draw 8192 colours, one ``no_grad`` forward, per-colour L1,
    keep the ``r * 8192`` worst and top the batch up with ``(1-r) * 8192`` fresh
    uniform colours.  The selection is done **per sample** (``k = round(r Q)``
    of each sample's own Q colours), which is what keeps the frozen ``B x Q``
    organisation intact; a global top-k over the flattened 8192 would give
    samples ragged colour counts.

    ``topk`` runs on the tensors' own device -- CPU and CUDA break ties
    differently (the where-side 0.296 IoU lesson).
    """
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"mining ratio must be in [0,1], got {ratio}")
    b, q = x_pool.shape[0], x_pool.shape[1]
    k = int(round(float(ratio) * q))
    k = max(0, min(k, q))
    if k == 0:
        return x_fresh[:, :q], target_fresh[:, :q], 0
    y = arm(z, x_pool)
    err = (y - target_pool.to(device=y.device, dtype=y.dtype)).abs().mean(dim=-1)
    idx = torch.topk(err, k, dim=1, largest=True, sorted=False).indices    # (B, k)
    gather = idx.unsqueeze(-1).expand(b, k, 3)
    x_hard = torch.gather(x_pool, 1, gather.to(device=x_pool.device))
    t_hard = torch.gather(target_pool, 1, gather.to(device=target_pool.device))
    n_new = q - k
    x = torch.cat([x_hard, x_fresh[:, :n_new]], dim=1)
    t = torch.cat([t_hard, target_fresh[:, :n_new]], dim=1)
    return x, t, int(k * b)


# --------------------------------------------------------------------------- #
# 7. one training step
# --------------------------------------------------------------------------- #
def train_step(arm: InterpcArm, fit: FitBatch, cfg: InterpcConfig, *,
               step: int, pair: PairBatch | None = None,
               teacher: EmaTeacher | None = None,
               critic: AcaiCritic | None = None,
               jac_state: dict[str, float] | None = None,
               mining_r: float | None = None,
               ) -> tuple[Tensor, dict[str, Any]]:
    """``L = L_GLUT + lambda_int(step) * L_interp`` and the ``steps.jsonl`` row.

    Returns ``(loss, row)``.  ``row`` carries every pre-registered first-row
    column of :meth:`InterpcConfig.step_columns` -- and, when
    ``lambda_int == 0``, deliberately **no** ``L_interp`` key, so the baseline
    row cannot look like the treatment row (runtime assertion (i)).

    The pair stream never touches ``L_GLUT`` (``EPR-026:429``): with
    ``lambda_int = 0`` the fit stream's forward, loss, gradients and sampling
    are EPR-024's, bit for bit.
    """
    y, aux = arm(fit.z, fit.x, return_aux=True)
    image_terms = None
    if cfg.loss_level >= 4 and fit.image is not None and fit.i_star is not None:
        from q3vl.whatb.criteria import compose_hat

        f_img = arm(fit.z, fit.image.movedim(-3, -1)).movedim(-1, -3)
        image_terms = (compose_hat(fit.image, fit.alpha if fit.alpha is not None else 1.0,
                                   f_img), fit.i_star)
    terms = glut_loss_terms(y, fit.target, aux, cfg, image_terms=image_terms)
    loss = terms["L_glut"]
    assert isinstance(loss, Tensor)

    row: dict[str, Any] = {
        "step": int(step),
        "L_rec": _log(terms["L_rec"]), "L_hc": _log(terms["L_hc"]),
        "L_sparse": _log(terms["L_sparse"]), "L_glut": _log(terms["L_glut"]),
        "n_colors": fit.n_colors,
        "n_luts_in_batch": len(set(fit.lut_ids)) if fit.lut_ids else int(fit.z.shape[0]),
        "mining_ratio": float(_mining_ratio_for(cfg, step, mining_r)),
        "n_hc_masked": int(terms["n_hc_masked"]),
        "n_degenerate_precision": int(aux.degenerate_precision.sum()),
        "oob_rate_train": float(aux.oob_mask().to(dtype=y.dtype).mean()),
    }
    if "L_img" in terms:
        row["L_img"] = _log(terms["L_img"])

    # ---- the arm's own term ------------------------------------------------
    w_int = interp_weight_at(step, cfg)
    if cfg.interp_enabled:
        if pair is None:
            raise ValueError(
                f"--interp-weight {cfg.interp_weight} is on and no PairBatch was "
                "handed in; a step that silently skips the arm's only change is "
                "exactly the 'defined but not wired' failure.")
        mixed = arm.transform_pair(pair.z_a, pair.z_b, pair.alpha, pair.x,
                                   return_aux=cfg.interp_hc)
        y_mix, aux_mix = mixed if cfg.interp_hc else (mixed, None)
        a = _broadcast_alpha(pair.alpha, y_mix)
        if cfg.interp_target == "gt_mix":
            with torch.no_grad():
                gt_mix = ((1.0 - a) * pair.values_a.to(device=y_mix.device, dtype=y_mix.dtype)
                          + a * pair.values_b.to(device=y_mix.device, dtype=y_mix.dtype))
        else:                    # ablation 8: ICT Eq.1's EMA pseudo-labels
            if teacher is None:
                raise ValueError("--interp-target ema_teacher needs an EmaTeacher")
            with torch.no_grad():
                gt_mix = ((1.0 - a) * teacher.transform(pair.z_a, pair.x)
                          + a * teacher.transform(pair.z_b, pair.x))
        l_int = interp_distance(y_mix, gt_mix, cfg.interp_dist)
        if cfg.interp_hc:        # ablation 9
            assert aux_mix is not None
            hc = glut_loss_terms(y_mix, gt_mix, aux_mix, cfg)["L_hc"]
            assert isinstance(hc, Tensor)
            l_int = l_int + cfg.lambda_hc * hc
        loss = loss + w_int * l_int
        row.update({"L_interp": _log(l_int), "lambda_interp": float(w_int),
                    "alpha_mean": _log(a.mean()), "n_pairs": pair.n_pairs,
                    "interp_where": cfg.interp_where,
                    "interp_target": cfg.interp_target})

    # ---- mutually exclusive alternate rows ---------------------------------
    if cfg.alternate == "acai":
        loss, extra = _acai_generator_term(arm, fit, pair, cfg, critic, loss)
        row.update(extra)
    elif cfg.alternate == "jacobian":
        term, extra = _jacobian_term(arm, fit, cfg, jac_state)
        loss = loss + term
        row.update(extra)

    row["loss"] = float(loss.detach())
    record_step_witness(row)
    return loss, row


def _acai_generator_term(arm: InterpcArm, fit: FitBatch, pair: PairBatch | None,
                         cfg: InterpcConfig, critic: AcaiCritic | None,
                         loss: Tensor) -> tuple[Tensor, dict[str, Any]]:
    """Alternate (a), generator half: ``+ lambda ||d_omega(x_hat_alpha)||^2`` (Eq.2).

    The critic's own objective (Eq.1) is stepped separately by
    :func:`acai_critic_loss` -- two optimisers, as in ``acai.py:89``.
    """
    if critic is None or pair is None:
        raise ValueError("--interp-alternate acai needs an AcaiCritic and a PairBatch")
    alpha = _acai_alpha(pair.alpha)
    y_mix = arm.transform_pair(pair.z_a, pair.z_b, alpha, critic.anchor)
    adv = (critic(y_mix) ** 2).mean()
    return loss + cfg.acai_lambda * adv, {"L_acai": _log(adv),
                                          "acai_alpha_mean": _log(alpha.mean())}


def _acai_alpha(alpha: Tensor) -> Tensor:
    """``alpha = 0.5 - |alpha - 0.5|`` -- ``acai.py:62-63`` verbatim."""
    return 0.5 - (alpha - 0.5).abs()


def acai_critic_loss(arm: InterpcArm, pair: PairBatch, cfg: InterpcConfig,
                     critic: AcaiCritic) -> tuple[Tensor, dict[str, Any]]:
    """ACAI Eq.1: ``||d(x_hat_a) - a||^2 + ||d(gamma x + (1-gamma) g(f(x)))||^2``.

    Domain mapping (NOVEL, the transfer itself has no precedent): ``x`` is a real
    library transform ``L_a`` evaluated on the anchor grid and ``g(f(x))`` is the
    arm's own reconstruction ``f_hat_a`` on the same grid.
    """
    alpha = _acai_alpha(pair.alpha)
    with torch.no_grad():
        y_mix = arm.transform_pair(pair.z_a, pair.z_b, alpha, critic.anchor)
        f_a = arm(pair.z_a, critic.anchor)
    real = pair.values_a
    if real.shape[1] != critic.anchor.shape[0]:
        raise ValueError(
            "the ACAI critic reads its own anchor grid; hand in values_a "
            f"evaluated on {critic.anchor.shape[0]} colours, got {real.shape[1]}")
    g = cfg.acai_gamma
    mixed_real = g * real.to(device=f_a.device, dtype=f_a.dtype) + (1.0 - g) * f_a
    a = alpha.reshape(-1).to(device=y_mix.device, dtype=y_mix.dtype)
    loss = ((critic(y_mix) - a) ** 2).mean() + (critic(mixed_real) ** 2).mean()
    return loss, {"L_acai_critic": float(loss.detach())}


def _jacobian_term(arm: InterpcArm, fit: FitBatch, cfg: InterpcConfig,
                   state: dict[str, float] | None) -> tuple[Tensor, dict[str, Any]]:
    """Alternate (b): ``(||J_z^T u||_2 - a)^2`` with ``a`` an EMA (decay 0.01).

    Smooth Diffusion ``train_smooth_diffusion.py:313-321`` in this domain:
    one ``autograd.grad(..., create_graph=True)`` of ``<f_{G(pi(z))}(x), u>``
    with respect to ``z``, ``u`` a random unit vector in function-value space;
    the anchor ``a`` follows the running mean rather than being pushed to 0
    (``:319``).
    """
    if state is None:
        raise ValueError("--interp-alternate jacobian needs a jac_state dict "
                         "carrying the EMA anchor across steps")
    z = fit.z.detach().clone().requires_grad_(True)
    y = arm(z, fit.x)
    u = torch.randn_like(y)
    u = u / u.reshape(u.shape[0], -1).norm(dim=1).clamp_min(1e-12).reshape(
        -1, *([1] * (u.dim() - 1)))
    grad, = torch.autograd.grad((y * u).sum(), z, create_graph=True)
    r = grad.reshape(grad.shape[0], -1).pow(2).sum(dim=1).sqrt()
    anchor = float(state.get("anchor", float(r.mean().detach())))
    anchor = anchor + cfg.jac_decay * (float(r.mean().detach()) - anchor)
    state["anchor"] = anchor
    reg = ((r - anchor) ** 2).mean()
    return cfg.jac_lambda * reg, {"L_jac": float(reg.detach()),
                                  "jac_anchor": anchor}


# --------------------------------------------------------------------------- #
# 8. optimiser / schedule (GLUT section 4.1 + App A.1)
# --------------------------------------------------------------------------- #
def build_optimizer(arm: InterpcArm, cfg: InterpcConfig,
                    extra_groups: Sequence[Mapping[str, Any]] = ()
                    ) -> torch.optim.Optimizer:
    """Adam (App A.1), base lr 1e-3, ``pi`` at 0.1x.  betas are PyTorch's default
    (0.9, 0.999) -- the paper does not give them; EPR-024:546 lists that as a
    declared deviation rather than a copied value."""
    if cfg.optimizer != "adam":
        raise ValueError("GLUT App A.1 specifies Adam; --optimizer takes 'adam'")
    groups = arm.param_groups() + [dict(g) for g in extra_groups]
    return torch.optim.Adam(groups, lr=cfg.lr)


def build_scheduler(opt: torch.optim.Optimizer, cfg: InterpcConfig):
    """Cosine annealing over the whole run (``T_max = total_steps``, App A.1)."""
    if cfg.scheduler != "cosine":
        raise ValueError("GLUT App A.1 specifies cosine annealing over the whole "
                         "training; --scheduler takes 'cosine'")
    return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.total_steps)


# --------------------------------------------------------------------------- #
# 9. guards
# --------------------------------------------------------------------------- #
class BaselineRowNotClean(AssertionError):
    """``lambda_int == 0`` but the first step row carries ``L_interp``.

    The baseline row of section 4.1 has to be EPR-024 bit-for-bit; a row that
    reports an interpolation loss is, whatever else it is, not that row.
    """


#: EPR-026 wrote this check first; it is shared now
#: (:class:`q3vl.whatb.guards.DegeneracyCheckNotRun`, enforced for every arm by
#: ``publish.assert_publishable``).  The name is kept as an alias so the arm's
#: own gate and its tests keep raising the same class as the shared one -- two
#: classes with the same name would make ``except`` catch only one of them.
DegeneracyCheckNotRun = _SharedDegeneracyCheckNotRun


class InterpPathSourceMismatch(AssertionError):
    """The board's ``interp_where`` differs from the evaluator's."""


class ContextCacheMismatch(AssertionError):
    """A read-out cache was produced by a different checkpoint / read-out kind."""


def reset_degeneracy_check() -> None:
    """Drop the in-process record (tests, and a runner re-entering training)."""
    _clear_degeneracy_check()


def degeneracy_check_ran() -> dict[str, Any] | None:
    """The shared witness -- one record for all six arms, set by the guard."""
    return _shared_degeneracy_check_ran()


def quick_eval_guard(arm: InterpcArm, z: Tensor, cfg: InterpcConfig, *,
                     x: Tensor | None = None, where: str = "quick_eval",
                     exit_process: bool = True) -> dict[str, Any]:
    """Run the three degeneracy assertions at the FIRST quick eval.

    ``f_hat`` is evaluated for ``B`` different conditions on the evaluation grid
    and handed to
    :func:`q3vl.whatb.degeneracy.assert_transform_not_degenerate`, which exits
    with ``SystemExit(2)`` (printing the three measured numbers next to their
    floors) if the transform is flat across colours, is the identity, or is the
    same for every sample.  The thresholds travel into ``run_setup.json`` via
    :meth:`InterpcConfig.degeneracy_thresholds`.
    """
    if z.dim() != 2 or z.shape[0] < 2:
        raise ValueError(
            "the degeneracy guard needs at least two distinct conditions: the "
            "cross-sample check is the one that catches 'one transform for every "
            f"instruction', and z has shape {tuple(z.shape)}")
    grid = arm.grid_eval if x is None else x
    with torch.no_grad():
        y = arm(z, grid)
    report = assert_transform_not_degenerate(
        y, grid, thresholds=cfg.degeneracy_thresholds, where=where,
        exit_process=exit_process,
        extra={"arm": ARM, "interp_weight": cfg.interp_weight,
               "n_conditions": int(z.shape[0])})
    # the witness is recorded inside assert_transform_not_degenerate (shared,
    # one record for all six arms); this only reads it back for the caller
    return dict(degeneracy_check_ran() or {"where": where, **report.as_dict()})


def assert_interp_step_columns(cfg: InterpcConfig, *,
                               steps_row: Mapping[str, Any] | None = None,
                               steps_path: Any = None
                               ) -> tuple[dict[str, Any], str]:
    """Runtime assertion (i) of ``EPR-026:464``, self-fetching, three tiers.

    * no row anywhere -> :class:`~q3vl.whatb.guards.StepsRowUnavailable`
      ("nobody handed me a row" -- a wiring failure, not a pass),
    * a row missing a promised column -> :class:`~q3vl.whatb.guards.LossColumnsMissing`
      ("the loss did not run"),
    * ``lambda_int == 0`` with an ``L_interp`` column ->
      :class:`BaselineRowNotClean`.

    The two exception types are the where-side SEGSAM / PRND lesson: the arm
    hook there received ``steps_row=None`` from one of two disagreeing call
    sites and reported a loss that had in fact run.
    """
    row, source = assert_first_step_columns(cfg.step_columns(),
                                            steps_row=steps_row,
                                            steps_path=steps_path)
    if not cfg.interp_enabled and "L_interp" in row:
        raise BaselineRowNotClean(
            f"--interp-weight is {cfg.interp_weight} (the EPR-024 baseline row) "
            f"but the first step row (source={source}) carries L_interp="
            f"{row['L_interp']!r}.  Row 0 of section 4.1 has to be EPR-024 "
            "bit-for-bit; this one is not.")
    return row, source


def assert_interp_path_source(board: Mapping[str, Any], cfg: InterpcConfig) -> str:
    """Runtime assertion (iv): the board's interpolation path is the trained one.

    ``--interp-where`` decides what ``f^cond_alpha`` means.  If the board was
    produced with one reading and the run trained the other, IP-A is measuring a
    path the arm never optimised, and nothing else on the board would show it.
    """
    got = ((board.get("interp") or {}).get("interp_where")
           or board.get("interp_where"))
    if got is None:
        raise InterpPathSourceMismatch(
            "the board records no interp_where; the evaluator and the trainer "
            "must agree on the interpolation path (EPR-026:464 (iv)) and an "
            "unrecorded path cannot be checked")
    if str(got) != cfg.interp_where:
        raise InterpPathSourceMismatch(
            f"board interp_where={got!r} but this run trains "
            f"{cfg.interp_where!r}; IP-A would score a path the arm never saw")
    return str(got)


def assert_context_caches(caches: Mapping[str, Mapping[str, Any]], *,
                          checkpoint: str, readout_kind: str = "seg_color"
                          ) -> dict[str, Any]:
    """Runtime assertion (iii): the four read-out caches match this base.

    ``none`` / ``shuffle`` / ``irrelevant`` / ``const``, each stamped with its
    ``checkpoint`` (shape of ``q3vl/whereb/gencontext.py:115-125, 163-172``).
    The three negative controls **must** be regenerated reasonings: a
    teacher-forced original reasoning keeps the true colour semantics at the
    ``<seg_color>`` position and the control silently measures nothing.
    """
    want = ("none", "shuffle", "irrelevant", "const")
    missing = [t for t in want if t not in caches]
    if missing:
        raise ContextCacheMismatch(
            f"missing read-out caches {missing}; N1/N2/N3 need their own "
            "regenerated-reasoning caches (frozen block item 6)")
    report: dict[str, Any] = {}
    for tag in want:
        meta = caches[tag]
        ck, kind = str(meta.get("checkpoint")), str(meta.get("readout_kind"))
        src = str(meta.get("context_source", ""))
        if ck != str(checkpoint):
            raise ContextCacheMismatch(
                f"cache {tag!r} was produced with checkpoint {ck!r}, this run is "
                f"on {checkpoint!r}")
        if kind != readout_kind:
            raise ContextCacheMismatch(
                f"cache {tag!r} has readout_kind={kind!r}, this run uses "
                f"{readout_kind!r}")
        if tag != "none" and src != "generated":
            raise ContextCacheMismatch(
                f"control cache {tag!r} has context_source={src!r}; the three "
                "negative controls must be regenerated (teacher-forced reasoning "
                "makes the control vacuous)")
        report[tag] = {"checkpoint": ck, "readout_kind": kind,
                       "context_source": src, "n": meta.get("n")}
    return report


def assert_publishable_interpc(board: Mapping[str, Any], cfg: InterpcConfig, *,
                               steps_row: Mapping[str, Any] | None = None,
                               steps_path: Any = None,
                               eval_only: bool = False,
                               require_degeneracy_check: bool = True
                               ) -> dict[str, Any]:
    """The arm's publication gate: shared checks + the two EPR-026 ones.

    Order matters: the degeneracy witness and the interpolation-path check come
    first, because both are cheap and both describe whether the numbers below
    them mean anything.
    """
    report: dict[str, Any] = {"arm": ARM}
    if require_degeneracy_check:
        state = degeneracy_check_ran()
        if not state:
            raise DegeneracyCheckNotRun(
                "the first-quick-eval degeneracy guard never ran for this "
                "process (q3vl.whatb.arms.interpc.quick_eval_guard).  A board "
                "published without it cannot tell a trained transform from a "
                "constant one -- which is how PRND and CONDINST spent 2.6 "
                "GPU-hours on a std-0 field.")
        report["degeneracy"] = state
    report["interp_where"] = assert_interp_path_source(board, cfg)
    if not eval_only:
        row, source = assert_interp_step_columns(cfg, steps_row=steps_row,
                                                 steps_path=steps_path)
        report["steps"] = {"source": source,
                           "values": {c: row[c] for c in cfg.step_columns()}}
        steps_row = row
    report["shared"] = assert_publishable(
        board, ARM, steps_row=steps_row, steps_path=steps_path,
        eval_only=eval_only, loss_level=cfg.loss_level,
        extra_step_columns=[c for c in cfg.step_columns()
                            if c not in step_columns_for(cfg.loss_level)],
        axes=AXES)
    return report


# --------------------------------------------------------------------------- #
# 10. the interpolation criteria (section 4.F)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def ipa_pair_errors(arm: InterpcArm, z_a: Tensor, z_b: Tensor,
                    values_a: Tensor, values_b: Tensor, x: Tensor, *,
                    alphas: Sequence[float] = IPA_ALPHA_GRID,
                    weights: Tensor | None = None) -> dict[str, Any]:
    """IP-A for one pair: the four columns section 4.F requires, on ``X_grid``.

    * ``arm`` -- ``dE00(f^cond_alpha, (1-a) L_a + a L_b)``;
    * ``output_mix`` -- the trivial column ``(1-a) f_hat_a + a f_hat_b`` against
      the same GT.  **Without this column an interpolation claim is void**
      (section 4.F failure modes): by construction it beats the conditional path
      almost everywhere;
    * ``endpoint`` -- the arm at ``alpha in {0, 1}``, i.e. its plain fit error;
    * the GLUT external reference is a constant of the protocol, not a
      measurement, and is carried in :func:`interp_extra_columns`.

    ``x`` is the query set (17^3 by default) and ``values_*`` are ``L_a(x)`` /
    ``L_b(x)``.  Everything stays on ``x``'s device.
    """
    z_a2 = z_a if z_a.dim() == 2 else z_a.unsqueeze(0)
    z_b2 = z_b if z_b.dim() == 2 else z_b.unsqueeze(0)
    f_a = arm(z_a2, x)[0]
    f_b = arm(z_b2, x)[0]
    va = values_a.to(device=f_a.device, dtype=f_a.dtype)
    vb = values_b.to(device=f_a.device, dtype=f_a.dtype)
    arm_col: list[float] = []
    mix_col: list[float] = []
    for a in alphas:
        alpha = f_a.new_full((1, 1), float(a))
        f_alpha = arm.transform_pair(z_a2, z_b2, alpha, x)[0]
        gt = (1.0 - float(a)) * va + float(a) * vb
        arm_col.append(float(function_distance(f_alpha, gt, weights)))
        trivial = ((1.0 - float(a)) * f_a + float(a) * f_b).clamp(0.0, 1.0)
        mix_col.append(float(function_distance(trivial, gt, weights)))
    return {"alphas": [float(a) for a in alphas],
            "arm": arm_col, "output_mix": mix_col,
            "endpoint": [arm_col[0], arm_col[-1]],
            "interp_grid": float(np.mean(arm_col)),
            "interp_grid_interior": float(np.mean(arm_col[1:-1])) if len(arm_col) > 2 else None,
            "metric": "dE00 on the query set (X_grid unless weights are given)"}


@torch.no_grad()
def ipb_pair_path(arm: InterpcArm, z_a: Tensor, z_b: Tensor, x: Tensor, *,
                  k: int = IPB_K, library: LibraryValues | None = None,
                  tau: float = 1e-3, weights: Tensor | None = None
                  ) -> dict[str, Any]:
    """IP-B for one pair: the six path quantities + ``d_lib`` + the two rates.

    ``path_len`` / ``chord`` / ``rho`` / ``sigma_bar`` / ``jump_max`` /
    ``mono_rate`` come from :func:`q3vl.whatb.criteria.path_quantities` on the
    clamped path (the headline-consistent function values, ``J`` **not**
    percentile-trimmed).

    ``oob_rate`` is measured **before** the clamp, which is what section 2.4
    defines (``Pr_x[f_alpha(x) not in [0,1]^3]`` clamp 前); the post-clamp figure
    that ``path_quantities`` computes on the clamped path is kept separately as
    ``oob_rate_postclamp`` rather than overwritten, so neither number can be
    mistaken for the other.  The degenerate-weight rate
    ``Pr_x[sum_j p_j o_j < tau]`` (proposition 2) is measured on the same
    forward.
    """
    if library is not None and tuple(library.x.shape) != tuple(x.shape):
        raise ValueError(
            f"d_lib is min_l D_X(f_alpha, L_l) on the SAME X as the path: the "
            f"library was evaluated on {tuple(library.x.shape)} and the path on "
            f"{tuple(x.shape)}")
    z_a2 = z_a if z_a.dim() == 2 else z_a.unsqueeze(0)
    z_b2 = z_b if z_b.dim() == 2 else z_b.unsqueeze(0)
    frames: list[Tensor] = []
    oob: list[float] = []
    degen: list[float] = []
    d_lib: list[float] = []
    for i in range(k + 1):
        a = float(i) / float(k)
        alpha = x.new_full((1, 1), a)
        y, aux = arm.transform_pair(z_a2, z_b2, alpha, x, return_aux=True)
        frames.append(y[0])
        oob.append(float(aux.oob_mask().to(dtype=y.dtype).mean()))
        degen.append(float(aux.degenerate_weight_mask(tau).to(dtype=y.dtype).mean()))
        if library is not None:
            d_lib.append(float(library.distance_to(y[0], metric="de00").min()))
    path = torch.stack(frames, dim=0)
    out = path_quantities(path, weights=weights)
    out["oob_rate_postclamp"] = out["oob_rate"]
    out["oob_rate"] = float(np.mean(oob))
    out["oob_rate_per_alpha"] = oob
    out["degenerate_weight_rate"] = float(np.mean(degen))
    out["tau"] = float(tau)
    if library is not None:
        out["d_lib"] = d_lib
        out["d_lib_mid"] = d_lib[len(d_lib) // 2]
    return out


#: GLUT App B.3 Table 7, CGLUT-32L, PSNR at the six alphas (external reference
#: only: MIT5K 100 images x 21 pairs of a 7-LUT library -- a different dataset
#: and a different library, so it is an order-of-magnitude reference and never a
#: paired comparison, EPR-026:577-579).
GLUT_TABLE7_REFERENCE: dict[str, Any] = {
    "source": "GLUT arXiv:2605.19889 App B.3 Table 7",
    "alphas": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "CGLUT-32L_Full_PSNR": [48.67, 35.44, 31.16, 31.33, 34.64, 47.95],
    "CGLUT-32L_SharedGeo_PSNR": [47.36, 38.46, 34.67, 34.47, 37.60, 46.18],
    "CGLUT-32L_Full_dE00": [0.59, 3.08, 4.48, 4.70, 3.23, 0.56],
    "CGLUT-32L_SharedGeo_dE00": [0.66, 2.11, 3.19, 2.96, 1.94, 0.68],
    "caveat": ("different dataset (MIT5K), different LUT library (7 styles), "
               "no additional blending constraint during training -- magnitude "
               "reference only, never a paired delta"),
}


def interp_extra_columns(ipa: Sequence[Mapping[str, Any]],
                         ipb: Sequence[Mapping[str, Any]],
                         cfg: InterpcConfig) -> dict[str, dict[str, Any]]:
    """The P1 columns for ``build_board(extra_columns=...)``.

    Produces the four keys ``assert_criteria_ran`` requires of a P1 arm --
    ``interp_grid`` / ``path_len`` / ``mono_rate`` / ``oob_rate`` -- plus the
    diagnostics that make them readable (the trivial ``output_mix`` column, the
    endpoint column, ``rho`` / ``sigma_bar`` / ``jump_max`` / ``chord``,
    ``d_lib``, the degenerate-weight rate and the external GLUT reference).

    ``interp_grid`` is the IP-A arm column averaged over the six-point alpha
    grid; the per-alpha vector travels in the same dict, so no reader has to
    guess which alpha a scalar came from.
    """
    cols: dict[str, dict[str, Any]] = {}
    if ipa:
        alphas = list(ipa[0]["alphas"])
        arm_per_alpha = np.asarray([r["arm"] for r in ipa], dtype=np.float64)
        mix_per_alpha = np.asarray([r["output_mix"] for r in ipa], dtype=np.float64)
        cols["interp_grid"] = {
            **describe([float(r["interp_grid"]) for r in ipa]),
            "alphas": alphas,
            "per_alpha_mean": [float(v) for v in arm_per_alpha.mean(axis=0)],
            "quantity": ("IP-A: dE00(f^cond_alpha, (1-a)L_a + a L_b) on the "
                         f"{cfg.eval_grid_n}^3 grid, averaged over the alpha grid"),
            "interp_where": cfg.interp_where,
        }
        cols["interp_output_mix"] = {
            **describe([float(np.mean(r["output_mix"])) for r in ipa]),
            "per_alpha_mean": [float(v) for v in mix_per_alpha.mean(axis=0)],
            "quantity": ("trivial column (1-a) f_hat_a + a f_hat_b against the "
                         "same GT; section 4.F: an interpolation claim without "
                         "this column is void"),
        }
        cols["interp_endpoint"] = {
            **describe([float(np.mean(r["endpoint"])) for r in ipa]),
            "quantity": "arm at alpha in {0,1} = the plain fit error",
        }
        cols["glut_external_reference"] = {"n": len(alphas), **GLUT_TABLE7_REFERENCE}
    if ipb:
        for key, quantity in (
                ("path_len", "sum_k D_X(f_k, f_{k+1}), K = %d" % cfg.ipb_k),
                ("chord", "D_X(f_0, f_1)"),
                ("rho", "path_len / chord (>= 1)"),
                ("sigma_bar", "std_k(delta_k) / (path_len / K)"),
                ("jump_max", "K * max_k delta_k, NO percentile trimming"),
                ("mono_rate", "sign agreement of the CIELab mean-b* readout"),
                ("oob_rate", "Pr_x[f_alpha(x) outside [0,1]^3] BEFORE the clamp"),
                ("degenerate_weight_rate",
                 "Pr_x[sum_j p_j o_j < tau] (proposition 2)"),
                ("d_lib_mid", "min_l D_X(f_0.5, L_l) over Lib_tr"),
        ):
            vals = [r.get(key) for r in ipb]
            if all(v is None for v in vals):
                continue
            col = {**describe(vals), "quantity": quantity}
            if key == "mono_rate":
                col["random_floor"] = 0.5
            if key == "oob_rate":
                col["postclamp"] = describe([r.get("oob_rate_postclamp") for r in ipb])
            cols[key] = col
    return cols


# --------------------------------------------------------------------------- #
# 11. run_setup
# --------------------------------------------------------------------------- #
def run_setup_block(cfg: InterpcConfig, arm: InterpcArm | None = None, *,
                    pair_index: PairIndex | None = None,
                    alpha_sampler: AlphaSampler | None = None,
                    extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Everything about this run that a reader needs and cannot re-derive."""
    block: dict[str, Any] = {
        "arm": ARM, "axes": list(AXES), "config": cfg.as_dict(),
        "frozen": {
            # MEASURED from --data; the frozen sft2seg literal is kept beside it
            "train_normal_n": cfg.train_n,
            "train_normal_n_frozen_v2seg": _TRAIN_NORMAL_N,
            "data": cfg.data,
            "batch_split": cfg.batch_split,
            "batch": f"{cfg.batch_samples}x{cfg.queries_per_sample}",
            "colors_per_step": cfg.colors_per_step,
            "colours_per_step": cfg.colors_per_step,
            "batch_split_step_matched_to_epr024": cfg.step_matched_to_epr024,
            "steps_per_epoch": cfg.steps_per_epoch,
            "total_steps": cfg.total_steps,
            "base_lr": cfg.lr,
            "clamp": cfg.clamp,
            "headline_formation": "I_hat = (1-a) * I + a * f_hat(I)",
            "hc_eps_c": cfg.hc_eps,
            "preregistered_keys": 12,
        },
        "required_criteria_axes": list(AXES),
        "step_columns": list(cfg.step_columns()),
        "degeneracy_thresholds": cfg.degeneracy_thresholds.as_dict(),
    }
    if arm is not None:
        block["model"] = arm.facts()
    if pair_index is not None:
        block["pair_index"] = pair_index.facts()
    if alpha_sampler is not None:
        block["alpha_sampler"] = alpha_sampler.facts()
    if extra:
        block.update(dict(extra))
    return block
