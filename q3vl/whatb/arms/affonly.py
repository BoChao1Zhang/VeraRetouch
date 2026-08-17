"""EPR-025 AFFONLY -- affine-only conditional generation (``12N + 12``).

Spec: ``experiments/prs/EPR-025_affine-only-conditional-head/PROPOSAL.md``
(819 lines, read in full).  The one structural change against the CGLUT carrier
of EPR-024::

    {mu_i, Sigma_i, o_i}   condition-generated  ->  condition-INDEPENDENT nn.Parameter
    theta_gen              22N + 12             ->  12N + 12  =  {M_i, b_i} u {G, g}

Everything else -- Eq.1-5, Eq.6-8, the sampling, the optimiser, the step count,
the criteria -- is unchanged (proposal section 1.1).  Because ``w_i(x)`` then no
longer depends on the condition, ``f`` is **linear in the generated parameters**
(proposition 1), which is what this arm exists to test.

What lives here
---------------
* :class:`AffineOnlyConfig`  -- every flag of section 3.4-(12), with the proposal's
  defaults, plus ``to_dict()`` for ``run_setup.json``;
* :class:`AffineOnlyHead`    -- ``pi`` + generator + carrier, with the three
  ``--share`` rows and the three ``--global-affine`` rows of section 4.1;
* :func:`loss_terms`         -- GLUT Eq.6-8 with the frozen ``L_hc`` mask;
* :func:`build_optimizer` / :func:`build_scheduler` -- Adam(1e-3) + cosine, the
  0.1x shared-geometry group, and the section 3.4-(8) startup assertion;
* :func:`build_training_colors` -- ruling 11.1-4 within-batch top-r mining;
* the arm's **three run-time assertions** of section 3.6 plus the batch-wide
  degeneracy guard, all of which refuse the board rather than warn;
* the P1 interpolation protocols (IP-A / IP-B) and the board wiring.

Frozen quantities this module hard-codes (cross-arm frozen block, byte-identical
in all six proposals; ``docs/HANDOFF_whatb_2026-08-15.md`` section 3.1)::

    train normal-only n = 93934 | B = 32 x Q = 256 = 8192 colours/step
    2936 steps/epoch | 117,440 steps (40 epochs) | clamp default "two"
    headline  Î = (1-a) ⊙ I + a ⊙ f̂(I)          | the twelve pre-registered keys

Discipline (each item paid for on the where side last week)
-----------------------------------------------------------
* every tensor entering a forward is moved with an explicit
  ``.to(device=ref.device, dtype=ref.dtype)``; the only constants are
  ``register_buffer(..., persistent=False)`` and there is no ``torch.tensor(...)``
  inside any ``forward``;
* every criterion is computed on the tensor's own device -- no ``.cpu()`` before
  a comparison or a ``topk``;
* the board-time assertion fetches its own first-step row three tiers deep
  (:mod:`q3vl.whatb.guards`), and "nobody handed me a row" and "the loss did not
  publish its columns" are different exceptions;
* the degeneracy guard runs at the **first** quick eval and leaves via
  ``SystemExit(2)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor, nn

from q3vl.whatb.caliber import (
    DATA_CHOICES,
    FROZEN_BATCH_SPLITS,
    effective_lambda_hc,
    effective_lambda_sparse,
    pure_l1_record,
    steps_per_epoch_of,
)
from q3vl.whatb.colorimetry import chroma_hue, srgb_to_lab
from q3vl.whatb.criteria import (
    PREREGISTERED_KEYS,
    build_board,
    function_distance,
    path_quantities,
    required_criteria,
)
from q3vl.whatb.generator import SEG_COLOR_HIDDEN_DIM, CGLUTGenerator, SegColorProjection
from q3vl.whatb.glut import (
    EPS,
    GlutAux,
    GlutCarrier,
    GlutParams,
    glut_forward,
    glut_geometry,
    n_params_glut,
)
from q3vl.whatb.guards import (
    DegeneracyReport,
    DegeneracyThresholds,
    assert_transform_not_degenerate,
)
from q3vl.whatb.publish import assert_publishable, step_columns_for
from q3vl.whatb.queries import (
    BATCH_SAMPLES,
    COLORS_PER_STEP,
    QUERIES_PER_SAMPLE,
    QuerySampler,
    mining_ratio,
    uniform_grid,
)

__all__ = [
    "ARM",
    "EPR",
    "AXES",
    "ARM_CRITERIA",
    "P1_CRITERIA",
    "SHARE_CHOICES",
    "GLOBAL_AFFINE_CHOICES",
    "MU_INIT_CHOICES",
    "LINEARITY_TOL",
    "DEGENERATE_WEIGHT_TAU",
    "FROZEN",
    "AffineOnlyConfig",
    "AffineOnlyHead",
    "LossTerms",
    "loss_terms",
    "build_optimizer",
    "build_scheduler",
    "build_training_colors",
    "train_step",
    "assert_shared_geometry_identical",
    "assert_affine_linearity",
    "assert_not_degenerate",
    "step0_witness",
    "degenerate_weight_rate",
    "interp_ip_a",
    "interp_ip_b",
    "required_columns",
    "arm_criteria_columns",
    "build_arm_board",
    "publish_board",
    "run_setup",
]

#: the ``--arm`` value; ``EPR`` is what the board is keyed by (``criteria.ARM_AXES``)
ARM = "AFFONLY"
EPR = "EPR-025"
#: proposal section 3.5: this is a P1 arm (and the premise-provider for P2)
AXES: tuple[str, ...] = ("P1",)

#: P1 additions to the twelve frozen keys (HANDOFF section 4.H code block)
P1_CRITERIA: tuple[str, ...] = ("interp_grid", "path_len", "mono_rate", "oob_rate")

#: this arm's own pre-registered columns -- proposal section 3.6's assertions 1/2
#: and proposition 2's execution line.  Missing or ``n = 0`` -> no board.
ARM_CRITERIA: tuple[str, ...] = (
    "shared_geom_identical",
    "affine_linearity_maxdev",
    "degenerate_weight_rate",
)

SHARE_CHOICES: tuple[str, ...] = ("none", "geo", "geo_opacity")
GLOBAL_AFFINE_CHOICES: tuple[str, ...] = ("affine", "residual", "none")
MU_INIT_CHOICES: tuple[str, ...] = ("grid", "random")

#: proposal section 3.6 assertion 2 -- "< 1e-5, and it must run in fp32
#: (bf16 machine epsilon is ~7.8e-3, so 1e-5 fails there by construction)".
LINEARITY_TOL: float = 1e-5
#: proposition 2's execution line: ``Pr_x[sum_j p_j o_j < tau]``, tau = 1e-3
#: (proposal section 3.5-F / :`4.2` "退化权重率 Pr_x[Σ_j p_j o_j < 10^-3]").
DEGENERATE_WEIGHT_TAU: float = 1e-3

#: the eight frozen numbers, carried into ``run_setup.json`` so a run that
#: silently deviated is visible on the artifact rather than only in a log line.
FROZEN: dict[str, Any] = {
    "train_split": "train",
    "train_winner_confidence": "normal",
    "train_n": 93934,
    "batch_samples": BATCH_SAMPLES,
    "queries_per_sample": QUERIES_PER_SAMPLE,
    "colors_per_step": COLORS_PER_STEP,
    "steps_per_epoch": 2936,
    "epochs": 40,
    "total_steps": 117440,
    "clamp_default": "two",
    "headline_formation": "I_hat = (1 - a) * I + a * f_hat(I)",
    "preregistered_keys": list(PREREGISTERED_KEYS),
    "required_criteria_epr025": list(required_criteria(EPR)),
}
assert FROZEN["steps_per_epoch"] == math.ceil(FROZEN["train_n"] / BATCH_SAMPLES)
assert FROZEN["total_steps"] == FROZEN["steps_per_epoch"] * FROZEN["epochs"]


# --------------------------------------------------------------------------- #
# 1. configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AffineOnlyConfig:
    """Every knob of proposal section 3.4-(12), at the proposal's own defaults.

    The three ablation switches of section 4.1 and 4.4 are ``share`` (rows 1/2/3),
    ``global_affine`` (rows 4 / 4'), ``shared_lr_scale`` (row 5), ``cond_dim``
    (row 6), ``n_gauss`` (row 7), ``mu_init`` (row 8), ``lambda_sparse`` (row 9)
    and ``zero_init_heads`` (row 10).  Nothing else in this file is tunable: the
    frozen block owns the batch organisation, the step count and the clamp.
    """

    # --- the experiment variable ------------------------------------------
    #: ``geo_opacity`` = this arm (share mu, Sigma AND o -> proposition 1 holds);
    #: ``geo`` = GLUT's own Shared Geometry (o still generated -> premise fails);
    #: ``none`` = Full Generation (= the EPR-024 form), row 1 of the main table.
    share: str = "geo_opacity"
    #: ``affine`` = Eq.5; ``residual`` = row 4 (``+ x`` instead of ``Gx + g``);
    #: ``none`` = row 4' (neither).
    global_affine: str = "affine"

    # --- structure (CGLUT App A.2 / section 2.3) ---------------------------
    n_gauss: int = 48
    cond_dim: int = 64
    gen_width: int = 128
    hidden_dim: int = SEG_COLOR_HIDDEN_DIM
    zero_init_heads: bool = True
    mu_init: str = "grid"
    sigma_init: float = 0.15
    opacity_logit_init: float = 4.0

    # --- carrier ------------------------------------------------------------
    clamp: str = "two"
    eps: float = EPS

    # --- loss (GLUT section 4.1 + the frozen L_hc mask) --------------------
    lambda_hc: float = 10.0
    lambda_sparse: float = 0.001
    eps_c: float = 1e-3
    loss_level: int = 3

    # --- optimiser (GLUT section 4.1 / App A.1) ----------------------------
    base_lr: float = 1e-3
    shared_lr_scale: float = 0.1
    weight_decay: float = 0.0
    grad_clip: float = 0.0          # "原文未提 -> 不裁"
    optimizer: str = "adam"
    scheduler: str = "cosine"

    # --- schedule / batching (frozen block) --------------------------------
    batch_samples: int = BATCH_SAMPLES
    queries: int = QUERIES_PER_SAMPLE
    #: which training corpora the population is drawn from (``--data``); the
    #: measured n is passed in, never a literal (:mod:`q3vl.whatb.caliber`).
    data: str = "v2seg"
    train_n: int = 93934
    epochs: int = 40
    total_steps: int = 117440
    mining_start_epoch: int = 5
    mining_end_epoch: int = 20
    mining_r_start: float = 0.10
    mining_r_end: float = 0.40
    seed: int = 20260810

    # --- assertions ---------------------------------------------------------
    linearity_pairs: int = 64
    linearity_grid: int = 17
    linearity_tol: float = LINEARITY_TOL
    shared_geom_probes: int = 64
    degenerate_weight_tau: float = DEGENERATE_WEIGHT_TAU
    degeneracy: DegeneracyThresholds = DegeneracyThresholds()

    def __post_init__(self) -> None:
        if self.share not in SHARE_CHOICES:
            raise ValueError(f"--share must be one of {SHARE_CHOICES}, got {self.share!r}")
        if self.global_affine not in GLOBAL_AFFINE_CHOICES:
            raise ValueError(
                f"--global-affine must be one of {GLOBAL_AFFINE_CHOICES}, got {self.global_affine!r}")
        if self.mu_init not in MU_INIT_CHOICES:
            raise ValueError(f"--mu-init must be one of {MU_INIT_CHOICES}, got {self.mu_init!r}")
        if self.mu_init == "random" and self.share == "none":
            raise ValueError(
                "--mu-init random is the ablation of the SHARED mu table (row 8); "
                "under --share none there is no mu table, mu is generated by a head")
        if self.clamp not in ("two", "one"):
            raise ValueError(f"--clamp must be 'two' or 'one', got {self.clamp!r}")
        if self.loss_level not in (1, 2, 3, 4):
            raise ValueError(f"--loss-level must be 1..4, got {self.loss_level}")
        if self.data not in DATA_CHOICES:
            raise ValueError(f"--data must be one of {DATA_CHOICES}, got {self.data!r}")

    # ---- derived arithmetic (all of it printed into run_setup) ------------
    @property
    def lambda_hc_effective(self) -> float:
        """``arms/carrier.py:347``'s ladder rule, through the shared caliber."""
        return effective_lambda_hc(self.lambda_hc, self.loss_level)

    @property
    def lambda_sparse_effective(self) -> float:
        """``arms/carrier.py:351``'s ladder rule, through the shared caliber."""
        return effective_lambda_sparse(self.lambda_sparse, self.loss_level)

    @property
    def steps_per_epoch(self) -> int:
        """``ceil(n / B)`` on the MEASURED population (never a literal)."""
        return steps_per_epoch_of(self.train_n, self.batch_samples)

    @property
    def batch_split(self) -> str:
        """``"BxQ"`` -- the row of the ONE shared table this run is on.

        Derived, so the name and the pair cannot drift (``arms/carrier.py``
        asserts ``name == f"{b}x{q}"`` for every ``BATCH_SPLITS`` entry).
        """
        return f"{int(self.batch_samples)}x{int(self.queries)}"

    @property
    def step_matched_to_epr024(self) -> bool:
        return self.batch_split in FROZEN_BATCH_SPLITS

    @property
    def theta_gen_dim(self) -> int:
        """Generated slice of ``Theta``: ``22N+12`` / ``13N+12`` / ``12N+12``."""
        n = self.n_gauss
        return {"none": 22 * n + 12, "geo": 13 * n + 12, "geo_opacity": 12 * n + 12}[self.share]

    @property
    def n_shared(self) -> int:
        """``10N`` -- mu(3) + Cholesky(6) + opacity logit(1) per Gaussian."""
        return 0 if self.share == "none" else 10 * self.n_gauss

    @property
    def colors_per_step(self) -> int:
        return self.batch_samples * self.queries

    def to_dict(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "degeneracy"}
        d["degeneracy_thresholds"] = self.degeneracy.as_dict()
        d["theta_gen_dim"] = self.theta_gen_dim
        d["n_shared_params"] = self.n_shared
        d["n_params_glut_total"] = n_params_glut(self.n_gauss)
        d["colors_per_step"] = self.colors_per_step
        d["colours_per_step"] = self.colors_per_step
        d["batch_split"] = self.batch_split
        d["batch_split_step_matched_to_epr024"] = self.step_matched_to_epr024
        d["steps_per_epoch"] = self.steps_per_epoch
        d["lambda_hc_effective"] = self.lambda_hc_effective
        d["lambda_sparse_effective"] = self.lambda_sparse_effective
        d["lambda_mono_effective"] = 0.0
        return d


# --------------------------------------------------------------------------- #
# 2. the head
# --------------------------------------------------------------------------- #
class AffineOnlyHead(nn.Module):
    """``pi -> CGLUT generator -> GLUT carrier``, with mu/Sigma/o shared.

    ``z`` is the frozen VLM read-out at ``<seg_color>`` -- ``(B, 2560)``.
    :meth:`project` is ``pi`` (the CGLUT ``e_l`` slot), :meth:`theta` produces the
    :class:`~q3vl.whatb.glut.GlutParams`, :meth:`transform` evaluates ``f`` on
    colour queries.  Condition-space interpolation (``f^cond``) is defined on the
    **post-pi** vector ``u`` -- the generator is the non-linear part and the
    proposal's ``f^cond`` is ``f_{G((1-a)z_a + a z_b)}`` with ``G`` the generator.

    ``--share geo_opacity`` (the arm) makes ``w_i(x)`` condition-independent, so
    :meth:`geometry_terms` returns the once-per-step
    ``(precision, logdet, opacity, degenerate)`` and every sample reuses it.  That
    reuse *is* the computational content of proposition 1.
    """

    def __init__(self, cfg: AffineOnlyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.proj = SegColorProjection(in_dim=cfg.hidden_dim, cond_dim=cfg.cond_dim)
        self.generator = CGLUTGenerator(
            cond_dim=cfg.cond_dim,
            hidden=cfg.gen_width,
            n_gauss=cfg.n_gauss,
            mode="full" if cfg.share == "none" else "affine_only",
            # row 4 (``+ x``) needs ``M_i = dM_i`` so step 0 is still the identity
            # (proposal section 4.1 row 4, parenthesis); rows "affine" / "none"
            # keep ``M_i = I + dM_i`` (section 3.4-(4)).
            m_residual=(cfg.global_affine != "residual"),
            zero_init_last=cfg.zero_init_heads,
            shared_sigma=cfg.sigma_init,
            shared_opacity_logit=cfg.opacity_logit_init,
            eps=cfg.eps,
        )
        # --share geo = GLUT's own Shared Geometry: {mu, Sigma} shared, o still
        # generated (section 3.2 of the paper, quoted at proposal :168).  The
        # opacity head is App A.2's "2 layers with adjusted output dimensions".
        self.head_opacity: nn.Module | None = None
        if cfg.share == "geo":
            h = cfg.gen_width
            self.head_opacity = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, cfg.n_gauss))
            if cfg.zero_init_heads:
                nn.init.zeros_(self.head_opacity[-1].weight)
                nn.init.zeros_(self.head_opacity[-1].bias)
        if cfg.mu_init == "random":
            self._randomise_mu()
        self.carrier = GlutCarrier(
            clamp=cfg.clamp, residual=(cfg.global_affine != "none"), eps=cfg.eps
        )
        # the only constants: no bare torch.tensor(...) in any forward below.
        self.register_buffer("eye3", torch.eye(3), persistent=False)
        self.register_buffer("zero3", torch.zeros(3), persistent=False)

    # -- construction helpers ------------------------------------------------
    def _randomise_mu(self) -> None:
        """Ablation row 8: GLUT B.4.6's random-mean control, private generator."""
        sg = self.generator.shared_geometry
        assert sg is not None
        gen = torch.Generator().manual_seed(int(self.cfg.seed))
        with torch.no_grad():
            sg.mu.copy_(torch.rand(sg.mu.shape, generator=gen))

    @property
    def shared_geometry(self):
        return self.generator.shared_geometry

    @property
    def n_gauss(self) -> int:
        return self.cfg.n_gauss

    # -- condition -> parameters --------------------------------------------
    def project(self, z: Tensor) -> Tensor:
        """``pi(z)``: ``(B, 2560) -> (B, d)``.  ``z`` is cast by the projection."""
        return self.proj(z)

    def theta_from_u(self, u: Tensor) -> GlutParams:
        """``G_theta(u)`` with this arm's two structural overrides applied."""
        params = self.generator(u)
        if self.head_opacity is not None:
            # --share geo only.  One extra encode of a (B, d) vector through three
            # 128-wide layers; the alternative is duplicating the generator's own
            # (tested) forward here, which is the thing that drifts.
            h = self.generator.encode(u)
            params = replace(params, opacity_logit=self.head_opacity(h))
        return self._apply_global_affine(params)

    def theta(self, z: Tensor) -> GlutParams:
        """``G_theta(pi(z))`` -- the full condition path."""
        return self.theta_from_u(self.project(z))

    def _apply_global_affine(self, params: GlutParams) -> GlutParams:
        """Rows 4 / 4' of section 4.1, expressed on ``{G, g}``.

        ``affine``   Eq.5, ``G`` and ``g`` generated (nothing to do);
        ``residual`` ``f = sum_i w_i f_i(x) + x``   -> ``G = I``, ``g = 0``, fixed;
        ``none``     ``f = sum_i w_i f_i(x)``       -> the carrier drops the term
                     (``residual=False``); ``G``/``g`` are zeroed so the recorded
                     parameters cannot be mistaken for a live global branch.
        """
        mode = self.cfg.global_affine
        if mode == "affine":
            return params
        b = params.batch_size
        ref = params.g_matrix
        eye = self.eye3.to(device=ref.device, dtype=ref.dtype)
        g_mat = eye.expand(b, 3, 3) if mode == "residual" else torch.zeros_like(ref)
        g_bias = self.zero3.to(device=ref.device, dtype=ref.dtype).expand(b, 3)
        return replace(params, g_matrix=g_mat, g_bias=g_bias)

    def geometry_terms(self) -> tuple[Tensor, Tensor, Tensor, Tensor] | None:
        """The once-per-step ``(precision, logdet, opacity, degenerate)``.

        ``None`` unless ``--share geo_opacity``: under ``geo`` the opacity is
        condition-dependent, so the carrier must recompute per sample (passing
        the shared tuple there would silently substitute the shared ``o``).
        """
        if self.cfg.share != "geo_opacity":
            return None
        return self.generator.shared_geometry_terms()

    # -- forward -------------------------------------------------------------
    def transform(
        self,
        z: Tensor | None,
        x: Tensor,
        *,
        params: GlutParams | None = None,
        geometry: tuple[Tensor, Tensor, Tensor, Tensor] | None = None,
        clamp: str | None = None,
        return_aux: bool = False,
        point_chunk: int | None = None,
    ) -> Tensor | tuple[Tensor, GlutAux]:
        """``f_theta(x)``.  ``x`` is ``(B, P, 3)`` or ``(P, 3)`` (shared queries)."""
        if params is None:
            if z is None:
                raise ValueError("transform() needs either z= or params=")
            params = self.theta(z)
        geom = self.geometry_terms() if geometry is None else geometry
        return glut_forward(
            x,
            params,
            clamp=self.carrier.clamp_mode if clamp is None else clamp,
            residual=self.carrier.residual,
            eps=self.carrier.eps,
            point_chunk=point_chunk,
            return_aux=return_aux,
            _geometry=geom,
        )

    def forward(self, z: Tensor, x: Tensor, **kw) -> Tensor | tuple[Tensor, GlutAux]:  # noqa: D102
        return self.transform(z, x, **kw)

    def transform_image(self, z: Tensor, img: Tensor, *, point_chunk: int | None = None) -> Tensor:
        """``f̂(I)`` for one ``(3, H, W)`` image in [0,1] -> ``(3, H, W)``.

        ``z`` is that image's condition, ``(2560,)`` or ``(1, 2560)``.  The image
        is moved onto the head's own (device, dtype) first -- the where-side
        MATTE arm died on exactly this (a CPU image tensor reaching an autocast
        region), so the cast is explicit and unconditional.
        """
        if img.dim() != 3 or img.shape[0] != 3:
            raise ValueError(f"img must be (3, H, W), got {tuple(img.shape)}")
        ref = self.proj.proj.weight
        z2 = z.reshape(1, -1).to(device=ref.device, dtype=ref.dtype)
        x = img.permute(1, 2, 0).unsqueeze(0).to(device=ref.device, dtype=ref.dtype)
        y = self.transform(z2, x, point_chunk=point_chunk)
        assert isinstance(y, Tensor)
        return y[0].permute(2, 0, 1).to(dtype=img.dtype)

    # -- bookkeeping ---------------------------------------------------------
    @property
    def config(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "arm": ARM,
            "epr": EPR,
            **self.cfg.to_dict(),
            "carrier": self.carrier.config,
            "generator": self.generator.config,
            "n_params_pi": sum(p.numel() for p in self.proj.parameters()),
            "n_params_generator": sum(p.numel() for p in self.generator.parameters()),
            "n_params_trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
        if self.head_opacity is not None:
            cfg["n_params_head_opacity"] = sum(p.numel() for p in self.head_opacity.parameters())
        return cfg

    def param_groups(self, base_lr: float | None = None, *, scale: float | None = None
                     ) -> list[dict[str, Any]]:
        """Two groups: everything at ``base_lr``, the shared tables at ``0.1x``.

        App A.1 puts "style embeddings and shared geometry parameters" on 0.1x
        the base lr; folding ``o`` into that group is this arm's NOVEL choice
        (``--shared-geom-lr-scale 1.0`` is ablation row 5).  ``pi`` is NOT in the
        group: it stands where CGLUT's learnable ``e_l`` stood but it is part of
        the generator path (section 3.3, "NOVEL 归属").
        """
        lr = self.cfg.base_lr if base_lr is None else float(base_lr)
        s = self.cfg.shared_lr_scale if scale is None else float(scale)
        sg = self.shared_geometry
        if sg is None:
            return [{"params": list(self.parameters()), "lr": lr, "name": "generator"}]
        geo = sg.parameters_list()
        geo_ids = {id(p) for p in geo}
        rest = [p for p in self.parameters() if id(p) not in geo_ids]
        # section 3.4-(8): "启动时断言「共享组的参数张量数 == 4 且总元素数 == 10N」"
        n_elem = sum(p.numel() for p in geo)
        if len(geo) != 4 or n_elem != 10 * self.cfg.n_gauss:
            raise AssertionError(
                f"shared-geometry lr group must hold 4 tensors totalling 10N = "
                f"{10 * self.cfg.n_gauss} elements; got {len(geo)} tensors / {n_elem} "
                "elements.  The 0.1x rule would then be applied to the wrong set.")
        return [
            {"params": rest, "lr": lr, "name": "generator"},
            {"params": geo, "lr": lr * s, "name": "shared_geometry"},
        ]


# --------------------------------------------------------------------------- #
# 3. loss -- GLUT Eq.6-8 verbatim, with the frozen L_hc mask
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LossTerms:
    """The three terms plus the columns ``steps.jsonl`` must carry."""

    total: Tensor
    l_rec: Tensor
    l_hc: Tensor
    l_sparse: Tensor
    n_hc_masked: int
    n_colors: int

    def as_row(self) -> dict[str, Any]:
        return {
            "L_total": float(self.total.detach()),
            "L_rec": float(self.l_rec.detach()),
            "L_hc": float(self.l_hc.detach()),
            "L_sparse": float(self.l_sparse.detach()),
            "n_hc_masked": int(self.n_hc_masked),
            "n_colors": int(self.n_colors),
        }


def loss_terms(
    y_hat: Tensor,
    y: Tensor,
    opacity: Tensor,
    *,
    lambda_hc: float = 10.0,
    lambda_sparse: float = 0.001,
    eps_c: float = 1e-3,
    eps: float = EPS,
    hc_mask: bool = True,
) -> LossTerms:
    """``L = ||ŷ-y||_1 + 10 * L_hc + 0.001 * R_sparse`` (GLUT Eq.6-8, section 4.1).

    ``L_hc`` follows the frozen block: ``h = (a, b) / max(C, eps_c)`` **and** the
    whole term multiplied by the hard mask ``1[C >= eps_c]``, ``eps_c = 1e-3``,
    with the masked-out point count logged as ``n_hc_masked``.  ``C`` and ``h``
    are taken from the **target** ``y`` (Eq.7: "C = sqrt(a^2+b^2), CIELab, 取自 y").

    ``R_sparse`` (Eq.8) reads the opacities; in this arm they are one global
    table, so the term is the same at every step of an epoch and does not depend
    on the batch -- a NOVEL consequence of the sharing, noted in section 3.2, the
    formula itself unchanged.  Ablation row 9 sets ``lambda_sparse = 0``.

    Reduction: ``mean`` over colours and channels for ``L_rec`` and over colours
    for ``L_hc``; the paper writes the norms without a reduction and every scale
    it quotes (lambda_hc = 10) is only meaningful against one, so it is recorded in
    ``run_setup`` as ``l_rec_reduction: mean``.
    """
    if y_hat.shape != y.shape:
        raise ValueError(f"y_hat {tuple(y_hat.shape)} != y {tuple(y.shape)}")
    yv = y.to(device=y_hat.device, dtype=y_hat.dtype)
    l_rec = (y_hat - yv).abs().mean()

    lab_hat = srgb_to_lab(y_hat)
    lab = srgb_to_lab(yv)
    c, h, valid = chroma_hue(lab, eps_c)
    _, h_hat, _ = chroma_hue(lab_hat, eps_c)
    term = c * (1.0 - (h_hat * h).sum(-1))
    if hc_mask:
        term = term * valid.to(term.dtype)
    l_hc = term.mean()
    n_hc_masked = int((~valid).sum())

    o = opacity.to(device=y_hat.device, dtype=y_hat.dtype)
    r_sparse = -(o * torch.log(o + eps) + (1.0 - o) * torch.log(1.0 - o + eps)).mean()

    total = l_rec + float(lambda_hc) * l_hc + float(lambda_sparse) * r_sparse
    return LossTerms(total, l_rec, l_hc, r_sparse, n_hc_masked, int(y.numel() // 3))


# --------------------------------------------------------------------------- #
# 4. optimiser / schedule
# --------------------------------------------------------------------------- #
def build_optimizer(head: AffineOnlyHead, cfg: AffineOnlyConfig | None = None
                    ) -> torch.optim.Optimizer:
    """Adam (NOT AdamW -- GLUT section 4.1 says Adam), ``wd = 0``, two lr groups."""
    cfg = cfg or head.cfg
    if cfg.optimizer != "adam":
        raise ValueError(
            f"GLUT section 4.1 says 'optimized using the Adam optimizer'; "
            f"{cfg.optimizer!r} is a deviation that has to be registered, not a flag")
    groups = head.param_groups(cfg.base_lr, scale=cfg.shared_lr_scale)
    return torch.optim.Adam(groups, lr=cfg.base_lr, weight_decay=cfg.weight_decay)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: AffineOnlyConfig,
                    total_steps: int | None = None):
    """Cosine annealing over the whole run (GLUT section 4.1, no warm-up)."""
    if cfg.scheduler != "cosine":
        raise ValueError(f"scheduler {cfg.scheduler!r} is not the paper's cosine annealing")
    t_max = int(total_steps or cfg.total_steps)
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=0.0)


# --------------------------------------------------------------------------- #
# 5. one training step (frozen batch organisation + ruling 11.1-4 mining)
# --------------------------------------------------------------------------- #
def build_training_colors(
    head: AffineOnlyHead,
    z: Tensor,
    sampler: QuerySampler,
    target_fn: Callable[[Tensor], Tensor],
    ratio: float,
    *,
    device: Any = "cpu",
    dtype: torch.dtype = torch.float32,
    geometry: tuple[Tensor, Tensor, Tensor, Tensor] | None = None,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """Ruling 11.1-4: within-batch top-``r`` resampling, no cross-step state.

    Draw ``B x Q`` uniform colours, score them with a ``no_grad`` forward, keep
    the ``r * Q`` hardest **per sample** (so the batch stays rectangular and the
    frozen ``B x Q = 8192`` holds), and top up with ``(1 - r) * Q`` fresh uniform
    colours.  The ``topk`` runs on the tensor's device: CPU and CUDA break ties
    differently and the where side lost 0.296 of an IoU to exactly that.
    """
    b = int(z.shape[0])
    q = sampler.q
    x = sampler.sample(b, q, device=device, dtype=dtype)
    y = target_fn(x)
    k = int(round(float(ratio) * q))
    k = max(0, min(k, q))
    if k == 0:
        return x, y, {"mining_ratio": float(ratio), "n_mined": 0, "n_fresh": q}

    with torch.no_grad():
        y_hat = head.transform(z, x, geometry=geometry)
        assert isinstance(y_hat, Tensor)
        err = (y_hat - y).abs().mean(dim=-1)                       # (B, Q)
    idx = torch.topk(err, k, dim=1, largest=True, sorted=False).indices
    gather = idx.unsqueeze(-1).expand(b, k, 3)
    x_hard = torch.gather(x, 1, gather)
    y_hard = torch.gather(y, 1, gather)

    x_new = sampler.sample(b, q - k, device=device, dtype=dtype)
    y_new = target_fn(x_new)
    return (
        torch.cat([x_hard, x_new], dim=1),
        torch.cat([y_hard, y_new], dim=1),
        {"mining_ratio": float(ratio), "n_mined": int(b * k), "n_fresh": int(b * (q - k))},
    )


def train_step(
    head: AffineOnlyHead,
    optimizer: torch.optim.Optimizer,
    z: Tensor,
    sampler: QuerySampler,
    target_fn: Callable[[Tensor], Tensor],
    *,
    epoch: float,
    lut_ids: Sequence[str] = (),
    scheduler: Any = None,
    cfg: AffineOnlyConfig | None = None,
    device: Any = "cpu",
    dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    """One optimiser step; returns the ``steps.jsonl`` row.

    The row carries every pre-registered column of HANDOFF section 4.H
    (``L_rec`` / ``L_hc`` / ``L_sparse`` / ``n_colors`` / ``n_luts_in_batch`` /
    ``mining_ratio`` / ``n_hc_masked``) plus this arm's diagnostics.  The
    publication gate reads the FIRST such row, three tiers deep.
    """
    cfg = cfg or head.cfg
    head.train()
    r = mining_ratio(epoch, start_epoch=cfg.mining_start_epoch, end_epoch=cfg.mining_end_epoch,
                     r_start=cfg.mining_r_start, r_end=cfg.mining_r_end)
    # geometry=None: the mining probe recomputes it inside its own ``no_grad``
    # block, so no graph is built for a forward whose only purpose is a topk.
    x, y, mine = build_training_colors(head, z, sampler, target_fn, r, device=device,
                                       dtype=dtype)
    geometry = head.geometry_terms()
    y_hat, aux = head.transform(z, x, geometry=geometry, return_aux=True)
    # the ladder gates the two optional weights (carrier.py:347/:351); at
    # --loss-level 3 (the default) both are cfg's own value, bit for bit.
    terms = loss_terms(y_hat, y, aux.opacity, lambda_hc=cfg.lambda_hc_effective,
                       lambda_sparse=cfg.lambda_sparse_effective,
                       eps_c=cfg.eps_c, eps=cfg.eps)

    optimizer.zero_grad(set_to_none=True)
    terms.total.backward()
    if cfg.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(head.parameters(), cfg.grad_clip)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    row = {
        **terms.as_row(),
        "n_luts_in_batch": len(set(lut_ids)) if lut_ids else int(z.shape[0]),
        **mine,
        "epoch": float(epoch),
        "lr": float(optimizer.param_groups[0]["lr"]),
        "n_degenerate_precision": int(aux.degenerate_precision.sum()),
        "degenerate_weight_rate": float(
            aux.degenerate_weight_mask(cfg.degenerate_weight_tau).to(y_hat.dtype).mean()),
    }
    if len(optimizer.param_groups) > 1:
        row["lr_shared_geometry"] = float(optimizer.param_groups[1]["lr"])
    return row


# --------------------------------------------------------------------------- #
# 6. the arm's three run-time assertions (section 3.6) + the degeneracy guard
# --------------------------------------------------------------------------- #
def assert_shared_geometry_identical(
    head: AffineOnlyHead, z: Tensor, *, n_probe: int | None = None, raise_on_fail: bool = True
) -> dict[str, Any]:
    """Assertion 1: ``(precision, logdet, o)`` are **bit-identical** across conditions.

    64 random conditions, one full forward each, ``torch.equal`` (not
    ``allclose``).  A difference means mu / Sigma / o are still on the condition
    path, i.e. the arm is not the arm.
    """
    n = int(n_probe or head.cfg.shared_geom_probes)
    zz = z[:n]
    if zz.shape[0] < 2:
        raise ValueError("assertion 1 needs at least two conditions")
    ref: tuple[Tensor, Tensor, Tensor] | None = None
    n_diff = 0
    max_dev = 0.0
    for i in range(zz.shape[0]):
        p = head.theta(zz[i : i + 1])
        prec, logdet, opac, _ = glut_geometry(p.chol_diag, p.chol_off, p.opacity_logit,
                                              eps=head.cfg.eps)
        cur = (prec.detach(), logdet.detach(), opac.detach())
        if ref is None:
            ref = cur
            continue
        if not all(torch.equal(a, b) for a, b in zip(ref, cur)):
            n_diff += 1
            max_dev = max(max_dev, *(float((a - b).abs().max()) for a, b in zip(ref, cur)))
    ok = n_diff == 0
    out = {
        "n": int(zz.shape[0]),
        "value": 1.0 if ok else 0.0,
        "identical": ok,
        "n_conditions_differing": n_diff,
        "max_abs_dev": max_dev,
        "share": head.cfg.share,
        "quantity": "torch.equal over (precision, logdet, opacity) across conditions",
    }
    if raise_on_fail and not ok:
        raise AssertionError(
            f"[{ARM}] assertion 1 FAILED: {n_diff}/{zz.shape[0] - 1} conditions produced a "
            f"different geometry (max |dev| = {max_dev:.3e}).  mu/Sigma/o are still "
            f"condition-dependent under --share {head.cfg.share}; proposition 1's premise "
            "does not hold and the board may not be published.")
    return out


def assert_affine_linearity(
    head: AffineOnlyHead,
    z: Tensor,
    *,
    alphas: Sequence[float] = (0.1, 0.3, 0.5, 0.7, 0.9),
    n_pairs: int | None = None,
    grid_n: int | None = None,
    tol: float | None = None,
    raise_on_fail: bool = True,
) -> dict[str, Any]:
    """Assertion 2: the numerical form of proposition 1 (parameter-space linearity).

    ``max_x | f_{(1-a)th_a + a th_b}(x) - ((1-a) f_{th_a}(x) + a f_{th_b}(x)) | < 1e-5``
    on the 17^3 grid, **before the clamp** and **in fp32** (bf16 machine epsilon
    is ~7.8e-3, so the tolerance is unreachable there by construction).  The
    interpolation is in ``theta_gen`` -- parameter space, not condition space.
    """
    cfg = head.cfg
    n_pairs = int(n_pairs or cfg.linearity_pairs)
    grid_n = int(grid_n or cfg.linearity_grid)
    tol = float(cfg.linearity_tol if tol is None else tol)
    if z.shape[0] < 2 * n_pairs:
        n_pairs = z.shape[0] // 2
    if n_pairs < 1:
        raise ValueError("assertion 2 needs at least two conditions")

    ref = head.proj.proj.weight
    za = z[: 2 * n_pairs : 2].to(device=ref.device, dtype=torch.float32)
    zb = z[1 : 2 * n_pairs : 2].to(device=ref.device, dtype=torch.float32)
    x = uniform_grid(grid_n, dtype=torch.float32, device=ref.device)

    with torch.no_grad():
        geom = head.geometry_terms()
        pa, pb = head.theta(za), head.theta(zb)
        fa = head.transform(None, x, params=pa, geometry=geom, clamp="none")
        fb = head.transform(None, x, params=pb, geometry=geom, clamp="none")
        assert isinstance(fa, Tensor) and isinstance(fb, Tensor)
        va, vb = pa.affine_flat(), pb.affine_flat()
        per_alpha: dict[str, float] = {}
        worst = 0.0
        for a in alphas:
            mix = pa.with_affine_flat((1.0 - a) * va + a * vb)
            f_mix = head.transform(None, x, params=mix, geometry=geom, clamp="none")
            assert isinstance(f_mix, Tensor)
            dev = float((f_mix - ((1.0 - a) * fa + a * fb)).abs().max())
            per_alpha[f"alpha_{a}"] = dev
            worst = max(worst, dev)

    ok = worst < tol
    out = {
        "n": int(n_pairs),
        "value": worst,
        "max_abs_dev": worst,
        "tol": tol,
        "per_alpha": per_alpha,
        "grid_n": grid_n,
        "dtype": "float32",
        "clamp": "none (pre-clamp, proposition 1 is stated pre-clamp)",
        "quantity": "max_x |f_{(1-a)th_a + a th_b}(x) - ((1-a) f_a(x) + a f_b(x))|",
        "passed": ok,
    }
    if raise_on_fail and not ok:
        raise AssertionError(
            f"[{ARM}] assertion 2 FAILED: max |f^par - f^fun| = {worst:.3e} >= {tol:.1e} "
            f"on the {grid_n}^3 grid.  Proposition 1's premise (mu/Sigma/o condition-"
            "independent) is not satisfied, so the sharing is not wired; refusing the board.")
    return out


def assert_not_degenerate(
    head: AffineOnlyHead,
    z: Tensor,
    x: Tensor | None = None,
    *,
    thresholds: DegeneracyThresholds | None = None,
    where: str = "quick_eval",
    exit_process: bool = True,
) -> DegeneracyReport:
    """The batch-wide guard: run it at the FIRST quick eval, never later only.

    Three conditions, any one of which stops the process with ``SystemExit(2)``:
    the transform is flat across query colours, it is the identity, or it is the
    same for every sample.  PRND and CONDINST burned 2.6 GPU-hours on a constant
    field because this check did not exist on the where side.
    """
    cfg = head.cfg
    ref = head.proj.proj.weight
    if x is None:
        x = uniform_grid(9, dtype=torch.float32, device=ref.device)
    xr = x.to(device=ref.device, dtype=torch.float32)
    with torch.no_grad():
        y = head.transform(z.to(device=ref.device, dtype=torch.float32), xr)
    assert isinstance(y, Tensor)
    return assert_transform_not_degenerate(
        y, xr,
        thresholds=thresholds or cfg.degeneracy,
        where=where,
        exit_process=exit_process,
        extra={"arm": ARM, "share": cfg.share, "n_gauss": cfg.n_gauss},
    )


def step0_witness(head: AffineOnlyHead, z: Tensor, *, grid_n: int = 17) -> dict[str, Any]:
    """``step0_maxabs_f_minus_id`` -- ruling 11.1-1's witness column.

    "各臂 step0 初始化按各自参考工作忠实移植 ... step0 不同不影响步数匹配下的
    headline 比较，只登记见证列 (EPR-024:612 的 step0_maxabs_f_minus_id, 17^3
    网格上 max|f_theta(x) - x|)".  With the zero-initialised heads this arm is at
    ``(1 - delta(x)) x``, so the witness is the size of proposition 2's ``delta``
    in fp32, not a training signal.  Recorded in ``run_setup.json`` before step 1.
    """
    ref = head.proj.proj.weight
    x = uniform_grid(grid_n, dtype=torch.float32, device=ref.device)
    with torch.no_grad():
        y = head.transform(z.to(device=ref.device, dtype=torch.float32), x)
    assert isinstance(y, Tensor)
    return {
        "step0_maxabs_f_minus_id": float((y - x.unsqueeze(0)).abs().max()),
        "grid_n": grid_n,
        "n_conditions": int(z.shape[0]),
        "zero_init_heads": head.cfg.zero_init_heads,
        "m_residual": head.generator.m_residual,
        "quantity": "max_x |f_theta(x) - x| on the uniform sRGB grid, before any step",
    }


def degenerate_weight_rate(head: AffineOnlyHead, z: Tensor, x: Tensor | None = None, *,
                           tau: float | None = None) -> dict[str, Any]:
    """Proposition 2's execution line: ``Pr_x[sum_j p_j(x) o_j < tau]``, tau = 1e-3.

    Non-zero means some region of the gamut is covered by no Gaussian, where
    ``f(x) = (1 - delta(x)) x`` with a non-negligible ``delta``.  Every board
    must carry the column.
    """
    cfg = head.cfg
    tau = float(cfg.degenerate_weight_tau if tau is None else tau)
    ref = head.proj.proj.weight
    if x is None:
        x = uniform_grid(17, dtype=torch.float32, device=ref.device)
    xr = x.to(device=ref.device, dtype=torch.float32)
    with torch.no_grad():
        _, aux = head.transform(z.to(device=ref.device, dtype=torch.float32), xr,
                                return_aux=True)
    mask = aux.degenerate_weight_mask(tau)
    rate = mask.to(torch.float32).mean()
    return {
        "n": int(z.shape[0]),
        "value": float(rate),
        "mean": float(rate),
        "tau": tau,
        "min_influence_sum": float(aux.influence_sum.min()),
        "n_queries": int(xr.shape[0]),
        "quantity": "Pr_x[sum_j p_j(x) o_j < tau] (pre-eps influence sum)",
    }


# --------------------------------------------------------------------------- #
# 7. P1 interpolation protocols (criteria section 4.F)
# --------------------------------------------------------------------------- #
def _lut_values(bank, lut_id: str, x: Tensor) -> Tensor:
    return bank.apply(x, lut_id)


def interp_ip_a(
    head: AffineOnlyHead,
    pairs: Sequence[tuple[Tensor, str, Tensor, str]],
    bank,
    *,
    alphas: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    grid_n: int = 17,
) -> dict[str, Any]:
    """IP-A (with GT): the four columns section 4.F requires, per alpha.

    ``pairs`` is ``(z_a, lut_a, z_b, lut_b)`` for two samples of the same source
    image.  GT is the function-space blend ``(1-a) L_a + a L_b`` on the 17^3 grid
    (pointwise identical to blending in image space).  Columns:

    ``f_cond``    the arm's condition-space path ``f_{G((1-a)u_a + a u_b)}``
    ``f_par``     the parameter-space path ``f_{(1-a)th_a + a th_b}``
    ``out_mix``   the trivial output-blend column ``(1-a) f̂_a + a f̂_b`` -- section
                  4.F is explicit that an interpolation claim without this column
                  is void, because by definition it beats the conditional path
    ``endpoint``  ``dE00(f̂_a, L_a)`` / ``dE00(f̂_b, L_b)``

    ``assertion 3`` lives here: under this arm ``f_par`` and ``out_mix`` must be
    the same numbers (proposition 1); the realised maximum difference is
    reported as ``assertion3_maxdev``.  It is measured **pre-clamp** -- the
    reported columns are the clamped values ``ŷ`` (that is what the headline
    metric consumes), and the clamp is exactly what breaks the identity at the
    gamut boundary, so comparing the two clamped columns would test the clamp
    rather than the linearity.
    """
    ref = head.proj.proj.weight
    x = uniform_grid(grid_n, dtype=torch.float32, device=ref.device)
    rows: dict[str, list[list[float]]] = {"f_cond": [], "f_par": [], "out_mix": []}
    endpoints: list[tuple[float, float]] = []
    a3_dev = 0.0
    with torch.no_grad():
        geom = head.geometry_terms()
        for z_a, lut_a, z_b, lut_b in pairs:
            ua = head.project(z_a.reshape(1, -1).to(device=ref.device, dtype=torch.float32))
            ub = head.project(z_b.reshape(1, -1).to(device=ref.device, dtype=torch.float32))
            pa, pb = head.theta_from_u(ua), head.theta_from_u(ub)
            fa = head.transform(None, x, params=pa, geometry=geom)
            fb = head.transform(None, x, params=pb, geometry=geom)
            fa_raw = head.transform(None, x, params=pa, geometry=geom, clamp="none")
            fb_raw = head.transform(None, x, params=pb, geometry=geom, clamp="none")
            assert isinstance(fa, Tensor) and isinstance(fb, Tensor)
            la = _lut_values(bank, lut_a, x)
            lb = _lut_values(bank, lut_b, x)
            endpoints.append((float(function_distance(fa[0], la)),
                              float(function_distance(fb[0], lb))))
            va, vb = pa.affine_flat(), pb.affine_flat()
            r_cond, r_par, r_mix = [], [], []
            for a in alphas:
                gt = (1.0 - a) * la + a * lb
                f_cond = head.transform(None, x, params=head.theta_from_u((1.0 - a) * ua + a * ub),
                                        geometry=geom)
                p_mix = pa.with_affine_flat((1.0 - a) * va + a * vb)
                f_par = head.transform(None, x, params=p_mix, geometry=geom)
                f_par_raw = head.transform(None, x, params=p_mix, geometry=geom, clamp="none")
                assert isinstance(f_cond, Tensor) and isinstance(f_par, Tensor)
                out_mix = (1.0 - a) * fa + a * fb
                out_mix_raw = (1.0 - a) * fa_raw + a * fb_raw
                a3_dev = max(a3_dev, float((f_par_raw - out_mix_raw).abs().max()))
                r_cond.append(float(function_distance(f_cond[0], gt)))
                r_par.append(float(function_distance(f_par[0], gt)))
                r_mix.append(float(function_distance(out_mix[0], gt)))
            rows["f_cond"].append(r_cond)
            rows["f_par"].append(r_par)
            rows["out_mix"].append(r_mix)

    def _mean(rs: list[list[float]]) -> list[float]:
        if not rs:
            return []
        return [float(sum(r[i] for r in rs) / len(rs)) for i in range(len(alphas))]

    per_alpha = {k: _mean(v) for k, v in rows.items()}
    flat = [v for r in rows["f_cond"] for v in r]
    return {
        "n": len(pairs),
        "value": float(sum(flat) / len(flat)) if flat else None,
        "mean": float(sum(flat) / len(flat)) if flat else None,
        "alphas": list(alphas),
        "per_alpha": per_alpha,
        "endpoint_a": [e[0] for e in endpoints],
        "endpoint_b": [e[1] for e in endpoints],
        "assertion3_maxdev": a3_dev,
        "assertion3_tol": LINEARITY_TOL,
        "assertion3_passed": a3_dev < LINEARITY_TOL,
        "grid_n": grid_n,
        "metric": "dE00 on the uniform sRGB grid",
        "quantity": "IP-A dE00(f_cond_alpha, (1-a) L_a + a L_b), mean over alphas and pairs",
    }


def interp_ip_b(
    head: AffineOnlyHead,
    pairs: Sequence[tuple[Tensor, Tensor]],
    *,
    k_steps: int = 20,
    grid_n: int = 17,
) -> dict[str, dict[str, Any]]:
    """IP-B (no GT): the six path quantities, each next to its trivial floor.

    The path is the **condition-space** one (``u_alpha = (1-a) u_a + a u_b``,
    CGLUT's own blending convention), sampled at ``alpha_k = k / K`` with K = 20.
    Returns one board column per quantity; ``jump_max`` is ``K * max_k delta_k``
    with **no percentile trimming** (StyleGAN's official PPL trims 1%/99% and
    hides exactly this).
    """
    ref = head.proj.proj.weight
    x = uniform_grid(grid_n, dtype=torch.float32, device=ref.device)
    per_path: list[dict[str, Any]] = []
    with torch.no_grad():
        geom = head.geometry_terms()
        for z_a, z_b in pairs:
            ua = head.project(z_a.reshape(1, -1).to(device=ref.device, dtype=torch.float32))
            ub = head.project(z_b.reshape(1, -1).to(device=ref.device, dtype=torch.float32))
            vals = []
            for k in range(k_steps + 1):
                a = k / float(k_steps)
                f = head.transform(None, x, params=head.theta_from_u((1.0 - a) * ua + a * ub),
                                   geometry=geom, clamp="none")
                assert isinstance(f, Tensor)
                vals.append(f[0])
            per_path.append(path_quantities(torch.stack(vals, dim=0)))

    def _col(key: str, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        xs = [p[key] for p in per_path if p.get(key) is not None]
        col: dict[str, Any] = {
            "n": len(xs),
            "value": float(sum(xs) / len(xs)) if xs else None,
            "mean": float(sum(xs) / len(xs)) if xs else None,
            "per_path": xs,
            "k_steps": k_steps,
        }
        col.update(dict(extra or {}))
        return col

    return {
        "path_len": _col("path_len", {"quantity": "sum_k D_X(f_k, f_k+1); collapse solution = 0"}),
        "chord": _col("chord"),
        "rho": _col("rho"),
        "sigma_bar": _col("sigma_bar", {"note": "collapse and pure linear fade both give 0"}),
        "jump_max": _col("jump_max", {"note": "K * max_k delta_k, no percentile trimming"}),
        "mono_rate": _col("mono_rate", {"random_floor": 0.5}),
        "oob_rate": _col("oob_rate", {"note": "pre-clamp out-of-gamut rate A(alpha)"}),
    }


# --------------------------------------------------------------------------- #
# 8. board wiring
# --------------------------------------------------------------------------- #
def required_columns() -> list[str]:
    """The twelve frozen keys + the P1 four + this arm's three (section 3.6)."""
    return sorted(set(required_criteria(EPR)) | set(ARM_CRITERIA))


def arm_criteria_columns(
    *,
    interp_a: Mapping[str, Any],
    interp_b: Mapping[str, Mapping[str, Any]],
    shared_geom: Mapping[str, Any],
    linearity: Mapping[str, Any],
    degenerate: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Assemble every column ``build_board`` does not own itself."""
    cols: dict[str, dict[str, Any]] = {
        "interp_grid": dict(interp_a),
        "shared_geom_identical": dict(shared_geom),
        "affine_linearity_maxdev": dict(linearity),
        "degenerate_weight_rate": dict(degenerate),
    }
    for key, col in interp_b.items():
        cols[key] = dict(col)
    return cols


def build_arm_board(rows: Sequence[Mapping[str, Any]], *, split: str,
                    extra_columns: Mapping[str, Mapping[str, Any]],
                    seed: int = 20260810, published: bool = True,
                    **board_extra: Any) -> dict[str, Any]:
    """``criteria.build_board`` keyed by the EPR (so ``ARM_AXES`` resolves) + the arm name."""
    board = build_board(rows, arm=EPR, split=split, extra_columns=extra_columns, seed=seed)
    board["arm_name"] = ARM
    board["axes"] = list(AXES)
    board["published"] = bool(published)
    board.update(board_extra)
    return board


def publish_board(board: Mapping[str, Any], *, steps_row: Mapping[str, Any] | None = None,
                  steps_path: Any = None, eval_only: bool = False,
                  loss_level: int = 3) -> dict[str, Any]:
    """The gate in front of ``metrics.json``: training side + criteria + headline.

    The steps row is resolved by the guards' three tiers (caller -> disk ->
    in-process witness) and the two failure modes raise different exceptions --
    the where-side contract mismatch that cost SEGSAM and PRND a run each.
    """
    return assert_publishable(
        board, EPR,
        steps_row=steps_row, steps_path=steps_path, eval_only=eval_only,
        loss_level=loss_level, required=required_columns(),
    )


def run_setup(head: AffineOnlyHead, *, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Everything the run must be able to prove about itself, in one dict."""
    cfg = head.cfg
    setup: dict[str, Any] = {
        "arm": ARM,
        "epr": EPR,
        "axes": list(AXES),
        "frozen": dict(FROZEN),
        "head": head.config,
        "required_columns": required_columns(),
        "step_columns": list(step_columns_for(cfg.loss_level)),
        "assertions": {
            "shared_geom_identical": {"n_probes": cfg.shared_geom_probes,
                                      "test": "torch.equal (not allclose)"},
            "affine_linearity_maxdev": {"n_pairs": cfg.linearity_pairs,
                                        "grid_n": cfg.linearity_grid,
                                        "tol": cfg.linearity_tol,
                                        "dtype": "float32",
                                        "clamp": "none"},
            "degenerate_weight_rate": {"tau": cfg.degenerate_weight_tau},
            "degeneracy_guard": cfg.degeneracy.as_dict(),
        },
        "loss": {
            "form": "L_rec + lambda_hc * L_hc + lambda_sparse * R_sparse (GLUT Eq.6-8)",
            "lambda_hc": cfg.lambda_hc,
            "lambda_sparse": cfg.lambda_sparse,
            **pure_l1_record(loss_level=cfg.loss_level,
                             lambda_hc=cfg.lambda_hc_effective,
                             lambda_sparse=cfg.lambda_sparse_effective),
            "l_rec_reduction": "mean",
            "l_hc_mask": f"hard mask 1[C >= {cfg.eps_c}], h = (a,b)/max(C, {cfg.eps_c})",
        },
        "optimizer": {
            "name": "Adam", "base_lr": cfg.base_lr, "weight_decay": cfg.weight_decay,
            "grad_clip": cfg.grad_clip, "scheduler": "cosine annealing, no warm-up",
            "shared_geometry_lr": cfg.base_lr * cfg.shared_lr_scale,
            "shared_geometry_lr_scale": cfg.shared_lr_scale,
            "shared_group": "{mu, chol_diag, chol_off, opacity_logit} (o in the group is NOVEL)",
        },
        "mining": {"rule": "within-batch top-r resampling (ruling 11.1-4)",
                   "epoch_start": cfg.mining_start_epoch, "epoch_end": cfg.mining_end_epoch,
                   "r_start": cfg.mining_r_start, "r_end": cfg.mining_r_end},
    }
    if extra:
        setup.update(dict(extra))
    return setup
