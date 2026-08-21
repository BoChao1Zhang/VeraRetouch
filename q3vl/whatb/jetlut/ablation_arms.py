"""EPR-032 additional ablation arms: J-GATE-AFF / J-4D / J-P15.

Three zero-randomness convex arms, all on the EPR-032 §10 caliber
(``X_fit = 33^3``, ``X_eval`` = the fixed 32,768 colours of ``65^3 \\ 33^3``,
pre-clamp, dE00, ADMM-LAD ``rho=100 / alpha=1.7 / 2000`` iterations) and all
routed through the *existing* solver, metrics and pool loaders:

======================  ======================================================
arm                     what changes relative to the §10 ladder
======================  ======================================================
``J-GATE-AFF``          gate := the AFFONLY checkpoint's learned shared
                        geometry (Shepard + eps + opacity, ``N = 48``), basis
                        := p1 jet.  ``P_dyn = 12*48 + 12 = 588``.
``J-4D``                atlas := 4-D ``(x, s)`` cell centres ``m^3 x m_s``,
                        exact POU in 4-D, monomials ``[1, d1, d2, d3, ds]``,
                        target ``y_s(x) = (1-s) x + s L(x)``.
                        ``P_dyn = 3*5*N4 + 12``.
``J-P15``               basis := ``[1, d1, d2, d3, d1^2, d2^2, d3^2]`` -- p2
                        without the three cross terms.  ``P_dyn = 21N + 12``.
======================  ======================================================

Nothing here re-implements the ADMM, the dE00 statistics, the LUT pools or the
GLUT gate.  ``run.fit_and_score`` owns the first two, ``run.all_pools`` the
third, and ``q3vl.whatb.glut.glut_forward`` the fourth -- this module only
supplies design matrices and the arm-specific runtime assertions.

Pools are pinned to the ``v20260804`` dataset version (EPR-030 NOTES item 6:
``run.py``'s pool loader otherwise follows the process default, which is
``cut-p45`` and gives 948/232 instead of 902/259).  ``assert_pool_sha`` refuses
to continue on any other id set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
from torch import Tensor

from q3vl.whatb import splits as S
from q3vl.whatb.colorimetry import delta_e00_srgb
from q3vl.whatb.glut import GlutParams, glut_forward
from q3vl.whatb.jetlut.core import Atlas, admm_lad, design_matrix, pou_weights
from q3vl.whatb.jetlut.gate import _interp_log

__all__ = [
    "Atlas4D",
    "DE00_STAT_KEYS",
    "GATE_XCHECK_TOL",
    "SHARED_GEOMETRY_KEYS",
    "assert_gate_matches_glut",
    "assert_nested_three",
    "assert_pool_sha",
    "assert_pou_4d",
    "de00_paired",
    "design_matrix_gate",
    "design_matrix_4d",
    "design_matrix_p15",
    "fill_distance_1d",
    "fit_score_pooled",
    "gate_weights",
    "load_shared_geometry",
    "lut_bounds",
    "lut_chunk_fields",
    "paired_de00_stats",
    "pooled_rhat",
    "pou_weights_4d",
    "reference_or_omitted",
    "theta_to_glut_params",
]

#: EPR-030 NOTES item 6 -- the only口径 whose pools reproduce §10's shas.
DATASET_VERSION = "v20260804"

#: PROPOSAL §10 / §11: ``lut_ids_sha`` (sha256 of "\n".join(sorted ids), 16 hex)
#: and n, for every pool ``run.all_pools`` can board whose id set is a fixed
#: function of the index (``train`` is not: it is the ``--n-train`` subsample).
POOL_SHA: dict[str, tuple[str, int]] = {
    "held_out": ("d3f17955816f39b1", 902),
    "t_lut_unseen": ("72f1dd33ce35a12c", 259),
    # PROPOSAL §11.1 / §11.2, the two full pools the §11 ladders were run on
    # (``run.all_pools``' own ``train_full`` / ``all``; both re-measured under
    # the pinned v20260804 index before being registered here).
    "train_full": ("06da414b16ea0c6c", 3149),
    "all": ("f964eba67b9cdc43", 4051),
}

#: the AFFONLY main arm whose shared geometry arm 1 borrows
AFFONLY_CKPT = "/home/bc/data/runs/what_b/whatb_AFFONLY_20260815/best.pt"

SHARED_GEOMETRY_KEYS = (
    "generator.shared_geometry.mu",
    "generator.shared_geometry.chol_diag",
    "generator.shared_geometry.chol_off",
    "generator.shared_geometry.opacity_logit",
)

#: PROPOSAL §10.2, held_out (n=902, c*=0.7, sha d3f17955816f39b1), fit-grid
#: pre-clamp p95 -- transcribed, never recomputed.  Only used when the on-disk
#: ``ladder_<pool>.json`` is absent; when it is present its sha is asserted and
#: its own rows are used.
REFERENCE_LADDER_P95: dict[str, dict[str, list[tuple[int, float]]]] = {
    "held_out": {
        "fitgrid": [(336, 2.5725), (780, 1.7511), (1512, 1.3247),
                    (2604, 1.0646), (4128, 0.8966),
                    (822, 1.6517), (1932, 1.1273), (3762, 0.8574),
                    (6492, 0.6961)],
    },
}
#: index into the list above: the first five entries are p=1, the rest p=2
_N_P1_REF = 5


# --------------------------------------------------------------------------- #
# pools
# --------------------------------------------------------------------------- #
def sha16(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16]


def pin_dataset_version(name: str = DATASET_VERSION, *, force: bool = False
                        ) -> str:
    """Pin the process-wide口径 before any index is read (EPR-030 NOTES 6).

    ``force`` is for tests only: a driver that has already parsed an index under
    another口径 must fail, not silently mix two.
    """
    return S.use_dataset_version(name, force=force).name


def assert_pool_sha(pool: str, ids: Sequence[str]) -> dict[str, object]:
    """Refuse to fit on a pool whose id set is not §10's.

    A drifted pool is the failure mode that silently makes every number in this
    EPR incomparable to the published ladder, and it is invisible in the output
    (the tables still look fine).  Registered pools are hard-asserted; an
    unregistered one is recorded, not guessed at.
    """
    got, n = sha16(ids), len(ids)
    rec = {"pool": pool, "n_lut": n, "lut_ids_sha": got,
           "dataset_version": S.active_dataset_version().name}
    if pool not in POOL_SHA:
        rec["expected"] = None
        return rec
    want_sha, want_n = POOL_SHA[pool]
    rec["expected"] = {"lut_ids_sha": want_sha, "n_lut": want_n}
    if got != want_sha or n != want_n:
        raise AssertionError(
            f"pool {pool!r} is {n} ids / sha {got}, PROPOSAL §10 registered "
            f"{want_n} / {want_sha}.  Active dataset version is "
            f"{S.active_dataset_version().name!r}; §10 was measured on "
            f"{DATASET_VERSION!r}.")
    return rec


# --------------------------------------------------------------------------- #
# arm 1 -- the AFFONLY shared geometry as the gate
# --------------------------------------------------------------------------- #
def load_shared_geometry(path: str | Path = AFFONLY_CKPT, *,
                         device: torch.device | str = "cpu",
                         dtype: torch.dtype = torch.float64
                         ) -> tuple[dict[str, Tensor], dict[str, object]]:
    """The four ``SharedGeometry`` tables of an AFFONLY checkpoint.

    Returns ``(tensors, provenance)``; the provenance carries the sha256 of each
    tensor's raw bytes *as stored* (fp32), so the arm's artefact pins the exact
    geometry it borrowed rather than a run directory that can be overwritten.
    """
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    missing = [k for k in SHARED_GEOMETRY_KEYS if k not in sd]
    if missing:
        raise KeyError(
            f"{path} has no shared geometry: missing {missing}.  This arm needs "
            "an AFFONLY run with --share geo_opacity (generator.py "
            "SharedGeometry), not a per-sample-geometry checkpoint.")
    prov: dict[str, object] = {
        "checkpoint": str(path), "step": ck.get("step"),
        "headline_normal_only": ck.get("headline_normal_only"),
        "tensors": {},
    }
    out: dict[str, Tensor] = {}
    for k in SHARED_GEOMETRY_KEYS:
        t = sd[k].detach().cpu().contiguous()
        prov["tensors"][k] = {
            "shape": list(t.shape), "dtype": str(t.dtype),
            "sha256": hashlib.sha256(t.numpy().tobytes()).hexdigest(),
        }
        out[k.rsplit(".", 1)[1]] = t.to(device=device, dtype=dtype)
    prov["n_gauss"] = int(out["mu"].shape[0])
    return out, prov


def _glut_params(geo: dict[str, Tensor], m_local: Tensor, b_local: Tensor,
                 g_matrix: Tensor, g_bias: Tensor) -> GlutParams:
    return GlutParams(
        mu=geo["mu"].unsqueeze(0), chol_diag=geo["chol_diag"].unsqueeze(0),
        chol_off=geo["chol_off"].unsqueeze(0),
        opacity_logit=geo["opacity_logit"].unsqueeze(0),
        m_local=m_local.unsqueeze(0), b_local=b_local.unsqueeze(0),
        g_matrix=g_matrix.unsqueeze(0), g_bias=g_bias.unsqueeze(0))


def gate_weights(x: Tensor, geo: dict[str, Tensor], *,
                 point_chunk: int | None = None,
                 affine: tuple[Tensor, Tensor, Tensor, Tensor] | None = None
                 ) -> Tensor:
    """``(Q, N)`` GLUT gate ``w_i(x) = p_i o_i / (sum_j p_j o_j + eps)``.

    Computed by ``glut.glut_forward(..., return_aux=True)`` -- the one
    implementation in the codebase (Eq.1-2, log-domain PDF, ``Sigma = L L^T +
    eps I``, ``eps = 1e-6``, non-exact POU).  ``affine`` only exists so the
    cross-check can prove the gate does not depend on it.
    """
    n = int(geo["mu"].shape[0])
    dev, dt = geo["mu"].device, geo["mu"].dtype
    if affine is None:
        affine = (torch.zeros((n, 3, 3), device=dev, dtype=dt),
                  torch.zeros((n, 3), device=dev, dtype=dt),
                  torch.zeros((3, 3), device=dev, dtype=dt),
                  torch.zeros((3,), device=dev, dtype=dt))
    params = _glut_params(geo, *affine)
    _, aux = glut_forward(x.unsqueeze(0).to(device=dev, dtype=dt), params,
                          clamp="none", residual=True, return_aux=True,
                          point_chunk=point_chunk)
    return aux.weights[0]


#: Tolerance floor of the gate cross-check.  On CPU the two gate evaluations are
#: bit-identical; on CUDA the ``sum_j p_j o_j`` reduction picks a different
#: accumulation order for a different point chunk and the weights differ at the
#: denormal level (measured 6.776e-21 on cuda:0, N=48, 4,096 probe colours) --
#: the same CPU/CUDA 1-ULP situation as the EPR-030 step-0 assertion.  ``w_i(x)``
#: lies in ``[0, 1]``, so 1e-12 is ~10 orders of magnitude below any real
#: mis-wiring (reading the affine coefficients, or a blocking-dependent gate,
#: both move the weights by O(1) -- see the reverse tests).
GATE_XCHECK_TOL = 1e-12


def assert_gate_matches_glut(geo: dict[str, Tensor], *, n_probe: int = 4096,
                             seed: int = 20260818,
                             tol: float = GATE_XCHECK_TOL) -> dict[str, object]:
    """Runtime assertion: the gate this arm fits on is ``glut_forward``'s gate.

    There is exactly one gate implementation (``glut.py``) and this arm imports
    it, so this is a *wiring* check, not an independent reimplementation: on
    ``n_probe`` random colours the weights must agree to ``tol`` when the local /
    global affine coefficients change (the gate must not read them) and when the
    point-chunk changes (the gate must not depend on blocking).  Either failure
    means the design matrix below is not the carrier it claims to be.

    The measured ``max_abs_diff`` is returned next to ``tol`` so the artefact
    carries the number, not just a boolean; ``bitwise_equal`` records whether the
    two evaluations were exactly equal (they are on CPU).
    """
    dev, dt = geo["mu"].device, geo["mu"].dtype
    n = int(geo["mu"].shape[0])
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.rand((n_probe, 3), generator=g, dtype=torch.float64).to(
        device=dev, dtype=dt)
    w_a = gate_weights(x, geo)
    noisy = (torch.randn((n, 3, 3), generator=g, dtype=torch.float64).to(dev, dt),
             torch.randn((n, 3), generator=g, dtype=torch.float64).to(dev, dt),
             torch.randn((3, 3), generator=g, dtype=torch.float64).to(dev, dt),
             torch.randn((3,), generator=g, dtype=torch.float64).to(dev, dt))
    w_b = gate_weights(x, geo, point_chunk=997, affine=noisy)
    diff = float((w_a - w_b).abs().max())
    if not (diff <= tol):
        raise AssertionError(
            "gate cross-check failed: glut_forward's w_i(x) changed when the "
            f"affine coefficients / point chunk changed (max abs diff "
            f"{diff:.3e} > {tol:g}) -- the design matrix would "
            "not be the GLUT carrier.")
    return {"n_probe": int(n_probe), "seed": int(seed),
            "max_abs_diff": diff, "tol": float(tol),
            "bitwise_equal": bool(torch.equal(w_a, w_b)),
            "weight_row_sum_min": float(w_a.sum(-1).min()),
            "weight_row_sum_max": float(w_a.sum(-1).max()),
            "weight_max": float(w_a.max())}


def design_matrix_gate(x: Tensor, geo: dict[str, Tensor], *, chunk: int = 4096,
                       point_chunk: int | None = None) -> Tensor:
    """``(Q, 4 + 4N)`` features of ``sum_i w_i(x)(M_i x + b_i) + G x + g``.

    Column layout mirrors ``core.design_matrix``: the global block ``[1, x]``
    first, then ``N`` local blocks ``w_i * [1, x]``.  Local monomials are the
    *unshifted* ``x`` (not ``(x - mu_i)/sigma``) so the coefficients are GLUT's
    own ``(M_i, b_i)`` and ``theta_to_glut_params`` is an identity relabelling.

    The gate is evaluated in a **single** ``glut_forward`` call and only the
    ``w_i * [1, x]`` assembly is blocked, so ``chunk`` is exactly inert: it moves
    no reduction.  Calling the gate per block instead would not be -- the
    ``sum_j p_j o_j`` denominator is a reduction whose kernel splits with the
    batch size, measured at ~1 ulp (1.4e-16, fp64) on 2 of 1000 points for
    ``chunk=97``, which is small but not zero and would make the design matrix a
    function of a blocking parameter.
    """
    n = int(geo["mu"].shape[0])
    w_all = gate_weights(x, geo, point_chunk=point_chunk)      # (Q, N)
    out = x.new_empty((x.shape[0], 4 + 4 * n))
    for s in range(0, x.shape[0], chunk):
        xb = x[s:s + chunk]
        w = w_all[s:s + chunk]                                 # (q, N)
        glob = torch.cat((torch.ones_like(xb[:, :1]), xb), dim=-1)   # (q, 4)
        local = w.unsqueeze(-1) * glob.unsqueeze(1)            # (q, N, 4)
        out[s:s + chunk] = torch.cat(
            (glob, local.reshape(xb.shape[0], -1)), dim=-1)
    return out


def theta_to_glut_params(theta: Tensor, geo: dict[str, Tensor]) -> GlutParams:
    """``(4+4N, 3)`` residual coefficients -> a ``GlutParams`` batch of 1.

    The fit is on the residual ``L(x) - x`` while GLUT's global branch emits the
    value, so the identity is folded into ``G`` here (``G_glut = G_fit + I``).
    """
    n = int(geo["mu"].shape[0])
    g_bias = theta[0]                                          # (3,)
    g_matrix = theta[1:4].transpose(0, 1).contiguous()         # (3, 3)
    g_matrix = g_matrix + torch.eye(3, device=theta.device, dtype=theta.dtype)
    loc = theta[4:].reshape(n, 4, 3)
    b_local = loc[:, 0, :].contiguous()                        # (N, 3)
    m_local = loc[:, 1:4, :].transpose(1, 2).contiguous()      # (N, 3, 3)
    return _glut_params(geo, m_local, b_local, g_matrix, g_bias)


def assert_carrier_matches_glut(x: Tensor, geo: dict[str, Tensor],
                                theta: Tensor, *, tol: float = 1e-9,
                                n_probe: int = 4096) -> dict[str, object]:
    """``x + Phi(x) theta`` must equal ``glut_forward`` on the relabelled theta.

    This is the assertion that the arm fits the GLUT carrier and not merely
    something built from GLUT's gate.
    """
    xs = x[:n_probe]
    lhs = xs + design_matrix_gate(xs, geo) @ theta[:, :3]
    rhs = glut_forward(xs.unsqueeze(0), theta_to_glut_params(theta[:, :3], geo),
                       clamp="none", residual=True)[0]
    err = float((lhs - rhs).abs().max())
    if not (err <= tol):
        raise AssertionError(
            f"carrier cross-check failed: |x + Phi theta - glut_forward| = "
            f"{err:.3e} > {tol:g}")
    return {"n_probe": int(xs.shape[0]), "max_abs_diff": err, "tol": tol}


# --------------------------------------------------------------------------- #
# arm 2 -- the 4-D (x, s) atlas
# --------------------------------------------------------------------------- #
def fill_distance_1d(m: int) -> float:
    """Fill distance of the ``m`` cell-centre grid on ``[0, 1]``: ``1/(2m)``."""
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    return 1.0 / (2.0 * m)


@dataclass(frozen=True)
class Atlas4D:
    """``m^3 x m_s`` cell centres on ``[0,1]^3 x [0,1]``, isotropic per block.

    ``sigma_x = c * h_m`` with ``h_m = sqrt(3)/(2m)`` (``core.fill_distance``)
    and ``sigma_s = c * h_ms`` with ``h_ms = 1/(2 m_s)`` -- the same ``c`` on
    both blocks, each against its own fill distance.  ``c`` is not recalibrated
    for 4-D (see NOTES).
    """

    m: int
    m_s: int
    c: float

    @property
    def n(self) -> int:
        return self.m ** 3 * self.m_s

    @property
    def sigma_x(self) -> float:
        return self.c * math.sqrt(3.0) / (2.0 * self.m)

    @property
    def sigma_s(self) -> float:
        return self.c * fill_distance_1d(self.m_s)

    def scale(self, *, device: torch.device | str = "cpu",
              dtype: torch.dtype = torch.float64) -> Tensor:
        return torch.tensor([self.sigma_x] * 3 + [self.sigma_s],
                            device=device, dtype=dtype)

    def centres(self, *, device: torch.device | str = "cpu",
                dtype: torch.dtype = torch.float64) -> Tensor:
        """``(N4, 4)``; x-major then s, matching ``Atlas.centres``' R-major."""
        ax = (torch.arange(self.m, device=device, dtype=dtype) + 0.5) / self.m
        sa = (torch.arange(self.m_s, device=device, dtype=dtype) + 0.5) / self.m_s
        r, g, b, s = torch.meshgrid(ax, ax, ax, sa, indexing="ij")
        return torch.stack([t.reshape(-1) for t in (r, g, b, s)], dim=-1)


def n_dynamic_params_4d(atlas: Atlas4D) -> int:
    """``3 (4 + 5 N4)`` -- 5 monomials ``[1, d1, d2, d3, ds]`` per primitive."""
    return 3 * (4 + 5 * atlas.n)


def pou_weights_4d(xs: Tensor, atlas: Atlas4D) -> Tensor:
    """``(Q, N4)`` exact 4-D POU.

    Implemented by pre-dividing both the queries and the centres by the
    anisotropic scale and handing the result to ``core.pou_weights`` with
    ``sigma = 1``: identical arithmetic, one implementation.
    """
    sc = atlas.scale(device=xs.device, dtype=xs.dtype)
    mu = atlas.centres(device=xs.device, dtype=xs.dtype) / sc
    return pou_weights(xs / sc, mu, 1.0)


POU4D_TOL = 1e-6


def assert_pou_4d(atlas: Atlas4D, *, n_probe: int = 4096, seed: int = 20260818,
                  tol: float = POU4D_TOL,
                  device: torch.device | str = "cpu") -> dict[str, object]:
    """Runtime assertion (in-arm): the 4-D POU rows sum to 1 on random ``(x,s)``.

    The exactness of the 4-D partition of unity is what makes the local jets a
    convex blend and the ``P_dyn`` bookkeeping honest; an underflowed or
    mis-scaled denominator silently rescales every local coefficient and the
    tables still look fine.  ``max|sum - 1|`` is returned so the artefact carries
    the measured deviation, not just a boolean.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    xs = torch.rand((n_probe, 4), generator=g, dtype=torch.float64).to(device)
    dev = float((pou_weights_4d(xs, atlas).sum(-1) - 1.0).abs().max())
    if not (dev < tol):
        raise AssertionError(
            f"4-D POU is not a partition of unity for m={atlas.m} "
            f"m_s={atlas.m_s} c={atlas.c}: max|sum_i pi_i(x,s) - 1| = "
            f"{dev:.3e} >= {tol:g} on {n_probe} random (x, s) points.")
    return {"m": atlas.m, "m_s": atlas.m_s, "c": atlas.c, "N4": atlas.n,
            "n_probe": int(n_probe), "seed": int(seed), "tol": float(tol),
            "max_abs_dev_from_one": dev}


def design_matrix_4d(xs: Tensor, atlas: Atlas4D, *, chunk: int = 2048) -> Tensor:
    """``(Q, 4 + 5 N4)`` for ``xs`` of shape ``(Q, 4)`` = ``[x_R,x_G,x_B,s]``.

    Global block is ``[1, x]`` only (no ``s``): the s-dependence lives entirely
    in the local jets, which is what makes ``P_dyn = 3*5*N4 + 12``.
    """
    if xs.shape[-1] != 4:
        raise ValueError(f"xs must be (Q, 4), got {tuple(xs.shape)}")
    sc = atlas.scale(device=xs.device, dtype=xs.dtype)
    mu = atlas.centres(device=xs.device, dtype=xs.dtype)
    out = xs.new_empty((xs.shape[0], 4 + 5 * atlas.n))
    for s0 in range(0, xs.shape[0], chunk):
        xb = xs[s0:s0 + chunk]                                  # (q, 4)
        pi = pou_weights_4d(xb, atlas)                          # (q, N4)
        delta = (xb.unsqueeze(1) - mu.unsqueeze(0)) / sc        # (q, N4, 4)
        mono = torch.cat((torch.ones_like(delta[..., :1]), delta), dim=-1)
        local = pi.unsqueeze(-1) * mono                         # (q, N4, 5)
        glob = torch.cat((torch.ones_like(xb[:, :1]), xb[:, :3]), dim=-1)
        out[s0:s0 + chunk] = torch.cat(
            (glob, local.reshape(xb.shape[0], -1)), dim=-1)
    return out


def stack_s(x: Tensor, s_values: Sequence[float]) -> Tensor:
    """``(Q * len(s), 4)`` -- every fit colour at every strength."""
    blocks = []
    for s in s_values:
        col = torch.full((x.shape[0], 1), float(s), device=x.device,
                         dtype=x.dtype)
        blocks.append(torch.cat((x, col), dim=-1))
    return torch.cat(blocks, dim=0)


def stack_s_residual(r: Tensor, s_values: Sequence[float]) -> Tensor:
    """``y_s(x) - x = s (L(x) - x)`` stacked in the same block order."""
    return torch.cat([float(s) * r for s in s_values], dim=0)


# --------------------------------------------------------------------------- #
# arm 3 -- the diagonal-second-order basis
# --------------------------------------------------------------------------- #
def _mono_p1(delta: Tensor) -> Tensor:
    d1, d2, d3 = delta.unbind(-1)
    return torch.stack([torch.ones_like(d1), d1, d2, d3], dim=-1)


def _mono_p15(delta: Tensor) -> Tensor:
    d1, d2, d3 = delta.unbind(-1)
    return torch.stack([torch.ones_like(d1), d1, d2, d3,
                        d1 * d1, d2 * d2, d3 * d3], dim=-1)


def _mono_p2(delta: Tensor) -> Tensor:
    d1, d2, d3 = delta.unbind(-1)
    return torch.stack([torch.ones_like(d1), d1, d2, d3,
                        d1 * d1, d2 * d2, d3 * d3,
                        d1 * d2, d1 * d3, d2 * d3], dim=-1)


MONOMIALS: dict[str, tuple[Callable[[Tensor], Tensor], int]] = {
    "p1": (_mono_p1, 4), "p15": (_mono_p15, 7), "p2": (_mono_p2, 10),
}


def design_matrix_mono(x: Tensor, atlas: Atlas, order: str, *,
                       chunk: int = 4096) -> Tensor:
    """``core.design_matrix`` generalised to an arbitrary monomial list.

    ``order='p1'`` / ``'p2'`` reproduce ``core.design_matrix(x, atlas, 1|2)``
    bit for bit (asserted by ``tests/test_epr032_ablation.py``); ``'p15'`` is the
    new diagonal-only second order.
    """
    mono_fn, k = MONOMIALS[order]
    mu = atlas.centres(device=x.device, dtype=x.dtype)
    sigma = atlas.sigma
    out = x.new_empty((x.shape[0], 4 + atlas.n * k))
    for s in range(0, x.shape[0], chunk):
        xb = x[s:s + chunk]
        pi = pou_weights(xb, mu, sigma)
        delta = (xb.unsqueeze(1) - mu.unsqueeze(0)) / sigma
        local = pi.unsqueeze(-1) * mono_fn(delta)
        glob = torch.cat((torch.ones_like(xb[:, :1]), xb), dim=-1)
        out[s:s + chunk] = torch.cat(
            (glob, local.reshape(xb.shape[0], -1)), dim=-1)
    return out


def design_matrix_p15(x: Tensor, atlas: Atlas, *, chunk: int = 4096) -> Tensor:
    return design_matrix_mono(x, atlas, "p15", chunk=chunk)


def n_dynamic_params_order(m: int, order: str) -> int:
    return 3 * (4 + m ** 3 * MONOMIALS[order][1])


NESTED_TOL = 1e-4

#: the three orders the nesting assertion needs; a run that boards a subset of
#: them produces zero checks, and a zero-check assertion cannot fail.
NESTED_ORDERS = ("p1", "p15", "p2")


def check_nested_three(rows: Sequence[dict], tol: float = NESTED_TOL
                       ) -> list[dict]:
    """``E_p1 >= E_p15 >= E_p2`` at matched ``m`` (G-wire, same form as §8.1).

    p15 is a strict subspace of p2 and a strict superspace of p1 on the same
    atlas and the same fit colours, so both inequalities are wiring facts.  A
    violation is a solver / design-matrix failure, never a result.
    """
    by = {(r["m"], r["order"]): r["l1_fit_mean_per_point"] for r in rows}
    out = []
    for m in sorted({m for m, _ in by}):
        e1, e15, e2 = (by.get((m, o)) for o in ("p1", "p15", "p2"))
        if e1 is None or e15 is None or e2 is None:
            continue
        ok1 = e15 <= e1 * (1.0 + tol)
        ok2 = e2 <= e15 * (1.0 + tol)
        out.append({"m": m, "l1_p1": e1, "l1_p15": e15, "l1_p2": e2,
                    "ratio_p15_p1": e15 / e1 if e1 else math.nan,
                    "ratio_p2_p15": e2 / e15 if e15 else math.nan,
                    "ok_p15_le_p1": bool(ok1), "ok_p2_le_p15": bool(ok2),
                    "ok": bool(ok1 and ok2)})
    return out


def assert_nested_three(rows: Sequence[dict], tol: float = NESTED_TOL, *,
                        orders: Sequence[str] | None = None,
                        n_m: int | None = None) -> list[dict]:
    """``check_nested_three`` + the coverage assertions that make it bite.

    ``check_nested_three`` skips any ``m`` that is missing one of the three
    orders, so a run boarded with ``--orders p1 p2`` yields ``checks == []`` and
    the assertion passes vacuously while ``nested_ok`` is still written as true.
    Three coverage conditions are therefore checked *before* the inequality:

    * ``orders`` (when given, the driver's ``--orders``) must contain all of
      ``NESTED_ORDERS``;
    * the rows themselves must carry all of ``NESTED_ORDERS``;
    * the number of complete triples must equal ``n_m`` (the driver's
      ``len(args.m)``) -- or, when ``n_m`` is not given, the number of distinct
      ``m`` present in the rows, which catches a triple that is complete
      corpus-wide but incomplete at some ``m``.
    """
    want = set(NESTED_ORDERS)
    if orders is not None:
        missing = want - set(orders)
        if missing:
            raise AssertionError(
                f"nested assertion needs orders {sorted(want)}, the run boarded "
                f"{list(orders)} (missing {sorted(missing)}): with a missing "
                "order every m is skipped, checks == [] and the assertion "
                "cannot fail.")
    have_rows = {r["order"] for r in rows if "order" in r}
    missing_rows = want - have_rows
    if missing_rows:
        raise AssertionError(
            f"nested assertion needs rows for orders {sorted(want)}, got "
            f"{sorted(have_rows)} (missing {sorted(missing_rows)}).")
    checks = check_nested_three(rows, tol)
    n_want = len(({r["m"] for r in rows if "order" in r}) if n_m is None
                 else range(n_m))
    if len(checks) != n_want:
        got = sorted(q["m"] for q in checks)
        raise AssertionError(
            f"nested assertion produced {len(checks)} complete p1/p15/p2 "
            f"triples at m={got}, expected {n_want}: some m is missing an "
            "order, and a skipped m is an unchecked m.")
    bad = [q for q in checks if not q["ok"]]
    if bad:
        raise AssertionError(
            "nested subspace inequality E_p1 >= E_p15 >= E_p2 violated at "
            f"{[q['m'] for q in bad]}: {bad}")
    return checks


# --------------------------------------------------------------------------- #
# shared: the two dE00 pairings
# --------------------------------------------------------------------------- #
#: the dE00 statistic blocks ``run.fit_and_score`` writes; each is emitted twice
#: by the three arms, ``*_legacy`` and ``*_paired`` (see ``de00_paired``).
DE00_STAT_KEYS = ("fitgrid_pre_clamp", "fitgrid_post_clamp",
                  "fitgrid_strata", "heldgrid_pre_clamp", "heldgrid_post_clamp")

#: restated in every artefact so a reader of the JSON alone knows which of the
#: two dE00 columns is which.
DE00_CALIBERS = {
    "legacy": "run.de00_of as published: r[:, 3s:3e].T.reshape(e-s, Q, 3), "
              "which pairs one channel at three neighbouring points as a "
              "colour.  The PROPOSAL §10 ladder and the reference_ladder rows "
              "are measured in this caliber, so the *_legacy columns are the "
              "horizontally comparable ones.",
    "paired": "ablation_arms.de00_paired: r[:, 3s:3e].reshape(Q, e-s, 3)"
              ".permute(1, 0, 2), i.e. column 3l+ch read as run.residuals "
              "writes it.  Same colorimetry call "
              "(colorimetry.delta_e00_srgb), corrected shape-taking only.",
}


def de00_paired(x: Tensor, r: Tensor, rhat: Tensor, *, clamp: bool,
                chunk: int = 64) -> Tensor:
    """``(Q, L)`` dE00 with the ``(point, LUT, channel)`` pairing corrected.

    ``run.de00_of`` takes the ``(Q, 3L)`` residual block as
    ``r[:, 3s:3e].T.reshape(e-s, Q, 3)``: after ``.T`` the flat order is
    ``[lut_s ch0 over all points, lut_s ch1 over all points, ...]``, so each
    length-3 triple is *one channel at three neighbouring points*, not one
    point's RGB.  ``y`` and ``yhat`` share that shape-taking but the base colour
    ``x`` does not, so the dE00 is evaluated on synthetic colours.  The correct
    take is ``r[:, 3s:3e].reshape(Q, e-s, 3).permute(1, 0, 2)`` (column ``3l+ch``,
    :func:`run.residuals`).

    Only the *shape-taking* differs -- the colorimetry is
    ``colorimetry.delta_e00_srgb``, the same call ``run.de00_of`` makes, so this
    is not a second dE00 implementation.  The arms report both columns:
    ``*_legacy`` (``run.de00_of``, the caliber the §10 ladder was measured in,
    kept for horizontal comparability) and ``*_paired`` (this one).
    """
    q = x.shape[0]
    lut_n = r.shape[1] // 3
    out = torch.empty((q, lut_n), dtype=torch.float32, device=x.device)
    xf = x.to(torch.float32)
    for s in range(0, lut_n, chunk):
        e = min(s + chunk, lut_n)
        # ``.contiguous()``: ``permute`` leaves the block in the query-major
        # layout, and ``colorimetry``'s sRGB->XYZ matmul picks a different
        # accumulation order for a strided input (~1e-5 relative on dE00).  A
        # contiguous block makes the value a function of the numbers only.
        y = xf.unsqueeze(0) + r[:, 3 * s:3 * e].reshape(
            q, e - s, 3).permute(1, 0, 2).to(torch.float32).contiguous()
        yh = xf.unsqueeze(0) + rhat[:, 3 * s:3 * e].reshape(
            q, e - s, 3).permute(1, 0, 2).to(torch.float32).contiguous()
        if clamp:
            yh = yh.clamp(0.0, 1.0)
        out[:, s:e] = delta_e00_srgb(yh, y).T
    return out


def paired_de00_stats(R, rec: dict, *, x_fit, x_eval, r_fit, r_eval, theta,
                      build_fit, build_eval,
                      strata: dict[str, Tensor] | None = None) -> dict:
    """Rename ``fit_and_score``'s dE00 blocks to ``*_legacy``, add ``*_paired``.

    ``run.py`` is frozen this round (its subcommand behaviour is the §10
    comparability anchor), so the corrected pairing is applied here, on the same
    ``theta`` and the same design matrices.  The design matrices are rebuilt --
    ``fit_and_score`` frees them -- which costs one extra build per grid and
    nothing else; the fit itself is not redone.
    """
    for k in DE00_STAT_KEYS:
        if k in rec:
            rec[f"{k}_legacy"] = rec.pop(k)
    for tag, xg, rg, build in (("fitgrid", x_fit, r_fit, build_fit),
                               ("heldgrid", x_eval, r_eval, build_eval)):
        f = build()
        rhat = f @ theta
        for clamp in (False, True):
            key = f"{tag}_{'post_clamp' if clamp else 'pre_clamp'}_paired"
            de = de00_paired(xg, rg, rhat, clamp=clamp)
            rec[key] = R._stats(de)
            if strata is not None and tag == "fitgrid" and not clamp:
                rec["fitgrid_strata_paired"] = {
                    k: R._stats(de[mask]) for k, mask in strata.items()
                    if int(mask.sum()) > 0}
            del de
        del f, rhat
    torch.cuda.empty_cache()
    return rec


# --------------------------------------------------------------------------- #
# shared: pool-dimension chunking (``--lut-chunk``)
# --------------------------------------------------------------------------- #
#: ``core.admm_lad`` carries ``r``/``z``/``u``/``z_old``/``ax``/``ax_hat`` at the
#: full ``(Q, 3L)`` shape, so its peak is ``~7 * Q * 3L * 8`` bytes: the J-4D
#: ``train_full`` fit (``Q = 5 * 33^3 = 179,685``, ``L = 3,149``) needs ~95 GiB
#: and OOMs on an 80/96 GiB card.  ``--lut-chunk N`` splits the *pool* (never the
#: colours, never the basis) into blocks of at most ``N`` LUT ids in registry
#: order and runs the existing ``run.fit_and_score`` once per block.
#:
#: What the split does and does not preserve:
#:
#: * the pool sha assertion still runs on the **full** pool, before the split
#:   (``_pool_ids``), and ``reference_or_omitted`` still keys off the full row
#:   count, so a chunked run writes exactly the reference a whole-pool run does;
#: * every pooled dE00 block is recomputed from the *concatenated per-LUT*
#:   ``(Q, L)`` field and handed to ``run._stats`` **once**, so ``mean`` / ``p95``
#:   / ``p99`` / ``max`` / the per-LUT views are the same reduction over the same
#:   multiset a whole-pool run performs -- no block-level statistic is ever
#:   combined into a pool-level one;
#: * ``l1_fit`` is likewise recomputed from a concatenated per-column ``primal``
#:   vector and summed once; ``gamut_violation_rate`` is rebuilt from integer
#:   counts;
#: * **the ADMM trajectory is not preserved.**  ``core.admm_lad`` adapts ``rho``
#:   on the batch-global residual norms (``r_pri``/``r_dual`` are Frobenius norms
#:   over *all* columns) and stops on a batch-global tolerance, so a block sees a
#:   different ``rho`` schedule and a different iterate than the whole pool does.
#:   The exact LAD minimiser is column-separable, the finite-iteration ADMM
#:   iterate is not.  ``core.py`` is frozen this round, so this is recorded, not
#:   patched: every block's ``admm_iters`` / ``admm_rho`` / ``gap_max`` /
#:   ``r_pri`` / ``r_dual`` is written to ``lut_chunk_solver`` on the row.
LUT_CHUNK_CONTRACT = {
    "split": "the pool's lut_id list, in registry order, into blocks of at most "
             "--lut-chunk ids; colours, design matrix, basis and ADMM "
             "parameters are untouched",
    "pool_sha": "asserted on the full pool before the split",
    "de00_blocks": "the per-block predictions are assembled into the whole-pool "
                   "(Q, 3L) field first, then run.de00_of / de00_paired and "
                   "run._stats are called once on the whole pool, exactly as a "
                   "whole-pool run does; no block-level statistic is combined, "
                   "and no dE00 is evaluated on a solver-block-shaped batch "
                   "(colorimetry.srgb_to_xyz is not batch-shape invariant)",
    "l1_fit": "recomputed as |rhat - r|.sum(0).sum() on the assembled "
              "whole-pool prediction (admm_lad's own resid at the whole-pool "
              "shape), not summed from the block L1s: torch's dim-0 reduction "
              "vectorises across the column axis, so a per-block sum is a "
              "different accumulation order",
    "gamut_violation_rate": "rebuilt from integer violation counts",
    "gap_mean": "block gap means, weighted by block column count",
    "r_pri / r_dual": "sqrt of the sum of the block squares (Frobenius over the "
                      "whole batch)",
    "tie_gap_rel": "pooled (L1 - L1_lam/10) / L1_lam/10, reconstructed from the "
                   "per-block ratios; the per-block values are in "
                   "lut_chunk_solver",
    "admm_iters / admm_rho / dual_feas": "max over blocks; per block in "
                                         "lut_chunk_solver",
    "not_preserved": "the ADMM trajectory: core.admm_lad adapts rho and stops "
                     "on batch-global residual norms, so a block's iterate is "
                     "not the whole pool's iterate.  The exact LAD minimiser is "
                     "column-separable; the finite-iteration ADMM iterate is "
                     "not.",
}

#: per-block solver diagnostics copied verbatim onto ``lut_chunk_solver``
_BLOCK_DIAG = ("admm_iters", "admm_rho", "gap_max", "gap_mean", "dual_feas",
               "r_pri", "r_dual", "fit_seconds", "l1_fit", "tie_gap_rel",
               "gamut_violation_rate")


def lut_bounds(n_lut: int, chunk: int | None) -> list[tuple[int, int]]:
    """``[(lo, hi), ...]`` half-open pool blocks of at most ``chunk`` ids.

    ``chunk <= 0`` (the default) and ``chunk >= n_lut`` both give the single
    block ``[(0, n_lut)]``, which routes through the *unchanged* whole-pool code
    path -- ``--lut-chunk 0`` is therefore bit-for-bit the behaviour that is
    already on the board.
    """
    if n_lut <= 0:
        raise ValueError(f"n_lut must be >= 1, got {n_lut}")
    if chunk is None or chunk <= 0 or chunk >= n_lut:
        return [(0, n_lut)]
    return [(s, min(s + chunk, n_lut)) for s in range(0, n_lut, chunk)]


def pooled_rhat(f: Tensor, thetas: Sequence[tuple[int, int, Tensor]], *,
                n_lut: int) -> Tensor:
    """``(Q, 3L)`` whole-pool prediction, assembled from the per-block thetas.

    The pooled field is materialised **before** any dE00 call, so
    ``run.de00_of`` / :func:`de00_paired` always see the whole pool in one call
    and tile it by their own fixed 64-LUT block, exactly as a whole-pool run
    does.  Calling the dE00 once per *solver* block instead would make the
    reported statistic a function of ``--lut-chunk``:
    ``colorimetry.srgb_to_xyz``'s ``lin @ M^T`` is not batch-shape invariant
    (measured on CPU fp32: a 3-LUT block and an 8-LUT block disagree by 3.1e-5
    dE00, ~3 ulp at dE00 ~ 10), which is the same blocking sensitivity
    :func:`de00_paired` already guards with its ``.contiguous()``.

    Column ``3l + ch`` is written at its pool position, so the assembled tensor
    is the one a whole-pool fit would have produced from the same coefficients.
    """
    if len(thetas) == 1:
        return f @ thetas[0][2]
    out = f.new_empty((f.shape[0], 3 * n_lut))
    for lo, hi, th in thetas:
        out[:, 3 * lo:3 * hi] = f @ th
    return out


def _pooled_de_into(R, rec: dict, tag: str, xg: Tensor, rg: Tensor,
                    rhat: Tensor,
                    strata: dict[str, Tensor] | None) -> None:
    """Both calibers x pre/post clamp (+ the strata split) for one grid."""
    for clamp in (False, True):
        ct = "post_clamp" if clamp else "pre_clamp"
        for caliber, fn in (("legacy", R.de00_of), ("paired", de00_paired)):
            de = fn(xg, rg, rhat, clamp=clamp)
            rec[f"{tag}_{ct}_{caliber}"] = R._stats(de)
            if strata is not None and tag == "fitgrid" and not clamp:
                rec[f"fitgrid_strata_{caliber}"] = {
                    k: R._stats(de[mask]) for k, mask in strata.items()
                    if int(mask.sum()) > 0}
            del de


def _merge_block_scalars(recs: Sequence[dict], bounds: Sequence[tuple[int, int]],
                         *, q_fit: int) -> dict:
    """Pool-level scalars from the per-block ``fit_and_score`` records.

    ``l1_fit`` / ``l1_fit_mean_per_point`` are *not* set here -- they are
    recomputed from the concatenated ``primal`` vector by
    :func:`fit_score_pooled`, which is a single reduction over the whole pool.
    """
    widths = [hi - lo for lo, hi in bounds]
    p_feat = {r["P_feat"] for r in recs}
    if len(p_feat) != 1:
        raise AssertionError(
            f"the pool blocks were fitted on different designs: P_feat={p_feat}")
    rec = dict(recs[0])
    for k in DE00_STAT_KEYS:
        rec.pop(k, None)
    n_col = float(sum(widths))
    rec["admm_iters"] = max(int(r["admm_iters"]) for r in recs)
    rec["admm_rho"] = max(float(r["admm_rho"]) for r in recs)
    rec["gap_max"] = max(float(r["gap_max"]) for r in recs)
    rec["dual_feas"] = max(float(r["dual_feas"]) for r in recs)
    rec["gap_mean"] = sum(float(r["gap_mean"]) * w
                          for r, w in zip(recs, widths)) / n_col
    rec["r_pri"] = math.sqrt(sum(float(r["r_pri"]) ** 2 for r in recs))
    rec["r_dual"] = math.sqrt(sum(float(r["r_dual"]) ** 2 for r in recs))
    rec["fit_seconds"] = sum(float(r["fit_seconds"]) for r in recs)
    # integer counts, so the pooled rate is the whole-pool rate exactly
    viol = n_tot = 0
    for r, w in zip(recs, widths):
        n = q_fit * w
        viol += int(round(float(r["gamut_violation_rate"]) * n))
        n_tot += n
    rec["gamut_violation_rate"] = viol / max(n_tot, 1)
    if "tie_gap_rel" in recs[0]:
        # tie_gap_rel = (L1 - base) / base with base the lam/10 solve's L1;
        # both terms are separable, so the pooled ratio is rebuilt from the
        # per-block L1 and the per-block ratio.
        l1 = sum(float(r["l1_fit"]) for r in recs)
        base = sum(float(r["l1_fit"]) / (1.0 + float(r["tie_gap_rel"]))
                   for r in recs)
        rec["tie_gap_rel"] = (l1 - base) / max(base, 1e-30)
    return rec


def fit_score_pooled(R, *, x_fit, x_eval, r_fit, r_eval, build_fit, build_eval,
                     meta: dict, max_iter: int,
                     strata: dict[str, Tensor] | None = None,
                     tie_probe: bool = False, lut_chunk: int = 0
                     ) -> tuple[dict, list[tuple[int, int, Tensor]]]:
    """``fit_and_score`` + :func:`paired_de00_stats`, over pool blocks.

    Returns ``(rec, thetas)`` with ``thetas`` the per-block
    ``(lut_lo, lut_hi, theta)`` in registry order.  A single block takes the
    whole-pool path unchanged (the two calls the arms already make, in the same
    order); several blocks take the assembled path described by
    :data:`LUT_CHUNK_CONTRACT`.
    """
    n_lut = r_fit.shape[1] // 3
    bounds = lut_bounds(n_lut, lut_chunk)
    if len(bounds) == 1:
        rec = R.fit_and_score(x_fit, x_eval, r_fit, r_eval, build_fit,
                              build_eval, meta, max_iter=max_iter,
                              strata=strata, tie_probe=tie_probe,
                              return_theta=True)
        theta = rec.pop("_theta")
        paired_de00_stats(R, rec, x_fit=x_fit, x_eval=x_eval, r_fit=r_fit,
                          r_eval=r_eval, theta=theta, build_fit=build_fit,
                          build_eval=build_eval, strata=strata)
        return rec, [(0, n_lut, theta)]

    recs: list[dict] = []
    thetas: list[tuple[int, int, Tensor]] = []
    for lo, hi in bounds:
        rb = r_fit[:, 3 * lo:3 * hi].contiguous()
        reb = r_eval[:, 3 * lo:3 * hi].contiguous()
        rc = R.fit_and_score(x_fit, x_eval, rb, reb, build_fit, build_eval,
                             meta, max_iter=max_iter, strata=strata,
                             tie_probe=tie_probe, return_theta=True)
        thetas.append((lo, hi, rc.pop("_theta")))
        recs.append(rc)
        del rb, reb
        torch.cuda.empty_cache()

    rec = _merge_block_scalars(recs, bounds, q_fit=x_fit.shape[0])
    rec["lut_chunk_solver"] = [
        {"block": i, "lut_lo": lo, "lut_hi": hi, "n_lut": hi - lo,
         **{k: rc[k] for k in _BLOCK_DIAG if k in rc}}
        for i, ((lo, hi), rc) in enumerate(zip(bounds, recs))]

    f = build_fit()
    rhat = pooled_rhat(f, thetas, n_lut=n_lut)
    del f
    _pooled_de_into(R, rec, "fitgrid", x_fit, r_fit, rhat, strata)
    # ``primal`` last, in place, on the pooled prediction that is about to be
    # freed: ``rhat - r_fit`` is ``admm_lad``'s own ``resid`` at the whole-pool
    # shape, so ``l1_fit`` is the same single reduction a whole-pool fit reports
    # -- and it costs no extra memory, which matters because ``rhat`` is already
    # the size of ``r_fit``.
    l1 = rhat.sub_(r_fit).abs_().sum(0).sum()
    rec["l1_fit"] = float(l1)
    rec["l1_fit_mean_per_point"] = float(l1 / r_fit.numel())
    del rhat, l1
    torch.cuda.empty_cache()
    f = build_eval()
    rhat = pooled_rhat(f, thetas, n_lut=n_lut)
    del f
    _pooled_de_into(R, rec, "heldgrid", x_eval, r_eval, rhat, None)
    del rhat
    torch.cuda.empty_cache()
    return rec, thetas


def lut_chunk_fields(ids: Sequence[str], chunk: int) -> dict:
    """The top-level ``lut_chunk`` record every arm artefact carries."""
    bounds = lut_bounds(len(ids), chunk)
    out: dict[str, object] = {"lut_chunk": int(chunk or 0)}
    if len(bounds) > 1:
        out["lut_chunk_aggregation"] = {
            "n_lut": len(ids), "n_blocks": len(bounds),
            "block_sizes": [hi - lo for lo, hi in bounds],
            **LUT_CHUNK_CONTRACT}
    return out


# --------------------------------------------------------------------------- #
# shared: the §10.2 reference ladder
# --------------------------------------------------------------------------- #
def reference_ladder(out: str | Path, pool: str, *, field: str = "p95",
                     grid: str = "fitgrid") -> dict:
    """The published §10.2 rows for ``pool``: from disk if present, else §10.2.

    On-disk rows are only accepted after their ``lut_ids_sha`` matches §10's, so
    a ladder rerun on a drifted pool cannot silently become the reference.
    """
    path = Path(out) / f"ladder_{pool}.json"
    if path.exists():
        doc = json.loads(path.read_text())
        want = POOL_SHA.get(pool, (None, None))[0]
        if want is not None and doc.get("lut_ids_sha") != want:
            raise AssertionError(
                f"{path} carries lut_ids_sha {doc.get('lut_ids_sha')}, §10 "
                f"registered {want}")
        rows = [{"p": r["p"], "P_dyn": r["P_dyn"],
                 "value": float(r[f"{grid}_pre_clamp"][field])}
                for r in doc["rows"]]
        return {"source": str(path), "c_star": doc.get("c_star"),
                "lut_ids_sha": doc.get("lut_ids_sha"), "field": field,
                "grid": grid, "rows": rows}
    tbl = REFERENCE_LADDER_P95.get(pool, {}).get(grid)
    if tbl is None or field != "p95":
        raise FileNotFoundError(
            f"no reference ladder for pool={pool} field={field} grid={grid}: "
            f"{path} is absent and PROPOSAL §10.2 is only transcribed for "
            "held_out / p95 / fitgrid")
    rows = [{"p": 1 if i < _N_P1_REF else 2, "P_dyn": pd, "value": v}
            for i, (pd, v) in enumerate(tbl)]
    return {"source": "PROPOSAL §10.2 (transcribed)", "c_star": 0.7,
            "lut_ids_sha": POOL_SHA[pool][0], "field": field, "grid": grid,
            "rows": rows}


def interp_reference(ref: dict, p: int, p_dyn: int) -> float | None:
    """log-P interpolation of the reference order-``p`` curve at ``p_dyn``."""
    pts = [(r["P_dyn"], r["value"]) for r in ref["rows"] if r["p"] == p]
    return _interp_log(pts, float(p_dyn))


def bracket_rows(ref: dict, p: int, p_dyn: int) -> list[dict]:
    """The two measured reference rows that bracket ``p_dyn`` (printed as-is)."""
    pts = sorted((r for r in ref["rows"] if r["p"] == p),
                 key=lambda r: r["P_dyn"])
    lo = [r for r in pts if r["P_dyn"] <= p_dyn]
    hi = [r for r in pts if r["P_dyn"] >= p_dyn]
    return [r for r in ([lo[-1]] if lo else []) + ([hi[0]] if hi else [])]


def reference_or_omitted(out: str | Path, pool: str, *, n_rows: int
                         ) -> tuple[dict | None, dict | None]:
    """``(reference_ladder, None)`` only when the row pool *is* the registered pool.

    ``--n-lut`` cuts the row pool for smoke runs while the published ladder is
    the full pool (902 / 259 / 3,149 / 4,051), so printing the two side by side
    puts an 8-LUT row next to a 902-LUT reference and every derived column
    (``rel_excess``,
    ``ref_*_interp_p95``, the bracket) is arithmetic between two different
    corpora.  In that case nothing referential is written at all: the artefact
    carries ``reference_omitted`` with the reason instead.
    """
    if pool not in POOL_SHA:
        return None, {"reason": f"pool {pool!r} is not a PROPOSAL-registered "
                                "pool, so there is no reference ladder for it",
                      "pool": pool, "n_lut_rows": int(n_rows)}
    want_n = POOL_SHA[pool][1]
    if n_rows != want_n:
        return None, {
            "reason": f"the row pool was cut to n={n_rows} (--n-lut) while the "
                      f"published reference ladder for {pool!r} is the full "
                      f"n={want_n} pool; a subset row next to a full-pool "
                      "reference makes every derived column meaningless, so no "
                      "reference or derived column is written",
            "pool": pool, "n_lut_rows": int(n_rows), "n_lut_reference": want_n}
    return reference_ladder(out, pool), None


def _ref_fields(ref: dict | None, omitted: dict | None) -> dict:
    """Exactly one of ``reference_ladder`` / ``reference_omitted`` in the JSON."""
    return {"reference_ladder": ref} if ref is not None \
        else {"reference_omitted": omitted}


# --------------------------------------------------------------------------- #
# drivers
# --------------------------------------------------------------------------- #
def _prep(args):
    """Pools + colours + residuals, all from ``run.py``'s own loaders."""
    from q3vl.whatb.jetlut import run as R

    pin_dataset_version(args.dataset_version)
    dev, dt = torch.device(args.device), torch.float64
    bank = R.open_bank()
    registry = R.all_pools(args.n_train)
    x = R.fit_colours(dt).to(dev)
    xe = R.eval_colours(dt).to(dev)
    return R, dev, dt, bank, registry, x, xe


def _pool_ids(registry, pool: str, n_lut: int | None) -> tuple[list[str], dict]:
    ids = registry[pool]
    sha_rec = assert_pool_sha(pool, ids)          # full pool, before any cut
    if n_lut is not None and n_lut < len(ids):
        ids = ids[:n_lut]
        sha_rec = {**sha_rec, "smoke_subset_n": len(ids),
                   "smoke_subset_sha": sha16(ids)}
    return ids, sha_rec


def _de00_columns(rec: dict, grid: str, field: str) -> str:
    """``mean/p95`` of both calibers, for the progress line."""
    return (f"{rec[f'{grid}_pre_clamp_legacy'][field]:.4f}L/"
            f"{rec[f'{grid}_pre_clamp_paired'][field]:.4f}P")


def cmd_jgate(args) -> None:
    """Arm 1: AFFONLY's learned shared geometry as the gate, p1 jet basis."""
    R, dev, dt, bank, registry, x, xe = _prep(args)
    geo, prov = load_shared_geometry(args.checkpoint, device=dev, dtype=dt)
    gate_check = assert_gate_matches_glut(geo, n_probe=args.n_gate_probe)
    n = prov["n_gauss"]
    p_dyn = 3 * (4 + 4 * n)
    print(f"[jgate] N={n}  P_dyn={p_dyn}  gate cross-check ok "
          f"({gate_check['n_probe']} colours, max|dw|="
          f"{gate_check['max_abs_diff']:.3e} <= tol {gate_check['tol']:g}, "
          f"bitwise={gate_check['bitwise_equal']})", flush=True)

    strata = {k: v.to(dev) for k, v in R.colour_strata(x).items()}
    for pool in args.pool:
        ids, sha_rec = _pool_ids(registry, pool, args.n_lut)
        r = R.residuals(bank, ids, x, dev, dt)
        re = R.residuals(bank, ids, xe, dev, dt)
        st = dict(strata)
        st["high_curvature"] = R.curvature_mask(r, R.GRID_FIT)
        st["rest"] = ~(st["high_curvature"] | st["boundary"]
                       | st["red_yellow"] | st["highlight"])
        meta = {"arm": "J-GATE-AFF", "gate": "affonly_shared_geometry",
                "basis": "p1", "m": None, "p": 1, "c": None, "N": n,
                "P_dyn": p_dyn, "pool": pool}
        build = lambda: design_matrix_gate(x, geo)        # noqa: E731
        build_e = lambda: design_matrix_gate(xe, geo)     # noqa: E731
        rec, thetas = fit_score_pooled(
            R, x_fit=x, x_eval=xe, r_fit=r, r_eval=re, build_fit=build,
            build_eval=build_e, meta=meta, max_iter=args.max_iter, strata=st,
            tie_probe=True, lut_chunk=args.lut_chunk)
        # a wiring check on the carrier, so it runs once per config on the first
        # block's theta -- not once per pool block
        rec["carrier_check"] = assert_carrier_matches_glut(x, geo, thetas[0][2])
        ref, omitted = reference_or_omitted(args.out, pool, n_rows=len(ids))
        if ref is not None:
            rec["reference_p1_bracket"] = bracket_rows(ref, 1, p_dyn)
            rec["reference_p1_interp_p95"] = interp_reference(ref, 1, p_dyn)
            rec["reference_de00_caliber"] = "legacy"
        print(f"[jgate:{pool}] n_lut={len(ids)} {rec['fit_seconds']:.1f}s "
              f"gap={rec['gap_max']:.1e} pre legacy/paired mean="
              f"{_de00_columns(rec, 'fitgrid', 'mean')} p95="
              f"{_de00_columns(rec, 'fitgrid', 'p95')} eval p95="
              f"{_de00_columns(rec, 'heldgrid', 'p95')}", flush=True)
        _dump(args.out, f"jgate_{pool}{args.tag}.json",
              {"seed": R.SEED, "arm": "J-GATE-AFF", "pool_check": sha_rec,
               "grid_fit": R.GRID_FIT, "grid_eval": R.GRID_EVAL,
               "n_eval_points": R.N_EVAL_POINTS,
               "checkpoint": prov, "gate_check": gate_check,
               "de00_calibers": DE00_CALIBERS,
               **lut_chunk_fields(ids, args.lut_chunk),
               **_ref_fields(ref, omitted),
               "strata_sizes": {k: int(v.sum()) for k, v in st.items()},
               "rows": [rec]})
        _dump(args.out, f"jgate_{pool}{args.tag}.DONE.json",
              {"pool": pool, "n_lut": len(ids), "P_dyn": p_dyn})
        del r, re
        torch.cuda.empty_cache()


def cmd_j4d(args) -> None:
    """Arm 2: the 4-D ``(x, s)`` atlas -- what the s axis costs the carrier."""
    R, dev, dt, bank, registry, x, xe = _prep(args)
    s_values = list(args.s_values)
    xs_fit = stack_s(x, s_values)
    xs_eval = stack_s(xe, s_values)
    # the design matrices are built from the 4-D rows (closed over below), but
    # every metric in ``fit_and_score`` -- dE00 and the gamut count -- consumes
    # the *colour* of each row, so it gets the 3-channel projection.  Passing the
    # 4-D tensor here broadcasts (Q,1,4) against (Q,l,3) and raises.
    xc_fit = xs_fit[:, :3].contiguous()
    xc_eval = xs_eval[:, :3].contiguous()
    # NOTES §4: the exact 4-D partition of unity is asserted *in the arm*, on
    # 4,096 random (x, s) points per boarded atlas, before any fit runs.
    pou_checks = {str(m): assert_pou_4d(
        Atlas4D(m=m, m_s=args.m_s, c=args.c_star_num),
        n_probe=args.n_pou_probe) for m in args.m}
    for m, chk in pou_checks.items():
        print(f"[j4d] POU check m={m} m_s={args.m_s} N4={chk['N4']}: "
              f"max|sum-1|={chk['max_abs_dev_from_one']:.3e} < {chk['tol']:g}",
              flush=True)
    for pool in args.pool:
        ids, sha_rec = _pool_ids(registry, pool, args.n_lut)
        r = R.residuals(bank, ids, x, dev, dt)
        re = R.residuals(bank, ids, xe, dev, dt)
        r4 = stack_s_residual(r, s_values)
        re4 = stack_s_residual(re, s_values)
        ref, omitted = reference_or_omitted(args.out, pool, n_rows=len(ids))
        rows = []
        for m in args.m:
            atlas = Atlas4D(m=m, m_s=args.m_s, c=args.c_star_num)
            p_dyn = n_dynamic_params_4d(atlas)
            meta = {"arm": "J-4D", "m": m, "m_s": args.m_s, "N4": atlas.n,
                    "c": atlas.c, "sigma_x": atlas.sigma_x,
                    "sigma_s": atlas.sigma_s, "P_dyn": p_dyn,
                    "s_values": s_values, "pool": pool}
            build = (lambda a=atlas: design_matrix_4d(xs_fit, a,
                                                      chunk=args.chunk))
            build_e = (lambda a=atlas: design_matrix_4d(xs_eval, a,
                                                        chunk=args.chunk))
            rec, thetas = fit_score_pooled(
                R, x_fit=xc_fit, x_eval=xc_eval, r_fit=r4, r_eval=re4,
                build_fit=build, build_eval=build_e, meta=meta,
                max_iter=args.max_iter, tie_probe=True,
                lut_chunk=args.lut_chunk)
            # the POU assertion is a per-atlas one-shot check run before any fit
            # (above); the row only carries its record, it is not re-run per block
            rec["pou_check"] = pou_checks[str(m)]
            # NOTES §2.9: ``l1_fit_mean_per_point`` divides by the *stacked*
            # element count 5Q*3L, the 3-D ladder's own column divides by Q*3L.
            rec["l1_fit_mean_per_colour_point"] = (
                rec["l1_fit"] / max(r.numel(), 1))
            rec["l1_denominators"] = {
                "l1_fit_mean_per_point": {"denominator": int(r4.numel()),
                                          "form": "len(s_values)*Q*3L"},
                "l1_fit_mean_per_colour_point": {"denominator": int(r.numel()),
                                                 "form": "Q*3L"}}
            rec["per_s"] = _per_s_stats(R, x, xe, r, re, thetas, atlas,
                                        s_values, chunk=args.chunk)
            if ref is not None and "1.0" in rec["per_s"]:
                rec["reference_p1_bracket"] = bracket_rows(ref, 1, p_dyn)
                rec["reference_p1_interp_p95"] = interp_reference(ref, 1, p_dyn)
                y1 = rec["reference_p1_interp_p95"]
                y4 = rec["per_s"]["1.0"]["fitgrid_pre_clamp_legacy"]["p95"]
                rec["s1_p95_vs_reference_p1"] = {
                    "jet4d_s1_p95": y4, "ref_p1_interp_p95": y1,
                    "rel_excess": (y4 - y1) / y1 if y1 else math.nan,
                    "de00_caliber": "legacy"}
            rows.append(rec)
            print(f"[j4d:{pool}] m={m} m_s={args.m_s} N4={atlas.n} P={p_dyn} "
                  f"{rec['fit_seconds']:.1f}s gap={rec['gap_max']:.1e} "
                  + "  ".join(
                      "s={:g}:p95={:.4f}L/{:.4f}P".format(
                          s,
                          rec['per_s'][f'{float(s)}']['fitgrid_pre_clamp_legacy']['p95'],
                          rec['per_s'][f'{float(s)}']['fitgrid_pre_clamp_paired']['p95'])
                      for s in s_values), flush=True)
            _dump(args.out, f"j4d_{pool}{args.tag}.json",
                  {"seed": R.SEED, "arm": "J-4D", "pool_check": sha_rec,
                   "grid_fit": R.GRID_FIT, "grid_eval": R.GRID_EVAL,
                   "n_eval_points": R.N_EVAL_POINTS, "s_values": s_values,
                   "de00_calibers": DE00_CALIBERS, "pou_check": pou_checks,
                   **lut_chunk_fields(ids, args.lut_chunk),
                   **_ref_fields(ref, omitted),
                   "proposition_explicit_gate_cost": PROP_EXPLICIT_GATE,
                   "rows": rows})
        _dump(args.out, f"j4d_{pool}{args.tag}.DONE.json",
              {"pool": pool, "n_lut": len(ids), "n_rows": len(rows)})
        del r, re, r4, re4
        torch.cuda.empty_cache()


#: NOTES proposition, restated in the artefact so a reader of the JSON alone
#: still has the analytic anchor the s-axis cost is measured against.
PROP_EXPLICIT_GATE = (
    "For the explicit-gate carrier F(x,s) = x + s Phi(x) theta with s >= 0, "
    "sum_{x,s} |s (Phi(x) theta - r(x))| = (sum_s s) * sum_x |Phi(x) theta - "
    "r(x)|, so the argmin over theta does not depend on the s sampling and the "
    "cost of the s axis is identically 0."
)


def _per_s_stats(R, x, xe, r, re, thetas: Sequence[tuple[int, int, Tensor]],
                 atlas: Atlas4D, s_values: Sequence[float], *,
                 chunk: int) -> dict:
    """dE00 per strength slice, on both grids, pre-/post-clamp, both calibers.

    ``thetas`` is the ``(lut_lo, lut_hi, theta)`` list :func:`fit_score_pooled`
    returns; each slice's ``(Q, L)`` field is assembled over the pool blocks and
    reduced by ``run._stats`` once, so a chunked run's per-s statistics are the
    same pooled reduction a whole-pool run performs.
    """
    out: dict[str, dict] = {}
    for s in s_values:
        rec: dict[str, dict] = {}
        for tag, xg, rg in (("fitgrid", x, r), ("heldgrid", xe, re)):
            f = design_matrix_4d(stack_s(xg, [s]), atlas, chunk=chunk)
            rhat = pooled_rhat(f, thetas, n_lut=rg.shape[1] // 3)
            tgt = float(s) * rg
            for clamp in (False, True):
                key = f"{tag}_{'post_clamp' if clamp else 'pre_clamp'}"
                for caliber, fn in (("legacy", R.de00_of),
                                    ("paired", de00_paired)):
                    de = fn(xg, tgt, rhat, clamp=clamp)
                    rec[f"{key}_{caliber}"] = R._stats(de)
                    del de
            del f, rhat, tgt
        out[f"{float(s)}"] = rec
    return out


def cmd_jp15(args) -> None:
    """Arm 3: p1 / p15 / p2 on one atlas -- the cross-term share of the p2 gain."""
    R, dev, dt, bank, registry, x, xe = _prep(args)
    strata = {k: v.to(dev) for k, v in R.colour_strata(x).items()}
    for pool in args.pool:
        ids, sha_rec = _pool_ids(registry, pool, args.n_lut)
        r = R.residuals(bank, ids, x, dev, dt)
        re = R.residuals(bank, ids, xe, dev, dt)
        st = dict(strata)
        st["high_curvature"] = R.curvature_mask(r, R.GRID_FIT)
        st["rest"] = ~(st["high_curvature"] | st["boundary"]
                       | st["red_yellow"] | st["highlight"])
        ref, omitted = reference_or_omitted(args.out, pool, n_rows=len(ids))
        rows = []
        for m in args.m:
            atlas = Atlas(m=m, c=args.c_star_num)
            for order in args.orders:
                p_dyn = n_dynamic_params_order(m, order)
                meta = {"arm": "J-P15", "m": m, "order": order,
                        "p": {"p1": 1, "p15": 1.5, "p2": 2}[order],
                        "c": atlas.c, "N": atlas.n, "h": atlas.h,
                        "sigma": atlas.sigma, "P_dyn": p_dyn, "pool": pool}
                build = (
                    (lambda a=atlas: design_matrix(x, a, 1)) if order == "p1"
                    else (lambda a=atlas: design_matrix(x, a, 2)) if order == "p2"
                    else (lambda a=atlas: design_matrix_p15(x, a)))
                build_e = (
                    (lambda a=atlas: design_matrix(xe, a, 1)) if order == "p1"
                    else (lambda a=atlas: design_matrix(xe, a, 2)) if order == "p2"
                    else (lambda a=atlas: design_matrix_p15(xe, a)))
                rec, thetas = fit_score_pooled(
                    R, x_fit=x, x_eval=xe, r_fit=r, r_eval=re,
                    build_fit=build, build_eval=build_e, meta=meta,
                    max_iter=args.max_iter, strata=st,
                    tie_probe=(order == "p15"), lut_chunk=args.lut_chunk)
                del thetas
                rows.append(rec)
                print(f"[jp15:{pool}] m={m} {order} P={p_dyn:5d} "
                      f"{rec['fit_seconds']:.1f}s gap={rec['gap_max']:.1e} "
                      f"l1/pt={rec['l1_fit_mean_per_point']:.6f} pre p95="
                      f"{_de00_columns(rec, 'fitgrid', 'p95')}", flush=True)
                _dump(args.out, f"jp15_{pool}{args.tag}.json",
                      {"seed": R.SEED, "arm": "J-P15", "pool_check": sha_rec,
                       "grid_fit": R.GRID_FIT, "grid_eval": R.GRID_EVAL,
                       "n_eval_points": R.N_EVAL_POINTS, "c_star": atlas.c,
                       "de00_calibers": DE00_CALIBERS,
                       **lut_chunk_fields(ids, args.lut_chunk),
                       **_ref_fields(ref, omitted), "rows": rows})
        nested = assert_nested_three(rows, orders=args.orders, n_m=len(args.m))
        _dump(args.out, f"jp15_{pool}{args.tag}.json",
              {"seed": R.SEED, "arm": "J-P15", "pool_check": sha_rec,
               "grid_fit": R.GRID_FIT, "grid_eval": R.GRID_EVAL,
               "n_eval_points": R.N_EVAL_POINTS, "c_star": args.c_star_num,
               "de00_calibers": DE00_CALIBERS,
               **lut_chunk_fields(ids, args.lut_chunk),
               **_ref_fields(ref, omitted), "nested": nested,
               "p15_share_of_p2_gain_legacy": _p15_share(rows, ref, "legacy"),
               "p15_share_of_p2_gain_paired": _p15_share(rows, None, "paired"),
               "strata_sizes": {k: int(v.sum()) for k, v in st.items()},
               "rows": rows})
        _dump(args.out, f"jp15_{pool}{args.tag}.DONE.json",
              {"pool": pool, "n_lut": len(ids), "n_rows": len(rows),
               "orders": list(args.orders), "n_nested_checks": len(nested),
               "nested_ok": bool(nested) and all(q["ok"] for q in nested)})
        for q in nested:
            print(f"  nested m={q['m']}: p1={q['l1_p1']:.6f} "
                  f"p15={q['l1_p15']:.6f} p2={q['l1_p2']:.6f}  ok", flush=True)
        del r, re
        torch.cuda.empty_cache()


def _p15_share(rows: Sequence[dict], ref: dict | None,
               caliber: str = "legacy") -> list[dict]:
    """``(p1 - p15) / (p1 - p2)`` on the fit-grid p95, at matched budget.

    p1 and p2 are read at ``P_dyn(p15)`` by log-P interpolation of *this run's*
    own p1 / p2 rows, so all three curves come from the same solver and the same
    pool; the published §10.2 ladder is carried alongside for reference only,
    and only on the ``legacy`` caliber -- it is the caliber §10 was measured in.
    """
    key = f"fitgrid_pre_clamp_{caliber}"

    def curve(order):
        return sorted((r["P_dyn"], r[key]["p95"])
                      for r in rows if r["order"] == order)

    c1, c2 = curve("p1"), curve("p2")
    out = []
    for r15 in sorted((r for r in rows if r["order"] == "p15"),
                      key=lambda r: r["P_dyn"]):
        pd = r15["P_dyn"]
        y1, y2 = _interp_log(c1, float(pd)), _interp_log(c2, float(pd))
        y15 = r15[key]["p95"]
        rec = {"m": r15["m"], "P_dyn": pd, "de00_caliber": caliber,
               "p15_p95": y15, "p1_interp_p95": y1, "p2_interp_p95": y2}
        if y1 is not None and y2 is not None and abs(y1 - y2) > 0:
            rec["share_of_p2_gain"] = (y1 - y15) / (y1 - y2)
        if ref is not None:
            rec["ref_p1_interp_p95"] = interp_reference(ref, 1, pd)
            rec["ref_p2_interp_p95"] = interp_reference(ref, 2, pd)
        out.append(rec)
    return out


def row_config_set(obj) -> set[tuple] | None:
    """``{(m, order)}`` of an artefact's ``rows``; ``None`` if it has no rows.

    ``(m, order)`` is the pair that identifies a ladder row across the three
    arms (``order`` is ``None`` outside J-P15), so it is what distinguishes "the
    same run, one more row" from "a different run's rows".
    """
    if not isinstance(obj, dict):
        return None
    rows = obj.get("rows")
    if not isinstance(rows, list):
        return None
    return {(r.get("m"), r.get("order")) for r in rows if isinstance(r, dict)}


def _mtime_stamp(path: Path) -> str:
    """``YYYYmmdd-HHMMSS`` of the file's own mtime -- reproducible from the file.

    Never a wall clock read: the stamp has to be recoverable by anybody looking
    at the renamed file later, and a clock read is not.
    """
    return time.strftime("%Y%m%d-%H%M%S", time.localtime(path.stat().st_mtime))


def _dump(out: str, name: str, obj, *, ts: str | None = None) -> Path | None:
    """Write ``obj``; move a *non-superseded* previous artefact aside first.

    2026-08-20: a single ``--m 6`` run wrote ``j4d_held_out.json`` on top of the
    m=3/4/5 artefact of the previous run and the three rows were gone -- the
    driver dumps the whole file on every row, so a narrower run silently
    replaces a wider one.  The guard: if the file on disk carries a ``rows``
    configuration ``(m, order)`` that this write does **not** carry, the old file
    is renamed ``<stem>.overwritten-<ts>.json`` first (``ts`` = the old file's
    own mtime unless the caller passes one).  A run's own incremental dumps grow
    the row set, so ``old ⊆ new`` and nothing is renamed.

    Returns the path the old file was moved to, or ``None``.
    """
    d = Path(out)
    d.mkdir(parents=True, exist_ok=True)
    path = d / name
    moved = None
    if path.exists():
        try:
            old = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            old = None
        old_cfg, new_cfg = row_config_set(old), row_config_set(obj)
        if old_cfg is not None and new_cfg is not None \
                and not old_cfg <= new_cfg:
            moved = d / f"{path.stem}.overwritten-{ts or _mtime_stamp(path)}.json"
            path.rename(moved)
            print(f"[dump] {name}: on-disk rows {sorted(map(str, old_cfg))} are "
                  f"not covered by this write {sorted(map(str, new_cfg))}; "
                  f"kept as {moved.name}", flush=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str))
    return moved


def add_arguments(ap: argparse.ArgumentParser) -> None:
    """Flags the three ablation subcommands add on top of ``run.py``'s."""
    ap.add_argument("--dataset-version", default=DATASET_VERSION,
                    choices=tuple(S.DATASET_VERSION_CHOICES),
                    help="pinned before any index read; §10's pools need "
                         "v20260804")
    ap.add_argument("--checkpoint", default=AFFONLY_CKPT,
                    help="AFFONLY run whose SharedGeometry is the arm-1 gate")
    ap.add_argument("--n-gate-probe", type=int, default=4096)
    ap.add_argument("--n-lut", type=int, default=None,
                    help="cut the pool to its first n ids (smoke only; the sha "
                         "assertion still runs on the full pool)")
    ap.add_argument("--n-pou-probe", type=int, default=4096,
                    help="random (x, s) points the J-4D POU assertion runs on")
    ap.add_argument("--m-s", type=int, default=3, help="J-4D strength cells")
    ap.add_argument("--s-values", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--orders", nargs="+", default=["p1", "p15", "p2"],
                    choices=tuple(MONOMIALS))
    ap.add_argument("--chunk", type=int, default=2048,
                    help="J-4D design-matrix point block")
    ap.add_argument("--lut-chunk", type=int, default=0,
                    help="solve the pool in blocks of at most N LUT ids "
                         "(registry order); 0 = whole pool, the behaviour "
                         "already on the board.  Only the ADMM batch is split "
                         "-- colours, basis, design matrix and ADMM parameters "
                         "are untouched, the pool sha is asserted on the full "
                         "pool before the split, and every pooled statistic is "
                         "recomputed from the concatenated per-LUT field.  See "
                         "ablation_arms.LUT_CHUNK_CONTRACT for what the split "
                         "does not preserve (the ADMM rho schedule).")


COMMANDS = {"jgate": cmd_jgate, "j4d": cmd_j4d, "jp15": cmd_jp15}


def main(argv: Sequence[str] | None = None) -> None:
    """Standalone entry point; ``run.py`` dispatches the same three commands."""
    from q3vl.whatb.jetlut.run import main as run_main

    sys.argv = [sys.argv[0]] + list(argv) if argv is not None else sys.argv
    run_main()


if __name__ == "__main__":
    main()
